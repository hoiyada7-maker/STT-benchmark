"""Voxtral Mini 4B Realtime streaming adapter (HF transformers VoxtralRealtimeForConditionalGeneration).

True streaming: model.generate() runs in a worker thread and pulls mel-feature chunks from a queue-backed
generator (the official streaming API); feed() pushes one 80 ms step and blocks until the model asks for the
next chunk, i.e. until the token for that step has been generated.  Tokens never get revised -> (text, text).

settings: "<delay_ms>" or "<delay_ms>-4bit"   delay multiple of 80 (80..1200, 2400); e.g. "480", "480-4bit".
  cpu: fp32 by default (CPU bf16 is unusably slow here: prefill >4 min; env VOXTRAL_CPU_DTYPE=bfloat16 to override); cuda: bf16, or 4-bit bitsandbytes with "-4bit".
"""
import os
import queue
import threading

import numpy as np

from .base import StreamEngine

STEP_MS = 80  # 1 text token = 8 mel frames x 10 ms hop = 1280 samples


class _Streamer:
    """Minimal HF streamer: collects generated token ids (first put() is the prompt)."""

    def __init__(self):
        self.ids, self._first = [], True

    def put(self, value):
        if self._first:
            self._first = False
            return
        self.ids.extend(value.reshape(-1).tolist())

    def end(self):
        pass


class Engine(StreamEngine):
    def __init__(self, device, setting):
        delay, _, flag = str(setting).partition("-")
        self.delay_ms = int(delay)
        self.q4 = flag == "4bit"
        self.device = "cuda" if str(device).startswith("cuda") else "cpu"
        self.name = "voxtral-realtime"
        self.model_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"
        self.setting = f"지연 {self.delay_ms}ms" + (" 4bit" if self.q4 else "")
        self.chunk_ms = STEP_MS
        self.thread = None

    def load(self):
        import torch
        from transformers import VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeProcessor

        self.torch = torch
        self.proc = VoxtralRealtimeProcessor.from_pretrained(self.model_id)
        # delay lives in the mistral_common audio config (tekken.json: transcription_delay_ms)
        self.proc.mistral_common_audio_config.transcription_delay_ms = float(self.delay_ms)
        kw = {}
        if self.device == "cuda" and self.q4:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
            kw["device_map"] = {"": 0}
        elif self.device == "cuda":
            kw.update(dtype=torch.bfloat16, device_map={"": 0})
        else:
            kw["dtype"] = getattr(torch, os.environ.get("VOXTRAL_CPU_DTYPE", "float32"))
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(self.model_id, **kw).eval()
        p = self.proc
        self.first_n = p.num_samples_first_audio_chunk
        self.win_n = p.num_samples_per_audio_chunk
        self.step_n = p.audio_length_per_tok * p.feature_extractor.hop_length
        self.half_win = p.feature_extractor.win_length // 2
        self.tail_pad = p.num_right_pad_tokens * p.raw_audio_length_per_tok
        self.reset()

    def _stop(self):
        if self.thread is not None:
            self.q.put(None)
            self.thread.join(timeout=60)
        self.thread = None

    def reset(self):
        self._stop()
        self.buf = np.zeros(0, np.float32)   # audio from absolute sample index self.base on
        self.base = 0
        self.next_start = None               # absolute start of next streaming window (None = first chunk pending)
        self.q = queue.Queue()
        self.ready = threading.Event()
        self.err = None
        self.streamer = _Streamer()

    def _feats(self, audio, first):
        inp = self.proc(audio, is_streaming=True, is_first_audio_chunk=first, return_tensors="pt")
        return inp.to(self.model.device, dtype=self.model.dtype)

    def _run(self, first_inputs):
        def gen():
            yield first_inputs.input_features
            while True:
                self.ready.set()              # previous step done; model wants the next chunk
                f = self.q.get()
                if f is None:
                    return
                yield f
        try:
            with self.torch.inference_mode():
                self.model.generate(input_ids=first_inputs.input_ids, input_features=gen(),
                                    num_delay_tokens=first_inputs.num_delay_tokens, streamer=self.streamer)
        except Exception as e:  # surfaced in feed()
            self.err = e
        finally:
            self.ready.set()

    def _wait(self):
        self.ready.wait()
        if self.err:
            raise self.err

    def _pump(self):
        """Push every window that is fully available in the buffer."""
        if self.next_start is None:
            if self.base + len(self.buf) < self.first_n:
                return
            inp = self._feats(self.buf[:self.first_n], True)
            self.next_start = self.proc.num_mel_frames_first_audio_chunk * self.proc.feature_extractor.hop_length \
                - self.half_win
            self.thread = threading.Thread(target=self._run, args=(inp,), daemon=True)
            self.thread.start()
            self._wait()
        while self.next_start + self.win_n <= self.base + len(self.buf):
            a = self.next_start - self.base
            f = self._feats(self.buf[a:a + self.win_n], False).input_features
            self.ready.clear()
            self.q.put(f)
            self._wait()
            self.next_start += self.step_n
        if self.next_start is not None and self.next_start - self.base > 0:   # drop consumed audio
            d = self.next_start - self.base
            self.buf, self.base = self.buf[d:], self.base + d

    def _text(self):
        return self.proc.tokenizer.decode(self.streamer.ids, skip_special_tokens=True).strip()

    def feed(self, chunk):
        self.buf = np.concatenate([self.buf, np.asarray(chunk, np.float32)])
        self._pump()
        t = self._text()
        return t, t

    def finish(self):
        self.buf = np.concatenate([self.buf, np.zeros(self.tail_pad, np.float32)])
        self._pump()
        self._stop()
        return self._text()
