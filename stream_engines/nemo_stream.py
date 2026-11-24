"""NeMo cache-aware streaming adapter (Nemotron-Speech / Nemotron-3.5 / Parakeet-Unified).

True streaming: audio is fed chunk by chunk; mel features are computed incrementally by NeMo's own
BatchedCacheFeatureBufferer (rolling feature buffer incl. pre-encode cache), the encoder carries its
attention / conv caches (`conformer_stream_step`), and the RNNT decoder carries partial hypotheses.
Mirrors nemo.collections.asr.inference CacheAwareRNNTPipeline (use_cache=True).

settings:
  nemotron-speech:70,R      R in {1,6,13}   chunk (R+1)*80 ms
  nemotron-3.5:56,R[@L]     R in {3,6,13}   chunk (R+1)*80 ms   (prompt L in {en-US,ko-KR,auto}, 기본 en-US)
  parakeet-unified:70,C,R   C in {2,7,13}, R in {0,1}  chunk C*80 ms, lookahead R*80 ms
"""
import copy
import os
import re

import numpy as np

from .base import StreamEngine

_TAG = re.compile(r"<[a-zA-Z]{2}-[a-zA-Z]{2}>")
_MODELS = {
    "nemotron-speech": "nvidia/nemotron-speech-streaming-en-0.6b",
    "nemotron-3.5": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "parakeet-unified": "nvidia/parakeet-unified-en-0.6b",
}
STEP_MS = 80  # encoder frame = 8 x 10 ms


class NemoStream(StreamEngine):
    def __init__(self, device, setting):
        setting, _, lang = setting.partition("@")   # 선택적 언어 접미사 (nemotron-3.5 전용)
        self.lang = lang or "en-US"
        fam, _, args = setting.partition(":")
        if fam not in _MODELS:
            raise ValueError(f"unknown nemo setting {setting!r}")
        a = [int(x) for x in args.split(",")]
        self.family, self.device = fam, ("cuda:0" if str(device).startswith("cuda") else "cpu")
        self.name = f"nemo-{fam}"
        self.model_id = _MODELS[fam]
        if fam == "parakeet-unified":
            self.left, self.C, self.R = a
            self.setting = f"[{self.left},{self.C},{self.R}] {self.C * STEP_MS}ms" + (f" + {self.R * STEP_MS}ms" if self.R else "")
        else:
            self.left, self.R = a
            self.C = self.R + 1
            self.setting = f"[{self.left},{self.R}] {self.C * STEP_MS}ms"
            self.R = 0  # lookahead is inside the chunk
        self.chunk_ms = self.C * STEP_MS
        self._sid = 0
        self._active = False

    # ------------------------------------------------------------------
    def load(self):
        import torch
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
        from nemo.collections.asr.inference.utils.pipeline_utils import make_preprocessor_deterministic

        self.torch = torch
        m = nemo_asr.models.ASRModel.from_pretrained(self.model_id, map_location="cpu")
        m = m.to(self.device)
        m.eval(); m.freeze()
        enc = m.encoder
        enc.set_default_att_context_size([self.left, self.C - 1])
        if hasattr(m, "set_inference_prompt"):
            m.set_inference_prompt(self.lang)
        self.model = m
        sc = enc.streaming_cfg
        pick = lambda v: v[1] if isinstance(v, list) else v
        self.pre = pick(sc.pre_encode_cache_size)
        self.model_chunk = pick(sc.chunk_size)          # mel frames per encoder step window
        self.shift = pick(sc.shift_size)                # mel frames consumed per step
        self.drop_extra = sc.drop_extra_pre_encoded
        assert self.shift * 10 == self.chunk_ms, (self.shift, self.chunk_ms)
        cfg = make_preprocessor_deterministic(copy.deepcopy(m.cfg))
        # +0.5 sample/frame slack so NeMo's int(secs*sr) / int(secs/stride) truncation can't lose one
        sr = 16000
        self.bufferer = BatchedCacheFeatureBufferer(
            num_slots=1, sample_rate=sr,
            buffer_size_in_secs=((self.pre + self.model_chunk) * 160 + 0.5) / sr,
            chunk_size_in_secs=(self.shift * 160 + 0.5) / sr,
            preprocessor_cfg=cfg.preprocessor, device=torch.device(self.device))
        assert self.bufferer.feature_buffer_len == self.pre + self.model_chunk, self.bufferer.feature_buffer_len
        assert self.bufferer.feature_chunk_len == self.shift
        self.reset()
        z = np.zeros(self.chunk_ms * 16, dtype=np.float32)   # warm-up (cudnn/kernels)
        for _ in range(3):
            self.feed(z)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        torch = self.torch
        if self._active:
            self.bufferer.free_slots([0])
        self._sid += 1
        self._first = True
        self._active = True
        self.cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        self.prev_hyp = None
        self.text = ""

    def _step(self, chunk):
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        torch = self.torch
        fr = Frame(samples=torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)),
                   stream_id=self._sid, is_first=self._first, is_last=False)
        self._first = False
        fbufs, _ = self.bufferer.update([fr])
        fb = fbufs[0].unsqueeze(0).to(self.device)
        ln = torch.tensor([fb.shape[2]], device=self.device)
        with torch.inference_mode():
            out = self.model.conformer_stream_step(
                processed_signal=fb, processed_signal_length=ln,
                cache_last_channel=self.cache[0], cache_last_time=self.cache[1], cache_last_channel_len=self.cache[2],
                keep_all_outputs=False, previous_hypotheses=self.prev_hyp,
                drop_extra_pre_encoded=self.drop_extra, return_transcription=True)
        self.cache = out[2:5]
        self.prev_hyp = out[5]
        ids = self.prev_hyp[0].y_sequence
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        self.text = re.sub(r"\s+", " ", _TAG.sub(" ", self.model.tokenizer.ids_to_text(ids))).strip()

    def feed(self, chunk):
        self._step(chunk)
        return self.text, self.text   # RNNT greedy never revises emitted tokens

    def finish(self):
        # flush lookahead / last frames with silence (>= 640 ms)
        n = max(1, -(-640 // self.chunk_ms))
        z = np.zeros(self.chunk_ms * 16, dtype=np.float32)
        for _ in range(n):
            self._step(z)
        return self.text


class NemoUnifiedStream(NemoStream):
    """Parakeet-Unified: NVIDIA supports only *buffered* streaming for this model (HF model card, NeMo
    examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py): every step re-encodes
    [left ctx | chunk | right ctx] raw audio with the chunked_limited_with_rc mask [left,C,R] (the model's
    'dcc' chunked convolution is not implemented in NeMo, and cache-aware streaming is not supported)
    and decodes only the chunk frames with a stateful RNNT label-looping decoder.
    R>0 is realised by prepending R*80 ms of silence, so output lags audio by exactly R steps."""

    def load(self):
        import torch
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.parts.utils.streaming_utils import ContextSize

        self.torch = torch
        m = nemo_asr.models.ASRModel.from_pretrained(self.model_id, map_location="cpu").to(self.device)
        m.eval(); m.freeze()
        if os.environ.get("NEMO_STREAM_BF16") and self.device != "cpu":
            m.to(torch.bfloat16)
        m.preprocessor.featurizer.dither = 0.0
        m.preprocessor.featurizer.pad_to = 0
        m.encoder.set_default_att_context_size([self.left, self.C, self.R])
        self.model = m
        step = 1280  # samples per encoder step (8 x 160)
        self.ctx = ContextSize(left=self.left * step, chunk=self.C * step, right=self.R * step)
        self.computer = m.decoding.decoding.decoding_computer
        self.reset()
        z = np.zeros(self.chunk_ms * 16, dtype=np.float32)   # warm-up
        for _ in range(3):
            self.feed(z)
        self.reset()

    def reset(self):
        from nemo.collections.asr.parts.utils.streaming_utils import StreamingBatchedAudioBuffer
        self.buf = StreamingBatchedAudioBuffer(batch_size=1, context_samples=self.ctx,
                                               dtype=self.torch.float32, device=self.device)
        self.state, self.hyps = None, None
        self.pending = np.zeros(self.R * 1280, dtype=np.float32)   # lead-in silence = lookahead
        self.text = ""

    def _step(self, chunk):
        torch = self.torch
        a = np.concatenate([self.pending, np.asarray(chunk, dtype=np.float32)])
        self.pending = a[:0]
        x = torch.from_numpy(a)[None].to(self.device)
        n = torch.tensor([x.shape[1]], device=self.device)
        self.buf.add_audio_batch_(x, audio_lengths=n, is_last_chunk=False, is_last_chunk_batch=torch.tensor([False], device=self.device))
        with torch.inference_mode():
            enc, enc_len = self.model(input_signal=self.buf.samples, input_signal_length=self.buf.context_size_batch.total())
            enc = enc.transpose(1, 2)
            ec = self.buf.context_size.subsample(factor=1280)
            ecb = self.buf.context_size_batch.subsample(factor=1280)
            enc = enc[:, ec.left:]
            hyp, self.state = self.computer(x=enc, out_len=ecb.chunk, prev_batched_state=self.state)
            if self.hyps is None:
                self.hyps = hyp
            else:
                self.hyps.merge_(hyp)
        from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
        ids = batched_hyps_to_hypotheses(self.hyps, batch_size=1)[0].y_sequence.tolist()
        self.text = re.sub(r"\s+", " ", self.model.tokenizer.ids_to_text(ids)).strip()


def create(device, setting):
    return (NemoUnifiedStream if setting.startswith("parakeet-unified") else NemoStream)(device, setting)
