"""Kyutai STT (PyTorch `moshi` package): Mimi (24 kHz, 1920-sample = 80 ms frames) + causal LM.

Follows delayed-streams-modeling/scripts/stt_from_file_pytorch.py: per-frame mimi.encode ->
lm_gen.step -> text token (pad ids 0 / existing_text_padding_id skipped). Greedy (temp=0).
Tokens never revise -> (text, text). Text lags audio by the model's fixed delay
(stt_config.audio_delay_seconds: 0.5 s for 1b, 2.5 s for 2.6b); finish() feeds delay+1 s silence.
reset() feeds audio_silence_prefix_seconds of silence (as the official script does) before the
stream; that cost is paid in reset(), not feed().
16k->24k: 1280 -> 1920 samples exactly (3/2). Done statefully with a 32-sample past context and
a 32-sample look-ahead (output delayed by 2 ms) so chunk edges see real neighbouring samples.
"""
import numpy as np

try:
    from .base import StreamEngine
except Exception:
    StreamEngine = object

_REPOS = {"1b": "kyutai/stt-1b-en_fr", "2.6b": "kyutai/stt-2.6b-en"}
_LABEL = {"1b": "1b (지연 0.5s)", "2.6b": "2.6b (지연 2.5s)"}
_K = 32  # resampler context / look-ahead in 16 kHz samples
_FRAME = 1920


class Engine(StreamEngine):
    name = "kyutai-stt"
    sample_rate = 16000
    chunk_ms = 80

    def __init__(self, device: str, setting: str):
        self.device, self.key = device, setting
        self.model_id = _REPOS[setting]
        self.setting = _LABEL[setting]
        self._stack = None

    def load(self):
        import os
        if self.device == "cpu":  # torch.compile needs MSVC `cl` (absent here); moshi's own opt-out
            os.environ.setdefault("NO_TORCH_COMPILE", "1")
        import torch
        from moshi.models import LMGen, loaders
        self.torch = torch
        dev = self.device
        info = loaders.CheckpointInfo.from_hf_repo(self.model_id)
        self.mimi = info.get_mimi(device=dev)
        self.tok = info.get_text_tokenizer()
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        lm = info.get_moshi(device=dev, dtype=dtype)
        self.lm_gen = LMGen(lm, temp=0, temp_text=0.0)
        self.prefix_s = info.stt_config.get("audio_silence_prefix_seconds", 1.0)
        self.delay_s = info.stt_config.get("audio_delay_seconds", 5.0)
        self.pad_id = (info.raw_config or {}).get("existing_text_padding_id", 3)
        self.reset()

    def reset(self):
        from contextlib import ExitStack
        if self._stack is not None:
            self._stack.close()
        self._stack = ExitStack()
        self._stack.enter_context(self.mimi.streaming(1))
        self._stack.enter_context(self.lm_gen.streaming(1))
        self._buf = np.zeros(_K, dtype=np.float32)  # 16 kHz history; first _K = past context (silence)
        self._pieces = []
        for _ in range(int(round(self.prefix_s * 12.5))):
            self._frame(np.zeros(_FRAME, dtype=np.float32))

    def _frame(self, a24):
        torch = self.torch
        x = torch.from_numpy(a24).to(self.device)[None, None]
        codes = self.mimi.encode(x)
        out = self.lm_gen.step(codes)
        if out is not None:
            t = int(out[0, 0, 0].cpu())
            if t not in (0, self.pad_id):
                self._pieces.append(self.tok.id_to_piece(t).replace("▁", " "))

    def feed(self, chunk):
        import torchaudio.functional as AF
        torch = self.torch
        chunk = np.asarray(chunk, dtype=np.float32)
        self._buf = np.concatenate([self._buf, chunk])
        # emit 16k samples [start-K+..]: window = buf[:len] ; we emit the segment ending K before the end
        win = self._buf[-(len(chunk) + 2 * _K):] if len(self._buf) >= len(chunk) + 2 * _K else \
            np.concatenate([np.zeros(len(chunk) + 2 * _K - len(self._buf), np.float32), self._buf])
        y = AF.resample(torch.from_numpy(win), 16000, 24000).numpy()
        y = y[int(1.5 * _K):int(1.5 * _K) + _FRAME]
        self._buf = self._buf[-(2 * _K + len(chunk)):]
        self._frame(np.ascontiguousarray(y, dtype=np.float32))
        t = self._text()
        return t, t

    def _text(self):
        return "".join(self._pieces).strip()

    def finish(self):
        z = np.zeros(self.chunk_ms * 16, dtype=np.float32)
        for _ in range(int(np.ceil((self.delay_s + 1.0) * 12.5))):
            self.feed(z)
        return self._text()
