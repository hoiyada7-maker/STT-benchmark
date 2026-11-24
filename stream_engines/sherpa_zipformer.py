"""sherpa-onnx streaming Zipformer (en-2023-06-26, chunk-16-left-128, fp32 ONNX, CPU).

chunk_ms=320: encoder metadata decode_chunk_len=32 input fbank frames (10 ms shift) = 320 ms
(chunk-size 16 post-embed frames @50 Hz; T=45 = 32 + 13 look-ahead frames, handled internally).
Endpointing disabled -> one continuous stream. Greedy transducer never revises -> (text, text).
num_threads=8.
"""
from pathlib import Path

import numpy as np

try:
    from .base import StreamEngine
except Exception:  # base.py not present / standalone test
    StreamEngine = object

_REPO = Path(__file__).resolve().parent.parent
_NAME = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
_CANDS = [_REPO / "models" / _NAME, _REPO / ".temp" / "2026-11-23-stream-bench" / "models" / _NAME]
_SUF = "epoch-99-avg-1-chunk-16-left-128"


class Engine(StreamEngine):
    name = "sherpa-zipformer"
    model_id = "csukuangfj/" + _NAME
    sample_rate = 16000
    chunk_ms = 320
    NUM_THREADS = 8

    def __init__(self, device: str = "cpu", setting: str = "chunk-16"):
        self.device = "cpu"  # ponytail: CPU only (pip sherpa-onnx wheel has no CUDA)
        self.setting = "chunk-16 (≈320ms)"
        self._rec = self._s = None

    def load(self):
        import sherpa_onnx
        d = next(p for p in _CANDS if p.exists())
        self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=str(d / f"encoder-{_SUF}.onnx"),
            decoder=str(d / f"decoder-{_SUF}.onnx"),
            joiner=str(d / f"joiner-{_SUF}.onnx"),
            num_threads=self.NUM_THREADS, sample_rate=16000, feature_dim=80,
            enable_endpoint_detection=False, decoding_method="greedy_search",
        )
        self.reset()

    def reset(self):
        self._s = self._rec.create_stream()

    def _text(self):
        while self._rec.is_ready(self._s):
            self._rec.decode_stream(self._s)
        return self._rec.get_result(self._s).strip()

    def feed(self, chunk):
        self._s.accept_waveform(16000, np.asarray(chunk, dtype=np.float32))
        t = self._text()
        return t, t

    def finish(self):
        self._s.accept_waveform(16000, np.zeros(int(0.66 * 16000), dtype=np.float32))  # tail padding
        self._s.input_finished()
        return self._text()
