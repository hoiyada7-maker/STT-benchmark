"""Parakeet 어댑터 (NVIDIA NeMo Parakeet-TDT, onnx-asr 런타임).

onnxruntime(CPU) 로 추론한다. 긴 오디오는 우선 통짜로 시도하고,
실패/과도한 지연 시 ~120초 창으로 나눠 이어 붙인다.
"""

import numpy as np

from .base import SttEngine

_CHUNK_SECONDS = 120.0


class ParakeetEngine(SttEngine):
    name = "parakeet"
    model_id = "nemo-parakeet-tdt-0.6b-v3"

    def load(self) -> None:
        import onnx_asr

        self._model = onnx_asr.load_model(self.model_id)
        self._chunked = False

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        try:
            return self._model.recognize(audio, sample_rate=sr)
        except Exception:
            # 전체 통짜 인식 실패 시 청크로 폴백.
            self._chunked = True
            step = int(_CHUNK_SECONDS * sr)
            parts = []
            for start in range(0, len(audio), step):
                chunk = audio[start : start + step]
                if len(chunk) == 0:
                    continue
                text = self._model.recognize(chunk, sample_rate=sr)
                if text:
                    parts.append(text.strip())
            return " ".join(parts).strip()
