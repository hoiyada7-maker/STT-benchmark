"""Moonshine 어댑터 (Useful Sensors Moonshine, 영어 전용 단문형 모델).

기본은 onnx-runtime 기반 useful-moonshine-onnx 패키지.
단문형 모델이라 한 번 호출에 64초 미만만 허용되므로, 30초 창으로 나눠 이어 붙인다.
onnx 백엔드 로드가 실패하면 transformers ASR 파이프라인(GPU, chunk_length_s=30)으로 폴백한다.
"""

import numpy as np

from .base import SttEngine

_CHUNK_SECONDS = 30.0


class MoonshineEngine(SttEngine):
    name = "moonshine"
    model_id = "moonshine/base"

    def load(self) -> None:
        self.backend = "onnx"
        try:
            import moonshine_onnx  # noqa: F401

            self._moonshine_onnx = moonshine_onnx
        except Exception:
            self.backend = "transformers"
            from transformers import pipeline

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model="UsefulSensors/moonshine-base",
                chunk_length_s=30,
                device=0,
            )

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        audio = np.asarray(audio, dtype=np.float32)

        if self.backend == "transformers":
            result = self._pipe(audio)
            return result["text"].strip()

        # onnx 백엔드: 단문형 모델이므로 30초 창으로 수동 분할.
        step = int(_CHUNK_SECONDS * sr)
        parts = []
        for start in range(0, len(audio), step):
            chunk = audio[start : start + step]
            if len(chunk) / sr < 0.1:
                continue
            texts = self._moonshine_onnx.transcribe(chunk, self.model_id)
            text = texts[0] if texts else ""
            if text:
                parts.append(text.strip())
        return " ".join(parts).strip()
