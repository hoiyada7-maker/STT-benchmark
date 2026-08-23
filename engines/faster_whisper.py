"""faster-whisper 어댑터 (OpenAI Whisper, CTranslate2 런타임).

로컬 GPU/CPU 추론. 내부 VAD 로 장형 오디오를 자동 세그먼트한다.
"""

import numpy as np

from .base import SttEngine


class FasterWhisperEngine(SttEngine):
    name = "faster-whisper"

    def __init__(self, language="en", device="cpu", model_size="small"):
        super().__init__(language=language, device=device)
        self.model_size = model_size
        self.model_id = f"whisper-{model_size}"

    def load(self) -> None:
        from faster_whisper import WhisperModel

        # GPU 면 float16, CPU 면 int8(속도) 로 기본 설정.
        # CTranslate2 는 "cuda:0" 대신 "cuda" 만 허용한다.
        if self.device.startswith("cuda"):
            self._model = WhisperModel(
                self.model_size, device="cuda", compute_type="float16"
            )
        else:
            self._model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8"
            )

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        # faster-whisper 는 자동 감지에 None 을 쓴다 ("auto" 는 유효 코드가 아님).
        lang = None if self.language in (None, "auto") else self.language
        segments, _info = self._model.transcribe(
            audio,
            language=lang,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip())

    def transcribe_words(self, audio: np.ndarray, sr: int = 16000):
        """단어별 (텍스트, 시작초, 끝초) 리스트. 미리보기 트리밍용(word_timestamps).

        시작/끝 초는 넘긴 audio 의 시작 기준이다.
        """
        lang = None if self.language in (None, "auto") else self.language
        segments, _info = self._model.transcribe(
            audio,
            language=lang,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
        )
        out = []
        for seg in segments:
            for w in (seg.words or []):
                t = w.word.strip()
                if t:
                    out.append((t, float(w.start), float(w.end)))
        return out
