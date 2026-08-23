"""SenseVoice 어댑터 (FunASR, Paraformer 계열).

베이스 프로젝트(D:\\pjt\\FunASR\\examples\\live_caption)가 쓰는 엔진.
장형 오디오는 fsmn-vad 로 세그먼트하고, 세그먼트 텍스트를 이어 붙인다.
"""

import numpy as np

from .base import SttEngine


class SenseVoiceEngine(SttEngine):
    name = "SenseVoice"
    model_id = "iic/SenseVoiceSmall"

    def load(self) -> None:
        from funasr import AutoModel

        self._model = AutoModel(
            model=self.model_id,
            trust_remote_code=False,
            disable_update=True,
            device=self.device,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
        )

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        res = self._model.generate(
            input=audio,
            cache={},
            language=self.language,
            use_itn=True,
            disable_pbar=True,
        )
        if not res:
            return ""
        # res: 세그먼트별 dict 목록. 각 세그먼트의 text 를 이어 붙인다.
        parts = []
        for seg in res:
            t = rich_transcription_postprocess(seg.get("text", ""))
            if t:
                parts.append(t)
        return " ".join(parts).strip()
