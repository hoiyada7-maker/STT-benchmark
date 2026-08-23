"""MOSS-Transcribe-Diarize 어댑터 (LLM 기반 ASR).

아키텍처: Whisper-Medium 인코더 + Qwen3-0.6B 텍스트 디코더 (transformers, trust_remote_code).
출력 형식 `[start_time][Sxx]text[end_time]` 에서 화자/타임스탬프 태그를 제거해
순수 텍스트만 반환한다.
"""

import os
import re
import tempfile

import numpy as np
import soundfile as sf

from .base import SttEngine

# `[0.48]`(시작/종료 타임스탬프) 와 `[S01]`(화자) 태그. 숫자·화자번호 모두 제거.
_TAG_RE = re.compile(r"\[\s*\d+(?:\.\d+)?\s*\]|\[\s*S\d+\s*\]", re.IGNORECASE)


class MossTranscribeDiarizeEngine(SttEngine):
    name = "MOSS-Transcribe-Diarize"
    model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

    def __init__(self, language="en", device="cuda:0", max_new_tokens=2048):
        super().__init__(language=language, device=device)
        self.max_new_tokens = max_new_tokens

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        from moss_transcribe_diarize.inference_utils import resolve_device

        device = resolve_device(self.device)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        self._device = device
        self._dtype = dtype
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True, dtype="auto"
            )
            .to(dtype=dtype)
            .to(device)
            .eval()
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        from moss_transcribe_diarize.inference_utils import (
            build_transcription_messages,
            generate_transcription,
        )

        # MOSS 경로 기반 API: 16k float32 배열을 임시 wav 로 저장해 넘긴다.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            sf.write(tmp_path, audio, samplerate=sr, subtype="FLOAT")
            messages = build_transcription_messages(tmp_path)
            result = generate_transcription(
                self._model,
                self._processor,
                messages,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                device=self._device,
                dtype=self._dtype,
            )
            return self._clean(result["text"])
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _clean(text: str) -> str:
        text = _TAG_RE.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()
