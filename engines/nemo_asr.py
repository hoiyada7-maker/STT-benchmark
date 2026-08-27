"""NVIDIA NeMo ASR 어댑터 (Parakeet-Unified / Nemotron-Speech / Nemotron-3.5).

세 모델 모두 `nemo.collections.asr` 의 `ASRModel.from_pretrained` 로 GPU 로드하고
`transcribe()` 로 추론한다. 다만 모델마다 호출 규약이 달라 variant 로 분기한다.

- parakeet-unified (영어)   : use_lhotse=True 로 호출해야 함(안 그러면 cfg.validation_ds
                              None 로 dataloader 셋업이 깨짐). 파일 경로 입력.
- nemotron-speech (영어)     : 기본 경로 입력으로 그대로 동작.
- nemotron-3.5 (다국어)      : prompt-conditioned. prompt=로케일("en-US"/"ko-KR") 로
                              언어를 지정하고, 출력에 섞여 나오는 <en-US>·<ko-KR>
                              언어 태그를 제거한다. numpy 배열 입력.

Windows 에서 dataloader 임시파일 잠금(WinError 32)을 피하려 num_workers=0 고정.
스트리밍/RNNT 계열이라 장문 처리는 안정적이지만, 8GB GPU OOM 대비로 실패 시
_CHUNK_SECONDS 창으로 나눠 이어 붙인다.
"""

import os
import re
import tempfile

import numpy as np

from .base import SttEngine

# variant -> (모델 id, prompt 조건부 여부)
_MODELS = {
    "parakeet-unified": ("nvidia/parakeet-unified-en-0.6b", False),
    "nemotron-speech": ("nvidia/nemotron-speech-streaming-en-0.6b", False),
    "nemotron-3.5": ("nvidia/nemotron-3.5-asr-streaming-0.6b", True),
}
_LOCALE = {"en": "en-US", "ko": "ko-KR"}
_TAG_RE = re.compile(r"<[a-zA-Z]{2}-[a-zA-Z]{2}>")
_CHUNK_SECONDS = 240.0


def _clean(text: str) -> str:
    """모델이 내뱉는 <en-US>·<ko-KR> 언어 태그와 잉여 공백 제거."""
    text = _TAG_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", text).strip()


class NemoAsrEngine(SttEngine):
    name = "nemo-asr"

    def __init__(self, language="en", device="cpu", variant="nemotron-speech"):
        super().__init__(language=language, device=device)
        v = str(variant).lower()
        if v not in _MODELS:
            v = "nemotron-speech"
        self.variant = v
        self._model_name, self._prompted = _MODELS[v]
        self.model_id = self._model_name.split("/")[-1]

    def load(self) -> None:
        import nemo.collections.asr as nemo_asr

        dev = "cuda:0" if str(self.device).startswith("cuda") else "cpu"
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self._model_name)
        self._model = self._model.to(dev)
        # parakeet-unified 는 cfg.validation_ds 가 None 이라 transcribe 의 dataloader
        # 셋업(_setup_transcribe_dataloader)이 깨진다. 최소 validation_ds 를 채워준다.
        from omegaconf import OmegaConf, open_dict

        if self._model.cfg.get("validation_ds") is None:
            with open_dict(self._model.cfg):
                self._model.cfg.validation_ds = OmegaConf.create({
                    "use_start_end_token": False,
                    "sample_rate": self._model.cfg.get("sample_rate", 16000),
                    "batch_size": 1,
                    "shuffle": False,
                })
        self._locale = _LOCALE.get(self.language[:2], "en-US")
        self._chunked = False

    def _kwargs(self) -> dict:
        kw = {"num_workers": 0, "verbose": False}
        if self.variant == "parakeet-unified":
            kw["use_lhotse"] = True
        if self._prompted:
            kw["prompt"] = self._locale
        return kw

    def _one(self, audio: np.ndarray, sr: int) -> str:
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        kw = self._kwargs()
        # prompt-conditioned 모델은 배열 입력이 안정적, 그 외는 임시 wav 경로 입력.
        if self._prompted:
            out = self._model.transcribe([audio], **kw)
        else:
            import soundfile as sf

            fd, path = tempfile.mkstemp(suffix=".wav", dir=os.environ.get("TEMP"))
            os.close(fd)
            try:
                sf.write(path, audio, sr)
                out = self._model.transcribe([path], **kw)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        text = out[0].text if out and hasattr(out[0], "text") else (out[0] if out else "")
        return _clean(text)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        try:
            return self._one(audio, sr)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            import torch

            torch.cuda.empty_cache()

        # OOM 폴백: 고정창으로 나눠 이어 붙인다.
        self._chunked = True
        step = int(_CHUNK_SECONDS * sr)
        parts = []
        for start in range(0, len(audio), step):
            chunk = audio[start : start + step]
            if len(chunk) == 0:
                continue
            t = self._one(chunk, sr)
            if t:
                parts.append(t)
        return " ".join(parts).strip()
