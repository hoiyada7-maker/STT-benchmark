"""Qwen3-ASR 어댑터 (Qwen3-ASR-0.6B / 1.7B / 1.7b-ko + Oruk Fourier 2B).

기본 변형(`0.6b` / `1.7b` / `1.7b-ko`)은 `qwen-asr` 패키지의 `Qwen3ASRModel` 로
GPU(bf16) 추론한다. 네이티브 HF `transformers >= 5.21` 포맷 체크포인트인
`oruk-fourier-2b` (`oruk/oruk-fourier-2b`, 영어 전용, CC BY-NC 4.0 비상업)는
`AutoProcessor` + `AutoModelForMultimodalLM` 으로 로드한다 (전용 transformers>=5.21 venv 권장).
긴 오디오는 120초 고정창으로 나눠 이어 붙인다.
언어는 full-name("English"/"Korean") 으로 넘겨야 한다.

같은 저장소의 Qwen3-ForcedAligner-0.6B 는 오디오→텍스트 전사를 하지 않고
(정답 텍스트↔오디오 타임스탬프 정렬 전용) WER/CER 대상이 아니라 제외한다.
"""

import importlib.metadata
import re

import numpy as np

from .base import SttEngine

_FOURIER_VARIANT = "oruk-fourier-2b"
_FOURIER_REVISION = "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
_NATIVE_MIN_TRANSFORMERS = (5, 21)

_MODELS = {
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
    # 한국어 특화 파인튜닝(같은 qwen-asr API, base 는 Qwen3-ASR-1.7B)
    "1.7b-ko": "TeamUNIVA/qwen3_asr_1.7b_ko_beta",
    # 영어 특화 파인튜닝(네이티브 HF transformers >= 5.21 포맷, CC BY-NC 4.0 비상업)
    _FOURIER_VARIANT: "oruk/oruk-fourier-2b",
}
_NATIVE_HF_REVISIONS = {
    _FOURIER_VARIANT: _FOURIER_REVISION,
}
_ENGLISH_ONLY_VARIANTS = {_FOURIER_VARIANT}
_LANG = {"en": "English", "ko": "Korean"}
_CHUNK_SECONDS = 120.0


def _transformers_version():
    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_at_least(version_text, minimum):
    if not version_text:
        return False
    parts = [int(p) for p in re.findall(r"\d+", str(version_text))[: len(minimum)]]
    while len(parts) < len(minimum):
        parts.append(0)
    return tuple(parts) >= minimum


def _normalize_variant(variant) -> str:
    v = str(variant).strip().lower()
    if v in ("oruk/oruk-fourier-2b", "oruk-fourier-2b", "fourier-2b"):
        return _FOURIER_VARIANT
    v = v.replace("qwen/", "").replace("qwen3-asr-", "")
    return v if v in _MODELS else "0.6b"


class Qwen3AsrEngine(SttEngine):
    name = "qwen3-asr"

    def __init__(self, language="en", device="cpu", variant="0.6b"):
        super().__init__(language=language, device=device)
        v = _normalize_variant(variant)
        self.variant = v
        self.model_id = _MODELS[v].split("/")[-1]
        self._native_hf = v in _NATIVE_HF_REVISIONS

    def load(self) -> None:
        import torch

        self._torch = torch
        self._chunked = False
        if self._native_hf:
            if self.variant in _ENGLISH_ONLY_VARIANTS and not str(self.language).lower().startswith("en"):
                raise ValueError(
                    f"{_MODELS[self.variant]} is English-only (CC BY-NC 4.0) "
                    f"and does not support language={self.language!r}."
                )
            installed = _transformers_version()
            if not _version_at_least(installed, _NATIVE_MIN_TRANSFORMERS):
                min_ver = ".".join(str(p) for p in _NATIVE_MIN_TRANSFORMERS)
                raise ImportError(
                    f"{_MODELS[self.variant]} requires transformers >= {min_ver} "
                    f"(found {installed or 'not installed'}). Run in a dedicated transformers>={min_ver} venv."
                )

            from transformers import AutoModelForMultimodalLM, AutoProcessor

            raw_dev = str(self.device).lower()
            if raw_dev.startswith("cuda"):
                dev = "cuda:0" if raw_dev == "cuda" else raw_dev
                dtype = torch.bfloat16
            elif raw_dev == "mps":
                dev = "mps"
                dtype = torch.bfloat16
            else:
                dev = "cpu"
                dtype = torch.float32
            self._dev = dev
            self._dtype = dtype
            repo_id = _MODELS[self.variant]
            revision = _NATIVE_HF_REVISIONS[self.variant]
            self._processor = AutoProcessor.from_pretrained(repo_id, revision=revision)
            self._model = (
                AutoModelForMultimodalLM.from_pretrained(repo_id, dtype=dtype, revision=revision)
                .to(dev)
                .eval()
            )
            self._lang = "en"
            return

        from qwen_asr import Qwen3ASRModel

        dev = "cuda:0" if str(self.device).startswith("cuda") else "cpu"
        self._model = Qwen3ASRModel.from_pretrained(
            _MODELS[self.variant],
            dtype=torch.bfloat16,
            device_map=dev,
            max_new_tokens=4096,
        )
        # 일부 파인튜닝 저장소(예: 1.7b-ko)는 chat_template.json 을 누락한다.
        # base Qwen3-ASR-1.7B 의 템플릿(동일 아키텍처)을 복원해 채운다.
        if getattr(self._model.processor, "chat_template", None) is None:
            import json

            from huggingface_hub import hf_hub_download

            p = hf_hub_download("Qwen/Qwen3-ASR-1.7B", "chat_template.json")
            with open(p, encoding="utf-8") as f:
                self._model.processor.chat_template = json.load(f)["chat_template"]
        self._lang = _LANG.get(self.language[:2], "English")

    def _one(self, audio: np.ndarray, sr: int) -> str:
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        if self._native_hf:
            if sr != 16000:
                from math import gcd

                from scipy.signal import resample_poly

                g = gcd(int(sr), 16000)
                audio = np.ascontiguousarray(resample_poly(audio, 16000 // g, int(sr) // g), dtype=np.float32)
            inputs = self._processor.apply_transcription_request(
                audio=audio,
                language=self._lang,
                return_tensors="pt",
            ).to(self._dev, dtype=self._dtype)
            with self._torch.inference_mode():
                output_ids = self._model.generate(**inputs, max_new_tokens=4096)
            generated = output_ids[:, inputs["input_ids"].shape[1] :]
            decoded = self._processor.batch_decode(generated, skip_special_tokens=True)
            text = (decoded[0] if isinstance(decoded, list) and decoded else str(decoded or "")).strip()
            if "<asr_text>" in text:
                text = text.split("<asr_text>", 1)[-1].strip()
            return text
        res = self._model.transcribe(audio=(audio, sr), language=self._lang)
        return res[0].text.strip() if res and res[0].text else ""

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        # Qwen3-ASR 은 발화 단위 모델이라 수분 단위 통짜 입력에서 앞부분을 통째로
        # 누락하는 등 장문 처리가 불안정하다(예외는 안 나므로 조용히 나쁜 결과가 됨).
        # 120초 이하 세그먼트는 시작까지 정확히 잡으므로, 긴 오디오는 항상 120초
        # 고정창으로 나눠 이어 붙인다(경계 ~4개, 페널티 미미).
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        step = int(_CHUNK_SECONDS * sr)
        if len(audio) <= step:
            return self._one(audio, sr)

        self._chunked = True
        parts = []
        for start in range(0, len(audio), step):
            chunk = audio[start : start + step]
            if len(chunk) == 0:
                continue
            text = self._one(chunk, sr)
            if text:
                parts.append(text)
        return " ".join(parts).strip()
