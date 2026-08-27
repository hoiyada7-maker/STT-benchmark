"""Qwen3-ASR 어댑터 (Qwen3-ASR-0.6B / 1.7B, transformers 백엔드).

`qwen-asr` 패키지의 Qwen3ASRModel 로 GPU(bf16) 추론한다. transcribe() 는
긴 오디오를 내부적으로 처리하지만, OOM/실패 시 ~120초 창으로 나눠 이어 붙인다.
언어는 full-name("English"/"Korean") 으로 넘겨야 한다.

같은 저장소의 Qwen3-ForcedAligner-0.6B 는 오디오→텍스트 전사를 하지 않고
(정답 텍스트↔오디오 타임스탬프 정렬 전용) WER/CER 대상이 아니라 제외한다.
"""

import numpy as np

from .base import SttEngine

_MODELS = {
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
    # 한국어 특화 파인튜닝(같은 qwen-asr API, base 는 Qwen3-ASR-1.7B)
    "1.7b-ko": "TeamUNIVA/qwen3_asr_1.7b_ko_beta",
}
_LANG = {"en": "English", "ko": "Korean"}
_CHUNK_SECONDS = 120.0


class Qwen3AsrEngine(SttEngine):
    name = "qwen3-asr"

    def __init__(self, language="en", device="cpu", variant="0.6b"):
        super().__init__(language=language, device=device)
        v = str(variant).lower().replace("qwen/", "").replace("qwen3-asr-", "")
        if v not in _MODELS:
            v = "0.6b"
        self.variant = v
        self.model_id = _MODELS[v].split("/")[-1]

    def load(self) -> None:
        import torch
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
        self._chunked = False

    def _one(self, audio: np.ndarray, sr: int) -> str:
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
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
