"""Underdog Bark 3.0 어댑터 (parakeet-tdt-0.6b-v3 파인튜닝, 영어 전용).

HF `ConwayResearch/Underdog-Bark-3.0` 의 `model.bark`(163MB, 자체 ANS 압축 포맷)를
공식 `bark_loader.py` 로 FP32 풀어 transformers `ParakeetForTDT` 에 올린다(PyTorch).
transformers>=5.19 + constriction 이 필요해 전용 venv(.venv-bark) 에서 실행한다.
공식 설정대로 앞에 0.25s 무음을 붙인다. 장문은 통짜 시도 후 OOM 이면 _CHUNK_SECONDS 창으로
나눠 이어 붙인다.
"""

import importlib.util

import numpy as np

from .base import SttEngine

_REPO = "ConwayResearch/Underdog-Bark-3.0"
_CHUNK_SECONDS = 120.0


class BarkEngine(SttEngine):
    name = "bark"
    model_id = "Underdog-Bark-3.0"
    #: True 면 GPU 에서도 처음부터 _CHUNK_SECONDS 창으로 돈다(8GB VRAM 초과 방지)
    force_chunk = False

    def load(self) -> None:
        from huggingface_hub import hf_hub_download

        loader_py = hf_hub_download(_REPO, "bark_loader.py")
        spec = importlib.util.spec_from_file_location("bark_loader", loader_py)
        self._loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._loader)
        dev = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self._model = self._loader.load_model(hf_hub_download(_REPO, "model.bark"), device=dev)
        self._chunked = False

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        # CPU 는 통짜 어텐션(수 GB 할당)이 할당 실패/크래시하므로 처음부터 청크로 돈다.
        if self.device.startswith("cuda") and not self.force_chunk:
            try:
                return self._loader.transcribe(self._model, audio, sr)
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
            if len(chunk):
                parts.append(self._loader.transcribe(self._model, chunk, sr))
        return " ".join(p for p in parts if p).strip()
