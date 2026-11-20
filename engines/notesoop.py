"""Notesoop-STT-Ko-v1 어댑터 (한국어 전용, 비자기회귀 CTC + KenLM 3-gram).

`google/embeddinggemma-2` 오디오 타워(동결) → 경량 Conformer 리더 → 25k 형태소 어휘 코사인 매칭 →
KenLM 3-gram 빔서치(beam 8, alpha 0.12, beta 0.40, 모델 카드 기본값) 로 구성된다.
모델 저장소가 자체 추론 코드(run_inference.py, src/*.py)를 함께 배포하므로 스냅샷을 받아
sys.path 에 올려 `EmbeddingGemmaSTT` 를 그대로 쓴다(코드 복제 없음).

전용 venv(.venv-notesoop, Python 3.12) 필요: embedding_gemma2 아키텍처는 최신 transformers(5.x)와
torchvision/Pillow(이미지 프로세서 로딩용), 그리고 kenlm(Windows 는 pip 소스 빌드, MinGW g++ 로 성공) 이 필요하다.
KenLM 이 없으면 조용히 Greedy CTC 로 떨어지므로 load() 에서 명시적으로 막는다.

한국어 전용이라 language 힌트는 쓰지 않는다. 모델은 파일 경로를 입력받으므로 배열을 임시 wav 로 쓴다.
리더 체크포인트가 청크(16초, 겹침 1초)·배치(4) 단위로 처리하는 것이 모델 고유 동작이다.
"""

import os
import sys
import tempfile

import numpy as np

from .base import SttEngine

_REPO = "Phsoft-ai/Notesoop-STT-Ko-v1"


class NotesoopEngine(SttEngine):
    name = "notesoop"
    model_id = "Notesoop-STT-Ko-v1"

    def __init__(self, language="ko", device="cuda:0"):
        super().__init__(language=language, device=device)
        self._chunked = True  # 모델 자체가 16초 창으로 나눠 이어 붙임

    def load(self) -> None:
        import kenlm  # noqa: F401  (없으면 Greedy 로 조용히 폴백하므로 먼저 실패시킨다)
        from huggingface_hub import snapshot_download

        root = snapshot_download(_REPO)
        if root not in sys.path:
            sys.path.insert(0, root)
        from run_inference import EmbeddingGemmaSTT

        dev = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self._stt = EmbeddingGemmaSTT(device=dev, use_kenlm=True, lm_alpha=0.12, lm_beta=0.40)
        if self._stt.decoder_kenlm is None:
            raise RuntimeError("KenLM 디코더 로드 실패")

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        import soundfile as sf

        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, audio, sr)
            text, _rtf = self._stt.transcribe(path, chunk_sec=16.0, overlap_sec=1.0, batch_size=4)
        finally:
            os.unlink(path)
        return text.strip()
