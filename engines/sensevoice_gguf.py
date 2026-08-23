"""SenseVoice-GGUF 어댑터 (llama.cpp / FunASR llama.cpp 런타임, CPU).

SenseVoice-Small 을 GGUF 로 양자화한 모델을 FunASR 의 llama.cpp 런타임
(단일 C++ 바이너리 `llama-funasr-sensevoice`) 로 실행한다. Python ML 스택이
아니라 사전 빌드된 네이티브 바이너리를 subprocess 로 호출한다.

- CPU 전용: `device` 인자는 받아들이지만 CPU 백엔드로 고정한다.
- 장문 오디오는 바이너리에 내장된 FSMN-VAD 로 세그먼트한다.
- 16kHz mono float32 numpy 배열을 입력으로 받아 텍스트를 반환한다 (base.py 규약).
"""

import os
import subprocess
import tempfile

import numpy as np

from .base import SttEngine

#: 프로젝트 .temp 아래 Tier3 검증 폴더의 기본 자산 위치
_DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".temp", "tier3_sensevoice_gguf",
)


class SenseVoiceGGUFEngine(SttEngine):
    name = "SenseVoice-GGUF"
    model_id = "FunAudioLLM/SenseVoiceSmall-GGUF (Q8_0)"

    def __init__(self, language="en", device="cpu", *, root=None):
        super().__init__(language=language, device="cpu")  # CPU 런타임 고정
        root = root or _DEFAULT_ROOT
        self._binary = os.path.join(root, "runtime", "llama-funasr-sensevoice.exe")
        self._model = os.path.join(root, "models", "sensevoice-small-q8.gguf")
        self._vad = os.path.join(root, "models", "fsmn-vad.gguf")

    def load(self) -> None:
        """바이너리·모델 경로가 존재하는지 확인한다 (서브프로세스 방식이라 추가 로드 없음)."""
        for label, path in (
            ("런타임 바이너리", self._binary),
            ("GGUF 모델", self._model),
            ("FSMN-VAD 모델", self._vad),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"[{self.name}] {label} 없음: {path}\n"
                    f"  .temp/tier3_sensevoice_gguf 아래에 런타임/모델을 확보하세요."
                )

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """16k mono float32 오디오를 임시 wav 로 저장 후 CLI 로 전사한다."""
        import soundfile as sf

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        fd, tmp_wav = tempfile.mkstemp(
            prefix="gguf_asr_", suffix=".wav", dir=_DEFAULT_ROOT,
        )
        os.close(fd)
        try:
            sf.write(tmp_wav, audio, sr, subtype="PCM_16")
            cmd = [
                self._binary,
                "-m", self._model,
                "--vad", self._vad,
                "-a", tmp_wav,
            ]
            proc = subprocess.run(
                cmd, capture_output=True, timeout=600,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"[{self.name}] CLI 실패 (exit {proc.returncode}):\n"
                    f"  stderr: {proc.stderr.decode('utf-8', 'replace')[-500:]}"
                )
            text = proc.stdout.decode("utf-8", "replace").strip()
            return text
        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
