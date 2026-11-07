"""VibeVoice-ASR-BitNet 어댑터 (microsoft/VibeASR.cpp 런타임, CPU).

VibeVoice-ASR-BitNet(1.5B; VAE=I8_S, LM=I2_S) 을 공식 C++ 런타임의 `asr_infer`
바이너리로 실행한다. Python ML 스택이 아니라 MinGW 로 빌드한 네이티브 바이너리를
subprocess 로 호출한다 (sensevoice_gguf.py 와 같은 방식).

- CPU 전용: `device` 인자는 받아들이지만 CPU 로 고정한다.
- CLI 에 언어 옵션이 없다 (다국어 모델이 오디오에서 언어를 판단). `language` 는 무시된다.
- 출력은 `[start - end] Speaker N: 내용` 줄들 → 발화 내용만 이어 붙여 반환한다.
- CLI 는 호출마다 모델을 로드한다(mmap). stderr 의 자체 타이밍을 `last_timing` 에 보관:
  load_ms(모델 로드), rtf(로드 제외 요청 구간 RTF), request_ms.
- 디코딩은 재현성을 위해 --greedy 고정.
- 장문은 60초(_CHUNK_SECONDS) 고정창으로 잘라 청크마다 CLI 를 호출한다. 600초를 한 번에
  넣으면 VAE 가 ~138 GB 연산 arena 를 요구해 실패한다(CLI 는 exit 0 + 빈 출력). 공식
  gradio 데모도 오디오를 청크로 나눈다. 청크가 2개 이상이면 `_chunked` = True.
  last_timing 은 청크 누적(load_ms 합, request_ms 합, rtf = request 합 / 오디오 길이).
"""

import glob
import os
import re
import subprocess
import tempfile

import numpy as np

from .base import SttEngine

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_RUNTIME = os.path.join(_ROOT, ".temp", "VibeASR.cpp")
#: winget(WinLibs) 설치 위치 — 실행 시 libstdc++ 등 DLL 을 찾으려면 PATH 에 필요
_WINLIBS_GLOB = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft", "WinGet", "Packages", "BrechtSanders.WinLibs*", "mingw64", "bin",
)

_CHUNK_SECONDS = 60.0
_MIN_TAIL_SECONDS = 0.1

_SEG_RE = re.compile(r"^\[[^\]]*\]\s*(?:Speaker\s+\S+:\s*)?(.*)$")


class VibeVoiceBitNetEngine(SttEngine):
    name = "vibevoice-bitnet"
    model_id = "bitnet-1.5b"

    def __init__(self, language="en", device="cpu", *, threads=8, runtime=None,
                 binary=None, vae_model=None, lm_model=None, mingw_bin=None):
        super().__init__(language=language, device="cpu")  # CPU 런타임 고정
        self.threads = threads
        runtime = runtime or _DEFAULT_RUNTIME
        models = os.path.join(runtime, "models", "vibeasr")
        self._binary = binary or os.path.join(runtime, "build", "bin", "asr_infer.exe")
        self._vae = vae_model or os.path.join(models, "vibeasr-vae-encoder-i8_s.gguf")
        self._lm = lm_model or os.path.join(models, "vibeasr-lm-i2_s-embed-q6_k.gguf")
        found = glob.glob(_WINLIBS_GLOB)
        self._mingw_bin = mingw_bin or (found[0] if found else None)
        self.last_timing = {}
        self._chunked = False

    def load(self) -> None:
        """바이너리·모델 경로가 존재하는지 확인한다 (서브프로세스 방식이라 추가 로드 없음)."""
        for label, path in (
            ("asr_infer 바이너리", self._binary),
            ("VAE GGUF", self._vae),
            ("LM GGUF", self._lm),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"[{self.name}] {label} 없음: {path}\n"
                    f"  .temp/VibeASR.cpp 를 MinGW 로 빌드하고 모델을 내려받으세요."
                )

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """16k mono float32 오디오를 60초 창으로 나눠 청크별로 전사하고 공백으로 잇는다."""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        step = int(_CHUNK_SECONDS * sr)
        chunks = [audio[i:i + step] for i in range(0, len(audio), step)
                  if len(audio[i:i + step]) >= _MIN_TAIL_SECONDS * sr]
        self._chunked = len(chunks) > 1
        texts, load_ms, req_ms = [], 0.0, 0.0
        for c in chunks:
            texts.append(self._run_chunk(c, sr))
            load_ms += self.last_timing["load_ms"]
            req_ms += self.last_timing["request_ms"] or 0.0
        self.last_timing = {
            "load_ms": load_ms,
            "request_ms": req_ms,
            "rtf": req_ms / (len(audio) / sr * 1000),
            "n_chunks": len(chunks),
        }
        return " ".join(t for t in texts if t)

    def _run_chunk(self, audio: np.ndarray, sr: int) -> str:
        """청크 하나를 임시 wav 로 저장 후 asr_infer 로 전사한다."""
        import soundfile as sf

        fd, tmp_wav = tempfile.mkstemp(prefix="vibeasr_", suffix=".wav")
        os.close(fd)
        env = dict(os.environ)
        if self._mingw_bin:
            env["PATH"] = self._mingw_bin + os.pathsep + env.get("PATH", "")
        try:
            sf.write(tmp_wav, audio, sr, subtype="PCM_16")
            cmd = [
                self._binary,
                "--vae-model", self._vae,
                "--lm-model", self._lm,
                "--audio", tmp_wav,
                "-t", str(self.threads),
                "--greedy",
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=3600, env=env)
            err = proc.stderr.decode("utf-8", "replace")
            if proc.returncode != 0:
                raise RuntimeError(
                    f"[{self.name}] CLI 실패 (exit {proc.returncode}):\n  stderr: {err[-500:]}"
                )
            self._parse_timing(err)
            out = proc.stdout.decode("utf-8", "replace")
            parts = []
            for line in out.splitlines():
                m = _SEG_RE.match(line.strip())
                if m and m.group(1).strip():
                    parts.append(m.group(1).strip())
            if "Error:" in err:  # CLI 가 오류에도 exit 0 을 반환함 (예: VAE arena 할당 실패)
                raise RuntimeError(f"[{self.name}] CLI 오류 (exit 0): {err[-300:]}")
            # ponytail: 화자 JSON 이 아닌 평문 출력(첫 글자가 '[' 아님)은 그대로 쓴다
            if not parts:
                parts = [out.strip()]
            return " ".join(parts)
        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

    def _parse_timing(self, err: str) -> None:
        """stderr 의 Timing Summary 에서 모델 로드 시간과 RTF 를 뽑는다."""
        def num(pat):
            m = re.search(pat, err)
            return float(m.group(1)) if m else None

        vae, lm = num(r"VAE model loading:\s+([\d.]+) ms"), num(r"LM model loading:\s+([\d.]+) ms")
        self.last_timing = {
            "load_ms": (vae or 0) + (lm or 0),
            "rtf": num(r"\n\s+RTF:\s+([\d.]+)"),
            "request_ms": num(r"\n\s+Total:\s+([\d.]+) ms"),
        }
