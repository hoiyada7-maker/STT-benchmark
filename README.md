# STT 엔진 벤치마크

동일 음성 소스를 여러 STT 엔진에 태워 **정답 대비 CER/WER/RTF 로 자동 채점**하는 도구.

원래 `liveCaption`(실시간 자막) 프로젝트에 함께 있던 벤치마크 부분을 목적 분리를 위해
떼어낸 저장소다. 엔진 어댑터 계층(`engines/`)은 라이브 캡션 쪽과 동일 인터페이스를 쓰므로,
양쪽 프로젝트가 각자 복사본을 갖는다.

## 구성

```
stt-benchmark/
  engines/     base.py · sensevoice.py · sensevoice_gguf.py · faster_whisper.py · moss.py
               # SttEngine 인터페이스 — 교체 가능한 STT 엔진 어댑터 계층
  benchmark/   score.py · dataset.py · run_benchmark.py
               # wav + 정답 txt 를 매칭해 엔진별 CER/WER/RTF 산출
  data/        audio/ · transcripts/ · raw/ · reports/   # 음성·정답·리포트 (gitignore)
  .docs/       벤치 결과·계획 문서 (gitignore)
```

- `engines/` — `SttEngine` 인터페이스. `load()` 후 `transcribe(audio) -> str`.
- `benchmark/dataset.py` — `data/audio/*.wav` 와 같은 이름의 `data/transcripts/*.txt`(정답)를 매칭.
- `benchmark/score.py` — `jiwer` 로 CER/WER 계산, 전사 시간으로 RTF 산출.
- `benchmark/run_benchmark.py` — 엔진별 배치 채점 후 `data/reports/` 에 결과 저장.

## 설치

```bash
pip install -r requirements.txt
# torch 는 CUDA 빌드 별도 설치 (GPU 사용 시)
#   https://pytorch.org/get-started/locally/
```

## 사용

`data/audio/<이름>.wav` 와 `data/transcripts/<이름>.txt`(정답 전사)를 같은 이름으로 둔다.

```bash
# 기본(sensevoice + faster-whisper), 영어, GPU
python -m benchmark.run_benchmark

# 엔진·언어 지정, CPU, 앞 60초만(스모크)
python -m benchmark.run_benchmark --engine faster-whisper --language en --cpu --limit 60
```

옵션:

| 옵션 | 설명 |
|---|---|
| `--engine` | 채점할 엔진 (반복 가능). 기본: `sensevoice`, `faster-whisper` |
| `--language` | 인식 언어 (기본: `en`) |
| `--cpu` | CPU 강제 (미지정 시 `cuda:0`) |
| `--limit` | 오디오 앞 N초만 사용 (0=전체). 스모크 테스트용 |

## 데이터 준비

`data/raw/` 의 YouTube 자막(json3)·오디오는 `yt-dlp` 로 받는다(`pip install yt-dlp`).
벤치 실험에 쓴 일회성 스크립트는 `.temp/` 에 있다(전사 생성·엔진별 재채점 등).
