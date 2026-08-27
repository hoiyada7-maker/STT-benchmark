# STT 엔진 벤치마크

*[English](README.md) · 한국어*

동일 음성 소스를 여러 STT 엔진에 태워 **정답 대비 CER / WER / RTF 로 자동 채점**하는 도구.

## 📊 벤치마크 결과

전체 결과 — 엔진별 관련 레포, 엔진·모델별 영어/한국어 점수, 지표 용어 설명과 해석:

- [Benchmark summary (English)](2026-08-23%20142956%20stt-benchmark-summary.md)
- **[벤치마크 요약 (한국어)](2026-08-23%20142956%20stt-benchmark-summary.ko.md)** ← 여기부터

요약 — 10분 클립 기준 정규화 WER 최고 (SenseVoice·Parakeet·Moonshine·MOSS·distil-\* 포함 전체 표는 요약 문서 참고):

| 언어 | 최고 엔진 / 모델 | WER | CER |
|---|---|---|---|
| 영어 | faster-whisper · large-v3-turbo | **0.0681** | 0.0485 |
| 한국어 | faster-whisper · large-v3-turbo | **0.1865** | 0.0855 |

> 측정 환경: CPU — 12th Gen Intel Core i7-12850HX · GPU — NVIDIA RTX A2000 8GB Laptop.

## 엔진

공통 `SttEngine` 인터페이스(`engines/`)로 엔진을 교체한다. 레지스트리 key:

| key | 엔진 | 언어 |
|---|---|---|
| `faster-whisper` | OpenAI Whisper (CTranslate2); small → large-v3-turbo, distil-\* | 99개 |
| `sensevoice` | FunASR / Paraformer (비자기회귀) | 50+ |
| `sensevoice-gguf` | SenseVoice 를 llama.cpp GGUF 런타임으로(CPU) | 50+ |
| `parakeet` | NVIDIA Parakeet TDT (onnx-asr) | 영어 + 유럽어 25종 (한국어 X) |
| `moonshine` | Useful Sensors Moonshine (ONNX, 단문형) | 영어 전용 |
| `moss` | MOSS-Transcribe-Diarize (Whisper 인코더 + LLM 디코더) | 50+ |
| `qwen3-asr` | Qwen3-ASR (Qwen3 LLM 백본); 0.6B/1.7B + 한국어 파인튜닝 1.7b-ko(TeamUNIVA), `--model` 로 선택 | 52개 (ko/en 포함) |
| `nemo-asr` | NVIDIA NeMo 2026 (Parakeet-Unified · Nemotron-Speech · Nemotron-3.5), `--model` 로 선택; 전용 Python 3.12 venv 필요 | 영어 전용(2종) + 40개 ko 포함(Nemotron-3.5) |
| `granite-speech` | IBM Granite-Speech 3.3 (Whisper 인코더 + Granite LLM, 2-pass); transformers 로 GPU 구동 | 영어 + 유럽어 5종 (한국어 X) |

## 구성

```
stt-benchmark/
  engines/     base.py · faster_whisper.py · sensevoice.py · sensevoice_gguf.py
               parakeet.py · moonshine.py · moss.py · qwen3_asr.py · nemo_asr.py · granite_speech.py
               # SttEngine 인터페이스 — 교체 가능한 STT 엔진 어댑터
  benchmark/   score.py · dataset.py · run_benchmark.py · graded.py
               # wav + 정답 txt 매칭, 엔진별 CER/WER/RTF 산출
  data/        audio/ · transcripts/ · raw/ · reports/   # 음성·정답·리포트 (gitignore)
```

- `engines/` — `SttEngine` 인터페이스. `load()` 후 `transcribe(audio) -> str`.
- `benchmark/dataset.py` — `data/audio/*.wav` 와 같은 이름의 `data/transcripts/*.txt`(정답) 매칭.
- `benchmark/score.py` — `jiwer` 로 CER/WER 계산, 전사 시간으로 RTF 산출.
- `benchmark/graded.py` — 단어 단위 채점지 생성(오인식 = ~~취소선~~, 정답 채움 = [**볼드**]).
- `benchmark/run_benchmark.py` — 엔진별 배치 채점 후 `data/reports/` 에 저장.

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

| 옵션 | 설명 |
|---|---|
| `--engine` | 채점할 엔진 (반복 가능). 기본: `sensevoice`, `faster-whisper` |
| `--model` | faster-whisper 모델 크기 (기본: `small`) |
| `--language` | 인식 언어 (기본: `en`) |
| `--cpu` | CPU 강제 (미지정 시 `cuda:0`) |
| `--limit` | 오디오 앞 N초만 사용 (0=전체). 스모크 테스트용 |
| `--graded` | 엔진·파일별 채점지 생성 → `data/reports/graded/` |

## 데이터 준비

`data/raw/` 의 YouTube 자막(json3)·오디오는 `yt-dlp` 로 받는다(`pip install yt-dlp`).
벤치 실험에 쓴 일회성 스크립트는 `.temp/` 에 있다(전사 생성·엔진별 재채점 등).
