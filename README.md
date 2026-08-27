# STT Engine Benchmark

*English (default) · [한국어](README.ko.md)*

Runs the same audio source through multiple STT engines and **auto-scores each against a reference transcript (CER / WER / RTF)**.

## 📊 Benchmark results

Full results — related repos by engine, English/Korean scores per engine & model, a metric glossary, and interpretation:

- **[Benchmark summary (English)](2026-08-23%20142956%20stt-benchmark-summary.md)** ← start here
- [벤치마크 요약 (한국어)](2026-08-23%20142956%20stt-benchmark-summary.ko.md)

Headline — best normalized WER on a 10-minute clip (see the summary for the full table incl. SenseVoice, Parakeet, Moonshine, MOSS, distil-\*):

| Language | Best engine / model | WER | CER |
|---|---|---|---|
| English | faster-whisper · large-v3-turbo | **0.0681** | 0.0485 |
| Korean  | faster-whisper · large-v3-turbo | **0.1865** | 0.0855 |

> Measured on: CPU — 12th Gen Intel Core i7-12850HX · GPU — NVIDIA RTX A2000 8GB Laptop.

## Engines

Swap engines behind a common `SttEngine` interface (`engines/`). Registry keys:

| key | engine | languages |
|---|---|---|
| `faster-whisper` | OpenAI Whisper (CTranslate2); models small → large-v3-turbo, distil-\* | 99 |
| `sensevoice` | FunASR / Paraformer (non-autoregressive) | 50+ |
| `sensevoice-gguf` | SenseVoice on llama.cpp GGUF runtime (CPU) | 50+ |
| `parakeet` | NVIDIA Parakeet TDT (onnx-asr) | English + 25 EU (no Korean) |
| `moonshine` | Useful Sensors Moonshine (ONNX, short-form) | English only |
| `moss` | MOSS-Transcribe-Diarize (Whisper enc + LLM dec) | 50+ |
| `qwen3-asr` | Qwen3-ASR (Qwen3 LLM backbone); 0.6B/1.7B + Korean fine-tune 1.7b-ko (TeamUNIVA), pick via `--model` | 52 (incl. ko/en) |
| `nemo-asr` | NVIDIA NeMo 2026 (Parakeet-Unified · Nemotron-Speech · Nemotron-3.5), pick via `--model`; needs a dedicated Python-3.12 venv | En-only (2) + 40 incl. ko (Nemotron-3.5) |
| `granite-speech` | IBM Granite-Speech 3.3 (Whisper encoder + Granite LLM, two-pass); runs on GPU via transformers | English + 5 EU (no Korean) |

## Layout

```
stt-benchmark/
  engines/     base.py · faster_whisper.py · sensevoice.py · sensevoice_gguf.py
               parakeet.py · moonshine.py · moss.py · qwen3_asr.py · nemo_asr.py · granite_speech.py
               # SttEngine interface — pluggable STT engine adapters
  benchmark/   score.py · dataset.py · run_benchmark.py · graded.py
               # match wav + reference txt, compute per-engine CER/WER/RTF
  data/        audio/ · transcripts/ · raw/ · reports/   # audio · references · reports (gitignored)
```

- `engines/` — the `SttEngine` interface: `load()`, then `transcribe(audio) -> str`.
- `benchmark/dataset.py` — matches `data/audio/*.wav` to same-named `data/transcripts/*.txt` (reference).
- `benchmark/score.py` — CER/WER via `jiwer`; RTF from transcription time.
- `benchmark/graded.py` — renders a word-level graded transcript (mis-transcribed = ~~strikethrough~~, corrections = [**bold**]).
- `benchmark/run_benchmark.py` — batch-scores per engine, writes to `data/reports/`.

## Install

```bash
pip install -r requirements.txt
# Install a CUDA build of torch separately for GPU use:
#   https://pytorch.org/get-started/locally/
```

## Usage

Place `data/audio/<name>.wav` and `data/transcripts/<name>.txt` (reference) under the same base name.

```bash
# default (sensevoice + faster-whisper), English, GPU
python -m benchmark.run_benchmark

# pick engine/language, CPU, first 60s only (smoke test)
python -m benchmark.run_benchmark --engine faster-whisper --language en --cpu --limit 60
```

| Option | Description |
|---|---|
| `--engine` | engine to score (repeatable). Default: `sensevoice`, `faster-whisper` |
| `--model` | faster-whisper model size (default: `small`) |
| `--language` | recognition language (default: `en`) |
| `--cpu` | force CPU (default: `cuda:0`) |
| `--limit` | use first N seconds only (0 = full). For smoke tests |
| `--graded` | save a graded transcript per engine/file → `data/reports/graded/` |

## Data prep

YouTube subtitles (json3) and audio under `data/raw/` are fetched with `yt-dlp` (`pip install yt-dlp`).
One-off scripts used for the benchmark experiments (reference building, per-engine re-scoring) live in `.temp/`.
