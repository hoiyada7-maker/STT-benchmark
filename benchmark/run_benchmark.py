"""배치 채점 러너.

data/ 의 모든 (오디오, 정답) 쌍에 대해 선택한 엔진들을 돌려
CER/WER/RTF 를 계산하고, 콘솔 표와 CSV/md 리포트를 출력한다.
등록된 13개 엔진(sensevoice, sensevoice-gguf, faster-whisper, moss, parakeet,
moonshine, qwen3-asr, nemo-asr, granite-speech, vibevoice-asr, notesoop, bark, lfm3-audio) 모두 --engine 으로 선택 가능하며,
faster-whisper 는 --model 로 모델 크기를, qwen3-asr(0.6b/1.7b/1.7b-ko/oruk-fourier-2b)·nemo-asr
(parakeet-unified/nemotron-speech/nemotron-3.5)는 --model 로 변형을 고를 수 있다.
granite-speech 는 영어 전용(한국어 미지원), nemo-asr 는 전용 .venv-nemo312 필요.
vibevoice-asr(7b/streaming-7b, 4-bit)는 --model 로 변형을 고르며 전용 venv(.venv-vibevoice,
.venv-vibevoice-stream) 필요.
notesoop(Notesoop-STT-Ko-v1)은 한국어 전용이며 전용 .venv-notesoop 필요.
bark(Underdog-Bark-3.0, parakeet-tdt-0.6b-v3 파인튜닝)는 영어 전용이며 전용 .venv-bark 필요.
lfm3-audio(LiquidAI LFM3-Audio-2B, bf16)는 transformers>=5.20.2 라 전용 .venv-lfm3 필요.
vibevoice-bitnet 은 VibeASR.cpp(MinGW 빌드, CPU, CLI 호출)로 실행한다 (.temp/VibeASR.cpp).
--graded 를 주면 엔진·파일별 채점지(md)도 저장한다.

실행:
    python -m benchmark.run_benchmark                 # 전체(2 엔진)
    python -m benchmark.run_benchmark --engine sensevoice
    python -m benchmark.run_benchmark --engine faster-whisper --model medium
    python -m benchmark.run_benchmark --limit 30      # 앞 30초만 스모크 테스트
    python -m benchmark.run_benchmark --graded        # 채점지(md)도 저장
"""

import argparse
import csv
import os
import sys

# Windows 콘솔(cp1252)에서 한글 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines import get_engine
from benchmark import dataset
from benchmark.graded import build_markdown
from benchmark.score import measure_transcribe, score_pair

REPORT_DIR = os.path.join(dataset.DATA_DIR, "reports")

# 한국어를 지원하지 않는 엔진
NO_KOREAN_ENGINES = {"parakeet", "moonshine", "granite-speech", "bark"}
# 한국어 전용 엔진 (영어 실행 시 건너뜀)
KOREAN_ONLY_ENGINES = {"notesoop"}


def build_engine(name, language, device, model):
    if name == "faster-whisper":
        return get_engine(name, language=language, device=device, model_size=model)
    if name in ("qwen3-asr", "nemo-asr", "vibevoice-asr"):
        return get_engine(name, language=language, device=device, variant=model)
    return get_engine(name, language=language, device=device)


def run(args):
    samples = dataset.list_samples()
    if not samples:
        print("data/audio 에 wav + data/transcripts 에 동일 이름 txt 가 없습니다.")
        return 1

    device = "cpu" if args.cpu else "cuda:0"
    rows = []

    for apath, tpath, key in samples:
        audio, sr = dataset.load_audio(apath)
        transcript = dataset.load_transcript(tpath)

        if args.limit:
            audio = audio[: int(sr * args.limit)]
            print(f"[경고] --limit 모드: 오디오만 앞 {args.limit}s 로 자름. "
                  f"정답은 전체라 지표가 부정확함(스모크 테스트 전용).")
        audio_seconds = len(audio) / sr

        print(f"\n=== {key} ({audio_seconds:.1f}s, {len(transcript.split())} 단어) ===")

        for eng_name in args.engines:
            if args.language.startswith("ko") and (
                eng_name in NO_KOREAN_ENGINES
                or (
                    eng_name == "qwen3-asr"
                    and str(args.model).strip().lower() in ("oruk-fourier-2b", "fourier-2b", "oruk/oruk-fourier-2b")
                )
            ):
                print(f"  [{eng_name}] 경고: 한국어({args.language})를 지원하지 않는 엔진/모델이라 건너뜀.")
                continue
            if not args.language.startswith("ko") and eng_name in KOREAN_ONLY_ENGINES:
                print(f"  [{eng_name}] 경고: 한국어 전용 엔진이라 {args.language} 에서는 건너뜀.")
                continue

            print(f"  [{eng_name}] 로딩...")
            engine = build_engine(eng_name, args.language, device, args.model)
            engine.load()
            print(f"  [{eng_name}] 인식 중...")
            text, elapsed = measure_transcribe(engine, audio, sr=sr)
            metrics = score_pair(transcript, text, audio_seconds=audio_seconds)
            rtf = elapsed / audio_seconds if audio_seconds else float("nan")
            row = {
                "file": key,
                "engine": engine.name,
                "model": engine.model_id,
                "wer_raw": f"{metrics['wer_raw']:.4f}",
                "cer_raw": f"{metrics['cer_raw']:.4f}",
                "wer_norm": f"{metrics['wer_norm']:.4f}",
                "cer_norm": f"{metrics['cer_norm']:.4f}",
                "rtf": f"{rtf:.3f}",
                "elapsed_s": f"{elapsed:.2f}",
                "audio_s": f"{audio_seconds:.1f}",
                "hyp_words": metrics["hyp_words"],
            }
            rows.append(row)
            print(
                f"    WER {metrics['wer_norm']:.4f} (raw {metrics['wer_raw']:.4f}) | "
                f"CER {metrics['cer_norm']:.4f} | RTF {rtf:.3f} | {elapsed:.1f}s"
            )
            print(f"    출력(앞 120): {text[:120]}")

            if args.graded:
                _save_graded(eng_name, engine, args, key, audio_seconds, transcript, text, metrics, rtf, elapsed)

    _save_report(rows)
    _print_table(rows)
    return 0


def _print_table(rows):
    print("\n" + "=" * 78)
    print(" 최종 결과")
    print("=" * 78)
    hdr = ("파일", "엔진", "모델", "WER(raw)", "CER(raw)", "WER(정규화)", "CER(정규화)", "RTF")
    print(" | ".join(hdr))
    print("-" * 78)
    for r in rows:
        print(
            f"{r['file'][:14]:<14} | {r['engine']:<14} | {r['model']:<20} | "
            f"{r['wer_raw']:>8} | {r['cer_raw']:>8} | {r['wer_norm']:>10} | "
            f"{r['cer_norm']:>10} | {r['rtf']:>5}"
        )


def _save_graded(eng_name, engine, args, key, audio_seconds, transcript, text, metrics, rtf, elapsed):
    meta = {
        "엔진 / 모델": f"{engine.name} · {engine.model_id}",
        "언어": args.language,
        "데이터": f"{key} ({audio_seconds:.0f}s)",
        "WER (raw / norm)": f"{metrics['wer_raw']:.4f} / {metrics['wer_norm']:.4f}",
        "CER (raw / norm)": f"{metrics['cer_raw']:.4f} / {metrics['cer_norm']:.4f}",
        "RTF": f"{rtf:.3f}",
        "처리시간": f"{elapsed:.1f}s",
    }
    md = build_markdown(transcript, text, title=f"채점지 — {engine.name} ({args.language})", meta=meta)

    slug = eng_name
    if eng_name == "faster-whisper":
        slug = f"{eng_name}_{args.model}"

    graded_dir = os.path.join(REPORT_DIR, "graded")
    os.makedirs(graded_dir, exist_ok=True)
    # 파일명에 데이터셋 key 를 넣어 여러 오디오를 돌려도 서로 덮어쓰지 않게 한다.
    path = os.path.join(graded_dir, f"graded_{args.language}_{key}_{slug}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"    채점지 저장: {path}")


def _save_report(rows):
    import datetime

    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(REPORT_DIR, f"report_{stamp}.csv")
    md_path = os.path.join(REPORT_DIR, f"report_{stamp}.md")

    keys = [
        "file", "engine", "model", "wer_raw", "cer_raw", "wer_norm", "cer_norm",
        "rtf", "elapsed_s", "audio_s", "hyp_words",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# STT 채점 리포트\n\n")
        f.write(f"> 생성: {stamp} · 정답은 YouTube 자동자막 기준(상대 비교)\n\n")
        f.write("| 파일 | 엔진 | 모델 | WER(raw) | CER(raw) | WER(정규화) | CER(정규화) | RTF |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['file']} | {r['engine']} | {r['model']} | {r['wer_raw']} | "
                f"{r['cer_raw']} | {r['wer_norm']} | {r['cer_norm']} | {r['rtf']} |\n"
            )

    print(f"\n리포트 저장: {csv_path}\n          {md_path}")


def main():
    ap = argparse.ArgumentParser(description="STT 엔진 배치 채점")
    ap.add_argument("--engine", action="append", dest="engines",
                    help="채점할 엔진 (반복 가능). 기본: sensevoice, faster-whisper. "
                         "가능: sensevoice, sensevoice-gguf, faster-whisper, moss, parakeet, moonshine, qwen3-asr, nemo-asr, granite-speech, vibevoice-asr, notesoop, bark, lfm3-audio")
    ap.add_argument("--model", default="small",
                    help="faster-whisper 모델 크기(기본: small), qwen3-asr 변형(0.6b/1.7b/1.7b-ko/oruk-fourier-2b), "
                         "또는 nemo-asr 변형(parakeet-unified/nemotron-speech/nemotron-3.5). 다른 엔진에는 영향 없음")
    ap.add_argument("--language", default="en", help="인식 언어 (기본: en)")
    ap.add_argument("--cpu", action="store_true", help="CPU 강제")
    ap.add_argument("--limit", type=float, default=0,
                    help="오디오 앞 N초만 사용 (0=전체). 스모크 테스트용")
    ap.add_argument("--graded", action="store_true",
                    help="엔진·파일별 채점지(md, 단어 diff)를 data/reports/graded/ 에 저장")
    args = ap.parse_args()

    if not args.engines:
        args.engines = ["sensevoice", "faster-whisper"]

    sys.exit(run(args))


if __name__ == "__main__":
    main()
