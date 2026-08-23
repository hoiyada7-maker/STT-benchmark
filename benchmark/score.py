"""채점 지표: 정규화, CER, WER, RTF.

정답과 엔진 출력을 비교해 문자/단어 오류율을 계산한다.
YouTube 자동자막 특성(화자 표시 `>>` 등)은 정규화로 제거하고,
"정규화 전/후" 두 벌을 모두 산출한다.
"""

import re
import string
import time

import jiwer


def normalize(text: str) -> str:
    """소문자화, 문장부호·화자 표시 제거, 공백 정규화."""
    text = text.lower()
    # YouTube 자동자막 화자 전환 표시 (예: ">> i'm doing a lot...")
    text = re.sub(r">+", " ", text)
    # 문장부호 제거 (아포스트로피는 유지: don't → dont 가 아닌 단어 단위 유지 위해
    # 아래에서 별도 처리 없이 남겨도 WER 에 큰 영향 없으므로 제거 대상에서 제외)
    text = "".join(ch for ch in text if ch not in string.punctuation or ch == "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_metric(fn, ref, hyp):
    try:
        return float(fn(ref, hyp))
    except Exception:
        return float("nan")


def score_pair(reference: str, hypothesis: str, audio_seconds: float = None):
    """정답/출력 한 쌍에 대한 지표를 dict 로 반환한다."""
    ref_norm = normalize(reference)
    hyp_norm = normalize(hypothesis)

    result = {
        "wer_raw": _safe_metric(jiwer.wer, reference, hypothesis),
        "cer_raw": _safe_metric(jiwer.cer, reference, hypothesis),
        "wer_norm": _safe_metric(jiwer.wer, ref_norm, hyp_norm),
        "cer_norm": _safe_metric(jiwer.cer, ref_norm, hyp_norm),
        "ref_words": len(reference.split()),
        "hyp_words": len(hypothesis.split()),
        "ref_chars": len(reference),
        "hyp_chars": len(hypothesis),
    }

    if audio_seconds:
        result["audio_seconds"] = audio_seconds
    return result


def measure_transcribe(engine, audio, sr=16000):
    """transcribe 호출을 타이밍 재고 (텍스트, 경과초) 를 반환한다."""
    t0 = time.perf_counter()
    text = engine.transcribe(audio, sr=sr)
    elapsed = time.perf_counter() - t0
    return text, elapsed
