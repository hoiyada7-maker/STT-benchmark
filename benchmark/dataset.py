"""채점 데이터셋 로더.

data/audio/*.wav 와 data/transcripts/*.txt 를 동일 파일명으로 매칭한다.
(예: talk_first10min.wav ↔ talk_first10min.txt)
"""

import os

import numpy as np
import soundfile as sf

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")


def list_samples():
    """(audio_path, transcript_path, key) 목록을 반환한다."""
    samples = []
    for fname in sorted(os.listdir(AUDIO_DIR)):
        if not fname.lower().endswith(".wav"):
            continue
        key = os.path.splitext(fname)[0]
        tpath = os.path.join(TRANSCRIPTS_DIR, key + ".txt")
        if not os.path.exists(tpath):
            continue
        samples.append((os.path.join(AUDIO_DIR, fname), tpath, key))
    return samples


def load_audio(path):
    """wav → (float32 mono 16k 배열, sr)."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def load_transcript(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def load_sample(key):
    """key 에 해당하는 (audio, sr, transcript) 를 반환한다."""
    apath = os.path.join(AUDIO_DIR, key + ".wav")
    tpath = os.path.join(TRANSCRIPTS_DIR, key + ".txt")
    audio, sr = load_audio(apath)
    transcript = load_transcript(tpath)
    return audio, sr, transcript
