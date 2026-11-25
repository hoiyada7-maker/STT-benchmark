"""Offline unit tests for native HF transformers>=5.21 Oruk Fourier 2B support in Qwen3AsrEngine."""
import sys
import types
from contextlib import contextmanager

import numpy as np
import pytest

from engines import get_engine
from engines import qwen3_asr as qwen


def test_variant_normalization_preserves_legacy_and_accepts_fourier_aliases():
    assert qwen.Qwen3AsrEngine(variant="0.6b").model_id == "Qwen3-ASR-0.6B"
    assert qwen.Qwen3AsrEngine(variant="1.7b").model_id == "Qwen3-ASR-1.7B"
    assert qwen.Qwen3AsrEngine(variant="1.7b-ko").model_id == "qwen3_asr_1.7b_ko_beta"
    assert qwen.Qwen3AsrEngine(variant="unknown").variant == "0.6b"

    for alias in ("oruk-fourier-2b", "fourier-2b", "oruk/oruk-fourier-2b", "ORUK-FOURIER-2B"):
        eng = get_engine("qwen3-asr", language="en", device="cpu", variant=alias)
        assert eng.variant == "oruk-fourier-2b"
        assert eng.model_id == "oruk-fourier-2b"
        assert eng._native_hf is True


def test_fourier_rejects_non_english_and_old_transformers(monkeypatch):
    eng_ko = qwen.Qwen3AsrEngine(language="ko", device="cpu", variant="oruk-fourier-2b")
    with pytest.raises(ValueError, match="English-only"):
        eng_ko.load()

    for bad in (None, "4.57.6", "5.20.9"):
        monkeypatch.setattr(qwen, "_transformers_version", lambda v=bad: v)
        eng_en = qwen.Qwen3AsrEngine(language="en", device="cpu", variant="oruk-fourier-2b")
        with pytest.raises(ImportError, match=r"transformers >= 5\.21"):
            eng_en.load()


def test_legacy_variant_still_uses_qwen_asr_package(monkeypatch):
    recorded = {}

    class FakeQwenModel:
        processor = types.SimpleNamespace(chat_template="<template>")

        def transcribe(self, audio=None, language=None):
            recorded["transcribe"] = (len(audio[0]), audio[1], language)
            return [types.SimpleNamespace(text=" Legacy output ")]

    fake_qwen_asr = types.ModuleType("qwen_asr")
    fake_qwen_asr.Qwen3ASRModel = types.SimpleNamespace(
        from_pretrained=lambda repo, **kw: (recorded.setdefault("load", (repo, kw)), FakeQwenModel())[1]
    )
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"

    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    eng = qwen.Qwen3AsrEngine(language="en", device="cpu", variant="1.7b")
    eng.load()
    out = eng.transcribe(np.zeros(16000, dtype=np.float32), sr=16000)

    assert recorded["load"][0] == "Qwen/Qwen3-ASR-1.7B"
    assert recorded["transcribe"] == (16000, 16000, "English")
    assert out == "Legacy output"


def test_fourier_native_hf_loads_pinned_revision_and_chunks_long_audio(monkeypatch):
    monkeypatch.setattr(qwen, "_transformers_version", lambda: "5.24.0")
    events = {"proc_loads": [], "model_loads": [], "requests": [], "chunks": 0}

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

        def __getitem__(self, item):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], slice):
                return FakeTensor((self.shape[0], self.shape[1] - (item[1].start or 0)))
            return self

    class FakeBatch(dict):
        def to(self, device, dtype=None):
            events["to"] = (device, dtype)
            return self

    class FakeProcessor:
        def apply_transcription_request(self, audio=None, language=None, return_tensors=None):
            events["requests"].append((len(audio), language, return_tensors))
            return FakeBatch({"input_ids": FakeTensor((1, 4))})

        def batch_decode(self, generated, skip_special_tokens=False):
            events["chunks"] += 1
            return [f"language English<asr_text> chunk {events['chunks']} "]

    class FakeModel:
        def to(self, dev):
            events["dev"] = dev
            return self

        def eval(self):
            events["eval"] = True
            return self

        def generate(self, **kwargs):
            return FakeTensor((1, 10))

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda repo, **kw: (events["proc_loads"].append((repo, kw)), FakeProcessor())[1]
    )
    fake_transformers.AutoModelForMultimodalLM = types.SimpleNamespace(
        from_pretrained=lambda repo, **kw: (events["model_loads"].append((repo, kw)), FakeModel())[1]
    )

    @contextmanager
    def inference_mode():
        yield

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.inference_mode = inference_mode

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    eng = qwen.Qwen3AsrEngine(language="en", device="cuda:0", variant="oruk-fourier-2b")
    eng.load()

    assert events["proc_loads"] == [("oruk/oruk-fourier-2b", {"revision": qwen._FOURIER_REVISION})]
    assert events["model_loads"] == [
        ("oruk/oruk-fourier-2b", {"dtype": "bfloat16", "revision": qwen._FOURIER_REVISION})
    ]
    assert events["dev"] == "cuda:0" and events["eval"] is True

    # 150 s audio splits into two chunks (120 s + 30 s) matching existing Qwen3-ASR chunking.
    text = eng.transcribe(np.zeros(150 * 16000, dtype=np.float32), sr=16000)
    assert eng._chunked is True
    assert text == "chunk 1 chunk 2"
    assert [frames for frames, *_ in events["requests"]] == [120 * 16000, 30 * 16000]


def test_run_benchmark_builds_fourier_and_skips_for_korean(tmp_path, monkeypatch):
    from benchmark import run_benchmark

    eng = run_benchmark.build_engine("qwen3-asr", "en", "cpu", "oruk-fourier-2b")
    assert eng.variant == "oruk-fourier-2b"
    assert eng.model_id == "oruk-fourier-2b"

    monkeypatch.setattr(run_benchmark.dataset, "list_samples", lambda: [("a.wav", "a.txt", "clip")])
    monkeypatch.setattr(run_benchmark.dataset, "load_audio", lambda p: (np.zeros(16000, dtype=np.float32), 16000))
    monkeypatch.setattr(run_benchmark.dataset, "load_transcript", lambda p: "안녕")
    monkeypatch.setattr(run_benchmark, "REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(
        run_benchmark,
        "build_engine",
        lambda *a, **kw: pytest.fail("oruk-fourier-2b must be skipped for Korean runs"),
    )

    args = types.SimpleNamespace(
        engines=["qwen3-asr"],
        model="oruk-fourier-2b",
        language="ko",
        cpu=True,
        limit=0,
        graded=False,
    )
    assert run_benchmark.run(args) == 0
