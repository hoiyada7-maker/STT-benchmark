"""LiquidAI LFM3-Audio-2B 어댑터 (transformers custom code, bf16/fp32).

`AutoModelForMultimodalLM` + `AutoProcessor`(trust_remote_code) 로 로드하며 transformers>=5.20.2
가 필요해 전용 .venv-lfm3 를 쓴다. 공식 ASR 프롬프트는 시스템 메시지
"Perform ASR in <English language name>." 이고, 텍스트만 받도록 suppress_mixed /
suppress_audio_out 을 켜서 greedy(generation_config 기본) 로 생성한다. 언어 힌트(English/Korean)를 넘긴다.
window_s 가 None 이면 전체를 한 번에, 값이 있으면 그 길이(초) 고정창으로 잘라 이어 붙인다.
"""

import numpy as np

from .base import SttEngine

_LANG = {"en": "English", "ko": "Korean"}


class Lfm3AudioEngine(SttEngine):
    name = "lfm3-audio"
    model_id = "LFM3-Audio-2B"

    def __init__(self, language="en", device="cuda:0", window_s=None, max_new_tokens=4096):
        super().__init__(language=language, device=device)
        self.window_s = window_s
        self.max_new_tokens = max_new_tokens
        self._chunked = False

    def load(self) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self._torch = torch
        dev = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self._dev = dev
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        mid = "LiquidAI/LFM3-Audio-2B"
        self._model = AutoModelForMultimodalLM.from_pretrained(
            mid, dtype=dtype, device_map=dev, trust_remote_code=True).eval()
        self._proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        self._prompt = f"Perform ASR in {_LANG.get(self.language[:2], 'English')}."

    def _one(self, audio: np.ndarray) -> str:
        conv = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": [{"type": "audio"}]},
        ]
        text = self._proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = self._proc(text=text, audio=[np.ascontiguousarray(audio, dtype=np.float32)], sampling_rate=16000,
                            return_tensors="pt").to(self._dev)
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, suppress_mixed=True,
                suppress_audio_out=True, return_dict_in_generate=True)
        return self._proc.decode(out.sequences[0, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        if not self.window_s or len(audio) <= self.window_s * sr:
            return self._one(audio)
        self._chunked = True
        step = int(self.window_s * sr)
        parts = [self._one(audio[i:i + step]) for i in range(0, len(audio), step)]
        return " ".join(p for p in parts if p).strip()
