"""IBM Granite Speech 어댑터 (granite-speech-3.3-2b, transformers 백엔드).

`AutoProcessor` + `AutoModelForSpeechSeq2Seq` 로 로드한다(모델 카드 기준 API).
호출 규약이 독특해서 그대로 옮긴다:
  1. tokenizer.apply_chat_template 로 시스템/유저 프롬프트를 문자열로 만든다.
     유저 프롬프트에 오디오 자리표시자 `<|audio|>` 를 넣어야 한다.
  2. processor(prompt, wav, ...) 에 위 문자열과 오디오 텐서(채널, 샘플)를 같이 넣는다.
  3. model.generate() 후, 입력 토큰 길이만큼 잘라내고 tokenizer.batch_decode 로
     디코드한다(입력 프롬프트가 그대로 에코되므로 앞부분을 잘라내야 함).
내부적으로 LoRA(rank=64) 어댑터를 쓰므로 `peft` 패키지가 필요하다.

영어 등 5개 언어만 지원하는 short-form 모델이라, 채점용 600초 클립을 한 번에
넣으면 문맥 예산을 넘겨 품질이 무너진다. 항상 30초(_CHUNK_SECONDS) 고정창으로
나눠 각각 전사한 뒤 이어 붙인다.
"""

import numpy as np
import torch

from .base import SttEngine

_CHUNK_SECONDS = 30.0
_SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: April 9, 2025.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)
_USER_PROMPT = "<|audio|>can you transcribe the speech into a written format?"


class GraniteSpeechEngine(SttEngine):
    name = "granite-speech"

    def __init__(self, language="en", device="cpu", variant="2b"):
        super().__init__(language=language, device=device)
        model_name = "ibm-granite/granite-speech-3.3-2b" if str(variant).lower() == "2b" \
            else f"ibm-granite/granite-speech-3.3-{variant}"
        self._model_name = model_name
        self.model_id = model_name.split("/")[-1]

    def load(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        dev = "cuda:0" if str(self.device).startswith("cuda") else "cpu"
        self._dev = dev
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._tokenizer = self._processor.tokenizer
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_name, device_map=dev, torch_dtype=torch.bfloat16
        )
        chat = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT},
        ]
        self._prompt = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        self._chunked = False

    def _one(self, audio: np.ndarray, sr: int) -> str:
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        model_inputs = self._processor(
            self._prompt, wav, device=self._dev, return_tensors="pt"
        ).to(self._dev)
        model_outputs = self._model.generate(
            **model_inputs, max_new_tokens=200, do_sample=False, num_beams=1
        )
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
        text = self._tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )
        return text[0].strip() if text else ""

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        step = int(_CHUNK_SECONDS * sr)
        if len(audio) <= step:
            return self._one(audio, sr)

        self._chunked = True
        parts = []
        # STFT(mel) 이 n_fft 보다 짧은 입력에서 깨지므로, 창 경계에서 남는 초소형
        # 꼬리(수십 ms)는 건너뛴다. 0.1s 미만은 유의미한 발화가 아니다.
        min_samples = int(0.1 * sr)
        for start in range(0, len(audio), step):
            chunk = audio[start : start + step]
            if len(chunk) < min_samples:
                continue
            text = self._one(chunk, sr)
            if text:
                parts.append(text)
        return " ".join(parts).strip()
