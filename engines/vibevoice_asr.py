"""Microsoft VibeVoice-ASR 어댑터 (7B / Streaming-7B, bitsandbytes 4-bit).

두 변형은 로딩 경로가 다르고 transformers 버전 요구가 충돌하므로 전용 venv 가 따로 필요하다.
  - "7b"           : HF 포트 `microsoft/VibeVoice-ASR-HF` (transformers>=5.3,
                     `VibeVoiceAsrForConditionalGeneration` + `AutoProcessor`) → .venv-vibevoice
  - "streaming-7b" : `microsoft/VibeVoice-ASR-Streaming-7B` 를 공식 저장소(`pip install -e`,
                     transformers<5 고정)의 `streaming_generate()` 로 구동 → .venv-vibevoice-stream
둘 다 LM(Linear)만 nf4(bf16 연산)로 양자화하고, 음향/의미 토크나이저·커넥터·lm_head 는
bf16 으로 둔다. flash-attn 없이 streaming 은 sdpa, 7b 는 기본 attn 을 쓴다.

VibeVoice-ASR 은 언어 지정 파라미터가 없는 자동 감지 모델이라 language 힌트는 쓰지 않는다
(en/ko 모두 동일 호출). 출력은 화자/타임스탬프가 붙은 구조화 JSON 이므로 Content 만 뽑아
공백으로 이어 붙인다. 7b 는 600초 클립을 한 번에 넣는다(최대 60분 지원).
"""

import re

import numpy as np

from .base import SttEngine

_HF_ID = {
    "7b": "microsoft/VibeVoice-ASR-HF",
    "streaming-7b": "microsoft/VibeVoice-ASR-Streaming-7B",
}
_SR = 24000  # 두 변형 모두 24 kHz 입력
_MAX_NEW_TOKENS = 20000  # 600초 한국어 JSON 출력도 넉넉히 담는 상한 (7b)
_SKIP = ["lm_head", "acoustic_tokenizer", "semantic_tokenizer", "acoustic_connector",
         "semantic_connector", "multi_modal_projector"]


class VibeVoiceAsrEngine(SttEngine):
    name = "vibevoice-asr"

    def __init__(self, language="en", device="cuda:0", variant="7b"):
        super().__init__(language=language, device=device)
        variant = str(variant).lower()
        if variant not in _HF_ID:
            raise ValueError(f"variant 는 {list(_HF_ID)} 중 하나여야 함: {variant!r}")
        self.variant = variant
        self._model_name = _HF_ID[variant]
        self.model_id = f"VibeVoice-ASR-{variant}-4bit"
        self._chunked = False

    def load(self) -> None:
        import torch
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("4-bit(bitsandbytes) 는 CUDA 가 필요하다")
        self._torch = torch
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=_SKIP,
        )
        if self.variant == "7b":
            from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                self._model_name, quantization_config=bnb, device_map="cuda:0",
                dtype=torch.bfloat16,  # 토크나이저 서브모델이 sdpa 미지원이라 기본 attn 사용
            ).eval()
            # 8GB 카드 보호: (1) 마스크+GQA 조합의 sdpa 가 math 커널로 폴백해 600초 프리필에
            # ~6GB 를 더 쓰므로 repeat_kv 경로(효율 커널)로 강제, (2) 토크나이저 청크를 60→10초로
            # 줄인다(모델 카드가 OOM 시 조정하라는 노브). 수치는 동일, 피크 VRAM 만 감소.
            import transformers.integrations.sdpa_attention as _sa

            _sa.use_gqa_in_sdpa = lambda *a, **k: False
            self._model.config.acoustic_tokenizer_chunk_size = 240000
        else:
            import json

            from huggingface_hub import hf_hub_download
            from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
            from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

            with open(hf_hub_download(self._model_name, "preprocessor_config.json")) as f:
                cfg = json.load(f)
            frame = cfg["speech_tok_compress_ratio"] / cfg["target_sample_rate"]
            self._chunk_sec = cfg["chunk_frames"] * frame
            self._delay_sec = cfg["lookahead_frames"] * frame
            self._tokenizer = VibeVoiceASRProcessor.from_pretrained(self._model_name).tokenizer
            self._model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                self._model_name, quantization_config=bnb, device_map="cuda:0",
                dtype=torch.bfloat16, attn_implementation="sdpa",
            ).eval()

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        import soxr

        torch = self._torch
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        if sr != _SR:
            audio = np.ascontiguousarray(soxr.resample(audio, sr, _SR), dtype=np.float32)
        if self.variant == "7b":
            return self._run_7b(audio)

        # streaming: 청크마다 방출된 텍스트를 이어 붙인다 (데모 스크립트와 동일 호출)
        parts = []
        with torch.inference_mode():
            for _i, _n, text in self._model.streaming_generate(
                audio_tensor=torch.from_numpy(audio), tokenizer=self._tokenizer,
                chunk_duration=self._chunk_sec, text_audio_delay=self._delay_sec,
                sample_rate=_SR, max_new_tokens_per_chunk=256, temperature=0.0,
            ):
                parts.append(text)
        return _clean("".join(parts))

    def _run_7b(self, audio: np.ndarray) -> str:
        torch = self._torch
        model = self._model
        inputs = self._processor.apply_transcription_request(audio=audio).to(
            model.device, model.dtype
        )
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)
        ids = ids[:, inputs["input_ids"].shape[1]:]
        parsed = self._processor.decode(ids, return_format="parsed")[0]
        if isinstance(parsed, list):
            return _clean(" ".join(str(d.get("Content", "")) for d in parsed if isinstance(d, dict)))
        # 파싱 실패 시 원문(문자열) — 마크업이 섞였을 수 있으므로 호출측이 점검한다
        return str(parsed).strip()


def _clean(text: str) -> str:
    """streaming 출력의 `Speaker N:` 라벨과 `[Music]` 류 비음성 태그를 제거하고 공백을 정규화한다."""
    text = re.sub(r"Speaker\s*\d+\s*:", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return " ".join(text.split())
