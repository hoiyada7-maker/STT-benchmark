"""STT 엔진 어댑터 패키지."""

from .base import SttEngine
from .sensevoice import SenseVoiceEngine
from .sensevoice_gguf import SenseVoiceGGUFEngine
from .faster_whisper import FasterWhisperEngine
from .moss import MossTranscribeDiarizeEngine
from .parakeet import ParakeetEngine
from .moonshine import MoonshineEngine
from .qwen3_asr import Qwen3AsrEngine
from .nemo_asr import NemoAsrEngine
from .granite_speech import GraniteSpeechEngine
from .vibevoice_bitnet import VibeVoiceBitNetEngine
from .vibevoice_asr import VibeVoiceAsrEngine
from .notesoop import NotesoopEngine
from .bark import BarkEngine
from .lfm3_audio import Lfm3AudioEngine

__all__ = [
    "SttEngine",
    "SenseVoiceEngine",
    "SenseVoiceGGUFEngine",
    "FasterWhisperEngine",
    "MossTranscribeDiarizeEngine",
    "ParakeetEngine",
    "MoonshineEngine",
    "Qwen3AsrEngine",
    "NemoAsrEngine",
    "GraniteSpeechEngine",
    "VibeVoiceBitNetEngine",
    "VibeVoiceAsrEngine",
    "NotesoopEngine",
    "BarkEngine",
    "Lfm3AudioEngine",
]


def get_engine(name: str, **kwargs) -> SttEngine:
    """이름으로 엔진 인스턴스를 만든다. 이름은 로스터의 key. """
    registry = {
        "sensevoice": SenseVoiceEngine,
        "sensevoice-gguf": SenseVoiceGGUFEngine,
        "faster-whisper": FasterWhisperEngine,
        "moss": MossTranscribeDiarizeEngine,
        "parakeet": ParakeetEngine,
        "moonshine": MoonshineEngine,
        "qwen3-asr": Qwen3AsrEngine,
        "nemo-asr": NemoAsrEngine,
        "granite-speech": GraniteSpeechEngine,
        "vibevoice-bitnet": VibeVoiceBitNetEngine,
        "vibevoice-asr": VibeVoiceAsrEngine,
        "notesoop": NotesoopEngine,
        "bark": BarkEngine,
        "lfm3-audio": Lfm3AudioEngine,
    }
    if name not in registry:
        raise KeyError(f"알 수 없는 엔진: {name!r} (가능: {list(registry)})")
    return registry[name](**kwargs)
