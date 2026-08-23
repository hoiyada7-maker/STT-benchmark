"""STT 엔진 어댑터 패키지."""

from .base import SttEngine
from .sensevoice import SenseVoiceEngine
from .sensevoice_gguf import SenseVoiceGGUFEngine
from .faster_whisper import FasterWhisperEngine
from .moss import MossTranscribeDiarizeEngine

__all__ = [
    "SttEngine",
    "SenseVoiceEngine",
    "SenseVoiceGGUFEngine",
    "FasterWhisperEngine",
    "MossTranscribeDiarizeEngine",
]


def get_engine(name: str, **kwargs) -> SttEngine:
    """이름으로 엔진 인스턴스를 만든다. 이름은 로스터의 key. """
    registry = {
        "sensevoice": SenseVoiceEngine,
        "sensevoice-gguf": SenseVoiceGGUFEngine,
        "faster-whisper": FasterWhisperEngine,
        "moss": MossTranscribeDiarizeEngine,
    }
    if name not in registry:
        raise KeyError(f"알 수 없는 엔진: {name!r} (가능: {list(registry)})")
    return registry[name](**kwargs)
