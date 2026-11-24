"""Streaming STT engine interface (shared by benchmark/run_stream.py and all adapters)."""


class StreamEngine:
    name: str = ""          # e.g. "nemo-nemotron-speech"
    model_id: str = ""
    setting: str = ""       # human label, e.g. "[70,6] 560ms"
    chunk_ms: int = 0       # model-native input chunk length in ms
    sample_rate = 16000     # harness always feeds 16 kHz mono float32

    def load(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Start a fresh stream."""
        raise NotImplementedError

    def feed(self, chunk):
        """chunk: float32 1-D, len == chunk_ms*16. Returns (stable_text, full_text) for the whole stream so far."""
        raise NotImplementedError

    def finish(self) -> str:
        """Flush and return the final full text."""
        raise NotImplementedError
