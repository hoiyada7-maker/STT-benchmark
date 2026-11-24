"""Lazy registry of streaming engines (each adapter imports heavy deps only when requested).

Adapter module contract: expose a factory `create(device, setting)` (also accepted: `get_engine`,
`build`), or define exactly one StreamEngine subclass taking `(device, setting)`.
"""
import importlib
import inspect

_MODULES = {
    "nemo": "stream_engines.nemo_stream",
    "sherpa-zipformer": "stream_engines.sherpa_zipformer",
    "kyutai": "stream_engines.kyutai_stt",
    "voxtral": "stream_engines.voxtral_rt",
}


def get_stream_engine(name, device, setting):
    if name not in _MODULES:
        raise ValueError(f"unknown stream engine {name!r}; choose from {sorted(_MODULES)}")
    mod = importlib.import_module(_MODULES[name])
    for fn in ("create", "get_engine", "build"):
        if hasattr(mod, fn):
            return getattr(mod, fn)(device, setting)
    from .base import StreamEngine
    cls = [c for _, c in inspect.getmembers(mod, inspect.isclass)
           if issubclass(c, StreamEngine) and c is not StreamEngine and c.__module__ == mod.__name__]
    if len(cls) != 1:
        raise RuntimeError(f"{mod.__name__}: need create(device, setting) or exactly one StreamEngine subclass")
    return cls[0](device, setting)
