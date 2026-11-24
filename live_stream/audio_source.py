"""오디오 캡처 모듈 (D:\\pjt\\liveCaption\\live\\audio_source.py 축약본).

마이크 또는 시스템 소리(WASAPI 루프백)를 16 kHz mono float32 로 변환해 콜백으로 흘린다.
WavSource 는 wav 파일을 실시간 속도로 재생해 같은 콜백 인터페이스로 흘린다(마이크 없이 테스트용).
"""
import math
import threading
import time

import numpy as np
import soundfile as sf
import soxr

TARGET_RATE = 16000


def list_devices():
    """입력 장치 목록: dict(index, name, channels, rate, kind) , kind in {"mic", "loopback"}."""
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    devices, lb_idx = [], set()
    try:
        try:
            for lb in p.get_loopback_device_info_generator():
                lb_idx.add(lb["index"])
                devices.append(dict(index=lb["index"], name=lb["name"], channels=int(lb["maxInputChannels"]),
                                    rate=int(lb["defaultSampleRate"]), kind="loopback"))
        except Exception:
            pass
        seen = set()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            key = (info["name"], int(info["maxInputChannels"]))
            if info["maxInputChannels"] <= 0 or i in lb_idx or key in seen:
                continue
            seen.add(key)
            devices.append(dict(index=i, name=info["name"], channels=key[1],
                                rate=int(info["defaultSampleRate"]), kind="mic"))
    finally:
        p.terminate()
    return devices


def default_loopback_index():
    """기본 스피커의 루프백 장치 index (없으면 None)."""
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        for lb in p.get_loopback_device_info_generator():
            if out["name"] in lb["name"]:
                return lb["index"]
    except Exception:
        pass
    finally:
        p.terminate()
    return None


class LevelMonitor:
    """여러 장치를 동시에 열어 실시간 입력 레벨(0~100)을 잰다 (liveCaption LevelMonitor 축약본). None = 열기 실패."""

    def __init__(self, device_indices):
        self._idx = list(device_indices)
        self._p, self._streams = None, []
        self.levels = {i: None for i in self._idx}

    def _cb(self, idx, ch):
        import pyaudiowpatch as pa

        def cb(data, frames, t, status):
            a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                a = a.reshape(-1, ch).mean(axis=1)
            rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
            db = 20 * math.log10(rms) if rms > 1e-6 else -100.0
            self.levels[idx] = int(max(0.0, min(100.0, (db + 60.0) / 60.0 * 100.0)))
            return (None, pa.paContinue)
        return cb

    def start(self):
        import pyaudiowpatch as pa
        self._p = pa.PyAudio()
        for i in self._idx:
            try:
                info = self._p.get_device_info_by_index(i)
                ch, rate = int(info["maxInputChannels"]), int(info["defaultSampleRate"])
                self._streams.append(self._p.open(
                    format=pa.paInt16, channels=ch, rate=rate, input=True, input_device_index=i,
                    frames_per_buffer=max(1, rate * 50 // 1000), stream_callback=self._cb(i, ch)))
                self.levels[i] = 0
            except Exception:
                self.levels[i] = None

    def stop(self):
        for s in self._streams:
            try:
                s.stop_stream()
                s.close()
            except Exception:
                pass
        self._streams.clear()
        if self._p is not None:
            self._p.terminate()
            self._p = None


class AudioSource:
    """선택한 장치에서 16 kHz mono float32 오디오를 on_audio(np.ndarray) 콜백으로 전달한다."""

    def __init__(self, device_index, on_audio, chunk_ms=100):
        import pyaudiowpatch as pyaudio
        self._pa = pyaudio
        self.on_audio = on_audio
        self._p = pyaudio.PyAudio()
        info = self._p.get_device_info_by_index(device_index)
        self.channels = int(info["maxInputChannels"])
        rate = int(info["defaultSampleRate"])
        self._rs = soxr.ResampleStream(rate, TARGET_RATE, 1, dtype="float32") if rate != TARGET_RATE else None
        self._stream = self._p.open(format=pyaudio.paInt16, channels=self.channels, rate=rate, input=True,
                                    input_device_index=device_index, frames_per_buffer=max(1, rate * chunk_ms // 1000),
                                    stream_callback=self._callback, start=False)

    def _callback(self, in_data, frame_count, time_info, status):
        a = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        if self.channels > 1:
            a = a.reshape(-1, self.channels).mean(axis=1)
        if self._rs is not None:
            a = self._rs.resample_chunk(a)
        if a.size:
            self.on_audio(a)
        return (None, self._pa.paContinue)

    def start(self):
        self._stream.start_stream()

    def stop(self):
        try:
            if self._stream.is_active():
                self._stream.stop_stream()
            self._stream.close()
        finally:
            self._p.terminate()


class WavSource:
    """wav 파일을 실시간 속도로 재생하며 on_audio 콜백에 흘린다 (AudioSource 와 같은 start/stop)."""

    def __init__(self, path, on_audio, chunk_ms=100):
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        self.audio = soxr.resample(a, sr, TARGET_RATE) if sr != TARGET_RATE else a
        self.on_audio, self.step, self.chunk_ms = on_audio, TARGET_RATE * chunk_ms // 1000, chunk_ms
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        t0 = time.perf_counter()
        for n, i in enumerate(range(0, len(self.audio), self.step), 1):
            # 벽시계 기준 페이싱(sleep 누적 오차 방지)
            if self._stop.wait(max(0.0, t0 + n * self.chunk_ms / 1000 - time.perf_counter())):
                return
            self.on_audio(self.audio[i:i + self.step])

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
