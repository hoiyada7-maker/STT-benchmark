"""NeMo 스트리밍 어댑터(stream_engines)를 쓰는 실시간 자막 오버레이.

저장소 루트에서 실행:
  .venv-nemo312/Scripts/python live_stream/live_caption_stream.py                 # 메뉴로 모델·장치 선택
  .venv-nemo312/Scripts/python live_stream/live_caption_stream.py --model 1       # GPU Parakeet-Unified [70,2,1]
  .venv-nemo312/Scripts/python live_stream/live_caption_stream.py --model 3 --lang ko-KR --device-index 12
  .venv-nemo312/Scripts/python live_stream/live_caption_stream.py --model 2 --wav data/audio/ted_first10min.wav \
      --quit-after 45 --log .temp/2026-11-24-live-stream/m2.log

모델: 1) GPU Parakeet-Unified [70,2,1] 160ms+80ms  2) GPU Parakeet-Unified [70,7,1] 560ms+80ms
      3) CPU Nemotron-3.5 [56,3] 320ms (--lang en-US|ko-KR|auto, ko-KR/auto 는 벤치 미측정)
종료: Esc / ✕ 버튼 / --quit-after 초.
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

# (메뉴 문구, setting, device)
MODELS = {
    1: ("GPU Parakeet-Unified (버퍼) [70,2,1] 160ms + 80ms", "parakeet-unified:70,2,1", "cuda"),
    2: ("GPU Parakeet-Unified (버퍼) [70,7,1] 560ms + 80ms", "parakeet-unified:70,7,1", "cuda"),
    3: ("CPU Nemotron-3.5 [56,3] 320ms", "nemotron-3.5:56,3", "cpu"),
}
LANGS = ["en-US", "ko-KR", "auto"]
BACKLOG_WARN_S = 5.0


def _ask(title, options, default=1):
    """번호 메뉴. Enter = default. 선택한 번호(1부터)를 돌려준다."""
    print(f"\n{title}")
    for i, o in enumerate(options, 1):
        print(f"  {i}) {o}" + ("  [기본]" if i == default else ""))
    while True:
        s = input(f"번호 [{default}]: ").strip()
        if not s:
            return default
        if s.isdigit() and 1 <= int(s) <= len(options):
            return int(s)


def _enable_ansi():
    """Windows 콘솔에서 ANSI 이스케이프(커서 이동)를 켠다."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetConsoleMode(k.GetStdHandle(-11), 7)
    except Exception:
        pass


def _level_bar(lv):
    if lv is None:
        return "신호: (열기 실패)"
    n = lv // 10
    return f"신호: {'█' * n}{'░' * (10 - n)} {lv:3d}"


def choose_device():
    from audio_source import LevelMonitor, default_loopback_index, list_devices
    devs = list_devices()
    if not devs:
        sys.exit("입력 장치가 없습니다.")
    lb = default_loopback_index()
    default = next((i for i, d in enumerate(devs, 1) if d["index"] == lb), 1)
    print("\n오디오 장치 (기본 = 시스템 소리 루프백) - 소리를 내면 신호가 올라갑니다")
    mon = LevelMonitor([d["index"] for d in devs])
    mon.start()
    _enable_ansi()

    def line(i, d):
        mark = "  [기본]" if i == default else ""
        # 신호 막대를 앞에 두고 이름을 잘라 한 줄에 맞춘다(줄이 넘치면 제자리 갱신이 어긋남)
        return (f"  {i}) {_level_bar(mon.levels.get(d['index']))}  [{d['kind']}] {d['name'][:40]}"
                f" idx={d['index']}{mark}")

    for i, d in enumerate(devs, 1):
        print(line(i, d))
    stop = threading.Event()

    def refresh():
        while not stop.wait(0.15):
            out = ["\x1b[s", f"\x1b[{len(devs)}F"]
            out += ["\x1b[2K" + line(i, d) + "\n" for i, d in enumerate(devs, 1)]
            out.append("\x1b[u")
            sys.stdout.write("".join(out))
            sys.stdout.flush()

    t = threading.Thread(target=refresh, daemon=True)
    t.start()
    try:
        while True:
            s = input(f"번호 [{default}]: ").strip()
            if not s:
                return devs[default - 1]["index"]
            if s.isdigit() and 1 <= int(s) <= len(devs):
                return devs[int(s) - 1]["index"]
    finally:
        stop.set()
        t.join(timeout=1)
        mon.stop()


class Pipeline:
    """오디오 버퍼 -> 엔진 스레드(chunk_ms 단위 feed) -> 오버레이/로그/콘솔 상태."""

    def __init__(self, eng, overlay, log_path):
        self.eng, self.overlay = eng, overlay
        self.buf, self.lock = np.zeros(0, dtype=np.float32), threading.Lock()
        self.stop_ev = threading.Event()
        self.log = open(log_path, "a", encoding="utf-8") if log_path else None
        self.feed_ms, self.max_backlog, self.n = [], 0.0, 0
        self.text, self.src = "", None

    def on_audio(self, a):
        with self.lock:
            self.buf = np.concatenate([self.buf, a])

    def backlog_s(self):
        return len(self.buf) / 16000

    def run(self, make_source, quit_after):
        self.overlay.set_text("모델 로딩 중...")
        t = time.perf_counter()
        self.eng.load()                          # 모델 로드 후에 오디오 스트림을 연다
        print(f"모델 로드 {time.perf_counter() - t:.1f}s", flush=True)
        if self.stop_ev.is_set():                # 로딩 중 창을 닫았으면 오디오를 열지 않는다
            return
        self.overlay.set_text("")
        self.src = make_source(self.on_audio)
        self.src.start()
        if quit_after:
            from PyQt6 import QtCore, QtWidgets
            threading.Timer(quit_after, lambda: QtCore.QMetaObject.invokeMethod(
                QtWidgets.QApplication.instance(), "quit", QtCore.Qt.ConnectionType.QueuedConnection)).start()
        n_chunk, last_print, warned = self.eng.chunk_ms * 16, 0.0, False
        while not self.stop_ev.is_set():
            with self.lock:
                chunk, self.buf = (self.buf[:n_chunk], self.buf[n_chunk:]) if len(self.buf) >= n_chunk else (None, self.buf)
            bl = self.backlog_s()
            self.max_backlog = max(self.max_backlog, bl)
            if chunk is None:
                self.stop_ev.wait(0.01)
                continue
            a = time.perf_counter()
            _, full = self.eng.feed(chunk)
            ms = (time.perf_counter() - a) * 1000
            self.feed_ms.append(ms)
            if full != self.text:
                self.text = full
                self.overlay.set_text(full)
                if self.log:
                    self.log.write(f"{time.strftime('%H:%M:%S')}\t{full}\n"); self.log.flush()
            if bl > BACKLOG_WARN_S and not warned:
                print(f"\n경고: 백로그 {bl:.1f}s 초과 - 모델이 실시간을 못 따라갑니다(오디오는 버리지 않음).", flush=True)
                warned = True
            if time.perf_counter() - last_print > 1:
                last_print = time.perf_counter()
                print(f"\r{self.eng.name} {self.eng.setting} | chunk {self.eng.chunk_ms}ms | feed {ms:.0f}ms | "
                      f"backlog {self.backlog_s():.2f}s   ", end="", flush=True)
        final = self.eng.finish()
        if self.log:
            self.log.write(f"{time.strftime('%H:%M:%S')}\tFINAL\t{final}\n"); self.log.close()

    def stop(self):
        self.stop_ev.set()
        if self.src:
            self.src.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=int, choices=[1, 2, 3])
    ap.add_argument("--lang", choices=LANGS, help="모델 3 전용")
    ap.add_argument("--device-index", type=int)
    ap.add_argument("--wav", help="장치 대신 wav 파일을 실시간 재생")
    ap.add_argument("--quit-after", type=float, help="오디오 시작 후 N초 뒤 자동 종료")
    ap.add_argument("--log", default=os.environ.get("CAPTION_LOG"), help="자막 갱신을 시각과 함께 기록할 파일")
    a = ap.parse_args()

    model = a.model or _ask("모델", [m[0] for m in MODELS.values()], 1)
    lang = a.lang
    if model == 3 and not lang:
        lang = LANGS[_ask("언어 (ko-KR/auto 는 벤치마크 미측정)", LANGS, 1) - 1]
    dev_idx = None
    if not a.wav:
        dev_idx = a.device_index if a.device_index is not None else choose_device()

    _, setting, device = MODELS[model]
    if model == 3:
        import psutil
        psutil.Process().cpu_affinity(list(range(16)))   # P-코어(0-15). torch/nemo import 전에 적용
        setting += f"@{lang}"

    from PyQt6 import QtWidgets
    from audio_source import AudioSource, WavSource
    from overlay import CaptionOverlay
    from stream_engines import get_stream_engine

    app = QtWidgets.QApplication(sys.argv)
    ov = CaptionOverlay()
    ov.show()
    eng = get_stream_engine("nemo", device, setting)
    pipe = Pipeline(eng, ov, a.log)
    make = (lambda cb: WavSource(a.wav, cb)) if a.wav else (lambda cb: AudioSource(dev_idx, cb))
    err = []

    def work():
        try:
            pipe.run(make, a.quit_after)
        except Exception as e:  # 엔진 스레드 예외를 콘솔에 드러내고 창을 닫는다
            import traceback; traceback.print_exc(); err.append(e)
            from PyQt6 import QtCore
            QtCore.QMetaObject.invokeMethod(app, "quit", QtCore.Qt.ConnectionType.QueuedConnection)

    th = threading.Thread(target=work)
    th.start()
    app.exec()
    pipe.stop()
    th.join()
    avg = sum(pipe.feed_ms) / len(pipe.feed_ms) if pipe.feed_ms else 0
    print(f"\n종료: feed {len(pipe.feed_ms)}회, 평균 {avg:.0f}ms, 최대 백로그 {pipe.max_backlog:.2f}s")
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
