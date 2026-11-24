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

실시간 영->한 번역(기본 켜짐, --no-translate 로 끔): 엔진의 확정 텍스트를 문장으로 잘라(segmenter.py)
Hy-MT2-1.8B(CPU, llama.cpp llama-server 포트 18090, translator.py)로 번역해 별도 오버레이(자막창 아래, 노란색,
최근 3문장)에 표시한다. --log 에는 `SENT<TAB>시각<TAB>영문`, `TR<TAB>시각<TAB>지연ms<TAB>영문<TAB>한글` 줄이 추가된다.
  .venv-nemo312/Scripts/python live_stream/live_caption_stream.py --model 1 --pause-s 1.5 --max-words 30 --tr-threads 8
문장 분할: ① . ? ! 뒤에 공백+다음 단어가 시작되면(약어·소수점 제외) ② 확정 텍스트가 --pause-s 초 안 늘면 남은 조각
          ③ 조각이 --max-words 단어를 넘으면 마지막 쉼표/세미콜론(없으면 max-words)에서 잘라 왼쪽을 확정.
CPU 분할: GPU STT(모델 1, 2) -> 번역 서버 코어 0-15, 스레드 8(기본).
          CPU STT(모델 3) -> STT 0-11, 번역 서버 12-15 스레드 4 (STT 와 CPU 를 나눠 써서 실시간이 밀릴 수 있음).
"""
import argparse
import json
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

    def __init__(self, eng, overlay, log_path, seg=None, tr=None):
        self.eng, self.overlay = eng, overlay
        self.seg, self.tr = seg, tr
        self.buf, self.lock = np.zeros(0, dtype=np.float32), threading.Lock()
        self.stop_ev = threading.Event()
        self.log = open(log_path, "a", encoding="utf-8") if log_path else None
        self.feed_ms, self.max_backlog, self.n = [], 0.0, 0
        self.text, self.src = "", None
        self.t0 = time.perf_counter()
        self.log_lock = threading.Lock()
        self.timeline, self.ta0 = None, None     # --timeline: 오디오 시작 기준 이벤트 기록
        self.last_in = time.perf_counter()       # 마지막 오디오 도착(또는 무음 보충) 시각

    def tl(self, **ev):
        if self.timeline and self.ta0 is not None:
            ev = {"t": round(time.perf_counter() - self.ta0, 3), **ev}
            with self.log_lock:
                self.timeline.write(json.dumps(ev, ensure_ascii=False) + "\n"); self.timeline.flush()

    def _log(self, line):
        if self.log:
            with self.log_lock:
                if not self.log.closed:
                    self.log.write(line + "\n"); self.log.flush()

    def _emit(self, sents):
        for en in sents:
            self._log(f"SENT\t{time.perf_counter() - self.t0:.2f}\t{en}")
            self.tl(type="sent", en=en)
            self.tr.submit(en)

    def on_audio(self, a):
        with self.lock:
            self.buf = np.concatenate([self.buf, a])
            self.last_in = time.perf_counter()

    def fill_silence(self, n_chunk, gap_s=0.2):
        """WASAPI loopback 은 소리가 없으면 콜백이 오지 않는다 -> 도착 공백만큼 무음을 채워 시간을 실시간으로 흐르게 한다."""
        with self.lock:
            now = time.perf_counter()
            if len(self.buf) < n_chunk and now - self.last_in > gap_s:
                self.buf = np.concatenate([self.buf, np.zeros(int((now - self.last_in) * 16000), dtype=np.float32)])
                self.last_in = now

    def backlog_s(self):
        return len(self.buf) / 16000

    def run(self, make_source, quit_after):
        self.overlay.set_text("모델 로딩 중...")
        t = time.perf_counter()
        self.eng.load()                          # 모델 로드 후에 오디오 스트림을 연다
        print(f"모델 로드 {time.perf_counter() - t:.1f}s", flush=True)
        if self.stop_ev.is_set():                # 로딩 중 창을 닫았으면 오디오를 열지 않는다
            return
        self.overlay.set_text("듣는 중... (소리를 재생하거나 말해 보세요)")   # 첫 자막이 나오면 전체 교체됨
        self.src = make_source(self.on_audio)
        self.ta0 = time.perf_counter()
        self.src.start()
        if quit_after:
            from PyQt6 import QtCore, QtWidgets
            threading.Timer(quit_after, lambda: QtCore.QMetaObject.invokeMethod(
                QtWidgets.QApplication.instance(), "quit", QtCore.Qt.ConnectionType.QueuedConnection)).start()
        n_chunk, last_print, warned = self.eng.chunk_ms * 16, 0.0, False
        self.last_in = time.perf_counter()
        while not self.stop_ev.is_set():
            self.fill_silence(n_chunk)
            with self.lock:
                chunk, self.buf = (self.buf[:n_chunk], self.buf[n_chunk:]) if len(self.buf) >= n_chunk else (None, self.buf)
            bl = self.backlog_s()
            self.max_backlog = max(self.max_backlog, bl)
            if chunk is None:
                if self.seg:
                    self._emit(self.seg.tick(self.text))
                self.stop_ev.wait(0.01)
                continue
            a = time.perf_counter()
            _, full = self.eng.feed(chunk)
            ms = (time.perf_counter() - a) * 1000
            self.feed_ms.append(ms)
            if full != self.text:
                self.text = full
                self.overlay.set_text(full)
                self.tl(type="cap", text=full)
                self._log(f"{time.strftime('%H:%M:%S')}\t{full}")
            if self.seg:
                self._emit(self.seg.feed(self.text))
            if bl > BACKLOG_WARN_S and not warned:
                print(f"\n경고: 백로그 {bl:.1f}s 초과 - 모델이 실시간을 못 따라갑니다(오디오는 버리지 않음).", flush=True)
                warned = True
            if time.perf_counter() - last_print > 1:
                last_print = time.perf_counter()
                print(f"\r{self.eng.name} {self.eng.setting} | chunk {self.eng.chunk_ms}ms | feed {ms:.0f}ms | "
                      f"backlog {self.backlog_s():.2f}s   ", end="", flush=True)
        final = self.eng.finish()
        if self.seg:                             # 남은 조각을 번역하고(짧은 timeout) 서버 종료
            self._emit(self.seg.flush(final or self.text))
            self.tr.stop(timeout=10)
        self._log(f"{time.strftime('%H:%M:%S')}\tFINAL\t{final}")
        if self.log:
            self.log.close()

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
    ap.add_argument("--timeline", help="자막/번역 이벤트를 오디오 시작 기준 시각과 함께 JSON 줄로 기록할 파일")
    ap.add_argument("--no-translate", action="store_true", help="영->한 번역 끄기")
    ap.add_argument("--pause-s", type=float, default=1.5, help="확정 텍스트가 이 시간 안 늘면 문장 확정")
    ap.add_argument("--max-words", type=int, default=30, help="이 단어 수를 넘으면 강제로 자름")
    ap.add_argument("--tr-threads", type=int, help="번역 서버 스레드(기본: GPU STT 8, CPU STT 4)")
    a = ap.parse_args()

    model = a.model or _ask("모델", [m[0] for m in MODELS.values()], 1)
    lang = a.lang
    if model == 3 and not lang:
        lang = LANGS[_ask("언어 (ko-KR/auto 는 벤치마크 미측정)", LANGS, 1) - 1]
    dev_idx = None
    if not a.wav:
        dev_idx = a.device_index if a.device_index is not None else choose_device()

    _, setting, device = MODELS[model]
    tr_on = not a.no_translate
    if model == 3:
        import psutil
        psutil.Process().cpu_affinity(list(range(12 if tr_on else 16)))   # P-코어. torch/nemo import 전에 적용
        setting += f"@{lang}"
        if tr_on:
            print("경고: CPU STT 와 번역이 CPU 를 나눠 씁니다(STT 0-11, 번역 12-15) - 실시간이 밀릴 수 있습니다.", flush=True)
    tr_cpus = list(range(12, 16)) if model == 3 else list(range(16))
    tr_threads = a.tr_threads or (4 if model == 3 else 8)

    from PyQt6 import QtWidgets
    from audio_source import AudioSource, WavSource
    from overlay import CaptionOverlay
    from stream_engines import get_stream_engine

    app = QtWidgets.QApplication(sys.argv)
    ov = CaptionOverlay()
    ov.show()
    seg = tr = tov = None
    delays = []
    if tr_on:
        from segmenter import SentenceSegmenter
        from translator import Translator
        ov.move(ov.x(), ov.y() - ov.height() - 10)       # 번역창이 아래 자리를 쓰도록 자막창을 위로
        tov = CaptionOverlay(color="#ffe98a", below=ov)
        tov.show()
        tov.set_text("번역 서버 로딩 중...")
        seg = SentenceSegmenter(a.pause_s, a.max_words)
        shown = []

        def on_tr(en, ko, t0, t1):
            shown.append(ko)
            pipe.tl(type="tr", en=en, ko=ko)
            tov.set_text("\n".join(shown))
            pipe._log(f"TR\t{t1 - pipe.t0:.2f}\t{(t1 - t0) * 1000:.0f}\t{en}\t{ko}")
            delays.append((t1 - t0) * 1000)

        tr = Translator(on_tr, tr_threads, tr_cpus)
    eng = get_stream_engine("nemo", device, setting)
    pipe = Pipeline(eng, ov, a.log, seg, tr)
    if a.timeline:
        pipe.timeline = open(a.timeline, "a", encoding="utf-8")
    make = (lambda cb: WavSource(a.wav, cb)) if a.wav else (lambda cb: AudioSource(dev_idx, cb))
    err = []

    def work():
        try:
            if tr:
                tr.start()                       # 서버 기동·워밍업
                tov.set_text("")
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
    if tr:
        tr.stop_server()
        if delays:
            print(f"\n번역 {len(delays)}문장, 지연 평균 {sum(delays) / len(delays):.0f}ms, 최대 {max(delays):.0f}ms", end="")
    print(f"\n종료: feed {len(pipe.feed_ms)}회, 평균 {avg:.0f}ms, 최대 백로그 {pipe.max_backlog:.2f}s")
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
