"""Hy-MT2-1.8B(Q4_K_M) CPU 영->한 번역기: llama.cpp llama-server(-ngl 0) + 순서 보장 작업 스레드.

벤치(translation-benchmark rebench 의 cpu_hymt2_1.8b_card)와 같은 모델 파일·raw 템플릿·카드 프롬프트·샘플링을 쓴다.
"""
import os
import queue
import subprocess
import threading
import time

import psutil
import requests

BENCH = r"D:\pjt\translation-benchmark\.temp"
SERVER = os.path.join(BENCH, "2026-11-13-chrf-bench", "llama_cpp", "llama-server.exe")
GGUF = "D:/ollama/models/blobs/sha256-dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699"
PORT = 18090                                   # 벤치는 18080/18081
TPL = "<｜hy_begin▁of▁sentence｜><｜hy_User｜>{p}<｜hy_Assistant｜>"
STOP = ["<｜hy_place▁holder▁no▁2｜>"]
CARD = ("Translate the following text into Korean. Note that you should only output the "
        "translated result without any additional explanation: {t}")


def _post(out):
    if "</think>" in out:
        out = out.split("</think>", 1)[1]
    return out.strip().strip('"').strip("'").strip()


class Translator:
    def __init__(self, on_result, threads=8, cpus=None, port=PORT, log_path=None):
        self.cb, self.threads, self.cpus, self.port = on_result, threads, cpus, port
        self.url = f"http://127.0.0.1:{port}"
        self.log_path, self.proc = log_path, None
        self.q = queue.Queue()
        self.th = threading.Thread(target=self._work, daemon=True)

    def _healthy(self):
        try:
            return requests.get(self.url + "/health", timeout=2).status_code == 200
        except Exception:
            return False

    def start(self, timeout=300):
        if not self._healthy():                # 이미 떠 있으면 재사용(종료도 하지 않는다)
            cmd = [SERVER, "-m", GGUF, "--port", str(self.port), "-c", "2048", "-ngl", "0", "-t", str(self.threads)]
            out = open(self.log_path, "w", encoding="utf-8") if self.log_path else subprocess.DEVNULL
            self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
            if self.cpus:
                psutil.Process(self.proc.pid).cpu_affinity(self.cpus)
            t0 = time.time()
            while not self._healthy():
                if self.proc.poll() is not None or time.time() - t0 > timeout:
                    self.stop_server()
                    raise RuntimeError("llama-server 기동 실패")
                time.sleep(1)
        self.translate("Hello.")               # 워밍업
        self.th.start()

    def translate(self, text):
        body = {"prompt": TPL.replace("{p}", CARD.replace("{t}", text)), "temperature": 0,
                "n_predict": 256, "stop": STOP}
        r = requests.post(self.url + "/completion", json=body, timeout=120)
        r.raise_for_status()
        return _post(r.json()["content"])

    def submit(self, en):
        self.q.put((en, time.perf_counter()))

    def _work(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            en, t0 = item
            try:
                ko = self.translate(en)
            except Exception as e:
                ko = f"[번역 실패: {e}]"
            self.cb(en, ko, t0, time.perf_counter())

    def stop(self, timeout=10):
        """대기열을 비울 때까지 timeout 초 기다린 뒤 우리가 띄운 서버만 종료."""
        self.q.put(None)
        if self.th.is_alive():
            self.th.join(timeout)
        self.stop_server()

    def stop_server(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except Exception:
                pass
