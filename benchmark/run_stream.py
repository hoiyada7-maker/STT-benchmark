"""Real-time streaming STT harness.

usage: python benchmark/run_stream.py --engine nemo --setting nemotron-speech:70,6 --device cuda
           [--clip-seconds 30] [--out DIR] [--fail-backlog 30] [--rt-final-backlog 2.0] [--tag NAME] [--affinity 0-15|0,2,4|0-7,16-23|none]

Feeds ted_first10min.wav to a stream_engines adapter in real time (one model-native chunk per
chunk period), measures per-chunk compute, backlog, word emit/final latency vs reference word
times, WER/CER, CPU/GPU/RAM utilisation.
realtime_ok = rtf < 1.0 and backlog after the last chunk <= --rt-final-backlog (s); backlog > --fail-backlog
mid-run aborts. A failed attempt (either kind) is retried once; two failures -> *_FAILED_REALTIME.json, exit 4. venv-agnostic (numpy, soundfile, jiwer, psutil; torch/pynvml optional).
--affinity (default 0-15 = i7-12850HX P-cores) pins the process before the engine loads; applied list recorded as cpu_affinity.
"""
import argparse, difflib, json, os, re, subprocess, sys, threading, time
import numpy as np
import psutil

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from benchmark import dataset                      # noqa: E402
from benchmark.score import score_pair, normalize  # noqa: E402

try:
    import pynvml; pynvml.nvmlInit(); _h = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _h = None

DEFAULT_OUT = os.path.join(ROOT, ".temp", "2026-11-23-stream-bench")
REF_TIMES = os.path.join(DEFAULT_OUT, "ref_word_times_ted.json")
WALL_LIMIT = 1200.0


# ---- resource sampling (same logic/fields as .claude/skills/stt-add-model/run_one.py) ----
def _smi():
    o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.split(",")
    return float(o[0]), float(o[1])


def gpu_sample():
    """returns (util %, mem used MB)"""
    if _h is not None:
        try:
            return pynvml.nvmlDeviceGetUtilizationRates(_h).gpu, pynvml.nvmlDeviceGetMemoryInfo(_h).used / 2**20
        except Exception:
            pass
    try:
        return _smi()
    except Exception:
        return 0.0, 0.0


def ram_mb():
    """RSS of this process + all children, MB"""
    p = psutil.Process(); t = 0
    for q in [p] + p.children(recursive=True):
        try: t += q.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    return t / 2**20


class Sampler(threading.Thread):
    def __init__(self, period=0.5):
        super().__init__(daemon=True); self.period = period; self.stop_ev = threading.Event(); self.s = []
    def run(self):
        psutil.cpu_percent(interval=None)  # prime
        while not self.stop_ev.wait(self.period):
            c = psutil.cpu_percent(interval=None)
            g, m = gpu_sample()
            self.s.append((round(time.time() - self.t0, 2), c, g, m, ram_mb()))
    def start(self):
        self.t0 = time.time(); super().start()
    def stop(self):
        self.stop_ev.set(); self.join()
        s = self.s[1:]  # discard first CPU sample
        def mm(i):
            v = [x[i] for x in s]
            return (round(min(v), 1), round(sum(v) / len(v), 1), round(max(v), 1)) if v else (None,) * 3
        c, g, r = mm(1), mm(2), mm(4)
        return dict(cpu_util_min=c[0], cpu_util_avg=c[1], cpu_util_max=c[2],
                    gpu_util_min=g[0], gpu_util_avg=g[1], gpu_util_max=g[2],
                    ram_gb_min=round(r[0] / 1024, 2) if s else None, ram_gb_avg=round(r[1] / 1024, 2) if s else None,
                    ram_gb_max=round(r[2] / 1024, 2) if s else None,
                    util_samples=len(s), gpu_mem_used_peak_mb=max([x[3] for x in s], default=None),
                    util_raw=[dict(t=x[0], cpu=x[1], gpu=x[2], gpu_mem_mb=round(x[3]), ram_mb=round(x[4])) for x in s])


# ---- one real-time pass ----
def run_attempt(eng, audio, chunk_n, fail_backlog):
    """returns dict(status ok|backlog|rtf2, snaps, proc_ms, backlogs, text, finish_ms, wall_s, chunks, audio_reached_s)"""
    chunk_s = chunk_n / 16000
    n_chunks = -(-len(audio) // chunk_n)
    eng.reset()
    snaps, proc_ms, backlogs = [], [], []
    last = None
    status = "ok"
    t0 = time.perf_counter()
    for i in range(n_chunks):
        c = audio[i * chunk_n:(i + 1) * chunk_n]
        if len(c) < chunk_n:
            c = np.concatenate([c, np.zeros(chunk_n - len(c), dtype=np.float32)])
        avail = (i + 1) * chunk_s
        lag = t0 + avail - time.perf_counter()
        if lag > 0:
            time.sleep(lag)
        a = time.perf_counter()
        stable, full = eng.feed(c)
        b = time.perf_counter()
        proc_ms.append((b - a) * 1000)
        backlogs.append(max(0.0, (b - t0) - avail))
        if (stable, full) != last:
            snaps.append((b - t0, stable, full)); last = (stable, full)
        if backlogs[-1] > fail_backlog:
            status = "backlog"; break
        if b - t0 > WALL_LIMIT:
            status = "rtf2"; break
    out = dict(status=status, snaps=snaps, proc_ms=proc_ms, backlogs=backlogs, chunks=len(proc_ms),
               audio_reached_s=len(proc_ms) * chunk_s, text="", finish_ms=None)
    if status == "ok":
        a = time.perf_counter()
        text = eng.finish()
        b = time.perf_counter()
        out["text"], out["finish_ms"] = text, (b - a) * 1000
        if last != (text, text):
            snaps.append((b - t0, text, text))
    out["wall_s"] = time.perf_counter() - t0
    return out


# ---- latency metrics ----
def _words(t):
    return normalize(t).split()


def latency_metrics(snaps, final_text, ref_words, clip_s):
    """snaps: [(t, stable, full)] ; ref_words: [{w,start,end}] ; returns dict"""
    res = dict(first_result_ms=None, word_emit_ms_avg=None, word_emit_ms_p95=None,
               word_final_ms_avg=None, word_final_ms_p95=None, matched_words=0)
    if not snaps:
        return res
    rw = [(normalize(w["w"]), w["start"], w["end"]) for w in ref_words if clip_s is None or w["end"] <= clip_s]
    rw = [x for x in rw if x[0] and " " not in x[0]]
    first = next((s for s in snaps if s[2].strip()), None)
    if first is not None and rw:
        res["first_result_ms"] = (first[0] - rw[0][1]) * 1000
    final = _words(final_text)
    K = len(final)
    if K == 0 or not rw:
        return res
    vocab = {}
    ids = lambda ws: np.array([vocab.setdefault(w, len(vocab)) for w in ws], dtype=np.int64)
    fin = ids(final)
    last_bad = np.full(K, -1, dtype=np.int64)  # last snapshot index where full word k != final word k
    cmax, ts, m = [], [], 0
    for i, (t, st, fu) in enumerate(snaps):
        w = ids(_words(fu)); L = min(len(w), K)
        last_bad[:L][w[:L] != fin[:L]] = i
        if len(w) < K:
            last_bad[len(w):] = i
        m = max(m, len(_words(st))); cmax.append(m); ts.append(t)
    ts = np.array(ts)
    emit_t = ts[np.minimum(last_bad + 1, len(ts) - 1)]
    fin_t = ts[np.minimum(np.searchsorted(np.array(cmax), np.arange(K) + 1, side="left"), len(ts) - 1)]
    sm = difflib.SequenceMatcher(None, final, [x[0] for x in rw], autojunk=False)
    em, fl = [], []
    for blk in sm.get_matching_blocks():
        for j in range(blk.size):
            k, r = blk.a + j, blk.b + j
            em.append((emit_t[k] - rw[r][2]) * 1000); fl.append((fin_t[k] - rw[r][2]) * 1000)
    if em:
        res.update(word_emit_ms_avg=float(np.mean(em)), word_emit_ms_p95=float(np.percentile(em, 95)),
                   word_final_ms_avg=float(np.mean(fl)), word_final_ms_p95=float(np.percentile(fl, 95)),
                   matched_words=len(em))
    return res


def run_attempts(eng, audio, chunk_n, secs, fail_backlog, rt_final):
    """up to 2 attempts; returns (attempts, res, util, attempt_history, fail_reason or None). res has rtf, final_backlog_s."""
    hist, reason = [], None
    for attempts in (1, 2):
        smp = Sampler(); smp.start()
        try:
            res = run_attempt(eng, audio, chunk_n, fail_backlog)
        finally:
            util = smp.stop()
        done_s = secs if res["status"] == "ok" else res["audio_reached_s"]  # aborted attempts: audio actually processed
        res["rtf"] = sum(res["proc_ms"]) / 1000 / done_s if done_s else 0.0
        res["final_backlog_s"] = res["backlogs"][-1] if res["backlogs"] else 0.0
        mb = max(res["backlogs"], default=0.0)
        hist.append(dict(status=res["status"], rtf=res["rtf"], final_backlog_s=res["final_backlog_s"], max_backlog_s=mb))
        if res["status"] == "backlog":
            reason = "backlog>fail_backlog"
        elif res["status"] == "ok" and res["rtf"] >= 1.0:
            reason = "rtf>=1"
        elif res["status"] == "ok" and res["final_backlog_s"] > rt_final:
            reason = f"final_backlog>{rt_final:g}"
        else:
            reason = None; break
        hist[-1]["fail_reason"] = reason
        print(f"attempt {attempts}: realtime failure ({reason}; rtf={res['rtf']:.3f} final_backlog={res['final_backlog_s']:.2f}s "
              f"chunks={res['chunks']})", flush=True)
    return attempts, res, util, hist, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-seconds", type=float, default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--fail-backlog", type=float, default=30.0)
    ap.add_argument("--rt-final-backlog", type=float, default=2.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--affinity", default="0-15", help='CPU list e.g. "0-15", "0,2,4", "0-7,16-23"; "none" disables')
    a = ap.parse_args()
    cpu_aff = None
    if a.affinity.lower() != "none":
        try:
            cpus = sorted({c for part in a.affinity.split(",") for c in (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1)
                                                                         if "-" in part else [int(part)])})
            psutil.Process().cpu_affinity(cpus); cpu_aff = psutil.Process().cpu_affinity()
        except Exception as e:
            print(f"WARNING: cpu_affinity({a.affinity}) failed: {e}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    base_name = a.tag or f"{a.engine}_{re.sub(r'[^A-Za-z0-9.]+', '-', a.setting).strip('-')}_{a.device}"
    if a.clip_seconds:
        base_name += f"_smoke{a.clip_seconds:g}s"
    base = os.path.join(a.out, base_name)
    if os.path.exists(base + ".json"):
        print("skip", base_name); return 0

    audio, sr = dataset.load_audio(os.path.join(ROOT, "data", "audio", "ted_first10min.wav"))
    assert sr == 16000, sr
    ref = dataset.load_transcript(os.path.join(ROOT, "data", "transcripts", "ted_first10min.txt"))
    ref_words = json.load(open(REF_TIMES, encoding="utf-8"))["words"]
    if a.clip_seconds:
        audio = audio[:int(a.clip_seconds * sr)]
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    secs = len(audio) / sr

    from stream_engines import get_stream_engine
    eng = get_stream_engine(a.engine, a.device, a.setting)
    t = time.perf_counter(); eng.load(); load_s = time.perf_counter() - t
    ram_after_load = round(ram_mb() / 1024, 2)
    torch = sys.modules.get("torch")
    use_cuda = bool(torch and a.device.startswith("cuda") and torch.cuda.is_available())
    if use_cuda: torch.cuda.reset_peak_memory_stats()
    chunk_n = eng.chunk_ms * 16

    attempts, res, util, hist, reason = run_attempts(eng, audio, chunk_n, secs, a.fail_backlog, a.rt_final_backlog)
    peak = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else None
    pm = res["proc_ms"]
    stats = dict(chunk_proc_ms_avg=float(np.mean(pm)) if pm else None,
                 chunk_proc_ms_p95=float(np.percentile(pm, 95)) if pm else None,
                 max_backlog_s=max(res["backlogs"], default=0.0))
    head = dict(engine=a.engine, engine_name=eng.name, setting=a.setting, setting_label=eng.setting,
                model_id=eng.model_id, device=a.device, chunk_ms=eng.chunk_ms, smoke=bool(a.clip_seconds),
                audio_seconds=secs, load_s=load_s, attempts=attempts, cpu_affinity=cpu_aff)

    if reason:
        extra = {}
        if res["status"] == "ok":
            m_ = score_pair(ref, res["text"], audio_seconds=secs)
            extra = dict(wer_raw=m_["wer_raw"], cer_raw=m_["cer_raw"], wer_norm=m_["wer_norm"], cer_norm=m_["cer_norm"],
                         **latency_metrics(res["snaps"], res["text"], ref_words, a.clip_seconds))
        rec = dict(head, status="FAILED_REALTIME", fail_reason=reason, chunks_processed=res["chunks"],
                   audio_seconds_reached=res["audio_reached_s"], fail_backlog_s=a.fail_backlog,
                   rt_final_backlog_s=a.rt_final_backlog, rtf=res["rtf"], final_backlog_s=res["final_backlog_s"],
                   attempt_history=hist, peak_vram_gb=peak, ram_gb_after_load=ram_after_load, **stats, **extra)
        json.dump(rec, open(base + "_FAILED_REALTIME.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("FAILED_REALTIME", base_name, json.dumps(rec, ensure_ascii=False)); return 4
    if res["status"] == "rtf2":
        open(base + "_SKIPPED_RTF2.txt", "w").write(f"wall > {WALL_LIMIT:.0f} s (RTF>=2) {base_name}\n")
        print("SKIPPED_RTF2", base_name); return 3

    text = res["text"]
    with open(base + "_snapshots.jsonl", "w", encoding="utf-8") as f:
        for t_, st, fu in res["snaps"]:
            f.write(json.dumps(dict(t=round(t_, 3), stable=st, full=fu), ensure_ascii=False) + "\n")
    m = score_pair(ref, text, audio_seconds=secs)
    lat = latency_metrics(res["snaps"], text, ref_words, a.clip_seconds)
    note = f"stream chunk {eng.chunk_ms} ms real-time paced"
    rec = dict(head, wer_raw=m["wer_raw"], cer_raw=m["cer_raw"], wer_norm=m["wer_norm"], cer_norm=m["cer_norm"],
               ref_words=m["ref_words"], hyp_words=m["hyp_words"],
               rtf=res["rtf"], final_backlog_s=res["final_backlog_s"],
               rt_final_backlog_s=a.rt_final_backlog, attempt_history=hist, wall_s=res["wall_s"], finish_ms=res["finish_ms"],
               realtime_ok=True, **stats, **lat,
               peak_vram_gb=peak, ram_gb_after_load=ram_after_load, **util)
    if a.clip_seconds:  # clip-matched WER against the whisper reference words (smoke sanity only)
        cr = " ".join(w["w"] for w in ref_words if w["end"] <= a.clip_seconds)
        rec["wer_norm_clipref"] = score_pair(cr, text)["wer_norm"]
        note += f"; smoke clip {a.clip_seconds:g}s (wer_norm vs full ref not meaningful, wer_norm_clipref vs whisper words)"
    rec["note"] = note
    open(base + ".txt", "w", encoding="utf-8").write(text)
    json.dump(rec, open(base + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    f_ = lambda v: "n/a" if v is None else f"{v:.0f}"
    print(f"{base_name}: chunk={eng.chunk_ms}ms wer_norm={m['wer_norm']:.3f} wer_clipref={rec.get('wer_norm_clipref')} "
          f"rtf={rec['rtf']:.3f} proc_avg={stats['chunk_proc_ms_avg']:.0f}ms max_backlog={stats['max_backlog_s']:.2f}s "
          f"emit_avg={f_(lat['word_emit_ms_avg'])}ms final_avg={f_(lat['word_final_ms_avg'])}ms "
          f"matched={lat['matched_words']} vram={peak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
