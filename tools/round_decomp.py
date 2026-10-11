"""round_decomp.py -- decompose ONE MTP verify round into real milliseconds.

Starts the production launcher (llmserver.bat, untouched -- this script never
re-states a single server flag) once, then replays three prompt styles x three
reps and reads llama.cpp's own per-request timing block.

The exact identity (same one ab_bench.py uses) is

    rounds           = predicted_n - draft_n_accepted      # each round emits 1 bonus token
    ms_per_round     = predicted_ms / rounds
    tokens_per_round = predicted_n / rounds
    depth            = draft_n / rounds                    # draft steps per round

`draft-mtp-adaptive` picks a different depth for different content, so the
spread across prompts/rep lets us regress the per-round wall time on depth:

    ms_per_round = fixed + step_ms * depth

    slope     = marginal cost of ONE draft step (MTP head forward + its overhead)
    intercept = the per-round cost paid even with no drafting at all
                = verify forward + sampling/rollback

That turns the earlier estimate ("~34 ms of drafting, ~16 ms of sampling" --
a residual from subtracting measured pieces) into two fitted numbers.

Reads only: the launcher is started and killed, nothing is written to the model
bundle. A port that already answers /health aborts the run before anything is
launched.
"""
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_bench as AB

ROOT = Path(__file__).resolve().parents[1]

# Three styles on purpose: the adaptive controller parks at a different depth
# for each, which is what gives the regression its leverage.
PROMPTS = [
    ("code.ringbuffer", "Implement a Python class RingBuffer with push, pop, peek, is_full, "
                        "is_empty and __len__, backed by a fixed-size list, with type hints "
                        "and docstrings. Output only the code."),
    ("prose.platform", "Describe the feeling of standing on a station platform late at night "
                       "as the last train pulls away. About 150 words of flowing prose."),
    ("zh.explain", "请解释一下推测解码为什么能加速大模型推理，它什么时候会失效，"
                   "以及接受率大概受哪些因素影响。用中文回答，三百字左右。"),
]
REPS = 3
MAX_TOKENS = 192


def headroom():
    """Refuse to run if the GPU already has a server on it."""
    try:
        # errors=replace: nvidia-smi output is not guaranteed UTF-8 and a strict
        # decode raises inside subprocess' reader thread (seen on this box).
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, errors="replace",
                             timeout=15).stdout.strip()
    except Exception as e:
        out = "nvidia-smi failed: %s" % e
    return out


def port_busy(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=3) as r:
            return r.status in (200, 503)
    except Exception:
        return False


def fetch_metrics(url):
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
            txt = r.read().decode("utf-8", "replace")
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("spec_decode_") and not line.startswith("#"):
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                out[parts[0].strip()] = float(parts[1])
    return out


def derive(tm):
    dn = tm.get("draft_n") or 0
    da = tm.get("draft_n_accepted") or 0
    pn = tm.get("predicted_n") or 0
    pm = tm.get("predicted_ms") or 0.0
    rounds = pn - da
    d = {"pred_n": pn, "draft_n": dn, "draft_acc": da, "predicted_ms": round(pm, 1),
         "tg": round(tm.get("predicted_per_second") or 0, 2)}
    if rounds > 0 and pm > 0:
        d["rounds"] = rounds
        d["ms_round"] = pm / rounds
        d["tok_round"] = pn / rounds
        d["depth"] = dn / rounds
        d["accept"] = da / dn if dn else None
    return d


def fit(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, r2) or None."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return a, b, r2


def main():
    port = 18080
    url = "http://127.0.0.1:%d" % port

    print("== pre-flight ==")
    print("gpu compute apps right now:")
    print("  %s" % (headroom().replace("\n", "\n  ") or "(none)"))
    if port_busy(url):
        print("ABORT: something already answers on %s -- not starting a second server." % url)
        return 2

    bat = ROOT / "llmserver.bat"
    if not bat.exists():
        print("ABORT: %s missing" % bat)
        return 2

    log = ROOT / "tools" / "_round_decomp.log"
    AB.kill_stragglers()
    runs, metrics = [], {}
    print("\n== launching %s ==" % bat.name)
    with open(log, "wb") as lf:
        proc = subprocess.Popen(["cmd", "/c", str(bat)], stdout=lf, stderr=subprocess.STDOUT,
                                cwd=str(bat.parent),
                                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        try:
            ok, why = AB.wait_health(proc, url)
            print("health: %s (%s)" % (ok, why))
            if not ok:
                print("FAILED to come up; see %s" % log)
                return 1

            for label, prompt in PROMPTS:
                for i in range(REPS):
                    t1 = time.time()
                    try:
                        resp = AB.ask(url, prompt, MAX_TOKENS)
                    except Exception as e:
                        print("  %-16s #%d ERROR %s" % (label, i + 1, e))
                        continue
                    tm = resp.get("timings") or {}
                    d = derive(tm)
                    if i == 0 and not runs:
                        print("  raw timings keys: %s" % sorted(tm.keys()))
                    d.update({"prompt": label, "rep": i + 1,
                              "wall_s": round(time.time() - t1, 2)})
                    runs.append(d)
                    if "ms_round" in d:
                        print("  %-16s #%d ms/round=%6.1f  tok/round=%.2f  depth=%.2f  "
                              "accept=%s  tg=%5.2f" % (
                                  label, i + 1, d["ms_round"], d["tok_round"], d["depth"],
                                  ("%.3f" % d["accept"]) if d["accept"] is not None else "-",
                                  d["tg"]))
                    else:
                        print("  %-16s #%d no spec stats in timings" % (label, i + 1))
            metrics = fetch_metrics(url)
        finally:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, text=True)
            try:
                proc.wait(timeout=20)
            except Exception:
                pass
            time.sleep(2)
            AB.kill_stragglers()

    (ROOT / "tools" / "_round_decomp.json").write_text(
        json.dumps({"runs": runs, "metrics": metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    good = [r for r in runs if "ms_round" in r]
    print("\n== per-round decomposition (%d usable runs) ==" % len(good))
    if not good:
        print("no usable runs")
        return 1

    # The bandwidth floor: one full weight read per round, nothing else.
    ms_round_med = statistics.median([r["ms_round"] for r in good])
    depth_med = statistics.median([r["depth"] for r in good])
    tok_med = statistics.median([r["tok_round"] for r in good])
    print("median: %.1f ms/round, %.2f tokens/round, depth %.2f" % (ms_round_med, tok_med, depth_med))
    print("per emitted token: %.1f ms  (=> %.1f t/s)" % (ms_round_med / tok_med,
                                                         1000.0 * tok_med / ms_round_med))

    f = fit([r["depth"] for r in good], [r["ms_round"] for r in good])
    if f:
        a, b, r2 = f
        print("\nfit  ms_round = %.1f + %.1f * depth      (r2=%.3f)" % (a, b, r2))
        print("  => marginal cost of one draft step : %.1f ms" % b)
        print("  => per-round cost at depth 0       : %.1f ms" % a)
        print("     (this is the verify forward + sampling/rollback + any fixed round cost)")
        # What the draft steps would cost if they were only their memory traffic.
        floor = 26.0 / 224.0
        print("  => the same step's memory traffic  : %.2f ms/step  (26 MB / 224 GB/s)" % floor)
        if b > 0:
            print("  => share of each draft step that is NOT math/bandwidth: %.0f%%"
                  % (100.0 * (1 - floor / b)))
        fixed_share = 100.0 * a / ms_round_med
        draft_share = 100.0 * (b * depth_med) / ms_round_med
        print("\n  round budget: %.1f%% round-fixed, %.1f%% drafting, %.1f%% unexplained"
              % (fixed_share, draft_share, max(0.0, 100 - fixed_share - draft_share)))
    if metrics:
        print("\nserver counters: %s" % json.dumps(metrics, sort_keys=True))
        if "spec_decode_num_drafts_total" in metrics:
            print("  (cross-check rounds from /metrics: %d)"
                  % int(metrics["spec_decode_num_drafts_total"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
