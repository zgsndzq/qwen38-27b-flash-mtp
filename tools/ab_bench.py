"""ab_bench.py -- one-command A/B benchmark for the .bat launchers in this bundle.

Design rule: this script NEVER re-states server flags. It launches the .bat
itself through cmd, so whatever the launcher says is what gets measured. That
makes it usable for any future variant (vocab-map, chain, adaptive, ...) without
editing a single line here.

What it does, per launcher:
  1. taskkill any leftover llama-server, launch the .bat, wait for /health == 200
     (a bare TCP connect is not enough: llama-server binds the port while it is
     still loading and then answers 503).
  2. replay a fixed prompt set N times, recording tg t/s, draft stats and peak VRAM.
  3. flush results to JSON after EVERY run, so a background job that gets killed
     at the ~2 minute mark still keeps all completed data.
  4. kill the whole process tree.

Derived per-round numbers use the exact identity
      rounds = predicted_n - draft_n_accepted        (each round emits 1 bonus token)
      mean draft depth d = draft_n / rounds
which is the runtime proof that an adaptive-depth launcher really is shallower
than a fixed-depth one. (Older scratch scripts used an approximate form of this
and understated the depth -- this one is exact.)

Usage:
  python ab_bench.py --bat llmserver.bat --bat llmserver-adaptive.bat
  python ab_bench.py --bat llmserver-adaptive.bat --prompts code --runs 5 --tag smoke
  python ab_bench.py --bat llmserver.bat --bat llmserver-adaptive.bat --dry-run

Exit code 0 = every config produced data, 1 = something failed to run.
"""

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# First entry of each set is byte-identical to the prompt used in the
# 2026-09-22 adaptive A/B, so results stay comparable with those numbers.
PROMPTS = {
    "code": [
        ("code.ringbuffer",
         "Implement a Python class RingBuffer with push, pop, peek, is_full, "
         "is_empty and __len__, backed by a fixed-size list, with type hints "
         "and docstrings. Output only the code."),
        ("code.shell",
         "Write a bash pipeline that finds the 10 largest files under a directory, "
         "prints path and human-readable size sorted descending, and tolerates "
         "spaces in filenames. Output only the command with a short explanation."),
    ],
    "prose": [
        ("prose.platform",
         "Describe the feeling of standing on a station platform late at night "
         "as the last train pulls away. About 150 words of flowing prose."),
        ("prose.idempotent",
         "Explain what idempotency means in distributed systems, why retries need "
         "it, and give one concrete example. About 150 words."),
    ],
    # Chinese: added 2026-09-22 because the mined table holds almost no CJK and
    # every scene-coverage run said that is where the shortlist turns negative.
    # These are the prompts that decide whether a mixed table is worth shipping.
    "zh": [
        ("zh.explain",
         "请解释一下推测解码为什么能加速大模型推理，它什么时候会失效，"
         "以及接受率大概受哪些因素影响。用中文回答，三百字左右。"),
        ("zh.chat",
         "我想把一台单显卡机器上的推理服务从每秒三十个 token 提到四十个，"
         "你会先看哪些指标，按什么顺序排查？请给出具体步骤和理由。"),
    ],
    # Chinese code: comments and docstrings are CJK, the identifiers are not --
    # the mixture a mixed table has to serve in one request.
    "zhcode": [
        ("zhcode.ringbuffer",
         "用 Python 实现一个环形缓冲区类，包含 push、pop、peek、is_full、is_empty "
         "和 __len__ 方法，要求有完整的类型标注和中文文档注释，并给出一个使用示例。"),
    ],
    "mixed": None,  # filled below: one from each set
}
PROMPTS["mixed"] = [PROMPTS["code"][0], PROMPTS["prose"][0]]
PROMPTS["scenes"] = [PROMPTS["code"][0], PROMPTS["zh"][0], PROMPTS["zhcode"][0]]


def gpu_used():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return -1


def kill_stragglers():
    subprocess.run(["taskkill", "/F", "/T", "/IM", "llama-server.exe"],
                   capture_output=True, text=True, errors="replace")
    time.sleep(2)


def wait_health(proc, url, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False, "process exited early (rc=%s)" % proc.returncode
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200:
                    return True, "%.1fs" % (time.time() - t0)
        except Exception:
            pass
        time.sleep(1)
    return False, "timeout after %ss" % timeout


def ask(url, prompt, max_tokens):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "cache_prompt": False, "stream": False,
    }).encode("utf-8")  # UTF-8 bytes: a str body gets mangled to '?' by some clients
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read().decode("utf-8"))


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.peak = 0

    def run(self):
        while not self.stop.is_set():
            u = gpu_used()
            if u > self.peak:
                self.peak = u
            time.sleep(1.0)


def derive(tm):
    """Exact per-round figures from a llama.cpp timings block."""
    dn = tm.get("draft_n") or 0
    da = tm.get("draft_n_accepted") or 0
    pn = tm.get("predicted_n") or 0
    rounds = pn - da
    out = {
        "tg": round(tm.get("predicted_per_second") or 0, 2),
        "pred_n": pn, "draft_n": dn, "draft_acc": da,
        "accept": round(da / dn, 4) if dn else None,
    }
    if rounds > 0:
        out["rounds"] = rounds
        out["depth"] = round(dn / rounds, 3)
        out["tok_per_round"] = round(pn / rounds, 3)
    return out


def run_config(bat, port, prompts, reps, max_tokens, out_path, results, key):
    url = "http://127.0.0.1:%d" % port
    log = out_path.parent / ("%s.log" % key)
    rec = {"bat": str(bat), "port": port, "log": str(log), "runs": [], "ok": False}
    results[key] = rec

    kill_stragglers()
    with open(log, "wb") as lf:
        print("[run ] %-24s (%s)" % (key, bat.name))
        # launch through cmd so the .bat decides every server flag
        proc = subprocess.Popen(["cmd", "/c", str(bat)], stdout=lf,
                                stderr=subprocess.STDOUT, cwd=str(bat.parent),
                                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        sampler = VramSampler()
        try:
            ok, why = wait_health(proc, url)
            rec["load"] = why
            if not ok:
                rec["err"] = why
                print("        FAILED: %s  (see %s)" % (why, log))
                return
            print("        healthy in %s" % why)
            sampler.start()
            for label, prompt in prompts:
                for i in range(reps):
                    t1 = time.time()
                    try:
                        resp = ask(url, prompt, max_tokens)
                        d = derive(resp.get("timings", {}) or {})
                        d.update({"prompt": label, "rep": i + 1,
                                  "wall_s": round(time.time() - t1, 2)})
                        rec["runs"].append(d)
                        print("        %-16s #%d  tg=%6.2f  depth=%s  accept=%s" % (
                            label, i + 1, d["tg"], d.get("depth"), d.get("accept")))
                    except Exception as e:
                        rec["runs"].append({"prompt": label, "rep": i + 1,
                                            "error": "%s: %s" % (type(e).__name__, e)})
                        print("        %-16s #%d  ERROR %s" % (label, i + 1, e))
                    # flush after every run: a killed job must not lose data
                    write_results(out_path, results)
        finally:
            sampler.stop.set()
            if sampler.is_alive():   # never started when the server failed to come up
                sampler.join(timeout=5)
            rec["peak_vram"] = sampler.peak
            rec["ok"] = any("tg" in r for r in rec["runs"])
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, text=True, errors="replace")
            try:
                proc.wait(timeout=20)
            except Exception:
                pass
            time.sleep(3)
            write_results(out_path, results)

    for group in sorted({r.get("prompt") for r in rec["runs"] if "tg" in r}):
        vals = [r["tg"] for r in rec["runs"] if r.get("prompt") == group and "tg" in r]
        if not vals:
            continue
        rec.setdefault("median", {})[group] = {
            "n": len(vals), "tg_median": round(statistics.median(vals), 2),
            "tg_all": vals, "min": min(vals), "max": max(vals),
            "depth_median": statistics.median([r["depth"] for r in rec["runs"]
                                               if r.get("prompt") == group and "depth" in r] or [0]),
            "accept_median": statistics.median([r["accept"] for r in rec["runs"]
                                                if r.get("prompt") == group and r.get("accept") is not None] or [0]),
        }
    print("[done] %s  %s" % (key, json.dumps(rec.get("median", {}), ensure_ascii=False)))


def write_results(path, results):
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def report(results, out_path):
    keys = list(results)
    print("\n" + "=" * 92)
    print("results -> %s" % out_path)
    groups = sorted({g for k in keys for g in (results[k].get("median") or {})})
    for g in groups:
        print("\n-- prompt set: %s" % g)
        print("   %-24s %8s %8s %8s %8s %8s" % ("config", "median", "min", "max", "depth", "accept"))
        for k in keys:
            m = (results[k].get("median") or {}).get(g)
            if not m:
                print("   %-24s %8s  (no data)" % (k, "-"))
                continue
            print("   %-24s %8.2f %8.2f %8.2f %8.2f %8s" % (
                k, m["tg_median"], m["min"], m["max"], m["depth_median"],
                m["accept_median"]))
        if len(keys) == 2 and all((results[k].get("median") or {}).get(g) for k in keys):
            a, b = [(results[k]["median"][g]) for k in keys]
            faster = "B" if b["tg_median"] > a["tg_median"] else "A"
            overlap = not (a["max"] < b["min"] or b["max"] < a["min"])
            print("   => %s faster. delta=%+.1f%%   spread overlap: %s" % (
                faster, (b["tg_median"] / a["tg_median"] - 1) * 100,
                "YES (not significant)" if overlap else "NO (5-vs-5 style separation)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bat", action="append", required=True,
                    help="launcher .bat, relative to bundle root or absolute (repeatable)")
    ap.add_argument("--prompts", default="mixed", choices=sorted(PROMPTS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bats = [Path(b) if Path(b).is_absolute() else (ROOT / b) for b in args.bat]
    missing = [b for b in bats if not b.exists()]
    if missing:
        for b in missing:
            print("MISSING: %s" % b)
        return 1

    tag = args.tag or ("x".join(b.stem.replace("llmserver", "ls") for b in bats) + "_" + args.prompts)
    out_path = Path(args.out) if args.out else (ROOT / "tools" / ("_ab_%s.json" % tag))
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("bundle   : %s" % ROOT)
    print("prompts  : %s  (%d x %d runs)" % (args.prompts, len(PROMPTS[args.prompts]), args.runs))
    for b in bats:
        print("bat      : %s" % b)
    print("out      : %s" % out_path)
    if args.dry_run:
        print("\n[DRY RUN] nothing launched.")
        return 0

    results = {}
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
            print("resuming: %d config(s) already in %s" % (len(results), out_path.name))
        except Exception:
            results = {}

    for b in bats:
        key = b.stem
        if (results.get(key) or {}).get("ok"):
            print("[skip] %s (already ok)" % key)
            continue
        run_config(b, args.port, PROMPTS[args.prompts], args.runs,
                   args.max_tokens, out_path, results, key)

    write_results(out_path, results)
    report(results, out_path)
    return 0 if all((results.get(b.stem) or {}).get("ok") for b in bats) else 1


if __name__ == "__main__":
    sys.exit(main())
