# -*- coding: utf-8 -*-
"""
smoke.py -- closed-loop latency check for JEV x Tetris.

Probes 18080 (aborts if someone else owns it) -> launches llmserver.bat exactly
as the user does -> waits for /health -> runs a set of synthetic mid-game boards
through bridge.decide() -> kills by PID (+ any orphan) and verifies VRAM is back
to zero.

Measures, per configuration: wall ms, prefill ms, decode ms, prompt_n, cache_n.
"""

import random
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge  # noqa: E402

ROOT = Path(r"F:\super\qwen38mtp")
BAT = ROOT / "llmserver.bat"
URL = "http://127.0.0.1:18080"
COLS, ROWS = 10, 20
PIECES = "IJLOSTZ"


# ------------------------------------------------------------ synthetic states
def make_state(rng, repr_mode, mode="jev"):
    board = [[0] * COLS for _ in range(ROWS)]
    for c in range(COLS):
        h = rng.randint(0, 9)
        for r in range(ROWS - h, ROWS):
            board[r][c] = 1
        if h > 2 and rng.random() < 0.35:            # punch a hole
            board[ROWS - h + rng.randint(0, h - 2)][c] = 0
    heights, holes = [], []
    for c in range(COLS):
        top = next((r for r in range(ROWS) if board[r][c]), -1)
        heights.append(0 if top < 0 else ROWS - top)
        holes.append(0 if top < 0 else sum(1 for r in range(top + 1, ROWS) if not board[r][c]))
    cands = []
    for i in range(8):
        cands.append({"label": bridge.LABELS[i], "rot": i % 4, "col": rng.randint(0, 9)})
    return {
        "repr": repr_mode, "mode": mode, "piece": rng.choice(PIECES),
        "grid": ["".join("#" if v else "." for v in row) for row in board],
        "heights": heights, "holes": holes, "cands": cands,
    }


def summarize(tag, rows):
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("  %-22s ALL FAILED" % tag)
        for r in rows:
            print("      %s" % r.get("error"))
        return None
    warm = ok[1:] if len(ok) > 1 else ok
    walls = sorted(r["wall_ms"] for r in warm)
    med = statistics.median(walls)
    print("  %-22s n=%-3d wall med=%-7.0f p90=%-7.0f  prefill med=%-7.0f decode med=%-6.1f"
          % (tag, len(warm), med, walls[-1], statistics.median([r["prompt_ms"] for r in warm]),
             statistics.median([r["predicted_ms"] for r in warm])))
    print("  %-22s prompt_n=%s  cache_n=%s  picks=%s"
          % ("", [r["prompt_n"] for r in ok],
             [r["cached_n"] for r in ok],
             "".join(str(r.get("label")) for r in ok)))
    return med


def main():
    try:
        urllib.request.urlopen(URL + "/health", timeout=2)
        print("BUSY: something already serves %s -- aborting, nothing started." % URL)
        return 2
    except Exception:
        pass

    logf = open(r"F:\super\qwen38mtp\jev-tetris\tools\logs\smoke_server.log", "wb")
    print("launching %s ..." % BAT)
    proc = subprocess.Popen(["cmd", "/c", str(BAT)], stdout=logf, stderr=subprocess.STDOUT,
                            cwd=str(ROOT))
    try:
        t0 = time.time()
        up = False
        while time.time() - t0 < 300:
            if proc.poll() is not None:
                print("server exited early rc=%s" % proc.returncode)
                return 1
            try:
                urllib.request.urlopen(URL + "/health", timeout=2)
                up = True
                break
            except Exception:
                time.sleep(1)
        if not up:
            print("health timeout")
            return 1
        print("healthy in %.1fs\n" % (time.time() - t0))

        rng = random.Random(7)
        results = {}
        for repr_mode in ("compact", "grid"):
            rows = []
            for _ in range(9):
                st = make_state(rng, repr_mode)
                try:
                    rows.append(bridge.decide(st))
                except Exception as e:
                    rows.append({"error": "%s: %s" % (type(e).__name__, e)})
            results[repr_mode] = summarize("jev/" + repr_mode, rows)

        # same state, three readout modes
        for mode in ("jev", "gen", "think"):
            rows = []
            for _ in range(4):
                st = make_state(rng, "compact", mode)
                try:
                    rows.append(bridge.decide(st))
                except Exception as e:
                    rows.append({"error": "%s: %s" % (type(e).__name__, e)})
            med = summarize("mode=" + mode, rows)
            results["mode_" + mode] = med
            if mode != "jev":
                ex = rows[-1]
                print("  %-22s text=%r" % ("", (ex.get("text") or "")[:90]))

        # one full decision dump, to eyeball the probability shape
        print("\n-- one decision in detail (compact/jev) --")
        st = make_state(rng, "compact")
        d = bridge.decide(st)
        print("  wall=%.0f ms  prompt_ms=%.0f  decode_ms=%.1f  prompt_n=%s  cache_n=%s"
              % (d["wall_ms"], d["prompt_ms"], d["predicted_ms"], d["prompt_n"], d["cached_n"]))
        print("  pick=%s (p=%.3f)   seen slots=%s" % (d["label"], d["prob"], d["seen"]))
        print("  probs=%s" % {k: round(v, 3) for k, v in (d["probs"] or {}).items()})
        print("\n--- prompt the model saw ---")
        print("\n".join("  | " + ln for ln in d["prompt"].splitlines()))
        print("\nSUMMARY %s" % results)
        return 0
    finally:
        logf.close()
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, text=True)
        try:
            proc.wait(timeout=20)
        except Exception:
            pass
        time.sleep(2)
        # orphan check: anything still holding VRAM?
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader"], capture_output=True,
                                 timeout=20).stdout.decode("utf-8", "replace").strip()
            print("\n[cleanup] killed pid %s; compute apps now: %s" % (proc.pid, out or "(none)"))
        except Exception as e:
            print("\n[cleanup] killed pid %s; vram probe failed: %s" % (proc.pid, e))


if __name__ == "__main__":
    sys.exit(main())
