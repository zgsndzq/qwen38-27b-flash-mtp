# -*- coding: utf-8 -*-
"""http_probe.py -- POST the exact payload the page sends, print the raw reply."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SERVER_BAT = Path(r"F:\super\qwen27b\llmserver.bat")
MODEL, UI = "http://127.0.0.1:18080", "http://127.0.0.1:18090"


def probe(url, t=2):
    try:
        urllib.request.urlopen(url, timeout=t)
        return True
    except Exception:
        return False


def main():
    for u in (MODEL + "/health", UI + "/api/health"):
        if probe(u):
            print("BUSY %s" % u)
            return 2
    log = open(HERE / "tools" / "logs" / "probe_server.log", "wb")
    m = b = None
    try:
        m = subprocess.Popen(["cmd", "/c", str(SERVER_BAT)], stdout=log,
                             stderr=subprocess.STDOUT, cwd=str(SERVER_BAT.parent))
        for _ in range(300):
            if probe(MODEL + "/health"):
                break
            time.sleep(1)
        print("model up")
        b = subprocess.Popen([sys.executable, str(HERE / "bridge.py")],
                             stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE))
        for _ in range(30):
            if probe(UI + "/api/health"):
                break
            time.sleep(1)
        print("bridge up\n")

        payload = {
            "repr": "compact", "mode": "jev", "piece": "T",
            "grid": [".........."] * 20,
            "heights": [8, 11, 6, 3, 9, 12, 7, 4, 10, 7],
            "holes": [1, 2, 1, 2, 5, 7, 2, 19, 5, 4],
            "cands": [{"label": L, "rot": i % 4, "col": i} for i, L in enumerate("ABCDEFGH")],
        }
        req = urllib.request.Request(UI + "/api/decide",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode("utf-8")
        print("--- raw reply (%d bytes) ---" % len(raw))
        print(raw[:1500])
        d = json.loads(raw)
        print("\nkeys        : %s" % sorted(d.keys()))
        print("label       : %r" % d.get("label"))
        print("prob        : %r" % d.get("prob"))
        print("probs type  : %s" % type(d.get("probs")).__name__)
        print("probs       : %s" % d.get("probs"))
        print("seen        : %s" % d.get("seen"))
        return 0
    finally:
        for p, n in ((b, "bridge"), (m, "model")):
            if p:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               capture_output=True, text=True)
        log.close()
        time.sleep(2)
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader"], capture_output=True,
                             timeout=20).stdout.decode("utf-8", "replace").strip()
        print("[cleanup] gpu compute apps: %s" % (out or "(none)"))


if __name__ == "__main__":
    sys.exit(main())
