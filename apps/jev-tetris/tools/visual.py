# -*- coding: utf-8 -*-
"""
visual.py -- one-shot end-to-end visual check for the JEV x Tetris page.

Brings up both halves in a single process (model server -> bridge), drives the
page through the visual-check wrapper (single-batch open/wait/screenshot, with
the blank-screenshot probe), and tears everything down by PID.

    python tools/visual.py                 # both shots
    python tools/visual.py jev             # only the JEV shot
"""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SERVER_BAT = Path(r"F:\super\qwen27b\llmserver.bat")
PS1 = Path(r"C:\Users\bestd\.workbuddy\skills\visual-check\scripts\visual-shot.ps1")
MODEL = "http://127.0.0.1:18080"
UI = "http://127.0.0.1:18090"

SHOTS = {
    "jev":  ("?auto=1&mode=jev&repr=compact&gap=0", 22000, "shot-jev.png"),
    "gen":  ("?auto=1&mode=gen&repr=grid&gap=0",    16000, "shot-gen-grid.png"),
}


def probe(url, timeout=2):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def wait_for(url, label, limit):
    t0 = time.time()
    while time.time() - t0 < limit:
        if probe(url):
            print("  %s up after %.1fs" % (label, time.time() - t0))
            return True
        time.sleep(1)
    print("  %s NEVER came up (%.0fs)" % (label, limit))
    return False


def kill(proc, name):
    if not proc:
        return
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                   capture_output=True, text=True)
    try:
        proc.wait(timeout=20)
    except Exception:
        pass
    print("  killed %s (pid %s)" % (name, proc.pid))


def main():
    want = sys.argv[1:] or list(SHOTS)
    for k in want:
        if k not in SHOTS:
            print("unknown shot %r; pick from %s" % (k, list(SHOTS)))
            return 2

    for url, who in ((MODEL + "/health", "model 18080"), (UI + "/api/health", "bridge 18090")):
        if probe(url):
            print("BUSY: %s already answers -- aborting, nothing started." % who)
            return 2

    model = bridge = None
    log = open(HERE / "tools" / "logs" / "visual_server.log", "wb")
    try:
        print("starting model server ...")
        model = subprocess.Popen(["cmd", "/c", str(SERVER_BAT)], stdout=log,
                                 stderr=subprocess.STDOUT, cwd=str(SERVER_BAT.parent))
        if not wait_for(MODEL + "/health", "model", 300):
            return 1

        print("starting bridge ...")
        bridge = subprocess.Popen([sys.executable, str(HERE / "bridge.py")],
                                  stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE))
        if not wait_for(UI + "/api/health", "bridge", 30):
            return 1

        rc = 0
        for key in want:
            qs, wait_ms, out = SHOTS[key]
            out_path = (HERE / out).as_posix()
            print("\n=== shot: %s  (%s, wait %ds) ===" % (key, qs, wait_ms // 1000))
            r = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PS1),
                 "-Url", UI + "/" + qs, "-Out", out_path, "-WaitMs", str(wait_ms),
                 "-W", "1600", "-H", "1020"],
                capture_output=True, text=True, errors="replace", timeout=240)
            for ln in (r.stdout or "").splitlines():
                if ln.strip():
                    print("   " + ln.strip())
            for ln in (r.stderr or "").splitlines():
                if ln.strip():
                    print("   ! " + ln.strip())
            print("   exit=%s" % r.returncode)
            rc = rc or r.returncode
        return rc
    finally:
        kill(bridge, "bridge")
        kill(model, "model server")
        log.close()
        time.sleep(2)
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader"], capture_output=True,
                                 timeout=20).stdout.decode("utf-8", "replace").strip()
            print("[cleanup] gpu compute apps now: %s" % (out or "(none)"))
        except Exception as e:
            print("[cleanup] vram probe failed: %s" % e)


if __name__ == "__main__":
    sys.exit(main())
