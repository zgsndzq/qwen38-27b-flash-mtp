#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closed loop for the JEV x Tetris demo:

    probe ports -> start llmserver (18080) -> start bridge (18090)
    -> run tools/play.js -> kill both by PID -> report

Aborts if either port is already busy (that would be somebody else's server).
"""
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # ...\qwen27b\jev-tetris
PKG = ROOT.parent                        # ...\qwen27b  (llmserver.bat lives here)
NODE = r"C:\Users\bestd\.workbuddy\binaries\node\versions\22.22.2-3\node.exe"
NO_WINDOW = 0x08000000


def listening(port):
    with socket.socket() as s:
        s.settimeout(0.6)
        return s.connect_ex(("127.0.0.1", port)) == 0


def http_ok(port, path, timeout=3):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def wait_http(port, path, seconds, label):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if http_ok(port, path):
            print("  %s up after %.1f s" % (label, time.time() - t0), flush=True)
            return True
        time.sleep(1.0)
    return False


def kill(pid, name):
    subprocess.run("taskkill /F /T /PID %d" % pid, shell=True,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("  killed %s (pid %d)" % (name, pid))


def main():
    maxp = sys.argv[1] if len(sys.argv) > 1 else "150"
    for p in (18080, 18090):
        if listening(p):
            print("ABORT: port %d is already in use - not touching it." % p)
            return 1

    srv = brg = None
    try:
        print("starting llama-server ...", flush=True)
        srv = subprocess.Popen(["cmd", "/c", str(PKG / "llmserver.bat")], cwd=str(PKG),
                               creationflags=NO_WINDOW,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_http(18080, "/health", 240, "model"):
            print("FAIL: model never answered")
            return 2

        print("starting bridge ...", flush=True)
        brg = subprocess.Popen([sys.executable, str(ROOT / "bridge.py")], cwd=str(ROOT),
                               creationflags=NO_WINDOW,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_http(18090, "/api/health", 20, "bridge"):
            print("FAIL: bridge never answered")
            return 2

        print("playing %s pieces with the real model ...\n" % maxp, flush=True)
        r = subprocess.run([NODE, str(HERE / "play.js"), maxp], cwd=str(ROOT),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=900)
        print(r.stdout)
        if r.stderr.strip():
            print("--- stderr ---\n" + r.stderr[-1500:])
        return r.returncode
    finally:
        if brg:
            kill(brg.pid, "bridge")
        if srv:
            kill(srv.pid, "llama-server")
        time.sleep(2)
        left = subprocess.run("tasklist /FI \"IMAGENAME eq llama-server.exe\"",
                              shell=True, capture_output=True, text=True,
                              encoding="utf-8", errors="replace").stdout or ""
        print("  llama-server still running:", "llama-server.exe" in left)


if __name__ == "__main__":
    sys.exit(main())
