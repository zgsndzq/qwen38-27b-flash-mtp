#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Free a TCP port: find the process listening on it, stop it, wait until it is gone.

    python free_port.py 18080

Shared by run_server.py and the mllm launcher, so every llmserver* entry point
starts from a clean port instead of dying on bind. Only the listener is touched;
nothing else is killed by name here.
"""
import os
import signal
import subprocess
import sys
import time

WIN = os.name == "nt"


def port_pid(port):
    """PID listening on `port`, or None."""
    if WIN:
        try:
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                                 capture_output=True, text=True, errors="replace").stdout
        except OSError:
            return None
        for line in out.splitlines():
            f = line.split()
            if len(f) < 5 or f[0].upper() != "TCP" or f[3].upper() != "LISTENING":
                continue
            if f[1].rsplit(":", 1)[-1] == str(port):
                try:
                    return int(f[4])
                except ValueError:
                    pass
        return None
    try:
        out = subprocess.run(["lsof", "-nP", "-iTCP:%d" % port, "-sTCP:LISTEN", "-t"],
                             capture_output=True, text=True).stdout.split()
        return int(out[0]) if out else None
    except (OSError, ValueError, IndexError):
        return None


def kill_pid(pid):
    if WIN:
        return subprocess.call(["taskkill", "/F", "/PID", str(pid)],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL) == 0
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except OSError:
        return False


def free_port(port, wait_s=15.0):
    """Stop the listener on `port`. True when the port is free afterwards."""
    pid = port_pid(port)
    if pid is None:
        return True
    print("[port] %d is held by pid %d, stopping it" % (port, pid), flush=True)
    kill_pid(pid)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if port_pid(port) is None:
            print("[port] %d is free" % port, flush=True)
            return True
        time.sleep(0.25)
    print("[port] %d is still held after %.0fs" % (port, wait_s), file=sys.stderr)
    return False


if __name__ == "__main__":
    try:
        p = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    except ValueError:
        sys.exit("usage: free_port.py PORT")
    sys.exit(0 if free_port(p) else 1)
