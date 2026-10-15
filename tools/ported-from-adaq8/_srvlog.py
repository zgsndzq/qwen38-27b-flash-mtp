# -*- coding: utf-8 -*-
"""_srvlog.py -- server stdout goes to RAM, not to disk.

User rule (2026-09-28): while testing, do not dribble the server log onto the SSD
line by line. Keep a bounded tail in memory, materialise a file only when somebody
explicitly asks for it (or when a test fails and you want the evidence).

Also measures how many bytes the server process actually read from / wrote to the
disk during its life -- that is the number that tells you whether a test run was
IO-bound or not.
"""
import ctypes
import os
import socket
import subprocess
import threading
import time
from ctypes import wintypes

ROOT = r"F:\super\qwen27b"
DEFAULT_KEEP = int(os.environ.get("SRVLOG_KEEP_BYTES", str(256 * 1024)))

MB = 1024.0 * 1024.0
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)


def port_busy(port=18080):
    s = socket.socket()
    s.settimeout(0.4)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        s.close()


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong)]


def proc_io(pid):
    """(read_bytes, write_bytes) cumulative for pid, or None."""
    h = _k32.OpenProcess(0x1000, False, pid)
    if not h:
        return None
    io = _IO_COUNTERS()
    ok = _k32.GetProcessIoCounters(h, ctypes.byref(io))
    _k32.CloseHandle(h)
    if not ok:
        return None
    return (io.ReadTransferCount, io.WriteTransferCount)


def find_server_pid(timeout=1.0, port=18080):
    """PID listening on the server port, or None.

    netstat is more reliable than `tasklist /FI IMAGENAME` here: the image name is
    not always what you expect, but the listening socket never lies."""
    deadline = time.time() + timeout
    while True:
        try:
            out = subprocess.check_output(["netstat", "-ano"],
                                          encoding="utf-8", errors="replace", timeout=20)
            for line in out.splitlines():
                if "LISTENING" in line and (":%d" % port) in line:
                    parts = line.split()
                    if parts and parts[-1].isdigit():
                        return int(parts[-1])
        except Exception:
            pass
        if time.time() >= deadline:
            return None
        time.sleep(0.2)


class MemTail:
    """Bounded in-RAM tail of a byte stream. Never touches the disk."""

    def __init__(self, keep=DEFAULT_KEEP):
        self.keep = keep
        self._buf = bytearray()
        self._lock = threading.Lock()
        self.total = 0

    def feed(self, chunk):
        with self._lock:
            self.total += len(chunk)
            self._buf += chunk
            if len(self._buf) > self.keep:
                del self._buf[:len(self._buf) - self.keep]

    def text(self):
        with self._lock:
            return bytes(self._buf).decode("utf-8", "replace")

    def tail(self, n=30):
        lines = self.text().splitlines()
        return "\n".join(lines[-n:])

    def dump(self, path):
        """Only call this when you actually want a file on disk."""
        with self._lock:
            with open(path, "wb") as f:
                f.write(bytes(self._buf))
        return path


class Server:
    """Start llmserver.bat with its stdout captured in RAM.

    Usage:
        with Server() as srv:
            srv.wait_health()
            ...
            print(srv.tail())
    """

    def __init__(self, bat=None, cwd=ROOT, keep=DEFAULT_KEEP, base="http://127.0.0.1:18080"):
        self.bat = bat or os.path.join(cwd, "llmserver.bat")
        self.cwd = cwd
        self.base = base
        self.tail_buf = MemTail(keep)
        self.proc = None
        self._reader = None
        self.io_after_load = None

    def start(self):
        if port_busy():
            raise RuntimeError(
                "port 18080 is already in use -- refusing to start a second server "
                "(only one llama-server may own the GPU). Kill the existing one first.")
        # bufsize=0 is essential: with a buffered reader, read(4096) blocks until
        # 4096 bytes accumulate (or EOF), so a live server appears to log nothing.
        # A raw FileIO returns each chunk as soon as it lands.
        self.proc = subprocess.Popen(["cmd", "/c", self.bat], cwd=self.cwd,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     bufsize=0)
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        return self

    def _pump(self):
        # read raw chunks; llama-server mixes \n and \r, block reads are safest
        try:
            while True:
                chunk = self.proc.stdout.read(4096)
                if not chunk:
                    break
                self.tail_buf.feed(chunk)
        except Exception:
            pass

    def wait_health(self, timeout=90.0, poll=0.4):
        import urllib.request
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if r.status == 200:
                        # capture what loading the model cost in disk IO
                        time.sleep(1.0)  # let the last mmap reads settle
                        pid = find_server_pid()
                        self.io_after_load = proc_io(pid) if pid else None
                        return True
            except Exception:
                pass
            time.sleep(poll)
        return False

    def load_io(self):
        """(read_MB, write_MB) the server spent getting to 'healthy'."""
        if not self.io_after_load:
            return None
        return (self.io_after_load[0] / MB, self.io_after_load[1] / MB)

    def tail(self, n=30):
        return self.tail_buf.tail(n)

    def disk_written_mb(self):
        pid = find_server_pid()
        c = proc_io(pid) if pid else None
        return c[1] / MB if c else None

    def kill(self):
        if self.proc and self.proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True)
        subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)
        if self.proc and self.proc.stdout:
            try:
                self.proc.stdout.close()
            except Exception:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.kill()
