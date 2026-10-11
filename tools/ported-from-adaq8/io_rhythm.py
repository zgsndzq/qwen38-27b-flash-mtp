# -*- coding: utf-8 -*-
"""
io_rhythm.py -- is the SSD moving in step with token output?

No admin needed. Four layers, sampled every 50 ms:

  A) device-level disk, via PDH     \\PhysicalDisk(_Total)\\% Disk Time + bytes
                                    \\PhysicalDisk(N ...)\\...  per drive
     PDH is the performance-counter API; it sees IO from EVERY source including
     the kernel, and "% Disk Time" reports the device being *busy* -- which
     catches activity that a byte delta would hide.

  B) memory pressure, via PDH       \\Memory\\Page Writes/sec, Pages Output/sec,
                                    Available MBytes, Committed Bytes
     THE key metric for "is the OS turning RAM pressure into disk writes".

  C) target process                 GetProcessIoCounters (read / write /
                                    OTHER = page-file traffic, i.e. disk IO that
                                    no file monitor ever reports) + hard faults.

  D) whole-system attribution       every process's deltas, scanned at 500 ms,
                                    so whoever writes gets named.

GPU power (NVML, every sample) is the token-rhythm proxy: decoding pulses the
card to ~55-60 W, idle sits at ~2.6 W / 210 MHz. Lining disk activity up against
that pulse is the entire question.

usage:  python io_rhythm.py [max_seconds] [active_seconds_needed] [pid]
        python io_rhythm.py 180 20
"""
import ctypes
import ctypes.wintypes as wt
import io
import os
import subprocess
import sys
import time

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)

MB = 1024.0 * 1024.0
OUT = r"F:\super\tmp\io_rhythm.txt"
SAMPLE = 0.05          # 50 ms
SLOW_EVERY = 10        # full process scan every 10 samples = 500 ms
ACTIVE_MW = 15000      # GPU power above this == generation in progress


# ---------------------------------------------------------------- structs ----
class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class PROCESSENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)), ("th32ModuleID", wt.DWORD),
        ("cntThreads", wt.DWORD), ("th32ParentProcessID", wt.DWORD),
        ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
        ("szExeFile", ctypes.c_char * 260),
    ]


class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("longValue", ctypes.c_long), ("doubleValue", ctypes.c_double),
                    ("largeValue", ctypes.c_longlong)]
    _anonymous_ = ("u",)
    _fields_ = [("CStatus", wt.DWORD), ("u", _U)]


# ------------------------------------------------------------- primitives ----
TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE = ctypes.c_void_p(-1).value
PQLI = 0x1000


def list_pids():
    out = {}
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == INVALID_HANDLE:
        return out
    try:
        e = PROCESSENTRY32()
        e.dwSize = ctypes.sizeof(PROCESSENTRY32)
        ok = k32.Process32First(snap, ctypes.byref(e))
        while ok:
            out[int(e.th32ProcessID)] = e.szExeFile.decode("ascii", "replace")
            ok = k32.Process32Next(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    return out


def proc_io(pid):
    h = k32.OpenProcess(PQLI, False, pid)
    if not h:
        return None
    try:
        c = IO_COUNTERS()
        if not k32.GetProcessIoCounters(h, ctypes.byref(c)):
            return None
        return (c.ReadTransferCount, c.WriteTransferCount, c.OtherTransferCount,
                c.ReadOperationCount, c.WriteOperationCount, c.OtherOperationCount)
    finally:
        k32.CloseHandle(h)


def proc_faults(pid):
    h = k32.OpenProcess(PQLI, False, pid)
    if not h:
        return None
    try:
        m = PROCESS_MEMORY_COUNTERS()
        m.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if not psapi.GetProcessMemoryInfo(h, ctypes.byref(m), m.cb):
            return None
        return (m.PageFaultCount, m.WorkingSetSize, m.PagefileUsage)
    finally:
        k32.CloseHandle(h)


# ------------------------------------------------------------------- pdh -----
class Pdh(object):
    """Thin PDH wrapper. add() returns an index usable with collect()."""

    PDH_FMT_DOUBLE = 0x00000200
    PDH_FMT_NOSCALE = 0x00001000

    def __init__(self):
        self.ok = False
        self.paths = []
        try:
            self.p = ctypes.WinDLL("pdh.dll")
        except OSError:
            return
        self.p.PdhOpenQueryW.argtypes = [wt.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.p.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wt.LPCWSTR,
                                                 ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.p.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self.p.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p, wt.DWORD,
                                                       ctypes.POINTER(wt.DWORD),
                                                       ctypes.POINTER(PDH_FMT_COUNTERVALUE)]
        self.p.PdhExpandWildCardPathW.argtypes = [wt.LPCWSTR, wt.LPCWSTR, wt.LPCWSTR,
                                                  ctypes.POINTER(wt.DWORD), wt.DWORD]
        self.q = ctypes.c_void_p()
        if self.p.PdhOpenQueryW(None, 0, ctypes.byref(self.q)) != 0:
            return
        self.cs = []
        self.ok = True

    def expand(self, wildcard):
        """\\PhysicalDisk(*)\\Disk Write Bytes/sec  ->  the real instance paths."""
        if not self.ok:
            return []
        need = wt.DWORD(0)
        self.p.PdhExpandWildCardPathW(None, wildcard, None, ctypes.byref(need), 0)
        if need.value == 0:
            return []
        buf = ctypes.create_unicode_buffer(need.value + 2)
        if self.p.PdhExpandWildCardPathW(None, wildcard, buf, ctypes.byref(need), 0) != 0:
            return []
        return [s for s in buf[:need.value].split("\0") if s]

    def add(self, path):
        if not self.ok:
            return -1
        c = ctypes.c_void_p()
        if self.p.PdhAddEnglishCounterW(self.q, path, 0, ctypes.byref(c)) != 0:
            self.paths.append((path, None))
            self.cs.append(None)
            return len(self.cs) - 1
        self.paths.append((path, True))
        self.cs.append(c)
        return len(self.cs) - 1

    def collect(self):
        if not self.ok:
            return []
        if self.p.PdhCollectQueryData(self.q) != 0:
            return [None] * len(self.cs)
        out = []
        for c in self.cs:
            if c is None:
                out.append(None)
                continue
            v = PDH_FMT_COUNTERVALUE()
            r = self.p.PdhGetFormattedCounterValue(c, self.PDH_FMT_DOUBLE, None,
                                                   ctypes.byref(v))
            out.append(v.doubleValue if r == 0 else None)
        return out

    def instances(self, obj):
        """kept only as a fallback -- the wildcard expander is what we use."""
        if not self.ok:
            return []
        buffer = ctypes.create_unicode_buffer(4096)
        need = wt.DWORD(4096)
        if self.p.PdhEnumObjectItemsW(None, None, obj, None, None,
                                      buffer, ctypes.byref(need), 0, 0) != 0:
            return []
        return [s for s in buffer[:need.value].split("\0") if s]


# -------------------------------------------------------------------- gpu ----
class NVML_UTIL(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class Gpu(object):
    def __init__(self):
        self.ok = False
        self.h = ctypes.c_void_p()
        self.n = None
        for dll in ("nvml.dll", r"C:\Windows\System32\nvml.dll"):
            try:
                self.n = ctypes.WinDLL(dll)
                break
            except OSError:
                self.n = None
        if self.n is None:
            return
        try:
            if self.n.nvmlInit_v2() != 0:
                return
            if self.n.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h)) != 0:
                return
            self.ok = True
        except Exception:
            self.ok = False

    def read(self):
        if not self.ok:
            return None, None
        u = NVML_UTIL()
        pu = ctypes.c_uint(0)
        self.n.nvmlDeviceGetUtilizationRates(self.h, ctypes.byref(u))
        self.n.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(pu))
        return u.gpu, pu.value


def find_listener(port):
    try:
        out = subprocess.check_output(["netstat", "-ano"], encoding="utf-8",
                                      errors="replace", timeout=10)
    except Exception:
        return None
    for line in out.splitlines():
        if "LISTENING" in line and (":%d" % port) in line:
            try:
                return int(line.split()[-1])
            except ValueError:
                pass
    return None


# ------------------------------------------------------------------- main ----
def main():
    max_s = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
    need_s = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    tgt = int(sys.argv[3]) if len(sys.argv) > 3 else find_listener(18080)

    lines = []

    def say(s=""):
        lines.append(s)
        print(s, flush=True)

    say("=" * 78)
    say("io_rhythm -- is the disk moving in step with token output?")
    say("  window   : up to %.0f s, stops early after %.0f s of GPU-active time"
        % (max_s, need_s))
    say("  sampling : %.0f ms      target pid: %s" % (SAMPLE * 1000, tgt))
    say("=" * 78)

    gpu = Gpu()
    say("nvml : %s" % ("ok  (power + util per sample)" if gpu.ok else "UNAVAILABLE"))

    pdh = Pdh()
    say("pdh  : %s" % ("ok" if pdh.ok else "UNAVAILABLE"))
    idx = {}
    if pdh.ok:
        insts = []
        for suffix in (r"\Disk Write Bytes/sec", r"\Disk Read Bytes/sec",
                       r"\% Disk Time", r"\Avg. Disk sec/Write"):
            for path in pdh.expand(r"\PhysicalDisk(*)" + suffix):
                m = path.split("(", 1)
                nm = m[1].split(")", 1)[0] if len(m) > 1 else "?"
                key = suffix.strip("\\").replace(" ", "_").replace("%", "pct").replace("/", "_")
                idx["disk_%s[%s]" % (key, nm)] = pdh.add(path)
                if nm not in insts:
                    insts.append(nm)
        for suffix in (r"\Disk Write Bytes/sec", r"\Disk Read Bytes/sec", r"\% Disk Time"):
            key = "total_" + suffix.strip("\\").replace(" ", "_").replace("%", "pct")
            idx[key] = pdh.add(r"\PhysicalDisk(_Total)" + suffix)
        say("       PhysicalDisk instances via wildcard: %s" % insts)
        idx["mem_pagew"] = pdh.add(r"\Memory\Page Writes/sec")
        idx["mem_pageout"] = pdh.add(r"\Memory\Pages Output/sec")
        idx["mem_avail"] = pdh.add(r"\Memory\Available MBytes")
        idx["mem_commit"] = pdh.add(r"\Memory\Committed Bytes")
        say("       counters added: %d" % len(idx))
    say()

    pids = list_pids()
    prev_all = {}
    for pid in pids:
        c = proc_io(pid)
        if c:
            prev_all[pid] = c

    # warm up PDH rate counters (first collect always reads 0)
    if pdh.ok:
        pdh.collect()
        time.sleep(0.2)

    prev_faults = 0
    prev_pf = 0
    if tgt:
        f0 = proc_faults(tgt)
        if f0:
            prev_faults, prev_pf = f0[0], f0[2]

    t0 = time.time()
    active = 0.0
    samples = []
    slow = []

    n = int(max_s / SAMPLE)
    for k in range(n):
        time.sleep(SAMPLE)
        t = time.time() - t0

        util, mw = gpu.read()
        is_active = (mw is not None and mw > ACTIVE_MW)
        if is_active:
            active += SAMPLE

        pv = pdh.collect() if pdh.ok else []
        pvv = {}
        for name, i in idx.items():
            pvv[name] = pv[i] if i is not None and i < len(pv) else None

        tr = tw = to = tf = 0
        if tgt:
            c = proc_io(tgt)
            if c and tgt in prev_all:
                pc = prev_all[tgt]
                tr, tw, to = c[0] - pc[0], c[1] - pc[1], c[2] - pc[2]
            if c:
                prev_all[tgt] = c
            f = proc_faults(tgt)
            if f:
                # DELTA, not the running total -- a page fault is the only
                # signature of a memory-mapped file being read back off disk
                tf = f[0] - prev_faults
                prev_faults = f[0]
                tpg = f[2] - prev_pf
                prev_pf = f[2]

        samples.append((t, util, mw, tr, tw, to, tf, pvv))

        if k % SLOW_EVERY == 0:
            cur_all = {}
            moved = []
            names = list_pids()
            for pid in names:
                c = proc_io(pid)
                if not c:
                    continue
                cur_all[pid] = c
                p = prev_all.get(pid)
                if not p:
                    continue
                dr, dw, do = c[0] - p[0], c[1] - p[1], c[2] - p[2]
                if dr or dw or do:
                    moved.append((pid, names.get(pid, "?"), dr, dw, do))
            if moved:
                slow.append((t, moved))
            prev_all = cur_all

        if active >= need_s:
            say(">> got %.1f s of GPU-active time after %.1f s -- stopping early"
                % (active, t))
            break

        if k and k % 200 == 0:
            say("  ... t=%5.1f s  active=%4.1f s  gpu=%s%% %s mW"
                % (t, active, util, mw))

    dur = time.time() - t0

    say()
    say("=" * 78)
    n_act = sum(1 for s in samples if s[2] and s[2] > ACTIVE_MW)
    n_idle = len(samples) - n_act
    say("RESULT over %.1f s (%.0f samples): %d active / %d idle"
        % (dur, len(samples), n_act, n_idle))
    say("=" * 78)

    act = [s for s in samples if s[2] and s[2] > ACTIVE_MW]
    idl = [s for s in samples if not (s[2] and s[2] > ACTIVE_MW)]
    disk_write_keys = [k for k in idx if "Write_Bytes" in k]

    # ---- memory pressure
    if pdh.ok:
        say()
        say("--- memory pressure (the only way RAM becomes disk writes) ---")
        for name in ("mem_pagew", "mem_pageout", "mem_avail", "mem_commit"):
            vals = [s[7].get(name) for s in samples if s[7].get(name) is not None]
            if not vals:
                say("  %-14s <no data>" % name)
                continue
            say("  %-14s max=%12.3f  mean=%12.3f  nonzero samples=%d/%d"
                % (name, max(vals), sum(vals) / len(vals),
                   sum(1 for v in vals if v > 0.001), len(vals)))

    # ---- device
    if pdh.ok:
        say()
        say("--- disk device, GPU-ACTIVE vs GPU-IDLE (the whole question) ---")
        keys = sorted(k for k in idx if k.startswith("disk_"))
        say("  %-34s %14s %14s %14s" % ("counter", "active max", "active mean", "idle mean"))
        for key in keys:
            av = [(s[7].get(key) or 0.0) for s in samples if s[2] and s[2] > ACTIVE_MW]
            iv = [(s[7].get(key) or 0.0) for s in samples if not (s[2] and s[2] > ACTIVE_MW)]
            if not av:
                continue
            am, aM = sum(av) / len(av), max(av)
            im = (sum(iv) / len(iv)) if iv else 0.0
            flag = ""
            if "Write_Bytes" in key and aM > 65536:
                flag = "   <-- WRITING"
            if "pct_Disk_Time" in key and aM > 5:
                flag = "   <-- DEVICE BUSY"
            say("  %-34s %14.0f %14.0f %14.0f%s" % (key, aM, am, im, flag))

    # ---- target process
    say()
    say("--- target pid %s ---" % tgt)
    tr = sum(s[3] for s in samples)
    tw = sum(s[4] for s in samples)
    to = sum(s[5] for s in samples)
    tf = sum(s[6] for s in samples)
    say("  read=%.4f MB   write=%.4f MB   page-file(other)=%.4f MB"
        % (tr / MB, tw / MB, to / MB))
    say("  page faults (mmap read-back signature) = %d  in %.1f s = %.1f/s"
        % (tf, dur, tf / dur if dur else 0))
    mov = [s for s in samples if s[3] or s[4] or s[5]]
    say("  samples where it moved ANY byte of IO: %d / %d" % (len(mov), len(samples)))
    if act:
        a = sum((s[3] + s[4] + s[5]) for s in act) / len(act)
        say("  during GPU-ACTIVE samples it moved %.1f B/sample" % a)

    # ---- rhythm test
    say()
    say("--- RHYTHM TEST: does disk write track GPU power (== token output)? ---")
    import math
    wkey = None
    for k in idx:
        if k.startswith("total_") and "Write_Bytes" in k:
            wkey = k
            break
    if pdh.ok and wkey:
        wt_ser = [(s[7].get(wkey) or 0.0) for s in samples]
        pw = [((s[2] or 0.0) / 1000.0) for s in samples]
        n_ = len(wt_ser)

        def pearson(a, b):
            m = min(len(a), len(b))
            ma, mb = sum(a[:m]) / m, sum(b[:m]) / m
            num = sum((a[i] - ma) * (b[i] - mb) for i in range(m))
            da = math.sqrt(sum((x - ma) ** 2 for x in a[:m]))
            db = math.sqrt(sum((x - mb) ** 2 for x in b[:m]))
            return (num / (da * db)) if da > 0 and db > 0 else float("nan")

        say("  pearson(watt, disk write B/s) = %+.3f   (1.0 == locked to tokens)"
            % pearson(pw, wt_ser))
        for lag in (-4, -2, 2, 4):
            if lag < 0:
                a, b = pw[-lag:], wt_ser[:lag]
            else:
                a, b = pw[:-lag], wt_ser[lag:]
            say("     at %+4d ms shift            = %+.3f" % (lag * SAMPLE * 1000,
                                                              pearson(a, b)))
        say()
        say("  largest disk-write samples in the window:")
        say("  %7s %8s %16s %8s %10s" % ("t(s)", "watt", "disk_w B/s", "gpu%", "target B"))
        top = sorted(range(n_), key=lambda i: -wt_ser[i])[:12]
        for i in sorted(top):
            s = samples[i]
            say("  %7.2f %8.1f %16.0f %8s %10d"
                % (s[0], (s[2] or 0) / 1000.0, wt_ser[i], s[1], s[3] + s[4] + s[5]))

    # ---- per-sample timeline, only around activity
    say()
    say("--- timeline: the 60 samples around the first GPU-active one ---")
    first = next((i for i, s in enumerate(samples) if s[2] and s[2] > ACTIVE_MW), None)
    if first is None:
        say("  <no GPU-active sample was captured -- generation never happened>")
    else:
        lo = max(0, first - 10)
        hi = min(len(samples), first + 50)
        say("  %7s %7s %7s %9s %11s %11s %12s %12s" %
            ("t(s)", "gpu%", "watt", "tgt_r(B)", "tgt_w(B)", "tgt_pg(B)", "disk_w(B/s)", "pgw/s"))
        for s in samples[lo:hi]:
            u, mw = s[1], s[2]
            dw = None
            for key in idx:
                if "Write_Bytes" in key and "total" not in key:
                    dw = s[7].get(key)
                    break
            ps = s[7].get("mem_pagew")
            say("  %7.2f %7s %7.1f %9d %11d %11d %12s %12s"
                % (s[0], u, (mw or 0) / 1000.0, s[3], s[4], s[5],
                   ("%.0f" % dw) if dw is not None else "-",
                   ("%.0f" % ps) if ps is not None else "-"))

    # ---- attribution
    say()
    say("--- who moved the system, over the whole window ---")
    if not slow:
        say("  <nobody: not one process moved a single byte of IO>")
    else:
        agg = {}
        for t, moved in slow:
            for pid, nm, dr, dw, do in moved:
                a = agg.setdefault((pid, nm), [0, 0, 0, 0])
                a[0] += dr; a[1] += dw; a[2] += do; a[3] += 1
        for (pid, nm), a in sorted(agg.items(), key=lambda x: -(x[1][0] + x[1][1] + x[1][2]))[:12]:
            say("  %-26s pid=%-6d read=%9.3f MB write=%9.4f MB pgfile=%9.4f MB (%d scans)"
                % (nm, pid, a[0] / MB, a[1] / MB, a[2] / MB, a[3]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[written to %s]" % OUT)


if __name__ == "__main__":
    main()
