"""Print where the tensor-data block starts in a gguf, walking the header the
same way llama.cpp does, so two files can be compared directly."""

import struct
import sys

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B",
       10: "Q", 11: "q", 12: "d"}


def walk(p):
    f = open(p, "rb")
    assert f.read(4) == b"GGUF"
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    align = 32

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n)

    def rval(t):
        nonlocal align
        if t == 8:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n)
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            ln = struct.unpack("<Q", f.read(8))[0]
            out = []
            for _ in range(ln):
                out.append(rval(et))
            return out
        if t not in SZ:
            raise SystemExit("unknown gguf type %d at %d" % (t, f.tell()))
        v = struct.unpack("<" + FMT[t], f.read(SZ[t]))[0]
        return v

    for _ in range(n_kv):
        k = rstr().decode("utf-8", "replace")
        t = struct.unpack("<I", f.read(4))[0]
        v = rval(t)
        if k == "general.alignment":
            align = v
        if k == "general.name":
            name = v.decode("utf-8", "replace") if isinstance(v, bytes) else v

    max_end = 0
    for _ in range(n_tensors):
        rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        f.read(8 * nd)
        f.read(4)
        off = struct.unpack("<Q", f.read(8))[0]
        max_end = max(max_end, off)

    raw_end = f.tell()
    aligned = raw_end + (-raw_end) % align
    print("%-46s v%d tensors=%d kv=%d align=%d" % (p.split("\\")[-1], ver,
                                                   n_tensors, n_kv, align))
    print("    name=%s" % name)
    print("    header raw end=%d  aligned=%d  max tensor offset=%d"
          % (raw_end, aligned, max_end))
    return aligned


for p in sys.argv[1:]:
    walk(p)