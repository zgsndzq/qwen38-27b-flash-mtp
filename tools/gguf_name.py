import struct
import sys

p = sys.argv[1] if len(sys.argv) > 1 else \
    r"F:\super\qwen27b\weights\Qwen3.8-27B-Flash-MTP.gguf"

# GGUF: magic(4) version(u32) tensor_count(u64) kv_count(u64) then kv pairs
# value type -> (reader, size)
T = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
    4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("B", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}


def r_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def r_val(f, t):
    if t == 8:
        return r_str(f)
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        # every element must be consumed or the stream desyncs
        keep = []
        for i in range(n):
            v = r_val(f, et)
            if i < 4:
                keep.append(v)
        return keep
    fmt, sz = T[t]
    return struct.unpack("<" + fmt, f.read(sz))[0]


with open(p, "rb") as f:
    assert f.read(4) == b"GGUF", "not a gguf"
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    print("gguf v%d  tensors=%d  kv=%d" % (ver, n_tensors, n_kv))
    for _ in range(n_kv):
        key = r_str(f)
        t = struct.unpack("<I", f.read(4))[0]
        val = r_val(f, t)
        if key.startswith("general.") or "name" in key.lower():
            print("  %-34s = %s" % (key, val))