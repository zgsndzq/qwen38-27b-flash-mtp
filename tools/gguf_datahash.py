"""sha256 of the tensor-data region of a gguf, starting at the aligned end of
the metadata. Two files with the same data hash differ only in their header."""

import hashlib
import struct
import sys


def data_start(p):
    with open(p, "rb") as f:
        assert f.read(4) == b"GGUF"
        struct.unpack("<I", f.read(4))
        struct.unpack("<Q", f.read(8))
        n_kv = struct.unpack("<Q", f.read(8))[0]
        align = 32
        for _ in range(n_kv):
            n = struct.unpack("<Q", f.read(8))[0]; f.read(n)
            t = struct.unpack("<I", f.read(4))[0]
            if t == 8:
                n = struct.unpack("<Q", f.read(8))[0]; f.read(n)
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                ln = struct.unpack("<Q", f.read(8))[0]
                for _ in range(ln):
                    if et == 8:
                        n = struct.unpack("<Q", f.read(8))[0]; f.read(n)
                    else:
                        f.read({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                                7: 1, 10: 8, 11: 8, 12: 8}[et])
            else:
                f.read({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                        10: 8, 11: 8, 12: 8}[t])
            if True:
                pass
        # tensor infos follow; walk them to find the end of the header
        # (name, n_dims, dims, type, offset)
        # re-open and skip using the count we already consumed is not possible,
        # so recompute: read n_tensors again from the start
    with open(p, "rb") as f:
        f.seek(8)
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        align = 32
        for _ in range(n_kv):
            n = struct.unpack("<Q", f.read(8))[0]; f.read(n)
            t = struct.unpack("<I", f.read(4))[0]
            if t == 8:
                n = struct.unpack("<Q", f.read(8))[0]; f.read(n)
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                ln = struct.unpack("<Q", f.read(8))[0]
                for _ in range(ln):
                    if et == 8:
                        n = struct.unpack("<Q", f.read(8))[0]; f.read(n)
                    else:
                        f.read({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                                7: 1, 10: 8, 11: 8, 12: 8}[et])
            else:
                f.read({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                        10: 8, 11: 8, 12: 8}[t])
        for _ in range(n_tensors):
            n = struct.unpack("<Q", f.read(8))[0]; f.read(n)
            nd = struct.unpack("<I", f.read(4))[0]
            f.read(8 * nd)
            f.read(4 + 8)
        end = f.tell()
    return end + (-end) % align


for p in sys.argv[1:]:
    off = data_start(p)
    h = hashlib.sha256()
    size = 0
    with open(p, "rb") as f:
        f.seek(off)
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
            size += len(b)
    print("%-46s data@%-9d %6.2f GB  %s" % (p.split("\\")[-1], off,
                                             size / 2**30, h.hexdigest()[:32]))