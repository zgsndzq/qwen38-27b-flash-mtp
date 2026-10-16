"""Rewrite one GGUF metadata value without touching a single tensor byte.

Changing general.name changes the size of the KV block, which shifts where the
tensor data starts. Tensor offsets are relative to that start, so they all shift
by the same delta and the data itself can be streamed across verbatim. What
comes out is bit-identical to the input apart from the metadata we changed.

    python gguf_rename.py <in.gguf> <out.gguf> <new general.name>
"""

import os
import struct
import sys

# GGUF value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
FIXED = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
         F32: "<f", U64: "<Q", I64: "<q", F64: "<d"}


def r_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n)


def r_raw(f, t):
    """Read a value and return its bytes exactly as they sit in the file."""
    if t == STRING:
        n = struct.unpack("<Q", f.read(8))[0]
        return struct.pack("<Q", n) + f.read(n)
    if t == ARRAY:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        parts = []
        for _ in range(n):
            parts.append(r_raw(f, et))
        return struct.pack("<I", et) + struct.pack("<Q", n) + b"".join(parts)
    if t == BOOL:
        return f.read(1)
    return f.read(struct.calcsize(FIXED[t]))


def enc_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def enc_raw(t, raw):
    return struct.pack("<I", t) + raw


def main():
    src, dst, new_name = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(src, "rb") as f:
        assert f.read(4) == b"GGUF", "not a GGUF"
        ver = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        kvs = []
        alignment = 32
        old_name = None
        for _ in range(n_kv):
            key = r_str(f).decode("utf-8", "replace")
            t = struct.unpack("<I", f.read(4))[0]
            raw = r_raw(f, t)
            if key == "general.name":
                old_name = raw[8:].decode("utf-8", "replace")   # drop the u64 length
                raw = enc_str(new_name)          # length prefix included, same shape as read_raw
                t = STRING
            if key == "general.alignment":
                alignment = struct.unpack("<Q", raw)[0]
            kvs.append((key, t, raw))

        tensors = []
        for _ in range(n_tensors):
            name = r_str(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            tensors.append((name, n_dims, dims, ttype, off))

        old_data_start = f.tell()
        old_pad = (-old_data_start) % alignment
        old_data_start += old_pad

    print("general.name : %r -> %r" % (old_name, new_name))
    print("tensors      : %d   alignment: %d" % (n_tensors, alignment))
    print("old data at  : %d" % old_data_start)

    # Rebuild. llama.cpp seeks to GGML_PAD(tell, alignment) after the tensor
    # infos (ggml/src/gguf.cpp:765) - it aligns UP to the nearest boundary and
    # knows nothing about extra padding. So the header has to end exactly where
    # the old one did, or the tensor offsets would all shift by a value we
    # cannot apply (the first tensor sits at offset 0, so a negative shift is
    # not representable).
    #
    # Shortening a value shortens the header, which would pull the aligned data
    # start earlier. The fix is a filler KV of exactly the missing byte count,
    # so the aligned start lands back on the old offset and every tensor offset
    # stays valid untouched.

    def build(kvs_extra):
        h = bytearray()
        h += b"GGUF" + struct.pack("<I", ver)
        h += struct.pack("<Q", n_tensors) + struct.pack("<Q", n_kv + len(kvs_extra))
        for key, t, raw in kvs:
            h += enc_str(key) + enc_raw(t, raw)
        for key, t, raw in kvs_extra:
            h += enc_str(key) + enc_raw(t, raw)
        for name, n_dims, dims, ttype, off in tensors:
            h += struct.pack("<Q", len(name)) + name
            h += struct.pack("<I", n_dims)
            for d in dims:
                h += struct.pack("<Q", d)
            h += struct.pack("<I", ttype)
            h += struct.pack("<Q", off)
        return h

    PAD_KEY = "general.ada_rename_pad"
    head = build([])
    padded = len(head) + ((-len(head)) % alignment)      # == GGML_PAD(len(head), align)
    need = old_data_start - padded
    if need:
        # a STRING kv costs 8 + len(key) + 4 + 8 + len(value); the key is
        # shortened when the gap is tight, the value absorbs the rest
        k = min(len(PAD_KEY), max(1, need - 21))
        key = PAD_KEY[:k]
        vlen = need - 20 - k
        assert vlen >= 0, "pad does not fit: need=%d key=%d" % (need, k)
        filler = (key, STRING, struct.pack("<Q", vlen) + b"\x00" * vlen)
        head = build([filler])
        print("filler kv    : %r, %d bytes" % (key, need))

    aligned_start = (-len(head)) % alignment + len(head)
    assert aligned_start == old_data_start, \
        "aligned start %d != original %d" % (aligned_start, old_data_start)
    print("header %d bytes, data still at %d, tensor offsets untouched"
          % (len(head), old_data_start))

    # rewrite the offset fields in place (they are the last 8 bytes of each entry)
    pos = 4 + 4 + 8 + 8          # "GGUF" + version + tensor_count + kv_count
    for key, t, raw in kvs:
        pos += 8 + len(key.encode()) + 4 + len(raw)
    pos += need                  # the filler entry
    for name, n_dims, dims, ttype, off in tensors:
        pos += 8 + len(name) + 4 + 8 * n_dims + 4
        struct.pack_into("<Q", head, pos, off)
        pos += 8

    assert pos == len(head), "header walk mismatch %d vs %d" % (pos, len(head))

    src_size = os.path.getsize(src)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(bytes(head))
        fout.write(b"\x00" * (old_data_start - len(head)))
        fin.seek(old_data_start)
        left = src_size - old_data_start
        done = 0
        while left:
            blk = fin.read(min(16 << 20, left))
            if not blk:
                break
            fout.write(blk)
            left -= len(blk)
            done += len(blk)
            if done % (1 << 30) < (16 << 20):
                print("   copied %.1f GB" % (done / 2**30))
    print("done: %s  (%.2f GB)" % (dst, os.path.getsize(dst) / 2**30))


if __name__ == "__main__":
    main()