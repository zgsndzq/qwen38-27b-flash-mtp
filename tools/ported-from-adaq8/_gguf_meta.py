#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump the metadata KV block of a GGUF (header only, no tensor data read)."""
import struct
import sys

PATH = r"F:\super\qwen27b\weights\Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"
OUT = r"F:\super\tmp\_gguf_meta.txt"

sink = open(OUT, 'w', encoding='utf-8')
sys.stdout = sink
sys.stderr = sink

# value type -> (struct fmt or None for special, size or None)
SCALAR = {
    0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2),
    4: ('<I', 4), 5: ('<i', 4), 6: ('<f', 4), 7: ('<?', 1),
    10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8),
}
TYPE_NAME = {0: 'u8', 1: 'i8', 2: 'u16', 3: 'i16', 4: 'u32', 5: 'i32',
             6: 'f32', 7: 'bool', 8: 'str', 9: 'arr', 10: 'u64', 11: 'i64', 12: 'f64'}


def read_str(f):
    (n,) = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8', 'replace')


def read_value(f, vtype):
    if vtype == 8:
        return read_str(f)
    if vtype == 9:
        (elem,) = struct.unpack('<I', f.read(4))
        (n,) = struct.unpack('<Q', f.read(8))
        if elem == 8 and n <= 8:
            return [read_str(f) for _ in range(n)]
        if elem in SCALAR:
            fmt, sz = SCALAR[elem]
            raw = f.read(sz * n)
            return list(struct.unpack('<' + fmt[1] * n, raw))
        return '<array of %d x %s>' % (n, TYPE_NAME.get(elem, elem))
    if vtype in SCALAR:
        fmt, sz = SCALAR[vtype]
        (v,) = struct.unpack(fmt, f.read(sz))
        return v
    raise ValueError('unknown value type %d' % vtype)


with open(PATH, 'rb') as f:
    magic = f.read(4)
    print('magic        : %r  %s' % (magic, 'OK' if magic == b'GGUF' else 'BAD'))
    (ver,) = struct.unpack('<I', f.read(4))
    (n_tensors,) = struct.unpack('<Q', f.read(8))
    (n_kv,) = struct.unpack('<Q', f.read(8))
    print('version      : %d' % ver)
    print('tensor_count : %d' % n_tensors)
    print('metadata_kv  : %d' % n_kv)
    print()

    for _ in range(n_kv):
        key = read_str(f)
        (vtype,) = struct.unpack('<I', f.read(4))
        val = read_value(f, vtype)
        shown = val
        if isinstance(val, str) and len(val) > 220:
            shown = val[:220] + '...(%d chars)' % len(val)
        if isinstance(val, list) and len(val) > 12:
            shown = str(val[:12]) + ' ...(%d items)' % len(val)
        print('%-34s %-5s %s' % (key, TYPE_NAME.get(vtype, vtype), shown))

    print()
    print('=== tensor info (first 12 + count by prefix) ===')
    from collections import Counter
    pref = Counter()
    first = []
    for i in range(n_tensors):
        name = read_str(f)
        (ndim,) = struct.unpack('<I', f.read(4))
        dims = struct.unpack('<' + 'Q' * ndim, f.read(8 * ndim))
        (ttype,) = struct.unpack('<I', f.read(4))
        (off,) = struct.unpack('<Q', f.read(8))
        pref[name.split('.')[0]] += 1
        if i < 12:
            first.append((name, dims, ttype))
    for n, d, t in first:
        print('  %-46s dims=%-24s type=%d' % (n, d, t))
    print()
    for k, c in sorted(pref.items(), key=lambda x: -x[1]):
        print('  %-28s %d' % (k, c))
