"""Pull just the MTP tensors out of Qwen/Qwen3.5-4B by HTTP range.

Safetensors layout: 8-byte little-endian header length, JSON header, then a
flat data blob. Only the 15 mtp.* tensors are fetched, so this costs ~220 MB
instead of the full 8 GB checkpoint.
"""
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://hf-mirror.com/Qwen/Qwen3.5-4B/resolve/main/"
SHARDS = [
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
]
OUT = Path(__file__).resolve().parent / "mtp"
DT = {  # safetensors dtype -> bytes
    "BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "U8": 1, "BOOL": 1,
}


def get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for shard in SHARDS:
        url = BASE + shard
        head = get(url, 0, 1 << 20)
        (n,) = struct.unpack("<Q", head[:8])
        if 8 + n > len(head):
            head = get(url, 0, 8 + n - 1)
        hdr = json.loads(head[8:8 + n])
        base_off = 8 + n
        todo = {k: v for k, v in hdr.items() if k.startswith("mtp.")}
        print(f"{shard}: header {n} B, {len(todo)} mtp tensors")
        for name, info in sorted(todo.items()):
            s, e = info["data_offsets"]
            dt = info["dtype"]
            expect = e - s
            assert expect == DT[dt] * int(__import__("math").prod(info["shape"])), (name, expect)
            t0 = time.time()
            blob = get(url, base_off + s, base_off + e - 1)
            if len(blob) != expect:
                sys.exit(f"{name}: got {len(blob)} bytes, expected {expect}")
            fn = name.replace(".", "_") + ".bin"
            (OUT / fn).write_bytes(blob)
            manifest[name] = {"file": fn, "dtype": dt, "shape": info["shape"], "bytes": expect}
            print(f"  {name:46s} {dt:4s} {str(info['shape']):14s} {expect/1e6:7.2f} MB  {time.time()-t0:.1f}s")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(v["bytes"] for v in manifest.values())
    print(f"\n{len(manifest)} tensors, {total/1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
