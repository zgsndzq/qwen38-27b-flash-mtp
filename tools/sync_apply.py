#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Put the merge result in place file by file, the way the sync plan says.

Three groups, because one rule cannot be right for all of them:

  ours   -- files we rewrote or extended (q3_0 dispatch, dspark, tier KV,
            speculative loop, our parameters). Take HEAD, then adapt to the
            upstream interfaces during the build.
  theirs -- files where upstream changed something we never touched, plus the
            interface headers we only add fields to (done separately).
  both   -- nothing here; a file that needs both is listed as ours and gets
            the upstream bits pasted in by hand.

    python tools/sync_apply.py            list what would happen
    python tools/sync_apply.py --apply    do it
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Our own code: take HEAD wholesale.
OURS = [
    "common/arg.cpp",
    "common/common.h",
    "common/speculative.cpp",
    "common/speculative.h",
    "common/sampling.cpp",
    "common/common.cpp",
    "common/chat.cpp",
    "README.md",
    "src/llama-graph.cpp",
    "src/llama-kv-cache.cpp",
    "src/llama-memory-recurrent.cpp",
    "src/llama-model.cpp",
    "src/llama-arch.cpp",
    "src/models/qwen35.cpp",
    "tests/test-backend-ops.cpp",
    "tests/test-chat.cpp",
    "tools/server/server-context.cpp",
    "ggml/src/ggml-cuda/ggml-cuda.cu",
    "ggml/src/ggml-cuda/fattn.cu",
    "ggml/src/ggml-cuda/fattn-mma-f16.cuh",
    "ggml/src/ggml-cuda/fattn-common.cuh",
    "ggml/src/ggml-cuda/fattn-vec.cuh",
    "ggml/src/ggml-cuda/common.cuh",
    "ggml/src/ggml-cuda/mmvq.cu",
    "ggml/src/ggml-cuda/mmq.cu",
    "ggml/src/ggml-cuda/mmq.cuh",
    "ggml/src/ggml-sycl/ggml-sycl.cpp",
    "ggml/src/ggml-sycl/mmvq.cpp",
    "ggml/src/ggml-sycl/vecdotq.hpp",
    "ggml/src/ggml-sycl/mmvq.hpp",
    "ggml/src/ggml-sycl/common.hpp",
    "ggml/src/ggml-sycl/convert.cpp",
    "ggml/src/ggml-sycl/fattn-mkl.cpp",
    "ggml/src/ggml-sycl/fusion.cpp",
    "ggml/src/ggml-sycl/fwht.cpp",
    "ggml/src/ggml-sycl/gemm.hpp",
]


def git(*args):
    return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True,
                          text=True, errors="surrogateescape")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    conf = [l.strip() for l in (ROOT / "conflicts.txt").read_text().splitlines() if l.strip()]
    ours = set(OURS)
    theirs = [f for f in conf if f not in ours]
    missing = [f for f in ours if f not in conf]

    print(f"take OURS   : {len([f for f in conf if f in ours])}")
    print(f"take THEIRS : {len(theirs)}")
    if missing:
        print(f"listed but not conflicting: {missing}")

    print("\n--- theirs (upstream version, then we paste our bits back) ---")
    for f in theirs:
        print(f"  {f}")

    if not args.apply:
        print("\n(dry run; add --apply)")
        return 0

    for f in theirs:
        data = subprocess.run(["git", "show", f"prism/prism:{f}"], cwd=ROOT,
                              capture_output=True).stdout
        (ROOT / "code" / f).write_bytes(data)
    print(f"\ntheirs written: {len(theirs)}")

    for f in [x for x in conf if x in ours]:
        data = subprocess.run(["git", "show", f"HEAD:code/{f}"], cwd=ROOT,
                              capture_output=True).stdout
        (ROOT / "code" / f).write_bytes(data)
    print(f"ours written: {len([f for f in conf if f in ours])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
