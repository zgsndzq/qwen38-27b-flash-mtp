"""make_variant.py -- derive variant launchers from the production .bat.

Hard rules this respects:
  * the variant MUST live in the bundle root (every %~dp0 path is relative to it)
  * output must be CRLF + no BOM + pure ASCII (cmd 936 decodes as GBK; odd CJK bytes eat the next line)
  * the production file is only ever READ, never modified
  * write tmp -> verify -> move; never overwrite in place

Every variant prints a unified diff of its non-rem lines, so the only change is
provably the intended one (a mangled argv here would invalidate the A/B).
"""

import difflib
import sys
from pathlib import Path

ROOT = Path("F:/super/qwen38mtp")
PROD = ROOT / "llmserver.bat"

VARIANTS = {
    # name -> list of (old, new, expected_occurrences)
    "llmserver-nograph.bat": [
        ("set GGML_CUDA_GRAPH_OPT=1", "set GGML_CUDA_DISABLE_GRAPHS=1", 1),
    ],
    "llmserver-nopdl.bat": [
        ("set GGML_CUDA_PDL=1", "set GGML_CUDA_PDL=0", 1),
    ],
    # drop the 24k draft shortlist -> drafting must project onto the full 248k vocab
    # (line continuation has to be removed too, or cmd chokes on a dangling ^)
    "llmserver-nomap.bat": [
        ("--spec-draft-n-min-adaptive 1 ^", "--spec-draft-n-min-adaptive 1", 1),
        # real CRLF prefix: removes the whole line, including its leading newline
        ("\r\n  --spec-draft-vocab-map \"%~dp0maps\\shortlist_24k_mix.vocab.txt\" --spec-draft-vocab-hot 0", "", 1),
    ],
    # pure decode baseline: no speculative decoding at all
    "llmserver-nomtp.bat": [
        ("--spec-type draft-mtp-adaptive", "--spec-type none", 1),
    ],
    # fixed-depth MTP: drops the adaptive controller (which sees every accept
    # result, so it may force a host<->device round trip on every draft step)
    "llmserver-fixed.bat": [
        ("--spec-type draft-mtp-adaptive", "--spec-type draft-mtp", 1),
    ],
    # -lv 4 (debug) prints the per-context breakdown the INFO level swallows:
    # model buffer, KV self size per context, compute buffer size, worst-case ubatch
    "llmserver-lv4.bat": [
        ('"%~dp0bin\\llama-server.exe" -m', '"%~dp0bin\\llama-server.exe" -lv 4 -m', 1),
    ],
    # same, with no speculative decoding at all: the two logs together say exactly
    # what the MTP machinery costs in VRAM (the 844 MiB idle delta)
    "llmserver-nomtp-lv4.bat": [
        ("--spec-type draft-mtp-adaptive", "--spec-type none", 1),
        ('"%~dp0bin\\llama-server.exe" -m', '"%~dp0bin\\llama-server.exe" -lv 4 -m', 1),
    ],
}


# DRAFT sweep: DRAFT sets both --spec-draft-n-max and (via need_n_rs_seq) the width
# of the GDN recurrent-state snapshot ring, so it trades VRAM against accept depth.
for _d in (1, 2, 4, 5):
    VARIANTS["llmserver-draft%d.bat" % _d] = [
        ('set "DRAFT=3"', 'set "DRAFT=%d"' % _d, 1),
    ]


# LLAMA_SPEC_PQ=0 (production) forces id-match verification; the upstream default
# is exact p/q rejection sampling (speculative.cpp:1502-1511). Crossed with DRAFT,
# and once with the shortlist removed (a truncated q inflates q(x) and hurts p/q).
VARIANTS["llmserver-pq.bat"] = [
    ("set LLAMA_SPEC_PQ=0", "set LLAMA_SPEC_PQ=1", 1),
]
for _d in (1, 2):
    VARIANTS["llmserver-pq-draft%d.bat" % _d] = [
        ("set LLAMA_SPEC_PQ=0", "set LLAMA_SPEC_PQ=1", 1),
        ('set "DRAFT=3"', 'set "DRAFT=%d"' % _d, 1),
    ]
VARIANTS["llmserver-pq-nomap.bat"] = [
    ("set LLAMA_SPEC_PQ=0", "set LLAMA_SPEC_PQ=1", 1),
    ("--spec-draft-n-min-adaptive 1 ^", "--spec-draft-n-min-adaptive 1", 1),
    ("\r\n  --spec-draft-vocab-map \"%~dp0maps\\shortlist_24k_mix.vocab.txt\" --spec-draft-vocab-hot 0", "", 1),
]


def code_lines(text):
    return [l.strip() for l in text.splitlines()
            if l.strip() and not l.strip().startswith("rem")]


def main():
    raw = PROD.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        print("ABORT: production bat has a BOM")
        return 1
    text = raw.decode("ascii")
    if b"\r\n" not in raw:
        print("ABORT: production bat is not CRLF")
        return 1
    print("source: %s  %d bytes  CRLF=%d  non-ascii=0" % (
        PROD.name, len(raw), raw.count(b"\r\n")))

    for name, subs in VARIANTS.items():
        out = text
        ok = True
        for old, new, expect in subs:
            n = out.count(old)
            if n != expect:
                print("SKIP %s: %r found %d times (expected %d)" % (name, old[:60], n, expect))
                ok = False
                break
            out = out.replace(old, new)
        if not ok:
            continue
        data = out.encode("ascii")
        if data.count(b"\r\n") > raw.count(b"\r\n"):
            print("SKIP %s: line count grew" % name)
            continue
        tmp = ROOT / (name + ".tmp")
        tmp.write_bytes(data)
        if tmp.read_bytes() != data:
            print("SKIP %s: tmp verify failed" % name)
            continue
        tmp.replace(ROOT / name)
        diff = [l for l in difflib.unified_diff(code_lines(text), code_lines(out),
                                                lineterm="", n=0)
                if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
        print("\n%-24s %d bytes  code-line diff:" % (name, len(data)))
        for l in diff:
            print("    %s" % l[:150])
    return 0


if __name__ == "__main__":
    sys.exit(main())
