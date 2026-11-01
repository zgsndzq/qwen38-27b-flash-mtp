#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Apply a per-block decision to a file left with conflict markers.

Nothing here decides anything: the caller passes one letter per conflict block,
in file order.

    o  keep ours
    t  keep theirs
    b  keep both, ours first
    d  keep both, theirs first

    python tools/pick.py code/ggml/src/ggml.c oo
    python tools/pick.py code/ggml/src/ggml-cuda/mmq.cu b
"""

import argparse
import re
import sys
from pathlib import Path

BLOCK = re.compile(
    r"^<{7}[^\n]*\n(?P<ours>.*?)(?:^\|{7}[^\n]*\n(?P<base>.*?))?^={7}\n(?P<theirs>.*?)^>{7}[^\n]*\n",
    re.S | re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("choices", help="one letter per block: o t b d")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    path = Path(args.file)
    text = path.read_bytes().decode("utf-8", "surrogateescape")
    blocks = list(BLOCK.finditer(text))
    if not blocks:
        print(f"{path}: no conflict markers")
        return 1

    choices = args.choices.strip()
    if len(choices) != len(blocks):
        print(f"{path}: {len(blocks)} blocks but {len(choices)} choices")
        for i, m in enumerate(blocks, 1):
            first = next((l.strip() for l in m.group("ours").splitlines() if l.strip()), "")
            print(f"  block {i}: {first[:70]}")
        return 2

    out = []
    last = 0
    for m, c in zip(blocks, choices):
        ours = m.group("ours") or ""
        theirs = m.group("theirs") or ""
        if c == "o":
            rep = ours
        elif c == "t":
            rep = theirs
        elif c == "b":
            rep = ours + theirs
        elif c == "d":
            rep = theirs + ours
        else:
            print(f"unknown choice {c}")
            return 2
        out.append(text[last:m.start()])
        out.append(rep)
        last = m.end()
    out.append(text[last:])
    new = "".join(out)

    if args.dry:
        print(f"{path}: {len(blocks)} blocks -> {''.join(choices)} (dry run)")
        return 0
    path.write_bytes(new.encode("utf-8", "surrogateescape"))
    print(f"{path}: {len(blocks)} blocks -> {choices}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
