#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Resolve the conflict markers left by sync_upstream.py.

Two kinds of block show up:

  * both sides added something different at the same spot (an #include, a
    switch case, an unrelated function) -- keep both, order does not matter;
  * both sides changed the same declarations -- keeping both duplicates a
    struct member, a field or a whole function.

The second kind is what broke the previous attempt. A block counts as a
duplicate when the identifiers it declares overlap on both sides, or when half
of its non-blank lines are identical after stripping whitespace; those keep
ours. Everything else is concatenated.

    python tools/resolve_conflicts.py            report only
    python tools/resolve_conflicts.py --apply    rewrite the files
"""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARK = re.compile(r"^<{7}", re.M)

BLOCK = re.compile(
    r"^<{7}[^\n]*\n(?P<ours>.*?)^\|{7}[^\n]*\n(?P<base>.*?)^={7}\n(?P<theirs>.*?)^>{7}[^\n]*\n",
    re.S | re.M)
BLOCK_NODIFF3 = re.compile(
    r"^<{7}[^\n]*\n(?P<ours>.*?)^={7}\n(?P<theirs>.*?)^>{7}[^\n]*\n",
    re.S | re.M)

KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "else", "case",
            "true", "false", "nullptr", "NULL", "break", "continue", "default",
            "do", "const", "static", "void", "int", "float", "bool", "auto"}

DECL = re.compile(r"\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*(?:=|;|\{)")


def declared_names(text):
    """Identifiers this block declares: `type name;`, `t name = ...`, `t f(...)`."""
    names = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("//", "#", "*", "/")):
            continue
        for m in DECL.finditer(s):
            name = m.group(1)
            if len(name) > 1 and name not in KEYWORDS:
                names.add(name)
    return names


def same_content(a, b):
    la = [l.strip() for l in a.splitlines() if l.strip()]
    lb = [l.strip() for l in b.splitlines() if l.strip()]
    if not la or not lb:
        return False
    common = len(set(la) & set(lb))
    return common / min(len(la), len(lb)) >= 0.5


def decide(ours, theirs):
    """(keep_ours_only, reason)"""
    shared = declared_names(ours) & declared_names(theirs)
    if shared:
        return True, "declares " + ", ".join(sorted(shared)[:3])
    if same_content(ours, theirs):
        return True, "same lines on both sides"
    return False, ""


def resolve(text):
    stats = {"keep_both": 0, "keep_ours": 0, "notes": []}

    def handler(m):
        ours, theirs = m.group("ours"), m.group("theirs")
        keep_ours, why = decide(ours, theirs)
        if keep_ours:
            stats["keep_ours"] += 1
            stats["notes"].append(why)
            return ours
        stats["keep_both"] += 1
        return ours + theirs

    text = BLOCK.sub(handler, text)
    text = BLOCK_NODIFF3.sub(handler, text)
    return text, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--files", nargs="*")
    args = ap.parse_args()

    if args.files:
        targets = [ROOT / f for f in args.files]
    else:
        targets = []
        for p in (ROOT / "code").rglob("*"):
            if not p.is_file() or p.suffix.lower() not in {
                    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp",
                    ".m", ".mm", ".txt", ".cmake"}:
                continue
            if b"\0" in p.read_bytes()[:4096]:
                continue
            targets.append(p)

    total = {"keep_both": 0, "keep_ours": 0}
    touched = 0
    for p in targets:
        text = p.read_bytes().decode("utf-8", "surrogateescape")
        if MARK.search(text) is None:
            continue
        text, stats = resolve(text)
        rel = p.relative_to(ROOT)
        touched += 1
        total["keep_both"] += stats["keep_both"]
        total["keep_ours"] += stats["keep_ours"]
        notes = "; ".join(sorted(set(stats["notes"]))[:2])
        flag = "  LEFT" if MARK.search(text) else ""
        print(f"  {str(rel):52s} both={stats['keep_both']:<3d} ours={stats['keep_ours']:<3d} {notes}{flag}")
        if args.apply:
            p.write_bytes(text.encode("utf-8", "surrogateescape"))

    print(f"\nfiles {touched}  keep-both {total['keep_both']}  keep-ours {total['keep_ours']}")
    if not args.apply:
        print("(report only; add --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
