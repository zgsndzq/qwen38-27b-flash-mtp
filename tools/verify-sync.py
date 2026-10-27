"""Audit a merged tree against upstream: every line upstream added must be in it.

git merge-file reports "clean" for a hunk it folds into a local rewrite while
dropping the upstream side, so a zero-conflict merge still has to be checked
line by line (common.cuh lost three context members that way on the 2026-10-27
sync; only the compiler caught it).

Absent lines are not automatically wrong: a line upstream added can be rejected
on purpose (PQ2_0-only code, the vision-encoder FA padding). The point is that
each one gets looked at.

    tools/verify-sync.py [base] [ref]
"""

import difflib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = "code"


def rev(path, rev):
    out = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{path}"],
                         capture_output=True, text=True, encoding="utf8", errors="ignore")
    return out.stdout.split("\n") if out.returncode == 0 else None


def norm(lines):
    return [l.strip() for l in lines if l.strip()]


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else open(os.path.join(ROOT, ".upstream-base"), encoding="utf8").read().strip()
    ref = sys.argv[2] if len(sys.argv) > 2 else "prism/prism"

    skip = set()
    skip_file = os.path.join(ROOT, ".upstream-skip")
    if os.path.exists(skip_file):
        skip = {l.strip() for l in open(skip_file, encoding="utf8") if l.strip()}

    changed = subprocess.run(["git", "-C", ROOT, "diff", "--name-only", base, ref],
                             capture_output=True, text=True, encoding="utf8").stdout.split("\n")

    total = 0
    for f in filter(None, changed):
        if f in skip:
            continue
        theirs = rev(f, ref)
        final_path = os.path.join(ROOT, PREFIX, f)
        if theirs is None or not os.path.exists(final_path):
            continue
        base_lines = rev(f, base)
        if base_lines is None:
            continue
        added = [l[1:] for l in difflib.unified_diff(norm(base_lines), norm(theirs), n=0, lineterm="")
                 if l.startswith("+") and not l.startswith("+++")]
        if not added:
            continue
        final = set(norm(open(final_path, encoding="utf8", errors="ignore").read().split("\n")))
        lost = [l for l in added if l.strip() not in final]
        if not lost:
            continue
        total += len(lost)
        print(f"== {f}: {len(lost)}/{len(added)} upstream-added lines absent")
        for l in lost:
            print("   -", l.strip()[:120])
    print(f"\ntotal absent lines: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
