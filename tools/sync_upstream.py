#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Three-way sync of code/ against an upstream commit, with path mapping.

git merge cannot do this: our history never had the upstream commit as a parent,
so there is no merge base and the tree layouts differ (upstream root <-> our
code/). This does the merge at the tree level instead:

    base   = the upstream commit we last synced to
    theirs = the upstream commit we want
    ours   = HEAD:code

Per file, three cases: only theirs changed (take it), only ours changed (keep
it), both changed (git merge-file). Report first, write with --apply.

    python tools/sync_upstream.py --base c96baf4 --theirs prism/prism
    python tools/sync_upstream.py --base c96baf4 --theirs prism/prism --apply
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git(*args, text=True):
    return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True,
                          text=text, errors="surrogateescape" if text else None)


def tree(rev, prefix=""):
    """{relpath: blob sha} for every file under prefix of rev."""
    spec = rev + ":" + prefix if prefix else rev + ":"
    out = git("ls-tree", "-r", spec).stdout
    result = {}
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) >= 3 and parts[1] == "blob":
            result[path] = parts[2]
    return result


def blob(sha):
    """Empty when the file does not exist on that side."""
    if sha is None:
        return ""
    return git("cat-file", "blob", sha).stdout


def merge3(ours, base, theirs):
    """git merge-file on three strings; returns (text, conflicts)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        (p / "ours").write_bytes(ours.encode("utf-8", "surrogateescape"))
        (p / "base").write_bytes(base.encode("utf-8", "surrogateescape"))
        (p / "theirs").write_bytes(theirs.encode("utf-8", "surrogateescape"))
        r = subprocess.run(["git", "merge-file", "-p", "--diff3",
                            str(p / "ours"), str(p / "base"), str(p / "theirs")],
                           capture_output=True)
        text = r.stdout.decode("utf-8", "surrogateescape")
        return text, max(r.returncode, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="上游 commit，我们上次同步到的点")
    ap.add_argument("--theirs", required=True, help="要同步到的上游 commit")
    ap.add_argument("--apply", action="store_true", help="写入文件（默认只报告）")
    args = ap.parse_args()

    base = tree(args.base)                       # upstream layout
    theirs = tree(args.theirs)
    ours = tree("HEAD", "code")                  # our layout

    added = only_theirs = only_ours = both = conflicts = 0
    conflict_files, written = [], 0

    for path in sorted(set(base) | set(theirs) | set(ours)):
        b, t, o = base.get(path), theirs.get(path), ours.get(path)

        if b == t:                               # 上游没动 -> 保留我们
            if o is not None:
                only_ours += 1
            continue
        if o == b:                               # 我们没动 -> 取上游
            if t is not None:
                only_theirs += 1
                if args.apply:
                    dest = ROOT / "code" / path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(git("cat-file", "blob", t, text=False).stdout)
                    written += 1
            else:
                if args.apply and (ROOT / "code" / path).exists():
                    (ROOT / "code" / path).unlink()
                    written += 1
            continue
        if o is None:                            # 上游新增的文件
            added += 1
            if args.apply and t:
                dest = ROOT / "code" / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(git("cat-file", "blob", t, text=False).stdout)
                written += 1
            continue
        if t is None:                            # 上游删了，我们改过 -> 留着并报出来
            conflict_files.append((path, "upstream deleted, we changed"))
            continue

        both += 1
        text, rc = merge3(blob(o), blob(b), blob(t))
        if rc:
            conflicts += 1
            conflict_files.append((path, "%d conflict(s)" % rc))
        if args.apply:
            dest = ROOT / "code" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(text.encode("utf-8", "surrogateescape"))
            written += 1

    print("base   %s" % args.base)
    print("theirs %s" % args.theirs)
    print()
    print("upstream only changed : %4d" % only_theirs)
    print("upstream new files    : %4d" % added)
    print("we only changed       : %4d" % only_ours)
    print("both changed          : %4d  (auto-mergeable unless listed)" % both)
    print("conflicts             : %4d" % conflicts)
    if args.apply:
        print("written               : %4d" % written)
    if conflict_files:
        print("\nneeds a human:")
        for path, why in conflict_files:
            print("  %-58s %s" % (path, why))
    return 0


if __name__ == "__main__":
    sys.exit(main())
