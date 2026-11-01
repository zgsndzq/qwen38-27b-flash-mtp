#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check that every switch in features.ini still has a reader in code/.

The startup log only says "applied N" -- that means the key was recognised and
handed to a handler, not that anything reads the value. This walks the keys and
looks for the two real control points:

    getenv("KEY")            an environment switch
    "--key" in arg.cpp       a common_arg handler

Exit code 1 when a key has no control point (the implementation it gated is
gone, most likely overwritten by an upstream merge).

    python tools/check_features.py [--verbose]
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
INI = ROOT / "features.ini"
EXTS = {".c", ".cpp", ".cu", ".h", ".cuh", ".hpp", ".mm"}


def read_ini():
    """[(section, key, value)] with comments and blanks dropped."""
    out, section = [], ""
    for line in INI.read_text(encoding="utf-8").splitlines():
        text = line.split(";")[0].split("#")[0].strip()
        if not text:
            continue
        if text.startswith("["):
            section = text.strip("[]").strip()
            continue
        if "=" in text:
            key, value = text.split("=", 1)
            out.append((section or "-", key.strip(), value.strip()))
    return out


def load_sources():
    return {p: p.read_text(encoding="utf-8", errors="replace")
            for p in CODE.rglob("*") if p.suffix in EXTS and p.is_file()}


def find_reader(key, sources, argcpp):
    """(kind, site) where the key is actually consumed, or (None, None)."""
    pattern = re.compile(r'getenv\s*\(\s*"%s"\s*\)' % re.escape(key))
    for path, text in sources.items():
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                return "env", "%s:%d" % (path.relative_to(ROOT), lineno)
    if ('"--%s"' % key) in argcpp or ('"%s"' % key) in argcpp:
        return "arg", "code/common/arg.cpp"
    return None, None


def main():
    verbose = "--verbose" in sys.argv[1:]
    sources = load_sources()
    argcpp = sources.get(CODE / "common" / "arg.cpp", "")
    keys = list(dict.fromkeys(read_ini()))

    width = max(len(k) for _, k, _ in keys)
    dead = []
    for section, key, value in keys:
        kind, site = find_reader(key, sources, argcpp)
        if kind is None:
            dead.append(key)
            print("%-*s  %-8s DEAD -- no getenv, no arg handler" % (width, key, section))
        elif verbose:
            print("%-*s  %-8s %-4s %s" % (width, key, section, kind, site))

    print("\n%d switches, %d without a reader" % (len(keys), len(dead)))
    if dead:
        print("missing: " + ", ".join(dead))
        print("A switch that is read nowhere means the code behind it is gone.")
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())
