# -*- coding: utf-8 -*-
"""Install freshly built ada-q8 binaries into F:\\super\\adaq8\\bin.

Only files whose md5 actually changed are touched; the previous copy of each is
kept next to it as `<name>.bak-<ts>` so a rollback is a single Move-Item.

Windows locks the binaries of a running server: the image cannot be written, but
it CAN be renamed. When a copy fails with PermissionError the old file is moved
aside as `<name>.inuse-<ts>` and the new one dropped in its place. The running
process keeps its existing mapping and is NOT disturbed -- the new code simply
takes effect at the next start.
"""
import hashlib
import io
import os
import shutil
import sys
import time

SRC = r"F:\super\qwen27b\ada-q8\build-q8\bin"
DST = r"F:\super\qwen27b\bin"
TS = time.strftime("%Y%m%d-%H%M%S")


def md5(p):
    h = hashlib.md5()
    with io.open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    names = sorted(n for n in os.listdir(SRC) if n.lower().endswith((".exe", ".dll")))
    if not names:
        print("no binaries in %s" % SRC)
        return 1
    print("source %s" % SRC)
    print("dest   %s" % DST)
    print()
    changed, same, added = [], [], []
    for n in names:
        s = os.path.join(SRC, n)
        d = os.path.join(DST, n)
        if not os.path.isfile(d):
            added.append(n)
            continue
        ms, md = md5(s), md5(d)
        if ms == md:
            same.append(n)
        else:
            changed.append((n, ms, md))

    print("CHANGED (%d):" % len(changed))
    for n, ms, md in changed:
        print("  %-26s src=%s  cur=%s" % (n, ms[:12], md[:12]))
    print("NEW (%d): %s" % (len(added), ", ".join(added) if added else "-"))
    print("UNCHANGED (%d): %s" % (len(same), ", ".join(same)))
    print()

    if os.environ.get("INSTALL_DRYRUN") == "1":
        print("dry run -- nothing copied")
        return 0

    locked = []
    for n, _, _ in changed:
        s = os.path.join(SRC, n)
        d = os.path.join(DST, n)
        shutil.copy2(d, d + ".bak-" + TS)
        try:
            shutil.copy2(s, d)
        except PermissionError:
            # the path is held by a running process -- step around it
            try:
                os.replace(d, d + ".inuse-" + TS)
                shutil.copy2(s, d)
                locked.append(n)
            except Exception as e:
                print("%-26s LOCKED and could not be moved aside: %s" % (n, e))
                print("   -> close the server and re-run this script")
                return 1
        ok = md5(d) == md5(s)
        print("%-26s installed  md5 match=%s  (backup %s.bak-%s)" % (n, ok, n, TS))
        if not ok:
            print("MD5 MISMATCH after copy for %s -- aborting" % n)
            return 1

    print()
    print("done: %d file(s) replaced, rollback = Move-Item bin\\*.bak-%s over the originals"
          % (len(changed), TS))
    if locked:
        print()
        print("NOTE: %d file(s) were in use by a running server and were moved aside as" % len(locked))
        print("      *.inuse-%s (the live process keeps running the OLD code):" % TS)
        for n in locked:
            print("        %s" % n)
        print("      close the server, then delete the *.inuse-%s files." % TS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
