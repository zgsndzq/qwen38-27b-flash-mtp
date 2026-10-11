"""make_draft_variants.py -- set the MTP draft depth cap in llmserver.bat.

WHY THIS EXISTS
  The adaptive depth controller (common/speculative-adaptive.h) only watches the
  ACCEPTANCE rate: a full accept climbs the depth, a miss drops it. It has no
  idea what a draft step costs. Measured on this box (tools/round_decomp.py),
  one draft step is ~10.2 ms while its own memory traffic is ~0.12 ms -- the
  marginal step is almost pure per-call fixed overhead. So the controller is
  happy to park at depth 2.2-2.7 even though a shallower round is cheaper.

  Scanning the cap with tools/ab_bench.py --prompts scenes (3 prompts x 3-4
  runs, one session per arm, same GPU):

      cap 3 (old)   39.80 / 34.94 / 43.02 t/s   accept .48/.38/.56   depth 2.2-2.7
      cap 2         40.02 / 39.54 / 45.49       accept .52/.51/.68   depth 1.9
      cap 1         41.11 / 39.21 / 44.52       accept .68/.61/.82   depth 0.99
    run 2:
      cap 2         40.67 / 37.51 / 42.12
      cap 1         39.64 / 37.52 / 42.64
      cap 3         35.61 / 35.88 / 41.81

  Merged over both rounds: cap 2 = 40.90, cap 1 = 40.78, cap 3 = 38.51.
  cap 1 and cap 2 are a wash on the mean, but cap 1 has roughly HALF the
  spread (worst single run 36.01 vs 32.89, range 1.9-3.7 vs 5.0-6.4) and it is
  the simplest config: at cap 1 the controller has no room to climb, so it is
  plain fixed-depth-1 drafting. That is what this file ships.

USAGE
  python tools/make_draft_variants.py 1 2       # scratch clones -> _ab_nN.bat
  python tools/make_draft_variants.py --apply 1 # patch llmserver.bat itself
  python tools/make_draft_variants.py --clean   # remove the scratch clones

  --apply archives the current launcher to
  F:\\super\\.archive\\<date>-draft<old>\\llmserver.bat.bak first, then writes
  through a .tmp file and swaps it in. Never overwrites in place: an earlier
  incident truncated this exact file to 2 bytes with a direct WriteAllBytes.
  Every clone/patch is re-read and asserted to be pure CRLF with zero non-ASCII
  bytes (cmd runs codepage 936 and a stray CJK byte eats the next line's `rem`).
"""
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "llmserver.bat"
ARCHIVE = Path(r"F:\super\.archive")
NAME = b'set "MODEL_NAME=Ternary-Bonsai-2-27B-SelfMTP-Q6K [qwen38mtp standalone]"'

OLD_HEADER = (b'rem  Measured 2026-09-22 (build-gdn, 3 prompts x 3 runs, median):\r\n'
              b'rem    DRAFT=3 -> 39.22 t/s, accept 0.549     (this file ships DRAFT=3)\r\n')
NEW_HEADER = (b'rem  Measured 2026-09-22 (build-gdn, 3 prompts x 3 runs, median):\r\n'
              b'rem    DRAFT=3 -> 39.22 t/s, accept 0.549\r\n')

OLD_NOTE = b'rem 3 = the 2026-09-22 head-A/B baseline. 1 = accept 0.75 (+20% on prose).\r\n'
NEW_NOTE = (b'rem DRAFT=1 SHIPPING (2026-09-24). One draft step measures ~10.2 ms while its own\r\n'
            b'rem memory traffic is ~0.12 ms, so the marginal step is NOT bandwidth-bound\r\n'
            b'rem and a shallower round wins. It is also not pure launch overhead:\r\n'
            b'rem --spec-chain merges the N draft decodes into one and measured +2% (noise).\r\n'
            b'rem Scanning the adaptive cap\r\n'
            b'rem (tools/ab_bench.py --prompts scenes, 3 prompts x 3-4 runs, same session):\r\n'
            b'rem   cap 3 (old)  39.80/34.94/43.02 t/s  accept .48/.38/.56  depth 2.2-2.7\r\n'
            b'rem   cap 2        40.02/39.54/45.49      accept .52/.51/.68  depth 1.94\r\n'
            b'rem   cap 1        41.11/39.21/44.52      accept .68/.61/.82  depth 0.99\r\n'
            b'rem cap 1 is ~+6% over cap 3 and has about half its spread (worst single run\r\n'
            b'rem 36.0 vs 32.9 at cap 2/3). At cap 1 the controller has no room to climb,\r\n'
            b'rem so this is plain fixed-depth-1 drafting. The controller only watches\r\n'
            b'rem acceptance -- it cannot see the 10.2 ms an extra step costs, which is\r\n'
            b'rem why the old cap 3 parked too deep.\r\n')


def check(raw, what):
    if raw.startswith(b"\xef\xbb\xbf"):
        raise SystemExit("%s has a BOM -- abort" % what)
    if raw.count(b"\n") != raw.count(b"\r\n"):
        raise SystemExit("%s is not pure CRLF -- abort" % what)
    bad = [i for i, b in enumerate(raw) if b >= 0x80]
    if bad:
        raise SystemExit("%s has %d non-ASCII byte(s), first at %d -- abort"
                         % (what, len(bad), bad[0]))
    return raw


def set_cap(raw, n):
    old = b'set "DRAFT=%d"' % current_cap(raw)
    if raw.count(old) != 1:
        raise SystemExit("cannot locate the DRAFT line (%r x%d)" % (old, raw.count(old)))
    out = raw.replace(old, b'set "DRAFT=%d"' % n)
    if n == 1 and OLD_NOTE in out:
        out = out.replace(OLD_NOTE, NEW_NOTE)
    if OLD_HEADER in out:
        out = out.replace(OLD_HEADER, NEW_HEADER)
    return out


def current_cap(raw):
    m = re.search(rb'set "DRAFT=(\d+)"', raw)
    if not m:
        raise SystemExit("no DRAFT= line in llmserver.bat")
    return int(m.group(1))


def write_checked(dst, data):
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dst)
    back = check(dst.read_bytes(), dst.name)
    if back != data:
        raise SystemExit("readback mismatch on %s" % dst)
    return back


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)

    if args == ["--clean"]:
        for p in sorted(ROOT.glob("_ab_n*.bat")):
            p.unlink()
            print("removed %s" % p.name)
        return 0

    if args[0] == "--apply":
        n = int(args[1])
        if not 1 <= n <= 3:
            raise SystemExit("refusing cap %d (the file's own note: >3 loses on code)" % n)
        raw = check(SRC.read_bytes(), "llmserver.bat")
        old = current_cap(raw)
        stamp = datetime.datetime.now().strftime("%Y%m%d")
        bak = ARCHIVE / ("%s-draft%d" % (stamp, old)) / "llmserver.bat.bak"
        bak.parent.mkdir(parents=True, exist_ok=True)
        bak.write_bytes(raw)
        print("archived %s -> %s (%d bytes)" % (SRC.name, bak, len(raw)))
        data = check(set_cap(raw, n), "patched llmserver.bat")
        write_checked(SRC, data)
        print("applied DRAFT=%d to %s (%d bytes, crlf=%s, non-ascii=0)" % (
            n, SRC.name, len(data), data.count(b"\r\n") == data.count(b"\n")))
        return 0

    for a in args:
        n = int(a)
        if not 1 <= n <= 3:
            raise SystemExit("refusing cap %d (the file's own note: >3 loses on code)" % n)
        data = check(set_cap(SRC.read_bytes(), n), "clone")
        data = data.replace(NAME, b'set "MODEL_NAME=AB draft cap %d [scratch, not production]"' % n)
        dst = ROOT / ("_ab_n%d.bat" % n)
        write_checked(dst, data)
        print("%s  %d bytes  DRAFT=%d  crlf=%s  non-ascii=0" % (
            dst.name, len(data), n, data.count(b"\r\n") == data.count(b"\n")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
