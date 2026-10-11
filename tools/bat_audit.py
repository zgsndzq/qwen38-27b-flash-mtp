"""bat_audit.py -- READ-ONLY health check for the .bat launchers in this bundle.

Three static checks together catch every launcher bug we have actually hit:

  1. non-ASCII bytes  -- a UTF-8 CJK comment read under cmd codepage 936 decodes
                         misaligned: the trailing half of a multi-byte char pairs
                         with the next line's first byte and EATS its `rem`, so
                         the rest of that line runs as a command
                         ("'xxx' is not recognized as an internal ...").
  2. line endings     -- cmd wants CRLF; a bare-LF line can be swallowed.
  3. UTF-8 BOM        -- cmd echoes the BOM bytes before @echo off.

Plus a semantic diff: it extracts the actual launch argv from each launcher and
prints the token delta. That is the check that answers "what does this variant
really change?" without trusting the comment block.

Nothing is ever written. Exit code 0 = all clean, 2 = at least one problem.

Usage:
  python bat_audit.py                    # audit every .bat in the bundle root
  python bat_audit.py a.bat b.bat        # audit specific files
  python bat_audit.py --diff a.bat b.bat # audit both, then show the argv delta
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A line whose first token is one of these is "intended as" a command line.
# If GBK decoding changes that first token, the line has been damaged.
CMD_HEADS = ("rem", "set", "echo", "@", "if", ")", "(", ":", "^", "title",
             "exit", "call", "for", "goto", "cd", "pause", "star", "taskkill")


def first_token(low_stripped):
    parts = low_stripped.split()
    return parts[0] if parts else ""


def gbk_damage(raw):
    """Return [(lineno, intended, decoded), ...] for lines broken under cp936."""
    try:
        intended = raw.decode("utf-8").split("\r\n")
    except UnicodeDecodeError:
        intended = raw.decode("utf-8", "replace").split("\r\n")
    decoded = raw.decode("gbk", "replace").split("\r\n")
    bad = []
    for i in range(min(len(intended), len(decoded))):
        want = intended[i].strip()
        if not want:
            continue
        low = want.lower()
        if not any(low.startswith(h) for h in CMD_HEADS):
            continue
        head = first_token(low)
        if head and not decoded[i].strip().lower().startswith(head):
            bad.append((i + 1, want[:52], decoded[i].strip()[:52]))
    return bad


def extract_argv(raw):
    """Best-effort: the logical launch command line, with `^` continuations joined."""
    text = raw.decode("utf-8", "replace").replace("\r\n", "\n")
    logical, buf = [], ""
    for line in text.split("\n"):
        s = line.rstrip()
        if s.endswith("^"):
            buf += s[:-1]
            continue
        if buf:
            buf += s
            logical.append(buf)
            buf = ""
        else:
            logical.append(s)
    # The launch line is the non-rem/non-set logical line that actually invokes
    # the server binary. Pass 1 keys on the binary because some launchers build
    # their flags into a %SPECARGS% variable first, so an "if ... set ..." line
    # can carry flags while not being the launch line. Pass 2 is the fallback.
    skip = ("rem", "set ", "set\"", "echo", "::", "if ", "if(")
    cands = []
    for line in logical:
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if any(low.startswith(h) for h in skip):
            continue
        cands.append(s)
    for keys in (("llama-server.exe", "%bin%", "%exe%"), ("--",)):
        hits = [s for s in cands if any(k in s.lower() for k in keys)]
        if hits:
            return hits[-1]
    return cands[-1] if cands else ""


def tokenize(argv):
    """Split the argv line into comparable tokens (whitespace kept simple)."""
    out, cur, q = [], "", None
    for ch in argv:
        if q:
            if ch == q:
                q = None
            else:
                cur += ch
            continue
        if ch in "\"'":
            q = ch
            continue
        if ch in " \t":
            if cur:
                out.append(cur)
                cur = ""
            continue
        cur += ch
    if cur:
        out.append(cur)
    return out


def audit(path):
    raw = path.read_bytes()
    non_ascii = sum(1 for b in raw if b > 127)
    crlf = raw.count(b"\r\n")
    bare_lf = raw.count(b"\n") - crlf
    bom = raw[:3] == b"\xef\xbb\xbf"
    damage = gbk_damage(raw)
    problems = []
    if non_ascii:
        problems.append("non-ASCII bytes=%d" % non_ascii)
    if bom:
        problems.append("UTF-8 BOM present")
    if bare_lf:
        problems.append("bare LF=%d" % bare_lf)
    if damage:
        problems.append("GBK-damaged lines=%d" % len(damage))
    print("%-28s %7d B  ASCII=%-5s CRLF=%-4d LF=%-3d BOM=%-5s damage=%d  %s" % (
        path.name, len(raw), non_ascii == 0, crlf, bare_lf, bom, len(damage),
        "OK" if not problems else "FAIL: " + "; ".join(problems)))
    for no, want, got in damage[:5]:
        print("      L%-4d want: %s" % (no, want))
        print("             got: %s" % got)
    return (not problems), extract_argv(raw)


def main():
    args = [a for a in sys.argv[1:] if a != "--diff"]
    want_diff = "--diff" in sys.argv

    files = [Path(a) if Path(a).is_absolute() else (ROOT / a) for a in args]
    if not files:
        files = sorted(ROOT.glob("*.bat"))
    missing = [f for f in files if not f.exists()]
    for f in missing:
        print("MISSING: %s" % f)
    files = [f for f in files if f.exists()]

    print("bundle root: %s" % ROOT)
    print("-" * 96)
    ok_all, argvs = True, []
    for f in files:
        ok, argv = audit(f)
        ok_all &= ok
        argvs.append((f, argv))

    if want_diff and len(argvs) == 2:
        (fa, aa), (fb, ab_) = argvs
        ta, tb = tokenize(aa), tokenize(ab_)
        print("-" * 96)
        print("argv diff:  %s  ->  %s" % (fa.name, fb.name))
        print("  A: %s" % aa)
        print("  B: %s" % ab_)
        import difflib
        only_a, only_b = [], []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb).get_opcodes():
            if tag in ("delete", "replace"):
                only_a.extend(ta[i1:i2])
            if tag in ("insert", "replace"):
                only_b.extend(tb[j1:j2])
        print("  A only (%d): %s" % (len(only_a), " ".join(only_a)))
        print("  B only (%d): %s" % (len(only_b), " ".join(only_b)))

    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
