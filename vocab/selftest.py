"""selftest.py -- checks that need neither a GPU nor a running server.

  python selftest.py                 # code + tokenizer + map-format checks
  python selftest.py <map.vocab.txt> # additionally validate a produced map file

The map-file validator is a faithful re-implementation of the checks in
``src/llama-mtp-vocab.h: llama_mtp_vocab_read``. That function throws on bad input,
so a bad file makes the server refuse to start -- which is a *loud* failure and
exactly why it is worth checking here instead of discovering it at launch:

  magic == "llama-mtp-vocab-v1"
  0 < n_vocab <= (1 << 24)          and  n_vocab == the model's vocabulary size
  0 < count <= max_size            where the server passes max_size = n_vocab
  every id in [0, n_vocab), no duplicates
  no trailing tokens after the list
"""

from __future__ import annotations

import sys
from pathlib import Path

VOCAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VOCAB_DIR))

import prompts                                    # noqa: E402
import shortlist                                  # noqa: E402
import tok                                        # noqa: E402

MAGIC = "llama-mtp-vocab-v1"


def check_map_file(path, n_vocab_expected: int | None = None):
    """Return (ok, messages). Mirrors llama_mtp_vocab_read()'s accept/reject rules."""
    msgs, ok = [], True

    def fail(m):
        nonlocal ok
        ok = False
        msgs.append("FAIL " + m)

    raw = Path(path).read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        fail("file starts with a UTF-8 BOM (the reader's >> magic would see garbage)")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as e:
        return False, ["FAIL not ASCII: %s" % e]

    toks = text.split()
    if len(toks) < 3:
        return False, ["FAIL too short to hold a header"]
    magic, n_vocab_s, count_s = toks[0], toks[1], toks[2]
    if magic != MAGIC:
        fail("magic is %r, expected %r" % (magic, MAGIC))
    if not n_vocab_s.lstrip("-").isdigit() or not count_s.lstrip("-").isdigit():
        return False, ["FAIL header numbers are not integers"]
    n_vocab, count = int(n_vocab_s), int(count_s)

    if not (0 < n_vocab <= (1 << 24)):
        fail("n_vocab %d outside (0, 1<<24]" % n_vocab)
    if n_vocab_expected is not None and n_vocab != n_vocab_expected:
        fail("n_vocab %d != model vocabulary %d (server would refuse)" % (n_vocab, n_vocab_expected))
    body = toks[3:]
    if count <= 0:
        fail("count %d <= 0" % count)
    if count > n_vocab:
        fail("count %d > n_vocab %d" % (count, n_vocab))
    if len(body) != count:
        fail("body has %d ids, header declares %d (trailing data would be rejected)"
             % (len(body), count))

    seen = set()
    dup = None
    for i, s in enumerate(body):
        if not (s.isdigit() or (s.startswith("-") and s[1:].isdigit())):
            fail("id #%d is not an integer: %r" % (i, s))
            break
        v = int(s)
        if v < 0 or v >= n_vocab:
            fail("id #%d = %d out of range [0,%d)" % (i, v, n_vocab))
            break
        if v in seen:
            dup = v
            break
        seen.add(v)
    if dup is not None:
        fail("duplicate id %d" % dup)

    if ok:
        msgs.append("OK   magic/n_vocab/count/range/uniqueness/no-trailing all pass "
                    "(n_vocab=%d, count=%d)" % (n_vocab, count))
    return ok, msgs


def main() -> int:
    rc = 0

    print("== prompts ==")
    cat = prompts.catalog()
    labels = [c[0] for c in cat]
    if len(labels) != len(set(labels)):
        print("  FAIL duplicate labels")
        rc = 1
    else:
        per = {}
        for _label, g, _p in cat:
            per[g] = per.get(g, 0) + 1
        print("  ok: %d prompts, unique labels, groups=%s" % (len(cat), sorted(per.items())))

    print("\n== gguf vocabulary ==")
    try:
        p2i, id_of, piece_of, n_vocab, tokens = shortlist.read_vocab_index(tok.DEFAULT_GGUF)
        print("  ok: n_vocab=%d  (the map header must carry exactly this number)" % n_vocab)
    except Exception as e:
        print("  FAIL could not read vocabulary: %s: %s" % (type(e).__name__, e))
        return 1

    print("\n== offline bpe ==")
    try:
        from tok import OfflineBpe
        bpe = OfflineBpe()
        print("  ggml.model=%s  pre=%s  merges=%d  specials=%s  exact_pre=%s"
              % (bpe.model_type, bpe.pre, len(bpe.merges),
                 "yes" if bpe._special_re else "no", bpe.exact_pre))
        for probe in ("def add(a, b):\n    return a + b\n",
                      "  return self.value\n",
                      "for i in range(10):\n    print(i)\n",
                      "#!/bin/bash\nset -euo pipefail\n"):
            ids = bpe.encode(probe)
            back = "".join(bpe.piece(i) for i in ids)
            ok = back == probe
            print("  %-6s %2d ids  roundtrip=%s  %r"
                  % ("ok" if ok else "DIFF", len(ids), "yes" if ok else "NO", probe[:34].replace("\n", "\\n")))
            if not ok:
                print("        pieces: %r" % [bpe.piece(i) for i in ids][:14])
                rc = 1

        # Pre-tokenizer behaviour. Round-tripping alone cannot prove the split is
        # right (any split re-encodes to the same bytes); these expectations come
        # from reading the alternation in llama-vocab.cpp by hand.
        print("  -- split checks (qwen35 alternation) --")
        for text, want in (
            # \p{N} is a SINGLE digit, so "10" is two blocks
            ("range(10)",           ["range", "(", "1", "0", ")"]),
            # rule 2's optional leading char is any non-letter/non-digit, so "(" or
            # ")" plus a letter run sticks together: "(a" is one block. Punctuation
            # runs with no letter after them fall to rule 4 and clump: "):" is one.
            ("def f(a, b):",        ["def", " f", "(a", ",", " b", "):"]),
            # a newline and an indent run never merge across the boundary
            ("\n    return x",      ["\n", "    ", "return", " x"]),
            # " ?[...]+" takes at most one leading space, so " =" clumps but the
            # trailing space does not join it
            ("a = 1\n",             ["a", " =", " ", "1", "\n"]),
            ("x += 10  # note\n",   ["x", " +=", " ", "1", "0", "  ", "#", " note", "\n"]),
        ):
            got = bpe._scanner(text)
            good = got == want
            print("  %-6s %-20r -> %r" % ("ok" if good else "FAIL", text, got))
            if not good:
                print("         expected %r" % (want,))
                rc = 1
    except Exception as e:
        print("  FAIL offline bpe: %s: %s" % (type(e).__name__, e))
        rc = 1

    print("\n== forced tokens resolve ==")
    miss = [s for s in shortlist.FORCED if id_of(s) is None]
    print("  %d/%d resolved; missing=%s"
          % (len(shortlist.FORCED) - len(miss), len(shortlist.FORCED), miss[:10] or "none"))

    print("\n== map writer / validator ==")
    tmp = VOCAB_DIR / "out" / "_selftest.vocab.txt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cand = [p2i[t] for t in tokens[:6]] + [i for i in (id_of("def"), id_of(" "), id_of("\n"))
                                           if i is not None]
    table = list(dict.fromkeys(cand))[:10]        # dedupe: the validator rejects duplicates
    shortlist.write_map(tmp, table, n_vocab)
    ok, msgs = check_map_file(tmp, n_vocab)
    for m in msgs:
        print("  " + m)
    if not ok:
        rc = 1
    # a deliberately wrong n_vocab must be rejected -- proving the validator bites
    bad = tmp.parent / "_selftest_bad.vocab.txt"
    shortlist.write_map(bad, table, n_vocab + 1)
    ok_bad, msgs_bad = check_map_file(bad, n_vocab)
    print("  negative control (wrong n_vocab): %s" % ("rejected as expected" if not ok_bad else "NOT REJECTED"))
    if ok_bad:
        rc = 1
    tmp.unlink(missing_ok=True)
    bad.unlink(missing_ok=True)

    for extra in sys.argv[1:]:
        print("\n== validate %s ==" % extra)
        ok, msgs = check_map_file(extra, n_vocab)
        for m in msgs:
            print("  " + m)
        if not ok:
            rc = 1

    print("\nselftest exit=%d" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
