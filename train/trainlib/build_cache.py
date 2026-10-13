# -*- coding: utf-8 -*-
"""Build a static lookup-decoding n-gram cache for llama-server from the
code training library (trainlib).

Why this file exists
--------------------
common/ngram-cache.cpp is a NON-AUTOREGRESSIVE candidate generator: the static
cache is looked up with exactly LLAMA_NGRAM_STATIC = 2 tokens and can hand back
a drafted token with no model forward at all. A static draft is only accepted
when the stored continuation distribution is concentrated:

    sum(count) >= 2      and      max(count) >= 66% * sum(count)

So the useful payload is bigrams only, and (for the aggressive build) only the
single most likely continuation, which makes the 66% test pass by construction.
That is the "Jev-style" primitive here: the candidate set is precomputed, the
score is an integer count, and no sequential decode is involved.

IMPORTANT: the static cache is ALSO used as a validator. When the context or
dynamic cache proposes a token, a token present in the static continuation set
of the current bigram gets its count multiplied by 100. So breadth has value
beyond the static path itself.

File format (common/ngram-cache.h, version 2, all integers little endian,
native sizes):

    magic        char[4]  "LNGC"
    version      u32      2
    n_vocab      u32
    vocab_hash   u64      FNV-1a over every token text + a NUL separator
    n_ngrams     u64
    payload_size u64
    payload_hash u64      FNV-1a of the payload bytes
    payload      n_ngrams entries of
                   ngram    i32[4]  padded with LLAMA_TOKEN_NULL (-1)
                   n_tokens i32     number of following (token, count) pairs
                   token i32, count i32   x n_tokens

Usage
    python build_cache.py --name precise --min-share 0.60 --min-count 4
    python build_cache.py --name wide    --min-share 0.00 --min-count 2
"""
import argparse
import collections
import json
import os
import struct
import sys
import time

VOCAB_DIR = r"F:\super\qwen38mtp\vocab"
LIB = r"F:\super\qwen38mtp\trainlib"
OUT_DIR = os.path.join(LIB, "lookup")

sys.path.insert(0, VOCAB_DIR)
from tok import OfflineBpe            # noqa: E402  exact qwen35 pre-tokenizer + BPE

MAGIC = b"LNGC"
VERSION = 2
LLAMA_NGRAM_MAX = 4
LLAMA_NGRAM_STATIC = 2
LLAMA_TOKEN_NULL = -1
FNV_INIT = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1

SKIP_DIRS = {"_tools", "lookup", "__pycache__"}


def fnv1a(h, data):
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


def vocab_identity(bpe):
    """Reproduce common_ngram_cache_get_vocab_id() byte for byte."""
    h = FNV_INIT
    for t in bpe.tokens:
        h = fnv1a(h, t.encode("utf-8", errors="surrogateescape"))
        h = fnv1a(h, b"\x00")
    return (len(bpe.tokens), h if h != 0 else 1)


def lib_files(lib):
    for cat in sorted(os.listdir(lib)):
        d = os.path.join(lib, cat)
        if not os.path.isdir(d) or cat in SKIP_DIRS or cat.startswith("."):
            continue
        for dp, dns, fns in os.walk(d):
            dns[:] = [x for x in dns if x not in SKIP_DIRS and not x.startswith(".")]
            for fn in sorted(fns):
                yield cat, os.path.join(dp, fn)


def count_bigrams(bpe, lib, limit_files=0, verbose=True):
    """(token_i, token_i+1) -> Counter(next token). Never joins two files."""
    flat = {}
    NV = bpe.n_vocab
    n_files = n_tokens = 0
    t0 = time.time()
    for cat, path in lib_files(lib):
        if limit_files and n_files >= limit_files:
            break
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        ids = bpe.encode(text)
        n_files += 1
        n_tokens += len(ids)
        for i in range(len(ids) - 2):
            k = (ids[i] * NV + ids[i + 1]) * NV + ids[i + 2]
            flat[k] = flat.get(k, 0) + 1
        if verbose and n_files % 200 == 0:
            print("  ... %d files, %d tokens, %d triples, %.0fs"
                  % (n_files, n_tokens, len(flat), time.time() - t0))
    print("  counted %d files / %d tokens / %d unique triples in %.0fs"
          % (n_files, n_tokens, len(flat), time.time() - t0))
    return flat, n_files, n_tokens


def aggregate(flat, n_vocab, topk):
    """bigram key -> (total, [(count, token)] sorted desc, capped at topk)."""
    per = {}
    for k, c in flat.items():
        bkey, cont = divmod(k, n_vocab)
        lst = per.get(bkey)
        if lst is None:
            per[bkey] = [c, c, cont]
        else:
            lst[0] += c
            # keep a short sorted-by-count tail
            lst.append(c)
            lst.append(cont)
            # rebuild only when it grew past 2*topk entries
            if len(lst) > 2 + 2 * topk * 2:
                pairs = sorted(zip(lst[1::2], lst[2::2]), reverse=True)[:topk]
                lst = [lst[0]] + [x for p in pairs for x in p]
                per[bkey] = lst
    return per


def select(per, min_count, min_share, topk, max_ngrams, n_vocab):
    """Apply the engine's own acceptance rule and keep the biggest survivors."""
    rows, n_natural = [], 0
    for bkey, lst in per.items():
        total = lst[0]
        if total < min_count:
            continue
        pairs = sorted(zip(lst[1::2], lst[2::2]), reverse=True)
        top_c, top_t = pairs[0]
        natural_ok = (100 * top_c >= 66 * total)
        if natural_ok:
            n_natural += 1
        if min_share > 0 and 100 * top_c < min_share * 100 * total:
            continue
        keep = pairs[:topk]
        rows.append((total, bkey, keep))
    rows.sort(reverse=True)                 # most frequent bigrams first
    if max_ngrams and len(rows) > max_ngrams:
        rows = rows[:max_ngrams]
    return rows, n_natural


def write_cache(path, rows, n_vocab, vocab_hash, header=False):
    """Write the cache. Default is the HEADERLESS layout, on purpose.

    A v2 header carries a vocab fingerprint (count + FNV-1a of every token text)
    and the loader REFUSES a file whose fingerprint differs -- but a mismatch is
    only a warning, the server still starts, and the table is silently ignored.
    Measured 2026-09-22: our fingerprint did not match the runtime's even though
    both sides count 248320 tokens, and the whole first A/B therefore ran with an
    EMPTY table (log: "static lookup cache ignored ... same size, different token
    texts"). Not worth reverse-engineering: the headerless layout is an older but
    fully supported format (ngram-cache.cpp:397-416) and it still validates every
    token id against the real vocab size, so a table built for another tokenizer
    cannot load nonsensically. The identity we would have written is kept in the
    sidecar .json instead.
    """
    payload = bytearray()
    for _total, bkey, keep in rows:
        b0, b1 = divmod(bkey, n_vocab)
        payload += struct.pack("<4i", b0, b1, LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL)
        payload += struct.pack("<i", len(keep))
        for c, t in keep:
            payload += struct.pack("<2i", t, c)
    if header:
        head = MAGIC + struct.pack("<IQQQQQ", VERSION, n_vocab, vocab_hash,
                                   len(rows), len(payload),
                                   fnv1a(FNV_INIT, bytes(payload)))
    else:
        head = b""
        # a headerless file must not START with the magic, or the loader would
        # read it as a v2 header and fail. Token ids are < 2^18, so the first
        # 4 bytes are 0x00..0x03 in byte 3 -- "LNGC" is unreachable.
        assert bytes(payload[:4]) != MAGIC, "headerless payload starts with the magic"
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(head)
        f.write(payload)
    os.replace(tmp, path)
    return len(payload) + len(head)


# Two builds of the same counts, so one encoding pass can feed both sides of an A/B:
#
#   precise  only bigrams whose top continuation already holds >=60% of the mass.
#            Small, high precision, and every entry is one the engine accepts on
#            its own terms.
#   wide     every bigram seen at least twice, forced down to its single most
#            likely continuation. The 66% test then passes by construction, so
#            the static path fires on almost any context -- more drafts, lower
#            precision. This is the aggressive end of the trade-off.
VARIANTS = {
    "precise": {"min_count": 4, "min_share": 0.60, "topk": 1, "max_ngrams": 0},
    "wide":    {"min_count": 2, "min_share": 0.00, "topk": 1, "max_ngrams": 500000},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None,
                    help="single custom build instead of --variants")
    ap.add_argument("--variants", default="precise,wide")
    ap.add_argument("--lib", default=LIB)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--min-count", type=int, default=4)
    ap.add_argument("--min-share", type=float, default=0.60)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--max-ngrams", type=int, default=0)
    ap.add_argument("--limit-files", type=int, default=0)
    args = ap.parse_args()

    if args.name:
        plan = [(args.name, {"min_count": args.min_count, "min_share": args.min_share,
                             "topk": args.topk, "max_ngrams": args.max_ngrams})]
    else:
        plan = [(n.strip(), VARIANTS[n.strip()])
                for n in args.variants.split(",") if n.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    print("loading tokenizer (offline, exact pre-tokenizer) ...")
    bpe = OfflineBpe(args.gguf)
    print("  n_vocab=%d merges=%d" % (bpe.n_vocab, len(bpe.merges)))

    flat, n_files, n_tokens = count_bigrams(bpe, args.lib, args.limit_files)
    if not flat:
        print("no n-grams; nothing to do")
        return 1

    topk_max = max(v["topk"] for _, v in plan)
    per = aggregate(flat, bpe.n_vocab, topk_max)
    del flat
    n_bigram_types = len(per)
    n_vocab, vocab_hash = vocab_identity(bpe)

    print("")
    print("bigram types seen : %d" % n_bigram_types)
    results = []
    for name, v in plan:
        rows, n_natural = select(per, v["min_count"], v["min_share"], v["topk"],
                                 v["max_ngrams"], bpe.n_vocab)
        path = os.path.join(args.out_dir, "%s.lngc" % name)
        nbytes = write_cache(path, rows, n_vocab, vocab_hash, header=False)
        meta = {
            "name": name, "file": path, "bytes": nbytes,
            "layout": "headerless (older format): the loader skips the vocab "
                      "fingerprint check, see write_cache() for why",
            "n_ngrams": len(rows), "n_vocab": n_vocab, "vocab_hash": vocab_hash,
            "ngram_order": LLAMA_NGRAM_STATIC,
            "min_count": v["min_count"], "min_share": v["min_share"],
            "topk": v["topk"], "max_ngrams": v["max_ngrams"],
            "corpus_files": n_files, "corpus_tokens": n_tokens,
            "bigram_types": n_bigram_types,
            "bigram_types_naturally_dominant": n_natural,
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(args.out_dir, "%s.json" % name), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        results.append((name, len(rows), nbytes, n_natural))
        print("  %-8s min_count=%-2d share=%.2f -> %7d n-grams  %6.2f MB  (%.1f%% of types fire)"
              % (name, v["min_count"], v["min_share"], len(rows),
                 nbytes / 1048576.0, 100.0 * n_natural / max(1, n_bigram_types)))
    print("-> %s" % args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
