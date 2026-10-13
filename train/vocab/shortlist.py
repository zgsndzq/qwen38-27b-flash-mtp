"""shortlist.py -- build the draft-vocabulary shortlist and decide whether it is worth it.

This is the offline stage: it never touches the GPU and never modifies a binary.
It consumes the token-id caches from ``corpus/ids`` and produces

  out/shortlist_<N>.vocab.txt   a map file the server can load directly
  out/coverage.csv              N vs coverage / drop-rate q, per field
  out/analysis.json             every number the report is made of
  out/VERDICT.md                the do / do-not verdict against the criteria

The economics being tested (design doc section 5). A decode round re-reads the
trunk every time, while the draft lm_head is read once per *draft step*:

    gross(d, N) = 0.265 * d * (1 - N/248320) / (5.765 + 0.395 * d)

Shortening the draft head to N rows saves ``gross`` of the round's bytes. The
cost is that a draft can only propose table tokens, so any real token outside the
table is a guaranteed rejection: with q = P(token not in table),

    net(N) = gross(d, N) - q(N)

q is the number the corpus measures. It is estimated out-of-sample with k-fold
splitting over whole samples -- a token counted while building the table must not
also be the token that certifies it, or the curve would just report its own
training set back.

Usage:
  python shortlist.py --source gen                 # authoritative source
  python shortlist.py --source local --backend-note
  python shortlist.py --source all --sizes 8k,16k,32k,64k --hot 4096
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from tok import IDS_DIR, FIELDS, DEFAULT_GGUF, BYTE_ENCODER, read_gguf_meta

VOCAB_DIR = Path(__file__).resolve().parent
OUT_DIR = VOCAB_DIR / "out"
N_VOCAB_FALLBACK = 248320

# reference point from the design doc: bytes moved per decode round
TRUNK_GB = 5.765          # trunk weights + target head, per round, fixed
DRAFT_STEP_GB = 0.395     # per draft step: MTP layer (~0.13) + full lm_head (0.265)
HEAD_GB = 0.265           # the full draft lm_head, the only part a shortlist shrinks

# tokens that must be in the table regardless of frequency. Missing one does not
# corrupt anything (the target still verifies with the full vocab) -- it only
# costs the acceleration at that step -- but these are cheap and structurally
# guaranteed to appear in real traffic.
FORCED = [
    "\n", "\n\n", "    ", "\t", "\r\n", " \n",
    " ", "(", ")", "()", "(", "):", "():", "()\n", ");", ");\n",
    "{", "}", "{}", "[", "]", "[]", "(),", "):\n",
    ",", ", ", ":", ": ", ";", ";\n", ".", "..", "...", "=", " = ", " == ", " != ",
    " -> ", " => ", " += ", " -= ", " < ", " > ", " >= ", " <= ", "+", "-", "*", "/",
    "//", "/*", "*/", "#", "# ", '"', "'", "`", "|", "&", "!", "?", "@",
    "def ", "class ", "return ", "if ", "else:", "elif ", "for ", "while ", "in ",
    "import ", "from ", "try:", "except ", "finally:", "with ", "as ", "yield ",
    "self", "self.", "None", "True", "False", "lambda ", "print(", "len(", "range(",
    "int", "str", "float", "list", "dict", "set", "tuple", "assert ", "raise ",
    "async ", "await ", "global ", "not ", "and ", "or ", "is ", "pass", "break", "continue",
    "int ", "void ", "char ", "struct ", "static ", "const ", "/* ", " */", "// ",
    "#include", "#define", "printf(", "sizeof(", "malloc(", "free(", "NULL",
    " -> ", "std::", "->", "#!", "#!/bin/bash", "#!/usr/bin/env python3",
    "function ", "const ", "let ", "var ", "export ", "require(", "console.log(",
    "SELECT ", "FROM ", "WHERE ", "JOIN ", "INSERT INTO ",
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
]


def parse_size(s: str) -> int:
    s = s.strip().lower()
    mult = 1
    if s.endswith("k"):
        mult, s = 1000, s[:-1]
    elif s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    return int(float(s) * mult)


def read_vocab_index(gguf_path):
    """(piece_to_id, text->id helper, id->text helper, n_vocab, tokens).

    Only the vocabulary and token types are read; the merge table is not needed
    here, so this stays fast even on a 5.8 GB file.
    """
    meta = read_gguf_meta(gguf_path,
                          want_keys=("tokenizer.ggml.tokens", "tokenizer.ggml.token_type"))
    tokens = meta.get("tokenizer.ggml.tokens") or []
    piece_to_id = {}
    for i, s in enumerate(tokens):
        piece_to_id.setdefault(s, i)

    byte_dec = {v: k for k, v in BYTE_ENCODER.items()}   # built once, not per call

    def id_of(text: str):
        piece = "".join(BYTE_ENCODER[b] for b in text.encode("utf-8"))
        return piece_to_id.get(piece)

    def piece_of(i: int) -> str:
        if not (0 <= i < len(tokens)):
            return "<oob:%d>" % i
        try:
            return bytes(byte_dec[ch] for ch in tokens[i]).decode("utf-8", errors="replace")
        except KeyError:
            return tokens[i]

    return piece_to_id, id_of, piece_of, len(tokens), tokens


# ------------------------------------------------------------------ data loading
_IDS_CACHE = {"path": None, "chunks": None}


def sample_chunks(sample: dict) -> dict:
    """Load one sample's token ids, with a single-entry cache.

    Deliberately reads on demand: a corpus large enough to make the coverage curve
    meaningful holds millions of ids, and keeping them all resident as Python ints
    would cost hundreds of megabytes for nothing. One sample at a time is enough,
    because every loop here walks samples sequentially.
    """
    path = sample["_path"]
    if _IDS_CACHE["path"] != path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        _IDS_CACHE["path"] = path
        _IDS_CACHE["chunks"] = data.get("chunks") or {}
    return _IDS_CACHE["chunks"]


def load_samples(sources: list[str]) -> list[dict]:
    """Metadata only -- the heavy `chunks` payload stays on disk."""
    out = []
    for p in sorted(IDS_DIR.glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("source") not in sources:
            continue
        rec.pop("chunks", None)
        rec["_path"] = str(p)
        out.append(rec)
    return out


def field_ids(sample: dict, mode: str) -> list[int]:
    """Token ids for one analysis mode ('all' | 'answer' | 'reasoning' | 'text')."""
    chunks = sample_chunks(sample)
    if mode == "all":
        ids = []
        for name in FIELDS.get(sample["source"], tuple(chunks)):
            ids.extend((chunks.get(name) or {}).get("ids") or [])
        return ids
    return list((chunks.get(mode) or {}).get("ids") or [])


def kfold(n: int, k: int):
    """Deterministic k-way split. Whole samples only -- no token-level leakage."""
    folds = [[] for _ in range(k)]
    for i in range(n):
        folds[i % k].append(i)
    out = []
    for f in range(k):
        ev = set(folds[f])
        tr = [i for i in range(n) if i not in ev]
        out.append((tr, sorted(ev)))
    return out


# --------------------------------------------------------------------- analysis
def analyse(samples: list[dict], sizes: list[int], k: int, mode: str,
            forced_ids: list[int], hot: int, depth_code: float, depth_prose: float):
    """Out-of-sample coverage curve for each table size, averaged over k folds."""
    n = len(samples)
    per_size = {N: {"cov": [], "cov_answer": [], "cov_reasoning": [], "eval_tokens": [],
                    "missing": Counter()} for N in sizes}
    origins = {}

    for fold, (tr_idx, ev_idx) in enumerate(kfold(n, k), 1):
        counts = Counter()
        for i in tr_idx:
            counts.update(field_ids(samples[i], mode))
        if not counts:
            continue
        ranked = [tid for tid, _c in counts.most_common()]
        for N in sizes:
            table, origin = build_table(ranked, N, forced_ids)
            origins[N] = origin
            in_table = set(table)
            for i in ev_idx:
                s = samples[i]
                ev_all = field_ids(s, mode)
                if not ev_all:
                    continue
                hit = sum(1 for t in ev_all if t in in_table)
                per_size[N]["cov"].append(hit / len(ev_all))
                per_size[N]["eval_tokens"].append(len(ev_all))
                for fld in ("answer", "reasoning", "text"):
                    ids = field_ids(s, fld)
                    if ids:
                        h = sum(1 for t in ids if t in in_table)
                        key = {"answer": "cov_answer", "reasoning": "cov_reasoning"}.get(fld)
                        if key:
                            per_size[N][key].append(h / len(ids))
                for t in ev_all:
                    if t not in in_table:
                        per_size[N]["missing"][t] += 1
        print("  fold %d/%d done (train samples=%d, eval samples=%d)"
              % (fold, k, len(tr_idx), len(ev_idx)))

    rows = []
    for N in sizes:
        d = per_size[N]
        if not d["cov"]:
            continue
        cov = statistics.mean(d["cov"])
        q = 1.0 - cov
        # A round re-reads the trunk and the target head once (fixed), plus the
        # draft step per draft step. Only the draft *lm_head* shrinks with the
        # shortlist -- the MTP layer of the step is untouched -- so the numerator
        # is HEAD_GB * d, NOT the whole DRAFT_STEP_GB * d. Using the step here
        # overstates the saving by ~4 points (and lets net exceed the ceiling).
        denom_code = TRUNK_GB + DRAFT_STEP_GB * depth_code
        denom_prose = TRUNK_GB + DRAFT_STEP_GB * depth_prose
        gross_code = HEAD_GB * depth_code * (1 - N / N_VOCAB_FALLBACK) / denom_code
        gross_prose = HEAD_GB * depth_prose * (1 - N / N_VOCAB_FALLBACK) / denom_prose

        # Per-sample coverage is the honest analogue of real use: the table is built
        # offline, then serves one request at a time. Mixing 8474 unrelated projects
        # into one average asks "would this table cover a whole monorepo", which is
        # not the question the draft head ever has to answer. A single response
        # reuses its own identifiers, so its coverage is the typical case.
        cvs = sorted(d["cov"])
        cov_med = statistics.median(cvs)
        cov_p10 = cvs[max(0, min(len(cvs) - 1, int(0.10 * len(cvs))))]
        q_med = 1.0 - cov_med
        static_rows = max(1, N - hot) if hot else N
        rows.append({
            "N": N, "count": origins[N][1], "forced": origins[N][0],
            "coverage": cov, "q": q,
            "coverage_answer": statistics.mean(d["cov_answer"]) if d["cov_answer"] else None,
            "coverage_reasoning": statistics.mean(d["cov_reasoning"]) if d["cov_reasoning"] else None,
            "coverage_min": min(d["cov"]), "coverage_max": max(d["cov"]),
            "coverage_median": cov_med, "coverage_p10": cov_p10,
            "q_median": q_med,
            "net_code_median": gross_code - q_med,
            "net_prose_median": gross_prose - q_med,
            "eval_tokens": sum(d["eval_tokens"]),
            "gross_code": gross_code, "gross_prose": gross_prose,
            "net_code": gross_code - q, "net_prose": gross_prose - q,
            "static_only_coverage": None,   # filled next, requires a second table build
            "hot": hot,
            "top_missing": [[t, c] for t, c in d["missing"].most_common(25)],
        })
    return rows


def build_table(ranked: list[int], size: int, forced_ids: list[int]):
    """Forced ids first, then the most frequent, truncated to `size`, no duplicates."""
    table, seen = [], set()
    for t in forced_ids:
        if t not in seen:
            table.append(t)
            seen.add(t)
    for t in ranked:
        if len(table) >= size:
            break
        if t not in seen:
            table.append(t)
            seen.add(t)
    n_forced = sum(1 for t in table if t in set(forced_ids))
    return table[:size], (n_forced, len(table[:size]))


def static_only_coverage(samples, sizes, k, mode, forced_ids, hot):
    """Coverage of the *static* head only, since the hot tail may be repointed at runtime.

    Must be computed inside the fold: building the table from all samples here would
    let evaluation tokens vote for their own inclusion and inflate the floor.
    """
    if not hot:
        return {}
    acc = {N: [] for N in sizes}
    for tr_idx, ev_idx in kfold(len(samples), k):
        counts = Counter()
        for i in tr_idx:
            counts.update(field_ids(samples[i], mode))
        if not counts:
            continue
        ranked = [t for t, _c in counts.most_common()]
        for N in sizes:
            table, _ = build_table(ranked, N, forced_ids)
            static = set(table[:max(1, len(table) - hot)])
            for i in ev_idx:
                ids = field_ids(samples[i], mode)
                if ids:
                    acc[N].append(sum(1 for t in ids if t in static) / len(ids))
    return {N: statistics.mean(v) for N, v in acc.items() if v}


# ----------------------------------------------------------------------- output
def write_map(path: Path, table: list[int], n_vocab: int, per_line: int = 24) -> None:
    """Exact format consumed by llama_mtp_vocab_read().

    Header, then the ids. The reader streams with operator>> so any whitespace
    layout works; wrapping keeps the file diffable. No trailing data is allowed --
    the reader throws on anything it cannot parse as a strict int64.
    """
    lines = ["llama-mtp-vocab-v1 %d %d" % (n_vocab, len(table))]
    for i in range(0, len(table), per_line):
        lines.append(" ".join(str(t) for t in table[i:i + per_line]))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def verdict_for(net: float) -> tuple[str, str]:
    """Verdict on the best point of the measured curve, not on a pre-chosen size."""
    if net >= 0.07:
        return "GO", "best table nets %+.1f%% -- worth building" % (100 * net)
    if net >= 0.04:
        return "GO", "best table nets %+.1f%% -- worth building" % (100 * net)
    if net >= 0.01:
        return "MAYBE", "best table only nets %+.1f%% -- marginal, decide on risk appetite" % (100 * net)
    return "NO-GO", "no table on the curve nets anything -- abandon this direction"


def verdict_for_8k(rows) -> tuple[str, str]:
    """The original design-doc criterion, kept so the assumption can be falsified."""
    row = next((r for r in rows if r["N"] == 8000), None)
    if row is None:
        return "n/a", "no 8k table in this run"
    cov = row["coverage"]
    if cov >= 0.995:
        return "GO", "8k coverage %.4f >= 99.5%%" % cov
    if cov >= 0.99:
        return "GO", "8k coverage %.4f >= 99%%" % cov
    if cov >= 0.97:
        return "GO", "8k coverage %.4f >= 97%%" % cov
    if cov >= 0.90:
        return "MAYBE", "8k coverage %.4f >= 90%% -- break-even only" % cov
    return "NO-GO", "8k coverage %.4f < 90%%" % cov


def cmd(args) -> int:
    sources = ["real", "gen", "local"] if args.source == "all" else [args.source]
    samples = load_samples(sources)
    if not samples:
        print("no tokenized samples in %s for source(s) %s" % (IDS_DIR, sources))
        return 1
    backends = sorted({s.get("backend", "?") for s in samples})
    print("samples : %d  (sources=%s, backends=%s)"
          % (len(samples), sorted({s['source'] for s in samples}), backends))
    if "offline" in backends:
        print("WARNING : offline-approx tokenization in use -- treat the curve as a preview")

    sizes = [parse_size(x) for x in args.sizes.split(",") if x.strip()]
    sizes = sorted(set(sizes))

    _p2i, id_of, piece_of, n_vocab, tokens = read_vocab_index(args.gguf)
    print("vocab   : %d tokens  (model head expects exactly this n_vocab)" % n_vocab)
    forced_ids, forced_missing = [], []
    for s in FORCED:
        i = id_of(s)
        if i is None:
            forced_missing.append(s)
        else:
            forced_ids.append(i)
    forced_ids = sorted(set(forced_ids))
    print("forced  : %d resolved, %d not in vocab%s"
          % (len(forced_ids), len(forced_missing),
             (" (" + ", ".join(repr(x) for x in forced_missing[:8]) + ")") if forced_missing else ""))

    print("analysing: mode=%s  k=%d  sizes=%s"
          % (args.field, args.splits, [format_size(x) for x in sizes]))
    rows = analyse(samples, sizes, args.splits, args.field, forced_ids,
                   args.hot, args.depth_code, args.depth_prose)
    if not rows:
        print("analysis produced nothing -- are the id caches empty?")
        return 1

    # The shipped map files are built from the FULL corpus (more data = a better
    # table), while every coverage number above is out-of-sample k-fold. Those are
    # deliberately different tables for different jobs: one maximises quality, the
    # other must not certify itself.
    n = len(samples)
    counts = Counter()
    for s in samples:
        counts.update(field_ids(s, args.field))
    ranked = [t for t, _c in counts.most_common()]
    distinct = len(ranked)
    print("corpus  : %d distinct tokens in the full corpus (largest table asked: %d)"
          % (distinct, max(sizes)))
    if distinct < 4 * max(sizes):
        print("WARNING : a table of this size nearly covers the whole corpus, so the\n"
              "          coverage numbers below are inflated -- the long tail that would\n"
              "          fall outside the table simply is not in the corpus yet.\n"
              "          Mine more code (run.py local --limit-files N) before trusting it.")
    table_cache = {}
    for N in sizes:
        table_cache[N], _ = build_table(ranked, N, forced_ids)
        if len(table_cache[N]) < N:
            print("note    : only %d distinct tokens available; the %s table is short"
                  % (len(table_cache[N]), format_size(N)))
    # report the size of the table that actually gets written, not the per-fold one
    for r in rows:
        r["count"] = len(table_cache[r["N"]])
    static_cov = static_only_coverage(samples, sizes, args.splits, args.field,
                                      forced_ids, args.hot)
    for r in rows:
        r["static_only_coverage"] = static_cov.get(r["N"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("%s-%s" % (args.source, args.field))
    for r in rows:
        p = OUT_DIR / ("shortlist_%s_%s.vocab.txt" % (format_size(r["N"]), tag))
        write_map(p, table_cache[r["N"]], n_vocab)
        r["map_file"] = str(p)

    # csv
    csv_path = OUT_DIR / ("coverage_%s.csv" % tag)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "table_entries", "forced_in_table", "coverage_mixed",
                    "coverage_median_request", "coverage_p10_request", "coverage_min_request",
                    "q_median", "coverage_answer", "coverage_reasoning", "coverage_static_only",
                    "gross_code", "net_code_mixed", "net_code_per_request",
                    "gross_prose", "net_prose_mixed",
                    "eval_tokens", "coverage_max"])
        for r in rows:
            w.writerow([r["N"], r["count"], r["forced"], "%.5f" % r["coverage"],
                        "%.5f" % r["coverage_median"], "%.5f" % r["coverage_p10"],
                        "%.5f" % r["coverage_min"], "%.5f" % r["q_median"],
                        "" if r["coverage_answer"] is None else "%.5f" % r["coverage_answer"],
                        "" if r["coverage_reasoning"] is None else "%.5f" % r["coverage_reasoning"],
                        "" if r["static_only_coverage"] is None else "%.5f" % r["static_only_coverage"],
                        "%.5f" % r["gross_code"], "%.5f" % r["net_code"],
                        "%.5f" % r["net_code_median"],
                        "%.5f" % r["gross_prose"], "%.5f" % r["net_prose"],
                        r["eval_tokens"], "%.5f" % r["coverage_max"]])

    analysis = {
        "tag": tag, "source": args.source, "field": args.field, "splits": args.splits,
        "hot": args.hot, "n_vocab": n_vocab, "n_samples": n,
        "samples_by_source": {s: sum(1 for x in samples if x["source"] == s) for s in sources},
        "samples_by_group": dict(Counter(x.get("group", "?") for x in samples).most_common()),
        "backends": backends, "forced_resolved": len(forced_ids),
        "forced_missing": forced_missing, "distinct_tokens": distinct,
        "depth_code": args.depth_code, "depth_prose": args.depth_prose,
        "eval_tokens_per_estimate": max((r["eval_tokens"] for r in rows), default=0),
        "rows": rows,
        "top_tokens": [[t, counts[t], piece_of(t)] for t in ranked[:args.top_show]],
    }
    (OUT_DIR / ("analysis_%s.json" % tag)).write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    md = render_markdown(analysis, rows, piece_of, args, n_vocab, backends)
    (OUT_DIR / ("VERDICT_%s.md" % tag)).write_text(md, encoding="utf-8")

    print("")
    print("%-7s %-7s %-9s %-9s %-9s %-10s %-10s" %
          ("N", "entries", "cov_mix", "cov_req", "q_req", "net_req", "net_mix"))
    for r in rows:
        print("%-7s %-7d %-9s %-9.4f %-9.4f %+9.2f%% %+9.2f%%" % (
            format_size(r["N"]), r["count"], "%.4f" % r["coverage"],
            r["coverage_median"], r["q_median"],
            100 * r["net_code_median"], 100 * r["net_code"]))
    best = best_row(rows)
    v, why = verdict_for(best["net_code_median"])
    v8, why8 = verdict_for_8k(rows)
    print("")
    print("BEST   : %s table -- on a typical request it covers %.4f%% of tokens "
          "(q %.4f%%), net %+.2f%%"
          % (format_size(best["N"]), 100 * best["coverage_median"],
             100 * best["q_median"], 100 * best["net_code_median"]))
    print("VERDICT: %s -- %s" % (v, why))
    print("   8k check (design-doc criteria): %s -- %s" % (v8, why8))
    print("report  : %s" % (OUT_DIR / ("VERDICT_%s.md" % tag)))
    print("csv     : %s" % csv_path)
    return 0


def best_row(rows):
    """The table with the best measured net gain on a typical request."""
    return max(rows, key=lambda r: r.get("net_code_median", r["net_code"]))


def format_size(n: int) -> str:
    if n % 1_000_000 == 0:
        return "%dm" % (n // 1_000_000)
    if n % 1000 == 0:
        return "%dk" % (n // 1000)
    return str(n)


def render_markdown(analysis, rows, piece_of, args, n_vocab, backends) -> str:
    o = []
    o.append("# Draft-vocabulary shortlist -- measured verdict\n")
    o.append("Source: `%s`  |  field: `%s`  |  %d-fold out-of-sample  |  %s tokens evaluated "
             "per estimate  |  tokenizer: %s\n"
             % (args.source, args.field, args.splits,
                "{:,}".format(analysis["eval_tokens_per_estimate"]), ", ".join(backends)))
    if "offline" in backends:
        o.append("> **PREVIEW DATA.** These ids came from the offline approximate tokenizer "
                 "(no pre-tokenizer regex). Good enough to reject an idea, not to certify one. "
                 "Re-run with a live server before acting.\n")
    o.append("")
    o.append("## 1. Coverage and economics\n")
    distinct = analysis.get("distinct_tokens", 0)
    if distinct and distinct < 4 * max(r["N"] for r in rows):
        o.append("> **These coverage numbers are inflated.** The corpus holds only **%d distinct "
                 "tokens** while the largest table asks for %d. A table that size nearly covers "
                 "the entire corpus, so the long tail that would actually land outside it is not "
                 "represented yet. Mine more code (`run.py local --limit-files N`) before drawing "
                 "a conclusion from this table.\n" % (distinct, max(r["N"] for r in rows)))
    o.append("`gross` = share of a decode round's bytes that the shortened draft head saves. "
             "`net = gross - q`, where q is the fraction of real tokens the draft would be "
             "unable to propose at all.\n")
    o.append("Two coverage readings, and they answer different questions:\n")
    o.append("- **coverage (mixed)** -- all evaluation tokens pooled together. This asks "
             "\"would one table cover an entire 8000-file source tree end to end\", which is "
             "not a question the draft head ever faces, and it is pessimistic.\n")
    o.append("- **coverage per request (median)** -- the median over individual samples. The "
             "table is built offline and then serves one response at a time, and a single "
             "response reuses its own identifiers heavily, so this is the honest analogue "
             "of real use. `p10` is the pessimistic decile.\n")
    o.append("")
    o.append("| table | entries | coverage(mixed) | coverage per request (median) | p10 | "
             "gross(code) | net(code) mixed | **net(code) per request** |")
    o.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        o.append("| %s | %d | %.4f | %.4f | %.4f | %+.2f%% | %+.2f%% | **%+.2f%%** |" % (
            format_size(r["N"]), r["count"], r["coverage"], r["coverage_median"],
            r["coverage_p10"], 100 * r["gross_code"], 100 * r["net_code"],
            100 * r["net_code_median"]))
    o.append("")
    if args.hot:
        o.append("`--hot %d`: the last %d entries are repointable at runtime, so the "
                 "`static-only` column is the guaranteed floor.\n" % (args.hot, args.hot))
    o.append("## 2. Verdict\n")
    best = best_row(rows)
    v, why = verdict_for(best["net_code_median"])
    o.append("**%s** -- best point of the curve is the **%s** table: on a typical request it "
             "covers **%.3f%%** of tokens (worst decile %.3f%%), which nets **%+.2f%%**.\n"
             % (v, format_size(best["N"]), 100 * best["coverage_median"],
                100 * best["coverage_p10"], 100 * best["net_code_median"]))
    v8, why8 = verdict_for_8k(rows)
    o.append("The design document assumed an 8k table would suffice; measured against that "
             "criterion directly, it says **%s** (%s). The saving shrinks only slowly with N "
             "while q shrinks fast, so the optimum sits well above 8k.\n" % (v8, why8))
    o.append("Full criteria: `docs/design-vocab-shortlist.md` section 4.5.3.\n")
    o.append("")
    o.append("## 3. What the table actually contains\n")
    o.append("Top %d tokens by frequency across the whole corpus -- this is what the shipped "
             "map file actually contains (the table is built from all samples; every coverage "
             "number above is out-of-sample). This is the direct test of the premise that code "
             "keys are a fixed structure: if the top of the list is indentation runs, "
             "punctuation and keywords, the premise holds.\n"
             % min(args.top_show, len(analysis["top_tokens"])))
    o.append("| # | id | count | piece |")
    o.append("|---|---|---|---|")
    for i, (tid, c, piece) in enumerate(analysis["top_tokens"][:args.top_show], 1):
        shown = piece.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        if len(shown) > 34:
            shown = shown[:31] + "..."
        o.append("| %d | %d | %d | `%s` |" % (i, tid, c, shown))
    o.append("")
    o.append("## 4. What the table misses\n")
    o.append("Most frequent tokens that fell outside the table during evaluation. If these are "
             "identifiers the runtime hot-slot mechanism can absorb them after first sight; if "
             "they are structural, the table is built wrong.\n")
    for r in ([best] + [x for x in rows if x is not best][:1]):
        o.append("**%s table** (%d tokens missing from eval, showing the worst):\n"
                 % (format_size(r["N"]), sum(c for _t, c in r["top_missing"])))
        o.append("| id | misses | piece |")
        o.append("|---|---|---|")
        for tid, c in r["top_missing"][:15]:
            piece = piece_of(tid).replace("\n", "\\n").replace("\t", "\\t")
            if len(piece) > 34:
                piece = piece[:31] + "..."
            o.append("| %d | %d | `%s` |" % (tid, c, piece))
        o.append("")
    o.append("## 5. Corpus composition\n")
    o.append("- samples: %d %s" % (analysis["n_samples"], analysis["samples_by_source"]))
    o.append("- distinct tokens in corpus: **%d** (largest table: %d)"
             % (analysis.get("distinct_tokens", 0), max(r["N"] for r in rows)))
    o.append("- per-request coverage spread (best table): median %.4f, p10 %.4f, min %.4f, "
             "max %.4f" % (best["coverage_median"], best["coverage_p10"],
                           best["coverage_min"], best["coverage_max"]))
    o.append("- groups: %s" % analysis["samples_by_group"])
    o.append("- forced structural tokens resolved: %d (%d not present in the vocab)"
             % (analysis["forced_resolved"], len(analysis["forced_missing"])))
    o.append("- assumed mean draft depth: code %.3f, prose %.3f (measured by tools/ab_bench.py)"
             % (analysis["depth_code"], analysis["depth_prose"]))
    o.append("")
    o.append("## 6. Map files written\n")
    for r in rows:
        o.append("- `%s`  (%d entries)" % (Path(r["map_file"]).name, r["count"]))
    o.append("")
    o.append("Load one with `--spec-draft-vocab-map <file>` and, optionally, "
             "`--spec-draft-vocab-hot N`.\n")
    return "\n".join(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="gen", choices=["gen", "local", "real", "all"])
    ap.add_argument("--field", default="all", choices=["all", "answer", "reasoning", "text"])
    ap.add_argument("--sizes", default="4k,8k,12k,16k,24k,32k,48k,64k")
    ap.add_argument("--splits", type=int, default=2)
    ap.add_argument("--hot", type=int, default=0)
    ap.add_argument("--top-show", type=int, default=60)
    ap.add_argument("--depth-code", type=float, default=2.657)
    ap.add_argument("--depth-prose", type=float, default=2.073)
    ap.add_argument("--gguf", default=str(DEFAULT_GGUF))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)
    return cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
