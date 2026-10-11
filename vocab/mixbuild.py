"""mixbuild.py -- build a shortlist from a *mixture* of corpora, and price the mix.

Why this exists. `shortlist.py` ranks tokens by raw frequency over whatever
corpus it is given, so the corpus with the most tokens wins outright. That is
exactly how the shipped table ended up as a code table: `local` contributes 24.2M
tokens against 67k for the served traffic, a ratio of 360:1, so the served
traffic could not influence a single row.

But the table length is a fixed budget and the objective is not "cover the biggest
corpus" -- it is "minimise dropped drafts on the traffic we actually serve". Those
differ whenever the corpora are not proportional to the traffic. So this tool lets
each pool contribute by *share* instead of by *size*:

    score(t) = SUM_p  w_p * (count_p(t) / tokens_p)

i.e. every pool is normalised before mixing, so `w_p` is literally "how much of the
table budget this pool competes for". Setting w_talk = 0 reproduces the shipped
code-only behaviour; raising it buys Chinese and prose rows at the cost of code
rows, and *how much that costs* is what the report measures.

Honesty rules
  * A held-out slice of every pool is never used to build the table, so the
    reported coverage is out-of-sample. With only 20 real samples, in-sample
    numbers would be a fantasy.
  * The headline metric is per-sample NET on the held-out *served traffic*, in the
    design doc's units:  net = gross(d, N) - q.
  * Covering a corpus and covering the traffic are different things, so the report
    prints both the token hit rate and the median per-sample q.

Usage
  python mixbuild.py --sweep                                  # the whole curve
  python mixbuild.py --sweep --size 32000
  python mixbuild.py --weights real=3,talk=2,code=2 --tag mix --out out/shortlist_24k_mix.vocab.txt
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

VOCAB_DIR = Path(__file__).resolve().parent
BUNDLE = VOCAB_DIR.parent
IDS_DIR = VOCAB_DIR / "corpus" / "ids"

N_VOCAB = 248320
TRUNK_GB = 5.765
DRAFT_STEP_GB = 0.395
HEAD_GB = 0.265
DEPTH = {"code": 2.657, "prose": 2.073}
PROSE_GROUPS = {"chat", "real", "zh", "zhcode", "zhthink"}

# Which pool a tokenized sample belongs to. Pools exist to be weighted against each
# other, so they follow the *kind* of traffic rather than file layout:
#   real -- assistant turns the model actually served (both fields)
#   talk -- conversational traffic from the prompt catalogue (english + chinese)
#   code -- mined human code and the model's own code answers
TALK_GROUPS = {"chat", "zh", "zhcode", "zhthink"}

FORCED = [
    "\n", "\n\n", "    ", "\t", "\r\n", " \n",
    " ", "(", ")", "()", "):", "()\n", ");", ");\n",
    "{", "}", "{}", "[", "]", "[]", "():", "(),", "):\n",
    ",", ", ", ":", ": ", ";", ";\n", ".", "..", "...", "=", " = ", " == ", " != ",
    " -> ", " => ", " += ", " -= ", " < ", " > ", " >= ", " <= ", "+", "-", "*", "/",
    "//", "/*", "*/", "#", "# ", '"', "'", "`", "|", "&", "!", "?", "@",
    "def ", "class ", "return ", "if ", "else:", "elif ", "for ", "while ", "in ",
    "import ", "from ", "try:", "except ", "finally:", "with ", "as ", "yield ",
    "self", "self.", "None", "True", "False", "lambda ", "print(", "len(", "range(",
    "int", "str", "float", "list", "dict", "set", "tuple", "assert ", "raise ",
    "async ", "await ", "global ", "not ", "and ", "or ", "is ", "pass", "break", "continue",
    "int ", "void", "char ", "struct ", "static ", "const ", "/* ", " */", "// ",
    "#include", "#define", "printf(", "sizeof(", "malloc(", "free(", "NULL",
    "std::", "->", "#!", "#!/bin/bash", "#!/usr/bin/env python3",
    "function ", "const ", "let ", "var ", "export ", "require(", "console.log(",
    "SELECT ", "FROM ", "WHERE ", "JOIN ", "INSERT INTO ",
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
]


def pool_of(source: str, group: str) -> str:
    if source == "real":
        return "real"
    if source == "local":
        return "code"
    return "talk" if group in TALK_GROUPS else "code"


def depth_kind(group: str) -> str:
    return "prose" if group.split("/")[0] in PROSE_GROUPS else "code"


def gross(n_table: int, kind: str) -> float:
    d = DEPTH[kind]
    return HEAD_GB * d * (1 - n_table / N_VOCAB) / (TRUNK_GB + DRAFT_STEP_GB * d)


# ------------------------------------------------------------------------ input
def load(holdout: float):
    """(counts per pool, held-out samples per pool).

    Counts come from the training slice only; the held-out slice is kept as raw
    id lists for scoring. One pass over corpus/ids.
    """
    counts = defaultdict(Counter)
    hold = defaultdict(list)
    seen = defaultdict(int)
    files = [p for p in sorted(IDS_DIR.glob("*.json")) if not p.name.startswith("_")]
    for p in files:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        chunks = rec.get("chunks") or {}
        source, group = rec.get("source", "?"), rec.get("group", "?")
        pool = pool_of(source, group)
        per_field = {f: list((chunks.get(f) or {}).get("ids") or [])
                     for f in ("reasoning", "answer", "text")}
        ids_all = [t for f in ("reasoning", "answer", "text") for t in per_field[f]]
        if not ids_all:
            continue
        seen[pool] += 1
        # Deterministic holdout: every 1/holdout-th sample, so it is reproducible
        # and spread across the pool rather than bunched at one end.
        keep_every = max(2, int(round(1.0 / holdout))) if holdout > 0 else 0
        is_holdout = keep_every and (seen[pool] % keep_every == 0)
        if is_holdout:
            for f, ids in per_field.items():
                if ids:
                    hold["%s/%s" % (pool, f)].append({"label": rec.get("label", p.stem),
                                                      "group": group, "ids": ids})
        else:
            counts[pool].update(ids_all)
    # The mined-code pool is ~360x the served traffic. Scoring all of its held-out
    # files at every sweep point buys no extra precision on a coverage number that is
    # already flat near 99.7%, so it is strided down to a cap.
    cap = 400
    for key in list(hold):
        if key.startswith("code/") and len(hold[key]) > cap:
            step = len(hold[key]) / cap
            hold[key] = [hold[key][int(i * step)] for i in range(cap)]
    return counts, hold


def build_table(counts, weights, size: int, forced_ids: list[int]):
    """Rank the union of all pools by the mixture score, then keep the top `size`."""
    # Normalise each pool once: w_p / tokens_p. Doing this inside the per-token loop
    # is O(rows * distinct_tokens) and takes minutes on a 24M-token pool.
    scale = {}
    for pool, c in counts.items():
        w = weights.get(pool, 0.0)
        n = sum(c.values())
        if w and n:
            scale[pool] = w / n
    score: Counter = Counter()
    for pool, s in scale.items():
        for t, k in counts[pool].items():
            score[t] += k * s
    ranked = [t for t, _ in score.most_common()]
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
    n_forced = len(set(forced_ids) & set(table))
    # Row provenance: which pool's score bought this row. Reported because "24k rows"
    # says nothing about whether the budget actually moved.
    origin: Counter = Counter()
    for t in table:
        best, bw = None, 0.0
        for pool, s in scale.items():
            v = counts[pool].get(t, 0) * s
            if v > bw:
                best, bw = pool, v
        origin[best or "forced"] += 1
    return table[:size], n_forced, origin


def read_forced(gguf=None):
    """Token ids for the structural forced pieces, resolved against the GGUF."""
    import shortlist
    _p2i, id_of, _p_of, n_vocab, _tokens = shortlist.read_vocab_index(
        gguf or shortlist.DEFAULT_GGUF)
    ids = []
    for s in FORCED:
        i = id_of(s)
        if i is not None:
            ids.append(i)
    return ids, n_vocab


def evaluate(table: list[int], hold: dict, n_table: int, only: str | None = None):
    in_t = set(table)
    rows = []
    for key in sorted(hold):
        ss = hold[key]
        if only and not key.startswith(only):
            continue
        if not ss:
            continue
        covs, hits, tot = [], 0, 0
        miss: Counter = Counter()
        for s in ss:
            h = 0
            for t in s["ids"]:
                if t in in_t:
                    h += 1
                else:
                    miss[t] += 1
            covs.append(h / len(s["ids"]))
            hits += h
            tot += len(s["ids"])
        q = 1.0 - statistics.median(covs)
        kind = depth_kind(ss[0]["group"])
        gg = gross(n_table, kind)
        rows.append({"key": key, "n": len(ss), "tokens": tot, "cov_mixed": hits / tot,
                     "cov_median": statistics.median(covs), "q": q, "gross": gg,
                     "net": gg - q, "depth": kind, "miss": miss.most_common(10)})
    return rows


def print_detail(rows: list[dict], title: str) -> None:
    print("\n%s" % title)
    print("%-22s %5s %10s %11s %11s %9s %9s %s"
          % ("held-out set", "n", "tokens", "cov(mixed)", "cov(median)", "q", "NET", "depth"))
    for r in sorted(rows, key=lambda r: (r["key"].split("/")[0], r["key"])):
        print("%-22s %5d %10d %11.5f %11.5f %9.5f %+8.2f%% %s"
              % (r["key"], r["n"], r["tokens"], r["cov_mixed"], r["cov_median"],
                 r["q"], 100 * r["net"], r["depth"]))
    for r in rows:
        if r["miss"]:
            bits = "; ".join("%d x%d" % (t, c) for t, c in r["miss"][:6])
            print("  dropped[%-18s] %s" % (r["key"], bits))


# ------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--size", type=int, default=24000)
    ap.add_argument("--holdout", type=float, default=0.25,
                    help="fraction of each pool reserved for scoring (0 = in-sample)")
    ap.add_argument("--weights", default="real=3,talk=2,code=2",
                    help="pool weights for a single build")
    ap.add_argument("--sweep", action="store_true",
                    help="build the same table at several talk weights and report the curve")
    ap.add_argument("--sweep-talk", default="0,1,2,3,4")
    ap.add_argument("--forced", type=int, default=1, help="include forced structural tokens")
    ap.add_argument("--out", default=None, help="write the map here")
    ap.add_argument("--tag", default="mix")
    ap.add_argument("--score-table", default=None,
                    help="score an existing map on the held-out slices and stop. "
                         "Use it to compare the shipped table against a built one on "
                         "identical holdout, which is the only fair comparison.")
    ap.add_argument("--gguf", default=None)
    args = ap.parse_args(argv)

    counts, hold = load(args.holdout)
    tot = {k: sum(v.values()) for k, v in counts.items()}
    held = {k: len(v) for k, v in hold.items()}
    print("pools : %s" % {k: "%d tokens" % v for k, v in sorted(tot.items())})
    print("held  : %s" % dict(sorted(held.items())))
    if not counts:
        print("no tokenized samples -- run the real/local/gen stages then tok")
        return 1

    if args.score_table:
        import groupcov
        p = Path(args.score_table)
        table = groupcov.read_map(p if p.is_absolute() else BUNDLE / p)
        print_detail(evaluate(table, hold, len(table)), "SCORED: %s (%d entries)"
                     % (p.name, len(table)))
        return 0

    forced_ids = []
    if args.forced:
        forced_ids, n_vocab = read_forced(args.gguf)
        print("forced: %d structural tokens resolved" % len(forced_ids))

    def parse_w(s):
        w = {}
        for part in s.split(","):
            k, _, v = part.partition("=")
            w[k.strip()] = float(v)
        return w

    settings = [(parse_w(args.weights), "chosen")]
    if args.sweep:
        settings = [({"real": 3.0, "talk": float(t), "code": 2.0}, "talk=%s" % t)
                    for t in args.sweep_talk.split(",")]

    print("\n%-10s %8s %8s %10s %10s %10s %10s"
          % ("setting", "rows", "talk_rows", "real q", "real NET", "code q", "code NET"))
    best = None
    for weights, label in settings:
        table, n_forced, origin = build_table(counts, weights, args.size, forced_ids)
        rows = evaluate(table, hold, len(table))
        by = {r["key"]: r for r in rows}
        real_all = _merge([by[k] for k in ("real/reasoning", "real/answer") if k in by])
        code_all = _merge([r for k, r in by.items() if k.startswith("code/")])
        talk_all = _merge([r for k, r in by.items() if k.startswith("talk/")])
        n_talk = origin.get("talk", 0)
        print("%-10s %8d %8d %10.5f %+9.2f%% %10.5f %+9.2f%%"
              % (label, len(table), n_talk, real_all.get("q", 0), 100 * real_all.get("net", 0),
                 code_all.get("q", 0), 100 * code_all.get("net", 0)))
        print("            row origin: %s  (forced=%d)" % (dict(origin.most_common()), n_forced))
        if talk_all:
            print("            talk q=%.5f NET=%+.2f%%" % (talk_all["q"], 100 * talk_all["net"]))
        score = real_all.get("net", -9)
        if best is None or score > best[0]:
            best = (score, weights, label, table, origin, rows)

    # ---- detail for the winner, on the same slices every contender saw
    score, weights, label, table, origin, rows = best
    print_detail(rows, "BEST-BY-SERVED-TRAFFIC: %s  (weights %s)" % (label, weights))

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = VOCAB_DIR / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("llama-mtp-vocab-v1 248320 %d\n" % len(table)
                       + " ".join(str(t) for t in table) + "\n", encoding="utf-8")
        print("\nwrote %s (%d entries)  weights=%s" % (out, len(table), weights))
    return 0


def _merge(rows: list[dict]) -> dict:
    """Token-weighted merge of several held-out buckets."""
    if not rows:
        return {}
    tok = sum(r["tokens"] for r in rows)
    if not tok:
        return {}
    q = sum(r["q"] * r["tokens"] for r in rows) / tok
    gg = sum(r["gross"] * r["tokens"] for r in rows) / tok
    return {"q": q, "gross": gg, "net": gg - q, "tokens": tok}


if __name__ == "__main__":
    raise SystemExit(main())
