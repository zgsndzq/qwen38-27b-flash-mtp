"""groupcov.py -- shortlist coverage broken down by prompt group, and by field.

`shortlist.py` reports one coverage curve for a whole corpus, averaged over
samples. That answers "is this table good enough *on the corpus it was built
from*". It does not answer what production actually asks, because production runs
**one long-lived table against mixed traffic**:

  * `maps/shortlist_24k.vocab.txt` is mined from *code* (the local source tree),
    but the same server also answers Chinese chat and instruction prose.
  * A token outside the table is a guaranteed draft rejection, so the cost is the
    drop rate q of *that* traffic -- not a corpus average that code dominates.
  * If q exceeds the gross byte saving, the shortlist is a net **loss** for that
    traffic, and the aggregate number hides it completely.

Two splits matter, and they answer different questions:

  ``--by group``  (default ``key`` = source:group)
      Code vs Chinese vs the served traffic. This is the traffic-mix question.
  ``--split-fields``
      Reasoning block vs answer text, inside each sample. This is the question the
      served traffic raised: the model *thinks in English* even when it answers in
      Chinese, so a table mined from English code can look fine on 62% of a
      Chinese request's tokens while losing badly on the rest. Pooling the two
      fields into one number hides exactly that.

Build tables with `run.py build` (it owns the structural forced-token logic);
score them here:

    python groupcov.py                                   # the shipped table
    python groupcov.py --table out/shortlist_24k_gen.vocab.txt
    python groupcov.py --table maps/shortlist_24k.vocab.txt --missing
    python groupcov.py --by source
    python groupcov.py --split-fields --missing 8         # thinking vs answer

Sources come from `run.py`: `real` (traffic the model actually served),
`gen` (the model answering the prompt catalogue), `local` (mined human code).

Economics are copied verbatim from `shortlist.py` so the two agree:

    gross(d, N) = 0.265 * d * (1 - N/248320) / (5.765 + 0.395 * d)
    net(N)      = gross(d, N) - q

`d` is the measured mean draft depth: 2.657 for code, 2.073 for prose. Which one
applies is decided per group -- chat and real traffic are prose, the rest is code.
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
TRUNK_GB = 5.765          # trunk weights + target head, per decode round, fixed
DRAFT_STEP_GB = 0.395     # per draft step: MTP layer (~0.13) + full lm_head (0.265)
HEAD_GB = 0.265           # the full draft lm_head, the only part a shortlist shrinks
DEPTH = {"code": 2.657, "prose": 2.073}
PROSE_GROUPS = {"chat", "real"}   # dialogue-shaped traffic; everything else is code
FIELDS = ("reasoning", "answer", "text")


def depth_kind(group: str) -> str:
    """Draft depth class for a bucket label, ignoring any '/field' suffix."""
    return "prose" if group.split("/")[0] in PROSE_GROUPS else "code"


def gross(n_table: int, kind: str) -> float:
    d = DEPTH[kind]
    return HEAD_GB * d * (1 - n_table / N_VOCAB) / (TRUNK_GB + DRAFT_STEP_GB * d)


# ------------------------------------------------------------------------ input
def read_map(path: Path) -> list[int]:
    """Parse a `llama-mtp-vocab-v1` map. Loud on malformed input, like the server."""
    toks = Path(path).read_text(encoding="utf-8").split()
    if len(toks) < 3 or toks[0] != "llama-mtp-vocab-v1":
        raise SystemExit("not a llama-mtp-vocab-v1 map: %s" % path)
    declared, count = int(toks[1]), int(toks[2])
    ids = [int(t) for t in toks[3:]]
    if len(ids) != count:
        raise SystemExit("%s declares %d entries but holds %d" % (path, count, len(ids)))
    if len(set(ids)) != len(ids):
        raise SystemExit("%s holds duplicate ids (the server rejects this)" % path)
    if declared != N_VOCAB:
        print("note: map targets n_vocab=%d, this tool assumes %d" % (declared, N_VOCAB))
    return ids


def load_samples(split_fields: bool = False) -> list[dict]:
    """Every tokenized sample.

    Default: the fields of one sample are concatenated, which is what the server
    sees. With ``split_fields`` each non-empty field becomes its own bucket, so
    the reasoning block and the answer are scored separately.
    """
    out = []
    for p in sorted(IDS_DIR.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        chunks = rec.get("chunks") or {}
        source, group = rec.get("source", "?"), rec.get("group", "?")
        if split_fields:
            for name in FIELDS:
                ids = list((chunks.get(name) or {}).get("ids") or [])
                if not ids:
                    continue
                out.append({"key": "%s:%s/%s" % (source, group, name), "source": source,
                            "group": "%s/%s" % (group, name),
                            "backend": rec.get("backend", "?"),
                            "label": rec.get("label", p.stem), "ids": ids})
            continue
        ids: list[int] = []
        for name in FIELDS:
            ids.extend((chunks.get(name) or {}).get("ids") or [])
        if not ids:
            continue
        out.append({"key": "%s:%s" % (source, group), "source": source, "group": group,
                    "backend": rec.get("backend", "?"),
                    "label": rec.get("label", p.stem), "ids": ids})
    return out


# --------------------------------------------------------------------- scoring
def score(buckets: dict[str, list[dict]], table: list[int], top_missing: int = 0):
    """Coverage + economics for each bucket against one table."""
    in_t = set(table)
    rows = []
    for key in sorted(buckets):
        ss = buckets[key]
        if not ss:
            continue
        covs, hits, tot = [], 0, 0
        missing: Counter = Counter()
        for s in ss:
            h = 0
            for t in s["ids"]:
                if t in in_t:
                    h += 1
                else:
                    missing[t] += 1
            covs.append(h / len(s["ids"]))
            hits += h
            tot += len(s["ids"])
        cov_med = statistics.median(covs)
        q_med = 1.0 - cov_med
        kind = depth_kind(ss[0]["group"])
        gg = gross(len(table), kind)
        rows.append({
            "key": key, "n_samples": len(ss), "n_tokens": tot,
            "coverage_mixed": hits / tot, "coverage_median": cov_med,
            "coverage_min": min(covs),
            "coverage_p10": sorted(covs)[max(0, int(0.10 * len(covs)) - 1)],
            "q_median": q_med, "gross": gg, "net": gg - q_med, "depth": kind,
            "backend": ss[0]["backend"],
            "top_missing": missing.most_common(top_missing) if top_missing else [],
        })
    return rows


def bucket_by(samples: list[dict], group_by: str) -> dict[str, list[dict]]:
    b: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        b[s[group_by]].append(s)
    return b


def print_rows(rows: list[dict], n_table: int, title: str) -> None:
    print("\n%s  (table=%d entries)" % (title, n_table))
    print("%-24s %7s %10s %11s %11s %8s %8s %9s %s"
          % ("group", "samples", "tokens", "cov(mixed)", "cov(median)", "q", "gross", "NET", "depth"))
    for r in rows:
        print("%-24s %7d %10d %11.5f %11.5f %8.5f %+7.2f%% %+8.2f%% %s"
              % (r["key"], r["n_samples"], r["n_tokens"], r["coverage_mixed"],
                 r["coverage_median"], r["q_median"], 100 * r["gross"], 100 * r["net"], r["depth"]))
    print("  cov = token hit rate; q = 1 - median per-sample coverage;")
    print("  NET = gross - q (the design doc's net saving). Negative NET = the table")
    print("  costs more speed on this traffic than it buys.")


def lazy_pieces():
    """Token -> literal piece, from the GGUF. None if unavailable."""
    try:
        from tok import OfflineBpe
        bpe = OfflineBpe()

        def piece(i: int) -> str:
            try:
                return bpe.piece(i)
            except Exception:
                return ""
        return piece
    except Exception as e:
        print("(piece lookup unavailable: %s)" % e)
        return None


# ------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--table", default=str(BUNDLE / "maps" / "shortlist_24k.vocab.txt"),
                    help="map file to score (default: the shipped one)")
    ap.add_argument("--by", default="key", choices=["key", "source", "group"])
    ap.add_argument("--split-fields", action="store_true",
                    help="score the reasoning block and the answer separately")
    ap.add_argument("--missing", type=int, nargs="?", const=6, default=0,
                    metavar="N", help="also list the N most-dropped tokens per group")
    ap.add_argument("--json", default=None, help="write the raw rows here")
    args = ap.parse_args(argv)

    samples = load_samples(split_fields=args.split_fields)
    if not samples:
        print("no tokenized samples in %s -- run 'python run.py local|gen' then 'tok'" % IDS_DIR)
        return 1
    srcs: Counter = Counter(s["source"] for s in samples)
    backends: Counter = Counter(s["backend"] for s in samples)
    print("samples=%d  by source=%s  backends=%s" % (len(samples), dict(srcs), dict(backends)))
    if "offline" in backends:
        print("NOTE: the `local` source is tokenized with the offline-approx tokenizer; "
              "its numbers are a preview. `real`/`gen` use the live tokenizer.")

    table = read_map(Path(args.table))
    rows = score(bucket_by(samples, args.by), table, args.missing)
    print_rows(rows, len(table), "table: %s" % Path(args.table).name)

    if args.missing:
        pieces = lazy_pieces()
        print("\nmost-dropped tokens per group (id x count  piece):")
        for r in rows:
            if not r["top_missing"]:
                continue
            bits = "; ".join("%d x%d %r" % (t, c, (pieces(t) if pieces else ""))
                             for t, c in r["top_missing"])
            print("  %-24s %s" % (r["key"], bits))

    if args.json:
        Path(args.json).write_text(
            json.dumps({"table": str(args.table), "by": args.by,
                        "split_fields": args.split_fields, "rows": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
