"""run.py -- one entry point for the whole draft-vocabulary-shortlist pipeline.

Stages, in dependency order:

  real    import the served traffic log       -> corpus/real/*.json    (no GPU)
  local   mine code already on disk      -> corpus/local/*.json     (no GPU)
  gen     let the model answer prompts   -> corpus/gen/*.json       (needs the server)
  tok     tokenize every corpus sample   -> corpus/ids/*.json       (server or offline)
  build   statistics + map files + verdict                          (no GPU)
  status  what exists so far
  all     real, local, gen, tok, build

Every stage is incremental: it skips work whose output already exists and is newer
than its input, so re-running after an interruption continues instead of
restarting. That is not decoration -- a long job on this machine can be reaped at
roughly the two minute mark, and the pipeline is built so that costs nothing.

Examples
  python run.py real                        # the traffic the model actually served
  python run.py local                       # free preview from the local source tree
  python run.py gen --limit 12              # small batch first
  python run.py gen --groups zh,zhcode      # only the Chinese groups
  python run.py tok --backend offline       # no GPU needed
  python run.py build --source real
  python run.py all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VOCAB_DIR = Path(__file__).resolve().parent
CORPUS = VOCAB_DIR / "corpus"
OUT_DIR = VOCAB_DIR / "out"


def stage_status(_args) -> int:
    print("vocab pipeline -- %s\n" % VOCAB_DIR)
    total_gb = 0.0
    for label, d, kind in (("real (served traffic) ", CORPUS / "real", "json"),
                           ("gen  (model-generated)", CORPUS / "gen", "json"),
                           ("local(real code tree)", CORPUS / "local", "json"),
                           ("ids  (tokenized)     ", CORPUS / "ids", "json"),
                           ("out  (tables+reports)", OUT_DIR, "*")):
        if not d.exists():
            print("  %-24s absent" % label)
            continue
        files = [p for p in d.glob("*.%s" % kind) if not p.name.startswith("_")] \
            if kind != "*" else [p for p in d.glob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in files) / 1e6
        total_gb += size / 1000
        print("  %-24s %4d files  %8.2f MB" % (label, len(files), size))
    print("")

    idx = CORPUS / "ids"
    if idx.exists():
        by_source, by_backend, tok_total = {}, {}, 0
        for p in sorted(idx.glob("*.json")):
            if p.name.startswith("_"):
                continue
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            by_source[r.get("source", "?")] = by_source.get(r.get("source", "?"), 0) + 1
            by_backend[r.get("backend", "?")] = by_backend.get(r.get("backend", "?"), 0) + 1
            tok_total += r.get("n_tokens") or 0
        print("  tokenized: %d tokens across %s (backends %s)"
              % (tok_total, by_source, by_backend))
        if "offline" in by_backend:
            print("  NOTE: offline-approx ids present; the coverage curve is a preview only.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("stage", choices=["real", "local", "gen", "tok", "verify",
                                      "build", "status", "all"])
    # service
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--bat", default=None, help="launcher .bat (default: bundle llmserver.bat)")
    ap.add_argument("--attach-only", action="store_true",
                    help="never launch a server; require one already running")
    # corpus
    ap.add_argument("--db", default=None,
                    help="conversation log to import in the real stage "
                         "(default: the ZhiYan agent's Agents.db)")
    ap.add_argument("--groups", default="all", help="prompt groups for gen, e.g. py,c,sh")
    ap.add_argument("--limit", type=int, default=0, help="max items this run (0 = all)")
    ap.add_argument("--limit-files", type=int, default=600, help="max local files to mine")
    ap.add_argument("--roots", action="append", help="roots to mine (repeatable)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    # tokenizer
    ap.add_argument("--backend", default="auto", choices=["auto", "service", "offline"])
    ap.add_argument("--only", default=None,
                    help="restrict the tok stage to these sources, e.g. gen,real "
                         "(avoids re-tokenizing 6k local samples through the server)")
    ap.add_argument("--gguf", default=None)
    # build
    ap.add_argument("--source", default="gen", choices=["gen", "local", "real", "all"])
    ap.add_argument("--field", default="all", choices=["all", "answer", "reasoning", "text"])
    ap.add_argument("--sizes", default="4k,8k,12k,16k,24k,32k,48k,64k")
    ap.add_argument("--splits", type=int, default=2)
    ap.add_argument("--hot", type=int, default=0)
    ap.add_argument("--top-show", type=int, default=60)
    ap.add_argument("--depth-code", type=float, default=2.657)
    ap.add_argument("--depth-prose", type=float, default=2.073)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    import corpus
    import tok
    import shortlist

    if args.gguf:
        # OfflineBpe reads tok.DEFAULT_GGUF at call time, so this takes effect.
        tok.DEFAULT_GGUF = Path(args.gguf)

    def do_real():
        print("== stage: real ==")
        return corpus.real(args)

    def do_local():
        print("== stage: local ==")
        return corpus.local(args)

    def do_gen():
        print("== stage: gen ==")
        return corpus.gen(args)

    def do_tok():
        print("== stage: tok ==")
        return tok.cmd(args)

    def do_build():
        print("== stage: build ==")
        argv_b = ["--source", args.source, "--field", args.field, "--sizes", args.sizes,
                  "--splits", str(args.splits), "--hot", str(args.hot),
                  "--top-show", str(args.top_show), "--depth-code", str(args.depth_code),
                  "--depth-prose", str(args.depth_prose)]
        if args.tag:
            argv_b += ["--tag", args.tag]
        if args.gguf:
            argv_b += ["--gguf", args.gguf]
        return shortlist.main(argv_b)

    if args.stage == "status":
        return stage_status(args)
    if args.stage == "real":
        return do_real()
    if args.stage == "local":
        return do_local()
    if args.stage == "gen":
        return do_gen()
    if args.stage == "tok":
        return do_tok()
    if args.stage == "verify":
        print("== stage: verify (offline tokenizer vs the live one) ==")
        return tok.consistency(n=args.limit, port=args.port, gguf=args.gguf)
    if args.stage == "build":
        return do_build()

    for fn in (do_real, do_local, do_gen, do_tok, do_build):      # stage == all
        rc = fn()
        if rc:
            print("stage failed (rc=%d); stopping" % rc)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
