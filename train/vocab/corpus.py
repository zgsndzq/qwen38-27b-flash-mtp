"""corpus.py -- collect the two corpora the shortlist is built from.

There are deliberately three independent sources, because they answer different
questions and one of them is free:

  ``real``  -- import the assistant turns the model actually served, straight out of
               the agent's SQLite log (thinking block included). This is the thing
               itself rather than a proxy for it, and the only source that contains
               the thinking block at all, but it is small (tens of rows).
               Costs nothing.
  ``gen``   -- let the bundle model answer the prompts.py catalogue. The scalable
               authoritative source: the draft head is trained to imitate the
               model's own next-token distribution, so the model's own output is
               the closest available proxy for what the draft has to guess.
               Costs GPU time (~25 s per sample).
  ``local`` -- mine a real source tree already on disk. Zero GPU cost, available
               immediately, and useful as a sanity upper bound: if the coverage
               curve on human-written code is already bad, model output will not
               rescue it. It is NOT the same distribution (humans write different
               identifier names), so it is a preview, not the verdict.

All of them write one JSON per sample so a run reaped at the ~2 minute mark keeps
everything it finished, and re-running the same command resumes instead of
repeating work.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import srv

VOCAB_DIR = Path(__file__).resolve().parent
CORPUS = VOCAB_DIR / "corpus"
GEN_DIR = CORPUS / "gen"
LOCAL_DIR = CORPUS / "local"
REAL_DIR = CORPUS / "real"

# Extensions worth mining. Deliberately excludes .md/.txt: prose would dilute the
# "code keys are a fixed structure" premise that this exercise depends on.
CODE_EXT = {".py", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".cuh", ".cu",
            ".sh", ".bash", ".bat", ".ps1", ".cmd",
            ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
            ".sql", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
            ".cmake", ".mk", ".make", ".go", ".rs", ".java"}

# Directory names that are build output, caches, vendored copies or archives.
# The toolchain entries matter: F:\super\tools\vs-layout holds Visual Studio
# redistributable installer scripts and Windows SDK headers, which poisoned the
# first full run (top tokens came out as `IDirect`, `IColor`, `PropertyValue`).
SKIP_DIRS = {"__pycache__", ".git", ".hg", ".svn", ".archive", "node_modules",
             "models", "cuda", "bin", "_donors", ".workbuddy", "delete",
             "dist", "obj", "site-packages", ".venv", "venv", "target",
             "bak", "tmp", "temp", "old", "backup", "backups",
             "vs-layout", "Windows Kits", "MSVC", "Redist", "NuGet", "dotnet",
             "SiteExtensions", "Common7", "VC", "WindowsApps", "Packages"}

# Extension buckets with a sampling quota each. Without a quota the biggest source
# tree wins by sheer file count -- the first dry run of this pipeline returned 600
# files that were all one-off ~/bak/_*.ps1 scratch scripts, i.e. nothing like the
# code the model is asked to produce.
EXT_BUCKETS = {
    "native": {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".cuh", ".cu"},
    "python": {".py"},
    "shell":  {".sh", ".bash", ".bat", ".ps1", ".cmd"},
    "web":    {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".sql"},
    "config": {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".cmake", ".mk", ".make"},
    "other":  {".go", ".rs", ".java"},
}
EXT_QUOTA = {"native": 0.38, "python": 0.30, "shell": 0.14, "web": 0.12,
             "config": 0.04, "other": 0.02}

# Build trees are named with many suffixes (build-win-kvmc, build-gdn, _dflash_src,
# _kvmc_backup, ...). Match them by prefix/pattern rather than listing every one.
SKIP_DIR_RE = re.compile(r"^(build|_dflash_src|_kvmc_backup|_mtp|_old)|(^build[-_])", re.I)

MAX_FILE_BYTES = 256 * 1024


def _bucket(ext: str) -> str:
    for name, exts in EXT_BUCKETS.items():
        if ext in exts:
            return name
    return "other"


def _quota_sample(files: list[Path], limit: int, seed: int = 20260922) -> list[Path]:
    """Take a quota per extension bucket, shuffling within each bucket.

    Shuffling matters: the file list arrives sorted, so taking the first N of a
    bucket would mine whichever directory sorts first and nothing else. A fixed
    seed keeps runs reproducible.
    """
    if not limit or len(files) <= limit:
        return files
    rng = random.Random(seed)
    groups: dict[str, list[Path]] = {}
    for p in files:
        groups.setdefault(_bucket(p.suffix.lower()), []).append(p)
    out: list[Path] = []
    taken: set = set()
    for name, q in EXT_QUOTA.items():
        g = list(groups.get(name, []))
        rng.shuffle(g)
        for p in g[:max(1, int(round(limit * q)))]:
            out.append(p)
            taken.add(p)
    # Backfill only from non-config buckets. When a bucket is short (this tree has
    # ~800 python files, well under the 30% quota) the shortfall used to be filled
    # in path order, and 'llm/.../json' sorts early enough to flood the corpus:
    # 1384 .json samples against a 240-file quota, which is why the raw frequency
    # list came back as digits and commas.
    rest = [p for p in files if p not in taken and _bucket(p.suffix.lower()) != "config"]
    rng.shuffle(rest)
    out.extend(rest)
    return out[:limit]


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _flush_index(directory: Path, index: dict) -> None:
    (directory / "_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ real log
# The authoritative corpus is the traffic the model actually served. The ZhiYan
# agent writes every assistant turn to Agents.db, thinking block included, and
# that is the token mix the draft head has to guess -- not a proxy for it. It is
# small (43 rows), so it complements `gen` rather than replacing it, but it is
# the only source that contains the thinking block that the launcher's default
# -rea off hides from every other stage.
DEFAULT_DB = Path(r"F:\bycode\ZhiYan\backend\data\Agents.db")
DB_FALLBACK = VOCAB_DIR / "Agents.db"


def real(args) -> int:
    import sqlite3

    db = Path(getattr(args, "db", None) or DEFAULT_DB)
    if not db.exists():
        if DB_FALLBACK.exists():
            print("log     : primary missing, falling back to %s" % DB_FALLBACK)
            db = DB_FALLBACK
        else:
            print("no conversation log found (looked for %s)" % DEFAULT_DB)
            return 1
    print("log     : %s (%.2f MB)" % (db, db.stat().st_size / 1e6))
    # Read-only URI: this file belongs to another application and must survive
    # the import untouched -- no journal writes, no schema migration on open.
    con = sqlite3.connect("file:%s?mode=ro" % str(db).replace("\\", "/"),
                          uri=True, timeout=20)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        rows = list(con.execute(
            "SELECT id, conversation_id, content, thinking, thinking_blocks, timestamp "
            "FROM messages WHERE role='assistant' ORDER BY timestamp"))
    finally:
        con.close()

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    index, skipped, with_think = {}, 0, 0
    for mid, conv, content, thinking, blocks, ts in rows:
        # thinking_blocks wraps the *same* text in a JSON list (measured ratio
        # ~0.96 against the plain column), so it is a fallback, never an addition.
        reasoning = (thinking or "").strip()
        if not reasoning and (blocks or "").strip():
            raw = blocks.strip()
            try:
                obj = json.loads(raw)
                if isinstance(obj, list):
                    reasoning = "\n\n".join(str(x) for x in obj).strip()
                elif isinstance(obj, dict):
                    reasoning = str(obj.get("text") or obj.get("content") or "").strip()
                else:
                    reasoning = raw
            except Exception:
                reasoning = raw
        answer = (content or "").strip()
        if len(answer) + len(reasoning) < 64:
            skipped += 1
            continue
        label = "%s__%s" % (re.sub(r"[^A-Za-z0-9._-]+", "_", str(conv or "conv"))[:40],
                            str(mid)[:16])
        _write(REAL_DIR / ("%s.json" % label),
               {"label": label, "group": "real", "conversation": conv, "ts": ts,
                "thinking": bool(reasoning), "reasoning": reasoning, "answer": answer})
        index[label] = {"conversation": conv, "has_thinking": bool(reasoning),
                        "n_answer": len(answer), "n_reasoning": len(reasoning)}
        with_think += 1 if reasoning else 0
    _flush_index(REAL_DIR, index)
    print("samples : %d  with_thinking=%d  skipped=%d" % (len(index), with_think, skipped))
    print("written -> %s" % REAL_DIR)
    return 0


# ------------------------------------------------------------------ generated
def gen(args) -> int:
    import prompts

    tasks = prompts.catalog(args.groups)
    if args.limit:
        tasks = tasks[:args.limit]
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    pending = [t for t in tasks if not (GEN_DIR / ("%s.json" % t[0])).exists()]
    done_n = len(tasks) - len(pending)
    print("catalogue=%d  already done=%d  pending=%d" % (len(tasks), done_n, len(pending)))
    if not pending:
        print("nothing to generate; use --overwrite to redo")
        return 0

    index = {}
    idx_path = GEN_DIR / "_index.json"
    if idx_path.exists():
        try:
            index = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}

    if args.dry_run:
        for label, group, _p in pending:
            print("  would generate %-22s [%s]" % (label, group))
        return 0

    log = VOCAB_DIR / "out" / "server_gen.log"
    with srv.Server(port=args.port, bat=args.bat, attach_only=args.attach_only, log=log) as s:
        print("server: %s" % s.attach_why)
        try:
            model_path = s.props().get("model_path", "?")
        except Exception:
            model_path = "?"
        for i, (label, group, prompt) in enumerate(pending, 1):
            t_label = "%s (%d/%d)" % (label, i, len(pending))
            # Only the thinking groups request the reasoning block; every other
            # group is generated exactly as the launcher serves (-rea off), so the
            # corpus matches the current configuration by default.
            want_think = group in getattr(prompts, "THINKING_GROUPS", ())
            # A thinking sample that stops at the default 1024 never reaches its
            # answer at all -- the block alone consumed the whole budget in the
            # probe. 2048 mirrors the launcher's --reasoning-budget.
            mt = max(args.max_tokens, 2048) if want_think else args.max_tokens
            try:
                resp = s.chat([{"role": "user", "content": prompt}],
                              max_tokens=mt, temperature=args.temperature,
                              seed=args.seed, thinking=True if want_think else None)
                ch = (resp.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                reasoning, answer = srv.pick_reasoning(msg)
                tm = resp.get("timings") or {}
                rec = {
                    "label": label, "group": group, "prompt": prompt,
                    "thinking": want_think,
                    "reasoning": reasoning, "answer": answer,
                    "finish_reason": ch.get("finish_reason"),
                    "n_reasoning": len(reasoning), "n_answer": len(answer),
                    "tg": round(tm.get("predicted_per_second") or 0, 2),
                    "tokens_predicted": tm.get("predicted_n"),
                    "model": model_path,
                }
                _write(GEN_DIR / ("%s.json" % label), rec)
                index[label] = {"group": group, "n_answer": len(answer),
                                "n_reasoning": len(reasoning), "tg": rec["tg"],
                                "thinking": want_think,
                                "finish_reason": rec["finish_reason"]}
                print("  [%s] %-22s ans=%-5d think=%-5d tg=%6.2f %s" % (
                    group, label, len(answer), len(reasoning), rec["tg"],
                    rec["finish_reason"] or ""))
            except Exception as e:                       # keep going: one failure must not
                print("  [%s] %-22s ERROR %s: %s" % (group, label, type(e).__name__, e))
                index[label] = {"group": group, "error": "%s: %s" % (type(e).__name__, e)}
            _flush_index(GEN_DIR, index)                 # survive a reaped job
    print("generated %d sample(s) -> %s" % (len(pending), GEN_DIR))
    return 0


# ---------------------------------------------------------------------- local
def _iter_code_files(roots: list[Path]):
    corpus_root = CORPUS.resolve()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in CODE_EXT:
                continue
            if p.name.startswith("_"):
                continue          # house convention: "_name" marks a scratch file
            if p.is_relative_to(corpus_root):
                continue          # never mine our own output back in (.json is whitelisted)
            parts = set(p.relative_to(root).parts[:-1])
            if parts & SKIP_DIRS:
                continue
            if any(SKIP_DIR_RE.search(part) for part in p.relative_to(root).parts[:-1]):
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES or p.stat().st_size < 32:
                    continue
            except OSError:
                continue
            yield p


def _slug(root: Path, p: Path) -> str:
    rel = str(p.relative_to(root)).replace("\\", "/")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", "%s__%s" % (root.name, rel))[:120]


def _root_of(p: Path, roots: list[Path]) -> Path | None:
    for root in roots:
        try:
            p.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def local(args) -> int:
    # Default to actual project trees rather than the drive root: the root also
    # holds toolchain payloads (F:\super\tools\vs-layout\*.ps1 installer scripts,
    # CUDA redistributables) that made it into the first run and are not code the
    # model would ever be asked to write.
    super_root = VOCAB_DIR.parents[1]
    default_roots = [super_root / "llm", super_root / "qwen27b", super_root / "llmchat"]
    roots = [Path(r).resolve() for r in (args.roots or [str(p) for p in default_roots if p.exists()])]
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    files = _quota_sample(list(_iter_code_files(roots)), args.limit_files)
    print("roots   : %s" % ", ".join(str(r) for r in roots))
    print("files   : %d" % len(files))
    if args.dry_run:
        for p in files[:40]:
            print("  %s" % p)
        return 0

    index, ext_counts, bytes_total, skipped = {}, {}, 0, 0
    for p in files:
        root = _root_of(p, roots)
        if root is None:
            skipped += 1
            continue
        text = None
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                text = p.read_text(encoding=enc)
                break
            except (UnicodeDecodeError, OSError):
                continue
        if not text or len(text) < 32:
            skipped += 1
            continue
        slug = _slug(root, p)
        rel = str(p.relative_to(root)).replace("\\", "/")
        _write(LOCAL_DIR / ("%s.json" % slug),
               {"label": slug, "group": "local", "root": str(root),
                "relpath": rel, "text": text})
        index[slug] = {"relpath": rel, "bytes": len(text.encode("utf-8"))}
        ext_counts[p.suffix.lower()] = ext_counts.get(p.suffix.lower(), 0) + 1
        bytes_total += len(text.encode("utf-8"))
    _flush_index(LOCAL_DIR, index)
    print("samples : %d  bytes=%d  skipped=%d" % (len(index), bytes_total, skipped))
    print("by ext  : %s" % sorted(ext_counts.items(), key=lambda kv: -kv[1])[:12])
    print("written -> %s" % LOCAL_DIR)
    return 0
