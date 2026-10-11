# -*- coding: utf-8 -*-
"""Build the code training library (trainlib) from F:\\github.

Balanced sampling: a bounded number of files per project per extension, classified
into a type tree, with a manifest so later runs only add what is genuinely new.

Layout
    trainlib/<category>/<project>/<relative path>
    trainlib/MANIFEST.csv     src,project,category,ext,bytes,lines,sha1,dest
    trainlib/STATS.json       per category / per project counts

Re-run any time (e.g. after adding a repo to F:\\github) to top the library up.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import sys

ROOT = r"F:\github"
LIB = r"F:\super\qwen38mtp\trainlib"

# ---------------------------------------------------------------- classification

# first match wins; ext -> category
CAT_OF_EXT = {}
CATEGORIES = [
    ("css", [".css", ".scss", ".sass", ".less", ".styl", ".wxss", ".pcss"]),
    ("html", [".html", ".htm", ".xhtml", ".vue", ".svelte", ".astro", ".ejs",
              ".hbs", ".handlebars", ".pug", ".jade", ".erb", ".twig", ".njk", ".liquid"]),
    ("js", [".js", ".mjs", ".cjs", ".jsx"]),
    ("ts", [".ts", ".tsx", ".mts", ".cts"]),
    ("python", [".py", ".pyi", ".pyw"]),
    ("markdown", [".md", ".markdown", ".mdx", ".rst", ".txt", ".adoc", ".asciidoc"]),
    ("sql", [".sql"]),
    ("shell", [".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd", ".nsh"]),
    ("config", [".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
                ".properties", ".editorconfig", ".example", ".webmanifest", ".plist"]),
    ("markup", [".xml", ".xsd", ".xsl", ".xslt", ".svg", ".html.erb"]),
    ("other", [".c", ".h", ".cc", ".cpp", ".hpp", ".hxx", ".cs", ".java", ".kt", ".kts",
               ".go", ".rs", ".rb", ".php", ".swift", ".m", ".mm", ".lua", ".r", ".jl",
               ".scala", ".dart", ".ex", ".exs", ".erl", ".hs", ".ml", ".fs", ".clj",
               ".zig", ".nim", ".v", ".j2", ".tpl", ".nix", ".bicep", ".ipynb", ".sse",
               ".proto", ".graphql", ".tf", ".hcl", ".dockerfile", ".gitignore",
               ".dockerignore", ".csv", ".jsonl", ".rs.in", ".in"]),
]
for _cat, _exts in CATEGORIES:
    for _e in _exts:
        CAT_OF_EXT.setdefault(_e, _cat)

# user focus first: python / ts / js / css / html / markdown get the budget first
PRIORITY = ["python", "ts", "js", "css", "html", "markdown", "sql", "shell",
            "config", "markup", "other"]

# per-extension caps (default: --per-ext). keep the noisy / low-signal ones small
EXT_CAP = {
    ".svg": 3, ".ipynb": 2, ".json": 6, ".jsonl": 3, ".csv": 3, ".txt": 6,
    ".xsd": 3, ".lock": 0, ".xml": 4, ".plist": 2, ".sse": 2, ".bicep": 1,
    ".in": 2, ".example": 2, ".webmanifest": 1,
}
CAT_CAP = {"markup": 8, "config": 24, "other": 30}

SKIP_DIRS = {
    "node_modules", ".git", ".svn", ".hg", "vendor", "dist", "build", "out",
    "target", "__pycache__", ".venv", "venv", "env", ".idea", ".vscode",
    "coverage", ".next", ".nuxt", ".cache", "bower_components", ".terraform",
    "pods", ".gradle", ".mvn", "obj", "bin", "debug", "release",
    "site-packages", ".tox", ".pytest_cache", ".mypy_cache", ".svelte-kit",
    ".angular", ".parcel-cache", "tmp", "temp", "bak", "backup", "old",
    ".vs", "artifacts", "htmlcov", ".cloudflare", ".npm", ".pnpm-store",
    "third_party", "3rdparty", "extern", "external", "deps", "test-results",
    ".turbo", ".vercel", ".netlify", "playwright-report", "storybook-static",
    "migrations", "__snapshots__", "fixtures", "golden", "snapshots",
}

# never collect these file names at all: secrets, lockfiles, minified bundles
SKIP_NAME_RE = re.compile(
    r"(^package-lock\.json$|^pnpm-lock\.yaml$|^yarn\.lock$|\.lock$|"
    r"^\.env|\.pem$|\.key$|\.p12$|\.pfx$|\.crt$|^id_rsa|"
    r"\.min\.(js|css)$|\.bundle\.(js|css)$|\.chunk\.(js|css)$|"
    r"\.map$|\.d\.ts\.map$|^\.DS_Store|"
    r"\.g\.(cs|ts)$|\.designer\.cs$|\.generated\.|\.autogen\.|"
    r"^LICENSE|^COPYING|^CHANGELOG)", re.I)

# path fragments that mark generated / vendored content
SKIP_PATH_RE = re.compile(
    r"(/vendor/|/third_party/|/3rdparty/|/external/|/\.cache/|"
    r"/generated/|/gen/|/auto_gen/|/__generated__/|/tsconfig|"
    r"/\.yarn/|/jquery[.-]|/bootstrap[.-]|/fontawesome|/normalize\.css$|"
    r"/swagger|/openapi)", re.I)

MIN_BYTES = 100
MAX_BYTES = 300 * 1024
MAX_LINES = 20000
MAX_LINE_LEN = 500


def category_of(path):
    ext = os.path.splitext(path)[1].lower()
    return CAT_OF_EXT.get(ext), ext


# ---------------------------------------------------------------- file scanning

def iter_files(project_dir):
    for dp, dns, fns in os.walk(project_dir):
        dns[:] = [d for d in dns
                  if d.lower() not in SKIP_DIRS and not d.startswith(".")]
        for fn in fns:
            yield os.path.join(dp, fn)


def read_ok(path):
    """Return (bytes, n_lines) when the file is clean text worth collecting."""
    try:
        if os.path.getsize(path) > MAX_BYTES:
            return None
        with open(path, "rb") as f:
            b = f.read()
    except OSError:
        return None
    if len(b) < MIN_BYTES:
        return None
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "\x00" in s:
        return None
    lines = s.split("\n")
    if len(lines) > MAX_LINES:
        return None
    longest = 0
    for ln in lines:
        if len(ln) > MAX_LINE_LEN:
            return None            # minified / machine-dumped
        if len(ln) > longest:
            longest = len(ln)
    return b, len(lines)


def safe_dest(lib, cat, project, rel):
    rel = rel.replace("\\", "/").lstrip("/")
    dest = os.path.join(lib, cat, project, *rel.split("/"))
    if len(dest) <= 225:
        return dest
    # too deep for windows: keep the name, shorten the middle with a hash tag
    rel_dir = os.path.dirname(rel)
    tag = hashlib.sha1(rel_dir.encode("utf-8", "replace")).hexdigest()[:8]
    dest = os.path.join(lib, cat, project, "_deep_" + tag, os.path.basename(rel))
    return dest


# ---------------------------------------------------------------- main

def load_manifest(path):
    seen_src, seen_sha, rows = set(), set(), []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                seen_src.add(r["src"])
                seen_sha.add(r["sha1"])
    return seen_src, seen_sha, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--lib", default=LIB)
    ap.add_argument("--projects", default="", help="comma list, default all")
    ap.add_argument("--per-ext", type=int, default=12)
    ap.add_argument("--per-project", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--fresh", action="store_true", help="wipe manifest and redo")
    args = ap.parse_args()

    lib = args.lib
    os.makedirs(lib, exist_ok=True)
    man_path = os.path.join(lib, "MANIFEST.csv")
    if args.fresh and os.path.isfile(man_path):
        os.remove(man_path)
    seen_src, seen_sha, rows = load_manifest(man_path)

    projects = [n for n in sorted(os.listdir(args.root))
                if os.path.isdir(os.path.join(args.root, n))]
    if args.projects:
        want = {s.strip() for s in args.projects.split(",") if s.strip()}
        projects = [p for p in projects if p in want]

    rnd = random.Random(args.seed)
    added, skipped, per_cat, per_proj = 0, 0, collections.Counter(), collections.Counter()
    log = []

    for project in projects:
        pdir = os.path.join(args.root, project)
        # bucket this project's candidates by extension
        by_ext = collections.defaultdict(list)
        for path in iter_files(pdir):
            rel = os.path.relpath(path, pdir)
            if SKIP_NAME_RE.search(os.path.basename(path)):
                continue
            if SKIP_PATH_RE.search("/" + rel.replace("\\", "/")):
                continue
            cat, ext = category_of(path)
            if cat is None:
                continue
            by_ext[ext].append(path)

        picked = []
        for ext, paths in by_ext.items():
            cap = EXT_CAP.get(ext, args.per_ext)
            if cap <= 0:
                continue
            paths.sort()
            if len(paths) > cap:
                rnd.shuffle(paths)
                paths = paths[:cap]
            picked.append((ext, paths))

        # priority order, then round-robin so a project does not fill up
        # with one language before the others get a chance
        picked.sort(key=lambda t: EXT_CAP.get(t[0], args.per_ext))
        max_len = max((len(p) for _, p in picked), default=0)
        ordered = []
        for i in range(max_len):
            for ext, paths in picked:
                if i < len(paths):
                    ordered.append(paths[i])

        n_proj = 0
        cat_seen = collections.Counter()
        for path in ordered:
            if n_proj >= args.per_project:
                break
            rel = os.path.relpath(path, pdir)
            cat, ext = category_of(path)
            if cat_seen[cat] >= CAT_CAP.get(cat, 10 ** 9):
                continue
            if path in seen_src:
                skipped += 1
                continue
            got = read_ok(path)
            if got is None:
                continue
            b, n_lines = got
            sha = hashlib.sha1(b).hexdigest()
            if sha in seen_sha:
                skipped += 1
                continue
            dest = safe_dest(lib, cat, project, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                with open(dest, "wb") as f:
                    f.write(b)
            except OSError:
                continue
            seen_src.add(path)
            seen_sha.add(sha)
            rows.append({
                "src": path, "project": project, "category": cat, "ext": ext,
                "bytes": len(b), "lines": n_lines, "sha1": sha,
                "dest": os.path.relpath(dest, lib).replace("\\", "/"),
            })
            added += 1
            n_proj += 1
            cat_seen[cat] += 1
            per_cat[cat] += 1
            per_proj[project] += 1
        log.append("[%s] added=%d (candidates=%d)" % (project, n_proj, len(ordered)))

    with open(man_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["src", "project", "category", "ext",
                                          "bytes", "lines", "sha1", "dest"])
        w.writeheader()
        w.writerows(rows)

    stats = {
        "root": args.root,
        "total_files": len(rows),
        "total_mb": round(sum(int(r["bytes"]) for r in rows) / 1048576.0, 2),
        "by_category": dict(sorted(per_cat.items())),
        "by_project": dict(sorted(per_proj.items())),
        "manifest": man_path,
    }
    with open(os.path.join(lib, "STATS.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    with open(os.path.join(lib, "_collect.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print("added=%d skipped=%d total=%d (%.2f MB)"
          % (added, skipped, len(rows), stats["total_mb"]))
    print(json.dumps(stats["by_category"], ensure_ascii=False))


if __name__ == "__main__":
    main()
