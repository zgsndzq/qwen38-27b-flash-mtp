#!/usr/bin/env bash
# Three-way merge of upstream llama.cpp commits into the code/ tree.
# The tree was reordered (ggml/ -> code/ggml/), so cherry-pick and rebase are
# permanently unusable: the only workable path is a path-mapped merge-file.
#
#   tools/sync-upstream.sh [ref] [--dry-run]
#
# base comes from .upstream-base (the upstream commit we last synced to).
# A file counts as "ours changed" only when it differs from base, so a clean
# file is replaced by theirs without any merge.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

REF="${1:-prism/prism}"
[ "${1:-}" = "--dry-run" ] && REF="prism/prism"
DRY=0
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done

PREFIX="code"
BASE_FILE=".upstream-base"
SKIP_FILE=".upstream-skip"

[ -f "$BASE_FILE" ] || { echo "missing $BASE_FILE (the upstream commit we are synced to)"; exit 1; }
BASE=$(tr -d '[:space:]' < "$BASE_FILE")
git cat-file -e "$BASE^{commit}" 2>/dev/null || { echo "base $BASE is not in the local object store; git fetch first"; exit 1; }
git cat-file -e "$REF^{commit}" 2>/dev/null || { echo "ref $REF is not in the local object store; git fetch first"; exit 1; }

map() { case "$1" in ggml/*|src/*|common/*|tests/*|tools/*|examples/*|include/*|cmake/*|app/*|vendor/*|benches/*|pocs/*|models/*|conversion/*|scripts/*|ci/*|docs/*|grammars/*|gguf-py/*|requirements/*|licenses/*) printf '%s/%s' "$PREFIX" "$1";; *) printf '%s/%s' "$PREFIX" "$1";; esac; }

STAMP=$(date +%Y%m%d-%H%M%S)
SAFE_REF=$(printf '%s' "$REF" | tr '/' '-')
REPORT="logs/sync-$SAFE_REF-$STAMP.md"
BACKUP="logs/sync-backup-$SAFE_REF-$STAMP"
mkdir -p logs

if [ "$DRY" = "0" ]; then mkdir -p "$BACKUP"; fi

ADDED=0 REUSED=0 MERGED=0 CONFLICTED=0 FAILED=0
CONFLICT_FILES=()
{
  echo "# upstream sync $BASE..$REF"
  echo
  echo "| file | result | conflict blocks |"
  echo "|---|---|---|"
} > "$REPORT"

while IFS= read -r f; do
  [ -z "$f" ] && continue
  if [ -f "$SKIP_FILE" ] && grep -qxF "$f" "$SKIP_FILE"; then
    printf '| `%s` | skipped by %s | - |\n' "$f" "$SKIP_FILE" >> "$REPORT"
    continue
  fi
  dst=$(map "$f")
  theirs_state=$(git cat-file -e "$REF:$f" 2>/dev/null && echo present || echo deleted)
  base_state=$(git cat-file -e "$BASE:$f" 2>/dev/null && echo present || echo absent)

  # Upstream deleted the file: only remove it when we never touched it.
  if [ "$theirs_state" = "deleted" ]; then
    if [ -f "$dst" ] && [ "$base_state" = "present" ] && [ "$(git show "$BASE:$f" | sha256sum)" = "$(sha256sum < "$dst")" ]; then
      [ "$DRY" = "0" ] && cp "$dst" "$BACKUP/" && rm -f "$dst"
      printf '| `%s` | removed (upstream) | - |\n' "$f" >> "$REPORT"
      CONFLICTED=$((CONFLICTED+1)); CONFLICT_FILES+=("$f (deleted upstream, local edits kept)")
    else
      printf '| `%s` | upstream deleted, local keeps it | - |\n' "$f" >> "$REPORT"
      CONFLICT_FILES+=("$f (upstream deleted, local edits kept)")
    fi
    continue
  fi

  mkdir -p "$(dirname "$dst")"
  # New upstream file or local copy absent: take theirs as-is.
  if [ ! -f "$dst" ]; then
    [ "$DRY" = "0" ] && git show "$REF:$f" > "$dst"
    printf '| `%s` | added | - |\n' "$f" >> "$REPORT"
    ADDED=$((ADDED+1)); continue
  fi
  # Binary files cannot merge.
  if git diff --numstat "$BASE" "$REF" -- "$f" 2>/dev/null | head -1 | grep -q $'^-\t-\t'; then
    [ "$DRY" = "0" ] && { cp "$dst" "$BACKUP/"; git show "$REF:$f" > "$dst"; }
    printf '| `%s` | binary, took theirs | - |\n' "$f" >> "$REPORT"
    MERGED=$((MERGED+1)); continue
  fi
  # Local file identical to base: no local work to protect.
  if [ "$base_state" = "present" ] && [ "$(git show "$BASE:$f" | sha256sum)" = "$(sha256sum < "$dst")" ]; then
    [ "$DRY" = "0" ] && git show "$REF:$f" > "$dst"
    printf '| `%s` | took theirs (local clean) | - |\n' "$f" >> "$REPORT"
    REUSED=$((REUSED+1)); continue
  fi

  TMP=$(mktemp -d)
  git show "$BASE:$f" > "$TMP/base"
  git show "$REF:$f"  > "$TMP/theirs"
  cp "$dst" "$TMP/ours"
  git merge-file -L "ours/$dst" -L "base/$f" -L "theirs/$f" -p "$TMP/ours" "$TMP/base" "$TMP/theirs" > "$TMP/merged"
  rc=$?
  nconf=$(grep -c '^<<<<<<<' "$TMP/merged" 2>/dev/null || echo 0)
  if [ "$rc" = "0" ]; then
    [ "$DRY" = "0" ] && { cp "$dst" "$BACKUP/"; cp "$TMP/merged" "$dst"; }
    printf '| `%s` | merged clean | 0 |\n' "$f" >> "$REPORT"
    MERGED=$((MERGED+1))
  elif [ "$rc" -gt 0 ] 2>/dev/null; then
    [ "$DRY" = "0" ] && { cp "$dst" "$BACKUP/"; cp "$TMP/merged" "$dst"; }
    printf '| `%s` | **conflict** | %s |\n' "$f" "$nconf" >> "$REPORT"
    CONFLICTED=$((CONFLICTED+1)); CONFLICT_FILES+=("$f ($nconf blocks)")
  else
    printf '| `%s` | **merge failed** | - |\n' "$f" >> "$REPORT"
    FAILED=$((FAILED+1))
  fi
  rm -rf "$TMP"
done < <(git diff --name-only "$BASE" "$REF")

{
  echo
  echo "added=$ADDED took-theirs=$REUSED merged=$MERGED conflicts=$CONFLICTED failed=$FAILED"
  if [ ${#CONFLICT_FILES[@]} -gt 0 ]; then
    echo
    echo "## needs a human"
    for c in "${CONFLICT_FILES[@]}"; do echo "- $c"; done
    [ "$DRY" = "0" ] && echo && echo "originals in $BACKUP/, resolve the markers then: git checkout -- <file> to abort"
  fi
  echo
  echo "## audit: upstream lines that did not reach the tree"
  echo '```'
  python "$(dirname "$0")/verify-sync.py" "$BASE" "$REF"
  echo '```'
} | tee -a "$REPORT"

[ "$DRY" = "1" ] && echo "(dry run: nothing written)"
exit 0
