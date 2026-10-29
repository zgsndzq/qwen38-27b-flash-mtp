#!/usr/bin/env bash
# port-palmer.sh — 把 Palmer fork 的单个提交按路径映射做三方合并到 code/ 树。
# 用法: tools/port-palmer.sh <ref> <commit> [commit...]
#   base = commit^:$f, theirs = commit:$f, ours = code/$f（缺失则直接取 theirs）
set -euo pipefail
cd "$(dirname "$0")/.."

REF="${1:?usage: port-palmer.sh <ref> <commit>...}"; shift
TS=$(date +%Y%m%d-%H%M%S)
BK="logs/port-backup-$TS"
mkdir -p "$BK"

tot=0; added=0; merged=0; conf=0
declare -a CONF_FILES=()

for c in "$@"; do
  echo "=== $c $(git log -1 --format='%s' "$c" | cut -c1-68)"
  for f in $(git show --pretty=format: --name-only "$c" | grep -v '^$'); do
    dst="code/$f"
    tot=$((tot+1))
    if [ ! -e "$dst" ]; then
      mkdir -p "$(dirname "$dst")"
      git show "$c:$f" > "$dst"
      echo "    +  $f (new)"
      added=$((added+1)); continue
    fi
    cp "$dst" "$BK/$(echo "$f" | tr '/' '_')"
    tmp=$(mktemp -d)
    git show "$c^:$f" > "$tmp/base" 2>/dev/null || : > "$tmp/base"
    git show "$c:$f"  > "$tmp/theirs"
    cp "$dst" "$tmp/ours"
    set +e
    git merge-file -L "ours" -L "base" -L "theirs" -p "$tmp/ours" "$tmp/base" "$tmp/theirs" > "$tmp/merged"
    rc=$?
    set -e
    if [ $rc -eq 0 ]; then
      cp "$tmp/merged" "$dst"; echo "    =  $f (clean)"; merged=$((merged+1))
    else
      cp "$tmp/merged" "$dst"; echo "    !  $f ($rc conflict blocks)"
      conf=$((conf+1)); CONF_FILES+=("$dst")
    fi
    rm -rf "$tmp"
  done
done

echo
echo "files=$tot new=$added clean=$merged conflicts=$conf"
if [ ${#CONF_FILES[@]} -gt 0 ]; then
  echo "备份: $BK"
  for f in "${CONF_FILES[@]}"; do echo "  - $f"; done
fi
