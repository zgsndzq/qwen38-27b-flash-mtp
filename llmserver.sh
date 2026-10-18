#!/usr/bin/env bash
# ===========================================================================
#  llama.cpp-qwen38-flash runtime launcher (Linux)
#  The Windows equivalent is llmserver.bat -- same parameters, same order.
#
#  Engine binary: build/src/bin (your own build) is preferred, build/bin is the
#  fallback. There is no prebuilt Linux binary, so in practice you build it.
#
#  All of OUR switches live in features.ini, which the engine reads itself
#  through --features-file. Only upstream-known parameters are set here.
#
#  NOTE: features.ini holds "spec-draft-vocab-map = maps/..." -- a path relative
#  to the working directory. That is why this script cds to its own folder.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SERVER=""
if [ -x build/src/bin/llama-server ]; then
  SERVER=build/src/bin
elif [ -x build/bin/llama-server ]; then
  SERVER=build/bin
else
  cat >&2 <<'EOF'
[error] llama-server not found. There is no prebuilt Linux binary -- build it:

  git clone https://github.com/zgsndzq/llama.cpp-qwen38-flash code
  cmake -B build/src -S code -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_QUANTS=all \
    -DLLAMA_CURL=OFF
  cmake --build build/src -j --target llama-server llama-bench llama-quantize

  CUDA_ARCH: 89 = Ada (4060/4070/4090), 86 = Ampere (3060/3080). Do not let
  CMake guess it -- a wrong arch builds fine and then fails to run.
  GGML_CUDA_FA_QUANTS=all matters: the default list omits the quantised-KV
  combos this model uses, and the runner then silently converts K/V to f16.
EOF
  exit 1
fi

# shared libraries: the built tree keeps them next to the binary
export LD_LIBRARY_PATH="$PWD/$SERVER:$PWD/build/bin:${LD_LIBRARY_PATH:-}"

MODEL="Qwen3.8-27B-Flash-MTP.gguf"
# context size; drop to 32768 on a 6 GB card (and kv-vram-cells in features.ini too)
CTX=65536
# KV precision. cells kept in VRAM = features.ini kv-vram-cells (must be <= CTX).
CTK=q4_0
CTV=q4_0
# thinking block token cap. -1 = unlimited, 0 = end at once, N>0 = hard cap.
REBUDGET=-1
# draft width ceiling, the adaptive controller stays at or below it.
DRAFT_N=4
# ngram lookup drafter
NMATCH=24
NGMIN=24
NGMAX=64
LOOKUP_N=$NGMAX

# a stale instance holds the port and its VRAM, so drop it before binding again
pkill -f llama-server >/dev/null 2>&1 || true
sleep 2

args=(
  -m "$PWD/weights/$MODEL"
  --host 127.0.0.1 --port 18080 --parallel 1 -rea off --no-ui
  -c "$CTX" -b 4096 -t 8 -ngl 99 -fa on -ctk "$CTK" -ctv "$CTV"
  --reasoning-budget "$REBUDGET"
  --spec-type ngram-mod,draft-mtp-adaptive --spec-draft-n-max "$DRAFT_N"
  --spec-lookup-n-max "$LOOKUP_N" --spec-ngram-mod-n-min "$NGMIN"
  --spec-ngram-mod-n-max "$NGMAX"
  --spec-ngram-mod-n-match "$NMATCH"
  --features-file features.ini
  --alias "Qwen3.8-27B-Flash-MTP"
)

# vision projector. VISION=1 mounts it, but only when the file is actually there.
VISION=1
MMPROJ="$PWD/weights/mmproj-Qwen3.8-27B-Q8_0.gguf"
if [ "$VISION" = "1" ]; then
  if [ -f "$MMPROJ" ]; then
    args+=(--mmproj "$MMPROJ" --image-min-tokens 1024)
  else
    echo "[warn] $MMPROJ not found - starting without vision" >&2
  fi
fi

exec "$SERVER/llama-server" "${args[@]}"
