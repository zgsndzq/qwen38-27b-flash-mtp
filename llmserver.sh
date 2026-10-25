#!/usr/bin/env bash
# ===========================================================================
#  Qwen3.8-27B-Flash-MTP  :18080   --  agent default profile  (Linux)
#
#    thinking ON, effort=low, cap 2048   |  q3_0 K/V, 88K cells, CTX 128K
#    slot save/restore across restarts   |  MTP + ngram draft, width <= 4
#
#  Same profile as llmserver.bat -- change one, change the other.
#  Our switches live in features.ini [qwen]; numbers live here.
#  features.ini paths are relative to the working directory, hence the cd.
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

  cmake -B build/src -S code -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_QUANTS=all -DLLAMA_CURL=OFF
  cmake --build build/src -j --target llama-server llama-bench llama-quantize

  ARCH: 89 = Ada (4060/4070/4090), 86 = Ampere (3060/3080) -- do not let CMake
  guess, a wrong arch builds fine and then fails to run.
  GGML_CUDA_FA_QUANTS=all: the default list omits the quantised-KV combos this
  model uses, and the runner then silently converts K/V to f16.
EOF
  exit 1
fi

# shared libraries: the built tree keeps them next to the binary
export LD_LIBRARY_PATH="$PWD/$SERVER:$PWD/build/bin:${LD_LIBRARY_PATH:-}"

MODEL="Qwen3.8-27B-Flash-MTP.gguf"
CTX=131072

# ---- K/V and VRAM tier ----------------------------------------------------
#  CELLS comes from CTK/CTV, and GGML_CUDA_FA_CHUNK must equal it, or deep
#  prefill leaves the fast path (~6x slower).
#  q4_0 = 18 KB/cell -> 65536 cells (1152 MiB)   [best accuracy]
#  q3_0 = 14 KB/cell -> 90112 cells (1232 MiB)   [current, fits 88K in VRAM]
#  Same type on both sides: a mixed pair has no FA kernel and copies to f16.
#  Cells past CELLS stay in pinned host RAM; CTX is unaffected.
#  6 GB card: drop to q4_0 / 65536 or lower.
CTK=q3_0
CTV=q3_0
case "$CTK$CTV" in
  q4_0q4_0) CELLS=65536 ;;
  *)        CELLS=90112 ;;
esac
export GGML_CUDA_FA_CHUNK="$CELLS"

# ---- thinking -------------------------------------------------------------
#  EFFORT: template default, a request may override (minimal|low|medium|high|xhigh)
#  REBUDGET: the only real cap -- -1 unlimited, 0 stop now, N>0 cap
#  REBMSG: tokenised verbatim before the end-of-thinking tag; plain text only
#  max_tokens is the client's budget (llmchat), never set here
EFFORT=low
REBUDGET=2048
REBMSG=" Budget exceeded, answer directly."

# ---- draft ----------------------------------------------------------------
#  DRAFT_N: adaptive width ceiling (2 = fastest measured, 4 = best quality)
#  NMATCH: preceding tokens that must match before the ngram lookup fires
#  NGMIN: consecutive matches to fire; NGMAX / LOOKUP_N: per-draft caps
#  BENCH=1 drops the ngram drafter (it warms up, so timing needs it off)
DRAFT_N=4
NMATCH=24
NGMIN=24
NGMAX=24
LOOKUP_N=$NGMAX
BENCH=0
SPECTYPE=ngram-mod,draft-mtp-adaptive
if [ "$BENCH" = "1" ]; then SPECTYPE=draft-mtp-adaptive; fi

# a stale instance holds the port and its VRAM
pkill -f llama-server >/dev/null 2>&1 || true
sleep 2

# llmchat saves and restores whole conversations here (~18 KB/token, gitignored)
mkdir -p data/slots

args=(
  -m "$PWD/weights/$MODEL"
  --host 127.0.0.1 --port 18080 --parallel 1 --no-ui
  --slot-save-path "$PWD/data/slots"
  -c "$CTX" -b 4096 -t 8 -ngl 99 -fa on -ctk "$CTK" -ctv "$CTV"
  --kv-vram-cells "$CELLS"
  -rea on --reasoning-effort "$EFFORT"
  --reasoning-budget "$REBUDGET" --reasoning-budget-message "$REBMSG"
  --spec-type "$SPECTYPE" --spec-draft-n-max "$DRAFT_N"
  --spec-lookup-n-max "$LOOKUP_N" --spec-ngram-mod-n-min "$NGMIN"
  --spec-ngram-mod-n-max "$NGMAX"
  --spec-ngram-mod-n-match "$NMATCH"
  --features-file features.ini --features-section qwen
  --alias "Qwen3.8-27B-Flash-MTP"
)

# vision: 0 = off. 1 mounts it only if the projector file exists.
VISION=0
MMPROJ="$PWD/weights/mmproj-Qwen3.8-27B-Q8_0.gguf"
if [ "$VISION" = "1" ] && [ -f "$MMPROJ" ]; then
  args+=(--mmproj "$MMPROJ" --image-min-tokens 1024)
fi

exec "$SERVER/llama-server" "${args[@]}"
