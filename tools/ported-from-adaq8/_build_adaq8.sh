#!/usr/bin/env bash
# 复现 _build_q8.ps1 的 portable toolchain 环境，然后用 ninja 构建。
# 用法: bash tools/_build_adaq8.sh [-n] [target...]
set -u

export TOOLS="F:/super/tools"
export MSVC="F:/super/tools/msvc/VC/Tools/MSVC/14.44.35207"
export SDK="F:/super/tools/sdk/c"
export SDKVER="10.0.22621.0"
export CUDA="F:/super/tools/cuda"

export PATH="/f/super/tools/msvc/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64:/f/super/tools/sdk/c/bin/10.0.22621.0/x64:/f/super/tools/cuda/bin:/f/super/tools/ninja:$PATH"
export INCLUDE="$MSVC/include;$SDK/Include/$SDKVER/um;$SDK/Include/$SDKVER/shared;$SDK/Include/$SDKVER/ucrt;$SDK/Include/$SDKVER/winrt"
export LIB="$MSVC/lib/x64;$SDK/ucrt/x64;$SDK/um/x64"
export CUDA_PATH="$CUDA"
export CUDA_HOME="$CUDA"
export VCToolsInstallDir="$MSVC/"

cd /f/super/llm/forks/ada-q8 || exit 3

DRY=""
if [ "${1:-}" = "-n" ]; then DRY="-n"; shift; fi
TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then TARGETS=(llama-server); fi

# The embedded server web UI is dead weight for us. With LLAMA_BUILD_UI=ON every configure runs
# `npm run build` (~65 s, and it FAILS in this offline setup), then tries two Hugging Face
# downloads that each die on a timeout, before ninja has compiled anything. One-time switch,
# guarded so it only reconfigures once and later runs pay nothing.
CMAKE="/c/Program Files/CMake/bin/cmake.exe"
if ! grep -q "^LLAMA_BUILD_UI:BOOL=OFF" build-q8/CMakeCache.txt 2>/dev/null; then
  echo "== cmake reconfigure: LLAMA_BUILD_UI=OFF (no embedded web UI, no npm, no HF fetch) =="
  "$CMAKE" -S . -B build-q8 -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF || exit 4
fi

echo "== ninja $DRY -C build-q8 -j 6 ${TARGETS[*]} =="
"$TOOLS/ninja/ninja.exe" $DRY -C build-q8 -j 6 "${TARGETS[@]}"
rc=$?
echo "NINJA_EXIT=$rc"
exit $rc
