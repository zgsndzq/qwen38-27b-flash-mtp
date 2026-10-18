# llama.cpp-qwen38-flash — llama.cpp fork for Qwen3.8-27B-Flash-MTP (PTQ1_0 + MTP)

A fork of [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (`prism`
branch), which is itself a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
Built for **Qwen3.8-27B-Flash-MTP** — Qwen3.8-27B in ternary PTQ1_0 — on an **NVIDIA Ada
(sm_89)** GPU, with speculative decoding driven by an **MTP draft head**.

Upstream baseline: `e8fd209c9` (`grammar: allow schema property order (#347)`).
Everything upstream has is carried over — this fork adds speed work on top of it.
The unmodified upstream README is kept as `README.upstream.md`.

## How the product is split

This repository is one half. The scripts a user actually touches are deliberately **not** here,
because they need to be reviewable and patchable — they used to live only inside a release zip.

| Where | What |
|---|---|
| **this repo** | the engine: PTQ1_0 kernels, MTP draft context, the speed work, the build scripts |
| **[qwen38-27b-flash-mtp](https://github.com/zgsndzq/qwen38-27b-flash-mtp)** | what you launch: `llmserver.bat` / `llmserver.sh`, **`features.ini`**, `llmchat\`, `maps\` |
| [Releases](https://github.com/zgsndzq/llama.cpp-qwen38-flash/releases) | prebuilt engine binaries (`build\bin` + `build\cuda`) |
| [Hugging Face](https://huggingface.co/SuperLogic/Qwen3.8-27B-Flash-MTP-GGUF) | the weights |

> ⚠️ **`features.ini` is not in this repository, and it is not compiled into the binary.**
> `llama-server` has no built-in default path for it — it is picked up at runtime through
> `--features-file features.ini`. A binary built from this source **without** that file behaves
> like upstream, not like us. Which switches we ship is decided in the runtime repo, not here.

The two repositories are meant to sit nested, which is the layout the build scripts assume
(`_build_q8.bat` writes to `..\build\`):

```bat
git clone https://github.com/zgsndzq/qwen38-27b-flash-mtp
cd qwen38-27b-flash-mtp
git clone https://github.com/zgsndzq/llama.cpp-qwen38-flash code
```

## Where are the weights

**Not here.** The gguf files live on Hugging Face:

### 👉 https://huggingface.co/SuperLogic/Qwen3.8-27B-Flash-MTP-GGUF

| File | Size | Needed for |
|---|---|---|
| `Qwen3.8-27B-Flash-MTP.gguf` | 5.79 GB | the model itself |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 0.59 GB | reading images |
| `SHA256SUMS` | — | verifying the two above |

Measured on the model card's reference machine (RTX 4060 Laptop, 8 GB): **70 t/s** with thinking
off, **100–200 t/s** in editing mode. Stock llama.cpp cannot load these weights — `PTQ1_0` and
the MTP draft context exist only in this fork.

## What is ours

Three layers. All of them are switchable at runtime, so none is hardcoded behaviour:

| Layer | What |
|---|---|
| `our delta: keep only the speed optimizations` | tiered KV cache (`--kv-vram-cells`), in-place quantized-KV flash attention, PTQ1_0 mat-vec up to 8 columns, `--spec-draft-window` / `-n-max-tail`, `GGML_CUDA_BATCH_INVARIANT`, `--reasoning-effort-*` server flags, `GGML_CUDA_OP_TIMING` |
| `port patch 0039` | `--spec-lookup-n-max`, a separate draft cap for the lookup drafters |
| `ada fork: local feature layer` | `--features-file` + `common/features.{h,cpp}`: reads an INI and applies it *before* argument parsing, so a switch can be flipped without recompiling. The default is always upstream behaviour |

## Build environment

| | |
|---|---|
| OS | Windows x64, or Linux (plain CMake — see below) |
| Compiler | MSVC (Visual Studio, any recent version) |
| CUDA | CUDA Toolkit with `nvcc` — **11.x or newer** |
| CMake | 3.21+ |
| Ninja | any |
| GPU | NVIDIA, compute capability **86 / 89** (the PTQ1_0 kernels are tuned for Ada) |

Nothing else is needed. All third-party code is vendored under `vendor/`.

## How to build

### Windows

1. Create your machine config from the template and edit it:

   ```bat
   copy build.local.example.bat build.local.bat
   ```

   Six values, all paths on *your* machine:

   | Variable | Meaning |
   |---|---|
   | `VCVARS` | `vcvars64.bat`, or a wrapper around it |
   | `CMAKE` | `cmake.exe` |
   | `NINJA` | `ninja.exe` |
   | `NVCC` | `nvcc.exe` from the CUDA Toolkit |
   | `CUDA_ARCH` | compute capability. **89 for Ada**, 86 for Ampere |
   | `JOBS` | parallelism. 4 keeps `nvcc` peak memory sane on a laptop GPU |

   The build scripts read this file and fail fast with a message if it is missing,
   so nothing is ever written into an unrelated directory.

2. Build:

   ```bat
   _build_q8.bat        rem configure + full build
   _build_inc.bat       rem incremental, only what ninja considers stale
   ```

   Binaries land in `..\build\src\bin\`, logs in `..\build\logs\`.
   The build never writes inside the source tree.

   `_build_clean.bat` wipes the build dir first. `_build_merge.bat` is `_build_q8.bat`
   without the FA pin. `_fa_reconfig.bat` re-applies the pin only.

3. That is all. The launcher in the runtime package accepts `..\build\src\bin\` directly, so
   there is usually nothing to copy — it prefers a local build over the prebuilt `..\build\bin\`.

### Linux

There is no `.sh` build script; use CMake directly. The two flags that are not optional are
`GGML_CUDA_FA_QUANTS=all` and an explicit `CMAKE_CUDA_ARCHITECTURES`.

```bash
cmake -B build/src -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_QUANTS=all \
  -DLLAMA_CURL=OFF
cmake --build build/src -j "$(nproc)" --target llama-server llama-bench llama-quantize
```

### `GGML_CUDA_FA_QUANTS`

The default build pins this to `all`, i.e. every `<K>-<V>` combination of the
flash-attention vector kernel is compiled. Without a matching kernel the runtime
**silently converts K and V to f16** — that changes both speed and numerics, and
the only symptom is one warning at startup. `all` removes that failure mode at the
cost of binary size. Shrink it again with:

```bat
set "FAQUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16"
```

### Line endings

`.gitattributes` pins `*.bat` to CRLF. Without it `core.autocrlf=input` normalises
batch files to LF, and **cmd mis-parses them** — a `rem` line swallows the command
that follows it.

## Running

`llama-server` takes the usual llama.cpp arguments. What matters for this model:

```bat
llama-server -m <model>.gguf ^
  -c 65536 -b 4096 -t 8 -ngl 99 -fa on ^
  -ctk q4_0 -ctv q4_0 ^
  --spec-type ngram-mod,draft-mtp-adaptive --spec-draft-n-max 4 ^
  --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 24 ^
  --spec-ngram-mod-n-max 64 --spec-lookup-n-max 64 ^
  --reasoning-budget -1 ^
  --features-file features.ini
```

`--features-file` resolves relative to the **working directory**, and `features.ini` in turn
holds `spec-draft-vocab-map = maps/shortlist_24k_mix.vocab.txt`, also relative. Run the server
from the package root, or turn both into absolute paths.

Notes:

- **KV cache is `q4_0`** — half the bytes per cell of `q8_0`, so the whole 64k
  window fits in the VRAM that `q8_0` needed for half of it.
- **`--features-file`** holds our switches. Precedence is command line >
  features file > default, so an explicit argument always wins.
- **`--reasoning-budget`** is an external cap on the thinking block, not a choice
  made for the model. `-1` means unlimited.
- **`GGML_CUDA_SHARED_POOL=1`** shares one transient pool between the target and
  the draft context, saving ~1.4 GB on an 8 GB card. Only safe with `--parallel 1`.
- **`GGML_CUDA_FA_SHORTQ_MMA=0`** keeps the in-place quantized-KV kernel for short
  verify batches. Setting it to 1 takes the upstream #189 detour, which was
  measured to cost decode speed on this model.
- The CUDA runtime dlls must sit next to the executable or on `PATH`; the packaged
  runtime carries them in `..\build\cuda\`.

## Layout

```
code\               this source tree
..\build\src\       CMake / Ninja output
..\build\bin\       runtime binaries (the prebuilt release lands here)
..\build\cuda\      CUDA runtime dlls
..\build\logs\      build logs
..\weights\         gguf models
..\features.ini     our runtime switches -- maintained in qwen38-27b-flash-mtp
```

## License

MIT, same as upstream llama.cpp.
