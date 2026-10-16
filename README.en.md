# qwen38-27b-flash-mtp — Qwen3.8-27B-Flash-MTP runtime
A ternary-quantised (`PTQ1_0`) Bonsai-2 27B with a **built-in MTP self-speculative draft head**.
5.79 GB of weights, runs on **an 8 GB consumer card**. Measured **70 t/s** with thinking off,
**100–200 t/s** in editing mode.

The engine is our fork of `PrismML-Eng/llama.cpp` — **stock llama.cpp cannot load this model**
(neither the `PTQ1_0` tensor type nor the MTP draft context exists upstream).

> 中文: see `README.md`

---

## 1. What this repository is

**Scripts and configuration only.** The binaries and the weights are not in git; each has its
own distribution channel:

| What | Where | Size | Required |
|---|---|---|---|
| `llmserver.bat` / `llmserver.sh` / `features.ini` / `llmchat\` / `maps\` | **this repo** (`git clone`) | < 1 MB | yes |
| engine binaries (`build\bin` + `build\cuda`) | [Releases](https://github.com/zgsndzq/llama.cpp-qwen38-flash/releases) | 566 MB | unless you build |
| weights `weights\*.gguf` | [Hugging Face](https://huggingface.co/SuperLogic/Qwen3.8-27B-Flash-MTP-GGUF) | 5.79 GB + 0.59 GB | yes |
| engine source (`code\`) | [zgsndzq/llama.cpp-qwen38-flash](https://github.com/zgsndzq/llama.cpp-qwen38-flash) | 350 MB | only to build |

Why the split: **the scripts have to be reviewable.** They used to exist only inside a 566 MB
zip, where nobody could diff them or send a pull request.

---

## 2. Running it in three steps

```bat
git clone https://github.com/zgsndzq/qwen38-27b-flash-mtp
cd qwen38-27b-flash-mtp
```

1. Extract the release binary zip **into this folder** → you get `build\bin\` and `build\cuda\bin\`
2. Download the two gguf files from Hugging Face into `weights\` → see `weights\README.md`
3. **Double-click `llmserver.bat`**

When you see these two lines, it is up:

```
load_model: loaded multimodal model, '...\weights\mmproj-...gguf'
llama_server: listening on http://127.0.0.1:18080
```

Then hit the API directly:

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

or use the bundled client: double-click `llmchat\llmchat.bat`.

**Requirements**: an NVIDIA driver. **8 GB** of VRAM; on 6 GB drop `CTX` in `llmserver.bat`
from `65536` to `32768`, and `kv-vram-cells` in `features.ini` must be `<=` that.

---

## 3. Layout

```
qwen38-27b-flash-mtp\
├── llmserver.bat        <- double-click to start (Windows)
├── llmserver.sh         <- ./llmserver.sh (Linux, parameters aligned)
├── features.ini         <- ALL of our switches live here
├── llmchat\             <- command-line client (optional, pure Python, portable)
├── maps\                <- draft shortlist, don't delete (features.ini points at it)
├── code\                <- [not in git] engine source, `git clone` it in
├── build\               <- [not in git] binaries, release zip or your own build
└── weights\             <- [not in git] weights, from Hugging Face
```

The directory names `weights\` and `maps\` are **not** cosmetic — the launcher and `features.ini`
look them up by name.

---

## 4. Where the configuration lives, and why the ini is not optional

Two files, no overlap:

| File | Contents |
|---|---|
| `llmserver.bat` / `llmserver.sh` | plain upstream parameters: model, port, `-c` (context), `-b` (batch), `-ngl`, `-fa`, speculative types |
| `features.ini` | **switches that exist only in this fork**; the engine reads the file itself (`--features-file`) |

### How the three relate

```
llmserver.bat / .sh   -->  command-line arguments  -->  llama-server
                           (things upstream knows)

features.ini          -->  --features-file         -->  llama-server
                           (things only this fork knows)
```

**Precedence: explicit command-line argument > `features.ini` > engine built-in default.**

The rule for deciding where a switch belongs is one line:

> **Does stock `llama-server` know this parameter? Yes -> `llmserver`. No -> `features.ini`.**

Where each launcher variable ends up on the command line:

| Variable | Becomes | Kind |
|---|---|---|
| `MODEL` | `-m weights\<name>.gguf` | upstream |
| `CTX` | `-c` | upstream |
| `CTK` / `CTV` | `-ctk` / `-ctv` | upstream |
| `REBUDGET` | `--reasoning-budget` | upstream |
| `DRAFT_N` | `--spec-draft-n-max` | upstream |
| `NMATCH` / `NGMIN` / `NGMAX` / `LOOKUP_N` | `--spec-ngram-mod-n-match` / `-n-min` / `-n-max` / `--spec-lookup-n-max` | upstream |
| `VISION` / `MMPROJ` | `--mmproj` + `--image-min-tokens` | upstream |
| — | `--spec-type ngram-mod,draft-mtp-adaptive` | **ours**: `draft-mtp-adaptive`, hardcoded because it must sit inside `--spec-type` next to upstream's `ngram-mod` |
| — | `--features-file features.ini` | **ours**: the one argument that wires the ini in |

And the keys that live in `features.ini` only, never on the command line:

| ini key | Why not in the launcher |
|---|---|
| `spec-draft-sampling` | no such upstream parameter |
| `spec-draft-vocab-map` | no such upstream parameter |
| `kv-vram-cells` | no such upstream parameter |
| `kq-mask-packed` | no such upstream parameter |
| `GGML_CUDA_*` / `LLAMA_*` | these are **environment variables**, not command-line arguments |
| `sampler-fast-topk` | no such upstream parameter |

**`features.ini` is not compiled in.** There is no built-in default path for it — the engine picks
it up at runtime through `--features-file features.ini`. **Without it you get upstream behaviour,
not our tuning.** A few of the keys that exist nowhere else:

| Key | Value | Engine default | What it does |
|---|---|---|---|
| `spec-draft-sampling` | `probabilistic` | `greedy` | draft is truly sampled at the request temperature; raises acceptance without changing the output distribution |
| `spec-draft-vocab-map` | `maps/shortlist_24k_mix.vocab.txt` | none | 24k draft-only shortlist for the MTP head |
| `kv-vram-cells` | `32768` | none | tiered KV: how many cells stay in VRAM, the rest read from pinned RAM |
| `GGML_CUDA_FA_SHORTQ_MMA` | `0` | see below | disables the upstream "quantised V + short query to f16" path |
| `LLAMA_CKPT_REUSE` | `1` | see below | reuses prompt-checkpoint buffers, saves ~1.4 GB |

Keep `GGML_CUDA_FA_SHORTQ_MMA = 0`. Measured on this model: with it on, one MTP request collapses
into a repeated token and acceptance drops to 0.12; with it off, the identical request is clean
(319 characters) and faster.

Explicit command-line arguments **win over** `features.ini`.

---

## 5. Building it yourself

When you changed the engine, or your GPU is not compute capability 89: see `build\README.md`.
In one line: `git clone .../llama.cpp-qwen38-flash code` → edit `code\build.local.bat` → run `code\_build_q8.bat`.

The result lands in `build\src\bin\`. `llmserver.bat` **accepts both locations and prefers your own
build** — no manual copying.

---

## 6. On Linux

The `.bat` is Windows-only, but **the stack itself is cross-platform** — the real work is one
`llama-server` plus a list of arguments. On Linux that is **`llmserver.sh`** in the same folder,
with parameters **aligned one-to-one** with `llmserver.bat`:

```bash
git clone https://github.com/zgsndzq/qwen38-27b-flash-mtp
cd qwen38-27b-flash-mtp

# there is no prebuilt Linux binary, so you build it
git clone https://github.com/zgsndzq/llama.cpp-qwen38-flash code
cmake -B build/src -S code -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_QUANTS=all \
  -DLLAMA_CURL=OFF
cmake --build build/src -j "$(nproc)" --target llama-server llama-bench llama-quantize

# weights come from Hugging Face into weights/ exactly as before
./llmserver.sh
```

Three things differ from Windows, everything else is identical:

| | Windows | Linux |
|---|---|---|
| start | double-click `llmserver.bat` | `chmod +x llmserver.sh` then `./llmserver.sh` |
| kill stale process | `taskkill /F /IM llama-server.exe` | `pkill -f llama-server` |
| library search path | `PATH` + `build\cuda\bin` | `LD_LIBRARY_PATH` (already set by the script) |
| build script | `code\_build_q8.bat` (fill in `build.local.bat`) | plain `cmake` (command above). There is **no** `.sh` build script |

⚠️ **On Linux, omitting `GGML_CUDA_FA_QUANTS=all` costs speed silently.** The CMake default list
does not contain the quantised K/V combinations this model uses; when the runner cannot find a
kernel it **quietly converts K/V to f16** (the `(slow)` path in the code) — no error, just slower
and numerically different. `-DGGML_CUDA_FA_QUANTS=all` is the only way to close that hole.

⚠️ **Do not let CMake guess `CUDA_ARCH`.** A wrong compute capability **builds fine and fails at
runtime**: Ada (4060 / 4070 / 4090) = `89`, Ampere (3060 / 3080) = `86`.

⚠️ **`-t 8` is a laptop core count — change it** for your machine (e.g. `-t $(nproc)`).

> `llmserver.sh` does nothing except `cd` to its own folder. You can just type the arguments out
> by hand and turn `--features-file features.ini` into an absolute path — the ini and the launcher
> are separable, as long as you also fix the relative `maps/...` path inside `features.ini`.

---

## 7. Gotchas

1. **Speed drops by an order of magnitude** — almost always KV spilling out of VRAM.
   `CTK/CTV=q4_0` with `CTX=65536` is one matched pair; switch to `q8_0` and `CTX` must be halved,
   because bytes per cell double.
2. **Output turns into repeated garbage** — check that `GGML_CUDA_FA_SHORTQ_MMA` is not enabled
   (the default `0` is correct).
3. **Editing a `.bat` produces a wall of `'xxx' is not recognized`** — the file was saved with
   **LF** line endings. cmd needs **CRLF**, and comments inside a `.bat` must be **pure ASCII**
   (non-ASCII comments under cmd(936) shift and eat the next line).
4. **Port already in use** — the launcher kills a stale `llama-server.exe` first on purpose,
   otherwise the VRAM is never released. To run two instances, comment those two lines out.
5. **Image requests rejected** — you need `weights\mmproj-*.gguf`. When it is missing the launcher
   disables vision automatically and prints a warning.
6. **`llama-server.exe not found`** — `build\` is empty. Either extract the release zip or build.

---

## 8. Provenance and licence

**Licence: Apache 2.0**

The weights are a **derivative work**, not an original model. Layer by layer, all Apache 2.0:

| Layer | Source | What was done |
|---|---|---|
| 1 | `Qwen/Qwen3.8-27B`, Copyright 2026 Alibaba Cloud | the base model, 27B hybrid attention (~75% linear + 25% full) |
| 2 | `prism-ml/Ternary-Bonsai-2-27B-gguf`, PrismML | matrix weights converted to ternary (`PTQ1_0`) |
| 3 | `ukisai/Swift-Bonsai-2-GGUF`, UkisAI | reasoning-efficiency fine-tune (~39.8% fewer thinking tokens, benchmarks held) |
| 4 | MTP draft head, from `ProCreations` | see `graft.donor.name` in the shipped gguf |
| 5 | **this project** | merged layers 3 and 4 into one GGUF and repacked it at precision. **No retraining.** |

The engine (`code\`, a fork of `PrismML-Eng/llama.cpp`) is **MIT**.

**Required attribution:** *"Created using Bonsai by Prism ML."*

Full statements: `LICENSE` (Apache 2.0) and `NOTICE` (per-layer copyright and changes) at the root.
