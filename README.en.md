# Qwen3.8-27B-Flash-MTP — runtime package

A 27B code model, ternary-quantised (`PTQ1_0`), with an MTP self-speculative draft head
**packed into the same gguf**. 5.79 GB of weights, fully resident on an **8 GB** consumer card.

The engine is a fork of `PrismML-Eng/llama.cpp` — upstream llama.cpp **cannot run this
model**: the `PTQ1_0` tensor type and the MTP draft context only exist here.

> 中文：见 `README.md`

---

## 1. Three places to download from

| Where | What | Size | Needed? |
|---|---|---|---|
| **Hugging Face** → `SuperLogic/Qwen3.8-27B-Flash-MTP-GGUF` | weights | 5.79 GB + 0.59 GB | **yes** |
| **GitHub Releases** → `qwen38-27b-flash-mtp-win64-<commit>.zip` | prebuilt binaries + CUDA runtime + this package | ≈600 MB | if you do not want to build |
| **GitHub source** → `zgsndzq/llama.cpp-qwen38-flash` | engine source | 400 MB | only if you build |
| This repo → `zgsndzq/qwen38-27b-flash-mtp` | launcher, `features.ini`, client, source | — | what you are reading |

A release zip only matches the engine revision it was built from; replace the whole
package, not individual dlls.

---

## 2. Layout

```
qwen38-27b-flash-mtp\
├── build\
│   ├── bin\            llama-server.exe + every dll (in the zip)
│   └── cuda\bin\       CUDA runtime (in the zip, no CUDA Toolkit needed)
├── weights\            ★ put the gguf here
│   ├── Qwen3.8-27B-Flash-MTP.gguf        main model, 5.79 GB
│   └── mmproj-Qwen3.8-27B-Q8_0.gguf      vision, 0.59 GB, only for images
├── maps\               24k draft shortlist (shipped, do not delete)
├── llmchat\            command-line client (optional)
├── code\               engine source (for building)
├── features.ini        this fork's switches
└── llmserver.bat       double-click to start
```

The `weights\` directory name is fixed (`%~dp0weights\` in the launcher); the file names
must match `MODEL=` in `llmserver.bat`.

---

## 3. Running it

No build step. Double-click `llmserver.bat`; these two lines mean it works:

```
load_model: initializing, n_slots = 1, n_ctx_slot = 131072
llama_server: listening on http://127.0.0.1:18080
```

Then:

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

Or use `llmchat\llmchat.bat` — it keeps the last conversation across restarts (`clear`
starts over).

**Requirements**: an NVIDIA driver, **6 GB VRAM minimum**. Defaults are `CTX=131072`
and `kv-vram-cells=32768` (half the KV in VRAM, half in host RAM). On a 6 GB card lower
`CTX` to `32768` and keep `kv-vram-cells ≤ CTX`.

**Baseline speed** (RTX 4060 Laptop, 8 GB, thinking off):

| Workload | Speed | Why |
|---|---|---|
| Code generation (ngram drafts hit often) | **70 t/s** | acceptance 0.65, 4.2 tokens per round |
| Editing an existing file (mostly verbatim) | **170 t/s** steady, 260 peak | acceptance 0.85, **41** tokens per round |
| Open-ended prose | 30–40 t/s | acceptance ~0.3, one token per round |
| prefill | **450–500 t/s** shallow → 265 at 32K → 213 at 64K | |

Decode is accounted per *round*: one forward pass reads all 5.79 GB of weights, and a
laptop 4060 moves ~215 GB/s, so a round costs **28–29 ms**.

- **Code generation**: 4.2 tokens per round ⇒ a bandwidth bound of ~145 t/s; the measured
  70 t/s is **48 %** of it, with the gap in launch overhead, the 5-row verify and sampling.
- **Editing**: 41 tokens per round, so the 29 ms of bandwidth is not the limit — verifying
  42 rows is: 2 × 27B × 42 ≈ 2.3 TFLOP in 245 ms ≈ **9 TFLOPS effective**. Speeding this
  up is about multi-row matmul kernel efficiency, not bandwidth.

---

## 4. Building it yourself

```bat
git clone https://github.com/zgsndzq/llama.cpp-qwen38-flash.git
cd llama.cpp-qwen38-flash
copy build.local.example.bat build.local.bat
_build_q8.bat
```

`build.local.bat` needs **no editing**: the template finds MSVC through `vswhere` (any
VS 2022 edition, BuildTools included), CMake/Ninja from PATH, a standalone install or
`pip install cmake ninja`, CUDA from `CUDA_PATH`, the standard install dirs or
`pip install nvidia-cuda-nvcc`, and builds for the local GPU by default
(`CUDA_ARCH=native`: 3060 → `86`, 4060/4070 → `89`).

Install at least one of: **VS2022 (MSVC), CUDA Toolkit, CMake, Ninja**.

Output lands in `build\src\bin\`; copy it over `build\bin\`. Log:
`build\logs\_build_q8.log`, last line `DONE`.

> After moving or renaming the tree the first build is a **full rebuild**: CMake bakes
> absolute paths into its cache and `_cache_guard.bat` wipes it. That is CMake, not a
> misconfiguration.

---

## 5. Configuration

| File | Holds |
|---|---|
| `llmserver.bat` | plain engine flags: model, port, `-c`, `-b`, `-ngl`, `-fa`, speculation, `--slot-save-path` |
| `features.ini` | switches only this fork has |

Worth knowing in `features.ini`:

- `kv-vram-cells` — KV cells kept in VRAM, the rest is read from host RAM. Must be ≤ `CTX`.
- `GGML_CUDA_FA_SPARSE` — page-sparse reads of the host tail; only pays off past ~100K.
- `GGML_CUDA_FA_CHUNK` — chunked deep prefill; memory scales with the chunk, not the context.
- `kq-mask-packed` — **commented out by default**: at 128K its f16 expansion costs 40 % of
  prefill (265 vs 430 t/s measured), for about 5 % of decode.
- `GGML_CUDA_FA_SHORTQ_MMA = 0` — **leave off**; enabling it produced repeated garbage.

Command-line flags override `features.ini`.

---

## 6. Pitfalls

1. **A `.bat` that suddenly reports `'xxx' is not recognized`** → it was saved with **LF**
   line endings. cmd's `^` continuation needs **CRLF**; keep batch comments ASCII.
2. **Speed drops by an order of magnitude** → KV spilled out of VRAM. `CTK/CTV=q4_0` with
   `CTX=131072` is one matched pair; switching to `q8_0` means halving `CTX`.
3. **Repeated garbage output** → check `GGML_CUDA_FA_SHORTQ_MMA` (0 is correct).
4. **Port busy** → the launcher kills a stale `llama-server.exe` on purpose, otherwise VRAM
   is not released.
5. **Image requests rejected** → `weights\mmproj-*.gguf` is missing; the launcher then
   disables vision and prints a warning.
6. **Resuming a conversation after a restart** → the server needs `--slot-save-path`
   (already set); `llmchat` saves and restores automatically, `clear` starts over.

---

## 7. Provenance and licence

**License: Apache 2.0**

The weights are a **derivative work**, layer by layer (all Apache 2.0):

| Layer | Source | What was done |
|---|---|---|
| 1 | `Qwen/Qwen3.8-27B`, Copyright 2026 Alibaba Cloud | the base model, 27B hybrid attention |
| 2 | `prism-ml/Ternary-Bonsai-2-27B-gguf`, PrismML | matrices converted to ternary (`PTQ1_0`) |
| 3 | `ukisai/Swift-Bonsai-2-GGUF`, UkisAI | reasoning-efficiency fine-tune |
| 4 | MTP draft head from `ProCreations` | see `graft.donor.name` in the gguf |
| 5 | **this project** | layers 3 and 4 merged and repacked, **no retraining** |

The engine (`code\`, a fork of `PrismML-Eng/llama.cpp`) is **MIT**.

**Required attribution:** *"Created using Bonsai by Prism ML."*

Full text in `LICENSE` (Apache 2.0) and `NOTICE` (per-layer copyright and changes).
