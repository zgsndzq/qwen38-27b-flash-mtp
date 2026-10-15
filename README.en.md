# ada-ternary — Ternary-Bonsai-2-27B runtime package

A ternary-quantised (`PTQ1_0`) Bonsai-2 27B with an **integrated MTP draft head**.
5.8 GB of weights, runs on an **8 GB consumer GPU**, measured 60–80 t/s.

The engine is a fork of `PrismML-Eng/llama.cpp` — **stock llama.cpp cannot run this model**.
Both the `PTQ1_0` tensor type and the MTP draft context exist only in this fork.

> 中文版见 `README.md`

---

## 1. Three places to download from

This is not one archive. It is three separate downloads:

| From | What | Size | Required |
|---|---|---|---|
| **Hugging Face** → `Ternary-Bonsai-2-27B-PTQ1_0-GGUF` | weights `*.gguf` | 5.8 GB (one model) | **yes** |
| **GitHub Releases** → `ada-ternary-win64-sm89-<commit>.zip` | prebuilt binaries + CUDA runtime + this package | 568 MB | unless you build it yourself |
| **GitHub source** → `llama.cpp-ada-ternary` | engine source | 350 MB | only if you want to build |

**Required engine commit: `b3e7c012c` or newer on a `sync-up-*` branch.** A mismatch fails at load time.

---

## 2. Layout

Unzip the release archive and drop the weights in:

```
ada-ternary\
├── build\
│   ├── bin\            llama-server.exe + all dlls (included in the zip)
│   └── cuda\bin\       CUDA runtime  (included - you do NOT need the CUDA Toolkit)
├── weights\            ★ put the gguf files here (create it yourself)
│   ├── Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf     main model
│   └── mmproj-Qwen3.8-27B-Q8_0.gguf                vision projector, only for images
├── maps\               draft shortlist (included, do not delete)
├── llmchat\            command-line client (optional)
├── features.ini        every switch this fork adds lives here
└── llmserver.bat       double-click to start
```

The `weights\` directory name is fixed — the launcher looks for `%~dp0weights\`.
The model filename must match `MODEL=` in `llmserver.bat`.

---

## 3. Run it

**No compiler needed.** Double-click `llmserver.bat`. You are up when you see:

```
load_model: loaded multimodal model, '...\weights\mmproj-...gguf'
llama_server: listening on http://127.0.0.1:18080
```

Then hit the API:

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

Or use the bundled client, `llmchat\llmchat.bat`.

**Requirements:** an NVIDIA driver. **8 GB VRAM** or more; on 6 GB drop `CTX` in
`llmserver.bat` from `65536` to `32768` (and keep `kv-vram-cells` in `features.ini` at or below it).

---

## 4. Build it yourself

For a different environment, or to modify the engine:

```bat
git clone https://github.com/professorpalmer/llama.cpp-ada-ternary.git
cd llama.cpp-ada-ternary
copy build.local.example.bat build.local.bat
notepad build.local.bat        :: fill in six values
_build_q8.bat
```

You need **VS2022 (MSVC) + CUDA Toolkit + CMake + Ninja**.

The six values in `build.local.bat` are documented in the template itself:

| Variable | What to put | Note |
|---|---|---|
| `VCVARS` | VS2022's `vcvars64.bat` | required — without it nvcc cannot find `cl.exe` |
| `CMAKE` / `NINJA` | paths to the two tools | |
| `NVCC` | `nvcc.exe` from your CUDA Toolkit | |
| `CUDA_ARCH` | **your GPU's compute capability** | ⚠️ the one value that can build fine and still not run: 4060/4070 = `89`, 3060 = `86`, 4090 = `89` |
| `JOBS` | parallelism, `4` on a laptop | nvcc will exhaust memory if set too high |

Output lands in `..\build\src\bin\`; copy it over `build\bin\` when done.
The log is `..\build\logs\_build_q8.log` — the last line reads `DONE` on success.

> Forgetting `build.local.bat` **does not fail silently** — the script writes
> `MISSING build.local.bat` and exits with code 9 instead of building somewhere unexpected.

---

## 5. Where the configuration lives

Two files, no overlap:

| File | Holds |
|---|---|
| `llmserver.bat` | parameters upstream understands: model, port, `-c` context, `-b` batch, `-ngl`, `-fa`, speculative type |
| `features.ini` | **switches that only exist in this fork**; the engine reads it via `--features-file` |

Two entries in `features.ini` worth knowing:

- `kv-vram-cells` — how many KV cells stay in VRAM (the rest reads pinned RAM).
  **Must be ≤ `CTX`.** Default `32768`.
- `GGML_CUDA_FA_SHORTQ_MMA = 0` — disables the upstream "quantised V with a short query goes to f16"
  detour. **Leave it off**: with it on, output degrades into repeated garbage in our measurements.

Explicit command-line arguments **take priority over** `features.ini`.

---

## 6. Traps

1. **Speed drops by an order of magnitude** → the KV cache spilled out of VRAM. `CTK/CTV=q4_0`
   with `CTX=65536` is one matched pair; going to `q8_0` doubles bytes per cell and requires
   halving `CTX`.
2. **Repeated garbage output** → check that `GGML_CUDA_FA_SHORTQ_MMA` is not enabled (0 is correct).
3. **A `.bat` you edited throws `'xxx' is not recognized`** → it was saved with **LF** line endings.
   cmd needs **CRLF**, and comments inside the batch files **must be pure ASCII** (non-ASCII bytes
   shift decoding under cmd's code page 936 and swallow the following line).
4. **Port already in use** → the launcher runs `taskkill` on any existing `llama-server.exe` first
   (deliberate; otherwise VRAM is not released). To run two instances, comment those lines out.
5. **Image requests rejected** → you need `weights\mmproj-*.gguf`. When the file is absent the
   launcher disables vision automatically and prints a warning.

---

## 7. Provenance and licence

**To be completed — required before upload.**

This is a **derivative work**: the trunk comes from `Swift-Bonsai-2` 27B (architecture `qwen35`).
What we produced is the **ternary quantisation and the MTP head conversion — no retraining**.
The licence follows the base model; we add no further restrictions.

See the model card on the weights repository for details.
