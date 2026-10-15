# qwen27b — Ternary-Bonsai-2-27B (PTQ1_0 + MTP) standalone runtime package

A **self-contained** inference package: everything needed to run `llmserver.bat` lives
inside this folder. Copy the folder to any drive on any Windows x64 machine and it runs
**with just an NVIDIA driver** — no CUDA Toolkit needed, the CUDA runtime ships with it.

There are **no absolute paths inside the package**; the launcher addresses everything
relative to `%~dp0`.

---

## Startup: two files

| File | Role |
|---|---|
| `llmserver.bat` | **The only launcher.** Holds plain engine arguments (`-c -b -t -ngl -fa -rea --spec-type --alias`). |
| `features.ini` | **The only feature entry point.** Holds everything the upstream engine does not know about: our fork's switches (`spec-draft-*`, `kv-vram-cells`, `kq-mask-packed`, ...) and all `GGML_CUDA_*` / `LLAMA_*` environment variables. |

The launcher passes `--features-file "%~dp0features.ini"` and **the engine reads it itself**.
So: change our features in `features.ini`, change the model / port / context in `llmserver.bat`.
An explicit command-line argument always **wins over** `features.ini`.

Experimental variants (fixed depth / nomap / lv4 / various draft depths) live in
`tools\_variants\`, **not** in the package root.

---

## Current recipe (2026-10-11)

| Item | Value | Where |
|---|---|---|
| Base model | `Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf` (in `weights\`) | `llmserver.bat` → `MODEL` |
| Vision head | `mmproj-Qwen3.8-27B-Q8_0.gguf` (mounted when `VISION=1`) | `llmserver.bat` |
| Context | `-c 65536` | `llmserver.bat` → `CTX` |
| Speculation | `--spec-type draft-mtp-adaptive` + `--spec-draft-n-max 4` | `llmserver.bat` |
| Thinking | `-rea off` (client can override via `chat_template_kwargs`) + `--reasoning-budget 2048` | `llmserver.bat` |
| Sampling | request temperature 0.6 (client `llmchat.bat --temp`) | client |
| KV cache | `q8_0` + tiered: `kv-vram-cells = 32768` (the rest reads from pinned RAM) | `features.ini` |
| Draft sampling | `spec-draft-sampling = probabilistic` (true sampling + rejection) | `features.ini` |
| Draft shortlist | `spec-draft-vocab-map = maps/shortlist_24k_mix.vocab.txt` (24k rows) | `features.ini` |
| Adaptive floor | `spec-draft-n-min-adaptive = 1` (ceiling is `--spec-draft-n-max 4`) | `features.ini` |
| Device pool | `GGML_CUDA_SHARED_POOL = 1` (target + draft share one pool; saves ~1.4 GB on an 8 GB card; **only safe with `--parallel 1`**) | `features.ini` |
| #189 short query | `GGML_CUDA_FA_SHORTQ_MMA = 0` (ignore upstream #189, keep the in-place q8_0 fast path) | `features.ini` |

⚠️ Bound to `--parallel 1`: one slot serves one session. Do **not** run two servers on the
same card — they fight over the port and the VRAM.

---

## Directory layout (split by nature, 2026-10-13)

One table for the top level; five kinds of thing go five ways — **Git follows only the code kind**:

| Path | Nature | In Git | Size | Distribution |
|---|---|---|---|---|
| `llmserver.bat` | run entry point (comments **must be pure ASCII**, see "Maintenance") | ✅ | 3 KB | Git |
| `features.ini` | **the only feature config** (English comments) | ✅ | 5 KB | Git |
| `maps\` | production draft shortlist (24k rows) | ✅ | 136 KB | Git |
| `llmchat\` | chat client (`chat.py` + launcher + `mem\`) | ✅ | 2.0 MB | Git |
| `apps\` | graphical apps: `jev-image\` (incl. `skill-chess-board.md`), `jev-tetris\` | ✅ | 1.5 MB | Git |
| `engine\` | engine-side archive: PTQ1 build scripts / `_dflash_bonsai.patch` / handover docs | ✅ | 141 KB | Git |
| `docs\` | measurement logs, design notes, port records | ✅ | 232 KB | Git |
| `tools\` | measurement scripts + `_variants\` (archived variants) | ✅ | ~330 KB | Git |
| `weights\` | 6 gguf files (current base + alternates + mmproj) | ❌ | 30 GB | **Hugging Face** |
| `build\bin\` | `llama-server.exe` + all dlls | ❌ | 331 MB | **GitHub Releases** |
| `build\cuda\` | CUDA runtime (cublas / cublasLt / cudart / nvrtc / nvJitLink) | ❌ | 697 MB | same release |
| `train\` | training work: `ternary\` `trainlib\` `vocab\` `calib\` `cute-triadic-gdn\` | partial | 3.1 GB | code→Git, corpora→HF |
| `data\huggingface\` | data preparation work | ❌ | 2.0 GB | **HF dataset** |

≈ **36 GB** in total, but **Git tracks only 125 files / 3.2 MB** — the big things are all in
`.gitignore`, each with its own channel, so `git status` only ever shows source and config.

⚠️ **The engine source is not in here.** It lives in `code\`
(a fork of `PrismML-Eng/llama.cpp`); this package only **consumes** its build output
(`build\bin\`).

---

## Relationship to `..\adaq8\`

`adaq8\` is an **old copy of this package plus an ARC evaluation workspace**. It runs the
same engine and the same gguf. Since 2026-10-13 **this package is the only main repo**
(`adaq8\` has no `.git` of its own):

- `adaq8\weights` and `adaq8\build\*` should become junctions pointing here, so weights and
  the CUDA runtime exist only once on the machine;
- engine sync direction: `adaq8\build\bin\*` → this package's `build\bin\` (done 2026-10-13,
  sha256 **15/15 identical**);
- `features.ini` is byte-identical on both sides; `llmserver.bat` differs only by this
  project's `VISION` block;
- the ngram knobs are unified at `NMATCH=24 / NGMIN=24`.

**Before removing `adaq8\`, relocate the three things only it has** — `train\ternary\`,
`data\huggingface\`, `train\cute-triadic-gdn\`. None of them belongs to this package.

The full port/sync record and rollback steps are in `docs\ported-from-adaq8\PORTED.md`.

---

## Tools (`tools\`)

| Tool | Purpose |
|---|---|
| `bat_audit.py` | **read-only** health check: non-ASCII / CRLF / BOM / GBK damage; `--diff a.bat b.bat` prints the **real command-line difference** (does not trust comments) |
| `ab_bench.py` | A/B any two launcher variants in one command: `--bat a.bat --bat b.bat`; starts the server, waits for `/health` 200, reports medians and a **zero-overlap verdict** |
| `mtp_head_probe.py` | zero-dependency minimal GGUF reader: vocab count, first tensor names/types, `prism.hadamard.*` — predicts whether a feature can attach to a model |
| `think_ab.py` | per-token throughput, thinking on vs off |
| `ctx_ab.py` | decode speed by context length; measures how much KV was reused (`cache_n`) |
| `gen_profile.py` | per-segment speed within one answer, aligned on the same time axis as `nvidia-smi` sampling — decides "blame the model or the GPU" |
| `keep_reasoning.py` | measures the real cost of keeping the thinking block across turns |
| `_variants\` | archived launcher variants (incl. the clean baseline arm `llmserver-nomap.bat`) |
| `ported-from-adaq8\` | probe scripts ported from adaq8 (network capture / endpoint checks) |

```bat
cd F:\super\qwen27b\tools
python bat_audit.py
python ab_bench.py --bat llmserver.bat --prompts scenes --runs 5 --tag mytest
python gen_profile.py --pad 0,2000 --max-tokens 2560 --width 128
python mtp_head_probe.py ..\weights\Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf
```

---

## Documentation

| File | Content |
|---|---|
| `docs\BENCH.md` | **in-package measurement log**: what worked, what was rejected, measurement rules, silent-failure criteria, mistakes made |
| `docs\VOCABMAP.md` | mechanism and payoff of the 24k draft shortlist |
| `docs\design-vocab-shortlist.md` | full derivation of the shortlist design (payoff recomputed from actual code use) |
| `docs\SPEEDUP-BUDGET.md` / `TERNARY-SPEEDUP-SURVEY.md` | speedup budget and survey |
| `docs\ported-from-adaq8\PORTED.md` | sync record with adaq8 and **how to roll back** |
| `docs\ported-from-adaq8\VOCAB-CJK.md` | shortlist and adaptive depth measured on Chinese / English / code |
| `docs\JEV.md` | JEV apps (vision / graphics) |

---

## Maintenance notes

1. **Comments in `llmserver.bat` and `features.ini` must be pure ASCII.** A UTF-8 Chinese
   comment is decoded as GBK by cmd (codepage 936) and desynchronises: when a line's Chinese
   byte count is odd it pairs with the first byte of the next line, swallowing that line's
   `rem`, and the rest is executed as commands, producing a pile of
   `'xxx' is not recognized`. **We hit this for real.**
   ⇒ After editing any bat, run `python tools\bat_audit.py`, then
   `--diff llmserver.bat <edited>` to confirm what actually changed.
2. **This package ships only the CUDA runtime, no build toolchain.** To rebuild the engine,
   go to `code\` (the full CUDA Toolkit is at `F:\super\tools\cuda`, including nvcc).
3. **If a new binary needs a dll that was trimmed from `cuda\bin`, startup fails immediately.**
   Re-test after swapping binaries.
4. **When changing the base model**: check `llmserver.bat`'s `MODEL` together with every
   model-bound item in `features.ini` (the shortlist path, for instance); the `VISION=1`
   mmproj must match the new base.
5. **Swapping the draft head needs no re-grafting**: same trunk, so just replace the gguf in
   `weights\`.
6. **Do not delete `F:\super\llm\_backup\bin-1012-prepartial\`** — it is currently the only
   engine rollback point (the in-package `bin.bak-*` were cleared in the 2026-10-13 split).

---

## Performance

**Scope warning:** the last **formal A/B** for this package was 2026-09-22, against the old
base model (`Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf` + shortlist + fixed depth). No formal A/B
has been run since the base change, so **do not compare the table below across base models**.

Measured 2026-09-22, **same session**, median of 5 runs each (full evidence in `docs\BENCH.md`):

| Config | Code | Prose | Peak VRAM |
|---|---|---|---|
| fixed depth 3 (baseline of that day) | 33.28 t/s | 30.28 t/s | 7523 MiB |
| ✅ `adaptive [1..3]` | **34.93 (+5.0%)** | **33.47 (+10.5%)** | 7557 MiB |
| `DRAFT=1` (different scope) | tie | **37.7 (+20%)** | — |

- **The adaptive win is real, not noise**: the prose arm was 5 vs 5 with **zero overlap**,
  and it ran second (heavier thermal throttling) yet still won by 10.5%.
- **Do not raise `--spec-draft-n-max`**: the controller climbs past the setpoint, per-round
  verify cost is roughly **linear in depth** while output is sub-linear, so deeper always loses.
- Swapping the draft head (Q5_0+Q8_0 → Q6_K/Q4_K) is worth only **+1~3%**, inside the noise.
- ⚠️ **Measurement noise is large**: the same config can spread **±12%** over 9 runs, and
  cross-session differences reach 7%. Any A/B must be **same-session**, ≥3 prompts × ≥3 runs,
  median.

Smoke measurements 2026-10-11 (new base + adaptive + `features.ini`, **probe scope, not A/B**):

| Scenario | Result |
|---|---|
| Prefill (short prompt) | 155 t/s |
| Prefill (long prompt, 4096 chunking) | 500 → 391 t/s (29.6k tokens) |
| Token generation (thinking off) | 20 ~ 41 t/s (strongly content-dependent) |
| Multi-turn incremental reuse | turn 2: `prompt_n = 22 / cache_n = 1124` (**proper increment**) |

---

## Provenance

| In package | Origin |
|---|---|
| `build\bin\` | `F:\super\qwen27b\code\..\build\src\bin` (rebuilt 2026-10-13 after merging 14 upstream commits) |
| `build\cuda\bin\` | `F:\super\tools\cuda\bin` (runtime dlls only; **no** nvcc / ptxas / nvdisasm) |
| `weights\` | 6 gguf files incl. the production base `Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf` (renamed from `models\` on 2026-10-13) |
| `engine\` | former `src\`: PTQ1 build scripts, `_dflash_bonsai.patch`, Ampere / SM86 handover docs (archive only) |

---

## Related

- `code\` — **engine source repo** (fork of `PrismML-Eng/llama.cpp`). `build\bin\` comes from
  it; rebuild the engine there (CUDA Toolkit at `F:\super\tools\cuda`).
- `..\llm\_backup\bin-1012-prepartial\` — **engine rollback point** (currently the only full
  binary backup).
- `..\adaq8\` — old copy plus ARC evaluation workspace, being merged into this package.
- `docs\ported-from-adaq8\PORTED.md` — sync record and rollback steps.
