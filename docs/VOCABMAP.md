# MTP draft-vocabulary shortlist — implemented, measured, re-mixed

> Status: **shipped into `llmserver.bat`** (2026-09-22), then **re-mixed the same day**
> after scene coverage showed the first table made Chinese traffic *slower*.
> Binary: `forks\llamAmpere_gdn\build-gdn` with three local source patches (below).
> Rollback: `F:\super\.archive\20260922-mixmap\llmserver.bat.bak`
> (pass 1 config) and `F:\super\.archive\20260922-vocabmap\llmserver.bat.bak` (pre-shortlist).

**Pass 2 result (3 prompts x 5 runs, median t/s); this is what ships now:**

| prompt | pass 1 (code-only map, fixed depth 3) | pass 2 (mixed map + adaptive) | delta |
|---|---|---|---|
| code.ringbuffer | 45.46 | **46.06** | +1.3% (not significant) |
| zh.explain (Chinese) | 26.18 | **33.98** | **+29.8%** (5-vs-5, no overlap) |
| zhcode (Chinese comments) | 32.65 | **37.74** | **+15.6%** (5-vs-5, no overlap) |

Draft acceptance moved with it: 0.62 / 0.22 / 0.36 -> 0.66 / 0.40 / 0.48, and the
effective draft depth fell from ~2.95 to 1.96-2.87. See "Mixed table" below.

## What it does

The MTP draft head (`output.weight`, `[5120, 248320]`, `ptq1_0`, **265 MB**) is re-read in
full on **every draft step**. The shortlist makes the drafter score only the 24 000 rows our
own code corpus actually uses, so a draft step reads **~26 MB instead of 265 MB**.

The target still verifies every draft against the **full** head, so the output distribution is
untouched: a worse draft can only cost speed, never correctness. Measured draft acceptance is
flat (0.56 → 0.58), which is the expected signature — the shortlist is a byte-count play, not a
quality trade.

## Measured (code prompts, `tools\ab_bench.py`)

| experiment | prompt | without | with | delta |
|---|---|---|---|---|
| smoke5 (1 run) | code.ringbuffer | 40.39 | **43.07** | +6.6% |
| smoke5 (1 run) | code.shell | 36.30 | **39.89** | +9.9% |
| prod1 (2 runs) | code.ringbuffer | 39.47 | **43.78** | +10.9% |
| prod1 (2 runs) | code.shell | 40.75 | 40.38 | −0.9% (flagged *not significant*) |

Acceptance in every pair: 0.581→0.607, 0.464→0.500, 0.561→0.578, 0.564→0.514.
Aggregate mean across all eight runs: 40.11 → 42.09 t/s = **+4.9%**; per-prompt medians +6~11%.
The design document predicted +7~9% from a pure byte account — that prediction holds.

## Scene coverage: this table is a code accelerator and a Chinese brake (2026-09-22)

The table is mined from **English code**, so it holds essentially no CJK tokens, and the
`MEASURED` section above used two code prompts. Score the *same* table per prompt group
(`vocab\groupcov.py`, service tokenizer) and the failure that the aggregate number hides
becomes visible:

| group | samples | tokens | coverage | q | gross | net |
|---|---|---|---|---|---|---|
| local (mined human code) | 5996 | 24,152,139 | 0.9976 | 0.0024 | +9.33% | **+9.10%** |
| py (model-written) | 1 | 1024 | 0.9785 | 0.0215 | +9.33% | +7.19% |
| sh | 1 | 1024 | 0.9678 | 0.0322 | +9.33% | +6.11% |
| js | 1 | 1024 | 0.9551 | 0.0449 | +9.33% | +4.84% |
| c | 1 | 1024 | 0.9492 | 0.0508 | +9.33% | +4.26% |
| **chat (zh)** | 4 | 4095 | **0.8320** | **0.1680** | +7.54% | **−9.26%** |

Inside the chat group the correlation *is* the story:

| prompt | prompt language | coverage | net |
|---|---|---|---|
| chat.plan | English | 0.9277 | +0.31% |
| chat.review | English | 0.9385 | +1.39% |
| chat.explain | **Chinese** | 0.7762 | **−14.85%** |
| chat.compare | **Chinese** | 0.6855 | **−23.91%** |

The dropped tokens are the proof -- every one of the chat group's worst misses is CJK:
`推测` x17, `如果` x11, `高频` x9, `概率` x9, `分布` x9, `比` x8.

Two consequences:

1. **The shipped config makes Chinese chat slower, not faster.** q (0.168) exceeds the gross
   byte saving (0.0754), so the table costs more speed than it buys on that traffic. A
   shortlist is only ever a win when the traffic looks like the corpus it was mined from.
2. **The two code-prompt numbers in `MEASURED` are a code-only workload.** They are honest,
   but they are not a description of the whole service.

Caveats: `net` is the design doc's model (`gross − q`), not an end-to-end A/B, and every group
outside `local` rests on 1-4 samples (`finish_reason=length`, capped at 1024 tokens). The
direction is structural -- a table mined from English code cannot cover CJK, and more samples
will not change that. Only the magnitude will move.

## Mixed table: the fix, and what it cost (2026-09-22, pass 2)

The table length is a fixed budget, so **who gets a row is zero-sum**. Pass 1 spent all 24 000
rows on English code; pass 2 spends ~4 000 on talk/CJK and ~3 000 on real served traffic and
gives up ~0.4pp of code coverage for it.

Three corpus sources, all tokenized by the **live** server tokenizer:

| pool | where it comes from | tokens |
|---|---|---|
| `code` | `corpus/local` (mined repo tree) + `gen` code groups | 17.9 M |
| `talk` | `corpus/gen` — the model answering 46 Chinese and 8 Chinese-code prompts, plus 6 thinking-block prompts | 41 k |
| `real` | `corpus/real` — real served traffic imported from `F:\bycode\ZhiYan\backend\data\Agents.db` (read-only) | 62 k |

`vocab\mixbuild.py` builds the table from weighted pools and keeps a **25% holdout** so its
scores are not in-sample. Same holdout, both tables:

| held-out set | pass 1 table | mixed table |
|---|---|---|
| `code/text` (mined code) | +9.09% | +8.71% |
| `code/answer` (model-written code) | −0.44% | +3.20% |
| `real/answer` (real traffic, body) | **−18.52%** | **−0.06%** |
| `real/reasoning` (real traffic, thinking) | **−5.98%** | **+2.79%** |
| `talk/answer` (Chinese answers) | **−16.79%** | **+3.78%** |
| `talk/reasoning` | **−30.00%** | **+0.46%** |

The talk weight curve is flat from 1 to 6 (real NET pinned at +1.48%, code at +8.70%): CJK
needs about 3.6 k rows and then saturates, so **there is no tuning left in that knob**. Shipped
at `real=3,talk=2,code=2`.

Then the surprise: on top of the mixed map, **adaptive draft depth** is worth far more than
the map choice. Both were measured in the same session (`tools\ab_bench.py --prompts scenes`):

| config | code | zh | zhcode |
|---|---|---|---|
| no shortlist, fixed depth 3 | 40.29 | 28.19 | 37.24 |
| code-only map, fixed depth 3 (pass 1) | 43.76 | 24.18 | 33.58 |
| mixed map, fixed depth 3 | 45.43 | 30.33 | 35.62 |
| **mixed map + adaptive depth (shipped)** | **45.06** | **34.97** | **40.23** |

So the honest attribution is: the mixed map is worth about +12.8% on code and +7.6% on Chinese
at fixed depth, and adaptive depth adds the rest. Pass 1's map was not merely neutral on
Chinese — it was **−14% against having no map at all**.

Verified engaged: `LLAMA_DRAFT_VOCAB_TRACE` recorded 2668 graph builds, every one
`-> COMPACT (n_sel rows only)`, `n_sel=24000`, `compact_ok=1`.

## The three source patches

`src/llama-graph.cpp`, `src/llama-context.{h,cpp}`, `src/llama-graph.h`:

1. **`ggml_type_name` whitelist** — `draft_vocab_direct()` did not list `GGML_TYPE_PTQ1_0`
   (`type=143`), so this model's head silently failed the eligibility check. One line.
2. **Hadamard rotation** — this model's `output.weight` is Hadamard-folded
   (`prism.hadamard.weight_names[0] == "output.weight"`), so the activation must be folded the
   same way before rows can be scored. The shortlist path used to `return nullptr` for such
   models, silently disabling the feature. Now mirrors `build_lora_mm`'s transform exactly.
3. **Compact head copy** — `init_draft_vocab()` also builds a `[5120, 24000]` `ptq1_0` tensor
   holding **exactly** the shortlisted rows (raw row bytes copied on the host, so the shortlist
   itself cannot change the weights' numerics), and the draft graph multiplies against that
   instead of the full head. Costs 26 MB of VRAM.

## Two dead ends worth remembering

**`ggml_get_rows` → matmul (the "packed" path) is a net loss.** `ggml_get_rows` always returns
F32 — `ggml.c` says so outright:

```c
ggml.c:4050   // TODO: implement non F32 return
ggml.c:4051   enum ggml_type type = GGML_TYPE_F32;
```

So gathering 24 000 rows materialises `24000 × 5120 × 4 B = 491 MB` of dense F32 **every draft
step**, ~19x the stored size of those very rows. Measured: **6.6 t/s vs 44.2 baseline = −85%**,
with acceptance *correct* (0.673). Right numbers, wrong bytes. A shortlist only pays if it stays
in the head's own quantised format.

**`mul_mat_id` with the shortlist as one-row experts (the "indexed" path) returns garbage.**
Draft acceptance collapsed to **exactly 0.0000** over 190 drafts — a signature no merely-worse
drafter produces. The CUDA path it selects (`mul_mat_vec_q_indexed_rows`) does not agree with
`ptq1_0`'s planar transposed activation layout: the dedicated PTQ1_0 mat-vec is gated behind
`!ids`, and the generic kernel that runs instead is wrong for it. Left in the code behind
`LLAMA_DRAFT_VOCAB_INDEXED=1` as a documented trap.

## Verifying it actually engaged (do this before trusting any number)

**The server's `--log-file` drops INFO logs coming from `src/`.** Only `common/`, `ggml/` and
server lines appear. So a shortlist that is silently disabled — or silently taking the wrong
branch — looks exactly like a working one, and the log gives you nothing.

Set the env var and the library appends one line per shortlist decision to a file:

```powershell
$env:LLAMA_DRAFT_VOCAB_TRACE = "F:/super/qwen38mtp/tools/vocabmap-trace.txt"
```

Healthy output looks like this (149 graph builds, all compact):

```
init_draft_vocab: head type=ptq1_0 ne=[5120,248320,1,1] n_embd=5120 n_sel=24000 row_sz=1120 n_hot=0 compact_ok=1
149 x graph: -> COMPACT (n_sel rows only)
```

`graph: -> PACKED` or `-> INDEXED` means the compact head is not reaching the graph — check that
`llmserver_graph_params` carries `draft_vocab_head` **and** that the `llm_graph_context`
constructor initialises it. (Missing the constructor line is exactly how this was caught: the
member's in-class `= nullptr` default hid it, and the symptom was a silent −85%.)

## Rebuild / retune

```powershell
python vocab\run.py build --source local --top-show 70   # re-measure table length on the corpus
python trainlib\build_cache.py                            # (unrelated: the n-gram table, see JEV.md)
```

Table length was measured on our own corpus: **24k is the optimum** (4k/8k leave too many tokens
out of the map; 64k buys nothing back). Swapping `maps\shortlist_24k_mix.vocab.txt` needs no
rebuild. (The pass-1 code-only map this line used to name is archived, see Files below.)

`HOT` must stay `0`: with hot slots the row set changes per request and **no compact copy is
built**, so the shortlist degenerates to the slow packed path. The flag is kept for the upstream
design, not for us.

## Files

| file | role |
|---|---|
| `maps\shortlist_24k_mix.vocab.txt` | **the shipped map** (24 000 ids, mixed pools) |
| `vocab\` | corpus collection (local / gen / real) + tokenizer + table builders |
| `vocab\mixbuild.py` | weighted-pool table builder with a 25% holdout (**the pass-2 builder**) |
| `vocab\groupcov.py` | per-group coverage/economics of any map file (the scene-coverage tool) |
| `llmserver.bat` | **the only launcher**: ships the mixed map + adaptive depth |

Intermediate control arms from pass 1 / pass 2 (code-only map, `proof_512` negative control,
`llmserver-{nomap,mixmap,mixmap-adaptive,adaptive,vocabmap*}.bat`) were moved on 2026-09-24 to
`F:\super\.archive\20260924-cleanup\` — see its `INDEX.md`. Nothing was deleted. The one worth
knowing about is `llmserver-nomap.bat`: it is the **only clean no-shortlist baseline**, so it is
what you restore if you ever need to re-measure the shortlist's net contribution.

## Rebuilding the mixed table

```powershell
python vocab\run.py gen --attach-only        # model answers its own prompt catalogue
python vocab\run.py real                     # import real served traffic (read-only)
python vocab\run.py tok --backend service --only gen,real --overwrite
python vocab\mixbuild.py --sweep --sweep-talk 0,1,2,3,4,6        # weight curve
python vocab\mixbuild.py --weights real=3,talk=2,code=2 --out maps\shortlist_24k_mix.vocab.txt
```

`--spec-draft-vocab-hot` must stay `0`: with hot slots the row set changes per request and
**no compact copy is built**, so the shortlist degenerates to the slow packed path.
Swapping the map file needs no rebuild of the binary.
