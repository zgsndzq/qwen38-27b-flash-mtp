# qwen27b → ada-q8 可移植优化普查

日期：2026-09-28 · 目标：把 `llamAmpere_gdn`（qwen27b 的 fork）做过的优化逐条对到 `ada-q8`（Bonsai-2-27B PTQ1_0）上，判定哪些真能提速。

方法：命令行选项集合做差集；两个 fork 的关键文件出 diff；CUDA 内核清单做差集后逐个判读量化类型/架构绑定。

---

## 0. 结论先行

**两个项目跑的是同一个模型。** `adaq8\models` 是 NTFS junction，指向 `qwen27b\models`，
物理上是同一份 `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf`（6,212,666,848 字节，同一 inode）。

GGUF 元数据（`tools/_gguf_meta.py` 直读 header）：

| key | value |
|---|---|
| `general.architecture` | **qwen35** |
| `general.size_label` | 27B |
| `qwen35.block_count` | 65 |
| `qwen35.nextn_predict_layers` | 1（有 MTP） |
| `qwen35.full_attention_interval` | 4 |
| `qwen35.ssm.conv_kernel` / `state_size` / `inner_size` | 4 / 128 / 6144 |
| `prism.hadamard.transform` | normalized-sylvester-walsh-hadamard |

⇒ **Bonsai 不是 dense Transformer，而是 Qwen3.5 的 GDN(SSM) + attention 混合架构**：
约 49 层 GDN 循环层 + 约 16 层 full attention。ada-q8 fork 侧也有对应的完整实现
（`ggml-cuda/gated_delta_net.cu`、`ssm-conv.cu`、`ssm-scan.cu`、`src/llama-memory-recurrent.cpp`）。

⚠️ **这修正了本文初稿的一个判断**：初稿曾把 GDN 类优化归入"结构上不适用"，理由是误判
Bonsai 为 dense。**实际适用**，已移到 §3.4。

于是两个 fork 的关系是：**同一个模型上的两条独立演化分支**。

- ada-q8 在 **block-draft（dspark/dflash）** 上走得更远（引用 113 处 vs gdn 57 处），且独有 `mmq-hopper-q1.cu`
- gdn 在 **TURBO 量化类型** 和 **MoE** 上走得更远 —— 这两块对 Bonsai（PTQ1_0 量化、非 MoE）**确实不适用**

24 个"gdn 独有开关"里，真正可考虑的只剩 4 项（见 §3）。

而 §1 的自适应已经实测完毕：**无收益**。

---

## 1. 自适应草稿深度：已移植，实测无收益

已按计划移植（`common/speculative-adaptive.h` + common.h/arg.cpp/speculative.cpp/server-context.cpp），编译通过，`--spec-draft-n-min-adaptive` 已出现在 `--help` 中，探针确认深度真的在动（平均 2.08–2.77，非恒定 3）。

**A/B（3 提示 × 5 次取中位，同会话，cache_prompt:false）**：

| arm | cn.prose | en.code | cn.code | 汇总中位 |
|---|---|---|---|---|
| fixed3 | 45.80 | 46.09 | 49.79 | **46.61** |
| adapt[2..3] | 46.74 | 44.03 | 47.11 | **46.10** |

`adapt[2..3] vs fixed3 = -1.09%，OVERLAP（不可分）`

接受率反而是自适应更高（0.469 / 0.425 / 0.459 vs 0.398 / 0.403 / 0.468），但深度被压到 2.1–2.3，两者抵消。

⇒ **ada-q8 上固定深度 3 已在最优点附近，自适应没有地板可兜。** 与 qwen27b 的结论相反（那边中文掉词严重）。原因：ada-q8 的短表是 pass-2 mix 表，中文覆盖率 98.9%，不存在"掉词"这个自适应要修的病。

**处置：保留代码（不删），生产 launcher 继续 `draft-mtp`。**

---

## 2. 搬不过来的（按绑定原因归类）

### 2.1 量化类型绑定 —— 最大一块

ada-q8 主模型是 **PTQ1_0 / PQ2_0**；gdn 独有以下内核**全部只服务 TURBO2_0 / TURBO3_0 / TURBO4_0 / Q5_CR / Q6_CR / Q8_CR**：

| 文件 | 大小 | 绑定 |
|---|---:|---|
| `mmvq-tq.cu` | 61 KB | TQ 三值 mmvq |
| `fattn-mma-turbo.cuh` + 22 个 template-instance | 8 KB | `DECL_FATTN_MMA_TURBO_ALL(..., GGML_TYPE_TURBO4_0/TURBO3_0/TURBO2_0)` |
| `turbo-quant.cuh` / `turbo-wht.cu` / `turbo-innerq.cu` | 40 KB | TurboQuant + WHT + InnerQ 均衡 |
| `mmv-cr.cu` | 15 KB | `block_q6_cr` / `q8_cr` / `q5_cr`，`static_assert(sizeof(block_q6_cr) == sizeof(block_q6_K))` |
| `convrot.cu` | 14 KB | GDN 的 conv+rope |

⇒ 换量化类型才用得上，不是"移植内核"能解决的。

### 2.2 模型结构绑定

- `--moe-cache` + `moe-devsort.cu` + `moe-weighted-reduction.cu`：MoE 专用。Bonsai 元数据里
  既没有 `expert_count` 也没有共享专家张量，不是 MoE ⇒ 不适用。
- `--dflash-defer-injection` / `--spec-synth-len` / `--spec-synth-rates`：dflash 专用
  （ada-q8 侧 dflash 引用反而更多，见 §0）。

⚠️ **初稿勘误**：`--gdn-replay` 原先被列在这里，理由是"选项说明自己写了 for Gated DeltaNet
models (Qwen3.5/3.5-MoE)"就判定 Bonsai 不匹配 —— 这是把模型名字当成了判据。
**Bonsai 的 `general.architecture` 就是 `qwen35`**（§0），正是该选项的目标架构。已移到 §3.4。

### 2.3 场景绑定（SLOTS=1 低延迟用不上）

`--lazy-mode` / `--kv-unified-per-slot` / `--ctx-size-draft` / `--cache-disk-path` / `--cache-disk-limit` / `--lookup-cache-dynamic-{max-ngrams,save,save-interval}` / `--slot-cache-key-{min-prefix,similarity}` / `--n-cpu-ffn` / `--log-jsonl`

全是容量、多槽、磁盘缓存、离线批量向的。当前生产是 `SLOTS=1` + `CTX=32768` 单槽对话，没有第二槽可调度，也没有磁盘缓存需求。

---

## 3. 真正架构无关、可考虑的 3 项

### 3.1 exact p/q drafting（`LLAMA_SPEC_PQ`）—— 推荐先算清楚再动手

gdn 自 2026-09-10 起默认开启（`pq_enabled`，`LLAMA_SPEC_PQ=0` 可关）：

> draft tokens are sampled at the request's temperature and verified by rejection sampling

ada-q8 完全没有这套代码（`pq_smpls` / `pq_enabled` / `LLAMA_SPEC_PQ` 全树 0 命中）。ada-q8 现在的 draft 采样器是：

```cpp
common_params_sampling sparams;
sparams.no_perf  = false;
sparams.top_k    = 10;
sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };   // 无 temp ⇒ greedy
```

即 **draft 取 argmax、target 按请求温度采样、验证走 id-match**。理论上 rejection sampling（接受率 `1 - TV(p,q)`）应高于 id-match（接受率 `p(argmax)`）。

**但实测给了负面信号。** 温度扫描（draft-mtp 固定深度 3，中文提示，192 token × 2 次）：

| temperature | tg t/s | accept |
|---|---|---|
| 0.0 | 45.96 / 46.24 | **0.4047 / 0.4047** |
| 0.3 | 45.98 / 44.45 | 0.4015 / 0.3783 |
| 0.6 | 47.34 / 45.37 | 0.4315 / 0.3962 |
| 1.0 | 44.30 / 41.58 | 0.3811 / 0.3333 |

**接受率对温度几乎不敏感** —— 连 target 走 greedy（temp=0）时也只有 0.4047。

含义：瓶颈**不是**"采样温度导致的分布失配"，而是 **MTP draft head 自身的 top-1 与 target 的 top-1 只有 ~40% 一致**。p/q 能拿到的收益是 `1 - TV(p, q_mtp)`，取决于 MTP head 分布能否逼近 target 分布，不是从 0.40 跳到 1.0。

⚠️ **caveat**：该实验未单独验证 `temperature` 是否真的透传到了 target sampler（输出内容未记录）。若透传失败，温度维度无效、结论不成立。**动手前应先确认这一点**（同一提示 temp=0 vs temp=1.0 的输出文本应有明显随机性差异）。

工作量：跨 `common.h`（draft params 加 `sampling` / `result_q`）+ `speculative.cpp`（约 6 处）。注意两 fork 的 `speculative.cpp` diff 有 **3284 行**，不能直接拷，要按 ada-q8 自己的 MTP 路径重写。

### 3.2 `--spec-chain`（chained MTP）—— 收益最确定，工作量最大

选项说明：

> chained MTP drafting: all draft tokens in one GPU decode

实现注释更明确：

> draft all n_max tokens with one chained decode (in-graph argmax feeds each next step); replaces n_max sequential draft decodes

当前 ada-q8 深度 3 = 3 次串行 draft decode + 1 次 verify。chain 把这 3 次合成 1 次。

**阻塞点**：`chain_graph` 需要

```cpp
llama_model_supports_mtp_chain(llama_get_model(ctx_dft))
```

而 ada-q8 的 `src/` 与 `include/` **全树没有这个函数**（gdn 在 `src/llama-context.cpp:4768` + `src/llama-ext.h:119`）。这是模型/图层的支持，不是 common 层能单独补的 ⇒ 要把 gdn 的 `llama-model/llama-graph` 侧的 chain 建图一起搬。

工作量：大（涉及 src 层图构建）。收益：直接省掉 N−1 次 draft decode 的调度与 launch 开销。

### 3.3 `fattn-swizzle.cuh` —— 中等工作量，长序列受益

gdn 的 `fattn-mma-f16.cuh:7` 有 `#include "fattn-swizzle.cuh"`，注释说明它让 f16 loader 和每次 `ldmatrix` 读都用 swizzled 布局；ada-q8 的同名文件是**不带 swizzle 的旧版**。

对 attention 占比高的场景（长上下文 prefill / 长序列）有用；当前 32768 ctx 单槽对话的 decode 是 mmv 主导，收益偏小。

### 3.4 GDN 循环层优化 —— 初稿误判，实际适用，**可能是最大的一块**

Bonsai 65 层里约 **49 层是 GDN(SSM) 循环层**（`full_attention_interval = 4`）。解码时这 49 层的
`ssm_conv` / `ssm_scan` / `gated_delta_net` 是主要计算量所在，而 ada-q8 fork 侧已有对应内核
（`gated_delta_net.cu`、`ssm-conv.cu`、`ssm-scan.cu`）—— 也就是说 gdn 在这块做的优化**能直接对上**。

- **`--gdn-replay`**（gdn 独有开关，ada-q8 全树无）：选项说明写
  "for Gated DeltaNet models (**Qwen3.5**/3.5-MoE) **with MTP rollback enabled**"，
  而 Bonsai 的 `general.architecture` 正是 `qwen35`、`nextn_predict_layers = 1`（MTP 已开）—— 完全吻合。
  作用：不为每个保留的草稿位置急切地存完整状态快照，改为记录 per-token 的小料再重放重建，
  ⇒ **同时省 VRAM 和 rollback 的快照带宽**。
  ⚠️ 当前 ada-q8 显存 **7673 / 8188 MiB** 已经贴着上限，这一项同时打"容量"和"速度"两个点。
- **`convrot.cu` + `convrot.cuh`**（14 KB，gdn 独有）：GDN 的 **conv + rope 融合**内核。
  ada-q8 有 `ssm-conv.cu` 但没有 conv 与 rope 的融合版。纯内核、不动计算图 ⇒ 可移植性高。

**优先级建议：先看 `--gdn-replay`。** 它是这 4 项里唯一同时命中"Bonsai 是 qwen35"、
"已开 MTP"、"显存贴顶"三个条件的。

---

## 4. 一句话建议

自适应这条路已经走完并证伪（§1）。剩下 4 项里：

1. **§3.4 的 `--gdn-replay` 优先级最高** —— 初稿把它误判掉，实际 Bonsai 就是它的目标架构，
   且当前显存贴顶，收益面最宽。
2. **§3.1 的 p/q**：先花 10 分钟、零代码把 temperature 透传验证掉 —— 它决定了 p/q 到底值不值得移植。
   若透传成立且接受率确实与温度无关，则 p/q 收益有限。
3. **§3.2 的 spec-chain** 收益最确定但工作量最大（要动 src 层建图），适合放在最后。
