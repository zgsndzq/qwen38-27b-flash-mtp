# 负面清单：试过、量过、确认没用（或有代价）的方向

这份文档记录**已经做过并有实测数据**的失败方向。目的是让后来的人（包括我们自己）不再重复走一遍。
每条都给出：做法 → 数字 → 结论。没有数据的猜测不在这里，那属于 `task.md` 的待办。

写下来的东西分三类：性能负例、正确性负例、工程陷阱。

---

## 0. 测量前提（先读这一节，否则下面的数字会互相打架）

**`BENCH=1` 是唯一的可比口径。**

`--spec-type ngram-mod,draft-mtp-adaptive` 里的 ngram lookup drafter 会在对话过程中把哈希表填满。
同一个 prompt 连续请求，吞吐会单调爬升：

```
第1轮 46 t/s → 第2轮 93 → 第3轮 116 → 第4轮 131 → 第5轮 156 → 第6轮 177
```

**3.9 倍的差值全部来自 ngram 缓存，与被测优化无关。** 冷启动阶段的任何 A/B 都是假的
（2026-10-19 晚因此作废了三条结论：kv_unified "+21%"、p_min 校准、扩列）。

| 口径 | 含义 | 27B 数值 |
|---|---|---|
| 生产 | ngram 开（`BENCH=0`） | 冷 46 → 热 177 t/s |
| 基准 | ngram 关（`BENCH=1`），只有 MTP | **45.5 t/s，稳定可重复** |

**优化只能作用在基准上**，ngram 会自动把它乘上去。所以“接受率提高”不是目标，
**每轮产出的 token 数**才是——ngram 之所以快，是因为它一次猜一大段；
任何“缩短草稿链”的手段（见 1.1）与它的方向天生相反。

---

## 1. 27B（`Qwen3.8-27B-Flash-MTP`，PTQ1_0 1.75 bpw，dense 65 层，hidden 5120）

### 1.1 草稿置信度门控（p_min / p_temp 标量校准）— 负例

**做法**：MTP 头报的是 top-1 概率，实测中位数 0.822，而 target 只接受 0.50，属于严重过自信。
于是加 `--spec-draft-p-min` / `--spec-draft-p-temp`（`p' = p^T`），在概率低时停止草稿。
接受率确实从 0.50 提到 0.77~0.99。

**数字**：

| 配置 | 基准 t/s | 生产 t/s |
|---|---|---|
| 基线（门控关闭） | **45.5** | 175~181 |
| 校准 0.45 / 1.6 | **43.2**（-5%） | 159（-9%） |

**结论**：**门控是负收益，不要开。** 原因不是概率不准，而是**门控砍掉草稿链**：
草稿比验证便宜，所以“浪费一半草稿换更长的链”比“每步都准但链短”更划算。
去掉排序开销（自己线性扫描求 max，不再 `common_sampler_get_candidates(smpl, true)` 排序）
后亏损从 -9% 缩到 -5%，但方向仍为负。

代码保留（`common.h` 的 `p_temp` 字段、`--spec-draft-p-temp`、`LLAMA_SPEC_DUMP`），
**默认关闭、零开销**。`LLAMA_SPEC_DUMP` 值得留着：正是它第一次暴露了 ngram 预热效应。

### 1.2 低秩 logit bias 校正（DSpark 思路）— 负例（原理上不成立）

**做法**：借用 DeepInfer 的 DSpark 思路，`logits' = logits + h @ U @ V`，用低秩矩阵把草稿
输出拉向主模型分布。

**为什么在本模型上不成立**（数据，不是推测）：

```
p0 ≥ 0.90 的步占 47.5%
这些步里第二候选的 p1 中位数 = 0.00364      ← 概率差 250 倍
把 top-2 顶到 top-1 需要的 logit 差 = 5.61
```

草稿头在自信时与第二名差了 250 倍，它**根本没考虑过其他可能性**。
线性低秩校正只能做 0.1~0.5 量级的微调，无法把排在几十名之后的 token 推到第一。
真要解决只能重训草稿头。

### 1.3 kv_unified — 负例（价值有限）

**做法**：上游 llama.cpp 的多序列做法是把 token 拼在 ne[1]（列维），`n_stream` 只出现在 KV 的 ne[3]。
加 `--kv-unified` 即可。

**数字**：短 prompt 非流式 +21%（44.3 → 53.8）；真实场景（`llmchat-dual.bat`，流式 + 钉 slot + KV 存档）
只有 **+7.6%**（57.8 → 62.2）。

**结论**：配置级收益太小，且短 prompt 的 +21% 是测量假象。

### 1.4 扩展 PT mat-vec 的列数上限 — 负例（本卡最优宽度是 4~5 列）

**做法**：`PTQ1_0_PT_MAX_COLS` 8 → 12，补齐 case 9~12 的模板实例，
`rows_per_item` 逐档压到 4 / 2 / 1 行。

**数字**（同 prompt / 300 token / 双请求）：

| 方案 | 合计 t/s |
|---|---|
| 门限 5（≤5 列 matvec，6+ 列 MMQ） | **62~63** |
| 12 列全 matvec，ROWS=4 | 13.0 |
| 12 列全 matvec，ROWS=2 | 23.3 |
| 12 列全 matvec，ROWS=1 | 24.6 |
| 2 列起全 MMQ（门限 1） | 46.8 |
| 12 列全 MMQ（每槽 6 列，DRAFT_N=5） | 16.0 |

**结论**：4060 Laptop（128-bit、8G）在 5120×17408 这个矩阵上，**一次调用的最优宽度就是 4~5 列**，
加宽到 8 列以上 matvec 与 MMQ 一起崩。原因不是寄存器溢出（ROWS 4→1 只从 13 提到 24.6），
而是 SMEM 预算 `rows_per_cta = 4096 / (ncols × 137)` 把每 CTA 的行数压到 1，
每个 CTA 都要重搬一遍激活、重做一遍归约，权重复用的收益被吃光。

**保留**：12 列的代码能力留在树里（`GGML_CUDA_PTQ1_MMVQ_MAX=12` 可开），但**门限必须停在 5**。

### 1.5 门限类误操作 — 负例

- `GGML_CUDA_PTQ1_MMVQ_MAX=8`：让 6~8 列走 matvec，反而更慢（cols=5 时 MMVQ 2.226ms vs MMQ 0.658ms）。
- `GGML_CUDA_PTQ1_MMVQ_MAX=24`：超过 `MMVQ_MAX_BATCH_SIZE` 硬上限，**assert 直接崩服务**。

### 1.6 FA 多流降级 / chunked 路径 — 负例（不适用）

chunked 路径要求 `K->ne[1] > 8192`，而 decode 阶段 K 的长度就是上下文本身，
不满足时自动退回；**对 decode 没有意义**，只影响 prefill。

### 1.7 GDN 双流税 — 撤回（无证据）

曾怀疑 hybrid 模型（GDN 循环状态）在多流下有额外代价，**op 级数据不支持**，撤回该结论。

### 1.8 树型草稿（tree verification）— 负例（引擎层不支持）

**做法**：给草稿开第二条分支序列，用 `llama_memory_seq_cp` 共享前缀，一次 decode 验两条分支。

**撞到的三道硬墙**（每道都是引擎断言）：

1. `n_seq_max` = `--parallel`，分支用 seq 1 时越界 → `llama_decode` 返回 -1（Invalid input batch）
2. 把 `n_seq_max` 改成 `n_parallel + 1` → 加载期 `GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)` 崩
3. 用 `--parallel 2` 拿到第二个 seq 并把请求钉在 slot 0 →
   **`llama-kv-cache.cpp:540 GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")`**

第 3 条是硬限制：qwen35 是 **hybrid（GDN 循环状态）**，它的 memory 不是 full KV buffer，
`seq_cp` 这类“共享前缀 cell”的用法根本不支持，换 f16 KV 也一样。

**结论**：在这两个 qwen35 hybrid 模型上，树验证只能走 EAGLE-4 那条路
（FA kernel 里加自定义树掩码 + GDN 分支状态），那是几百行 CUDA 的活，
在 speculative.cpp 层加几行解决不了。

### 1.9 EAGLE-4 草稿移植 — 未跑通（接受率 0，剩余两个嫌疑）

官方 `EAGLE-Corp/Qwen3.8-27B-EAGLE4` 转成我们引擎能读的 GGUF（`tools/convert_eagle4_to_gguf.py`），
能加载、能出草稿，但 **acceptance 0.00000（0/351）**，草稿 top-1 概率 0.11~0.16、跨 24.8 万词表接近均匀。

**已排除**（都有实测）：

| 怀疑点 | 结论 |
|---|---|
| 目标特征没进来 | 三层 mean\|x\| = 0.16 / 1.17 / 2.29，5120 维全非零，encoder 输出 1.19 |
| 拼接顺序反了 | 官方 `x = cat(input_layernorm(embeds), hidden_norm(hidden))`，与我们一致 |
| `fc_norm` 语义 | 官方 `_project_target_features`：三路各自 RMSNorm 后 concat 再 fc，与我们的实现一致 |

**已按官方参考补上的两处**（保留，可能就是解法的一部分）：

- `eagle3.norm_output`（config 里为 true）：官方 `return normed, [normed if self.norm_output else summed]`，
  即下一轮的 g_embd 必须是**归一化后**的 hidden，我们原来一直传 prenorm
- `fc_norm.0/1/2` 三个张量 + per-aux norm 图；token 嵌入改走 `build_embd_rows`（Hadamard 折叠还原）

**剩下两个嫌疑**：

1. 输出头。官方文档写的是 “LM head: the low-rank candidate head”，我们借用的是 target 那个
   折叠过的 PTQ1_0 全量头（`build_lora_mm` 会按 `hadamard_rotations` 转一次）。
   可试把 `efficient_lm_head`（rank 128）打进 GGUF。
2. 特征层语义。`llama_set_embeddings_layer_inp(3)` 取的是“进入第 3 层的 hidden”，
   而 HF 的 `eagle_aux_hidden_state_layer_ids` 多半指“第 3 层的输出”，差一层。

### 1.10 扩散模型做草稿 — 负例（成本结构不成立）

扩散语言模型生成 N 个 token 要跑 T 步去噪，每步都是一次全模型前向。
我们只有 1 层 MTP 头，草稿本来就便宜，扩散的“每 token 草稿成本”是它的 4 倍以上；
而且显存只剩 293 MiB，加不下第二个模型。**关键是这个位置已经被 ngram 占满了**：
ngram 一次最多猜 24 个 token，成本接近零。

### 1.11 其它已排除项

- 权重驻留/融合（把权重常驻 SRAM）：8 GB 卡上没有余量。
- 稀疏 attention（`GGML_CUDA_FA_SPARSE`）：我们实现了打分选页，但**没有 tile map**，
  属于他们文档里“比 dense 还慢 33%”的那一版（见第 4 节）。

---

## 2. 9B（`Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0`，PQ2_0 + MTP 头，qwen35 hybrid 33 层，hidden 4096）

这条线与 27B **同架构族**（都是 qwen35 hybrid，`full_attention_interval=4`，GDN，1 个 nextn 层），
**词表也完全一致（248320）**，所以 27B 的 24K 草稿短词表可以直接复用。

### 2.1 负例

| 方向 | 数字 | 结论 |
|---|---|---|
| **DFlash2 块草稿器** | 纯解码 77.3 / 77.8 t/s；配 DFlash2 n-max 4 → 68.2 / 71.5 | **比纯解码慢 10%**，官方 USAGE.md 也这么说，只在 copy-heavy 场景才可能赚 |
| DFlash2 n-max 16 | 26.9 t/s（accept 0.11） | 多猜的 slot 大量被拒，验证成本白花 |
| MiMo-9B（q5_k）当 target 配 DFlash2 | 43.7 t/s | 尺寸能凑合，但草稿器没针对它训练过（accept 只有一半） |
| 27B 的 `GGML_CUDA_PTQ1_MMVQ_MAX` | 无效 | 9B 是 **PQ2_0**，不走 PT kernel |
| 27B 的 `kv-vram-cells` 分层 | 不需要 | 128K q4_0 KV 在这个模型上只有约 0.7 GB，全放显存；分层只会引入 PCIe 访问 |
| 27B 的 `GGML_CUDA_FA_SPARSE` | 不需要 | 同上，没有 host tail 可稀疏 |
| `--effort` 分档 | 空操作 | 模板的 `chat_template_caps` 写着 `supports_reasoning_effort = False` |
| 冗余文件 | 2.25 GB | 带 MTP 头的文件已包含全部主干权重，不带头的那个是冗余 |

### 2.2 正例（都实测过）

| 做法 | 数字 |
|---|---|
| 换 MTP 单文件 | 77.3 → **111~115 t/s（+47%）** |
| **草稿宽度 n-max = 2** | 112 t/s；n-max 4 掉到 96~105（accept 0.51）→ **9B 上 n-max 2 最优** |
| 对齐 27B 配置形态（ngram-mod + adaptive MTP + 24K 短词表 + `LLAMA_MTP_DRAFT_UBATCH=64` + q4_0 KV + 128K） | 冷 112 → 热 **399 / 480 / 486 t/s**（accept 0.96），显存 4065 MiB |
| 思考开关 | 模板支持 `enable_thinking`，服务端 `-rea off` 就是默认关；`--think off` 生效 |

**注意 DRAFT_N 的模型差异**：27B 用 4，9B 用 2。这不是配置漂移，是实测差异：
9B 上 n-max 4 让 accept 从 0.64 掉到 0.51；27B 上 n-max 2 与 4 速度相同（187 vs 186 t/s）、
acceptance 同为 0.98。PrismML 上游 #354 已把 CUDA 默认草稿深度改成 2。

### 2.3 9B 的独立配置

`features.ini` 的 `[bonsai]` 节（`[qwen]` 的 7 项专用项不适用于它）：

```ini
LLAMA_MTP_DRAFT_UBATCH = 64
spec-draft-sampling = probabilistic
spec-draft-n-min-adaptive = 1
spec-draft-vocab-map = maps/shortlist_24k_mix.vocab.txt
kv-vram-cells = 0
```

---

## 3. 跨模型通用结论

1. **4~5 列是这张卡的一次调用的最优宽度**（matvec 与 MMQ 都如此）。
2. **ngram 是乘数**：吞吐 = 基准 × ngram 倍数（27B 是 45.5 × 3.9）。任何优化只能作用在基准上。
3. **草稿宽度宁小勿大**：多猜的 slot 大量被拒，验证成本白花。
4. **每轮产出的 token 数**才是目标，接受率不是。
5. **两个模型不能同时跑**（5.79 + 2.49 GB > 8 GB），切换用计划任务。
6. 量化类型**不需要**和 target 一致：草稿可以是 f16/q8_0，target 是 PTQ1_0 也没问题，
   只要词表、hidden 宽度、以及 hidden 的**基**（Hadamard 折叠必须在查表后还原）一致。

---

## 4. 外部项目的负面结论（`professorpalmer/bonsai-ada-surgery`，与我们同源）

他们的模型就是 PrismML 的 PTQ1_0 27B，和我们同一套三值权重，因此他们的负面清单可信度很高。
**我们的 fork 比他们的 bundle 更靠前**（0053 关 host-tail graph、FA tile map、0047 tier partial staging、
0054 GDN chunked prefill、FA_PREFILL_F16 我们都有），但他们记录的死路同样适用：

| 想法 | 结果 |
|---|---|
| host row 交给 CPU 算 | 每层 192k 行约 110 ms（6 核） vs 走 PCIe 约 19 ms |
| sparse prefill（MAXQ=4096） | prefill +6%，但 10 个代码片段只对 8 个、12 个 needle 只中 11 个 |
| 选择范围扩到 VRAM 段（SINKS） | decode 36.3 vs 22.8，但 needle 7/12 —— 检索能力掉了 |
| `FA_V_SKIP=1`（跳过权重低于 exp(-20) 的 V 行） | 完全精确，但 decode 19.7 → 19.7：瓶颈是拷贝不是算 |
| 深上下文把 prefill micro-batch 提到 1024 / 2048 | **-36% / -85%**，512 最好 |
| MTP tail 4（8 GB 卡） | 多 3 个 rollback plane，cell 从 46,592 掉到 20,992，acceptance 0.79 → 0.55 |

他们的正例数字可当目标：读 host tail 时关 CUDA graph，192k decode 22.8 → 29.4 t/s（4070 12 GB），
128k +16%、192k +9%、245k +8%（2060 SUPER 8 GB）；带 tile map 的 sparse host-tail 读取在 160k 是 ×1.9。

---

## 5. 工程陷阱（性能之外的“负例”）

1. **`.bat` 必须 ASCII + CRLF，注释一律英文。** cmd 在 GBK 控制台下遇到中文注释会把整行当命令执行
   （报 “‘lookup’ 不是内部或外部命令”），LF 换行同样让 cmd 解析崩。
2. **ninja 不追踪 `.cuh` 依赖**：改了头文件必须 `touch` 对应的 `.cu`，否则是空编译。
3. **编译前必须先停服务**，`llama-server.exe` 会锁住 dll。
4. **草稿侧的日志进不了 server 日志**：`LLAMA_LOG_INFO` 和 `fprintf(stderr)` 都会被吞，
   只能用 `fopen` 写文件打点。
5. **heredoc 会破坏 python 补丁里的 `\n`**：写 C 字符串里的换行要用 `chr(92) + "n"` 拼，
   否则会变成真换行导致 “error C2001: newline in constant”。
6. **`git diff --stat` 报几千行** 通常是换行符（CRLF/LF）整体翻转，不是代码改动；
   提交前按 HEAD 的换行符逐个恢复。
7. **短 prompt + 非流式请求会高估收益**（见第 0 节的 ngram 效应）。
