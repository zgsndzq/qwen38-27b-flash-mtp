# JEV：把"逻辑引擎"接在 MTP 前面 —— v1 实现与实测结论

> 用户 2026-09-22 的想法原话：
> "我们为什么不考虑代码补全技术呢？要提速，就要有超人的想象力……比如我们把逻辑引擎结合进去。
> 把 Jev 这种技术放进来。让模型在预测词上输出更快。"
>
> 📦 **2026-09-24**：`llmserver-jev.bat` / `llmserver-jev-wide.bat` 已移入
> `F:\super\.archive\20260924-cleanup\bat\`（未删除）。本文是**否决记录**，结论不变：
> n-gram 接在 MTP 前面是负收益，**别重复这条实验**。
> **结论：v1 做出来了、表真的加载上了、还是输了——写代码 −36%（温和表）到 −63%（激进表）。**
> 但这一轮不是白跑：我们拿到了**剂量-反应式**的因果证据，知道它为什么输、修哪里，
> 而且**是在抓到并修掉一个静默失效之后**才敢下这个结论。

---

## 1. Jev 到底是什么（先把名词钉死）

TypeSafe（创始人 Diogo Almeida，ChatGPT 共同发明人之一）2026-09-15 发布的 **Jev**。
它**不是**更快的 LLM，而是把"决策"从"生成"里剥出来的另一种模型：

| | 自回归 LLM | Jev |
|---|---|---|
| 输出 | 逐 token 顺序生成文本 | 一次并行查询给出**带类型的值** |
| 采样 | 逐个 token 采样 | **hardware-aware 并行采样**，所有选项同时打分、两阶段选择 |
| 候选数 | 全词表 | 预定义 schema，上限 255 个离散选项 |
| 训练 | RLHF / RLVR | **RLCD**（Reinforcement Learning for Calibrated Decisions），优化概率校准 |
| 延迟 | 3–329 s（对话级） | 70–500 ms |
| 计费 | 输入+输出都算 | 输出不计费（没有自回归生成） |

本机 `F:\github\openjev\` 是对这个**接口模式的开放复现**（不是 Jev 的权重）。它把可迁移的
那一半量化得很干净：同一个冻结的 Qwen3.5-4B、同一个 state、21 条 yes/no 判据、一张 3090 ——
**直接读 typed logits：1.023 s，输出 0 个 token** vs 自回归吐一个 JSON 数组：5.332 s / 111 token = **5.21×**。

**可迁移的那一条：不要生成你已经能用一次前向读出来的东西。**
不可迁移的是"决策原语"本身——那解决的是路由/分类/打分，不是"把代码写得快"。

---

## 2. v1 长这样：让确定性引擎先手出草稿，MTP 收尾

本机这套 llama.cpp fork **已经内置**了这个原语，不用新写模型：
`common/ngram-cache.cpp` 查一张预计算的 token 表直接给出下一个 token，**完全不跑模型前向**。
候选集预先算好、分数是个整数计数、一次查表并行给出 —— 形态和 Jev 的并行采样是同一个东西。

```
trainlib/                     真实代码语料（CSS/HTML/JS/TS/Py/文档…）
  └ build_cache.py  ──►  lookup/precise.lngc   41,587 条 / 1.11 MB
                         lookup/wide.lngc     210,479 条 / 5.62 MB
                                  │
llmserver-jev.bat ────────────────┘
  --spec-type ngram-cache,draft-mtp-adaptive
  --lookup-cache-static <表>
  [--lookup-cache-dynamic <自学习表> --lookup-cache-dynamic-save]
```

`--lookup-cache-dynamic-save` 是"越用越快"那一半：服务把自己服务过的会话里的 n-gram 写回文件，
下次启动带着上次的经验起来。默认关掉（原因见 §6 原因 1）。

---

## 3. 引擎的硬约束（读代码得到的，不是猜的）

建表之前必须知道这五件事，否则表怎么建都是错的：

1. **静态表只用 bigram 查。** `common/ngram-cache.cpp:178-182` 用 `LLAMA_NGRAM_STATIC`
   （`ngram-cache.h:12` = **2**）取键 ⇒ 文件里存 1/3/4-gram 全是死数据。
2. **只有"续接足够集中"的 bigram 才出草稿。** `ngram-cache.cpp:106-111`：
   `sum(count) >= 2` 且 `top1 >= 66% × sum(count)`，否则放弃。
3. **静态表还兼任判官。** `ngram-cache.cpp:139-148`：context/dynamic 表提的候选，
   若出现在静态表的续接集合里，score 被乘 **100**。所以覆盖面本身也有价值。
4. **多个 speculator 是"优先级抢先"，不是相加。** `speculative.cpp:3819-3868`
   按顺序问每个 impl，谁先产出非空 draft 谁赢（`dp.drafting=false`、`impl_last` 记它），
   后面的 impl 这轮不再跑。顺序**硬编码**在 `speculative.cpp:3597`：

   ```
   ngram-simple > ngram-mod > ngram-cache > draft-simple > draft-eagle3
                > draft-mtp > draft-mtp-adaptive
   ```

   → **所有 n-gram 引擎都排在 MTP 前面。** 这条是本次失败的全部根因。
   另：`speculative.cpp:3703` `const uint16_t n_draft = 8; // TODO get from config?`
   —— ngram-cache 每轮**无条件**草拟 8 个 token，**命令行没有任何开关能压短它**。

5. **缓存文件带 v2 头会被"词表指纹"校验拒收，而拒收只是警告、不是报错。**
   头部含 `n_vocab` + `vocab_hash`（`ngram-cache.cpp:241-261`：对每个 token 文本做 FNV-1a、
   每个后补一个 NUL 字节）。**实测 2026-09-22：我按代码复刻的指纹与运行时不符**（两边都数 248320 个 token），
   服务照常启动、`/health` 照常 200，日志里只有一行
   `W ... static lookup cache ignored: ... (same size, different token texts)`。
   ⇒ 改用**无头（older format）布局**：`ngram-cache.cpp:397-416` 明确支持，
   跳过指纹校验、**但仍然用真实词表尺寸校验每个 token id**，且日志会打印
   `has no header (older format), vocab identity not checked, N n-grams` —— 这行正好当"真加载上了"的证据。
   指纹本身仍记在 `lookup/*.json` 里备查。

---

## 4. 实测

### 4.1 权威数据（`--tag jev3`，**表已确认加载**，同会话，mixed ×3 取中位）

写代码（`code.ringbuffer`，192 max_tokens）：

| 配置 | t/s 中位 | 区间 | depth | accept | token/轮 | 单轮耗时 | vs 基线 |
|---|---|---|---|---|---|---|---|
| `llmserver-adaptive.bat`（只 MTP） | **45.75** | 41.48–46.84 | 2.906 | 0.6739 | 2.958 | 64.7 ms | — |
| `llmserver-jev.bat`（precise 表 41k） | 29.04 | 27.78–34.73 | 3.044 | 0.4191 | 2.276 | 78.4 ms | **−36.5%** |
| `llmserver-jev-wide.bat`（wide 表 210k） | **17.14** | 16.07–18.35 | 3.261 | **0.1744** | 1.569 | 91.5 ms | **−62.5%** |

写文章（`prose.platform`）：32.45 / 29.63 / 21.79 —— 温和表 −8.7%，激进表 −32.9%。

### 4.2 对照：上一轮（`--tag jev2`）表**静默没加载**时的数字

| 配置 | code t/s | depth | accept |
|---|---|---|---|
| `llmserver-adaptive.bat` | 47.41 | 2.738 | 0.7135 |
| `llmserver-jev.bat`（此时表被忽略，引擎只剩上下文表） | 32.93 | 3.19 | 0.5195 |
| `llmserver-jev-wide.bat`（同上） | 29.98 | 3.276 | 0.4872 |

⇒ 两个会话的基线相差 3.5%（45.75 vs 47.41），所以跨会话那一格只能当参考。
但方向是清楚的：**把语料表真的接上去之后，比"只有上下文表"还要更差，而且覆盖率越高越差。**

### 4.3 剂量-反应（这是本轮最有价值的一张表）

| 引擎能看到的东西 | code t/s | accept |
|---|---|---|
| 什么都没有（纯 MTP） | 45.75 | 0.674 |
| 仅当前上下文（语料表被忽略） | 32.93 † | 0.520 † |
| + 41k 条自然集中 bigram | 29.04 | 0.419 |
| + 210k 条强制取 top-1 的 bigram | **17.14** | **0.174** |

† 跨会话参考值。**单调下降**：引擎能"自信"说出下一个 token 的机会越多，吞吐越低。
如果 n-gram 引擎是在"帮忙"，曲线应该往上；它一路往下，说明它在**抢轮次**。

复现：

```powershell
python tools\ab_bench.py --bat llmserver-adaptive.bat --bat llmserver-jev.bat `
                         --bat llmserver-jev-wide.bat --prompts mixed --runs 3 --tag jev3
```

---

## 5. 机制诊断

看那对反向变化的量：**depth 涨**（2.906 → 3.044 → 3.261）、**accept 掉**（0.674 → 0.419 → 0.174）。
如果 MTP 还在赢，两项应该和基线一样。depth 顶着上限、accept 塌下去，
说明**大多数轮是 n-gram 引擎赢的，而它赢下来的草稿更差**。三个原因按贡献排序：

1. **优先级反转（架构层，主因）**：表一命中就把 MTP 这一轮**整个踢掉**。它不是在"补充"MTP，
   而是在"替换"一个 accept 0.67 的草稿器。所以表越勤奋损失越大 —— 这就是 §4.3 剂量-反应的来源。
2. **bigram 的"自信"和"对"没关系（数据层）**：wide 表把每条 bigram 强制压成 top-1，
   于是文件里每条都是"100% 集中"，必然通过引擎的 66% 门槛；但在真实上下文里
   **accept 只有 0.174** —— 也就是**它自信说出的下一个 token，82.6% 是错的**。
   语料里的全局高频 ≠ 当前上下文里的正确。
3. **草稿长度失控把验证批量撑大（成本层）**：`n_draft = 8` 写死 ⇒ "错在第一个 token 的 8 草稿"
   照样占满一轮验证。证据：token/轮从 2.958 掉到 1.569（−47%），**单轮耗时却从 64.7 涨到 91.5 ms（+41%）**，
   两下相乘就是 −62.5%。多出来的开销全在验证更长的草稿批上，白烧。

一句话：**在这个代码里，n-gram 表拿到的每一轮，都是从一个更好的草稿器手里抢来的；
抢来之后既没给出更多 token，又让每一轮更贵。**

---

## 6. 两个静默失效（方法论，比结论更值钱）

1. **`/health` 200 不证明缓存加载成功。** 词表指纹不符时只打一行 `W`，服务完全正常，
   引擎退化成"只有上下文表"。**第一版 A/B 整轮都是在这个状态下跑的**，
   直到去 grep 服务日志才发现。⇒ **凡"表/权重/插件类"参数，必须 grep 日志找加载证据行**
   （本线是 `static lookup cache loaded: ... (N n-grams)`），不能只看服务起来没起来。
2. **我自己造的文件格式必须验"能被真引擎读进去"**，而不是"我按文档写对了"。
   这次的指纹算法是从 `ngram-cache.cpp:241-261` 逐行复刻的，**照样对不上**
   （原因未追查，`token_get_text` 恒非空，`n_vocab` 也对得上；不值得为它反编译）。
   ⇒ 改用有官方支持、且日志会给确认行的**无头布局**，用可观测性换掉不可观测的正确性假设。

---

## 7. 那正确的接法是什么

方向没死，但要换接法。按性价比排序：

1. **真正有价值的是"链式"：MTP 出 k 个 → n-gram 从 MTP 草稿的末尾**接下去**。**
   这样两者是叠加而不是抢，才是"逻辑引擎 + 神经草稿器"的正确形态。
   注意 fork 里已有 `chain` 机制（`params.chain` / `LLAMA_SPEC_CHAIN`），但本线早前实测
   **是静默空操作**（无日志、无效果），要真得改源码 —— 这是下一步的正解。
2. **顺手必修：把 `n_draft = 8` 变成可配置，或至少收到 2。**
   第 5 节原因 3 的成本证据说它每轮都在白烧；即使不接 MTP 也该做。
3. **优先级翻转**（把 `has_ngram_cache` 挪到 `has_draft_mtp_adaptive` 之后，`speculative.cpp:3597-3630`）
   —— 但改完表只在"MTP 一个草稿都产不出"的轮里有用，**期望增益≈0**，不值得单独重编。
4. **换赛道（最像 Jev 的活法）：别做 next-token，做整段惯用式选取。**
   Jev 那个 5.21× 来自"一次前向读出 21 个判据、0 个输出 token"，对应到写代码不是"猜下一个 token"，
   而是 **FIM 式的整块选取**（闭括号 + 换行 + 下一个函数签名这种 10~20 token 一块）：
   命中一次就是一次大胜，而且天然不会去抢单 token 的活。
5. **（并行的那条线）** 词表短表 `--spec-draft-vocab-map` 的挂点在 `speculative.cpp:3463-3469`，
   打的是草稿头的带宽开销，与本线不冲突。

---

## 8. 这次实验杀死了什么、没杀死什么

**杀死了：**
- "把 bigram 表放在 MTP 前面当草稿器" —— 写代码 −36.5%（温和）/−62.5%（激进），同会话、区间不重叠。
- "语料集中度 = 上下文置信度" 这个隐含假设。被强制成 100% 集中度的表，实战 accept 只有 0.174。
- "n-gram 引擎是免费午餐" —— 它确实不跑前向，但它**抢轮次**，而轮次是要钱的。

**没杀死：**
- Jev 的核心主张（一次前向读结果，别逐 token 生成）。那条在"决策 / 分类 / 整段选取"上依然成立，
  `openjev/` 的 5.21× 就是证据。
- "代码有固定结构、可以预计算" 这个判断。这是**数据层**的问题（bigram 太浅、且没有语法约束），
  不是判断错了 —— 用它去做**更长的、带语法约束的、且不抢 MTP 轮次**的候选生成仍是对的方向。

---

## 9. 交付物清单

| 文件 | 作用 |
|---|---|
| `trainlib/` | 从 `F:\github` 21 个项目采样的代码训练库，980 文件 / 8.79 MB / 2.8M token，按类型分树、可增量补充 |
| `trainlib/collect.py` | 采集器（配额、去重、增量、跳过生成物与密钥） |
| `trainlib/build_cache.py` | 训练库 → `lookup/*.lngc`（复刻引擎接受规则与文件格式，用无头布局保证可加载） |
| `llmserver-jev.bat` | v1 启动器：`ngram-cache,draft-mtp-adaptive` + precise 表（默认） |
| `llmserver-jev-wide.bat` | 激进版（wide 表 + 自学习表），保留以便复现"更差"这个结论 |
| `tools/_ab_jev1.json` `_ab_jev2.json` `_ab_jev3.json` | 原始测量数据（jev3 = 表已加载的权威数据） |

`llmserver.bat`（生产）与 `llmserver-adaptive.bat` 全程未改动。
