# 一个模型，两个接口：对话 + JEV 判据，跑在 8 GB 笔记本上

> 2026-09-24 定稿。本文只讲**怎么做到的**和**参数是什么**，实测账在 `BENCH.md`，否决记录在 `JEV.md`。

---

## 0. 一句话

不是两个模型，不是两个进程，不是两份显存。
**一个 `llama-server.exe`、一份 6.21 GB 权重、一个 slot 池、同一个 `/v1/chat/completions`。**
对话和判据的差别**只在请求体这几行 JSON 里**。

```jsonc
// 判据：只读 logits，不解码任何 token
{ "max_tokens": 1, "logprobs": true, "top_logprobs": 40,
  "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
  "chat_template_kwargs": {"enable_thinking": false} }

// 对话：正常自回归生成
{ "max_tokens": 512, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
  "chat_template_kwargs": {"enable_thinking": true} }
```

---

## 1. 原理：为什么一个模型能同时干这两件事

两者在计算上是**同一条路径**，只是走多远不同：

```
prompt ──► prefill（N 个 token 一次前向）──► 最后一层的 logits
                                                 │
                    ┌────────────────────────────┴────────────────────────────┐
                    │                                                         │
      【判据】直接读候选槽位的 logprob                      【对话】采样一个 token
       A: -0.02  B: -3.71  C: -5.44 …                        → 塞回输入，再来一次
       softmax 归一化 → 选最大 → 结束                          （自回归，直到 EOS 或 max_tokens）
       输出 token 数 = 0，额外解码 = 0 次
```

- **判据就是「一次前向 + 读结果」，把自回归那一步砍掉。** 候选是有限集合（A~H、yes/no、路由名…），它们本来就在词表里，模型对它们本来就有 logits —— 不需要生成，直接读。
- **对话就是「一次前向 + 采样 + 循环」。** 同一个最后一层、同一份 logits，只是默认只取 argmax/采样值、丢掉其余。
- 所以两个接口**共用的不是"能力"，是同一份权重、同一套 prefill/decode 通路、同一块 KV 显存**。llama.cpp 的 server 本来就是多接口共享一个模型实例（`/v1/chat/completions`、`/completion`、`/embeddings`…），我们做的是**把这个"读 logits 的子集"当成一个正式接口来用**。

### 1.1 判据侧三个必须对的参数

| 参数 | 值 | 为什么必须是这个值 |
|---|---|---|
| `logprobs` + `top_logprobs` | `true` / `40` | 不返回 `top_logprobs` 就只能拿到采样出来的那**一个** token 的概率，没法在选项集合上比较 |
| `temperature` / `top_k` / `top_p` / `min_p` | `1.0` / `0` / `1.0` / `0.0` | **关掉一切截断**。任何截断都会把选项从 `top_logprobs` 里剔掉，导致 softmax 分母失真、概率全错。判据要的是**原始分布** |
| `repeat_penalty` | `1.0` | 同上，任何重排都会污染分布 |
| `chat_template_kwargs.enable_thinking` | `false` | 判据要的是「下一个 token 就是答案」，不能让模板先塞一个思考块 |
| `max_tokens` | `1` | 只是为了让 API 合法返回一个 logprob 槽位，**实际不产生任何新 token** |

拿到 `top_logprobs` 后，在**候选标签**上重新归一化（`bridge.py:104` `_softmax_restricted`）：

```
p(L) = exp(lp_L - max) / Σ_L' exp(lp_L' - max)      L ∈ {A…H}
```

得到的是**校准过的概率**，可以直接做门控 —— 实测里模型唯一算错的那道题（17×24），它的概率是 **0.668**，其余全在 0.99+。**概率真的携带不确定性**，这是生成式给不了的东西。

### 1.2 为什么这套在 8 GB 上能成立

| 项 | 大小 | 说明 |
|---|---|---|
| 权重 | **6.21 GB 恒定** | PTQ1_0 三值 ≈ 1.8 bit/权重 × 27B；与上下文长度、batch 无关 |
| KV | **73 KB/token** | 65 层 × 4 KV head × 256 维 × q4_0（0.5625 B） |
| KV @ 32k | 2.4 GB | ⇒ 权重 + KV ≈ **8.6 GB**，`-c 32768` 是这张卡的上限附近 |
| MTP 草稿短表 | **26 MB/步** | 没有短表时每步读 265 MB（`init_draft_vocab` 的紧凑副本） |

**权重压到 6.21 GB 是全部前提。** 27B 在 4 bit 下约 15 GB，8 GB 卡装不下；三值 1.8 bit 才塞得进。
这就是"基于 Qwen3.8-27B 在 8 GB 笔记本里"这句话的真实技术含量 —— 不是模型小，是**格式特殊**（PTQ1_0 + 自训 MTP 头 + 短表 + `--kv-mean-center` 标定 + 定制 GDN kernel）。

---

## 2. 串行交替：切来切去为什么几乎不花钱

**这是本方案最值钱的部分。**

### 2.1 三层机制

```
第 1 层  同一份权重、同一个 slot 池
        ⇒ 两个接口天然轮流用同一个 slot，显存零增量

第 2 层  主机内存 prompt cache（--cache-ram 默认 8192 MiB）
        会话切走时 server_slot::prompt_save 把「整段序列状态」搬进主机 RAM：
          · KV cache
          · GDN 的 recurrent 状态      ← hybrid 模型必须有这一份
          · MTP 草稿 context
        （llama-memory-hybrid.cpp:192/199，两半都写）
        切回来时 prompt_load 原样灌回

第 3 层  ubatch 边界的增量复用
        hybrid 的快照只能按 ubatch 边界保存
        （llama-memory-hybrid.cpp:87-91 [TAG_RECURRENT_ROLLBACK_SPLITS]）
        ⇒ 命中缓存后的重算量 ≈ 「1 个 -ub + 5 token」，与尾巴多长无关
```

### 2.2 实测：冷 → 热

| 场景 | 冷（首次） | 热（切回来） | `prompt_n` |
|---|---|---|---|
| 9659 token 会话 A/B 交替 | 19.7 s / 17.7 s | **0.45 s / 0.59 s** | 9659 → **10** |
| 4.2k 文档：对话建前缀 → 判据复用 | — | 判据 `prompt_n` **4204 → 1048** | 复用 75% |

**功能上也是对的**：共享同一段 system prompt 的两个会话，各自报出**自己的**口令，无串味。

### 2.3 两条硬条件（不满足就付全量）

| 条件 | 后果 |
|---|---|
| **system prompt 必须一致**（第一条消息逐 token 相同，差异只放尾巴） | 换掉 system ⇒ 全量重算 |
| **`enable_thinking` 不能频繁来回切** | 它改的是模板渲染出来的 **token 序列**，等于不同前缀 ⇒ **首次跨设置付一次全量**，之后两边状态各留一份 `cache-ram`，来回切只付 ~1 个 ubatch |

⚠️ 改历史**中间**任何一条消息，也是全量重算（前缀被截断）。

### 2.4 所以"交替工作"实际长这样

```
t=0s    对话开始（enable_thinking=true, 32768 ctx）
        └─ 首次 prefill 全量，之后每轮只新增 token
t=…     用户切去编辑器问一句  ──►  判据请求（enable_thinking=false）
        └─ 第一次：跨设置，付一次全量（~2.2 s @512tok state）
        └─ 以后每次：命中缓存，只重算 1 个 ubatch ⇒ 0.69 s
t=…     切回对话
        └─ 命中对话那份 cache-ram，付 ~1 个 ubatch 就回来了
```

**这就是"两个 API 共享一个模型"的全部成本结构：一次性跨设置的全量，之后每次 1 个 ubatch。**

### 2.5 `-ub 256` 是这条线的关键旋钮（已发货）

判据的延迟**正比于重算量**，而重算量 ≈ `-ub`。6 条判据实测：

| `-ub` | 判据延迟 | prefill（16536 tok） | decode |
|---|---|---|---|
| 1024 | 2.18 s | 466.7 t/s | 35.5 t/s |
| **256** | **0.70 s** | 462.0 t/s（**−1%**） | 35.1 t/s |
| 128 | 0.45 s | 444.7 t/s（−4.9%） | 35.2 t/s |
| 64 | 0.35 s | 361.4 t/s（−29%） | 34.8 t/s |

⇒ **`-ub 256` 是纯赚**（判据快 3.1×，对话 prefill 只慢 1%，decode 不变）。`ub128` 可选，`ub64` 排除。
**判据的精度与 `-ub` 无关**（读的是同一层 logits）。

---

## 3. 并行：真正能开几路，什么时候不该开

### 3.1 结论表（`-c 32768`，同卡实测）

| 配置 | 每路 t/s | 合计 | vs 单路 |
|---|---|---|---|
| 单路，MTP **on** | 37.3 | 37.3 | — |
| 单路，MTP off | 31.3 | 31.3 | — |
| 2 路，MTP **on** | 18.8 / 16.2 | **31.2** | **比单路还亏** ❌ |
| 2 路，MTP off | 23.7 / 23.7 | **46.4** | **+48%** ✅ |
| 10 路，MTP off | — | 45.6 | ≈ 2 路 ⇒ **2 路即饱和** |

**两条命令级规则：**

```
单会话交互（人机对话 + 判据交替）：  SLOTS=1，MTP on     → 37.3 t/s + 判据 0.7 s
多路并发（多个 agent 同时跑）：      SLOTS=2，MTP off    → 合计 46.4 t/s（代价：每槽 ctx 腰斩）
```

### 3.2 为什么"开 MTP 反而不能并行"

这不是调参问题，是**结构问题**：

- MTP 推测解码要求「起草 → 验证」在一个 ubatch 里配对完成。
- 两个 slot 各跑各的 draft→verify 节奏，**永远错相位**，合不到同一个 ubatch ⇒ 白开。
- 上游佐证：ggml 的 #23269 承认 "parallel decoding with MTP remains highly suboptimal"，#29002 指出 hybrid 模型的 co-batching 会破坏 recurrent/short-conv 状态（贪心发散、~25% 中途 EOG）。我们 fork 直接在 `speculative.cpp:3478` 把 `n_rs_seq` 置 0 回避了。

⇒ **MTP 和并行是二选一**，别指望相加。三种配置已穷尽（31.2 / 29.0 / 46.4），别重复实验。

### 3.3 还有一个更好的并行形态（待做）

**批判据（batched decisions）**：N 个判据共享**已缓存的 state**、并发发出，一次前向算 N 条序列。

为什么这个形态最值：**权重搬运量与 batch 无关** —— 一次前向读满 6.21 GB，出 1 个结果还是 N 个结果，搬运量一样。判据又不需要逐 token 循环，天生适合打包。

需要 `SLOTS≥2`（要改 bat），是下一步最该做的事。

---

## 4. 效果与速度账

### 4.1 判据

| 项目 | 实测 |
|---|---|
| 短 state agent 决策 | **11/12** 正确（错的那条概率仅 0.668） |
| 长文档（4.2k token）判据 | **6/6** |
| 答案稳定性 | 18 次请求逐字一致 |
| 延迟（短 state） | **~0.56 s** |
| 延迟（长文档，首次读入 8.6 s 之后） | **每条 0.69 s** |

### 4.2 三态对比（俄罗斯方块 demo，同题同模型）

| 模式 | 每手耗时 | 相对 |
|---|---|---|
| **判据（JEV）** | **646 ms**（中位） | 1× |
| 生成式（`max_tokens:12`） | 898 ms | 1.4× |
| 思考模式（512 token） | **15030 ms** | **23×** |

整局成绩：存活 150 块、消 56 行、挑中引擎最优 **82.7%**。

⚠️ 注意一个容易误读的点：判据的 646 ms 里 **prefill 占大头**（state 300 多 token），不是"读 logits 便宜"。公平对照下判据与生成式基本打平（563 vs 557 ms）—— **判据的收益不在单次延迟，在于①输出天然结构化、②概率可做门控、③批量时一次前向摊多份**。

### 4.3 提示词的设计规则（实测得出）

**必须把每个选项的后果列成可比数字**，否则模型等于随机：

```
A r0 c3 L1 H2 T5 B3 S-8
B r1 c7 L0 H4 T9 B5 S-21
...
ANSWER:
```

- 只给 `rotate/column` 时：模型乱选（最高概率 21%），离线 40 局 **25 块就堆死 / 0 行**。
- 带上后果列 + system 一句「S 越大越好」后：**150 块 / 56 行 / 不堆死**。

⇒ **让 LLM 当决策头，喂"选项 + 后果"比喂"选项 + 规则"有效得多。** 模型不擅长自己在脑子里评估规则，擅长在已算好的整数上比大小。

---

## 5. 参数速查（照抄即可）

### 5.1 服务端（`llmserver.bat`，1679 B，6 个旋钮）

```bat
set "SLOTS=1"      rem 1 = 单会话独占整个 -c；2 路已饱和
set "CTX=32768"    rem 越大越吃显存（73 KB/token）
set "DRAFT=3"      rem 草稿深度上限；短提示 1 更快，长输入 3 更好
set "REA=on"       rem 思考开关，客户端可覆盖
set "EFFORT=low"   rem 仅 thinking 时生效
set "TEMP=0.6"

"%~dp0bin\llama-server.exe" -m "%~dp0models\...-mtp.gguf" ^
  --host 127.0.0.1 --port 18080 --parallel %SLOTS% -rea %REA% ^
  -c %CTX% -b 4096 -ub 256 -t 8 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 ^
  --temp %TEMP% --top-p 0.95 --top-k 20 --reasoning-effort %EFFORT% ^
  --reasoning-preserve --kv-mean-center "%~dp0calib\kv-bias....gguf" ^
  --reasoning-budget 2048 --reasoning-budget-message "\n\nBudget exceeded, answer directly.\n" ^
  --spec-type draft-mtp-adaptive --spec-draft-n-max %DRAFT% --spec-draft-n-min-adaptive 1 ^
  --spec-draft-vocab-map "%~dp0maps\shortlist_24k_mix.vocab.txt" --spec-draft-vocab-hot 0
```

关键几行的作用：

| 参数 | 作用 |
|---|---|
| `-ub 256` | 判据延迟 2.2 s → 0.7 s（唯一生效的旋钮） |
| `-fa on -ctk q4_0 -ctv q4_0 --kv-mean-center` | KV 量化 + 偏置标定，省显存；三者必须配套 |
| `--spec-type draft-mtp-adaptive` | 自训 MTP 头，单路 31.3 → 37.3 t/s |
| `--spec-draft-vocab-map` + `--spec-draft-vocab-hot 0` | 短表：每步读 26 MB 而非 265 MB；中文 +29.8% |
| `-c 32768` | 权重 6.21 + KV 2.4 ≈ 8.6 GB，8 GB 卡的上限档 |

⚠️ `--cache-ram`（默认 **8192 MiB**）、`cache_prompt`（默认 true）、`--cache-idle-slots`（默认 on）**都不用显式写**，是交替复用能工作的原因。

### 5.2 客户端

| | 判据 | 对话 |
|---|---|---|
| `max_tokens` | `1` | 512+ |
| `logprobs` / `top_logprobs` | `true` / `40` | 不需要 |
| 采样 | `temp 1.0 / top_k 0 / top_p 1.0`（全关） | `temp 0.6 / top_p 0.95 / top_k 20` |
| `enable_thinking` | `false` | `true` |
| `cache_prompt` | `true` | `true` |

---

## 6. 诚实定位：哪些是通用的，哪些是我们的

免得把 llama.cpp 自带的能力算成自己的发明：

| | 归属 |
|---|---|
| `max_tokens:1 + logprobs` 读判据 | **通用机制**（llama.cpp 自带，openjev 复现的是这个模式） |
| 多接口共享一个模型实例 | **通用机制**（server 的设计） |
| 主机内存 prompt cache / `--cache-ram` | **通用机制**（但"对话+判据共享同一份缓存、只付 1 个 ubatch"这个用法要自己验） |
| `--parallel` 多 slot | **通用机制** |
| **PTQ1_0 三值格式 + 自训 MTP 头** | **我们的**（27B → 6.21 GB，8 GB 卡的前提） |
| **24k 短表（保持头量化格式的紧凑副本）** | **我们的**（中文 +29.8%、中文注释代码 +15.6%） |
| **`--kv-mean-center` 偏置标定 + 配套 q4_0 KV** | **我们的** |
| **定制 GDN kernel（避免 upstream co-batching 状态损坏）** | **我们的** |
| **`-ub 256` 这个具体档位** | **我们的实测结论**（不是文档里的推荐值） |

**真正少见的是这个组合**：把「一次前向读判据」当**与生成并列的一等接口**，接在同一个 server 实例、同一份显存上，并且量化验证了「两者共享 KV、来回切只付 1 个 ubatch、互不串味」。
通用零件搭出一个能用的整机 —— 亮点在整机，不在零件。

---

## 7. 入口与复现

| 要做什么 | 去哪 |
|---|---|
| 起服务 | `F:\super\qwen38mtp\llmserver.bat` → `127.0.0.1:18080` |
| 看判据跑起来长什么样 | `F:\super\jev-tetris\start.bat`（一键：起服务 → 桥接 18090 → 开浏览器） |
| 判据提示词唯一出处 | `F:\super\jev-tetris\bridge.py` 的 `SYS` / `build_user` |
| 测判据速度 | `F:\super\jev-tetris\tools\playtest.py <块数>` |
| 测三种模式对照 | 页面上的模式切换（jev / gen / think） |
| 判据机制验证脚本 | `F:\super\tmp\jev_poc.py`、`jev_v2.py`、`jev_v3.py`、`jev_share.py`、`jev_share2.py`、`ub_side_effects.py` |
| 完整 Benchmark 账 | `docs\BENCH.md` |
| n-gram 那条线为什么被否决 | `docs\JEV.md` |
