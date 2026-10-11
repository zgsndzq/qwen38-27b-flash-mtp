# Ternary 提速调研：4060 上还能不能翻翻

> 2026-09-25。外部调研（论文 / llama.cpp 上游 PR / 社区实测）+ 与我们自身实测的对照。
> 结论先给：**能，但不在「更聪明的三值算法」上翻，在两个工程点上翻。**

## 0. 我们的基线（自测，作为全部推算的输入）

| 量 | 值 | 来源 |
|---|---|---|
| 权重 | 6.21 GB 恒定（PTQ1_0） | 实测 |
| 显存带宽 | **≈224 GB/s**（由 32.5 t/s × 6.21 GB ÷ 0.90 反推） | 推算 |
| 裸解码 | 32.5 t/s（= 202 GB/s ≈ **90% 带宽**） | `llama-bench tg128` |
| 轮时间结构 | `ms_round = 30.1 + 10.2 × depth`（r²=0.985） | 实测 |
| prefill | 540 t/s（pp2048） | 实测 |
| 单会话 A/B | MTP on 37.3 / off 31.3 | A/B 实验 |
| L2 | **32 MB**（RTX 4060 Laptop） | 规格 |

---

## 1. 先算清楚：极限在哪

### 1.1 主干前向已经到顶，没有余量

6.21 GB ÷ 224 GB/s = **27.7 ms**，实测固定项 **30.1 ms** ⇒ 92% 带宽效率。
⇒ 主干单次前向**不可能再快**。旁证：我们自己算过「算力侧砍光也只值 11%」。

### 1.2 起草步骤是唯一的窟窿

| | 搬运量 | 带宽极限 | 实测 | 低效倍数 |
|---|---|---|---|---|
| 主干 1 次 | 6.21 GB | 27.7 ms | 30.1 ms | 1.09× |
| MTP 起草 1 步 | ~266 MB | **1.19 ms** | **10.2 ms** | **8.6×** |

⇒ 一个 round（depth=3）：60.7 ms 里，**约 27 ms 是起草的纯开销**。
⇒ 若把起草治到带宽极限：round = 30.1 + 3.6 = 33.7 ms，accept 不变（3.48 token/round）
⇒ **57 t/s → 103 t/s = 1.8×**。

**这是「翻翻」的全部希望所在，且只在这一件事上。**

## 2. 关键发现：这个 8.6× 是已知的系统性问题，外部独立复现了

外部实测（`leanzero.net`，Qwen3.6-27B + MTP，Apple Silicon）：

> 起草一步需要 65 层（290.5 MiB）+ 输出投影（248320 词表，1.04 GB）。合计约 1.35 GB。
> 主模型 43.3 ms 搬 18.3 GB ⇒ 约 420 GB/s。按这个速率 1.35 GB 该花 3.2 ms，实际花 **30 ms**。
> **起草步骤每字节的效率比主模型自己的前向差 9 倍。**

他们的机器、模型、量化都不一样，但**低效倍数与我们几乎一致（9× vs 8.6×）**。
⇒ 这不是我们的配置问题，是 `draft-mtp` 实现的通性。

作者还点出两个我们可能也中招的点：
- **起草步要跑 output projection**（1.04 GB，权重与主模型共享）——这是「266 MB」这个估算之外的一大块质量。
- **`--spec-draft-p-min` 是唯一把它救回来的旋钮**（他试出 n-max 3 + p-min 0.8 → +4%）。
  ⚠️ 我们的 bat 当前是 `--spec-draft-p-min 0`（等于关掉这个闸门）。

另一篇（`authon.dev`）列了 MTP 失效三因，**第一条我们能一行 env 就验**：
> **CUDA graph 重捕获**：MTP 每步接受的 token 数可变 ⇒ 动态形状 ⇒ 图每步重捕 ⇒ 图优化的收益全丢，**每步开销反而上升**。

我们 bat 里开着 `GGML_CUDA_GRAPH_OPT=1` ⇒ 这条必须验。

## 3. 网上现成的 ternary 改造，逐条裁决

| 方案 | 是什么 | 对我们 |
|---|---|---|
| **BITCOS**（arXiv 2609.16338，2026-09） | 位图+压缩符号，2−z bits/weight，**直接拿 PrismML fork 当基准** | ❌ **实测对我们 0.96×** |
| **T-MAC**（MSRA, EuroSys'25） | LUT 免反量化 mpGEMM，CPU 4~5× | ❌ CPU/NPU 专用 |
| **bitnet.cpp**（Microsoft） | I2_S / TL1 / TL2 LUT kernel；GPU 支持 2025-05 才上 | ❌ 主战场是 CPU |
| **tritllm-kernel**（2026） | 免乘法 ternary GEMV，Ada `dp4a`，RTX 4080 Laptop **5.8× vs FP16 cuBLAS** | ⚠️ 见下 |
| **TQ2_0 vs TQ1_0**（llama.cpp #8151） | aligned 2-bit vs 5-trit/byte | ✅ **关键线索** |
| **稀疏 ternary GEMM**（ACM TECS 26） | 只存非零索引 + 加法替代乘法 | ❌ 见下 |
| **Codebook 1.5bpw**（ISLPED'26） | 64 项码本 + 6-bit 索引 | ❌ 要 VQ 微调 + 专用硬件 |
| **Leech lattice VRAM 布局** | bit-plane 优于 one-hot mask | ❌ 要重打包重训，收益 1.1~1.4× |

### 3.1 BITCOS：**论文里测的就是我们的模型，结论是对我们无效**

论文测了 29 个 SOTA 三值 checkpoint，**Bonsai 27B 在列**：

| 模型 | 零密度 | BITCOS bits/weight |
|---|---|---|
| CAT-Q Qwen3-1.7B | **51.48%（最高）** | 1.485 |
| BitNet b1.58 2B4T | 42.19% | 1.578 |
| **Bonsai 27B** | **29.66%（最低）** | 1.703 |

BITCOS 的判据是零密度 z > 0.375 才赢过五进制打包 —— **Bonsai 27B 是最低的 29.66%**，
所以论文自己给出 **0.96×**（略差于五进制打包）。

⇒ 两个独立证据同时指向：**我们的权重格式已经接近实际极限，没有「更紧凑」的空间可挖。**
（论文另注：全文**无任何 NVIDIA/CUDA 结果**，GPU 侧只测 Intel Xe2。）

### 3.2 TQ2_0 vs TQ1_0：**「解包成本 > 省下的字节」，这条直接指向我们的 prefill**

llama.cpp #8151 的实测（CPU）：

| 格式 | bpw | 相对速度 |
|---|---|---|
| TQ1_0（5 trit/byte） | **1.6875** | 基准 |
| TQ2_0（aligned 2-bit） | 2.0625 | **快 1.66~2.44×**（客户端 CPU） |

原因写得很直白：**TQ1_0 footprint 最小但 prefill/decode 吞吐更低** ——
五进制打包要额外解包，**这个开销比省下的字节更贵**。

对我们的意义**只在 prefill**：
- prefill 540 t/s ⇒ 权重流量仅 **1.64 GB/s**（6.21 GB / 2048 token / 3.79 s）⇒ **远不是带宽受限，是算力/解包受限**
- ⇒ 这里才有「换格式」的空间。decode 相反：1.69 bpw 已经吃到 90% 带宽，换 2.06 bpw 会让权重涨到 7.5 GB、decode **变慢**。

⚠️ 但**没有任何 NVIDIA 实测数据**：BITCOS 论文明确没测 NVIDIA；TQ2_0 的 1.66~2.44× 是 CPU 数字。
⇒ 属于「值得试，但不能承诺」的一档。

### 3.3 tritllm：**看起来最香，但我们吃不到**

它在 RTX 4080 Laptop 上 5.8× vs FP16 cuBLAS、**有效带宽 439 GB/s**；4090 上 **1940 GB/s**。
1940 > 4090 的 HBM 带宽（~1008 GB/s） ⇒ **它吃的是 L2**（作者自己说：L2 大小与加速比强相关，
当层能装进 L2 时有效带宽达 HBM 的 2×）。

**我们的 L2 只有 32 MB，权重 6.21 GB —— 装不进。这条红利拿不到。**
且它的对比基线是 FP16 cuBLAS，不是我们的 ternary kernel，5.8× 不可直接外推。

### 3.4 稀疏 ternary GEMM：我们是最没料可挖的那一档

ACM TECS 2026「加法替代乘法」的稀疏 GEMM：50% 稀疏下 4× 理论加速，实测比 Eigen/PyTorch/cuSPARSE
快 1.3~6.9×（RTX-3080Ti 上 Llama-3 8B 只有 7 t/s）。
它靠「只存非零索引」省字节 —— 但 **我们的零密度是 29.66%，29 个模型里最低**。
⇒ 起点就比设计它的场景差一半以上。

## 4. 上游现成可合的补丁（llama.cpp 2026）

| PR | 内容 | 打哪 |
|---|---|---|
| **#26150** | **CUDA 支持 Q2_0**（超低比特 2-bit） | decode/prefill 都可能 |
| **#26079** | **mvq↔MMQ 交叉点运行时可调**（batch>1；专治小显存 GPU 加大 batch 的退化） | 并发 |
| MMQ stream-k 降开销 | fastdiv + 动态 tiling，**明确针对 prefill 阶段** | **prefill** |
| #26120 | MMQ tile 放不进 smem 时回退 cuBLAS | 稳定性 |
| #26135 | `/metrics` + `GET /memory` 显存监控端点 | 观测 |
| #26160 | QFX16/32 无损 2.05× 压缩 | 对我们 1.69bpw 权重无意义 |

`--spec-type mtp` **已改名为 `draft-mtp`**（旧名静默失效）—— 我们是新名，没问题。

## 5. 一个反直觉的发现：MTP 可能正在拖慢我们的 agent 场景

外部两条实现级事实：

1. **「MTP 让 prompt processing 慢约 17%」**（`saram.io`）⇒ 我们相当于一直在付一条 prefill 税。
2. **「盈亏平衡点约 50 个生成 token」**，少于 50 token 时 prefill 税吃掉收益。
3. **「不要开并行槽，`-np 1` 是必需」**、**不要与多模态同用**、**MoE 上收益不明显** —— 前两条与我们已确立的硬规则一致。

对照我们的流量形状：
- **agent 一轮 84% 在 prefill**；**JEV 判据只出 1 个 token**
- ⇒ 在这两种形状上，MTP 的 17% prefill 税**极可能是净亏**
- 且我们自己测到的 MTP 增益只有 **1.19×**（37.3 vs 31.3），远低于社区宣称的 1.71~1.85×
  ⇒ 说明我们已经在「带宽受限 ⇒ 推测解码收益缩小」的那一段梯度上

⚠️ 我们的速度结论**全部是在「聊天形状」（中等提示、长输出）上测的**。
agent 形状（长提示、短输出、含 1-token 判据）**从未单独测过**。

## 6. 结论

### 能「翻翻」的两条路（都不需要重训、不需要新模型）

1. **起草步骤 8.6× 低效 → 目标 1.8×（decode 侧）**
   算术上成立，外部独立复现，但**网上没有现成补丁** —— 只能自己测出来再改。
2. **prefill 的解包开销 + 上游 MMQ 优化 → 目标 1.2~1.6×（prefill 侧）**
   prefill 是算力受限不是带宽受限，且是 agent 的主战场。

### 明确「翻不动」的四条

① 更紧凑的三值格式（我们已是 29 个模型里最紧凑的，BITCOS 实测 0.96×）
② L2 红利（32 MB 装不进 6.21 GB）
③ CPU 侧 LUT 类方案（T-MAC / bitnet.cpp）
④ 稀疏化（我们零密度最低，最没料）

## 7. 建议的验证顺序（全部零训练，最便宜的先来）

| # | 实验 | 验什么 | 成本 |
|---|---|---|---|
| 1 | `GGML_CUDA_DISABLE_GRAPHS=1` 对照 | 起草步是否每步重捕 CUDA graph | 一行 env |
| 2 | `--spec-draft-p-min 0.8`（现为 0） | 外部唯一救回来的旋钮 | 改一个数 |
| 3 | 起草头 Q6_K/Q4_K → Q8_0/Q4_0 对照 | K-quant 慢 kernel 假设 | 重导头 |
| 4 | **MTP on/off 用 agent 形状重测** | MTP 在 84%-prefill 流量上是否净亏 | 换 trace |
| 5 | 合 #26150 + #26079 + MMQ stream-k | prefill 上游红利 | 重编 |

⚠️ 全部按测量铁律：≥3 提示 × ≥3 次取中位、**同会话**、测前确认同卡只有一个 server、
A/B 两臂同设 `cache_prompt:false`。

---

## 8. 换专用引擎 NInfer？—— 换不上，但它给了我们三条硬东西

> 2026-09-25 追加。调研对象：`github.com/Neroued/ninfer`（Apache-2.0，2026-06-26 首次提交）。

### 8.1 三道墙：装不上，不是"跑得慢"

| 墙 | 它要的 | 我们有的 | 性质 |
|---|---|---|---|
| **架构** | `sm_120a`（RTX 5090 / Blackwell）。README 原文：**"The build rejects CUDA architectures other than `sm_120a`"** | RTX 4060 Laptop = **sm_89**（Ada AD107） | 🔴 **硬**：`cmake` 阶段直接拒绝，**一个 binary 都不产出** |
| **系统** | 64-bit Linux + CUDA 13.1 + CMake 3.28 + C++20 + Ninja + FFmpeg dev + libcurl≥7.85；**无 install target、无预编译包**，必须自己建 | Windows | 🔴 **硬** |
| **权重** | `.ninfer` 容器：`groupwise-int`（Qwen3.6-27B = **16.29 GiB** / 5.03 bpw）或 **NVFP4**（Blackwell 原生 FP4 格式） | 三值 PTQ1_0，**6.21 GB** / 1.69 bpw | 🔴 **硬**：① 8 GB 显存**装不下 16 GiB 的一半**；② 官方支持的格式是 **BF16 / INT8 / FP8 / NVFP4 / groupwise-int —— 里面没有 ternary，没有 TQ1_0 / PTQ1_0** |

**连带损失**（即使三道墙都拆掉）：自训 MTP 头（accept 0.827）、24k 草稿短表（中文 +29.8%）、
`--kv-mean-center` 标定、定制 GDN kernel、三值权重 —— **换引擎等于全部作废**。

⚠️ 另有：NVFP4 是 **Blackwell 原生**格式，Ada 没有原生 FP4 tensor core。

### 8.2 它的速度数字**不能外推**到我们身上

- 它对 llama.cpp 的优势是**对着 5-bit 配置**比出来的（`Q5KM` 权重 / `q5_1` KV ≈ **5.5 bpw**，
  另一份对比用 NVFP4+FP8 KV）——**字节量和它自己同一档（5.03 bpw）**，所以 kernel 效率优势才能显出来。
  引用的数字：4.7× prefill（1K ctx，5090）、1.4× decode（1K）、2.8×（128K）。
- 我们是 **1.69 bpw**，**字节量上已领先 2.4~3×**。decode 是带宽受限 ⇒ **字节量直接决定速度**。
- 社区初步重跑已打折：543 → **~448 t/s**；另有人指出它 **27B prefill 未必胜过 NVFP4 的 vLLM**。
- 它自己的 5090 单请求数字（Qwen3.8-27B nvfp4）：prefill 7,680 tok **8,340 t/s**、260k tok **2,203 t/s**、
  结构化 MTP3 decode **219.8 t/s**。换算：27B × 4bpw ≈ 15 GiB ⇒ 若按 2,203 t/s 的 prefill 流量约
  **114 GB/s**，而 5090 有 1,792 GB/s ⇒ **它 prefill 也远未吃满带宽**，说明"专精引擎"并没有凭空变出带宽。

### 8.3 ⭐ 但它证明/给了三条对我们很硬的东西

#### ① GDN hybrid 可以 co-batch —— 上游说"需大重构"，一个个人项目已经做出来了

NInfer 的 kernel 清单明确包含 **`gated delta rule`**（= Gated DeltaNet = 我们的 GDN recurrent），
它跑的 `Qwen/Qwen3.8-27B` 就是 GDN hybrid 架构。而它的并发实测：

| 并发 | Qwen3.8-27B nvfp4 稳态 decode | MTP3 accept |
|---|---|---|
| C=1 | 143.8 tok/s | 48.9% |
| C=2 | 267.6 t/s（**1.86×**） | 48.1% |
| C=4 | 461.1 t/s（3.21×） | 45.8% |
| C=8 | **766.6 t/s（5.33×）** | 46.0% |

**8 路、接受率 46% 稳定不崩。** 对照 llama.cpp 现状：#29002 说 hybrid co-batching 会
**破坏 recurrent 状态、约 25% 中途 EOG**；#23269 承认"需大重构"。

⇒ **我们"2 路即饱和"的实测没错**（那是 llama.cpp 当前实现的事实），
**但"架构上不可能"这个推论要修正**。差别在实现：第三方资料描述它用了
**"replay-based SSM optimizations"** —— 这正是我们 `llamAmpere_gdn` 里
`[TAG_RECURRENT_ROLLBACK_SPLITS]` 那条线**要走到的终点**（我们目前只能对齐 ubatch 边界回滚，
它做到了跨并发槽的完整状态回放）。

⚠️ 诚信标注：`replay-based SSM` 出自二手英文综述（内容风格疑似 SEO 生成），**非官方 README**；
但**官方并发表是硬数据**，"GDN 能 co-batch"这个架构结论成立。

#### ② `--lm-head-draft`：专用 LM head 草稿路径 —— 正打在我们最大的窟窿上

我们的起草步低效 **8.6×**（266 MB 该 1.19 ms、实测 10.2 ms），外部独立复现 **9×**，
且 `leanzero.net` 指出起草步除主干外**还要跑 output projection（1.04 GB，与主模型共享词表权重）**。

NInfer 把这块做成了**独立选项**：默认推荐配置里就是 `--spec mtp --draft-tokens 3 --lm-head-draft`，
文档称 "a dedicated LM-head draft path"。
⇒ **命名本身证实：output projection / LM head 的草稿开销可以被单独优化掉**——
这正是我们 §6 第 1 条要走的方向，且它已经证明这条能落地。

#### ③ pinned host memory 做状态槽 —— 我们能改，直打切换延迟

NInfer 的服务器参数：`--device-state-slots 2 --host-state-slots 8 --host-kv-mib 8192`，
README 原文：**"two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host KV"**。

我们的现状（**源码核对过**）：
- `tools/server/server-task.h:588 struct server_prompt_data` 用 **`std::vector<uint8_t> main/drft`**
  ⇒ **普通堆内存，不是 pinned**。
- 同一份 fork 里 `src/llama-context.cpp:2883` 的注释证明作者知道 pinned 的价值：
  「a pinned reallocation costs a cudaFreeHost/cudaMallocHost pair (**~60 ms at 90 MiB**)」
  ⇒ **输出缓冲区是 pinned 的，prompt cache 不是。**

⇒ H2D/D2H 走 pinned 可做零拷贝 DMA —— **这正是我们"切换 0.5 s"那 0.5 s 的成本来源**。
可改、可测、不碰主干。（次要点：它的 `--host-kv-mib` 默认量级 8 GiB，和我们的 `--cache-ram 8192` 巧合一致。）

### 8.4 顺带一个有意思的对照：它的 MTP 接受率比我们低 20~34 个点

| | 底座 | MTP 头 | accept |
|---|---|---|---|
| **我们** | Qwen3.8-27B 三值 PTQ1_0 | **自训** | **0.827** |
| NInfer | Qwen3.8-27B nvfp4 | 原厂头 + FP4 量化 | 0.489（C=1） |
| NInfer | Qwen3.6-27B nvfp4 | 原厂头 | 0.693（C=1）／0.682~0.690（groupwise-int） |

⚠️ 口径不完全可比（提示类型不同：它用 AIME/结构化 fixture，我们用混合提示；量化也不同）。
但方向清楚：**自训 MTP 头是我们的真资产，而它恰好是 NInfer 这个配置最弱的一环。**
换引擎 = 丢掉我们最强的那块，去补它最弱的那块。

### 8.5 可抄清单（按性价比排序）

| # | 抄什么 | 打哪 | 成本 | 优先级 |
|---|---|---|---|---|
| 1 | **exact-batch CUDA Graph** —— 它的 decode 描述是 "**exact-batch** CUDA Graph"，即**把形状钉住再捕图** | 我们开着 `GGML_CUDA_GRAPH_OPT=1`，但 MTP 每步接受 token 数可变 ⇒ 动态形状 ⇒ **每步重捕，图优化白给** | `GGML_CUDA_DISABLE_GRAPHS=1` 一行 env 先验 | ⭐ 立刻 |
| 2 | **独立草稿头（LM-head draft path）** | 起草步 8.6× 低效（decode 侧最大单点） | 改源码 | ⭐ 高 |
| 3 | **prompt cache 改 pinned host 内存** | 切换延迟 0.5 s | 改源码，改动小 | 中 |
| 4 | device/host 两级槽位设计（2 + 8） | 我们的 `--cache-ram` 已同构 | 已具备 | — |
| 5 | `--preserve-thinking` ↔ 我们 `--reasoning-preserve` | 已有 | 已具备 | — |

### 8.6 结论一句话

**换引擎：不行（sm_120a 编译拒绝 + 16 GiB 装不进 8 GB + 它的格式没有三值）。**
**但在同一批调研里，它替我们验证了两件事**：GDN hybrid 的 co-batch 是**实现问题不是架构问题**；
起草头的 LM-head 开销**可以独立摘掉**。
**再加上一条我们自己查源码查出来的**：prompt cache 用的是普通堆内存，**改成 pinned 就能白拿一段切换延迟**。

⇒ **不要换引擎，要抄它的做法——而且抄的三条全都落在我们已有的窟窿上。**
