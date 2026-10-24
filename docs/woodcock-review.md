# woodcock 调研：有什么能借，有什么已经证明不用碰

调研对象：`AtomicBot-ai/woodcock`（2026-10-22 建仓，MIT，8 stars，Python/MLX）
相关文档：`docs/research.md`（论文索引）、`docs/journal.md`（逐日实验与失败记录）、
`docs/design-vocab-trim.md`、`docs/design-draft-length.md`

**结论先说**：没有可直接移植的代码（它是 MLX/Python，C++ 部分只占 1 KB 的
CMakeLists）。真正有价值的是**两条被它们证伪的方向**和**一处方法论差异**。

---

## 1. 它是什么，凭什么快

一句话：**EAGLE-3 风格的训练型 drafter + 词表裁剪 + 按实测延迟选草稿长度**，
在 Apple Silicon 上把 Qwen3-1.7B 从 120 tok/s 拉到 240 tok/s（2.00×）。

它的四层归因（`journal.md`，1.7B，M4 Max）：

| 改动 | 效果 |
|---|---|
| 读 target 三层 hidden state 的 1 层 drafter + 单次同步的链式验证 | 比 mlx-lm 快 1.24×（全词表 head） |
| draft head 从 151,936 削到 16k 最常用 token | draft step 1.75 → 0.61 ms；tokens/round 2.09 → 2.07（几乎不变） |
| 每轮按实测延迟选链长（不是固定 γ） | 每个 domain 都不低于最佳固定 γ（math 选 4.6，multilingual 选 2.0） |
| 用 9× 数据重训 drafter（80k 条 target 自写答案） | tokens/round 2.15 → 2.94；1.56× → 2.00× |

**它的核心命题**：acceptance 是天花板，所有工程优化只能逼近它。
最后那 +28% 是**重训模型**换来的，不是改代码。

---

## 2. 我们和它的对照

| | woodcock | 我们（27B / RTX 4060） |
|---|---|---|
| 目标模型 | Qwen3-1.7B / 4B，bf16 | 27B PTQ1_0（5.79 GB） |
| 草稿机制 | 训练过的 1 层 drafter，fuse target 的 low/mid/high hidden | 模型自带 MTP nextn 头 + ngram-mod 查表 |
| 加速 | 2.00× / 2.23× | **3.9×**（46–48 → 177–187 t/s，ngram 主导） |
| 词表裁剪 | 16k / 151,936 = **10.5%** | 24k / 248,320 = **9.7%** |
| 深度策略 | 每轮用延迟模型算 γ* = argmax E(γ)/T(γ) | 爬山控制器（accept/reject 计数 + 手调阈值） |
| 验证成本 | n=1..8 只贵 32%（9.36 → 12.38 ms） | 同样平坦（DRAFT_N=2 与 4 实测同速） |
| tree 草稿 | 已实现（EAGLE-2 风格 top-k 树 + 递归拒绝采样） | 未做，只有线性链 |

**加速比不可直接比**：他们从 120 tok/s 起跳，我们从 46 tok/s 起跳。我们的
3.9× 里绝大部分是 ngram（BENCH=1 切掉后立刻掉回 60–80），MTP 单独大约只值 1.4×
（按 α≈0.6、c≈0.19 代入 L23 公式算出来 γ*=2、上限 1.44×）。

---

## 3. 值得借鉴：一处

### 3.1 深度选择从"爬山"换成"成本模型"

**他们的做法**（`design-draft-length.md`）：每轮直接算

```
E(γ) = (1 − α^(γ+1)) / (1 − α)          # Leviathan et al. Eq.1
T(γ) = t_v(γ+1) + γ·t_s + t_u           # 实测延迟，不是常数
γ*   = argmax_γ E(γ)/T(γ)
α̂    = (S + m·α₀) / (S + F + m)          # 窗口估计，W=16, m=4, α₀=0.6
```

关键是 **t_v(γ+1) 用实测值而不是假设为常数**（他们测出 n=1..8 从 9.36 涨到
12.38 ms），以及 α̂ 用带先验的窗口估计而不是裸计数。

**我们的做法**（`common/speculative-adaptive.h`，92 行）：连续全接受记
`n_climb`，累积漏接受记 `n_drop`，靠两套阈值往上爬/往下掉，阈值是手调的阶梯
（floor 和 depth 处便宜、中间贵）。**没有任何延迟/成本输入。**

**差异的实际影响**：

- 爬山是反应式的，换 domain 要试几次才知道该升还是降；成本模型每轮直接算。
  他们实测 auto 在每个 domain 都不低于最佳固定值（math 4.6 / code 3.1 / chat 2.5 /
  multilingual 2.0）。
- 但在我们的卡上 γ 的敏感度本来就低（t_v 平坦 + c 小），27B 上 DRAFT_N=2 和 4 同速。
  **9B 上就敏感了**（n-max 2 → 112 t/s，n-max 4 → 96–105），所以这条在 9B 上可能值
  几个百分点，27B 上大概率不值。

**要做的话**：需要 t_v(n) 和 t_s 的实测表（一个一次性标定脚本），然后替换
`speculative-adaptive.h` 的爬升逻辑。不是小改动，但边界清晰。

---

## 4. 已被证伪：两条（省时间，别做）

### 4.1 扩大 draft 词表 —— 无用

我们的直觉会是"24k 只覆盖 95% 的 token，扩到 48k 也许接受率更高"。
他们实测打掉了：

| head | cover | tokens/round |
|---|---|---|
| full 151,936 | 100% | 2.09 |
| 48k | 98.5% | — |
| 32k | 96.8% | — |
| **16k** | **92.8%** | **2.07** |

原文：*"The truncated head costs almost no acceptance ... the drafter rarely
guessed the rare tokens anyway."* multilingual 只有 86% 覆盖，tokens/round 仍从
1.59 只掉到 1.58。

**含义**：词表裁剪的收益全部来自 draft step 变快，**不来自接受率**。
在 GPU 上 head 不是带宽瓶颈（我们 27B 的瓶颈是权重流式读取），所以：
- 扩大 24k → 无收益，别做。
- 缩小到 16k → 在我们平台上也不会有速度收益（head 本来就不慢），别做。
- **24k 这个值是对的，不动。**

### 4.2 tree 草稿 —— 他们做了但没报收益

他们完整实现了 EAGLE-2 风格 top-k 树（`policies.py` 的 `StaticPolicy` /
`Plan.children_for` / `select_verify`，含 T>0 的递归拒绝采样和无偏性负控测试），
但**全部 benchmark 都是 chain，没有一条 tree 的加速数据**。

对照 4.1 的结论（acceptance 是天花板）+ 我们的 γ 不敏感（27B 2 与 4 同速），
tree 在我们这里的预期收益：**验证宽度换接受率，但我们已经不缺验证宽度
（DRAFT_N=4 = 5 列，MMQ 高效区），收益面很窄。** 不值得开。

---

## 5. 不可借鉴：两条

1. **重训 drafter**：他们的 +28% 来自 9× 数据重训（80k 条 target 自写答案，
   H100 上 6400 步，峰值 46.7 GB）。我们有现成的 MTP 头，再训一个 drafter 的
   成本远超收益；而且 27B 的 MTP acceptance 已经 0.98（next-token）。
2. **single-sync chain**：那是统一内存架构下的 host 同步优化。CUDA 上是异步
   launch，本来就没有每轮一次同步的问题。

---

## 6. 一个方法论值得学

`journal.md` 里每一项改动都带独立测量的贡献值，失败的和成功的都记（包括
"bf16 下 40 条只有 12–18 条 greedy 一致，指向 target 的 batched-vs-single 数值差异"
这种悬案）。我们自己的 `docs/negative-results.md` 已经是这个格式，
这次对账也印证了它的价值——**他们替我们测掉了两个方向**。

---

## 7. 结论

- 没有代码可搬。
- **一处可做**：9B 上的深度控制器，用实测延迟 + α 窗口估计替换爬山阈值。
  27B 上不划算（γ 不敏感）。
- **两处已经不用做**：扩大 shortlist、tree 草稿。
- 我们现在的组合（MTP + ngram + 24k shortlist）在方法论上没有落伍，
  差别只在它训了模型、我们用了查表。
