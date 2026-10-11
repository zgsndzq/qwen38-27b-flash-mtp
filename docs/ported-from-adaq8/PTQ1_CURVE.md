# PTQ1_0 列数曲线 + 一轮解码的真实成本拆解

> 2026-09-28，`adaq8` 项目。全部数字本机实测（RTX 4060 Laptop，256 GB/s，8188 MiB）。
> 工具 `tools/speed_probe.py`；原始数据 `bench/arms_{ptq1,opt,optv,draft,bat,graph}.json`。
> 提示词固定 `skill.txt`，每臂 2~3 次取中位，同卡同时只有一个 server。

---

## 0. 一句话结论

**「带宽到头」只对了 55%，而且那 55% 是死的、剩下 45% 是活的。**

```
一轮 54.0 ms（depth 3，ctx 3.5k，18.5 rounds/s，3.28 token/轮 = 60.7 t/s）
  = 33.6 ms  权重搬运（5.93 GB @ 176 GB/s）        <- 真到顶，别动
  + 3 × 7.0 ms  起草步                             <- 带宽账只值 2.1 ms，3.3× 空闲
```

⇒ 可撬动的是**后 38%**，而且它是**每步固定开销**问题，不是搬运问题。

---

## 1. 新锚点：三个干净的实测点

| 配置 | 接受率 | tg | tokens/轮 | **rounds/s** | **ms/轮** |
|---|---:|---:|---:|---:|---:|
| `--spec-type none` | — | 29.74（29.92/29.66/29.74） | 1.00 | 29.8 | **33.6** |
| depth 1 | **0.910**（.910/.889/.936） | 48.02（48.02/47.82/49.17） | 1.91 | 25.1 | **39.8** |
| depth 3 | 0.760 | 60.69（56.61/65.79/60.69） | 3.28 | 18.5 | **54.0** |

**两个独立算式给出同一个起草步成本：**
- depth 1 − 无推测 = 39.8 − 33.6 = **6.2 ms**
- (depth 3 − depth 1) / 2 = (54.0 − 39.8) / 2 = **7.1 ms**

**一步起草该搬的字节**（`gguf_bytes.py` 实测）：MTP 层 `blk.64` = **266 MB** + `output.weight` 全表 248320 词 = **278 MB** = **544 MB**
⇒ 按 256 GB/s 值 **2.1 ms**。**实测 7.0 ms = 3.3× 低效。**

⚠️ 注意 `output.weight` 是**全量 248k 词表**。我们自己的树里 `--spec-draft-vocab-map`（24k 热词短表，265 MB → 26 MB）**没有移植进 ada-q8**，所以每一步起草都在读整张表。

---

## 2. 验证批宽（depth）曲线 —— 这条轴从来没测过

`--spec-draft-n-max n` ⇒ 验证批 = `n+1` 列。

| 臂 | 列数 | 路径 | 中位 tg | 接受率 | tokens/轮 | **rounds/s** | ms/轮 | 线性模型 33.6+7d | 超支 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **D3** | 4 | PT | **60.69** | 0.760 | 3.28 | **18.5** | 54.0 | 54.6 | −0.6 |
| D4PT | 5 | PT | 55.83 | 0.765 | 4.06 | 13.8 | 72.5 | 61.6 | **+10.9** |
| D5PT | 6 | PT | 55.69 | 0.709 | 4.55 | 12.2 | 82.0 | 68.6 | **+13.4** |
| D6PT | 7 | PT | 44.37 | 0.569 | 4.41 | 10.1 | 99.0 | 75.6 | **+23.4** |
| D7PT | 8 | PT | 27.03 | 0.506 | 4.54 | 5.95 | 168.0 | 82.6 | **+85.4** |
| D7MQ | 8 | MMQ | 32.37 | 0.579 | 5.05 | 6.41 | 156.0 | — | — |

**判决：depth 3 最优。** 但「为什么」有两层，第二层才是真因（见 §5）。

### 2.1 第一层：`GGML_CUDA_PTQ1_MMVQ_MAX` 默认 = 4

`ada-q8/ggml/src/ggml-cuda/mmvq.cu:299`：

```cpp
// the tile path overtakes it at 5+ columns on Ada (RTX 4070, Bonsai 2 27B:
// mat-vec 155 t/s vs MMQ 244 t/s at n=8; at n=5 172 vs 162).
static const int max_cols = [] {
    const char * e = getenv("GGML_CUDA_PTQ1_MMVQ_MAX");
    return e ? atoi(e) : (ggml_cuda_batch_invariant() ? MMVQ_MAX_BATCH_SIZE : 4);
}();
```

- 默认 **4** ⇒ 5 列（depth 4）就被踹出 PT 内核、走 MMQ。
- 设 `=8` 后 depth 4 走 PT：**55.83** vs MMQ 的 45.08（**+24%**）。
- 8 列时 MMQ(32.37) > PT(27.03) —— **他们注释里那句被本机数据独立验证**，所以只能提到 5，不能提到 8。
- 已落到 bat：`set "GGML_CUDA_PTQ1_MMVQ_MAX=5"`（≤4 列时代码根本不读这个值 ⇒ depth 3 逐位不变）。

---

## 3. 逐节点 GPU 剖面（他们 HEAD commit 自带的探针）

`GGML_CUDA_OP_TIMING=1` + `GGML_CUDA_DISABLE_GRAPHS=1`，**并且必须 `-lv 4`**
（否则 ggml 的 INFO 行根本不进日志 —— 生产 bat 日志里 `load_tensors` 出现 **0** 次）。
`timing = op_timing && !use_cuda_graph`，所以**开着图时这个探针完全空转**。

纯解码段，每张图：

| 节点 | depth 3 | depth 6 |
|---|---:|---:|
| `MUL_MAT ptq1_0 5120x17408 cols=N`（FFN up/gate） | 3.463 ms | 3.464 ms |
| `MUL_MAT ptq1_0 17408x5120 cols=N`（FFN down） | 1.798 | 1.694 |
| `MUL_MAT ptq1_0 5120x248320 cols=1`（**起草步的 LM head**） | 0.843 | **1.800** |
| `CPY f32 cols=1` | 1.077 | 0.680 |
| 合计 / 图 | **17.72** | **16.39** |

### 3.1 三条硬结论

1. **`5120x17408` 在 4 列与 7 列下逐位相同（3.463 / 3.464 ms）**
   ⇒ PT 内核**确实列数无关**（权重解一次、所有列共享）。
   单实例 3.463/32 = **0.108 ms**，19.5 MB ÷ 0.108 ms = **180 GB/s = 70%**。矩阵乘没毛病。
2. **所以验证批宽不是瓶颈 —— 涨的是起草步**（LM head 那一行 0.843 → 1.800 正是起草步数 3 → 6）。
3. 每图总时 17.72 vs 16.39 几乎不变，而 rounds/s 从 18.5 掉到 10.1
   ⇒ **差别在「每轮几张图」，不在「每张图多慢」。**

---

## 4. 被证伪 / 被纠正的四条

| 说法 | 判决 | 证据 |
|---|---|---|
| `GGML_CUDA_GRAPH_OPT=1` 值 1.5× | **本树不成立**，疑似负收益 | D3G=64.18 vs D3=67.52（同 npredict 1024）。`ggml-cuda.cu:5239` 读这个 env，它是**多流并发**（QKV fork/join）开关，默认关。**bat 里那行是 inherited cargo**，待 3 轮复核后删。 |
| 「depth 4 悬崖」是内核列数上限 | **常数记错了** | 不是硬编码，是 `GGML_CUDA_PTQ1_MMVQ_MAX` 默认 4。 |
| `GGML_CUDA_DISABLE_GRAPHS` 不存在 | **错，存在** | `common.cuh:1393`。OPT 臂真的关了图。 |
| `GGML_CUDA_OP_TIMING=1` 会拖慢 25% | **那不是它** | 它的 25% 掉速来自逐节点 event 开销，不是配置差异（D3 同 npredict = 67.52 vs OPT3 = 45.55）。 |

---

## 5. 第二层：深度上限的真因是**显存**，不是带宽

`--spec-draft-n-max 7` 时 prefill 从 **497.7 t/s 崩到 70.3 t/s（7.1×）**：

```
D3   run1: prompt eval =  5991 ms / 2982 tokens (497.74 t/s)
D7PT run1: prompt eval = 42420 ms / 2982 tokens ( 70.30 t/s)
```

算术（RS 环每级 150 MiB 由 `docs/SPEEDUP-BUDGET.md` §7.1 实测）：

| 项 | 字节 |
|---|---|
| 主干权重 | 5.65 GB |
| MTP 层 `blk.64` | 254 MB |
| KV VRAM 头（`--kv-vram-cells 24000` × 34.0 KB/位置） | 835 MB |
| compute buffer | ~180 MB |
| **GDN 递归状态快照环 = 150 MiB × (1 + DRAFT)** | D3 → 600 MB / **D7 → 1200 MB** |
| **D3 合计** | **7.52 GB** ✅ |
| **D7 合计** | **8.12 GB** ❌ > 8.00 GB |

**超支量与 RS 环同步增长**（§2 最后一列：d≤3 无超支，d=4 +11 ms，d=6 +23 ms，d=7 +85 ms）
⇒ 这是**典型的换页/降级曲线**，不是内核效率曲线。

**⇒ `GDN_RS × (1+DRAFT)` 是唯一与 DRAFT 成正比的显存项。** 想上 depth ≥5 必须先给它腾地方（降 TIER / 降 ctx / 换 KV 格式），**或者改掉「每级存一份全量快照」的设计**（只快照被污染层 / delta 编码）—— 那是纯算法问题。

---

## 6. 下一阶段议程（按杠杆排序）

### 6.1 零编译

| # | 动作 | 预期 | 状态 |
|---|---|---|---|
| 1 | bat 加 `GGML_CUDA_PTQ1_MMVQ_MAX=5` | 修 `DRAFT_TAIL=4` 那条路径：5 列 **+24%** | ✅ **已落**（备份 `llmserver.bat.bak_20260928-100238`） |
| 2 | 删 `GGML_CUDA_GRAPH_OPT=1` | 实测疑似 **−5%** ⇒ 删掉可能 +5% | ⏳ 需同会话 3 轮复核再动（2 轮不足以分辨 ±12% 散布） |
| 3 | 复核 `DRAFT_TAIL=4` | depth 4 每轮 +34%、只多 24% token ⇒ 浅上下文净亏；但深上下文走 host tail 时才可能赚 | ⏳ 需 24k 上下文实测 |

### 6.2 需要编译 —— 把我们的补丁移植进 ada-q8（用户指令第 3 步）

已核对：`ada-q8` 树里我们的补丁**一个都没有**
（`spec-draft-vocab-map` / `vocab-hot` / `draft-mtp-adaptive` / `-n-min-adaptive` / `LLAMA_SPEC_PQ` / `SPC_TIMING` / `SPC_SRV_TIMING` grep 全部 = 0）。

| 补丁 | 作用 | 预期 |
|---|---|---|
| `--spec-draft-vocab-map` / `--vocab-hot` | 起草步 LM head 265 MB → 26 MB | 每步 −0.93 ms（**−13%**） |
| **`SPC_TIMING` + `SPC_SRV_TIMING`** | **拆开那 7.0 ms 里剩下 ~5 ms 固定开销的唯一仪器**（输出形如 `pre/draft/dec/launch/sync/post/acc`） | 定位用 |
| `draft-mtp-adaptive` + `-n-min-adaptive` | 按位置动态深度 | 接受率↑ |

### 6.3 算法级（真正的「没有上限」）

| # | 方向 | 算术 |
|---|---|---|
| 1 | **把 depth 个起草步合成一张图 / 一个持久内核。** 现在每步一次 launch+sync+sample | 固定开销 ×6 → ×1：depth 6 时 round 46 → ~36 ms ⇒ **~122 t/s（2.0×）** |
| 2 | **GDN RS 快照环瘦身**（只快照被污染层 / delta 编码） | 解锁 depth 5-7；列数已证明不花时间 |
| 3 | **长上下文才是主战场。** 3.5k 时 KV 只占每轮搬运 **2%**；按 34.0 KB/位置，24k 占 14%、**64k 占 37%** | 现在所有结论都建立在 KV 影响最小的点上 ⇒ 下一轮必须在 24k/64k 重测整条曲线 |

### 6.4 不要再试的

- `GGML_CUDA_GRAPH_OPT=1`（0 或负收益）
- depth ≥4 的 MMQ 路径（≤3 层时 PT 快 24%）
- `--spec-draft-window` 与深度抢参数（窗口是 KV 窗，不是起草次数）

---

## 7. 模型摘要（可直接引用）

```
tg = (1 + accept × DRAFT) / round
round_ms ≈ 33.6 (5.93 GB @ 176 GB/s，上限)  +  7.0 × DRAFT  (带宽账 2.1 ms/步，3.3× 空闲)
         ↑ 前项只在 RS 环保住 8 GB 显存时成立；超了之后超支 ∝ 环宽 = 150 MiB × (1+DRAFT)
```

## 8. 复现命令

```bat
cd F:\super\adaq8
python tools\speed_probe.py --list
python tools\speed_probe.py --arm D3 --arm D4PT --arm D5PT --arm D6PT --arm D7PT --arm D7MQ --runs 3
python tools\speed_probe.py --arm PLAIN --arm D1 --runs 3 --npredict 1024
python tools\speed_probe.py --arm OPT3v --arm OPT6v --runs 1 --npredict 1024   rem -lv 4 才出剖面
python tools\speed_probe.py --bat F:/super/adaq8/llmserver.bat --label PROD --runs 2
python tools\gguf_meta.py  models\Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf block_count head_count embedding_length
python tools\patch_bat_ptq1.py    rem 已执行；写 tmp -> 校验 -> Move-Item，非原地覆盖
```

本轮向 `tools/speed_probe.py` 新增的臂：
`D3 D4PT D5PT D6PT D7PT D7MQ D3G OPT3 OPT6 OPT3v OPT6v PLAIN D1`
