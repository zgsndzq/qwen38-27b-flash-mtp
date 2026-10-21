# q3_0 KV — 实施计划（暂缓，待做）

> 状态：**设计上ready，代码未动手。** 2026-10-21 查清全部改动点后归档，等专机做。
> 目标：在 8GB 卡上把 KV 每cell 从 18,432 B 降到 14,336 B（**-22%**），
> 同样显存预算下cell 从 65,536 提到约 84,000，让更多上下文留在 VRAM 而不是掉进 host tail吃 PCIe。

---

## 0. 为什么最终必须做这件事（决策链）

1. 要 128K 上下文 → 显存不够（权重 5.79 GB 占死大部分，8GB 卡余量有限）。
2. 显存不够 → 只能把更多 KV cell 放进 VRAM（`kv-vram-cells` 现在 65,536）。
3. 提 cell 上限受权重挤压 → 唯一剩下的杠杆是**让每个 cell 变小**。
4. 4 bit 以下本树没有可用的对称块类型 → **只能新增 q3_0**。

实测锚点：64K 是分水岭，64K 以上 prefill 深度处 KV 开始走 PCIe（host tail）。
多出来的约 18K cell 直接决定有多少上下文能留在显存。

---

## 1. 前置结论：q3_0 在本树完全不存在（2026-10-21 核查）

全树搜 `GGML_TYPE_Q3_0` **零命中**。逐环节确认：

| 环节 | 文件 | 现状 |
|---|---|---|
| 类型枚举 | `ggml/include/ggml.h:389` | 无 Q3_0。私有槽位 **43..141 空着**（`PQ2_0=142`、`PTQ1_0=143` 占高位），可用 **43** |
| block 结构 | `ggml/src/ggml-common.h` | 无 `block_q3_0` |
| CPU 量化/反量化 | `ggml/src/ggml.c`、`ggml-quants.c` | 无 |
| CUDA 转换 | `ggml/src/ggml-cuda/convert.cu` | 无分支（6 处switch 位置见下） |
| CUDA type_traits | `ggml/src/ggml-cuda/common.cuh` | 无（Q4_0 在 1107 行） |
| FA vec_dot | `ggml/src/ggml-cuda/fattn-common.cuh:185` | `vec_dot_fattn_vec_KQ_q4_0` 存在，q3_0 无 |
| FA dequant V | `ggml/src/ggml-cuda/fattn-common.cuh:445` | `dequantize_V_q4_0` 存在，q3_0 无 |
| FA 分发 | `fattn-common.cuh:658` / `:680` | `get_vec_dot_KQ()` / `get_dequantize_V()` 只分发 7 种 |
| FA 白名单 | `ggml/src/ggml-cuda/fattn.cu:402` | `ggml_cuda_fattn_kv_type_supported()` 同白名单 |
| CLI 白名单 | `common/arg.cpp:305` | `kv_cache_types` 9 种，不含 Q3_0 |

**这是新增一个量化类型，不是移植一条已有线路。** Q4_0 全树约 40 处引用点可作施工参照。

---

## 2. 不能复用 `block_q3_K`（省事路线是死的）

树里已有 `block_q3_K`（3.4375 bpw），数字上接近，看着能省事。**但不能用**：

```
block_q3_K:  QK_K=256 的 super-block
             hmask[QK_K/8] + qs[QK_K/4] + scales[12]（6-bit 量化）+ ggml_half d
```

两个硬伤：
1. **scale 本身还要6-bit 量化**（`scales[12]` + `d` 两级），反量化路径与 q4_0 完全不同。
2. KV 需要的是**简单对称块**（每 32 元素独立 scale），FA 内核里没有 q3_K 的 vec_dot。

所以要**照 `block_q4_0` 新造**，而不是改编 q3_K：

```c
#define QK3_0 32
typedef struct {
    ggml_half d;            // delta
    uint8_t  qs[QK3_0*3/8];// 32 个 3-bit 值 -> 12 字节
} block_q3_0;              // 2 + 12 = 14 字节/ 32 元素 = 3.5 bpw
static_assert(sizeof(block_q3_0) == sizeof(ggml_half) + QK3_0*3/8, "...");
```

值域映射（对称、无zero point，与 q4_0 同构）：`q ∈ [-4, 3]`，`x ≈ d * q`。

---

## 3. 收益与代价（必须先算清楚再动手）

**收益**：
- 每 cell 18,432 B → 14,336 B（**-22%**）
- 1.18 GB 预算下 cell 65,536 →约 84,000
- 64K 以上不再立刻进 host tail，prefill 与 decode 少走 PCIe

**代价**：
- 4.5 bpw → 3.5 bpw，**K 的精度下降**。长上下文"大海捞针"最敏感（见下）。

---

## 4. 精度判断（避免把事情做错方向）

**先纠正一个常见误解**：KV 量化误差**不随上下文变长而累积**。每个 token 的 K/V 是
独立定点误差，没有跨 token 的递推路径，128K 不会比 8K "偏得更远"。

真正的风险是**检索任务放大 K 的误差**：上下文越长，"大海捞针"越依赖精确点积匹配。
而 K 与 V 的敏感度差别很大：
- **K** 进 softmax 前的 logits，误差**直接改变注意力分布** → 敏感
- **V** 参与加权求和，误差**被平均掉** → 宽容

**结论：`-ctk` 与 `-ctv` 不必取同一档。** 最省的组合是 **K 保 q4_0、V 降到 q3_0**：
容量省一半左右，精度损失小得多。引擎天然支持分档，两个参数本来就是独立的。

`docs/kv-mean-center.md` 里的 `--kv-mean-center` 只作用于 **q4_0 的 K**，
若最终给K 上q3_0，这个补偿手段失效，需另想办法（可考虑给 q3_0 也做mean-center）。

**动手前建议先测**：q4_0 vs q5_0 的 logit KLD（对 f16 baseline）。
若q4_0→q5_0 的 KLD 已经很小，说明瓶颈不在 KV 精度，q3_0 这套实现不值得做。

---

## 5. 施工顺序（每步可单独验证）

| 步 | 内容 | 文件 | 验证方式 | 风险 |
|---|---|---|---|---|
| 1 | 枚举 + block + CPU 量化/反量化 + type_traits | `ggml.h:435`、`ggml-common.h`、`ggml.c`、`ggml-quants.c/h` | `test-backend-ops` 单独验证正确性 | **低**（纯 CPU） |
| 2 | CUDA 双向转换 | `convert.cu`（6 处 switch：553/618/686/749/778/807）、`common.cuh:1107` | 与 CPU 实现逐元素比对 | 低 |
| 3 | **FA 三套内核** | `fattn-common.cuh:185/445/658/680`、`fattn-mma-f16.cuh`（Q4_0 有 12 处）、`fattn-vec.cuh`（4 处）、`fattn-tile.cu` | **必须确认三套都命中** | **高** |
| 4 | 白名单 + 分档开关 | `fattn.cu:402`、`common/arg.cpp:305`、`features.ini` | 实测 cell 数与速度 | 低 |

### 第 3 步是全部风险所在

**vec / tile / mma-f16 三套内核都要有模板实例化。漏掉一套不会报错，
会静默回退到 f16 中转** —— 那样22% 显存省下来了但速度全丢，等于白做。

所以第 3 步做完必须**实测确认三套都走到q3_0 分支**，而不是靠推断。
可用 `GGML_CUDA_FA_LOG=1` 看 `mmfa.log` 里实际的 kernel 路由与 type_k，
确认没有出现 f16 中转。

---

## 6. 建议的落地形态：分档，而不是全局替换

不要把q3_0 设成全局默认。建议：

```
短上下文（< 64K）   -ctk q4_0  -ctv q4_0    精度优先
长上下文（> 64K）   -ctk q4_0  -ctv q3_0    K 保精度、V 换容量（推荐）
                    或        -ctk q3_0 -ctv q3_0   最省，精度损失最大
```

按第 4 节的敏感度分析，**K 保 q4_0、V 降 q3_0 是性价比最高的一档**，
建议作为第一个要测的组合。

配套可以在 `features.ini` 里做成按档切换（`-ctk`/`-ctv` 本来就是 features 能驱动的参数：
任何非 env 形状的键都会变成 `--key value`），例如拆成 `[qwen]`（默认高精度）
与 `[qwen-long]`（长上下文低容量）两个 section，由不同 bat 选择。

---

## 7. 参照实现

`bonsai-ada-surgery` 的 **patch 0056** 就是这件事：8GB 预设从 bundle-20261015b 起
把 KV 从 q4_0 换成 q3_0，为的是在同样显存里塞进 131k 而不是更少。
可作为对照读，但注意**我们树的 FA 内核结构与他们不完全一致**，不能直接 cherry-pick。

他们的其他 prefill 相关手段（多数我们已有）：
- packed 1-bit KQ mask（`kq-mask-packed`，我们因 128K 下展开开销更大而关闭）
- prefill int8 activation plane（+18% prefill，KL 0.00028）
- level-decode chunk 按 buffer 占用切（`GGML_MIRAI_LEVELS_MIB`）

---

## 8. 相关文件速查

- 计划归档时：`src/docs/negative-results.md` 第 6 节（q3_0 缺失核查）
- 慢路径阈值（与本计划独立但同样影响 prefill）：`features.ini` 的 `GGML_CUDA_FA_CHUNK`