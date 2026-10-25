# 我们相对上游 PrismML-Eng/llama.cpp 的私有部分

定位：**这是 `prism` 分支的分支**。上游的改动照单全收，我们的改动单独记账。
每次同步上游时按这份清单核对，少一项就是能力丢了。

同步上游用 `prism` remote（已配好）：

```bash
git fetch --depth=1 prism prism          # 只拿最新一个提交,够 diff
git diff --ignore-cr-at-eol prism/prism:ggml HEAD:code/ggml     # 看差异
```

---

## 一、私有实现（上游没有，合并时必须保留）

### 1. q3_0 — 3-bit K/V 缓存类型

14 B/块（2 B f16 scale + 12 B 顺序 3-bit 位流）对 q4_0 的 18 B，省 22%。
只是缓存类型，`-ctk q3_0` 切换，量化在线做。

| 文件 | 加了什么 |
|---|---|
| `ggml/include/ggml.h` | `GGML_TYPE_Q3_0 = 43` |
| `ggml/src/ggml-common.h` | `block_q3_0`、`QK3_0`、`QR3_0`、`QI3_0` |
| `ggml/src/ggml-quants.c/h` | `quantize_row_q3_0_ref` / `dequantize_row_q3_0` |
| `ggml/src/ggml.c` | type_traits + `quantize_chunk` case |
| `ggml/src/ggml-cpu/quants.c/h`、`ggml-cpu.c` | CPU 侧注册 |
| `ggml/src/ggml-cuda/common.cuh` | `q3_0_unpack3` + type_traits |
| `ggml/src/ggml-cuda/cpy-utils.cuh`、`cpy.cu` | 双向编解码 + dispatch（**KV 写入路径**） |
| `ggml/src/ggml-cuda/convert.cu`、`dequantize.cuh` | 反量化，6 个 dispatch |
| `ggml/src/ggml-cuda/set-rows.cu` | set_rows 量化写 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | F32↔Q3_0 拷贝白名单 |
| `ggml/src/ggml-cuda/fattn-common.cuh` | `vec_dot_fattn_vec_KQ_q3_0` + `dequantize_V_q3_0` + 两个分发分支 |
| `ggml/src/ggml-cuda/fattn.cu` | 白名单 + `FATTN_VEC_CASES_ALL_D(Q3_0,Q3_0)` |
| `common/arg.cpp` | `-ctk/-ctv` 白名单 |
| `ggml/cmake/common.cmake` | `FA_TYPES` 加 q3_0 |
| `template-instances/generate_cu_files.py` | `TYPES_KV` 加 q3_0 |
| `template-instances/*q3_0*.cu` | **31 个**（16 mma + 15 vec），脚本生成 |

缺任意一环不会报错，只会**静默回退到 f16 中转**——显存省了、速度全丢。

### 2. 分层 KV（`--kv-vram-cells`）的完整实现

上游 #319 是另一套更简单的版本。**我们的更细，不要被它覆盖**：

| 机制 | 位置 |
|---|---|
| `cuMemCreate` HOST location + 一个 VMM range | `ggml-cuda.cu`（`ggml_backend_cuda_tier_alloc`） |
| staging alias（tail 拷进 VRAM staging，op 读 alias） | `ggml_cuda_tier_stage_pages` |
| `set_rows` 重定向到 staging（写不别名读） | `ggml_cuda_tier_unalias` |
| 跨 op 异步预取（copy engine 搬下一个 op 的 host 行） | `ggml_cuda_tier_graph_begin` |
| 四个子开关 | `features.ini` 的 `GGML_CUDA_KV_TIER_*` |
| tier 大小 | `llmserver.bat` 的 `CELLS` ← `CTK`/`CTV` |

上游 #319 的效果描述只有"prefill reads the tail once per op"（146→280 tok/s at 180k），
我们已经有这个，还有他们没提的 redirect 和 prefetch。

### 3. 其它私有改动

| 文件 | 改动 |
|---|---|
| `llmchat/chat.py` + `README.md` | 渲染帧三件套（system/think/effort）、思考块回填、`--no-sys` |
| `tools/server/server-context.cpp` | **槽位续接修复**：`mem_pos_max + 1 >= n_past` 取代 `pos_min == 0`；`acc per pos` 上 INFO |
| `ggml/src/ggml-cuda/mmvq-ptq1_0.cuh` | `PTQ1_0_PT_ROWS_68`（默认 1，行为不变） |
| `llmserver.bat` | `CELLS` 由 `CTK`/`CTV` 推导、`REBUDGET=2048` |
| `features.ini` | 开关集合（数字全部移到 bat） |

---

## 二、已从上游同步的（不用再挑）

| 上游 | 内容 | 落点 |
|---|---|---|
| `4c3c02b9` #308 | FA 8 列 config `nbatch_fa 64 → 32` | `fattn-mma-f16.cuh` |
| `b8639015` #378 | graph key 混入首节点 `ne[]` + `n_nodes` | `ggml-cuda.cu` |
| `23f6463d` #380 | thin F32 走 mmvf 到 8 列（+6% MTP） | `mmvf.cu` |

已在代码里注明 `Upstream PrismML-Eng/llama.cpp <sha>, ported <date>`，搜这个前缀就能找到全部。

---

## 三、同步上游的规矩

1. **先看差异分类**，不要直接 merge：
   ```bash
   git diff --ignore-cr-at-eol --numstat prism/prism:ggml HEAD:code/ggml
   ```
   新增为 0 的文件 = 纯上游新增，可以整文件覆盖；
   删除为 0 的 = 我们独有，不要动；
   两边都有数 = **冲突区，必须逐 hunk 看**。

2. **冲突区只有 3 个**（2026-10-25 盘点）：
   `ggml-cuda.cu`、`fattn-mma-f16.cuh`、`mmvq-ptq1_0.cuh`。
   前两个的冲突来自 q3_0/tier KV，第三个来自 `PT_ROWS_68`。

3. **不要为了"差异归零"合入我们用不到的代码**：PQ2_0 的 Hopper 内核
   （`mmq-pq2-j128.cuh`）、`gated_delta_net_l8.cuh`（上游自己 0.99–1.02×）、
   `norm.cu` 的 Hadamard 融合（27B neutral）。这些留着只会扩大维护面。

4. **合并后跑一遍 `features.ini` 自检**：每个开关都得有代码在读。
   开关还在、行为没了 = 对应实现被覆盖了。这是最省事的回归。

---

## 四、当前差异（2026-10-25 11:10）

`ggml-cuda` 忽略 EOL 后：

| 类别 | 文件数 | 行数 |
|---|---|---|
| 上游有、我们没有（可整文件拿） | 11 | 1773 |
| 我们有、上游没有（q3_0 相关） | 7 | 141 |
| 双方都改（冲突区） | 16 | — |

11 个纯上游文件里，值得拿的只有 `top-k-select.cuh`（296）+ `top-k.cu`（19）和
`cpy.cuh`（12）；`mmq-pq2-j128.cuh`（921）是 9B 的 PQ2_0 prefill，为 9B 线可以拿；
剩下 500 多行是上面第 3 条说的"故意不要"。
