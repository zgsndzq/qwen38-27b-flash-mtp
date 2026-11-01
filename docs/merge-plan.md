# 合并上游 5628c733 的计划

## 一、起点

| | |
|---|---|
| 我们同步到的上游点 | `.upstream-base` = `c96baf4`（2026-10-26） |
| 要合到的上游 | `prism/prism` = `5628c733`（2026-10-30） |
| 我们的树 | `code/`（上游根 ↔ `code/`，靠 `tools/sync-upstream.sh` 的 `map()`） |

**关键事实：上一次 sync 没有真的同步。**

`bdb67e241` 的 message 写着 `sync upstream prism 909f0ca..c96baf4`，但 `--stat` 只有 **17 个文件**：

```
- take fwht/norm/quantize/ssm-conv/mmvf/mmvq fusion updates
- reject c96baf4 (FA head-72 pad, vision-only; vision disabled)
- skip tests/test-backend-ops.cpp permanently (.upstream-skip)
```

那是**挑着摘的**，不是一个跨 1303 个提交的合并。所以 `code/` 里 sycl、tests、部分 src 还停在 `1d0970e5b`（10-19 导入源码那一刻），和上游差了 11 天。这就是为什么现在 diff 会出现 `ggml-sycl.cpp -1883 行`这种数字 —— 不是我们删的，是我们一直没拿。

## 二、差距

`python tools/sync_upstream.py --base c96baf4 --theirs prism/prism`：

```
upstream only changed : 1812     上游改了、我们没动   -> 整取上游
upstream new files    :    1     上游新增             -> 直接加
we only changed       : 2129     我们改了、上游没动   -> 原样保留
both changed          :   96     双方都动             -> 三方合并
conflicts             :   49     合并后仍有冲突       -> 手工
```

49 个里 `tests/test-backend-ops.cpp` 被 `.upstream-skip` 永久跳过，**实际要处理 48 个**。

## 三、48 个冲突怎么处理

按“我们改了多少”分三类。判据：`git diff c96baf4:<f> HEAD:code/<f>`。

### A 类 · 我们只加了锚点（11 个文件，每个 ≤12 行）

上游整文件拿走，然后把我们的几行重新打上去。

| 文件 | 我们 | 上游 | 我们的锚点是什么 |
|---|---|---|---|
| `ggml/include/ggml-rpc.h` | +2/-2 | +4/-4 | q3_0 的类型枚举登记 |
| `ggml/src/ggml-cuda/mmq.cuh` | +1/-0 | +346/-280 | include `ggml-cuda-dbg.h` |
| `ggml/src/ggml-sycl/common.hpp` | +0/-9 | +190/-51 | 我们删过的 sycl 片段，**取上游即可** |
| `src/CMakeLists.txt` | +1/-0 | +45/-36 | 加一个源文件 |
| `src/llama-arch.cpp` | +7/-4 | +688/-434 | q3_0 归档项 |
| `src/llama-arch.h` | +4/-2 | +193/-36 | 同上，枚举 |
| `src/llama-ext.h` | +0/-10 | +29/-0 | **取上游** |
| `src/llama-impl.h` | +0/-6 | +6/-1 | **取上游** |
| `src/llama-model-loader.cpp` | +0/-7 | +203/-47 | **取上游** |
| `src/llama-model.h` | +4/-2 | +198/-15 | q3_0 字段 |
| `src/llama-quant.cpp` | +0/-6 | +210/-90 | **取上游** |

标“取上游”的 5 个：我们的 diff 是净删除，等于我们落后，没有私有改动要保。

### B 类 · 我们的实现（27 个文件，必须逐 hunk 看）

| 文件 | 我们 | 上游 | 里面是什么 |
|---|---|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | +1174/-224 | +1352/-271 | tier KV 全套 + q3_0 拷贝白名单 + shared pool |
| `ggml/src/ggml-cuda/fattn.cu` | +1084/-112 | +196/-11 | q3_0 白名单 + `FATTN_VEC_CASES_ALL_D` |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | +265/-50 | +346/-197 | q3_0 的 mma 实例路径 |
| `ggml/src/ggml-cuda/fattn-common.cuh` | +192/-21 | +53/-13 | `vec_dot_fattn_vec_KQ_q3_0` + V 反量化 |
| `ggml/src/ggml-cuda/common.cuh` | +30/-6 | +95/-49 | `q3_0_unpack3` + type_traits |
| `ggml/src/ggml-cuda/mmvq.cu` | +30/-56 | +383/-40 | PTQ1_0 的 MMVQ 列上限 |
| `ggml/src/ggml-cuda/mmq.cu` | +18/-1 | +228/-65 | `GGML_CUDA_MMQ_LOG` 包装（今天加的） |
| `ggml/src/ggml.c` | +49/-48 | +223/-27 | q3_0 的 type_traits + `quantize_chunk` |
| `ggml/src/ggml-backend.cpp` | +11/-83 | +381/-197 | 需逐 hunk 确认，疑似落后 |
| `common/arg.cpp` | +216/-4 | +243/-99 | `-ctk/-ctv` 白名单 + `--features-file` |
| `common/speculative.cpp` | +463/-349 | +764/-462 | MTP 草稿 + `SPC_SRV_TIMING` |
| `common/sampling.cpp` | +146/-47 | +176/-4 | `sampler-fast-topk` |
| `common/common.h` | +74/-18 | +183/-35 | 参数结构 |
| `common/speculative.h` | +18/-0 | +20/-2 | 纯新增，我们的保留 |
| `src/llama-context.cpp` | +214/-264 | +597/-159 | 上下文 + 遥测 |
| `src/llama-graph.cpp` | +186/-69 | +769/-313 | GDN/fusion 相关 |
| `src/llama-graph.h` | +22/-24 | +145/-57 | 同上 |
| `src/llama-kv-cache.cpp` | +168/-86 | +418/-115 | tier KV 的 host 侧 |
| `src/llama-kv-cache.h` | +2/-19 | +41/-4 | 疑似落后 |
| `src/llama-memory-recurrent.cpp` | +173/-157 | +143/-14 | `LLAMA_RS_REPLAY` |
| `src/llama-context.h` | +25/-38 | +60/-10 | 同上 |
| `tools/server/server-context.cpp` | +448/-18 | +1971/-331 | 槽位续接 + `ARC_SLOT_RESUME` / `LLAMA_ACC_PER_POS` |
| `src/models/qwen35.cpp` | +9/-519 | +47/-41 | **净删除 519 行，先确认是不是落后** |

### C 类 · 整取上游，不手工（10 个文件）

| 文件 | 我们 | 上游 | 结论 |
|---|---|---|---|
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | +28/-1883 | +1773/-321 | 取上游 |
| `ggml/src/ggml-sycl/mmvq.cpp` | +25/-1139 | +976/-88 | 取上游 |
| `ggml/src/ggml-sycl/vecdotq.hpp` | +0/-156 | +434/-149 | 取上游 |
| `ggml/src/ggml-sycl/fwht.cpp` | +0/-198 | +42/-30 | 取上游 |
| `ggml/src/ggml-sycl/gemm.hpp` | +0/-48 | +12/-4 | 取上游 |
| `ggml/src/ggml-sycl/mmvq.hpp` | +0/-14 | +44/-0 | 取上游 |
| `ggml/src/ggml-sycl/convert.cpp` | +4/-114 | +123/-186 | 取上游 |
| `ggml/src/ggml-sycl/fusion.cpp` | +2/-15 | +193/-10 | 取上游 |
| `tests/test-chat.cpp` | +0/-199 | +1315/-127 | 取上游 |
| `ggml/src/ggml-sycl/fattn-mkl.cpp` | +48/-34 | +30/-25 | 需看一眼那 48 行是不是我们的 |

我们不用 SYCL，这一批纯粹是为了让树的基准对齐，合进去不影响 CUDA 路径。

## 四、执行步骤

**0. 先固化当前状态**

```
git status 干净
tools/check_features.py            # 33 switches, 0 without a reader
```

**1. 干跑，确认分类**

```
bash tools/sync-upstream.sh prism/prism --dry-run
```

它逐文件判 `added / took theirs / merged / conflict`，报告落在 `logs/sync-*.md`。这一步慢，在 Windows 上单次十几分钟，因为每个文件要跑几次 `git show`。

**2. 真跑**

```
bash tools/sync-upstream.sh prism/prism
```

- 改之前每个文件都 cp 到 `logs/sync-backup-*/`
- 冲突文件会被写入 `<<<<<<<` 标记

**3. 按 A/B/C 处理冲突**

- C 类：`git checkout` 之后 `git show prism/prism:<path> > code/<path>` 直接覆盖
- A 类：取上游，然后把锚点补回去（`type_traits` 一行、`switch` 一个 case、CMake 一行）
- B 类：逐个看 `<<<<<<<` 块，原则是**上游的新逻辑照收，我们的特性挂在它后面**

**4. 编译**

```
cmd //c code\_build_inc.bat
grep BUILD_EXIT build/logs/_build_inc.log
```

改过 `.cuh` 或 cmake 文件会触发全量，10–15 分钟。

**5. 三道自检**

```
tools/check_features.py                      # 33 个开关都得有读点
python tools/verify-sync.py c96baf4 prism/prism   # 上游行有没有漏进来
python tools/sync_upstream.py --base prism/prism --theirs prism/prism   # 应为 0 差异
```

**6. 服务冒烟**

```
llmserver.bat → /health → 一次 chat 请求 → 看 [slot] / [speed] 行
```

**7. 更新记录**

```
echo 5628c733 > .upstream-base
```

## 五、风险点

**上游改了 q3_0 依赖的结构**
`fattn-mma-f16.cuh` 上游 +346/-197，我们的 31 个 q3_0 实例全靠它。合完先编这一个文件，通过了再编全部。

**`GGML_OP_COUNT` 会变**
上次 `9d37ac9fb` 编不过就是栽在这：`ggml.c` 用了 `GGML_OP_GATED_DELTA_NET_REPLAY`，`ggml.h` 里没有这个枚举。上游每次加 op 都会改 `GGML_OP_COUNT`，合完必须确认 `ggml.h` 的枚举和我们代码里用的一致。

**`verify-sync.py` 的净删除**
C 类那批（sycl 等）现在在 dev 上是“少 1883 行”的状态，合完之后这部分应该归零。如果合完还有残留，说明上游那部分又前进了。

**`.upstream-skip`**
`tests/test-backend-ops.cpp` 是明确永久跳过的（上游 2867 行改动，我们不跟）。这个文件不参与冲突统计，别误判成漏合。
