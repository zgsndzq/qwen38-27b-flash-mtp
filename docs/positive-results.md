# 正面清单：实测有效、已启用的优化

与 `negative-results.md` 配对。那边记录"试过、量过、确认没用"，这边记录
**"有效、已落在配置里、别再弄丢"**。

每条包含：做法 / 实测数据 / 配置位置。没有实测数据的会标注 `（未实测）`。

**测量前提**（与负面清单第 0 节相同，不重复）：测速必须 `BENCH=1` 隔离 ngram 缓存，
或用充分预热的同一起点。短 prompt + 非流式请求会高估收益。

---

## 1. 投机解码：吞吐的主要来源

### 1.1 ngram-mod 查找草稿 — 最大单一乘数

| | 冷启动 | 热态 |
|---|---|---|
| 27B | 46–48 t/s | **177–187 t/s** |
| 9B | 112 t/s | **480 t/s** |

3.9 倍。任何别的优化都只作用在基准上，这是乘数本身。
配置：`llmserver.bat` 的 `NGMIN/NGMAX/NMATCH/LOOKUP_N` + `--spec-type ngram-mod`。
BENCH=1 时切掉它以测裸速。

### 1.2 ngram 表落盘 — 重启不冷

`spec-ngram-mod-cache = data/ngram-mod.cache`（`[qwen]`）。
表是 16 MB 固定 hash，写回时机=表变脏且距上次写盘 ≥30 秒，硬杀也留得住。
实测重启后第一轮 **47.5 → 182.8 t/s**。

注意：**每个模型一个文件**，词表不同不能共用（27B 与 9B 词表相同可共用短名单，
但 ngram 表按 vocab 索引，仍建议分开）。

### 1.3 MTP 草稿头（模型自带 nextn 层）

`--spec-type draft-mtp-adaptive`，`DRAFT_N=4`（27B，用户硬规矩）。
实测 acceptance 0.98（27B）/ 0.53–0.61 mean len 2.6–2.7（9B）。
9B 上 **n-max 2 优于 4**（112 vs 96–105 t/s），27B 上 2 与 4 同速，按 4 配。

### 1.4 24K 草稿短名单

`spec-draft-vocab-map = maps/shortlist_24k_mix.vocab.txt`（`[qwen]` 与 `[bonsai]` 共用）。
两个模型同为 248320 词表，文件可直接复用。

### 1.5 拒绝采样

`spec-draft-sampling = probabilistic`（两个 section 都有）。
按请求温度做真采样 + 拒绝采样，比 match 模式质量更稳。

---

## 2. KV 与显存：128K 能跑起来靠这些

### 2.1 分层 KV（kv-vram-cells=65536）— 128K 的前提

`[qwen]` 的 `kv-vram-cells = 65536`：前 64K cell 在显存，其余在 pinned RAM。
CTX 131072 时 host tail 约 1.18 GB RAM（18 KB/token）。
实测：设成 32768 时 64K 序列掉到 **12.8 t/s**；65536 时装下且整机
**7858/8188 MiB**。设成 =CTX 会 OOM。

### 2.2 target/draft 共享显存池

`GGML_CUDA_SHARED_POOL = 1`（common）。8 GB 卡上省约 **1.4 GB**。
仅 `--parallel 1` 下安全。

### 2.3 生产上下文配置

`llmserver.bat`：`CTX=131072`、`--parallel 1`、`-fa on`、
`-ctk q4_0 -ctv q4_0`、`-b 4096`、`-t 8`、`-ngl 99`、`VISION=0`（无 mmproj）。
9B 用 128K + q4_0 KV 全显存（2.49 GB 权重，KV 只 0.7 GB，`kv-vram-cells=0`）。

---

## 3. 注意力内核

### 3.1 FA_CHUNK=65536 — 2026-10-21 修的最大坑

详见 features.ini 长注释与负面清单 1.6 的撤回记录。要点：
C 是**慢路径触发阈值**不是显存旋钮，8192 会让所有 >8K 的 prefill 掉进
「拷贝→转 f16→折叠」。实测 12K ×5.6、16K ×7.4、24K ×8.3、32K ×12.1、41K ×14.6；
66,247 token 端到端 215 s / 307.96 t/s。

### 3.2 FA_SHORTQ_MMA=0 — 保护 MTP verify 热点

短查询（Q 3–4 行）+ 量化 V cache 时**不**走 f16 MMA 内核（否则每遍把整个
K/V 反量化到 scratch）。实测：开=某 MTP 请求重复一个 token、acceptance 掉到
0.12；关=同一请求干净出 319 字符且更快（27.1 vs 24.9 t/s）。

### 3.3 FA_SPARSE — host tail 稀疏读取

`GGML_CUDA_FA_SPARSE = 8192,64,1024`（`[qwen]`）：保留 VRAM cell + 最后 R 个
+ B 个最佳 host 页，每步 host 流量从整个 tail 降到 B 个 cell。
上游同族 4070 实测 160k decode **13.8 → 27.3 t/s**（我们的卡未复测，
近似检索，注意 needle 风险）。

### 3.4 kq-mask-packed 在 128K 保持关闭

3315-token prompt 实测：packed off → pp 430 t/s / tg 38.7；on → pp 265 / tg 40.5。
打包 mask 的同步展开循环无法与 cp_async 重叠，128K 下比一次 f16 展开还贵。
decode 侧偏好 packed 约 5%，但 prefill 亏 40%，整体关。（配置里已注释掉。）

### 3.5 PT mat-vec / MMQ 交叉点=5

`GGML_CUDA_PTQ1_MMVQ_MAX = 5`（`[qwen]`）。RTX 4090 实测：n=8 时
mat-vec 155 t/s vs MMQ 244；n=5 时 172 vs 162。**5 是拐点，别动**
（扩到 8/12、或降到 1 全是负例，见负面清单 1.4）。

### 3.6 移植自上游 / 同源项目的内核

已在本树且验证过：0053 关 host-tail graph（192k decode 22.8 → 29.4 t/s，
128k +16% / 192k +9% / 245k +8%）、FA tile map、0047 tier partial staging、
0054 GDN chunked prefill、`GGML_CUDA_FA_PREFILL_F16`、#307 MMA FA GQA>4
（本树实现比上游宽：gqa_ratio % 8/4/2 三档全开）。

---

## 4. 引擎与工具链

### 4.1 MTP 草稿微批

`LLAMA_MTP_DRAFT_UBATCH = 64`（`[qwen]`/`[bonsai]`）。

### 4.2 自适应草稿深度下限

`spec-draft-n-min-adaptive = 1`：控制器从这里起步、不低于此，
上限由启动脚本的 `--spec-draft-n-max` 管。

### 4.3 BENCH 开关 — 测速基础设施

`llmserver.bat` 的 `BENCH=1`：保留 MTP 头、去掉 ngram 查找草稿，
使冷热可测、可复现。生产 `BENCH=0` 拿 ngram 最大收益。
**没有这个开关之前的所有对比数字都不可信**（3.9 倍冷热差）。

### 4.4 埋点体系（都在 common 节）

- `LLAMA_PERF_LOG = 2` + `LLAMA_PERF_LOG_FILE = logs/perf.csv`：
  每次 decode 一行，含 ctx/kind/耗时/最大位置；level 2 加投机接受数。
  实时看：`python tools\perf_monitor.py logs\perf.csv`
- `SPC_SRV_TIMING = 1`：服务端每轮时间拆解（pre/draft/dec/round/post）
- `GGML_CUDA_FA_LOG = 1` + `GGML_CUDA_MMQ_LOG = 1`：内核路由与 tile 选择，
  写文件不打屏（`GGML_CUDA_LOG_FILE`/`GGML_CUDA_LOG_MAX`，采样写满即停）

**FA_CHUNK 那个坑就是这套埋点抓出来的**——按 max_pos 分段看 prefill 速率曲线，
平滑衰减是深度问题，断崖是开关问题。

### 4.5 features.ini 分节机制

一个文件多模型：`[common]` / `[qwen]` / `[bonsai]` / `[lfm]`，
`--features-section X` 选择。启动脚本只放通用运行参数。
环境变量形状的键导出为环境变量，其余变 `--key value`。

### 4.6 其它已启用开关

- `sampler-fast-topk = auto`（common）：拒绝采样时关、常规采样时开
- `LLAMA_CKPT_REUSE = 1`（common）：evicted checkpoint 缓冲复用
- 27B 侧 `-no-ui -rea off` 等生产项；Sage 曾用 `-rea on` + budget 4096
  （该模型已删，配置已清）

---

## 5. 跨模型通用结论（正面）

1. **ngram 是乘数**：吞吐 = 基准 × ngram 倍数。优化只能作用在基准上。
2. **草稿量化类型不必与 target 一致**：草稿 f16/q8_0、target PTQ1_0 没问题，
   只要词表、hidden 宽度、hidden 的基（Hadamard 折叠须在查表后还原）一致。
3. **4~5 列是这张卡单次调用的最优宽度**（matvec 与 MMQ 皆然），
   177 GB/s 已到带宽 82%——再加宽是物理限制，不是调优问题。
4. **同族模型配置可迁移**：qwen35 hybrid（GDN + full_attention_interval 4 +
   nextn）的 27B / 9B / Sage 之间，MTP、ngram、shortlist、features 机制全部通用；
   不通用的只有量化专属项（PTQ1_0 / PQ2_0）与分层 KV 前提。
5. **"对 decode 无害"不等于"无害"**：FA_CHUNK 只影响 prefill，差点被当成
   设计使然（负面清单 1.6 的教训，放这里提醒）。
