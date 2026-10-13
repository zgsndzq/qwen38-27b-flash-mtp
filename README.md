# qwen38mtp — Bonsai-2-27B 三值量化 + MTP 自投机 独立运行包

一个**自包含**的推理运行包：把 `llmserver.bat` 跑起来所需的一切都装在这个文件夹里。
整个文件夹可以拷到任意盘符、任意 Windows x64 机器上，**只要装了 NVIDIA 驱动就能跑**
（不需要 CUDA Toolkit —— CUDA 运行时已随包携带）。

包内**没有任何绝对路径**，启动脚本用 `%~dp0` 相对寻址。

---

## 启动：两个文件

| 文件 | 角色 |
|---|---|
| `llmserver.bat` | **唯一启动器**。只放引擎认识的普通运行参数（`-c -b -t -ngl -fa -rea --spec-type --alias` 等）。 |
| `features.ini` | **唯一特性入口**。放上游引擎不认识的东西 —— 我们 fork 的开关（`spec-draft-*`、`kv-vram-cells`、`kq-mask-packed` …）以及全部 `GGML_CUDA_*` / `LLAMA_*` 环境变量。 |

启动器里用的是 `--features-file "%~dp0features.ini"`，**引擎自己读**这个文件。
⇒ 想改我们的特性，只改 `features.ini`；想改底座/端口/上下文，只改 `llmserver.bat`。
命令行显式给的参数**优先于** `features.ini`。

实验变体（固定深度 / nomap / lv4 / 各 draft 深度 …）在 `tools\_variants\`，**不在包根**。

---

## 当前配方（2026-10-11）

| 项 | 值 | 在哪 |
|---|---|---|
| 底座 | `Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf`（在 `weights\`） | `llmserver.bat` → `MODEL` |
| 视觉头 | `mmproj-Qwen3.8-27B-Q8_0.gguf`（`VISION=1` 时挂） | `llmserver.bat` |
| 上下文 | `-c 65536` | `llmserver.bat` → `CTX` |
| 投机 | `--spec-type draft-mtp-adaptive` + `--spec-draft-n-max 4` | `llmserver.bat` |
| 思考 | `-rea off`（客户端可用 `chat_template_kwargs` 覆盖）+ `--reasoning-budget 2048` | `llmserver.bat` |
| 采样 | 请求温度 0.6（客户端 `llmchat.bat --temp`） | 客户端 |
| KV | q8_0 + tiered：`kv-vram-cells = 32768`（超出部分读 pinned RAM） | `features.ini` |
| 草稿采样 | `spec-draft-sampling = probabilistic`（真采样 + 拒绝采样） | `features.ini` |
| 草稿短表 | `spec-draft-vocab-map = maps/shortlist_24k_mix.vocab.txt`（24k 行） | `features.ini` |
| 自适应地板 | `spec-draft-n-min-adaptive = 1`（天花板是 `--spec-draft-n-max 4`） | `features.ini` |
| 显存池 | `GGML_CUDA_SHARED_POOL = 1`（target + draft 共池，8G 卡省 ~1.4G；**仅在 `--parallel 1` 下安全**） | `features.ini` |
| #189 短 query | `GGML_CUDA_FA_SHORTQ_MMA = 0`（忽略上游 #189，保留原地 q8_0 快路） | `features.ini` |

⚠️ 与 `--parallel 1` 绑定：一个 slot 只服务一个会话。同一张卡**不要同时跑两份服务**（端口和显存都会冲突）。

---

## 目录结构（2026-10-13 按性质拆分）

顶层只有一张表，五类东西五种去处 —— **Git 只跟「代码」那一类**：

| 路径 | 性质 | 进 Git | 大小 | 分发渠道 |
|---|---|---|---|---|
| `llmserver.bat` | 运行入口（注释 **必须纯 ASCII**，见"维护注意"） | ✅ | 3 KB | Git |
| `features.ini` | **唯一特性配置**（英文注释） | ✅ | 5 KB | Git |
| `maps\` | 生产草稿短表（24k 行） | ✅ | 136 KB | Git |
| `llmchat\` | 对话客户端（`chat.py` + 启动 bat + `mem\`） | ✅ | 2.0 MB | Git |
| `apps\` | 图形应用：`jev-image\`（含 `skill-chess-board.md`）、`jev-tetris\` | ✅ | 1.5 MB | Git |
| `engine\` | 引擎侧留档：PTQ1 构建脚本 / `_dflash_bonsai.patch` / 交接文档 | ✅ | 141 KB | Git |
| `docs\` | 实测账、设计稿、移植记录（见下） | ✅ | 232 KB | Git |
| `tools\` | 测量脚本 + `_variants\`（归档变体） | ✅ | ~330 KB | Git |
| `weights\` | 6 个 gguf（当前底座 + 备选 + mmproj） | ❌ | 30 GB | **Hugging Face** |
| `build\bin\` | `llama-server.exe` + 全部 dll | ❌ | 331 MB | **GitHub Releases** |
| `build\cuda\` | CUDA 运行时（cublas / cublasLt / cudart / nvrtc / nvJitLink） | ❌ | 697 MB | Releases（同包） |
| `train\` | 训练工程：`ternary\` `trainlib\` `vocab\` `calib\` `cute-triadic-gdn\` | 部分 | 3.1 GB | 代码→Git，语料/产物→HF |
| `data\huggingface\` | 数据准备工程 | ❌ | 2.0 GB | **HF dataset** |

合计 **≈ 36 GB**，但 **Git 只跟踪 125 个文件 / 3.2 MB** —— 大件全在 `.gitignore` 里，
各有各的渠道，`git status` 以后只会看到源码和配置。

⚠️ **引擎源码不在这里**。它是 `F:\super\qwen38mtp\ada-q8`（对 `PrismML-Eng/llama.cpp` 的 fork），
本包只**消费**它的编译产物（`build\bin\`）。

---

## 与 `..\adaq8\` 的关系（正在合并）

`adaq8\` 是本包的一个**旧副本 + ARC 评测工作区**，跑的是同一份引擎 + 同一份 gguf。
**2026-10-13 起本包是唯一主仓**（`adaq8\` 没有自己的 `.git`）：

- `adaq8\weights` `adaq8\build\*` 应改成指向本包的 junction ⇒ 权重与 CUDA 运行时全机只有一份；
- 引擎同步方向：`adaq8\build\bin\*` → 本包 `build\bin\`（2026-10-13 做过，sha256 **15/15 一致**）；
- 两边 `features.ini` 逐字一致；`llmserver.bat` 只差本项目独有的 `VISION` 视觉开关段；
- 两边的 ngram 参数已统一为 `NMATCH=24 / NGMIN=24`。

**要拆掉 `adaq8\` 之前，先安置它独有的三样** —— `train\ternary\`、`data\huggingface\`、
`train\cute-triadic-gdn\`，它们并不属于本包。

移植/同步的完整记录与回退方式见 `docs\ported-from-adaq8\PORTED.md`。

---

## 包内工具（`tools\`）

| 工具 | 作用 |
|---|---|
| `bat_audit.py` | **只读**体检：非 ASCII / CRLF / BOM / GBK 损伤行；`--diff a.bat b.bat` 打印**真正的命令行差异**（不信任注释） |
| `ab_bench.py` | 一条命令 A/B 任意 bat 变体：`--bat a.bat --bat b.bat`，自动起服务、等 `/health` 200、给中位数与**零重叠判定** |
| `mtp_head_probe.py` | 零依赖最小 GGUF 读取器：读词表条数、头张量名/类型、`prism.hadamard.*` —— 预判某特性能否挂上某模型 |
| `think_ab.py` | 思考开 / 关的每 token 吞吐对照 |
| `ctx_ab.py` | 按上下文长度分段给 decode 速度，量 KV 复用了多少（`cache_n`） |
| `gen_profile.py` | 单条回答内分段速度 + `nvidia-smi` 采样按同一时间轴对齐 —— 判定"该甩锅给模型还是 GPU" |
| `keep_reasoning.py` | 测"思考块跨轮保留"的真实代价 |
| `_variants\` | 归档的启动器变体（含 README 说的"唯一干净基线臂" `llmserver-nomap.bat`） |
| `ported-from-adaq8\` | 从 adaq8 移植过来的探针脚本（网络抓包 / 端点校验等） |

```bat
cd F:\super\qwen38mtp\tools
python bat_audit.py
python ab_bench.py --bat llmserver.bat --prompts scenes --runs 5 --tag mytest
python gen_profile.py --pad 0,2000 --max-tokens 2560 --width 128
python mtp_head_probe.py ..\weights\Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf
```

---

## 文档

| 文件 | 内容 |
|---|---|
| `docs\BENCH.md` | **包内实测账**：什么有效、什么否决、测量铁律、静默失效判据、踩过的坑 |
| `docs\VOCABMAP.md` | 草稿短表（24k）的机理与收益账 |
| `docs\design-vocab-shortlist.md` | 短表设计的完整推导（按代码用途重算收益） |
| `docs\SPEEDUP-BUDGET.md` / `TERNARY-SPEEDUP-SURVEY.md` | 提速预算与调研 |
| `docs\ported-from-adaq8\PORTED.md` | 与 adaq8 的同步记录、**回退方式** |
| `docs\ported-from-adaq8\VOCAB-CJK.md` | 中文/英文/代码三种内容下短表与自适应深度的实测 |
| `docs\JEV.md` | JEV 应用（视觉/图形）说明 |

---

## 维护注意

1. **`llmserver.bat` 和 `features.ini` 的注释必须是纯 ASCII。** UTF-8 中文注释在 cmd(936) 下按 GBK 解码会错位：某行中文字节数为奇数时会与下一行首字节配对、吃掉下一行的 `rem`，后面被当命令执行，报一堆 `'xxx' is not recognized`。**已实测踩过。**
   ⇒ 改完任何 bat 都跑一次 `python tools\bat_audit.py`，再加 `--diff llmserver.bat <改后的>` 确认"真正改了什么"。
2. **本包只带 CUDA 运行时，没有编译工具链。** 要重编引擎，得回到 `F:\super\qwen38mtp\ada-q8`（完整 CUDA 在 `F:\super\tools\cuda`，含 nvcc）。
3. **`cuda\bin` 里精简掉的 dll 若被新二进制需要，会启动即失败。** 换二进制后要重测。
4. **换底座模型时**：`llmserver.bat` 的 `MODEL` 与 `features.ini` 里任何与模型绑定的项（如短表路径）要一起核；`VISION=1` 的 mmproj 需与新底座匹配。
5. **换草稿头不必重新嫁接**：trunk 相同，直接替换 weights 里的 gguf 即可。
6. **不要把 `F:\super\llm\_backup\bin-1012-prepartial\` 当垃圾删掉** —— 它是当前唯一的引擎回退点
   （包内的 `bin.bak-*` 已随 2026-10-13 的拆分清掉）。

---

## 性能

**口径提醒：** 本包最后一次**正式 A/B** 是 2026-09-22，基线是旧底座
（`Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf` + 短表 + 固定深度）。换底座后的正式 A/B 尚未做，
下面的表**不要跨底座直接比对**。

2026-09-22 实测，**同会话**、各 5 次取中位数（完整证据链见 `docs\BENCH.md`）：

| 配置 | 代码 | 散文 | 显存峰值 |
|---|---|---|---|
| 固定深度 3（当次基准） | 33.28 t/s | 30.28 t/s | 7523 MiB |
| ✅ `adaptive [1..3]` | **34.93（+5.0%）** | **33.47（+10.5%）** | 7557 MiB |
| `DRAFT=1`（另一口径） | 打平 | **37.7（+20%）** | — |

- **自适应是真赢，不是噪声**：散文那组 5 对 5 **零重叠**，而且它跑在后面（热降频更重）还赢 10.5%。
- **`--spec-draft-n-max` 别往上调**：控制器会爬过设定值，每轮验证成本近似**线性于深度**、产出是次线性的，越深越亏。
- 换草稿头（Q5_0+Q8_0 → Q6_K/Q4_K）本身只值 **+1~3%（噪声内）**。
- ⚠️ **测量噪声很大**：同配置 9 次 tg 散布可达 **±12%**，跨会话差异有 7%。任意 A/B 必须**同会话**、≥3 提示 × ≥3 次取中位数。

2026-10-11 冒烟实测（新底座 + adaptive + `features.ini`，**探针口径，非 A/B**）：

| 场景 | 结果 |
|---|---|
| 预填充（短提示） | 155 t/s |
| 预填充（长提示，4096 分块起） | 500 → 391 t/s（29.6k token 全程） |
| 逐字生成（关思考） | 20 ~ 41 t/s（与回答内容强相关） |
| 多轮增量复用 | 第 2 轮 `prompt_n = 22 / cache_n = 1124`（**正常增量**） |

---

## 来源（Provenance）

| 包内 | 原位置 |
|---|---|
| `build\bin\` | `F:\super\qwen38mtp\ada-q8\build-q8\bin`（2026-10-13 合入 14 个上游提交后重编） |
| `build\cuda\bin\` | `F:\super\tools\cuda\bin`（只取运行时 dll；**未带** nvcc / ptxas / nvdisasm） |
| `weights\` | `Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf`（生产底座）等 6 个 gguf（2026-10-13 从 `models\` 改名） |
| `engine\` | 原 `src\`：PTQ1 构建脚本、`_dflash_bonsai.patch`、Ampere / SM86 交接文档（仅留档） |

---

## 相关

- `..\llm\forks\ada-q8\` —— **引擎源码仓**（fork of `PrismML-Eng/llama.cpp`）。`build\bin\` 从它来；
  要重编引擎去那里（CUDA Toolkit 在 `F:\super\tools\cuda`）。
- `..\llm\_backup\bin-1012-prepartial\` —— **引擎回退点**（当前唯一整套二进制备份）。
- `..\adaq8\` —— 旧副本 + ARC 评测工作区，正在合并进本包。
- `docs\ported-from-adaq8\PORTED.md` —— 同步记录与回退步骤。
