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
| 底座 | `Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf` | `llmserver.bat` → `MODEL` |
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

## 目录结构（2026-10-11 清理后）

| 路径 | 内容 | 大小 |
|---|---|---|
| `llmserver.bat` | 唯一启动器（注释 **必须纯 ASCII**，见"维护注意"） | 1.3 KB |
| `features.ini` | 唯一特性配置（英文注释） | 5.2 KB |
| `bin\` | `llama-server.exe` + 全部 dll | **15 个文件** / 268 MB |
| `bin.bak-before-adaq8sync-1010\` | **唯一保留的回退点**：同步 adaq8 引擎之前的那份 | 258 MB |
| `cuda\` | CUDA 运行时（cublas / cublasLt / cudart / nvrtc / nvJitLink） | 698 MB |
| `models\` | 6 个 gguf（当前底座 + 备选 + mmproj） | 30 GB |
| `maps\` | 生产草稿短表（24k 行） | 136 KB |
| `calib\` | kv-bias 校准数据（**历史，当前配方不再使用**） | 100 KB |
| `docs\` | 实测账、设计稿、移植记录（见下） | 232 KB |
| `tools\` | 测量脚本 + `_variants\`（归档变体） | ~330 KB |
| `src\` | 历史构建脚本 / patch / 设计文档（**仅留档，不影响运行**） | 141 KB |
| `jev-image\` | 应用：图像 JEV（含 `skill-chess-board.md`） | 582 KB |
| `jev-tetris\` | 应用：俄罗斯方块 JEV | 952 KB |
| `trainlib\` | 语料/词表训练库 | 19 MB |
| `vocab\` | 词表构建工程（含 `out\` 产物） | 234 MB |

合计 **≈ 31.4 GB**（其中 `models\` 占 30 GB）。

---

## 与 `..\adaq8\` 的关系

两个项目**跑的是同一份引擎 + 同一份 gguf**，只差目录和启动配置：

- `adaq8\models` 和 `adaq8\cuda` 都是**指向本包的软链** ⇒ 权重与 CUDA 运行时全机只有一份；
- 引擎同步方向：`adaq8\bin\*` → 本包 `bin\`（2026-10-10 做过一次，sha256 15/15 一致）；
- 两边 `features.ini` 逐字一致，`llmserver.bat` 只差本项目独有的 `VISION` 视觉开关段。

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
python mtp_head_probe.py ..\models\Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf
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
2. **本包只带 CUDA 运行时，没有编译工具链。** 要重编引擎，得回到 `F:\super\llm\forks\ada-q8`（完整 CUDA 在 `F:\super\tools\cuda`，含 nvcc）。
3. **`cuda\bin` 里精简掉的 dll 若被新二进制需要，会启动即失败。** 换二进制后要重测。
4. **换底座模型时**：`llmserver.bat` 的 `MODEL` 与 `features.ini` 里任何与模型绑定的项（如短表路径）要一起核；`VISION=1` 的 mmproj 需与新底座匹配。
5. **换草稿头不必重新嫁接**：trunk 相同，直接替换 weights 里的 gguf 即可。
6. **不要把 `bin.bak-before-adaq8sync-1010` 当垃圾删掉** —— 它是当前唯一的引擎回退点。

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
| `bin\` | `F:\super\llm\forks\ada-q8\build-q8\bin`（2026-10-10 合并上游后重编，含 `--features-file` 机制） |
| `bin.bak-before-adaq8sync-1010\` | 2026-10-10 同步前的旧引擎（v1 全解码缓存 + 手工缝合的 #220 GDN gather 折叠） |
| `cuda\bin\` | `F:\super\tools\cuda\bin`（只取运行时 dll；**未带** nvcc / ptxas / nvdisasm） |
| `models\` | `llm\models\`（Bench：`Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf`） |
| `src\` | `llm\forks\llamAmpere_gdn\` 的构建脚本与设计文档（仅留档） |

---

## 相关

- `..\adaq8\` —— 姊妹项目，**同一份引擎 + 同一份 gguf**，多了 ARC 评测工作区。软链指向本包的 `models\` / `cuda\`。
- `..\llmchat\` —— 独立对话客户端（`chat.py` + `llmchat.bat` + `imgtest.bat`），连本包起的服务用。system prompt 在 `llmchat\sys.txt`。
- `..\llm\docs\Runbook.md` —— 完整工程手册（运行时开关 / 自适应深度）。
- `docs\ported-from-adaq8\PORTED.md` —— 同步记录与回退步骤。
