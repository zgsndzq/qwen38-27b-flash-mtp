# 移植记录：qwen27b 现在跑 adaq8 的运行时

日期：2026-09-28

**一句话**：`qwen27b` 换上了 `adaq8` 的二进制与出餐配方。两者本来就是同一个模型
（`adaq8\models` 是指向 `qwen27b\models` 的 junction），所以这次只是把落后一整代的
**运行时 + 配方**搬回来，模型、JEV、训练库一律没动。

---

## 1. 移植了什么

| 项目 | 来源 | 落到 |
|---|---|---|
| 运行时二进制 | `adaq8\bin\*.dll / *.exe` | `qwen27b\bin\`（ada-q8，build 462，`c8b8993a6`，sm_89） |
| 出餐配方 | `adaq8\llmserver.bat` | `qwen27b\llmserver.bat`（按本项目重写，见 §3） |
| 草稿短表设置 | `--spec-draft-vocab-map` | 指向本项目的 `maps\shortlist_24k_mix.vocab.txt` |
| 说明文档 | `adaq8\bench\*` | `docs\ported-from-adaq8\`（含 `adaq8-README.md`） |
| 工具脚本 | `adaq8\tools\*` | `tools\ported-from-adaq8\`（路径已改写为本项目的） |
| 测试提示词 | `adaq8\skill.txt` | `skill.txt`（探针脚本依赖） |

## 2. 保留了什么（一个没动）

`jev-tetris\`、`jev-image\`、`trainlib\`、`vocab\`、`tools\`（原有的）、`src\`、`calib\`、`maps\`、`models\`、`cuda\`

**JEV 兼容性已核对**：两个 JEV 只用 `/health`、`/v1/chat/completions`、`/props`，
**没有用 `/embeddings`** —— 而 `--embeddings` 与 MTP 互斥，所以换运行时对 JEV 零影响。

## 3. 配置项改了什么，为什么

| 项 | 原来 | 现在 | 理由 |
|---|---|---|---|
| KV 缓存 | `q4_0` + `--kv-mean-center` | `q8_0`（已是代码默认，不再传） | 干净重测下 q4_0 是负收益 |
| `LLAMA_ATTN_ROT_DISABLE=1` | 开 | **删** | 它属于上面那套 kv-bias 标定，随标定一起弃用 |
| 推测类型 | `draft-mtp-adaptive` + `--spec-draft-n-min-adaptive 1` | `draft-mtp`（固定深度 3） | 自适应已移植并实测：3 提示 × 5 次，fixed3 46.61 vs adapt 46.10 = **−1.09%，重叠不可分**。短表中文覆盖 98.9%，不存在"掉词缩深度"这个自适应要治的病 |
| `EFFORT` | `low` | `medium` + `--reasoning-effort-allow medium` | effort 是**提示词注入开关**，不是长度偏好：模板对 `low`/`xhigh` 各往 system 塞一句话，只有 `medium` 一字不加。`-allow` 是夹子，否则客户端发 `high` 会让模板抛异常 → 每条请求 HTTP 500 |
| `--reasoning-budget 2048` | 开 | **删** | 到点会往思考块**中间**强注英文 "Budget exceeded, answer directly." 并截断 |
| `LLAMA_SPEC_PQ=0` | 关 p/q | **删**（走默认） | 温度扫描显示接受率几乎不随温度变（temp 0 → 0.4047，temp 0.6 → 0.4315），p/q 在这里买不到东西 |
| `GGML_CUDA_GRAPH_OPT=1` | 开 | **删** | 实测开反而 67.5 → 64.2 t/s |
| `GGML_CUDA_PDL=1` | 开 | **删** | 本就是默认，且在 sm_89 上是 no-op（PDL 要 sm_90 起） |
| `VISION` 强制降深度 | `VISION=1` 时把 DRAFT 压到 1 | **不再压** | 深度 3 是实测最优，挂 mmproj 不改变这点 |

## 4. 实测（本机 RTX 4060 Laptop，换完之后）

`tools\ported-from-adaq8\smoke.py`，走 `/v1/chat/completions` + 开思考：

```
health OK in 10.5s
vram used/total  = 7931 / 8188 MiB
decode  t/s      = 45.33      <== tg
prompt  t/s      = 107.0
draft accept     = 0.4042 (154 / 381), mean len = 2.21
load disk IO     = 1139.5 MB read / 0.0 MB written
```

启动证据（证明参数真的进了内核，而不只是"服务起来了"）：

```
MTP draft context uses the draft-only vocabulary shortlist '...\maps\shortlist_24k_mix.vocab.txt'
draft context sized to 24832 tokens (spec-draft-window 16384, target 32768)
n_slots = 1, n_ctx_slot = 32768
speculative decoding enabled: draft-mtp
```

裸速（不带思考、走 `/completion`）参考 `RESULT.md`：launcher 62.9–63.6 t/s，接受率 0.82。

⚠️ **显存**：`VISION=1`（为 `jev-image` 保留）挂了 mmproj，占用 7931/8188，只剩 257 MiB。
不跑识图时把 `VISION` 改成 `0` 可省约 288 MiB。

## 5. 怎么回退

⚠️ **2026-10-06 更新：旧二进制备份 `bin.bak-20260928-preadaq8\` 与 `llmserver.bat.bak-20260928-preadaq8`
已被清理，下面这套回退步骤不再可用。** 要回退现在只能重新编译指定 commit。

```
ren bin bin.adaq8            （或删掉）
ren bin.bak-20260928-preadaq8 bin
copy /y llmserver.bat.bak-20260928-preadaq8 llmserver.bat
```

## 6. 重建（改了 fork 之后）

`tools\ported-from-adaq8\_build_adaq8.sh` 仍然把产物装到 `adaq8\bin`（脚本里的目标路径没改，
因为 fork 是两个项目共用的）。重建后需要再手动把 `adaq8\bin\*.dll`、`*.exe` 拷到 `qwen27b\bin`。

## 7. `--gdn-replay`（GDN verify tape）—— 已评估，建议不搬

见 `PORT-SURVEY.md`。`perf/gdn-verify-tape-replay` 分支（2026-09-27，作者 bri-prism）已读并评估：
**对我们是负的，不建议 cherry-pick。**

| 事实 | 出处 |
|---|---|
| 作者实测端到端 **慢 3.4%**（M5 Pro Metal），默认关闭 | commit message 原文 |
| 它的收益机制是"放弃 fused rows-mode state read/scatter" | `delta-net-base.cpp`: 触发条件 `tape_mode && state_rows == nullptr` |
| rejected round 要多跑一个 replay op | commit message |
| 只是给未来 "fused tape write" 打的地基，本身不是优化 | commit message：`kept as the base for a fused tape write` |
| CUDA 后端**完全未验证**（作者只在 CPU/Metal 测） | 测试只覆盖 CPU/Metal |

我们的情况放大了负面项：MTP 实测 `acceptance 0.492 / mean len 2.48` —— 一半 token 不接受，
**回滚是常态，replay op 会频繁触发**。而且 rows-mode 正是我们在 CUDA 上省时间的东西，
上游 `8e18ce3be`(#318) 刚把它的正确性修好。**保持现状（rows-mode 快照路径 + #318）。**

真正值得留意的是上游把 **fused tape write** 做出来之后——那版才可能翻盘。

## 8. 同步记录：2026-10-06

**上游代码更新 + 重编 + 二进制同步。**

| 项 | 状态 |
|---|---|
| fork HEAD | `4dfe1bb1b`（merge 上游 `origin/prism` 4 个提交） |
| 上游那 4 个提交 | #265 ARM NEON PQ2_0 / #291 Vulkan / **#318 CUDA GDN 修复** / #304 DFlash2 opt-in lookup |
| 对我们有效 | **只有 #318**（CUDA GDN gather ids 内存复用时的正确性修复。Bonsai-2 是 qwen35+GDN 混合架构，65 层里约 49 层 GDN） |
| 其余三个 | ARM / Vulkan / DFlash2（我们走 CUDA + draft-mtp）**零影响** |
| 编译 | `F:\super\qwen27b\ada-q8\_build_q8.bat`，`BUILD_EXIT=0`，243/243，零 error，sm_89 |
| 验证方式 | `.obj` mtime 晚于源码（`--version` 串是 CMake configure 缓存，不可当判据） |
| 产物 | 15 个（5 exe / 10 dll），已铺到 `build-q8\bin` + `adaq8\bin` + `qwen27b\bin`，**sha256 三处一致** |
| 模型 | **无需同步** —— `adaq8\models` 是指向 `qwen27b\models` 的 junction，本就是同一份 |

### 底座新增（Swift）

| 文件 | 字节 | 说明 |
|---|---|---|
| `Swift-Bonsai-2-PTQ1_0.gguf` | 5,946,648,960 | ukisai 原厂主干，**无 MTP 草稿头**（851 张量 / `block_count=64`） |
| `Swift-Bonsai-2-PTQ1_0-mtp.gguf` | 6,212,666,880 | 嫁接我们自己的 `blk.64` 头后的成品（866 张量 / `block_count=65`） |

⚠️ **直接喂原厂件会硬退**：`context type MTP requested but model doesn't contain MTP layers`。
要用就用 `-mtp` 那份。原厂件留着是嫁接流程的输入，删了下次重做要重下（4 分钟）。

Swift 实测（同 prompt / 4500 token 上下文，对照旧 Ternary 底座）：
`acceptance 0.49228 vs 0.48867`、`mean len 2.48 vs 2.47`、`50.17 vs 49.81 t/s` —— **无损失，三项微涨**。

### 当前 `llmserver.bat` 与 adaq8 的 4 处差异（**有意的，不要盲目统一**）

| 行 | adaq8 | qwen27b | 性质 |
|---|---|---|---|
| 底座 `-m` | `Swift-Bonsai-2-PTQ1_0-mtp.gguf` | `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf` | 本项目仍在用旧底座做对照 |
| `REBUDGET` | 0 | 2048 | 思考预算护栏，两边策略不同 |
| `DRAFT_NTAIL` | 4 | 3 | **未实测**（`DRAFT_NTAIL=4` 是 `GGML_CUDA_PTQ1_MMVQ_MAX=5` 的最后一格，3 vs 4 一直挂着待验证） |
| 文档注释 | `docs\spec-*.md` | `docs\ported-from-adaq8\PORTED.md` | 各自指向自己的文档 |

## 9. 同步记录：2026-10-10（引擎 + 运行全量对齐）

**从 adaq8 全量同步，只保留本项目的应用级开关。上面 §8 那张「4 处有意差异」表从此刻起作废。**

| 项 | 内容 |
|---|---|
| 引擎 | `adaq8\bin\*` 15 个文件 → `qwen27b\bin\`，**sha256 15/15 一致**（ada-q8，10-10 合并上游 `origin/prism` 后重编，含 `--features-file` 机制） |
| 回退点 | `bin.bak-before-adaq8sync-1010\`（15 文件 / 258M）、`llmserver.bat.bak-before-adaq8sync-1010` |
| 底座 | 换成 adaq8 的 `Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf`（原 `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf`） |
| 配置机制 | 换成 `features.ini`（`--features-file`）：fork 特性全部搬进该文件，bat 只留运行时参数 |
| 调参 | 全对齐 adaq8：`draft-mtp-adaptive` + `--spec-draft-n-max 4`、`kv-vram-cells 32768`、`-rea off`、`p_min` 关、`probabilistic` 草稿采样、24k 短表 |

### 保留的应用级内容（本次一个没动）

- **VISION**（`--mmproj` 视觉头，`jev-image` 识图用）—— 在 `llmserver.bat` 里单独一段，保持 `VISION=0/1` 开关。
- **思考 / 记忆注入** —— 在客户端：`jev-image\bridge.py`、`jev-tetris\bridge.py` 每条请求带 `chat_template_kwargs.enable_thinking` 与系统提示词。这些文件未被触碰。
- `jev-image\` `jev-tetris\` `trainlib\` `vocab\` `src\` `calib\` `maps\` `tools\` —— 全部原状。

### 现在 `llmserver.bat` 与 adaq8 只剩两处差异

| 行 | adaq8 | qwen27b | 性质 |
|---|---|---|---|
| 文件头注释 | 指向 `docs\spec-旋钮配置与扫参.md` | 指向本文档 | 各自文档 |
| VISION 块 | 无（只有注释） | `set "VISION=0/1"` + `%VISARG%` | 本项目独有（`jev-image` 需要） |

`features.ini` 与 adaq8 **逐字一致**（仅第 2 行标题不同）。
