# adaq8 — 独立实验项目（Ada 内核手术栈）

**本目录与 `..\qwen27b` 是两个独立项目。`qwen27b` 一个字节都不改。**

## 这是什么

把 `professorpalmer` 的 **bonsai-q8-product**（他的「最新最全」分支）单独跑起来，
用**和 qwen27b 完全一样的 llmserver / llmchat 双 bat 模式**，并按**他公布的配方**调到位。

| | 内容 |
|---|---|
| 上游 fork | `https://github.com/professorpalmer/llama.cpp-ada-ternary`，分支 `bonsai-q8-product` |
| commit | `c8b8993a6` → `version: 0.2.0-dev (build 462)` |
| 参考配方 | `https://github.com/professorpalmer/bonsai-ada-surgery`（README 的 Windows 旋钮 + Linux 参考命令） |
| 源码 worktree | `F:\super\qwen27b\ada-q8`（git worktree，base = PrismML `9a9394a`） |
| 构建目录 | `F:\super\qwen27b\ada-q8\build-q8`（`CMAKE_CUDA_ARCHITECTURES=89` = Ada sm_89） |
| 本机 GPU | RTX 4060 Laptop 8 GB，compute_cap **8.9 = Ada**，带宽 **256 GB/s**（他们的测试卡 RTX 4070 同为 Ada，504 GB/s） |

## 实测结果

完整数据与推理：**`bench\RESULT.md`**。提示词固定 `skill.txt`，9 个臂 × 3 次取中位。

| | tg (t/s) | 接受率 |
|---|---:|---:|
| **本启动器（他们的配方）** | **62.9 – 63.6** | **0.82** |
| 旧配置（q4_0 KV + 校准 + 无窗口 + c32k） | 60.7 | 0.79 |
| 关掉起草窗口 | 61.8 | 0.79 |
| 起草深度 2 / **4** | 58.7 / **45.1** | 0.86 / 0.73 |
| ubatch 512 / 1024 | 60.5 / 56.8 | 0.78 / 0.78 |

**按带宽折算，他们的 4070 在 4k 上下文是 83 t/s；我们只有它 0.508 倍的带宽，
按比例应得 42 t/s，实测 63.6 = 归一化预期的 1.5 倍。**

## 怎么跑

1. 双击 `llmserver.bat` —— 等窗口出现 `main: server is listening on http://127.0.0.1:18080`
2. 双击 `llmchat\llmchat.bat` —— 开聊

想换旋钮直接改 `llmserver.bat` 顶部的 `SLOTS / CTX / VISION / REA / EFFORT / DRAFT_MAP`
（`TIER` 由 `CTX` 派生）。其余全是 **fork 的代码默认值**，bat 里不再出现：
`common/common.h` 管主+草稿两个 ctx 的 K/V q8_0、`-ub 256`、起草窗口 16384、起草尾段 4、
`temp 0.6 / top-k 20 / min-p 0`；`ggml-cuda/mmvq.cu` 管 PTQ1_0 的 PT-vs-MMQ 交叉点 5。
改完必须重编：`bash tools/_build_adaq8.sh`。

## 目录结构（镜像 qwen27b 的模式）

```
adaq8\
  llmserver.bat          <- 起服务 :18080（他们的二进制 + 他们的配方）
  bin\                   <- 真副本：ada-q8\build-q8\bin（12 个文件，含 ggml-cuda.dll 241 MB）
  calib\                 <- 真副本：kv-bias 校准 + corpus（只有 KV=q4_0 时才用）
  maps\                  <- 真副本：shortlist_24k_mix.vocab.txt（我们私有补丁的资产，他们的树用不上）
  models\                <- junction -> ..\qwen27b\models   （6.4 GB，不复制）
  cuda\                  <- junction -> ..\qwen27b\cuda     （698 MB，不复制）
  llmchat\               <- 客户端模式整套
    chat.py              <- 与主项目逐字节相同（md5 已校验）
    llmchat.bat          <- :18080
    sys.txt              <- system prompt
    mem\                 <- 记忆唯一来源（文本 + 图片）
  tools\                 <- 只留 5 个还在用的
    smoke.py             <- 30s 冒烟：起 bat → 等 health → 打启动证据 → 发一次请求 → 杀进程
    _srvlog.py           <- server stdout 全程留内存（smoke 依赖它；拒绝逐行落盘）
    _build_adaq8.sh      <- 重编 fork
    _install_bin.py      <- 把 build-q8\bin 里 md5 变了的文件装进 .\bin
    _swap_bat.py         <- 安全换 bat：写 sidecar → 校验 CRLF/ASCII → Move-Item
  bench\                 <- RESULT.md / PTQ1_CURVE.md / PLAN-COMBOS.md（结论）
  archive\2026-09-28-wrapup\  <- 一次性产物：bat/bin/fork 备份、探针、补丁脚本、逐臂日志
```

## 配置与 qwen27b 的差异

### ① 私有开关：短表**已经移植进来了**（不再是「删」）

`--spec-draft-vocab-map` 这套**是我们真正的原创**。他们的树里原本没有，09-28 整条移植过来：
`src/llama-mtp-vocab.h`（214 行）+ 5 处接线（`llama-context.h/.cpp`、`llama-graph.h/.cpp`、
`common/{common.h,arg.cpp,speculative.cpp}`、`include/llama.h`）。所以 adaq8 现在**也用短表**。

| 开关 | adaq8 |
|---|---|
| `--spec-type draft-mtp-adaptive` | 不用（那是主项目的自适应档） |
| `--spec-draft-n-min-adaptive 1` | 不用 |
| `--spec-draft-vocab-map maps\shortlist_24k_mix.vocab.txt` | **用**：起草一步读 26 MB 而非 265 MB |
| `--spec-draft-vocab-hot` | 保持代码默认 **0**（非 0 会让紧凑副本失效并**静默**退回全头） |

### ② 按他们的配方改掉的部分（都实测过）

| 项 | qwen27b | adaq8 | 为什么 |
|---|---|---|---|
| K/V 精度 | `-ctk/-ctv q4_0` + `--kv-mean-center` 校准 | **`-ctk/-ctv q8_0`**，不挂校准 | 他们的原话：q4_0 翻转的 top token 是 q8_0 的 **12 倍**；而速度上两者打平（18.01 vs 18.28 轮/s） |
| KV 旋转 | `LLAMA_ATTN_ROT_DISABLE=1` 强制关 | **开着**（默认） | 那个 env 是校准文件的配套要求，挂了校准就没有理由关掉这个质量特性 |
| 上下文 | `-c 32768` 全显存 q4_0 | **`-c 32768` 全显存 q8_0** | 短表上线后 32k 够用；`CTX>32768` 会自动派生 `--kv-vram-cells 24000` 分层 |
| K/V 显存占用 | 576 MiB | 835 MiB（+1.42 GB 主机） | 视觉关掉后腾出的显存用在这里 |
| 起草窗口 | 无 | **`--spec-draft-window 16384`** | 任意深度起草成本恒定 ⇒ 接受率 0.789 → 0.823 |
| 起草缓存 | 默认 F16 | **`-ctkd q8_0 -ctvd q8_0`** | 他们的配方 |
| 起草尾段 | 无 | `--spec-draft-n-max-tail 4` | 越过分层线后的起草宽度（我们深度不够，暂不生效） |

### ③ 保持原样的（都验证过是最优）

- `-b 4096 -ub 256`（ub 512/1024 两项都更差，1024 的 prefill 从 490 掉到 217 t/s）
- `-t 8 -ngl 99 -fa on --parallel 1`。⚠️ `--parallel 1` **不是噪音行**：服务端 example 把
  `n_parallel` 预设成 `-1`（auto→4），结构体初值 1 不生效；删掉变 4 slot，实测 **8.2 t/s**（基线 61.7）
- `-rea on`（本机默认开思考）
- `GGML_CUDA_GRAPH_OPT=1` / `GGML_CUDA_PDL=1` —— **CUDA 图是 1.5 倍**：关掉后同一配置 63.6 → 41.3 t/s
- MMA 解码路由、FWHT 自量化 Hadamard、GDN 状态 gather 融合、PTQ1_0 多列 mat-vec：源码里都是**默认开**，不需要开关

### ④ 未采用他们的 harness-proofing

`--reasoning-effort-allow` / `--reasoning-max-tokens-floor` / `--jinja`：那是给 Cline、Open WebUI
这类会发 `effort:"high"` 的客户端防 HTTP 500 用的，与速度无关，我们的客户端不发。

## 他们的新旋钮（源码位置与实测判决）

`common/common.h:common_params_speculative_draft` 里新增，我们把结论一次说清：

- `--spec-draft-window N` —— 草稿上下文只保留最近 N 行。
  源码注释：*"16k rows accept as many drafts as the full history at 131k"*。
  **实测：16384 最优**（w4k 62.17 / w8k 62.28 / w16k 63.56）——调小省不下时间，只掉接受率。
- `--spec-draft-depth-max N` —— 超过 N token 停止起草（默认 0 = 永不停止）。
  他们靠窗口机制把这个上限**取消了**，所以保持 0。
- `--spec-draft-n-max-tail N` —— 越过 `--kv-vram-cells` 分层线后的起草宽度。
- `--kv-vram-cells N` —— 分层 KV：前 N 格留显存，其余落 pinned host RAM 并映射进同一设备地址空间。
  `34.8 KB/q8_0 位置`（他们「262144 位置 = 9.1 GB KV」反推，与实测显存对得上）。

> ⚠️ **纠正一条早先的错判**：前面 A/B/C/D 四臂是在 **`-rea on`（思考开着）** 下跑的，
> 生成的 token 大量落在 `<think>` 里，且 C 臂用的是 **q4_0 目标 KV + F16 起草缓存**。
> 因此「`--spec-draft-window` 是负收益」的结论**不成立**——那是被思考档和缓存类型污染的。
> 关掉思考 + q8_0 双向缓存后，窗口是**正收益**（+3.6% 接受率）。

## 测速工具

```bash
cd F:/super/adaq8
python tools/speed_probe.py --list                  # 列出全部内置臂的完整命令行
python tools/speed_probe.py --arm T3 --runs 3       # 单臂，用内置参数
python tools/speed_probe.py --bat llmserver.bat --label FINAL   # 直接测 bat 本身
```

工具会：先杀掉 18080 的占用者 → 起服务 → 等 `/health` → 发 `skill.txt` → 抓**服务器自己的**
`timings` 与日志里的 `draft acceptance` / `mean len` → 杀自己的进程树 → 写 `bench/arms.json`。

两个坑已内置修掉，值得记：
- 他们的 llama-server 新版本**不给 `-m` 就进 router 模式**（不加载模型、所有请求 400），
  所以臂定义里必须带 `-m`。
- 服务日志里 `draft acceptance = ... , mean len = X` 的 X 就是「每轮产出几个 token」，
  有了它才能把「接受率」和「内核快慢」分开算（`rounds/s = tg ÷ mean_len`）。

## 构建要点（复现用）

- `-DGGML_NATIVE=OFF` **必须**显式关掉，并显式开 `GGML_AVX2=ON / GGML_BMI2=ON / GGML_FMA=ON / GGML_F16C=ON`。
  MSVC 下 `GGML_NATIVE=ON` 会让 `FindSIMD` 把 AVX/AVX2/BMI2/SSE42 全留成 OFF **且不报错**，
  CPU 量化内核退化成标量（本项目实测慢 12.5×）。
  判据：`grep -iE "^GGML_(AVX2|BMI2):" build-q8/CMakeCache.txt` 必须 ON，
  且 `grep -oE "/arch:AVX2|__BMI2__" build-q8/build.ninja` 非空。
- 构建走 `cmd /c _build_q8.bat` + **Bash 前台**（超时会自动后台化且不杀进程）。
  PowerShell 工具的后台任务吃 120 s 上限，到点连根杀，子进程变孤儿继续跑、日志被切在半句警告上
  —— 看着像编译错误，实际日志里零个 `error`。
- 同一 build 目录**只允许一个 ninja**。被杀的旧任务留下的孤儿 ninja 会把日志截断重写。
- 这个树编出来的 `llama-server-impl.dll` 只有 3.1 MB（ada-combo 的同名文件 13 MB）**不是残件**：
  日志写明 `UI: no assets available - building without an embedded UI`，离线拉不到 Web UI 资源，
  纯 API 跑分不受影响。

## 收工状态（2026-09-28）

**结论在 `bench\*.md`，配方在 fork 源码默认值里，bat 只剩 6 个旋钮。**

### 这次收工时拆掉的脚手架

| 拆掉的 | 它原本是干什么的 |
|---|---|
| `src/llama-dv-trace.h` + 4 处调用 | 短表移植期的分支追踪。**它在 MTF 起草图构建里被调用，即每生成一个 token 写几次盘** —— 形态就是「一 token 一次 SSD 写」。移植完成后已整条删除，`LLAMA_DRAFT_VOCAB_TRACE` 这个 env 也不存在了 |
| `tools\speed_probe.py` / `ab_bench.py` / `e2e_smoke.py` | 扫参 / A/B / 端到端，配方定稿后无用 |
| `tools\_port_shortlist*.py` / `_bake_defaults.py` / `_patch_*.py` | 一次性补丁，结果已经落在源码里 |
| `tools\disk_probe / disk_blame / io_* / mem_state / gguf_meta` | 排查「谁在动盘」用的一次性探针 |
| `bench\_*.log` × 26 | 逐臂服务日志，结论已写进三个 .md |
| `bin\*.bak-*` / `llmserver.bat.bak_*` / fork 里的 `*.bak-*` | 换装与改源码时的备份 |

全部在 `archive\2026-09-28-wrapup\`（**没有删除，移动即可还原**）。

### 关于「SSD 鸡鸡叫」

实测过：推理**全程不碰盘**。进程累计 IO 在 10 秒采样里一动不动
（读 41 MB / 写 **0.0** MB —— 41 MB 是加载时的元数据，模型本身走的页缓存），
全系统 15 秒里所有进程读 1.00 MB / 写 0.82 MB，而 `llama-server` **不在榜上**。

动盘的是别人：`SiYuan.exe`（思源笔记）持续 ~2.5 MB/s 写入、每秒上千次写操作；
`node.exe`（编辑器侧）偶发一秒 6000 次小写。真正属于本项目的只有**每次启动加载模型
的 5.79 GB 读**（页缓存没命中时），以及 `speed_probe.py` 那种 9 臂 × 3 次 = 27 次起停。

**规矩**：`llama-server` 的 stdout 一律留在内存（`tools\_srvlog.py` 的 `MemTail`），
只在失败时 dump 一份。不要给 token 路径上加任何写盘。
