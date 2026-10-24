# Qwen3.8-27B-Flash-MTP — 运行包

三值量化（`PTQ1_0`）的 27B 代码模型，MTP 自投机草稿头**烘在同一个 gguf 里**。
权重 5.79 GB，**8 GB 显存的消费级卡就能整模型驻留**。

引擎是对 `PrismML-Eng/llama.cpp` 的 fork —— 官方 llama.cpp **跑不了这个模型**
（`PTQ1_0` 张量类型、MTP 草稿上下文都是这个 fork 才有的）。

> English: see `README.en.md`

---

## 一、三处下载

| 从哪 | 拿什么 | 多大 | 必须吗 |
|---|---|---|---|
| **Hugging Face** → `SuperLogic/Qwen3.8-27B-Flash-MTP-GGUF` | 权重 | 5.79 GB + 0.59 GB | **必须** |
| **GitHub Releases** → `qwen38-27b-flash-mtp-win64-<commit>.zip` | 预编译二进制 + CUDA 运行时 + 本包 | ≈600 MB | 不想自己编就要 |
| **GitHub 源码仓** → `zgsndzq/llama.cpp-qwen38-flash` | 引擎源码 | 400 MB | 想自己编才要 |
| 本仓 → `zgsndzq/qwen38-27b-flash-mtp` | 启动脚本 + `features.ini` + 客户端 + 源码 | — | 就是你现在看的这个 |

**预编译 zip 与引擎版本要配对**：换引擎必须整包换，不要只替换 dll。

---

## 二、怎么摆

解压 zip，把权重放进去：

```
qwen38-27b-flash-mtp\
├── build\
│   ├── bin\            ← llama-server.exe + 全部 dll（zip 自带）
│   └── cuda\bin\       ← CUDA 运行时（zip 自带，**不用装 CUDA Toolkit**）
├── weights\            ← ★ gguf 放这里
│   ├── Qwen3.8-27B-Flash-MTP.gguf        主模型 5.79 GB
│   └── mmproj-Qwen3.8-27B-Q8_0.gguf      视觉 0.59 GB，要读图才需要
├── maps\               ← 24k 草稿短名单（自带，别删）
├── llmchat\            ← 命令行客户端（可选）
├── code\               ← 引擎源码（自己编才用）
├── features.ini        ← fork 自己的开关都在这里
└── llmserver.bat       ← 双击启动
```

`weights\` 目录名不能改（启动脚本按 `%~dp0weights\` 找），
文件名要和 `llmserver.bat` 里的 `MODEL=` 一致。

---

## 三、跑起来

**不需要编译**，双击 `llmserver.bat`。看到这两行就成了：

```
load_model: initializing, n_slots = 1, n_ctx_slot = 131072
llama_server: listening on http://127.0.0.1:18080
```

然后打 API：

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```

或者用附带的客户端 `llmchat\llmchat.bat`（支持 `clear` 清空、退出后重开可续上次对话）。

**前置条件**：NVIDIA 驱动，显存 **6 GB 起**。默认 `CTX=131072`、`kv-vram-cells=32768`
（KV 一半在显存、一半在主机内存）。6 GB 卡把 `CTX` 调小到 `32768`，同时 `features.ini`
的 `kv-vram-cells` 也要 ≤ 它。

**基准速度**（RTX 4060 Laptop 8 GB，thinking off）：

| 场景 | 速度 | 为什么 |
|---|---|---|
| 代码生成（ngram 草稿命中率高） | **70 t/s** | 接受率 0.65，每轮带 4.2 token |
| 改写已有文件（几乎原样输出） | **170 t/s** 稳态，峰值 260 | 接受率 0.85，每轮带 **41** token |
| 开放式闲聊 | 30–40 t/s | 接受率 0.3 左右，每轮 1 个 |
| prefill | **450–500 t/s**（浅）→ 265（32K）→ 213（64K） | |

decode 的账按「每轮」算：一轮前向读一遍 5.79 GB 权重，8 GB 笔记本卡有效带宽约
215 GB/s ⇒ **28–29 ms 一轮**。

- **代码生成**：一轮带 4.2 token ⇒ 带宽上限 ≈145 t/s，实测 70 = **48%**，差距在启动开销、5 行 verify、采样。
- **改写**：一轮带 41 token，带宽只要 29 ms，但 42 行的 verify 变成**算力瓶颈**
  （2 × 27B × 42 ≈ 2.3 TFLOP / 245 ms ≈ **9 TFLOPS 有效**）。这一段提速度靠的是多行矩阵乘的 kernel 效率，不是带宽。

---

## 四、编译：三个必读（会浪费一下午的坑）

自己编译只需要一条命令，但有三件事不知道就会浪费一下午。

### 1. 从 IDE / 自动化工具里编译时，nvcc 会"无声失败"

**症状**：`_build_inc.bat` 跑完 `BUILD_EXIT=1`，所有 CUDA 文件 `FAILED`，但日志里
**一条 `error:` 都没有**。直接手跑 `nvcc -c 空文件.cu` 也是 `EXIT=1` 且**零输出**，
而 `nvcc --version`、`ptxas --version`、`cicc --help` 全部正常。

**原因**：有些 IDE / agent 运行时会往进程环境里注入几十 KB 的变量（MCP 配置、shim 路径、
编码/沙箱参数）。nvcc 给 `cicc` / `cudafe++` / `ptxas` 传环境时被这些内容撑坏，
子进程起不来，nvcc 又不报错，于是表现为"退出码 1 + 静默"。

**判定**：`nvcc --use-local-env -c 任意空.cu` 能成功 → 就是这个原因。

**现在的处理**：`ggml/src/ggml-cuda/CMakeLists.txt` 已经无条件加了

```cmake
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--use-local-env>")
```

`--use-local-env` 让 nvcc 用当前进程的环境重新解析，在正常终端里是无害的空操作，
所以**不需要你做什么**。如果你是老版本代码遇到这个症状，先确认这行在不在。

> 别去改 `build/src/CMakeFiles/rules.ninja` —— 每次 CMake 重新配置都会覆盖它。
> 要改就改 CMakeLists。

### 2. 改过 CMake 文件之后第一次编译是全量

改了 `CMakeLists.txt` / `*.cmake` 后，ninja 会先 `Re-running CMake...`，
然后**全部 500+ 个目标重建**（编译选项变了，所有 obj 失效）。16 核机器大约 10–15 分钟，
中途超时或被打断**不用从头来**，`_build_inc.bat` 再跑一次就从断点继续。

### 3. 增量编译 vs 全量编译

| 场景 | 命令 | 说明 |
|---|---|---|
| 改 `.cpp` / `.cu` / `.cuh` | `_build_inc.bat` | 只重编受影响的目标 |
| 换过目录、改过 CMake、编译结果可疑 | `_build_q8.bat` | 清掉 CMake cache 重配 |
| 只改 `.cuh`（头文件） | `_build_inc.bat`，但见下 | ninja **不追踪 `.cuh` 依赖**，改了头文件要 `touch` 对应的 `.cu`，否则是空编译 |

产物在 `build\src\bin\`，日志 `build\logs\_build_inc.log`，最后一行 `BUILD_EXIT=0` +
`SERVER_OK` 才算成功。

**编译前先停服务**：`llama-server.exe` 会锁住 `ggml-cuda.dll`，链接必然失败。

```bat
taskkill /F /IM llama-server.exe
```

---

## 五、自己编译的完整步骤

```bat
git clone https://github.com/zgsndzq/llama.cpp-qwen38-flash.git
cd llama.cpp-qwen38-flash
copy build.local.example.bat build.local.bat
_build_q8.bat
```

**`build.local.bat` 不用手改**：模板会自己找工具链 —— MSVC 用 `vswhere`（任意 VS 2022
版本，含 BuildTools）、CMake/Ninja 支持独立安装或 `pip install cmake ninja`、
CUDA 支持 `CUDA_PATH`、标准安装目录或 `pip install nvidia-cuda-nvcc`。
显卡架构默认按本机自动探测（`CUDA_ARCH=native` → 3060 得 `86`、4070/4060 得 `89`）。

需要装：**VS2022（MSVC）+ CUDA Toolkit + CMake + Ninja** 中至少一套。

| 变量 | 默认 | 说明 |
|---|---|---|
| `CUDA_ARCH` | `native` | 唯一会让"编译成功但跑不起来"的值，交给自动探测最稳 |
| `JOBS` | 核数−4，上限 12 | 太大 nvcc 会吃爆内存 |

产物落在 `build\src\bin\`，编完覆盖到 `build\bin\` 即可。
日志在 `build\logs\_build_q8.log`，最后一行 `DONE` 就成功了。

> 换过目录（甚至只是改名）之后第一次编译是**全量**：CMake 把绝对路径写进 cache，
> `_cache_guard.bat` 会检测到并清掉重配。这是 CMake 的性质，不是配置问题。

---

## 六、配置在哪

| 文件 | 放什么 |
|---|---|
| `llmserver.bat` | 引擎认识的普通参数：模型、端口、上下文 `-c`、批大小 `-b`、`-ngl`、`-fa`、投机类型、`--slot-save-path` |
| `features.ini` | **这个 fork 才有的开关**，引擎自己读 |

`features.ini` 里值得知道的几条：

- `kv-vram-cells` —— KV 有多少留在显存，超出的读主机内存。**必须 ≤ `CTX`**，默认 `32768`。
- `GGML_CUDA_FA_SPARSE` —— 主机尾部按页稀疏读取，**只在 >100K 上下文才真正生效**。
- `GGML_CUDA_FA_CHUNK` —— 深上下文的 prefill 按块跑，显存按块算而不是按上下文算。
- `kq-mask-packed` —— **默认注释掉**。128K 下它的 f16 展开会吃掉 40% prefill（实测 265 vs 430 t/s），
  代价是 decode 差约 5%。
- `GGML_CUDA_FA_SHORTQ_MMA = 0` —— **别开**，实测输出会变成重复乱码。

命令行显式给的参数**优先于** `features.ini`。

---

## 七、几个会踩的坑

1. **改了 `.bat` 却报一堆 `'xxx' is not recognized`** → 文件被存成了 **LF** 换行。
   cmd 的 `^` 续行只认 **CRLF**；含中文注释也建议转成 ASCII，cmd(936) 下会错位吃掉下一行。
2. **速度掉一个数量级** → KV 溢出显存。`CTK/CTV=q4_0` 配 `CTX=131072` 是一组；
   改成 `q8_0` 必须把 `CTX` 减半。
3. **输出变成重复乱码** → 检查 `GGML_CUDA_FA_SHORTQ_MMA` 是否被打开（默认 0 是对的）。
4. **端口被占** → 启动脚本会先 `taskkill` 掉旧的 `llama-server.exe`（故意为之，否则显存不释放）。
5. **图片请求被拒** → 需要 `weights\mmproj-*.gguf`；不在时启动脚本会自动关掉视觉并打印警告。
6. **重启后想续同一段对话** → 服务端要带 `--slot-save-path`（默认已带），
   `llmchat` 会自动保存/恢复；`clear` 才是重新开始。
7. **编译时所有 CUDA 文件 `FAILED`、日志里却没有任何 `error:`** → nvcc 被进程环境撑死了。
   判定：手跑 `nvcc --use-local-env -c 空文件.cu` 能过；不加参数就 `EXIT=1` 且零输出。
   处理：确认 `ggml/src/ggml-cuda/CMakeLists.txt` 里有
   `add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--use-local-env>")`。
   从 IDE / agent 工具里跑编译时最容易碰上，普通 cmd 窗口不会。
8. **改了 `build/src/CMakeFiles/rules.ninja` 但下次编译又变回去** → 那是生成文件。
   任何 CMake 重新配置都会重写它。要加编译参数改 `CMakeLists.txt`。
9. **改过 CMake 文件后第一次编译要十几年分钟** → 属于正常：编译选项变了，所有 obj 失效。
   中途断掉重跑 `_build_inc.bat` 会从断点继续，不用清干净重来。
10. **链接失败、报 `ggml-cuda.dll` 被占用** → 服务没停。`taskkill /F /IM llama-server.exe`。

---

## 八、出处与许可证

**License: Apache 2.0**

权重是**衍生作品**，不是原创模型。逐层如下（每一层都是 Apache 2.0）：

| 层 | 出处 | 做了什么 |
|---|---|---|
| 1 | `Qwen/Qwen3.8-27B`　Copyright 2026 Alibaba Cloud | 原始模型，27B 混合注意力（约 75% 线性 + 25% 全注意力） |
| 2 | `prism-ml/Ternary-Bonsai-2-27B-gguf`　PrismML | 把矩阵权重转成三值（PTQ1_0） |
| 3 | `ukisai/Swift-Bonsai-2-GGUF`　UkisAI | reasoning-efficiency 微调 |
| 4 | MTP 草稿头，来自 `ProCreations` | 见包内 gguf 的 `graft.donor.name` |
| 5 | **本项目** | 合并第 3、4 层并按精度重打包，**未重新训练** |

引擎（`code\`，fork of `PrismML-Eng/llama.cpp`）是 **MIT**。

**必填署名：** *"Created using Bonsai by Prism ML."*

完整声明见包根 `LICENSE`（Apache 2.0 全文）与 `NOTICE`（逐层版权与改动说明）。
