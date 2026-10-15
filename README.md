# ada-ternary-runtime — Swift-Bonsai-2-27B 运行包

三值量化（`PTQ1_0`）的 Bonsai-2 27B，自带 **MTP 自投机草稿头**。
权重 5.79 GB，**8 GB 显存的消费级卡就能跑**，实测 **70 t/s**（关思考）、
**100–200 t/s**（编辑模式）。

引擎是我们对 `PrismML-Eng/llama.cpp` 的 fork —— 官方 llama.cpp **跑不了这个模型**
（`PTQ1_0` 张量类型和 MTP 草稿上下文都只存在于这个 fork）。

> English: see `README.en.md`

---

## 一、这个仓是什么

**只有脚本和配置。** 二进制和权重不进 git，各有各的分发渠道：

| 东西 | 从哪拿 | 多大 | 必须吗 |
|---|---|---|---|
| `llmserver.bat` / `llmserver.sh` / `features.ini` / `llmchat\` / `maps\` | **本仓**（`git clone`） | < 1 MB | **必须** |
| 引擎二进制（`build\bin` + `build\cuda`） | [Releases](https://github.com/zgsndzq/ada-ternary/releases) | 566 MB | 不想自己编就要 |
| 权重 `weights\*.gguf` | [Hugging Face](https://huggingface.co/SuperLogic/Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km) | 5.79 GB + 0.59 GB | **必须** |
| 引擎源码（`code\`） | [zgsndzq/ada-ternary](https://github.com/zgsndzq/ada-ternary) | 350 MB | 想自己编才要 |

这样分的原因：**脚本要能提 PR、能看 diff**。以前它们只活在一个 566 MB 的 zip 里，
没人能改、没人能审。

---

## 二、三步跑起来

```bat
git clone https://github.com/zgsndzq/ada-ternary-runtime
cd ada-ternary-runtime
```

1. 把 Release 的二进制 zip 解压到**当前目录** → 得到 `build\bin\` 和 `build\cuda\bin\`
2. 从 Hugging Face 下两个 gguf 放进 `weights\` → 详见 `weights\README.md`
3. **双击 `llmserver.bat`**

看到这两行就成了：

```
load_model: loaded multimodal model, '...\weights\mmproj-...gguf'
llama_server: listening on http://127.0.0.1:18080
```

然后直接打 API：

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```

或者用附带的客户端：双击 `llmchat\llmchat.bat`。

**前置条件**：NVIDIA 驱动。显存 **8 GB** 起；6 GB 要把 `llmserver.bat` 里的 `CTX`
从 `65536` 调到 `32768`，同时 `features.ini` 的 `kv-vram-cells` 也要 ≤ 它。

---

## 三、目录长什么样

```
ada-ternary-runtime\
├── llmserver.bat        ← 双击启动（Windows）
├── llmserver.sh         ← ./llmserver.sh（Linux，参数逐个对齐）
├── features.ini         ← ★ 我们的配置全在这里
├── llmchat\             ← 命令行客户端（可选，纯 Python，跨平台）
├── maps\                ← 草稿短名单，别删（features.ini 引它）
├── code\                ← 【不进 git】引擎源码，git clone 进来
├── build\               ← 【不进 git】二进制，解压 Release 或自己编
└── weights\             ← 【不进 git】权重，从 HF 下
```

`weights\` 和 `maps\` 这两个目录名**不能改**（启动脚本和 `features.ini` 按名字找）。

---

## 四、配置在哪 —— 为什么那个 ini 不能少

分两个文件，职责不重叠：

| 文件 | 放什么 |
|---|---|
| `llmserver.bat` / `llmserver.sh` | 引擎认识的普通参数：模型、端口、上下文 `-c`、批大小 `-b`、`-ngl`、`-fa`、投机类型 |
| `features.ini` | **这个 fork 才有的开关**，引擎自己读（`--features-file`） |

### 三者的关系

```
llmserver.bat / .sh   ──►  命令行参数   ──►  llama-server
                          （上游认得的）

features.ini          ──►  --features-file  ──►  llama-server
                          （只有这个 fork 认得的）
```

**优先级：命令行显式参数 > `features.ini` > 引擎内置默认。**

判断一个开关该写哪儿，就一句话：

> **上游原版 `llama-server` 认不认这个参数？认 → 写 `llmserver`；不认 → 写 `features.ini`。**

对照表（左边是 `llmserver.bat` 里的变量，右边是它变成的命令行）：

| 变量 | 变成 | 类别 |
|---|---|---|
| `MODEL` | `-m weights\<名>.gguf` | 上游 |
| `CTX` | `-c` | 上游 |
| `CTK` / `CTV` | `-ctk` / `-ctv` | 上游 |
| `REBUDGET` | `--reasoning-budget` | 上游 |
| `DRAFT_N` | `--spec-draft-n-max` | 上游 |
| `NMATCH` / `NGMIN` / `NGMAX` / `LOOKUP_N` | `--spec-ngram-mod-n-match` / `-n-min` / `-n-max` / `--spec-lookup-n-max` | 上游 |
| `VISION` / `MMPROJ` | `--mmproj` + `--image-min-tokens` | 上游 |
| — | `--spec-type ngram-mod,draft-mtp-adaptive` | ⚠️ `draft-mtp-adaptive` 是**我们的**（写死在命令行里，因为它必须在 `--spec-type` 里和上游的 `ngram-mod` 并列）|
| — | `--features-file features.ini` | ⚠️ **我们的**：就是这一条把 ini 接进来的 |

反过来，**不在 `llmserver` 里、只在 `features.ini` 里的**：

| ini 键 | 为什么不写进命令行 |
|---|---|
| `spec-draft-sampling` | 上游没有这个参数 |
| `spec-draft-vocab-map` | 上游没有 |
| `kv-vram-cells` | 上游没有 |
| `kq-mask-packed` | 上游没有 |
| `GGML_CUDA_*` / `LLAMA_*` | 这些是**环境变量**，不是命令行参数 |
| `sampler-fast-topk` | 上游没有 |

⚠️ **`features.ini` 不参与编译。** 引擎里没有它的默认路径 —— 它是运行时被
`--features-file features.ini` 读进来的。**少了它，你拿到的是上游行为，不是我们的调校。**

只在 ini 里的几条（举几个例子）：

| 键 | 值 | 引擎默认 | 作用 |
|---|---|---|---|
| `spec-draft-sampling` | `probabilistic` | `greedy` | 草稿按目标温度真采样，只提接受率、不改输出分布 |
| `spec-draft-vocab-map` | `maps/shortlist_24k_mix.vocab.txt` | 无 | MTP 草稿头的 24k 短名单 |
| `kv-vram-cells` | `32768` | 无 | KV 分箱：留在显存多少 cell，超出的读 pinned RAM |
| `GGML_CUDA_FA_SHORTQ_MMA` | `0` | 见下 | 关掉上游"量化 V + 短 query 转 f16"那条路 |
| `LLAMA_CKPT_REUSE` | `1` | 见下 | 复用 prompt checkpoint 缓冲，省约 1.4 GB |

`GGML_CUDA_FA_SHORTQ_MMA = 0` **别开**：实测会把输出变成重复的乱码
（同样一个请求，开=重复一个 token、接受率掉到 0.12，关=干净 319 字符且更快）。

命令行显式给的参数**优先于** `features.ini`。

---

## 五、想自己编

改了引擎、或者显卡算力不是 89 的时候：见 `build\README.md`。
一句话是 `git clone .../ada-ternary code` → 填 `code\build.local.bat` → `code\_build_q8.bat`。

编出来的东西在 `build\src\bin\`，`llmserver.bat` **两个位置都认，优先用你自己编的**，
不需要手工拷贝。

---

## 六、在 Linux 上用

`.bat` 是 Windows 的，但**这套东西本身是跨平台的** —— 真正干活的是一个 `llama-server`
加一串参数。Linux 上对应的是同目录的 **`llmserver.sh`**，参数与 `llmserver.bat` **逐个对齐**：

```bash
git clone https://github.com/zgsndzq/ada-ternary-runtime
cd ada-ternary-runtime

# 没有预编译的 Linux 二进制，必须自己编
git clone https://github.com/zgsndzq/ada-ternary code
cmake -B build/src -S code -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_QUANTS=all \
  -DLLAMA_CURL=OFF
cmake --build build/src -j "$(nproc)" --target llama-server llama-bench llama-quantize

# 权重照样从 HF 下到 weights/（见 weights/README.md）
./llmserver.sh
```

三处和 Windows 不同，其余全部一样：

| | Windows | Linux |
|---|---|---|
| 启动 | 双击 `llmserver.bat` | `chmod +x llmserver.sh` 后 `./llmserver.sh` |
| 杀旧进程 | `taskkill /F /IM llama-server.exe` | `pkill -f llama-server` |
| dll / so 搜索路径 | `PATH` + `build\cuda\bin` | `LD_LIBRARY_PATH`（脚本已设） |
| 编译脚本 | `code\_build_q8.bat`（填 `build.local.bat`） | 直接 `cmake`（见上面的命令）。**没有** `.sh` 版构建脚本 |

⚠️ **Linux 上没有 `GGML_CUDA_FA_QUANTS=all` 会静默掉速。**
CMake 默认的 FA kernel 列表不含这个模型用的量化 K/V 组合，
运行时找不到就**默默把 K/V 转成 f16**（代码里那条 `(slow)` 路径）—— 不报错，只是变慢、数值也变。
`-DGGML_CUDA_FA_QUANTS=all` 是把它堵掉的唯一办法。

⚠️ **`CUDA_ARCH` 别让 CMake 猜。** 填错算力**能编过、跑起来才炸**：
Ada（4060 / 4070 / 4090）= `89`，Ampere（3060 / 3080）= `86`。

⚠️ **`-t 8` 是笔记本核数，Linux 上按机器改**（比如 `-t $(nproc)`）。

> `llmserver.sh` 只做了 `cd` 到脚本所在目录，除此之外没干别的事 ——
> 直接照着它手敲那一串参数、把 `--features-file features.ini` 换成绝对路径，也一样跑得起来。
> **ini 和脚本是可以分开的**，前提是 `features.ini` 里 `spec-draft-vocab-map` 那个
> `maps/...` 相对路径你得跟着改成对的。

---

## 七、几个会踩的坑

1. **速度掉一个数量级** → 十有八九是 KV 溢出显存。`CTK/CTV=q4_0` 配 `CTX=65536` 是一组；
   改成 `q8_0` 就必须把 `CTX` 减半，否则每 cell 字节翻倍、装不下。
2. **输出变成重复乱码** → 检查 `GGML_CUDA_FA_SHORTQ_MMA` 有没有被打开（默认 0 是对的）。
3. **改了 `.bat` 却报一堆 `'xxx' is not recognized`** → 文件被存成了 **LF** 换行。
   cmd 需要 **CRLF**，而且 bat 里的注释**必须是纯 ASCII**（中文注释在 cmd(936) 下会错位吃掉下一行）。
4. **端口被占** → 启动脚本会先 `taskkill` 掉旧的 `llama-server.exe`（故意为之，否则显存不放）。
   想同时跑两个实例，得先把那两行注释掉。
5. **图片请求被拒** → 需要 `weights\mmproj-*.gguf`。文件不在时启动脚本会自动关掉视觉并打一条警告。
6. **`llama-server.exe not found`** → `build\` 是空的。要么解压 Release 的 zip，要么自己编。

---

## 八、出处与许可证

**License: Apache 2.0**

权重是**衍生作品**，不是原创模型。逐层如下（每一层都是 Apache 2.0）：

| 层 | 出处 | 做了什么 |
|---|---|---|
| 1 | `Qwen/Qwen3.8-27B`　Copyright 2026 Alibaba Cloud | 原始模型，27B 混合注意力（约 75% 线性 + 25% 全注意力） |
| 2 | `prism-ml/Ternary-Bonsai-2-27B-gguf`　PrismML | 把矩阵权重转成三值（PTQ1_0） |
| 3 | `ukisai/Swift-Bonsai-2-GGUF`　UkisAI | reasoning-efficiency 微调（思考 token 减约 39.8%，基准分持平） |
| 4 | MTP 草稿头，来自 `ProCreations` | 见包内 gguf 的 `graft.donor.name` |
| 5 | **本项目** | 把第 3 层与第 4 层合并成一个 GGUF 并按精度重新打包，**未重新训练** |

引擎（`code\`，fork of `PrismML-Eng/llama.cpp`）是 **MIT**。

**必填署名：** *"Created using Bonsai by Prism ML."*

完整声明见根目录 `LICENSE`（Apache 2.0 全文）与 `NOTICE`（逐层版权与改动说明）。
