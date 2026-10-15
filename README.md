# ada-ternary — Ternary-Bonsai-2-27B 运行包

三值量化（`PTQ1_0`）的 Bonsai-2 27B，自带 **MTP 自投机草稿头**。
权重 5.8 GB，**8 GB 显存的消费级卡就能跑**，实测 60–80 t/s。

引擎是对 `PrismML-Eng/llama.cpp` 的 fork —— 官方 llama.cpp **跑不了这个模型**
（`PTQ1_0` 张量类型和 MTP 草稿上下文都是这个 fork 才有的）。

> English: see `README.en.md`

---

## 一、三处下载

这套东西不是一个压缩包，分三处：

| 从哪 | 拿什么 | 多大 | 必须吗 |
|---|---|---|---|
| **Hugging Face** → `Ternary-Bonsai-2-27B-PTQ1_0-GGUF` | 权重 `*.gguf` | 5.8 GB（单模型） | **必须** |
| **GitHub Releases** → `ada-ternary-win64-sm89-<commit>.zip` | 预编译二进制 + CUDA 运行时 + 本包 | 568 MB | 不想自己编就要 |
| **GitHub 源码仓** → `llama.cpp-ada-ternary` | 引擎源码 | 350 MB | 想自己编才要 |

**要求的引擎版本：`b3e7c012c` 或更新的 `sync-up-*` 分支。** 版本不对会加载失败。

---

## 二、怎么摆

解压 Release 的 zip，得到一个文件夹，把权重放进去：

```
ada-ternary\
├── build\
│   ├── bin\            ← llama-server.exe + 全部 dll（zip 自带）
│   └── cuda\bin\       ← CUDA 运行时（zip 自带，**不用装 CUDA Toolkit**）
├── weights\            ← ★ 把 gguf 放这里，自己新建
│   ├── Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf     主模型
│   └── mmproj-Qwen3.8-27B-Q8_0.gguf                视觉，要读图才需要
├── maps\               ← 草稿短名单（自带，别删）
├── llmchat\            ← 命令行客户端（可选）
├── features.ini        ← fork 自己的开关都在这里
└── llmserver.bat       ← 双击启动
```

`weights\` 这个目录名不能改（启动脚本按 `%~dp0weights\` 找），
模型文件名要和 `llmserver.bat` 里的 `MODEL=` 一致。

---

## 三、跑起来

**不需要编译**，双击 `llmserver.bat`。看到这两行就成了：

```
load_model: loaded multimodal model, '...\weights\mmproj-...gguf'
llama_server: listening on http://127.0.0.1:18080
```

然后打 API：

```bash
curl http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```

或者用附带的客户端 `llmchat\llmchat.bat`。

**前置条件**：NVIDIA 驱动。显存 **8 GB** 起；6 GB 要把 `llmserver.bat` 里的
`CTX` 从 `65536` 调到 `32768`（同时 `features.ini` 的 `kv-vram-cells` 也要 ≤ 它）。

---

## 四、想自己编译

环境不一样、或者想改引擎的时候：

```bat
git clone https://github.com/professorpalmer/llama.cpp-ada-ternary.git
cd llama.cpp-ada-ternary
copy build.local.example.bat build.local.bat
notepad build.local.bat        :: 填 6 个值
_build_q8.bat
```

需要装：**VS2022（MSVC）+ CUDA Toolkit + CMake + Ninja**。

`build.local.bat` 要填的 6 个值，模板里都写了注释：

| 变量 | 填什么 | 注意 |
|---|---|---|
| `VCVARS` | VS2022 的 `vcvars64.bat` | 必须 —— 没有它 nvcc 找不到 `cl.exe` |
| `CMAKE` / `NINJA` | 两个工具的可执行文件 | |
| `NVCC` | CUDA Toolkit 里的 `nvcc.exe` | |
| `CUDA_ARCH` | **你的显卡算力** | ⚠️ 唯一会让"编译成功但跑不起来"的值：4060/4070 = `89`，3060 = `86`，4090 = `89` |
| `JOBS` | 并行度，笔记本填 `4` | 太大 nvcc 会吃爆内存 |

产物落在 `..\build\src\bin\`，编完覆盖到 `build\bin\` 即可。
日志在 `..\build\logs\_build_q8.log`，最后一行 `DONE` 就成功了。

> 忘了建 `build.local.bat` **不会静默出错** —— 脚本会写 `MISSING build.local.bat`
> 然后以退出码 9 结束，不会在别的地方乱建目录。

---

## 五、配置在哪

分两个文件，职责不重叠：

| 文件 | 放什么 |
|---|---|
| `llmserver.bat` | 引擎认识的普通参数：模型、端口、上下文 `-c`、批大小 `-b`、`-ngl`、`-fa`、投机类型 |
| `features.ini` | **这个 fork 才有的开关**，引擎自己读（`--features-file`） |

`features.ini` 里值得知道的两条：

- `kv-vram-cells` —— KV 有多少留在显存，超出的读 pinned RAM。**必须 ≤ `CTX`**，默认 `32768`。
- `GGML_CUDA_FA_SHORTQ_MMA = 0` —— 关掉上游那条"量化 V + 短 query 转 f16"的路径。
  **别开**：实测会把输出变成重复的乱码。

命令行显式给的参数**优先于** `features.ini`。

---

## 六、几个会踩的坑

1. **速度掉一个数量级** → 十有八九是 KV 溢出显存。`CTK/CTV=q4_0` 配 `CTX=65536` 是一组；
   改成 `q8_0` 就必须把 `CTX` 减半，否则每 cell 字节翻倍、装不下。
2. **输出变成重复乱码** → 检查 `GGML_CUDA_FA_SHORTQ_MMA` 有没有被打开（默认 0 是对的）。
3. **改了 `.bat` 却报一堆 `'xxx' is not recognized`** → 文件被存成了 **LF** 换行。
   cmd 需要 **CRLF**，而且 bat 里的注释**必须是纯 ASCII**（中文注释在 cmd(936) 下会错位吃掉下一行）。
4. **端口被占** → 启动脚本会先 `taskkill` 掉旧的 `llama-server.exe`（故意为之，否则显存不放）。
   想同时跑两个实例，得先把那两行注释掉。
5. **图片请求被拒** → 需要 `weights\mmproj-*.gguf`。文件不在时启动脚本会自动关掉视觉并打印一条警告。

---

## 七、出处与许可证

**待补 —— 上传前必须确认。**

这是**衍生作品**：主干来自 `Swift-Bonsai-2` 27B（架构 `qwen35`），我们做的是
**三值量化 + MTP 头转换，没有重新训练**。许可证跟随 base model，不额外施加限制。

详见权重仓库的 model card。
