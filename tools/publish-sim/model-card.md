---
language: [zh, en]
base_model:
  - Qwen/Qwen3.8-27B
  - prism-ml/Ternary-Bonsai-2-27B-gguf
  - ukisai/Swift-Bonsai-2-GGUF
library_name: llama.cpp
pipeline_tag: text-generation
license: apache-2.0
tags: [ternary, PTQ1_0, quantization, GGUF, MTP, speculative-decoding, llama.cpp, local-inference, 0token]
---

# Qwen3.8-27B-Flash-MTP — 27B 三值 + MTP：8 GB 笔记本显卡跑到 70 t/s

**让普通用户进入 0 Token 时代。**

不需要租 API、不需要云、不需要买卡。一台 **8 GB 显存的普通笔记本**，在自己家里跑 27B 级推理模型：

| 模式 | 实测速度 |
|---|---|
| **关思考** | **70 t/s** |
| **编辑模式** | **100–200 t/s** |

作为对照：同一台机器上，LM Studio 跑 Qwen3.8-27B 的 4-bit 量化，**大约 1–2 t/s**。

![70 t/s on an 8 GB laptop](https://cdn-uploads.huggingface.co/production/uploads/6aafcc4fc71ed6fa0c2eb60a/8GGwBjrAhAOFto6HztEpI.png)

*上图：8 GB 普通笔记本上的实测输出速度。*

---

## 为什么能做到

27B 从 ~54 GB（FP16）压到 **5.79 GB**，靠的是四件事叠加：

| 技术 | 效果 | 来源 |
|---|---|---|
| **三值量化 + 配套 CUDA 内核** | 27B 压到 5.79 GB，单卡可跑 | PrismML Bonsai-2 主干 + 本 fork 的内核 |
| **CUDA Graph** | 提速约 1.5× | llama.cpp / ggml |
| **MTP 推测解码，深度 3** | 一次前向多出 3 个 token | 本 fork（草稿头内嵌在 gguf 里，`blk.64.nextn`） |
| **起草词表 248k → 24k** | 起草一步只读 25.6 MB，**省 90% 字节** | **我们自己的优化** |

最后一条是关键：MTP 草稿头的 `output.weight` 是全词表 `[5120, 248320]`（265 MB），每个起草步都要整读一遍。
把起草打分限制在自己语料真正用到的 24,000 行上，每步只读 ~26 MB。
**目标模型验证时仍然用全词表，所以输出分布一个 bit 都不变 —— 只省字节，不伤质量**（实测接受率持平）。

---

## 三值量化没有伤推理能力

三值化压缩的是**记忆的精度**，不是推理能力 —— 后者可以通过注入更丰富的知识库记忆和图像记忆恢复。

![象棋棋盘布局](https://cdn-uploads.huggingface.co/production/uploads/6aafcc4fc71ed6fa0c2eb60a/-qSeUbT_Hyiyrv6Ev3R0O.png)

*上图：开 low 思考档，模型生成的中国象棋棋盘布局。*

---

## 完整复现方案

跑起来要三样东西：**脚本和配置**、**引擎二进制**、**权重**。权重就是这个仓库。
脚本和二进制在 GitHub —— 刻意分成两个仓，因为脚本得能被人读、被人改：

| 拿什么 | 在哪 |
|---|---|
| 启动脚本 `llmserver.bat` / `.sh`、**`features.ini`**、命令行客户端 | 👉 **[github.com/zgsndzq/qwen38-27b-flash-mtp](https://github.com/zgsndzq/qwen38-27b-flash-mtp)** |
| 预编译 Windows 运行时（含 CUDA 运行时 dll，**不用装 CUDA Toolkit**） | [Releases](https://github.com/zgsndzq/llama.cpp-qwen38-flash/releases) |
| 引擎源码 + 构建脚本（想自己编才要） | [github.com/zgsndzq/llama.cpp-qwen38-flash](https://github.com/zgsndzq/llama.cpp-qwen38-flash) |

⚠️ **`features.ini` 不在引擎源码仓里，也不是编译进二进制的。** 引擎没有它的内置默认路径，
它是启动时用 `--features-file features.ini` 读进来的。少了它，自己编出来的二进制跑的是
**上游行为**，不是我们的调校（比如 `spec-draft-vocab-map` 那条 24k 起草短名单、`kv-vram-cells`
的 KV 分箱、`spec-draft-sampling = probabilistic`）。

最快的一条路：

```bat
git clone https://github.com/zgsndzq/qwen38-27b-flash-mtp
cd qwen38-27b-flash-mtp
:: 把 Release 的二进制包解压到当前目录，把本仓库两个 gguf 放进 weights\
llmserver.bat
```

复现要点（细节见 runtime 仓的 README）：

- 采样 `--temp 0.6 --top-k 20 --top-p 0.95`
- KV 缓存用 `q4_0`，长上下文靠 tiered KV（`kv-vram-cells`）把超出的部分放 pinned host 内存
- 自投机必须带 `--spec-type draft-mtp-adaptive`，否则 MTP 草稿头不参与起草，速度掉一半以上
- 起草短名单 `--spec-draft-vocab-map maps/shortlist_24k_mix.vocab.txt`

---

## 文件

| 文件 | 大小 | 说明 |
|---|---|---|
| `Qwen3.8-27B-Flash-MTP.gguf` | 5.79 GB | **主模型**。三值主干 + 内嵌 MTP 草稿头（866 张量），嵌入/输出层用 q4_K_M 保住精度 |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 0.59 GB | 视觉投影器，要读图才需要 |

> sha256 见本仓库 `SHA256SUMS`。

**只有这两个。** 我们本地开发期间试过的其他变体（不同 MTP donor、不同嵌入精度、以及一个没有
MTP 头的版本）都**不发布** —— 免得下载者拿到一个"看起来一样但用不了自投机"的文件。

---

## 怎么跑

```bat
llama-server.exe ^
  -m Qwen3.8-27B-Flash-MTP.gguf ^
  --host 127.0.0.1 --port 18080 --parallel 1 ^
  -c 65536 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 ^
  --spec-type ngram-mod,draft-mtp-adaptive --spec-draft-n-max 4
```

- **必须用本 fork 的引擎**。`PTQ1_0` 张量类型和 MTP 草稿上下文都是这个 fork 才有的，
  官方 llama.cpp 加载会直接报 unknown tensor type。
- 视觉：加 `--mmproj mmproj-Qwen3.8-27B-Q8_0.gguf --image-min-tokens 1024`。
- 8 GB 卡建议 `-c 65536`；6 GB 卡降到 32768。

---

## 出处与许可

**License: Apache 2.0**

本仓库是**衍生作品**，不是原创模型。完整链条（逐层均为 Apache 2.0）：

| 层 | 出处 | 做了什么 |
|---|---|---|
| 1 | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)　Copyright 2026 Alibaba Cloud | 原始模型（27B 混合注意力，约 75% 线性 + 25% 全注意力） |
| 2 | [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)　PrismML | 矩阵权重转三值（PTQ1_0，ternary g128 + 块内 Hadamard 旋转 + FP16 组缩放） |
| 3 | [`ukisai/Swift-Bonsai-2-GGUF`](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF)　UkisAI | reasoning-efficiency 微调（官方报告思考 token 减约 39.8%，基准分持平） |
| 4 | MTP 草稿头，来自 `ProCreations` | 本仓库 gguf 的 `graft.donor.name` 指向它 |
| 5 | **本仓库 + `zgsndzq/llama.cpp-qwen38-flash`** | 合并第 3、4 层为一个 GGUF，并按精度重新打包。**未重新训练**；速度优化与起草短名单是本项目做的 |

引擎侧（`zgsndzq/llama.cpp-qwen38-flash`，fork of `PrismML-Eng/llama.cpp`）为 **MIT**。

**必填署名：** *"Created using Bonsai by Prism ML."*

使用本仓库需同时遵守上述各层的许可条款。完整声明见 `NOTICE` 与 `LICENSE`。

---

## 引用

```bibtex
@misc{swift-bonsai-2,
  title  = {Qwen3.8-27B-Flash-MTP: 27B ternary (PTQ1\_0) + MTP, 70 tok/s on 8 GB},
  note   = {Derived from Qwen3.8-27B via PrismML Ternary Bonsai 2 and UkisAI Swift Bonsai 2},
  year   = {2026}
}
```
