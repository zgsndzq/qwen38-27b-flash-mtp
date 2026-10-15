---
language: [zh, en]
base_model: Swift-Bonsai-2-27B          # TODO 确认上游仓库与许可证
library_name: llama.cpp
pipeline_tag: text-generation
tags: [ternary, PTQ1_0, quantization, GGUF, MTP, speculative-decoding, llama.cpp]
---

# Ternary-Bonsai-2-27B (PTQ1_0 + MTP)

三值量化（1.58-bit class，GGUF `PTQ1_0`）的 Bonsai-2 27B，并自带 **MTP 自投机草稿头**，
用在一块消费级 8 GB 显存卡上就能跑（RTX 4060 Laptop 实测 ~60-80 t/s）。

权重体积从 27B 的 fp16（约 54 GB）压到 **5.8 GB**，是"能在本地跑"与"不能"的分界线。

---

## 需要哪个引擎

**这些权重不是官方 llama.cpp 能跑的。** `PTQ1_0` 这个张量类型和 MTP 草稿上下文都是
本 fork 才有的，官方引擎会在加载时直接报 unknown tensor type。

| | |
|---|---|
| 引擎仓库 | `professorpalmer/llama.cpp-ada-ternary`（fork of `PrismML-Eng/llama.cpp`） |
| **要求的 commit** | **`{{ENGINE_COMMIT}}`**（或更新的 `sync-up-*` 分支） |
| 分支 | `sync-up-1013-stage1` |

引擎侧的构建说明见该仓库 README；预编译二进制在同一仓库的 **Releases** 页。

---

## 文件

| 文件 | 大小 | 说明 |
|---|---|---|
| `Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf` | 5.79 GB | **推荐**。三值主干 + MTP 头，嵌入/输出层用 q4_K_M 保住精度 |
| `Swift-Bonsai-2-PTQ1_0-mtp.gguf` | 5.79 GB | 同上，全三值，最小 |
| `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf` | 5.79 GB | 早期命名，等价 |
| `Swift-Bonsai-2-PTQ1_0-mtp-pro.gguf` | 5.96 GB | pro 配置，多留了几层高精度 |
| `Swift-Bonsai-2-PTQ1_0.gguf` | 5.54 GB | **无 MTP 头**，不支持自投机 |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 0.59 GB | 视觉投影器（可选，要读图才需要） |

> sha256 见本仓库 `SHA256SUMS`。

---

## 怎么跑

```bat
llama-server.exe ^
  -m Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf ^
  --host 127.0.0.1 --port 18080 --parallel 1 ^
  -c 65536 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 ^
  --spec-type ngram-mod,draft-mtp-adaptive --spec-draft-n-max 4
```

要点：

- **`--spec-type` 必须带上 `draft-mtp-adaptive`**，否则 MTP 头不会被用作草稿，速度掉一半以上。
- `-ctk/-ctv q4_0`：KV 用 q4_0，64k 窗口才能在 8 GB 卡上完整驻留显存。
  改成 `q8_0` 时窗口要相应减半，否则 KV 溢出到内存、速度掉一个数量级。
- 视觉：加 `--mmproj mmproj-Qwen3.8-27B-Q8_0.gguf`，建议同时给 `--image-min-tokens 1024`。
- 完整配方（含 fork 自己的开关文件 `features.ini`）见引擎仓库的 runtime 说明。

---

## 量化方式

主干权重为 **PTQ1_0**：三值 {-1, 0, +1} 加一组 per-group 缩放因子，约 1.7 bpw。
非三值的部分（嵌入、输出投影、MTP 头）按文件名后缀保留更高精度（`q4km` / `pro`）。
`down_proj` 与注意力的一部分保持 fp16，用来压住三值化带来的质量损失。

MTP 头是从主干数据里一并转换出来的，不是后加的独立 drafter —— 它和主干共享词表与
隐层，所以草稿的接受率高、显存不额外涨。

---

## 评测

本仓库**不附带**完整 benchmark。我们自己测过的两个点，供参考（RTX 4060 Laptop 8 GB，
q4_0 KV，64k 上下文）：

| 场景 | 结果 |
|---|---|
| 生成速度（代码/结构化文本，ngram + MTP 命中） | 60-80 t/s |
| 生成速度（自由散文，草稿命中率低） | 约 40 t/s |
| 视觉：读 777x858 中国象棋棋盘，32 个棋子 | 28/32 全对（错的两个是冷僻字 `俥`/`傌`，被读成同一个形近字） |

---

## 许可证与出处

**待补 — 必须在上传前确认。**

本仓库是**衍生作品**：主干来自 `Swift-Bonsai-2` 27B（`general.name = Swift-Bonsai-2-merged-PTQ1_0`，
架构 `qwen35`），我们只做了三值量化与 MTP 头的转换，**没有重新训练**。

因此：

- 本仓库**不能**声明为原创模型，许可证**跟随 base model**，不额外施加限制
- 使用者需同时遵守 base model 的许可条款
- 若 base model 有商用限制，本量化版同样受限

> TODO（上传前必须填）：base model 的 HF 仓库地址、其许可证名称、以及本量化版的产出方署名。

---

## 引用

```bibtex
@misc{ternary-bonsai-2-27b,
  title  = {Ternary-Bonsai-2-27B (PTQ1\_0 + MTP)},
  note   = {Ternary quantisation of Swift-Bonsai-2-27B with an integrated MTP draft head},
  year   = {2026}
}
```
