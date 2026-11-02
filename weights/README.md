# weights — 这里放模型权重

**权重不在 git 里**，在 Hugging Face：

### 👉 https://huggingface.co/SuperLogic/Qwen3.8-27B-Flash-MTP-GGUF

下载后放到这个目录，文件名**不要改**（`llmserver.bat` 按名字找）：

| 文件 | 大小 | 必须吗 |
|---|---|---|
| `Qwen3.8-27B-Flash-MTP.gguf` | 5.79 GB | **必须**，主模型 |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 0.59 GB | 要读图片才需要 |

放好之后目录长这样：

```
weights\
├── Qwen3.8-27B-Flash-MTP.gguf
└── mmproj-Qwen3.8-27B-Q8_0.gguf
```

> ⚠️ 目录名 `weights` 不能改，启动脚本按 `%~dp0weights\` 找。
>
> 没有 mmproj 时，启动脚本会自动关掉视觉并打印一条 `[warn]` —— 不会启动失败，
> 只是发图片给模型会被拒。

## 另一条线：3D 场景小模型（`llmserver3d.bat`）

第三方微调，不是 Qwen 官方发布：https://huggingface.co/ccore/Qwen3.5-4B-3D-Coder
（base 是 Qwen3.5-4B，专攻 Three.js 3D 场景生成）。同样按名字放进来，别改文件名：

| 文件 | 大小 | 必须吗 |
|---|---|---|
| `Qwen3.5-4B-3D-Coder-Q8_0.gguf` | 4.17 GB | **必须**，`llmserver3d.bat` 用 |
| `mmproj-Qwen3.5-4B-BF16.gguf` | 0.63 GB | 要读图片才需要，当前未下载 |

这个 GGUF 里 `qwen35.nextn_predict_layers = 0`，即**没有 MTP 头**，所以
`llmserver3d.bat` 不传 `--spec-type`，`features.ini` 那几个 MTP 调参段也套不上。
