# weights — 这里放模型权重

**权重不在 git 里**，在 Hugging Face：

### 👉 https://huggingface.co/SuperLogic/Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km

下载后放到这个目录，文件名**不要改**（`llmserver.bat` 按名字找）：

| 文件 | 大小 | 必须吗 |
|---|---|---|
| `Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf` | 5.79 GB | **必须**，主模型 |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 0.59 GB | 要读图片才需要 |

放好之后目录长这样：

```
weights\
├── Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf
└── mmproj-Qwen3.8-27B-Q8_0.gguf
```

> ⚠️ 目录名 `weights` 不能改，启动脚本按 `%~dp0weights\` 找。
>
> 没有 mmproj 时，启动脚本会自动关掉视觉并打印一条 `[warn]` —— 不会启动失败，
> 只是发图片给模型会被拒。
