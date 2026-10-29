# AGENTS.md

Qwen3.8-27B-Flash-MTP 在 8 GB 显卡上的本地推理分支（llama.cpp fork）。

## 跑起来

- 服务端：`llmserver.bat`（Windows）/ `llmserver.sh`，两者都只调 `run_server.py`。
  **数字在 `server.json`**，功能开关在 `features.ini` 的 `[qwen]` 节。
- 客户端：`llmchat/llmchat.bat`，配置在 `llmchat/config.json`。
- 编译：`code/_build_inc.bat` 增量，产物进 `build/src/bin`；改过 cmake 文件后首次是全量。

## 分支口径

- 只有**我们优化过、能提速**的代码才归我们的特性与配置：代码里默认开，`features.ini` 里关。
- 没优化过的上游代码保持上游现状，不要顺手改默认值——我们的改动只服务提速。
- 开关"读了但没效果" = 代码在合并里丢了，比编译不过更危险。

## 改的时候要注意

- `GGML_CUDA_FA_CHUNK` 必须等于 `--kv-vram-cells`，换的时候连 KV 类型一起换。
- nvcc 要 `--use-local-env`，已固化在 `ggml-cuda/CMakeLists.txt`。
- `.bat/.cmd/.ps1/.ini` 是 CRLF，其余源码 LF，工作区与 HEAD 保持一致。
- 起不来先加 `-lv 4`：日志最后一行之后才是崩溃点，`failed to remove sequence` 是 GDN 记忆不能回滚。
