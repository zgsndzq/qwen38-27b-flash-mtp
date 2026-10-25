# AGENTS.md — 动这个仓库之前必须知道的事

**第一优先：用户指令。** 本文与用户当次要求冲突时，一律以用户为准。

细节见 `README.md` 和 `docs/`。

## 提交

- 不要擅自 `git add/commit`。
- 用户说“提交并推送”就全部推送，没实测的在 commit message 里注一句。

## 配置

- 数字在 `server.json`，开关在 `features.ini`；bat / sh 只调 `run_server.py`。
- `GGML_CUDA_FA_CHUNK` 必须等于 `--kv-vram-cells`，换的时候连 KV 类型一起换。
- 服务端不改客户端下发的 `max_tokens` / `temp`。

## 编译

- nvcc 要 `--use-local-env`，已固化在 `ggml-cuda/CMakeLists.txt`。不要手改 `rules.ninja`。
- 改过 cmake 文件后首次是全量，中途断了重跑会续。
- 改了 `.cuh` 要 `touch` 对应的 `.cu`，ninja 不追踪头文件。

## 换行符

`.bat/.cmd/.ps1/.ini` 是 CRLF，其余源码 LF，工作区跟 HEAD 一致；否则 diff 会变成整个文件重写。

## 起不来的时候

先加 `-lv 4`，最后一行之后才是崩溃点。`failed to remove sequence` 是 GDN 记忆不能回滚，不是显存。

## 不要做的

- 不要为了“验证一下”自己起服务、发推理请求。
- 不要把 `build/`、`weights/`、`data/`、`logs/` 放进 git。
