# AGENTS.md — 动这个仓库之前必须知道的事

细节在 `README.md`（编译/运行）、`docs/negative-results.md`（试过没用）、
`docs/positive-results.md`（有效）。这里只写容易做错的地方。

## 提交

- 不要擅自 `git add/commit`，用户说提交才提交。
- 用户说“提交并推送”就是**全部推送**。哪部分没实测，在 commit message 和回复里注一句，
  别把提交扣在本地等第二次授权。
- 提交前看 `git diff --numstat`，数字明显大于预期就是 EOL 污染。

## 配置的单一来源

- **数字在 `server.json`**，`llmserver.bat` / `.sh` 只调 `run_server.py`；开关在 `features.ini`。
- `GGML_CUDA_FA_CHUNK` 必须等于 `--kv-vram-cells`，换的时候连 KV 类型一起换。两者不等，
  深 context 会掉进 f16 慢路径（实测 6 倍）。
- 服务端只定能力边界，不改客户端下发的 `max_tokens` / `temp`。

## 编译

- **nvcc 必须 `--use-local-env`**，已固化在 `ggml/src/ggml-cuda/CMakeLists.txt`；缺了会全
  CUDA 文件 FAILED 而日志一条 error 都没有。不要手改 `rules.ninja`，cmake 重配会覆盖。
- 改过 `CMakeLists.txt` / `*.cmake` 后第一次是全量（10–15 min），中途断了重跑会续。
- ninja 不追踪 `.cuh` 依赖：改了头文件要 `touch` 对应的 `.cu`。

## 换行符

`.gitattributes` 里 `*.bat *.cmd *.ps1 *.ini` 是 CRLF（HEAD 存 LF）；其余源码工作区 EOL
必须和 HEAD 一致，否则 diff 会变成整个文件重写。

## 起不来的时候

先加 `-lv 4`。日志停在 `initializing, n_slots=1` 不代表 KV 分配失败，**最后一行之后才是
崩溃点**。崩在 `failed to remove sequence` 是 GDN 记忆不能回滚，不是显存。

## 测速

必须 `BENCH=1`（切掉 ngram）或充分预热，否则冷热不可比。

## 不要做的

- 不要为了“验证一下”自己起服务、发推理请求，改完停手。
- 不要把 `build/`、`weights/`、`data/`、`logs/` 放进 git。
- 不要在没有 `mock/` + `bench/` 数据的情况下重写 mat-vec 内核（链式加宽已证明只 +0.4%）。
