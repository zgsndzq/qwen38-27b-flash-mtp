# AGENTS.md — 动这个仓库之前必须知道的事

写给接手这块代码的 AI 代理和人。详细背景在 `README.md`（编译/运行）、
`docs/negative-results.md`（试过且没用）、`docs/positive-results.md`（有效且已启用）。

## 提交纪律：没跑过的东西不许 push

| 改动 | commit | push 到 origin |
|---|---|---|
| 文档 / 注释 / README | 可以 | 可以 |
| 代码，只编译过 | 可以 | **不行** |
| 代码 + 实际启动并跑通 / 量过 | 可以 | 可以 |

**编译通过不等于能跑。** 反例：q3_0 KV 类型做完了、`BUILD_EXIT=0`、31 个模板实例、
dll 从 329 MB 涨到 374 MB，看着完全"完成"——但服务一次都没起过，就被推上远端了。
远端只放确定能跑的状态；未验证的留在工作区或本地 commit。

用户说"提交并推送"就是**全部推送**。哪部分没实测，在 commit message 和回复里注一句，
别把提交扣在本地等第二次授权——把话再说一遍比推错更烦。

## 编译

- **nvcc 必须带 `--use-local-env`**，已固化在 `code/ggml/src/ggml-cuda/CMakeLists.txt`。
  缺了它的症状很隐蔽：ninja 报一堆 `FAILED`，日志里**一条 `error:` 都没有**，
  连没改过的 CUDA 文件也"失败"。原因是启动进程的环境变量太多，nvcc 的子进程
  （cudafe++/cicc/ptxas）起不来。
- **不要手改 `build/src/CMakeFiles/rules.ninja`**：cmake 一重配就覆盖。踩过两次。
- 改过任何 `CMakeLists.txt` / `*.cmake` 之后第一次是**全量**（500+ 目标，10–15 min）。
  中途超时不用重来，重跑 `_build_inc.bat` 会续。
- ninja **不追踪 `.cuh` 依赖**：改了头文件要 `touch` 对应的 `.cu`，否则是空编译。

## 换行符

仓库根有 `.gitattributes`：`*.bat *.cmd *.ps1 *.ini` 一律 `eol=crlf`
（**工作区 CRLF，HEAD 里存 LF**）。其余源码工作区 EOL 必须和 HEAD 一致，
否则 git 按二进制比，diff 变成整个文件重写（`ggml-cuda.cu` 8 行改动显示 14841 行）。

提交前自检：`git diff --numstat | awk '{a+=$1;d+=$2} END{print a,d}'`，
数字明显大于预期就是 EOL 污染。

## 起不来的时候

**先开 `-lv 4`。** 默认 verbosity 的日志会停在 `initializing, n_slots=1` 就没了，
看着像 KV 分配失败，实际可能崩在后面十几步。**日志最后一行之后就是崩溃点。**

已知的坑：`data/ngram-mod.cache` 的 `save()` 写出的文件 `load()` 读不了
（新生成的一样崩），所以 `spec-ngram-mod-cache` 在 `features.ini` 里是注释掉的。
表现为在 ngram 表初始化之后访问违例（`0xC0000005`）。

## 测速

- 必须 `BENCH=1`（切掉 ngram）或充分预热同一 warm-up，否则数字不可比。
- 短 prompt + 非流式请求会高估收益。

## 配置的单一来源

`llmserver.bat` 管所有**数字**（`CTX`/`CELLS`/`CTK`/`CTV`/`REBUDGET`…），
`features.ini` 只管**开关**。`CELLS` 由 `CTK`/`CTV` 推导、`GGML_CUDA_FA_CHUNK` 跟着
`CELLS`，三者永远同值——chunk pool 由 `FA_CHUNK` 定尺，两个值不等的话，
"KV 还在显存"和"开始分块"的边界会落在两个不同的 cell 上，收益白扔。

## 不要做的事

- 不要为了"验证一下"就自己起服务、发推理请求。改完配置停手，让用户起。
- 不要把 40 万行以外的生成物（`build/`、`weights/`、`data/`、`logs/`）放进 git。
- 不要在没有 `mock/` + `bench/` 数据支撑的情况下重写 mat-vec 内核；
  链式加宽已经用模拟证明只会 +0.4%（`docs/negative-results.md` 1.4 / 1.13）。
