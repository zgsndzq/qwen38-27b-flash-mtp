# llmchat — 独立对话客户端

`python llm.py chat` 的**独立化版本**：单文件、零项目依赖、把这个文件夹拷到哪都能用。

它**只负责对话**——连上一个已经跑着的 llama-server（比如同包 `..\llmserver.bat` 起的那个），
自己**不加载任何模型**。

## 怎么用

**双击 `llmchat.bat`** 即可。参数都在文件顶部的变量区，跟 `llmserver.bat` 一个风格。

命令行等价写法：

```
cd F:\super\llmchat
python chat.py --temp 0.6 --think on --effort medium
```

## 文件

| 文件 | 作用 |
|---|---|
| `chat.py` | 客户端本体（stdlib only，`prompt_toolkit` 可选）。改行为改这里 |
| `llmchat.bat` | 启动器：填参数 → 调 chat.py。**注释必须纯 ASCII**（原因见文件里那段说明） |
| `sys.txt` | system prompt（UTF-8）。**改这里就能换人设**；没有这个文件、也没配 `SYS_FILE`，就**一条 system 消息都不发**（没有内置兜底） |

## 参数（命令行 / bat 变量对照）

| chat.py 参数 | bat 变量 | 说明 |
|---|---|---|
| `--base-url` | `HOST`+`PORT` | 服务端地址，默认 `http://127.0.0.1:18080` |
| `--temp` | `TEMP` | 采样温度。**留空 = 不下发**，沿用服务端 `--temp` |
| `--think` | `THINK` | `on` / `off` / `server`（= 不下发，沿用服务端 `-rea`） |
| `--effort` | `EFFORT` | `xhigh` / `medium` / `low`，仅思考开启时生效 |
| `--timeout` | `TIMEOUT` | 等待服务端**无数据**的秒数，默认 600 |
| `--max-tokens` | `MAXTOK` | 单轮上限，留空 = 不限制（答到 EOS） |
| `--sys` / `--sys-file` | `SYS_FILE` | system prompt；文件优先 |
| `--hide-thinking` | `EXTRA` | 不打印思考内容（请求照常开思考） |
| `--no-tui` | `EXTRA` | 禁用多行 TUI |

`MODEL_NAME` 只是窗口标题，**不影响行为** —— llama-server 服务的是它自己启动时加载的那个 gguf，
客户端没法选模型。启动时客户端会探测 `/v1/models` 并把服务端实际加载的模型路径打出来。

## 会话内元命令（本地拦截，不发给模型）

| 命令 | 作用 |
|---|---|
| `quit` / `q` / `exit` / `退出` | 退出 |
| `clear` / `cls` / `清空` | 清空上下文（保留 system prompt） |
| `/paste` … 单独一行 `/end` | 非 TTY（管道）下粘贴多行文本 |

## 缓存/槽位的硬规则：前缀必须逐 token 可复现

服务端用 `--slot-save-path` 起的槽位只支持**严格延长**：新请求的 prompt 必须是已存
token 序列的延长，差一个 token 就丢掉整段 KV 全量重算（20K 上下文约 2 分钟）。

前缀由三样东西决定，缺一不可，必须一起存、一起重放：

| 成员 | 作用 | 存在哪 |
|---|---|---|
| system 正文 | 有就放在最前，没有就一条都不发 | 会话文件 `sys_frame` |
| 思考开关 | 模板只有 `enable_thinking` 为真才解析 effort 并注入那段 system | `think_frame` |
| 思考档位 | 同上，字面量进 system 段 | `effort_frame` |

规则：
1. **配什么就用什么**。没有 `--sys` / `--sys-file` / `SYS_FILE` 就不发 system 消息，
   不做任何"内置默认"发挥。
2. **历史里存实际发出的那一版**（含 `<knowledge>`、含图片），事后不改写。
3. 槽位存盘之后那一次 `session_save()` 也**必须**把三帧一起写回；只写 `slot_tokens`
   会把帧覆盖成 null，下次启动就用空帧重放，前缀在 system 段（约第 20 个 token）分叉。
4. **帧变了就是槽位作废，不是把旧帧套回当前配置**。启动时拿当前帧和存档帧比：
   一致才读槽位；不一致就清掉槽位、按**这一次**的配置跑，并明确打印差异
   （`思考开关 关 -> 开`）。上一版强行沿用旧帧，表现是"打开思考却没思考"。
5. **思考块必须回填进历史**（默认 `keep_thinking=True`）。模板看到 assistant 带
   `reasoning_content` 就渲染 `<think>…</think>`，而服务端 KV 里本来就有这些 token；
   客户端丢掉它，prompt 就比 KV 少一段，公共前缀在**第一个带思考的 assistant 轮**断掉，
   后面全部重算。想省上下文用 `--no-keep-thinking`，那是拿缓存换空间。



超时保护的是**心跳**，不是"整体耗时"：预填充（prompt eval）期间服务端**一个字都不发**，
长上下文下 pp 再快也要几十秒到几分钟。所以长提示词报 timeout **不是服务挂了**，
调大 `TIMEOUT` 或 `LLAMA_CHAT_TIMEOUT` 即可。

## 与原 `python llm.py chat` 的关系

参数名**完全一致**（`--sys` / `--base-url` / `--temp` / `--think` / `--effort`），
原来的命令把 `python llm.py chat` 换成 `python chat.py` 就能直接跑。
行为也一致：多轮、SSE 流式逐字打印、思考先出回答后出、`quit`/`clear` 元命令、同一个 `LLAMA_CHAT_TIMEOUT`。

差别只有：想显式沿用服务端时，原版靠"不给 `--think`"，这里多了个显式的 `--think server`（`on`/`off` 用法不变）。

## TUI（多行输入）

装了 `prompt_toolkit` 且是交互终端时自动启用：粘贴大段文档整块提交、**Esc+Enter** 手动换行、↑/↓ 历史。
没装则退化为单行 + `/paste`。想强制单行：`--no-tui`。

## 验证记录（2026-09-22）

连 18080 上正在运行的服务实测（管道喂脚本，非 TTY 路径）：

```
client argv = chat.py --base-url http://127.0.0.1:18080 --temp 0.6 --think on --effort medium --max-tokens 400
rc = 0   elapsed = 4.8s   stderr 空
底座: http://127.0.0.1:18080  temp=0.6  思考=开(medium)
服务端模型: Ternary-Bonsai-2-27B
[思考] ...（思考块正常输出）
[回答] 快速排序是一种通过选取基准元素…的分治算法。
你> 上下文已清空（保留 system prompt，历史 0 条）      <- clear 生效
[回答] 2                                              <- 第二轮问答正常
你> 再见！
```

`llmchat.bat` 单独验证（`cmd /c llmchat.bat`，喂 `quit`）：rc = 0，
正确拾取 `sys.txt`、正确组装 `--think on --effort medium`、`pause` 正常。

## 依赖

- Python 3（只用标准库；`prompt_toolkit` 可选）
- 一个正在运行的 llama-server（例：同包 `..\llmserver.bat`）
