# llmchat

小对话客户端，用来测已经在跑的 llama-server（先开 `..\llmserver.bat`）。
只用 Python 标准库，不加载模型。

## 用法

双击 `llmchat.bat` 即可。**配置全在同目录的 `config.json`**，改完直接生效；
命令行给的参数会覆盖它，例如：

```
python chat.py --think server --temp 0.2
```

要换一套配置就多放一个文件。双击用时改 `llmchat.bat` 里的 `CONFIG` 那一行
（`set "CONFIG=config.bonsai.json"`）；命令行给文件名则优先：

```
llmchat.bat config.bonsai.json
llmchat.bat config.bonsai.json --temp 0.2
```

`CONFIG` 默认是 `config.json`。

配置文件的键和参数同名（下划线写法）：

```json
{
  "base_url": "http://127.0.0.1:18080",
  "think": "on",
  "effort": "low",
  "temp": 0.6,
  "max_tokens": null,
  "timeout": 600,
  "sys_file": "",
  "keep_thinking": true
}
```

## 主要参数

| 参数 | 取值 / 默认 | 作用 |
|---|---|---|
| `think` | `on` / `off` / `server` | 思考开关；`server` = 不发，用服务端 `-rea`。也认 `true`/`false`/`1`/`0`/`yes`/`no` |
| `effort` | `low`；或 `xhigh` `medium`；留空 | 思考强度；留空 = 服务端默认 |
| `temp` | 数字；`null` | 采样温度；`null` = 用服务端 `--temp` |
| `max_tokens` | 数字；`null` | 单轮回答上限；`null` = 到 EOS |
| `timeout` | 秒，默认 600 | **多久没收到数据**，不是总时长 |
| `sys_file` | 路径；空 | system prompt 文件，空 = 不发 |
| `base_url` | 默认 `http://127.0.0.1:18080` | 服务端地址 |
| `mem` | `0`/`1`/`2`/`3`，默认 `0` | 注入 `mem\`：1 文本、2 图片、3 都要 |
| `mem_scope` | 目录名；默认 `all` | 用 `mem\` 下哪个架子（图库/记忆分家） |
| `mem_pick` | `one`/`all`，默认 `all` | 图片：一句只命中最相关一张，或整架全注入 |

其余命令行开关：`--hide-thinking` 只看答案、`--no-tui` 单行输入、`--once` 问一句就退、
`--image 图` 给这一句配图、`--log 文件` 同时写盘、`--quiet` 不屏显。

## 图片输入

服务端得挂上 projector（`llmserver3d.bat` 在 `weights\` 里有 mmproj 时自动挂）。两种方式：

- `/img PATH` 或 `--image PATH`：只跟着你发的那一句走。
- `mem\<架子>\` 图库：每张图配一个**同名** `.md` 标签，说话时命中就把那张挂上，
  一句最多一张，命中不了就不带图。不与图同名的 `.md` 是常驻约定，第一轮注入。
  `llmchat3d.bat` 走 `config.3d.json`（`mem=3`、`scene` 架子、`pick=one`）。
  细节和 `/mem` 命令都在 [mem\README.md](mem/README.md)。

## 会话内命令

`quit` / `q` 退出，`clear` 清空上下文并删槽位文件，`/paste` 到 `/end` 贴长文，
`/img PATH` 配图，`/mem` 看架子、`/mem find QUERY` 试命中。

## 注意

- 会话和槽位文件按端口区分（`llmchat_18080.slot`），换端口不串。
- 重启后能续接，前提是历史一条没改（含思考块）；改了就从头发一遍。带图的那条同理：
  历史里存的是图片路径，发送时才编码成 data URL，所以存档不会被 base64 撑大。
- 图片吃的是上下文窗口，下限 1024 token/张；跑一轮看 `[speed] prefill N tok` 是真实值。
- `llmchat.bat` 和 `llmchat3d.bat` 必须是 ASCII + CRLF。

## 记忆 A/B 对照（llmchat-mem.bat）

同一个 prompt 发两次：一次不带记忆、一次带记忆，左右两栏并排，用来看记忆库到底有没有用。

```
llmchat-mem.bat               交互，一行一个 prompt
llmchat-mem.bat -f q.txt      批量，文件里每行一个
llmchat-mem.bat --code        代码档（temp 0.2 / presence 0.3）
llmchat-mem.bat --stack       窄终端改成上下两段
```

- 不带历史、不落 session，纯单轮
- 两侧用同一组采样参数和同一个 seed（默认 42），只有记忆开闭不同
- 不带记忆那侧靠引擎的请求级开关 `{"mem": false}`，需要服务端
  `features.ini` 里 `LLAMA_MEM_REQ_TOGGLE = 1`（默认已开）
- 连 `config.json` 里的 `base_url`（18080），可用 `--base-url` 覆盖

底部两列数字是「耗时 · 生成 token · t/s」；若两侧输出完全一致，会提示这条 prompt 没触发记忆。
