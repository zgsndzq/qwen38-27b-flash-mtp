# llmchat

小对话客户端，用来测已经在跑的 llama-server（先开 `..\llmserver.bat`）。
只用 Python 标准库，不加载模型。

## 用法

双击 `llmchat.bat` 即可。**配置全在同目录的 `config.json`**，改完直接生效；
命令行给的参数会覆盖它，例如：

```
python chat.py --think server --temp 0.2
```

`config.json` 的键和参数同名（下划线写法）：

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
| `think` | `on` / `off` / `server` | 思考开关；`server` = 不发，用服务端 `-rea` |
| `effort` | `low`；或 `xhigh` `medium`；留空 | 思考强度；留空 = 服务端默认 |
| `temp` | 数字；`null` | 采样温度；`null` = 用服务端 `--temp` |
| `max_tokens` | 数字；`null` | 单轮回答上限；`null` = 到 EOS |
| `timeout` | 秒，默认 600 | **多久没收到数据**，不是总时长 |
| `sys_file` | 路径；空 | system prompt 文件，空 = 不发 |
| `base_url` | 默认 `http://127.0.0.1:18080` | 服务端地址 |

其余命令行开关：`--hide-thinking` 只看答案、`--no-tui` 单行输入、`--once` 问一句就退、
`--log 文件` 同时写盘、`--quiet` 不屏显。

## 会话内命令

`quit` / `q` 退出，`clear` 清空上下文并删槽位文件，`/paste` 到 `/end` 贴长文。

## 注意

- 会话和槽位文件按端口区分（`llmchat_18080.slot`），换端口不串。
- 重启后能续接，前提是历史一条没改（含思考块）；改了就从头发一遍。
- `llmchat.bat` 必须是 ASCII + CRLF。
