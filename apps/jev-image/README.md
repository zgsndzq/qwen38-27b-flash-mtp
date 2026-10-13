# jev-image —— 图像 JEV 接口 + 网页

两种用法，同一个判据内核：

- **网页**：上传一张图 → 问它 → 出答案（`run.bat`，UI 在 **18091**）；
- **命令行**：给「图 + 问题 + 选项」判分，返回**答案字母 + 该字母在选项上的归一化概率**。

形态对标外部榜 Image JevBench 的 System One 决策题（那道题的校准轴，纯字母输出恒得 0）。
判据链路与 `llmchat\sat\sat_test.py` 的 `jev()` 完全同形，两边数字可以直接比：

```
max_tokens = 1  +  logprobs  +  top_logprobs = 40  +  enable_thinking = false  +  采样全关
```

一次前向只吐 1 个 token，取它在选项字母集合上重归一化（没进 top-40 的按 logprob = -30 计）。

## 跑起来

模型服务**不在本目录启动**，先在别处起 `..\llmserver.bat`（固定 **18080**；要识图必须
`VISION=1`，否则视觉头不挂、图会被静默忽略），然后：

```
run.bat                      不带参数 = 起网页（浏览器自动打开 18091）
```

`run.bat` 只做三件事：找 python → 检查 18080 是否在答（不在答就只警告，**不替你启服务**）
→ 起 `bridge.py` 并打开浏览器。**关掉那个窗口 = 停 UI。**

```
UI      http://127.0.0.1:18091/
model   http://127.0.0.1:18080/
```

## 网页（可视化）

上面是**吸顶操作栏**，所有操作都在那儿，不用滚动就能用：图（左上小方块 / 拖到框里 / `Ctrl+V` 粘贴）
→ 问题 → 候选（可留空）→ 识别。`Enter` 直接跑。

图从浏览器上传后缩到**最长边 ≤ 1280 px** 再转 PNG —— 控制请求体大小，也不影响模型自己的
视觉编码（mmproj 有固定输入尺寸）。页面是纯 data URI POST 给本机 bridge、再原样转发到 18080，**图不落盘**。

空白时结果区有一个 **▷ 用示例图跑一次判据** 按钮（取 `llmchat\mem\chess.png` + 4 个候选直接跑）；
带 `?demo=1` 打开也会自动跑同一套 —— 给人看效果或截图取证用。

### 两种回答方式（按「候选」栏空不空自动选，没有开关）

| 形态 | 什么时候走 | 调用 |
|---|---|---|
| **直接回答** | 候选栏空着 | `max_tokens=320`、温度 0、关思考 → 一句话答案 |
| **候选判据 (JEV)** | 候选栏填了 ≥2 项 | `max_tokens=1` + `logprobs` + `top_logprobs=40`，一次前向**零解码**，各候选概率直接读 logits |

判据模式最多 8 个候选（逗号 / 顿号 / 换行分隔）。结果按概率降序排成
`序号 | 候选 | 长度条 | 百分比`，序号就是名次，**第一名不做任何特殊样式**；右上角给
`top-1` 与 `mass`（概率覆盖 = 选项字母在完整词表分布里的总概率，偏低说明模型本来想说别的，
这次答案不可信）。

给模型的选项标记仍是 `A/B/C/D…`（判据协议要求单 token 字母，也要和
`llmchat\sat\sat_test.py` 的 `jev()` 同形），**只在页面上换成 1/2/3**；
想看模型的原始输入就展开「模型实际看到的内容」。

主题是**白底深字**（对比度 正文 17.3:1 / 次要 7.1:1 / 最弱 5.1:1），不是暗色 —— 之前那版暗色是错的。

## 命令行

`run.bat` 后面加 `--cli` 就把参数原样交给 `image_jev.py`：

```
run.bat --cli --image ..\..\llmchat\mem\chess.png --question "河界上写的是哪四个字？" --options "楚河汉界,楚河漢界"
run.bat --cli --batch samples\chess.jsonl --out out\preds.jsonl
run.bat --cli --score out\preds.jsonl
run.bat --cli --batch samples\chess.jsonl --dry          只看拼装、不发请求
```

服务地址默认 `http://127.0.0.1:18080`，可用环境变量 `LLAMA_CHAT_URL` 或 `--url` 覆盖。

## 题目文件格式（jsonl，一行一题）

```json
{"id":"q1","image":"F:/path/pic.png","question":"...","options":["甲","乙","丙"],"gold":"B"}
```

`gold` 可省（省了就不打分）。`options` 最多 8 个（A–H）。`image` 省掉就是**无图对照题**。

## 输出（out\preds.jsonl 每题一行）

`answer` 选项字母 · `prob` 选中项概率 · `probs` 全部选项概率 · `mass` 选项字母在完整词表里的
总概率 · `valid` 是否解析出字母 · `latency_ms` · `prompt_n` · `gpu_s`（`prompt_ms + predicted_ms`）
· `usd_per_1000`。

`--score` 输出：准确率（无效/不可解析算错）、`valid_rate`、10-bin top-label **ECE**、
`mass_mean`、p50/p95 延迟、**GPU 秒/题**、**USD/1000**。成本口径与榜上一致 = 实测 GPU 秒 × 单价。

## 基线（2026-09-26，chess.png，5 题）

```
4/5 正确 · valid_rate 1.0 · mass 0.99 · p50 2.31 s · gpu 1.76 s/题 · USD 0.49/1000 @$1/h
错题 = 「河界下方第一行有几枚红兵」答 7（真值 5，p=0.436）—— 数数类问题视觉容易错
```

## 目录

| 文件 | 作用 |
|---|---|
| `bridge.py` | 网页的同源后端：收 data URI、拼 prompt、转发 18080、算好概率回给页面（stdlib only） |
| `index.html` | 页面本体（白底深字、吸顶操作栏、候选概率列表、耗时/成本） |
| `image_jev.py` | 判据内核（stdlib only，单文件）；CLI 也走它 |
| `run.bat` | 启动器：默认起网页；带 `--cli` 转命令行 |
| `samples\chess.jsonl` | 5 题样例（题目里的图用 `llmchat\mem\chess.png`，**没有另拷副本**） |
| `shots\ui_empty.png` | 打开时的样子 |
| `shots\ui.png` | 判据结果（`?demo=1` 自动跑一遍后截的） |
| `out\` | 预测落盘处 |

## 成本轴往哪砍（榜上我们唯一塌的轴）

1. **前缀复用**：system + 题目模板固定，只 prefill 题目本身（网页默认已开 `cache_prompt`）；
2. **图片按内容 sha1 缓存编码**（接口内已做，进程级）；
3. 降图尺寸 / 视觉 token 数。
