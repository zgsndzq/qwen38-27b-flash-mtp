# vocab —— 草稿词表短路（`--spec-draft-vocab-map`）的离线训练与分析

这个文件夹只干一件事：**造一张"代码高频 token"短表，并测出它值不值得挂到草稿头上**。

它不重编任何东西、不改生产 `llmserver.bat`、不需要 GPU（生成语料那一步除外）。

设计依据：`../docs/design-vocab-shortlist.md`（机理、两个静默阻断、收益模型、判据表）。
实测账：`../docs/BENCH.md`。

---

## 为什么是"代码专用"词表

草稿头 `output.weight [5120, 248320]`（265 MB）**每出一个草稿 token 就要整份读一遍**。
而一段真实代码用到的词，高度集中：`\n`、`    `（4 空格缩进）、`self.`、`def `、`return `、
`():`、` =` 这些在 BPE 里**各自就是一个 token**。把表从 248320 缩到 8k，草稿头按 `N/248320` 等比变小。

省下的是**按步计费的项**，trunk 是每轮固定项 ⇒ **草稿越深，短路越值钱**（代码场景深度 2.657 > 散文 2.073）。

天花板：**代码 10.0% / 散文 8.1%**（草稿头占每轮字节的比例）。
代价：草稿只能在表内提词，落在表外的 token 必失；一阶模型下 **净收益 = 毛节省 − 掉词率 q**。
⇒ 所以 **q 必须小于毛节省**，这就是下面那张覆盖率曲线要回答的问题。

---

## 目录

```
vocab/
├── run.py            ← 唯一入口（阶段分派）
├── srv.py              服务生命周期 + HTTP（绝不杀不是自己起的服务）
├── prompts.py          83 条代码任务提示词（py/c/sh/js/sql/algo/dbg + 少量指令）
├── corpus.py           两个语料源：gen（模型生成）/ local（真实代码库）
├── tok.py              分词：service（权威）/ offline（精确复刻 qwen35 切分器）
├── shortlist.py        统计 → 出表 → 覆盖率曲线 → 判据结论
├── selftest.py         离线自检（含 map 文件格式校验器）
├── corpus/
│   ├── gen/            模型生成的样本（每样本一个 json）
│   ├── local/          从真实代码库抽的样本
│   └── ids/            分词结果缓存（每样本一个 json，按字段存 id 列表）
└── out/                表文件 + coverage csv + VERDICT.md + analysis.json
```

---

## 快速开始

```bat
:: 1. 零成本预演：直接从本机真实代码库抽语料（不需要 GPU）
python run.py local --limit-files 400

:: 2. 分词（offline 后端，不需要 GPU）
python run.py tok --backend offline

:: 3. 出表和曲线
python run.py build --source local

:: 4. 看进度
python run.py status
```

要拿**权威**数据（模型自己生成的语料 → 最贴合草稿要猜的目标分布）：

```bat
python run.py gen --limit 12          :: 先小批量
python run.py tok                     :: 有服务就用 service 后端（精确、权威）
python run.py build --source gen
python run.py all                     :: 或一次跑完 local+gen+tok+build
```

每个阶段都是**增量**的：已有产物且比输入新就跳过，重跑即续跑。
（本机长跑后台任务约 **2 分钟**会被回收，所以每一步都落盘，被砍了不丢数据。）

---

## 两个语料源，别混

| | `gen`（模型生成） | `local`（真实代码库） |
|---|---|---|
| 成本 | 需 GPU，约 25 s/条 | 零成本，立即可用 |
| 权威性 | **权威** —— 草稿头模仿的就是模型自己的输出分布 | 参考 —— 人写的代码，标识符命名风格不同 |
| 用途 | 最终判据 | 先跑通、拿趋势；人写代码都覆盖不了就别谈模型输出 |

`--source all` 会同时评估两者，能看出"能不能用真实代码库造表"。

---

## 两个分词后端

| | `service`（`/tokenize`） | `offline` |
|---|---|---|
| 来源 | 跑着的服务，模型自己的 tokenizer | 直接读 GGUF 的 `tokenizer.ggml.tokens` / `.merges` |
| 精度 | 权威 | **精确复刻** `qwen35` pre-tokenizer（见下） |
| 需要 | GPU + 服务 | 只需要 GGUF 文件（几秒读完 header） |

`--backend auto`（默认）有服务就用 service，没有就退回 offline。

**offline 为什么可信**：`tokenizer.ggml.pre = "qwen35"` 对应的切分规则在
`forks/llamAmpere/src/llama-vocab.cpp`（`LLAMA_VOCAB_PRE_TYPE_QWEN35`）里是写死的，
`tok.py: scan_qwen35()` 逐条复刻了它，用标准库 `unicodedata` 判 Unicode 类别（不依赖 PyPI，
本机装不上）。

两个容易搞错的细节，都已按源码实现并在 `selftest.py` 里断言：

- `\p{N}` 是**单个数字字符** —— `10` 是两块，不是一块
- `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` 的"可选前导字符"可以是**任意**非字母非数字字符
  ⇒ `(a` 是一块、`):` 是一块，但 `\n` 和 `    ` 永远不跨边界合并

```bat
python selftest.py                              :: 自检：切分断言 + map 格式
python run.py tok --backend service             :: 拿权威数据跑一次，两版对照
```

---

## 输出

- `out/shortlist_<N>_<tag>.vocab.txt` —— 直接喂给 `--spec-draft-vocab-map` 的表
- `out/coverage_<tag>.csv` —— N vs 覆盖率/q/净收益
- `out/VERDICT_<tag>.md` —— **判据结论** + 表内容 + 表外高频 token（诊断漏了什么）
- `out/analysis_<tag>.json` —— 报告用到的全部原始数字

表文件格式（`src/llama-mtp-vocab.h`，**格式错会让服务起不来**，是响亮失败）：

```
llama-mtp-vocab-v1 248320 <count>
<id0> <id1> ... <id_{count-1}>
```

- `n_vocab` 必须**恰好等于**模型的 248320（服务端用模型的词表大小做上限，所以表长可开到接近全量）
- 无重复、无尾部多余数据。`selftest.py` 里的校验器复刻了 C++ 那套检查，含负例自测。

---

## 判据（`docs/design-vocab-shortlist.md` §4.5.3）

| 8k 表覆盖率 | 掉词率 q | 对照毛节省 10.0% | 结论 |
|---|---|---|---|
| ≥ 99.5% | ≤ 0.5% | 10.0 − 0.5 | **净 ≈ +9.5%，几乎拿满** |
| ≥ 99% | 1% | 10.0 − 1.0 | 净 ≈ +9%，值得做 |
| ≥ 97% | 3% | 10.0 − 3.0 | 净 ≈ +7%，仍值得 |
| ≥ 90% | 10% | 保本 | 靠自适应深度兜底，危险 |
| < 90% | > 10% | 负 | 放弃这条线 |

**下行有地板**：推荐脚本用 `--spec-draft-n-min-adaptive 1`，词表再烂最多把深度压到 1，
而深度 1 实测 **40.91 t/s**（比 `DRAFT=3` 的 39.22 还高）；target 始终全词表验证
⇒ **输出质量零损失**。所以结构是「上行 +7~9% / 下行 ~0 / 质量零风险」。

---

## 还差什么才能真跑起来

这张表现在还**挂不上**，两个阻断都在 `llama-graph.cpp`（详见设计文档 §2）：

1. 头的量化类型是 `PTQ1_0`（type 143），不在 `draft_vocab_direct()` 的白名单里 → 一行修改
2. 头被 `prism.hadamard` 折叠了 → 短路函数里要照搬 `build_lora_mm` 的旋转（约 20~30 行）

两者都是**静默回退**：日志里那句 `uses the draft-only vocabulary shortlist` 只证明表被读进去了，
**不证明图里真走了短路**。所以验收必须做**功能反证**：故意用 512 条极短表，接受率必须崩。
