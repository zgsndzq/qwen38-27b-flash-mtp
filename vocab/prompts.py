"""prompts.py -- the code-task prompt catalogue that seeds the corpus.

Why a catalogue instead of a few prompts: the shortlist is a frequency table, so
its quality is bounded by how representative the corpus is. Coverage has to span
the languages and the *shapes* of answer we actually consume:

  * several languages (python / c / shell / javascript / sql),
  * long answers, not only short snippets -- indentation runs, punctuation runs
    and idiomatic fragments only show up in multi-hundred-token responses,
  * reasoning-heavy tasks, because the thinking block is a large share of real
    traffic and its token mix differs from the code it precedes,
  * ordinary instruction prose, since prompts and their wrappers are predicted
    too,
  * plain Chinese instruction traffic, because the shortlist budget is zero-sum:
    rows spent on English code identifiers are rows not spent on CJK tokens, and
    measurement (not assumption) has to decide that split.

Each entry is (label, prompt); the enclosing list is what gives it its group, and
``catalog()`` returns (label, group, prompt) triples. Keeping the group in the
returned triple lets the analysis slice "code only" vs "including the thinking
block" without re-running anything.

Weights used by gen: groups are expanded round-robin, so a larger group simply
contributes proportionally more samples.
"""

# --------------------------------------------------------------------- python
PY = [
    ("py.ringbuffer", "Implement a Python class RingBuffer backed by a fixed-size list, with push, pop, "
                      "peek, is_full, is_empty, __len__ and an iterator. Type hints, docstrings, and a "
                      "short usage example at the end."),
    ("py.lru", "Implement an LRU cache in pure Python using a dict plus a doubly linked list. Explain why "
               "OrderedDict would be simpler and when the manual version is worth it."),
    ("py.decorator", "Explain and implement a retry decorator that takes max_attempts, base_delay and an "
                     "exception tuple, uses exponential backoff with jitter, and preserves the wrapped "
                     "function signature."),
    ("py.asyncio", "Write an async Python worker pool that consumes a queue, runs at most N tasks "
                   "concurrently with asyncio.Semaphore, handles cancellation cleanly, and reports "
                   "progress. Include error handling."),
    ("py.pytest", "Write a pytest module for a function that parses a duration string like '1h30m' or "
                  "'90s' into seconds. Cover parametrised cases, edge cases and error cases."),
    ("py.csv", "Write a Python script that reads a large CSV, aggregates a numeric column by a key column, "
               "writes the result sorted descending, and streams the input so memory stays flat. Use only "
               "the standard library."),
    ("py.refactor", "Refactor a 200-line function that both validates input and writes files into three "
                    "smaller functions with a clear data flow. Show the before/after structure and explain "
                    "the seams you chose."),
    ("py.dataclass", "Explain dataclasses, slots, frozen, field(default_factory=...) and __post_init__ "
                     "with a practical example of a config object loaded from JSON, including validation."),
    ("py.logparser", "Write a Python log parser that extracts timestamp, level, module and message from "
                     "lines like '2026-09-22 20:11:03,412 WARNING worker.queue - queue depth 91', then "
                     "summarises counts per level per hour."),
    ("py.argparse", "Build a CLI with argparse that has subcommands init, add and list, a global --verbose "
                    "flag, sensible defaults, and gives clean error messages. Show the help output too."),
]

# -------------------------------------------------------------------------- c
C = [
    ("c.hashmap", "Implement an open-addressing hash map in C with linear probing, tombstones, growth at "
                  "70% load, and a string-key specialisation. Show the header and the implementation."),
    ("c.arena", "Implement a bump/arena allocator in C: init from a malloc'd block, aligned allocation, "
                "save/restore checkpoints, and a reset that keeps the memory. Explain alignment handling."),
    ("c.ringbuf", "Write a lock-free single-producer single-consumer ring buffer in C11 with atomics. "
                  "Explain the memory ordering choices at each store."),
    ("c.makefile", "Write a Makefile for a small C project with two object directories, automatic header "
                   "dependencies, debug and release configurations, and a clean target."),
    ("c.threadpool", "Implement a thread pool in C using pthreads, a task queue with a mutex and condition "
                     "variable, graceful shutdown, and a way to wait for completion."),
    ("c.parse", "Write a recursive-descent parser in C for arithmetic expressions with precedence, "
                "parentheses and unary minus, returning an AST you can evaluate."),
]

# ----------------------------------------------------------------- shell / ops
SH = [
    ("sh.topfiles", "Write a bash pipeline that finds the 10 largest files under a directory, prints path "
                    "and human-readable size sorted descending, and tolerates spaces in filenames. Explain "
                    "each stage."),
    ("sh.watchdog", "Write a systemd unit and a small wrapper script that restarts a service when its "
                    "health endpoint fails three times in a row, with backoff and logging."),
    ("sh.awklog", "Write an awk program that summarises an nginx access log: requests per minute, 4xx/5xx "
                  "rates, and the top ten clients, output as a compact table."),
    ("sh.deploy", "Write a bash deployment script with set -euo pipefail, a lock file, a rollback on "
                  "failure, and coloured status output. Explain the trap handling."),
    ("sh.git", "Explain and demonstrate git commands for finding which commit introduced a bug: bisect, "
               "log -S for a string, log -L for a function range, and blame with ignore-revs."),
]

# --------------------------------------------------------------- js / frontend
JS = [
    ("js.debounce", "Implement debounce and throttle in modern JavaScript with leading/trailing options, "
                    "cancel and flush methods, and explain the timer bookkeeping."),
    ("js.fetch", "Write a JavaScript helper that fetches JSON with timeout, retry with backoff, "
                 "cancellation via AbortController, and a typed error for non-2xx responses."),
    ("js.dom", "Explain event delegation and write a vanilla JavaScript component that renders a sortable, "
               "filterable table from a JSON array without a framework."),
    ("js.promise", "Explain the microtask queue with concrete ordering examples mixing setTimeout, "
                   "Promise.then and async/await, then implement a tiny promise pool."),
]

# ------------------------------------------------------------------------ sql
SQL = [
    ("sql.report", "Write a SQL query that produces a monthly revenue report with month-over-month growth "
                   "percentage from an orders table, including months with no orders."),
    ("sql.schema", "Design a schema for a multi-tenant ticketing system with tenants, projects, tickets, "
                   "comments, tags and attachments. Give the DDL, indexes and the reasoning."),
    ("sql.opt", "Explain how you would diagnose and fix a slow query: reading EXPLAIN ANALYZE output, "
                "composite index ordering, covering indexes and avoiding functions on indexed columns."),
]

# ---------------------------------------------------------------------- algos
ALGO = [
    ("algo.dijkstra", "Implement Dijkstra with a binary heap, explain why decrease-key is usually replaced "
                      "by lazy insertion, and give the complexity of both variants."),
    ("algo.lru_clock", "Compare LRU, LFU and CLOCK page replacement: implement CLOCK with a reference bit, "
                       "and explain the trade-offs with concrete access patterns."),
    ("algo.mergesort", "Implement an iterative bottom-up merge sort over an array of ints, in place where "
                       "possible, and explain why the recursive version is usually faster in practice."),
    ("algo.trie", "Implement a trie for prefix search with insert, search and starts_with, plus a "
                  "memory-efficient variant. Show the complexity of each operation."),
]

# --------------------------------------------------------------------- debug
DEBUG = [
    ("dbg.segfault", "A C program segfaults only in release builds. Walk through a systematic diagnosis: "
                     "compiler flags, undefined behaviour, stack vs heap, and how to use ASan and gdb."),
    ("dbg.leak", "Explain how to find a memory leak in a long-running Python service: tracemalloc, gc "
                 "referrers, object growth by type, and the usual culprits such as caches and cycles."),
    ("dbg.flaky", "A test fails about one run in twenty. Describe a method for turning that into a "
                  "reproducible case, and list the usual causes of flakiness in tests."),
    ("dbg.slow", "A service is slow only under load. Explain how to separate the hypotheses: queueing, "
                 "lock contention, GC pauses, connection pool exhaustion and downstream timeouts."),
]

# ------------------------------------------- instruction prose (small slice)
CHAT = [
    ("chat.plan", "I have a 5 GB model that must serve 40 tokens per second on one 8 GB GPU. Outline a "
                  "plan for finding the bottleneck, listing what you would measure first and why."),
    ("chat.review", "Review this design: a CLI tool that reads a config file, expands environment "
                    "variables, validates the result and then starts a server. Point out the failure "
                    "modes and the improvements you would make."),
    ("chat.explain", "解释一下推测解码（speculative decoding）里接受率的含义，以及为什么草稿深度加大之后"
                     "接受率反而可能下降。用具体例子说明。"),
    ("chat.compare", "对比两种方案：把草稿词表缩小到高频 token，与提高草稿深度。分别说明它们的收益来源、"
                     "代价，以及什么情况下各自的期望收益为正。"),
]

# ------------------------------------------------------------------- chinese
# Plain Chinese instruction traffic. Added after the scene-coverage measurement
# showed the table built from English code is a net LOSS on Chinese answers
# (~-9% modelled), for a structural reason: a table mined from English code
# cannot contain CJK tokens at all, so no amount of extra code samples moves it.
# The table length is a fixed budget, so what is really being decided here is
# the *share* of the budget CJK gets -- and that share should follow measured
# traffic, not the assumption "code dominates" that WEIGHTS used to encode.
#
# Deliberately varied shapes (explain / compare / how-to / debug / review /
# estimate / plan / checklist / API doc / plain chat) so the CJK token
# distribution is not dominated by one topic's vocabulary.
ZH = [
    ("zh.explain", "用中文解释一下推测解码里“接受率”的含义，以及为什么把草稿深度从 2 加到 4 之后接受率"
                   "反而可能下降。请举一个具体的例子走一遍。"),
    ("zh.compare", "对比两种提速手段：缩小草稿词表 与 提高草稿深度。用中文分别说明它们的收益来源、"
                   "代价，以及在什么条件下各自的期望收益为正。"),
    ("zh.howto", "我要在一台只有 8 GB 显存的机器上编译并启动 llama.cpp 的推理服务，请用中文给出完整步骤："
                 "环境准备、编译参数、启动参数、验证方法。每一步说明为什么这么做。"),
    ("zh.debug", "一个服务在低负载下正常，压力上来之后响应时间从 200 毫秒涨到 3 秒。请用中文列出排查顺序："
                 "先量什么、看哪些指标，以及如何把结论缩小到某个子系统。"),
    ("zh.review", "帮我评审这段设计：一个命令行工具读取配置文件、展开环境变量、校验之后启动服务。"
                  "请用中文指出可能的失败模式和改进建议。"),
    ("zh.summary", "把下面这类技术讨论整理成要点：混合专家模型的推理瓶颈、批处理对吞吐的影响、"
                 "以及显存与带宽的取舍。请用中文分条列出，每条给一句理由。"),
    ("zh.metrics", "请用中文说明如何评估一个大模型的生成速度：首字延迟、解码速度、并发吞吐各自的定义、"
                   "测量方法，以及它们之间会互相trade-off的原因。"),
    ("zh.estimation", "用中文估算：从内存里读取一个 24000 行、每行 5120 维的量化权重矩阵，每生成一个 token "
                      "需要多少带宽？如果显存带宽是 400 GB/s，这一步理论上要多久？请写出计算过程。"),
    ("zh.tradeoff", "在消费级显卡上部署大模型时，量化精度、上下文长度、并发数三者互相挤压显存。"
                    "请用中文分析三者的取舍关系，并给出针对 8 GB 显卡的建议配置。"),
    ("zh.checklist", "请用中文给出一份“上线前检查清单”：覆盖依赖版本、配置文件、日志、异常处理、"
                     "性能回归和回滚方案。每个条目说明检查方法和合格标准。"),
    ("zh.plan", "我打算在两周内把一个大模型的推理速度提升 30%。请用中文制定计划："
                "先做什么、如何度量基线、怎样安排实验、什么情况下放弃某条路线。"),
    ("zh.answer", "有用户反馈模型在中文对话里回答得比英文慢。请用中文给出你的分析思路、"
                  "需要收集哪些证据，以及可能的原因清单。"),
    ("zh.doc", "请用中文写一段 API 文档，说明一个命令行参数的含义、取值、默认值和副作用，"
               "并给出两个使用示例和常见错误的解释。"),
    ("zh.teach", "用一个生活化的比喻向初学者解释什么是 KV 缓存、它为什么占显存、"
                 "以及为什么上下文越长显存占用越大。请用中文，控制在十句话以内。"),
    ("zh.risk", "请用中文分析一项把本地模型接入生产流水线的风险：数据合规、输出稳定性、"
                "故障回退和成本波动。针对每项风险给出具体的缓解措施。"),
    ("zh.optimize", "我的推理服务在单张显卡上速度为每秒 30 个 token，图形处理器占用率却只有 60%，"
                    "功耗也跑不满。请用中文给出优化方向，并按预期收益从高到低排序。"),
]

# Chinese *comments inside code*. This is the shape real traffic actually has:
# the draft head sees Chinese and ASCII interleaved token by token, and a table
# mined from English code has neither the CJK tokens nor their mixed-order
# contexts. Kept separate from ZH so the analysis can tell "Chinese prose" from
# "Chinese inside code", which have different token mixes.
ZHCODE = [
    ("zc.ringbuffer", "用 Python 实现一个环形缓冲区类，支持 push、pop、peek、判满、判空和迭代；"
                      "所有文档字符串和行内注释都用中文，最后给一段使用示例。"),
    ("zc.sql", "写一段带中文注释的 SQL：按用户维度统计最近 30 天的订单金额、笔数和客单价，"
               "按金额倒序取前 100 名，注释说明每个子查询的目的。"),
    ("zc.batch", "写一个 Windows 批处理脚本，把指定目录下的日志按日期归档并删除 30 天前的旧文件；"
                 "每一步用中文注释说明，包含错误处理。"),
    ("zc.cparse", "写一个 C 函数，从配置文件里读取键值对并解析成结构体；用中文注释说明边界条件、"
                  "内存归属和出错时的返回约定。"),
    ("zc.tsapi", "写一个 TypeScript 工具函数，把任意对象转换成查询字符串；用中文注释说明参数含义、"
                 "边界情况和抛出的异常类型。"),
    ("zc.test", "给一个计算校验和的函数写单元测试，测试用例的中文注释要说明每个用例覆盖的场景，"
                "包含边界值和异常输入。"),
    ("zc.shell", "写一个带中文注释的 shell 备份脚本：压缩指定目录、按时间戳命名、保留最近七份、"
                 "失败时报警，注释说明保留策略和并发保护。"),
    ("zc.annotate", "给一段没有注释的代码补上中文注释和文档字符串，说明每个函数的作用、"
                    "参数和返回值，并解释为什么这样拆分。"),
]

# Chinese reasoning. Real chat traffic spends a large share of its tokens in the
# thinking block, and that block was absent from every corpus built so far --
# the production launcher runs with -rea off, so nothing ever produced one. These
# prompts are generated with enable_thinking, and the corpus records the
# reasoning and the answer separately (FIELDS already treats them as two chunks),
# so a table can be scored with and without the thinking block.
ZHTHINK = [
    ("zt.analyze", "一步一步分析：为什么把草稿深度从 2 提高到 4 之后，推测解码的接受率反而下降了？"
                   "先列出可能的机制，再逐条判断哪个解释更站得住脚，最后给结论。"),
    ("zt.diagnose", "逐步排查：一个服务在低负载下正常，并发升高后响应时间明显变长。"
                    "请先把现象拆成可测量的假设，再说明每一步要收集什么证据、如何排除。"),
    ("zt.design", "分步骤设计一个缓存淘汰方案：先分析访问模式，再比较几种策略的代价，最后给出选择和建议参数。"),
    ("zt.math", "逐步计算：一个 24000 行、每行 5120 维的量化权重，每生成一个 token 需要读取多少数据？"
                "在 400 GB/s 带宽下理想耗时是多少？请写出每一步的推导。"),
    ("zt.compare", "先列判据，再逐条对比两种提速方案：缩小草稿词表 和 提高草稿深度。"
                   "最后说明在什么条件下各自更优。"),
    ("zt.decide", "请先说明你的思考过程，再给出最终建议：在 8 GB 显存的显卡上，"
                  "量化精度、上下文长度和并发数应该优先牺牲哪一个？"),
]

GROUPS = {"py": PY, "c": C, "sh": SH, "js": JS, "sql": SQL, "algo": ALGO, "dbg": DEBUG,
          "chat": CHAT, "zh": ZH, "zhcode": ZHCODE, "zhthink": ZHTHINK}

# Groups whose samples are generated with the thinking block enabled. The
# launcher ships with -rea off, so every other group matches the current serving
# configuration; these exist so the table can also cover the block that shows up
# the moment thinking is switched on.
THINKING_GROUPS = {"zhthink"}

# Relative sampling weight per group. Code still leads, because that is the bulk
# of the traffic this head drafts for -- but Chinese is no longer a token gesture:
# the scene-coverage run showed the old all-code table actively costs ~9% on
# Chinese answers, so the CJK share of the budget is a measured quantity now.
#
# The corpus deliberately OVER-covers Chinese. The table length is a fixed
# budget, so a corpus that contains little CJK cannot produce a table with much
# of it, while a CJK-rich corpus can always be dialled *down* at build time. The
# split belongs in the build step (see mixbuild.py), not in the sampling weights.
WEIGHTS = {"py": 3, "c": 2, "sh": 2, "js": 2, "sql": 1, "algo": 2, "dbg": 2,
           "chat": 1, "zh": 2, "zhcode": 1, "zhthink": 1}


def catalog(groups: str = "all"):
    """Return [(label, group, prompt)], expanded by WEIGHTS, in a stable order.

    Ordering is interleaved by (prompt index, repetition round, group) rather than
    grouped by language. That matters because a background run can be reaped at the
    ~2 minute mark: whatever prefix got produced should already span many languages
    instead of being all Python.

    Labels are always unique -- a repeated sample gets a ``-r2`` / ``-r3`` suffix --
    because the generator writes one file per label and duplicates would overwrite
    each other.
    """
    want = list(GROUPS) if groups == "all" else [g.strip() for g in groups.split(",") if g.strip()]
    unknown = [g for g in want if g not in GROUPS]
    if unknown:
        raise ValueError("unknown group(s): %s (have %s)" % (",".join(unknown), ",".join(GROUPS)))

    pools = {g: list(GROUPS[g]) for g in want}
    rounds = max(WEIGHTS[g] for g in want)
    longest = max(len(p) for p in pools.values())

    out = []
    for i in range(longest):
        for r in range(rounds):
            for g in want:
                if r >= WEIGHTS[g] or i >= len(pools[g]):
                    continue
                label, prompt = pools[g][i]
                if r > 0:
                    label = "%s-r%d" % (label, r + 1)
                out.append((label, g, prompt))
    return out


def _self_test():
    cat = catalog()
    labels = [c[0] for c in cat]
    assert len(labels) == len(set(labels)), "duplicate label in catalogue"
    assert all(len(c) == 3 for c in cat)
    per = {}
    for label, g, _p in cat:
        per[g] = per.get(g, 0) + 1
    print("catalogue ok: %d prompts, groups=%s" % (len(cat), sorted(per.items())))


if __name__ == "__main__":
    _self_test()
