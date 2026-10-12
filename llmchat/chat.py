#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""llmchat —— 独立的 llama-server 对话客户端（单文件，零第三方依赖）。

等价于项目里的 `python llm.py chat`，但把这个文件夹拷到任何地方都能用：
**不依赖 llm/ 的任何包**，只用 Python 标准库。

    python chat.py                                        # 连默认地址 127.0.0.1:18080
    python chat.py --base-url http://127.0.0.1:18081      # 换底座
    python chat.py --temp 0.6 --think on --effort medium
    python chat.py --sys "你是成语接龙助手"
    python chat.py --sys-file sys.txt                     # system prompt 从 UTF-8 文本读
    python chat.py --timeout 1200                          # 覆盖 LLAMA_CHAT_TIMEOUT

会话内元命令（本地拦截，不发给模型）：
    quit / q / exit / 退出        退出
    clear / cls / 清空            清空上下文（保留 system prompt）
    非 TTY 下（管道）：/paste 进入粘贴模式，单独一行 /end 结束

环境变量：
    LLAMA_CHAT_URL        默认 base_url（--base-url 优先）
    LLAMA_CHAT_TIMEOUT    等待服务端「无数据」的上限秒数，默认 600

为什么是「无数据超时」而不是「整体超时」：预填充（prompt eval）期间服务端一个字都不发，
长上下文下 pp 再快也要几十秒到几分钟。所以超时保护的是「心跳」，
长提示词/长上下文报 timeout 时调大 LLAMA_CHAT_TIMEOUT 即可，不是服务挂了。

TUI（多行输入）：装了 prompt_toolkit 且是交互终端时自动启用 ——
粘贴大段文档整块提交、Esc+Enter 手动换行、↑/↓ 历史。没装则退化为单行 + /paste。
"""

import argparse
import json
import os
import socket
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = os.environ.get("LLAMA_CHAT_URL", "http://127.0.0.1:18080")
DEFAULT_SYS = "你是一个有用的中文 AI 助手，请准确、完整地回答用户的问题。"


def fill_sys(text):
    """填 system prompt 里的运行时占位符（每次启动现算，不写死）。

        {date}     YYYY-MM-DD
        {weekday}  一..日

    没装占位符的文本原样返回 ⇒ 老 sys.txt 不受影响。
    """
    if not text or "{" not in text:
        return text
    import datetime
    d = datetime.date.today()
    return (text.replace("{date}", d.isoformat())
                .replace("{weekday}", "一二三四五六日"[d.weekday()]))


DEFAULT_TIMEOUT = float(os.environ.get("LLAMA_CHAT_TIMEOUT", "600"))
CTX_LIMIT = int(os.environ.get("LLAMA_CHAT_CTX", "32768"))   # 仅用于水位提示

EXIT_CMDS = ("quit", "exit", "q", "/quit", "/exit", "退出")
CLEAR_CMDS = ("clear", "/clear", "cls", "/cls", "清空", "清除上下文")
PASTE_CMDS = ("/paste", "/多行")
END_CMDS = ("/end", ":::end")

# 输出被重定向到文件时，用 UTF-8 写，避免 Windows 默认 cp936 把中文写坏。
# 终端（TTY）不动，交给 Python 的 console writer 处理。
if not sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# --------------------------------------------------------------------------
# 交互输入（沿用 llm/cli/input_box.py 的行为，可选依赖 prompt_toolkit）
# --------------------------------------------------------------------------
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    _HAS_TUI = True
except ImportError:
    _HAS_TUI = False


def _make_bindings():
    """multiline 下反转按键：Enter 提交、Esc+Enter 换行（默认是反的，会导致对话卡死）。"""
    from prompt_toolkit.filters import emacs_insert_mode, vi_insert_mode
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.key_binding.bindings.named_commands import get_by_name

    kb = KeyBindings()
    insert = vi_insert_mode | emacs_insert_mode

    @kb.add("enter", filter=insert)
    def _accept(event):
        get_by_name("accept-line").call(event)

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    return kb


def _new_session():
    return PromptSession(multiline=True, history=InMemoryHistory(),
                         key_bindings=_make_bindings())


def _read_message(prompt="你> "):
    """非 TUI：单行直接返回；输入 /paste 进入粘贴模式，/end 结束。"""
    line = input(prompt).strip()
    if line.lower() not in PASTE_CMDS:
        return line
    print("粘贴模式：粘贴内容后，单独一行 /end 结束（空行保留）")
    lines = []
    while True:
        part = input("… ")
        if part.strip() in END_CMDS:
            break
        lines.append(part)
    return "\n".join(lines).strip("\n") if lines else ""


def read_user(session=None, prompt="你> "):
    """统一入口：TTY + 有 prompt_toolkit → TUI 多行；否则退化 /paste 模式。"""
    if _HAS_TUI and sys.stdin.isatty():
        while True:
            text = (session or _new_session()).prompt(prompt)
            if text.strip():
                return text
    return _read_message(prompt)


def stream_printer(hide_thinking=False):
    """on_delta(kind, text)：思考/回答分别边到边打印。attrs: .streamed 记录是否已流式。"""
    st = {"thinking": False, "content": False}

    def on_delta(kind, text):
        if kind == "thinking":
            st["content"] = False
            if hide_thinking:
                if not st["thinking"]:
                    print("\n[思考中...]", end="", flush=True)
                    st["thinking"] = True
                return
            if not st["thinking"]:
                print("\n[思考] ", end="", flush=True)
                st["thinking"] = True
            print(text, end="", flush=True)
        else:
            st["thinking"] = False
            if not st["content"]:
                print("\n\n[回答] ", end="", flush=True)
                st["content"] = True
            print(text, end="", flush=True)

    on_delta.streamed = st
    return on_delta


# --------------------------------------------------------------------------
# llama-server 调用（把 llm/engine/gguf_client.py 的流式核心内联进来）
# --------------------------------------------------------------------------
def probe(base_url, timeout=5.0):
    """/v1/models 探活。返回 (ok, 服务端报的模型 id 或 None, 错误)。"""
    try:
        with urllib.request.urlopen(base_url + "/v1/models", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rows = data.get("data") or []
        return True, (rows[0].get("id") if rows else None), None
    except Exception as e:
        return False, None, e


def chat_sse(messages, base_url, temperature, max_new_tokens=None, on_delta=None,
             think=None, effort=None, timeout=DEFAULT_TIMEOUT):
    """流式 chat：SSE 逐增量回调 on_delta(kind, text)。

    - temperature 传数字会顶掉服务端 --temp；None = 不下发，沿用服务端
    - think=True/False → chat_template_kwargs.enable_thinking（覆盖服务端 -rea）
    - effort="xhigh"|"medium"|"low" → chat_template_kwargs.reasoning_effort
    返回 (reasoning, content) 完整拼接。
    """
    body = {"model": "gguf", "messages": messages, "stream": True}
    if temperature is not None:
        body["temperature"] = temperature
    if max_new_tokens is not None:
        body["max_tokens"] = max_new_tokens
    kwargs = {}
    if think is not None:
        kwargs["enable_thinking"] = bool(think)
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    if kwargs:
        body["chat_template_kwargs"] = kwargs

    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"})

    reasoning, content = [], []
    timings = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:                       # 逐行读 SSE
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "timings" in obj:               # 最后一个 chunk 带计时
                    timings = obj["timings"]
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                r = delta.get("reasoning_content")
                c = delta.get("content")
                if r:
                    reasoning.append(r)
                    if on_delta:
                        on_delta("thinking", r)
                if c:
                    content.append(c)
                    if on_delta:
                        on_delta("content", c)
    except Exception as e:
        hint = ""
        if isinstance(e, socket.timeout):
            hint = (f"\n等待 {timeout:.0f}s 内服务端无任何数据。若是长提示词/长上下文，"
                    f"这是预填充耗时过长（不是服务没起来）；\n"
                    f"调大环境变量 LLAMA_CHAT_TIMEOUT（秒）或加 --timeout 1200 即可。")
        raise ConnectionError(
            f"llama-server 流式调用失败（{base_url}）：{e}{hint}") from e
    return "".join(reasoning), "".join(content), (timings or {})


# --------------------------------------------------------------------------
def _meta(text):
    """会话级元命令 → 'exit' / 'clear' / None（None = 当普通消息发给模型）。"""
    t = (text or "").strip().lower()
    if t in EXIT_CMDS:
        return "exit"
    if t in CLEAR_CMDS:
        return "clear"
    return None


def _parse_think(v):
    """'on'/'off' → True/False；'server' → None（沿用服务端 -rea）。"""
    if v is None or v == "server":
        return None
    return v == "on"




def _build_parser():
    ap = argparse.ArgumentParser(
        prog="chat.py",
        description="llama-server 对话客户端（独立版，零第三方依赖）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例：\n"
               "  python chat.py --temp 0.6 --think on --effort medium\n"
               "  python chat.py --base-url http://127.0.0.1:18081 --think server\n")
    ap.add_argument("--base-url", default=DEFAULT_URL,
                    help=f"llama-server 地址（默认 {DEFAULT_URL}）")
    ap.add_argument("--sys", default=None, help="system prompt（默认内置中文提示词）")
    ap.add_argument("--sys-file", default=None,
                    help="从 UTF-8 文本文件读 system prompt（优先于 --sys）")
    ap.add_argument("--temp", type=float, default=None,
                    help="采样温度；不给 = 不下发，沿用服务端 --temp")
    ap.add_argument("--think", choices=["on", "off", "server"], default="on",
                    help="思考开关（默认 on，会覆盖服务端 -rea off）；"
                         "server = 沿用服务端 -rea；off = 关闭")
    ap.add_argument("--effort", choices=["xhigh", "medium", "low"], default=None,
                    help="思考强度；仅在思考开启时生效")
    ap.add_argument("--timeout", type=float, default=None,
                    help=f"无数据超时秒数（默认 {DEFAULT_TIMEOUT:.0f}，可用 LLAMA_CHAT_TIMEOUT 覆盖）")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="单轮回答上限；不给 = 不限制（答到 EOS）")
    ap.add_argument("--hide-thinking", action="store_true",
                    help="不打印思考内容（仅影响显示，请求照常开启思考）")
    ap.add_argument("--keep-thinking", action="store_true",
                    help="把思考块留在历史里（默认不留：它每轮都占满一个思考预算的上下文，"
                         "长会话轮数会被砍到五分之一左右）")
    ap.add_argument("--no-tui", action="store_true", help="禁用多行 TUI，用单行 + /paste")
    ap.add_argument("--mem", choices=["auto", "on", "off"], default="auto",
                    help="记忆注入：auto=关键词命中才注入（默认）· "
                         "on=强制注入全部记忆（A/B 对照用）· off=完全不注入（基线）")
    ap.add_argument("--once", metavar="问题", default=None,
                    help="非交互：只问这一句，答完即退（A/B 对照用）")
    ap.add_argument("--mem-dir", metavar="目录", default=None,
                    help="记忆库目录（默认 mem\\）。用于冗余度 A/B，如 --mem-dir mem\\rich")
    ap.add_argument("--save", metavar="文件", default=None,
                    help="把本轮最终回答原文写到 UTF-8 文件（配合 --once）")
    return ap


# --------------------------------------------------------------------------
# 记忆体（POC 起于 2026-09-25；2026-09-26 判决：**主载体用自然语言**）
#
#   条目格式   **自然语言**（推荐）。一份正常的说明文档就对了：
#                概述 → 分节细节 → 示例 → 例外，分层写，同一件事可以换句话
#                再说一遍（冗余是特性不是浪费）。理由：AI 是自然语言模型，
#                它的全部先验都在自然语言里；四元组是一套它没见过的记号，
#                读进来要先翻译一遍，而且"一行一事实"的格式天然禁止把同一
#                事实换个说法重述——那恰恰是最有效的加固手段。
#              实测（redundancy_ab，14 题×6 臂）：六行四元组骨架 0.787/11 命中，
#                改写成自然语言逐条展开 0.922/14 命中 ⇒ 赢的是表述方式与覆盖。
#              四元组（.quads）保留可用，它适合"一行一事实"的精确数字/坐标表。
#   存储位置   mem\ 下任意文本文件（.md 推荐 / .txt / .quads），递归；文件名随意
#   记忆来源   mem\ 下的文本（文本记忆）和图片（视觉记忆）—— 里面有什么就用什么，
#              换内容只需要换 mem 里的文件，不用改代码、不用复制
#   注入方式   改写**本轮** user content，不进 messages 历史 ⇒ 不污染后续轮
#
# 端到端链路：用户提问 → 加载 mem 里的记忆 → inject() 把记忆本体拼进本轮 content
# --------------------------------------------------------------------------
MEM_DIR = Path(__file__).resolve().with_name("mem")

# 文本记忆 = mem\ 下所有文本文件（递归）。换内容 = 换 mem 里那个文件，不用改代码。
# 格式不写死：.md 是推荐载体，.quads 是历史格式（精确表场景），两者都能读。
TEXT_SUFFIX = (".md", ".markdown", ".txt", ".quads")

# 注入的只有记忆本体：整段用 <knowledge> 圈起来，此外**一个字都不加**。
# 不写"下面是…记忆"、不写"以它为准"、不写"用户问题："、不写"必须…" —— 任何框架语、
# 分隔线、指令句都是在注入提示词，不是在注入记忆（模板一加，模型就去执行模板而不是照记忆）。
# 顺序：**用户原话在最前（任务是第一优先），记忆作为参考跟在后面**；然后是
# 文本在前、图在后（同一轮 content：用户原话 → knowledge → image）。
KNOW_OPEN = "<knowledge>"
KNOW_CLOSE = "</knowledge>"


# --------------------------------------------------------------------------
#   记忆开关 —— 就这一个变量，改数字就能切换，A/B 直接对比
#     0 = 无记忆      纯基线，什么都不注入
#     1 = 文本记忆    只注入 mem\ 下的文本文件（.md 推荐，四元组 .quads 也可）
#     2 = 图像记忆    只注入 mem\ 下那张图（视觉 token）        ← 当前启用
#     3 = 混合记忆    图 + 文本一起给
#                     ⚠ 实测两者会打架：判据题从 0.47 掉到 0.33，模型被文本
#                     记忆拉回标准开局 ⇒ 同一事实别两种形态同时给
#   不想改文件就临时用环境变量覆盖：  set LLAMA_CHAT_MEM=1
# --------------------------------------------------------------------------
try:
    MEM_MODE = int(os.environ.get("LLAMA_CHAT_MEM", "2"))
except ValueError:          # env 被写坏（空串/非数字）时退回默认，别崩在这
    MEM_MODE = 2

# 视觉记忆 = mem\ 下所有图片（递归）。不复制、不放别处：换内容就换 mem 里那张图。
IMG_SUFFIX = (".png", ".jpg", ".jpeg", ".webp", ".gif")

# 历史里是否保留图：True = 发出去的带图消息原样留在上下文里，后续轮次（"换个颜色"/"再改改"）
# 模型还看得见，不会变瞎。
# ⚠️ 不要"把旧图退化成占位文字"：改写已发出的历史 ⇒ prompt cache 全废 ⇒ 每轮全量重 prefill
#    （实测表观吞吐掉到 ~5 t/s）。代价是图会累积，靠 [ctx 水位] 提示盯着。
VISION_KEEP = True


def _data_url(path):
    import base64
    import mimetypes
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii"))


def load_vision_bank():
    """mem 下所有图片 → {文件名: data url}。换记忆 = 换 mem 里那张图。"""
    bank = {}
    if not MEM_DIR.is_dir():
        return bank
    for path in sorted(MEM_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMG_SUFFIX:
            continue
        url = _data_url(path)
        if url:
            bank[path.stem] = url
    return bank


def know_block(hits):
    """记忆本体：<knowledge> 圈住，此外一个字不加（不含用户原话）。"""
    return "%s\n%s\n%s" % (KNOW_OPEN, "\n\n".join(b for _, b in hits), KNOW_CLOSE)


def inject_vision(user_text, hits, text_block=None):
    """用户原话在最前（任务优先），随后文本记忆、再图。除 <knowledge> 外不加任何标记。"""
    if not hits:
        return user_text, []
    parts = [{"type": "text", "text": user_text}]
    if text_block:
        parts.append({"type": "text", "text": text_block})
    parts += [{"type": "image_url", "image_url": {"url": url}} for _, url in hits]
    return parts, [k for k, _ in hits]


def load_bank(mem_dir=MEM_DIR):
    """mem 下所有文本文件 → {文件名: 正文}。换记忆 = 换 mem 里那个文件。"""
    bank = {}
    if not mem_dir.is_dir():
        return bank
    for path in sorted(mem_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIX:
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            bank[path.stem] = text
    return bank


def inject(user_text, hits):
    """纯文本档：用户原话在最前，记忆本体跟在后面。"""
    if not hits:
        return user_text, []
    return user_text + "\n\n" + know_block(hits), [k for k, _ in hits]


def main(argv=None):
    a = _build_parser().parse_args(argv)

    if a.sys_file:
        try:
            # utf-8-sig：记事本另存为 UTF-8 with BOM 时自动剥掉 BOM，
            # 否则 BOM 会混进 system prompt 开头。
            with open(a.sys_file, encoding="utf-8-sig") as f:
                sys_prompt = f.read().strip()
        except OSError as e:
            print(f"[ERROR] 读不到 --sys-file {a.sys_file}：{e}")
            return 2
        if not sys_prompt:
            print(f"[ERROR] --sys-file 内容为空：{a.sys_file}")
            return 2
    else:
        sys_prompt = a.sys or DEFAULT_SYS

    sys_prompt = fill_sys(sys_prompt)      # {date} / {weekday}

    think_flag = _parse_think(a.think)
    timeout = a.timeout or DEFAULT_TIMEOUT
    base_url = a.base_url.rstrip("/")

    # 启动探活 + 显示服务端实际加载的模型
    ok, model_id, err = probe(base_url)
    think_txt = ("服务端默认" if think_flag is None else ("开" if think_flag else "关"))
    print(f"底座: {base_url}  "
          f"temp={a.temp if a.temp is not None else '服务端默认'}  "
          f"思考={think_txt}" + (f"({a.effort})" if a.effort else ""))
    if ok:
        print(f"服务端模型: {model_id or '(未报告)'}")
    else:
        print(f"[警告] 探活失败：{err}\n       （先双击 llmserver.bat / llmchat 对应的服务端再试）")

    print(f"\nsystem: {sys_prompt}")
    tip = ("粘贴大段文档自动整块提交；Esc+Enter 回车换行"
           if (_HAS_TUI and sys.stdin.isatty() and not a.no_tui)
           else "输入 '/paste' 粘贴大段文档")
    print(f"输入 'quit' 退出，'clear' 清空上下文，{tip}\n")

    session = None
    if _HAS_TUI and sys.stdin.isatty() and not a.no_tui:
        session = _new_session()

    messages = [{"role": "system", "content": sys_prompt}]
    mem_done = False        # 记忆只注入首轮（clear 后重置）

    # 记忆库：MEM_MODE 决定装哪一类；文本记忆还受 --mem 控制
    use_text = MEM_MODE in (1, 3)
    use_vision = MEM_MODE in (2, 3)
    _MODE_NAME = {0: "无记忆（纯基线）", 1: "文本记忆", 2: "图像记忆",
                  3: "混合记忆（图+文本，实测会打架）"}
    print(f"记忆模式: {MEM_MODE} = {_MODE_NAME.get(MEM_MODE, '未知')}")

    vbank = load_vision_bank() if use_vision else {}
    if vbank:
        n_img = len(set(vbank.values()))      # 多个触发词可能指向同一张图
        print(f"  视觉: {n_img} 张   命中词[{', '.join(sorted(vbank))}]")

    bank = {} if (not use_text or a.mem == "off") else load_bank(
        Path(a.mem_dir).expanduser().resolve() if a.mem_dir else MEM_DIR)
    if bank:
        n_lines = sum(t.count("\n") + 1 for t in bank.values())
        print(f"  文本: {len(bank)} 块 / {n_lines} 行   [{', '.join(bank)}]  注入={a.mem}")

    if not vbank and not bank:
        print("  （本模式没有加载到任何记忆）")
    print()

    while True:
        if a.once is not None:                  # 非交互单次模式
            user, once = a.once, True
            print(f"你> {user}")
        else:
            once = False
            try:
                user = read_user(session, "你> ")
            except (EOFError, KeyboardInterrupt):
                break
        if not user:
            break

        meta = _meta(user)                  # 元命令本地拦截，不进上下文
        if meta == "exit":
            break
        if meta == "clear":
            messages = [{"role": "system", "content": sys_prompt}]
            mem_done = False                # 清空上下文 ⇒ 允许下一轮重新做首轮注入
            print(f"上下文已清空（保留 system prompt，历史 {len(messages) - 1} 条）\n")
            continue

        # ---- 记忆注入（**只在首轮**）----
        # 首轮： 用户原话 → <knowledge> 记忆本体 → 图，一次性写进上下文；
        # 之后每轮只发原话，一律用普通字符串，不再附加任何记忆。
        # 理由：已发出的前缀一字不改 ⇒ prompt cache 稳定命中，上下文结构从第一轮
        # 起就定型；记忆也不会随轮数重复堆积。记忆仍在历史里，模型每轮都看得见。
        # 要换记忆内容：clear 后重来（或重开会话）。
        t_keys = []
        if mem_done:
            sent, keys = user, []
        elif use_vision:
            text_block = None
            if use_text and bank:
                text_block = know_block(list(bank.items()))
                t_keys = list(bank)
            sent, v_keys = inject_vision(user, list(vbank.items()), text_block)
            keys = t_keys + v_keys
        else:
            sent, keys = inject(user, list(bank.items()))
        if keys:
            if isinstance(sent, str):                       # 纯文本档
                note = "文本 %d 块 ~%d 字符" % (len(keys), len(sent))
            else:                                           # 多模态档：原话 + 文本 + 图
                n_img = sum(1 for p in sent if p.get("type") == "image_url")
                note = "图片 %d 张" % n_img
                if t_keys:
                    note += " + 文本 %d 块" % len(t_keys)
            print(f"[记忆命中·首轮] {', '.join(keys)}  → 注入 {note}\n")
        elif not mem_done and a.mem != "off":
            print("[记忆未命中] 首轮不注入\n")
        mem_done = True

        # 历史里存的是"实际发出去的那一版"：首轮带记忆本体/图，之后每轮就是原话。
        sent_msgs = messages + [{"role": "user", "content": sent}]
        try:
            printer = stream_printer(hide_thinking=a.hide_thinking)
            reasoning, content, timings = chat_sse(
                sent_msgs, base_url, a.temp, a.max_tokens,
                printer, think=think_flag, effort=a.effort, timeout=timeout)
        except ConnectionError as e:
            print(f"\n底座不可用：{e}\n")
            if once:
                return 1
            continue

        if printer.streamed["content"] or printer.streamed["thinking"]:
            print()                          # 流式已逐字印出，只补换行
        else:
            print(f"\n底座> {content}")

        # 回传思考块：服务端 --reasoning-preserve 生效的前提是客户端先把它发回去。
        # 视觉记忆留在历史里（否则"换个颜色""再改改"时模型已经看不见图了）。
        # 存进历史的一律是"实际发出去的那一版"，且此后永不改写 —— 改写已发出的消息会让
        # prompt cache 当轮全废、每轮全量重 prefill（实测表观吞吐掉到 ~5 t/s）。代价是
        # 图会累积，靠 [ctx 水位] 提示盯着，必要时 clear 重来。
        if VISION_KEEP and isinstance(sent, list):
            messages.append({"role": "user", "content": sent})
        else:
            messages.append({"role": "user", "content": user})
        # 思考块默认【不进历史】：它每轮都要吃掉整个思考长度（预算 4096 ⇒ 每轮 4096 tok），
        # 而下一轮模型真正需要的是自己说过的答案。不回传让上下文每轮只增长"输入+答案"，
        # 长会话的轮数直接翻几倍；前缀从第一轮起就固定，prompt cache 不受影响。
        # 思考过程照常流式打印，只是不留在历史里（--keep-thinking 恢复旧行为）。
        assistant = {"role": "assistant", "content": content}
        if reasoning and a.keep_thinking:
            assistant["reasoning_content"] = reasoning
        messages.append(assistant)

        if a.save:
            try:
                Path(a.save).write_text(content, encoding="utf-8")
                print(f"  [已保存] {a.save}")
            except OSError as e:
                print(f"  [保存失败] {e}")

        ctx = (timings.get("prompt_n") or 0) + (timings.get("cache_n") or 0)
        if ctx:
            print(f"  [ctx {ctx} tok · 水位 {100.0 * ctx / CTX_LIMIT:.0f}%]")
        # 速度：服务端 timings 里本来就有，之前只取了 prompt_n/cache_n 算水位，
        # 速度从没打印过 —— llmserver 只把它打在自己那个窗口里，这里看不到。
        # pp = 预填充（每轮新增的 prompt），tg = 逐字生成，就是体感速度。
        pp = timings.get("prompt_per_second")
        tg = timings.get("predicted_per_second")
        if pp or tg:
            seg = []
            if pp:
                seg.append(f"预填充 {pp:.1f} t/s ({timings.get('prompt_n') or 0} tok)")
            if tg:
                seg.append(f"生成 {tg:.1f} t/s ({timings.get('predicted_n') or 0} tok)")
            print("  [速度] " + " · ".join(seg))
        print()

        if once:
            return 0

    print("再见！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
