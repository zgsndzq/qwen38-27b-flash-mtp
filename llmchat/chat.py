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
import urllib.parse
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


def _seed_messages(sys_prompt):
    """开新上下文的消息表：有 system 才放 system（--no-sys 时是空表）。"""
    if sys_prompt is None:
        return []
    return [{"role": "system", "content": sys_prompt}]


DEFAULT_TIMEOUT = float(os.environ.get("LLAMA_CHAT_TIMEOUT", "600"))
# 水位提示的分母。以前写死 32768，服务端改成 65536 / 131072 之后它还在报 28% 那种
# 假水位。现在优先问服务端 /props 的 n_ctx，拿不到才退回环境变量。
CTX_LIMIT = int(os.environ.get("LLAMA_CHAT_CTX", "131072"))


def ctx_limit(base_url):
    """服务端真实 n_ctx；查不到就返回 CTX_LIMIT。只在第一次用到时问一次。"""
    global _CTX_PROBED
    if _CTX_PROBED is None:
        _CTX_PROBED = 0
        try:
            with urllib.request.urlopen(base_url + "/props", timeout=3) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            n = (d.get("default_generation_settings") or {}).get("n_ctx") or d.get("n_ctx")
            if n:
                _CTX_PROBED = int(n)
        except Exception:
            pass
        if not _CTX_PROBED:
            _CTX_PROBED = CTX_LIMIT
    return _CTX_PROBED


_CTX_PROBED = None

EXIT_CMDS = ("quit", "exit", "q", "/quit", "/exit", "退出")
CLEAR_CMDS = ("clear", "/clear", "cls", "/cls", "清空", "清除上下文")
PASTE_CMDS = ("/paste", "/多行")
END_CMDS = ("/end", ":::end")

# When stdout is a file, force UTF-8: Windows' default cp936 mangles non-ASCII.
# A real TTY is left alone for Python's console writer.
if not sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Interactive input. prompt_toolkit is optional; falls back to plain input().
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
# llama-server client: streaming core inlined, stdlib only.
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
             think=None, effort=None, timeout=DEFAULT_TIMEOUT, id_slot=None, cache_prompt=None):
    """流式 chat：SSE 逐增量回调 on_delta(kind, text)。

    - temperature 传数字会顶掉服务端 --temp；None = 不下发，沿用服务端
    - think=True/False → chat_template_kwargs.enable_thinking（覆盖服务端 -rea）
    - effort="xhigh"|"medium"|"low" → chat_template_kwargs.reasoning_effort
    返回 (reasoning, content) 完整拼接。
    """
    body = {"model": "gguf", "messages": messages, "stream": True}
    if id_slot is not None:
        body["id_slot"] = int(id_slot)     # 钉在指定槽：双会话各自绑定，KV 才能复用
    if cache_prompt is not None:
        body["cache_prompt"] = bool(cache_prompt)
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
        elif "400" in str(e):
            hint = ("\n服务端拒绝了请求（400）。最常见的原因：当前会话长度超过了服务端单槽的"
                    "上下文容量（比如双槽 32K 时每槽只有 16384 tok）。\n"
                    "在 llmchat 里输入 clear 清空本会话即可继续；或把服务端切回大上下文配置。")
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
    ap.add_argument("--no-sys", action="store_true",
                    help="完全不下发 system 消息：前缀里只有用户消息（纯模型调用）")
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
# Memory. Source is whatever sits in mem\\ (text files and images): change the
# files, not the code. It is written into THIS turn's content only and never into
# message history, so later turns are unaffected.
#
# Use prose, not records. Measured on redundancy_ab (14 questions x 6 arms): a
# six-line quad skeleton scored 0.787 / 11 hits, the same facts as prose scored
# 0.922 / 14 hits. Prose wins because the model's priors live in natural language,
# and a "one fact per line" format structurally forbids restating a fact in other
# words - which is the strongest reinforcement available.
#
#   Layout    any text file under mem\, read recursively (.md recommended;
#             .txt and .quads also work). File names do not matter.
#             .quads is kept for exact number/coordinate tables, one fact per line.
#   Images    any image under mem\ is a vision-memory candidate.
#   Inject    rewrites this turn's user content; never enters messages history.
#
# End to end: user asks -> memory loaded from mem\\ -> inject() splices the memory
# body into this turn's content.
# --------------------------------------------------------------------------
MEM_DIR = Path(__file__).resolve().with_name("mem")

# --------------------------------------------------------------------------
# 会话落盘 + KV 槽位缓存（重启后秒恢复）
#
# 服务端用 --slot-save-path <dir> 启动时，/slots/0?action=save|restore|erase 能把
# 整段 KV（含 48 层 GDN 的递推状态和 MTP 草稿上下文）存成文件再读回来。重启后
# 继续同一段对话只读回槽位、不重算整段 prompt：16k 前缀约从 50 s 变成 1 s 级。
# 前提是**新请求必须是已存 token 的延长**；中间改过历史就退化成全量重读，不会出错。
# --------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"          # 运行数据（不属于 logs/，删 logs 不影响）
SESSION_FILE = DATA_DIR / "llmchat_session.json"
SLOT_NAME = "llmchat.slot"
SLOT_FILE = Path(__file__).resolve().parent.parent / "data" / "slots" / SLOT_NAME
SLOT_MIN = int(os.environ.get("LLAMA_SLOT_MIN", "8192"))   # 短于这个长度不值得来回存盘
SLOT_GROW = int(os.environ.get("LLAMA_SLOT_GROW", "2048"))  # 上下文长这么多才重存一次（439MB 不便宜）
_slot_off = False                                          # 服务端没配 --slot-save-path 时置位


def apply_profile(base_url):
    """按端口给会话与 KV 存档加后缀，两个模型的对话不会混在同一个文件里。

    18080（默认端口）保持原文件名，老会话不受影响；其它端口（18081 = LFM）加 _<port>。
    参数是已经 rstrip('/') 过的 base_url。
    """
    global SESSION_FILE, SLOT_NAME, SLOT_FILE
    try:
        port = str(urllib.parse.urlparse(base_url).port or "")
    except ValueError:
        port = ""
    if not port or port == "18080":
        return
    SESSION_FILE = DATA_DIR / f"llmchat_session_{port}.json"
    SLOT_NAME = f"llmchat_{port}.slot"
    SLOT_FILE = Path(__file__).resolve().parent.parent / "data" / "slots" / SLOT_NAME


def slot_action(base_url, action, filename=None, timeout=180, slot_id=0):
    """POST /slots/0?action=...；文件名走 JSON body（这个构建的路由是 POST，GET 会 404）。"""
    global _slot_off
    if _slot_off and action != "erase":
        return None
    url = f"{base_url}/slots/{int(slot_id)}?action={action}"
    body = json.dumps({"filename": filename} if filename else {}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "{}")
    except Exception as e:
        # restore 失败很常见（第一次跑、文件被删），不能因此把 save 也关掉
        if action == "save" and not _slot_off:
            _slot_off = True
            print(f"  [槽位] 保存失败：{e}")
        elif action == "restore":
            print(f"  [槽位] 读回未生效：{e}")
        return None


def session_save(messages, tokens, slot_tokens=None):
    """会话落盘。slot_tokens = 上次成功存槽位时的 token 数，用来决定下次要不要重存。"""
    try:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"messages": messages, "tokens": int(tokens), "slot": SLOT_NAME}
        if slot_tokens is not None:
            data["slot_tokens"] = int(slot_tokens)
        else:
            old = session_load() or {}
            if old.get("slot_tokens"):
                data["slot_tokens"] = int(old["slot_tokens"])
        SESSION_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def session_load():
    try:
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def session_clear(base_url):
    try:
        SESSION_FILE.unlink()
    except OSError:
        pass
    slot_action(base_url, "erase")

# Text memory = every text file under mem\, recursively.
TEXT_SUFFIX = (".md", ".markdown", ".txt", ".quads")

# Only the raw memory goes in, wrapped in <knowledge> and nothing else. No lead-in
# ("here is your memory"), no "follow this", no "the user asks:" - any framing text
# makes the model execute the framing instead of reading the memory.
# Order: user's own words first (the task has priority), memory after as reference;
# within the turn: original text -> <knowledge> -> image.
KNOW_OPEN = "<knowledge>"
KNOW_CLOSE = "</knowledge>"


# --------------------------------------------------------------------------
#   MEM_MODE  the single switch; change the number to A/B compare.
#     0 = none      pure baseline, nothing injected
#     1 = text      text files under mem\ only
#     2 = vision    a single image from mem\ (vision tokens)     <- current
#     3 = mixed     image + text together
#                   WARNING measured to fight itself: the JEV probe drops from
#                   0.47 to 0.33 - the text pulls the model back to a standard
#                   opening. Never give one fact in two shapes at once.
#   Override without editing:   set LLAMA_CHAT_MEM=1
# --------------------------------------------------------------------------
try:
    MEM_MODE = int(os.environ.get("LLAMA_CHAT_MEM", "2"))
except ValueError:          # env 被写坏（空串/非数字）时退回默认，别崩在这
    MEM_MODE = 2

# Vision memory = every image under mem\, recursively. No copies anywhere else.
IMG_SUFFIX = (".png", ".jpg", ".jpeg", ".webp", ".gif")

# Keep images in history: the message that was sent stays in the context verbatim,
# so the model can still see the picture on later turns ("change the colour").
# Do NOT replace old images with placeholder text: rewriting an already-sent message
# invalidates the whole prompt cache and re-prefills everything each turn (measured
# apparent throughput falls to ~5 t/s). The price of keeping them is that images
# accumulate - watch the [ctx] watermark.
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

    if a.no_sys:
        # Pure model call: build no system message and read no sys file.
        # None means "no system" - an empty string would fall back to DEFAULT_SYS
        # through `a.sys or DEFAULT_SYS`.
        sys_prompt = None
    elif a.sys_file:
        try:
            # utf-8-sig: strip the BOM that Notepad writes for "UTF-8 with BOM",
            # otherwise it ends up at the front of the system prompt.
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

    if sys_prompt is not None:
        sys_prompt = fill_sys(sys_prompt)  # {date} / {weekday}

    think_flag = _parse_think(a.think)
    timeout = a.timeout or DEFAULT_TIMEOUT
    base_url = a.base_url.rstrip("/")
    apply_profile(base_url)

    # Probe the server and show the model it actually loaded.
    ok, model_id, err = probe(base_url)
    think_txt = ("服务端默认" if think_flag is None else ("开" if think_flag else "关"))
    print(f"底座: {base_url}  "
          f"temp={a.temp if a.temp is not None else '服务端默认'}  "
          f"思考={think_txt}" + (f"({a.effort})" if a.effort else ""))
    if ok:
        print(f"服务端模型: {model_id or '(未报告)'}")
    else:
        print(f"[警告] 探活失败：{err}\n       （先双击 llmserver.bat / llmchat 对应的服务端再试）")

    print("\nsystem: %s" % (sys_prompt if sys_prompt is not None else "(无 — 纯模型调用)"))
    tip = ("粘贴大段文档自动整块提交；Esc+Enter 回车换行"
           if (_HAS_TUI and sys.stdin.isatty() and not a.no_tui)
           else "输入 '/paste' 粘贴大段文档")
    print(f"输入 'quit' 退出，'clear' 清空上下文，{tip}\n")

    session = None
    if _HAS_TUI and sys.stdin.isatty() and not a.no_tui:
        session = _new_session()

    messages = _seed_messages(sys_prompt)
    mem_done = False        # 记忆只注入首轮（clear 后重置）

    # 上次的会话：消息 + 槽位。恢复后 mem_done=True，避免把记忆再注入一遍。
    saved = session_load()
    need_restore = False
    if saved and saved.get("messages"):
        messages = saved["messages"]
        mem_done = True
        need_restore = True
        print(f"已恢复上次会话：{len(messages)} 条消息、约 {saved.get('tokens') or 0} tok"
              f"（输入 clear 重新开始）\n")

    # MEM_MODE picks the kind; text memory is additionally gated by --mem.
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
            messages = _seed_messages(sys_prompt)
            mem_done = False                # 清空上下文 ⇒ 允许下一轮重新做首轮注入
            session_clear(base_url)         # 落盘的会话和槽位一起清掉
            _n_sys = 0 if sys_prompt is None else 1
            _tail = "，无 system" if sys_prompt is None else ""
            print(f"上下文已清空（历史 {len(messages) - _n_sys} 条{_tail}；"
                  f"输入历史已清）\n")
            continue

        # ---- memory injection: FIRST TURN ONLY ----
        # Turn 1 writes user text -> <knowledge> -> image into the context once.
        # Later turns send the user's words alone, as a plain string, with no memory
        # attached. Reason: an unchanged prefix keeps the prompt cache hit every turn,
        # and the memory never piles up. It stays in history, so the model still sees
        # it. To change the memory, clear (or restart the session).
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

        # History keeps exactly what was sent: memory + image on turn 1, plain
        # user text afterwards.
        sent_msgs = messages + [{"role": "user", "content": sent}]

        # 上次的槽位只在恢复后的第一轮读一次；服务端会取最长公共前缀，改过历史
        # 就自动退化成重算，不会读进错误的 KV。
        if need_restore:
            need_restore = False
            # 文件不在就别发请求了：服务端会回 400「No available space in KV cache」，
            # 那条日志很容易被误读成 KV 满了。
            if (saved.get("tokens") or 0) >= SLOT_MIN and SLOT_FILE.exists():
                r = slot_action(base_url, "restore", SLOT_NAME)
                if r and r.get("n_restored"):
                    print(f"  [槽位] 读回 {r['n_restored']} tok，本轮只算新增部分")
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

        # Send the thinking block back: the server's --reasoning-preserve only works
        # if the client returns it. Images stay in history so later turns can still
        # see them. What goes into history is always the version that was actually
        # sent, and it is never rewritten afterwards - rewriting a sent message
        # destroys the prompt cache and re-prefills every turn (~5 t/s measured).
        if VISION_KEEP and isinstance(sent, list):
            messages.append({"role": "user", "content": sent})
        else:
            messages.append({"role": "user", "content": user})
        # The thinking block does NOT go into history by default: it costs its full
        # length every turn (budget 4096 -> 4096 tokens/turn), while the next turn
        # only really needs the answer. Dropping it means the context grows by
        # input+answer per turn, which multiplies how many turns fit in a session;
        # the prefix is fixed from turn 1 so the prompt cache still hits.
        # It is still streamed to the screen; --keep-thinking restores the old way.
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
            limit = ctx_limit(base_url)
            print(f"  [ctx {ctx} tok · 水位 {100.0 * ctx / limit:.0f}% / {limit}]")
            # 会话落盘 + 长前缀存槽位：下次启动（或换个会话再回来）直接读回，
            # 不用重算整段 prompt。短于 SLOT_MIN 的不存；上下文没长出 SLOT_GROW
            # 也不重存——一个槽位文件 28 KB/token，16k 就 440 MB。
            session_save(messages, ctx)
            prev = int(saved.get("slot_tokens") or 0) if isinstance(saved, dict) else 0
            if ctx >= SLOT_MIN and (not SLOT_FILE.exists() or ctx - prev >= SLOT_GROW):
                r = slot_action(base_url, "save", SLOT_NAME)
                if r and r.get("n_saved"):
                    if not isinstance(saved, dict):
                        saved = {}
                    saved["slot_tokens"] = ctx
                    session_save(messages, ctx, slot_tokens=ctx)
                    print(f"  [槽位] 已存 {r['n_saved']} tok"
                          f"（{r.get('n_written', 0) / 1e6:.0f} MB，"
                          f"{r.get('timings', {}).get('save_ms', 0):.0f} ms）")
        # Speed: the server already reports it in timings, but the launcher only
        # prints it in its own window. pp = prefill (new prompt this turn),
        # tg = token generation, which is the speed you actually feel.
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
