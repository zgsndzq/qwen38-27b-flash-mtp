# -*- coding: utf-8 -*-
"""A/B 对照：同一个 prompt，一次不带记忆、一次带记忆，左右两栏输出。

不带历史、不落 session，纯单轮对比。依赖引擎的请求级开关：
    features.ini  LLAMA_MEM_REQ_TOGGLE = 1
    请求体        {"mem": false}   -> 跳过注入

用法：
    llmchat-mem.bat                     交互，输入一行 prompt
    llmchat-mem.bat -f prompts.txt      每行一个 prompt，批量跑
    llmchat-mem.bat --code              代码任务采样档（低温 + 轻惩罚）
    llmchat-mem.bat --stack             窄终端：上下两段而不是左右两栏
"""

import argparse
import json
import os
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

if os.name == "nt":
    os.system("")  # 打开 ANSI 转义

CFG = Path(__file__).resolve().parent / "config.json"

DIM, BOLD, RESET = "\x1b[90m", "\x1b[1m", "\x1b[0m"
CYAN, YELLOW = "\x1b[36m", "\x1b[33m"


def cwidth(s):
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def wrap(text, width):
    """按显示宽度折行（中文按 2 列算），保留原有换行"""
    out = []
    for para in text.splitlines():
        if not para:
            out.append("")
            continue
        line, w = [], 0
        for ch in para:
            cw = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
            if w + cw > width and line:
                out.append("".join(line))
                line, w = [], 0
            line.append(ch)
            w += cw
        out.append("".join(line))
    return out


def pad(s, width):
    d = width - cwidth(s)
    return s + " " * max(d, 0)


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 本机服务，绕开代理


def post(url, payload, timeout=600):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with OPENER.open(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    dt = time.time() - t0
    msg = data["choices"][0]["message"]
    text = msg.get("content") or ""
    usage = data.get("usage") or {}
    mem = data.get("memory") or {}
    return text, dt, usage, mem


def ask(base, prompt, mem, args):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "seed": args.seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": args.think == "on"},
    }
    if not mem:
        payload["mem"] = False
    return post(base.rstrip("/") + "/v1/chat/completions", payload)


def render_columns(left, right, lw, rw, lname, rname):
    lrows, rrows = wrap(left, lw), wrap(right, rw)
    h = max(len(lrows), len(rrows))
    head = (BOLD + CYAN + pad("─ " + lname, lw) + RESET + " │ "
            + BOLD + YELLOW + pad("─ " + rname, rw) + RESET)
    print(head)
    for i in range(h):
        a = lrows[i] if i < len(lrows) else ""
        b = rrows[i] if i < len(rrows) else ""
        print(pad(a, lw) + " │ " + b)
    print(DIM + "─" * lw + "─┼─" + "─" * rw + RESET)


def show_memory(mem):
    """把这一轮注入的记忆打印出来，便于逐个对照"""
    ents = mem.get("entries") or []
    if not ents:
        print(DIM + "  [记忆] 这一轮没有命中任何条目" + RESET)
        return
    print(f"{BOLD}{YELLOW}  [记忆] 注入 {mem.get('tokens', 0)} tok / {len(ents)} 条{RESET}")
    for e in ents:
        name = (e.get("title") or "").strip() or f"#{e.get('idx', '?')}"
        if e.get("pinned"):
            tag = f"{DIM}pinned{RESET}"
        else:
            tag = f"{DIM}score {e.get('score', 0):.3f}{RESET}"
        print(f"    · {name}  {tag}  {e.get('tokens', 0)} tok")
    pv = (mem.get("preview") or "").strip()
    if pv:
        w = os.get_terminal_size().columns if sys.stdout.isatty() else 100
        print(DIM + "    ── 注入内容（全文）" + RESET)
        for ln in pv.splitlines():
            for row in wrap(ln, max(40, w - 6)) or [""]:
                print(DIM + "    " + row + RESET)


def one_round(base, prompt, args):
    print(f"\n{BOLD}Q{RESET} {prompt}")
    ltext, ltime, lusage, _ = ask(base, prompt, False, args)
    rtext, rtime, rusage, rmem = ask(base, prompt, True, args)

    term = os.get_terminal_size().columns if sys.stdout.isatty() else 100
    if args.stack or term < 90:
        print(f"\n{BOLD}{CYAN}[ 无记忆 ]{RESET}")
        print(ltext)
        print(f"\n{BOLD}{YELLOW}[ 有记忆 ]{RESET}")
        print(rtext)
    else:
        colw = max(28, (min(term, args.max_width) - 3) // 2)
        render_columns(ltext, rtext, colw, colw, "无记忆", "有记忆")

    def stat(t, u):
        tok = u.get("completion_tokens") or 0
        return f"{t:5.1f}s · {tok:4d} tok · {tok / max(t, 0.01):4.0f} t/s"

    print(DIM + "  " + pad(stat(ltime, lusage), 40) + "  " + stat(rtime, rusage) + RESET)
    show_memory(rmem)
    if rtext == ltext:
        print(DIM + "  (两侧输出完全一致 —— 这条 prompt 没有触发记忆)" + RESET)


def main():
    cfg = {}
    if CFG.exists():
        try:
            cfg = json.loads(CFG.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=cfg.get("base_url", "http://127.0.0.1:18080"))
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--presence-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=-1, help="-1 = 不限制，直到模型自己收尾")
    ap.add_argument("--max-width", type=int, default=140)
    ap.add_argument("--think", default="off", choices=["on", "off"])
    ap.add_argument("--stack", action="store_true")
    ap.add_argument("--code", action="store_true", help="代码档：低温、轻惩罚")
    ap.add_argument("-f", "--file", default="")
    args = ap.parse_args()

    if args.code:
        args.temp, args.presence_penalty, args.top_p = 0.2, 0.3, 0.95

    base = args.base_url
    print(f"{DIM}server {base} · temp {args.temp} · seed {args.seed} · max_tokens {args.max_tokens}{RESET}")

    if args.file:
        for line in Path(args.file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                one_round(base, line.strip(), args)
        return

    print(DIM + "输入 prompt 回车；空行退出" + RESET)
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        one_round(base, q, args)


if __name__ == "__main__":
    main()
