#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does enabling --reasoning-preserve actually cost anything?

Two things have to line up for a reasoning trace to survive into the next turn:

  1. the server must be told to keep it          (--reasoning-preserve)
  2. the CLIENT must send it back                (assistant msg with
                                                 "reasoning_content")

Server-side flags alone are inert if the client drops the trace, which is
exactly what llmchat\\chat.py line ~341 does today: it appends only
{"role":"assistant","content": content}.

So this probe measures all three combinations and reports what each costs:
turn-by-turn TTFT, prompt tokens charged per turn, and how fast the context
grows (which is what eventually hits both TTFT and the 32k context cap).

Usage:  python keep_reasoning.py --rounds 4 [--max-tokens 512]
The server must already be running; nothing here starts or kills it.
"""
import argparse
import json
import sys
import time
import urllib.request

QUESTIONS = [
    "请推导一下：为什么二叉搜索树退化为链表后查找复杂度变成 O(n)？给我一个具体的插入顺序例子。",
    "把这个例子改成红黑树会怎样？只需要讲关键的旋转发生在哪一步。",
    "那么 AVL 和红黑树在这个插入顺序下的旋转次数差多少？",
    "最后用三句话总结我该怎么选型。",
]


def post(url, payload, timeout=3600):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read().decode("utf-8", "replace"))
    return time.perf_counter() - t0, obj


def run(url, rounds, max_tokens, keep_reasoning):
    """keep_reasoning=False reproduces what llmchat/chat.py does today."""
    msgs = []
    rows = []
    for q in QUESTIONS[:rounds]:
        msgs.append({"role": "user", "content": q})
        payload = {
            "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "cache_prompt": True, "stream": False,
        }
        wall, obj = post(url, payload)
        msg = (obj.get("choices") or [{}])[0].get("message") or {}
        tm = obj.get("timings") or {}
        reasoning = msg.get("reasoning_content") or ""
        answer = msg.get("content") or ""
        rows.append({
            "turn": len(rows) + 1,
            "prompt_n": tm.get("prompt_n") or 0,
            "cache_n": tm.get("cache_n") or 0,
            "ctx": (tm.get("prompt_n") or 0) + (tm.get("cache_n") or 0),
            "prefill_s": round((tm.get("prompt_ms") or 0) / 1000.0, 2),
            "wall_s": round(wall, 2),
            "reason_chars": len(reasoning),
            "answer_chars": len(answer),
            "pred_n": tm.get("predicted_n") or 0,
            "tps": round(tm.get("predicted_per_second") or 0, 2),
            "sent_reason": keep_reasoning and len(rows) > 0,
        })
        assistant = {"role": "assistant", "content": answer}
        if keep_reasoning and reasoning:
            assistant["reasoning_content"] = reasoning
        msgs.append(assistant)
    return rows


def show(title, rows):
    print("\n=== %s ===" % title)
    print("%5s %9s %9s %8s %10s %8s %7s %9s %8s" %
          ("turn", "prompt_n", "cache_n", "ctx", "prefill_s", "wall_s",
           "tps", "reason_c", "sent"))
    for r in rows:
        print("%5d %9d %9d %8d %10.2f %8.2f %7.2f %9d %8s" %
              (r["turn"], r["prompt_n"], r["cache_n"], r["ctx"],
               r["prefill_s"], r["wall_s"], r["tps"],
               r["reason_chars"], "yes" if r["sent_reason"] else "no"))
    if len(rows) > 1:
        first, last = rows[0]["prefill_s"], rows[-1]["prefill_s"]
        growth = (rows[-1]["ctx"] - rows[0]["ctx"]) / max(len(rows) - 1, 1)
        print("  turn1 prefill %.2fs -> turn%d prefill %.2fs | ctx grows ~%.0f tok/turn"
              % (first, len(rows), last, growth))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--mode", default="both",
                    choices=["drop", "keep", "both"],
                    help="drop = client does not resend reasoning (today's chat.py); "
                         "keep = client resends it")
    args = ap.parse_args(argv)

    modes = []
    if args.mode in ("drop", "both"):
        modes.append(("CLIENT DROPS reasoning (what llmchat/chat.py does today)", False))
    if args.mode in ("keep", "both"):
        modes.append(("CLIENT RESENDS reasoning_content", True))

    for title, keep in modes:
        rows = run(args.url, args.rounds, args.max_tokens, keep)
        show(title, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
