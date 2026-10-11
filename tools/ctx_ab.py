"""Diagnose the "second turn starts ~30% slower, then recovers" effect.

For each turn of a cumulative conversation we record:

  - what the server charged to the *prompt*  (prompt_n / prompt_ms / cache_n)
  - time to first token
  - per-segment decode speed inside one response (tokens 1-64, 65-192, 193+)

The per-segment split is the whole point. A single end-to-end number cannot tell
apart three very different causes:

  1. prompt cost leaking into the average  -> impossible here, the server's
     `predicted_per_second` divides by (t_gen_last - t_prompt_last), so prompt
     time is excluded. If the first segment is slow anyway, it is real decode.
  2. long-context attention getting dearer  -> every segment slows down roughly
     proportionally, no recovery inside a turn.
  3. per-turn warm-up (CUDA graph capture, cache occupancy, slot reset) -> the
     first segment is slow and later ones recover.

Usage:
    python tools/ctx_ab.py [--rounds 5] [--max-tokens 384] [--runs 2]
                           [--no-history] [--url http://127.0.0.1:18080]
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

# Five ordinary Chinese questions; each answer becomes part of the next prompt.
QUESTIONS = [
    "用中文简单说明一下 ring buffer 为什么要用取模运算。",
    "那如果读写指针都用无符号整数溢出代替取模，会有什么坑？",
    "再说说它在多线程无锁队列里通常怎么落地。",
    "顺带解释一下伪共享对这个结构的影响。",
    "最后给一段最小的 C 语言实现，带中文注释。",
]


def post_stream(url, payload):
    """Yield (wall_time, n_chunks, timings_or_None, delta_text) per SSE chunk."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json; charset=utf-8",
                 "Accept": "text/event-stream"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        n = 0
        timings = None
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            n += 1
            if "timings" in obj:
                timings = obj["timings"]
            delta = ""
            ch = obj.get("choices") or [{}]
            msg = ch[0].get("delta") if ch else None
            if isinstance(msg, dict):
                delta = msg.get("content") or ""
            yield time.perf_counter() - t0, n, timings, delta
        if timings is None:
            yield time.perf_counter() - t0, n, None


def one_turn(url, messages, max_tokens, cache_prompt):
    payload = {
        "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "cache_prompt": cache_prompt, "stream": True,
    }
    stamps = [(0.0, 0)]
    text_parts = []
    last_timings = None
    for t, n, timings, delta in post_stream(url, payload):
        stamps.append((t, n))
        if delta:
            text_parts.append(delta)
        if timings:
            last_timings = timings
    return stamps, last_timings, "".join(text_parts)


def seg_rate(stamps, lo, hi):
    """Tokens per second between cumulative token index lo and hi."""
    pts = {n: t for t, n in stamps}
    ks = sorted(pts)
    if not ks or ks[-1] <= lo:
        return None
    t_lo, t_hi = None, None
    prev = None
    for k in ks:
        if prev is not None and prev < lo <= k:
            t_lo = pts[prev] + (pts[k] - pts[prev]) * (lo - prev) / (k - prev)
        if prev is not None and prev < hi <= k:
            t_hi = pts[prev] + (pts[k] - pts[prev]) * (hi - prev) / (k - prev)
        prev = k
    if t_lo is None:
        t_lo = pts[ks[0]]
    if t_hi is None:
        t_hi = pts[ks[-1]]
    dt = t_hi - t_lo
    return (hi - lo) / dt if dt > 0 else None


def acc_of(timings):
    if not timings:
        return None
    d = timings.get("draft_n") or 0
    a = timings.get("draft_n_accepted") or 0
    return (a / d) if d else None


def make_row(qi, stamps, timings, extra=None):
    total_tok = stamps[-1][1]
    ttft = stamps[1][0] if len(stamps) > 1 else None
    tm = timings or {}
    row = {
        "turn": qi,
        "prompt_n": tm.get("prompt_n") or 0,
        "cache_n": tm.get("cache_n") or 0,
        "prompt_ms": tm.get("prompt_ms") or 0.0,
        "pp_tps": tm.get("prompt_per_second") or 0.0,
        "ttft": ttft,
        "pred_n": tm.get("predicted_n") or total_tok,
        "server_tps": tm.get("predicted_per_second") or 0.0,
        "seg1": seg_rate(stamps, 1, min(64, total_tok)),
        "seg2": seg_rate(stamps, 64, min(192, total_tok)),
        "seg3": seg_rate(stamps, 192, total_tok),
        "accept": acc_of(tm),
    }
    if extra:
        row.update(extra)
    return row


def run_pass(url, questions, max_tokens, keep_history, cache_prompt):
    history = []
    rows = []
    for qi, q in enumerate(questions, 1):
        messages = list(history) + [{"role": "user", "content": q}]
        stamps, timings, answer = one_turn(url, messages, max_tokens, cache_prompt)
        rows.append(make_row(qi, stamps, timings))
        if keep_history:
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": answer or ""})
    return rows


# ----------------------------------------------------------- padded-prefix mode
# The conversation above grows the context in "realistic" chunks. That is noisy:
# every turn differs in content, so a 20% gap is hard to attribute. This mode
# pins the context length itself -- one question, several prefix sizes -- so the
# only variable left is how many KV entries attention has to walk per step.

FILLER_UNIT = (
    "下面是一段用于填充上下文的参考资料，它本身不需要你处理、总结或回应，"
    "只是为了让这次请求的输入足够长。\n"
    "A ring buffer is a fixed-size queue backed by a contiguous array. When either "
    "index reaches the end it wraps around to the beginning, which is normally done "
    "with a modulo operation. On power-of-two capacities the compiler turns the "
    "modulo into a bitwise AND, which is why every serious implementation rounds the "
    "capacity up. The cost is memory you paid for but cannot address.\n"
    "def push(self, item):\n"
    "    if (self.head + 1) & self.mask == self.tail:\n"
    "        raise BufferError('ring buffer is full')\n"
    "    self.buf[self.head] = item\n"
    "    self.head = (self.head + 1) & self.mask\n"
    "    return True\n"
)


def tokenize_len(url, text):
    body = json.dumps({"content": text}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url + "/tokenize", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=120) as r:
        obj = json.load(r)
    return len(obj.get("tokens") or [])


def pad_to(url, question, target):
    if target <= 0:
        return question
    unit = tokenize_len(url, FILLER_UNIT) or 1
    text = FILLER_UNIT * max(1, target // unit + 1)
    for _ in range(8):
        cur = tokenize_len(url, text + "\n\n" + question)
        if cur <= target * 1.02:
            break
        cut = int(len(text) * (target / max(cur, 1)) * 0.97)
        text = text[:max(cut, 1)]
    return text + "\n\n" + question


def run_padded(url, question, targets, max_tokens, cache_prompt):
    rows = []
    for i, target in enumerate(targets, 1):
        content = pad_to(url, question, target)
        messages = [{"role": "user", "content": content}]
        stamps, timings, _ = one_turn(url, messages, max_tokens, cache_prompt)
        rows.append(make_row(i, stamps, timings, {"target": target}))
    return rows


def summarize(all_runs, n_turns):
    print("%-5s %8s %8s %9s %8s %9s %9s %9s %9s %9s" % (
        "turn", "prompt_n", "cache_n", "prefill_s", "TTFT_s", "seg1-64", "seg65-192",
        "seg193+", "server", "accept"))
    for ti in range(n_turns):
        group = [r[ti] for r in all_runs if len(r) > ti]
        if not group:
            continue
        med = lambda k: statistics.median([g[k] for g in group if g[k] is not None]) \
            if any(g[k] is not None for g in group) else None
        f = lambda v, s="%9.2f": (s % v) if v is not None else "%9s" % "-"
        print("%-5d %8d %8d %9.2f %8s %s %s %s %9.2f %9s" % (
            group[0]["turn"],
            int(med("prompt_n") or 0),
            int(med("cache_n") or 0),
            (med("prompt_ms") or 0) / 1000.0,
            f(med("ttft"), "%8.2f"),
            f(med("seg1")),
            f(med("seg2")),
            f(med("seg3")),
            med("server_tps") or 0.0,
            f(med("accept"), "%9.3f"),
        ))
    print("\n  server = /v1 timings.predicted_per_second (pure decode, prompt excluded)")
    print("  seg*   = wall-clock speed measured from streamed chunk timestamps")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--no-history", action="store_true",
                    help="every turn starts from a clean context (control arm)")
    ap.add_argument("--cache-prompt", action="store_true",
                    help="let the server reuse KV across turns")
    ap.add_argument("--pad-tokens", default=None,
                    help="comma list of context sizes; each size is one single-turn "
                         "request padded to that many prompt tokens")
    args = ap.parse_args(argv)

    py = None
    if args.pad_tokens:
        targets = [int(x) for x in args.pad_tokens.split(",") if x.strip()]
        all_runs = []
        for i in range(args.runs):
            rows = run_padded(args.url, QUESTIONS[0], targets,
                              args.max_tokens, args.cache_prompt)
            all_runs.append(rows)
            print("  [pass %d/%d done]" % (i + 1, args.runs), file=sys.stderr)
        print("\n=== SAME QUESTION, PINNED CONTEXT SIZE ===")
        print("  (turn = index into --pad-tokens; prompt_n is what the server saw)")
        summarize(all_runs, len(targets))
        return 0

    qs = QUESTIONS[:args.rounds]
    all_runs = []
    for i in range(args.runs):
        all_runs.append(run_pass(args.url, qs, args.max_tokens,
                                 not args.no_history, args.cache_prompt))
        print("  [pass %d/%d done]" % (i + 1, args.runs), file=sys.stderr)
    print("\n=== %s ===" % ("CLEAN CONTEXT (control)" if args.no_history
                            else "CUMULATIVE CONTEXT (what you are seeing)"))
    summarize(all_runs, len(qs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
