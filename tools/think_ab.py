"""think_ab.py -- does thinking cost throughput? A/B against a LIVE server.

The question it answers is not "is thinking slower per answer" (it is, trivially:
it emits more tokens) but "is thinking slower PER TOKEN". If reasoning text runs
at the same tokens/second as ordinary text, then a thinking answer costs only its
extra tokens and nothing else -- which is exactly what you want, because real
traffic spends 62-69% of its output inside the reasoning block.

It attaches to an already-running server and never restarts it: thinking is
toggled per request with chat_template_kwargs.enable_thinking (the launcher's
`-rea` only sets the default). So this can be run while the server window is
being watched, and it will not kill anyone's process.

Usage:
  python think_ab.py                        # 3 prompts x 3 runs, both modes
  python think_ab.py --runs 5 --max-tokens 512
  python think_ab.py --prompts code --url http://127.0.0.1:18080

Exit code 0 = both modes produced data.
"""

import argparse
import json
import statistics
import sys
import urllib.request

PROMPTS = {
    "code": [
        ("code.ringbuffer",
         "Implement a Python class RingBuffer with push, pop, peek, is_full, "
         "is_empty and __len__, backed by a fixed-size list, with type hints "
         "and docstrings. Output only the code."),
        ("code.shell",
         "Write a bash pipeline that finds the 10 largest files under a directory, "
         "prints path and human-readable size sorted descending, and tolerates "
         "spaces in filenames. Output only the command with a short explanation."),
    ],
    "zh": [
        ("zh.explain",
         "请解释一下推测解码为什么能加速大模型推理，它什么时候会失效，"
         "以及接受率大概受哪些因素影响。用中文回答，三百字左右。"),
        ("zh.optimize",
         "我想把一台单显卡机器上的推理服务从每秒三十个 token 提到四十个，"
         "你会先看哪些指标，按什么顺序排查？请给出具体步骤和理由。"),
    ],
    "mixed": None,
}
PROMPTS["mixed"] = [PROMPTS["code"][0], PROMPTS["zh"][0]]


def ask(url, prompt, max_tokens, thinking, timeout=1800):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "cache_prompt": False, "stream": False,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    body = json.dumps(payload).encode("utf-8")  # str bodies get mangled by some clients
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def stats(resp):
    tm = resp.get("timings") or {}
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    dn = tm.get("draft_n") or 0
    da = tm.get("draft_n_accepted") or 0
    pn = tm.get("predicted_n") or 0
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return {
        "tg": round(tm.get("predicted_per_second") or 0, 2),
        "pp": round(tm.get("prompt_per_second") or 0, 2),
        "pred_n": pn,
        "n_reasoning": len(reasoning),
        "n_answer": len(msg.get("content") or ""),
        "accept": round(da / dn, 4) if dn else None,
        "depth": round(dn / (pn - da), 3) if (pn - da) > 0 else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--prompts", default="mixed", choices=sorted(PROMPTS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=384)
    args = ap.parse_args(argv)

    try:
        with urllib.request.urlopen(args.url + "/health", timeout=5) as r:
            if r.status != 200:
                print("server not healthy")
                return 1
    except Exception as exc:
        print("cannot reach %s: %s" % (args.url, exc))
        return 1

    print("url      : %s" % args.url)
    print("runs     : %d   max_tokens: %d" % (args.runs, args.max_tokens))
    print()
    header = "%-16s %-9s %8s %8s %8s %8s %9s %9s"
    print(header % ("prompt", "mode", "tg", "pp", "pred_n", "think_ch", "accept", "depth"))
    print("-" * 84)

    rows = []
    for label, prompt in PROMPTS[args.prompts]:
        for mode, thinking in (("off", False), ("on", True)):
            got = []
            for _ in range(args.runs):
                try:
                    got.append(stats(ask(args.url, prompt, args.max_tokens, thinking)))
                except Exception as exc:
                    print("  !! %s/%s failed: %s" % (label, mode, exc))
            if not got:
                continue
            med = lambda k: statistics.median([g[k] for g in got if g[k] is not None])
            acc = med("accept")
            dep = med("depth")
            row = {
                "prompt": label, "mode": mode, "n": len(got),
                "tg": med("tg"), "pp": med("pp"), "pred_n": med("pred_n"),
                "n_reasoning": med("n_reasoning"), "accept": acc, "depth": dep,
            }
            rows.append(row)
            print("%-16s %-9s %8.2f %8.2f %8d %8d %9s %9s" % (
                label, mode, row["tg"], row["pp"], row["pred_n"], row["n_reasoning"],
                ("%.3f" % acc) if acc is not None else "-",
                ("%.2f" % dep) if dep is not None else "-"))
        print("-" * 84)

    print()
    ok = False
    for label, _ in PROMPTS[args.prompts]:
        off = next((r for r in rows if r["prompt"] == label and r["mode"] == "off"), None)
        on = next((r for r in rows if r["prompt"] == label and r["mode"] == "on"), None)
        if not off or not on:
            continue
        ok = True
        delta = 100.0 * (on["tg"] - off["tg"]) / off["tg"] if off["tg"] else 0.0
        print("%-16s tg %.2f (off) -> %.2f (on)   %+.1f%%   | accept %.3f -> %.3f"
              % (label, off["tg"], on["tg"], delta,
                 off["accept"] or 0, on["accept"] or 0))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
