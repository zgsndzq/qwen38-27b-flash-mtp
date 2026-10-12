#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片记忆饱和测试 —— mem 里放 N 张图就全量注入 N 张，测多图检索的上限。

判据：8 张图长得一模一样，只有名字和编号不同。问"写着「X」的那张图，编号是多少？"
      ⇒ 模型必须在 N 张视觉 token 里定位到名字匹配的那张、再读编号。
      N=1 时它只要读一张图；N 越大，"在多少张图里找到"这个检索负担越重。

图片是按 600x400 截图得到的，每张约几百个视觉 token —— 所以图片轴的上限
会明显低于文本轴（ctx 相同，但单张的占用大得多）。

用法：
    python sat_img.py --levels 1,2,4,8
    python sat_img.py --probes 1,4 --levels 4
    python sat_img.py --dry
"""

import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHAT_PY = ROOT.parent / "chat.py"
LIB = ROOT / "img_lib"
RUN = ROOT / "img_run"
PARK = ROOT / "img_park"
OUT_JSON = ROOT / "_img_result.json"

UPSTREAM = os.environ.get("JEV_UPSTREAM", "http://127.0.0.1:18080")
LABELS = "ABCD"
HI = 0.55
SYS = "你在做单选题。只回答选项字母。"
LEVELS = [1, 2, 4, 8]
REP = int(os.environ.get("SAT_REP", "1"))
ORDER = os.environ.get("SAT_ORDER", "after")     # 图片轴默认只测 after（文本轴已验证位置影响小）

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 与 make_imgs.py 完全相同的种子与序列 —— 图与题必须对得上
BEASTS = ["青鸾", "白泽", "玄武", "朱雀", "麒麟", "饕餮", "鲲鹏", "九尾"]
NIMG = 8
_rnd = random.Random(31415)
_used, CODES = set(), []
while len(CODES) < NIMG:
    _c = _rnd.randint(1000, 9999)
    if _c not in _used:
        _used.add(_c)
        CODES.append(_c)


def make_probe(i):
    """第 i 张图的题：问名字对应的编号。干扰项都不在 8 个编号里。"""
    opts = [str(CODES[i - 1])]
    while len(opts) < 4:
        x = _rnd.randint(1000, 9999)
        if x not in _used and str(x) not in opts:
            opts.append(str(x))
    r = random.Random(6000 + i)
    r.shuffle(opts)
    letter = LABELS[opts.index(str(CODES[i - 1]))]
    return dict(id="I%02d" % i, img=i,
                q="图里写着「%s」的那张图，它的编号是多少？" % BEASTS[i - 1],
                opts=dict(zip(LABELS, opts)), ans=letter)


PROBES = {i: make_probe(i) for i in range(1, NIMG + 1)}


def set_n(n):
    """让 img_run/ 里恰好是前 n 张图（只 Move，不删除）。"""
    RUN.mkdir(parents=True, exist_ok=True)
    PARK.mkdir(parents=True, exist_ok=True)
    seen = set()
    for d in (LIB, RUN, PARK):
        for f in sorted(d.glob("*.png")):
            if f.name in seen:
                continue
            seen.add(f.name)
            want = RUN if int(f.stem[1:]) <= n else PARK
            if d != want:
                shutil.move(str(f), str(want))
    return sorted(p.name for p in RUN.glob("*.png"))


def _norm(tok):
    t = (tok or "").strip()
    if not t:
        return None
    c = t[0].upper()
    if c not in LABELS:
        return None
    if len(t) > 1 and any(ch.isalnum() for ch in t[1:].strip()):
        return None
    return c


def _post(payload, timeout=600):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        UPSTREAM + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def jev(parts, ans):
    payload = {
        "max_tokens": 1,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
        "repeat_penalty": 1.0,
        "logprobs": True, "top_logprobs": 40,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": parts}],
        "cache_prompt": False,
        "stream": False,
    }
    t0 = time.time()
    try:
        resp = _post(payload)
    except Exception as e:
        return dict(p=-1.0, pick="ERR", probs={}, err=str(e)[:140],
                    ms=round((time.time() - t0) * 1000.0, 1), prompt_n=None)
    ms = (time.time() - t0) * 1000.0
    ch = (resp.get("choices") or [{}])[0]
    lp = (ch.get("logprobs") or {}).get("content") or []
    top = (lp[0].get("top_logprobs") or []) if lp else []
    by = {}
    for ent in top:
        L = _norm(ent.get("token"))
        if L and L not in by:
            by[L] = ent.get("logprob")
    vals = [float(by.get(L, -30.0)) for L in LABELS]
    m = max(vals)
    ws = [math.exp(v - m) for v in vals]
    s = sum(ws) or 1.0
    probs = {L: w / s for L, w in zip(LABELS, ws)}
    return dict(p=round(probs[ans], 4),
                pick=max(LABELS, key=lambda L: probs[L]),
                probs={L: round(probs[L], 4) for L in LABELS},
                ms=round(ms, 1),
                prompt_n=(resp.get("timings") or {}).get("prompt_n"))


def ask(probe, vbank):
    """图全放题目之后、紧贴 ANSWER（与文本轴 after 同构）。"""
    head = "\n".join([probe["q"]] +
                     ["%s. %s" % (L, probe["opts"][L]) for L in LABELS])
    imgs = [{"type": "image_url", "image_url": {"url": u}}
            for _, u in sorted(vbank.items())]
    if ORDER == "before":
        parts = imgs + [{"type": "text", "text": head + "\n\nANSWER:"}]
    else:
        parts = ([{"type": "text", "text": head}] + imgs +
                 [{"type": "text", "text": "ANSWER:"}])
    rs = [jev(parts, probe["ans"]) for _ in range(REP)]
    return dict(
        p=round(statistics.median([x["p"] for x in rs]), 4),
        pick=rs[0]["pick"],
        prompt_n=rs[0]["prompt_n"],
        ms=round(statistics.median([x["ms"] for x in rs]), 1),
        err=rs[0].get("err"),
        probs=rs[0]["probs"],
    )


def pick_probes(n):
    if n <= 1:
        return [1]
    ids = [1, min(n, NIMG)]
    if n >= 4:
        ids.insert(1, max(2, n // 2))
    return sorted(set(ids))


def main():
    args = sys.argv[1:]
    if "--dry" in args:
        for i in range(1, NIMG + 1):
            p = PROBES[i]
            print("[%s] %s" % (p["id"], p["q"]))
            for L in LABELS:
                print("   %s. %s%s" % (L, p["opts"][L], "  <=" if L == p["ans"] else ""))
        print("\n图数 = %d，目录 %s" % (len(list(LIB.glob("*.png"))), LIB))
        return 0
    if "--levels" in args:
        global LEVELS
        LEVELS = [int(x) for x in args[args.index("--levels") + 1].split(",")]
    only = None
    if "--probes" in args:
        only = [int(x) for x in args[args.index("--probes") + 1].split(",")]

    try:
        with urllib.request.urlopen(UPSTREAM + "/health", timeout=5) as r:
            h = json.loads(r.read().decode("utf-8") or "{}")
        print("服务 %s  health=%s  order=%s rep=%d" % (UPSTREAM, h.get("status", "?"), ORDER, REP))
    except Exception as e:
        print("服务不可用 %s : %s" % (UPSTREAM, e))
        return 2

    spec = importlib.util.spec_from_file_location("chat", str(CHAT_PY))
    chat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chat)

    rows, t_start = [], time.time()
    line = "%-4s %-5s %-5s %8s %8s %8s  %s" % ("N", "探针", "档位", "p", "prompt_n", "ms", "pick")
    print(line)
    print("-" * len(line))

    for n in [0] + LEVELS:
        files = set_n(n)
        chat.MEM_DIR = RUN
        vbank = chat.load_vision_bank() if n else {}
        if n and not vbank:
            print("!! N=%d 但图库为空 —— set_n 失效，本轮无效" % n)
            return 3
        for i in (only if only else (pick_probes(n) if n else [1])):
            probe = PROBES[i]
            r = ask(probe, vbank)
            tag = ("最老" if i == 1 else ("最新" if i == n else "中间"))
            rows.append(dict(n=n, probe=probe["id"], tag=tag, order=ORDER,
                             files=len(files), **r))
            print("%-4d %-5s %-5s %8.3f %8s %8s  %s%s" % (
                n, probe["id"], tag, r["p"], r["prompt_n"], r["ms"], r["pick"],
                ("  " + r["err"]) if r.get("err") else ""))
        sys.stdout.flush()

    by = {}
    for r in rows:
        by.setdefault(r["n"], []).append(r["p"])
    print("-" * len(line))
    print("档位汇总:")
    for n in sorted(by):
        ps = by[n]
        hit = sum(1 for p in ps if p >= HI)
        print("  N=%-2d 图  平均 p=%.3f  命中 %d/%d" % (n, sum(ps) / len(ps), hit, len(ps)))

    OUT_JSON.write_text(json.dumps(
        dict(levels=LEVELS, rep=REP, order=ORDER, hi=HI,
             elapsed_s=round(time.time() - t_start, 1), rows=rows),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n用时 %.0f s   结果 -> %s" % (time.time() - t_start, OUT_JSON))
    return 0


if __name__ == "__main__":
    sys.exit(main())
