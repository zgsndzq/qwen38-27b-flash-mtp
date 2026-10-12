#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆饱和测试 —— 「有几个文件就注入几个」的上限在哪。

做法：
  mem 目录里放 N 个文件 → 全部注入 → 用 JEV 判据题问其中某一条的事实，
  看正确项的归一化概率随 N 怎么变。

为什么用虚构事实：
  32 个虚构星港的编号，模型先验里根本不存在 ⇒ 答对只可能是"读到了记忆"，
  不存在"其实它本来就知道"的污染（之前 RGB/原子序数那类就有这个毛病）。

为什么文件严格嵌套：
  N=4 的那 4 个文件就是 N=8 里的前 4 个 ⇒ 同一条记忆（D1）在各档之间
  字节完全一致，唯一的变量就是"库里还塞了多少别的"，这才是纯净的稀释曲线。

不碰 F:/super/llmchat/mem（用户的记忆目录），测试库建在 sat/ 下。
用 run/ + park/ 两个目录靠 Move 切换启用数量 —— 只改名不删除，避开删除配额。

用法：
    python sat_test.py --gen          只生成测试库，不跑模型
    python sat_test.py                跑全量（0 基线 + 6 档）
    python sat_test.py --levels 1,4,16
    python sat_test.py --dry          打印题目与文件，不调模型
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

ROOT = Path(__file__).resolve().parent          # F:\super\llmchat\sat
CHAT_PY = ROOT.parent / "chat.py"               # F:\super\llmchat\chat.py
LIB = ROOT / "lib"                              # 全部 32 个记忆文件（只写一次）
RUN = ROOT / "run"                              # 当前启用的（load 这里）
PARK = ROOT / "park"                            # 暂时关掉的
OUT_JSON = ROOT / "_result.json"
OUT_TXT = ROOT / "_report.txt"

UPSTREAM = os.environ.get("JEV_UPSTREAM", "http://127.0.0.1:18080")
LABELS = "ABCD"
HI = 0.55                                        # 「命中」阈值（与 probe.py 一致）
SYS = "你在做单选题。只回答选项字母。"
LEVELS = [1, 2, 4, 8, 16, 32]
REP = int(os.environ.get("SAT_REP", "2"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------- 测试库生成
# 名字用 15×15 组合，保证唯一 —— 同名会让"问哪个港"本身变歧义
_A = list("赤青玄白紫墨金碧苍丹蔚绛黛皓朱")
_B = list("霄霜澜桦电玉乌落霞微风云雪月岚")
NAMES = [a + b for a in _A for b in _B]            # 225 个
GALAXY = ["天璇", "天枢", "摇光", "开阳", "玉衡", "天玑", "天权", "紫微"]
NDOM = 160                                          # 库容量（要探 ctx 上限，得比 32 大得多）

TPL = """# 星港档案 · {nm}港

{nm}港是一座货运星港，隶属{gal}星系，主要承接跨星系的散货与冷链运输。

- 港口编号：{code}
- 常驻舰船：{ships} 艘
- 领航员代号：{pilot}

港区共设三个泊位区，常年由领航员轮班值守。所有进出港的船只都必须先
向调度台报备自己的港口编号，调度台凭编号核对船籍。{nm}港每年吞吐量在
{gal}星系内排名靠前。
"""


def build_spec():
    """确定性生成 32 个域的事实（种子固定，每次跑都一样）。"""
    rnd = random.Random(20260926)
    specs, used = {}, set()
    for i in range(1, NDOM + 1):
        while True:
            code = rnd.randint(1000, 9999)
            if code not in used:
                used.add(code)
                break
        specs[i] = dict(
            nm=NAMES[i - 1],
            gal=GALAXY[(i - 1) % len(GALAXY)],
            code=code,
            ships=rnd.randint(11, 99),
            pilot="%s-%03d" % (rnd.choice("KLMNPQRSTVWXYZ"), rnd.randint(100, 999)),
        )
    # 干扰项：3 个既不在库里、也不重复的 4 位数
    for i in range(1, NDOM + 1):
        dis = []
        while len(dis) < 3:
            x = rnd.randint(1000, 9999)
            if x not in used and x not in dis:
                dis.append(x)
        specs[i]["dis"] = dis
    return specs


SPECS = build_spec()


def gen_lib():
    """把 32 个记忆文件写进 lib/（已存在则跳过，不覆盖）。"""
    LIB.mkdir(parents=True, exist_ok=True)
    written = 0
    for i in range(1, NDOM + 1):
        name = "d%02d.md" % i
        # 文件可能正躺在 run/ 或 park/（上一轮搬走的），别重复生成一套
        if any((d / name).exists() for d in (LIB, RUN, PARK)):
            continue
        (LIB / name).write_text(TPL.format(**SPECS[i]), encoding="utf-8")
        written += 1
    return written


def make_probe(i):
    """第 i 个域的判据题：问它的港口编号，选项位置按域号固定打乱。"""
    s = SPECS[i]
    opts = [str(s["code"])] + [str(x) for x in s["dis"]]
    r = random.Random(7000 + i)
    r.shuffle(opts)
    letter = LABELS[opts.index(str(s["code"]))]
    return dict(id="D%02d" % i, dom=i,
                q="星港「%s港」的港口编号是多少？" % s["nm"],
                opts=dict(zip(LABELS, opts)), ans=letter)


PROBES = {i: make_probe(i) for i in range(1, NDOM + 1)}


def set_n(n):
    """让 run/ 里恰好是前 n 个文件（只 Move，不删除）。

    文件可能在三个地方（lib / run / park），所以先把当前位置找出来再搬到目标目录 ——
    v1 只在 run/park 之间搬，首次调用时文件还在 lib，结果一个都没搬过去，
    整个实验跑成了"空注入"（prompt_n 恒定 38），白跑一轮。
    """
    RUN.mkdir(parents=True, exist_ok=True)
    PARK.mkdir(parents=True, exist_ok=True)
    seen = set()                    # 文件可能已不在 lib（上一轮搬走了），三个目录都要扫
    for d in (LIB, RUN, PARK):
        for f in sorted(d.glob("*.md")):
            if f.name in seen:
                continue
            seen.add(f.name)
            want = RUN if int(f.stem[1:]) <= n else PARK
            if d != want:
                shutil.move(str(f), str(want))
    return sorted(p.name for p in RUN.glob("*.md"))


# ---------------------------------------------------------------- JEV 判据
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


def _post(payload, timeout=300):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        UPSTREAM + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def jev(user_content, ans):
    payload = {
        "max_tokens": 1,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
        "repeat_penalty": 1.0,
        "logprobs": True, "top_logprobs": 40,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": user_content}],
        "cache_prompt": False,          # 每次全量 prefill，杜绝缓存假加速
        "stream": False,
    }
    t0 = time.time()
    try:
        resp = _post(payload)
    except Exception as e:              # 超 ctx / 服务拒绝 —— 记下来，别让整轮崩掉
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


# ---------------------------------------------------------------- 主流程
def ask(probe, bank, order="after"):
    """判据题拼装：问题 + 选项 + 记忆 + ANSWER:（llmchat 的真实形态是"问题在前记忆在后"）。

    order="before" 把记忆挪到问题之前 —— 作 recency 对照：如果一个在后全灭、在前能中，
                 说明卡的是注意力位置，不是"记不住"。
    """
    head = "\n".join([probe["q"]] +
                     ["%s. %s" % (L, probe["opts"][L]) for L in LABELS])
    hits = list(bank.items())
    if hits:
        kb = chat.know_block(hits)
        content = (head + "\n\n" + kb if order == "after" else kb + "\n\n" + head)
        content += "\n\nANSWER:"
    else:
        content = head + "\n\nANSWER:"
    rs = [jev(content, probe["ans"]) for _ in range(REP)]
    return dict(
        p=round(statistics.median([x["p"] for x in rs]), 4),
        pick=rs[0]["pick"],
        prompt_n=rs[0]["prompt_n"],
        ms=round(statistics.median([x["ms"] for x in rs]), 1),
        probs=rs[0]["probs"],
    )


def pick_probes(n):
    """每档问哪几条：第 1 条（最老）、中间那条、最新那条。"""
    if n <= 1:
        return [1]
    ids = [1, n]
    if n >= 4:
        ids.insert(1, max(2, n // 2))
    return sorted(set(ids))


def main():
    args = sys.argv[1:]
    dry = "--dry" in args
    if "--gen" in args:
        k = gen_lib()
        print("已生成 %d 个新文件到 %s（共 %d 个域）" % (k, LIB, NDOM))
        return 0
    if "--levels" in args:
        global LEVELS
        LEVELS = [int(x) for x in args[args.index("--levels") + 1].split(",")]
    only = None
    if "--probes" in args:              # 指定问哪几个域，用来单独验某道题
        only = [int(x) for x in args[args.index("--probes") + 1].split(",")]

    if "--setn" in args:                       # 自检：只搬家，不调模型
        n = int(args[args.index("--setn") + 1])
        fs = set_n(n)
        print("run/  %2d 个: %s" % (len(fs), ", ".join(fs[:8])))
        print("park/ %2d 个" % len(list(PARK.glob("*.md"))))
        print("lib/  %2d 个" % len(list(LIB.glob("*.md"))))
        return 0

    gen_lib()

    if dry:
        print("LEVELS = %s   REP = %d" % (LEVELS, REP))
        for i in [1, 2, 4, 8, 16, 32]:
            if i > NDOM:
                continue
            p = PROBES[i]
            print("\n[D%02d] %s" % (i, p["q"]))
            for L in LABELS:
                print("   %s. %s%s" % (L, p["opts"][L], "  <=" if L == p["ans"] else ""))
        print("\n文件数 = %d，每文件约 %d 字符" % (
            len(list(LIB.glob("*.md"))), len(TPL.format(**SPECS[1]))))
        return 0

    try:
        with urllib.request.urlopen(UPSTREAM + "/health", timeout=5) as r:
            h = json.loads(r.read().decode("utf-8") or "{}")
        print("服务 %s  health=%s" % (UPSTREAM, h.get("status", "?")))
    except Exception as e:
        print("服务不可用 %s : %s" % (UPSTREAM, e))
        return 2

    rows, t_start = [], time.time()
    line = "%-4s %-5s %-5s %-7s %8s %8s %8s  %s" % (
        "N", "探针", "档位", "记忆位", "p", "prompt_n", "ms", "pick")
    print(line)
    print("-" * len(line))

    for n in [0] + LEVELS:
        files = set_n(n)
        bank = chat.load_bank(RUN) if n else {}
        kchars = sum(len(v) for v in bank.values())
        if n and not bank:              # 空注入会让整轮变成噪声，直接停，别浪费
            print("!! N=%d 但注入为空 —— set_n 失效，本轮无效" % n)
            return 3
        for i in (only if only else (pick_probes(n) if n else [1])):
            probe = PROBES[i]
            tag = ("最老" if i == 1 else ("最新" if i == n else "中间"))
            for order in ("after", "before"):
                r = ask(probe, bank, order)
                rows.append(dict(n=n, probe=probe["id"], tag=tag, order=order,
                                 ans=probe["ans"], files=len(files),
                                 kchars=kchars, **r))
                print("%-4d %-5s %-5s %-7s %8.3f %8s %8s  %s" % (
                    n, probe["id"], tag, order, r["p"], r["prompt_n"],
                    r["ms"], r["pick"]))
        sys.stdout.flush()

    # ---- 汇总 ----
    print("-" * len(line))
    by = {}
    for r in rows:
        by.setdefault(r["n"], {}).setdefault(r["order"], []).append(r)
    print("档位汇总（同档各探针的平均 p）:")
    for n in sorted(by):
        first = next(iter(by[n].values()))[0]
        for order in ("after", "before"):
            rs = by[n].get(order, [])
            if not rs:
                continue
            ps = [x["p"] for x in rs]
            hit = sum(1 for p in ps if p >= HI)
            print("  N=%-3d 记忆在%-2s 平均 p=%.3f  命中 %d/%d  注入 %d 块 / %d 字符" % (
                n, ("后" if order == "after" else "前"), sum(ps) / len(ps),
                hit, len(ps), first["files"], first["kchars"]))

    OUT_JSON.write_text(json.dumps(
        dict(levels=LEVELS, rep=REP, hi=HI, elapsed_s=round(time.time() - t_start, 1),
             rows=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n用时 %.0f s   结果 -> %s" % (time.time() - t_start, OUT_JSON))
    return 0


# 只在需要时 import（dry/gen 不依赖 chat.py）
chat = None
if not ({"--gen", "--dry"} & set(sys.argv[1:])):
    spec = importlib.util.spec_from_file_location("chat", str(CHAT_PY))
    chat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chat)
    chat.MEM_DIR = RUN              # load_vision_bank 用全局；load_bank 显式传


if __name__ == "__main__":
    sys.exit(main())
