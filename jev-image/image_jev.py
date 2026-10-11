#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image JEV 接口 —— 给「一张图 + 问题 + 选项」，回「字母 + 概率」。

对标 Image JevBench（benchmarkheaven.com/image-jev-bench）的 System One 决策形态：
一道题 = (图, 问题, 选项) -> (答案字母, 该字母在选项上的归一化概率)。
概率是它校准轴的唯一输入 —— 只给字母不给概率的系统，校准轴恒为 0。

判据链路与 sat/sat_test.py 完全一致（两处数字可以直接比）：
    max_tokens=1 + logprobs + top_logprobs=40 + enable_thinking=False + 采样全关
    => 一次前向只吐 1 个 token，取它的概率分布，在选项字母集合上重归一化。

成本口径（榜上成本轴 = 实测 GPU 秒 x GPU 小时单价）：
    回传 timings.prompt_ms / predicted_ms => gpu_s，再折算 USD/1000 决策。
    cache_prompt 默认 False（每次都全量 prefill），测的是最坏情况。

用法：
    # 单题
    python image_jev.py --image mem/chess.png --question "..." --options "甲,乙,丙"
    # 批量（每行 {"id","image","question","options","gold"}）
    python image_jev.py --batch items.jsonl --out preds.jsonl
    # 打分（准确率 + 10-bin ECE + 延迟/成本）
    python image_jev.py --score preds.jsonl --usd-per-gpu-hour 1.0
    # 只看拼装不发请求
    python image_jev.py --batch items.jsonl --dry
"""

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = (os.environ.get("LLAMA_CHAT_URL") or os.environ.get("JEV_UPSTREAM")
       or "http://127.0.0.1:18080")
SYS = "你在做单选题。只回答选项字母。"
TOPK = 40
LABELS = "ABCDEFGH"
MISSING = -30.0            # 没进 top-40 的选项按此 logprob 计（exp(-30) ~ 1e-13）

_URL_CACHE = {}


def data_url(path):
    """图片 -> data URI。参数可以是**本地路径**（CLI 走这条），也可以是浏览器
    直接给的 **data: URI**（bridge/网页走这条，图不落盘）。按内容 sha1 缓存，
    同一批里重复的图只编码一次。"""
    src = str(path)
    if src.startswith("data:"):
        head, _, b64 = src.partition(",")
        if not head.endswith(";base64"):
            return src, hashlib.sha1(src.encode("utf-8")).hexdigest(), 0
        try:
            raw = base64.b64decode(b64 + "=" * (-len(b64) % 4))
        except Exception:
            raw = b""
        return src, hashlib.sha1(raw).hexdigest(), len(raw)
    p = Path(path)
    raw = p.read_bytes()
    sha = hashlib.sha1(raw).hexdigest()
    if sha not in _URL_CACHE:
        mime = mimetypes.guess_type(p.name)[0] or "image/png"
        _URL_CACHE[sha] = "data:%s;base64,%s" % (
            mime, base64.b64encode(raw).decode("ascii"))
    return _URL_CACHE[sha], sha, len(raw)


def strip_prefix(opt):
    """选项文本自带 "A. xxx" 前缀时剥掉，免得和我们加的 A. 打架。"""
    t = str(opt).strip()
    if len(t) > 1 and t[0].upper() in LABELS and t[1] in ".、:：)） ":
        return t[2:].strip()
    return t


def build_prompt(question, options, labels):
    """与 sat_test.ask() 同形：问题 + 各选项 + 空行 + ANSWER:（选项在最后锚住答案位）。"""
    lines = [str(question).strip()]
    lines += ["%s. %s" % (L, strip_prefix(o)) for L, o in zip(labels, options)]
    return "\n".join(lines) + "\n\nANSWER:"


def norm_label(tok, labels):
    """token -> 选项字母。只认「单个字母」或「字母+标点」；"AB"/"A1" 这类多字符 token 不算。"""
    t = (tok or "").strip()
    if not t:
        return None
    c = t[0].upper()
    if c not in labels:
        return None
    if len(t) > 1 and any(ch.isalnum() for ch in t[1:].strip()):
        return None
    return c


def _post(payload, timeout):
    req = urllib.request.Request(
        URL.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def decide(question, options, image=None, cache=False, timeout=300,
           usd_per_gpu_hour=1.0):
    """一道题 -> 一个决策。返回 dict（含 probs / mass / gpu_s / usd_per_1000）。"""
    options = list(options)
    if not options:
        raise ValueError("options 不能为空")
    if len(options) > len(LABELS):
        raise ValueError("最多支持 %d 个选项" % len(LABELS))
    labels = LABELS[:len(options)]
    text = build_prompt(question, options, labels)

    sha, nbytes = None, 0
    if image:
        url, sha, nbytes = data_url(image)
        # 顺序：文本（任务）在前，图在后 —— 与 llmchat 的注入顺序一致。
        content = [{"type": "text", "text": text},
                   {"type": "image_url", "image_url": {"url": url}}]
    else:
        content = text

    payload = {
        "max_tokens": 1,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
        "repeat_penalty": 1.0,
        "logprobs": True, "top_logprobs": TOPK,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": content}],
        "cache_prompt": bool(cache),
        "stream": False,
    }

    base = dict(question=question, options=options, image=image,
                image_sha1=sha, image_bytes=nbytes, labels=labels)
    t0 = time.time()
    try:
        resp = _post(payload, timeout)
    except Exception as e:
        base.update(valid=False, answer=None, prob=0.0, probs={}, mass=0.0,
                    raw_token=None, error=str(e)[:200],
                    latency_ms=round((time.time() - t0) * 1000.0, 1),
                    prompt_n=None, gpu_s=None, usd_per_1000=None)
        return base

    latency_ms = (time.time() - t0) * 1000.0
    ch = (resp.get("choices") or [{}])[0]
    lp = (ch.get("logprobs") or {}).get("content") or []
    top = (lp[0].get("top_logprobs") or []) if lp else []

    by = {}                                   # 同一字母的多个 token 形态取概率最高的那个
    for ent in top:
        L = norm_label(ent.get("token"), labels)
        if L and L not in by:
            by[L] = float(ent.get("logprob", MISSING))

    lps = [by.get(L, MISSING) for L in labels]
    m = max(lps)
    ws = [math.exp(v - m) for v in lps]
    s = sum(ws) or 1.0
    probs = {L: w / s for L, w in zip(labels, ws)}
    # 有效概率覆盖：选项字母在完整词表分布里的总概率。低 = 模型本来想说别的（答案不可信）。
    mass = min(sum(math.exp(v) for v in lps), 1.0)
    answer = max(labels, key=lambda L: probs[L]) if by else None

    tm = resp.get("timings") or {}
    gpu_s = ((tm.get("prompt_ms") or 0.0) + (tm.get("predicted_ms") or 0.0)) / 1000.0
    base.update(
        valid=bool(by),
        answer=answer,
        prob=round(probs[answer], 4) if answer else 0.0,
        probs={L: round(probs[L], 4) for L in labels},
        mass=round(mass, 6),
        raw_token=(lp[0].get("token") if lp else None),
        latency_ms=round(latency_ms, 1),
        prompt_n=tm.get("prompt_n"),
        gpu_s=round(gpu_s, 4),
        usd_per_1000=round(gpu_s * usd_per_gpu_hour / 3.6, 5),
    )
    return base


def score(rows, rate=1.0):
    """对着 gold 打分：准确率（无效/不可解析算错）+ 10-bin top-label ECE + 延迟/成本。"""
    done = [r for r in rows if r.get("gold") not in (None, "")]
    n = len(done)
    if not n:
        return dict(items=0, note="没有带 gold 的条目")
    hit = sum(1 for r in done if r.get("valid") and r.get("answer") == r["gold"])
    cnt = [0] * 10
    conf = [0.0] * 10
    cor = [0] * 10
    for r in done:
        c = float(r.get("prob") or 0.0)
        b = min(int(c * 10), 9)
        cnt[b] += 1
        conf[b] += c
        cor[b] += 1 if (r.get("valid") and r.get("answer") == r["gold"]) else 0
    ece = sum(cnt[b] / n * abs(conf[b] / cnt[b] - cor[b] / cnt[b])
              for b in range(10) if cnt[b])
    lat = sorted(r["latency_ms"] for r in done if r.get("latency_ms"))
    gpu = [r["gpu_s"] for r in done if r.get("gpu_s")]
    return dict(
        items=n,
        accuracy=round(hit / n, 4),
        hits="%d/%d" % (hit, n),
        valid_rate=round(sum(1 for r in done if r.get("valid")) / n, 4),
        ece10=round(ece, 4),
        mass_mean=round(statistics.mean([r.get("mass") or 0.0 for r in done]), 4),
        latency_p50=round(statistics.median(lat), 1) if lat else None,
        latency_p95=round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 1) if lat else None,
        gpu_s_mean=round(statistics.mean(gpu), 4) if gpu else None,
        usd_per_1000=round(statistics.mean(gpu) * rate / 3.6, 5) if gpu else None,
    )


def load_items(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            it = json.loads(line)
            it.setdefault("id", "item-%d" % i)
            opts = it["options"]
            if isinstance(opts, str):
                it["options"] = [o for o in opts.replace("|", ",").split(",") if o.strip()]
            out.append(it)
    return out


def run_batch(items, out_path=None, cache=False, usd_per_gpu_hour=1.0, quiet=False):
    rows = []
    fh = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for k, it in enumerate(items, 1):
            r = decide(it["question"], it["options"], image=it.get("image"),
                       cache=cache, usd_per_gpu_hour=usd_per_gpu_hour)
            r["id"] = it.get("id")
            if it.get("gold") is not None:
                r["gold"] = it["gold"]
            rows.append(r)
            if fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            if not quiet:
                print("[%d/%d] %s -> %s p=%.3f %s ms" % (
                    k, len(items), r["id"], r["answer"], r["prob"],
                    r.get("error") or ("%.0f" % r["latency_ms"])), file=sys.stderr)
    finally:
        if fh:
            fh.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description="Image JEV 接口（图 + 问题 + 选项 -> 字母 + 概率）")
    ap.add_argument("--image")
    ap.add_argument("--question")
    ap.add_argument("--options", help="逗号或 | 分隔；也可写成 JSON 数组")
    ap.add_argument("--batch", help="items.jsonl，每行 {id,image,question,options,gold}")
    ap.add_argument("--out", help="把逐题结果写成 jsonl（榜上要的 item-level outputs）")
    ap.add_argument("--score", help="对 --out 产出的 jsonl 打分")
    ap.add_argument("--usd-per-gpu-hour", type=float, default=1.0,
                    help="成本轴单价；榜上按实测 GPU 秒 x 该价折算（默认 1.0）")
    ap.add_argument("--cache", action="store_true",
                    help="打开 prompt cache（默认关，测最坏情况）")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--dry", action="store_true", help="只拼装打印，不发请求")
    ap.add_argument("--url", help="覆盖服务地址")
    a = ap.parse_args()

    global URL, SYS
    if a.url:
        URL = a.url

    if a.score:
        rows = [json.loads(l) for l in open(a.score, "r", encoding="utf-8")
                if l.strip()]
        print(json.dumps(score(rows, a.usd_per_gpu_hour), ensure_ascii=False, indent=2))
        return 0

    if a.batch:
        items = load_items(a.batch)
        if a.dry:
            for it in items:
                opts = it["options"]
                print("--- %s ---" % it.get("id"))
                print(build_prompt(it["question"], opts, LABELS[:len(opts)]))
                print("[图] %s" % (it.get("image") or "(无)"))
            return 0
        rows = run_batch(items, a.out, a.cache, a.usd_per_gpu_hour)
        s = score(rows) if any("gold" in r for r in rows) else None
        print(json.dumps({"n": len(rows), "score": s}, ensure_ascii=False, indent=2))
        return 0

    if not a.question or not a.options:
        ap.error("要么给 --question/--options，要么给 --batch")
    opts = json.loads(a.options) if a.options.strip().startswith("[") else \
        [o for o in a.options.replace("|", ",").split(",") if o.strip()]
    if a.dry:
        print(build_prompt(a.question, opts, LABELS[:len(opts)]))
        return 0
    r = decide(a.question, opts, image=a.image, cache=a.cache,
               timeout=a.timeout, usd_per_gpu_hour=a.usd_per_gpu_hour)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
