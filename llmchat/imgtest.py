#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imgtest.py — 极简「首轮塞一张图」多轮测试器（零第三方依赖，不依赖 chat.py）

唯一目的：看清楚多轮对话时，服务端第 2 轮起到底是「增量复用」还是「全量重算」。

每轮结束打一行判据：
    [本轮] prompt_n=<这一轮真正送进模型算的 token>  cache_n=<从缓存复用的 token>

  * prompt_n ≈ 整段上下文(上万)、cache_n ≈ 0   → 全量重算（缓存没接上）⚠️
  * prompt_n 只有几十~几百、cache_n 很大        → 增量（正常）✅

第一轮把**一张**图塞进 user content；之后每轮只发原话，历史原样追加、永不改写
（改写已发出的消息会让 prompt cache 当轮全废）。

用法（双击 imgtest.bat 即可，也可加参数）：
    python imgtest.py
    python imgtest.py --img mem\\chess1.png --think on
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_URL = os.environ.get("LLAMA_CHAT_URL", "http://127.0.0.1:18080")
IMG_SUFFIX = (".png", ".jpg", ".jpeg", ".webp", ".gif")
DEFAULT_SYS = "你是一个有用的中文 AI 助手，请准确、完整地回答用户的问题。"


def pick_image(arg=None):
    """默认取 mem\\ 下按文件名排序的第一张图；--img 可指定任意一张。"""
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = (HERE / arg).resolve()
        return p if p.is_file() else None
    mem = HERE / "mem"
    if mem.is_dir():
        for p in sorted(mem.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_SUFFIX:
                return p
    return None


def data_url(path):
    raw = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii"))


def probe(base_url, timeout=5.0):
    """/v1/models 探活。返回 (ok, 模型id 或 错误信息)。"""
    try:
        with urllib.request.urlopen(base_url + "/v1/models", timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        rows = d.get("data") or []
        return True, (rows[0].get("id") if rows else None)
    except Exception as e:                                    # noqa: BLE001
        return False, str(e)


def send(messages, base_url, temperature, max_tokens, think, timeout):
    """非流式一发。返回 (content, reasoning_content, timings)。"""
    body = {"model": "gguf", "messages": messages, "stream": False,
            "cache_prompt": True}                             # 显式声明，服务端默认也是 true
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens:
        body["max_tokens"] = max_tokens
    if think is not None:
        body["chat_template_kwargs"] = {"enable_thinking": bool(think)}

    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read().decode("utf-8"))

    ch = (obj.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return (msg.get("content") or "", msg.get("reasoning_content") or "",
            obj.get("timings") or {}, obj.get("usage") or {})


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")               # Windows 控制台中文
    except Exception:                                          # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(
        prog="imgtest.py",
        description="首轮塞一张图的多轮 prefill 复用测试器（零依赖）")
    ap.add_argument("--url", default=DEFAULT_URL, help="llama-server 地址")
    ap.add_argument("--img", default=None, help="首轮要塞的图（默认 mem\\ 下第一张）")
    ap.add_argument("--sys-file", default=None, help="system prompt 文件（默认 sys.txt，若有）")
    ap.add_argument("--temp", type=float, default=0.6, help="采样温度")
    ap.add_argument("--max-tokens", type=int, default=512, help="单轮回答上限（0=不限）")
    ap.add_argument("--think", choices=["on", "off"], default="off",
                    help="思考开关（默认 off：测 prefill 复用不需要，更快）")
    ap.add_argument("--timeout", type=float, default=600.0, help="请求超时秒数")
    a = ap.parse_args(argv)

    url = a.url.rstrip("/")
    img = pick_image(a.img)
    if img is None:
        print("[错误] 找不到图片：请用 --img 指定一张，或把图放进 %s" % (HERE / "mem"))
        return 2
    img_url = data_url(img)                                    # 只准备这一次，之后复用

    sys_file = Path(a.sys_file) if a.sys_file else (HERE / "sys.txt")
    if sys_file.is_file():
        sys_prompt = sys_file.read_text(encoding="utf-8-sig").strip() or DEFAULT_SYS
    else:
        sys_prompt = DEFAULT_SYS

    think = (a.think == "on")

    ok, info = probe(url)
    print("=" * 70)
    print(" imgtest — 首轮带【一张】图的多轮 prefill 复用测试器")
    print(" 服务端 : %s" % url)
    if ok:
        print(" 模型   : %s" % (info or "(未报告)"))
    else:
        print(" [警告] 探活失败：%s" % info)
        print("        请先启动同包的 llmserver.bat（且 VISION=1，已挂 --mmproj）")
    print(" 首轮图 : %s  (%.0f KB)" % (img.name, img.stat().st_size / 1024.0))
    print(" 参数   : think=%s  temp=%s  max_tokens=%s" % (a.think, a.temp, a.max_tokens or "不限"))
    print(" 提示词 : %s" % (sys_file.name if sys_file.is_file() else "(内置)"))
    print("=" * 70)
    print(" 输入 'quit' 退出 · 'clear' 清空上下文重来")
    print(" 判据：看第 2 轮起的 [本轮] 行 —— prompt_n 很小 = 增量 OK / ≈上万 = 全量 ⚠️")
    print()

    messages = [{"role": "system", "content": sys_prompt}]
    turn = 0
    while True:
        try:
            q = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        if q.lower() == "clear":
            messages = [{"role": "system", "content": sys_prompt}]
            turn = 0
            print("（上下文已清空，下一轮重新算首轮——会重新带图）\n")
            continue

        turn += 1
        if turn == 1:
            content = [{"type": "text", "text": q},
                       {"type": "image_url", "image_url": {"url": img_url}}]
            shown = "原话 + 1 张图"
        else:
            content = q
            shown = "纯文本"

        messages.append({"role": "user", "content": content})
        try:
            text, _reasoning, t, usage = send(
                messages, url, a.temp, a.max_tokens, think, a.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:400]
            print("[HTTP %s] %s\n（若是图片相关报错，检查服务端是否用 VISION=1 启动）\n" % (e.code, body))
            messages.pop()
            turn -= 1
            continue
        except Exception as e:                                 # noqa: BLE001
            print("[请求失败] %s\n" % e)
            messages.pop()
            turn -= 1
            continue

        print("\nAI> " + (text if text else "(空回答)"))
        messages.append({"role": "assistant", "content": text})

        pn = t.get("prompt_n")
        cn = t.get("cache_n")
        pp = t.get("prompt_per_second")
        tg = t.get("predicted_per_second")
        print("-" * 70)
        print("[本轮] 第 %d 轮（%s）" % (turn, shown))
        if pn is not None:
            n_cache = cn if cn is not None else 0
            if n_cache > 0:
                tag = "增量 ✅ 复用 %s tok" % n_cache
            else:
                tag = "全量重算 ⚠️ 缓存未命中"
            print("       prompt_n=%-7s cache_n=%-7s → %s" % (pn, n_cache, tag))
        elif usage:
            print("       usage.prompt_tokens=%s （服务端未回 timings）" % usage.get("prompt_tokens"))
        if pp or tg:
            seg = []
            if pp:
                seg.append("预填充 %.1f t/s" % pp)
            if tg:
                seg.append("生成 %.1f t/s" % tg)
            print("       " + " · ".join(seg))
        print()

    print("再见！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
