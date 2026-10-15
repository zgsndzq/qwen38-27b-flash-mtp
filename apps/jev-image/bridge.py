#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image JEV -- bridge + UI server  (upload an image, ask about it, get the answer)

The page is same-origin; every "look at the image" is one request to the local
llama-server on 127.0.0.1:18080:

  judge  (= JEV)   max_tokens=1 + logprobs + top_logprobs=40 + no thinking
                   -> exactly one forward pass: the option letters' probabilities
                      are read straight off the logits, zero tokens decoded.
  describe         max_tokens=N, temperature 0
                   -> the everyday image-QA shape: just say what it is.

The image arrives from the browser as a data URI and is forwarded as-is (never
touched the disk).  The bridge exists so the prompt lives in one place and the
browser does not have to deal with CORS.

Stdlib only.  Python 3.8+.
"""

import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import image_jev as IJ                                            # noqa: E402

UPSTREAM = (os.environ.get("JEV_UPSTREAM") or IJ.URL).rstrip("/")
LISTEN = ("127.0.0.1", 18091)
LABELS = IJ.LABELS
CACHE = True                     # UI 场景复用前缀；CLI 仍以 False 测最坏情况

SYS_DESC = (
    "You are looking at an image and answering a question about it. "
    "Answer directly: short, concrete, no greetings, do not restate the question. "
    "If something is not visible, say so instead of guessing."
)

# 无图对照那一臂专用的 system：不能再说"你在看图"，否则模型只会回一句
# "我无法查看图片"，对照就变成废话。没有图时让它照自己的先验直接答。
SYS_DESC_BLIND = (
    "Answer the question directly, from what you already know: "
    "short, concrete, no greetings, do not restate the question."
)


# ----------------------------------------------------------------- 中止通道
# 点停止必须落成"把上游连接掐掉"，发取消请求没有用：
#   非流式请求在服务端等的是 req.should_stop（tools/server/server-context.cpp:4558），
#   它只认 TCP 连接断开；DELETE /v1/stream 那套只对带 X-Conversation-Id 的流式请求
#   生效（server-stream.cpp:622-628）。所以桥上唯一能真停住 GPU 的动作，就是从
#   另一个线程把这条上游连接关掉 —— 服务端立刻看到断开并取消这个 slot。
# 前端给每次请求带一个 stop_key，连接就登记在这个 key 下。
_LIVE = {}                       # stop_key -> _TrackedConn
_LIVE_LOCK = threading.Lock()
_PLAIN_OPENER = urllib.request.build_opener()


class _TrackedConn(http.client.HTTPConnection):
    """连上时把自己登记进 _LIVE，好让 /api/stop 从另一个线程关掉它。"""

    def __init__(self, *a, stop_key=None, **kw):
        super().__init__(*a, **kw)
        self.stop_key = stop_key

    def connect(self):
        super().connect()
        if self.stop_key:
            with _LIVE_LOCK:
                _LIVE[self.stop_key] = self


def _tracking_opener(stop_key):
    class _H(urllib.request.HTTPHandler):
        def __init__(self):
            super().__init__()

        def http_open(self, req):
            return self.do_open(
                lambda *a, **kw: _TrackedConn(*a, stop_key=stop_key, **kw), req)
    return urllib.request.build_opener(_H)


def _abort(stop_key):
    """掐掉该 key 下在飞的上游连接；返回是否真掐到了。"""
    with _LIVE_LOCK:
        conn = _LIVE.pop(stop_key, None)
    if conn is None:
        return False
    try:
        if conn.sock is not None:
            conn.sock.shutdown(socket.SHUT_RDWR)   # 这一下把 FIN 发出去，服务端立刻看到
    except OSError:
        pass
    try:
        conn.close()
    except OSError:
        pass
    return True


# ----------------------------------------------------------------- upstream
def _post(path, payload, timeout=600, stop_key=None):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        UPSTREAM + path, data=body,
        headers={"Content-Type": "application/json; charset=utf-8"})
    opener = _tracking_opener(stop_key) if stop_key else _PLAIN_OPENER
    try:
        with opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    finally:
        if stop_key:
            with _LIVE_LOCK:
                _LIVE.pop(stop_key, None)


def _health(timeout=3):
    with urllib.request.urlopen(UPSTREAM + "/health", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def _content(text, image):
    """图放在文本后面（与 llmchat 的注入顺序一致）。"""
    if image:
        return [{"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image}}]
    return text


def _timings(resp, t0, usd_per_gpu_hour=1.0):
    tm = resp.get("timings") or {}
    gpu_s = ((tm.get("prompt_ms") or 0.0) + (tm.get("predicted_ms") or 0.0)) / 1000.0
    return {
        "latency_ms": round((time.time() - t0) * 1000.0, 1),
        "prompt_n": tm.get("prompt_n"),
        "cached_n": tm.get("cache_n"),
        "prompt_ms": round(tm.get("prompt_ms") or 0.0, 1),
        "predicted_ms": round(tm.get("predicted_ms") or 0.0, 1),
        "predicted_n": tm.get("predicted_n"),
        "gpu_s": round(gpu_s, 4),
        "usd_per_1000": round(gpu_s * usd_per_gpu_hour / 3.6, 5),
    }


# ----------------------------------------------------------------- two modes
def ask_jev(question, options, image, timeout=600, usd=1.0, cache=CACHE):
    """判据：字母 + 概率。全部归一化逻辑复用 image_jev.decide()。"""
    r = IJ.decide(question, options, image=image, cache=cache,
                  timeout=timeout, usd_per_gpu_hour=usd)
    r.pop("image", None)                      # 别把几 MB 的 data URI 回给页面
    labels = list(r.get("labels") or [])       # decide() 给的是字符串 "ABCD"，转成数组给前端
    ans = r.get("answer")
    return {
        "mode": "jev",
        "has_image": bool(image),
        "prompt": IJ.build_prompt(question, options, labels),
        "option_texts": list(options),
        "labels": labels,
        "probs": r.get("probs") or {},
        "mass": r.get("mass"),
        "answer": {
            "label": ans,
            "text": options[labels.index(ans)] if ans in labels else None,
            "prob": r.get("prob"),
            "raw_token": r.get("raw_token"),
            "valid": r.get("valid"),
        } if ans else None,
        "text": None,
        "image_sha1_8": (r.get("image_sha1") or "")[:8] or None,
        "image_bytes": r.get("image_bytes"),
        "latency_ms": r.get("latency_ms"),
        "prompt_n": r.get("prompt_n"),
        "cached_n": None,
        "prompt_ms": r.get("prompt_ms"),
        "predicted_ms": r.get("predicted_ms"),
        "gpu_s": r.get("gpu_s"),
        "usd_per_1000": r.get("usd_per_1000"),
        "error": r.get("error"),
    }


def ask_desc(question, image, max_tokens=320, timeout=600, usd=1.0, cache=CACHE,
             sys_prompt=SYS_DESC, stop_key=None):
    """生成式：最常见的识图问答 —— 直接给一句话答案。"""
    payload = {
        "max_tokens": int(max_tokens),
        "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
        "repeat_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": _content(question, image)}],
        "cache_prompt": bool(cache),
        "stream": False,
    }
    t0 = time.time()
    try:
        resp = _post("/v1/chat/completions", payload, timeout, stop_key=stop_key)
    except Exception as e:
        out = {"mode": "gen", "has_image": bool(image), "prompt": question,
               "option_texts": [], "labels": [], "probs": {}, "mass": None,
               "answer": None, "text": None,
               "error": "%s: %s" % (type(e).__name__, str(e)[:300])}
        out.update({k: None for k in ("latency_ms", "prompt_n", "cached_n",
                                      "prompt_ms", "predicted_ms", "gpu_s",
                                      "usd_per_1000")})
        out["latency_ms"] = round((time.time() - t0) * 1000.0, 1)
        return out
    ch = (resp.get("choices") or [{}])[0]
    txt = ((ch.get("message") or {}).get("content") or "").strip()
    out = {"mode": "gen", "has_image": bool(image), "prompt": question,
           "option_texts": [], "labels": [], "probs": {}, "mass": None,
           "answer": None, "text": txt, "error": None,
           "image_sha1_8": None, "image_bytes": None}
    out.update(_timings(resp, t0, usd))
    return out


def ask(req):
    q = (req.get("question") or "").strip() or "这张图里是什么？"
    img = req.get("image") or None
    opts = [str(o).strip() for o in (req.get("options") or []) if str(o).strip()]
    mode = (req.get("mode") or "").strip().lower() or ("jev" if opts else "gen")
    usd = float(req.get("usd_per_gpu_hour") or 1.0)
    timeout = float(req.get("timeout") or 600)
    cache = bool(req.get("cache", CACHE))
    stop_key = (req.get("stop_key") or "").strip() or None

    if mode == "jev" and len(opts) < 2:
        return {"error": "判据模式至少需要 2 个选项"}

    def once(with_image):
        if mode == "jev":
            # 判据模式是 max_tokens=1 的"一次前向"，没有可中止的解码过程，不接 stop_key
            return ask_jev(q, opts, img if with_image else None,
                           timeout=timeout, usd=usd, cache=cache)
        return ask_desc(q, img if with_image else None,
                        max_tokens=int(req.get("max_tokens") or 320),
                        timeout=timeout, usd=usd, cache=cache,
                        sys_prompt=SYS_DESC if with_image else SYS_DESC_BLIND,
                        stop_key=stop_key)

    out = once(True)
    out["mode"] = mode
    out["has_image"] = bool(img)
    if req.get("control"):
        out["control"] = once(False)
    else:
        out["control"] = None
    return out


# ------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # keep the console readable
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        raw = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path      # drop ?query
        if path in ("/", "/index.html"):
            page = ROOT / "index.html"
            if not page.exists():
                return self._send(500, {"error": "index.html missing"})
            return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        if path.startswith("/api/health"):
            try:
                h = _health()
                props = {}
                try:
                    with urllib.request.urlopen(UPSTREAM + "/props", timeout=3) as r:
                        props = json.loads(r.read().decode("utf-8"))
                except Exception:
                    pass
                mp = props.get("model_path", "")
                mod = props.get("modalities") or {}
                return self._send(200, {
                    "up": True, "status": h, "model": mp,
                    "vision": bool(mod.get("vision")),
                    "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
                })
            except Exception as e:
                return self._send(200, {"up": False, "error": str(e)})
        if path.startswith("/api/demo"):
            # 只给页面的 ?demo=1 用：把一张示例图交给浏览器自动跑一遍（截图/演示）。
            p = Path(os.environ.get("JEV_DEMO_IMAGE")
                     or r"F:\super\qwen27b\llmchat\mem\chess.png")
            if not p.exists():
                return self._send(404, {"error": "demo image not found: %s" % p})
            url, sha, n = IJ.data_url(str(p))
            return self._send(200, {"name": p.name, "bytes": n, "sha1_8": sha[:8], "image": url})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path.startswith("/api/stop"):
            key = ""
            try:
                n = int(self.headers.get("Content-Length") or 0)
                key = (json.loads(self.rfile.read(n).decode("utf-8")) or {}).get("stop_key") or ""
            except Exception:
                pass
            return self._send(200, {"ok": True, "killed": bool(key) and _abort(key)})
        if not path.startswith("/api/ask"):
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception as e:
            return self._send(400, {"error": "bad json: %s" % e})
        try:
            return self._send(200, ask(req))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            return self._send(200, {"error": "upstream HTTP %s: %s" % (e.code, detail)})
        except Exception as e:
            return self._send(200, {"error": "%s: %s" % (type(e).__name__, e)})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else LISTEN[1]
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    print("Image JEV  ->  http://127.0.0.1:%d/" % port)
    print("looks go to %s" % UPSTREAM)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
