#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JEV x Tetris -- bridge + UI server.

The page is same-origin, and every AI move is exactly one JEV-style call to the
local llama-server on 127.0.0.1:18080:

    max_tokens = 1  +  logprobs  +  top_logprobs = 40

i.e. one forward pass, read the logits of the candidate letter slots, decode
zero extra tokens.  The bridge only exists so the prompt lives in one place and
the browser does not have to deal with CORS.

Stdlib only.  Python 3.8+.
"""

import http.client
import json
import math
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
UPSTREAM = "http://127.0.0.1:18080"
LISTEN = ("127.0.0.1", 18090)
LABELS = "ABCDEFGH"

SYS = (
    "You choose the best move in Tetris, on a board 10 columns wide and 20 rows tall. "
    "For every legal placement you are given r rotation, c column, L lines cleared, "
    "H holes left, T resulting top height, B bumpiness, and S the engine's quality score. "
    "Higher S is always better: answer the letter with the largest S. "
    "(If S is absent, prefer the most L, then the fewest H, then the lowest T.) "
    "Answer with exactly one uppercase letter from the list."
)


# --------------------------------------------------------------------- prompt
def build_user(state):
    """state -> one user message.  Everything before the dynamic block is fixed,
    so the server's prompt cache can reuse it."""
    cands = state.get("cands") or []
    piece = state.get("piece", "?")
    lines = []
    if state.get("repr") == "grid":
        lines.append("BOARD")
        lines.extend(state.get("grid") or [])
    else:
        lines.append("COLUMN HEIGHTS (left to right): " + ",".join(
            str(int(h)) for h in (state.get("heights") or [])))
    lines.append("PIECE: " + str(piece))
    lines.append("PLACEMENTS")
    for c in cands:
        lines.append("%s r%d c%d L%d H%d T%d B%d S%d" % (
            c["label"], int(c.get("rot", 0)), int(c.get("col", 0)),
            int(c.get("L", 0)), int(c.get("H", 0)), int(c.get("T", 0)),
            int(c.get("B", 0)), int(c.get("S", 0))))
    lines.append("ANSWER:")
    return "\n".join(lines)


def messages(state):
    return [
        {"role": "system", "content": SYS},
        {"role": "user", "content": build_user(state)},
    ]


# ----------------------------------------------------------------- stop channel
# Pressing stop must actually kill the upstream connection; a cancel request does
# nothing. A non-streaming request on the server waits on req.should_stop
# (tools/server/server-context.cpp:4558) and that only reacts to the TCP connection
# dropping; the DELETE /v1/stream route applies only to streaming requests carrying
# X-Conversation-Id (server-stream.cpp:622-628). So the only thing that really stops
# the GPU from the bridge is closing this upstream connection from another thread -
# the server sees the drop and cancels the slot immediately.
# The page sends a stop_key with each request; the connection is registered under it.
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
    body = json.dumps(payload).encode("utf-8")
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


def _norm_token(tok):
    """'A' / ' A' / 'A.' / 'A)' -> 'A'."""
    t = (tok or "").strip()
    if not t:
        return None
    c = t[0].upper()
    if c not in LABELS:
        return None
    if len(t) > 1:
        rest = t[1:].strip()
        if rest and any(ch.isalnum() for ch in rest):
            return None          # 'APPLE' is not option A
    return c


def _softmax_restricted(got, labels):
    """got: {label: logprob or None} -> normalised probs over the labels."""
    vals = []
    for L in labels:
        v = got.get(L)
        vals.append(-30.0 if v is None else float(v))
    m = max(vals)
    ws = [math.exp(v - m) for v in vals]
    s = sum(ws) or 1.0
    return {L: w / s for L, w in zip(labels, ws)}


def decide(state, stop_key=None):
    t0 = time.time()
    msgs = messages(state)
    mode = state.get("mode") or "jev"
    labels = [c["label"] for c in (state.get("cands") or [])]

    if mode == "jev":
        payload = {
            "max_tokens": 1,
            "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
            "repeat_penalty": 1.0,
            "logprobs": True, "top_logprobs": 40,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    elif mode == "gen":
        payload = {
            "max_tokens": 12, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    else:  # think
        payload = {
            "max_tokens": 512, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    payload.update({"messages": msgs, "cache_prompt": True, "stream": False})

    resp = _post("/v1/chat/completions", payload, stop_key=stop_key)
    wall = (time.time() - t0) * 1000.0
    choice = (resp.get("choices") or [{}])[0]
    tm = resp.get("timings") or {}

    out = {
        "mode": mode,
        "wall_ms": round(wall, 1),
        "prompt_n": tm.get("prompt_n"),
        "cached_n": tm.get("cache_n"),
        "prompt_ms": round(tm.get("prompt_ms") or 0.0, 1),
        "predicted_ms": round(tm.get("predicted_ms") or 0.0, 1),
        "predicted_n": tm.get("predicted_n"),
        "prompt": build_user(state),
        "labels": labels,
    }

    if mode == "jev":
        lp = (choice.get("logprobs") or {}).get("content") or []
        top = (lp[0].get("top_logprobs") or []) if lp else []
        by_tok = {}
        for ent in top:
            L = _norm_token(ent.get("token"))
            if L and L not in by_tok:
                by_tok[L] = ent.get("logprob")
        probs = _softmax_restricted(by_tok, labels)
        best = max(labels, key=lambda L: probs[L]) if labels else None
        out.update({"label": best, "probs": probs,
                    "prob": round(probs[best], 4) if best else None,
                    "seen": sorted(by_tok.keys()),
                    "text": None})
    else:
        txt = (choice.get("message") or {}).get("content") or ""
        pick = None
        for ch in txt:
            if ch.upper() in labels:
                pick = ch.upper()
                break
        out.update({"label": pick, "probs": None, "prob": None,
                    "text": txt.strip()[:400]})
    return out


# ------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):      # keep the console readable
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
                return self._send(200, {"up": True, "status": h,
                                        "model": props.get("model_path", "")})
            except Exception as e:
                return self._send(200, {"up": False, "error": str(e)})
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
        if not path.startswith("/api/decide"):
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            state = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception as e:
            return self._send(400, {"error": "bad json: %s" % e})
        try:
            key = (state.get("stop_key") or "").strip() or None
            return self._send(200, decide(state, stop_key=key))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            return self._send(200, {"error": "upstream HTTP %s: %s" % (e.code, detail)})
        except Exception as e:
            return self._send(200, {"error": "%s: %s" % (type(e).__name__, e)})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else LISTEN[1]
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    print("JEV x Tetris  ->  http://127.0.0.1:%d/" % port)
    print("decisions go to %s" % UPSTREAM)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
