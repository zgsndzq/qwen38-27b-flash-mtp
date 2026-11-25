# -*- coding: utf-8 -*-
"""One directly-written memory block. No training, no weights touched.

The key was written offline with Google EmbeddingGemma 2 (CPU, -ngl 0).
At prompt time we match the query, and if it hits we hand the whole block
to the model as a system message. Symbol matching works with zero services;
an optional CPU embedding server adds semantic recall.
"""
import json
import math
import os
import urllib.request
from pathlib import Path

from log import log

HERE = Path(__file__).resolve().parent / "recall"
EMBED_URL = os.environ.get("RECALL_EMBED_URL", "http://127.0.0.1:18090")
DIM = 256
COS_MID = 0.68

_blocks = None


def _load():
    global _blocks
    if _blocks is None:
        cfg = json.loads((HERE / "blocks.json").read_text(encoding="utf-8"))
        for b in cfg:
            b["text"] = (HERE / b["text_file"]).read_text(encoding="utf-8")
        _blocks = cfg
    return _blocks


def _sym(q, symbols):
    return min(sum(1 for s in symbols if s in q), 4) / 4.0


def _embed(q):
    """Optional: ask the CPU embedding server. Never blocks more than 0.4s."""
    if os.environ.get("RECALL_EMBED", "1") == "0":
        return None
    body = json.dumps({"input": ["task: search result | query: " + q]},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(EMBED_URL + "/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=0.4) as r:
            d = json.loads(r.read().decode("utf-8"))
        v = d["data"][0]["embedding"][:DIM]
    except Exception:
        return None
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def apply(messages, q):
    """Return messages with the recalled block merged in, or unchanged on a miss."""
    if os.environ.get("RECALL", "1") == "0":
        return messages
    for b in _load():
        sym = _sym(q, b["symbols"])
        vec = _embed(q)
        if vec is None:
            hit, cos, why = sym > 0, None, "symbol"
        else:
            cos = sum(a * c for a, c in zip(vec, b["key"]))
            logit = (cos - COS_MID) * 40.0 + sym * 6.0
            hit = 1 / (1 + math.exp(-logit)) > 0.5
            why = "cos+sym"
        if not hit:
            log(f"[recall] miss  {b['id']} ({why})")
            continue
        tag = "以下是从记忆层提取到的内容，请据此作答：\n\n" + b["text"]
        log(f"[recall] HIT   {b['id']} ({why}"
            + (f" cos={cos:+.3f}" if cos is not None else "") + f" sym={sym:.2f})"
            + f" +{len(b['text'])} chars")
        out = list(messages)
        if out and out[0]["role"] == "system" and isinstance(out[0]["content"], str):
            out[0] = {"role": "system", "content": out[0]["content"] + "\n\n" + tag}
        else:
            out.insert(0, {"role": "system", "content": tag})
        return out
    return messages
