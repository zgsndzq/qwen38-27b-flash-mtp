#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""llama-server HTTP layer: probe, streaming chat, slot control. Stdlib only.

The timeout is "no data from the server", not total wall time: during prompt
eval the server sends nothing, and on a long context that legitimately takes
minutes. When a long prompt times out, raise --timeout; the server is fine.
"""

import json
import os
import socket
import urllib.request

DEFAULT_URL = os.environ.get("LLAMA_CHAT_URL", "http://127.0.0.1:18080")
DEFAULT_TIMEOUT = float(os.environ.get("LLAMA_CHAT_TIMEOUT", "600"))

# Denominator for the context gauge: ask the server /props for n_ctx, fall
# back to this environment variable only when the probe fails.
CTX_LIMIT = int(os.environ.get("LLAMA_CHAT_CTX", "131072"))
_CTX_PROBED = None

_slot_off = False       # set when the server has no --slot-save-path; stops retries


def probe(base_url, timeout=5.0):
    """/v1/models liveness check. Returns (ok, model id or None, error)."""
    try:
        with urllib.request.urlopen(base_url + "/v1/models", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rows = data.get("data") or []
        return True, (rows[0].get("id") if rows else None), None
    except Exception as e:
        return False, None, e


def ctx_limit(base_url):
    """Real n_ctx from the server; CTX_LIMIT if it cannot be read. Probed once."""
    global _CTX_PROBED
    if _CTX_PROBED is None:
        _CTX_PROBED = 0
        try:
            with urllib.request.urlopen(base_url + "/props", timeout=3) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            n = (d.get("default_generation_settings") or {}).get("n_ctx") or d.get("n_ctx")
            if n:
                _CTX_PROBED = int(n)
        except Exception:
            pass
        if not _CTX_PROBED:
            _CTX_PROBED = CTX_LIMIT
    return _CTX_PROBED


def chat_sse(messages, base_url, temperature, max_new_tokens=None, on_delta=None,
             think=None, effort=None, timeout=DEFAULT_TIMEOUT, id_slot=None,
             cache_prompt=None):
    """Streaming chat. Calls on_delta(kind, text); returns (reasoning, content, timings).

    temperature: a number overrides the server --temp, None sends nothing.
    think=True/False -> chat_template_kwargs.enable_thinking (overrides -rea).
    effort -> chat_template_kwargs.reasoning_effort.
    """
    body = {"model": "gguf", "messages": messages, "stream": True}
    if id_slot is not None:
        body["id_slot"] = int(id_slot)     # pin to a slot so multi-session KV reuse works
    if cache_prompt is not None:
        body["cache_prompt"] = bool(cache_prompt)
    if temperature is not None:
        body["temperature"] = temperature
    if max_new_tokens is not None:
        body["max_tokens"] = max_new_tokens
    kwargs = {}
    if think is not None:
        kwargs["enable_thinking"] = bool(think)
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    if kwargs:
        body["chat_template_kwargs"] = kwargs

    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"})

    reasoning, content = [], []
    timings = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:                       # one SSE line at a time
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "timings" in obj:               # the last chunk carries them
                    timings = obj["timings"]
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                r = delta.get("reasoning_content")
                c = delta.get("content")
                if r:
                    reasoning.append(r)
                    if on_delta:
                        on_delta("thinking", r)
                if c:
                    content.append(c)
                    if on_delta:
                        on_delta("content", c)
    except Exception as e:
        hint = ""
        if isinstance(e, socket.timeout):
            hint = (f"\nNo data from the server within {timeout:.0f}s. On a long prompt or "
                    f"context this is prompt eval taking its time, not a dead server.\n"
                    f"Raise LLAMA_CHAT_TIMEOUT (seconds) or pass --timeout 1200.")
        elif "400" in str(e):
            hint = ("\nThe server rejected the request (400). The usual cause is the "
                    "session exceeding the per-slot context size.\nType clear to start "
                    "a new context and continue.")
        raise ConnectionError(
            f"llama-server streaming call failed ({base_url}): {e}{hint}") from e
    return "".join(reasoning), "".join(content), (timings or {})


def slot_action(base_url, action, filename=None, timeout=180, slot_id=0):
    """POST /slots/<id>?action=save|restore|erase; the filename goes in the body."""
    global _slot_off
    if _slot_off and action != "erase":
        return None
    url = f"{base_url}/slots/{int(slot_id)}?action={action}"
    body = json.dumps({"filename": filename} if filename else {}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "{}")
    except Exception as e:
        # restore fails routinely (first run, file deleted); do not disable save
        if action == "save" and not _slot_off:
            _slot_off = True
            print(f"  [slot] save failed: {e}")
        elif action == "restore":
            print(f"  [slot] restore had no effect: {e}")
        return None
