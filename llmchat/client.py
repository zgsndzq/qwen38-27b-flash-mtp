#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Everything that talks to llama-server: probe, chat, slots."""

import json
import socket
import urllib.request
from dataclasses import dataclass

from log import log


def _given(fields):
    """Drop the fields left at None: absent means "keep the server default"."""
    return {k: v for k, v in fields.items() if v is not None}


@dataclass
class ChatRequest:
    """One /v1/chat/completions call, built from whatever the caller set."""

    messages: list
    temperature: float = None
    max_tokens: int = None
    think: bool = None
    effort: str = None

    def body(self):
        return {
            "model": "gguf",
            "messages": self.messages,
            "stream": True,
            **_given({"temperature": self.temperature, "max_tokens": self.max_tokens}),
            "chat_template_kwargs": _given({"enable_thinking": self.think,
                                            "reasoning_effort": self.effort}),
        }


class Client:
    """One llama-server endpoint."""

    def __init__(self, base_url, timeout):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._n_ctx = 0
        self._slot_off = False       # server has no --slot-save-path; stop retrying

    # ---- liveness and size ----------------------------------------------
    def probe(self):
        """/v1/models check. Returns (ok, model id or None, error)."""
        try:
            with urllib.request.urlopen(self.base_url + "/v1/models", timeout=5) as r:
                rows = json.loads(r.read().decode("utf-8")).get("data") or []
            return True, (rows[0].get("id") if rows else None), None
        except Exception as e:
            return False, None, e

    def n_ctx(self):
        """Slot size from /props, asked once. 0 when the server will not say."""
        if self._n_ctx:
            return self._n_ctx
        try:
            with urllib.request.urlopen(self.base_url + "/props", timeout=3) as r:
                d = json.loads(r.read().decode("utf-8"))
            n = (d.get("default_generation_settings") or {}).get("n_ctx") or d.get("n_ctx")
            self._n_ctx = int(n or 0)
        except Exception:
            pass
        return self._n_ctx

    # ---- chat ------------------------------------------------------------
    def chat(self, req, on_delta=None):
        """POST a ChatRequest and stream it. Returns (reasoning, content, timings)."""
        http = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(req.body(), ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"})

        reasoning, content = [], []
        timings = None
        try:
            with urllib.request.urlopen(http, timeout=self.timeout) as resp:
                for raw in resp:                   # one SSE line at a time
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    if "timings" in obj:           # the last chunk carries them
                        timings = obj["timings"]
                    delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                    self._collect("thinking", delta.get("reasoning_content"),
                                  reasoning, on_delta)
                    self._collect("content", delta.get("content"), content, on_delta)
        except Exception as e:
            raise ConnectionError(
                f"llama-server call failed ({self.base_url}): {e}{self._hint(e)}") from e
        return "".join(reasoning), "".join(content), (timings or {})

    @staticmethod
    def _collect(kind, text, sink, on_delta):
        if not text:
            return
        sink.append(text)
        if on_delta:
            on_delta(kind, text)

    def _hint(self, e):
        if isinstance(e, socket.timeout):
            return (f"\nNo data within {self.timeout:.0f}s. On a long prompt that is prompt "
                    f"eval working, not a dead server -- raise --timeout or "
                    f"LLAMA_CHAT_TIMEOUT.")
        if "400" in str(e):
            return ("\nThe server rejected the request (400), usually because the session "
                    "is larger than the slot. Type clear and continue.")
        return ""

    # ---- slots -----------------------------------------------------------
    def slot(self, action, filename=None, slot_id=0):
        """POST /slots/<id>?action=save|restore|erase; the filename goes in the body."""
        if self._slot_off and action != "erase":
            return None
        http = urllib.request.Request(
            f"{self.base_url}/slots/{int(slot_id)}?action={action}",
            data=json.dumps({"filename": filename} if filename else {}).encode("utf-8"),
            method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(http, timeout=180) as r:
                return json.loads(r.read().decode("utf-8") or "{}")
        except Exception as e:
            # restore fails routinely (first run, file deleted); do not disable save
            if action == "save":
                self._slot_off = True
                log(f"  [slot] save failed: {e}")
            elif action == "restore":
                log(f"  [slot] restore had no effect: {e}")
            return None
