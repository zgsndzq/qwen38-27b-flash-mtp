#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Session persistence, KV slot archive, render frames.

With --slot-save-path the server can write a whole KV state (including the GDN
recurrent state and the MTP draft context) to a file and read it back, so a
restart resumes the same conversation without re-prefilling the prompt.

This only works when the new request is an extension of the stored tokens.
sys_frame / think_frame / effort_frame decide the prefix the template emits;
if any of them changes, the server drops the whole KV and re-computes. That is
why every session_save must write all three frames -- a missing field makes the
next start replay with the default frame and the prefix forks at the system
block.
"""

import json
import os
import urllib.parse
from pathlib import Path

from srvapi import slot_action

DATA_DIR = Path(__file__).resolve().parent / "data"
SESSION_FILE = DATA_DIR / "llmchat_session.json"
SLOT_NAME = "llmchat.slot"
SLOT_DIR = Path(__file__).resolve().parent.parent / "data" / "slots"
SLOT_FILE = SLOT_DIR / SLOT_NAME

SLOT_MIN = int(os.environ.get("LLAMA_SLOT_MIN", "8192"))     # shorter than this is not worth a file
SLOT_GROW = int(os.environ.get("LLAMA_SLOT_GROW", "2048"))   # re-save only after this much growth (28 KB/token)


def apply_profile(base_url):
    """Add a port suffix to the session and slot names so two models never mix.

    The default port 18080 keeps the plain names; others (18081 = LFM) get
    _<port>. base_url must already be rstrip('/')'d.
    """
    global SESSION_FILE, SLOT_NAME, SLOT_FILE
    try:
        port = str(urllib.parse.urlparse(base_url).port or "")
    except ValueError:
        port = ""
    if not port or port == "18080":
        return
    SESSION_FILE = DATA_DIR / f"llmchat_session_{port}.json"
    SLOT_NAME = f"llmchat_{port}.slot"
    SLOT_FILE = SLOT_DIR / SLOT_NAME


def ensure_slot_dir():
    """Recreate the slot directory: it can vanish if data/ is cleaned while the
    server keeps running, and the server refuses to write a slot without it."""
    try:
        SLOT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def session_save(messages, tokens, slot_tokens=None, sys_frame=None,
                 effort_frame=None, think_frame=None):
    """Save the session. slot_tokens = token count of the last saved slot."""
    try:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"messages": messages, "tokens": int(tokens), "slot": SLOT_NAME,
                "sys_frame": sys_frame, "effort_frame": effort_frame,
                "think_frame": think_frame}
        if slot_tokens is not None:
            data["slot_tokens"] = int(slot_tokens)
        else:
            old = session_load() or {}
            if old.get("slot_tokens"):
                data["slot_tokens"] = int(old["slot_tokens"])
        SESSION_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def session_load():
    try:
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def session_clear(base_url):
    """Drop the archive, the in-memory slot and the slot files on disk.

    The server's erase only clears the slot in memory. The file written by
    --slot-save-path stays on disk, so a later start would restore a stale
    prefix of hundreds of MB that no longer matches the (now empty) history.
    """
    try:
        SESSION_FILE.unlink()
    except OSError:
        pass
    slot_action(base_url, "erase")
    for p in (SLOT_FILE, SLOT_FILE.parent / (SLOT_NAME + ".draft")):
        try:
            p.unlink()
        except OSError:
            pass
