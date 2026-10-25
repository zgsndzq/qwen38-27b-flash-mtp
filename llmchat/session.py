#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Session archive, slot files, and the render frames they depend on."""

import json
import os
import urllib.parse
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SESSION_FILE = DATA_DIR / "llmchat_session.json"
SLOT_NAME = "llmchat.slot"
SLOT_DIR = Path(__file__).resolve().parent.parent / "data" / "slots"
SLOT_FILE = SLOT_DIR / SLOT_NAME



def apply_profile(base_url):
    """Name the session and slot files after the port in base_url."""
    global SESSION_FILE, SLOT_NAME, SLOT_FILE
    port = urllib.parse.urlparse(base_url).port or 0
    tag = str(port) if port else "default"
    SESSION_FILE = DATA_DIR / f"llmchat_session_{tag}.json"
    SLOT_NAME = f"llmchat_{tag}.slot"
    SLOT_FILE = SLOT_DIR / SLOT_NAME


def ensure_slot_dir():
    """Recreate the slot directory; the server will not write a slot without it."""
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


def drop_slot_files():
    """Delete the slot files on disk. The server's erase only clears memory."""
    for p in (SLOT_FILE, SLOT_FILE.parent / (SLOT_NAME + ".draft")):
        try:
            p.unlink()
        except OSError:
            pass


def session_clear(client):
    """Drop the archive, the in-memory slot and the slot files."""
    try:
        SESSION_FILE.unlink()
    except OSError:
        pass
    client.slot("erase")
    drop_slot_files()
