#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Input and display; prompt_toolkit is optional."""

import sys

from log import log

EXIT_CMDS = ("quit", "exit", "q", "/quit", "/exit")
CLEAR_CMDS = ("clear", "/clear", "cls", "/cls")
PASTE_CMDS = ("/paste",)
END_CMDS = ("/end", ":::end")

# Force UTF-8 when redirected: cp936 mangles non-ASCII. A real TTY is left alone.
if not sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    HAS_TUI = True
except ImportError:
    HAS_TUI = False


def _make_bindings():
    """Enter submits, Esc+Enter inserts a newline (the default is the reverse)."""
    from prompt_toolkit.filters import emacs_insert_mode, vi_insert_mode
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.key_binding.bindings.named_commands import get_by_name

    kb = KeyBindings()
    insert = vi_insert_mode | emacs_insert_mode

    @kb.add("enter", filter=insert)
    def _accept(event):
        get_by_name("accept-line").call(event)

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    return kb


def new_session():
    return PromptSession(multiline=True, history=InMemoryHistory(),
                         key_bindings=_make_bindings())


def _read_plain(prompt="you> "):
    """No TUI: one line, or /paste followed by lines until /end."""
    line = input(prompt).strip()
    if line.lower() not in PASTE_CMDS:
        return line
    log.raw("paste mode: paste, then a single /end line to finish (blank lines kept)")
    lines = []
    while True:
        part = input("... ")
        if part.strip() in END_CMDS:
            break
        lines.append(part)
    return "\n".join(lines).strip("\n") if lines else ""


def read_user(session=None, prompt="you> "):
    """Single entry point: TUI multiline when available, /paste otherwise."""
    if not (HAS_TUI and sys.stdin.isatty()):
        return _read_plain(prompt)
    ps = session or new_session()
    while True:
        text = ps.prompt(prompt)
        if text.strip():
            return text


def stream_printer(hide_thinking=False):
    """on_delta(kind, text); the returned callable carries .streamed."""
    st = {"thinking": False, "content": False}

    def on_delta(kind, text):
        if kind == "thinking":
            st["content"] = False
            if hide_thinking:
                if not st["thinking"]:
                    log.raw("\n[thinking...]")
                    st["thinking"] = True
                return
            if not st["thinking"]:
                log.raw("\n[thinking] ")
                st["thinking"] = True
            log.raw(text)
        else:
            st["thinking"] = False
            if not st["content"]:
                log.raw("\n\n[answer] ")
                st["content"] = True
            log.raw(text)

    on_delta.streamed = st
    return on_delta


def meta(text):
    """Session meta command -> 'exit' / 'clear' / None (None = send to the model)."""
    t = (text or "").strip().lower()
    if t in EXIT_CMDS:
        return "exit"
    if t in CLEAR_CMDS:
        return "clear"
    return None
