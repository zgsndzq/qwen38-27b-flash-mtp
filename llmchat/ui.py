#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Input and display: TUI/paste input, streaming printer, meta commands.

prompt_toolkit is optional; without it we fall back to single-line input
plus the /paste mode.
"""

import sys

EXIT_CMDS = ("quit", "exit", "q", "/quit", "/exit")
CLEAR_CMDS = ("clear", "/clear", "cls", "/cls")
PASTE_CMDS = ("/paste",)
END_CMDS = ("/end", ":::end")

# Force UTF-8 when stdout is redirected: Windows' default cp936 mangles
# non-ASCII. A real TTY is left alone for Python's console writer.
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
    """Invert the multiline keys: Enter submits, Esc+Enter inserts a newline.

    The default is the other way round, which makes a chat session look hung.
    """
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
    print("paste mode: paste, then a single /end line to finish (blank lines kept)")
    lines = []
    while True:
        part = input("... ")
        if part.strip() in END_CMDS:
            break
        lines.append(part)
    return "\n".join(lines).strip("\n") if lines else ""


def read_user(session=None, prompt="you> "):
    """Single entry point: TUI multiline when available, /paste otherwise."""
    if HAS_TUI and sys.stdin.isatty():
        while True:
            text = (session or new_session()).prompt(prompt)
            if text.strip():
                return text
    return _read_plain(prompt)


def stream_printer(hide_thinking=False):
    """on_delta(kind, text): thinking and answer printed as they arrive.

    The returned callable carries .streamed, so the caller knows whether
    anything was already printed.
    """
    st = {"thinking": False, "content": False}

    def on_delta(kind, text):
        if kind == "thinking":
            st["content"] = False
            if hide_thinking:
                if not st["thinking"]:
                    print("\n[thinking...]", end="", flush=True)
                    st["thinking"] = True
                return
            if not st["thinking"]:
                print("\n[thinking] ", end="", flush=True)
                st["thinking"] = True
            print(text, end="", flush=True)
        else:
            st["thinking"] = False
            if not st["content"]:
                print("\n\n[answer] ", end="", flush=True)
                st["content"] = True
            print(text, end="", flush=True)

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
