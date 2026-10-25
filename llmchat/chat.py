#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""llmchat -- a standalone llama-server chat client (no third-party deps).

    python chat.py                                    # 127.0.0.1:18080
    python chat.py --base-url http://127.0.0.1:18081  # another server
    python chat.py --temp 0.6 --think on --effort low

In-session meta commands (intercepted locally, never sent to the model):
    quit / q / exit        leave
    clear / cls            drop the context (the system prompt is kept)
    not a TTY: /paste enters paste mode, a single /end line finishes it

Environment:
    LLAMA_CHAT_URL         default base_url (--base-url wins)
    LLAMA_CHAT_TIMEOUT     "no data" timeout in seconds, default 600
    LLAMA_SLOT_MIN         shorter than this, no slot file, default 8192
    LLAMA_SLOT_GROW        re-save only after this much growth, default 2048

Layout: srvapi.py is the HTTP layer, session.py keeps the archive and slots,
ui.py owns input and display, this file is the CLI and the main loop.
"""

import argparse
import sys
from pathlib import Path

import session as sess
from srvapi import (DEFAULT_TIMEOUT, DEFAULT_URL, chat_sse, ctx_limit,
                    probe, slot_action)
from ui import HAS_TUI, meta, new_session, read_user, stream_printer

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def fill_sys(text):
    """Expand runtime placeholders in the system prompt: {date}, {weekday}.

    Text without placeholders is returned unchanged, so an old sys.txt still
    works as-is.
    """
    if not text or "{" not in text:
        return text
    import datetime
    d = datetime.date.today()
    return (text.replace("{date}", d.isoformat())
                .replace("{weekday}", WEEKDAYS[d.weekday()]))


def seed_messages(sys_prompt):
    """Message list for a fresh context: a system message only if configured."""
    if sys_prompt is None:
        return []
    return [{"role": "system", "content": sys_prompt}]


def parse_think(v):
    """'on'/'off' -> True/False; 'server' -> None (keep the server's -rea)."""
    if v is None or v == "server":
        return None
    return v == "on"


def build_parser():
    ap = argparse.ArgumentParser(
        prog="chat.py",
        description="llama-server chat client (standalone, stdlib only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python chat.py --temp 0.6 --think on --effort low\n"
               "  python chat.py --base-url http://127.0.0.1:18081 --think server\n")
    ap.add_argument("--base-url", default=DEFAULT_URL,
                    help=f"llama-server address (default {DEFAULT_URL})")
    ap.add_argument("--sys", default=None,
                    help="system prompt; without it no system message is sent")
    ap.add_argument("--sys-file", default=None,
                    help="read the system prompt from a UTF-8 file (wins over --sys)")
    ap.add_argument("--no-sys", action="store_true",
                    help="send no system message at all (raw model call)")
    ap.add_argument("--temp", type=float, default=None,
                    help="sampling temperature; omit to keep the server --temp")
    ap.add_argument("--think", choices=["on", "off", "server"], default="on",
                    help="thinking switch (default on; the server is on + low too); "
                         "server = keep the server's -rea")
    ap.add_argument("--effort", choices=["xhigh", "medium", "low"], default=None,
                    help="thinking effort; omit to keep the server default (low)")
    ap.add_argument("--timeout", type=float, default=None,
                    help=f"no-data timeout in seconds (default {DEFAULT_TIMEOUT:.0f})")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="per-turn answer limit; omit for no limit (run to EOS)")
    ap.add_argument("--hide-thinking", action="store_true",
                    help="do not print the thinking block (the request still thinks)")
    ap.add_argument("--no-keep-thinking", dest="keep_thinking", action="store_false",
                    help="do not keep the thinking block in history. Dropping it makes "
                         "the next prompt shorter than the server KV, so the shared "
                         "prefix breaks at the first thinking turn and everything "
                         "is re-computed")
    ap.add_argument("--no-tui", action="store_true",
                    help="disable the multiline TUI, use single line + /paste")
    ap.add_argument("--once", metavar="QUESTION", default=None,
                    help="non-interactive: ask this one question and exit")
    ap.add_argument("--save", metavar="FILE", default=None,
                    help="write the final answer to a UTF-8 file (with --once)")
    ap.set_defaults(keep_thinking=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)

    if a.no_sys:
        sys_prompt = None
    elif a.sys_file:
        try:
            # utf-8-sig strips the BOM Notepad leaves at the head of the prompt
            with open(a.sys_file, encoding="utf-8-sig") as f:
                sys_prompt = f.read().strip()
        except OSError as e:
            print(f"[ERROR] cannot read --sys-file {a.sys_file}: {e}")
            return 2
        if not sys_prompt:
            print(f"[ERROR] --sys-file is empty: {a.sys_file}")
            return 2
    else:
        # no built-in system prompt: opt in via --sys / --sys-file. A built-in
        # fallback would change the prefix between runs and invalidate the slot
        # and the prompt cache from the system block on.
        sys_prompt = a.sys

    if sys_prompt is not None:
        sys_prompt = fill_sys(sys_prompt)       # {date} / {weekday}
    sys_frame = sys_prompt
    think_flag = parse_think(a.think)
    timeout = a.timeout or DEFAULT_TIMEOUT
    base_url = a.base_url.rstrip("/")
    sess.apply_profile(base_url)
    sess.ensure_slot_dir()

    ok, model_id, err = probe(base_url)
    think_txt = ("server default" if think_flag is None else ("on" if think_flag else "off"))
    print(f"server: {base_url}  "
          f"temp={a.temp if a.temp is not None else 'server default'}  "
          f"thinking={think_txt}" + (f"({a.effort})" if a.effort else ""))
    if ok:
        print(f"model: {model_id or '(not reported)'}")
    else:
        print(f"[warn] probe failed: {err}\n       (start llmserver.bat first)")

    print("\nsystem: %s" % (sys_prompt if sys_prompt is not None else "(none - raw call)"))
    tip = ("paste a large document and it is submitted in one block; Esc+Enter adds a line"
           if (HAS_TUI and sys.stdin.isatty() and not a.no_tui)
           else "type '/paste' to paste a large document")
    print(f"type 'quit' to exit, 'clear' to drop the context, {tip}\n")

    session = None
    if HAS_TUI and sys.stdin.isatty() and not a.no_tui:
        session = new_session()

    messages = seed_messages(sys_prompt)

    # Previous session: messages + slot.
    saved = sess.session_load()
    need_restore = False
    if saved and saved.get("messages"):
        messages = saved["messages"]
        need_restore = True

        # The frames only decide whether the old slot is still usable; they never
        # override what the user chose this time. A changed frame forks the prefix
        # at the first token, so the slot is dropped and we run with the new one.
        frame_diff = []
        if "sys_frame" in saved and saved.get("sys_frame") != sys_prompt:
            frame_diff.append("system")
        if "think_frame" in saved and saved.get("think_frame") != think_flag:
            frame_diff.append("thinking %s -> %s" % (
                "on" if saved.get("think_frame") else "off", think_txt))
        if "effort_frame" in saved and saved.get("effort_frame") != a.effort:
            frame_diff.append("effort %s -> %s" % (saved.get("effort_frame"), a.effort))

        if frame_diff:
            need_restore = False
            slot_action(base_url, "erase")      # drop it, so the next start does not read hundreds of MB
            print("(saved frames differ: %s)" % "; ".join(frame_diff))
            print("  slot dropped, this turn is a full re-compute; later turns cache again.")

        # The system message in history must match this run's frame (no-op if equal)
        if messages and messages[0]["role"] == "system":
            messages = seed_messages(sys_prompt) + messages[1:]
        elif sys_prompt is not None:
            messages = seed_messages(sys_prompt) + messages
        _frame = "none" if sys_prompt is None else "system %d chars" % len(sys_prompt)
        print("(frames: %s, thinking=%s, effort=%s)" % (
            _frame, think_txt, a.effort or "template default"))
        print(f"resumed the previous session: {len(messages)} messages, "
              f"about {saved.get('tokens') or 0} tok (type clear to start over)\n")

    while True:
        if a.once is not None:
            user, once = a.once, True
            print(f"you> {user}")
        else:
            once = False
            try:
                user = read_user(session, "you> ")
            except (EOFError, KeyboardInterrupt):
                break
        if not user:
            break

        cmd = meta(user)                     # meta commands never enter the context
        if cmd == "exit":
            break
        if cmd == "clear":
            messages = seed_messages(sys_prompt)
            sess.session_clear(base_url)     # the archive and the slot go together
            _n_sys = 0 if sys_prompt is None else 1
            _tail = ", no system" if sys_prompt is None else ""
            print(f"context cleared ({len(messages) - _n_sys} messages{_tail}; "
                  f"input history cleared)\n")
            continue

        # History always keeps exactly what was sent, and is never rewritten:
        # editing a sent message forks the prefix, which invalidates both the
        # prompt cache and the slot, and every turn becomes a full re-compute.
        sent_msgs = messages + [{"role": "user", "content": user}]

        # The slot is read back once, on the first turn after a resume. The server
        # takes the longest common prefix, so an edited history degrades to a
        # re-compute instead of reading wrong KV.
        restored_n = 0
        if need_restore:
            need_restore = False
            # If the file is gone, do not send the request: the server answers 400
            # "No available space in KV cache", which reads like a full cache.
            if (saved.get("tokens") or 0) >= sess.SLOT_MIN and sess.SLOT_FILE.exists():
                r = slot_action(base_url, "restore", sess.SLOT_NAME)
                if r and r.get("n_restored"):
                    restored_n = int(r["n_restored"])
                    print(f"  [slot] restored {restored_n} tok")

        try:
            printer = stream_printer(hide_thinking=a.hide_thinking)
            reasoning, content, timings = chat_sse(
                sent_msgs, base_url, a.temp, a.max_tokens,
                printer, think=think_flag, effort=a.effort, timeout=timeout)
        except ConnectionError as e:
            print(f"\nserver unavailable: {e}\n")
            if once:
                return 1
            continue

        # "Restored" is not the same as "reused this turn": prompt_n is the only
        # trustworthy evidence.
        if restored_n:
            pp_n = int(timings.get("prompt_n") or 0)
            if pp_n > max(256, restored_n // 8):
                print(f"  [slot] missed: restored {restored_n} tok but re-computed "
                      f"{pp_n} tok (saved frames differ, type clear to rebuild)")

        if printer.streamed["content"] or printer.streamed["thinking"]:
            print()                          # already streamed, just end the line
        else:
            print(f"\nmodel> {content}")

        messages.append({"role": "user", "content": user})
        # The thinking block has to go back into history: the template renders
        # <think>...</think> only when an assistant message carries
        # reasoning_content, and the server KV already holds those tokens.
        # Dropping them makes the next prompt shorter than the KV, the shared
        # prefix breaks at the first thinking turn, and everything is re-computed.
        assistant = {"role": "assistant", "content": content}
        if reasoning and a.keep_thinking:
            assistant["reasoning_content"] = reasoning
        messages.append(assistant)

        if a.save:
            try:
                Path(a.save).write_text(content, encoding="utf-8")
                print(f"  [saved] {a.save}")
            except OSError as e:
                print(f"  [save failed] {e}")

        ctx = (timings.get("prompt_n") or 0) + (timings.get("cache_n") or 0)
        if ctx:
            limit = ctx_limit(base_url)
            print(f"  [ctx {ctx} tok - {100.0 * ctx / limit:.0f}% of {limit}]")
            sess.session_save(messages, ctx, sys_frame=sys_frame,
                              effort_frame=a.effort, think_frame=think_flag)
            prev = int(saved.get("slot_tokens") or 0) if isinstance(saved, dict) else 0
            if ctx >= sess.SLOT_MIN and (not sess.SLOT_FILE.exists()
                                         or ctx - prev >= sess.SLOT_GROW):
                r = slot_action(base_url, "save", sess.SLOT_NAME)
                if r and r.get("n_saved"):
                    if not isinstance(saved, dict):
                        saved = {}
                    saved["slot_tokens"] = ctx
                    # all three frames in the same write, or the next start replays
                    # with default frames, forks the prefix and the slot is wasted
                    sess.session_save(messages, ctx, slot_tokens=ctx, sys_frame=sys_frame,
                                      effort_frame=a.effort, think_frame=think_flag)
                    print(f"  [slot] saved {r['n_saved']} tok "
                          f"({r.get('n_written', 0) / 1e6:.0f} MB, "
                          f"{r.get('timings', {}).get('save_ms', 0):.0f} ms)")

        pp = timings.get("prompt_per_second")
        tg = timings.get("predicted_per_second")
        if pp or tg:
            seg = []
            if pp:
                seg.append(f"prefill {pp:.1f} t/s ({timings.get('prompt_n') or 0} tok)")
            if tg:
                seg.append(f"decode {tg:.1f} t/s ({timings.get('predicted_n') or 0} tok)")
            print("  [speed] " + " - ".join(seg))
        print()

        if once:
            return 0

    print("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
