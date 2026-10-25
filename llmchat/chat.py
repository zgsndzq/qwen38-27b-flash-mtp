#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small chat client for an already running llama-server."""

import argparse
import json
import sys
from pathlib import Path

import session as sess
from log import log, setup as log_setup
from client import ChatRequest, Client
from ui import HAS_TUI, meta, new_session, read_user, stream_printer

# CLI options as (config key, flags, add_argument kwargs); config.json sets the defaults.
OPTIONS = (
    ("base_url", ("--base-url",),
     {"default": None, "help": "server address (required)"}),
    ("sys", ("--sys",),
     {"default": None, "help": "system prompt; without it none is sent"}),
    ("sys_file", ("--sys-file",),
     {"default": None, "help": "read the system prompt from a UTF-8 file (wins over --sys)"}),
    ("temp", ("--temp",),
     {"type": float, "default": None, "help": "sampling temperature (omit = server --temp)"}),
    ("think", ("--think",),
     {"choices": ("on", "off", "server"), "default": "on",
      "help": "thinking switch (server = keep the server's -rea)"}),
    ("effort", ("--effort",),
     {"choices": ("xhigh", "medium", "low"), "default": None,
      "help": "thinking effort (omit = server default)"}),
    ("timeout", ("--timeout",),
     {"type": float, "default": None, "help": "seconds without data (required)"}),
    ("slot_min", ("--slot-min",),
     {"type": int, "default": None, "help": "no slot file below this many tokens"}),
    ("slot_grow", ("--slot-grow",),
     {"type": int, "default": None, "help": "re-save the slot after this much growth"}),
    ("max_tokens", ("--max-tokens",),
     {"type": int, "default": None, "help": "per-turn answer limit (omit = run to EOS)"}),
    ("hide_thinking", ("--hide-thinking",),
     {"action": "store_true", "help": "do not print the thinking block"}),
    ("keep_thinking", ("--no-keep-thinking",),
     {"dest": "keep_thinking", "action": "store_false", "default": True,
      "help": "drop thinking from history (saves context, breaks the slot cache)"}),
    ("no_tui", ("--no-tui",),
     {"action": "store_true", "help": "single-line input + /paste"}),
    ("once", ("--once",),
     {"metavar": "QUESTION", "default": None,
      "help": "non-interactive: ask this question and exit"}),
    ("save", ("--save",),
     {"metavar": "FILE", "default": None,
      "help": "write the final answer to a UTF-8 file (with --once)"}),
    ("log", ("--log",),
     {"metavar": "FILE", "default": None,
      "help": "append everything printed to a file as well"}),
    ("quiet", ("--quiet",),
     {"action": "store_true", "help": "no console output (pair with --log)"}),
)


REQUIRED = ("base_url", "timeout", "slot_min", "slot_grow")


def load_config():
    """config.json next to this file; a missing or broken one just means defaults."""
    try:
        with open(Path(__file__).resolve().parent / "config.json", encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def build_parser(conf=None):
    conf = conf or {}
    ap = argparse.ArgumentParser(
        prog="chat.py",
        description="chat client for an already running llama-server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python chat.py --temp 0.6 --think on --effort low\n"
               "  python chat.py --base-url http://127.0.0.1:18081 --think server\n")
    for key, flags, kwargs in OPTIONS:
        kw = dict(kwargs)
        kw["default"] = conf.get(key, kw.get("default"))
        ap.add_argument(*flags, **kw)
    return ap


def load_sys(a):
    """(system prompt, exit code); None means send no system message."""
    if not a.sys_file:
        return a.sys, 0
    try:
        with open(a.sys_file, encoding="utf-8-sig") as f:   # strips Notepad's BOM
            text = f.read().strip()
    except OSError as e:
        log(f"[ERROR] cannot read --sys-file {a.sys_file}: {e}")
        return None, 2
    if not text:
        log(f"[ERROR] --sys-file is empty: {a.sys_file}")
        return None, 2
    return text, 0


def seed_messages(sys_prompt):
    """Message list for a fresh context: a system message only if configured."""
    return [] if sys_prompt is None else [{"role": "system", "content": sys_prompt}]


def parse_think(v):
    """'on'/'off' -> True/False; 'server' -> None (keep the server's -rea)."""
    return None if v == "server" else v == "on"


class Chat:
    """History, slot bookkeeping, and one turn at a time."""

    def __init__(self, a, client, sys_prompt, think):
        self.a = a
        self.client = client
        self.sys_prompt = sys_prompt
        self.think = think
        self.slot_min = a.slot_min
        self.slot_grow = a.slot_grow
        self.messages = seed_messages(sys_prompt)
        self.saved = sess.session_load() or {}
        self.slot_seen = int(self.saved.get("slot_tokens") or 0)
        self.need_restore = bool(self.saved.get("messages"))
        self.psession = None

    # ---- resume ---------------------------------------------------------
    def frame_diff(self):
        """Frames that changed since the archive was written; empty = slot fits."""
        old = self.saved
        out = []
        if old.get("sys_frame") != self.sys_prompt:
            out.append("system")
        if self.think is not None and old.get("think_frame") is not None \
                and old.get("think_frame") != self.think:
            out.append("thinking")
        if self.a.effort is not None and old.get("effort_frame") is not None \
                and old.get("effort_frame") != self.a.effort:
            out.append("effort")
        return out

    def with_system(self, messages):
        """Put this run's system message at the head, or drop a stale one."""
        if messages and messages[0]["role"] == "system":
            return seed_messages(self.sys_prompt) + messages[1:]
        if self.sys_prompt is None:
            return messages
        return seed_messages(self.sys_prompt) + messages

    def resume(self):
        """Load the archive and drop its slot when the frames no longer match."""
        if not self.need_restore:
            return
        self.messages = self.with_system(list(self.saved["messages"]))
        diff = self.frame_diff()
        if diff:
            self.need_restore = False
            self.client.slot("erase")
            log("(frames changed: %s) slot dropped, this turn re-computes" % ", ".join(diff))
        log("resumed %d messages / %s tok"
            % (len(self.messages), self.saved.get("tokens") or 0))

    def restore(self):
        """Read the slot file back, once, on the first turn after a resume."""
        if not self.need_restore:
            return 0
        self.need_restore = False
        if (self.saved.get("tokens") or 0) < self.slot_min or not sess.SLOT_FILE.exists():
            return 0
        r = self.client.slot("restore", sess.SLOT_NAME)
        n = int((r or {}).get("n_restored") or 0)
        if n:
            log(f"  [slot] restored {n} tok")
        return n

    # ---- one turn -------------------------------------------------------
    def turn(self, user):
        """Send one message. False means the server was unreachable."""
        sent = self.messages + [{"role": "user", "content": user}]
        restored = self.restore()
        req = ChatRequest(sent, temperature=self.a.temp, max_tokens=self.a.max_tokens,
                          think=self.think, effort=self.a.effort)
        try:
            printer = stream_printer(hide_thinking=self.a.hide_thinking)
            reasoning, content, timings = self.client.chat(req, printer)
        except ConnectionError as e:
            log(f"\nserver unavailable: {e}\n")
            return False

        self.report_hit(restored, timings)
        if printer.streamed["content"] or printer.streamed["thinking"]:
            log()                       # already streamed, just end the line
        else:
            log(f"\nmodel> {content}")
        self.messages.append({"role": "user", "content": user})
        self.messages.append(self.assistant(content, reasoning))
        if self.a.save:
            self.save_answer(content)
        self.keep_context(timings)
        self.report_speed(timings)
        log()
        return True

    def assistant(self, content, reasoning):
        """History entry; the thinking block goes back as the server sent it."""
        if not (reasoning and self.a.keep_thinking):
            return {"role": "assistant", "content": content}
        return {"role": "assistant", "content": content, "reasoning_content": reasoning}

    def report_hit(self, restored, timings):
        """prompt_n is the only proof that the restored slot was really reused."""
        pp = int(timings.get("prompt_n") or 0)
        if restored and pp > max(256, restored // 8):
            log(f"  [slot] missed: restored {restored} tok but re-computed {pp} tok")

    def save_answer(self, content):
        try:
            Path(self.a.save).write_text(content, encoding="utf-8")
            log(f"  [saved] {self.a.save}")
        except OSError as e:
            log(f"  [save failed] {e}")

    def keep_context(self, timings):
        """Archive the session, and the slot too once it grew enough."""
        ctx = (timings.get("prompt_n") or 0) + (timings.get("cache_n") or 0)
        if not ctx:
            return
        limit = self.client.n_ctx()
        share = "" if not limit else " - %d%% of %d" % (100.0 * ctx / limit, limit)
        log(f"  [ctx {ctx} tok{share}]")
        self.write_session(ctx)
        grown = ctx >= self.slot_min and (not sess.SLOT_FILE.exists()
                                         or ctx - self.slot_seen >= self.slot_grow)
        if grown:
            self.save_slot(ctx)

    def write_session(self, ctx, slot_tokens=None):
        """All three frames in one write, or the next start replays defaults."""
        sess.session_save(self.messages, ctx, slot_tokens=slot_tokens,
                          sys_frame=self.sys_prompt, effort_frame=self.a.effort,
                          think_frame=self.think)

    def save_slot(self, ctx):
        r = self.client.slot("save", sess.SLOT_NAME)
        if not (r and r.get("n_saved")):
            return
        self.slot_seen = ctx
        self.write_session(ctx, slot_tokens=ctx)
        log(f"  [slot] saved {r['n_saved']} tok ({r.get('n_written', 0) / 1e6:.0f} MB, "
            f"{r.get('timings', {}).get('save_ms', 0):.0f} ms)")

    def report_speed(self, timings):
        seg = []
        for key, nkey, label in (("prompt_per_second", "prompt_n", "prefill"),
                                 ("predicted_per_second", "predicted_n", "decode")):
            tps = timings.get(key)
            if tps:
                seg.append(f"{label} {tps:.1f} t/s ({timings.get(nkey) or 0} tok)")
        if seg:
            log("  [speed] " + " - ".join(seg))

    # ---- loop -----------------------------------------------------------
    def clear(self):
        self.messages = seed_messages(self.sys_prompt)
        sess.session_clear(self.client)
        log("context cleared\n")

    def read(self):
        if self.a.once is not None:
            log(f"you> {self.a.once}")
            return self.a.once
        try:
            return read_user(self.psession, "you> ")
        except (EOFError, KeyboardInterrupt):
            return None

    def run(self):
        while True:
            user = self.read()
            if not user:
                break
            cmd = meta(user)
            if cmd == "exit":
                break
            if cmd == "clear":
                self.clear()
                continue
            ok = self.turn(user)
            if self.a.once is not None:
                return 0 if ok else 1
        log("bye")
        return 0


def main(argv=None):
    a = build_parser(load_config()).parse_args(argv)
    log_setup(a.log, echo=not a.quiet)
    missing = [k for k in REQUIRED if getattr(a, k) is None]
    if missing:
        log("[ERROR] config.json: missing " + ", ".join(missing))
        return 2
    sys_prompt, rc = load_sys(a)
    if rc:
        return rc
    base_url = a.base_url.rstrip("/")
    sess.apply_profile(base_url)
    sess.ensure_slot_dir()

    client = Client(base_url, a.timeout)
    ok, model_id, err = client.probe()
    if not ok:
        log(f"[ERROR] no llama-server at {base_url}: {err}")
        return 2
    log(f"{model_id or 'model'} @ {base_url}")

    think = parse_think(a.think)
    chat = Chat(a, client, sys_prompt, think)
    use_tui = HAS_TUI and sys.stdin.isatty() and not a.no_tui
    chat.psession = new_session() if use_tui else None
    chat.resume()
    return chat.run()


if __name__ == "__main__":
    sys.exit(main())
