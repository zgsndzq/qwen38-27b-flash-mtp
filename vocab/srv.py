"""srv.py -- llama-server lifecycle + HTTP helpers for the vocab-shortlist pipeline.

Rules inherited from the project runbook (each one cost real time):

  * Wait on ``/health == 200``, NOT on a TCP connect. llama-server binds the port
    while it is still loading the weights and answers 503 until the model is up,
    so a TCP probe declares success far too early.
  * Launch the bundle ``.bat`` through cmd, so the launcher owns every server
    flag. This module never restates flags (same design rule as tools/ab_bench.py).
  * NEVER kill a server we did not start. If a healthy server already sits on the
    port we attach to it and leave it alone on exit. Killing someone else's
    session is the one unforgivable bug in this file.
  * Request bodies are UTF-8 *bytes*. A ``str`` body gets encoded with the local
    code page by some clients and silently turns Chinese into '?'.

Usage:
    with Server(port=18080) as srv:          # attach if healthy, else launch
        ids = srv.tokenize("def f(x): return x")
        r   = srv.chat([{"role": "user", "content": "..."}], max_tokens=256)
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
DEFAULT_BAT = BUNDLE / "llmserver.bat"
DEFAULT_PORT = 18080


def gpu_used() -> int:
    """MiB currently used on GPU 0, or -1 when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return -1


def kill_llama_servers() -> None:
    """Kill llama-server.exe only. Never touches any other process."""
    subprocess.run(["taskkill", "/F", "/T", "/IM", "llama-server.exe"],
                   capture_output=True, text=True)
    time.sleep(2)


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str, payload: dict, timeout: float):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class ServerError(RuntimeError):
    pass


class Server:
    """A llama-server we either attached to or launched ourselves."""

    def __init__(self, port: int = DEFAULT_PORT, bat=None, attach_only: bool = False,
                 log: Path | None = None, load_timeout: int = 600):
        self.port = port
        self.bat = Path(bat) if bat else DEFAULT_BAT
        self.attach_only = attach_only
        self.load_timeout = load_timeout
        self.log = Path(log) if log else (BUNDLE / "vocab" / "out" / "server.log")
        self.proc: subprocess.Popen | None = None
        self.owned = False          # True only when *we* spawned it
        self.attach_why = ""

    # ---------------------------------------------------------------- lifecycle
    @property
    def url(self) -> str:
        return "http://127.0.0.1:%d" % self.port

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def ensure(self) -> "Server":
        """Attach to a healthy server if there is one, else launch the .bat."""
        if self.health():
            self.attach_why = "attached to an already running server (left untouched)"
            return self
        if self.attach_only:
            raise ServerError("no healthy server on %s and --attach-only was given" % self.url)
        self._launch()
        return self

    def _launch(self) -> None:
        if not self.bat.exists():
            raise ServerError("launcher not found: %s" % self.bat)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        before = gpu_used()
        with open(self.log, "wb") as lf:
            self.proc = subprocess.Popen(
                ["cmd", "/c", str(self.bat)], stdout=lf, stderr=subprocess.STDOUT,
                cwd=str(self.bat.parent),
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        self.owned = True
        t0 = time.time()
        while time.time() - t0 < self.load_timeout:
            if self.proc.poll() is not None:
                raise ServerError("launcher exited early rc=%s (see %s)"
                                  % (self.proc.returncode, self.log))
            if self.health():
                self.attach_why = "launched %s, healthy in %.1fs (vram %s -> %s MiB)" % (
                    self.bat.name, time.time() - t0, before, gpu_used())
                return
            time.sleep(1)
        raise ServerError("timeout after %ss (see %s)" % (self.load_timeout, self.log))

    def cleanup(self) -> None:
        """Kill the tree only if we launched it. Attached servers are left running."""
        if not self.owned or self.proc is None:
            return
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                       capture_output=True, text=True)
        try:
            self.proc.wait(timeout=20)
        except Exception:
            pass
        time.sleep(3)

    def __enter__(self) -> "Server":
        return self.ensure()

    def __exit__(self, *exc) -> None:
        self.cleanup()

    # -------------------------------------------------------------------- HTTP
    def props(self) -> dict:
        return _get(self.url + "/props", timeout=20)

    def tokenize(self, text: str, add_special: bool = False,
                 parse_special: bool = True) -> list[int]:
        """Authoritative tokenization, straight from the loaded model's tokenizer.

        Deliberately NOT chunked: splitting the text would break BPE merges across
        the chunk boundary and invent tokens that the model would never emit.
        """
        if not text:
            return []
        out = _post(self.url + "/tokenize",
                    {"content": text, "add_special": add_special,
                     "parse_special": parse_special}, timeout=1800)
        return [int(t) for t in out.get("tokens", [])]

    def chat(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.6,
             top_p: float = 0.95, top_k: int = 20, seed: int | None = None,
             timeout: float = 3600, stream: bool = False, thinking: bool | None = None) -> dict:
        payload = {
            "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "top_k": top_k, "cache_prompt": False, "stream": stream,
        }
        if seed is not None:
            payload["seed"] = seed
        # The launcher pins -rea off, but the fork lets a client override the
        # thinking mode per request. That is the only way to obtain a thinking
        # block without restarting the server someone else is serving from.
        if thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        return _post(self.url + "/v1/chat/completions", payload, timeout=timeout)


def pick_reasoning(msg: dict) -> tuple[str, str]:
    """Split a chat message into (reasoning, answer).

    Different builds expose the thinking block under different keys, and some
    inline it in ``content``. Handle all of them so the corpus keeps whatever the
    model actually produced.
    """
    reasoning = ""
    for key in ("reasoning_content", "reasoning", "thinking"):
        v = msg.get(key)
        if isinstance(v, str) and v.strip():
            reasoning = v
            break
    content = msg.get("content") or ""
    if not reasoning:
        # some templates inline  thinking...<｜end▁of▁thinking｜> inside content
        for open_tag, close_tag in ((" thinking", "<｜end▁of▁thinking｜>"), ("<thinking>", "</thinking>")):
            i, j = content.find(open_tag), content.find(close_tag)
            if i != -1 and j > i:
                reasoning = content[i + len(open_tag):j]
                content = content[:i] + content[j + len(close_tag):]
                break
    return reasoning.strip(), content.strip()
