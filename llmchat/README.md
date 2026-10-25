# llmchat — standalone chat client

Talks to an already running llama-server (for example the one started by
`..\llmserver.bat`). It never loads a model itself, and it only needs the
Python standard library.

## Run it

Double-click `llmchat.bat`. Its settings are variables at the top of the file,
in the same style as `llmserver.bat`.

Command line equivalent:

```
python chat.py --temp 0.6 --think on --effort low
```

## Files

| File | Purpose |
|---|---|
| `chat.py` | entry point: CLI, session resume, main loop |
| `srvapi.py` | HTTP layer: probe, streaming chat, slot control |
| `session.py` | session archive, KV slot files, render frames |
| `ui.py` | input (TUI / paste), streaming printer, meta commands |
| `llmchat.bat` | launcher: fills in the parameters and calls chat.py. **ASCII only** |
| `sys.txt` | system prompt (UTF-8, off by default). Without a system prompt configured, no system message is sent at all — there is no built-in fallback |

## Arguments

| chat.py | bat variable | Meaning |
|---|---|---|
| `--base-url` | — | server address, default `http://127.0.0.1:18080` |
| `--temp` | — | sampling temperature. empty = send nothing, keep the server `--temp` |
| `--think` | `THINK` | `on` / `off` / `server` (server = send nothing, keep `-rea`) |
| `--effort` | `EFFORT` | `xhigh` / `medium` / `low`; omit to keep the server default |
| `--timeout` | — | seconds of **no data** from the server, default 600 |
| `--max-tokens` | — | per-turn answer limit; omit for no limit (run to EOS) |
| `--sys` / `--sys-file` | `SYS` + `SYS_FILE` | system prompt; the file wins |
| `--hide-thinking` | — | do not print thinking (the request still thinks) |
| `--no-tui` | — | disable the multiline TUI |

The window title is cosmetic: the server always serves the gguf it loaded, and
the client cannot pick a model. On start it probes `/v1/models` and prints what
the server actually has.

## In-session meta commands (intercepted locally)

| Command | Effect |
|---|---|
| `quit` / `q` / `exit` | leave |
| `clear` / `cls` | drop the context, keep the system prompt |
| `/paste` … a single `/end` line | paste multiline text when stdin is not a TTY |

## Slots and caches: the prefix must be reproducible token for token

A slot saved with `--slot-save-path` only supports a **strict extension**: the
new prompt has to extend the stored token sequence, and a single token of
difference makes the server drop the whole KV and re-compute.

The prefix is decided by three things, and they must be saved and replayed
together:

| Part | Role | Stored as |
|---|---|---|
| system text | first message if configured, otherwise none is sent | `sys_frame` |
| thinking switch | the template injects the effort sentence only when `enable_thinking` is true | `think_frame` |
| thinking effort | the literal that goes into the system block | `effort_frame` |

Rules:

1. **Send what is configured.** No `--sys` / `--sys-file` means no system
   message, with no "sensible default" invented on top.
2. **History keeps exactly what was sent**, and is never rewritten afterwards.
3. After saving a slot, the following `session_save()` must write all three
   frames again; writing only `slot_tokens` resets them to null and the next
   start replays an empty frame, forking the prefix at the system block.
4. **A changed frame invalidates the slot** — it is not a reason to force the
   old frame back on. On start the saved frames are compared with this run's: on
   a match the slot is restored, on a difference the slot is dropped and the run
   proceeds with the current settings, printing the difference.
5. **The thinking block goes back into history** (default). The template renders
   `<think>...</think>` only when an assistant message carries
   `reasoning_content`, and the server KV already holds those tokens. Dropping
   them makes the next prompt shorter than the KV, so the shared prefix breaks
   at the first thinking turn and everything after it is re-computed. Use
   `--no-keep-thinking` to trade the cache for context space.

## Timeouts

The timeout guards the **heartbeat**, not total wall time: during prompt eval
the server sends nothing, which on a long context legitimately takes minutes.
A timeout on a long prompt does not mean the server is down — raise
`--timeout` or `LLAMA_CHAT_TIMEOUT`.

## TUI

Enabled automatically when `prompt_toolkit` is installed and stdin is a
terminal: pasting a large document submits it in one block, **Esc+Enter** adds
a newline, up/down recall history. Without it the client falls back to
single-line input plus `/paste`. Force single line with `--no-tui`.

## Dependencies

- Python 3 (standard library only; `prompt_toolkit` is optional)
- A running llama-server, e.g. `..\llmserver.bat`
