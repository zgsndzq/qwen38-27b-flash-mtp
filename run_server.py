#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Start llama-server from server.json. Shared by llmserver.bat and llmserver.sh.

All the numbers live in server.json; this file only turns them into argv.
GGML_CUDA_FA_CHUNK is always set to cells -- the two must stay equal or deep
prefill leaves the fast path.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIN = os.name == "nt"
BIN = "llama-server.exe" if WIN else "llama-server"


def load_config():
    with open(HERE / "server.json", encoding="utf-8-sig") as f:
        return json.load(f)


def find_server():
    """Our own build first, then the release zip layout. None if neither exists."""
    for rel in ("build/src/bin", "build/bin"):
        if (HERE / rel / BIN).exists():
            return HERE / rel
    return None


def prepare_env(server_dir):
    """Libraries sit next to the binary; on Windows the CUDA dlls ship in build/cuda."""
    path = os.environ.get("PATH" if WIN else "LD_LIBRARY_PATH", "")
    parts = [str(server_dir), str(HERE / "build" / "bin")]
    if WIN and (HERE / "build" / "cuda" / "bin").is_dir():
        parts.insert(0, str(HERE / "build" / "cuda" / "bin"))
        os.environ["CUDA_PATH"] = str(HERE / "build" / "cuda")
    os.environ["PATH" if WIN else "LD_LIBRARY_PATH"] = \
        (os.pathsep).join(parts + ([path] if path else []))


def kill_stale():
    """A leftover instance holds the port and its VRAM."""
    cmd = ["taskkill", "/F", "/IM", BIN] if WIN else ["pkill", "-f", "llama-server"]
    subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)


def build_args(c, server_dir):
    """The command line, straight from server.json."""
    args = [
        str(server_dir / BIN),
        "-m", str(HERE / "weights" / c["model"]),
        "--host", c["host"], "--port", str(c["port"]),
        "--parallel", str(c["parallel"]), "--no-ui",
        "--slot-save-path", str(HERE / "data" / "slots"),
        "-c", str(c["ctx"]), "-b", str(c["batch"]), "-t", str(c["threads"]),
        "-ngl", str(c["ngl"]), "-fa", "on" if c["flash_attn"] else "off",
        "-ctk", c["ctk"], "-ctv", c["ctv"],
        "--kv-vram-cells", str(c["cells"]),
        "-rea", "on" if c["thinking"] else "off",
        "--reasoning-effort", c["effort"],
        "--reasoning-budget", str(c["reasoning_budget"]),
        "--reasoning-budget-message", c["reasoning_budget_message"],
        "--spec-type", spec_type(c),
        "--spec-draft-n-max", str(c["draft_n"]),
        "--spec-lookup-n-max", str(c["ngram_n_max"]),
        "--spec-ngram-mod-n-min", str(c["ngram_n_min"]),
        "--spec-ngram-mod-n-max", str(c["ngram_n_max"]),
        "--spec-ngram-mod-n-match", str(c["ngram_n_match"]),
        "--features-file", str(HERE / c["features_file"]),
        "--features-section", c["features_section"],
        "--alias", c["alias"],
    ]
    mmproj = HERE / c["mmproj"]
    if c["vision"] and mmproj.exists():
        args += ["--mmproj", str(mmproj), "--image-min-tokens", str(c["image_min_tokens"])]
    return args


def spec_type(c):
    """bench drops the ngram drafter: it warms up, so timing needs it off."""
    return "draft-mtp-adaptive" if c["bench"] else "ngram-mod,draft-mtp-adaptive"


def main():
    if not (HERE / "server.json").exists():
        print("[error] server.json not found next to this script", file=sys.stderr)
        return 1
    c = load_config()
    server_dir = find_server()
    if server_dir is None:
        print("[error] llama-server not found.\n"
              "        prebuilt : extract the release zip into this folder\n"
              "        from src : build code into build/src/bin first", file=sys.stderr)
        return 1
    try:
        args = build_args(c, server_dir)
    except KeyError as e:
        print(f"[error] server.json: missing {e}", file=sys.stderr)
        return 1
    prepare_env(server_dir)
    os.environ["GGML_CUDA_FA_CHUNK"] = str(c["cells"])
    (HERE / "data" / "slots").mkdir(parents=True, exist_ok=True)
    if c.get("kill_stale", True):
        kill_stale()
    return subprocess.call(args)


if __name__ == "__main__":
    sys.exit(main())
