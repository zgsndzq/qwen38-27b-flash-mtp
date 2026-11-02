#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Start llama-server from a server profile. Shared by the llmserver*.bat files.

All the numbers live in the json; this file only turns them into argv.
Profiles: server.json is the default, server.<name>.json is picked by naming it
on the command line (llmserver.bat bonsai -> server.bonsai.json). Keys marked
optional below may be left out; a profile that does not need a feature simply
does not mention it.
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
DEFAULT_CONF = "server.json"

# "debug": true in a profile turns every server trace on, writing to files
DEBUG_ALL = {
    "SPC_SRV_TIMING": 1,
    "GGML_CUDA_FA_LOG": 1,
    "GGML_CUDA_LOG_FILE": "logs/mmfa.log",
    "LLAMA_PERF_LOG": 2,
    "LLAMA_PERF_LOG_FILE": "logs/perf.csv",
}


def config_path(name=None):
    """server.json, or server.<name>.json for a named profile.

    A name that is a path (.json suffix or a directory part) is taken as-is, so
    an absolute file outside the tree still works."""
    if not name:
        return HERE / DEFAULT_CONF
    p = Path(name)
    if p.suffix == ".json" or p.parent != Path("."):
        return p if p.is_absolute() else HERE / p
    return HERE / f"server.{name}.json"


def load_config(name=None):
    with open(config_path(name), encoding="utf-8-sig") as f:
        return json.load(f)


def list_profiles():
    """Names usable on the command line, i.e. the server.*.json files here."""
    return sorted(p.name[len("server."):-len(".json")] for p in HERE.glob("server.*.json"))


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
    """The command line, straight from the profile.

    Required: model host port ctx batch threads ngl parallel flash_attn ctk ctv
              alias features_file features_section draft_n ngram_n_{match,min,max}
    Optional: cells thinking effort reasoning_budget reasoning_budget_message
              vision mmproj image_min_tokens log_file bench kill_stale
              debug   true = every server trace on, into logs/mmfa.log and
                      logs/perf.csv; a dict instead overrides single keys.
                      Omitted = all off, nothing is written.
    """
    args = [
        str(server_dir / BIN),
        "-m", str(HERE / "weights" / c["model"]),
        "--host", c["host"], "--port", str(c["port"]),
        "--parallel", str(c["parallel"]), "--no-ui",
        "--slot-save-path", str(HERE / "data" / "slots"),
        "-c", str(c["ctx"]), "-b", str(c["batch"]), "-t", str(c["threads"]),
        "-ngl", str(c["ngl"]), "-fa", "on" if c["flash_attn"] else "off",
        "-ctk", c["ctk"], "-ctv", c["ctv"],
        "--spec-type", spec_type(c),
        "--spec-draft-sampling", "probabilistic",
        "--spec-draft-n-max", str(c["draft_n"]),
        "--spec-lookup-n-max", str(c["ngram_n_max"]),
        "--spec-ngram-mod-n-min", str(c["ngram_n_min"]),
        "--spec-ngram-mod-n-max", str(c["ngram_n_max"]),
        "--spec-ngram-mod-n-match", str(c["ngram_n_match"]),
        "--features-file", str(HERE / c["features_file"]),
        "--features-section", c["features_section"],
        "--alias", c["alias"],
    ]
    # A profile only mentions what it uses: bonsai has no tiered KV (cells) and no
    # reasoning knobs, and leaving them out is the point of the split.
    if c.get("cells"):
        args += ["--kv-vram-cells", str(c["cells"])]
    if "thinking" in c:
        args += ["-rea", "on" if c["thinking"] else "off"]
    if c.get("effort"):
        args += ["--reasoning-effort", c["effort"]]
    if c.get("reasoning_budget"):
        args += ["--reasoning-budget", str(c["reasoning_budget"])]
    if c.get("reasoning_budget_message"):
        args += ["--reasoning-budget-message", c["reasoning_budget_message"]]
    if c.get("log_file"):
        args += ["--log-file", str(HERE / c["log_file"])]
    mmproj = HERE / c["mmproj"] if c.get("mmproj") else None
    if c.get("vision") and mmproj is not None and mmproj.exists():
        args += ["--mmproj", str(mmproj), "--image-min-tokens", str(c["image_min_tokens"])]
    return args


def spec_type(c):
    """bench drops the ngram drafter: it warms up, so timing needs it off."""
    return "draft-mtp-adaptive" if c.get("bench") else "ngram-mod,draft-mtp-adaptive"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else None
    path = config_path(name)
    if not path.exists():
        print(f"[error] config not found: {path}", file=sys.stderr)
        alts = list_profiles()
        if alts:
            print("        profiles here: " + ", ".join(alts), file=sys.stderr)
            print(f"        usage: llmserver.bat [{'|'.join(alts)}]", file=sys.stderr)
        return 1
    c = load_config(name)
    server_dir = find_server()
    if server_dir is None:
        print("[error] llama-server not found.\n"
              "        prebuilt : extract the release zip into this folder\n"
              "        from src : build code into build/src/bin first", file=sys.stderr)
        return 1
    try:
        args = build_args(c, server_dir)
    except KeyError as e:
        print(f"[error] {path.name}: missing {e}", file=sys.stderr)
        return 1
    print(f"[profile] {path.name}  model={c['model']}  port={c['port']}")
    model = HERE / "weights" / c["model"]
    if not model.exists():
        print(f"[error] model not found: {model}", file=sys.stderr)
        return 1
    prepare_env(server_dir)
    if c.get("cells"):
        os.environ["GGML_CUDA_FA_CHUNK"] = str(c["cells"])
    # "debug": true = every trace; a dict overrides single keys; absent = all off
    dbg = c.get("debug")
    if dbg is True:
        dbg = DEBUG_ALL
    for k, v in (dbg if isinstance(dbg, dict) else {}).items():
        if isinstance(v, str) and v.startswith("logs/"):
            v = str(HERE / v)
        os.environ[str(k)] = str(v)
    if isinstance(dbg, dict) and dbg:
        (HERE / "logs").mkdir(parents=True, exist_ok=True)
        print("[debug] " + "  ".join(f"{k}={os.environ[str(k)]}" for k in dbg), flush=True)
    (HERE / "data" / "slots").mkdir(parents=True, exist_ok=True)
    if c.get("log_file"):
        (HERE / "logs").mkdir(parents=True, exist_ok=True)
    if c.get("kill_stale", True):
        kill_stale()
    return subprocess.call(args)


if __name__ == "__main__":
    sys.exit(main())
