#!/usr/bin/env bash
# Config file to load. A file named on the command line wins.
#   server.json         -> llama-server
#   server.bonsai.json  -> bonsai
CONFIG="server.json"
set -e
cd "$(dirname "$(readlink -f "$0")")"
exec python3 run_server.py "${1:-$CONFIG}"
