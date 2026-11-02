#!/usr/bin/env bash
# Everything is in server.json; pass a name to use server.<name>.json instead.
#   ./llmserver.sh            -> server.json
#   ./llmserver.sh bonsai     -> server.bonsai.json
set -e
cd "$(dirname "$(readlink -f "$0")")"
exec python3 run_server.py "$@"
