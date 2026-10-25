#!/usr/bin/env bash
# Everything is in server.json; this only starts it.
set -e
cd "$(dirname "$(readlink -f "$0")")"
exec python3 run_server.py
