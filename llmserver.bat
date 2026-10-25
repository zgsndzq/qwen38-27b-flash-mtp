@echo off
cd /d "%~dp0"

rem Everything is in server.json; this only starts it.
rem PY: full path if 'python' is not on PATH
set "PY=python"

title Qwen3.8-27B-Flash-MTP
"%PY%" "%~dp0run_server.py"
pause
