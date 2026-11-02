@echo off
cd /d "%~dp0"

rem Everything is in server.json; pass a name to use server.<name>.json instead.
rem   llmserver.bat          -> server.json
rem   llmserver.bat bonsai   -> server.bonsai.json
rem PY: full path if 'python' is not on PATH
set "PY=python"

title llama-server %1
"%PY%" "%~dp0run_server.py" %*
pause
