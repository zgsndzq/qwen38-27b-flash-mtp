@echo off
cd /d %~dp0

rem Settings live in config.json; flags passed on this line override them.
rem PY: full path if 'python' is not on PATH
set "PY=python"

title llmchat
"%PY%" "%~dp0chat.py" %*
pause
