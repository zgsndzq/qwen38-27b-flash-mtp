@echo off
cd /d %~dp0

rem Settings live in config.json; a first bare word picks config.<word>.json:
rem   llmchat.bat           -> config.json
rem   llmchat.bat bonsai    -> config.bonsai.json
rem Flags passed on this line override whatever the file says.
rem PY: full path if 'python' is not on PATH
set "PY=python"

title llmchat %1
"%PY%" "%~dp0chat.py" %*
pause
