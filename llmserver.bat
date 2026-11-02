@echo off
cd /d "%~dp0"

rem Config file to load. A file named on the command line wins.
set "CONFIG=server-q40.json"
rem PY: full path if 'python' is not on PATH
set "PY=python"

if not "%~1"=="" set "CONFIG=%~1"

title llama-server %CONFIG%
"%PY%" "%~dp0run_server.py" %CONFIG%
pause
