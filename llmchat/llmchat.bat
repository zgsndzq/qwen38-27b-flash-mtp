@echo off
cd /d "%~dp0"

rem Config file to load. A file named on the command line wins; flags after it pass through.
set "CONFIG=config.json"
rem PY: full path if 'python' is not on PATH
set "PY=python"

if "%~1"=="" (set "ARGV=%CONFIG%") else (set "ARGV=%*")

title llmchat %ARGV%
"%PY%" "%~dp0chat.py" %ARGV%
pause
