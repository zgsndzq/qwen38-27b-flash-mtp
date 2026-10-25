@echo off
cd /d %~dp0

rem talks to an already running llama-server (start llmserver.bat first)
rem PY: full path if 'python' is not on PATH
set "PY=python"
rem THINK: on / off / server -- server = keep the server's -rea
set "THINK=on"
rem EFFORT: xhigh / medium / low -- empty = send nothing, use the template default
set "EFFORT=low"
set "EFFARG="
if not "%EFFORT%"=="" set "EFFARG=--effort %EFFORT%"
rem SYS: 0 = no system message, 1 = read SYS_FILE
set "SYS=0"
set "SYS_FILE=sys-mtp.txt"
set "SYSARG="
if "%SYS%"=="0" set "SYSARG=--no-sys"
if "%SYS%"=="1" if exist "%~dp0%SYS_FILE%" set "SYSARG=--sys-file "%~dp0%SYS_FILE%""

title llmchat [Qwen3.8-27B-Flash-MTP] :18080
"%PY%" "%~dp0chat.py" --base-url http://127.0.0.1:18080 --temp 0.6 --think %THINK% %EFFARG% %SYSARG% %*
pause
