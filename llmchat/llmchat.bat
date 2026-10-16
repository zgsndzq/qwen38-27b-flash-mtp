@echo off
cd /d %~dp0

rem this launcher only talks to an already running llama-server (start the package's llmserver.bat first)
rem PY: full path if 'python' is not on PATH
set "PY=python"
rem THINK: on / off / server -- server = keep the server's -rea
set "THINK=off"
rem EFFORT: xhigh / medium / low -- leave EMPTY to send nothing and let the chat
rem template use its own default. medium = the model mid-depth reasoning mode; with
rem the server budget no longer capped it gets to finish, which is what the old 2048
rem cap was cutting off. xhigh reasons deeper, but on a ternary-quantised model a
rem longer chain also has more room to drift, so medium is the chosen mode.
set "EFFORT=medium"
set "EFFARG="
if not "%EFFORT%"=="" set "EFFARG=--effort %EFFORT%"
rem MEM: memory switch -- 0 = none / 1 = text (mem\*.md) / 2 = vision image / 3 = mixed (text+image)
rem      same as MEM_MODE in chat.py; this one wins, so there is no need to edit the .py
rem      mode 2 and 3 need the server started with --mmproj (set VISION=1 in the package's llmserver.bat)
rem      memory is injected on the FIRST turn only; type clear to re-inject
set "LLAMA_CHAT_MEM=0"

set "SYS=0"
set "SYS_FILE=sys-mtp.txt"
set "SYSARG="
if "%SYS%"=="0" set "SYSARG=--no-sys"
if "%SYS%"=="1" if exist "%~dp0%SYS_FILE%" set "SYSARG=--sys-file "%~dp0%SYS_FILE%""

title llmchat [Qwen3.8-27B-Flash-MTP-Qwen3.8-27B-Flash-MTP] :18080
"%PY%" "%~dp0chat.py" --base-url http://127.0.0.1:18080 --temp 0.6 --think %THINK% %EFFARG% %SYSARG% %*
pause
