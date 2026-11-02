@echo off
cd /d %~dp0

rem llmchat wired to llmserver3d.bat: same port (18080), the Three.js system
rem prompt from sys3d.txt, and thinking on (the model was trained with thinking
rem traces). These flags come before %*, so anything you add wins. Settings that
rem are not set here still come from config.json. This file must stay ASCII + CRLF.
rem PY: full path if 'python' is not on PATH
set "PY=python"

title llmchat 3D
"%PY%" "%~dp0chat.py" --sys-file "%~dp0sys3d.txt" --think on %*
pause
