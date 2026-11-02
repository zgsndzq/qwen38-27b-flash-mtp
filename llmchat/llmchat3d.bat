@echo off
cd /d %~dp0

rem llmchat wired to llmserver3d.bat through the "3d" profile = config.3d.json:
rem port 18080, the Three.js system prompt from sys3d.txt, thinking on, and mem=3
rem on the "scene" shelf (its conventions go in on the first turn). "3d" is the
rem profile name, so anything you add here wins over that file. Settings config.3d.json
rem leaves out still come from chat.py defaults. Keep this file ASCII + CRLF.
rem Two ways to feed a picture: "/img PATH" attaches to one message; a picture in
rem mem\scene\ with a same-named tag file gets attached on its own whenever what you
rem type matches it (one per message, none when nothing fits). See mem\README.md,
rem and "/mem find QUERY" to check what a phrase would hit before you rely on it.
rem If pictures blow the slot, lower CTX in llmserver3d.bat before adding more.
rem PY: full path if 'python' is not on PATH
set "PY=python"

title llmchat 3D
"%PY%" "%~dp0chat.py" 3d %*
pause
