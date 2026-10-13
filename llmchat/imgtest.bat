@echo off
cd /d %~dp0

rem imgtest -- minimal multi-turn client that puts ONE image in the FIRST turn only.
rem It answers one question: does the server reuse the prefix from turn 2 on?
rem Needs a running llama-server on :18080 with --mmproj (set VISION=1 in ..\llmserver.bat).
rem Usage: imgtest.bat                                   (default: first image in mem\)
rem        imgtest.bat --img mem\chess1.png --think on
rem        imgtest.bat --url http://127.0.0.1:18081

set "PY=python"
where python >nul 2>nul || set "PY=C:\Users\bestd\.workbuddy\binaries\python\versions\3.13.12\python.exe"

title imgtest [1st-turn image only] :18080
"%PY%" "%~dp0imgtest.py" %*
pause
