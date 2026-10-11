@echo off
setlocal
cd /d "%~dp0"
title Image JEV   UI 18091 : model 18080

rem --------------------------------------------------------------- python
set "PY="
where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY if exist "%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe" set "PY=%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe"
if not defined PY (
  echo [ERROR] python not found. Install python, or set PY=... in this file.
  pause
  exit /b 1
)

rem ------------------------------------------------ command line passthrough
rem   run.bat --cli --image mem\chess.png --question "..." --options "A,B,C"
rem   run.bat --cli --batch samples\chess.jsonl --out out\preds.jsonl
rem   run.bat --cli --score out\preds.jsonl
if "%~1"=="--cli" goto cli

rem --------------------------------------------- the model server, port 18080
rem Deliberately NOT started here: run ..\llmserver.bat in another window first.
curl -s -m 3 http://127.0.0.1:18080/health >nul 2>nul
if errorlevel 1 echo [WARN] 18080 is not answering - start ..\llmserver.bat, the page shows "offline" until then.

if not exist "%~dp0out" mkdir "%~dp0out"

echo.
echo   UI      http://127.0.0.1:18091/
echo   model   http://127.0.0.1:18080/   start ..\llmserver.bat yourself
echo   stop    close this window, or press Ctrl+C
echo   CLI     run.bat --cli --image mem\chess.png --question "..." --options "A,B,C"
echo.

rem set JEV_NO_BROWSER=1 to skip opening the browser (headless / scripted runs)
if not "%JEV_NO_BROWSER%"=="1" start "" "http://127.0.0.1:18091/"
"%PY%" "%~dp0bridge.py"

pause
exit /b 0

:cli
shift
"%PY%" "%~dp0image_jev.py" %*
pause
exit /b 0
