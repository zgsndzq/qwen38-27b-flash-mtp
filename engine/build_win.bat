@echo off
rem ===========================================================================
rem  Build the MTP-capable llama-server / llama-cli natively on Windows.
rem  Source : F:\super\llm\llamAmpere   (feature/v0.3.1-bonsai2)
rem  Log    : F:\super\_buildwin3.log
rem  This runs detached from the agent process tree on purpose - long builds
rem  get killed if they live inside a tool-call process.
rem ===========================================================================
setlocal
call F:\super\tools\vcvars.bat
if errorlevel 1 ( echo VCVARS_FAILED > F:\super\_buildwin3.log & exit /b 1 )

cd /d F:\super\llm\llamAmpere\build-win
echo BUILD_START %DATE% %TIME% > F:\super\_buildwin3.log
F:\super\tools\ninja\ninja.exe -j 6 llama-server llama-cli >> F:\super\_buildwin3.log 2>&1
set RC=%ERRORLEVEL%
echo BUILD_EXIT=%RC% %DATE% %TIME% >> F:\super\_buildwin3.log
if exist bin\llama-server.exe (echo EXE_OK >> F:\super\_buildwin3.log) else (echo EXE_MISSING >> F:\super\_buildwin3.log)
echo DONE >> F:\super\_buildwin3.log
endlocal
