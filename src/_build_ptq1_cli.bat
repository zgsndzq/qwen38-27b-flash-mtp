@echo off
setlocal
call F:\super\tools\vcvars.bat
cd /d F:\super\llm\forks\llamAmpere\build-ptq1
F:\super\tools\ninja\ninja.exe -j 6 llama-cli
echo CLI_BUILD_EXIT=%ERRORLEVEL%
