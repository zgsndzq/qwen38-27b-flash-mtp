@echo off
rem Baseline (pristine-code) build of llama-server + llama-bench, detached-safe.
setlocal
call F:\super\tools\vcvars.bat
if errorlevel 1 ( echo VCVARS_FAILED > F:\super\_ptq1_base.log & exit /b 1 )
cd /d F:\super\llm\forks\llamAmpere\build-ptq1
echo BUILD_START %DATE% %TIME% > F:\super\_ptq1_base.log
F:\super\tools\ninja\ninja.exe -j 6 llama-server llama-bench >> F:\super\_ptq1_base.log 2>&1
set RC=%ERRORLEVEL%
echo BUILD_EXIT=%RC% %DATE% %TIME% >> F:\super\_ptq1_base.log
if exist bin\llama-bench.exe (echo BENCH_OK >> F:\super\_ptq1_base.log) else (echo BENCH_MISSING >> F:\super\_ptq1_base.log)
echo DONE >> F:\super\_ptq1_base.log
endlocal
