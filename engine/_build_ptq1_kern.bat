@echo off
rem Incremental build of the pr-ptq1-mmv (commit1+2) kernel version into the SAME build-ptq1 dir.
rem Only mmvq.cu / quantize.cu changed -> ninja recompiles those + relinks.
setlocal
call F:\super\tools\vcvars.bat
if errorlevel 1 ( echo VCVARS_FAILED > F:\super\_ptq1_kern.log & exit /b 1 )
cd /d F:\super\llm\forks\llamAmpere\build-ptq1
echo BUILD_START %DATE% %TIME% > F:\super\_ptq1_kern.log
F:\super\tools\ninja\ninja.exe -j 6 llama-server llama-bench >> F:\super\_ptq1_kern.log 2>&1
set RC=%ERRORLEVEL%
echo BUILD_EXIT=%RC% %DATE% %TIME% >> F:\super\_ptq1_kern.log
if exist bin\llama-bench.exe (echo BENCH_OK >> F:\super\_ptq1_kern.log) else (echo BENCH_MISSING >> F:\super\_ptq1_kern.log)
echo DONE >> F:\super\_ptq1_kern.log
endlocal
