@echo off
rem Ternary-Bonsai-2-9B (Qwen3.5-9B backbone, PQ2_0 + MTP head) on port 18080.
rem Same profile shape as llmserver.bat (the 27B): ngram lookup + adaptive MTP,
rem q4_0 KV, 128K context. This file must stay ASCII + CRLF.
setlocal
set "HERE=%~dp0"
set "SERVER="
if exist "%~dp0build\src\bin\llama-server.exe" set "SERVER=%~dp0build\src\bin"
if not defined SERVER if exist "%~dp0build\bin\llama-server.exe" set "SERVER=%~dp0build\bin"
if not defined SERVER (
  echo llama-server.exe not found under build - build it first.
  exit /b 1
)
if exist "%~dp0build\cuda\bin" (
  set "PATH=%~dp0build\cuda\bin;%PATH%"
  set "CUDA_PATH=%~dp0build\cuda"
)
set "PATH=%SERVER%;%~dp0build\bin;%PATH%"

set "MODEL=Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf"
rem 2.49 GB of weights + q4_0 KV: the whole 128K window stays in VRAM.
set "CTX=131072"
set "CTK=q4_0"
set "CTV=q4_0"

rem draft width ceiling. the 27B runs 4; this target measured n-max 2 -> 112 t/s
rem and n-max 4 -> 96-105, so 2 is the ceiling here.
set "DRAFT_N=2"

rem ngram lookup drafter, same knobs as the 27B.
set "NMATCH=24"
set "NGMIN=24"
set "NGMAX=24"
set "LOOKUP_N=%NGMAX%"
rem BENCH=1 drops the ngram lookup and keeps the MTP head, for repeatable timing.
set "BENCH=0"
set "SPECTYPE=ngram-mod,draft-mtp-adaptive"
if "%BENCH%"=="1" set "SPECTYPE=draft-mtp-adaptive"

set "PORT=18080"
if not exist "%~dp0logs" mkdir "%~dp0logs"
if not exist "%~dp0data\slots" mkdir "%~dp0data\slots"
if not exist "%~dp0weights\%MODEL%" (
  echo model not found: "%~dp0weights\%MODEL%"
  exit /b 1
)
title Ternary-Bonsai-2-9B :%PORT%
cd /d "%~dp0"
taskkill /F /IM llama-server.exe >nul 2>nul
ping -n 3 127.0.0.1 >nul
rem no mmproj was downloaded for this model, so there is no vision path.
"%SERVER%\llama-server.exe" -m "%~dp0weights\%MODEL%" ^
  --host 127.0.0.1 --port %PORT% --parallel 1 -rea off --no-ui ^
  --slot-save-path "%~dp0data\slots" ^
  -c %CTX% -b 4096 -t 8 -ngl 99 -fa on -ctk %CTK% -ctv %CTV% ^
  --spec-type %SPECTYPE% --spec-draft-n-max %DRAFT_N% ^
  --spec-lookup-n-max %LOOKUP_N% --spec-ngram-mod-n-min %NGMIN% ^
  --spec-ngram-mod-n-max %NGMAX% ^
  --spec-ngram-mod-n-match %NMATCH% ^
  --features-file features.ini --features-section bonsai ^
  --alias "Ternary-Bonsai-2-9B" ^
  --log-file "%~dp0logs\bonsai_server.log"
endlocal
