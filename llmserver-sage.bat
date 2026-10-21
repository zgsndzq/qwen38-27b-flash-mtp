@echo off
setlocal
cd /d "%~dp0"

rem ===========================================================================
rem  llama.cpp-qwen38-flash runtime launcher : Sage A1.5 (InternScience)
rem
rem  Sage-A1.5-Q4_K_M.gguf : arch qwen35moe, 41 layers, hidden 2048, 16 heads /
rem  2 KV heads (GQA 8), 256 experts with 8 active (2.6B active), one nextn
rem  layer, so it carries an MTP head like the 27B does. 262144 context,
rem  Q4_K_M, 20.22 GB.
rem
rem  Our switches live in features.ini under [sage]. This file must stay
rem  ASCII + CRLF - cmd breaks on anything else.
rem ===========================================================================

set "SERVER="
if exist "%~dp0build\src\bin\llama-server.exe" set "SERVER=%~dp0build\src\bin"
if not defined SERVER if exist "%~dp0build\bin\llama-server.exe" set "SERVER=%~dp0build\bin"
if not defined SERVER (
  echo [error] llama-server.exe not found.
  echo         prebuilt : extract the release zip into this folder
  echo         from src : copy code\build.local.example.bat code\build.local.bat
  echo                    then run code\_build_q8.bat
  pause
  exit /b 1
)

if exist "%~dp0build\cuda\bin" (
  set "PATH=%~dp0build\cuda\bin;%PATH%"
  set "CUDA_PATH=%~dp0build\cuda"
)
set "PATH=%SERVER%;%~dp0build\bin;%PATH%"

rem GPU layers. The weights are 20.22 GB and the card is 8 GB, so -ngl 99 is an
rem instant OOM (it dies allocating the 248 MiB MTP draft context). Only 41
rem layers exist, so the ceiling is low anyway. Start at 14 and raise it if
rem nvidia-smi shows room; every layer left on the CPU is read over PCIe.
rem MoE note: 256 experts with 8 active, so the expert tensors a decode step
rem touches are only the active ones -- leaving some layers on the CPU costs
rem less here than it would on a dense model, but the shared/active-expert
rem matmuls are what actually matter for tg.
set "NGPU=14"
set "MODEL=sage-a15\Sage-A1.5-Q4_K_M.gguf"
rem context. KV is about 9 KB/token (16 attn layers x 2 KV heads x 256,
rem q4_0, K and V), so 32768 is about 290 MiB and fits next to a partial model;
rem 131072 would be about 1.1 GB and does not. Raise this only after checking
rem nvidia-smi against NGPU.
set "CTX=32768"
rem KV precision. cells kept in VRAM = features.ini kv-vram-cells (must be <= CTX).
rem q4_0 on 16 attention layers with 2 KV heads of 256: about 9 KB per token.
set "CTK=q4_0"
set "CTV=q4_0"
set "REBUDGET=-1"
set "DRAFT_N=4"
set "NMATCH=24"
set "NGMIN=24"
set "NGMAX=24"
set "LOOKUP_N=%NGMAX%"
rem BENCH=1 keeps the MTP head and drops the ngram lookup drafter.
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
title Sage-A1.5-Q4_K_M :%PORT%

rem a stale instance keeps :18080 and its VRAM, so kill it before binding again.
taskkill /F /IM llama-server.exe >nul 2>nul
ping -n 3 127.0.0.1 >nul

rem vision projector: this repo shipped mmproj next to the weights.
set "VISION=1"
set "MMPROJ=%~dp0weights\sage-a15\Sage-A1.5-mmproj.gguf"
set "VISARG="
if "%VISION%"=="1" if exist "%MMPROJ%" set "VISARG=--mmproj %MMPROJ% --image-min-tokens 1024"

"%SERVER%\llama-server.exe" -m "%~dp0weights\%MODEL%" ^
  --host 127.0.0.1 --port %PORT% --parallel 1 -rea off --no-ui ^
  --slot-save-path "%~dp0data\slots" ^
  -c %CTX% -b 2048 -t 8 -ngl %NGPU% -fa on -ctk %CTK% -ctv %CTV% ^
  --reasoning-budget %REBUDGET% ^
  --spec-type %SPECTYPE% --spec-draft-n-max %DRAFT_N% ^
  --spec-lookup-n-max %LOOKUP_N% --spec-ngram-mod-n-min %NGMIN% ^
  --spec-ngram-mod-n-max %NGMAX% ^
  --spec-ngram-mod-n-match %NMATCH% ^
  --features-file features.ini --features-section sage ^
  --alias "Sage-A1.5-Q4_K_M" ^
  %VISARG% ^
  --log-file "%~dp0logs\sage_server.log"
endlocal
