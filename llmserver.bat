@echo off
setlocal
cd /d "%~dp0"

rem ===========================================================================
rem  Qwen3.8-27B-Flash-MTP  :18080   --  agent default profile
rem
rem    thinking ON, effort=low, cap 2048   |  q3_0 K/V, 88K cells, CTX 128K
rem    slot save/restore across restarts   |  MTP + ngram draft, width <= 4
rem
rem  Binary: build\src\bin (own build), else build\bin (release zip).
rem  Our switches live in features.ini [qwen]; numbers live here.
rem  ASCII + CRLF only - cmd mis-parses anything else.
rem ===========================================================================

set "SERVER="
if exist "%~dp0build\src\bin\llama-server.exe" set "SERVER=%~dp0build\src\bin"
if not defined SERVER if exist "%~dp0build\bin\llama-server.exe" set "SERVER=%~dp0build\bin"
if not defined SERVER (
  echo [error] llama-server.exe not found.
  echo         prebuilt : extract the release zip into this folder
  echo         from src : run code\_build_q8.bat first
  pause
  exit /b 1
)

rem CUDA runtime dlls: shipped under build\cuda in the zip, on PATH otherwise
if exist "%~dp0build\cuda\bin" (
  set "PATH=%~dp0build\cuda\bin;%PATH%"
  set "CUDA_PATH=%~dp0build\cuda"
)
set "PATH=%SERVER%;%~dp0build\bin;%PATH%"

set "MODEL=Qwen3.8-27B-Flash-MTP.gguf"
set "CTX=131072"

rem ---- K/V and VRAM tier ----------------------------------------------------
rem  CELLS comes from CTK/CTV and GGML_CUDA_FA_CHUNK must stay equal to it,
rem  otherwise deep prefill leaves the fast path and drops ~6x.
rem  q4_0 = 18 KB/cell -> 65536 (1152 MiB)  |  q3_0 = 14 KB/cell -> 90112 (1232 MiB)
rem  same type on both sides: a mixed pair has no FA kernel and copies to f16
rem  cells past CELLS stay in pinned host RAM; CTX is unaffected
rem  6 GB card: drop to q4_0 / 65536 or lower
set "CTK=q3_0"
set "CTV=q3_0"
set "CELLS=90112"
if "%CTK%%CTV%"=="q4_0q4_0" set "CELLS=65536"
if "%CTK%%CTV%"=="q3_0q3_0" set "CELLS=90112"
set "GGML_CUDA_FA_CHUNK=%CELLS%"

rem ---- thinking -------------------------------------------------------------
rem  EFFORT: template default, overridable per request (minimal|low|medium|high|xhigh)
rem  REBUDGET: the only real cap -- -1 unlimited, 0 stop now, N>0 cap
rem  REBMSG: tokenised verbatim before the end-of-thinking tag; plain text only
rem  max_tokens is the client's budget (llmchat), never set here
set "EFFORT=low"
set "REBUDGET=2048"
set "REBMSG= Budget exceeded, answer directly."

rem ---- draft ----------------------------------------------------------------
rem  DRAFT_N: adaptive width ceiling (2 = fastest, 4 = best quality)
rem  NMATCH: preceding tokens that must match before the ngram lookup fires
rem  NGMIN: consecutive matches to fire; NGMAX / LOOKUP_N: per-draft caps
rem  BENCH=1 drops the ngram drafter (it warms up, so timing needs it off)
set "DRAFT_N=4"
set "NMATCH=24"
set "NGMIN=24"
set "NGMAX=24"
set "LOOKUP_N=%NGMAX%"
set "BENCH=0"
set "SPECTYPE=ngram-mod,draft-mtp-adaptive"
if "%BENCH%"=="1" set "SPECTYPE=draft-mtp-adaptive"

title Qwen3.8-27B-Flash-MTP :18080

rem a stale instance holds the port and its VRAM
taskkill /F /IM llama-server.exe >nul 2>nul
ping -n 3 127.0.0.1 >nul

rem vision projector, mounted only when VISION=1 and the file exists
set "VISION=0"
set "MMPROJ=%~dp0weights\mmproj-Qwen3.8-27B-Q8_0.gguf"
set "VISARG="
if "%VISION%"=="1" if exist "%MMPROJ%" set "VISARG=--mmproj %MMPROJ% --image-min-tokens 1024"

rem llmchat saves and restores whole conversations here (~18 KB/token, gitignored)
if not exist "%~dp0data\slots" mkdir "%~dp0data\slots"

"%SERVER%\llama-server.exe" -m "%~dp0weights\%MODEL%" ^
  --host 127.0.0.1 --port 18080 --parallel 1 --no-ui ^
  --slot-save-path "%~dp0data\slots" ^
  -c %CTX% -b 4096 -t 8 -ngl 99 -fa on -ctk %CTK% -ctv %CTV% --kv-vram-cells %CELLS% ^
  -rea on --reasoning-effort %EFFORT% ^
  --reasoning-budget %REBUDGET% --reasoning-budget-message "%REBMSG%" ^
  --spec-type %SPECTYPE% --spec-draft-n-max %DRAFT_N% ^
  --spec-lookup-n-max %LOOKUP_N% --spec-ngram-mod-n-min %NGMIN% ^
  --spec-ngram-mod-n-max %NGMAX% ^
  --spec-ngram-mod-n-match %NMATCH% ^
  --features-file features.ini --features-section qwen ^
  --alias "Qwen3.8-27B-Flash-MTP" ^
  %VISARG%
