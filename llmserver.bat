@echo off
setlocal
cd /d "%~dp0"

rem ===========================================================================
rem  llama.cpp-qwen38-flash runtime launcher
rem
rem  Where the engine binary comes from -- this script accepts both layouts:
rem    1. prebuilt release  : extract the release zip here  -> build\bin
rem    2. built from source : code\_build_q8.bat writes it  -> build\src\bin
rem  Your own build wins when both exist.
rem
rem  All of OUR switches live in features.ini, which the engine reads itself
rem  through --features-file. Only upstream-known parameters are set here.
rem  This file must stay ASCII + CRLF - cmd breaks on anything else.
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

rem CUDA runtime dlls. The prebuilt zip ships them under build\cuda; a user who
rem built from source already has the CUDA Toolkit on PATH.
if exist "%~dp0build\cuda\bin" (
  set "PATH=%~dp0build\cuda\bin;%PATH%"
  set "CUDA_PATH=%~dp0build\cuda"
)
set "PATH=%SERVER%;%~dp0build\bin;%PATH%"

set "MODEL=Qwen3.8-27B-Flash-MTP.gguf"
set "CTX=131072"
rem KV precision + VRAM cell budget, all configured HERE (single source).
rem features.ini no longer carries the numbers: --kv-vram-cells and GGML_CUDA_FA_CHUNK
rem below are derived from CTK/CTV, so the pair can never desync.
rem   q4_0/q4_0  18 KB/cell  -> 65536 cells  (1152 MiB)  best accuracy
rem   q4_0/q3_0  NOT USABLE: mixed types have no compiled FA kernel pair,
rem   every op falls back to a full f16 copy (10x slower). Same type only.
rem   q3_0/q3_0  14 KB/cell  -> 90112 cells  (1232 MiB)  set CTK/CTV to q3_0 for 90K
rem Per cell = 4 KV heads x 256 wide x K+V x 16 attn layers. Cells past the
rem tier live in pinned host RAM (CUDA VMM); CTX stays 131072 either way.
rem Sizes leave ~400 MiB headroom: filling the card triggers VMM page
rem migration on the weights and decode drops ~10x (measured, see task.md T16 #16).
set "CTK=q3_0"
set "CTV=q3_0"
rem tier size lookup for the (K,V) pair above
set "CELLS=90112"
if "%CTK%%CTV%"=="q4_0q4_0" set "CELLS=65536"
if "%CTK%%CTV%"=="q3_0q3_0" set "CELLS=90112"
set "GGML_CUDA_FA_CHUNK=%CELLS%"
rem thinking block token cap. -1 = unlimited, 0 = end at once, N>0 = hard cap.
rem This is the ONLY knob that actually bounds thinking length. reasoning_effort is just
rem a sentence the chat template writes into the system block - the model may ignore it.
rem Standard for this model, see docs/ONE-MODEL-TWO-APIS.md: 2048 plus a message.
set "REBUDGET=2048"
rem The message is tokenized verbatim and forced just before the end-of-thinking tag,
rem so it must be plain text - no \n escapes, nothing is unescaped anywhere.
set "REBMSG= Budget exceeded, answer directly."
rem draft width ceiling, the adaptive controller stays at or below it.
set "DRAFT_N=4"
rem ngram lookup drafter. NMATCH = preceding tokens that must match before it looks,
rem NGMIN = minimum consecutive match to fire, NGMAX / LOOKUP_N = per-draft caps.
set "NMATCH=24"
rem BENCH=1 keeps the MTP draft head and drops the ngram lookup drafter. The ngram cache
rem fills up over a conversation, so a cold first request (~46 t/s) and a warm sixth (~177)
rem are not comparable -- set this to 1 when timing an A/B, back to 0 for production.
set "BENCH=0"
set "SPECTYPE=ngram-mod,draft-mtp-adaptive"
if "%BENCH%"=="1" set "SPECTYPE=draft-mtp-adaptive"
set "NGMIN=24"
set "NGMAX=24"
set "LOOKUP_N=%NGMAX%"

title Qwen3.8-27B-Flash-MTP :18080

rem a stale instance keeps :18080 and its VRAM, so kill it before binding again.
taskkill /F /IM llama-server.exe >nul 2>nul
ping -n 3 127.0.0.1 >nul

rem vision projector. VISION=1 mounts it, but only when the file is actually there.
rem Without mmproj the server refuses every image request, so mount what exists only.
set "VISION=0"
set "MMPROJ=%~dp0weights\mmproj-Qwen3.8-27B-Q8_0.gguf"
set "VISARG="
if "%VISION%"=="1" if exist "%MMPROJ%" set "VISARG=--mmproj %MMPROJ% --image-min-tokens 1024"

rem KV slot store: llmchat saves and restores a whole conversation here.
rem about 18 KB per token, kept under logs so it never enters git.
if not exist "%~dp0data\slots" mkdir "%~dp0data\slots"

"%SERVER%\llama-server.exe" -m "%~dp0weights\%MODEL%" ^
  --host 127.0.0.1 --port 18080 --parallel 1 -rea off --no-ui ^
  --slot-save-path "%~dp0data\slots" ^
  -c %CTX% -b 4096 -t 8 -ngl 99 -fa on -ctk %CTK% -ctv %CTV% --kv-vram-cells %CELLS% ^
  --reasoning-budget %REBUDGET% --reasoning-budget-message "%REBMSG%" ^
  --spec-type %SPECTYPE% --spec-draft-n-max %DRAFT_N% ^
  --spec-lookup-n-max %LOOKUP_N% --spec-ngram-mod-n-min %NGMIN% ^
  --spec-ngram-mod-n-max %NGMAX% ^
  --spec-ngram-mod-n-match %NMATCH% ^
  --features-file features.ini --features-section qwen ^
  --alias "Qwen3.8-27B-Flash-MTP" ^
  %VISARG%
