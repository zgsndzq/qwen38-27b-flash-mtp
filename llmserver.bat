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
rem KV precision. cells kept in VRAM = features.ini kv-vram-cells (must be <= CTX).
set "CTK=q4_0"
set "CTV=q4_0"
rem thinking block token cap. -1 = unlimited, 0 = end at once, N>0 = hard cap.
set "REBUDGET=-1"
rem draft width ceiling, the adaptive controller stays at or below it.
set "DRAFT_N=4"
rem ngram lookup drafter. NMATCH = preceding tokens that must match before it looks,
rem NGMIN = minimum consecutive match to fire, NGMAX / LOOKUP_N = per-draft caps.
set "NMATCH=24"
set "NGMIN=24"
set "NGMAX=64"
set "LOOKUP_N=%NGMAX%"

title Qwen3.8-27B-Flash-MTP :18080

rem a stale instance keeps :18080 and its VRAM, so kill it before binding again.
taskkill /F /IM llama-server.exe >nul 2>nul
ping -n 3 127.0.0.1 >nul

rem vision projector. VISION=1 mounts it, but only when the file is actually there.
rem Without mmproj the server refuses every image request, so mount what exists only.
set "VISION=1"
set "MMPROJ=%~dp0weights\mmproj-Qwen3.8-27B-Q8_0.gguf"
set "VISARG="
if "%VISION%"=="1" if exist "%MMPROJ%" (
  set "VISARG=--mmproj "%MMPROJ%" --image-min-tokens 1024"
) else (
  echo [warn] %MMPROJ% not found - starting without vision
)

rem KV 槽位存盘目录：llmchat 用它保存/恢复整段 KV（重启后秒续同一段对话）。
rem 每个槽位文件约 18 KB/token（16k 前缀 ≈ 290 MB），放 logs 下不进 git。
if not exist "%~dp0logs\slots" mkdir "%~dp0logs\slots"

"%SERVER%\llama-server.exe" -m "%~dp0weights\%MODEL%" ^
  --host 127.0.0.1 --port 18080 --parallel 1 -rea off --no-ui ^
  --slot-save-path "%~dp0logs\slots" ^
  -c %CTX% -b 4096 -t 8 -ngl 99 -fa on -ctk %CTK% -ctv %CTV% ^
  --reasoning-budget %REBUDGET% ^
  --spec-type ngram-mod,draft-mtp-adaptive --spec-draft-n-max %DRAFT_N% ^
  --spec-lookup-n-max %LOOKUP_N% --spec-ngram-mod-n-min %NGMIN% ^
  --spec-ngram-mod-n-max %NGMAX% ^
  --spec-ngram-mod-n-match %NMATCH% ^
  --features-file features.ini ^
  --alias "Qwen3.8-27B-Flash-MTP" ^
  %VISARG%
