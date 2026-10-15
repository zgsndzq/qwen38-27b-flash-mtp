@echo off
cd /d %~dp0

rem Our fork's own switches live in features.ini, read by the engine via --features-file.
rem Only upstream-known parameters are set here.
rem Kept on purpose in this project: the vision projector for jev-image.

set "MODEL=Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf"
set "CTX=65536"
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

set PATH=%~dp0build\cuda\bin;%PATH%
set CUDA_PATH=%~dp0build\cuda

title Ternary-Bonsai-2-27B-ADA-Q8 :18080

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

"%~dp0build\bin\llama-server.exe" -m "%~dp0weights\%MODEL%" ^
  --host 127.0.0.1 --port 18080 --parallel 1 -rea off --no-ui ^
  -c %CTX% -b 4096 -t 8 -ngl 99 -fa on -ctk %CTK% -ctv %CTV% ^
  --reasoning-budget %REBUDGET% ^
  --spec-type ngram-mod,draft-mtp-adaptive --spec-draft-n-max %DRAFT_N% ^
  --spec-lookup-n-max %LOOKUP_N% --spec-ngram-mod-n-min %NGMIN% ^
  --spec-ngram-mod-n-max %NGMAX% ^
  --spec-ngram-mod-n-match %NMATCH% ^
  --features-file features.ini ^
  --alias "Ternary-Bonsai-2-27B" ^
  %VISARG%
