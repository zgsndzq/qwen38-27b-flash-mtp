@echo off
cd /d %~dp0

rem Recipe and rollback: docs\ported-from-adaq8\PORTED.md
rem Our fork features live in features.ini - edit them there, not here.

rem ---- runtime parameters, all recognised by the upstream engine ----
rem model file, relative to \models
set "MODEL=Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf"
rem context window
set "CTX=65536"
rem cap on tokens inside the thinking block, 0 ends it immediately
set "REBUDGET=2048"
rem ceiling on tokens drafted per step; the adaptive controller stays at or below it
set "DRAFT_N=4"

rem this fork's CUDA runtime
set PATH=%~dp0cuda\bin;%PATH%
set CUDA_PATH=%~dp0cuda

title Ternary-Bonsai-2-27B-ADA-Q8 :18080

rem ---- application switches, kept out of the upstream recipe ----
rem vision: 1 mounts the projector for jev-image, 0 leaves it off (saves about 288 MiB)
set "VISION=1"
set "VISARG="
if "%VISION%"=="1" set "VISARG=--mmproj "%~dp0models\mmproj-Qwen3.8-27B-Q8_0.gguf""
"%~dp0bin\llama-server.exe" -m "%~dp0models\%MODEL%" ^
  --features-file "%~dp0features.ini" ^
  --host 127.0.0.1 --port 18080 --parallel 1 -rea off ^
  -c %CTX% -b 4096 -t 8 -ngl 99 -fa on ^
  --reasoning-budget %REBUDGET% ^
  --spec-type draft-mtp-adaptive --spec-draft-n-max %DRAFT_N% ^
  --alias "Ternary-Bonsai-2-27B" ^
  %VISARG%
