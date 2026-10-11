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
rem ngram lookup drafter (engine patch 0039): drafts text already present in the context.
rem LOOKUP_N caps one draft; NGMIN is the minimum consecutive match it needs to fire.
rem Measured 2026-10-11, copy-heavy scenes (free writing is unaffected, ~43 t/s either way):
rem   rewrite a config file: 76 t/s with MTP alone -> 92 t/s with the lookup drafter.
rem Lower NGMIN fires more often but accepts less; 16 is the knee (8 and 12 measured slower).
set "LOOKUP_N=16"
set "NGMIN=16"

rem this fork's CUDA runtime
set PATH=%~dp0cuda\bin;%PATH%
set CUDA_PATH=%~dp0cuda

title Ternary-Bonsai-2-27B-ADA-Q8 :18080

rem ---- application switches, kept out of the upstream recipe ----
rem vision: 1 mounts the projector for jev-image; 0 leaves it off (measured: only ~43 MiB, not 288)
rem image-min-tokens: Qwen-VL wants >= 1024 image tokens; below that grounding accuracy drops
set "VISION=1"
set "VISARG="
if "%VISION%"=="1" set "VISARG=--mmproj "%~dp0models\mmproj-Qwen3.8-27B-Q8_0.gguf" --image-min-tokens 1024"
rem speculative decoding: the lookup drafter runs first and falls back to the MTP draft
rem when it has nothing to copy, so free-form text keeps the plain MTP path.
"%~dp0bin\llama-server.exe" -m "%~dp0models\%MODEL%" ^
  --features-file "%~dp0features.ini" ^
  --host 127.0.0.1 --port 18080 --parallel 1 -rea off ^
  -c %CTX% -b 4096 -t 8 -ngl 99 -fa on ^
  --reasoning-budget %REBUDGET% ^
  --spec-type ngram-mod,draft-mtp-adaptive --spec-draft-n-max %DRAFT_N% ^
  --spec-lookup-n-max %LOOKUP_N% --spec-ngram-mod-n-min %NGMIN% ^
  --alias "Ternary-Bonsai-2-27B" ^
  %VISARG%
