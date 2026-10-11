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
rem Measured 2026-10-11 on this box (MTP alone is the baseline; free writing, the control,
rem stays at ~43 t/s in every configuration; output text verified unchanged):
rem   MTP alone              : copy  65.0  rewrite  76.2
rem   n_match 24 (default)   : copy  65.3  rewrite  94.7
rem   n_match 12             : copy  86.7  rewrite 118.8
rem   n_match  8  <- chosen  : copy 107.5  rewrite 108.3
rem n_match is the lookup window: how many preceding tokens must appear verbatim in the
rem context before the drafter fires. The engine default is 24 and almost never fires here.
set "NMATCH=8"
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
  --spec-ngram-mod-n-match %NMATCH% ^
  --alias "Ternary-Bonsai-2-27B" ^
  %VISARG%
