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
rem NMATCH   how many preceding tokens must match before it looks at all.
rem NGMIN    minimum consecutive match it needs to fire (below ~16 it fires more but accepts less).
rem NGMAX    ceiling on one ngram draft (--spec-ngram-mod-n-max).
rem LOOKUP_N cap on one lookup draft; pinned to NGMAX so it can chain as far as the table can.
rem 2026-10-12 LOOKUP_N 16 -> NGMAX: the cap was tighter than the table itself, so a long
rem stretch of reused text got cut at 16 even though the drafter could chain 64.
rem Measured 2026-10-11 (MTP alone is the baseline; free writing stays ~43 t/s everywhere;
rem output text verified unchanged):
rem   short copy / config rewrite : copy 65 -> 65 t/s, rewrite 76 -> 95 t/s
rem   a long code-writing task    : 71 t/s, same as MTP alone, no stalls
rem A shorter window scores much higher on copy/rewrite (n_match=8 gave 108/108) but ruins
rem code writing: on a 1800-token HTML task it fell to 51 t/s with visible stalls, because a
rem short window matches the wrong place in freshly generated code. An adaptive stand-down
rem was tried and did not fix it. 24 is the value that never regresses anything.
set "NMATCH=24"
set "NGMAX=64"
set "LOOKUP_N=%NGMAX%"
set "NGMIN=16"

rem this fork's CUDA runtime
set PATH=%~dp0cuda\bin;%PATH%
set CUDA_PATH=%~dp0cuda

title Ternary-Bonsai-2-27B-ADA-Q8 :18080

rem ---- kill a leftover server from a previous run ----
rem A stale instance keeps :18080 bound and its VRAM allocated, so a new start fails
rem to bind (or the client ends up talking to a half-dead server). /F terminates it.
tasklist /FI "IMAGENAME eq llama-server.exe" 2>nul | find /I "llama-server.exe" >nul
if not errorlevel 1 echo [llmserver] killing a leftover instance still holding :18080
taskkill /F /IM llama-server.exe >nul 2>nul
rem let Windows release the port and the device memory before we bind again
ping -n 3 127.0.0.1 >nul

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
  --spec-ngram-mod-n-max %NGMAX% --spec-ngram-mod-n-match %NMATCH% ^
  --alias "Ternary-Bonsai-2-27B" ^
  %VISARG%
