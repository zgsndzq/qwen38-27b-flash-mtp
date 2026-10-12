@echo off
cd /d %~dp0

rem ROLLBACK BUILD 2026-10-12 - engine = b6aa68094, the 2026-10-10 00:34 binary.
rem Backups: F:\super\llm\_backup\  (bin-newengine-1012\, ada-q8-delta-1012-full.patch)
rem This engine predates --features-file, so what features.ini used to hold is written out here.

rem ---- runtime parameters, all recognised by the upstream engine ----
rem model file, relative to \models
set "MODEL=Swift-Bonsai-2-PTQ1_0-mtp-pro-q4km.gguf"
rem context window
set "CTX=65536"
rem KV cache precision + how many cells stay in VRAM (must not exceed CTX).
rem q4_0 is half the bytes per cell of q8_0, so the WHOLE 64k window fits in the
rem same 1.06 GB that q8_0 used for only 32768 cells. Equal VRAM, no host spill,
rem and no mid-chat drop at the old 32768 line. Set KVVRAM = CTX for all-in-VRAM.
set "CTK=q4_0"
set "CTV=q4_0"
set "KVVRAM=65536"
rem tokens allowed inside the thinking block: -1 = no cap (the model stops when it is
rem done), 0 = end immediately, N>0 = hard cap. Set to -1 on 2026-10-12.
rem The block no longer enters the history (chat.py), so it costs nothing across turns -
rem but inside a single turn it still occupies KV. If it never stops it can eat the whole
rem 64k window and leave no room for the answer; this architecture cannot shift. Watch
rem the "[ctx NNNN tok - water XX%]" line llmchat prints each turn.
set "REBUDGET=-1"
rem ceiling on tokens drafted per step; the adaptive controller stays at or below it
set "DRAFT_N=4"
rem ngram lookup drafter: drafts text that is already present in the context.
rem NMATCH   how many preceding tokens must match before it looks at all.
rem NGMIN    minimum consecutive match it needs to fire.
rem NGMAX    ceiling on one ngram draft (--spec-ngram-mod-n-max).
rem NOTE this rollback engine has no --spec-lookup-n-max (patch 0039), so an ngram draft
rem      is additionally cut at --spec-draft-n-max and cannot chain all the way to NGMAX.
rem      That option comes back when patch 0039 is re-applied on top of this engine.
rem Measured 2026-10-11 on this model (MTP alone is the baseline): rewrite 76 -> 95 t/s,
rem code writing unchanged, free writing unchanged. n_match below 16 stalls code writing.
set "NMATCH=24"
set "NGMAX=64"
set "NGMIN=24"

rem ---- engine switches that used to live in features.ini ----
rem This engine predates --features-file. All three are native environment variables.
rem   GGML_CUDA_PTQ1_MMVQ_MAX : mat-vec column count where PT hands over to MMQ; 5 keeps
rem                             4 draft tokens on the cheap in-place path.
rem   GGML_CUDA_SHARED_POOL    : target and draft share one arena (~1.4 GB on an 8 GB card,
rem                             only safe while --parallel is 1).
rem   LLAMA_MTP_DRAFT_UBATCH   : micro-batch of the draft context.
set "GGML_CUDA_PTQ1_MMVQ_MAX=5"
set "GGML_CUDA_SHARED_POOL=1"
set "LLAMA_MTP_DRAFT_UBATCH=64"

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
  --host 127.0.0.1 --port 18080 --parallel 1 -rea off ^
  -c %CTX% -b 4096 -t 8 -ngl 99 -fa on -ctk %CTK% -ctv %CTV% ^
  --reasoning-budget %REBUDGET% ^
  --spec-draft-sampling probabilistic ^
  --spec-draft-vocab-map maps/shortlist_24k_mix.vocab.txt ^
  --spec-draft-n-min-adaptive 1 ^
  --kv-vram-cells %KVVRAM% --kq-mask-packed ^
  --spec-type ngram-mod,draft-mtp-adaptive --spec-draft-n-max %DRAFT_N% ^
  --spec-ngram-mod-n-min %NGMIN% --spec-ngram-mod-n-max %NGMAX% ^
  --spec-ngram-mod-n-match %NMATCH% ^
  --alias "Ternary-Bonsai-2-27B" ^
  %VISARG%
