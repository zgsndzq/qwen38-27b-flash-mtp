@echo off
cd /d %~dp0

rem == the only options worth changing ==
rem SLOTS: parallel slots; 1 = one conversation owns all of -c (2 ways is already saturated)
set "SLOTS=1"
rem CTX: 8192 / 16384 / 24576 / 32768 / 65536; larger = more VRAM
set "CTX=32768"
rem DRAFT: draft depth cap; 1 is faster on short prompts, 3 wins on long inputs
set "DRAFT=3"
rem REA: off / on / auto -- client may override
set "REA=on"
rem EFFORT: xhigh / medium / low -- only used when thinking is on
set "EFFORT=low"
rem TEMP: sampling temperature
set "TEMP=0.6"

set PATH=%~dp0build\cuda\bin;%PATH%
set CUDA_PATH=%~dp0build\cuda
set LLAMA_SPEC_PQ=0
set GGML_CUDA_GRAPH_OPT=1
set GGML_CUDA_PDL=1
rem must match the kv-bias calibration
set LLAMA_ATTN_ROT_DISABLE=1

title Ternary-Bonsai-2-27B-SelfMTP-Q6K :18080

rem -ub 256: finer prompt reuse (a cached-doc decision goes 2.2 s -> 0.7 s) for ~1% of prefill speed
rem q4_0 KV is required by --kv-mean-center; the draft map keeps one draft step at 26 MB not 265 MB
"%~dp0build\bin\llama-server.exe" -m "%~dp0weights\Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf" ^
  --host 127.0.0.1 --port 18080 --parallel %SLOTS% -rea %REA% ^
  -c %CTX% -b 4096 -ub 256 -t 8 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 ^
  --temp %TEMP% --top-p 0.95 --top-k 20 --reasoning-effort %EFFORT% ^
  --reasoning-preserve --kv-mean-center "%~dp0calib\kv-bias.Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf" ^
  --reasoning-budget 2048 --reasoning-budget-message "\n\nBudget exceeded, answer directly.\n" ^
  --spec-type none --spec-draft-n-max %DRAFT% --spec-draft-n-min-adaptive 1 ^
  --spec-draft-vocab-map "%~dp0maps\shortlist_24k_mix.vocab.txt" --spec-draft-vocab-hot 0
