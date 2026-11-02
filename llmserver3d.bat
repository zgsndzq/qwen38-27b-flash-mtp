@echo off
rem Qwen3.5-4B-3D-Coder Q8_0 (ccore fine-tune of Qwen3.5-4B) on port 18080.
rem Flags follow the model card: all layers on GPU, flash attn on, 64K ctx,
rem parallel 1, and for Q8_0 the KV cache at q8_0. Sampling defaults come from
rem the card's table because llmchat only sends temperature. This file must stay
rem ASCII + CRLF.
rem
rem 18080 is shared with llmserver.bat and llmserver-bonsai.bat - one server at a
rem time, and llmchat names its session after the port, so the chat history carries
rem over when you switch models. Type /clear in llmchat after swapping. This kills
rem any running llama-server to take the VRAM.
setlocal
set "HERE=%~dp0"
set "SERVER="
if exist "%~dp0build\src\bin\llama-server.exe" set "SERVER=%~dp0build\src\bin"
if not defined SERVER if exist "%~dp0build\bin\llama-server.exe" set "SERVER=%~dp0build\bin"
if not defined SERVER (
  echo llama-server.exe not found under build - build it first.
  exit /b 1
)
if exist "%~dp0build\cuda\bin" (
  set "PATH=%~dp0build\cuda\bin;%PATH%"
  set "CUDA_PATH=%~dp0build\cuda"
)
set "PATH=%SERVER%;%~dp0build\bin;%PATH%"

set "MODEL=Qwen3.5-4B-3D-Coder-Q8_0.gguf"
set "MMPROJ=mmproj-Qwen3.5-4B-BF16.gguf"
rem 4.17 GB of weights + q8_0 KV: 64K is the size the card was tested at, and it
rem keeps the whole window inside 8 GB of VRAM. Raise only if you also drop ctk.
set "CTX=65536"
set "CTK=q8_0"
set "CTV=q8_0"

rem No --spec-type here on purpose: this GGUF carries qwen35.nextn_predict_layers=0,
rem so there is no MTP head to draft from, and the card warns against MTP as well.
rem No --features-file either - both [qwen] and [bonsai] sections tune MTP/PQ paths
rem that this plain Q8_0 file does not take.

rem Vision is on when the projector is present - same idiom as run_server.py, which
rem enables it only if the file exists. Name it explicitly: llama-server otherwise
rem tries to fetch an mmproj from the internet. One projector serves both quants and
rem uses the stock Qwen3.5-4B visual encoder, per the model card. Image tokens come
rem out of the same 64K window, and the projector costs extra VRAM - if it will not
rem fit in 8 GB, cut CTX to 32768 before touching anything else.
set "VIS="
if exist "%~dp0weights\%MMPROJ%" set "VIS=--mmproj "%~dp0weights\%MMPROJ%" --image-min-tokens 1024"
if not defined VIS echo note: weights\%MMPROJ% missing - starting text-only

set "PORT=18080"
if not exist "%~dp0logs" mkdir "%~dp0logs"
if not exist "%~dp0data\slots" mkdir "%~dp0data\slots"
if not exist "%~dp0weights\%MODEL%" (
  echo model not found: "%~dp0weights\%MODEL%"
  exit /b 1
)
title Qwen3.5-4B-3D-Coder :%PORT%
cd /d "%~dp0"
taskkill /F /IM llama-server.exe >nul 2>nul
ping -n 3 127.0.0.1 >nul
rem -rea auto lets the chat template decide; llmchat overrides it per request
rem through chat_template_kwargs.enable_thinking.
"%SERVER%\llama-server.exe" -m "%~dp0weights\%MODEL%" ^
  --host 127.0.0.1 --port %PORT% --parallel 1 --no-ui ^
  --slot-save-path "%~dp0data\slots" ^
  -c %CTX% -b 4096 -t 8 -ngl 99 -fa on -ctk %CTK% -ctv %CTV% %VIS% ^
  --temp 0.7 --top-p 0.95 --top-k 20 --repeat-penalty 1.0 ^
  -rea auto ^
  --alias "Qwen3.5-4B-3D-Coder" ^
  --log-file "%~dp0logs\3d_server.log"
endlocal
