@echo off
rem LFM2.5-24B-A2B (lfm2moe, IQ2_XXS) on port 18081.
rem One engine binary, two model profiles: this is the MoE profile.
rem This file must stay ASCII + CRLF - cmd breaks on anything else.
setlocal
set "HERE=%~dp0"
set "SERVER="
if exist "%~dp0build\src\bin\llama-server.exe" set "SERVER=%~dp0build\src\bin"
if not defined SERVER if exist "%~dp0build\bin\llama-server.exe" set "SERVER=%~dp0build\bin"
if not defined SERVER (
  echo llama-server.exe not found under build - build it first.
  exit /b 1
)
rem runtime dlls live next to the binary and under build\bin
if exist "%~dp0build\cuda\bin" (
  set "PATH=%~dp0build\cuda\bin;%PATH%"
  set "CUDA_PATH=%~dp0build\cuda"
)
set "PATH=%SERVER%;%~dp0build\bin;%PATH%"
set "MODEL=LFM2.5-24B-A2B-UD-IQ2_XXS.gguf"
set "MODELDIR=%~dp0..\hf\lfm2.5-24b-a2b"
rem full 128K context: 6.9 GB of weights + q4_0 KV measured at 7895/8188 MiB
set "CTX=131072"
set "PORT=18081"
if not exist "%~dp0logs" mkdir "%~dp0logs"
title LFM2.5-24B-A2B :%PORT%
rem features.ini is relative: the working directory must be src
cd /d "%~dp0"
rem a stale instance keeps the port and its VRAM
taskkill /F /IM llama-server.exe >nul 2>nul
ping -n 3 127.0.0.1 >nul
"%SERVER%\llama-server.exe" -m "%MODELDIR%\%MODEL%" ^
  --host 127.0.0.1 --port %PORT% --parallel 1 -rea off --no-ui ^
  -c %CTX% -b 2048 -t 8 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 ^
  --features-file features.ini --features-section lfm ^
  --alias "LFM2.5-24B-A2B" ^
  --log-file "%~dp0logs\lfm_server.log"
endlocal
