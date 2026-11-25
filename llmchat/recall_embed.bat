@echo off
rem Optional semantic recall server. CPU only (-ngl 0): 0 MiB VRAM, 27B keeps the GPU.
rem Without it recall falls back to symbol matching and still works.
set "BIN=%~dp0..\build\src\bin"
set "PATH=%BIN%;%~dp0..\build\cuda\bin;%PATH%"
start "recall-embed" /min "%BIN%\llama-server.exe" -m "E:\models\embeddinggemma2\embeddinggemma-2-UD-Q4_K_XL.gguf" --embeddings --pooling mean --embd-normalize 2 -c 2048 -ngl 0 --host 127.0.0.1 --port 18090 --no-webui
echo recall embedding server on 18090 (CPU, 0 MiB VRAM)
