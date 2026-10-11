@echo off
setlocal
set "BIN=F:\super\llm\forks\llamAmpere\build-ptq1\bin"
set "MODEL=F:\super\llm\models\Ternary-Bonsai-2-27B-Base\weights\Ternary-Bonsai-2-27B-PTQ1_0.gguf"
set "PATH=%BIN%;F:\super\tools\cuda\bin;%PATH%"
"%BIN%\llama-cli.exe" -m "%MODEL%" -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --temp 0 -n 96 -p "The capital of France is" < NUL 2> F:\super\_ptq1_gen_err.txt 1> F:\super\_ptq1_gen_out.txt
echo GEN_EXIT=%ERRORLEVEL%
endlocal
