@echo off
rem Usage: _bench_ptq1.bat <outfile_tag>
rem Same invocation for baseline and kernel so the tg numbers are directly comparable.
setlocal
set "BIN=F:\super\llm\forks\llamAmpere\build-ptq1\bin"
set "MODEL=F:\super\llm\models\Ternary-Bonsai-2-27B-Base\weights\Ternary-Bonsai-2-27B-PTQ1_0.gguf"
set "PATH=%BIN%;F:\super\tools\cuda\bin;%PATH%"
set "TAG=%~1"
echo ==== bench %TAG% %DATE% %TIME% ==== > F:\super\_ptq1_bench_%TAG%.txt
"%BIN%\llama-bench.exe" -m "%MODEL%" -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -p 0 -n 32,128,512 -r 3 >> F:\super\_ptq1_bench_%TAG%.txt 2>&1
echo BENCH_DONE=%ERRORLEVEL% >> F:\super\_ptq1_bench_%TAG%.txt
endlocal
