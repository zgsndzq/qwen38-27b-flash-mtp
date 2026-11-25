@echo off
rem 27B memory variant: published model untouched, memory is a separate artifact. ASCII + CRLF.
cd /d "%~dp0"
rem memory master switch. features.ini leaves LLAMA_MEMORY unset on purpose, so this
rem profile is the only one that turns the layer on.
set "LLAMA_MEMORY=1"
python "%~dp0run_server.py" server.mem.json
pause