@echo off
rem A/B: same prompt twice, with and without memory injection (no history).
rem Keep this file ASCII + CRLF.
setlocal
set "PY=python"
python "%~dp0mem_ab.py" %*
exit /b %ERRORLEVEL%
