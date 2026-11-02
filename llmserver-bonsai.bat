@echo off
rem Ternary-Bonsai-2-9B (Qwen3.5-9B backbone, PQ2_0 + MTP head) on port 18080.
rem The numbers live in server.bonsai.json; this keeps the old name working.
rem Keep this file ASCII + CRLF.
cd /d "%~dp0"
set "PY=python"
title Ternary-Bonsai-2-9B
"%PY%" "%~dp0run_server.py" bonsai
pause
