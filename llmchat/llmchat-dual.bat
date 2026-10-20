@echo off
rem Dual-session chat on a 2-slot server: two prompts, then both run side by side.
rem Keep this file ASCII + CRLF (cmd breaks on anything else).
setlocal
python "%~dp0dual.py" %*
exit /b %ERRORLEVEL%
