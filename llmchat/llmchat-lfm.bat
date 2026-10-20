@echo off
cd /d %~dp0

rem Chat front-end for llmserver-lfm.bat (:18081).
rem chat.py keys the session file and the KV slot by port, so this never
rem mixes history with the qwen chat on 18080.
rem This file must stay ASCII + CRLF - cmd breaks on anything else.

rem PY: full path if "python" is not on PATH
set "PY=python"
rem memory injection: 0 = none, 1 = text, 2 = vision, 3 = mixed
set "LLAMA_CHAT_MEM=0"

title llmchat [LFM2.5-24B-A2B] :18081
rem temp 0.1 / top_k 50 / repeat 1.05 come from the model card the GGUF
rem itself carries (general.sampling.*). no --think/--effort: lfm2moe has no
rem reasoning channel, so those switches would only send dead template args.
"%PY%" "%~dp0chat.py" --base-url http://127.0.0.1:18081 --temp 0.1 --no-sys --mem off %*
pause
