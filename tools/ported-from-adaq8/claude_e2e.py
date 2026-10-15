#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end: does the real Claude Code CLI talk to our 18080 server?

Starts llmserver.bat, waits for /health, runs `claude -p <prompt>` with the
user's own settings (no env overrides), records stdout/stderr, kills the server.
"""
import io
import os
import subprocess
import time
import urllib.request

ROOT = r'F:\super\qwen27b'
OUT = r'F:\super\tmp\_claude_e2e.txt'
fh = io.open(OUT, 'w', encoding='utf-8')


def log(s):
    fh.write(s + '\n')
    fh.flush()


def health():
    try:
        return 'ok' in urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=5)\
            .read().decode('utf-8', 'replace')
    except Exception:
        return False


subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
time.sleep(2)
p = subprocess.Popen(['cmd', '/c', 'llmserver.bat'], cwd=ROOT,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t0 = time.time()
while time.time() - t0 < 300:
    if health():
        break
    time.sleep(1)
log('health ready in %.1f s' % (time.time() - t0))

claude = os.path.join(os.environ['APPDATA'], 'npm', 'claude.cmd')
log('claude : %s' % claude)
for prompt in ['回答两个字：你好',
               '用一句话说明 2+2 等于几，不要用工具。']:
    log('')
    log('===== prompt: %s =====' % prompt)
    t1 = time.time()
    try:
        r = subprocess.run([claude, '-p', prompt, '--model', 'Ternary-Bonsai-2-27B'],
                           capture_output=True, timeout=600,
                           text=True, encoding='utf-8', errors='replace')
        log('rc=%s  wall=%.1f s' % (r.returncode, time.time() - t1))
        log('--- stdout ---')
        log((r.stdout or '')[:3000])
        log('--- stderr ---')
        log((r.stderr or '')[:2000])
    except subprocess.TimeoutExpired:
        log('TIMEOUT after 600 s')

subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
log('')
log('cleanup done')
fh.close()
