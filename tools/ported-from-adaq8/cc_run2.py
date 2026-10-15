#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Server on 18080 + raw TCP dump proxy on 18081 + real Claude Code via 18081."""
import io
import os
import subprocess
import sys
import time
import urllib.request

ROOT = r'F:\super\qwen27b'
PY = sys.executable
OUT = r'F:\super\tmp\_cc_run2.txt'
fh = io.open(OUT, 'w', encoding='utf-8')


def log(s):
    fh.write(s + '\n')
    fh.flush()


def ok(url):
    try:
        return 'ok' in urllib.request.urlopen(url, timeout=5).read().decode('utf-8', 'replace')
    except Exception:
        return False


subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
time.sleep(2)
srv = subprocess.Popen(['cmd', '/c', 'llmserver.bat'], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t0 = time.time()
while time.time() - t0 < 300:
    if ok('http://127.0.0.1:18080/health'):
        break
    time.sleep(1)
log('server up in %.1f s' % (time.time() - t0))

px = subprocess.Popen([PY, os.path.join(ROOT, 'tools', 'ported-from-adaq8', 'tcp_dump.py')],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
log('proxy started rc-pending')

claude = os.path.join(os.environ['APPDATA'], 'npm', 'claude.cmd')
settings = r'F:\super\tmp\_claude_proxy_settings.json'
try:
    t1 = time.time()
    r = subprocess.run([claude, '-p', '回答两个字：你好', '--settings', settings],
                       capture_output=True, timeout=240, text=True,
                       encoding='utf-8', errors='replace')
    log('claude rc=%s wall=%.1f s' % (r.returncode, time.time() - t1))
    log('--- stdout ---')
    log((r.stdout or '')[:2000])
    log('--- stderr ---')
    log((r.stderr or '')[:800])
except subprocess.TimeoutExpired:
    log('claude TIMEOUT 240s')

px.kill()
subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
log('cleanup done')
fh.close()
