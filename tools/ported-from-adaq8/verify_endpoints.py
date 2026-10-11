#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the endpoints Claude Code actually needs, on the live launcher.

Starts llmserver.bat, waits for /health, then hits:
  GET  /props                        -> n_ctx / n_slots / model path
  GET  /v1/models                    -> the API model name (--alias)
  POST /v1/messages                  -> Anthropic messages API (non-stream)
  POST /v1/messages                  -> the same, streamed
Kills by image name at the end.  Read-only otherwise.
"""
import io
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

ROOT = r'F:\super\qwen38mtp'
PORT = 18080
OUT = r'F:\super\tmp\_api_verify.txt'
fh = io.open(OUT, 'w', encoding='utf-8')


def log(s):
    fh.write(s + '\n')
    fh.flush()


def post(path, obj, raw=False):
    data = json.dumps(obj).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (PORT, path), data=data,
                                 headers={'Content-Type': 'application/json',
                                          'x-api-key': 'local',
                                          'anthropic-version': '2023-06-01'})
    r = urllib.request.urlopen(req, timeout=300)
    body = r.read().decode('utf-8', 'replace')
    return body if raw else json.loads(body)


def get(path):
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (PORT, path),
                                 headers={'x-api-key': 'local'})
    return urllib.request.urlopen(req, timeout=30).read().decode('utf-8', 'replace')


subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
time.sleep(2)

env = dict(os.environ)
p = subprocess.Popen(['cmd', '/c', 'llmserver.bat'], cwd=ROOT, env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t0 = time.time()
ready = False
while time.time() - t0 < 300:
    try:
        if 'ok' in get('/health'):
            ready = True
            break
    except Exception:
        pass
    time.sleep(1)
log('health ready in %.1f s (pid=%s)' % (time.time() - t0, p.pid))
if not ready:
    log('SERVER NEVER CAME UP')
    subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
    fh.close()
    raise SystemExit(1)

try:
    props = json.loads(get('/props'))
    log('n_ctx   : %s' % props.get('n_ctx'))
    log('n_slots : %s' % props.get('n_slots'))
    log('model   : %s' % props.get('model_path'))
    log('default gen params: %s' % json.dumps(props.get('default_generation_settings', {}).get('params', {}))[:200])
except Exception as e:
    log('props failed: %r' % (e,))

log('')
log('--- GET /v1/models ---')
try:
    m = json.loads(get('/v1/models'))
    log(json.dumps(m, ensure_ascii=False)[:800])
except Exception as e:
    log('failed: %r' % (e,))

body = {'model': 'Ternary-Bonsai-2-27B',
        'max_tokens': 48,
        'messages': [{'role': 'user', 'content': '用一句话说明你是什么模型。'}]}

log('')
log('--- POST /v1/messages (non-stream, anthropic) ---')
try:
    t1 = time.time()
    r = post('/v1/messages', body)
    log('wall %.2f s' % (time.time() - t1))
    log(json.dumps(r, ensure_ascii=False)[:1200])
except urllib.error.HTTPError as e:
    log('HTTP %d: %s' % (e.code, e.read().decode('utf-8', 'replace')[:800]))
except Exception as e:
    log('failed: %r' % (e,))

log('')
log('--- POST /v1/messages (stream, first 6 events) ---')
try:
    req = urllib.request.Request('http://127.0.0.1:%d/v1/messages' % PORT,
                                 data=json.dumps(dict(body, stream=True)).encode('utf-8'),
                                 headers={'Content-Type': 'application/json',
                                          'x-api-key': 'local',
                                          'anthropic-version': '2023-06-01'})
    r = urllib.request.urlopen(req, timeout=300)
    n = 0
    for line in r:
        s = line.decode('utf-8', 'replace').rstrip()
        if s:
            log('  %s' % s[:160])
            n += 1
        if n >= 12:
            break
except urllib.error.HTTPError as e:
    log('HTTP %d: %s' % (e.code, e.read().decode('utf-8', 'replace')[:600]))
except Exception as e:
    log('failed: %r' % (e,))

log('')
log('--- POST /v1/messages/count_tokens ---')
try:
    r = post('/v1/messages/count_tokens', {'model': body['model'], 'messages': body['messages']})
    log(json.dumps(r, ensure_ascii=False)[:300])
except Exception as e:
    log('failed: %r' % (e,))

subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
log('')
log('cleanup done')
fh.close()
