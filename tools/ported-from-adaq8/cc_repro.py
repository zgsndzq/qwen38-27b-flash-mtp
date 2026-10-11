#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce Claude Code's exact /v1/messages shapes against the live server."""
import io
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

ROOT = r'F:\super\qwen38mtp'
PORT = 18080
OUT = r'F:\super\tmp\_cc_repro.txt'
fh = io.open(OUT, 'w', encoding='utf-8')


def log(s):
    fh.write(s + '\n')
    fh.flush()


def post(path, obj):
    data = json.dumps(obj).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (PORT, path), data=data,
                                 headers={'Content-Type': 'application/json',
                                          'x-api-key': 'local',
                                          'anthropic-version': '2023-06-01'})
    try:
        r = urllib.request.urlopen(req, timeout=240)
        return r.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        return 'HTTP %d: %s' % (e.code, e.read().decode('utf-8', 'replace')[:500])


subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
time.sleep(2)
p = subprocess.Popen(['cmd', '/c', 'llmserver.bat'], cwd=ROOT,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t0 = time.time()
while time.time() - t0 < 300:
    try:
        req = urllib.request.Request('http://127.0.0.1:%d/health' % PORT)
        if 'ok' in urllib.request.urlopen(req, timeout=5).read().decode():
            break
    except Exception:
        time.sleep(1)
log('ready in %.1f s' % (time.time() - t0))

SYS_TXT = 'You are Claude Code, Anthropic official CLI for Claude.'

cases = {
    'A system as plain string':
        {'system': SYS_TXT,
         'messages': [{'role': 'user', 'content': 'hi'}]},
    'B system as array with cache_control (CC shape)':
        {'system': [{'type': 'text', 'text': SYS_TXT,
                     'cache_control': {'type': 'ephemeral'}}],
         'messages': [{'role': 'user', 'content': 'hi'}]},
    'C CC full shape: system array + tools + stream':
        {'system': [{'type': 'text', 'text': SYS_TXT,
                     'cache_control': {'type': 'ephemeral'}}],
         'tools': [{'name': 'Bash', 'description': 'run shell',
                    'input_schema': {'type': 'object', 'properties': {'command': {'type': 'string'}},
                                     'required': ['command']}}],
         'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}],
         'stream': True, 'max_tokens': 64},
    'D system array of TWO blocks':
        {'system': [{'type': 'text', 'text': SYS_TXT},
                    {'type': 'text', 'text': 'Second part.'}],
         'messages': [{'role': 'user', 'content': 'hi'}]},
    'E empty system string':
        {'system': '',
         'messages': [{'role': 'user', 'content': 'hi'}]},
}

for name, body in cases.items():
    log('')
    log('=== %s ===' % name)
    body.setdefault('model', 'Ternary-Bonsai-2-27B')
    body.setdefault('max_tokens', 32)
    t1 = time.time()
    r = post('/v1/messages', body)
    log('wall %.1f s' % (time.time() - t1))
    log(r[:700])

subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
log('')
log('cleanup done')
fh.close()
