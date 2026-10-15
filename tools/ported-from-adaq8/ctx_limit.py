#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How far can -c actually go on this 8 GB card?  Not a guess: start the real
server, read n_ctx back from /props, read VRAM from nvidia-smi, kill by PID.

Usage: ctx_limit.py 131072 262144
"""
import io
import os
import subprocess
import sys
import time
import urllib.request

ROOT = r'F:\super\qwen27b'
BIN = os.path.join(ROOT, 'bin', 'llama-server.exe')
MODEL = os.path.join(ROOT, 'models', 'Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf')
MAP = os.path.join(ROOT, 'maps', 'shortlist_24k_mix.vocab.txt')
PORT = 18099
OUT = r'F:\super\tmp\_ctx_limit.txt'


def log(s, fh):
    fh.write(s + '\n')
    fh.flush()


def get(path):
    r = urllib.request.urlopen('http://127.0.0.1:%d%s' % (PORT, path), timeout=10)
    return r.read().decode('utf-8', 'replace')


def vram():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                            '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=15).stdout.strip()
        u, t = o.split(',')
        return int(u.strip()), int(t.strip())
    except Exception:
        return None, None


def run(ctx, tier, fh):
    log('', fh)
    log('===== -c %d  %s =====' % (ctx, ('--kv-vram-cells %d' % tier) if tier else '(no tier)'), fh)
    env = dict(os.environ)
    env['PATH'] = os.path.join(ROOT, 'cuda', 'bin') + ';' + env.get('PATH', '')
    env['CUDA_PATH'] = os.path.join(ROOT, 'cuda')
    cmd = [BIN, '-m', MODEL, '--host', '127.0.0.1', '--port', str(PORT),
           '--parallel', '1', '-rea', 'off', '-c', str(ctx), '-b', '4096', '-t', '8',
           '-ngl', '99', '-fa', 'on', '--spec-type', 'draft-mtp',
           '--spec-draft-vocab-map', MAP]
    if tier:
        cmd += ['--kv-vram-cells', str(tier)]
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding='utf-8', errors='replace')
    ok = False
    deadline = t0 + 300
    while time.time() < deadline:
        if p.poll() is not None:
            log('SERVER EXITED rc=%s' % p.returncode, fh)
            try:
                tail = p.stdout.read()
            except Exception:
                tail = ''
            log('--- tail ---\n%s' % tail[-3000:], fh)
            return False
        try:
            h = get('/health')
            if '"ok"' in h or 'ok' in h:
                ok = True
                break
        except Exception:
            pass
        time.sleep(1.0)
    if not ok:
        log('TIMEOUT waiting for /health', fh)
        try:
            p.kill()
        except Exception:
            pass
        return False
    log('load + health ready in %.1f s' % (time.time() - t0), fh)
    try:
        import json
        props = json.loads(get('/props'))
        log('n_ctx        : %s' % props.get('n_ctx'), fh)
        log('n_slots      : %s' % props.get('n_slots'), fh)
        log('model_params : %s' % props.get('model_params'), fh)
    except Exception as e:
        log('props failed: %r' % (e,), fh)
    u, t = vram()
    if u is not None:
        log('vram         : %d / %d MiB' % (u, t), fh)
    return True


def main():
    fh = io.open(OUT, 'w', encoding='utf-8')
    try:
        for arg in sys.argv[1:]:
            if ':' in arg:
                c, t = arg.split(':')
                run(int(c), int(t), fh)
            else:
                c = int(arg)
                run(c, 24000 if c > 32768 else 0, fh)
    finally:
        subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'],
                       capture_output=True, text=True)
        time.sleep(2)
        log('', fh)
        log('cleanup done', fh)
        fh.close()


main()
