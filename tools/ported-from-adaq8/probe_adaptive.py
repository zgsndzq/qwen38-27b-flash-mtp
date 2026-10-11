#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot probe: start a launcher variant, wait for /health, fire a prompt, dump evidence, kill.

Everything happens inside this single process so the service never outlives the call:
    python tools/probe_adaptive.py --spec draft-mtp-adaptive --out F:/super/tmp/_probe.txt

The launcher is derived from the production llmserver.bat by substituting the
--spec-type value, so nothing else can drift (the bat stays the single source of truth).
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = r'F:\super\qwen38mtp'
PROD_BAT = os.path.join(ROOT, 'llmserver.bat')
PORT = 18080


def log(msg, fh):
    print(msg)
    fh.write(msg + '\n')
    fh.flush()


def make_variant(spec_value, extra=()):
    """llmserver.bat with only the --spec-type value (and any extras) changed."""
    with io.open(PROD_BAT, 'r', encoding='utf-8', newline='') as f:
        raw = f.read()
    assert '--spec-type draft-mtp' in raw, 'production bat does not use draft-mtp'
    out = raw.replace('--spec-type draft-mtp', '--spec-type ' + spec_value)
    for e in extra:
        out = out.rstrip() + ' ' + e + '\n'
    # the bat is a cmd script: keep it CRLF + pure ASCII
    out = out.replace('\r\n', '\n').replace('\n', '\r\n')
    out.encode('ascii')
    name = 'llmserver-probe.bat'
    p = os.path.join(ROOT, name)
    with io.open(p, 'w', encoding='ascii', newline='') as f:
        f.write(out)
    return p


def wait_health(timeout=180.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen('http://127.0.0.1:%d/health' % PORT, timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(0.5)
    return None


def post(path, payload, timeout=600):
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (PORT, path), data=body,
                                 headers={'Content-Type': 'application/json; charset=utf-8'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def kill_tree(pid):
    subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def free_port():
    """Refuse to run when 18080 is already taken -- two servers on one card is the rule."""
    import socket
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return None if s.connect_ex(('127.0.0.1', PORT)) != 0 else -1
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default='draft-mtp-adaptive')
    ap.add_argument('--extra', action='append', default=[])
    ap.add_argument('--prompt', default=None)
    ap.add_argument('--n-predict', type=int, default=256)
    ap.add_argument('--chat', action='store_true')
    ap.add_argument('--dump', action='store_true')
    ap.add_argument('--temp', action='append', type=float, default=[],
                    help='repeatable; each value produces one request pass (omitted = server default)')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--out', default=r'F:\super\tmp\_probe_adaptive.txt')
    a = ap.parse_args()
    temps = a.temp if a.temp else [None]

    fh = io.open(a.out, 'w', encoding='utf-8')
    busy = free_port()
    if busy:
        log('ABORT: port %d already in use by pid %d' % (PORT, busy), fh)
        return 2

    bat = make_variant(a.spec, a.extra)
    log('variant bat : %s' % bat, fh)
    log('spec-type   : %s' % a.spec, fh)

    logpath = os.path.join(ROOT, 'tools', '_probe_server.log')
    lf = io.open(logpath, 'w', encoding='utf-8', errors='replace')
    proc = subprocess.Popen(['cmd.exe', '/c', bat], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    try:
        dt = wait_health()
        if dt is None:
            log('ABORT: /health never returned 200', fh)
            return 3
        log('health 200 after %.1f s' % dt, fh)

        prompt = a.prompt or io.open(os.path.join(ROOT, 'skill.txt'), encoding='utf-8').read()
        log('%-8s %-4s %9s %9s %9s %8s' % ('temp', 'run', 'tg t/s', 'accept', 'pred_ms', 'tokens'), fh)
        for t in temps:
            for rep in range(a.repeat):
                if a.chat:
                    payload = {'messages': [{'role': 'user', 'content': prompt}],
                               'max_tokens': a.n_predict, 'cache_prompt': False, 'stream': False}
                else:
                    payload = {'prompt': prompt, 'n_predict': a.n_predict,
                               'cache_prompt': False, 'stream': False, 'temperature': 0.6}
                if t is not None:
                    payload['temperature'] = t
                t0 = time.time()
                try:
                    res = post('/v1/chat/completions' if a.chat else '/completion', payload)
                except urllib.error.HTTPError as e:
                    log('HTTP %d: %s' % (e.code, e.read().decode('utf-8', 'replace')[:800]), fh)
                    return 4
                wall = time.time() - t0
                tm = res.get('timings', {})
                tr = res.get('tokens_predicted') or tm.get('predicted_n') or 0
                pms = tm.get('predicted_ms') or 0
                tg = (tr / (pms / 1000.0)) if (tr and pms) else 0.0
                dn = tm.get('draft_n') or 0
                da = tm.get('draft_n_accepted') or 0
                log('%-8s %-4d %9.2f %9.4f %9.1f %8d   wall=%.2fs' % (
                    ('%.2f' % t) if t is not None else 'default', rep, tg,
                    (da / dn) if dn else 0.0, pms, tr, wall), fh)
                if a.dump and t is temps[0] and rep == 0:
                    log('--- full response (truncated) ---', fh)
                    log(json.dumps(res, ensure_ascii=False)[:4000], fh)
    finally:
        kill_tree(proc.pid)
        time.sleep(1.5)
        lf.flush()
        lf.close()

    # --- server log evidence ---
    txt = io.open(logpath, encoding='utf-8', errors='replace').read()
    log('', fh)
    log('=== server log: adaptive / spec / offload evidence ===', fh)
    keys = ('adaptive', 'spec ', 'draft', 'offload', 'CUDA0 model buffer', 'n_ctx', 'shortlist',
            'graphs reused', 'load_tensors', 'slot release')
    for line in txt.splitlines():
        low = line.lower()
        if any(k.lower() in low for k in keys):
            log(line[:220], fh)
    log('', fh)
    log('=== server log tail ===', fh)
    for line in txt.splitlines()[-25:]:
        log(line[:220], fh)
    fh.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
