#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B the draft-depth arms of adaq8: fixed 3 vs adaptive [1..3] vs adaptive [2..3].

Same-session per arm, >=3 prompts x >=3 runs, median. cache_prompt:false on every
arm (the rule). Each arm = the production llmserver.bat with only --spec-type (and
any extra flag) changed, so the bat stays the single source of truth.

    python tools/ab_adaptive.py
"""
import io
import json
import os
import re
import socket
import statistics as st
import subprocess
import sys
import time
import urllib.request

ROOT = r'F:\super\qwen38mtp'
PROD_BAT = os.path.join(ROOT, 'llmserver.bat')
PORT = 18080
LOG = os.path.join(ROOT, 'tools', '_ab_server.log')
OUT = r'F:\super\qwen38mtp\docs\ported-from-adaq8\ADAPTIVE-AB.txt'

ARMS = [
    ('fixed3',    'draft-mtp',          []),
    ('adapt[1..3]', 'draft-mtp-adaptive', []),
    ('adapt[2..3]', 'draft-mtp-adaptive', ['--spec-draft-n-min-adaptive', '2']),
]

PROMPTS = [
    ('cn.prose', '请用中文详细解释 Transformer 里的自注意力机制，包括 Q/K/V 的作用。'),
    ('en.code',  'Write a C++ implementation of a lock-free single-producer single-consumer ring '
                 'buffer. Explain the memory ordering choices you made.'),
    ('cn.code',  '用 C++ 实现一个 LRU 缓存，要求 get 和 put 都是 O(1)，注释和说明都用中文。'),
]

N_PREDICT = 192
RUNS = 3

ACC_RE = re.compile(r'draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)')

def port_busy():
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return s.connect_ex(('127.0.0.1', PORT)) == 0
    finally:
        s.close()


def make_variant(spec, extra, name='llmserver-ab.bat'):
    raw = io.open(PROD_BAT, encoding='utf-8', newline='').read()
    assert '--spec-type draft-mtp' in raw
    out = raw.replace('--spec-type draft-mtp', '--spec-type ' + spec)
    if extra:
        out = out.rstrip() + ' ' + ' '.join(extra) + '\n'
    out = out.replace('\r\n', '\n').replace('\n', '\r\n')
    out.encode('ascii')          # .bat must stay pure ASCII
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


def ask(prompt):
    body = json.dumps({'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': N_PREDICT, 'cache_prompt': False, 'stream': False}).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % PORT, data=body,
                                 headers={'Content-Type': 'application/json; charset=utf-8'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode('utf-8'))


def median(xs):
    return st.median(xs) if xs else float('nan')


def main():
    global RUNS, OUT
    args = sys.argv[1:]
    if '--runs' in args:
        RUNS = int(args[args.index('--runs') + 1])
    only = None
    if '--arms' in args:
        only = args[args.index('--arms') + 1].split(',')
    if '--out' in args:
        OUT = args[args.index('--out') + 1]
    arms = [a for a in ARMS if only is None or a[0] in only]

    fh = io.open(OUT, 'w', encoding='utf-8')

    def log(m):
        print(m)
        fh.write(m + '\n')
        fh.flush()

    if port_busy():
        log('ABORT: port %d already in use' % PORT)
        return 2

    results = {}
    lf = None
    for label, spec, extra in arms:
        bat = make_variant(spec, extra)
        log('')
        log('=' * 78)
        log('ARM %-12s  spec=%-18s extra=%s' % (label, spec, ' '.join(extra) or '-'))
        log('=' * 78)
        if lf:
            lf.close()
        lf = io.open(LOG, 'w', encoding='utf-8', errors='replace')
        proc = subprocess.Popen(['cmd.exe', '/c', bat], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
        try:
            dt = wait_health()
            if dt is None:
                log('ABORT: /health never came up')
                return 3
            log('health 200 after %.1f s' % dt)
            log('%-10s %-4s %8s %9s %9s %9s' % ('prompt', 'run', 'tg t/s', 'accept', 'depth', 'tok/round'))
            rows = []
            for pname, ptext in PROMPTS:
                for i in range(RUNS):
                    res = ask(ptext)
                    tm = res['timings']
                    pn, pms = tm['predicted_n'], tm['predicted_ms']
                    tg = pn / (pms / 1000.0)
                    dn, da = tm['draft_n'], tm['draft_n_accepted']
                    rounds = pn - da
                    depth = dn / rounds if rounds else 0.0
                    tpr = pn / rounds if rounds else 0.0
                    rows.append((pname, tg, da / dn if dn else 0.0, depth, tpr))
                    log('%-10s %-4d %8.2f %9.4f %9.3f %9.3f' % (pname, i, tg, rows[-1][2], depth, tpr))
            results[label] = rows
        finally:
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2.0)
            lf.flush()
        # controller evidence straight from the server log
        txt = io.open(LOG, encoding='utf-8', errors='replace').read()
        accs = ACC_RE.findall(txt)
        log('server log draft lines: %d' % len(accs))
        if accs:
            log('  first: accept=%s mean_len=%s   last: accept=%s mean_len=%s'
                % (accs[0][0], accs[0][3], accs[-1][0], accs[-1][3]))

    # ---------------------------------------------------------------- summary
    log('')
    log('=' * 78)
    log('SUMMARY  (median per prompt, %d runs each)' % RUNS)
    log('=' * 78)
    log('%-12s %-10s %9s %9s %9s %9s' % ('arm', 'prompt', 'tg t/s', 'accept', 'depth', 'tok/round'))
    for label, _, _ in arms:
        for pname, _ in PROMPTS:
            sel = [r for r in results[label] if r[0] == pname]
            log('%-12s %-10s %9.2f %9.4f %9.3f %9.3f' % (
                label, pname, median([r[1] for r in sel]), median([r[2] for r in sel]),
                median([r[3] for r in sel]), median([r[4] for r in sel])))
    log('')
    log('%-12s %9s %9s   (all prompts pooled)' % ('arm', 'median tg', 'min..max'))
    for label, _, _ in arms:
        tg = [r[1] for r in results[label]]
        log('%-12s %9.2f %9.2f..%.2f' % (label, median(tg), min(tg), max(tg)))
    log('')
    base = [r[1] for r in results['fixed3']]
    for label, _, _ in arms[1:]:
        arm = [r[1] for r in results[label]]
        ov = 'OVERLAP (not separable)' if min(arm) <= max(base) else 'ZERO OVERLAP'
        log('%-12s vs fixed3: %+6.2f%%   %s' % (label, 100.0 * (median(arm) / median(base) - 1.0), ov))
    fh.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
