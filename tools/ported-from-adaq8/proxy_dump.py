#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump what the client really sends: a pass-through proxy in front of 18080.

Listens on 18081, forwards every request verbatim, and records method, path and
the request body (JSON pretty-printed, truncated) plus the response status.
"""
import io
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UP = 'http://127.0.0.1:18080'
PORT = 18081
OUT = r'F:\super\tmp\_proxy_dump.txt'
lock = threading.Lock()
fh = io.open(OUT, 'w', encoding='utf-8')


def log(s):
    with lock:
        fh.write(s + '\n')
        fh.flush()


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _do(self):
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n) if n else b''
        log('')
        log('===== %s %s  (%d bytes) =====' % (self.command, self.path, len(body)))
        if body:
            try:
                obj = json.loads(body.decode('utf-8', 'replace'))
                log('keys: %s' % sorted(obj.keys()))
                if 'system' in obj:
                    s = obj['system']
                    log('system type: %s' % type(s).__name__)
                    if isinstance(s, list):
                        log('system blocks: %d, types=%s'
                            % (len(s), [b.get('type') for b in s if isinstance(b, dict)]))
                        head = ''
                        for b in s:
                            if isinstance(b, dict) and b.get('type') == 'text':
                                head += b.get('text', '')
                        log('system head: %r' % head[:400])
                    else:
                        log('system head: %r' % str(s)[:400])
                if 'messages' in obj:
                    log('messages: %d' % len(obj['messages']))
                    for i, m in enumerate(obj['messages'][:8]):
                        c = m.get('content')
                        if isinstance(c, list):
                            desc = '[%s]' % ','.join(str(b.get('type')) for b in c if isinstance(b, dict))
                        else:
                            desc = repr(str(c)[:120])
                        log('  [%d] role=%-10s %s' % (i, m.get('role'), desc[:200]))
                log('--- body (first 2500 chars) ---')
                log(json.dumps(obj, ensure_ascii=False)[:2500])
            except Exception as e:
                log('not json (%r); first 600: %r' % (e, body[:600]))
        req = urllib.request.Request(UP + self.path, data=body if body else None,
                                     method=self.command)
        for k, v in self.headers.items():
            if k.lower() in ('host', 'content-length', 'connection', 'accept-encoding'):
                continue
            req.add_header(k, v)
        try:
            r = urllib.request.urlopen(req, timeout=600)
            data = r.read()
            status, hdrs = r.status, r.headers
        except urllib.error.HTTPError as e:
            data, status, hdrs = e.read(), e.code, e.headers
        except Exception as e:
            log('upstream error: %r' % (e,))
            self.send_response(502)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        log('<- %s  (%d bytes)  %s' % (status, len(data), data[:160]))
        self.send_response(status)
        ct = hdrs.get('Content-Type', 'application/json')
        self.send_header('Content-Type', ct)
        if 'text/event-stream' in ct:
            self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == '__main__':
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), H)
    log('proxy up on %d -> %s' % (PORT, UP))
    srv.serve_forever()
