#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Byte-level TCP proxy 18081 -> 18080. Logs every raw request body, forwards
the rest untouched. No HTTP parsing, nothing to break."""
import io
import socket
import threading

UP = ('127.0.0.1', 18080)
PORT = 18081
OUT = r'F:\super\tmp\_raw_dump.txt'
lock = threading.Lock()
fh = io.open(OUT, 'w', encoding='utf-8', buffering=1)


def log(s):
    with lock:
        fh.write(s + '\n')


def pipe(src, dst, tag, cap=None):
    total = 0
    try:
        while True:
            d = src.recv(65536)
            if not d:
                break
            total += len(d)
            if tag == 'C' and total - len(d) < 400000:
                log('>> %s' % d.decode('utf-8', 'replace')[:6000])
            dst.sendall(d)
    except Exception:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except Exception:
            pass


def handle(cs):
    try:
        us = socket.create_connection(UP, timeout=10)
    except Exception as e:
        log('CONNECT FAIL: %r' % (e,))
        cs.close()
        return
    log('=== new conn ===')
    t1 = threading.Thread(target=pipe, args=(cs, us, 'C'), daemon=True)
    t2 = threading.Thread(target=pipe, args=(us, cs, 'S'), daemon=True)
    t1.start(); t2.start()
    t1.join(3600)


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(('127.0.0.1', PORT))
srv.listen(16)
log('raw proxy on %d -> %s' % (PORT, UP))
while True:
    cs, _ = srv.accept()
    threading.Thread(target=handle, args=(cs,), daemon=True).start()
