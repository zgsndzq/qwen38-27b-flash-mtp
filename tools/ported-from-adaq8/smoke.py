# -*- coding: utf-8 -*-
"""30s 冒烟：起真 bat -> 等 health -> 打启动证据 -> 发一次请求 -> 抓 tg/accept -> 杀进程。

路径规矩（2026-09-28 修）：默认走 **/v1/chat/completions + enable_thinking**，也就是
客户端真正用的那条路（会过 chat template、会开思考）。旧的 /completion 原始补全既不过
模板也不开思考，它给的 t/s 和用户体感的 t/s 不是一回事，会误导。要旧的走 SMOKE_MODE=raw。

日志规矩（用户 2026-09-28）：server 的 stdout 全程留在内存里，不逐行落盘；
只有真失败时才 dump 一份到 bench/ 供排查。顺带报告这次启动到底从磁盘读了多少 MB ——
模型是 5790 MB，如果这个数接近它，那本次启动就是磁盘受限的。

启动证据（硬规则 6：服务起来了 != 参数生效了）：health 200 之后，从内存日志里
筛出真正能证明"参数进了内核"的行（KV 类型 / n_ubatch / draft 窗口 / 短表 / 卸载）。
改完参数只看 t/s 是不够的，必须能指着日志说"这行变了"。

规矩：请求发出后 SMOKE_REQ_TIMEOUT（默认 30s）内没有完整回答就立刻停服务。
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _srvlog import Server

ROOT = r"F:\super\qwen38mtp"
BASE = "http://127.0.0.1:18080"
FAIL_DUMP = os.path.join(ROOT, "bench", "_smoke_fail.log")
LOAD_TIMEOUT = float(os.environ.get("SMOKE_LOAD_TIMEOUT", "90"))
REQ_TIMEOUT = float(os.environ.get("SMOKE_REQ_TIMEOUT", "30"))
NPREDICT = int(os.environ.get("SMOKE_NPREDICT", "512"))
MODE = os.environ.get("SMOKE_MODE", "chat")          # chat = real path | raw = /completion
MODEL_MB = 5790.0

PROMPT = ("Write a haiku about the sea, then explain in one sentence why the tide "
          "comes in twice a day.")

# 这些子串是"参数真的进了内核"的证据，不是普通日志噪音
EVIDENCE_KEYS = (
    "n_ctx ",
    "n_batch ",
    "n_ubatch ",
    "K (",                      # KV cache 真实类型：K (q8_0) / V (q8_0)
    "n_rs_seq",
    "offloaded",
    "speculative decoding enabled",
    "draft-only vocabulary",
    "draft context sized",
    "n_ctx_slot",
    "n_slots",
    "failed to",
    "error",
)


def evidence(text, limit=22):
    seen, out = set(), []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s in seen:
            continue
        low = s.lower()
        if any(k.lower() in low for k in EVIDENCE_KEYS):
            seen.add(s)
            out.append(s)
    return out[-limit:]


def build_request():
    if MODE == "raw":
        return BASE + "/completion", {
            "prompt": PROMPT,
            "n_predict": NPREDICT,
            "temperature": 0.0,
            "cache_prompt": False,
        }
    return BASE + "/v1/chat/completions", {
        "model": "gguf",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": NPREDICT,
        "temperature": 0.6,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
    }


def main():
    srv = Server()
    try:
        t0 = time.time()
        try:
            srv.start()
        except RuntimeError as e:
            print("REFUSING TO START: %s" % e)
            return 5

        # --- 1) 等 health -------------------------------------------------
        if not srv.wait_health(LOAD_TIMEOUT):
            print("NO HEALTH in %.0fs" % LOAD_TIMEOUT)
            print("--- server log tail ---")
            print(srv.tail(25))
            print("dumped to %s" % srv.tail_buf.dump(FAIL_DUMP))
            return 3
        print("health OK in %.1fs" % (time.time() - t0))

        li = srv.load_io()
        if li:
            print("load disk IO     = %.1f MB read / %.1f MB written   (model file %.0f MB)"
                  % (li[0], li[1], MODEL_MB))
        else:
            print("load disk IO     = <unavailable>")

        # VRAM 是 KV 类型(q8_0 vs F16 差约 800 MB)唯一能从外部看到的证据：
        # llama 层的 INFO 日志被服务端滤掉了，C/ubatch 看不到，只能看显存账。
        try:
            _v = os.popen('nvidia-smi --query-gpu=memory.used,memory.total '
                          '--format=csv,noheader,nounits').read().strip().splitlines()[0]
        except Exception as e:
            _v = "<unavailable: %r>" % e
        print("vram used/total  = %s MiB   (q8_0 @ c131072/tier24000 measures ~7839)" % _v)

        # --- 1b) 启动证据 --------------------------------------------------
        if os.environ.get("SMOKE_FULLLOG") == "1":
            print("--- FULL in-RAM startup log (%d bytes) ---" % len(srv.tail_buf._buf))
            for ln in srv.tail_buf.text().splitlines():
                print("  " + ln.strip())
        else:
            print("--- startup evidence (proves the flags reached the kernels) ---")
            ev = evidence(srv.tail_buf.text())
            if ev:
                for ln in ev:
                    print("  " + ln)
            else:
                print("  <none matched -- treat as 'flags NOT confirmed active'>")

        # --- 2) 发一次请求，硬超时 ----------------------------------------
        url, payload_in = build_request()
        print("--- request: %s  mode=%s  n_predict=%d ---" % (url, MODE, NPREDICT))
        body = json.dumps(payload_in).encode("utf-8")
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        t1 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print("REQUEST FAILED after %.1fs: %r" % (time.time() - t1, e))
            print("--- server log tail ---")
            print(srv.tail(25))
            print("dumped to %s" % srv.tail_buf.dump(FAIL_DUMP))
            return 4
        dt = time.time() - t1

        tm = payload.get("timings", {})
        if MODE == "raw":
            txt = (payload.get("content") or "").strip()
            reason = ""
        else:
            msg = (payload.get("choices") or [{}])[0].get("message") or {}
            txt = (msg.get("content") or "").strip()
            reason = (msg.get("reasoning_content") or "").strip()

        print("--- SMOKE RESULT ---")
        print("wall            = %.2f s" % dt)
        print("prompt_n        = %s" % tm.get("prompt_n"))
        print("predicted_n     = %s" % tm.get("predicted_n"))
        print("prompt t/s      = %.1f" % (tm.get("prompt_per_second") or 0))
        print("decode  t/s     = %.2f   <== tg" % (tm.get("predicted_per_second") or 0))
        print("first token     = %.0f ms" % (tm.get("prompt_ms") or 0))

        # 草稿接受率就是"思考段为什么慢"的唯一判据，之前的冒烟根本没打它。
        dn, da = tm.get("draft_n"), tm.get("draft_n_accepted")
        if dn:
            print("draft accept    = %.4f (%s accepted / %s generated)"
                  % (float(da or 0) / dn, da, dn))
        else:
            print("draft accept    = <not in timings; see the log tail below>")

        if reason:
            print("thinking chars  = %d" % len(reason))
        print("content chars   = %d" % len(txt))
        print("text head       = %s" % (txt[:110].replace("\n", " / ") or reason[:110].replace("\n", " / ")))

        # 请求之后才出现的证据：draft 接受率、每轮 token 数、tg 曲线
        print("--- server log tail (after the request) ---")
        tail = srv.tail_buf.text().splitlines()
        keep = [l.strip() for l in tail
                if any(k in l for k in ("draft acceptance", "eval time", "print_timing: id",
                                        "tg =", "graphs reused", "n_gen", "Budget exceeded"))]
        for ln in (keep[-12:] or [l.strip() for l in tail[-8:]]):
            print("  " + ln)
        return 0
    finally:
        srv.kill()
        time.sleep(0.6)
        print("--- server stdout kept in RAM: %s bytes captured, %d bytes retained, "
              "0 bytes written to disk ---"
              % (srv.tail_buf.total, len(srv.tail_buf._buf)))


if __name__ == "__main__":
    sys.exit(main())
