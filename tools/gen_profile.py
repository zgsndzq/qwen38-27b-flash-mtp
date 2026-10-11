"""Why does tg start ~30% low and climb for a minute?

`tg` in the server log divides by (t_last_gen - t_first_gen), so prompt time is
NOT included -- whatever makes it climb is real decode. This pairs two
measurements taken on the same wall clock:

  * per-128-token decode speed inside one long response (from SSE timestamps)
  * nvidia-smi samples at 1 Hz while that response is being produced

If the two curves move together it is the GPU (clocks, power, memory paging);
if speed climbs while the GPU sits flat it is the model side -- most likely
speculative decoding drafting better later in the response.

Usage:
    python tools/gen_profile.py [--pad 0,2000] [--max-tokens 2560] [--sample 1.0]
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request

QUESTION = ("请详细讲一下推测解码在工程上是怎么落地的：草稿模型怎么选、验证阶段怎么批量、"
            "接受率掉了要怎么排查，以及它和批处理之间的取舍。尽量展开。")

GPU_QUERY = ("timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,"
             "clocks.max.sm,temperature.gpu")


class GpuSampler(threading.Thread):
    def __init__(self, interval):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self.t0 = None
        self.stop = threading.Event()

    def run(self):
        self.t0 = time.perf_counter()
        while not self.stop.is_set():
            t_rel = time.perf_counter() - self.t0
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=" + GPU_QUERY,
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                row = [x.strip() for x in out.split(",")]
                if len(row) >= 7:
                    self.samples.append((t_rel, float(row[1]), float(row[2]),
                                         float(row[3]), float(row[4]),
                                         float(row[5]), float(row[6])))
            except Exception:
                pass
            self.stop.wait(self.interval)


def stream_request(url, messages, max_tokens):
    body = json.dumps({
        "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "cache_prompt": False, "stream": True,
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json; charset=utf-8",
                 "Accept": "text/event-stream"})
    stamps, timings = [], None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            stamps.append(time.perf_counter() - t0)
            if "timings" in obj:
                timings = obj["timings"]
    return stamps, timings


def segment_speeds(stamps, width, nmax=None):
    """Speed of every `width` tokens, using the timestamps of their arrival."""
    n = len(stamps)
    if nmax:
        n = min(n, nmax)
    out = []
    for lo in range(0, n - width, width):
        dt = stamps[lo + width] - stamps[lo]
        if dt > 0:
            out.append((lo + width, width / dt))
    return out


def gpu_window(samples, t_a, t_b):
    inside = [s for s in samples if t_a <= s[0] <= t_b]
    if not inside:
        return None
    avg = lambda i: sum(s[i] for s in inside) / len(inside)
    return {"util": avg(1), "mem": avg(2), "power": avg(3),
            "sm": avg(4), "sm_max": avg(5), "temp": avg(6)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--max-tokens", type=int, default=2560)
    ap.add_argument("--pad", default="0,2000",
                    help="prompt sizes to test, in tokens (0 = bare question)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--sample", type=float, default=1.0)
    args = ap.parse_args(argv)

    sys.path.insert(0, "F:\\super\\qwen38mtp\\tools")
    from ctx_ab import pad_to  # reuse the /tokenize-driven padding helper

    for target in [int(x) for x in args.pad.split(",")]:
        question = pad_to(args.url, QUESTION, target)
        messages = [{"role": "user", "content": question}]

        gpu = GpuSampler(args.sample)
        gpu.start()
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)  # wake up driver
            time.sleep(3.0)
            base_t = time.perf_counter() - gpu.t0
            stamps, timings = stream_request(args.url, messages, args.max_tokens)
        finally:
            gpu.stop.set()
            gpu.join(timeout=5)

        tm = timings or {}
        pn = tm.get("prompt_n") or 0
        pms = tm.get("prompt_ms") or 0.0
        print("\n#### prompt target ~%d, server saw prompt_n=%d, prefill %.2f s (%.0f t/s)"
              % (target, pn, pms / 1000.0, tm.get("prompt_per_second") or 0))
        if tm:
            print("     server: predicted_n=%d  predicted_per_second=%.2f  accept=%s"
                  % (tm.get("predicted_n") or 0,
                     tm.get("predicted_per_second") or 0,
                     (tm.get("draft_n_accepted") or 0) / max(tm.get("draft_n") or 1, 1)))

        idle = gpu_window(gpu.samples, 0, base_t)
        if idle:
            print("     idle   : util %3.0f%%  mem %5.0f MiB  power %5.1f W  sm %4.0f/%4.0f MHz  %3.0f C"
                  % (idle["util"], idle["mem"], idle["power"], idle["sm"], idle["sm_max"], idle["temp"]))

        # The request starts at base_t; north of that everything is decode.
        start = base_t
        print("     %-14s %9s | %s" % ("tokens", "t/s", "GPU over that window"))
        for upto, tps in segment_speeds(stamps, args.width):
            # arrival timestamps are relative to the request, shift into gpu clock
            pass
        # re-walk with absolute gpu-clock times
        for upto, tps in segment_speeds(stamps, args.width):
            t_end = start + stamps[upto]
            t_beg = start + stamps[upto - args.width]
            g = gpu_window(gpu.samples, t_beg, t_end)
            if not g:
                continue
            print("     %4d-%-9d %9.2f | util %3.0f%%  mem %5.0f MiB  power %5.1f W  "
                  "sm %4.0f/%4.0f MHz  %3.0f C"
                  % (upto - args.width + 1, upto, tps, g["util"], g["mem"],
                     g["power"], g["sm"], g["sm_max"], g["temp"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
