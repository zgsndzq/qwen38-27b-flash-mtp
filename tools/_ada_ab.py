# -*- coding: utf-8 -*-
"""
A/B driver: production qwen38mtp binary vs the ada-combo (professorpalmer bonsai-combo) build.

Arms (identical prompts / sampling; only binary + spec block + env differ):
  P_prod : production bin + production bat args verbatim (adaptive + vocab-map, VISION=1)
  P_base : production bin + plain draft-mtp n-max 1              <- isolation control
  N_min  : ada-combo bin + same args as P_base                   <- pure kernel delta
  N_rec  : ada-combo bin + professorpalmer's shipped recipe       <- the "breakthrough" arm

  N_min vs P_base = kernel delta only
  N_rec vs P_prod = what the user would actually feel

Also: greedy (temp 0) identity check across arms + long-prompt prefill probe.
Nothing existing is modified: services start and stop inside this process.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT     = r"F:\super\qwen38mtp"
PROD_BIN = os.path.join(ROOT, "bin", "llama-server.exe")
ADA_BIN  = r"F:\super\llm\forks\ada-combo\build-ada\bin\llama-server.exe"

MODEL  = os.path.join(ROOT, "models", "Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf")
MMPROJ = os.path.join(ROOT, "models", "mmproj-Qwen3.8-27B-Q8_0.gguf")
CALIB  = os.path.join(ROOT, "calib",  "kv-bias.Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf")
VMAP   = os.path.join(ROOT, "maps",   "shortlist_24k_mix.vocab.txt")

PORT     = 18080
BASE_URL = "http://127.0.0.1:%d" % PORT
OUT      = r"F:\super\qwen38mtp\tools\_ada_ab_out.txt"

RUNS_PER_PROMPT = 3
N_PREDICT       = 256

PROMPTS = [
    ("code_en", "Write a Python function that merges two sorted lists into one sorted list, without using sorted() or heapq. Then state its time complexity."),
    ("prose_cn", "\u8bf7\u7528\u4e2d\u6587\u8bb2\u89e3\u4e00\u4e0b\uff1a\u4e3a\u4ec0\u4e48\u5927\u6a21\u578b\u63a8\u7406\u65f6 KV \u7f13\u5b58\u80fd\u52a0\u901f\uff0c\u4ee5\u53ca\u5b83\u4f1a\u5360\u7528\u591a\u5c11\u663e\u5b58\u3002"),
    ("code_cn", "\u7528\u4e2d\u6587\u6ce8\u91ca\u5199\u4e00\u4e2a Python \u7c7b\uff0c\u5b9e\u73b0\u4e00\u4e2a\u56fa\u5b9a\u5bb9\u91cf\u7684 LRU \u7f13\u5b58\uff08\u53cc\u5411\u94fe\u8868 + \u54c8\u5e0c\u8868\uff09\uff0c\u5e76\u8bf4\u660e\u6bcf\u4e2a\u64cd\u4f5c\u7684\u590d\u6742\u5ea6\u3002"),
]

GREEDY_PROMPT = "What is 17 + 25? Answer with the number only, then explain in one short sentence."
LOREM = ("The following is a technical report about sparse attention. " * 180)


def common_args():
    return [
        "-m", MODEL,
        "--host", "127.0.0.1", "--port", str(PORT), "--parallel", "1", "-rea", "on",
        "-c", "32768", "-b", "4096", "-ub", "256", "-t", "8",
        "-ngl", "99", "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0",
        "--temp", "0.6", "--top-p", "0.95", "--top-k", "20",
        "--reasoning-effort", "low", "--reasoning-preserve",
        "--kv-mean-center", CALIB,
        "--reasoning-budget", "2048",
        "--reasoning-budget-message", "\n\nBudget exceeded, answer directly.\n",
        "--mmproj", MMPROJ,
    ]


def recipe_args():
    a = common_args()
    a[a.index("-b") + 1] = "2048"
    a[a.index("-ub") + 1] = "512"
    return a


BASE_SPEC = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]

PROD_SPEC = [
    "--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "1",
    "--spec-draft-n-min-adaptive", "1",
    "--spec-draft-vocab-map", VMAP, "--spec-draft-vocab-hot", "0",
]

# professorpalmer's shipped recipe (start-server.ps1), limited to flags bonsai-combo actually has.
# bonsai-combo has no --kv-vram-cells, so no --spec-draft-window / --spec-draft-n-max-tail.
RECIPE_SPEC = [
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
    "-ctkd", "q4_0", "-ctvd", "q4_0",
    "--spec-draft-depth-max", "24576",
    "--backend-sampling",
]

ARMS = [
    ("P_prod", PROD_BIN, common_args() + PROD_SPEC, {}),
    ("P_base", PROD_BIN, common_args() + BASE_SPEC, {}),
    ("N_min",  ADA_BIN,  common_args() + BASE_SPEC, {}),
    ("N_rec",  ADA_BIN,  recipe_args() + RECIPE_SPEC, {"GGML_CUDA_BATCH_INVARIANT": "1"}),
]


def child_env(extra=None):
    e = dict(os.environ)
    e["PATH"] = os.path.join(ROOT, "cuda", "bin") + os.pathsep + r"F:\super\tools\cuda\bin" + os.pathsep + e.get("PATH", "")
    e["CUDA_PATH"] = os.path.join(ROOT, "cuda")
    e["LLAMA_ATTN_ROT_DISABLE"] = "1"
    e["LLAMA_SPEC_PQ"] = "0"
    e["GGML_CUDA_GRAPH_OPT"] = "1"
    e["GGML_CUDA_PDL"] = "1"
    if extra:
        e.update(extra)
    return e


def port_lines():
    try:
        r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=60)
        return [l for l in r.stdout.splitlines() if (":%d " % PORT) in l]
    except Exception:
        return []


def kill_leftovers():
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/NH"],
                           capture_output=True, text=True, timeout=60)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].lower() == "llama-server.exe":
                subprocess.run(["taskkill", "/F", "/PID", parts[1]], capture_output=True, timeout=60)
    except Exception as e:
        print("kill_leftovers:", e)


def health(timeout=900, proc=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            return -1
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(1.0)
    return -2


def post(path, body, timeout=1200):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE_URL + path, data=data,
                                headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def chat(prompt, n_predict, temp=None):
    body = {"messages": [{"role": "user", "content": prompt}],
            "n_predict": n_predict, "cache_prompt": False, "stream": False}
    if temp is not None:
        body["temperature"] = temp
        body["top_p"] = 1.0
        body["top_k"] = 1
        body["seed"] = 1234
    t0 = time.time()
    resp = post("/v1/chat/completions", body)
    wall = time.time() - t0
    try:
        txt = resp["choices"][0]["message"]["content"] or ""
    except Exception:
        txt = json.dumps(resp, ensure_ascii=False)[:400]
    return wall, (resp.get("timings") or {}), txt


def vram():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip()
    except Exception:
        return "?"


def run_arm(name, binary, args, extra_env, results):
    rec = {"arm": name, "binary": binary, "prompts": {}, "greedy": None, "prefill": None}
    results[name] = rec
    print("\n===== ARM %s =====" % name)
    print("bin:", binary)
    if not os.path.exists(binary):
        rec["error"] = "BINARY MISSING"
        print("BINARY MISSING")
        return

    kill_leftovers()
    busy = port_lines()
    if busy:
        rec["error"] = "PORT BUSY"
        print("PORT BUSY -> skip arm:", busy[:2])
        return

    logf = open(r"F:\super\qwen38mtp\tools\_ada_%s.log" % name, "wb")
    proc = subprocess.Popen([binary] + args, stdout=logf, stderr=subprocess.STDOUT,
                            env=child_env(extra_env), cwd=ROOT)
    try:
        t = health(timeout=900, proc=proc)
        if t < 0:
            rec["error"] = "HEALTH FAILED rc=%s" % t
            print("HEALTH FAILED", t)
            return
        rec["load_s"] = round(t, 1)
        print("health 200 after %.1f s" % t)
        time.sleep(3.0)
        rec["vram_mib"] = vram()
        print("vram used:", rec["vram_mib"], "MiB")

        try:
            chat("Say OK.", 8, temp=0.0)
        except Exception as e:
            print("warmup failed:", e)

        for pname, ptext in PROMPTS:
            rows = []
            for i in range(RUNS_PER_PROMPT):
                wall, tim, txt = chat(ptext, N_PREDICT)
                rows.append({"wall": round(wall, 2), "pp": tim.get("prompt_per_second"),
                             "tg": tim.get("predicted_per_second"),
                             "pn": tim.get("prompt_n"), "cn": tim.get("cache_n"),
                             "dn": tim.get("draft_n"), "da": tim.get("draft_n_accepted"),
                             "pred_n": tim.get("predicted_n"), "len": len(txt)})
                print("  %-9s run%d wall=%.2f pp=%.1f tg=%.2f draft=%s/%s" % (
                    pname, i, wall, tim.get("prompt_per_second") or 0,
                    tim.get("predicted_per_second") or 0,
                    tim.get("draft_n"), tim.get("draft_n_accepted")))
            tgs = [r["tg"] for r in rows if r["tg"]]
            pps = [r["pp"] for r in rows if r["pp"]]
            dns = [r["dn"] for r in rows if r["dn"]]
            das = [r["da"] for r in rows if r["da"]]
            rec["prompts"][pname] = {
                "runs": rows,
                "tg_median": round(statistics.median(tgs), 2) if tgs else None,
                "tg_max": round(max(tgs), 2) if tgs else None,
                "pp_median": round(statistics.median(pps), 2) if pps else None,
                "draft_n": int(statistics.median(dns)) if dns else None,
                "draft_acc": int(statistics.median(das)) if das else None,
            }

        _, _, gtxt = chat(GREEDY_PROMPT, 128, temp=0.0)
        rec["greedy"] = gtxt
        print("  greedy:", repr(gtxt[:110]))

        wall, tim, _ = chat(LOREM + "\n\nSummarize the above in one sentence.", 24, temp=0.0)
        rec["prefill"] = {"wall": round(wall, 2), "pp": tim.get("prompt_per_second"),
                          "pn": tim.get("prompt_n"), "tg": tim.get("predicted_per_second")}
        print("  prefill: prompt_n=%s pp=%.1f wall=%.2f" % (
            tim.get("prompt_n"), tim.get("prompt_per_second") or 0, wall))
    finally:
        try:
            proc.terminate()
            time.sleep(2)
        except Exception:
            pass
        if proc.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        try:
            logf.close()
        except Exception:
            pass
        time.sleep(1.5)


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    results = {}
    for name, binary, args, xenv in ARMS:
        if only and name not in only:
            continue
        run_arm(name, binary, args, xenv, results)

    L = []
    A = L.append
    A("=" * 78)
    A("A/B RESULT  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    A("=" * 78)
    for name, _, _, _ in ARMS:
        r = results.get(name)
        if not r:
            continue
        A("")
        A("[%s] load=%ss vram=%s MiB %s" % (name, r.get("load_s"), r.get("vram_mib"), r.get("error", "")))
        A("  bin: %s" % r["binary"])
        for pn, pr in (r.get("prompts") or {}).items():
            A("   %-9s tg_med=%-7s tg_max=%-7s pp_med=%-8s draft=%s/%s" % (
                pn, pr["tg_median"], pr["tg_max"], pr["pp_median"], pr["draft_n"], pr["draft_acc"]))
        pf = r.get("prefill")
        if pf:
            A("   prefill   prompt_n=%s pp=%.1f wall=%ss" % (pf.get("pn"), pf.get("pp") or 0, pf.get("wall")))

    A("")
    A("--- greedy identity ---")
    gs = {}
    for n, _, _, _ in ARMS:
        r = results.get(n)
        if r:
            gs[n] = r.get("greedy")
    for n, g in gs.items():
        A("  %-8s %r" % (n, (g or "")[:90]))
    A("  ==> %s" % ("IDENTICAL across all arms" if len(set(gs.values())) == 1 else "DIFFER"))

    def tg(name, pn):
        try:
            return results[name]["prompts"][pn]["tg_median"]
        except Exception:
            return None

    A("")
    A("--- ratios (tg median) ---")
    for pn, _ in PROMPTS:
        p, b, nm, nr = tg("P_prod", pn), tg("P_base", pn), tg("N_min", pn), tg("N_rec", pn)
        A("  %-9s P_prod=%-7s P_base=%-7s N_min=%-7s N_rec=%-7s | N_min/P_base=%s  N_rec/P_prod=%s" % (
            pn, p, b, nm, nr,
            ("%.3fx" % (nm / b)) if (b and nm) else "-",
            ("%.3fx" % (nr / p)) if (p and nr) else "-"))

    A("")
    A("--- prefill ratios ---")
    try:
        A("  P_prod pp=%.1f | N_min pp=%.1f | N_rec pp=%.1f | N_rec/P_prod=%s" % (
            results["P_prod"]["prefill"]["pp"] or 0,
            results["N_min"]["prefill"]["pp"] or 0,
            results["N_rec"]["prefill"]["pp"] or 0,
            "%.3fx" % ((results["N_rec"]["prefill"]["pp"] or 0) / (results["P_prod"]["prefill"]["pp"] or 1))))
    except Exception as e:
        A("  (incomplete: %s)" % e)

    txt = "\n".join(L)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(txt)
    with open(r"F:\super\qwen38mtp\tools\_ada_ab.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(txt)
    print("\nok -> " + OUT)


if __name__ == "__main__":
    main()
