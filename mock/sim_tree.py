#!/usr/bin/env python3
"""树式投机解码的蒙特卡洛模拟。

不做任何解析近似：真的逐轮抛硬币，看 target 在每层接不接受，统计实际 gain。
跑 1e6~1e7 轮，输出 gain 分布和对应的 tok/s。

两种候选分布模型：
  geometric  每层 width 个候选，至少命中一个的概率 1-(1-p)^width（独立假设，偏乐观）
  zipf       候选按 Zipf(s) 衰减，算 top-W 的累积概率（更接近真实草稿分布，偏保守）

用法：
    python mock/sim_tree.py                     # 用 params.json 默认值
    python mock/sim_tree.py --rounds 10000000   # 改轮数
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_params(path=None):
    with open(path or os.path.join(HERE, "params.json"), encoding="utf-8") as f:
        return json.load(f)


def zipf_topw(p, width, s):
    """Zipf 模型：候选按概率降序，第 k 名权重 k^-s。

    给定 top-1 概率 p，其余质量按 1/k^s 摊给第 2 名以后，
    返回 target 落在 top-W 里的累积概率（比几何假设保守）。
    """
    if width <= 1:
        return float(p)
    j = np.arange(2, 1024, dtype=np.float64)
    w = j ** (-s)
    return float(min(1.0, p + (1.0 - p) * w[:width - 1].sum() / w.sum()))


def extend_per_depth(per_depth, depth):
    """empirical：直接用实测的逐层单候选命中率。

    实测深度不够时按最后一个比值外推（比值取 p[-1]/p[-2]，夹在 [0, 1]）。
    这比假设 geometric 或 zipf 都更接近真值——那两者本来就是我们猜的形状。
    """
    if not per_depth:
        raise SystemExit("model=empirical 需要 draft.per_depth；先跑 calibrate.py --write")
    p = np.asarray(per_depth, dtype=np.float64)
    if p.size == 0:
        raise SystemExit("draft.per_depth 为空")
    if p.size < depth:
        r = (p[-1] / p[-2]) if p.size > 1 and p[-2] > 0 else 0.85
        r = float(min(1.0, max(0.0, r)))
        ext = p[-1] * (r ** np.arange(1, depth - p.size + 1))
        p = np.concatenate([p, ext])
    return np.clip(p[:depth], 0.0, 1.0)


def simulate(alpha, decay, model, width, budget, n_rounds, seed, chunk,
             zipf_s=1.1, per_depth=None):
    """返回每轮的 gain（1 + 接受深度）。"""
    depth = max(1, budget // width)
    # 每层的单候选命中率：empirical 用实测，其余两种按 alpha 逐层衰减生成
    if model == "empirical":
        p_layers = extend_per_depth(per_depth, depth)
    else:
        p_layers = np.array([alpha * (decay ** k) for k in range(depth)], dtype=np.float64)
    if model in ("geometric", "empirical"):
        hit = 1.0 - (1.0 - p_layers) ** width
    elif model == "zipf":
        hit = np.array([zipf_topw(p, width, zipf_s) for p in p_layers],
                       dtype=np.float64)
    else:
        raise ValueError(model)

    rng = np.random.default_rng(seed)
    gains = np.empty(n_rounds, dtype=np.int8)
    t0 = time.time()
    done = 0
    while done < n_rounds:
        n = min(chunk, n_rounds - done)
        r = rng.random((n, depth))
        # 每层是否通过：r[:, k] < hit[k]
        passed = r < hit[None, :]
        # 第一个 False 的位置就是接受深度；全通过则 depth
        accepted = np.where(passed.all(axis=1), depth, passed.argmin(axis=1))
        gains[done:done + n] = (1 + accepted).astype(np.int8)
        done += n
        if done % (chunk * 4) == 0 or done == n_rounds:
            el = time.time() - t0
            print(f"    {done/1e6:.2f}M 轮  {el:.1f}s  "
                  f"平均 gain {gains[:done].mean():.3f}", file=sys.stderr)
    return gains, depth, hit


# 实测：1M 轮 1.20 s，10M 轮 7.41 s -> 边际约 0.70 us/轮，解释器启动约 0.55 s。
# 这两个数只用于"预估耗时"的预算守卫，不影响任何模拟结果。
SEC_PER_ROUND = 0.70e-6
STARTUP_S = 0.55


def guard_seconds(rounds, budget_s, force):
    """用户硬规矩：任何模拟都不许跑超过 1 分钟。

    超预算不静默截断（那会让"10M 轮"的结论对不上），而是报出可负担的轮数，
    要么按建议改 --rounds，要么显式 --force。
    """
    est = STARTUP_S + rounds * SEC_PER_ROUND
    if est <= budget_s or force:
        return rounds
    max_rounds = int((budget_s - STARTUP_S) / SEC_PER_ROUND // 100_000 * 100_000)
    sys.exit(
        "\n!! 超出时间预算：%.2fM 轮预计 %.1f s > %.0f s\n"
        "   上限约 %.1fM 轮（%.1f s）\n"
        "   用 --rounds %d，或 --force 强制跑。\n"
        % (rounds / 1e6, est, budget_s,
           max_rounds / 1e6, STARTUP_S + max_rounds * SEC_PER_ROUND, max_rounds)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--widths", default=None, help="逗号分隔，覆盖 params")
    ap.add_argument("--budget-seconds", type=float, default=None,
                    help="时间预算，默认取 params.sim.budget_seconds（60）")
    ap.add_argument("--force", action="store_true", help="无视时间预算")
    args = ap.parse_args()

    P = load_params(args.params)
    hw, dr, tr, sim, bl = P["hardware"], P["draft"], P["tree"], P["sim"], P["baseline"]
    alpha = args.alpha if args.alpha is not None else dr["alpha"]
    model = args.model or dr["model"]
    decay = dr["decay"]
    widths = ([int(x) for x in args.widths.split(",")] if args.widths
              else tr["widths"])
    budget = tr["budget"]
    rounds = args.rounds if args.rounds is not None else sim["n_rounds"]
    budget_s = (args.budget_seconds if args.budget_seconds is not None
                else sim.get("budget_seconds", 60))
    rounds = guard_seconds(rounds, budget_s, args.force)

    t_step = hw["weight_gb"] * 1e9 / (hw["bandwidth_gbps"] * 1e9)
    base_ts = 1.0 / t_step

    t0 = time.monotonic()

    print(f"\n=== 树式投机解码模拟 ===")
    print(f"硬件  : {hw['weight_gb']} GB / {hw['bandwidth_gbps']} GB/s "
          f"-> 每步 {t_step*1e3:.1f} ms，链式 n=1 上限 {base_ts:.1f} tok/s")
    print(f"草稿  : alpha={alpha} decay={decay} model={model}")
    print(f"预算  : {budget} 节点/轮，轮数 {rounds/1e6:.1f}M，seed={sim['seed']}\n")

    # 基准：链式 n
    bn = bl["chain_n"]
    bg, _, _ = simulate(alpha, decay, model, 1, bn, rounds, sim["seed"], sim["chunk"],
                        per_depth=dr.get("per_depth"))
    b_ts = bg.mean() / t_step

    print(f"\n{'width':>6} {'depth':>6} {'平均gain':>9} {'P50':>5} {'P95':>5} "
          f"{'tok/s':>8} {'vs链式':>8}")
    print("-" * 62)
    rows = []
    for w in widths:
        gains, depth, hit = simulate(alpha, decay, model, w, budget,
                                     rounds, sim["seed"], sim["chunk"],
                                     per_depth=dr.get("per_depth"))
        g = float(gains.mean())
        p50 = float(np.percentile(gains, 50))
        p95 = float(np.percentile(gains, 95))
        ts = g / t_step
        rows.append((w, depth, g, ts, ts / b_ts))
        print(f"{w:>6} {depth:>6} {g:>9.3f} {p50:>5.0f} {p95:>5.0f} "
              f"{ts:>8.1f} {ts/b_ts:>7.2f}x")

    best = max(rows, key=lambda r: r[3])
    print("-" * 62)
    print(f"基准  : 链式 n={bn} -> {b_ts:.1f} tok/s（gain {bg.mean():.3f}）")
    print(f"最优  : width={best[0]} depth={best[1]} -> {best[3]:.1f} tok/s "
          f"({best[4]:.2f}x vs 链式)")

    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    tag = f"a{alpha}_{model}_{rounds//1000}k"
    path = os.path.join(out, f"tree_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "alpha": alpha, "decay": decay, "model": model,
            "budget": budget, "rounds": rounds, "seed": sim["seed"],
            "t_step_ms": t_step * 1e3, "base_ts": base_ts,
            "baseline_chain_n": bn, "baseline_gain": float(bg.mean()),
            "baseline_ts": b_ts,
            "rows": [{"width": w, "depth": d, "gain": g, "ts": ts,
                      "speedup_vs_chain": sp} for w, d, g, ts, sp in rows],
            "best": {"width": best[0], "depth": best[1], "ts": best[3]},
        }, f, indent=2, ensure_ascii=False)
    print(f"\n耗时  : {time.monotonic() - t0:.1f} s（预算 {budget_s:.0f} s）")
    print(f"结果已写入 {path}")


if __name__ == "__main__":
    main()
