#!/usr/bin/env python3
"""从 server.log 标定真实的草稿接受率，输出 sim_tree.py 能用的参数。

日志里每轮有一行：
    draft acceptance = 0.44737 (   17 accepted /    38 generated), mean len =  1.77

accepted/generated 是整轮的，不是每层的。要还原逐层命中率，用递推：
    若一轮开 n 个草稿、接受 a 个，则第 k 层通过的次数 ≈ a_k
    单候选命中率 p_k ≈ a_k / n_k
样本少时只能估计总体 alpha，层间衰减取默认值。

用法：
    python mock/calibrate.py --log src/logs/server.log
    python mock/calibrate.py --log src/logs/server.log --write   # 更新 params.json
"""
import argparse
import json
import os
import re
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("需要 numpy：用 C:/Users/bestd/.workbuddy/binaries/python/envs/default/Scripts/python.exe 跑")

HERE = os.path.dirname(os.path.abspath(__file__))
RE = re.compile(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), "
                r"mean len =\s*([\d.]+)")
# 逐层存活曲线（server-context.cpp 以 INFO 打印）。括号里是 S(i) = P(len > i)，
# 因为底层计数器是累积的：接受 k 个 token 会把 0..k-1 全部 +1。
# 所以第 i 层的边际单候选命中率 = S(i)/S(i-1)，S(-1) 记为 1。
RE_POS = re.compile(r"acc per pos = \(([^)]*)\), steps = (\d+)")


def parse(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE.search(line)
            if m:
                rows.append({
                    "rate": float(m.group(1)),
                    "accepted": int(m.group(2)),
                    "generated": int(m.group(3)),
                    "mean_len": float(m.group(4)),
                })
    return rows


def parse_pos(path):
    """回收逐层存活曲线，按 steps 加权累加。"""
    acc = None
    steps_all = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_POS.search(line)
            if not m:
                continue
            rates = [float(x) for x in m.group(1).split(",") if x.strip()]
            n = int(m.group(2))
            if acc is None:
                acc = [0.0] * len(rates)
            elif len(rates) > len(acc):
                acc += [0.0] * (len(rates) - len(acc))
            for i, r in enumerate(rates):
                acc[i] += r * n
            steps_all += n
    if acc is None or steps_all == 0:
        return None
    return [a / steps_all for a in acc], steps_all


def diagnose(S):
    """由存活曲线判断候选分布形状。

    returns dict: marginals / ratios / 两种形状假设的拟合误差
    """
    if not S:
        return None
    marg = []
    prev = 1.0
    for s in S:
        marg.append(s / prev if prev > 0 else 0.0)
        prev = s
    # 边际比：geometric 假设下应恒等于 alpha（比值全 1）
    ratios = [marg[i + 1] / marg[i] if marg[i] > 0 else 0.0
              for i in range(len(marg) - 1)]

    idx = np.arange(len(S), dtype=np.float64)
    obs = np.asarray(S, dtype=np.float64)
    out = {"marginals": marg, "ratios": ratios}

    # H1 geometric: S(i) = a^(i+1)  -> log S 对 i 线性
    y = np.log(np.clip(obs, 1e-9, None))
    A = np.vstack([idx, np.ones_like(idx)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = np.exp(A @ coef)
    out["geo_alpha"] = float(np.exp(coef[0]))
    out["geo_rmse"] = float(np.sqrt(np.mean((pred - obs) ** 2)))

    # H2 power-law: S(i) = S0 * (i+1)^(-k) -> log S 对 log(i+1) 线性
    x = np.log(idx + 1.0)
    A2 = np.vstack([x, np.ones_like(x)]).T
    coef2, *_ = np.linalg.lstsq(A2, y, rcond=None)
    pred2 = np.exp(A2 @ coef2)
    out["pl_k"] = float(-coef2[0])
    out["pl_rmse"] = float(np.sqrt(np.mean((pred2 - obs) ** 2)))

    # 判定的核心不是"哪个族拟合更好"，而是**边际是否恒定**：
    # geometric 的充要条件就是 p_i 不随深度变（边际比全 1）。
    # 边际明显衰减时两个族都不对，此时唯一诚实的做法是用实测 empirical，
    # 而不是拿 zipf 去硬套——zipf 的 s 本来也是拍的。
    out["ratio_mean"] = float(np.mean(ratios)) if ratios else 1.0
    out["ratio_min"] = float(min(ratios)) if ratios else 1.0
    out["flat"] = out["ratio_mean"] >= 0.92
    out["shape"] = "geometric" if out["flat"] else "long-tailed(use empirical)"
    return out


def summarize(rows):
    if not rows:
        return None
    n = len(rows)
    acc = sum(r["accepted"] for r in rows)
    gen = sum(r["generated"] for r in rows)
    rates = sorted(r["rate"] for r in rows)
    mean_len = sum(r["mean_len"] for r in rows) / n
    # 由 mean len 反推首层 alpha：e = 1/(1-a) 是链式极限，
    # 但一轮只有 n_draft 个，用 e = 1 + a + a^2 + ... 到 n_draft 项
    # 反解 alpha 不便，直接给两个候选估计
    alpha_from_len = 1.0 - 1.0 / mean_len if mean_len > 1 else 0.0
    alpha_from_rate = acc / gen if gen else 0.0
    return {
        "rounds": n,
        "accepted": acc,
        "generated": gen,
        "rate_all": acc / gen if gen else 0.0,
        "rate_median": rates[n // 2],
        "rate_p10": rates[max(0, n // 10)],
        "rate_p90": rates[min(n - 1, n * 9 // 10)],
        "mean_len": mean_len,
        "alpha_from_mean_len": alpha_from_len,
        "alpha_from_rate": alpha_from_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(HERE, "..", "src", "logs", "server.log"))
    ap.add_argument("--write", action="store_true", help="写回 params.json 的 draft.alpha")
    args = ap.parse_args()

    path = os.path.abspath(args.log)
    if not os.path.exists(path):
        sys.exit(f"找不到 {path}")
    rows = parse(path)
    s = summarize(rows)
    if not s:
        sys.exit("日志里没有 'draft acceptance' 行（需要 --verbosity >= 2 且发生过投机解码）")

    print(f"日志: {path}")
    print(f"轮数: {s['rounds']}")
    print(f"总接受: {s['accepted']}/{s['generated']} = {s['rate_all']:.4f}")
    print(f"逐轮接受率: P10={s['rate_p10']:.3f} 中位={s['rate_median']:.3f} "
          f"P90={s['rate_p90']:.3f}")
    print(f"mean len: {s['mean_len']:.2f}")
    print(f"\nalpha 估计:")
    print(f"  accepted/generated   -> {s['alpha_from_rate']:.3f}（整轮平均，偏低）")
    print(f"  1 - 1/mean_len       -> {s['alpha_from_mean_len']:.3f}（链式极限反推）")

    if s["rounds"] < 30:
        print(f"\n!! 只有 {s['rounds']} 轮，样本太少，标定不可靠。"
              f"需要至少几百轮（一条长回答大约 50-200 轮）。")

    # ---- 逐层存活曲线：唯一能区分 geometric / zipf 的证据 ----
    pos = parse_pos(path)
    diag = None
    if pos is None:
        print("\n逐层接受率: 日志里没有 'acc per pos' 行。")
        print("  这条线在 server-context.cpp 里已改为 INFO 打印，需要重编后才会出现；")
        print("  在此之前只能用上面的 mean len 反推 alpha，形状仍是假设。")
    else:
        S, steps_all = pos
        diag = diagnose(S)
        print(f"\n逐层存活 S(i)=P(len>i)（{steps_all} 次验证步）:")
        for i, sv in enumerate(S):
            mr = diag["marginals"][i]
            print(f"  i={i}  S={sv:.4f}  边际 p={mr:.4f}")
        print(f"  边际比 p[i+1]/p[i] = "
              + ", ".join(f"{r:.3f}" for r in diag["ratios"]))
        print(f"\n形状判定（看边际是否恒定，不看哪个族拟合更好）:")
        print(f"  边际比均值 {diag['ratio_mean']:.3f}，最小 {diag['ratio_min']:.3f}")
        print(f"  geometric : alpha={diag['geo_alpha']:.3f}  RMSE={diag['geo_rmse']:.4f}")
        print(f"  power-law : k={diag['pl_k']:.3f}       RMSE={diag['pl_rmse']:.4f}")
        if diag["flat"]:
            print(f"  -> 边际基本恒定，geometric 假设成立（alpha≈{diag['marginals'][0]:.3f}）")
        else:
            print(f"  -> 边际随深度衰减（比值 {diag['ratio_mean']:.2f}），")
            print(f"     两个族都不对，必须用 empirical 实测序列，不要拿 zipf 硬套")

    if args.write:
        p = os.path.join(HERE, "params.json")
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        if diag is not None:
            # 实测形状优先：直接喂逐层边际，不再假设 geometric/zipf
            cfg["draft"]["model"] = "empirical"
            cfg["draft"]["alpha"] = round(diag["marginals"][0], 3)
            cfg["draft"]["per_depth"] = [round(x, 4) for x in diag["marginals"]]
        else:
            cfg["draft"]["alpha"] = round(s["alpha_from_mean_len"], 3)
        cfg["draft"]["_calibrated_from"] = {
            "log": os.path.relpath(path),
            "rounds": s["rounds"],
            "mean_len": round(s["mean_len"], 3),
            "shape": diag["shape"] if diag else None,
            "marginal_ratio_mean": round(diag["ratio_mean"], 3) if diag else None,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n已写入 {p}（model={cfg['draft']['model']}, "
              f"alpha={cfg['draft']['alpha']}）")
    else:
        print("\n加 --write 把结果写回 params.json")


if __name__ == "__main__":
    main()
