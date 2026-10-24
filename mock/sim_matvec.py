#!/usr/bin/env python3
"""PTQ1_0 matvec 调参模拟器 —— 不编译任何 CUDA，先算出该不该做、能做到多少。

回答两个问题，顺序不能反：
  1. 值不值得：列数 n 的收益上限是多少（只看 acceptance α，与 kernel 无关）
  2. 做不做得成：寄存器 / SMEM / occupancy 在哪一档崩

收益模型（Leviathan et al. arXiv:2211.17192 Eq.1）：
    chain: e(n) = 1 + a1 + a1*a2 + ...     一次权重读取最多服务 n+1 个 token
    tree : 每层 width 个候选，该层至少命中一个的概率 1-(1-p)^width，
           深度 depth = budget/width
带宽是硬墙：
    t_step = W_bytes / BW
    tok/s  = e(n) / t_step
"""
import argparse

W_BYTES = 5.79e9
BW = 272e9            # RTX 4060 GDDR6
SM_REGS = 65536       # 每 SM 32-bit 寄存器
SM_SMEM = 100 * 1024  # Ada 每 SM SMEM（默认上限）
SM_THREADS = 1536     # Ada 每 SM thread
MAX_REGS = 255

REG_BASE = 34         # 索引/地址/循环/dp4a 中间量
REG_UNPACK = 12       # 解包缓冲（warp 内 128 trit -> 32 B 原始 -> 32 个 uint32 视图）


def chain_gain(n, a, decay=0.0):
    total, p = 1.0, 1.0
    for k in range(n):
        p *= max(a - decay * k, 0.0)
        total += p
    return total


def tree_gain(budget, width, p):
    depth = max(1, budget // width)
    hit = 1.0 - (1.0 - p) ** width
    total, q = 1.0, 1.0
    for _ in range(depth):
        q *= hit
        total += q
    return total, depth, hit


def regs_for(ncols, nrows, scheme):
    if scheme == "smem_partials":
        return REG_BASE + REG_UNPACK + 4
    # warp-per-row：每 thread 持 ncols 个 float 累加器 + ncols 个激活 int32
    return REG_BASE + REG_UNPACK + ncols * 2 + max(0, nrows // 8)


def smem_for(ncols, nrows, bprp, scheme, gate=True):
    x2 = 2 if gate else 1
    if scheme == "smem_partials":
        return ncols * nrows * (bprp + 1) * 4 * x2
    return (32 + 4 * 32 + nrows * 4) * x2


def feasible(ncols, nrows, threads, bprp):
    """返回 (regs, smem_bytes, blocks_per_sm, util)。"""
    regs = regs_for(ncols, nrows, "warp_per_row")
    smem = smem_for(ncols, nrows, bprp, "warp_per_row")
    if regs > MAX_REGS or smem > SM_SMEM:
        return regs, smem, 0, 0.0
    blocks = min(SM_REGS // (threads * regs), SM_THREADS // threads, SM_SMEM // smem)
    util = 1.0 if blocks >= 6 else (0.9 if blocks >= 4 else (0.75 if blocks >= 2 else 0.4))
    return regs, smem, blocks, util


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.60)
    ap.add_argument("--decay", type=float, default=0.06)
    ap.add_argument("--bprp", type=int, default=4)
    ap.add_argument("--threads", type=int, default=256)
    args = ap.parse_args()

    t_step = W_BYTES / BW          # 秒
    base_ts = 1.0 / t_step
    a = args.alpha
    g5 = chain_gain(5, a, args.decay)

    print(f"权重 {W_BYTES/1e9:.2f} GB / 带宽 {BW/1e9:.0f} GB/s "
          f"-> 每步 {t_step*1e3:.1f} ms，纯带宽上限 {base_ts:.1f} tok/s")
    print(f"alpha(首层)={a}  逐层衰减={args.decay}  "
          f"n->inf 极限 gain={1/(1-a) if a < 1 else float('inf'):.2f}\n")

    print("== 1. 收益天花板（只看 acceptance，与 kernel 无关）==")
    print(f"{'n':>3} {'gain':>6} {'tok/s':>8} {'vs n=1':>8} {'vs n=5':>8}")
    for n in (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48):
        g = chain_gain(n, a, args.decay)
        print(f"{n:>3} {g:>6.2f} {g/t_step:>8.1f} {g/chain_gain(1,a):>7.2f}x {g/g5:>7.2f}x")

    print("\n== 2. 可行性（warp-per-row，nrows=8）==")
    print(f"{'n':>3} {'regs':>5} {'smem':>7} {'blk/SM':>7} {'util':>5} {'tok/s':>8}  判")
    best = None
    for n in range(1, 49):
        regs, smem, blocks, util = feasible(n, 8, args.threads, args.bprp)
        g = chain_gain(n, a, args.decay)
        ts = g / t_step * util
        ok = blocks > 0
        if n in (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 40, 48) or not ok:
            print(f"{n:>3} {regs:>5} {smem/1024:>6.1f}K {blocks:>7} "
                  f"{util:>5.2f} {ts:>8.1f}  {'ok' if ok else '溢出'}")
        if ok and (best is None or ts > best[1]):
            best = (n, ts, regs, blocks, g)

    print("\n== 3. 同样预算改走 tree ==")
    print(f"{'width':>6} {'depth':>6} {'每层命中':>9} {'gain':>6} {'tok/s':>8} {'vs chain n=5':>13}")
    best_tree = None
    for w in (1, 2, 3, 4, 6, 8):
        budget = 48
        tg, depth, hit = tree_gain(budget, w, a)
        util = feasible(budget, 8, args.threads, args.bprp)[3]
        ts = tg / t_step * util
        print(f"{w:>6} {depth:>6} {hit:>9.3f} {tg:>6.2f} {ts:>8.1f} {tg/g5:>12.2f}x")
        if best_tree is None or ts > best_tree[1]:
            best_tree = (w, ts, tg)

    print("\n== 结论 ==")
    g48 = chain_gain(48, a, args.decay)
    if best:
        print(f"链式最优 n={best[0]} -> {best[1]:.1f} tok/s "
              f"(regs={best[2]}, blk/SM={best[3]})")
    if best_tree:
        print(f"树式最优 width={best_tree[0]} -> {best_tree[1]:.1f} tok/s "
              f"(gain {best_tree[2]:.2f})")
    print(f"\nn=5 已拿到极限收益的 {g5/(1/(1-a))*100:.0f}%；"
          f"n=48 相对 n=5 只有 {(g48/g5-1)*100:+.1f}%")
    if g48 / g5 < 1.15:
        print("-> 链式加宽在 alpha<=0.6 时是白做：n>5 之后每多一列只多 accept 一个，"
              "而 gain 已经收敛到 1/(1-alpha)。")
        if best_tree and best_tree[1] > best[1] * 1.15:
            print(f"-> 但同样的 48 列预算改走 tree，能到 {best_tree[1]:.1f} tok/s，"
                  f"因为每层 width 个候选把该层命中率抬到 {tree_gain(48, best_tree[0], a)[2]:.3f}。")
        else:
            print("-> tree 在这个 alpha 下也没有明显优势，宽列这条路整体不划算。")


if __name__ == "__main__":
    main()
