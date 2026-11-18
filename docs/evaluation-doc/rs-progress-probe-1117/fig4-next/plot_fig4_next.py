"""新版 Fig. 4：只画「下一步走法」(next move) 的线性解码准确率，对比 A=MotionMem 与 B=Frozen Wan pooling。

数值全部从 rsprog-probe-summary.json 读取，不硬编码。
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY = HERE.parent / "records" / "rsprog-probe-summary.json"
DEFAULT_OUT = HERE

# Okabe-Ito 色盲友好配色：A 蓝，B 灰，前缀对照黑色点线，时间基线朱红点划线
C_A = "#0072B2"
C_B = "#8C8C8C"
C_PRE = "#222222"
C_TIME = "#D55E00"
LS_ACT = (0, (4, 2))
LS_PRE = (0, (1, 1.5))
LS_TIME = (0, (5, 2, 1, 2))
NAME = {"A": "MotionMem (ours)", "B": "Frozen Wan pooling"}
COLOR = {"A": C_A, "B": C_B}
DIFFS = ["easy", "medium", "hard", "xhard"]


def curve_arrays(entry):
    c = entry["curve"]
    mean = np.array([p["mean"] for p in c])
    lo = np.array([p["ci95"][0] for p in c])
    hi = np.array([p["ci95"][1] for p in c])
    return np.arange(len(c)), mean, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    args = ap.parse_args()

    with open(args.summary) as f:
        d = json.load(f)
    M = d["models"]
    t_base = d["time_baseline"]["u"]["next"]["mean"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.45, 1.0]})

    # ---------- (a) 逐层曲线 ----------
    sel = {}
    for m in ("B", "A"):  # A 后画，压在上面
        e = M[m]["memmod"]["next"]
        x, mu, lo, hi = curve_arrays(e)
        ax.fill_between(x, lo, hi, color=COLOR[m], alpha=0.22, lw=0, zorder=2 if m == "B" else 4)
        ax.plot(x, mu, color=COLOR[m], lw=2.2, zorder=3 if m == "B" else 5)
        xa, mua, _, _ = curve_arrays(M[m]["action"]["next"])
        ax.plot(xa, mua, color=COLOR[m], lw=1.1, ls=LS_ACT, zorder=3)
        layer = Counter(e["layer_by_fold"]).most_common(1)[0][0]
        sel[m] = (layer, e["selected"]["mean"], hi[layer])
        ax.plot([layer], [mu[layer]], marker="o", ms=5, mfc="white", mec=COLOR[m], mew=1.6, zorder=6)

    # 前缀（无记忆）对照：A、B 两条曲线取平均合并成一条（两者 selected 均为 0.38）
    xp, pa, _, _ = curve_arrays(M["A"]["prefix_cat"]["next"])
    _, pb, _, _ = curve_arrays(M["B"]["prefix_cat"]["next"])
    ax.plot(xp, (pa + pb) / 2, color=C_PRE, lw=1.0, ls=LS_PRE, zorder=2)
    ax.axhline(t_base, color=C_TIME, lw=1.0, ls=LS_TIME, zorder=1)

    ax.set_xlim(-0.4, 18.4)
    ax.set_ylim(0.2, 0.86)
    ax.set_xticks(range(0, 19, 3))
    ax.set_yticks(np.arange(0.2, 0.81, 0.1))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Next-move decoding accuracy")

    la, sa, ya = sel["A"]
    lb, sb, yb = sel["B"]
    ax.text(la, ya + 0.012, f"{sa:.2f}", ha="center", va="bottom", color=C_A, fontweight="bold", fontsize=8.5)
    ax.text(lb, yb + 0.012, f"{sb:.2f}", ha="center", va="bottom", color="#5A5A5A", fontweight="bold", fontsize=8.5)

    ha = [Line2D([], [], color=C_A, lw=2.2, label=NAME["A"]),
          Line2D([], [], color=C_B, lw=2.2, label=NAME["B"]),
          Line2D([], [], color="#444444", lw=2.2, label="memory read-out (95% CI band)"),
          Line2D([], [], color="#444444", lw=1.1, ls=LS_ACT, label="action tokens"),
          Line2D([], [], color=C_PRE, lw=1.0, ls=LS_PRE, label="prefix (no memory)"),
          Line2D([], [], color=C_TIME, lw=1.0, ls=LS_TIME, label="time-only")]

    # ---------- (b) 分难度柱状图 ----------
    w = 0.36
    diffs_out = {}
    for i, dk in enumerate(DIFFS):
        vals = {}
        for m, off in (("A", -w / 2), ("B", w / 2)):
            v = M[m]["memmod"]["next"]["by_difficulty"][dk]["value"]
            vals[m] = v
            bx.bar(i + off, v["mean"], w, color=COLOR[m], edgecolor="white", lw=0.5, zorder=2)
            bx.errorbar(i + off, v["mean"], yerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]],
                        color="#222222", lw=1.0, capsize=2.5, capthick=1.0, zorder=3)
        u = M["A"]["memmod"]["next"]["by_difficulty"][dk]["u"]["mean"]
        bx.hlines(u, i - w - 0.04, i + w + 0.04, color=C_TIME, lw=1.2, ls=LS_TIME, zorder=4)
        diff = vals["A"]["mean"] - vals["B"]["mean"]
        diffs_out[dk] = diff
        separated = vals["A"]["ci95"][0] > vals["B"]["ci95"][1]  # 只用「CI 不重叠」这一事实
        top = max(vals["A"]["ci95"][1], vals["B"]["ci95"][1])
        bx.text(i, top + 0.025, f"{diff:+.2f}", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold" if separated else "normal", color=C_A if separated else "#777777")
    bx.set_xticks(range(len(DIFFS)))
    bx.set_xticklabels(DIFFS)
    bx.set_xlim(-0.6, len(DIFFS) - 0.4)
    bx.set_ylim(0, 1.0)
    bx.set_yticks(np.arange(0, 1.01, 0.2))
    bx.set_xlabel("Episode difficulty")
    bx.set_ylabel("Accuracy (memory read-out)")
    # 两个面板共用一份图例（颜色与时间基线线型在 (b) 中含义相同）
    fig.legend(handles=ha, loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=3, frameon=False,
               handlelength=2.2, columnspacing=2.0, labelspacing=0.3, borderaxespad=0)

    fig.subplots_adjust(left=0.068, right=0.995, bottom=0.15, top=0.81, wspace=0.24)
    fig.text(0.004, 0.86, "(a)", fontsize=9.5, fontweight="bold", va="top", ha="left")
    fig.text(bx.get_position().x0 - 0.088, 0.86, "(b)", fontsize=9.5, fontweight="bold", va="top", ha="left")
    fig.savefig(out / "fig4-next.png", dpi=200)
    fig.savefig(out / "fig4-next.pdf")

    a_sel = M["A"]["memmod"]["next"]["selected"]["mean"]
    b_sel = M["B"]["memmod"]["next"]["selected"]["mean"]
    print("diff_all=" + ",".join(f"{k}:{v:+.3f}" for k, v in diffs_out.items()))
    print(f"FIG4_NEXT=PASS a_sel={a_sel:.3f} b_sel={b_sel:.3f} "
          f"diffs={diffs_out['medium']:+.3f},{diffs_out['hard']:+.3f},{diffs_out['xhard']:+.3f}")


if __name__ == "__main__":
    main()
