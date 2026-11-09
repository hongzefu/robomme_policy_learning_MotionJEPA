"""探索性：共同示教轨迹上 MotionMem − FrameSamp 的配对差（同组、训练折内选层、按局重采样 95% 区间）总览。
用法：python plot_angle_forest.py <angles.json> <png>
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = json.load(open(sys.argv[1]))["cross_model"]
reads = [("to_press", "Steps to press (−MAE, steps)", True), ("M", "Cube speed M (acc.)", False),
         ("c", "Passes elapsed (acc.)", False), ("c_strict", "Passes elapsed, look-alike frame pairs (acc.)", False),
         ("press", "Press now? (acc.)", False), ("remaining", "Passes remaining (acc.)", False),
         ("cube_pos", "Cube position (−MAE)", True)]
groups = [("text", "Text", "#eb6834"), ("current", "Current frame", "#1baf7a"), ("action", "Action token", "#2a78d6"),
          ("history", "History frames", "#888888")]
groups = [g for g in groups if g[0] in A]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 2.4]}, sharey=False)
for ax, subset in ((axes[0], [r for r in reads if r[0] == "to_press"]), (axes[1], [r for r in reads if r[0] != "to_press"])):
    ylabels = []
    for i, (r, name, _) in enumerate(subset):
        ylabels.append(name)
        for j, (g, gname, color) in enumerate(groups):
            if r not in A[g]:
                continue
            d = A[g][r]["mot_minus_ctx"]
            y = i + (j - (len(groups) - 1) / 2) * 0.18
            ax.errorbar(d["mean"], y, xerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]], fmt="o", color=color,
                        ms=4, capsize=2, lw=1.2, label=gname if i == 0 else None)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(subset))); ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_ylim(len(subset) - 0.5, -0.5); ax.grid(axis="x", alpha=0.25)
axes[0].set_xlabel("MotionMem − FrameSamp (steps; > 0 = MotionMem better)")
axes[1].set_xlabel("MotionMem − FrameSamp (accuracy / −MAE; > 0 = MotionMem better)")
axes[1].legend(fontsize=8, frameon=False, loc="lower right")
fig.suptitle("StopCube, common official demos (98 eps): paired model difference per readout, episode-bootstrap 95% CI  [exploratory]",
             fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(sys.argv[2], dpi=150)
print("PLOT_DONE", sys.argv[2])
