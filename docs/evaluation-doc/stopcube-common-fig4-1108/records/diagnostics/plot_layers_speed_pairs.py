"""探索性：逐层读速度 M（准确率）与画面几乎相同配对点的已过次数（配对准确率），两模型、Text／Current／Action 组，共同示教轨迹。
数据：explore_angles 的 angles.json（layers 字段，全折外、每局等权）。
用法：python plot_layers_speed_pairs.py <angles.json> <png>
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = json.load(open(sys.argv[1]))["layers"]
groups = [("text", "Text", "#eb6834"), ("current", "Current frame", "#1baf7a"), ("action", "Action token", "#2a78d6")]
models = [("ctx", "FrameSamp", "--", 1.3), ("mot", "MotionMem", "-", 1.9)]
panels = [("M", "Cube speed M (accuracy; majority class ≈ 0.42 by episode)"), ("c_strict", "Passes elapsed on look-alike frame pairs (both points right)")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, (r, title) in zip(axes, panels):
    for g, gname, color in groups:
        for m, mname, ls, lw in models:
            L = A.get(f"{m}|{g}")
            if not L:
                continue
            xs = sorted(int(k) for k in L)
            ax.plot(xs, [L[str(x)][r]["acc"] for x in xs], ls, color=color, lw=lw, label=f"{mname} · {gname}")
    ax.set_title(title, fontsize=10); ax.set_xlabel("Layer (−1 = input)"); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    ax.set_xticks(range(-1, 18, 2))
axes[0].set_ylabel("Out-of-fold accuracy")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.suptitle("StopCube, common official demos (98 eps, 2344 pts): layer-wise decoding  [exploratory]", fontsize=11)
fig.tight_layout(rect=(0, 0.12, 1, 0.93)); fig.savefig(sys.argv[2], dpi=150); print("PLOT_DONE")
