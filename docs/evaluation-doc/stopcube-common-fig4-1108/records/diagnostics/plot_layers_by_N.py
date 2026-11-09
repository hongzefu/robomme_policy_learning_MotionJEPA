"""探索性：Fig. 4 同款逐层 R²（to_press），按 N≤3／N≥4 分开评估（探针在全部点上训练，只在评估时分档）。
用法：python plot_layers_by_N.py <explore 输出目录> <png>   （需先跑 layers_by_N.py）
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, png = Path(sys.argv[1]), sys.argv[2]
R = json.loads((out / "layers_by_N.json").read_text())
groups = [("text", "Text", "#eb6834"), ("current", "Current frame", "#1baf7a"), ("action", "Action token", "#2a78d6")]
models = [("ctx", "FrameSamp", "--"), ("mot", "MotionMem", "-")]
tags = [("N<=3", "Low memory demand (N = 2, 3)"), ("N>=4", "High memory demand (N = 4, 5)")]
x = list(range(-1, 18)); labels = ["In"] + [str(i) for i in range(18)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for ax, (tag, title) in zip(axes, tags):
    for g, gname, color in groups:
        for m, mname, ls in models:
            pts = [(l, R[tag][f"{m}|{g}|{l}"][0]) for l in x if f"{m}|{g}|{l}" in R[tag]]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=color, lw=1.9 if m == "mot" else 1.3,
                    label=f"{mname} · {gname}")
    ax.axhline(R[tag]["ref|t_N"][0], color="#444444", ls=":", lw=1.2, label="Oracle [t, N]")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Layer"); ax.grid(alpha=0.25)
    ax.set_ylim(-1.4, 0.8)
axes[0].set_ylabel("Out-of-fold R² (steps to press)")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
fig.suptitle("StopCube, common official demos: layer-wise probe split by memory demand  [exploratory]", fontsize=11)
fig.tight_layout(rect=(0, 0.13, 1, 0.94))
fig.savefig(png, dpi=150)
print("PLOT_DONE", png)
