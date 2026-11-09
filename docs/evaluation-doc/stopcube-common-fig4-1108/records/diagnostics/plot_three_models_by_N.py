"""探索性：三个模型（FrameSamp、MotionMem concat、MotionMem split）Text 组读 to_press 的逐层 R²，按 N≤3／N≥4 分档评估（探针在全部点上训练）。
数据：explore/new/layers_by_N.json（FrameSamp、concat）与 explore/split-vs-fs/layers_by_N.json（split）。用法：python plot_three_models_by_N.py <explore 目录> <png>
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = Path(sys.argv[1]); a = json.loads((E / "new/layers_by_N.json").read_text()); b = json.loads((E / "split-vs-fs/layers_by_N.json").read_text())
series = [("FrameSamp (14%)", a, "ctx", "#2a78d6", "--"), ("MotionMem concat (43%)", a, "mot", "#eb6834", "-"), ("MotionMem split (66%)", b, "mot", "#7a2e0e", "-")]
x = list(range(-1, 18))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, (tag, title) in zip(axes, (("N<=3", "Low memory demand (N = 2, 3)"), ("N>=4", "High memory demand (N = 4, 5)"))):
    for name, R, m, c, ls in series:
        ax.plot(x, [R[tag][f"{m}|text|{l}"][0] for l in x], ls, color=c, lw=1.9, label=name)
    ax.axhline(a[tag]["ref|t_N"][0], color="#555555", ls=":", lw=1.2, label="Oracle [t, N]")
    ax.set_title(title, fontsize=11); ax.set_xlabel("Layer (−1 = input)"); ax.set_xticks(range(-1, 18, 2)); ax.grid(alpha=0.25)
    ax.set_ylim(-1.3, 0.9)
axes[0].set_ylabel("Out-of-fold R² (steps to press), Text tokens")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
fig.suptitle("StopCube, common official demos: three policies (StopCube success in legend)  [exploratory]", fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 0.93)); fig.savefig(sys.argv[2], dpi=150); print("PLOT_DONE")
