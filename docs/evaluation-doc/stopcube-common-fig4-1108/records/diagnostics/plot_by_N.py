"""探索性：共同示教轨迹上，Text 组探针读数按目标次数 N 拆开（训练折内选层，按局等权，局级 95% 区间）。
用法：python plot_by_N.py <hover_points.json> <explore 输出目录> <png>
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.oracle_press_prob import bootstrap

points, out, png = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
s = hp.load_samples(points)
P = np.load(out / "predictions.npz")
A = json.loads((out / "angles.json").read_text())

def composed(model, group, readout):
    lab = hp.READOUT_LABEL[readout]
    chosen = A["selected"][f"{model}|{group}"][readout]["layer_by_fold"]
    res = np.full(len(s["t"]), np.nan)
    for fold, layer in enumerate(chosen):
        part = s["outer"] == fold
        res[part] = P[f"{model}|{group}|{layer}|{lab}"][part]
    return res

def ep(readout, pred, mask):
    truth = s["labels"][hp.READOUT_LABEL[readout]]
    v = hp.readout_episodes(s, readout, np.where(mask, pred, np.nan), truth)
    if readout in hp.REGRESSION_LABELS:
        v = -v
    return v[np.isfinite(v)]

Ns = [2, 3, 4, 5]
panels = [("to_press", "Steps to press: MAE (steps, lower better)"), ("M", "Cube speed M: accuracy"),
          ("c", "Passes elapsed: accuracy")]
series = [("mot", "MotionMem Text", "#eb6834", "-o"), ("ctx", "FrameSamp Text", "#2a78d6", "-s")]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
dump = {}
for ax, (readout, title) in zip(axes, panels):
    for model, name, color, style in series:
        pred = composed(model, "text", readout)
        m, lo, hi = [], [], []
        for N in Ns:
            b = bootstrap(ep(readout, pred, s["N"] == N), 2000, 42)
            m.append(b["mean"]); lo.append(b["ci95"][0]); hi.append(b["ci95"][1])
        ax.plot(Ns, m, style, color=color, label=name, lw=1.8, ms=5)
        ax.fill_between(Ns, lo, hi, color=color, alpha=0.15, lw=0)
        dump[f"{readout}|{model}"] = m
    ref = P[f"ref|t_N|{hp.READOUT_LABEL[readout]}"]
    rm = [float(np.mean(ep(readout, ref, s["N"] == N))) for N in Ns]
    ax.plot(Ns, rm, "--", color="#555555", lw=1.1, label="Oracle [t, N]")
    dump[f"{readout}|oracle"] = rm
    ax.set_title(title, fontsize=10)
    ax.set_xticks(Ns)
    ax.set_xlabel("Target pass count N (memory demand)")
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("StopCube, common official demos (98 eps, 2344 pts): probe on Text tokens, split by N  [exploratory]", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(png, dpi=150)
Path(png).with_suffix(".json").write_text(json.dumps(dump, indent=1))
print("PLOT_DONE", png)
