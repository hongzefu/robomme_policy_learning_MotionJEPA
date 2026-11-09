"""探索性：N≤3 的局上，三个模型 Text 组读 to_press 的逐层折外预测（PCA64、分折种子 42），相邻模型的局级 MAE 配对差（按局重采样 2000 次）。
FrameSamp、concat 取 explore/new/predictions.npz；split 取 explore/split-vs-fs/predictions.npz 的 mot 键。层 0、4、8、12、17。
用法：python three_model_N4_paired.py <hover_points.json> <explore 目录>
"""
import json, sys
from pathlib import Path
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import episode_mae
from scripts.interpret.analysis.oracle_press_prob import bootstrap

s = hp.load_samples(sys.argv[1]); E = Path(sys.argv[2])
A = np.load(E / "new/predictions.npz"); B = np.load(E / "split-vs-fs/predictions.npz")
y = s["labels"]["to_press"]; hi = s["N"] <= 3
out = {}
for l in (0, 4, 8, 12, 17):
    preds = {"FrameSamp": A[f"ctx|text|{l}|to_press"], "concat": A[f"mot|text|{l}|to_press"], "split": B[f"mot|text|{l}|to_press"]}
    e = {k: episode_mae(y[hi], v[hi], s["groups"][hi]) for k, v in preds.items()}
    row = {}
    for a, b in (("FrameSamp", "concat"), ("concat", "split"), ("FrameSamp", "split")):
        keys = sorted(set(e[a]) & set(e[b]))
        row[f"{a}-{b}"] = bootstrap([e[a][k] - e[b][k] for k in keys], 2000, 42)
    out[str(l)] = row
    print(f"L{l}: " + " | ".join(f"{k} MAE diff {v['mean']:+.1f}[{v['ci95'][0]:.1f},{v['ci95'][1]:.1f}]" for k, v in row.items()))
(E / "three_model_N3_paired.json").write_text(json.dumps(out, indent=1))
