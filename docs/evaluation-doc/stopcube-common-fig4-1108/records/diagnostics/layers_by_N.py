"""探索性：to_press 逐层折外读数按 N 分档（N≤3 与 N≥4），每局等权 R²／MAE，用 explore_angles 的全折外预测。
用法：python layers_by_N.py <hover_points.json> <explore 输出目录>
"""
import json, sys
from pathlib import Path
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import metrics

s = hp.load_samples(sys.argv[1]); out = Path(sys.argv[2]); P = np.load(out / "predictions.npz")
y = s["labels"]["to_press"]
res = {}
for tag, mask in (("N<=3", s["N"] <= 3), ("N>=4", s["N"] >= 4), ("all", np.ones(len(y), bool))):
    for key in P.files:
        parts = key.split("|")
        if parts[-1] != "to_press":
            continue
        pred = np.where(mask, P[key], np.nan)
        m = metrics(y, pred, s["groups"])
        res.setdefault(tag, {})["|".join(parts[:-1])] = (m["r2"], m["mae"])
(out / "layers_by_N.json").write_text(json.dumps(res, indent=1))
for tag in res:
    print("==", tag, "oracle", [round(v, 3) for v in res[tag]["ref|t_N"]], "time", [round(v, 3) for v in res[tag]["ref|t"]])
    for m, g in (("ctx", "text"), ("mot", "text"), ("ctx", "current"), ("mot", "current")):
        r = [res[tag][f"{m}|{g}|{l}"] for l in range(-1, 18)]
        print(f"  {m} {g} r2:", " ".join(f"{a:.2f}" for a, _ in r))
        print(f"  {m} {g} mae:", " ".join(f"{b:.0f}" for _, b in r))
