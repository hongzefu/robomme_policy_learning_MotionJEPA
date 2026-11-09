"""把探索性结果导出成稿件候选图用的 CSV（同列名约定、模型名 baseline／motion、组名同 Fig. 4）。
用法：python export_candidates.py <hover_points.json> <explore/new 目录> <输出目录>
产物：beyond_oracle_by_layer.csv、by_N.csv、oracle_variants.csv、paired_diff.csv
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.oracle_press_prob import bootstrap
from scripts.interpret.analysis.rollout_probe import metrics

pts, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
MODEL = {"ctx": "baseline", "mot": "motion"}
GROUP = {"text": "Text", "current": "Current frame", "action": "Action token", "history": "History frames", "motion": "Motion"}
s = hp.load_samples(pts)

B = json.loads((src / "beyond_oracle.json").read_text())
with open(out / "beyond_oracle_by_layer.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "group", "layer", "layer_index", "r2", "mae_steps", "mae_gain_steps", "gain_ci_low", "gain_ci_high"])
    for key, c in B["cells"].items():
        m, g, l = key.split("|"); l = int(l)
        w.writerow([MODEL[m], GROUP[g], "input" if l == -1 else l, l, c["r2"], c["mae"], c["mae_gain"], *c["mae_gain_ci95"]])

oh = hp._one_hot; y = s["labels"]["to_press"]
variants = {"time_only": [oh(s["t"])], "time_and_target_count": [oh(s["t"]), oh(s["N"])],
            "time_target_count_speed": [oh(s["t"]), oh(s["N"]), oh(s["M"])]}
with open(out / "oracle_variants.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["reference", "inputs", "r2", "mae_steps"])
    for name, cols in variants.items():
        X = np.hstack(cols); p, _, _, _ = hp.crossfit(lambda fold: X, s, y[:, None], False)
        m = metrics(y, p[:, 0], s["groups"])
        w.writerow([name, {"time_only": "t", "time_and_target_count": "t,N", "time_target_count_speed": "t,N,M"}[name], m["r2"], m["mae"]])

P = np.load(src / "predictions.npz"); A = json.loads((src / "angles.json").read_text())
def composed(model, group, readout):
    lab = hp.READOUT_LABEL[readout]; chosen = A["selected"][f"{model}|{group}"][readout]["layer_by_fold"]
    res = np.full(len(s["t"]), np.nan)
    for fold, layer in enumerate(chosen):
        part = s["outer"] == fold; res[part] = P[f"{model}|{group}|{layer}|{lab}"][part]
    return res
def ep(readout, pred, mask):
    v = hp.readout_episodes(s, readout, np.where(mask, pred, np.nan), s["labels"][hp.READOUT_LABEL[readout]])
    return v
with open(out / "by_N.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "group", "readout", "metric", "N", "value", "ci_low", "ci_high", "n_episodes"])
    for readout, metric in (("to_press", "mae_steps"), ("M", "accuracy"), ("c", "accuracy")):
        sources = [(MODEL[m], GROUP[g], composed(m, g, readout)) for m in ("ctx", "mot") for g in ("text", "current", "action")]
        sources.append(("reference", "time_and_target_count", P[f"ref|t_N|{hp.READOUT_LABEL[readout]}"]))
        for mname, gname, pred in sources:
            for N in (2, 3, 4, 5):
                v = ep(readout, pred, s["N"] == N); v = v[np.isfinite(v)]
                if readout == "to_press": v = -v
                b = bootstrap(v, 2000, 42)
                w.writerow([mname, gname, readout, metric, N, b["mean"], *b["ci95"], len(v)])
with open(out / "paired_diff.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["group", "readout", "motion_minus_baseline", "ci_low", "ci_high", "n_episodes", "unit"])
    for g, rows in A["cross_model"].items():
        if g not in GROUP: continue
        for r, x in rows.items():
            d = x["mot_minus_ctx"]
            w.writerow([GROUP[g], r, d["mean"], *d["ci95"], d["n_episodes"], "-MAE (steps or position) for regression; accuracy otherwise; >0 = motion better"])
print("EXPORT_DONE", out)
