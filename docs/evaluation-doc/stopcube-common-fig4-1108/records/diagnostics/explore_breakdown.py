"""探索性：按目标次数 N、速度 M 拆开跨模型差（用 explore_angles 的折外预测与训练折内选层）。
用法：python explore_breakdown.py <hover_points.json> <explore 输出目录>
"""
import json, sys
from pathlib import Path
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.oracle_press_prob import bootstrap

points, out = sys.argv[1], Path(sys.argv[2])
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

def ep_scores(readout, pred, mask):
    truth = s["labels"][hp.READOUT_LABEL[readout]]
    return hp.readout_episodes(s, readout, np.where(mask, pred, np.nan), truth)

report = {}
for (gm, gc) in (("text", "text"), ("current", "current"), ("motion", "history")):
    for readout in ("to_press", "M", "c", "c_strict"):
        a, b = composed("mot", gm, readout), composed("ctx", gc, readout)
        for key, values in (("N", s["N"]), ("M", s["M"])):
            for v in np.unique(values):
                mask = values == v
                ea, eb = ep_scores(readout, a, mask), ep_scores(readout, b, mask)
                keep = np.isfinite(ea) & np.isfinite(eb)
                if keep.sum() < 3:
                    continue
                r = bootstrap(ea[keep] - eb[keep], 2000, 42)
                report.setdefault(f"mot_{gm}-ctx_{gc}|{readout}", {})[f"{key}={int(v)}"] = {
                    "mot": float(np.nanmean(ea[keep])), "ctx": float(np.nanmean(eb[keep])),
                    "diff": r["mean"], "ci95": r["ci95"], "n_episodes": int(keep.sum())}
(out / "breakdown.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
for k, v in report.items():
    print(k)
    for kk, x in v.items():
        print(f"  {kk}: mot={x['mot']:.3f} ctx={x['ctx']:.3f} diff={x['diff']:.3f} ci=[{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}] n={x['n_episodes']}")
