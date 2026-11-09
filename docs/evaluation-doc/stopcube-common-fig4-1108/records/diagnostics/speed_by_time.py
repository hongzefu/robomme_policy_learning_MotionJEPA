"""探索性：读速度 M 的准确率按时间步 t 分档（训练折内选层的折外预测，同 explore_angles），两模型 Text、History 组。
每档每局等权；同档内跨模型配对差按局重采样。用法：python speed_by_time.py <hover_points.json> <explore/new 目录>
"""
import json, sys
from pathlib import Path
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.oracle_press_prob import bootstrap

s = hp.load_samples(sys.argv[1]); src = Path(sys.argv[2])
P = np.load(src / "predictions.npz"); A = json.loads((src / "angles.json").read_text())
def comp(m, g):
    ch = A["selected"][f"{m}|{g}"]["M"]["layer_by_fold"]; r = np.full(len(s["t"]), np.nan)
    for f, l in enumerate(ch):
        r[s["outer"] == f] = P[f"{m}|{g}|{l}|M"][s["outer"] == f]
    return r
bins = [(0, 80), (80, 120), (120, 160), (160, 200), (200, 1000)]
out = {}
for g in ("text", "history"):
    pm, pc = comp("mot", g), comp("ctx", g)
    for lo, hi in bins:
        mask = (s["t"] >= lo) & (s["t"] < hi)
        am = hp.readout_episodes(s, "M", np.where(mask, pm, np.nan), s["labels"]["M"])
        ac = hp.readout_episodes(s, "M", np.where(mask, pc, np.nan), s["labels"]["M"])
        k = np.isfinite(am) & np.isfinite(ac); d = bootstrap(am[k] - ac[k], 2000, 42)
        out[f"{g}|{lo}-{hi}"] = {"mot": float(np.nanmean(am[k])), "ctx": float(np.nanmean(ac[k])), "diff": d, "n_eps": int(k.sum()), "n_pts": int(mask.sum())}
        print(f"{g} t∈[{lo},{hi}): pts={mask.sum()} eps={k.sum()} mot={np.nanmean(am[k]):.2f} ctx={np.nanmean(ac[k]):.2f} diff={d['mean']:+.3f}[{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}]")
(src / "speed_by_time.json").write_text(json.dumps(out, indent=1))
