"""探索性：临近决策时刻的读数。只在评估时取距按下 ≤ 阈值的点（探针仍在全部点上训练，训练折内选层的折外预测，同 explore_angles）。
报两模型 Text／Current／Action 组的 MAE 与配对差（按局重采样），以及 oracle [t,N] 同子集的 MAE。阈值 48、64、96 步。
用法：python near_decision.py <hover_points.json> <explore/new 目录>
"""
import json, sys
from pathlib import Path
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import episode_mae
from scripts.interpret.analysis.oracle_press_prob import bootstrap

s = hp.load_samples(sys.argv[1]); src = Path(sys.argv[2])
P = np.load(src / "predictions.npz"); A = json.loads((src / "angles.json").read_text())
y = s["labels"]["to_press"]
def comp(m, g):
    ch = A["selected"][f"{m}|{g}"]["to_press"]["layer_by_fold"]; r = np.full(len(y), np.nan)
    for f, l in enumerate(ch):
        r[s["outer"] == f] = P[f"{m}|{g}|{l}|to_press"][s["outer"] == f]
    return r
out = {}
for thr in (48, 64, 96):
    mask = y <= thr
    ref = episode_mae(y[mask], P["ref|t_N|to_press"][mask], s["groups"][mask])
    row = {"n_points": int(mask.sum()), "n_episodes": len(ref), "oracle_mae": float(np.mean(list(ref.values())))}
    for g in ("text", "current", "action"):
        a = episode_mae(y[mask], comp("mot", g)[mask], s["groups"][mask]); b = episode_mae(y[mask], comp("ctx", g)[mask], s["groups"][mask])
        d = bootstrap([b[k] - a[k] for k in sorted(set(a) & set(b))], 2000, 42)
        row[g] = {"mot_mae": float(np.mean(list(a.values()))), "ctx_mae": float(np.mean(list(b.values()))), "ctx_minus_mot": d}
    out[str(thr)] = row
    print(f"thr<={thr}: pts={row['n_points']} eps={row['n_episodes']} oracle={row['oracle_mae']:.1f} | " + " | ".join(
        f"{g}: mot={row[g]['mot_mae']:.1f} ctx={row[g]['ctx_mae']:.1f} diff={row[g]['ctx_minus_mot']['mean']:.1f}[{row[g]['ctx_minus_mot']['ci95'][0]:.1f},{row[g]['ctx_minus_mot']['ci95'][1]:.1f}]" for g in ("text", "current", "action")))
(src / "near_decision.json").write_text(json.dumps(out, indent=1))
