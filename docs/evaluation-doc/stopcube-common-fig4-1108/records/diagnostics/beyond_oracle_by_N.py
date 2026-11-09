"""探索性：修正口径（ONEHOT_SCALE=100）的「[t,N] + 组特征」折外预测，按 N≤3／N≥4 分档，跨模型局级 MAE 配对差（FrameSamp − MotionMem，按局重采样）。
数据：explore/scale100/beyond_oracle.npz（键 ctx|g|l、mot|g|l、base）。用法：python beyond_oracle_by_N.py <hover_points.json> <npz> [组,组]
"""
import sys
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import episode_mae
from scripts.interpret.analysis.oracle_press_prob import bootstrap

s = hp.load_samples(sys.argv[1]); P = np.load(sys.argv[2]); y = s["labels"]["to_press"]
for tag, mask in (("N<=3", s["N"] <= 3), ("N>=4", s["N"] >= 4)):
    for g in (sys.argv[3].split(",") if len(sys.argv) > 3 else ("text", "history")):
        for l in (0, 8, 17):
            ec = episode_mae(y[mask], P[f"ctx|{g}|{l}"][mask], s["groups"][mask]); em = episode_mae(y[mask], P[f"mot|{g}|{l}"][mask], s["groups"][mask])
            d = bootstrap([ec[k] - em[k] for k in sorted(set(ec) & set(em))], 2000, 42)
            print(f"{tag} {g} L{l}: FS-MM MAE diff {d['mean']:+.1f}[{d['ci95'][0]:.1f},{d['ci95'][1]:.1f}] n_eps={d['n_episodes']}")
