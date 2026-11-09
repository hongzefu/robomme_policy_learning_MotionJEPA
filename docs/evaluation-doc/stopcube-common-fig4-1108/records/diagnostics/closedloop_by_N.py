"""探索性：旧 Fig. 4（各自闭环 50 局，q4 点集，第 17 层折外预测）按目标次数 N 拆开的 MAE。
输入：stopcube-rollout-1105 的 fit-predictions.npz 与 stopcube-oracle-1104 的 oracle_table.json（取 target_pass）。
用法：python closedloop_by_N.py <fit-predictions.npz> <oracle_table.json>
"""
import json, sys
import numpy as np
from scripts.interpret.analysis.oracle_press_prob import bootstrap

z = np.load(sys.argv[1]); table = json.load(open(sys.argv[2]))["episodes"]
N_of = {k: v["target_pass"] for k, v in table.items()}
res = {}
for model, groups in (("ctx", ("文字", "当前视觉", "action", "历史视觉", "time_N")), ("mot", ("文字", "当前视觉", "action", "motion", "time_N"))):
    y = z[f"{model}/q4/y"]; g = z[f"{model}/q4/groups"]
    N = np.array([N_of[x] for x in g])
    for grp in groups:
        p = z[f"{model}/q4/{grp}"]
        for n in sorted(set(N.tolist())):
            eps = sorted(set(g[N == n].tolist()))
            per = [float(np.abs(y[g == e] - p[g == e]).mean()) for e in eps]
            b = bootstrap(per, 2000, 42)
            res[f"{model}|{grp}|N={n}"] = (b["mean"], b["ci95"], len(eps))
for k, (m, ci, n) in res.items():
    print(f"{k}: MAE={m:.1f} ci=[{ci[0]:.1f},{ci[1]:.1f}] eps={n}")
