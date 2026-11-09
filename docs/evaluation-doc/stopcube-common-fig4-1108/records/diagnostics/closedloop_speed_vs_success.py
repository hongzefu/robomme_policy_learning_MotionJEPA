"""探索性：闭环（stopcube-rollout-1105，各模型自己的轨迹，Q4 样本集）里，每局「读对速度 M 的比例」（指定组、层的折外预测）与该局成功与否的关系。
用 closedloop_beyond_oracle.make_samples 取样；速度读数为一对多岭（训练折内 PCA64，按局 5 折）；成功取 align.json 的 task_success。
报：成功局与失败局的平均读对比例、差值的按局重采样区间、以及按「读对比例 ≥0.5」分组的成功率。
用法：python closedloop_speed_vs_success.py <rollout 产物根>
"""
import json, os, sys
from pathlib import Path
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from closedloop_beyond_oracle import make_samples, COL
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_attention import load_align
from scripts.interpret.analysis.oracle_press_prob import bootstrap

root = Path(sys.argv[1]); align = load_align(root)
for model, cells in (("mot", [("text", 0), ("history", 8), ("motion", -1)]), ("ctx", [("text", 8), ("history", 8)])):
    eps = align["models"][model]
    s, prefix, action, _ = make_samples(model, root, eps)
    keys = sorted({k for k in eps if eps[k].get("valid") and eps[k]["uses"]["q4"]}, key=lambda k: int(k.split(":")[1]))
    succ = {i: int(eps[k].get("task_success") or 0) for i, k in enumerate(keys)}
    for g, l in cells:
        X = prefix[:, l + 1, COL[g]].astype(np.float64)
        cl, inv = np.unique(s["M"], return_inverse=True)
        pred, _, _, _ = hp.crossfit(lambda f: X, s, np.eye(len(cl))[inv], True)
        m = np.full(len(inv), np.nan); k = np.isfinite(pred[:, 0]); m[k] = cl[pred[k].argmax(1)]
        acc = {int(e): float(np.mean(m[s["groups"] == e] == s["M"][s["groups"] == e])) for e in np.unique(s["groups"])}
        ys = [acc[e] for e in acc if succ.get(e) == 1]; ns = [acc[e] for e in acc if succ.get(e) == 0]
        rng = np.random.default_rng(42)
        diffs = [np.mean(rng.choice(ys, len(ys))) - np.mean(rng.choice(ns, len(ns))) for _ in range(2000)]
        hi = [succ[e] for e in acc if acc[e] >= 0.5]; lo = [succ[e] for e in acc if acc[e] < 0.5]
        Mep = {int(e): int(s["M"][s["groups"] == e][0]) for e in acc}
        strat = []
        for Mv in sorted(set(Mep.values())):
            a1 = [acc[e] for e in acc if Mep[e] == Mv and succ.get(e) == 1]; a0 = [acc[e] for e in acc if Mep[e] == Mv and succ.get(e) == 0]
            if a1 and a0:
                strat.append((Mv, len(a1), len(a0), np.mean(a1) - np.mean(a0)))
        print("   within-M:", " ".join(f"M={Mv}: succ{n1}/fail{n0} diff={d:+.3f}" for Mv, n1, n0, d in strat))
        print(f"{model} {g} L{l}: eps={len(acc)} succ={len(ys)} | acc succ={np.mean(ys):.2f} fail={np.mean(ns):.2f} diff={np.mean(ys)-np.mean(ns):+.3f}[{np.quantile(diffs,0.025):+.3f},{np.quantile(diffs,0.975):+.3f}] | success rate acc>=0.5: {sum(hi)}/{len(hi)}  acc<0.5: {sum(lo)}/{len(lo)}")
