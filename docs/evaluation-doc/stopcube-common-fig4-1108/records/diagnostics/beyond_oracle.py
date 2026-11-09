"""探索性：oracle [t,N] 独热 + 组特征（训练折内标准化 + PCA 64）拼接后读 to_press，减去只用 [t,N] 的读数 = 表征提供的 t、N 之外的信息。
方法同 hover_probe（按局 5 折、内层选 λ、每局等权）；PCA 只在每折训练行上拟合。
用法：python beyond_oracle.py <hover_points.json> <回放库根> <输出 json> <workers>
"""
import json, multiprocessing, os, sys
from pathlib import Path
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import episode_mae, episode_weights, metrics
from scripts.interpret.analysis.oracle_press_prob import bootstrap

COL = {"history": 0, "motion": 1, "current": 2, "text": 3}
_S = {}

def task(args):
    model, group, layer = args
    s = _S["s"]; y = s["labels"]["to_press"][:, None]
    if group == "action":
        X = _S["actions"][model][:, layer].astype(np.float64)
    else:
        X = _S["banks"][model][:, layer + 1, COL[group]].astype(np.float64)
    T = hp.time_features(s, "t_N")
    def feats(fold):
        train = s["outer"] != fold
        pca = hp.fit_pca(X[train], episode_weights(s["groups"][train]))
        return np.hstack([T, hp.pca_transform(pca, X)])
    pred, _, leak, _ = hp.crossfit(feats, s, y, False)
    return (model, group, layer), pred[:, 0], leak

def main():
    points, root, out, workers = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
    s = hp.load_samples(points); banks, actions = {}, {}
    for m in ("ctx", "mot"):
        zs = [np.load(Path(root) / m / f"episode_{k}.npz") for k in s["keys"].tolist()]
        banks[m] = np.concatenate([z["hidden_prefix"] for z in zs])
        if all("hidden_action" in z.files for z in zs):
            actions[m] = np.concatenate([z["hidden_action"] for z in zs])
    _S.update(s=s, banks=banks, actions=actions)
    y = s["labels"]["to_press"]
    base, _, _, _ = hp.crossfit(lambda f: hp.time_features(s, "t_N"), s, y[:, None], False)
    base = base[:, 0]
    jobs = [(m, g, l) for m in ("ctx", "mot") for g in ("text", "current", "history") + (("motion",) if m == "mot" else ())
            for l in range(-1, 18)]
    jobs += [(m, "action", l) for m in actions for l in range(18)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = pool.map(task, jobs, chunksize=1)
    mb = metrics(y, base, s["groups"]); eb = episode_mae(y, base, s["groups"])
    report = {"oracle": mb, "cells": {}, "leak": int(sum(r[2] for r in res))}
    preds = {}
    for (m, g, l), p, _ in res:
        mm = metrics(y, p, s["groups"]); ep = episode_mae(y, p, s["groups"])
        d = bootstrap([eb[k] - ep[k] for k in sorted(set(eb) & set(ep))], 2000, 42)
        report["cells"][f"{m}|{g}|{l}"] = {"r2": mm["r2"], "mae": mm["mae"], "r2_gain": mm["r2"] - mb["r2"],
                                          "mae_gain": d["mean"], "mae_gain_ci95": d["ci95"]}
        preds[(m, g, l)] = p
    out.write_text(json.dumps(report, indent=1))
    np.savez_compressed(out.with_suffix(".npz"), base=base, **{f"{m}|{g}|{l}": v for (m, g, l), v in preds.items()})
    # 跨模型：同点同折，按局先相减再重采样（正 = MotionMem 误差小）
    cross = {}
    for g in ("text", "current", "history", "action"):
        for l in range(-1, 18):
            a_, b_ = preds.get(("mot", g, l)), preds.get(("ctx", g, l))
            if a_ is None or b_ is None:
                continue
            ea, eb = episode_mae(y, a_, s["groups"]), episode_mae(y, b_, s["groups"])
            cross[f"{g}|{l}"] = bootstrap([eb[k] - ea[k] for k in sorted(set(ea) & set(eb))], 2000, 42)
    report["cross_model_mae_ctx_minus_mot"] = cross
    out.write_text(json.dumps(report, indent=1))
    for g in ("text", "current", "history", "action"):
        ls = [l for l in range(-1, 18) if f"{g}|{l}" in cross]
        print("cross", g, " ".join(f"{cross[f'{g}|{l}']['mean']:.0f}[{cross[f'{g}|{l}']['ci95'][0]:.0f},{cross[f'{g}|{l}']['ci95'][1]:.0f}]" for l in ls))
    print("oracle", round(mb["r2"], 3), round(mb["mae"], 1), "leak", report["leak"])
    for m in ("ctx", "mot"):
        for g in ("text", "current", "history", "action"):
            ls = [l for l in range(-1, 18) if f"{m}|{g}|{l}" in report["cells"]]
            print(m, g, "r2:", " ".join(f"{report['cells'][f'{m}|{g}|{l}']['r2']:.2f}" for l in ls))
            print(m, g, "maegain:", " ".join(f"{report['cells'][f'{m}|{g}|{l}']['mae_gain']:.0f}" for l in ls))

if __name__ == "__main__":
    main()
