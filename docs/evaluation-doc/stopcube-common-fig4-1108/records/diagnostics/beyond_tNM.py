"""探索性：在 [t,N,M]（独热×100，R² 0.926）之上再加组特征（训练折内 PCA64），看表征是否还有 M 以外的信息。
两模型 Text、History、Current、Action 组，第 0、8、17 层；同点跨模型 MAE 差与各自相对 [t,N,M] 的增量（按局重采样）。
用法：python beyond_tNM.py <hover_points.json> <回放库根> <输出 json> <workers>
"""
import json, multiprocessing, os, sys
from pathlib import Path
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import episode_mae, episode_weights, metrics
from scripts.interpret.analysis.oracle_press_prob import bootstrap

COL = {"history": 0, "current": 2, "text": 3}
_S = {}

def task(args):
    model, group, layer = args
    s = _S["s"]; b = _S["banks"][model]
    X = (b["action"][:, layer] if group == "action" else b["prefix"][:, layer + 1, COL[group]]).astype(np.float64)
    T = _S["T"]
    def f(fold):
        tr = s["outer"] != fold; pca = hp.fit_pca(X[tr], episode_weights(s["groups"][tr]))
        return np.hstack([T, hp.pca_transform(pca, X)])
    p, _, leak, _ = hp.crossfit(f, s, s["labels"]["to_press"][:, None], False)
    return args, p[:, 0], leak

def main():
    pts, root, outp, workers = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    s = hp.load_samples(pts); oh = hp._one_hot
    T = np.hstack([oh(s["t"]), oh(s["N"]), oh(s["M"])]) * 100.0
    banks = {}
    for m in ("ctx", "mot"):
        zs = [np.load(root / m / f"episode_{k}.npz") for k in s["keys"].tolist()]
        banks[m] = {"prefix": np.concatenate([z["hidden_prefix"] for z in zs]), "action": np.concatenate([z["hidden_action"] for z in zs])}
    _S.update(s=s, banks=banks, T=T)
    y = s["labels"]["to_press"]
    base, _, _, _ = hp.crossfit(lambda f: T, s, y[:, None], False); base = base[:, 0]
    mb = metrics(y, base, s["groups"]); eb = episode_mae(y, base, s["groups"])
    jobs = [(m, g, l) for m in ("ctx", "mot") for g in ("text", "history", "current", "action") for l in (0, 8, 17)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = {a: p for a, p, _ in pool.map(task, jobs, chunksize=1)}
    rep = {"tNM": mb, "cells": {}}
    print(f"[t,N,M] alone r2={mb['r2']:.3f} mae={mb['mae']:.1f}")
    for g in ("text", "history", "current", "action"):
        for l in (0, 8, 17):
            out = {}
            for m in ("ctx", "mot"):
                mm = metrics(y, res[(m, g, l)], s["groups"]); e = episode_mae(y, res[(m, g, l)], s["groups"])
                out[m] = {"r2": mm["r2"], "mae": mm["mae"], "gain": bootstrap([eb[k] - e[k] for k in eb], 2000, 42)}
            ec, em = episode_mae(y, res[("ctx", g, l)], s["groups"]), episode_mae(y, res[("mot", g, l)], s["groups"])
            out["ctx_minus_mot"] = bootstrap([ec[k] - em[k] for k in ec], 2000, 42)
            rep["cells"][f"{g}|{l}"] = out
            print(f"{g} L{l}: ctx r2={out['ctx']['r2']:.3f} gain {out['ctx']['gain']['mean']:+.1f} | mot r2={out['mot']['r2']:.3f} gain {out['mot']['gain']['mean']:+.1f}[{out['mot']['gain']['ci95'][0]:.1f},{out['mot']['gain']['ci95'][1]:.1f}] | ctx-mot {out['ctx_minus_mot']['mean']:+.1f}[{out['ctx_minus_mot']['ci95'][0]:.1f},{out['ctx_minus_mot']['ci95'][1]:.1f}]")
    outp.write_text(json.dumps(rep, indent=1, default=float)); print("TNM_DONE")

if __name__ == "__main__":
    main()
