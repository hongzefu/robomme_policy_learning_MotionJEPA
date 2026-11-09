"""探索性：旧闭环 rollout（stopcube-rollout-1105，各模型自己的 50 局，Q4 = 距正确窗口中心）上做「[t,N] 独热 + 组特征」增量。
样本选取、标签与 rollout_probe 相同（load_align／load_bank／build_samples，label=q4）；折用 rollout_probe.hash_fold（两模型同映射）；
拟合同 hover_probe（训练折内标准化 + PCA 64、内层按局选 λ、每局等权）。每个模型只在自己的轨迹上算，跨模型只比「增量」。
用法：python closedloop_beyond_oracle.py <rollout 产物根> <输出 json> <workers>
"""
import json, multiprocessing, os, sys
from pathlib import Path
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis import rollout_probe as rp
from scripts.interpret.analysis.rollout_attention import load_align
from scripts.interpret.analysis.oracle_press_prob import bootstrap

COL = {"history": 0, "motion": 1, "current": 2, "text": 3}
_S = {}

def make_samples(model, root, episodes):
    bank = rp.load_bank(root, model, episodes)
    raw = rp.build_samples(bank, episodes, "q4")
    keys = sorted(set(raw["groups"].tolist()), key=lambda k: int(k.split(":")[1]))
    kid = {k: i for i, k in enumerate(keys)}
    groups = np.asarray([kid[k] for k in raw["groups"]], dtype=np.int64)
    uk, gi = np.unique(groups, return_inverse=True)
    M = np.asarray([episodes[k]["move_interval"] for k in raw["groups"]])
    s = {"groups": groups, "keys": uk, "group_index": gi, "t": raw["t"].astype(np.int64), "N": raw["n_pass"].astype(np.int64),
         "M": M, "y": raw["y"],
         "outer": np.asarray([rp.hash_fold(k, rp.OUTER_FOLDS, "outer") for k in raw["groups"]]),
         "inner": np.asarray([rp.hash_fold(k, rp.INNER_FOLDS, "inner") for k in raw["groups"]])}
    return s, bank["prefix"][raw["rows"]], bank["action"][raw["rows"]], bank["motion_present"]

def fit(s, feats):
    p, _, leak, _ = hp.crossfit(feats, s, s["y"][:, None], False)
    return p[:, 0], leak

def task(args):
    model, group, layer = args
    s, prefix, action = _S[model]
    X = (action[:, layer] if group == "action" else prefix[:, layer + 1, COL[group]]).astype(np.float64)
    T = np.hstack([hp._one_hot(s["t"]), hp._one_hot(s["N"])])
    def feats(fold):
        train = s["outer"] != fold
        pca = hp.fit_pca(X[train], rp.episode_weights(s["groups"][train]))
        return np.hstack([T, hp.pca_transform(pca, X)])
    p, leak = fit(s, feats)
    return args, p, leak

def main():
    root, out, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    align = load_align(root)
    jobs = []
    for model in ("ctx", "mot"):
        s, prefix, action, motion = make_samples(model, root, align["models"][model])
        _S[model] = (s, prefix, action)
        gs = ["text", "current", "history"] + (["motion"] if motion else [])
        jobs += [(model, g, l) for g in gs for l in range(-1, 18)] + [(model, "action", l) for l in range(18)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = pool.map(task, jobs, chunksize=1)
    report = {"note": "探索性；闭环各自轨迹；q4", "models": {}, "cells": {}, "leak": 0}
    base_ep = {}
    for model in ("ctx", "mot"):
        s = _S[model][0]
        refs = {}
        for name, cols in {"t_N": [hp._one_hot(s["t"]), hp._one_hot(s["N"])],
                           "t_N_M": [hp._one_hot(s["t"]), hp._one_hot(s["N"]), hp._one_hot(s["M"])]}.items():
            X = np.hstack(cols); p, _ = fit(s, lambda f: X)
            refs[name] = rp.metrics(s["y"], p, s["groups"])
            if name == "t_N":
                base_ep[model] = rp.episode_mae(s["y"], p, s["groups"])
        report["models"][model] = {"n_points": int(len(s["y"])), "n_episodes": int(len(s["keys"])), "refs": refs}
    for (model, group, layer), p, leak in res:
        s = _S[model][0]; report["leak"] += leak
        m = rp.metrics(s["y"], p, s["groups"]); e = rp.episode_mae(s["y"], p, s["groups"]); b = base_ep[model]
        d = bootstrap([b[k] - e[k] for k in sorted(set(b) & set(e))], 2000, 42)
        report["cells"][f"{model}|{group}|{layer}"] = {"r2": m["r2"], "mae": m["mae"], "mae_gain": d["mean"], "ci95": d["ci95"]}
    out.write_text(json.dumps(report, indent=1, default=float))
    print("leak", report["leak"])
    for model in ("ctx", "mot"):
        r = report["models"][model]
        print(model, "points", r["n_points"], "eps", r["n_episodes"], "t_N", round(r["refs"]["t_N"]["r2"], 3), round(r["refs"]["t_N"]["mae"], 1),
              "t_N_M", round(r["refs"]["t_N_M"]["r2"], 3))
        for g in ("text", "current", "history", "motion", "action"):
            ls = [l for l in range(-1, 18) if f"{model}|{g}|{l}" in report["cells"]]
            if not ls: continue
            print(f"  {g} r2:", " ".join(f"{report['cells'][f'{model}|{g}|{l}']['r2']:.2f}" for l in ls))
            print(f"  {g} gain:", " ".join(f"{report['cells'][f'{model}|{g}|{l}']['mae_gain']:.0f}" for l in ls))
    print("CLOSEDLOOP_DONE")

if __name__ == "__main__":
    main()
