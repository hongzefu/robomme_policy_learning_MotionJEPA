"""探索性：跨 N 外推。探针只在 N≤3 的局上训练、在 N≥4 的局上测（fold A），以及反方向（fold B）。
特征：组特征训练集内标准化 + PCA64（同 hover_probe.crossfit）；λ 在训练局内按局 4 折选。标签 to_press 与 M（速度，分类）。
参考：[t] 与 [t,N] 用连续值（独热无法外推到未见过的 N），以及 [t,N] 连续 + N·t 交互。
用法：python cross_N_generalize.py <hover_points.json> <回放库根> <输出 json> <workers>
"""
import json, multiprocessing, os, sys
from pathlib import Path
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import metrics

COL = {"history": 0, "motion": 1, "current": 2, "text": 3}
_S = {}

def run(X, use_pca):
    s = _S["s"]; Y, layout = _S["Y"], _S["layout"]
    pred, _, leak, _ = hp.crossfit(lambda f: X, s, Y, use_pca)
    out = {}
    for lab in ("to_press", "M"):
        p = hp.decode(pred, layout[(lab, None)])
        for name, fold in (("train_le3_test_ge4", 0), ("train_ge4_test_le3", 1)):
            m = s["outer"] == fold
            if lab == "to_press":
                r = metrics(s["labels"]["to_press"], np.where(m, p, np.nan), s["groups"])
                out[f"{lab}|{name}"] = {"r2": r["r2"], "mae": r["mae"]}
            else:
                out[f"{lab}|{name}"] = {"acc": float(np.nanmean(hp.readout_episodes(s, "M", np.where(m, p, np.nan), s["labels"]["M"])))}
    return out, leak

def task(args):
    model, group, layer = args
    bank = _S["banks"][model]
    X = (bank["action"][:, layer] if group == "action" else bank["prefix"][:, layer + 1, COL[group]]).astype(np.float64)
    return args, *run(X, True)

def main():
    pts, root, out, workers = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    s = hp.load_samples(pts)
    s["outer"] = np.where(s["N"] >= 4, 0, 1)
    Y, layout = hp.build_targets(s)
    banks = {}
    for m in ("ctx", "mot"):
        zs = [np.load(root / m / f"episode_{k}.npz") for k in s["keys"].tolist()]
        banks[m] = {"prefix": np.concatenate([z["hidden_prefix"] for z in zs]), "action": np.concatenate([z["hidden_action"] for z in zs])}
    _S.update(s=s, Y=Y, layout=layout, banks=banks)
    t, N = s["t"].astype(float), s["N"].astype(float)
    refs = {name: run(X, False)[0] for name, X in {"[t]": t[:, None], "[t,N]": np.stack([t, N], 1),
                                                    "[t,N,tN]": np.stack([t, N, t * N], 1)}.items()}
    jobs = [(m, g, l) for m in ("ctx", "mot") for g in ("text", "current", "history") + (("motion",) if m == "mot" else ())
            for l in range(-1, 18)] + [(m, "action", l) for m in ("ctx", "mot") for l in range(18)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = pool.map(task, jobs, chunksize=1)
    report = {"refs": refs, "cells": {f"{m}|{g}|{l}": o for (m, g, l), o, _ in res}, "leak": int(sum(r[2] for r in res))}
    out.write_text(json.dumps(report, indent=1, default=float))
    print("leak", report["leak"])
    for k, v in refs.items():
        print("ref", k, {a: {x: round(y, 3) for x, y in b.items()} for a, b in v.items()})
    for key in ("to_press|train_le3_test_ge4", "M|train_le3_test_ge4", "to_press|train_ge4_test_le3"):
        metric = "r2" if key.startswith("to_press") else "acc"
        for m in ("ctx", "mot"):
            for g in ("text", "current", "history", "action"):
                ls = [l for l in range(-1, 18) if f"{m}|{g}|{l}" in report["cells"]]
                print(key, m, g, " ".join(f"{report['cells'][f'{m}|{g}|{l}'][key][metric]:.2f}" for l in ls))
    print("CROSSN_DONE")

if __name__ == "__main__":
    main()
