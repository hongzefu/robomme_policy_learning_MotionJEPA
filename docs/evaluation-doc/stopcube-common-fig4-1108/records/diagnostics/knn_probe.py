"""探索性：非线性稳健性检查。训练折内标准化 + PCA64（同 hover_probe.fit_pca，每局等权），再用 k 近邻（欧氏距离，k∈{5,15,45} 在训练折内按局 4 折选）
读 to_press（回归，近邻均值）与速度 M（分类，近邻多数）。外层 5 折同 hover_probe。层取 4、8、12、17。
用法：python knn_probe.py <hover_points.json> <回放库根> <输出 json> <workers>
"""
import json, multiprocessing, os, sys
from pathlib import Path
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import metrics, episode_weights, episode_mae
from scripts.interpret.analysis.oracle_press_prob import bootstrap

COL = {"history": 0, "current": 2, "text": 3}
KS = (5, 15, 45)
LAYERS = (4, 8, 12, 17)
_S = {}

def knn_predict(Xtr, ytr, Xte, k, classify):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d, k, axis=1)[:, :k]
    nb = ytr[idx]
    if not classify:
        return nb.mean(1)
    vals = np.unique(ytr)
    counts = np.stack([(nb == v).sum(1) for v in vals], 1)
    return vals[counts.argmax(1)]

def fit_fold(X, s, train, test, y, classify):
    pca = hp.fit_pca(X[train], episode_weights(s["groups"][train]))
    Ztr, Zte = hp.pca_transform(pca, X[train]), hp.pca_transform(pca, X[test])
    inner = s["inner"][train]; ytr = y[train]
    best, best_err = KS[0], np.inf
    for k in KS:
        err = 0.0
        for f in np.unique(inner):
            h = inner == f
            if h.all() or (~h).sum() <= k:
                continue
            p = knn_predict(Ztr[~h], ytr[~h], Ztr[h], k, classify)
            err += np.mean(p != ytr[h]) if classify else np.mean(np.abs(p - ytr[h]))
        if err < best_err:
            best, best_err = k, err
    return knn_predict(Ztr, ytr, Zte, best, classify)

def task(args):
    model, group, layer = args
    s = _S["s"]; bank = _S["banks"][model]
    X = (bank["action"][:, layer] if group == "action" else bank["prefix"][:, layer + 1, COL[group]]).astype(np.float64)
    out = {}
    for lab, classify in (("to_press", False), ("M", True)):
        y = s["labels"][lab]; pred = np.full(len(y), np.nan)
        for fold in range(hp.OUTER_FOLDS):
            test = s["outer"] == fold
            pred[test] = fit_fold(X, s, ~test, test, y, classify)
        out[lab] = pred
    return args, out

def main():
    pts, root, outp, workers = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    s = hp.load_samples(pts); banks = {}
    for m in ("ctx", "mot"):
        zs = [np.load(root / m / f"episode_{k}.npz") for k in s["keys"].tolist()]
        banks[m] = {"prefix": np.concatenate([z["hidden_prefix"] for z in zs]), "action": np.concatenate([z["hidden_action"] for z in zs])}
    _S.update(s=s, banks=banks)
    jobs = [(m, g, l) for m in ("ctx", "mot") for g in ("text", "current", "history", "action") for l in LAYERS]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = dict(pool.map(task, jobs, chunksize=1))
    rep = {"cells": {}, "cross": {}}
    y = s["labels"]["to_press"]
    for (m, g, l), o in res.items():
        r = metrics(y, o["to_press"], s["groups"])
        acc = float(np.nanmean(hp.readout_episodes(s, "M", o["M"], s["labels"]["M"])))
        rep["cells"][f"{m}|{g}|{l}"] = {"to_press_r2": r["r2"], "to_press_mae": r["mae"], "M_acc": acc}
    for g in ("text", "current", "history", "action"):
        for l in LAYERS:
            a, b = res[("mot", g, l)], res[("ctx", g, l)]
            ea, eb = episode_mae(y, a["to_press"], s["groups"]), episode_mae(y, b["to_press"], s["groups"])
            ma = hp.readout_episodes(s, "M", a["M"], s["labels"]["M"]); mb = hp.readout_episodes(s, "M", b["M"], s["labels"]["M"])
            keep = np.isfinite(ma) & np.isfinite(mb)
            rep["cross"][f"{g}|{l}"] = {"to_press_mae_ctx_minus_mot": bootstrap([eb[k] - ea[k] for k in sorted(set(ea) & set(eb))], 2000, 42),
                                       "M_acc_mot_minus_ctx": bootstrap(ma[keep] - mb[keep], 2000, 42)}
    outp.write_text(json.dumps(rep, indent=1, default=float))
    for g in ("text", "current", "history", "action"):
        for l in LAYERS:
            c, d = rep["cells"][f"ctx|{g}|{l}"], rep["cells"][f"mot|{g}|{l}"]; x = rep["cross"][f"{g}|{l}"]
            t, mm = x["to_press_mae_ctx_minus_mot"], x["M_acc_mot_minus_ctx"]
            print(f"{g} L{l}: r2 ctx={c['to_press_r2']:.2f} mot={d['to_press_r2']:.2f} maediff={t['mean']:.1f}[{t['ci95'][0]:.1f},{t['ci95'][1]:.1f}] | M ctx={c['M_acc']:.2f} mot={d['M_acc']:.2f} diff={mm['mean']:.3f}[{mm['ci95'][0]:.3f},{mm['ci95'][1]:.3f}]")
    print("KNN_DONE")

if __name__ == "__main__":
    main()
