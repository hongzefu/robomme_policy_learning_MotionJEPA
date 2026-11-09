"""探索性阴性对照：整局打乱 (M,N)（每局换成另一局的 M、N，t 不变），用 hover_points.point_labels 重算 to_press，
[t,N] 特征也用打乱后的 N。比较「[t,N_shuf]（独热×100）+ 组特征」与「只用 [t,N_shuf]」的 MAE 差。打乱 5 次（seed 0–4）。
组：两模型 Text、History，第 8、17 层；PCA 64。真实标签下的增量见 beyond_oracle_scale100.json。
用法：python shuffle_control.py <hover_points.json> <回放库根> <输出 json> <workers>
"""
import json, multiprocessing, os, sys
from pathlib import Path
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import episode_mae, episode_weights
from scripts.interpret.hover_points import point_labels

COL = {"history": 0, "text": 3}
_S = {}

def shuffled(s, seed):
    rng = np.random.default_rng(seed); keys = s["keys"]; donor = keys[rng.permutation(len(keys))]
    first = {int(k): int(np.flatnonzero(s["groups"] == k)[0]) for k in keys}
    mp = {int(k): first[int(d)] for k, d in zip(keys, donor)}
    N = np.empty(len(s["t"]), int); y = np.empty(len(s["t"]))
    for i, (k, t) in enumerate(zip(s["groups"].tolist(), s["t"].tolist())):
        src = mp[k]; M, n = int(s["M"][src]), int(s["N"][src]); N[i] = n; y[i] = point_labels(t, M, n)["to_press"]
    return N, y

def task(args):
    model, group, layer, seed = args
    s = _S["s"]; X = _S["banks"][model][:, layer + 1, COL[group]].astype(np.float64)
    N, y = _S["shuf"][seed]
    T = np.hstack([hp._one_hot(s["t"]), hp._one_hot(N)]) * 100.0
    base, _, _, _ = hp.crossfit(lambda f: T, s, y[:, None], False)
    def f1(fold):
        tr = s["outer"] != fold; pca = hp.fit_pca(X[tr], episode_weights(s["groups"][tr]))
        return np.hstack([T, hp.pca_transform(pca, X)])
    p, _, _, _ = hp.crossfit(f1, s, y[:, None], False)
    eb, ep = episode_mae(y, base[:, 0], s["groups"]), episode_mae(y, p[:, 0], s["groups"])
    return args, float(np.mean([eb[k] - ep[k] for k in eb]))

def main():
    pts, root, outp, workers = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    s = hp.load_samples(pts)
    banks = {m: np.concatenate([np.load(root / m / f"episode_{k}.npz")["hidden_prefix"] for k in s["keys"].tolist()]) for m in ("ctx", "mot")}
    _S.update(s=s, banks=banks, shuf={sd: shuffled(s, sd) for sd in range(5)})
    jobs = [(m, g, l, sd) for m in ("ctx", "mot") for g in ("text", "history") for l in (8, 17) for sd in range(5)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = dict(pool.map(task, jobs, chunksize=1))
    rep = {}
    for m in ("ctx", "mot"):
        for g in ("text", "history"):
            for l in (8, 17):
                v = [res[(m, g, l, sd)] for sd in range(5)]
                rep[f"{m}|{g}|{l}"] = {"gain_per_shuffle": v, "mean": float(np.mean(v))}
                print(f"{m} {g} L{l}: shuffled gain mean {np.mean(v):+.1f} range [{min(v):+.1f},{max(v):+.1f}]")
    outp.write_text(json.dumps(rep, indent=1)); print("SHUFFLE_DONE")

if __name__ == "__main__":
    main()
