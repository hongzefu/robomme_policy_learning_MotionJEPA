"""探索性：分折种子稳健性。外层 5 折与内层 4 折的按局哈希种子换成 5 个不同值（原值 42 之外再取 1、7、123、2026），
重算 (a) [t,N]（独热×100）+ 组特征读 to_press 的跨模型 MAE 差；(b) 读速度 M 的跨模型准确率差。组 Text、History，层 0、8、17，PCA 64。
用法：python fold_seed_robustness.py <hover_points.json> <回放库根> <输出 json> <workers>
"""
import json, multiprocessing, os, sys
from pathlib import Path
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import episode_mae, episode_weights, hash_fold, OUTER_FOLDS, INNER_FOLDS
from scripts.interpret.analysis.oracle_press_prob import bootstrap

COL = {"history": 0, "text": 3}
SEEDS = (42, 1, 7, 123, 2026)
_S = {}

def with_seed(s, seed):
    t = dict(s)
    t["outer"] = np.asarray([hash_fold(int(k), OUTER_FOLDS, "outer", seed=seed) for k in s["groups"]])
    t["inner"] = np.asarray([hash_fold(int(k), INNER_FOLDS, "inner", seed=seed) for k in s["groups"]])
    return t

def task(args):
    model, group, layer, seed = args
    s = with_seed(_S["s"], seed); X = _S["banks"][model][:, layer + 1, COL[group]].astype(np.float64)
    T = hp.time_features(s, "t_N") * 100.0
    def f1(fold):
        tr = s["outer"] != fold; pca = hp.fit_pca(X[tr], episode_weights(s["groups"][tr]))
        return np.hstack([T, hp.pca_transform(pca, X)])
    p1, _, _, _ = hp.crossfit(f1, s, s["labels"]["to_press"][:, None], False)
    cl, inv = np.unique(s["M"], return_inverse=True)
    p2, _, _, _ = hp.crossfit(lambda f: X, s, np.eye(len(cl))[inv], True)
    m = np.full(len(inv), np.nan); k = np.isfinite(p2[:, 0]); m[k] = cl[p2[k].argmax(1)]
    return args, p1[:, 0], m

def main():
    pts, root, outp, workers = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    s = hp.load_samples(pts); banks = {m: np.concatenate([np.load(root / m / f"episode_{k}.npz")["hidden_prefix"] for k in s["keys"].tolist()]) for m in ("ctx", "mot")}
    _S.update(s=s, banks=banks)
    jobs = [(m, g, l, sd) for m in ("ctx", "mot") for g in ("text", "history") for l in (0, 8, 17) for sd in SEEDS]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = {a: (p, mm) for a, p, mm in pool.map(task, jobs, chunksize=1)}
    y = s["labels"]["to_press"]; rep = {}
    for g in ("text", "history"):
        for l in (0, 8, 17):
            for sd in SEEDS:
                (pc, mc), (pm, mm) = res[("ctx", g, l, sd)], res[("mot", g, l, sd)]
                ec, em = episode_mae(y, pc, s["groups"]), episode_mae(y, pm, s["groups"])
                a = bootstrap([ec[k] - em[k] for k in sorted(set(ec) & set(em))], 2000, 42)
                ac = hp.readout_episodes(s, "M", mc, s["labels"]["M"]); am = hp.readout_episodes(s, "M", mm, s["labels"]["M"])
                kk = np.isfinite(ac) & np.isfinite(am); b = bootstrap(am[kk] - ac[kk], 2000, 42)
                rep[f"{g}|{l}|{sd}"] = {"beyond_mae_ctx_minus_mot": a, "M_acc_mot_minus_ctx": b}
                print(f"{g} L{l} seed{sd}: beyond {a['mean']:+.1f}[{a['ci95'][0]:.1f},{a['ci95'][1]:.1f}] | M diff {b['mean']:+.3f}[{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}]")
    outp.write_text(json.dumps(rep, indent=1)); print("FOLD_ROBUST_DONE")

if __name__ == "__main__":
    main()
