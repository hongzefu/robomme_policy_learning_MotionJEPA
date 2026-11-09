"""探索性：样本效率。外层训练局按 hover_probe.curve_filter 取 25%／50%／100%（测试折不变），重算
(a) Text、History 组读速度 M 的准确率（训练折内 PCA64，一对多岭）；(b) [t,N]（独热×100）+ 组特征读 to_press 的 MAE 相对只用 [t,N] 的减少。
两模型，层 0、8、17；跨模型配对差按局重采样。用法：python learning_curve.py <hover_points.json> <回放库根> <输出 json> <workers>
"""
import json, multiprocessing, os, sys
from pathlib import Path
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import episode_mae, episode_weights
from scripts.interpret.analysis.oracle_press_prob import bootstrap

COL = {"history": 0, "text": 3}
FRACS = (0.25, 0.5, 1.0)
_S = {}

def task(args):
    model, group, layer, frac = args
    s = _S["s"]; X = _S["banks"][model][:, layer + 1, COL[group]].astype(np.float64)
    filt = hp.curve_filter(s, frac)
    cl, inv = np.unique(s["M"], return_inverse=True)
    pm, _, _, _ = hp.crossfit(lambda f: X, s, np.eye(len(cl))[inv], True, train_filter=filt)
    m = np.full(len(inv), np.nan); k = np.isfinite(pm[:, 0]); m[k] = cl[pm[k].argmax(1)]
    T = hp.time_features(s, "t_N") * 100.0
    def f(fold):
        tr = filt(s["outer"] != fold); pca = hp.fit_pca(X[tr], episode_weights(s["groups"][tr]))
        return np.hstack([T, hp.pca_transform(pca, X)])
    pt, _, _, _ = hp.crossfit(f, s, s["labels"]["to_press"][:, None], False, train_filter=filt)
    pb, _, _, _ = hp.crossfit(lambda fo: T, s, s["labels"]["to_press"][:, None], False, train_filter=filt)
    return args, m, pt[:, 0], pb[:, 0]

def main():
    pts, root, outp, workers = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    s = hp.load_samples(pts)
    banks = {m: np.concatenate([np.load(root / m / f"episode_{k}.npz")["hidden_prefix"] for k in s["keys"].tolist()]) for m in ("ctx", "mot")}
    _S.update(s=s, banks=banks)
    jobs = [(m, g, l, fr) for m in ("ctx", "mot") for g in ("text", "history") for l in (0, 8, 17) for fr in FRACS]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = {a: r for a, *r in pool.map(task, jobs, chunksize=1)}
    y = s["labels"]["to_press"]; rep = {}
    for g in ("text", "history"):
        for l in (0, 8, 17):
            for fr in FRACS:
                row = {}
                for m in ("ctx", "mot"):
                    mm, pt, pb = res[(m, g, l, fr)]
                    acc = hp.readout_episodes(s, "M", mm, s["labels"]["M"])
                    eb, et = episode_mae(y, pb, s["groups"]), episode_mae(y, pt, s["groups"])
                    row[m] = {"M_acc": acc, "gain": {k: eb[k] - et[k] for k in eb if k in et}}
                a, b = row["mot"]["M_acc"], row["ctx"]["M_acc"]; kk = np.isfinite(a) & np.isfinite(b)
                dM = bootstrap(a[kk] - b[kk], 2000, 42)
                gm, gc = row["mot"]["gain"], row["ctx"]["gain"]
                dG = bootstrap([gm[k] - gc[k] for k in sorted(set(gm) & set(gc))], 2000, 42)
                rep[f"{g}|{l}|{fr}"] = {"M_ctx": float(np.nanmean(b)), "M_mot": float(np.nanmean(a)), "M_diff": dM,
                                        "gain_ctx": float(np.mean(list(gc.values()))), "gain_mot": float(np.mean(list(gm.values()))), "gain_diff": dG}
                r = rep[f"{g}|{l}|{fr}"]
                print(f"{g} L{l} frac{fr}: M ctx={r['M_ctx']:.2f} mot={r['M_mot']:.2f} diff {dM['mean']:+.3f}[{dM['ci95'][0]:.3f},{dM['ci95'][1]:.3f}] | gain ctx={r['gain_ctx']:+.1f} mot={r['gain_mot']:+.1f} diff {dG['mean']:+.1f}[{dG['ci95'][0]:.1f},{dG['ci95'][1]:.1f}]")
    outp.write_text(json.dumps(rep, indent=1, default=float)); print("CURVE_DONE")

if __name__ == "__main__":
    main()
