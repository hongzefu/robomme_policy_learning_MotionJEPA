"""探索性：PCA 维度稳健性。对核心结论在 PCA 16／32／64／128 维下重算：
(a) [t,N]（独热×100）+ 组特征读 to_press，同点跨模型 MAE 差（FrameSamp − MotionMem）；
(b) 组特征读速度 M（分类），跨模型准确率差（MotionMem − FrameSamp）。组 Text、History，层 0、8、17。其余同 hover_probe。
用法：python pca_dim_robustness.py <hover_points.json> <回放库根> <输出 json> <workers>
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
_S = {}

def task(args):
    model, group, layer, dim = args
    s = _S["s"]; X = _S["banks"][model][:, layer + 1, COL[group]].astype(np.float64)
    T = hp.time_features(s, "t_N") * 100.0
    def feats(fold):
        tr = s["outer"] != fold
        pca = hp.fit_pca(X[tr], episode_weights(s["groups"][tr]), dim=dim)
        return np.hstack([T, hp.pca_transform(pca, X)])
    y = s["labels"]["to_press"]
    p1, _, _, _ = hp.crossfit(feats, s, y[:, None], False)
    def feats2(fold):
        tr = s["outer"] != fold
        pca = hp.fit_pca(X[tr], episode_weights(s["groups"][tr]), dim=dim)
        return hp.pca_transform(pca, X)
    cl, inv = np.unique(s["M"], return_inverse=True); Y = np.eye(len(cl))[inv]
    p2, _, _, _ = hp.crossfit(feats2, s, Y, False)
    m = np.full(len(y), np.nan); k = np.isfinite(p2[:, 0]); m[k] = cl[p2[k].argmax(1)]
    return args, p1[:, 0], m

def main():
    pts, root, outp, workers = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    s = hp.load_samples(pts); banks = {}
    for m in ("ctx", "mot"):
        banks[m] = np.concatenate([np.load(root / m / f"episode_{k}.npz")["hidden_prefix"] for k in s["keys"].tolist()])
    _S.update(s=s, banks=banks)
    jobs = [(m, g, l, d) for m in ("ctx", "mot") for g in ("text", "history") for l in (0, 8, 17) for d in (16, 32, 64, 128)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = {a: (p, mm) for a, p, mm in pool.map(task, jobs, chunksize=1)}
    y = s["labels"]["to_press"]; rep = {}
    for g in ("text", "history"):
        for l in (0, 8, 17):
            for d in (16, 32, 64, 128):
                (pc, mc), (pm, mm) = res[("ctx", g, l, d)], res[("mot", g, l, d)]
                ec, em = episode_mae(y, pc, s["groups"]), episode_mae(y, pm, s["groups"])
                a = bootstrap([ec[k] - em[k] for k in sorted(set(ec) & set(em))], 2000, 42)
                ac = hp.readout_episodes(s, "M", mc, s["labels"]["M"]); am = hp.readout_episodes(s, "M", mm, s["labels"]["M"])
                keep = np.isfinite(ac) & np.isfinite(am); b = bootstrap(am[keep] - ac[keep], 2000, 42)
                rep[f"{g}|{l}|{d}"] = {"beyond_mae_ctx_minus_mot": a, "M_acc_mot_minus_ctx": b,
                                      "M_acc_ctx": float(np.nanmean(ac)), "M_acc_mot": float(np.nanmean(am))}
                print(f"{g} L{l} pca{d}: beyond {a['mean']:+.1f}[{a['ci95'][0]:.1f},{a['ci95'][1]:.1f}] | M ctx={np.nanmean(ac):.2f} mot={np.nanmean(am):.2f} diff {b['mean']:+.3f}[{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}]")
    outp.write_text(json.dumps(rep, indent=1))
    print("PCA_ROBUST_DONE")

if __name__ == "__main__":
    main()
