"""探索性：三个模型 Text 组读 to_press 的逐层折外 R²，按 N≥4 分档评估（探针在全部点上训练），PCA 维度 16／32／64／128（分折种子固定 42）。
检查「N≥4 时每层都是 FrameSamp < concat < split」在不同分折下是否成立。用法：python three_model_by_N_pca_dim.py <hover_points.json> <输出 json> <workers>
"""
import json, multiprocessing, os, sys
from pathlib import Path
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import metrics, hash_fold, OUTER_FOLDS, INNER_FOLDS

ROOT = Path("v1-store/evaluation/stopcube-common-fig4-1108")
MODELS = {"FrameSamp": ROOT / "replay/ctx", "concat": ROOT / "replay/mot", "split": ROOT / "replay-split/mot"}
SEEDS = (16, 32, 64, 128)  # 此处复用为 PCA 维度
_S = {}

def task(args):
    name, layer, seed = args
    s = dict(_S["s"])
    dim = seed
    X = _S["banks"][name][:, layer + 1, 3].astype(np.float64)
    y = s["labels"]["to_press"]
    from scripts.interpret.analysis.rollout_probe import episode_weights
    def feats(fold):
        tr = s["outer"] != fold
        pca = hp.fit_pca(X[tr], episode_weights(s["groups"][tr]), dim=dim)
        return hp.pca_transform(pca, X)
    p, _, _, _ = hp.crossfit(feats, s, y[:, None], False)
    hi = s["N"] >= 4
    return args, metrics(y, np.where(hi, p[:, 0], np.nan), s["groups"])["r2"]

def main():
    pts, outp, workers = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    s = hp.load_samples(pts)
    banks = {n: np.concatenate([np.load(d / f"episode_{k}.npz")["hidden_prefix"] for k in s["keys"].tolist()]) for n, d in MODELS.items()}
    _S.update(s=s, banks=banks)
    jobs = [(n, l, sd) for n in MODELS for l in range(0, 18) for sd in SEEDS]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = dict(pool.map(task, jobs, chunksize=1))
    ordered = 0; total = 0; viol = []
    for l in range(0, 18):
        for sd in SEEDS:
            a, b, c = res[("FrameSamp", l, sd)], res[("concat", l, sd)], res[("split", l, sd)]
            total += 1
            if a < b < c:
                ordered += 1
            else:
                viol.append((l, sd, round(a, 3), round(b, 3), round(c, 3)))
    outp.write_text(json.dumps({"cells": {f"{n}|{l}|{sd}": v for (n, l, sd), v in res.items()}, "ordered": ordered, "total": total, "violations": viol}, indent=1))
    print(f"N>=4 ordered FrameSamp<concat<split: {ordered}/{total}")
    for v in viol:
        print("  violation layer", v[0], "seed", v[1], "FS/concat/split", v[2:])
    print("BYN_PCA_DONE")

if __name__ == "__main__":
    main()
