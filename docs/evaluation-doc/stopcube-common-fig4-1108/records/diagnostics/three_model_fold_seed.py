"""探索性：三个模型（FrameSamp、MotionMem concat、MotionMem split）Text 组读 to_press 的折外 R²，固定层 0／8／17，5 个分折种子（42、1、7、123、2026）。
方法同 hover_probe（训练折内标准化 + PCA64、内层选 λ、每局等权）。用法：python three_model_fold_seed.py <hover_points.json> <输出 json> <workers>
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
SEEDS = (42, 1, 7, 123, 2026)
_S = {}

def task(args):
    name, layer, seed = args
    s = dict(_S["s"])
    s["outer"] = np.asarray([hash_fold(int(k), OUTER_FOLDS, "outer", seed=seed) for k in s["groups"]])
    s["inner"] = np.asarray([hash_fold(int(k), INNER_FOLDS, "inner", seed=seed) for k in s["groups"]])
    X = _S["banks"][name][:, layer + 1, 3].astype(np.float64)
    y = s["labels"]["to_press"]
    p, _, _, _ = hp.crossfit(lambda f: X, s, y[:, None], True)
    return args, metrics(y, p[:, 0], s["groups"])["r2"]

def main():
    pts, outp, workers = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    s = hp.load_samples(pts)
    banks = {n: np.concatenate([np.load(d / f"episode_{k}.npz")["hidden_prefix"] for k in s["keys"].tolist()]) for n, d in MODELS.items()}
    _S.update(s=s, banks=banks)
    jobs = [(n, l, sd) for n in MODELS for l in (0, 8, 17) for sd in SEEDS]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = dict(pool.map(task, jobs, chunksize=1))
    rep = {f"{n}|{l}": [res[(n, l, sd)] for sd in SEEDS] for n in MODELS for l in (0, 8, 17)}
    outp.write_text(json.dumps(rep, indent=1))
    for l in (0, 8, 17):
        print(f"L{l}: " + " | ".join(f"{n} {min(rep[f'{n}|{l}']):.3f}–{max(rep[f'{n}|{l}']):.3f}" for n in MODELS))
    print("THREE_SEED_DONE")

if __name__ == "__main__":
    main()
