"""探索性：旧闭环 rollout（各模型自己的轨迹，Q4 样本集）上逐层读速度 M（分类，一对多岭、训练折内 PCA64、按局 5 折、每局等权）。
复用 closedloop_beyond_oracle.make_samples。用法：python closedloop_speed.py <rollout 产物根> <输出 json> <workers>
"""
import json, multiprocessing, os, sys
from pathlib import Path
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from closedloop_beyond_oracle import make_samples, COL
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_attention import load_align

_S = {}

def acc(s, pred):
    keep = np.isfinite(pred)
    return float(np.nanmean(hp.episode_mean_array(pred == s["M"], keep, s)))

def task(args):
    model, group, layer = args
    s, prefix, action = _S[model]
    X = (action[:, layer] if group == "action" else prefix[:, layer + 1, COL[group]]).astype(np.float64)
    classes, inv = np.unique(s["M"], return_inverse=True); Y = np.eye(len(classes))[inv]
    pred, _, leak, _ = hp.crossfit(lambda f: X, s, Y, True)
    out = np.full(len(inv), np.nan); k = np.isfinite(pred[:, 0]); out[k] = classes[pred[k].argmax(1)]
    return args, acc(s, out), leak

def main():
    root, outp, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    align = load_align(root); jobs = []; base = {}
    for model in ("ctx", "mot"):
        s, prefix, action, motion = make_samples(model, root, align["models"][model]); _S[model] = (s, prefix, action)
        gs = ["text", "current", "history"] + (["motion"] if motion else [])
        jobs += [(model, g, l) for g in gs for l in range(-1, 18)] + [(model, "action", l) for l in range(18)]
        ep_M = {k: s["M"][s["groups"] == k][0] for k in s["keys"]}
        vals, cnt = np.unique(list(ep_M.values()), return_counts=True); base[model] = float(cnt.max() / cnt.sum())
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = pool.map(task, jobs, chunksize=1)
    rep = {"majority_by_episode": base, "cells": {f"{m}|{g}|{l}": a for (m, g, l), a, _ in res}, "leak": int(sum(r[2] for r in res))}
    outp.write_text(json.dumps(rep, indent=1))
    print("leak", rep["leak"], "majority", base)
    for m in ("ctx", "mot"):
        for g in ("text", "current", "history", "motion", "action"):
            ls = [l for l in range(-1, 18) if f"{m}|{g}|{l}" in rep["cells"]]
            if ls:
                print(m, g, " ".join(f"{rep['cells'][f'{m}|{g}|{l}']:.2f}" for l in ls))
    print("CL_SPEED_DONE")

if __name__ == "__main__":
    main()
