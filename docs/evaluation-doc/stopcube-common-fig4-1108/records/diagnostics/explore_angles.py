"""探索性：在共同示教轨迹的回放库上，逐层逐组保存两个模型的折外预测，并算跨模型配对比较。

只读 1105 的 hover_points.json 与回放库（或本轮带 hidden_action 的新库），只用 CPU。
方法完全复用 hover_probe：按局 5 折、训练折内标准化 + PCA 64、内层按局 4 折选 λ、每局等权。
选层：每折按训练折内层折外分数选层（同 hover_probe.select_layers），不看测试折。
跨模型差：两模型同点同折，按局先相减再局级重采样（bootstrap 2000 次，seed 42）。

这是探索性分析，不是预先登记的主读数；结论须标「探索性」。

用法：python explore_angles.py <hover_points.json> <回放库根> <输出目录> <workers>
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.oracle_press_prob import bootstrap
from scripts.interpret.analysis.rollout_probe import episode_mae, metrics

REPEATS, SEED = 2000, 42
GROUPS = {"history": 0, "motion": 1, "current": 2, "text": 3, "latest_motion": 4}
READS = ("M", "c", "c_strict", "press", "remaining", "to_press", "cube_pos")
_S = {}


def load(points, root):
    samples = hp.load_samples(points)
    banks = {}
    for model in ("ctx", "mot"):
        prefix, tokens, action = [], [], []
        for key in samples["keys"].tolist():
            with np.load(Path(root) / model / f"episode_{key}.npz", allow_pickle=False) as z:
                assert np.array_equal(z["t"], samples["t"][samples["groups"] == key])
                prefix.append(z["hidden_prefix"].astype(np.float32))
                tokens.append(z["group_tokens"])
                if "hidden_action" in z.files:
                    action.append(z["hidden_action"].astype(np.float32))
        tokens = np.concatenate(tokens)
        bank = {"prefix": np.concatenate(prefix), "active": [g for g, c in GROUPS.items() if (tokens[:, c] > 0).all()]}
        if action and len(action) == len(prefix):
            bank["action"] = np.concatenate(action)
            bank["active"].append("action")
        banks[model] = bank
    return samples, banks


def features(model, group, layer):
    """layer=-1 为输入层（prefix 第 0 项）；动作组没有输入层。"""
    bank = _S["banks"][model]
    if group == "action":
        return bank["action"][:, layer]
    return bank["prefix"][:, layer + 1, GROUPS[group]]


def task(args):
    model, group, layer = args
    samples, Y, layout = _S["samples"], _S["Y"], _S["layout"]
    X = features(model, group, layer).astype(np.float64)
    pred, inner, leak, _ = hp.crossfit(lambda fold: X, samples, Y, True)
    out = {label: hp.decode(pred, layout[(label, None)]) for label in hp.LABELS}
    inner_dec = {fold: {label: hp.decode(v, layout[(label, None)]) for label in hp.LABELS} for fold, v in inner.items()}
    return (model, group, layer), out, inner_dec, leak


def score(samples, readout, prediction, mask):
    truth = samples["labels"][hp.READOUT_LABEL[readout]]
    p = np.where(mask, prediction, np.nan)
    return hp.readout_episodes(samples, readout, p, truth)


def per_episode(samples, readout, prediction):
    """每局分数，越大越好（回归取 −|误差|）。"""
    return hp.readout_episodes(samples, readout, prediction, samples["labels"][hp.READOUT_LABEL[readout]])


def headline(samples, readout, prediction):
    truth = samples["labels"][hp.READOUT_LABEL[readout]]
    if readout in hp.REGRESSION_LABELS:
        m = metrics(truth, prediction, samples["groups"])
        return {"r2": m["r2"], "mae": m["mae"]}
    return {"acc": hp.readout_value(samples, readout, prediction, truth)}


def main():
    points, root, out_dir, workers = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, banks = load(points, root)
    Y, layout = hp.build_targets(samples)
    _S.update(samples=samples, banks=banks, Y=Y, layout=layout)
    jobs = []
    for model, bank in banks.items():
        for group in bank["active"]:
            layers = range(18) if group == "action" else range(-1, 18)
            jobs += [(model, group, layer) for layer in layers]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        results = pool.map(task, jobs, chunksize=1)
    preds, inners, leak = {}, {}, 0
    for key, out, inner, lk in results:
        preds[key], inners[key] = out, inner
        leak += lk
    # 时间基线
    refs = {}
    for kind in ("t", "t_N"):
        pred, _, _, _ = hp.crossfit(lambda fold, k=kind: hp.time_features(samples, k), samples, Y, False)
        refs[kind] = {label: hp.decode(pred, layout[(label, None)]) for label in hp.LABELS}
    np.savez_compressed(out_dir / "predictions.npz", **{
        f"{m}|{g}|{l}|{lab}": v for (m, g, l), out in preds.items() for lab, v in out.items()},
        **{f"ref|{k}|{lab}": v for k, o in refs.items() for lab, v in o.items()})

    outer = samples["outer"]
    report = {"note": "探索性；按局 5 折；选层在训练折内", "group_leak": leak, "n_points": int(len(samples["t"])),
              "refs": {k: {r: headline(samples, r, o[hp.READOUT_LABEL[r]]) for r in READS} for k, o in refs.items()},
              "layers": {}, "selected": {}, "cross_model": {}}
    for (m, g, l), out in sorted(preds.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        report["layers"].setdefault(f"{m}|{g}", {})[str(l)] = {r: headline(samples, r, out[hp.READOUT_LABEL[r]]) for r in READS}
    selected_pred = {}
    for model, bank in banks.items():
        for group in bank["active"]:
            layers = list(range(18)) if group == "action" else list(range(-1, 18))
            for r in READS:
                lab = hp.READOUT_LABEL[r]
                scores = np.full((len(layers), hp.OUTER_FOLDS), np.nan)
                for i, l in enumerate(layers):
                    for fold, dec in inners[(model, group, l)].items():
                        train = outer != fold
                        vals = score(samples, r, dec[lab], train)
                        scores[i, fold] = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
                chosen = hp.select_layers(scores)
                by_layer = np.stack([preds[(model, group, l)][lab] for l in layers])
                pred = hp.compose_prediction(by_layer, chosen, outer)
                selected_pred[(model, group, r)] = pred
                report["selected"].setdefault(f"{model}|{group}", {})[r] = {
                    **headline(samples, r, pred), "layer_by_fold": [None if c is None else layers[c] for c in chosen]}
    # 跨模型：同组名配对，加「每个模型最好组」的配对
    for g in ("text", "current", "history", "action"):
        for r in READS:
            a, b = selected_pred.get(("mot", g, r)), selected_pred.get(("ctx", g, r))
            if a is None or b is None:
                continue
            ea, eb = per_episode(samples, r, a), per_episode(samples, r, b)
            keep = np.isfinite(ea) & np.isfinite(eb)
            report["cross_model"].setdefault(g, {})[r] = {"mot_minus_ctx": bootstrap(ea[keep] - eb[keep], REPEATS, SEED),
                                                          "unit": "回归为 −MAE 差（正 = mot 误差小），分类为准确率差"}
    # mot 记忆组（history/motion/latest）对 ctx history
    for g in ("motion", "latest_motion"):
        for r in READS:
            a, b = selected_pred.get(("mot", g, r)), selected_pred.get(("ctx", "history", r))
            if a is None or b is None:
                continue
            ea, eb = per_episode(samples, r, a), per_episode(samples, r, b)
            keep = np.isfinite(ea) & np.isfinite(eb)
            report["cross_model"].setdefault(f"mot_{g}_vs_ctx_history", {})[r] = {
                "mot_minus_ctx": bootstrap(ea[keep] - eb[keep], REPEATS, SEED)}
    (out_dir / "angles.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"EXPLORE_DONE points={report['n_points']} jobs={len(jobs)} leak={leak}", flush=True)


if __name__ == "__main__":
    main()
