"""RouteStick 进度探针与新版 Fig. 4（计划 1117-routestick-progress-probe-attn 第一部分 §3、§5，第二部分 B1 第 4 行）。

主环境 CPU，只用 numpy（出图另需 matplotlib）。输入：
  * R1 的 rsprog_points.json（schema rsprog-points-1）：顶层 episodes[]（k、difficulty、L、demo_bounds…）、
    顶层 points[]（k、t、u、j（0 起）、tau、remaining、cur_move、next_move（末步 null）…）、pairs[]（k、t_a、t_b、
    front_maxdiff、wrist_maxdiff，按 H5 timestep t 对齐）、dropped[]（剔除局，本脚本排除）。默认值即 R1 实际键名，
    见 DEFAULT_FIELDS；--field 仅备用。
  * R2 的每局 npz：<replay-root>/<模型>/episode_<k>.npz（--npz-pattern 可改），本脚本读 u、hidden_prefix (N,19,2,W)
    （组 0 = image、1 = lang）、hidden_action (N,18,W_a)、hidden_memmod (N,18,W_a)、mem_frame_mean、mem_motion_mean、
    mem_motion_latest (N,1024)。点按 (k, u) 与标签表对齐；两个模型取共同点集。

方法（沿用 1105 StopCube 探针）：按局 5 折（rollout_probe.hash_fold，两模型同一映射），每局等权；特征在外层训练折内
标准化 + PCA 64 维（hover_probe.fit_pca/pca_transform），岭回归 / 一对多分类（hover_probe.ridge_path/fit_targets），
λ 在外层训练折内按局分 4 折选。逐层曲线报每一层的折外读数；「主读数层」在外层训练折内部按内层折外分数逐折选。
时间基线：执行偏移 u 独热；oracle：(u, L) 联合独热。打乱对照：同难度内整局换路线，按本局 u 重算标签，20 次。

读数：τ 用局等权 R²（每局贡献 r_e = 1 − mse_e / sst，sst 为局等权总方差，r_e 的均值恰为 R²，局级 bootstrap 对 r_e
重采样、sst 固定）；分类用局等权准确率。判定行 PROBE 只表示流程完整（无泄漏、基线与打乱对照齐全、值非空），不表示
解码成功。主读数（计划 X8）：模型 A 动作组的 τ（R²）与下一步走法（准确率）；其余都是探索性读数。
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

# 必须在 numpy 之前设置（理由见 rollout_probe 文件头：单线程 BLAS 在 aspen 上快一个数量级）。
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from scripts.interpret.analysis.hover_probe import fit_pca, fit_targets, pca_transform, ridge_path
from scripts.interpret.analysis.oracle_press_prob import bootstrap, write_json
from scripts.interpret.analysis.rollout_probe import (INNER_FOLDS, OUTER_FOLDS, SEED, SHUFFLES, episode_weights,
                                                      hash_fold, select_lambda)

__all__ = ["ridge_path", "select_lambda"]   # 与 hover_probe 同一套拟合；select_lambda 供单测对拍 λ 网格

MODELS = ("A", "B")
MODEL_NAME = {"A": "MotionMem", "B": "Frozen Wan pooling"}
MAIN_MODEL = "A"
PREFIX_LEVELS, LAYERS, LAST_LAYER = 19, 18, 17
POSITIVE_CONTROL_TOL = 0.8
POSITIVE_CONTROL_LAYER = 17
REPEATS = 2000
STEP_FRAMES = 50
DIFFICULTIES = ("easy", "medium", "hard", "xhard")

# 特征组：(组名, 层数)；prefix_cat 是前缀 image 与 lang 拼接，作 Fig. 4 的「前缀」线。
GROUPS = {"prefix_image": PREFIX_LEVELS, "prefix_lang": PREFIX_LEVELS, "prefix_cat": PREFIX_LEVELS,
          "action": LAYERS, "memmod": LAYERS, "mem_frame_mean": 1, "mem_motion_mean": 1, "mem_motion_latest": 1}
REGRESSION = ("tau", "remaining", "j")
CLASSIFICATION = ("next", "cur")
LABELS = REGRESSION + CLASSIFICATION
MAIN_LABELS = ("tau", "next")
SHUFFLE_GROUPS = ("action",)

# 默认按 R1 实际输出（schema rsprog-points-1，scripts/interpret/rsprog_points.py 模块 docstring 第二节）：
# 顶层 episodes[]、points[]（点不挂在局下）、pairs[]（用 H5 timestep t 对齐，t = e0 + u）、dropped[]（剔除局）。
DEFAULT_FIELDS = {
    "episodes": "episodes", "k": "k", "difficulty": "difficulty", "L": "L", "points": "points",
    "t": "t", "u": "u", "j": "j", "tau": "tau", "remaining": "remaining", "cur": "cur_move", "next": "next_move",
    "pairs": "pairs", "pair_k": "k", "pair_t_a": "t_a", "pair_t_b": "t_b", "dropped": "dropped",
}
PIX_TOL = 8   # 同帧异史严格档：front/wrist 逐像素最大差都 ≤ 8（R1 输出已只留过阈值的配对，这里再核一次）

_STATE = {}


# ---------------------------------------------------------------- 读入

def load_points(path, fields=None):
    """读 R1 的标签表，摊平成逐点列表；排除 dropped[] 里的局。返回 {"rows", "pairs_raw"}。

    点在顶层 points[]（每点带 k）；L 与难度取自所属局。pairs_raw 是 (k, t_a, t_b)。
    """
    f = {**DEFAULT_FIELDS, **(fields or {})}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    dropped = {int(entry[f["k"]]) for entry in data.get(f["dropped"], []) or []}
    episodes = {int(ep[f["k"]]): ep for ep in data[f["episodes"]] if int(ep[f["k"]]) not in dropped}
    rows = []
    for point in data[f["points"]]:
        k = int(point[f["k"]])
        if k not in episodes:
            continue
        ep = episodes[k]
        L, j, nxt = int(ep[f["L"]]), int(point[f["j"]]), point.get(f["next"])
        rows.append({"k": k, "difficulty": str(ep[f["difficulty"]]), "L": L, "t": int(point[f["t"]]),
                     "u": int(point[f["u"]]), "j": j, "tau": float(point[f["tau"]]),
                     "remaining": float(point.get(f["remaining"], L - j)),
                     "cur": str(point[f["cur"]]), "next": None if nxt is None else str(nxt)})
    tol = data.get("pix_tol", PIX_TOL)
    pairs = []
    for entry in data.get(f["pairs"], []) or []:
        if entry.get("front_maxdiff", 0) > tol or entry.get("wrist_maxdiff", 0) > tol:
            continue
        k = int(entry[f["pair_k"]])
        if k in episodes:
            pairs.append((k, int(entry[f["pair_t_a"]]), int(entry[f["pair_t_b"]])))
    return {"rows": rows, "pairs_raw": pairs, "dropped": sorted(dropped)}


def npz_path(replay_root, pattern, model, k):
    return Path(replay_root) / pattern.format(model=model, k=k)


def _read_bank(replay_root, pattern, model, keys):
    """读一个模型全部局；返回 {k: {字段: 数组}}，缺局跳过。"""
    bank = {}
    for k in keys:
        path = npz_path(replay_root, pattern, model, k)
        if not path.exists():
            continue
        with np.load(path) as data:
            bank[k] = {name: data[name] for name in ("u", "hidden_prefix", "hidden_action", "hidden_memmod",
                                                      "mem_frame_mean", "mem_motion_mean", "mem_motion_latest")
                       if name in data.files}
    return bank


def build_samples(points, banks):
    """按 (k, u) 把标签表与各模型回放库对齐，只留所有模型都有的点。返回 (samples, 特征库, 计数)。"""
    rows = points["rows"]
    available = None
    for bank in banks.values():
        keys = {(k, int(u)) for k, ep in bank.items() for u in ep["u"].tolist()}
        available = keys if available is None else available & keys
    available = available or set()
    keep = sorted((r for r in rows if (r["k"], r["u"]) in available), key=lambda r: (r["k"], r["u"]))
    counts = {"label_points": len(rows), "matched_points": len(keep), "dropped_episodes": len(points.get("dropped", [])),
              "episodes": len({r["k"] for r in keep}), "dropped": len(rows) - len(keep)}
    if not keep:
        raise SystemExit("找不到标签表与回放库的共同点；核对 --npz-pattern 与 (k, u)")
    keys = sorted({r["k"] for r in keep})
    index_of = {k: i for i, k in enumerate(keys)}
    groups = np.asarray([str(r["k"]) for r in keep])
    samples = {
        "keys": keys, "groups": groups, "group_index": np.asarray([index_of[r["k"]] for r in keep]),
        "k": np.asarray([r["k"] for r in keep]), "u": np.asarray([r["u"] for r in keep]),
        "L": np.asarray([r["L"] for r in keep]), "difficulty": np.asarray([r["difficulty"] for r in keep]),
        "outer": np.asarray([hash_fold(g, OUTER_FOLDS, "outer", SEED) for g in groups]),
        "inner": np.asarray([hash_fold(g, INNER_FOLDS, "inner", SEED) for g in groups]),
    }
    move_classes = sorted({r["cur"] for r in keep} | {r["next"] for r in keep if r["next"] is not None})
    code = {name: i for i, name in enumerate(move_classes)}
    labels = {name: np.asarray([r[name] for r in keep], dtype=np.float64) for name in REGRESSION}
    labels["cur"] = np.asarray([code[r["cur"]] for r in keep], dtype=np.float64)
    labels["next"] = np.asarray([np.nan if r["next"] is None else code[r["next"]] for r in keep], dtype=np.float64)
    samples["labels"] = labels
    samples["move_classes"] = move_classes
    samples["n_classes"] = len(move_classes)
    samples["moves"] = episode_moves(keep, code)
    row_of = {(r["k"], r["t"]): i for i, r in enumerate(keep)}
    pair_rows = [(row_of[(k, a)], row_of[(k, b)]) for k, a, b in points["pairs_raw"] if (k, a) in row_of and (k, b) in row_of]
    samples["pairs"] = np.asarray(pair_rows, dtype=np.int64).reshape(-1, 2)
    counts["pairs"] = len(points["pairs_raw"])
    counts["pairs_matched"] = len(pair_rows)

    features = {}
    for model, bank in banks.items():
        order = {k: {int(u): i for i, u in enumerate(ep["u"].tolist())} for k, ep in bank.items()}
        take = [(r["k"], order[r["k"]][r["u"]]) for r in keep]

        def gather(name, take=take, bank=bank):
            return np.stack([bank[k][name][i] for k, i in take]).astype(np.float32)

        prefix = gather("hidden_prefix")
        features[model] = {"prefix_image": prefix[:, :, 0], "prefix_lang": prefix[:, :, 1],
                           "action": gather("hidden_action"), "memmod": gather("hidden_memmod"),
                           "mem_frame_mean": gather("mem_frame_mean")[:, None],
                           "mem_motion_mean": gather("mem_motion_mean")[:, None],
                           "mem_motion_latest": gather("mem_motion_latest")[:, None]}
    return samples, features, counts


def episode_moves(rows, code):
    """每局各步走法（由 cur 读出；每步至少有一个点，因为步长 50 帧、取点 16 帧）。"""
    moves = {}
    for r in rows:
        moves.setdefault(r["k"], {})[r["j"]] = code[r["cur"]]
    return {k: [steps.get(j, -1) for j in range(max(steps) + 1)] for k, steps in moves.items()}


def feature_matrix(features, model, group, level):
    bank = features[model]
    if group == "prefix_cat":
        return np.hstack([bank["prefix_image"][:, level], bank["prefix_lang"][:, level]]).astype(np.float64)
    return bank[group][:, level].astype(np.float64)


def one_hot(values):
    classes, inverse = np.unique(values, return_inverse=True, axis=0)
    result = np.zeros((len(inverse), len(classes)))
    result[np.arange(len(inverse)), inverse.reshape(-1)] = 1.0
    return result


def time_features(samples, kind):
    """[u]：执行偏移独热；[u,L]：(u, L) 联合独热（τ 是两者的确定函数，所以这是 oracle）。"""
    if kind == "u":
        return one_hot(samples["u"])
    return one_hot(np.stack([samples["u"], samples["L"]], axis=1))


# ---------------------------------------------------------------- 打乱对照

def shuffled_labels(samples, shuffles=SHUFFLES, seed=SEED):
    """同难度内整局换路线（L 与逐步走法一起换），按本局 u 重算 τ、j、剩余与走法；u ≥ 50L′ 的点标签记 NaN。"""
    generator = np.random.default_rng(seed)
    keys = samples["keys"]
    difficulty_of = {k: samples["difficulty"][samples["group_index"] == i][0] for i, k in enumerate(keys)}
    L_of = {k: int(samples["L"][samples["group_index"] == i][0]) for i, k in enumerate(keys)}
    result = []
    for _ in range(shuffles):
        mapping = {}
        for difficulty in sorted(set(difficulty_of.values())):
            members = [k for k in keys if difficulty_of[k] == difficulty]
            for k, donor in zip(members, generator.permutation(members).tolist()):
                mapping[k] = donor
        labels = {name: np.full(len(samples["u"]), np.nan) for name in LABELS}
        for row, (k, u) in enumerate(zip(samples["k"].tolist(), samples["u"].tolist())):
            donor = mapping[k]
            L, moves = L_of[donor], samples["moves"][donor]
            j = u // STEP_FRAMES
            if j >= L:
                continue
            labels["tau"][row] = u / (STEP_FRAMES * L)
            labels["j"][row] = j
            labels["remaining"][row] = L - j
            labels["cur"][row] = moves[j] if j < len(moves) and moves[j] >= 0 else np.nan
            if j + 1 < L and j + 1 < len(moves) and moves[j + 1] >= 0:
                labels["next"][row] = moves[j + 1]
        result.append(labels)
    return result


# ---------------------------------------------------------------- 交叉拟合

def project_folds(X, samples, use_pca):
    """外层每折：训练折内标准化 + PCA（或原样），返回 [(训练下标, 测试下标, 训练特征, 测试特征) | None]。"""
    groups, outer = samples["groups"], samples["outer"]
    folds = []
    for fold in range(OUTER_FOLDS):
        test = outer == fold
        train = ~test
        if not test.any() or not train.any():
            folds.append(None)
            continue
        if use_pca:
            pca = fit_pca(X[train], episode_weights(groups[train]))
            folds.append((np.flatnonzero(train), np.flatnonzero(test), pca_transform(pca, X[train]), pca_transform(pca, X[test])))
        else:
            folds.append((np.flatnonzero(train), np.flatnonzero(test), X[train], X[test]))
    return folds


def group_leak(samples):
    groups, outer = samples["groups"], samples["outer"]
    return sum(len(set(groups[outer != f].tolist()) & set(groups[outer == f].tolist())) for f in range(OUTER_FOLDS))


def cv_label(folds, samples, y, kind):
    """一个标签的折外预测与逐折内层分数。y 中 NaN 的点不参与拟合与评分。返回 (oos (N,), inner_scores (折,))。"""
    groups, inner = samples["groups"], samples["inner"]
    n_classes = samples["n_classes"]
    oos = np.full(len(y), np.nan)
    scores = np.full(OUTER_FOLDS, np.nan)
    valid = np.isfinite(y)
    for fold, item in enumerate(folds):
        if item is None:
            continue
        train_idx, test_idx, train_x, test_x = item
        tr, te = valid[train_idx], valid[test_idx]
        if tr.sum() < 2:
            continue
        y_train = y[train_idx][tr]
        if kind == "class":
            Y = np.zeros((len(y_train), n_classes))
            Y[np.arange(len(y_train)), y_train.astype(int)] = 1.0
        else:
            Y = y_train[:, None]
        fitted = fit_targets(train_x[tr], Y, groups[train_idx][tr], inner[train_idx][tr],
                             test_x[te] if te.any() else test_x[:1])
        if fitted is None:
            continue
        test_pred, inner_pred = fitted
        decode = (lambda v: v.argmax(axis=1).astype(float)) if kind == "class" else (lambda v: v[:, 0])
        if te.any():
            oos[test_idx[te]] = decode(test_pred)
        inner_values = decode(inner_pred)
        g = groups[train_idx][tr]
        finite = np.isfinite(inner_pred).all(axis=1)
        scores[fold] = score(y_train[finite], inner_values[finite], g[finite], kind)
    return oos, scores


def _episode_index(groups):
    _, index = np.unique(groups, return_inverse=True)
    return index


def r2_episodes(y, prediction, groups):
    """局等权 R² 的逐局贡献 r_e（均值即 R²）。返回 (r_e 数组, 局键)。"""
    keep = np.isfinite(y) & np.isfinite(prediction)
    y, prediction, groups = y[keep], prediction[keep], groups[keep]
    if not len(y):
        return np.zeros(0), np.zeros(0)
    keys, index = np.unique(groups, return_inverse=True)
    counts = np.bincount(index)
    weights = 1.0 / counts[index]
    mean = (weights * y).sum() / weights.sum()
    sst = np.bincount(index, weights=(y - mean) ** 2)
    sst = float((sst / counts).mean())
    mse = np.bincount(index, weights=(y - prediction) ** 2) / counts
    if sst <= 0:
        return np.full(len(keys), np.nan), keys
    return 1.0 - mse / sst, keys


def accuracy_episodes(y, prediction, groups):
    keep = np.isfinite(y) & np.isfinite(prediction)
    if not keep.any():
        return np.zeros(0), np.zeros(0)
    keys, index = np.unique(groups[keep], return_inverse=True)
    counts = np.bincount(index)
    return np.bincount(index, weights=(y[keep] == prediction[keep]).astype(float)) / counts, keys


def episode_values(y, prediction, groups, kind):
    return (accuracy_episodes if kind == "class" else r2_episodes)(y, prediction, groups)


def score(y, prediction, groups, kind):
    values, _ = episode_values(np.asarray(y, float), np.asarray(prediction, float), np.asarray(groups), kind)
    return float(np.nanmean(values)) if len(values) and np.isfinite(values).any() else float("nan")


def label_kind(label):
    return "class" if label in CLASSIFICATION else "reg"


# ---------------------------------------------------------------- 任务

def run_feature(model, group, level):
    """一个 (模型, 组, 层) 的全部标签；动作组另跑打乱对照。"""
    samples = _STATE["samples"]
    X = feature_matrix(_STATE["features"], model, group, level)
    folds = project_folds(X, samples, use_pca=True)
    result = {}
    for label in LABELS:
        result[label] = cv_label(folds, samples, samples["labels"][label], label_kind(label))
    if group in SHUFFLE_GROUPS:
        result["shuffles"] = [{label: cv_label(folds, samples, labels[label], label_kind(label)) for label in MAIN_LABELS}
                              for labels in _STATE["shuffled"]]
    return result


def run_time(kind):
    samples = _STATE["samples"]
    folds = project_folds(time_features(samples, kind), samples, use_pca=False)
    result = {label: cv_label(folds, samples, samples["labels"][label], label_kind(label)) for label in LABELS}
    if kind == "u":
        result["shuffles"] = [{label: cv_label(folds, samples, labels[label], label_kind(label)) for label in MAIN_LABELS}
                              for labels in _STATE["shuffled"]]
    return result


def _run_task(task):
    began = time.monotonic()
    value = run_time(task[1]) if task[0] == "time" else run_feature(*task)
    return task, value, time.monotonic() - began


def run_tasks(tasks, workers):
    results = {}
    if workers <= 1:
        iterator = map(_run_task, tasks)
        for done, (task, value, elapsed) in enumerate(iterator, 1):
            results[task] = value
            print(f"RSPROG_PROBE_TASK done={done}/{len(tasks)} task={task} seconds={elapsed:.1f}", flush=True)
    else:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            for done, (task, value, elapsed) in enumerate(pool.imap_unordered(_run_task, tasks, chunksize=1), 1):
                results[task] = value
                print(f"RSPROG_PROBE_TASK done={done}/{len(tasks)} task={task} seconds={elapsed:.1f}", flush=True)
    return results


# ---------------------------------------------------------------- 汇总

def stat(y, prediction, groups, kind, repeats=REPEATS):
    values, _ = episode_values(y, prediction, groups, kind)
    values = values[np.isfinite(values)] if len(values) else values
    return bootstrap(values, repeats, SEED)


def diff_stat(y, left, right, groups, kind, repeats=REPEATS):
    """left − right 的逐局差（同一组局），局级 bootstrap。"""
    a, keys_a = episode_values(y, left, groups, kind)
    b, keys_b = episode_values(y, right, groups, kind)
    common = sorted(set(keys_a.tolist()) & set(keys_b.tolist()))
    da = dict(zip(keys_a.tolist(), a.tolist()))
    db = dict(zip(keys_b.tolist(), b.tolist()))
    values = np.asarray([da[k] - db[k] for k in common if np.isfinite(da[k]) and np.isfinite(db[k])])
    return bootstrap(values, repeats, SEED)


def select_layers(scores):
    """scores (层, 折)：每折取内层分最高的层；无分数的折为 None。"""
    chosen = []
    for fold in range(scores.shape[1]):
        column = scores[:, fold]
        chosen.append(int(np.nanargmax(column)) if np.isfinite(column).any() else None)
    return chosen


def compose(by_level, chosen, outer):
    result = np.full(by_level.shape[1], np.nan)
    for fold, level in enumerate(chosen):
        if level is not None:
            part = outer == fold
            result[part] = by_level[level][part]
    return result


def pair_accuracy(samples, prediction, truth, repeats=REPEATS):
    """同帧异史严格档：配对两点下一步走法都读对才算对；逐局配对准确率后局级 bootstrap。"""
    pairs = samples["pairs"]
    if not len(pairs):
        return {"n_pairs": 0, **bootstrap([], repeats, SEED)}
    a, b = pairs[:, 0], pairs[:, 1]
    keep = np.isfinite(prediction[a]) & np.isfinite(prediction[b]) & np.isfinite(truth[a]) & np.isfinite(truth[b])
    correct = (prediction[a] == truth[a]) & (prediction[b] == truth[b])
    groups = samples["groups"][a][keep]
    if not keep.any():
        return {"n_pairs": 0, **bootstrap([], repeats, SEED)}
    values, _ = accuracy_episodes(correct[keep].astype(float), np.ones(keep.sum()), groups)
    return {"n_pairs": int(keep.sum()), **bootstrap(values, repeats, SEED)}


def by_difficulty(samples, label, prediction, references):
    y, kind = samples["labels"][label], label_kind(label)
    out = {}
    for difficulty in sorted(set(samples["difficulty"].tolist()), key=lambda d: (DIFFICULTIES + (d,)).index(d)):
        mask = samples["difficulty"] == difficulty
        y_d = np.where(mask, y, np.nan)
        entry = {"value": stat(y_d, prediction, samples["groups"], kind)}
        for name, ref in references.items():
            entry[name] = stat(y_d, ref, samples["groups"], kind)
            entry[f"minus_{name}"] = diff_stat(y_d, prediction, ref, samples["groups"], kind)
        out[difficulty] = entry
    return out


def _shuffle_value(samples, shuffled, task_results, levels, label):
    """一次打乱：逐层折外预测 + 逐折选层（只看内层分），返回读数。"""
    kind = label_kind(label)
    preds = np.stack([task_results[level][label][0] for level in levels])
    scores = np.stack([task_results[level][label][1] for level in levels])
    chosen = select_layers(scores)
    prediction = compose(preds, chosen, samples["outer"])
    return score(shuffled[label], prediction, samples["groups"], kind)


def summarize(samples, results, models, counts, repeats=REPEATS):
    """把任务结果汇总成 summary 与出图用的预测。"""
    groups, outer = samples["groups"], samples["outer"]
    time_u, time_uL = results[("time", "u")], results[("time", "uL")]
    references = {"u": {label: time_u[label][0] for label in LABELS}, "uL": {label: time_uL[label][0] for label in LABELS}}
    leak = group_leak(samples)
    summary = {"plan": "1117-routestick-progress-probe-attn", "counts": counts, "group_leak": leak,
               "move_classes": samples["move_classes"], "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
               "repeats": repeats, "shuffles": len(_STATE.get("shuffled", [])), "main_model": MAIN_MODEL,
               "main_labels": list(MAIN_LABELS), "time_baseline": {}, "models": {}}
    for kind_name, res in (("u", time_u), ("uL", time_uL)):
        summary["time_baseline"][kind_name] = {
            label: stat(samples["labels"][label], res[label][0], groups, label_kind(label), repeats) for label in LABELS}
        if kind_name == "u":
            summary["time_baseline"]["u"]["pair_next"] = pair_accuracy(samples, res["next"][0], samples["labels"]["next"], repeats)
            summary["time_baseline"]["u"]["shuffled"] = {
                label: [score(shuf[label], s[label][0], groups, label_kind(label)) for shuf, s in
                        zip(_STATE["shuffled"], res["shuffles"])] for label in MAIN_LABELS}
    predictions = {"references": references, "layer": {}, "selected": {}}
    for model in models:
        entry = {}
        for group, levels in GROUPS.items():
            level_ids = list(range(levels))
            task = {level: results[(model, group, level)] for level in level_ids}
            per_label = {}
            for label in LABELS:
                kind = label_kind(label)
                y = samples["labels"][label]
                curve = [stat(y, task[level][label][0], groups, kind, repeats) for level in level_ids]
                preds = np.stack([task[level][label][0] for level in level_ids])
                scores = np.stack([task[level][label][1] for level in level_ids])
                chosen = select_layers(scores)
                selected = compose(preds, chosen, outer)
                predictions["layer"][(model, group, label)] = preds
                predictions["selected"][(model, group, label)] = selected
                block = {"curve": curve, "layer_by_fold": chosen, "selected": stat(y, selected, groups, kind, repeats),
                         "minus_u": diff_stat(y, selected, references["u"][label], groups, kind, repeats),
                         "minus_uL": diff_stat(y, selected, references["uL"][label], groups, kind, repeats)}
                if label in MAIN_LABELS:
                    block["by_difficulty"] = by_difficulty(samples, label, selected,
                                                           {"u": references["u"][label], "uL": references["uL"][label]})
                if label == "next":
                    block["pair_next"] = pair_accuracy(samples, selected, y, repeats)
                    block["pair_next_minus_u"] = None
                per_label[label] = block
            if group in SHUFFLE_GROUPS:
                per_label["shuffled"] = {}
                for label in MAIN_LABELS:
                    values = []
                    for index, shuffled in enumerate(_STATE["shuffled"]):
                        per_level = [{label: task[level]["shuffles"][index][label]} for level in level_ids]
                        values.append(_shuffle_value(samples, shuffled, per_level, level_ids, label))
                    per_label["shuffled"][label] = {"values": values, "mean": float(np.nanmean(values)) if values else None}
            entry[group] = per_label
        if "action" in entry:
            pc = entry["action"]["cur"]["curve"][POSITIVE_CONTROL_LAYER]["mean"]
            entry["positive_control"] = {"layer": POSITIVE_CONTROL_LAYER, "acc": pc,
                                         "pass": pc is not None and pc >= POSITIVE_CONTROL_TOL}
            pair_model = entry["action"]["next"]["pair_next"]
            pair_base = summary["time_baseline"]["u"]["pair_next"]
            entry["action"]["next"]["pair_next_minus_u"] = (
                None if pair_model["mean"] is None or pair_base["mean"] is None else pair_model["mean"] - pair_base["mean"])
        summary["models"][model] = entry
    return summary, predictions


def verdicts(summary):
    lines = [f"GROUP_LEAK={'PASS' if summary['group_leak'] == 0 else 'FAIL'} leak={summary['group_leak']}"]
    for model, entry in summary["models"].items():
        pc = entry.get("positive_control")
        if pc is not None:
            acc = "nan" if pc["acc"] is None else f"{pc['acc']:.3f}"
            lines.append(f"POSITIVE_CONTROL={'PASS' if pc['pass'] else 'FAIL'} model={model} layer={pc['layer']} acc={acc}")
        for label in MAIN_LABELS:
            block = entry["action"][label]
            base = summary["time_baseline"]["u"][label]
            shuffled = entry["action"].get("shuffled", {}).get(label, {}).get("values", [])
            ok = (summary["group_leak"] == 0 and block["selected"]["mean"] is not None and base["mean"] is not None
                  and math.isfinite(block["selected"]["mean"]) and len(shuffled) == summary["shuffles"] and summary["shuffles"] > 0)
            value = block["selected"]["mean"]
            lines.append(f"PROBE={'PASS' if ok else 'FAIL'} model={model} label={label} group=action "
                         f"value={_fmt(value)} minus_u={_fmt(block['minus_u']['mean'])} "
                         f"shuffle_mean={_fmt(np.nanmean(shuffled) if shuffled else None)}"
                         + (" main=1" if model == MAIN_MODEL else " main=0"))
    return lines


def _fmt(value):
    return "nan" if value is None or not np.isfinite(value) else f"{value:.3f}"


# ---------------------------------------------------------------- 出图

PLOT_GROUPS = (("prefix_cat", "Prefix (image+text)", "#1baf7a", 2.2),
               ("memmod", "Memory modulation", "#eb6834", 2.2),
               ("action", "Action", "#2a78d6", 3.4))
INK, MUTED, GRID = "#1d1d1d", "#555555", "#dddddd"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans", "axes.unicode_minus": False})
    return plt


def _clean(axis):
    for name in ("top", "right"):
        axis.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        axis.spines[name].set_color(MUTED)
    axis.tick_params(colors=MUTED)


def _x_of(group, levels):
    """前缀第 0 级是进第 0 层前的输入，画在 x=−1；层输出画在层号上。"""
    return np.arange(levels) - (1 if levels == PREFIX_LEVELS else 0)


def _draw_layerwise(axes, summary, models):
    base = summary["time_baseline"]["u"]["tau"]["mean"]
    handles = {}
    for axis, model in zip(axes, models):
        entry = summary["models"][model]
        for group, name, color, width in PLOT_GROUPS:
            curve = [c["mean"] if c["mean"] is not None else np.nan for c in entry[group]["tau"]["curve"]]
            x = _x_of(group, len(curve))
            (line,) = axis.plot(x, curve, color=color, linewidth=width, label=name, solid_capstyle="round")
            axis.plot(x[-1], curve[-1], "o", color=color, markersize=4.5, markeredgecolor="white", markeredgewidth=0.8)
            handles[name] = line
        if base is not None:
            handles["Time baseline [u]"] = axis.axhline(base, color="#888888", linestyle="--", linewidth=1.6,
                                                       label="Time baseline [u]")
        axis.axhline(0, color=GRID, linewidth=1.0, zorder=0)
        axis.set_title(MODEL_NAME.get(model, model), fontsize=12, fontweight="bold", color=INK)
        axis.set_xlabel("Layer", color=MUTED)
        axis.set_xticks([0, 5, 10, 15])
        _clean(axis)
    axes[0].set_ylabel(r"Probe $R^2$ (progress $\tau$)", color=MUTED)
    return handles


def _fit_line(x, y):
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2 or np.ptp(x[keep]) == 0:
        return None
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    return slope, intercept


def _draw_fit(axes, samples, predictions, models, titles=False):
    y = samples["labels"]["tau"]
    handles = {}
    for column, (axis, model) in enumerate(zip(axes, models)):
        for group, name, color, width in PLOT_GROUPS:
            preds = predictions["layer"][(model, group, "tau")]
            prediction = preds[-1]
            if group == "action":
                axis.scatter(y, prediction, s=5, color=color, alpha=0.12, linewidths=0, zorder=1)
            fit = _fit_line(y, prediction)
            if fit is not None:
                (line,) = axis.plot([0, 1], [fit[1], fit[0] + fit[1]], color=color, linewidth=width, label=name, zorder=3)
                handles[name] = line
        axis.plot([0, 1], [0, 1], color="#888888", linestyle=":", linewidth=1.5, zorder=2)
        if column == 0:
            axis.text(0.62, 0.9, "$y = x$", color=MUTED, fontsize=10, style="italic", transform=axis.transAxes)
        axis.set_xlim(0, 1)
        axis.set_ylim(-0.1, 1.1)
        axis.set_xlabel(r"True progress $\tau$", color=MUTED)
        if titles:
            axis.set_title(MODEL_NAME.get(model, model), fontsize=12, fontweight="bold", color=INK)
        _clean(axis)
    axes[0].set_ylabel("Predicted (layer 17)", color=MUTED)
    return handles


def plot_figures(samples, summary, predictions, output):
    """新版 Fig. 4：progress-fig4（上下两排），另把上排、下排各存一张。返回写出的路径。"""
    plt = _plt()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODELS if m in summary["models"]] or list(summary["models"])
    written = []

    def save(figure, stem):
        for suffix in ("png", "pdf"):
            path = output / f"{stem}.{suffix}"
            figure.savefig(path, dpi=200 if suffix == "png" else None, bbox_inches="tight")
            written.append(path)
        plt.close(figure)

    def legend(figure, handles, top):
        figure.legend(list(handles.values()), list(handles.keys()), loc="upper center", ncol=len(handles), frameon=False,
                      bbox_to_anchor=(0.5, top), handlelength=1.8, columnspacing=1.2, fontsize=9.5)

    figure, axes = plt.subplots(2, len(models), figsize=(3.3 * len(models) + 0.4, 6.2), squeeze=False)
    for column in range(1, len(models)):
        axes[0, column].sharey(axes[0, 0])
        axes[1, column].sharey(axes[1, 0])
    handles = _draw_layerwise(axes[0], summary, models)
    _draw_fit(axes[1], samples, predictions, models)
    for column in range(1, len(models)):
        axes[0, column].tick_params(labelleft=False)
        axes[1, column].tick_params(labelleft=False)
    figure.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.6)
    legend(figure, handles, 1.0)
    save(figure, "progress-fig4")

    figure, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models) + 0.4, 3.4), squeeze=False, sharey=True)
    handles = _draw_layerwise(axes[0], summary, models)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    legend(figure, handles, 1.0)
    save(figure, "progress-layerwise")

    figure, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models) + 0.4, 3.4), squeeze=False, sharey=True)
    handles = _draw_fit(axes[0], samples, predictions, models, titles=True)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    legend(figure, handles, 1.0)
    save(figure, "progress-fit")
    return written


# ---------------------------------------------------------------- 入口

def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def run(points_path, replay_root, output, models=MODELS, workers=1, fields=None, pattern="{model}/episode_{k}.npz",
        shuffles=SHUFFLES, repeats=REPEATS, plot=True):
    points = load_points(points_path, fields)
    keys = sorted({r["k"] for r in points["rows"]})
    banks = {}
    for model in models:
        bank = _read_bank(replay_root, pattern, model, keys)
        if bank:
            banks[model] = bank
        else:
            print(f"RSPROG_PROBE 找不到模型 {model} 的回放库，跳过", flush=True)
    if not banks:
        raise SystemExit("找不到任何回放库")
    samples, features, counts = build_samples(points, banks)
    print(f"RSPROG_PROBE_DATA points={counts['matched_points']} episodes={counts['episodes']} "
          f"dropped={counts['dropped']} pairs={counts['pairs_matched']}/{counts['pairs']}", flush=True)
    _STATE.clear()
    _STATE.update(samples=samples, features=features, shuffled=shuffled_labels(samples, shuffles))
    tasks = [("time", "u"), ("time", "uL")]
    tasks += [(model, group, level) for model in banks for group, levels in GROUPS.items() for level in range(levels)]
    results = run_tasks(tasks, workers)
    summary, predictions = summarize(samples, results, list(banks), counts, repeats)
    lines = verdicts(summary)
    summary["verdicts"] = lines
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "rsprog-probe-summary.json", _jsonable(summary))
    if plot:
        summary["figures"] = [str(p) for p in plot_figures(samples, summary, predictions, output)]
    for line in lines:
        print(line, flush=True)
    return summary, predictions, samples


def _parse_fields(items):
    fields = {}
    for item in items or []:
        name, _, value = item.partition("=")
        if not value or name not in DEFAULT_FIELDS:
            raise SystemExit(f"--field 格式为 名=json键，名取自 {sorted(DEFAULT_FIELDS)}：{item}")
        fields[name] = value
    return fields


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--points", required=True, help="R1 的 rsprog_points.json")
    parser.add_argument("--replay-root", required=True, help="R2 回放根目录")
    parser.add_argument("--output", required=True, help="输出目录（JSON 与图）")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--npz-pattern", default="{model}/episode_{k}.npz", help="相对 replay-root 的每局 npz 路径模板")
    parser.add_argument("--field", action="append", help="字段名覆盖，如 next=next_move；可重复")
    parser.add_argument("--pairs-key", default=None, help="同帧异史配对表在 JSON 顶层的键（默认 pairs）")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shuffles", type=int, default=SHUFFLES)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    fields = _parse_fields(args.field)
    if args.pairs_key:
        fields["pairs"] = args.pairs_key
    run(args.points, args.replay_root, args.output, tuple(args.models), args.workers, fields, args.npz_pattern,
        args.shuffles, args.repeats, plot=not args.no_plot)


if __name__ == "__main__":
    main()
