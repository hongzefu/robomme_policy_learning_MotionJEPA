"""StopCube 悬停期逐层探针与出图（计划 1105-stopcube-hover-probe 的 A4–A9、B5）。

主环境 CPU，只用 numpy。输入是 W1 的 hover_points.json 和 W2 的回放库
<replay-root>/<ctx|mot>/episode_<k>.npz（键 t、hidden_prefix、group_tokens）。

方法：按局 5 折、每局等权。特征组在外层训练折内标准化再 PCA 到 64 维，PCA 得分不再标准化，
所以这里另写只中心化的岭路径（闭式解和 λ 网格同 rollout_probe）；λ 在外层训练折内按局分 4 折选，
内层不重算主成分。分类一对多，每类各选 λ，预测取最大。时间基线 [t]、[t,N] 用独热，不经 PCA。
单报一层时在外层训练折内部按验证结果选层，每折可以不同。

判定只保证流程（无泄漏、对照齐全、结果非空），不代表解码成功；结论只到「可解码」。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

# 必须在 numpy 之前设置，理由见 rollout_probe 文件头。
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from scripts.interpret.analysis.oracle_press_prob import bootstrap, write_json
from scripts.interpret.analysis.rollout_attention import GRID, INK, MODEL_STYLE, MUTED, clean_axis, setup_plot
from scripts.interpret.analysis.rollout_probe import (INNER_FOLDS, LAMBDAS, LINE_COLOR, OUTER_FOLDS, REPEATS, SEED, SHUFFLES,
                                                      episode_mae, episode_weights, fit_standardizer, hash_fold,
                                                      mae_difference, metrics)
from scripts.interpret.hover_points import MOVE_INTERVALS, point_labels


# 第 3 维下标 → 特征组。基线的 motion 与 latest_motion 两列是空的，由 group_tokens 判定后跳过。
FEATURES = ("history", "motion", "current", "text", "latest_motion")
FEATURE_COLUMN = dict(zip(FEATURES, range(5)))
FEATURE_NAME = {"history": "历史视觉", "motion": "motion", "current": "当前视觉", "text": "文字",
                "latest_motion": "最近 motion token"}
FEATURE_COLOR = {key: LINE_COLOR[FEATURE_NAME[key]] for key in FEATURES[:4]}
FEATURE_COLOR["latest_motion"] = LINE_COLOR["动作 token"]

MODEL_ORDER = ("mot", "ctx")
LAYERS = 18
INPUT_INDEX = 0
LAST_LAYER = 17
PCA_DIM = 64
POSITIVE_CONTROL_TOL = 0.8
CURVE_FRACTIONS = (0.25, 0.5, 1.0)
WINDOW_STRIDE = 32
PROGRESS_BIN = 16

CLASS_LABELS = ("M", "c", "press", "remaining")
REGRESSION_LABELS = ("to_press", "cube_pos")
LABELS = CLASS_LABELS + REGRESSION_LABELS
LABEL_NAME = {"M": "速度 M", "c": "已过次数", "press": "该不该按", "remaining": "剩余次数", "to_press": "距按下步数",
              "cube_pos": "方块位置"}
READOUTS = ("M", "c", "c_strict", "press", "remaining", "to_press", "cube_pos")
READOUT_LABEL = {name: name for name in LABELS} | {"c_strict": "c"}
# 预先写死的两个主读数（A7）：特征组 → (标签, 读数)。只在 motion 模型上有。
MAIN_MODEL = "mot"
MAIN_READOUTS = {"latest_motion": ("M", "M"), "motion": ("c", "c_strict")}
# 打乱只对主读数和时间基线；时间基线额外覆盖逐层图的三行。
TIME_SHUFFLE_READOUTS = ("M", "c_strict", "press")
TIME_SHUFFLE_LABELS = ("M", "c", "press")
PROGRESS_FEATURE = {"mot": "motion", "ctx": "history"}

# 计划 A7 按 100 局 2344 个点算出的两组时刻；数据是这份全量时必须对上。
PLAN_POINTS = 2344
PLAN_COMMON_T = sorted(set(range(56, 233, 8)) - {88, 120, 152, 200, 208})
PLAN_SAME_TIME_T = [56, 96, 104, 112, 128, 136, 144, 160, 168, 176, 184, 192, 216, 224, 232]

_STATE = {}


# ---------- 读入 ----------

def load_samples(points_path):
    """把 hover_points.json 摊平成按点对齐的数组；没有点的局不进入。"""
    with open(points_path, encoding="utf-8") as handle:
        episodes = json.load(handle)["episodes"]
    groups, ts, Ms, Ns, labels = [], [], [], [], {name: [] for name in LABELS}
    position, pending = {}, []
    for key in sorted(episodes, key=int):
        item = episodes[key]
        for point in item["points"]:
            index = len(ts)
            position[(int(key), point["t"])] = index
            groups.append(int(key))
            ts.append(point["t"])
            Ms.append(item["M"])
            Ns.append(item["N"])
            labels["M"].append(item["M"])
            for name in ("c", "press", "remaining", "to_press", "cube_pos"):
                labels[name].append(point[name])
            if point["strict"] is True:
                pending.append((index, int(key), point["pair_t"]))
    pairs = np.asarray([(index, position[(key, paired)]) for index, key, paired in pending], dtype=np.int64).reshape(-1, 2)
    groups = np.asarray(groups, dtype=np.int64)
    keys, group_index = np.unique(groups, return_inverse=True)
    outer = {int(key): hash_fold(int(key), OUTER_FOLDS, "outer") for key in keys}
    inner = {int(key): hash_fold(int(key), INNER_FOLDS, "inner") for key in keys}
    return {"groups": groups, "keys": keys, "group_index": group_index,
            "t": np.asarray(ts, dtype=np.int64), "M": np.asarray(Ms, dtype=np.int64), "N": np.asarray(Ns, dtype=np.int64),
            "labels": {name: np.asarray(values, dtype=np.float64) for name, values in labels.items()},
            "pairs": pairs,
            "outer": np.asarray([outer[int(key)] for key in groups]), "inner": np.asarray([inner[int(key)] for key in groups])}


def load_bank(replay_root, label, samples):
    """读一个模型的回放库。t 必须与标签表逐点相同；一列要么整列为空（不参与），要么整列有内容。"""
    prefix, tokens = [], []
    for key in samples["keys"].tolist():
        with np.load(Path(replay_root) / label / f"episode_{key}.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["t"], samples["t"][samples["groups"] == key]):
                raise ValueError(f"回放的 t 与标签表不符：{label} 第 {key} 局")
            prefix.append(np.asarray(archive["hidden_prefix"], dtype=np.float32))
            tokens.append(np.asarray(archive["group_tokens"]))
    prefix, tokens = np.concatenate(prefix), np.concatenate(tokens)
    active = []
    for feature in FEATURES:
        present = tokens[:, FEATURE_COLUMN[feature]] > 0
        if present.all():
            active.append(feature)
        elif present.any():
            raise ValueError(f"{label} 的 {feature} 列只有部分点有 token：{int(present.sum())}/{len(present)}")
    return {"prefix": prefix, "active": active}


def time_sets(samples):
    """三种速度都有点的 t，以及其中次数不全相同的 t（同时刻档）。"""
    common, same_time = [], []
    for t in np.unique(samples["t"]).tolist():
        here = samples["t"] == t
        if set(samples["M"][here].tolist()) >= set(MOVE_INTERVALS):
            common.append(t)
            if len(np.unique(samples["labels"]["c"][here])) > 1:
                same_time.append(t)
    return common, same_time


def check_plan_times(samples, common, same_time):
    """全量数据必须对上计划列出的时刻；合成数据点数不同，不核对。返回是否核对过。"""
    if len(samples["t"]) != PLAN_POINTS:
        return False
    if common != PLAN_COMMON_T or same_time != PLAN_SAME_TIME_T:
        raise ValueError(f"时刻集合与计划不符：common={common} same_time={same_time}")
    return True


def time_features(samples, kind):
    """时间基线：t 独热，kind='t_N' 时再拼 N 独热。"""
    columns = [_one_hot(samples["t"])]
    if kind == "t_N":
        columns.append(_one_hot(samples["N"]))
    return np.hstack(columns)


def _one_hot(values):
    classes, inverse = np.unique(values, return_inverse=True)
    result = np.zeros((len(values), len(classes)))
    result[np.arange(len(values)), inverse] = 1.0
    return result


def shuffled_params(samples, shuffles=SHUFFLES, seed=SEED):
    """整局换 (M,N)，按本局 t 用 point_labels 重算；超出原类别的值原样保留，不裁剪不丢点。"""
    generator = np.random.default_rng(seed)
    keys = samples["keys"]
    first = {int(key): int(np.flatnonzero(samples["groups"] == key)[0]) for key in keys}
    result = []
    for _ in range(shuffles):
        donor = keys[generator.permutation(len(keys))]
        mapping = {int(key): first[int(other)] for key, other in zip(keys, donor)}
        labels = {name: np.empty(len(samples["t"])) for name in LABELS}
        for index, (key, t) in enumerate(zip(samples["groups"].tolist(), samples["t"].tolist())):
            source = mapping[key]
            M, N = int(samples["M"][source]), int(samples["N"][source])
            labels["M"][index] = M
            for name, value in point_labels(t, M, N).items():
                labels[name][index] = value
        result.append(labels)
    return result


# ---------- 拟合 ----------

def fit_pca(X, weights, dim=PCA_DIM):
    """外层训练折内标准化 + 加权 PCA。样本数小于维度时走 Gram 矩阵。返回 (mean, std, components (D,k))。"""
    mean, std = fit_standardizer(X, weights)
    scaled = (X - mean) / std * np.sqrt(weights)[:, None]
    if scaled.shape[0] < scaled.shape[1]:
        values, vectors = np.linalg.eigh(scaled @ scaled.T)
    else:
        values, vectors = np.linalg.eigh(scaled.T @ scaled)
    order = np.argsort(values)[::-1][:dim]
    values, vectors = values[order], vectors[:, order]
    keep = values > max(values.max(initial=0.0), 0.0) * 1e-10
    values, vectors = values[keep], vectors[:, keep]
    if scaled.shape[0] < scaled.shape[1]:
        vectors = scaled.T @ vectors / np.sqrt(values)
    return mean, std, vectors


def pca_transform(pca, X):
    mean, std, components = pca
    return (X - mean) / std @ components


def ridge_path(X, Y, weights, lambdas=LAMBDAS):
    """多目标岭路径，只中心化不标准化。返回 (mean, betas (Λ,D,K), intercept (K,))。

    闭式解同 rollout_probe.fit_path；维度大于样本数时走对偶形式。
    """
    total = weights.sum()
    mean = (weights[:, None] * X).sum(axis=0) / total
    intercept = (weights[:, None] * Y).sum(axis=0) / total
    Z, centered = X - mean, Y - intercept
    if Z.shape[1] > Z.shape[0]:
        root = np.sqrt(weights)[:, None]
        scaled = Z * root
        values, vectors = np.linalg.eigh(scaled @ scaled.T)
        basis = scaled.T @ vectors
        projected = vectors.T @ (root * centered)
    else:
        values, vectors = np.linalg.eigh((Z * weights[:, None]).T @ Z)
        basis = vectors
        projected = vectors.T @ (Z.T @ (weights[:, None] * centered))
    values = np.clip(values, 0.0, None)
    shrunk = projected[None] / (values[None, :, None] + np.asarray(lambdas)[:, None, None])
    return mean, basis @ shrunk, intercept


def predict_path(path, X):
    mean, betas, intercept = path
    return (X - mean) @ betas + intercept


def fit_targets(train_x, Y, groups, inner, test_x, lambdas=LAMBDAS):
    """每个目标在训练折内按局分内层折选 λ。返回 (测试折预测, 训练折内层折外预测)；拆不出内层折时返回 None。"""
    weights = episode_weights(groups)
    inner_pred = np.full((len(lambdas), *Y.shape), np.nan)
    errors, used = np.zeros((len(lambdas), Y.shape[1])), 0
    for fold in np.unique(inner):
        hold = inner == fold
        if hold.all():
            continue
        values = predict_path(ridge_path(train_x[~hold], Y[~hold], weights[~hold], lambdas), train_x[hold])
        errors += (weights[hold][None, :, None] * (values - Y[hold][None]) ** 2).sum(axis=1)
        inner_pred[:, hold] = values
        used += 1
    if not used:
        return None
    best = errors.argmin(axis=0)
    columns = np.arange(Y.shape[1])
    test = predict_path(ridge_path(train_x, Y, weights, lambdas), test_x)
    return test[best, :, columns].T, inner_pred[best, :, columns].T


def build_targets(samples, shuffled=None, shuffle_labels=()):
    """把全部标签摊成一张目标矩阵，同一次分解求解。返回 (Y, layout)，layout 的键是 (标签, 打乱序号或 None)。"""
    columns, layout, start = [], {}, 0

    def add(name, shuffle, values):
        nonlocal start
        if name in CLASS_LABELS:
            classes, inverse = np.unique(values, return_inverse=True)
            block = np.zeros((len(values), len(classes)))
            block[np.arange(len(values)), inverse] = 1.0
        else:
            classes, block = None, values[:, None]
        columns.append(block)
        layout[(name, shuffle)] = (start, start + block.shape[1], classes)
        start += block.shape[1]

    for name in LABELS:
        add(name, None, samples["labels"][name])
    for name in shuffle_labels:
        for index, labels in enumerate(shuffled):
            add(name, index, labels[name])
    return np.hstack(columns), layout


def decode(values, entry):
    """分类取最大，回归取值；没有预测的行保持 NaN。"""
    start, stop, classes = entry
    if classes is None:
        return values[:, start]
    result = np.full(len(values), np.nan)
    keep = np.isfinite(values[:, start])
    result[keep] = classes[values[keep, start:stop].argmax(axis=1)]
    return result


def crossfit(features_for_fold, samples, Y, use_pca, train_filter=None):
    """外层 5 折。features_for_fold(fold) 返回该折用的特征矩阵，None 表示跳过这一折。"""
    groups, outer = samples["groups"], samples["outer"]
    prediction = np.full(Y.shape, np.nan)
    inner_values, leak, train_episodes = {}, 0, []
    for fold in range(OUTER_FOLDS):
        test = outer == fold
        if not test.any():
            continue
        train = ~test if train_filter is None else train_filter(~test)
        leak += len(set(groups[train].tolist()) & set(groups[test].tolist()))
        X = features_for_fold(fold)
        if X is None or not train.any():
            continue
        train_episodes.append(len(np.unique(groups[train])))
        if use_pca:
            pca = fit_pca(X[train], episode_weights(groups[train]))
            train_x, test_x = pca_transform(pca, X[train]), pca_transform(pca, X[test])
        else:
            train_x, test_x = X[train], X[test]
        fitted = fit_targets(train_x, Y[train], groups[train], samples["inner"][train], test_x)
        if fitted is None:
            continue
        prediction[test] = fitted[0]
        values = np.full(Y.shape, np.nan)
        values[train] = fitted[1]
        inner_values[fold] = values
    return prediction, inner_values, leak, train_episodes


# ---------- 读数定义 ----------

def episode_mean_array(values, keep, samples):
    """逐局均值，下标同 samples['keys']；没有点的局是 NaN。"""
    size = len(samples["keys"])
    index = samples["group_index"][keep]
    counts = np.bincount(index, minlength=size)
    sums = np.bincount(index, weights=np.asarray(values, dtype=np.float64)[keep], minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts


def pair_correct(prediction, truth, pairs):
    """同帧配对：t 和 t+2M 都读对才算对。返回 (对否, 两点都有预测)。"""
    first, second = pairs[:, 0], pairs[:, 1]
    keep = np.isfinite(prediction[first]) & np.isfinite(prediction[second])
    return (prediction[first] == truth[first]) & (prediction[second] == truth[second]), keep


def pair_episode_array(prediction, truth, samples):
    """严格档的逐局配对准确率。"""
    pairs = samples["pairs"]
    correct, keep = pair_correct(prediction, truth, pairs)
    size = len(samples["keys"])
    index = samples["group_index"][pairs[:, 0]][keep]
    counts = np.bincount(index, minlength=size)
    sums = np.bincount(index, weights=correct[keep].astype(np.float64), minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts


def readout_episodes(samples, readout, prediction, truth):
    """每局一个分数（越大越好），下标同 samples['keys']；选层和打乱对照共用。"""
    if readout == "c_strict":
        return pair_episode_array(prediction, truth, samples)
    keep = np.isfinite(prediction)
    if READOUT_LABEL[readout] in REGRESSION_LABELS:
        return episode_mean_array(-np.abs(prediction - truth), keep, samples)
    return episode_mean_array(prediction == truth, keep, samples)


def readout_value(samples, readout, prediction, truth):
    values = readout_episodes(samples, readout, prediction, truth)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def nan_to_none(value):
    return None if value is None or not np.isfinite(value) else float(value)


def as_dict(values, samples):
    return {int(key): float(value) for key, value in zip(samples["keys"], values) if np.isfinite(value)}


def difference(left, right):
    """left−right 的局级区间；差值先按局相减，只比两边都有的局。"""
    shared = sorted(set(left) & set(right))
    return bootstrap([left[key] - right[key] for key in shared], REPEATS, SEED)


def accuracy_block(samples, prediction, truth, mask=None, references=None):
    mask = np.ones(len(truth), dtype=bool) if mask is None else mask
    keep = np.isfinite(prediction) & mask
    per_episode = as_dict(episode_mean_array(prediction == truth, keep, samples), samples)
    result = {"value": float(np.mean(list(per_episode.values()))) if per_episode else None,
              "n_episodes": len(per_episode), "n_samples": int(keep.sum()), "n_no_prediction": int((mask & ~keep).sum())}
    for name, reference in (references or {}).items():
        other = as_dict(episode_mean_array(reference == truth, np.isfinite(reference) & mask, samples), samples)
        result[f"minus_{name}"] = difference(per_episode, other)
    return result


def strict_block(samples, prediction, truth, references=None):
    per_episode = as_dict(pair_episode_array(prediction, truth, samples), samples)
    result = {"value": float(np.mean(list(per_episode.values()))) if per_episode else None,
              "n_episodes": len(per_episode), "n_pairs": len(samples["pairs"])}
    for name, reference in (references or {}).items():
        result[f"minus_{name}"] = difference(per_episode, as_dict(pair_episode_array(reference, truth, samples), samples))
    return result


def regression_block(samples, prediction, truth, references=None):
    groups = samples["groups"]
    result = metrics(truth, prediction, groups)
    per_episode = episode_mae(truth, prediction, groups)
    for name, reference in (references or {}).items():
        result[f"minus_{name}"] = mae_difference(per_episode, episode_mae(truth, reference, groups), REPEATS, SEED)
    return result


def speed_block(samples, prediction, truth, references, common):
    result = accuracy_block(samples, prediction, truth, references=references)
    result["by_M"] = {str(M): accuracy_block(samples, prediction, truth, samples["M"] == M, references)
                      for M in MOVE_INTERVALS}
    result["common_t"] = accuracy_block(samples, prediction, truth, np.isin(samples["t"], common), references)
    return result


def same_time_block(samples, prediction, truth, same_time, baseline):
    """同时刻档：按 t 分别报准确率，并按 M 拆开；baseline 是 [t] 基线在该 t 的预测。"""
    result = {}
    for t in same_time:
        here = samples["t"] == t
        row = {"n": int(here.sum()), "value": _point_accuracy(prediction, truth, here), "by_M": {}}
        if baseline is not None:
            row["time"] = _point_accuracy(baseline, truth, here)
            row["time_by_M"] = {}
        for M in MOVE_INTERVALS:
            part = here & (samples["M"] == M)
            row["by_M"][str(M)] = _point_accuracy(prediction, truth, part)
            if baseline is not None:
                row["time_by_M"][str(M)] = _point_accuracy(baseline, truth, part)
        result[str(t)] = row
    return result


def _point_accuracy(prediction, truth, mask):
    keep = mask & np.isfinite(prediction)
    return float((prediction[keep] == truth[keep]).mean()) if keep.any() else None


def window_block(samples, prediction, truth, references):
    """B5：窗口每 32 步更新，t % 32 就是距最近窗口更新的步数。"""
    phase = samples["t"] % WINDOW_STRIDE
    return {str(value): accuracy_block(samples, prediction, truth, phase == value, references)
            for value in np.unique(phase).tolist()}


def label_references(references, label, extra=None, compare=None):
    """取某个标签的对照预测：时间基线，外加同层的当前视觉或文字组。"""
    result = {kind: predictions[label] for kind, predictions in (references or {}).items()}
    if compare and extra in compare:
        result[extra] = compare[extra][label]
    return result


def summarize_cell(samples, predictions, times, references=None, compare=None, window=False):
    """一个（特征组, 层）的全部读数。references 是 {基线名: 各标签预测}；compare 是同层当前视觉、文字组的预测。"""
    truth = samples["labels"]
    common, same_time = times
    base = {label: label_references(references, label) for label in LABELS}
    cell = {
        "M": speed_block(samples, predictions["M"], truth["M"], base["M"], common),
        "c": {"full": accuracy_block(samples, predictions["c"], truth["c"],
                                     references=label_references(references, "c", "current", compare)),
              "strict": strict_block(samples, predictions["c"], truth["c"], base["c"]),
              "same_time": same_time_block(samples, predictions["c"], truth["c"], same_time, base["c"].get("t"))},
        "press": accuracy_block(samples, predictions["press"], truth["press"],
                                references=label_references(references, "press", "text", compare)),
        "remaining": accuracy_block(samples, predictions["remaining"], truth["remaining"],
                                    references=label_references(references, "remaining", "text", compare)),
        "to_press": regression_block(samples, predictions["to_press"], truth["to_press"], base["to_press"]),
        "cube_pos": regression_block(samples, predictions["cube_pos"], truth["cube_pos"], base["cube_pos"]),
    }
    if window:
        cell["c"]["by_window_update"] = window_block(samples, predictions["c"], truth["c"], base["c"])
    return cell


def readout_block(samples, readout, prediction, references, common):
    """单个读数的结果块，给按训练折选出来的层用。"""
    truth = samples["labels"][READOUT_LABEL[readout]]
    if readout == "M":
        return speed_block(samples, prediction, truth, references, common)
    if readout == "c_strict":
        return strict_block(samples, prediction, truth, references)
    if readout in REGRESSION_LABELS:
        return regression_block(samples, prediction, truth, references)
    return accuracy_block(samples, prediction, truth, references=references)


def select_layers(scores):
    """scores (层, 折)：每折取内层验证分最高的层；这一折没有分数时为 None。"""
    chosen = []
    for fold in range(scores.shape[1]):
        column = scores[:, fold]
        chosen.append(int(np.nanargmax(column)) if np.isfinite(column).any() else None)
    return chosen


def compose_prediction(by_layer, layer_by_fold, outer):
    """每折用自己选的层的折外预测拼起来。"""
    result = np.full(by_layer.shape[-1], np.nan)
    for fold, layer in enumerate(layer_by_fold):
        if layer is not None:
            part = outer == fold
            result[part] = by_layer[layer][part]
    return result


def most_common_layer(layer_by_fold):
    """判定行只能写一个层号：取折间最常见的，并列取小的；每折的层在 JSON 里。"""
    counts = Counter(layer for layer in layer_by_fold if layer is not None)
    return min(counts, key=lambda layer: (-counts[layer], layer)) if counts else None


def curve_filter(samples, fraction):
    """外层训练局按局编号的哈希排序取前若干比例。"""

    def apply(train):
        keys = np.unique(samples["groups"][train]).tolist()
        keys.sort(key=lambda key: hashlib.sha256(f"{SEED}:curve:{key}".encode()).hexdigest())
        keep = keys[:math.ceil(len(keys) * fraction)]
        return train & np.isin(samples["groups"], keep)

    return apply


# ---------- 任务（多进程）----------

def run_probe_task(model, feature, index):
    samples, shuffled = _STATE["samples"], _STATE["shuffled"]
    X = _STATE["banks"][model]["prefix"][:, index, FEATURE_COLUMN[feature]].astype(np.float64)
    main = MAIN_READOUTS.get(feature) if model == MAIN_MODEL and index != INPUT_INDEX else None
    return evaluate(lambda fold: X, samples, shuffled, True, (main[0],) if main else (), (main[1],) if main else ())


def run_time_task(kind):
    samples, shuffled = _STATE["samples"], _STATE["shuffled"]
    X = time_features(samples, kind)
    return evaluate(lambda fold: X, samples, shuffled, False, TIME_SHUFFLE_LABELS if kind == "t" else (), ())


def run_curve_task(feature, readout, fraction, layer_by_fold):
    """学习曲线用 100% 训练时选出的层（选层只看过训练折），主成分在缩减后的训练局上重算。"""
    samples = _STATE["samples"]
    prefix = _STATE["banks"][MAIN_MODEL]["prefix"]
    column = FEATURE_COLUMN[feature]

    def features(fold):
        layer = layer_by_fold[fold]
        return None if layer is None else prefix[:, layer + 1, column].astype(np.float64)

    Y, layout = build_targets(samples)
    prediction, _, leak, train_episodes = crossfit(features, samples, Y, True, curve_filter(samples, fraction))
    label = READOUT_LABEL[readout]
    decoded = decode(prediction, layout[(label, None)])
    return {"value": nan_to_none(readout_value(samples, readout, decoded, samples["labels"][label])),
            "n_train_episodes_mean": float(np.mean(train_episodes)) if train_episodes else 0.0, "leak": leak}


def evaluate(features_for_fold, samples, shuffled, use_pca, shuffle_labels, shuffle_readouts):
    """跑一组特征：返回各标签折外预测、各读数的内层验证分，以及打乱标签上的同样两项。"""
    Y, layout = build_targets(samples, shuffled, shuffle_labels)
    prediction, inner_values, leak, _ = crossfit(features_for_fold, samples, Y, use_pca)
    result = {"pred": {name: decode(prediction, layout[(name, None)]) for name in LABELS}, "leak": leak, "inner": {},
              "shuffle_pred": {}, "shuffle_inner": {}}
    for readout in READOUTS:
        label = READOUT_LABEL[readout]
        result["inner"][readout] = _inner_scores(samples, readout, inner_values, layout[(label, None)], samples["labels"][label])
    for label in shuffle_labels:
        result["shuffle_pred"][label] = np.stack([decode(prediction, layout[(label, index)]) for index in range(len(shuffled))])
    for readout in shuffle_readouts:
        label = READOUT_LABEL[readout]
        result["shuffle_inner"][readout] = np.stack(
            [_inner_scores(samples, readout, inner_values, layout[(label, index)], shuffled[index][label])
             for index in range(len(shuffled))])
    return result


def _inner_scores(samples, readout, inner_values, entry, truth):
    scores = np.full(OUTER_FOLDS, np.nan)
    for fold, values in inner_values.items():
        scores[fold] = readout_value(samples, readout, decode(values, entry), truth)
    return scores


def run_task(task):
    began = time.monotonic()
    kind, *arguments = task
    function = {"probe": run_probe_task, "time": run_time_task, "curve": run_curve_task}[kind]
    return task, function(*arguments), time.monotonic() - began


def run_tasks(tasks, workers, stage):
    """按任务多进程。fork 共享只读的隐状态库，不重复占内存。"""
    results, seconds = {}, []

    def collect(iterator):
        for done, (task, value, elapsed) in enumerate(iterator, 1):
            results[task] = value
            seconds.append(elapsed)
            print(f"HOVER_PROBE_TASK stage={stage} done={done}/{len(tasks)} seconds={elapsed:.1f}", flush=True)

    if workers <= 1:
        collect(map(run_task, tasks))
    else:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            collect(pool.imap_unordered(run_task, tasks, chunksize=1))
    return results, seconds


# ---------- 汇总 ----------

def analyze(samples, banks, workers, smoke, shuffles=SHUFFLES):
    """跑全部任务并汇总成 summary。返回 (summary, 给出图用的预测)。"""
    _STATE.update(samples=samples, banks=banks, shuffled=shuffled_params(samples, shuffles))
    shuffled = _STATE["shuffled"]
    common, same_time = time_sets(samples)
    checked = check_plan_times(samples, common, same_time)
    times = (common, same_time)

    tasks = [("time", "t"), ("time", "t_N")]
    for model, bank in banks.items():
        for feature in bank["active"]:
            if smoke and not (model == MAIN_MODEL and feature in MAIN_READOUTS):
                continue
            tasks.extend(("probe", model, feature, layer + 1) for layer in range(LAYERS))
            if not smoke:
                tasks.append(("probe", model, feature, INPUT_INDEX))
    results, seconds = run_tasks(tasks, workers, "probe")
    leak = sum(value["leak"] for value in results.values())

    time_results = {kind: results[("time", kind)] for kind in ("t", "t_N")}
    references = {kind: time_results[kind]["pred"] for kind in time_results}
    time_baseline = {kind: summarize_cell(samples, references[kind], times, window=True) for kind in references}
    time_shuffle = _time_shuffle(samples, shuffled, time_results["t"])
    summary = {"schema_version": 1, "smoke": smoke, "n_points": len(samples["t"]), "n_episodes": len(samples["keys"]),
               "n_strict_pairs": len(samples["pairs"]),
               "method": {"folds": {"outer": OUTER_FOLDS, "inner": INNER_FOLDS, "seed": SEED, "rule": "sha256(seed:盐:局编号) 取模"},
                          "pca_dim": PCA_DIM, "lambdas": LAMBDAS.tolist(),
                          "bootstrap": {"repeats": REPEATS, "seed": SEED, "unit": "episode"}, "shuffles": len(shuffled)},
               "times": {"common_t": common, "same_time_t": same_time, "checked_against_plan": checked},
               "time_baseline": {**time_baseline, "shuffle_t": time_shuffle}, "models": {},
               "note": "流程判定不代表解码成功；结论只到「可解码」。方块位置的 minus_* 是 MAE 差。主读数以外都是探索性。"}

    selected_prediction = {}
    for model, bank in banks.items():
        features = {}
        for feature in bank["active"]:
            if ("probe", model, feature, 1) not in results:
                continue
            features[feature] = _summarize_feature(samples, results, model, feature, references, times, smoke,
                                                   selected_prediction)
        summary["models"][model] = {"active_features": list(features), "features": features}

    curve_tasks = []
    for feature, (_, readout) in MAIN_READOUTS.items():
        layers = summary["models"][MAIN_MODEL]["features"][feature]["selected"][readout]["layer_by_fold"]
        curve_tasks.extend(("curve", feature, readout, fraction, tuple(layers)) for fraction in CURVE_FRACTIONS)
    curves, curve_seconds = run_tasks(curve_tasks, workers, "curve")
    for task, value in curves.items():
        leak += value["leak"]
        _, feature, readout, fraction, _ = task
        target = summary["models"][MAIN_MODEL]["features"][feature]
        target.setdefault("learning_curve", {}).setdefault(readout, []).append(
            {"fraction": fraction, "value": value["value"], "n_train_episodes_mean": value["n_train_episodes_mean"]})
    for feature, (_, readout) in MAIN_READOUTS.items():
        summary["models"][MAIN_MODEL]["features"][feature]["learning_curve"][readout].sort(key=lambda item: item["fraction"])

    summary["group_leak"] = leak
    summary["task_seconds"] = {"probe_tasks": len(seconds), "probe_mean": float(np.mean(seconds)),
                               "curve_tasks": len(curve_seconds), "curve_mean": float(np.mean(curve_seconds))}
    return summary, {"selected": selected_prediction, "references": references}


def _summarize_feature(samples, results, model, feature, references, times, smoke, selected_prediction):
    outer = samples["outer"]

    def cell(index):
        compare = {name: results[("probe", model, name, index)]["pred"] for name in ("current", "text")
                   if ("probe", model, name, index) in results}
        return summarize_cell(samples, results[("probe", model, feature, index)]["pred"], times, references, compare,
                              window=model == MAIN_MODEL)

    entry = {"name": FEATURE_NAME[feature], "layers": [cell(layer + 1) for layer in range(LAYERS)], "selected": {}}
    if not smoke:
        entry["input_level"] = cell(INPUT_INDEX)
    per_layer = [results[("probe", model, feature, layer + 1)] for layer in range(LAYERS)]
    for readout in READOUTS:
        label = READOUT_LABEL[readout]
        layer_by_fold = select_layers(np.stack([item["inner"][readout] for item in per_layer]))
        prediction = compose_prediction(np.stack([item["pred"][label] for item in per_layer]), layer_by_fold, outer)
        selected_prediction[(model, feature, readout)] = prediction
        entry["selected"][readout] = {"layer": most_common_layer(layer_by_fold), "layer_by_fold": layer_by_fold,
                                      **readout_block(samples, readout, prediction, label_references(references, label), times[0])}
    if model == MAIN_MODEL and feature in MAIN_READOUTS:
        label, readout = MAIN_READOUTS[feature]
        entry["shuffle"] = {readout: _probe_shuffle(samples, _STATE["shuffled"], per_layer, label, readout)}
    return entry


def _probe_shuffle(samples, shuffled, per_layer, label, readout):
    """打乱对照也走同样的训练折内选层，否则「18 层取最好」的乐观偏差会被算到真实读数头上。"""
    outer = samples["outer"]
    scores = np.stack([item["shuffle_inner"][readout] for item in per_layer])
    by_layer = np.stack([item["shuffle_pred"][label] for item in per_layer])
    values = []
    for index, labels in enumerate(shuffled):
        layers = select_layers(scores[:, index])
        prediction = compose_prediction(by_layer[:, index], layers, outer)
        values.append(nan_to_none(readout_value(samples, readout, prediction, labels[label])))
    finite = [value for value in values if value is not None]
    return {"n": len(shuffled), "values": values, "mean": float(np.mean(finite)) if finite else None,
            "max": max(finite) if finite else None}


def _time_shuffle(samples, shuffled, result):
    entry = {}
    for readout in TIME_SHUFFLE_READOUTS:
        label = READOUT_LABEL[readout]
        values = [nan_to_none(readout_value(samples, readout, result["shuffle_pred"][label][index], labels[label]))
                  for index, labels in enumerate(shuffled)]
        finite = [value for value in values if value is not None]
        entry[readout] = {"values": values, "mean": float(np.mean(finite)) if finite else None}
    return entry


# ---------- 判定 ----------

def verdicts(summary):
    """返回 (判定行, 是否全部 PASS)。PROBE 的 PASS 只表示流程完整，不表示解码成功。"""
    lines, passed = [], True
    leak = summary["group_leak"]
    lines.append(f"GROUP_LEAK={'PASS' if leak == 0 else 'FAIL'} leak={leak}")
    passed &= leak == 0
    if not summary["smoke"]:
        for model, item in summary["models"].items():
            r2 = item["features"]["current"]["layers"][LAST_LAYER]["cube_pos"]["r2"]
            good = r2 is not None and r2 >= POSITIVE_CONTROL_TOL
            passed &= good
            lines.append(f"POSITIVE_CONTROL={'PASS' if good else 'FAIL'} r2={_number(r2)} tol={POSITIVE_CONTROL_TOL} model={model}")
    for feature, (_, readout) in MAIN_READOUTS.items():
        entry = summary["models"][MAIN_MODEL]["features"][feature]
        selected = entry["selected"][readout]
        gap = selected["minus_t"]
        shuffled = entry["shuffle"][readout]["mean"]
        complete = (leak == 0 and selected["value"] is not None and selected["layer"] is not None
                    and gap["mean"] is not None and shuffled is not None
                    and summary["time_baseline"]["shuffle_t"][readout]["mean"] is not None)
        passed &= complete
        interval = gap["ci95"] or [None, None]
        lines.append(f"PROBE={'PASS' if complete else 'FAIL'} model={MAIN_MODEL} label={readout} feature={feature} "
                     f"layer={selected['layer']} value={_number(selected['value'])} minus_time={_number(gap['mean'])} "
                     f"ci=[{_number(interval[0])},{_number(interval[1])}] shuffled={_number(shuffled)}")
    return lines, passed


def _number(value):
    return "null" if value is None else f"{value:.4f}"


# ---------- 出图 ----------

ROWS = (("M", "速度 M", lambda cell: cell["M"]["value"]),
        ("c_strict", "已过次数（严格档）", lambda cell: cell["c"]["strict"]["value"]),
        ("press", "该不该按", lambda cell: cell["press"]["value"]))


def plot_layerwise(summary, output):
    """逐层准确率：行是标签，列是模型；水平线是 [t] 基线和整局换参数对照。"""
    plt = setup_plot(output)
    models = [model for model in MODEL_ORDER if model in summary["models"]]
    figure, axes = plt.subplots(len(ROWS), len(models), figsize=(5.8 * len(models) + 0.6, 3.1 * len(ROWS)), squeeze=False,
                                sharex=True, sharey=True)
    handles = {}
    baseline, shuffle_time = summary["time_baseline"]["t"], summary["time_baseline"]["shuffle_t"]
    for row, (readout, title, getter) in enumerate(ROWS):
        for column, model in enumerate(models):
            axis = axes[row, column]
            for feature in FEATURES:
                entry = summary["models"][model]["features"].get(feature)
                if entry is None:
                    continue
                values = [getter(cell) for cell in entry["layers"]]
                line, = axis.plot(range(LAYERS), [np.nan if value is None else value for value in values],
                                  color=FEATURE_COLOR[feature], linewidth=1.5, label=FEATURE_NAME[feature])
                handles.setdefault(FEATURE_NAME[feature], line)
            value = getter(baseline)
            if value is not None:
                handles.setdefault("[t] 基线", axis.axhline(value, color=INK, linestyle="--", linewidth=1.2))
            value = shuffle_time[readout]["mean"]
            if value is not None:
                handles.setdefault("整局换参数·[t] 基线", axis.axhline(value, color=MUTED, linestyle=":", linewidth=1.4))
            if model == MAIN_MODEL:
                for feature, (_, main) in MAIN_READOUTS.items():
                    if main == readout and feature in summary["models"][model]["features"]:
                        value = summary["models"][model]["features"][feature]["shuffle"][main]["mean"]
                        if value is not None:
                            handles.setdefault("整局换参数·主读数探针",
                                               axis.axhline(value, color=MUTED, linestyle="-.", linewidth=1.2))
            axis.set_title(f"{MODEL_STYLE[model][0]} · {title}", loc="left", fontsize=10.5, color=INK)
            axis.set_ylim(0, 1.03)
            axis.set_xticks([*range(0, LAYERS - 1, 2), LAST_LAYER])
            axis.grid(color=GRID, linewidth=0.7)
            clean_axis(axis)
            if column == 0:
                axis.set_ylabel("折外准确率", color=MUTED)
            if row == len(ROWS) - 1:
                axis.set_xlabel("层号（0～17）", color=MUTED)
    figure.legend(handles=list(handles.values()), labels=list(handles), loc="upper center", ncol=4, frameon=False, fontsize=9.5)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = Path(output) / "hover-layerwise.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def binned_curve(y, prediction):
    """按真实距按下步数分箱取均值，箱宽 16 步。"""
    keep = np.isfinite(prediction)
    bins = np.floor(y[keep] / PROGRESS_BIN).astype(np.int64)
    centers, means = [], []
    for value in np.unique(bins).tolist():
        centers.append((value + 0.5) * PROGRESS_BIN)
        means.append(float(prediction[keep][bins == value].mean()))
    return centers, means


def plot_progress(samples, summary, predictions, output):
    """距按下步数：折外预测对真值；两个模型并排，另画时间基线的预测。"""
    plt = setup_plot(output)
    models = [model for model in MODEL_ORDER if model in summary["models"]]
    figure, axes = plt.subplots(1, len(models), figsize=(5.8 * len(models), 4.8), squeeze=False, sharex=True, sharey=True)
    y = samples["labels"]["to_press"]
    for column, model in enumerate(models):
        axis = axes[0, column]
        feature = PROGRESS_FEATURE[model]
        prediction = predictions["selected"][(model, feature, "to_press")]
        selected = summary["models"][model]["features"][feature]["selected"]["to_press"]
        name, color, _ = MODEL_STYLE[model]
        axis.scatter(y, prediction, s=7, color=color, alpha=0.18, linewidths=0)
        axis.plot(*binned_curve(y, prediction), color=color, linewidth=2,
                  label=f"{FEATURE_NAME[feature]}（选层，MAE={_number(selected['mae'])}）")
        for kind, style, title in (("t", "--", "[t] 基线"), ("t_N", ":", "[t,N] 基线")):
            mae = summary["time_baseline"][kind]["to_press"]["mae"]
            axis.plot(*binned_curve(y, predictions["references"][kind]["to_press"]), color=INK, linestyle=style, linewidth=1.3,
                      label=f"{title}（MAE={_number(mae)}）")
        low, high = float(y.min()), float(y.max())
        axis.plot([low, high], [low, high], color=MUTED, linewidth=0.8, label="预测等于真值")
        axis.set_title(name, loc="left", fontsize=10.5, color=INK)
        axis.set_xlabel("真实距按下步数", color=MUTED)
        if column == 0:
            axis.set_ylabel("折外预测（步）", color=MUTED)
        axis.grid(color=GRID, linewidth=0.7)
        clean_axis(axis)
        axis.legend(frameon=False, fontsize=9, loc="upper left")
    figure.tight_layout()
    path = Path(output) / "hover-progress.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def same_time_table(samples, summary, predictions):
    """固定 t 下三种速度的已过次数准确率：真实次数、motion 组、基线的历史视觉组、[t] 基线。"""
    truth = samples["labels"]["c"]
    sources = {model: (predictions["selected"][(model, PROGRESS_FEATURE[model], "c")], PROGRESS_FEATURE[model])
               for model in summary["models"]}
    rows = []
    for t in summary["times"]["same_time_t"]:
        row = {"t": t, "c_by_M": {}, "time_by_M": {}, "models": {model: {} for model in sources}}
        for M in MOVE_INTERVALS:
            part = (samples["t"] == t) & (samples["M"] == M)
            row["c_by_M"][str(M)] = int(truth[part][0])
            row["time_by_M"][str(M)] = _point_accuracy(predictions["references"]["t"]["c"], truth, part)
            for model, (prediction, _) in sources.items():
                row["models"][model][str(M)] = _point_accuracy(prediction, truth, part)
        rows.append(row)
    return {"features": {model: source[1] for model, source in sources.items()}, "rows": rows}


def plot_same_time(table, output):
    plt = setup_plot(output)
    models = [model for model in MODEL_ORDER if model in table["features"]]
    columns = ["t"] + [f"真实次数\nM={M}" for M in MOVE_INTERVALS]
    for model in models:
        columns += [f"{MODEL_STYLE[model][0]}·{FEATURE_NAME[table['features'][model]]}\nM={M}" for M in MOVE_INTERVALS]
    columns += [f"[t] 基线\nM={M}" for M in MOVE_INTERVALS]
    cells = []
    for row in table["rows"]:
        cell = [str(row["t"])] + [str(row["c_by_M"][str(M)]) for M in MOVE_INTERVALS]
        for model in models:
            cell += [_short(row["models"][model][str(M)]) for M in MOVE_INTERVALS]
        cell += [_short(row["time_by_M"][str(M)]) for M in MOVE_INTERVALS]
        cells.append(cell)
    if not cells:
        cells = [["—"] * len(columns)]
    figure, axis = plt.subplots(figsize=(1.02 * len(columns), 0.34 * len(cells) + 1.0))
    axis.axis("off")
    widths = [0.035] + [0.055] * 3 + [(1 - 0.035 - 0.165) / (len(columns) - 4)] * (len(columns) - 4)
    drawn = axis.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center", colWidths=widths)
    drawn.auto_set_font_size(False)
    drawn.set_fontsize(7.5)
    drawn.scale(1, 1.35)
    for (row, _), cell in drawn.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#eeeeea")
            cell.set_height(cell.get_height() * 1.9)
    path = Path(output) / "hover-sametime.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _short(value):
    return "—" if value is None else f"{value:.2f}"


# ---------- 入口 ----------

def run(points, replay_root, output, workers, smoke=False, shuffles=SHUFFLES):
    began = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples = load_samples(points)
    labels = (MAIN_MODEL,) if smoke else MODEL_ORDER
    banks = {label: load_bank(replay_root, label, samples) for label in labels}
    summary, predictions = analyze(samples, banks, workers, smoke, shuffles)
    lines, passed = verdicts(summary)
    summary["verdicts"] = lines
    if not smoke:
        summary["same_time_table"] = same_time_table(samples, summary, predictions)
    write_json(output / "hover-summary.json", summary)
    if not smoke:
        plot_layerwise(summary, output)
        plot_progress(samples, summary, predictions, output)
        plot_same_time(summary["same_time_table"], output)
    return lines, passed, time.monotonic() - began


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", type=Path, required=True, help="hover_points.json")
    parser.add_argument("--replay-root", type=Path, required=True, help="回放库根，含 ctx/ 与 mot/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--smoke", action="store_true", help="只做主读数，不出图，用于估时")
    args = parser.parse_args(argv)
    lines, passed, seconds = run(args.points, args.replay_root, args.output, args.workers, args.smoke)
    for line in lines:
        print(line, flush=True)
    print(f"HOVER_PROBE_DONE seconds={seconds:.1f}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
