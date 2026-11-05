"""StopCube 真实 rollout 的 Q3、Q4 线性探针（1105 计划 K5、K6、K9，§三.5）。

主环境 CPU，只用 numpy。样本是每模型每次到位后的推理：
  Q3 y=trigger_step−chunk_start_env_step，局取 uses.q3 且已触发，只取 infer_seq≤I_trigger；
  Q4 y=window_center−chunk_start_env_step（可为负），局取 uses.q4，取到本局结束。
岭回归闭式解，维度大于样本数时走对偶形式。每个样本按 1/该局样本数加权，外层 5 折按局划分，两个模型共用
episode→折映射；正则系数只在训练折内部按局分 4 折选择，标准化只用训练折。

判据只保证流程（PROBE_REPORT）；科学结论不由代码下，只能到「可解码」。

⚠ 打乱标签的含义：整局换的是该局的标签锚点（触发步或窗口中心），即 y′=另一局的锚点−t。推理在同一 16 步网格上，
这等于把另一局的整条标签序列按同一 t 对齐搬过来。t 本身不被打乱，所以读得出 t 的特征在打乱后 R² 仍可能明显
大于 0，这不是过拟合；summary 同时给出时间基线在同一份打乱标签上的 R²，两者要一起读。
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

# 必须在 numpy 之前设置。16 核且负载约 18 的 aspen 上实测，780×2048 的一次 ridge_cv：多线程 BLAS 33.0 s，单线程 1.58 s。
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from scripts.interpret.analysis.oracle_press_prob import bootstrap, write_json
from scripts.interpret.analysis.rollout_attention import (GRID, INK, MODEL_STYLE, MODELS, MUTED, clean_axis, load_align,
                                                          offline_path, scalar, setup_plot)


PREFIX_GROUPS = ("历史视觉", "motion", "当前视觉", "文字")
MOTION_COLUMN, CURRENT_COLUMN = 1, 2
LABELS = ("q3", "q4")
LABEL_NAME = {"q3": "Q3 离自己按", "q4": "Q4 离正确窗口"}
LAMBDAS = np.logspace(-3, 5, 9)
OUTER_FOLDS, INNER_FOLDS = 5, 4
SEED, REPEATS, SHUFFLES = 42, 2000, 20
SHUFFLE_WARN = 0.1
MAIN_LAYER = 17
LAST_PREFIX = 18
LAYERS = 18


def hash_fold(key, folds, salt="outer", seed=SEED):
    """只由局编号和 seed 决定，所以两个模型映射相同，别的局存在与否不影响这一局。"""
    digest = hashlib.sha256(f"{seed}:{salt}:{key}".encode()).hexdigest()
    return int(digest, 16) % folds


def episode_weights(groups):
    """每局权重之和相等（长局不主导），整体均值归一到 1。"""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights * len(weights) / weights.sum()


def fit_standardizer(X, weights):
    total = weights.sum()
    mean = (weights[:, None] * X).sum(axis=0) / total
    std = np.sqrt((weights[:, None] * (X - mean) ** 2).sum(axis=0) / total)
    return mean, np.where(std > 0, std, 1.0)


def fit_path(X, y, weights, lambdas=LAMBDAS):
    """同一次分解求整条正则路径。返回 (mean, std, betas (D,Λ), intercept)。

    样本数小于维度走对偶：β=ZᵀW^½(K+λI)⁻¹W^½y，K=W^½ZZᵀW^½；否则走原始形式。
    """
    mean, std = fit_standardizer(X, weights)
    Z = (X - mean) / std
    intercept = float((weights * y).sum() / weights.sum())
    centered = y - intercept
    if Z.shape[1] > Z.shape[0]:
        root = np.sqrt(weights)
        scaled = Z * root[:, None]
        values, vectors = np.linalg.eigh(scaled @ scaled.T)
        basis = scaled.T @ vectors
        projected = vectors.T @ (root * centered)
    else:
        values, vectors = np.linalg.eigh((Z * weights[:, None]).T @ Z)
        basis = vectors
        projected = vectors.T @ (Z.T @ (weights * centered))
    values = np.clip(values, 0.0, None)
    betas = basis @ (projected[None, :] / (values[None, :] + np.asarray(lambdas)[:, None])).T
    return mean, std, betas, intercept


def predict_path(path, X):
    mean, std, betas, intercept = path
    return (X - mean) / std @ betas + intercept


def select_lambda(X, y, groups, weights, lambdas=LAMBDAS, seed=SEED):
    """训练折内部按局分 4 折；内层每折各自标准化。拆不出内层折时返回 None。"""
    folds = np.asarray([hash_fold(key, INNER_FOLDS, "inner", seed) for key in groups])
    errors, used = np.zeros(len(lambdas)), 0
    for fold in np.unique(folds):
        hold = folds == fold
        if hold.all():
            continue
        path = fit_path(X[~hold], y[~hold], weights[~hold], lambdas)
        residual = predict_path(path, X[hold]) - y[hold][:, None]
        errors += (weights[hold][:, None] * residual ** 2).sum(axis=0)
        used += 1
    return int(np.argmin(errors)) if used else None


def ridge_cv(X, y, groups, lambdas=LAMBDAS, seed=SEED):
    """外层 5 折折外预测；拆不出训练或内层折的样本预测记 NaN。group_leak 按实际下标集合核对。"""
    X, y, groups = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(groups)
    weights = episode_weights(groups)
    folds = np.asarray([hash_fold(key, OUTER_FOLDS, "outer", seed) for key in groups])
    prediction = np.full(len(y), np.nan)
    leak, chosen = 0, {}
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        leak += len(set(groups[train].tolist()) & set(groups[test].tolist()))
        if not train.any():
            continue
        index = select_lambda(X[train], y[train], groups[train], weights[train], lambdas, seed)
        if index is None:
            continue
        chosen[int(fold)] = float(lambdas[index])
        path = fit_path(X[train], y[train], weights[train], lambdas[index:index + 1])
        prediction[test] = predict_path(path, X[test])[:, 0]
    return {"prediction": prediction, "group_leak": leak, "lambda_by_fold": chosen}


def episode_mae(y, prediction, groups):
    keep = np.isfinite(prediction)
    return {key: float(np.abs(y[keep & (groups == key)] - prediction[keep & (groups == key)]).mean())
            for key in np.unique(groups[keep]).tolist()}


def metrics(y, prediction, groups):
    """折外 MAE（步）与 R²，都是每局等权。"""
    keep = np.isfinite(prediction)
    if not keep.any():
        return {"mae": None, "r2": None, "n_samples": 0, "n_episodes": 0, "n_no_prediction": int(len(y))}
    per_episode = episode_mae(y, prediction, groups)
    weights = episode_weights(groups[keep])
    center = (weights * y[keep]).sum() / weights.sum()
    total = (weights * (y[keep] - center) ** 2).sum()
    residual = (weights * (y[keep] - prediction[keep]) ** 2).sum()
    return {"mae": float(np.mean(list(per_episode.values()))), "r2": float(1 - residual / total) if total > 0 else None,
            "n_samples": int(keep.sum()), "n_episodes": len(per_episode), "n_no_prediction": int((~keep).sum())}


def mae_difference(left, right, repeats=REPEATS, seed=SEED):
    """left−right 的逐局 MAE 差，局级重采样；只比两边都有预测的局。"""
    shared = sorted(set(left) & set(right))
    return bootstrap([left[key] - right[key] for key in shared], repeats, seed)


def select_samples(item, label):
    """单局样本 [(infer_seq, t)] 和锚点；不进入该问题的局返回空。"""
    empty = ([], None)
    reach = item.get("reach_step")
    if not item.get("valid") or reach is None:
        return empty
    if label == "q3":
        if not item["uses"]["q3"] or item.get("trigger_step") is None:
            return empty
        anchor, limit = item["trigger_step"], item["I_trigger"]
    else:
        if not item["uses"]["q4"]:
            return empty
        anchor, limit = item["window_center"], None
    chosen = [(point["infer_seq"], point["chunk_start_env_step"]) for point in item["inferences"]
              if point["chunk_start_env_step"] >= reach and (limit is None or point["infer_seq"] <= limit)]
    return chosen, anchor


def load_bank(root, model, episodes):
    """读两个标签要用的全部隐状态，只读一次。"""
    index, prefix, action, tokens = {}, [], [], []
    for key, item in episodes.items():
        seqs = {seq for label in LABELS for seq, _ in select_samples(item, label)[0]}
        for seq in sorted(seqs):
            with np.load(offline_path(root, model, item["nonce"], seq), allow_pickle=False) as archive:
                if str(scalar(archive["episode_key"])) != key or int(scalar(archive["infer_seq"])) != seq:
                    raise ValueError(f"离线记录身份不符：{model} {key} {seq}")
                index[(key, seq)] = len(prefix)
                prefix.append(np.asarray(archive["hidden_prefix"], dtype=np.float32))
                action.append(np.asarray(archive["hidden_action"], dtype=np.float32))
                tokens.append(np.asarray(archive["hidden_group_tokens"]))
    if not prefix:
        return {"index": {}, "prefix": None, "action": None, "motion_present": False}
    return {"index": index, "prefix": np.stack(prefix), "action": np.stack(action),
            "motion_present": bool(np.stack(tokens)[:, MOTION_COLUMN].max() > 0)}


def build_samples(bank, episodes, label):
    rows, y, t, n_pass, anchors, groups = [], [], [], [], [], []
    for key, item in episodes.items():
        chosen, anchor = select_samples(item, label)
        for seq, start in chosen:
            rows.append(bank["index"][(key, seq)])
            y.append(anchor - start)
            t.append(start)
            n_pass.append(item["target_pass"])
            anchors.append(anchor)
            groups.append(key)
    return {"rows": np.asarray(rows, dtype=np.int64), "y": np.asarray(y, dtype=np.float64),
            "t": np.asarray(t, dtype=np.float64), "n_pass": np.asarray(n_pass, dtype=np.float64),
            "anchor": np.asarray(anchors, dtype=np.float64), "groups": np.asarray(groups)}


def restrict(samples, keys):
    keep = np.isin(samples["groups"], sorted(keys))
    return {name: value[keep] for name, value in samples.items()}


def active_columns(motion_present):
    """K9：基线没有 motion，只拼三组。"""
    return [column for column in range(len(PREFIX_GROUPS)) if motion_present or column != MOTION_COLUMN]


def k9_layer0(prefix, motion_present):
    """prefix (n,19,4,W) → 前缀各组第 0 层输入拼接 (n,G·W)。"""
    return prefix[:, 0, active_columns(motion_present)].reshape(len(prefix), -1)


def time_features(samples):
    """时间基线只用 t 和 N，不用 M。"""
    return np.stack([samples["t"], samples["n_pass"]], axis=1)


def shuffle_labels(samples, generator):
    """整局一起换：y′=另一局的锚点−t。"""
    keys = np.unique(samples["groups"])
    mapping = dict(zip(keys.tolist(), generator.permutation(len(keys)).tolist()))
    anchor_by_key = {key: samples["anchor"][samples["groups"] == key][0] for key in keys.tolist()}
    order = keys.tolist()
    return np.asarray([anchor_by_key[order[mapping[key]]] for key in samples["groups"].tolist()]) - samples["t"]


class Tally:
    """累计全部 ridge_cv 调用的 group_leak。"""

    def __init__(self):
        self.leak = 0

    def run(self, X, y, groups):
        result = ridge_cv(X, y, groups)
        self.leak += result["group_leak"]
        return result


def evaluate(tally, X, samples, y=None):
    y = samples["y"] if y is None else y
    fit = tally.run(X, y, samples["groups"])
    return {**metrics(y, fit["prediction"], samples["groups"]), "lambda_by_fold": fit["lambda_by_fold"]}, \
        episode_mae(y, fit["prediction"], samples["groups"]), fit["prediction"]


def main_readouts(tally, bank, samples, repeats=REPEATS, seed=SEED):
    """主读数和三个回归对照，以及逐局 MAE 差的区间。"""
    prefix = bank["prefix"][samples["rows"]]
    readouts = {"main": bank["action"][samples["rows"], MAIN_LAYER],
                "time_baseline": time_features(samples),
                "layer0": k9_layer0(prefix, bank["motion_present"]),
                "current_only": prefix[:, LAST_PREFIX, CURRENT_COLUMN]}
    result, per_episode, prediction = {}, {}, None
    for name, X in readouts.items():
        result[name], per_episode[name], values = evaluate(tally, X, samples)
        if name == "main":
            prediction = values
    result["main_minus_time"] = mae_difference(per_episode["main"], per_episode["time_baseline"], repeats, seed)
    result["main_minus_layer0"] = mae_difference(per_episode["main"], per_episode["layer0"], repeats, seed)
    result["main_minus_current_only"] = mae_difference(per_episode["main"], per_episode["current_only"], repeats, seed)
    return result, per_episode, prediction


def layer_curves(tally, bank, samples):
    """逐层描述：动作 token 和前缀四组（每层输出与同组第 0 层对照）；基线的 motion 列显式缺失。"""
    action, prefix = [], {name: [] for name in PREFIX_GROUPS}
    prefix0 = {}
    rows = samples["rows"]
    for layer in range(LAYERS):
        action.append(evaluate(tally, bank["action"][rows, layer], samples)[0])
    for column, name in enumerate(PREFIX_GROUPS):
        if column not in active_columns(bank["motion_present"]):
            prefix[name], prefix0[name] = None, None
            continue
        for layer in range(LAYERS):
            prefix[name].append(evaluate(tally, bank["prefix"][rows, layer + 1, column], samples)[0])
        prefix0[name] = evaluate(tally, bank["prefix"][rows, 0, column], samples)[0]
    return {"action": action, "prefix": prefix, "prefix_layer0": prefix0}


def shuffle_controls(tally, bank, samples, shuffles=SHUFFLES, seed=SEED):
    generator = np.random.default_rng(seed)
    X = bank["action"][samples["rows"], MAIN_LAYER]
    baseline = time_features(samples)
    probe_r2, time_r2 = [], []
    for _ in range(shuffles):
        y = shuffle_labels(samples, generator)
        probe_r2.append(evaluate(tally, X, samples, y)[0]["r2"])
        time_r2.append(evaluate(tally, baseline, samples, y)[0]["r2"])
    finite = [value for value in probe_r2 if value is not None]
    return {"n": shuffles, "r2": probe_r2, "r2_max": max(finite) if finite else None,
            "r2_mean": float(np.mean(finite)) if finite else None, "time_baseline_r2": time_r2}


def run_label(tally, bank, samples, repeats=REPEATS, seed=SEED, shuffles=SHUFFLES):
    result, per_episode, prediction = main_readouts(tally, bank, samples, repeats, seed)
    keep = np.isfinite(prediction)
    result.update(n_samples=len(samples["y"]), n_episodes=len(np.unique(samples["groups"])),
                  negative_labels=int((samples["y"] < 0).sum()),
                  shuffle=shuffle_controls(tally, bank, samples, shuffles, seed),
                  layers=layer_curves(tally, bank, samples),
                  scatter={"y": samples["y"][keep].tolist(), "prediction": prediction[keep].tolist()})
    return result, per_episode


def common_episodes(tally, banks, samples_by_model, label, repeats=REPEATS, seed=SEED):
    """在两个模型都有样本的局上重算主读数；差值只描述，不归因。"""
    models = list(banks)
    if len(models) != 2:
        return {"episodes": [], "models": {}, "right_minus_left_main": None}
    shared = set(samples_by_model[models[0]][label]["groups"].tolist()) & set(samples_by_model[models[1]][label]["groups"].tolist())
    if not shared:
        return {"episodes": [], "models": {}, "right_minus_left_main": None}
    result, per_episode = {}, {}
    for model in models:
        subset = restrict(samples_by_model[model][label], shared)
        result[model], per_episode[model], _ = main_readouts(tally, banks[model], subset, repeats, seed)
    return {"episodes": sorted(shared), "models": result, "direction": f"{models[1]}_minus_{models[0]}",
            "right_minus_left_main": mae_difference(per_episode[models[1]]["main"], per_episode[models[0]]["main"],
                                                    repeats, seed)}


def run_probes(banks, episodes_by_model, repeats=REPEATS, seed=SEED, shuffles=SHUFFLES):
    tally = Tally()
    per_model = {}
    samples_by_model = {model: {label: build_samples(banks[model], episodes_by_model[model], label) for label in LABELS}
                        for model in banks}
    for model in banks:
        per_model[model] = {"motion_present": banks[model]["motion_present"], "labels": {}}
        for label in LABELS:
            per_model[model]["labels"][label], _ = run_label(tally, banks[model], samples_by_model[model][label],
                                                             repeats, seed, shuffles)
    common = {label: common_episodes(tally, banks, samples_by_model, label, repeats, seed) for label in LABELS}
    return {"models": per_model, "common_episodes": common, "group_leak": tally.leak}


def verdict(summary):
    """流程判定：有泄漏或对照缺失才 FAIL；打乱标签超过 0.1 只警告，不改判据。"""
    models = summary["models"]
    entries = [entry for item in models.values() for entry in item["labels"].values()]
    controls = all(entry[name]["mae"] is not None for entry in entries
                   for name in ("main", "time_baseline", "layer0", "current_only"))
    controls = controls and all(entry["shuffle"]["r2_max"] is not None for entry in entries)
    n_labels = min((len(item["labels"]) for item in models.values()), default=0)
    shuffle_max = max((entry["shuffle"]["r2_max"] for entry in entries if entry["shuffle"]["r2_max"] is not None), default=None)
    passed = len(models) == len(MODELS) and n_labels == len(LABELS) and summary["group_leak"] == 0 and controls
    text = "null" if shuffle_max is None else f"{shuffle_max:.6g}"
    line = (f"PROBE_REPORT={'PASS' if passed else 'FAIL'} models={len(models)} labels={n_labels} "
            f"group_leak={summary['group_leak']} shuffle_r2_max={text}")
    warnings = []
    if shuffle_max is not None and shuffle_max > SHUFFLE_WARN:
        warnings.append(f"⚠⚠⚠ 警告：打乱标签 R²={shuffle_max:.4f} 超过 {SHUFFLE_WARN}。整局换的是锚点，t 没有打乱，"
                        "读得出 t 的探针 R² 也会大于 0；须与 shuffle.time_baseline_r2 对照后判断。判据不改。")
    return line, passed, warnings


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def plot_table(summary, output):
    """主读数表：MAE（步）/R²，以及探针减时间基线的区间。"""
    plt = setup_plot(output)
    cells = []
    for model, item in summary["models"].items():
        for label, entry in item["labels"].items():
            gap = entry["main_minus_time"]
            interval = "—" if gap["ci95"] is None else f"{gap['mean']:.2f} [{gap['ci95'][0]:.2f},{gap['ci95'][1]:.2f}]"
            cells.append([MODEL_STYLE[model][0], LABEL_NAME[label], f"{entry['n_episodes']}/{entry['n_samples']}"]
                         + [f"{fmt(entry[name]['mae'])} / {fmt(entry[name]['r2'], 3)}"
                            for name in ("main", "time_baseline", "layer0", "current_only")]
                         + [interval, fmt(entry["shuffle"]["r2_max"], 3)])
    columns = ["模型", "标签", "局/样本", "第 17 层动作", "时间基线", "第 0 层", "只用当前画面", "探针−时间基线 MAE", "打乱 R² 最大"]
    figure, axis = plt.subplots(figsize=(14, 0.38 * (len(cells) + 1) + 0.5))
    axis.axis("off")
    drawn = axis.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center",
                       colWidths=[0.08, 0.11, 0.07, 0.12, 0.12, 0.12, 0.12, 0.17, 0.09])
    drawn.auto_set_font_size(False)
    drawn.set_fontsize(9.5)
    drawn.scale(1, 1.5)
    for (row, _), cell in drawn.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#eeeeea")
    path = Path(output) / "probe-main-table.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


LINE_COLOR = dict(zip(("动作 token", *PREFIX_GROUPS), ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")))


def plot_layers(summary, output):
    """逐层 MAE 曲线：两个模型、两个标签；时间基线作横线。"""
    plt = setup_plot(output)
    models = list(summary["models"])
    figure, axes = plt.subplots(len(models), len(LABELS), figsize=(11, 3.6 * len(models)), squeeze=False, sharex=True)
    handles = {}
    for row, model in enumerate(models):
        for column, label in enumerate(LABELS):
            axis = axes[row, column]
            entry = summary["models"][model]["labels"][label]
            series = {"动作 token": entry["layers"]["action"], **entry["layers"]["prefix"]}
            for name, values in series.items():
                if values is None:
                    continue
                line, = axis.plot(range(LAYERS), [np.nan if item["mae"] is None else item["mae"] for item in values],
                                  color=LINE_COLOR[name], linewidth=2 if name == "动作 token" else 1.3, label=name)
                handles.setdefault(name, line)
            baseline = entry["time_baseline"]["mae"]
            if baseline is not None:
                handles.setdefault("时间基线", axis.axhline(baseline, color=MUTED, linestyle="--", linewidth=1.2, label="时间基线"))
            axis.set_title(f"{MODEL_STYLE[model][0]} · {LABEL_NAME[label]}", loc="left", fontsize=10.5, color=INK)
            axis.grid(color=GRID, linewidth=0.7)
            clean_axis(axis)
            if column == 0:
                axis.set_ylabel("折外 MAE（步）", color=MUTED)
            axis.set_xticks(range(0, LAYERS, 2))
            if row == len(models) - 1:
                axis.set_xlabel("层（0～17）", color=MUTED)
    ordered = [handles[name] for name in ("动作 token", *PREFIX_GROUPS, "时间基线") if name in handles]
    figure.legend(handles=ordered, loc="upper center", ncol=len(ordered), frameon=False, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = Path(output) / "probe-layers.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_scatter(summary, output):
    """主读数的折外预测对真值散点。"""
    plt = setup_plot(output)
    models = list(summary["models"])
    figure, axes = plt.subplots(len(models), len(LABELS), figsize=(10, 4.2 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        for column, label in enumerate(LABELS):
            axis = axes[row, column]
            entry = summary["models"][model]["labels"][label]
            y, prediction = np.asarray(entry["scatter"]["y"]), np.asarray(entry["scatter"]["prediction"])
            _, color, marker = MODEL_STYLE[model]
            axis.scatter(y, prediction, s=10, color=color, marker=marker, alpha=0.5, linewidths=0)
            if len(y):
                low, high = float(min(y.min(), prediction.min())), float(max(y.max(), prediction.max()))
                axis.plot([low, high], [low, high], color=MUTED, linewidth=0.9)
            axis.set_title(f"{MODEL_STYLE[model][0]} · {LABEL_NAME[label]}  MAE={fmt(entry['main']['mae'])}  R²={fmt(entry['main']['r2'], 3)}",
                           loc="left", fontsize=10, color=INK)
            axis.set_xlabel("真值（步）", color=MUTED)
            axis.set_ylabel("折外预测（步）", color=MUTED)
            axis.grid(color=GRID, linewidth=0.7)
            clean_axis(axis)
    figure.tight_layout()
    path = Path(output) / "probe-scatter.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def run(root, shuffles=SHUFFLES):
    root = Path(root)
    align = load_align(root)
    episodes = {model: align["models"][model] for model in MODELS if model in align["models"]}
    banks = {model: load_bank(root, model, items) for model, items in episodes.items()}
    summary = run_probes(banks, episodes, shuffles=shuffles)
    summary.update(schema_version=1, table=align["table"],
                   folds={"outer": OUTER_FOLDS, "inner": INNER_FOLDS, "seed": SEED, "rule": "sha256(seed:盐:局编号) 取模"},
                   lambdas=LAMBDAS.tolist(), bootstrap={"repeats": REPEATS, "seed": SEED, "unit": "episode"},
                   note="流程判定不代表解码成功；结论只到「可解码」，不写因果。")
    line, passed, warnings = verdict(summary)
    summary.update(verdict=line, warnings=warnings)
    output = root / "analysis"
    write_json(output / "probe-summary.json", summary)
    plot_table(summary, output)
    plot_layers(summary, output)
    plot_scatter(summary, output)
    return line, passed, warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="产物根")
    args = parser.parse_args(argv)
    line, passed, warnings = run(args.root)
    for warning in warnings:
        print(warning, flush=True)
    print(line, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
