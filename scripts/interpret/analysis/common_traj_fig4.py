"""StopCube 共同轨迹 Fig. 4 三表导出（计划 1108-stopcube-common-trajectory-action-plan 的 K5–K7、B2）。

方法：两个模型在同一批官方示教观测上前向（回放库），只解码一个回归标签 to_press（距正确窗口步数，
M·N − M//2 − t）。探针完全复用 hover_probe：外层按局 5 折、每局等权，特征组在外层训练折内标准化再 PCA 到
64 维，λ 在外层训练折内按局分 4 折选；时间参考线 [t]（time_only）与 [t,N]（time_and_target_count）用独热、
不经 PCA。每个（模型、组、层）一次 crossfit，按任务多进程（fork）。

输入：
  --points        1105 的 hover_points.json（取点与标签）；
  --replay-root   回放库根，含 ctx/ 与 mot/，每局 episode_<k>.npz，键 t、hidden_prefix (n,19,5,2048)
                  （第 2 维 0 为输入层、k+1 为第 k 层；第 3 维 0 历史视觉、1 motion、2 当前视觉、3 文字、4 最近 motion）、
                  group_tokens (n,5)，可选 hidden_action (n,18,W_a)（第 k 项为第 k 层）；
  --repro-summary 可选，1105 的 hover-summary.json，用来逐值核对本入口能复现 1105 的 to_press R²。

组：Text=列 3、Current frame=列 2、History frames=列 0、Motion=列 1（group_tokens 整列为 0 的组不参与，
即基线没有 Motion）、Action token=hidden_action（库里没有该键就跳过）。模型名 ctx→baseline、mot→motion。
层：layer=input、layer_index=-1 用 hidden_prefix 第 0 项；layer=k、layer_index=k 用 hidden_prefix 第 k+1 项
或 hidden_action 第 k 项。

输出（--output 目录）：
  r2_by_layer.csv      model,group,layer,layer_index,r2,mae_steps
  fit_lines.csv        model,group,r2,slope,slope_ci_low,slope_ci_high,intercept,n_points,n_episodes
                       （第 17 层；y=折外预测、x=真值，每局等权加权最小二乘；slope 区间按局重采样 2000 次，seed 42）
  reference_lines.csv  model,reference,metric,target,value,ci_low,ci_high,note
  summary.json         全部数值、参数与判定行

判定行只保证流程（取点相同、无泄漏、oracle 两次相等、结果非空、能复现 1105），不代表解码成功。
"""
from __future__ import annotations

import argparse
import csv
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

from scripts.interpret.analysis.hover_probe import (INNER_FOLDS, LAMBDAS, OUTER_FOLDS, PCA_DIM, build_targets, crossfit,
                                                    decode, load_samples, time_features)
from scripts.interpret.analysis.oracle_press_prob import write_json
from scripts.interpret.analysis.rollout_probe import REPEATS, SEED, episode_weights, metrics


LABEL = "to_press"
MODEL_ORDER = ("ctx", "mot")
MODEL_NAME = {"ctx": "baseline", "mot": "motion"}
# 前缀组：输出名 → hidden_prefix 第 3 维下标。顺序即导出顺序（Action token 插在 Current frame 之后，同旧表）。
PREFIX_COLUMN = {"Text": 3, "Current frame": 2, "History frames": 0, "Motion": 1}
ACTION_GROUP = "Action token"
GROUP_ORDER = ("Text", "Current frame", ACTION_GROUP, "History frames", "Motion")
FIT_GROUPS = ("Text", "Current frame", ACTION_GROUP)
# 1105 hover-summary.json 里的特征名。
REPRO_FEATURE = {"Text": "text", "Current frame": "current", "History frames": "history", "Motion": "motion"}
LAYERS = 18
FIT_LAYER = 17
INPUT_LAYER = -1
REFERENCES = {"time_only": "t", "time_and_target_count": "t_N"}
REFERENCE_NOTE = "common demo trajectory"
ORACLE_TOL = 1e-12
REPRO_TOL = 1e-9

_STATE = {}


# ---------- 读入 ----------

def bank_points(replay_root, label, keys):
    """读一个模型库里每局的 t，返回（局, t）集合；缺文件的局不进集合。只用于 SAME_POINTS。"""
    points = set()
    for key in keys:
        path = Path(replay_root) / label / f"episode_{key}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as archive:
            points.update((int(key), int(t)) for t in archive["t"].tolist())
    return points


def same_points(sets):
    """各模型库（局, t）集合的对称差大小（两两并起来）。返回 (点数, 差异数)。"""
    sets = list(sets)
    union = set().union(*sets) if sets else set()
    common = set.intersection(*sets) if sets else set()
    return len(common), len(union - common)


def load_bank(replay_root, label, samples):
    """读一个模型的回放库，只保留要用的组，float32。t 必须与 hover_points 逐局逐点相同。

    返回 {"groups": {组名: (n, 层数, D) 数组}, "layer_index": {组名: [层号...]}}。
    前缀组层号为 -1..17（数组第 0 项对应 -1）；Action token 层号为 0..17。
    """
    prefix, tokens, action = [], [], []
    for key in samples["keys"].tolist():
        with np.load(Path(replay_root) / label / f"episode_{key}.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["t"], samples["t"][samples["groups"] == key]):
                raise ValueError(f"回放的 t 与 hover_points 不符：{label} 第 {key} 局")
            prefix.append(np.asarray(archive["hidden_prefix"][:, :, list(PREFIX_COLUMN.values())], dtype=np.float32))
            tokens.append(np.asarray(archive["group_tokens"]))
            action.append(np.asarray(archive["hidden_action"], dtype=np.float32) if "hidden_action" in archive.files else None)
    prefix, tokens = np.concatenate(prefix), np.concatenate(tokens)
    groups, layer_index = {}, {}
    for position, (name, column) in enumerate(PREFIX_COLUMN.items()):
        present = tokens[:, column] > 0
        if not present.any():
            continue
        if not present.all():
            raise ValueError(f"{label} 的 {name} 列只有部分点有 token：{int(present.sum())}/{len(present)}")
        groups[name] = np.ascontiguousarray(prefix[:, :, position])
        layer_index[name] = list(range(INPUT_LAYER, LAYERS))
    del prefix
    if all(item is not None for item in action):
        groups[ACTION_GROUP] = np.concatenate(action)
        layer_index[ACTION_GROUP] = list(range(LAYERS))
    elif any(item is not None for item in action):
        raise ValueError(f"{label} 只有部分局有 hidden_action")
    return {"groups": groups, "layer_index": layer_index}


def layer_slice(group, layer):
    """层号 → 组数组第 1 维下标。"""
    return layer + 1 if group != ACTION_GROUP else layer


# ---------- 拟合 ----------

def targets(samples, all_labels=False):
    """to_press 目标矩阵与解码项。all_labels=True 时把 hover_probe 的全部标签放进 Y、只取 to_press 列。"""
    if all_labels:
        Y, layout = build_targets(samples)
        return Y, layout[(LABEL, None)]
    return samples["labels"][LABEL][:, None], (0, 1, None)


def probe(X, samples, use_pca, all_labels=False):
    """一次外层 5 折 crossfit，返回 (to_press 折外预测, 泄漏数)。"""
    Y, entry = targets(samples, all_labels)
    prediction, _, leak, _ = crossfit(lambda fold: X, samples, Y, use_pca)
    return decode(prediction, entry), leak


def run_task(task):
    began = time.monotonic()
    kind, model, name, layer = task
    samples = _STATE["samples"]
    if kind == "probe":
        group = _STATE["banks"][model]["groups"][name]
        X = group[:, layer_slice(name, layer)].astype(np.float64)
        prediction, leak = probe(X, samples, True, _STATE["all_labels"])
    else:
        prediction, leak = probe(time_features(samples, REFERENCES[name]), samples, False, _STATE["all_labels"])
    return task, {"pred": prediction, "leak": leak}, time.monotonic() - began


def run_tasks(tasks, workers):
    """按任务多进程。fork 共享只读的隐状态库，不重复占内存。"""
    results, seconds = {}, []

    def collect(iterator):
        for done, (task, value, elapsed) in enumerate(iterator, 1):
            results[task] = value
            seconds.append(elapsed)
            print(f"COMMON_FIG4_TASK done={done}/{len(tasks)} task={'/'.join(map(str, task))} seconds={elapsed:.1f}",
                  flush=True)

    if workers <= 1:
        collect(map(run_task, tasks))
    else:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            collect(pool.imap_unordered(run_task, tasks, chunksize=1))
    return results, seconds


# ---------- 拟合线 ----------

def fit_line(x, y, groups, repeats=REPEATS, seed=SEED):
    """每局等权的加权最小二乘 y = slope·x + intercept，slope 区间按局重采样。

    每局权重和相等（同 episode_weights），所以整体的加权矩只是各局局内均值之和：
    把每局压成 (1, x̄, ȳ, x²均值, xy均值) 五个量，重采样时按每局被抽中的次数加权求和即可，
    与「把抽中的局复制成新局、再用 episode_weights 重新加权拟合」逐式相同。
    """
    keep = np.isfinite(y) & np.isfinite(x)
    x, y, groups = x[keep], y[keep], groups[keep]
    keys, inverse = np.unique(groups, return_inverse=True)
    if len(keys) == 0:
        return {"slope": None, "intercept": None, "slope_ci": None, "n_points": 0, "n_episodes": 0}
    counts = np.bincount(inverse).astype(np.float64)

    def mean(values):
        return np.bincount(inverse, weights=values) / counts

    stats = np.stack([np.ones(len(keys)), mean(x), mean(y), mean(x * x), mean(x * y)], axis=1)

    def solve(total):
        n, sx, sy, sxx, sxy = total.T
        denominator = n * sxx - sx * sx
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = (n * sxy - sx * sy) / denominator
        return slope, (sy - slope * sx) / n

    slope, intercept = solve(stats.sum(axis=0)[None])
    generator = np.random.default_rng(seed)
    draws = generator.choice(len(keys), size=(repeats, len(keys)), replace=True)
    multiplicity = np.stack([np.bincount(row, minlength=len(keys)) for row in draws]).astype(np.float64)
    boot, _ = solve(multiplicity @ stats)
    boot = boot[np.isfinite(boot)]
    interval = np.quantile(boot, [0.025, 0.975]).tolist() if len(boot) else None
    return {"slope": _finite(slope[0]), "intercept": _finite(intercept[0]), "slope_ci": interval,
            "n_points": int(len(x)), "n_episodes": int(len(keys))}


def weighted_line_reference(x, y, groups):
    """直接用 episode_weights 做加权最小二乘，给测试核对 fit_line 的点估计。"""
    weights = episode_weights(groups)
    design = np.stack([x, np.ones_like(x)], axis=1) * np.sqrt(weights)[:, None]
    solution, *_ = np.linalg.lstsq(design, y * np.sqrt(weights), rcond=None)
    return float(solution[0]), float(solution[1])


# ---------- 汇总 ----------

def analyze(samples, banks, workers, all_labels=False):
    """跑全部任务并汇总。返回 summary（不含判定行）与折外预测。"""
    _STATE.update(samples=samples, banks=banks, all_labels=all_labels)
    tasks = []
    for model, bank in banks.items():
        tasks.extend(("time", model, name, None) for name in REFERENCES)
        for name in GROUP_ORDER:
            if name in bank["groups"]:
                tasks.extend(("probe", model, name, layer) for layer in bank["layer_index"][name])
    results, seconds = run_tasks(tasks, workers)
    truth, groups = samples["labels"][LABEL], samples["groups"]
    leak = sum(value["leak"] for value in results.values())

    models = {}
    for model, bank in banks.items():
        entry = {"name": MODEL_NAME[model], "groups": {}, "references": {}}
        for name in REFERENCES:
            entry["references"][name] = _clean(metrics(truth, results[("time", model, name, None)]["pred"], groups))
        time_r2 = entry["references"]["time_only"]["r2"]
        for name in GROUP_ORDER:
            if name not in bank["groups"]:
                continue
            layers = []
            for layer in bank["layer_index"][name]:
                cell = _clean(metrics(truth, results[("probe", model, name, layer)]["pred"], groups))
                cell["layer"] = "input" if layer == INPUT_LAYER else str(layer)
                cell["layer_index"] = layer
                cell["minus_time"] = None if cell["r2"] is None or time_r2 is None else cell["r2"] - time_r2
                layers.append(cell)
            group_entry = {"layers": layers, "best_layer": _best_layer(layers)}
            if name in FIT_GROUPS:
                prediction = results[("probe", model, name, FIT_LAYER)]["pred"]
                fit = fit_line(truth, prediction, groups)
                fit["r2"] = next(cell["r2"] for cell in layers if cell["layer_index"] == FIT_LAYER)
                fit["layer"] = FIT_LAYER
                group_entry["fit_line"] = fit
            entry["groups"][name] = group_entry
        models[model] = entry

    summary = {"schema_version": 1, "label": LABEL, "n_points": int(len(samples["t"])),
               "n_episodes": int(len(samples["keys"])),
               "method": {"folds": {"outer": OUTER_FOLDS, "inner": INNER_FOLDS, "seed": SEED,
                                    "rule": "sha256(seed:盐:局编号) 取模"},
                          "pca_dim": PCA_DIM, "lambdas": LAMBDAS.tolist(), "targets": "all_labels" if all_labels else LABEL,
                          "fit_line": {"layer": FIT_LAYER, "weights": "episode_weights", "bootstrap_repeats": REPEATS,
                                       "bootstrap_seed": SEED, "unit": "episode", "quantiles": [0.025, 0.975]},
                          "workers": workers},
               "model_name": {model: MODEL_NAME[model] for model in banks},
               "group_leak": int(leak), "models": models,
               "task_seconds": {"tasks": len(seconds), "mean": float(np.mean(seconds)) if seconds else 0.0},
               "note": "流程判定不代表解码成功；只到「可解码」。共同轨迹：两个模型在同一批示教观测上开环前向。"}
    return summary, results


def _best_layer(layers):
    values = [cell["r2"] if cell["r2"] is not None else -np.inf for cell in layers]
    if not np.isfinite(values).any():
        return None
    return layers[int(np.argmax(values))]["layer_index"]


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _clean(value):
    return {key: (_finite(item) if isinstance(item, float) else item) for key, item in value.items()}


# ---------- 判定 ----------

def verdicts(summary, points, diff, repro=None):
    """返回 (判定行, 是否全部 PASS)。"""
    lines, passed = [], True
    good = diff == 0
    passed &= good
    lines.append(f"SAME_POINTS={'PASS' if good else 'FAIL'} points={points} diff={diff}")
    leak = summary["group_leak"]
    passed &= leak == 0
    lines.append(f"GROUP_LEAK={'PASS' if leak == 0 else 'FAIL'} leak={leak}")
    oracle = [summary["models"][model]["references"]["time_and_target_count"]["r2"] for model in summary["models"]]
    good = len(oracle) == 2 and None not in oracle and abs(oracle[0] - oracle[1]) <= ORACLE_TOL
    passed &= good
    lines.append(f"ORACLE_EQUAL={'PASS' if good else 'FAIL'} " + " ".join(
        f"r2_{letter}={'null' if value is None else repr(value)}" for letter, value in zip("ab", oracle)))
    for model, entry in summary["models"].items():
        for name, group in entry["groups"].items():
            for pick, layer in (("last", FIT_LAYER), ("best", group["best_layer"])):
                cell = next((cell for cell in group["layers"] if cell["layer_index"] == layer), None)
                r2 = None if cell is None else cell["r2"]
                good = r2 is not None
                passed &= good
                lines.append(f"PROBE={'PASS' if good else 'FAIL'} model={entry['name']} group={name.replace(' ', '_')} "
                             f"layer={'null' if cell is None else cell['layer']} r2={_number(r2)} "
                             f"minus_time={_number(None if cell is None else cell['minus_time'])} pick={pick}")
    if repro is not None:
        passed &= repro["pass"]
        lines.append(f"REPRO_1105={'PASS' if repro['pass'] else 'FAIL'} maxabs={_scientific(repro['maxabs'])} "
                     f"compared={repro['compared']} missing={repro['missing']}")
    return lines, passed


def compare_1105(summary, reference):
    """把前缀组每层 to_press R²（及时间参考线）与 1105 hover-summary.json 逐值比较。"""
    diffs, missing, rows = [], 0, []
    for model, entry in summary["models"].items():
        old_model = reference["models"].get(model)
        for name, group in entry["groups"].items():
            if name not in REPRO_FEATURE:
                continue
            old = None if old_model is None else old_model["features"].get(REPRO_FEATURE[name])
            for cell in group["layers"]:
                if old is None:
                    old_value = None
                elif cell["layer_index"] == INPUT_LAYER:
                    old_value = old.get("input_level", {}).get(LABEL, {}).get("r2")
                else:
                    old_value = old["layers"][cell["layer_index"]][LABEL]["r2"]
                rows.append(_repro_row(model, name, cell["layer"], cell["r2"], old_value, diffs))
                missing += rows[-1]["diff"] is None
        for name, kind in REFERENCES.items():
            old_value = reference["time_baseline"].get(kind, {}).get(LABEL, {}).get("r2")
            rows.append(_repro_row(model, f"reference:{name}", None, entry["references"][name]["r2"], old_value, diffs))
            missing += rows[-1]["diff"] is None
    maxabs = max(diffs) if diffs else None
    return {"pass": bool(diffs) and missing == 0 and maxabs <= REPRO_TOL, "maxabs": maxabs, "tol": REPRO_TOL,
            "compared": len(diffs), "missing": int(missing), "rows": rows}


def _repro_row(model, group, layer, new, old, diffs):
    diff = None if new is None or old is None else abs(new - old)
    if diff is not None:
        diffs.append(diff)
    return {"model": model, "group": group, "layer": layer, "r2": new, "r2_1105": old, "diff": diff}


def _number(value):
    return "null" if value is None else f"{value:.4f}"


def _scientific(value):
    return "null" if value is None else f"{value:.3e}"


# ---------- 导出 ----------

def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])


def export_tables(summary, output):
    output = Path(output)
    r2_rows, fit_rows, reference_rows = [], [], []
    for model, entry in summary["models"].items():
        name = entry["name"]
        for group, item in entry["groups"].items():
            for cell in item["layers"]:
                r2_rows.append([name, group, cell["layer"], cell["layer_index"], cell["r2"], cell["mae"]])
            if "fit_line" in item:
                fit = item["fit_line"]
                low, high = fit["slope_ci"] or (None, None)
                fit_rows.append([name, group, fit["r2"], fit["slope"], low, high, fit["intercept"], fit["n_points"],
                                 fit["n_episodes"]])
        for reference in REFERENCES:
            values = entry["references"][reference]
            for metric, key in (("r2", "r2"), ("mae", "mae")):
                reference_rows.append([name, reference, metric, None, values[key], None, None, REFERENCE_NOTE])
    write_csv(output / "r2_by_layer.csv", ("model", "group", "layer", "layer_index", "r2", "mae_steps"), r2_rows)
    write_csv(output / "fit_lines.csv", ("model", "group", "r2", "slope", "slope_ci_low", "slope_ci_high", "intercept",
                                         "n_points", "n_episodes"), fit_rows)
    write_csv(output / "reference_lines.csv", ("model", "reference", "metric", "target", "value", "ci_low", "ci_high",
                                               "note"), reference_rows)


# ---------- 入口 ----------

def run(points, replay_root, output, workers, repro_summary=None, all_labels=False, models=MODEL_ORDER):
    began = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples = load_samples(points)
    keys = samples["keys"].tolist()
    expected = set(zip(samples["groups"].tolist(), samples["t"].tolist()))
    sets = [bank_points(replay_root, model, keys) for model in models]
    n_points, diff = same_points(sets)
    mismatch = {model: len(found ^ expected) for model, found in zip(models, sets)}
    if any(mismatch.values()):
        line = f"SAME_POINTS=FAIL points={n_points} diff={diff}"
        print(line, f"vs_hover_points={mismatch}", flush=True)
        raise ValueError(f"回放库与 hover_points 的取点不符：{mismatch}")
    banks = {model: load_bank(replay_root, model, samples) for model in models}
    summary, _ = analyze(samples, banks, workers, all_labels)
    repro = None
    if repro_summary is not None:
        with open(repro_summary, encoding="utf-8") as handle:
            repro = compare_1105(summary, json.load(handle))
        summary["repro_1105"] = repro
    lines, passed = verdicts(summary, n_points, diff, repro)
    summary["same_points"] = {"points": n_points, "diff": diff}
    summary["verdicts"] = lines
    summary["seconds"] = time.monotonic() - began
    summary["inputs"] = {"points": str(points), "replay_root": str(replay_root),
                         "repro_summary": None if repro_summary is None else str(repro_summary)}
    write_json(output / "summary.json", summary)
    export_tables(summary, output)
    return lines, passed, summary["seconds"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", type=Path, required=True, help="hover_points.json")
    parser.add_argument("--replay-root", type=Path, required=True, help="回放库根，含 ctx/ 与 mot/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--repro-summary", type=Path, default=None, help="1105 hover-summary.json，给出时打印 REPRO_1105")
    parser.add_argument("--all-label-targets", action="store_true",
                        help="把 hover_probe 的全部标签放进 Y、只取 to_press（默认只放 to_press）")
    args = parser.parse_args(argv)
    lines, passed, seconds = run(args.points, args.replay_root, args.output, args.workers, args.repro_summary,
                                 args.all_label_targets)
    for line in lines:
        print(line, flush=True)
    print(f"COMMON_FIG4_DONE seconds={seconds:.1f}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
