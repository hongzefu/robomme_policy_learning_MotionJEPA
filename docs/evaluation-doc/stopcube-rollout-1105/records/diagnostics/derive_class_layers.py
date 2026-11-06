"""一次性（CPU，不训练模型、不推理）：把 Q4 回归的换算分类从第 17 层补到所有层。

回归本身和 rollout_probe 相同：训练时不知道配对，也不知道分类目标，只拟合「离正确窗口还有几步」。
原分析只留了逐层 R²，没留逐层折外预测，所以这里按同样的折和 λ 选择重算一遍。
闸门：每层 R² 要等于 probe-summary.json；第 17 层换算结果要等于 class-derived.json；配对数要等于 derive_class.py。
用法：python derive_class_layers.py <产物根> <probe-summary.json> <class-derived.json> <输出目录> <进程数>
"""
import json
import multiprocessing
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from derive_class import CUBE_TOL, EXPECTED, TCP_TOL, stat
from scripts.interpret.analysis import rollout_probe as P
from scripts.interpret.analysis.rollout_attention import load_align

BANKS, SAMPLES = {}, {}


def fit_one(task):
    """task=(模型, 读数名, 层下标)。层下标 -1 是输入，0～17 是第 k 层输出。"""
    model, reader, layer = task
    rows = SAMPLES[model]["rows"]
    if reader == "action":
        X = BANKS[model]["action"][rows, layer]
    else:
        X = BANKS[model]["prefix"][rows, layer + 1, P.PREFIX_GROUPS.index(reader)]
    fit = P.ridge_cv(X, SAMPLES[model]["y"], SAMPLES[model]["groups"])
    r2 = P.metrics(SAMPLES[model]["y"], fit["prediction"], SAMPLES[model]["groups"])["r2"]
    return task, fit["prediction"], r2, fit["group_leak"]


def truth_and_pairs(model, items, samples):
    """决策点的 M、N、真值和同位置配对，规则与 derive_class.py 相同。"""
    y, groups = samples["y"], samples["groups"]
    M, N, T = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    points, cursor = {}, 0
    for key, item in items.items():
        if not item.get("valid") or item.get("reach_step") is None or not item["uses"]["q4"]:
            continue
        first = {}
        for line in open(item["trace_path"]):
            row = json.loads(line)
            if row.get("event") == "step" and row.get("row_in_chunk") == 0:
                first[row["infer_seq"]] = row
        trigger = item.get("trigger_step")
        for point in item["inferences"]:
            t = point["chunk_start_env_step"]
            if t < item["reach_step"]:
                continue
            if groups[cursor] != key or abs(y[cursor] - (item["window_center"] - t)) > 1e-6:
                raise SystemExit(f"点序不符：{model} {key} {t}")
            M[cursor], N[cursor], T[cursor] = item["move_interval"], item["target_pass"], t
            if point["infer_seq"] in first and (trigger is None or t <= trigger):
                row = first[point["infer_seq"]]
                points.setdefault(key, []).append((cursor, np.asarray(row["cube_position_before"][:2]), np.asarray(row["tcp_before"])))
            cursor += 1
    if cursor != len(y):
        raise SystemExit("点数不符")
    c = np.maximum(0, np.floor((T + np.floor(M / 2)) / M))
    truth = {"c": c, "press": (c + 1 == N).astype(int), "remaining": N - c}
    if not (((y > 0) & (y <= M)).astype(int) == truth["press"]).all() or not (np.ceil(y / M) == truth["remaining"]).all():
        raise SystemExit("真值换算与标签不符")
    pairs = {"c": [], "press": []}
    for key, chosen in points.items():
        for i, left in enumerate(chosen):
            for right in chosen[i + 1:]:
                if np.linalg.norm(left[1] - right[1]) > CUBE_TOL or np.linalg.norm(left[2] - right[2]) > TCP_TOL:
                    continue
                for name in pairs:
                    if truth[name][left[0]] != truth[name][right[0]]:
                        pairs[name].append((left[0], right[0], key))
    for name, (count, episodes) in EXPECTED[model].items():
        if (len(pairs[name]), len({p[2] for p in pairs[name]})) != (count, episodes):
            raise SystemExit(f"配对数不符：{model} {name}")
    return M, N, truth, pairs


def score(prediction, M, N, truth, pairs, groups):
    guess = {"c": N - np.ceil(prediction / M), "press": ((prediction > 0) & (prediction <= M)).astype(int)}
    entry = {}
    for label in ("press", "c"):
        hit = guess[label] == truth[label]
        entry[label] = {"point": stat(hit.astype(float), groups),
                        "pair": stat([float(hit[a] and hit[b]) for a, b, _ in pairs[label]], [p[2] for p in pairs[label]])}
    return entry


def main():
    root, probe_path, class_path, output, workers = sys.argv[1:6]
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    probe, reference = json.load(open(probe_path)), json.load(open(class_path))
    align = load_align(root)
    tasks, context = [], {}
    for model in ("ctx", "mot"):
        items = align["models"][model]
        BANKS[model] = P.load_bank(root, model, items)
        SAMPLES[model] = P.build_samples(BANKS[model], items, "q4")
        context[model] = truth_and_pairs(model, items, SAMPLES[model])
        tasks += [(model, "action", layer) for layer in range(P.LAYERS)]
        for column in P.active_columns(BANKS[model]["motion_present"]):
            tasks += [(model, P.PREFIX_GROUPS[column], layer) for layer in range(-1, P.LAYERS)]
    print(f"TASKS={len(tasks)}", flush=True)
    results = {}
    leak = 0
    with multiprocessing.get_context("fork").Pool(int(workers)) as pool:
        for count, (task, prediction, r2, group_leak) in enumerate(pool.imap_unordered(fit_one, tasks), 1):
            results[task] = (prediction, r2)
            leak += group_leak
            if count % 10 == 0:
                print(f"PROGRESS {count}/{len(tasks)}", flush=True)
    summary, saved, checked = {}, {}, 0
    for model in ("ctx", "mot"):
        M, N, truth, pairs = context[model]
        groups = SAMPLES[model]["groups"]
        layers = probe["models"][model]["labels"]["q4"]["layers"]
        summary[model] = {"pairs": {name: {"n": len(v), "episodes": len({p[2] for p in v})} for name, v in pairs.items()}, "readers": {}}
        time_fit = P.ridge_cv(P.time_features(SAMPLES[model]), SAMPLES[model]["y"], groups)
        summary[model]["time_N"] = score(time_fit["prediction"], M, N, truth, pairs, groups)
        for (name, reader, layer), (prediction, r2) in sorted(results.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
            if name != model:
                continue
            if reader == "action":
                expected = layers["action"][layer]["r2"]
            elif layer == -1:
                expected = layers["prefix_layer0"][reader]["r2"]
            else:
                expected = layers["prefix"][reader][layer]["r2"]
            if abs(r2 - expected) > 1e-6:
                raise SystemExit(f"{model} {reader} 层 {layer} R² 与原结果不同：{r2} 对 {expected}")
            checked += 1
            entry = score(prediction, M, N, truth, pairs, groups)
            if layer == P.MAIN_LAYER:
                old = reference[model]["readers"][reader]
                for label in ("press", "c"):
                    for kind in ("point", "pair"):
                        if abs(entry[label][kind]["mean"] - old[label][kind]["mean"]) > 1e-9:
                            raise SystemExit(f"{model} {reader} 第 17 层 {label} {kind} 与 class-derived 不同")
            summary[model]["readers"].setdefault(reader, {})[str(layer)] = {"r2": r2, **entry}
            saved[f"{model}/{reader}/{layer}"] = prediction.astype(np.float32)
    (output / "class-layers.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    np.savez(output / "class-layers-predictions.npz", **saved)
    print(f"CLASS_LAYERS=PASS fits={len(tasks)} r2_checked={checked} layer17_match=1 pairs_match=1 group_leak={leak}", flush=True)


if __name__ == "__main__":
    main()
