"""一次性（CPU，不训练、不推理）：把已有 Q4 回归的第 17 层折外预测，用真实 M、N 换算成三个分类。

该不该按：0 < y ≤ M；剩余次数：ceil(y/M)；已过次数：N − 剩余次数。同一局 N 固定，后两者准确率相同。
配对：同一局内方块 xy 距离 ≤ 2 cm、TCP 距离 ≤ 1 cm，且标签不同；都取触发前（含）的决策点。
用法：python derive_class.py <align.json> <fit-predictions.npz> <输出 json> <输出 npz>
"""
import json
import sys

import numpy as np

CUBE_TOL, TCP_TOL = 0.02, 0.01
READERS = ("action", "文字", "当前视觉", "历史视觉", "motion", "time_N")
EXPECTED = {"ctx": {"c": (155, 28), "press": (73, 17)}, "mot": {"c": (277, 27), "press": (146, 26)}}


def by_episode(values, keys):
    table = {}
    for value, key in zip(values, keys):
        table.setdefault(key, []).append(float(value))
    return {key: float(np.mean(items)) for key, items in table.items()}


def stat(values, keys, repeats=2000, seed=42):
    table = by_episode(values, keys)
    items = np.asarray(list(table.values()))
    draws = np.random.default_rng(seed).integers(0, len(items), (repeats, len(items)))
    low, high = np.percentile(items[draws].mean(axis=1), [2.5, 97.5])
    return {"mean": float(items.mean()), "ci95": [float(low), float(high)], "episodes": len(items), "n": len(values)}


def main():
    align_path, fit_path, out_json, out_npz = sys.argv[1:5]
    align, fit = json.load(open(align_path)), np.load(fit_path)
    summary, saved = {}, {}
    for model in ("ctx", "mot"):
        y, groups = fit[f"{model}/q4/y"], fit[f"{model}/q4/groups"]
        M, N, T = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
        points, cursor = {}, 0
        for key, item in align["models"][model].items():
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
        for key, items in points.items():
            for i, left in enumerate(items):
                for right in items[i + 1:]:
                    if np.linalg.norm(left[1] - right[1]) > CUBE_TOL or np.linalg.norm(left[2] - right[2]) > TCP_TOL:
                        continue
                    for name in pairs:
                        if truth[name][left[0]] != truth[name][right[0]]:
                            pairs[name].append((left[0], right[0], key))
        for name, (count, episodes) in EXPECTED[model].items():
            if (len(pairs[name]), len({p[2] for p in pairs[name]})) != (count, episodes):
                raise SystemExit(f"配对数不符：{model} {name}")
        summary[model] = {"points": len(y), "episodes": int(len(np.unique(groups))),
                          "pairs": {name: {"n": len(v), "episodes": len({p[2] for p in v})} for name, v in pairs.items()}, "readers": {}}
        press = np.asarray([p[:2] for p in pairs["press"]], dtype=np.int64)
        press_truth = truth["press"][press]
        order = np.where(press_truth[:, 0, None] == 1, press, press[:, ::-1])  # 第 0 列恒为该按的点
        saved[f"{model}/press_pairs_yes_no"] = order
        saved[f"{model}/press_true_ratio"] = y / M
        for reader in READERS:
            key = f"{model}/q4/{reader}"
            if key not in fit.files:
                continue
            prediction = fit[key]
            guess = {"c": N - np.ceil(prediction / M), "remaining": np.ceil(prediction / M),
                     "press": ((prediction > 0) & (prediction <= M)).astype(int)}
            entry = {}
            for label, pair_name in (("c", "c"), ("remaining", "c"), ("press", "press")):
                hit = guess[label] == truth[label]
                entry[label] = {"point": stat(hit.astype(float), groups),
                                "pair": stat([float(hit[a] and hit[b]) for a, b, _ in pairs[pair_name]], [p[2] for p in pairs[pair_name]])}
            summary[model]["readers"][reader] = entry
            saved[f"{model}/ratio/{reader}"] = prediction / M
            print(f"{model} {reader} press 配对 {entry['press']['pair']['mean']:.3f} 逐点 {entry['press']['point']['mean']:.3f} "
                  f"已过次数 配对 {entry['c']['pair']['mean']:.3f} 逐点 {entry['c']['point']['mean']:.3f}", flush=True)
    json.dump(summary, open(out_json, "w"), ensure_ascii=False, indent=1)
    np.savez(out_npz, **saved)
    print("CLASS_DERIVED=PASS pairs_match=1 truth_roundtrip=1", flush=True)


if __name__ == "__main__":
    main()
