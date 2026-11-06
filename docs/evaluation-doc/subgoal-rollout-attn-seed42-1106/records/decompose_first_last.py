"""描述性拆解：把对角富集分成「读示范首段」「读示范末段」「真对位（中间子目标）」。只读已存产物，不改判据，事后补做。

用法：decompose_first_last.py <产物根> <输出 json>
每局先平均、每局等权，局级重采样 2000 次。富集 = share 均值 ÷ uniform 均值。
- first_later：执行 j≥2 时读示范第 1 段（此时第 1 段不在对角上）。
- last_earlier：执行 j<m 时读示范第 m 段（此时第 m 段不在对角上）。
- diag_middle：1<j<m 时读示范第 j 段，首尾都不在内，只有这一格算真对位。
- diag_j1：j=1 时读第 1 段；首段偏好与对位在这一格分不开。
- reach：每局走到的最大 j、成败。
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.interpret.analysis.oracle_press_prob import bootstrap
from scripts.interpret.analysis.subgoal_attention import MODELS, READERS, STREAMS, TASKS, analyse_episode, discover_shards, load_episode, rollout_counts


def enrich(rows, pick):
    """rows 是一局的逐次读数；pick(j,m) 给要读的段号或 None。返回一局的富集，无样本为 None。"""
    share, uniform = [], []
    for row in rows:
        segment = pick(row["j"])
        if segment is None:
            continue
        share.append(float(row["share"][segment - 1]))
        uniform.append(float(row["uniform"][segment - 1]))
    if not share or np.mean(uniform) <= 0:
        return None
    return float(np.mean(share) / np.mean(uniform))


def summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n_episodes": 0, "mean": None, "ci95": None}
    return bootstrap(np.asarray(values, dtype=float))


root, out = Path(sys.argv[1]), Path(sys.argv[2])
result = {"note": "事后补做的描述性拆解，未预先写入判据；未校正多重比较", "cells": {}, "reach": {}}
episodes = []
for model in MODELS:
    counts = rollout_counts(discover_shards(root / model))
    for key, (record, _, run) in sorted(counts["accepted"].items()):
        episodes.append(analyse_episode(load_episode(run, root / model / "offline", record), model))
for task in TASKS:
    for model in MODELS:
        group = [ep for ep in episodes if ep["task"] == task and ep["model"] == model]
        reach, jobs = Counter(), Counter()
        for ep in group:
            top = max(inf["j"] for inf in ep["inferences"])
            reach[f"{'成功' if ep['success'] else '失败'}:走到第{top}段"] += 1
            jobs.update(inf["j"] for inf in ep["inferences"])
        result["reach"][f"{task}/{model}"] = {
            "m>=2": sum(ep["m"] >= 2 for ep in group), "success": sum(ep["success"] for ep in group),
            "fail_at_j1_m>=2": sum(not ep["success"] and ep["m"] >= 2 and max(i["j"] for i in ep["inferences"]) == 1 for ep in group),
            "reach_j2_m>=2": sum(ep["m"] >= 2 and max(i["j"] for i in ep["inferences"]) >= 2 for ep in group),
            "inferences_by_j": {str(k): v for k, v in sorted(jobs.items())}}
        for reader in READERS:
            for stream in STREAMS[model]:
                weighting = {"frame": "single", "motion": "overlap"}[stream]
                first, last, middle, j1, j_m = [], [], [], [], []
                for ep in group:
                    if ep["m"] < 2:
                        continue
                    rows, m = ep["records"][(reader, stream, weighting)], ep["m"]
                    first.append(enrich(rows, lambda j: 1 if j >= 2 else None))
                    last.append(enrich(rows, lambda j: m if j < m else None))
                    middle.append(enrich(rows, lambda j: j if 1 < j < m else None))
                    j1.append(enrich(rows, lambda j: 1 if j == 1 else None))
                    j_m.append(enrich(rows, lambda j: m if j == m else None))
                cell = {"first_later": summary(first), "last_earlier": summary(last), "diag_middle": summary(middle),
                        "diag_j1": summary(j1), "diag_jm": summary(j_m)}
                result["cells"][f"{task}/{model}/{reader}/{stream}"] = cell
                fmt = lambda c: f"{c['mean']:.2f}[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}]n={c['n_episodes']}" if c["mean"] is not None else "无"
                print(f"{task:11s} {model} {reader:6s} {stream:6s} 读首段(j≥2)={fmt(cell['first_later'])} 读末段(j<m)={fmt(cell['last_earlier'])} 中段对位={fmt(cell['diag_middle'])} j=1对角={fmt(cell['diag_j1'])}", flush=True)
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
for key, value in result["reach"].items():
    print(key, json.dumps(value, ensure_ascii=False), flush=True)
print("DECOMPOSE=DONE", flush=True)
