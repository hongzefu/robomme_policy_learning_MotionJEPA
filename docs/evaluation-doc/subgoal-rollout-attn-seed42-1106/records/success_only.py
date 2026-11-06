"""只看成功局的对位读数。事后补做的描述，只读已存产物，未预先写入判据，未校正多重比较。

用法：success_only.py <产物根> <输出 json> <输出 png>
读数与主分析同一套函数（a2_cell、judge_tier）。另加：
- first_later、diag_middle、last_earlier：含义同 decompose_first_last.py。
- j1_success、j1_fail：第 1 个子目标时读示范第 1 段的富集，成功局和失败局都有这一段，可直接比。
"""
import json
import sys
from pathlib import Path

import numpy as np

from scripts.interpret.analysis.oracle_press_prob import bootstrap
from scripts.interpret.analysis.subgoal_attention import (MAIN_WEIGHTING, MODELS, READERS, STREAMS, TASKS, _clean, _pyplot, a2_cell,
                                                          analyse_episode, discover_shards, load_episode, rollout_counts)

REPEATS, SEED = 2000, 42


def enrich(rows, pick):
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


def fmt(c):
    return f"{c['mean']:.2f}[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}]n={c['n_episodes']}" if c["mean"] is not None else "无"


root, out_json, out_png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
episodes = []
for model in MODELS:
    counts = rollout_counts(discover_shards(root / model))
    for key, (record, _, run) in sorted(counts["accepted"].items()):
        episodes.append(analyse_episode(load_episode(run, root / model / "offline", record), model))
result = {"note": "只看成功局，事后补做的描述；未预先写入判据，未校正多重比较；按结局筛选有选择偏差", "structure": {}, "cells": {}}
matrices = {}
for task in TASKS:
    for model in MODELS:
        group = [ep for ep in episodes if ep["task"] == task and ep["model"] == model]
        wins = [ep for ep in group if ep["success"]]
        result["structure"][f"{task}/{model}"] = {
            "success": len(wins), "m_counts": {str(m): sum(ep["m"] == m for ep in wins) for m in sorted({ep["m"] for ep in wins})},
            "difficulty": {d: sum(ep["difficulty"] == d for ep in wins) for d in ("easy", "medium", "hard")},
            "inferences": sum(len(ep["inferences"]) for ep in wins),
            "all_reached_m": all(max(i["j"] for i in ep["inferences"]) == ep["m"] for ep in wins)}
        print(f"{task}/{model}", json.dumps(result["structure"][f"{task}/{model}"], ensure_ascii=False), flush=True)
        for reader in READERS:
            for stream in STREAMS[model]:
                weighting = MAIN_WEIGHTING[stream]
                cell = a2_cell(wins, reader, stream, weighting, REPEATS, SEED)
                first, last, middle = [], [], []
                for ep in wins:
                    if ep["m"] >= 2:
                        rows, m = ep["records"][reader, stream, weighting], ep["m"]
                        first.append(enrich(rows, lambda j: 1 if j >= 2 else None))
                        last.append(enrich(rows, lambda j: m if j < m else None))
                        middle.append(enrich(rows, lambda j: j if 1 < j < m else None))
                j1 = {flag: summary([enrich(ep["records"][reader, stream, weighting], lambda j: 1 if j == 1 else None)
                                     for ep in group if ep["m"] >= 2 and ep["success"] == flag]) for flag in (True, False)}
                saved = {k: cell[k] for k in ("n_episodes", "D", "slope", "slope_undefined_episodes", "by_j", "offsets", "tier", "direction", "uniform_centroid_slope_mean", "matrix")}
                saved.update(first_later=summary(first), diag_middle=summary(middle), last_earlier=summary(last), j1_success=j1[True], j1_fail=j1[False])
                result["cells"][f"{task}/{model}/{reader}/{stream}"] = saved
                matrices[task, model, reader, stream] = cell["matrix"]
                print(f"{task:11s} {model} {reader:6s} {stream:6s} n={cell['n_episodes']} D={fmt(cell['D'])} 斜率={fmt(cell['slope'])} {cell['tier']} 首段={fmt(saved['first_later'])} 对位={fmt(saved['diag_middle'])} 末段={fmt(saved['last_earlier'])} j1成功={fmt(j1[True])} j1失败={fmt(j1[False])}", flush=True)
                print("   by_j", {k: (round(v["mean"], 2), v["n_episodes"]) for k, v in cell["by_j"].items() if v["mean"] is not None}, flush=True)
out_json.write_text(json.dumps(_clean(result), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

plt = _pyplot()
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
cmap = LinearSegmentedColormap.from_list("diverge", ["#2a78d6", "#f0efec", "#e34948"])
cmap.set_bad("#d9d8d2")
cells = [("ctx", "action", "frame"), ("ctx", "text", "frame"), ("mot", "action", "frame"), ("mot", "action", "motion"), ("mot", "text", "frame"), ("mot", "text", "motion")]
names = {"ctx": "基线", "mot": "motion", "action": "动作", "text": "文字", "frame": "帧流", "motion": "motion 流"}
vmax = max([2.0] + [np.nanmax(np.array(m["enrichment"], float)) for m in matrices.values() if m])
fig, axes = plt.subplots(2, len(cells), figsize=(18, 6.6), constrained_layout=True)
image = None
for r, task in enumerate(TASKS):
    for c, key in enumerate(cells):
        ax, matrix = axes[r, c], matrices[(task,) + key]
        n = result["cells"][f"{task}/{key[0]}/{key[1]}/{key[2]}"]["n_episodes"]
        ax.set_title(f"{task}\n{names[key[0]]} · {names[key[1]]} · {names[key[2]]}（{n} 局）", fontsize=9)
        if not matrix:
            ax.set_axis_off()
            continue
        values = np.array(matrix["enrichment"], float)
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, interpolation="nearest", aspect="equal", norm=TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=vmax))
        ax.set_xticks(range(len(values)), [str(i + 1) for i in range(len(values))], fontsize=8)
        ax.set_yticks(range(len(values)), [str(i + 1) for i in range(len(values))], fontsize=8)
        for j, jp in zip(*np.where(np.isfinite(values))):
            ax.text(jp, j, f"{values[j, jp]:.1f}", ha="center", va="center", fontsize=7)
        if r == 1:
            ax.set_xlabel("读示范第 j′ 段")
        if c == 0:
            ax.set_ylabel("执行第 j 段")
fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.6, label="share ÷ uniform（1 为按 token 均分）")
fig.suptitle("只看成功局的子目标对位矩阵（事后补做；不同 m 的局按相同 j、j′ 叠加；motion 流按「有重叠就算」）")
fig.savefig(out_png, dpi=130)
print("SUCCESS_ONLY=DONE", flush=True)
