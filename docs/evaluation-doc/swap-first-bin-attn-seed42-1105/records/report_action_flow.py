"""汇总 compute_action_flow.py 的逐局结果，写 action-flow-summary.json。

报告：逐层读取、动作最终来源（三个 alpha）、来源经哪层哪组到动作、记忆内部互读、头级集中度、patch、文字片段、选对减选错。
统计单位是局，局级重采样 2000 次（种子 42）；头级和文字片段属事后挑选，未校正。

用法：report_action_flow.py <action-flow 目录> <输出 json>
"""
import json
from pathlib import Path
import sys

import numpy as np

TASKS = ("ButtonUnmaskSwap", "VideoUnmaskSwap")
EVENTS = ("reveal", "cover", "swap", "after")


def boot(values):
    values = np.asarray(values, float)
    means = values[np.random.default_rng(42).integers(0, values.size, (2000, values.size))].mean(0 if values.ndim == 0 else 1)
    return {"n": int(values.size), "mean": float(values.mean()), "lo": float(np.quantile(means, .025)), "hi": float(np.quantile(means, .975))}


def diff_boot(a, b):
    rng = np.random.default_rng(42)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a[rng.integers(0, a.size, (2000, a.size))].mean(1) - b[rng.integers(0, b.size, (2000, b.size))].mean(1)
    return {"n_a": int(a.size), "n_b": int(b.size), "diff": float(a.mean() - b.mean()), "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975))}


def main(root, out):
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    groups = index["groups"]
    g = {n: i for i, n in enumerate(groups)}
    eps = index["episodes"]
    data = {}
    for e in eps:
        with np.load(root / e["file"]) as z:
            data[e["model"], e["key"]] = {k: z[k] for k in z.files}
    result = {"groups": groups, "alphas": index["alphas"], "cells": {}}
    for model in ("motion", "baseline"):
        for task in TASKS:
            sub = [e for e in eps if e["model"] == model and e["task"] == task]
            arrs = [data[model, e["key"]] for e in sub]
            direct = np.stack([a["direct"] for a in arrs])  # (局,层,头,q,k)
            counts = np.stack([a["counts"] for a in arrs])
            cell = {"episodes": len(sub), "token_counts": counts.mean(0).tolist()}
            mem = counts[:, :8].sum(1, keepdims=True)
            cell["uniform_memory_token_share"] = (counts[:, :8] / mem * 100).mean(0).tolist()
            cell["direct_layer_pct"] = (direct.mean((0, 2)) * 100).tolist()
            for alpha in index["alphas"]:
                final = np.stack([a[f"rollout_{alpha}"][-1, g["action"]] for a in arrs]) * 100
                route = np.stack([a[f"route_{alpha}"] for a in arrs]) * 100  # (局,层,via,source)
                roll = np.stack([a[f"rollout_{alpha}"] for a in arrs]) * 100  # (局,层+1,组,来源)
                memory_final = final[:, :8].sum(1, keepdims=True)
                share = final[:, :8] / np.maximum(memory_final, 1e-12) * 100
                uniform = counts[:, :8] / mem * 100
                cell[f"alpha_{alpha}"] = {
                    "action_final": {n: boot(final[:, i]) for i, n in enumerate(groups)},
                    "action_memory_share": {n: boot(share[:, i]) for i, n in enumerate(groups[:8])},
                    "action_memory_enrich": {n: boot(share[:, i] / np.maximum(uniform[:, i], 1e-9)) for i, n in enumerate(groups[:8]) if uniform[:, i].mean() > 0},
                    "route_via_source": route.sum(1).mean(0).tolist(),
                    "route_layer_source": route.sum(2).mean(0).tolist(),
                    "route_layer_via_source": route.mean(0).tolist(),
                    "carry_by_layer": roll.mean(0).tolist()}
                labels = np.array([e["label"] for e in sub])
                cell[f"alpha_{alpha}"]["correct_minus_wrong"] = {n: diff_boot(final[labels == "correct", i], final[labels == "wrong", i]) for i, n in enumerate(groups[:8]) if counts[:, i].mean() > 0}
            # 头级：超额 = 这段读取量 − 该流读取量 × token 占比；流内质量不足 1% 的头不计
            heads = {}
            for query in ("text", "action"):
                for stream, members in (("frame", list(range(4))), ("motion", list(range(4, 8)))):
                    if counts[:, members].sum() == 0:
                        continue
                    q = direct[:, :, :, g[query]][:, :, :, members]
                    stream_mass = q.sum(-1)
                    frac = counts[:, members] / np.maximum(counts[:, members].sum(1, keepdims=True), 1)
                    for ei, event in enumerate(EVENTS):
                        excess = q[..., ei] - stream_mass * frac[:, ei, None, None]
                        usable = stream_mass.mean(0) >= .01
                        mean_excess = np.where(usable, excess.mean(0), -9)
                        order = np.dstack(np.unravel_index(np.argsort(-mean_excess, axis=None), mean_excess.shape))[0]
                        top = []
                        for l, h in order[:6]:
                            if mean_excess[l, h] <= 0:
                                continue
                            share = (q[:, l, h, ei] / np.maximum(stream_mass[:, l, h], 1e-9)).mean()
                            top.append({"head": f"L{l}H{h}", "excess_pct": float(mean_excess[l, h] * 100), "event_mass_pct": float(q[:, l, h, ei].mean() * 100),
                                        "stream_mass_pct": float(stream_mass[:, l, h].mean() * 100), "share_pct": float(share * 100), "uniform_pct": float(frac[:, ei].mean() * 100),
                                        "support": int((excess[:, l, h] > 0).sum()), "episodes": len(sub)})
                        heads[f"{query}|{stream}|{event}"] = top
            cell["top_heads"] = heads
            spatial = np.stack([a["spatial"] for a in arrs]) * 100
            cell["spatial_pct"] = spatial.mean(0).tolist()
            pieces = {}
            for e in sub:
                for p, v in e["pieces"].items():
                    pieces.setdefault(p, []).append(v)
            cell["text_pieces"] = {p: {"episodes": len(v), "mean_pct": (np.mean(v, 0) * 100).tolist()} for p, v in pieces.items()}
            result["cells"][f"{model}|{task}"] = cell
    out.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ACTION_FLOW_REPORT=PASS cells={len(result['cells'])}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
