"""当前帧 token 有没有融合历史：逐层统计 prefix 内 query 组读 key 组的注意力。

输入窗口表（records/<模型>-windows.json）和 <模型>/offline/<nonce>/{infer,prefix}_*.npz。只读。
query 组：当前第三人称图像、当前手腕图像、文字；key 组：历史帧、motion、两路当前图像、文字。
每个 query 组的值 = 组内有效 query 行的平均（每行和为 1），先头平均成层，再按窗口内推理平均、按局平均。
两跳估计（attention-only，忽略 MLP 和残差）：图像 l 层读文字的量 × 文字在 0..l-1 层读记忆的平均。
末行 CURRENT_FUSION=PASS episodes=<n> image_to_memory_max=<x>；图像读记忆超过 1e-6 判 FAIL。

用法：analyze_current_frame_fusion.py <records 目录> <产物根> <输出 json> [--workers n] [--limit-episodes n]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

LAYERS, HEADS = 18, 8
FRAME_TOKENS, IMAGE_BASE, IMAGE_WRIST = 512, 256, 256
QUERIES = ("image_base", "image_wrist", "text")
KEYS = ("frame", "motion", "image_base", "image_wrist", "text")


def groups(z, P):
    mem, order, mask = int(z["MEM"]), np.asarray(z["mem_order"]), np.asarray(z["mem_mask"], bool)
    txt = int(z["txt_len"])
    out = {"frame": np.where(mask & (order < FRAME_TOKENS))[0], "motion": np.where(mask & (order >= FRAME_TOKENS))[0],
           "image_base": np.arange(mem, mem + IMAGE_BASE), "image_wrist": np.arange(mem + IMAGE_BASE, mem + IMAGE_BASE + IMAGE_WRIST),
           "text": np.arange(mem + IMAGE_BASE + IMAGE_WRIST, mem + IMAGE_BASE + IMAGE_WRIST + txt)}
    if mem + IMAGE_BASE + IMAGE_WRIST + txt > P:
        raise ValueError("分组越界")
    return out


def one_episode(job):
    acc = np.zeros((len(job["window"]), len(QUERIES), len(KEYS), LAYERS, HEADS))
    two_hop = np.zeros((len(job["window"]), 2, LAYERS))
    for i, seq in enumerate(job["window"]):
        d = Path(job["dir"])
        with np.load(d / f"infer_{seq:04d}.npz") as z:
            meta = {k: z[k] for k in ("MEM", "mem_order", "mem_mask", "txt_len", "P")}
        with np.load(d / f"prefix_{seq:04d}.npz") as z:
            A = z["probs_prefix"].astype(np.float32)
        g = groups(meta, int(meta["P"]))
        for q, qn in enumerate(QUERIES):
            rows = A[:, :, g[qn], :]
            for k, kn in enumerate(KEYS):
                acc[i, q, k] = rows[:, :, :, g[kn]].sum(-1).mean(-1)
        mean_layer = acc[i].mean(-1)
        text_to_memory = mean_layer[2, 0] + mean_layer[2, 1]
        for v in range(2):
            image_to_text = mean_layer[v, 4]
            for layer in range(1, LAYERS):
                two_hop[i, v, layer] = image_to_text[layer] * text_to_memory[:layer].mean()
        del A
    return {"key": job["key"], "task": job["task"], "model": job["model"], "matrix": acc.mean(0), "two_hop": two_hop.mean(0)}


def boot(values):
    picks = np.random.default_rng(42).integers(0, values.shape[0], (2000, values.shape[0]))
    means = values[picks].mean(1)
    return {"mean": values.mean(0).tolist(), "lo": np.quantile(means, .025, axis=0).tolist(), "hi": np.quantile(means, .975, axis=0).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit-episodes", type=int)
    args = p.parse_args()
    jobs = []
    for model, tag in (("motion", "mot"), ("baseline", "ctx")):
        data = json.loads((args.records / f"{tag}-windows.json").read_text(encoding="utf-8"))
        eps = [e for e in data["episodes"] if e.get("window")]
        for e in eps[:args.limit_episodes] if args.limit_episodes else eps:
            jobs.append({"key": e["episode_key"], "task": e["task"], "model": model, "window": e["window"],
                         "dir": str(args.root / tag / "offline" / e["episode_nonce"])})
    print(f"开始 episodes={len(jobs)}", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(one_episode, jobs), 1):
            rows.append(row)
            print(f"完成 {i}/{len(jobs)} {row['model']} {row['key']}", flush=True)
    result = {"queries": QUERIES, "keys": KEYS, "cells": {}}
    worst = 0.0
    for model in ("motion", "baseline"):
        for task in ("ButtonUnmaskSwap", "VideoUnmaskSwap"):
            sub = [r for r in rows if r["model"] == model and r["task"] == task]
            if not sub:
                continue
            stack = np.stack([r["matrix"] for r in sub]) * 100
            worst = max(worst, float(stack[:, :2, :2].max()))
            cell = {"episodes": len(sub), "two_hop_pct": {"image_base": boot(np.stack([r["two_hop"][0] for r in sub]) * 100),
                                                         "image_wrist": boot(np.stack([r["two_hop"][1] for r in sub]) * 100)}}
            for q, qn in enumerate(QUERIES):
                cell[qn] = {kn: boot(stack[:, q, k].mean(-1)) for k, kn in enumerate(KEYS)}
            cell["image_to_text_heads"] = stack[:, :2, 4].mean((0, 1)).tolist()
            cell["text_to_motion_heads"] = stack[:, 2, 1].mean(0).tolist()
            cell["text_to_frame_heads"] = stack[:, 2, 0].mean(0).tolist()
            result["cells"][f"{model}|{task}"] = cell
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"CURRENT_FUSION={'PASS' if worst <= 1e-4 else 'FAIL'} episodes={len(rows)} image_to_memory_max_pct={worst:.3e}")


if __name__ == "__main__":
    main()
