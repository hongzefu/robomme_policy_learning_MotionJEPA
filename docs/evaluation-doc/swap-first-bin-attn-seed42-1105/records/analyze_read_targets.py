"""文字、当前图像（经文字两跳）、动作 token 分别读了哪段历史帧和哪些 motion 窗。

输入同 analyze_current_frame_fusion.py；只读。窗口内每次推理：
- 文字：prefix 的文字 query 行，先头平均、再文字 token 平均。
- 动作：infer 的 probs_action，按去噪步、动作 query、头平均。
- 图像两跳（attention-only 估计，忽略 MLP 和残差）：图像 query 在 l 层读每个文字 token 的量 ×
  该文字 token 在 0..l-1 层读这段记忆的平均。图像直接读记忆由掩码恒为 0。
事件：开局 [0,32)、盖住 [32,64)、交换 [64,64+50k)、其后。帧按帧时刻归类；
motion 窗内容范围 [t,t+32)，演示段截断到 exec_start_idx；与交换有重叠即算交换；否则按起点归类。
均匀 = 该事件的 token 数占该流 token 数。结果按局平均。
另按文字 token 片段统计第 0、1 层读记忆的量（片段来自 episode_window.json 的 text_token_pieces）。
末行 READ_TARGETS=PASS episodes=<n>。

用法：analyze_read_targets.py <records 目录> <产物根> <episodes 目录> <输出 json> [--workers n] [--limit-episodes n]
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

LAYERS = 18
FRAME_TOKENS, TOKENS_PER_FRAME, IMAGE_TOKENS = 512, 16, 512
EVENTS = ("reveal", "cover", "swap", "after")
QUERIES = ("text", "image_2hop", "action")
BIN = 16


def event_of_frame(t, end):
    return "reveal" if t < 32 else "cover" if t < 64 else "swap" if t < end else "after"


def event_of_window(t, d, end):
    stop = min(t + 32, d) if t < d else t + 32
    if t < end and stop > 64:
        return "swap"
    return "reveal" if t < 32 else "cover" if t < 64 else "after"


def one_episode(job):
    end = 64 + 50 * job["k"]
    d = Path(job["dir"])
    pieces_mass, frame_time, motion_time, uniform = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), defaultdict(list)
    events = {q: {s: {e: [] for e in EVENTS} for s in ("frame", "motion")} for q in QUERIES}
    for seq in job["window"]:
        with np.load(d / f"infer_{seq:04d}.npz") as z:
            mem, order, mask = int(z["MEM"]), np.asarray(z["mem_order"]), np.asarray(z["mem_mask"], bool)
            txt = int(z["txt_len"])
            frame_times, motion_times = np.asarray(z["frame_times"]), np.asarray(z["motion_times"]).reshape(-1)
            exec_start = int(z["exec_start_idx"])
            action = np.asarray(z["probs_action"], np.float32).mean(axis=(0, 2, 3))
        with np.load(d / f"prefix_{seq:04d}.npz") as z:
            prefix = z["probs_prefix"].astype(np.float32).mean(1)  # (层, P, P)
        image = np.arange(mem, mem + IMAGE_TOKENS)
        text = np.arange(mem + IMAGE_TOKENS, mem + IMAGE_TOKENS + txt)
        fpos = np.where(mask & (order < FRAME_TOKENS))[0]
        mpos = np.where(mask & (order >= FRAME_TOKENS))[0]
        ftime = frame_times[order[fpos] // TOKENS_PER_FRAME]
        mtime = motion_times[order[mpos] - FRAME_TOKENS]
        text_rows = prefix[:, text][:, :, :mem]  # (层, 文字, 记忆)
        cum = np.zeros_like(text_rows)
        for layer in range(1, LAYERS):
            cum[layer] = text_rows[:layer].mean(0)
        img_to_text = prefix[:, image][:, :, text].mean(1)  # (层, 文字)
        memory_views = {"text": text_rows.mean(1), "image_2hop": np.einsum("lt,ltm->lm", img_to_text, cum), "action": action[:, :mem]}
        for q, vec in memory_views.items():
            for stream, pos, times, label in (("frame", fpos, ftime, lambda t: event_of_frame(t, end)), ("motion", mpos, mtime, lambda t: event_of_window(t, exec_start, end))):
                values = vec[:, pos]
                labels = np.array([label(int(t)) for t in times]) if len(times) else np.array([], dtype=object)
                for e in EVENTS:
                    events[q][stream][e].append(values[:, labels == e].sum(1) if len(labels) else np.zeros(LAYERS))
                keys = times // BIN * BIN if stream == "frame" else times
                target = frame_time if stream == "frame" else motion_time
                for key in np.unique(keys):
                    target[q][int(key)].append(values[:, keys == key].sum(1))
        for stream, pos, times, label in (("frame", fpos, ftime, lambda t: event_of_frame(t, end)), ("motion", mpos, mtime, lambda t: event_of_window(t, exec_start, end))):
            labels = [label(int(t)) for t in times]
            for e in EVENTS:
                uniform[f"{stream}|{e}"].append(labels.count(e) / max(len(labels), 1))
        for i, piece in enumerate(job["pieces"][:txt]):
            pieces_mass[piece].append(np.array([text_rows[:2, i, fpos].sum(1), text_rows[:2, i, mpos].sum(1)]))
        del prefix, action, text_rows, cum
    n = len(job["window"])
    return {"task": job["task"], "k": job["k"], "model": job["model"], "key": job["key"],
            "events": {q: {s: {e: np.mean(v, 0).tolist() for e, v in blocks.items()} for s, blocks in by.items()} for q, by in events.items()},
            "frame_time": {q: {t: (np.sum(v, 0) / n).tolist() for t, v in by.items()} for q, by in frame_time.items()},
            "motion_time": {q: {t: (np.sum(v, 0) / n).tolist() for t, v in by.items()} for q, by in motion_time.items()},
            "uniform": {k: float(np.mean(v)) for k, v in uniform.items()},
            "pieces": {p: np.mean(v, 0).tolist() for p, v in pieces_mass.items()}}


def boot(values):
    picks = np.random.default_rng(42).integers(0, values.size, (2000, values.size))
    means = values[picks].mean(1)
    return {"mean": float(values.mean()), "lo": float(np.quantile(means, .025)), "hi": float(np.quantile(means, .975))}


def summarize(rows, by_time=True):
    out = {"episodes": len(rows), "events": {}, "uniform": {k: float(np.mean([r["uniform"][k] for r in rows])) for k in rows[0]["uniform"]}}
    for q in QUERIES:
        out["events"][q] = {}
        for s in ("frame", "motion"):
            stacks = {e: np.stack([np.asarray(r["events"][q][s][e]) for r in rows]) * 100 for e in EVENTS}
            total = sum(stacks.values())
            out["events"][q][s] = {e: {"by_layer": stacks[e].mean(0).tolist(), "all_layers": boot(stacks[e].mean(1)),
                                       "share_of_stream": boot(np.divide(stacks[e].mean(1), np.maximum(total.mean(1), 1e-12)) * 100)} for e in EVENTS}
    if by_time:
        for name in ("frame_time", "motion_time"):
            out[name] = {}
            for q in QUERIES:
                acc = defaultdict(list)
                for r in rows:
                    for t, v in r[name].get(q, {}).items():  # 基线没有 motion 窗
                        acc[t].append(v)
                keep = {t: v for t, v in acc.items() if len(v) >= .5 * len(rows)}
                out[name][q] = {str(t): (np.mean(v, 0) * 100).tolist() for t, v in sorted(keep.items())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("episodes", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit-episodes", type=int)
    args = p.parse_args()
    jobs = []
    for model, tag, prefix in (("motion", "mot", "with_motion"), ("baseline", "ctx", "without_motion")):
        data = json.loads((args.records / f"{tag}-windows.json").read_text(encoding="utf-8"))
        eps = [e for e in data["episodes"] if e.get("window")]
        for e in eps[:args.limit_episodes] if args.limit_episodes else eps:
            detail = json.loads((args.episodes / f"{prefix}_{e['episode_key'].replace(':', '_')}" / "episode_window.json").read_text(encoding="utf-8"))
            jobs.append({"key": e["episode_key"], "task": e["task"], "k": int(e["swap_times"]), "model": model, "window": e["window"],
                         "pieces": detail["records"][0]["text_token_pieces"], "dir": str(args.root / tag / "offline" / e["episode_nonce"])})
    print(f"开始 episodes={len(jobs)}", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(one_episode, jobs), 1):
            rows.append(row)
            print(f"完成 {i}/{len(jobs)} {row['model']} {row['key']}", flush=True)
    result = {"events": EVENTS, "queries": QUERIES, "cells": {}}
    for model in ("motion", "baseline"):
        for task in ("ButtonUnmaskSwap", "VideoUnmaskSwap"):
            sub = [r for r in rows if r["model"] == model and r["task"] == task]
            if not sub:
                continue
            cell = {"all": summarize(sub, by_time=False), "by_k": {}}
            for k in sorted({r["k"] for r in sub}):
                cell["by_k"][str(k)] = summarize([r for r in sub if r["k"] == k])
            pieces = defaultdict(list)
            for r in sub:
                for piece, v in r["pieces"].items():
                    pieces[piece].append(v)
            cell["text_pieces"] = {piece: {"episodes": len(v), "mean_pct": (np.mean(v, 0) * 100).tolist()} for piece, v in pieces.items()}
            result["cells"][f"{model}|{task}"] = cell
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"READ_TARGETS=PASS episodes={len(rows)}")


if __name__ == "__main__":
    main()
