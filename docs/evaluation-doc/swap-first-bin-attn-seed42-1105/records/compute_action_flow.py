"""全链路注意力：开局、交换等事件的信息怎样一路到动作 token。只读已保存的 prefix、infer。

token 分 13 组：历史帧 4 段、motion 窗 4 段（开局/盖住/交换/其后）、第三人称图像、手腕图像、文字、动作。
motion 窗内容范围 [t,t+32)，演示段截断到 exec_start_idx，与交换有重叠即算交换。

每局每次推理计算：
1. direct：逐层逐头，每个 query 组读每个 key 组的量（组内 token 平均，每行和为 1）。
2. rollout：头平均后 R[l+1] = alpha R[l] + (1-alpha) A[l] R[l]，R 为每个 token 由各来源组成的比例（注意力近似，忽略 MLP 和 value 范数）。
   动作组按去噪步、动作 query 平均成一行，读 prefix 第 l 层输入的 R[l]。
3. route：把动作最终来源分解到 (进入层, 中转组, 来源)。动作读自身只改变保留比例，不算新中转。
4. spatial：文字、动作读历史帧的 4×4 patch 分布（层平均）。
5. pieces：文字 token 片段第 0、1 层读记忆各段的量。

判定：掩码（图像读记忆、记忆读图像/文字恒为 0）、R 行和、route 守恒。
末行 ACTION_FLOW=PASS|FAIL episodes=<n> mask_max=<x> row_err=<x> route_err=<x>。

用法：compute_action_flow.py <records 目录> <产物根> <episodes 目录> <输出目录> [--workers n] [--limit-episodes n]
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
FRAME_TOKENS, TOKENS_PER_FRAME, IMAGE_BASE, IMAGE_WRIST = 512, 16, 256, 256
EVENTS = ("reveal", "cover", "swap", "after")
GROUPS = tuple(f"frame_{e}" for e in EVENTS) + tuple(f"motion_{e}" for e in EVENTS) + ("image_base", "image_wrist", "text", "action")
G = {name: i for i, name in enumerate(GROUPS)}
ALPHAS = (0.5, 0.2, 0.8)


def frame_event(t, end):
    return "reveal" if t < 32 else "cover" if t < 64 else "swap" if t < end else "after"


def motion_event(t, d, end):
    stop = min(t + 32, d) if t < d else t + 32
    if t < end and stop > 64:
        return "swap"
    return "reveal" if t < 32 else "cover" if t < 64 else "after"


def token_groups(meta, P, end):
    mem, order, mask, txt = int(meta["MEM"]), meta["mem_order"], meta["mem_mask"], int(meta["txt_len"])
    labels = np.full(P, -1, np.int64)
    ft, mt = meta["frame_times"], meta["motion_times"]
    d = int(meta["exec_start_idx"])
    for s in range(mem):
        if not mask[s]:
            continue
        o = int(order[s])
        if o < FRAME_TOKENS:
            labels[s] = G["frame_" + frame_event(int(ft[o // TOKENS_PER_FRAME]), end)]
        else:
            labels[s] = G["motion_" + motion_event(int(mt[o - FRAME_TOKENS]), d, end)]
    labels[mem:mem + IMAGE_BASE] = G["image_base"]
    labels[mem + IMAGE_BASE:mem + IMAGE_BASE + IMAGE_WRIST] = G["image_wrist"]
    labels[mem + IMAGE_BASE + IMAGE_WRIST:mem + IMAGE_BASE + IMAGE_WRIST + txt] = G["text"]
    return labels


def one_inference(d, seq, end, pieces):
    with np.load(d / f"infer_{seq:04d}.npz") as z:
        meta = {k: z[k] for k in ("MEM", "mem_order", "mem_mask", "txt_len", "P", "frame_times", "motion_times", "exec_start_idx")}
        meta["motion_times"] = np.asarray(meta["motion_times"]).reshape(-1)
        action = np.asarray(z["probs_action"], np.float32).mean(axis=(0, 3))  # (层, 头, P+20)
    with np.load(d / f"prefix_{seq:04d}.npz") as z:
        A = z["probs_prefix"].astype(np.float32)
    P = int(meta["P"])
    labels = token_groups(meta, P, end)
    n_groups = len(GROUPS)
    weights = np.zeros((P, n_groups), np.float32)
    valid = labels >= 0
    weights[np.where(valid)[0], labels[valid]] = 1
    counts = weights.sum(0)
    direct = np.zeros((LAYERS, HEADS, n_groups, n_groups), np.float32)
    query_mean = weights / np.maximum(counts, 1)
    for layer in range(LAYERS):
        for head in range(HEADS):
            direct[layer, head, :n_groups - 1, :n_groups - 1] = (query_mean.T @ (A[layer, head] @ weights))[:n_groups - 1, :n_groups - 1]
            direct[layer, head, G["action"], :n_groups - 1] = (action[layer, head, :P] @ weights)[:n_groups - 1]
            direct[layer, head, G["action"], G["action"]] = action[layer, head, P:].sum()
    mem_idx = [G[x] for x in GROUPS[:8]]
    image_idx = [G["image_base"], G["image_wrist"]]
    mask_max = max(float(direct[:, :, image_idx][:, :, :, mem_idx].max()),
                   float(direct[:, :, mem_idx][:, :, :, image_idx + [G["text"], G["action"]]].max()))
    mean_a = A.mean(1)
    action_mean = action.mean(1)
    rollouts, routes, row_err, route_err = {}, {}, 0.0, 0.0
    for alpha in ALPHAS:
        R = weights.copy()
        r_act = np.zeros(n_groups)
        r_act[G["action"]] = 1
        track = np.zeros((LAYERS + 1, n_groups, n_groups), np.float32)
        track[0] = query_mean.T @ R
        track[0, G["action"]] = r_act
        route = np.zeros((LAYERS, n_groups - 1, n_groups), np.float64)
        keep_init = 1.0
        for layer in range(LAYERS):
            row = action_mean[layer]
            read = row[:P] @ R  # 动作读 prefix 第 layer 层输入
            per_via = np.zeros((n_groups - 1, n_groups))
            for g in range(n_groups - 1):
                per_via[g] = (row[:P] * weights[:, g]) @ R
            self_mass = float(row[P:].sum())
            factor = alpha + (1 - alpha) * self_mass
            route *= factor
            keep_init *= factor
            route[layer] = (1 - alpha) * per_via
            r_act = factor * r_act + (1 - alpha) * read
            rows = mean_a[layer]
            rowsum = rows.sum(1)
            active = rowsum > .5
            next_r = R.copy()
            next_r[active] = alpha * R[active] + (1 - alpha) * (rows[active] @ R)
            R = next_r
            track[layer + 1] = query_mean.T @ R
            track[layer + 1, G["action"]] = r_act
        rebuilt = route.sum((0, 1))
        rebuilt[G["action"]] += keep_init
        route_err = max(route_err, float(np.abs(rebuilt - r_act).max()))
        row_err = max(row_err, float(np.abs(R[valid].sum(1) - 1).max()), float(abs(r_act.sum() - 1)))
        rollouts[alpha], routes[alpha] = track, route
    # 空间与文字片段
    mem = int(meta["MEM"])
    spatial = np.zeros((2, len(EVENTS), 16), np.float32)
    text_pos = np.where(labels == G["text"])[0]
    views = (mean_a.mean(0)[text_pos].mean(0)[:mem], action_mean.mean(0)[:mem])
    frame_tokens = [(s, EVENTS.index(GROUPS[labels[s]].split("_")[1]), int(meta["mem_order"][s]) % TOKENS_PER_FRAME) for s in range(mem) if 0 <= labels[s] < 4]
    for s, e, patch in frame_tokens:
        spatial[0, e, patch] += views[0][s]
        spatial[1, e, patch] += views[1][s]
    piece_rows = {}
    for i, s in enumerate(text_pos):
        if i < len(pieces):
            piece_rows.setdefault(pieces[i], []).append(mean_a[:2, s, :P] @ weights[:, :8])
    del A, action, mean_a, action_mean
    return {"direct": direct, "rollouts": rollouts, "routes": routes, "spatial": spatial, "counts": counts,
            "mask_max": mask_max, "row_err": row_err, "route_err": route_err,
            "pieces": {p: np.mean(v, 0) for p, v in piece_rows.items()}}


def one_episode(job):
    end = 64 + 50 * job["k"]
    results = [one_inference(Path(job["dir"]), seq, end, job["pieces"]) for seq in job["window"]]
    avg = lambda f: np.mean([f(r) for r in results], 0)
    data = {"direct": avg(lambda r: r["direct"]), "spatial": avg(lambda r: r["spatial"]), "counts": avg(lambda r: r["counts"])}
    for alpha in ALPHAS:
        data[f"rollout_{alpha}"] = avg(lambda r: r["rollouts"][alpha])
        data[f"route_{alpha}"] = avg(lambda r: r["routes"][alpha])
    out = Path(job["out"]) / f"{job['model']}_{job['key'].replace(':', '_')}.npz"
    np.savez_compressed(out, **data)
    pieces = {}
    for r in results:
        for p, v in r["pieces"].items():
            pieces.setdefault(p, []).append(v)
    return {"key": job["key"], "task": job["task"], "k": job["k"], "model": job["model"], "label": job["label"], "outcome": job["outcome"],
            "mask_max": max(r["mask_max"] for r in results), "row_err": max(r["row_err"] for r in results), "route_err": max(r["route_err"] for r in results),
            "pieces": {p: np.mean(v, 0).tolist() for p, v in pieces.items()}, "file": out.name}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("episodes", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-episodes", type=int)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for model, tag, prefix in (("motion", "mot", "with_motion"), ("baseline", "ctx", "without_motion")):
        data = json.loads((args.records / f"{tag}-windows.json").read_text(encoding="utf-8"))
        eps = [e for e in data["episodes"] if e.get("window")]
        for e in eps[:args.limit_episodes] if args.limit_episodes else eps:
            detail = json.loads((args.episodes / f"{prefix}_{e['episode_key'].replace(':', '_')}" / "episode_window.json").read_text(encoding="utf-8"))
            jobs.append({"key": e["episode_key"], "task": e["task"], "k": int(e["swap_times"]), "model": model, "window": e["window"], "label": e["label"],
                         "outcome": e["outcome"], "pieces": detail["records"][0]["text_token_pieces"], "dir": str(args.root / tag / "offline" / e["episode_nonce"]), "out": str(args.out)})
    print(f"开始 episodes={len(jobs)}", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(one_episode, jobs), 1):
            rows.append(row)
            print(f"完成 {i}/{len(jobs)} {row['model']} {row['key']} mask={row['mask_max']:.2e} row={row['row_err']:.2e} route={row['route_err']:.2e}", flush=True)
    (args.out / "index.json").write_text(json.dumps({"groups": GROUPS, "alphas": ALPHAS, "episodes": rows}, ensure_ascii=False) + "\n", encoding="utf-8")
    mask, row_err, route = max(r["mask_max"] for r in rows), max(r["row_err"] for r in rows), max(r["route_err"] for r in rows)
    ok = mask <= 1e-6 and row_err <= 1e-2 and route <= 1e-3
    print(f"ACTION_FLOW={'PASS' if ok else 'FAIL'} episodes={len(rows)} mask_max={mask:.3e} row_err={row_err:.3e} route_err={route:.3e}")


if __name__ == "__main__":
    main()
