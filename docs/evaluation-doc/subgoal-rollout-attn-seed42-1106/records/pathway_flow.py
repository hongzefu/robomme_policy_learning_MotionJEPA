"""PatternLock、RouteStick 上的全链路注意力：帧吸收 motion、文字读记忆、动作吸收。公式同 Swap 的 compute_action_flow.py。

token 分 8 组：示范帧、执行帧、示范 motion、执行 motion、第三人称图像、手腕图像、文字、动作。
每次推理算 direct（逐层逐头，读者组读来源组，组内平均）与 alpha=0.5、0.2、0.8 的 rollout、route；每局先平均。
累积只是注意力近似，忽略 MLP 和 value 范数。判定：掩码、R 行和、route 守恒。末行 PATHWAY_FLOW=PASS|FAIL。

用法：
  pathway_flow.py compute <full-attn 根> <输出目录> [--workers n] [--limit n]
  pathway_flow.py crosscheck <Swap infer 目录> <seq> <end> <compute_action_flow.py 所在目录>
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

LAYERS, HEADS = 18, 8
FRAME_TOKENS, TOKENS_PER_FRAME, IMAGE_BASE, IMAGE_WRIST = 512, 16, 256, 256
GROUPS = ("frame_demo", "frame_exec", "motion_demo", "motion_exec", "image_base", "image_wrist", "text", "action")
G = {name: i for i, name in enumerate(GROUPS)}
MEMORY = [G["frame_demo"], G["frame_exec"], G["motion_demo"], G["motion_exec"]]
IMAGES = [G["image_base"], G["image_wrist"]]
ALPHAS = (0.5, 0.2, 0.8)


def token_labels(meta, P):
    mem, order, mask, txt = int(meta["MEM"]), meta["mem_order"], meta["mem_mask"], int(meta["txt_len"])
    labels = np.full(P, -1, np.int64)
    frame_times, motion_times, demo = meta["frame_times"], meta["motion_times"], int(meta["exec_start_idx"])
    for s in range(mem):
        if not mask[s]:
            continue
        o = int(order[s])
        if o < FRAME_TOKENS:
            labels[s] = G["frame_demo"] if int(frame_times[o // TOKENS_PER_FRAME]) < demo else G["frame_exec"]
        else:
            labels[s] = G["motion_demo"] if int(motion_times[o - FRAME_TOKENS]) < demo else G["motion_exec"]
    labels[mem:mem + IMAGE_BASE] = G["image_base"]
    labels[mem + IMAGE_BASE:mem + IMAGE_BASE + IMAGE_WRIST] = G["image_wrist"]
    labels[mem + IMAGE_BASE + IMAGE_WRIST:mem + IMAGE_BASE + IMAGE_WRIST + txt] = G["text"]
    return labels


def one_inference(directory, seq):
    with np.load(directory / f"infer_{seq:04d}.npz") as z:
        meta = {k: z[k] for k in ("MEM", "mem_order", "mem_mask", "txt_len", "P", "frame_times", "motion_times", "exec_start_idx", "infer_seq")}
        meta["motion_times"] = np.asarray(meta["motion_times"]).reshape(-1)
        action = np.asarray(z["probs_action"], np.float32).mean(axis=(0, 3))  # (层, 头, P+20)
    with np.load(directory / f"prefix_{seq:04d}.npz") as z:
        A = z["probs_prefix"].astype(np.float32)
    P = int(meta["P"])
    labels = token_labels(meta, P)
    n = len(GROUPS)
    weights = np.zeros((P, n), np.float32)
    valid = labels >= 0
    weights[np.where(valid)[0], labels[valid]] = 1
    counts = weights.sum(0)
    direct = np.zeros((LAYERS, HEADS, n, n), np.float32)
    query_mean = weights / np.maximum(counts, 1)
    for layer in range(LAYERS):
        for head in range(HEADS):
            direct[layer, head, :n - 1, :n - 1] = (query_mean.T @ (A[layer, head] @ weights))[:n - 1, :n - 1]
            direct[layer, head, G["action"], :n - 1] = (action[layer, head, :P] @ weights)[:n - 1]
            direct[layer, head, G["action"], G["action"]] = action[layer, head, P:].sum()
    mask_max = max(float(direct[:, :, IMAGES][:, :, :, MEMORY].max()),
                   float(direct[:, :, MEMORY][:, :, :, IMAGES + [G["text"], G["action"]]].max()))
    mean_a, action_mean = A.mean(1), action.mean(1)
    rollouts, routes, row_err, route_err = {}, {}, 0.0, 0.0
    for alpha in ALPHAS:
        R = weights.copy()
        r_act = np.zeros(n)
        r_act[G["action"]] = 1
        track = np.zeros((LAYERS + 1, n, n), np.float32)
        track[0] = query_mean.T @ R
        track[0, G["action"]] = r_act
        route = np.zeros((LAYERS, n - 1, n), np.float64)
        keep = 1.0
        for layer in range(LAYERS):
            row = action_mean[layer]
            read = row[:P] @ R
            per_via = np.zeros((n - 1, n))
            for g in range(n - 1):
                per_via[g] = (row[:P] * weights[:, g]) @ R
            self_mass = float(row[P:].sum())
            factor = alpha + (1 - alpha) * self_mass
            route *= factor
            keep *= factor
            route[layer] = (1 - alpha) * per_via
            r_act = factor * r_act + (1 - alpha) * read
            rows = mean_a[layer]
            active = rows.sum(1) > .5
            next_r = R.copy()
            next_r[active] = alpha * R[active] + (1 - alpha) * (rows[active] @ R)
            R = next_r
            track[layer + 1] = query_mean.T @ R
            track[layer + 1, G["action"]] = r_act
        rebuilt = route.sum((0, 1))
        rebuilt[G["action"]] += keep
        route_err = max(route_err, float(np.abs(rebuilt - r_act).max()))
        row_err = max(row_err, float(np.abs(R[valid].sum(1) - 1).max()), float(abs(r_act.sum() - 1)))
        rollouts[alpha], routes[alpha] = track, route
    return {"direct": direct, "rollouts": rollouts, "routes": routes, "counts": counts, "mask_max": mask_max, "row_err": row_err, "route_err": route_err}


def one_episode(job):
    directory = Path(job["dir"])
    seqs = sorted(int(p.stem.split("_")[1]) for p in directory.glob("infer_*.npz"))
    results = [one_inference(directory, seq) for seq in seqs]
    avg = lambda f: np.mean([f(r) for r in results], 0)
    data = {"direct": avg(lambda r: r["direct"]), "counts": avg(lambda r: r["counts"])}
    for alpha in ALPHAS:
        data[f"rollout_{alpha}"] = avg(lambda r: r["rollouts"][alpha])
        data[f"route_{alpha}"] = avg(lambda r: r["routes"][alpha])
    np.savez(Path(job["out"]) / f"{directory.name}.npz", **data)
    return {"nonce": directory.name, "inferences": len(seqs), "mask_max": max(r["mask_max"] for r in results),
            "row_err": max(r["row_err"] for r in results), "route_err": max(r["route_err"] for r in results)}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("compute", "crosscheck"))
parser.add_argument("paths", nargs="+")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--limit", type=int)
args = parser.parse_args()
if args.mode == "crosscheck":
    directory, seq, end, record = Path(args.paths[0]), int(args.paths[1]), int(args.paths[2]), args.paths[3]
    sys.path.insert(0, record)
    import compute_action_flow as old
    theirs, ours = old.one_inference(directory, seq, end, []), one_inference(directory, seq)
    # 12 组与 8 组都折成：帧、motion、第三人称、手腕、文字、动作
    fold_old = [list(range(4)), list(range(4, 8)), [8], [9], [10], [11]]
    fold_new = [[0, 1], [2, 3], [4], [5], [6], [7]]
    def fold(matrix, parts, counts):
        out = np.zeros((6, 6))
        for i, q in enumerate(parts):
            w = counts[q] / max(counts[q].sum(), 1e-9)
            for j, k in enumerate(parts):
                out[i, j] = (w[:, None] * matrix[np.ix_(q, k)]).sum()
        return out
    err_direct = max(float(np.abs(fold(theirs["direct"][l].mean(0), fold_old, theirs["counts"]) - fold(ours["direct"][l].mean(0), fold_new, ours["counts"])).max()) for l in range(LAYERS))
    err_roll = max(float(np.abs(fold(theirs["rollouts"][a][l], fold_old, theirs["counts"]) - fold(ours["rollouts"][a][l], fold_new, ours["counts"])).max()) for a in ALPHAS for l in range(LAYERS + 1))
    ok = err_direct < 1e-3 and err_roll < 1e-3
    print(f"PATHWAY_CROSSCHECK={'PASS' if ok else 'FAIL'} direct_err={err_direct:.3g} rollout_err={err_roll:.3g} tol=1e-3 counts={ours['counts'].tolist()}", flush=True)
    raise SystemExit(0 if ok else 1)
root, out = Path(args.paths[0]), Path(args.paths[1])
out.mkdir(parents=True, exist_ok=True)
jobs = [{"dir": str(p), "out": str(out)} for p in sorted(root.iterdir()) if p.is_dir() and any(p.glob("infer_*.npz"))]
if args.limit:
    jobs = jobs[:args.limit]
with ProcessPoolExecutor(args.workers) as pool:
    rows = list(pool.map(one_episode, jobs))
line = (f"episodes={len(rows)} inferences={sum(r['inferences'] for r in rows)} mask_max={max(r['mask_max'] for r in rows):.3e} "
        f"row_err={max(r['row_err'] for r in rows):.3e} route_err={max(r['route_err'] for r in rows):.3e}")
ok = rows and max(r["mask_max"] for r in rows) <= 1e-3 and max(r["row_err"] for r in rows) <= 2e-3 and max(r["route_err"] for r in rows) <= 1e-3
(out / "index.json").write_text(json.dumps({"groups": GROUPS, "alphas": ALPHAS, "episodes": rows}, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"PATHWAY_FLOW={'PASS' if ok else 'FAIL'} {line}", flush=True)
raise SystemExit(0 if ok else 1)
