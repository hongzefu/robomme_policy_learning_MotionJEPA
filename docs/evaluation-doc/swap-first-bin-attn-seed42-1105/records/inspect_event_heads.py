"""闭合那次推理里，专门读开局或交换的几个头具体读哪些帧、patch 和 motion 窗。只读。

头是看完 action-flow 汇总后事后挑的，属探索性。每局只取 G 那次推理。
用法：inspect_event_heads.py <records 目录> <产物根> <输出 json> [--workers n]
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

MEM_FRAME = 512
HEADS = {"text": [(0, 3), (0, 5), (1, 1), (1, 5), (0, 4), (2, 6), (0, 2)], "action": [(3, 0), (16, 6), (13, 6), (17, 1), (14, 0), (16, 0), (16, 7)]}


def one(job):
    d = Path(job["dir"])
    seq = job["G"]
    with np.load(d / f"infer_{seq:04d}.npz") as z:
        mem, order, mask, txt = int(z["MEM"]), np.asarray(z["mem_order"]), np.asarray(z["mem_mask"], bool), int(z["txt_len"])
        ft, mt = np.asarray(z["frame_times"]), np.asarray(z["motion_times"]).reshape(-1)
        action = np.asarray(z["probs_action"], np.float32).mean(axis=(0, 3))
    with np.load(d / f"prefix_{seq:04d}.npz") as z:
        A = z["probs_prefix"].astype(np.float32)
    text = np.arange(mem + 512, mem + 512 + txt)
    out = {}
    fpos = np.where(mask & (order < MEM_FRAME))[0]
    mpos = np.where(mask & (order >= MEM_FRAME))[0]
    times = ft[order[fpos] // 16]
    patch = order[fpos] % 16
    wins = mt[order[mpos] - MEM_FRAME]
    for query, heads in HEADS.items():
        for l, h in heads:
            vec = A[l, h, text].mean(0)[:mem] if query == "text" else action[l, h, :mem]
            rec = {"frame_total": float(vec[fpos].sum()), "motion_total": float(vec[mpos].sum()), "frame_t": {}, "patch_reveal": np.zeros(16).tolist(), "window": {}}
            for t in np.unique(times):
                rec["frame_t"][int(t)] = float(vec[fpos][times == t].sum())
            pr = np.zeros(16)
            sel = times < 32
            np.add.at(pr, patch[sel], vec[fpos][sel])
            rec["patch_reveal"] = pr.tolist()
            for t in np.unique(wins):
                rec["window"][int(t)] = float(vec[mpos][wins == t].sum())
            out[f"{query}|L{l}H{h}"] = rec
    return {"model": job["model"], "task": job["task"], "k": job["k"], "key": job["key"], "heads": out}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    jobs = []
    for model, tag in (("motion", "mot"), ("baseline", "ctx")):
        data = json.loads((args.records / f"{tag}-windows.json").read_text(encoding="utf-8"))
        for e in data["episodes"]:
            if e.get("window"):
                jobs.append({"model": model, "task": e["task"], "k": int(e["swap_times"]), "key": e["episode_key"], "G": int(e["G"]), "dir": str(args.root / tag / "offline" / e["episode_nonce"])})
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, jobs))
    result = {}
    for model in ("motion", "baseline"):
        for task in ("ButtonUnmaskSwap", "VideoUnmaskSwap"):
            sub = [r for r in rows if r["model"] == model and r["task"] == task]
            cell = {}
            for name in sub[0]["heads"]:
                f_total = np.mean([r["heads"][name]["frame_total"] for r in sub])
                m_total = np.mean([r["heads"][name]["motion_total"] for r in sub])
                bins, wins = defaultdict(list), defaultdict(list)
                for r in sub:
                    rec = r["heads"][name]
                    local = defaultdict(float)
                    for t, v in rec["frame_t"].items():
                        local[int(t) // 8 * 8] += v
                    for t, v in local.items():
                        bins[t].append(v)
                    for t, v in rec["window"].items():
                        wins[int(t)].append(v)
                n = len(sub)
                cell[name] = {"frame_total_pct": float(f_total * 100), "motion_total_pct": float(m_total * 100),
                              "frame_bin8_pct": {str(t): float(np.sum(v) / n * 100) for t, v in sorted(bins.items())},
                              "window_pct": {str(t): float(np.sum(v) / n * 100) for t, v in sorted(wins.items())},
                              "patch_reveal_pct": (np.mean([r["heads"][name]["patch_reveal"] for r in sub], 0) * 100).tolist()}
            result[f"{model}|{task}"] = cell
    args.out.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"EVENT_HEADS=PASS episodes={len(rows)}")


if __name__ == "__main__":
    main()
