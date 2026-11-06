"""从正式 motion rollout 记录提取 motion 编码，生成替换实验的供体文件和供体映射。

每局取最后一次推理的 motion_emb、motion_times（有效行在前），写：
  <out>/full/<task>_ep<id>.npz   全部窗口，E1 用
  <out>/swap/<task>_ep<id>.npz   只含与交换区间重叠的窗口（t < 64+50k 且 t+33 > 64），E2 用
  <out>/swap_map.json            每局 -> 同任务、同交换次数、同 bin 数组内按局号循环的下一局
组内只有一局时没有合法供体，停止并报错，不退回其他分组。

用法：python derive_motion_donors.py <windows.json> <attn-root> [<attn-root> ...] --out <目录>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def last_record(attn_roots, nonce):
    paths = [path for root in attn_roots for path in sorted(Path(root).glob(f"{nonce}/rollout_*.npz"))]
    if not paths:
        raise SystemExit(f"nonce={nonce} 没有 rollout 记录")
    return paths[-1]


def extract(path):
    """返回 (起点, 编码)；每窗 1 个 token。"""
    with np.load(path, allow_pickle=False) as z:
        times = np.asarray(z["motion_times"], np.int32)
        emb = np.asarray(z["obs/motion_emb"], np.float32)[0]
    k = int((times >= 0).sum())
    if k == 0 or np.any(times[k:] != -1) or np.any(np.diff(times[:k]) <= 0):
        raise SystemExit(f"{path}：motion_times 不是有效行在前的升序")
    return times[:k], emb[:k]


def overlaps_swap(times, swap_times):
    return (times < 64 + 50 * swap_times) & (times + 33 > 64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("windows", type=Path)
    parser.add_argument("attn_roots", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    episodes = json.loads(args.windows.read_text(encoding="utf-8"))["episodes"]
    (args.out / "full").mkdir(parents=True, exist_ok=False)
    (args.out / "swap").mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    windows_total = swap_total = 0
    for entry in episodes:
        name = f"{entry['task']}_ep{entry['episode_id']}.npz"
        times, emb = extract(last_record(args.attn_roots, entry["episode_nonce"]))
        np.savez(args.out / "full" / name, start_frames=times, embs=emb)
        keep = overlaps_swap(times, int(entry["swap_times"]))
        if not keep.any():
            raise SystemExit(f"{entry['episode_key']}：没有与交换区间重叠的窗口")
        np.savez(args.out / "swap" / name, start_frames=times[keep], embs=emb[keep])
        windows_total += times.size
        swap_total += int(keep.sum())
        groups[entry["task"], entry["swap_times"], entry["bin_count"]].append(int(entry["episode_id"]))
    swap_map = {}
    for (task, swap_times, bin_count), ids in sorted(groups.items()):
        ids = sorted(ids)
        if len(ids) < 2:
            raise SystemExit(f"{task} swap={swap_times} bin={bin_count} 只有 {len(ids)} 局，没有合法供体")
        for index, episode_id in enumerate(ids):
            swap_map[f"{task}:{episode_id}"] = f"{task}:{ids[(index + 1) % len(ids)]}"
    (args.out / "swap_map.json").write_text(json.dumps(swap_map, indent=1) + "\n", encoding="utf-8")
    sizes = {f"{k[0]}|swap={k[1]}|bin={k[2]}": len(v) for k, v in sorted(groups.items())}
    print(f"MOTION_DONORS=PASS episodes={len(episodes)} windows={windows_total} swap_windows={swap_total} groups={len(groups)}")
    print("DONOR_GROUPS=" + json.dumps(sizes, ensure_ascii=False))


if __name__ == "__main__":
    main()
