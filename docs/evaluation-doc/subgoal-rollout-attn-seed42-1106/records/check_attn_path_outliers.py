"""去掉基线 attn_path_maxabs>0.05 的 4 局后 D 的变化。用法：check_attn_path_outliers.py <analysis/episodes 目录>"""
import glob
import json
import sys
import numpy as np
BAD = {"RouteStick:40", "PatternLock:23", "PatternLock:30", "RouteStick:31"}
for task in ("PatternLock", "RouteStick"):
    for reader in ("action", "text"):
        full, kept = [], []
        for path in glob.glob(f"{sys.argv[1]}/ctx_{task}_*.json"):
            ep = json.load(open(path, encoding="utf-8"))
            if ep["m"] < 2 or ep["readouts"][f"{reader}/frame/single"]["D"] is None:
                continue
            full.append(ep["readouts"][f"{reader}/frame/single"]["D"])
            if ep["episode_key"] not in BAD:
                kept.append(full[-1])
        print(f"{task} {reader} 全部={np.mean(full):.3f} n={len(full)} 去异常局={np.mean(kept):.3f} n={len(kept)}")
