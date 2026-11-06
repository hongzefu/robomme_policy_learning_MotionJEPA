"""逐层读取：文字、动作各自读帧记忆、motion 记忆、当前画面、文字的量。只读已存产物，事后补做。

用法：layer_flow.py <产物根> <输出 json>。头平均，每局先平均，每局等权；单位是占该读者注意力的百分比。
"""
import glob
import json
import sys
from pathlib import Path
import numpy as np
root, out = Path(sys.argv[1]), Path(sys.argv[2])
BLOCKS = ("历史视觉", "motion", "当前视觉", "文字", "其余")
accepted = {}
for model in ("mot", "ctx"):
    for path in glob.glob(str(root / model / "s*" / "run" / "accepted" / "*.terminal.json")):
        record = json.load(open(path, encoding="utf-8"))
        accepted[model, record["episode_nonce"]] = record
result = {}
for model in ("mot", "ctx"):
    per = {}
    for path in glob.glob(str(root / model / "offline" / "*" / "offline_*.npz")):
        z = np.load(path)
        names = [str(n) for n in z["group_names"]]
        key = str(z["episode_key"])
        for reader, array in (("action", z["group_mass_action"]), ("text", z["group_mass_text"])):
            layers = np.stack([array[names.index(b)].mean(1) for b in BLOCKS], 1)      # (层, 块)
            per.setdefault((key.split(":")[0], reader, key), []).append(layers)
    for task in ("PatternLock", "RouteStick"):
        for reader in ("action", "text"):
            eps = [np.mean(v, 0) for k, v in per.items() if k[0] == task and k[1] == reader]
            mean = np.mean(eps, 0) * 100
            result[f"{task}/{model}/{reader}"] = {"blocks": BLOCKS, "by_layer_pct": mean.round(3).tolist(), "n_episodes": len(eps)}
            print(task, model, reader, "局", len(eps))
            for layer in range(18):
                print(f"  L{layer:<2d} 帧 {mean[layer,0]:6.2f} motion {mean[layer,1]:6.2f} 当前 {mean[layer,2]:6.2f} 文字 {mean[layer,3]:6.2f} 动作 {mean[layer,4]:6.2f}")
out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
