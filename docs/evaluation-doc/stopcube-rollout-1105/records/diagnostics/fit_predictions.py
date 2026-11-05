"""一次性重算（CPU，不新增推理）：第 17 层各组与时间基线的折外预测，供拟合直线图使用。

只调用 rollout_probe 的函数和同一折划分，不改分析代码。R² 与原 probe-summary.json 不同就报错。
用法：python fit_predictions.py <产物根> <原 probe-summary.json> <输出目录>
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from scripts.interpret.analysis import rollout_probe as P
from scripts.interpret.analysis.rollout_attention import load_align


def main():
    root, original, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    output.mkdir(parents=True, exist_ok=True)
    old = json.load(open(original))
    align = load_align(root)
    saved, summary, checked = {}, {}, 0
    tally = P.Tally()
    for model in ("ctx", "mot"):
        items = align["models"][model]
        bank = P.load_bank(root, model, items)
        summary[model] = {}
        for label in P.LABELS:
            samples = P.build_samples(bank, items, label)
            rows = samples["rows"]
            reference = old["models"][model]["labels"][label]
            readers = {"action": (bank["action"][rows, P.MAIN_LAYER], reference["layers"]["action"][P.MAIN_LAYER]["r2"]),
                       "time_N": (P.time_features(samples), reference["time_baseline"]["r2"]),
                       "time": (samples["t"][:, None], None)}
            for column, name in enumerate(P.PREFIX_GROUPS):
                if column in P.active_columns(bank["motion_present"]):
                    readers[name] = (bank["prefix"][rows, P.LAST_PREFIX, column],
                                     reference["layers"]["prefix"][name][P.MAIN_LAYER]["r2"])
            summary[model][label] = {"n_samples": int(len(rows)), "n_episodes": int(len(np.unique(samples["groups"])))}
            saved[f"{model}/{label}/y"] = samples["y"]
            saved[f"{model}/{label}/groups"] = samples["groups"]
            for name, (X, expected) in readers.items():
                entry, _, prediction = P.evaluate(tally, X, samples)
                if expected is not None:
                    if abs(entry["r2"] - expected) > 1e-6:
                        raise SystemExit(f"{model} {label} {name} R² 与原结果不同：{entry['r2']} 对 {expected}")
                    checked += 1
                saved[f"{model}/{label}/{name}"] = prediction
                summary[model][label][name] = {"r2": entry["r2"], "mae": entry["mae"]}
                print(f"FIT model={model} label={label} reader={name} r2={entry['r2']:.4f} mae={entry['mae']:.2f}", flush=True)
    np.savez(output / "fit-predictions.npz", **saved)
    (output / "fit-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(f"FIT_R2_MATCH=PASS checked={checked} group_leak={tally.leak}", flush=True)


if __name__ == "__main__":
    main()
