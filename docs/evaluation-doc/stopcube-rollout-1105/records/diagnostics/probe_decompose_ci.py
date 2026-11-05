"""一次性（不进仓库，CPU）：分解探针相对时间基线的按局 MAE 差和区间，同一折划分。"""
import json, os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
from scripts.interpret.analysis import rollout_probe as P
from scripts.interpret.analysis.rollout_attention import load_align
align = load_align(root); out = {}
C = {n: i for i, n in enumerate(P.PREFIX_GROUPS)}
for model in ("ctx", "mot"):
    items = align["models"][model]; bank = P.load_bank(root, model, items); out[model] = {}
    for label in P.LABELS:
        s = P.build_samples(bank, items, label); pre = bank["prefix"][s["rows"]]; t = P.Tally(); time = P.time_features(s)
        F = {"时间基线": time, "全部K9": P.k9_layer0(pre, bank["motion_present"]),
             "文字+历史帧": np.concatenate([pre[:, 0, C["文字"]], pre[:, 0, C["历史视觉"]]], 1)}
        if bank["motion_present"]:
            F["文字+motion"] = np.concatenate([pre[:, 0, C["文字"]], pre[:, 0, C["motion"]]], 1)
            F["[t,N]+motion"] = np.concatenate([time, pre[:, 0, C["motion"]]], 1)
            F["motion单独"] = pre[:, 0, C["motion"]]
        E = {k: P.evaluate(t, X, s) for k, X in F.items()}
        res = {"mae": {k: E[k][0]["mae"] for k in E}, "r2": {k: E[k][0]["r2"] for k in E}}
        res["minus_time"] = {k: P.mae_difference(E[k][1], E["时间基线"][1]) for k in E if k != "时间基线"}
        if "文字+motion" in E:
            res["文字+motion_minus_文字+历史帧"] = P.mae_difference(E["文字+motion"][1], E["文字+历史帧"][1])
        out[model][label] = res
        print(model, label, {k: (round(v["mean"], 1), [round(x, 1) for x in v["ci95"]], v["n_episodes"]) for k, v in res["minus_time"].items()}, flush=True)
        if "文字+motion_minus_文字+历史帧" in res:
            v = res["文字+motion_minus_文字+历史帧"]; print("  文字+motion−文字+历史帧", round(v["mean"], 1), [round(x, 1) for x in v["ci95"]], flush=True)
(root / "analysis" / "probe-decompose-ci.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
print("DONE leak", sum(1 for _ in []), flush=True)
