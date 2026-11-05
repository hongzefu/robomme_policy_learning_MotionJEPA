"""一次性分解（不进仓库，CPU，不新增推理）：第 0 层拼接探针的信息来自哪几组。

只复用 rollout_probe 的函数和同一折划分；输出写 analysis/probe-decompose.json，作为留档的计划外描述。
"""
import json, os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
from scripts.interpret.analysis import rollout_probe as P
from scripts.interpret.analysis.rollout_attention import load_align

align = load_align(root)
out = {}
COL = {name: i for i, name in enumerate(P.PREFIX_GROUPS)}
for model in ("ctx", "mot"):
    items = align["models"][model]
    bank = P.load_bank(root, model, items)
    out[model] = {}
    for label in P.LABELS:
        samples = P.build_samples(bank, items, label)
        prefix = bank["prefix"][samples["rows"]]
        tally = P.Tally()
        feats = {}
        names = ["文字", "历史视觉", "当前视觉"] + (["motion"] if bank["motion_present"] else [])
        for name in names:
            feats[f"pt:{name}"] = prefix[:, 0, COL[name]]
        for name in names[1:]:
            feats[f"pt:文字+{name}"] = np.concatenate([prefix[:, 0, COL["文字"]], prefix[:, 0, COL[name]]], axis=1)
        feats["pt:全部(K9)"] = P.k9_layer0(prefix, bank["motion_present"])
        time = P.time_features(samples)
        feats["时间基线[t,N]"] = time
        feats["[t,N]+pt:文字"] = np.concatenate([time, prefix[:, 0, COL["文字"]]], axis=1)
        if bank["motion_present"]:
            feats["[t,N]+pt:motion"] = np.concatenate([time, prefix[:, 0, COL["motion"]]], axis=1)
        feats["[t,N]+pt:历史视觉"] = np.concatenate([time, prefix[:, 0, COL["历史视觉"]]], axis=1)
        result = {}
        for name, X in feats.items():
            m, _, _ = P.evaluate(tally, X, samples)
            result[name] = {"mae": m["mae"], "r2": m["r2"], "n_samples": m["n_samples"], "n_episodes": m["n_episodes"]}
            print(model, label, name, round(m["mae"], 1), round(m["r2"], 3), flush=True)
        result["group_leak"] = tally.leak
        out[model][label] = result
(root / "analysis" / "probe-decompose.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
print("PROBE_DECOMPOSE=DONE leak=", sum(v["group_leak"] for m in out.values() for v in m.values()), flush=True)
