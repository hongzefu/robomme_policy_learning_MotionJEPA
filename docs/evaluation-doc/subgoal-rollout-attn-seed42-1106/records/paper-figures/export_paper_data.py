"""导出论文主图（fig_pathway）全部原始数据为 CSV，供其他仓库画图；不依赖本仓库代码。

数据源与 plot_pathway_figure.py 相同：
- 逐局通路 npz：v1-store/evaluation/subgoal-rollout-attn-seed42-1106/mot/pathway/<nonce>.npz（direct 形状 (18 层, 8 头, 8 组, 8 组)）
- 任务归属：…/mot/*/run/accepted/*.terminal.json
- 序列长度：任一 …/mot/full-attn/<nonce>/infer_0000.npz
- 子目标对位：records/summary.json（A2）、success-only.json、decompose.json

校验：导出后读回 pathway_heatmap.csv 的 pooled 行与 fig_pathway-layers.json 的 heatmap_pct 逐格比对（参考 JSON 存两位小数，容差 5.1e-3），
以及 subgoal_alignment_matrix.csv 与 summary.json 的 enrichment 逐格比对（容差 1e-6）；末行 PAPER_DATA_EXPORT=PASS|FAIL files=<n> …。
用法：export_paper_data.py [仓库根]
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/hongzefu/robomme_policy_learning_MotionJEPA")
REC = REPO / "docs/evaluation-doc/subgoal-rollout-attn-seed42-1106/records"
STORE = REPO / "v1-store/evaluation/subgoal-rollout-attn-seed42-1106/mot"
OUT = REC / "paper-data"
OUT.mkdir(exist_ok=True)

GROUPS8 = ("frame_demo", "frame_exec", "motion_demo", "motion_exec", "image_base", "image_wrist", "text", "action")
BLOCKS = [("frames", [0, 1]), ("motion", [2, 3]), ("image_3rd", [4]), ("image_wrist", [5]), ("language", [6]), ("action", [7])]
TASKS = ("PatternLock", "RouteStick")

# ----------------------------------------------------------------------------- 读逐局 npz
task_of = {}
for path in glob.glob(str(STORE / "*" / "run" / "accepted" / "*.terminal.json")):
    r = json.load(open(path, encoding="utf-8"))
    task_of[r["episode_nonce"]] = r["episode_key"].split(":")[0]
episodes = {t: [] for t in TASKS}
for path in sorted((STORE / "pathway").glob("*.npz")):
    with np.load(path) as z:
        episodes[task_of[path.stem]].append({"direct": z["direct"].mean(1), "direct_max": z["direct"].max(1), "counts": z["counts"],
                                             "carry": z["rollout_0.5"], **{f"route_{a}": z[f"route_{a}"] for a in ("0.2", "0.5", "0.8")}})
index = json.load(open(STORE / "pathway" / "index.json", encoding="utf-8"))
inferences = sum(e["inferences"] for e in index["episodes"])


def fold6(direct, counts):
    M = np.zeros((direct.shape[0], 6, 6))
    for a, (_, q) in enumerate(BLOCKS):
        w = counts[q] / max(counts[q].sum(), 1e-9) if len(q) > 1 else np.ones(1)
        for b, (_, k) in enumerate(BLOCKS):
            M[:, a, b] = (w[None, :, None] * direct[:, q][:, :, k]).sum((1, 2))
    return M


def frame_carry(carry, counts):
    w = counts[[0, 1]] / counts[[0, 1]].sum()
    return np.array([(w[:, None] * c[[0, 1]][:, [2, 3]]).sum() for c in carry])


per_task = {}
for t in TASKS:
    items = episodes[t]
    per_task[t] = {
        "episodes": len(items),
        "M6": np.mean([fold6(i["direct"], i["counts"]) for i in items], 0) * 100,
        # 与 plot_pathway_figure.py 同口径：先对局平均 rollout 与 token 数，再按平均 token 数加权折帧
        "carry": frame_carry(np.mean([i["carry"] for i in items], 0), np.mean([i["counts"] for i in items], 0)) * 100,
        "counts": np.mean([i["counts"] for i in items], 0),
        "motion_tokens": float(np.mean([i["counts"][[2, 3]].sum() for i in items])),
        "route": {a: np.mean([i[f"route_{a}"].sum(0) for i in items], 0)[:, [2, 3]].sum(1) * 100 for a in ("0.2", "0.5", "0.8")},
    }
pooled = {"M6": np.mean([per_task[t]["M6"] for t in TASKS], 0), "carry": np.mean([per_task[t]["carry"] for t in TASKS], 0)}
mask_zero8 = np.max([i["direct_max"].max(0) for t in TASKS for i in episodes[t]], 0) == 0
with np.load(next((STORE / "full-attn").glob("*/infer_0000.npz"))) as z:
    MEM, TXT, P, HORIZON = int(z["MEM"]), int(z["txt_len"]), int(z["P"]), int(z["probs_action"].shape[3])
    N_IMG = (P - MEM - int(z["probs_text"].shape[2])) // 2
    FRAME_TOKENS = int((np.asarray(z["mem_order"]) < 512).sum())
sizes = {"frames": FRAME_TOKENS, "motion": None, "image_3rd": N_IMG, "image_wrist": N_IMG, "language": TXT, "action": HORIZON}

files = []


def write(name, header, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    files.append((name, len(rows)))


# 1 掩码块矩阵
rows = []
for a, (qn, qs) in enumerate(BLOCKS):
    for b, (kn, ks) in enumerate(BLOCKS):
        allowed = not all(mask_zero8[q, k] for q in qs for k in ks)
        rows.append([qn, kn, int(allowed), "causal" if (qn == "action" and kn == "action") else ("full" if allowed else "masked"),
                     sizes[qn] if sizes[qn] is not None else f"{per_task['PatternLock']['motion_tokens']:.2f}|{per_task['RouteStick']['motion_tokens']:.2f}",
                     sizes[kn] if sizes[kn] is not None else f"{per_task['PatternLock']['motion_tokens']:.2f}|{per_task['RouteStick']['motion_tokens']:.2f}"])
write("mask_blocks.csv", ["query_block", "key_block", "allowed", "kind", "query_tokens", "key_tokens"], rows)

# 2 逐层 6×6 块注意力（pooled + 两任务）
rows = []
for scope, M in (("pooled", pooled["M6"]), *((t, per_task[t]["M6"]) for t in TASKS)):
    for l in range(18):
        for a, (qn, _) in enumerate(BLOCKS):
            for b, (kn, _) in enumerate(BLOCKS):
                rows.append([scope, l, qn, kn, f"{M[l, a, b]:.6f}"])
write("layer_block_attention.csv", ["scope", "layer", "reader_block", "source_block", "attention_pct"], rows)

# 3 热图 7 行 × 18 层
ROWS = [("frames ← motion", "A", (0, 1)), ("frames: cum. motion content", "A", None), ("action ← frames", "A", (5, 0)),
        ("language ← motion", "B", (4, 1)), ("language ← frames", "B", (4, 0)), ("action ← language", "B", (5, 4)),
        ("action ← motion", "-", (5, 1))]
rows = []
for scope, M, carry in (("pooled", pooled["M6"], pooled["carry"]), *((t, per_task[t]["M6"], per_task[t]["carry"]) for t in TASKS)):
    for name, route, idx in ROWS:
        for l in range(18):
            v = carry[l + 1] if idx is None else M[l, idx[0], idx[1]]
            rows.append([scope, name, route, l, f"{v:.6f}"])
write("pathway_heatmap.csv", ["scope", "edge", "route", "layer", "value_pct"], rows)

# 4 帧累积 motion 含量（第 l 层输入，l = 0..18）
rows = []
for scope, carry in (("pooled", pooled["carry"]), *((t, per_task[t]["carry"]) for t in TASKS)):
    for l in range(19):
        rows.append([scope, l, f"{carry[l]:.6f}"])
write("frame_motion_content.csv", ["scope", "layer_input", "motion_content_pct"], rows)

# 5 动作里来自 motion 的注意力质量与末跳分解
rows = []
for t in TASKS:
    for a, via in per_task[t]["route"].items():
        total = via.sum()
        rows.append([t, a, f"{total:.6f}", f"{via[[0, 1]].sum() / total:.6f}", f"{via[6] / total:.6f}", f"{via[[4, 5]].sum() / total:.6f}",
                     f"{via[[2, 3]].sum() / total:.6f}", per_task[t]["episodes"], f"{per_task[t]['motion_tokens']:.3f}"])
write("route_shares.csv", ["task", "alpha", "motion_content_in_action_pct", "via_frames_frac", "via_language_frac", "via_images_frac",
                           "via_motion_frac", "n_episodes", "mean_motion_tokens"], rows)

# 6 子目标对位矩阵（全部 A2 格）与 7 摘要
summary = json.load(open(REC / "summary.json", encoding="utf-8"))
succ = json.load(open(REC / "success-only.json", encoding="utf-8"))
decomp = json.load(open(REC / "decompose.json", encoding="utf-8"))
mat_rows, sum_rows = [], []
for task, models in summary["A2"].items():
    for model, readers in models.items():
        for reader, streams in readers.items():
            for stream, weights in streams.items():
                for weight, c in weights.items():
                    E, N = c["matrix"]["enrichment"], c["matrix"]["n_episodes_per_cell"]
                    for j, row in enumerate(E):
                        for jp, v in enumerate(row):
                            if v is not None:
                                mat_rows.append([task, model, reader, stream, weight, j + 1, jp + 1, f"{v:.6f}", N[j][jp]])
                    key = f"{task}/{model}/{reader}/{stream}"
                    sc = succ["cells"].get(key, {}).get("D") if weight == {"frame": "single", "motion": "overlap"}[stream] else None
                    dc = decomp["cells"].get(key, {}) if weight == {"frame": "single", "motion": "overlap"}[stream] else {}
                    sum_rows.append([task, model, reader, stream, weight, c["n_episodes"], f"{c['D']['mean']:.6f}", f"{c['D']['ci95'][0]:.6f}",
                                     f"{c['D']['ci95'][1]:.6f}", f"{c['slope']['mean']:.6f}", f"{c['slope']['ci95'][0]:.6f}", f"{c['slope']['ci95'][1]:.6f}",
                                     c["tier"], f"{sc['mean']:.6f}" if sc else "", f"{sc['ci95'][0]:.6f}" if sc else "", f"{sc['ci95'][1]:.6f}" if sc else "",
                                     sc["n"] if sc and "n" in sc else (succ["cells"][key].get("n_episodes", "") if sc else ""),
                                     *(f"{dc[k]['mean']:.6f}" if k in dc else "" for k in ("first_later", "diag_middle", "last_earlier"))])
write("subgoal_alignment_matrix.csv", ["task", "model", "reader", "stream", "weight", "executing_j", "demo_segment_jprime", "enrichment", "n_episodes"], mat_rows)
write("subgoal_alignment_summary.csv", ["task", "model", "reader", "stream", "weight", "n_episodes", "D_mean", "D_ci_low", "D_ci_high", "slope_mean",
                                        "slope_ci_low", "slope_ci_high", "tier", "success_only_D_mean", "success_only_D_ci_low", "success_only_D_ci_high",
                                        "success_only_n", "first_segment_mean", "true_alignment_mean", "last_segment_mean"], sum_rows)

# ----------------------------------------------------------------------------- 校验
ref = json.load(open(REC / "paper-figures" / "fig_pathway-layers.json", encoding="utf-8"))
hm_ref = np.array(ref["heatmap_pct"])
hm_csv = np.zeros_like(hm_ref)
with open(OUT / "pathway_heatmap.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["scope"] == "pooled":
            hm_csv[[n for n, *_ in ROWS].index(r["edge"]), int(r["layer"])] = float(r["value_pct"])
err_hm = float(np.abs(hm_csv - hm_ref).max())
err_m6 = float(np.abs(pooled["M6"] - np.array(ref["layer_matrix_pct"])).max())
mask_ok = [[not a for a in row] for row in ref["masked"]] == [[bool(int(r[2])) for r in rows_m] for rows_m in
                                                              [[x for x in csv.reader(open(OUT / "mask_blocks.csv", encoding="utf-8"))][1:][i * 6:(i + 1) * 6] for i in range(6)]]
cell = summary["A2"]["RouteStick"]["mot"]["text"]["motion"]["overlap"]["matrix"]["enrichment"]
csv_cell = {(int(r["executing_j"]), int(r["demo_segment_jprime"])): float(r["enrichment"]) for r in csv.DictReader(open(OUT / "subgoal_alignment_matrix.csv", encoding="utf-8"))
            if (r["task"], r["model"], r["reader"], r["stream"], r["weight"]) == ("RouteStick", "mot", "text", "motion", "overlap")}
err_a2 = max(abs(csv_cell[(j + 1, jp + 1)] - v) for j, row in enumerate(cell) for jp, v in enumerate(row) if v is not None)
ok = err_hm <= 5.1e-3 and err_m6 <= 5.1e-3 and mask_ok and err_a2 <= 1e-6   # 参考 JSON 存两位小数、CSV 存六位
for name, n in files:
    print(f"{name}: {n} 行")
print(f"PAPER_DATA_EXPORT={'PASS' if ok else 'FAIL'} files={len(files)} heatmap_err={err_hm:.2e} matrix_err={err_m6:.2e} mask_match={int(mask_ok)} "
      f"a2_err={err_a2:.2e} episodes={sum(per_task[t]['episodes'] for t in TASKS)} inferences={inferences}")
raise SystemExit(0 if ok else 1)
