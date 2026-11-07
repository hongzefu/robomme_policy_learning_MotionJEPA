"""汇总 pathway_flow.py 的逐局结果，按任务列三跳读数。每局等权，局级重采样 2000 次。帧按 token 数加权。

用法：pathway_summary.py <产物根> <模型> <pathway 目录> <输出 json>
"""
import glob
import json
import sys
from pathlib import Path
import numpy as np
np.set_printoptions(precision=1, suppress=True, linewidth=220)
root, model, flow, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
GROUPS = json.load(open(flow / "index.json", encoding="utf-8"))["groups"]
G = {n: i for i, n in enumerate(GROUPS)}
FRAME, MOTION, IMAGE, TEXT, ACTION = [0, 1], [2, 3], [4, 5], G["text"], G["action"]
task_of = {}
for path in glob.glob(str(root / model / "*" / "run" / "accepted" / "*.terminal.json")):
    record = json.load(open(path, encoding="utf-8"))
    task_of[record["episode_nonce"]] = (record["episode_key"].split(":")[0], record["outcome"] == "success")


def boot(values, repeats=2000, seed=42):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], "n": len(values)}


def per_episode(z):
    direct, counts = z["direct"].mean(1) * 100, z["counts"]                  # (层, q, k)
    w = counts[FRAME] / counts[FRAME].sum()
    carry = z["rollout_0.5"] * 100
    frame_carry = np.array([(w[:, None] * carry[l][FRAME][:, MOTION]).sum() for l in range(19)])
    frame_read = np.array([(w[:, None] * direct[l][FRAME][:, MOTION]).sum() for l in range(18)])
    text_frame, text_motion = direct[:, TEXT][:, FRAME].sum(1), direct[:, TEXT][:, MOTION].sum(1)
    item = {"frame_read_motion": frame_read, "frame_carry_motion": frame_carry, "text_read_frame": text_frame, "text_read_motion": text_motion,
            "text_via_frame_motion": text_frame / 100 * frame_carry[:18], "action_read_text": direct[:, ACTION, TEXT],
            "action_read_frame": direct[:, ACTION][:, FRAME].sum(1), "action_read_motion": direct[:, ACTION][:, MOTION].sum(1),
            "action_read_image": direct[:, ACTION][:, IMAGE].sum(1), "motion_tokens": counts[MOTION].sum()}
    for alpha in ("0.2", "0.5", "0.8"):
        via = z[f"route_{alpha}"].sum(0) * 100
        m = via[:, MOTION].sum(1)
        item[f"route_{alpha}"] = {"total": m.sum(), "frame": m[FRAME].sum(), "motion": m[MOTION].sum(), "text": m[TEXT], "image": m[IMAGE].sum()}
    return item


episodes = {}
for path in sorted(flow.glob("*.npz")):
    task, success = task_of[path.stem]
    with np.load(path) as z:
        episodes.setdefault(task, []).append(per_episode(z))
result = {}
for task, items in sorted(episodes.items()):
    mean = {k: np.mean([i[k] for i in items], 0) for k in items[0] if not k.startswith("route_")}
    print(f"== {task} {model} 局={len(items)} motion token 约 {mean['motion_tokens']:.1f}")
    for key, label in (("frame_read_motion", "帧读 motion%"), ("frame_carry_motion", "帧累积 motion%（第 l 层输入）"), ("text_read_frame", "文字读帧%"), ("text_read_motion", "文字读 motion%"),
                       ("text_via_frame_motion", "文字经帧拿 motion%"), ("action_read_text", "动作读文字%"), ("action_read_frame", "动作读帧%"), ("action_read_motion", "动作读 motion%")):
        print(f"{label}\n  {mean[key]}")
    cell = {"episodes": len(items), "by_layer": {k: np.asarray(v).tolist() for k, v in mean.items()}, "route": {}}
    for alpha in ("0.2", "0.5", "0.8"):
        total = np.array([i[f"route_{alpha}"]["total"] for i in items])
        row = {"total": boot(total)}
        for via in ("frame", "text", "image", "motion"):
            values = np.array([i[f"route_{alpha}"][via] for i in items])
            row[via] = float(values.sum() / total.sum() * 100)
            row[via + "_abs"] = boot(values)
        cell["route"][alpha] = row
        print(f"alpha={alpha} 动作里来自 motion {row['total']['mean']:.2f}% [{row['total']['ci95'][0]:.2f},{row['total']['ci95'][1]:.2f}]：经帧 {row['frame']:.0f}% 经文字 {row['text']:.0f}% 经图像 {row['image']:.0f}% 经 motion {row['motion']:.0f}%")
    result[task] = cell
out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("PATHWAY_SUMMARY=DONE", flush=True)
