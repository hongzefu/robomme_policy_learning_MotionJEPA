#!/usr/bin/env python3
"""从 episode 清单换算 16 任务的运动记忆时序指标（官方数据集口径，纯计算、秒级）。

输入是 ``scan_manifest.py build`` 产出的 ``episode_manifest.json``（逐集
``num_timesteps`` / ``exec_start_idx``），输出每任务的「中位集」快照与逐任务分布，
供「运动记忆时序数轴」按 train split 示范长度作图。

**不重新实现任何公式**——段长、每段网格数、可见窗口、帧路采样全部 import 现有实现：
``datastore.motion_store`` 的 ``build_index_entries`` / ``max_visible_count`` /
``segment_grid_starts``，以及 ``shared.sampling.even_sampling_indices``。

「中位集」取 ``statistics.median_high``（100 集时 = 按 num_timesteps 排序的第 51 小），
用该集**真实的 (num_timesteps, exec_start_idx) 配对**，而不是分别取中位数——后者会造出
数据集里不存在的组合。t 取该集最后一帧 ``num_timesteps - 1``。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import motion_store as ms          # noqa: E402
from mme_vla_suite.shared.sampling import even_sampling_indices  # noqa: E402

FRAME_BUDGET = 32          # = budget 512 // (token_per_image 16 × num_views 1)，与在线/训练同值
MOTION_BUDGET = 96         # motion.budget，零截断契约上限

# 官方四组分类（scripts/training/compute_results.py::TASK_SUITES）
TASK_SUITES = {
    "Counting":    ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"],
    "Persistent":  ["ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"],
    "Referential": ["PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"],
    "Behavior":    ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"],
}
# examples/robomme/utils.py::TASK_WITH_VIDEO_DEMO
TASK_WITH_VIDEO_DEMO = {
    "VideoUnmask", "VideoUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
    "VideoRepick", "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
}


def task_of(h5_file: str) -> str:
    return h5_file.replace("record_dataset_", "").replace(".h5", "")


def dist(xs: list[int]) -> dict:
    s = sorted(xs)
    return {"min": s[0], "p25": s[len(s) // 4], "median": statistics.median_high(s),
            "p75": s[3 * len(s) // 4], "max": s[-1], "mean": round(sum(s) / len(s), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    entries = ms.build_index_entries(manifest)      # 段基址与每段 num_grid 全部由它算

    by_task: dict[str, list] = {}
    for e in entries:
        by_task.setdefault(task_of(e.h5_file), []).append(e)

    tasks_out = {}
    for task, eps in sorted(by_task.items()):
        # 中位集：按 num_timesteps 排序取 median_high 那一集，保留真实的 (nt, es) 配对
        nts = [e.num_timesteps for e in eps]
        med_nt = statistics.median_high(nts)
        med = sorted([e for e in eps if e.num_timesteps == med_nt], key=lambda e: e.raw_ep_idx)[0]

        t = med.num_timesteps - 1
        n_demo = ms.seg_num_grid(med.demo.seg_len)
        n_exec_vis = len(ms.visible_motion_rows(med, t)[0]) - n_demo
        total_vis = ms.max_visible_count(med)
        fidx = even_sampling_indices(t, FRAME_BUDGET)
        gaps = [b - a for a, b in zip(fidx, fidx[1:])]

        tasks_out[task] = {
            "suite": next(k for k, v in TASK_SUITES.items() if task in v),
            "has_demo": task in TASK_WITH_VIDEO_DEMO,
            "episodes": len(eps),
            "median_episode": {
                "raw_ep_idx": med.raw_ep_idx,
                "num_timesteps": med.num_timesteps,
                "exec_start_idx": med.exec_start_idx,
                "t": t,
                "demo_seg_len": med.demo.seg_len,
                "exec_seg_len": med.exec.seg_len,
                "demo_windows": n_demo,
                "exec_windows": n_exec_vis,
                "motion_tokens": total_vis,
                "budget_pct": round(total_vis / MOTION_BUDGET * 100, 1),
                "frame_samples": len(fidx),
                "delta_mean": round(t / (FRAME_BUDGET - 1), 2) if t >= FRAME_BUDGET else None,
                "delta_min": min(gaps) if gaps else None,
                "delta_max": max(gaps) if gaps else None,
                "frames_in_demo": sum(1 for f in fidx if f < med.exec_start_idx),
                "demo_grid_starts": ms.segment_grid_starts(med.demo.seg_len)[:4],
            },
            "num_timesteps": dist(nts),
            "exec_start_idx": dist([e.exec_start_idx for e in eps]),
            "motion_tokens": dist([ms.max_visible_count(e) for e in eps]),
            "total_timesteps": sum(nts),
            "total_exec_samples": sum(e.num_timesteps - e.exec_start_idx for e in eps),
        }

    totals = ms.index_totals(entries)
    out = {
        "source_manifest": args.manifest,
        "split": "train",
        "frame_budget": FRAME_BUDGET,
        "motion_budget": MOTION_BUDGET,
        "grid": {"stride": ms.GRID_STRIDE, "window_frames": ms.WINDOW_FRAMES,
                 "origin": ms.GRID_ORIGIN, "direction": ms.WINDOW_DIRECTION},
        "totals": {
            "tasks": len(tasks_out),
            "episodes": len(entries),
            "timesteps": sum(e.num_timesteps for e in entries),
            "exec_samples": sum(e.num_timesteps - e.exec_start_idx for e in entries),
            **totals,
        },
        "suites": TASK_SUITES,
        "tasks": tasks_out,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    print(f"任务数 {len(tasks_out)}  episode {out['totals']['episodes']}  "
          f"timestep {out['totals']['timesteps']}  exec 样本 {out['totals']['exec_samples']}  "
          f"窗口行 {totals['rows']}（demo {totals['demo_rows']} + exec {totals['exec_rows']}）")
    print(f"{'任务':<18}{'组':<12}{'demo':>5}{'中位集 nt':>10}{'es':>6}{'token':>7}"
          f"{'Δ':>7}{'32帧落demo':>11}{'单集最大token':>14}")
    for task, d in sorted(tasks_out.items(), key=lambda kv: (kv[1]["suite"], kv[0])):
        m = d["median_episode"]
        print(f"{task:<18}{d['suite']:<12}{'有' if d['has_demo'] else '无':>5}"
              f"{m['num_timesteps']:>10}{m['exec_start_idx']:>6}{m['motion_tokens']:>7}"
              f"{m['delta_mean'] or 0:>7.1f}{m['frames_in_demo']:>11}{d['motion_tokens']['max']:>14}")
    print(f"清单已写 {args.out}")


if __name__ == "__main__":
    main()
