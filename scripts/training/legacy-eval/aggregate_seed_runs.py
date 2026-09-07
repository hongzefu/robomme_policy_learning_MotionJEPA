#!/usr/bin/env python
"""跨 seed / 跨任务批次的评估结果汇总（多 seed 对照专用）。

背景：merge_eval_shards.py 只处理「单 run、单 seed」，其 --seed 仅用于拼路径，
输出里没有 seed 维度；而本轮要把同一组权重在多个 policy 采样 seed 下的结果并排比。
另外为绕开「单个 eval.py 进程建满 27 个仿真环境即崩」的上限，本轮按任务分批起跑，
每批一个独立 RUN_NAME（否则第二批发现 log.json 已存在会空转退出），于是同一个
(组, seed) 的结果散在 4 个批次 × N 个 worker 目录里，也需要在这里合并。

读取路径（与 eval_shard.sh 的落盘口径一致）：
  <eval-root>/<run前缀>-s<seed>-<Task>-w<k>/ckpt<ckpt_id>/seed<seed>/progress.json
progress.json 是 {task: {ep: true|false|"error"}}，由 eval.py 每评完一集写一次；
success/fail/timeout 三分只能从 videos/<task>_ep<N>_<flag>_..mp4 的文件名反解
（沿用 merge_eval_shards.py 的 _VIDEO_RE）；每集的环境 seed 与难度取自 benchmark 元数据。

用法：
  uv run --no-sync python scripts/training/prod/aggregate_seed_runs.py \
    --group official:official-ctx:79999:4 --group motion:awsprod40k-motion:39999:2 \
    --seeds 42,7,2024 --tasks ButtonUnmask,VideoUnmask,ButtonUnmaskSwap,VideoUnmaskSwap \
    --metadata-dir <robomme env_metadata 目录> --out-dir <留档 records 目录>

判定行：SEED_AGG=DONE|INCOMPLETE groups=<g> seeds=<s> episodes=<n>/<N> errors=<e>
产物：<out-dir>/{summary.txt, per_episode_by_seed.json}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics as st
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_VIDEO_RE = re.compile(r"^(?P<task>[A-Za-z]+)_ep(?P<ep>\d+)_(?P<flag>[a-z]+)_")


def load_test_meta(meta_dir: pathlib.Path, task: str) -> dict[int, dict]:
    """读 benchmark 元数据里每集固定的环境 seed 与难度（换 policy seed 不会改变它们）。"""
    p = meta_dir / "test" / f"record_dataset_{task}_metadata.json"
    if not p.exists():
        return {}
    payload = json.loads(p.read_text())
    return {int(r["episode"]): {"seed": r.get("seed"), "difficulty": r.get("difficulty")}
            for r in payload.get("records", [])}


def collect_one(eval_root: pathlib.Path, prefix: str, ckpt_id: str, shards: int,
                seed: str, task: str) -> tuple[dict[int, object], dict[int, str], dict[int, str]]:
    """收一个 (组, seed, 任务) 的逐集结果。返回 (结果, flag, 来源 worker)。"""
    results: dict[int, object] = {}
    flags: dict[int, str] = {}
    origin: dict[int, str] = {}
    for k in range(shards):
        shard_dir = eval_root / f"{prefix}-s{seed}-{task}-w{k}" / f"ckpt{ckpt_id}" / f"seed{seed}"
        prog = shard_dir / "progress.json"
        if prog.exists():
            try:
                payload = json.loads(prog.read_text())
            except json.JSONDecodeError:      # 评估中途读到半写状态，跳过本片本轮
                payload = {}
            for ep, val in payload.get(task, {}).items():
                results[int(ep)] = val
                origin[int(ep)] = f"w{k}"
        videos = shard_dir / "videos"
        if videos.is_dir():
            for mp4 in videos.glob("*.mp4"):
                m = _VIDEO_RE.match(mp4.name)
                if m and m.group("task") == task:
                    flags[int(m.group("ep"))] = m.group("flag")
    return results, flags, origin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=str(_REPO_ROOT / "v1-store/evaluation"))
    ap.add_argument("--group", action="append", required=True,
                    help="<组名>:<run前缀>:<ckpt_id>:<shards>，可重复")
    ap.add_argument("--seeds", required=True, help="逗号分隔的 policy 采样 seed")
    ap.add_argument("--tasks", default="ButtonUnmask,VideoUnmask,ButtonUnmaskSwap,VideoUnmaskSwap")
    ap.add_argument("--episodes-per-task", type=int, default=50)
    ap.add_argument("--metadata-dir",
                    default=str(_REPO_ROOT / "third_party/robomme_benchmark/src/robomme/env_metadata"))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    eval_root = pathlib.Path(args.eval_root)
    meta_dir = pathlib.Path(args.metadata_dir)
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    N = args.episodes_per_task

    groups = []
    for spec in args.group:
        parts = spec.split(":")
        if len(parts) != 4:
            print(f"错误: --group 格式应为 <组名>:<run前缀>:<ckpt_id>:<shards>，实为 {spec!r}", file=sys.stderr)
            return 2
        groups.append({"name": parts[0], "prefix": parts[1], "ckpt": parts[2], "shards": int(parts[3])})

    meta = {t: load_test_meta(meta_dir, t) for t in tasks}
    if not any(meta.values()):
        print(f"警告: --metadata-dir 没读到任何元数据（{meta_dir}），逐集 env seed / 难度将为空", file=sys.stderr)

    # cells[(组, seed, 任务)] = {"n","success","error","rate","missing"}
    cells: dict[tuple[str, str, str], dict] = {}
    # per_ep[(组, 任务, ep)] = {"env_seed","difficulty","results":{seed:{...}}}
    per_ep: dict[tuple[str, str, int], dict] = {}
    n_done = n_err = 0

    for g in groups:
        for seed in seeds:
            for task in tasks:
                res, flags, origin = collect_one(eval_root, g["prefix"], g["ckpt"], g["shards"], seed, task)
                errs = sorted(e for e, v in res.items() if v == "error")
                succ = sum(1 for v in res.values() if v is True)
                n = len(res)
                missing = [e for e in range(N) if e not in res]
                cells[(g["name"], seed, task)] = {
                    "n": n, "success": succ, "error": len(errs),
                    "rate": (succ / n) if n else None, "missing": missing,
                }
                n_done += n
                n_err += len(errs)
                for ep, val in res.items():
                    key = (g["name"], task, ep)
                    slot = per_ep.setdefault(key, {
                        "group": g["name"], "task": task, "episode": ep,
                        "env_seed": meta[task].get(ep, {}).get("seed"),
                        "difficulty": meta[task].get(ep, {}).get("difficulty"),
                        "results": {},
                    })
                    slot["results"][seed] = {
                        "success": (val is True), "flag": flags.get(ep), "shard": origin.get(ep),
                        "error": (val == "error"),
                    }

    N_total = len(groups) * len(seeds) * len(tasks) * N
    status = "DONE" if (n_done == N_total and n_err == 0) else "INCOMPLETE"
    head = (f"SEED_AGG={status} groups={len(groups)} seeds={len(seeds)} tasks={len(tasks)} "
            f"episodes={n_done}/{N_total} errors={n_err}")

    lines = [head, ""]
    for g in groups:
        lines.append(f"## {g['name']}（{g['prefix']}-s<seed>-<Task>，ckpt{g['ckpt']}，{g['shards']} worker/批）")
        lines.append("")
        lines.append("| 任务 | " + " | ".join(f"seed {s}" for s in seeds) + " | 均值 | 标准差 |")
        lines.append("|---|" + "---|" * (len(seeds) + 2))
        task_means = []
        for task in tasks:
            cs = [cells[(g["name"], seed, task)] for seed in seeds]
            cols = []
            rates = []
            for c in cs:
                if c["n"] == 0:
                    cols.append("—")
                else:
                    cols.append(f"{c['success']}/{c['n']} ({c['rate']:.1%})")
                    rates.append(c["rate"])
            mean_s = f"{st.mean(rates):.1%}" if rates else "—"
            sd_s = f"{st.stdev(rates):.1%}" if len(rates) >= 2 else "NA"
            lines.append(f"| {task} | " + " | ".join(cols) + f" | {mean_s} | {sd_s} |")
            task_means.append(rates)
        # 四任务均值（每个 seed 各算一次，再跨 seed 统计）
        per_seed_mean = []
        cols = []
        for i, seed in enumerate(seeds):
            rs = [cells[(g["name"], seed, task)]["rate"] for task in tasks
                  if cells[(g["name"], seed, task)]["rate"] is not None]
            if len(rs) == len(tasks):
                m = st.mean(rs)
                per_seed_mean.append(m)
                cols.append(f"{m:.1%}")
            else:
                cols.append("—")
        mean_s = f"{st.mean(per_seed_mean):.1%}" if per_seed_mean else "—"
        sd_s = f"{st.stdev(per_seed_mean):.1%}" if len(per_seed_mean) >= 2 else "NA"
        lines.append(f"| **四任务均值** | " + " | ".join(cols) + f" | **{mean_s}** | **{sd_s}** |")
        lines.append("")
        holes = [(seed, task, cells[(g["name"], seed, task)]["missing"])
                 for seed in seeds for task in tasks if cells[(g["name"], seed, task)]["missing"]]
        if holes:
            lines.append("缺口（seed 任务 缺失集号）：")
            for seed, task, ms in holes:
                shown = ",".join(str(m) for m in ms[:12]) + ("…" if len(ms) > 12 else "")
                lines.append(f"  seed{seed} {task} 缺 {len(ms)} 集: {shown}")
            lines.append("")

    # 跨 seed 稳定性：三次都成 / 都败 / 有翻转
    lines.append("## 逐集跨 seed 稳定性（仅统计该组该集在所有 seed 上都有结果的）")
    lines.append("")
    lines.append("| 组 | 全成 | 全败 | 翻转 | 覆盖集数 |")
    lines.append("|---|---|---|---|---|")
    for g in groups:
        allw = alll = flip = cov = 0
        for (gname, task, ep), slot in per_ep.items():
            if gname != g["name"]:
                continue
            got = [slot["results"][s]["success"] for s in seeds if s in slot["results"]]
            if len(got) != len(seeds):
                continue
            cov += 1
            if all(got):
                allw += 1
            elif not any(got):
                alll += 1
            else:
                flip += 1
        lines.append(f"| {g['name']} | {allw} | {alll} | {flip} | {cov} |")
    lines.append("")

    lines.append("逐集明细（组 任务 ep env_seed 难度 " + " ".join(f"s{s}" for s in seeds) + "）：")
    for key in sorted(per_ep, key=lambda k: (k[0], k[1], k[2])):
        slot = per_ep[key]
        cols = []
        for s in seeds:
            r = slot["results"].get(s)
            cols.append("-" if r is None else ("E" if r["error"] else ("1" if r["success"] else "0")))
        lines.append(f"  {slot['group']} {slot['task']} {slot['episode']:2d} {slot['env_seed']} "
                     f"{slot['difficulty']} " + " ".join(cols))

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "per_episode_by_seed.json").write_text(
        json.dumps({"line": head, "seeds": seeds, "tasks": tasks,
                    "groups": groups,
                    "cells": {f"{gn}|{sd}|{tk}": v for (gn, sd, tk), v in cells.items()},
                    "episodes": [per_ep[k] for k in sorted(per_ep, key=lambda k: (k[0], k[1], k[2]))]},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(head)
    print(f"写出 {out/'summary.txt'} 与 {out/'per_episode_by_seed.json'}")
    return 0 if status == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
