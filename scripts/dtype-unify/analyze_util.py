#!/usr/bin/env python3
"""稳态步时与 GPU util 统计（AGENTS 16 口径）。

**为什么另写一个**：`scripts/bottleneck-bench-v2/analyze_gpu_util.py` 期望的文件名与
列格式（`gpu_util_dense.csv`，legacy 为 `epoch秒,卡号,util%,显存MiB`）与本轮 speed /
正确性 run 采集的 `nvidia-smi --query-gpu=timestamp,index,utilization.gpu,
utilization.memory,memory.used,power.draw --format=csv,noheader` 不同，直接喂会解析错。
本脚本按实际采集格式算，口径严格照 AGENTS 16：

- **禁止以中位数作标题结论**——必报 util **均值**、**0% 采样占比**、**慢步/非慢步分层
  均值**；中位数只作附列。
- 稳态窗默认丢弃前 `--warmup` 步（JIT 编译与 worker 起步）。
- 摘要步及其下一步默认剔除（`--digest-steps`）：一次完整 TrainState 摘要是数十秒级的
  纯停顿，正确性 run 不剔除就没法看步时。

用法：
    uv run scripts/dtype-unify/analyze_util.py <records_dir> --steps 1000 \
        [--warmup 50] [--digest-steps 0,100,...] [--util-csv <path>]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import statistics


def _load_steps(rec: pathlib.Path) -> list[tuple[int, float]]:
    rows = []
    with (rec / "metrics.jsonl").open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("step") is not None and r.get("wall_time"):
                    rows.append((int(r["step"]), float(r["wall_time"])))
    rows.sort()
    return rows


def _load_util(path: pathlib.Path) -> list[tuple[float, int, float]]:
    """解析 nvidia-smi csv：timestamp, index, util.gpu %, util.mem %, mem MiB, power W。"""
    out = []
    with path.open() as f:
        for line in f:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                ts = _dt.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                out.append((ts, int(parts[1]), float(parts[2].rstrip(" %"))))
            except (ValueError, IndexError):
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="稳态步时与 util 统计（AGENTS 16 口径）")
    ap.add_argument("records_dir", type=pathlib.Path)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--digest-steps", default="", help="摘要步（逗号分隔），该步及下一步剔除")
    ap.add_argument("--util-csv", type=pathlib.Path, default=None)
    ap.add_argument("--slow-factor", type=float, default=1.5, help="慢步阈值 = 该倍数 × 稳态中位")
    ap.add_argument("--report", type=pathlib.Path, default=None)
    args = ap.parse_args()

    rec = args.records_dir
    rows = _load_steps(rec)
    digest = {int(x) for x in args.digest_steps.split(",") if x.strip()}

    # 逐步步时 = 相邻 log 时刻之差；剔除 warmup 前、摘要步及其下一步
    deltas: list[tuple[int, float]] = []
    for (s0, t0), (s1, t1) in zip(rows, rows[1:], strict=False):
        if s1 != s0 + 1 or s1 < args.warmup:
            continue
        if s1 in digest or s1 - 1 in digest:
            continue
        deltas.append((s1, t1 - t0))
    vals = [d for _, d in deltas]
    if not vals:
        raise SystemExit("稳态窗内没有可用步时——检查 --warmup / --digest-steps")

    med = statistics.median(vals)
    slow_cut = args.slow_factor * med
    slow = [(s, d) for s, d in deltas if d > slow_cut]
    fast = [(s, d) for s, d in deltas if d <= slow_cut]
    srt = sorted(vals)
    p10 = srt[int(0.10 * (len(srt) - 1))]
    p90 = srt[int(0.90 * (len(srt) - 1))]

    out: dict = {
        "records_dir": str(rec),
        "steps": args.steps,
        "warmup": args.warmup,
        "n_steady": len(vals),
        "step_median_s": round(med, 4),
        "step_mean_s": round(statistics.fmean(vals), 4),
        "step_p10_s": round(p10, 4),
        "step_p90_s": round(p90, 4),
        "slow_steps": [s for s, _ in slow],
        "slow_mean_s": round(statistics.fmean([d for _, d in slow]), 4) if slow else None,
        "fast_mean_s": round(statistics.fmean([d for _, d in fast]), 4) if fast else None,
        "epoch_hours": round(395289 / 8 * statistics.fmean(vals) / 3600, 3),
    }

    util_csv = args.util_csv or (rec / "util-lms500.csv")
    if util_csv.exists():
        samples = _load_util(util_csv)
        # 稳态时间窗 = 第一个与最后一个稳态步的 log 时刻
        t_lo = next(t for s, t in rows if s >= args.warmup)
        t_hi = rows[-1][1]
        win = [(ts, gi, u) for ts, gi, u in samples if t_lo <= ts <= t_hi]
        # 摘要步停顿区间（该步 log 时刻的前后各一步）不计入 util——它是纯 device_get 停顿
        pause = []
        by_step = dict(rows)
        for s in sorted(digest):
            if s in by_step and (s + 1) in by_step:
                pause.append((by_step[s], by_step[s + 1]))
        def in_pause(ts: float) -> bool:
            return any(a <= ts <= b for a, b in pause)
        clean = [(ts, gi, u) for ts, gi, u in win if not in_pause(ts)]
        base = clean or win
        us = [u for _, _, u in base]
        out["util"] = {
            "csv": str(util_csv),
            "n_samples": len(base),
            "n_excluded_pause": len(win) - len(clean),
            "mean_pct": round(statistics.fmean(us), 2) if us else None,
            "median_pct": round(statistics.median(us), 2) if us else None,
            "zero_pct_share": round(100.0 * sum(1 for u in us if u == 0) / len(us), 2) if us else None,
            "per_gpu_mean": {
                str(g): round(statistics.fmean([u for _, gi, u in base if gi == g]), 2)
                for g in sorted({gi for _, gi, _ in base})
            },
        }
    else:
        out["util"] = None

    print(json.dumps(out, indent=2, ensure_ascii=False))
    u = out["util"]
    print(
        "UTIL_SUMMARY "
        f"step_mean={out['step_mean_s']}s step_median={out['step_median_s']}s "
        f"p10={out['step_p10_s']} p90={out['step_p90_s']} n={out['n_steady']} "
        f"slow={len(out['slow_steps'])} "
        f"util_mean={u['mean_pct'] if u else 'NA'}% "
        f"zero_share={u['zero_pct_share'] if u else 'NA'}%"
    )
    if args.report:
        args.report.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
