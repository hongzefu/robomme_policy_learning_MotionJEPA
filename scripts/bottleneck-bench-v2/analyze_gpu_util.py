#!/usr/bin/env python
"""GPU 利用率正确判读分析（bottleneck-bench v2）。

判读口径（对应 AGENTS.md 第 16 条）：标题结论只允许用稳态窗口内的
util 均值、0% 采样占比、慢步/非慢步分层均值；**禁止用中位数做标题结论**——
v1-e2e-b64 的教训：约 1/3 采样 <100%（大量为 0%）被「中位 100%」掩盖，
均值实际只有 69-70%。中位数只出现在文末附录区，仅供交叉核对。

输入（<record_dir>/ 下）：
  metrics.jsonl        逐步 wall_time/loss（必需）
  gpu_util_dense.csv   dense 通道（nvidia-smi -lms 500，主判读依据；
                       格式：时间戳字符串,卡号,util%,显存MiB，时间戳为节点本地时区毫秒精度）
  gpu_util.csv         legacy 通道（v1 同款 15s 采样，epoch秒,卡号,util%,显存MiB；
                       与 v1-e2e-b64 基线同口径对照用）
  nfs_read.csv         可选（时间戳,normal_read累计,server_read累计）

缺 dense 文件时自动退回用 legacy 通道做主判读并打印告警（供回归 v1 旧数据）。

用法：uv run scripts/bottleneck-bench-v2/analyze_gpu_util.py <record_dir> --steps 600
"""

import argparse
import bisect
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

# v1-e2e-b64 基线（8C/4w/64G，300 步，15s 采样），供对照行
V1_BASELINE = {
    "run": "v1-e2e-b64",
    "step_median": 6.933,
    "util_mean": "69-70%",
    "slow_wall_share": 0.33,
    "slow_util_mean": 54.0,
    "fast_util_mean": 79.0,
}
EPOCH_STEPS = 6176  # epoch 实测口径换算用步数
SLOW_THRESH = 8.0   # 慢步阈值（秒），沿用 v1 分析口径


def load_steps(record_dir: Path, steps: int, warmup: int):
    rows = [json.loads(l) for l in open(record_dir / "metrics.jsonl") if '"loss"' in l]
    assert len(rows) == steps, f"metrics 行数 {len(rows)} != {steps}"
    losses = [r["loss"]["dec"] for r in rows]
    bad = [v for v in losses if not math.isfinite(v)]
    assert not bad, f"非有限 loss: {bad[:5]}"
    t = {r["step"]: r["wall_time"] for r in rows}
    deltas = {s: t[s] - t[s - 1] for s in range(warmup + 1, steps)}
    return t, deltas, losses


def load_dense(path: Path):
    """dense 通道：时间戳为节点本地时区字符串（与 metrics 的 time.time() 同钟同节点，
    按本机本地时区换算 epoch；集群与本机同处 America/Detroit）。"""
    out = []
    for line in open(path):
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != 4:
            continue
        try:
            ts = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            out.append((ts, int(parts[1]), int(parts[2]), int(parts[3])))
        except ValueError:
            continue
    return out


def load_legacy(path: Path):
    out = []
    for line in open(path):
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) != 4:
            continue
        try:
            out.append((float(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
        except ValueError:
            continue
    return out


def pct(vals, cond):
    return 100.0 * sum(1 for v in vals if cond(v)) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir", type=Path)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()
    d = args.record_dir

    t, deltas, losses = load_steps(d, args.steps, args.warmup)
    t_lo, t_hi = t[args.warmup], t[args.steps - 1]
    dvals = sorted(deltas.values())
    n = len(dvals)
    step_med = statistics.median(dvals)
    p10, p90 = dvals[n // 10], dvals[n * 9 // 10]

    # 慢步区间（(t[s-1], t[s]]，delta > 阈值），采样按落入区间与否分层
    slow_iv = sorted((t[s - 1], t[s]) for s, dt in deltas.items() if dt > SLOW_THRESH)
    slow_starts = [a for a, _ in slow_iv]
    slow_wall = sum(b - a for a, b in slow_iv)
    total_wall = sum(dvals)

    def in_slow(ts):
        i = bisect.bisect_right(slow_starts, ts) - 1
        return i >= 0 and ts <= slow_iv[i][1]

    dense_path = d / "gpu_util_dense.csv"
    legacy_path = d / "gpu_util.csv"
    if dense_path.exists():
        primary, channel = load_dense(dense_path), "dense(500ms)"
    else:
        print("告警: 缺 gpu_util_dense.csv，主判读退回 legacy 15s 通道（仅回归旧数据时预期）")
        primary, channel = load_legacy(legacy_path), "legacy(15s)"
    win = [r for r in primary if t_lo <= r[0] <= t_hi]
    assert win, "稳态窗口内无 GPU 采样"

    utils = [u for _, _, u, _ in win]
    slow_u = [u for ts, _, u, _ in win if in_slow(ts)]
    fast_u = [u for ts, _, u, _ in win if not in_slow(ts)]
    gpus = sorted({i for _, i, _, _ in win})

    print(f"===== 标题结论区（主通道 {channel}，稳态窗口=[step{args.warmup}, step{args.steps - 1}]，n采样={len(win)}）=====")
    print(f"RESULT 步时中位={step_med:.3f}s (n={n}, p10={p10:.3f}, p90={p90:.3f})  吞吐={64 / step_med:.2f} 样本/s")
    print(f"RESULT GPU util 均值={statistics.mean(utils):.1f}%  0%采样占比={pct(utils, lambda v: v == 0):.1f}%  <100%采样占比={pct(utils, lambda v: v < 100):.1f}%")
    for g in gpus:
        gu = [u for _, i, u, _ in win if i == g]
        print(f"RESULT   卡{g}: util 均值={statistics.mean(gu):.1f}%  0%占比={pct(gu, lambda v: v == 0):.1f}%")
    print(f"RESULT 慢步(>{SLOW_THRESH:.0f}s)墙钟占比={100 * slow_wall / total_wall:.1f}%  慢步时段 util 均值={statistics.mean(slow_u) if slow_u else float('nan'):.1f}%  非慢步时段={statistics.mean(fast_u) if fast_u else float('nan'):.1f}%")
    print(f"RESULT 显存峰值={max(m for _, _, _, m in win)} MiB")
    print(f"RESULT epoch({EPOCH_STEPS}步)={EPOCH_STEPS * step_med:.0f}s ≈ {EPOCH_STEPS * step_med / 3600:.2f} 小时")
    print(f"RESULT loss n={len(losses)} min={min(losses):.4f} max={max(losses):.4f} 末值={losses[-1]:.4f}")

    nfs_path = d / "nfs_read.csv"
    if nfs_path.exists():
        nfs = [l.strip().split(",") for l in open(nfs_path) if l.strip()]
        nwin = [(float(a), int(c)) for a, b, c in nfs if t_lo <= float(a) <= t_hi]
        if len(nwin) >= 2:
            mbps = (nwin[-1][1] - nwin[0][1]) / (nwin[-1][0] - nwin[0][0]) / 1e6
            print(f"RESULT NFS实测={mbps:.0f} MB/s（公式口径 {1.20e3 / step_med:.0f} MB/s = 1.20GB/步时）")

    print(f"----- 对照 v1 基线 {V1_BASELINE['run']}（8C/4w/64G）-----")
    print(f"对照 步时中位 {V1_BASELINE['step_median']:.3f}s → {step_med:.3f}s；util 均值 {V1_BASELINE['util_mean']} → {statistics.mean(utils):.1f}%；"
          f"慢步墙钟 {V1_BASELINE['slow_wall_share'] * 100:.0f}% → {100 * slow_wall / total_wall:.1f}%；"
          f"慢步/非慢步 util {V1_BASELINE['slow_util_mean']:.0f}%/{V1_BASELINE['fast_util_mean']:.0f}% → "
          f"{statistics.mean(slow_u) if slow_u else float('nan'):.0f}%/{statistics.mean(fast_u) if fast_u else float('nan'):.0f}%")

    if dense_path.exists() and legacy_path.exists():
        lwin = [r for r in load_legacy(legacy_path) if t_lo <= r[0] <= t_hi]
        if lwin:
            lmean = statistics.mean([u for _, _, u, _ in lwin])
            print(f"----- legacy 对照区（15s 通道，与 v1 采样口径一致，n={len(lwin)}）-----")
            print(f"legacy util 均值={lmean:.1f}%  dense−legacy 均值差={statistics.mean(utils) - lmean:+.1f}pp（验证 v1 的 15s 采样有无系统性偏差）")

    # 附录：中位数仅供交叉核对——禁止用作标题结论（中位数假象见文件头注释）
    print("----- 附录（勿作标题结论）-----")
    print(f"附录 GPU util 中位={statistics.median(utils):.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
