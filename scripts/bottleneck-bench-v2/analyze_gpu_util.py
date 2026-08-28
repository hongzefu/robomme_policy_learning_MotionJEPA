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

# S7.5 判据/字节帐依赖格式层（venv 内已装 mme_vla_suite 包）
from mme_vla_suite.datastore import framesamp_store as _fs
from mme_vla_suite.datastore import load_manifest

# D 节主判据表（必达档；--accept 时机器判定）
ACCEPT_THRESHOLDS = {"step_med_s": 5.00, "util_mean_pct": 90.0,
                     "zero_pct": 5.0, "slow_wall_pct": 5.0, "epoch_h": 8.6}


def per_step_read_bytes(record_dir: Path) -> tuple[float, str]:
    """S7.5：每步读盘从 env.json(backend/dataset_path) + history_config + 清单现场
    推导（替代硬编码 1.20 GB）。返回 (bytes_per_step, 口径说明)；推导不了时回退
    legacy 全量清单公式并注明。"""
    env = {}
    try:
        env = json.load(open(record_dir / "env.json"))
    except (OSError, json.JSONDecodeError):
        pass
    batch = int(env.get("batch_size") or env.get("argv_batch") or 64)
    backend = env.get("MMEVLA_DATA_BACKEND") or env.get("backend") or "legacy"
    manifest_path = env.get("manifest_path_resolved") or env.get("manifest_path") or \
        str(Path(__file__).resolve().parents[2] / "v1-store" / "episode_manifest.json")
    per_frame = _fs.IMAGE_ROW_BYTES if backend == "packed" else _fs.SOURCE_NPY_SIZE
    manifest = load_manifest(manifest_path)
    mean_frames = _fs.mean_sampled_frames(manifest, 32)
    per_sample = _fs.SOURCE_PKL_BYTES_FLOOR + mean_frames * per_frame
    return batch * per_sample, f"backend={backend} b{batch} mean_frames={mean_frames:.3f}"

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


def extra_summary(dirs: list[Path], steps: int, warmup: int) -> int:
    """S7.5：吃多个 record_dir 的附加判据汇总（D 节）：w4 与 w8 步时差 ≤3%。"""
    med_by_w = {}
    for d in dirs:
        env = json.load(open(d / "env.json"))
        w = int(env.get("num_workers") or env.get("workers"))
        _, deltas, _ = load_steps(d, steps, warmup)
        med_by_w[w] = statistics.median(sorted(deltas.values()))
        print(f"EXTRA dir={d.name} workers={w} 步时中位={med_by_w[w]:.3f}s")
    rc = 0
    if 4 in med_by_w and 8 in med_by_w:
        diff = abs(med_by_w[4] - med_by_w[8]) / min(med_by_w[4], med_by_w[8]) * 100
        ok = diff <= 3.0
        print(f"E2E_EXTRA w4w8_step_diff={diff:.2f}% 阈值=3% → "
              f"{'PASS' if ok else 'FAIL（CPU 侧仍未松绑）'}")
        rc = 0 if ok else 1
    else:
        print("E2E_EXTRA 缺 w4/w8 对，无法判附加判据")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir", type=Path, nargs="?", default=None)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--accept", action="store_true",
                    help="S8b 主判据表 5 项机器判定（E2E_ACCEPT 单行，FAIL 非零退出）；"
                         "默认关=现状行为（回归旧 record 不受影响）")
    ap.add_argument("--extra", type=Path, nargs="+", default=None,
                    help="附加判据汇总入口：传多个 record_dir（含 w4/w8）")
    args = ap.parse_args()
    if args.extra:
        return extra_summary(args.extra, args.steps, args.warmup)
    if args.record_dir is None:
        ap.error("缺 record_dir（单目录分析），或改用 --extra <dirs…> 汇总模式")
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
            # S7.5：每步读盘从 env.json+清单现场推导，替代硬编码 1.20 GB
            try:
                step_bytes, note = per_step_read_bytes(d)
                print(f"RESULT NFS实测={mbps:.0f} MB/s"
                      f"（公式口径 {step_bytes / 1e6 / step_med:.0f} MB/s = "
                      f"{step_bytes / 1e9:.3f}GB/步时，{note}）")
            except Exception as e:  # noqa: BLE001 —— 推导失败降级注明，不掩盖分析
                print(f"RESULT NFS实测={mbps:.0f} MB/s（公式口径推导失败: {e}）")

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

    # S7.5：主判据表 5 项机器判定（D 节；--accept 时启用，FAIL 非零退出）
    if args.accept:
        vals = {"step_med_s": step_med,
                "util_mean_pct": statistics.mean(utils),
                "zero_pct": pct(utils, lambda v: v == 0),
                "slow_wall_pct": 100 * slow_wall / total_wall,
                "epoch_h": EPOCH_STEPS * step_med / 3600}
        checks = {"step_med_s": vals["step_med_s"] <= ACCEPT_THRESHOLDS["step_med_s"],
                  "util_mean_pct": vals["util_mean_pct"] >= ACCEPT_THRESHOLDS["util_mean_pct"],
                  "zero_pct": vals["zero_pct"] <= ACCEPT_THRESHOLDS["zero_pct"],
                  "slow_wall_pct": vals["slow_wall_pct"] <= ACCEPT_THRESHOLDS["slow_wall_pct"],
                  "epoch_h": vals["epoch_h"] <= ACCEPT_THRESHOLDS["epoch_h"]}
        verdict = "PASS" if all(checks.values()) else "FAIL"
        detail = " ".join(
            f"{k}={vals[k]:.3f}{'✓' if checks[k] else '✗(阈' + str(ACCEPT_THRESHOLDS[k]) + ')'}"
            for k in vals)
        print(f"E2E_ACCEPT={verdict} {detail}")
        return 0 if verdict == "PASS" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
