#!/usr/bin/env python
"""GPU 利用率正确判读分析（bottleneck-bench v2；L0 重立量具版，v1-95util）。

判读口径（对应 AGENTS.md 第 16 条 + v1-95util L0）：
- 标题结论只用稳态窗口内的 **真实墙钟步时均值**、util 均值、0% 采样占比、
  active_util（非零采样条件均值）；**中位数一律只作附录，禁作标题结论**
  ——v1-e2e-b64 的教训：约 1/3 采样 <100% 被「中位 100%」掩盖，均值实际仅 69-70%。
- `step_mean` = 稳态窗口首末 metrics `wall_time` 差 ÷ 稳态步数，log1/log100 两档同式；
- util 三项永远来自 dense 500ms 通道（缺 dense 退 legacy 须显式告警并标 DEGRADED）；
- log100 生产档（env.json `log_interval` > 1）没有逐步步时：p10/p90、慢步分层、
  相位分组、slow_wall_pct 自动跳过并在输出写明——这些诊断只在 log1 对照档可算；
- 判据为 **E2E95_ACCEPT 五项**（--accept 时机器判定）；旧 E2E_ACCEPT/E2E_EXTRA 已废弃，
  旧性能族基线（v1-g0-speed-r2 等）一并作废，不再输出对照行。

输入（<record_dir>/ 下）：
  metrics.jsonl        训练日志标量（log1 逐步 / log100 每 100 步区间均值，必需）
  env.json             run 口径（log_interval / epoch_steps / batch_size / num_workers /
                       prefetch_factor / 线程数等；由 sbatch write_env 写入）
  gpu_util_dense.csv   dense 通道（nvidia-smi -lms 500，util 主判读依据）
  gpu_util.csv         legacy 通道（15s 采样，交叉核对用）
  nfs_read.csv         可选（时间戳,normal_read累计,server_read累计）

用法：uv run scripts/bottleneck-bench-v2/analyze_gpu_util.py <record_dir> --steps 600 [--accept]
"""

import argparse
import bisect
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

# S7.5 字节帐依赖格式层（venv 内已装 mme_vla_suite 包）
from mme_vla_suite.datastore import framesamp_store as _fs
from mme_vla_suite.datastore import load_manifest

# E2E95_ACCEPT 五项（v1-95util 判据表；--accept 时机器判定，FAIL 非零退出）
ACCEPT95_THRESHOLDS = {
    "util_mean_pct": 95.0,    # ≥，dense 稳态均值
    "zero_pct": 3.8,          # ≤，dense 稳态 0% 采样占比（工程目标 ≤2%）
    "active_util_pct": 98.0,  # ≥，非零采样条件均值
    "step_mean_s": 5.013,     # ≤，稳态真实墙钟 ÷ 稳态步数（建议直接做到 ≤5.00）
    "epoch_h": 8.6,           # ≤，epoch_steps × step_mean 换算展示（非独立自由度）
}
SLOW_THRESH = 8.0   # 慢步阈值（秒）：legacy 档 p90 16.6s 时代所定，packed 档 p90≈7.0s，
                    # 仅作 log1 诊断区回归护栏，不入判据
EPOCH_STEPS_FALLBACK = 6176   # 仅旧 record（env.json 无 epoch_steps）回退用，须告警


def per_step_read_bytes(record_dir: Path, env: dict) -> tuple[float, str]:
    """S7.5：每步读盘从 env.json(backend/dataset_path) + 清单现场推导。"""
    batch = int(env.get("batch_size") or 64)
    backend = env.get("MMEVLA_DATA_BACKEND") or env.get("backend") or "legacy"
    manifest_path = env.get("manifest_path_resolved") or env.get("manifest_path") or \
        str(Path(__file__).resolve().parents[2] / "v1-store" / "episode_manifest.json")
    per_frame = _fs.IMAGE_ROW_BYTES if backend == "packed" else _fs.SOURCE_NPY_SIZE
    manifest = load_manifest(manifest_path)
    mean_frames = _fs.mean_sampled_frames(manifest, 32)
    per_sample = _fs.SOURCE_PKL_BYTES_FLOOR + mean_frames * per_frame
    return batch * per_sample, f"backend={backend} b{batch} mean_frames={mean_frames:.3f}"


def load_env(record_dir: Path) -> dict:
    try:
        return json.load(open(record_dir / "env.json"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_metrics(record_dir: Path):
    rows = [json.loads(l) for l in open(record_dir / "metrics.jsonl") if '"loss"' in l]
    losses = [r["loss"]["dec"] for r in rows]
    bad = [v for v in losses if not math.isfinite(v)]
    assert not bad, f"非有限 loss: {bad[:5]}"
    return {r["step"]: r["wall_time"] for r in rows}, losses


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
    ap.add_argument("--accept", action="store_true",
                    help="E2E95_ACCEPT 五项机器判定（FAIL 非零退出）；默认关=只出报表")
    args = ap.parse_args()
    d = args.record_dir

    env = load_env(d)
    # log100 口径判别靠 env.json 的 log_interval 字段（sbatch write_env 写入），
    # 不靠猜 metrics 行距；旧 record 无该字段 → 按 log1 处理（旧数据全是 log1）
    log_interval = int(env.get("log_interval") or 1)
    if "log_interval" not in env:
        print("告警: env.json 无 log_interval 字段（旧 record），按 log1 处理")
    if "epoch_steps" in env:
        epoch_steps = int(env["epoch_steps"])
    else:
        epoch_steps = EPOCH_STEPS_FALLBACK
        print(f"告警: env.json 无 epoch_steps 字段（旧 record），回退 {epoch_steps}")
    batch = int(env.get("batch_size") or 64)
    if "batch_size" not in env:
        print("告警: env.json 无 batch_size 字段，回退 batch=64")
    workers = env.get("num_workers")

    t, losses = load_metrics(d)
    expected = [s for s in range(args.steps) if s % log_interval == 0]
    # 尾段口径（L0 已知损失）：train.py 末步非日志步时尾段指标不落盘，metrics 只到
    # 最后一个日志步；步集合必须与 log_interval 调度严格一致，防口径漂移
    assert sorted(t) == expected, \
        f"metrics 步集合与 log_interval={log_interval} 调度不符: " \
        f"n={len(t)} 期望 {len(expected)}（首末 {min(t, default=None)}/{max(t, default=None)}）"

    # 稳态窗口：首个 ≥warmup 的日志步 → 末个日志步；step_mean = 真实墙钟差 ÷ 步数，两档同式
    steady = [s for s in sorted(t) if s >= args.warmup]
    assert len(steady) >= 2, f"稳态窗口内日志步不足 2 个（warmup={args.warmup}）"
    s_lo, s_hi = steady[0], steady[-1]
    t_lo, t_hi = t[s_lo], t[s_hi]
    n_steady = s_hi - s_lo
    step_mean = (t_hi - t_lo) / n_steady
    epoch_h = epoch_steps * step_mean / 3600

    # 逐步统计只在 log1 可算（log100 的 metrics 是区间均值，天然没有逐步步时）
    per_step = log_interval == 1
    if per_step:
        deltas = {s: t[s] - t[s - 1] for s in range(s_lo + 1, args.steps)}
        dvals = sorted(deltas.values())
        n = len(dvals)
        p10, p90 = dvals[n // 10], dvals[n * 9 // 10]
        step_med = statistics.median(dvals)
        slow_iv = sorted((t[s - 1], t[s]) for s, dt in deltas.items() if dt > SLOW_THRESH)
        slow_starts = [a for a, _ in slow_iv]
        slow_wall = sum(b - a for a, b in slow_iv)
        total_wall = sum(dvals)

        def in_slow(ts):
            i = bisect.bisect_right(slow_starts, ts) - 1
            return i >= 0 and ts <= slow_iv[i][1]

    # util 三项永远来自 dense 500ms 采样，与日志粒度无关，两档一致
    dense_path = d / "gpu_util_dense.csv"
    legacy_path = d / "gpu_util.csv"
    degraded = False
    if dense_path.exists():
        primary, channel = load_dense(dense_path), "dense(500ms)"
    else:
        degraded = True
        print("告警: 缺 gpu_util_dense.csv，主判读退回 legacy 15s 通道——util 三项判读"
              "降级（DEGRADED），仅回归旧数据时预期")
        primary, channel = load_legacy(legacy_path), "legacy(15s) DEGRADED"
    win = [r for r in primary if t_lo <= r[0] <= t_hi]
    assert win, "稳态窗口内无 GPU 采样"

    utils = [u for _, _, u, _ in win]
    nz = [u for u in utils if u > 0]
    util_mean = statistics.mean(utils)
    zero_pct = pct(utils, lambda v: v == 0)
    active_util = statistics.mean(nz) if nz else float("nan")
    gpus = sorted({i for _, i, _, _ in win})

    print(f"===== 标题结论区（主通道 {channel}，log_interval={log_interval}，"
          f"稳态窗口=[step{s_lo}, step{s_hi}]，n采样={len(win)}）=====")
    print(f"RESULT 步时均值={step_mean:.3f}s（真实墙钟 {t_hi - t_lo:.1f}s ÷ {n_steady} 步）"
          f"  吞吐={batch / step_mean:.2f} 样本/s (batch={batch})")
    print(f"RESULT GPU util 均值={util_mean:.1f}%  0%采样占比={zero_pct:.1f}%  "
          f"active_util(非零条件均值)={active_util:.1f}%  <100%采样占比={pct(utils, lambda v: v < 100):.1f}%")
    for g in gpus:
        gu = [u for _, i, u, _ in win if i == g]
        gnz = [u for u in gu if u > 0]
        print(f"RESULT   卡{g}: util 均值={statistics.mean(gu):.1f}%  "
              f"0%占比={pct(gu, lambda v: v == 0):.1f}%  "
              f"active_util={statistics.mean(gnz) if gnz else float('nan'):.1f}%")
    print(f"RESULT 显存峰值={max(m for _, _, _, m in win)} MiB")
    print(f"RESULT epoch({epoch_steps}步)={epoch_steps * step_mean:.0f}s ≈ {epoch_h:.2f} 小时"
          f"（= epoch_steps × step_mean 换算展示）")
    print(f"RESULT loss n={len(losses)} min={min(losses):.4f} max={max(losses):.4f} 末值={losses[-1]:.4f}")

    nfs_path = d / "nfs_read.csv"
    if nfs_path.exists():
        nfs = [l.strip().split(",") for l in open(nfs_path) if l.strip()]
        nwin = [(float(a), int(c)) for a, b, c in nfs if t_lo <= float(a) <= t_hi]
        if len(nwin) >= 2:
            mbps = (nwin[-1][1] - nwin[0][1]) / (nwin[-1][0] - nwin[0][0]) / 1e6
            try:
                step_bytes, note = per_step_read_bytes(d, env)
                print(f"RESULT NFS实测={mbps:.0f} MB/s"
                      f"（公式口径 {step_bytes / 1e6 / step_mean:.0f} MB/s = "
                      f"{step_bytes / 1e9:.3f}GB/步时，{note}）")
            except Exception as e:  # noqa: BLE001 —— 推导失败降级注明，不掩盖分析
                print(f"RESULT NFS实测={mbps:.0f} MB/s（公式口径推导失败: {e}）")

    if per_step:
        slow_u = [u for ts, _, u, _ in win if in_slow(ts)]
        fast_u = [u for ts, _, u, _ in win if not in_slow(ts)]
        print("----- log1 诊断区（仅对照档可算；诊断列，不计入 E2E95_ACCEPT 判定）-----")
        print(f"诊断 步时 p10={p10:.3f}s p90={p90:.3f}s (n={n})")
        print(f"诊断 slow_wall_pct(>{SLOW_THRESH:.0f}s)={100 * slow_wall / total_wall:.1f}%"
              f"（参考线 ≤5%）  慢步时段 util 均值={statistics.mean(slow_u) if slow_u else float('nan'):.1f}%"
              f"  非慢步时段={statistics.mean(fast_u) if fast_u else float('nan'):.1f}%")
        if workers:
            w = int(workers)
            by_phase: dict[int, list[float]] = {}
            for s, dt in deltas.items():
                by_phase.setdefault(s % w, []).append(dt)
            parts = []
            for r in sorted(by_phase):
                vs = by_phase[r]
                n_slow = sum(1 for v in vs if v > SLOW_THRESH)
                parts.append(f"r{r}:均值{statistics.mean(vs):.2f}s/慢{n_slow}")
            print(f"诊断 相位分组 step%{w}（n/档≈{len(deltas) // w}）: " + "  ".join(parts))
    else:
        print(f"----- log1 诊断区：跳过（log_interval={log_interval} 生产档无逐步步时；"
              f"p10/p90、慢步分层、相位分组、slow_wall_pct 天然不可算，细节诊断回 log1 对照档查）-----")

    if not degraded and legacy_path.exists():
        lwin = [r for r in load_legacy(legacy_path) if t_lo <= r[0] <= t_hi]
        if lwin:
            lmean = statistics.mean([u for _, _, u, _ in lwin])
            print(f"----- legacy 对照区（15s 通道，n={len(lwin)}）-----")
            print(f"legacy util 均值={lmean:.1f}%  dense−legacy 均值差={util_mean - lmean:+.1f}pp"
                  f"（验证 15s 稀疏采样有无系统性偏差）")

    # 附录：中位数仅供交叉核对——禁止用作标题结论（中位数假象见文件头注释）
    print("----- 附录（勿作标题结论）-----")
    print(f"附录 GPU util 中位={statistics.median(utils):.0f}%")
    if per_step:
        print(f"附录 步时中位={step_med:.3f}s")

    # E2E95_ACCEPT 五项机器判定（v1-95util 判据表；--accept 时启用，FAIL 非零退出）
    if args.accept:
        vals = {"util_mean_pct": util_mean,
                "zero_pct": zero_pct,
                "active_util_pct": active_util,
                "step_mean_s": step_mean,
                "epoch_h": epoch_h}
        checks = {"util_mean_pct": vals["util_mean_pct"] >= ACCEPT95_THRESHOLDS["util_mean_pct"],
                  "zero_pct": vals["zero_pct"] <= ACCEPT95_THRESHOLDS["zero_pct"],
                  "active_util_pct": vals["active_util_pct"] >= ACCEPT95_THRESHOLDS["active_util_pct"],
                  "step_mean_s": vals["step_mean_s"] <= ACCEPT95_THRESHOLDS["step_mean_s"],
                  "epoch_h": vals["epoch_h"] <= ACCEPT95_THRESHOLDS["epoch_h"]}
        verdict = "PASS" if all(checks.values()) else "FAIL"
        detail = " ".join(
            f"{k}={vals[k]:.3f}{'✓' if checks[k] else '✗(阈' + str(ACCEPT95_THRESHOLDS[k]) + ')'}"
            for k in vals)
        tag = " DEGRADED" if degraded else ""
        print(f"E2E95_ACCEPT={verdict}{tag} {detail}")
        return 0 if verdict == "PASS" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
