#!/usr/bin/env python3
"""dense GPU util CSV 降采样归档器（v1-prod-trend-10h）。

**为什么需要**：`gl_e2e_fix.sbatch` 的 dense 通道是 `nvidia-smi -lms 500` 流式采样，
4 卡 × 2 Hz ≈ 1.2 MB/h。600 步的 bench 只有约 1 MB，现有留档一律**逐字节原样进 git**
（`docs/training-doc/v1-framesamp-e2e/records/README.md` 明写「逐字节拷贝」，实测字节数
完全相同）。但一次 10 小时的正式训练会产出约 12 MB / 29 万行，80k 步的完整训练更是 5–8 GB
——原样进 git 不可持续（AGENTS 14 要求大产物留在不进 git 的 `v1-store/`）。

**处置**（用户 2026-08-28 拍板「dense 采样文件降采样归档」）：全分辨率原件留在
`v1-store/`，只把降采样版归档进 `docs/training-doc/<run_name>/records/`；
`result.md` 写明原件绝对路径与降采样倍率。**判读永远跑全分辨率原件**——降采样只服务归档，
不参与任何 util 结论（AGENTS 16 要求采样间隔显著小于步时，5 秒间隔对 4.7 秒步时不合格）。

**按卡分组抽稀**：dense CSV 每个采样轮次会为每张卡各写一行（时间戳相差微秒级）。
若按「每 N 行取一行」naive 抽稀，4 卡会被切得不均（N 与卡数不互质时甚至只剩某一张卡）。
本工具对**每张卡各自计数**，保证各卡采样点数一致、时间轴对齐。

用法：uv run scripts/train-prod/downsample_util_csv.py <src.csv> <dst.csv> [--every 10]
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def downsample(src: pathlib.Path, dst: pathlib.Path, every: int) -> tuple[int, int, int]:
    """按卡分组每 every 个采样保留 1 个；各卡首末样本恒保留（保住时间跨度）。

    返回 (读入行数, 写出行数, 卡数)。无法解析卡号的行原样保留（不静默丢数据）。
    """
    lines = src.read_text().splitlines()
    # 先统计每张卡的总样本数，好判定"末样本"
    total_per_gpu: dict[str, int] = {}
    for line in lines:
        parts = line.split(",")
        if len(parts) >= 2:
            total_per_gpu[parts[1].strip()] = total_per_gpu.get(parts[1].strip(), 0) + 1

    seen: dict[str, int] = {}
    kept: list[str] = []
    for line in lines:
        parts = line.split(",")
        if len(parts) < 2:
            kept.append(line)          # 解析不了的行原样保留
            continue
        gpu = parts[1].strip()
        idx = seen.get(gpu, 0)
        seen[gpu] = idx + 1
        is_first = idx == 0
        is_last = idx == total_per_gpu[gpu] - 1
        if is_first or is_last or idx % every == 0:
            kept.append(line)
    dst.write_text("\n".join(kept) + ("\n" if kept else ""))
    return len(lines), len(kept), len(total_per_gpu)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=pathlib.Path, help="全分辨率 gpu_util_dense.csv")
    ap.add_argument("dst", type=pathlib.Path, help="降采样输出路径")
    ap.add_argument("--every", type=int, default=10,
                    help="每张卡每 N 个采样保留 1 个（默认 10 = 500ms → 5s 间隔）")
    args = ap.parse_args()
    if args.every < 1:
        raise SystemExit("--every 必须 ≥1")
    if not args.src.exists():
        raise SystemExit(f"源文件不存在: {args.src}")
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    n_in, n_out, n_gpu = downsample(args.src, args.dst, args.every)
    src_mb = args.src.stat().st_size / 1e6
    dst_mb = args.dst.stat().st_size / 1e6
    print(f"DOWNSAMPLE_OK every={args.every} gpus={n_gpu} "
          f"lines {n_in} → {n_out} ({100 * n_out / n_in:.1f}%)  "
          f"size {src_mb:.2f}MB → {dst_mb:.2f}MB")
    print(f"  全分辨率原件（判读以它为准）: {args.src.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
