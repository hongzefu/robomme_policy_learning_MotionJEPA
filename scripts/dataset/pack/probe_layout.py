#!/usr/bin/env python3
"""源 npy 布局探针（v2 计划 5.1「探针脚本固化」）。

复核 pack --reader slice 依赖的数据格式常量（历次交互探针的固化版）：
  st_size == 602,951
  image_emb_4x4 数据段偏移 262,595（(1,16,2048) bf16，65,536 B）
  pos_emb_4x4  数据段偏移 541,352（(1,16,768)  f32，49,152 B）
  state_emb    数据段偏移 602,906（(8,)        f32，32 B）
对每个抽样文件：decode 出三键 → raw.find 定位字节段 → 与常量比对；
任一不符即非零退出（此时禁用 --reader slice，改用 decode 档并复核格式变更来源）。

用法：
  uv run python scripts/data-pack-framesamp/probe_layout.py --source <4task-gl 根> [--n 30]
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import framesamp_store as fs  # noqa: E402


def probe_one(path: pathlib.Path) -> list[str]:
    """返回该文件的失配清单（空 = 全部常量成立）。"""
    bad: list[str] = []
    size = path.stat().st_size
    if size != fs.SOURCE_NPY_SIZE:
        bad.append(f"st_size={size} != {fs.SOURCE_NPY_SIZE}")
        return bad   # 大小都不对，偏移无从谈起
    raw = path.read_bytes()
    with path.open("rb") as f:
        d = np.load(f, allow_pickle=True).item()
    for key, offset, shape, dtype in (
            (fs.IMAGE_KEY, fs.SOURCE_IMAGE_OFFSET, (1,) + fs.IMAGE_ROW_SHAPE, fs.IMAGE_DTYPE),
            (fs.POS_KEY, fs.SOURCE_POS_OFFSET, (1,) + fs.POS_ROW_SHAPE, np.float32),
            (fs.STATE_KEY, fs.SOURCE_STATE_OFFSET, fs.STATE_ROW_SHAPE, np.float32)):
        arr = d[key]
        if arr.shape != shape or arr.dtype != dtype:
            bad.append(f"{key} 形制 {arr.shape}/{arr.dtype} != {shape}/{np.dtype(dtype)}")
            continue
        blob = arr.tobytes()
        found = raw.find(blob)
        if found != offset:
            bad.append(f"{key} 偏移 {found} != {offset}")
        if raw[offset:offset + len(blob)] != blob:
            bad.append(f"{key} 窗口字节与 decode 不符")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="4task-gl 源库根")
    ap.add_argument("--n", type=int, default=30, help="随机抽样帧数（另含首 episode 首帧）")
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    src = pathlib.Path(args.source)
    ep_dirs = sorted((src / "features").glob("episode_*"),
                     key=lambda p: int(p.name.split("_")[1]))
    if not ep_dirs:
        raise SystemExit(f"未找到 features/episode_*: {src}")
    rng = random.Random(args.seed)
    picks = [ep_dirs[0] / "token_emb_0.npy"]
    for _ in range(args.n):
        ep = rng.choice(ep_dirs)
        frames = list(ep.glob("token_emb_*.npy"))
        picks.append(rng.choice(frames))

    n_bad = 0
    for p in picks:
        bad = probe_one(p)
        if bad:
            n_bad += 1
            print(f"FAIL {p}: {'; '.join(bad)}")
    if n_bad:
        print(f"PROBE_LAYOUT=FAIL files={len(picks)} bad={n_bad} —— "
              f"禁用 --reader slice，改用 decode 并复核格式变更")
        raise SystemExit(1)
    print(f"PROBE_LAYOUT=PASS files={len(picks)} "
          f"st_size={fs.SOURCE_NPY_SIZE} offsets=({fs.SOURCE_IMAGE_OFFSET},"
          f"{fs.SOURCE_POS_OFFSET},{fs.SOURCE_STATE_OFFSET})")


if __name__ == "__main__":
    main()
