#!/usr/bin/env python3
"""补下 RoboMME 官方 16 任务的原始 H5（公开集 ``Yinpei/robomme_data_h5``，train split，每任务 100 episode）。

**为什么需要**：环境 B 起初只下了 v1 的 4 个目标任务（``paths.sh`` 的 ``TARGET_TASKS``），
按官方数据集口径统计全部 16 个任务的 episode 长度与 demo 前缀时缺另外 12 个。

HF 上 16 个任务**全部是 ``.tar.xz``、没有裸 ``.h5``**（xz 流式压缩不能随机访问），
所以只能整包下载 + 解压，无法只取 metadata。已存在 ``record_dataset_<Task>.h5`` 的任务自动跳过；
``.tar.xz`` 与解压件都保留，与 2026-09-04 下载 4 个时一致。

sha256 逐文件算出后写 manifest（AGENTS 15），供后续与其他环境对拍同源。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

REPO_ID = "Yinpei/robomme_data_h5"

# 与 examples/robomme/utils.py::TASK_NAME_LIST 同序（四组各 4 个）
ALL_TASKS = [
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes",
    "ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def fetch_one(task: str, raw_dir: pathlib.Path) -> dict:
    from huggingface_hub import hf_hub_download

    h5 = raw_dir / f"record_dataset_{task}.h5"
    xz = raw_dir / f"record_dataset_{task}.h5.tar.xz"
    t0 = time.perf_counter()

    if h5.exists() and h5.stat().st_size > 0:
        _log(f"[{task}] 已有 h5，跳过")
        return {"task": task, "skipped": True, "h5_bytes": h5.stat().st_size}

    if not (xz.exists() and xz.stat().st_size > 0):
        _log(f"[{task}] 下载 tar.xz 开始")
        hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                        filename=xz.name, local_dir=str(raw_dir))
        _log(f"[{task}] 下载完成 {xz.stat().st_size / 1e9:.2f} GB ({time.perf_counter() - t0:.0f}s)")
    else:
        _log(f"[{task}] tar.xz 已存在，跳过下载")

    _log(f"[{task}] 解压开始")
    subprocess.run(["tar", "-xJf", str(xz), "-C", str(raw_dir)], check=True)
    if not (h5.exists() and h5.stat().st_size > 0):
        raise RuntimeError(f"[{task}] 解压后缺 {h5.name}")
    _log(f"DONE_{task} h5_bytes={h5.stat().st_size} 用时={time.perf_counter() - t0:.0f}s")
    return {"task": task, "skipped": False, "h5_bytes": h5.stat().st_size}


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/scratch/hongze/robomme_data_h5")
    ap.add_argument("--tasks", default="", help="逗号分隔；默认全部 16 个")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--manifest-out", default="", help="写 sha256 清单的路径（留空则不算 sha256）")
    args = ap.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in args.tasks.split(",") if t] or ALL_TASKS
    unknown = [t for t in tasks if t not in ALL_TASKS]
    if unknown:
        raise SystemExit(f"错误: 未知任务 {unknown}，只接受 {ALL_TASKS}")

    _log(f"=== FETCH_H5 raw_dir={raw_dir} tasks={len(tasks)} jobs={args.jobs} start={time.strftime('%F %T')} ===")
    t0 = time.perf_counter()
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(fetch_one, t, raw_dir): t for t in tasks}
        for fut in cf.as_completed(futs):
            results.append(fut.result())     # 任一任务失败即抛出，不吞异常

    got = sorted(r["task"] for r in results)
    _log(f"全部完成 tasks={len(got)} 新取={sum(1 for r in results if not r['skipped'])} "
         f"用时={time.perf_counter() - t0:.0f}s")

    if args.manifest_out:
        _log("算 sha256（逐文件，AGENTS 15）…")
        entries = {}
        with cf.ThreadPoolExecutor(max_workers=min(8, args.jobs * 2)) as ex:
            futs = {ex.submit(sha256_of, raw_dir / f"record_dataset_{t}.h5"): t for t in got}
            for fut in cf.as_completed(futs):
                t = futs[fut]
                p = raw_dir / f"record_dataset_{t}.h5"
                entries[f"record_dataset_{t}.h5"] = {"size": p.stat().st_size, "sha256": fut.result()}
                _log(f"  sha256 {t} 完成")
        out = pathlib.Path(args.manifest_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"repo_id": REPO_ID, "raw_dir": str(raw_dir),
             "files": dict(sorted(entries.items()))}, indent=2, ensure_ascii=False) + "\n")
        _log(f"清单已写 {out}")


if __name__ == "__main__":
    main()
