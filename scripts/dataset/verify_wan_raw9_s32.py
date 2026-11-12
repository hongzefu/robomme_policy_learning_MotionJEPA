#!/usr/bin/env python3
"""闸 G1：raw9 stride-32 motion 表的独立逐字节校验（1112-4task-modul-wanraw9-s32-50k-plan.html §2、第二部分 1.1 第 4 行）。

不 import 导出脚本，也不调用 motion_store 的网格公式：本脚本自己从清单重算 stride-32 index（段键、num_grid、row_base、
窗起点）与 budget_max，自己从源 latent 段 metadata（stride 16）挑出偶数 m（``seg_offset % 32 == 0``），把
``<latents>/<段>.bin`` 第 m 个 chunk 的 589,824 原字节与表第 m' = m/2 行逐字节比较。源字节读自本机解包后的
``wan-latents/<段>.bin``（tar 已删），身份锚点是 ``--checksums``（bucket ``checksums/sha256-source-files.txt``）里
``wan-latents/<名>`` 的成员摘要：每段 ``.bin`` 与 ``.metadata.json`` 的 sha256 都须与之相等。

另外经 store 读 API 核表自身契约：``MotionMeta.load``（布局、raw9 provenance）、fast 档（清单身份、表大小）、
full 档（整表 sha256）；任何一项抛错都记为 FAIL（不吞，原因写进判定行）。

并行：按段切任务，``multiprocessing.Pool(--workers).imap`` 有序分发；表以只读 ``np.memmap`` 打开。

末行判定：``RAW9_TABLE=PASS rows=… demo=… exec=… budget_max=… mismatch=0``；行数 / demo / exec / budget_max 与
``--expect-*``（默认为主会话 11-12 实测 36484 / 18367 / 18117 / 71）任一不符或 mismatch>0 即 FAIL、退出码 1。本工具只读。
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import motion_store as ms  # noqa: E402
from mme_vla_suite.datastore.manifest import load_manifest  # noqa: E402

# 本脚本自持的几何常量（与导出侧独立书写，互为对照）
CHUNK_SHAPE = (9, 16, 32, 32)
CHUNK_BYTES = 9 * 16 * 32 * 32 * 4          # 589,824
SRC_STRIDE = 16
NEW_STRIDE = 32
WINDOW = 33
DEMO_MIN_REAL = 17
EXEC_MIN_REAL = 33
LAYOUT_NAME = "motion-raw9-grid32-demopad17-v1"
ARC_PREFIX = "wan-latents/"


def seg_key(h5_file: str, raw_ep_idx: int, seg: str) -> str:
    name = os.path.basename(h5_file)
    return f"{name[len('record_dataset_'):-len('.h5')]}_ep{int(raw_ep_idx)}_{seg}"


def expected_index(manifest: dict) -> list[dict]:
    """独立重算：逐 episode 先 demo 后 exec，num_chunks = max(0, L − (min_real − 1))，num_grid = ⌈num_chunks / 32⌉。"""
    out = []
    cursor = 0
    for g, ep in enumerate(manifest["episodes"]):
        es, nt = int(ep["exec_start_idx"]), int(ep["num_timesteps"])
        segs = {}
        for seg, L, min_real, start in (("demo", es, DEMO_MIN_REAL, 0), ("exec", nt - es, EXEC_MIN_REAL, es)):
            nc = max(0, L - (min_real - 1))
            ng = (nc + NEW_STRIDE - 1) // NEW_STRIDE
            segs[seg] = {"key": seg_key(ep["h5_file"], ep["raw_ep_idx"], seg), "row_base": cursor if ng else None,
                         "num_grid": ng, "seg_len": L, "seg_start": start}
            cursor += ng
        out.append({"g": g, "es": es, "nt": nt, **segs})
    return out


def budget_of(ep: dict) -> int:
    """该 episode 最后一帧 t = nt−1 时可见的窗数：demo s=32m、s+16 ≤ es−1；exec u=32m、u+32 ≤ t−es。"""
    es, t = ep["es"], ep["nt"] - 1
    n = sum(1 for m in range(ep["demo"]["num_grid"]) if NEW_STRIDE * m + (DEMO_MIN_REAL - 1) <= es - 1)
    n += sum(1 for m in range(ep["exec"]["num_grid"]) if NEW_STRIDE * m + (WINDOW - 1) <= t - es)
    return n


def load_members(path: pathlib.Path) -> dict[str, str]:
    members = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64 and parts[1].strip().startswith(ARC_PREFIX):
            members[parts[1].strip()] = parts[0]
    return members


_MEMBERS: dict[str, str] = {}


def _init_worker(members: dict[str, str]) -> None:
    """Pool initializer：成员摘要字典每个 worker 只接收一次，不随每个段任务 pickle。"""
    global _MEMBERS
    _MEMBERS = members


def check_segment(job: dict) -> dict:
    """独立比对一段：源成员摘要 → 偶数 m 选取 → 表第 row_base+m' 行与源第 m 个 chunk 逐字节相等。"""
    root = pathlib.Path(job["latents_root"])
    key = job["key"]
    problems: list[str] = []
    data = (root / f"{key}.bin").read_bytes()
    meta_bytes = (root / f"{key}.metadata.json").read_bytes()
    for name, b in ((f"{key}.bin", data), (f"{key}.metadata.json", meta_bytes)):
        if _MEMBERS.get(ARC_PREFIX + name) != hashlib.sha256(b).hexdigest():
            problems.append(f"{name} sha256 与成员清单不符")
    meta = json.loads(meta_bytes.decode("utf-8"))
    rows_meta = meta.get("rows") or []
    if meta.get("segment") != key or meta.get("seg_start_global") != job["seg_start"] \
            or meta.get("grid_stride") != SRC_STRIDE or len(data) != len(rows_meta) * CHUNK_BYTES:
        problems.append(f"源 metadata 身份/网格不符: segment={meta.get('segment')} "
                        f"grid_stride={meta.get('grid_stride')} rows={len(rows_meta)} bytes={len(data)}")
    picked = [r for r in rows_meta if int(r.get("seg_offset", -1)) % NEW_STRIDE == 0]
    if len(picked) != job["num_grid"]:
        problems.append(f"偶数 m 窗数 {len(picked)} != 新 index num_grid {job['num_grid']}")
    mismatch = 0
    padded = 0
    table = np.memmap(job["table"], dtype="<f4", mode="r", shape=(job["num_rows"],) + CHUNK_SHAPE)
    try:
        for i, r in enumerate(picked[:job["num_grid"]]):
            m = int(r["m"])
            if int(r["seg_offset"]) != SRC_STRIDE * m or m % 2 or m // 2 != i \
                    or int(r["start_global_frame"]) != job["seg_start"] + NEW_STRIDE * i:
                problems.append(f"m={m} 不是第 {i} 个偶数窗或起点不符: {r}")
                mismatch += 1
                continue
            src = data[m * CHUNK_BYTES:(m + 1) * CHUNK_BYTES]
            if table[job["row_base"] + i].tobytes() != src:
                mismatch += 1
            padded += int((r.get("pad_frames") or 0) > 0)
    finally:
        del table
    return {"key": key, "problems": problems, "mismatch": mismatch, "padded": padded}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="<lib>/motion-wan-raw9-s32")
    ap.add_argument("--latents", required=True, help="<lib>/wan-latents（解包后的运行时目录）")
    ap.add_argument("--checksums", required=True, help="bucket checksums/sha256-source-files.txt（成员摘要清单）")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--expect-rows", type=int, default=36484)
    ap.add_argument("--expect-demo", type=int, default=18367)
    ap.add_argument("--expect-exec", type=int, default=18117)
    ap.add_argument("--expect-budget-max", type=int, default=71)
    args = ap.parse_args()

    table_root = pathlib.Path(args.table).resolve()
    latents_root = pathlib.Path(args.latents).resolve()

    def fail(reason: str) -> None:
        print(f"RAW9_TABLE=FAIL reason={reason}", flush=True)
        raise SystemExit(1)

    # 一、表自身契约（store 读 API）：任何一项抛错即 FAIL
    try:
        meta = ms.MotionMeta.load(table_root)
        if meta.raw.get("layout") != LAYOUT_NAME or tuple(meta.row_shape) != CHUNK_SHAPE \
                or meta.grid_stride != NEW_STRIDE:
            fail(f"layout/row_shape/grid_stride={meta.raw.get('layout')}/{meta.row_shape}/{meta.grid_stride}")
        ms.run_fast_checks(meta)
        ms.run_full_checks(meta)
        manifest = load_manifest(meta.manifest_path)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — 判定行须写出原因，不吞退出码
        fail(f"{type(e).__name__}: {str(e)[:200]}")
    members = load_members(pathlib.Path(args.checksums).resolve())
    if not members:
        fail("成员清单里没有 wan-latents/ 行")

    # 二、独立重算 index 与 budget_max，与表 index 逐段比
    want = expected_index(manifest)
    index_mismatch = 0
    if len(want) != len(meta.entries):
        index_mismatch += 1
    for w, e in zip(want, meta.entries):
        for seg in ("demo", "exec"):
            s = getattr(e, seg)
            if (w[seg]["key"], w[seg]["row_base"], w[seg]["num_grid"], w[seg]["seg_len"]) != \
                    (ms.segment_key(e.h5_file, e.raw_ep_idx, seg), s.row_base, s.num_grid, s.seg_len):
                index_mismatch += 1
    rows = sum(w[s]["num_grid"] for w in want for s in ("demo", "exec"))
    demo = sum(w["demo"]["num_grid"] for w in want)
    exe = sum(w["exec"]["num_grid"] for w in want)
    budget_max = max((budget_of(w) for w in want), default=0)
    if rows != meta.num_rows:
        index_mismatch += 1

    # 三、逐段逐字节比对
    jobs = [{"key": w[seg]["key"], "row_base": w[seg]["row_base"], "num_grid": w[seg]["num_grid"],
             "seg_start": w[seg]["seg_start"], "latents_root": str(latents_root),
             "table": str(table_root / ms.MOTION_TABLE_RELPATH), "num_rows": meta.num_rows}
            for w in want for seg in ("demo", "exec") if w[seg]["num_grid"]]
    mismatch = 0
    padded = 0
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(members,)) as pool:
        for res in pool.imap(check_segment, jobs, chunksize=1):
            if res["problems"]:
                index_mismatch += len(res["problems"])
                print(f"[G1] {res['key']}: {res['problems'][:3]}", flush=True)
            mismatch += res["mismatch"]
            padded += res["padded"]

    total_mismatch = mismatch + index_mismatch
    ok = (total_mismatch == 0 and rows == args.expect_rows and demo == args.expect_demo
          and exe == args.expect_exec and budget_max == args.expect_budget_max)
    line = (f"RAW9_TABLE={'PASS' if ok else 'FAIL'} rows={rows} demo={demo} exec={exe} "
            f"budget_max={budget_max} mismatch={total_mismatch}")
    if index_mismatch:
        line += f" index_mismatch={index_mismatch}"
    print(f"[G1] 新表补帧窗 padded={padded}", flush=True)
    print(line, flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
