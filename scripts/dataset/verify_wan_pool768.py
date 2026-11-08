#!/usr/bin/env python3
"""G3：768 池化 motion 表的独立重算与对齐校验（1108-4task-modul-wantp768-50k-plan.html §5 G3、第二部分 1.2）。

不复用 ``wan_pool768.py``：每窗用 float64 的 torch 参考独立重算——``Z[1:9]`` 在 float64 下取时间均值，再
``torch.nn.functional.adaptive_avg_pool2d(·, (6,8))``，C→H→W 展平——与表行（float32）逐元素比 ``max_abs``，
阈值 1e-6。并行方式与导出相同：按 motion_index 段切任务，``multiprocessing.Pool(--workers).imap`` 有序分发，
``OMP_NUM_THREADS=1``，torch 单线程。

同时核：
- 表的 MotionMeta / fast / full 档（契约、池化 provenance、清单身份、表 sha256）；
- 新表 motion_index.json 与 ``--motion-ref`` 的逐字节相同、逐段键 / row_base / num_grid 相同；
- 逐段 latent metadata 的段键、num_grid、窗起点（demo 0,16,…；exec es+0,16,…）与 index 相同；
- 行数 / demo / exec / 补帧窗计数等于期望（默认 71316 = 35913 + 35403，补帧窗 1600）；全表有限。

末行判定：``POOL768_TABLE=PASS rows=… demo=… exec=… padded=… index_mismatch=0 max_abs=<x>``，否则 FAIL 且退出码 1。
本工具只读，不改表。
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import motion_store as ms  # noqa: E402

LATENT_SHAPE = (9, 16, 32, 32)
LATENT_CHUNK_BYTES = int(np.prod(LATENT_SHAPE)) * 4
TOL = 1e-6


def reference_rows(lat: np.ndarray):
    """float64 torch 参考：(n,9,16,32,32) → (n,768) float64。"""
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    z = torch.from_numpy(lat.astype(np.float64))
    a = z[:, 1:9].mean(dim=1)                              # (n,16,32,32)
    p = F.adaptive_avg_pool2d(a, (6, 8))                   # (n,16,6,8)
    return p.reshape(p.shape[0], 768).numpy()


def check_segment(job: dict) -> dict:
    root = pathlib.Path(job["latents_root"])
    key = job["key"]
    ng = job["num_grid"]
    problems: list[str] = []
    data = (root / f"{key}.bin").read_bytes()
    if len(data) != ng * LATENT_CHUNK_BYTES:
        return {"key": key, "problems": [f"latent 字节数 {len(data)} != {ng}×{LATENT_CHUNK_BYTES}"],
                "max_abs": float("inf"), "padded": 0, "finite": False}
    meta = json.loads((root / f"{key}.metadata.json").read_text(encoding="utf-8"))
    for k in ("segment", "num_grid", "segment_kind", "seg_start_global"):
        want = {"segment": key, "num_grid": ng, "segment_kind": job["segment"],
                "seg_start_global": job["seg_start"]}[k]
        if meta.get(k) != want:
            problems.append(f"metadata.{k}={meta.get(k)!r} != {want!r}")
    rows_meta = meta.get("rows") or []
    if len(rows_meta) != ng:
        problems.append(f"metadata.rows 条数 {len(rows_meta)} != {ng}")
    padded = 0
    for m, r in enumerate(rows_meta[:ng]):
        if r.get("start_global_frame") != job["seg_start"] + ms.GRID_STRIDE * m:
            problems.append(f"m={m} start_global_frame={r.get('start_global_frame')}")
        padded += int((r.get("pad_frames") or 0) > 0)
    lat = np.frombuffer(data, dtype="<f4").reshape((ng,) + LATENT_SHAPE)
    table = np.memmap(job["table"], dtype="<f4", mode="r", shape=(job["num_rows"], 768))
    got = np.array(table[job["row_base"]:job["row_base"] + ng], dtype=np.float32)
    del table
    finite = bool(np.isfinite(got).all())
    ref = reference_rows(lat)
    max_abs = float(np.abs(got.astype(np.float64) - ref).max()) if finite else float("inf")
    return {"key": key, "problems": problems, "max_abs": max_abs, "padded": padded, "finite": finite}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="<lib>/motion-wan-pool768")
    ap.add_argument("--latents", required=True, help="<lib>/wan-latents")
    ap.add_argument("--motion-ref", required=True, help="<lib>/motion（上次 encoder 表）")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--expect-rows", type=int, default=71316)
    ap.add_argument("--expect-demo", type=int, default=35913)
    ap.add_argument("--expect-exec", type=int, default=35403)
    ap.add_argument("--expect-padded", type=int, default=1600)
    args = ap.parse_args()

    table_root = pathlib.Path(args.table).resolve()
    ref_root = pathlib.Path(args.motion_ref).resolve()
    latents_root = pathlib.Path(args.latents).resolve()
    meta = ms.MotionMeta.load(table_root)                     # 含池化 provenance 闸
    ms.run_fast_checks(meta)
    ms.run_full_checks(meta)
    ref = ms.MotionMeta.load(ref_root)

    index_mismatch = 0
    if (table_root / ms.INDEX_RELPATH).read_bytes() != (ref_root / ms.INDEX_RELPATH).read_bytes():
        print("[G3] motion_index.json 与 motion-ref 字节不同", flush=True)
        index_mismatch += 1
    if len(meta.entries) != len(ref.entries):
        index_mismatch += 1
    for a, b in zip(meta.entries, ref.entries):
        for seg in ms.SEGMENTS:
            sa, sb = getattr(a, seg), getattr(b, seg)
            if (ms.segment_key(a.h5_file, a.raw_ep_idx, seg), sa.row_base, sa.num_grid, sa.seg_len, a.exec_start_idx) != \
                    (ms.segment_key(b.h5_file, b.raw_ep_idx, seg), sb.row_base, sb.num_grid, sb.seg_len, b.exec_start_idx):
                index_mismatch += 1
    totals = ms.index_totals(list(meta.entries))

    jobs = []
    for e in meta.entries:
        for seg in ms.SEGMENTS:
            s = getattr(e, seg)
            if s.num_grid == 0:
                continue
            jobs.append({"key": ms.segment_key(e.h5_file, e.raw_ep_idx, seg), "segment": seg,
                         "row_base": s.row_base, "num_grid": s.num_grid,
                         "seg_start": 0 if seg == "demo" else e.exec_start_idx,
                         "latents_root": str(latents_root), "table": str(table_root / ms.MOTION_TABLE_RELPATH),
                         "num_rows": meta.num_rows})
    max_abs = 0.0
    padded = 0
    nonfinite = 0
    with mp.Pool(args.workers) as pool:
        for res in pool.imap(check_segment, jobs, chunksize=1):
            if res["problems"]:
                index_mismatch += len(res["problems"])
                print(f"[G3] {res['key']}: {res['problems'][:3]}", flush=True)
            nonfinite += int(not res["finite"])
            max_abs = max(max_abs, res["max_abs"])
            padded += res["padded"]

    ok = (index_mismatch == 0 and nonfinite == 0 and max_abs <= TOL
          and totals["rows"] == args.expect_rows == meta.num_rows
          and totals["demo_rows"] == args.expect_demo and totals["exec_rows"] == args.expect_exec
          and padded == args.expect_padded)
    line = (f"POOL768_TABLE={'PASS' if ok else 'FAIL'} rows={totals['rows']} demo={totals['demo_rows']} "
            f"exec={totals['exec_rows']} padded={padded} index_mismatch={index_mismatch} max_abs={max_abs:.3e}")
    if nonfinite:
        line += f" nonfinite_segments={nonfinite}"
    print(line, flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
