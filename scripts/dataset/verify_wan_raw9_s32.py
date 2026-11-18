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

``--stride 16``（1117-e2e-enc-train-plan.html 第二部分一节本文件行、三节 G2）：校验步长 16 bf16 新表
``motion-raw9bf16-grid16-demopad17-v1``。偶数检验退化为恒等（取全部 m）；内容校验为「新表行 == bf16(源 latent 第 m 个
chunk) 逐字节相等」，bf16 参照由本脚本直接 ``astype(ml_dtypes.bfloat16)`` 独立写出（不 import 导出脚本，也不 import
``raw_bf16``）。须给 ``--motion-ref``（768 维表）：期望行数 / demo / exec / budget_max 从它的 index 用本脚本自持公式
重算，且新表 index 与它逐行相等（含 demo_min_real / demo_tail_pad spec 与 grid_stride）；给 ``--s32-table`` 时再做
s16 表偶数 m 行 == bf16(s32 表对应行) 逐字节相等（s32 表 table_sha256 须等于 ``--expect-s32-table-sha256``）。
输出三行判定（任一 FAIL 退出码 1）：

  INDEX_VS_MOTION768=PASS rows=71316 mismatches=0
  S16_EVEN_VS_S32=PASS rows=36484 mismatches=0
  EXPORT_RAW16=PASS rows=71316 dtype=bf16 row_bytes=294912 demo=… exec=… budget_max=… budget_limit=160 mismatch=0
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

import ml_dtypes  # noqa: E402
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
# 步长 16 bf16 新表（1117 计划）
S16_LAYOUT_NAME = "motion-raw9bf16-grid16-demopad17-v1"
BF16 = np.dtype(ml_dtypes.bfloat16)
S16_ROW_BYTES = 9 * 16 * 32 * 32 * 2        # 294,912
S32_TABLE_SHA256 = "0e6657af118e65c762b86dfd86ce0794289108b3be37ad5017fe9c26d1442d19"


def seg_key(h5_file: str, raw_ep_idx: int, seg: str) -> str:
    name = os.path.basename(h5_file)
    return f"{name[len('record_dataset_'):-len('.h5')]}_ep{int(raw_ep_idx)}_{seg}"


def expected_index(manifest: dict, stride: int = NEW_STRIDE) -> list[dict]:
    """独立重算：逐 episode 先 demo 后 exec，num_chunks = max(0, L − (min_real − 1))，num_grid = ⌈num_chunks / stride⌉。"""
    out = []
    cursor = 0
    for g, ep in enumerate(manifest["episodes"]):
        es, nt = int(ep["exec_start_idx"]), int(ep["num_timesteps"])
        segs = {}
        for seg, L, min_real, start in (("demo", es, DEMO_MIN_REAL, 0), ("exec", nt - es, EXEC_MIN_REAL, es)):
            nc = max(0, L - (min_real - 1))
            ng = (nc + stride - 1) // stride
            segs[seg] = {"key": seg_key(ep["h5_file"], ep["raw_ep_idx"], seg), "row_base": cursor if ng else None,
                         "num_grid": ng, "num_chunks": nc, "seg_len": L, "seg_start": start}
            cursor += ng
        out.append({"g": g, "es": es, "nt": nt, **segs})
    return out


def budget_of(ep: dict, stride: int = NEW_STRIDE) -> int:
    """该 episode 最后一帧 t = nt−1 时可见的窗数：demo s=Sm、s+16 ≤ es−1；exec u=Sm、u+32 ≤ t−es（S = stride）。"""
    es, t = ep["es"], ep["nt"] - 1
    n = sum(1 for m in range(ep["demo"]["num_grid"]) if stride * m + (DEMO_MIN_REAL - 1) <= es - 1)
    n += sum(1 for m in range(ep["exec"]["num_grid"]) if stride * m + (WINDOW - 1) <= t - es)
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


def check_segment_s16(job: dict) -> dict:
    """s16 一段：源成员摘要 → 全部 m（恒等选取）→ 表第 row_base+m 行 == astype(bf16)(源第 m 个 chunk) 逐字节；
    给了 s32 表时，偶数 m 再比 表第 row_base+m 行 == astype(bf16)(s32 表第 s32_row_base+m/2 行)。"""
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
    picked = [r for r in rows_meta if int(r.get("seg_offset", -1)) % SRC_STRIDE == 0]     # s16：恒等选取
    if len(picked) != job["num_grid"]:
        problems.append(f"窗数 {len(picked)} != 新 index num_grid {job['num_grid']}")
    mismatch = 0
    even_rows = 0
    even_mismatch = 0
    src_all = np.frombuffer(data, dtype="<f4")
    table = np.memmap(job["table"], dtype=BF16, mode="r", shape=(job["num_rows"],) + CHUNK_SHAPE)
    t32 = None
    if job.get("s32_table"):
        t32 = np.memmap(job["s32_table"], dtype="<f4", mode="r", shape=(job["s32_num_rows"],) + CHUNK_SHAPE)
    try:
        for i, r in enumerate(picked[:job["num_grid"]]):
            m = int(r["m"])
            if int(r["seg_offset"]) != SRC_STRIDE * m or m != i or int(r["start_global_frame"]) != job["seg_start"] + SRC_STRIDE * i:
                problems.append(f"m={m} 不是第 {i} 个窗或起点不符: {r}")
                mismatch += 1
                continue
            got = table[job["row_base"] + i].tobytes()
            src = src_all[m * CHUNK_BYTES // 4:(m + 1) * CHUNK_BYTES // 4]
            if got != src.astype(BF16).tobytes():
                mismatch += 1
            if t32 is not None and m % 2 == 0:
                even_rows += 1
                if m // 2 >= job["s32_num_grid"] or \
                        got != np.asarray(t32[job["s32_row_base"] + m // 2]).astype(BF16).tobytes():
                    even_mismatch += 1
        if t32 is not None and even_rows != job["s32_num_grid"]:
            problems.append(f"偶数 m 行数 {even_rows} != s32 index num_grid {job['s32_num_grid']}")
            even_mismatch += 1
    finally:
        del table, t32
    return {"key": key, "problems": problems, "mismatch": mismatch, "even_rows": even_rows,
            "even_mismatch": even_mismatch}


def index_vs_ref(meta, ref) -> int:
    """新表 index 与 768 维表 index 逐行比：身份五字段、两段 (row_base, num_grid, num_chunks, seg_len)、spec 三键与步长。"""
    bad = 0
    if meta.spec.fields() != ref.spec.fields() or meta.grid_stride != ref.grid_stride:
        bad += 1
    if len(meta.entries) != len(ref.entries) or meta.num_rows != ref.num_rows:
        bad += 1
    for a, b in zip(meta.entries, ref.entries):
        ka = (a.g, a.h5_file, a.raw_ep_idx, a.num_timesteps, a.exec_start_idx)
        kb = (b.g, b.h5_file, b.raw_ep_idx, b.num_timesteps, b.exec_start_idx)
        for seg in ("demo", "exec"):
            sa, sb = getattr(a, seg), getattr(b, seg)
            if ka != kb or (sa.row_base, sa.num_grid, sa.num_chunks, sa.seg_len) != \
                    (sb.row_base, sb.num_grid, sb.num_chunks, sb.seg_len):
                bad += 1
    return bad


def main_s16(args) -> None:
    """--stride 16：INDEX_VS_MOTION768、S16_EVEN_VS_S32（给 --s32-table 时）、EXPORT_RAW16 三行判定。"""
    table_root = pathlib.Path(args.table).resolve()
    latents_root = pathlib.Path(args.latents).resolve()

    def fail(reason: str) -> None:
        print(f"EXPORT_RAW16=FAIL reason={reason}", flush=True)
        raise SystemExit(1)

    if not args.motion_ref:
        fail("--stride 16 须给 --motion-ref（768 维表）")
    try:
        meta = ms.MotionMeta.load(table_root)
        t = meta.raw["tables"][ms.MOTION_KEY]
        if meta.raw.get("layout") != S16_LAYOUT_NAME or tuple(meta.row_shape) != CHUNK_SHAPE \
                or meta.grid_stride != SRC_STRIDE or t.get("dtype") != "bfloat16" or meta.row_bytes != S16_ROW_BYTES:
            fail(f"layout/row_shape/grid_stride/dtype/row_bytes={meta.raw.get('layout')}/{meta.row_shape}/"
                 f"{meta.grid_stride}/{t.get('dtype')}/{meta.row_bytes}")
        ms.run_fast_checks(meta)
        ms.run_full_checks(meta)
        manifest = load_manifest(meta.manifest_path)
        ref = ms.MotionMeta.load(pathlib.Path(args.motion_ref).resolve())
        if ref.manifest_sha256 != meta.manifest_sha256:
            fail("motion-ref 与新表绑定的清单不同")
        s32 = None
        if args.s32_table:
            s32 = ms.MotionMeta.load(pathlib.Path(args.s32_table).resolve())
            if s32.raw.get("layout") != LAYOUT_NAME or s32.manifest_sha256 != meta.manifest_sha256:
                fail(f"s32 表 layout / 清单不符: {s32.raw.get('layout')}")
            if s32.table_sha256 != args.expect_s32_table_sha256:
                fail(f"s32 表 table_sha256={s32.table_sha256} != 期望 {args.expect_s32_table_sha256}")
            ms.run_fast_checks(s32)
            ms.run_full_checks(s32)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — 判定行须写出原因，不吞退出码
        fail(f"{type(e).__name__}: {str(e)[:200]}")
    members = load_members(pathlib.Path(args.checksums).resolve())
    if not members:
        fail("成员清单里没有 wan-latents/ 行")

    # 一、INDEX_VS_MOTION768
    idx_bad = index_vs_ref(meta, ref)
    print(f"INDEX_VS_MOTION768={'PASS' if idx_bad == 0 else 'FAIL'} rows={meta.num_rows} mismatches={idx_bad}",
          flush=True)

    # 二、期望值：从 768 表 index 用自持公式重算（独立于 ms 网格公式）；与清单重算的 s16 index 逐段比
    want = expected_index(manifest, SRC_STRIDE)
    ref_eps = [{"es": e.exec_start_idx, "nt": e.num_timesteps, "demo": {"num_grid": e.demo.num_grid},
                "exec": {"num_grid": e.exec.num_grid}} for e in ref.entries]
    exp_rows = ref.num_rows
    exp_demo = sum(e.demo.num_grid for e in ref.entries)
    exp_exec = sum(e.exec.num_grid for e in ref.entries)
    exp_budget = max((budget_of(e, SRC_STRIDE) for e in ref_eps), default=0)
    index_mismatch = 0
    if len(want) != len(meta.entries):
        index_mismatch += 1
    for w, e in zip(want, meta.entries):
        for seg in ("demo", "exec"):
            s_ = getattr(e, seg)
            if (w[seg]["key"], w[seg]["row_base"], w[seg]["num_grid"], w[seg]["num_chunks"], w[seg]["seg_len"]) != \
                    (ms.segment_key(e.h5_file, e.raw_ep_idx, seg), s_.row_base, s_.num_grid, s_.num_chunks, s_.seg_len):
                index_mismatch += 1
    rows = sum(w[x]["num_grid"] for w in want for x in ("demo", "exec"))
    demo = sum(w["demo"]["num_grid"] for w in want)
    exe = sum(w["exec"]["num_grid"] for w in want)
    budget_max = max((budget_of(w, SRC_STRIDE) for w in want), default=0)
    if rows != meta.num_rows:
        index_mismatch += 1

    # 三、逐段逐字节：新表 == bf16(源)；偶数 m == bf16(s32 行)
    s32_by = {}
    if s32 is not None:
        for e in s32.entries:
            for seg in ("demo", "exec"):
                s_ = getattr(e, seg)
                s32_by[(e.g, seg)] = (s_.row_base, s_.num_grid)
    jobs = []
    for g, w in enumerate(want):
        for seg in ("demo", "exec"):
            if not w[seg]["num_grid"]:
                continue
            job = {"key": w[seg]["key"], "row_base": w[seg]["row_base"], "num_grid": w[seg]["num_grid"],
                   "seg_start": w[seg]["seg_start"], "latents_root": str(latents_root),
                   "table": str(table_root / ms.MOTION_TABLE_RELPATH), "num_rows": meta.num_rows}
            if s32 is not None:
                rb, ng = s32_by.get((g, seg), (None, 0))
                job.update(s32_table=str(pathlib.Path(args.s32_table).resolve() / ms.MOTION_TABLE_RELPATH),
                           s32_num_rows=s32.num_rows, s32_row_base=rb, s32_num_grid=ng)
            jobs.append(job)
    mismatch = 0
    even_rows = 0
    even_mismatch = 0
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(members,)) as pool:
        for res in pool.imap(check_segment_s16, jobs, chunksize=1):
            if res["problems"]:
                index_mismatch += len(res["problems"])
                print(f"[G2] {res['key']}: {res['problems'][:3]}", flush=True)
            mismatch += res["mismatch"]
            even_rows += res["even_rows"]
            even_mismatch += res["even_mismatch"]

    ok_even = True
    if s32 is not None:
        if even_rows != s32.num_rows:
            even_mismatch += 1
        ok_even = even_mismatch == 0
        print(f"S16_EVEN_VS_S32={'PASS' if ok_even else 'FAIL'} rows={even_rows} mismatches={even_mismatch}",
              flush=True)
    total_mismatch = mismatch + index_mismatch
    ok = (total_mismatch == 0 and rows == exp_rows == args.expect_rows and demo == exp_demo and exe == exp_exec
          and budget_max == exp_budget and budget_max <= args.budget_limit)
    line = (f"EXPORT_RAW16={'PASS' if ok else 'FAIL'} rows={rows} dtype=bf16 row_bytes={meta.row_bytes} "
            f"demo={demo} exec={exe} budget_max={budget_max} budget_limit={args.budget_limit} mismatch={total_mismatch}")
    if index_mismatch:
        line += f" index_mismatch={index_mismatch}"
    if not ok:
        line += f" expect_rows={args.expect_rows} ref_rows={exp_rows} ref_demo={exp_demo} ref_exec={exp_exec} " \
                f"ref_budget_max={exp_budget}"
    print(line, flush=True)
    if not (ok and ok_even and idx_bad == 0):
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="<lib>/motion-wan-raw9-s32")
    ap.add_argument("--latents", required=True, help="<lib>/wan-latents（解包后的运行时目录）")
    ap.add_argument("--checksums", required=True, help="bucket checksums/sha256-source-files.txt（成员摘要清单）")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--expect-rows", type=int, default=None)
    ap.add_argument("--expect-demo", type=int, default=18367)
    ap.add_argument("--expect-exec", type=int, default=18117)
    ap.add_argument("--expect-budget-max", type=int, default=71)
    ap.add_argument("--stride", type=int, choices=(16, 32), default=32, help="被校验表的步长（默认 32）")
    ap.add_argument("--motion-ref", help="--stride 16：768 维表根（期望值与 index 逐行比对的来源）")
    ap.add_argument("--s32-table", help="--stride 16：s32 raw9 表根（给出即做 S16_EVEN_VS_S32）")
    ap.add_argument("--expect-s32-table-sha256", default=S32_TABLE_SHA256)
    ap.add_argument("--budget-limit", type=int, default=160, help="--stride 16：budget_max 上限")
    args = ap.parse_args()
    if args.expect_rows is None:                       # 默认：s32 为 11-12 实测 36484，s16 为 768 表行数 71316
        args.expect_rows = 71316 if args.stride == 16 else 36484
    if args.stride == 16:
        main_s16(args)
        return

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
