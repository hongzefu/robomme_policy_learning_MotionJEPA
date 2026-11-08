#!/usr/bin/env python3
"""Wan raw latent 零参数池化 768 维 motion 表导出（1108-4task-modul-wantp768-50k-plan.html §3.2、第二部分 1.2）。

按上次 encoder motion 表 ``<motion-ref>/meta/motion_index.json`` 的段序（episode 序，先 demo 后 exec）逐段读
``<latents>/<段>.bin``（num_grid × (9,16,32,32) float32 裸字节，extract_wan.py 写出），逐窗调
``mme_vla_suite.shared.wan_pool768.pool_raw_wan_latent_768``，按 row_base 顺序写新表：

  <out>/motion_token.f32.bin           (rows, 768) f32；行序与 motion-ref 逐行相同
  <out>/meta/motion_index.json         motion-ref 的 motion_index.json 原字节照抄（段键、row_base、窗起点逐项相同）
  <out>/meta/row_digests.blake2b.bin   逐行 blake2b-128
  <out>/meta/store_meta.json           布局 motion-768-grid16-demopad17-v1；provenance 记 encoder_kind / readout /
                                       encoder_ckpt=null / parameter_count=0 / pool_source_sha256 / latent 根成员摘要 /
                                       manifest canonical identity

并行：段级任务经 ``multiprocessing.Pool(--workers).imap``（有序、chunksize=1）分发，worker 只做纯 NumPy 池化，
主进程按 row_base 顺序写，所以输出字节与进程数无关（store_meta 里只有 ``exporter`` 块记进程数与时刻）。

身份核对：
- ``--manifest`` 经 load_manifest 重算 canonical identity，须等于 motion-ref store_meta 的 manifest_sha256，
  且 motion-ref 的 index 与清单逐 episode 身份相同；
- 每段 latent 的 ``.bin`` 与 ``.metadata.json`` 的 sha256 须等于 ``--checksums`` 成员清单里
  ``wan-latents/<名>`` 那一行（bucket ``checksums/sha256-source-files.txt``，pack_and_hash.py 按 arcname 逐成员写入；
  ``sha256-shards.txt`` 只有 tar 对象摘要，不是成员清单，传错即拒）；``.bin`` 还核 ``.bin.sha256`` 边车；
- 逐段 metadata 的段键、g、h5_file、raw_ep_idx、段类型、num_grid、窗起点与 motion-ref index 一致；
- 逐段 metadata 的 vae 块覆盖 motion-ref provenance.vae 的全部键且值相同（新表 vae 块沿用旧表）。

输出根事先存在即拒绝；写完经真实读 API ``MotionStore.rows`` 全表读回比对逐行摘要，通过才把 status 写成 verified。
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import motion_store as ms  # noqa: E402
from mme_vla_suite.datastore.manifest import load_manifest  # noqa: E402
from mme_vla_suite.shared import wan_pool768 as wp  # noqa: E402

LATENT_SHAPE = (9, 16, 32, 32)
LATENT_CHUNK_BYTES = int(np.prod(LATENT_SHAPE)) * 4      # 589,824，与 wan_common.CHUNK_BYTES 同值
LATENT_ARC_PREFIX = "wan-latents"                        # pack_and_hash.py build_jobs 里 wan-latents 的 arc_prefix
POOL_SOURCE_RELPATH = "src/mme_vla_suite/shared/wan_pool768.py"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_member_checksums(path: pathlib.Path) -> tuple[dict[str, str], str]:
    """读成员清单，返回 {arcname: sha256}（只取 wan-latents/ 前缀行）与这些行排序后拼接的 sha256。"""
    members: dict[str, str] = {}
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"{path} 行格式不是 '<sha256>  <路径>': {raw!r}")
        sha, name = parts[0], parts[1].strip()
        if not name.startswith(LATENT_ARC_PREFIX + "/"):
            continue
        if name in members and members[name] != sha:
            raise ValueError(f"{path} 成员 {name} 出现两次且摘要不同")
        members[name] = sha
        lines.append(f"{sha}  {name}")
    if not members:
        raise ValueError(f"{path} 里没有任何 {LATENT_ARC_PREFIX}/ 成员行；--checksums 须为 bucket "
                         f"checksums/sha256-source-files.txt（成员摘要），不是 sha256-shards.txt（tar 对象摘要）")
    digest = sha256_bytes(("\n".join(sorted(lines)) + "\n").encode("utf-8"))
    return members, digest


def segment_jobs(entries: list[ms.IndexEntry]) -> list[dict]:
    """按 motion-ref 行序契约列出段任务（num_grid=0 的段不出现）。"""
    jobs = []
    for e in entries:
        for seg in ms.SEGMENTS:
            s = getattr(e, seg)
            if s.num_grid == 0:
                continue
            jobs.append({"key": ms.segment_key(e.h5_file, e.raw_ep_idx, seg), "segment": seg, "g": e.g,
                         "h5_file": e.h5_file, "raw_ep_idx": e.raw_ep_idx, "row_base": s.row_base,
                         "num_grid": s.num_grid, "seg_len": s.seg_len,
                         "seg_start": 0 if seg == "demo" else e.exec_start_idx})
    return jobs


def check_segment_metadata(meta: dict, job: dict) -> int:
    """逐段 metadata 与 motion-ref index 的身份、网格、窗起点核对；返回本段补帧窗数。"""
    key = job["key"]
    want = {"segment": key, "g": job["g"], "h5_file": job["h5_file"], "raw_ep_idx": job["raw_ep_idx"],
            "segment_kind": job["segment"], "seg_start_global": job["seg_start"], "seg_len": job["seg_len"],
            "num_grid": job["num_grid"], "latent_shape": list(LATENT_SHAPE), "latent_dtype": "float32",
            "chunk_bytes": LATENT_CHUNK_BYTES, "grid_stride": ms.GRID_STRIDE, "window_frames": ms.WINDOW_FRAMES}
    for k, v in want.items():
        if meta.get(k) != v:
            raise ValueError(f"{key} metadata.{k}={meta.get(k)!r} != 期望 {v!r}")
    rows = meta.get("rows")
    if not isinstance(rows, list) or len(rows) != job["num_grid"]:
        raise ValueError(f"{key} metadata.rows 条数与 num_grid={job['num_grid']} 不符")
    padded = 0
    for m, r in enumerate(rows):
        off = ms.GRID_STRIDE * m
        if r.get("m") != m or r.get("seg_offset") != off or r.get("start_global_frame") != job["seg_start"] + off:
            raise ValueError(f"{key} m={m} 窗起点与网格不符: {r}")
        pad = r.get("pad_frames")
        if type(pad) is not int or pad < 0 or (pad > 0 and job["segment"] != "demo"):
            raise ValueError(f"{key} m={m} pad_frames={pad!r} 非法")
        padded += int(pad > 0)
    return padded


def pool_segment(job: dict) -> dict:
    """worker：读一段 latent → 核边车与 metadata → 逐窗池化，返回该段 (num_grid×768) f32 字节与摘要。"""
    root = pathlib.Path(job["latents_root"])
    key = job["key"]
    bin_path = root / f"{key}.bin"
    meta_path = root / f"{key}.metadata.json"
    data = bin_path.read_bytes()
    ng = job["num_grid"]
    if len(data) != ng * LATENT_CHUNK_BYTES:
        raise ValueError(f"{bin_path} 字节数 {len(data)} != {ng} × {LATENT_CHUNK_BYTES}")
    bin_sha = sha256_bytes(data)
    sidecar = (root / f"{key}.bin.sha256").read_text(encoding="utf-8").split()[0]
    if sidecar != bin_sha:
        raise ValueError(f"{bin_path} sha256 {bin_sha[:16]}… != 边车 {sidecar[:16]}…")
    meta_bytes = meta_path.read_bytes()
    meta = json.loads(meta_bytes.decode("utf-8"))
    if meta.get("sha256") != bin_sha or meta.get("bytes") != len(data):
        raise ValueError(f"{meta_path} 记录的 sha256 / bytes 与 .bin 不符")
    padded = check_segment_metadata(meta, job)
    lat = np.frombuffer(data, dtype="<f4").reshape((ng,) + LATENT_SHAPE)
    out = np.empty((ng, 768), dtype=np.float32)
    for m in range(ng):
        out[m] = wp.pool_raw_wan_latent_768(np.ascontiguousarray(lat[m]))[0]
    return {"key": key, "row_base": job["row_base"], "num_grid": ng, "segment": job["segment"],
            "rows": out.tobytes(), "bin_sha256": bin_sha, "metadata_sha256": sha256_bytes(meta_bytes),
            "vae": meta.get("vae"), "padded": padded}


def atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def meta_bytes(meta: dict) -> bytes:
    return json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latents", required=True, help="<lib>/wan-latents（解包后的运行时目录）")
    ap.add_argument("--motion-ref", required=True, help="<lib>/motion（上次 encoder motion 表，提供段序与 index）")
    ap.add_argument("--manifest", required=True, help="<lib>/meta/episode_manifest.json")
    ap.add_argument("--checksums", required=True, help="bucket checksums/sha256-source-files.txt（成员摘要清单）")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True, help="<lib>/motion-wan-pool768（必须事先不存在）")
    args = ap.parse_args()
    t0 = time.perf_counter()
    started_at = _now()
    if args.workers < 1:
        raise SystemExit("--workers 须 ≥ 1")

    out = pathlib.Path(args.out).absolute()
    if out.exists() or out.is_symlink():
        raise SystemExit(f"错误: 输出根已存在，拒绝写入: {out}")
    latents_root = pathlib.Path(args.latents).resolve()
    ref_root = pathlib.Path(args.motion_ref).resolve()
    manifest_path = pathlib.Path(args.manifest).resolve()
    checksums_path = pathlib.Path(args.checksums).resolve()

    manifest = load_manifest(manifest_path)                       # canonical identity 现场重算
    ref = ms.MotionMeta.load(ref_root)
    if ref.manifest_sha256 != manifest["sha256"]:
        raise SystemExit(f"错误: motion-ref 绑定清单 {ref.manifest_sha256[:16]}… != --manifest {manifest['sha256'][:16]}…")
    entries = list(ref.entries)
    ms.check_index_against_manifest(entries, manifest)
    if ref.spec != ms.LAYOUT_SPECS[ms.LAYOUT] or ref.raw["layout"] != ms.LAYOUT:
        raise SystemExit(f"错误: motion-ref 布局 {ref.raw['layout']} != {ms.LAYOUT}")
    index_bytes = (ref_root / ms.INDEX_RELPATH).read_bytes()
    totals = ms.index_totals(entries)
    ref_vae = ref.provenance.get("vae")
    if not isinstance(ref_vae, dict) or not ref_vae:
        raise SystemExit("错误: motion-ref provenance.vae 缺失")
    members, members_digest = load_member_checksums(checksums_path)
    latents_meta = latents_root / "metadata.json"
    if not latents_meta.is_file():
        raise SystemExit(f"错误: latent 根缺 metadata.json: {latents_meta}")

    jobs = segment_jobs(entries)
    for j in jobs:
        j["latents_root"] = str(latents_root)
    print(f"[export] episodes={len(entries)} segments={len(jobs)} rows={totals['rows']} "
          f"(demo {totals['demo_rows']} + exec {totals['exec_rows']}) workers={args.workers}", flush=True)

    out.mkdir(parents=True)
    (out / "meta").mkdir()
    table_path = out / ms.MOTION_TABLE_RELPATH
    tmp = table_path.with_name(table_path.name + ".tmp")
    sha = hashlib.sha256()
    digests = bytearray()
    offset = 0
    padded = 0
    seg_rows = {"demo": 0, "exec": 0}
    with open(tmp, "wb") as f, mp.Pool(args.workers) as pool:
        for n, res in enumerate(pool.imap(pool_segment, jobs, chunksize=1)):
            key = res["key"]
            if res["row_base"] * ms.MOTION_ROW_BYTES != offset:
                raise RuntimeError(f"{key} row_base={res['row_base']} 与写游标 {offset // ms.MOTION_ROW_BYTES} 不符")
            for name, got in ((f"{LATENT_ARC_PREFIX}/{key}.bin", res["bin_sha256"]),
                              (f"{LATENT_ARC_PREFIX}/{key}.metadata.json", res["metadata_sha256"])):
                if members.get(name) != got:
                    raise RuntimeError(f"{name} sha256 {got[:16]}… 与成员清单 {str(members.get(name))[:16]}… 不符")
            vae = res["vae"] or {}
            if any(vae.get(k) != v for k, v in ref_vae.items()):
                raise RuntimeError(f"{key} 的 latent vae 块与 motion-ref provenance.vae 不同")
            rows = res["rows"]
            f.write(rows)
            sha.update(rows)
            for m in range(res["num_grid"]):
                h = hashlib.blake2b(digest_size=ms.ROW_DIGEST_BYTES)
                h.update(rows[m * ms.MOTION_ROW_BYTES:(m + 1) * ms.MOTION_ROW_BYTES])
                digests += h.digest()
            offset += len(rows)
            padded += res["padded"]
            seg_rows[res["segment"]] += res["num_grid"]
            if (n + 1) % 500 == 0:
                print(f"[export] {n + 1}/{len(jobs)} 段 {offset // ms.MOTION_ROW_BYTES} 行", flush=True)
        f.flush()
        os.fsync(f.fileno())
    if offset != totals["rows"] * ms.MOTION_ROW_BYTES:
        raise RuntimeError(f"表字节数 {offset} != rows × {ms.MOTION_ROW_BYTES}")
    if seg_rows != {"demo": totals["demo_rows"], "exec": totals["exec_rows"]}:
        raise RuntimeError(f"分段行数 {seg_rows} 与 index totals 不符")
    os.replace(tmp, table_path)
    table_sha = sha.hexdigest()
    if ms.sha256_file(table_path) != table_sha:
        raise RuntimeError("表落盘后 sha256 与写入流不符")

    atomic_write_bytes(out / ms.INDEX_RELPATH, index_bytes)
    atomic_write_bytes(out / ms.ROW_DIGESTS_RELPATH, bytes(digests))
    spec = ms.LAYOUT_SPECS[ms.LAYOUT]
    old_prov = ref.provenance
    git_head = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False).stdout.strip()
    provenance = {
        "encoder_kind": wp.POOL768_ENCODER_KIND,
        "readout": wp.POOL768_READOUT,
        "encoder_ckpt": None,
        "parameter_count": wp.POOL768_PARAMETER_COUNT,
        "pool_source_relpath": POOL_SOURCE_RELPATH,
        "pool_source_sha256": wp.pool_source_sha256(),
        "pool_output_shape": list(wp.POOL768_SHAPE),
        "manifest_sha256": manifest["sha256"],
        "motion_index_sha256": ref.motion_index_sha256,
        "vae": ref_vae,
        "latent_root": {
            "path": str(latents_root),
            "metadata_sha256": ms.sha256_file(latents_meta),
            "member_checksums_path": str(checksums_path),
            "member_checksums_file_sha256": ms.sha256_file(checksums_path),
            "member_prefix": LATENT_ARC_PREFIX + "/",
            "member_count": len(members),
            "members_sha256": members_digest,
            "segments_checked": len(jobs),
        },
        "motion_ref": {
            "path": str(ref_root),
            "store_meta_sha256": ms.sha256_file(ref_root / ms.META_RELPATH),
            "motion_index_sha256": ref.motion_index_sha256,
            "table_sha256": ref.table_sha256,
        },
        "mj_repo_commit": old_prov.get("mj_repo_commit"),
        "source_pin": old_prov.get("source_pin"),
        "raw_dir": old_prov.get("raw_dir"),
        "input_manifest_sha256": old_prov.get("input_manifest_sha256"),
        "padded_windows": padded,
    }
    import importlib.metadata as md
    meta = {
        "schema": ms.META_SCHEMA, "layout": ms.LAYOUT, "status": "packed",
        **spec.fields(),
        "byte_order": ms.BYTE_ORDER, "array_order": ms.ARRAY_ORDER,
        "grid_stride": ms.GRID_STRIDE, "window_frames": ms.WINDOW_FRAMES, "grid_origin": ms.GRID_ORIGIN,
        "window_direction": ms.WINDOW_DIRECTION, "truncation_policy": ms.TRUNCATION_POLICY,
        "frame_size": ms.FRAME_SIZE,
        "tables": {ms.MOTION_KEY: {"row_shape": list(ms.MOTION_ROW_SHAPE), "dtype": "float32",
                                   "row_bytes": ms.MOTION_ROW_BYTES, "num_rows": totals["rows"],
                                   "relpath": ms.MOTION_TABLE_RELPATH, "byte_count": offset,
                                   "sha256": table_sha}},
        "num_rows": totals["rows"], "totals": totals,
        "row_order": ref.raw.get("row_order"),
        "manifest_sha256": manifest["sha256"],
        "manifest_path": str(manifest_path),
        "motion_index_relpath": ms.INDEX_RELPATH, "motion_index_sha256": ref.motion_index_sha256,
        "provenance": provenance,
        "verify": None,
        "row_digests": {"relpath": ms.ROW_DIGESTS_RELPATH, "algo": "blake2b-128", "covered_rows": totals["rows"],
                        "coverage": ms.ROW_DIGEST_COVERAGE, "byte_count": len(digests),
                        "sha256": sha256_bytes(bytes(digests))},
        # 只有本块随进程数与时刻变化；确定性比对剔除本块
        "exporter": {"script": "scripts/dataset/export_wan_pool768.py", "git_commit": git_head,
                     "python": sys.version.split()[0], "numpy": md.version("numpy"),
                     "workers": args.workers, "segments": len(jobs),
                     "started_at": started_at, "finished_at": None},
    }
    atomic_write_bytes(out / ms.META_RELPATH, meta_bytes(meta))

    # 真实读 API 全表读回：逐行 blake2b 与写入流摘要比对，过了才写 verified
    store = ms.MotionStore(out, manifest_path=str(manifest_path), verify_level="full")
    back = bytearray()
    for r in range(store.num_rows):
        h = hashlib.blake2b(digest_size=ms.ROW_DIGEST_BYTES)
        h.update(store.table[r].tobytes())
        back += h.digest()
    if not np.isfinite(store.table).all():
        raise RuntimeError("新表含非有限值")
    store.close()
    if bytes(back) != bytes(digests):
        raise RuntimeError("读回逐行摘要与写入流不符")
    conclusion = (f"EXPORT_POOL768=PASS rows={totals['rows']} demo={totals['demo_rows']} exec={totals['exec_rows']} "
                  f"padded={padded} segments={len(jobs)} table_sha256={table_sha}")
    meta["status"] = "verified"
    meta["verify"] = {"scanned": totals["rows"], "mismatches": 0, "method": "MotionStore 全表读回逐行 blake2b",
                      "conclusion": conclusion}
    meta["exporter"]["finished_at"] = _now()
    atomic_write_bytes(out / ms.META_RELPATH, meta_bytes(meta))
    ms.MotionMeta.load(out)
    print(f"[export] 完成，耗时 {time.perf_counter() - t0:.1f}s", flush=True)
    print(conclusion, flush=True)


if __name__ == "__main__":
    main()
