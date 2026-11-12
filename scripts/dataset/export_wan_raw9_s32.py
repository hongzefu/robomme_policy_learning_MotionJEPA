#!/usr/bin/env python3
"""Wan raw latent 直读 motion 表导出：stride 32、每行 (9,16,32,32) f32 原字节（1112-4task-modul-wanraw9-s32-50k-plan.html
§2、第二部分 1.1 第 2/3 行、1.2）。

源是解包后的 ``<latents>/<段>.bin``（stride-16 网格，num_grid16 × 589,824 B，extract_wan.py 写出）。段内取 stride-16 网格的
偶数 m（新行号 m' = m/2，即段内起点 0,32,64,…，demo / exec 各自从段起点起算），新表第 m' 行 = 源 ``.bin`` 第 m 个
chunk 的 589,824 原字节，不经任何浮点运算、不做 z-score / 裁剪 / dtype 转换：

  <out>/motion_token.f32.bin           (rows, 9,16,32,32) f32；行序 = 清单 canonical 序逐 episode，先 demo 后 exec
  <out>/meta/motion_index.json         由 ms.build_index_entries(manifest, LAYOUT_SPECS[LAYOUT_RAW9]) 新算（grid_stride=32）
  <out>/meta/row_digests.blake2b.bin   逐行 blake2b-128
  <out>/meta/store_meta.json           布局 motion-raw9-grid32-demopad17-v1；provenance 记 encoder_kind=wan_raw9 / readout /
                                       encoder_ckpt=null / parameter_count=0 / raw9_source_sha256 / latent 根成员摘要 /
                                       manifest canonical identity / budget_max

并行：段级任务经 ``multiprocessing.Pool(--workers).imap``（有序、chunksize=1）分发；worker 读源段、核身份，把偶数 chunk
的原字节以 ``os.pwrite`` 直接写到输出表的 ``row_base × 589,824`` 偏移处，只回传逐行摘要（不经管道搬 21.5 GB）。
每行写入位置只由 index 决定，所以输出字节与进程数无关（store_meta 里只有 ``exporter`` 块记进程数与时刻）。

身份核对（同 export_wan_pool768.py）：
- ``--manifest`` 经 load_manifest 重算 canonical identity，须等于 ``--motion-ref`` store_meta 的 manifest_sha256，
  且 motion-ref 的 index（stride-16 源网格）与清单逐 episode 身份相同；``--motion-ref`` 只用于给出源网格与沿用字段
  （vae 块、input_manifest_sha256、mj_repo_commit、source_pin、raw_dir），新表 index 不照抄它；
- 每段 latent 的 ``.bin`` 与 ``.metadata.json`` 的 sha256 须等于 ``--checksums`` 成员清单里 ``wan-latents/<名>`` 那一行
  （bucket ``checksums/sha256-source-files.txt``；``sha256-shards.txt`` 只有 tar 对象摘要，传错即拒）；``.bin`` 还核边车；
- 逐段 metadata 的段键、g、h5_file、raw_ep_idx、段类型、num_grid、窗起点与源网格一致；偶数 m 的
  ``start_global_frame == seg_start + 32m'``；逐段断言 ``num_grid32 == len(range(0, num_chunks, 32)) == ⌈num_grid16/2⌉``；
- 逐段 metadata 的 vae 块覆盖 motion-ref provenance.vae 的全部键且值相同。

输出根事先存在即拒绝；写完经真实读 API ``MotionStore``（verify_level=full，只读 memmap）全表读回比对逐行摘要、核全有限，
通过才把 status 写成 verified。末行 ``EXPORT_RAW9=PASS rows=… demo=… exec=… padded=… budget_max=… table_sha256=…``，
budget_max = 新 index 上逐 episode ``max_visible_count`` 的最大值。
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
from mme_vla_suite.shared import wan_raw9 as wr  # noqa: E402

LATENT_SHAPE = wr.RAW9_INPUT_SHAPE
LATENT_CHUNK_BYTES = wr.RAW9_ROW_BYTES                   # 589,824，与 wan_common.CHUNK_BYTES 同值
LATENT_ARC_PREFIX = "wan-latents"                        # pack_and_hash.py build_jobs 里 wan-latents 的 arc_prefix
RAW9_SOURCE_RELPATH = "src/mme_vla_suite/shared/wan_raw9.py"
SOURCE_GRID_STRIDE = ms.GRID_STRIDE                      # 源 latent 的网格步长 16
NEW_STRIDE = wr.RAW9_STRIDE                              # 32
STEP = NEW_STRIDE // SOURCE_GRID_STRIDE                  # 取每第 2 个源窗（偶数 m）
if STEP * SOURCE_GRID_STRIDE != NEW_STRIDE:
    raise RuntimeError("新步长须为源步长的整数倍")


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


def segment_jobs(src_entries: list[ms.IndexEntry], new_entries: list[ms.IndexEntry]) -> list[dict]:
    """按新 index 行序列出段任务（新 num_grid=0 的段不出现）；逐段断言新旧网格的偶数 m 对应关系。"""
    if len(src_entries) != len(new_entries):
        raise ValueError("源 index 与新 index 的 episode 数不同")
    jobs = []
    for a, b in zip(src_entries, new_entries):
        if (a.g, a.h5_file, a.raw_ep_idx, a.num_timesteps, a.exec_start_idx) != \
                (b.g, b.h5_file, b.raw_ep_idx, b.num_timesteps, b.exec_start_idx):
            raise ValueError(f"源 index 与新 index 的 episode 身份不同 g={a.g}")
        for seg in ms.SEGMENTS:
            s16, s32 = getattr(a, seg), getattr(b, seg)
            if s16.num_chunks != s32.num_chunks or s16.seg_len != s32.seg_len:
                raise ValueError(f"g={a.g} {seg} 源/新 num_chunks 或 seg_len 不同")
            ng32 = len(range(0, s32.num_chunks, NEW_STRIDE))
            if s32.num_grid != ng32 or ng32 != len(range(0, s16.num_grid, STEP)):
                raise ValueError(f"g={a.g} {seg} num_grid32={s32.num_grid} 与 ⌈num_grid16/2⌉ / 公式 {ng32} 不符")
            if s32.num_grid == 0:
                continue
            jobs.append({"key": ms.segment_key(a.h5_file, a.raw_ep_idx, seg), "segment": seg, "g": a.g,
                         "h5_file": a.h5_file, "raw_ep_idx": a.raw_ep_idx, "row_base": s32.row_base,
                         "num_grid16": s16.num_grid, "num_grid32": s32.num_grid, "seg_len": s16.seg_len,
                         "seg_start": 0 if seg == "demo" else a.exec_start_idx})
    return jobs


def check_segment_metadata(meta: dict, job: dict) -> int:
    """逐段 metadata 与源网格（stride 16）的身份、网格、窗起点核对；返回新表（偶数 m）里的补帧窗数。"""
    key = job["key"]
    want = {"segment": key, "g": job["g"], "h5_file": job["h5_file"], "raw_ep_idx": job["raw_ep_idx"],
            "segment_kind": job["segment"], "seg_start_global": job["seg_start"], "seg_len": job["seg_len"],
            "num_grid": job["num_grid16"], "latent_shape": list(LATENT_SHAPE), "latent_dtype": "float32",
            "chunk_bytes": LATENT_CHUNK_BYTES, "grid_stride": SOURCE_GRID_STRIDE, "window_frames": ms.WINDOW_FRAMES}
    for k, v in want.items():
        if meta.get(k) != v:
            raise ValueError(f"{key} metadata.{k}={meta.get(k)!r} != 期望 {v!r}")
    rows = meta.get("rows")
    if not isinstance(rows, list) or len(rows) != job["num_grid16"]:
        raise ValueError(f"{key} metadata.rows 条数与 num_grid={job['num_grid16']} 不符")
    padded = 0
    for m, r in enumerate(rows):
        off = SOURCE_GRID_STRIDE * m
        if r.get("m") != m or r.get("seg_offset") != off or r.get("start_global_frame") != job["seg_start"] + off:
            raise ValueError(f"{key} m={m} 窗起点与网格不符: {r}")
        pad = r.get("pad_frames")
        if type(pad) is not int or pad < 0 or (pad > 0 and job["segment"] != "demo"):
            raise ValueError(f"{key} m={m} pad_frames={pad!r} 非法")
        if m % STEP == 0:
            if r["start_global_frame"] != job["seg_start"] + NEW_STRIDE * (m // STEP):
                raise ValueError(f"{key} m={m} 新行 m'={m // STEP} 起点与 stride {NEW_STRIDE} 网格不符")
            padded += int(pad > 0)
    return padded


def export_segment(job: dict) -> dict:
    """worker：读一段源 latent → 核边车与 metadata → 偶数 chunk 原字节 pwrite 到输出表，返回逐行摘要。"""
    root = pathlib.Path(job["latents_root"])
    key = job["key"]
    bin_path = root / f"{key}.bin"
    meta_path = root / f"{key}.metadata.json"
    data = bin_path.read_bytes()
    ng16 = job["num_grid16"]
    if len(data) != ng16 * LATENT_CHUNK_BYTES:
        raise ValueError(f"{bin_path} 字节数 {len(data)} != {ng16} × {LATENT_CHUNK_BYTES}")
    bin_sha = sha256_bytes(data)
    sidecar = (root / f"{key}.bin.sha256").read_text(encoding="utf-8").split()[0]
    if sidecar != bin_sha:
        raise ValueError(f"{bin_path} sha256 {bin_sha[:16]}… != 边车 {sidecar[:16]}…")
    meta_bytes = meta_path.read_bytes()
    meta = json.loads(meta_bytes.decode("utf-8"))
    if meta.get("sha256") != bin_sha or meta.get("bytes") != len(data):
        raise ValueError(f"{meta_path} 记录的 sha256 / bytes 与 .bin 不符")
    padded = check_segment_metadata(meta, job)
    digests = bytearray()
    view = memoryview(data)
    fd = os.open(job["table_tmp"], os.O_WRONLY)
    try:
        for mp_, m in enumerate(range(0, ng16, STEP)):
            chunk = view[m * LATENT_CHUNK_BYTES:(m + 1) * LATENT_CHUNK_BYTES]
            pos = (job["row_base"] + mp_) * LATENT_CHUNK_BYTES
            done = 0
            while done < LATENT_CHUNK_BYTES:
                done += os.pwrite(fd, chunk[done:], pos + done)
            h = hashlib.blake2b(digest_size=ms.ROW_DIGEST_BYTES)
            h.update(chunk)
            digests += h.digest()
        if mp_ + 1 != job["num_grid32"]:
            raise RuntimeError(f"{key} 写出 {mp_ + 1} 行 != num_grid32 {job['num_grid32']}")
    finally:
        os.close(fd)
    return {"key": key, "row_base": job["row_base"], "num_grid": job["num_grid32"], "segment": job["segment"],
            "digests": bytes(digests), "bin_sha256": bin_sha, "metadata_sha256": sha256_bytes(meta_bytes),
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
    ap.add_argument("--manifest", required=True, help="<lib>/meta/episode_manifest.json")
    ap.add_argument("--checksums", required=True, help="bucket checksums/sha256-source-files.txt（成员摘要清单）")
    ap.add_argument("--motion-ref", required=True,
                    help="<lib>/motion-wan-pool768（stride-16 源网格表，给出源行序与沿用的 vae / input_manifest 等字段）")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True, help="<lib>/motion-wan-raw9-s32（必须事先不存在）")
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
    src_entries = list(ref.entries)
    ms.check_index_against_manifest(src_entries, manifest)
    if ref.spec != ms.LAYOUT_SPECS[ms.LAYOUT] or ref.raw["layout"] != ms.LAYOUT:
        raise SystemExit(f"错误: motion-ref 布局 {ref.raw['layout']} != 源 latent 布局 {ms.LAYOUT}")
    ref_vae = ref.provenance.get("vae")
    if not isinstance(ref_vae, dict) or not ref_vae:
        raise SystemExit("错误: motion-ref provenance.vae 缺失")
    members, members_digest = load_member_checksums(checksums_path)
    latents_meta = latents_root / "metadata.json"
    if not latents_meta.is_file():
        raise SystemExit(f"错误: latent 根缺 metadata.json: {latents_meta}")

    spec = ms.LAYOUT_SPECS[ms.LAYOUT_RAW9]
    new_entries = ms.build_index_entries(manifest, spec)
    ref_index = json.loads((ref_root / ms.INDEX_RELPATH).read_text(encoding="utf-8"))
    index = ms.index_payload(manifest, new_entries, spec=spec, layout=ms.LAYOUT_RAW9,
                             mj_repo_commit=ref_index.get("mj_repo_commit"))
    index_bytes = (json.dumps(index, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    index_sha = sha256_bytes(index_bytes)
    if ms.parse_index(json.loads(index_bytes), ctx="新 motion_index.json") != new_entries:
        raise RuntimeError("新 index 序列化后解析不回原条目")
    totals = ms.index_totals(new_entries)
    budget_max = max((ms.max_visible_count(e) for e in new_entries), default=0)

    jobs = segment_jobs(src_entries, new_entries)
    out.mkdir(parents=True)
    (out / "meta").mkdir()
    table_path = out / ms.MOTION_TABLE_RELPATH
    tmp = table_path.with_name(table_path.name + ".tmp")
    total_bytes = totals["rows"] * LATENT_CHUNK_BYTES
    with open(tmp, "wb") as f:
        f.truncate(total_bytes)
    for j in jobs:
        j["latents_root"] = str(latents_root)
        j["table_tmp"] = str(tmp)
    print(f"[export] episodes={len(new_entries)} segments={len(jobs)} rows={totals['rows']} "
          f"(demo {totals['demo_rows']} + exec {totals['exec_rows']}) budget_max={budget_max} "
          f"workers={args.workers}", flush=True)

    digests = bytearray()
    cursor = 0
    padded = 0
    seg_rows = {"demo": 0, "exec": 0}
    with mp.Pool(args.workers) as pool:
        for n, res in enumerate(pool.imap(export_segment, jobs, chunksize=1)):
            key = res["key"]
            if res["row_base"] != cursor:
                raise RuntimeError(f"{key} row_base={res['row_base']} 与行游标 {cursor} 不符")
            for name, got in ((f"{LATENT_ARC_PREFIX}/{key}.bin", res["bin_sha256"]),
                              (f"{LATENT_ARC_PREFIX}/{key}.metadata.json", res["metadata_sha256"])):
                if members.get(name) != got:
                    raise RuntimeError(f"{name} sha256 {got[:16]}… 与成员清单 {str(members.get(name))[:16]}… 不符")
            vae = res["vae"] or {}
            if any(vae.get(k) != v for k, v in ref_vae.items()):
                raise RuntimeError(f"{key} 的 latent vae 块与 motion-ref provenance.vae 不同")
            if len(res["digests"]) != res["num_grid"] * ms.ROW_DIGEST_BYTES:
                raise RuntimeError(f"{key} 回传摘要条数不符")
            digests += res["digests"]
            cursor += res["num_grid"]
            padded += res["padded"]
            seg_rows[res["segment"]] += res["num_grid"]
            if (n + 1) % 500 == 0:
                print(f"[export] {n + 1}/{len(jobs)} 段 {cursor} 行", flush=True)
    if cursor != totals["rows"]:
        raise RuntimeError(f"写出行数 {cursor} != index totals.rows {totals['rows']}")
    if seg_rows != {"demo": totals["demo_rows"], "exec": totals["exec_rows"]}:
        raise RuntimeError(f"分段行数 {seg_rows} 与 index totals 不符")
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    if tmp.stat().st_size != total_bytes:
        raise RuntimeError(f"表字节数 {tmp.stat().st_size} != rows × {LATENT_CHUNK_BYTES}")
    os.replace(tmp, table_path)
    table_sha = ms.sha256_file(table_path)

    atomic_write_bytes(out / ms.INDEX_RELPATH, index_bytes)
    atomic_write_bytes(out / ms.ROW_DIGESTS_RELPATH, bytes(digests))
    old_prov = ref.provenance
    git_head = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False).stdout.strip()
    provenance = {
        "encoder_kind": wr.RAW9_ENCODER_KIND,
        "readout": wr.RAW9_READOUT,
        "encoder_ckpt": None,
        "parameter_count": wr.RAW9_PARAMETER_COUNT,
        "raw9_source_relpath": RAW9_SOURCE_RELPATH,
        "raw9_source_sha256": wr.source_sha256(),
        "row_shape": list(LATENT_SHAPE),
        "source_grid_stride": SOURCE_GRID_STRIDE,
        "source_m_rule": "even",
        "manifest_sha256": manifest["sha256"],
        "motion_index_sha256": index_sha,
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
            "layout": ref.raw["layout"],
            "store_meta_sha256": ms.sha256_file(ref_root / ms.META_RELPATH),
            "motion_index_sha256": ref.motion_index_sha256,
            "table_sha256": ref.table_sha256,
        },
        "mj_repo_commit": old_prov.get("mj_repo_commit"),
        "source_pin": old_prov.get("source_pin"),
        "raw_dir": old_prov.get("raw_dir"),
        "input_manifest_sha256": old_prov.get("input_manifest_sha256"),
        "padded_windows": padded,
        "budget_max": budget_max,
    }
    import importlib.metadata as md
    meta = {
        "schema": ms.META_SCHEMA, "layout": ms.LAYOUT_RAW9, "status": "packed",
        **spec.fields(),
        "byte_order": ms.BYTE_ORDER, "array_order": ms.ARRAY_ORDER,
        "grid_stride": spec.grid_stride, "window_frames": ms.WINDOW_FRAMES, "grid_origin": ms.GRID_ORIGIN,
        "window_direction": ms.WINDOW_DIRECTION, "truncation_policy": ms.TRUNCATION_POLICY,
        "frame_size": ms.FRAME_SIZE,
        "tables": {ms.MOTION_KEY: {"row_shape": list(spec.row_shape), "dtype": "float32",
                                   "row_bytes": spec.row_bytes, "num_rows": totals["rows"],
                                   "relpath": ms.MOTION_TABLE_RELPATH, "byte_count": total_bytes,
                                   "sha256": table_sha}},
        "num_rows": totals["rows"], "totals": totals,
        "row_order": index["row_order"],
        "manifest_sha256": manifest["sha256"],
        "manifest_path": str(manifest_path),
        "motion_index_relpath": ms.INDEX_RELPATH, "motion_index_sha256": index_sha,
        "provenance": provenance,
        "verify": None,
        "row_digests": {"relpath": ms.ROW_DIGESTS_RELPATH, "algo": "blake2b-128", "covered_rows": totals["rows"],
                        "coverage": ms.ROW_DIGEST_COVERAGE, "byte_count": len(digests),
                        "sha256": sha256_bytes(bytes(digests))},
        # 只有本块随进程数与时刻变化；确定性比对剔除本块
        "exporter": {"script": "scripts/dataset/export_wan_raw9_s32.py", "git_commit": git_head,
                     "python": sys.version.split()[0], "numpy": md.version("numpy"),
                     "workers": args.workers, "segments": len(jobs),
                     "started_at": started_at, "finished_at": None},
    }
    atomic_write_bytes(out / ms.META_RELPATH, meta_bytes(meta))

    # 真实读 API（只读 memmap）全表读回：逐行 blake2b 与 worker 回传摘要比对、逐行核有限，过了才写 verified
    store = ms.MotionStore(out, manifest_path=str(manifest_path), verify_level="full")
    if not isinstance(store.table, np.memmap):
        raise RuntimeError("raw9 表未以 memmap 打开")
    back = bytearray()
    nonfinite = 0
    for r in range(store.num_rows):
        row = store.table[r]
        h = hashlib.blake2b(digest_size=ms.ROW_DIGEST_BYTES)
        h.update(row.tobytes())
        back += h.digest()
        nonfinite += int(not np.isfinite(row).all())
    store.close()
    if bytes(back) != bytes(digests):
        raise RuntimeError("读回逐行摘要与写入摘要不符")
    if nonfinite:
        raise RuntimeError(f"新表含非有限值（{nonfinite} 行）")
    conclusion = (f"EXPORT_RAW9=PASS rows={totals['rows']} demo={totals['demo_rows']} exec={totals['exec_rows']} "
                  f"padded={padded} budget_max={budget_max} table_sha256={table_sha}")
    meta["status"] = "verified"
    meta["verify"] = {"scanned": totals["rows"], "mismatches": 0,
                      "method": "MotionStore（只读 memmap）全表读回逐行 blake2b + 有限性", "conclusion": conclusion}
    meta["exporter"]["finished_at"] = _now()
    atomic_write_bytes(out / ms.META_RELPATH, meta_bytes(meta))
    ms.MotionMeta.load(out)
    print(f"[export] 完成，耗时 {time.perf_counter() - t0:.1f}s", flush=True)
    print(conclusion, flush=True)


if __name__ == "__main__":
    main()
