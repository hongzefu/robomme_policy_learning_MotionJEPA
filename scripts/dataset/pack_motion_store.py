#!/usr/bin/env python3
"""motion 离线表打包 / 校验工具（motion-memory-plan.md 第二部分一节 1.1）。

把 encoder 阶段落在 ``<lib>/motion-tokens/<段>.f32.bin`` 的逐段 token 按**行序契约**拼成一张表：

  <out>/motion_token.f32.bin        (rows, 768) f32 裸字节；行序 = 清单 canonical_order 逐 episode，
                                    每 episode 先 demo 后 exec，段内网格升序（motion_store.build_index_entries 派生）
  <out>/meta/motion_index.json      段基址表（唯一身份来源），store_meta 记其 sha256
  <out>/meta/store_meta.json        唯一契约，两阶段写：pack→"packed"、verify→"verified"
  <out>/meta/row_digests.blake2b.bin verify 产出的逐行 blake2b-128
  <out>/meta/pack_progress.jsonl    逐段落盘记录（续跑用）

子命令：
  pack     逐段读 token 文件（核 sidecar sha256 + 字节数）→ 顺序写表 + read-after-write → 写 index / meta 阶段 1
  verify   经真实读 API ``MotionStore.rows`` 全表逐行对源 token 文件 memcmp（零遗漏）+ row_digests + meta 回填
  report   打印 meta 摘要

事务协议照 ``pack_framesamp_store.py``（同一份 meta/pack.lock 语义：O_CREAT|O_EXCL；残锁 --resume 接管、
异 host --force-break-lock；verify 回填后才释放）。格式常量、index 公式、MotionMeta、读 API 一律
import 自 ``mme_vla_suite.datastore.motion_store``，绝不复制。本模块顶层不 import jax / torch。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import uuid

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import motion_store as ms  # noqa: E402
from mme_vla_suite.datastore.manifest import load_manifest  # noqa: E402

_WAN_DIR = _HERE / "wan"


# ══ 锁协议（与 pack_framesamp_store.acquire_lock 同语义）════════════════════════


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(store_root: pathlib.Path, *, resume: bool, force_break: bool, phase: str) -> dict:
    lock = store_root / ms.LOCK_RELPATH
    lock.parent.mkdir(parents=True, exist_ok=True)
    payload = {"build_uuid": uuid.uuid4().hex, "host": socket.gethostname(), "pid": os.getpid(),
               "phase": phase, "started_at": _now()}
    blob = (json.dumps(payload, ensure_ascii=False, indent=1) + "\n").encode()
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, blob)
            os.fsync(fd)
        finally:
            os.close(fd)
        return payload
    except FileExistsError:
        pass
    old_text = lock.read_text(encoding="utf-8", errors="replace")
    try:
        old = json.loads(old_text)
    except json.JSONDecodeError:
        old = {}
    same_host = old.get("host") == payload["host"]
    if same_host and old.get("pid") == os.getpid():
        pass
    elif same_host and isinstance(old.get("pid"), int) and _pid_alive(old["pid"]):
        raise RuntimeError(f"pack.lock 属同 host 存活进程 pid={old['pid']}，拒跑: {lock}")
    elif same_host:
        if not resume:
            raise RuntimeError(f"pack.lock 残锁（同 host、pid 不存活）: {lock}\n{old_text}"
                               f"确认无并发后用 --resume 显式接管")
    else:
        if not force_break:
            raise RuntimeError(f"pack.lock 属异 host（跨 host 无法判活），一律拒跑: {lock}\n{old_text}"
                               f"确认异 host 无进程后用 --force-break-lock 破锁")
        print(f"⚠ 异 host 锁全文:\n{old_text}", flush=True)
        if input("确认破异 host 锁并接管？输入 yes 继续: ").strip() != "yes":
            raise RuntimeError("未确认破锁，退出")
    tmp = lock.with_suffix(".lock.tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, lock)
    return payload


def release_lock(store_root: pathlib.Path) -> None:
    (store_root / ms.LOCK_RELPATH).unlink(missing_ok=True)


def atomic_write_bytes(path: pathlib.Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ══ 源 token 文件 ═══════════════════════════════════════════════════════════════


def _segment_key(entry: ms.IndexEntry, seg: str) -> str:
    return ms.segment_key(entry.h5_file, entry.raw_ep_idx, seg)


def read_segment_tokens(tokens_root: pathlib.Path, key: str, num_grid: int) -> tuple[bytes, dict]:
    """读一段 token 文件：核字节数 + sidecar sha256 + metadata 可解析；返回 (原始字节, metadata)。"""
    p = tokens_root / f"{key}.f32.bin"
    if not p.is_file():
        raise FileNotFoundError(f"缺段 token 文件: {p}")
    size = p.stat().st_size
    if size != num_grid * ms.MOTION_ROW_BYTES:
        raise ValueError(f"{p} 字节数 {size} != {num_grid} × {ms.MOTION_ROW_BYTES}")
    want = (tokens_root / f"{key}.f32.bin.sha256").read_text().split()[0]
    data = p.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise ValueError(f"{p} sha256 {got[:16]}… != sidecar {want[:16]}…")
    meta = json.loads((tokens_root / f"{key}.metadata.json").read_text(encoding="utf-8"))
    if int(meta.get("num_grid", -1)) != num_grid or meta.get("sha256") != want:
        raise ValueError(f"{p} metadata 与文件不符（num_grid / sha256）")
    arr = np.frombuffer(data, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{p} 含非有限值")
    return data, meta


def iter_rows_in_order(entries: list[ms.IndexEntry]):
    """按行序契约产出 (entry, seg, key, row_base, num_grid)。"""
    for e in entries:
        for seg in ms.SEGMENTS:
            s = getattr(e, seg)
            if s.num_grid == 0:
                continue
            yield e, seg, _segment_key(e, seg), s.row_base, s.num_grid


def _load_json(p: pathlib.Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def gather_provenance(latents_root: pathlib.Path, tokens_root: pathlib.Path, entries: list[ms.IndexEntry],
                      manifest: dict, index_sha: str) -> dict:
    """store_meta.provenance：SOURCE_PIN、VAE info、encoder info（各取自逐段 metadata 并断言跨段唯一）、
    逐 worker 指纹、encoder_state_sha256 清单、wan-latents/metadata.json sha。"""
    pin = _load_json(_WAN_DIR / "SOURCE_PIN.json")
    vae_infos: dict[str, dict] = {}
    enc_infos: dict[str, dict] = {}
    workers: dict[str, dict] = {}
    state_sha: dict | None = None
    same_keys_vae = ("vae_id", "vae_state_sha256", "vae_dtype", "latent_mode", "batch", "tf32", "amp",
                     "torch", "cuda", "cudnn", "diffusers", "module_sha256", "encoder_src_sha256", "flags",
                     "env", "gpu_name", "compute_cap", "driver")
    same_keys_enc = ("checkpoint", "checkpoint_sha256", "checkpoint_epoch", "arch", "state_key", "precision",
                     "amp", "tf32", "batch", "vae_id", "motion_dims", "torch", "cuda", "cudnn",
                     "module_sha256", "encoder_src_sha256", "env", "gpu_name", "compute_cap", "driver")
    for _e, _seg, key, _, _ng in iter_rows_in_order(entries):
        lm = _load_json(latents_root / f"{key}.metadata.json")
        tm = _load_json(tokens_root / f"{key}.metadata.json")
        v = {k: lm["vae"].get(k) for k in same_keys_vae}
        vae_infos[json.dumps(v, sort_keys=True, default=str)] = v
        enc = {k: tm["encoder"].get(k) for k in same_keys_enc}
        enc["flags"] = tm.get("encoder_flags")
        enc_infos[json.dumps(enc, sort_keys=True, default=str)] = enc
        for w in (lm["worker"], tm["worker"]):
            wid = f"{w.get('hostname')}:{w.get('gpu_uuid')}:{w.get('worker')}:{w.get('pid')}"
            workers[wid] = w
        if state_sha is None:
            state_sha = tm["encoder_state_sha256"]
        elif state_sha != tm["encoder_state_sha256"]:
            raise ValueError(f"{key} 的 encoder_state_sha256 清单与其他段不同（不是同一份权重）")
    if len(vae_infos) != 1:
        raise ValueError(f"跨段 VAE provenance 不唯一（{len(vae_infos)} 种）：{list(vae_infos.values())}")
    if len(enc_infos) != 1:
        raise ValueError(f"跨段 encoder provenance 不唯一（{len(enc_infos)} 种）：{list(enc_infos.values())}")
    vae = next(iter(vae_infos.values()))
    enc = next(iter(enc_infos.values()))
    if vae["module_sha256"] != pin["source_sha256"] or enc["module_sha256"] != pin["source_sha256"]:
        raise ValueError("逐段 provenance 的 module_sha256 != SOURCE_PIN.source_sha256")
    # 跨 worker 硬件/软件唯一性（1.3）：gpu_name / compute_cap / driver / torch / cudnn / git_commit / mj_commit
    uniq_keys = ("gpu_name", "compute_cap", "driver_version", "torch", "cudnn_version", "git_commit", "mj_commit")
    for k in uniq_keys:
        vals = {str(w.get(k)) for w in workers.values()}
        if len(vals) != 1:
            raise ValueError(f"跨 worker {k} 不唯一: {sorted(vals)}")
    ckpt_name = enc["checkpoint"]
    epoch = int(ckpt_name.replace("checkpoint_epoch_", "").replace(".pt", ""))
    if epoch != int(enc["checkpoint_epoch"]):
        raise ValueError(f"checkpoint 名解析出的 epoch {epoch} != ckpt 内记录 {enc['checkpoint_epoch']}")
    latents_meta = latents_root / "metadata.json"
    return {
        "manifest_sha256": manifest["sha256"],
        "motion_index_sha256": index_sha,
        "mj_repo_commit": pin["mj_repo_commit"],
        "source_pin": pin,
        "vae": vae,
        "encoder": {"run_name": "wan-v8-filter10-72ep-a", "checkpoint_name": ckpt_name, "epoch": epoch,
                    "state_key": "encoder", "batch": 1, **enc},
        "encoder_state_sha256": state_sha,
        "workers": list(workers.values()),
        "worker_unique_keys": list(uniq_keys),
        "latents_metadata_sha256": ms.sha256_file(latents_meta) if latents_meta.is_file() else None,
        "latents_root": str(latents_root.resolve()),
        "tokens_root": str(tokens_root.resolve()),
    }


# ══ 子命令 ════════════════════════════════════════════════════════════════════


def cmd_pack(args) -> None:
    t_all = time.perf_counter()
    manifest = load_manifest(args.manifest)
    entries = ms.build_index_entries(manifest)
    totals = ms.index_totals(entries)
    tokens_root = pathlib.Path(args.tokens).resolve()
    latents_root = pathlib.Path(args.latents).resolve()
    store_root = pathlib.Path(args.out).resolve()
    if store_root.is_symlink():
        raise RuntimeError(f"输出根是符号链接，拒绝写入: {store_root}")
    store_root.mkdir(parents=True, exist_ok=True)
    (store_root / "meta").mkdir(exist_ok=True)
    meta_path = store_root / ms.META_RELPATH
    lock = acquire_lock(store_root, resume=args.resume, force_break=args.force_break_lock, phase="pack")
    try:
        if meta_path.exists() and not args.resume:
            raise RuntimeError(f"store_meta.json 已存在: {meta_path}；重打包须显式 --resume")
        pin = _load_json(_WAN_DIR / "SOURCE_PIN.json")
        index = ms.index_payload(manifest, entries, mj_repo_commit=pin["mj_repo_commit"])
        index_bytes = (json.dumps(index, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        index_sha = hashlib.sha256(index_bytes).hexdigest()
        print(f"[pack] episodes={len(entries)} rows={totals['rows']} (exec {totals['exec_rows']} + "
              f"demo {totals['demo_rows']}) index_sha256={index_sha[:16]}…", flush=True)

        table_path = store_root / ms.MOTION_TABLE_RELPATH
        tmp = table_path.with_name(table_path.name + ".tmp")
        progress_path = store_root / ms.PROGRESS_RELPATH
        sha = hashlib.sha256()
        offset = 0
        n_seg = 0
        with open(tmp, "w+b") as f, open(progress_path, "w", encoding="utf-8") as pf:
            fd = f.fileno()
            for e, seg, key, row_base, ng in iter_rows_in_order(entries):
                if row_base * ms.MOTION_ROW_BYTES != offset:
                    raise RuntimeError(f"{key} row_base={row_base} 与写游标 {offset // ms.MOTION_ROW_BYTES} 不符")
                data, _ = read_segment_tokens(tokens_root, key, ng)
                f.write(data)
                f.flush()
                back = os.pread(fd, len(data), offset)        # read-after-write
                if back != data:
                    raise RuntimeError(f"写侧校验失败: {key} 读回不符")
                sha.update(back)
                offset += len(data)
                n_seg += 1
                pf.write(json.dumps({"segment": key, "g": e.g, "seg": seg, "row_base": row_base,
                                     "num_grid": ng, "at": _now()}) + "\n")
            os.fsync(fd)
        if offset != totals["rows"] * ms.MOTION_ROW_BYTES:
            raise RuntimeError(f"表字节数 {offset} != rows × {ms.MOTION_ROW_BYTES}")
        os.replace(tmp, table_path)
        table_sha = sha.hexdigest()
        if ms.sha256_file(table_path) != table_sha:
            raise RuntimeError("表落盘后 sha256 与写入流不符")

        atomic_write_bytes(store_root / ms.INDEX_RELPATH, index_bytes)
        prov = gather_provenance(latents_root, tokens_root, entries, manifest, index_sha)
        git_head = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=False).stdout.strip()
        import importlib.metadata as md
        meta = {
            "schema": ms.META_SCHEMA, "layout": ms.LAYOUT, "status": "packed",
            "byte_order": ms.BYTE_ORDER, "array_order": ms.ARRAY_ORDER,
            "grid_stride": ms.GRID_STRIDE, "window_frames": ms.WINDOW_FRAMES, "grid_origin": ms.GRID_ORIGIN,
            "window_direction": ms.WINDOW_DIRECTION, "truncation_policy": ms.TRUNCATION_POLICY,
            "frame_size": ms.FRAME_SIZE,
            "tables": {ms.MOTION_KEY: {"row_shape": list(ms.MOTION_ROW_SHAPE), "dtype": "float32",
                                       "row_bytes": ms.MOTION_ROW_BYTES, "num_rows": totals["rows"],
                                       "relpath": ms.MOTION_TABLE_RELPATH, "byte_count": offset,
                                       "sha256": table_sha}},
            "num_rows": totals["rows"], "totals": totals,
            "row_order": index["row_order"],
            "manifest_sha256": manifest["sha256"],
            "manifest_path": str(pathlib.Path(args.manifest).resolve()),
            "motion_index_relpath": ms.INDEX_RELPATH, "motion_index_sha256": index_sha,
            "provenance": prov,
            "packer": {"build_uuid": lock["build_uuid"], "host": lock["host"], "git_commit": git_head,
                       "python": sys.version.split()[0], "numpy": md.version("numpy"),
                       "segments": n_seg, "started_at": lock["started_at"], "finished_at": _now()},
            "verify": None, "row_digests": None,
        }
        atomic_write_bytes(meta_path, json.dumps(meta, ensure_ascii=False, indent=1).encode())
        ms.MotionMeta.load(store_root)     # 自检：契约能读回（含 index sha 现场重算）
        print(f"[pack] 表 {offset:,} B / {totals['rows']} 行 / {n_seg} 段, sha256={table_sha[:16]}…; "
              f"meta 阶段1 落盘（status=packed）; 耗时 {time.perf_counter() - t_all:.1f}s", flush=True)
        print("PACK_MOTION_DONE=1 （pack.lock 保留，verify 回填后才释放）", flush=True)
    except BaseException:
        print("[pack] 失败：pack.lock 与半成品保留", flush=True)
        raise


def cmd_verify(args) -> None:
    t_all = time.perf_counter()
    store_root = pathlib.Path(args.store).resolve()
    meta = ms.MotionMeta.load(store_root)
    manifest_path = args.manifest or meta.manifest_path
    tokens_root = pathlib.Path(args.tokens or meta.provenance["tokens_root"]).resolve()
    manifest = load_manifest(manifest_path)
    if manifest["sha256"] != meta.manifest_sha256:
        raise RuntimeError("verify: 清单指纹与 meta 不符")
    entries = ms.build_index_entries(manifest)
    if [dataclass_tuple(e) for e in entries] != [dataclass_tuple(e) for e in meta.entries]:
        raise RuntimeError("verify: 现场按清单重算的 index 与 meta 内 motion_index 不同")
    started_at = _now()
    had_lock = (store_root / ms.LOCK_RELPATH).exists()
    acquire_lock(store_root, resume=(args.resume if had_lock else False),
                 force_break=args.force_break_lock, phase="verify")

    store = ms.MotionStore(store_root, meta=meta, manifest_path=manifest_path, verify_level="full")
    scanned = 0
    mismatches: list[tuple[int, str]] = []
    digests = bytearray()
    for _e, _seg, key, row_base, ng in iter_rows_in_order(entries):
        src, _ = read_segment_tokens(tokens_root, key, ng)
        rows = store.rows(np.arange(row_base, row_base + ng))          # 真实读 API
        for m in range(ng):
            st = rows[m].tobytes()
            if st != src[m * ms.MOTION_ROW_BYTES:(m + 1) * ms.MOTION_ROW_BYTES]:
                mismatches.append((row_base + m, key))
            h = hashlib.blake2b(digest_size=ms.ROW_DIGEST_BYTES)
            h.update(st)
            digests += h.digest()
            scanned += 1
    if scanned != meta.num_rows:
        raise RuntimeError(f"verify 扫描行数 {scanned} != num_rows {meta.num_rows}（有遗漏）")
    if mismatches:
        print(f"VERIFY_MOTION=FAIL scanned={scanned} mismatches={len(mismatches)}")
        for row, key in mismatches[:20]:
            print(f"  失配 row={row} segment={key}")
        print("⚠ FAIL：不回填 meta、保留 pack.lock")
        raise SystemExit(1)
    atomic_write_bytes(store_root / ms.ROW_DIGESTS_RELPATH, bytes(digests))
    raw = dict(meta.raw)
    raw["status"] = "verified"
    raw["verify"] = {"scanned": scanned, "mismatches": 0, "started_at": started_at, "finished_at": _now(),
                     "conclusion": f"VERIFY_MOTION=PASS scanned={scanned} mismatches=0"}
    raw["row_digests"] = {"relpath": ms.ROW_DIGESTS_RELPATH, "algo": "blake2b-128", "covered_rows": scanned,
                          "coverage": ms.ROW_DIGEST_COVERAGE, "byte_count": len(digests),
                          "sha256": hashlib.sha256(bytes(digests)).hexdigest()}
    atomic_write_bytes(store_root / ms.META_RELPATH, json.dumps(raw, ensure_ascii=False, indent=1).encode())
    ms.MotionMeta.load(store_root)
    release_lock(store_root)
    print(f"[verify] meta 回填（status=verified）+ row_digests 落盘, 耗时 {time.perf_counter() - t_all:.1f}s",
          flush=True)
    print(f"VERIFY_MOTION=PASS scanned={scanned} mismatches=0")


def dataclass_tuple(e: ms.IndexEntry) -> tuple:
    return (e.g, e.h5_file, e.raw_ep_idx, e.num_timesteps, e.exec_start_idx,
            (e.demo.row_base, e.demo.num_grid, e.demo.num_chunks, e.demo.seg_len),
            (e.exec.row_base, e.exec.num_grid, e.exec.num_chunks, e.exec.seg_len))


def cmd_report(args) -> None:
    meta = ms.MotionMeta.load(pathlib.Path(args.store))
    r = meta.raw
    print(f"layout={r['layout']} status={r['status']} num_rows={meta.num_rows} totals={r['totals']}")
    print(f"manifest={meta.manifest_path}\n  sha256={meta.manifest_sha256}")
    print(f"motion_index_sha256={meta.motion_index_sha256}")
    print(f"encoder={r['provenance']['encoder'].get('checkpoint_sha256')} vae={r['provenance']['vae'].get('vae_state_sha256')}")
    print(f"packer={r.get('packer')}\nverify={r.get('verify')}\nrow_digests={r.get('row_digests')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--manifest", required=True)
    p.add_argument("--tokens", required=True, help="<lib>/motion-tokens")
    p.add_argument("--latents", required=True, help="<lib>/wan-latents（只读 metadata 取 provenance）")
    p.add_argument("--out", required=True, help="<lib>/motion")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-break-lock", action="store_true")
    p.set_defaults(func=cmd_pack)
    p = sub.add_parser("verify")
    p.add_argument("--store", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--tokens", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-break-lock", action="store_true")
    p.set_defaults(func=cmd_verify)
    p = sub.add_parser("report")
    p.add_argument("--store", required=True)
    p.set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
