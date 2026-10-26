#!/usr/bin/env python3
"""motion 离线表打包 / 校验工具（0901-motion-memory-plan.md 第二部分一节 1.1）。

把 encoder 阶段落在 ``<lib>/motion-tokens/<段>.f32.bin`` 的逐段 token 按**行序契约**拼成一张表：

  <out>/motion_token.f32.bin        (rows, 8, 256) f32 裸字节（8,192 B/行）；行序 = 清单 canonical_order 逐 episode，
                                    每 episode 先 demo 后 exec，段内网格升序（motion_store.build_index_entries 派生）
  <out>/meta/motion_index.json      段基址表（唯一身份来源），store_meta 记其 sha256
  <out>/meta/store_meta.json        唯一契约，两阶段写：pack→"packed"、verify→"verified"
  <out>/meta/row_digests.blake2b.bin verify 产出的逐行 blake2b-128
  <out>/meta/pack_progress.jsonl    逐段落盘记录（续跑用）

子命令：
  pack     逐段读 token 文件（核 sidecar sha256 + 字节数）→ 顺序写表 + read-after-write → 写 index / meta 阶段 1
  verify   经真实读 API ``MotionStore.rows`` 全表逐行对源 token 文件 memcmp（零遗漏）+ row_digests + meta 回填

历史 VAE 兼容（S5-B）：``pack --restore-receipt <S5-A receipt> --expected-ckpt-sha256 <final>`` 时 VAE 与新 encoder
分两组校验（见 ``load_vae_compat`` / ``check_vae_group``），写表前预检；此模式下 ``verify`` 还要求
``compare_wan.py tokens`` 已对本张 packed 表写下 ``meta/encoder_bitexact.json`` PASS，否则拒绝进入 verified。
  report   打印 meta 摘要

事务协议照 ``pack_framesamp_store.py``（同一份 meta/pack.lock 语义：O_CREAT|O_EXCL；残锁 --resume 接管、
异 host --force-break-lock；verify 回填后才释放）。格式常量、index 公式、MotionMeta、读 API 一律
import 自 ``mme_vla_suite.datastore.motion_store``，绝不复制。本模块顶层不 import jax / torch。
"""

from __future__ import annotations

import argparse
import dataclasses
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


def read_segment_tokens(tokens_root: pathlib.Path, key: str, num_grid: int, *,
                        spec: ms.LayoutSpec | None = None) -> tuple[bytes, dict]:
    """读一段 token 文件：核字节数 + sidecar sha256 + metadata 可解析；返回 (原始字节, metadata)。"""
    spec = ms.LAYOUT_SPECS[ms.LAYOUT] if spec is None else spec
    p = tokens_root / f"{key}.f32.bin"
    if not p.is_file():
        raise FileNotFoundError(f"缺段 token 文件: {p}")
    size = p.stat().st_size
    if size != num_grid * spec.row_bytes:
        raise ValueError(f"{p} 字节数 {size} != {num_grid} × {spec.row_bytes}")
    want = (tokens_root / f"{key}.f32.bin.sha256").read_text().split()[0]
    data = p.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise ValueError(f"{p} sha256 {got[:16]}… != sidecar {want[:16]}…")
    meta = json.loads((tokens_root / f"{key}.metadata.json").read_text(encoding="utf-8"))
    if int(meta.get("num_grid", -1)) != num_grid or meta.get("sha256") != want:
        raise ValueError(f"{p} metadata 与文件不符（num_grid / sha256）")
    if spec.row_shape == ms.MOTION_ROW_SHAPE:
        contract = {"layout": ms.LAYOUT, "row_shape": list(spec.row_shape), "dtype": "float32",
                    "token_dim": int(np.prod(spec.row_shape)), "token_bytes": spec.row_bytes,
                    "bytes": size}
        if any(meta.get(k) != v for k, v in contract.items()):
            raise ValueError(f"{p} 原始 motion 合同不符，期望 {contract}")
    if spec == ms.LAYOUT_SPECS[ms.STAGE2_LAYOUT]:
        contract = {"schema": "stage2-memory-export-v1", "layout": ms.STAGE2_LAYOUT,
                    "row_shape": [1, 1024], "dtype": "float32", "token_dim": 1024,
                    "token_bytes": 4096, "bytes": size, "num_grid": num_grid, "grid_stride": 32, "window_frames": 33}
        if any(type(meta.get(k)) is not type(v) or meta[k] != v for k, v in contract.items()):
            raise ValueError(f"{p} stage2 行合同不符")
        if (any(type(v) is not int for v in meta["row_shape"])
                or not isinstance(meta.get("source_grid"), list)
                or any(type(v) is not int for v in meta["source_grid"])
                or meta["source_grid"] != list(range(0, num_grid * 2, 2))):
            raise ValueError(f"{p} source_grid 不是连续偶数格")
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


# ══ 历史 VAE 兼容（S5-B，对抗检查 D3/D4）════════════════════════════════════════
#
# 历史 VLA latent 由 S5-A 的 restore_mixed8_vla_latents.py 恢复并封口（restore_receipt.json）。这些 latent 的
# VAE 记录是当时的适配件/源提交，与本次 encoder 的 SOURCE_PIN 必然不同。因此 provenance 分两组：
#   VAE 组     ：组内跨段/跨 worker 严格唯一，且逐字段等于 receipt 记录的**历史值**（精确匹配，不通用放宽）；
#   encoder 组 ：组内严格唯一，且等于当前 SOURCE_PIN、显式 ckpt SHA、冻结 config 与同一份 state 摘要。
# 不传 receipt 时行为与旧版相同（两组共用 SOURCE_PIN、全体 worker 唯一）。

RECEIPT_SCHEMA = "mixed8-vla-latent-restore/v1"
VAE_IDENTITY_KEYS = ("vae_id", "vae_state_sha256", "vae_dtype", "latent_mode", "batch", "tf32", "amp",
                     "torch", "cuda", "cudnn", "diffusers", "module_sha256", "encoder_src_sha256")
SAME_KEYS_VAE = ("vae_id", "vae_state_sha256", "vae_dtype", "latent_mode", "batch", "tf32", "amp",
                 "torch", "cuda", "cudnn", "diffusers", "cublas_pkg", "cudnn_pkg", "module_sha256",
                 "encoder_src_sha256", "flags", "env", "gpu_name", "compute_cap", "sm_count", "driver")
SAME_KEYS_ENC = ("checkpoint", "checkpoint_sha256", "checkpoint_epoch", "config_sha256", "arch", "state_key", "precision",
                 "amp", "tf32", "batch", "vae_id", "motion_dims", "torch", "cuda", "cudnn", "diffusers",
                 "cublas_pkg", "cudnn_pkg", "module_sha256", "encoder_src_sha256", "env", "gpu_name",
                 "compute_cap", "sm_count", "driver")
WORKER_UNIQUE_KEYS = ("gpu_name", "compute_cap", "driver_version", "torch", "cudnn_version", "git_commit", "mj_commit")
CHUNK_BYTES = 9 * 16 * 32 * 32 * np.dtype(np.float32).itemsize
BITEXACT_RELPATH = "meta/encoder_bitexact.json"     # compare_wan.py tokens 在 packed 阶段写；兼容模式 verify 必需


@dataclasses.dataclass(frozen=True)
class HistoricalPins:
    """历史缓存的钉死期望值（计划第二部分 0.1 / S5-A）。CLI 不提供覆盖，测试可传合成值。"""
    total_metadata_sha256: str = "bda20fd101f57271a9b18db295679256cbf12d1d5c52e1fab80b936f7d9f1061"
    vae_state_sha256: str = "9980d252230c265cc2869466a74f85f5ee45b01ea9521bbb31159f90b75fe6d0"
    episode_manifest_sha256: str = "c6a063dfbfc6da404dd8f1d3ddc38e9ba6e25abca81417fe742872401e7f97df"
    input_manifest_sha256: str = "d7dd651edb8cb9652fa647a73a2d0d3c4be64d6876e1d7a186a5e8c75b02aafb"
    manifest_canonical_sha256: str = "d6a3ddb2b579b5cf50cd66fbefedd36478766e2b4cc07833a7efae29769742ad"


def load_vae_compat(receipt_path: pathlib.Path, latents_root: pathlib.Path, *,
                    manifest_path: pathlib.Path | None = None, oracle_report: pathlib.Path | None = None,
                    pins: HistoricalPins = HistoricalPins()) -> dict:
    """核验 S5-A restore_receipt.json 并返回历史 VAE 兼容记录（不读 Torch）。

    只认 receipt 的完整封口：schema/状态/无新 encoder 身份、恢复目录 == latents_root 的上级、总 metadata 文件
    SHA 等于钉死值且等于 receipt、全量文件清单 SHA 现场重算一致、VAE 历史身份字段齐全且自洽、父库 manifest
    文件字节 SHA 一致。任一不符即 ValueError。返回值不含可伪造的“兼容开关”，只含经核实的历史身份。
    """
    receipt_path = pathlib.Path(receipt_path).resolve()
    latents_root = pathlib.Path(latents_root).resolve()
    rc = _load_json(receipt_path)
    if rc.get("schema_version") != RECEIPT_SCHEMA or rc.get("status") != "PASS":
        raise ValueError("restore receipt 的 schema_version 或 status 不符")
    if rc.get("new_encoder_identity") is not None:
        raise ValueError("restore receipt 不得携带新 encoder 身份")
    out_dir = pathlib.Path(rc.get("out_dir", "/nonexistent")).resolve()
    layout = rc.get("layout") or {}
    if out_dir != receipt_path.parent or (out_dir / str(layout.get("latents", "?"))).resolve() != latents_root:
        raise ValueError("restore receipt 指向的恢复目录与 --latents 不同")
    total_meta = latents_root / "metadata.json"
    total_sha = ms.sha256_file(total_meta)
    lat = rc.get("latents") or {}
    if not (total_sha == lat.get("total_metadata_sha256") == pins.total_metadata_sha256):
        raise ValueError("恢复的总 metadata SHA 与 receipt/钉死值不符")
    files_list = out_dir / str(lat.get("files_list", "restore_files.sha256"))
    if not files_list.is_file() or ms.sha256_file(files_list) != lat.get("files_list_sha256"):
        raise ValueError("restore 文件清单缺失或 SHA 与 receipt 不符")
    listed = {}
    for line in files_list.read_text(encoding="utf-8").splitlines():
        sha, _, rel = line.partition("  ")
        listed[rel] = sha
    if listed.get(f"{layout.get('latents')}/metadata.json") != total_sha:
        raise ValueError("restore 文件清单中的总 metadata SHA 不符")
    vae = rc.get("vae") or {}
    ident = vae.get("identity") or {}
    if set(ident) != set(VAE_IDENTITY_KEYS) or any(ident.get(k) in (None, "") for k in VAE_IDENTITY_KEYS):
        raise ValueError("receipt 的历史 VAE 身份字段不全")
    if not (ident["vae_state_sha256"] == vae.get("vae_state_sha256") == pins.vae_state_sha256):
        raise ValueError("receipt 的 VAE state SHA 与钉死值不符")
    if (vae.get("historical_module_sha256") != ident["module_sha256"]
            or vae.get("historical_encoder_src_sha256") != ident["encoder_src_sha256"]):
        raise ValueError("receipt 的历史源码 SHA 与身份字段不自洽")
    gits, mjs = vae.get("worker_git_commits"), vae.get("worker_mj_commits")
    if not (isinstance(gits, list) and len(gits) == 1 and isinstance(mjs, list) and len(mjs) == 1):
        raise ValueError("receipt 的历史 VAE worker 提交必须各自唯一")
    parent = rc.get("parent") or {}
    if (parent.get("episode_manifest_sha256") != pins.episode_manifest_sha256
            or parent.get("input_manifest_sha256") != pins.input_manifest_sha256
            or parent.get("manifest_canonical_sha256") != pins.manifest_canonical_sha256):
        raise ValueError("receipt 的父库 manifest SHA 与钉死值不符")
    if manifest_path is not None:
        manifest_path = pathlib.Path(manifest_path)
        if ms.sha256_file(manifest_path) != pins.episode_manifest_sha256:
            raise ValueError("episode_manifest.json 不是父库原字节")
        if ms.sha256_file(manifest_path.parent / "input_manifest.json") != pins.input_manifest_sha256:
            raise ValueError("input_manifest.json 不是父库原字节")
    if oracle_report is not None:
        want = (rc.get("oracle") or {}).get("vae_report_sha256")
        if want is None or ms.sha256_file(oracle_report) != want:
            raise ValueError("历史 VAE oracle 报告与 receipt 记录不符")
    return {"mode": "historical-restore-receipt", "receipt_path": str(receipt_path),
            "receipt_sha256": ms.sha256_file(receipt_path), "schema_version": RECEIPT_SCHEMA,
            "total_metadata_sha256": total_sha, "identity": dict(ident),
            "historical_module_sha256": ident["module_sha256"],
            "historical_encoder_src_sha256": ident["encoder_src_sha256"],
            "worker_git_commits": list(gits), "worker_mj_commits": list(mjs),
            "oracle_coverage": (rc.get("oracle") or {}).get("coverage")}


def _worker_id(w: dict) -> str:
    return f"{w.get('hostname')}:{w.get('gpu_uuid')}:{w.get('worker')}:{w.get('pid')}"


def _unique_workers(workers: dict, what: str) -> None:
    for k in WORKER_UNIQUE_KEYS:
        vals = {str(w.get(k)) for w in workers.values()}
        if len(vals) != 1:
            raise ValueError(f"{what}跨 worker {k} 不唯一: {sorted(vals)}")


def check_vae_group(latents_root: pathlib.Path, keys: list[str], compat: dict) -> tuple[dict, list[dict]]:
    """历史 VAE 组：总 metadata（钉死 SHA）逐段 sha/bytes ↔ 段 metadata ↔ sidecar；组内身份严格唯一并
    逐字段等于 receipt 历史值；worker 组内硬件/提交唯一且等于 receipt。不读 bin（bin 由 gather 单遍核验）。"""
    if ms.sha256_file(latents_root / "metadata.json") != compat["total_metadata_sha256"]:
        raise ValueError("总 metadata 在兼容核验后被改动")
    total = _load_json(latents_root / "metadata.json")
    vae_infos: dict[str, dict] = {}
    workers: dict[str, dict] = {}
    for key in keys:
        seg = total["segments"].get(key)
        lm = _load_json(latents_root / f"{key}.metadata.json")
        side = (latents_root / f"{key}.bin.sha256").read_text(encoding="utf-8").split()[0]
        if seg is None or not (seg["sha256"] == lm.get("sha256") == side) or seg["bytes"] != lm.get("bytes"):
            raise ValueError(f"{key} 段 metadata/sidecar 与固定总 metadata 不符")
        v = {k: lm["vae"].get(k) for k in SAME_KEYS_VAE}
        vae_infos[json.dumps(v, sort_keys=True, default=str)] = v
        workers[_worker_id(lm["worker"])] = lm["worker"]
    if len(vae_infos) != 1:
        raise ValueError(f"跨段 VAE provenance 不唯一（{len(vae_infos)} 种）")
    vae = next(iter(vae_infos.values()))
    diff = [k for k in VAE_IDENTITY_KEYS if vae.get(k) != compat["identity"][k]]
    if diff:
        raise ValueError(f"VAE 组身份与 restore receipt 历史值不符: {diff}")
    _unique_workers(workers, "VAE 组")
    w0 = next(iter(workers.values()))
    if [w0.get("git_commit")] != compat["worker_git_commits"] or [w0.get("mj_commit")] != compat["worker_mj_commits"]:
        raise ValueError("VAE 组 worker 的 git/mj 提交与 restore receipt 不符")
    return vae, list(workers.values())


def gather_provenance(latents_root: pathlib.Path, tokens_root: pathlib.Path, entries: list[ms.IndexEntry],
                      manifest: dict, index_sha: str, *, encoder_run_dir: pathlib.Path,
                      raw_dir: pathlib.Path, input_manifest_path: pathlib.Path,
                      vae_compat: dict | None = None, expected_ckpt_sha256: str | None = None) -> dict:
    """store_meta.provenance：SOURCE_PIN、VAE info、encoder info（各取自逐段 metadata 并断言跨段唯一）、
    逐 worker 指纹、encoder_state_sha256 清单、wan-latents/metadata.json sha。

    ``vae_compat``（load_vae_compat 的返回）给出时 VAE 与 encoder 分组校验，见上方说明；此时
    ``expected_ckpt_sha256`` 必填。cmd_pack 在写表之前调用（预检），不等写完整张表才发现不兼容。"""
    pin = _load_json(_WAN_DIR / "SOURCE_PIN.json")
    if vae_compat is not None and not expected_ckpt_sha256:
        raise ValueError("历史 VAE 兼容模式必须显式给出最终 encoder ckpt SHA")
    keys = [key for _e, _seg, key, _, _ng in iter_rows_in_order(entries)]
    total_segments = None
    vae_hist = workers_vae_hist = None
    if vae_compat is not None:
        vae_hist, workers_vae_hist = check_vae_group(latents_root, keys, vae_compat)
        total_segments = _load_json(latents_root / "metadata.json")["segments"]
    vae_infos: dict[str, dict] = {}
    enc_infos: dict[str, dict] = {}
    workers: dict[str, dict] = {}
    workers_enc: dict[str, dict] = {}
    state_sha: dict | None = None
    for _e, _seg, key, _, _ng in iter_rows_in_order(entries):
        lm = _load_json(latents_root / f"{key}.metadata.json")
        tm = _load_json(tokens_root / f"{key}.metadata.json")
        latent_path = latents_root / f"{key}.bin"
        blocks = tm.get("input_block_sha256")
        if not isinstance(blocks, list) or len(blocks) != _ng or latent_path.stat().st_size != _ng * CHUNK_BYTES:
            raise ValueError(f"{key} latent 块数或每块指纹数量不符")
        whole = hashlib.sha256()                   # 单遍：整文件 SHA 与逐块指纹同时核
        with latent_path.open("rb") as stream:
            for expected in blocks:
                buf = stream.read(CHUNK_BYTES)
                whole.update(buf)
                if hashlib.sha256(buf).hexdigest() != expected:
                    raise ValueError(f"{key} latent 逐块指纹不符")
        if tm.get("input_latent_sha256") != whole.hexdigest():
            raise ValueError(f"{key} 的 motion 未绑定当前 latent 字节")
        if total_segments is not None and total_segments[key]["sha256"] != whole.hexdigest():
            raise ValueError(f"{key} latent 字节与固定总 metadata 记录不符")
        actual_run = pathlib.Path(tm["encoder"]["run_dir"]).resolve()
        if actual_run != encoder_run_dir.resolve():
            raise ValueError(f"{key} 的 encoder 目录与 --encoder-run-dir 不同")
        v = {k: lm["vae"].get(k) for k in SAME_KEYS_VAE}
        vae_infos[json.dumps(v, sort_keys=True, default=str)] = v
        enc = {k: tm["encoder"].get(k) for k in SAME_KEYS_ENC}
        enc["flags"] = tm.get("encoder_flags")
        enc_infos[json.dumps(enc, sort_keys=True, default=str)] = enc
        workers[_worker_id(lm["worker"])] = lm["worker"]
        workers[_worker_id(tm["worker"])] = tm["worker"]
        workers_enc[_worker_id(tm["worker"])] = tm["worker"]
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
    if vae_compat is None:
        if vae["module_sha256"] != pin["source_sha256"] or enc["module_sha256"] != pin["source_sha256"]:
            raise ValueError("逐段 provenance 的 module_sha256 != SOURCE_PIN.source_sha256")
    else:
        if vae != vae_hist:
            raise ValueError("VAE 组预检与逐段复核结果不同")
        if enc["module_sha256"] != pin["source_sha256"]:
            raise ValueError("encoder 组 module_sha256 != 当前 SOURCE_PIN.source_sha256")
        if enc["checkpoint_sha256"] != expected_ckpt_sha256:
            raise ValueError("encoder 组 checkpoint SHA 与显式最终 ckpt SHA 不同")
    if ms.sha256_file(_WAN_DIR / "wan_motion_infer.py") != pin["source_sha256"]:
        raise ValueError("当前推理源码与 SOURCE_PIN.source_sha256 不符")
    if enc["arch"] != "wan-latent-v10-st" or enc["motion_dims"] != list(ms.MOTION_ROW_SHAPE) or enc["state_key"] != "encoder":
        raise ValueError("encoder 架构、原始槽形状或 EMA state_key 与 v10 raw8 合同不符")
    src_groups = (vae, enc) if vae_compat is None else (enc,)
    if pin.get("encoder_source_sha256") and any(
            info["encoder_src_sha256"] != pin["encoder_source_sha256"] for info in src_groups):
        raise ValueError("VAE/encoder 记录的 encoder 源码与 SOURCE_PIN 不符")
    # 跨 worker 硬件/软件唯一性（1.3）：gpu_name / compute_cap / driver / torch / cudnn / git_commit / mj_commit
    uniq_keys = WORKER_UNIQUE_KEYS
    group = workers if vae_compat is None else workers_enc
    _unique_workers(group, "" if vae_compat is None else "encoder 组")
    if any(w.get("mj_commit") != pin["mj_repo_commit"] for w in group.values()):
        raise ValueError("worker 的 MotionJEPA 源提交与 SOURCE_PIN 不符")
    ckpt_name = enc["checkpoint"]
    checkpoint_path = encoder_run_dir / ckpt_name
    if not checkpoint_path.is_file() or ms.sha256_file(checkpoint_path) != enc["checkpoint_sha256"]:
        raise ValueError("encoder 目录中的 checkpoint 与逐段 provenance 不符")
    if ms.sha256_file(encoder_run_dir / "config.yaml") != enc["config_sha256"]:
        raise ValueError("encoder 冻结 config 与逐段 provenance 不符")
    epoch = int(ckpt_name.replace("checkpoint_epoch_", "").replace(".pt", ""))
    if epoch != int(enc["checkpoint_epoch"]):
        raise ValueError(f"checkpoint 名解析出的 epoch {epoch} != ckpt 内记录 {enc['checkpoint_epoch']}")
    latents_meta = latents_root / "metadata.json"
    prov = {
        "manifest_sha256": manifest["sha256"],
        "motion_index_sha256": index_sha,
        "mj_repo_commit": pin["mj_repo_commit"],
        "source_pin": pin,
        "vae": vae,
        "encoder": {"run_name": encoder_run_dir.resolve().name, "checkpoint_name": ckpt_name, "epoch": epoch,
                    "state_key": "encoder", "batch": 1, **enc},
        "encoder_state_sha256": state_sha,
        "workers": list(workers.values()),
        "worker_unique_keys": list(uniq_keys),
        "latents_metadata_sha256": ms.sha256_file(latents_meta) if latents_meta.is_file() else None,
        "latents_root": str(latents_root.resolve()),
        "tokens_root": str(tokens_root.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "input_manifest_sha256": ms.sha256_file(input_manifest_path),
    }
    if vae_compat is not None:
        # 兼容记录旁挂在新 store；历史 latent metadata 原样不动，历史日期/源码 SHA 不改写成本次值。
        prov["vae_compat"] = vae_compat
        prov["workers_vae"] = workers_vae_hist
        prov["workers_encoder"] = list(workers_enc.values())
        prov["worker_unique_scope"] = "per_group"
        prov["encoder_state_digest"] = hashlib.sha256(
            json.dumps(state_sha, sort_keys=True).encode()).hexdigest()
    return prov


# ══ 子命令 ════════════════════════════════════════════════════════════════════


def validate_stage2_segments(latents_root, tokens_root, entries):
    """精确段集合、偶数格映射及补帧契约；合法零窗段仍由索引保留。"""
    expected = {key for _, _, key, _, _ in iter_rows_in_order(entries)}
    for suffix in (".f32.bin", ".f32.bin.sha256", ".metadata.json"):
        actual = {p.name[:-len(suffix)] for p in tokens_root.glob(f"*{suffix}")}
        if actual != expected:
            raise ValueError(f"stage2 段集合不符 {suffix}: 缺 {sorted(expected - actual)[:5]}，多 {sorted(actual - expected)[:5]}")
    total = _load_json(latents_root / "metadata.json")
    if set(total["segments"]) != expected:
        raise ValueError("stage2 恢复 latent 段集合与 manifest 非零段集合不同")
    summary = dict(real_plus_pad_ne_33=0, exec_padded=0, pad_not_last_demo=0,
                   pad_source_mismatch=0, demo_tail_padded=0, demo_tail_dropped=0,
                   legal_zero_segments=sum(getattr(e, s).seg_len > 0 and getattr(e, s).num_grid == 0
                                           for e in entries for s in ms.SEGMENTS))
    for entry, segment, key, _, ng in iter_rows_in_order(entries):
        lm = _load_json(latents_root / f"{key}.metadata.json")
        _, tm = read_segment_tokens(tokens_root, key, ng, spec=entry.spec)
        info = getattr(entry, segment)
        start = 0 if segment == "demo" else entry.exec_start_idx
        legacy_count = ms.seg_num_grid(info.seg_len, entry.spec.min_real(segment))
        fixed = {"segment": key, "segment_kind": segment, "seg_start_global": start, "seg_len": info.seg_len}
        if (any(tm.get(k) != v or lm.get(k) != v for k, v in fixed.items())
                or lm.get("num_grid") != legacy_count or lm.get("grid_stride") != 16
                or lm.get("window_frames") != 33 or len(lm.get("rows", [])) != legacy_count
                or len(tm.get("rows", [])) != ng):
            raise ValueError(f"{key} 段元数据与索引合同不符")
        if ms.canonical_json_bytes(lm["rows"]) != ms.canonical_json_bytes(total["segments"][key].get("rows")):
            raise ValueError(f"{key} 原始行身份与 receipt 固定的总 metadata 不符")
        for j, source_grid in enumerate(tm["source_grid"]):
            row = tm["rows"][j]
            real = min(33, info.seg_len - 32 * j)
            pad = 33 - real
            want = {"m": source_grid, "seg_offset": 32 * j, "start_global_frame": start + 32 * j,
                    "real_frames": real, "pad_frames": pad,
                    "pad_source_frame": start + info.seg_len - 1 if pad else None,
                    "input_shape": [33, 256, 256, 3], "input_dtype": "uint8"}
            if (ms.canonical_json_bytes(row) != ms.canonical_json_bytes(lm["rows"][source_grid])
                    or any(type(row.get(k)) is not type(v) or row[k] != v for k, v in want.items())):
                raise ValueError(f"{key} 第 {j} 行不等于原偶数格，或 padding/起点身份不符")
            ms.require_sha256(row.get("input_frames_sha256"), f"{key} input_frames_sha256")
            if pad and (segment != "demo" or j != ng - 1 or real < 17):
                raise ValueError(f"{key} 非 demo 末窗补帧或真实帧不足17")
            summary["demo_tail_padded"] += int(pad > 0)
        if segment == "demo" and 0 < info.seg_len - ng * 32 <= 16:
            summary["demo_tail_dropped"] += 1
    return summary


def _stage2_validation(validation_root, *, manifest, latent_sha, checkpoint_sha, source_pin, numerics, online_vae):
    """读取已封口的 G0/G2/G3，要求输入摘要与关键数值指标完整绑定。"""
    samples_path = validation_root / "windows.json"
    samples = _load_json(samples_path)
    counts = samples.get("counts", {})
    windows = samples.get("windows", [])
    if (samples.get("schema") != "stage2-validation-inputs-v1" or samples.get("gate") != "G0"
            or samples.get("status") != "PASS" or samples.get("seed") != 0
            or samples.get("manifest_sha256") != manifest["sha256"]
            or samples.get("latent_metadata_sha256") != latent_sha
            or counts.get("windows") != 64 or len(windows) != 64 or counts.get("tasks") != 16
            or counts.get("sha_mismatches") != 0 or counts.get("demo_tail", 0) < 16):
        raise ValueError("stage2 G0 证据不完整或绑定了不同输入")
    tasks, identities, tail_count = {}, set(), 0
    for window in windows:
        required = ("key", "task", "episode", "segment_kind", "source_grid", "latent_path", "latent_block_sha256",
                    "frame_ids", "real_frames", "pad_frames", "pad_source_frame", "input_frames_sha256",
                    "sources", "source_frame_indices")
        if any(k not in window for k in required):
            raise ValueError("stage2 G0 窗口必填字段不全")
        tasks[window["task"]] = tasks.get(window["task"], 0) + 1
        if (len(window.get("frame_ids", [])) != 33 or len(window.get("source_frame_indices", [])) != 33
                or not window.get("sources") or type(window.get("source_grid")) is not int
                or window["source_grid"] < 0 or window["source_grid"] % 2
                or type(window["episode"]) is not int or window["episode"] < 0
                or window["segment_kind"] not in ms.SEGMENTS
                or window["key"] != f"{window['task']}_ep{window['episode']}_{window['segment_kind']}"
                or not isinstance(window["latent_path"], str) or not window["latent_path"]):
            raise ValueError("stage2 G0 窗口来源不完整")
        ident = (window["key"], window["source_grid"])
        if ident in identities:
            raise ValueError("stage2 G0 样本身份重复")
        identities.add(ident)
        real, pad = window["real_frames"], window["pad_frames"]
        frames = window["frame_ids"]
        if (type(real) is not int or type(pad) is not int or real + pad != 33 or pad < 0
                or real < (17 if window["segment_kind"] == "demo" else 33)
                or any(type(frame) is not int or frame < 0 for frame in frames)):
            raise ValueError("stage2 G0 真实帧与补帧数量不符")
        if (frames != list(range(frames[0], frames[0] + real)) + [frames[real - 1]] * pad
                or window["pad_source_frame"] != (frames[real - 1] if pad else None)):
            raise ValueError("stage2 G0 帧索引或补帧来源不符")
        tail_count += int(pad > 0)
        kind = "demo_h5" if window["segment_kind"] == "demo" else "exec_pkl"
        for source in window["sources"]:
            if (not isinstance(source, dict) or source.get("kind") != kind
                    or not isinstance(source.get("path"), str) or not source["path"]):
                raise ValueError("stage2 G0 源文件身份不全")
            ms.require_sha256(source.get("sha256"), "G0.sources.sha256")
        for index in window["source_frame_indices"]:
            if (not isinstance(index, dict) or type(index.get("source")) is not int
                    or not 0 <= index["source"] < len(window["sources"])
                    or type(index.get("index")) is not int or index["index"] < 0
                    or (kind == "exec_pkl" and index["index"] != 0)):
                raise ValueError("stage2 G0 源帧索引不符")
        for key in ("input_frames_sha256", "latent_block_sha256"):
            ms.require_sha256(window.get(key), f"G0.{key}")
    if len(tasks) != 16 or set(tasks.values()) != {4} or tail_count != counts["demo_tail"]:
        raise ValueError("stage2 G0 必须每任务4窗")
    samples_sha = ms.sha256_file(samples_path)
    evidence = {"G0": {"path": str(samples_path.resolve()), "sha256": samples_sha}}
    for gate in ("G2", "G3"):
        path = validation_root / f"{gate}.json"
        record = _load_json(path)
        if (record.get("schema") != "stage2-validation-gate-v1" or record.get("gate") != gate
                or record.get("status") != "PASS" or record.get("checkpoint_sha256") != checkpoint_sha
                or record.get("samples_sha256") != samples_sha):
            raise ValueError(f"stage2 {gate} 证据缺失、失败或 SHA 绑定不符")
        if (ms.canonical_json_bytes(record.get("source_pin")) != ms.canonical_json_bytes(source_pin)
                or ms.canonical_json_bytes(record.get("numerics")) != ms.canonical_json_bytes(numerics)):
            raise ValueError(f"stage2 {gate} 与当前实际源码或数值环境不同源")
        metrics = record.get("metrics", {})
        if gate == "G2":
            if (ms.canonical_json_bytes(record.get("online_vae")) != ms.canonical_json_bytes(online_vae)
                    or metrics.get("windows") != 64
                    or metrics.get("demo_tail") != counts["demo_tail"]
                    or not isinstance(metrics.get("worst_mse"), (int, float))
                    or not 0 <= metrics["worst_mse"] <= 1e-6
                    or not isinstance(metrics.get("worst_max"), (int, float))
                    or not 0 <= metrics["worst_max"] <= 0.01):
                raise ValueError("stage2 G2 数值指标不完整或超过阈值")
        else:
            want = dict(windows=32, seed=0, max_abs=0, geometry_all_true=1, leaves=65,
                        state_key="model", exact=1, live_negative_rejected=1)
            if any(metrics.get(k) != v for k, v in want.items()):
                raise ValueError("stage2 G3 EMA 严格核对或独立32窗证据不足")
        evidence[gate] = {"path": str(path.resolve()), "sha256": ms.sha256_file(path)}
    return evidence


def gather_stage2_provenance(latents_root, tokens_root, entries, manifest, index_sha, *,
                            encoder_run_dir, raw_dir, input_manifest_path, vae_compat,
                            expected_ckpt_sha256, validation_root, source_pin_path, manifest_file_sha256):
    """stage2 专属来源链：整 latent 与偶数块分别核 SHA，保留全部 encoder 身份。"""
    if vae_compat is None:
        raise ValueError("stage2 必须使用已封口的历史 VAE receipt")
    ms.require_sha256(expected_ckpt_sha256, "stage2 expected_ckpt_sha256")
    validate_stage2_segments(latents_root, tokens_root, entries)
    keys = [key for _, _, key, _, _ in iter_rows_in_order(entries)]
    vae, vae_workers = check_vae_group(latents_root, keys, vae_compat)
    total = _load_json(latents_root / "metadata.json")
    pin = _load_json(source_pin_path)
    if ms.sha256_file(_WAN_DIR / "wan_motion_infer.py") != pin.get("vae_adapter_sha256"):
        raise ValueError("stage2 SOURCE_PIN 的 VAE 适配件 SHA 与实际代码不符")
    if ms.sha256_file(source_pin_path.parent / "uv.lock") != pin.get("dependency_lock_sha256"):
        raise ValueError("stage2 SOURCE_PIN 的依赖锁 SHA 与实际文件不符")
    source_grid, identity, workers = {}, None, []
    for entry, _, key, _, ng in iter_rows_in_order(entries):
        tm = _load_json(tokens_root / f"{key}.metadata.json")
        lm = _load_json(latents_root / f"{key}.metadata.json")
        # S1 对 manifest 文件原字节取摘要；store/G0 的 manifest_sha256 是清单内部规范摘要。
        if tm.get("manifest_sha256") != manifest_file_sha256:
            raise ValueError(f"{key} S1 清单文件 SHA 与当前 episode_manifest 原字节不同")
        groups = {k: tm.get(k) for k in ("encoder", "encoder_state_sha256", "source_pin", "numerics")}
        if any(not isinstance(v, dict) or not v for v in groups.values()):
            raise ValueError(f"{key} stage2 身份字段缺失")
        encoder = dict(groups["encoder"])
        if "run_dir" in encoder and pathlib.Path(encoder["run_dir"]).resolve() != encoder_run_dir.resolve():
            raise ValueError(f"{key} encoder_run_dir 与显式输入不同")
        # 路径只用来定位；其余身份逐段必须完全相同。
        encoder.pop("run_dir", None)
        groups["encoder"] = encoder
        if identity is None:
            identity = groups
        elif ms.canonical_json_bytes(groups) != ms.canonical_json_bytes(identity):
            raise ValueError(f"{key} encoder/状态/pin/数值身份与其他段不同")
        projection = {k: pin.get(k) for k in ("mj_repo_commit", "export_source_sha256", "encoder_source_sha256")}
        if groups["source_pin"] != projection:
            raise ValueError(f"{key} S1 source_pin 与 S4 三字段投影不同")
        path = latents_root / f"{key}.bin"
        old_count = lm["num_grid"]
        blocks = tm.get("input_block_sha256")
        if path.stat().st_size != old_count * CHUNK_BYTES or not isinstance(blocks, list) or len(blocks) != ng:
            raise ValueError(f"{key} latent 字节数或偶数块 SHA 数量不符")
        whole = hashlib.sha256()
        with path.open("rb") as stream:
            for g in range(old_count):
                block = stream.read(CHUNK_BYTES)
                whole.update(block)
                if g % 2 == 0 and ms.sha256_bytes(block) != blocks[g // 2]:
                    raise ValueError(f"{key} 偶数格 {g} latent 字节 SHA 不符")
        if not (whole.hexdigest() == tm.get("input_latent_sha256") == lm.get("sha256") == total["segments"][key]["sha256"]):
            raise ValueError(f"{key} 整 latent SHA 不符（含未选中的奇数块）")
        source_grid[key] = tm["source_grid"]
        if not isinstance(tm.get("worker"), dict) or not tm["worker"]:
            raise ValueError(f"{key} 缺实际导出 worker 记录")
        workers.append(tm["worker"])
    if identity is None:
        raise ValueError("stage2 表不能为空")
    enc = identity["encoder"]
    if enc.get("run_name") != encoder_run_dir.name or enc.get("checkpoint_sha256") != expected_ckpt_sha256:
        raise ValueError("stage2 encoder run 或 checkpoint SHA 与显式输入不同")
    for name, expected in ((enc["checkpoint_name"], enc["checkpoint_sha256"]), ("config.yaml", enc["config_sha256"])):
        if pathlib.Path(name).name != name or ms.sha256_file(encoder_run_dir / name) != expected:
            raise ValueError(f"stage2 encoder 实体文件 SHA 不符: {name}")
    export = _load_json(tokens_root / "export_record.json")
    if (export.get("schema") != "stage2-memory-export-v1" or export.get("scope") != "full"
            or export.get("segments") != len(keys) or export.get("rows") != sum(e.demo.num_grid + e.exec.num_grid for e in entries)
            or len(export.get("segment_keys", [])) != len(keys) or set(export["segment_keys"]) != set(keys)
            or export.get("manifest_sha256") != manifest_file_sha256):
        raise ValueError("stage2 export_record 未完整覆盖清单")
    for group in identity:
        actual = export.get(group)
        if group == "encoder" and isinstance(actual, dict):
            actual = {k: v for k, v in actual.items() if k != "run_dir"}
        if ms.canonical_json_bytes(actual) != ms.canonical_json_bytes(identity[group]):
            raise ValueError(f"stage2 export_record 的 {group} 与段身份不同")
    latent_sha = ms.sha256_file(latents_root / "metadata.json")
    online_vae = {k: vae[k] for k in ("vae_id", "vae_state_sha256", "vae_dtype", "latent_mode", "batch", "tf32", "amp")}
    online_vae.update(adapter_sha256=pin["vae_adapter_sha256"],
                      preprocess=dict(frame_size=256, input_dtype="uint8", input_layout="THWC", window_frames=33),
                      numerics=identity["numerics"])
    evidence = _stage2_validation(validation_root, manifest=manifest, latent_sha=latent_sha,
                                 checkpoint_sha=expected_ckpt_sha256, source_pin=pin,
                                 numerics=identity["numerics"], online_vae=online_vae)
    evidence.update(restore_receipt={"path": vae_compat["receipt_path"], "sha256": vae_compat["receipt_sha256"]},
                    SOURCE_PIN={"path": str(source_pin_path.resolve()), "sha256": ms.sha256_file(source_pin_path)})
    provenance = {**identity, "source_pin": pin, "vae": vae,
                  "stage2_compat": {"schema": "stage2-compat-v1", "restore_receipt_sha256": vae_compat["receipt_sha256"],
                                    "historical_vae": vae_compat["identity"], "online_vae": online_vae, "evidence": evidence},
                  "manifest_sha256": manifest["sha256"], "motion_index_sha256": index_sha,
                  "episode_manifest_file_sha256": manifest_file_sha256,
                  "mj_repo_commit": pin["mj_repo_commit"], "source_grid_sha256": ms.sha256_bytes(ms.canonical_json_bytes(source_grid)),
                  "latents_metadata_sha256": latent_sha, "latents_root": str(latents_root.resolve()),
                  "tokens_root": str(tokens_root.resolve()), "raw_dir": str(raw_dir.resolve()),
                  "input_manifest_sha256": ms.sha256_file(input_manifest_path), "vae_compat": vae_compat,
                  "workers_vae": vae_workers, "workers_encoder": workers}
    ms.validate_stage2_provenance(provenance)
    return provenance


def check_raw_sources(manifest_path: pathlib.Path, latents_root: pathlib.Path,
                      raw_dir: pathlib.Path, oracle_report: pathlib.Path) -> pathlib.Path:
    """在写表前核对输入清单、Wan、oracle 三方绑定，并核 oracle 所验的元数据身份。"""
    input_path = manifest_path.parent / "input_manifest.json"
    inputs = _load_json(input_path)
    latent_meta = _load_json(latents_root / "metadata.json")
    report = _load_json(oracle_report)
    manifest = load_manifest(manifest_path)
    spec = ms.LAYOUT_SPECS[ms.LAYOUT]
    if any(latent_meta.get(k) != v for k, v in spec.fields().items()):
        raise ValueError("Wan metadata 的窗口契约与目标布局不同")
    if (report.get("demo_min_real_frames"), report.get("exec_min_real_frames")) != (spec.demo_min_real, spec.exec_min_real):
        raise ValueError("oracle 报告的窗口契约与目标布局不同")
    sources = [inputs, latent_meta, report, manifest]
    want = raw_dir.resolve()
    if any("raw_dir" not in source or pathlib.Path(source["raw_dir"]).resolve() != want for source in sources):
        raise ValueError("输入清单、Wan、oracle 与 --raw-dir 的三方目录绑定不同")
    if (report.get("manifest_sha256") != manifest["sha256"]
            or report.get("tested_metadata_sha256") != ms.sha256_file(latents_root / "metadata.json")):
        raise ValueError("oracle 报告与当前清单或 Wan metadata 不同源")
    if report.get("frame_mismatches") != 0 or report.get("metadata_mismatches") != 0:
        raise ValueError("oracle 全量输入核验未通过，拒绝打包")
    return input_path


def cmd_pack(args) -> None:
    t_all = time.perf_counter()
    manifest = load_manifest(args.manifest)
    stage2 = getattr(args, "layout", "legacy") == "stage2"
    layout = ms.STAGE2_LAYOUT if stage2 else ms.LAYOUT
    spec = ms.LAYOUT_SPECS[layout]
    if stage2 and (getattr(args, "latent_grid_subsample", None) != 2
                   or not getattr(args, "stage2_validation", None)
                   or not getattr(args, "stage2_source_pin", None)
                   or not getattr(args, "restore_receipt", None)):
        raise ValueError("stage2 必须显式提供 subsample=2、G0/G2/G3 目录、SOURCE_PIN 和恢复 receipt")
    if stage2 and args.force_break_lock:
        raise ValueError("stage2 禁止强制破锁")
    entries = ms.build_index_entries(manifest, spec)
    totals = ms.index_totals(entries)
    tokens_root = pathlib.Path(args.tokens).resolve()
    latents_root = pathlib.Path(args.latents).resolve()
    raw_dir = pathlib.Path(args.raw_dir).resolve()
    encoder_run_dir = pathlib.Path(args.encoder_run_dir).resolve()
    oracle_report = (pathlib.Path(args.oracle_report) if args.oracle_report
                     else latents_root.parent / "oracle/wan-mj/vae_report.json")
    input_manifest_path = check_raw_sources(
        pathlib.Path(args.manifest).resolve(), latents_root, raw_dir, oracle_report)
    vae_compat = None
    if getattr(args, "restore_receipt", None):
        if not args.expected_ckpt_sha256:
            raise RuntimeError("--restore-receipt 必须同时给出 --expected-ckpt-sha256")
        vae_compat = load_vae_compat(pathlib.Path(args.restore_receipt), latents_root,
                                     manifest_path=pathlib.Path(args.manifest), oracle_report=oracle_report)
    elif getattr(args, "expected_ckpt_sha256", None):
        raise RuntimeError("--expected-ckpt-sha256 仅用于历史 VAE 兼容模式（须配 --restore-receipt）")
    output = pathlib.Path(args.out).absolute()
    if stage2 and any(path.is_symlink() for path in (output, *output.parents)):
        raise RuntimeError(f"stage2 输出路径不能穿过符号链接: {output}")
    store_root = output.resolve()
    if store_root.is_symlink():
        raise RuntimeError(f"输出根是符号链接，拒绝写入: {store_root}")
    store_root.mkdir(parents=True, exist_ok=True)
    (store_root / "meta").mkdir(exist_ok=True)
    meta_path = store_root / ms.META_RELPATH
    lock = acquire_lock(store_root, resume=args.resume, force_break=args.force_break_lock, phase="pack")
    try:
        if meta_path.exists() and not args.resume:
            raise RuntimeError(f"store_meta.json 已存在: {meta_path}；重打包须显式 --resume")
        pin = _load_json(pathlib.Path(args.stage2_source_pin) if stage2 else _WAN_DIR / "SOURCE_PIN.json")
        index = ms.index_payload(manifest, entries, spec=spec, layout=layout, mj_repo_commit=pin["mj_repo_commit"])
        index_bytes = (json.dumps(index, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        index_sha = hashlib.sha256(index_bytes).hexdigest()
        print(f"[pack] episodes={len(entries)} rows={totals['rows']} (exec {totals['exec_rows']} + "
              f"demo {totals['demo_rows']}) index_sha256={index_sha[:16]}…", flush=True)
        # 写表前预检全部来源（兼容模式含 VAE/encoder 分组）；旧 compare 记录作废，须对新表重做。
        (store_root / BITEXACT_RELPATH).unlink(missing_ok=True)
        gather = gather_stage2_provenance if stage2 else gather_provenance
        extra = {"validation_root": pathlib.Path(args.stage2_validation),
                 "source_pin_path": pathlib.Path(args.stage2_source_pin),
                 "manifest_file_sha256": ms.sha256_file(args.manifest)} if stage2 else {}
        prov = gather(latents_root, tokens_root, entries, manifest, index_sha,
                                 encoder_run_dir=encoder_run_dir, raw_dir=raw_dir,
                                 input_manifest_path=input_manifest_path, vae_compat=vae_compat,
                                 expected_ckpt_sha256=getattr(args, "expected_ckpt_sha256", None), **extra)
        print(f"PACK_PROVENANCE_PRECHECK=PASS mode={'vae_compat' if vae_compat else 'single_pin'} "
              f"segments={sum(1 for _ in iter_rows_in_order(entries))}", flush=True)

        table_path = store_root / ms.MOTION_TABLE_RELPATH
        tmp = table_path.with_name(table_path.name + ".tmp")
        progress_path = store_root / ms.PROGRESS_RELPATH
        sha = hashlib.sha256()
        offset = 0
        n_seg = 0
        with open(tmp, "w+b") as f, open(progress_path, "w", encoding="utf-8") as pf:
            fd = f.fileno()
            for e, seg, key, row_base, ng in iter_rows_in_order(entries):
                if row_base * spec.row_bytes != offset:
                    raise RuntimeError(f"{key} row_base={row_base} 与写游标 {offset // spec.row_bytes} 不符")
                data, _ = read_segment_tokens(tokens_root, key, ng, spec=spec)
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
        if offset != totals["rows"] * spec.row_bytes:
            raise RuntimeError(f"表字节数 {offset} != rows × {spec.row_bytes}")
        os.replace(tmp, table_path)
        table_sha = sha.hexdigest()
        if ms.sha256_file(table_path) != table_sha:
            raise RuntimeError("表落盘后 sha256 与写入流不符")

        atomic_write_bytes(store_root / ms.INDEX_RELPATH, index_bytes)
        git_head = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=False).stdout.strip()
        import importlib.metadata as md
        meta = {
            "schema": ms.META_SCHEMA, "layout": layout, "status": "packed",
            **spec.fields(),
            "byte_order": ms.BYTE_ORDER, "array_order": ms.ARRAY_ORDER,
            "grid_stride": spec.grid_stride, "window_frames": ms.WINDOW_FRAMES, "grid_origin": ms.GRID_ORIGIN,
            "window_direction": ms.WINDOW_DIRECTION, "truncation_policy": ms.TRUNCATION_POLICY,
            "frame_size": ms.FRAME_SIZE,
            "tables": {ms.MOTION_KEY: {"row_shape": list(spec.row_shape), "dtype": "float32",
                                       "row_bytes": spec.row_bytes, "num_rows": totals["rows"],
                                       "relpath": ms.MOTION_TABLE_RELPATH, "byte_count": offset,
                                       "sha256": table_sha}},
            "num_rows": totals["rows"], "totals": totals,
            "row_order": index["row_order"],
            "manifest_sha256": manifest["sha256"], "manifest_scope": "full",
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
        print("PACK_MOTION_DONE=1 raw_dir_match=1 （pack.lock 保留，verify 回填后才释放）", flush=True)
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
    entries = ms.build_index_entries(manifest, meta.spec)
    if [dataclass_tuple(e) for e in entries] != [dataclass_tuple(e) for e in meta.entries]:
        raise RuntimeError("verify: 现场按清单重算的 index 与 meta 内 motion_index 不同")
    stage2 = meta.raw["layout"] == ms.STAGE2_LAYOUT
    if stage2 and (args.force_break_lock or not getattr(args, "check_padding", False)):
        raise ValueError("stage2 verify 必须 --check-padding 且禁止强制破锁")
    padding = validate_stage2_segments(pathlib.Path(meta.provenance["latents_root"]), tokens_root, entries) if stage2 else {}
    if stage2:
        source_grid = {key: list(range(0, ng * 2, 2)) for _, _, key, _, ng in iter_rows_in_order(entries)}
        if ms.sha256_bytes(ms.canonical_json_bytes(source_grid)) != meta.provenance.get("source_grid_sha256"):
            raise ValueError("verify source_grid 摘要与当前索引不符")
        if ms.sha256_file(pathlib.Path(meta.provenance["latents_root"]) / "metadata.json") != meta.provenance["latents_metadata_sha256"]:
            raise ValueError("verify 历史总 metadata 已变动")
    if stage2 or meta.provenance.get("vae_compat") is not None:
        check_bitexact_record(store_root, meta.raw)
    if stage2:
        for label, evidence in meta.provenance["stage2_compat"]["evidence"].items():
            if ms.sha256_file(evidence["path"]) != evidence["sha256"]:
                raise ValueError(f"verify 来源证据已变动: {label}")
    started_at = _now()
    had_lock = (store_root / ms.LOCK_RELPATH).exists()
    if stage2 and had_lock and not args.resume:
        raise ValueError("stage2 verify 接管 pack 锁须显式 --resume")
    acquire_lock(store_root, resume=(args.resume if had_lock else False),
                 force_break=args.force_break_lock, phase="verify")

    store = ms.MotionStore(store_root, meta=meta, manifest_path=manifest_path, verify_level="full")
    scanned = 0
    mismatches: list[tuple[int, str]] = []
    digests = bytearray()
    for _e, _seg, key, row_base, ng in iter_rows_in_order(entries):
        src, _ = read_segment_tokens(tokens_root, key, ng, spec=meta.spec)
        rows = store.rows(np.arange(row_base, row_base + ng))          # 真实读 API
        for m in range(ng):
            st = rows[m].tobytes()
            if st != src[m * meta.row_bytes:(m + 1) * meta.row_bytes]:
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
    if stage2:
        raw["verify"].update(padding, nonfinite=0, segment_set_mismatch=0)
    raw["row_digests"] = {"relpath": ms.ROW_DIGESTS_RELPATH, "algo": "blake2b-128", "covered_rows": scanned,
                          "coverage": ms.ROW_DIGEST_COVERAGE, "byte_count": len(digests),
                          "sha256": hashlib.sha256(bytes(digests)).hexdigest()}
    atomic_write_bytes(store_root / ms.META_RELPATH, json.dumps(raw, ensure_ascii=False, indent=1).encode())
    ms.MotionMeta.load(store_root)
    release_lock(store_root)
    print(f"[verify] meta 回填（status=verified）+ row_digests 落盘, 耗时 {time.perf_counter() - t_all:.1f}s",
          flush=True)
    details = " nonfinite=0 segment_set_mismatch=0 " + " ".join(f"{k}={v}" for k, v in padding.items()) if stage2 else ""
    print(f"VERIFY_MOTION=PASS scanned={scanned} mismatches=0{details}")


def check_bitexact_record(store_root: pathlib.Path, raw_meta: dict) -> dict:
    """兼容模式 verify 前置：必须已有 compare_wan.py tokens 对**本张 packed 表**写下的 PASS 记录（R2）。"""
    path = store_root / BITEXACT_RELPATH
    if not path.is_file():
        raise RuntimeError("verify: 缺 compare_wan.py tokens 的逐位比较记录，拒绝进入 verified")
    rec = _load_json(path)
    table = raw_meta["tables"][ms.MOTION_KEY]
    want = {"status": "PASS", "compared": int(raw_meta["num_rows"]), "mismatches": 0,
            "store_table_sha256": table["sha256"], "motion_index_sha256": raw_meta["motion_index_sha256"],
            "checkpoint_sha256": raw_meta["provenance"]["encoder"]["checkpoint_sha256"]}
    if raw_meta["layout"] == ms.STAGE2_LAYOUT:
        want.update(schema="stage2-encoder-bitexact-v1", scope="full", layout=ms.STAGE2_LAYOUT,
                    source_grid_sha256=raw_meta["provenance"]["source_grid_sha256"])
        if ms.sha256_file(store_root / ms.MOTION_TABLE_RELPATH) != table["sha256"]:
            raise RuntimeError("verify: 当前表与认证所绑定的 SHA 不符")
        if ms.sha256_file(store_root / ms.INDEX_RELPATH) != raw_meta["motion_index_sha256"]:
            raise RuntimeError("verify: 当前索引与认证所绑定的 SHA 不符")
    bad = [k for k, v in want.items() if type(rec.get(k)) is not type(v) or rec[k] != v]
    if bad:
        raise RuntimeError(f"verify: 逐位比较记录与当前表不符 {bad}")
    return rec


def dataclass_tuple(e: ms.IndexEntry) -> tuple:
    return (e.g, e.h5_file, e.raw_ep_idx, e.num_timesteps, e.exec_start_idx, e.spec,
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
    p.add_argument("--layout", choices=("legacy", "stage2"), default="legacy")
    p.add_argument("--latent-grid-subsample", type=int, default=1)
    p.add_argument("--stage2-validation", default=None)
    p.add_argument("--stage2-source-pin", default=None)
    p.add_argument("--manifest", required=True)
    p.add_argument("--tokens", required=True, help="<lib>/motion-tokens")
    p.add_argument("--latents", required=True, help="<lib>/wan-latents（只读 metadata 取 provenance）")
    p.add_argument("--out", required=True, help="<lib>/motion")
    p.add_argument("--encoder-run-dir", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--oracle-report", default=None, help="默认 <lib>/oracle/wan-mj/vae_report.json")
    p.add_argument("--restore-receipt", default=None,
                   help="S5-A restore_receipt.json：历史 VAE 与新 encoder 分组校验（须配 --expected-ckpt-sha256）")
    p.add_argument("--expected-ckpt-sha256", default=None, help="兼容模式下新 encoder 最终 ckpt 的显式 SHA256")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-break-lock", action="store_true")
    p.set_defaults(func=cmd_pack)
    p = sub.add_parser("verify")
    p.add_argument("--store", required=True)
    p.add_argument("--check-padding", action="store_true")
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
