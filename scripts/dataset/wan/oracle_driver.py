#!/usr/bin/env python3
"""D2 / D3 oracle 驱动：经 MotionJEPA 的 uv 环境调**原版** ``wan_motion_infer`` 产出对拍真值。

对应 0901-motion-memory-plan.md 第一部分 4.3 与第二部分 1.4「Wan 侧」。只依赖 stdlib + numpy + h5py + 原版模块
（``sys.path`` 指向 ``<MotionJEPA>/scripts/inference-example``），**禁止** import ``mme_vla_suite`` / jax / openpi；
起手 ``check_env()`` + ``pin_numerics()``；``load_vae(..., expected_state_sha256=9980d252…)``；
``load_encoder`` 用与被测**同一份**拷贝 ``v1-store/external/motionjepa/<run>/``（否则 provenance 的 run_dir 必不等）。

  vae      独立读取 ``episode_manifest.json``，按段长 ``range(0, max(0, L-32), 16)`` 重算全部
           ``(segment, m, start_global_frame)``，逐项反查被测 ``wan-latents/metadata.json``（集合、num_grid、seg_len、
           起点）；从官方 h5 读期望的 33 帧，送入原版 ``encode_chunk`` 前把 uint8 sha256 与被测记录的
           ``input_frames_sha256`` 逐窗比较；落 ``<out>/<段>.bin``（同构、同 chunk 序）+ ``vae_report.json``
  encoder  对**我方** ``wan-latents/<段>.bin`` 跑原版 ``motion_token``，按行序契约落
           ``<out>/motion_token.f32.bin`` + ``encoder_report.json``（含 77 张量 sha256 清单与 provenance）
  aggregate 多卡分片收拢：``vae`` / ``encoder`` 加 ``--shard-idx i --num-shards n`` 时按 ``expected_segments`` 稳定序
           取 ``segs[i::n]``，各片只写自己的段文件与 ``*_report.shard<i>of<n>.json``（encoder 片另写
           ``motion_token.shard<i>of<n>.f32.bin``）；``aggregate --manifest … --out …`` 核片数齐全、口径一致后合成
           与单进程逐字节同构的 ``vae_report.json`` / ``motion_token.f32.bin`` + ``encoder_report.json``，
           ``compare_wan.py`` 不感知分片。环境 B 8×A100 用 8 片（400 ep ≈ 6,700 窗单进程 ≈ 100 min → ≈ 13 min）。

用法（MotionJEPA 项目只读，uv --no-sync）：
  PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy HF_HOME=v1-store/cache/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \\
  uv run --project /nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA --no-sync python \\
      scripts/dataset/wan/oracle_driver.py vae --manifest <lib>/meta/episode_manifest.json --raw-dir <h5 目录> \\
      --latents <lib>/wan-latents --out <lib>/oracle/wan-mj
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
from decimal import Decimal
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "assets"))

import assets_lock as al  # noqa: E402

# 环境 A 默认 turbo 只读副本；环境 B 用环境变量 MJ_REPO（paths.sh 同名）指向 /scratch/hongze/MotionJEPA
MJ_REPO_DEFAULT = os.environ.get("MJ_REPO", "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA")
ENCODER_RUN_DIR_DEFAULT = str(_REPO_ROOT / "v1-store" / "external" / "motionjepa" / "wan-full1600-filter2-b176x4-72ep-a")
VAE_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _expected_ckpt(args):
    """ckpt 期望值：默认取 ASSETS_LOCK.json 钉死的那份，显式传 SKIP 才跳过。

    本文件是**探针 / 对拍工具**，探一个未入 lock 的 run_dir 是它的本职，所以保留 SKIP 出口；
    生产路径（encode_motion.py 的 required=True、run_local.py、motion_sidecar.py）不提供 SKIP。
    """
    if args.expected_ckpt_sha256 == "SKIP":
        return None
    return args.expected_ckpt_sha256 or al.expected_sha256("motionjepa_ckpt")


# 口径常量（独立于被测实现书写；与计划 2.2 / 4.1 同值）
STRIDE = 16
WINDOW = 33
FRAME_SIZE = 256
LAT_SHAPE = (9, 16, 32, 32)
CHUNK_BYTES = 9 * 16 * 32 * 32 * 4
TOKEN_DIM = 768
TOKEN_BYTES = TOKEN_DIM * 4          # f32 一行

for _name in ("mme_vla_suite", "jax", "openpi"):
    if _name in sys.modules:
        raise SystemExit(f"oracle 驱动禁止载入 {_name}")


def import_orig(mj_repo: str):
    d = pathlib.Path(mj_repo) / "scripts" / "inference-example"
    if not (d / "wan_motion_infer.py").is_file():
        raise SystemExit(f"缺原版模块: {d / 'wan_motion_infer.py'}")
    sys.path.insert(0, str(d))
    import wan_motion_infer as W  # noqa: E402
    return W


def manifest_sha256(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def load_manifest(path: str) -> dict:
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if payload.get("sha256") != manifest_sha256(payload):
        raise SystemExit(f"清单 sha256 不符: {path}")
    return payload


def task_of(h5_file: str) -> str:
    n = os.path.basename(h5_file)
    assert n.startswith("record_dataset_") and n.endswith(".h5"), h5_file
    return n[len("record_dataset_"):-3]


def expected_segments(manifest: dict, *, demo_min_real: int, exec_min_real: int) -> list[dict]:
    """独立重算 demo 与 exec；两段分别取最少真实帧数，不导入被测公式。"""
    if not 1 <= demo_min_real <= WINDOW or exec_min_real != WINDOW:
        raise SystemExit("oracle 要求 demo 最少真实帧在 [1,33]，exec 固定 33")
    out = []
    for ep in manifest["episodes"]:
        T, es = int(ep["num_timesteps"]), int(ep["exec_start_idx"])
        for seg, start, L in (("demo", 0, es), ("exec", es, T - es)):
            min_real = demo_min_real if seg == "demo" else exec_min_real
            starts = list(range(0, max(0, L - (min_real - 1)), STRIDE))
            if not starts:
                continue
            out.append({"key": f"{task_of(ep['h5_file'])}_ep{int(ep['raw_ep_idx'])}_{seg}",
                        "g": int(ep["global_episode_idx"]), "h5_file": ep["h5_file"],
                        "raw_ep_idx": int(ep["raw_ep_idx"]), "seg": seg, "start": start, "len": L,
                        "starts": starts, "min_real": min_real})
    return out


def read_frames(h5_path: str, raw_ep_idx: int, t0: int, n: int) -> np.ndarray:
    import h5py
    out = np.empty((n, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    with h5py.File(h5_path, "r") as f:
        ep = f[f"episode_{raw_ep_idx}"]
        for i in range(n):
            ep[f"timestep_{t0 + i}/obs/front_rgb"].read_direct(out[i])
    return out


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------- vae ----------------

def sample_selected(sample_spec: str | None, key: str, m: int, *, padded: bool) -> bool:
    """确定式抽样：SHA256 前八字节按大端解释，模 10000 与整数阈值比较。"""
    if sample_spec is None:
        return True
    match = re.fullmatch(r"padded:all,rest:(0(?:\.\d{1,4})?|1(?:\.0{1,4})?),seed:(\d+)", sample_spec)
    if match is None:
        raise SystemExit(f"抽样规格不合法: {sample_spec!r}")
    threshold = int(Decimal(match.group(1)) * 10000)
    value = int.from_bytes(hashlib.sha256(f"{int(match.group(2))}:{key}:{m}".encode()).digest()[:8], "big")
    return padded or value % 10000 < threshold

def cmd_vae(args):
    import torch
    W = import_orig(args.mj_repo)
    W.check_env()
    W.pin_numerics()
    W.check_versions(strict=True, with_diffusers=True)
    torch.manual_seed(0)
    device = torch.device("cuda")
    manifest = load_manifest(args.manifest)
    if pathlib.Path(args.raw_dir).resolve() != pathlib.Path(manifest["raw_dir"]).resolve():
        raise SystemExit("--raw-dir 与清单绑定的 raw_dir 不同")
    segs = expected_segments(manifest, demo_min_real=args.demo_min_real, exec_min_real=args.exec_min_real)
    lat_root = pathlib.Path(args.latents)
    tested = json.loads((lat_root / "metadata.json").read_text(encoding="utf-8"))
    schema = tested.get("schema")
    if schema not in (2, 3):
        raise SystemExit(f"被测 metadata.json schema={schema} 不在 {{2,3}}")
    if schema == 2 and (args.demo_min_real, args.exec_min_real) != (33, 33):
        raise SystemExit("schema=2 只接受历史 33/33 契约")
    if schema == 3:
        contract = {"demo_min_real_frames": args.demo_min_real, "exec_min_real_frames": args.exec_min_real,
                    "demo_tail_pad": "none" if args.demo_min_real == 33 else "repeat_last"}
        if any(tested.get(k) != v for k, v in contract.items()):
            raise SystemExit("被测 metadata 的补帧契约与 oracle 显式参数不同")
        if pathlib.Path(tested["raw_dir"]).resolve() != pathlib.Path(args.raw_dir).resolve():
            raise SystemExit("Wan metadata 与 oracle 的 raw_dir 不同")
    sample_selected(args.sample_spec, "规格检查", 0, padded=False)
    for k, v in (("grid_stride", STRIDE), ("window_frames", WINDOW), ("grid_origin", "segment_start"),
                 ("window_direction", "forward"), ("truncation_policy", "none")):
        if tested.get(k) != v:
            raise SystemExit(f"被测 metadata {k}={tested.get(k)!r} != oracle 口径 {v!r}")
    exp_keys = {s["key"] for s in segs}
    got_keys = set(tested["segments"])
    if exp_keys != got_keys:
        raise SystemExit(f"段集合不符: 缺 {sorted(exp_keys - got_keys)[:5]} 多 {sorted(got_keys - exp_keys)[:5]}")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    segs = _shard(segs, args)
    vae, vinfo = W.load_vae(VAE_ID, device, expected_state_sha256=W.VAE_STATE_SHA256_EXPECTED)
    print(f"[oracle-vae] 原版模块 {W.__file__} sha256={W.sha256_file(W.__file__)[:16]}… "
          f"vae_state={vinfo['vae_state_sha256'][:16]}… segments={len(segs)} shard={args.shard_idx}/{args.num_shards}", flush=True)

    frame_mismatches, meta_mismatches, n_windows = 0, 0, 0
    metadata_rows_checked = 0
    sampled = []
    per_seg = {}
    t_all = time.perf_counter()
    for i, s in enumerate(segs):
        tm = tested["segments"][s["key"]]
        rows = tm["rows"]
        if int(tm["num_grid"]) != len(s["starts"]) or int(tm["seg_len"]) != s["len"] or len(rows) != len(s["starts"]):
            meta_mismatches += 1
            print(f"  ✗ {s['key']} metadata num_grid/seg_len/rows 与重算不符: "
                  f"{tm['num_grid']}/{tm['seg_len']}/{len(rows)} vs {len(s['starts'])}/{s['len']}", flush=True)
            continue
        frames = read_frames(str(pathlib.Path(args.raw_dir) / s["h5_file"]), s["raw_ep_idx"], s["start"], s["len"])
        blob = bytearray()
        seg_frame_bad = 0
        for m, off in enumerate(s["starts"]):
            r = rows[m]
            metadata_rows_checked += 1
            if int(r["m"]) != m or int(r["seg_offset"]) != off or int(r["start_global_frame"]) != s["start"] + off:
                meta_mismatches += 1
                print(f"  ✗ {s['key']} m={m} 行记录 {r} 与重算 (off={off}, f={s['start'] + off}) 不符", flush=True)
            real = min(WINDOW, s["len"] - off)
            pad = WINDOW - real
            source_frame = s["start"] + s["len"] - 1 if pad else None
            if real < s["min_real"] or (s["seg"] == "exec" and pad):
                raise SystemExit("oracle 独立切窗越过段契约")
            if schema == 3 and (r.get("real_frames"), r.get("pad_frames"), r.get("pad_source_frame")) != (real, pad, source_frame):
                meta_mismatches += 1
            if schema == 3 and any(k not in r for k in ("real_frames", "pad_frames", "pad_source_frame")):
                meta_mismatches += 1
            if r.get("input_shape") != [WINDOW, FRAME_SIZE, FRAME_SIZE, 3] or r.get("input_dtype") != "uint8":
                meta_mismatches += 1
            # 独立装配：先分配固定 33 帧，尾部直接广播本段最后一帧。
            win = np.empty((WINDOW, FRAME_SIZE, FRAME_SIZE, 3), np.uint8)
            win[:real] = frames[off:off + real]
            if pad:
                win[real:] = frames[-1]
            got_sha = sha_arr(win)
            if got_sha != r["input_frames_sha256"]:
                frame_mismatches += 1
                seg_frame_bad += 1
            if sample_selected(args.sample_spec, s["key"], m, padded=bool(pad)):
                sampled.append({"segment": s["key"], "m": m})
                W.pin_numerics()
                lat = W.encode_chunk(vae, win, device)                  # 原版
                blob += lat.cpu().numpy().astype(np.float32).tobytes()
                n_windows += 1
            else:
                blob += bytes(CHUNK_BYTES)                            # 保持每段定长，未抽中行不作为数值真值。
        p = out / f"{s['key']}.bin"
        atomic_write(p, bytes(blob))
        sha = sha_file(p)
        atomic_write(out / f"{s['key']}.bin.sha256", f"{sha}  {s['key']}.bin\n".encode())
        per_seg[s["key"]] = {"num_grid": len(s["starts"]), "sha256": sha, "frame_mismatches": seg_frame_bad}
        if (i + 1) % 10 == 0 or i + 1 == len(segs):
            print(f"[oracle-vae] {i + 1}/{len(segs)} 段, windows={n_windows} frame_mismatches={frame_mismatches} "
                  f"({time.perf_counter() - t_all:.0f}s)", flush=True)
    report = {"manifest_sha256": manifest["sha256"], "segments": per_seg, "windows": n_windows,
              "windows_encoded": n_windows, "metadata_rows_checked": metadata_rows_checked,
              "raw_dir": str(pathlib.Path(args.raw_dir).resolve()), "sample_spec": args.sample_spec,
              "demo_min_real_frames": args.demo_min_real, "exec_min_real_frames": args.exec_min_real,
              "frame_mismatches": frame_mismatches, "metadata_mismatches": meta_mismatches,
              "tested_metadata_sha256": sha_file(lat_root / "metadata.json"),
              "elapsed_s": time.perf_counter() - t_all, "vae_provenance": vinfo,
              "orig_module_sha256": W.sha256_file(W.__file__), "mj_repo": os.path.abspath(args.mj_repo)}
    sample_name = _shard_name("sampled_windows", args, ".json") if args.num_shards > 1 else "sampled_windows.json"
    atomic_write(out / sample_name, json.dumps({"sample_spec": args.sample_spec, "windows": sampled}, ensure_ascii=False).encode())
    if args.num_shards > 1:
        report["shard"] = {"idx": args.shard_idx, "num": args.num_shards}
        (out / _shard_name("vae_report", args, ".json")).write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=str))
        print(f"ORACLE_VAE=DONE shard={args.shard_idx}/{args.num_shards} metadata_rows={metadata_rows_checked} windows={n_windows} "
              f"frame_mismatches={frame_mismatches} metadata_mismatches={meta_mismatches} "
              f"elapsed={report['elapsed_s']:.0f}s", flush=True)
    else:
        (out / "vae_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        print(f"ORACLE_VAE=DONE metadata_rows={metadata_rows_checked} windows={n_windows} frame_mismatches={frame_mismatches} "
              f"metadata_mismatches={meta_mismatches} elapsed={report['elapsed_s']:.0f}s", flush=True)
    if frame_mismatches or meta_mismatches:
        raise SystemExit(1)


# ---------------- encoder ----------------

def cmd_encoder(args):
    import torch
    W = import_orig(args.mj_repo)
    W.check_env()
    W.pin_numerics()
    W.check_versions(strict=True, with_diffusers=False)
    torch.manual_seed(0)
    device = torch.device("cuda")
    manifest = load_manifest(args.manifest)
    segs = _shard(expected_segments(manifest, demo_min_real=args.demo_min_real, exec_min_real=args.exec_min_real), args)
    lat_root = pathlib.Path(args.latents)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    encoder, einfo, use_amp = W.load_encoder(args.encoder_run_dir, args.checkpoint, device,
                                             expected_sha256=_expected_ckpt(args))
    sd = encoder.state_dict()
    state_sha = {k: hashlib.sha256(sd[k].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                 for k in sorted(sd)}
    print(f"[oracle-enc] ckpt={einfo['checkpoint_sha256'][:16]}… epoch={einfo['checkpoint_epoch']} amp={einfo['amp']} "
          f"tensors={len(state_sha)} affine_finite={bool(torch.isfinite(encoder.latents_std).all())}", flush=True)
    # 行序契约：清单序逐 episode，demo 先 exec 后（segs 已按此序生成）
    table = bytearray()
    row_map = []
    t_all = time.perf_counter()
    n = 0
    for s in segs:
        p = lat_root / f"{s['key']}.bin"
        want = (lat_root / f"{s['key']}.bin.sha256").read_text().split()[0]
        if sha_file(p) != want:
            raise SystemExit(f"{p} sha256 与 sidecar 不符")
        if p.stat().st_size != len(s["starts"]) * CHUNK_BYTES:
            raise SystemExit(f"{p} 字节数与重算 num_grid 不符")
        for m in range(len(s["starts"])):
            blk = W.read_latent_block(str(p), m)
            W.pin_numerics()
            tok = W.motion_token(encoder, blk, use_amp, device)          # 原版
            table += tok.astype(np.float32).tobytes()
            row_map.append({"row": n, "segment": s["key"], "m": m, "g": s["g"], "seg": s["seg"]})
            n += 1
    if args.num_shards > 1:
        tbl = out / _shard_name("motion_token", args, ".f32.bin")
        atomic_write(tbl, bytes(table))
        report = {"rows": n, "table_sha256": sha_file(tbl), "row_map": row_map,
                  "encoder_provenance": einfo, "encoder_state_sha256": state_sha,
                  "elapsed_s": time.perf_counter() - t_all, "orig_module_sha256": W.sha256_file(W.__file__),
                  "shard": {"idx": args.shard_idx, "num": args.num_shards}}
        (out / _shard_name("encoder_report", args, ".json")).write_text(
            json.dumps(report, ensure_ascii=False, indent=1, default=str))
        print(f"ORACLE_ENCODER=DONE shard={args.shard_idx}/{args.num_shards} rows={n} "
              f"elapsed={report['elapsed_s']:.0f}s", flush=True)
        return
    atomic_write(out / "motion_token.f32.bin", bytes(table))
    report = {"rows": n, "table_sha256": sha_file(out / "motion_token.f32.bin"), "row_map": row_map,
              "encoder_provenance": einfo, "encoder_state_sha256": state_sha,
              "elapsed_s": time.perf_counter() - t_all, "orig_module_sha256": W.sha256_file(W.__file__)}
    (out / "encoder_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print(f"ORACLE_ENCODER=DONE rows={n} elapsed={report['elapsed_s']:.0f}s", flush=True)


# ---------------- 分片与收拢 ----------------

def _shard(segs: list[dict], args) -> list[dict]:
    """按 expected_segments 稳定序取模分片；num_shards=1 时原样返回（单进程口径不变）。"""
    n, i = int(args.num_shards), int(args.shard_idx)
    if n < 1 or not (0 <= i < n):
        raise SystemExit(f"--shard-idx {i} / --num-shards {n} 非法")
    return segs if n == 1 else segs[i::n]


def _shard_name(stem: str, args, suffix: str) -> str:
    return f"{stem}.shard{int(args.shard_idx)}of{int(args.num_shards)}{suffix}"


def _load_shards(out: pathlib.Path, stem: str, n: int) -> list[dict]:
    reps = []
    for i in range(n):
        p = out / f"{stem}.shard{i}of{n}.json"
        if not p.is_file():
            raise SystemExit(f"缺分片报告 {p}")
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("shard") != {"idx": i, "num": n}:
            raise SystemExit(f"{p} 的 shard 字段 {r.get('shard')} 与文件名不符")
        reps.append(r)
    return reps


def _same_across(reps: list[dict], key: str, what: str):
    vals = [json.dumps(r.get(key), sort_keys=True, default=str) for r in reps]
    if len(set(vals)) != 1:
        raise SystemExit(f"{what}: 各片 {key} 不一致")
    return reps[0].get(key)


def cmd_aggregate(args):
    """把 n 片的报告 / 表合成与单进程逐字节同构的产物；任一片缺失、口径不一致或段集合与清单重算不符即 FAIL。"""
    manifest = load_manifest(args.manifest)
    segs = expected_segments(manifest, demo_min_real=args.demo_min_real, exec_min_real=args.exec_min_real)
    out = pathlib.Path(args.out)
    n = int(args.num_shards)
    did = []
    if args.kind in ("vae", "both"):
        reps = _load_shards(out, "vae_report", n)
        per_seg = {}
        for r in reps:
            dup = set(per_seg) & set(r["segments"])
            if dup:
                raise SystemExit(f"vae: 段 {sorted(dup)[:5]} 出现在多个分片")
            per_seg.update(r["segments"])
        exp_keys = {s["key"] for s in segs}
        if set(per_seg) != exp_keys:
            raise SystemExit(f"vae: 分片段集合 ≠ 清单重算: 缺 {sorted(exp_keys - set(per_seg))[:5]} 多 {sorted(set(per_seg) - exp_keys)[:5]}")
        for s in segs:
            if not (out / f"{s['key']}.bin").is_file() or not (out / f"{s['key']}.bin.sha256").is_file():
                raise SystemExit(f"vae: 缺段文件 {s['key']}.bin(.sha256)")
        report = {
            "manifest_sha256": _same_across(reps, "manifest_sha256", "vae"),
            "raw_dir": _same_across(reps, "raw_dir", "vae"),
            "sample_spec": _same_across(reps, "sample_spec", "vae"),
            "demo_min_real_frames": _same_across(reps, "demo_min_real_frames", "vae"),
            "exec_min_real_frames": _same_across(reps, "exec_min_real_frames", "vae"),
            "metadata_rows_checked": sum(int(r["metadata_rows_checked"]) for r in reps),
            "windows_encoded": sum(int(r["windows_encoded"]) for r in reps),
            "segments": {s["key"]: per_seg[s["key"]] for s in segs},
            "windows": sum(int(r["windows"]) for r in reps),
            "frame_mismatches": sum(int(r["frame_mismatches"]) for r in reps),
            "metadata_mismatches": sum(int(r["metadata_mismatches"]) for r in reps),
            "tested_metadata_sha256": _same_across(reps, "tested_metadata_sha256", "vae"),
            "elapsed_s": max(float(r["elapsed_s"]) for r in reps),
            "elapsed_s_sum": sum(float(r["elapsed_s"]) for r in reps),
            "vae_provenance": _same_across(reps, "vae_provenance", "vae"),
            "orig_module_sha256": _same_across(reps, "orig_module_sha256", "vae"),
            "mj_repo": _same_across(reps, "mj_repo", "vae"),
            "aggregated_from_shards": n,
        }
        if report["manifest_sha256"] != manifest["sha256"]:
            raise SystemExit("vae: 分片报告的 manifest_sha256 与 --manifest 不同")
        if (report["demo_min_real_frames"], report["exec_min_real_frames"]) != (args.demo_min_real, args.exec_min_real):
            raise SystemExit("vae: 分片窗口契约与 aggregate 显式参数不同")
        if report["metadata_rows_checked"] != sum(len(s["starts"]) for s in segs):
            raise SystemExit("vae: 全量元数据检查行数不足")
        merged = []
        for i in range(n):
            sample = json.loads((out / f"sampled_windows.shard{i}of{n}.json").read_text())
            if sample["sample_spec"] != report["sample_spec"]:
                raise SystemExit("vae: 分片抽样表规格不一致")
            merged.extend(sample["windows"])
        keys = {(r["segment"], int(r["m"])) for r in merged}
        if len(keys) != len(merged) or len(merged) != report["windows"] or report["windows_encoded"] != report["windows"]:
            raise SystemExit("vae: 抽样集合重复或编码计数不符")
        atomic_write(out / "sampled_windows.json", json.dumps({"sample_spec": report["sample_spec"], "windows": merged}).encode())
        (out / "vae_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        print(f"ORACLE_VAE=DONE metadata_rows={report['metadata_rows_checked']} windows={report['windows']} frame_mismatches={report['frame_mismatches']} "
              f"metadata_mismatches={report['metadata_mismatches']} elapsed={report['elapsed_s']:.0f}s shards={n}", flush=True)
        did.append("vae")
        if report["frame_mismatches"] or report["metadata_mismatches"]:
            raise SystemExit(1)
    if args.kind in ("encoder", "both"):
        reps = _load_shards(out, "encoder_report", n)
        # 各片的 (segment, m) → token 字节；按清单序 demo 先 exec 后重排回单进程行序
        tok = {}
        for i, r in enumerate(reps):
            tbl = out / f"motion_token.shard{i}of{n}.f32.bin"
            if not tbl.is_file() or sha_file(tbl) != r["table_sha256"]:
                raise SystemExit(f"encoder: 片 {i} 表缺失或 sha256 与报告不符")
            raw = tbl.read_bytes()
            if len(raw) != int(r["rows"]) * TOKEN_BYTES or len(r["row_map"]) != int(r["rows"]):
                raise SystemExit(f"encoder: 片 {i} 行数 {r['rows']} 与表字节 {len(raw)} / row_map 不符")
            for j, rm in enumerate(r["row_map"]):
                k = (rm["segment"], int(rm["m"]))
                if k in tok:
                    raise SystemExit(f"encoder: 行 {k} 出现在多个分片")
                tok[k] = (raw[j * TOKEN_BYTES:(j + 1) * TOKEN_BYTES], rm)
        table = bytearray()
        row_map = []
        for s in segs:
            for m in range(len(s["starts"])):
                k = (s["key"], m)
                if k not in tok:
                    raise SystemExit(f"encoder: 清单重算行 {k} 在各片中缺失")
                b, rm = tok.pop(k)
                table += b
                row_map.append({"row": len(row_map), "segment": s["key"], "m": m, "g": int(rm["g"]), "seg": rm["seg"]})
        if tok:
            raise SystemExit(f"encoder: 各片多出 {len(tok)} 行不在清单重算内")
        atomic_write(out / "motion_token.f32.bin", bytes(table))
        report = {"rows": len(row_map), "table_sha256": sha_file(out / "motion_token.f32.bin"), "row_map": row_map,
                  "encoder_provenance": _same_across(reps, "encoder_provenance", "encoder"),
                  "encoder_state_sha256": _same_across(reps, "encoder_state_sha256", "encoder"),
                  "elapsed_s": max(float(r["elapsed_s"]) for r in reps),
                  "elapsed_s_sum": sum(float(r["elapsed_s"]) for r in reps),
                  "orig_module_sha256": _same_across(reps, "orig_module_sha256", "encoder"),
                  "aggregated_from_shards": n}
        (out / "encoder_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        print(f"ORACLE_ENCODER=DONE rows={report['rows']} elapsed={report['elapsed_s']:.0f}s shards={n}", flush=True)
        did.append("encoder")
    print(f"ORACLE_AGGREGATE=DONE kinds={','.join(did)} shards={n}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mj-repo", default=MJ_REPO_DEFAULT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("vae")
    p.add_argument("--manifest", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--latents", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--demo-min-real", type=int, required=True)
    p.add_argument("--exec-min-real", type=int, required=True)
    p.add_argument("--sample-spec", default=None)
    p.set_defaults(func=cmd_vae)
    p = sub.add_parser("encoder")
    p.add_argument("--manifest", required=True)
    p.add_argument("--latents", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoder-run-dir", default=ENCODER_RUN_DIR_DEFAULT)
    p.add_argument("--checkpoint", default="checkpoint_epoch_72.pt")
    p.add_argument("--expected-ckpt-sha256", default="",
                   help="默认按 ASSETS_LOCK.json 校；探未入 lock 的 run_dir 时显式传 SKIP")
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--demo-min-real", type=int, required=True)
    p.add_argument("--exec-min-real", type=int, required=True)
    p.set_defaults(func=cmd_encoder)
    p = sub.add_parser("aggregate", help="把 --num-shards 片的 vae / encoder 产物合成单进程同构产物")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--kind", choices=["vae", "encoder", "both"], default="both")
    p.add_argument("--demo-min-real", type=int, required=True)
    p.add_argument("--exec-min-real", type=int, required=True)
    p.set_defaults(func=cmd_aggregate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
