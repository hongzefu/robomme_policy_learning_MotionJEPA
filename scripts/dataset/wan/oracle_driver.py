#!/usr/bin/env python3
"""D2 / D3 oracle 驱动：经 MotionJEPA 的 uv 环境调**原版** ``wan_motion_infer`` 产出对拍真值。

对应 motion-memory-plan.md 第一部分 4.3 与第二部分 1.4「Wan 侧」。只依赖 stdlib + numpy + h5py + 原版模块
（``sys.path`` 指向 ``<MotionJEPA>/scripts/inference-example``），**禁止** import ``mme_vla_suite`` / jax / openpi；
起手 ``check_env()`` + ``pin_numerics()``；``load_vae(..., expected_state_sha256=9980d252…)``；
``load_encoder`` 用与被测**同一份**拷贝 ``v1-store/external/motionjepa/<run>/``（否则 provenance 的 run_dir 必不等）。

  vae      独立读取 ``episode_manifest.json``，按段长 ``range(0, max(0, L-32), 16)`` 重算全部
           ``(segment, m, start_global_frame)``，逐项反查被测 ``wan-latents/metadata.json``（集合、num_grid、seg_len、
           起点）；从官方 h5 读期望的 33 帧，送入原版 ``encode_chunk`` 前把 uint8 sha256 与被测记录的
           ``input_frames_sha256`` 逐窗比较；落 ``<out>/<段>.bin``（同构、同 chunk 序）+ ``vae_report.json``
  encoder  对**我方** ``wan-latents/<段>.bin`` 跑原版 ``motion_token``，按行序契约落
           ``<out>/motion_token.f32.bin`` + ``encoder_report.json``（含 77 张量 sha256 清单与 provenance）

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
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "assets"))

import assets_lock as al  # noqa: E402

MJ_REPO_DEFAULT = "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/MotionJEPA"
ENCODER_RUN_DIR_DEFAULT = str(_REPO_ROOT / "v1-store" / "external" / "motionjepa" / "wan-v8-filter10-72ep-a")
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


def expected_segments(manifest: dict) -> list[dict]:
    """独立重算：每 episode demo=[0,es)、exec=[es,T)；起点 range(0, max(0, L-32), 16)。"""
    out = []
    for ep in manifest["episodes"]:
        T, es = int(ep["num_timesteps"]), int(ep["exec_start_idx"])
        for seg, start, L in (("demo", 0, es), ("exec", es, T - es)):
            starts = list(range(0, max(0, L - (WINDOW - 1)), STRIDE))
            if not starts:
                continue
            out.append({"key": f"{task_of(ep['h5_file'])}_ep{int(ep['raw_ep_idx'])}_{seg}",
                        "g": int(ep["global_episode_idx"]), "h5_file": ep["h5_file"],
                        "raw_ep_idx": int(ep["raw_ep_idx"]), "seg": seg, "start": start, "len": L,
                        "starts": starts})
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

def cmd_vae(args):
    import torch
    W = import_orig(args.mj_repo)
    W.check_env()
    W.pin_numerics()
    W.check_versions(strict=True, with_diffusers=True)
    torch.manual_seed(0)
    device = torch.device("cuda")
    manifest = load_manifest(args.manifest)
    segs = expected_segments(manifest)
    lat_root = pathlib.Path(args.latents)
    tested = json.loads((lat_root / "metadata.json").read_text(encoding="utf-8"))
    if tested.get("schema") != 2:
        raise SystemExit(f"被测 metadata.json schema={tested.get('schema')} != 2")
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
    vae, vinfo = W.load_vae(VAE_ID, device, expected_state_sha256=W.VAE_STATE_SHA256_EXPECTED)
    print(f"[oracle-vae] 原版模块 {W.__file__} sha256={W.sha256_file(W.__file__)[:16]}… "
          f"vae_state={vinfo['vae_state_sha256'][:16]}… segments={len(segs)}", flush=True)

    frame_mismatches, meta_mismatches, n_windows = 0, 0, 0
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
            if int(r["m"]) != m or int(r["seg_offset"]) != off or int(r["start_global_frame"]) != s["start"] + off:
                meta_mismatches += 1
                print(f"  ✗ {s['key']} m={m} 行记录 {r} 与重算 (off={off}, f={s['start'] + off}) 不符", flush=True)
            win = np.ascontiguousarray(frames[off:off + WINDOW])
            got_sha = sha_arr(win)
            if got_sha != r["input_frames_sha256"]:
                frame_mismatches += 1
                seg_frame_bad += 1
            W.pin_numerics()
            lat = W.encode_chunk(vae, win, device)                      # 原版
            blob += lat.cpu().numpy().astype(np.float32).tobytes()
            n_windows += 1
        p = out / f"{s['key']}.bin"
        atomic_write(p, bytes(blob))
        sha = sha_file(p)
        atomic_write(out / f"{s['key']}.bin.sha256", f"{sha}  {s['key']}.bin\n".encode())
        per_seg[s["key"]] = {"num_grid": len(s["starts"]), "sha256": sha, "frame_mismatches": seg_frame_bad}
        if (i + 1) % 10 == 0 or i + 1 == len(segs):
            print(f"[oracle-vae] {i + 1}/{len(segs)} 段, windows={n_windows} frame_mismatches={frame_mismatches} "
                  f"({time.perf_counter() - t_all:.0f}s)", flush=True)
    report = {"manifest_sha256": manifest["sha256"], "segments": per_seg, "windows": n_windows,
              "frame_mismatches": frame_mismatches, "metadata_mismatches": meta_mismatches,
              "tested_metadata_sha256": sha_file(lat_root / "metadata.json"),
              "elapsed_s": time.perf_counter() - t_all, "vae_provenance": vinfo,
              "orig_module_sha256": W.sha256_file(W.__file__), "mj_repo": os.path.abspath(args.mj_repo)}
    (out / "vae_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print(f"ORACLE_VAE=DONE windows={n_windows} frame_mismatches={frame_mismatches} "
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
    segs = expected_segments(manifest)
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
    atomic_write(out / "motion_token.f32.bin", bytes(table))
    report = {"rows": n, "table_sha256": sha_file(out / "motion_token.f32.bin"), "row_map": row_map,
              "encoder_provenance": einfo, "encoder_state_sha256": state_sha,
              "elapsed_s": time.perf_counter() - t_all, "orig_module_sha256": W.sha256_file(W.__file__)}
    (out / "encoder_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print(f"ORACLE_ENCODER=DONE rows={n} elapsed={report['elapsed_s']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mj-repo", default=MJ_REPO_DEFAULT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("vae")
    p.add_argument("--manifest", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--latents", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_vae)
    p = sub.add_parser("encoder")
    p.add_argument("--manifest", required=True)
    p.add_argument("--latents", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoder-run-dir", default=ENCODER_RUN_DIR_DEFAULT)
    p.add_argument("--checkpoint", default="checkpoint_epoch_72.pt")
    p.add_argument("--expected-ckpt-sha256", default="",
                   help="默认按 ASSETS_LOCK.json 校；探未入 lock 的 run_dir 时显式传 SKIP")
    p.set_defaults(func=cmd_encoder)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
