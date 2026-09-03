#!/usr/bin/env python3
"""Wan VAE 网格窗抽取 worker：按段领任务，每段 num_grid 个 33 帧窗口 → 复制件 ``encode_chunk`` → ``wan-latents/<段>.bin``。

对应 motion-memory-plan.md 第一部分 4.2 第 4 步与第二部分 1.2 / 1.3。本脚本只做「读输入 → 调复制件
``encode_chunk`` → 落盘」，不复写任何 🔒 数值语句；起手 ``check_env()`` + ``pin_numerics()`` +
``check_versions()``，``load_vae(..., expected_state_sha256=9980d252…)``；B=1、33 帧一次喂。

产物（每段三件，原子落盘）：
  <out>/<段>.bin              num_grid × 589,824 B：组优先 (9,16,32,32) f32 裸字节，chunk 序 = 网格序
  <out>/<段>.bin.sha256       "<sha256>  <段>.bin"
  <out>/<段>.metadata.json    schema 2 单段清单：rows[{m, seg_offset, start_global_frame, input_shape,
                              input_dtype, input_frames_sha256}]、sha256、worker 指纹与 VAE info
``input_frames_sha256`` 在紧邻 ``encode_chunk`` 调用前对最终 C 连续的 33 帧 uint8 原始字节计算（1.2 契约）。

领任务：工作项按 num_grid LPT 降序排队，``os.open(<out>/_claims/_claim_<段>, O_CREAT|O_EXCL)`` 领一项、
完成即 unlink；续跑判据 = ``.bin`` 字节数 + ``.sha256`` sidecar + metadata 可解析，残缺段先清再重做。
由 ``scripts/dataset/run_local.py --stage wan`` 每 GPU 起一个本进程（``CUDA_VISIBLE_DEVICES`` 只暴露一张卡）。

用法：
  UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=v1-store/venvs/wan HF_HOME=v1-store/cache/hf HF_HUB_OFFLINE=1 \\
  CUDA_VISIBLE_DEVICES=1 uv run --project scripts/dataset/wan --no-sync python scripts/dataset/wan/extract_wan.py \\
      --manifest <lib>/meta/episode_manifest.json --raw-dir <h5 目录> --out <lib>/wan-latents --worker gpu1
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_HERE))

import wan_common as wc  # noqa: E402

VAE_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
METADATA_SCHEMA = 2


def load_infer_module():
    spec = importlib.util.spec_from_file_location("wan_motion_infer", str(_HERE / "wan_motion_infer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def process_segment(W, torch, vae, device, item: dict, raw_dir: pathlib.Path, out_dir: pathlib.Path,
                    fingerprint: dict, vae_info: dict) -> dict:
    key = item["key"]
    frames = wc.read_segment_frames(str(raw_dir / item["h5_file"]), item["raw_ep_idx"],
                                    item["seg_start"], item["seg_len"])
    ng = item["num_grid"]
    if ng != wc.seg_num_grid(item["seg_len"]):
        raise RuntimeError(f"{key} num_grid 与公式不符")
    final = out_dir / f"{key}.bin"
    tmp = final.with_name(final.name + f".tmp.{os.getpid()}")
    rows = []
    import hashlib
    sha = hashlib.sha256()
    with open(tmp, "wb") as f:
        for m in range(ng):
            off = wc.GRID_STRIDE * m
            if off + wc.WINDOW_FRAMES > item["seg_len"]:
                raise RuntimeError(f"{key} m={m} 窗口越段（seg_len={item['seg_len']}）")
            window = np.ascontiguousarray(frames[off:off + wc.WINDOW_FRAMES])
            if window.shape != (wc.WINDOW_FRAMES, wc.FRAME_SIZE, wc.FRAME_SIZE, 3) or window.dtype != np.uint8:
                raise RuntimeError(f"{key} m={m} 窗口形制异常 {window.shape} {window.dtype}")
            in_sha = wc.sha256_bytes(window)               # 紧邻 encode_chunk 之前、对最终连续输入计算
            W.pin_numerics()
            lat = W.encode_chunk(vae, window, device)      # 🔒 复制件；(9,16,32,32) f32 on device
            blob = lat.cpu().numpy().astype(np.float32, copy=False)
            if blob.shape != wc.LAT_SHAPE or not blob.flags["C_CONTIGUOUS"]:
                blob = np.ascontiguousarray(blob, dtype=np.float32)
            b = blob.tobytes()
            if len(b) != wc.CHUNK_BYTES:
                raise RuntimeError(f"{key} m={m} latent 字节数 {len(b)} != {wc.CHUNK_BYTES}")
            f.write(b)
            sha.update(b)
            rows.append({"m": m, "seg_offset": off, "start_global_frame": item["seg_start"] + off,
                         "input_shape": list(window.shape), "input_dtype": "uint8",
                         "input_frames_sha256": in_sha})
        f.flush()
        os.fsync(f.fileno())
    size = tmp.stat().st_size
    if size != ng * wc.CHUNK_BYTES:
        raise RuntimeError(f"{key} 字节数帐不符: {size} != {ng} × {wc.CHUNK_BYTES}")
    disk_sha = wc.sha256_file(tmp)
    if disk_sha != sha.hexdigest():
        raise RuntimeError(f"{key} 读回 sha256 与写入流不符")
    os.replace(tmp, final)
    dfd = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    wc.atomic_write_bytes(out_dir / f"{key}.bin.sha256", f"{disk_sha}  {key}.bin\n".encode())
    meta = {
        "schema": METADATA_SCHEMA, "segment": key, "g": item["g"], "h5_file": item["h5_file"],
        "raw_ep_idx": item["raw_ep_idx"], "segment_kind": item["segment"],
        "seg_start_global": item["seg_start"], "seg_len": item["seg_len"],
        "num_grid": ng, "num_chunks": item["num_chunks"],
        "grid_stride": wc.GRID_STRIDE, "window_frames": wc.WINDOW_FRAMES,
        "grid_origin": wc.GRID_ORIGIN, "window_direction": wc.WINDOW_DIRECTION,
        "truncation_policy": wc.TRUNCATION_POLICY, "frame_size": wc.FRAME_SIZE,
        "latent_shape": list(wc.LAT_SHAPE), "latent_dtype": "float32", "chunk_bytes": wc.CHUNK_BYTES,
        "rows": rows, "sha256": disk_sha, "bytes": size,
        "worker": fingerprint, "vae": vae_info,
    }
    wc.write_json(out_dir / f"{key}.metadata.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", required=True, help="<lib>/wan-latents")
    ap.add_argument("--worker", required=True, help="worker 标签（如 gpu1）")
    ap.add_argument("--segments", nargs="*", default=None, help="只跑这些段键（默认全部）")
    ap.add_argument("--max-items", type=int, default=0, help="最多处理几段（探针用；0=不限）")
    args = ap.parse_args()

    import torch
    W = load_infer_module()
    pin = wc.load_source_pin(_HERE)
    W.check_env()
    W.pin_numerics()
    W.check_versions(strict=True, with_diffusers=True)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    device = torch.device("cuda")

    manifest = wc.load_manifest(args.manifest)
    items = wc.lpt_order(wc.list_segments(manifest))
    if args.segments:
        want = set(args.segments)
        items = [it for it in items if it["key"] in want]
        missing = want - {it["key"] for it in items}
        if missing:
            raise SystemExit(f"--segments 里有清单外的段: {sorted(missing)}")
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    claims = out_dir / "_claims"
    raw_dir = pathlib.Path(args.raw_dir)

    vae, vinfo = W.load_vae(VAE_ID, device, expected_state_sha256=W.VAE_STATE_SHA256_EXPECTED)
    fingerprint = wc.worker_fingerprint(_REPO_ROOT, pin["mj_repo_commit"],
                                        extra={"worker": args.worker, "module_sha256": pin["source_sha256"]})
    print(f"[extract] worker={args.worker} gpu={fingerprint['gpu_name']} uuid={fingerprint['gpu_uuid']} "
          f"torch={fingerprint['torch']} cudnn={fingerprint['cudnn_version']} vae_state={vinfo['vae_state_sha256'][:16]}… "
          f"items_total={len(items)}", flush=True)

    prog = wc.Progress("extract")
    skipped = 0
    for it in items:
        if args.max_items and prog.items >= args.max_items:
            break
        key = it["key"]
        expect = it["num_grid"] * wc.CHUNK_BYTES
        if wc.segment_outputs_complete(out_dir, key, expect):
            skipped += 1
            continue
        if not wc.try_claim(claims, key, args.worker):
            continue
        try:
            wc.purge_segment_outputs(out_dir, key)
            t0 = time.perf_counter()
            W.pin_numerics()
            process_segment(W, torch, vae, device, it, raw_dir, out_dir, fingerprint, vinfo)
            prog.done(key, it["num_grid"], time.perf_counter() - t0)
        finally:
            wc.release_claim(claims, key)
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    print(f"WORKER_DONE stage=wan worker={args.worker} items={prog.items} windows={prog.windows} "
          f"skipped={skipped} elapsed={time.perf_counter() - prog.t0:.0f}s peak_mib={peak:.0f}", flush=True)


if __name__ == "__main__":
    main()
