#!/usr/bin/env python3
"""MotionJEPA encoder 编码 worker：按段领任务，读 ``wan-latents/<段>.bin`` 每块 → 复制件 ``motion_token`` →
``motion-tokens/<段>.f32.bin``（num_grid × 768 f32）。

对应 motion-memory-plan.md 第一部分 4.2 第 5 步与第二部分 1.5。只做「读 latent → 调复制件 ``motion_token`` → 落盘」，
不复写任何 🔒 数值语句；起手 ``check_env()`` + ``pin_numerics()`` + ``check_versions()``；
``load_encoder(run_dir, ckpt, expected_sha256=…)`` 整份 strict（affine buffer 从 ckpt 带入，禁 ``load_wan_latent_stats``）；
B=1 硬约束；autocast 由 run 配置决定（bf16）。CLI 刻意不设 ``--encoder-key`` / ``--tf32`` / ``--amp``。

产物（每段三件，原子落盘）：
  <out>/<段>.f32.bin            num_grid × 3,072 B，行序 = 网格序
  <out>/<段>.f32.bin.sha256
  <out>/<段>.metadata.json      输入 .bin sha256、每行 latent 块 sha256、encoder info、worker 指纹

用法：
  UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=v1-store/venvs/wan CUDA_VISIBLE_DEVICES=1 \\
  uv run --project scripts/dataset/wan --no-sync python scripts/dataset/wan/encode_motion.py \\
      --manifest <lib>/meta/episode_manifest.json --latents <lib>/wan-latents --out <lib>/motion-tokens \\
      --encoder-run-dir v1-store/external/motionjepa/wan-v8-filter10-72ep-a --checkpoint checkpoint_epoch_72.pt \\
      --expected-ckpt-sha256 <sha> --worker gpu1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

METADATA_SCHEMA = 1


def load_infer_module():
    spec = importlib.util.spec_from_file_location("wan_motion_infer", str(_HERE / "wan_motion_infer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_latent_bin(path: pathlib.Path, num_grid: int) -> np.ndarray:
    size = path.stat().st_size
    if size != num_grid * wc.CHUNK_BYTES:
        raise RuntimeError(f"{path} 字节数 {size} != {num_grid} × {wc.CHUNK_BYTES}")
    side = path.with_name(path.name + ".sha256")
    want = side.read_text().split()[0]
    got = wc.sha256_file(path)
    if got != want:
        raise RuntimeError(f"{path} sha256 {got[:16]}… != sidecar {want[:16]}…")
    arr = np.fromfile(path, dtype=np.float32)
    arr.shape = (num_grid,) + wc.LAT_SHAPE
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--latents", required=True, help="<lib>/wan-latents")
    ap.add_argument("--out", required=True, help="<lib>/motion-tokens")
    ap.add_argument("--encoder-run-dir", required=True)
    ap.add_argument("--checkpoint", default="checkpoint_epoch_72.pt")
    ap.add_argument("--expected-ckpt-sha256", required=True)
    ap.add_argument("--worker", required=True)
    ap.add_argument("--segments", nargs="*", default=None)
    args = ap.parse_args()

    import torch
    W = load_infer_module()
    pin = wc.load_source_pin(_HERE)
    W.check_env()
    W.pin_numerics()
    W.check_versions(strict=True, with_diffusers=False)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    device = torch.device("cuda")

    manifest = wc.load_manifest(args.manifest)
    items = wc.lpt_order(wc.list_segments(manifest))
    if args.segments:
        want = set(args.segments)
        items = [it for it in items if it["key"] in want]
    lat_dir = pathlib.Path(args.latents)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    claims = out_dir / "_claims"

    encoder, einfo, use_amp = W.load_encoder(args.encoder_run_dir, args.checkpoint, device,
                                             expected_sha256=args.expected_ckpt_sha256)
    # 77 张量 sha256 清单（D3 附加判据：两侧逐键相等）
    sd = encoder.state_dict()
    state_sha = {k: hashlib.sha256(sd[k].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                 for k in sorted(sd)}
    fingerprint = wc.worker_fingerprint(_REPO_ROOT, pin["mj_repo_commit"],
                                        extra={"worker": args.worker, "module_sha256": pin["source_sha256"]})
    print(f"[encode] worker={args.worker} gpu={fingerprint['gpu_name']} ckpt={einfo['checkpoint_sha256'][:16]}… "
          f"epoch={einfo['checkpoint_epoch']} amp={einfo['amp']} tensors={len(state_sha)} items_total={len(items)}",
          flush=True)

    prog = wc.Progress("encode")
    skipped = 0
    for it in items:
        key = it["key"]
        ng = it["num_grid"]
        final = out_dir / f"{key}.f32.bin"
        if wc.segment_outputs_complete(out_dir, key + ".f32", ng * wc.TOKEN_BYTES) \
                and (out_dir / f"{key}.metadata.json").is_file():
            skipped += 1
            continue
        if not wc.try_claim(claims, key, args.worker):
            continue
        try:
            for suffix in (".f32.bin", ".f32.bin.sha256", ".metadata.json"):
                (out_dir / f"{key}{suffix}").unlink(missing_ok=True)
            t0 = time.perf_counter()
            lat_path = lat_dir / f"{key}.bin"
            lats = read_latent_bin(lat_path, ng)
            toks = np.empty((ng, wc.TOKEN_DIM), dtype=np.float32)
            block_sha = []
            for m in range(ng):
                blk = torch.from_numpy(np.ascontiguousarray(lats[m]))      # (9,16,32,32) f32
                block_sha.append(wc.sha256_bytes(lats[m]))
                W.pin_numerics()
                toks[m] = W.motion_token(encoder, blk, use_amp, device)     # 🔒 复制件；B=1
            blob = np.ascontiguousarray(toks).tobytes()
            if len(blob) != ng * wc.TOKEN_BYTES:
                raise RuntimeError(f"{key} token 字节数 {len(blob)} != {ng} × {wc.TOKEN_BYTES}")
            wc.atomic_write_bytes(final, blob)
            sha = wc.sha256_file(final)
            wc.atomic_write_bytes(out_dir / f"{key}.f32.bin.sha256", f"{sha}  {key}.f32.bin\n".encode())
            wc.write_json(out_dir / f"{key}.metadata.json", {
                "schema": METADATA_SCHEMA, "segment": key, "g": it["g"], "segment_kind": it["segment"],
                "num_grid": ng, "token_dim": wc.TOKEN_DIM, "token_bytes": wc.TOKEN_BYTES,
                "input_latent_bin": str(lat_path), "input_latent_sha256": wc.sha256_file(lat_path),
                "input_block_sha256": block_sha, "sha256": sha, "bytes": len(blob),
                "encoder": {k: v for k, v in einfo.items() if k != "flags"},
                "encoder_flags": einfo.get("flags"),
                "encoder_state_sha256": state_sha,
                "worker": fingerprint,
            })
            prog.done(key, ng, time.perf_counter() - t0)
        finally:
            wc.release_claim(claims, key)
    print(f"WORKER_DONE stage=encode worker={args.worker} items={prog.items} windows={prog.windows} "
          f"skipped={skipped} elapsed={time.perf_counter() - prog.t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
