#!/usr/bin/env python3
"""独立 Torch 环境的一窗真实提取与原源码复算；不启动在线服务。"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from pair4_fixture import CHECKPOINT
from pair4_fixture import CHECKPOINT_SHA
from pair4_fixture import FRAME_COUNT
from pair4_fixture import H5_NAME
from pair4_fixture import LAYOUT
from pair4_fixture import MAIN_REPO
from pair4_fixture import PARENT
from pair4_fixture import PARENT_MANIFEST_SHA
from pair4_fixture import REGISTERED_RUN
from pair4_fixture import REPO
from pair4_fixture import SEGMENT
from pair4_fixture import array_sha
from pair4_fixture import checked_root
from pair4_fixture import import_script
from pair4_fixture import load_fixture
from pair4_fixture import read_json
from pair4_fixture import sha256_file
from pair4_fixture import write_new_json


def _state_sha(encoder) -> dict[str, str]:
    return {key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in sorted(encoder.state_dict().items())}


def _check_encoder(info: dict, run: Path) -> None:
    expected = {"checkpoint": CHECKPOINT, "checkpoint_sha256": CHECKPOINT_SHA, "checkpoint_epoch": 6,
                "arch": "wan-latent-v10-st", "state_key": "encoder", "motion_dims": [8, 256]}
    if Path(info["run_dir"]).resolve() != run or any(info.get(k) != v for k, v in expected.items()):
        raise ValueError("实际encoder身份不是已批准的external epoch6/raw8合同")


def _independent_inputs(root: Path, manifest: dict) -> tuple[np.ndarray, dict]:
    """独立重读父PKL和staged H5，重算段、窗口、元数据与文件摘要。"""
    import h5py

    fixture = root / "fixture"
    latents = fixture / "wan-latents"
    inputs = read_json(fixture / "meta/input_manifest.json")
    total = read_json(latents / "metadata.json")
    segment = read_json(latents / f"{SEGMENT}.metadata.json")
    raw_meta = read_json(root / "raw8" / f"{SEGMENT}.metadata.json")
    frames = np.empty((FRAME_COUNT, 256, 256, 3), np.uint8)
    frame_mismatches, metadata_mismatches = 0, 0
    metadata_checks = 0

    def check(ok) -> None:
        nonlocal metadata_checks, metadata_mismatches
        metadata_checks += 1
        metadata_mismatches += int(not ok)

    with h5py.File(fixture / "raw" / H5_NAME, "r") as h5:
        for t in range(FRAME_COUNT):
            parent_bytes = (PARENT / "source/data" / f"{t}.pkl").read_bytes()
            sample = pickle.loads(parent_bytes)
            image = sample["image"]
            staged = h5[f"episode_0/timestep_{t}/obs/front_rgb"][:]
            frame_mismatches += int(image.dtype != np.uint8 or image.shape != (256, 256, 3)
                                    or staged.dtype != image.dtype or staged.shape != image.shape
                                    or staged.tobytes() != image.tobytes())
            frames[t] = image
            check((fixture / "source/data" / f"{t}.pkl").read_bytes() == parent_bytes)
            check(hashlib.sha256(parent_bytes).hexdigest() == inputs["frames"][t]["parent_pkl_sha256"])
            check(array_sha(image) == inputs["frames"][t]["image_sha256"])
    check(len(manifest["episodes"]) == 1)
    ep = manifest["episodes"][0]
    check((ep["global_episode_idx"], ep["raw_ep_idx"], ep["num_timesteps"], ep["exec_start_idx"]) == (0, 0, 33, 0))
    # 独立公式不用生产 list_segments；分别枚举 demo 与 exec 的完整窗口。
    expected_windows = []
    for kind, begin, length, minimum in (("demo", 0, ep["exec_start_idx"], 17),
                                          ("exec", ep["exec_start_idx"], ep["num_timesteps"] - ep["exec_start_idx"], 33)):
        for m, offset in enumerate(range(0, max(0, length - minimum + 1), 16)):
            real = min(33, length - offset)
            expected_windows.append((kind, m, begin + offset, real, 33 - real))
    check(expected_windows == [("exec", 0, 0, 33, 0)])
    check(set(total["segments"]) == {SEGMENT})
    check(total["totals"] == {"segments": 1, "windows": 1, "bytes": 589824})
    check(total["manifest_sha256"] == manifest["sha256"] == inputs["manifest_sha256"])
    check(inputs["parent_manifest_sha256"] == PARENT_MANIFEST_SHA)
    check(array_sha(frames) == inputs["input_frames_sha256"])
    for obj in (inputs, total, manifest):
        check(Path(obj["raw_dir"]).resolve() == (fixture / "raw").resolve())
    check(total["demo_min_real_frames"] == 17 and total["exec_min_real_frames"] == 33
          and total["demo_tail_pad"] == "repeat_last")
    for key, value in {"schema": 3, "segment": SEGMENT, "g": 0, "h5_file": H5_NAME, "raw_ep_idx": 0,
                       "segment_kind": "exec", "seg_start_global": 0, "seg_len": 33, "num_grid": 1,
                       "num_chunks": 1, "min_real": 33, "latent_shape": [9, 16, 32, 32],
                       "latent_dtype": "float32", "chunk_bytes": 589824}.items():
        check(segment.get(key) == value)
    for key, value in {"grid_stride": 16, "window_frames": 33, "grid_origin": "segment_start",
                       "window_direction": "forward", "truncation_policy": "none", "frame_size": 256}.items():
        check(total.get(key) == value and segment.get(key) == value)
    expected_row = {"m": 0, "seg_offset": 0, "start_global_frame": 0,
                    "input_shape": [33, 256, 256, 3], "input_dtype": "uint8",
                    "input_frames_sha256": array_sha(frames), "real_frames": 33,
                    "pad_frames": 0, "pad_source_frame": None}
    check(segment["rows"] == [expected_row])
    check(total["segments"][SEGMENT]["rows"] == [expected_row])
    for key in ("num_grid", "seg_len", "seg_start_global", "segment_kind", "g", "sha256", "bytes", "min_real"):
        check(total["segments"][SEGMENT][key] == segment[key])
    for path, meta, expected_bytes in ((latents / f"{SEGMENT}.bin", segment, 589824),
                                        (root / "raw8" / f"{SEGMENT}.f32.bin", raw_meta, 8192)):
        actual_sha = sha256_file(path)
        check(path.stat().st_size == expected_bytes == meta["bytes"])
        check(actual_sha == meta["sha256"])
        check(path.with_name(path.name + ".sha256").read_text().split() == [actual_sha, path.name])
    latent_sha = sha256_file(latents / f"{SEGMENT}.bin")
    check(raw_meta["input_latent_sha256"] == latent_sha)
    check(raw_meta["input_block_sha256"] == [latent_sha])
    check(Path(raw_meta["input_latent_bin"]).resolve() == (latents / f"{SEGMENT}.bin").resolve())
    check(raw_meta["layout"] == LAYOUT and raw_meta["row_shape"] == [8, 256]
          and raw_meta["dtype"] == "float32" and raw_meta["num_grid"] == 1)
    audit = {"frame_mismatches": frame_mismatches, "metadata_mismatches": metadata_mismatches,
             "checked_frames": FRAME_COUNT, "metadata_checks": metadata_checks,
             "metadata_rows_checked": len(expected_windows), "input_frames_sha256": array_sha(frames)}
    if frame_mismatches or metadata_mismatches:
        write_new_json(fixture / "oracle/input_failure.json", audit)
        raise ValueError(f"独立单窗输入核验失败：{audit}")
    return frames, audit


def run_source(root: Path, *, encoder_run_dir: Path, checkpoint: str,
               expected_ckpt_sha256: str, max_windows: int, progress: dict | None = None) -> dict:
    started = time.perf_counter()
    progress = {} if progress is None else progress
    progress["stage"] = "preflight"
    root = checked_root(root)
    run = encoder_run_dir.resolve()
    if max_windows != 1 or run != REGISTERED_RUN or checkpoint != CHECKPOINT or expected_ckpt_sha256 != CHECKPOINT_SHA:
        raise ValueError("本入口仅接受已批准注册权重的一窗冒烟")
    record = load_fixture(root)
    progress.setdefault("gates", {})["REGISTERED_SOURCE"] = {"status": "PASS", **record["registration"]}
    paths = {k: Path(v) for k, v in record["paths"].items()}
    for target in (paths["latents"], paths["raw8"], paths["oracle"]):
        if target.exists():
            raise FileExistsError(f"新提取目录必须为空且未创建，拒绝缓存命中或覆盖：{target}")
    wan_dir = REPO / "scripts/dataset/wan"
    sys.path.insert(0, str(wan_dir))
    import extract_wan
    import torch
    import wan_common as wc

    manifest = wc.load_manifest(paths["manifest"])
    items = wc.list_segments(manifest)
    if len(items) != 1 or items[0]["key"] != SEGMENT or items[0]["num_grid"] != 1:
        raise ValueError("夹具不是唯一完整的33帧exec窗口")
    pin = wc.load_source_pin(wan_dir)
    infer = extract_wan.load_infer_module()
    infer.check_env()
    infer.pin_numerics()
    infer.check_versions(strict=True, with_diffusers=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("本次离线阶段必须仅暴露一张CUDA卡")
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    torch.cuda.init()  # 设备枚举不初始化上下文；显存统计要求CUDA已初始化。
    torch.cuda.reset_peak_memory_stats(device)
    fingerprint = wc.worker_fingerprint(REPO, pin["mj_repo_commit"],
                                        extra={"worker": "paired4-one-window", "module_sha256": pin["source_sha256"]})
    for target in (paths["latents"], paths["raw8"], paths["oracle"]):
        target.mkdir(parents=True)
    progress["stage"] = "source_vae"
    vae, vae_info = infer.load_vae(extract_wan.VAE_ID, device, expected_state_sha256=infer.VAE_STATE_SHA256_EXPECTED)
    extract_wan.process_segment(infer, torch, vae, device, items[0], paths["raw_dir"], paths["latents"], fingerprint, vae_info)
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    latent_path = paths["latents"] / f"{SEGMENT}.bin"
    latent = np.fromfile(latent_path, dtype="<f4").reshape(9, 16, 32, 32)
    progress["stage"] = "source_encoder"
    encoder, encoder_info, use_amp = infer.load_encoder(str(run), checkpoint, device, expected_sha256=expected_ckpt_sha256)
    _check_encoder(encoder_info, run)
    encoder_state_sha = _state_sha(encoder)
    infer.pin_numerics()
    raw = infer.motion_token(encoder, torch.from_numpy(latent), use_amp, device)
    if raw.dtype != np.float32 or raw.shape != (8, 256) or not np.isfinite(raw).all():
        raise ValueError("提取接口未返回有限fp32原始8×256")
    raw_path = paths["raw8"] / f"{SEGMENT}.f32.bin"
    wc.atomic_write_bytes(raw_path, np.ascontiguousarray(raw).tobytes())
    raw_sha = sha256_file(raw_path)
    wc.atomic_write_bytes(raw_path.with_name(raw_path.name + ".sha256"), f"{raw_sha}  {raw_path.name}\n".encode())
    wc.write_json(paths["raw8"] / f"{SEGMENT}.metadata.json", {
        "schema": 1, "segment": SEGMENT, "g": 0, "segment_kind": "exec", "num_grid": 1,
        "layout": LAYOUT, "row_shape": [8, 256], "dtype": "float32", "token_dim": 2048, "token_bytes": 8192,
        "input_latent_bin": str(latent_path), "input_latent_sha256": sha256_file(latent_path),
        "input_block_sha256": [array_sha(latent)], "sha256": raw_sha, "bytes": raw.nbytes,
        "encoder": {k: v for k, v in encoder_info.items() if k != "flags"},
        "encoder_flags": encoder_info.get("flags"), "encoder_state_sha256": encoder_state_sha,
        "worker": fingerprint,
    })
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    aggregate = import_script("pair4_original_aggregate", REPO / "scripts/dataset/run_local.py")
    aggregate.aggregate_segments(paths["latents"], paths["manifest"], "wan")
    aggregate.aggregate_segments(paths["raw8"], paths["manifest"], "encode")

    # 总metadata先封口；独立读输入与原源码复算都发生在封口以后。
    metadata_sha = sha256_file(paths["latents"] / "metadata.json")
    progress["stage"] = "oracle_inputs"
    parent_frames, audit = _independent_inputs(root, manifest)
    original_path = Path(pin["mj_repo_path"]) / pin["source_path"]
    if sha256_file(original_path) != pin["upstream_source_sha256"]:
        raise ValueError("oracle原源码不是SOURCE_PIN中锁定的上游文件")
    original = import_script("pair4_independent_upstream_infer", original_path)
    if Path(original.__file__).resolve() != original_path.resolve():
        raise ValueError("独立oracle错误命中同名模块缓存")
    original.check_env()
    original.pin_numerics()
    original.check_versions(strict=True, with_diffusers=True)
    progress["stage"] = "oracle_vae"
    vae, oracle_vae_info = original.load_vae(extract_wan.VAE_ID, device,
                                            expected_state_sha256=original.VAE_STATE_SHA256_EXPECTED)
    windows_encoded = 0
    oracle_latent = original.encode_chunk(vae, parent_frames, device).cpu().numpy()
    windows_encoded += 1
    latent_mismatches = int(oracle_latent.dtype != latent.dtype or oracle_latent.shape != latent.shape
                            or oracle_latent.tobytes() != latent.tobytes())
    wc.atomic_write_bytes(paths["oracle"] / f"{SEGMENT}.bin", oracle_latent.tobytes())
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    original.pin_numerics()
    progress["stage"] = "oracle_encoder"
    encoder, oracle_encoder_info, use_amp = original.load_encoder(str(run), checkpoint, device,
                                                                  expected_sha256=expected_ckpt_sha256)
    _check_encoder(oracle_encoder_info, run)
    oracle_state_sha = _state_sha(encoder)
    if oracle_state_sha != encoder_state_sha:
        raise ValueError("独立oracle严格加载的encoder逐张量指纹不同")
    original.pin_numerics()
    oracle_raw = original.motion_token(encoder, torch.from_numpy(np.ascontiguousarray(oracle_latent)), use_amp, device)
    if oracle_raw.dtype != np.float32 or oracle_raw.shape != (2048,):
        raise ValueError("上游原始输出不符合flat2048合同")
    oracle_raw = oracle_raw.reshape(8, 256)
    encoder_mismatches = int(oracle_raw.tobytes() != raw.tobytes())
    wc.atomic_write_bytes(paths["oracle"] / "motion_token.f32.bin", oracle_raw.tobytes())
    report = {**audit, "manifest_sha256": manifest["sha256"], "raw_dir": str(paths["raw_dir"]),
              "scope": "smoke_one_window", "checked_windows": audit["metadata_rows_checked"],
              "windows": windows_encoded, "windows_encoded": windows_encoded,
              "sample_spec": None, "demo_min_real_frames": 17, "exec_min_real_frames": 33,
              "tested_metadata_sha256": metadata_sha, "latent_mismatches": latent_mismatches,
              "encoder_mismatches": encoder_mismatches,
              "segments": {SEGMENT: {"num_grid": 1, "sha256": array_sha(oracle_latent),
                                      "frame_mismatches": audit["frame_mismatches"]}},
              "vae_provenance": oracle_vae_info, "encoder_provenance": oracle_encoder_info,
              "encoder_state_sha256": oracle_state_sha, "orig_module_sha256": sha256_file(original_path),
              "mj_repo": pin["mj_repo_path"], "elapsed_s": time.perf_counter() - started}
    write_new_json(paths["oracle"] / "vae_report.json", report)
    if latent_mismatches or encoder_mismatches:
        raise ValueError("独立原源码的latent/raw8数值复算未逐位通过")
    if sha256_file(paths["latents"] / "metadata.json") != metadata_sha:
        raise ValueError("oracle执行过程中总metadata发生变化")
    progress["stage"] = "seal_report"
    artifacts = {str(path.relative_to(root)): sha256_file(path)
                 for directory in (paths["latents"], paths["raw8"], paths["oracle"])
                 for path in sorted(directory.iterdir()) if path.is_file()}
    artifacts["fixture/fixture_manifest.json"] = sha256_file(root / "fixture/fixture_manifest.json")
    result = {"status": "PASS", "manifest_sha256": manifest["sha256"], "artifacts_sha256": artifacts,
              "gates": {"REGISTERED_SOURCE": {"status": "PASS", **record["registration"]},
                        "SOURCE_FRESH": {"status": "PASS", "checked_windows": 1, **audit,
                                         "latent_mismatches": latent_mismatches, "encoder_mismatches": encoder_mismatches}},
              "input_frames_sha256": array_sha(parent_frames), "latent_sha256": array_sha(latent),
              "raw8_sha256": raw_sha, "source_pin": pin, "worker": fingerprint,
              "elapsed_s": time.perf_counter() - started, "peak_cuda_mib": torch.cuda.max_memory_allocated(device) / 2**20,
              "paths": record["paths"]}
    write_new_json(root / "reports/source-smoke.json", result)
    del encoder
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, required=True)
    parser.add_argument("--encoder-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-ckpt-sha256", required=True)
    args = parser.parse_args()
    # 在导入diffusers前固定现有本机缓存，缺资产即离线报错，不暗中下载。
    os.environ.setdefault("HF_HOME", str(MAIN_REPO / "v1-store/cache/hf"))
    os.environ.setdefault("XDG_CACHE_HOME", str(MAIN_REPO / "v1-store/cache/xdg"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    started = time.perf_counter()
    progress = {"stage": "preflight", "gates": {
        "REGISTERED_SOURCE": {"status": "NOT_RUN"}, "SOURCE_FRESH": {"status": "NOT_RUN"}}}
    try:
        result = run_source(**vars(args), progress=progress)
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        progress["gates"]["SOURCE_FRESH"] = {"status": "FAIL", "reason": reason}
        peak = None
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            with contextlib.suppress(Exception):
                peak = torch.cuda.max_memory_allocated() / 2**20
        failure = {"status": "FAIL", **progress, "reason": reason,
                   "elapsed_s": time.perf_counter() - started, "peak_cuda_mib": peak}
        target = checked_root(args.root) / "reports/source-smoke.json"
        if not target.exists():
            write_new_json(target, failure)
        print(f"SOURCE_FRESH=FAIL stage={progress['stage']} reason={reason}", flush=True)
        raise
    print(f"SOURCE_FRESH=PASS checked_windows=1 raw8_bytes=8192 elapsed_s={result['elapsed_s']:.3f}", flush=True)


if __name__ == "__main__":
    main()
