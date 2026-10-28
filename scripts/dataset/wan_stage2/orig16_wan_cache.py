#!/usr/bin/env python3
"""固定版本官方 Wan 缓存的规划、整文件恢复、逐窗导入和 G2 输入身份核验。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
from pathlib import PurePosixPath
import pickle
import re
import shutil
import socket
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "wan"))
import wan_common as wc  # noqa: E402

REPO = "HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1"
REVISION = "384d5db410be713a49357bf3f55ca4f4d2f559a0"
KIND = "published-cache-with-local-fill"
PLAN_SCHEMA = "orig16-wan-cache-plan-v1"
RECEIPT_SCHEMA = "orig16-wan-cache-import-v1"
PREPROCESS = {
    "frame_size": 256,
    "input_dtype": "uint8",
    "input_layout": "THWC",
    "window_frames": 33,
    "augmentation": "none",
}
LATENT_SHAPES = {"group_first": (9, 16, 32, 32), "channel_first": (16, 9, 32, 32)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_file(path):
    return wc.sha256_file(path)


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wc.write_json(path, value)


def safe_path(root, relative):
    """拒绝绝对路径、父目录跳转与指向根外的符号链接。"""
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or ".." in rel.parts or "\\" in relative:
        raise ValueError(f"非法相对路径: {relative}")
    root = Path(root).resolve()
    result = root.joinpath(*rel.parts)
    if not result.resolve().is_relative_to(root):
        raise ValueError(f"路径越过根目录: {relative}")
    return result


def read_sums(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if not match:
            raise ValueError("SHA256SUMS 行格式非法")
        sha, name = match.groups()
        safe_path(Path(path).parent, name)
        if name in result:
            raise ValueError(f"SHA256SUMS 路径重复: {name}")
        result[name] = sha
    return result


def check_file(path, record):
    path = Path(path)
    if path.stat().st_size != record["bytes"] or sha_file(path) != record["sha256"]:
        raise ValueError(f"文件字节数或 SHA 不符: {path}")


def check_provenance(provenance):
    layout = provenance.get("axis_order")
    if (
        layout not in LATENT_SHAPES
        or provenance.get("latent_shape") != list(LATENT_SHAPES[layout])
        or provenance.get("dtype") != "float32"
        or provenance.get("byte_order") != "little"
        or provenance.get("chunk_bytes") != wc.CHUNK_BYTES
        or provenance.get("affine_normalized") is not False
        or provenance.get("window_frames") != 33
        or provenance.get("stride") != 1
        or not isinstance(provenance.get("numeric_mode"), dict)
        or not provenance["numeric_mode"]
    ):
        raise ValueError("发布 provenance 的轴序、原始 latent 或窗口合同不符")
    return layout


def source_files(input_manifest, published):
    got = [
        {"name": name, "sha256": rec["sha256"], "bytes": rec["size"]}
        for name, rec in sorted(input_manifest["files"].items())
    ]
    want = sorted(
        (
            {"name": rec["h5"]["file"], "sha256": rec["h5"]["sha256"], "bytes": rec["h5"]["size"]}
            for rec in published["files"].values()
        ),
        key=lambda x: x["name"],
    )
    if not got or got != want or input_manifest.get("count") != len(got):
        raise ValueError("目标 input manifest 与发布源 H5 集合不相等")
    for row in got:
        if Path(row["name"]).name != row["name"] or not re.fullmatch("[0-9a-f]{64}", row["sha256"]):
            raise ValueError("源 H5 名称或 SHA 非法")
    return got


def build_plan(manifest_path, input_path, index_path):
    """仅读清单和发布索引；不读 H5 像素，也不隐式下载。"""
    import pyarrow.parquet as pq

    manifest_path, input_path, index_path = (Path(p).resolve() for p in (manifest_path, input_path, index_path))
    manifest = wc.load_manifest(manifest_path)
    root = index_path.parent
    sums = read_sums(root / "SHA256SUMS")
    names = [
        "source_manifest.json",
        "provenance.json",
        "index/episodes.parquet",
        "index/segments.parquet",
        "index/windows.parquet",
    ]
    metadata = []
    for name in names:
        path = safe_path(root, name)
        rec = {"path": name, "sha256": sums[name], "bytes": path.stat().st_size}
        check_file(path, rec)
        metadata.append(rec)
    provenance = read_json(root / "provenance.json")
    layout = check_provenance(provenance)
    sources = source_files(read_json(input_path), read_json(root / "source_manifest.json"))
    source_map = {x["name"]: x for x in sources}
    episodes = {}
    for row in pq.read_table(index_path / "episodes.parquet").to_pylist():
        key = (Path(row["h5_file"]).name, int(row["episode"]))
        if key in episodes or source_map[key[0]]["sha256"] != row["source_sha256"]:
            raise ValueError("发布 episode 重复或源 H5 SHA 不符")
        if wc.task_of_h5(key[0]) != row["task"]:
            raise ValueError("发布 episode 任务名不符")
        episodes[key] = row
    seen, globals_seen = set(), set()
    for ep in manifest["episodes"]:
        key = (ep["h5_file"], ep["raw_ep_idx"])
        published = episodes[key]
        if key in seen or ep["global_episode_idx"] in globals_seen:
            raise ValueError("目标 episode 重复")
        seen.add(key)
        globals_seen.add(ep["global_episode_idx"])
        demo_end, completed, total = (published[k] for k in ("demo_end", "completed_index", "num_timesteps"))
        if not 0 <= demo_end <= completed <= total or ep["exec_start_idx"] != demo_end or ep["num_timesteps"] != total:
            raise ValueError("目标 episode 的 D/C/T 与发布索引不符")
    if seen != set(episodes):
        raise ValueError("目标 episode 集合与固定发布不相等")
    published_segments = {}
    for row in pq.read_table(index_path / "segments.parquet").to_pylist():
        key = row["name"]
        if key in published_segments:
            raise ValueError("发布 segment 重复")
        ep = episodes[(Path(row["h5_file"]).name, row["episode"])]
        expected_lo = 0 if row["segment"] == "demo" else ep["demo_end"]
        expected_hi = ep["demo_end"] if row["segment"] == "demo" else ep["completed_index"]
        if (
            row["segment"] not in ("demo", "exec")
            or row["lo"] != expected_lo
            or row["hi"] != expected_hi
            or row["frames"] != expected_hi - expected_lo
            or row["num_chunks"] != max(0, row["frames"] - 32)
            or key != f"{row['task']}_ep{row['episode']}_{row['segment']}"
        ):
            raise ValueError(f"发布 segment 边界不符: {key}")
        if row["num_chunks"]:
            safe_path(root, row["binfile"])
        elif row["binfile"] is not None:
            raise ValueError("零完整窗口的发布段不应有 binfile")
        published_segments[key] = row
    items = wc.list_segments(manifest)
    windows, copy_lookup, files = [], {}, {}
    for item in items:
        ep = episodes[(item["h5_file"], item["raw_ep_idx"])]
        for m in range(item["num_grid"]):
            offset = m * wc.GRID_STRIDE
            start = item["seg_start"] + offset
            real = min(33, item["seg_len"] - offset)
            pad = 33 - real
            copy = real == 33 and start + 33 <= (ep["demo_end"] if item["segment"] == "demo" else ep["completed_index"])
            row = {
                "key": item["key"],
                "m": m,
                "route": "copy" if copy else "fill",
                "h5_file": item["h5_file"],
                "raw_ep_idx": item["raw_ep_idx"],
                "segment_kind": item["segment"],
                "start_global_frame": start,
                "seg_offset": offset,
                "real_frames": real,
                "pad_frames": pad,
                "pad_source_frame": start + real - 1 if pad else None,
                "frame_ids": list(range(start, start + real)) + [start + real - 1] * pad,
                "source_file": None,
                "source_row": None,
                "source_offset": None,
                "source_layout": None,
                "output_file": item["key"] + ".bin",
                "output_offset": m * wc.CHUNK_BYTES,
            }
            if copy:
                segment = published_segments[item["key"]]
                file = segment["binfile"]
                row.update(
                    source_file=file, source_row=offset, source_offset=offset * wc.CHUNK_BYTES, source_layout=layout
                )
                files[file] = {"path": file, "sha256": sums[file], "bytes": segment["num_chunks"] * wc.CHUNK_BYTES}
                copy_lookup[(item["key"], offset)] = row
            windows.append(row)
    found = set()
    for batch in pq.ParquetFile(index_path / "windows.parquet").iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            key = (f"{row['task']}_ep{row['episode']}_{row['segment']}", row["start"])
            if key not in copy_lookup:
                continue
            target = copy_lookup[key]
            if (
                key in found
                or row["binfile"] != target["source_file"]
                or row["byte_offset"] != target["source_offset"]
                or row["valid_frames"] != 33
                or row["original_frame_idx"] != target["start_global_frame"]
                or row["window_start"] != target["start_global_frame"]
                or row["window_end"] != target["start_global_frame"] + 33
            ):
                raise ValueError(f"发布窗口行映射不符: {key}")
            found.add(key)
    if found != set(copy_lookup):
        raise ValueError("发布索引缺少可复制窗口")
    counts = {
        "windows": len(windows),
        "copy": len(copy_lookup),
        "fill": len(windows) - len(copy_lookup),
        "grid32": sum((x["num_grid"] + 1) // 2 for x in items),
        "segments": len(items),
        "download_files": len(files),
        "download_bytes": sum(x["bytes"] for x in files.values()),
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "kind": KIND,
        "repo": REPO,
        "revision": REVISION,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["sha256"],
        "manifest_file_sha256": sha_file(manifest_path),
        "input_manifest_path": str(input_path),
        "input_manifest_sha256": sha_file(input_path),
        "published_root": str(root),
        "sha256sums_sha256": sha_file(root / "SHA256SUMS"),
        "published_metadata": metadata,
        "published_files": sorted(files.values(), key=lambda x: x["path"]),
        "provenance": provenance,
        "source_h5_files": sources,
        "segments": items,
        "windows": windows,
        "counts": counts,
    }
    plan["sha256"] = canonical_sha(plan)
    return plan


def load_plan(path):
    plan = read_json(path)
    body = {k: v for k, v in plan.items() if k != "sha256"}
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("repo") != REPO
        or plan.get("revision") != REVISION
        or plan.get("sha256") != canonical_sha(body)
    ):
        raise ValueError("缓存计划身份或摘要不符")
    return plan


def fetch_plan(plan, revision, out, *, downloader=None):
    """仅固定 revision，下载完整文件，已有文件仍逐字节核 SHA。"""
    if revision != REVISION or plan["revision"] != revision or plan["repo"] != REPO:
        raise ValueError("恢复必须指定固定的 40 位发布 revision")
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sums_path = out / "SHA256SUMS"
    if not sums_path.exists():
        downloaded = Path(
            downloader(repo_id=REPO, repo_type="dataset", revision=revision, filename="SHA256SUMS", local_dir=str(out))
        )
        if downloaded.resolve() != sums_path.resolve():
            shutil.copyfile(downloaded, sums_path)
    if sha_file(sums_path) != plan["sha256sums_sha256"]:
        raise ValueError("恢复的 SHA256SUMS 与计划不同")
    sums = read_sums(sums_path)
    records = plan["published_metadata"] + plan["published_files"]
    for record in records:
        name = record["path"]
        path = safe_path(out, name)
        if sums.get(name) != record["sha256"]:
            raise ValueError(f"发布 SHA256SUMS 与计划文件 SHA 不符: {name}")
        if not path.exists():
            downloaded = Path(
                downloader(repo_id=REPO, repo_type="dataset", revision=revision, filename=name, local_dir=str(out))
            )
            if downloaded.resolve() != path.resolve():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(downloaded, path)
        check_file(path, record)
    result = {
        "schema": "orig16-wan-cache-fetch-v1",
        "status": "PASS",
        "plan_sha256": plan["sha256"],
        "repo": REPO,
        "revision": revision,
        "sha256sums_sha256": plan["sha256sums_sha256"],
        "files": records,
    }
    write_json(out / "fetch-receipt.json", result)
    return result


def verify_inputs(plan, cache_root, raw_dir):
    """导入前核整文件，不能把下载收据作为当前磁盘内容的替代证明。"""
    if (
        sha_file(plan["manifest_path"]) != plan["manifest_file_sha256"]
        or sha_file(plan["input_manifest_path"]) != plan["input_manifest_sha256"]
    ):
        raise ValueError("计划生成后目标清单发生变化")
    if sha_file(Path(cache_root) / "SHA256SUMS") != plan["sha256sums_sha256"]:
        raise ValueError("缓存 SHA256SUMS 与计划不符")
    sums = read_sums(Path(cache_root) / "SHA256SUMS")
    for record in plan["published_metadata"] + plan["published_files"]:
        if sums.get(record["path"]) != record["sha256"]:
            raise ValueError("缓存文件摘要不属于固定发布")
        check_file(safe_path(cache_root, record["path"]), record)
    if read_json(Path(cache_root) / "provenance.json") != plan["provenance"]:
        raise ValueError("实际发布 provenance 与计划不符")
    rebuilt = build_plan(plan["manifest_path"], plan["input_manifest_path"], Path(cache_root) / "index")
    if any(rebuilt[k] != value for k, value in plan.items() if k not in ("sha256", "published_root")):
        raise ValueError("计划映射与当前清单及发布索引重新推导的结果不同")
    for record in plan["source_h5_files"]:
        check_file(safe_path(raw_dir, record["name"]), record)


def read_window(raw_dir, row):
    frames = wc.read_segment_frames(
        str(safe_path(raw_dir, row["h5_file"])), row["raw_ep_idx"], row["start_global_frame"], row["real_frames"]
    )
    if row["pad_frames"]:
        frames = np.concatenate((frames, np.repeat(frames[-1:], row["pad_frames"], axis=0)))
    return np.ascontiguousarray(frames)


def copy_latent(cache_root, row):
    with safe_path(cache_root, row["source_file"]).open("rb") as stream:
        stream.seek(row["source_offset"])
        blob = stream.read(wc.CHUNK_BYTES)
    if len(blob) != wc.CHUNK_BYTES:
        raise ValueError("发布源块越界")
    source_shape = LATENT_SHAPES[row["source_layout"]]
    latent = np.frombuffer(blob, dtype="<f4").reshape(source_shape)
    if row["source_layout"] == "channel_first":
        latent = latent.transpose(1, 0, 2, 3)
    return np.ascontiguousarray(latent)


def make_encoder():
    """GPU worker 内延迟导入；数值路径唯一委托既有 encode_chunk。"""
    import torch

    spec = importlib.util.spec_from_file_location("wan_motion_infer", HERE.parent / "wan/wan_motion_infer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_env()
    module.pin_numerics()
    module.check_versions(strict=True, with_diffusers=True)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise ValueError("缺窗补算需要 CUDA")
    vae, info = module.load_vae(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "cuda", expected_state_sha256=module.VAE_STATE_SHA256_EXPECTED
    )

    def encode(window):
        module.pin_numerics()
        return module.encode_chunk(vae, window, "cuda").cpu().numpy()

    return (
        encode,
        {
            "weights": "fp32",
            "amp": False,
            "batch": 1,
            "latent_mode": "mode",
            "use_tiling": False,
            "use_slicing": False,
            "flags": module.numerics_snapshot(),
            "torch": torch.__version__,
            "cudnn": torch.backends.cudnn.version(),
            "cuda": torch.version.cuda,
            "gpu_name": info["gpu_name"],
            "driver": info["driver"],
            "diffusers": info["diffusers"],
            "adapter_sha256": sha_file(HERE.parent / "wan/wan_motion_infer.py"),
        },
        info,
    )


def import_segment(plan, item, rows, cache_root, raw_dir, out, encode, fill_numeric, vae_info, worker):
    """单段顺序写入；每窗保留实际输入 SHA 与可逆布局映射。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    final = out / (item["key"] + ".bin")
    tmp = final.with_name(final.name + f".tmp.{os.getpid()}")
    frames = wc.read_segment_frames(
        str(safe_path(raw_dir, item["h5_file"])), item["raw_ep_idx"], item["seg_start"], item["seg_len"]
    )
    records, stream_sha = [], hashlib.sha256()
    try:
        with tmp.open("wb") as stream:
            for row in rows:
                window = frames[row["seg_offset"] : row["seg_offset"] + row["real_frames"]]
                if row["pad_frames"]:
                    window = np.concatenate((window, np.repeat(window[-1:], row["pad_frames"], axis=0)))
                window = np.ascontiguousarray(window)
                input_sha = wc.sha256_bytes(window)
                latent = copy_latent(cache_root, row) if row["route"] == "copy" else encode(window)
                if latent.shape != wc.LAT_SHAPE or latent.dtype != np.dtype("float32") or not np.isfinite(latent).all():
                    raise ValueError("导入 latent 形状、dtype 或 finite 不符")
                blob = np.ascontiguousarray(latent, dtype="<f4").tobytes()
                stream.write(blob)
                stream_sha.update(blob)
                records.append(
                    {
                        **row,
                        "input_shape": [33, 256, 256, 3],
                        "input_dtype": "uint8",
                        "input_frames_sha256": input_sha,
                        "latent_sha256": wc.sha256_bytes(blob),
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())
        expected = {"bytes": item["num_grid"] * wc.CHUNK_BYTES, "sha256": stream_sha.hexdigest()}
        check_file(tmp, expected)
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    wc.atomic_write_bytes(out / f"{item['key']}.bin.sha256", f"{expected['sha256']}  {final.name}\n".encode())
    meta = {
        "schema": 3,
        "segment": item["key"],
        "g": item["g"],
        "h5_file": item["h5_file"],
        "raw_ep_idx": item["raw_ep_idx"],
        "segment_kind": item["segment"],
        "seg_start_global": item["seg_start"],
        "seg_len": item["seg_len"],
        "num_grid": item["num_grid"],
        "num_chunks": item["num_chunks"],
        "min_real": item["min_real"],
        "grid_stride": 16,
        "window_frames": 33,
        "grid_origin": "segment_start",
        "window_direction": "forward",
        "truncation_policy": "none",
        "frame_size": 256,
        "latent_shape": list(wc.LAT_SHAPE),
        "latent_dtype": "float32",
        "chunk_bytes": wc.CHUNK_BYTES,
        "rows": records,
        **expected,
        "worker": worker,
        "vae": vae_info,
        "numeric_mode": fill_numeric,
        "plan_sha256": plan["sha256"],
        "importer_sha256": sha_file(__file__),
    }
    write_json(out / f"{item['key']}.metadata.json", meta)
    return meta


def worker_import(plan_path, cache_root, raw_dir, out, gpu, keys):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    plan = load_plan(plan_path)
    grouped = {key: [] for key in keys}
    for row in plan["windows"]:
        if row["key"] in grouped:
            grouped[row["key"]].append(row)
    encode, numeric, info = None, {}, {}
    if any(r["route"] == "fill" for rows in grouped.values() for r in rows):
        encode, numeric, info = make_encoder()
        if (
            info["vae_id"] != plan["provenance"]["vae_id"]
            or info["vae_state_sha256"] != plan["provenance"]["vae_state_sha256"]
        ):
            raise ValueError("缺窗 VAE 与发布缓存权重不同源")
    worker = {
        "worker": f"gpu{gpu}",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "gpu_uuid": wc.gpu_uuid_of_visible(),
        "cuda_visible_devices": str(gpu),
    }
    for item in plan["segments"]:
        if item["key"] in grouped:
            import_segment(plan, item, grouped[item["key"]], cache_root, raw_dir, out, encode, numeric, info, worker)
            print(f"WAN_CACHE_SEGMENT=PASS key={item['key']} windows={item['num_grid']} worker=gpu{gpu}", flush=True)


def aggregate_import(plan, out, raw_dir):
    out = Path(out).resolve()
    records, files, segments, workers, modes = [], [], {}, {}, []
    expected = {(r["key"], r["m"]): r for r in plan["windows"]}
    for item in plan["segments"]:
        meta = read_json(out / f"{item['key']}.metadata.json")
        if meta.get("plan_sha256") != plan["sha256"] or meta.get("importer_sha256") != sha_file(__file__):
            raise ValueError("分段产物来自不同计划或导入器")
        file = {"path": item["key"] + ".bin", "sha256": meta["sha256"], "bytes": meta["bytes"]}
        check_file(out / file["path"], file)
        if len(meta["rows"]) != item["num_grid"] or file["bytes"] != item["num_grid"] * wc.CHUNK_BYTES:
            raise ValueError("分段行数不符")
        with (out / file["path"]).open("rb") as stream:
            for m, row in enumerate(meta["rows"]):
                want = expected[(item["key"], m)]
                if any(row.get(k) != v for k, v in want.items()):
                    raise ValueError("逐窗映射与计划不符")
                blob = stream.read(wc.CHUNK_BYTES)
                if (
                    wc.sha256_bytes(blob) != row["latent_sha256"]
                    or not np.isfinite(np.frombuffer(blob, dtype="<f4")).all()
                ):
                    raise ValueError("逐窗 latent SHA 或 finite 不符")
                records.append(row)
        segments[item["key"]] = {
            k: meta[k]
            for k in (
                "num_grid",
                "seg_len",
                "seg_start_global",
                "segment_kind",
                "g",
                "rows",
                "sha256",
                "bytes",
                "min_real",
            )
        }
        segments[item["key"]]["worker"] = meta["worker"]["worker"]
        workers[meta["worker"]["worker"]] = meta["worker"]
        if meta["numeric_mode"]:
            modes.append(meta["numeric_mode"])
        files.append(file)
    if len(records) != plan["counts"]["windows"] or any(x != modes[0] for x in modes):
        raise ValueError("导入窗口计数或 worker 数值配置不一致")
    metadata = {
        "schema": 3,
        "stage": "wan",
        "grid_stride": 16,
        "window_frames": 33,
        "frame_size": 256,
        "demo_min_real_frames": 17,
        "exec_min_real_frames": 33,
        "demo_tail_pad": "repeat_last",
        "grid_origin": "segment_start",
        "window_direction": "forward",
        "truncation_policy": "none",
        "raw_dir": str(Path(raw_dir).resolve()),
        "manifest_sha256": plan["manifest_sha256"],
        "manifest_path": plan["manifest_path"],
        "segments": segments,
        "workers": list(workers.values()),
        "totals": {"segments": len(segments), "windows": len(records), "bytes": sum(x["bytes"] for x in files)},
    }
    write_json(out / "metadata.json", metadata)
    provenance = plan["provenance"]
    return {
        "schema": RECEIPT_SCHEMA,
        "kind": KIND,
        "status": "PASS",
        "published": {
            "repo": REPO,
            "revision": REVISION,
            "sha256sums_sha256": plan["sha256sums_sha256"],
            "files": plan["published_metadata"] + plan["published_files"],
        },
        "source_h5_files": plan["source_h5_files"],
        **{
            k: plan[k]
            for k in (
                "manifest_path",
                "manifest_sha256",
                "manifest_file_sha256",
                "input_manifest_path",
                "input_manifest_sha256",
            )
        },
        "importer_sha256": sha_file(__file__),
        "plan_sha256": plan["sha256"],
        "latents": {"root": str(out), "metadata_sha256": sha_file(out / "metadata.json"), "files": files},
        "vae": {
            "vae_id": provenance["vae_id"],
            "vae_state_sha256": provenance["vae_state_sha256"],
            "latent_mode": "mode",
        },
        "preprocess": PREPROCESS,
        "numeric_modes": {"copy": provenance["numeric_mode"], "fill": modes[0] if modes else {"status": "NOT_USED"}},
        "windows": records,
        "counts": {k: plan["counts"][k] for k in ("windows", "copy", "fill")},
    }


def import_plan(plan_path, cache_root, raw_dir, out, receipt_path, gpus):
    plan = load_plan(plan_path)
    if Path(receipt_path).exists():
        raise FileExistsError("导入收据已存在，拒绝覆盖")
    gpu_ids = gpus.split(",")
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(x not in list("01234567") for x in gpu_ids):
        raise ValueError("GPU 必须是 0–7 范围内互不重复的编号")
    verify_inputs(plan, cache_root, raw_dir)
    # 每段完整归属一个 worker，避免共享写入同一个目标文件。
    shards, loads = [[] for _ in gpu_ids], [0 for _ in gpu_ids]
    for item in wc.lpt_order(plan["segments"]):
        index = min(range(len(loads)), key=loads.__getitem__)
        shards[index].append(item["key"])
        loads[index] += item["num_grid"]
    ctx = multiprocessing.get_context("spawn")
    workers = [
        ctx.Process(target=worker_import, args=(str(plan_path), str(cache_root), str(raw_dir), str(out), gpu, keys))
        for gpu, keys in zip(gpu_ids, shards, strict=True)
        if keys
    ]
    for worker in workers:
        worker.start()
    try:
        for worker in workers:
            worker.join()
        if any(worker.exitcode for worker in workers):
            raise RuntimeError(f"导入 worker 失败: {[w.exitcode for w in workers]}")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
    receipt = aggregate_import(plan, out, raw_dir)
    validate_receipt(receipt)
    write_json(receipt_path, receipt)
    return receipt


def validate_receipt(receipt):
    # 生产消费与工具共用唯一合同，禁止依赖测试夹具。
    sys.path.insert(0, str(ROOT / "src"))
    from mme_vla_suite.datastore.motion_store import validate_wan_cache_import_receipt

    return validate_wan_cache_import_receipt(receipt)


def check_import(samples_path, receipt_path, raw_dir):
    receipt, samples = read_json(receipt_path), read_json(samples_path)
    mapping = validate_receipt(receipt)
    receipt_sha = sha_file(receipt_path)
    if (
        samples.get("gate") != "G0"
        or samples.get("status") != "PASS"
        or samples.get("cache_import_receipt_sha256") != receipt_sha
        or samples.get("manifest_sha256") != receipt["manifest_sha256"]
        or samples.get("latent_metadata_sha256") != receipt["latents"]["metadata_sha256"]
    ):
        raise ValueError("G0 与导入收据身份不符")
    if (
        sha_file(receipt["manifest_path"]) != receipt["manifest_file_sha256"]
        or sha_file(receipt["input_manifest_path"]) != receipt["input_manifest_sha256"]
    ):
        raise ValueError("导入后的清单身份改变")
    manifest = wc.load_manifest(receipt["manifest_path"])
    if manifest["sha256"] != receipt["manifest_sha256"]:
        raise ValueError("清单内部摘要与收据不符")
    expected = {}
    episodes = {(ep["h5_file"], ep["raw_ep_idx"]): ep for ep in manifest["episodes"]}
    for item in wc.list_segments(manifest):
        for m in range(item["num_grid"]):
            start = item["seg_start"] + 16 * m
            real = min(33, item["seg_len"] - 16 * m)
            expected[item["key"], m] = {
                "h5_file": item["h5_file"],
                "raw_ep_idx": item["raw_ep_idx"],
                "segment_kind": item["segment"],
                "start_global_frame": start,
                "real_frames": real,
                "pad_frames": 33 - real,
                "pad_source_frame": start + real - 1 if real < 33 else None,
                "frame_ids": list(range(start, start + real)) + [start + real - 1] * (33 - real),
            }
    if set(expected) != set(mapping):
        raise ValueError("收据窗口全集与目标清单不同")
    for identity, fields in expected.items():
        if any(mapping[identity].get(k) != v for k, v in fields.items()):
            raise ValueError("收据窗口帧号与目标清单重新推导的结果不同")
    raw_sources = {r["name"]: r for r in receipt["source_h5_files"]}
    pinned_inputs = read_json(receipt["input_manifest_path"])
    pinned_sources = {
        name: {"name": name, "sha256": record["sha256"], "bytes": record["size"]}
        for name, record in pinned_inputs["files"].items()
    }
    if raw_sources != pinned_sources or pinned_inputs["count"] != len(pinned_sources):
        raise ValueError("收据源 H5 集合与固定 input manifest 不同")
    for record in raw_sources.values():
        check_file(safe_path(raw_dir, record["name"]), record)
    latents = receipt["latents"]
    root = Path(latents["root"])
    if sha_file(root / "metadata.json") != latents["metadata_sha256"]:
        raise ValueError("导入总 metadata 被篡改")
    files = {r["path"]: r for r in latents["files"]}
    checked, seen, tasks = set(), set(), Counter()
    counts = Counter()
    for sample in samples["windows"]:
        identity = (sample["key"], sample["source_grid"])
        if identity in seen or identity[1] % 2:
            raise ValueError("G0 重复窗口或非 grid32 窗口")
        seen.add(identity)
        row = mapping[identity]
        for key in ("frame_ids", "input_frames_sha256", "real_frames", "pad_frames", "pad_source_frame"):
            if sample[key] != row[key]:
                raise ValueError(f"G0 帧合同不符: {identity} {key}")
        if sample["latent_block_sha256"] != row["latent_sha256"]:
            raise ValueError("G0 latent 摘要不符")
        if sample["segment_kind"] != row["segment_kind"] or sample["episode"] != row["raw_ep_idx"]:
            raise ValueError("G0 episode 或段身份不符")
        route = "copy" if row["route"] == "copy" else row["segment_kind"] + "-fill"
        source = {
            "kind": "raw_h5",
            "path": str(safe_path(raw_dir, row["h5_file"]).resolve()),
            "sha256": raw_sources[row["h5_file"]]["sha256"],
            "episode": row["raw_ep_idx"],
        }
        if sample.get("route") != row["route"] or sample.get("cache_route") != route:
            raise ValueError("G0 路线不符")
        task = wc.task_of_h5(row["h5_file"])
        if sample["task"] != task:
            raise ValueError("G0 任务身份不符")
        tasks[task] += 1
        frames = read_window(raw_dir, row)
        if (
            frames.shape != (33, 256, 256, 3)
            or frames.dtype != np.uint8
            or wc.sha256_bytes(frames) != row["input_frames_sha256"]
        ):
            raise ValueError("原 H5 窗口 RGB、帧序或补帧摘要不符")
        if row["segment_kind"] == "demo":
            if sample.get("sources") != [source] or sample.get("source_frame_indices") != [
                {"source": 0, "index": f} for f in row["frame_ids"]
            ]:
                raise ValueError("G0 demo 原 H5 来源或帧序不符")
        else:
            ep = episodes[row["h5_file"], row["raw_ep_idx"]]
            if (
                sample.get("raw_h5") != source
                or len(sample.get("sources", [])) != 33
                or sample.get("source_frame_indices") != [{"source": i, "index": 0} for i in range(33)]
            ):
                raise ValueError("G0 exec 原 H5 来源或帧序不符")
            for i, record in enumerate(sample["sources"]):
                raw = Path(record["path"]).read_bytes()
                if record.get("kind") != "exec_pkl" or wc.sha256_bytes(raw) != record.get("sha256"):
                    raise ValueError("G0 执行 PKL 来源或 SHA 不符")
                data = pickle.loads(raw)
                for name, value in (
                    ("epis_idx", ep["global_episode_idx"]),
                    ("step_idx", row["frame_ids"][i]),
                    ("exec_start_idx", ep["exec_start_idx"]),
                ):
                    array = data.get(name)
                    if (
                        not isinstance(array, np.ndarray)
                        or array.shape != (1,)
                        or array.dtype != np.int32
                        or array.item() != value
                    ):
                        raise ValueError("G0 执行 PKL episode 或帧索引不符")
                flag, pixels = data.get("is_demo"), data.get("image")
                if (
                    not isinstance(flag, np.ndarray)
                    or flag.shape != (1,)
                    or flag.dtype != np.bool_
                    or flag.item()
                    or not isinstance(pixels, np.ndarray)
                    or pixels.shape != (256, 256, 3)
                    or pixels.dtype != np.uint8
                    or not np.array_equal(pixels, frames[i])
                ):
                    raise ValueError("G0 执行 PKL 像素与原 H5 不一致")
        path = safe_path(root, row["output_file"])
        if Path(sample["latent_path"]).resolve() != path.resolve():
            raise ValueError("G0 latent 路径不符")
        if path not in checked:
            check_file(path, files[row["output_file"]])
            checked.add(path)
        with path.open("rb") as stream:
            stream.seek(row["output_offset"])
            blob = stream.read(wc.CHUNK_BYTES)
        if wc.sha256_bytes(blob) != row["latent_sha256"] or not np.isfinite(np.frombuffer(blob, dtype="<f4")).all():
            raise ValueError("最终 latent 块 SHA 或 finite 不符")
        counts["windows"] += 1
        counts["demo_tail"] += row["segment_kind"] == "demo" and row["pad_frames"] > 0
        counts["copy" if row["route"] == "copy" else row["segment_kind"] + "_fill"] += 1
    if (
        counts["windows"] != 64
        or len(tasks) != 16
        or set(tasks.values()) != {4}
        or counts["demo_tail"] < 16
        or any(counts[k] < 1 for k in ("copy", "demo_fill", "exec_fill"))
    ):
        raise ValueError("G2 必须覆盖 64 窗、16 任务及三种输入来源")
    return {
        "schema": "stage2-validation-gate-v1",
        "gate": "G2",
        "status": "PASS",
        "kind": KIND,
        "samples_sha256": sha_file(samples_path),
        "cache_import_receipt_sha256": receipt_sha,
        "manifest_sha256": receipt["manifest_sha256"],
        "latent_metadata_sha256": latents["metadata_sha256"],
        "vae": receipt["vae"],
        "preprocess": receipt["preprocess"],
        "numeric_modes": receipt["numeric_modes"],
        "output_numerics": {"status": "NOT_REQUIRED", "reason": "user_decision_same_input"},
        "metrics": {**counts, "sha_mismatches": 0, "frame_mismatches": 0, "shape_mismatches": 0, "nonfinite": 0},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "plan": ("manifest", "input-manifest", "published-index", "out"),
        "fetch": ("plan", "revision", "out"),
        "import": ("plan", "cache-root", "raw-dir", "out", "receipt", "gpus"),
        "check": ("samples", "receipt", "raw-dir", "out"),
    }
    for command, arguments in commands.items():
        child = sub.add_parser(command)
        for argument in arguments:
            child.add_argument("--" + argument, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = build_plan(args.manifest, args.input_manifest, args.published_index)
        write_json(args.out, result)
    elif args.command == "fetch":
        result = fetch_plan(load_plan(args.plan), args.revision, args.out)
    elif args.command == "import":
        result = import_plan(args.plan, args.cache_root, args.raw_dir, args.out, args.receipt, args.gpus)
    else:
        result = check_import(args.samples, args.receipt, args.raw_dir)
        if Path(args.out).exists():
            raise FileExistsError("G2 已存在，拒绝覆盖")
        write_json(args.out, result)
    counts = result.get("counts", result.get("metrics", {}))
    if args.command == "plan":
        print(f"WAN_CACHE_AUDIT=PASS candidates={counts['windows']} route={KIND}")
    elif args.command == "import":
        print(
            f"WAN_CACHE_IMPORT=PASS copied={counts['copy']} filled={counts['fill']} "
            f"total={counts['windows']} input_mismatches=0 nonfinite=0"
        )
        print(f"WAN_INPUT_COVERAGE=PASS windows={counts['windows']} missing=0 duplicate=0 mismatches=0")
    elif args.command == "check":
        print(f"G2_INPUT_IDENTITY=PASS windows={counts['windows']} mismatches=0")
        print("G2_NUMERICS=NOT_REQUIRED reason=user_decision_same_input")
    else:
        print("WAN_CACHE_FETCH=PASS " + " ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, pickle.UnpicklingError) as exc:
        print(f"WAN_CACHE=FAIL error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
