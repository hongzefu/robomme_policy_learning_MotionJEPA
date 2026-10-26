#!/usr/bin/env python3
"""固定 G0 原帧清单：只读缓存和本地原图，逐窗确认补帧后的输入字节身份。"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import random
import sys
import tempfile

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.manifest import load_manifest

CHUNK_BYTES = 589824
GRID_STRIDE = 16
WINDOW_FRAMES = 33
TASKS = 16
PER_TASK = 4
MIN_DEMO_TAIL = 16


def canonical_json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def require_sha256(value, context):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{context} 缺少合法 SHA256")


def require_file(path):
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"缺少本地输入文件: {path}；不自动下载或重建")
    return path


def read_json(path):
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def select_windows(manifest, latents, *, tasks=TASKS, per_task=PER_TASK, min_demo_tail=MIN_DEMO_TAIL):
    """按排序后的候选和 seed=0 抽样，优先轮转覆盖不同任务的 demo 补帧尾窗。"""
    grouped = {}
    total = read_json(latents / "metadata.json")
    expected = {"schema": 3, "stage": "wan", "grid_stride": GRID_STRIDE,
                "window_frames": WINDOW_FRAMES, "frame_size": 256,
                "demo_min_real_frames": 17, "exec_min_real_frames": 33,
                "demo_tail_pad": "repeat_last", "grid_origin": "segment_start",
                "window_direction": "forward", "truncation_policy": "none",
                "manifest_sha256": manifest["sha256"]}
    if any(total.get(k) != v for k, v in expected.items()):
        raise ValueError("历史 latent 总 metadata 与清单或窗口合同不符")
    episode_ids, segment_keys = set(), set()
    for ep in manifest["episodes"]:
        for field in ("global_episode_idx", "raw_ep_idx", "num_timesteps", "exec_start_idx", "exec_sample_offset"):
            if type(ep.get(field)) is not int or ep[field] < 0:
                raise ValueError(f"episode 清单的 {field} 必须为非负整数")
        if ep["global_episode_idx"] in episode_ids:
            raise ValueError("episode 清单存在重复 global_episode_idx")
        episode_ids.add(ep["global_episode_idx"])
        task = ms.task_of_h5(ep["h5_file"])
        grouped.setdefault(task, [])
        for segment, length in ms.segment_lengths(ep["num_timesteps"], ep["exec_start_idx"]).items():
            minimum = 17 if segment == "demo" else 33
            # 固定历史格距，不能随 stage2 存储布局的格距变化。
            count = len(range(0, max(0, length - minimum + 1), GRID_STRIDE))
            key = ms.segment_key(ep["h5_file"], ep["raw_ep_idx"], segment)
            if key in segment_keys:
                raise ValueError(f"episode 清单存在重复段 {key}")
            segment_keys.add(key)
            if not count:
                continue
            lm = read_json(latents / f"{key}.metadata.json")
            bound = total["segments"].get(key)
            start_global = 0 if segment == "demo" else ep["exec_start_idx"]
            segment_expected = {"schema": 3, "segment": key, "segment_kind": segment,
                                "g": ep["global_episode_idx"], "h5_file": ep["h5_file"],
                                "raw_ep_idx": ep["raw_ep_idx"], "seg_len": length,
                                "seg_start_global": start_global, "grid_stride": GRID_STRIDE,
                                "window_frames": WINDOW_FRAMES, "min_real": minimum,
                                "num_grid": count, "num_chunks": max(0, length - minimum + 1),
                                "latent_shape": [9, 16, 32, 32], "latent_dtype": "float32",
                                "chunk_bytes": CHUNK_BYTES, "bytes": count * CHUNK_BYTES}
            bound_fields = ("sha256", "bytes", "rows", "num_grid", "seg_len", "seg_start_global",
                            "segment_kind", "g", "min_real")
            if (not isinstance(bound, dict) or any(lm.get(k) != v for k, v in segment_expected.items())
                    or len(lm.get("rows", [])) != count
                    or any(k not in bound or lm.get(k) != bound[k] for k in bound_fields)):
                raise ValueError(f"{key} 历史 metadata 与清单或固定总表不符")
            require_sha256(lm.get("sha256"), f"{key} sha256")
            for old_grid in range(0, count, 2):
                row = lm["rows"][old_grid]
                start = start_global + old_grid * GRID_STRIDE
                real = min(33, length - old_grid * GRID_STRIDE)
                pad = 33 - real
                want = {"m": old_grid, "seg_offset": old_grid * GRID_STRIDE, "start_global_frame": start,
                        "real_frames": real, "pad_frames": pad,
                        "pad_source_frame": start + real - 1 if pad else None,
                        "input_shape": [33, 256, 256, 3], "input_dtype": "uint8"}
                if any(row.get(k) != v for k, v in want.items()):
                    raise ValueError(f"{key} 格 {old_grid} 的原始帧合同不符")
                require_sha256(row.get("input_frames_sha256"), f"{key} input_frames_sha256")
                grouped[task].append((ep, segment, key, old_grid, row))
    if len(grouped) != tasks or any(len(v) < per_task for v in grouped.values()):
        raise ValueError("G0 任务数或每任务候选数不足")
    rng = random.Random(0)
    tails, selected = {}, {task: [] for task in sorted(grouped)}
    for task in selected:
        grouped[task].sort(key=lambda x: (x[2], x[3]))
        rng.shuffle(grouped[task])
        tails[task] = [x for x in grouped[task] if x[1] == "demo" and x[4]["pad_frames"] > 0]
    needed = min_demo_tail
    while needed:
        progressed = False
        for task in selected:
            if tails[task] and len(selected[task]) < per_task - 1 and needed:
                selected[task].append(tails[task].pop())
                needed -= 1
                progressed = True
        if not progressed:
            raise ValueError(f"G0 demo 补帧尾窗不足，尚缺 {needed}")
    for task, chosen in selected.items():
        # 每任务保留真实执行窗，端到端验证同时覆盖两个来源。
        execution = next((x for x in grouped[task] if x[1] == "exec"), None)
        if execution is None:
            raise ValueError(f"G0 {task} 缺执行窗")
        chosen.append(execution)
        chosen.extend([x for x in grouped[task] if x not in chosen][:per_task - len(chosen)])
    return [x for chosen in selected.values() for x in chosen]


def materialize_window(item, *, source, demo_root, scope, run_manifest, file_sha_cache):
    """返回完整像素和显式源索引；demo 输出编号与原来源 episode 分开核对。"""
    ep, segment, key, old_grid, row = item
    start, real, pad = row["start_global_frame"], row["real_frames"], row["pad_frames"]
    frame_ids = list(range(start, start + real)) + [row["pad_source_frame"]] * pad
    sources, indices, frames = [], [], []

    def digest(path):
        path = require_file(path).resolve()
        if path not in file_sha_cache:
            file_sha_cache[path] = ms.sha256_file(path)
        return file_sha_cache[path]

    if segment == "demo":
        path = demo_root / key.removesuffix("_demo") / "video_demo.h5"
        complete = path.with_suffix(".complete.json")
        record = read_json(complete)
        origin = scope.get(key)
        if origin is None:
            raise ValueError(f"control/scope.json 缺输出段 {key}")
        identity = record.get("identity", {})
        fields = ("difficulty", "episode_group", "h5_sha256", "hi", "lo", "member", "output_episode_index", "seed", "suffix")
        if (any(k not in identity or k not in origin or identity[k] != origin[k] for k in fields)
                or identity.get("output_episode_index") != ep["raw_ep_idx"]
                or identity.get("lo") != 0 or identity.get("hi") != ep["exec_start_idx"]
                or origin.get("name") != key or identity.get("suffix") != "demo"
                or origin.get("task") != ms.task_of_h5(ep["h5_file"])):
            raise ValueError(f"{key} 完成记录、输出编号与 control 映射不同源")
        require_sha256(identity["h5_sha256"], f"{key} 原来源 H5 sha256")
        sha = digest(path)
        relative = str(path.resolve().relative_to(demo_root.parent.resolve()))
        if sha != record.get("sha256") or sha != run_manifest["raw_sha256"].get(relative):
            raise ValueError(f"{key} demo H5 文件摘要与完成记录或 control 不符")
        sources.append({"kind": "demo_h5", "path": str(path.resolve()), "sha256": sha,
                        "complete_path": str(complete.resolve()), "complete_sha256": digest(complete),
                        "identity": identity})
        with h5py.File(path, "r") as handle:
            if "frames" not in handle:
                raise ValueError(f"{path} 缺少 frames 数据集")
            data = handle["frames"]
            if data.shape != (ep["exec_start_idx"], 256, 256, 3) or data.dtype != np.uint8:
                raise ValueError(f"{key} demo frames 形制不符")
            for frame in frame_ids:
                frames.append(data[frame])
                indices.append({"source": 0, "index": frame})
    else:
        for frame in frame_ids:
            sample = ep["exec_sample_offset"] + frame - ep["exec_start_idx"]
            path = source / "data" / f"{sample}.pkl"
            raw = require_file(path).read_bytes()
            try:
                data = pickle.loads(raw)
            except (pickle.UnpicklingError, EOFError, ValueError) as exc:
                raise ValueError(f"{path} 执行样本无法读取: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{path} 执行样本必须是字典")
            for field, expected in (("epis_idx", ep["global_episode_idx"]), ("step_idx", frame),
                                    ("exec_start_idx", ep["exec_start_idx"])):
                value = data.get(field)
                if (not isinstance(value, np.ndarray) or value.shape != (1,) or value.dtype != np.int32
                        or value.item() != expected):
                    raise ValueError(f"{path} 的 {field} 与清单或 (1,) int32 合同不符")
            is_demo = data.get("is_demo")
            if (not isinstance(is_demo, np.ndarray) or is_demo.shape != (1,)
                    or is_demo.dtype != np.bool_ or is_demo.item()):
                raise ValueError(f"{path} 执行窗却标记为 demo")
            image = data.get("image")
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (256, 256, 3):
                raise ValueError(f"{path} image 形制不符")
            indices.append({"source": len(sources), "index": 0})
            sources.append({"kind": "exec_pkl", "path": str(path.resolve()), "sha256": ms.sha256_bytes(raw)})
            frames.append(image)
    pixels = np.ascontiguousarray(np.stack(frames))
    if ms.sha256_bytes(pixels.tobytes()) != row["input_frames_sha256"]:
        raise ValueError(f"{key} 格 {old_grid} 原帧 SHA 与缓存不一致")
    return pixels, {"key": key, "task": ms.task_of_h5(ep["h5_file"]), "episode": ep["raw_ep_idx"],
                    "segment_kind": segment, "source_grid": old_grid, "frame_ids": frame_ids,
                    **{k: row[k] for k in ("real_frames", "pad_frames", "pad_source_frame", "input_frames_sha256")},
                    "sources": sources, "source_frame_indices": indices}


def build_inputs(manifest_path, latents, source, demo_root):
    latents, source, demo_root = (pathlib.Path(p).resolve() for p in (latents, source, demo_root))
    manifest = load_manifest(require_file(manifest_path))
    selected = select_windows(manifest, latents)
    control = demo_root.parent / "control"
    scope_path = control / "scope.json"
    scope_record = read_json(scope_path)
    if scope_record.get("status") != "PASS":
        raise ValueError(f"{scope_path} 不是通过的来源映射")
    scope_list = scope_record["entries"]
    scope = {x["name"]: x for x in scope_list}
    if len(scope) != len(scope_list):
        raise ValueError("control 段身份重复")
    run_path = control / "run_manifest.json"
    run = read_json(run_path)
    if not scope_record.get("profile") or scope_record["profile"] != run.get("profile"):
        raise ValueError("control scope 与 run_manifest 的 profile 不一致")
    windows, cache = [], {}
    for item in selected:
        _, row = materialize_window(item, source=source, demo_root=demo_root, scope=scope,
                                    run_manifest=run, file_sha_cache=cache)
        path = require_file(latents / f"{row['key']}.bin")
        if path not in cache:
            lm = read_json(latents / f"{row['key']}.metadata.json")
            if path.stat().st_size != lm["bytes"] or ms.sha256_file(path) != lm["sha256"]:
                raise ValueError(f"{path} latent 文件字节数或 SHA 与历史 metadata 不一致")
            cache[path] = lm["sha256"]
        with path.open("rb") as stream:
            stream.seek(row["source_grid"] * CHUNK_BYTES)
            block = stream.read(CHUNK_BYTES)
        if len(block) != CHUNK_BYTES:
            raise ValueError(f"{path} latent 块不足")
        row.update(latent_path=str(path.resolve()), latent_block_sha256=ms.sha256_bytes(block))
        windows.append(row)
    counts = {"windows": len(windows), "tasks": len({w["task"] for w in windows}),
              "demo_tail": sum(w["segment_kind"] == "demo" and w["pad_frames"] > 0 for w in windows),
              "sha_mismatches": 0}
    if (counts["windows"] != TASKS * PER_TASK or counts["tasks"] != TASKS
            or counts["demo_tail"] < MIN_DEMO_TAIL
            or len({(w["key"], w["source_grid"]) for w in windows}) != TASKS * PER_TASK
            or any(sum(w["task"] == task for w in windows) != PER_TASK for task in {w["task"] for w in windows})):
        raise ValueError("G0 抽样结果未满足固定的 64 窗、16 任务和 demo 尾窗合同")
    return {"schema": "stage2-validation-inputs-v1", "gate": "G0", "status": "PASS", "seed": 0,
            "manifest_sha256": manifest["sha256"], "latent_metadata_sha256": ms.sha256_file(latents / "metadata.json"),
            "control": {"scope": {"path": str(scope_path), "sha256": ms.sha256_file(scope_path)},
                        "run_manifest": {"path": str(run_path), "sha256": ms.sha256_file(run_path)}},
            "windows": windows, "counts": counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "latents", "source", "out"):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    parser.add_argument("--demo-root", type=pathlib.Path,
                        default=pathlib.Path("/scratch/hongze/dataset-local-a100-16task-mixed8-200/data-raw"))
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        raise FileExistsError(f"G0 清单已经存在，拒绝覆盖: {args.out}")
    result = build_inputs(args.manifest, args.latents.resolve(), args.source.resolve(), args.demo_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.out.parent, prefix=f".{args.out.name}.", delete=False) as stream:
            tmp = pathlib.Path(stream.name)
            stream.write(canonical_json_bytes(result) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        # 同目录硬链接原子发布，目标若被并发创建则失败，绝不覆盖。
        os.link(tmp, args.out)
    finally:
        if tmp is not None:
            tmp.unlink()
    print(f"RAW_WINDOWS_READY=PASS windows=64 tasks=16 demo_tail={result['counts']['demo_tail']} sha_mismatches=0")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, pickle.UnpicklingError) as exc:
        print(f"RAW_WINDOWS_READY=FAIL error={exc}", file=sys.stderr)
        sys.exit(1)
