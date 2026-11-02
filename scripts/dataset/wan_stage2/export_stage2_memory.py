#!/usr/bin/env python3
"""只读恢复的 latent，按偶数格导出阶段二运动表，并核验所有 IO 身份。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import tempfile

import numpy as np
import torch
import yaml

from motion_jepa.export.stage2_memory import (
    IDENTITY_FIELDS, canonical_json, check_dropout, encode_windows, file_sha256,
    load_ema_encoder,
)


SCHEMA = "stage2-memory-export-v1"
LAYOUT = "motion-stage2-1x1024-grid32-demopad17-v1"
LATENT_SHAPE = (9, 16, 32, 32)
LATENT_BYTES = 589824
ROW_BYTES = 4096


def read_json(path):
    """拒绝重复键与非标准数字，避免两种解析器看到不同身份。"""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON 重复键：{key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"JSON 非有限数字：{value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def require_equal(actual, expected, label):
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(f"{label} 不匹配")


def valid_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("SHA256 必须为 64 位小写十六进制")
    return value


def integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须为非负整数")
    return value


def parse_expected_counts(value):
    """命令行显式指定全表段数、grid32 行数，不预设某个数据集规模。"""
    if not re.fullmatch(r"[0-9]+,[0-9]+", value):
        raise argparse.ArgumentTypeError("expected-counts 必须为 <正整数段数>,<正整数行数>")
    counts = tuple(int(part) for part in value.split(","))
    if min(counts) < 1:
        raise argparse.ArgumentTypeError("expected-counts 的段数与行数必须均为正整数")
    return counts


def verify_input_lineage(checkpoint_path, input_manifest_path, *, lineage_build_spec=None):
    """比较训练源与 VLA 输入清单的完整 H5 集合、字节数和 SHA256。"""
    try:
        config = yaml.safe_load((checkpoint_path.parent / "config.yaml").read_text())
        data_root = config["data"]["data_root"]
        if not isinstance(data_root, str) or not data_root:
            raise ValueError("config.yaml 缺少有效 data.data_root")
        build_spec = read_json(Path(lineage_build_spec) if lineage_build_spec is not None
                               else Path(data_root) / "control" / "build_spec.json")
        source_files = build_spec["identity"]["source_pin"]["files"]
        input_files = read_json(input_manifest_path)["files"]
        if not isinstance(source_files, list) or not source_files:
            raise ValueError("训练 source_pin.files 必须为非空列表")
        if not isinstance(input_files, dict) or not input_files:
            raise ValueError("input-manifest.files 必须为非空文件名映射")
        source, inputs = {}, {}
        for entry in source_files:
            path = entry["path"]
            if not isinstance(path, str) or not path:
                raise ValueError("训练 source_pin.files.path 无效")
            name = Path(path).name
            if not re.fullmatch(r"record_dataset_[A-Za-z0-9]+\.h5", name):
                raise ValueError(f"训练来源 H5 文件名无效：{name}")
            if name in source:
                raise ValueError(f"训练来源 H5 文件名重复：{name}")
            source[name] = {"size": integer(entry["bytes"], f"{name}.bytes"),
                            "sha256": valid_sha(entry["sha256"])}
        for name, entry in input_files.items():
            if not re.fullmatch(r"record_dataset_[A-Za-z0-9]+\.h5", name):
                raise ValueError(f"input-manifest H5 文件名无效：{name}")
            inputs[name] = {"size": integer(entry["size"], f"{name}.size"),
                            "sha256": valid_sha(entry["sha256"])}
        mismatches = sorted(name for name in source.keys() | inputs.keys()
                            if source.get(name) != inputs.get(name))
        if mismatches:
            raise ValueError(f"训练源与 input-manifest 不匹配；mismatches={len(mismatches)} "
                             f"files={mismatches}")
    except Exception as error:
        print(f"MJ_CLI_LINEAGE=FAIL error={type(error).__name__} detail={error}", flush=True)
        raise
    print(f"MJ_CLI_LINEAGE=PASS files={len(source)} mismatches=0", flush=True)


def manifest_segments(manifest):
    """独立从 episode 清单推导非零段，demo 最少 17 帧，exec 最少 33 帧。"""
    segments, seen = {}, set()
    for episode in manifest["episodes"]:
        g = integer(episode["global_episode_idx"], "global_episode_idx")
        if g in seen:
            raise ValueError("manifest 包含重复 episode 编号")
        seen.add(g)
        h5_name = episode["h5_file"]
        match = re.fullmatch(r"record_dataset_([A-Za-z0-9]+)\.h5", h5_name)
        if match is None:
            raise ValueError(f"manifest 的 h5_file 无效：{h5_name}")
        raw = integer(episode["raw_ep_idx"], "raw_ep_idx")
        total = integer(episode["num_timesteps"], "num_timesteps")
        split = integer(episode["exec_start_idx"], "exec_start_idx")
        if split > total:
            raise ValueError("exec_start_idx 超出 episode 长度")
        for kind, start, length, minimum in (("demo", 0, split, 17),
                                              ("exec", split, total - split, 33)):
            count = max(0, (length - minimum) // 16 + 1)
            if not count:
                continue
            key = f"{match[1]}_ep{raw}_{kind}"
            if key in segments:
                raise ValueError(f"manifest 段 key 重复：{key}")
            segments[key] = {"segment": key, "g": g, "h5_file": h5_name,
                             "raw_ep_idx": raw, "segment_kind": kind,
                             "seg_start_global": start, "seg_len": length,
                             "num_grid": count, "min_real": minimum}
    if not segments:
        raise ValueError("manifest 没有任何非零段")
    return dict(sorted(segments.items()))


def verify_sidecar(path, digest):
    sidecar = Path(str(path) + ".sha256")
    fields = sidecar.read_text().strip().split()
    if len(fields) not in (1, 2) or fields[0] != digest:
        raise ValueError(f"SHA sidecar 与实际文件不同：{sidecar}")
    if len(fields) == 2 and fields[1] not in (path.name, "*" + path.name):
        raise ValueError(f"SHA sidecar 文件名不同：{sidecar}")


def source_segment(root, key, expected):
    """完整校验源段索引、整文件 SHA 及偶数块 SHA，绝不改写历史元数据。"""
    path, meta_path = root / f"{key}.bin", root / f"{key}.metadata.json"
    meta = read_json(meta_path)
    for field, wanted in expected.items():
        require_equal(meta[field], wanted, f"{key}.{field}")
    contract = {"schema": 3, "grid_stride": 16, "window_frames": 33,
                "grid_origin": "segment_start", "window_direction": "forward",
                "truncation_policy": "none", "frame_size": 256,
                "latent_shape": list(LATENT_SHAPE), "latent_dtype": "float32",
                "chunk_bytes": LATENT_BYTES,
                "bytes": expected["num_grid"] * LATENT_BYTES,
                "num_chunks": max(0, expected["seg_len"] - expected["min_real"] + 1)}
    for field, wanted in contract.items():
        require_equal(meta[field], wanted, f"{key}.{field}")
    if path.stat().st_size != contract["bytes"] or len(meta["rows"]) != expected["num_grid"]:
        raise ValueError(f"源段字节数或行数错误：{key}")
    for index, row in enumerate(meta["rows"]):
        real = min(33, expected["seg_len"] - index * 16)
        fields = {"m": index, "seg_offset": index * 16,
                  "start_global_frame": expected["seg_start_global"] + index * 16,
                  "input_shape": [33, 256, 256, 3], "input_dtype": "uint8",
                  "real_frames": real, "pad_frames": 33 - real,
                  "pad_source_frame": expected["seg_start_global"] + expected["seg_len"] - 1
                  if real < 33 else None}
        for field, wanted in fields.items():
            require_equal(row[field], wanted, f"{key}.rows[{index}].{field}")
        valid_sha(row["input_frames_sha256"])
    digest, blocks = hashlib.sha256(), []
    with path.open("rb") as stream:
        for index in range(expected["num_grid"]):
            block = stream.read(LATENT_BYTES)
            if len(block) != LATENT_BYTES:
                raise ValueError(f"源段读取不完整：{key}")
            digest.update(block)
            if index % 2 == 0:
                blocks.append(hashlib.sha256(block).hexdigest())
        if stream.read(1):
            raise ValueError(f"源段含多余尾字节：{key}")
    require_equal(digest.hexdigest(), valid_sha(meta["sha256"]), f"{key} 源 SHA256")
    verify_sidecar(path, digest.hexdigest())
    return {"path": path, "metadata": meta, "source_grid": list(range(0, expected["num_grid"], 2)),
            "input_latent_sha256": digest.hexdigest(), "input_block_sha256": blocks,
            "input_metadata_sha256": file_sha256(meta_path)}


def read_window(source, index, device):
    with source["path"].open("rb") as stream:
        stream.seek(source["source_grid"][index] * LATENT_BYTES)
        block = stream.read(LATENT_BYTES)
    if len(block) != LATENT_BYTES or hashlib.sha256(block).hexdigest() != source["input_block_sha256"][index]:
        raise ValueError("读取时 latent 块与已校验身份不同")
    array = np.frombuffer(block, dtype="<f4").reshape(1, *LATENT_SHAPE).copy()
    if not np.isfinite(array).all():
        raise ValueError("latent 块含非有限值")
    return torch.from_numpy(array).to(device)


def atomic_new(path, content):
    """先完整写临时文件再独占发布，不覆盖已存在的用户产物。"""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def output_metadata(source, info, manifest_sha, worker):
    old = source["metadata"]
    grid = source["source_grid"]
    return {"schema": SCHEMA, "layout": LAYOUT,
            **{name: old[name] for name in ("segment", "g", "h5_file", "raw_ep_idx",
                                            "segment_kind", "seg_start_global", "seg_len")},
            "grid_stride": 32, "window_frames": 33, "row_shape": [1, 1024],
            "dtype": "float32", "token_dim": 1024, "token_bytes": ROW_BYTES,
            "num_grid": len(grid), "bytes": len(grid) * ROW_BYTES,
            "source_grid": grid, "rows": [deepcopy(old["rows"][i]) for i in grid],
            **{name: source[name] for name in ("input_latent_sha256", "input_block_sha256",
                                               "input_metadata_sha256")},
            **{name: deepcopy(info[name]) for name in IDENTITY_FIELDS},
            "manifest_sha256": manifest_sha, "worker": worker}


def verify_output(out, key, expected):
    """恢复只接受三件套齐全且内容、来源及环境全部匹配的段。"""
    path = out / f"{key}.f32.bin"
    meta = read_json(out / f"{key}.metadata.json")
    for field, wanted in expected.items():
        if field != "worker":
            require_equal(meta[field], wanted, f"{key} 输出 {field}")
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"输出段字节数错误：{key}")
    digest = file_sha256(path)
    require_equal(meta["sha256"], digest, f"{key} 输出 SHA256")
    verify_sidecar(path, digest)
    values = np.fromfile(path, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError(f"输出含非有限值：{key}")
    return meta


def export(args):
    """参数已定时执行一次导出；全量与冒烟的根记录明确分开。"""
    if args.grid_subsample != 2 or args.batch != 1:
        raise ValueError("阶段二导出固定 grid-subsample=2、batch=1")
    if args.max_segments is not None and args.max_segments < 1:
        raise ValueError("max-segments 必须为正整数")
    if args.check_dropout not in (0, 64):
        raise ValueError("正式 dropout 诊断固定为 64 窗")
    scope = "full" if args.max_segments is None else "smoke"
    expected_counts = getattr(args, "expected_counts", None)
    if scope == "full" and (not isinstance(expected_counts, (tuple, list))
                            or len(expected_counts) != 2
                            or any(type(value) is not int or value < 1 for value in expected_counts)):
        raise ValueError("全表导出必须提供 --expected-counts <正整数段数>,<正整数行数>")
    if not getattr(args, "input_manifest", None):
        raise ValueError("导出必须提供 --input-manifest")
    valid_sha(args.expected_sha256)
    checkpoint_path = Path(args.ckpt).resolve(strict=True)
    input_manifest_path = Path(args.input_manifest).resolve(strict=True)
    verify_input_lineage(checkpoint_path, input_manifest_path,
                         lineage_build_spec=getattr(args, "lineage_build_spec", None))
    latent_root = Path(args.latents).resolve(strict=True)
    manifest_path = Path(args.manifest).resolve(strict=True)
    out = Path(args.out).absolute()
    if any(parent.is_symlink() for parent in (out, *out.parents)):
        raise ValueError("输出路径不能通过符号链接写入")
    out = out.resolve()
    for protected in (latent_root, checkpoint_path.parent, manifest_path.parent,
                      input_manifest_path.parent):
        if out == protected or protected in out.parents:
            raise ValueError("输出不能位于输入或源 checkpoint 目录内")
    manifest_sha = file_sha256(manifest_path)
    segments = manifest_segments(read_json(manifest_path))
    expected_keys = set(segments)
    for suffix in (".bin", ".metadata.json", ".bin.sha256"):
        actual = {path.name.removesuffix(suffix) for path in latent_root.glob(f"*{suffix}")}
        if actual != expected_keys:
            raise ValueError(f"latent {suffix} 段集合与 manifest 不符；"
                             f"缺少={sorted(expected_keys - actual)[:5]} 多余={sorted(actual - expected_keys)[:5]}")
    keys = list(segments)
    if scope == "full":
        counts = (len(keys), sum((entry["num_grid"] + 1) // 2 for entry in segments.values()))
        require_equal(counts, expected_counts, "全表段数与行数（expected-counts）")
    else:
        keys = keys[:args.max_segments]
    if out.exists() and (not args.resume or not out.is_dir()):
        raise FileExistsError("输出已存在；仅显式 --resume 可校验已有产物")
    allowed = {f"{key}{suffix}" for key in keys
               for suffix in (".f32.bin", ".f32.bin.sha256", ".metadata.json")}
    allowed.add("export_record.json")
    if out.exists():
        for path in out.iterdir():
            if path.name not in allowed or not path.is_file() or path.is_symlink():
                raise ValueError(f"输出目录含未知、部分或符号链接产物：{path}")
    encoder, info = load_ema_encoder(str(checkpoint_path), device=args.device,
                                     expected_sha256=args.expected_sha256)
    worker = {"pid": os.getpid(), "hostname": socket.gethostname(), "device": args.device,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "checkpoint_path": str(checkpoint_path), "latents_path": str(latent_root),
              "output_path": str(out)}
    root_identity = {"schema": SCHEMA, "scope": scope, "segments": len(keys),
                     "rows": sum((segments[key]["num_grid"] + 1) // 2 for key in keys),
                     "segment_keys": keys, "manifest_sha256": manifest_sha,
                     **{name: deepcopy(info[name]) for name in IDENTITY_FIELDS}}
    record_path = out / "export_record.json"
    if record_path.exists():
        previous = read_json(record_path)
        for field, wanted in root_identity.items():
            require_equal(previous[field], wanted, f"根记录 {field}")
        # 已完成记录不得掩盖被删掉的段；不在这种情况下补写。
        if {path.name for path in out.iterdir()} != allowed:
            raise ValueError("已完成根记录引用的段文件不完整")
    dropout_inputs, dropout_result, prepared = [], None, {}
    if args.check_dropout:
        for key in keys:
            source = source_segment(latent_root, key, segments[key])
            prepared[key] = source
            for index in range(len(source["source_grid"])):
                dropout_inputs.append(read_window(source, index, "cpu"))
                if len(dropout_inputs) == args.check_dropout:
                    break
            if len(dropout_inputs) == args.check_dropout:
                break
        if len(dropout_inputs) != args.check_dropout:
            raise ValueError("所选段不足 64 窗，无法完成 dropout 诊断")
        dropout_result = check_dropout(encoder, (window.to(args.device) for window in dropout_inputs))
        print("DROPOUT_OFF=PASS " + " ".join(f"{k}={v}" for k, v in dropout_result.items()), flush=True)
        dropout_inputs.clear()
    out.mkdir(parents=True, exist_ok=args.resume)
    written = []
    for key in keys:
        source = prepared.pop(key) if key in prepared else source_segment(latent_root, key, segments[key])
        expected = output_metadata(source, info, manifest_sha, worker)
        paths = [out / f"{key}{suffix}" for suffix in (".f32.bin", ".f32.bin.sha256", ".metadata.json")]
        exists = [path.exists() for path in paths]
        if any(exists):
            if not args.resume or not all(exists):
                raise ValueError(f"段三件套不完整或未允许恢复：{key}")
            meta = verify_output(out, key, expected)
        else:
            rows = [encode_windows(encoder, read_window(source, i, args.device)).cpu().numpy()
                    for i in range(len(source["source_grid"]))]
            content = np.concatenate(rows).astype("<f4", copy=False).tobytes(order="C")
            if len(content) != expected["bytes"]:
                raise RuntimeError("编码输出字节数错误")
            digest = hashlib.sha256(content).hexdigest()
            meta = {**expected, "sha256": digest}
            atomic_new(paths[0], content)
            atomic_new(paths[1], f"{digest}  {paths[0].name}\n".encode())
            atomic_new(paths[2], canonical_json(meta) + b"\n")
            verify_output(out, key, expected)
        written.append({"key": key, "sha256": meta["sha256"], "num_grid": meta["num_grid"],
                        "metadata_sha256": file_sha256(paths[2])})
        print(f"STAGE2_SEGMENT=PASS key={key} rows={meta['num_grid']}", flush=True)
    record = {**root_identity, "files": written, "worker": worker, "dropout": dropout_result}
    if record_path.exists():
        require_equal(read_json(record_path)["files"], written, "恢复根记录 files")
    else:
        atomic_new(record_path, canonical_json(record) + b"\n")
    if scope == "full":
        print(f"STAGE2_EXPORT=PASS segments={len(keys)} rows={root_identity['rows']}", flush=True)
    else:
        print(f"STAGE2_EXPORT=PASS scope=smoke segments={len(keys)} rows={root_identity['rows']}", flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ckpt", "expected-sha256", "latents", "manifest", "input-manifest", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-counts", type=parse_expected_counts,
                        help="全表必填的 <段数>,<grid32行数>；冒烟不强制核对计数")
    parser.add_argument("--lineage-build-spec",
                        help="显式指定迁移后的训练来源 build_spec.json；默认沿用 config.data.data_root")
    parser.add_argument("--grid-subsample", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--check-dropout", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.max_segments is None and args.expected_counts is None:
        parser.error("全表导出必须提供 --expected-counts <段数>,<grid32行数>")
    try:
        export(args)
    except Exception as error:
        print(f"STAGE2_EXPORT=FAIL error={type(error).__name__} detail={error}", flush=True)
        raise


if __name__ == "__main__":
    main()
