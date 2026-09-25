#!/usr/bin/env python3
"""逐物理 episode 验证 counting source 是公开全集 source 的完整逐位子集。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import struct
import sys

import check_orig80k_sources as sources
import numpy as np

REQUIRED_SAMPLE_KEYS = {
    "image", "wrist_image", "actions", "state", "prompt", "is_demo", "exec_start_idx",
    "step_idx", "epis_idx", "simple_subgoal", "grounded_subgoal", "simple_subgoal_online",
    "grounded_subgoal_online",
}


def _exact_entries(directory: Path, expected: set[str], *, directories: bool = False) -> None:
    """同时拒绝缺项、多项、非普通文件；不把交集误当成完整覆盖。"""
    seen = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name not in expected:
                raise ValueError(f"多余产物: {directory / entry.name}")
            valid = entry.is_dir(follow_symlinks=False) if directories else entry.is_file(follow_symlinks=False)
            if not valid:
                raise ValueError(f"产物类型不符或为链接: {directory / entry.name}")
            seen.add(entry.name)
    if seen != expected:
        raise ValueError(f"缺少产物: {directory / min(expected - seen)}")


def _bytes_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            chunk = a.read(1 << 20)
            if chunk != b.read(1 << 20):
                return False
            if not chunk:
                return True


def feature_difference(left: Path, right: Path) -> str:
    """字节不同时附带定位量；数值接近不会把字节失败变成通过。"""
    if left.suffix != ".npy":
        return "非 NPY 文件字节不同"
    try:
        a, b = (np.load(path, allow_pickle=True).item() for path in (left, right))
        if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(b):
            return "特征键集合不同"
        differences = []
        for key in sorted(a):
            x, y = np.asarray(a[key]), np.asarray(b[key])
            if x.shape != y.shape or x.dtype != y.dtype:
                differences.append(f"{key}:shape/dtype不同")
                continue
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            if not (np.isfinite(xf).all() and np.isfinite(yf).all()):
                differences.append(f"{key}:含非有限值")
            else:
                delta = float(np.max(np.abs(xf - yf))) if x.size else 0.0
                differences.append(f"{key}:max_abs_diff={delta:.17g}")
        return "; ".join(differences)
    except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        return f"无法解码差异详情: {exc}"


def compare_value(left, right, label: str) -> None:
    """递归比较解码后的全部键；数组使用 dtype、shape 与 C 序原始字节。"""
    if type(left) is not type(right):
        raise ValueError(f"{label}: 类型不同 {type(left).__name__}/{type(right).__name__}")
    if isinstance(left, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f"{label}: dtype/shape 不同 {left.dtype}/{left.shape} vs {right.dtype}/{right.shape}")
        if left.dtype.hasobject:
            raise ValueError(f"{label}: 不接受含对象指针、无法定义原始数值字节的数组")
        if left.tobytes(order="C") != right.tobytes(order="C"):
            raise ValueError(f"{label}: 数组原始字节不同")
    elif isinstance(left, np.generic):
        if left.dtype.hasobject or left.dtype != right.dtype or left.tobytes() != right.tobytes():
            raise ValueError(f"{label}: NumPy 标量 dtype/字节不同或含对象")
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise ValueError(f"{label}: 键集合不同")
        for key in left:
            compare_value(left[key], right[key], f"{label}.{key}")
    elif isinstance(left, tuple | list):
        if len(left) != len(right):
            raise ValueError(f"{label}: 序列长度不同")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            compare_value(a, b, f"{label}[{index}]")
    elif isinstance(left, float):
        if struct.pack("!d", left) != struct.pack("!d", right):
            raise ValueError(f"{label}: 浮点标量原始字节不同")
    elif left is None or isinstance(left, str | bytes | bool | int):
        if left != right:
            raise ValueError(f"{label}: 标量或字符串不同")
    else:
        raise ValueError(f"{label}: 不支持的样本字段类型 {type(left).__name__}")


def _sample_identity(sample: dict, episode: dict, step: int, label: str) -> None:
    if not isinstance(sample, dict) or not set(sample) >= REQUIRED_SAMPLE_KEYS:
        raise ValueError(f"{label}: pkl 缺少必要字段或顶层不是字典")
    for key, value in (("epis_idx", episode["global_episode_idx"]), ("step_idx", step),
                       ("exec_start_idx", episode["exec_start_idx"])):
        array = sample[key]
        if not isinstance(array, np.ndarray) or array.shape != (1,) or array.dtype != np.dtype("int32"):
            raise ValueError(f"{label}.{key}: 身份字段必须为 (1,) int32")
        if int(array[0]) != value:
            raise ValueError(f"{label}.{key}: 身份映射错误，实际 {array[0]}，期望 {value}")
    demo = sample["is_demo"]
    if not isinstance(demo, np.ndarray) or demo.shape != (1,) or demo.dtype != np.dtype("bool") or bool(demo[0]):
        raise ValueError(f"{label}.is_demo: 执行样本必须为 (1,) bool 且为 False")


def compare_sample_files(left: Path, right: Path, full_ep: dict, subset_ep: dict, step: int) -> None:
    with left.open("rb") as stream:
        a = pickle.load(stream)
        if stream.read(1):
            raise ValueError(f"pkl 尾部有多余数据: {left}")
    with right.open("rb") as stream:
        b = pickle.load(stream)
        if stream.read(1):
            raise ValueError(f"pkl 尾部有多余数据: {right}")
    identity = f"{full_ep['h5_file']}#{full_ep['raw_ep_idx']}/step={step}"
    _sample_identity(a, full_ep, step, f"{identity}/full")
    _sample_identity(b, subset_ep, step, f"{identity}/subset")
    if set(a) != set(b):
        raise ValueError(f"{identity}: pkl 键集合不同")
    for key in a:
        if key != "epis_idx":
            compare_value(a[key], b[key], f"{identity}.{key}")


def check_subset(full: Path, subset: Path) -> dict:
    full, subset = Path(full), Path(subset)
    full_manifest = sources.read_manifest(full / "meta/episode_manifest.json")
    subset_manifest = sources.read_manifest(subset / "meta/episode_manifest.json")
    totals = sources.compare_episode_manifests(subset_manifest, full_manifest, sources.COUNTING_TASKS)
    for root, manifest in ((full, full_manifest), (subset, subset_manifest)):
        source = root / "source"
        _exact_entries(source / "features", {f"episode_{ep['global_episode_idx']}" for ep in manifest["episodes"]},
                       directories=True)
        _exact_entries(source / "data", {f"{index}.pkl" for index in range(manifest["totals"]["exec_samples"])})
    full_by_identity = {(ep["h5_file"], ep["raw_ep_idx"]): ep for ep in full_manifest["episodes"]}
    samples = features = 0
    for index, subset_ep in enumerate(subset_manifest["episodes"], 1):
        identity = (subset_ep["h5_file"], subset_ep["raw_ep_idx"])
        full_ep = full_by_identity[identity]
        left_dir = full / "source/features" / f"episode_{full_ep['global_episode_idx']}"
        right_dir = subset / "source/features" / f"episode_{subset_ep['global_episode_idx']}"
        names = {f"token_emb_{step}.npy" for step in range(subset_ep["num_timesteps"])} | {"kept_indices.json"}
        for directory in (left_dir, right_dir):
            _exact_entries(directory, names)
        for name in sorted(names):
            left, right = left_dir / name, right_dir / name
            if not _bytes_equal(left, right):
                raise ValueError(f"feature 字节不符 {identity}/{name}: {feature_difference(left, right)}")
            features += int(name.endswith(".npy"))
        for local_index, step in enumerate(range(subset_ep["exec_start_idx"], subset_ep["num_timesteps"])):
            left = full / "source/data" / f"{full_ep['exec_sample_offset'] + local_index}.pkl"
            right = subset / "source/data" / f"{subset_ep['exec_sample_offset'] + local_index}.pkl"
            compare_sample_files(left, right, full_ep, subset_ep, step)
            samples += 1
        if index % 10 == 0 or index == len(subset_manifest["episodes"]):
            print(f"SUBSET_PROGRESS episodes={index}/{totals['episodes']} frames={features} samples={samples}", flush=True)
    if samples != totals["exec_samples"] or features != totals["timesteps"]:
        raise ValueError(f"子集实际比较数量不符: frames={features} samples={samples}")
    return {"episodes": totals["episodes"], "samples": samples, "frames": features,
            "full_manifest_sha256": full_manifest["sha256"], "subset_manifest_sha256": subset_manifest["sha256"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--subset", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = check_subset(args.full, args.subset)
    except (OSError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        print(f"SUBSET_EQ=FAIL {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"SUBSET_EQ=PASS episodes={result['episodes']} samples={result['samples']} "
          "feature_mismatch=0 sample_mismatch=0", flush=True)
    print(f"SUBSET_MANIFEST full={result['full_manifest_sha256']} subset={result['subset_manifest_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
