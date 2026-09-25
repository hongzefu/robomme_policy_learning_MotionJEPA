#!/usr/bin/env python3
"""绑定原版 80k 的公开 H5 摘要与 episode 身份；只读清单，不重读大型 H5。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore.manifest import manifest_sha256  # noqa: E402

FULL_TASKS = (
    "BinFill", "ButtonUnmask", "ButtonUnmaskSwap", "InsertPeg", "MoveCube", "PatternLock",
    "PickHighlight", "PickXtimes", "RouteStick", "StopCube", "SwingXtimes", "VideoPlaceButton",
    "VideoPlaceOrder", "VideoRepick", "VideoUnmask", "VideoUnmaskSwap",
)
COUNTING_TASKS = ("BinFill", "PickXtimes", "StopCube", "SwingXtimes")
EPISODES_PER_TASK = 100
FULL_TOTALS = {"episodes": 1600, "timesteps": 768897, "exec_samples": 476857}
COUNTING_TOTALS = {"episodes": 400, "timesteps": 189035, "exec_samples": 189035}
CONTENT_FIELDS = ("num_timesteps", "exec_start_idx", "exec_samples")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 键重复: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"JSON 含非有限常量: {value}")


def read_json(path: str | Path) -> dict:
    """读取严格 JSON，避免重复键被解析器静默覆盖。"""
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                       parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须为对象: {path}")
    return value


def read_manifest(path: str | Path) -> dict:
    value = read_json(path)
    if value.get("sha256") != manifest_sha256(value):
        raise ValueError(f"episode 清单 sha256 不符: {path}")
    return value


def file_names(tasks) -> set[str]:
    return {f"record_dataset_{task}.h5" for task in tasks}


def parse_tasks(raw: str | None) -> tuple[str, ...]:
    """本轮只允许公开全集或完整四任务 counting，不提供缩小验收的 CLI。"""
    if raw is None:
        return FULL_TASKS
    tasks = tuple(part.strip() for part in raw.split(","))
    if len(tasks) != len(set(tasks)) or set(tasks) != set(COUNTING_TASKS):
        raise ValueError(f"--tasks 必须恰为 {','.join(COUNTING_TASKS)}，不能缺失或重复")
    return COUNTING_TASKS


def _integer(value, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} 必须为不小于 {minimum} 的整数，实际为 {value!r}")
    return value


def validate_manifest(manifest: dict, tasks, totals: dict, *, label: str) -> list[tuple[str, int]]:
    """核验身份唯一、连续编号/偏移和总账；分片安排不参与内容判定。"""
    if type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError(f"{label}: 不支持的 manifest version")
    names = file_names(tasks)
    if manifest.get("canonical_order") != sorted(names):
        raise ValueError(f"{label}: canonical_order 不是指定任务的规范文件序")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{label}: episodes 必须为列表")
    identities = []
    seen = set()
    counts = Counter()
    exec_offset = total_offset = 0
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"{label}: episode {index} 不是对象")
        name = episode.get("h5_file")
        if name not in names:
            raise ValueError(f"{label}: 非目标 H5 {name!r}")
        raw_index = _integer(episode.get("raw_ep_idx"), f"{label} raw_ep_idx")
        identity = (name, raw_index)
        if identity in seen:
            raise ValueError(f"{label}: episode 物理身份重复 {identity}")
        seen.add(identity)
        identities.append(identity)
        counts[name] += 1
        n = _integer(episode.get("num_timesteps"), f"{identity} num_timesteps", minimum=1)
        start = _integer(episode.get("exec_start_idx"), f"{identity} exec_start_idx")
        samples = _integer(episode.get("exec_samples"), f"{identity} exec_samples")
        if start > n or samples != n - start:
            raise ValueError(f"{label}: {identity} 执行区间与 exec_samples 不符")
        for key, expected in (("global_episode_idx", index), ("exec_sample_offset", exec_offset),
                              ("total_sample_offset", total_offset)):
            if _integer(episode.get(key), f"{identity} {key}") != expected:
                raise ValueError(f"{label}: {identity} {key} 不连续，期望 {expected}")
        exec_offset += samples
        total_offset += n
    if counts != Counter(dict.fromkeys(names, EPISODES_PER_TASK)):
        raise ValueError(f"{label}: 每任务必须恰为 {EPISODES_PER_TASK} 集，实际为 {dict(counts)}")
    got = {"episodes": len(episodes), "timesteps": total_offset, "exec_samples": exec_offset}
    saved = manifest.get("totals")
    if not isinstance(saved, dict) or set(saved) != set(got):
        raise ValueError(f"{label}: totals 字段集合不符")
    for key, value in saved.items():
        _integer(value, f"{label} totals.{key}")
    if saved != got or got != totals:
        raise ValueError(f"{label}: totals 不符，记录 {saved}，实算 {got}，要求 {totals}")
    return identities


def compare_episode_manifests(current: dict, reference: dict, tasks) -> dict:
    reference_ids = validate_manifest(reference, FULL_TASKS, FULL_TOTALS, label="参考全集")
    if reference_ids != sorted(reference_ids):
        raise ValueError("参考全集 episode 顺序不是规范物理身份序")
    full = set(tasks) == set(FULL_TASKS)
    totals = FULL_TOTALS if full else COUNTING_TOTALS
    current_ids = validate_manifest(current, tasks, totals, label="本轮清单")
    expected_ids = [identity for identity in reference_ids if identity[0] in file_names(tasks)]
    if set(current_ids) != set(expected_ids):
        raise ValueError("episode 物理身份集合与参考投影不符（缺失或多余）")
    if full and current_ids != expected_ids:
        raise ValueError("完整库 episode 顺序与参考不符")
    expected = dict(zip(reference_ids, reference["episodes"], strict=True))
    for identity, episode in zip(current_ids, current["episodes"], strict=True):
        for key in CONTENT_FIELDS:
            if episode[key] != expected[identity][key]:
                raise ValueError(f"episode 内容不符: {identity} {key}")
    return totals


def _input_files(document: dict, names: set[str], label: str) -> dict:
    files = document.get("files")
    if not isinstance(files, dict) or set(files) != names:
        raise ValueError(f"{label}: H5 文件集合不符（缺失或多余）")
    if "count" in document and _integer(document["count"], f"{label} count") != len(files):
        raise ValueError(f"{label}: count 与文件集合不符")
    for name, entry in files.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{label}: {name} 摘要条目不是对象")
        _integer(entry.get("size"), f"{label} {name} size", minimum=1)
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"{label}: {name} sha256 必须为完整小写十六进制摘要")
    return files


def check_sources(input_path, reference_input_path, manifest_path, reference_manifest_path, tasks) -> dict:
    reference_input = read_json(reference_input_path)
    reference_files = _input_files(reference_input, file_names(FULL_TASKS), "参考输入")
    files = _input_files(read_json(input_path), file_names(tasks), "本轮输入")
    for name, entry in files.items():
        if any(entry[key] != reference_files[name][key] for key in ("size", "sha256")):
            raise ValueError(f"H5 来源 pin 不符: {name} size/sha256")
    current, reference = read_manifest(manifest_path), read_manifest(reference_manifest_path)
    totals = compare_episode_manifests(current, reference, tasks)
    return {
        "files": len(files), **totals,
        "manifest_sha256": current["sha256"], "reference_manifest_sha256": reference["sha256"],
        "input_manifest_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "reference_input_sha256": hashlib.sha256(Path(reference_input_path).read_bytes()).hexdigest(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--reference-input", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--tasks", default=None, help="仅 counting 模式传完整四任务 CSV")
    args = parser.parse_args(argv)
    try:
        result = check_sources(args.input_manifest, args.reference_input, args.manifest,
                               args.reference_manifest, parse_tasks(args.tasks))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ORIG80K_SOURCES=FAIL {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"INPUT_PIN=PASS files={result['files']}", flush=True)
    print(f"EPISODE_IDENTITY=PASS episodes={result['episodes']} timesteps={result['timesteps']} "
          f"samples={result['exec_samples']}", flush=True)
    print("SOURCE_PROVENANCE=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
