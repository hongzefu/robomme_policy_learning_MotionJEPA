"""episode_manifest.json 读取（sha256 fail-loud）——消费侧唯一真值源入口。

schema 与 `scripts/data-preprocess-GL/scan_manifest.py`（建库侧，原样保留）一致：
顶层 version / raw_dir / canonical_order / num_shards / totals / shard_load_timesteps /
episodes / sha256；每个 episode 记 global_episode_idx / h5_file / raw_ep_idx /
num_timesteps / exec_start_idx / exec_samples / exec_sample_offset /
total_sample_offset / shard_idx。

该目录名含连字符不是合法包名、跨目录 import 不成立，故 sha256 规范化算法在此
独立实现（与 scan_manifest.manifest_sha256 逐字同口径：剔除 sha256 字段后
sort_keys + 紧凑分隔符的 JSON 取 sha256）；打包工具、FrameSampDataset、对拍工具
一律从本包 import，不得各自复制。
"""

from __future__ import annotations

import hashlib
import json
import pathlib


def manifest_sha256(payload: dict) -> str:
    """对不含 sha256 字段的规范化 JSON 取摘要（与 scan_manifest.py 同口径）。"""
    body = {k: v for k, v in payload.items() if k != "sha256"}
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_manifest(path: str | pathlib.Path) -> dict:
    """读取并校验清单：sha256 不符（被改动）即显式 raise，绝不静默放行。"""
    p = pathlib.Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    expect = manifest_sha256(payload)
    got = payload.get("sha256")
    if got != expect:
        raise ValueError(
            f"清单 sha256 不符（已被改动？）: {p}: 记录 {got} != 现算 {expect}")
    return payload
