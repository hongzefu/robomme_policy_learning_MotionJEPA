#!/usr/bin/env python3
"""集群预处理链路的守卫回归测试。

这批守卫全部来自 2026-08-24 的对抗审查：它们的共同毛病是「失败真正发生时会静默放行」，
所以每条用例都刻意制造那个失败，断言守卫必须亮红灯。守卫将来还会被改，
一次性验证脚本改一次就得重写一次，因此固化成 pytest 用例随仓库走。

跑法（禁止裸 python，见 AGENTS.md 第 3 条）：
  UV_LINK_MODE=copy JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
    uv run pytest scripts/data-preprocess-GL/test_guards.py -q

`JAX_PLATFORMS=cpu` 是必需的：build_shard 的导入链会拉起 jax，不设它会去抢 GPU。
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_HERE))


# ── 原子写：字节必须零变化（第一层 bitexact 按字节比 kept_indices.json）────────────
def test_atomic_write_json_bytes_unchanged(tmp_path: pathlib.Path) -> None:
    from mme_vla_suite.dataset_builder.build_robomme_dataset import atomic_write_json

    obj = [[0, 0, 1], [3, 0, 7], [12, 5, 63]]
    p = tmp_path / "kept_indices.json"
    atomic_write_json(str(p), obj)

    # 与旧写法 json.dump(obj, f) 的字节逐字节相同——分隔符/编码任何变化都会让第一层炸
    assert p.read_bytes() == json.dumps(obj).encode()
    # 成功路径不留 .tmp 残留
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_json_overwrites_existing(tmp_path: pathlib.Path) -> None:
    from mme_vla_suite.dataset_builder.build_robomme_dataset import atomic_write_json

    p = tmp_path / "kept_indices.json"
    p.write_text("[[9,9,9]]")
    atomic_write_json(str(p), [[1, 1, 1]])
    assert json.loads(p.read_text()) == [[1, 1, 1]]


# ── 分片续跑的完整性判据 ──────────────────────────────────────────────────────
def _mini_shard_lib(root: pathlib.Path, *, num_timesteps: int, exec_offset: int,
                    exec_samples: int, kept_content: str | None) -> dict:
    """造一个只有单 episode 的最小产物库，返回它的清单条目。

    kept_content=None 表示不写 kept_indices.json；传字符串则原样写入
    （用于制造半截 JSON、空文件这类「文件存在但内容坏了」的场景）。
    """
    d = root / "features" / "episode_0"
    d.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    for i in range(num_timesteps):
        np.save(d / f"token_emb_{i}.npy", np.zeros(2, dtype=np.float32))
    for i in range(exec_offset, exec_offset + exec_samples):
        (root / "data" / f"{i}.pkl").write_bytes(pickle.dumps({"epis_idx": 0}))
    if kept_content is not None:
        (d / "kept_indices.json").write_text(kept_content)
    return {
        "global_episode_idx": 0,
        "num_timesteps": num_timesteps,
        "exec_sample_offset": exec_offset,
        "exec_samples": exec_samples,
    }


@pytest.mark.parametrize(
    ("kept_content", "expected", "why"),
    [
        ("[[0,0,1]]", True, "内容合法 → 完整"),
        ("[[0,0,1", False, "半截 JSON（写到一半被杀）→ 必须判不完整"),
        ("", False, "空壳文件（open 已创建、内容未落盘）→ 必须判不完整"),
        (None, False, "文件不存在 → 不完整"),
    ],
)
def test_episode_is_complete_checks_json_content(
    tmp_path: pathlib.Path, kept_content: str | None, expected: bool, why: str
) -> None:
    from build_shard import ShardProcessor

    ep = _mini_shard_lib(tmp_path, num_timesteps=3, exec_offset=5, exec_samples=2,
                         kept_content=kept_content)
    proc = ShardProcessor(str(tmp_path / "raw"), str(tmp_path))
    assert proc.episode_is_complete(ep) is expected, why


def test_episode_is_complete_rejects_missing_pkl(tmp_path: pathlib.Path) -> None:
    """kept_indices 合法、token_emb 齐全，但 exec 区间缺了一个 pkl —— 旧判据会放行。"""
    from build_shard import ShardProcessor

    ep = _mini_shard_lib(tmp_path, num_timesteps=3, exec_offset=5, exec_samples=2,
                         kept_content="[[0,0,1]]")
    (tmp_path / "data" / "6.pkl").unlink()
    proc = ShardProcessor(str(tmp_path / "raw"), str(tmp_path))
    assert proc.episode_is_complete(ep) is False


def test_episode_is_complete_rejects_missing_token_emb(tmp_path: pathlib.Path) -> None:
    from build_shard import ShardProcessor

    ep = _mini_shard_lib(tmp_path, num_timesteps=3, exec_offset=0, exec_samples=1,
                         kept_content="[[0,0,1]]")
    (tmp_path / "features" / "episode_0" / "token_emb_2.npy").unlink()
    proc = ShardProcessor(str(tmp_path / "raw"), str(tmp_path))
    assert proc.episode_is_complete(ep) is False


# ── finalize 的完整性核验 ─────────────────────────────────────────────────────
def _mini_manifest(num_timesteps: int, exec_samples: int) -> dict:
    return {
        "episodes": [{
            "global_episode_idx": 0, "h5_file": "a.h5", "raw_ep_idx": 0,
            "num_timesteps": num_timesteps, "exec_sample_offset": 0,
            "exec_samples": exec_samples,
        }],
        "totals": {"exec_samples": exec_samples, "timesteps": num_timesteps},
    }


def test_check_completeness_flags_bad_json(tmp_path: pathlib.Path) -> None:
    from finalize_checks import check_completeness

    _mini_shard_lib(tmp_path, num_timesteps=2, exec_offset=0, exec_samples=1,
                    kept_content="[[0,0,1")          # 半截
    errs, _ = check_completeness(_mini_manifest(2, 1), str(tmp_path))
    assert any("不是合法 JSON" in e for e in errs), errs


def test_check_completeness_flags_tmp_residue(tmp_path: pathlib.Path) -> None:
    from finalize_checks import check_completeness

    _mini_shard_lib(tmp_path, num_timesteps=2, exec_offset=0, exec_samples=1,
                    kept_content="[[0,0,1]]")
    (tmp_path / "features" / "episode_0" / "kept_indices.json.tmp").write_text("[[0,0")
    errs, _ = check_completeness(_mini_manifest(2, 1), str(tmp_path))
    assert any("残留原子写临时文件" in e for e in errs), errs


def test_check_completeness_passes_clean_lib(tmp_path: pathlib.Path) -> None:
    from finalize_checks import check_completeness

    _mini_shard_lib(tmp_path, num_timesteps=2, exec_offset=0, exec_samples=1,
                    kept_content="[[0,0,1]]")
    errs, stats = check_completeness(_mini_manifest(2, 1), str(tmp_path))
    assert errs == []
    assert stats == {"execution_samples": 1, "total_samples": 2}
