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


# ── NaN/Inf 零容差通道 ────────────────────────────────────────────────────────
def test_metrics_nan_flag_and_cosine_stays_one() -> None:
    """这条同时钉住「为什么必须独立成通道」：余弦对 NaN 行给的是满分 1.0。"""
    from compare_datasets import metrics

    a = np.ones((4, 8), dtype=np.float32)
    b = a.copy()
    b[1, 3] = np.nan
    m = metrics(a, b)
    assert m["has_nonfinite"] is True
    # 含 NaN 的那一行落进 ~ok 分支、cos 保留初值 1.0 —— 两道余弦判据依然稳稳过阈值
    # （比的是生产阈值而非精确 1.0：其余全一行的余弦本身就是 0.999999999999…）
    assert m["min_cosine"] >= 0.999
    assert m["p5_cosine"] >= 0.9999


def test_metrics_same_nan_bitpattern_is_false_pass() -> None:
    """两侧同一位置同一 NaN 位模式：bitwise_equal 假通过，只有 has_nonfinite 抓得住。"""
    from compare_datasets import metrics

    a = np.ones((2, 4), dtype=np.float32)
    a[0, 0] = np.nan
    m = metrics(a, a.copy())
    assert m["bitwise_equal"] is True
    assert m["has_nonfinite"] is True


def test_grid_metrics_nan_flag() -> None:
    from compare_datasets import grid_metrics

    a = np.ones((3, 3), dtype=np.float32)
    b = a.copy()
    b[2, 2] = np.inf
    assert grid_metrics(a, b)["has_nonfinite"] is True


def test_agg_nonfinite_not_folded_and_report_is_valid_json() -> None:
    from compare_datasets import Agg

    agg = Agg()
    agg.add({"bitwise_equal": True, "same_bit_frac": 1.0, "max_ulp": 0,
             "min_cosine": 1.0, "max_abs_diff": 0.0})
    agg.add({"bitwise_equal": True, "same_bit_frac": float("nan"), "max_ulp": 0,
             "min_cosine": float("nan"), "max_abs_diff": float("nan"),
             "err_floor_rel": float("nan"), "has_nonfinite": True})
    assert agg.has_nonfinite is True
    assert agg.n_nonfinite == 1
    assert agg.n == 2
    # NaN 一旦折叠进数值字段，json.dumps 会写出裸 NaN token —— 那不是合法 JSON，jq 会炸
    dumped = json.dumps(agg.as_dict())
    assert "NaN" not in dumped
    assert json.loads(dumped)["nonfinite_items"] == 1


def _ns(**kw):
    import argparse as _ap
    base = {"min_cosine": 0.999, "min_p5_cosine": 0.9999, "max_err_floor_rel": 0.05}
    base.update(kw)
    return _ap.Namespace(**base)


def _agg_with(**kw):
    from compare_datasets import Agg
    agg = Agg()
    agg.n = 1
    for k, v in kw.items():
        setattr(agg, k, v)
    return agg


@pytest.mark.parametrize("mode", ["bitexact", "crossarch", "downstream"])
def test_verdict_nonfinite_always_fails(mode: str) -> None:
    from compare_datasets import verdict

    aggs = {"state_emb": _agg_with(has_nonfinite=True, n_nonfinite=1)}
    assert verdict(mode, aggs, _ns()), f"{mode} 下 NaN 未判死"


# ── pos_emb 必须参与判定（审查前它在 crossarch/downstream 下只被打印）──────────
@pytest.mark.parametrize("mode", ["bitexact", "crossarch", "downstream"])
def test_verdict_pos_emb_is_zero_tolerance(mode: str) -> None:
    from compare_datasets import verdict

    bad = {"pos_emb_8x8": _agg_with(bitwise_equal=False)}
    good = {"pos_emb_8x8": _agg_with(bitwise_equal=True)}
    assert verdict(mode, bad, _ns()), f"{mode} 下 pos_emb 不逐位相同却未判死"
    assert verdict(mode, good, _ns()) == []


def test_verdict_ds_pos_emb_is_zero_tolerance() -> None:
    from compare_datasets import verdict

    aggs = {"ds_pos_emb": _agg_with(bitwise_equal=False)}
    assert verdict("downstream", aggs, _ns())


def test_verdict_ds_frames_present_is_zero_tolerance() -> None:
    from compare_datasets import verdict

    aggs = {"ds_frames_present": _agg_with(bitwise_equal=False)}
    fails = verdict("downstream", aggs, _ns())
    assert any("选帧目标文件" in f for f in fails), fails


def test_verdict_equiv_thresholds_still_fire() -> None:
    """判据重构不能把原有的三条等价阈值弄丢。"""
    from compare_datasets import verdict

    assert verdict("crossarch", {"image_emb_8x8": _agg_with(min_cosine=0.9)}, _ns())
    assert verdict("crossarch", {"image_emb_8x8": _agg_with(min_p5_cosine=0.99)}, _ns())
    assert verdict("crossarch", {"image_emb_8x8": _agg_with(max_err_floor_rel=0.9)}, _ns())
    # bitexact 模式下 image_emb 走零容差、不走阈值
    assert verdict("bitexact", {"image_emb_8x8": _agg_with(min_cosine=0.9)}, _ns()) == []


def test_deprecated_thresholds_are_rejected() -> None:
    """删掉的废弃参数必须明确报错，不能静默忽略（使用者会以为阈值生效）。"""
    import subprocess

    r = subprocess.run(
        [sys.executable, str(_HERE / "compare_datasets.py"), "--mode", "bitexact",
         "--manifest", "x", "--a_lib", "x", "--b_lib", "x", "--max_ulp", "1"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode != 0
    assert "unrecognized arguments" in r.stderr


# ── pkl / kept_indices 的聚合必须如实（审查前恒填 True）────────────────────────
def test_compare_episode_pkl_reports_false(tmp_path: pathlib.Path) -> None:
    from compare_datasets import Agg, compare_episode

    ep = {"h5_file": "a.h5", "raw_ep_idx": 0, "num_timesteps": 1, "exec_samples": 1}
    for side, arr in (("A", np.zeros(3, dtype=np.float32)), ("B", np.ones(3, dtype=np.float32))):
        d = tmp_path / side / "features" / "episode_0"
        d.mkdir(parents=True)
        (d / "kept_indices.json").write_text("[[0,0,1]]")
        np.save(d / "token_emb_0.npy", np.array({"state_emb": np.zeros(2, dtype=np.float32)}))
        (tmp_path / side / "data").mkdir(parents=True)
        (tmp_path / side / "data" / "0.pkl").write_bytes(
            pickle.dumps({"epis_idx": 0, "actions": arr}))

    aggs = {"kept_indices": Agg(), "pkl": Agg()}
    errs: list[str] = []
    a = {"local_g": 0, "exec_offset": 0, "ep": ep}
    compare_episode(str(tmp_path / "A"), str(tmp_path / "B"), a, dict(a), [0], aggs, errs)
    assert errs, "pkl 数组不同却没报错"
    assert aggs["pkl"].bitwise_equal is False, "pkl 聚合仍恒填 True"


def test_compare_episode_kept_indices_reports_false(tmp_path: pathlib.Path) -> None:
    from compare_datasets import Agg, compare_episode

    ep = {"h5_file": "a.h5", "raw_ep_idx": 0, "num_timesteps": 1, "exec_samples": 1}
    for side, kept in (("A", "[[0,0,1]]"), ("B", "[[0,0,2]]")):
        d = tmp_path / side / "features" / "episode_0"
        d.mkdir(parents=True)
        (d / "kept_indices.json").write_text(kept)
        np.save(d / "token_emb_0.npy", np.array({"state_emb": np.zeros(2, dtype=np.float32)}))
        (tmp_path / side / "data").mkdir(parents=True)
        (tmp_path / side / "data" / "0.pkl").write_bytes(pickle.dumps({"epis_idx": 0}))

    aggs = {"kept_indices": Agg(), "pkl": Agg()}
    errs: list[str] = []
    a = {"local_g": 0, "exec_offset": 0, "ep": ep}
    compare_episode(str(tmp_path / "A"), str(tmp_path / "B"), a, dict(a), [0], aggs, errs)
    # 审查前失败时根本不 add（n==0），verdict 的 `n == 0: continue` 会整项跳过
    assert aggs["kept_indices"].n == 1
    assert aggs["kept_indices"].bitwise_equal is False
