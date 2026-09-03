#!/usr/bin/env python3
"""集群预处理链路的守卫回归测试。

这批守卫全部来自 2026-08-24 的对抗审查：它们的共同毛病是「失败真正发生时会静默放行」，
所以每条用例都刻意制造那个失败，断言守卫必须亮红灯。守卫将来还会被改，
一次性验证脚本改一次就得重写一次，因此固化成 pytest 用例随仓库走。

跑法（禁止裸 python，见 AGENTS.md 第 3 条）：
  UV_LINK_MODE=copy JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
    uv run pytest scripts/dataset/test_guards.py -q

`JAX_PLATFORMS=cpu` 是必需的：build_shard 的导入链会拉起 jax，不设它会去抢 GPU。

v2-motionmem 起（集群链路删除后）新增：motion 表格式层与 wan 子项目公共件的常量 / 公式互证、
复制件 SOURCE_PIN 哨兵、motion_index 行序与起点可见集合的边界用例、本机 worker 的 claim 协议。
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
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
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


# ── 分片指纹的跨片同源断言 ────────────────────────────────────────────────────
def _fp(**kw) -> dict:
    base = {"host": "gl1500", "slurm_job": "1", "gpu_device_kind": "NVIDIA A40",
            "jax": "0.5.3", "jaxlib": "0.5.3", "git_commit": "abc123",
            "resource_tier": {"cpus_per_task": "2", "mem_per_node_mb": "24576"}}
    base.update(kw)
    return base


def test_aggregate_fingerprints_all_same() -> None:
    from finalize_checks import aggregate_shard_fingerprints

    docs = [(f"_shard{i}of3.json", {"schema_version": 2, "fingerprint": _fp(host=f"gl150{i}")})
            for i in range(3)]
    agg, errs = aggregate_shard_fingerprints(docs)
    assert errs == []
    assert agg["gpu_device_kind"] == "NVIDIA A40"
    # host 允许不同（array task 本就落在不同节点），只记录不断言
    assert agg["hosts"] == ["gl1500", "gl1501", "gl1502"]
    assert agg["shards_with_fingerprint"] == 3


def test_aggregate_fingerprints_detects_mismatch() -> None:
    from finalize_checks import aggregate_shard_fingerprints

    docs = [("_shard0of2.json", {"schema_version": 2, "fingerprint": _fp()}),
            ("_shard1of2.json", {"schema_version": 2, "fingerprint": _fp(git_commit="deadbee")})]
    _, errs = aggregate_shard_fingerprints(docs)
    assert any("git_commit 不一致" in e for e in errs), errs


def test_aggregate_fingerprints_detects_jax_mismatch() -> None:
    from finalize_checks import aggregate_shard_fingerprints

    docs = [("_shard0of2.json", {"schema_version": 2, "fingerprint": _fp()}),
            ("_shard1of2.json", {"schema_version": 2, "fingerprint": _fp(jax="0.6.0")})]
    _, errs = aggregate_shard_fingerprints(docs)
    assert any("jax 不一致" in e for e in errs), errs


def test_aggregate_fingerprints_flags_legacy_sidecar() -> None:
    """产出于指纹引入之前的旧 sidecar：必须 fail-loud 且消息可读。"""
    from finalize_checks import aggregate_shard_fingerprints

    docs = [("_shard0of1.json", {"shard_idx": 0, "steps": 100})]
    _, errs = aggregate_shard_fingerprints(docs)
    assert any("产出于指纹字段引入之前" in e for e in errs), errs


def test_aggregate_fingerprints_flags_unavailable() -> None:
    """指纹采集失败等于没有指纹，不能当同源放行。"""
    from finalize_checks import aggregate_shard_fingerprints

    docs = [("_shard0of1.json",
             {"schema_version": 2, "fingerprint": _fp(gpu_device_kind="unavailable: no gpu")})]
    _, errs = aggregate_shard_fingerprints(docs)
    assert any("采集失败" in e for e in errs), errs


# ── 建库域冻结副本 sha256 哨兵（commitV4.0 起冻结，防与训练侧「好心同步」发散）────
# 常量在 V4.0 落地时固化：三个叶子文件与 shared/ 源逐字节相同，
# mem_buffer.py 恰好改 3 行 import（shared.→dataset_builder.）。
# V4.4（计划 V3.12）之后 shared/ 侧源文件或删或改，这四条哨兵是唯一在岗的冻结证明。
_FROZEN_BUILDER_SHA256 = {
    "data_utils.py": "20f9e7163dff43c9adb6ab7443f2427cdeb4eeabeaf763ef5ffeb7be8214f4ee",
    "posemb_3d.py": "65e232935010b0dfebb46288c74da7ce0ce638cc8db508cd56cfb84e72caba0b",
    "siglip_tokenizer.py": "72fb842327467a4d7cb0f770a514278d67b20721c84d59c82e6cae25f4ce0858",
    "mem_buffer.py": "76e20064c3bcd0b9619ab2472b46643f28fc96f72e446623467c09b5214e6075",
}


@pytest.mark.parametrize("fname", sorted(_FROZEN_BUILDER_SHA256))
def test_builder_copy_frozen_sha256(fname: str) -> None:
    import hashlib
    p = _REPO_ROOT / "src" / "mme_vla_suite" / "dataset_builder" / fname
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    assert got == _FROZEN_BUILDER_SHA256[fname], (
        f"建库域冻结副本 {fname} 内容已变（sha256 {got}）。该文件自 commitV4.0 起冻结，"
        "任何改动（包括从训练侧同步）都会破坏建库产物一致性证明；如确需变更须用户拍板并更新常量。"
    )


# ── 建库输出 --force 闸：目标已存在且未给 force 时必须拒绝，且目录原样保留 ────────
def test_dataset_processor_refuses_existing_output_without_force(
    tmp_path: pathlib.Path,
) -> None:
    from mme_vla_suite.dataset_builder.build_robomme_dataset import DatasetProcessor

    out = tmp_path / "existing_out"
    out.mkdir()
    sentinel = out / "precious.txt"
    sentinel.write_text("不可误删")

    with pytest.raises(FileExistsError, match="--force"):
        DatasetProcessor(
            raw_data_path=str(tmp_path / "raw"),
            preprocessed_data_path=str(out),
        )
    # 拒绝路径必须零副作用：目录与内容原样保留
    assert sentinel.read_text() == "不可误删"


# ── v2-motionmem：motion 表格式层与 wan 子项目公共件必须逐项同值 ─────────────────
def _wan_common():
    import importlib.util
    spec = importlib.util.spec_from_file_location("wan_common", _HERE / "wan" / "wan_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wan_common_constants_match_motion_store() -> None:
    from mme_vla_suite.datastore import motion_store as ms
    wc = _wan_common()
    for k in ("GRID_STRIDE", "WINDOW_FRAMES", "GRID_ORIGIN", "WINDOW_DIRECTION", "TRUNCATION_POLICY", "FRAME_SIZE"):
        assert getattr(wc, k) == getattr(ms, k), k
    assert wc.TOKEN_BYTES == ms.MOTION_ROW_BYTES and wc.TOKEN_DIM == ms.MOTION_ROW_SHAPE[0]
    for L in range(0, 1300):
        assert wc.seg_num_chunks(L) == ms.seg_num_chunks(L), L
        assert wc.seg_num_grid(L) == ms.seg_num_grid(L), L
    # 计划 2.2 / 4.1 的实测锚点：VideoPlaceOrder ep4 demo 1118 + exec 293 → 68 + 17；exec 段不截尾
    assert ms.seg_num_grid(1118) == 68 and ms.seg_num_grid(293) == 17
    assert ms.seg_num_grid(32) == 0 and ms.seg_num_grid(33) == 1 and ms.seg_num_grid(48) == 1 and ms.seg_num_grid(49) == 2


def test_source_pin_matches_copied_module() -> None:
    import hashlib
    pin = json.loads((_HERE / "wan" / "SOURCE_PIN.json").read_text(encoding="utf-8"))
    got = hashlib.sha256((_HERE / "wan" / "wan_motion_infer.py").read_bytes()).hexdigest()
    assert got == pin["source_sha256"], "复制件 wan_motion_infer.py 与 SOURCE_PIN.source_sha256 不符（复制件被改动）"
    assert pin["mj_repo_commit"] == "2a484ad960ed6155321dc34def9011eb119f857f"


def _mini_manifest_for_motion(specs):
    """specs = [(num_timesteps, exec_start_idx), ...] → 带 sha256 的迷你清单。"""
    from mme_vla_suite.datastore.manifest import manifest_sha256
    eps, eo, to = [], 0, 0
    for g, (nt, es) in enumerate(specs):
        eps.append({"global_episode_idx": g, "h5_file": "record_dataset_T.h5", "raw_ep_idx": g,
                    "num_timesteps": nt, "exec_start_idx": es, "exec_samples": nt - es,
                    "exec_sample_offset": eo, "total_sample_offset": to, "shard_idx": 0})
        eo += nt - es
        to += nt
    m = {"version": 1, "raw_dir": "/x", "canonical_order": ["record_dataset_T.h5"], "num_shards": 1,
         "totals": {"episodes": len(eps), "timesteps": to, "exec_samples": eo},
         "shard_load_timesteps": [to], "episodes": eps}
    m["sha256"] = manifest_sha256(m)
    return m


def test_motion_index_roundtrip_and_totals() -> None:
    from mme_vla_suite.datastore import motion_store as ms
    m = _mini_manifest_for_motion([(291, 0), (338, 66), (586, 216), (40, 0), (32, 0)])
    entries = ms.build_index_entries(m)
    payload = ms.index_payload(m, entries, mj_repo_commit="abc")
    back = ms.parse_index(payload)
    assert [e for e in back] == [e for e in entries]
    t = ms.index_totals(entries)
    # 291 exec → 17；demo 66 → 3、exec 272 → 15；demo 216 → 12、exec 370 → 22；exec 40 → 1；exec 32 → 0
    assert (t["exec_rows"], t["demo_rows"]) == (17 + 15 + 22 + 1, 3 + 12)
    assert entries[0].demo.row_base is None and entries[0].exec.row_base == 0
    assert entries[1].demo.row_base == 17 and entries[1].exec.row_base == 20
    ms.check_index_against_manifest(entries, m)
    # 篡改 row_base 必 raise
    bad = json.loads(json.dumps(payload))
    bad["entries"][1]["exec"]["row_base"] = 21
    with pytest.raises(ValueError, match="row_base"):
        ms.parse_index(bad)


def test_visible_motion_rows_boundaries() -> None:
    from mme_vla_suite.datastore import motion_store as ms
    m = _mini_manifest_for_motion([(338, 66), (291, 0), (100, 33), (100, 32)])
    e = ms.build_index_entries(m)
    # demo 段 66 帧 → 起点 0,16,32（32+32=64 ≤ 65）；exec 段 t-es<32 时无 exec 起点
    r, f = ms.visible_motion_rows(e[0], 66)
    assert f.tolist() == [0, 16, 32] and r.tolist() == [0, 1, 2]
    r, f = ms.visible_motion_rows(e[0], 66 + 31)
    assert f.tolist() == [0, 16, 32]
    r, f = ms.visible_motion_rows(e[0], 66 + 32)          # 第一个 exec 起点 u=0 恰好可见
    assert f.tolist() == [0, 16, 32, 66] and r.tolist() == [0, 1, 2, 3]
    r, f = ms.visible_motion_rows(e[0], 66 + 47)
    assert f.tolist() == [0, 16, 32, 66]
    r, f = ms.visible_motion_rows(e[0], 66 + 48)
    assert f.tolist() == [0, 16, 32, 66, 82]
    assert ms.max_visible_count(e[0]) == 3 + 15
    # es=0：t=31 无起点，t=32 一个
    assert ms.visible_motion_rows(e[1], 31)[0].size == 0
    assert ms.visible_motion_rows(e[1], 32)[1].tolist() == [0]
    # demo 恰好 33 帧 → 1 个 demo 起点；demo 32 帧 → 0 个
    assert ms.visible_motion_rows(e[2], 33)[1].tolist() == [0]
    assert ms.visible_motion_rows(e[3], 32)[1].size == 0
    with pytest.raises(ValueError):
        ms.visible_motion_rows(e[0], 65)                   # t 落在 demo 段不是 exec 样本


def test_segment_key_and_task_parsing() -> None:
    from mme_vla_suite.datastore import motion_store as ms
    assert ms.segment_key("record_dataset_VideoUnmaskSwap.h5", 3, "exec") == "VideoUnmaskSwap_ep3_exec"
    with pytest.raises(ValueError):
        ms.task_of_h5("foo.h5")


def test_worker_claim_is_exclusive(tmp_path: pathlib.Path) -> None:
    from build_shard import release_claim_episode, try_claim_episode
    assert try_claim_episode(str(tmp_path), 7, "gpu0") is True
    assert try_claim_episode(str(tmp_path), 7, "gpu1") is False
    release_claim_episode(str(tmp_path), 7)
    assert try_claim_episode(str(tmp_path), 7, "gpu1") is True
    wc = _wan_common()
    claims = tmp_path / "_claims"
    assert wc.try_claim(claims, "T_ep0_exec", "gpu0") and not wc.try_claim(claims, "T_ep0_exec", "gpu1")
    wc.release_claim(claims, "T_ep0_exec")
    assert wc.try_claim(claims, "T_ep0_exec", "gpu1")


def test_wan_common_list_segments_matches_index() -> None:
    from mme_vla_suite.datastore import motion_store as ms
    wc = _wan_common()
    m = _mini_manifest_for_motion([(291, 0), (338, 66), (586, 216), (32, 0)])
    segs = wc.list_segments(m)
    entries = ms.build_index_entries(m)
    exp = []
    for e in entries:
        for seg in ms.SEGMENTS:
            s = getattr(e, seg)
            if s.num_grid:
                exp.append((ms.segment_key(e.h5_file, e.raw_ep_idx, seg), s.num_grid, s.seg_len))
    assert [(x["key"], x["num_grid"], x["seg_len"]) for x in segs] == exp
    assert [x["key"] for x in wc.lpt_order(segs)][0] == "T_ep2_exec"


def test_worker_processor_refreshes_completeness_snapshot(tmp_path: pathlib.Path) -> None:
    """分片模式缓存一次 data/ 快照；worker 模式下另一 worker 后写的 pkl 必须被看见，否则会 purge 重做。"""
    from build_shard import ShardProcessor, WorkerProcessor

    ep = _mini_shard_lib(tmp_path, num_timesteps=3, exec_offset=5, exec_samples=2, kept_content="[[0,0,1]]")
    shard = ShardProcessor(str(tmp_path / "raw"), str(tmp_path))
    worker = WorkerProcessor(str(tmp_path / "raw"), str(tmp_path))
    assert shard.episode_is_complete(ep) and worker.episode_is_complete(ep)
    # 「另一 worker」此刻才写出 episode 1 的产物
    d = tmp_path / "features" / "episode_1"
    d.mkdir()
    np.save(d / "token_emb_0.npy", np.zeros(2, dtype=np.float32))
    (d / "kept_indices.json").write_text("[[0,0,1]]")
    (tmp_path / "data" / "9.pkl").write_bytes(pickle.dumps({"epis_idx": 1}))
    ep1 = {"global_episode_idx": 1, "num_timesteps": 1, "exec_sample_offset": 9, "exec_samples": 1}
    assert shard.episode_is_complete(ep1) is False, "分片模式快照语义保持不变"
    assert worker.episode_is_complete(ep1) is True, "worker 模式必须刷新快照"


# ── v2-motionmem S2：MotionStore 的 spawn / pickle 契约（motion-memory-plan.md 2.8）────────────────
def _real_lib() -> pathlib.Path | None:
    lib = _REPO_ROOT / "v1-store" / "datasets" / "4task-motion-40ep"
    alt = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", "")) / "datasets" / "4task-motion-40ep" if os.environ.get("MMEVLA_V1_STORE") else None
    for p in (lib, alt):
        if p is not None and (p / "motion" / "meta" / "store_meta.json").is_file() and (p / "framesamp" / "meta" / "store_meta.json").is_file():
            return p
    return None


def test_motion_store_pickle_refused_and_owner_pid() -> None:
    from mme_vla_suite.datastore import motion_store as ms
    lib = _real_lib()
    if lib is None:
        pytest.skip("缺 40 ep 真实库")
    store = ms.MotionStore(lib / "motion")
    assert store.owner_pid == os.getpid() and store.num_rows == 772 and store.table.shape == (772, 768)
    with pytest.raises(TypeError, match="禁止 pickle"):
        pickle.dumps(store)
    store.close()


def _worker_probe(lib_str: str, q) -> None:
    """spawn worker：构造后取一个样本，回报 MotionStore owner_pid 与本进程 pid。"""
    import os as _os
    import sys as _sys
    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
    from mme_vla_suite.training.dataloader import _create_framesamp_dataset
    import json as _json, types as _types, numpy as _np, omegaconf
    lib = pathlib.Path(lib_str)
    v1 = pathlib.Path(_os.environ.get("MMEVLA_V1_STORE", str(pathlib.Path(__file__).resolve().parents[2] / "v1-store")))
    ns = _json.load(open(v1 / "train-assets/mme_vla_suite/robomme/norm_stats.json"))["norm_stats"]["state"]
    st = _types.SimpleNamespace(q01=_np.array(ns["q01"]), q99=_np.array(ns["q99"]), mean=_np.array(ns["mean"]), std=_np.array(ns["std"]))
    _os.environ["MMEVLA_MOTION_STORE"] = str(lib / "motion")
    hc = omegaconf.OmegaConf.load(pathlib.Path(__file__).resolve().parents[2] / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context-motion.yaml")
    ds = _create_framesamp_dataset(str(lib / "framesamp"), _types.SimpleNamespace(norm_stats={"state": st}, use_quantile_norm=True), hc, 20)
    state = ds.__getstate__()
    d = ds[100]
    q.put({"pid": _os.getpid(), "owner_pid": ds._mstore.owner_pid, "getstate_mstore_none": state["_mstore"] is None,
           "getstate_store_none": state["_store"] is None, "k": int(d["motion_mask"].sum())})


def test_framesamp_dataset_motion_store_lazy_in_spawn_worker() -> None:
    """dataset 主进程只读 meta；spawn worker 内懒构造 MotionStore（owner_pid == worker pid）；__getstate__ 剥离双 store。"""
    import multiprocessing as mp
    lib = _real_lib()
    if lib is None:
        pytest.skip("缺 40 ep 真实库")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker_probe, args=(str(lib), q))
    p.start()
    res = q.get(timeout=600)
    p.join(timeout=60)
    assert p.exitcode == 0
    assert res["owner_pid"] == res["pid"] != os.getpid()
    assert res["getstate_mstore_none"] and res["getstate_store_none"]
    assert res["k"] == 5
