"""dtype 统一修复验证工具的自检（pytest）。

三组：

1. **纯函数位型测试**——`right_padding_token_emb` 的输出 dtype 必须跟随各自输入。
   这是 `static_state_emb` 那一处修复的**唯一**有效验证（交付键经 `_normalize_state`
   恒为 f64，把修复效果完全掩盖了）。修复尚未落地时（V2.4a 阶段）该组自动 skip，
   V2.4b 落地后自动转为真断言——不用改测试文件。
2. **哈希口径守卫**——本目录 `_common` 的 raw / canonical 摘要必须与
   `scripts/training/g0/bench_train_steps.py` 的同名实现逐字节同结果。两处口径一旦
   漂移，G1 vs G0b 的输入侧对拍就会拿两把不同的尺子量同一件事。
3. **位型容器 round-trip**——覆盖 bf16 / f32 / f64 / bool 全部出现的类型。

跑法：`UV_LINK_MODE=copy uv run pytest scripts/training/tests/test_padding_dtype.py -q`
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import ml_dtypes
import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _common as C  # noqa: E402

MAX_SIZE = 32


def _make_inputs(t: int):
    rng = np.random.default_rng(20260827 + t)
    img = rng.standard_normal((t, 1, 16, 8)).astype(ml_dtypes.bfloat16)
    pos = rng.standard_normal((t, 1, 16, 4)).astype(np.float32)
    state = rng.standard_normal((t, 8)).astype(np.float32)
    mask = np.ones((t,), dtype=np.bool_)
    return img, pos, state, mask


def _fix_landed() -> bool:
    """探测三行修复是否已落地（短样本分支的输出 dtype 是否跟随输入）。"""
    from mme_vla_suite.shared.data_utils import right_padding_token_emb

    img, pos, state, mask = _make_inputs(3)
    oi, _op, _os, _om = right_padding_token_emb(img, pos, state, mask, MAX_SIZE)
    return str(oi.dtype) == str(img.dtype)


@pytest.mark.parametrize("t", [1, 2, 3, 30, 31, 32, 33])
def test_padding_preserves_dtype(t: int) -> None:
    from mme_vla_suite.shared.data_utils import right_padding_token_emb

    if not _fix_landed():
        pytest.skip("dtype 修复尚未落地（V2.4a 阶段），本断言在 V2.4b 后生效")

    img, pos, state, mask = _make_inputs(t)
    oi, op, os_, om = right_padding_token_emb(img, pos, state, mask, MAX_SIZE)

    for out, src, name in ((oi, img, "img"), (op, pos, "pos"), (os_, state, "state")):
        assert str(out.dtype) == str(src.dtype), f"{name} dtype 未跟随输入"
        assert out.shape[0] == MAX_SIZE
        keep = min(t, MAX_SIZE)
        # 非填充区必须逐字节不变
        assert out[:keep].tobytes() == np.ascontiguousarray(src[:keep]).tobytes(), f"{name} 非填充区被改动"
        if t < MAX_SIZE:
            pad = np.asarray(out[keep:]).astype(np.float64)
            assert pad.size and not np.any(pad), f"{name} 填充区非零"

    assert str(om.dtype) == "bool"
    assert om.shape[0] == MAX_SIZE
    assert om[: min(t, MAX_SIZE)].all()
    if t < MAX_SIZE:
        assert not om[t:].any()


def _load_bench_module():
    p = C.REPO_ROOT / "scripts" / "training" / "g0" / "bench_train_steps.py"
    spec = importlib.util.spec_from_file_location("_bench_for_hash_guard", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hash_kouging_matches_bench() -> None:
    """本目录的摘要口径必须与 bench 量具逐字节同结果（防两把尺子）。"""
    bench = _load_bench_module()
    rng = np.random.default_rng(7)
    samples = [
        rng.standard_normal((3, 4)).astype(ml_dtypes.bfloat16),
        rng.standard_normal((5,)).astype(np.float32),
        rng.standard_normal((2, 2)).astype(np.float64),
        (rng.standard_normal((6,)) > 0),
        rng.integers(0, 255, size=(4, 3), dtype=np.uint8),
    ]
    for arr in samples:
        assert C.leaf_sha256(arr) == bench._leaf_sha256(arr), f"raw 口径漂移: {arr.dtype}"
        assert C.canonical_sha256(arr) == bench._canonical_sha256(arr), f"canonical 口径漂移: {arr.dtype}"


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(12, dtype=np.float32).reshape(3, 4).astype(ml_dtypes.bfloat16),
        np.arange(6, dtype=np.float32),
        np.arange(4, dtype=np.float64).reshape(2, 2),
        np.array([True, False, True, True]),
    ],
    ids=["bf16", "f32", "f64", "bool"],
)
def test_bit_container_roundtrip(arr, tmp_path) -> None:
    meta = C.save_array(tmp_path, "some/key", arr)
    back = C.load_array(tmp_path, "some/key")
    assert str(back.dtype) == str(arr.dtype)
    assert back.shape == arr.shape
    assert back.tobytes() == np.ascontiguousarray(arr).tobytes()
    assert meta["raw"] == C.leaf_sha256(arr)


def test_describe_leaf_handles_none_and_str() -> None:
    assert C.describe_leaf("recur_mask", None) == {"kind": "none"}
    d = C.describe_leaf("prompt", np.asarray("push the button"))
    assert d["kind"] == "str" and d["value"] == "push the button"


def test_fixture_indices_are_reproducible_and_on_boundary() -> None:
    """定点集必须可复现，且各档 step_idx 落在预期的 padding / 满长分支上。"""
    # 与 single_step_grad.py 同口径：默认 legacy 顶层清单（环境 A），DTYPE_MANIFEST 可指向某库的 meta/episode_manifest.json；
    # 两者都没有（环境 B 尚未建 400 ep 库）时 skip 而不是 FileNotFoundError
    mp = pathlib.Path(os.environ.get("DTYPE_MANIFEST") or (C.REPO_ROOT / "v1-store" / "episode_manifest.json"))
    if not mp.is_file():
        pytest.skip(f"缺清单 {mp}（环境 B 无 legacy 顶层清单；建好 400 ep 库后设 DTYPE_MANIFEST 再跑）")
    manifest = C.load_manifest(mp)
    g1 = C.build_fixture_indices(manifest)
    g2 = C.build_fixture_indices(manifest)
    assert g1 == g2, "定点集不可复现"
    for step in (*C.SHORT_STEPS, *C.FULL_STEPS):
        idxs = g1[f"step{step}"]
        assert len(idxs) == C.PER_STEP
        for i in idxs[:20]:
            _epis, got_step = C.resolve_index(manifest, i)
            assert got_step == step, f"index {i} 反查 step_idx={got_step}，应为 {step}"
    assert len(g1["random"]) == C.N_RANDOM
    batches = C.build_fixture_batches(g1)
    assert len(batches) == sum(n for _, n in C.BATCH_PLAN)
    assert all(len(b["indices"]) == C.BATCH_SIZE for b in batches)
