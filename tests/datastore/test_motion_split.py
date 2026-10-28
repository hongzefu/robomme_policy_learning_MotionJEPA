"""阶段二拆分的逐位合同、窗口容量与真实 loader／在线组包对拍。"""

import copy
import functools
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.motion_adapter import (
    PAIRING,
    STAGE2_LAYOUT,
    STAGE2_READOUT,
    STAGE2_SHAPE,
    STAGE2_SPLIT_READOUT,
    STAGE2_SPLIT_SHAPE,
    motion_policy_spec,
    pair_adjacent_tokens,
    split_stage2_tokens,
)
from mme_vla_suite.policies import policy as online
from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
from mme_vla_suite.shared.sampling import MEM_ORDER_SENTINEL, memory_order, pad_times
from mme_vla_suite.training.framesamp_dataset import FrameSampDataset, check_motion_contract

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/training"))
from test_framesamp_motion_stage2 import dataset, stage2_fx  # noqa: E402, F401
from test_motion_store_stage2 import baseline_module, stage2_identity  # noqa: E402


def split_config(**overrides):
    return dict(source_shape=[1, 1024], readout=STAGE2_SPLIT_READOUT,
                num_tokens=4, dim=256, budget=384, **overrides)


@pytest.mark.parametrize("shape", [(1, 1024), (0, 1, 1024), (7, 1, 1024), (2, 3, 1, 1024)])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_split_preserves_every_bit_and_pool_group(shape, noncontiguous):
    bits = np.random.default_rng(42).integers(0, 2**32, shape, dtype=np.uint32)
    if bits.size:
        bits.flat[:4] = [0, 0x80000000, 0x7FC00001, 0x7F800000]
    raw = bits.view(np.float32)
    if noncontiguous:
        raw = raw[..., ::-1]
    actual = split_stage2_tokens(raw)
    assert actual.shape == shape[:-2] + STAGE2_SPLIT_SHAPE
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    assert actual.tobytes() == raw.tobytes()
    for group in range(4):
        assert actual[..., group, :].tobytes() == raw[..., 0, group * 256:(group + 1) * 256].tobytes()


@pytest.mark.parametrize("raw", [np.zeros((4, 256), np.float32), np.zeros(1024, np.float32),
                                  np.zeros((1, 1024), np.float64), np.zeros((8, 256), np.float32), []])
def test_split_rejects_wrong_dtype_shape_and_repeated_split(raw):
    with pytest.raises((TypeError, ValueError)):
        split_stage2_tokens(raw)


def test_split_policy_spec_uses_window_budget_and_same_disk_layout():
    spec = motion_policy_spec(split_config())
    assert (spec.source_shape, spec.readout, spec.num_tokens, spec.dim) == (
        STAGE2_SHAPE, STAGE2_SPLIT_READOUT, 4, 256)
    assert (spec.budget, spec.window_budget, spec.layouts) == (384, 96, (STAGE2_LAYOUT,))


@pytest.mark.parametrize("budget", [0, -4, 1, 95, 383, 385, True, 384.0, "384"])
def test_split_rejects_invalid_budget(budget):
    cfg = split_config()
    cfg["budget"] = budget
    with pytest.raises(ValueError, match="budget"):
        motion_policy_spec(cfg)


@pytest.mark.parametrize("field,value", [("source_shape", [4, 256]), ("source_shape", None),
                                        ("source_shape", [True, 1024]), ("num_tokens", 1),
                                        ("num_tokens", True), ("dim", 1024)])
def test_split_rejects_wrong_source_and_output_contract(field, value):
    cfg = split_config()
    cfg[field] = value
    with pytest.raises(ValueError):
        motion_policy_spec(cfg)


@pytest.mark.parametrize("field", ["source_shape", "num_tokens", "dim", "budget"])
def test_split_requires_explicit_contract(field):
    cfg = split_config()
    del cfg[field]
    with pytest.raises(ValueError, match="缺字段"):
        motion_policy_spec(cfg)


@pytest.mark.parametrize("readout", [STAGE2_READOUT, STAGE2_SPLIT_READOUT])
@pytest.mark.parametrize("fault", [None, "arch", "weight_type", "checkpoint_sha256", "source_pin"])
def test_both_stage2_readouts_enforce_provenance(readout, fault):
    cfg = split_config(checkpoint_sha256="a" * 64)
    if readout == STAGE2_READOUT:
        cfg.update(readout=readout, num_tokens=1, dim=1024, budget=96)
    provenance = stage2_identity()
    meta = SimpleNamespace(raw={"layout": STAGE2_LAYOUT}, row_shape=STAGE2_SHAPE, provenance=provenance)
    if fault == "source_pin":
        del provenance[fault]
    elif fault:
        provenance["encoder"][fault] = "b" * 64 if fault == "checkpoint_sha256" else "错误身份"
    if fault:
        with pytest.raises(ValueError):
            check_motion_contract(cfg, meta)
    else:
        assert check_motion_contract(cfg, meta).readout == readout


def padded_split(count):
    ds = FrameSampDataset.__new__(FrameSampDataset)
    ds._motion_budget, ds._motion_window_budget, ds._motion_num_tokens = 384, 96, 4
    raw = np.arange(count * 1024, dtype=np.float32).reshape(count, 1, 1024)
    pos = np.arange(count * 256, dtype=np.float32).reshape(count, 256)
    starts = np.arange(count, dtype=np.int64) * 32
    return raw, pos, starts, ds._pad_motion(split_stage2_tokens(raw), pos, starts, count)


@pytest.mark.parametrize("count", [0, 1, 95, 96])
def test_split_padding_metadata_and_896_order(count):
    raw, pos, starts, (emb, pe, mask, times) = padded_split(count)
    assert emb.shape == pe.shape == (384, 256)
    assert emb.dtype == pe.dtype == np.float32 and mask.dtype == np.bool_
    assert emb[:4 * count].tobytes() == raw.tobytes()
    np.testing.assert_array_equal(pe[:4 * count], np.repeat(pos, 4, axis=0))
    np.testing.assert_array_equal(mask, np.arange(384) < 4 * count)
    np.testing.assert_array_equal(times[:4 * count], np.repeat(starts, 4))
    assert not np.any(emb[4 * count:]) and not np.any(pe[4 * count:])
    assert np.all(times[4 * count:] == MEM_ORDER_SENTINEL)
    frame_times = pad_times([0, 32, 64, 96], 32)
    order = memory_order(frame_times, 16, times)
    assert order.shape == (896,) and order.dtype == np.int32
    np.testing.assert_array_equal(np.sort(order), np.arange(896))
    identities = [(int(frame_times[i // 16]), 0, i) for i in range(512)]
    identities += [(int(times[i]), 1, 512 + i) for i in range(384)]
    np.testing.assert_array_equal(order, [row[2] for row in sorted(identities)])
    inverse = np.argsort(order)
    for window in range(96):
        positions = inverse[512 + window * 4:512 + (window + 1) * 4]
        np.testing.assert_array_equal(positions, positions[0] + np.arange(4))
    if count:
        assert inverse[:16].max() < inverse[512]


def test_split_rejects_97_windows_without_truncation():
    with pytest.raises(ValueError, match="97 > 窗口容量 96"):
        padded_split(97)
    mem = FrameSampMemory.__new__(FrameSampMemory)
    mem.exec_start_idx = 0
    mem.motion_budget = 96
    mem.motion_window = 33
    mem.motion_stride = 32
    mem.demo_min_real = 17
    with pytest.raises(RuntimeError, match="97 > motion.budget 96"):
        mem._prepare_motion(3104)


def test_split_loader_rejects_97_window_episode_at_construction(dataset, monkeypatch):
    cfg = copy.deepcopy(dataset.config)
    cfg.motion.update(readout=STAGE2_SPLIT_READOUT, num_tokens=4, dim=256, budget=384)
    monkeypatch.setattr(ms, "max_visible_count", lambda entry: 97)
    with pytest.raises(ValueError, match="97 > 窗口容量 96"):
        dataset.create(cfg)


@pytest.mark.parametrize("fusion,dim", [("context", 2048), ("modulation", 1024)])
@pytest.mark.parametrize("tokens_per_image", [16, 64])
@pytest.mark.parametrize("readout", [STAGE2_READOUT, STAGE2_SPLIT_READOUT])
def test_loader_and_online_policy_same_samples_bitwise(dataset, monkeypatch, fusion, dim, tokens_per_image, readout):
    cfg = copy.deepcopy(dataset.config)
    cfg.integration_type, cfg.memory_token_dim = fusion, dim
    cfg.token_per_image = tokens_per_image
    if readout == STAGE2_SPLIT_READOUT:
        cfg.motion.update(readout=readout, num_tokens=4, dim=256, budget=384)
    dataset.frame_meta.spec.tokens_per_frame = tokens_per_image
    ds = dataset.create(cfg)
    monkeypatch.delenv("MMEVLA_MOTION_OVERFLOW", raising=False)
    monkeypatch.setattr(online.nnx_utils, "module_jit", lambda fn: fn)
    monkeypatch.setattr(online, "FrameSampMemory", functools.partial(FrameSampMemory, max_steps=192))
    raw_by_start = {}
    model = SimpleNamespace(history_config=cfg, sample_actions=lambda *a: None, vision_encode=lambda *a: None)
    norm = {"state": SimpleNamespace(mean=np.zeros(8), std=np.ones(8))}
    policy = online.MME_VLA_Policy(model, norm_stats=norm,
                                  motion_enc_fn=lambda window, start: raw_by_start[start])
    assert policy._motion_cfg["budget"] == 96 and policy._motion_cfg["dim"] == 1024
    assert cfg.motion.budget == (384 if readout == STAGE2_SPLIT_READOUT else 96)

    def image_rows(rows):
        return np.broadcast_to(np.asarray(rows, np.float32)[:, None, None],
                               (len(rows), tokens_per_image, 2048)).copy()

    def state_rows(rows):
        return np.broadcast_to(np.asarray(rows, np.float32)[:, None], (len(rows), 8)).copy()

    for ep in dataset.fixture.fx.manifest["episodes"]:
        policy.reset()
        mem = policy.mem_buffer
        es, length, g = ep["exec_start_idx"], ep["num_timesteps"], ep["global_episode_idx"]
        mem.exec_start_idx = es
        mem._raw_frames = dict.fromkeys(range(length), np.zeros((256, 256, 3), np.uint8))
        entry = ds._motion_entries[g]
        rows, starts = ms.visible_motion_rows(entry, length - 1)
        raw_by_start.clear()
        raw_by_start.update(zip(starts, ds._ensure_motion_store().rows(rows).reshape(-1, 1024), strict=True))
        for start in starts:
            # 真正经过底层编码形状检查，防止误把 256 维策略合同传给 1024 维 sidecar。
            mem._encode_window(int(start))
        for frame in range(length):
            row = int(ds._row_base[g]) + frame
            mem._history_feats[frame] = {mem.image_key: image_rows([row]),
                mem.pos_key: mem.pos_emb[frame:frame + 1], "state_emb": state_rows([row])[0]}
        frame_store = SimpleNamespace(read_image_rows=image_rows, state_rows=state_rows,
                                      pos_rows=lambda frames: mem.pos_emb[frames])
        ds._ensure_store = lambda: frame_store
        for step in sorted({es, min(es + 31, length - 1), min(es + 32, length - 1), length - 1}):
            policy.step_idx = step
            actual = policy._prepare_history({})
            expected = ds[ep["exec_sample_offset"] + step - es]
            assert actual["mem_order"].shape == ((896,) if readout == STAGE2_SPLIT_READOUT else (608,))
            for key, value in actual.items():
                assert value.dtype == expected[key].dtype and value.shape == expected[key].shape, key
                assert value.tobytes() == expected[key].tobytes(), (g, step, key)
    ds.close()


def test_legacy_concat_and_pair_adapter_unchanged():
    concat = motion_policy_spec(dict(source_shape=[1, 1024], readout=STAGE2_READOUT,
                                    num_tokens=1, dim=1024, budget=96))
    assert (concat.dim, concat.budget, concat.window_budget, concat.num_tokens) == (1024, 96, 96, 1)
    old = baseline_module("src/mme_vla_suite/datastore/motion_adapter.py", "baseline_motion_adapter_split")
    cfg = dict(source_shape=[8, 256], readout=PAIRING, num_tokens=4, dim=512, budget=640)
    assert vars(motion_policy_spec(cfg)) == vars(old.motion_policy_spec(cfg))
    raw = np.random.default_rng(42).integers(0, 2**32, (7, 8, 256), dtype=np.uint32).view(np.float32)
    assert pair_adjacent_tokens(raw).tobytes() == old.pair_adjacent_tokens(raw).tobytes()
