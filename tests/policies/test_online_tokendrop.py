"""在线 tokendrop 与建库域 MemoryBuffer 的对拍；编码器为确定性替身，不跑真实模型。"""

import functools
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from omegaconf import OmegaConf

from mme_vla_suite.dataset_builder.mem_buffer import MemoryBuffer
from mme_vla_suite.policies import policy as policy_module
from mme_vla_suite.policies.framesamp_memory import FrameSampMemory

REPO = Path(__file__).resolve().parents[2]
MAX_STEPS = 132  # 多视角夹具按 step*v 取位置表，65 帧 × 2 视角需要 130 行


def stub_encoder(images):
    """每帧 token 随像素变化，能查出帧与 patch 的搬运错误；输出 (t,v,256,2048) 供三档池化。"""
    t, v = images.shape[:2]
    frame_values = np.asarray(images).mean(axis=(2, 3, 4), dtype=np.float32)
    channels = np.arange(2048, dtype=np.float32) / 2048
    tokens = np.arange(256, dtype=np.float32)[:, None] + channels[None, :]
    return (frame_values[:, :, None, None] + tokens[None, None]).astype(np.dtype("bfloat16"))


def make_frames(count, size=64, views=1, seed=0):
    """混合静止 patch（分数 <1e-4 被跳过）、并列分数（靠后续字段定序）与随机 patch。"""
    rng = np.random.default_rng(seed)
    cell = size // 8
    frames = np.zeros((count, views, size, size, 3), np.uint8)
    for index in range(count):
        for view in range(views):
            frame = frames[index, view]
            frame[:] = 40
            # 第 0 行随帧变化且 8 个 patch 差异相同：分数并列，只能按帧、视角、patch 定序。
            frame[:cell] = (index * 3) % 256
            # 第 1 行每帧随机，第 2 行起静止。
            frame[cell:2 * cell] = rng.integers(0, 256, frame[cell:2 * cell].shape, dtype=np.uint8)
    states = np.asarray([[index + axis / 16 for axis in range(8)] for index in range(count)], np.float32)
    return frames, states


def upstream_memory(views=1, keptsize=2048, stride=8):
    return MemoryBuffer(
        num_views=views, compute_token_drop_score=True, token_drop_stride=stride,
        token_drop_keptsize=keptsize, prepare_buffer=True, vision_enc_fn=stub_encoder, max_steps=MAX_STEPS,
    )


def online_memory(views=1, keptsize=2048, stride=8):
    return FrameSampMemory(
        num_views=views, vision_enc_fn=stub_encoder, token_per_image=64, max_steps=MAX_STEPS,
        token_drop_stride=stride, token_drop_keptsize=keptsize,
    )


def feed_online(memory, frames, states, sizes):
    start = 0
    for size in sizes:
        ids = list(range(start, min(start + size, len(frames))))
        if ids:
            memory.add_buffer(frames[ids], states[ids], ids)
        start += size
    assert start >= len(frames)


def feed_upstream(memory, frames, states):
    for index in range(len(frames)):
        memory.add_buffer(frames[index:index + 1], states[index:index + 1], [index])


def snapshot_heap(memory):
    return sorted((float(score), step, view, patch) for score, step, view, patch in memory.scored_token_heap)


@pytest.mark.parametrize("views,keptsize,stride,sizes", [
    (1, 2048, 8, [1, 16, 16, 16, 16, 16]),   # Button 类：首包 1 帧
    (1, 2048, 8, [33, 16, 16]),              # Video 类：首包 D+1 帧
    (1, 150, 8, [65]),                       # 容量淘汰；首帧 64 个不淘汰
    (1, 10, 8, [5, 7, 53]),                  # 容量小于首帧 token 数
    (2, 300, 8, [20, 45]),                   # 多视角
    (1, 2048, 1, [65]),                      # stride=1 时 step 0 也进打分分支
])
def test_scoring_and_prepare_match_upstream(views, keptsize, stride, sizes):
    frames, states = make_frames(65, views=views)
    upstream = upstream_memory(views, keptsize, stride)
    online = online_memory(views, keptsize, stride)
    feed_upstream(upstream, frames, states)
    feed_online(online, frames, states, sizes)

    assert online.token_drop_last_frame == upstream.token_drop_last_frame
    assert snapshot_heap(online) == snapshot_heap(upstream)
    upstream_indices = upstream.get_token_dropping_indices()
    assert online.get_token_dropping_indices() == upstream_indices
    # 取索引不得破坏堆（每步 infer 都要取）。
    assert online.get_token_dropping_indices() == upstream_indices

    gather = upstream.default_history_feats_gather_fn
    for step in (0, 7, 8, 23, 40, 64):
        for budget in (32, 64, 1024):
            expected_indices = upstream.filter_token_dropping_indices(upstream_indices, step, budget)
            actual_indices = online.filter_token_dropping_indices(online.get_token_dropping_indices(), step, budget)
            assert actual_indices == expected_indices
            expected = upstream.prepare_token_dropping(step, budget, gather)
            actual = online.prepare_token_dropping(step, budget)
            for left, right in zip(actual, expected, strict=True):
                assert left.dtype == right.dtype and left.shape == right.shape
                assert left.tobytes() == right.tobytes()
            assert actual[0].shape == (budget, 2048) and actual[0].dtype == np.float32
            assert actual[1].shape == (budget, 768) and actual[1].dtype == np.float32
            assert actual[2].shape == (budget, 8) and actual[2].dtype == np.float32
            assert actual[3].dtype == np.bool_ and int(actual[3].sum()) == min(budget, len(expected_indices))


def test_real_frame_size_256_matches_upstream():
    frames, states = make_frames(17, size=256)
    upstream = upstream_memory()
    online = online_memory()
    feed_upstream(upstream, frames, states)
    feed_online(online, frames, states, [1, 16])
    assert online.get_token_dropping_indices() == upstream.get_token_dropping_indices()


def test_step_zero_push_does_not_evict():
    frames, states = make_frames(1)
    memory = online_memory(keptsize=10)
    memory.add_buffer(frames, states, [0])
    assert len(memory.scored_token_heap) == 64


def test_score_tie_breaks_by_later_fields():
    frames, states = make_frames(16)
    memory = online_memory()
    feed_online(memory, frames, states, [16])
    tied = [(step, patch) for step, view, patch in memory.get_token_dropping_indices() if patch < 8 and step == 7]
    assert tied == [(7, patch) for patch in range(8)]


def test_only_reference_frame_is_kept_and_input_not_aliased():
    frames, states = make_frames(20)
    memory = online_memory()
    feed_online(memory, frames, states, [20])
    assert np.array_equal(memory._token_drop_ref, frames[15])
    frames[15] = 0
    assert not np.array_equal(memory._token_drop_ref, frames[15])
    assert memory._raw_frames == {}
    assert set(memory._history_feats[3]) == {"image_emb_8x8", "pos_emb_8x8", "state_emb"}


def test_clear_resets_scoring_state():
    frames, states = make_frames(17)
    memory = online_memory()
    feed_online(memory, frames, states, [17])
    first = memory.get_token_dropping_indices()
    memory.clear()
    assert memory.scored_token_heap == [] and memory.token_drop_last_frame == -1 and memory._token_drop_ref is None
    feed_online(memory, frames, states, [17])
    assert memory.get_token_dropping_indices() == first


def test_disabled_memory_has_no_scoring_and_same_frame_sampling():
    frames, states = make_frames(33)
    off = FrameSampMemory(vision_enc_fn=stub_encoder, token_per_image=64, max_steps=MAX_STEPS)
    on = online_memory()
    assert off.token_drop_enabled is False
    feed_online(off, frames, states, [33])
    feed_online(on, frames, states, [33])
    assert off.scored_token_heap == [] and off._token_drop_ref is None and off.token_drop_last_frame == -1
    outputs = [
        memory.prepare_frame_sampling(32, 1024, 64, memory.default_history_feats_gather_fn)
        for memory in (off, on)
    ]
    for left, right in zip(*outputs, strict=True):
        assert left.dtype == right.dtype and left.tobytes() == right.tobytes()


def test_invalid_inputs_rejected_without_writes():
    with pytest.raises(ValueError, match="token_per_image"):
        FrameSampMemory(vision_enc_fn=stub_encoder, token_per_image=16, max_steps=MAX_STEPS, token_drop_stride=8)
    memory = online_memory()
    frames = np.zeros((1, 1, 60, 60, 3), np.uint8)
    with pytest.raises(ValueError, match="整除"):
        memory.add_buffer(frames, np.zeros((1, 8), np.float32), [0])
    assert memory.n_steps == 0 and memory.scored_token_heap == []


def make_policy(monkeypatch, config):
    monkeypatch.setattr(policy_module.nnx_utils, "module_jit", lambda fn: fn)
    monkeypatch.setattr(policy_module, "FrameSampMemory", functools.partial(FrameSampMemory, max_steps=MAX_STEPS))
    model = SimpleNamespace(history_config=config, sample_actions=lambda *a: None, vision_encode=stub_encoder)
    return policy_module.MME_VLA_Policy(
        model, norm_stats={"state": SimpleNamespace(mean=np.ones(8), std=np.ones(8) * 2)})


def load_config(perceptual_type, *, token_per_image=64, budget=1024):
    config = OmegaConf.load(REPO / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-context.yaml")
    config.perceptual_memory.type = perceptual_type
    config.token_per_image = token_per_image
    config.budget = budget
    return config


def test_policy_token_dropping_uses_stride_and_prepare_branch(monkeypatch):
    policy = make_policy(monkeypatch, load_config("token_dropping"))
    assert policy.mem_buffer.token_drop_enabled and policy.mem_buffer.token_drop_stride == 8
    frames, states = make_frames(33)
    policy.add_buffer({"images": frames[:17], "state": states[:17]})
    policy.add_buffer({"images": frames[17:], "state": states[17:]})

    inputs = policy._prepare_history({"marker": 1})
    upstream = upstream_memory()
    feed_upstream(upstream, frames, states)
    image, position, state, mask = upstream.prepare_token_dropping(32, 1024, upstream.default_history_feats_gather_fn)
    assert inputs["marker"] == 1
    assert inputs["static_image_emb"].tobytes() == image.tobytes()
    assert inputs["static_pos_emb"].tobytes() == position.tobytes()
    assert inputs["static_mask"].tobytes() == mask.tobytes()
    assert inputs["static_state_emb"].tobytes() == policy._normalize_state(state).tobytes()
    assert "mem_order" not in inputs and "motion_emb" not in inputs

    first = policy.mem_buffer
    policy.reset()
    assert policy.mem_buffer is not first and policy.mem_buffer.scored_token_heap == []


def test_policy_frame_sampling_path_keeps_scoring_off(monkeypatch):
    policy = make_policy(monkeypatch, load_config("frame_sampling", token_per_image=16, budget=512))
    assert policy.mem_buffer.token_drop_enabled is False
    frames, states = make_frames(17)
    policy.add_buffer({"images": frames, "state": states})
    inputs = policy._prepare_history({})
    assert inputs["static_image_emb"].shape == (512, 2048)
    assert policy.mem_buffer.scored_token_heap == []


def test_policy_rejects_token_dropping_with_motion(monkeypatch):
    config = load_config("token_dropping")
    config.motion.enabled = True
    with pytest.raises(ValueError, match="token_dropping"):
        make_policy(monkeypatch, config)
