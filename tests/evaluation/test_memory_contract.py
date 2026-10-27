"""阶段一 E3 的处理器合同：独立 token、位置行、容量与排序真值。"""

import numpy as np
import pytest

from frame_fixture import DEMO_LENGTHS, EXEC_QUERIES, SENTINEL, expected_order, feed_packet, feed_scheduler, make_memory, oracle_windows
from mme_vla_suite.shared.sampling import even_sampling_indices, memory_order, pad_times


@pytest.mark.parametrize("demo_length", DEMO_LENGTHS)
def test_motion_assembly_shape_dtype_mask_positions_and_times(demo_length):
    memory = make_memory()
    feed_scheduler(memory, list(range(demo_length + 1)), demo_length)
    previous = 0
    for query in EXEC_QUERIES:
        feed_scheduler(memory, list(range(demo_length + previous + 1, demo_length + query + 1)), demo_length)
        expected = oracle_windows(demo_length, query)
        count = len(expected)
        if count > 96:
            with pytest.raises(RuntimeError, match="budget"):
                memory._prepare_motion(demo_length + query)
        else:
            emb, pos, mask, times = memory._prepare_motion(demo_length + query)
            assert emb.shape == (96, 1024) and emb.dtype == np.float32
            assert pos.shape == (96, 256) and pos.dtype == np.float32
            assert mask.shape == (96,) and mask.dtype == np.bool_
            assert times.shape == (96,) and times.dtype == np.int64
            starts = [window.start for window in expected]
            expected_emb = np.zeros((96, 1024), np.float32)
            expected_pos = np.zeros((96, 256), np.float32)
            for index, start in enumerate(starts):
                expected_emb[index] = start + np.arange(1024, dtype=np.float32) / 2048
                expected_pos[index] = start * 1024 + np.arange(256, dtype=np.float32) / 256
            np.testing.assert_array_equal(emb, expected_emb)
            np.testing.assert_array_equal(pos, expected_pos)
            np.testing.assert_array_equal(mask, np.arange(96) < count)
            np.testing.assert_array_equal(times, starts + [SENTINEL] * (96 - count))
        previous = query


@pytest.mark.parametrize("step", [0, 1, 7, 8, 31, 32, 33, 63, 64, 65, 1301, 3110, 4095])
def test_visual_sampling_and_608_slot_stable_permutation(step):
    # 用有理数向下取整定义八帧均匀采样，独立于 numpy.linspace。
    expected_frames = list(range(step + 1)) if step < 8 else [slot * step // 7 for slot in range(8)]
    actual_frames = even_sampling_indices(step, 8)
    assert actual_frames == expected_frames
    frame_times = expected_frames + [SENTINEL] * (8 - len(expected_frames))
    for demo_length in (0, 17, 50):
        starts = [window.start for window in oracle_windows(demo_length, step)]
        if len(starts) > 96:
            continue
        motion_times = starts + [SENTINEL] * (96 - len(starts))
        order = memory_order(frame_times, 64, motion_times)
        assert order.shape == (608,) and order.dtype == np.int32
        np.testing.assert_array_equal(order, expected_order(frame_times, motion_times))
        assert sorted(order.tolist()) == list(range(608))


def test_duplicate_timestamps_preserve_visual_patch_and_motion_slot_order():
    frame_times = [32, 0, 32, SENTINEL, 0, SENTINEL, 64, 64]
    motion_times = [32, 0, 32, 64, 0] + [SENTINEL] * 91
    actual = memory_order(frame_times, 64, motion_times)
    np.testing.assert_array_equal(actual, expected_order(frame_times, motion_times))
    for index, time in enumerate(frame_times):
        positions = [int(np.flatnonzero(actual == (index * 64 + patch))[0]) for patch in range(64)]
        assert positions == list(range(positions[0], positions[0] + 64))


def test_pad_times_dtype_sentinel_and_rejection():
    actual = pad_times([0, 32, 50], 96)
    np.testing.assert_array_equal(actual, [0, 32, 50] + [SENTINEL] * 93)
    assert actual.dtype == np.int64
    with pytest.raises(ValueError):
        pad_times(list(range(97)), 96)
    with pytest.raises(ValueError):
        pad_times([SENTINEL], 96)


def test_clear_discards_episode_frames_tokens_and_grid_state():
    memory = make_memory()
    feed_packet(memory, list(range(18)), 17)
    feed_packet(memory, list(range(18, 50)), 17)
    assert memory._history_feats_motion
    memory.clear()
    assert memory.n_steps == 0
    assert not memory._raw_frames and not memory._history_feats_motion
    assert memory.exec_start_idx is None
    assert memory._next_grid_start_demo == memory._next_grid_start_exec == 0
    # 诊断计时、调用累计数并非窗口缓存；核第二局不会重复使用前局 token。
    memory.motion_enc.calls.clear()
    feed_packet(memory, [0], 0)
    assert not memory._history_feats_motion
    feed_packet(memory, list(range(1, 33)), 0)
    assert memory.motion_enc.calls == [(0, tuple(range(33)))]


def test_real_ingestion_visual_padding_and_state_alignment():
    memory = make_memory()
    feed_packet(memory, [0], 0)
    feed_packet(memory, [1], 0)
    emb, pos, states, mask = memory.prepare_frame_sampling(1, 512, 64, memory.default_history_feats_gather_fn)
    assert emb.shape == (512, 2) and emb.dtype == np.dtype("bfloat16")
    assert pos.shape == (512, 256) and pos.dtype == np.float32
    assert states.shape == (512, 8) and states.dtype == np.float32
    np.testing.assert_array_equal(mask, np.arange(512) < 128)
    expected_visual = np.stack((np.arange(64) + 1, np.arange(64) + 65), axis=-1).astype(np.dtype("bfloat16"))
    np.testing.assert_array_equal(emb[:128], np.tile(expected_visual, (2, 1)))
    np.testing.assert_array_equal(states[:64], np.broadcast_to(np.arange(8, dtype=np.float32) / 16, (64, 8)))
    np.testing.assert_array_equal(states[64:128], np.broadcast_to(1 + np.arange(8, dtype=np.float32) / 16, (64, 8)))
    assert not np.any(emb[128:]) and not np.any(pos[128:]) and not np.any(states[128:])
