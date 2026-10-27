"""阶段一 E2：独立帧号真值、真实首包及执行续包；不执行仿真或神经编码器。"""

import numpy as np
import pytest

from frame_fixture import DEMO_LENGTHS, EXEC_QUERIES, feed_packet, feed_scheduler, make_memory, memory_snapshot, oracle_windows
from mme_vla_suite.datastore import motion_store


@pytest.mark.parametrize("demo_length,count,last_pad", [
    (0, 0, None), (1, 0, None), (16, 0, None), (17, 1, 16),
    (32, 1, 1), (33, 1, 0), (48, 1, 0), (49, 2, 16),
    (50, 2, 15), (64, 2, 1), (65, 2, 0), (1808, 56, 0), (1809, 57, 16),
])
def test_oracle_agrees_with_hand_counted_boundary_examples(demo_length, count, last_pad):
    demo = oracle_windows(demo_length, 0)
    assert len(demo) == count
    if demo:
        assert demo[-1].pad == last_pad
        assert demo[-1].real + demo[-1].pad == 33
    for query, exec_count in zip(EXEC_QUERIES, (0, 1, 1, 1, 2, 2, 40), strict=True):
        assert len(oracle_windows(demo_length, query)) == count + exec_count


@pytest.mark.parametrize("demo_length", DEMO_LENGTHS)
def test_online_scheduler_matches_independent_frame_oracle(demo_length):
    memory = make_memory()
    feed_scheduler(memory, list(range(demo_length + 1)), demo_length)
    previous = 0
    for query in EXEC_QUERIES:
        feed_scheduler(memory, list(range(demo_length + previous + 1, demo_length + query + 1)), demo_length)
        expected = oracle_windows(demo_length, query)
        assert memory.motion_enc.calls == [(window.start, window.frame_ids) for window in expected]
        assert memory.visible_motion_frames(demo_length + query) == [window.start for window in expected]
        assert memory.motion_encode_calls == len(expected)
        previous = query


@pytest.mark.parametrize("demo_length", DEMO_LENGTHS)
@pytest.mark.parametrize("query", EXEC_QUERIES)
def test_offline_visible_rows_match_independent_oracle(demo_length, query):
    manifest = {"episodes": [dict(global_episode_idx=0, h5_file="独立夹具.h5", raw_ep_idx=0,
                                  num_timesteps=demo_length + 1302, exec_start_idx=demo_length)]}
    entry = motion_store.build_index_entries(manifest, motion_store.LAYOUT_SPECS[motion_store.STAGE2_LAYOUT])[0]
    rows, frames = motion_store.visible_motion_rows(entry, demo_length + query)
    expected = oracle_windows(demo_length, query)
    np.testing.assert_array_equal(frames, [window.start for window in expected])
    np.testing.assert_array_equal(rows, np.arange(len(expected), dtype=np.int64))
    assert frames.dtype == rows.dtype == np.int64


@pytest.mark.parametrize("demo_length", [0, 17, 50])
def test_complete_first_packet_and_real_sixteen_frame_schedule(demo_length):
    memory = make_memory()
    feed_packet(memory, list(range(demo_length + 1)), demo_length)
    for query in (0, 16, 32, 48, 64):
        if query:
            feed_packet(memory, list(range(demo_length + query - 15, demo_length + query + 1)), demo_length)
        assert memory.motion_enc.calls == [(window.start, window.frame_ids) for window in oracle_windows(demo_length, query)]
        assert memory.n_steps == demo_length + query + 1
    assert len([start for start, _ in memory.motion_enc.calls if start >= demo_length]) == 2


@pytest.mark.parametrize("chunks", [(65,), (1,) * 65, (7, 9, 15, 1, 1, 30, 2)])
def test_execution_packet_partition_does_not_change_windows(chunks):
    memory = make_memory()
    demo_length = 17
    feed_packet(memory, list(range(demo_length + 1)), demo_length)
    query = 0
    for count in chunks:
        feed_packet(memory, list(range(demo_length + query + 1, demo_length + query + count + 1)), demo_length)
        query += count
        assert memory.motion_enc.calls == [(window.start, window.frame_ids) for window in oracle_windows(demo_length, query)]
    assert query == 65


@pytest.mark.parametrize("fault", ["partial_demo", "missing_demo", "negative_index", "duplicate_in_packet", "unordered_first", "missing_exec0", "position_overflow", "wrong_dtype", "wrong_size"])
def test_invalid_first_packet_is_rejected_without_partial_state(fault):
    memory = make_memory()
    before = memory_snapshot(memory)
    demo_length, ids, kwargs = 17, list(range(18)), {}
    if fault == "partial_demo":
        ids = list(range(10))
    elif fault == "missing_demo":
        ids.remove(7)
    elif fault == "negative_index":
        demo_length, ids = 0, [-1, 0]
    elif fault == "duplicate_in_packet":
        demo_length, ids = 0, [0, 0]
    elif fault == "unordered_first":
        demo_length, ids = 0, [1, 0]
    elif fault == "missing_exec0":
        ids = list(range(17))
    elif fault == "position_overflow":
        demo_length, ids = 0, [0, 4096]
    elif fault == "wrong_dtype":
        kwargs = {"dtype": np.float32}
    elif fault == "wrong_size":
        kwargs = {"frame_size": 224}
    with pytest.raises((ValueError, RuntimeError)):
        feed_packet(memory, ids, demo_length, **kwargs)
    assert memory_snapshot(memory) == before, f"非法首包 {fault} 污染了记忆状态"
    feed_packet(memory, list(range(18)), 17)
    assert memory.motion_enc.calls == [(window.start, window.frame_ids) for window in oracle_windows(17, 0)]


@pytest.mark.parametrize("fault", ["duplicate_existing", "missing_frame", "future_frame", "unordered_exec", "cross_segment", "boundary_drift"])
def test_invalid_execution_packet_is_rejected_without_partial_state(fault):
    memory = make_memory()
    feed_packet(memory, list(range(18)), 17)
    before = memory_snapshot(memory)
    boundary, ids = 17, [18]
    if fault == "duplicate_existing":
        ids = [17]
    elif fault == "missing_frame":
        ids = [19]
    elif fault == "future_frame":
        ids = [50]
    elif fault == "unordered_exec":
        ids = [19, 18]
    elif fault == "cross_segment":
        ids = [16, 18]
    elif fault == "boundary_drift":
        boundary = 18
    with pytest.raises((ValueError, RuntimeError)):
        feed_packet(memory, ids, boundary)
    assert memory_snapshot(memory) == before, f"非法执行包 {fault} 污染了记忆状态"
    feed_packet(memory, [18], 17)
    assert memory.n_steps == 19


def test_future_query_cannot_read_unreceived_history():
    memory = make_memory()
    feed_packet(memory, [0], 0)
    before = memory_snapshot(memory)
    with pytest.raises((ValueError, RuntimeError)):
        memory._prepare_motion(32)
    assert memory_snapshot(memory) == before
