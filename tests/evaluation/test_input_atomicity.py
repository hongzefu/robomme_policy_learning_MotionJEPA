"""在线输入拒绝零写入及合法路径字节对拍；不运行真实神经编码器。"""

from pathlib import Path
import subprocess
import types

import numpy as np
import pytest

from frame_fixture import feed_packet, make_memory, memory_snapshot, packet
from mme_vla_suite.policies.policy import MME_VLA_Policy


BASE = "2789ae42f3166788ec7d5efc4baf29d8d5a1bc4f"


def bare_policy(memory):
    policy = MME_VLA_Policy.__new__(MME_VLA_Policy)
    policy.mem_buffer = memory
    policy.motion_enabled = True
    policy.exec_start_idx = 0
    policy.step_idx = -1
    return policy


@pytest.mark.parametrize("fault", [
    "partial_demo", "missing_exec0", "image_dtype", "image_shape", "state_dtype",
    "state_shape", "negative_boundary", "boundary_drift", "continuation_state", "overlong_first",
    "boolean_boundary", "numpy_boolean_boundary", "positive_fractional_boundary", "negative_fractional_boundary",
])
def test_policy_rejection_preserves_boundary_step_and_memory(fault):
    memory = make_memory()
    policy = bare_policy(memory)
    ids, boundary = list(range(18)), 17
    if fault in ("boundary_drift", "continuation_state"):
        images, states = packet(ids)
        policy.add_buffer(dict(images=images, state=states, exec_start_idx=17))
        ids, boundary = [18], 18 if fault == "boundary_drift" else 0
    elif fault == "partial_demo":
        ids = list(range(10))
    elif fault == "missing_exec0":
        ids = list(range(17))
    elif fault == "negative_boundary":
        boundary = -1
    elif fault == "overlong_first":
        ids = list(range(19))
    elif fault in ("boolean_boundary", "numpy_boolean_boundary", "positive_fractional_boundary"):
        # 帧数故意与错误强转后的 D=1 匹配，避免首包长度检查掩盖类型漏洞。
        ids = [0, 1]
        boundary = {"boolean_boundary": True, "numpy_boolean_boundary": np.bool_(True),
                    "positive_fractional_boundary": 1.5}[fault]
    elif fault == "negative_fractional_boundary":
        ids, boundary = [0], -0.5
    images, states = packet(ids)
    if fault == "image_dtype":
        images = images.astype(np.float32)
    elif fault == "image_shape":
        images = images[:, 0]
    elif fault == "state_dtype":
        states = states.astype(np.float64)
    elif fault in ("state_shape", "continuation_state"):
        states = states[:, :7]
    before = (policy.exec_start_idx, policy.step_idx, memory_snapshot(memory))
    with pytest.raises(ValueError):
        policy.add_buffer(dict(images=images, state=states, exec_start_idx=boundary))
    assert (policy.exec_start_idx, policy.step_idx, memory_snapshot(memory)) == before
    # 同一实例能接受后续合法包，不需要恢复或清空。
    ids = [18] if policy.step_idx >= 0 else list(range(18))
    images, states = packet(ids)
    policy.add_buffer(dict(images=images, state=states, exec_start_idx=17))
    assert policy.exec_start_idx == 17 and policy.step_idx == ids[-1]


def test_policy_accepts_numpy_integer_boundary_and_zero_continuation_marker():
    policy = bare_policy(make_memory())
    images, states = packet(list(range(18)))
    policy.add_buffer(dict(images=images, state=states, exec_start_idx=np.int64(17)))
    images, states = packet([18])
    policy.add_buffer(dict(images=images, state=states, exec_start_idx=np.int32(0)))
    assert policy.exec_start_idx == 17 and policy.step_idx == 18


def test_memory_rejects_structural_errors_before_encoder_or_state_write():
    for fault in ("empty", "length", "boolean_index", "fractional_index", "negative_boundary",
                  "boolean_boundary", "fractional_boundary", "overflow_boundary", "missing_boundary", "overlong_first"):
        memory = make_memory()
        images, states = packet([0])
        ids, boundary = [0], 0
        if fault == "empty":
            images, states, ids = images[:0], states[:0], []
        elif fault == "length":
            ids = [0, 1]
        elif fault == "boolean_index":
            ids = [False]
        elif fault == "fractional_index":
            ids = [0.0]
        elif fault == "negative_boundary":
            boundary = -1
        elif fault == "boolean_boundary":
            boundary = False
        elif fault == "fractional_boundary":
            boundary = 0.0
        elif fault == "overflow_boundary":
            boundary = 4096
        elif fault == "missing_boundary":
            boundary = None
        elif fault == "overlong_first":
            images, states = packet([0, 1])
            ids = [0, 1]
        before = memory_snapshot(memory)
        memory.vision_enc = lambda _: pytest.fail("非法输入不应调用视觉编码器")
        with pytest.raises(ValueError):
            memory.add_buffer(images, states, ids, exec_start_idx=boundary)
        assert memory_snapshot(memory) == before, fault


@pytest.fixture(scope="module")
def baseline_memory_class():
    root = Path(__file__).resolve().parents[2]
    path = "src/mme_vla_suite/policies/framesamp_memory.py"
    source = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=root, text=True)
    reference = types.ModuleType("framesamp_memory_fixed_baseline")
    exec(compile(source, f"{BASE}:{path}", "exec"), reference.__dict__)
    return reference.FrameSampMemory


@pytest.mark.parametrize("demo_length", [0, 17, 50])
def test_legal_first_and_sixteen_frame_packets_match_base_bytes(baseline_memory_class, demo_length):
    reference, candidate = make_memory(), make_memory()
    reference.__class__ = baseline_memory_class
    for ids in [list(range(demo_length + 1))] + [
        list(range(demo_length + offset + 1, demo_length + offset + 17))
        for offset in range(0, 64, 16)
    ]:
        for memory in (reference, candidate):
            feed_packet(memory, ids, demo_length)
        # 快照逐键包含 dtype、shape、原始字节摘要、窗口帧号和全部缓存状态。
        assert memory_snapshot(reference) == memory_snapshot(candidate)
        for left, right in zip(reference._prepare_motion(ids[-1]), candidate._prepare_motion(ids[-1]), strict=True):
            assert left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
        for left, right in zip(
            reference.prepare_frame_sampling(ids[-1], 512, 64, reference.default_history_feats_gather_fn),
            candidate.prepare_frame_sampling(ids[-1], 512, 64, candidate.default_history_feats_gather_fn), strict=True,
        ):
            assert left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
