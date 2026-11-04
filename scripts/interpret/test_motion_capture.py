"""实际运动输入与全部编码调用轨迹的 CPU 定向验证。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.interpret import serve_policy_attn as SA


class Memory:
    def __init__(self, dtype, valid):
        self.emb = np.arange(18, dtype=dtype).reshape(3, 6)
        self.pos = np.arange(12, dtype=dtype).reshape(3, 4) / 8
        self.mask = np.asarray(valid, bool)
        self.times = np.asarray([0, 16, 32], np.int64)
        self.calls = 0

    def _prepare_motion(self, step_idx):
        self.calls += 1
        return self.emb, self.pos, self.mask, self.times

    def get_frame_sampling_indices(self, step_idx, budget, token_per_image):
        return [0, 4]


class Policy:
    """组包后变换内容和有效位，以区分缓存返回值与实际输入。"""

    def __init__(self, dtype, valid, motion=True):
        self.dtype, self.valid, self.motion_enabled = dtype, valid, motion
        self.config = SimpleNamespace(budget=4, token_per_image=1, num_views=1,
                                      motion=SimpleNamespace(budget=3))
        self.metadata = {"model_kind": "stage2_pool4_concat" if motion else "history"}
        self.reset()

    def reset(self):
        self.mem_buffer = Memory(self.dtype, self.valid)
        self.step_idx, self.exec_start_idx = 40, 7

    def add_buffer(self, obs):
        pass

    def infer(self, obs):
        fields = dict(static_mask=np.asarray([[True, True, False, False]]),
                      tokenized_prompt_mask=np.asarray([[True, False]]))
        if self.motion_enabled:
            emb, pos, mask, _ = self.mem_buffer._prepare_motion(self.step_idx)
            actual_mask = mask.copy()
            actual_mask[1] = False
            fields.update(motion_emb=(emb + 2)[None], motion_pos=(pos + 3)[None],
                          motion_mask=actual_mask[None],
                          mem_order=np.asarray([[4, 0, 6, 1, 5, 2, 3]], np.int32))
        self.observation = SimpleNamespace(**fields)
        return {"actions": self._sample_actions(None, self.observation), "infer_time_ms": 1.25}


class Adapter:
    def __init__(self, motion=True):
        self.MEM = 7 if motion else 4
        self.P = self.MEM + 2
        self.actions = np.arange(160, dtype=np.float64).reshape(20, 8)

    def make_sample_fn(self):
        def sample(rng, observation, **kwargs):
            self.received = observation
            self.before = {key: value.copy() for key, value in vars(observation).items()}
            self.last_attn = {
                "P": self.P, "MEM": self.MEM,
                "probs_action": np.zeros((10, 18, 8, 20, self.P + 20), np.float16),
                "probs_text": np.zeros((18, 8, 64, self.P), np.float16),
            }
            return self.actions
        return sample


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("valid", [[False, False, False], [True, True, True]])
def test_actual_input_complete_and_actions_unchanged(tmp_path, monkeypatch, dtype, valid):
    policy, adapter = Policy(dtype, valid), Adapter()
    obs = {"observation/state": np.arange(8, dtype=np.float64), "prompt": "按按钮"}
    state_before = obs["observation/state"].copy()
    policy._sample_actions = adapter.make_sample_fn()
    reference = policy.infer(obs)
    expected_observation = {key: value.copy() for key, value in vars(policy.observation).items()}

    recorder = SA.AttnRecorder(policy, adapter, tmp_path, save_top_frames=0)
    recorder.install()
    monkeypatch.setitem(SA._RPC, "op", "reset")
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "actual", "episode_id": 8})
    policy.reset()
    for seq in range(2):
        result = policy.infer(obs)
        assert result["actions"] is reference["actions"] is adapter.actions
        assert result["actions"].dtype == np.float64 and result["infer_time_ms"] == 1.25
        assert adapter.received is policy.observation
        for key, before in adapter.before.items():
            np.testing.assert_array_equal(getattr(policy.observation, key), before)
            np.testing.assert_array_equal(before, expected_observation[key])
        np.testing.assert_array_equal(obs["observation/state"], state_before)
        assert policy.mem_buffer.calls == seq + 1
        with np.load(tmp_path / "actual" / f"infer_{seq:04d}.npz") as record:
            assert set(record.files) == set(SA.NPZ_KEYS + SA.ABLATE_NPZ_KEYS + SA.MOTION_INPUT_NPZ_KEYS)
            for key, shape in (("motion_emb", (3, 6)), ("motion_pos", (3, 4))):
                assert record[key].shape == shape and record[key].dtype == dtype
                np.testing.assert_array_equal(record[key], expected_observation[key][0])
            mask = expected_observation["motion_mask"][0]
            np.testing.assert_array_equal(record["motion_mask"], mask)
            np.testing.assert_array_equal(record["motion_times"], np.where(mask, [0, 16, 32], -1))
            order = record["mem_order"]
            np.testing.assert_array_equal(record["mem_mask"], np.r_[True, True, False, False, mask][order])
            for slot in np.flatnonzero(mask):
                key = np.flatnonzero(order == 4 + slot)
                assert key.size == 1 and record["mem_mask"][key[0]]
            assert int(record["infer_seq"]) == seq and int(record["step_idx"]) == 40
            assert int(record["exec_start_idx"]) == 7
            np.testing.assert_array_equal(record["actions"], adapter.actions.astype(np.float32))
    info = json.loads((tmp_path / "actual" / "episode.json").read_text())
    assert info["reset_rpc"]["episode_id"] == 8 and info["infers"] == 2


def test_no_motion_keeps_empty_legacy_shapes(tmp_path, monkeypatch):
    policy, adapter = Policy(np.float32, [False] * 3, motion=False), Adapter(motion=False)
    recorder = SA.AttnRecorder(policy, adapter, tmp_path, save_top_frames=0)
    recorder.install()
    monkeypatch.setitem(SA._RPC, "op", "reset")
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "without"})
    policy.reset()
    policy.infer({"observation/state": np.zeros(8)})
    with np.load(tmp_path / "without" / "infer_0000.npz") as record:
        assert set(record.files) == set(SA.NPZ_KEYS + SA.ABLATE_NPZ_KEYS)
        assert record["motion_mask"].shape == record["motion_times"].shape == (0,)
        assert record["motion_mask"].dtype == bool and record["motion_times"].dtype == np.int32
    assert policy.mem_buffer.calls == 0


def test_all_encoder_calls_survive_repeated_starts(tmp_path):
    class Client:
        def __init__(self):
            self.calls = []
            self.buffer = np.empty(6, np.float32)

        def __call__(self, frames, start):
            self.calls.append(start)
            self.buffer[:] = len(self.calls) + np.arange(6)
            return self.buffer

    inner = Client()
    client = SA.MotionContentClient(inner, record_dir=tmp_path)
    client.begin_episode("calls")
    frames = np.zeros((33, 1), np.uint8)
    expected = []
    for start in [16, 0, 16, 16]:
        returned = client(frames, start)
        assert returned is inner.buffer
        expected.append(returned.copy())
        client.flush()
    with np.load(tmp_path / "calls.npz") as record:
        assert record["start_frames"].tolist() == [16, 0]
        assert record["counts"].tolist() == [3, 1]
        np.testing.assert_array_equal(record["embs"], expected[:2])
        assert record["call_start_frames"].tolist() == inner.calls == [16, 0, 16, 16]
        assert record["call_start_frames"].dtype == np.int32
        assert record["call_seq"].dtype == np.int64 and record["call_seq"].tolist() == [0, 1, 2, 3]
        assert record["call_embs"].dtype == np.float32 and record["call_embs"].shape == (4, 6)
        np.testing.assert_array_equal(record["call_embs"], expected)
    assert not frames.any() and client.replaced == 0 and client.flush() is None
    client.begin_episode("next")
    client(frames, 32)
    client.flush()
    with np.load(tmp_path / "next.npz") as record:
        assert record["call_start_frames"].tolist() == [32] and record["call_seq"].tolist() == [0]


def test_legacy_donor_without_call_history_remains_readable(tmp_path):
    emb = np.arange(12, dtype=np.float32).reshape(2, 6)
    np.savez(tmp_path / "ButtonUnmaskSwap_ep8.npz", start_frames=np.asarray([0, 16], np.int32),
             embs=emb, counts=np.ones(2, np.int32), mode=np.str_("none"))
    donor = SA.load_donor(tmp_path, "ButtonUnmaskSwap:8")
    calls = []

    def inner(frames, start):
        calls.append(start)
        return np.full(6, -1, np.float32)

    client = SA.MotionContentClient(inner, mode="swap", donor=donor)
    result = client(np.zeros((33, 1), np.uint8), 16)
    np.testing.assert_array_equal(result, emb[1])
    assert calls == [16] and client.replace_log == [[16, 16]]
