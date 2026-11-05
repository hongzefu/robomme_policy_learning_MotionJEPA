"""询问批量、完整输入、离线回放及专用 RPC 的 CPU 核心测试。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from scripts.evaluation.runtime_contract import RpcSession, payload_sha256
from scripts.interpret import offline_query as OQ
from scripts.interpret import serve_policy_attn as SA


def observation():
    return HistAugObservation(images={"base": jnp.ones((1, 4, 4, 3), jnp.float32)},
        image_masks={"base": jnp.ones((1,), bool)}, state=jnp.arange(32, dtype=jnp.float32)[None],
        tokenized_prompt=jnp.ones((1, 4), jnp.int32), tokenized_prompt_mask=jnp.ones((1, 4), bool),
        static_image_emb=jnp.ones((1, 4, 6), jnp.bfloat16), static_mask=jnp.ones((1, 4), bool),
        static_pos_emb=jnp.ones((1, 4, 6), jnp.float32), static_state_emb=jnp.ones((1, 4, 8), jnp.float32),
        motion_emb=jnp.ones((1, 3, 6), jnp.bfloat16), motion_pos=jnp.ones((1, 3, 6), jnp.float32),
        motion_mask=jnp.asarray([[True, False, False]]), mem_order=jnp.arange(7, dtype=jnp.int32)[None])


class Policy:
    def __init__(self):
        self._seed = 42
        self.config = SimpleNamespace(budget=4, token_per_image=1, num_views=1)
        self._model = SimpleNamespace(action_horizon=20, action_dim=32)
        self._sample_kwargs = {"num_steps": 10}
        self.metadata = {"model_kind": "测试模型"}
        self.sample_calls, self.transform_calls, self.builds = [], 0, 0
        self.observation = observation()
        self.reset()

    def reset(self):
        self._rng = jax.random.key(42)
        self.step_idx, self.exec_start_idx = 10, 0
        self.mem_buffer = SimpleNamespace(get_frame_sampling_indices=lambda *args: [0, 4, 8])

    def _sample_actions(self, rng, obs, *, noise=None, num_steps=10):
        self.sample_calls.append((obs, noise))
        n = obs.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (n, 20, 32))
        for leaf in jax.tree.leaves(obs):
            assert leaf.shape[0] == n
        return noise + obs.state[:, None, :]

    def _output_transform(self, outputs):
        assert outputs["actions"].shape == (20, 32)
        assert outputs["state"].shape == (32,)
        self.transform_calls += 1
        return {"actions": outputs["actions"][:, :8] + outputs["state"][:8]}

    def infer(self, obs):
        self.builds += 1
        self._rng, key = jax.random.split(self._rng)
        values = self._sample_actions(key, self.observation, **self._sample_kwargs)
        outputs = self._output_transform({"state": np.asarray(self.observation.state[0]), "actions": np.asarray(values[0])})
        outputs["infer_time_ms"] = 1.0
        return outputs


def test_recursive_roundtrip_and_batch(tmp_path):
    obs = observation()
    path = tmp_path / "obs.npz"
    np.savez(path, **OQ.pack_observation(obs))
    with np.load(path, allow_pickle=False) as archive:
        restored = OQ.unpack_observation(archive, HistAugObservation)
    assert restored.token_ar_mask is None
    assert restored.static_image_emb.dtype == ml_dtypes.bfloat16
    for left, right in zip(jax.tree.leaves(obs), jax.tree.leaves(restored), strict=True):
        assert left.shape == right.shape and left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    batched = OQ.repeat_observation(restored, 10)
    for original, batch in zip(jax.tree.leaves(obs), jax.tree.leaves(batched), strict=True):
        assert batch.shape[0] == 10 and batch.dtype == original.dtype
        np.testing.assert_array_equal(batch, np.repeat(original, 10, axis=0))


def test_noise_identity_and_prediction_prefix():
    left = OQ.query_noise(42, "StopCube:16", 2, 10, 20, 32)
    np.testing.assert_array_equal(left, OQ.query_noise(42, "StopCube:16", 2, 10, 20, 32))
    np.testing.assert_array_equal(left[:2], OQ.query_noise(42, "StopCube:16", 2, 2, 20, 32))
    assert not np.array_equal(left, OQ.query_noise(42, "StopCube:16", 3, 10, 20, 32))
    assert not np.array_equal(left, OQ.query_noise(42, "StopCube:15", 2, 10, 20, 32))


@pytest.mark.parametrize("batch", [2, 10])
def test_record_original_path_and_offline_replay(tmp_path, monkeypatch, batch):
    policy = Policy()
    recorder = SA.QueryRecorder(policy, tmp_path, 10, model_id="motion", batch_size=batch)
    recorder.install()
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "same", "episode_id": 16})
    policy.reset()
    rng = np.asarray(jax.random.key_data(policy._rng)).copy()
    context = {"mode": "oracle_query", "episode_key": "StopCube:16", "env_step": 1,
               "pre_reach": True, "raw_state": np.arange(8, dtype=np.float32)}
    outputs = policy.infer({"observation/state": context["raw_state"], "oracle_context": context})
    assert policy.builds == 1 and len(policy.sample_calls) == 10 // batch
    assert policy.transform_calls == 11
    np.testing.assert_array_equal(rng, jax.random.key_data(policy._rng))
    assert outputs["actions"].shape == (20, 8) and outputs["query_actions"].shape == (10, 20, 8)
    np.testing.assert_array_equal(outputs["actions"], outputs["query_actions"][0])
    path = outputs["query_record"]
    with np.load(path, allow_pickle=False) as archive:
        assert archive["raw_state"].dtype == np.float32
        assert archive["actions_normalized"].shape == (10, 20, 32)
        assert archive["noise"].shape == (10, 20, 32)
        assert str(archive["episode_key"]) == "StopCube:16"
        assert int(archive["query_seq"]) == 0 and bool(archive["pre_reach"])
    replay = OQ.replay_query(policy, path)
    assert replay["status"] == "PASS" and replay["same_path_maxabs"] == 0
    assert json.loads((tmp_path / "same" / "episode.json").read_text())["queries"] == 1
    assert not list(tmp_path.rglob("infer_*.npz"))


def test_calibration_uses_original_infer_without_query(tmp_path, monkeypatch):
    policy = Policy()
    recorder = SA.QueryRecorder(policy, tmp_path, 10)
    recorder.install()
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "cal"})
    policy.reset()
    rng = np.asarray(jax.random.key_data(policy._rng)).copy()
    outputs = policy.infer({"oracle_context": {"mode": "oracle_calibrate"}})
    assert outputs["actions"].shape == (20, 8) and "query_actions" not in outputs
    assert not np.array_equal(rng, jax.random.key_data(policy._rng))
    assert not list(tmp_path.rglob("*.npz"))


def test_nonfinite_observation_is_rejected():
    with pytest.raises(ValueError, match="非有限"):
        OQ.pack_observation({"state": np.asarray([[np.nan]], np.float32)})


def test_offline_rejects_changed_prediction_but_reports_attention_difference(tmp_path, monkeypatch):
    policy = Policy()
    recorder = SA.QueryRecorder(policy, tmp_path, 2)
    recorder.install()
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "offline"})
    policy.reset()
    outputs = policy.infer({"oracle_context": {"mode": "oracle_query", "episode_key": "StopCube:0",
        "env_step": 1, "pre_reach": False, "raw_state": np.zeros(8, np.float32)}})
    adapter = SimpleNamespace(run=lambda rng, obs, noise, **kwargs: (noise + obs.state[:, None] + 0.1,))
    result = OQ.replay_query(policy, outputs["query_record"], adapter=adapter)
    assert result["status"] == "PASS" and result["attention"] == "ok"
    assert result["attn_path_maxabs"] > 0.09
    with np.load(outputs["query_record"], allow_pickle=False) as archive:
        record = dict(archive)
    record["actions"][0, 0, 0] += 0.01
    damaged = tmp_path / "changed.npz"
    np.savez(damaged, **record)
    assert OQ.replay_query(policy, damaged)["status"] == "FAIL"
    record["actions"][0, 0, 0] = np.nan
    np.savez(damaged, **record)
    assert OQ.replay_query(policy, damaged)["status"] == "FAIL"


def reset_payload():
    return {"reset": True, "oracle_context": {"mode": "oracle_query", "episode_key": "StopCube:16"},
            "eval_rpc": {"episode_nonce": "nonce", "expected_run_identity_sha256": "run", "expected_server_nonce": "server"}}


def history_payload(session, count=4, boundary=0):
    payload = {"add_buffer": True, "images": np.zeros((count, 1), np.uint8),
               "state": np.zeros((count, 8), np.float32), "exec_start_idx": boundary,
               "oracle_context": {"mode": "oracle_query", "episode_key": "StopCube:16"}}
    payload["eval_rpc"] = {"episode_nonce": "nonce", "batch_seq": session.batch_seq,
        "first_global_frame": session.next_frame, "frame_count": count, "payload_sha256": payload_sha256(payload)}
    return payload


def initialize(cls):
    session = cls({"run_identity_sha256": "run", "server_nonce": "server"})
    payload = reset_payload()
    session.commit(session.validate(payload), payload["eval_rpc"])
    return session


def test_rpc_relaxes_only_cadence():
    session = initialize(SA.query_session_class("oracle_query"))
    for count, boundary in [(3, 2), (4, 0), (4, 0), (16, 0)]:
        payload = history_payload(session, count, boundary)
        session.commit(session.validate(payload), payload["eval_rpc"])
    assert session.next_frame == 27
    payload = history_payload(session)
    payload["eval_rpc"]["first_global_frame"] -= 1
    with pytest.raises(ValueError, match="历史顺序"):
        session.validate(payload)
    payload = history_payload(session)
    payload["state"][0, 0] = 1
    with pytest.raises(ValueError, match="payload"):
        session.validate(payload)
    payload = history_payload(session)
    payload["oracle_context"]["episode_key"] = "StopCube:15"
    with pytest.raises(ValueError, match="身份"):
        session.validate(payload)
    normal = initialize(RpcSession)
    payload = history_payload(normal, 1)
    normal.commit(normal.validate(payload), payload["eval_rpc"])
    with pytest.raises(ValueError, match="续包帧数"):
        normal.validate(history_payload(normal))


def test_rpc_first_packet_and_infer_order_are_strict():
    session = initialize(SA.query_session_class())
    with pytest.raises(ValueError, match="首包"):
        session.validate(history_payload(session, 4))
    payload = history_payload(session, 1)
    session.commit(session.validate(payload), payload["eval_rpc"])
    inference = {"oracle_context": payload["oracle_context"], "eval_rpc": {"episode_nonce": "nonce",
        "replan_seq": 0, "last_global_frame": 0}}
    session.commit(session.validate(inference), inference["eval_rpc"])
    with pytest.raises(ValueError, match="infer序号"):
        session.validate(inference)
    release = {"release_attempt": True, "eval_rpc": {"episode_nonce": "nonce"}}
    assert session.validate(release) == "release"
