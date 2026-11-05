"""真实 rollout 输入记录的 CPU 测试：假 policy、不加载权重、不启动服务。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.serving import websocket_policy_server
from scripts.evaluation import manifest as M
from scripts.interpret import attn_adapter
from scripts.interpret import offline_query as OQ
from scripts.interpret import run_local as local
from scripts.interpret import serve_policy_attn as SA
from scripts.training import serve_policy

SPLIT = "stage2_pool4_split_v1"


def observation(motion_mask=(True, True, False, False)):
    size = len(motion_mask)
    return HistAugObservation(images={"base": jnp.ones((1, 4, 4, 3), jnp.float32)},
        image_masks={"base": jnp.ones((1,), bool)}, state=jnp.arange(32, dtype=jnp.float32)[None],
        tokenized_prompt=jnp.ones((1, 4), jnp.int32), tokenized_prompt_mask=jnp.ones((1, 4), bool),
        static_image_emb=jnp.ones((1, 4, 6), jnp.bfloat16), static_mask=jnp.ones((1, 4), bool),
        static_pos_emb=jnp.ones((1, 4, 6), jnp.float32), static_state_emb=jnp.ones((1, 4, 8), jnp.float32),
        motion_emb=jnp.ones((1, size, 6), jnp.bfloat16), motion_pos=jnp.ones((1, size, 6), jnp.float32),
        motion_mask=jnp.asarray([motion_mask]), mem_order=jnp.arange(4 + size, dtype=jnp.int32)[None])


class Buffer:
    def __init__(self, times):
        self.times = times
        self.motion_calls = 0

    def get_frame_sampling_indices(self, step_idx, budget, token_per_image):
        return [0, 4, step_idx]

    def _prepare_motion(self, step_idx):
        self.motion_calls += 1
        return None, None, None, np.asarray(self.times)


class Policy:
    """形状和调用顺序与 policy.Policy.infer 一致：先拆 RNG，再组包，再采样，最后输出变换。"""

    def __init__(self, *, motion=False, readout=SPLIT, times=(4, 8)):
        self._seed = 42
        self.motion_enabled = motion
        self.times = times
        self.config = SimpleNamespace(budget=4, token_per_image=1, num_views=1, motion=SimpleNamespace(readout=readout))
        self._model = SimpleNamespace(action_horizon=20, action_dim=32)
        self._sample_kwargs = {"num_steps": 10}
        self.metadata = {"model_kind": "测试模型"}
        self.sample_calls, self.buffers = [], []
        self.observation = observation((True,) * 6 + (False,) * 2 if motion and readout == SPLIT else (True, False))
        self.reset()

    def reset(self):
        self._rng = jax.random.key(42)
        self.step_idx, self.exec_start_idx = 10, 0
        self.mem_buffer = Buffer(self.times)
        self.buffers.append(self.mem_buffer)

    def _sample_actions(self, rng, obs, *, noise=None, num_steps=10):
        self.sample_calls.append({"rng": rng, "noise": noise, "num_steps": num_steps})
        if noise is None:
            noise = jax.random.normal(rng, (obs.state.shape[0], 20, 32))
        return noise + obs.state[:, None, :]

    def _output_transform(self, outputs):
        return {"actions": outputs["actions"][:, :8] + outputs["state"][:8]}

    def infer(self, obs):
        if self.motion_enabled:
            self.mem_buffer._prepare_motion(self.step_idx)
        self._rng, key = jax.random.split(self._rng)
        values = self._sample_actions(key, self.observation, **self._sample_kwargs)
        outputs = self._output_transform({"state": np.asarray(self.observation.state[0]), "actions": np.asarray(values[0])})
        outputs["infer_time_ms"] = 1.0
        return outputs


def install(tmp_path, monkeypatch, *, nonce="ep", **kwargs):
    policy = Policy(**kwargs)
    SA.RolloutRecorder(policy, tmp_path, model_id="ckpt").install()
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": nonce})
    policy.reset()
    return policy


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def test_original_path_rng_and_outputs_unchanged(tmp_path, monkeypatch):
    plain = Policy()
    policy = install(tmp_path, monkeypatch)
    for _ in range(3):
        expected, actual = plain.infer({}), policy.infer({})
        np.testing.assert_array_equal(expected["actions"], actual["actions"])
    np.testing.assert_array_equal(jax.random.key_data(plain._rng), jax.random.key_data(policy._rng))
    assert len(policy.sample_calls) == 3
    for call in policy.sample_calls:
        assert call["noise"] is None and call["num_steps"] == 10


def test_noise_is_the_noise_the_sampler_drew(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch)
    outputs = policy.infer({})
    record = load(tmp_path / "ep" / "rollout_0000.npz")
    key = policy.sample_calls[0]["rng"]
    np.testing.assert_array_equal(record["sample_rng"], jax.random.key_data(key))
    np.testing.assert_array_equal(record["noise"], jax.random.normal(key, (1, 20, 32)))
    np.testing.assert_array_equal(record["actions_normalized"], record["noise"] + np.arange(32, dtype=np.float32))
    np.testing.assert_array_equal(record["actions"][0], outputs["actions"])


def test_record_keys_shapes_and_files(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch)
    policy.infer({})
    policy.step_idx = 14
    policy.infer({})
    directory = tmp_path / "ep"
    assert sorted(path.name for path in directory.iterdir()) == ["episode.json", "rollout_0000.npz", "rollout_0001.npz"]
    assert json.loads((directory / "episode.json").read_text())["infers"] == 2
    record = load(directory / "rollout_0001.npz")
    expected = set(OQ.pack_observation(policy.observation)) | {"noise", "actions_normalized", "actions", "sample_rng",
        "sample_batch_size", "sample_kwargs", "step_idx", "infer_seq", "frame_times", "motion_times", "model_id"}
    assert set(record) == expected
    assert not {"episode_key", "env_step", "raw_state"} & set(record)
    assert not [key for key in record if key.startswith(("probs_", "infer_")) and key != "infer_seq"]
    assert record["noise"].shape == record["actions_normalized"].shape == (1, 20, 32)
    assert record["actions"].shape == (1, 20, 8) and record["actions"].dtype == np.float32
    assert record["sample_rng"].dtype == np.uint32 and record["sample_rng"].shape == (2,)
    assert int(record["sample_batch_size"]) == 1 and record["sample_batch_size"].dtype == np.int64
    assert json.loads(str(record["sample_kwargs"])) == {"num_steps": 10}
    assert int(record["step_idx"]) == 14 and int(record["infer_seq"]) == 1
    assert record["frame_times"].tolist() == [0, 4, 14] and record["frame_times"].dtype == np.int64
    assert record["motion_times"].shape == (0,) and record["motion_times"].dtype == np.int32
    assert str(record["model_id"]) == "ckpt"


def test_replay_query_consumes_record_as_is(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch)
    policy.infer({})
    path = tmp_path / "ep" / "rollout_0000.npz"
    with np.load(path, allow_pickle=False) as archive:
        restored = OQ.unpack_observation(archive, HistAugObservation)
    assert restored.static_image_emb.dtype == ml_dtypes.bfloat16
    fresh = Policy()
    result = OQ.replay_query(fresh, path)
    assert result["status"] == "PASS" and result["same_path_maxabs"] == 0 and result["predictions"] == 1
    assert fresh.sample_calls[0]["noise"] is not None and fresh.sample_calls[0]["num_steps"] == 10
    damaged = load(path)
    damaged["noise"] = damaged["noise"] + 0.01
    changed = tmp_path / "changed.npz"
    np.savez(changed, **damaged)
    assert OQ.replay_query(Policy(), changed)["status"] == "FAIL"


def test_readback_detects_a_changed_file(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch)
    policy.infer({})
    path = tmp_path / "ep" / "rollout_0000.npz"
    record = load(path)
    SA.RolloutRecorder._check_readback(path, OQ.pack_observation(policy.observation) | {
        key: value for key, value in record.items() if not key.startswith("obs")})
    with pytest.raises(RuntimeError, match="读回不符: noise"):
        SA.RolloutRecorder._check_readback(path, OQ.pack_observation(policy.observation) | {
            key: value + 1 if key == "noise" else value for key, value in record.items() if not key.startswith("obs")})
    with pytest.raises(RuntimeError, match="读回键不符"):
        SA.RolloutRecorder._check_readback(path, {"noise": record["noise"]})


def test_motion_times_split_readout_and_rehook_after_reset(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch, motion=True)
    policy.infer({})
    assert policy.mem_buffer.motion_calls == 1
    first = load(tmp_path / "ep" / "rollout_0000.npz")
    assert first["motion_times"].tolist() == [4, 4, 4, 4, 8, 8, -1, -1]
    assert first["motion_times"].dtype == np.int32
    policy.times = (12, 16)
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": "ep2"})
    policy.reset()
    assert policy.mem_buffer is policy.buffers[-1]
    policy.infer({})
    assert policy.mem_buffer.motion_calls == 1
    assert load(tmp_path / "ep2" / "rollout_0000.npz")["motion_times"].tolist() == [12, 12, 12, 12, 16, 16, -1, -1]


def test_motion_times_without_split(tmp_path, monkeypatch):
    policy = install(tmp_path, monkeypatch, motion=True, readout="other")
    policy.infer({})
    assert load(tmp_path / "ep" / "rollout_0000.npz")["motion_times"].tolist() == [4, -1]


def test_infer_before_reset_is_rejected(tmp_path):
    policy = Policy()
    SA.RolloutRecorder(policy, tmp_path).install()
    with pytest.raises(RuntimeError, match="reset"):
        policy.infer({})


def server_args(tmp_path, **changes):
    values = dict(attn_out=tmp_path / "attn", save_top_frames=3, save_prefix=False, ablate="none", record_motion=None,
                  swap_donor=None, query_record=0, query_batch_size=None, oracle_mode="", rollout_record=True, seed=42,
                  policy=SimpleNamespace(dir="/ckpt"))
    return SimpleNamespace(**(values | changes))


@pytest.mark.parametrize("changes", [dict(query_record=10), dict(ablate="mask_vis"), dict(save_prefix=True),
                                     dict(record_motion="/tmp/x"), dict(swap_donor="StopCube:1")])
def test_main_rejects_conflicts(tmp_path, changes):
    with pytest.raises(SystemExit, match="互斥"):
        SA.main(server_args(tmp_path, **changes))


def patch_main(monkeypatch, policy):
    calls = {"rpc": None}
    monkeypatch.setattr(serve_policy, "create_policy", lambda args: policy)
    monkeypatch.setattr(serve_policy, "main", lambda args: serve_policy.create_policy(args))
    monkeypatch.setattr(SA, "_install_rpc_tracking", lambda **kwargs: calls.update(rpc=kwargs))
    return calls


def test_main_installs_recorder_without_adapter(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("rollout 记录不得构造 AttnAdapter")

    monkeypatch.setattr(attn_adapter, "AttnAdapter", forbidden)
    policy = Policy()
    calls = patch_main(monkeypatch, policy)
    SA.main(server_args(tmp_path))
    assert calls["rpc"] == {"query_record": 0, "oracle_mode": ""}
    recorder = policy._sample_actions.__self__
    assert isinstance(recorder, SA.RolloutRecorder) and recorder.model_id == "/ckpt"


def test_main_without_flag_keeps_attention_path(tmp_path, monkeypatch):
    class Reached(Exception):
        pass

    def adapter(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(attn_adapter, "AttnAdapter", adapter)
    patch_main(monkeypatch, Policy())
    with pytest.raises(Reached):
        SA.main(server_args(tmp_path, rollout_record=False))


def test_rpc_tracking_uses_default_session(monkeypatch):
    from scripts.evaluation.runtime_contract import RpcSession

    monkeypatch.setattr(websocket_policy_server, "RpcSession", RpcSession)
    SA._install_rpc_tracking(query_record=0)
    assert websocket_policy_server.RpcSession.__mro__[1] is RpcSession


def test_args_class_flag():
    import tyro

    Args = SA._make_args_class()
    tail = ["policy:checkpoint", "--policy.config", "cfg", "--policy.dir", "/ckpt"]
    assert tyro.cli(Args, args=["--attn-out", "/tmp/a", *tail]).rollout_record is False
    assert tyro.cli(Args, args=["--attn-out", "/tmp/a", "--rollout-record", *tail]).rollout_record is True


def base_spec(tmp_path, **changes):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema_version": 1, "episodes": {}}))
    values = dict(repo="/repo", policy_python="/policy/python", sim_python="/sim/python", stage2_python="/stage2/python",
                  checkpoint="/ckpt", policy_config="cfg", model_kind="stage2_pool4_concat", cuda="0",
                  attn_out="/out/attn", run_root="/out/run", manifest="/out/manifest.json", xla_mem_fraction=0.5,
                  stopcube_mode="normal", stopcube_config=str(config), model_seed=42, run_identity_sha256="a" * 64)
    return values | changes


def write_spec(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


@pytest.mark.parametrize("changes", [dict(stopcube_mode=None, stopcube_config=None), dict(stopcube_mode="hold_extended"),
                                     dict(stopcube_mode="oracle_query"), dict(rollout_record="true"),
                                     dict(rollout_record=1)])
def test_run_local_rejects_invalid_spec(tmp_path, changes):
    spec = base_spec(tmp_path) | {"rollout_record": True} | changes
    with pytest.raises(SystemExit, match="rollout_record"):
        local.load_spec(write_spec(tmp_path, spec))


def test_run_local_accepts_normal_mode(tmp_path):
    assert local.load_spec(write_spec(tmp_path, base_spec(tmp_path, rollout_record=True)))["rollout_record"] is True
    other = dict(base_spec(tmp_path, rollout_record=False), stopcube_mode="hold_extended")
    assert local.load_spec(write_spec(tmp_path, other))["rollout_record"] is False


def test_run_local_command_and_identity(tmp_path):
    spec = base_spec(tmp_path)
    off = local.build_server_command(spec, "nonce", 9000)
    assert "--rollout-record" not in off
    assert local.build_server_command(dict(spec, rollout_record=False), "nonce", 9000) == off
    on = local.build_server_command(dict(spec, rollout_record=True), "nonce", 9000)
    index = on.index("--rollout-record")
    assert index < on.index("policy:checkpoint") and on[:index] + on[index + 1:] == off
    manifest = {"sha256": "b" * 64}
    identity = local.run_identity(spec, manifest)
    # 缺键时身份必须仍是改动前的字段集合
    assert identity == M.digest({"manifest_seal": "b" * 64, "checkpoint": "/ckpt", "policy_config": "cfg",
        "model_kind": "stage2_pool4_concat", "model_seed": 42, "stopcube_mode": "normal",
        "stopcube_config_sha256": local.hashlib.sha256(open(spec["stopcube_config"], "rb").read()).hexdigest()})
    assert local.run_identity(dict(spec, rollout_record=False), manifest) == identity
    assert local.run_identity(dict(spec, rollout_record=True), manifest) != identity


@pytest.mark.parametrize("on, kept, count", [(True, "rollout_0000.npz", 1), (False, "infer_0000.npz", 1),
                                             (True, "infer_0000.npz", 0), (False, "rollout_0000.npz", 0)])
def test_collect_counts_the_right_files(tmp_path, on, kept, count):
    attn, output = tmp_path / "attn", tmp_path / "client"
    (attn / "n").mkdir(parents=True)
    (attn / "n" / kept).write_bytes(b"")
    (attn / "n" / "episode.json").write_text("{}")
    (output / "traces" / "n" / "meta").mkdir(parents=True)
    (output / "traces" / "n" / "meta" / "hold_receipt.json").write_text(json.dumps({"trace_path": "t", "video_path": "v"}))
    terminal = dict(episode_key="StopCube:16", episode_nonce="n", task_name="StopCube", episode_id=16, env_seed=1,
                    difficulty="easy", chunk_id="c", run_identity_sha256="r", outcome="success", rpc_seq=3)
    (output / "x.terminal.json").write_text(json.dumps(terminal))
    spec = base_spec(tmp_path, attn_out=str(attn), run_root=str(tmp_path / "run"), rollout_record=on)
    row = local.collect(spec, tmp_path, output, [{"episode_key": "StopCube:16", "task_name": "StopCube", "episode_id": 16}])[0]
    assert row["npz"] == count and row["infers"] == 1
