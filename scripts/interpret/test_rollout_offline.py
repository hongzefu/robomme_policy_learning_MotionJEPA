"""离线计算入口的 CPU 测试：真实 RolloutRecorder 写记录，假 policy 和假适配器，不加载权重。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from scripts.interpret import rollout_offline as RO
from scripts.interpret import serve_policy_attn as SA

EXPECTED_KEYS = {"group_names", "group_mass", "group_tokens_all", "hidden_prefix", "hidden_action", "hidden_group_tokens",
                 "p_actions", "raw_state", "episode_key", "infer_seq", "step_idx", "chunk_start_env_step",
                 "same_path_maxabs", "attn_path_maxabs", "input_sha256", "model_id", "frame_times", "motion_times",
                 "parity_x0_maxabs", "parity_group_mass_maxabs", "parity_group_total_maxabs", "parity_motion_rel_maxabs"}


def make_observation(motion):
    base = dict(images={"base": jnp.ones((1, 4, 4, 3), jnp.float32)}, image_masks={"base": jnp.ones((1,), bool)},
                state=jnp.arange(32, dtype=jnp.float32)[None], tokenized_prompt=jnp.ones((1, 4), jnp.int32),
                tokenized_prompt_mask=jnp.asarray([[True, True, True, False]]),
                static_image_emb=jnp.ones((1, 4, 6), jnp.bfloat16), static_mask=jnp.asarray([[True, True, True, False]]),
                static_pos_emb=jnp.ones((1, 4, 6), jnp.float32), static_state_emb=jnp.ones((1, 4, 8), jnp.float32))
    if not motion:
        return HistAugObservation(**base)
    # 8 个 motion token，前 6 个有效；记忆顺序把 motion 排在帧之前
    return HistAugObservation(**base, motion_emb=jnp.ones((1, 8, 6), jnp.bfloat16), motion_pos=jnp.ones((1, 8, 6), jnp.float32),
                              motion_mask=jnp.asarray([[True] * 6 + [False] * 2]),
                              mem_order=jnp.asarray([[4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3]], dtype=jnp.int32))


class Buffer:
    def get_frame_sampling_indices(self, step_idx, budget, token_per_image):
        return [0, 4, 8, step_idx]

    def _prepare_motion(self, step_idx):
        return None, None, None, np.asarray([4, 8])


class FakePolicy:
    """调用顺序与 policy.Policy.infer 一致；动作只由噪声和 state 决定。"""

    def __init__(self, motion=False):
        self.motion_enabled = motion
        self.config = SimpleNamespace(budget=4, token_per_image=1, num_views=1,
                                      motion=SimpleNamespace(readout="stage2_pool4_split_v1"))
        self._model = SimpleNamespace(action_horizon=20, action_dim=32)
        self._sample_kwargs = {"num_steps": 10}
        self.observation = make_observation(motion)
        self.sample_batches = []
        self.reset()

    def reset(self):
        self._rng = jax.random.key(42)
        self.step_idx = 16
        self.mem_buffer = Buffer()

    def _sample_actions(self, rng, obs, *, noise=None, num_steps=10):
        if noise is None:
            noise = jax.random.normal(rng, (obs.state.shape[0], 20, 32))
        self.sample_batches.append(noise.shape[0])
        return noise + obs.state[:, None, :]

    def _output_transform(self, outputs):
        return {"actions": outputs["actions"][:, :8] + outputs["state"][:8]}

    def infer(self, obs):
        if self.motion_enabled:
            self.mem_buffer._prepare_motion(self.step_idx)
        self._rng, key = jax.random.split(self._rng)
        values = self._sample_actions(key, self.observation, **self._sample_kwargs)
        return self._output_transform({"state": np.asarray(self.observation.state[0]), "actions": np.asarray(values[0])})


class FakeAdapter:
    """形状同 AttnAdapter.run；概率在全部 key 上均匀，组质量可手算。"""

    def __init__(self, *, offset=0.0, x0_delta=0.0, probs_hook=None):
        """x0_delta 只加在捕获开的 x_0 上；probs_hook(probs, capture_hidden) 就地改动作注意力。"""
        self.offset, self.x0_delta, self.probs_hook = offset, x0_delta, probs_hook
        self.calls = []

    def layout(self, observation):
        mem = int(np.asarray(observation.mem_order).size) if observation.mem_order is not None else 4
        return {"P": mem + 2 + 4, "MEM": mem, "IMG": 2, "TXT": 4, "n_img": 1}

    def run(self, rng, observation, *, noise, num_steps, out_f16, capture_hidden=False, group_masks=None):
        self.calls.append({"capture_hidden": capture_hidden, "noise": np.asarray(noise), "num_steps": num_steps,
                           "out_f16": out_f16, "group_masks": group_masks})
        keys = self.layout(observation)["P"] + 20
        x_0 = noise + observation.state[:, None, :] + self.offset
        if capture_hidden:
            x_0 = x_0 + self.x0_delta
        probs = np.full((num_steps, 18, 1, 8, 20, keys), 1 / keys, np.float32)
        if self.probs_hook is not None:
            self.probs_hook(probs, capture_hidden)
        text = np.zeros((18, 1, 8, 4, keys - 20), np.float32)
        lay = self.layout(observation)
        if not capture_hidden:
            return x_0, probs, text, num_steps, lay
        hidden = {"hidden_prefix": np.ones((19, 4, 5), np.float32), "hidden_action": np.ones((18, 3), np.float32),
                  "group_tokens": np.asarray(group_masks).sum(axis=1).astype(np.int32)}
        return x_0, probs, text, num_steps, lay, hidden


def write_episode(root, monkeypatch, *, nonce="ep", key="StopCube:16", motion=False, infers=2, trace=True, terminal=True):
    """真实记录器写 rollout_*.npz，按 stopcube_runner 的字段写轨迹，按 run_local.collect 的文件名写 accepted。"""
    policy = FakePolicy(motion)
    SA.RolloutRecorder(policy, root / "attn", model_id="/ckpt").install()
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": nonce})
    policy.reset()
    rows = []
    for seq in range(infers):
        policy.step_idx = 16 + seq * 16
        policy.infer({})
        for row in range(2):
            rows.append({"event": "step", "episode_key": key, "infer_seq": seq, "row_in_chunk": row,
                         "chunk_start_env_step": 100 + seq * 16, "state_before": [float(seq * 10 + row)] * 8})
    if trace:
        directory = root / "run" / "attempts" / "000-a" / "client" / "traces" / nonce
        directory.mkdir(parents=True)
        lines = [{"event": "episode_start", "episode_key": "错的", "infer_seq": 0, "row_in_chunk": 0}, *rows]
        (directory / "step_trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in lines))
    if terminal:
        accepted = root / "run" / "accepted"
        accepted.mkdir(parents=True, exist_ok=True)
        (accepted / f"{key}.{nonce[:8]}.terminal.json").write_text(json.dumps({"episode_key": key, "episode_nonce": nonce}))
    return policy


def arguments(root, **changes):
    values = dict(attn_root=root / "attn", run_root=root / "run", output=root / "offline", seed=42,
                  parity="first", query_batch_size=2, episodes=None)
    return SimpleNamespace(**(values | changes))


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def test_module_points_to_this_tree():
    assert Path(RO.__file__).resolve().parent == Path(__file__).resolve().parent


def test_trace_mapping_takes_row_zero_of_step_events(tmp_path, monkeypatch):
    write_episode(tmp_path, monkeypatch)
    rows = RO.load_trace(tmp_path / "run", "ep")
    assert sorted(rows) == [0, 1]
    assert rows[1]["state_before"] == [10.0] * 8 and rows[1]["row_in_chunk"] == 0
    assert rows[0]["episode_key"] == "StopCube:16"


def test_accepted_nonce_without_trace_stops(tmp_path, monkeypatch):
    policy = write_episode(tmp_path, monkeypatch, trace=False)
    with pytest.raises(SystemExit, match="step_trace"):
        RO.run(policy, FakeAdapter(), arguments(tmp_path))
    assert not (tmp_path / "offline").exists()


def test_missing_infer_seq_stops(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch)
    path = next((tmp_path / "run").rglob("step_trace.jsonl"))
    kept = [line for line in path.read_text().splitlines() if json.loads(line)["infer_seq"] != 1]
    path.write_text("\n".join(kept) + "\n")
    with pytest.raises(SystemExit, match="infer_seq=1"):
        RO.run(policy, FakeAdapter(), arguments(tmp_path))
    assert (tmp_path / "offline" / "ep" / "offline_0000.npz").exists()
    assert not (tmp_path / "offline" / "ep" / "offline_0001.npz").exists()


def test_replay_fail_stops_before_writing(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=1)
    path = tmp_path / "attn" / "ep" / "rollout_0000.npz"
    damaged = load(path)
    damaged["noise"] = damaged["noise"] + 0.01
    np.savez(path, **damaged)
    adapter = FakeAdapter()
    with pytest.raises(SystemExit) as stopped:
        RO.run(policy, adapter, arguments(tmp_path))
    assert stopped.value.code == 1
    assert "ROLLOUT_REPLAY=FAIL" in capsys.readouterr().out
    assert adapter.calls == [] and not (tmp_path / "offline").exists()


def parity_values(line):
    """把判定行里的 key=value 解析成字典。"""
    return dict(item.split("=", 1) for item in line.split()[1:])


def parity_line(output, prefix="HIDDEN_PARITY=PASS"):
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    assert len(lines) == 1, output
    return lines[0]


def history_key_shift(delta, *, layer=None, head=None):
    """捕获关时只改第 0 个 key（基线属历史视觉）；layer/head 为 None 表示全部层或头。"""

    def hook(probs, capture_hidden):
        if not capture_hidden:
            probs[:, slice(None) if layer is None else layer, :, slice(None) if head is None else head, :, 0] += delta

    return hook


def motion_mass(on, off):
    """把 motion 的 6 个有效 key 设成给定总质量（motion 模型，key 0～5）。"""

    def hook(probs, capture_hidden):
        probs[..., :6] = (on if capture_hidden else off) / 6

    return hook


def expect_parity_fail(tmp_path, monkeypatch, capsys, adapter, *, motion=False):
    policy = write_episode(tmp_path, monkeypatch, infers=1, motion=motion)
    with pytest.raises(SystemExit) as stopped:
        RO.run(policy, adapter, arguments(tmp_path))
    assert stopped.value.code == 1
    output = capsys.readouterr().out
    assert "HIDDEN_PARITY=PASS" not in output
    assert not (tmp_path / "offline" / "ep" / "offline_0000.npz").exists()
    line = parity_line(output, "HIDDEN_PARITY=FAIL")
    values = parity_values(line)
    assert values["points"] == "1" and values["path"].endswith("rollout_0000.npz")
    return values


def test_parity_x0_over_tolerance_fails(tmp_path, monkeypatch, capsys):
    values = expect_parity_fail(tmp_path, monkeypatch, capsys, FakeAdapter(x0_delta=0.03))
    assert float(values["x0_maxabs"]) == pytest.approx(0.03, abs=1e-4)
    assert float(values["group_mass_maxabs"]) == 0 and values["motion_rel_maxabs"] == "null"


def test_parity_single_head_over_tolerance_fails(tmp_path, monkeypatch, capsys):
    # 单头差 0.03 超限，跨 18 层 8 头平均后总质量差只有 0.03/144，未超限
    values = expect_parity_fail(tmp_path, monkeypatch, capsys, FakeAdapter(probs_hook=history_key_shift(0.03, layer=3, head=2)))
    assert float(values["group_mass_maxabs"]) == pytest.approx(0.03, abs=1e-6)
    assert float(values["group_total_maxabs"]) == pytest.approx(0.03 / 144, abs=1e-7)


def test_parity_group_total_over_tolerance_fails(tmp_path, monkeypatch, capsys):
    # 全部层头都差 0.002：单头未超限，总质量超限
    values = expect_parity_fail(tmp_path, monkeypatch, capsys, FakeAdapter(probs_hook=history_key_shift(0.002)))
    assert float(values["group_mass_maxabs"]) == pytest.approx(0.002, abs=1e-6)
    assert float(values["group_total_maxabs"]) == pytest.approx(0.002, abs=1e-6)


def test_parity_motion_relative_over_tolerance_fails(tmp_path, monkeypatch, capsys):
    # motion 总质量很小，绝对差都未超限，只有相对差超限
    values = expect_parity_fail(tmp_path, monkeypatch, capsys, FakeAdapter(probs_hook=motion_mass(1e-3, 1.2e-3)), motion=True)
    assert float(values["motion_rel_maxabs"]) == pytest.approx(0.2 / 1.2, abs=1e-5)
    assert float(values["group_mass_maxabs"]) <= RO.TOL_GROUP and float(values["group_total_maxabs"]) <= RO.TOL_TOTAL


def test_parity_within_tolerance_passes_with_measured_values(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=1, motion=True)
    adapter = FakeAdapter(x0_delta=0.01, probs_hook=motion_mass(0.1, 0.1005))
    assert RO.run(policy, adapter, arguments(tmp_path)) == 0
    output = capsys.readouterr().out
    line = parity_line(output)
    values = parity_values(line)
    assert values["points"] == "1"
    assert float(values["x0_maxabs"]) == pytest.approx(0.01, abs=1e-4)
    assert float(values["group_mass_maxabs"]) == pytest.approx(5e-4, abs=1e-6)
    assert float(values["group_total_maxabs"]) == pytest.approx(5e-4, abs=1e-6)
    assert float(values["motion_rel_maxabs"]) == pytest.approx(5e-4 / 0.1005, abs=1e-5)
    assert line.endswith("tol_x0=0.02 tol_group=0.02 tol_total=0.001 tol_motion_rel=0.1")
    assert "logic_f32" not in line
    note = parity_values(parity_line(output, "HIDDEN_PARITY_NOTE"))
    assert note["count"] == "0" and note["note_threshold"] == "0.003"
    assert float(note["group_mass_maxabs"]) == pytest.approx(5e-4, abs=1e-6)
    result = load(tmp_path / "offline" / "ep" / "offline_0000.npz")
    for name in ("x0_maxabs", "group_mass_maxabs", "group_total_maxabs", "motion_rel_maxabs"):
        assert float(result[f"parity_{name}"]) == pytest.approx(float(values[name]), rel=1e-6)


def test_parity_summary_takes_maximum_over_points(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    calls = []

    def hook(probs, capture_hidden):
        if not capture_hidden:
            calls.append(1)
            probs[..., 0] += 2e-4 if len(calls) == 1 else 5e-4

    assert RO.run(policy, FakeAdapter(probs_hook=hook), arguments(tmp_path, parity="all")) == 0
    values = parity_values(parity_line(capsys.readouterr().out))
    assert values["points"] == "2"
    assert float(values["group_total_maxabs"]) == pytest.approx(5e-4, abs=1e-6)
    first = load(tmp_path / "offline" / "ep" / "offline_0000.npz")
    assert float(first["parity_group_total_maxabs"]) == pytest.approx(2e-4, abs=1e-6)


def test_merge_parity_keeps_motion_reading_across_null():
    base = dict(x0_maxabs=0.1, group_mass_maxabs=0.2, group_total_maxabs=0.3)
    merged = RO.merge_parity(None, base | {"motion_rel_maxabs": None})
    merged = RO.merge_parity(merged, dict(x0_maxabs=0.05, group_mass_maxabs=0.4, group_total_maxabs=0.3, motion_rel_maxabs=0.02))
    merged = RO.merge_parity(merged, base | {"motion_rel_maxabs": None})
    assert merged == dict(x0_maxabs=0.1, group_mass_maxabs=0.4, group_total_maxabs=0.3, motion_rel_maxabs=0.02)


def test_parity_baseline_skips_motion_relative(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=1)
    assert RO.run(policy, FakeAdapter(), arguments(tmp_path)) == 0
    values = parity_values(parity_line(capsys.readouterr().out))
    assert values["motion_rel_maxabs"] == "null"
    result = load(tmp_path / "offline" / "ep" / "offline_0000.npz")
    assert np.isnan(result["parity_motion_rel_maxabs"])
    assert float(result["parity_x0_maxabs"]) == 0


def test_parity_motion_off_zero_and_on_zero_passes(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=1, motion=True)
    assert RO.run(policy, FakeAdapter(probs_hook=motion_mass(0.0, 0.0)), arguments(tmp_path)) == 0
    assert parity_values(parity_line(capsys.readouterr().out))["motion_rel_maxabs"] == "0"


def test_parity_motion_off_zero_and_on_nonzero_fails(tmp_path, monkeypatch, capsys):
    values = expect_parity_fail(tmp_path, monkeypatch, capsys, FakeAdapter(probs_hook=motion_mass(1e-5, 0.0)), motion=True)
    assert values["motion_rel_maxabs"] == "inf"


def test_parity_note_counts_points_and_keeps_maximum(tmp_path, monkeypatch, capsys):
    # 三个点的单头差依次为 0.01、0.001、0.004：两点超过 0.003，都未超限；只在汇总时打印一行
    deltas = iter((0.01, 0.001, 0.004))

    def hook(probs, capture_hidden):
        if not capture_hidden:
            probs[:, 0, :, 0, :, 0] += next(deltas)

    policy = write_episode(tmp_path, monkeypatch, infers=3)
    assert RO.run(policy, FakeAdapter(probs_hook=hook), arguments(tmp_path, parity="all")) == 0
    output = capsys.readouterr().out
    parity_line(output)
    values = parity_values(parity_line(output, "HIDDEN_PARITY_NOTE"))
    assert set(values) == {"count", "group_mass_maxabs", "note_threshold"}
    assert values["count"] == "2" and values["note_threshold"] == "0.003"
    assert float(values["group_mass_maxabs"]) == pytest.approx(0.01, abs=1e-6)


def test_parity_does_not_compare_single_probability_elements(tmp_path, monkeypatch, capsys):
    # 单元素差 0.4，平均 10 个扩散步和 16 个动作位置后组质量差 0.0025，未超限、也不计入信息行
    def hook(probs, capture_hidden):
        if not capture_hidden:
            probs[0, 0, 0, 0, 0, 0] += 0.4

    policy = write_episode(tmp_path, monkeypatch, infers=1)
    assert RO.run(policy, FakeAdapter(probs_hook=hook), arguments(tmp_path)) == 0
    output = capsys.readouterr().out
    assert float(parity_values(parity_line(output))["group_mass_maxabs"]) == pytest.approx(0.0025, abs=1e-6)
    assert parity_values(parity_line(output, "HIDDEN_PARITY_NOTE"))["count"] == "0"


def test_attention_and_hidden_come_from_capture_on_run(tmp_path, monkeypatch, capsys):
    # 输出的组质量不受捕获关那次运行影响
    policy = write_episode(tmp_path, monkeypatch, infers=1)
    RO.run(policy, FakeAdapter(probs_hook=history_key_shift(5e-4)), arguments(tmp_path))
    result = load(tmp_path / "offline" / "ep" / "offline_0000.npz")
    np.testing.assert_allclose(result["group_mass"][:, 3, 2], result["group_tokens_all"] / 30, atol=1e-5)


def test_parity_not_compared_records_nan(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    assert RO.run(policy, FakeAdapter(), arguments(tmp_path, parity="first")) == 0
    first = load(tmp_path / "offline" / "ep" / "offline_0000.npz")
    second = load(tmp_path / "offline" / "ep" / "offline_0001.npz")
    assert float(first["parity_group_mass_maxabs"]) == 0
    for name in ("x0_maxabs", "group_mass_maxabs", "group_total_maxabs", "motion_rel_maxabs"):
        assert np.isnan(second[f"parity_{name}"]) and second[f"parity_{name}"].dtype == np.float64


@pytest.mark.parametrize("mode, expected", [("first", 1), ("all", 3)])
def test_parity_frequency(tmp_path, monkeypatch, capsys, mode, expected):
    policy = write_episode(tmp_path, monkeypatch, infers=3)
    adapter = FakeAdapter()
    assert RO.run(policy, adapter, arguments(tmp_path, parity=mode)) == 0
    plain = [call for call in adapter.calls if not call["capture_hidden"]]
    assert len(plain) == expected and sum(call["capture_hidden"] for call in adapter.calls) == 3
    values = parity_values(parity_line(capsys.readouterr().out))
    assert values["points"] == str(expected) and float(values["x0_maxabs"]) == 0


def test_output_keys_values_and_no_temp_files(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch)
    adapter = FakeAdapter(offset=0.25)
    assert RO.run(policy, adapter, arguments(tmp_path)) == 0
    directory = tmp_path / "offline" / "ep"
    assert sorted(path.name for path in directory.iterdir()) == ["offline_0000.npz", "offline_0001.npz"]
    result = load(directory / "offline_0001.npz")
    assert set(result) == EXPECTED_KEYS
    assert result["p_actions"].shape == (10, 20, 8)
    assert result["raw_state"].tolist() == [10.0] * 8 and result["raw_state"].dtype == np.float32
    assert str(result["episode_key"]) == "StopCube:16" and int(result["infer_seq"]) == 1
    assert int(result["step_idx"]) == 32 and int(result["chunk_start_env_step"]) == 116
    assert float(result["same_path_maxabs"]) == 0
    assert float(result["attn_path_maxabs"]) == pytest.approx(0.25, abs=1e-4)      # 假动作是 float32，大 state 上有舍入
    assert str(result["model_id"]) == "/ckpt"
    assert result["frame_times"].tolist() == [0, 4, 8, 32]
    names = result["group_names"].tolist()
    assert names[:5] == ["motion", "历史视觉", "当前视觉", "文字", "其余"]
    assert result["group_mass"].shape == (len(names), 18, 8) and result["group_mass"].dtype == np.float32
    assert result["group_tokens_all"].shape == (len(names),)
    # 均匀概率下组质量 = token 数 / key 总数；P=10，共 30 个 key
    np.testing.assert_allclose(result["group_mass"][:, 3, 2], result["group_tokens_all"] / 30, atol=1e-5)
    assert result["hidden_prefix"].shape == (19, 4, 5) and result["hidden_action"].shape == (18, 3)
    output = capsys.readouterr().out
    assert "ROLLOUT_REPLAY=PASS points=2 same_path_maxabs=0 tol=1e-6" in output
    assert "OFFLINE_RUN=PASS records=2 skipped=0 skipped_nonce=0 attn_path_maxabs=0.25" in output
    assert output.count("OFFLINE_EPISODE") == 1


def test_attention_uses_actual_noise_and_p_uses_query_noise(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=1)
    adapter = FakeAdapter()
    policy.sample_batches.clear()
    RO.run(policy, adapter, arguments(tmp_path))
    record = load(tmp_path / "attn" / "ep" / "rollout_0000.npz")
    captured = [call for call in adapter.calls if call["capture_hidden"]][0]
    np.testing.assert_array_equal(captured["noise"], record["noise"])
    assert captured["num_steps"] == 10 and captured["out_f16"] is False
    # 重放 batch=1 一次；P 用动作 batch=2 五次
    assert policy.sample_batches == [1] + [2] * 5
    noise = RO.query_noise(42, "StopCube:16", 0, 10, 20, 32)
    expected = noise[:, :, :8] + np.arange(8, dtype=np.float32) * 2
    np.testing.assert_allclose(load(tmp_path / "offline" / "ep" / "offline_0000.npz")["p_actions"], expected, atol=1e-5)


def test_resume_skips_matching_and_rejects_changed_input(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch)
    args = arguments(tmp_path)
    RO.run(policy, FakeAdapter(offset=0.5), args)
    capsys.readouterr()
    output = tmp_path / "offline" / "ep" / "offline_0000.npz"
    before = output.read_bytes()
    adapter = FakeAdapter()
    assert RO.run(policy, adapter, args) == 0
    text = capsys.readouterr().out
    # 跳过的记录仍计入全局最大值
    assert "OFFLINE_RUN=PASS records=0 skipped=2 skipped_nonce=0 attn_path_maxabs=0.5" in text
    assert "HIDDEN_PARITY=SKIPPED points=0" in text and "HIDDEN_PARITY=PASS" not in text
    assert "HIDDEN_PARITY_NOTE" not in text
    assert "ROLLOUT_REPLAY=PASS points=2" in text
    assert adapter.calls == [] and output.read_bytes() == before
    path = tmp_path / "attn" / "ep" / "rollout_0000.npz"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(SystemExit, match="input_sha256"):
        RO.run(policy, adapter, args)
    assert output.read_bytes() == before


def test_resume_processes_only_missing_records(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch)
    RO.run(policy, FakeAdapter(), arguments(tmp_path))
    (tmp_path / "offline" / "ep" / "offline_0001.npz").unlink()
    capsys.readouterr()
    adapter = FakeAdapter()
    RO.run(policy, adapter, arguments(tmp_path))
    text = capsys.readouterr().out
    assert "OFFLINE_RUN=PASS records=1 skipped=1" in text
    # 第一份已在上次比过，本次不再比
    assert "HIDDEN_PARITY=SKIPPED points=0" in text and len(adapter.calls) == 1


def test_motion_windows_follow_mem_order():
    observation = SimpleNamespace(static_mask=np.array([[1, 1, 0]], bool), mem_order=np.array([[5, 0, 3, 7, 1, 4, 2, 6]]))
    groups = {"motion": np.zeros(13, bool)}
    groups["motion"][[0, 2, 5, 7]] = True      # 记忆位 → order 5、3、4、6 → motion token 2、0、1、3
    result = RO.motion_window_groups(observation, groups, np.array([8, 4, 8, 12, -1]), 8)
    assert list(result) == ["motion窗:4", "motion窗:8", "motion窗:12"]
    assert np.flatnonzero(result["motion窗:4"]).tolist() == [5]
    assert np.flatnonzero(result["motion窗:8"]).tolist() == [0, 2]
    assert np.flatnonzero(result["motion窗:12"]).tolist() == [7]
    assert all(len(mask) == 13 for mask in result.values())


def test_motion_window_rejects_invalid_time():
    observation = SimpleNamespace(static_mask=np.array([[1]], bool), mem_order=np.array([[1, 0]]))
    groups = {"motion": np.array([True, False, False])}
    with pytest.raises(ValueError, match="无效位"):
        RO.motion_window_groups(observation, groups, np.array([-1]), 2)
    with pytest.raises(ValueError, match="超出"):
        RO.motion_window_groups(observation, groups, np.zeros(0, int), 2)


def test_motion_model_windows_and_hidden_mask_order(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, motion=True, infers=1)
    adapter = FakeAdapter()
    RO.run(policy, adapter, arguments(tmp_path))
    result = load(tmp_path / "offline" / "ep" / "offline_0000.npz")
    names = result["group_names"].tolist()
    tokens = dict(zip(names, result["group_tokens_all"].tolist()))
    assert tokens["motion窗:4"] == 4 and tokens["motion窗:8"] == 2 and tokens["motion"] == 6
    assert result["motion_times"].tolist() == [4, 4, 4, 4, 8, 8, -1, -1]
    masks = [call for call in adapter.calls if call["capture_hidden"]][0]["group_masks"]
    # P = 12 + 2 + 4；记忆位 0～7 是 motion，8～10 是有效帧
    assert masks.shape == (4, 18) and masks.dtype == bool
    assert np.flatnonzero(masks[0]).tolist() == [8, 9, 10]            # 历史视觉
    assert np.flatnonzero(masks[1]).tolist() == list(range(6))        # motion
    assert np.flatnonzero(masks[2]).tolist() == [12, 13]              # 当前视觉
    assert np.flatnonzero(masks[3]).tolist() == [14, 15, 16]          # 文字
    assert result["hidden_group_tokens"].tolist() == [3, 6, 2, 3]


def test_baseline_has_no_motion_windows_and_empty_motion_group(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=1)
    adapter = FakeAdapter()
    RO.run(policy, adapter, arguments(tmp_path))
    result = load(tmp_path / "offline" / "ep" / "offline_0000.npz")
    names = result["group_names"].tolist()
    assert not [name for name in names if name.startswith("motion窗:")]
    assert result["group_tokens_all"][names.index("motion")] == 0
    assert result["motion_times"].shape == (0,)
    masks = [call for call in adapter.calls if call["capture_hidden"]][0]["group_masks"]
    assert masks.shape == (4, 10)
    assert result["hidden_group_tokens"].tolist() == [3, 0, 2, 3]


def test_episode_filter_and_missing_episode(tmp_path, monkeypatch, capsys):
    # 第 16 局没有轨迹：过滤必须在读轨迹之前，不碰别的局
    write_episode(tmp_path, monkeypatch, nonce="a", key="StopCube:16", infers=1, trace=False)
    policy = write_episode(tmp_path, monkeypatch, nonce="b", key="StopCube:17", infers=1)
    assert RO.run(policy, FakeAdapter(), arguments(tmp_path, episodes=["StopCube:17"])) == 0
    assert sorted(path.name for path in (tmp_path / "offline").iterdir()) == ["b"]
    with pytest.raises(SystemExit, match="StopCube:99"):
        RO.run(policy, FakeAdapter(), arguments(tmp_path, episodes=["StopCube:17", "StopCube:99"]))


def test_stale_nonce_is_skipped_and_counted(tmp_path, monkeypatch, capsys):
    # 旧 nonce 没有 terminal：一份没有轨迹，一份轨迹不完整，都不处理、不报错
    write_episode(tmp_path, monkeypatch, nonce="old1", key="StopCube:16", infers=1, trace=False, terminal=False)
    write_episode(tmp_path, monkeypatch, nonce="old2", key="StopCube:16", infers=2, terminal=False)
    path = next((tmp_path / "run" / "attempts").rglob("old2/step_trace.jsonl"))
    path.write_text(path.read_text().splitlines()[0] + "\n")
    policy = write_episode(tmp_path, monkeypatch, nonce="new", key="StopCube:16", infers=1)
    assert RO.run(policy, FakeAdapter(), arguments(tmp_path)) == 0
    text = capsys.readouterr().out
    assert "OFFLINE_SKIP nonce=old1 reason=no_terminal" in text
    assert "OFFLINE_SKIP nonce=old2 reason=no_terminal" in text
    assert "OFFLINE_RUN=PASS records=1 skipped=0 skipped_nonce=2 " in text
    assert sorted(path.name for path in (tmp_path / "offline").iterdir()) == ["new"]


def test_only_stale_nonces_is_not_a_pass(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=1, terminal=False)
    with pytest.raises(SystemExit, match="records=0"):
        RO.run(policy, FakeAdapter(), arguments(tmp_path))


def test_empty_input_is_not_a_pass(tmp_path, monkeypatch, capsys):
    (tmp_path / "attn").mkdir()
    with pytest.raises(SystemExit, match="records=0"):
        RO.run(FakePolicy(), FakeAdapter(), arguments(tmp_path))
    assert "OFFLINE_RUN=PASS" not in capsys.readouterr().out


def test_non_finite_attention_path_stops(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=1)
    with pytest.raises(ValueError, match="非有限"):
        RO.run(policy, FakeAdapter(offset=float("nan")), arguments(tmp_path))
    assert not (tmp_path / "offline" / "ep" / "offline_0000.npz").exists()


@pytest.fixture(scope="module", autouse=True)
def _verdict_line(request):
    """pytest 收尾时打印判定行，排在 -q 摘要之后。"""
    session = request.session

    class _Plugin:
        def pytest_unconfigure(self):
            verdict = "PASS" if session.testsfailed == 0 else "FAIL"
            print(f"OFFLINE_UNIT={verdict} tests={session.testscollected}")

    request.config.pluginmanager.register(_Plugin(), "offline-unit-verdict")
    yield
