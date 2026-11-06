"""窗口离线注意力的 CPU 测试：真实 RolloutRecorder 写记录，假 policy 和假适配器，不加载权重。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from scripts.interpret import first_bin_offline as FB
from scripts.interpret import serve_policy_attn as SA

TXT, TXT_LEN, IMG = 64, 19, 2
BASE_DTYPES = {"P": "int64", "MEM": "int64", "step_idx": "int64", "exec_start_idx": "int64", "infer_seq": "int64",
               "mem_order": "int32", "mem_mask": "bool", "frame_times": "int32", "motion_times": "int32", "motion_mask": "bool",
               "txt_len": "int64", "actions": "float32", "state": "float32", "top_frames": "int32",
               "probs_action": "float16", "probs_text": "float16"}
PREFIX_DTYPES = {"probs_prefix": "float16", "P": "int64", "MEM": "int64", "infer_seq": "int64", "step_idx": "int64"}


def make_observation(motion):
    prompt_mask = np.arange(TXT) < TXT_LEN
    base = dict(images={"base": jnp.ones((1, 4, 4, 3), jnp.float32)}, image_masks={"base": jnp.ones((1,), bool)},
                state=jnp.arange(32, dtype=jnp.float32)[None], tokenized_prompt=jnp.ones((1, TXT), jnp.int32),
                tokenized_prompt_mask=jnp.asarray(prompt_mask[None]),
                static_image_emb=jnp.ones((1, 4, 6), jnp.bfloat16), static_mask=jnp.asarray([[True, True, True, False]]),
                static_pos_emb=jnp.ones((1, 4, 6), jnp.float32), static_state_emb=jnp.ones((1, 4, 8), jnp.float32))
    if not motion:
        return HistAugObservation(**base)
    # 8 个 motion token，前 6 个有效；记忆顺序把 motion 排在帧之前
    return HistAugObservation(**base, motion_emb=jnp.full((1, 8, 6), 2.0, jnp.float32),
                              motion_pos=jnp.full((1, 8, 5), 3.0, jnp.float32),
                              motion_mask=jnp.asarray([[True] * 6 + [False] * 2]),
                              mem_order=jnp.asarray([[4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3]], dtype=jnp.int32))


class Buffer:
    def get_frame_sampling_indices(self, step_idx, budget, token_per_image):
        return [0, step_idx]      # 只有两帧，其余位应填 -1

    def _prepare_motion(self, step_idx):
        return None, None, None, np.asarray([4, 8])


class FakePolicy:
    """调用顺序与 policy.Policy.infer 一致；动作只由噪声和 state 决定。"""

    def __init__(self, motion=False):
        self.motion_enabled = motion
        self.config = SimpleNamespace(budget=4, token_per_image=1, num_views=1,
                                      motion=SimpleNamespace(readout="stage2_pool4_split_v1", budget=8))
        self._model = SimpleNamespace(action_horizon=20, action_dim=32)
        self._sample_kwargs = {"num_steps": 10}
        self.observation = make_observation(motion)
        self.reset()

    def reset(self):
        self._rng = jax.random.key(42)
        self.step_idx = 16
        self.mem_buffer = Buffer()

    def _sample_actions(self, rng, obs, *, noise=None, num_steps=10):
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
        return self._output_transform({"state": np.asarray(self.observation.state[0]), "actions": np.asarray(values[0])})


class FakeAdapter:
    """形状同 AttnAdapter.run(save_prefix=True)：保留 batch 轴；概率在有效 key 上均匀，padding 为 0。"""

    def __init__(self, *, offset=0.0, hook=None, steps=None):
        self.offset, self.hook, self.steps = offset, hook, steps
        self.calls = []

    def run(self, rng, observation, *, noise, num_steps, out_f16, save_prefix):
        self.calls.append(dict(rng=np.asarray(rng), noise=np.asarray(noise), num_steps=num_steps, out_f16=out_f16,
                               save_prefix=save_prefix))
        motion = observation.mem_order is not None
        static = np.asarray(observation.static_mask)[0]
        if motion:
            full = np.r_[static, np.asarray(observation.motion_mask)[0]]
            mem_mask = full[np.asarray(observation.mem_order)[0]]
        else:
            mem_mask = static
        MEM = len(mem_mask)
        P = MEM + IMG + TXT
        prompt = np.arange(TXT) < TXT_LEN
        valid = np.r_[mem_mask, np.ones(IMG, bool), prompt]
        action_keys = np.r_[valid, np.ones(20, bool)]
        action = np.where(action_keys, 1 / action_keys.sum(), 0).astype(np.float16)
        row = np.where(valid, 1 / valid.sum(), 0).astype(np.float16)
        probs = {"action": np.broadcast_to(action, (num_steps, 18, 1, 8, 20, P + 20)).copy(),
                 "text": np.broadcast_to(row, (18, 1, 8, TXT, P)).copy(), "prefix": np.broadcast_to(row, (18, 1, 8, P, P)).copy()}
        probs["text"][:, :, :, ~prompt] = 0
        probs["prefix"][:, :, :, ~valid] = 0
        if self.hook is not None:
            self.hook(probs)
        x_0 = noise + observation.state[:, None, :] + self.offset
        layout = {"P": P, "MEM": MEM, "IMG": IMG, "TXT": TXT, "n_img": 1}
        return x_0, probs["action"], probs["text"], self.steps or num_steps, layout, probs["prefix"]


def write_episode(root, monkeypatch, *, nonce="ep", key="ButtonUnmaskSwap:0", shard="g0", motion=False, infers=6, row_changes=None):
    """真实记录器写 rollout_*.npz；轨迹字段同 1.1；accepted 文件名同 run_local.collect。"""
    policy = FakePolicy(motion)
    SA.RolloutRecorder(policy, root / shard / "attn", model_id="/ckpt").install()
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": nonce})
    policy.reset()
    rows = []
    for seq in range(infers):
        policy.step_idx = 16 + seq * 16
        policy.infer({})
        for row in range(2):
            rows.append({"event": "step", "episode_key": key, "infer_seq": seq, "row_in_chunk": row, "exec_start_idx": 7,
                         "chunk_start_env_step": 100 + seq * 16, "state_before": [float(seq * 10 + row)] * 8, **(row_changes or {})})
    directory = root / shard / "run" / "attempts" / "000-a" / "client" / "traces" / nonce
    directory.mkdir(parents=True)
    (directory / "step_trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    accepted = root / shard / "run" / "accepted"
    accepted.mkdir(parents=True, exist_ok=True)
    (accepted / f"{key}.{nonce[:8]}.terminal.json").write_text(json.dumps({"episode_key": key, "episode_nonce": nonce}))
    return policy


def write_windows(root, entries):
    path = root / "windows.json"
    path.write_text(json.dumps({"schema_version": 1, "episodes": entries}))
    return path


def entry(key="ButtonUnmaskSwap:0", shard="g0", G=4, window=None, nonce="ep"):
    """字段名同 first_bin_windows 输出。"""
    window = list(range(max(0, G - 3), G + 1)) if window is None else window
    return {"episode_nonce": nonce, "episode_key": key, "shard": shard, "G": G, "window": window,
            "window_status": "complete" if len(window) == 4 else "incomplete"}


def arguments(root, shard="g0"):
    return SimpleNamespace(windows=root / "windows.json", run_root=[root / "g0" / "run", root / "g1" / "run"],
                           attn_root=[root / "g0" / "attn", root / "g1" / "attn"], output=root / "offline", shard=shard)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def names(root, nonce="ep"):
    return sorted(path.name for path in (root / "offline" / nonce).iterdir())


def test_module_points_to_this_tree():
    assert Path(FB.__file__).resolve().parent == Path(__file__).resolve().parent


def test_only_window_records_are_processed(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch)
    write_windows(tmp_path, [entry(G=4)])
    adapter = FakeAdapter()
    assert FB.run(policy, adapter, arguments(tmp_path)) == 0
    assert len(adapter.calls) == 4
    assert names(tmp_path) == [f"{kind}_{seq:04d}.npz" for kind in ("infer", "prefix") for seq in (1, 2, 3, 4)]


def test_top_level_must_be_schema_v1(tmp_path, monkeypatch):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    (tmp_path / "windows.json").write_text(json.dumps([entry(G=1)]))
    with pytest.raises(SystemExit, match="schema_version"):
        FB.run(policy, FakeAdapter(), arguments(tmp_path))


def test_incomplete_window_near_start_is_processed(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch)
    write_windows(tmp_path, [entry(G=1)])
    assert entry(G=1)["window_status"] == "incomplete"
    assert FB.run(policy, FakeAdapter(), arguments(tmp_path)) == 0
    assert names(tmp_path) == ["infer_0000.npz", "infer_0001.npz", "prefix_0000.npz", "prefix_0001.npz"]


def test_window_inconsistent_with_G_stops(tmp_path, monkeypatch):
    policy = write_episode(tmp_path, monkeypatch)
    write_windows(tmp_path, [entry(G=4, window=[1, 4])])
    with pytest.raises(SystemExit, match="window"):
        FB.run(policy, FakeAdapter(), arguments(tmp_path))


@pytest.mark.parametrize("status", ["no_pick", "no_close_before_pick"])
def test_episode_without_window_is_skipped_and_reported(tmp_path, monkeypatch, capsys, status):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    write_windows(tmp_path, [{"episode_nonce": "none", "episode_key": "ButtonUnmaskSwap:9", "shard": "g0", "G": None,
                              "window": [], "window_status": status}, entry(G=1)])
    assert FB.run(policy, FakeAdapter(), arguments(tmp_path)) == 0
    output = capsys.readouterr().out
    assert f"OFFLINE_SKIP episode=ButtonUnmaskSwap:9 reason={status}" in output
    assert "no_window=1" in output and "ROLLOUT_REPLAY=PASS points=2 " in output


def test_unknown_window_status_stops(tmp_path, monkeypatch):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    write_windows(tmp_path, [entry(G=1) | {"window_status": "别的"}])
    with pytest.raises(SystemExit, match="window_status"):
        FB.run(policy, FakeAdapter(), arguments(tmp_path))


def test_shard_filter_runs_only_own_episodes(tmp_path, monkeypatch, capsys):
    # 第 0 局的轨迹不存在也不碰：分片过滤必须在读轨迹之前
    write_episode(tmp_path, monkeypatch, nonce="a", key="ButtonUnmaskSwap:0", shard="g0", infers=2)
    (tmp_path / "g0" / "run" / "attempts").rename(tmp_path / "g0" / "run" / "moved")
    policy = write_episode(tmp_path, monkeypatch, nonce="b", key="ButtonUnmaskSwap:1", shard="g1", infers=2)
    write_windows(tmp_path, [entry("ButtonUnmaskSwap:0", "g0", 1, nonce="a"), entry("ButtonUnmaskSwap:1", "g1", 1, nonce="b")])
    assert FB.run(policy, FakeAdapter(), arguments(tmp_path, "g1")) == 0
    assert sorted(path.name for path in (tmp_path / "offline").iterdir()) == ["b"]


def test_empty_shard_is_not_a_pass(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    write_windows(tmp_path, [entry(shard="g0", G=1)])
    with pytest.raises(SystemExit, match="没有可处理的记录"):
        FB.run(policy, FakeAdapter(), arguments(tmp_path, "g1"))
    assert "=PASS" not in capsys.readouterr().out


def test_replay_failure_exits_nonzero_before_attention(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch)
    path = tmp_path / "g0" / "attn" / "ep" / "rollout_0002.npz"
    damaged = load(path)
    damaged["noise"] = damaged["noise"] + 0.01
    np.savez(path, **damaged)
    adapter = FakeAdapter()
    write_windows(tmp_path, [entry(G=4)])
    with pytest.raises(SystemExit) as stopped:
        FB.run(policy, adapter, arguments(tmp_path))
    assert stopped.value.code == 1
    output = capsys.readouterr().out
    assert "ROLLOUT_REPLAY=FAIL points=1" in output and "ROLLOUT_REPLAY=PASS" not in output
    assert len(adapter.calls) == 1            # 第 1 条算完，第 2 条在重放时停
    assert names(tmp_path) == ["infer_0001.npz", "prefix_0001.npz"]


def test_resume_skips_finished_and_rejects_changed_input(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch)
    write_windows(tmp_path, [entry(G=4)])
    FB.run(policy, FakeAdapter(offset=0.5), arguments(tmp_path))
    capsys.readouterr()
    output = tmp_path / "offline" / "ep" / "infer_0001.npz"
    before = output.read_bytes()
    adapter = FakeAdapter()
    assert FB.run(policy, adapter, arguments(tmp_path)) == 0
    text = capsys.readouterr().out
    assert adapter.calls == [] and output.read_bytes() == before
    assert "written=0 skipped=4" in text
    # 跳过的记录仍计入最大值
    line = text.splitlines()[-1]
    assert line.startswith("ATTN_WINDOW=PASS records=4 ")
    assert float(line.split("attn_path_maxabs=")[1]) == pytest.approx(0.5, abs=1e-4)      # 假动作是 float32，大 state 上有舍入
    path = tmp_path / "g0" / "attn" / "ep" / "rollout_0001.npz"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(SystemExit, match="input_sha256"):
        FB.run(policy, adapter, arguments(tmp_path))
    assert output.read_bytes() == before


def test_resume_recomputes_only_incomplete_records(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch)
    write_windows(tmp_path, [entry(G=4)])
    FB.run(policy, FakeAdapter(), arguments(tmp_path))
    (tmp_path / "offline" / "ep" / "infer_0002.npz").unlink()
    (tmp_path / "offline" / "ep" / "prefix_0003.npz").unlink()
    capsys.readouterr()
    adapter = FakeAdapter()
    assert FB.run(policy, adapter, arguments(tmp_path)) == 0
    assert len(adapter.calls) == 2 and "written=2 skipped=2" in capsys.readouterr().out
    assert len(names(tmp_path)) == 8


@pytest.mark.parametrize("motion", [False, True])
def test_key_table_dtypes_and_shapes(tmp_path, monkeypatch, capsys, motion):
    policy = write_episode(tmp_path, monkeypatch, motion=motion)
    write_windows(tmp_path, [entry(G=4)])
    assert FB.run(policy, FakeAdapter(), arguments(tmp_path)) == 0
    infer = load(tmp_path / "offline" / "ep" / "infer_0002.npz")
    prefix = load(tmp_path / "offline" / "ep" / "prefix_0002.npz")
    motion_keys = set(SA.MOTION_INPUT_NPZ_KEYS) if motion else set()
    assert set(infer) == set(SA.NPZ_KEYS) | motion_keys | set(FB.EXTRA_NPZ_KEYS)
    assert set(BASE_DTYPES) == set(SA.NPZ_KEYS) and set(PREFIX_DTYPES) == set(SA.PREFIX_NPZ_KEYS) == set(prefix)
    assert {key: str(infer[key].dtype) for key in SA.NPZ_KEYS} == BASE_DTYPES
    assert {key: str(prefix[key].dtype) for key in SA.PREFIX_NPZ_KEYS} == PREFIX_DTYPES
    assert str(infer["same_path_maxabs"].dtype) == str(infer["attn_path_maxabs"].dtype) == "float64"
    assert infer["input_sha256"].dtype.kind == "U"
    MEM = 12 if motion else 4
    P = MEM + IMG + TXT
    assert int(infer["P"]) == P and int(infer["MEM"]) == MEM
    # 去 batch 轴后的形状
    assert infer["probs_action"].shape == (10, 18, 8, 20, P + 20)
    assert infer["probs_text"].shape == (18, 8, 64, P)
    assert prefix["probs_prefix"].shape == (18, 8, P, P)
    assert infer["actions"].shape == (20, 8) and infer["state"].shape == (8,)
    assert infer["frame_times"].shape == (4,) and infer["mem_order"].shape == infer["mem_mask"].shape == (MEM,)
    for key in ("P", "MEM", "infer_seq", "step_idx"):
        assert int(prefix[key]) == int(infer[key])


def test_baseline_rebuilt_values(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch)
    write_windows(tmp_path, [entry(G=4)])
    FB.run(policy, FakeAdapter(offset=0.25), arguments(tmp_path))
    record = load(tmp_path / "g0" / "attn" / "ep" / "rollout_0002.npz")
    infer = load(tmp_path / "offline" / "ep" / "infer_0002.npz")
    assert int(infer["infer_seq"]) == 2 and int(infer["step_idx"]) == 48 and int(infer["exec_start_idx"]) == 7
    assert infer["frame_times"].tolist() == [0, 48, -1, -1]
    assert infer["mem_order"].tolist() == [0, 1, 2, 3] and infer["mem_mask"].tolist() == [True, True, True, False]
    assert infer["motion_times"].shape == (0,) and infer["motion_mask"].shape == (0,)
    assert int(infer["txt_len"]) == TXT_LEN
    # 物理状态来自轨迹 row_in_chunk==0 行，不是观测里的归一化 state
    assert infer["state"].tolist() == [20.0] * 8
    np.testing.assert_array_equal(infer["actions"], record["actions"][0])
    # 帧 0、1 有效且质量相等，并列时保持帧序；padding 帧不进
    assert infer["top_frames"].tolist() == [0, 48]
    assert float(infer["same_path_maxabs"]) == 0
    assert float(infer["attn_path_maxabs"]) == pytest.approx(0.25, abs=1e-4)
    assert str(infer["input_sha256"]) == FB.fingerprint(tmp_path / "g0" / "attn" / "ep" / "rollout_0002.npz")["sha256"]


def test_motion_rebuilt_values_follow_mem_order(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, motion=True, infers=2)
    write_windows(tmp_path, [entry(G=1)])
    assert FB.run(policy, FakeAdapter(), arguments(tmp_path)) == 0
    infer = load(tmp_path / "offline" / "ep" / "infer_0001.npz")
    assert infer["mem_order"].tolist() == [4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3]
    assert infer["motion_mask"].tolist() == [True] * 6 + [False] * 2
    assert infer["motion_times"].tolist() == [4, 4, 4, 4, 8, 8, -1, -1]
    assert infer["mem_mask"].tolist() == [True] * 6 + [False] * 2 + [True, True, True, False]
    assert infer["motion_emb"].shape == (8, 6) and infer["motion_emb"].dtype == np.float32
    assert infer["motion_pos"].shape == (8, 5) and infer["motion_pos"].dtype == np.float32
    assert float(infer["motion_emb"][0, 0]) == 2.0 and float(infer["motion_pos"][0, 0]) == 3.0


def test_adapter_gets_recorded_noise_rng_and_flags(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    write_windows(tmp_path, [entry(G=1)])
    adapter = FakeAdapter()
    FB.run(policy, adapter, arguments(tmp_path))
    record = load(tmp_path / "g0" / "attn" / "ep" / "rollout_0001.npz")
    call = adapter.calls[1]
    np.testing.assert_array_equal(call["noise"], record["noise"])
    np.testing.assert_array_equal(call["rng"], record["sample_rng"])
    assert call["num_steps"] == 10 and call["out_f16"] is True and call["save_prefix"] is True


def test_final_lines_are_verdicts(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch)
    write_windows(tmp_path, [entry(G=4)])
    FB.run(policy, FakeAdapter(offset=0.25), arguments(tmp_path))
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-2] == "ROLLOUT_REPLAY=PASS points=4 same_path_maxabs=0 tol=1e-6"
    assert lines[-1].startswith("ATTN_WINDOW=PASS records=4 rowsum_err=")
    assert 0 <= float(lines[-1].split("rowsum_err=")[1].split()[0]) <= FB.ROWSUM_TOL
    assert sum(line.startswith("ROLLOUT_REPLAY_RECORD") for line in lines) == 4      # 首条是运行型的起跑判据


def bad_rowsum(probs):
    probs["action"][0, 0, 0, 0, 0, 0] += 0.01


def bad_text_rowsum(probs):
    probs["text"][0, 0, 0, 0, 0] += 0.01


def bad_prefix_rowsum(probs):
    probs["prefix"][0, 0, 0, 0, 0] += 0.01


def non_finite(probs):
    probs["prefix"][0, 0, 0, 0, 0] = np.nan


def action_padding_key(probs):
    # 基线第 3 个记忆位是 padding；从第 0 位挪质量，行和不变，只有 padding 判据能抓到
    probs["action"][..., 3] = 1e-3
    probs["action"][..., 0] -= 1e-3


def text_padding_query(probs):
    probs["text"][:, :, :, TXT - 1, 0] = 1e-3


def prefix_padding_query(probs):
    probs["prefix"][:, :, :, 3, 0] = 1e-3


@pytest.mark.parametrize("hook, reason", [(bad_rowsum, "rowsum"), (bad_text_rowsum, "rowsum"), (bad_prefix_rowsum, "rowsum"),
                                          (non_finite, "probs_prefix_non_finite"), (action_padding_key, "action_padding_key"),
                                          (text_padding_query, "text_padding_query"), (prefix_padding_query, "prefix_padding_query")])
def test_invalid_attention_stops_without_writing(tmp_path, monkeypatch, capsys, hook, reason):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    write_windows(tmp_path, [entry(G=1)])
    with pytest.raises(SystemExit) as stopped:
        FB.run(policy, FakeAdapter(hook=hook), arguments(tmp_path))
    assert stopped.value.code == 1
    output = capsys.readouterr().out
    assert "ATTN_WINDOW=FAIL" in output and f"reason={reason} " in output and "ATTN_WINDOW=PASS" not in output
    assert not (tmp_path / "offline" / "ep").exists()


def test_non_finite_x0_stops(tmp_path, monkeypatch, capsys):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    write_windows(tmp_path, [entry(G=1)])
    with pytest.raises(SystemExit) as stopped:
        FB.run(policy, FakeAdapter(offset=float("nan")), arguments(tmp_path))
    assert stopped.value.code == 1 and "reason=x_0_non_finite" in capsys.readouterr().out


def test_denoise_step_mismatch_stops(tmp_path, monkeypatch):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    write_windows(tmp_path, [entry(G=1)])
    with pytest.raises(RuntimeError, match="实际步数"):
        FB.run(policy, FakeAdapter(steps=9), arguments(tmp_path))


def test_missing_record_or_trace_field_stops(tmp_path, monkeypatch):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    write_windows(tmp_path, [entry(G=4)])
    with pytest.raises(SystemExit, match="infer_seq=2"):         # 窗口 1～4，轨迹只有 0、1
        FB.run(policy, FakeAdapter(), arguments(tmp_path))
    (tmp_path / "g0" / "attn" / "ep" / "rollout_0001.npz").unlink()
    write_windows(tmp_path, [entry(G=1)])
    with pytest.raises(SystemExit, match="rollout 记录应恰有 1 份"):
        FB.run(policy, FakeAdapter(), arguments(tmp_path))


def test_missing_exec_start_idx_is_not_filled(tmp_path, monkeypatch):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    path = next((tmp_path / "g0" / "run").rglob("step_trace.jsonl"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row.pop("exec_start_idx")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_windows(tmp_path, [entry(G=1)])
    with pytest.raises(SystemExit, match="exec_start_idx"):
        FB.run(policy, FakeAdapter(), arguments(tmp_path))


def test_episode_not_accepted_stops(tmp_path, monkeypatch):
    policy = write_episode(tmp_path, monkeypatch, infers=2)
    write_windows(tmp_path, [entry(G=1, nonce="别的")])
    with pytest.raises(SystemExit, match="恰有 1 条"):
        FB.run(policy, FakeAdapter(), arguments(tmp_path))
    write_windows(tmp_path, [entry(key="ButtonUnmaskSwap:7", G=1)])      # nonce 对但局号不符
    with pytest.raises(SystemExit, match="恰有 1 条"):
        FB.run(policy, FakeAdapter(), arguments(tmp_path))


def test_parser_repeatable_roots_and_shard():
    argv = ["--windows", "w.json", "--run-root", "a", "--run-root", "b", "--attn-root", "c", "--attn-root", "d",
            "--output", "o", "--ckpt", "k", "--config", "cfg", "--shard", "g1"]
    args = FB.build_parser().parse_args(argv)
    assert [str(path) for path in args.run_root] == ["a", "b"] and [str(path) for path in args.attn_root] == ["c", "d"]
    assert args.shard == "g1" and args.seed == 42
    with pytest.raises(SystemExit):
        FB.build_parser().parse_args(argv[:-1] + ["g2"])


@pytest.fixture(scope="module", autouse=True)
def _verdict_line(request):
    """pytest 收尾时打印判定行，排在 -q 摘要之后。"""
    session = request.session

    class _Plugin:
        def pytest_unconfigure(self):
            verdict = "PASS" if session.testsfailed == 0 else "FAIL"
            print(f"W3_CPU={verdict} tests={session.testscollected}")

    request.config.pluginmanager.register(_Plugin(), "w3-cpu-verdict")
    yield
