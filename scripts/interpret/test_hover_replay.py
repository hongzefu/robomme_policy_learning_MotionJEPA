"""示教回放入口的 CPU 测试：桩 policy、桩适配器和小合成 H5，不加载权重；前缀路径用假模型跑真实的 HistoryBlock 展开。"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from mme_vla_suite.datastore.motion_adapter import STAGE2_SPLIT_READOUT  # noqa: E402
from mme_vla_suite.models.integration import history_observation as ho  # noqa: E402
from mme_vla_suite.models.integration import utils as hu  # noqa: E402
from mme_vla_suite.policies import policy_config  # noqa: E402
from mme_vla_suite.shared.sampling import memory_order, pad_times  # noqa: E402
from mme_vla_suite.training import config as training_config  # noqa: E402
from openpi.models import model as openpi_model  # noqa: E402
from scripts.interpret import attn_adapter as A  # noqa: E402
from scripts.interpret import hover_replay as HR  # noqa: E402
from scripts.interpret import test_rollout_hidden as THR  # noqa: E402
from scripts.interpret.offline_query import pack_observation  # noqa: E402

FRAME = 4
BUDGET = 4              # 4 帧记忆，每帧 1 个 token
WINDOW_BUDGET = 3       # 3 个 motion 窗口，每窗 4 个 token
MOTION_TOKENS = WINDOW_BUDGET * 4
IMG_TOKENS, TXT = 4, 4
W = 3


class Buffer:
    motion_budget = WINDOW_BUDGET

    def __init__(self, motion):
        self.motion = motion

    def get_frame_sampling_indices(self, step_idx, budget, token_per_image):
        return sorted(set(np.linspace(0, step_idx, min(budget, step_idx + 1), dtype=np.int32).tolist()))

    def visible_motion_frames(self, step_idx):
        if not self.motion:
            raise AssertionError("基线不应读 motion 窗口")
        return list(range(0, step_idx - 32 + 1, 32))


class StubPolicy:
    """调用顺序与 MME_VLA_Policy 一致；观测里的 state 编码 step_idx 和批大小无关的帧内容。"""

    def __init__(self, motion):
        self.motion_enabled = motion
        self.config = SimpleNamespace(budget=BUDGET, token_per_image=1, num_views=1,
                                      motion={"readout": STAGE2_SPLIT_READOUT} if motion else None)
        self.reset_calls = 0
        self.reset()

    def reset(self):
        self.reset_calls += 1
        self.step_idx = -1
        self.mem_buffer = Buffer(self.motion_enabled)
        self.batches = []
        self.elements = []
        self.images_seen = []
        self.states_seen = []

    def add_buffer(self, obs):
        assert obs["exec_start_idx"] == 0
        self.batches.append(len(obs["images"]))
        self.images_seen.append(obs["images"])
        self.states_seen.append(obs["state"])
        self.step_idx += len(obs["images"])

    def _prepare_history(self, element):
        self.elements.append(dict(element))
        return element

    def _input_transform(self, element):
        frames = self.mem_buffer.get_frame_sampling_indices(self.step_idx, BUDGET, 1)
        static_mask = np.zeros(BUDGET, bool)
        static_mask[:len(frames)] = True
        inputs = {"image": {"base_0_rgb": element["observation/image"].astype(np.float32),
                            "left_wrist_0_rgb": element["observation/wrist_image"].astype(np.float32)},
                  "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_},
                  "state": np.full(8, self.step_idx, np.float32),
                  "tokenized_prompt": np.ones(TXT, np.int32),
                  "tokenized_prompt_mask": np.asarray([True, True, True, False]),
                  "static_image_emb": np.full((BUDGET, W), self.step_idx, np.float32),
                  "static_pos_emb": np.ones((BUDGET, W), np.float32),
                  "static_state_emb": np.ones((BUDGET, W), np.float32),
                  "static_mask": static_mask}
        if self.motion_enabled:
            windows = self.mem_buffer.visible_motion_frames(self.step_idx)
            motion_mask = np.zeros(MOTION_TOKENS, bool)
            motion_mask[:len(windows) * 4] = True
            inputs.update(motion_emb=np.ones((MOTION_TOKENS, W), np.float32), motion_pos=np.ones((MOTION_TOKENS, W), np.float32),
                          motion_mask=motion_mask,
                          mem_order=memory_order(pad_times(frames, BUDGET), 1,
                                                 np.repeat(pad_times(windows, WINDOW_BUDGET), 4)))
        return inputs


class StubAdapter:
    AH, AD = 2, 2

    def __init__(self, motion, *, error=0.0):
        self.motion = motion
        self.error = error
        self.masks = []
        self.full_calls = []

    def layout(self, observation):
        mem = BUDGET + (MOTION_TOKENS if self.motion else 0)
        return {"P": mem + IMG_TOKENS + TXT, "MEM": mem, "IMG": IMG_TOKENS, "TXT": TXT}

    def _hidden(self, observation, masks, error=0.0):
        step = float(observation.state[0, 0])
        hidden = np.tile((step + error) * (1 + np.arange(masks.shape[0], dtype=np.float32))[None, :, None], (19, 1, W))
        return {"hidden_prefix": jnp.asarray(hidden, jnp.float32), "group_tokens": jnp.asarray(masks.sum(axis=1), jnp.int32)}

    def run_prefix(self, observation, masks):
        self.masks.append(np.asarray(masks))
        return self._hidden(observation, masks)

    def run(self, rng, observation, *, noise, num_steps, out_f16, capture_hidden, group_masks):
        self.full_calls.append(dict(num_steps=num_steps, out_f16=out_f16, capture_hidden=capture_hidden))
        return None, self._hidden(observation, group_masks, self.error)


def write_h5(path, steps):
    with h5py.File(path, "w") as h5_file:
        for k, count in steps.items():
            episode = h5_file.create_group(f"episode_{k}")
            for i in range(count):
                obs = episode.create_group(f"timestep_{i}/obs")
                obs["front_rgb"] = np.full((FRAME, FRAME, 3), i % 251, np.uint8)
                obs["wrist_rgb"] = np.full((FRAME, FRAME, 3), (i + 1) % 251, np.uint8)
                obs["joint_state"] = np.arange(7, dtype=np.float32) + i
                obs["gripper_state"] = np.asarray([0.5 + i, -7.0], np.float32)


def make_entry(ts, k=0):
    return {"episode": k, "prompt": f"press {k}", "points": [{"t": t, "c": 1} for t in ts]}


@pytest.fixture
def h5_path(tmp_path):
    path = tmp_path / "demo.h5"
    write_h5(path, {0: 80, 1: 40, 2: 5})
    return path


@pytest.mark.parametrize("last_t,batch,want", [
    (32, 8, [(0, 1), (1, 9), (9, 17), (17, 25), (25, 33)]),
    (32, 16, [(0, 1), (1, 17), (17, 33)]),
    (0, 8, [(0, 1)]),
])
def test_feed_batches_land_on_multiples(last_t, batch, want):
    assert HR.feed_batches(last_t, batch) == want


def test_replay_batches_stop_at_last_point(h5_path):
    policy = StubPolicy(False)
    with h5py.File(h5_path) as h5_file:
        result = HR.replay_episode(policy, StubAdapter(False), h5_file["episode_0"], make_entry([16, 32]), 8)
    assert policy.batches == [1, 8, 8, 8, 8]
    assert policy.step_idx == 32                 # 第 33 帧之后没有读
    assert result["t"].tolist() == [16, 32]
    assert result["t"].dtype == np.int64
    assert result["hidden_prefix"].shape == (2, 19, 5, W) and result["hidden_prefix"].dtype == np.float32
    assert result["group_tokens"].shape == (2, 5) and result["group_tokens"].dtype == np.int32
    assert result["hidden_prefix"][0, 0, 0, 0] == 16     # 取点时 step_idx 就是 t
    assert result["hidden_prefix"][1, 0, 0, 0] == 32


def test_replay_eval_pace(h5_path):
    policy = StubPolicy(False)
    with h5py.File(h5_path) as h5_file:
        HR.replay_episode(policy, StubAdapter(False), h5_file["episode_0"], make_entry([16, 32]), 16)
    assert policy.batches == [1, 16, 16]


def test_point_off_batch_boundary_rejected(h5_path):
    with h5py.File(h5_path) as h5_file, pytest.raises(ValueError, match="批边界"):
        HR.replay_episode(StubPolicy(False), StubAdapter(False), h5_file["episode_0"], make_entry([8, 16]), 16)


def test_state_and_element_assembly(h5_path):
    policy = StubPolicy(False)
    with h5py.File(h5_path) as h5_file:
        HR.replay_episode(policy, StubAdapter(False), h5_file["episode_0"], make_entry([8]), 8)
    images = np.concatenate(policy.images_seen)
    states = np.concatenate(policy.states_seen)
    assert images.shape == (9, 1, FRAME, FRAME, 3) and images.dtype == np.uint8
    assert states.dtype == np.float32
    np.testing.assert_array_equal(states[3], np.asarray([3, 4, 5, 6, 7, 8, 9, 3.5], np.float32))   # 关节 7 维 + gripper_state[:1]
    assert (images[3] == 3).all()
    element = policy.elements[0]
    assert set(element) == {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
    assert (element["observation/image"] == 8).all() and element["observation/image"].shape == (FRAME, FRAME, 3)
    assert (element["observation/wrist_image"] == 9).all()
    np.testing.assert_array_equal(element["observation/state"], states[8])
    assert element["prompt"] == "press 0"


def window_tokens(observation_order, mask):
    return sorted((observation_order[np.flatnonzero(mask)] - BUDGET).tolist())


def test_masks_latest_window_motion(h5_path):
    policy, adapter = StubPolicy(True), StubAdapter(True)
    seen = []

    def on_point(t, observation, masks, record, hidden):
        seen.append((t, np.asarray(observation.mem_order[0]), masks, record))

    with h5py.File(h5_path) as h5_file:
        HR.replay_episode(policy, adapter, h5_file["episode_0"], make_entry([16, 40, 72]), 8, on_point=on_point)
    p = adapter.layout(None)["P"]
    by_t = {item[0]: item for item in seen}
    # t=16：还没有窗口；t=40：只有窗口 0；t=72：窗口 0、32，最新的是 32，对应 token 4..7
    assert not by_t[16][2][4].any()
    assert window_tokens(by_t[40][1], by_t[40][2][4]) == [0, 1, 2, 3]
    assert window_tokens(by_t[72][1], by_t[72][2][4]) == [4, 5, 6, 7]
    for _, _, masks, record in seen:
        assert masks.shape == (5, p) and masks.dtype == bool
        assert record["motion_times"].shape == (MOTION_TOKENS,)
    assert by_t[72][2][1].sum() == 8        # motion 组是两个窗口全部 8 个 token


def test_masks_baseline_motion_columns_empty(h5_path):
    policy, adapter = StubPolicy(False), StubAdapter(False)
    with h5py.File(h5_path) as h5_file:
        result = HR.replay_episode(policy, adapter, h5_file["episode_0"], make_entry([72]), 8)
    masks = adapter.masks[0]
    assert masks.shape == (5, adapter.layout(None)["P"])
    assert not masks[1].any() and not masks[4].any()
    assert result["group_tokens"][0, 4] == 0
    assert result["group_tokens"][0, 0] > 0 and result["group_tokens"][0, 2] == IMG_TOKENS
    assert result["group_tokens"][0, 3] == 3    # 文字有效 token


def run_kwargs(tmp_path):
    return dict(output=tmp_path / "out", label="ctx", ckpt="/ckpt/79999", config="cfg")


def make_points():
    return {"schema_version": 1, "episodes": {"0": make_entry([16, 32], 0), "1": make_entry([8], 1), "2": make_entry([], 2)}}


def test_run_writes_keys_and_verdict(tmp_path, h5_path, capsys):
    points = make_points()
    with h5py.File(h5_path) as h5_file:
        assert HR.run_episodes(StubPolicy(False), StubAdapter(False), h5_file, points, **run_kwargs(tmp_path)) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "REPLAY_RUN=PASS model=ctx points=3 expected=3"
    assert any(line.startswith("REPLAY_EPISODE episode=0 points=2 seconds=") and "vision_seconds=" in line
               and "prefix_seconds=" in line for line in out)
    directory = tmp_path / "out" / "ctx"
    assert sorted(path.name for path in directory.iterdir()) == ["episode_0.npz", "episode_1.npz"]     # 没有点的局不写
    with np.load(directory / "episode_0.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"t", "hidden_prefix", "group_tokens", "episode", "label", "fingerprint"}
        assert archive["t"].tolist() == [16, 32] and archive["t"].dtype == np.int64
        assert archive["hidden_prefix"].shape == (2, 19, 5, W) and archive["hidden_prefix"].dtype == np.float32
        assert archive["group_tokens"].dtype == np.int32
        assert int(archive["episode"]) == 0 and str(archive["label"]) == "ctx"
        assert str(archive["fingerprint"]) == HR.episode_fingerprint(points["episodes"]["0"], "/ckpt/79999", "cfg")


def test_run_episode_subset_expected(tmp_path, h5_path, capsys):
    with h5py.File(h5_path) as h5_file:
        assert HR.run_episodes(StubPolicy(False), StubAdapter(False), h5_file, make_points(), episodes=[1],
                               **run_kwargs(tmp_path)) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "REPLAY_RUN=PASS model=ctx points=1 expected=1"
    assert [path.name for path in (tmp_path / "out" / "ctx").iterdir()] == ["episode_1.npz"]


def test_fingerprint_skip_and_mismatch(tmp_path, h5_path, capsys):
    points = make_points()
    kwargs = run_kwargs(tmp_path)
    with h5py.File(h5_path) as h5_file:
        HR.run_episodes(StubPolicy(False), StubAdapter(False), h5_file, points, **kwargs)
        path = tmp_path / "out" / "ctx" / "episode_0.npz"
        before = path.read_bytes()
        capsys.readouterr()
        policy = StubPolicy(False)
        assert HR.run_episodes(policy, StubAdapter(False), h5_file, points, **kwargs) == 0     # 再跑一遍全部跳过
        out = capsys.readouterr().out
        assert "REPLAY_EPISODE" not in out and "REPLAY_RUN=PASS model=ctx points=3 expected=3" in out
        assert policy.batches == []
        changed = json.loads(json.dumps(points))
        changed["episodes"]["0"]["points"][0]["c"] = 2
        with pytest.raises(SystemExit, match="不覆盖"):
            HR.run_episodes(StubPolicy(False), StubAdapter(False), h5_file, changed, **kwargs)
        with pytest.raises(SystemExit, match="不覆盖"):
            HR.run_episodes(StubPolicy(False), StubAdapter(False), h5_file, points, **{**kwargs, "ckpt": "/other"})
    assert path.read_bytes() == before


def test_verdict_fail_exit_code(tmp_path, h5_path, capsys):
    """已有文件的点数和 points 文件对不上时 REPLAY_RUN=FAIL。"""
    points = {"episodes": {"0": make_entry([16, 32], 0)}}
    kwargs = run_kwargs(tmp_path)
    path = tmp_path / "out" / "ctx" / "episode_0.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, t=np.asarray([16]), fingerprint=np.str_(HR.episode_fingerprint(points["episodes"]["0"], "/ckpt/79999", "cfg")))
    with h5py.File(h5_path) as h5_file:
        assert HR.run_episodes(StubPolicy(False), StubAdapter(False), h5_file, points, **kwargs) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "REPLAY_RUN=FAIL model=ctx points=1 expected=2"


def test_temporary_file_replaced_only_on_success(tmp_path, h5_path, monkeypatch):
    points = {"episodes": {"0": make_entry([8], 0)}}
    directory = tmp_path / "out" / "ctx"
    real_replace = os.replace
    seen = []

    def spy(source, target):
        seen.append((str(source), str(target)))
        assert np.load(source, allow_pickle=False)["t"].tolist() == [8]     # 换入前内容已完整
        assert not os.path.exists(target)
        real_replace(source, target)

    monkeypatch.setattr(HR.os, "replace", spy)
    with h5py.File(h5_path) as h5_file:
        HR.run_episodes(StubPolicy(False), StubAdapter(False), h5_file, points, **run_kwargs(tmp_path))
    assert len(seen) == 1 and seen[0][1] == str(directory / "episode_0.npz")
    assert ".tmp.npz" in seen[0][0]
    assert [path.name for path in directory.iterdir()] == ["episode_0.npz"]


def write_rollout(path, policy, adapter_motion, h5_episode, t, *, mutate=None):
    """用回放观测造一份和 RolloutRecorder.build_record 同写法的记录。"""
    captured = {}

    def on_point(step, observation, masks, record, hidden):
        captured.update(observation=observation, record=record)

    HR.replay_episode(policy, StubAdapter(adapter_motion), h5_episode, make_entry([t]), 8, on_point=on_point)
    observation, record = captured["observation"], captured["record"]
    frame_times = record["frame_times"][record["frame_times"] != HR.MEM_ORDER_SENTINEL]
    motion_times = np.zeros(0, np.int32)
    if adapter_motion:
        motion_times = np.where(np.asarray(observation.motion_mask).reshape(-1), record["motion_times"], -1).astype(np.int32)
    saved = pack_observation(observation)
    saved.update(step_idx=np.int64(t), frame_times=np.asarray(frame_times, np.int64), motion_times=motion_times)
    if mutate is not None:
        mutate(saved)
    np.savez(path, **saved)


@pytest.mark.parametrize("motion", [False, True])
def test_layout_same_and_each_difference(tmp_path, h5_path, motion):
    with h5py.File(h5_path) as h5_file:
        episode = h5_file["episode_0"]
        path = tmp_path / "rollout_0000.npz"
        write_rollout(path, StubPolicy(motion), motion, episode, 64)
        captured = {}

        def on_point(step, observation, masks, record, hidden):
            captured.update(observation=observation, record=record)

        HR.replay_episode(StubPolicy(motion), StubAdapter(motion), episode, make_entry([64]), 16, on_point=on_point)
        assert HR.layout_differences(captured["observation"], captured["record"], path) == []
        assert list(HR.rollout_steps(tmp_path)) == [64]

        def shift_frames(saved):
            saved["frame_times"] = saved["frame_times"] + 1

        write_rollout(path, StubPolicy(motion), motion, episode, 64, mutate=shift_frames)
        assert HR.layout_differences(captured["observation"], captured["record"], path) == ["frame_times"]

        def reshape_schema(saved):
            schema = json.loads(str(saved["obs_schema"]))
            schema["items"]["static_mask"]["dtype"] = "int32"
            saved["obs/static_mask"] = saved["obs/static_mask"].astype(np.int32)
            saved["obs_schema"] = np.str_(json.dumps(schema, ensure_ascii=False))

        write_rollout(path, StubPolicy(motion), motion, episode, 64, mutate=reshape_schema)
        assert HR.layout_differences(captured["observation"], captured["record"], path) == ["obs_schema"]
        if motion:
            def swap_order(saved):
                order = saved["obs/mem_order"].copy()
                order[0, [0, 1]] = order[0, [1, 0]]
                saved["obs/mem_order"] = order

            write_rollout(path, StubPolicy(motion), motion, episode, 64, mutate=swap_order)
            assert HR.layout_differences(captured["observation"], captured["record"], path) == ["mem_order"]

            def shift_motion(saved):
                saved["motion_times"] = saved["motion_times"] + 1

            write_rollout(path, StubPolicy(motion), motion, episode, 64, mutate=shift_motion)
            assert HR.layout_differences(captured["observation"], captured["record"], path) == ["motion_times"]


def test_precision_readings_definition():
    # 3 个点，每组 19 层、2 维
    prefix = np.zeros((3, 19, 2, 2), np.float32)
    prefix[1, :, 0] = 1.0
    prefix[2, :, 0] = 3.0
    full = prefix.copy()
    full[0, 5, 0, 1] += 0.1
    tokens = np.asarray([[4, 0], [4, 0], [4, 0]])
    readings = HR.precision_readings(prefix, full, tokens, names=("甲", "乙"))
    assert len(readings) == 1                   # 空组不报
    name, ratio, error = readings[0]
    assert name == "甲"
    np.testing.assert_allclose(error, 0.1, rtol=1e-5)
    np.testing.assert_allclose(ratio, 1.5 / 0.1, rtol=1e-5)    # 相邻距离 1 和 2 的中位数是 1.5


@pytest.mark.parametrize("motion", [False, True])
def test_check_pass_and_lines(tmp_path, h5_path, capsys, motion):
    entry = make_entry([40, 48, 56, 64, 72])
    attn = tmp_path / "attn"
    attn.mkdir()
    adapter = StubAdapter(motion, error=0.0)
    with h5py.File(h5_path) as h5_file:
        write_rollout(attn / "rollout_0003.npz", StubPolicy(motion), motion, h5_file["episode_0"], 64)
        policy = StubPolicy(motion)
        assert HR.check_episode(policy, adapter, h5_file["episode_0"], entry, attn, seed=42) == 0
    out = capsys.readouterr().out
    assert "REPLAY_PARITY_LAYOUT compared=1" in out
    assert "REPLAY_PARITY=PASS layout_mismatch=0 points=2 maxabs=0" in out      # 16 的倍数：48、64
    groups = [line.split("group=")[1].split()[0] for line in out.splitlines() if line.startswith("HIDDEN_PRECISION=")]
    assert groups == (list(HR.REPLAY_GROUPS) if motion else ["历史视觉", "当前视觉", "文字"])
    assert all(line.startswith("HIDDEN_PRECISION=PASS") for line in out.splitlines() if "group=" in line)
    assert len(adapter.full_calls) == 5         # 每个取点一次全路径
    assert adapter.full_calls[0] == dict(num_steps=10, out_f16=False, capture_hidden=True)
    assert policy.reset_calls == 3              # 构造 1 次，两遍回放各 1 次
    assert policy.batches == [1, 16, 16, 16, 16]


def test_check_precision_fail_exit_code(tmp_path, h5_path, capsys):
    attn = tmp_path / "attn"
    attn.mkdir()
    with h5py.File(h5_path) as h5_file:
        code = HR.check_episode(StubPolicy(False), StubAdapter(False, error=10.0), h5_file["episode_0"],
                                make_entry([40, 48, 56, 64]), attn, seed=42)
    out = capsys.readouterr().out
    assert code == 1
    assert "REPLAY_PARITY=PASS" in out and "HIDDEN_PRECISION=FAIL group=历史视觉" in out


def test_check_batch_difference_fail(tmp_path, h5_path, capsys):
    """第二遍回放的隐状态差超过容差时 REPLAY_PARITY=FAIL。"""
    attn = tmp_path / "attn"
    attn.mkdir()
    policy = StubPolicy(False)

    class Drifting(StubAdapter):
        def run_prefix(self, observation, masks):
            return self._hidden(observation, masks, 0.002 * (policy.reset_calls - 2))

    with h5py.File(h5_path) as h5_file:
        code = HR.check_episode(policy, Drifting(False), h5_file["episode_0"], make_entry([48, 64]), attn, seed=42)
    out = capsys.readouterr().out
    assert code == 1
    assert "REPLAY_PARITY_LAYOUT compared=0" in out
    assert "REPLAY_PARITY=FAIL layout_mismatch=0 points=2 maxabs=" in out


def test_check_layout_mismatch_fail(tmp_path, h5_path, capsys):
    attn = tmp_path / "attn"
    attn.mkdir()

    def shift_frames(saved):
        saved["frame_times"] = saved["frame_times"] + 1

    with h5py.File(h5_path) as h5_file:
        write_rollout(attn / "rollout_0000.npz", StubPolicy(False), False, h5_file["episode_0"], 64, mutate=shift_frames)
        code = HR.check_episode(StubPolicy(False), StubAdapter(False), h5_file["episode_0"], make_entry([48, 64]), attn, seed=42)
    out = capsys.readouterr().out
    assert code == 1
    assert "REPLAY_LAYOUT_DIFF step=64 fields=frame_times" in out
    assert "REPLAY_PARITY=FAIL layout_mismatch=1 points=2 maxabs=0" in out


@dataclasses.dataclass
class StubModelConfig:
    dtype: str = "bfloat16"

    def load(self, params, remove_extra_params):
        return SimpleNamespace(config=self)


@pytest.mark.parametrize("contract", [None, "contracts/runtime-mot.json"])
def test_eval_target_contract_reaches_create_trained_policy(tmp_path, h5_path, monkeypatch, contract):
    calls = []

    def create_trained_policy(train_config, checkpoint_dir, **kwargs):
        calls.append(kwargs)
        policy = StubPolicy(False)
        policy._model = SimpleNamespace(config=StubModelConfig())
        return policy

    monkeypatch.setattr(policy_config, "create_trained_policy", create_trained_policy)
    monkeypatch.setattr(training_config, "get_config", lambda name: name)
    monkeypatch.setattr(openpi_model, "restore_params", lambda path, dtype: {})
    monkeypatch.setattr(A, "AttnAdapter", lambda shim: StubAdapter(False))
    points = tmp_path / "points.json"
    points.write_text(json.dumps({"episodes": {}}), encoding="utf-8")
    argv = ["run", "--points", str(points), "--h5", str(h5_path), "--ckpt", str(tmp_path), "--config", "cfg",
            "--label", "ctx", "--output", str(tmp_path / "out"), "--stage2-python", "/py/stage2"]
    if contract is not None:
        argv += ["--eval-target-contract", contract]
    assert HR.main(argv) == 0
    assert calls == [{"seed": 42, "motion_stub": False, "stage2_python": "/py/stage2",
                      "eval_target_contract": None if contract is None else Path(contract)}]


@pytest.fixture(scope="module")
def prefix_env():
    patch = pytest.MonkeyPatch()
    original = hu.get_config
    fake = {name: THR._fake_config(name) for name in ("fake_prefix", "fake_action")}
    patch.setattr(hu, "get_config", lambda v: fake.get(v) or original(v))
    patch.setattr(ho, "preprocess_observation", lambda _rng, o, **_: o)
    obs = THR._make_obs()
    adapter = A.AttnAdapter(A._ModelShim(THR._FakeModel(obs)), require_flags=False)
    yield SimpleNamespace(adapter=adapter, obs=obs)
    patch.undo()


def test_run_prefix_matches_capture_hidden(prefix_env):
    adapter, obs = prefix_env.adapter, prefix_env.obs
    noise = jax.random.normal(jax.random.key(1), (1, THR.AH, THR.AD), jnp.float32)
    full = adapter.run(jax.random.key(0), obs, noise=noise, num_steps=THR.STEPS, out_f16=False,
                       capture_hidden=True, group_masks=THR.GROUPS)[-1]
    prefix = adapter.run_prefix(obs, THR.GROUPS)
    assert set(prefix) == {"hidden_prefix", "group_tokens"}
    assert prefix["hidden_prefix"].shape == (19, THR.GROUPS.shape[0], THR.W) and prefix["hidden_prefix"].dtype == jnp.float32
    assert prefix["group_tokens"].dtype == jnp.int32
    np.testing.assert_allclose(np.asarray(prefix["hidden_prefix"]), np.asarray(full["hidden_prefix"]), atol=1e-4)
    np.testing.assert_array_equal(np.asarray(prefix["group_tokens"]), np.asarray(full["group_tokens"]))


def test_run_prefix_rejects_batch_two(prefix_env):
    obs = jax.tree.map(lambda x: jnp.repeat(x, 2, axis=0) if x.ndim else x, prefix_env.obs)
    with pytest.raises(ValueError, match="batch=1"):
        prefix_env.adapter.run_prefix(obs, THR.GROUPS)
