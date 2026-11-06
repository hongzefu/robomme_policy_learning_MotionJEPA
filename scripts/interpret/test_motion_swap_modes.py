"""motion_swap_exact、motion_shift 两个替换模式的 CPU 测试：假 client、假策略，不导入 robomme。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import manifest as M
from scripts.interpret import run_local as local
from scripts.interpret import serve_policy_attn as SA

FRAMES = np.zeros((33, 1), np.uint8)
DIM = 6


class Inner:
    """每个起点的原编码互不相同，且复用同一块缓冲，用来查返回值与内部状态有没有别名。"""

    def __init__(self):
        self.calls = []
        self.buffer = np.empty(DIM, np.float32)

    def __call__(self, frames, start=None):
        self.calls.append(start)
        self.buffer[:] = original(start or 0)
        return self.buffer


def original(start):
    return np.full(DIM, 1000 + start, np.float32)


def donor_value(start):
    return np.full(DIM, -start - 1, np.float32)


def make_donor(starts=(0, 16, 48)):
    return {"name": "ButtonUnmaskSwap:8", "start_frames": np.asarray(starts, np.int32),
            "embs": np.stack([donor_value(s) for s in starts])}


def run(client, starts):
    return [client(FRAMES, start).copy() for start in starts]


# ―― motion_swap_exact ――

def test_exact_hit_replaces_and_miss_keeps_original():
    client = SA.MotionContentClient(Inner(), mode="swap_exact", donor=make_donor())
    client.begin_episode("e")
    out = run(client, [0, 16, 32, 48, 64])
    for result, expected in zip(out, [donor_value(0), donor_value(16), original(32), donor_value(48), original(64)]):
        np.testing.assert_array_equal(result, expected)
    assert client.replaced == 3
    assert client.replace_log == [[0, 0], [16, 16], [48, 48]]
    assert client.inner.calls == [0, 16, 32, 48, 64]
    summary = client.summary()
    assert summary["mode"] == "swap_exact" and summary["donor"] == "ButtonUnmaskSwap:8"
    assert summary["n_windows"] == 5 and summary["n_replaced"] == 3


def test_exact_differs_from_swap_only_on_miss():
    donor = make_donor()
    fallback = SA.MotionContentClient(Inner(), mode="swap", donor=donor)
    exact = SA.MotionContentClient(Inner(), mode="swap_exact", donor=donor)
    # 起点 32 不在供体里：swap 就近回退到 16 并计数；exact 保留原编码且不计数
    np.testing.assert_array_equal(fallback(FRAMES, 32), donor_value(16))
    np.testing.assert_array_equal(exact(FRAMES, 32), original(32))
    assert fallback.replaced == 1 and fallback.replace_log == [[32, 16]]
    assert exact.replaced == 0 and exact.replace_log == []


def test_exact_needs_donor_start_and_matching_shape():
    with pytest.raises(ValueError, match="须给供体"):
        SA.MotionContentClient(Inner(), mode="swap_exact")
    client = SA.MotionContentClient(Inner(), mode="swap_exact", donor=make_donor())
    with pytest.raises(RuntimeError, match="motion_swap_exact 需要起点帧号"):
        client(FRAMES)
    wrong = {"name": "x", "start_frames": np.asarray([0], np.int32), "embs": np.zeros((1, DIM + 1), np.float32)}
    with pytest.raises(RuntimeError, match="形状"):
        SA.MotionContentClient(Inner(), mode="swap_exact", donor=wrong)(FRAMES, 0)


def test_exact_record_keeps_original_and_donor_starts(tmp_path):
    client = SA.MotionContentClient(Inner(), mode="swap_exact", donor=make_donor(), record_dir=tmp_path)
    client.begin_episode("rec")
    run(client, [16, 32])
    client.flush()
    with np.load(tmp_path / "rec.npz") as record:
        np.testing.assert_array_equal(record["embs"], np.stack([original(16), original(32)]))
        assert record["donor_start_frames"].tolist() == [16, -1]
        assert str(record["mode"]) == "swap_exact" and str(record["donor"]) == "ButtonUnmaskSwap:8"


# ―― motion_shift ――

def test_shift_first_kept_then_previous_original():
    client = SA.MotionContentClient(Inner(), mode="shift")
    client.begin_episode("e")
    out = run(client, [0, 16, 32, 48])
    expected = [original(0), original(0), original(16), original(32)]
    for result, value in zip(out, expected):
        np.testing.assert_array_equal(result, value)
    assert client.replaced == 3
    assert client.replace_log == [[16, 0], [32, 16], [48, 32]]
    assert client.summary()["n_replaced"] == 3 and client.summary()["donor"] is None


def test_shift_repeated_call_is_stable():
    client = SA.MotionContentClient(Inner(), mode="shift")
    client.begin_episode("e")
    first = run(client, [0, 16, 32])
    again = run(client, [0, 16, 32, 16])
    for result, value in zip(again, first + [first[1]]):
        np.testing.assert_array_equal(result, value)
    # 重复调用不推进「上一个窗口」：新窗口 48 仍取 32 的原编码
    np.testing.assert_array_equal(client(FRAMES, 48), original(32))
    # 首窗重复调用不算替换；其余每次实际返回替换值的调用都计数
    assert client.replace_log == [[16, 0], [32, 16], [16, 0], [32, 16], [16, 0], [48, 32]]
    assert client.replaced == 6


def test_shift_output_never_aliases_inner_buffer():
    inner = Inner()
    client = SA.MotionContentClient(inner, mode="shift")
    client.begin_episode("e")
    held = client(FRAMES, 0)
    assert held is not inner.buffer
    client(FRAMES, 16)
    np.testing.assert_array_equal(held, original(0))
    # 改调用方拿到的数组不污染内部状态
    held[:] = -7
    np.testing.assert_array_equal(client(FRAMES, 32), original(16))
    np.testing.assert_array_equal(client(FRAMES, 0), original(0))


def test_shift_new_episode_clears_state():
    client = SA.MotionContentClient(Inner(), mode="shift")
    client.begin_episode("a")
    run(client, [0, 16, 32])
    assert client.replaced == 2
    client.begin_episode("b")
    assert client.replaced == 0 and client.replace_log == []
    # 新局首窗保留，且取不到上一局的窗口
    np.testing.assert_array_equal(client(FRAMES, 16), original(16))
    np.testing.assert_array_equal(client(FRAMES, 32), original(16))
    assert client.replace_log == [[32, 16]]


def test_shift_needs_start():
    with pytest.raises(RuntimeError, match="motion_shift 需要起点帧号"):
        SA.MotionContentClient(Inner(), mode="shift")(FRAMES)


def test_shift_record_keeps_true_original(tmp_path):
    client = SA.MotionContentClient(Inner(), mode="shift", record_dir=tmp_path)
    client.begin_episode("rec")
    run(client, [0, 16])
    client.flush()
    with np.load(tmp_path / "rec.npz") as record:
        np.testing.assert_array_equal(record["embs"], np.stack([original(0), original(16)]))
        assert str(record["mode"]) == "shift" and "donor_start_frames" not in record.files


def test_shift_on_real_memory_call_order():
    """按 framesamp_memory 的真实调用公式（demo 段起点 + exec 段起点，严格递增、不重复）核对。"""
    es, stride = 40, 16
    starts = [s for s in range(0, es, stride)] + [es + u for u in range(0, 3 * stride, stride)]
    assert starts == sorted(set(starts))
    client = SA.MotionContentClient(Inner(), mode="shift")
    client.begin_episode("e")
    out = run(client, starts)
    np.testing.assert_array_equal(out[0], original(starts[0]))
    for i in range(1, len(starts)):
        np.testing.assert_array_equal(out[i], original(starts[i - 1]))
    assert client.replaced == len(starts) - 1


# ―― 既有模式不变 ――

def test_existing_modes_unchanged():
    none = SA.MotionContentClient(inner := Inner())
    assert none(FRAMES, 16) is inner.buffer and none.replaced == 0 and none.replace_log == []
    zero = SA.MotionContentClient(Inner(), mode="zero")
    np.testing.assert_array_equal(zero(FRAMES, 16), np.zeros(DIM, np.float32))
    assert zero.replaced == 1 and zero.replace_log == [[16, -1]]
    swap = SA.MotionContentClient(Inner(), mode="swap", donor=make_donor())
    out = run(swap, [16, 32, 100])
    for result, value in zip(out, [donor_value(16), donor_value(16), donor_value(48)]):
        np.testing.assert_array_equal(result, value)
    assert swap.replaced == 3 and swap.replace_log == [[16, 16], [32, 16], [100, 48]]


@pytest.mark.parametrize("name, motion", [
    ("none", None), ("mask_motion", None), ("mask_vis", None), ("heads:L3H0", None),
    ("motion_zero", {"kind": "zero"}), ("motion_swap:/d", {"kind": "swap", "dir": "/d"}),
])
def test_parse_existing_names_unchanged(name, motion):
    assert SA.parse_ablate(name)["motion"] == motion
    assert SA.parse_ablate(name)["name"] == name


# ―― 解析 ――

def test_parse_new_modes():
    assert SA.parse_ablate("motion_swap_exact:/d")["motion"] == {"kind": "swap_exact", "dir": "/d"}
    assert SA.parse_ablate("motion_shift")["motion"] == {"kind": "shift"}
    combined = SA.parse_ablate("motion_shift+heads:L3H0")
    assert combined["motion"] == {"kind": "shift"} and combined["heads"] == ((3, 0),)
    assert combined["name"] == "motion_shift+heads:L3H0"


@pytest.mark.parametrize("name, message", [
    ("motion_swap_exact:", "motion_swap_exact: 后须给供体目录"),
    ("motion_swap_exact", "未知项"),
    ("motion_shift:/d", "未知项"),
    ("motion_swap_exact:/a+motion_swap_exact:/b", "只能出现一次"),
    ("motion_shift+motion_zero", "只能出现一次"),
    ("motion_shift+motion_swap:/d", "只能出现一次"),
    ("motion_swap:/d+motion_swap_exact:/d", "只能出现一次"),
])
def test_parse_errors(name, message):
    with pytest.raises(ValueError, match=message):
        SA.parse_ablate(name)


# ―― AttnRecorder ――

def fake_policy(motion=True):
    config = SimpleNamespace(budget=4, token_per_image=1, num_views=1, motion=SimpleNamespace(budget=3))
    return SimpleNamespace(config=config, motion_enabled=motion, reset=None, add_buffer=None, infer=None)


def write_donor(directory):
    np.savez(directory / "ButtonUnmaskSwap_ep8.npz", start_frames=np.asarray([0, 16], np.int32),
             embs=np.stack([donor_value(0), donor_value(16)]), counts=np.ones(2, np.int32), mode=np.str_("none"))


@pytest.mark.parametrize("name", ["motion_swap_exact:/d", "motion_shift"])
def test_recorder_rejects_baseline(tmp_path, name):
    with pytest.raises(SystemExit, match="需要 motion 模型"):
        SA.AttnRecorder(fake_policy(motion=False), None, tmp_path, ablate=name)


def test_recorder_donor_pairing(tmp_path):
    write_donor(tmp_path)
    with pytest.raises(SystemExit, match="必须同时给出"):
        SA.AttnRecorder(fake_policy(), None, tmp_path, ablate=f"motion_swap_exact:{tmp_path}")
    with pytest.raises(SystemExit, match="必须同时给出"):
        SA.AttnRecorder(fake_policy(), None, tmp_path, ablate="motion_shift", swap_donor="ButtonUnmaskSwap:8")
    recorder = SA.AttnRecorder(fake_policy(), None, tmp_path, ablate=f"motion_swap_exact:{tmp_path}",
                               swap_donor="ButtonUnmaskSwap:8")
    assert recorder.donor["name"] == "ButtonUnmaskSwap:8"
    assert SA.AttnRecorder(fake_policy(), None, tmp_path, ablate="motion_shift").donor is None


@pytest.mark.parametrize("kind, ablate", [("swap_exact", "motion_swap_exact:{d}"), ("shift", "motion_shift")])
def test_recorder_installs_client_with_mode(tmp_path, kind, ablate):
    write_donor(tmp_path)
    policy = fake_policy()
    policy._motion_client = Inner()
    policy.mem_buffer = SimpleNamespace(motion_enc=policy._motion_client)
    recorder = SA.AttnRecorder(policy, None, tmp_path, ablate=ablate.format(d=tmp_path),
                               swap_donor="ButtonUnmaskSwap:8" if kind == "swap_exact" else None)
    recorder._install_motion_client()
    assert policy._motion_client is recorder.motion_client is policy.mem_buffer.motion_enc
    assert recorder.motion_client.mode == kind


# ―― run_local ――

def base_spec(tmp_path, **changes):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": []}))
    values = dict(repo="/repo", policy_python="/policy/python", sim_python="/sim/python",
                  stage2_python="/stage2/python", checkpoint="/ckpt", policy_config="cfg",
                  model_kind="stage2_pool4_concat", cuda="0", attn_out="/out/attn", run_root="/out/run",
                  manifest=str(manifest), xla_mem_fraction=0.5, model_seed=42, run_identity_sha256="a" * 64)
    return values | changes


def load(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return local.load_spec(path)


def test_spec_exact_requires_swap_map(tmp_path):
    with pytest.raises(SystemExit, match="swap_map"):
        load(tmp_path, base_spec(tmp_path, ablate="motion_swap_exact:/d"))
    spec = load(tmp_path, base_spec(tmp_path, ablate="motion_swap_exact:/d", swap_map="T:1=T:2"))
    assert spec["swap_map"] == {"T:1": "T:2"}


def test_spec_shift_refuses_swap_map(tmp_path):
    assert load(tmp_path, base_spec(tmp_path, ablate="motion_shift"))["ablate"] == "motion_shift"
    with pytest.raises(SystemExit, match="swap_map"):
        load(tmp_path, base_spec(tmp_path, ablate="motion_shift", swap_map="T:1=T:2"))


def test_spec_swap_map_rule_for_existing_modes(tmp_path):
    with pytest.raises(SystemExit, match="swap_map"):
        load(tmp_path, base_spec(tmp_path, ablate="motion_swap:/d"))
    with pytest.raises(SystemExit, match="swap_map"):
        load(tmp_path, base_spec(tmp_path, swap_map="T:1=T:2"))
    with pytest.raises(SystemExit, match="swap_map"):
        load(tmp_path, base_spec(tmp_path, ablate="motion_zero", swap_map="T:1=T:2"))


@pytest.mark.parametrize("changes", [
    {"ablate": "motion_shift"},
    {"ablate": "motion_swap_exact:/d", "swap_map": "T:1=T:2"},
])
def test_spec_new_modes_need_motion_model(tmp_path, changes):
    with pytest.raises(SystemExit, match="只适用于"):
        load(tmp_path, base_spec(tmp_path, model_kind="history", **changes))


def test_server_command_new_modes(tmp_path):
    base = local.build_server_command(base_spec(tmp_path), "nonce", 9000)
    index = base.index("policy:checkpoint")

    shift = local.build_server_command(base_spec(tmp_path, ablate="motion_shift"), "nonce", 9000)
    assert shift == base[:index] + ["--ablate", "motion_shift"] + base[index:]

    spec = base_spec(tmp_path, ablate="motion_swap_exact:/d", swap_map="T:1=T:2")
    exact = local.build_server_command(spec, "nonce", 9000, episodes=[{"episode_key": "T:1"}])
    assert exact == base[:index] + ["--ablate", "motion_swap_exact:/d", "--swap-donor", "T:2"] + base[index:]


def test_server_command_existing_modes_unchanged(tmp_path):
    base = local.build_server_command(base_spec(tmp_path), "nonce", 9000)
    index = base.index("policy:checkpoint")
    spec = base_spec(tmp_path, ablate="motion_swap:/d", swap_map="T:1=T:2")
    swap = local.build_server_command(spec, "nonce", 9000, episodes=[{"episode_key": "T:1"}])
    assert swap == base[:index] + ["--ablate", "motion_swap:/d", "--swap-donor", "T:2"] + base[index:]
    zero = local.build_server_command(base_spec(tmp_path, ablate="motion_zero"), "nonce", 9000)
    assert zero == base[:index] + ["--ablate", "motion_zero"] + base[index:]


def test_identity_includes_new_names_and_keeps_old(tmp_path):
    manifest = {"sha256": "b" * 64}
    core = {"manifest_seal": "b" * 64, "checkpoint": "/ckpt", "policy_config": "cfg",
            "model_kind": "stage2_pool4_concat", "model_seed": 42}
    assert local.run_identity(base_spec(tmp_path), manifest) == M.digest(core)
    assert local.run_identity(base_spec(tmp_path, ablate="motion_zero"), manifest) == M.digest(core | {"ablate": "motion_zero"})
    assert local.run_identity(base_spec(tmp_path, ablate="motion_swap:/d", swap_map="T:1=T:2"), manifest) == M.digest(
        core | {"ablate": "motion_swap:/d", "swap_map": {"T:1": "T:2"}})
    assert local.run_identity(base_spec(tmp_path, ablate="motion_shift"), manifest) == M.digest(core | {"ablate": "motion_shift"})
    assert local.run_identity(base_spec(tmp_path, ablate="motion_swap_exact:/d", swap_map="T:1=T:2"), manifest) == M.digest(
        core | {"ablate": "motion_swap_exact:/d", "swap_map": {"T:1": "T:2"}})


# ―― 不存注意力 ――

REPLACING = ["motion_swap:{d}", "motion_swap_exact:{d}", "motion_shift"]
P, MEM, AH = 16, 7, SA.ACTION_HORIZON
ACTIONS = np.arange(AH * SA.ACTION_DIM, dtype=np.float32).reshape(AH, SA.ACTION_DIM)


def attn_recorder(tmp_path, ablate="none", **kwargs):
    """假策略走完 reset → add_buffer → infer：infer 经 _sample_actions（真实的记录器遮蔽）并调用 motion client。"""
    write_donor(tmp_path)
    out = tmp_path / "attn"
    policy = fake_policy()
    policy.metadata = {"model_kind": "stage2_pool4_concat"}
    policy.step_idx, policy.exec_start_idx = 0, 0
    policy._motion_client = Inner()

    def make_buffer():
        motion_times = np.asarray([0, 16, -1], np.int32)
        return SimpleNamespace(motion_enc=policy._motion_client,
                               _prepare_motion=lambda step: (None, None, None, motion_times),
                               get_frame_sampling_indices=lambda step, budget, tpi: np.asarray([0, step], np.int64))

    policy.mem_buffer = make_buffer()
    policy.reset = lambda: setattr(policy, "mem_buffer", make_buffer())

    def add_buffer(obs):
        policy.step_idx += len(obs["images"])

    policy.add_buffer = add_buffer
    observation = SimpleNamespace(static_mask=np.ones((1, 4), bool), motion_mask=np.asarray([[True, True, False]]),
                                  mem_order=np.arange(7)[None], tokenized_prompt_mask=np.ones((1, 3), bool),
                                  motion_emb=np.zeros((1, 3, DIM), np.float32), motion_pos=np.zeros((1, 3, DIM), np.float32))

    def infer(obs):
        policy.mem_buffer._prepare_motion(policy.step_idx)
        for start in (0, 16):
            policy.mem_buffer.motion_enc(FRAMES, start)
        return {"actions": policy._sample_actions(None, observation)}

    policy.infer = infer
    adapter = SimpleNamespace(last_attn=None)

    def make_sample_fn():
        def sample(rng, obs, **kw):
            adapter.last_attn = {"P": P, "MEM": MEM,
                                 "probs_action": np.zeros((SA.NUM_STEPS, SA.NUM_LAYERS, SA.NUM_HEADS, AH, P + AH), np.float16),
                                 "probs_text": np.zeros((SA.NUM_LAYERS, SA.NUM_HEADS, SA.TEXT_LEN, P), np.float16),
                                 "probs_prefix": np.zeros((SA.NUM_LAYERS, SA.NUM_HEADS, P, P), np.float16)}
            return ACTIONS.copy()
        return sample

    adapter.make_sample_fn = make_sample_fn
    donor = "ButtonUnmaskSwap:8" if ablate.startswith(("motion_swap:", "motion_swap_exact:")) else None
    recorder = SA.AttnRecorder(policy, adapter, out, ablate=ablate.format(d=tmp_path), swap_donor=donor, **kwargs)
    recorder.install()
    return recorder, policy, out


def drive(policy, out, infers=2):
    """跑一局，返回 (逐次动作, episode 目录)。"""
    policy.reset()
    actions = []
    for _ in range(infers):
        policy.add_buffer({"images": np.zeros((1, 1, 2, 2, 3), np.uint8), "state": np.zeros((1, 8), np.float32)})
        result = policy.infer({"observation/state": np.zeros(8, np.float32), "prompt": "p"})
        actions.append(result["actions"])
    directory, = [path for path in out.iterdir() if path.is_dir()]
    return actions, directory


@pytest.mark.parametrize("ablate", REPLACING)
def test_replacement_writes_no_infer_and_full_episode_json(tmp_path, ablate):
    recorder, policy, out = attn_recorder(tmp_path, ablate)
    assert recorder.save_attn is False and recorder.save_top_frames == 0
    actions, directory = drive(policy, out)
    assert [path.name for path in directory.iterdir()] == ["episode.json"]
    for result in actions:
        np.testing.assert_array_equal(result, ACTIONS)
    info = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    assert info["attn_saved"] is False and info["infers"] == 2 and info["n_masked_keys"] == [0, 0]
    assert info["ablate"] == ablate.format(d=tmp_path) and info["prompt"] == "p"
    content = info["motion_content"]
    assert content["n_windows"] == 2 and content["n_replaced"] > 0
    assert content["n_replaced"] == len(content["replace_log"])


def test_unsaved_keeps_replace_log_and_masks_like_saved(tmp_path):
    """同一 ablate 的不存路径与存路径（只有 motion_zero 能对照）逐字段一致，只差文件与 attn_saved。"""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, saved_policy, saved_out = attn_recorder(tmp_path / "a", "motion_zero+mask_motion")
    _, unsaved_policy, unsaved_out = attn_recorder(tmp_path / "b", "motion_zero+mask_motion", save_attn=False)
    saved_actions, saved_dir = drive(saved_policy, saved_out)
    unsaved_actions, unsaved_dir = drive(unsaved_policy, unsaved_out)
    for left, right in zip(saved_actions, unsaved_actions):
        np.testing.assert_array_equal(left, right)
    assert not list(unsaved_dir.glob("*.npz")) and len(list(saved_dir.glob("infer_*.npz"))) == 2
    left = json.loads((saved_dir / "episode.json").read_text(encoding="utf-8"))
    right = json.loads((unsaved_dir / "episode.json").read_text(encoding="utf-8"))
    assert right.pop("attn_saved") is False and "attn_saved" not in left
    assert left["n_masked_keys"] == right["n_masked_keys"] == [2, 2]
    for key in ("episode_nonce", "reset_unix_time", "server_pid", "save_top_frames"):
        left.pop(key), right.pop(key)
    assert left == right


def test_replacement_with_masks_counts_keys(tmp_path):
    _, policy, out = attn_recorder(tmp_path, "motion_shift+mask_motion")
    _, directory = drive(policy, out, infers=1)
    info = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    assert info["n_masked_keys"] == [2] and not list(directory.glob("*.npz"))


@pytest.mark.parametrize("ablate", REPLACING)
def test_replacement_rejects_explicit_save(tmp_path, ablate):
    with pytest.raises(SystemExit, match="不能显式要求 save_attn"):
        attn_recorder(tmp_path, ablate, save_attn=True)
    spec = base_spec(tmp_path, ablate=ablate.format(d="/d"), save_attn=True)
    if "swap" in ablate:
        spec["swap_map"] = "T:1=T:2"
    with pytest.raises(SystemExit, match="save_attn=true"):
        load(tmp_path, spec)


@pytest.mark.parametrize("ablate", REPLACING)
def test_replacement_rejects_save_prefix(tmp_path, ablate):
    with pytest.raises(SystemExit, match="互斥"):
        attn_recorder(tmp_path, ablate, save_prefix=True)
    spec = base_spec(tmp_path, ablate=ablate.format(d="/d"), save_prefix=True)
    if "swap" in ablate:
        spec["swap_map"] = "T:1=T:2"
    with pytest.raises(SystemExit, match="互斥"):
        load(tmp_path, spec)


def test_none_default_still_writes(tmp_path):
    recorder, policy, out = attn_recorder(tmp_path)
    assert recorder.save_attn is True
    actions, directory = drive(policy, out)
    assert sorted(path.name for path in directory.glob("*.npz")) == ["infer_0000.npz", "infer_0001.npz"]
    np.testing.assert_array_equal(actions[0], ACTIONS)
    assert "attn_saved" not in json.loads((directory / "episode.json").read_text(encoding="utf-8"))


def test_none_explicit_false_does_not_write(tmp_path):
    recorder, policy, out = attn_recorder(tmp_path, save_attn=False)
    assert recorder.save_attn is False
    actions, directory = drive(policy, out)
    assert [path.name for path in directory.iterdir()] == ["episode.json"]
    np.testing.assert_array_equal(actions[1], ACTIONS)
    info = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    assert info["attn_saved"] is False and info["infers"] == 2 and "motion_content" not in info


def test_save_false_excludes_save_prefix(tmp_path):
    with pytest.raises(SystemExit, match="互斥"):
        attn_recorder(tmp_path, save_attn=False, save_prefix=True)
    with pytest.raises(SystemExit, match="互斥"):
        load(tmp_path, base_spec(tmp_path, save_attn=False, save_prefix=True))
    assert attn_recorder(tmp_path, save_attn=True, save_prefix=True)[0].save_attn is True


@pytest.mark.parametrize("changes", [
    {"rollout_record": True, "bin_trace": True},
    {"stopcube_mode": "oracle_query", "stopcube_config": "/c.json"},
])
def test_save_false_excludes_other_recorders(tmp_path, changes):
    with pytest.raises(SystemExit):
        load(tmp_path, base_spec(tmp_path, save_attn=False, **changes))


@pytest.mark.parametrize("value", ["false", 0, None])
def test_save_attn_must_be_bool(tmp_path, value):
    with pytest.raises(SystemExit, match="必须为 bool"):
        load(tmp_path, base_spec(tmp_path, save_attn=value))


@pytest.mark.parametrize("name", [
    "none", "mask_motion", "heads:L3H0", "motion_zero", "motion_zero+heads:L3H0",
    "motion_swap:/d", "motion_swap_exact:/d", "motion_shift", "motion_shift+mask_vis", "mask_vis+motion_swap:/d",
])
def test_driver_and_server_agree_on_replacement(name):
    assert local.replaces_motion({"ablate": name}) == SA.replaces_motion(SA.parse_ablate(name))


@pytest.mark.parametrize("name", ["none", "mask_motion", "mask_vis", "heads:L3H0", "motion_zero"])
def test_other_modes_still_save(tmp_path, name):
    assert SA.resolve_save_attn(SA.parse_ablate(name)) is True
    assert local.saves_attn({"ablate": name}) is True
    assert SA.resolve_save_attn(SA.parse_ablate(name), False) is False
    assert local.saves_attn({"ablate": name, "save_attn": False}) is False


def test_cli_exposes_no_save_attn(tmp_path):
    tyro = pytest.importorskip("tyro")
    cls = SA._make_args_class()
    common = ["--attn-out", str(tmp_path), "policy:checkpoint", "--policy.config", "c", "--policy.dir", "d"]
    assert tyro.cli(cls, args=common).no_save_attn is False
    assert tyro.cli(cls, args=["--no-save-attn", *common]).no_save_attn is True


def test_command_and_identity_missing_key_unchanged(tmp_path):
    manifest = {"sha256": "b" * 64}
    base = local.build_server_command(base_spec(tmp_path), "nonce", 9000)
    index = base.index("policy:checkpoint")
    off = base_spec(tmp_path, save_attn=False)
    assert local.build_server_command(off, "nonce", 9000) == base[:index] + ["--no-save-attn"] + base[index:]
    on = base_spec(tmp_path, save_attn=True)
    assert local.build_server_command(on, "nonce", 9000) == base
    core = {"manifest_seal": "b" * 64, "checkpoint": "/ckpt", "policy_config": "cfg",
            "model_kind": "stage2_pool4_concat", "model_seed": 42}
    assert local.run_identity(base_spec(tmp_path), manifest) == M.digest(core)
    assert local.run_identity(on, manifest) == M.digest(core)
    assert local.run_identity(off, manifest) == M.digest(core | {"save_attn": False})
    # 替换实验缺键：命令与身份都与加本功能之前逐字相同，由服务端自行不写
    shift = base_spec(tmp_path, ablate="motion_shift")
    assert "--no-save-attn" not in local.build_server_command(shift, "nonce", 9000)
    assert local.run_identity(shift, manifest) == M.digest(core | {"ablate": "motion_shift"})
    # 替换实验显式 false：并入身份并显式传参
    both = base_spec(tmp_path, ablate="motion_shift", save_attn=False)
    assert "--no-save-attn" in local.build_server_command(both, "nonce", 9000)
    assert local.run_identity(both, manifest) == M.digest(core | {"ablate": "motion_shift", "save_attn": False})


def collect_row(tmp_path, spec_changes, *, infers=2, terminal=True, files=()):
    """按 run_local.collect 读取的目录形状造一局，返回 collect 的单局结果。"""
    attn, output = tmp_path / "attn", tmp_path / "client"
    (attn / "n").mkdir(parents=True)
    output.mkdir()
    (attn / "n" / "episode.json").write_text(json.dumps({"infers": infers}))
    for name in files:
        (attn / "n" / name).write_bytes(b"")
    fields = dict(episode_key="T:1", episode_nonce="n", task_name="T", episode_id=1, env_seed=1, difficulty="easy",
                  chunk_id="c", run_identity_sha256="r", outcome="fail", rpc_seq=5)
    (output / f"n.{'terminal' if terminal else 'started'}.json").write_text(json.dumps(fields))
    spec = base_spec(tmp_path, attn_out=str(attn), run_root=str(tmp_path / "run"), **spec_changes)
    row, = local.collect(spec, tmp_path, output, [{"episode_key": "T:1", "task_name": "T", "episode_id": 1}])
    return row


@pytest.mark.parametrize("changes", [
    {"ablate": "motion_shift"},
    {"ablate": "motion_swap:/d", "swap_map": {"T:1": "T:2"}},
    {"ablate": "motion_swap_exact:/d", "swap_map": {"T:1": "T:2"}},
    {"save_attn": False},
])
def test_collect_unsaved_judged_by_episode_json(tmp_path, changes):
    row = collect_row(tmp_path, changes)
    assert row["npz"] == 0 and row["attn_saved"] is False and row["episode_infers"] == 2
    assert local.row_ok(row) is True


def test_collect_unsaved_fails_without_infers(tmp_path):
    assert local.row_ok(collect_row(tmp_path, {"ablate": "motion_shift"}, infers=0)) is False


def test_collect_unsaved_fails_without_terminal(tmp_path):
    row = collect_row(tmp_path, {"ablate": "motion_shift"}, terminal=False)
    assert row["terminal"] is False and local.row_ok(row) is False


def test_collect_unsaved_fails_on_stray_infer_file(tmp_path):
    row = collect_row(tmp_path, {"ablate": "motion_shift"}, files=["infer_0000.npz"])
    assert row["npz"] == 1 and local.row_ok(row) is False


def test_collect_saved_judgement_unchanged(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    # rpc_seq=5 → infers=2；npz 数须相等，且行里不多出新键
    good = collect_row(tmp_path / "a", {}, files=["infer_0000.npz", "infer_0001.npz"])
    assert good["npz"] == good["infers"] == 2 and "attn_saved" not in good and local.row_ok(good) is True
    short = collect_row(tmp_path / "b", {}, files=["infer_0000.npz"])
    assert local.row_ok(short) is False
