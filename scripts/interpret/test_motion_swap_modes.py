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
