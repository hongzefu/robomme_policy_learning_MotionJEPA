"""S7 单测：--ablate 解析、记忆区 key 遮挡的位置计算、命令形状 / 身份透传、serve 侧记录，以及头遮挡 attention 类的数值语义（CPU）。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.interpret import run_local as RL
from scripts.interpret import serve_policy_attn as SA

SENTINEL = int(np.iinfo(np.int32).max)


# ―――――――――――――――――――――――― 解析 ――――――――――――――――――――――――

def test_parse_ablate_presets():
    assert SA.parse_ablate("none") == {"name": "none", "masks": frozenset(), "heads": (), "motion": None}
    assert SA.parse_ablate(None)["name"] == "none"
    assert SA.parse_ablate("mask_motion")["masks"] == {"mask_motion"}
    assert SA.parse_ablate("mask_vis")["masks"] == {"mask_vis"}
    got = SA.parse_ablate("heads:L16H0,L3H0")
    assert got["masks"] == frozenset() and got["heads"] == ((3, 0), (16, 0))
    combo = SA.parse_ablate("mask_vis+heads:L14H6")
    assert combo["masks"] == {"mask_vis"} and combo["heads"] == ((14, 6),) and combo["name"] == "mask_vis+heads:L14H6"


@pytest.mark.parametrize("bad", ["mask_all", "heads:", "heads:L18H0", "heads:L0H8", "heads:3,0", "mask_vis+foo"])
def test_parse_ablate_rejects(bad):
    with pytest.raises(ValueError):
        SA.parse_ablate(bad)


# ―――――――――――――――――――――――― 记忆区 key 遮挡位置 ――――――――――――――――――――――――

def _synthetic(seed=0):
    """motion 布局：512 帧 token（32 帧 × 16）+ 96 motion，mem_order 为随机置换；frame_times 前 20 帧有效。"""
    mem_order = np.random.default_rng(seed).permutation(608).astype(np.int32)
    frame_times = np.full(32, -1, np.int32)
    frame_times[:20] = np.arange(20) * 3          # 全域帧号 0,3,...,57；< 32 的是前 11 帧（0..30）
    return mem_order, frame_times


def test_mask_motion_positions():
    mem_order, frame_times = _synthetic()
    mask = SA.memory_key_mask({"mask_motion"}, mem_order, frame_times, budget=512, tokens_per_frame=16)
    assert mask.shape == (608,) and mask.sum() == 96
    assert np.array_equal(np.flatnonzero(mask), np.flatnonzero(mem_order >= 512))


def test_mask_vis_positions():
    mem_order, frame_times = _synthetic()
    mask = SA.memory_key_mask({"mask_vis"}, mem_order, frame_times, budget=512, tokens_per_frame=16)
    expect = np.zeros(608, bool)
    for s, src in enumerate(mem_order):
        if src < 512 and frame_times[src // 16] >= 0 and frame_times[src // 16] < 32:
            expect[s] = True
    assert np.array_equal(mask, expect)
    assert mask.sum() == 11 * 16                  # 帧号 0,3,...,30 共 11 帧，每帧 16 token
    assert not (mask & (mem_order >= 512)).any()  # motion token 不被 mask_vis 选中
    frames_hit = set((mem_order[mask] // 16).tolist())
    assert frames_hit == set(range(11))


def test_mask_vis_baseline_identity_and_valid():
    frame_times = np.full(32, -1, np.int32)
    frame_times[:10] = np.arange(10) * 4          # 帧号 0..36；< 32 的是前 8 帧
    order = np.arange(512)
    mask = SA.memory_key_mask({"mask_vis"}, order, frame_times, budget=512, tokens_per_frame=16)
    assert np.array_equal(np.flatnonzero(mask), np.arange(8 * 16))
    valid = np.ones(512, bool)
    valid[:16] = False                            # 有效位取与：无效 token 不计入
    mask_v = SA.memory_key_mask({"mask_vis"}, order, frame_times, budget=512, tokens_per_frame=16, valid=valid)
    assert np.array_equal(np.flatnonzero(mask_v), np.arange(16, 8 * 16))


def test_mask_combined_is_union():
    mem_order, frame_times = _synthetic(1)
    kw = dict(budget=512, tokens_per_frame=16)
    both = SA.memory_key_mask({"mask_vis", "mask_motion"}, mem_order, frame_times, **kw)
    a = SA.memory_key_mask({"mask_vis"}, mem_order, frame_times, **kw)
    b = SA.memory_key_mask({"mask_motion"}, mem_order, frame_times, **kw)
    assert np.array_equal(both, a | b) and not (a & b).any()
    assert not SA.memory_key_mask(set(), mem_order, frame_times, **kw).any()


# ―――――――――――――――――――――――― 命令形状 / 身份 ――――――――――――――――――――――――

def _cmd_spec(**kw):
    spec = {"policy_python": "/p/python", "stage2_python": "/s/python", "run_identity_sha256": "a" * 64,
            "checkpoint": "/ck", "policy_config": "cfg", "model_kind": "stage2_pool4_concat", "cuda": "1",
            "attn_out": "/attn"}
    spec.update(kw)
    return spec


@pytest.mark.parametrize("ablate", [None, "none", "mask_motion", "mask_vis+heads:L14H6"])
def test_server_command_ablate(ablate):
    spec = _cmd_spec() if ablate is None else _cmd_spec(ablate=ablate)
    command = RL.build_server_command(spec, "n" * 32, 9100)
    i = command.index("policy:checkpoint")
    assert command[i:] == ["policy:checkpoint", "--policy.config", "cfg", "--policy.dir", "/ck"]
    if ablate in (None, "none"):
        assert "--ablate" not in command
        assert command[i - 2:i] == ["--attn-out", "/attn"]
    else:
        assert command.count("--ablate") == 1
        assert command[i - 2:i] == ["--ablate", ablate] and command[i - 4:i - 2] == ["--attn-out", "/attn"]


def test_server_command_ablate_with_prefix():
    command = RL.build_server_command(_cmd_spec(ablate="mask_vis", save_prefix=True), "n" * 32, 9100)
    i = command.index("policy:checkpoint")
    assert command[i - 5:i] == ["--attn-out", "/attn", "--save-prefix", "--ablate", "mask_vis"]


def test_run_identity_ablate():
    manifest = {"sha256": "f" * 64}
    spec = {"checkpoint": "/ck", "policy_config": "cfg", "model_kind": "history", "model_seed": 42}
    base = RL.run_identity(spec, manifest)
    assert RL.run_identity(dict(spec, ablate="none"), manifest) == base       # none 时身份与 S7 之前相同
    vis = RL.run_identity(dict(spec, ablate="mask_vis"), manifest)
    assert vis != base and vis != RL.run_identity(dict(spec, ablate="heads:L3H0"), manifest)


# ―――――――――――――――――――――――― serve 侧记录 ――――――――――――――――――――――――

class FakeMem:
    def get_frame_sampling_indices(self, step_idx, budget, token_per_image):
        return list(range(0, step_idx + 1, 4))[:32]

    def _prepare_motion(self, step_idx):
        mask = np.zeros(96, bool)
        mask[:3] = True
        times = np.full(96, SENTINEL, np.int64)
        times[:3] = [0, 16, 32]
        return np.zeros((96, 768), np.float32), np.zeros((96, 4), np.float32), mask, times


class FakePolicy:
    def __init__(self, motion):
        self.motion_enabled = motion
        self.config = SimpleNamespace(budget=512, token_per_image=8, num_views=2, motion=SimpleNamespace(budget=96))
        self.metadata = {"model_kind": "stage2_pool4_concat" if motion else "history"}
        self.reset()

    def reset(self):
        self.mem_buffer = FakeMem()
        self.step_idx, self.exec_start_idx = -1, 0

    def add_buffer(self, obs):
        self.step_idx += len(obs["images"])

    def infer(self, obs):
        n_frames = len(self.mem_buffer.get_frame_sampling_indices(self.step_idx, 512, 8))
        static_mask = np.repeat(np.arange(32) < n_frames, 16)[None]
        prompt_mask = (np.arange(64) < 21)[None]
        if self.motion_enabled:
            motion_emb, motion_pos, motion_mask, _ = self.mem_buffer._prepare_motion(self.step_idx)
            mem_order = np.random.default_rng(0).permutation(608).astype(np.int32)[None]
            observation = SimpleNamespace(static_mask=static_mask, motion_mask=motion_mask[None], mem_order=mem_order,
                                          motion_emb=motion_emb[None], motion_pos=motion_pos[None],
                                          tokenized_prompt_mask=prompt_mask)
        else:
            observation = SimpleNamespace(static_mask=static_mask, motion_mask=None, mem_order=None,
                                          tokenized_prompt_mask=prompt_mask)
        actions = self._sample_actions(None, observation)
        return {"actions": actions, "state": np.zeros(8, np.float32), "infer_time_ms": 1.0}


class KwAdapter:
    """记录 sample_fn 收到的 mask_keys。"""

    def __init__(self, P, MEM):
        self.P, self.MEM, self.last_attn, self.seen = P, MEM, None, []

    def make_sample_fn(self):
        def sample(rng, observation, **kw):
            self.seen.append(kw.get("mask_keys"))
            self.last_attn = {"probs_action": np.zeros((10, 18, 8, 20, self.P + 20), np.float16),
                              "probs_text": np.zeros((18, 8, 64, self.P), np.float16), "P": self.P, "MEM": self.MEM}
            return np.zeros((20, 8), np.float32)
        return sample


def _run_one(tmp_path, monkeypatch, motion, ablate, nonce):
    P, MEM = (1184, 608) if motion else (1088, 512)
    policy = FakePolicy(motion)
    adapter = KwAdapter(P, MEM)
    recorder = SA.AttnRecorder(policy, adapter, tmp_path / "attn", save_top_frames=0, ablate=ablate)
    recorder.install()
    monkeypatch.setitem(SA._RPC, "op", "reset")
    monkeypatch.setitem(SA._RPC, "rpc", {"episode_nonce": nonce})
    policy.reset()
    policy.add_buffer({"images": np.zeros((40, 2, 4, 4, 3), np.uint8), "state": np.zeros((40, 8), np.float32)})
    policy.infer({"observation/state": np.zeros(8), "prompt": "p"})
    return adapter, tmp_path / "attn" / nonce


@pytest.mark.parametrize("motion,ablate", [(False, "none"), (True, "none"), (False, "mask_vis"), (True, "mask_vis"),
                                           (True, "mask_motion"), (True, "heads:L3H0")])
def test_recorder_ablate_record(tmp_path, monkeypatch, motion, ablate):
    adapter, ep = _run_one(tmp_path, monkeypatch, motion, ablate, "n1")
    # FakeMem 采到 10 帧，帧号 0,4,...,36；< 32 的是前 8 帧
    mem_order = np.random.default_rng(0).permutation(608) if motion else np.arange(512)
    masks = SA.parse_ablate(ablate)["masks"]
    if masks:
        mk = adapter.seen[0]
        assert mk.dtype == bool and mk.shape == (608 if motion else 512,)
        if ablate == "mask_vis":
            expect = (mem_order < 512) & (mem_order // 16 < 8)
        else:                                                       # mask_motion：只遮有效的 3 个 motion token
            expect = (mem_order >= 512) & (mem_order - 512 < 3)
        assert np.array_equal(mk, expect)
        n = int(expect.sum())
    else:
        assert adapter.seen == [None]                              # 无 key 遮挡时不传 mask_keys
        n = 0
    with np.load(ep / "infer_0000.npz") as z:
        expected_keys = set(SA.NPZ_KEYS) | set(SA.ABLATE_NPZ_KEYS)
        if motion:
            expected_keys |= set(SA.MOTION_INPUT_NPZ_KEYS)
        assert set(z.files) == expected_keys
        assert str(z["ablate"]) == ablate and z["n_masked_keys"].dtype == np.int64 and int(z["n_masked_keys"]) == n
    info = json.loads((ep / "episode.json").read_text())
    assert info["ablate"] == ablate and info["n_masked_keys"] == [n]


def test_recorder_mask_motion_baseline_exits(tmp_path):
    with pytest.raises(SystemExit):
        SA.AttnRecorder(FakePolicy(False), KwAdapter(1088, 512), tmp_path / "attn", ablate="mask_motion")


def test_serve_help_has_ablate():
    pytest.importorskip("tyro")
    import dataclasses
    try:
        Args = SA._make_args_class()
    except Exception as exc:                                       # noqa: BLE001
        pytest.skip(f"serve_policy 不可导入: {type(exc).__name__}")
    names = {f.name: f.default for f in dataclasses.fields(Args)}
    assert names["ablate"] == "none"


# ―――――――――――――――――――――――― 头遮挡 attention 类（CPU 数值）――――――――――――――――――――――――

def test_ablate_attention_semantics():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from openpi.models import gemma
    from scripts.interpret import attn_adapter as AA
    import mv_model_adapter as MVA

    cfg = gemma.get_config("dummy")                                # 8 头、1 个 kv 头、head_dim 16
    B, T = 1, 6
    x = jax.random.normal(jax.random.key(0), (B, T, cfg.width), jnp.float32)
    pos = jnp.arange(T)[None]
    mask = jnp.tril(jnp.ones((B, 1, T, T), bool))
    Patched = MVA._make_patched_attention()
    params = Patched(configs=(cfg,)).init(jax.random.key(1), [x], pos, mask, None)

    def apply(cls, p):
        MVA._CAPTURE.clear()
        (out,), _ = cls(configs=(cfg,)).apply(p, [x], pos, mask, None)
        return np.asarray(out), np.asarray(MVA._CAPTURE[0])

    ref, probs_ref = apply(Patched, params)
    same, probs_same = apply(AA._make_ablate_attention(()), params)
    assert np.array_equal(ref, same) and np.array_equal(probs_ref, probs_same)    # 不遮 = 原类逐位一致
    for heads in [(2,), (0, 5)]:
        abl, probs_abl = apply(AA._make_ablate_attention(heads), params)
        assert np.array_equal(probs_abl, probs_ref)                               # 概率不受影响
        w = params["params"]["attn_vec_einsum"]["w"]                               # (N, H, D)
        pz = jax.tree.map(lambda a: a, params)
        pz["params"]["attn_vec_einsum"]["w"] = w.at[jnp.asarray(heads)].set(0.0)
        wz, _ = apply(Patched, pz)
        np.testing.assert_allclose(abl, wz, atol=1e-6, rtol=0)                    # 该头对输出贡献为 0
        assert np.abs(abl - ref).max() > 1e-4                                     # 确实改变输出
    with pytest.raises(ValueError):
        AA.normalize_heads(((18, 0),))
    assert AA.normalize_heads([(16, 0), (3, 0), (3, 0)]) == ((3, 0), (16, 0))
