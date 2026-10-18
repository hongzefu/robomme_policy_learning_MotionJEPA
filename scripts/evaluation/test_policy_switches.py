"""src/mme_vla_suite/policies 四处开关式小改的单测（1017-test-xhard5-12per 4.3 第 5 项，各一条主测）。

需要服务端依赖（jax、einops、openpi）；用评估服务端环境跑，并把 worktree 的 src 置前：
``JAX_PLATFORMS=cpu PYTHONPATH=<worktree>/src <v1-store/envs/policy-eval-server>/bin/python -m pytest scripts/evaluation/test_policy_switches.py -q``
第一条测试先断言 ``mme_vla_suite`` 确实从本仓库（本 worktree）的 src 导入。
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]

import mme_vla_suite  # noqa: E402
from mme_vla_suite.policies import framesamp_memory as fm  # noqa: E402
from mme_vla_suite.policies import motion_client as mc  # noqa: E402
from mme_vla_suite.policies import policy as policy_mod  # noqa: E402
from mme_vla_suite.policies import policy_config as pc  # noqa: E402


def test_imports_from_this_checkout():
    assert Path(mme_vla_suite.__file__).resolve() == REPO / "src/mme_vla_suite/__init__.py"


# ── ① MMEVLA_MOTION_ONLINE_GPU ────────────────────────────────────────────────
def test_online_gpu_env_overrides_snapshot(monkeypatch):
    monkeypatch.delenv("MMEVLA_MOTION_ONLINE_GPU", raising=False)
    assert pc._motion_online_gpu({"online_gpu": 4}) == 4          # 默认：快照卡号，行为不变
    assert pc._motion_online_gpu({}) == 1
    monkeypatch.setenv("MMEVLA_MOTION_ONLINE_GPU", "0")
    assert pc._motion_online_gpu({"online_gpu": 4}) == "0"


# ── ② MMEVLA_MOTION_PROV_RELAX ────────────────────────────────────────────────
def _client(sidecar):
    client = object.__new__(mc.MotionEncoderClient)
    client.provenance = sidecar
    client.closed = False
    client.close = lambda: setattr(client, "closed", True)
    return client


def _prov(**hw):
    vae = {k: "same" for k in mc.PROV_SAME_VAE}
    enc = {k: "same" for k in mc.PROV_SAME_ENC}
    for k, v in hw.items():
        if k in vae:
            vae[k] = v
        if k in enc:
            enc[k] = v
    return {"vae": vae, "encoder": enc}


def test_prov_relax_exempts_only_listed_keys(monkeypatch):
    store = _prov()
    a40 = _prov(gpu_name="A40", compute_cap="8.6", sm_count="84", driver="550")
    monkeypatch.delenv("MMEVLA_MOTION_PROV_RELAX", raising=False)
    with pytest.raises(RuntimeError):                               # 默认全键硬比对
        _client(a40).check_provenance(store)
    monkeypatch.setenv("MMEVLA_MOTION_PROV_RELAX", "gpu_name,compute_cap,sm_count,driver")
    client = _client(a40)
    client.check_provenance(store)                                  # 四键 × vae/encoder 两段都放行
    assert len(client.relaxed_diffs) == 8 and not client.closed
    bad = _prov(gpu_name="A40", torch="2.0")                        # 四键之外的软件键不等仍 raise
    with pytest.raises(RuntimeError):
        _client(bad).check_provenance(store)


# ── ③ MMEVLA_MOTION_OVERFLOW=resample（_motion_quota / _even_pick / MOTION_DOWNSAMPLE）──────────
def _memory(es, overflow, budget=160):
    mem = object.__new__(fm.FrameSampMemory)
    mem.exec_start_idx = es
    mem.motion_window, mem.motion_stride, mem.demo_min_real = 33, 16, 17
    mem.motion_budget, mem.motion_overflow = budget, overflow
    mem.motion_dim, mem.motion_pos_dim = 4, 2
    mem.pos_emb = np.zeros((4096, 1, 2), np.float32)
    mem.motion_downsample_steps = mem.motion_downsample_max_k = mem.motion_last_k = 0
    return mem


@pytest.mark.parametrize("es,k_demo,keep_demo", [(1247, 77, 77), (1500, 93, 80)])
def test_overflow_resample_fills_budget(es, k_demo, keep_demo):
    step = es + 2000                                                # max_steps=2000：执行段 124 窗
    raising = _memory(es, "raise")
    frames = raising.visible_motion_frames(step)
    assert len(frames) == k_demo + 124
    raising._history_feats_motion = {f: np.ones(4, np.float32) for f in frames}
    with pytest.raises(RuntimeError):                               # 默认 raise：零截断契约不变
        raising._prepare_motion(step)
    mem = _memory(es, "resample")
    mem._history_feats_motion = raising._history_feats_motion
    emb, pos, mask, times = mem._prepare_motion(step)
    assert mask.all() and len(mask) == 160
    assert mem.motion_downsample_steps == 1 and mem.motion_last_k == k_demo + 124
    assert fm._motion_quota(k_demo, 124, 160) == (keep_demo, 160 - keep_demo)
    picked = fm._even_pick(124, 160 - keep_demo)
    assert picked[0] == 0 and picked[-1] == 123 and len(set(picked)) == len(picked)
    assert fm._even_pick(5, 5) == [0, 1, 2, 3, 4]


# ── ④ policy.infer 回传 motion 计数 ─────────────────────────────────────────────
@pytest.mark.parametrize("motion", [False, True])
def test_infer_returns_motion_counts(monkeypatch, motion):
    import jax
    import jax.numpy as jnp
    monkeypatch.setattr(policy_mod, "HistAugObservation",
                        SimpleNamespace(from_dict=lambda d: SimpleNamespace(state=d["state"])))
    pol = object.__new__(policy_mod.MME_VLA_Policy)
    pol.mem_buffer = SimpleNamespace(n_steps=1, motion_last_k=201, motion_downsample_steps=3,
                                     motion_budget=160, motion_overflow="resample")
    pol._prepare_history = lambda x: x
    pol._input_transform = lambda x: x
    pol._output_transform = lambda x: x
    pol._sample_actions = lambda rng, obs: jnp.zeros((1, 16, 8))
    pol._sample_kwargs = {}
    pol._rng = jax.random.key(0)
    pol.motion_enabled = motion
    out = pol.infer({"state": np.zeros(8, np.float32)})
    assert out["actions"].shape == (16, 8)
    if motion:
        assert (out["motion_k"], out["motion_downsample_steps"], out["motion_budget"], out["motion_overflow"]) \
            == (201, 3, 160, "resample")
    else:
        assert not any(k.startswith("motion_") for k in out)       # 非 motion 模型：回包与改动前相同
