"""attn_adapter 隐状态捕获的 CPU 定向测试：用假模型跑真实的 HistoryBlock 展开路径，不加载权重。

假模型只把 embed_prefix / embed_suffix / action_out_proj 换成小维度的简单实现，18 层 HistoryBlock、
注意力捕获和去噪 while_loop 走的都是 attn_adapter 的生产代码。真实权重上的一致性留给 S2 的 HIDDEN_PARITY。
"""
from __future__ import annotations

import dataclasses
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.nnx as nnx  # noqa: E402
import flax.struct  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from scripts.interpret import attn_adapter as A  # noqa: E402
from mme_vla_suite.models.integration import history_gemma as hg  # noqa: E402
from mme_vla_suite.models.integration import history_observation as ho  # noqa: E402
from mme_vla_suite.models.integration import utils as hu  # noqa: E402
from mme_vla_suite.models.integration.history_pi0 import make_attn_mask  # noqa: E402
from openpi.models import gemma  # noqa: E402

W, W_A, AH, AD = 32, 24, 20, 4
MEM, TXT, N_IMG, IMG_EACH = 6, 3, 2, 2
P = MEM + N_IMG * IMG_EACH + TXT
STEPS = 10
# 第 2 组是空组（对应基线没有 motion）；第 3 组含一个 padding token，有效 token 只有 2 个
GROUPS = np.zeros((4, P), bool)
GROUPS[0, 0:4] = True
GROUPS[1, 4:6] = True
GROUPS[3, P - 3:P] = True
PADDED = P - 1


@flax.struct.dataclass
class _FakeObs:
    state: jax.Array
    prefix: jax.Array
    prefix_valid: jax.Array
    tokenized_prompt: jax.Array
    images: dict


def _fake_config(variant):
    return dataclasses.replace(hu.get_config("dummy"), width=W if variant == "fake_prefix" else W_A, depth=18,
                               mlp_dim=16, head_dim=4)


def _params_tree(key, obs):
    """按生产结构初始化：layers 每个叶子首维是层数，final_norm_1 为动作路径的 AdaRMS。"""
    block = hg.HistoryBlock(configs=(_fake_config("fake_prefix"), _fake_config("fake_action")), integration_type="context")
    h = obs.prefix.astype(jnp.float32)
    hs = jnp.zeros((1, AH, W_A), jnp.float32)
    ad = jnp.ones((1, 8), jnp.float32)
    T = P + AH
    mask = jnp.ones((1, 1, T, T), bool)
    pos = jnp.arange(T)[None]
    keys = jax.random.split(key, 19)
    layers = [block.init(k, [h, hs], None, pos, mask, [None, ad], None, None, True)["params"] for k in keys[:18]]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *layers)
    final = gemma.RMSNorm().init(keys[18], hs, ad)["params"]
    return stacked, final


def _to_nnx(tree):
    if isinstance(tree, dict):
        return nnx.Dict({k: _to_nnx(v) for k, v in tree.items()})
    return nnx.Param(tree)


class _FakeModel(nnx.Module):
    use_history = True
    integration_type = "context"
    action_horizon = AH
    action_dim = AD

    def __init__(self, obs):
        self.config = SimpleNamespace(dtype="float32", paligemma_variant="fake_prefix", action_expert_variant="fake_action")
        layers, final = _params_tree(jax.random.key(3), obs)
        self.PaliGemma = nnx.Dict(llm=nnx.Dict(layers=_to_nnx(layers), final_norm_1=_to_nnx(final)))
        rngs = nnx.Rngs(5)
        self.action_in = nnx.Linear(AD, W_A, rngs=rngs)
        self.action_out_proj = nnx.Linear(W_A, AD, rngs=rngs)

    def embed_memory(self, o):
        return o.prefix[:, :MEM], o.prefix_valid[:, :MEM], None, None

    def embed_prefix(self, o):
        return o.prefix, o.prefix_valid, jnp.zeros(o.prefix_valid.shape, bool), None

    def embed_suffix(self, o, x_t, time):
        st = self.action_in(x_t)
        mask = jnp.ones((x_t.shape[0], AH), bool)
        ar = jnp.zeros(mask.shape, bool).at[:, 0].set(True)
        ad = jnp.stack([time * (i + 1.0) for i in range(8)], axis=-1)
        return st, mask, ar, None, ad


def _make_obs():
    rng = np.random.default_rng(7)
    valid = np.ones((1, P), bool)
    valid[0, PADDED] = False
    prefix = rng.normal(size=(1, P, W)).astype(np.float32)
    return _FakeObs(state=jnp.zeros((1, 2)), prefix=jnp.asarray(prefix), prefix_valid=jnp.asarray(valid),
                    tokenized_prompt=jnp.zeros((1, TXT), jnp.int32),
                    images={"a": jnp.zeros((1, 1)), "b": jnp.zeros((1, 1))})


@pytest.fixture(scope="module")
def env():
    patch = pytest.MonkeyPatch()
    original = hu.get_config
    fake = {name: _fake_config(name) for name in ("fake_prefix", "fake_action")}
    patch.setattr(hu, "get_config", lambda v: fake.get(v) or original(v))
    patch.setattr(ho, "preprocess_observation", lambda _rng, o, **_: o)
    obs = _make_obs()
    model = _FakeModel(obs)
    adapter = A.AttnAdapter(A._ModelShim(model), require_flags=False)
    noise = jax.random.normal(jax.random.key(1), (1, AH, AD), jnp.float32)
    yield SimpleNamespace(adapter=adapter, obs=obs, noise=noise, model=model)
    patch.undo()


@pytest.fixture(scope="module")
def outputs(env):
    key = jax.random.key(0)
    off = env.adapter.run(key, env.obs, noise=env.noise, num_steps=STEPS, out_f16=False)
    on = env.adapter.run(key, env.obs, noise=env.noise, num_steps=STEPS, out_f16=False,
                         capture_hidden=True, group_masks=GROUPS)
    return SimpleNamespace(off=off, on=on)


@pytest.fixture(scope="module", autouse=True)
def _verdict_line(request):
    """pytest 收尾时打印判定行，排在 -q 摘要之后。"""
    session = request.session

    class _Plugin:
        def pytest_unconfigure(self):
            verdict = "PASS" if session.testsfailed == 0 else "FAIL"
            print(f"HIDDEN_UNIT={verdict} tests={session.testscollected}")

    request.config.pluginmanager.register(_Plugin(), "hidden-unit-verdict")
    yield


def test_adapter_module_points_to_this_tree():
    here = os.path.dirname(os.path.abspath(__file__))
    assert os.path.dirname(os.path.abspath(A.__file__)) == here


def test_group_mean_matches_hand_calculation():
    tokens = jnp.array([[1., 10.], [3., 30.], [5., 50.], [100., 100.]])
    masks = jnp.array([[True, True, False, False],     # (1+3)/2、(10+30)/2
                       [False, False, True, True],     # 第 3 个 token 无效，只剩第 2 个
                       [False, False, False, False]])  # 空组
    valid = jnp.array([True, True, True, False])
    mean, counts = A.group_token_mean(tokens, masks, valid)
    np.testing.assert_array_equal(np.asarray(mean), [[2., 20.], [5., 50.], [0., 0.]])
    np.testing.assert_array_equal(np.asarray(counts), [2, 1, 0])
    assert mean.dtype == jnp.float32 and counts.dtype == jnp.int32


def test_group_mean_ignores_nan_outside_group_and_empty_group_is_zero():
    tokens = jnp.array([[1., jnp.inf], [jnp.nan, jnp.nan], [3., 4.]])
    masks = jnp.array([[False, True, True], [False, True, False]])
    mean, counts = A.group_token_mean(tokens, masks, jnp.array([True, False, True]))
    np.testing.assert_array_equal(np.asarray(mean), [[3., 4.], [0., 0.]])
    np.testing.assert_array_equal(np.asarray(counts), [1, 0])


def test_action_mean_uses_first_16_of_last_ah():
    hs = jnp.arange(23.0)[:, None]                     # 前 3 个不是动作 token
    assert float(A.action_token_mean(hs, 20)[0]) == pytest.approx(np.arange(3, 19).mean())


def test_sampling_outputs_bitwise_equal(outputs):
    for name, off, on in zip(("x_0", "probs_action", "probs_text"), outputs.off[:3], outputs.on[:3]):
        assert np.array_equal(np.asarray(off), np.asarray(on)), name
    assert int(outputs.on[3]) == STEPS
    assert len(outputs.off) == 5 and len(outputs.on) == 6


def test_sampling_outputs_bitwise_equal_with_f16_and_save_prefix(env):
    key = jax.random.key(0)
    off = env.adapter.run(key, env.obs, noise=env.noise, num_steps=STEPS, save_prefix=True)
    on = env.adapter.run(key, env.obs, noise=env.noise, num_steps=STEPS, save_prefix=True,
                         capture_hidden=True, group_masks=GROUPS)
    assert len(off) == 6 and len(on) == 7 and set(on[-1]) == {"hidden_prefix", "hidden_action", "group_tokens"}
    for index in (0, 1, 2, 5):
        assert np.array_equal(np.asarray(off[index]), np.asarray(on[index])), index
    assert on[-1]["hidden_prefix"].dtype == jnp.float32 and on[-1]["hidden_action"].dtype == jnp.float32


def test_hidden_shapes_counts_and_empty_group(outputs):
    hidden = outputs.on[-1]
    assert set(hidden) == {"hidden_prefix", "hidden_action", "group_tokens"}
    assert hidden["hidden_prefix"].shape == (19, 4, W) and hidden["hidden_prefix"].dtype == jnp.float32
    assert hidden["hidden_action"].shape == (18, W_A) and hidden["hidden_action"].dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(hidden["group_tokens"]), [4, 2, 0, 2])
    prefix = np.asarray(hidden["hidden_prefix"])
    assert np.isfinite(prefix).all() and np.isfinite(np.asarray(hidden["hidden_action"])).all()
    assert (prefix[:, 2] == 0).all()
    assert not (prefix[:, 0] == 0).all()


def _manual_rollout(env):
    """不经 f_sample_attn，用普通 HistoryBlock 独立重算一遍隐状态，作为手算对照。"""
    cfgs = env.adapter.cfgs
    block = hg.HistoryBlock(configs=cfgs, integration_type="context")
    llm = env.adapter.gstate.to_pure_dict()["PaliGemma"]["llm"]
    layers, final = llm["layers"], llm["final_norm_1"]
    pt, pm, par_, pna = env.model.embed_prefix(env.obs)
    mask = make_attn_mask(pm, par_, pna)
    pos = jnp.cumsum(pm, axis=1) - 1
    h = pt
    rows = [pt[0]]
    caches = []
    for l in range(18):
        out, kv = block.apply({"params": jax.tree.map(lambda x: x[l], layers)}, [h, None], None, pos, mask[:, None],
                              [None, None], None, None, True)
        h = out[0]
        rows.append(h[0])
        caches.append(kv)
    x_t, time, sums = env.noise, 1.0, np.zeros((18, W_A))
    dt = -1.0 / STEPS
    for _ in range(STEPS):
        st, sm, sar, _, ad = env.model.embed_suffix(env.obs, x_t, jnp.broadcast_to(time, (1,)))
        full = jnp.concatenate([jnp.broadcast_to(pm[:, None], (1, AH, P)), make_attn_mask(sm, sar)], axis=-1)
        spos = jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
        hs = st
        for l in range(18):
            out, _ = block.apply({"params": jax.tree.map(lambda x: x[l], layers)}, [None, hs], caches[l], spos,
                                 full[:, None], [None, ad], None, None, True)
            hs = out[1]
            sums[l] += np.asarray(hs[0, :16]).mean(axis=0)
        hs, _ = gemma.RMSNorm().apply({"params": final}, hs, ad)
        x_t = x_t + dt * env.model.action_out_proj(hs[:, -AH:])
        time += dt
    valid = np.asarray(pm[0])
    prefix = np.zeros((19, 4, W))
    for r, row in enumerate(rows):
        for g in range(4):
            members = GROUPS[g] & valid
            if members.any():
                prefix[r, g] = np.asarray(row)[members].mean(axis=0)
    return prefix, sums / STEPS, np.asarray(x_t)


def test_hidden_values_match_independent_rollout(env, outputs):
    prefix, action, x_0 = _manual_rollout(env)
    hidden = outputs.on[-1]
    np.testing.assert_allclose(np.asarray(hidden["hidden_prefix"]), prefix, atol=1e-5)
    np.testing.assert_allclose(np.asarray(hidden["hidden_action"]), action, atol=1e-5)
    np.testing.assert_allclose(np.asarray(outputs.on[0]), x_0, atol=1e-5)
    # 第 0 项严格是进入第 0 层之前的 pt，无需容差
    expected = np.asarray(env.obs.prefix[0])[GROUPS[0]].mean(axis=0)
    np.testing.assert_allclose(np.asarray(hidden["hidden_prefix"])[0, 0], expected, atol=1e-6)


def test_off_call_form_unchanged(env):
    """关闭时 f_sample_attn 收到的实参必须和接入捕获之前逐字相同。"""
    seen = []
    adapter = env.adapter
    original = adapter.f_sample_attn

    def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return original(*args, **kwargs)

    adapter.f_sample_attn = spy
    try:
        adapter.run(jax.random.key(0), env.obs, noise=env.noise, num_steps=STEPS, out_f16=False)
        adapter.run(jax.random.key(0), env.obs, noise=env.noise, num_steps=STEPS, out_f16=False, save_prefix=True)
    finally:
        adapter.f_sample_attn = original
    assert [len(args) for args, _ in seen] == [7, 7]
    assert seen[0][1] == {} and seen[1][1] == {"save_prefix": True}
    assert seen[0][0][3] is env.noise and seen[0][0][4:] == (STEPS, MEM + N_IMG * IMG_EACH, False)


@pytest.mark.parametrize("kwargs", [
    {"mask_keys": np.zeros(P, bool)},
    {"ablate_heads": ((3, 0),)},
])
def test_capture_conflicts_with_ablation(env, kwargs):
    with pytest.raises(ValueError, match="capture_hidden"):
        env.adapter.run(jax.random.key(0), env.obs, noise=env.noise, num_steps=STEPS,
                        capture_hidden=True, group_masks=GROUPS, **kwargs)


def test_bad_arguments_rejected(env):
    key = jax.random.key(0)
    with pytest.raises(ValueError, match="group_masks"):
        env.adapter.run(key, env.obs, noise=env.noise, num_steps=STEPS, group_masks=GROUPS)
    with pytest.raises(ValueError, match="group_masks"):
        env.adapter.run(key, env.obs, noise=env.noise, num_steps=STEPS, capture_hidden=True)
    with pytest.raises(ValueError, match="group_masks"):
        env.adapter.run(key, env.obs, noise=env.noise, num_steps=STEPS, capture_hidden=True, group_masks=GROUPS[:, :-1])


def test_real_baseline_config_abstract_shapes(monkeypatch):
    """真实基线配置（不建参数值、不加载权重）：W、W_a 按配置读取。"""
    from mme_vla_suite.models.config.utils import get_history_config
    from mme_vla_suite.training import config as _config
    monkeypatch.chdir(A.C.REPO_ROOT)
    train_config = _config.get_config("mme_vla_suite_b128")
    history = get_history_config("perceptual-framesamp-context.yaml")
    model_cfg = dataclasses.replace(train_config.model, history_config=history, use_history=True)
    model = nnx.eval_shape(model_cfg.create, jax.random.key(0))
    adapter = A.AttnAdapter(A._ModelShim(model), require_flags=False)
    obs = A._robomme_obs_spec(model.config)
    lay = adapter.layout(obs)
    masks = jax.ShapeDtypeStruct((4, lay["P"]), jnp.bool_)
    out = jax.eval_shape(
        lambda s, r, o, g: adapter.f_sample_attn(s, r, o, None, STEPS, lay["MEM"] + lay["IMG"], True,
                                                 capture_hidden=True, group_masks=g),
        adapter.gstate, jax.random.key(0), obs, masks)
    hidden = out[-1]
    assert tuple(hidden["hidden_prefix"].shape) == (19, 4, adapter.cfgs[0].width)
    assert tuple(hidden["hidden_action"].shape) == (18, adapter.cfgs[1].width)
    assert hidden["hidden_prefix"].dtype == jnp.float32 and hidden["hidden_action"].dtype == jnp.float32
