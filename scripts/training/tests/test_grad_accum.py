"""梯度累积（--grad-accum-steps）的 CPU 单测。

覆盖三件事：
- V2 线性性：确定性玩具 loss 下，K=1/2/4 累积梯度与全 batch 梯度一致（fp32 相对误差 < 1e-6）；
- V3 切分：B=128、K=2、4 个模拟设备，每样本恰好进一个 micro、每 micro 每设备 16 个；
- K=1 时 train_step 不经过 split_microbatches / accumulate_microbatch_grads（走原路径），
  K>1 时 train_step 端到端可跑，且在 rng 无关的玩具模型上与 K=1 的 loss/梯度范数一致。

运行（CPU）：
  XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  PYTHONPATH=<worktree>/src:<worktree>/scripts/training uv run --no-sync python -m pytest \
  scripts/training/tests/test_grad_accum.py -q -s
"""

import os

# 必须在 import jax 之前设置，模拟 4 个 CPU 设备。
_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _flags:
    os.environ["XLA_FLAGS"] = (_flags + " --xla_force_host_platform_device_count=4").strip()
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import train as train_mod


# ── 玩具模型：小 MLP + MSE，无随机 ────────────────────────────────────────


class _Llm(nnx.Module):
    def __init__(self, rngs):
        self.proj = nnx.Linear(16, 8, rngs=rngs)


class _PaliGemma(nnx.Module):
    def __init__(self, rngs):
        self.llm = _Llm(rngs)


class ToyModel(nnx.Module):
    """仿 HistoryPi0 的最小接口：PaliGemma.llm 子树 + compute_loss(rng, obs, actions, train)。"""

    def __init__(self, rngs):
        self.inp = nnx.Linear(12, 16, rngs=rngs)
        self.PaliGemma = _PaliGemma(rngs)

    def __call__(self, x):
        h = jnp.tanh(self.inp(x))
        return self.PaliGemma.llm.proj(h)

    def compute_loss(self, rng, observation, actions, train=True):
        del rng, train  # 确定性：不用随机数
        pred = self(observation["x"])
        return jnp.mean((pred - actions) ** 2, axis=-1)  # 逐样本 loss，形状 (b,)


def _toy_batch(b, seed=0):
    r = np.random.default_rng(seed)
    obs = {"x": jnp.asarray(r.normal(size=(b, 12)), jnp.float32)}
    act = jnp.asarray(r.normal(size=(b, 8)), jnp.float32)
    return obs, act


def _full_grad(model, obs, act):
    def loss_fn(m, o, a):
        return jnp.mean(m.compute_loss(None, o, a))

    return nnx.value_and_grad(loss_fn)(model, obs, act)


def _accum_grad(model, obs, act, k):
    graphdef, state = nnx.split(model)

    def loss_fn(m, o, a):
        return jnp.mean(m.compute_loss(None, o, a))

    def micro_lg(i, micro):
        m = nnx.merge(graphdef, state)
        o, a = micro
        return nnx.value_and_grad(loss_fn)(m, o, a)

    return train_mod.accumulate_microbatch_grads(micro_lg, train_mod.split_microbatches((obs, act), k), k)


def _max_rel(a_tree, b_tree):
    worst = 0.0
    for a, b in zip(jax.tree.leaves(a_tree), jax.tree.leaves(b_tree), strict=True):
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        denom = max(np.max(np.abs(b)), 1e-30)
        worst = max(worst, float(np.max(np.abs(a - b)) / denom))
    return worst


# ── V2 线性性 ────────────────────────────────────────────────────────────


def test_grad_accum_linear():
    model = ToyModel(nnx.Rngs(0))
    obs, act = _toy_batch(64)
    full_loss, full_grads = nnx.jit(_full_grad)(model, obs, act)
    worst = 0.0
    for k in (1, 2, 4):
        loss, grads = nnx.jit(_accum_grad, static_argnums=3)(model, obs, act, k)
        worst = max(worst, _max_rel(grads, full_grads), _max_rel(loss, full_loss))
        for g, f in zip(jax.tree.leaves(grads), jax.tree.leaves(full_grads), strict=True):
            assert g.dtype == f.dtype
    verdict = "PASS" if worst < 1e-6 else "FAIL"
    print(f"\nGRAD_ACCUM_LINEAR={verdict} ks=2,4 max_rel={worst:.3e}")
    assert worst < 1e-6


# ── V3 切分 ──────────────────────────────────────────────────────────────


def test_micro_split():
    assert jax.device_count() == 4, f"需要 4 个模拟设备，实际 {jax.device_count()}"
    b, k, n_dev = 128, 2, 4
    mesh = sharding.make_mesh(1)  # (batch=4, fsdp=1)，与 --fsdp-devices 1 的数据轴切法一致
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    idx = jax.device_put(jnp.arange(b, dtype=jnp.int32), data_sharding)

    with sharding.set_mesh(mesh):
        micro = jax.jit(lambda x: train_mod.split_microbatches(x, k))(idx)
        # 每个 micro 再经 train_step 使用的同一约束，确认仍按数据轴均分
        micros = [jax.jit(sharding.activation_sharding_constraint)(micro[i]) for i in range(k)]

    seen = []
    per_device_counts = set()
    for i, m in enumerate(micros):
        vals = np.asarray(m)
        assert np.array_equal(vals, np.arange(i, b, k)), f"micro {i} 不是样本 {i}::{k}"
        devs = {}
        for shard in m.addressable_shards:
            devs.setdefault(shard.device.id, []).extend(np.asarray(shard.data).tolist())
        assert len(devs) == n_dev, f"micro {i} 只落在 {len(devs)} 个设备上"
        per_device_counts.update(len(v) for v in devs.values())
        seen.extend(vals.tolist())
    dup = len(seen) - len(set(seen))
    miss = len(set(range(b)) - set(seen))
    ok = dup == 0 and miss == 0 and per_device_counts == {b // (k * n_dev)}
    per_dev = per_device_counts.pop() if len(per_device_counts) == 1 else sorted(per_device_counts)
    print(f"\nMICRO_SPLIT={'PASS' if ok else 'FAIL'} k={k} per_device={per_dev} dup={dup} miss={miss}")
    assert ok


# ── train_step 分支 ──────────────────────────────────────────────────────


def _toy_state():
    model = ToyModel(nnx.Rngs(1))
    graphdef, params = nnx.split(model)
    tx = optax.adamw(1e-3)
    return training_utils.TrainState(
        step=jnp.int32(0), params=params, model_def=graphdef,
        opt_state=tx.init(params.filter(nnx.Param)), tx=tx, ema_decay=0.99, ema_params=params,
    )


def _toy_config(k):
    return types.SimpleNamespace(
        grad_accum_steps=k, trainable_filter=nnx.Param, model=types.SimpleNamespace(use_history=False)
    )


def _run_train_step(k):
    batch = _toy_batch(32, seed=3)
    with at.disable_typechecking():
        return jax.jit(lambda s, b: train_mod.train_step(_toy_config(k), jax.random.key(0), s, b))(_toy_state(), batch)


def test_k1_does_not_use_accum(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("K=1 不应调用梯度累积路径")

    monkeypatch.setattr(train_mod, "split_microbatches", _boom)
    monkeypatch.setattr(train_mod, "accumulate_microbatch_grads", _boom)
    _, info = _run_train_step(1)
    assert np.isfinite(float(info["loss"]))


def test_train_step_k2_matches_k1():
    s1, i1 = _run_train_step(1)
    s2, i2 = _run_train_step(2)
    for key in ("loss", "grad_norm", "llm_grad_norm"):
        rel = abs(float(i1[key]) - float(i2[key])) / max(abs(float(i1[key])), 1e-30)
        assert rel < 1e-5, f"{key}: K=1 {float(i1[key])} vs K=2 {float(i2[key])}"
    assert int(s2.step) == 1
    assert _max_rel(s2.params, s1.params) < 1e-5
    assert _max_rel(s2.ema_params, s1.ema_params) < 1e-5


def test_train_step_k2_under_fsdp_mesh():
    """在 4 设备 FSDP mesh 下跑 K=2，覆盖 micro / 累加器分片约束代码路径（玩具参数小于 4MiB 会被复制）。"""
    mesh = sharding.make_mesh(4)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    batch = jax.device_put(_toy_batch(32, seed=3), data_sharding)
    _, i1 = _run_train_step(1)
    with at.disable_typechecking(), sharding.set_mesh(mesh):
        _, i2 = jax.jit(lambda s, b: train_mod.train_step(_toy_config(2), jax.random.key(0), s, b))(_toy_state(), batch)
    rel = abs(float(i1["loss"]) - float(i2["loss"])) / abs(float(i1["loss"]))
    assert rel < 1e-5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
