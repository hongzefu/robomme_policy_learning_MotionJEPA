"""--resume-full 完整续跑状态的 CPU 单测（G4C-S4）。

覆盖：
- 玩具 TrainState（含 optax adamw opt_state、EMA）在 4 卡 FSDP mesh 上 save→restore 逐位一致、sharding 一致；
- manifest 完整性；原子替换：第二次保存中途失败时旧 resume/ 原样保留、仍可还原；成功后只留 1 份；
- 缺件 / sha256 被改 / schema 错 / 关键字段不一致 / 叶子结构不一致 时 fail-loud；
- 旧 --resume / --overwrite 仍报错；--resume-full 与 --resum-ckpt-id 互斥；全新 run 的 dataloader 种子不变。

运行（CPU）：
  XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  PYTHONPATH=<worktree>/src:<worktree>/scripts/training uv run --no-sync python -m pytest \
  scripts/training/tests/test_resume_full.py -q -s
末尾打印判定行 RESUME_UNIT=PASS tests=<n>。
"""

import os

# 必须在 import jax 之前设置，模拟 4 个 CPU 设备。
_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _flags:
    os.environ["XLA_FLAGS"] = (_flags + " --xla_force_host_platform_device_count=4").strip()
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import ast
import dataclasses
import json
import pathlib
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import openpi.shared.array_typing as at
import openpi.training.checkpoints as ckpts
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import train as train_mod

# 每个测试函数体跑到最后一行才登记；模块结束时全部登记才判 PASS
_PASSED = []
_N_TESTS = 4


def teardown_module(module):
    verdict = "PASS" if len(_PASSED) == _N_TESTS else "FAIL"
    print(f"\nRESUME_UNIT={verdict} tests={len(_PASSED)}")


# ── 玩具模型与 TrainState ─────────────────────────────────────────────────


class _Llm(nnx.Module):
    def __init__(self, rngs):
        self.proj = nnx.Linear(64, 32, rngs=rngs)


class _PaliGemma(nnx.Module):
    def __init__(self, rngs):
        self.llm = _Llm(rngs)


class ToyModel(nnx.Module):
    def __init__(self, rngs):
        self.inp = nnx.Linear(16, 64, rngs=rngs)
        self.PaliGemma = _PaliGemma(rngs)

    def __call__(self, x):
        return self.PaliGemma.llm.proj(jnp.tanh(self.inp(x)))

    def compute_loss(self, rng, observation, actions, train=True):
        del rng, train
        return jnp.mean((self(observation["x"]) - actions) ** 2, axis=-1)


CONFIG_FIELDS = {"batch_size": 32, "grad_accum_steps": 2, "num_train_steps": 100, "seed": 7, "exp_name": "toy"}


def _toy_state(seed):
    model = ToyModel(nnx.Rngs(seed))
    graphdef, params = nnx.split(model)
    tx = optax.adamw(1e-3)
    return training_utils.TrainState(
        step=jnp.int32(0), params=params, model_def=graphdef,
        opt_state=tx.init(params.filter(nnx.Param)), tx=tx, ema_decay=0.99, ema_params=params,
    )


def _mesh_and_sharding(state):
    mesh = sharding.make_mesh(4)  # (batch=1, fsdp=4)
    shape = jax.eval_shape(lambda: state)
    # 玩具参数很小，min_size_mbytes=0 强制按 FSDP 轴切分以覆盖多卡分片路径
    return mesh, sharding.fsdp_sharding(shape, mesh, min_size_mbytes=0)


def _trained_state():
    """FSDP 下跑 3 步 train_step，得到非零 AdamW 动量与 step=3 的状态。"""
    state = _toy_state(0)
    mesh, st_sharding = _mesh_and_sharding(state)
    state = jax.device_put(state, st_sharding)
    cfg = types.SimpleNamespace(grad_accum_steps=1, trainable_filter=nnx.Param,
                                model=types.SimpleNamespace(use_history=False))
    r = np.random.default_rng(0)
    rng = jax.random.key(CONFIG_FIELDS["seed"])
    with at.disable_typechecking(), sharding.set_mesh(mesh):
        step_fn = jax.jit(lambda s, b: train_mod.train_step(cfg, rng, s, b), out_shardings=(st_sharding, None))
        for _ in range(3):
            batch = ({"x": jnp.asarray(r.normal(size=(8, 16)), jnp.float32)},
                     jnp.asarray(r.normal(size=(8, 32)), jnp.float32))
            state, _ = step_fn(state, batch)
    return state, st_sharding, rng


def _fresh_target(state):
    """与 state 同结构、同 sharding、但值全不同的还原目标（模拟 main 里新初始化的 train_state）。"""
    return jax.tree.map(lambda x: jax.device_put(np.full(x.shape, 0.5, x.dtype), x.sharding), state)


def _assert_bitwise(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert np.array_equal(np.asarray(x), np.asarray(y)), "还原值与保存值不逐位一致"
        assert x.sharding.is_equivalent_to(y.sharding, x.ndim), "还原 sharding 与保存时不一致"


# ── 逐位一致 ─────────────────────────────────────────────────────────────


def test_roundtrip_bitwise(tmp_path):
    assert jax.device_count() == 4
    state, st_sharding, rng = _trained_state()
    assert int(state.step) == 3
    # 至少一个叶子真的跨 4 卡分片
    assert any(len({s.device for s in x.addressable_shards}) == 4 and x.addressable_shards[0].data.size < x.size
               for x in jax.tree.leaves(state))
    # AdamW 动量非零（证明存的是训练后的 opt_state）
    assert any(float(jnp.abs(x).sum()) > 0 for x in jax.tree.leaves(state.opt_state) if x.ndim > 0)

    final = ckpts.save_resume_state(tmp_path, state, 3, config_fields=CONFIG_FIELDS, rng=rng)
    assert sorted(os.listdir(tmp_path)) == ["resume"]
    manifest = json.loads((pathlib.Path(final) / "manifest.json").read_text())
    assert manifest["schema"] == ckpts.RESUME_SCHEMA and manifest["step"] == 3
    assert manifest["config"] == CONFIG_FIELDS
    assert manifest["num_leaves"] == len(jax.tree.leaves(state)) == len(manifest["leaves"])
    assert manifest["files_sha256"] and all(len(v) == 64 for v in manifest["files_sha256"].values())
    assert manifest["rng"]["key_data"] == np.asarray(jax.random.key_data(rng)).tolist()

    target = _fresh_target(state)
    restored, m2 = ckpts.restore_resume_state(tmp_path, target, expected_config=CONFIG_FIELDS)
    assert m2["step"] == 3 and int(restored.step) == 3
    _assert_bitwise(restored, state)
    # ShapeDtypeStruct 目标同样可用（train.py::main 的用法）
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), target)
    restored2, _ = ckpts.restore_resume_state(tmp_path, abstract, expected_config=CONFIG_FIELDS)
    _assert_bitwise(restored2, state)
    _PASSED.append("roundtrip_bitwise")


# ── 原子替换 ─────────────────────────────────────────────────────────────


def test_atomic_replace_on_failure(tmp_path, monkeypatch):
    state, st_sharding, rng = _trained_state()
    ckpts.save_resume_state(tmp_path, state, 3, config_fields=CONFIG_FIELDS, rng=rng)
    before = json.loads((tmp_path / "resume" / "manifest.json").read_text())

    # 第二次保存（step=4）在 state 写完、manifest 写之前失败
    state4 = dataclasses.replace(state, step=state.step + 1)
    real_sha = ckpts._sha256_tree

    def _boom(root):
        raise OSError("模拟磁盘写满")

    monkeypatch.setattr(ckpts, "_sha256_tree", _boom)
    with pytest.raises(OSError):
        ckpts.save_resume_state(tmp_path, state4, 4, config_fields=CONFIG_FIELDS, rng=rng)
    monkeypatch.setattr(ckpts, "_sha256_tree", real_sha)

    # 旧 resume/ 原样保留且仍可校验、还原
    after = json.loads((tmp_path / "resume" / "manifest.json").read_text())
    assert after == before and after["step"] == 3
    restored, _ = ckpts.restore_resume_state(tmp_path, _fresh_target(state), expected_config=CONFIG_FIELDS)
    _assert_bitwise(restored, state)
    assert any(n.startswith("resume.tmp-") for n in os.listdir(tmp_path))  # 半份只留在 tmp

    # 下次成功保存清掉半份、替换旧份，只剩 1 份
    ckpts.save_resume_state(tmp_path, state4, 4, config_fields=CONFIG_FIELDS, rng=rng)
    assert sorted(os.listdir(tmp_path)) == ["resume"]
    assert json.loads((tmp_path / "resume" / "manifest.json").read_text())["step"] == 4

    # step 参数与 state.step 不一致即拒绝
    with pytest.raises(ckpts.ResumeStateError):
        ckpts.save_resume_state(tmp_path, state4, 9, config_fields=CONFIG_FIELDS, rng=rng)
    _PASSED.append("atomic_replace_on_failure")


# ── fail-loud ────────────────────────────────────────────────────────────


def test_fail_loud(tmp_path):
    state, st_sharding, rng = _trained_state()
    # 缺 manifest
    with pytest.raises(ckpts.ResumeStateError, match="不存在"):
        ckpts.read_resume_manifest(tmp_path, expected_config=CONFIG_FIELDS)

    ckpts.save_resume_state(tmp_path, state, 3, config_fields=CONFIG_FIELDS, rng=rng)
    mpath = tmp_path / "resume" / "manifest.json"
    good = mpath.read_text()

    # 关键字段逐项不一致
    for key, val in (("batch_size", 64), ("grad_accum_steps", 1), ("num_train_steps", 50),
                     ("seed", 8), ("exp_name", "other")):
        with pytest.raises(ckpts.ResumeStateError, match=key):
            ckpts.read_resume_manifest(tmp_path, expected_config={**CONFIG_FIELDS, key: val})

    # schema 错 / 缺字段
    bad = json.loads(good)
    bad["schema"] = "v0"
    mpath.write_text(json.dumps(bad))
    with pytest.raises(ckpts.ResumeStateError, match="schema"):
        ckpts.read_resume_manifest(tmp_path, expected_config=CONFIG_FIELDS)
    bad = json.loads(good)
    del bad["files_sha256"]
    mpath.write_text(json.dumps(bad))
    with pytest.raises(ckpts.ResumeStateError, match="files_sha256"):
        ckpts.read_resume_manifest(tmp_path, expected_config=CONFIG_FIELDS)
    mpath.write_text(good)

    # 叶子结构不一致（目标少一个子树）
    other = _toy_state(5)
    other = dataclasses.replace(other, ema_params=None)
    with pytest.raises(ckpts.ResumeStateError, match="叶子结构"):
        ckpts.restore_resume_state(tmp_path, other, expected_config=CONFIG_FIELDS)

    # state 文件内容被改 → sha256 失败
    files = sorted(p for p in (tmp_path / "resume" / "state").rglob("*") if p.is_file())
    victim = max(files, key=lambda p: p.stat().st_size)
    data = bytearray(victim.read_bytes())
    data[-1] ^= 0xFF
    victim.write_bytes(bytes(data))
    with pytest.raises(ckpts.ResumeStateError, match="sha256"):
        ckpts.read_resume_manifest(tmp_path, expected_config=CONFIG_FIELDS)
    # 文件缺失 → sha256 失败
    victim.unlink()
    with pytest.raises(ckpts.ResumeStateError, match="sha256"):
        ckpts.read_resume_manifest(tmp_path, expected_config=CONFIG_FIELDS)
    _PASSED.append("fail_loud")


# ── 入口护栏与种子 ───────────────────────────────────────────────────────


def _flags(**kw):
    base = dict(overwrite=False, resume=False, resume_full=False, resum_ckpt_id=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_entry_guards_and_seed():
    with pytest.raises(ValueError, match="--overwrite / --resume"):
        train_mod.check_resume_flags(_flags(resume=True))
    with pytest.raises(ValueError, match="--overwrite / --resume"):
        train_mod.check_resume_flags(_flags(overwrite=True))
    with pytest.raises(ValueError, match="--overwrite / --resume"):
        train_mod.check_resume_flags(_flags(resume=True, resume_full=True))
    with pytest.raises(ValueError, match="resum-ckpt-id"):
        train_mod.check_resume_flags(_flags(resume_full=True, resum_ckpt_id=1000))
    train_mod.check_resume_flags(_flags(resume_full=True))
    train_mod.check_resume_flags(_flags())

    # TrainConfig 只新增 resume_full，默认 False
    import mme_vla_suite.training.config as cfg_mod
    fields = {f.name: f for f in dataclasses.fields(cfg_mod.TrainConfig)}
    assert fields["resume_full"].default is False

    # 全新 run（start_step=0）种子即 config.seed；续跑按步号偏移
    assert train_mod.resume_effective_seed(42, 0) == 42
    assert train_mod.resume_effective_seed(42, 5001) == 42 + 1000003 * 5001

    # main 中 create_data_loader 的 seed 来自 data_seed，且非 resume_full 时 data_seed = config.seed
    src = pathlib.Path(train_mod.__file__).read_text()
    tree = ast.parse(src)
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main_fn) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "create_data_loader"]
    assert len(calls) == 1
    seed_kw = next(k for k in calls[0].keywords if k.arg == "seed")
    assert isinstance(seed_kw.value, ast.Name) and seed_kw.value.id == "data_seed"
    assert "data_seed = config.seed" in src
    _PASSED.append("entry_guards_and_seed")
