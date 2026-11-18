"""S1 单测（计划 1117-e2e-enc-train-plan 二节 S1 行验收）：JAX encoder、vendored 参考、窗口清单、G1/G1c 冒烟。

运行（worktree 内 CPU）::

    JAX_PLATFORMS=cpu uv run --no-sync python -m pytest scripts/training/tests/test_e2e_encoder.py -q

需要 v1-store（P1 源 latent、P3 checkpoint）的用例在数据缺失时 skip。S2 的
``mme_vla_suite.shared.raw_bf16.raw9_f32_to_bf16`` 未合入时，G1b/G1c 用例注入同契约替身
（``np.float32 → ml_dtypes.bfloat16``，RNE）并以 warning 标明；合入后自动改用真模块。
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import types
import warnings

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch
from flax import nnx

WT = pathlib.Path(__file__).resolve().parents[3]
G0 = WT / "scripts/training/g0"
sys.path.insert(0, str(WT / "src"))
sys.path.insert(0, str(G0))
sys.path.insert(0, str(G0 / "ref_torch"))

import e2e_windows_v1 as W  # noqa: E402
import wan_latent_encoder_d3213de as REF  # noqa: E402
from mme_vla_suite.models.representation import wan_e2e_encoder as E  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
P3 = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/external/motionjepa/"
                  "wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt")
LATENTS = W.DEFAULT_DATASET_ROOT / "wan-latents"
need_ckpt = pytest.mark.skipif(not P3.exists(), reason="P3 checkpoint 不在盘")
need_latents = pytest.mark.skipif(not LATENTS.is_dir(), reason="P1 wan-latents 不在盘")


def _enc(depth=1, dtype=jnp.float32, seed=0):
    return E.WanLatentMotionEncoderJax(layers=depth, rngs=nnx.Rngs(seed), dtype=dtype)


def _ref(depth=1, seed=0):
    torch.manual_seed(seed)
    return REF.build_reference(depth, latents_mean=E.LATENTS_MEAN, latents_std=E.LATENTS_STD).eval()


def _x(n=3, seed=0):
    return (np.random.default_rng(seed).standard_normal((n, 9, 16384)) * 2.0).astype(np.float32)


@contextlib.contextmanager
def _raw_bf16_available():
    try:
        import mme_vla_suite.shared.raw_bf16  # noqa: F401
        yield
        return
    except ImportError:
        pass
    warnings.warn("S2 raw_bf16 未合入：注入同契约替身 raw9_f32_to_bf16（np.float32→ml_dtypes.bfloat16）")
    mod = types.ModuleType("mme_vla_suite.shared.raw_bf16")
    mod.raw9_f32_to_bf16 = lambda x: np.asarray(x, np.float32).astype(ml_dtypes.bfloat16)
    sys.modules["mme_vla_suite.shared.raw_bf16"] = mod
    try:
        yield
    finally:
        sys.modules.pop("mme_vla_suite.shared.raw_bf16", None)


# ── 结构 ───────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("in_dtype", [np.float32, ml_dtypes.bfloat16])
def test_shape(dtype, in_dtype):
    m = _enc(1, dtype)
    y = m(jnp.asarray(_x(5).astype(in_dtype)))
    assert y.shape == (5, 768) and y.dtype == dtype


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        _enc()(jnp.zeros((2, 8, 16384)))


@pytest.mark.parametrize("depth,want", [(1, 20_252_928), (8, 69_808_896)])
def test_param_count(depth, want):
    m = nnx.eval_shape(lambda: _enc(depth))
    assert E.count_params(m) == want
    assert int(sum(p.numel() for p in REF.build_reference(depth).parameters())) == want
    # 参数树只有 Param：depth=1 为 10 个叶（无标准化常量、无 RNG 状态）
    leaves = jax.tree_util.tree_leaves(nnx.state(m))
    assert len(leaves) == 3 + 7 * depth
    assert len(jax.tree_util.tree_leaves(nnx.state(m, nnx.Param))) == len(leaves)


def test_from_yaml_dict_config():
    cfg = {"layers": 1, "width": 768, "heads": 12, "mlp_ratio": 4, "dropout": 0.0, "norm": "vae_config"}
    m = E.WanLatentMotionEncoderJax(cfg, rngs=nnx.Rngs(0), dtype=jnp.bfloat16)
    assert m.depth == 1 and m.dtype == jnp.bfloat16 and len(m.blocks) == 1
    assert E.WanLatentMotionEncoderJax(cfg, rngs=nnx.Rngs(0), layers=8).depth == 8
    with pytest.raises(ValueError):
        E.WanLatentMotionEncoderJax({**cfg, "bogus": 1}, rngs=nnx.Rngs(0))


def test_swiglu_hidden_2048():
    assert E.swiglu_hidden(768, 4) == 2048
    m = _enc(1)
    ffn = m.blocks[0].ffn
    assert ffn.hidden == 2048
    assert ffn.gate_proj.kernel.value.shape == (768, 2048)
    assert ffn.up_proj.kernel.value.shape == (768, 2048)
    assert ffn.down_proj.kernel.value.shape == (2048, 768)
    assert REF.build_reference(1).blocks[0].ffn.gate_proj.weight.shape == (2048, 768)


def test_explicit_init_uniform_fan_in():
    m = _enc(1, seed=3)
    for lin, fan_in in [(m.input_proj, 16384), (m.blocks[0].attn.to_qkv, 768),
                        (m.blocks[0].ffn.down_proj, 2048), (m.output_proj, 768)]:
        k = np.asarray(lin.kernel.value)
        bound = 1.0 / np.sqrt(fan_in)
        assert lin.use_bias is False and lin.bias.value is None
        assert np.abs(k).max() <= bound * (1 + 1e-6)
        assert abs(k.std() / (bound / np.sqrt(3.0)) - 1) < 0.02      # U(±b) 的 std = b/√3（lecun_normal 为 1/√fan_in）
    for norm in (m.blocks[0].norm1, m.blocks[0].norm2, m.final_norm):
        assert np.array_equal(np.asarray(norm.weight.value), np.ones(768, np.float32))


def test_normalization_constants_nonstate_fp32():
    m = _enc(1)
    mean = np.asarray(E.LATENTS_MEAN, np.float32)
    std = np.asarray(E.LATENTS_STD, np.float32)
    assert np.abs(mean - np.asarray(E.VAE_CONFIG_LATENTS_MEAN, np.float32)).max() <= 5e-7
    assert np.abs(std - np.asarray(E.VAE_CONFIG_LATENTS_STD, np.float32)).max() <= 3.3e-5   # P8 舍入级
    paths = ["/".join(map(str, p)) for p, _ in jax.tree_util.tree_flatten_with_path(nnx.state(m))[0]]
    assert not any("latents" in p for p in paths)
    x = _x(2)
    z = np.asarray(m.normalize(jnp.asarray(x)))
    want = ((x.reshape(2, 9, 16, 1024) - mean[:, None]) / std[:, None]).reshape(2, 9, 16384)
    assert z.dtype == np.float32 and np.array_equal(z, want)
    # bf16 输入：先升 f32 再标准化，与 f32 输入（同值）逐位相同
    xb = x.astype(ml_dtypes.bfloat16)
    mb = _enc(1, jnp.bfloat16)
    assert np.array_equal(np.asarray(mb(jnp.asarray(xb))), np.asarray(mb(jnp.asarray(xb.astype(np.float32)))))


def test_constants_survive_bf16_param_restore():
    """参数树整体转 bf16 再写回（模拟 restore_params(dtype=bf16)）后，标准化常量仍是指定的 fp32 值。"""
    m = _enc(1, jnp.bfloat16)
    st = nnx.state(m, nnx.Param)
    nnx.update(m, jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16), st))
    assert m.latents_mean == E.LATENTS_MEAN and m.latents_std == E.LATENTS_STD
    assert len(jax.tree_util.tree_leaves(nnx.state(m))) == 10
    x = _x(1)
    z = np.asarray(m.normalize(jnp.asarray(x)))
    assert z.dtype == np.float32 and np.array_equal(z, np.asarray(_enc(1).normalize(jnp.asarray(x))))


@need_ckpt
def test_constants_equal_ckpt_buffers_bitwise():
    sd = REF.load_ckpt_encoder_state(P3)
    for k, ref in (("latents_mean", E.LATENTS_MEAN), ("latents_std", E.LATENTS_STD)):
        got = sd[k].numpy().astype(np.float32).ravel()
        assert np.array_equal(got.view(np.uint32), np.asarray(ref, np.float32).view(np.uint32))


def test_rmsnorm_order_matches_d3213de():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((4, 9, 768)) * 3).astype(ml_dtypes.bfloat16)
    w = (1 + 0.1 * rng.standard_normal(768)).astype(np.float32)
    norm = E.RMSNorm(768)
    norm.weight.value = jnp.asarray(w)
    y = np.asarray(norm(jnp.asarray(x)))
    x32 = x.astype(np.float32)
    r = 1.0 / np.sqrt(np.mean(x32 ** 2, -1, keepdims=True) + 1e-6)
    want = (x32 * r).astype(ml_dtypes.bfloat16).astype(np.float32) * w   # 归一化 → cast 回 bf16 → 乘 f32 weight
    assert y.dtype == np.float32                                           # bf16 × f32 → f32（同 torch 提升）
    assert np.abs(y - want).max() <= 1e-6 * np.abs(want).max() + np.abs(want).max() * 2 ** -8
    wrong = ((x32 * r) * w).astype(ml_dtypes.bfloat16).astype(np.float32)  # 先乘 weight 再 cast（错误顺序）
    assert not np.array_equal(y, wrong)
    tn = REF.RMSNorm(768)
    with torch.no_grad():
        tn.weight.copy_(torch.from_numpy(w))
        ty = tn(torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16)).float().numpy()
    ulp = np.abs(want) * 2 ** -7
    assert np.all(np.abs(y - ty) <= ulp + 1e-12)                           # 与 torch 至多 1 个 bf16 ulp
    assert (y == ty).mean() > 0.99


def test_causality():
    m = _enc(2, seed=5)
    x = _x(2, seed=7)

    def hidden(xx):
        h = m.input_proj(m.normalize(jnp.asarray(xx)).astype(m.dtype))
        cos, sin = E.rope_cos_sin(9, 64)
        for blk in m.blocks:
            h = blk(h, cos, sin)
        return np.asarray(h)

    h0 = hidden(x)
    for j in (0, 4, 8):
        x2 = x.copy()
        x2[:, j] += 5.0
        h1 = hidden(x2)
        assert np.array_equal(h0[:, :j], h1[:, :j]), f"改第 {j} 个 latent 影响了前 {j} 个位置"
        assert not np.allclose(h0[:, j], h1[:, j])
    # 读出只取最后位置：改任一位置都影响输出
    assert not np.allclose(np.asarray(m(jnp.asarray(x))), np.asarray(m(jnp.asarray(x2))))


@pytest.mark.parametrize("depth", [1, 2])
def test_small_forward_backward_parity_vs_torch(depth):
    import port_e2e_encoder_parity as P

    t = _ref(depth, seed=11)
    jm = P.build_jax_from_torch(t, depth, jnp.float32)
    x = _x(4, seed=2)
    xb = P.batches(x, 2)
    yt, gt = P.torch_forward_backward(t, xb, n_total=4)
    yj, gj = P.jax_forward_backward(jm, xb, n_total=4)
    assert P.rel_rmse(yj, yt) <= 1e-5
    assert sorted(gt) == sorted(gj) and len(gt) == 3 + 7 * depth
    assert max(P.rel_l2(gj[k], gt[k]) for k in gt) <= 1e-4


def test_state_dict_mapping_rejects_bad_keys():
    t = _ref(1)
    sd = dict(t.state_dict())
    m = _enc(1)
    sd_extra = {**sd, "blocks.1.norm1.weight": sd["blocks.0.norm1.weight"]}
    with pytest.raises(KeyError):
        E.load_torch_state_dict(m, sd_extra)
    sd_missing = {k: v for k, v in sd.items() if k != "final_norm.weight"}
    with pytest.raises(KeyError):
        E.load_torch_state_dict(m, sd_missing)
    sd_badstat = {**sd, "latents_std": sd["latents_std"] * 1.001}
    with pytest.raises(ValueError):
        E.load_torch_state_dict(m, sd_badstat)


# ── F1：参数树只有字符串键（openpi 真实权重加载器 flatten_dict(sep="/") 的前提）──────────────────────
@pytest.mark.parametrize("depth", [1, 8])
def test_param_tree_keys_are_str(depth):
    from flax.traverse_util import flatten_dict

    m = _enc(depth)
    flat = flatten_dict(nnx.state(m, nnx.Param).to_pure_dict(), sep="/")   # 整数键会在这里 TypeError
    arrays = {k: v for k, v in flat.items() if v is not None}
    want_blocks = {f"blocks/{E.block_attr(i)}" for i in range(depth)}
    got_blocks = {"/".join(k.split("/")[:2]) for k in arrays if k.startswith("blocks/")}
    assert got_blocks == want_blocks
    assert len(arrays) == 3 + 7 * depth and len(m.blocks) == depth
    assert [b is m.blocks[i] for i, b in enumerate(m.blocks)] == [True] * depth
    # torch 键映射与参数树一一对应：param_groups 覆盖全部数组叶，梯度布局键 = torch state dict 键
    groups = E.param_groups(m)
    assert sorted(groups) == sorted(["input_proj", "final_norm", "output_proj"] + [f"block{i}" for i in range(depth)])
    assert sum(len(g) for g in groups.values()) == len(arrays)
    grads = nnx.state(m, nnx.Param)
    tl = E.nnx_grads_to_torch_layout(m, grads)
    torch_keys = {k for k in REF.build_reference(depth).state_dict()
                  if not k.endswith(("cos_cache", "sin_cache", "latents_mean", "latents_std"))}
    assert set(tl) == torch_keys


def test_real_spec_tree_through_merge_params():
    """G4 失败路径：真实规格（mme_vla_suite_b128_80k + e2eenc YAML）的参数形状树经
    ``weight_loaders._merge_params`` 合并（只含非 encoder 键的合成 loaded），再按 train.py
    ``_load_weights_and_validate`` 做 ``check_pytree_equality``。"""
    import dataclasses

    from flax.traverse_util import flatten_dict
    from omegaconf import OmegaConf

    import openpi.shared.array_typing as at
    from mme_vla_suite.models.representation import percep_mem as pm
    from mme_vla_suite.training import config as _config
    from openpi.training import weight_loaders

    yaml = WT / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-e2eenc.yaml"
    cfg = _config.get_config("mme_vla_suite_b128_80k")
    mc = dataclasses.replace(cfg.model, history_config=OmegaConf.load(yaml), use_history=True)
    model = nnx.eval_shape(mc.create, jax.random.key(0))
    shape_tree = nnx.state(model).to_pure_dict()          # 同 train.py：train_state_shape.params.to_pure_dict()
    flat_ref = flatten_dict(shape_tree, sep="/")            # 改前在此 TypeError: sequence item 3 …
    enc_pre = f"mem_encoder/{pm.E2E_ENCODER_ATTR}/"
    enc = sorted(k for k, v in flat_ref.items() if k.startswith(enc_pre) and v is not None)
    assert len(enc) == 10 and sum(k.startswith(enc_pre + "blocks/block_0/") for k in enc) == 7
    loaded = {k: np.broadcast_to(np.zeros((), v.dtype), v.shape)
              for k, v in flat_ref.items() if v is not None and not k.startswith(enc_pre)}
    from flax.traverse_util import unflatten_dict
    merged = weight_loaders._merge_params(unflatten_dict(loaded, sep="/"), shape_tree, missing_regex=".*")
    at.check_pytree_equality(expected=shape_tree, got=merged, check_shapes=True, check_dtypes=True)
    fm = flatten_dict(merged, sep="/")
    assert all(isinstance(fm[k], jax.ShapeDtypeStruct) for k in enc)          # encoder 叶取自形状树（随机初值）
    assert all(isinstance(fm[k], np.ndarray) for k in loaded)                 # 其余叶取自 loaded
    print(f"\nE2E_MERGE_PARAMS=PASS ref_leaves={len(flat_ref)} loaded={len(loaded)} enc_from_init={len(enc)}")


# ── vendored 参考 ──────────────────────────────────────────────────────────────────────────────
def test_vendored_file_provenance():
    """vendored 正文逐字还原 d3213de：去掉文件头与附加段、把 mixin 块换回原 import 行后 sha256 等于原文件。"""
    import hashlib

    src = (G0 / "ref_torch/wan_latent_encoder_d3213de.py").read_text()
    assert "d3213de673d6b71b4ff46f6a568c9b79560231a9" in src
    body = src[src.index('"""Motion encoder for the v7'):]
    body = body[:body.index("\n\n# " + "=" * 93 + "\n# vendored 附加")]
    begin = "# ==== BEGIN vendored wan_latent_dit.py:35-65 (d3213de) ====\n"
    end = "# ==== END vendored wan_latent_dit.py:35-65 ====\n"
    pre, rest = body.split(begin)
    mixin, post = rest.split(end)
    assert hashlib.sha256(mixin.encode()).hexdigest() == \
        "e2e517eb53370ee4bc8b58e27c70894a12450c9d1b08b296a0c84f973f58fe8b"
    orig = pre + "from .wan_latent_dit import LatentAffineMixin\n" + post
    assert hashlib.sha256(orig.encode()).hexdigest() == \
        "bd8775d64356d224f9ad945dc908b4830041417fb05487e40ee8a5dcdcf6ea44"    # d3213de wan_latent_encoder.py


@need_ckpt
def test_vendored_strict_load_ckpt_depth8():
    model, missing, unexpected = REF.strict_load_ckpt(P3, depth=8)
    assert missing == [] and unexpected == []
    assert REF.count_params(model) == 69_808_896


# ── 窗口清单 ───────────────────────────────────────────────────────────────────────────────────
def test_windows_manifest_contract():
    m, sha = W.load_manifest(W.DEFAULT_MANIFEST)              # sha256 与提交记录（E2E_WINDOWS_V1_SHA256）一致
    assert sha == W.E2E_WINDOWS_V1_SHA256
    ws = m["windows"]
    assert len(ws) == m["num_windows"] >= 2000
    tasks = {w["task"] for w in ws}
    assert tasks == {"BinFill", "RouteStick", "VideoRepick", "VideoUnmaskSwap"}
    per_task = {t: sum(w["task"] == t for w in ws) for t in tasks}
    assert len(set(per_task.values())) == 1                   # 四任务等量分层
    for t in tasks:
        assert any(w["task"] == t and w["demopad"] for w in ws)  # 每任务都有 demopad 尾窗
        assert any(w["task"] == t and w["kind"] == "exec" for w in ws)
    assert sum(w["task"] == "BinFill" and w["kind"] == "demo" for w in ws) >= 100   # BinFill 带 demo 样本
    for w in ws:
        assert w["demopad"] == (w["kind"] == "demo" and w["pad_frames"] > 0)
        assert w["byte_offset"] == w["m"] * W.CHUNK_BYTES
        assert set(m["fields"]) <= set(w)
    rows = [w["table_row"] for w in ws]
    assert rows == sorted(rows) and len(set(rows)) == len(rows)
    assert all(0 <= r < m["table_rows_768"] for r in rows)
    assert [w["idx"] for w in ws] == list(range(len(ws)))
    with pytest.raises(RuntimeError):
        W.load_manifest(W.DEFAULT_MANIFEST, expect_sha="0" * 64)


@need_latents
def test_windows_manifest_reads_source():
    m, _ = W.load_manifest()
    sel = W.select_windows(m, 8)
    assert {w["task"] for w in sel} == {"BinFill", "RouteStick", "VideoRepick", "VideoUnmaskSwap"}
    x = W.read_window_latents(LATENTS, sel, verify=True)      # 逐块 sha256 校验
    assert x.shape == (8, 9, 16384) and x.dtype == np.float32 and np.isfinite(x).all()
    idx = json.loads((W.DEFAULT_DATASET_ROOT / "motion/meta/motion_index.json").read_text())
    e = next(e for e in idx["entries"] if e["g"] == sel[0]["g"])
    assert e[sel[0]["kind"]]["row_base"] + sel[0]["m"] == sel[0]["table_row"]


# ── G1 / G1b / G1c 冒烟（CPU 小批）──────────────────────────────────────────────────────────────
def _run_main(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(argv)
    return rc, buf.getvalue()


@need_latents
def test_g1c_negatives_fail_as_expected_cpu():
    import e2e_bf16_ref_parity as G1C

    with _raw_bf16_available():
        rc, out = _run_main(G1C.main, ["--max-windows", "16", "--batch", "8"])
    line = [ln for ln in out.splitlines() if ln.startswith("E2E_BF16_REF=")][0]
    assert "neg_x2=FAIL_AS_EXPECTED" in line and "neg_stopgrad=FAIL_AS_EXPECTED" in line, out
    assert line.startswith("E2E_BF16_REF=PASS") and rc == 0, out


@need_latents
def test_g1_depth1_seed_and_g1b_cpu_smoke():
    import port_e2e_encoder_parity as P

    with _raw_bf16_available():
        rc, out = _run_main(P.main, ["--depth", "1", "--init", "seed", "--max-windows", "8", "--batch", "8"])
    assert rc == 0, out
    assert any(ln.startswith("E2E_PORT_PARITY=PASS windows=8 ") and "depth=1 init=seed params=20252928 "
               "fwd_rel_rmse<=1e-4 grad_rel_rmse<=1e-3" in ln for ln in out.splitlines()), out
    assert any(ln.startswith("E2E_BF16_FLOOR=RECORDED f=") for ln in out.splitlines()), out


@need_latents
@need_ckpt
def test_g1_depth8_ckpt_cpu_smoke():
    import port_e2e_encoder_parity as P

    rc, out = _run_main(P.main, ["--depth", "8", "--init", "ckpt", "--max-windows", "8", "--batch", "8"])
    assert rc == 0, out
    assert any(ln.startswith("E2E_PORT_PARITY=PASS windows=8 ") and "depth=8 init=ckpt params=69808896" in ln
               for ln in out.splitlines()), out
