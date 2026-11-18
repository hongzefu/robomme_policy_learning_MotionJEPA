"""S3 训练侧单测（计划 1117-e2e-enc-train-plan.html 第二部分「二 子代理分配表」S3 行验收）。

覆盖：
① e2e YAML 与 MotionMEM YAML 只差 motion 节的约定键；
② R3 关闭态：wantp768 / wanraw9s32 / MotionMEM YAML 的参数树与 v11 原锚点（叶数 65、树摘要）一致，e2e 只多 encoder 叶；
③ inputs_spec：e2e 下 motion_emb (b,1440,16384) bf16、motion_pos/mask 160 槽、mem_order 672；
④ history_observation 的 e2e 形状 typecheck（并回归 raw9 720 与 768 的 160）；
⑤ ``_fill_motion_e2e``：k = 0 / 1 / 141 / 160 四档，普通取数与预分配取数 shape/dtype/全部字节/尾部零一致，
   预分配路径缓冲地址不变；表行字节原样（bf16，不转换）；pos/mask/mem_order 与 768 路径同式；
⑥ L1：``TransformedDataset.__getitems__`` 按数据集 ``raw9_motion_dtype`` 整批 bf16 预分配，共享内存往返字节不变；
⑦ 768 分支不被 e2e YAML 误入：真实 framesamp 库 + 768 表上，e2e YAML 构造 FrameSampDataset 与 ``_motion_gates`` 都 raise；
⑧ ``_e2e_motion_gate`` 配对：kind/readout/table_kind/layout/dtype/几何任一不符即 raise；
⑨ train.py：``init_history_config`` 的 e2e 分支（真实入口，MotionMeta 用夹具）、诊断键、逐步裁剪日志、步 0 输出；
⑩ 标准化常量：保存 → bf16 恢复 → 合并回模型后仍为 fp32 指定值，严格参数树无多余叶。

S1 的 ``wan_e2e_encoder`` 未合入时，注册同名最小替身（只验 S3 的接线，替身有 ``_IS_TEST_STUB``）；
S2 的 bf16 布局未合入时，依赖真表布局的用例跳过并在汇报中列出。
运行：``JAX_PLATFORMS=cpu uv run --no-sync python -m pytest scripts/training/tests/test_e2e_dataset.py -q -s``
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import types
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import ml_dtypes
import numpy as np
import pytest

WT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "src"))
sys.path.insert(0, str(WT / "scripts" / "training"))
YAML_DIR = WT / "src/mme_vla_suite/models/config/robomme"
E2E_YAML = YAML_DIR / "perceptual-framesamp-modul-8frame-8x8-e2eenc.yaml"
MOTION_YAML = YAML_DIR / "perceptual-framesamp-modul-8frame-8x8-motion.yaml"
V11_YAML = YAML_DIR / "perceptual-framesamp-modul-8frame-8x8-wantp768.yaml"
RAW9_YAML = YAML_DIR / "perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml"
V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store"))
LIB = V1 / "datasets/4task-v2-1600ep-604f16da"
BF16 = np.dtype(ml_dtypes.bfloat16)

# v11 原锚点（test_raw9_dataset.py：e59cde3 = c55a6bc 代码上 wantp768.yaml 的 nnx.eval_shape 参数树摘要）
V11_N_LEAVES = 65
V11_TREE_SHA256 = "51f537d1b96c40af546bee0fda0c3e3982c4c4b1f8af6716b2e66b4022e38020"

# ── S1 encoder 替身（仅在 S1 未合入时注册）──────────────────────────────────────────────────────
_STUB_SRC = '''"""测试用最小替身：与 S1 WanLatentMotionEncoderJax 同名（1117 计划一节 wan_e2e_encoder.py 行；只验 S3 接线）。"""
import flax.nnx as nnx
import jax.numpy as jnp
_IS_TEST_STUB = True
STUB_MEAN = tuple(0.05 * c - 0.3 for c in range(16))
STUB_STD = tuple(1.0 + 0.1 * c for c in range(16))
_init = nnx.initializers.variance_scaling(1 / 3, "fan_in", "uniform")


class _Norm(nnx.Module):
    def __init__(self, dim):
        self.weight = nnx.Param(jnp.ones((dim,), jnp.float32))

    def __call__(self, x):
        x32 = x.astype(jnp.float32)
        y = (x32 * jnp.reciprocal(jnp.sqrt(jnp.mean(x32 * x32, -1, keepdims=True) + 1e-6))).astype(x.dtype)
        return y * self.weight


class _Block(nnx.Module):
    def __init__(self, dim, dtype, rngs):
        self.norm1 = _Norm(dim)
        self.ffn = nnx.Linear(dim, dim, use_bias=False, kernel_init=_init, dtype=dtype, rngs=rngs)

    def __call__(self, x):
        return x + self.ffn(self.norm1(x))


class WanLatentMotionEncoderJax(nnx.Module):
    def __init__(self, depth, dim=768, heads=12, dropout=0.0, dtype=jnp.float32, *, rngs):
        self.dtype = dtype
        self.mean = STUB_MEAN          # 非状态常量（Python 元组，不进参数树）
        self.std = STUB_STD
        self.input_proj = nnx.Linear(16384, dim, use_bias=False, kernel_init=_init, dtype=dtype, rngs=rngs)
        self.blocks = [_Block(dim, dtype, rngs) for _ in range(int(depth))]
        self.final_norm = _Norm(dim)
        self.output_proj = nnx.Linear(dim, dim, use_bias=False, kernel_init=_init, dtype=dtype, rngs=rngs)

    def __call__(self, z):
        n, t = z.shape[0], z.shape[1]
        x = z.astype(jnp.float32).reshape(n, t, 16, 1024)
        m = jnp.asarray(self.mean, jnp.float32)[None, None, :, None]
        s = jnp.asarray(self.std, jnp.float32)[None, None, :, None]
        x = ((x - m) / s).reshape(n, t, 16384).astype(self.dtype)
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)
        x = self.final_norm(x[:, -1])
        return self.output_proj(x)
'''


def install_encoder_stub() -> bool:
    """S1 模块不存在时注册替身；返回 True 表示用的是真模块。"""
    name = "mme_vla_suite.models.representation.wan_e2e_encoder"
    try:
        mod = importlib.import_module(name)
        return not getattr(mod, "_IS_TEST_STUB", False)
    except ModuleNotFoundError:
        import mme_vla_suite.models.representation as pkg
        stub = types.ModuleType(name)
        exec(compile(_STUB_SRC, "<wan_e2e_encoder stub>", "exec"), stub.__dict__)
        sys.modules[name] = stub
        pkg.wan_e2e_encoder = stub
        return False


import importlib  # noqa: E402

S1_REAL = install_encoder_stub()

from mme_vla_suite.datastore import motion_store as ms  # noqa: E402
from mme_vla_suite.models.representation import percep_mem as pm  # noqa: E402
from mme_vla_suite.shared.sampling import even_sampling_indices, memory_order, pad_times  # noqa: E402

S2_LAYOUT = pm.E2E_LAYOUT in getattr(ms, "LAYOUT_SPECS", {})


def _say(capsys, line: str) -> None:
    with capsys.disabled():
        print(f"\n{line}")


# ══ ① YAML ════════════════════════════════════════════════════════════════════════════════════


def test_yaml_only_motion_keys_differ(capsys):
    from omegaconf import OmegaConf

    a = OmegaConf.to_container(OmegaConf.load(MOTION_YAML))
    b = OmegaConf.to_container(OmegaConf.load(E2E_YAML))
    ma, mb = a.pop("motion"), b.pop("motion")
    assert a == b
    assert ma.pop("store_path") == "v1-store/datasets/4task-v2-1600ep-604f16da/motion"
    assert ma.pop("source_run").endswith("#encoder")
    want = {"store_path": "v1-store/datasets/4task-v2-1600ep-604f16da/motion-wan-raw9-s16",
            "encoder_kind": "wan_e2e", "readout": "wan_e2e_enc768_s16_v1", "raw_dim": 16384,
            "latents_per_window": 9,
            "encoder": {"layers": 1, "width": 768, "heads": 12, "mlp_ratio": 4, "dropout": 0.0, "norm": "vae_config"}}
    for k, v in want.items():
        assert mb.pop(k) == v, k
    assert "tokens_per_window" not in mb
    assert ma == mb                      # dim 768 / budget 160 / stride 16 / 其余键逐值不变
    assert (mb["dim"], mb["budget"], mb["stride"]) == (768, 160, 16)
    _say(capsys, "E2E_YAML=PASS only_motion_keys=1 tokens_per_window=absent layers=1")


# ══ ② ③ 参数树、R3、inputs_spec ═══════════════════════════════════════════════════════════════


def _tree(yaml_path):
    import jax
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict
    from omegaconf import OmegaConf
    from mme_vla_suite.training import config as _config

    hc = OmegaConf.load(yaml_path)
    cfg = _config.get_config("mme_vla_suite_b128_80k")
    mc = dataclasses.replace(cfg.model, history_config=hc, use_history=True)
    model = nnx.eval_shape(mc.create, jax.random.key(0))
    flat = {k: v for k, v in flatten_dict(nnx.state(model, nnx.Param).to_pure_dict()).items() if v is not None}
    lines = sorted("/".join(map(str, k)) + ":" + str(tuple(v.shape)) + ":" + str(v.dtype) for k, v in flat.items())
    return hc, mc, model, flat, lines


def _sha(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def test_r3_closed_trees_hit_v11_anchor(capsys):
    _, _, m11, _, l11 = _tree(V11_YAML)
    assert len(l11) == V11_N_LEAVES and _sha(l11) == V11_TREE_SHA256
    assert m11.mem_encoder.motion_e2e is False and not hasattr(m11.mem_encoder, pm.E2E_ENCODER_ATTR)
    _, _, mm, _, lm = _tree(MOTION_YAML)
    assert lm == l11                     # MotionMEM 与 v11 的树同形（两者 motion 节只差 store/kind）
    assert not hasattr(mm.mem_encoder, pm.E2E_ENCODER_ATTR)
    _, _, mr, _, lr = _tree(RAW9_YAML)
    swapped = sorted(l.replace("mem_encoder/motion_encoder_static/kernel:(17152, 1024)",
                               "mem_encoder/motion_encoder_static/kernel:(1536, 1024)") for l in lr)
    assert _sha(swapped) == V11_TREE_SHA256 and not hasattr(mr.mem_encoder, pm.E2E_ENCODER_ATTR)
    _say(capsys, f"R3_TREE=PASS v11_leaves={len(l11)} sha={_sha(l11)[:16]} motionmem=same raw9=same(except kernel)")


def test_e2e_tree_adds_only_encoder_and_specs(capsys):
    from mme_vla_suite.models.integration.history_pi0 import _motion_specs

    hc, mc, model, flat, lines = _tree(E2E_YAML)
    _, _, _, _, lm = _tree(MOTION_YAML)
    pre = f"mem_encoder/{pm.E2E_ENCODER_ATTR}/"
    enc = [l for l in lines if l.startswith(pre)]
    rest = [l for l in lines if not l.startswith(pre)]
    assert enc and rest == lm, "e2e 树去掉 encoder 叶后须与 MotionMEM 树完全相同"
    assert all("img" not in l.split(":")[0] and "llm" not in l.split(":")[0] for l in enc)
    n_enc = sum(int(np.prod(v.shape)) for k, v in flat.items() if k[:2] == ("mem_encoder", pm.E2E_ENCODER_ATTR))
    if S1_REAL:
        assert n_enc == pm.E2E_POLICY_ENCODER_PARAMS, n_enc
    me = model.mem_encoder
    assert (me.motion_budget, me.motion_tokens, me.motion_dim, me.motion_emb_len, me.motion_emb_dim, me.motion_lpw) == \
        (160, 160, 768, 1440, 16384, 9)
    sp = _motion_specs(hc, 4)
    assert {k: (tuple(v.shape), str(v.dtype)) for k, v in sp.items()} == {
        "motion_emb": ((4, 1440, 16384), "bfloat16"), "motion_pos": ((4, 160, 256), "float32"),
        "motion_mask": ((4, 160), "bool"), "mem_order": ((4, 672), "int32")}
    # 冻结过滤器不命中 encoder 叶
    import flax.nnx as nnx
    frozen = nnx.state(model, nnx.All(nnx.Param, mc.get_freeze_filter()))
    from flax.traverse_util import flatten_dict
    assert not any(k[:2] == ("mem_encoder", pm.E2E_ENCODER_ATTR) for k in flatten_dict(frozen.to_pure_dict()))
    _say(capsys, f"E2E_PARAM_TREE=PASS enc_leaves={len(enc)} enc_params={n_enc} s1_real={int(S1_REAL)} "
                 f"rest_equals_motionmem=1 spec_motion_emb=(4,1440,16384,bf16)")


# ══ ④ history_observation typecheck ═══════════════════════════════════════════════════════════


def _obs(yaml_path, b=2, pos_len=None):
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from mme_vla_suite.training import config as _config

    hc = OmegaConf.load(yaml_path)
    mc = dataclasses.replace(_config.get_config("mme_vla_suite_b128_80k").model, history_config=hc, use_history=True)
    spec, _ = mc.inputs_spec(batch_size=b)
    arr = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), spec)
    fields = {f.name: getattr(arr, f.name) for f in dataclasses.fields(arr)}
    if pos_len is not None:
        fields["motion_pos"] = jnp.zeros((b, pos_len, 256), jnp.float32)
    return HistAugObservation(**fields)


def test_history_observation_typecheck(capsys):
    import jaxtyping

    o = _obs(E2E_YAML)
    assert o.motion_emb.shape == (2, 1440, 16384) and str(o.motion_emb.dtype) == "bfloat16"
    assert o.motion_pos.shape == (2, 160, 256) and o.motion_mask.shape == (2, 160)
    r = _obs(RAW9_YAML)
    assert r.motion_emb.shape == (2, 720, 16384) and r.motion_pos.shape == (2, 720, 256)
    m = _obs(MOTION_YAML)
    assert m.motion_emb.shape == (2, 160, 768)
    with pytest.raises((jaxtyping.TypeCheckError, TypeError, ValueError)):
        _obs(E2E_YAML, pos_len=161)       # pos 与 mask 仍共用 l4：161 ≠ 160 必拒
    _say(capsys, "E2E_OBS_TYPECHECK=PASS e2e=(1440,16384)/160/160 raw9=720/720/720 m768=160/160/160 neg=rejected")


# ══ ⑤ _fill_motion_e2e ════════════════════════════════════════════════════════════════════════

ROW_SHAPE = (9, 16, 32, 32)
FX_EPISODES = ((60, 10), (2600, 1000))     # (num_timesteps, exec_start_idx)：g0 无 demo 窗；g1 demo 62 窗 + exec 98 窗


class _BF16Store:
    def __init__(self, table):
        self._t = table

    def rows_into(self, rows, out):
        r = np.asarray(rows, dtype=np.int64)
        if out.shape != (len(r),) + ROW_SHAPE or out.dtype != BF16:
            raise ValueError(f"rows_into out {out.shape} {out.dtype}")
        out[...] = self._t[r]
        return out


class _PosStore:
    def __init__(self, pos):
        self._pos = pos

    def pos_rows(self, frames):
        return self._pos[np.asarray(frames, dtype=np.int64)]


@pytest.fixture(scope="module")
def fx():
    rng = np.random.default_rng(3)
    spec = SimpleNamespace(demo_min_real=17, exec_min_real=33, demo_tail_pad="repeat_last", grid_stride=16,
                           row_shape=ROW_SHAPE)
    entries, cursor = [], 0
    for g, (nt, es) in enumerate(FX_EPISODES):
        segs = {}
        for seg, seg_len, min_real in (("demo", es, 17), ("exec", nt - es, 33)):
            n_chunks = max(0, seg_len - (min_real - 1))
            n16 = len(range(0, n_chunks, 16))
            segs[seg] = ms.SegmentInfo(row_base=cursor if n16 else None, num_grid=n16, num_chunks=n_chunks,
                                       seg_len=seg_len)
            cursor += n16
        entries.append(SimpleNamespace(g=g, h5_file=f"Fix{g}.h5", raw_ep_idx=0, num_timesteps=nt, exec_start_idx=es,
                                       demo=segs["demo"], exec=segs["exec"], spec=spec))
    # 正态数值的 bf16 表（训练冒烟要前向数值合理）；取数只搬运字节，按字节比对
    table = rng.standard_normal((cursor,) + ROW_SHAPE).astype(BF16)
    pos = rng.standard_normal((max(nt for nt, _ in FX_EPISODES), 64, 768)).astype(np.float32)
    return SimpleNamespace(entries=entries, table=table, pos=pos, rows=cursor)


def _ds(budget=160):
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset

    ds = FrameSampDataset.__new__(FrameSampDataset)
    ds._motion_budget = budget
    ds._motion_tokens = budget
    ds._motion_dim = 768
    ds._motion_pos_dim = 256
    ds._max_frames = 8
    ds._tokens_per_frame = 64
    ds._motion_lpw = 9
    ds._motion_raw_dim = 16384
    ds.raw9_motion_shape = (budget * 9, 16384)
    ds.raw9_motion_dtype = BF16
    return ds


CASES = {0: (0, 15), 1: (0, 10 + 32), 141: (1, 1000 + 16 * 78 + 32), 160: (1, 2599)}


@pytest.mark.parametrize("k", sorted(CASES))
def test_fill_motion_e2e_plain_vs_prealloc(fx, k, capsys):
    g, t = CASES[k]
    e = fx.entries[g]
    rows_m, f_m = ms.visible_motion_rows(e, t)
    assert len(rows_m) == k, (k, len(rows_m))
    ds = _ds()
    frames = np.asarray(even_sampling_indices(t, 8), dtype=np.int64)
    st, ps = _BF16Store(fx.table), _PosStore(fx.pos)
    a = ds._fill_motion_e2e(e, t, st, ps, frames)
    big = np.full((2, 1440, 16384), 7, dtype=np.uint16).view(BF16)      # 预分配缓冲先填脏值
    slot = big[1]
    addr = slot.__array_interface__["data"][0]
    b = ds._fill_motion_e2e(e, t, st, ps, frames, out_emb=slot)
    assert b[0] is slot and b[0].__array_interface__["data"][0] == addr
    for x, y in zip(a, b):
        assert x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
    emb, pos, mask, order = a
    assert emb.shape == (1440, 16384) and emb.dtype == BF16
    n = 9 * k
    assert emb[:n].tobytes() == fx.table[rows_m].tobytes()               # 表行原字节（不转换）
    assert not emb[n:].view(np.uint16).any() and not b[0][n:].view(np.uint16).any()   # 尾部零
    # pos / mask / mem_order 与 768 路径（_pad_motion + memory_order）同式
    mpos = (np.ascontiguousarray(fx.pos[f_m][:, 0, :256]) if k else np.zeros((0, 256), np.float32))
    _, rpos, rmask, rtimes = ds._pad_motion(np.zeros((k, 768), np.float32), mpos, f_m, k)
    assert pos.tobytes() == rpos.tobytes() and mask.tobytes() == rmask.tobytes()
    assert np.array_equal(order, memory_order(pad_times(frames, 8), 64, rtimes))
    assert pos.shape == (160, 256) and mask.shape == (160,) and order.shape == (672,) and order.dtype == np.int32
    _say(capsys, f"E2E_FILL=PASS k={k} prealloc_eq=1 addr_kept=1 tail_zero=1 bytes_from_table=1")


def test_fill_motion_e2e_overflow_and_bad_out(fx):
    e = fx.entries[1]
    ds = _ds(budget=144)
    with pytest.raises(ValueError, match="零截断"):
        ds._fill_motion_e2e(e, 2599, _BF16Store(fx.table), _PosStore(fx.pos), np.arange(8))
    ds = _ds()
    with pytest.raises(ValueError, match="out_emb"):
        ds._fill_motion_e2e(e, 2599, _BF16Store(fx.table), _PosStore(fx.pos), np.arange(8),
                            out_emb=np.zeros((1440, 16384), np.float32))


# ══ ⑥ L1 整批 bf16 预分配 ═══════════════════════════════════════════════════════════════════


class _L1DS:
    raw9_motion_shape = (1440, 16384)
    raw9_motion_dtype = BF16

    def __init__(self, fx):
        self.fx = fx
        self.ds = _ds()

    def __len__(self):
        return 4

    def _item(self, i, out):
        g, t = CASES[sorted(CASES)[i]]
        emb, pos, mask, order = self.ds._fill_motion_e2e(self.fx.entries[g], t, _BF16Store(self.fx.table),
                                                         _PosStore(self.fx.pos), np.arange(8), out_emb=out)
        return {"motion_emb": emb, "motion_pos": pos, "motion_mask": mask, "mem_order": order,
                "x": np.full((3,), i, np.float32)}

    def __getitem__(self, i):
        return self._item(i, None)

    def getitem_into(self, i, out):
        return self._item(i, out)


def test_l1_prealloc_bf16_roundtrip(fx, capsys):
    from openpi.training import data_loader as odl

    base = _L1DS(fx)
    tds = odl.TransformedDataset(base, [])
    pre = tds.__getitems__([0, 1, 2, 3])
    assert isinstance(pre, odl._PreCollated)
    batch = odl._from_shared_torch(odl._collate_fn_shm(pre))
    m = batch["motion_emb"]
    assert m.shape == (4, 1440, 16384) and m.dtype == BF16
    for i in range(4):
        assert m[i].tobytes() == base[i]["motion_emb"].tobytes()
    assert batch["motion_pos"].shape == (4, 160, 256) and batch["mem_order"].dtype == np.int32
    _say(capsys, "E2E_L1=PASS batch=(4,1440,16384) dtype=bfloat16 bytes_eq=1")


# ══ ⑦ ⑧ 768 分支不被 e2e YAML 误入 / 配对闸 ════════════════════════════════════════════════════


def _data_config():
    from mme_vla_suite.training import config as _config

    cfg = _config.get_config("mme_vla_suite_b128_80k")
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(
        cfg.data, assets=dataclasses.replace(cfg.data.assets, assets_dir=str(V1 / "train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"))))
    return cfg.data.create(cfg.assets_dirs, cfg.model)


need_lib = pytest.mark.skipif(not (LIB / "framesamp-8x8").exists() or not (LIB / "motion").exists(),
                              reason="本机无 v1-store 1600ep 库")


@need_lib
def test_e2e_yaml_never_enters_768_branch(capsys):
    from omegaconf import OmegaConf
    from mme_vla_suite.datastore import StoreMeta
    from mme_vla_suite.training import dataloader as mdl
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset

    dc = _data_config()
    hc_e2e = OmegaConf.load(E2E_YAML)
    hc_768 = OmegaConf.load(MOTION_YAML)
    fs = LIB / "framesamp-8x8"
    # 回归：MotionMEM YAML + 768 表照常构造
    ds768 = FrameSampDataset(str(fs), dc, hc_768, 20, motion_root=str(LIB / "motion"))
    assert not ds768._motion_e2e and not hasattr(ds768, "raw9_motion_shape")
    # e2e YAML 指向 768 表：数据集按 e2e 分支校验表行形状，必拒（不会静默按 768 取数）
    with pytest.raises(ValueError, match="row_shape"):
        FrameSampDataset(str(fs), dc, hc_e2e, 20, motion_root=str(LIB / "motion"))
    # 闸：e2e YAML 指向 768 表必拒（table_kind 不配对）
    os.environ["MMEVLA_MOTION_STORE"] = str(LIB / "motion")
    try:
        with pytest.raises(ValueError, match="table_kind|layout|dtype"):
            mdl._motion_gates(hc_e2e, StoreMeta.load(fs))
        assert mdl._motion_gates(hc_768, StoreMeta.load(fs)) == str(LIB / "motion")
    finally:
        os.environ.pop("MMEVLA_MOTION_STORE", None)
    _say(capsys, "E2E_NOT_768=PASS dataset_rejects=1 gate_rejects=1 motionmem_regress=1")


def _fake_meta(**over):
    prov = {"table_kind": pm.E2E_TABLE_KIND, "table_op": "f32_to_bf16_reorder", "parameter_count": 0,
            "raw9_source_sha256": "0" * 64}
    prov.update(over.pop("prov", {}))
    spec = SimpleNamespace(grid_stride=16, row_shape=ROW_SHAPE, dtype=over.pop("dtype", BF16),
                           demo_min_real=17, demo_tail_pad="repeat_last")
    return SimpleNamespace(provenance=prov, raw={"layout": over.pop("layout", pm.E2E_LAYOUT)}, spec=spec)


def test_e2e_gate_pairing(capsys):
    from omegaconf import OmegaConf
    from mme_vla_suite.training import dataloader as mdl

    mcfg = OmegaConf.load(E2E_YAML).motion
    mdl._e2e_motion_gate(mcfg, _fake_meta())
    bad = [
        (lambda c: c.__setitem__("readout", "wan_raw9_flat16384_s32_v1"), {}),
        (lambda c: c.__setitem__("source_run", "x/checkpoint_epoch_1.pt#encoder"), {}),
        (lambda c: c.__setitem__("tokens_per_window", 9), {}),
        (lambda c: c.__setitem__("latents_per_window", 8), {}),
        (lambda c: c.__setitem__("stride", 32), {}),
        (None, {"prov": {"table_kind": "wan_raw9"}}),
        (None, {"layout": "motion-raw9-grid32-demopad17-v1"}),
        (None, {"dtype": np.dtype(np.float32)}),
    ]
    for mut, meta_over in bad:
        c = OmegaConf.create(OmegaConf.to_container(mcfg))
        if mut:
            mut(c)
        with pytest.raises(ValueError):
            mdl._e2e_motion_gate(c, _fake_meta(**meta_over))
    _say(capsys, f"E2E_GATE=PASS negatives={len(bad)} all_rejected=1")


# ══ ⑨ train.py：provenance / 诊断 ════════════════════════════════════════════════════════════


def _load_train():
    spec = importlib.util.spec_from_file_location("train_under_test", str(WT / "scripts/training/train.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@need_lib
def test_init_history_config_e2e_branch(tmp_path, monkeypatch, capsys):
    """真实 init_history_config 入口：MotionMeta 用夹具（S2 新表未导出），_motion_gates 走真实闸前半段由夹具替代。"""
    from omegaconf import OmegaConf
    from mme_vla_suite.training import dataloader as mdl
    from mme_vla_suite.training import config as _config

    train = _load_train()
    hc = OmegaConf.load(E2E_YAML)
    fake = _fake_meta()
    fake.manifest_sha256, fake.motion_index_sha256, fake.table_sha256 = "a" * 64, "b" * 64, "c" * 64
    fake.provenance["vae"] = {"name": "Wan2.1"}
    fake_root = tmp_path / "e2e_table"
    (fake_root / "meta").mkdir(parents=True)
    (fake_root / "meta" / "store_meta.json").write_text("{}")
    monkeypatch.setattr(mdl, "_motion_gates", lambda h, fm: str(fake_root))
    monkeypatch.setattr(ms.MotionMeta, "load", staticmethod(lambda root: fake))
    cfg = _config.get_config("mme_vla_suite_b128_80k")
    cfg = dataclasses.replace(cfg, exp_name="t", model=dataclasses.replace(cfg.model, history_config=E2E_YAML.name),
                              checkpoint_base_dir=str(tmp_path / "ck"))
    cfg.checkpoint_dir.mkdir(parents=True)
    train.init_history_config(cfg, hc, framesamp_root=str(LIB / "framesamp-8x8"))
    prov = json.loads((cfg.checkpoint_dir / "motion_provenance.json").read_text())
    assert prov["encoder"] is None and prov["encoder_kind"] == "wan_e2e"
    assert prov["readout"] == "wan_e2e_enc768_s16_v1" and prov["table_kind"] == "wan_raw_bf16_s16_v1"
    assert prov["raw9_source_sha256"] == "0" * 64 and prov["vae"] == {"name": "Wan2.1"}
    assert prov["policy_encoder"] == {"init": "random", "trainable": True, "layers": 1,
                                      "parameter_count": 20252928, "spec_src": "d3213de"}
    assert "pool_source_sha256" not in prov
    snap = OmegaConf.load(cfg.checkpoint_dir / "history_config.resolved.yaml")
    assert snap.motion.encoder_kind == "wan_e2e" and snap.motion.latents_per_window == 9
    # 消费方（S4 写集）：check_checkpoint.py 的 e2e 分支合入后才能闭环
    cc = WT / "scripts/evaluation/check_checkpoint.py"
    consumed = "wan_e2e" in cc.read_text()
    _say(capsys, f"E2E_PROVENANCE=PASS policy_encoder_params=20252928 table_param_count_separate=1 "
                 f"check_checkpoint_e2e_branch={'present' if consumed else 'PENDING_S4'}")


def test_train_diag_helpers(tmp_path, capsys):
    import jax.numpy as jnp
    import optax

    train = _load_train()
    d = tmp_path / "diag"
    d.mkdir()
    infos = [{"grad_norm": jnp.float32(2.0 + i), "clip_coef": jnp.float32(1 / (2.0 + i)),
              "clip_coef_ex_enc": jnp.float32(0.6), "e2e_enc_grad_norm": jnp.float32(0.5), "loss": jnp.float32(1)}
             for i in range(3)]
    train._e2e_write_clip_steps(d, [7, 8, 9], infos)
    rows = [json.loads(l) for l in (d / "clip_steps.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [7, 8, 9] and "loss" not in rows[0]
    assert abs(rows[1]["clip_coef"] - 1 / 3) < 1e-6
    # 窗口清单行号解析：列表 / windows 对象，缺行号即拒
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"windows": [{"task": "a", "row": 3}, {"table_row": 5}]}))
    r, sha = train._e2e_load_window_rows(p)
    assert r.tolist() == [3, 5] and len(sha) == 64
    p.write_text(json.dumps([{"task": "a"}]))
    with pytest.raises(ValueError):
        train._e2e_load_window_rows(p)
    x = np.random.default_rng(0).standard_normal((64, 8))
    assert 1.0 < train._e2e_effective_rank(x) <= 8.0
    assert abs(train._e2e_effective_rank(np.outer(np.arange(10.0), np.ones(4)))-1.0) < 1e-6
    _say(capsys, "E2E_DIAG_HELPERS=PASS clip_steps_per_step=1 window_rows=1 eff_rank=1")


# ══ ⑩ 标准化常量与严格参数树（保存 → bf16 恢复 → 合并）══════════════════════════════════════════


def test_norm_constants_survive_bf16_restore(tmp_path, capsys):
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict
    from omegaconf import OmegaConf
    import orbax.checkpoint as ocp
    from openpi.models import model as _omodel

    mcfg = OmegaConf.load(E2E_YAML).motion
    enc = pm.build_e2e_encoder(mcfg, nnx.Rngs(params=jax.random.key(1)), jnp.bfloat16)
    # 非状态常量：模块属性里不是 Variable / Module 的数组或浮点元组（标准化 mean / std 等）
    consts = {k: v for k, v in vars(enc).items()
              if not isinstance(v, (nnx.Variable, nnx.Module, list, dict))
              and (isinstance(v, (np.ndarray, jax.Array))
                   or (isinstance(v, tuple) and v and all(isinstance(x, float) for x in v)))}
    assert consts, "encoder 上找不到标准化常量（非状态属性）"
    params = nnx.state(enc, nnx.Param).to_pure_dict()
    ck = (tmp_path / "params").resolve()
    with ocp.PyTreeCheckpointer() as c:
        c.save(ck, {"params": params})
    restored = _omodel.restore_params(ck, restore_type=np.ndarray, dtype=jnp.bfloat16)
    want = {tuple(map(str, k)): v for k, v in flatten_dict(params).items()}      # orbax 恢复后列表下标为字符串（replace_by_pure_dict 会转回 int）
    got = {tuple(map(str, k)): v for k, v in flatten_dict(restored).items()}
    assert set(got) == set(want), f"严格参数树：多余 {set(got) - set(want)} 缺 {set(want) - set(got)}"
    gd, st = nnx.split(enc)
    st.replace_by_pure_dict(jax.tree.map(jnp.asarray, restored))
    enc2 = nnx.merge(gd, st)
    for k, v in consts.items():
        v2 = getattr(enc2, k)
        a, b = np.asarray(v), np.asarray(v2)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), k
        if a.dtype.kind == "f":
            assert a.dtype in (np.float32, np.float64) or isinstance(v, tuple), (k, a.dtype)
    z = jnp.asarray(np.random.default_rng(0).standard_normal((2, 9, 16384)).astype(BF16))
    assert enc2(z).shape == (2, 768)
    _say(capsys, f"E2E_NORM_CONST=PASS consts={sorted(consts)} restore_bf16_strict_tree=1 s1_real={int(S1_REAL)}")



# ══ ⑪ CPU 1 步训练冒烟：真实 init_train_state + train_step（小 variant）、诊断键、逐步裁剪日志、步 0 输出 ════════════


class _FixMeta:
    def __init__(self, table):
        self.spec = SimpleNamespace(dtype=BF16, row_shape=ROW_SHAPE, grid_stride=16)
        self._t = table


class _FixStore:
    def __init__(self, root, meta=None, **_):
        self._t = meta._t

    def rows_into(self, rows, out):
        out[...] = self._t[np.asarray(rows, np.int64)]
        return out

    def close(self):
        pass


def test_train_step_cpu_smoke(fx, tmp_path, monkeypatch, capsys):
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    import openpi.training.sharding as sharding
    import openpi.training.weight_loaders as wl
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from mme_vla_suite.training import config as _config

    train = _load_train()
    hc = OmegaConf.load(E2E_YAML)
    cfg = _config.get_config("mme_vla_suite_b128_80k")
    cfg = dataclasses.replace(
        cfg, exp_name="smoke", checkpoint_base_dir=str(tmp_path / "ck"), batch_size=1, fsdp_devices=1,
        weight_loader=wl.NoOpWeightLoader(),
        model=dataclasses.replace(cfg.model, history_config=hc, use_history=True,
                                  paligemma_variant="gemma_150m", action_expert_variant="gemma_300m"))
    assert train._e2e_diag_enabled(cfg)
    mesh = sharding.make_mesh(1)
    state, _ = train.init_train_state(cfg, jax.random.key(0), mesh, resume=False)
    # 步 0 输出：窗口清单夹具 + 表夹具（S2 读表合入前用夹具替身 MotionMeta / MotionStore）
    man = tmp_path / "w.json"
    man.write_text(json.dumps({"windows": [{"row": int(r)} for r in range(0, fx.rows, 7)]}))
    monkeypatch.setenv("E2E_WINDOWS_MANIFEST", str(man))
    monkeypatch.setattr(ms.MotionMeta, "load", staticmethod(lambda root: _FixMeta(fx.table)))
    monkeypatch.setattr(ms, "MotionStore", _FixStore)
    cfg.checkpoint_dir.mkdir(parents=True)
    p0 = train._e2e_step0_diagnostics(cfg, state, "fixture-root")
    d0 = json.loads(p0.read_text())
    assert d0["windows"] == len(range(0, fx.rows, 7)) and d0["finite"] and d0["out_shape"] == [d0["windows"], 768]
    assert (p0.parent / "0.npy").exists()
    # 1 步 train_step：合成批（motion_emb 取表行 bf16 原字节，前 k 槽有效）
    spec, act_spec = cfg.model.inputs_spec(batch_size=1)
    arr = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), spec)
    fields = {f.name: getattr(arr, f.name) for f in dataclasses.fields(arr)}
    ds = _ds()
    emb, mpos, mmask, order = ds._fill_motion_e2e(fx.entries[1], 2599, _BF16Store(fx.table), _PosStore(fx.pos),
                                                  np.arange(8))
    fields.update(motion_emb=jnp.asarray(emb[None]), motion_pos=jnp.asarray(mpos[None]),
                  motion_mask=jnp.asarray(mmask[None]), mem_order=jnp.asarray(order[None]),
                  static_mask=jnp.ones_like(fields["static_mask"]),
                  tokenized_prompt_mask=jnp.ones_like(fields["tokenized_prompt_mask"]),
                  image_masks={k: jnp.ones_like(v) for k, v in fields["image_masks"].items()},
                  static_image_emb=jnp.asarray(np.random.default_rng(1).standard_normal(
                      fields["static_image_emb"].shape).astype(np.float32)))
    obs = HistAugObservation(**fields)
    batch = (obs, jnp.asarray(np.random.default_rng(2).standard_normal(act_spec.shape).astype(np.float32)))
    step_fn = jax.jit(lambda r, s, b: train.train_step(cfg, r, s, b))
    with sharding.set_mesh(mesh):
        mid_state, info0 = step_fn(jax.random.key(1), state, batch)
        new_state, info = step_fn(jax.random.key(1), mid_state, batch)
    info0, info = jax.device_get(info0), jax.device_get(info)
    keys = sorted(k for k in info if k.startswith("e2e_") or k.startswith("clip"))
    assert {"clip_coef", "clip_coef_ex_enc", "e2e_enc_grad_norm", "e2e_enc_update_param_ratio"} <= set(keys)
    assert any(k.startswith("e2e_enc_grad_norm/") for k in keys)
    assert all(np.isfinite(float(v)) for i in (info0, info) for v in i.values())
    # 步 0：记忆路（mem_enc_norm 与 encoder）梯度恰为 0——MotionMEM / wanraw9s32 同配置实测同样为 0、步 1 起非零，
    # 是 modulation 记忆注入的既有初始化性质，不是 E2E 引入；步 1 起 encoder 各组梯度应非零
    assert float(info0["mem_enc_norm"]) == 0.0 and float(info0["e2e_enc_grad_norm"]) == 0.0
    assert float(info["e2e_enc_grad_norm"]) > 0 and all(float(info[k]) > 0 for k in keys if k.startswith("e2e_enc_grad_norm/"))
    train._e2e_write_clip_steps(train._e2e_diag_dir(cfg), [0, 1], [info0, info])
    clip = [json.loads(l) for l in (cfg.checkpoint_dir / "diagnostics/clip_steps.jsonl").read_text().splitlines()]
    assert [c["step"] for c in clip] == [0, 1]
    assert abs(clip[1]["clip_coef"] - min(1.0, 1.0 / float(info["grad_norm"]))) < 1e-5
    ts = train._e2e_token_stats(new_state, batch)
    assert np.isfinite(ts["e2e_enc_out_std"])
    _say(capsys, f"E2E_TRAIN_SMOKE=PASS steps=2 step0_mem_grad=0 diag_keys={len(keys)} loss={float(info['loss']):.4f} "
                 f"grad_norm={float(info['grad_norm']):.4f} enc_grad={float(info['e2e_enc_grad_norm']):.4g} "
                 f"diag0={p0.name} eff_rank={d0['effective_rank']:.2f} clip_steps=2 s1_real={int(S1_REAL)}")
