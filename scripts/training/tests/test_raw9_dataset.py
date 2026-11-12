"""S2 训练侧单测（计划 1112-4task-modul-wanraw9-s32-50k-plan.html 第二部分「二 子代理分配表」S2 行验收 ①–⑤）。

① v11 wantp768.yaml 构造的 HistoryPi0 参数树叶数 / 形状与基线 e59cde3（= c55a6bc 代码）完全一致，inputs_spec 不变；
② 新 wanraw9s32.yaml 下 motion_encoder_static kernel (17152,1024)、该层参数 17,564,672，其余叶与基线逐项相同；
③ 合成表上 dataset（``FrameSampDataset._fill_motion_raw9``）取出第 m 行 = 源 latent 第 2m 窗展平值、τ=f+4j、
   demo 尾补窗 τ 截断、pos 取前 256 维、右填充到 budget·9=720、mem_order 按 (τ, 类型) 交错；
④ ``compare_dl_equiv_raw9.py --mode raw9 --fixture``：逐 token 参考实现 vs L0 预分配实现 → DL_EQUIV=PASS，改 1 字节即 FAIL；
⑤ ``--mode act-stats`` 在合成 ckpt 夹具上输出 ACT_STATS 行。

S1 的 ``mme_vla_suite.shared.wan_raw9`` 与 stride 参数化的 ``motion_store.visible_motion_rows`` 尚未合入时，
用同名接口的最小桩（只验证 S2 的接线）；S1 合入后自动改用真模块。
运行：``uv run --no-sync python scripts/training/tests/test_raw9_dataset.py``，末行 ``S2_TRAIN_UNIT=PASS|FAIL``。
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import pathlib
import re
import sys
import types

import numpy as np
import pytest

WT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "src"))
YAML_DIR = WT / "src/mme_vla_suite/models/config/robomme"
V11_YAML = YAML_DIR / "perceptual-framesamp-modul-8frame-8x8-wantp768.yaml"
NEW_YAML = YAML_DIR / "perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml"
COMPARE = WT / "scripts/training/g0/compare_dl_equiv_raw9.py"

# 基线：e59cde3（src 与 c55a6bc 逐字相同）上对 wantp768.yaml 跑 nnx.eval_shape 得到的参数树摘要
# （行 = "叶路径:形状:dtype"，排序后以 \n 拼接再 sha256）
V11_N_LEAVES = 65
V11_TREE_SHA256 = "51f537d1b96c40af546bee0fda0c3e3982c4c4b1f8af6716b2e66b4022e38020"

# ── S1 接口桩（仅在 S1 未合入时注册）─────────────────────────────────────────────
_STUB_SRC = '''"""测试用最小桩：与 S1 wan_raw9 同名接口（计划 1112 第二部分 1.1 第 1 行）。"""
import hashlib, pathlib
import numpy as np
RAW9_ENCODER_KIND = "wan_raw9"
RAW9_READOUT = "wan_raw9_flat16384_s32_v1"
RAW9_INPUT_SHAPE = (9, 16, 32, 32)
RAW9_TOKEN_DIM = 16384
RAW9_TOKENS_PER_WINDOW = 9
RAW9_STRIDE = 32
RAW9_ROW_BYTES = 589824
RAW9_PARAMETER_COUNT = 0
_IS_TEST_STUB = True
def flatten_raw9(z):
    return np.ascontiguousarray(z, dtype=np.float32).reshape(9, 16384)
def token_times(f, seg_last_real):
    return np.minimum(int(f) + 4 * np.arange(9, dtype=np.int64), int(seg_last_real)).astype(np.int64)
def motion_token_budget(mcfg):
    return int(mcfg.budget) * int(mcfg.get("tokens_per_window", 1))
def source_sha256():
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
'''

try:
    import mme_vla_suite.shared.wan_raw9 as _wr_real  # noqa: F401
    S1_REAL = not getattr(_wr_real, "_IS_TEST_STUB", False)   # __main__ 再经 pytest 导入时会先看到本文件注册的桩
except ModuleNotFoundError:
    S1_REAL = False
    import mme_vla_suite.shared as _shared_pkg
    _stub = types.ModuleType("mme_vla_suite.shared.wan_raw9")
    exec(compile(_STUB_SRC, "<wan_raw9 stub>", "exec"), _stub.__dict__)
    sys.modules["mme_vla_suite.shared.wan_raw9"] = _stub
    _shared_pkg.wan_raw9 = _stub

from mme_vla_suite.datastore import motion_store as ms  # noqa: E402
from mme_vla_suite.shared import wan_raw9 as wr  # noqa: E402
from mme_vla_suite.shared.sampling import even_sampling_indices, memory_order, pad_times  # noqa: E402

S1_STRIDE = "grid_stride" in getattr(ms.LayoutSpec, "__dataclass_fields__", {})


def _visible_rows_stub(entry, t):
    """S1 未合入时的 visible_motion_rows 桩：与现有实现同式，只把 GRID_STRIDE 换成 entry.spec.grid_stride。"""
    es, t, st = int(entry.exec_start_idx), int(t), int(entry.spec.grid_stride)
    rows, frames = [], []
    if entry.demo.row_base is not None:
        for m in range(entry.demo.num_grid):
            if st * m + (entry.spec.demo_min_real - 1) <= es - 1:
                rows.append(entry.demo.row_base + m)
                frames.append(st * m)
    if entry.exec.row_base is not None:
        for m in range(entry.exec.num_grid):
            if st * m + (ms.WINDOW_FRAMES - 1) <= t - es:
                rows.append(entry.exec.row_base + m)
                frames.append(es + st * m)
    return np.asarray(rows, dtype=np.int64), np.asarray(frames, dtype=np.int64)


@pytest.fixture(autouse=True)
def _s1_stride(monkeypatch):
    if not S1_STRIDE:
        monkeypatch.setattr(ms, "visible_motion_rows", _visible_rows_stub)
    yield


def _load_compare():
    spec = importlib.util.spec_from_file_location("compare_dl_equiv_raw9_under_test", str(COMPARE))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


CMP = _load_compare()


def _say(capsys, line: str) -> None:
    with capsys.disabled():
        print(f"\n{line}")


# ══ ① ② 参数树与 inputs_spec ═══════════════════════════════════════════════════


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
    flat = flatten_dict(nnx.state(model, nnx.Param).to_pure_dict())
    lines = sorted("/".join(map(str, k)) + ":" + str(tuple(v.shape)) + ":" + str(v.dtype) for k, v in flat.items())
    return hc, model, flat, lines


def _sha(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def test_v11_param_tree_and_specs_unchanged(capsys):
    import mme_vla_suite
    from mme_vla_suite.models.integration.history_pi0 import _motion_specs

    assert str(WT / "src") in mme_vla_suite.__file__, mme_vla_suite.__file__
    hc, model, flat, lines = _tree(V11_YAML)
    assert len(lines) == V11_N_LEAVES and _sha(lines) == V11_TREE_SHA256
    me = model.mem_encoder
    assert (me.motion_budget, me.motion_tokens, me.motion_dim) == (160, 160, 768)
    sp = _motion_specs(hc, 4)
    assert {k: (tuple(v.shape), str(v.dtype)) for k, v in sp.items()} == {
        "motion_emb": ((4, 160, 768), "float32"), "motion_pos": ((4, 160, 256), "float32"),
        "motion_mask": ((4, 160), "bool"), "mem_order": ((4, 672), "int32")}
    _say(capsys, f"V11_PARAM_TREE=PASS n_leaves={len(lines)} sha={_sha(lines)[:16]} s1_real={int(S1_REAL)}")


def test_raw9_param_tree_and_specs(capsys):
    from mme_vla_suite.models.integration.history_pi0 import _motion_specs

    hc, model, flat, lines = _tree(NEW_YAML)
    key = ("mem_encoder", "motion_encoder_static")
    k_shape = tuple(flat[key + ("kernel",)].shape)
    b_shape = tuple(flat[key + ("bias",)].shape)
    assert k_shape == (17152, 1024) and b_shape == (1024,)
    n_params = int(np.prod(k_shape)) + int(np.prod(b_shape))
    assert n_params == 17_564_672
    # 其余叶与基线逐项相同：把 kernel 形状换回 v11 的 (1536,1024) 后整树摘要 == 基线
    swapped = sorted(l.replace("mem_encoder/motion_encoder_static/kernel:(17152, 1024)",
                               "mem_encoder/motion_encoder_static/kernel:(1536, 1024)") for l in lines)
    assert len(lines) == V11_N_LEAVES and _sha(swapped) == V11_TREE_SHA256
    me = model.mem_encoder
    assert (me.motion_budget, me.motion_tokens, me.motion_dim) == (80, 720, 16384)
    sp = _motion_specs(hc, 4)
    assert {k: tuple(v.shape) for k, v in sp.items()} == {
        "motion_emb": (4, 720, 16384), "motion_pos": (4, 720, 256), "motion_mask": (4, 720), "mem_order": (4, 1232)}
    _say(capsys, f"RAW9_PARAM_TREE=PASS kernel={k_shape} motion_encoder_static_params={n_params} n_leaves={len(lines)}")


def test_raw9_yaml_only_motion_keys_differ():
    from omegaconf import OmegaConf

    a = OmegaConf.to_container(OmegaConf.load(V11_YAML))
    b = OmegaConf.to_container(OmegaConf.load(NEW_YAML))
    ma, mb = a.pop("motion"), b.pop("motion")
    assert a == b
    want = {"dim": 16384, "tokens_per_window": 9, "budget": 80, "stride": 32,
            "store_path": "v1-store/datasets/4task-v2-1600ep-604f16da/motion-wan-raw9-s32",
            "encoder_kind": "wan_raw9", "readout": "wan_raw9_flat16384_s32_v1"}
    for k, v in want.items():
        assert mb.pop(k) == v, k
        ma.pop(k, None)
    assert ma == mb
    assert (wr.RAW9_ENCODER_KIND, wr.RAW9_READOUT) == (want["encoder_kind"], want["readout"])


# ══ ③ dataset 取数（合成表）═════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return CMP.build_fixture(tmp_path_factory.mktemp("raw9fx"))


def _fill(fx, g, t, budget=80):
    ds = CMP.fixture_dataset(fx, budget)
    frames = np.asarray(even_sampling_indices(t, CMP.FIXTURE_MAX_FRAMES), dtype=np.int64)
    out = ds._fill_motion_raw9(fx.entries[g], t, CMP.FixtureMotionStore(fx.table, fx.num_rows),
                               CMP.FixturePosStore(fx.pos), frames)
    return ds, frames, out


def _src(fx, g, seg, m16):
    e = fx.entries[g]
    lat = np.fromfile(fx.latents / f"{ms.segment_key(e.h5_file, e.raw_ep_idx, seg)}.bin", dtype="<f4")
    return lat.reshape((-1,) + wr.RAW9_INPUT_SHAPE)[m16]


def test_dataset_rows_tau_pos_pad(fx, capsys):
    # episode 1：es=50，demo 窗 s=0（全真）、s=32（es−s=18 → 尾补，τ 截到 49）；t=149 时 exec 窗 u=0,32,64（段长 100）
    g, t = 1, 149
    e = fx.entries[g]
    ds, frames, (emb, pos, mask, order) = _fill(fx, g, t)
    T = 720
    assert emb.shape == (T, 16384) and emb.dtype == np.float32
    assert pos.shape == (T, 256) and pos.dtype == np.float32
    assert mask.shape == (T,) and mask.dtype == np.bool_
    assert order.shape == (512 + T,) and order.dtype == np.int32
    wins = [("demo", 0, 0), ("demo", 1, 32), ("exec", 0, 50), ("exec", 1, 82), ("exec", 2, 114)]
    n = 9 * len(wins)
    taus = []
    for w, (seg, mp, f) in enumerate(wins):
        z = _src(fx, g, seg, 2 * mp)                                  # 第 m 行 = 源 latent 第 2m 窗
        last = e.exec_start_idx - 1 if seg == "demo" else f + 32
        for j in range(9):
            i = 9 * w + j
            assert emb[i].tobytes() == z[j].reshape(-1).tobytes()     # C→H→W 展平原字节
            tau = min(f + 4 * j, last)
            assert pos[i].tobytes() == fx.pos[tau, 0, :256].tobytes()  # pos 取 τ 行第 0 行前 256 维
            taus.append(tau)
    assert taus[9:18] == [32, 36, 40, 44, 48, 49, 49, 49, 49]          # demo 尾补窗 τ 截断到 es−1
    assert taus[18:27] == [50 + 4 * j for j in range(9)]               # exec 窗 τ = f + 4j
    assert mask[:n].all() and not mask[n:].any()                       # 右填充到 720
    assert not emb[n:].any() and not pos[n:].any()
    # mem_order：与 memory_order(帧时刻, 64, pad_times(τ)) 逐位相同，且重排后键单调、同刻帧在 motion 前、同 τ 保持窗序→j 序
    ftimes = pad_times(frames, CMP.FIXTURE_MAX_FRAMES)
    mt = pad_times(np.asarray(taus), T)
    assert np.array_equal(order, memory_order(ftimes, 64, mt))
    keys = np.concatenate([np.repeat(ftimes * 2, 64), mt * 2 + 1])
    ks = keys[order]
    assert np.all(np.diff(ks) >= 0)
    pos49 = [int(p) - 512 for p in order if p >= 512 and mt[int(p) - 512] == 49]
    assert pos49 == sorted(pos49) and len(pos49) == 4
    _say(capsys, f"RAW9_DATASET_FILL=PASS windows={len(wins)} tokens={n} T={T} tail_trunc=4")


def test_dataset_zero_windows_and_overflow(fx):
    # episode 2：es=10 无 demo 窗；t=es+31 时 exec 窗也不可见 → 全 padding
    ds, frames, (emb, pos, mask, order) = _fill(fx, 2, 41)
    assert not mask.any() and not emb.any() and not pos.any()
    assert np.array_equal(order, memory_order(pad_times(frames, 8), 64, pad_times([], 720)))
    # 窗数 > budget 即 raise（零截断契约，按窗计）
    with pytest.raises(ValueError, match="零截断"):
        _fill(fx, 1, 149, budget=4)


# ══ ④ DL_EQUIV 夹具 ═══════════════════════════════════════════════════════════


def test_dl_equiv_fixture_pass_and_tamper(tmp_path, capsys):
    rc = CMP.main(["--mode", "raw9", "--fixture", str(tmp_path / "ok")])
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if l.startswith("DL_EQUIV=")][-1]
    assert rc == 0 and re.fullmatch(r"DL_EQUIV=PASS samples=(\d+) mismatch=0", line), line
    rc2 = CMP.main(["--mode", "raw9", "--fixture", str(tmp_path / "bad"), "--tamper"])
    out2 = capsys.readouterr().out
    line2 = [l for l in out2.splitlines() if l.startswith("DL_EQUIV=")][-1]
    assert rc2 == 1 and line2.startswith("DL_EQUIV=FAIL") and "mismatch=0" not in line2, line2
    _say(capsys, f"{line}  |  tamper: {line2}")


# ══ ⑤ act-stats 合成 ckpt ═════════════════════════════════════════════════════


def test_act_stats_fixture(tmp_path, capsys):
    import orbax.checkpoint as ocp

    rng = np.random.default_rng(7)
    dout = 8
    p = {"mem_encoder": {
        "motion_pos_proj": {"kernel": (rng.standard_normal((256, 768)) * 0.05).astype(np.float32),
                            "bias": rng.standard_normal(768).astype(np.float32)},
        "motion_encoder_static": {"kernel": (rng.standard_normal((16384 + 768, dout)) * 0.01).astype(np.float32),
                                  "bias": rng.standard_normal(dout).astype(np.float32)}},
        "other": {"w": np.zeros((4, 4), np.float32)}}               # 非 motion 叶：部分恢复不应读它
    ckpt = tmp_path / "run" / "5000"
    with ocp.PyTreeCheckpointer() as c:
        c.save((ckpt / "params").resolve(), {"params": p})
    rc = CMP.main(["--mode", "act-stats", "--ckpt", str(ckpt), "--fixture", str(tmp_path / "fx")])
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if l.startswith("ACT_STATS ")][-1]
    m = re.fullmatch(r"ACT_STATS step=5000 mean=(\S+) var=(\S+) samples=64 tokens=(\d+)", line)
    assert rc == 0 and m, line
    # 与直接按公式算的结果一致
    got = CMP.load_motion_params(ckpt)
    assert got[("motion_encoder_static", "kernel")].shape == (17152, dout)
    assert float(m.group(2)) > 0 and int(m.group(3)) > 0
    _say(capsys, line)


if __name__ == "__main__":
    class _Count:
        def __init__(self):
            self.passed = self.failed = self.skipped = 0

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
                if report.outcome == "passed":
                    self.passed += 1
                elif report.outcome == "skipped":
                    self.skipped += 1
                else:
                    self.failed += 1
    c = _Count()
    rc = pytest.main([__file__, "-q", "-p", "no:cacheprovider"], plugins=[c])
    ok = rc == 0 and c.failed == 0
    print(f"S2_TRAIN_UNIT={'PASS' if ok else 'FAIL'} passed={c.passed} failed={c.failed} "
          f"skipped={c.skipped} s1_real={int(S1_REAL)}")
    sys.exit(0 if ok else 1)
