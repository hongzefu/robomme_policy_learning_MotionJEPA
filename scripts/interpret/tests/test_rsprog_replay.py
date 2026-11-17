"""计划 1117 B4 R2 的 CPU 单测：attn_adapter 的 modulation 支持与 rsprog_replay 的喂帧、点表、记忆 token 表、npz 字段。

不跑完整模型前向；modulation 的形状用抽象求值（需要本机有 checkpoint 的 run 快照，缺失则跳过）。
"""
from __future__ import annotations

import json
import os
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from scripts.interpret import attn_adapter as A
from scripts.interpret import rsprog_replay as R

MODUL_CKPT = A.C.V1 / "assets/vla-4task-modul-motion-50000/50000"
CTX_OFFICIAL = A.C.V1 / "models/official-mme-vla/perceptual-framesamp-context/79999"
CTX_MOTION = A.C.V1 / "assets/vla-o16cc-59999/59999"


# ── 喂帧节奏与点表 ─────────────────────────────────────────────────────────────────────────
def test_eval_batches_follow_eval_cadence():
    assert R.eval_batches(10, 10) == [(0, 11)]
    assert R.eval_batches(10, 42) == [(0, 11), (11, 27), (27, 43)]
    ends = [stop - 1 for _, stop in R.eval_batches(5, 5 + 16 * 4)]
    assert ends == [5, 21, 37, 53, 69]
    ends8 = {stop - 1 for _, stop in R.eval_batches(5, 69, batch=8)}
    assert set(ends) <= ends8                       # 精度对照节奏的批边界覆盖全部推理点
    with pytest.raises(ValueError):
        R.check_eval_points(5, [5, 20])


def test_load_points_variants(tmp_path):
    a = {"episodes": [{"difficulty": "easy", "episode": 1000, "e0": 100, "seed": 3400000,
                       "points": [{"t": 116, "u": 16}, {"t": 100, "u": 0}]},
                      {"difficulty": "xhard", "episode": 1000, "exec_start_idx": 300, "points": [{"u": 0}]}]}
    pa = tmp_path / "a.json"
    pa.write_text(json.dumps(a))
    pts = R.load_points(pa)
    assert list(pts) == ["easy/1000", "xhard/1000"]
    assert pts["easy/1000"]["ts"] == [100, 116] and pts["easy/1000"]["us"] == [0, 16]
    assert pts["xhard/1000"]["ts"] == [300]
    assert pts["easy/1000"]["noise_id"] != pts["xhard/1000"]["noise_id"]
    b = {"episodes": {"medium/1001": {"difficulty": "medium", "demo_len": 50, "points": [{"t": 66}]}}}
    pb = tmp_path / "b.json"
    pb.write_text(json.dumps(b))
    assert R.load_points(pb)["medium/1001"]["episode"] == 1001
    bad = {"episodes": [{"difficulty": "easy", "episode": 1, "e0": 10, "points": [{"t": 20}]}]}
    pc = tmp_path / "c.json"
    pc.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        R.load_points(pc)


def test_shards_partition():
    pts = {f"{d}/{k}": None for d in R.DIFFICULTIES for k in range(1000, 1050)}
    shards = [set(R.select_episodes(pts, None, f"{i}/3")) for i in range(3)]
    assert set().union(*shards) == set(pts)
    assert sum(len(s) for s in shards) == len(pts)
    assert all(len(s) > 40 for s in shards)
    assert R.select_episodes(pts, ["hard/1003"], None) == ["hard/1003"]
    with pytest.raises(SystemExit):
        R.select_episodes(pts, ["1003"], None)        # 局号在四个难度里重复


# ── 记忆 token 表 ─────────────────────────────────────────────────────────────────────────
def _fake_policy(e0, t, n_frames_budget=8, tpf=64, motion_budget=160, window=33, stride=16, demo_min=17):
    from mme_vla_suite.shared.sampling import pad_times

    frames = list(np.linspace(0, t, n_frames_budget).round().astype(int))

    def visible(step):
        out, s = [], 0
        while s + demo_min - 1 <= e0 - 1:
            out.append(s)
            s += stride
        u = 0
        while u + window - 1 <= step - e0:
            out.append(e0 + u)
            u += stride
        return out

    vis = visible(t)
    mb = types.SimpleNamespace(
        get_frame_sampling_indices=lambda step, budget, tpi: frames,
        _prepare_motion=lambda step: (None, None, None, pad_times(vis, motion_budget)),
        exec_start_idx=e0, motion_window=window)
    cfg = types.SimpleNamespace(budget=n_frames_budget * tpf, token_per_image=tpf, num_views=1, motion={})
    return types.SimpleNamespace(config=cfg, mem_buffer=mb, motion_enabled=True), frames, vis


def test_memory_token_table_types_and_times():
    from mme_vla_suite.shared.sampling import memory_order, pad_times
    e0, t, tpf = 100, 196, 64
    policy, frames, vis = _fake_policy(e0, t)
    ft = pad_times(frames, 8)
    mt = pad_times(vis, 160)
    order = memory_order(ft, tpf, mt)
    static = np.ones(8 * tpf, bool)
    motion_mask = np.zeros(160, bool)
    motion_mask[:len(vis)] = True
    obs = types.SimpleNamespace(static_mask=static[None], motion_mask=motion_mask[None], mem_order=order[None])
    tab = R.memory_token_table(policy, obs, t)
    v = tab["valid"]
    assert v.sum() == 8 * tpf + len(vis)
    assert (tab["type"][v] == 1).sum() == len(vis)
    mot = v & (tab["type"] == 1)
    starts = tab["t0"][mot]
    assert sorted(starts.tolist()) == sorted(vis)
    for s, e in zip(tab["t0"][mot], tab["t1"][mot]):
        assert e == (min(s + 32, e0 - 1) if s < e0 else s + 32)
    fr = v & (tab["type"] == 0)
    assert np.array_equal(tab["t0"][fr], tab["t1"][fr])
    assert set(tab["t0"][fr].tolist()) == set(frames)
    assert np.all(np.diff(np.where(v, tab["t0"], 10**9)[v]) >= 0)      # 记忆序按时间升序
    bad = types.SimpleNamespace(static_mask=obs.static_mask, motion_mask=obs.motion_mask,
                                mem_order=np.roll(order, 1)[None])
    with pytest.raises(RuntimeError):
        R.memory_token_table(policy, bad, t)


# ── MemoryAttention / MemoryRMSNorm 重写 ─────────────────────────────────────────────────────
def test_mem_attention_rewrite_matches_original():
    d, rowsum, masked = R.mem_attention_equivalence(seed=1, T=6, S=9)
    assert d <= 1e-5
    assert rowsum <= 1e-5
    assert masked == 0.0


def _small_block():
    from openpi.models.gemma import Config
    c0 = Config(width=64, depth=1, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=16)
    c1 = Config(width=1024, depth=1, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=16)
    return (c0, c1)


def test_history_block_capture_equivalence():
    """小配置 HistoryBlock(modulation)：换成重写版后输出逐值一致；记录项可还原 mem_mod_vec 与 FiLM 量。"""
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.models.integration import history_gemma as hg
    cfgs = _small_block()
    block = hg.HistoryBlock(configs=cfgs, integration_type="modulation")
    ks = jax.random.split(jax.random.key(0), 5)
    P, T, S = 6, 20, 11
    xp = jax.random.normal(ks[0], (1, P, 64))
    xs = jax.random.normal(ks[1], (1, T, 1024))
    mem = jax.random.normal(ks[2], (1, S, 1024))
    mmask = jnp.asarray([[True] * 8 + [False] * 3])
    is_mot = jnp.asarray([False] * 4 + [True] * 4 + [False] * 3)
    pos = jnp.arange(P + T)[None]
    mask = jnp.ones((1, 1, P + T, P + T), bool)
    args = ([xp, xs], None, pos, mask, [None, None], [None, mem], [None, mmask], True)
    params = block.init(ks[3], *args)
    # 让 FiLM 的 Dense 偏置非零，F(x,0) 与纯 rms(x) 可区分
    params = jax.tree_util.tree_map_with_path(
        lambda p, x: x + 0.1 if "mem_rms_norm_ffn" in jax.tree_util.keystr(p) and "bias" in jax.tree_util.keystr(p) else x,
        params)
    ref, _ = block.apply(params, *args)
    with A._capture_attn(hg, A._make_patched_attention()) as cap, A._capture_mem(hg) as mcap:
        mine, _ = block.apply(params, *args)
        rec = {r["kind"]: r for r in mcap}
        probs = cap[0]
    assert set(rec) == {"attn", "film"}
    for a, b in zip(ref, mine):
        assert float(jnp.abs(a - b).max()) <= 1e-5
    # 原模块对同一输入的 mem_mod_vec
    orig = hg.MemoryAttention().apply({"params": params["params"]["mem_attn"]}, rec["attn"]["x_in"], mem, mmask)
    assert float(jnp.abs(orig - rec["attn"]["out"]).max()) <= 1e-5
    stats = A.modulation_point_stats(rec["attn"], rec["film"], probs, is_mot, action_horizon=T, img_end=4, prefix_len=P)
    assert stats["attn_mem"].shape == (S,)
    assert abs(float(stats["attn_mem"].sum()) - 1.0) <= 1e-5
    assert float(stats["attn_mem"][8:].max()) == 0.0
    assert float(stats["mem_rowsum_err"]) <= 1e-5 and float(stats["main_rowsum_err"]) <= 1e-5
    assert abs(float(stats["attn_main"].sum()) - 1.0) <= 1e-5
    assert float(stats["film"][0]) > 0.0 and 0.0 < float(stats["film"][1]) < 1.0
    # FiLM 总量：独立按定义重算
    x = rec["film"]["out"]
    base = rec["film"]["base"]
    want = (jnp.linalg.norm(x - base, axis=-1) / jnp.linalg.norm(base, axis=-1))[0, :16].mean()
    assert abs(float(want) - float(stats["film"][0])) <= 1e-5
    allm = A.modulation_point_stats(rec["attn"], rec["film"], probs, jnp.ones(S, bool), action_horizon=T, img_end=4,
                                    prefix_len=P)
    nom = A.modulation_point_stats(rec["attn"], rec["film"], probs, jnp.zeros(S, bool), action_horizon=T, img_end=4,
                                   prefix_len=P)
    assert abs(float(allm["film"][1]) - 1.0) <= 1e-5 and float(nom["film"][1]) == 0.0


def test_memory_input_means():
    import jax.numpy as jnp
    x = jnp.arange(5 * 4, dtype=jnp.float32).reshape(5, 4)
    valid = jnp.asarray([True, True, True, True, False])
    mot = jnp.asarray([False, True, False, True, True])
    fr, mo, latest = A.memory_input_means(x, valid, mot)
    assert np.allclose(fr, np.asarray(x)[[0, 2]].mean(0))
    assert np.allclose(mo, np.asarray(x)[[1, 3]].mean(0))
    assert np.allclose(latest, np.asarray(x)[3])
    _, mo0, latest0 = A.memory_input_means(x, valid, jnp.zeros(5, bool))
    assert float(jnp.abs(mo0).max()) == 0.0 and float(jnp.abs(latest0).max()) == 0.0


# ── npz 字段 ──────────────────────────────────────────────────────────────────────────────
def test_pack_episode_matches_plan_table(tmp_path):
    rows = R.synthetic_rows(n=4, S=12, W=8)
    arrays = R.pack_episode(rows)
    assert R.check_npz_schema(arrays) == []
    lens = [int(r["table"]["valid"].sum()) for r in rows]
    M = max(lens)
    assert arrays["attn_mem"].shape == (4, 18, M) and list(arrays["mem_len"]) == lens
    for i, r in enumerate(rows):
        v = r["table"]["valid"]
        assert np.array_equal(arrays["attn_mem"][i, :, :lens[i]], r["attn_mem"][:, v])
        assert np.all(arrays["mem_type"][i, lens[i]:] == -1) and np.all(arrays["mem_t0"][i, lens[i]:] == -1)
        assert np.all(arrays["attn_mem"][i, :, lens[i]:] == 0)
    path = tmp_path / "A" / "easy" / "episode_1000.npz"
    R.write_episode(path, arrays, {"episode": 1000, "difficulty": "easy", "model": "A", "e0": 10, "samples": 5,
                                   "seed": 42, "fingerprint": "f"})
    with np.load(path, allow_pickle=False) as z:
        loaded = {k: z[k] for k in z.files}
    assert R.check_npz_schema(loaded) == []
    assert str(loaded["model"]) == "A" and int(loaded["action_samples"]) == 5
    assert R.existing_points(path, "f") == 4
    with pytest.raises(SystemExit):
        R.existing_points(path, "g")


def test_precision_ratio():
    a = np.cumsum(np.ones((5, 18, 3)), axis=0)
    ratio, err = R.precision_ratio(a, a + 0.01)
    assert abs(err - 0.01) < 1e-12 and abs(ratio - 100) < 1e-6
    assert R.precision_ratio(a, a)[0] == float("inf")


def test_selftest_cli():
    assert R.main(["--selftest"]) == 0


# ── 适配器形状（抽象求值，不加载权重）──────────────────────────────────────────────────────────
@pytest.mark.skipif(not (MODUL_CKPT.parent / "history_config.resolved.yaml").is_file(), reason="无 modulation run 快照")
def test_modulation_adapter_shapes_and_layout():
    os.chdir(A.C.REPO_ROOT)
    ad, obs, lay = A.modulation_abstract(MODUL_CKPT, "mme_vla_suite_b128_80k")
    assert ad.modulation and lay["MEM"] == 0 and lay["IMG"] == lay["P"] - lay["TXT"]
    assert lay["MEM_TOKENS"] == 512 + 160 and lay["P"] == 576 and lay["TXT"] == 64
    masks = R.prefix_group_masks(lay)
    assert masks.shape == (2, lay["P"]) and masks[0, :lay["IMG"]].all() and masks[1, lay["IMG"]:].all()
    assert not (masks[0] & masks[1]).any()
    out = A.modulation_capture_shapes(ad, obs, lay)
    hid = out[-1]
    S, Wa, W = lay["MEM_TOKENS"], 1024, 2048
    want = {"hidden_prefix": (19, 2, W), "hidden_action": (18, Wa), "hidden_memmod": (18, Wa), "attn_mem": (18, S),
            "attn_main": (18, 3), "film_rel": (18, 2), "mem_frame_mean": (Wa,), "mem_motion_mean": (Wa,),
            "mem_motion_latest": (Wa,), "mem_valid": (S,), "mem_is_motion": (S,)}
    for k, sh in want.items():
        assert tuple(hid[k].shape) == sh, k
    zero = A.modulation_capture_shapes(ad, obs, lay, zero_mem=True)
    assert all(tuple(a.shape) == tuple(b.shape) for a, b in zip(out[:4], zero[:4]))
    # 不捕获时也能抽象求值（make_sample_fn 路径）
    import jax
    plain = jax.eval_shape(lambda s, r, o: ad.f_sample_attn(s, r, o, None, 10, lay["MEM"] + lay["IMG"], True),
                           ad.gstate, jax.random.key(0), obs)
    assert tuple(plain[0].shape) == (1, 20, 32)
    assert tuple(plain[1].shape) == (10, 18, 1, 8, 20, lay["P"] + 20)


@pytest.mark.skipif(not (CTX_OFFICIAL.is_dir() and CTX_MOTION.is_dir()), reason="无 context checkpoint")
def test_context_dry_shapes_unchanged(capsys):
    args = types.SimpleNamespace(official_ckpt=str(CTX_OFFICIAL), official_config="mme_vla_suite_b128",
                                 motion_ckpt=str(CTX_MOTION), motion_config="mme_vla_suite_orig16_b128_60k", num_steps=10)
    assert A.dry_shapes(args) == 0
    text = capsys.readouterr().out
    assert "ATTN_DRY_SHAPES=PASS official_ctx512:P=1088,MEM=512 motion_concat:P=1184,MEM=608" in text
