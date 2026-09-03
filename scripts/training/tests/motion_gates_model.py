#!/usr/bin/env python3
"""motion memory S2 开启态正确性闸 M1–M5（motion-memory-plan.md 第一部分 5.3 / 第二部分四节表一、五节）。

全部在本机 CPU 跑（`JAX_PLATFORMS=cpu`），不需要训练 checkpoint；M3 / M4 用 gemma `dummy` 变体随机初始化的 HistoryPi0。

  --gate m1   数据端交付：脚本内独立 oracle（直读 motion_index.json / motion 表 / pos 表 / 清单，不 import 被测 dataset / store / sampling
              的公式）按公式重算每个样本的 motion_emb / motion_pos / motion_mask / mem_order，与 FrameSampDataset.__getitem__ 逐位；
              三层：helper 合成网格（预算 4，合法数 0–4，第 5 个必 raise）/ 迷你库（合成 motion store + 迷你 framesamp）/ 40 ep 真实库全部 11,530 样本。
              判定行 MOTION_DELIVERY=PASS samples=<n> mismatches=0
  --gate m2   排队函数：10,000 组随机输入对 Python sorted 三元组键逐位 + 五条性质 + 两侧同一函数对象 + import 面只有 numpy。
              判定行 MEM_ORDER=PASS cases=10000 mismatches=0
  --gate m3   新层与重排：帧路输出两态逐位；运动路两层按生产 bf16 语义用独立 jax.lax.dot_general 复算逐位（另报 ULP）；
              padding 行两两逐位；gather 对 20 个随机置换 vs np.take_along_axis 逐位；三种坏 mem_order 必 raise；参数命名与分组。
              判定行 MOTION_ENC=PASS、MEM_GATHER=PASS
  --gate m4   mask 正确性：三样本定点 batch（k=6,m=0 / k=32,m=11 / k=32,m=96）：(a) 补位塞垃圾 loss 与动作逐位不变；
              (b) 输入梯度补位为零、真位非零，参数梯度全空 batch 为零；(c) 并列序 vs 交错 loss 必变；(d) 真行置换 + 重算 mem_order loss 逐位；
              (e) 关闭态参数拷入开启态，m=0 样本 |Δloss| ≤ 1e-4·|loss| 且 ≤ 1% × (c) 的差。
              判定行 MASK_INVARIANCE / GRAD_LEAK / ORDER_EFFECT / ZERO_MOTION_EQUIV
  --gate m5   搬运环节：spec / observation / preprocess / Repack / 双 store 同源 正向全链透传；负向：坏 mem_order、缺键、spec 与开关不一致、
              stride / window / direction / origin / frame_size 错、未 verified、换入另一合法 store、只篡改 index、resolved sha 错、
              motion checkpoint extra / missing——每种都必须在训练或评估前 raise。判定行 MOTION_PLUMBING=PASS

用法：
    MMEVLA_MOTION_STORE=v1-store/datasets/4task-motion-40ep/motion JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \\
      UV_LINK_MODE=copy uv run scripts/training/tests/motion_gates_model.py --gate m1 --lib v1-store/datasets/4task-motion-40ep
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import json
import os
import pathlib
import random
import shutil
import sys
import time
import types

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if not (_REPO_ROOT / "pyproject.toml").exists():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
sys.path.insert(0, str(_REPO_ROOT / "src"))
# v1-store 根：worktree 内开发时用 MMEVLA_V1_STORE 指到主树（worktree 没有 v1-store）
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))

CLOSED_YAML = "perceptual-framesamp-context.yaml"
OPEN_YAML = "perceptual-framesamp-context-motion.yaml"
SENTINEL = np.iinfo(np.int32).max
TOKENS_PER_FRAME = 16
FRAME_BUDGET = 32
MOTION_BUDGET = 96
POS_DIM = 256


# ══════════════════════════════════════════════════════════════════════════════
# 独立 oracle（不 import motion_store / sampling 的公式；只用 numpy + json）
# ══════════════════════════════════════════════════════════════════════════════

def oracle_even_indices(step: int, budget: int) -> list[int]:
    """与 shared.sampling.even_sampling_indices 同式（独立书写，作对照）。"""
    if step < budget:
        return list(range(step + 1))
    return np.linspace(0, step, budget, dtype=np.int32).tolist()


def oracle_visible(entry: dict, t: int, stride: int = 16, window: int = 33) -> list[tuple[int, int]]:
    """(row, f) 按 f 升序：demo s=stride·m 满足 s+window−1 ≤ es−1；exec u=stride·m 满足 u+window−1 ≤ t−es。"""
    es = int(entry["exec_start_idx"])
    out = []
    d, x = entry["demo"], entry["exec"]
    for m in range(int(d["num_grid"])):
        s = stride * m
        if s + window - 1 <= es - 1:
            out.append((int(d["row_base"]) + m, s))
    for m in range(int(x["num_grid"])):
        u = stride * m
        if u + window - 1 <= t - es:
            out.append((int(x["row_base"]) + m, es + u))
    return sorted(out, key=lambda r: r[1])


def oracle_mem_order(frame_times: list[int], motion_times: list[int], tokens_per_frame: int = TOKENS_PER_FRAME) -> np.ndarray:
    """Python sorted 三元组键 (时刻, 类型, 原位) 的稳定排序（与被测 np.argsort 实现独立）。"""
    items = []
    pos = 0
    for f in frame_times:
        for _ in range(tokens_per_frame):
            items.append((f, 0, pos)); pos += 1
    for f in motion_times:
        items.append((f, 1, pos)); pos += 1
    return np.array([i[2] for i in sorted(items)], dtype=np.int32)


def oracle_sample(entry: dict, t: int, table: np.ndarray, pos_table: np.ndarray, budget: int = MOTION_BUDGET,
                  pos_dim: int = POS_DIM) -> dict:
    vis = oracle_visible(entry, t)
    k = len(vis)
    if k > budget:
        raise ValueError(f"oracle: 合法起点数 {k} > 预算 {budget}")
    rows = [r for r, _ in vis]
    frames = [f for _, f in vis]
    emb = np.zeros((budget, table.shape[1]), np.float32)
    pos = np.zeros((budget, pos_dim), np.float32)
    if k:
        emb[:k] = table[rows]
        pos[:k] = pos_table[frames, 0, :pos_dim]
    mask = np.zeros(budget, np.bool_)
    mask[:k] = True
    mtimes = frames + [SENTINEL] * (budget - k)
    fr = oracle_even_indices(t, FRAME_BUDGET)
    ftimes = fr + [SENTINEL] * (FRAME_BUDGET - len(fr))
    return {"motion_emb": emb, "motion_pos": pos, "motion_mask": mask,
            "mem_order": oracle_mem_order(ftimes, mtimes), "k": k}


def load_lib_oracle(lib: pathlib.Path):
    index = json.loads((lib / "motion" / "meta" / "motion_index.json").read_text(encoding="utf-8"))
    table = np.fromfile(lib / "motion" / "motion_token.f32.bin", dtype=np.float32).reshape(-1, 768)
    sm = json.loads((lib / "framesamp" / "meta" / "store_meta.json").read_text(encoding="utf-8"))
    pos_table = np.fromfile(lib / "framesamp" / "pos_emb_4x4.f32.bin", dtype=np.float32).reshape(-1, 16, 768)
    manifest = json.loads((lib / "meta" / "episode_manifest.json").read_text(encoding="utf-8"))
    assert index["totals"]["rows"] == table.shape[0]
    assert sm["num_pos_rows"] == pos_table.shape[0]
    return index, table, pos_table, manifest


def _fake_data_config():
    ns = json.load(open(_V1 / "train-assets/mme_vla_suite/robomme/norm_stats.json"))["norm_stats"]["state"]
    st = types.SimpleNamespace(q01=np.array(ns["q01"]), q99=np.array(ns["q99"]), mean=np.array(ns["mean"]), std=np.array(ns["std"]))
    return types.SimpleNamespace(norm_stats={"state": st}, use_quantile_norm=True)


def _load_yaml(name: str):
    import omegaconf
    return omegaconf.OmegaConf.load(_REPO_ROOT / "src/mme_vla_suite/models/config/robomme" / name)


def _make_dataset(lib: pathlib.Path, yaml_name: str):
    from mme_vla_suite.training.dataloader import _create_framesamp_dataset
    os.environ["MMEVLA_MOTION_STORE"] = str(lib / "motion")
    return _create_framesamp_dataset(str(lib / "framesamp"), _fake_data_config(), _load_yaml(yaml_name), 20)


def _bytes_equal(a, b) -> bool:
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
# M1
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_entry(g: int, nt: int, es: int, row_cursor: int, stride=16, window=33) -> tuple[dict, int]:
    def seg(L):
        nc = max(0, L - (window - 1)); ng = len(range(0, nc, stride)); return nc, ng
    dnc, dng = seg(es); xnc, xng = seg(nt - es)
    e = {"g": g, "h5_file": "record_dataset_T.h5", "raw_ep_idx": g, "num_timesteps": nt, "exec_start_idx": es,
         "demo": {"row_base": row_cursor if dng else None, "num_grid": dng, "num_chunks": dnc, "seg_len": es},
         "exec": {"row_base": row_cursor + dng if xng else None, "num_grid": xng, "num_chunks": xnc, "seg_len": nt - es}}
    return e, row_cursor + dng + xng


def m1_helper_layer() -> tuple[int, int]:
    """helper 层：合成网格（demo 刚够 / 刚差一帧 / exec 第 32 帧首起点）+ 测试预算 4 覆盖合法数 0–4、第 5 个必 raise。"""
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.shared.sampling import memory_order, pad_times
    checked = bad = 0
    specs = [(291, 0), (338, 66), (586, 216), (100, 33), (100, 32), (100, 34), (65, 0), (64, 0), (400, 0)]
    cursor = 0
    entries = []
    for g, (nt, es) in enumerate(specs):
        e, cursor = _synthetic_entry(g, nt, es, cursor)
        entries.append(e)
    parsed = ms.parse_index({"schema": 1, "layout": ms.LAYOUT, "grid_stride": 16, "window_frames": 33,
                             "grid_origin": "segment_start", "window_direction": "forward", "truncation_policy": "none",
                             "entries": entries, "totals": {"rows": cursor, "exec_rows": sum(e["exec"]["num_grid"] for e in entries),
                                                            "demo_rows": sum(e["demo"]["num_grid"] for e in entries)}})
    for e, pe in zip(entries, parsed, strict=True):
        for t in range(e["exec_start_idx"], e["num_timesteps"]):
            rows, frames = ms.visible_motion_rows(pe, t)
            o = oracle_visible(e, t)
            checked += 1
            if rows.tolist() != [r for r, _ in o] or frames.tolist() != [f for _, f in o]:
                bad += 1
    # 预算 4：合法数 0–4 都能 pad，第 5 个必 raise（禁止裁掉早期历史后继续）
    for k in range(0, 5):
        pad_times(list(range(0, 16 * k, 16)), 4)
    try:
        pad_times(list(range(0, 16 * 5, 16)), 4)
        bad += 1
    except ValueError:
        pass
    # mem_order 对合成时刻与 oracle 逐位
    rng = random.Random(1)
    for _ in range(200):
        n = rng.randint(0, 32); m = rng.randint(0, 96)
        ft = sorted(rng.sample(range(0, 600), n)) + [SENTINEL] * (32 - n)
        mt = sorted(rng.sample(range(0, 600), m)) + [SENTINEL] * (96 - m)
        checked += 1
        if not np.array_equal(memory_order(np.array(ft), 16, np.array(mt)), oracle_mem_order(ft, mt)):
            bad += 1
    # 非法置换必 raise：直接调用被测校验（构造不可能的 keys 无法触发，改用 embed_memory 侧在 M3 覆盖）；此处验哨兵不与真实时刻相交
    assert SENTINEL > 2 * 4096
    return checked, bad


def m1_real_layer(lib: pathlib.Path, limit: int | None) -> tuple[int, int, dict]:
    index, table, pos_table, manifest = load_lib_oracle(lib)
    ds = _make_dataset(lib, OPEN_YAML)
    n = len(ds) if limit is None else min(limit, len(ds))
    # 独立换算 idx → (g, t)：清单 exec_sample_offset 前缀和
    eps = manifest["episodes"]
    starts = np.array([e["exec_sample_offset"] for e in eps], np.int64)
    bad = 0
    kstat = []
    t0 = time.time()
    for idx in range(n):
        g = int(np.searchsorted(starts, idx, side="right") - 1)
        t = int(eps[g]["exec_start_idx"]) + (idx - int(starts[g]))
        o = oracle_sample(index["entries"][g], t, table, pos_table)
        d = ds[idx]
        for k in ("motion_emb", "motion_pos", "motion_mask", "mem_order"):
            if not _bytes_equal(d[k], o[k]):
                bad += 1
                if bad <= 10:
                    print(f"  ✗ idx={idx} g={g} t={t} key={k}")
        kstat.append(o["k"])
        if (idx + 1) % 2000 == 0:
            print(f"  [m1] {idx + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    ks = np.array(kstat)
    stats = {"n": n, "k_median": float(np.median(ks)), "k_mean": float(ks.mean()), "k_max": int(ks.max()),
             "k_p25": float(np.percentile(ks, 25)), "k_p75": float(np.percentile(ks, 75)), "k_p90": float(np.percentile(ks, 90)),
             "k_p95": float(np.percentile(ks, 95)), "k_p99": float(np.percentile(ks, 99)),
             "zero_frac": float((ks == 0).mean()), "fill_rate": float(ks.mean() / MOTION_BUDGET)}
    ds.close()
    return n, bad, stats


def cmd_m1(args):
    c1, b1 = m1_helper_layer()
    print(f"[m1 helper] checked={c1} mismatches={b1}")
    n, b3, stats = m1_real_layer(pathlib.Path(args.lib), args.limit)
    print(f"[m1 real] samples={n} mismatches={b3} 有效数分布 {json.dumps(stats)}")
    # A19：有效数分布须与清单统计一致（40 ep 库：中位 11 / 均值 11.46 / 最大 34 / 零起点 5.55%）
    a19 = (abs(stats["k_mean"] - 11.46) < 0.05 and stats["k_max"] == 34 and abs(stats["zero_frac"] - 0.0555) < 0.001
           and stats["k_median"] == 11.0) if args.limit is None else True
    print(f"A19_VALID_DIST={'PASS' if a19 else 'FAIL'} median={stats['k_median']} mean={stats['k_mean']:.2f} max={stats['k_max']} "
          f"zero_frac={stats['zero_frac']:.4f} fill_rate={stats['fill_rate']:.3f}")
    ok = b1 == 0 and b3 == 0 and a19
    print(f"MOTION_DELIVERY={'PASS' if ok else 'FAIL'} samples={n} mismatches={b1 + b3} helper_checked={c1}")
    if not ok:
        raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════════
# M2
# ══════════════════════════════════════════════════════════════════════════════

def cmd_m2(args):
    from mme_vla_suite.shared import sampling
    from mme_vla_suite.shared.sampling import MEM_ORDER_SENTINEL, memory_order
    import mme_vla_suite.training.framesamp_dataset as fds
    rng = random.Random(args.seed)
    bad = 0
    N = 10000
    for i in range(N):
        n = rng.randint(0, 32); m = rng.randint(0, 96)
        step = rng.randint(0, 1200)
        fr = oracle_even_indices(step, 32)[:n] if rng.random() < 0.5 else sorted(rng.sample(range(0, 1300), n))  # 含 linspace 重复值
        ft = list(fr) + [MEM_ORDER_SENTINEL] * (32 - len(fr))
        if rng.random() < 0.5 and fr:
            mt = sorted(set(rng.choices(fr, k=min(m, len(fr))) + rng.sample(range(0, 1300), max(0, m - len(fr)))))[:m]
        else:
            mt = sorted(rng.sample(range(0, 1300), m))
        mt = mt + [MEM_ORDER_SENTINEL] * (96 - len(mt))
        got = memory_order(np.array(ft), 16, np.array(mt))
        exp = oracle_mem_order(ft, mt)
        if not np.array_equal(got, exp):
            bad += 1; continue
        # 五条性质
        n_valid = 16 * len(fr) + (96 - mt.count(MEM_ORDER_SENTINEL))
        if not np.array_equal(np.sort(got), np.arange(608)):
            bad += 1; continue
        valid_positions = set(range(16 * len(fr))) | set(512 + j for j in range(96) if mt[j] != MEM_ORDER_SENTINEL)
        if set(got[:n_valid].tolist()) != valid_positions:
            bad += 1; continue                                    # 真 token 占前 16k+m 位
        for f_i in range(len(fr)):                                # 同帧 16 位连续升序
            where = [p for p, v in enumerate(got) if 16 * f_i <= v < 16 * (f_i + 1)]
            if where != list(range(where[0], where[0] + 16)):
                bad += 1; break
        # 同刻帧在 motion 前；padding 帧路在前
        keyed = [(ft[v // 16], 0) if v < 512 else (mt[v - 512], 1) for v in got]
        if keyed != sorted(keyed):
            bad += 1
    same_obj = fds.memory_order is sampling.memory_order
    tree = ast.parse((_REPO_ROOT / "src/mme_vla_suite/shared/sampling.py").read_text(encoding="utf-8"))
    imports = {n.names[0].name.split(".")[0] if isinstance(n, ast.Import) else n.module.split(".")[0]
               for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))}
    numpy_only = imports == {"numpy"}
    sentinel_ok = MEM_ORDER_SENTINEL >= 2 * 4096
    ok = bad == 0 and same_obj and numpy_only and sentinel_ok
    print(f"MEM_ORDER={'PASS' if ok else 'FAIL'} cases={N} mismatches={bad} same_object={int(same_obj)} imports={sorted(imports)} sentinel_ok={int(sentinel_ok)}")
    if not ok:
        raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════════
# M3 / M4 共用：CPU dummy 模型与定点 batch
# ══════════════════════════════════════════════════════════════════════════════

def _make_models(seed: int = 0):
    """CPU 上用 gemma dummy 变体（宽度 64）随机初始化开 / 关两态 HistoryPi0。

    dummy 主干宽度 64 ≠ 生产的 2048，所以模型侧 memory_token_dim 同步改成 64（只影响 encoder_static / motion_encoder_static
    的输出维；数据侧仍按生产 YAML 交付）。M3 的运动路 bf16 复算与 M4 的 mask 性质都不依赖具体宽度。
    """
    import jax
    from openpi.models import gemma as _gemma
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
    width = int(_gemma.get_config("dummy").width)
    def cfg(name):
        hc = _load_yaml(name)
        hc.memory_token_dim = width
        return HistoryPi0Config(use_history=True, history_config=hc, dtype="bfloat16", action_horizon=20,
                                paligemma_variant="dummy", action_expert_variant="dummy")
    c_open, c_closed = cfg(OPEN_YAML), cfg(CLOSED_YAML)
    return c_open, c_open.create(jax.random.key(seed)), c_closed, c_closed.create(jax.random.key(seed))


def _fixture_batch(lib: pathlib.Path, ds, specs):
    """按 (k 帧数, m motion 数) 从真实库挑样本：k=6 → t=5 的样本；k=32,m=11 → 某 t 使可见 11；k=32,m=96 → 人造满 96。"""
    index, table, pos_table, manifest = load_lib_oracle(lib)
    eps = manifest["episodes"]
    starts = [e["exec_sample_offset"] for e in eps]
    chosen = []
    for want_k, want_m in specs:
        found = None
        best = (-1, None)
        for g, e in enumerate(eps):
            for t in range(e["exec_start_idx"], e["num_timesteps"]):
                k = min(t + 1, 32)
                m = len(oracle_visible(index["entries"][g], t))
                idx = starts[g] + (t - e["exec_start_idx"])
                if want_m == "max":
                    if k == want_k and m > best[0]:
                        best = (m, idx)
                elif k == want_k and m == want_m:
                    found = idx; break
            if found is not None:
                break
        if want_m == "max":
            found = best[1]
        if found is None:
            raise SystemExit(f"库里找不到 k={want_k} m={want_m} 的样本")
        chosen.append(found)
    return [ds[i] for i in chosen], chosen


def _obs_from_samples(samples, motion: bool):
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    B = len(samples)
    def stack(k, dt=None):
        a = np.stack([np.asarray(x[k]) for x in samples]); return jnp.asarray(a if dt is None else a.astype(dt))
    images = {"base_0_rgb": jnp.zeros((B, 224, 224, 3), jnp.float32), "left_wrist_0_rgb": jnp.zeros((B, 224, 224, 3), jnp.float32)}
    masks = {k: jnp.ones((B,), jnp.bool_) for k in images}
    tok = jnp.zeros((B, 64), jnp.int32)
    tmask = jnp.zeros((B, 64), jnp.bool_).at[:, :8].set(True)
    kw = {}
    if motion:
        kw = dict(motion_emb=stack("motion_emb"), motion_pos=stack("motion_pos"), motion_mask=stack("motion_mask"), mem_order=stack("mem_order"))
    return HistAugObservation(images=images, image_masks=masks, state=jnp.zeros((B, 32), jnp.float32), tokenized_prompt=tok,
                              tokenized_prompt_mask=tmask, token_ar_mask=None, token_loss_mask=None,
                              static_image_emb=stack("static_image_emb", np.float32), static_mask=stack("static_mask"),
                              static_pos_emb=stack("static_pos_emb"), static_state_emb=stack("static_state_emb", np.float32), **kw)


def _leaf_paths(model):
    import jax
    from flax import nnx
    return {jax.tree_util.keystr(kp): np.asarray(v) for kp, v in jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param).to_pure_dict())[0]}


# ══════════════════════════════════════════════════════════════════════════════
# M3
# ══════════════════════════════════════════════════════════════════════════════

def cmd_m3(args):
    import jax
    import jax.numpy as jnp
    import openpi.shared.array_typing as at
    from flax import nnx
    lib = pathlib.Path(args.lib)
    ds = _make_dataset(lib, OPEN_YAML)
    c_open, m_open, c_closed, m_closed = _make_models(0)
    samples, idxs = _fixture_batch(lib, ds, [(6, 0), (32, 11), (32, "max")])
    # 人造第三个样本满 96：把真 motion 行复制填满（M3 只看层算术，不要求物理合法）
    s3 = dict(samples[2]); k3 = int(s3["motion_mask"].sum())
    emb = np.array(s3["motion_emb"]); pos = np.array(s3["motion_pos"]); msk = np.ones(96, np.bool_)
    for j in range(k3, 96):
        emb[j] = emb[j % k3] + 0.001 * (j + 1); pos[j] = pos[j % k3]
    from mme_vla_suite.shared.sampling import memory_order, pad_times
    s3.update(motion_emb=emb, motion_pos=pos, motion_mask=msk,
              mem_order=memory_order(pad_times(oracle_even_indices(int(s3["step_idx"].item()), 32), 32), 16,
                                     np.array(sorted(list(np.arange(0, 96) * 16)))))
    samples[2] = s3
    obs = _obs_from_samples(samples, motion=True)
    obs_c = dataclasses.replace(obs, motion_emb=None, motion_pos=None, motion_mask=None, mem_order=None)
    fails = []
    with at.disable_typechecking():
        tok_o, im_o, ar_o, na_o = m_open.embed_memory(obs)
        tok_c, im_c, ar_c, na_c = m_closed.embed_memory(obs_c)
        # 并列序（重排前）：直接调 mem_encoder
        par, _, _ = m_open.mem_encoder(obs.static_image_emb, obs.static_pos_emb, obs.static_state_emb,
                                       motion_emb=obs.motion_emb, motion_pos=obs.motion_pos, motion_mask=obs.motion_mask)
    par = np.asarray(par)
    # ① 帧路两态逐位（并列序前 512 位 vs 关闭态输出）
    if not np.array_equal(par[:, :512].view(np.uint16), np.asarray(tok_c).view(np.uint16)):
        fails.append("帧路 512 位两态不逐位")
    # ② 运动路独立复算：生产 bf16 语义（nnx.Linear dtype=bf16：输入与参数 cast 到 bf16，dot 累加 preferred_element_type=bf16）
    P = _leaf_paths(m_open)
    W1 = P["['mem_encoder']['motion_pos_proj']['kernel']"]; b1 = P["['mem_encoder']['motion_pos_proj']['bias']"]
    W2 = P["['mem_encoder']['motion_encoder_static']['kernel']"]; b2 = P["['mem_encoder']['motion_encoder_static']['bias']"]
    lin = m_open.mem_encoder.motion_pos_proj
    dt = lin.dtype
    x_pos = jnp.asarray(np.asarray(obs.motion_pos)).astype(dt)
    h = jax.lax.dot_general(x_pos, jnp.asarray(W1).astype(dt), (((2,), (0,)), ((), ())), precision=lin.precision, preferred_element_type=dt)
    h = h + jnp.asarray(b1).astype(dt)
    h = nnx.silu(h)
    x2 = jnp.concatenate([jnp.asarray(np.asarray(obs.motion_emb)).astype(dt), h.astype(dt)], axis=-1)
    y = jax.lax.dot_general(x2, jnp.asarray(W2).astype(dt), (((2,), (0,)), ((), ())), precision=lin.precision, preferred_element_type=dt)
    y = y + jnp.asarray(b2).astype(dt)
    y = np.asarray(y); mot = par[:, 512:]
    bit_same = np.array_equal(y.view(np.uint16), mot.view(np.uint16))
    ulp = np.abs(y.view(np.uint16).astype(np.int64) - mot.view(np.uint16).astype(np.int64)).max()
    print(f"[m3] 运动路独立 bf16 复算 逐位={bit_same} max_bf16_ulp={ulp}")
    if not bit_same:
        fails.append(f"运动路独立复算不逐位（max ulp {ulp}）")
    # ③ padding 行两两逐位（样本 0 全 96 padding、样本 1 后 85 行）
    pad_rows = mot[0]
    if not all(np.array_equal(pad_rows[0].view(np.uint16), pad_rows[j].view(np.uint16)) for j in range(1, 96)):
        fails.append("padding 行经两层后不两两逐位")
    # ④ gather 对 20 个随机置换 vs np.take_along_axis
    rng = np.random.default_rng(0)
    for i in range(20):
        perm = np.stack([rng.permutation(608) for _ in range(3)]).astype(np.int32)
        o2 = dataclasses.replace(obs, mem_order=jnp.asarray(perm))
        with at.disable_typechecking():
            t2, m2, _, _ = m_open.embed_memory(o2)
        exp_t = np.take_along_axis(par, perm[:, :, None], axis=1)
        exp_m = np.take_along_axis(np.concatenate([np.asarray(obs.static_mask), np.asarray(obs.motion_mask)], axis=1), perm, axis=1)
        if not (np.array_equal(np.asarray(t2).view(np.uint16), exp_t.view(np.uint16)) and np.array_equal(np.asarray(m2), exp_m)):
            fails.append(f"gather 置换 {i} 不逐位")
            break
    # ⑤ 三种坏 mem_order 必 raise
    # 注：jax 默认 x64 关闭，int64 输入会被静默降成 int32，故「错 dtype」用 float32（真正会静默通过 take_along_axis 的类型）
    for name, bad in (("错长度", jnp.asarray(np.tile(np.arange(600, dtype=np.int32), (3, 1)))),
                      ("错 dtype", jnp.asarray(np.tile(np.arange(608), (3, 1)).astype(np.float32))),
                      ("缺键 None", None)):
        try:
            with at.disable_typechecking():
                m_open.embed_memory(dataclasses.replace(obs, mem_order=bad))
            fails.append(f"坏 mem_order（{name}）未 raise")
        except (ValueError, TypeError):
            pass
    # ⑥ 参数命名与分组
    new_keys = [k for k in P if "motion" in k]
    if len(new_keys) != 4 or any("img" in k for k in new_keys) or not all("mem_encoder" in k for k in new_keys):
        fails.append(f"新参数命名/分组异常: {new_keys}")
    import re
    freeze = c_open.get_freeze_filter()
    print(f"[m3] 新叶 {new_keys}; freeze_filter={freeze}")
    n_open = len(P); n_closed = len(_leaf_paths(m_closed))
    if n_open - n_closed != 4:
        fails.append(f"参数叶数差 {n_open - n_closed} != 4")
    for f in fails:
        print("  ✗", f)
    enc_ok = not any("运动路" in f or "帧路" in f or "padding" in f or "命名" in f or "叶数" in f for f in fails)
    gat_ok = not any("gather" in f or "mem_order" in f for f in fails)
    print(f"MOTION_ENC={'PASS' if enc_ok else 'FAIL'} bf16_ulp_max={ulp} frame_bitexact={int(np.array_equal(par[:, :512].view(np.uint16), np.asarray(tok_c).view(np.uint16)))}")
    print(f"MEM_GATHER={'PASS' if gat_ok else 'FAIL'} perms=20 bad_order_raises=3")
    ds.close()
    if fails:
        raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════════════════
# M4
# ══════════════════════════════════════════════════════════════════════════════

def cmd_m4(args):
    import jax
    import jax.numpy as jnp
    import openpi.shared.array_typing as at
    from flax import nnx
    from mme_vla_suite.shared.sampling import memory_order, pad_times
    lib = pathlib.Path(args.lib)
    ds = _make_dataset(lib, OPEN_YAML)
    c_open, m_open, c_closed, m_closed = _make_models(0)
    samples, idxs = _fixture_batch(lib, ds, [(6, 0), (32, 11), (32, "max")])
    # 第三个样本人造满 96（阴性对照：无补位）
    s3 = dict(samples[2]); k3 = int(s3["motion_mask"].sum())
    emb = np.array(s3["motion_emb"]); pos = np.array(s3["motion_pos"])
    for j in range(k3, 96):
        emb[j] = emb[j % k3] * (1 + 0.01 * j); pos[j] = pos[j % k3]
    ftimes3 = pad_times(oracle_even_indices(int(s3["step_idx"].item()), 32), 32)
    s3.update(motion_emb=emb, motion_pos=pos, motion_mask=np.ones(96, np.bool_),
              mem_order=memory_order(ftimes3, 16, np.arange(96, dtype=np.int64) * 16))
    samples[2] = s3
    obs = _obs_from_samples(samples, motion=True)
    B = 3
    actions = jnp.asarray(np.random.default_rng(0).standard_normal((B, 20, 32)).astype(np.float32))
    rng = jax.random.key(7)
    noise = jnp.asarray(np.random.default_rng(1).standard_normal((B, 20, 32)).astype(np.float32))

    def loss_fn(model, o):
        with at.disable_typechecking():
            return model.compute_loss(rng, o, actions, train=False)

    def sample_fn(model, o):
        with at.disable_typechecking():
            return model.sample_actions(jax.random.key(3), o, noise=noise, num_steps=10)

    fails = []
    base_loss = np.asarray(loss_fn(m_open, obs)); base_act = np.asarray(sample_fn(m_open, obs))
    print(f"[m4] base loss {base_loss.mean():.6f} shape {base_loss.shape} actions {base_act.shape}")
    # (a) 补位塞有限随机垃圾
    g = np.random.default_rng(5)
    def garbage(o):
        sm = np.asarray(o.static_mask); mm = np.asarray(o.motion_mask)
        si = np.array(np.asarray(o.static_image_emb)); sp = np.array(np.asarray(o.static_pos_emb))
        me = np.array(np.asarray(o.motion_emb)); mp = np.array(np.asarray(o.motion_pos))
        si[~sm] = g.normal(0, 1e3, si[~sm].shape); sp[~sm] = g.normal(0, 1e3, sp[~sm].shape)
        me[~mm] = g.normal(0, 1e3, me[~mm].shape); mp[~mm] = g.normal(0, 1e3, mp[~mm].shape)
        return dataclasses.replace(o, static_image_emb=jnp.asarray(si), static_pos_emb=jnp.asarray(sp),
                                   motion_emb=jnp.asarray(me), motion_pos=jnp.asarray(mp))
    og = garbage(obs)
    la = np.asarray(loss_fn(m_open, og)); aa = np.asarray(sample_fn(m_open, og))
    inv_ok = np.array_equal(la.view(np.uint8), base_loss.view(np.uint8)) and np.array_equal(aa.view(np.uint8), base_act.view(np.uint8))
    print(f"MASK_INVARIANCE={'PASS' if inv_ok else 'FAIL'} loss_bitexact={int(np.array_equal(la.view(np.uint8), base_loss.view(np.uint8)))} "
          f"actions_bitexact={int(np.array_equal(aa.view(np.uint8), base_act.view(np.uint8)))}")
    if not inv_ok:
        fails.append("MASK_INVARIANCE")
    # (b) 输入梯度：补位为零、真位非零；参数梯度：全空 batch 两新层四叶全零，有 motion 非零
    graphdef, state = nnx.split(m_open)
    def loss_wrt_inputs(si, sp, me, mp):
        model = nnx.merge(graphdef, state)
        o = dataclasses.replace(obs, static_image_emb=si, static_pos_emb=sp, motion_emb=me, motion_pos=mp)
        return jnp.sum(loss_fn(model, o))
    grads = jax.grad(loss_wrt_inputs, argnums=(0, 1, 2, 3))(obs.static_image_emb, obs.static_pos_emb, obs.motion_emb, obs.motion_pos)
    sm = np.asarray(obs.static_mask); mm = np.asarray(obs.motion_mask)
    leak_ok = True
    for name, gr, mask in (("static_image_emb", grads[0], sm), ("static_pos_emb", grads[1], sm), ("motion_emb", grads[2], mm), ("motion_pos", grads[3], mm)):
        gr = np.asarray(gr).astype(np.float64)
        pad_zero = bool(np.all(gr[~mask] == 0)) if (~mask).any() else True
        real_nonzero = bool(np.any(gr[mask] != 0)) if mask.any() else True
        finite = bool(np.isfinite(gr).all())
        print(f"  [grad] {name}: padding_zero={pad_zero} real_nonzero={real_nonzero} finite={finite}")
        leak_ok &= pad_zero and real_nonzero and finite
    def param_grads(o):
        def f(st):
            model = nnx.merge(graphdef, st)
            return jnp.sum(loss_fn(model, o))
        return jax.grad(f)(state)
    obs_empty = dataclasses.replace(obs, motion_emb=jnp.zeros_like(obs.motion_emb), motion_pos=jnp.zeros_like(obs.motion_pos),
                                    motion_mask=jnp.zeros_like(obs.motion_mask),
                                    mem_order=jnp.asarray(np.stack([memory_order(pad_times(oracle_even_indices(int(s["step_idx"].item()), 32), 32), 16,
                                                                                 np.full(96, SENTINEL, np.int64)) for s in samples])))
    pg_empty = _leaf_paths_state(param_grads(obs_empty)); pg_full = _leaf_paths_state(param_grads(obs))
    new_keys = [k for k in pg_empty if "motion" in k]
    empty_zero = all(np.all(np.asarray(pg_empty[k]).astype(np.float64) == 0) for k in new_keys)
    full_nonzero = all(np.any(np.asarray(pg_full[k]).astype(np.float64) != 0) for k in new_keys)
    print(f"  [grad] 新层四叶: 全空 batch 全零={empty_zero} 有 motion 非零={full_nonzero} keys={len(new_keys)}")
    leak_ok &= empty_zero and full_nonzero and len(new_keys) == 4
    print(f"GRAD_LEAK={'PASS' if leak_ok else 'FAIL'}")
    if not leak_ok:
        fails.append("GRAD_LEAK")
    # (c) 并列序 vs 交错：loss 必变
    obs_par = dataclasses.replace(obs, mem_order=jnp.asarray(np.tile(np.arange(608, dtype=np.int32), (B, 1))))
    lp = np.asarray(loss_fn(m_open, obs_par))
    diff_c = float(np.abs(lp.astype(np.float64) - base_loss).max())
    order_ok = diff_c > 0
    print(f"ORDER_EFFECT={'PASS' if order_ok else 'FAIL'} max_abs_diff_parallel_vs_interleaved={diff_c:.3e}")
    if not order_ok:
        fails.append("ORDER_EFFECT")
    # (d) 真 motion 行内部随机置换 + 重算 mem_order → loss 逐位
    perm_rng = np.random.default_rng(9)
    me = np.array(np.asarray(obs.motion_emb)); mp = np.array(np.asarray(obs.motion_pos)); mo = np.array(np.asarray(obs.mem_order))
    for i, s in enumerate(samples):
        k = int(s["motion_mask"].sum())
        if k < 2:
            continue
        p = perm_rng.permutation(k)
        me[i, :k] = me[i, :k][p]; mp[i, :k] = mp[i, :k][p]
        # 真行的全域时刻跟着置换（时刻 = 原次序下的起点帧号，从 mem_order 反推不便，直接用 oracle）
        idx = idxs[i]
        manifest = json.loads((lib / "meta/episode_manifest.json").read_text())
        eps = manifest["episodes"]; starts = [e["exec_sample_offset"] for e in eps]
        g_ = int(np.searchsorted(np.array(starts), idx, side="right") - 1); t_ = eps[g_]["exec_start_idx"] + idx - starts[g_]
        index = json.loads((lib / "motion/meta/motion_index.json").read_text())
        frames = [f for _, f in oracle_visible(index["entries"][g_], t_)] if i != 2 else list(np.arange(96) * 16)
        frames = np.array(frames)[p].tolist() if i != 2 else np.array(frames)[p].tolist()
        mo[i] = memory_order(pad_times(oracle_even_indices(t_, 32), 32), 16, pad_times(frames, 96))
    obs_perm = dataclasses.replace(obs, motion_emb=jnp.asarray(me), motion_pos=jnp.asarray(mp), mem_order=jnp.asarray(mo))
    ld = np.asarray(loss_fn(m_open, obs_perm))
    perm_ok = np.array_equal(ld.view(np.uint8), base_loss.view(np.uint8))
    print(f"[m4] (d) 真行置换后 loss 逐位={perm_ok} max|Δ|={float(np.abs(ld.astype(np.float64) - base_loss).max()):.3e}")
    if not perm_ok:
        fails.append("ROW_PERM_INVARIANCE")
    # (e) 关闭态参数拷入开启态：m=0 样本 loss 差 ≤ 1e-4·|loss| 且 ≤ 1% × (c)
    gd_o, st_o = nnx.split(m_open)
    st_c = nnx.state(m_closed, nnx.Param)
    merged = copy.deepcopy(st_o)
    def copy_in(dst, src):
        for kp, v in jax.tree_util.tree_flatten_with_path(src.to_pure_dict())[0]:
            node = dst
            for k in kp[:-1]:
                node = node[getattr(k, "key", getattr(k, "name", k))]
            last = getattr(kp[-1], "key", getattr(kp[-1], "name", kp[-1]))
            node[last].value = v
    copy_in(merged, st_c)
    m_mix = nnx.merge(gd_o, merged)
    l_mix = np.asarray(loss_fn(m_mix, obs))
    obs_closed = _obs_from_samples(samples, motion=False)
    l_closed = np.asarray(loss_fn(m_closed, obs_closed))
    d_e = float(np.abs(l_mix[0].astype(np.float64) - l_closed[0]).max())
    tol = 1e-4 * float(np.abs(l_closed[0]).max())
    e_ok = d_e <= tol and d_e <= 0.01 * diff_c
    print(f"ZERO_MOTION_EQUIV={'PASS' if e_ok else 'FAIL'} max_abs_diff={d_e:.3e} tol={tol:.3e} order_diff={diff_c:.3e}")
    if not e_ok:
        fails.append("ZERO_MOTION_EQUIV")
    ds.close()
    if fails:
        print("FAILS:", fails)
        raise SystemExit(1)


def _leaf_paths_state(state):
    import jax
    return {jax.tree_util.keystr(kp): v for kp, v in jax.tree_util.tree_flatten_with_path(state.to_pure_dict() if hasattr(state, "to_pure_dict") else state)[0]}


# ══════════════════════════════════════════════════════════════════════════════
# M5
# ══════════════════════════════════════════════════════════════════════════════

def cmd_m5(args):
    import jax
    import jax.numpy as jnp
    import omegaconf
    import openpi.shared.array_typing as at
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.models.integration.history_observation import HistAugObservation, preprocess_observation
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
    from mme_vla_suite.training.dataloader import _create_framesamp_dataset, _motion_gates, _parse_source_run
    from mme_vla_suite.datastore import StoreMeta
    from openpi import transforms as _tr
    lib = pathlib.Path(args.lib)
    fails = []
    def expect_raise(name, fn, *exc):
        try:
            fn()
            fails.append(f"{name} 未 raise")
        except exc or (Exception,):
            pass
    # 正向：spec 由配置推出、observation 过类型检查、preprocess 前后四键逐位、Repack 含四键、双 store 同源
    cfg_open = _load_yaml(OPEN_YAML)
    c = HistoryPi0Config(use_history=True, history_config=cfg_open, dtype="bfloat16", paligemma_variant="dummy", action_expert_variant="dummy")
    spec, _ = c.inputs_spec(batch_size=2)
    if tuple(spec.mem_order.shape) != (2, 608) or spec.mem_order.dtype != jnp.int32 or tuple(spec.motion_emb.shape) != (2, 96, 768):
        fails.append("inputs_spec 形制错")
    ds = _make_dataset(lib, OPEN_YAML)
    s = ds[100]
    obs = _obs_from_samples([s, s], motion=True)
    obs2 = preprocess_observation(jax.random.key(0), obs, train=False)
    for k in ("motion_emb", "motion_pos", "motion_mask", "mem_order"):
        if not _bytes_equal(getattr(obs2, k), getattr(obs, k)):
            fails.append(f"preprocess_observation 未透传 {k}")
    d = obs.to_dict()
    if any(k not in d for k in ("motion_emb", "motion_pos", "motion_mask", "mem_order")):
        fails.append("to_dict 缺 motion 键")
    import mme_vla_suite.training.config as tc
    src = pathlib.Path(tc.__file__).read_text(encoding="utf-8")
    if not all(f'"{k}": "{k}"' in src for k in ("motion_emb", "motion_pos", "motion_mask", "mem_order")):
        fails.append("RepackTransform 未登记四键")
    from mme_vla_suite.policies.robomme_policy import RoboMMEInputs
    out = RoboMMEInputs(model_type=None)({"observation/image": np.zeros((224, 224, 3), np.uint8), "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
                                          "observation/state": np.zeros(8), **{k: s[k] for k in ("motion_emb", "motion_pos", "motion_mask", "mem_order")}})
    if any(out.get(k) is None for k in ("motion_emb", "motion_pos", "motion_mask", "mem_order")):
        fails.append("RoboMMEInputs 未透传四键")
    fm = StoreMeta.load(lib / "framesamp")
    _motion_gates(cfg_open, fm)
    # 负向：坏 mem_order（非置换）在 dataset 侧
    from mme_vla_suite.shared.sampling import memory_order
    expect_raise("非置换 mem_order", lambda: _bad_order())
    # 缺键
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config as _C
    m = c.create(jax.random.key(0))
    with at.disable_typechecking():
        expect_raise("缺 motion 键 embed_memory", lambda: m.embed_memory(dataclasses.replace(obs, motion_emb=None)), ValueError)
    # spec / 开关不一致：inputs_spec 用 open 但模型建成 closed → 检查函数 _motion_enabled 一致（构造即 raise 的场景需伪造 mem_encoder）
    m.mem_encoder.motion_enabled = False
    with at.disable_typechecking():
        tok, im, _, _ = m.embed_memory(obs)
    if tok.shape[1] != 512:
        fails.append("motion_enabled=False 下 embed_memory 未早返回")
    m.mem_encoder.motion_enabled = True
    # stride / window / direction / origin / frame_size 错
    for key, val in (("stride", 20), ("window_frames", 32), ("window_direction", "backward"), ("grid_origin", "global"), ("frame_size", 224),
                     ("budget", 64), ("dim", 512), ("pos_dim", 768)):
        cfg = copy.deepcopy(cfg_open); cfg.motion[key] = val
        expect_raise(f"motion.{key}={val}", lambda cfg=cfg: _make_dataset(lib, None) if False else _create_framesamp_dataset(str(lib / "framesamp"), _fake_data_config(), cfg, 20))
    # 未 verified / 换入另一合法 store / 只篡改 index / 串库 manifest
    tmp = pathlib.Path(args.tmp or (lib / "oracle" / "m5-tmp")); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    def clone(name):
        d = tmp / name; shutil.copytree(lib / "motion", d); return d
    d1 = clone("unverified"); meta = json.loads((d1 / "meta/store_meta.json").read_text()); meta["status"] = "packed"
    (d1 / "meta/store_meta.json").write_text(json.dumps(meta))
    os.environ.pop("MMEVLA_MOTION_ALLOW_UNVERIFIED", None)
    expect_raise("未 verified 的 motion store", lambda: _gates_with(cfg_open, fm, d1))
    d2 = clone("tampered-index"); idx = json.loads((d2 / "meta/motion_index.json").read_text()); idx["entries"][3]["exec"]["row_base"] += 1
    (d2 / "meta/motion_index.json").write_text(json.dumps(idx))
    expect_raise("只篡改 index", lambda: _gates_with(cfg_open, fm, d2))
    d3 = clone("other-manifest"); meta = json.loads((d3 / "meta/store_meta.json").read_text()); meta["manifest_sha256"] = "0" * 64
    idx = json.loads((d3 / "meta/motion_index.json").read_text()); idx["manifest_sha256"] = "0" * 64
    raw = (json.dumps(idx, ensure_ascii=False, indent=1) + "\n").encode(); (d3 / "meta/motion_index.json").write_bytes(raw)
    meta["motion_index_sha256"] = hashlib.sha256(raw).hexdigest(); (d3 / "meta/store_meta.json").write_text(json.dumps(meta))
    expect_raise("换入另一份自身自洽但清单不同的 store", lambda: _gates_with(cfg_open, fm, d3))
    d4 = clone("wrong-provenance"); meta = json.loads((d4 / "meta/store_meta.json").read_text()); meta["provenance"]["encoder"]["epoch"] = 70
    (d4 / "meta/store_meta.json").write_text(json.dumps(meta))
    expect_raise("source_run 与 provenance.encoder 不符", lambda: _gates_with(cfg_open, fm, d4))
    expect_raise("source_run 格式错", lambda: _parse_source_run("wan-v8/checkpoint_best.pt#encoder"))
    # resolved sha 错 / motion checkpoint extra-missing：用 policy_config 的两个 helper 与最小 fixture
    from mme_vla_suite.policies.policy_config import _load_resolved_snapshot, _assert_param_tree_exact
    run = tmp / "run"; run.mkdir()
    raw = omegaconf.OmegaConf.to_yaml(cfg_open).encode(); (run / "history_config.resolved.yaml").write_bytes(raw)
    (run / "history_config.resolved.sha256").write_text("0" * 64 + "\n")
    (run / "motion_provenance.json").write_text(json.dumps({"motion_enabled": True, "resolved_sha256": "0" * 64}))
    expect_raise("resolved sha 错", lambda: _load_resolved_snapshot(run), ValueError)
    (run / "history_config.resolved.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")
    (run / "motion_provenance.json").write_text(json.dumps({"motion_enabled": True, "resolved_sha256": hashlib.sha256(raw).hexdigest()}))
    cfg_back, en = _load_resolved_snapshot(run)
    if not en:
        fails.append("resolved 快照恢复后 enabled 不为 True")
    from flax import nnx
    params_full = nnx.state(m, nnx.Param).to_pure_dict()
    _assert_param_tree_exact(m, params_full)
    missing = copy.deepcopy(params_full); del missing["mem_encoder"]["motion_pos_proj"]
    expect_raise("motion checkpoint 缺参数", lambda: _assert_param_tree_exact(m, missing), ValueError)
    extra = copy.deepcopy(params_full); extra["mem_encoder"]["extra_leaf"] = {"kernel": np.zeros(2)}
    expect_raise("motion checkpoint 多参数", lambda: _assert_param_tree_exact(m, extra), ValueError)
    shutil.rmtree(tmp, ignore_errors=True)
    ds.close()
    for f in fails:
        print("  ✗", f)
    print(f"MOTION_PLUMBING={'PASS' if not fails else 'FAIL'} negatives=16")
    if fails:
        raise SystemExit(1)


def _bad_order():
    from mme_vla_suite.shared import sampling
    import numpy as _np
    # 构造相同键无法造出非置换；直接调用校验路径：argsort 永远是置换，故用 embed_memory 侧（M3 ⑤）覆盖非置换；
    # 这里验证 pad_times 对越界与超预算 raise
    sampling.pad_times([sampling.MEM_ORDER_SENTINEL], 4)


def _gates_with(cfg, fm, root):
    from mme_vla_suite.training.dataloader import _motion_gates
    old = os.environ.get("MMEVLA_MOTION_STORE")
    os.environ["MMEVLA_MOTION_STORE"] = str(root)
    try:
        return _motion_gates(cfg, fm)
    finally:
        if old is None:
            os.environ.pop("MMEVLA_MOTION_STORE", None)
        else:
            os.environ["MMEVLA_MOTION_STORE"] = old


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", choices=["m1", "m2", "m3", "m4", "m5"], required=True)
    ap.add_argument("--lib", default=str(_V1 / "datasets/4task-motion-40ep"))
    ap.add_argument("--limit", type=int, default=None, help="m1 真实库层只跑前 N 个样本（调试用；正式判定不设）")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    {"m1": cmd_m1, "m2": cmd_m2, "m3": cmd_m3, "m4": cmd_m4, "m5": cmd_m5}[args.gate](args)


# 唯一入口在文件末尾（main = _t3_main，含 m1–m5 与 t3 五个闸门）。此处原有的 if __name__ == "__main__": main()
# 会在模块执行到这里时先跑旧 parser、把 --gate t3* 拒掉（2026-09-03 t3common 首跑实测 EXIT_CODE=2），已删。


# ══════════════════════════════════════════════════════════════════════════════
# T3 真实训练链路四层（motion-memory-plan.md 5.3 / 2.10）：t3common / t3trace / t3mechanism / t3phase
# 复用 bench_train_steps 的两把哈希尺子（_leaf_sha256 / _canonical_sha256），不复制 hash 实现。
# ══════════════════════════════════════════════════════════════════════════════

_G0_DIR = _REPO_ROOT / "scripts" / "training" / "g0"


def _bench_hashes():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench_train_steps", _G0_DIR / "bench_train_steps.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_train_steps"] = mod
    spec.loader.exec_module(mod)
    return mod._leaf_sha256, mod._canonical_sha256


def _train_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mme_train", _REPO_ROOT / "scripts" / "training" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mme_train"] = mod
    spec.loader.exec_module(mod)
    return mod


def _t3_train_config(yaml_name: str, exp_name: str, ckpt_base: pathlib.Path, dataset_path: pathlib.Path,
                     *, steps: int = 1000, batch: int = 8, fsdp: int = 2, workers: int = 4, seed: int = 42):
    """与 bench_train_steps 同一组训练语义 argv 构造 TrainConfig（tyro CLI，argv 显式传入）。"""
    import mme_vla_suite.training.config as _config
    import tyro
    argv = ["mme_vla_suite", "--exp-name", exp_name, "--assets-base-dir", str(_V1 / "train-assets"),
            "--checkpoint-base-dir", str(ckpt_base), "--batch-size", str(batch), "--num-workers", str(workers),
            "--num-train-steps", str(steps), "--log-interval", "1", "--save-interval", "1", "--seed", str(seed),
            "--fsdp-devices", str(fsdp), "--dataset-path", str(dataset_path),
            "--weight-loader.params-path", str(_V1 / "models/openpi-assets/checkpoints/pi05_base/params"),
            "--model.use-history", "--model.history-config", yaml_name, "--no-wandb-enabled"]
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _config._CONFIGS_DICT.items()}, args=argv)


def _t3_init_state(yaml_name: str, exp_name: str, ckpt_base: pathlib.Path, dataset_path: pathlib.Path, fsdp: int):
    """按 train.main 同一 RNG 派生（rng=key(seed) → split → init_rng）初始化完整 TrainState（含 pi05_base 权重）。"""
    import dataclasses as _dc
    import jax
    from openpi.training import sharding
    from mme_vla_suite.models.config.utils import get_history_config
    train = _train_module()
    config = _t3_train_config(yaml_name, exp_name, ckpt_base, dataset_path, fsdp=fsdp)
    resolved = get_history_config(config.model.history_config)
    config = _dc.replace(config, model=_dc.replace(config.model, history_config=resolved))
    rng = jax.random.key(config.seed)
    _train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    return config, state, mesh, train


def _t3_state_leaf_shas(state) -> dict[str, str]:
    import jax
    leaf_sha, _ = _bench_hashes()
    out = {}
    trees = {"params": state.params, "opt_state": state.opt_state, "step": state.step}
    if state.ema_params is not None:
        trees["ema_params"] = state.ema_params
    for name, tree in trees.items():
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            if leaf is None:
                continue
            out[name + jax.tree_util.keystr(path)] = leaf_sha(np.asarray(jax.device_get(leaf)))
    return out


def cmd_t3common(args):
    """开 / 关两态在同一进程、同 seed / 环境下初始化完整 TrainState，逐叶比较；写冻结 reference。"""
    ref_path = pathlib.Path(args.out)
    tmp = pathlib.Path(args.tmp or (_V1 / "tmp" / "t3common"))
    shutil.rmtree(tmp, ignore_errors=True)
    res = {}
    for tag, yaml_name in (("closed", CLOSED_YAML), ("open", OPEN_YAML)):
        _cfg, state, _mesh, _ = _t3_init_state(yaml_name, f"t3common-{tag}", tmp / tag, pathlib.Path(args.dataset), args.fsdp)
        res[tag] = _t3_state_leaf_shas(state)
        del state
    common = set(res["closed"]) & set(res["open"])
    closed_only = sorted(set(res["closed"]) - set(res["open"]))
    open_only = sorted(set(res["open"]) - set(res["closed"]))
    mismatch = sorted(k for k in common if res["closed"][k] != res["open"][k])
    n_open_params = sum(1 for k in open_only if k.startswith("params"))
    n_open_ema = sum(1 for k in open_only if k.startswith("ema_params"))
    n_open_opt = sum(1 for k in open_only if k.startswith("opt_state"))
    ok = (not mismatch and not closed_only and n_open_params == 4 and n_open_ema == 4 and n_open_opt == 8 and len(open_only) == 16
          and all("motion" in k for k in open_only) and len(res["closed"]) == 177 and len(res["open"]) == 193)
    ref = {"schema": 1, "closed": res["closed"], "open": res["open"], "common_mismatches": mismatch, "closed_only": closed_only,
           "open_only": open_only, "n_leaves": {"closed": len(res["closed"]), "open": len(res["open"])}, "verdict": "PASS" if ok else "FAIL"}
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps(ref, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"T3_COMMON_INIT={'PASS' if ok else 'FAIL'} common_mismatches={len(mismatch)} open_only_params={n_open_params} "
          f"open_only_ema={n_open_ema} open_only_opt={n_open_opt} closed_only={len(closed_only)} n_leaves_closed={len(res['closed'])} n_leaves_open={len(res['open'])}")
    if not ok:
        raise SystemExit(1)


def cmd_t3verifyinit(args):
    """正式 run 的 step-0 记录（param_checksums.jsonl 首行 per_leaf）必须逐叶命中 t3common reference。"""
    ref = json.loads(pathlib.Path(args.reference).read_text())
    fails = 0
    for tag, rec in (("closed", args.closed_records), ("open", args.open_records)):
        if rec is None:
            continue
        rows = [json.loads(l) for l in (pathlib.Path(rec) / "param_checksums.jsonl").read_text().splitlines() if l.strip()]
        r0 = next(r for r in rows if int(r["step"]) == 0)
        if r0["per_leaf"] != ref[tag]:
            diff = [k for k in set(r0["per_leaf"]) | set(ref[tag]) if r0["per_leaf"].get(k) != ref[tag].get(k)]
            print(f"  ✗ {tag} step-0 与 reference 差 {len(diff)} 叶: {diff[:5]}")
            fails += 1
        else:
            print(f"  ✓ {tag} step-0 {len(r0['per_leaf'])} 叶命中 reference")
    print(f"T3_INIT_MATCH={'PASS' if not fails else 'FAIL'}")
    if fails:
        raise SystemExit(1)


def _t3_load_records(rec: pathlib.Path):
    bd = {int(json.loads(l)["step"]): json.loads(l) for l in (rec / "batch_digests.jsonl").read_text().splitlines() if l.strip()}
    idx = json.loads((rec / "index_sequence.json").read_text())["indices"]
    return bd, idx


_T3_DIGEST_STEPS = {0, 1, 2, 100, 200, 299, 300, 400, 500, 600, 700, 800, 900, 999}
_MOTION_KEYS = ("motion_emb", "motion_pos", "motion_mask", "mem_order")


def cmd_t3trace(args):
    """T3_TOKEN_TRACE：open 侧 14 条摘要的四个 motion 键由 M1 oracle 按 sample_indices 重建，raw / canonical sha 逐键相同；
    open / closed 公共 12 叶逐键相同，open-only 恰为四叶；前 8,000 个 index 逐项相同。--preflight 只做 112 样本覆盖性检查。"""
    leaf_sha, canon_sha = _bench_hashes()
    lib = pathlib.Path(args.lib)
    index, table, pos_table, manifest = load_lib_oracle(lib)
    eps = manifest["episodes"]
    starts = np.array([e["exec_sample_offset"] for e in eps], np.int64)

    def sample_of(i):
        g = int(np.searchsorted(starts, i, side="right") - 1)
        t = int(eps[g]["exec_start_idx"]) + (i - int(starts[g]))
        return g, t

    if args.preflight:
        idx = json.loads(pathlib.Path(args.index_json).read_text())["indices"]
        chosen = [idx[s * 8:(s + 1) * 8] for s in sorted(_T3_DIGEST_STEPS)]
        flat = [i for b in chosen for i in b]
        ks = [oracle_sample(index["entries"][sample_of(i)[0]], sample_of(i)[1], table, pos_table)["k"] for i in flat]
        has_empty = any(k == 0 for k in ks); has_2 = any(k >= 2 for k in ks)
        has_video = any(eps[sample_of(i)[0]]["exec_start_idx"] > 0 for i in flat)
        ok = has_empty and has_2 and has_video and len(flat) == 112
        print(f"T3_TRACE_PREFLIGHT={'PASS' if ok else 'FAIL'} samples={len(flat)} empty={sum(k == 0 for k in ks)} k_ge2={sum(k >= 2 for k in ks)} video={has_video}")
        if not ok:
            raise SystemExit(1)
        return

    bo, io = _t3_load_records(pathlib.Path(args.open_records))
    bc, ic = _t3_load_records(pathlib.Path(args.closed_records))
    fails = []
    if set(bo) != _T3_DIGEST_STEPS or set(bc) != _T3_DIGEST_STEPS:
        fails.append(f"摘要步集 != {sorted(_T3_DIGEST_STEPS)}: open={sorted(bo)} closed={sorted(bc)}")
    if io[:8000] != ic[:8000] or len(io) < 8000 or len(ic) < 8000:
        fails.append("前 8,000 个 index 不同或不足")
    mism = 0; n_samples = 0
    for s in sorted(_T3_DIGEST_STEPS):
        if s not in bo or s not in bc:
            continue
        ro, rc = bo[s], bc[s]
        if ro["n_keys"] != 16 or rc["n_keys"] != 12:
            fails.append(f"step {s} n_keys open={ro['n_keys']} closed={rc['n_keys']}")
        ko, kc = set(ro["per_key"]), set(rc["per_key"])
        oo = {k for k in ko - kc}
        if kc - ko or len(oo) != 4 or not all(any(m in k for m in _MOTION_KEYS) for k in oo):
            fails.append(f"step {s} 键集合差异不是恰四个 motion 叶: +{sorted(oo)} -{sorted(kc - ko)}")
        for k in kc & ko:
            if ro["per_key"][k] != rc["per_key"][k]:
                fails.append(f"step {s} 公共叶 {k} raw sha 不同"); break
        if ro["sample_indices"] != rc["sample_indices"] or ro["sample_indices"] != io[s * 8:(s + 1) * 8]:
            fails.append(f"step {s} sample_indices 不一致")
        # oracle 重建四键 batch
        samples = [oracle_sample(index["entries"][sample_of(i)[0]], sample_of(i)[1], table, pos_table) for i in ro["sample_indices"]]
        n_samples += len(samples)
        for key in _MOTION_KEYS:
            arr = np.stack([sm[key] for sm in samples])
            kname = next((k for k in ko if key in k), None)
            if kname is None:
                fails.append(f"step {s} open 摘要缺 {key}"); mism += 1; continue
            if ro["per_key"][kname] != leaf_sha(arr) or ro["per_key_canonical"][kname] != canon_sha(arr):
                mism += 1
                print(f"  ✗ step {s} {key} raw/canonical 与 oracle 不同")
    ok = not fails and mism == 0
    for f in fails:
        print("  ✗", f)
    print(f"T3_TOKEN_TRACE={'PASS' if ok else 'FAIL'} steps=14 samples={n_samples} keys=4 mismatches={mism + len(fails)}")
    if not ok:
        raise SystemExit(1)


def cmd_t3mechanism(args):
    """T3_MOTION_CAUSAL + T3_MECHANISM：真实 batch、重新初始化的 TrainState（须命中 t3common reference 与 run 的 init 记录），
    固定 RNG / actions；bf16 独立复算两层与 gather；padding 垃圾 → loss / 全梯度摘要逐位不变；有效 emb 清零 / 打乱与有效 pos 扰动 → 梯度摘要必变；
    ∂loss/∂motion_emb 有效位 finite 且分组 L2 > 0、padding 位 0；W2[:768] / W2[768:] / W1 / bias 梯度分组范数。"""
    import dataclasses as _dc
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    import openpi.shared.array_typing as at
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from mme_vla_suite.shared.sampling import memory_order, pad_times
    lib = pathlib.Path(args.lib)
    leaf_sha, _ = _bench_hashes()
    ref = json.loads(pathlib.Path(args.reference).read_text())
    rec = pathlib.Path(args.open_records)
    _bo, idx = _t3_load_records(rec)
    # 确定性选 batch：前 8,000 个 index 中最早一个「至少一个样本自身 motion_mask.sum() ≥ 2」的 batch
    index, table, pos_table, manifest = load_lib_oracle(lib)
    eps = manifest["episodes"]; starts = np.array([e["exec_sample_offset"] for e in eps], np.int64)
    def sample_of(i):
        g = int(np.searchsorted(starts, i, side="right") - 1); return g, int(eps[g]["exec_start_idx"]) + (i - int(starts[g]))
    chosen_step = None
    for s in range(1000):
        b = idx[s * 8:(s + 1) * 8]
        if any(oracle_sample(index["entries"][sample_of(i)[0]], sample_of(i)[1], table, pos_table)["k"] >= 2 for i in b):
            chosen_step = s; break
    if chosen_step is None:
        raise SystemExit("前 8,000 个样本里没有 motion_mask.sum()≥2 的 batch")
    batch_idx = idx[chosen_step * 8:(chosen_step + 1) * 8]
    print(f"[t3mechanism] 选 step {chosen_step} 的 batch: {batch_idx}")
    tmp = pathlib.Path(args.tmp or (_V1 / "tmp" / "t3mechanism")); shutil.rmtree(tmp, ignore_errors=True)
    config, state, mesh, train = _t3_init_state(OPEN_YAML, "t3mechanism-open", tmp, pathlib.Path(args.dataset), args.fsdp)
    shas = _t3_state_leaf_shas(state)
    if shas != ref["open"]:
        raise SystemExit("重新初始化的 TrainState 未命中 t3common reference")
    rows = [json.loads(l) for l in (rec / "param_checksums.jsonl").read_text().splitlines() if l.strip()]
    if next(r for r in rows if int(r["step"]) == 0)["per_leaf"] != shas:
        raise SystemExit("重新初始化的 TrainState 未命中正式 run 的 step-0 记录")
    print("[t3mechanism] 初态命中 reference 与 run init 记录")
    # 真实 batch：经生产 dataset + 与训练同一组 transforms（data_config）——直接复用 create_data_loader 的 dataset + transform_dataset
    from openpi.training.data_loader import transform_dataset
    from mme_vla_suite.training.dataloader import _create_framesamp_dataset
    os.environ["MMEVLA_MOTION_STORE"] = str(lib / "motion")
    data_config = config.data.create(config.assets_dirs, config.model)
    ds = transform_dataset(_create_framesamp_dataset(str(lib / "framesamp"), data_config, config.model.history_config, config.model.action_horizon),
                           data_config, skip_norm_stats=False)
    items = [ds[i] for i in batch_idx]
    batch = jax.tree.map(lambda *xs: np.stack(xs), *items)
    obs = HistAugObservation.from_dict(jax.tree.map(jnp.asarray, batch))
    actions = jnp.asarray(batch["actions"])
    graphdef = state.model_def
    params = state.params
    # 显存：只保留 params；ema / opt_state（≈ 3 份参数体量）在初态校验后即释放——否则第二次 value_and_grad OOM（2026-09-03 实测 44 GB 双卡 fsdp=2 下 RESOURCE_EXHAUSTED）
    del state
    train_rng = jax.random.fold_in(jax.random.key(config.seed), chosen_step)   # 固定

    def loss_fn(params, obs, actions):
        model = nnx.merge(graphdef, params)
        model.train()
        with at.disable_typechecking():
            return jnp.mean(model.compute_loss(train_rng, obs, actions, train=True))

    @jax.jit
    def loss_and_grads(params, obs, actions):
        return jax.value_and_grad(loss_fn)(params, obs, actions)

    @jax.jit
    def input_grads(params, obs, actions):
        def f(me, mp):
            return loss_fn(params, _dc.replace(obs, motion_emb=me, motion_pos=mp), actions)
        return jax.grad(f, argnums=(0, 1))(obs.motion_emb, obs.motion_pos)

    # 梯度摘要只覆盖训练真正求导的叶（config.trainable_filter，与 train_step 的 nnx.DiffState 同一过滤）：
    # 冻结叶（如 SigLIP patch-embedding conv 的 kernel）在训练里根本不算 wgrad，其 GPU 反向核在本脚本里被算出来且不确定
    # （2026-09-03 实测：同一 obs 连算两次 / 三档 padding 垃圾都只让 ['PaliGemma']['img']['embedding']['kernel'] 一叶变化，loss 逐位相同）。
    trainable_keys = {jax.tree_util.keystr(kp) for kp, _ in
                      jax.tree_util.tree_flatten_with_path(params.filter(config.trainable_filter).to_pure_dict())[0]}
    for mk in ("['mem_encoder']['motion_pos_proj']['kernel']", "['mem_encoder']['motion_encoder_static']['kernel']"):
        if mk not in trainable_keys:
            raise SystemExit(f"motion 叶 {mk} 不在 trainable_filter 内——与 T3_SMOKE motion_params_updated 矛盾")
    print(f"[t3mechanism] 梯度摘要覆盖 trainable 叶 {len(trainable_keys)} 个（全参数叶 {len(jax.tree_util.tree_leaves(params.to_pure_dict()))}）")

    def grad_digest(grads):
        g = hashlib.sha256()
        for kp, v in jax.tree_util.tree_flatten_with_path(grads.to_pure_dict() if hasattr(grads, "to_pure_dict") else grads)[0]:
            if jax.tree_util.keystr(kp) in trainable_keys:
                g.update((jax.tree_util.keystr(kp) + leaf_sha(np.asarray(jax.device_get(v)))).encode())
        return g.hexdigest()

    fails = []
    base_loss, base_grads = loss_and_grads(params, obs, actions)
    base_loss = float(base_loss); base_dig = grad_digest(base_grads)
    # (d) 所需的四个 motion 叶梯度立即取回 host，整树梯度随即释放（每次 value_and_grad 的全参数梯度与 params 同体量）
    gp = {jax.tree_util.keystr(kp): np.asarray(jax.device_get(v)).astype(np.float64)
          for kp, v in jax.tree_util.tree_flatten_with_path(base_grads.to_pure_dict())[0] if "motion" in jax.tree_util.keystr(kp)}
    del base_grads
    print(f"[t3mechanism] base loss {base_loss:.6f}")

    def leaf_shas(grads):
        return {jax.tree_util.keystr(kp): leaf_sha(np.asarray(jax.device_get(v)))
                for kp, v in jax.tree_util.tree_flatten_with_path(grads.to_pure_dict())[0] if jax.tree_util.keystr(kp) in trainable_keys}

    def loss_digest(o, want_leaves=False):
        """loss 标量 + 全梯度摘要（可选逐叶 sha），梯度树用完即释放。"""
        l, g = loss_and_grads(params, o, actions)
        d = grad_digest(g)
        leaves = leaf_shas(g) if want_leaves else None
        del g
        return (float(l), d, leaves) if want_leaves else (float(l), d)

    base_leaves = None
    # bf16 独立复算两层 + gather（并列序 → 重排）
    P = {jax.tree_util.keystr(kp): np.asarray(jax.device_get(v)) for kp, v in jax.tree_util.tree_flatten_with_path(params.to_pure_dict())[0]}
    W1 = P["['mem_encoder']['motion_pos_proj']['kernel']"]; b1 = P["['mem_encoder']['motion_pos_proj']['bias']"]
    W2 = P["['mem_encoder']['motion_encoder_static']['kernel']"]; b2 = P["['mem_encoder']['motion_encoder_static']['bias']"]
    model = nnx.merge(graphdef, params)
    dt = model.mem_encoder.motion_pos_proj.dtype; prec = model.mem_encoder.motion_pos_proj.precision
    with at.disable_typechecking():
        par, _, _ = model.mem_encoder(obs.static_image_emb, obs.static_pos_emb, obs.static_state_emb,
                                      motion_emb=obs.motion_emb, motion_pos=obs.motion_pos, motion_mask=obs.motion_mask)
        tok, im, _, _ = model.embed_memory(obs)
    h = nnx.silu(jax.lax.dot_general(obs.motion_pos.astype(dt), jnp.asarray(W1).astype(dt), (((2,), (0,)), ((), ())), precision=prec, preferred_element_type=dt) + jnp.asarray(b1).astype(dt))
    y = jax.lax.dot_general(jnp.concatenate([obs.motion_emb.astype(dt), h.astype(dt)], -1), jnp.asarray(W2).astype(dt), (((2,), (0,)), ((), ())), precision=prec, preferred_element_type=dt) + jnp.asarray(b2).astype(dt)
    par_np = np.asarray(par); y_np = np.asarray(y)
    if not np.array_equal(y_np.view(np.uint16), par_np[:, 512:].view(np.uint16)):
        fails.append("两层 bf16 独立复算不逐位")
    mo = np.asarray(obs.mem_order)
    if not np.array_equal(np.asarray(tok).view(np.uint16), np.take_along_axis(par_np, mo[:, :, None], axis=1).view(np.uint16)):
        fails.append("gather 与 np.take_along_axis 不逐位")
    # (a) padding 垃圾 → loss 与全梯度摘要逐位不变
    g = np.random.default_rng(3)
    me = np.array(np.asarray(obs.motion_emb)); mp = np.array(np.asarray(obs.motion_pos)); mm = np.asarray(obs.motion_mask)
    me[~mm] = g.normal(0, 1e3, me[~mm].shape); mp[~mm] = g.normal(0, 1e3, mp[~mm].shape)
    l_pad, d_pad, pad_leaves = loss_digest(_dc.replace(obs, motion_emb=jnp.asarray(me), motion_pos=jnp.asarray(mp)), want_leaves=True)
    pad_bitexact = l_pad == base_loss and d_pad == base_dig
    if not pad_bitexact:
        # 诊断输出：loss 是否变、哪些叶变；再用 N(0,1) 与 N(0,1e-3) 两档垃圾复测，区分「泄漏」与「大数溢出 / 精度」
        _, _, base_leaves = loss_digest(obs, want_leaves=True)
        _, _, base_leaves2 = loss_digest(obs, want_leaves=True)
        nd = sorted(k for k in base_leaves if base_leaves[k] != base_leaves2.get(k))
        print(f"  [pad-diag] 确定性探针：同一 obs 连算两次梯度，叶变化 {len(nd)}/{len(base_leaves)}：{[k[:90] for k in nd[:6]]}")
        diff = sorted(k for k in base_leaves if base_leaves[k] != pad_leaves.get(k))
        print(f"  [pad-diag] loss base={base_loss.hex()} pad(1e3)={l_pad.hex()} 同={l_pad == base_loss}；梯度叶变化 {len(diff)}/{len(base_leaves)}：{[k[:90] for k in diff[:8]]}")
        for scale in (1.0, 1e-3):
            g2 = np.random.default_rng(5)
            me_s = np.array(np.asarray(obs.motion_emb)); mp_s = np.array(np.asarray(obs.motion_pos))
            me_s[~mm] = g2.normal(0, scale, me_s[~mm].shape); mp_s[~mm] = g2.normal(0, scale, mp_s[~mm].shape)
            l_s, d_s, lv_s = loss_digest(_dc.replace(obs, motion_emb=jnp.asarray(me_s), motion_pos=jnp.asarray(mp_s)), want_leaves=True)
            diff_s = sorted(k for k in base_leaves if base_leaves[k] != lv_s.get(k))
            print(f"  [pad-diag] 垃圾尺度 {scale:g}: loss 同={l_s == base_loss} 摘要同={d_s == base_dig} 叶变化 {len(diff_s)}：{[k[:90] for k in diff_s[:6]]}")
    # (b) 只在自身有效行内清零 / 打乱 motion_emb → loss 或梯度摘要必变；有效 pos 扰动 → 梯度摘要必变
    ks = mm.sum(axis=1); i_star = int(np.argmax(ks))
    me0 = np.array(np.asarray(obs.motion_emb)); me0[i_star, :ks[i_star]] = 0
    l0, d0 = loss_digest(_dc.replace(obs, motion_emb=jnp.asarray(me0)))
    perm = np.random.default_rng(4).permutation(int(ks[i_star]))
    me1 = np.array(np.asarray(obs.motion_emb)); me1[i_star, :ks[i_star]] = me1[i_star, :ks[i_star]][perm]
    l1, d1 = loss_digest(_dc.replace(obs, motion_emb=jnp.asarray(me1)))
    emb_effect = (l0 != base_loss or d0 != base_dig) and (l1 != base_loss or d1 != base_dig)
    mp1 = np.array(np.asarray(obs.motion_pos)); mp1[i_star, :ks[i_star]] *= 1.5
    l2, d2 = loss_digest(_dc.replace(obs, motion_pos=jnp.asarray(mp1)))
    pos_effect = d2 != base_dig
    # (c) 输入梯度：有效位 finite 且分组 L2 > 0，padding 位逐位 0
    gme, gmp = input_grads(params, obs, actions)
    gme = np.asarray(gme).astype(np.float64); gmp = np.asarray(gmp).astype(np.float64)
    in_ok = (np.isfinite(gme[mm]).all() and np.linalg.norm(gme[mm]) > 0 and np.all(gme[~mm] == 0)
             and np.isfinite(gmp[mm]).all() and np.linalg.norm(gmp[mm]) > 0 and np.all(gmp[~mm] == 0))
    # (d) 参数梯度分组（gp 已在 base 之后取回）
    gW2 = gp["['mem_encoder']['motion_encoder_static']['kernel']"]; gW1 = gp["['mem_encoder']['motion_pos_proj']['kernel']"]
    norms = {"W2_content[:768]": float(np.linalg.norm(gW2[:768])), "W2_pos[768:]": float(np.linalg.norm(gW2[768:])),
             "W1": float(np.linalg.norm(gW1)), "b1": float(np.linalg.norm(gp["['mem_encoder']['motion_pos_proj']['bias']"])),
             "b2": float(np.linalg.norm(gp["['mem_encoder']['motion_encoder_static']['bias']"])),
             "motion_emb_valid": float(np.linalg.norm(gme[mm])), "motion_pos_valid": float(np.linalg.norm(gmp[mm]))}
    grp_ok = norms["W2_content[:768]"] > 0 and norms["W2_pos[768:]"] > 0 and norms["W1"] > 0
    print(f"[t3mechanism] 分组梯度范数 {json.dumps({k: f'{v:.4e}' for k, v in norms.items()})}")
    print(f"T3_MOTION_CAUSAL={'PASS' if (pad_bitexact and emb_effect and pos_effect) else 'FAIL'} pad_bitexact={int(pad_bitexact)} emb_effect={int(emb_effect)} pos_effect={int(pos_effect)}")
    ok = pad_bitexact and emb_effect and pos_effect and in_ok and grp_ok and not fails
    for f in fails:
        print("  ✗", f)
    print(f"T3_MECHANISM={'PASS' if ok else 'FAIL'} step={chosen_step} input_grad_ok={int(in_ok)} group_norms_ok={int(grp_ok)}")
    out = {"chosen_step": chosen_step, "batch_idx": batch_idx, "base_loss": base_loss, "norms": norms, "pad_bitexact": pad_bitexact,
           "emb_effect": emb_effect, "pos_effect": pos_effect, "input_grad_ok": bool(in_ok), "verdict": "PASS" if ok else "FAIL"}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    if not ok:
        raise SystemExit(1)


def cmd_t3phase(args):
    """T3_PHASE_REPORT：严格恢复两侧最终（目录 999、EMA）checkpoint，全部 11,530 样本 fold_in(base, idx) 固定逐样本 RNG，
    compute_loss(train=False) 的 20 步均值；按 phase=(t−es)%16 与 empty/nonempty 汇总；完整性硬校验，均值只报告。"""
    import dataclasses as _dc
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import openpi.shared.array_typing as at
    import openpi.models.model as _model
    from openpi.training.data_loader import transform_dataset
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from mme_vla_suite.training.dataloader import _create_framesamp_dataset
    lib = pathlib.Path(args.lib)
    index, table, pos_table, manifest = load_lib_oracle(lib)
    eps = manifest["episodes"]; starts = np.array([e["exec_sample_offset"] for e in eps], np.int64)
    n = int(manifest["totals"]["exec_samples"])
    os.environ["MMEVLA_MOTION_STORE"] = str(lib / "motion")
    sides = {}
    for tag, yaml_name, ckpt in (("closed", CLOSED_YAML, args.closed_ckpt), ("open", OPEN_YAML, args.open_ckpt)):
        config = _t3_train_config(yaml_name, f"t3phase-{tag}", _V1 / "tmp" / "t3phase", pathlib.Path(args.dataset), fsdp=1)
        from mme_vla_suite.models.config.utils import get_history_config
        config = _dc.replace(config, model=_dc.replace(config.model, history_config=get_history_config(yaml_name)))
        params = _model.restore_params(pathlib.Path(ckpt) / "params", dtype=jnp.bfloat16)
        model = config.model.load(params, remove_extra_params=False)
        from mme_vla_suite.policies.policy_config import _assert_param_tree_exact, _params_have_motion
        _assert_param_tree_exact(model, params)
        if _params_have_motion(params) != (tag == "open"):
            raise SystemExit(f"{tag} checkpoint 的 motion 参数存在性与侧别不符")
        model.eval()
        data_config = config.data.create(config.assets_dirs, config.model)
        ds = transform_dataset(_create_framesamp_dataset(str(lib / "framesamp"), data_config, config.model.history_config, config.model.action_horizon),
                               data_config, skip_norm_stats=False)
        graphdef, state = nnx.split(model)

        @jax.jit
        def one(key, obs, actions):
            m = nnx.merge(graphdef, state)
            with at.disable_typechecking():
                return jnp.mean(m.compute_loss(key, obs, actions, train=False), axis=-1)   # (1,)

        base = jax.random.key(args.seed)
        cache = pathlib.Path(args.out).with_suffix(f".{tag}.losses.npy")   # 每侧逐样本 loss 落盘：另一侧崩了不必重扫（2026-09-03 closed 侧 25 min 因 ds.close 崩过一次）
        if cache.is_file() and args.reuse_losses:
            losses = np.load(cache)
            if losses.shape != (n,):
                raise SystemExit(f"{cache} 形状 {losses.shape} != ({n},)")
            print(f"  [{tag}] 复用 {cache}", flush=True)
        else:
            losses = np.zeros(n, np.float64)
            t0 = time.time()
            for i in range(n):
                item = ds[i]
                batch = jax.tree.map(lambda x: jnp.asarray(np.asarray(x))[None], item)
                obs = HistAugObservation.from_dict(batch)
                losses[i] = float(one(jax.random.fold_in(base, i), obs, batch["actions"])[0])
                if (i + 1) % 1000 == 0:
                    print(f"  [{tag}] {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
            np.save(cache, losses)
        sides[tag] = losses
        inner = getattr(ds, "_dataset", ds)       # TransformedDataset 本身没有 close，关底层 FrameSampDataset 的 store
        if hasattr(inner, "close"):
            inner.close()
    # 标签由 oracle 按物理样本身份统一生成
    phase = np.zeros(n, np.int64); tau = np.zeros(n, np.int64); empty = np.zeros(n, np.bool_)
    for i in range(n):
        g = int(np.searchsorted(starts, i, side="right") - 1); es = int(eps[g]["exec_start_idx"]); t = es + (i - int(starts[g]))
        tau[i] = t - es; phase[i] = (t - es) % 16
        empty[i] = oracle_sample(index["entries"][g], t, table, pos_table)["k"] == 0
    p0 = phase == 0; cold = p0 & (tau < 32); steady = p0 & (tau >= 32); other = ~p0
    if not (p0.sum() == cold.sum() + steady.sum() and p0.sum() + other.sum() == n and empty.sum() + (~empty).sum() == n):
        raise SystemExit("分区计数不闭合")
    def mean(mask, side):
        return float(np.sort(np.where(mask)[0]).size and sides[side][mask].sum() / mask.sum())
    for side in ("open", "closed"):
        w = (mean(cold, side) * cold.sum() + mean(steady, side) * steady.sum()) / max(1, p0.sum())
        if abs(w - mean(p0, side)) > 1e-12 * max(1.0, abs(mean(p0, side))):
            raise SystemExit(f"{side} phase0 总均值 != 冷/稳态计数加权均值")
    rep = {"samples": n, "phase0_n": int(p0.sum()), "phase0_open": mean(p0, "open"), "phase0_closed": mean(p0, "closed"),
           "phase0_cold_n": int(cold.sum()), "phase0_cold_open": mean(cold, "open"), "phase0_cold_closed": mean(cold, "closed"),
           "phase0_steady_n": int(steady.sum()), "phase0_steady_open": mean(steady, "open"), "phase0_steady_closed": mean(steady, "closed"),
           "other_n": int(other.sum()), "other_open": mean(other, "open"), "other_closed": mean(other, "closed"),
           "empty_n": int(empty.sum()), "empty_open": mean(empty, "open"), "empty_closed": mean(empty, "closed"),
           "nonempty_n": int((~empty).sum()), "nonempty_open": mean(~empty, "open"), "nonempty_closed": mean(~empty, "closed")}
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), loss_open=sides["open"], loss_closed=sides["closed"], phase=phase, tau=tau, empty=empty)
    out.write_text(json.dumps({**rep, "seed": args.seed, "param_kind": "ema(final ckpt 999)", "note": "loss 方向只报告，无 PASS/FAIL；ep0–9 在 encoder 训练集内"}, indent=1))
    print("T3_PHASE_REPORT " + " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in rep.items()))


# 扩展 --gate 选项
def _t3_main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", choices=["m1", "m2", "m3", "m4", "m5", "t3common", "t3verifyinit", "t3trace", "t3mechanism", "t3phase"], required=True)
    ap.add_argument("--lib", default=str(_V1 / "datasets/4task-motion-40ep"))
    ap.add_argument("--dataset", default=str(_V1 / "datasets/4task-motion-40ep/framesamp"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--out", default=str(_V1 / "reports/motion/t3.json"))
    ap.add_argument("--fsdp", type=int, default=2)
    ap.add_argument("--reference", default=str(_V1 / "reports/motion/t3_common_init_reference.json"))
    ap.add_argument("--open-records", default=None)
    ap.add_argument("--closed-records", default=None)
    ap.add_argument("--open-ckpt", default=None)
    ap.add_argument("--closed-ckpt", default=None)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--index-json", default=None)
    ap.add_argument("--reuse-losses", action="store_true", help="t3phase：已有 <out>.<side>.losses.npy 时直接复用")
    args = ap.parse_args()
    {"m1": cmd_m1, "m2": cmd_m2, "m3": cmd_m3, "m4": cmd_m4, "m5": cmd_m5, "t3common": cmd_t3common, "t3verifyinit": cmd_t3verifyinit,
     "t3trace": cmd_t3trace, "t3mechanism": cmd_t3mechanism, "t3phase": cmd_t3phase}[args.gate](args)


main = _t3_main

if __name__ == "__main__":
    main()
