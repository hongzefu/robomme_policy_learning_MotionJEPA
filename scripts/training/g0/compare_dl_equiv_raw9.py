#!/usr/bin/env python3
"""wan_raw9 训练侧数据等价闸 / v11 回归 / 每 ckpt 激活统计（计划 1112-4task-modul-wanraw9-s32-50k-plan.html
§3.4、第二部分 1.2 本文件行、闸 G3；owner S2）。

三种模式：

``--mode raw9``（闸 G3 ``DL_EQUIV``）
    对同一批确定性抽样 idx，比较「逐 token 参考实现」与 ``FrameSampDataset`` 的 L0 预分配实现给模型的四键
    ``motion_emb / motion_pos / motion_mask / mem_order`` 是否逐字节相等（dtype、shape、``tobytes()`` 三项）。
    参考实现不经 motion 表：直接从解包后的源 latent ``<latents>/<段键>.bin``（stride 16，第 m′ 个 stride 32 窗 =
    源第 2m′ 窗，每窗 (9,16,32,32) float32 原字节）读窗，逐 token 手写 C→H→W 展平、τ_j = min(f+4j, last_real)
    （demo 窗 last_real = es−1、exec 窗 last_real = f+32）、pos = 帧路 pos 表 τ 行第 0 行前 pos_dim 维、mask、
    右填充，以及按 (全域时刻×2+类型, 拼接位置) 的 Python 排序手写 mem_order（不调 ``shared/sampling.py``）。
    判定行 ``DL_EQUIV=PASS samples=<n> mismatch=0``；S5 的 L1/L2/L3 每级复用本模式。

    - 真实库：``--history-yaml <wanraw9s32.yaml> --framesamp <lib>/framesamp-8x8 --latents <lib>/wan-latents``；
    - 合成夹具：``--fixture <空目录>``（单测与离线自检用；``--tamper`` 先把 motion 表改 1 字节，预期 FAIL）。

``--mode regress-v11``（闸 G2 训练侧 ``V11_REGRESS``）
    在 v11 yaml（wantp768）下对 256 个确定性样本取八键（static_image_emb / static_pos_emb / static_state_emb /
    static_mask + motion 四键）的 dtype、shape、sha256 摘要，并取 ``HistoryPi0`` 参数树（叶路径、形状、dtype）摘要，
    写 ``--dump``；给 ``--ref-dump``（由 c55a6bc 只读 worktree 侧同脚本 dump 得到）时逐项比较，
    判定行 ``V11_REGRESS=PASS samples=<n> mismatch=0 leaves_equal=1``。本模式顶层不 import wan_raw9，可在 c55a6bc 树上跑。

``--mode act-stats``（第一部分 §5 运行中记录项）
    对 ``--ckpt <run>/<step>`` 只部分恢复 ``motion_pos_proj`` 与 ``motion_encoder_static`` 四个叶，用固定 seed 的
    64 个样本在 numpy float32 里前向到 ``motion_encoder_static`` 输出（只统计 motion_mask 为真的 token），
    打印 ``ACT_STATS step=<step> mean=<x> var=<y> samples=64 tokens=<n>``。由主会话在 E 阶段每份 ckpt 落盘后跑
    （``JAX_PLATFORMS=cpu``，≤5 min），不进 runner。

用法（主检出，环境取法见计划第二部分「二 子代理分配表」）：
  JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/g0/compare_dl_equiv_raw9.py --mode raw9 \\
      --history-yaml src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml \\
      --framesamp v1-store/datasets/4task-v2-1600ep-604f16da/framesamp-8x8 \\
      --latents v1-store/datasets/4task-v2-1600ep-604f16da/wan-latents --samples 256
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]

MOTION_KEYS = ("motion_emb", "motion_pos", "motion_mask", "mem_order")
KEYS8 = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask") + MOTION_KEYS
SENTINEL = int(np.iinfo(np.int32).max)       # 与 shared/sampling.py::MEM_ORDER_SENTINEL 同值（参考实现独立写死）

# 参考实现自带的几何常量（不 import wan_raw9 / motion_store 的常量，独立写死，与计划 §2 / §3.1 一致）
REF_STRIDE = 32
REF_SRC_STRIDE = 16
REF_LATENT_SHAPE = (9, 16, 32, 32)
REF_LATENT_BYTES = 9 * 16 * 32 * 32 * 4      # 589,824
REF_TOKEN_DIM = 16 * 32 * 32                 # 16,384
REF_WINDOW_LAST = 32                         # 33 帧窗：[f, f+32]


# ── 比较小件 ─────────────────────────────────────────────────────────────────


def _digest(a) -> list:
    a = np.asarray(a)
    return [str(a.dtype), list(a.shape), hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()]


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def _sample_idxs(n_total: int, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    n = min(int(n), int(n_total))
    return sorted(int(i) for i in rng.choice(int(n_total), size=n, replace=False))


# ── 逐 token 参考实现（只用 numpy 与 Python 循环；不调 dataset / sampling / wan_raw9）─────────────────


def ref_windows(entry, step: int) -> list[tuple[str, int, int]]:
    """stride 32 合法窗（手写）：返回 [(段, m′, 全域起点 f)]，按 f 升序（demo 全在 exec 前）。"""
    es = int(entry.exec_start_idx)
    out = []
    for mp in range(int(entry.demo.num_grid)):
        s = REF_STRIDE * mp
        if s + (int(entry.spec.demo_min_real) - 1) <= es - 1:
            out.append(("demo", mp, s))
    for mp in range(int(entry.exec.num_grid)):
        u = REF_STRIDE * mp
        if u + REF_WINDOW_LAST <= int(step) - es:
            out.append(("exec", mp, es + u))
    return out


def ref_mem_order(frame_times: list[int], tokens_per_frame: int, motion_times: list[int]) -> np.ndarray:
    """手写交错次序：位置 i 的键 = (时刻×2+类型, i)，Python sorted 后取 i（等价稳定排序）。"""
    keys = []
    for t in frame_times:
        for _ in range(tokens_per_frame):
            keys.append(t * 2 + 0)
    for t in motion_times:
        keys.append(t * 2 + 1)
    order = [i for _, i in sorted((k, i) for i, k in enumerate(keys))]
    return np.asarray(order, dtype=np.int32)


def reference_motion(entry, step: int, frames: list[int], *, read_latent, pos_lookup, pos_dim: int,
                     n_tokens: int, max_frames: int, tokens_per_frame: int) -> dict:
    """逐 token 构造四键。read_latent(段, entry, 源窗号) → (9,16,32,32) f32；pos_lookup(t) → (768,) f32。"""
    es = int(entry.exec_start_idx)
    emb = np.zeros((n_tokens, REF_TOKEN_DIM), dtype=np.float32)
    pos = np.zeros((n_tokens, pos_dim), dtype=np.float32)
    mask = np.zeros((n_tokens,), dtype=np.bool_)
    mtimes: list[int] = []
    i = 0
    for seg, mp, f in ref_windows(entry, step):
        z = read_latent(seg, entry, 2 * mp)
        last = es - 1 if seg == "demo" else f + REF_WINDOW_LAST
        for j in range(REF_LATENT_SHAPE[0]):
            if i >= n_tokens:
                raise ValueError(f"参考实现：token 数超过 T={n_tokens}（零截断契约被破坏）")
            flat = np.empty((REF_TOKEN_DIM,), dtype=np.float32)
            c_n, h_n, w_n = REF_LATENT_SHAPE[1:]
            for c in range(c_n):                    # C→H→W：index = 1024c + 32h + w
                flat[c * h_n * w_n:(c + 1) * h_n * w_n] = z[j, c].reshape(-1)
            emb[i] = flat
            tau = min(f + 4 * j, last)
            pos[i] = pos_lookup(tau)[:pos_dim]
            mask[i] = True
            mtimes.append(tau)
            i += 1
    mtimes_p = mtimes + [SENTINEL] * (n_tokens - len(mtimes))
    ftimes_p = list(frames) + [SENTINEL] * (max_frames - len(frames))
    return {"motion_emb": emb, "motion_pos": pos, "motion_mask": mask,
            "mem_order": ref_mem_order(ftimes_p, tokens_per_frame, mtimes_p)}


def compare_motion(got: dict, ref: dict) -> list[str]:
    return [k for k in MOTION_KEYS if not _same(got[k], ref[k])]


# ── 合成夹具（单测与 --fixture 共用）────────────────────────────────────────────


class FixtureMotionStore:
    """夹具 motion 表：只读 memmap，接口同 S1 的 ``MotionStore.rows_into``。"""

    def __init__(self, table_path: pathlib.Path, num_rows: int):
        self._t = np.memmap(table_path, dtype="<f4", mode="r", shape=(num_rows,) + REF_LATENT_SHAPE)

    def rows_into(self, rows, out) -> None:
        r = np.asarray(rows, dtype=np.int64)
        if out.shape != (len(r),) + REF_LATENT_SHAPE:
            raise ValueError(f"rows_into out 形状 {out.shape}")
        out[...] = self._t[r]


class FixturePosStore:
    def __init__(self, pos: np.ndarray):
        self._pos = pos

    def pos_rows(self, frames) -> np.ndarray:
        return self._pos[np.asarray(frames, dtype=np.int64)]


FIXTURE_EPISODES = (
    # (num_timesteps, exec_start_idx)：覆盖 demo 尾补窗（es−s ∈ [17,32]）、无 demo 窗（es<17）、多 exec 窗
    (180, 70),     # demo 窗 s=0,32（s=32：es−s=38 全真）；s=64 不合法(es-16=54)；exec 段长 110
    (150, 50),     # demo s=0（全真）、s=32（es−s=18 → 尾补，τ 截断到 49）
    (90, 10),      # demo 无合法窗；exec 段长 80
)
FIXTURE_TPF = 64
FIXTURE_MAX_FRAMES = 8
FIXTURE_POS_DIM = 256


def build_fixture(root: pathlib.Path, seed: int = 0) -> SimpleNamespace:
    """在 root 下写合成源 latent（stride 16）与 stride 32 表（偶数窗原字节），返回夹具描述。"""
    from mme_vla_suite.datastore import motion_store as ms

    root.mkdir(parents=True, exist_ok=True)
    lat_root = root / "wan-latents"
    lat_root.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    spec = SimpleNamespace(demo_min_real=17, exec_min_real=33, demo_tail_pad="repeat_last",
                           grid_stride=REF_STRIDE, row_shape=REF_LATENT_SHAPE)
    entries, table_rows, cursor = [], [], 0
    for g, (nt, es) in enumerate(FIXTURE_EPISODES):
        h5 = f"record_dataset_Fix{g}.h5"
        segs = {}
        for seg, seg_len, min_real in (("demo", es, 17), ("exec", nt - es, 33)):
            n_chunks = max(0, seg_len - (min_real - 1))
            n16 = len(range(0, n_chunks, REF_SRC_STRIDE))
            n32 = len(range(0, n_chunks, REF_STRIDE))
            lat = rng.standard_normal((n16,) + REF_LATENT_SHAPE).astype("<f4")
            (lat_root / f"{ms.segment_key(h5, 0, seg)}.bin").write_bytes(lat.tobytes())
            base = cursor if n32 else None
            for mp in range(n32):
                table_rows.append(lat[2 * mp])
            cursor += n32
            segs[seg] = ms.SegmentInfo(row_base=base, num_grid=n32, num_chunks=n_chunks, seg_len=seg_len)
        entries.append(SimpleNamespace(g=g, h5_file=h5, raw_ep_idx=0, num_timesteps=nt, exec_start_idx=es,
                                       demo=segs["demo"], exec=segs["exec"], spec=spec))
    table_path = root / "motion_token.f32.bin"
    table_path.write_bytes(np.stack(table_rows).astype("<f4").tobytes())
    max_t = max(nt for nt, _ in FIXTURE_EPISODES)
    pos = rng.standard_normal((max_t, FIXTURE_TPF, 768)).astype(np.float32)
    samples = [(g, t) for g, (nt, es) in enumerate(FIXTURE_EPISODES) for t in range(es, nt, 3)]
    return SimpleNamespace(root=root, latents=lat_root, table=table_path, num_rows=cursor, entries=entries,
                           pos=pos, samples=samples)


def fixture_dataset(fx, budget: int = 80):
    """夹具上的 L0 实现：裸 FrameSampDataset 实例只设 _fill_motion_raw9 用到的属性。"""
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    from mme_vla_suite.shared import wan_raw9 as wr

    ds = FrameSampDataset.__new__(FrameSampDataset)
    ds._motion_budget = int(budget)
    import omegaconf
    ds._motion_tokens = wr.motion_token_budget(omegaconf.OmegaConf.create({"budget": budget, "tokens_per_window": 9}))
    ds._motion_dim = REF_TOKEN_DIM
    ds._motion_pos_dim = FIXTURE_POS_DIM
    ds._max_frames = FIXTURE_MAX_FRAMES
    ds._tokens_per_frame = FIXTURE_TPF
    return ds


def _fixture_read_latent(lat_root: pathlib.Path):
    from mme_vla_suite.datastore import motion_store as ms

    def read(seg, entry, m16):
        p = lat_root / f"{ms.segment_key(entry.h5_file, entry.raw_ep_idx, seg)}.bin"
        return np.fromfile(p, dtype="<f4", count=REF_LATENT_BYTES // 4,
                           offset=int(m16) * REF_LATENT_BYTES).reshape(REF_LATENT_SHAPE)
    return read


def run_fixture_equiv(root: pathlib.Path, *, tamper: bool = False, budget: int = 80) -> tuple[int, int, str]:
    from mme_vla_suite.shared.sampling import even_sampling_indices

    fx = build_fixture(root)
    if tamper:                                    # 改表第 0 行第 101 字节（夹具第 0 号 episode 的 demo 窗 0 恒可见）
        b = bytearray(fx.table.read_bytes())
        b[101] ^= 0x01
        fx.table.write_bytes(bytes(b))
    ds = fixture_dataset(fx, budget)
    mstore = FixtureMotionStore(fx.table, fx.num_rows)
    pstore = FixturePosStore(fx.pos)
    read = _fixture_read_latent(fx.latents)
    mism, first = 0, ""
    for g, t in fx.samples:
        frames = even_sampling_indices(t, FIXTURE_MAX_FRAMES)
        got = dict(zip(MOTION_KEYS, ds._fill_motion_raw9(fx.entries[g], t, mstore, pstore,
                                                         np.asarray(frames, dtype=np.int64))))
        ref = reference_motion(fx.entries[g], t, frames, read_latent=read, pos_lookup=lambda tau: fx.pos[tau, 0],
                               pos_dim=FIXTURE_POS_DIM, n_tokens=ds._motion_tokens,
                               max_frames=FIXTURE_MAX_FRAMES, tokens_per_frame=FIXTURE_TPF)
        bad = compare_motion(got, ref)
        if bad:
            mism += 1
            first = first or f"g={g},t={t}:{','.join(bad)}"
    return len(fx.samples), mism, first


# ── 真实库 ───────────────────────────────────────────────────────────────────


def _build_dataset(args, history_yaml: pathlib.Path):
    import omegaconf
    from mme_vla_suite.training import config as _config
    from mme_vla_suite.training.dataloader import _create_framesamp_dataset

    train_config = _config.get_config(args.train_config)
    if args.assets_dir:
        train_config = dataclasses.replace(train_config, data=dataclasses.replace(
            train_config.data, assets=dataclasses.replace(train_config.data.assets, assets_dir=str(args.assets_dir))))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    hc = omegaconf.OmegaConf.load(history_yaml)
    if args.motion_root:
        hc.motion.store_path = str(pathlib.Path(args.motion_root).resolve())
    ds = _create_framesamp_dataset(str(args.framesamp), data_config, hc, int(train_config.model.action_horizon))
    return ds, hc, train_config


def run_real_equiv(args) -> tuple[int, int, str]:
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.shared.sampling import even_sampling_indices

    ds, hc, _ = _build_dataset(args, args.history_yaml)
    if not getattr(ds, "_motion_raw9", False):
        raise SystemExit("错误: --history-yaml 不是 wan_raw9 配置（dataset._motion_raw9 为假）")
    lat_root = pathlib.Path(args.latents)

    def read(seg, entry, m16):
        p = lat_root / f"{ms.segment_key(entry.h5_file, entry.raw_ep_idx, seg)}.bin"
        if p.stat().st_size < (int(m16) + 1) * REF_LATENT_BYTES:
            raise ValueError(f"源 latent {p} 不含第 {m16} 窗")
        return np.fromfile(p, dtype="<f4", count=REF_LATENT_BYTES // 4,
                           offset=int(m16) * REF_LATENT_BYTES).reshape(REF_LATENT_SHAPE)

    store = ds._ensure_store()
    n_tokens = int(hc.motion.budget) * int(hc.motion.tokens_per_window)
    mism, first = 0, ""
    idxs = _sample_idxs(len(ds), args.samples, args.seed)
    for idx in idxs:
        got = ds[idx]
        g, t = int(ds._epis_of[idx]), int(ds._step_of[idx])
        frames = even_sampling_indices(t, ds._max_frames)
        ref = reference_motion(ds._motion_entries[g], t, frames, read_latent=read,
                               pos_lookup=lambda tau: store.pos_rows(np.asarray([tau]))[0, 0],
                               pos_dim=int(hc.motion.pos_dim), n_tokens=n_tokens,
                               max_frames=ds._max_frames, tokens_per_frame=ds._tokens_per_frame)
        bad = compare_motion(got, ref)
        if bad:
            mism += 1
            first = first or f"idx={idx}:{','.join(bad)}"
    return len(idxs), mism, first


def param_tree_digest(train_config, hc) -> dict:
    import jax
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict

    mc = dataclasses.replace(train_config.model, history_config=hc, use_history=True)
    model = nnx.eval_shape(mc.create, jax.random.key(0))
    flat = flatten_dict(nnx.state(model, nnx.Param).to_pure_dict())
    lines = sorted("/".join(map(str, k)) + ":" + str(tuple(v.shape)) + ":" + str(v.dtype) for k, v in flat.items())
    return {"n_leaves": len(lines), "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(), "leaves": lines}


def run_regress_v11(args) -> int:
    ds, hc, train_config = _build_dataset(args, args.history_yaml)
    idxs = _sample_idxs(len(ds), args.samples, args.seed)
    dump = {"history_yaml": str(args.history_yaml), "seed": args.seed, "samples": []}
    for idx in idxs:
        d = ds[idx]
        dump["samples"].append({"idx": idx, "keys": {k: (None if d.get(k) is None else _digest(d[k])) for k in KEYS8}})
    dump["param_tree"] = param_tree_digest(train_config, hc)
    if args.dump:
        pathlib.Path(args.dump).write_text(json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[dump] {args.dump} samples={len(idxs)} n_leaves={dump['param_tree']['n_leaves']}")
    if not args.ref_dump:
        print("V11_REGRESS=DUMP_ONLY（未给 --ref-dump）")
        return 0
    ref = json.loads(pathlib.Path(args.ref_dump).read_text(encoding="utf-8"))
    ref_by = {s["idx"]: s["keys"] for s in ref["samples"]}
    mism = sum(1 for s in dump["samples"] if ref_by.get(s["idx"]) != s["keys"])
    missing = sum(1 for s in dump["samples"] if s["idx"] not in ref_by)
    leaves_equal = int(ref["param_tree"]["sha256"] == dump["param_tree"]["sha256"]
                       and ref["param_tree"]["n_leaves"] == dump["param_tree"]["n_leaves"])
    ok = mism == 0 and missing == 0 and leaves_equal == 1 and len(ref["samples"]) == len(dump["samples"])
    print(f"V11_REGRESS={'PASS' if ok else 'FAIL'} samples={len(idxs)} mismatch={mism} leaves_equal={leaves_equal}")
    return 0 if ok else 1


# ── act-stats ────────────────────────────────────────────────────────────────


def load_motion_params(ckpt: pathlib.Path) -> dict:
    """部分恢复 motion_pos_proj / motion_encoder_static 的 kernel 与 bias（numpy），不读其余参数。"""
    import jax
    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict, unflatten_dict

    path = (pathlib.Path(ckpt) / "params").resolve()
    with ocp.PyTreeCheckpointer() as c:
        meta = c.metadata(path)
        tree = meta.tree if hasattr(meta, "tree") else meta
        flat = flatten_dict(tree["params"])
        want = {k: v for k, v in flat.items()
                if any(n in map(str, k) for n in ("motion_pos_proj", "motion_encoder_static"))}
        item = {"params": unflatten_dict(want)}
        restore_args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item)
        got = c.restore(path, args=ocp.args.PyTreeRestore(item=item, restore_args=restore_args, transforms={}))
    out = {}
    for k, v in flatten_dict(got["params"]).items():
        ks = [str(x) for x in k]
        if ks[-1] == "value":
            ks = ks[:-1]
        mod = "motion_pos_proj" if "motion_pos_proj" in ks else "motion_encoder_static"
        if ks[-1] not in ("kernel", "bias") or (mod, ks[-1]) in out:
            raise ValueError(f"ckpt 里 motion 叶不唯一或不认识: {'/'.join(ks)}")
        out[(mod, ks[-1])] = np.asarray(v, dtype=np.float32)
    if len(out) != 4:
        raise ValueError(f"ckpt 缺 motion 叶，只找到 {sorted(out)}")
    return out


def act_stats(params: dict, samples) -> tuple[float, float, int]:
    wp, bp = params[("motion_pos_proj", "kernel")], params[("motion_pos_proj", "bias")]
    w, b = params[("motion_encoder_static", "kernel")], params[("motion_encoder_static", "bias")]
    s1 = s2 = 0.0
    n_el = n_tok = 0
    for emb, pos, mask in samples:
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            continue
        e = np.asarray(emb, dtype=np.float32)[m]
        hp = np.asarray(pos, dtype=np.float32)[m] @ wp + bp
        hp = hp * (1.0 / (1.0 + np.exp(-hp)))                         # silu
        y = np.concatenate([e, hp.astype(np.float32)], axis=-1) @ w + b
        y = y.astype(np.float64)
        s1 += float(y.sum())
        s2 += float((y * y).sum())
        n_el += y.size
        n_tok += int(m.sum())
    if n_el == 0:
        raise ValueError("64 个样本没有任何有效 motion token")
    mean = s1 / n_el
    return mean, s2 / n_el - mean * mean, n_tok


def run_act_stats(args) -> int:
    from mme_vla_suite.shared.sampling import even_sampling_indices

    ckpt = pathlib.Path(args.ckpt)
    params = load_motion_params(ckpt)
    n = 64
    if args.fixture:
        fx = build_fixture(pathlib.Path(args.fixture))
        ds = fixture_dataset(fx)
        ms_, ps_ = FixtureMotionStore(fx.table, fx.num_rows), FixturePosStore(fx.pos)
        pairs = [fx.samples[i] for i in _sample_idxs(len(fx.samples), n, args.seed)]

        def gen():
            for g, t in pairs:
                o = ds._fill_motion_raw9(fx.entries[g], t, ms_, ps_,
                                         np.asarray(even_sampling_indices(t, FIXTURE_MAX_FRAMES), np.int64))
                yield o[0], o[1], o[2]
        used = len(pairs)
    else:
        ds, _, _ = _build_dataset(args, args.history_yaml)
        idxs = _sample_idxs(len(ds), n, args.seed)

        def gen():
            for idx in idxs:
                d = ds[idx]
                yield d["motion_emb"], d["motion_pos"], d["motion_mask"]
        used = len(idxs)
    mean, var, n_tok = act_stats(params, gen())
    step = ckpt.name
    print(f"ACT_STATS step={step} mean={mean:.6e} var={var:.6e} samples={used} tokens={n_tok}")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("raw9", "regress-v11", "act-stats"))
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--history-yaml", type=pathlib.Path,
                    default=REPO / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml")
    ap.add_argument("--framesamp", type=pathlib.Path, help="framesamp-8x8 库根")
    ap.add_argument("--latents", type=pathlib.Path, help="解包后的源 latent 根 wan-latents/（raw9 模式参考实现）")
    ap.add_argument("--motion-root", type=pathlib.Path, help="覆盖 yaml motion.store_path（绝对化）")
    ap.add_argument("--train-config", default="mme_vla_suite_b128_80k")
    ap.add_argument("--assets-dir", type=pathlib.Path, help="覆盖 TrainConfig.data.assets.assets_dir")
    ap.add_argument("--fixture", type=pathlib.Path, help="合成夹具目录（raw9 / act-stats）")
    ap.add_argument("--tamper", action="store_true", help="夹具：先把 motion 表改 1 字节（预期 DL_EQUIV=FAIL）")
    ap.add_argument("--dump", type=pathlib.Path)
    ap.add_argument("--ref-dump", type=pathlib.Path)
    ap.add_argument("--ckpt", type=pathlib.Path, help="act-stats：<run>/<step> 目录（含 params/）")
    args = ap.parse_args(argv)

    import mme_vla_suite
    print(f"[env] mme_vla_suite={mme_vla_suite.__file__}")
    if args.mode == "raw9":
        if args.fixture:
            n, mism, first = run_fixture_equiv(args.fixture, tamper=args.tamper)
        else:
            if args.framesamp is None or args.latents is None:
                ap.error("--mode raw9 真实库需要 --framesamp 与 --latents（或用 --fixture）")
            n, mism, first = run_real_equiv(args)
        ok = mism == 0 and n > 0
        tail = "" if ok else f" first={first}"
        print(f"DL_EQUIV={'PASS' if ok else 'FAIL'} samples={n} mismatch={mism}{tail}")
        return 0 if ok else 1
    if args.mode == "regress-v11":
        return run_regress_v11(args)
    if args.ckpt is None:
        ap.error("--mode act-stats 需要 --ckpt")
    return run_act_stats(args)


if __name__ == "__main__":
    sys.exit(main())
