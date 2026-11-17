"""RouteStick 进度探针与注意力图的专家回放（计划 1117-routestick-progress-probe-attn-plan.html B1 第 3 行、B4 R2、B5、B6 第 4–5 步）。

按评测节奏喂帧（同 examples/robomme/eval.py：首包 = 示范段 [0, e0) 加执行段第 0 帧，exec_start_idx = e0；之后每包 16 帧，
每包后推理一次，推理点为 t = e0 + u，u = 0、16、32…），在 R1 给出的每个推理点上用 AttnAdapter（modulation）捕获
逐层隐状态与聚合注意力，K 个噪声样本（缺省 5）在本入口里取平均，每局写一个 npz。

bf16 的 policy 只用来逐帧生成记忆（motion 走真实编码器 sidecar）与装配观测；捕获由 float32 模型在 highest 精度下做，
与 hover_replay.build_models 相同。

输入 --points：R1 的 rsprog_points.json（schema "rsprog-points-1"，见 rsprog_points.py 文件头）。本脚本只用：
  episodes[]：k（全局序号 0–199）、difficulty、episode（每难度 1000–1049）、e0、n_frames；
  points[]（顶层，按 k 归局）：k、difficulty、episode、t、u（t − e0 = u，u 为 16 的倍数）；
  dropped[]：其中的 k 整局跳过。局的唯一键为全局序号 k。
H5：与 R1 points 子命令同一查找顺序——<h5-root>/RouteStick/<difficulty>/record_dataset_RouteStick.h5（清单生成布局），
再 <h5-root>/<difficulty>/record_dataset_RouteStick.h5；局名 episode_<episode>；prompt 取 setup/task_goal[0]。
回放前用 H5 的 is_video_demo 核 e0，不符即报错。

npz（<out>/<A|B>/episode_<k>.npz，k 为 R1 全局序号 0–199，与 R3 rsprog_probe 缺省 --npz-pattern 一致；N 为本局推理点数，M 为本局各点有效记忆 token 数的最大值，按记忆序紧排、
不足补齐，mem_len 记每点实际个数）：
  t、u                     (N,)         int64
  hidden_prefix            (N,19,2,W)   f32   前缀两组（0=image、1=lang）逐级有效 token 均值（第 0 级为进第 0 层前）
  hidden_action            (N,18,W_a)   f32   动作位（后 AH 个 token 的前 16 个）均值，去噪步平均，再对 K 个噪声平均
  hidden_memmod            (N,18,W_a)   f32   每层 mem_mod_vec 在同一动作位上的均值，去噪步平均，K 平均
  attn_mem                 (N,18,M)     f32   每层 MemoryAttention 份额（头、动作位、去噪步、K 平均），补齐位为 0
  mem_type                 (N,M)        int8  0 帧 token、1 motion token、−1 补齐
  mem_t0、mem_t1           (N,M)        int32 帧 token 为帧号（两列相同）；motion token 为窗口首末帧
                                              （示范段窗口末帧截到 e0−1，与 demo_tail_pad=repeat_last 的真实帧一致）；补齐 −1
  mem_len                  (N,)         int32
  attn_main                (N,18,3)     f32   动作 token 主注意力在 (lang, image, action) 上的份额，头、动作位、去噪步、K 平均
  film_rel                 (N,18,2)     f32   [:, :, 0] = ‖F(x,m) − F(x,0)‖/‖F(x,0)‖；[:, :, 1] = ‖Σ_motion α·V‖/‖Σ_all α·V‖
                                              （定义见 attn_adapter 文件头），K 平均
  mem_frame_mean、mem_motion_mean、mem_motion_latest  (N,1024) f32   mem_seq 的帧 token 均值、motion 均值、最新 motion token
另有元数据键（不在计划字段表里，只供核对）：k、episode、difficulty、model、e0、action_samples、seed、fingerprint、
attn_mem_rowsum_err、attn_main_rowsum_err、memmod_maxabs_pt（(N,) f32，每点 K 个样本的最大值），
memmod_maxabs（f32 标量 = memmod_maxabs_pt 的局内最大值，供 R3 rsprog_attn 的 ATTN_AGG）。

噪声：第 j 份噪声用 hover_replay.action_noise(seed, k, t, j)，k 为全局序号（四个难度唯一）。

用法（JAX 确定性环境变量与日志由调用方给）：
  python -m scripts.interpret.rsprog_replay --model A --ckpt <…/50000> --points rsprog_points.json --h5-root <根> \
      --out <输出根> [--episodes 0 153 | --shard 0/3] [--samples 5] [--seed 42]   # --episodes 为全局序号 k
  闸门（只读一局，不写 npz）：--mem-blind-check、--parity-check、--precision-check（可同时给），对 --episodes 的第一局
  （缺省本分片第一局）跑，--check-max-points 限制点数。
  --selftest：不加载权重的 CPU 自检（喂帧节奏、点表规范化、MemoryAttention 重写对原模块、npz 打包形状）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import types
import uuid

import numpy as np

DIFFICULTIES = ("easy", "medium", "hard", "xhard")
POINTS_SCHEMA = "rsprog-points-1"
TASK = "RouteStick"
MODEL_KINDS = {"A": "modul_v10_768", "B": "wan_fixedpool768"}
DEFAULT_CONFIG = "mme_vla_suite_b128_80k"
EVAL_BATCH = 16
K_SAMPLES = 5
NUM_STEPS = 10
PARITY_TOL = 1e-3
PRECISION_MIN_RATIO = 10
ROWSUM_TOL = 1e-4
MEMMOD_TOL = 1e-3
FORMAT_VERSION = 1
PREFIX_GROUPS = ("image", "lang")
AVERAGED_KEYS = ("hidden_action", "hidden_memmod", "attn_mem", "attn_main", "film_rel")
NOISE_FREE_KEYS = ("hidden_prefix", "mem_frame_mean", "mem_motion_mean", "mem_motion_latest", "mem_valid", "mem_is_motion")
ERROR_KEYS = ("attn_mem_rowsum_err", "attn_main_rowsum_err", "memmod_maxabs")

# 计划 B4 R2 行的字段表：名字 → (dtype, 形状模板)；模板里 N/M/W/Wa 为符号
NPZ_SPEC = {
    "t": ("int64", ("N",)), "u": ("int64", ("N",)),
    "hidden_prefix": ("float32", ("N", 19, 2, "W")),
    "hidden_action": ("float32", ("N", 18, "Wa")),
    "hidden_memmod": ("float32", ("N", 18, "Wa")),
    "attn_mem": ("float32", ("N", 18, "M")),
    "mem_type": ("int8", ("N", "M")),
    "mem_t0": ("int32", ("N", "M")), "mem_t1": ("int32", ("N", "M")),
    "mem_len": ("int32", ("N",)),
    "attn_main": ("float32", ("N", 18, 3)),
    "film_rel": ("float32", ("N", 18, 2)),
    "mem_frame_mean": ("float32", ("N", 1024)),
    "mem_motion_mean": ("float32", ("N", 1024)),
    "mem_motion_latest": ("float32", ("N", 1024)),
}


# ══════════════════════════════════════════ 点表与分片 ══════════════════════════════════════════
def normalize_entry(ep, points):
    """rsprog-points-1 的一局记录 + 该局的点 → {"k", "difficulty", "episode", "e0", "n_frames", "ts", "us", "h5", "noise_id"}。"""
    k = int(ep["k"])
    difficulty = str(ep["difficulty"])
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"k={k} 未知难度 {difficulty!r}")
    episode, e0 = int(ep["episode"]), int(ep["e0"])
    pts = sorted(points, key=lambda p: int(p["t"]))
    for p in pts:
        if (str(p["difficulty"]), int(p["episode"])) != (difficulty, episode):
            raise ValueError(f"k={k} 的点 (difficulty, episode) 与局记录不符：{p}")
        if int(p["t"]) - e0 != int(p["u"]):
            raise ValueError(f"k={k} 点 t={p['t']} u={p['u']} 与 e0={e0} 不符")
    ts = [int(p["t"]) for p in pts]
    if len(set(ts)) != len(ts):
        raise ValueError(f"k={k} 推理点重复")
    check_eval_points(e0, ts)
    n_frames = ep.get("n_frames")
    if n_frames is not None and ts and max(ts) >= int(n_frames):
        raise ValueError(f"k={k} 推理点 {max(ts)} 超出 n_frames={n_frames}")
    return {"k": k, "difficulty": difficulty, "episode": episode, "e0": e0,
            "n_frames": None if n_frames is None else int(n_frames), "ts": ts, "us": [t - e0 for t in ts],
            "h5": ep.get("h5"), "noise_id": k}


def load_points(path):
    """R1 的 rsprog_points.json（schema rsprog-points-1：顶层 episodes[] 与顶层 points[]，点按 k 归局；
    dropped[] 里的 k 跳过）→ {k: 规范化记录}，按 k 排序。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != POINTS_SCHEMA:
        raise ValueError(f"{path} 的 schema={raw.get('schema')!r}，期望 {POINTS_SCHEMA!r}")
    dropped = {int(d["k"]) for d in raw.get("dropped", [])}
    by_k = {}
    for p in raw["points"]:
        by_k.setdefault(int(p["k"]), []).append(p)
    out = {}
    for ep in raw["episodes"]:
        k = int(ep["k"])
        if k in dropped:
            continue
        if k in out:
            raise ValueError(f"点表里 k={k} 重复")
        out[k] = normalize_entry(ep, by_k.pop(k, []))
    stray = sorted(set(by_k) - dropped)
    if stray:
        raise ValueError(f"点表里有不属于任何保留局的点：k={stray[:5]}")
    return dict(sorted(out.items()))


def shard_of(k, n):
    """按全局序号 k 的 sha256 分片（与机器、顺序无关）。"""
    return int(hashlib.sha256(str(int(k)).encode("utf-8")).hexdigest(), 16) % int(n)


def select_episodes(points, episodes=None, shard=None):
    """--episodes 为全局序号 k 的列表；--shard i/n 按 k 哈希。"""
    keys = list(points)
    if episodes:
        want = [int(e) for e in episodes]
        missing = [e for e in want if e not in points]
        if missing:
            raise SystemExit(f"--episodes 里的 k 不在点表（或已剔除）：{missing}")
        keys = want
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"--shard {shard} 非法")
        keys = [k for k in keys if shard_of(k, n) == i]
    return keys


# ══════════════════════════════════════════ 喂帧节奏 ══════════════════════════════════════════
def check_eval_points(e0, ts, batch=EVAL_BATCH):
    bad = [t for t in ts if t < e0 or (t - e0) % batch]
    if bad:
        raise ValueError(f"推理点必须是 e0 + 16k（e0={e0}）：{bad[:5]}")


def eval_batches(e0, last_t, batch=EVAL_BATCH):
    """评测节奏的帧区间 [start, stop)：首包 [0, e0+1)（示范段加执行段第 0 帧），之后每包 batch 帧，喂到 last_t。"""
    if last_t < e0:
        raise ValueError(f"last_t={last_t} < e0={e0}")
    out = [(0, e0 + 1)]
    start = e0 + 1
    while start <= last_t:
        stop = min(start + batch, last_t + 1)
        out.append((start, stop))
        start = stop
    return out


def h5_exec_start(h5_episode):
    """H5 里第一帧 is_video_demo == False 的帧号。"""
    n = sum(1 for k in h5_episode.keys() if k.startswith("timestep_"))
    for i in range(n):
        if not bool(h5_episode[f"timestep_{i}/info/is_video_demo"][()]):
            return i, n
    raise ValueError("整局都是示范帧")


# ══════════════════════════════════════════ 记忆 token 表 ══════════════════════════════════════════
def memory_token_table(policy, observation, t, *, valid_from_adapter=None):
    """记忆序（mem_order 重排后）下每个 token 的 (valid, type, t0, t1)，均为 (S,) numpy。
    帧路时刻与 _prepare_history 同（get_frame_sampling_indices），motion 起点取 mem_buffer._prepare_motion(t) 的同一返回；
    用 memory_order 重算置换并与观测里的 mem_order 比对，不一致即报错。"""
    from mme_vla_suite.datastore.motion_adapter import STAGE2_SPLIT_READOUT
    from mme_vla_suite.shared.sampling import MEM_ORDER_SENTINEL, memory_order, pad_times
    config = policy.config
    budget, tpi, nv = int(config.budget), int(config.token_per_image), int(config.num_views)
    tpf = tpi * nv
    max_frames = budget // tpf
    frame_times = pad_times(policy.mem_buffer.get_frame_sampling_indices(t, budget, tpi), max_frames)
    static = np.asarray(observation.static_mask).reshape(-1).astype(bool)
    n_frame_tok = static.size
    if observation.mem_order is None:
        order = np.arange(n_frame_tok)
        motion_times = np.zeros(0, np.int64)
        valid = static
    else:
        order = np.asarray(observation.mem_order).reshape(-1).astype(np.int64)
        _, _, _, motion_times = policy.mem_buffer._prepare_motion(int(t))
        if config.motion.get("readout") == STAGE2_SPLIT_READOUT:
            motion_times = np.repeat(motion_times, 4)
        expect = memory_order(frame_times, tpf, motion_times)
        if not np.array_equal(expect, order):
            raise RuntimeError(f"t={t} 重算的 mem_order 与观测不一致")
        valid = np.concatenate([static, np.asarray(observation.motion_mask).reshape(-1).astype(bool)])[order]
    if valid_from_adapter is not None and not np.array_equal(np.asarray(valid_from_adapter, bool), valid):
        raise RuntimeError(f"t={t} 适配器的 mem_valid 与观测 mask 不一致")
    es = int(policy.mem_buffer.exec_start_idx) if policy.motion_enabled else 0
    window = int(policy.mem_buffer.motion_window) if policy.motion_enabled else 0
    S = order.size
    mtype = np.where(order >= n_frame_tok, 1, 0).astype(np.int8)
    t0 = np.full(S, -1, np.int64)
    t1 = np.full(S, -1, np.int64)
    fr = order < n_frame_tok
    t0[fr] = frame_times[order[fr] // tpf]
    t1[fr] = t0[fr]
    mo = ~fr
    if mo.any():
        start = np.asarray(motion_times, np.int64)[order[mo] - n_frame_tok]
        end = start + window - 1
        end = np.where(start < es, np.minimum(end, es - 1), end)
        t0[mo], t1[mo] = start, end
    sentinel = (t0 == MEM_ORDER_SENTINEL)
    if (valid & sentinel).any():
        raise RuntimeError(f"t={t} 有效 token 的时刻是哨兵")
    return {"valid": valid, "type": mtype, "t0": t0, "t1": t1}


def prefix_group_masks(layout):
    """bool[2,P]：image = [0, IMG)，lang = [IMG, P)；modulation 前缀不含记忆。"""
    P, IMG = layout["P"], layout["IMG"]
    if layout["MEM"] != 0:
        raise ValueError("rsprog 回放只用于 modulation 模型（前缀不含记忆）")
    m = np.zeros((2, P), bool)
    m[0, :IMG] = True
    m[1, IMG:] = True
    return m


# ══════════════════════════════════════════ 单点捕获 ══════════════════════════════════════════
def capture_point(adapter, observation, masks, *, seed, noise_id, t, samples, zero_mem=False, return_actions=False):
    """K 份噪声各跑一次 adapter.run(capture_hidden=True)，平均 AVERAGED_KEYS；其余取第 0 份。返回 numpy dict。"""
    import jax

    sums, first, errs, actions = {}, None, {k: 0.0 for k in ERROR_KEYS}, []
    noise_free_diff = 0.0
    with jax.default_matmul_precision("highest"):
        for j in range(samples):
            rng, noise = action_noise_key(seed, noise_id, t, j, (1, adapter.AH, adapter.AD))
            out = adapter.run(rng, observation, noise=noise, num_steps=NUM_STEPS, out_f16=False, capture_hidden=True,
                              group_masks=masks, zero_mem=zero_mem)
            hid, x0 = jax.device_get((out[-1], out[0]))
            hid = {k: np.asarray(v) for k, v in hid.items()}
            if return_actions:
                actions.append(np.asarray(x0))
            for k in AVERAGED_KEYS:
                sums[k] = sums.get(k, 0.0) + hid[k].astype(np.float64)
            for k in ERROR_KEYS:
                errs[k] = max(errs[k], float(hid[k]))
            if first is None:
                first = hid
            else:
                noise_free_diff = max(noise_free_diff, *(float(np.abs(hid[k].astype(np.float64)
                                                                      - first[k].astype(np.float64)).max())
                                                         for k in NOISE_FREE_KEYS if hid[k].size))
    res = {k: (sums[k] / samples).astype(np.float32) for k in AVERAGED_KEYS}
    for k in NOISE_FREE_KEYS:
        res[k] = first[k]
    res.update(errs)
    res["noise_free_maxabs"] = noise_free_diff
    if return_actions:
        res["actions"] = actions
    return res


def action_noise_key(seed, noise_id, t, j, shape):
    from scripts.interpret.hover_replay import action_noise
    return action_noise(seed, noise_id, t, j, shape)


# ══════════════════════════════════════════ 回放一局 ══════════════════════════════════════════
def read_point_inputs(h5_episode, start, stop):
    from scripts.interpret.hover_replay import read_steps
    return read_steps(h5_episode, start, stop)


def episode_prompt(entry, h5_episode):
    goal = h5_episode["setup/task_goal"][()]
    goal = goal[0] if np.ndim(goal) else goal
    return goal.decode("utf-8") if isinstance(goal, bytes) else str(goal)


def replay_episode(policy, adapter, h5_episode, entry, *, seed, samples, ts=None, batch=EVAL_BATCH, on_point=None,
                   capture=True):
    """按评测节奏喂帧（batch 可改，只用于 HIDDEN_PRECISION 的对照节奏），在 ts（缺省 entry 全部点）上捕获。
    on_point(t, observation, masks, record) 在每点调用；capture=False 时不做捕获（只给 on_point 用）。返回行列表。"""
    from scripts.interpret.hover_replay import build_observation
    e0 = entry["e0"]
    e0_h5, n_steps = h5_exec_start(h5_episode)
    if e0_h5 != e0:
        raise RuntimeError(f"k={entry['k']} 点表 e0={e0} 与 H5 is_video_demo 边界 {e0_h5} 不符")
    ts = list(entry["ts"]) if ts is None else list(ts)
    if not ts:
        return []
    check_eval_points(e0, ts, EVAL_BATCH)
    if max(ts) >= n_steps:
        raise RuntimeError(f"k={entry['k']} 推理点 {max(ts)} 超出局长 {n_steps}")
    wanted = set(ts)
    prompt = episode_prompt(entry, h5_episode)
    rows = []
    policy.reset()
    for start, stop in eval_batches(e0, max(ts), batch):
        frames, states = read_point_inputs(h5_episode, start, stop)
        policy.add_buffer({"images": frames, "state": states, "exec_start_idx": int(e0)})
        t = stop - 1
        if t not in wanted:
            continue
        if int(policy.step_idx) != t:
            raise RuntimeError(f"policy.step_idx={policy.step_idx} != t={t}")
        observation = build_observation(policy, {
            "observation/image": frames[-1, 0],
            "observation/wrist_image": h5_episode[f"timestep_{t}/obs/wrist_rgb"][()],
            "observation/state": states[-1],
            "prompt": prompt,
        })
        masks = prefix_group_masks(adapter.layout(observation))
        row = {"t": t, "u": t - e0}
        if capture:
            started = time.perf_counter()
            res = capture_point(adapter, observation, masks, seed=seed, noise_id=entry["noise_id"], t=t, samples=samples)
            row.update(res)
            row["table"] = memory_token_table(policy, observation, t, valid_from_adapter=res["mem_valid"])
            row["seconds"] = time.perf_counter() - started
        rows.append(row)
        if on_point is not None:
            on_point(t, observation, masks, row)
    if [r["t"] for r in rows] != sorted(ts):
        raise RuntimeError(f"k={entry['k']} 实际推理点 {[r['t'] for r in rows]} 与点表不一致")
    return rows


def pack_episode(rows):
    """行列表 → 计划 B4 字段表的数组（attn_mem 与记忆 token 表按有效 token 紧排，补齐到本局最大 M）。"""
    n = len(rows)
    lens = [int(r["table"]["valid"].sum()) for r in rows]
    M = max(lens) if lens else 0
    out = {"t": np.asarray([r["t"] for r in rows], np.int64), "u": np.asarray([r["u"] for r in rows], np.int64),
           "mem_len": np.asarray(lens, np.int32)}
    attn = np.zeros((n, 18, M), np.float32)
    mtype = np.full((n, M), -1, np.int8)
    t0 = np.full((n, M), -1, np.int32)
    t1 = np.full((n, M), -1, np.int32)
    for i, r in enumerate(rows):
        v = r["table"]["valid"]
        k = lens[i]
        attn[i, :, :k] = np.asarray(r["attn_mem"], np.float32)[:, v]
        mtype[i, :k] = r["table"]["type"][v]
        t0[i, :k] = r["table"]["t0"][v]
        t1[i, :k] = r["table"]["t1"][v]
    out.update(attn_mem=attn, mem_type=mtype, mem_t0=t0, mem_t1=t1)
    for k in ("hidden_prefix", "hidden_action", "hidden_memmod", "attn_main", "film_rel",
              "mem_frame_mean", "mem_motion_mean", "mem_motion_latest"):
        out[k] = np.stack([np.asarray(r[k], np.float32) for r in rows]) if rows else np.zeros((0,), np.float32)
    for k in ("attn_mem_rowsum_err", "attn_main_rowsum_err"):
        out[k] = np.asarray([r[k] for r in rows], np.float32)
    pt = np.asarray([r["memmod_maxabs"] for r in rows], np.float32)
    out["memmod_maxabs_pt"] = pt                                       # 逐点（K 个样本的最大值）
    out["memmod_maxabs"] = np.float32(pt.max() if pt.size else 0.0)    # 局级标量，R3 rsprog_attn 的 ATTN_AGG 读这个
    return out


def check_npz_schema(arrays):
    """按 NPZ_SPEC 核对字段名、dtype、形状；返回问题列表（空 = 通过）。"""
    problems = []
    sym = {}
    for name, (dtype, shape) in NPZ_SPEC.items():
        if name not in arrays:
            problems.append(f"缺 {name}")
            continue
        a = arrays[name]
        if str(a.dtype) != dtype:
            problems.append(f"{name} dtype {a.dtype} != {dtype}")
        if a.ndim != len(shape):
            problems.append(f"{name} ndim {a.ndim} != {len(shape)}")
            continue
        for d, s in zip(a.shape, shape):
            if isinstance(s, str):
                if sym.setdefault(s, d) != d:
                    problems.append(f"{name} 维 {s}={d} 与前面 {sym[s]} 不一致")
            elif d != s:
                problems.append(f"{name} 形状 {a.shape} 不符 {shape}")
    return problems


def fingerprint(entry, model, ckpt, config, samples, seed):
    payload = {"entry": {k: entry[k] for k in ("k", "difficulty", "episode", "e0", "ts", "us")}, "model": model, "ckpt": str(ckpt),
               "config": config, "samples": int(samples), "seed": int(seed), "format": FORMAT_VERSION}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_episode(path, arrays, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.npz"
    extra = {"k": np.int64(meta["k"]), "episode": np.int64(meta["episode"]), "difficulty": np.str_(meta["difficulty"]), "model": np.str_(meta["model"]),
             "e0": np.int64(meta["e0"]), "action_samples": np.int64(meta["samples"]), "seed": np.int64(meta["seed"]),
             "fingerprint": np.str_(meta["fingerprint"])}
    np.savez(temporary, **arrays, **extra)
    os.replace(temporary, path)


def existing_points(path, fp):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["fingerprint"]) != fp:
            raise SystemExit(f"{path} 的指纹与当前输入不一致，不覆盖")
        return len(archive["t"])


def h5_path_of(entry, h5_root):
    """与 R1 points 子命令同一查找顺序：<h5-root>/RouteStick/<d>/，再 <h5-root>/<d>/。"""
    if entry.get("h5"):
        return Path(entry["h5"])
    for cand in (Path(h5_root) / TASK / entry["difficulty"], Path(h5_root) / entry["difficulty"]):
        path = cand / f"record_dataset_{TASK}.h5"
        if path.is_file():
            return path
    raise FileNotFoundError(f"k={entry['k']} 在 {h5_root} 下找不到 {TASK}/{entry['difficulty']}/ 或 "
                            f"{entry['difficulty']}/ 的 record_dataset_{TASK}.h5")


def npz_path(out, model, k):
    """与 R3 rsprog_probe 缺省 --npz-pattern "{model}/episode_{k}.npz" 一致。"""
    return Path(out) / model / f"episode_{int(k)}.npz"


def run_episodes(policy, adapter, points, keys, *, model, ckpt, config, h5_root, out, samples, seed):
    import h5py
    expected = sum(len(points[k]["ts"]) for k in keys)
    total, worst = 0, {k: 0.0 for k in ERROR_KEYS}
    for key in keys:
        entry = points[key]
        if not entry["ts"]:
            continue
        path = npz_path(out, model, key)
        fp = fingerprint(entry, model, ckpt, config, samples, seed)
        count = existing_points(path, fp)
        if count is not None:
            total += count
            print(f"REPLAY_EPISODE_SKIP k={key} points={count}", flush=True)
            continue
        started = time.perf_counter()
        with h5py.File(h5_path_of(entry, h5_root), "r") as h5:
            rows = replay_episode(policy, adapter, h5[f"episode_{entry['episode']}"], entry, seed=seed, samples=samples)
        arrays = pack_episode(rows)
        problems = check_npz_schema(arrays)
        if problems:
            raise RuntimeError(f"{key} npz 字段不符：{problems}")
        write_episode(path, arrays, {"k": entry["k"], "episode": entry["episode"], "difficulty": entry["difficulty"], "model": model,
                                     "e0": entry["e0"], "samples": samples, "seed": seed, "fingerprint": fp})
        for k in ERROR_KEYS:
            worst[k] = max(worst[k], float(np.max(arrays[k])) if np.size(arrays[k]) else 0.0)
        total += len(rows)
        point_s = float(np.mean([r["seconds"] for r in rows]))
        print(f"REPLAY_EPISODE k={key} difficulty={entry['difficulty']} episode={entry['episode']} points={len(rows)} M={arrays['attn_mem'].shape[-1]} "
              f"seconds={time.perf_counter() - started:.1f} point_seconds={point_s:.2f} "
              f"rowsum_maxerr={max(float(arrays['attn_mem_rowsum_err'].max()), float(arrays['attn_main_rowsum_err'].max())):.3g} "
              f"memmod_maxabs={float(arrays['memmod_maxabs']):.3g}", flush=True)
    rowsum = max(worst["attn_mem_rowsum_err"], worst["attn_main_rowsum_err"])
    agg = rowsum <= ROWSUM_TOL and worst["memmod_maxabs"] <= MEMMOD_TOL
    print(f"ATTN_AGG_CAPTURE={'PASS' if agg else 'FAIL'} rowsum_maxerr={rowsum:.3g} memmod_maxabs={worst['memmod_maxabs']:.3g}",
          flush=True)
    passed = total == expected and agg
    print(f"REPLAY_RUN={'PASS' if passed else 'FAIL'} model={model} episodes={len(keys)} points={total} expected={expected}",
          flush=True)
    return 0 if passed else 1


# ══════════════════════════════════════════ 闸门 ══════════════════════════════════════════
def mem_blind_check(policy, adapter, h5_episode, entry, *, seed, max_points):
    """同一观测把 mem_seq 换成全零：前缀各层隐状态应逐位不变，动作组应变化。只用 1 份噪声。"""
    ts = entry["ts"][-max_points:]
    worst_prefix, worst_action, worst_zero = 0.0, 0.0, 0.0

    def on_point(t, observation, masks, row):
        nonlocal worst_prefix, worst_action, worst_zero
        a = capture_point(adapter, observation, masks, seed=seed, noise_id=entry["noise_id"], t=t, samples=1)
        z = capture_point(adapter, observation, masks, seed=seed, noise_id=entry["noise_id"], t=t, samples=1, zero_mem=True)
        worst_prefix = max(worst_prefix, float(np.abs(a["hidden_prefix"].astype(np.float64)
                                                      - z["hidden_prefix"].astype(np.float64)).max()))
        worst_action = max(worst_action, float(np.abs(a["hidden_action"].astype(np.float64)
                                                      - z["hidden_action"].astype(np.float64)).max()))
        worst_zero = max(worst_zero, float(np.abs(z["mem_frame_mean"]).max()), float(np.abs(z["mem_motion_mean"]).max()))

    replay_episode(policy, adapter, h5_episode, entry, seed=seed, samples=1, ts=ts, on_point=on_point, capture=False)
    ok = worst_prefix == 0.0 and worst_action > 0.0 and worst_zero == 0.0
    print(f"MEM_BLIND={'PASS' if ok else 'FAIL'} prefix_maxabs={worst_prefix:.9g} action_maxabs={worst_action:.6g} "
          f"points={len(ts)} k={entry['k']}", flush=True)
    return ok


def parity_check(policy, adapter, h5_episode, entry, *, seed, max_points):
    """同一噪声（第 0 份）：捕获路径的动作 vs 同一 float32 模型经 module_jit(model.sample_actions)（即
    policy._sample_actions 的同一构造）的动作；另打印 bf16 policy._sample_actions 的差作参考（不进判定）。"""
    import jax
    from openpi.shared import nnx_utils
    prod32 = nnx_utils.module_jit(adapter.model.sample_actions)
    ts = entry["ts"][:max_points]
    diffs, diffs_bf16 = [], []

    def on_point(t, observation, masks, row):
        rng, noise = action_noise_key(seed, entry["noise_id"], t, 0, (1, adapter.AH, adapter.AD))
        res = capture_point(adapter, observation, masks, seed=seed, noise_id=entry["noise_id"], t=t, samples=1,
                            return_actions=True)
        mine = res["actions"][0].astype(np.float64)
        with jax.default_matmul_precision("highest"):
            ref = np.asarray(jax.device_get(prod32(rng, observation, noise=noise)), np.float64)
        bf16 = np.asarray(jax.device_get(policy._sample_actions(rng, observation, noise=noise)), np.float64)
        diffs.append(float(np.abs(mine - ref).max()))
        diffs_bf16.append(float(np.abs(mine - bf16).max()))

    replay_episode(policy, adapter, h5_episode, entry, seed=seed, samples=1, ts=ts, on_point=on_point, capture=False)
    maxabs = max(diffs, default=float("nan"))
    ok = bool(diffs) and maxabs <= PARITY_TOL
    print(f"REPLAY_PARITY_BF16_INFO points={len(diffs_bf16)} maxabs={max(diffs_bf16, default=float('nan')):.6g}", flush=True)
    print(f"REPLAY_PARITY={'PASS' if ok else 'FAIL'} points={len(diffs)} maxabs={maxabs:.9g} tol={PARITY_TOL:g} "
          f"k={entry['k']}", flush=True)
    return ok


def precision_ratio(main, other):
    """(相邻点距离中位数 ÷ 两条路径的最大差, 最大差)；main/other (n, ...) 同点同形。距离：相邻点同层逐元素最大差，
    跨层与相邻对取中位数（与 hover_replay.action_distance 同口径）。两路径逐位相同时比值记 inf。"""
    a = np.asarray(main, np.float64)
    b = np.asarray(other, np.float64)
    error = float(np.abs(a - b).max())
    if len(a) < 2:
        return float("nan"), error
    flat = a.reshape(a.shape[0], a.shape[1], -1) if a.ndim >= 3 else a.reshape(a.shape[0], 1, -1)
    distance = float(np.median(np.abs(flat[1:] - flat[:-1]).max(axis=-1)))
    if error == 0.0:
        return float("inf"), error
    return distance / error, error


def precision_check(policy, adapter, h5_episode, entry, *, seed, max_points, samples):
    """HIDDEN_PRECISION（沿用 StopCube 口径）：同一局用评测节奏（16）与另一节奏（8）各回放一遍，同点同噪声；
    捕获误差 = 两遍同一特征的最大差，距离 = 相邻推理点同层最大差的中位数；比值 ≥ 10 判 PASS。"""
    ts = entry["ts"][:max_points]
    a = replay_episode(policy, adapter, h5_episode, entry, seed=seed, samples=samples, ts=ts, batch=EVAL_BATCH)
    b = replay_episode(policy, adapter, h5_episode, entry, seed=seed, samples=samples, ts=ts, batch=EVAL_BATCH // 2)
    ok = True
    groups = {"prefix_image": lambda r: r["hidden_prefix"][:, 0], "prefix_lang": lambda r: r["hidden_prefix"][:, 1],
              "action": lambda r: r["hidden_action"], "memmod": lambda r: r["hidden_memmod"],
              "attn_mem": lambda r: r["attn_mem"], "film_rel": lambda r: r["film_rel"]}
    for name, get in groups.items():
        ratio, error = precision_ratio(np.stack([get(r) for r in a]), np.stack([get(r) for r in b]))
        passed = ratio >= PRECISION_MIN_RATIO
        ok &= passed
        print(f"HIDDEN_PRECISION={'PASS' if passed else 'FAIL'} group={name} min_dist_over_err={ratio:.6g} "
              f"path_maxabs={error:.9g} points={len(ts)}", flush=True)
    return ok


# ══════════════════════════════════════════ 模型 ══════════════════════════════════════════
def build_models(args):
    """沿用 hover_replay.build_models：bf16 policy（真实 motion sidecar）+ float32 模型的 AttnAdapter。"""
    from scripts.interpret import hover_replay
    ns = types.SimpleNamespace(ckpt=Path(args.ckpt), config=args.config, seed=args.seed, stage2_python=None,
                               eval_target_contract=None)
    policy, adapter = hover_replay.build_models(ns)
    kind = policy.metadata.get("model_kind")
    if kind != MODEL_KINDS[args.model]:
        raise SystemExit(f"--model {args.model} 期望 model_kind={MODEL_KINDS[args.model]}，checkpoint 实为 {kind}")
    if not adapter.modulation:
        raise SystemExit("rsprog 回放只用于 modulation 模型")
    print(f"[rsprog] model={args.model} kind={kind} ckpt={Path(args.ckpt).resolve()} integration={adapter.integration_type}",
          flush=True)
    return policy, adapter


# ══════════════════════════════════════════ CPU 自检 ══════════════════════════════════════════
def mem_attention_equivalence(seed=0, T=5, S=7):
    """随机小张量上：重写版 MemoryAttention 对 history_gemma.MemoryAttention 的最大差（f32），以及概率行和误差。"""
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.models.integration import history_gemma as hg
    from scripts.interpret.attn_adapter import _MEM_CAPTURE, _make_mem_capture_classes
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    x = jax.random.normal(k1, (1, T, 1024), jnp.float32)
    mem = jax.random.normal(k2, (1, S, 1024), jnp.float32)
    mask = jnp.asarray([[True] * (S - 2) + [False, False]])
    orig = hg.MemoryAttention()
    params = orig.init(k3, x, mem, mask)
    ref = orig.apply(params, x, mem, mask)
    cap_cls, _ = _make_mem_capture_classes()
    _MEM_CAPTURE.clear()
    mine = cap_cls().apply(params, x, mem, mask)
    rec = _MEM_CAPTURE[-1]
    probs = np.asarray(rec["probs"])
    _MEM_CAPTURE.clear()
    return (float(jnp.abs(ref - mine).max()), float(np.abs(probs.sum(-1) - 1).max()),
            float(probs[..., S - 2:].max()))


def selftest():
    ok = True
    b = eval_batches(10, 42)
    good = b[0] == (0, 11) and b[1] == (11, 27) and b[2] == (27, 43) and len(b) == 3
    ok &= good
    print(f"SELFTEST eval_batches={b} ok={int(good)}", flush=True)
    entry = normalize_entry({"k": 103, "difficulty": "medium", "episode": 1003, "e0": 10, "n_frames": 200},
                            [{"k": 103, "difficulty": "medium", "episode": 1003, "t": 26, "u": 16},
                             {"k": 103, "difficulty": "medium", "episode": 1003, "t": 10, "u": 0}])
    good = entry["ts"] == [10, 26] and entry["us"] == [0, 16] and entry["noise_id"] == 103
    ok &= good
    print(f"SELFTEST normalize ok={int(good)}", flush=True)
    d, rs, masked = mem_attention_equivalence()
    good = d <= 1e-5 and rs <= 1e-5 and masked == 0.0
    ok &= good
    print(f"SELFTEST mem_attention maxabs={d:.3g} rowsum_err={rs:.3g} masked_max={masked:.3g} ok={int(good)}", flush=True)
    rows = synthetic_rows(n=3, S=12, W=8, Wa=1024)
    arrays = pack_episode(rows)
    problems = check_npz_schema(arrays)
    ok &= not problems
    print(f"SELFTEST npz_schema problems={problems}", flush=True)
    print(f"RSPROG_SELFTEST={'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def synthetic_rows(n=3, S=12, W=8, Wa=1024, seed=0):
    """合成行（不跑模型），只用于打包与字段表自检。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        valid = np.zeros(S, bool)
        valid[: S - 2 - (n - 1 - i)] = True
        mtype = (np.arange(S) % 3 == 2).astype(np.int8)
        t0 = np.arange(S, dtype=np.int64) * 4
        rows.append({"t": 10 + 16 * i, "u": 16 * i,
                     "hidden_prefix": rng.normal(size=(19, 2, W)).astype(np.float32),
                     "hidden_action": rng.normal(size=(18, Wa)).astype(np.float32),
                     "hidden_memmod": rng.normal(size=(18, Wa)).astype(np.float32),
                     "attn_mem": rng.random((18, S)).astype(np.float32),
                     "attn_main": rng.random((18, 3)).astype(np.float32),
                     "film_rel": rng.random((18, 2)).astype(np.float32),
                     "mem_frame_mean": rng.normal(size=1024).astype(np.float32),
                     "mem_motion_mean": rng.normal(size=1024).astype(np.float32),
                     "mem_motion_latest": rng.normal(size=1024).astype(np.float32),
                     "attn_mem_rowsum_err": 0.0, "attn_main_rowsum_err": 0.0, "memmod_maxabs": 0.0,
                     "table": {"valid": valid, "type": mtype, "t0": t0, "t1": t0 + 32 * mtype}})
    return rows


# ══════════════════════════════════════════ 入口 ══════════════════════════════════════════
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(MODEL_KINDS))
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--h5-root", type=Path)
    parser.add_argument("--episodes", nargs="+", type=int, default=None, help="全局序号 k（R1 的 0–199）")
    parser.add_argument("--shard", default=None, help="i/n：按 k 的 sha256 取第 i 片")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--samples", type=int, default=K_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--mem-blind-check", action="store_true")
    parser.add_argument("--parity-check", action="store_true")
    parser.add_argument("--precision-check", action="store_true")
    parser.add_argument("--check-max-points", type=int, default=1000)
    parser.add_argument("--check-samples", type=int, default=1, help="--precision-check 每点噪声份数")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    for name in ("model", "ckpt", "points", "h5_root"):
        if getattr(args, name) is None:
            parser.error(f"缺 --{name.replace('_', '-')}")
    checks = args.mem_blind_check or args.parity_check or args.precision_check
    if not checks and args.out is None:
        parser.error("全量回放需要 --out")
    if args.samples < 1:
        parser.error("--samples 至少 1")
    points = load_points(args.points)
    keys = select_episodes(points, args.episodes, args.shard)
    if not keys:
        raise SystemExit("选中 0 局")
    print(f"[rsprog] episodes={len(keys)} points={sum(len(points[k]['ts']) for k in keys)} first={keys[0]}", flush=True)
    policy, adapter = build_models(args)
    client = getattr(policy, "_motion_client", None)
    try:
        if checks:
            import h5py
            entry = points[keys[0]]
            ok = True
            with h5py.File(h5_path_of(entry, args.h5_root), "r") as h5:
                ep = h5[f"episode_{entry['episode']}"]
                if args.mem_blind_check:
                    ok &= mem_blind_check(policy, adapter, ep, entry, seed=args.seed, max_points=args.check_max_points)
                if args.parity_check:
                    ok &= parity_check(policy, adapter, ep, entry, seed=args.seed, max_points=args.check_max_points)
                if args.precision_check:
                    ok &= precision_check(policy, adapter, ep, entry, seed=args.seed, max_points=args.check_max_points,
                                          samples=args.check_samples)
            return 0 if ok else 1
        return run_episodes(policy, adapter, points, keys, model=args.model, ckpt=Path(args.ckpt).resolve(),
                            config=args.config, h5_root=args.h5_root, out=args.out, samples=args.samples, seed=args.seed)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
