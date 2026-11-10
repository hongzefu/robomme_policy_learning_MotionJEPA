#!/usr/bin/env python3
"""多步局逐子目标位置 motion 交换的准备脚本（计划 1109-subgoal-position-swap-all-lengths-plan.html 第一部分三–七节；
第二部分〇′开工审查裁决 F1–F12、一、二节 SP-W1 行）。

任务：PatternLock（m = 节点数 − 1）、RouteStick（m = 步数）；难度 easy / medium / hard 都用。

子命令：
  scan  --out <O> --task <T> [--seed0 S] [--max-count 1000] [--workers 8] [--margin 1.3] [--no-env-verify]
        须在 official-sim 解释器里跑（PYTHONPATH=<仓库>:<仓库>/src:<robomme_benchmark/src>）。两步：
        ① 复算（只用 torch 随机数，按场景源码顺序）：seed 从 S 起步长 100，每个 seed 依循环顺序（从 easy/medium/hard 中
           第 i%3 个起）找第一个「该难度复算出的 m 层还没凑够 配额×margin」的难度，凑满全部 m 层或扫完 max-count 即停；
        ② 环境核实（EnvRunner + 自建 metadata <O>/scan_meta/<T>/，与 eval 同一路径）：reset 后读 selected_buttons 得
           完整路径（PL 节点格、方向；RS 格、左右、绕向）、m、示范长度 es 与示范每帧子目标序号 demo_task_index。
        跳过官方 train/test/val metadata 里的 seed；SceneGenerationError 计 skipped。写 <O>/scan_<T>.json，末行
        SP_SCAN=PASS|FAIL task=<T> seeds=<核实可用数> skipped=<k> banned=<k> errors=<k> replay_mismatch=<k> m=<各层可用数>
  pool  --out <O> [--overwrite]
        读两份 scan，每任务按〇′ F1 的 m 配额（PL m=1:8,2:12,3:18,4:20,5:20,6:21,7:21；RS m=2:15,3:15,4:20,5:20,6:25,7:25，
        共 120 = PLFULL_EPISODES）分层取局（层内按 env_seed 升序；某层不足时缺额顺次补给其他层仍有余量的层），局号按
        env_seed 升序 0..119；写 <O>/metadata/record_dataset_<T>_metadata.json（plfull 口径）、<O>/manifest.json、
        <O>/pool.json。每任务一行 SP_POOL=PASS|FAIL task=<T> eps=<n> m=<各层> short=<缺额> overlap=<与官方 seed 交集>，
        另一行 SP_POOL_LIMIT。
  specs --out <O>/<M> --model motion|wan --commit <sha> [路径根参数] [Wan 参数]
        生成 smoke（每任务局号 0 各 1 局 N，录 motion → donors --smoke → 第 1 步 Z 各 1 局）、glsmoke（同局在第一个 GL
        worker 上跑 N，GL_SMOKE 用）、n（全部 240 局 N，录 motion）三阶段 spec，manifests、queues、env-aspen.sh /
        env-gl.sh、launch_stage.sh（aspen tmux）、gl_launch_stage.sh（GL 端 srun 塞进占位）、plan.json。
        spec 带 subgoal_trace=true、profile=plfull、seed 42。会话前缀 sps-。env 文件只设 SPS_M（本模型目录），不碰调用方 O。
  donors --out <O>/<M> [--variant all|core] [--smoke] [--cross-frac 0.5]
        读 N 阶段终局、回执（m、demo_task_index、exec_start_idx）、step_trace 与 <M>/record/*.npz：
        SP_SEGMENT（示范帧段连续、覆盖 0..es−1、段数 = m；RS 每段 50 帧）；W_j 两版（all = 真实帧与段相交的示范窗，
        core = 真实帧完全落在段内的窗）；逐步复原 N 走到第几步；只对「N 走到第 j 步起点」的 (局, j) 干预：
        Bsame（另一局同位置 j、子目标不同且从接收局第 j 步起点可执行）、Bcross（另一局不同位置 j′，条件同上，按
        default_rng(0) 选一半 (局,j)）、Z（零）。供体按 W 内顺序对齐，供体窗少时用供体该段之后的下一个窗补齐（〇′ F7）。
        合成供体键 ep = 10000000 + recv×100 + slot，slot = j×10 + 组号（all：Bsame=1、Bcross=2、Z=3；core：4、5、6）。
        供体对齐后编码与接收局自身同位置原编码逐位相同的 (局,j,组) 跳过（SP_GROUPS 的 same_as_recv）。
        写 <M>/donors_sp/<task>_ep<合成号>.npz 并读回核对；打印 SP_SEGMENT、SP_RECV、DONOR_FILES、DONOR_DISTINCT
        （PASS|WARN，只记录不阻断）、SP_GROUPS；再生成 multieq、swap（Bsame+Z）、cross（Bcross）、seeds（每任务 20 对 seed 1）四阶段 spec。

路径根参数（〇′ F5）：所有写死路径都可覆盖，默认 aspen 值。aspen 端 --repo（运行工作树，默认主检出下
v1-store/worktrees/sp-swap-run）、--store、--bench-src、--sim-python、--policy-python、--stage2-python、--mmevla-store、
--checkpoint、--target-contract、--log-dir；GL 端 --gl-jobs/--gl-per-job 给 worker 数，--gl-release（GL 仓库 = release
目录）、--gl-store（默认 …-eval/v1-store）、--gl-bench-src（默认 <release>/third_party/robomme_benchmark/src）、
--gl-sim-python/--gl-policy-python/--gl-stage2-python（默认 $GLSTORE/envs/{official-sim,policy,stage2}/bin/python）、
--gl-checkpoint（必给）、--gl-target-contract、--gl-log-dir。模型项里以 aspen store / aspen 仓库开头的字符串在 GL spec 里
自动换成 GL store / release。全部产物根 <O> 由 --out 给（实际用 NFS turbo 路径，两端同路径可见）。

帧号与窗口（沿用 FP F1）：帧 k = 第 k 步之后的观测（帧 0 为 reset）；示范窗起点 s=0,32,… 满足 s+16 ≤ es−1，真实帧
[s, min(s+32, es−1)]；执行窗起点 es+32u。示范帧子目标：第 j 步 = demo_task_index == j（〇′ F9），段写成闭区间
[a_j, b_j−1]（〇′ F8）。执行段第 j 步的子目标序号 E_j = m + 2 + j（任务表：0 NO RECORD、1..m 示范移动、m+1 复位、
m+2 复位到起摆姿态、m+3..2m+2 执行移动）。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import re
import shlex
import sys

import numpy as np

from scripts.evaluation import manifest as M
from scripts.interpret import pl_full_prep as P
from scripts.interpret.pl_swap_prep import MAIN, SPEC_TEMPLATE, XLA_MEM_FRACTION

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- 常量
TASKS = ("PatternLock", "RouteStick")
SEED0 = {"PatternLock": 2300000, "RouteStick": 2400000}      # R6 / 〇′ F3
SEED_STEP = 100
MAX_COUNT = 1000
DIFFICULTIES = ("easy", "medium", "hard")
QUOTA = {"PatternLock": {1: 8, 2: 12, 3: 18, 4: 20, 5: 20, 6: 21, 7: 21},
         "RouteStick": {2: 15, 3: 15, 4: 20, 5: 20, 6: 25, 7: 25}}            # 〇′ F1
PER_TASK = 120
PL_CONFIGS = {"easy": (3, (2, 4)), "medium": (4, (3, 5)), "hard": (5, (4, 8))}    # grid、节点数区间
PL_SPACING, PL_CENTER = 0.1, (-0.1, 0.0)
PL_THRESH = 0.025                     # PatternLock.goal_thresh
RS_CONFIGS = {"easy": ((2, 3), False), "medium": ((4, 5), False), "hard": ((4, 7), True)}
RS_ROUTE = (0, 2, 4, 6, 8)
RS_POSTS = (1, 3, 5, 7)
RS_CENTER, RS_SPACING, RS_COLS = (-0.1, 0.0), 0.07, 9
RS_THRESH = 0.03
RS_STEP_FRAMES = 50
SWING_SHORT = {"clockwise": "cw", "counterclockwise": "ccw"}
MOTION_WINDOW = 33
MOTION_STRIDE = 32
DEMO_MIN_REAL = 17
MAX_STEPS = 1300
SYN_BASE = 10000000
GROUP_NO = {"all": {"Bsame": 1, "Bcross": 2, "Z": 3}, "core": {"Bsame": 4, "Bcross": 5, "Z": 6}}
SWAP_GROUPS = ("Bsame", "Bcross", "Z")
PAIR_SEED = 0
MAIN_SEED = 42
SEED_CHECK = 1
SEEDCHECK_SWAP, SEEDCHECK_N = 15, 5   # 每任务 20 对
MULTIEQ_PER_GROUP = 3
CROSS_FRAC = 0.5
DONOR_DIR = "donors_sp"
STAGES = ("smoke", "glsmoke", "n", "multieq", "swap", "cross", "seeds", "swapcore", "crosscore")
SPEC_MAX = 20
CHUNK = 5

# aspen 默认路径根（〇′ F5：全部可经命令行覆盖）
ASPEN = {
    "repo": str(MAIN / "v1-store/worktrees/sp-swap-run"),
    "store": str(MAIN / "v1-store"),
    "bench_src": str(MAIN / "third_party/robomme_benchmark/src"),
    "sim_python": SPEC_TEMPLATE["sim_python"],
    "policy_python": SPEC_TEMPLATE["policy_python"],
    "stage2_python": SPEC_TEMPLATE["stage2_python"],
    "mmevla_store": str(MAIN / "v1-store"),
    "log_dir": str(MAIN / "v1-store/logs/sp-swap-1110"),
}
GL_STORE = "/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store"
ASPEN_WORKERS = ("w00", "w01", "w10", "w11")
ASPEN_CUDA = {"w00": "0", "w01": "0", "w10": "1", "w11": "1"}
ASPEN_ORDER = ("w00", "w10", "w01", "w11")
EVAL_PY = REPO / "examples/robomme/eval.py"

MODELS = {
    "motion": {"checkpoint": str(MAIN / "v1-store/assets/vla-o16cc-59999/59999"),
               "policy_config": "mme_vla_suite_orig16_b128_60k",
               "model_kind": "stage2_pool4_concat",
               "target_contract": SPEC_TEMPLATE["target_contract"]},
    "wan": {"checkpoint": None, "policy_config": None, "model_kind": None, "target_contract": None},
}


def meta_name(task):
    return f"record_dataset_{task}_metadata.json"


def episode_key(task, ep):
    return f"{task}:{int(ep)}"


def split_key(key):
    task, _, ep = str(key).rpartition(":")
    return task, int(ep)


def record_name(task, ep):
    return f"{task}_ep{int(ep)}.npz"


def slot_of(j, group, variant="all"):
    return int(j) * 10 + GROUP_NO[variant][group]


def synthetic_ep(recv_ep, j, group, variant="all"):
    return SYN_BASE + int(recv_ep) * 100 + slot_of(j, group, variant)


def parse_synthetic(ep):
    """合成号 → (接收局, j, 组, 口径)。"""
    rest = int(ep) - SYN_BASE
    recv, slot = divmod(rest, 100)
    j, g = divmod(slot, 10)
    for variant, table in GROUP_NO.items():
        for group, no in table.items():
            if no == g:
                return recv, j, group, variant
    raise ValueError(f"非法合成号 {ep}")


# ---------------------------------------------------------------- 几何：PatternLock
def pl_grid(difficulty):
    return PL_CONFIGS[difficulty][0]


def pl_rc(cell, grid):
    return divmod(int(cell), int(grid))


def pl_dir(a, b, grid):
    (r0, c0), (r1, c1) = pl_rc(a, grid), pl_rc(b, grid)
    return (r1 - r0, c1 - c0)


def pl_step(start, d, grid):
    """从 start 走方向 d 的格；出盘 None。"""
    r, c = pl_rc(start, grid)
    r2, c2 = r + int(d[0]), c + int(d[1])
    return r2 * grid + c2 if 0 <= r2 < grid and 0 <= c2 < grid else None


def pl_xy(cell, grid):
    """PatternLock._load_scene：x = −0.1 + (行 − (G−1)/2)·0.1，y = (列 − (G−1)/2)·0.1。"""
    r, c = pl_rc(cell, grid)
    h = (grid - 1) / 2
    return (PL_CENTER[0] + (r - h) * PL_SPACING, PL_CENTER[1] + (c - h) * PL_SPACING)


def pl_replay(env_seed, difficulty, find_path=None):
    """按 PatternLock._load_scene 顺序复算路径（与 pl_swap_prep.replay_path 同法，按难度换 grid 与长度区间）。"""
    import torch
    if find_path is None:
        from robomme.robomme_env.utils.adjacent import find_path_0_to_8 as find_path
    grid, (lo, hi) = PL_CONFIGS[difficulty]
    g = torch.Generator()
    g.manual_seed(int(env_seed))
    path = None
    for _ in range(1000):
        start, end = torch.randperm(grid * grid, generator=g)[:2].tolist()
        path, _, _, _ = find_path(start=start, target=end, R=grid, C=grid, diagonals=True, generator=g)
        if lo <= len(path) <= hi:
            break
    return {"cells": [int(x) for x in path], "grid": grid}


# ---------------------------------------------------------------- 几何：RouteStick
def rs_xy(cell, theta):
    x0 = RS_CENTER[0]
    y0 = RS_CENTER[1] + (int(cell) - (RS_COLS - 1) / 2) * RS_SPACING
    return (x0 * math.cos(theta) - y0 * math.sin(theta), x0 * math.sin(theta) + y0 * math.cos(theta))


def rs_target(start_cell, side):
    """从路线格 start_cell 按 side（路线下标 ±1）走一步的目标格；出界 None。"""
    i = RS_ROUTE.index(int(start_cell)) + int(side)
    return RS_ROUTE[i] if 0 <= i < len(RS_ROUTE) else None


def rs_replay(env_seed, difficulty):
    """按 RouteStick._load_scene 的随机数顺序复算（同 rs_full_prep.replay_route，按难度换配置）。"""
    import torch
    (lo, hi), backtrack = RS_CONFIGS[difficulty]
    g = torch.Generator()
    g.manual_seed(int(env_seed))
    theta = math.radians((torch.rand(1, generator=g).item() * 60) - 30)
    for _ in RS_POSTS:
        torch.rand(3, generator=g)
    steps = int(torch.randint(lo, hi + 1, (1,), generator=g).item())
    idxs = [torch.randint(0, len(RS_ROUTE), (1,), generator=g).item()]
    for _ in range(steps):
        cur = idxs[-1]
        prev = idxs[-2] if len(idxs) > 1 else None
        neighbors = [n for n in (cur - 1, cur + 1) if 0 <= n < len(RS_ROUTE)]
        cands = neighbors if (backtrack or prev is None) else ([n for n in neighbors if n != prev] or neighbors)
        idxs.append(cands[torch.randint(0, len(cands), (1,), generator=g).item()])
    swings = [SWING_SHORT["clockwise" if torch.rand(1, generator=g).item() < 0.5 else "counterclockwise"]
              for _ in range(steps)]
    return {"cells": [RS_ROUTE[int(i)] for i in idxs], "swings": swings, "theta": float(theta)}


# ---------------------------------------------------------------- 子目标描述（scan 条目 / 局池条目 → 每步）
def subgoals(e):
    """返回第 1..m 步的子目标列表：PL {start, target, dir}；RS {start, target, side, swing}。"""
    cells = e["cells"]
    out = []
    for j in range(1, len(cells)):
        a, b = int(cells[j - 1]), int(cells[j])
        if e["task"] == "PatternLock":
            out.append({"start": a, "target": b, "dir": list(pl_dir(a, b, e["grid"]))})
        else:
            side = RS_ROUTE.index(b) - RS_ROUTE.index(a)
            out.append({"start": a, "target": b, "side": int(side), "swing": e["swings"][j - 1]})
    return out


def subgoal_differs(task, recv_sg, donor_sg):
    """子目标不同：PL 方向不同；RS 左右或绕向不同。"""
    if task == "PatternLock":
        return tuple(recv_sg["dir"]) != tuple(donor_sg["dir"])
    return recv_sg["side"] != donor_sg["side"] or recv_sg["swing"] != donor_sg["swing"]


def donor_dir_target(task, recv_e, j, donor_sg):
    """供体子目标从接收局第 j 步实际起点（= 路径第 j−1 格）出发到达的格；出盘 None（不可执行）。"""
    start = int(recv_e["cells"][j - 1])
    if task == "PatternLock":
        return pl_step(start, donor_sg["dir"], recv_e["grid"])
    return rs_target(start, donor_sg["side"])


# ---------------------------------------------------------------- 窗口
def demo_starts(es):
    out, s = [], 0
    while s + (DEMO_MIN_REAL - 1) <= int(es) - 1:
        out.append(s)
        s += MOTION_STRIDE
    return out


def exec_starts(es, max_steps=MAX_STEPS):
    out, u = [], 0
    while u + (MOTION_WINDOW - 1) <= int(max_steps):
        out.append(int(es) + u)
        u += MOTION_STRIDE
    return out


def all_starts(es, max_steps=MAX_STEPS):
    return demo_starts(es) + exec_starts(es, max_steps)


def real_range(f, es):
    f, es = int(f), int(es)
    if f < es:
        return f, min(f + MOTION_WINDOW - 1, es - 1)
    return f, f + MOTION_WINDOW - 1


def next_start_after(f, es, max_steps=MAX_STEPS):
    later = [s for s in all_starts(es, max_steps) if s > int(f)]
    return min(later) if later else None


def segments_from_labels(labels, es, m):
    """demo_task_index（长度 ≥ es）→ ([(a_j, b_j−1)] 闭区间 j=1..m 或 None, 问题)。只看帧 0..es−1：
    须为按 1..m 顺序的连续非空块（〇′ F9）。"""
    es, m = int(es), int(m)
    lab = [int(x) for x in list(labels)[:es]]
    if len(lab) != es or es <= 0:
        return None, f"len={len(lab)}<es={es}"
    segs, cur, a = [], lab[0], 0
    for k in range(1, es + 1):
        if k == es or lab[k] != cur:
            segs.append((cur, a, k - 1))
            if k < es:
                cur, a = lab[k], k
    order = [s[0] for s in segs]
    if order != list(range(1, m + 1)):
        return None, f"order={order}"
    return [(a_, b_) for _, a_, b_ in segs], None


def segment_ok(task, segs, es):
    """SP_SEGMENT 单局：连续、覆盖 0..es−1；RS 每段 50 帧。返回问题或 None。"""
    if not segs:
        return "no_segments"
    if segs[0][0] != 0 or segs[-1][1] != int(es) - 1:
        return "cover"
    for (a0, b0), (a1, _) in zip(segs, segs[1:]):
        if a1 != b0 + 1:
            return "gap"
    if task == "RouteStick" and any(b - a + 1 != RS_STEP_FRAMES for a, b in segs):
        return "rs_len"
    return None


def window_sets(seg, es):
    """一段 [a, b]（闭区间）→ (all, core) 示范窗起点列表。"""
    a, b = seg
    w_all, w_core = [], []
    for s in demo_starts(es):
        lo, hi = real_range(s, es)
        if lo <= b and hi >= a:
            w_all.append(s)
            if lo >= a and hi <= b:
                w_core.append(s)
    return w_all, w_core


# ---------------------------------------------------------------- scan（official-sim）
def split_metadata_seeds(task, bench_src):
    seeds = set()
    for ds in ("train", "test", "val"):
        path = Path(bench_src) / "robomme/env_metadata" / ds / meta_name(task)
        if path.is_file():
            seeds |= {int(r["seed"]) for r in json.loads(path.read_text())["records"]}
    return seeds


def replay_entry(task, seed, difficulty):
    if task == "PatternLock":
        r = pl_replay(seed, difficulty)
        return {"cells": r["cells"], "grid": r["grid"], "m": len(r["cells"]) - 1}
    r = rs_replay(seed, difficulty)
    return {"cells": r["cells"], "swings": r["swings"], "theta": r["theta"], "m": len(r["cells"]) - 1}


def choose_candidates(task, seeds, replay_fn, margin, banned=()):
    """seeds 升序；每个 seed 从 DIFFICULTIES[i%3] 起循环找第一个复算 m 层还缺的难度。返回 (候选, 计数, 扫了几个)。
    replay_fn(seed, difficulty) → {"m": …, …}。"""
    quota = QUOTA[task]
    want = {m: int(math.ceil(q * margin)) for m, q in quota.items()}
    got = {m: 0 for m in quota}
    cands, scanned = [], 0
    for i, seed in enumerate(seeds):
        if all(got[m] >= want[m] for m in quota):
            break
        scanned += 1
        if seed in banned:
            continue
        for k in range(len(DIFFICULTIES)):
            diff = DIFFICULTIES[(i + k) % len(DIFFICULTIES)]
            rep = replay_fn(seed, diff)
            m = int(rep["m"])
            if m in got and got[m] < want[m]:
                got[m] += 1
                cands.append(dict(rep, task=task, env_seed=int(seed), difficulty=diff, scan_index=i))
                break
    return cands, got, scanned


def scan_metadata(task, cands):
    records = [{"task": task, "episode": i, "seed": int(c["env_seed"]), "difficulty": c["difficulty"]}
               for i, c in enumerate(cands)]
    return {"env_id": task, "record_count": len(records), "records": records}


def env_path(task, u):
    """reset 后从环境读路径：节点按 buttons_grid 身份换成格号；RS 另读绕向。"""
    ids = {id(b): i for i, b in enumerate(u.buttons_grid)}
    cells = [ids[id(b)] for b in u.selected_buttons]
    out = {"cells": cells, "m": len(cells) - 1}
    if task == "RouteStick":
        out["swings"] = [SWING_SHORT.get(s, s) for s in u.swing_directions]
    return out


def verify_one(task, ep, cand, meta_dir, video_dir):
    from examples.robomme.env_runner import EnvRunner
    from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError
    from scripts.interpret.subgoal_trace_runner import DEMO_KEY, find_demonstration_wrapper
    entry = dict(cand)
    runner = None
    try:
        runner = EnvRunner(task, str(video_dir), max_steps=MAX_STEPS, dataset="test", difficulty=cand["difficulty"],
                           metadata_dir=str(meta_dir))
        runner.make_env(int(ep))
        wrapper = find_demonstration_wrapper(runner.env)
        original = wrapper._augment_obs_and_info
        u = runner.env.unwrapped

        def augment(obs, info, action):
            obs, info = original(obs, info, action)
            info[DEMO_KEY] = int(u.current_task_index)
            return obs, info
        wrapper._augment_obs_and_info = augment
        try:
            result = runner.get_init_obs()
        finally:
            del wrapper._augment_obs_and_info
        labels = [int(v) for v in runner.env.demonstration_data[4][DEMO_KEY]]
        es = len(result["images"]) - 1
        real = env_path(task, u)
        entry.update(replay_cells=cand["cells"], replay_swings=cand.get("swings"), replay_m=cand["m"])
        entry.update(real, es=int(es), demo_task_index=labels, status="ok",
                     actual_difficulty=str(u.difficulty))
        entry["replay_match"] = (real["cells"] == cand["cells"]
                                 and (task != "RouteStick" or real["swings"] == cand.get("swings")))
        segs, why = segments_from_labels(labels, es, real["m"])
        entry["segments"] = segs
        entry["segment_problem"] = why or (segment_ok(task, segs, es) if segs else None)
        if entry["actual_difficulty"] != cand["difficulty"]:
            raise RuntimeError(f"难度未生效：{entry['actual_difficulty']} ≠ {cand['difficulty']}")
    except SceneGenerationError as error:
        entry.update(status="skipped", reason=f"SceneGenerationError: {error}")
    except Exception as error:   # 记原因继续
        entry.update(status="error", reason=f"{type(error).__name__}: {error}")
    finally:
        if runner is not None and getattr(runner, "env", None) is not None:
            try:
                runner.env.close()
            except Exception:
                pass
    return entry


def _verify_chunk(args):
    task, items, meta_dir, video_dir = args
    return [verify_one(task, ep, cand, meta_dir, video_dir) for ep, cand in items]


def scan_verdict(task, entries, banned):
    ok = [e for e in entries if e["status"] == "ok"]
    skipped = sum(1 for e in entries if e["status"] == "skipped")
    errors = sum(1 for e in entries if e["status"] == "error")
    mism = sum(1 for e in ok if e.get("replay_match") is False)
    segbad = sum(1 for e in ok if e.get("segment_problem"))
    per_m = {}
    for e in ok:
        if not e.get("segment_problem"):
            per_m[e["m"]] = per_m.get(e["m"], 0) + 1
    verdict = "PASS" if errors == 0 and ok else "FAIL"
    mtxt = ",".join(f"{m}:{per_m.get(m, 0)}" for m in QUOTA[task])
    return (f"SP_SCAN={verdict} task={task} seeds={len(ok)} skipped={skipped} banned={banned} errors={errors} "
            f"replay_mismatch={mism} segment_bad={segbad} m={mtxt}")


def cmd_scan(args):
    out = Path(args.out).resolve()
    task = args.task
    seed0 = args.seed0 if args.seed0 is not None else SEED0[task]
    seeds = [int(seed0) + SEED_STEP * i for i in range(int(args.max_count))]
    banned = split_metadata_seeds(task, args.bench_src)
    cands, got, scanned = choose_candidates(task, seeds, lambda s, d: replay_entry(task, s, d), args.margin, banned)
    n_banned = sum(1 for s in seeds[:scanned] if s in banned)
    print(f"SP_SCAN_REPLAY task={task} scanned={scanned} candidates={len(cands)} "
          + " ".join(f"m{m}={got[m]}" for m in sorted(got)), flush=True)
    if args.no_env_verify:
        entries = [dict(c, status="ok", replay_match=None, es=None, demo_task_index=None, segments=None,
                        segment_problem=None, verified=False) for c in cands]
    else:
        meta_dir = out / "scan_meta" / task
        P._write_once(meta_dir / meta_name(task), P._json_bytes(scan_metadata(task, cands)), args.overwrite)
        video_dir = out / "scan_tmp" / task
        video_dir.mkdir(parents=True, exist_ok=True)
        workers = max(1, int(args.workers))
        items = list(enumerate(cands))
        chunks = [items[i::workers] for i in range(workers)]
        entries = []
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for part in pool.map(_verify_chunk, [(task, c, str(meta_dir), str(video_dir)) for c in chunks if c]):
                entries.extend(part)
                print(f"SCAN_PART task={task} done={len(entries)}/{len(cands)}", flush=True)
        for e in entries:
            e["verified"] = True
        entries.sort(key=lambda e: e["scan_index"])
        for e in entries:
            if e["status"] == "error":
                print(f"SCAN_ERROR seed={e['env_seed']} {e['reason']}", flush=True)
    line = scan_verdict(task, entries, n_banned)
    payload = {"task": task, "seed0": int(seed0), "step": SEED_STEP, "scanned": scanned, "margin": float(args.margin),
               "quota": {str(k): v for k, v in QUOTA[task].items()}, "verdict": line, "entries": entries}
    P._write_json(out / f"scan_{task}.json", payload)
    print(line, flush=True)
    return 0 if line.startswith("SP_SCAN=PASS") else 1


# ---------------------------------------------------------------- pool
def stratified_take(entries, quota, per_task):
    """按 m 分层，层内 env_seed 升序取配额；缺额顺次补给其他仍有余量的层（m 降序，长局优先）。"""
    layers = {}
    for e in sorted(entries, key=lambda x: x["env_seed"]):
        layers.setdefault(int(e["m"]), []).append(e)
    take = {m: min(q, len(layers.get(m, []))) for m, q in quota.items()}
    short = per_task - sum(take.values())
    for m in sorted(quota, reverse=True):
        if short <= 0:
            break
        extra = min(short, len(layers.get(m, [])) - take[m])
        if extra > 0:
            take[m] += extra
            short -= extra
    picked = [e for m in quota for e in layers.get(m, [])[:take[m]]]
    return sorted(picked, key=lambda e: e["env_seed"]), take, max(short, 0)


def pool_episode(task, i, e):
    ep = {"task_name": task, "episode_id": i, "env_seed": int(e["env_seed"]), "difficulty": e["difficulty"],
          "m": int(e["m"]), "cells": [int(c) for c in e["cells"]], "es": e.get("es"),
          "segments": e.get("segments"), "scan_index": int(e["scan_index"])}
    if task == "PatternLock":
        ep["grid"] = pl_grid(e["difficulty"])
    else:
        ep["swings"] = list(e["swings"])
        ep["theta"] = float(e["theta"])
    return ep


def metadata_payload(task, episodes):
    records = [{"task": task, "episode": int(e["episode_id"]), "seed": int(e["env_seed"]), "difficulty": e["difficulty"]}
               for e in episodes]
    return {"env_id": task, "record_count": len(records), "records": records}


def manifest_for(episodes, meta_shas, chunk=CHUNK):
    rows = [{"task_name": e["task_name"], "episode_id": int(e["episode_id"]), "env_seed": int(e["env_seed"]),
             "difficulty": e["difficulty"]} for e in episodes]
    return M.make_manifest(rows, metadata_sha256=dict(meta_shas), chunk_size=chunk, official=False)


def eval_episode_limit(path=EVAL_PY):
    try:
        m = re.search(r"^PLFULL_EPISODES\s*=\s*(\d+)", Path(path).read_text(), re.M)
    except OSError:
        return None
    return int(m.group(1)) if m else None


def entry_task(e, task):
    e = dict(e)
    e["task"] = task
    if task == "PatternLock":
        e.setdefault("grid", pl_grid(e["difficulty"]))
    return e


def cmd_pool(args):
    out = Path(args.out).resolve()
    all_eps, info, lines, ok_all, shas = [], {}, [], True, {}
    for task in TASKS:
        scan = P._read_json(out / f"scan_{task}.json")
        usable = [e for e in scan["entries"] if e["status"] == "ok" and not e.get("segment_problem")]
        picked, take, short = stratified_take(usable, QUOTA[task], PER_TASK)
        eps = [pool_episode(task, i, e) for i, e in enumerate(picked)]
        banned = split_metadata_seeds(task, args.bench_src)
        overlap = sorted({e["env_seed"] for e in eps} & banned)
        data = P._json_bytes(metadata_payload(task, eps))
        shas[meta_name(task)] = hashlib.sha256(data).hexdigest()
        P._write_once(out / "metadata" / meta_name(task), data, args.overwrite)
        ok = len(eps) == PER_TASK and not overlap
        ok_all &= ok
        info[task] = {"usable": len(usable), "take": {str(k): v for k, v in take.items()}, "short": short,
                      "overlap": overlap}
        lines.append(f"SP_POOL={'PASS' if ok else 'FAIL'} task={task} eps={len(eps)} "
                     f"m={','.join(f'{m}:{take[m]}' for m in QUOTA[task])} short={short} overlap={len(overlap)}")
        all_eps += eps
    limit = eval_episode_limit()
    limit_ok = limit is not None and PER_TASK <= limit
    lines.append(f"SP_POOL_LIMIT={'PASS' if limit_ok else 'FAIL'} eval_plfull_episodes={limit} per_task={PER_TASK}")
    manifest = manifest_for(all_eps, shas)
    P._write_once(out / "manifest.json", P._json_bytes(manifest), args.overwrite)
    pool = {"tasks": info, "per_task": PER_TASK, "metadata_sha256": shas, "manifest_seal": manifest["sha256"],
            "numbering": "每任务局号按 env_seed 升序 0..119", "episodes": all_eps}
    P._write_once(out / "pool.json", P._json_bytes(pool), args.overwrite)
    for line in lines:
        print(line, flush=True)
    return 0 if ok_all and limit_ok else 1


def load_pool(pool_root):
    pool = P._read_json(Path(pool_root) / "pool.json")
    by_key = {}
    for e in pool["episodes"]:
        e = dict(e, task=e["task_name"])
        by_key[episode_key(e["task_name"], e["episode_id"])] = e
    return pool, by_key


# ---------------------------------------------------------------- 站点（路径根）与 worker
def site_roots(args):
    """返回 {"aspen": {...}, "gl": {...}|None}。GL 的 store / release 前缀用于替换模型项里的 aspen 路径。"""
    aspen = {k: str(getattr(args, k) or ASPEN[k]) for k in ASPEN}
    gl = None
    if int(args.gl_jobs) > 0:
        if not args.gl_release:
            raise SystemExit("--gl-jobs > 0 时必须给 --gl-release（GL 端 release 目录）")
        store = args.gl_store or GL_STORE
        gl = {"repo": args.gl_release, "store": store,
              "bench_src": args.gl_bench_src or f"{args.gl_release}/third_party/robomme_benchmark/src",
              "sim_python": args.gl_sim_python or f"{store}/envs/official-sim/bin/python",
              "policy_python": args.gl_policy_python or f"{store}/envs/policy/bin/python",
              "stage2_python": args.gl_stage2_python or f"{store}/envs/stage2/bin/python",
              "mmevla_store": args.gl_mmevla_store or store,
              "log_dir": args.gl_log_dir or f"{store}/logs/sp-swap-1110"}
    return {"aspen": aspen, "gl": gl}


def remap(value, src_prefixes, dst_prefixes):
    """字符串以 src 前缀开头时换成对应 dst 前缀（先长后短）；其他类型原样（dict/list 递归）。"""
    if isinstance(value, dict):
        return {k: remap(v, src_prefixes, dst_prefixes) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v, src_prefixes, dst_prefixes) for v in value]
    if isinstance(value, str):
        for src, dst in sorted(zip(src_prefixes, dst_prefixes), key=lambda x: -len(x[0])):
            if value == src or value.startswith(src.rstrip("/") + "/"):
                return dst.rstrip("/") + value[len(src.rstrip("/")):]
    return value


def gl_model(model, aspen, gl, gl_checkpoint=None, gl_target_contract=None, gl_extra=None):
    src = [aspen["store"], aspen["mmevla_store"], str(MAIN)]
    dst = [gl["store"], gl["mmevla_store"], gl["repo"]]
    out = remap(dict(model), src, dst)
    if gl_checkpoint:
        out["checkpoint"] = gl_checkpoint
    if gl_target_contract:
        out["target_contract"] = gl_target_contract
    if gl_extra:
        out["extra"] = dict(gl_extra)
    return out


def make_workers(gl_jobs, gl_per_job):
    """worker 表：[(名, 站点, cuda, GL 占位序号或 None)]；分配顺序 aspen 两卡交替，再 GL 逐占位。"""
    out = [(w, "aspen", ASPEN_CUDA[w], None) for w in ASPEN_ORDER]
    for j in range(int(gl_jobs)):
        for k in range(int(gl_per_job)):
            out.append((f"g{j}{chr(ord('a') + k)}", "gl", "0", j))
    return out


class Builder:
    """收集 spec、清单、plan 行与队列行；按 worker 所在站点选路径根与模型项。"""

    def __init__(self, out, model_name, models, roots, workers, frac, meta_dir, meta_shas, seal=None):
        self.out, self.model_name, self.models, self.roots = Path(out), model_name, models, roots
        self.workers = {w[0]: w for w in workers}
        self.order = [w[0] for w in workers]
        self.frac, self.meta_dir, self.meta_shas = float(frac), Path(meta_dir), dict(meta_shas)
        self.specs, self.rows, self.queue = {}, [], {}

    def site(self, worker):
        return self.workers[worker][1]

    def add(self, name, stage, worker, episodes, *, seed=MAIN_SEED, ablate="none", swap_map=None, record=None,
            chunk=CHUNK, record_dir=None, rows=()):
        if name in self.specs:
            raise ValueError(f"spec 重名 {name}")
        site = self.site(worker)
        roots = self.roots[site]
        run_dir = self.out / "runs" / name
        spec = dict(SPEC_TEMPLATE)
        spec.update(policy_python=roots["policy_python"], sim_python=roots["sim_python"],
                    stage2_python=roots["stage2_python"])
        model = self.models[site]
        spec.update({k: v for k, v in model.items() if k != "extra"})
        spec.update(model.get("extra") or {})
        spec.update(repo=roots["repo"], model_seed=int(seed), cuda=self.workers[worker][2],
                    attn_out=str(run_dir / "attn"), run_root=str(run_dir / "run"), xla_mem_fraction=self.frac,
                    manifest=str(self.out / "manifests" / f"{name}.json"), ablate=ablate, profile="plfull",
                    plfull_metadata_dir=str(self.meta_dir), subgoal_trace=True)
        if swap_map:
            spec["swap_map"] = dict(swap_map)
        if record:
            spec["record_motion"] = str(record)
        manifest = manifest_for(episodes, self.meta_shas, chunk=chunk)
        self.specs[name] = (spec, manifest, stage, worker)
        self.queue.setdefault((stage, worker), []).append(
            f"run_spec {stage} {name} {shlex.quote(str(self.out / 'specs' / f'{name}.json'))}\n")
        for row in rows:
            row = dict(row)
            row.update(stage=stage, worker=worker, site=site, seed=int(seed), spec=name,
                       attn_out=spec["attn_out"], run_root=spec["run_root"],
                       record_dir=str(record_dir or self.out / "record"))
            self.rows.append(row)
        return spec

    def add_cmd(self, stage, worker, text):
        self.queue.setdefault((stage, worker), []).append(f"run_cmd {stage} {shlex.quote(text)}\n")

    def write(self, stages):
        for name, (spec, manifest, stage, worker) in self.specs.items():
            P._write_json(self.out / "manifests" / f"{name}.json", manifest)
            P._write_json(self.out / "specs" / f"{name}.json", spec)
        written = []
        (self.out / "queues").mkdir(parents=True, exist_ok=True)
        for stage in stages:
            for worker in self.order:
                lines = self.queue.get((stage, worker), [])
                n_spec = sum(1 for x in lines if x.startswith("run_spec "))
                site = self.site(worker)
                path = self.out / "queues" / f"{stage}-{worker}.sh"
                text = [QUEUE_HEAD.format(stage=stage, worker=worker, cuda=self.workers[worker][2], n=n_spec,
                                          env=shlex.quote(str(self.out / f"env-{site}.sh")), frac=self.frac)]
                text += lines
                text.append(QUEUE_TAIL.format(stage=stage, worker=worker, n=n_spec))
                path.write_text("".join(text))
                path.chmod(0o755)
                written.append(path)
        return written


def distribute(items, order):
    """静态分片：按序轮转分给 worker。返回 {worker: [item]}。"""
    out = {}
    for i, item in enumerate(items):
        out.setdefault(order[i % len(order)], []).append(item)
    return out


# ---------------------------------------------------------------- 模板
ENV_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/interpret/sp_swap_prep.py specs 生成（站点 {site}）。用法：source env-{site}.sh <GPU> [显存比例，默认 {frac:.2f}]
# 运行树 {run_tree}：aspen 须为 git 工作树且 HEAD = {commit}、clean；GL 为 release 目录，若有 .release_sha 须等于 {commit}。
# 只设 SPS_M（本模型目录），不覆盖调用方的 O（R5）。
set -o pipefail
gpu=$1; frac=${{2:-{frac:.2f}}}
export CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_MEM_FRACTION=$frac
export PYTHONUNBUFFERED=1 UV_CACHE_DIR={store}/cache/uv XDG_CACHE_HOME={store}/cache HF_HOME={store}/cache/hf TORCH_HOME={store}/cache/torch JAX_COMPILATION_CACHE_DIR={store}/cache/jax MMEVLA_V1_STORE={mmevla} OPENPI_DATA_HOME={store}/models HF_HUB_OFFLINE=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
T={run_tree}
export PYTHONPATH=$T:$T/src:{bench}
cd "$T" || return 2
if [ -d .git ] || [ -f .git ]; then
  test "$(git rev-parse HEAD)" = {commit} || {{ echo "ENV_CHECK=FAIL reason=head"; return 2; }}
  test -z "$(git status --porcelain)" || {{ echo "ENV_CHECK=FAIL reason=dirty"; return 2; }}
elif [ -f .release_sha ]; then
  test "$(cat .release_sha)" = {commit} || {{ echo "ENV_CHECK=FAIL reason=release_sha"; return 2; }}
else
  echo "ENV_CHECK=WARN reason=no_git_no_release_sha"
fi
SPS_M={out}
py={py}
sim={sim}
echo "ENV_CHECK=PASS site={site} head={commit} gpu=$gpu frac=$frac"
"""

QUEUE_HEAD = """#!/usr/bin/env bash
# 由 scripts/interpret/sp_swap_prep.py 生成：阶段 {stage}、worker {worker}（GPU {cuda}），{n} 个 spec。
# 某 spec rc≠0 记失败并继续；run_cmd 失败即退出。末行 SPS_QUEUE，有失败时非零退出。
source {env} {cuda} {frac:.2f} || exit $?
fail=0
run_spec() {{
  local stage=$1 name=$2 spec=$3
  "$py" -m scripts.interpret.run_local --spec "$spec"
  local rc=$?
  echo "SPS_SPEC stage=$stage spec=$name rc=$rc"
  if [ "$rc" -ne 0 ]; then fail=$((fail+1)); fi
}}
run_cmd() {{
  local stage=$1 cmd=$2
  eval "$cmd"
  local rc=$?
  echo "SPS_CMD stage=$stage rc=$rc"
  if [ "$rc" -ne 0 ]; then echo "SPS_QUEUE stage=$stage worker={worker} specs={n} failed=cmd"; exit "$rc"; fi
}}
"""

QUEUE_TAIL = """echo "SPS_QUEUE stage={stage} worker={worker} specs={n} failed=$fail"
[ "$fail" -eq 0 ]
"""

LAUNCH_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/interpret/sp_swap_prep.py specs 生成（aspen 端）。用法：bash launch_stage.sh <{stages}>
# 为该阶段有 spec 的 aspen worker 各起一个 detached tmux（sps-{model}-<阶段>-<worker>），日志 {logs}/sps-{model}-<阶段>-<worker>.log。
# 顺序：smoke → glsmoke（GL）→ n → （sp_swap_prep donors）→ multieq → swap → cross → seeds。
SPS_M={out}
st=$1
case "$st" in {stages}) ;; *) echo "未知阶段 $st"; exit 2;; esac
mkdir -p {logs}
for w in {workers}; do
  q=$SPS_M/queues/$st-$w.sh
  [ -f "$q" ] && grep -q '^run_spec ' "$q" || continue
  name=sps-{model}-$st-$w; log={logs}/$name.log
  if tmux has-session -t "=$name" 2>/dev/null; then echo "SKIP_EXISTS $name"; continue; fi
  if [ -e "$log" ]; then echo "SKIP_LOG_EXISTS $log（重跑前先把旧日志改名留档）"; continue; fi
  tmux new-session -d -s "$name" "set -o pipefail; PYTHONUNBUFFERED=1 bash $q 2>&1 | tee $log; echo \\"EXIT_CODE=\\$?\\" >> $log"
  echo "STARTED $name $log"
done
"""

GL_LAUNCH_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/interpret/sp_swap_prep.py specs 生成（GL 登录节点上运行）。用法：bash gl_launch_stage.sh <{stages}> <占位JobID_0> … <占位JobID_{last}>
# 每个 GL worker 起一个 detached tmux（sps-{model}-<阶段>-<worker>），里面 srun 塞进对应占位（R4，不写 --qos）：
#   srun --jobid=<hold> --overlap --exact --ntasks=1 --cpus-per-task={cpus} --gpu_cmode=shared bash <队列>
# 日志 {logs}/sps-{model}-<阶段>-<worker>.log。
SPS_M={out}
st=$1; shift
case "$st" in {stages}) ;; *) echo "未知阶段 $st"; exit 2;; esac
holds=("$@")
[ "${{#holds[@]}}" -eq {jobs} ] || {{ echo "需要 {jobs} 个占位 JobID，得到 ${{#holds[@]}}"; exit 2; }}
mkdir -p {logs}
for spec in {workers}; do
  w=${{spec%%:*}}; j=${{spec##*:}}; hold=${{holds[$j]}}
  q=$SPS_M/queues/$st-$w.sh
  [ -f "$q" ] && grep -q '^run_spec ' "$q" || continue
  name=sps-{model}-$st-$w; log={logs}/$name.log
  if tmux has-session -t "=$name" 2>/dev/null; then echo "SKIP_EXISTS $name"; continue; fi
  if [ -e "$log" ]; then echo "SKIP_LOG_EXISTS $log（重跑前先把旧日志改名留档）"; continue; fi
  tmux new-session -d -s "$name" "set -o pipefail; srun --jobid=$hold --overlap --exact --ntasks=1 --cpus-per-task={cpus} --gpu_cmode=shared bash $q 2>&1 | tee $log; echo \\"EXIT_CODE=\\$?\\" >> $log"
  echo "STARTED $name hold=$hold $log"
done
"""


# ---------------------------------------------------------------- specs
def model_entry(name, overrides=None):
    entry = dict(MODELS[name])
    entry.update({k: v for k, v in (overrides or {}).items() if v is not None})
    missing = [k for k in ("checkpoint", "policy_config", "model_kind", "target_contract") if not entry.get(k)]
    if missing:
        raise SystemExit(f"模型 {name} 缺 {missing}：Wan 项须用 --checkpoint/--policy-config/--model-kind/--target-contract 给出")
    return entry


def n_row(e, **extra):
    row = {"group": "N", "task": e["task_name"], "ep": int(e["episode_id"]), "m": int(e["m"]), "j": None,
           "slot": None, "donor": None, "donor_src": None, "donor_j": None, "variant": None}
    row.update(extra)
    return row


def smoke_episodes(pool_by_key):
    return [pool_by_key[episode_key(t, 0)] for t in TASKS if episode_key(t, 0) in pool_by_key]


def smoke_w(e):
    """smoke 第 1 步 Z 的 W_1（all 版）：由扫描的段与 es 算，与 cmd_donors_smoke 写的起点一致。"""
    if not e.get("segments") or e.get("es") is None:
        return None
    return window_sets(tuple(e["segments"][0]), e["es"])[0]


def build_stage_specs(b, pool_by_key, gl_first=None):
    out = b.out
    smoke = out / "smoke"
    eps = smoke_episodes(pool_by_key)
    b.add("smoke-n", "smoke", "w00", eps, record=smoke / "record", record_dir=smoke / "record",
          rows=[n_row(e) for e in eps])
    b.add_cmd("smoke", "w00", f"\"$py\" -m scripts.interpret.sp_swap_prep donors --out {shlex.quote(str(out))} --smoke")
    b.add("smoke-z", "smoke", "w00", eps, ablate=f"motion_swap_exact:{smoke / DONOR_DIR}",
          swap_map={episode_key(e["task_name"], e["episode_id"]):
                    episode_key(e["task_name"], synthetic_ep(e["episode_id"], 1, "Z")) for e in eps},
          record_dir=smoke / "record",
          rows=[n_row(e, group="Z", j=1, slot=slot_of(1, "Z"), donor=synthetic_ep(e["episode_id"], 1, "Z"),
                      variant="all", w_starts=smoke_w(e)) for e in eps])
    if gl_first:
        b.add("glsmoke-n", "glsmoke", gl_first, eps, record=out / "glsmoke" / "record",
              record_dir=out / "glsmoke" / "record", rows=[n_row(e) for e in eps])
    keys = sorted(pool_by_key, key=lambda k: (-pool_by_key[k]["m"], TASKS.index(split_key(k)[0]), split_key(k)[1]))
    per = distribute(keys, b.order)
    for worker in b.order:
        mine = per.get(worker, [])
        for i in range(0, len(mine), SPEC_MAX):
            chunk = [pool_by_key[k] for k in mine[i:i + SPEC_MAX]]
            b.add(f"n-{worker}-{i // SPEC_MAX:02d}", "n", worker, chunk, record=out / "record",
                  rows=[n_row(e) for e in chunk])


def make_builder(out, model_name, models, roots, workers, frac, pool):
    pool_root = Path(out).parent
    meta_dir = pool_root / "metadata"
    shas = {name: hashlib.sha256((meta_dir / name).read_bytes()).hexdigest() for name in pool["metadata_sha256"]}
    if shas != pool["metadata_sha256"]:
        raise SystemExit("metadata 与 pool.json 记录的 sha256 不符")
    return Builder(out, model_name, models, roots, workers, frac, meta_dir, shas)


def cmd_specs(args):
    out = Path(args.out).resolve()
    if not args.commit or len(args.commit) < 7:
        raise SystemExit("--commit 须给运行工作树的 commit sha")
    if out.name != args.model:
        raise SystemExit(f"--out 末级目录须为模型名 {args.model}（约定 <O>/<M>）")
    roots = site_roots(args)
    extra = json.loads(args.extra) if args.extra else None
    model = model_entry(args.model, {"checkpoint": args.checkpoint, "policy_config": args.policy_config,
                                     "model_kind": args.model_kind, "target_contract": args.target_contract})
    if extra:
        model["extra"] = extra
    models = {"aspen": model}
    if roots["gl"] is not None:
        if not args.gl_checkpoint:
            raise SystemExit("--gl-jobs > 0 时必须给 --gl-checkpoint（GL 端权重目录，两端目录名可能不同）")
        gl_extra = json.loads(args.gl_extra) if args.gl_extra else None
        models["gl"] = gl_model(model, roots["aspen"], roots["gl"], args.gl_checkpoint, args.gl_target_contract,
                                gl_extra)
        if not args.gl_target_contract and not Path(models["gl"]["target_contract"]).exists():
            raise SystemExit(f"GL contract 按改根推出的路径不存在：{models['gl']['target_contract']}；"
                             "请先把 contract 同步到 GL store 同一相对路径，或用 --gl-target-contract 指定")
    pool, pool_by_key = load_pool(out.parent)
    plan_path = out / "plan.json"
    if plan_path.exists() and not args.overwrite:
        raise SystemExit(f"{plan_path} 已存在；确需重写 spec 加 --overwrite（不删任何运行目录）")
    out.mkdir(parents=True, exist_ok=True)
    workers = make_workers(args.gl_jobs if roots["gl"] else 0, args.gl_per_job)
    b = make_builder(out, args.model, models, roots, workers, args.frac, pool)
    gl_first = next((w[0] for w in workers if w[1] == "gl"), None)
    build_stage_specs(b, pool_by_key, gl_first)
    for site, r in roots.items():
        if r is None:
            continue
        (out / f"env-{site}.sh").write_text(ENV_TEMPLATE.format(
            site=site, frac=args.frac, run_tree=r["repo"], commit=args.commit, store=r["store"], mmevla=r["mmevla_store"],
            bench=r["bench_src"], out=out, py=r["policy_python"], sim=r["sim_python"]))
    launch = out / "launch_stage.sh"
    launch.write_text(LAUNCH_TEMPLATE.format(out=out, model=args.model, logs=roots["aspen"]["log_dir"],
                                             workers=" ".join(w[0] for w in workers if w[1] == "aspen"),
                                             stages="|".join(STAGES)))
    launch.chmod(0o755)
    if roots["gl"] is not None:
        gl = out / "gl_launch_stage.sh"
        gl.write_text(GL_LAUNCH_TEMPLATE.format(out=out, model=args.model, logs=roots["gl"]["log_dir"],
                                                workers=" ".join(f"{w[0]}:{w[3]}" for w in workers if w[1] == "gl"),
                                                stages="|".join(STAGES), jobs=int(args.gl_jobs),
                                                last=int(args.gl_jobs) - 1, cpus=int(args.gl_cpus_per_worker)))
        gl.chmod(0o755)
    queues = b.write(STAGES)
    plan = {"tasks": list(TASKS), "model": args.model, "commit": args.commit, "out": str(out),
            "pool_root": str(out.parent), "record_dir": str(out / "record"), "models": models, "roots": roots,
            "workers": [list(w) for w in workers], "manifest_seal": pool["manifest_seal"], "frac": float(args.frac),
            "runs": b.rows}
    P._write_json(plan_path, plan)
    counts = {st: sum(1 for r in b.rows if r["stage"] == st) for st in ("smoke", "glsmoke", "n")}
    print("SPS_SPECS model=" + args.model + " " + " ".join(f"{k}={v}" for k, v in counts.items())
          + f" specs={len(b.specs)} queues={len(queues)} workers={len(workers)}", flush=True)
    return 0


def builder_from_plan(out, plan, pool):
    workers = [tuple(w) for w in plan["workers"]]
    return make_builder(out, plan["model"], plan["models"], plan["roots"], workers, plan.get("frac", XLA_MEM_FRACTION),
                        pool)


# ---------------------------------------------------------------- 读 N 阶段
def read_step_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("event", "step") == "step":
                    rows.append(row)
    return rows


def find_run(run_root, attn_out, key):
    run_root = Path(run_root)
    terminals = sorted(run_root.glob(f"accepted/{key}.*.terminal.json"), key=lambda p: p.stat().st_mtime)
    out = {"key": key, "terminal": None, "receipt": None, "trace": None, "motion_content": None, "terminals": len(terminals)}
    if not terminals:
        return out
    terminal = P._read_json(terminals[-1])
    out["terminal"] = terminal
    nonce = terminal["episode_nonce"]
    traces = sorted(run_root.glob(f"attempts/*/client/traces/{nonce}/step_trace.jsonl"))
    if traces:
        out["trace"] = traces[0]
        rp = traces[0].parent / "meta" / "hold_receipt.json"
        if rp.is_file():
            out["receipt"] = P._read_json(rp)
    episode = Path(attn_out) / nonce / "episode.json"
    if episode.is_file():
        out["motion_content"] = P._read_json(episode).get("motion_content")
    return out


def load_record(path):
    if path is None or not Path(path).is_file():
        return None
    with np.load(path) as z:
        sf = np.asarray(z["start_frames"], np.int64).reshape(-1)
        embs = np.asarray(z["embs"], np.float32)
    if embs.ndim != 2 or embs.shape[0] != sf.size:
        return None
    rec = {}
    for i, s in enumerate(sf.tolist()):
        rec.setdefault(int(s), embs[i].copy())
    return rec


# ---------------------------------------------------------------- 逐步复原
def nearest_cell(task, xy, e):
    """离 xy 最近的格（PL 全部按钮；RS 路线格）。"""
    if task == "PatternLock":
        cells = range(e["grid"] * e["grid"])
        centers = {c: pl_xy(c, e["grid"]) for c in cells}
    else:
        centers = {c: rs_xy(c, e["theta"]) for c in RS_ROUTE}
    return min(centers, key=lambda c: math.hypot(xy[0] - centers[c][0], xy[1] - centers[c][1]))


def first_hit(task, tcps, start_cell, e):
    """段内第一次水平距离 < 阈值的非起点格；返回 (格或 None, 命中下标或 None)。"""
    if task == "PatternLock":
        centers = {c: pl_xy(c, e["grid"]) for c in range(e["grid"] * e["grid"]) if c != int(start_cell)}
        thresh = PL_THRESH
    else:
        centers = {c: rs_xy(c, e["theta"]) for c in RS_ROUTE if c != int(start_cell)}
        thresh = RS_THRESH
    for i, tcp in enumerate(tcps):
        x, y = float(tcp[0]), float(tcp[1])
        hits = sorted((math.hypot(x - cx, y - cy), c) for c, (cx, cy) in centers.items())
        if hits and hits[0][0] < thresh:
            return hits[0][1], i
    return None, None


def swing_side(points, prev_xy, curr_xy):
    """RouteStick.direction_fail 的平均叉积：> 0 cw，< 0 ccw，其他 None。"""
    lx, ly = curr_xy[0] - prev_xy[0], curr_xy[1] - prev_xy[1]
    if math.hypot(lx, ly) < 1e-6 or not points:
        return None
    cross = [lx * (float(p[1]) - prev_xy[1]) - ly * (float(p[0]) - prev_xy[0]) for p in points]
    avg = sum(cross) / len(cross)
    return "cw" if avg > 0 else "ccw" if avg < 0 else None


def reconstruct(task, e, rows, m=None, success=False):
    """step_trace 行 → 每步 [{i, start, reached, completed, swing, how}]（只到第一个未完成步为止）。
    第 i 步完成 = 某行 task_index_before == E_i 且 task_index_after > E_i（E_i = m+2+i），到达格 = 该行 tcp_after 最近格；
    未完成的那一步按几何取第一次到达的非起点格（RS 另算绕向），段 = 上一步完成后的各点。整局 success 时末步可无跳变。"""
    m = int(m if m is not None else e["m"])
    tcps = [r.get("tcp_after") for r in rows]
    steps, pos, start = [], 0, int(e["cells"][0])
    for i in range(1, m + 1):
        E = m + 2 + i
        done_at = None
        for k in range(pos, len(rows)):
            tb, ta = rows[k].get("task_index_before"), rows[k].get("task_index_after")
            if tb is not None and ta is not None and int(tb) == E and int(ta) > E:
                done_at = k
                break
        seg_pts = [t for t in tcps[pos:(done_at + 1 if done_at is not None else len(rows))] if t is not None]
        if done_at is not None and tcps[done_at] is not None:
            reached, how = nearest_cell(task, tcps[done_at], e), "jump"
            pts = seg_pts
        elif done_at is None and success and i == m and seg_pts:
            reached, how = nearest_cell(task, seg_pts[-1], e), "success_end"
            done_at = len(rows) - 1
            pts = seg_pts
        else:
            reached, hit = first_hit(task, seg_pts, start, e)
            how = "geom" if reached is not None else "none"
            pts = seg_pts[:hit + 1] if hit is not None else seg_pts
        swing = None
        if task == "RouteStick" and reached is not None:
            swing = swing_side(pts, rs_xy(start, e["theta"]), rs_xy(reached, e["theta"]))
        steps.append({"i": i, "start": start, "reached": reached, "completed": done_at is not None, "swing": swing,
                      "how": how})
        if done_at is None:
            break
        pos, start = done_at + 1, int(reached)
    return steps


def completed_steps(steps):
    n = 0
    for s in steps:
        if not s["completed"]:
            break
        n += 1
    return n


# ---------------------------------------------------------------- donors：读 N
def episode_info(e, run, record_dir):
    """一局 N：status ok / no_terminal / no_receipt / no_trace / segment_bad / m_mismatch。"""
    task, ep = e["task_name"], int(e["episode_id"])
    info = {"task": task, "ep": ep, "key": episode_key(task, ep), "m": int(e["m"]), "status": "ok"}
    if run["terminal"] is None:
        info["status"] = "no_terminal"
        return info
    if run["receipt"] is None or run["trace"] is None:
        info["status"] = "no_receipt" if run["trace"] is not None else "no_trace"
        return info
    rc = run["receipt"]
    es = rc.get("exec_start_idx")
    if rc.get("m") is not None and int(rc["m"]) != int(e["m"]):
        info.update(status="m_mismatch", receipt_m=rc.get("m"))
        return info
    segs, why = segments_from_labels(rc.get("demo_task_index") or [], es or 0, e["m"])
    why = why or segment_ok(task, segs, es)
    info["es"] = None if es is None else int(es)
    if why:
        info.update(status="segment_bad", segment_problem=why)
        return info
    windows = [window_sets(s, es) for s in segs]
    rows = read_step_rows(run["trace"])
    success = run["terminal"].get("outcome") == "success"
    steps = reconstruct(task, e, rows, e["m"], success)
    rec_path = Path(record_dir) / record_name(task, ep)
    info.update(segments=[list(s) for s in segs], W_all=[w[0] for w in windows], W_core=[w[1] for w in windows],
                steps=steps, completed=completed_steps(steps), outcome=run["terminal"].get("outcome"),
                record=str(rec_path) if rec_path.is_file() else None)
    return info


def collect_n(plan, pool_by_key, record_dir):
    infos = {}
    for row in plan["runs"]:
        if row["stage"] != "n":
            continue
        key = episode_key(row["task"], row["ep"])
        infos[key] = episode_info(pool_by_key[key], find_run(row["run_root"], row["attn_out"], key), record_dir)
    return infos


# ---------------------------------------------------------------- donors：分组
def aligned_rows(recv_W, donor_W, donor_rec, donor_es):
    """接收局 W 第 k 窗 ← 供体 W 第 k 窗；供体窗少时用供体段之后的下一个窗补齐（〇′ F7），多出的不用。
    返回 (embs 或 None, 补齐数, 源起点, 问题)。"""
    srcs = list(donor_W[:len(recv_W)])
    pad = len(recv_W) - len(srcs)
    if pad > 0:
        nxt = next_start_after(max(donor_W), donor_es) if donor_W else None
        if nxt is None:
            return None, pad, srcs, "no_pad_window"
        srcs += [nxt] * pad
    missing = [s for s in srcs if s not in donor_rec]
    if missing:
        return None, pad, srcs, f"donor_missing_starts={missing}"
    return np.stack([donor_rec[s] for s in srcs]).astype(np.float32), pad, srcs, None


def recv_usable(v):
    """可作接收局：N 状态 ok、有录制 npz、终局不是 error。"""
    return v["status"] == "ok" and bool(v.get("record")) and v.get("outcome") != "error"


def recv_positions(infos, task, variant="all"):
    """可干预的 (接收局键, j)：接收局可用（recv_usable），N 走到第 j 步起点（前 j−1 步完成），W_j 非空。"""
    out = []
    for key in sorted((k for k, v in infos.items() if v["task"] == task and recv_usable(v)),
                      key=lambda k: split_key(k)[1]):
        v = infos[key]
        Ws = v["W_all"] if variant == "all" else v["W_core"]
        for j in range(1, v["m"] + 1):
            if v["completed"] >= j - 1 and Ws[j - 1]:
                out.append((key, j))
    return out


def assign_groups(infos, pool_by_key, task, records, variant="all", cross_frac=CROSS_FRAC, seed=PAIR_SEED):
    """返回 (runs, notes)。Bsame：另一局同位置 j、子目标不同、从接收局第 j 步起点可执行；Bcross：另一局 j′≠j，
    条件同上，只对 default_rng(seed) 选出的一半 (局,j) 做；Z：零。供体须有 N 录制且对齐可行。每 (局,j,组) 一个供体，
    各组一个 default_rng(seed)，按 (局,j) 升序逐个 choice。"""
    Wkey = "W_all" if variant == "all" else "W_core"
    positions = recv_positions(infos, task, variant)
    donors = sorted((v for v in infos.values() if v["task"] == task and v["status"] == "ok" and records.get(v["key"])),
                    key=lambda v: v["ep"])
    n_cross = int(math.ceil(len(positions) * cross_frac))
    rng_pick = np.random.default_rng(seed)
    cross_set = set()
    if positions and n_cross:
        idx = rng_pick.choice(len(positions), size=min(n_cross, len(positions)), replace=False)
        cross_set = {positions[i] for i in idx.tolist()}
    rng = {"Bsame": np.random.default_rng(seed), "Bcross": np.random.default_rng(seed)}
    runs, notes = [], {"no_donor": [], "infeasible": 0}
    for key, j in positions:
        r = infos[key]
        re_ = pool_by_key[key]
        rsg = subgoals(re_)[j - 1]
        rW = r[Wkey][j - 1]
        base = {"recv": key, "task": task, "ep": r["ep"], "m": r["m"], "j": j, "variant": variant}
        runs.append(dict(base, group="Z", donor_src=None, donor_j=None, w_starts=list(rW)))
        for group in ("Bsame", "Bcross"):
            if group == "Bcross" and (key, j) not in cross_set:
                continue
            cands = []
            for d in donors:
                if d["key"] == key:
                    continue
                de = pool_by_key[d["key"]]
                dsgs = subgoals(de)
                jprimes = [j] if group == "Bsame" else [x for x in range(1, d["m"] + 1) if x != j]
                for jp in jprimes:
                    if jp > d["m"] or not d[Wkey][jp - 1]:
                        continue
                    dsg = dsgs[jp - 1]
                    if not subgoal_differs(task, rsg, dsg) or donor_dir_target(task, re_, j, dsg) is None:
                        continue
                    if aligned_rows(rW, d[Wkey][jp - 1], records[d["key"]], d["es"])[0] is None:
                        notes["infeasible"] += 1
                        continue
                    cands.append((d["key"], jp))
            if not cands:
                notes["no_donor"].append(f"{key}:{j}:{group}")
                continue
            pick = cands[int(rng[group].integers(len(cands)))]
            runs.append(dict(base, group=group, donor_src=pick[0], donor_j=int(pick[1]), w_starts=list(rW)))
    for run in runs:
        run["slot"] = slot_of(run["j"], run["group"], variant)
        run["donor"] = synthetic_ep(run["ep"], run["j"], run["group"], variant)
    return runs, notes


# ---------------------------------------------------------------- donors：合成文件
def donor_payload(run, infos, records):
    W = run["w_starts"]
    rec = records[run["recv"]]
    dim = next(iter(rec.values())).shape[0]
    meta = {"pad": 0, "src_starts": None}
    if run["group"] == "Z":
        return W, np.zeros((len(W), dim), np.float32), meta
    d = infos[run["donor_src"]]
    Wkey = "W_all" if run["variant"] == "all" else "W_core"
    embs, pad, srcs, why = aligned_rows(W, d[Wkey][run["donor_j"] - 1], records[run["donor_src"]], d["es"])
    if embs is None:
        return None, None, why
    meta.update(pad=pad, src_starts=srcs)
    return W, embs, meta


def write_donor_files(folder, runs, infos, records):
    folder = Path(folder)
    problems, done, integrity = [], [], 0
    stats = {"files": 0, "pad_runs": 0, "pad_windows": 0, "maxabs": 0.0}
    for run in runs:
        starts, embs, meta = donor_payload(run, infos, records)
        if starts is None:
            problems.append(f"{run['recv']}:{run['j']}:{run['group']}:{meta}")
            continue
        path = folder / record_name(run["task"], run["donor"])
        P._savez_atomic(path, start_frames=np.asarray(starts, np.int32), embs=embs.astype(np.float32),
                        counts=np.ones(len(starts), np.int32))
        with np.load(path) as back:
            b_starts, b_embs = np.asarray(back["start_frames"]).tolist(), np.asarray(back["embs"], np.float32)
        if b_starts != [int(s) for s in starts] or b_embs.shape != embs.shape:
            problems.append(f"{run['recv']}:{run['j']}:{run['group']}:readback_shape")
            integrity += 1
            continue
        dev = float(np.max(np.abs(b_embs - embs))) if embs.size else 0.0
        stats["maxabs"] = max(stats["maxabs"], dev)
        if dev != 0.0:
            problems.append(f"{run['recv']}:{run['j']}:{run['group']}:maxabs={dev}")
            integrity += 1
            continue
        stats["files"] += 1
        if meta["pad"]:
            stats["pad_runs"] += 1
            stats["pad_windows"] += meta["pad"]
        done.append(dict(run, pad=meta["pad"], src_starts=meta["src_starts"]))
    ok = integrity == 0
    line = (f"DONOR_FILES={'PASS' if ok else 'FAIL'} files={stats['files']} skipped={len(problems)} "
            f"integrity={integrity} pad_runs={stats['pad_runs']} pad_windows={stats['pad_windows']} "
            f"maxabs={stats['maxabs']:g}")
    return ok, line, problems, done


def drop_same_as_recv(runs, infos, records):
    """剔除供体对齐后编码与接收局自身同位置原编码逐位相同的干预 run（这次「交换」等于没换）。
    接收局 W 窗原编码取自其 N 录制；录制缺窗或供体对齐不可行时不判、原样保留（由 write_donor_files 处理）。
    返回 (保留的 runs, 被剔除的 "<接收局>:<j>:<组>" 列表)。Z 组不判。"""
    kept, same = [], []
    for run in runs:
        if run["group"] not in ("Bsame", "Bcross"):
            kept.append(run)
            continue
        rec = records.get(run["recv"])
        W = run["w_starts"]
        starts, embs, _ = donor_payload(run, infos, records)
        if rec is None or starts is None or any(f not in rec for f in W):
            kept.append(run)
            continue
        own = np.stack([rec[f] for f in W]).astype(np.float32)
        if own.shape == embs.shape and own.tobytes() == embs.astype(np.float32).tobytes():
            same.append(f"{run['recv']}:{run['j']}:{run['group']}")
            continue
        kept.append(run)
    return kept, same


def distinct_line(runs, infos, records, pool_by_key=None, same_as_recv=0):
    """各 (供体局, j′) 源编码（对齐前，供体 W_j′ 拼接）两两按字节比较。只记录不阻断：本计划每 (局,j,组) 只有
    一个供体，不同供体之间全同不影响读数（如 PatternLock 不同 seed 同场景同路径，示范编码逐位相同）；
    真正有意义的「供体与接收局同编码」由 drop_same_as_recv 剔除，其跳过数 same_as_recv 一并写进判定行。
    给 pool_by_key 时对每个全同对注明两侧子目标是否相同（subgoal_differs 口径），统计 same_subgoal / diff_subgoal；
    子目标不同而编码相同是要记录的发现，不阻断。identical 的每项为 (a, b, 子目标相同 True|False|None)。"""
    srcs = sorted({(r["donor_src"], r["donor_j"], r["variant"]) for r in runs if r["group"] in ("Bsame", "Bcross")})
    digests = {}
    for key, jp, variant in srcs:
        rec = records.get(key)
        W = infos[key]["W_all" if variant == "all" else "W_core"][jp - 1]
        if rec is None or any(f not in rec for f in W):
            continue
        digests[(key, jp, variant)] = hashlib.sha256(np.stack([rec[f] for f in W]).astype(np.float32).tobytes()).hexdigest()
    keys = sorted(digests)
    identical = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if digests[a] != digests[b]:
                continue
            same_sg = None
            if pool_by_key is not None:
                ea, eb = pool_by_key[a[0]], pool_by_key[b[0]]
                same_sg = not subgoal_differs(ea["task"], subgoals(ea)[a[1] - 1], subgoals(eb)[b[1] - 1])
            identical.append((a, b, same_sg))
    ok = not identical and len(digests) == len(srcs)
    n_same = sum(1 for t in identical if t[2] is True)
    n_diff = sum(1 for t in identical if t[2] is False)
    line = (f"DONOR_DISTINCT={'PASS' if ok else 'WARN'} donors={len(digests)} identical_pairs={len(identical)} "
            f"same_subgoal={n_same} diff_subgoal={n_diff} same_as_recv={same_as_recv}")
    return line, ok, identical


def segment_line(infos):
    rows = [v for v in infos.values() if v["status"] in ("ok", "segment_bad")]
    bad = [v["key"] for v in rows if v["status"] == "segment_bad"]
    lines = []
    for task in TASKS:
        mine = [v for v in rows if v["task"] == task]
        tb = [v for v in mine if v["status"] == "segment_bad"]
        lines.append(f"SP_SEGMENT={'PASS' if not tb and mine else 'FAIL'} task={task} eps={len(mine)} bad={len(tb)}")
    return lines, bad


def recv_line(task, infos, positions):
    mine = [v for v in infos.values() if v["task"] == task]
    ok = [v for v in mine if v["status"] == "ok"]
    by_m = {}
    for key, j in positions:
        m = infos[key]["m"]
        by_m.setdefault(m, {}).setdefault(j, 0)
        by_m[m][j] += 1
    txt = " ".join(f"m{m}:" + ",".join(f"j{j}={n}" for j, n in sorted(js.items())) for m, js in sorted(by_m.items()))
    no_rec = sum(1 for v in ok if not v.get("record"))
    err = sum(1 for v in ok if v.get("outcome") == "error")
    return (f"SP_RECV task={task} eps={len(mine)} ok={len(ok)} success={sum(1 for v in ok if v['outcome'] == 'success')} "
            f"no_record={no_rec} error_outcome={err} positions={len(positions)} {txt}")


# ---------------------------------------------------------------- donors：阶段 spec
def run_row(run):
    return {"group": run["group"], "task": run["task"], "ep": int(run["ep"]), "m": int(run["m"]), "j": int(run["j"]),
            "slot": run["slot"], "donor": int(run["donor"]), "donor_src": run.get("donor_src"),
            "donor_j": run.get("donor_j"), "variant": run["variant"], "w_starts": run.get("w_starts"),
            "pad": run.get("pad", 0), "src_starts": run.get("src_starts")}


def seedcheck_runs(done, infos, seed=PAIR_SEED):
    """每任务 20 对：15 个干预局 + 5 个 N 局（接收局），default_rng(seed) 抽，用 seed 1 再跑。"""
    out = []
    for task in TASKS:
        rng = np.random.default_rng(seed)
        swaps = [r for r in done if r["task"] == task and r["group"] in ("Bsame", "Z")]
        if swaps:
            idx = sorted(rng.choice(len(swaps), size=min(SEEDCHECK_SWAP, len(swaps)), replace=False).tolist())
            out += [dict(swaps[i], seed=SEED_CHECK) for i in idx]
        recvs = sorted({r["recv"] for r in done if r["task"] == task}, key=lambda k: split_key(k)[1])
        if recvs:
            idx = sorted(rng.choice(len(recvs), size=min(SEEDCHECK_N, len(recvs)), replace=False).tolist())
            out += [{"recv": recvs[i], "task": task, "ep": split_key(recvs[i])[1], "m": infos[recvs[i]]["m"],
                     "group": "N", "seed": SEED_CHECK} for i in idx]
    return out


def multieq_runs(done):
    chosen, used = [], set()
    for group in ("Bsame", "Z"):
        n = 0
        for r in done:
            if r["group"] != group or r["recv"] in used:
                continue
            chosen.append(r)
            used.add(r["recv"])
            n += 1
            if n == MULTIEQ_PER_GROUP:
                break
    return chosen


def pack_runs(runs, key_fn, cap=SPEC_MAX):
    """按 key 装箱，箱内接收局唯一、≤ cap（同一 (j, 组) 的不同接收局进同一 spec，〇′ F11）。返回 [(key, [run])]。"""
    return [(key, [t[2] for t in members]) for key, members in P.pack([(key_fn(r), r["recv"], r) for r in runs], cap)]


def build_swap_specs(b, pool_by_key, done, seed_runs, variant="all"):
    ablate = f"motion_swap_exact:{b.out / DONOR_DIR}"
    tag = "" if variant == "all" else "core"
    aspen = [w for w in b.order if b.site(w) == "aspen"]
    chosen = multieq_runs(done) if variant == "all" else []
    for i, run in enumerate(chosen):
        b.add(f"multieq{tag}-single-{i:02d}", "multieq", aspen[i % len(aspen)], [pool_by_key[run["recv"]]],
              ablate=ablate, swap_map={run["recv"]: episode_key(run["task"], run["donor"])}, chunk=1,
              rows=[dict(run_row(run), mvariant="single", pair_index=i)])
    if chosen:
        b.add(f"multieq{tag}-multi", "multieq", aspen[-1], [pool_by_key[r["recv"]] for r in chosen], ablate=ablate,
              swap_map={r["recv"]: episode_key(r["task"], r["donor"]) for r in chosen},
              rows=[dict(run_row(r), mvariant="multi", pair_index=i) for i, r in enumerate(chosen)])
    stages = (("swap" + tag, [r for r in done if r["group"] in ("Bsame", "Z")], MAIN_SEED),
              ("cross" + tag, [r for r in done if r["group"] == "Bcross"], MAIN_SEED))
    for stage, members, seed in stages:
        bins = pack_runs(sorted(members, key=lambda r: (r["j"], r["group"], r["task"], r["ep"])),
                         lambda r: (r["j"], r["group"]))
        per = distribute(bins, b.order)
        for worker in b.order:
            for i, ((j, group), runs) in enumerate(per.get(worker, [])):
                b.add(f"{stage}-{worker}-j{j}-{group}-{i:02d}", stage, worker,
                      [pool_by_key[r["recv"]] for r in runs], seed=seed, ablate=ablate,
                      swap_map={r["recv"]: episode_key(r["task"], r["donor"]) for r in runs},
                      rows=[run_row(r) for r in runs])
    bins = pack_runs(seed_runs, lambda r: ("none" if r["group"] == "N" else "sp"))
    per = distribute(bins, b.order)
    for worker in b.order:
        for i, (kind, runs) in enumerate(per.get(worker, [])):
            eps = [pool_by_key[r["recv"]] for r in runs]
            name = f"seeds-{worker}-{kind}-{i:02d}"
            if kind == "none":
                b.add(name, "seeds", worker, eps, seed=SEED_CHECK, rows=[n_row(pool_by_key[r["recv"]]) for r in runs])
            else:
                b.add(name, "seeds", worker, eps, seed=SEED_CHECK, ablate=ablate,
                      swap_map={r["recv"]: episode_key(r["task"], r["donor"]) for r in runs},
                      rows=[run_row(r) for r in runs])


def cmd_donors_smoke(out):
    pool, pool_by_key = load_pool(out.parent)
    problems, files = [], 0
    for e in smoke_episodes(pool_by_key):
        rec = load_record(out / "smoke" / "record" / record_name(e["task_name"], e["episode_id"]))
        if rec is None or not e.get("segments") or e.get("es") is None:
            problems.append(f"{e['task_name']}:{e['episode_id']}:missing_record_or_scan")
            continue
        dim = next(iter(rec.values())).shape[0]
        W = smoke_w(e)
        path = out / "smoke" / DONOR_DIR / record_name(e["task_name"], synthetic_ep(e["episode_id"], 1, "Z"))
        P._savez_atomic(path, start_frames=np.asarray(W, np.int32), embs=np.zeros((len(W), dim), np.float32),
                        counts=np.ones(len(W), np.int32))
        with np.load(path) as back:
            if np.asarray(back["start_frames"]).tolist() != W or float(np.max(np.abs(back["embs"]))) != 0.0:
                problems.append(f"{e['task_name']}:{e['episode_id']}:readback")
                continue
        files += 1
    for p in problems:
        print(f"DONOR_PROBLEM {p}", flush=True)
    line = f"DONOR_SMOKE={'PASS' if not problems else 'FAIL'} files={files} problems={len(problems)}"
    print(line, flush=True)
    return 0 if not problems else 1


def cmd_donors(args):
    out = Path(args.out).resolve()
    if args.smoke:
        return cmd_donors_smoke(out)
    plan = P._read_json(out / "plan.json")
    pool, pool_by_key = load_pool(out.parent)
    infos = collect_n(plan, pool_by_key, out / "record")
    records = {k: load_record(v.get("record")) for k, v in infos.items() if v.get("record")}
    seg_lines, _ = segment_line(infos)
    for line in seg_lines:
        print(line, flush=True)
    all_runs, notes = [], {}
    for task in TASKS:
        print(recv_line(task, infos, recv_positions(infos, task, args.variant)), flush=True)
        runs, notes[task] = assign_groups(infos, pool_by_key, task, records, args.variant, args.cross_frac)
        runs, notes[task]["same_as_recv"] = drop_same_as_recv(runs, infos, records)
        all_runs += runs
    ok, line, problems, done = write_donor_files(out / DONOR_DIR, all_runs, infos, records)
    n_same_recv = sum(len(notes[t]["same_as_recv"]) for t in TASKS)
    dline, _, identical = distinct_line(done, infos, records, pool_by_key, n_same_recv)
    for p in problems:
        print(f"DONOR_PROBLEM {p}", flush=True)
    for a, b_, same_sg in identical:
        sg = {True: "same", False: "diff", None: "unknown"}[same_sg]
        print(f"DONOR_IDENTICAL {a} {b_} subgoal={sg}", flush=True)
    print(line, flush=True)
    print(dline, flush=True)
    for task in TASKS:
        cnt = {g: sum(1 for r in done if r["task"] == task and r["group"] == g) for g in SWAP_GROUPS}
        print(f"SP_GROUPS task={task} variant={args.variant} " + " ".join(f"{g}={n}" for g, n in cnt.items())
              + f" no_donor={len(notes[task]['no_donor'])} infeasible={notes[task]['infeasible']}"
              + f" same_as_recv={len(notes[task]['same_as_recv'])}", flush=True)
    for task in TASKS:
        for item in notes[task]["same_as_recv"]:
            print(f"DONOR_SAME_AS_RECV {item}", flush=True)
    seed_runs = seedcheck_runs(done, infos) if args.variant == "all" else []
    b = builder_from_plan(out, plan, pool)
    build_swap_specs(b, pool_by_key, done, seed_runs, args.variant)
    keep_stage = ("multieq", "swap", "cross", "seeds") if args.variant == "all" else ("swapcore", "crosscore")
    b.write(keep_stage)
    plan["runs"] = [r for r in plan["runs"] if r["stage"] not in keep_stage] + b.rows
    plan.setdefault("recv", {})[args.variant] = sorted({f"{r['recv']}#{r['j']}" for r in done})
    plan.setdefault("eligibility", {})
    for k, v in infos.items():
        plan["eligibility"][k] = {kk: v.get(kk) for kk in ("status", "m", "es", "segments", "W_all", "W_core",
                                                           "completed", "outcome", "segment_problem")}
    plan.setdefault("group_notes", {})[args.variant] = notes
    plan.setdefault("donor_problems", {})[args.variant] = {"count": len(problems), "integrity_ok": bool(ok),
                                                          "items": problems}
    plan.setdefault("donor_distinct", {})[args.variant] = {
        "line": dline, "same_as_recv": n_same_recv,
        "identical": [{"a": list(a), "b": list(b_), "same_subgoal": sg} for a, b_, sg in identical]}
    P._write_json(out / "plan.json", plan)
    counts = {}
    for row in b.rows:
        counts[f"{row['stage']}:{row['group']}"] = counts.get(f"{row['stage']}:{row['group']}", 0) + 1
    print("SPS_SWAP_SPECS " + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f" total={len(b.rows)} specs={len(b.specs)}", flush=True)
    print(f"DONORS={'PASS' if ok else 'FAIL'} problems={len(problems)} integrity_ok={int(ok)} runs={len(done)} "
          f"same_as_recv={n_same_recv}", flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------- 命令行
def add_root_args(p):
    for k in ASPEN:
        p.add_argument("--" + k.replace("_", "-"), dest=k, default=None, help=f"aspen 端，默认 {ASPEN[k]}")
    p.add_argument("--gl-jobs", type=int, default=0, help="GL 单卡占位数（0 = 只用 aspen）")
    p.add_argument("--gl-per-job", type=int, default=2)
    p.add_argument("--gl-cpus-per-worker", type=int, default=2)
    p.add_argument("--gl-release", default=None, help="GL 端 release 目录（= GL 仓库）")
    p.add_argument("--gl-store", default=None, help=f"默认 {GL_STORE}")
    p.add_argument("--gl-bench-src", default=None)
    p.add_argument("--gl-sim-python", default=None)
    p.add_argument("--gl-policy-python", default=None)
    p.add_argument("--gl-stage2-python", default=None)
    p.add_argument("--gl-mmevla-store", default=None)
    p.add_argument("--gl-log-dir", default=None)
    p.add_argument("--gl-checkpoint", default=None)
    p.add_argument("--gl-target-contract", default=None)
    p.add_argument("--gl-extra", default=None, help="GL spec 的额外键（JSON；缺省按路径前缀换算 --extra）")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--out", required=True)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--seed0", type=int, default=None)
    p.add_argument("--max-count", type=int, default=MAX_COUNT)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--margin", type=float, default=1.3, help="每层候选 = ceil(配额×margin)，留给核实失败")
    p.add_argument("--no-env-verify", action="store_true", help="只复算不建环境（仅调试；pool 需要核实过的 es 与段）")
    p.add_argument("--bench-src", default=ASPEN["bench_src"])
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("pool")
    p.add_argument("--out", required=True)
    p.add_argument("--bench-src", default=ASPEN["bench_src"])
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("specs")
    p.add_argument("--out", required=True, help="<O>/<M>")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--commit", required=True)
    p.add_argument("--frac", type=float, default=XLA_MEM_FRACTION)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--policy-config", default=None)
    p.add_argument("--model-kind", default=None)
    p.add_argument("--target-contract", default=None)
    p.add_argument("--extra", default=None)
    p.add_argument("--overwrite", action="store_true")
    add_root_args(p)
    p = sub.add_parser("donors")
    p.add_argument("--out", required=True, help="<O>/<M>")
    p.add_argument("--variant", choices=("all", "core"), default="all")
    p.add_argument("--cross-frac", type=float, default=CROSS_FRAC)
    p.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    return {"scan": cmd_scan, "pool": cmd_pool, "specs": cmd_specs, "donors": cmd_donors}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
