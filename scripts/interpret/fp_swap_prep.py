#!/usr/bin/env python3
"""三个交换任务「第一次拿起」motion 交换的准备脚本（计划 1109-swap-tasks-first-pick-motion-swap-plan.html
第一部分三–七节；第二部分〇′开工审查裁决 F1–F12、一、二、四节）。

任务：VideoUnmaskSwap、ButtonUnmaskSwap、VideoRepick；难度只用 easy、medium（按 seed 序号交替，显式写进自建 metadata）。

四个子命令：
  scan  --out <O> --task <T> --seed0 <S> --count 300 [--workers 6]
        须在 official-sim 解释器里跑（PYTHONPATH=<仓库>:<仓库>/src:<robomme_benchmark/src>）。按 eval 同一条建环境路径
        （examples.robomme.env_runner.EnvRunner + 自建 metadata 目录 <O>/scan_meta/<T>/）建环境：Video 类只 reset（示范
        脚本里已含交换），ButtonUnmaskSwap reset 后用「保持当前关节角 + 官方 reset 夹爪命令」走到最后一次交换结束。
        读 swap_times、swap_schedule（换成物体下标）、目标下标，按交换序列算 moved 与 swap_blind_index，Video 类另记示范长度
        es。跳过 train/test/val 官方 metadata 里出现的 seed；SceneGenerationError 跳过计数。写 <O>/scan_<T>.json，末行
        FP_SCAN=PASS|FAIL task=<T> seeds=<可用数> skipped=<场景生成失败> banned=<与官方重合跳过> errors=<其他异常>
  pool  --out <O> [--per-task 120]
        读三份 scan，每任务按（难度，swap_times，moved）分层轮转取局，局号按 env_seed 升序 0..per_task−1；写
        <O>/metadata/record_dataset_<T>_metadata.json（plfull 口径）、<O>/manifest.json、<O>/pool.json。每任务一行
        FP_POOL=PASS|FAIL task=<T> eps=<n> cells=<k> overlap=<与官方 seed 交集>，另一行 FP_POOL_LIMIT（eval.py plfull 局号上限）。
  specs --out <O>/<M> --model motion|wan --commit <sha> --run-tree <DIR> [Wan 参数同 rs_full_prep]
        生成 smoke（每任务局号 0 各 1 局 → donors --smoke → Zall 1 局/任务）、n（全部局池，录 motion）两阶段 spec、
        manifests、queues、env.sh、launch_stage.sh、plan.json；会话前缀 fps-；spec 带 bin_trace=true、stop_after_first_pick=true。
        env.sh 只设 FPS_M（本模型目录），不碰调用方变量 O（R5）。
  donors --out <O>/<M> [--wan-gate [8]] [--smoke]
        读 n 阶段终局、回执、首行轨迹与 <M>/record/*.npz：按 F1 算每局交换段窗口 W、S0 起点、W 末窗末帧；按 F3/F4 定
        接收局与供体资格；按第五节分组（Z、Rnd、S0、A×2、B×2、Zall），A/B 供体用 default_rng(0) 抽；按 F2 把全部组写成
        合成供体键 npz（<M>/donors_fp/<task>_ep<10000000+recv×100+slot>.npz），A/B 按交换段内顺序对齐并改写起点。
        打印 DONOR_FILES、DONOR_DISTINCT（改写前源编码）、FP_RECV；再生成 multieq、swap、seeds 三阶段 spec。

逐子目标位置扩展（计划 1109-subgoal-position-swap-all-lengths-plan.html〇′ F2、F5、F7、F8、F10、F11，SP-W2）：
  scan  另加 --difficulties easy,medium,hard（缺省 easy,medium，旧行为不变；按 seed 序号轮转）与 --tag（缺省：难度非缺省时
        取难度名拼接）；带 tag 时写 <O>/scan_<T>_<tag>.json，元数据目录 <O>/scan_meta/<T>-<tag>/。
        F2 补扫：VUS --seed0 2050000、BUS --seed0 2150000 各 --count 150 --difficulties hard。
  pool  --by-swap-times [--per-n 30]：读 <O>/scan_<T>.json 与全部 <O>/scan_<T>_*.json，每任务按 swap_times n=1/2/3 各取
        per-n 局（SWAP_TIMES_SOURCES：VR 全部取 easy/medium；VUS/BUS 的 n=1、2 取 easy/medium，n=3 取 hard），格内按
        （难度，moved）轮转。判定行 FP_POOL 另带 bins=n1:..,n2:..,n3:..。不加该开关时与旧模式完全相同（只读 scan_<T>.json）。
  specs 路径根参数（F5）：--main、--store、--bench-src、--policy-python、--sim-python、--stage2-python、--mmevla-store、
        --log-dir、--release（运行树是 git archive 解出的 release 目录，env.sh 不查 git，改查目录名与 commit 前缀一致）、
        --workers 名=GPU,…；预设 --gl-store <GLSTORE> 一次设好 store、三个解释器（<GLSTORE>/envs/{policy,official-sim,
        stage2}/bin/python）、MMEVLA_V1_STORE、--release、16 个 worker gl00..gl15（GPU 0），bench_src 缺省取
        <run-tree>/third_party/robomme_benchmark/src。spec 里以旧 STORE/MAIN 开头的路径整体改根。路径根与 worker 表写进
        plan.json，donors 沿用。全部缺省时与旧行为逐字节一致。
  donors --by-k：按第 k 次交换的窗口 W_k（交换帧区间 [start_k+1, end_k+1]，与 segment_starts 窗口真实帧闭区间相交，F8）
        给每个 (接收局, k≤n) 生成 Bsame_k（同 n 另一局的 W_k，default_rng(0) 每 (任务,k) 一个，按 W_k 内顺序对齐、沿用补齐
        规则 F7）与 Z_k（零）；合成键 10000000+recv×100+slot，slot=20+2k+组（Bsame=0、Z=1）。写 kswap（seed 42）、
        kseeds（每任务 default_rng(0) 抽 KSEED_PER_TASK 局，seed 1）两阶段 spec；同 (k,组) 的接收局装进同一 spec（F11）。
        打印 FP_K_GROUPS、DONOR_FILES；不动旧组的行。

trace 与回执字段（FP-W1 提供）只在 receipt_fields、trace_row_fields 两个函数里读，合并后按实际字段名在这里改。
窗口与帧号（F1）：帧 k = 第 k 步之后的观测（帧 0 为 reset）；示范窗起点 s=0,32,… 满足 s+16 ≤ es−1，真实帧
[s, min(s+32, es−1)]；执行窗起点 es+32u，真实帧 [f, f+32]，最多到 es+MAX_STEPS；交换帧区间 [第一次 start+1, 最后 end+1]
（再加帧偏移：Video 示范帧号 = elapsed，偏移 0；Button 帧号 = es + elapsed − initial_env_step）。Video 类 W 只取示范窗，
Button 只取执行窗（第一部分第三节）。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import shlex
import sys

import numpy as np

from scripts.evaluation import manifest as M
from scripts.interpret import pl_full_prep as P
from scripts.interpret.pl_swap_prep import BENCH_SRC, SPEC_TEMPLATE, STORE, XLA_MEM_FRACTION

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- 常量
TASKS = ("VideoUnmaskSwap", "ButtonUnmaskSwap", "VideoRepick")
VIDEO_TASKS = ("VideoUnmaskSwap", "VideoRepick")
BUTTON_TASKS = ("ButtonUnmaskSwap",)
SEED0 = {"VideoUnmaskSwap": 2000000, "ButtonUnmaskSwap": 2100000, "VideoRepick": 2200000}   # R6
SEED_STEP = 100
SCAN_COUNT = 300
DIFFICULTIES = ("easy", "medium")           # 按 seed 序号交替（F12）
OBJECT_ATTR = {"VideoUnmaskSwap": "spawned_bins", "ButtonUnmaskSwap": "spawned_bins", "VideoRepick": "spawned_cubes"}
PICK_SUBGOAL_OFFSET = {"ButtonUnmaskSwap": 2, "VideoUnmaskSwap": 0, "VideoRepick": 0}   # 先按两个按钮
MOTION_WINDOW = 33
MOTION_STRIDE = 32
DEMO_MIN_REAL = 17                          # 两模型配置 demo_min_real_frames: 17、demo_tail_pad: repeat_last
MAX_STEPS = 1300                            # eval plfull 固定 1300
PER_TASK = 120
SYN_BASE = 10000000                         # F2：ep = SYN_BASE + recv×100 + slot
SLOTS = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "S0": 5, "Z": 6, "Rnd": 7, "Zall": 8}
SLOT_GROUP = {"A1": "A", "A2": "A", "B1": "B", "B2": "B", "S0": "S0", "Z": "Z", "Rnd": "Rnd", "Zall": "Zall"}
GROUPS = ("N", "Z", "Rnd", "S0", "A", "B", "Zall")
SWAP_GROUPS = ("Z", "Rnd", "S0", "A", "B", "Zall")
GATE_GROUPS = ("Z", "S0", "Zall")           # Wan 门槛触发时只跑 N、Z、S0、Zall
WAN_GATE_DEFAULT = 8
PAIR_SEED = 0
MAIN_SEED = 42
SEEDCHECK_PER_GROUP = 5
SEEDCHECK_SEEDS = (1, 2)
MULTIEQ_GROUPS = ("S0", "A", "B", "Z")
MULTIEQ_PER_GROUP = 3
DONOR_DIR = "donors_fp"
BASE_STAGES = ("smoke", "n", "multieq", "swap", "seeds")
K_STAGES = ("kswap", "kseeds")
STAGES = BASE_STAGES + K_STAGES
# ---- 按第 k 次交换（SP-W2）
K_GROUPS = ("Bsame_k", "Z_k")
K_GROUP_CODE = {"Bsame_k": 0, "Z_k": 1}
K_SLOT_BASE = 20                            # slot = 20 + 2k + 组号；旧 slot 1–8 不冲突
K_MAX = 3
K_SLOTS = {f"{g}{k}": K_SLOT_BASE + 2 * k + c for k in range(1, K_MAX + 1) for g, c in K_GROUP_CODE.items()}
KSEED_PER_TASK = 20
KSEED_SEED = 1
PER_N = 30
SWAP_TIMES_NS = (1, 2, 3)
SWAP_TIMES_SOURCES = {                      # 〇′ F2：VR 不用 hard；VUS/BUS 的 n=3 只在 hard
    "VideoRepick": {1: ("easy", "medium"), 2: ("easy", "medium"), 3: ("easy", "medium")},
    "VideoUnmaskSwap": {1: ("easy", "medium"), 2: ("easy", "medium"), 3: ("hard",)},
    "ButtonUnmaskSwap": {1: ("easy", "medium"), 2: ("easy", "medium"), 3: ("hard",)},
}
GL_WORKERS = tuple(f"gl{i:02d}" for i in range(16))   # 8 个占位 × 每卡 2 worker；srun 内可见 GPU 为 0
RUN_TREE = STORE / "worktrees/fp-swap-run"
LOG_DIR = STORE / "logs/fp-swap-1110"
OFFICIAL_META = BENCH_SRC / "robomme/env_metadata"
EVAL_PY = REPO / "examples/robomme/eval.py"

MODELS = {
    "motion": dict(P.MODELS["motion"]),
    "wan": {"checkpoint": None, "policy_config": None, "model_kind": None, "target_contract": None},
}


def meta_name(task):
    return f"record_dataset_{task}_metadata.json"


def episode_key(task, ep):
    return f"{task}:{int(ep)}"


def record_name(task, ep):
    return f"{task}_ep{int(ep)}.npz"


def split_key(key):
    task, _, ep = str(key).rpartition(":")
    return task, int(ep)


def synthetic_ep(recv_ep, slot):
    code = SLOTS[slot] if slot in SLOTS else K_SLOTS[slot]
    return SYN_BASE + int(recv_ep) * 100 + code


def k_slot(group, k):
    return f"{group}{int(k)}"


# ---------------------------------------------------------------- 交换序列组合
def swap_outcome(n_objects, target_index, schedule):
    """按交换序列组合槽位：slot[i] 为物体 i 当前所在槽位（初始槽位 = 下标）；每次交换 (a,b) 两物体互换槽位
    （swap_flat_two_lane 交换两物体位置）。返回 (moved, swap_blind_index)：moved = 目标最终槽位 ≠ 原槽位；
    swap_blind_index = 结束时占着目标原槽位的物体下标。"""
    slot = list(range(int(n_objects)))
    for item in schedule:
        a, b = int(item[0]), int(item[1])
        if not (0 <= a < n_objects and 0 <= b < n_objects) or a == b:
            raise ValueError(f"交换对下标非法: {item}（物体数 {n_objects}）")
        slot[a], slot[b] = slot[b], slot[a]
    t = int(target_index)
    blind = slot.index(t)
    return slot[t] != t, blind


def ordered_schedule(schedule):
    """交换按时间先后（start、end）排序；第 k 次交换 = 排序后第 k 条（1 起）。"""
    return sorted((list(x) for x in schedule), key=lambda x: (int(x[2]), int(x[3])))


def _slots_after(n_objects, schedule):
    slot = list(range(int(n_objects)))
    for item in schedule:
        a, b = int(item[0]), int(item[1])
        if not (0 <= a < n_objects and 0 <= b < n_objects) or a == b:
            raise ValueError(f"交换对下标非法: {item}（物体数 {n_objects}）")
        slot[a], slot[b] = slot[b], slot[a]
    return slot


def swap_blind_k(n_objects, target_index, schedule, k):
    """若第 k 次交换没发生（其余交换照常）时目标所在的物体：按「漏掉第 k 次」的序列组合出目标的信念槽位，
    返回实际（全部交换后）占着该槽位的物体下标。漏掉全部交换时退化为 swap_blind_index。"""
    sched = ordered_schedule(schedule)
    k = int(k)
    if not 1 <= k <= len(sched):
        raise ValueError(f"k={k} 超出交换次数 {len(sched)}")
    actual = _slots_after(n_objects, sched)
    belief = _slots_after(n_objects, sched[:k - 1] + sched[k:])[int(target_index)]
    return actual.index(belief)


def classify_pick_k(first_pick, target_index, blind_k):
    """correct / swap_blind_k（blind_k ≠ 目标时才可能）/ other / none。"""
    if first_pick is None:
        return "none"
    idx = int(first_pick["index"])
    if idx == int(target_index):
        return "correct"
    if blind_k is not None and int(blind_k) != int(target_index) and idx == int(blind_k):
        return "swap_blind_k"
    return "other"


# ---------------------------------------------------------------- 窗口（F1）
def demo_starts(es):
    out, s = [], 0
    while s + (DEMO_MIN_REAL - 1) <= int(es) - 1:
        out.append(s)
        s += MOTION_STRIDE
    return out


def exec_starts(es, max_steps=MAX_STEPS):
    """执行窗起点 es+32u：33 帧到齐（f+32 ≤ es+max_steps，帧 es+k 为第 k 步之后的观测）。"""
    out, u = [], 0
    while u + (MOTION_WINDOW - 1) <= int(max_steps):
        out.append(int(es) + u)
        u += MOTION_STRIDE
    return out


def all_starts(es, max_steps=MAX_STEPS):
    return demo_starts(es) + exec_starts(es, max_steps)


def real_range(f, es):
    """起点 f 的真实帧范围（含两端）；示范窗尾部补帧不算真实帧。"""
    f, es = int(f), int(es)
    if f < es:
        return f, min(f + MOTION_WINDOW - 1, es - 1)
    return f, f + MOTION_WINDOW - 1


def frame_offset(task, es, initial_env_step):
    """交换步号 → 全域帧号的偏移：Video 示范帧号 = elapsed（0）；Button 执行帧 es+k 对应 elapsed = initial+k。"""
    if task in BUTTON_TASKS and initial_env_step is not None:
        return int(es) - int(initial_env_step)
    return 0


def swap_frames(schedule, offset=0):
    """交换帧区间 [第一次 start+1, 最后 end+1]（闭区间，F1）。"""
    if not schedule:
        return None
    return min(int(x[2]) for x in schedule) + 1 + offset, max(int(x[3]) for x in schedule) + 1 + offset


def segment_starts(task, es, max_steps=MAX_STEPS):
    """W 的候选起点：Video 只取示范窗，Button 只取执行窗。"""
    return demo_starts(es) if task in VIDEO_TASKS else exec_starts(es, max_steps)


def swap_windows(task, es, schedule, offset=0, max_steps=MAX_STEPS):
    """W = 真实帧范围与交换帧区间相交的起点（升序）。"""
    span = swap_frames(schedule, offset)
    if span is None:
        return []
    lo, hi = span
    out = []
    for f in segment_starts(task, es, max_steps):
        a, b = real_range(f, es)
        if a <= hi and b >= lo:
            out.append(f)
    return out


def swap_frames_k(schedule, k, offset=0):
    """第 k 次交换的帧区间 [start_k+1, end_k+1]（闭区间）。"""
    sched = ordered_schedule(schedule)
    if not 1 <= int(k) <= len(sched):
        return None
    item = sched[int(k) - 1]
    return int(item[2]) + 1 + offset, int(item[3]) + 1 + offset


def windows_in_span(task, es, span, max_steps=MAX_STEPS):
    """segment_starts 中真实帧闭区间与 span（闭区间）相交的起点（升序，〇′ F8）。"""
    if span is None:
        return []
    lo, hi = span
    return [f for f in segment_starts(task, es, max_steps) if real_range(f, es)[0] <= hi and real_range(f, es)[1] >= lo]


def swap_windows_k(task, es, schedule, k, offset=0, max_steps=MAX_STEPS):
    """W_k：第 k 次交换帧区间相交的窗口起点。"""
    return windows_in_span(task, es, swap_frames_k(schedule, k, offset), max_steps)


def s0_start(task, es, schedule, offset=0, max_steps=MAX_STEPS):
    """S0：真实帧范围全在第一次交换 start（帧号）及以前的最后一个起点；没有返回 None。"""
    if not schedule:
        return None
    first = min(int(x[2]) for x in schedule) + offset
    cands = [f for f in segment_starts(task, es, max_steps) if real_range(f, es)[1] <= first]
    return max(cands) if cands else None


def last_window_end(W, es):
    return real_range(max(W), es)[1] if W else None


def next_start_after(f, es, max_steps=MAX_STEPS):
    """供体补齐用：交换段之后的第一个窗起点（示范之后接执行段）。"""
    later = [s for s in all_starts(es, max_steps) if s > int(f)]
    return min(later) if later else None


# ---------------------------------------------------------------- 回执与轨迹（FP-W1 字段集中在这里）
def _flag(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return bool(any(_flag(v) for v in value))
    return bool(value)


def receipt_fields(receipt):
    """hold_receipt.json → 规范化 dict。FP-W1（3a51940）实际字段：task_name、episode_id、env_seed、difficulty、
    target_index（int）、swap_times、swap_schedule=[[a,b,start,end],…]（a/b 未确定为 null）、
    first_pick_subgoal_index（第一次拿起子目标下标：VideoUnmaskSwap=1、ButtonUnmaskSwap=2、VideoRepick 按名字找）、
    stop_after_first_pick、truncated_first_pick、first_pick = null 或 {index, env_step（步后 elapsed）, action_index,
    is_target, subgoal_index_at_pick（步前 timestep）, subgoal_index_after_pick, successflag, failureflag, outcome}；
    沿用旧回执的 exec_start_idx（es）与 initial_env_step。schedule_complete=False 表示有未确定的交换对。"""
    target = receipt.get("target_index")
    if target is None:
        target = receipt.get("first_target_bin_index")
    fp = receipt.get("first_pick")
    first_pick = None
    if fp:
        sub = fp.get("subgoal_index_at_pick")
        first_pick = {"index": int(fp["index"]), "env_step": int(fp["env_step"]), "is_target": bool(fp["is_target"]),
                      "action_index": fp.get("action_index"), "subgoal_index_at_pick": None if sub is None else int(sub),
                      "successflag": _flag(fp.get("successflag")), "failureflag": _flag(fp.get("failureflag")),
                      "outcome": fp.get("outcome")}
    es = receipt.get("exec_start_idx")
    raw = receipt.get("swap_schedule") or []
    complete = all(item is not None and all(v is not None for v in item) for item in raw)
    schedule = [[int(v) for v in item] for item in raw] if complete else [list(item or []) for item in raw]
    return {"task": receipt.get("task_name"), "episode_id": receipt.get("episode_id"), "env_seed": receipt.get("env_seed"),
            "difficulty": receipt.get("difficulty"), "target_index": None if target is None else int(target),
            "swap_times": None if receipt.get("swap_times") is None else int(receipt["swap_times"]),
            "schedule": schedule, "schedule_complete": complete and bool(raw),
            "es": None if es is None else int(es), "initial_env_step": receipt.get("initial_env_step"),
            "first_pick": first_pick, "first_pick_subgoal_index": receipt.get("first_pick_subgoal_index"),
            "truncated_first_pick": receipt.get("truncated_first_pick")}


def trace_row_fields(row):
    """step_trace.jsonl 一行 → env_step（步后）、objects、subgoal_index、action_index。W1 实际行字段为 env_step_after
    （步后，优先取 env_step 键以防改名）、objects（步后，与 bins_after 同值；非扩展模式缺失时回退 bins_after）、
    subgoal_index；拿起那一行另带 first_pick（与回执同，本脚本以回执为准）。"""
    env_step = row.get("env_step", row.get("env_step_after"))
    objects = row.get("objects", row.get("bins_after"))
    return {"env_step": None if env_step is None else int(env_step), "objects": objects,
            "subgoal_index": None if row.get("subgoal_index") is None else int(row["subgoal_index"]),
            "action_index": row.get("action_index"), "exec_start_idx": row.get("exec_start_idx")}


def read_step_rows(path, limit=None):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event", "step") == "step":
                rows.append(row)
                if limit and len(rows) >= limit:
                    break
    return rows


def find_run(run_root, attn_out, key):
    """一局在 run_root 下的最新终局、回执、轨迹路径、episode.json 的 motion_content。缺什么对应值为 None。"""
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
        receipt_path = traces[0].parent / "meta" / "hold_receipt.json"
        if receipt_path.is_file():
            out["receipt"] = P._read_json(receipt_path)
    episode = Path(attn_out) / nonce / "episode.json"
    if episode.is_file():
        out["motion_content"] = P._read_json(episode).get("motion_content")
    return out


def expected_pick_subgoal(task, rf, first_row):
    """第一次拿起子目标的下标：回执给出则用之；否则 = 执行段首行子目标下标 + 偏移（Button 先按两个按钮 +2）。"""
    if rf.get("first_pick_subgoal_index") is not None:
        return int(rf["first_pick_subgoal_index"])
    if first_row is None or first_row["subgoal_index"] is None:
        return None
    return first_row["subgoal_index"] + PICK_SUBGOAL_OFFSET[task]


def episode_labels(task, rf, n_objects=None):
    """回执 → 交换标签与窗口：moved、swap_blind_index、W、S0、W 末窗末帧、帧偏移。"""
    es = rf["es"] if rf["es"] is not None else 0
    offset = frame_offset(task, es, rf["initial_env_step"])
    sched = rf["schedule"]
    n_obj = n_objects if n_objects is not None else (max([max(x[0], x[1]) for x in sched] + [rf["target_index"]]) + 1)
    moved, blind = swap_outcome(n_obj, rf["target_index"], sched)
    W = swap_windows(task, es, sched, offset)
    ks = range(1, len(sched) + 1)
    return {"moved": bool(moved), "swap_blind_index": int(blind), "W": W, "S0": s0_start(task, es, sched, offset),
            "w_end": last_window_end(W, es), "offset": offset, "es": es, "n_objects": int(n_obj),
            "Wk": {k: swap_windows_k(task, es, sched, k, offset) for k in ks},
            "blind_k": {k: swap_blind_k(n_obj, rf["target_index"], sched, k) for k in ks}}


def classify_pick(first_pick, target_index, swap_blind_index, moved):
    """correct / swap_blind（只对 moved 局有意义）/ other / none。"""
    if first_pick is None:
        return "none"
    idx = int(first_pick["index"])
    if idx == int(target_index):
        return "correct"
    if moved and idx == int(swap_blind_index):
        return "swap_blind"
    return "other"


def pick_status(task, rf, labels, first_row):
    """返回 dict：on_turn（拿起时子目标已轮到第一次拿起）、before_buttons、f3_ok、pick_frame。"""
    fp = rf["first_pick"]
    expected = expected_pick_subgoal(task, rf, first_row)
    out = {"expected_subgoal": expected, "on_turn": None, "before_buttons": False, "f3_ok": True, "pick_frame": None}
    if fp is not None:
        sub = fp["subgoal_index_at_pick"]
        out["on_turn"] = expected is not None and sub is not None and sub == expected
        out["before_buttons"] = task in BUTTON_TASKS and expected is not None and sub is not None and sub < expected
        if task in BUTTON_TASKS:
            initial = rf["initial_env_step"] if rf["initial_env_step"] is not None else 0
            out["pick_frame"] = labels["es"] + int(fp["env_step"]) - int(initial)
            out["f3_ok"] = labels["w_end"] is not None and out["pick_frame"] >= labels["w_end"]
    return out


# ---------------------------------------------------------------- scan（official-sim）
def scan_seeds(task, seed0, count, difficulties=DIFFICULTIES):
    diffs = tuple(difficulties)
    return [(i, int(seed0) + SEED_STEP * i, diffs[i % len(diffs)]) for i in range(int(count))]


def parse_difficulties(text):
    diffs = tuple(d.strip() for d in str(text).split(",") if d.strip())
    bad = [d for d in diffs if d not in ("easy", "medium", "hard")]
    if not diffs or bad:
        raise SystemExit(f"--difficulties 只能取 easy/medium/hard 的逗号列表：{text}")
    return diffs


def scan_tag(difficulties, tag=None):
    """扫描输出后缀：显式 tag 优先；难度为缺省 easy,medium 时无后缀（旧文件名）；否则难度名拼接。"""
    if tag:
        return str(tag)
    return None if tuple(difficulties) == DIFFICULTIES else "-".join(difficulties)


def scan_file(out, task, tag=None):
    return Path(out) / (f"scan_{task}.json" if not tag else f"scan_{task}_{tag}.json")


def split_metadata_seeds(task, official_meta=None):
    seeds = set()
    root = Path(official_meta) if official_meta else OFFICIAL_META
    for ds in ("train", "test", "val"):
        path = root / ds / meta_name(task)
        if path.is_file():
            seeds |= {int(r["seed"]) for r in json.loads(path.read_text())["records"]}
    return seeds


def scan_metadata(task, items):
    records = [{"task": task, "episode": int(i), "seed": int(seed), "difficulty": diff} for i, seed, diff in items]
    return {"env_id": task, "record_count": len(records), "records": records}


def env_labels(task, u):
    """reset（及 Button 走完交换）后从环境读交换标签：物体按身份换成下标。"""
    objects = list(getattr(u, OBJECT_ATTR[task]))
    target = u.target_cube_1 if task == "VideoRepick" else u.selected_bins[0]
    index = {id(obj): i for i, obj in enumerate(objects)}
    if id(target) not in index:
        raise RuntimeError("目标物体不在物体列表里")
    schedule = []
    for item in u.swap_schedule:
        a, b, start, end = item
        if a is None or b is None or id(a) not in index or id(b) not in index:
            raise RuntimeError(f"交换对未确定或不在物体列表: {item}")
        schedule.append([index[id(a)], index[id(b)], int(start), int(end)])
    swap_times = int(u.swap_times)
    if len(schedule) != swap_times:
        raise RuntimeError(f"swap_schedule 条数 {len(schedule)} ≠ swap_times {swap_times}")
    return {"n_objects": len(objects), "target_index": index[id(target)], "swap_times": swap_times, "swap_schedule": schedule}


def _elapsed(u):
    value = u.elapsed_steps
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    return int(value)


def _hold_action(u):
    from robomme.robomme_env.utils import reset_panda
    q = u.agent.robot.get_qpos()
    q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
    q = np.asarray(q, np.float64).reshape(-1)
    gripper = float(reset_panda.get_reset_panda_param("action")[-1])
    return np.asarray(list(q[:7]) + [gripper], dtype=np.float64)


def scan_one(task, ep, seed, difficulty, meta_dir, video_dir):
    """一个 seed：按 eval 同一路径建环境，返回扫描条目（status ok / skipped / error）。"""
    from examples.robomme.env_runner import EnvRunner
    from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError
    entry = {"scan_index": int(ep), "env_seed": int(seed), "difficulty": difficulty}
    runner = None
    try:
        runner = EnvRunner(task, str(video_dir), max_steps=MAX_STEPS, dataset="test", difficulty=difficulty,
                           metadata_dir=str(meta_dir))
        runner.make_env(int(ep))
        result = runner.get_init_obs()
        u = runner.env.unwrapped
        es = len(result["images"]) - 1
        steps = 0
        if task in BUTTON_TASKS:
            hold = _hold_action(u)
            end = max(int(x[3]) for x in u.swap_schedule)
            while _elapsed(u) < end:
                _, stop, outcome = runner.step(hold)
                steps += 1
                if stop:
                    raise RuntimeError(f"保持不动走到第 {steps} 步环境终止: {outcome}")
                if steps > end + 10:
                    raise RuntimeError("保持不动超过交换结束步仍未到达")
        labels = env_labels(task, u)
        moved, blind = swap_outcome(labels["n_objects"], labels["target_index"], labels["swap_schedule"])
        entry.update(labels, status="ok", moved=bool(moved), swap_blind_index=int(blind),
                     es=int(es) if task in VIDEO_TASKS else 0, hold_steps=steps, actual_difficulty=str(u.difficulty))
        if entry["actual_difficulty"] != difficulty:
            raise RuntimeError(f"难度未生效：{entry['actual_difficulty']} ≠ {difficulty}")
    except SceneGenerationError as error:
        entry.update(status="skipped", reason=f"SceneGenerationError: {error}")
    except Exception as error:   # 记原因、继续下一个 seed；FP_SCAN 以 errors 计
        entry.update(status="error", reason=f"{type(error).__name__}: {error}")
    finally:
        if runner is not None and getattr(runner, "env", None) is not None:
            try:
                runner.env.close()
            except Exception:
                pass
    return entry


def _scan_chunk(args):
    task, items, meta_dir, video_dir = args
    return [scan_one(task, ep, seed, diff, meta_dir, video_dir) for ep, seed, diff in items]


def scan_verdict(task, entries):
    ok = [e for e in entries if e["status"] == "ok"]
    skipped = sum(1 for e in entries if e["status"] == "skipped")
    banned = sum(1 for e in entries if e["status"] == "banned")
    errors = sum(1 for e in entries if e["status"] == "error")
    bad = [e for e in ok if e["swap_times"] != len(e["swap_schedule"]) or not 0 <= e["target_index"] < e["n_objects"]]
    verdict = "PASS" if errors == 0 and not bad and ok else "FAIL"
    return f"FP_SCAN={verdict} task={task} seeds={len(ok)} skipped={skipped} banned={banned} errors={errors + len(bad)}"


def cmd_scan(args):
    out = Path(args.out).resolve()
    task = args.task
    diffs = parse_difficulties(args.difficulties)
    tag = scan_tag(diffs, args.tag)
    items = scan_seeds(task, args.seed0, args.count, diffs)
    banned = split_metadata_seeds(task, official_meta(args))
    keep = [it for it in items if it[1] not in banned]
    sub = task if not tag else f"{task}-{tag}"
    meta_dir = out / "scan_meta" / sub
    P._write_once(meta_dir / meta_name(task), P._json_bytes(scan_metadata(task, keep)), args.overwrite)
    video_dir = out / "scan_tmp" / sub
    video_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers))
    chunks = [keep[i::workers] for i in range(workers)]
    entries = [{"scan_index": i, "env_seed": s, "difficulty": d, "status": "banned"} for i, s, d in items if s in banned]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for part in pool.map(_scan_chunk, [(task, c, str(meta_dir), str(video_dir)) for c in chunks if c]):
            entries.extend(part)
            print(f"SCAN_PART task={task} done={len(entries)}/{len(items)}", flush=True)
    entries.sort(key=lambda e: e["scan_index"])
    for e in entries:
        if e["status"] == "error":
            print(f"SCAN_ERROR seed={e['env_seed']} {e['reason']}", flush=True)
    line = scan_verdict(task, entries)
    payload = {"task": task, "seed0": int(args.seed0), "step": SEED_STEP, "count": int(args.count),
               "difficulties": list(diffs), "metadata_dir": str(meta_dir), "verdict": line, "entries": entries}
    if tag:
        payload["tag"] = tag
    P._write_json(scan_file(out, task, tag), payload)
    print(line, flush=True)
    return 0 if line.startswith("FP_SCAN=PASS") else 1


# ---------------------------------------------------------------- pool
def cell_of(e):
    return (e["difficulty"], int(e["swap_times"]), bool(e["moved"]))


def stratified_pick(entries, per_task):
    """按（难度，swap_times，moved）分格，格内按 env_seed 升序；格按键排序后轮转逐个取，直到取满。"""
    cells = {}
    for e in sorted(entries, key=lambda x: x["env_seed"]):
        cells.setdefault(cell_of(e), []).append(e)
    order = sorted(cells)
    picked, pos = [], {k: 0 for k in order}
    while len(picked) < per_task:
        progressed = False
        for k in order:
            if len(picked) >= per_task:
                break
            if pos[k] < len(cells[k]):
                picked.append(cells[k][pos[k]])
                pos[k] += 1
                progressed = True
        if not progressed:
            break
    counts = {f"{k[0]}|n{k[1]}|{'moved' if k[2] else 'stay'}": pos[k] for k in order}
    avail = {f"{k[0]}|n{k[1]}|{'moved' if k[2] else 'stay'}": len(cells[k]) for k in order}
    return sorted(picked, key=lambda e: e["env_seed"]), counts, avail


def official_meta(args):
    bench = getattr(args, "bench_src", None)
    return Path(bench) / "robomme/env_metadata" if bench else None


def load_scan_entries(out, task, all_scans=False):
    """读 <O>/scan_<T>.json（all_scans 时另读全部 scan_<T>_*.json）。返回 (全部条目, 可用条目, 文件名列表)。
    同一 env_seed 出现在多份扫描里：内容相同只留一份，不同则报错（避免两份难度口径混进同一局）。"""
    out = Path(out)
    files = [scan_file(out, task)] if scan_file(out, task).is_file() or not all_scans else []
    if all_scans:
        files += sorted(out.glob(f"scan_{task}_*.json"))
    if not files:
        raise SystemExit(f"{out} 下没有 {task} 的扫描文件")
    entries, usable, seen = [], [], {}
    for path in files:
        for e in P._read_json(path)["entries"]:
            entries.append(e)
            if e["status"] != "ok":
                continue
            prev = seen.get(int(e["env_seed"]))
            if prev is not None:
                if {k: v for k, v in prev.items() if k != "scan_index"} != {k: v for k, v in e.items() if k != "scan_index"}:
                    raise SystemExit(f"{task} env_seed={e['env_seed']} 在多份扫描里内容不同")
                continue
            seen[int(e["env_seed"])] = e
            usable.append(e)
    return entries, usable, [p.name for p in files]


def by_swap_times_pick(task, entries, per_n=PER_N, sources=None):
    """每个 n ∈ {1,2,3} 只从 SWAP_TIMES_SOURCES[task][n] 的难度里取 per_n 局，格内（难度，n，moved）轮转。
    返回 (picked 按 env_seed 升序, counts 格→取数, avail 格→可用数, bins n→取数)。"""
    sources = (sources or SWAP_TIMES_SOURCES)[task]
    picked, counts, avail, bins = [], {}, {}, {}
    for n in SWAP_TIMES_NS:
        mine = [e for e in entries if int(e["swap_times"]) == n and e["difficulty"] in sources[n]]
        got, c, a = stratified_pick(mine, per_n)
        picked += got
        counts.update(c)
        avail.update(a)
        bins[n] = len(got)
    return sorted(picked, key=lambda e: e["env_seed"]), counts, avail, bins


def pool_episodes(task, picked):
    eps = []
    for i, e in enumerate(picked):
        eps.append({"task_name": task, "episode_id": i, "env_seed": int(e["env_seed"]), "difficulty": e["difficulty"],
                    "swap_times": int(e["swap_times"]), "moved": bool(e["moved"]), "swap_schedule": e["swap_schedule"],
                    "target_index": int(e["target_index"]), "swap_blind_index": int(e["swap_blind_index"]),
                    "n_objects": int(e["n_objects"]), "es": int(e.get("es", 0)), "scan_index": int(e["scan_index"])})
    return eps


def metadata_payload(task, episodes):
    records = [{"task": task, "episode": int(e["episode_id"]), "seed": int(e["env_seed"]), "difficulty": e["difficulty"]}
               for e in episodes]
    return {"env_id": task, "record_count": len(records), "records": records}


def manifest_for(episodes, meta_shas, chunk=P.CHUNK):
    rows = [{"task_name": e["task_name"], "episode_id": int(e["episode_id"]), "env_seed": int(e["env_seed"]),
             "difficulty": e["difficulty"]} for e in episodes]
    return M.make_manifest(rows, metadata_sha256=dict(meta_shas), chunk_size=chunk, official=False)


def eval_episode_limit(path=EVAL_PY):
    """examples/robomme/eval.py 的 PLFULL_EPISODES（plfull 口径局号上限）；读不到返回 None。"""
    try:
        m = re.search(r"^PLFULL_EPISODES\s*=\s*(\d+)", Path(path).read_text(), re.M)
    except OSError:
        return None
    return int(m.group(1)) if m else None


def cmd_pool(args):
    out = Path(args.out).resolve()
    by_n = bool(getattr(args, "by_swap_times", False))
    per_n = int(getattr(args, "per_n", PER_N))
    per_task = per_n * len(SWAP_TIMES_NS) if by_n else int(args.per_task)
    all_eps, tasks_info, lines, ok_all = [], {}, [], True
    meta_shas = {}
    for task in TASKS:
        if by_n:
            scanned, usable, files = load_scan_entries(out, task, all_scans=True)
            picked, counts, avail, bins = by_swap_times_pick(task, usable, per_n)
        else:
            scan = P._read_json(out / f"scan_{task}.json")
            scanned, files, bins = scan["entries"], [f"scan_{task}.json"], None
            usable = [e for e in scan["entries"] if e["status"] == "ok"]
            picked, counts, avail = stratified_pick(usable, per_task)
        eps = pool_episodes(task, picked)
        banned = split_metadata_seeds(task, official_meta(args))
        overlap = sorted({e["env_seed"] for e in eps} & banned)
        meta_bytes = P._json_bytes(metadata_payload(task, eps))
        meta_shas[meta_name(task)] = hashlib.sha256(meta_bytes).hexdigest()
        P._write_once(out / "metadata" / meta_name(task), meta_bytes, args.overwrite)
        ok = len(eps) == per_task and not overlap and (bins is None or all(v == per_n for v in bins.values()))
        ok_all &= ok
        tasks_info[task] = {"scanned": len(scanned), "usable": len(usable), "cells_taken": counts,
                            "cells_available": avail, "overlap": overlap}
        line = f"FP_POOL={'PASS' if ok else 'FAIL'} task={task} eps={len(eps)} cells={len(counts)} overlap={len(overlap)}"
        if by_n:
            tasks_info[task].update(scan_files=files, bins={str(k): v for k, v in bins.items()})
            line += " bins=" + ",".join(f"n{k}:{v}" for k, v in bins.items())
        lines.append(line)
        all_eps.extend(eps)
    limit = eval_episode_limit()
    limit_ok = limit is not None and per_task <= limit
    lines.append(f"FP_POOL_LIMIT={'PASS' if limit_ok else 'FAIL'} eval_plfull_episodes={limit} per_task={per_task}")
    manifest = manifest_for(all_eps, meta_shas)
    P._write_once(out / "manifest.json", P._json_bytes(manifest), args.overwrite)
    pool = {"tasks": tasks_info, "per_task": per_task, "metadata_sha256": meta_shas, "manifest_seal": manifest["sha256"],
            "numbering": "每任务局号按 env_seed 升序 0..per_task−1", "episodes": all_eps}
    if by_n:
        pool.update(mode="by_swap_times", per_n=per_n,
                    sources={t: {str(n): list(d) for n, d in v.items()} for t, v in SWAP_TIMES_SOURCES.items()})
    P._write_once(out / "pool.json", P._json_bytes(pool), args.overwrite)
    for line in lines:
        print(line, flush=True)
    return 0 if ok_all and limit_ok else 1


def load_pool(pool_root):
    pool = P._read_json(Path(pool_root) / "pool.json")
    by_key = {episode_key(e["task_name"], e["episode_id"]): e for e in pool["episodes"]}
    return pool, by_key


# ---------------------------------------------------------------- specs
QUEUE_HEAD = P.QUEUE_HEAD.replace("pl_full_prep.py", "fp_swap_prep.py")
QUEUE_TAIL = P.QUEUE_TAIL
assert "\nO={out}\n" in P.ENV_TEMPLATE
# R5：env.sh 被调用方 source，不得覆盖调用方的 O；本模型目录改叫 FPS_M。
ENV_TEMPLATE = P.ENV_TEMPLATE.replace("pl_full_prep.py", "fp_swap_prep.py").replace("\nO={out}\n", "\nFPS_M={out}\n")

LAUNCH_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/interpret/fp_swap_prep.py specs 生成。用法：bash launch_stage.sh <{stages}>
# 为该阶段有 spec 的 worker 各起一个 detached tmux（fps-{model}-<阶段>-<worker>），日志 {logs}/fps-{model}-<阶段>-<worker>.log。
# 顺序：smoke → n → （fp_swap_prep donors）→ multieq → swap → seeds；后三阶段要先有 donors_fp/。
FPS_M={out}
st=$1
case "$st" in {stages}) ;; *) echo "未知阶段 $st"; exit 2;; esac
mkdir -p {logs}
for w in {workers}; do
  q=$FPS_M/queues/$st-$w.sh
  [ -f "$q" ] && grep -q '^run_spec ' "$q" || continue
  name=fps-{model}-$st-$w; log={logs}/$name.log
  if tmux has-session -t "=$name" 2>/dev/null; then echo "SKIP_EXISTS $name"; continue; fi
  if [ -e "$log" ]; then echo "SKIP_LOG_EXISTS $log（重跑前先把旧日志改名留档）"; continue; fi
  tmux new-session -d -s "$name" "set -o pipefail; PYTHONUNBUFFERED=1 bash $q 2>&1 | tee $log; echo \\"EXIT_CODE=\\$?\\" >> $log"
  echo "STARTED $name $log"
done
"""


ENV_GIT_CHECK = ('test "$(git rev-parse HEAD)" = {commit} || {{ echo "ENV_CHECK=FAIL reason=head"; return 2; }}\n'
                 'test -z "$(git status --porcelain)" || {{ echo "ENV_CHECK=FAIL reason=dirty"; return 2; }}\n')
assert ENV_GIT_CHECK in ENV_TEMPLATE and "MMEVLA_V1_STORE={store}" in ENV_TEMPLATE
# F5：release 目录（git archive 解出，无 .git）不查 git；改查目录名与 commit 互为前缀、且入口脚本在。
ENV_RELEASE_CHECK = ('b=$(basename "$T"); case "{commit}" in "$b"*) ;; *) case "$b" in {commit}*) ;; '
                     '*) echo "ENV_CHECK=FAIL reason=release_name"; return 2;; esac;; esac\n'
                     'test -f "$T/scripts/interpret/run_local.py" || {{ echo "ENV_CHECK=FAIL reason=release_files"; return 2; }}\n')


def default_roots():
    """旧行为的路径根（aspen 主检出）。"""
    return {"main": str(P.MAIN), "store": str(STORE), "bench_src": str(BENCH_SRC),
            "policy_python": SPEC_TEMPLATE["policy_python"], "sim_python": SPEC_TEMPLATE["sim_python"],
            "stage2_python": SPEC_TEMPLATE["stage2_python"], "mmevla_store": str(STORE), "log_dir": str(LOG_DIR),
            "release": False}


def resolve_roots(args):
    """命令行 → 路径根。--gl-store 是预设：store、三个解释器、MMEVLA_V1_STORE、release；bench_src 缺省取
    <run-tree>/third_party/robomme_benchmark/src。显式参数优先于预设。"""
    roots = default_roots()
    gl = getattr(args, "gl_store", None)
    store = getattr(args, "store", None) or gl
    if store:
        roots.update(store=str(store), mmevla_store=str(store), log_dir=str(Path(store) / "logs/fp-swap-1110"))
    if gl:
        g = Path(gl)
        roots.update(policy_python=str(g / "envs/policy/bin/python"), sim_python=str(g / "envs/official-sim/bin/python"),
                     stage2_python=str(g / "envs/stage2/bin/python"), release=True,
                     bench_src=str(Path(args.run_tree) / "third_party/robomme_benchmark/src"))
    for key in ("main", "bench_src", "policy_python", "sim_python", "stage2_python", "mmevla_store", "log_dir"):
        value = getattr(args, key, None)
        if value:
            roots[key] = str(value)
    if getattr(args, "release", False):
        roots["release"] = True
    return roots


def parse_workers(text, gl=False):
    """'名=GPU,…' → (写出顺序, 轮转顺序, 名→GPU)。缺省为旧 4 worker（轮转两卡交替）；--gl-store 缺省 gl00..gl15。"""
    if not text:
        if gl:
            return list(GL_WORKERS), list(GL_WORKERS), {w: "0" for w in GL_WORKERS}
        return list(P.WORKERS), list(P.ASSIGN_ORDER), dict(P.WORKER_CUDA)
    cuda = {}
    for part in str(text).split(","):
        name, _, gpu = part.strip().partition("=")
        if not name or not gpu or name in cuda or not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise SystemExit(f"--workers 格式为 名=GPU,…（名只含字母数字下划线、不重复）：{text}")
        cuda[name] = gpu
    return list(cuda), list(cuda), cuda


def reroot(value, roots):
    """以旧 STORE / MAIN 开头的路径换成新根（STORE 优先，它在 MAIN 之下）。"""
    if not isinstance(value, str):
        return value
    for old, key in ((str(STORE), "store"), (str(P.MAIN), "main")):
        if roots[key] != old and (value == old or value.startswith(old + "/")):
            return roots[key] + value[len(old):]
    return value


def render_env(roots, frac, run_tree, commit, out):
    text = ENV_TEMPLATE.replace("MMEVLA_V1_STORE={store}", "MMEVLA_V1_STORE={mmevla}")
    if roots.get("release"):
        text = text.replace(ENV_GIT_CHECK, ENV_RELEASE_CHECK)
    return text.format(frac=frac, run_tree=run_tree, commit=commit, store=roots["store"], bench=roots["bench_src"],
                       out=out, py=roots["policy_python"], sim=roots["sim_python"], mmevla=roots["mmevla_store"])


class Builder(P.Builder):
    """沿用 pl_full_prep.Builder；清单为多任务自建 metadata，spec 改用 bin_trace + 拿起即停。
    roots / workers 缺省时与旧行为一致；给出时 spec 路径改根、worker 与 GPU 用自定义表。"""

    def __init__(self, *a, meta_shas=None, roots=None, workers=None, **kw):
        super().__init__(*a, **kw)
        self.meta_shas = dict(meta_shas or {})
        self.roots = dict(roots or default_roots())
        order, assign, cuda = workers or (list(P.WORKERS), list(P.ASSIGN_ORDER), dict(P.WORKER_CUDA))
        self.workers, self.assign_order, self.worker_cuda = list(order), list(assign), dict(cuda)

    def round_robin(self, items):
        return [(self.assign_order[i % len(self.assign_order)], item) for i, item in enumerate(items)]

    def add_spec(self, name, stage, worker, episodes, *, seed=MAIN_SEED, profile="plfull", ablate="none",
                 swap_map=None, record=None, manifest=None, chunk=P.CHUNK, record_dir=None, rows=()):
        """同 pl_full_prep.Builder.add_spec，GPU 取自 self.worker_cuda。"""
        if name in self.specs:
            raise ValueError(f"spec 重名 {name}")
        if worker not in self.worker_cuda:
            raise ValueError(f"未知 worker {worker}")
        run_dir = self.out / "runs" / name
        spec = P.base_spec(self.model, self.run_tree)
        spec.update(model_seed=int(seed), cuda=self.worker_cuda[worker], attn_out=str(run_dir / "attn"),
                    run_root=str(run_dir / "run"), xla_mem_fraction=self.frac,
                    manifest=str(self.out / "manifests" / f"{name}.json"), ablate=ablate)
        if profile == "plfull":
            spec.update(profile="plfull", plfull_metadata_dir=str(self.meta_dir))
        if swap_map:
            spec["swap_map"] = dict(swap_map)
        if record:
            spec["record_motion"] = str(record)
        if manifest is None:
            manifest = manifest_for(episodes, self.meta_shas, chunk=chunk)
        self.specs[name] = (spec, manifest, stage, worker)
        self.queue.setdefault((stage, worker), []).append(
            f"run_spec {stage} {name} {shlex.quote(str(self.out / 'specs' / f'{name}.json'))}\n")
        for row in rows:
            row = dict(row)
            row.update(stage=stage, worker=worker, cuda=self.worker_cuda[worker], seed=int(seed), spec=name,
                       spec_path=str(self.out / "specs" / f"{name}.json"), attn_out=spec["attn_out"],
                       run_root=spec["run_root"], record_dir=str(record_dir or self.out / "record"))
            self.rows.append(row)
        return spec

    def add(self, name, stage, worker, episodes, **kw):
        chunk = kw.pop("chunk", P.CHUNK)
        kw["manifest"] = manifest_for(episodes, self.meta_shas, chunk=chunk)
        spec = self.add_spec(name, stage, worker, episodes, **kw)
        spec.pop("subgoal_trace", None)          # 与 bin_trace 互斥
        spec.update(bin_trace=True, stop_after_first_pick=True)
        for key in list(spec):
            spec[key] = reroot(spec[key], self.roots)
        spec.update(policy_python=self.roots["policy_python"], sim_python=self.roots["sim_python"],
                    stage2_python=self.roots["stage2_python"])
        return spec

    def write(self, stages):
        for name, (spec, manifest, stage, worker) in self.specs.items():
            P._write_json(self.out / "manifests" / f"{name}.json", manifest)
            P._write_json(self.out / "specs" / f"{name}.json", spec)
        written = []
        (self.out / "queues").mkdir(parents=True, exist_ok=True)
        for stage in stages:
            for worker in self.workers:
                lines = self.queue.get((stage, worker), [])
                n_spec = sum(1 for x in lines if x.startswith("run_spec "))
                path = self.out / "queues" / f"{stage}-{worker}.sh"
                text = [QUEUE_HEAD.format(stage=stage, worker=worker, cuda=self.worker_cuda[worker], n=n_spec,
                                          env=shlex.quote(str(self.out / "env.sh")), frac=self.frac)]
                text += lines
                text.append(QUEUE_TAIL.format(stage=stage, worker=worker, n=n_spec))
                path.write_text("".join(text))
                path.chmod(0o755)
                written.append(path)
        return written


def make_builder(out, model_name, model, run_tree, frac, pool, roots=None, workers=None):
    pool_root = Path(out).parent
    meta_dir = pool_root / "metadata"
    shas = {name: hashlib.sha256((meta_dir / name).read_bytes()).hexdigest() for name in pool["metadata_sha256"]}
    if shas != pool["metadata_sha256"]:
        raise SystemExit("metadata 与 pool.json 记录的 sha256 不符")
    return Builder(out, pool_root, model_name, model, run_tree, frac, meta_dir, None, meta_shas=shas, roots=roots,
                   workers=workers)


def plan_workers(plan):
    """plan.json 记录的 worker 表 → Builder 的 workers 参数；旧 plan 没有该键时为旧 4 worker。"""
    w = plan.get("workers")
    if not w:
        return None
    return list(w["order"]), list(w["assign"]), dict(w["cuda"])


def model_entry(name, overrides=None):
    entry = dict(MODELS[name])
    entry.update({k: v for k, v in (overrides or {}).items() if v is not None})
    missing = [k for k in ("checkpoint", "policy_config", "model_kind", "target_contract") if not entry.get(k)]
    if missing:
        raise SystemExit(f"模型 {name} 缺 {missing}：Wan 项须用 --checkpoint/--policy-config/--model-kind/--target-contract 给出")
    return entry


def n_row(e, **extra):
    row = {"group": "N", "task": e["task_name"], "ep": int(e["episode_id"]), "slot": None, "donor": None,
           "donor_src": None, "swap_times": int(e["swap_times"]), "moved": bool(e["moved"])}
    row.update(extra)
    return row


def smoke_episodes(pool_by_key):
    return [pool_by_key[episode_key(t, 0)] for t in TASKS if episode_key(t, 0) in pool_by_key]


def build_stage_specs(b, pool_by_key, stages):
    out = b.out
    if "smoke" in stages:
        smoke = out / "smoke"
        eps = smoke_episodes(pool_by_key)
        w0 = b.workers[0]
        b.add("smoke-n", "smoke", w0, eps, record=smoke / "record", record_dir=smoke / "record",
              rows=[n_row(e) for e in eps])
        b.add_cmd("smoke", w0, f"\"$py\" -m scripts.interpret.fp_swap_prep donors --out {shlex.quote(str(out))} --smoke")
        b.add("smoke-zall", "smoke", w0, eps, ablate=f"motion_swap_exact:{smoke / DONOR_DIR}",
              swap_map={episode_key(e["task_name"], e["episode_id"]):
                        episode_key(e["task_name"], synthetic_ep(e["episode_id"], "Zall")) for e in eps},
              record_dir=smoke / "record",
              rows=[dict(n_row(e), group="Zall", slot="Zall", donor=synthetic_ep(e["episode_id"], "Zall")) for e in eps])
    if "n" in stages:
        keys = sorted(pool_by_key, key=lambda k: (TASKS.index(split_key(k)[0]), split_key(k)[1]))
        per_worker = {}
        for worker, key in b.round_robin(keys):
            per_worker.setdefault(worker, []).append(key)
        for worker in b.workers:
            mine = per_worker.get(worker, [])
            for i in range(0, len(mine), P.SPEC_MAX):
                eps = [pool_by_key[k] for k in mine[i:i + P.SPEC_MAX]]
                b.add(f"n-{worker}-{i // P.SPEC_MAX:02d}", "n", worker, eps, record=out / "record",
                      rows=[n_row(e) for e in eps])


def cmd_specs(args):
    out = Path(args.out).resolve()
    if not args.commit or len(args.commit) < 7:
        raise SystemExit("--commit 须给运行工作树的 commit sha")
    if out.name != args.model:
        raise SystemExit(f"--out 末级目录须为模型名 {args.model}（约定 <O>/<M>）")
    extra = json.loads(args.extra) if args.extra else None
    model = model_entry(args.model, {"checkpoint": args.checkpoint, "policy_config": args.policy_config,
                                     "model_kind": args.model_kind, "target_contract": args.target_contract})
    if extra:
        model["extra"] = extra
    pool, pool_by_key = load_pool(out.parent)
    plan_path = out / "plan.json"
    if plan_path.exists() and not args.overwrite:
        raise SystemExit(f"{plan_path} 已存在；确需重写 spec 加 --overwrite（不删任何运行目录）")
    out.mkdir(parents=True, exist_ok=True)
    roots = resolve_roots(args)
    workers = parse_workers(args.workers, gl=bool(args.gl_store))
    b = make_builder(out, args.model, model, args.run_tree, args.frac, pool, roots=roots, workers=workers)
    stages = ("smoke", "n")
    build_stage_specs(b, pool_by_key, stages)
    (out / "env.sh").write_text(render_env(roots, args.frac, args.run_tree, args.commit, out))
    launch = out / "launch_stage.sh"
    launch.write_text(LAUNCH_TEMPLATE.format(out=out, model=args.model, logs=roots["log_dir"],
                                             workers=" ".join(b.workers), stages="|".join(STAGES)))
    launch.chmod(0o755)
    queues = b.write(BASE_STAGES)   # multieq/swap/seeds 先写空队列，donors 再覆盖；kswap/kseeds 由 donors --by-k 写
    plan = {"tasks": list(TASKS), "model": args.model, "commit": args.commit, "run_tree": str(args.run_tree),
            "out": str(out), "pool_root": str(out.parent), "record_dir": str(out / "record"), "model_entry": model,
            "manifest_seal": pool["manifest_seal"], "frac": float(args.frac), "gate": None, "runs": b.rows}
    if roots != default_roots() or b.workers != list(P.WORKERS):
        plan.update(roots=roots, workers={"order": b.workers, "assign": b.assign_order, "cuda": b.worker_cuda})
    P._write_json(plan_path, plan)
    counts = {st: sum(1 for r in b.rows if r["stage"] == st) for st in stages}
    print("FPS_SPECS model=" + args.model + " " + " ".join(f"{k}={v}" for k, v in counts.items())
          + f" specs={len(b.specs)} queues={len(queues)}", flush=True)
    return 0


# ---------------------------------------------------------------- donors：读 N 阶段
def load_record(path):
    """录制 npz → {起点: emb}；缺文件返回 None；重复起点取第一次。"""
    if not Path(path).is_file():
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


def complete_schedule(schedule, scan_schedule):
    """回执交换表里未确定的交换对（Button 拿起即停后 idx2 为 None）用局池 / 扫描的同 seed 同难度交换表补全。
    返回 (补全后的表或 None, 每条是否由回执确定的掩码)。条数或起止步不符、已确定的交换对与扫描不符时返回 None
    （已确定部分的不符留给 FP_REPLAY 计数，这里只在无法补全时放弃）。"""
    determined = [all(v is not None for v in (item or [])) and len(item or []) == 4 for item in schedule]
    if all(determined) and schedule:
        return [[int(v) for v in item] for item in schedule], determined
    if not scan_schedule or len(scan_schedule) != len(schedule):
        return None, determined
    out = []
    for item, ok, ref in zip(schedule, determined, scan_schedule):
        if ok:
            out.append([int(v) for v in item])
            continue
        item = list(item or [None] * 4) + [None] * (4 - len(item or []))
        if any(item[i] is not None and int(item[i]) != int(ref[i]) for i in range(4)):
            return None, determined
        out.append([int(v) for v in ref])
    return out, determined


def apply_completion(rf, scan_entry):
    """就地把 rf 的交换表补全；返回 True 表示可用（完整或已补全），False 为 excluded。"""
    rf["raw_schedule"] = [list(item or []) for item in rf["schedule"]]
    sched, mask = complete_schedule(rf["schedule"], (scan_entry or {}).get("swap_schedule"))
    rf["determined"] = mask
    rf["completed_from_scan"] = sched is not None and not all(mask)
    if sched is None:
        return False
    rf["schedule"], rf["schedule_complete"] = sched, True
    return True


def episode_info(task, ep, run, record_dir, scan_entry=None):
    """一局 N 阶段读数 → 标签、窗口、拿起、资格。status ∈ ok / excluded（交换表补不全或缺目标）/ no_terminal /
    no_receipt / no_trace。"""
    info = {"task": task, "ep": int(ep), "key": episode_key(task, ep), "status": "ok", "record": None}
    if run["terminal"] is None:
        info["status"] = "no_terminal"
        return info
    if run["receipt"] is None or run["trace"] is None:
        info["status"] = "no_receipt" if run["trace"] is not None else "no_trace"
        return info
    rf = receipt_fields(run["receipt"])
    if not apply_completion(rf, scan_entry) or rf["target_index"] is None:
        info["status"] = "excluded"
        return info
    first = read_step_rows(run["trace"], limit=1)
    first_row = trace_row_fields(first[0]) if first else None
    n_obj = len(first_row["objects"]) if first_row and first_row["objects"] else None
    labels = episode_labels(task, rf, n_obj)
    st = pick_status(task, rf, labels, first_row)
    record_path = Path(record_dir) / record_name(task, ep)
    rec = load_record(record_path)
    has_w = rec is not None and all(f in rec for f in labels["W"])
    info.update(rf=rf, labels=labels, pick=st, outcome=run["terminal"].get("outcome"),
                swap_times=rf["swap_times"], moved=labels["moved"], record=str(record_path) if rec is not None else None,
                record_has_w=bool(has_w), category=classify_pick(rf["first_pick"], rf["target_index"],
                                                                labels["swap_blind_index"], labels["moved"]))
    fp = rf["first_pick"]
    info["recv_ok"] = bool(fp is not None and fp["is_target"] and st["on_turn"] and st["f3_ok"] and has_w and labels["W"])
    info["donor_ok"] = bool(st["f3_ok"] and has_w and labels["W"])
    return info


def collect_n(plan, record_dir, pool_by_key=None):
    infos = {}
    for row in plan["runs"]:
        if row["stage"] != "n":
            continue
        key = episode_key(row["task"], row["ep"])
        infos[key] = episode_info(row["task"], row["ep"], find_run(row["run_root"], row["attn_out"], key), record_dir,
                                  (pool_by_key or {}).get(key))
    return infos


# ---------------------------------------------------------------- donors：分组
def assign_groups(infos, task, few=False, records=None):
    """返回 (runs, notes)：runs 为 [{group, slot, recv, donor_src, fallback}]，按接收局升序。
    A：同 swap_times、同 moved；B：同 swap_times、moved 相反（VideoRepick 无候选时取 swap_times=2、moved=否）。
    给出 records 时先剔除按顺序对齐不可行的候选（供体窗或补齐窗不在录制里），计入 notes["infeasible"]。
    A/B 每组一个 default_rng(0)，按接收局升序逐局 choice(2, replace=False)。"""
    mine = {k: v for k, v in infos.items() if v["task"] == task and v["status"] == "ok"}
    recvs = sorted((v for v in mine.values() if v["recv_ok"]), key=lambda v: v["ep"])
    donors = sorted((v for v in mine.values() if v["donor_ok"]), key=lambda v: v["ep"])
    rng = {"A": np.random.default_rng(PAIR_SEED), "B": np.random.default_rng(PAIR_SEED)}
    runs, notes = [], {"short": [], "fallback": [], "no_s0": [], "infeasible": []}

    def feasible(r, d):
        if records is None:
            return True
        rec = records.get(d["key"])
        if rec is None:
            return False
        return aligned_rows(r["labels"]["W"], d["labels"]["W"], rec, d["labels"]["es"])[0] is not None

    def filt(r, group, cands):
        ok = [d for d in cands if feasible(r, d)]
        kept = {d["key"] for d in ok}
        notes["infeasible"] += [f"{r['key']}:{group}:{d['key']}" for d in cands if d["key"] not in kept]
        return ok
    for r in recvs:
        base = {"recv": r["key"], "task": task, "ep": r["ep"]}
        runs.append(dict(base, group="Z", slot="Z", donor_src=None))
        if not few:
            runs.append(dict(base, group="Rnd", slot="Rnd", donor_src=None))
        if r["labels"]["S0"] is not None:
            runs.append(dict(base, group="S0", slot="S0", donor_src=r["key"]))
        else:
            notes["no_s0"].append(r["key"])
        if not few:
            for group in ("A", "B"):
                want_moved = r["moved"] if group == "A" else not r["moved"]
                cands = filt(r, group, [d for d in donors if d["key"] != r["key"] and d["swap_times"] == r["swap_times"]
                                        and d["moved"] == want_moved])
                fallback = False
                if group == "B" and not cands and task == "VideoRepick":
                    cands = filt(r, group, [d for d in donors if d["key"] != r["key"] and d["swap_times"] == 2
                                            and not d["moved"]])
                    fallback = True
                    notes["fallback"].append(r["key"])
                k = min(2, len(cands))
                if k < 2:
                    notes["short"].append(f"{r['key']}:{group}:{len(cands)}")
                picks = sorted(rng[group].choice(len(cands), size=k, replace=False).tolist()) if k else []
                for j, idx in enumerate(picks):
                    slot = f"{group}{j + 1}"
                    runs.append(dict(base, group=group, slot=slot, donor_src=cands[idx]["key"], fallback=fallback))
        runs.append(dict(base, group="Zall", slot="Zall", donor_src=None))
    for run in runs:
        run.setdefault("fallback", False)
        run["donor"] = synthetic_ep(run["ep"], run["slot"])
    return runs, notes


# ---------------------------------------------------------------- donors：合成供体文件
def random_rows(rows, seed):
    rng = np.random.default_rng(int(seed))
    out = np.empty_like(rows, dtype=np.float32)
    for i, row in enumerate(rows):
        v = rng.standard_normal(row.shape).astype(np.float64)
        v *= float(np.linalg.norm(row.astype(np.float64))) / float(np.linalg.norm(v))
        out[i] = v.astype(np.float32)
    return out


def aligned_rows(recv_W, donor_W, donor_rec, donor_es):
    """A/B：接收局 W 第 k 窗 ← 供体 W 第 k 窗；供体窗不够时用供体交换后第一个窗补齐。
    返回 (embs[len(recv_W), D] 或 None, 补齐数, 源起点列表, 问题)。"""
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


def donor_payload(run, infos, records):
    """一条交换局 → (starts, embs, meta) 或 (None, None, 问题)。"""
    r = infos[run["recv"]]
    W = r["labels"]["W"]
    rec = records[run["recv"]]
    dim = next(iter(rec.values())).shape[0]
    slot = run["slot"]
    meta = {"pad": 0, "src_starts": None}
    if slot == "Z":
        return W, np.zeros((len(W), dim), np.float32), meta
    if slot == "Rnd":
        return W, random_rows(np.stack([rec[f] for f in W]), run["donor"]), meta
    if slot == "S0":
        s0 = r["labels"]["S0"]
        if s0 not in rec:
            return None, None, "s0_missing"
        meta["src_starts"] = [s0] * len(W)
        return W, np.stack([rec[s0]] * len(W)).astype(np.float32), meta
    if slot == "Zall":
        starts = all_starts(r["labels"]["es"])
        return starts, np.zeros((len(starts), dim), np.float32), meta
    d = infos[run["donor_src"]]
    embs, pad, srcs, why = aligned_rows(W, d["labels"]["W"], records[run["donor_src"]], d["labels"]["es"])
    if embs is None:
        return None, None, why
    meta.update(pad=pad, src_starts=srcs)
    return W, embs, meta


def write_donor_files(folder, runs, infos, records):
    """写 folder/<task>_ep<合成号>.npz 并读回核对。返回 (ok, 判定行, problems, done_runs)。
    单局问题（donor_missing_starts / no_pad_window / s0_missing）只记 problems 并跳过该局该组，不影响 ok；
    读回不符（形状、逐位、范数）同样跳过该局，但判 DONOR_FILES=FAIL（文件写坏是脚本问题）。"""
    folder = Path(folder)
    problems, done, integrity = [], [], 0
    stats = {"files": 0, "pad_runs": 0, "pad_windows": 0, "maxabs": 0.0, "norm_maxrel": 0.0}
    for run in runs:
        starts, embs, meta = donor_payload(run, infos, records)
        if starts is None:
            problems.append(f"{run['recv']}:{run['slot']}:{meta}")
            continue
        path = folder / record_name(run["task"], run["donor"])
        P._savez_atomic(path, start_frames=np.asarray(starts, np.int32), embs=embs.astype(np.float32),
                        counts=np.ones(len(starts), np.int32))
        with np.load(path) as back:
            b_starts, b_embs = np.asarray(back["start_frames"]).tolist(), np.asarray(back["embs"], np.float32)
        if b_starts != [int(s) for s in starts] or b_embs.shape != embs.shape:
            problems.append(f"{run['recv']}:{run['slot']}:readback_shape")
            integrity += 1
            continue
        if run["slot"] == "Rnd":
            ref = np.stack([records[run["recv"]][f] for f in starts]).astype(np.float64)
            n0, n1 = np.linalg.norm(ref, axis=1), np.linalg.norm(b_embs.astype(np.float64), axis=1)
            rel = float(np.max(np.abs(n1 - n0) / np.where(n0 > 0, n0, np.nan))) if len(n0) else 0.0
            rel = rel if np.isfinite(rel) else float("inf")
            stats["norm_maxrel"] = max(stats["norm_maxrel"], rel)
            if not rel < 1e-5:
                problems.append(f"{run['recv']}:Rnd:norm_rel={rel:g}")
                integrity += 1
                continue
        else:
            dev = float(np.max(np.abs(b_embs - embs))) if embs.size else 0.0
            stats["maxabs"] = max(stats["maxabs"], dev)
            if dev != 0.0:
                problems.append(f"{run['recv']}:{run['slot']}:maxabs={dev}")
                integrity += 1
                continue
        stats["files"] += 1
        if meta["pad"]:
            stats["pad_runs"] += 1
            stats["pad_windows"] += meta["pad"]
        done.append(dict(run, w_starts=[int(s) for s in starts], pad=meta["pad"], src_starts=meta["src_starts"]))
    ok = integrity == 0
    line = (f"DONOR_FILES={'PASS' if ok else 'FAIL'} files={stats['files']} skipped={len(problems)} "
            f"integrity={integrity} pad_runs={stats['pad_runs']} pad_windows={stats['pad_windows']} "
            f"maxabs={stats['maxabs']:g} norm_maxrel={stats['norm_maxrel']:.2e}")
    return ok, line, problems, done


def distinct_line(runs, infos, records):
    """A/B 供体源编码（改写前，供体 W 各窗拼接）按字节两两比较；逐位相同算一对。"""
    srcs = sorted({r["donor_src"] for r in runs if r["group"] in ("A", "B")})
    digests = {}
    for key in srcs:
        rec, W = records.get(key), infos[key]["labels"]["W"]
        if rec is None or any(f not in rec for f in W):
            continue
        digests[key] = hashlib.sha256(np.stack([rec[f] for f in W]).astype(np.float32).tobytes()).hexdigest()
    identical = [(a, b) for i, a in enumerate(sorted(digests)) for b in sorted(digests)[i + 1:] if digests[a] == digests[b]]
    ok = not identical and len(digests) == len(srcs)
    return (f"DONOR_DISTINCT={'PASS' if ok else 'FAIL'} donors={len(digests)} identical_pairs={len(identical)}",
            ok, identical)


def recv_line(task, infos):
    mine = [v for v in infos.values() if v["task"] == task]
    ok = [v for v in mine if v["status"] == "ok"]
    picked = [v for v in ok if v["rf"]["first_pick"] is not None]
    return (f"FP_RECV task={task} eps={len(mine)} ok={len(ok)} picked={len(picked)} "
            f"recv={sum(v['recv_ok'] for v in ok)} donors={sum(v['donor_ok'] for v in ok)} "
            f"f3_bad={sum(1 for v in ok if not v['pick']['f3_ok'])} "
            f"before_buttons={sum(1 for v in ok if v['pick']['before_buttons'])} "
            f"not_on_turn={sum(1 for v in picked if not v['pick']['on_turn'])}")


# ---------------------------------------------------------------- donors：阶段 spec
def run_row(run, infos):
    r = infos[run["recv"]]
    return {"group": run["group"], "task": run["task"], "ep": int(run["ep"]), "slot": run["slot"],
            "donor": int(run["donor"]) if run.get("donor") is not None else None, "donor_src": run.get("donor_src"),
            "swap_times": r["swap_times"], "moved": r["moved"], "w_starts": run.get("w_starts"),
            "pad": run.get("pad", 0), "fallback": run.get("fallback", False)}


def seedcheck_runs(swap_runs, recv_keys):
    """每任务每组（含 N）default_rng(0) 抽 5 局，seed 1、2 各跑一次。"""
    out = []
    for task in TASKS:
        for group in GROUPS:
            if group == "N":
                pool = [{"recv": k, "task": task, "ep": split_key(k)[1], "group": "N", "slot": None, "donor": None,
                         "donor_src": None} for k in sorted(recv_keys, key=lambda k: split_key(k)[1])
                        if split_key(k)[0] == task]
            else:
                pool = [r for r in swap_runs if r["task"] == task and r["group"] == group]
            if not pool:
                continue
            rng = np.random.default_rng(PAIR_SEED)
            idx = sorted(rng.choice(len(pool), size=min(SEEDCHECK_PER_GROUP, len(pool)), replace=False).tolist())
            for seed in SEEDCHECK_SEEDS:
                out += [dict(pool[i], seed=seed) for i in idx]
    return out


def multieq_runs(swap_runs):
    chosen, used = [], set()
    for group in MULTIEQ_GROUPS:
        n = 0
        for r in swap_runs:
            if r["group"] != group or r["recv"] in used:
                continue
            chosen.append(r)
            used.add(r["recv"])
            n += 1
            if n == MULTIEQ_PER_GROUP:
                break
    return chosen


def build_swap_specs(b, pool_by_key, infos, swap_runs, seed_runs):
    ablate = f"motion_swap_exact:{b.out / DONOR_DIR}"
    chosen = multieq_runs(swap_runs)
    multi_w = b.workers[min(2, len(b.workers) - 1)]          # 缺省 4 worker 时 = 旧的 w10；singles = w00、w01、w11
    singles = [w for w in b.workers if w != multi_w][:3] or [multi_w]
    for i, run in enumerate(chosen):
        b.add(f"multieq-single-{i:02d}", "multieq", singles[i % len(singles)], [pool_by_key[run["recv"]]], ablate=ablate,
              swap_map={run["recv"]: episode_key(run["task"], run["donor"])}, chunk=1,
              rows=[dict(run_row(run, infos), variant="single", pair_index=i)])
    if chosen:
        b.add("multieq-multi", "multieq", multi_w, [pool_by_key[r["recv"]] for r in chosen], ablate=ablate,
              swap_map={r["recv"]: episode_key(r["task"], r["donor"]) for r in chosen},
              rows=[dict(run_row(r, infos), variant="multi", pair_index=i) for i, r in enumerate(chosen)])
    for stage, members in (("swap", [dict(r, seed=MAIN_SEED) for r in swap_runs]), ("seeds", seed_runs)):
        per_worker = {}
        for worker, run in b.round_robin(members):
            per_worker.setdefault(worker, []).append(run)
        for worker in b.workers:
            items = [((run["seed"], "none" if run["group"] == "N" else "fp"), run["recv"], run)
                     for run in per_worker.get(worker, [])]
            for i, ((seed, kind), bin_) in enumerate(P.pack(items)):
                runs = [item[2] for item in bin_]
                name = f"{stage}-{worker}-{kind}-s{seed}-{i:02d}"
                eps = [pool_by_key[r["recv"]] for r in runs]
                if kind == "none":
                    b.add(name, stage, worker, eps, seed=seed, rows=[n_row(pool_by_key[r["recv"]]) for r in runs])
                else:
                    b.add(name, stage, worker, eps, seed=seed, ablate=ablate,
                          swap_map={r["recv"]: episode_key(r["task"], r["donor"]) for r in runs},
                          rows=[run_row(r, infos) for r in runs])


def cmd_donors_smoke(out):
    """smoke：用 pool 的扫描标签与 smoke 录制写 Zall 合成供体（每任务局号 0）。"""
    pool, pool_by_key = load_pool(out.parent)
    problems, files = [], 0
    for e in smoke_episodes(pool_by_key):
        rec = load_record(out / "smoke" / "record" / record_name(e["task_name"], e["episode_id"]))
        if rec is None:
            problems.append(f"{e['task_name']}:{e['episode_id']}:missing_record")
            continue
        dim = next(iter(rec.values())).shape[0]
        starts = all_starts(e["es"])
        path = out / "smoke" / DONOR_DIR / record_name(e["task_name"], synthetic_ep(e["episode_id"], "Zall"))
        P._savez_atomic(path, start_frames=np.asarray(starts, np.int32), embs=np.zeros((len(starts), dim), np.float32),
                        counts=np.ones(len(starts), np.int32))
        with np.load(path) as back:
            if np.asarray(back["start_frames"]).tolist() != starts or float(np.max(np.abs(back["embs"]))) != 0.0:
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
    if getattr(args, "by_k", False):
        return cmd_donors_k(out)
    plan = P._read_json(out / "plan.json")
    pool, pool_by_key = load_pool(out.parent)
    infos = collect_n(plan, out / "record", pool_by_key)
    records = {k: load_record(v["record"]) for k, v in infos.items() if v.get("record")}
    gate_min = args.wan_gate
    swap_runs, notes, gate = [], {}, {}
    for task in TASKS:
        print(recv_line(task, infos), flush=True)
        n_recv = sum(1 for v in infos.values() if v["task"] == task and v["status"] == "ok" and v["recv_ok"])
        few = gate_min is not None and n_recv < gate_min
        gate[task] = {"recv": n_recv, "few": few, "groups": ["N", *GATE_GROUPS] if few else list(GROUPS)}
        runs, notes[task] = assign_groups(infos, task, few=few, records=records)
        swap_runs += runs
    ok, line, problems, done = write_donor_files(out / DONOR_DIR, swap_runs, infos, records)
    dline, dok, identical = distinct_line(done, infos, records)
    for problem in problems:
        print(f"DONOR_PROBLEM {problem}", flush=True)
    for a, b_ in identical:
        print(f"DONOR_IDENTICAL {a} {b_}", flush=True)
    print(line, flush=True)
    print(dline, flush=True)
    for task in TASKS:
        n = notes[task]
        print(f"FP_GROUPS task={task} gate={'few' if gate[task]['few'] else ('ok' if gate_min is not None else 'off')} "
              f"recv={gate[task]['recv']} short={len(n['short'])} fallback={len(n['fallback'])} no_s0={len(n['no_s0'])} "
              f"infeasible={len(n['infeasible'])} excluded={sum(1 for v in infos.values() if v['task'] == task and v['status'] == 'excluded')}",
              flush=True)
    # 单局问题只跳过该局该组；只有 A/B 源编码逐位相同（DONOR_DISTINCT=FAIL）才整体不生成后续 spec。
    if not dok:
        print("DONORS_SPECS=SKIP reason=donor_distinct", flush=True)
        return 1
    recv_keys = sorted({r["recv"] for r in done})
    seed_runs = seedcheck_runs(done, recv_keys)
    b = make_builder(out, plan["model"], plan["model_entry"], plan["run_tree"], plan.get("frac", XLA_MEM_FRACTION), pool,
                     roots=plan.get("roots"), workers=plan_workers(plan))
    build_swap_specs(b, pool_by_key, infos, done, seed_runs)
    b.write(("multieq", "swap", "seeds"))
    plan["runs"] = [r for r in plan["runs"] if r["stage"] not in ("multieq", "swap", "seeds")] + b.rows
    plan["gate"] = {"wan_gate": gate_min, "tasks": gate}
    plan["recv"] = recv_keys
    plan["eligibility"] = {k: {"status": v["status"], "recv_ok": v.get("recv_ok"), "donor_ok": v.get("donor_ok"),
                               "f3_ok": (v.get("pick") or {}).get("f3_ok"), "on_turn": (v.get("pick") or {}).get("on_turn"),
                               "before_buttons": (v.get("pick") or {}).get("before_buttons"),
                               "W": (v.get("labels") or {}).get("W"), "S0": (v.get("labels") or {}).get("S0"),
                               "w_end": (v.get("labels") or {}).get("w_end")} for k, v in infos.items()}
    plan["group_notes"] = notes
    plan["donor_problems"] = {"count": len(problems), "integrity_ok": bool(ok), "items": problems}
    P._write_json(out / "plan.json", plan)
    counts = {}
    for row in b.rows:
        counts[f"{row['stage']}:{row['group']}"] = counts.get(f"{row['stage']}:{row['group']}", 0) + 1
    print("FPS_SWAP_SPECS " + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f" total={len(b.rows)} specs={len(b.specs)}", flush=True)
    print(f"DONORS={'PASS' if ok else 'FAIL'} problems={len(problems)} integrity_ok={int(ok)} swap_runs={len(done)}",
          flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------- donors --by-k：按第 k 次交换（SP-W2）
def k_windows(info, k):
    return list(((info.get("labels") or {}).get("Wk") or {}).get(int(k)) or [])


def assign_k_groups(infos, task, records=None):
    """返回 (runs, notes)。接收局 = recv_ok（N 第一次拿起满足 F3/F4 资格）且 n ≥ k、W_k 非空且 W_k 全在录制里；
    Bsame_k 供体 = 同 swap_times 的另一局（donor_ok、W_k 非空、按 W_k 内顺序对齐可行），每 (任务,k) 一个
    default_rng(0)，按接收局升序逐局 choice 1 个；Z_k 为零向量。"""
    mine = {k_: v for k_, v in infos.items() if v["task"] == task and v["status"] == "ok"}
    recvs = sorted((v for v in mine.values() if v["recv_ok"]), key=lambda v: v["ep"])
    donors = sorted((v for v in mine.values() if v["donor_ok"]), key=lambda v: v["ep"])
    runs, notes = [], {"short": [], "infeasible": [], "no_wk": []}

    def has(rec_key, W):
        rec = (records or {}).get(rec_key) if records is not None else None
        return records is None or (rec is not None and all(f in rec for f in W))

    def feasible(r, d, k):
        if records is None:
            return True
        rec = records.get(d["key"])
        return rec is not None and aligned_rows(k_windows(r, k), k_windows(d, k), rec, d["labels"]["es"])[0] is not None
    for k in range(1, K_MAX + 1):
        rng = np.random.default_rng(PAIR_SEED)
        for r in recvs:
            if int(r["swap_times"]) < k:
                continue
            Wr = k_windows(r, k)
            if not Wr or not has(r["key"], Wr):
                notes["no_wk"].append(f"{r['key']}:k{k}")
                continue
            base = {"recv": r["key"], "task": task, "ep": r["ep"], "k": k, "swap_times": int(r["swap_times"])}
            cands = [d for d in donors if d["key"] != r["key"] and int(d["swap_times"]) == int(r["swap_times"])
                     and k_windows(d, k)]
            ok = [d for d in cands if feasible(r, d, k)]
            notes["infeasible"] += [f"{r['key']}:k{k}:{d['key']}" for d in cands if d not in ok]
            if ok:
                idx = int(rng.choice(len(ok)))
                runs.append(dict(base, group="Bsame_k", slot=k_slot("Bsame_k", k), donor_src=ok[idx]["key"]))
            else:
                notes["short"].append(f"{r['key']}:k{k}")
            runs.append(dict(base, group="Z_k", slot=k_slot("Z_k", k), donor_src=None))
    for run in runs:
        run["donor"] = synthetic_ep(run["ep"], run["slot"])
        run["fallback"] = False
    return runs, notes


def donor_payload_k(run, infos, records):
    """一条 kswap 局 → (starts=W_k, embs, meta) 或 (None, None, 问题)。"""
    r = infos[run["recv"]]
    W = k_windows(r, run["k"])
    rec = records[run["recv"]]
    dim = next(iter(rec.values())).shape[0]
    meta = {"pad": 0, "src_starts": None}
    if run["group"] == "Z_k":
        return W, np.zeros((len(W), dim), np.float32), meta
    d = infos[run["donor_src"]]
    embs, pad, srcs, why = aligned_rows(W, k_windows(d, run["k"]), records[run["donor_src"]], d["labels"]["es"])
    if embs is None:
        return None, None, why
    meta.update(pad=pad, src_starts=srcs)
    return W, embs, meta


def write_k_donor_files(folder, runs, infos, records):
    """写 kswap 合成供体并逐位读回核对；返回 (ok, 判定行, problems, done)。"""
    folder = Path(folder)
    problems, done, integrity, pads = [], [], 0, [0, 0]
    for run in runs:
        starts, embs, meta = donor_payload_k(run, infos, records)
        if starts is None:
            problems.append(f"{run['recv']}:{run['slot']}:{meta}")
            continue
        path = folder / record_name(run["task"], run["donor"])
        P._savez_atomic(path, start_frames=np.asarray(starts, np.int32), embs=embs.astype(np.float32),
                        counts=np.ones(len(starts), np.int32))
        with np.load(path) as back:
            ok = (np.asarray(back["start_frames"]).tolist() == [int(s) for s in starts]
                  and np.array_equal(np.asarray(back["embs"], np.float32), embs.astype(np.float32)))
        if not ok:
            problems.append(f"{run['recv']}:{run['slot']}:readback")
            integrity += 1
            continue
        if meta["pad"]:
            pads[0] += 1
            pads[1] += meta["pad"]
        done.append(dict(run, w_starts=[int(s) for s in starts], pad=meta["pad"], src_starts=meta["src_starts"]))
    ok = integrity == 0
    line = (f"DONOR_FILES={'PASS' if ok else 'FAIL'} mode=k files={len(done)} skipped={len(problems)} "
            f"integrity={integrity} pad_runs={pads[0]} pad_windows={pads[1]}")
    return ok, line, problems, done


def k_run_row(run, infos):
    row = run_row(run, infos)
    row.update(k=int(run["k"]), src_starts=run.get("src_starts"))
    return row


def kseed_runs(done):
    """每任务 default_rng(0) 从 kswap 局里抽 KSEED_PER_TASK 局，seed KSEED_SEED 再跑一次。"""
    out = []
    for task in TASKS:
        pool = [r for r in done if r["task"] == task]
        if not pool:
            continue
        rng = np.random.default_rng(PAIR_SEED)
        idx = sorted(rng.choice(len(pool), size=min(KSEED_PER_TASK, len(pool)), replace=False).tolist())
        out += [dict(pool[i], seed=KSEED_SEED) for i in idx]
    return out


def build_k_specs(b, pool_by_key, infos, done, seed_runs):
    """kswap / kseeds：同 (seed, k, 组) 的接收局装进同一 spec（每 spec ≤ SPEC_MAX、接收局唯一，F11），箱按序轮转给 worker。"""
    ablate = f"motion_swap_exact:{b.out / DONOR_DIR}"
    for stage, members in (("kswap", [dict(r, seed=MAIN_SEED) for r in done]), ("kseeds", seed_runs)):
        items = sorted((((r["seed"], r["k"], r["group"]), r["recv"], r) for r in members),
                       key=lambda x: (x[0], TASKS.index(x[2]["task"]), x[2]["ep"]))
        bins = P.pack(items)
        counter = {}
        for worker, (key, bin_) in b.round_robin(bins):
            seed, k, group = key
            i = counter.get((worker, key), 0)
            counter[(worker, key)] = i + 1
            runs = [item[2] for item in bin_]
            name = f"{stage}-{worker}-{group.replace('_', '')}{k}-s{seed}-{i:02d}"
            b.add(name, stage, worker, [pool_by_key[r["recv"]] for r in runs], seed=seed, ablate=ablate,
                  swap_map={r["recv"]: episode_key(r["task"], r["donor"]) for r in runs},
                  rows=[k_run_row(r, infos) for r in runs])


def cmd_donors_k(out):
    plan = P._read_json(out / "plan.json")
    pool, pool_by_key = load_pool(out.parent)
    infos = collect_n(plan, out / "record", pool_by_key)
    records = {k: load_record(v["record"]) for k, v in infos.items() if v.get("record")}
    runs, notes = [], {}
    for task in TASKS:
        print(recv_line(task, infos), flush=True)
        r, notes[task] = assign_k_groups(infos, task, records=records)
        runs += r
    ok, line, problems, done = write_k_donor_files(out / DONOR_DIR, runs, infos, records)
    for problem in problems:
        print(f"DONOR_PROBLEM {problem}", flush=True)
    print(line, flush=True)
    for task in TASKS:
        n = notes[task]
        per = {}
        for r in done:
            if r["task"] == task:
                per[(r["swap_times"], r["k"], r["group"])] = per.get((r["swap_times"], r["k"], r["group"]), 0) + 1
        print(f"FP_K_GROUPS task={task} short={len(n['short'])} no_wk={len(n['no_wk'])} infeasible={len(n['infeasible'])} "
              + " ".join(f"n{a}k{k}{g}={v}" for (a, k, g), v in sorted(per.items())), flush=True)
    seed_runs = kseed_runs(done)
    b = make_builder(out, plan["model"], plan["model_entry"], plan["run_tree"], plan.get("frac", XLA_MEM_FRACTION), pool,
                     roots=plan.get("roots"), workers=plan_workers(plan))
    build_k_specs(b, pool_by_key, infos, done, seed_runs)
    b.write(K_STAGES)
    plan["runs"] = [r for r in plan["runs"] if r["stage"] not in K_STAGES] + b.rows
    plan["k_recv"] = sorted({r["recv"] for r in done})
    plan["k_eligibility"] = {k: {"status": v["status"], "recv_ok": v.get("recv_ok"), "donor_ok": v.get("donor_ok"),
                                 "swap_times": v.get("swap_times"),
                                 "Wk": {str(kk): w for kk, w in ((v.get("labels") or {}).get("Wk") or {}).items()},
                                 "blind_k": {str(kk): w for kk, w in ((v.get("labels") or {}).get("blind_k") or {}).items()}}
                             for k, v in infos.items()}
    plan["k_group_notes"] = notes
    plan["k_donor_problems"] = {"count": len(problems), "integrity_ok": bool(ok), "items": problems}
    P._write_json(out / "plan.json", plan)
    counts = {}
    for row in b.rows:
        counts[f"{row['stage']}:{row['group']}"] = counts.get(f"{row['stage']}:{row['group']}", 0) + 1
    print("FPS_K_SPECS " + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f" total={len(b.rows)} specs={len(b.specs)}", flush=True)
    print(f"DONORS_K={'PASS' if ok else 'FAIL'} problems={len(problems)} runs={len(done)}", flush=True)
    return 0 if ok else 1


def add_root_args(p):
    """F5 路径根参数（specs 用；scan / pool 只用 --bench-src 找官方 metadata）。"""
    p.add_argument("--main", default=None, help="替换 spec 里以旧 MAIN 开头的路径")
    p.add_argument("--store", default=None, help="v1-store 根（缓存、MMEVLA_V1_STORE 缺省、spec 里旧 STORE 前缀）")
    p.add_argument("--bench-src", default=None, help="robomme_benchmark/src（PYTHONPATH 与官方 metadata）")
    p.add_argument("--policy-python", default=None)
    p.add_argument("--sim-python", default=None)
    p.add_argument("--stage2-python", default=None)
    p.add_argument("--mmevla-store", default=None, help="MMEVLA_V1_STORE（缺省 = --store）")
    p.add_argument("--log-dir", default=None, help="launch_stage.sh 的日志目录（缺省 <store>/logs/fp-swap-1110）")
    p.add_argument("--release", action="store_true", help="运行树是 release 目录（无 .git），env.sh 改查目录名")
    p.add_argument("--workers", default=None, help="名=GPU,…（缺省 w00=0,w01=0,w10=1,w11=1；--gl-store 时 gl00..gl15=0）")
    p.add_argument("--gl-store", default=None,
                   help="GreatLakes 预设：store/解释器/MMEVLA_V1_STORE 取该目录，--release，16 worker，bench_src 取运行树")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("--out", required=True)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--seed0", type=int, required=True)
    p.add_argument("--count", type=int, default=SCAN_COUNT)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--difficulties", default=",".join(DIFFICULTIES), help="逗号列表，按 seed 序号轮转（缺省 easy,medium）")
    p.add_argument("--tag", default=None, help="扫描文件后缀（缺省：难度非 easy,medium 时取难度名拼接）")
    p.add_argument("--bench-src", default=None, help="robomme_benchmark/src（找官方 metadata）")
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("pool")
    p.add_argument("--out", required=True)
    p.add_argument("--per-task", type=int, default=PER_TASK)
    p.add_argument("--by-swap-times", action="store_true", help="每任务按 n=1/2/3 各取 --per-n 局（读全部扫描文件）")
    p.add_argument("--per-n", type=int, default=PER_N)
    p.add_argument("--bench-src", default=None, help="robomme_benchmark/src（找官方 metadata）")
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("specs")
    p.add_argument("--out", required=True, help="<O>/<M>")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--commit", required=True, help="运行工作树应处的 commit（env.sh 核对）")
    p.add_argument("--run-tree", default=str(RUN_TREE))
    p.add_argument("--frac", type=float, default=XLA_MEM_FRACTION)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--policy-config", default=None)
    p.add_argument("--model-kind", default=None)
    p.add_argument("--target-contract", default=None)
    p.add_argument("--extra", default=None, help="并入 spec 的额外键（JSON 对象）")
    p.add_argument("--overwrite", action="store_true")
    add_root_args(p)
    p = sub.add_parser("donors")
    p.add_argument("--by-k", action="store_true", help="按第 k 次交换生成 Bsame_k / Z_k 与 kswap、kseeds 阶段")
    p.add_argument("--out", required=True, help="<O>/<M>")
    p.add_argument("--wan-gate", type=int, nargs="?", const=WAN_GATE_DEFAULT, default=None,
                   help="某任务接收局 < 该值（缺省 8）时该任务只跑 N、Z、S0、Zall")
    p.add_argument("--smoke", action="store_true", help="只处理 smoke 阶段录制")
    args = parser.parse_args(argv)
    return {"scan": cmd_scan, "pool": cmd_pool, "specs": cmd_specs, "donors": cmd_donors}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
