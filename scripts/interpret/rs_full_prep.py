#!/usr/bin/env python3
"""RouteStick 第一个子目标 motion 交换「全因子」的准备脚本（计划 1109-routestick-swap-full-factorial-plan.html
第二版：第一部分二、三节；第二部分一、二、四节；主代理追加：不跑官方 50 局重评，不生成 repro 阶段）。

RouteStick 没有单步局（easy length=[2,3]），所以只用「恰好两步」的局，只换第一个子目标对应的示范 motion 窗口
（起点 0、32；W0 组只换起点 0），只读第一步。

第一步类型 = (起点 index 0..4, 左右, 绕向)：
  起点 index 指 route_button_indices=[0,2,4,6,8] 中的下标；左右用 side ∈ {+1,-1} 表示下标增减。
  side=+1 记 "L"、side=-1 记 "R"：与环境 _stick_side 的措辞一致（目标 y 大于前一目标 y 记 left；theta ∈ [−30°,30°]
  时列号越大 y 越大，所以下标增大 = left）。绕向 cw/ccw 即环境 swing_directions 的 clockwise/counterclockwise。
  端点只能单向，共 8 种（起点，左右）× 2 绕向 = 16 种；类型序号 k 按 起点升序 → side（−1 在前）→ cw 在前 排出。

局号编排（写进 pool.json）：类型 k 的 3 局按 env_seed 升序，第 1 局 = R 局号 2k，第 2 局 = R 局号 2k+1（R 共 0–31），
第 3 局 = S 局号 32+k（S 共 32–47）。

四个子命令（只用 CPU）：
  replay-check [--seeds 20]
          在 official-sim 解释器里用 gym.make 直接建真实 RouteStick 环境（obs_mode=state、不渲染），reset 后读
          selected_buttons 对应格号、swing_directions 与 9 个目标格的 xy，与复算函数 replay_route 逐项比较。
          判定行 RS_REPLAY=PASS|FAIL seeds=<n> mismatches=<k>（另打印 RS_REPLAY_XY maxerr=<m>，目标坐标的最大偏差）。
  pool    --out <O>
          新扫 env_seed 1400000 起、步长 100、共 4000 个，跳过 RouteStick train/test/val 清单 seed，只留 steps==2；
          每类型取 3 局。写 pool.json、metadata/record_dataset_RouteStick_metadata.json、manifest.json、pairs.json；
          判定行 RS_POOL=PASS eps=48 types=16 overlap=0 与
          RS_PAIRS=PASS A=32 Bsw=32 Bside=48 C=96 D1=96 D2=208 W0=80 seedcheck=60。
  specs   --out <O>/<M> --model motion|wan --commit <sha> [--checkpoint …]
          smoke（w00 顺序：N 2 局 → donors --smoke → 交换 2 局 → 置零 1 局）、multieq（12 个交换局，一局一服务端 vs 连跑）、
          n（R+S 48 局，录 motion）三阶段的 spec、清单、队列、env.sh、launch_stage.sh、plan.json；会话前缀 rsf-。
  donors  --out <O>/<M> [--wan-gate 8] [--smoke]
          读 n 录制：donor_trim/（S 16 局，起点 0、32 两行）、donor_w0/（S 16 局，只留起点 0）、zero/ 与 random/（R 32 局，
          两行同形全零 / 逐行同范数随机，种子 = 局号）；读回校验，打印 DONOR_TRIM、DONOR_DISTINCT（S 集两行编码两两不同）；
          再按 N 结果（接收局原样整局成功才进）生成 swap（seed 42：Z、Rnd、A、Bsw、Bside、C、D1、D2）、w0（seed 42，
          donor_w0）、seeds（seed 1、2，donor_trim）三阶段。--wan-gate 8：R 原样成功 < 8 时只生成 Z、Rnd。

spec、队列、Builder、装箱等与任务无关的部分 import 自 pl_full_prep（只读，不改它）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

import numpy as np

from scripts.evaluation import manifest as M
from scripts.interpret import pl_full_prep as P
from scripts.interpret.pl_swap_prep import BENCH_SRC, SPEC_TEMPLATE, STORE, XLA_MEM_FRACTION

# ---------------------------------------------------------------- 常量
TASK = "RouteStick"
DIFFICULTY = "easy"
CONFIGS = {"easy": ((2, 3), False), "medium": ((4, 5), False), "hard": ((4, 7), True)}   # length、backtrack
ROUTE = (0, 2, 4, 6, 8)             # route_button_indices：目标格
POSTS = (1, 3, 5, 7)                # 立柱格
GRID_CENTER = (-0.1, 0.0)
SPACING = 0.07
N_COLS = 9
SCAN_START, SCAN_STEP, SCAN_COUNT = 1400000, 100, 4000
N_TYPES = 16
N_RECV = 32                         # R 局号 0–31
S_OFFSET = 32                       # S 局号 = 32 + 类型序号
N_EPS = 48
PAIR_SEED = 0
SEEDCHECK_PER_GROUP = 10
SEEDCHECK_SEEDS = (1, 2)
MAIN_SEED = 42
PAIR_GROUPS = ("A", "Bsw", "Bside", "C", "D1", "D2")
SWAP_GROUPS = ("Z", "Rnd") + PAIR_GROUPS
EXPECTED_PAIRS = {"A": 32, "Bsw": 32, "Bside": 48, "C": 96, "D1": 96, "D2": 208}
D2_IN, D2_OUT = 144, 64
W0_FROM = ("Bsw", "Bside")
STARTS_TRIM = (0, 32)               # 第一步覆盖的两个示范窗口起点
STARTS_W0 = (0,)
WAN_GATE_DEFAULT = 8
MULTIEQ_PER_GROUP = 2
STAGES = ("smoke", "multieq", "n", "swap", "w0", "seeds")
SMOKE_R, SMOKE_S = 0, 33            # 类型 0 (0,L,cw) 的 R 第 1 局 与 类型 1 (0,L,ccw) 的 S 局：互为 Bsw
RUN_TREE = STORE / "worktrees/rs-full-run"
LOG_DIR = STORE / "logs/rs-swap-full-1109"
META_NAME = f"record_dataset_{TASK}_metadata.json"
OFFICIAL_META = BENCH_SRC / "robomme/env_metadata"
SIDE_LABEL = {1: "L", -1: "R"}
SWING_SHORT = {"clockwise": "cw", "counterclockwise": "ccw"}
ABLATE_DIR = {"Z": "zero", "Rnd": "random", "A": "donor_trim", "Bsw": "donor_trim", "Bside": "donor_trim",
              "C": "donor_trim", "D1": "donor_trim", "D2": "donor_trim", "W0": "donor_w0"}

MODELS = {
    "motion": dict(P.MODELS["motion"]),
    "wan": {"checkpoint": None, "policy_config": None, "model_kind": None, "target_contract": None},
}


def episode_key(ep):
    return f"{TASK}:{int(ep)}"


def record_name(ep):
    return f"{TASK}_ep{int(ep)}.npz"


# ---------------------------------------------------------------- 复算与几何
def replay_route(env_seed, difficulty=DIFFICULTY):
    """按 RouteStick._load_scene 的随机数顺序复算：theta=rand(1)·60−30 度 → 4 根立柱各 rand(3) 颜色 →
    steps=randint(lo,hi+1) → generate_dynamic_walk（start=randint(0,5)；每步候选邻居，不许回头时剔除上一格、
    无路才回头，randint 选）→ 每步 rand(1)<0.5 为 clockwise。返回 dict。"""
    import torch
    (lo, hi), backtrack = CONFIGS[difficulty]
    g = torch.Generator()
    g.manual_seed(int(env_seed))
    theta = math.radians((torch.rand(1, generator=g).item() * 60) - 30)
    for _ in POSTS:
        torch.rand(3, generator=g)
    steps = int(torch.randint(lo, hi + 1, (1,), generator=g).item())
    idxs = [torch.randint(0, len(ROUTE), (1,), generator=g).item()]
    for _ in range(steps):
        cur = idxs[-1]
        prev = idxs[-2] if len(idxs) > 1 else None
        neighbors = [n for n in (cur - 1, cur + 1) if 0 <= n < len(ROUTE)]
        if backtrack or prev is None:
            cands = neighbors
        else:
            cands = [n for n in neighbors if n != prev] or neighbors
        idxs.append(cands[torch.randint(0, len(cands), (1,), generator=g).item()])
    swings = ["clockwise" if torch.rand(1, generator=g).item() < 0.5 else "counterclockwise" for _ in range(steps)]
    return {"env_seed": int(env_seed), "theta": float(theta), "steps": steps, "route_idx": [int(i) for i in idxs],
            "cells": [ROUTE[int(i)] for i in idxs], "swings": swings}


def cell_xy(cell, theta):
    """格中心世界坐标（_load_scene：1 行 9 列、间距 0.07、中心 (−0.1,0)，再整体绕原点转 theta）。"""
    x0 = GRID_CENTER[0]
    y0 = GRID_CENTER[1] + (int(cell) - (N_COLS - 1) / 2) * SPACING
    return (x0 * math.cos(theta) - y0 * math.sin(theta), x0 * math.sin(theta) + y0 * math.cos(theta))


def step1_type(route):
    """(起点 index, side ±1, 'cw'|'ccw')。"""
    s, n = int(route["route_idx"][0]), int(route["route_idx"][1])
    return (s, n - s, SWING_SHORT[route["swings"][0]])


def all_types():
    """16 种第一步类型，按 起点升序 → side（−1 在前）→ cw 在前。"""
    out = []
    for s in range(len(ROUTE)):
        for side in (-1, 1):
            if 0 <= s + side < len(ROUTE):
                for sw in ("cw", "ccw"):
                    out.append((s, side, sw))
    return out


def type_label(t):
    return f"{t[0]}{SIDE_LABEL[t[1]]}-{t[2]}"


def target_cell(start_idx, side):
    """从起点 index 按 side 走一步到达的目标格号；出界返回 None。"""
    j = int(start_idx) + int(side)
    return ROUTE[j] if 0 <= j < len(ROUTE) else None


# ---------------------------------------------------------------- replay-check
def real_route(env_seed, difficulty=DIFFICULTY):
    """建真实 RouteStick 环境（与评估同一注册类），reset 后读路线、绕向与 9 个目标格 xy。"""
    import gymnasium as gym
    import robomme.robomme_env  # noqa: F401  注册任务
    env = gym.make(TASK, seed=int(env_seed), difficulty=difficulty, obs_mode="state", render_mode=None,
                   control_mode="pd_joint_pos")
    try:
        env.reset()
        u = env.unwrapped
        ids = {id(b): i for i, b in enumerate(u.buttons_grid)}
        cells = [ids[id(b)] for b in u.selected_buttons]
        xy = []
        for b in u.buttons_grid:
            p = b.pose.p
            p = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
            p = np.asarray(p, np.float64).reshape(-1)
            xy.append([float(p[0]), float(p[1])])
        return {"cells": cells, "swings": list(u.swing_directions), "xy": xy}
    finally:
        env.close()


def cmd_replay_check(args):
    seeds = [SCAN_START + SCAN_STEP * i for i in range(args.seeds)]
    mismatches, maxerr = 0, 0.0
    for seed in seeds:
        rep = replay_route(seed)
        real = real_route(seed)
        err = max(math.hypot(real["xy"][c][0] - cell_xy(c, rep["theta"])[0],
                             real["xy"][c][1] - cell_xy(c, rep["theta"])[1]) for c in range(N_COLS))
        maxerr = max(maxerr, err)
        ok = rep["cells"] == real["cells"] and rep["swings"] == real["swings"] and err < 1e-5
        mismatches += 0 if ok else 1
        print(f"REPLAY seed={seed} replay={rep['cells']}/{[SWING_SHORT[s] for s in rep['swings']]} "
              f"real={real['cells']}/{[SWING_SHORT.get(s, s) for s in real['swings']]} xyerr={err:.2e} ok={int(ok)}",
              flush=True)
    print(f"RS_REPLAY_XY maxerr={maxerr:.2e}", flush=True)
    print(f"RS_REPLAY={'PASS' if mismatches == 0 else 'FAIL'} seeds={len(seeds)} mismatches={mismatches}", flush=True)
    return 0 if mismatches == 0 else 1


# ---------------------------------------------------------------- pool
def scan_seeds():
    return [SCAN_START + SCAN_STEP * i for i in range(SCAN_COUNT)]


def split_metadata_seeds(dataset):
    data = json.loads((OFFICIAL_META / dataset / META_NAME).read_text())
    return {int(r["seed"]) for r in data["records"]}


def choose_pool(two_step):
    """two_step: [route dict]（steps==2）。每类型按 env_seed 升序取 3 局，返回 (48 局, 每类型可用数)。"""
    types = all_types()
    index = {t: k for k, t in enumerate(types)}
    by_type = {t: [] for t in types}
    for route in sorted(two_step, key=lambda r: r["env_seed"]):
        if route["steps"] != 2:
            raise ValueError(f"env_seed {route['env_seed']} 不是两步局")
        by_type[step1_type(route)].append(route)
    short = [type_label(t) for t, v in by_type.items() if len(v) < 3]
    if short:
        raise SystemExit(f"这些类型不足 3 局: {short}")
    episodes = []
    for t in types:
        k = index[t]
        for pick, (ep, role) in enumerate(((2 * k, "R"), (2 * k + 1, "R"), (S_OFFSET + k, "S"))):
            r = by_type[t][pick]
            episodes.append({"episode_id": ep, "role": role, "type": k, "type_label": type_label(t),
                             "env_seed": r["env_seed"], "path": r["cells"], "route_idx": r["route_idx"],
                             "swings": r["swings"], "theta": r["theta"], "start": t[0], "side": t[1],
                             "swing": t[2], "target": target_cell(t[0], t[1])})
    episodes.sort(key=lambda e: e["episode_id"])
    return episodes, {type_label(t): len(v) for t, v in by_type.items()}


def metadata_payload(episodes):
    records = [{"task": TASK, "episode": e["episode_id"], "seed": e["env_seed"], "difficulty": DIFFICULTY}
               for e in episodes]
    return {"env_id": TASK, "record_count": len(records), "records": records}


def manifest_for(episodes, meta_sha, chunk=P.CHUNK):
    rows = [{"task_name": TASK, "episode_id": int(e["episode_id"]), "env_seed": int(e["env_seed"]),
             "difficulty": DIFFICULTY} for e in episodes]
    return M.make_manifest(rows, metadata_sha256={META_NAME: meta_sha}, chunk_size=chunk, official=False)


def make_pair(recv, donor, rt, dt):
    own = target_cell(rt[0], rt[1])
    return {"recv": recv, "donor": donor, "recv_type": list(rt), "donor_type": list(dt),
            "recv_start": rt[0], "recv_side": rt[1], "recv_swing": rt[2],
            "donor_start": dt[0], "donor_side": dt[1], "donor_swing": dt[2],
            "own": own, "donor_abs": target_cell(dt[0], dt[1]), "donor_dir_cell": target_cell(rt[0], dt[1])}


def pair_group(rt, dt):
    same_start, same_side, same_swing = rt[0] == dt[0], rt[1] == dt[1], rt[2] == dt[2]
    if same_start and same_side:
        return "A" if same_swing else "Bsw"
    if same_start:
        return "Bside"
    if same_side:
        return "C" if same_swing else "D1"
    return "D2"


def build_pairs(types=None):
    """接收 R 局 2k、2k+1（类型 k）× 供体 S 局 32+j（类型 j）全配对。D2 全跑不抽样，标盘内 / 出盘；
    W0 = Bsw + Bside 同批配对；seed 复核从六组各抽 10 对（每组 default_rng(0)）。"""
    types = types or all_types()
    rows = {g: [] for g in PAIR_GROUPS}
    for k, rt in enumerate(types):
        for recv in (2 * k, 2 * k + 1):
            for j, dt in enumerate(types):
                pair = make_pair(recv, S_OFFSET + j, rt, dt)
                group = pair_group(rt, dt)
                if group == "D2":
                    pair["stratum"] = "in" if pair["donor_dir_cell"] is not None else "out"
                rows[group].append(pair)
    w0 = [dict(p, w0_from=g) for g in W0_FROM for p in rows[g]]
    seedcheck = {}
    for group in PAIR_GROUPS:
        rng = np.random.default_rng(PAIR_SEED)
        pool = rows[group]
        idx = sorted(rng.choice(len(pool), size=SEEDCHECK_PER_GROUP, replace=False).tolist())
        seedcheck[group] = [pool[i] for i in idx]
    return {"types": [{"type": k, "label": type_label(t), "start": t[0], "side": t[1], "swing": t[2],
                       "target": target_cell(t[0], t[1])} for k, t in enumerate(types)],
            "counts": {g: len(rows[g]) for g in PAIR_GROUPS},
            "d2_strata": {s: sum(1 for p in rows["D2"] if p["stratum"] == s) for s in ("in", "out")},
            "all": rows, "w0": w0, "seedcheck": seedcheck, "seedcheck_seeds": list(SEEDCHECK_SEEDS),
            "rng_seed": PAIR_SEED}


def pairs_line(pairs):
    c = pairs["counts"]
    seedcheck = sum(len(v) for v in pairs["seedcheck"].values())
    ok = (all(c[g] == EXPECTED_PAIRS[g] for g in PAIR_GROUPS) and pairs["d2_strata"] == {"in": D2_IN, "out": D2_OUT}
          and len(pairs["w0"]) == c["Bsw"] + c["Bside"] == 80 and seedcheck == SEEDCHECK_PER_GROUP * len(PAIR_GROUPS))
    return (f"RS_PAIRS={'PASS' if ok else 'FAIL'} A={c['A']} Bsw={c['Bsw']} Bside={c['Bside']} C={c['C']} "
            f"D1={c['D1']} D2={c['D2']} W0={len(pairs['w0'])} seedcheck={seedcheck}"), ok


def cmd_pool(args):
    out = Path(args.out).resolve()
    split_seeds = {ds: split_metadata_seeds(ds) for ds in ("train", "test", "val")}
    banned = set().union(*split_seeds.values())
    seeds = [s for s in scan_seeds() if s not in banned]
    two, hist = [], {}
    for i, seed in enumerate(seeds):
        route = replay_route(seed)
        hist[route["steps"]] = hist.get(route["steps"], 0) + 1
        if route["steps"] == 2:
            two.append(route)
        if (i + 1) % 1000 == 0:
            print(f"SCAN {i + 1}/{len(seeds)} two_step={len(two)}", flush=True)
    episodes, per_type = choose_pool(two)
    chosen = {e["env_seed"] for e in episodes}
    overlap = {ds: sorted(chosen & s) for ds, s in split_seeds.items()}
    n_overlap = sum(len(v) for v in overlap.values())
    recheck = sum(1 for e in episodes if replay_route(e["env_seed"])["cells"] != e["path"])
    types_covered = len({e["type"] for e in episodes if e["role"] == "R"})
    meta = metadata_payload(episodes)
    meta_bytes = P._json_bytes(meta)
    meta_sha = hashlib.sha256(meta_bytes).hexdigest()
    P._write_once(out / "metadata" / META_NAME, meta_bytes, args.overwrite)
    manifest = manifest_for(episodes, meta_sha)
    P._write_once(out / "manifest.json", P._json_bytes(manifest), args.overwrite)
    pool = {"task": TASK, "difficulty": DIFFICULTY,
            "scan": {"start": SCAN_START, "step": SCAN_STEP, "count": SCAN_COUNT, "skipped_split": SCAN_COUNT - len(seeds),
                     "scanned": len(seeds), "two_step": len(two), "steps_hist": {str(k): v for k, v in sorted(hist.items())}},
            "numbering": "类型 k 的 3 局按 env_seed 升序：R 局号 2k、2k+1，S 局号 32+k",
            "side_note": "side=+1 记 L（下标增大，环境 _stick_side 措辞 left），side=-1 记 R",
            "per_type_two_step": per_type, "overlap": overlap, "recheck_mismatch": recheck,
            "metadata_sha256": meta_sha, "manifest_seal": manifest["sha256"], "episodes": episodes}
    P._write_once(out / "pool.json", P._json_bytes(pool), args.overwrite)
    pairs = build_pairs()
    P._write_once(out / "pairs.json", P._json_bytes(pairs), args.overwrite)
    print(f"SCAN_DONE scanned={len(seeds)} two_step={len(two)} min_per_type={min(per_type.values())}", flush=True)
    ok_pool = (len(episodes) == N_EPS and types_covered == N_TYPES and n_overlap == 0 and recheck == 0
               and all(len(e["path"]) == 3 for e in episodes))
    print(f"RS_POOL={'PASS' if ok_pool else 'FAIL'} eps={len(episodes)} types={types_covered} overlap={n_overlap}",
          flush=True)
    line, ok_pairs = pairs_line(pairs)
    print(line, flush=True)
    return 0 if ok_pool and ok_pairs else 1


# ---------------------------------------------------------------- specs
QUEUE_HEAD = P.QUEUE_HEAD.replace("pl_full_prep.py", "rs_full_prep.py")
QUEUE_TAIL = P.QUEUE_TAIL
ENV_TEMPLATE = P.ENV_TEMPLATE.replace("pl_full_prep.py", "rs_full_prep.py")

LAUNCH_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/interpret/rs_full_prep.py specs 生成。用法：bash launch_stage.sh <{stages}>
# 为该阶段有 spec 的 worker 各起一个 detached tmux（rsf-{model}-<阶段>-<worker>），日志 {logs}/rsf-{model}-<阶段>-<worker>.log。
# 顺序：smoke → n → （rs_full_prep donors）→ multieq → swap → w0 → seeds；multieq 与 swap/w0/seeds 要先有供体文件。
O={out}
st=$1
case "$st" in {stages}) ;; *) echo "未知阶段 $st"; exit 2;; esac
mkdir -p {logs}
for w in {workers}; do
  q=$O/queues/$st-$w.sh
  [ -f "$q" ] && grep -q '^run_spec ' "$q" || continue
  name=rsf-{model}-$st-$w; log={logs}/$name.log
  if tmux has-session -t "=$name" 2>/dev/null; then echo "SKIP_EXISTS $name"; continue; fi
  if [ -e "$log" ]; then echo "SKIP_LOG_EXISTS $log（重跑前先把旧日志改名留档）"; continue; fi
  tmux new-session -d -s "$name" "set -o pipefail; PYTHONUNBUFFERED=1 bash $q 2>&1 | tee $log; echo \\"EXIT_CODE=\\$?\\" >> $log"
  echo "STARTED $name $log"
done
"""


class Builder(P.Builder):
    """沿用 pl_full_prep.Builder（spec 键与上一轮一致），只换清单（RouteStick）与队列模板。"""

    def add(self, name, stage, worker, episodes, **kw):
        if kw.get("manifest") is None:
            kw["manifest"] = manifest_for(episodes, self.meta_sha, chunk=kw.pop("chunk", P.CHUNK))
        else:
            kw.pop("chunk", None)
        return self.add_spec(name, stage, worker, episodes, **kw)

    def write(self, stages):
        for name, (spec, manifest, stage, worker) in self.specs.items():
            P._write_json(self.out / "manifests" / f"{name}.json", manifest)
            P._write_json(self.out / "specs" / f"{name}.json", spec)
        written = []
        (self.out / "queues").mkdir(parents=True, exist_ok=True)
        for stage in stages:
            for worker in P.WORKERS:
                lines = self.queue.get((stage, worker), [])
                n_spec = sum(1 for x in lines if x.startswith("run_spec "))
                path = self.out / "queues" / f"{stage}-{worker}.sh"
                text = [QUEUE_HEAD.format(stage=stage, worker=worker, cuda=P.WORKER_CUDA[worker], n=n_spec,
                                          env=shlex.quote(str(self.out / "env.sh")), frac=self.frac)]
                text += lines
                text.append(QUEUE_TAIL.format(stage=stage, worker=worker, n=n_spec))
                path.write_text("".join(text))
                path.chmod(0o755)
                written.append(path)
        return written


def run_row(group, recv, donor, pool_by_id, **extra):
    re_ = pool_by_id[int(recv)]
    row = {"group": group, "ep": int(recv), "donor": None if donor is None else int(donor),
           "path": list(re_["path"]), "route_idx": list(re_["route_idx"]), "swings": list(re_["swings"]),
           "theta": float(re_["theta"]), "recv_type": [re_["start"], re_["side"], re_["swing"]],
           "donor_path": None, "donor_type": None}
    if donor is not None and int(donor) != int(recv):
        d = pool_by_id[int(donor)]
        row["donor_path"] = list(d["path"])
        row["donor_type"] = [d["start"], d["side"], d["swing"]]
    row.update(extra)
    return row


def multieq_pairs(pairs):
    """六组各按表顺序取 2 对，12 对接收局两两不同。"""
    chosen, used = [], set()
    for group in PAIR_GROUPS:
        n = 0
        for p in pairs["all"][group]:
            if p["recv"] in used:
                continue
            chosen.append((group, p))
            used.add(p["recv"])
            n += 1
            if n == MULTIEQ_PER_GROUP:
                break
    return chosen


def build_stage_specs(b, pool_by_id, pairs, stages):
    out = b.out
    if "smoke" in stages:
        smoke = out / "smoke"
        eps = [pool_by_id[SMOKE_R], pool_by_id[SMOKE_S]]
        b.add("smoke-n", "smoke", "w00", eps, record=smoke / "record", record_dir=smoke / "record",
              rows=[run_row("N", e["episode_id"], None, pool_by_id) for e in eps])
        b.add_cmd("smoke", "w00", f"\"$py\" -m scripts.interpret.rs_full_prep donors --out {shlex.quote(str(out))} --smoke")
        b.add("smoke-swap", "smoke", "w00", eps, ablate=f"motion_swap_exact:{smoke / 'donor_trim'}",
              swap_map={episode_key(SMOKE_R): episode_key(SMOKE_S), episode_key(SMOKE_S): episode_key(SMOKE_R)},
              record_dir=smoke / "record",
              rows=[run_row("Bsw", SMOKE_R, SMOKE_S, pool_by_id), run_row("Bsw", SMOKE_S, SMOKE_R, pool_by_id)])
        b.add("smoke-zero", "smoke", "w00", [pool_by_id[SMOKE_R]], ablate=f"motion_swap_exact:{smoke / 'zero'}",
              swap_map={episode_key(SMOKE_R): episode_key(SMOKE_R)}, record_dir=smoke / "record",
              rows=[run_row("Z", SMOKE_R, SMOKE_R, pool_by_id)])
    if "multieq" in stages:
        chosen = multieq_pairs(pairs)
        ablate = f"motion_swap_exact:{out / 'donor_trim'}"
        single_workers = ("w00", "w01", "w11")
        for i, (group, p) in enumerate(chosen):
            worker = single_workers[i % len(single_workers)]
            b.add(f"multieq-single-{i:02d}", "multieq", worker, [pool_by_id[p["recv"]]], ablate=ablate,
                  swap_map={episode_key(p["recv"]): episode_key(p["donor"])}, chunk=1,
                  rows=[run_row(group, p["recv"], p["donor"], pool_by_id, variant="single", pair_index=i)])
        b.add("multieq-multi", "multieq", "w10", [pool_by_id[p["recv"]] for _, p in chosen], ablate=ablate,
              swap_map={episode_key(p["recv"]): episode_key(p["donor"]) for _, p in chosen},
              rows=[run_row(g, p["recv"], p["donor"], pool_by_id, variant="multi", pair_index=i)
                    for i, (g, p) in enumerate(chosen)])
    if "n" in stages:
        ids = sorted(pool_by_id)
        for w_i, worker in enumerate(P.ASSIGN_ORDER):
            mine = [pool_by_id[i] for i in ids[w_i::len(P.ASSIGN_ORDER)]]
            b.add(f"n-{worker}", "n", worker, mine, record=out / "record",
                  rows=[run_row("N", e["episode_id"], None, pool_by_id) for e in mine])


def load_pool(pool_root):
    pool = P._read_json(Path(pool_root) / "pool.json")
    pairs = P._read_json(Path(pool_root) / "pairs.json")
    pool_by_id = {int(e["episode_id"]): e for e in pool["episodes"]}
    if sorted(pool_by_id) != list(range(N_EPS)):
        raise SystemExit("pool.json 局号不是 0–47")
    return pool, pairs, pool_by_id


def make_builder(out, model_name, model, run_tree, frac, pool):
    pool_root = out.parent
    meta_dir = pool_root / "metadata"
    meta_sha = hashlib.sha256((meta_dir / META_NAME).read_bytes()).hexdigest()
    if meta_sha != pool["metadata_sha256"]:
        raise SystemExit("metadata 与 pool.json 记录的 sha256 不符")
    return Builder(out, pool_root, model_name, model, run_tree, frac, meta_dir, meta_sha)


def model_entry(name, overrides=None):
    entry = dict(MODELS[name])
    entry.update({k: v for k, v in (overrides or {}).items() if v is not None})
    missing = [k for k in ("checkpoint", "policy_config", "model_kind", "target_contract") if not entry.get(k)]
    if missing:
        raise SystemExit(f"模型 {name} 缺 {missing}：Wan 项须用 --checkpoint/--policy-config/--model-kind/--target-contract 给出")
    return entry


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
    pool, pairs, pool_by_id = load_pool(out.parent)
    plan_path = out / "plan.json"
    if plan_path.exists() and not args.overwrite:
        raise SystemExit(f"{plan_path} 已存在；确需重写 spec 加 --overwrite（不删任何运行目录）")
    out.mkdir(parents=True, exist_ok=True)
    b = make_builder(out, args.model, model, args.run_tree, args.frac, pool)
    stages = ("smoke", "multieq", "n")
    build_stage_specs(b, pool_by_id, pairs, stages)
    (out / "env.sh").write_text(ENV_TEMPLATE.format(frac=args.frac, run_tree=args.run_tree, commit=args.commit,
                                                    store=STORE, bench=BENCH_SRC, out=out,
                                                    py=SPEC_TEMPLATE["policy_python"], sim=SPEC_TEMPLATE["sim_python"]))
    launch = out / "launch_stage.sh"
    launch.write_text(LAUNCH_TEMPLATE.format(out=out, model=args.model, logs=LOG_DIR, workers=" ".join(P.WORKERS),
                                             stages="|".join(STAGES)))
    launch.chmod(0o755)
    queues = b.write(STAGES)   # swap/w0/seeds 先写空队列，donors 再覆盖
    plan = {"task": TASK, "model": args.model, "commit": args.commit, "run_tree": str(args.run_tree), "out": str(out),
            "pool_root": str(out.parent), "record_dir": str(out / "record"), "model_entry": model,
            "manifest_seal": pool["manifest_seal"], "frac": float(args.frac), "gate": None, "runs": b.rows}
    P._write_json(plan_path, plan)
    counts = {st: sum(1 for r in b.rows if r["stage"] == st) for st in stages}
    print("RSF_SPECS model=" + args.model + " " + " ".join(f"{k}={v}" for k, v in counts.items())
          + f" specs={len(b.specs)} queues={len(queues)}", flush=True)
    return 0


# ---------------------------------------------------------------- donors
def pick_rows(path, starts):
    """录制 npz 中起点 ∈ starts 的行（按 starts 顺序）→ ((embs[n,D], counts[n]), None)；缺任一行或重复返回 (None, 原因)。"""
    if not Path(path).is_file():
        return None, "missing_record"
    with np.load(path) as z:
        sf = np.asarray(z["start_frames"], np.int32).reshape(-1)
        embs = np.asarray(z["embs"], np.float32)
        counts = np.asarray(z["counts"], np.int32).reshape(-1) if "counts" in z.files else np.ones_like(sf)
    if embs.ndim != 2 or embs.shape[0] != sf.size:
        return None, "bad_shape"
    idx = []
    for s in starts:
        hit = np.flatnonzero(sf == int(s))
        if hit.size != 1:
            return None, f"start{s}_rows={hit.size}"
        idx.append(int(hit[0]))
    cnt = counts[idx].astype(np.int32) if counts.size == sf.size else np.ones(len(idx), np.int32)
    return (embs[idx].copy(), cnt), None


def random_rows(rows, ep):
    """逐行同 L2 范数、随机方向；一个 numpy 生成器（种子 = 局号）依次生成各行。"""
    rng = np.random.default_rng(int(ep))
    out = np.empty_like(rows, dtype=np.float32)
    for i, row in enumerate(rows):
        v = rng.standard_normal(row.shape).astype(np.float64)
        v *= float(np.linalg.norm(row.astype(np.float64))) / float(np.linalg.norm(v))
        out[i] = v.astype(np.float32)
    return out


def make_donor_files(base, donor_eps, recv_eps):
    """base/record → donor_trim/、donor_w0/（donor_eps）、zero/、random/（recv_eps）。返回 (ok, 行, problems, rows)。"""
    base = Path(base)
    problems, rows = [], {}
    stats = {"donor": 0, "w0": 0, "zero": 0, "random": 0, "rows": 0, "maxabs": 0.0, "norm_maxrel": 0.0}
    for ep in sorted(set(donor_eps) | set(recv_eps)):
        got, why = pick_rows(base / "record" / record_name(ep), STARTS_TRIM)
        if got is None:
            problems.append(f"ep{ep}:{why}")
            continue
        two, counts = got
        rows[ep] = two
        targets = []
        if ep in donor_eps:
            targets.append(("donor", base / "donor_trim", STARTS_TRIM, two, counts))
            targets.append(("w0", base / "donor_w0", STARTS_W0, two[:1], counts[:1]))
        if ep in recv_eps:
            targets.append(("zero", base / "zero", STARTS_TRIM, np.zeros_like(two), counts))
            targets.append(("random", base / "random", STARTS_TRIM, random_rows(two, ep), counts))
        for kind, folder, starts, emb, cnt in targets:
            path = folder / record_name(ep)
            P._savez_atomic(path, start_frames=np.asarray(starts, np.int32), embs=emb.astype(np.float32),
                            counts=np.asarray(cnt, np.int32))
            with np.load(path) as back:
                b_starts, b_embs = np.asarray(back["start_frames"]).tolist(), np.asarray(back["embs"], np.float32)
            stats["rows"] = max(stats["rows"], len(b_starts))
            ref = two[:len(starts)]
            if b_starts != list(starts) or b_embs.shape != ref.shape:
                problems.append(f"ep{ep}:{kind}_shape")
                continue
            if kind == "random":
                n0 = np.linalg.norm(ref.astype(np.float64), axis=1)
                n1 = np.linalg.norm(b_embs.astype(np.float64), axis=1)
                rel = float(np.max(np.abs(n1 - n0) / np.where(n0 > 0, n0, np.nan)))
                rel = rel if np.isfinite(rel) else float("inf")
                stats["norm_maxrel"] = max(stats["norm_maxrel"], rel)
                if not rel < 1e-5:
                    problems.append(f"ep{ep}:random_norm_rel={rel:g}")
            else:
                dev = float(np.max(np.abs(b_embs - (0.0 if kind == "zero" else ref))))
                stats["maxabs"] = max(stats["maxabs"], dev)
                if dev != 0.0:
                    problems.append(f"ep{ep}:{kind}_maxabs={dev}")
            stats[kind] += 1
    ok = not problems
    line = (f"DONOR_TRIM={'PASS' if ok else 'FAIL'} donors={stats['donor']} w0={stats['w0']} zeros={stats['zero']} "
            f"randoms={stats['random']} rows={stats['rows']} maxabs={stats['maxabs']:g} norm_maxrel={stats['norm_maxrel']:.2e}")
    return ok, line, problems, rows


def distinct_line(rows, donor_eps):
    """S 集两行编码（起点 0、32 拼起来）两两比较，逐位相同算一对。"""
    have = [ep for ep in sorted(donor_eps) if ep in rows]
    identical = []
    for i, a in enumerate(have):
        for b_ in have[i + 1:]:
            if float(np.max(np.abs(rows[a] - rows[b_]))) == 0.0:
                identical.append((a, b_))
    ok = not identical and len(have) == len(donor_eps)
    return f"DONOR_DISTINCT={'PASS' if ok else 'FAIL'} donors={len(have)} identical_pairs={len(identical)}", ok, identical


def n_results(plan):
    out = {}
    for row in plan["runs"]:
        if row["stage"] != "n":
            continue
        term = P.latest_terminal(row["run_root"], episode_key(row["ep"]))
        out[int(row["ep"])] = None if term is None else int(term.get("outcome") == "success")
    return out


def swap_runs(pairs, success, gate_few):
    """返回 ([(stage, group, recv, donor, seed, extra)], skipped)：接收局原样整局成功才进。"""
    runs, skipped = [], []
    ok = lambda r: success.get(int(r)) == 1
    def put(run):
        (runs if ok(run[2]) else skipped).append(run)
    for group in ("Z", "Rnd"):
        for k in range(N_RECV):
            put(("swap", group, k, k, MAIN_SEED, {}))
    if not gate_few:
        for group in PAIR_GROUPS:
            for p in pairs["all"][group]:
                extra = {"stratum": p["stratum"]} if "stratum" in p else {}
                put(("swap", group, p["recv"], p["donor"], MAIN_SEED, extra))
        for p in pairs["w0"]:
            put(("w0", "W0", p["recv"], p["donor"], MAIN_SEED, {"w0_from": p["w0_from"]}))
        for seed in SEEDCHECK_SEEDS:
            for group in PAIR_GROUPS:
                for p in pairs["seedcheck"][group]:
                    extra = {"stratum": p["stratum"]} if "stratum" in p else {}
                    put(("seeds", group, p["recv"], p["donor"], seed, extra))
    return runs, skipped


def build_swap_specs(b, pool_by_id, runs):
    for stage in ("swap", "w0", "seeds"):
        members = [r for r in runs if r[0] == stage]
        per_worker = {}
        for worker, run in P.round_robin(members):
            per_worker.setdefault(worker, []).append(run)
        for worker in P.WORKERS:
            items = [((run[4], ABLATE_DIR[run[1]]), run[2], run) for run in per_worker.get(worker, [])]
            for i, ((seed, kind), bin_) in enumerate(P.pack(items)):
                name = f"{stage}-{worker}-{kind}-s{seed}-{i:02d}"
                recvs = [item[1] for item in bin_]
                b.add(name, stage, worker, [pool_by_id[r] for r in recvs], seed=seed,
                      ablate=f"motion_swap_exact:{b.out / kind}",
                      swap_map={episode_key(item[2][2]): episode_key(item[2][3]) for item in bin_},
                      rows=[run_row(item[2][1], item[2][2], item[2][3], pool_by_id, **item[2][5]) for item in bin_])


def cmd_donors(args):
    out = Path(args.out).resolve()
    if args.smoke:
        ok, line, problems, _ = make_donor_files(out / "smoke", {SMOKE_R, SMOKE_S}, {SMOKE_R})
        for problem in problems:
            print(f"DONOR_PROBLEM {problem}", flush=True)
        print(line, flush=True)
        return 0 if ok else 1
    plan = P._read_json(out / "plan.json")
    pool, pairs, pool_by_id = load_pool(out.parent)
    success = n_results(plan)
    donor_eps = set(range(S_OFFSET, N_EPS))
    recv_eps = set(range(N_RECV))
    ok, line, problems, rows = make_donor_files(out, donor_eps, recv_eps)
    dline, dok, identical = distinct_line(rows, donor_eps)
    for problem in problems:
        print(f"DONOR_PROBLEM {problem}", flush=True)
    for a, b_ in identical:
        print(f"DONOR_IDENTICAL {a} {b_}", flush=True)
    print(line, flush=True)
    print(dline, flush=True)
    missing_n = sorted(ep for ep in range(N_EPS) if success.get(ep) is None)
    r_success = sum(1 for k in range(N_RECV) if success.get(k) == 1)
    s_success = sum(1 for k in donor_eps if success.get(k) == 1)
    gate_min = args.wan_gate
    gate_few = gate_min is not None and r_success < gate_min
    print(f"N_RESULT r_success={r_success}/{N_RECV} s_success={s_success}/{len(donor_eps)} missing={len(missing_n)} "
          f"gate={'few' if gate_few else ('ok' if gate_min is not None else 'off')}", flush=True)
    if not (ok and dok):
        print("DONORS_SPECS=SKIP reason=donor_files", flush=True)
        return 1
    runs, skipped = swap_runs(pairs, success, gate_few)
    b = make_builder(out, plan["model"], plan["model_entry"], plan["run_tree"], plan.get("frac", XLA_MEM_FRACTION), pool)
    build_swap_specs(b, pool_by_id, runs)
    b.write(("swap", "w0", "seeds"))
    plan["runs"] = [r for r in plan["runs"] if r["stage"] not in ("swap", "w0", "seeds")] + b.rows
    plan["gate"] = {"wan_gate": gate_min, "r_success": r_success, "few": gate_few, "min": gate_min,
                    "groups": ["N", "Z", "Rnd"] if gate_few else ["N", *SWAP_GROUPS, "W0"]}
    plan["skipped_recv_failed"] = [[r[0], r[1], r[2], r[3], r[4]] for r in skipped]
    plan["n_missing"] = missing_n
    P._write_json(out / "plan.json", plan)
    counts = {}
    for row in b.rows:
        counts[f"{row['stage']}:{row['group']}"] = counts.get(f"{row['stage']}:{row['group']}", 0) + 1
    print("RSF_SWAP_SPECS " + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f" total={len(b.rows)} specs={len(b.specs)} skipped={len(skipped)}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("replay-check")
    p.add_argument("--seeds", type=int, default=20)
    p = sub.add_parser("pool")
    p.add_argument("--out", required=True)
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
    p = sub.add_parser("donors")
    p.add_argument("--out", required=True, help="<O>/<M>")
    p.add_argument("--wan-gate", type=int, nargs="?", const=WAN_GATE_DEFAULT, default=None,
                   help="R 原样成功 < 该值（缺省 8）时只生成 Z、Rnd")
    p.add_argument("--smoke", action="store_true", help="只处理 smoke 阶段录制")
    args = parser.parse_args(argv)
    return {"replay-check": cmd_replay_check, "pool": cmd_pool, "specs": cmd_specs, "donors": cmd_donors}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
