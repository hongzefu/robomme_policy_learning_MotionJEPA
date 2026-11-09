#!/usr/bin/env python3
"""PatternLock 单步 motion token 交换的准备脚本（计划 1108-patternlock-single-step-motion-swap-plan.html 第二部分一）。

三个子命令：
  paths   按 PatternLock._load_scene 的随机数顺序复算 8 局路径，写 <out>/paths.json，末行
          PL_PATHS=PASS|FAIL eps=8 mismatches=<n>。需在 official-sim 解释器里运行（要 torch 与
          robomme.robomme_env.utils.adjacent），PYTHONPATH 含 robomme_benchmark/src。
  specs   生成全部 run_local spec、按阶段的 4 个 worker 队列脚本、env.sh 与 plan.json：
            <out>/specs/<组>-seed<s>-<worker>-m<片>.json
            <out>/queues/<阶段>-<worker>.sh       阶段 ∈ smoke1、smoke2、stage2、stage3a、stage3b
            <out>/env.sh                          队列脚本 source；核对运行工作树 HEAD == --commit 且 clean
            <out>/plan.json                       130 局逐条：阶段、worker、组、seed、局、供体、spec、attn_out、run_root
          运行产物目录 <out>/<组>/seed<s>/<worker>/{attn,run}：同一 run_root 只归一个 worker，
          不同 worker 绝不并发共用。清单用 1106 的 manifest-s0..s3（按 ep%16 分片，每局取含它的那份）。
  donors  读 <out>/record/PatternLock_ep<id>.npz，只留 start_frames==0 那一行写 <out>/donor_trim/（只对供体局），
          同形状全零写 <out>/zero/（8 局都写）；读回核对后打印
          DONOR_TRIM=PASS|FAIL donors=<n> zeros=<n> rows=<最大行数> maxabs=<最大偏差>。
          --episodes 5,20 只处理指定局（冒烟用）。

组：N 原样（seed 42 另加 record_motion）、Z 示范窗置零（motion_swap_exact:<out>/zero，自映射）、
A/B/C 交换（motion_swap_exact:<out>/donor_trim），每个 seed 一份。只用 CPU，不起模型与仿真。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import uuid

import numpy as np

MAIN = Path("/data/hongzefu/robomme_policy_learning_MotionJEPA")
STORE = MAIN / "v1-store"
REF_1106 = STORE / "evaluation/subgoal-rollout-attn-seed42-1106"
RUN_TREE = STORE / "worktrees/pl-swap-run"
BENCH_SRC = MAIN / "third_party/robomme_benchmark/src"
TASK = "PatternLock"
DIFFICULTY = "easy"
GRID = 3
LENGTH_RANGE = (2, 4)   # easy 档 config：grid 3、length [2,4]

# ep → (env_seed, 路径)；路径于 2026-11-08 用 official-sim 复算（计划第一部分第一节表）
EPISODES = {
    5: (650500, (1, 5)),
    28: (652800, (1, 5)),
    20: (652000, (1, 0)),
    48: (654800, (2, 1)),
    25: (652500, (4, 3)),
    44: (654400, (0, 1)),
    29: (652900, (7, 4)),
    32: (653200, (3, 0)),
}
EP_ORDER = (5, 28, 20, 48, 25, 44, 29, 32)
SEEDS = (42, 1, 2, 3, 4)
RECORD_SEED = 42
# 条件 → [(接收局, 供体局)]（计划第一部分第三节）
CONDITIONS = {
    "A": ((5, 28), (28, 5)),
    "B": ((20, 5), (5, 20), (28, 20)),
    "C": ((20, 25), (25, 48), (48, 20), (29, 32), (32, 29)),
}
GROUPS = ("N", "Z", "A", "B", "C")
WORKERS = ("w00", "w01", "w10", "w11")
WORKER_CUDA = {"w00": "0", "w01": "0", "w10": "1", "w11": "1"}
ASSIGN_ORDER = ("w00", "w10", "w01", "w11")   # 轮转时两卡交替
STAGES = ("smoke1", "smoke2", "stage2", "stage3a", "stage3b")
SMOKE_EPS = (5, 20)
MANIFEST_SHARDS = 4
XLA_MEM_FRACTION = 0.30

SPEC_TEMPLATE = {
    "policy_python": str(MAIN / ".venv/bin/python"),
    "sim_python": str(STORE / "envs/official-sim/bin/python"),
    "stage2_python": str(STORE / "envs/stage2/bin/python"),
    "purpose": "validation",
    "save_top_frames": 0,
    "save_prefix": False,
    "startup_timeout_seconds": 900,
    "rpc_timeout_seconds": 600,
    "checkpoint": str(STORE / "assets/vla-o16cc-59999/59999"),
    "policy_config": "mme_vla_suite_orig16_b128_60k",
    "model_kind": "stage2_pool4_concat",
    "target_contract": str(STORE / "evaluation/interpret-1103/contracts/runtime-mot.json"),
    "subgoal_trace": True,
    "save_attn": False,
}


# ---------------------------------------------------------------- 几何
def cell_rc(cell):
    return divmod(int(cell), GRID)


def direction(path):
    """第一步的 (Δ行, Δ列)。"""
    (r0, c0), (r1, c1) = cell_rc(path[0]), cell_rc(path[1])
    return (r1 - r0, c1 - c0)


def button_xy(cell):
    """easy 按钮中心世界坐标：x = −0.1 + (行 − 1)·0.1，y = (列 − 1)·0.1（PatternLock._load_scene）。"""
    r, c = cell_rc(cell)
    return (-0.1 + (r - 1) * 0.1, (c - 1) * 0.1)


def episode_key(ep):
    return f"{TASK}:{int(ep)}"


def record_name(ep):
    return f"{TASK}_ep{int(ep)}.npz"


# ---------------------------------------------------------------- paths
def _load_find_path():
    from robomme.robomme_env.utils.adjacent import find_path_0_to_8
    return find_path_0_to_8


def replay_path(env_seed, difficulty=DIFFICULTY, find_path=None):
    """按 PatternLock._load_scene 顺序复算：Generator(seed) → 循环 randperm(G*G)[:2] + find_path_0_to_8(diagonals=True)，
    直到路径长度落在档位区间（最多 1000 次，同源码）。"""
    import torch
    if difficulty != DIFFICULTY:
        raise ValueError("只支持 easy 档")
    find_path = find_path or _load_find_path()
    generator = torch.Generator()
    generator.manual_seed(int(env_seed))
    path = None
    for _ in range(1000):
        start, end = torch.randperm(GRID * GRID, generator=generator)[:2].tolist()
        path, _, _, _ = find_path(start=start, target=end, R=GRID, C=GRID, diagonals=True, generator=generator)
        if LENGTH_RANGE[0] <= len(path) <= LENGTH_RANGE[1]:
            break
    return [int(x) for x in path]


def cmd_paths(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, mismatches = {}, 0
    for ep in EP_ORDER:
        seed, expected = EPISODES[ep]
        got = replay_path(seed)
        ok = tuple(got) == tuple(expected) and len(got) == 2
        mismatches += 0 if ok else 1
        rows[str(ep)] = {"env_seed": seed, "path": got, "expected": list(expected), "m": len(got) - 1, "ok": ok}
        print(f"PATH ep={ep} env_seed={seed} path={got} expected={list(expected)} ok={int(ok)}", flush=True)
    _write_json(out / "paths.json", {"task": TASK, "difficulty": DIFFICULTY, "episodes": rows})
    line = f"PL_PATHS={'PASS' if mismatches == 0 else 'FAIL'} eps={len(EP_ORDER)} mismatches={mismatches}"
    print(line, flush=True)
    return 0 if mismatches == 0 else 1


# ---------------------------------------------------------------- 运行清单
def all_runs():
    """130 局：[(组, seed, 接收局, 供体局或 None)]，组内同 seed 接收局唯一。"""
    runs = []
    for seed in SEEDS:
        runs += [("N", seed, ep, None) for ep in EP_ORDER]
        runs += [("Z", seed, ep, ep) for ep in EP_ORDER]
        for group in ("A", "B", "C"):
            runs += [(group, seed, ep, donor) for ep, donor in CONDITIONS[group]]
    return runs


def stage_of(run):
    group, seed, ep, donor = run
    if seed == RECORD_SEED and group == "N":
        return "smoke1" if ep in SMOKE_EPS else "stage2"
    if seed == RECORD_SEED and ((group == "B" and (ep, donor) == (20, 5)) or (group == "Z" and ep == 20)):
        return "smoke2"
    if seed == 1 and group == "N":
        return "stage3a"
    return "stage3b"


def assign(runs):
    """各阶段 (run) 依序轮转到 4 个 worker（两卡交替）。返回 [(阶段, worker, run)]。"""
    out = []
    for stage in STAGES:
        members = [run for run in runs if stage_of(run) == stage]
        for i, run in enumerate(members):
            out.append((stage, ASSIGN_ORDER[i % len(ASSIGN_ORDER)], run))
    return out


def manifest_shards(manifest_dir):
    """ep → (片号, manifest 路径)；核对 env_seed 与表一致。"""
    found = {}
    for shard in range(MANIFEST_SHARDS):
        path = Path(manifest_dir) / f"manifest-s{shard}.json"
        data = json.loads(path.read_text())
        for episode in data["episodes"]:
            if episode.get("task_name") != TASK or int(episode["episode_id"]) not in EPISODES:
                continue
            ep = int(episode["episode_id"])
            if int(episode["env_seed"]) != EPISODES[ep][0] or episode.get("difficulty", DIFFICULTY) != DIFFICULTY:
                raise SystemExit(f"manifest {path} 中 {episode_key(ep)} 的 env_seed/难度与表不符: {episode}")
            if ep in found:
                raise SystemExit(f"{episode_key(ep)} 同时出现在 {found[ep][1]} 与 {path}")
            found[ep] = (shard, str(path))
    missing = sorted(set(EPISODES) - set(found))
    if missing:
        raise SystemExit(f"1106 manifest 缺这些局: {missing}")
    return found


def ablate_for(group, out):
    if group == "N":
        return "none"
    if group == "Z":
        return f"motion_swap_exact:{out / 'zero'}"
    return f"motion_swap_exact:{out / 'donor_trim'}"


def run_dirs(out, group, seed, worker):
    base = Path(out) / group / f"seed{seed}" / worker
    return base / "attn", base / "run"


def write_specs(out, plan_rows, shards, run_tree=RUN_TREE, frac=XLA_MEM_FRACTION):
    """plan_rows: [(阶段, worker, run)]。每个 (组, seed, worker, 片) 一份 spec；交换 spec 的 swap_map 恰为
    排在该 spec 下的接收局。返回 {(组, seed, worker, 片): spec 路径}。"""
    out = Path(out)
    grouped = {}
    for stage, worker, (group, seed, ep, donor) in plan_rows:
        key = (group, seed, worker, shards[ep][0])
        grouped.setdefault(key, []).append((ep, donor))
    paths = {}
    (out / "specs").mkdir(parents=True, exist_ok=True)
    for (group, seed, worker, shard), members in sorted(grouped.items(), key=lambda kv: (GROUPS.index(kv[0][0]),) + kv[0][1:]):
        attn, run = run_dirs(out, group, seed, worker)
        spec = dict(SPEC_TEMPLATE)
        spec.update(repo=str(run_tree), model_seed=int(seed), manifest=shards[members[0][0]][1],
                    cuda=WORKER_CUDA[worker], attn_out=str(attn), run_root=str(run), xla_mem_fraction=float(frac),
                    ablate=ablate_for(group, out))
        if group == "N" and seed == RECORD_SEED:
            spec["record_motion"] = str(out / "record")
        if seed != RECORD_SEED:   # 客户端 official 口径要求 model_seed==42；其余 seed 走 testx（easy 局与 official 等价）
            spec.update(profile="testx", max_steps=1300)
        if group != "N":
            spec["swap_map"] = {episode_key(ep): episode_key(donor) for ep, donor in sorted(members)}
        path = out / "specs" / f"{group}-seed{seed}-{worker}-m{shard}.json"
        _write_json(path, spec)
        paths[(group, seed, worker, shard)] = path
    return paths


ENV_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/interpret/pl_swap_prep.py specs 生成。用法：source env.sh <GPU> [显存比例，默认 {frac:.2f}]
# 环境变量照抄 1106 env.sh；运行工作树固定 {run_tree}，HEAD 必须为 {commit} 且工作区 clean。
set -o pipefail
gpu=$1; frac=${{2:-{frac:.2f}}}
export CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_MEM_FRACTION=$frac
export PYTHONUNBUFFERED=1 UV_CACHE_DIR={store}/cache/uv XDG_CACHE_HOME={store}/cache HF_HOME={store}/cache/hf TORCH_HOME={store}/cache/torch JAX_COMPILATION_CACHE_DIR={store}/cache/jax MMEVLA_V1_STORE={store} OPENPI_DATA_HOME={store}/models HF_HUB_OFFLINE=1
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'
T={run_tree}
export PYTHONPATH=$T:$T/src:{bench}
cd "$T" || return 2
test "$(git rev-parse HEAD)" = {commit} || {{ echo "ENV_CHECK=FAIL reason=head"; return 2; }}
test -z "$(git status --porcelain)" || {{ echo "ENV_CHECK=FAIL reason=dirty"; return 2; }}
O={out}
py={py}
sim={sim}
echo "ENV_CHECK=PASS head={commit} gpu=$gpu frac=$frac"
"""

QUEUE_HEAD = """#!/usr/bin/env bash
# 由 scripts/interpret/pl_swap_prep.py specs 生成：阶段 {stage}、worker {worker}（GPU {cuda}），{n} 局。
# 某局 rc≠0 记失败并继续下一局；末行 PLS_QUEUE，有失败时以非零退出。
source {env} {cuda} {frac:.2f} || exit $?
fail=0
run_one() {{
  local group=$1 seed=$2 ep=$3 spec=$4
  "$py" -m scripts.interpret.run_local --spec "$spec" --episodes "PatternLock:$ep"
  local rc=$?
  echo "PLS_EPISODE stage={stage} group=$group seed=$seed ep=$ep rc=$rc"
  if [ "$rc" -ne 0 ]; then fail=$((fail+1)); fi
}}
"""

QUEUE_TAIL = """echo "PLS_QUEUE stage={stage} worker={worker} runs={n} failed=$fail"
[ "$fail" -eq 0 ]
"""


def write_queues(out, plan_rows, spec_paths, shards, frac=XLA_MEM_FRACTION):
    out = Path(out)
    (out / "queues").mkdir(parents=True, exist_ok=True)
    written = []
    for stage in STAGES:
        for worker in WORKERS:
            members = [run for st, w, run in plan_rows if st == stage and w == worker]
            path = out / "queues" / f"{stage}-{worker}.sh"
            lines = [QUEUE_HEAD.format(stage=stage, worker=worker, cuda=WORKER_CUDA[worker], n=len(members),
                                       env=shlex.quote(str(out / "env.sh")), frac=frac)]
            for group, seed, ep, donor in members:
                spec = spec_paths[(group, seed, worker, shards[ep][0])]
                lines.append(f"run_one {group} {seed} {ep} {shlex.quote(str(spec))}\n")
            lines.append(QUEUE_TAIL.format(stage=stage, worker=worker, n=len(members)))
            path.write_text("".join(lines))
            path.chmod(0o755)
            written.append(path)
    return written


def cmd_specs(args):
    out = Path(args.out).resolve()
    if not args.commit or len(args.commit) < 7:
        raise SystemExit("--commit 须给运行工作树的 commit sha")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    shards = manifest_shards(args.manifest_dir)
    runs = all_runs()
    plan_rows = assign(runs)
    spec_paths = write_specs(out, plan_rows, shards, run_tree=Path(args.run_tree), frac=args.frac)
    env_text = ENV_TEMPLATE.format(frac=args.frac, run_tree=args.run_tree, commit=args.commit, store=STORE,
                                   bench=BENCH_SRC, out=out, py=SPEC_TEMPLATE["policy_python"],
                                   sim=SPEC_TEMPLATE["sim_python"])
    (out / "env.sh").write_text(env_text)
    queues = write_queues(out, plan_rows, spec_paths, shards, frac=args.frac)
    plan = []
    for stage, worker, (group, seed, ep, donor) in plan_rows:
        attn, run = run_dirs(out, group, seed, worker)
        plan.append({"stage": stage, "worker": worker, "cuda": WORKER_CUDA[worker], "group": group, "seed": seed,
                     "ep": ep, "donor": donor, "path": list(EPISODES[ep][1]),
                     "donor_path": list(EPISODES[donor][1]) if donor is not None else None,
                     "spec": str(spec_paths[(group, seed, worker, shards[ep][0])]),
                     "attn_out": str(attn), "run_root": str(run)})
    _write_json(out / "plan.json", {"commit": args.commit, "run_tree": str(args.run_tree), "out": str(out),
                                    "record_dir": str(out / "record"), "seeds": list(SEEDS), "runs": plan})
    counts = {stage: sum(1 for row in plan if row["stage"] == stage) for stage in STAGES}
    print("SPECS " + " ".join(f"{k}={v}" for k, v in counts.items())
          + f" total={len(plan)} specs={len(spec_paths)} queues={len(queues)}", flush=True)
    return 0


# ---------------------------------------------------------------- donors
def donor_episodes():
    return sorted({donor for pairs in CONDITIONS.values() for _, donor in pairs})


def _savez_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def make_donors(out, episodes=None):
    """返回 (ok, info)。episodes=None 时处理全部 8 局；缺录制、无第 0 帧行、多行均判 FAIL。"""
    out = Path(out)
    wanted = list(EP_ORDER) if episodes is None else [int(e) for e in episodes]
    unknown = sorted(set(wanted) - set(EPISODES))
    if unknown:
        raise SystemExit(f"--episodes 含不在 8 局清单内的局: {unknown}")
    donors_set = set(donor_episodes())
    problems, donors, zeros, rows_max, maxabs = [], 0, 0, 0, 0.0
    for ep in wanted:
        src = out / "record" / record_name(ep)
        if not src.is_file():
            problems.append(f"ep{ep}:missing_record")
            continue
        with np.load(src) as z:
            starts = np.asarray(z["start_frames"], np.int32).reshape(-1)
            embs = np.asarray(z["embs"], np.float32)
            counts = np.asarray(z["counts"], np.int32).reshape(-1) if "counts" in z.files else np.ones_like(starts)
        idx = np.flatnonzero(starts == 0)
        if idx.size != 1 or embs.ndim != 2 or embs.shape[0] != starts.size:
            problems.append(f"ep{ep}:start0_rows={idx.size}")
            continue
        i = int(idx[0])
        row = embs[i:i + 1].copy()
        count = counts[i:i + 1].astype(np.int32) if counts.size == starts.size else np.ones(1, np.int32)
        targets = []
        if ep in donors_set:
            targets.append(("donor", out / "donor_trim" / record_name(ep), row))
        targets.append(("zero", out / "zero" / record_name(ep), np.zeros_like(row)))
        for kind, path, emb in targets:
            _savez_atomic(path, start_frames=np.zeros(1, np.int32), embs=emb.astype(np.float32), counts=count)
            with np.load(path) as back:   # 读回核对
                b_starts, b_embs = np.asarray(back["start_frames"]), np.asarray(back["embs"], np.float32)
            rows_max = max(rows_max, int(b_starts.size))
            if b_starts.size != 1 or int(b_starts[0]) != 0 or b_embs.shape != (1, embs.shape[1]):
                problems.append(f"ep{ep}:{kind}_shape")
            dev = float(np.max(np.abs(b_embs - (embs[i:i + 1] if kind == "donor" else 0.0))))
            maxabs = max(maxabs, dev)
            if dev != 0.0:
                problems.append(f"ep{ep}:{kind}_maxabs={dev}")
            if kind == "donor":
                donors += 1
            else:
                zeros += 1
    ok = not problems
    line = (f"DONOR_TRIM={'PASS' if ok else 'FAIL'} donors={donors} zeros={zeros} rows={rows_max} "
            f"maxabs={maxabs:g}")
    return ok, {"line": line, "problems": problems, "episodes": wanted}


def cmd_donors(args):
    episodes = [x for x in args.episodes.split(",") if x.strip()] if args.episodes else None
    ok, info = make_donors(args.out, episodes)
    for problem in info["problems"]:
        print(f"DONOR_PROBLEM {problem}", flush=True)
    print(info["line"], flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------- 公共
def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paths")
    p.add_argument("--out", required=True)
    p = sub.add_parser("specs")
    p.add_argument("--out", required=True)
    p.add_argument("--commit", required=True, help="运行工作树应处的 commit（env.sh 核对）")
    p.add_argument("--manifest-dir", default=str(REF_1106))
    p.add_argument("--run-tree", default=str(RUN_TREE))
    p.add_argument("--frac", type=float, default=XLA_MEM_FRACTION)
    p = sub.add_parser("donors")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", default=None, help="只处理这些局号，如 5,20")
    args = parser.parse_args(argv)
    return {"paths": cmd_paths, "specs": cmd_specs, "donors": cmd_donors}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
