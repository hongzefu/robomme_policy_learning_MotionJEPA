#!/usr/bin/env python3
"""StopCube motion 速度替换的准备脚本（计划 1110-stopcube-motion-speed-swap-plan.html 第一部分 K1–K11、§3.1–§3.3、§5；
第二部分「〇′ 开工审查裁决」B1–B5、N1–N7 与 1.3 节；以〇′为准）。

子命令：
  pool  --out <O> [--start 2500000] [--per-cell 6] [--bench-src …] [--overwrite]
        从 start 起逐个种子用 hover_points.task_params 复算 (M,N)，跳过官方 train/test/val metadata 的种子，
        12 格 (M∈{60,80,120} × N∈{2..5}) 各取前 per-cell 个；局号 0..71 按 (M, N, 种子) 升序分配；difficulty 取官方
        StopCube metadata 的众数（各取值计数记进 sc-pool.json）。写 <O>/metadata/record_dataset_StopCube_metadata.json
        （键 seed，口径同 sp_swap_prep.metadata_payload）与 <O>/sc-pool.json（每局 ep、seed、M、N、difficulty、metadata
        sha256、冒烟局）。冒烟接收局 = M∈{60,80,120} 且 N=3 的各第一局。末行
        SC_POOL=PASS|FAIL eps=<n> cells=<k> per_cell=<p>，另一行 SC_POOL_LIMIT（eval.py 的 PLFULL_EPISODES 上限）。
  specs --out <O>/<M> --model motion|wan --stage n|donor|iv --commit <sha> [--smoke] [路径根与 GL 参数，同 sp_swap_prep]
        n：每接收局 N（model_seed 42，record_motion=<根>/record/n）与 S（model_seed 1，不录），stopcube_mode=normal，
           stopcube_config=<根>/normal_cfg.json（{"schema_version":1,"episodes":{}}）。
        donor：按 slot（0=M′60、1=M′80、2=M′120）拆独立 spec（每份 spec 内每接收局只出现一次），stopcube_mode=speed_donor，
           stopcube_config=<根>/donor_cfg/slot<s>.json（由 donors 生成），record_motion=<根>/record/donor-s<s>。
        iv：按 slot 拆独立 spec，ablate=motion_swap_exact:<根>/donors_sc，swap_map 接收局 → 合成键，model_seed 42，不录。
        所有 spec：profile=plfull、plfull_metadata_dir=<O>/metadata、save_attn=false、不带 subgoal_trace。
        --smoke 只用冒烟 3 局，全部产物（runs、specs、queues、plan.json、record、donors_sc）写在 <O>/<M>/smoke/ 下。
        生成 env-aspen.sh / env-gl.sh、queues/<阶段>-<worker>.sh、launch_stage.sh（aspen tmux，会话前缀 scs-）、
        holds/<阶段>-hold<j>.sh 与 gl_launch_stage.sh（GL：每个占位只起一个 srun 作业步，内含两条通道，第二条错开 45 秒）。
  donors --out <O>/<M> [--smoke] [--overwrite]
        （动作表按「步前值」：first_env_step = N 轨迹第一行的 env_step_before，actions[i] = env_step_before ==
        first_env_step+i 的 executed_action，len(actions) == T_h − first_env_step；主会话 11-10 契约更正）
        读 n 阶段 N 回合终局与 step_trace.jsonl（stopcube_windows.read_trace，步号须连续）：press_N = 第一个
        button_pressed_after 为真的 env_step_after；T_h = 就位步（stopcube_oracle.hover_measurement：TCP 距按钮水平
        ≤ 0.03 且高度 < 0.2 的官方 button_hover，连续 3 步高度极差 ≤ 1 mm）；无就位取 press_N − 16；都没有取最后一步；
        并保证 2 ≤ T_h ≤（press_N 或最后一步）。供体配置 actions[i] = env_step_before == first_env_step+i 的
        executed_action，取 env_step_before ∈ [first_env_step, T_h−1]（真实环境首个物理步为 1）；hold_qpos = actions[−1]；horizon = max(M,M′)·N + 32。写三个 slot 的配置，
        末行 SC_DONOR_PREP。
  donors --out <O>/<M> --finalize [--smoke]
        读 record/donor-s<s>/StopCube_ep<r>.npz → 写 donors_sc/StopCube_ep<syn>.npz（start_frames int32、embs float32、
        counts 全 1），写后读回逐位比对；核对供体回执 button_ever_pressed == false、move_interval_applied == M′（不过即不写
        该供体）；REPLAY_EQ（M′ == M 的 slot，起点 s 满足 s+32 < T_h 的窗口与 record/n 同起点编码比对，阈值 1e-4，超过记
        FAIL 但不中断写供体）；MOVE_OVERRIDE（供体轨迹每行 move_interval == M′，geometry_window 解出的各次经过中心与
        M′·k − M′/2 偏差 ≤ 3 步，k = 1..N）；DONOR_FILES。

合成供体键 syn = 20000000 + recv×10 + slot（〇′ N4）。帧号：帧 f = 第 f 个动作执行后的观测（帧 0 为 reset 观测，
对应 env_step 1）；motion 窗起点 s = 32u，覆盖帧 [s, s+32]。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

import numpy as np

from scripts.interpret import pl_full_prep as P
from scripts.interpret import sp_swap_prep as S
from scripts.interpret.hover_points import MOVE_INTERVALS, press_step, task_params
from scripts.interpret.pl_swap_prep import MAIN, SPEC_TEMPLATE, XLA_MEM_FRACTION

# ---------------------------------------------------------------- 常量
TASK = "StopCube"
SEED_START = 2500000
PER_CELL = 6
TARGET_PASSES = (2, 3, 4, 5)
CELLS = tuple((m, n) for m in MOVE_INTERVALS for n in TARGET_PASSES)
SMOKE_N = 3
MAX_SCAN = 200000
SYN_BASE = 20000000
SLOTS = {0: 60, 1: 80, 2: 120}          # slot → M′
MAIN_SEED = 42
SEED_CHECK = 1
STOP_TIME_OVERRIDE = 99                 # 〇′ B1：只为关掉超时判败
HORIZON_PAD = 32
FIRST_ENV_STEP = 1
PRESS_FALLBACK = 16
HOVER_XY, HOVER_Z = 0.03, 0.2           # subgoal_evaluate_func.button_hover 默认阈值
MOTION_WINDOW = 33
MOTION_STRIDE = 32
REPLAY_TOL = 1e-4
MOVE_TOL = 3
STAGES = ("n", "donor", "iv")
SPEC_MAX = 20
CHUNK = 5
POOL_NAME = "sc-pool.json"
DONOR_DIR = "donors_sc"
CFG_DIR = "donor_cfg"
NORMAL_CFG = "normal_cfg.json"
GL_STAGGER = 45
GL_CPUS = 4

ASPEN = dict(S.ASPEN)
ASPEN.update(repo=str(MAIN / "v1-store/worktrees/sc-speed-run"), log_dir=str(MAIN / "v1-store/logs/sc-speed-1110"))
GL_STORE = S.GL_STORE
ASPEN_CUDA = S.ASPEN_CUDA
ASPEN_ORDER = S.ASPEN_ORDER

WAN_CKPT = ("v1-store/assets/vla-o16wan/vla59k-8bf3feb94861d042a78834104a702aa5d4a91c6f7e3b2c12c94899b1d23b8f97/59999")
MODELS = {
    "motion": {"checkpoint": str(MAIN / "v1-store/assets/vla-o16cc-59999/59999"),
               "policy_config": "mme_vla_suite_orig16_b128_60k",
               "model_kind": "stage2_pool4_concat",
               "target_contract": SPEC_TEMPLATE["target_contract"]},
    "wan": {"checkpoint": str(MAIN / WAN_CKPT),
            "policy_config": "mme_vla_suite_orig16_b128_60k",
            "model_kind": "wan_fixedpool1024",
            "target_contract": "none"},
}


# ---------------------------------------------------------------- 键与公式
def episode_key(ep):
    return f"{TASK}:{int(ep)}"


def record_name(ep):
    return f"{TASK}_ep{int(ep)}.npz"


def syn_ep(recv, slot):
    if int(slot) not in SLOTS or int(recv) < 0:
        raise ValueError(f"非法接收局或 slot：{recv}, {slot}")
    return SYN_BASE + int(recv) * 10 + int(slot)


def parse_syn(ep):
    """合成号 → (接收局, slot)。"""
    rest = int(ep) - SYN_BASE
    recv, slot = divmod(rest, 10)
    if rest < 0 or slot not in SLOTS:
        raise ValueError(f"非法合成号 {ep}")
    return recv, slot


def slot_of(m_prime):
    for slot, mp in SLOTS.items():
        if mp == int(m_prime):
            return slot
    raise ValueError(f"M′ 只能是 {tuple(SLOTS.values())}")


def delta_star(M, Mp, N):
    """完全按 token 时钟的预测偏移 Δ* = T(M′,N) − T(M,N)，T = M·N − M/2（§3.1）。"""
    return press_step(int(Mp), int(N)) - press_step(int(M), int(N))


def horizon(M, Mp, N):
    """供体回合主动结束步 H = max(M,M′)·N + 32（〇′ B1）。"""
    return max(int(M), int(Mp)) * int(N) + HORIZON_PAD


def speed_group(M, Mp):
    """〇′ N1：M′ > M 记 slow（token 更慢），M′ < M 记 fast，相等记 same。"""
    return "same" if int(Mp) == int(M) else ("slow" if int(Mp) > int(M) else "fast")


# ---------------------------------------------------------------- pool
def official_seeds(bench_src):
    return S.split_metadata_seeds(TASK, bench_src)


def official_difficulty(bench_src):
    """官方 StopCube metadata（train/test/val）的 difficulty 计数与众数（并列取字典序最小）。"""
    counts = Counter()
    for ds in ("train", "test", "val"):
        path = Path(bench_src) / "robomme/env_metadata" / ds / S.meta_name(TASK)
        if path.is_file():
            counts.update(str(r["difficulty"]) for r in json.loads(path.read_text())["records"])
    if not counts:
        return None, {}
    top = max(counts.values())
    return sorted(k for k, v in counts.items() if v == top)[0], dict(counts)


def build_pool(start, per_cell, banned=(), params_fn=task_params, max_scan=MAX_SCAN):
    """逐个种子复算 (M,N)，每格取前 per_cell 个。返回 (候选 [{seed,M,N}], 扫描数, 命中官方种子数)。"""
    banned = set(banned)
    got = {c: [] for c in CELLS}
    scanned = hit = 0
    seed = int(start)
    while any(len(v) < per_cell for v in got.values()):
        if scanned >= max_scan:
            raise SystemExit(f"扫了 {scanned} 个种子仍未凑满：{ {c: len(v) for c, v in got.items()} }")
        scanned += 1
        if seed in banned:
            hit += 1
        else:
            M, N = params_fn(seed)
            cell = (int(M), int(N))
            if cell in got and len(got[cell]) < per_cell:
                got[cell].append({"seed": seed, "M": cell[0], "N": cell[1]})
        seed += 1
    picked = [e for c in CELLS for e in got[c]]
    return picked, scanned, hit


def number_pool(picked, difficulty):
    """局号按 (M, N, 种子) 升序 0..n−1；冒烟局 = N=3 时每个 M 的第一局。"""
    eps = []
    for i, e in enumerate(sorted(picked, key=lambda x: (x["M"], x["N"], x["seed"]))):
        eps.append({"task_name": TASK, "episode_id": i, "ep": i, "env_seed": int(e["seed"]), "seed": int(e["seed"]),
                    "M": int(e["M"]), "N": int(e["N"]), "difficulty": difficulty})
    smoke = []
    for M in MOVE_INTERVALS:
        first = next((e for e in eps if e["M"] == M and e["N"] == SMOKE_N), None)
        if first is not None:
            smoke.append(first["ep"])
    return eps, smoke


def metadata_payload(eps):
    return S.metadata_payload(TASK, eps)


def pool_line(eps, per_cell, banned):
    cells = Counter((e["M"], e["N"]) for e in eps)
    overlap = sorted({e["env_seed"] for e in eps} & set(banned))
    ok = (len(cells) == len(CELLS) and all(cells[c] == per_cell for c in CELLS) and len(eps) == per_cell * len(CELLS)
          and not overlap)
    return (f"SC_POOL={'PASS' if ok else 'FAIL'} eps={len(eps)} cells={len(cells)} per_cell={per_cell}"
            + ("" if not overlap else f" overlap={len(overlap)}")), ok


def cmd_pool(args):
    out = Path(args.out).resolve()
    banned = official_seeds(args.bench_src)
    difficulty, counts = official_difficulty(args.bench_src)
    if difficulty is None:
        raise SystemExit(f"读不到官方 StopCube metadata：{args.bench_src}")
    picked, scanned, hit = build_pool(args.start, args.per_cell, banned)
    eps, smoke = number_pool(picked, difficulty)
    data = P._json_bytes(metadata_payload(eps))
    name = S.meta_name(TASK)
    sha = hashlib.sha256(data).hexdigest()
    P._write_once(out / "metadata" / name, data, args.overwrite)
    line, ok = pool_line(eps, args.per_cell, banned)
    limit = S.eval_episode_limit()
    limit_ok = limit is not None and len(eps) <= limit
    pool = {"task": TASK, "start": int(args.start), "per_cell": int(args.per_cell), "scanned": scanned,
            "banned_hits": hit, "difficulty": difficulty, "official_difficulty_counts": counts,
            "difficulty_rule": "官方 train/test/val StopCube metadata 的众数",
            "metadata_sha256": {name: sha}, "numbering": "局号按 (M, N, 种子) 升序 0..n−1",
            "smoke": smoke, "episodes": [{k: e[k] for k in ("ep", "seed", "M", "N", "difficulty")} for e in eps],
            "verdict": line}
    P._write_once(out / POOL_NAME, P._json_bytes(pool), args.overwrite)
    print(f"SC_POOL_SCAN start={args.start} scanned={scanned} banned_hits={hit} difficulty={difficulty} "
          f"official_counts={json.dumps(counts, sort_keys=True)} smoke={smoke}", flush=True)
    print(f"SC_POOL_LIMIT={'PASS' if limit_ok else 'FAIL'} eval_plfull_episodes={limit} eps={len(eps)}", flush=True)
    print(line, flush=True)
    return 0 if ok and limit_ok else 1


def load_pool(pool_root):
    """读 sc-pool.json → (pool, {局号: 局条目})；条目补 task_name / episode_id / env_seed 以供 manifest。"""
    pool = P._read_json(Path(pool_root) / POOL_NAME)
    by_ep = {}
    for e in pool["episodes"]:
        by_ep[int(e["ep"])] = {"task_name": TASK, "episode_id": int(e["ep"]), "ep": int(e["ep"]),
                               "env_seed": int(e["seed"]), "seed": int(e["seed"]), "M": int(e["M"]),
                               "N": int(e["N"]), "difficulty": e["difficulty"]}
    return pool, by_ep


def layout(out, smoke=False):
    """--out <O>/<M> → (本阶段根目录, 局池根 <O>)；冒烟根为 <O>/<M>/smoke。"""
    out = Path(out).resolve()
    return (out / "smoke" if smoke else out), out.parent


def recv_eps(pool, by_ep, smoke):
    return sorted(int(e) for e in pool["smoke"]) if smoke else sorted(by_ep)


# ---------------------------------------------------------------- 模板（会话前缀 scs-）
ENV_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/interpret/sc_speed_prep.py specs 生成（站点 {site}）。用法：source env-{site}.sh <GPU> [显存比例，默认 {frac:.2f}]
# 运行树 {run_tree}：aspen 须为 git 工作树且 HEAD = {commit}、clean；GL 为 release 目录，若有 .release_sha 须等于 {commit}。
# 只设 SCS_M（本阶段根目录），不覆盖调用方的 O。
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
SCS_M={root}
py={py}
sim={sim}
echo "ENV_CHECK=PASS site={site} head={commit} gpu=$gpu frac=$frac"
"""

QUEUE_HEAD = """#!/usr/bin/env bash
# 由 scripts/interpret/sc_speed_prep.py 生成：阶段 {stage}、worker {worker}（GPU {cuda}），{n} 个 spec。
# 某 spec rc≠0 记失败并继续。末行 SCS_QUEUE，有失败时非零退出。
source {env} {cuda} {frac:.2f} || exit $?
fail=0
run_spec() {{
  local stage=$1 name=$2 spec=$3
  "$py" -m scripts.interpret.run_local --spec "$spec"
  local rc=$?
  echo "SCS_SPEC stage=$stage spec=$name rc=$rc"
  if [ "$rc" -ne 0 ]; then fail=$((fail+1)); fi
}}
"""

QUEUE_TAIL = """echo "SCS_QUEUE stage={stage} worker={worker} specs={n} failed=$fail"
[ "$fail" -eq 0 ]
"""

LAUNCH_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/interpret/sc_speed_prep.py specs 生成（aspen 端）。用法：bash launch_stage.sh <{stages}>
# 为该阶段有 spec 的 aspen worker 各起一个 detached tmux（{prefix}-<阶段>-<worker>），日志 {logs}/{prefix}-<阶段>-<worker>.log。
# 顺序：n →（sc_speed_prep donors）→ donor →（donors --finalize）→ iv →（analysis.sc_speed_readout）。
SCS_M={root}
st=$1
case "$st" in {stages}) ;; *) echo "未知阶段 $st"; exit 2;; esac
mkdir -p {logs}
for w in {workers}; do
  q=$SCS_M/queues/$st-$w.sh
  [ -f "$q" ] && grep -q '^run_spec ' "$q" || continue
  name={prefix}-$st-$w; log={logs}/$name.log
  if tmux has-session -t "=$name" 2>/dev/null; then echo "SKIP_EXISTS $name"; continue; fi
  if [ -e "$log" ]; then echo "SKIP_LOG_EXISTS $log（重跑前先把旧日志改名留档）"; continue; fi
  tmux new-session -d -s "$name" "set -o pipefail; PYTHONUNBUFFERED=1 bash $q 2>&1 | tee $log; echo \\"EXIT_CODE=\\$?\\" >> $log"
  echo "STARTED $name $log"
done
"""

HOLD_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/interpret/sc_speed_prep.py specs 生成（GL 占位 {hold} 内的唯一作业步，阶段 {stage}）。
# 两条通道各跑一个 worker 队列，第二条错开 {stagger} 秒启动；每条通道日志 {logs}/{prefix}-{stage}-<worker>.log，
# 尾行 EXIT_CODE=。全部通道结束后，本次启动的任一通道非零即非零退出（因 SKIP_LOG_EXISTS 跳过的旧日志不计入）。
SCS_M={root}
mkdir -p {logs}
pids=()
chans=()
started=0
for w in {workers}; do
  q=$SCS_M/queues/{stage}-$w.sh
  [ -f "$q" ] && grep -q '^run_spec ' "$q" || continue
  log={logs}/{prefix}-{stage}-$w.log
  if [ -e "$log" ]; then echo "SKIP_LOG_EXISTS $log"; continue; fi
  if [ "$started" -gt 0 ]; then sleep {stagger}; fi
  ( set -o pipefail; PYTHONUNBUFFERED=1 bash "$q" 2>&1 | tee "$log"; echo "EXIT_CODE=$?" >> "$log" ) &
  pids+=($!)
  chans+=("$w")
  started=$((started+1))
  echo "CHANNEL_STARTED $w $log"
done
rc=0
for p in "${{pids[@]}}"; do wait "$p" || rc=1; done
for w in "${{chans[@]}}"; do
  log={logs}/{prefix}-{stage}-$w.log
  grep -q '^EXIT_CODE=0$' "$log" || rc=1
done
echo "SCS_HOLD stage={stage} hold={hold} channels=$started rc=$rc"
exit $rc
"""

GL_LAUNCH_TEMPLATE = """#!/usr/bin/env bash
# 由 scripts/interpret/sc_speed_prep.py specs 生成（GL 登录节点上运行）。用法：bash gl_launch_stage.sh <{stages}> <占位JobID_0> … <占位JobID_{last}>
# 每个占位起一个 detached tmux（{prefix}-<阶段>-h<j>），里面只起一个作业步（K9）：
#   srun --jobid=<hold> --overlap --exact --ntasks=1 --cpus-per-task={cpus} --gpu_cmode=shared bash holds/<阶段>-hold<j>.sh
# 作业步内两条通道错开 {stagger} 秒；会话日志 {logs}/{prefix}-<阶段>-h<j>.log，通道日志见 hold 脚本。
SCS_M={root}
st=$1; shift
case "$st" in {stages}) ;; *) echo "未知阶段 $st"; exit 2;; esac
holds=("$@")
[ "${{#holds[@]}}" -eq {jobs} ] || {{ echo "需要 {jobs} 个占位 JobID，得到 ${{#holds[@]}}"; exit 2; }}
mkdir -p {logs}
for j in $(seq 0 {last}); do
  h=$SCS_M/holds/$st-hold$j.sh
  [ -f "$h" ] || continue
  grep -q 'NO_WORK' "$h" && continue
  hold=${{holds[$j]}}
  name={prefix}-$st-h$j; log={logs}/$name.log
  if tmux has-session -t "=$name" 2>/dev/null; then echo "SKIP_EXISTS $name"; continue; fi
  if [ -e "$log" ]; then echo "SKIP_LOG_EXISTS $log（重跑前先把旧日志改名留档）"; continue; fi
  tmux new-session -d -s "$name" "set -o pipefail; srun --jobid=$hold --overlap --exact --ntasks=1 --cpus-per-task={cpus} --gpu_cmode=shared bash $h 2>&1 | tee $log; echo \\"EXIT_CODE=\\$?\\" >> $log"
  echo "STARTED $name hold=$hold $log"
done
"""


# ---------------------------------------------------------------- spec 构建
def make_workers(gl_jobs, gl_per_job, aspen=True):
    out = [(w, "aspen", ASPEN_CUDA[w], None) for w in ASPEN_ORDER] if aspen else []
    for j in range(int(gl_jobs)):
        for k in range(int(gl_per_job)):
            out.append((f"g{j}{chr(ord('a') + k)}", "gl", "0", j))
    if not out:
        raise SystemExit("没有任何 worker：--no-aspen 时须给 --gl-jobs > 0")
    return out


def site_roots(args):
    roots = S.site_roots(args)
    roots["aspen"] = {k: str(getattr(args, k) or ASPEN[k]) for k in ASPEN}
    if roots["gl"] is not None and not args.gl_log_dir:
        roots["gl"]["log_dir"] = f"{roots['gl']['store']}/logs/sc-speed-1110"
    return roots


def model_entry(name, overrides=None):
    entry = dict(MODELS[name])
    entry.update({k: v for k, v in (overrides or {}).items() if v is not None})
    missing = [k for k in ("checkpoint", "policy_config", "model_kind", "target_contract") if not entry.get(k)]
    if missing:
        raise SystemExit(f"模型 {name} 缺 {missing}")
    return entry


class SCBuilder:
    """收集 spec、清单、plan 行与队列行；按 worker 所在站点选路径根与模型项（同 sp_swap_prep.Builder 的写法，
    另加 StopCube 键、去掉 subgoal_trace）。"""

    def __init__(self, root, models, roots, workers, frac, meta_dir, meta_shas):
        self.root, self.models, self.roots = Path(root), models, roots
        self.workers = {w[0]: tuple(w) for w in workers}
        self.order = [w[0] for w in workers]
        self.frac, self.meta_dir, self.meta_shas = float(frac), Path(meta_dir), dict(meta_shas)
        self.specs, self.rows, self.queue = {}, [], {}

    def site(self, worker):
        return self.workers[worker][1]

    def add(self, name, stage, worker, episodes, *, seed, mode, config, ablate="none", swap_map=None, record=None,
            rows=()):
        if name in self.specs:
            raise ValueError(f"spec 重名 {name}")
        site = self.site(worker)
        roots = self.roots[site]
        run_dir = self.root / "runs" / name
        spec = dict(SPEC_TEMPLATE)
        spec.pop("subgoal_trace", None)
        spec.update(policy_python=roots["policy_python"], sim_python=roots["sim_python"],
                    stage2_python=roots["stage2_python"])
        model = self.models[site]
        spec.update({k: v for k, v in model.items() if k != "extra"})
        spec.update(model.get("extra") or {})
        spec.update(repo=roots["repo"], model_seed=int(seed), cuda=self.workers[worker][2],
                    attn_out=str(run_dir / "attn"), run_root=str(run_dir / "run"), xla_mem_fraction=self.frac,
                    manifest=str(self.root / "manifests" / f"{name}.json"), ablate=ablate, profile="plfull",
                    plfull_metadata_dir=str(self.meta_dir), stopcube_mode=mode, stopcube_config=str(config),
                    save_attn=False)
        if swap_map:
            spec["swap_map"] = dict(swap_map)
        if record:
            spec["record_motion"] = str(record)
        manifest = S.manifest_for(episodes, self.meta_shas, chunk=CHUNK)
        self.specs[name] = (spec, manifest, stage, worker)
        self.queue.setdefault((stage, worker), []).append(
            f"run_spec {stage} {name} {shlex.quote(str(self.root / 'specs' / f'{name}.json'))}\n")
        for row in rows:
            row = dict(row)
            row.update(stage=stage, worker=worker, site=site, seed=int(seed), spec=name, attn_out=spec["attn_out"],
                       run_root=spec["run_root"], record_dir=str(record) if record else None)
            self.rows.append(row)
        return spec

    def write(self, stage):
        for name, (spec, manifest, st, worker) in self.specs.items():
            P._write_json(self.root / "manifests" / f"{name}.json", manifest)
            P._write_json(self.root / "specs" / f"{name}.json", spec)
        (self.root / "queues").mkdir(parents=True, exist_ok=True)
        written = []
        for worker in self.order:
            lines = self.queue.get((stage, worker), [])
            n_spec = sum(1 for x in lines if x.startswith("run_spec "))
            path = self.root / "queues" / f"{stage}-{worker}.sh"
            text = [QUEUE_HEAD.format(stage=stage, worker=worker, cuda=self.workers[worker][2], n=n_spec,
                                      env=shlex.quote(str(self.root / f"env-{self.site(worker)}.sh")), frac=self.frac)]
            text += lines
            text.append(QUEUE_TAIL.format(stage=stage, worker=worker, n=n_spec))
            path.write_text("".join(text))
            path.chmod(0o755)
            written.append(path)
        return written


def ep_row(e, group, **extra):
    row = {"group": group, "ep": int(e["ep"]), "M": int(e["M"]), "N": int(e["N"]), "slot": None, "M_prime": None,
           "syn": None, "kind": None}
    row.update(extra)
    return row


def distribute(items, order):
    return S.distribute(items, order)


def chunks(seq, size=SPEC_MAX):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def build_n(b, eps, by_ep):
    """n 阶段：每接收局 N（seed 42，录 record/n）与 S（seed 1，不录）。"""
    cfg = b.root / NORMAL_CFG
    P._write_once(cfg, P._json_bytes({"schema_version": 1, "episodes": {}}))
    per = distribute(list(eps), b.order)
    for worker in b.order:
        for i, part in enumerate(chunks(per.get(worker, []))):
            es = [by_ep[r] for r in part]
            b.add(f"n-N-{worker}-{i:02d}", "n", worker, es, seed=MAIN_SEED, mode="normal", config=cfg,
                  record=b.root / "record" / "n", rows=[ep_row(e, "N") for e in es])
            b.add(f"n-S-{worker}-{i:02d}", "n", worker, es, seed=SEED_CHECK, mode="normal", config=cfg,
                  rows=[ep_row(e, "S") for e in es])


def slot_items(root, eps, by_ep, need):
    """返回 ([(slot, 接收局)], 缺失 [(slot, 接收局)])：donor 阶段要求 donor_cfg/slot<s>.json 含该局；
    iv 阶段要求 donors_sc 里有合成供体文件。"""
    items, missing = [], []
    for slot in SLOTS:
        if need == "cfg":
            path = Path(root) / CFG_DIR / f"slot{slot}.json"
            if not path.is_file():
                raise SystemExit(f"缺供体配置 {path}：先跑 donors")
            have = {int(k.rpartition(":")[2]) for k in P._read_json(path)["episodes"]}
        else:
            have = {r for r in eps if (Path(root) / DONOR_DIR / record_name(syn_ep(r, slot))).is_file()}
        items += [(slot, r) for r in eps if r in have]
        missing += [(slot, r) for r in eps if r not in have]
    return items, missing


def missing_gate(missing, allow):
    """打印 SCS_SPECS_MISSING 与逐项缺失；缺失 > 0 且未加 --allow-missing 时返回 False（调用方非零退出）。"""
    print(f"SCS_SPECS_MISSING={len(missing)}", flush=True)
    for slot, r in missing:
        print(f"SCS_MISSING slot={slot} recv={episode_key(r)}", flush=True)
    return not missing or bool(allow)


def build_slot_stage(b, stage, items, by_ep):
    """donor / iv：按 slot 拆独立 spec，每份 spec 内每接收局只出现一次（〇′ N4）。"""
    per = distribute(sorted(items), b.order)
    for worker in b.order:
        mine = per.get(worker, [])
        bins = P.pack([(slot, r) for slot, r in mine], SPEC_MAX)
        for i, (slot, members) in enumerate(bins):
            es = [by_ep[m[1]] for m in members]
            mp = SLOTS[slot]
            rows = [ep_row(e, "D" if stage == "donor" else "R", slot=slot, M_prime=mp, syn=syn_ep(e["ep"], slot),
                           kind=speed_group(e["M"], mp)) for e in es]
            if stage == "donor":
                b.add(f"donor-s{slot}-{worker}-{i:02d}", stage, worker, es, seed=MAIN_SEED, mode="speed_donor",
                      config=b.root / CFG_DIR / f"slot{slot}.json", record=b.root / "record" / f"donor-s{slot}",
                      rows=rows)
            else:
                b.add(f"iv-s{slot}-{worker}-{i:02d}", stage, worker, es, seed=MAIN_SEED, mode="normal",
                      config=b.root / NORMAL_CFG, ablate=f"motion_swap_exact:{b.root / DONOR_DIR}",
                      swap_map={episode_key(e["ep"]): episode_key(syn_ep(e["ep"], slot)) for e in es}, rows=rows)


def write_launchers(root, stage_tag, model, roots, workers, gl_jobs):
    prefix = f"scs-{model}{stage_tag}"
    aspen_w = [w[0] for w in workers if w[1] == "aspen"]
    launch = root / "launch_stage.sh"
    launch.write_text(LAUNCH_TEMPLATE.format(root=root, prefix=prefix, logs=roots["aspen"]["log_dir"],
                                             workers=" ".join(aspen_w), stages="|".join(STAGES)))
    launch.chmod(0o755)
    if roots["gl"] is None:
        return []
    gl = root / "gl_launch_stage.sh"
    gl.write_text(GL_LAUNCH_TEMPLATE.format(root=root, prefix=prefix, logs=roots["gl"]["log_dir"],
                                            stages="|".join(STAGES), jobs=int(gl_jobs), last=int(gl_jobs) - 1,
                                            cpus=GL_CPUS, stagger=GL_STAGGER))
    gl.chmod(0o755)
    return [gl]


def write_holds(root, stage, model_tag, workers, logs, queue_has):
    """每个 GL 占位一份 holds/<阶段>-hold<j>.sh；该占位两条通道都没活时写 NO_WORK 标记。"""
    (root / "holds").mkdir(parents=True, exist_ok=True)
    by_hold = {}
    for w in workers:
        if w[1] == "gl":
            by_hold.setdefault(int(w[3]), []).append(w[0])
    paths = []
    for j, ws in sorted(by_hold.items()):
        path = root / "holds" / f"{stage}-hold{j}.sh"
        busy = [w for w in ws if queue_has(stage, w)]
        if not busy:
            path.write_text(f"#!/usr/bin/env bash\n# NO_WORK：阶段 {stage} 占位 {j} 没有 spec\nexit 0\n")
        else:
            path.write_text(HOLD_TEMPLATE.format(hold=j, stage=stage, stagger=GL_STAGGER, logs=logs,
                                                 prefix=f"scs-{model_tag}", root=root, workers=" ".join(ws)))
        path.chmod(0o755)
        paths.append(path)
    return paths


def cmd_specs(args):
    out = Path(args.out).resolve()
    if out.name != args.model:
        raise SystemExit(f"--out 末级目录须为模型名 {args.model}（约定 <O>/<M>）")
    if not args.commit or len(args.commit) < 7:
        raise SystemExit("--commit 须给运行工作树的 commit sha")
    root, pool_root = layout(out, args.smoke)
    pool, by_ep = load_pool(pool_root)
    meta_dir = pool_root / "metadata"
    shas = {name: hashlib.sha256((meta_dir / name).read_bytes()).hexdigest() for name in pool["metadata_sha256"]}
    if shas != pool["metadata_sha256"]:
        raise SystemExit("metadata 与 sc-pool.json 记录的 sha256 不符")
    roots = site_roots(args)
    model = model_entry(args.model, {"checkpoint": args.checkpoint, "policy_config": args.policy_config,
                                     "model_kind": args.model_kind, "target_contract": args.target_contract})
    if args.extra:
        model["extra"] = json.loads(args.extra)
    models = {"aspen": model}
    if roots["gl"] is not None:
        if not args.gl_checkpoint:
            raise SystemExit("--gl-jobs > 0 时必须给 --gl-checkpoint（GL 端权重目录，两端目录名可能不同）")
        gl_extra = json.loads(args.gl_extra) if args.gl_extra else None
        models["gl"] = S.gl_model(model, roots["aspen"], roots["gl"], args.gl_checkpoint, args.gl_target_contract,
                                  gl_extra)
    workers = make_workers(args.gl_jobs if roots["gl"] else 0, args.gl_per_job, not args.no_aspen)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    plan = P._read_json(plan_path) if plan_path.exists() else {
        "task": TASK, "model": args.model, "root": str(root), "pool_root": str(pool_root), "smoke": bool(args.smoke),
        "runs": [], "stages": {}}
    if plan["model"] != args.model:
        raise SystemExit(f"plan.json 属于模型 {plan['model']}，不是 {args.model}")
    if args.stage in plan["stages"] and not args.overwrite:
        raise SystemExit(f"阶段 {args.stage} 已生成；确需重写加 --overwrite（不删任何运行目录）")
    b = SCBuilder(root, models, roots, workers, args.frac, meta_dir, shas)
    eps = recv_eps(pool, by_ep, args.smoke)
    if args.stage == "n":
        build_n(b, eps, by_ep)
    else:
        items, missing = slot_items(root, eps, by_ep, "cfg" if args.stage == "donor" else "donors")
        if not missing_gate(missing, args.allow_missing):
            print("缺失 (slot, 接收局) > 0：未生成任何 spec；确认后加 --allow-missing 放行", flush=True)
            return 2
        if not items:
            raise SystemExit(f"阶段 {args.stage} 没有可用的 (slot, 接收局)")
        build_slot_stage(b, args.stage, items, by_ep)
    for site, r in roots.items():
        if r is None:
            continue
        (root / f"env-{site}.sh").write_text(ENV_TEMPLATE.format(
            site=site, frac=args.frac, run_tree=r["repo"], commit=args.commit, store=r["store"], mmevla=r["mmevla_store"],
            bench=r["bench_src"], root=root, py=r["policy_python"], sim=r["sim_python"]))
    tag = "-smoke" if args.smoke else ""
    queues = b.write(args.stage)
    write_launchers(root, tag, args.model, roots, workers, args.gl_jobs)
    holds = []
    if roots["gl"] is not None:
        holds = write_holds(root, args.stage, f"{args.model}{tag}", workers, roots["gl"]["log_dir"],
                            lambda st, w: bool(b.queue.get((st, w))))
    plan["runs"] = [r for r in plan["runs"] if r["stage"] != args.stage] + b.rows
    plan["stages"][args.stage] = {"commit": args.commit, "workers": [list(w) for w in workers], "models": models,
                                  "roots": roots, "frac": float(args.frac), "specs": sorted(b.specs)}
    plan["metadata_sha256"] = shas
    P._write_json(plan_path, plan)
    counts = Counter(r["group"] for r in b.rows)
    print(f"SCS_SPECS model={args.model} stage={args.stage} smoke={int(args.smoke)} "
          + " ".join(f"{g}={n}" for g, n in sorted(counts.items()))
          + f" specs={len(b.specs)} queues={len(queues)} holds={len(holds)} workers={len(workers)}", flush=True)
    return 0


# ---------------------------------------------------------------- donors：读 N 回合
def first_press(rows):
    return next((r["env_step_after"] for r in rows if r.get("button_pressed_after") is True), None)


def hover_rows(rows, button):
    """把 stopcube_runner 的逐步行换成 stopcube_oracle.hover_measurement 需要的字段（官方 button_hover 判据）。"""
    out = []
    for r in rows:
        tcp = r.get("tcp_after")
        ok = (isinstance(tcp, (list, tuple)) and len(tcp) == 3 and button is not None
              and math.hypot(tcp[0] - button[0], tcp[1] - button[1]) <= HOVER_XY and tcp[2] < HOVER_Z)
        out.append({"env_step_after": r["env_step_after"], "tcp_after": tcp, "phase": "hold",
                    "button_hover_after": bool(ok), "button_pressed_after": bool(r.get("button_pressed_after"))})
    return out


def reach_step(rows, button):
    from scripts.interpret.stopcube_oracle import hover_measurement
    if button is None:
        return None
    reach, _ = hover_measurement(hover_rows(rows, button))
    return reach


def hold_step(rows, button):
    """T_h 规则（〇′ B3）：就位步 → press_N − 16 → 最后一步；并保证 2 ≤ T_h ≤（press_N − 1 或最后一步），
    即回放动作（步前值 < T_h）不含产生按下的那一步。
    返回 (T_h, 来源, press_N, reach)。"""
    press = first_press(rows)
    last = rows[-1]["env_step_after"]
    reach = reach_step(rows, button)
    if reach is not None:
        th, src = reach, "reach"
    elif press is not None:
        th, src = press - PRESS_FALLBACK, "press-16"
    else:
        th, src = last, "last"
    upper = press - 1 if press is not None else last
    th = max(2, min(int(th), int(upper)))
    return th, src, press, reach


def replay_actions(rows, t_hold, first=FIRST_ENV_STEP):
    """actions[j−first] = env_step_before == j 的 executed_action，j = first..T_h−1；缺步或非有限 8 维即报错。"""
    by = {}
    for r in rows:
        if r["env_step_before"] in by:
            raise ValueError(f"重复 env_step_before={r['env_step_before']}")
        by[r["env_step_before"]] = r
    out = []
    for j in range(int(first), int(t_hold)):
        r = by.get(j)
        if r is None:
            raise ValueError(f"缺 env_step_before={j}")
        a = r.get("executed_action")
        if not (isinstance(a, list) and len(a) == 8 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in a)):
            raise ValueError(f"env_step_before={j} 的 executed_action 不是有限 8 维")
        out.append([float(x) for x in a])
    if not out:
        raise ValueError("回放动作为空（T_h 须 ≥ 2）")
    return out


def donor_entry(env_seed, M, N, Mp, actions, t_hold, first_env_step=FIRST_ENV_STEP):
    """len(actions) == t_hold − first_env_step；env_step_before ≥ t_hold 的步由运行器用 hold_qpos。"""
    if len(actions) != int(t_hold) - int(first_env_step):
        raise ValueError("动作数须等于 t_hold − first_env_step")
    return {"env_seed": int(env_seed), "move_interval_override": int(Mp), "stop_time_override": STOP_TIME_OVERRIDE,
            "first_env_step": int(first_env_step), "actions": [list(a) for a in actions], "t_hold": int(t_hold),
            "hold_qpos": list(actions[-1]), "horizon": horizon(M, Mp, N), "orig_move_interval": int(M),
            "target_pass": int(N)}


def find_sc_run(row):
    """sp_swap_prep.find_run + 供体回执 meta/speed_donor_receipt.json。"""
    run = S.find_run(row["run_root"], row["attn_out"], episode_key(row["ep"]))
    run["donor_receipt"] = None
    if run["trace"] is not None:
        p = Path(run["trace"]).parent / "meta" / "speed_donor_receipt.json"
        if p.is_file():
            run["donor_receipt"] = P._read_json(p)
    return run


def load_trace(path):
    from scripts.interpret.stopcube_windows import read_trace
    return read_trace(path)


def n_info(row, e):
    """一局 N：status ok / no_terminal / error_outcome / no_trace / trace_bad / bad_first_step / params_mismatch / no_actions。"""
    info = {"ep": int(e["ep"]), "M": e["M"], "N": e["N"], "status": "ok"}
    run = find_sc_run(row)
    if run["terminal"] is None:
        info["status"] = "no_terminal"
        return info
    info["outcome"] = run["terminal"].get("outcome")
    if info["outcome"] == "error":
        info["status"] = "error_outcome"
        return info
    if run["trace"] is None:
        info["status"] = "no_trace"
        return info
    try:
        trace = load_trace(run["trace"])
    except ValueError as error:
        info.update(status="trace_bad", reason=str(error))
        return info
    rows = trace["rows"]
    if not rows or not isinstance(rows[0].get("env_step_before"), int) or rows[0]["env_step_before"] < 1:
        info["status"] = "bad_first_step"
        return info
    first = int(rows[0]["env_step_before"])
    if rows[0].get("move_interval") != e["M"] or rows[0].get("target_pass") != e["N"]:
        info.update(status="params_mismatch", trace_M=rows[0].get("move_interval"), trace_N=rows[0].get("target_pass"))
        return info
    button = (run["receipt"] or {}).get("extra", {}).get("button_position") or trace["meta"].get("button_position")
    th, src, press, reach = hold_step(rows, button)
    th = max(th, first + 1)
    info.update(trace=str(run["trace"]), press_N=press, reach=reach, t_hold=th, t_hold_source=src,
                last_step=rows[-1]["env_step_after"], first_env_step=first)
    try:
        info["actions"] = replay_actions(rows, th, first)
    except ValueError as error:
        info.update(status="no_actions", reason=str(error))
    return info


def cmd_donors_prep(args, root, plan, by_ep):
    infos = {}
    for row in plan["runs"]:
        if row["stage"] == "n" and row["group"] == "N":
            infos[int(row["ep"])] = n_info(row, by_ep[int(row["ep"])])
    cfgs = {slot: {} for slot in SLOTS}
    for ep, v in sorted(infos.items()):
        if v["status"] != "ok":
            print(f"SC_DONOR_INVALID ep={ep} status={v['status']} {v.get('reason', '')}".rstrip(), flush=True)
            continue
        e = by_ep[ep]
        for slot, mp in SLOTS.items():
            cfgs[slot][episode_key(ep)] = donor_entry(e["env_seed"], e["M"], e["N"], mp, v["actions"], v["t_hold"],
                                                        v["first_env_step"])
    for slot, eps in cfgs.items():
        P._write_once(root / CFG_DIR / f"slot{slot}.json", P._json_bytes({"schema_version": 1, "episodes": eps}),
                      args.overwrite)
    ok = [v for v in infos.values() if v["status"] == "ok"]
    src = Counter(v["t_hold_source"] for v in ok)
    plan["donor_prep"] = {str(ep): {k: v.get(k) for k in ("status", "reason", "press_N", "reach", "t_hold", "first_env_step",
                                                          "t_hold_source", "last_step", "outcome", "trace")}
                          for ep, v in sorted(infos.items())}
    P._write_json(root / "plan.json", plan)
    verdict = "PASS" if infos and len(ok) == len(infos) else "FAIL"
    print(f"SC_DONOR_PREP={verdict} eps={len(infos)} ok={len(ok)} invalid={len(infos) - len(ok)} "
          f"th_reach={src['reach']} th_press16={src['press-16']} th_last={src['last']} slots={len(SLOTS)}", flush=True)
    return 0 if verdict == "PASS" else 1


# ---------------------------------------------------------------- donors --finalize
def replay_starts(starts, t_hold):
    """REPLAY_EQ 的比较窗口：起点 s 满足 s+32 < T_h（窗内帧全由回放动作产生）。"""
    return sorted(int(s) for s in starts if int(s) + MOTION_WINDOW - 1 < int(t_hold))


def write_speed_donor(path, starts, embs):
    """写合成供体 npz 并读回逐位比对；返回问题字符串或 None。"""
    starts = [int(s) for s in starts]
    embs = np.asarray(embs, np.float32)
    P._savez_atomic(path, start_frames=np.asarray(starts, np.int32), embs=embs,
                    counts=np.ones(len(starts), np.int32))
    with np.load(path) as back:
        if (np.asarray(back["start_frames"]).tolist() != starts or back["embs"].dtype != np.float32
                or back["embs"].shape != embs.shape or back["embs"].tobytes() != embs.tobytes()
                or np.asarray(back["counts"]).tolist() != [1] * len(starts)):
            return "readback"
    return None


def move_override_check(trace, receipt, Mp, N):
    """供体轨迹每行 move_interval == M′，geometry_window 各次经过中心与 M′·k − M′/2 的偏差（k = 1..N）。
    供体行内 target_pass 是改写后的 99，所以 definition 的 target_pass 一律取配置的 N，cube_half_size 取回执。
    返回 (ok, maxdev, 原因)。"""
    from scripts.interpret.stopcube_windows import geometry_window
    rows = trace["rows"]
    bad_rows = sum(1 for r in rows if r.get("move_interval") != Mp)
    if bad_rows:
        return False, None, f"move_interval_rows={bad_rows}"
    half = (receipt or {}).get("cube_half_size")
    geo = geometry_window([trace], {"move_interval": int(Mp), "target_pass": int(N), "cube_half_size": half})
    visits = {v["target_pass"]: v for v in geo["visits"]}
    if geo["window_status"] != "resolved" or any(k not in visits for k in range(1, int(N) + 1)):
        return False, None, "unresolved"
    devs = [abs((visits[k]["in"] + visits[k]["out"]) / 2 - (Mp * k - Mp / 2)) for k in range(1, int(N) + 1)]
    maxdev = max(devs)
    return maxdev <= MOVE_TOL, maxdev, None if maxdev <= MOVE_TOL else f"dev={maxdev}"


def replay_line(model, replay):
    """REPLAY_EQ 判定行：比较窗口总数为 0、任一局缺窗或 maxabs 超阈值即 FAIL；zero_window_eps = windows=0 的局数。"""
    maxabs = max((x["maxabs"] for x in replay if x["maxabs"] is not None), default=None)
    windows = sum(x["windows"] for x in replay)
    ok = (bool(replay) and windows > 0
          and all(x["maxabs"] is not None and x["maxabs"] <= REPLAY_TOL and not x["missing"] for x in replay))
    return (f"REPLAY_EQ={'PASS' if ok else 'FAIL'} model={model} eps={len(replay)} windows={windows} "
            f"maxabs={maxabs if maxabs is None else f'{maxabs:.3g}'} "
            f"zero_window_eps={sum(1 for x in replay if x['windows'] == 0)}")


def cmd_donors_finalize(args, root, plan, by_ep):
    prep = plan.get("donor_prep") or {}
    n_rec = root / "record" / "n"
    problems, written, replay, moves = [], [], [], []
    for row in sorted((r for r in plan["runs"] if r["stage"] == "donor"), key=lambda r: (r["slot"], r["ep"])):
        r, slot, mp = int(row["ep"]), int(row["slot"]), int(row["M_prime"])
        e = by_ep[r]
        tag = f"{episode_key(r)}:s{slot}"
        run = find_sc_run(row)
        rc = run.get("donor_receipt")
        if run["terminal"] is None or run["trace"] is None:
            problems.append(f"{tag}:no_run")
            continue
        if rc is None:
            problems.append(f"{tag}:no_receipt")
            continue
        if rc.get("button_ever_pressed") is not False or rc.get("move_interval_applied") != mp:
            problems.append(f"{tag}:receipt pressed={rc.get('button_ever_pressed')} applied={rc.get('move_interval_applied')}")
            continue
        try:
            trace = load_trace(run["trace"])
            ok, dev, why = move_override_check(trace, rc, mp, e["N"])
        except ValueError as error:
            ok, dev, why = False, None, f"trace_bad:{error}"
        moves.append({"key": tag, "ok": ok, "maxdev": dev, "why": why})
        rec = S.load_record(Path(row["record_dir"]) / record_name(r))
        if not rec:
            problems.append(f"{tag}:no_record")
            continue
        starts = sorted(rec)
        why_w = write_speed_donor(root / DONOR_DIR / record_name(syn_ep(r, slot)), starts,
                                  np.stack([rec[s] for s in starts]))
        if why_w:
            problems.append(f"{tag}:{why_w}")
            continue
        written.append(syn_ep(r, slot))
        if mp == e["M"]:
            th = (prep.get(str(r)) or {}).get("t_hold")
            nrec = S.load_record(n_rec / record_name(r))
            if th is None or not nrec:
                replay.append({"ep": r, "windows": 0, "maxabs": None, "missing": ["no_n_record_or_t_hold"]})
                continue
            cmp_starts = replay_starts(nrec, th)
            missing = [s for s in cmp_starts if s not in rec]
            devs = [float(np.max(np.abs(nrec[s] - rec[s]))) for s in cmp_starts if s in rec]
            replay.append({"ep": r, "t_hold": th, "windows": len(devs), "maxabs": max(devs) if devs else 0.0,
                           "missing": missing})
    maxabs = max((x["maxabs"] for x in replay if x["maxabs"] is not None), default=None)
    mv_ok = bool(moves) and all(m["ok"] for m in moves)
    mv_dev = max((m["maxdev"] for m in moves if m["maxdev"] is not None), default=None)
    lines = [
        f"DONOR_FILES={'PASS' if not problems and written else 'FAIL'} files={len(written)} problems={len(problems)}",
        replay_line(plan["model"], replay),
        f"MOVE_OVERRIDE={'PASS' if mv_ok else 'FAIL'} eps={len(moves)} maxdev={mv_dev}",
    ]
    for p in problems:
        print(f"DONOR_PROBLEM {p}", flush=True)
    for m in moves:
        if not m["ok"]:
            print(f"MOVE_OVERRIDE_BAD {m['key']} {m['why']}", flush=True)
    plan["finalize"] = {"written": sorted(written), "problems": problems, "replay_eq": replay, "move_override": moves,
                        "lines": lines}
    P._write_json(root / "plan.json", plan)
    for line in lines:
        print(line, flush=True)
    return 0 if all("=PASS" in l for l in lines) else 1


def cmd_donors(args):
    root, pool_root = layout(args.out, args.smoke)
    plan = P._read_json(root / "plan.json")
    _, by_ep = load_pool(pool_root)
    if args.finalize:
        return cmd_donors_finalize(args, root, plan, by_ep)
    return cmd_donors_prep(args, root, plan, by_ep)


# ---------------------------------------------------------------- 命令行
def add_root_args(p):
    for k in ASPEN:
        p.add_argument("--" + k.replace("_", "-"), dest=k, default=None, help=f"aspen 端，默认 {ASPEN[k]}")
    p.add_argument("--no-aspen", action="store_true", help="不用 aspen 的 4 个 worker（只用 GL）")
    p.add_argument("--gl-jobs", type=int, default=0, help="GL 单卡占位数（0 = 只用 aspen）")
    p.add_argument("--gl-per-job", type=int, default=2)
    p.add_argument("--gl-release", default=None, help="GL 端 release 目录（= GL 仓库）")
    p.add_argument("--gl-store", default=None, help=f"默认 {GL_STORE}")
    p.add_argument("--gl-bench-src", default=None)
    p.add_argument("--gl-sim-python", default=None)
    p.add_argument("--gl-policy-python", default=None)
    p.add_argument("--gl-stage2-python", default=None)
    p.add_argument("--gl-mmevla-store", default=None)
    p.add_argument("--gl-log-dir", default=None)
    p.add_argument("--gl-checkpoint", default=None, help="GL 端权重目录（--gl-jobs > 0 时必给）")
    p.add_argument("--gl-target-contract", default=None)
    p.add_argument("--gl-extra", default=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=int, default=SEED_START)
    p.add_argument("--per-cell", type=int, default=PER_CELL)
    p.add_argument("--bench-src", default=ASPEN["bench_src"])
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("specs")
    p.add_argument("--out", required=True, help="<O>/<M>")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--stage", "--phase", dest="stage", required=True, choices=STAGES)
    p.add_argument("--commit", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--frac", type=float, default=XLA_MEM_FRACTION)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--policy-config", default=None)
    p.add_argument("--model-kind", default=None)
    p.add_argument("--target-contract", default=None)
    p.add_argument("--extra", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-missing", action="store_true", help="donor/iv 阶段有缺失 (slot, 接收局) 时仍生成")
    add_root_args(p)
    p = sub.add_parser("donors")
    p.add_argument("--out", required=True, help="<O>/<M>")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return {"pool": cmd_pool, "specs": cmd_specs, "donors": cmd_donors}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
