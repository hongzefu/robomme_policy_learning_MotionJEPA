#!/usr/bin/env python3
"""L6 闭环探针汇总（TIC 计划 2.5，审计第 3 条）：读 `serve_policy_probe.py` 的 jsonl + eval 的
`progress.json` + eval stdout 日志 + probe server 日志，出 7 条阻断判定行 + 1 条观察行 + 唯一成功行 `TIC_EVAL=`。

**为什么不能像 L1–L5 那样逐字节比**：评估跑 benchmark 的 test split，训练库里没有这些 episode，
没有「训练侧真值」可比。改比不变量——每一条都是「在线交付必须自洽且与训练口径同式」的硬约束：

| 判定行 | 性质 | 查什么 |
|---|---|---|
| `EVAL_EPISODE_MAP` | 阻断 | 探针 reset 次数 == eval 日志 episode 行数 == progress.json 条目数，且逐集 (task, ep) 顺序、`exec_start_idx`、`task_goal` 三重对上 |
| `EVAL_TAU_K` | 阻断 | 每集决策时刻是 `es, es+16, es+32, …`；k ≤ motion budget；报 τ_max / k_max / headroom |
| `EVAL_K_FORMULA` | 阻断 | 本文件内**独立重写** demo(es)+exec(es,τ) 起点公式，与探针记的 `motion_frames` 逐元素比 |
| `EVAL_ORDER_LEGAL` | 阻断 | 本文件内**独立重写** `key = 2·时刻 + 类型` 的稳定排序（不 import `shared/sampling`），与 `mem_order` 逐元素比；核排列合法性、dtype、长度、两条 mask 的前缀 True |
| `EVAL_BACKEND` | 阻断 | server 日志 `PROBE_ENV` 行：后端是 gpu、在线现算 pos 表与训练库 pos 小表逐位相同（`motion_gpu_override` 同行打印，留档核生产路径，不参与判定） |
| `EVAL_PROMPT` | 阻断 | 训练侧 4 任务 pkl 的 `prompt` 过同一 `PaligemmaTokenizer` 得 token 集合；在线 `tok_ids` 不在其中即 FAIL |
| `EVAL_NO_RAISE` | 阻断 | 没有任何一集因抛错被记成 `"error"`；timeout 由「该集 infer 次数 == ⌈(max_steps+1)/16⌉」判并与视频文件名交叉核 |
| `EVAL_DIST_OBS` | 观察 | 在线 k / τ 分布 vs 训练库按同一公式重算的分布 |

**eval 侧日志与 progress.json 的真实形制**（读 `examples/robomme/eval.py` 与 `docs/training-doc/eval-awsprod40k-b128-motion/records/` 实测）：
- eval stdout **没有逐集成功 / 失败 / timeout 结果行**。每集只有三行：
  `[robomme] env for task <task> episode <n> setup finished` / `task_goal: <goal>` / `exec_start_idx: <es>`；
  异常集另有 `Error evaluating episode <n> for task <task>: <e>`，续评时有 `... already evaluated, skipping...`。
- `progress.json` 是 `{task: {"<ep>": true|false|"error"}}`；**timeout 与普通失败都写 false**，无法区分。
- 三分（success / fail / timeout）的唯一来源是 `save_dir/videos/<task>_ep<n>_<flag>_<goal>_<difficulty>.mp4` 文件名
  （`eval.py::EpisodeEvaluator.eval_each_episode` 写、`legacy-eval/merge_eval_shards.py` 也这么读）。
  故 timeout 主判据取探针 infer 次数、交叉核取视频文件名，不指望日志里有 timeout 行。

用法：
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= UV_LINK_MODE=copy uv run --no-sync python \
    scripts/training/g0/summarize_eval_probe.py \
      --probe v1-store/reports/tic/eval_probe.jsonl \
      --progress v1-store/evaluation/<run>/ckpt39999/seed42/progress.json \
      --eval-log v1-store/logs/tic-evalprobe.log --server-log v1-store/logs/tic-evalprobe.server.log \
      --lib v1-store/datasets/4task-motion-400ep --budget 96 --max-steps 1300
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import pickle
import re
import statistics as stat
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if not (_REPO_ROOT / "pyproject.toml").is_file():
    raise SystemExit(f"错误: 仓库根解析失败 {_REPO_ROOT}（缺 pyproject.toml）")
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))

# ── 在线侧口径常量（与 history_config.resolved.yaml 一致；由 PROBE_ENV 行现场核对，不写死进判据）──
SENTINEL = int(np.iinfo(np.int32).max)          # 与 shared/sampling.MEM_ORDER_SENTINEL 同值，此处**独立定义**
MOTION_WINDOW = 33
MOTION_STRIDE = 16

_EP_RE = re.compile(r"^\[robomme\] env for task (?P<task>\S+) episode (?P<ep>\d+) setup finished\s*$")
_GOAL_RE = re.compile(r"^task_goal: (?P<goal>.*)$")
_ES_RE = re.compile(r"^exec_start_idx: (?P<es>\d+)\s*$")
_ERR_RE = re.compile(r"^Error evaluating episode (?P<ep>\d+) for task (?P<task>\S+): (?P<msg>.*)$")
_SKIP_RE = re.compile(r"^\[robomme\] episode (?P<ep>\d+) already evaluated, skipping\.\.\.\s*$")
_VIDEO_RE = re.compile(r"^(?P<task>[A-Za-z]+)_ep(?P<ep>\d+)_(?P<flag>[a-z]+)_")
_PROBE_ENV_RE = re.compile(r"\bPROBE_ENV\b(?P<rest>.*)$")


# --------------------------------------------------------------------------
# 独立重算（刻意不 import mme_vla_suite.shared.sampling —— 两侧各写一份才有对拍意义）
# --------------------------------------------------------------------------
def motion_frames_formula(es: int, t: int, window: int = MOTION_WINDOW, stride: int = MOTION_STRIDE) -> list[int]:
    """合法运动窗起点（全域帧号，升序）：demo 段 [0, es) 与 exec 段 [es, t] 各按 segment_start 网格 forward 取。"""
    out: list[int] = []
    s = 0
    while s + (window - 1) <= es - 1:
        out.append(s)
        s += stride
    u = 0
    while u + (window - 1) <= t - es:
        out.append(es + u)
        u += stride
    return out


def frames_sampled_formula(t: int, max_frames: int) -> list[int]:
    """帧路选帧（与 even_sampling_indices 同式，此处独立重写）。"""
    if t < max_frames:
        return list(range(t + 1))
    return [int(x) for x in np.linspace(0, t, max_frames, dtype=np.int32)]


def memory_order_formula(frames_sampled, motion_frames, max_frames: int, tokens_per_frame: int,
                         motion_budget: int) -> np.ndarray:
    """608 位交错次序：key = 时刻×2 + 类型（帧 0 / 运动 1），稳定排序；padding 位记哨兵落尾。"""
    ft = np.full(max_frames, SENTINEL, dtype=np.int64)
    ft[: len(frames_sampled)] = np.asarray(frames_sampled, dtype=np.int64)
    mt = np.full(motion_budget, SENTINEL, dtype=np.int64)
    mt[: len(motion_frames)] = np.asarray(motion_frames, dtype=np.int64)
    keys = np.concatenate([np.repeat(ft * 2 + 0, tokens_per_frame), mt * 2 + 1])
    return np.argsort(keys, kind="stable").astype(np.int32)


# --------------------------------------------------------------------------
# 读入
# --------------------------------------------------------------------------
def read_probe(path: pathlib.Path) -> list[dict]:
    recs = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"错误: probe jsonl 第 {ln} 行不可解析: {e}")
    if not recs:
        raise SystemExit(f"错误: probe jsonl 为空: {path}")
    return recs


def read_eval_log(path: pathlib.Path) -> dict:
    """按出现顺序抽 (task, ep, goal, es) 四元组，另收错误 / 跳过行。"""
    episodes: list[dict] = []
    errors: list[dict] = []
    skips: list[str] = []
    api_abort = 0
    cur: dict | None = None
    for raw in path.open(encoding="utf-8", errors="replace"):
        line = raw.rstrip("\n")
        m = _EP_RE.match(line)
        if m:
            cur = {"task": m.group("task"), "ep": int(m.group("ep")), "goal": None, "es": None}
            episodes.append(cur)
            continue
        if cur is not None:
            m = _GOAL_RE.match(line)
            if m and cur["goal"] is None:
                cur["goal"] = m.group("goal")
                continue
            m = _ES_RE.match(line)
            if m and cur["es"] is None:
                cur["es"] = int(m.group("es"))
                continue
        m = _ERR_RE.match(line)
        if m:
            errors.append({"task": m.group("task"), "ep": int(m.group("ep")), "msg": m.group("msg")})
            continue
        m = _SKIP_RE.match(line)
        if m:
            skips.append(line)
            continue
        if line.startswith("API calling error, aborting"):
            api_abort += 1
    return {"episodes": episodes, "errors": errors, "skips": skips, "api_abort": api_abort}


def read_probe_env(path: pathlib.Path) -> dict:
    """从 server 日志抓最后一条 PROBE_ENV 行，解析成 key=value 字典。"""
    found = None
    for raw in path.open(encoding="utf-8", errors="replace"):
        m = _PROBE_ENV_RE.search(raw)
        if m:
            found = m.group("rest")
    if found is None:
        raise SystemExit(f"错误: server 日志里没有 PROBE_ENV 行: {path}")
    out: dict[str, str] = {}
    for tok in found.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def count_tracebacks(path: pathlib.Path) -> tuple[int, int, list[str]]:
    """分块数 server 日志里的 Traceback，把 websockets 握手噪声与真实策略侧异常分开。

    起 server 的脚本用「connect 一下就关」探测端口就绪（`legacy-eval/eval_shard.*.sh` 与
    `tests/run_t3_eval_obs.sh` 都是），每探一次 websockets 就打一个
    `EOFError: connection closed while reading HTTP request line` / `handshake_exc` 的 Traceback。
    这类块里没有任何仓库代码帧，一律算噪声；只有块内出现 `mme_vla_suite/` 或 `/openpi/` 帧的才算真异常。
    """
    real = noise = 0
    heads: list[str] = []
    block: list[str] | None = None

    def flush(b: list[str] | None) -> None:
        nonlocal real, noise
        if not b:
            return
        body = "".join(b)
        if "mme_vla_suite/" in body or "/openpi/" in body or "site-packages/openpi" in body:
            real += 1
            tail = [x.strip() for x in b if x.strip()][-1:]
            heads.extend(tail)
        else:
            noise += 1

    for raw in path.open(encoding="utf-8", errors="replace"):
        if "Traceback (most recent call last)" in raw:
            flush(block)
            block = [raw]
            continue
        if block is not None:
            if raw.startswith((" ", "\t")) or raw.strip() == "" or ":" in raw:
                block.append(raw)
                if raw.strip() == "":
                    flush(block)
                    block = None
            else:
                flush(block)
                block = None
    flush(block)
    return real, noise, heads[:3]


def read_video_flags(videos_dir: pathlib.Path) -> dict[tuple[str, int], str]:
    flags: dict[tuple[str, int], str] = {}
    if not videos_dir.is_dir():
        return flags
    for mp4 in sorted(videos_dir.glob("*.mp4")):
        m = _VIDEO_RE.match(mp4.name)
        if m:
            flags[(m.group("task"), int(m.group("ep")))] = m.group("flag")
    return flags


def train_prompts(lib: pathlib.Path, pkl_root: pathlib.Path) -> dict[str, set[str]]:
    """按任务收训练侧 pkl 的 prompt 原文集合。

    每个 episode 的 `prompt` 由建库时 `task_goal.lower()` 一次写定（`build_robomme_dataset.py`），
    episode 内所有 exec 样本同值，故每集只读它的第一条 `data/<exec_sample_offset>.pkl`。
    """
    man = json.loads((lib / "meta" / "episode_manifest.json").read_text(encoding="utf-8"))
    per_task: dict[str, set[str]] = collections.defaultdict(set)
    for ep in man["episodes"]:
        if int(ep["exec_samples"]) <= 0:
            continue
        task = re.sub(r"^record_dataset_|\.h5$", "", ep["h5_file"])
        idx = int(ep["exec_sample_offset"])
        p = pkl_root / f"{idx}.pkl"
        if not p.is_file():
            raise SystemExit(f"错误: 训练 pkl 缺失: {p}（--train-pkl-root 指错？）")
        with p.open("rb") as f:
            d = pickle.load(f)
        per_task[task].add(str(d["prompt"]))
    return dict(per_task)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, help="serve_policy_probe.py 的 jsonl")
    ap.add_argument("--progress", required=True, help="<save_dir>/progress.json")
    ap.add_argument("--eval-log", required=True, help="eval.py 的 stdout/stderr 全量日志")
    ap.add_argument("--server-log", required=True, help="probe server 日志（含 PROBE_ENV 行）")
    ap.add_argument("--lib", default=str(_V1 / "datasets/4task-motion-400ep"), help="训练库根")
    ap.add_argument("--train-pkl-root", default=None,
                    help="训练 pkl 根（默认 <lib>/source/data，即 FrameSampDataset.__getitem__ 读的那一份）")
    ap.add_argument("--videos-dir", default=None, help="默认 <progress 所在目录>/videos，用于 timeout 交叉核")
    ap.add_argument("--budget", type=int, default=96, help="motion token 预算")
    ap.add_argument("--max-steps", type=int, default=1300, help="eval.py 的 --args.max_steps")
    ap.add_argument("--max-frames", type=int, default=32, help="帧路最多帧数 = budget/(token_per_image·num_views)")
    ap.add_argument("--tokens-per-frame", type=int, default=16, help="token_per_image × num_views")
    ap.add_argument("--expect-episodes", type=int, default=24, help="预期总集数（4 任务 × 6 集）")
    ap.add_argument("--expect-tasks", type=int, default=4)
    ap.add_argument("--out", default=None, help="JSON 报告落点（默认 v1-store/reports/tic/eval_probe_summary.json）")
    args = ap.parse_args()

    lib = pathlib.Path(args.lib)
    pkl_root = pathlib.Path(args.train_pkl_root) if args.train_pkl_root else lib / "source" / "data"
    progress_path = pathlib.Path(args.progress)
    videos_dir = pathlib.Path(args.videos_dir) if args.videos_dir else progress_path.parent / "videos"
    out_path = pathlib.Path(args.out) if args.out else _V1 / "reports" / "tic" / "eval_probe_summary.json"

    recs = read_probe(pathlib.Path(args.probe))
    elog = read_eval_log(pathlib.Path(args.eval_log))
    penv = read_probe_env(pathlib.Path(args.server_log))
    prog = json.loads(progress_path.read_text(encoding="utf-8"))
    vflags = read_video_flags(videos_dir)
    server_tracebacks, handshake_noise, tb_heads = count_tracebacks(pathlib.Path(args.server_log))

    by_ep: dict[int, list[dict]] = collections.OrderedDict()
    for r in recs:
        by_ep.setdefault(int(r["episode_seq"]), []).append(r)
    for seq, rs in by_ep.items():
        rs.sort(key=lambda x: int(x["infer_seq"]))

    report: dict = {"lines": [], "blocking": {}, "observe": {}}
    lines: list[str] = []
    blocking: dict[str, bool] = {}

    def emit(tag: str, ok: bool | None, body: str) -> None:
        if ok is None:
            line = f"{tag} {body}"
        else:
            line = f"{tag}={'PASS' if ok else 'FAIL'} {body}"
            blocking[tag] = bool(ok)
        lines.append(line)
        print(line, flush=True)

    # ── 1. EVAL_EPISODE_MAP ────────────────────────────────────────────────
    resets = len(by_ep)
    seqs = sorted(by_ep)
    seq_contiguous = seqs == list(range(1, resets + 1))
    log_eps = elog["episodes"]
    prog_entries = [(t, int(e)) for t, eps in prog.items() for e in eps]
    tasks_seen = sorted({e["task"] for e in log_eps})
    per_task_counts = collections.Counter(e["task"] for e in log_eps)
    n_pt = sorted(set(per_task_counts.values()))
    one_to_one = 1
    es_match = goal_match = 0
    map_detail: list[dict] = []
    if resets == len(log_eps) and resets == len(prog_entries) and seq_contiguous:
        for i, le in enumerate(log_eps):
            rs = by_ep[i + 1]
            es_ok = all(int(r["es"]) == int(le["es"]) for r in rs) if le["es"] is not None else False
            goal_ok = all(str(r["prompt_text"]) == str(le["goal"]) for r in rs) if le["goal"] is not None else False
            in_prog = (le["task"], le["ep"]) in set(prog_entries)
            es_match += int(es_ok)
            goal_match += int(goal_ok)
            if not (es_ok and goal_ok and in_prog):
                one_to_one = 0
            map_detail.append({"episode_seq": i + 1, "task": le["task"], "ep": le["ep"],
                               "es_ok": es_ok, "goal_ok": goal_ok, "in_progress": in_prog})
    else:
        one_to_one = 0
    map_ok = bool(
        one_to_one == 1
        and resets == args.expect_episodes
        and len(tasks_seen) == args.expect_tasks
        and not elog["skips"]
    )
    emit("EVAL_EPISODE_MAP", map_ok,
         f"resets={resets} tasks={len(tasks_seen)} per_task={','.join(str(x) for x in n_pt) or 'NA'} "
         f"log_episodes={len(log_eps)} progress_entries={len(prog_entries)} one_to_one={one_to_one} "
         f"es_match={es_match}/{len(log_eps)} goal_match={goal_match}/{len(log_eps)} "
         f"seq_contiguous={int(seq_contiguous)} resumed_skips={len(elog['skips'])} "
         f"expect_episodes={args.expect_episodes} expect_tasks={args.expect_tasks}")
    report["episode_map"] = map_detail

    # ── 2. EVAL_TAU_K ──────────────────────────────────────────────────────
    infer_points = len(recs)
    tau_max = max(int(r["t"]) for r in recs)
    k_max = max(int(r["k"]) for r in recs)
    es_values = sorted({int(r["es"]) for r in recs})
    tau_bad: list[dict] = []
    for seq, rs in by_ep.items():
        for j, r in enumerate(rs):
            want = int(r["es"]) + 16 * j
            if int(r["t"]) != want or int(r["infer_seq"]) != j + 1:
                tau_bad.append({"episode_seq": seq, "infer_seq": r["infer_seq"], "t": r["t"], "want_t": want})
    k_over = [r for r in recs if int(r["k"]) > args.budget]
    k_len_bad = [r for r in recs if int(r["k"]) != len(r["motion_frames"])]
    tau_ok = not tau_bad and not k_over and not k_len_bad
    emit("EVAL_TAU_K", tau_ok,
         f"episodes={resets} infer_points={infer_points} tau_max={tau_max} k_max={k_max} budget={args.budget} "
         f"headroom={args.budget - k_max} es_values={','.join(str(x) for x in es_values)} "
         f"tau_mismatches={len(tau_bad)} "
         f"k_over_budget={len(k_over)} k_len_mismatch={len(k_len_bad)}")
    if args.budget - k_max == 0:
        print("  ⚠ headroom=0：再长一帧就会触发 FrameSampMemory._prepare_motion 的零截断 raise", flush=True)

    # ── 3. EVAL_K_FORMULA ──────────────────────────────────────────────────
    kf_mis = 0
    fs_mis = 0
    for r in recs:
        want = motion_frames_formula(int(r["es"]), int(r["t"]))
        if [int(x) for x in r["motion_frames"]] != want:
            kf_mis += 1
        want_fs = frames_sampled_formula(int(r["t"]), args.max_frames)
        if [int(x) for x in r["frames_sampled"]] != want_fs:
            fs_mis += 1
    emit("EVAL_K_FORMULA", kf_mis == 0 and fs_mis == 0,
         f"points={infer_points} mismatches={kf_mis} frames_sampled_mismatches={fs_mis} "
         f"window={MOTION_WINDOW} stride={MOTION_STRIDE} max_frames={args.max_frames}")

    # ── 4. EVAL_ORDER_LEGAL ────────────────────────────────────────────────
    n_slots = args.max_frames * args.tokens_per_frame + args.budget
    nonperm = 0
    dtype_bad = 0
    len_bad = 0
    order_mis = 0
    smask_bad = 0
    mmask_bad = 0
    for r in recs:
        order = np.asarray(r["mem_order"], dtype=np.int64)
        if str(r["mem_order_dtype"]) != "int32":
            dtype_bad += 1
        if int(r["mem_order_len"]) != n_slots or order.size != n_slots:
            len_bad += 1
            continue
        if not np.array_equal(np.sort(order), np.arange(n_slots, dtype=np.int64)):
            nonperm += 1
        want = memory_order_formula(r["frames_sampled"], r["motion_frames"], args.max_frames,
                                    args.tokens_per_frame, args.budget)
        if not np.array_equal(order.astype(np.int32), want):
            order_mis += 1
        sm = str(r["static_mask"])
        n_true = len(r["frames_sampled"]) * args.tokens_per_frame
        if len(sm) != n_slots - args.budget or sm != "1" * n_true + "0" * (len(sm) - n_true):
            smask_bad += 1
        mm = str(r["motion_mask"])
        k = int(r["k"])
        if len(mm) != args.budget or mm != "1" * k + "0" * (args.budget - k):
            mmask_bad += 1
    order_ok = not (nonperm or dtype_bad or len_bad or order_mis or smask_bad or mmask_bad)
    emit("EVAL_ORDER_LEGAL", order_ok,
         f"points={infer_points} nonperm={nonperm} dtype_int32={int(dtype_bad == 0)} len{n_slots}={int(len_bad == 0)} "
         f"expected_order_mismatches={order_mis} static_mask_bad={smask_bad} motion_mask_bad={mmask_bad}")

    # ── 5. EVAL_BACKEND ────────────────────────────────────────────────────
    backend = penv.get("backend", "?")
    pt_sha = penv.get("pos_table_sha", "")
    st_sha = penv.get("store_pos_sha", "")
    pos_equal = int(bool(pt_sha) and pt_sha == st_sha)
    backend_ok = bool(backend == "gpu" and pos_equal == 1)
    emit("EVAL_BACKEND", backend_ok,
         f"backend={backend} pos_rows={penv.get('pos_rows')} pos_table_sha={pt_sha[:16]}… "
         f"store_pos_sha={st_sha[:16]}… equal={pos_equal} motion_enabled={penv.get('motion_enabled')} "
         f"motion_gpu_override={penv.get('motion_gpu_override')} devices={penv.get('devices')}")

    # ── 6. EVAL_PROMPT ─────────────────────────────────────────────────────
    from mme_vla_suite.training.config import PaligemmaTokenizer  # 与在线 model_transforms 同一个类

    max_token_len = int(penv.get("max_token_len", 64))
    tok = PaligemmaTokenizer(max_token_len)
    tr_prompts = train_prompts(lib, pkl_root)
    all_train_texts = sorted({p for ps in tr_prompts.values() for p in ps})
    # `discrete_state_input`（PROBE_ENV 现场给出）决定 tokenize 是否吃 state：
    #  - 0（本 run）：token 只由文本决定，可预建 token 集合一次比全部记录；
    #  - 1：token 还依赖当刻离散化 state，只能逐记录用 state_tok 重算候选。
    dsi = penv.get("discrete_state_input", "0") == "1"
    state_dim = int(penv.get("state_dim", 8))
    train_tok: dict[tuple[int, ...], str] = {}
    if not dsi:
        for p in all_train_texts:
            ids, _ = tok.tokenize(p, None)
            train_tok[tuple(int(x) for x in np.asarray(ids).reshape(-1))] = p
    in_set = 0
    text_in_set = 0
    online_texts = sorted({str(r["prompt_text"]) for r in recs})
    bad_examples: list[dict] = []
    for r in recs:
        key = tuple(int(x) for x in r["tok_ids"])
        if not dsi:
            hit = key in train_tok
        else:
            stv = np.asarray(r["state_tok"], dtype=np.float32)[:state_dim]
            hit = any(tuple(int(x) for x in np.asarray(tok.tokenize(p, stv)[0]).reshape(-1)) == key
                      for p in all_train_texts)
        if hit:
            in_set += 1
        elif len(bad_examples) < 3:
            bad_examples.append({"episode_seq": r["episode_seq"], "infer_seq": r["infer_seq"],
                                 "prompt_text": r["prompt_text"], "tok_ids_head": r["tok_ids"][:12]})
        if str(r["prompt_text"]) in all_train_texts:
            text_in_set += 1
    prompt_ok = in_set == infer_points
    emit("EVAL_PROMPT", prompt_ok,
         f"tasks={len(tr_prompts)} episodes={resets} distinct_prompt_text={len(online_texts)} "
         f"train_distinct_prompt={len(all_train_texts)} train_distinct_tok={len(train_tok) if not dsi else 'NA(state-dependent)'} "
         f"tok_in_trainset={in_set}/{infer_points} text_in_trainset={text_in_set}/{infer_points} "
         f"discrete_state_input={int(dsi)} "
         f"tokenizer=PaligemmaTokenizer(max_len={max_token_len},strip,underscore,newline;no_lower)")
    for b in bad_examples:
        print(f"  ✗ 在线 prompt 不在训练集 token 集合: {b}", flush=True)
    report["prompt_bad_examples"] = bad_examples

    # ── 7. EVAL_NO_RAISE ───────────────────────────────────────────────────
    prog_errors = [(t, e) for t, eps in prog.items() for e, v in eps.items() if v == "error"]
    prog_unknown = [(t, e, v) for t, eps in prog.items() for e, v in eps.items()
                    if not isinstance(v, bool) and v != "error"]
    expect_timeout_infers = -(-(args.max_steps + 1) // 16)          # ⌈(max_steps+1)/16⌉
    probe_timeout_seqs = [seq for seq, rs in by_ep.items() if len(rs) >= expect_timeout_infers]
    video_timeouts = {k for k, v in vflags.items() if v == "timeout"}
    cross = 0
    if map_detail:
        seq2te = {d["episode_seq"]: (d["task"], d["ep"]) for d in map_detail}
        cross = sum(1 for s in probe_timeout_seqs if seq2te.get(s) in video_timeouts)
    no_raise_ok = bool(
        not prog_errors and not prog_unknown and not elog["errors"]
        and elog["api_abort"] == 0 and server_tracebacks == 0
    )
    emit("EVAL_NO_RAISE", no_raise_ok,
         f"episodes={resets} errors={len(prog_errors)} timeouts={len(probe_timeout_seqs)} "
         f"unknown={len(prog_unknown)} log_error_lines={len(elog['errors'])} api_abort={elog['api_abort']} "
         f"server_tracebacks={server_tracebacks} handshake_noise={handshake_noise} "
         f"expect_timeout_infers={expect_timeout_infers} "
         f"timeout_video_cross={cross}/{len(probe_timeout_seqs)} videos_seen={len(vflags)}")
    for e in elog["errors"][:3]:
        print(f"  ✗ eval 日志错误行: {e}", flush=True)
    for h in tb_heads:
        print(f"  ✗ server 侧异常: {h}", flush=True)

    # ── 8. EVAL_DIST_OBS（观察）────────────────────────────────────────────
    ks = [int(r["k"]) for r in recs]
    man = json.loads((lib / "meta" / "episode_manifest.json").read_text(encoding="utf-8"))
    train_ks: list[int] = []
    train_tau_max = 0
    for ep in man["episodes"]:
        es = int(ep["exec_start_idx"])
        T = int(ep["num_timesteps"])
        train_tau_max = max(train_tau_max, T - 1)
        for t in range(es, T):
            train_ks.append(len(motion_frames_formula(es, t)))
    emit("EVAL_DIST_OBS", None,
         f"online_k_median={stat.median(ks)} mean={stat.mean(ks):.2f} max={max(ks)} | "
         f"train_k_median={stat.median(train_ks)} mean={stat.mean(train_ks):.2f} max={max(train_ks)} "
         f"train_samples={len(train_ks)} | online_tau_max={tau_max} train_tau_max={train_tau_max}")

    # ── 唯一成功行 ─────────────────────────────────────────────────────────
    n_pass = sum(1 for v in blocking.values() if v)
    n_tot = len(blocking)
    ok = n_pass == n_tot
    head = (f"TIC_EVAL={'PASS' if ok else 'FAIL'} episodes={resets} tasks={len(tasks_seen)} "
            f"infer_points={infer_points} blocking={n_pass}/{n_tot} observe=1")
    lines.append(head)
    print(head, flush=True)
    if not ok:
        print("  FAIL 项: " + ", ".join(k for k, v in blocking.items() if not v), file=sys.stderr, flush=True)

    report["lines"] = lines
    report["blocking"] = blocking
    report["probe_env"] = penv
    report["online_prompts"] = online_texts
    report["train_prompts"] = {k: sorted(v) for k, v in tr_prompts.items()}
    report["all_train_texts"] = all_train_texts
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"报告落 {out_path}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
