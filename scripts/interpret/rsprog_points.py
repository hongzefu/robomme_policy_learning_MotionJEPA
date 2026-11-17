#!/usr/bin/env python3
"""RouteStick 进度探针（计划 1117-routestick-progress-probe-attn-plan.html，子任务 R1）：局池、spec、取点、标签、
同帧异史配对、记忆 token 归段表。

一、局池与 spec（``build-pool``）
  全局序号 k = 0..199，难度按 k 分段：k // 50 → easy / medium / hard / xhard；源 seed = 3400000 + 100·k，
  只用于 ``samep_rs_variants.rebuild_spec`` 复算路线与布局（与 steering 同一套函数，难度显式传入）；
  每难度内局号 episode = 1000 + k % 50。环境 seed 由生成脚本按 seed_layout 的布局公式由局号现算，与源 seed 无关。
  每条路线逐条过 ``samep_rs_variants.check_spec``。输出：
    <out>/specs/<difficulty>.json  单组规格文件（顶层 spec_schema_version；记录含 SPEC_RECORD_REQUIRED 全部键，
                                   spec_sha256 按子模块 spec_record_sha256 同一算法：剔除 spec_sha256/collision 后
                                   canonical JSON 的 sha256）
    <out>/specs/manifest.json      清单（manifest_version=1，4 组 RouteStick × 难度，各指向同目录的单组文件，
                                   episodes 1000–1049）。子模块 --difficulty 只接受 easy/medium/hard 比例串，
                                   xhard 只能经清单生成：``--episode-specs manifest.json``，产物落
                                   ``<output>/RouteStick/<difficulty>/``
    <out>/pool.json                sidecar：每局 k、seed（源 seed）、difficulty、episode、L、rotation_deg、
                                   node_slots、nodes、directions、sides（每步 left/right）、moves（每步 A/B/C/D）、
                                   spec_sha256
  末行打印 ``SPEC_LOAD=PASS|FAIL``：用子模块 load_episode_specs 原样加载清单（4 组 × 50 局）与 4 份单组文件
  （单组只对 easy/medium/hard 走 CLI 难度一致检查；xhard 单组按文件难度加载）。

  左右（sides）：与 RouteStick._load_scene 内 ``_stick_side`` 同式——目标格 y 坐标大于上一格 y 坐标记 left，
  否则 right；格中心坐标按 ``rs_full_prep.cell_xy`` 同式（1 行 9 列、间距 0.07、中心 (−0.1, 0)，绕原点转 theta）。
  走法四类（与 H5 setup/available_multi_choices 一致，官方 data_0226 实查）：
    A = left + clockwise，B = right + clockwise，C = left + counterclockwise，D = right + counterclockwise。

二、取点与标签（``points``）
  输入 pool.json 与生成产物根：每难度一个 ``record_dataset_RouteStick.h5``，默认路径
  ``<data-root>/RouteStick/<difficulty>/record_dataset_RouteStick.h5``（清单生成的布局），也接受
  ``<data-root>/<difficulty>/record_dataset_RouteStick.h5``。H5 内局名 ``episode_<局号>``（生成脚本按 job.episode
  命名），与 pool 的 (difficulty, episode) 一一对应；setup/seed 是环境 seed，原样记入输出。
  帧时间一律是 H5 的 timestep 序号（0 起，连续）。
    e0 = 第一帧 is_video_demo == False；示范段边界 = [0, e0) 内 is_subgoal_boundary 帧；
    执行段边界 = [e0, 总帧数) 内 is_subgoal_boundary 且 is_completed 为假的帧（is_completed 为真的那条边界是
    完成标记，官方示教里落在末段倒数第 7 帧，不是步边界）。
    推理点 u = 0, 16, 32, …，u < 执行段长（= 总帧数 − e0；规则局即 50L），帧 t = e0 + u。
    j = 点所在的执行步（0 起，按实际边界），τ = (j + (t − b_j)/(b_{j+1} − b_j)) / L（规则局等于 u/(50L)），
    tau_time = u/(50L)，remaining = L − j，cur_move = moves[j]，next_move = moves[j+1]（最后一步为 null），
    button_slot / button_cell = 本步出发按钮（node_slots[j] / nodes[j]），target_slot = node_slots[j+1]，
    step_start = 该点是否本步第一个推理点（t − b_j < 16）。
  GEN_SPEC 逐局核：示范段长 = 50L、示范边界恰为 0, 50, …, 50(L−1)、示范与执行每帧 choice_action 等于 spec 走法、
  is_video_demo 只在 e0 处翻转一次（之前全真、之后全假）、执行步边界数 = L 且首边界 = e0。不符即剔除并计数；
  执行段边界数对、但段长不全为 50 或执行段长 ≠ 50L 的局按实际边界算标签，计入 irregular，不算 mismatch。
  H5 缺局计入 missing（也算剔除）。剔除数 > 10% 局池时 FAIL。
  判定行：``GEN_SPEC=PASS|FAIL eps=<保留局数> mismatches=<不符局数> dropped=<剔除局数> irregular=<n> missing=<n>``

  同帧异史配对（计划 §4、B2）：同一局内两个 step_start 点、button_slot 相同（j 不同），帧 t 的 front_rgb 与
  wrist_rgb 逐像素最大差都 ≤ 8，记为严格档配对。判定行 ``PAIRS n=<同按钮候选对数> kept=<过像素阈值对数>``。

  输出 rsprog_points.json 模式（schema "rsprog-points-1"）：
    {"schema": .., "pool": <pool.json 路径>, "data_root": .., "pix_tol": 8, "stride": 16, "seglen": 50,
     "episodes": [{"k", "seed", "env_seed", "difficulty", "episode", "L", "node_slots", "nodes", "directions",
                   "sides", "moves", "demo_bounds": [[a, b], …]（闭区间，示范第 j′=1..L 段）,
                   "exec_bounds": [[a, b], …]（闭区间，执行第 j=0..L−1 步，末步止于总帧数−1）,
                   "e0", "n_frames", "irregular"}],
     "points": [{"k", "difficulty", "episode", "t", "u", "j", "tau", "tau_time", "remaining", "cur_move",
                 "next_move", "button_slot", "button_cell", "target_slot", "step_start"}],
     "pairs": [{"k", "t_a", "t_b", "j_a", "j_b", "button_slot", "front_maxdiff", "wrist_maxdiff"}],
     "dropped": [{"k", "difficulty", "episode", "reason"}],
     "verdicts": {"GEN_SPEC": <判定行>, "PAIRS": <判定行>}}

三、记忆 token 归段（``mem_token_segments``，供 R3 的 rsprog_attn 用）
  mem_type 0 = 帧 token（mem_t0 = mem_t1 = 帧时间），1 = motion token（窗口首、末帧），−1 = 补齐。
  返回同形状 int 数组：帧 token = 所在示范段 j′（1 起，执行帧为 0）；motion token = 窗口内落在同一示范段的帧数
  ≥ ⌈2/3·窗口帧数⌉（33 帧窗口即 ≥ 22 帧）的那段，否则 0（跨段过渡或执行段）；补齐 = −1。
  口径复用 ``analysis/subgoal_attention.py`` 的 segment_of_frame / segments_of_window（逐帧段号表、窗口截到 e0），
  本文件只包一层把 demo_bounds 变成逐帧段号表，不改原文件。

用法（worktree 环境见计划 B4）：
  python -m scripts.interpret.rsprog_points build-pool --out <目录> [--seed-start 3400000] [--per-difficulty 50]
      [--episode-start 1000] [--bench-root <benchmark 仓库根>]
  python -m scripts.interpret.rsprog_points points --pool <out>/pool.json --data-root <生成产物根>
      --out <rsprog_points.json> [--stride 16] [--pix-tol 8]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

from scripts.interpret import samep_rs_variants as V

TASK = "RouteStick"
DIFFICULTIES = ("easy", "medium", "hard", "xhard")
SEED_START = 3_400_000
SEED_STRIDE = 100
PER_DIFFICULTY = 50
EPISODE_START = 1000
SEGLEN = V.SEGLEN                     # 50 帧一步
STRIDE = 16                           # 评测节奏：执行段每 16 步推理一次
PIX_TOL = 8
MAX_DROP_FRAC = 0.10
MOTION_MAJORITY = 2.0 / 3.0           # 33 帧窗口 → ≥ 22 帧
GRID_CENTER = (-0.1, 0.0)
SPACING = 0.07
N_COLS = 9
MOVES = {("left", V.CW): "A", ("right", V.CW): "B", ("left", V.CCW): "C", ("right", V.CCW): "D"}
H5_NAME = f"record_dataset_{TASK}.h5"
SPEC_SCHEMA_VERSION = 1
MANIFEST_VERSION = 1
GENERATOR_VERSION = "rsprog-points-1117"
POINTS_SCHEMA = "rsprog-points-1"
REPO = Path(__file__).resolve().parents[2]
MAIN_CHECKOUT = Path("/data/hongzefu/robomme_policy_learning_MotionJEPA")
DEFAULT_BENCH_ROOT = MAIN_CHECKOUT / "third_party/robomme_benchmark"


# ---------------------------------------------------------------- spec 散列（与子模块 spec_record_sha256 同算法）
def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def spec_sha256(record):
    payload = {k: v for k, v in record.items() if k not in ("spec_sha256", "collision")}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_generator(bench_root=DEFAULT_BENCH_ROOT):
    """按文件路径只读导入子模块 scripts/generate_dataset_newseed.py（不改子模块）。"""
    path = Path(bench_root) / "scripts/generate_dataset_newseed.py"
    if not path.is_file():
        raise SystemExit(f"找不到生成脚本：{path}")
    name = "_rsprog_generate_dataset_newseed"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 局池
def cell_xy(cell, theta):
    """格中心世界坐标（与 rs_full_prep.cell_xy、RouteStick._load_scene 同式）；theta 为弧度。"""
    x0 = GRID_CENTER[0]
    y0 = GRID_CENTER[1] + (int(cell) - (N_COLS - 1) / 2) * SPACING
    return (x0 * math.cos(theta) - y0 * math.sin(theta), x0 * math.sin(theta) + y0 * math.cos(theta))


def route_sides(nodes, rotation_deg):
    """每步左右：目标格 y > 上一格 y 记 left（RouteStick._stick_side 同式）。"""
    th = math.radians(float(rotation_deg))
    out = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        out.append("left" if cell_xy(b, th)[1] > cell_xy(a, th)[1] else "right")
    return out


def route_moves(sides, directions):
    return [MOVES[(s, d)] for s, d in zip(sides, directions)]


def difficulty_of(k, per_difficulty=PER_DIFFICULTY):
    return DIFFICULTIES[int(k) // int(per_difficulty)]


def build_pool(seed_start=SEED_START, per_difficulty=PER_DIFFICULTY, episode_start=EPISODE_START,
               seed_stride=SEED_STRIDE):
    """返回 (pool 行列表, {难度: [spec 记录]})；每条路线过 check_spec，不过即抛错。"""
    rows, specs = [], {d: [] for d in DIFFICULTIES}
    for k in range(len(DIFFICULTIES) * int(per_difficulty)):
        d = difficulty_of(k, per_difficulty)
        seed = int(seed_start) + int(seed_stride) * k
        ep = int(episode_start) + k % int(per_difficulty)
        rec = V.rebuild_spec(seed, d, ep)
        bad = V.check_spec(rec, d)
        if bad:
            raise ValueError(f"k={k} seed={seed} {d}: check_spec 不过 {bad}")
        rec["spec_sha256"] = spec_sha256(rec)
        a = rec["actions"]
        sides = route_sides(a["nodes"], rec["layout"]["rotation_deg"])
        rows.append({"k": k, "seed": seed, "difficulty": d, "episode": ep, "L": int(rec["objects"]["L"]),
                     "rotation_deg": rec["layout"]["rotation_deg"], "node_slots": list(a["node_slots"]),
                     "nodes": list(a["nodes"]), "directions": list(a["directions"]), "sides": sides,
                     "moves": route_moves(sides, a["directions"]), "spec_sha256": rec["spec_sha256"]})
        specs[d].append(rec)
    return rows, specs


def spec_document(difficulty, records, seed_start=SEED_START):
    sampling = {"cfg": V.CFG[difficulty], "route": V.ROUTE, "seglen": SEGLEN}
    return {"spec_schema_version": SPEC_SCHEMA_VERSION, "task": TASK, "difficulty": difficulty,
            "generator_seed": int(seed_start), "derived_seed": DIFFICULTIES.index(difficulty),
            "generator_version": GENERATOR_VERSION,
            "sampling_config_sha256": hashlib.sha256(canonical_json(sampling).encode("utf-8")).hexdigest(),
            "episodes": records}


def write_specs(rows, specs, out, seed_start=SEED_START):
    out = Path(out)
    sdir = out / "specs"
    sdir.mkdir(parents=True, exist_ok=True)
    groups = []
    for d in DIFFICULTIES:
        (sdir / f"{d}.json").write_text(json.dumps(spec_document(d, specs[d], seed_start), ensure_ascii=False,
                                                   indent=1), encoding="utf-8")
        groups.append({"task": TASK, "difficulty": d, "spec_path": f"{d}.json",
                       "episodes": [int(r["episode"]) for r in specs[d]]})
    (sdir / "manifest.json").write_text(json.dumps({"manifest_version": MANIFEST_VERSION, "groups": groups},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "pool.json").write_text(json.dumps({"schema": "rsprog-pool-1", "seed_start": int(seed_start),
                                               "episodes": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return sdir


def spec_load_check(sdir, bench_root=DEFAULT_BENCH_ROOT):
    """用子模块 load_episode_specs 原样加载清单与 4 份单组文件；返回 (ok, 说明行列表)。"""
    gen = load_generator(bench_root)
    sdir = Path(sdir)
    lines, ok = [], True
    groups = gen.load_episode_specs(sdir / "manifest.json", Path(bench_root), Path(bench_root) / "_rsprog_out")
    got = sorted((g.task, g.difficulty, len(g.episodes)) for g in groups)
    want = sorted((TASK, d, len(json.loads((sdir / f"{d}.json").read_text())["episodes"])) for d in DIFFICULTIES)
    if got != want:
        ok = False
    for g in groups:
        lines.append(f"manifest {g.task}/{g.difficulty}: episodes={len(g.episodes)} "
                     f"[{min(g.episodes)}..{max(g.episodes)}] out={Path(g.output_root).name}")
        for e in g.episodes:
            if spec_sha256(dict(g.records[e])) != g.records[e]["spec_sha256"]:
                ok = False
    for d in DIFFICULTIES:
        cli = d if d != "xhard" else None
        single = gen.load_episode_specs(sdir / f"{d}.json", Path(bench_root), Path(bench_root) / "_rsprog_out",
                                        task=TASK, difficulty=cli)
        if len(single) != 1 or single[0].difficulty != d:
            ok = False
    return ok, lines


# ---------------------------------------------------------------- H5 取点
def _scalar(ds):
    v = ds[()]
    return v.decode() if isinstance(v, bytes) else v


def read_episode_meta(group):
    """逐帧读 is_video_demo、is_subgoal_boundary、is_completed、choice（A–D）。"""
    names = [k for k in group if k.startswith("timestep_")]
    idx = sorted(int(k.split("_")[1]) for k in names)
    if idx != list(range(len(idx))):
        raise ValueError("timestep 序号不连续")
    demo, bnd, comp, choice = [], [], [], []
    for i in idx:
        ts = group[f"timestep_{i}"]
        demo.append(bool(ts["info/is_video_demo"][()]))
        bnd.append(bool(ts["info/is_subgoal_boundary"][()]))
        comp.append(bool(ts["info/is_completed"][()]))
        choice.append(json.loads(_scalar(ts["action/choice_action"])).get("choice"))
    return {"demo": demo, "bnd": bnd, "comp": comp, "choice": choice,
            "env_seed": int(group["setup/seed"][()]) if "setup/seed" in group else None}


def episode_bounds(meta, L, moves, seglen=SEGLEN):
    """返回 (信息字典, 问题列表)；问题非空即该局 mismatch。"""
    demo, bnd, comp, choice = meta["demo"], meta["bnd"], meta["comp"], meta["choice"]
    n = len(demo)
    problems = []
    if not n or not demo[0]:
        return None, ["首帧不是示范帧"]
    e0 = next((i for i, v in enumerate(demo) if not v), None)
    if e0 is None:
        return None, ["没有执行帧"]
    if any(demo[e0:]):
        problems.append("is_video_demo 在 e0 之后又出现真")
    if e0 != seglen * L:
        problems.append(f"示范段长 {e0} ≠ 50L={seglen * L}")
    dstarts = [i for i in range(e0) if bnd[i]]
    if dstarts != [seglen * j for j in range(L)]:
        problems.append(f"示范边界 {dstarts[:12]} 不是 0,50,…")
    demo_bounds = [[a, b - 1] for a, b in zip(dstarts, dstarts[1:] + [e0])] if dstarts else []
    estarts = [i for i in range(e0, n) if bnd[i] and not comp[i]]
    if len(estarts) != L or (estarts and estarts[0] != e0):
        problems.append(f"执行步边界 {estarts[:12]} 数目或起点不符（L={L}，e0={e0}）")
    exec_bounds = [[a, b - 1] for a, b in zip(estarts, estarts[1:] + [n])]
    if len(demo_bounds) == L:
        for j, (a, b) in enumerate(demo_bounds):
            if any(c != moves[j] for c in choice[a:b + 1]):
                problems.append(f"示范第 {j + 1} 段 choice 与 spec {moves[j]} 不符")
    if len(exec_bounds) == L:
        for j, (a, b) in enumerate(exec_bounds):
            if any(c != moves[j] for c in choice[a:b + 1]):
                problems.append(f"执行第 {j} 步 choice 与 spec {moves[j]} 不符")
    irregular = (n - e0 != seglen * L) or any(b - a + 1 != seglen for a, b in exec_bounds)
    info = {"e0": e0, "n_frames": n, "demo_bounds": demo_bounds, "exec_bounds": exec_bounds,
            "irregular": bool(irregular)}
    return info, problems


def episode_points(row, info, stride=STRIDE, seglen=SEGLEN):
    """按评测节奏取点并打标签。"""
    L, e0, n = int(row["L"]), info["e0"], info["n_frames"]
    eb = info["exec_bounds"]
    pts = []
    for u in range(0, n - e0, int(stride)):
        t = e0 + u
        j = max(i for i, (a, _) in enumerate(eb) if a <= t)
        a, b = eb[j]
        frac = (t - a) / float(b - a + 1)
        pts.append({"k": row["k"], "difficulty": row["difficulty"], "episode": row["episode"], "t": t, "u": u,
                    "j": j, "tau": (j + frac) / L, "tau_time": u / float(seglen * L), "remaining": L - j,
                    "cur_move": row["moves"][j], "next_move": row["moves"][j + 1] if j + 1 < L else None,
                    "button_slot": int(row["node_slots"][j]), "button_cell": int(row["nodes"][j]),
                    "target_slot": int(row["node_slots"][j + 1]), "step_start": (t - a) < int(stride)})
    return pts


def _maxdiff(group, ta, tb, key):
    x = np.asarray(group[f"timestep_{ta}/obs/{key}"][()], dtype=np.int16)
    y = np.asarray(group[f"timestep_{tb}/obs/{key}"][()], dtype=np.int16)
    return int(np.abs(x - y).max())


def same_frame_pairs(group, pts, pix_tol=PIX_TOL):
    """同局、两个 step_start 点、出发按钮相同；返回 (候选数, 过阈值配对列表)。"""
    starts = [p for p in pts if p["step_start"]]
    n, kept = 0, []
    for i in range(len(starts)):
        for k in range(i + 1, len(starts)):
            p, q = starts[i], starts[k]
            if p["button_slot"] != q["button_slot"] or p["j"] == q["j"]:
                continue
            n += 1
            fd = _maxdiff(group, p["t"], q["t"], "front_rgb")
            wd = _maxdiff(group, p["t"], q["t"], "wrist_rgb")
            if fd <= pix_tol and wd <= pix_tol:
                kept.append({"k": p["k"], "t_a": p["t"], "t_b": q["t"], "j_a": p["j"], "j_b": q["j"],
                             "button_slot": p["button_slot"], "front_maxdiff": fd, "wrist_maxdiff": wd})
    return n, kept


def find_h5(data_root, difficulty):
    root = Path(data_root)
    for cand in (root / TASK / difficulty / H5_NAME, root / difficulty / H5_NAME):
        if cand.is_file():
            return cand
    return None


def run_points(pool_path, data_root, stride=STRIDE, pix_tol=PIX_TOL, seglen=SEGLEN):
    import h5py
    rows = json.loads(Path(pool_path).read_text())["episodes"]
    by_d = {}
    for r in rows:
        by_d.setdefault(r["difficulty"], []).append(r)
    episodes, points, pairs, dropped = [], [], [], []
    mism = irregular = missing = n_cand = 0
    for d in DIFFICULTIES:
        if d not in by_d:
            continue
        path = find_h5(data_root, d)
        handle = h5py.File(path, "r") if path is not None else None
        try:
            for r in by_d[d]:
                name = f"episode_{r['episode']}"
                if handle is None or name not in handle:
                    missing += 1
                    dropped.append({"k": r["k"], "difficulty": d, "episode": r["episode"], "reason": "missing"})
                    continue
                group = handle[name]
                meta = read_episode_meta(group)
                info, problems = episode_bounds(meta, int(r["L"]), r["moves"], seglen)
                if problems:
                    mism += 1
                    dropped.append({"k": r["k"], "difficulty": d, "episode": r["episode"],
                                    "reason": "; ".join(problems)})
                    continue
                irregular += int(info["irregular"])
                pts = episode_points(r, info, stride, seglen)
                c, kept = same_frame_pairs(group, pts, pix_tol)
                n_cand += c
                pairs += kept
                points += pts
                episodes.append({**{k: r[k] for k in ("k", "seed", "difficulty", "episode", "L", "node_slots",
                                                      "nodes", "directions", "sides", "moves")},
                                 "env_seed": meta["env_seed"], **info})
        finally:
            if handle is not None:
                handle.close()
    n_drop = len(dropped)
    ok = n_drop <= MAX_DROP_FRAC * len(rows)
    gen = (f"GEN_SPEC={'PASS' if ok else 'FAIL'} eps={len(episodes)} mismatches={mism} dropped={n_drop} "
           f"irregular={irregular} missing={missing}")
    pl = f"PAIRS n={n_cand} kept={len(pairs)}"
    return {"schema": POINTS_SCHEMA, "pool": str(pool_path), "data_root": str(data_root), "pix_tol": pix_tol,
            "stride": stride, "seglen": seglen, "episodes": episodes, "points": points, "pairs": pairs,
            "dropped": dropped, "verdicts": {"GEN_SPEC": gen, "PAIRS": pl}}


# ---------------------------------------------------------------- 记忆 token 归段
def _subgoal_attention():
    from scripts.interpret.analysis import subgoal_attention as S
    return S


def demo_task_index(demo_bounds, exec_start=None):
    """demo_bounds（示范第 1..L 段的闭区间帧范围）→ 逐帧段号表（长 exec_start，未覆盖帧为 0）。"""
    e0 = int(exec_start) if exec_start is not None else (int(demo_bounds[-1][1]) + 1 if demo_bounds else 0)
    dti = np.zeros(e0, dtype=np.int64)
    for j, (a, b) in enumerate(demo_bounds, start=1):
        dti[int(a):min(int(b) + 1, e0)] = j
    return dti


def mem_token_segments(mem_type, mem_t0, mem_t1, demo_bounds, exec_start=None, majority=MOTION_MAJORITY):
    """每个记忆 token 所属示范段 j′（1 起）；0 = 过渡/非示范，−1 = 补齐。形状同 mem_type。"""
    S = _subgoal_attention()
    mt = np.asarray(mem_type)
    t0 = np.broadcast_to(np.asarray(mem_t0), mt.shape)
    t1 = np.broadcast_to(np.asarray(mem_t1), mt.shape)
    dti = demo_task_index(demo_bounds, exec_start)
    e0 = len(dti)
    out = np.full(mt.shape, -1, dtype=np.int64)
    cache = {}
    for idx in np.ndindex(mt.shape):
        kind = int(mt[idx])
        if kind < 0:
            continue
        a, b = int(t0[idx]), int(t1[idx])
        key = (kind, a, b)
        if key not in cache:
            if kind == 0:
                cache[key] = 0 if a >= e0 else int(S.segment_of_frame(a, dti, e0))
            elif kind == 1:
                w = b - a + 1
                if w <= 0:
                    raise ValueError(f"motion 窗口首末帧非法：{a}..{b}")
                counts = S.segments_of_window(a, dti, e0, window=w) if a >= 0 else {}
                counts = {s: c for s, c in counts.items() if s > 0}
                need = math.ceil(majority * w - 1e-9)
                best = max(counts.items(), key=lambda kv: kv[1]) if counts else (0, 0)
                cache[key] = int(best[0]) if best[1] >= need else 0
            else:
                raise ValueError(f"未知 mem_type {kind}")
        out[idx] = cache[key]
    return out


# ---------------------------------------------------------------- CLI
def cmd_build_pool(args):
    rows, specs = build_pool(args.seed_start, args.per_difficulty, args.episode_start)
    sdir = write_specs(rows, specs, args.out, args.seed_start)
    for d in DIFFICULTIES:
        ls = [r["L"] for r in rows if r["difficulty"] == d]
        dist = {L: ls.count(L) for L in sorted(set(ls))}
        print(f"{d}: episodes={len(ls)} L分布={dist} 局号 {min(r['episode'] for r in rows if r['difficulty'] == d)}"
              f"..{max(r['episode'] for r in rows if r['difficulty'] == d)}")
    print(f"CHECK_SPEC=PASS routes={len(rows)} bad=0")
    ok, lines = spec_load_check(sdir, args.bench_root)
    for line in lines:
        print(line)
    print(f"SPEC_LOAD={'PASS' if ok else 'FAIL'} groups={len(DIFFICULTIES)} episodes={len(rows)} "
          f"manifest={sdir / 'manifest.json'}")
    return 0 if ok else 1


def cmd_points(args):
    res = run_points(args.pool, args.data_root, args.stride, args.pix_tol)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    by_d = {}
    for p in res["points"]:
        by_d[p["difficulty"]] = by_d.get(p["difficulty"], 0) + 1
    print(f"points={len(res['points'])} 按难度={by_d} → {args.out}")
    for row in res["dropped"][:20]:
        print(f"  剔除 k={row['k']} {row['difficulty']}/ep{row['episode']}: {row['reason']}")
    print(res["verdicts"]["GEN_SPEC"])
    print(res["verdicts"]["PAIRS"])
    return 0 if res["verdicts"]["GEN_SPEC"].startswith("GEN_SPEC=PASS") else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-pool", help="复算 200 局路线，写 4 份单组 spec、清单与 pool.json")
    b.add_argument("--out", required=True)
    b.add_argument("--seed-start", type=int, default=SEED_START)
    b.add_argument("--per-difficulty", type=int, default=PER_DIFFICULTY)
    b.add_argument("--episode-start", type=int, default=EPISODE_START)
    b.add_argument("--bench-root", default=str(DEFAULT_BENCH_ROOT))
    p = sub.add_parser("points", help="从生成的 H5 取点、打标签、算配对")
    p.add_argument("--pool", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--pix-tol", type=int, default=PIX_TOL)
    args = ap.parse_args(argv)
    return cmd_build_pool(args) if args.cmd == "build-pool" else cmd_points(args)


if __name__ == "__main__":
    sys.exit(main())
