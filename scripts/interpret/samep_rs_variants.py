#!/usr/bin/env python3
"""RouteStick 同一局变体：从 (seed, 难度) 重建 episode_spec，生成第 j 步的 S_swing / S_side / S_both 变体并录示范
（计划 1115-4task-routestick-videorepick-steering-plan.html 第二部分 §1.1、§2.1、§4.2、§6）。

只读仿真、不跑策略：每个变体 reset 一次，读示范帧（front + wrist 逐帧 md5）、示范每帧子目标序号、末端轨迹，
与原样重建局比对。重建顺序与 docs/evaluation-doc/steer-4task-1115/records/spec-probe/rs_spec_probe.py 相同
（theta → 4 柱色 → 段数 → 起点 → 每步邻居 → 每步绕向），该脚本实测 3 局与自然局逐帧一致。

用法（official-sim 解释器，-I 隔离，benchmark 源码显式给出）：
  <sim-python> -I scripts/interpret/samep_rs_variants.py build --bench-src <b4e97f2 的 src> --repo <仓库根>
      (--pool <O>/pool.json | --episodes 3100003:hard,3100010:medium) --out <目录> [--workers 4] [--groups S_swing,S_side,S_both]
  输出 <目录>/variants.jsonl：每局一行 kind=base（自然局与原样重建比对），每 (局, j, 组) 一行 kind=variant；
  末尾三行判定（口径见计划 §6）：
    RS_SPEC_REBUILD=PASS|FAIL eps=<n> frame_mismatch=<k> errors=<e>
    RS_SEGLEN=PASS|FAIL eps=<n> variants=<m> bad=<k>
    RS_VARIANT=PASS|FAIL variants=<n> early_diff=<a> swing_mismatch=<k> skipped_endpoint=<s> errors=<e>

帧与段（实测 3100010 medium：示范 201 帧，demo_task_index 在帧 0/50/100/150 变为 1/2/3/4，帧 200 起为执行段）：
帧 k = 第 k 步之后的观测；第 k 步在帧 k−1 的子目标下执行。因此第 j 段的「内容帧」= 标签段 [a_j, b_j] 整体后移 1，
即 [50(j−1)+1, 50j]，与 probe 的首个不同帧（j=2 → 51、j=1 → 1）一致；本脚本记录的 segments 均为内容帧闭区间。

绕向两种口径（与 RouteStick.direction_fail / probe 同式：平均叉积 > 0 记 clockwise）：
  标签口径 swing_label：相对「本变体 prev → 本段目标」有向线；
  物理弧侧 swing_phys：相对「原局 prev → 原局本段目标」有向线（S_swing 两口径相同；S_side 第 j 段两口径相反）。
S_side 规则：第 j 步改去另一侧邻居，之后每步保持原相对方向（槽位差），出界则反弹；第 j 步起点在端点（槽 0 或 4）
不生成 S_side / S_both。生成时用 sampling_config.parameters.configs[难度].backtrack=True 放开回头（只影响校验，
示范规划与物理不受影响）；环境不校验槽位下界（-1 会被当成 8），这里自查 0 ≤ slot ≤ 4。
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys

TASK = "RouteStick"
CFG = {"easy": ((2, 3), False), "medium": ((4, 5), False), "hard": ((4, 7), True), "xhard": ((8, 10), True)}
ROUTE = (0, 2, 4, 6, 8)
POSTS = (1, 3, 5, 7)
SEGLEN = 50
GROUPS = ("S_swing", "S_side", "S_both")
CW, CCW = "clockwise", "counterclockwise"
MAX_STEPS_WITHOUT_DEMO = 3002          # 与 probe 相同；只影响执行段步数上限，reset 内示范不受影响


# ---------------------------------------------------------------- 纯函数：spec 重建与变体
def rebuild_spec(seed, difficulty, episode=0):
    """按 RouteStick._load_scene 自然局的随机数顺序复算 episode_spec（同 rs_spec_probe.replay）。"""
    import torch
    (lo, hi), back = CFG[difficulty]
    g = torch.Generator()
    g.manual_seed(int(seed))
    theta = (torch.rand(1, generator=g).item() * 60) - 30
    rgbs = [torch.rand(3, generator=g).tolist() for _ in range(4)]
    steps = int(torch.randint(lo, hi + 1, (1,), generator=g).item())
    idx = [int(torch.randint(0, 5, (1,), generator=g).item())]
    for _ in range(steps):
        cur = idx[-1]
        prev = idx[-2] if len(idx) > 1 else None
        nb = [n for n in (cur - 1, cur + 1) if 0 <= n < 5]
        c = nb if (back or prev is None) else ([n for n in nb if n != prev] or nb)
        idx.append(int(c[torch.randint(0, len(c), (1,), generator=g).item()]))
    dirs = [CW if torch.rand(1, generator=g).item() < 0.5 else CCW for _ in range(steps)]
    return {"task": TASK, "episode": int(episode), "difficulty": difficulty,
            "layout": {"rotation_deg": float(theta), "obstacle_rgb": [[float(v) for v in c] for c in rgbs]},
            "objects": {"L": steps},
            "actions": {"nodes": [ROUTE[i] for i in idx], "node_slots": idx, "directions": dirs}}


def flip_dir(d):
    if d not in (CW, CCW):
        raise ValueError(f"非法绕向 {d!r}")
    return CCW if d == CW else CW


def side_slots(slots, j):
    """第 j 步（1 起）改去另一侧邻居，之后每步保持原槽位差、出界反弹；起点在端点返回 None。"""
    s = [int(v) for v in slots]
    j = int(j)
    if not 1 <= j <= len(s) - 1:
        raise ValueError(f"j={j} 超出段数 {len(s) - 1}")
    alt = s[j - 1] - (s[j] - s[j - 1])
    if not 0 <= alt < len(ROUTE):
        return None
    new = s[:j] + [alt]
    for i in range(j, len(s) - 1):
        step = s[i + 1] - s[i]
        nxt = new[-1] + step
        if not 0 <= nxt < len(ROUTE):
            nxt = new[-1] - step
        new.append(nxt)
    return new


def check_spec(spec, difficulty, allow_backtrack=None):
    """复现环境侧校验（RouteStick._load_scene spec 分支）并自查 0 ≤ slot ≤ 4；返回问题列表（空 = 合法）。"""
    (lo, hi), back = CFG[difficulty]
    back = back if allow_backtrack is None else bool(allow_backtrack)
    a = spec["actions"]
    L = int(spec["objects"]["L"])
    nodes, slots, dirs = list(a["nodes"]), [int(v) for v in a["node_slots"]], list(a["directions"])
    out = []
    if not lo <= L <= hi:
        out.append(f"L={L} 不在 [{lo},{hi}]")
    if len(nodes) != L + 1 or len(slots) != L + 1:
        out.append("节点数 ≠ L+1")
    if any(not 0 <= v < len(ROUTE) for v in slots):
        out.append(f"槽位越界 {slots}")
    elif [ROUTE[v] for v in slots] != [int(v) for v in nodes]:
        out.append("nodes 与 node_slots 不一致")
    for i in range(min(L, len(slots) - 1)):
        if abs(slots[i + 1] - slots[i]) != 1:
            out.append(f"第 {i} 段不相邻")
        if not back and i > 0 and slots[i + 1] == slots[i - 1] and 0 < slots[i] < len(ROUTE) - 1:
            out.append(f"第 {i} 段非端点回头（backtrack=False）")
    if len(dirs) != L or any(d not in (CW, CCW) for d in dirs):
        out.append("directions 长度或取值非法")
    return out


def make_variant(spec, j, group):
    """返回 (变体 spec, 是否需放开 backtrack) 或 None（S_side/S_both 起点在端点）。"""
    j = int(j)
    v = copy.deepcopy(spec)
    a = v["actions"]
    if group not in GROUPS:
        raise ValueError(f"未知组 {group}")
    if group in ("S_side", "S_both"):
        new = side_slots(a["node_slots"], j)
        if new is None:
            return None
        a["node_slots"] = new
        a["nodes"] = [ROUTE[i] for i in new]
    if group in ("S_swing", "S_both"):
        a["directions"][j - 1] = flip_dir(a["directions"][j - 1])
    return v, group != "S_swing"


def content_segments(L, seglen=SEGLEN):
    """理论内容段：第 j 段 = [seglen(j−1)+1, seglen·j]（闭区间）。"""
    return [(seglen * (j - 1) + 1, seglen * j) for j in range(1, int(L) + 1)]


def segments_from_labels(labels, m):
    """demo_task_index（逐帧，帧 0..es）→ 内容段 [(a_j+1, b_j+1)]，j=1..m；只看帧 0..es−1 的标签，须为 1..m 顺序
    连续块（与 sp_swap_prep.segments_from_labels 同判据），返回 (段或 None, 问题)。"""
    lab = [int(x) for x in labels]
    es = len(lab) - 1
    if es <= 0:
        return None, "empty"
    blocks, cur, a = [], lab[0], 0
    for k in range(1, es + 1):
        if k == es or lab[k] != cur:
            blocks.append((cur, a, k - 1))
            if k < es:
                cur, a = lab[k], k
    order = [b[0] for b in blocks]
    if order != list(range(1, int(m) + 1)):
        return None, f"order={order}"
    return [(a_ + 1, b_ + 1) for _, a_, b_ in blocks], None


def seglen_ok(n_frames, segs, L, seglen=SEGLEN):
    """RS_SEGLEN 单局：帧数 = 1 + seglen·L，且每段恰 seglen 帧。"""
    if int(n_frames) != 1 + seglen * int(L) or segs is None or len(segs) != int(L):
        return False
    return all(b - a + 1 == seglen for a, b in segs)


def swing_of(points, a_xy, b_xy):
    """平均叉积（direction_fail 同式）：> 0 clockwise，< 0 counterclockwise，否则 None。"""
    lx, ly = b_xy[0] - a_xy[0], b_xy[1] - a_xy[1]
    if math.hypot(lx, ly) < 1e-9 or not len(points):
        return None
    cr = sum(lx * (float(p[1]) - a_xy[1]) - ly * (float(p[0]) - a_xy[0]) for p in points) / len(points)
    return CW if cr > 0 else CCW if cr < 0 else None


def post_of(a_cell, b_cell):
    """被绕立柱：两相邻路线格的中点格；返回 (立柱序号 0..3, 格号)。"""
    mid = (int(a_cell) + int(b_cell)) // 2
    if abs(int(a_cell) - int(b_cell)) != 2 or mid not in POSTS:
        raise ValueError(f"{a_cell}→{b_cell} 不是相邻路线格")
    return POSTS.index(mid), mid


def first_diff(a, b):
    """两组逐帧摘要的首个不同下标；完全相同返回 None；长度不同且公共部分相同返回较短长度。"""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def demo_swings(eef_xy, segs, cells, ref_cells, cell_xy):
    """逐段示范绕向：标签口径（相对 cells[k−1]→cells[k]）与物理弧侧口径（相对 ref_cells[k−1]→ref_cells[k]）。"""
    lab, phys = [], []
    for k, (a, b) in enumerate(segs, start=1):
        pts = eef_xy[a:b + 1]
        lab.append(swing_of(pts, cell_xy[cells[k - 1]], cell_xy[cells[k]]))
        phys.append(swing_of(pts, cell_xy[ref_cells[k - 1]], cell_xy[ref_cells[k]])
                    if k < len(ref_cells) else None)
    return lab, phys


# ---------------------------------------------------------------- 仿真
def native_sampling(difficulty=None, backtrack=False):
    """b4e97f2 的 NATIVE_SAMPLING 副本；backtrack=True 时在 configs 里放开该难度回头（_resolve_sampling_config 用 setdefault 接受）。"""
    from robomme.robomme_env.RouteStick import NATIVE_SAMPLING, RouteStick
    samp = {"parameters": copy.deepcopy(NATIVE_SAMPLING["parameters"]),
            "positions": copy.deepcopy(NATIVE_SAMPLING["positions"])}
    if backtrack:
        cfgs = copy.deepcopy(RouteStick.configs)
        cfgs[difficulty] = dict(cfgs[difficulty])
        cfgs[difficulty]["backtrack"] = True
        samp["parameters"]["configs"] = cfgs
    return samp


def reset_demo(seed, difficulty, spec=None, sampling=None):
    """建环境并 reset，返回示范摘要。spec=None 为自然局（gym.make 显式传难度，同 probe）。"""
    import gymnasium as gym
    import numpy as np
    import robomme.robomme_env  # noqa: F401  注册环境
    from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
    from robomme.env_record_wrapper.episode_config_resolver import _wrap_env
    from robomme.injection_candidates import RUNTIME
    kw = dict(RUNTIME["kwargs"])
    kw.update(seed=int(seed), difficulty=difficulty)
    if spec is not None:
        kw.update(episode_spec=spec, sampling_config=sampling)
    env = _wrap_env(gym.make(TASK, **kw), env_id=TASK, action_space="joint_angle", gui_render=False,
                    max_steps_without_demo=MAX_STEPS_WITHOUT_DEMO)
    try:
        w = env
        while not isinstance(w, DemonstrationWrapper):
            w = w.env
        original = w._augment_obs_and_info
        u = env.unwrapped

        def augment(obs, info, action):
            obs, info = original(obs, info, action)
            info["samep_task_index"] = int(u.current_task_index)
            return obs, info
        w._augment_obs_and_info = augment
        try:
            obs, _ = env.reset()
        finally:
            del w._augment_obs_and_info
        labels = [int(v) for v in env.demonstration_data[4]["samep_task_index"]]
        front, wrist = obs["front_rgb_list"], obs["wrist_rgb_list"]
        hashes = [hashlib.md5(np.ascontiguousarray(f).tobytes() + np.ascontiguousarray(g).tobytes()).hexdigest()
                  for f, g in zip(front, wrist)]
        eef = np.asarray(obs["eef_state_list"], dtype=np.float64)[:, :2]
        cell_xy = {i: tuple(float(x) for x in np.asarray(b.pose.p).reshape(-1)[:2]) for i, b in enumerate(u.buttons_grid)}
        ids = {id(b): i for i, b in enumerate(u.buttons_grid)}
        return {"hashes": hashes, "labels": labels, "eef": eef, "cell_xy": cell_xy,
                "cells": [ids[id(b)] for b in u.selected_buttons], "dirs": list(u.swing_directions),
                "difficulty": str(u.difficulty), "n_frames": len(hashes)}
    finally:
        env.close()


def process_episode(item, groups=GROUPS):
    """一局：自然局、原样重建、每 (j, 组) 变体；返回 jsonl 行列表（第一行 kind=base）。"""
    seed, diff, ep = int(item["seed"]), item["difficulty"], item.get("episode_id")
    base = {"kind": "base", "seed": seed, "difficulty": diff, "episode_id": ep, "status": "ok"}
    rows = [base]
    try:
        spec = rebuild_spec(seed, diff, episode=ep or 0)
        L = int(spec["objects"]["L"])
        base.update(spec=spec, L=L, sampling_config=native_sampling())
        nat = reset_demo(seed, diff)
        reb = reset_demo(seed, diff, spec, base["sampling_config"])
        segs, why = segments_from_labels(reb["labels"], L)
        base.update(n_frames=reb["n_frames"], natural_n_frames=nat["n_frames"], segments=segs, segment_problem=why,
                    frame_first_diff=first_diff(nat["hashes"], reb["hashes"]),
                    route_match=(nat["cells"] == reb["cells"] == [int(c) for c in spec["actions"]["nodes"]]),
                    dirs_match=(nat["dirs"] == reb["dirs"] == spec["actions"]["directions"]),
                    actual_difficulty=reb["difficulty"])
        base["frame_mismatch"] = (base["frame_first_diff"] is not None or not base["route_match"]
                                  or not base["dirs_match"] or reb["difficulty"] != diff)
        base["seglen_ok"] = seglen_ok(reb["n_frames"], segs, L)
        if segs is not None:
            lab, _ = demo_swings(reb["eef"], segs, reb["cells"], reb["cells"], reb["cell_xy"])
            base["demo_swings_label"] = lab
        base["usable"] = not base["frame_mismatch"] and base["seglen_ok"]
    except Exception as error:   # 记原因继续，判定行计 errors
        base.update(status="error", reason=f"{type(error).__name__}: {error}", usable=False)
        return rows
    orig_cells = [int(c) for c in spec["actions"]["nodes"]]
    for j in range(1, L + 1):
        for group in groups:
            row = {"kind": "variant", "seed": seed, "difficulty": diff, "episode_id": ep, "j": j, "group": group,
                   "L": L, "base_usable": base["usable"]}
            rows.append(row)
            made = make_variant(spec, j, group)
            if made is None:
                row.update(status="skipped", reason="endpoint", usable=False)
                continue
            vspec, backtrack = made
            problems = check_spec(vspec, diff, allow_backtrack=True if backtrack else None)
            sampling = native_sampling(diff, backtrack)
            row.update(spec=vspec, sampling_config=sampling, backtrack_override=backtrack)
            if problems:
                row.update(status="invalid", reason="; ".join(problems), usable=False)
                continue
            try:
                var = reset_demo(seed, diff, vspec, sampling)
            except Exception as error:
                row.update(status="error", reason=f"{type(error).__name__}: {error}", usable=False)
                continue
            vsegs, why = segments_from_labels(var["labels"], L)
            fd = first_diff(reb["hashes"], var["hashes"])
            expect = SEGLEN * (j - 1) + 1
            vcells = [int(c) for c in vspec["actions"]["nodes"]]
            row.update(status="ok", n_frames=var["n_frames"], segments=vsegs, segment_problem=why,
                       seglen_ok=seglen_ok(var["n_frames"], vsegs, L), first_diff=fd, expected_first_diff=expect,
                       early_diff=(fd is not None and fd < expect), no_diff=fd is None,
                       route_match=var["cells"] == vcells, dirs_match=var["dirs"] == vspec["actions"]["directions"],
                       spec_dirs=list(vspec["actions"]["directions"]))
            posts = []
            for k in range(1, L + 1):
                p, cell = post_of(vcells[k - 1], vcells[k])
                posts.append({"post": p, "cell": cell, "rgb": vspec["layout"]["obstacle_rgb"][p]})
            row["posts"] = posts
            orig_post = post_of(orig_cells[j - 1], orig_cells[j])[0]
            row["post_changed_j"] = posts[j - 1]["post"] != orig_post
            if vsegs is not None:
                lab, phys = demo_swings(var["eef"], vsegs, vcells, orig_cells, var["cell_xy"])
                row.update(demo_swings_label=lab, demo_swings_phys=phys)
                row["swing_mismatch"] = lab[j - 1] != vspec["actions"]["directions"][j - 1]
            else:
                row["swing_mismatch"] = True
            row["usable"] = bool(base["usable"] and row["seglen_ok"] and not row["early_diff"] and not row["no_diff"]
                                 and not row["swing_mismatch"] and row["route_match"] and row["dirs_match"])
    return rows


def _process(args):
    item, groups, bench_src, repo = args
    _setup_path(bench_src, repo)
    return process_episode(item, groups)


def _setup_path(bench_src, repo):
    for p in (repo, str(Path(repo) / "src") if repo else None, bench_src):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def verdicts(rows):
    """三条判定行（计划 §6）。"""
    bases = [r for r in rows if r["kind"] == "base"]
    vs = [r for r in rows if r["kind"] == "variant"]
    b_err = sum(1 for r in bases if r["status"] == "error")
    mism = sum(1 for r in bases if r.get("frame_mismatch"))
    ok_b = [r for r in bases if r["status"] == "ok"]
    ran = [r for r in vs if r["status"] == "ok"]
    seg_bad = sum(1 for r in ok_b if not r.get("seglen_ok")) + sum(1 for r in ran if not r.get("seglen_ok"))
    early = sum(1 for r in ran if r.get("early_diff"))
    swing = sum(1 for r in ran if r.get("swing_mismatch"))
    skipped = sum(1 for r in vs if r["status"] == "skipped")
    v_err = sum(1 for r in vs if r["status"] in ("error", "invalid"))
    return [
        f"RS_SPEC_REBUILD={'PASS' if mism == 0 and b_err == 0 and bases else 'FAIL'} eps={len(bases)} "
        f"frame_mismatch={mism} errors={b_err}",
        f"RS_SEGLEN={'PASS' if seg_bad == 0 and ok_b else 'FAIL'} eps={len(ok_b)} variants={len(ran)} bad={seg_bad}",
        f"RS_VARIANT={'PASS' if early == 0 and v_err == 0 and ran else 'FAIL'} variants={len(ran)} early_diff={early} "
        f"swing_mismatch={swing} skipped_endpoint={skipped} errors={v_err} "
        f"usable={sum(1 for r in vs if r.get('usable'))}",
    ]


def load_items(args):
    if args.pool:
        pool = json.loads(Path(args.pool).read_text())
        return [{"seed": int(e["env_seed"]), "difficulty": e["difficulty"], "episode_id": int(e["episode_id"])}
                for e in pool["episodes"] if e.get("task_name", e.get("task")) == TASK]
    out = []
    for i, part in enumerate(x for x in args.episodes.split(",") if x.strip()):
        seed, diff = part.split(":")
        if diff not in CFG:
            raise SystemExit(f"未知难度 {diff}")
        out.append({"seed": int(seed), "difficulty": diff, "episode_id": i})
    return out


def cmd_build(args):
    _setup_path(args.bench_src, args.repo)
    items = load_items(args)
    groups = tuple(g for g in args.groups.split(",") if g)
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        raise SystemExit(f"未知组 {bad}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "variants.jsonl"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} 已存在；确需重写加 --overwrite")
    rows = []
    workers = max(1, int(args.workers))
    jobs = [(it, groups, args.bench_src, args.repo) for it in items]
    if workers == 1:
        parts = map(_process, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        parts = pool.map(_process, jobs)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for i, part in enumerate(parts):
            for row in part:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.extend(part)
            b = part[0]
            print(f"SAMEP_EP done={i + 1}/{len(items)} seed={b['seed']} difficulty={b['difficulty']} "
                  f"status={b['status']} usable={b.get('usable')} variants={len(part) - 1}", flush=True)
    tmp.replace(path)
    if workers > 1:
        pool.shutdown()
    lines = verdicts(rows)
    (out / "verdicts.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line, flush=True)
    return 0 if all("=PASS" in x for x in lines) else 1


def read_variants(path):
    """读 variants.jsonl → (bases {(seed, 难度): 行}, variants [行])。"""
    bases, variants = {}, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["kind"] == "base":
            bases[(int(row["seed"]), row["difficulty"])] = row
        else:
            variants.append(row)
    return bases, variants


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--bench-src", required=True, help="robomme_benchmark@b4e97f2 的 src 目录（插到 sys.path 最前）")
    p.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pool", help="sp_swap_prep pool 输出的 pool.json（只取 RouteStick 局）")
    g.add_argument("--episodes", help="seed:难度,…（局号按顺序 0..）")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--groups", default=",".join(GROUPS))
    p.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return {"build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
