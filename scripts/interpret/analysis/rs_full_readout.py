#!/usr/bin/env python3
"""RouteStick 第一个子目标 motion 交换全因子读数（计划 1109-routestick-swap-full-factorial-plan.html 第一部分三节、
第二部分一节 rs_full_readout 行；修订一：Wan 门槛的 PARTIAL 写法、另报环境官方第一子目标成败）。

两种入口（只用 CPU）：
  --root <O>/<M>                       单模型读数：读 <O>/<M>/plan.json（rs_full_prep specs + donors 写出）列出的每局
  --root <O> --intersect motion,wan    两模型都原样整局成功的接收局上的并排表
输出写 <root>/readout/（readout.json、readout.md；交集 intersect.json、intersect.md），可用 --out 改。

每局只读第一步（first_step_segment）：
  轨迹 = attempts/*/client/traces/<nonce>/step_trace.jsonl 的 step 行（只含执行段），取 tcp_after；从第一个 step 起，
  到第一次水平距离 < 0.03 m 的「非起点目标格」（route 目标格 0、2、4、6、8 中除起点外的格；同步多格取最近）为止。
  目标 / 立柱坐标按 rs_full_prep.cell_xy 用接收局的转角 theta 复算。
  到达归类 classify_target：own（自己第一步目标）/ donor_dir（从自己起点按供体第一步左右推出的格）/ donor_abs（供体第一步
  的绝对目标格）/ other / none，重合时 own > donor_dir > donor_abs。
  绕向 swing_side：照 RouteStick.direction_fail，line_vec = 到达格 − 起点格，对段内每点 rel = xy − 起点格，
  cross = line_vec[0]·rel[1] − line_vec[1]·rel[0]，均值 > 0 为 clockwise、< 0 为 counterclockwise、= 0 记 none；
  没有到达任何非起点目标时绕向记 none（没有可对照的连线）。
  绕向归类：own_swing（与自己第一步绕向相同；与供体相同时也记 own）/ donor_swing / other_swing（与两者都不同，只会出现在
  供体绕向与自己相同的组）/ none。
  第一步成功 = 到达 own 且 own_swing。另报整局成功（终局 outcome），以及环境官方第一子目标成败 official_step1：
  步轨迹的 task_index_after 是否越过执行段第一个子目标的序号（第一个 step 行的 task_index_before，缺失时用回执
  demo_task_index 末帧）；取不到记 None 并在读数里单列计数。

判定行：
  RS_DEMO_LAYOUT=PASS|PARTIAL|FAIL model=<m> eps=<n> bad=<k>   N 组：budget.demo_frames==100、demo_windows==3、
                                                              回执 D==100、m==2、demo_task_index 第 0–49 帧同一子目标、第 50–99 帧另一子目标
  SWAP_SCOPE=PASS|PARTIAL|FAIL model=<m> runs=<n> count_mismatch=<k> scope_bad=<k>
        replace_log 每条 [s, s] 且起点集合恰为 {0,32}（W0 组 {0}）；n_replaced == 接收局 N 录制中这些起点的 counts 之和
  MULTI_SWAP_EQ=PASS|PARTIAL|FAIL runs=<n> differ=<k>      一局一服务端 vs 连跑：到达格、实际绕向、整局成败逐局相同
  RS_SWAP_FULL=PASS|PARTIAL|FAIL model=<m> runs=<n> invalid=<k> N=<a>/48 Z=… Rnd=… A=… Bsw_swing=… Bside_dir=… C_own=…
        D1_swing=… D2_dir=… D2_abs=… W0_swing=… W0_dir=…
    N 为整局成功；Z、Rnd、A 为第一步成功；Bsw_swing、D1_swing、W0_swing（W0 中 Bsw 来源）为 donor_swing；Bside_dir、
    W0_dir（W0 中 Bside 来源）、D2_dir 为到达 donor_dir；C_own 为到达 own；D2_abs 为到达 donor_abs。分母均为该组有效局。
    Wan 门槛触发：RS_SWAP_FULL=PARTIAL model=wan reason=few_success r_success=<k> groups=N,Z,Rnd …
  RS_SWAP_INTERSECT models=<a,b> recv=<k> <m>:Z=…,Rnd=…,…     两模型都原样整局成功的接收局上的同口径读数
另有信息行（不是验收项）：SEED_FIRST_STEP、OFFICIAL_STEP1。成功率与各类占比只是读数，不设及格线。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from scripts.interpret.analysis.pl_swap_readout import attempts_for, bootstrap_ci, read_json
from scripts.interpret.rs_full_prep import (N_EPS, N_RECV, PAIR_GROUPS, ROUTE, STARTS_TRIM, STARTS_W0, SWING_SHORT, TASK,
                                            cell_xy, record_name, target_cell)

REACH_THRESH = 0.03                 # is_obj_swing_onto 的 distance_threshold
REACH_CATS = ("own", "donor_dir", "donor_abs", "other", "none")
SWING_CATS = ("own_swing", "donor_swing", "other_swing", "none")
GROUPS = ("N", "Z", "Rnd") + PAIR_GROUPS + ("W0",)
SWAP_GROUPS = ("Z", "Rnd") + PAIR_GROUPS
MAIN_STAGES = ("n", "swap", "w0")
SCOPE_STAGES = ("smoke", "multieq", "swap", "w0", "seeds")
MAIN_SEED = 42
DEMO_FRAMES, DEMO_WINDOWS, STEP_FRAMES = 100, 3, 50


# ---------------------------------------------------------------- 几何与归类
def first_step_segment(tcps, start_cell, theta, thresh=REACH_THRESH):
    """返回 (到达格或 None, 段内点列表)。段 = 从第一个点到第一次到达（含）；没到达时为全部点。"""
    centers = {c: cell_xy(c, theta) for c in ROUTE if c != int(start_cell)}
    for i, tcp in enumerate(tcps):
        x, y = float(tcp[0]), float(tcp[1])
        hits = [(math.hypot(x - cx, y - cy), c) for c, (cx, cy) in centers.items()]
        hits = [h for h in hits if h[0] < thresh]
        if hits:
            return min(hits)[1], tcps[:i + 1]
    return None, list(tcps)


def swing_side(points, prev_xy, curr_xy):
    """RouteStick.direction_fail 的平均叉积：> 0 clockwise，< 0 counterclockwise，= 0 / 无点 / 两点重合 None。"""
    lx, ly = curr_xy[0] - prev_xy[0], curr_xy[1] - prev_xy[1]
    if math.hypot(lx, ly) < 1e-6 or not points:
        return None
    cross = [lx * (float(p[1]) - prev_xy[1]) - ly * (float(p[0]) - prev_xy[0]) for p in points]
    avg = sum(cross) / len(cross)
    return "clockwise" if avg > 0 else "counterclockwise" if avg < 0 else None


def classify_target(cell, own, donor_dir=None, donor_abs=None):
    """优先级 own > donor_dir > donor_abs > other；None 记 none。"""
    if cell is None:
        return "none"
    if cell == own:
        return "own"
    if donor_dir is not None and cell == donor_dir:
        return "donor_dir"
    if donor_abs is not None and cell == donor_abs:
        return "donor_abs"
    return "other"


def classify_swing(actual, own_swing, donor_swing=None):
    """actual / own_swing / donor_swing 均为 'cw'|'ccw'（或 None）。"""
    if actual is None:
        return "none"
    if actual == own_swing:
        return "own_swing"
    if donor_swing is not None and actual == donor_swing:
        return "donor_swing"
    return "other_swing"


def targets(row):
    """(own, donor_dir, donor_abs, own_swing, donor_swing)；无供体时后三者 None（donor_dir 出界也 None）。"""
    rs, rside, rsw = row["recv_type"]
    own = target_cell(rs, rside)
    dt = row.get("donor_type")
    if not dt:
        return own, None, None, rsw, None
    return own, target_cell(rs, dt[1]), target_cell(dt[0], dt[1]), rsw, dt[2]


# ---------------------------------------------------------------- 读一局
def read_steps(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("event") == "step" and r.get("tcp_after") is not None:
                    rows.append(r)
    return rows


def official_step1(steps, receipt):
    """环境官方第一子目标是否完成：task_index_after 越过执行段第一个子目标序号。取不到返回 (None, 原因)。"""
    first = steps[0].get("task_index_before") if steps else None
    if first is None and receipt and receipt.get("demo_task_index"):
        first = receipt["demo_task_index"][-1]
    after = [s.get("task_index_after") for s in steps if s.get("task_index_after") is not None]
    if first is None or not after:
        return None, "no_task_index"
    return int(max(after) > int(first)), None


def layout_bad(terminal, receipt):
    """RS_DEMO_LAYOUT 的逐局检查，返回问题列表（空 = 合格）。"""
    bad = []
    budget = terminal.get("budget") or {}
    if budget.get("demo_frames") != DEMO_FRAMES:
        bad.append(f"demo_frames={budget.get('demo_frames')}")
    if budget.get("demo_windows") != DEMO_WINDOWS:
        bad.append(f"demo_windows={budget.get('demo_windows')}")
    if receipt is None:
        return bad + ["no_receipt"]
    if receipt.get("D") != DEMO_FRAMES or receipt.get("m") != 2:
        bad.append(f"D={receipt.get('D')},m={receipt.get('m')}")
    t = list(receipt.get("demo_task_index") or [])
    if len(t) < DEMO_FRAMES:
        return bad + [f"demo_task_index_len={len(t)}"]
    first, second = set(t[:STEP_FRAMES]), set(t[STEP_FRAMES:DEMO_FRAMES])
    if len(first) != 1 or len(second) != 1 or first == second:
        bad.append(f"step_split={sorted(first)}|{sorted(second)}")
    return bad


def load_run(row):
    """一局读数；status ∈ ok / missing（未跑）/ invalid（跑了但读不全）。"""
    key = f"{TASK}:{int(row['ep'])}"
    run_root, attn_out = Path(row["run_root"]), Path(row["attn_out"])
    out = {k: row.get(k) for k in ("stage", "worker", "group", "seed", "ep", "donor", "spec", "variant", "pair_index",
                                    "record_dir", "recv_type", "donor_type", "stratum", "w0_from")}
    out["attempts"] = attempts_for(run_root, key)
    terminals = sorted(run_root.glob(f"accepted/{key}.*.terminal.json"), key=lambda p: p.stat().st_mtime)
    if not terminals:
        out["status"] = "invalid" if out["attempts"] else "missing"
        out["reason"] = "no_terminal"
        return out
    terminal = read_json(terminals[-1])
    nonce = terminal["episode_nonce"]
    out.update(nonce=nonce, outcome=terminal.get("outcome"), success=int(terminal.get("outcome") == "success"),
               terminals=len(terminals), budget=terminal.get("budget"))
    episode_path = attn_out / nonce / "episode.json"
    out["motion_content"] = read_json(episode_path).get("motion_content") if episode_path.is_file() else None
    traces = sorted(run_root.glob(f"attempts/*/client/traces/{nonce}/step_trace.jsonl"))
    if not traces:
        out["status"], out["reason"] = "invalid", "no_trace"
        return out
    receipt_path = traces[0].parent / "meta" / "hold_receipt.json"
    receipt = read_json(receipt_path) if receipt_path.is_file() else None
    steps = read_steps(traces[0])
    tcps = [[float(v) for v in s["tcp_after"]] for s in steps]
    theta = float(row["theta"])
    start = int(row["path"][0])
    reached, seg = first_step_segment(tcps, start, theta)
    actual = None
    if reached is not None:
        actual = swing_side(seg, cell_xy(start, theta), cell_xy(reached, theta))
    actual_s = SWING_SHORT.get(actual) if actual else None
    own, ddir, dabs, own_sw, donor_sw = targets(row)
    reach_cat = classify_target(reached, own, ddir, dabs)
    swing_cat = classify_swing(actual_s, own_sw, donor_sw)
    off, off_why = official_step1(steps, receipt)
    out.update(trace=str(traces[0]), steps=len(tcps), seg_len=len(seg), reached=reached, swing_actual=actual_s,
               own=own, donor_dir_cell=ddir, donor_abs_cell=dabs, own_swing=own_sw, donor_swing=donor_sw,
               category=reach_cat, swing_category=swing_cat,
               step1_success=int(reach_cat == "own" and swing_cat == "own_swing"),
               official_step1=off, official_step1_reason=off_why, layout_bad=layout_bad(terminal, receipt),
               status="ok")
    if row["group"] != "N" and out["motion_content"] is None:
        out["status"], out["reason"] = "invalid", "no_episode_json"
    return out


# ---------------------------------------------------------------- 判定
def judge_layout(runs, model):
    rows = [r for r in runs if r["stage"] == "n" and r["status"] == "ok"]
    bad = {r["ep"]: r["layout_bad"] for r in rows if r.get("layout_bad")}
    verdict = "FAIL" if bad else ("PASS" if len(rows) == N_EPS else "PARTIAL")
    return f"RS_DEMO_LAYOUT={verdict} model={model} eps={len(rows)} bad={len(bad)}", {"eps": len(rows), "bad": bad}


def record_counts(record_dir, ep, starts):
    """接收局 N 录制中给定起点的 counts 之和；缺文件或缺行返回 None。"""
    path = Path(record_dir) / record_name(ep)
    if not path.is_file():
        return None
    with np.load(path) as z:
        sf = np.asarray(z["start_frames"]).reshape(-1)
        counts = np.asarray(z["counts"]).reshape(-1) if "counts" in z.files else np.ones_like(sf)
    total = 0
    for s in starts:
        hit = np.flatnonzero(sf == int(s))
        if hit.size != 1:
            return None
        total += int(counts[int(hit[0])])
    return total


def judge_scope(runs, model, planned):
    details, mismatch, bad = [], 0, 0
    for r in runs:
        if r["stage"] not in SCOPE_STAGES or r["group"] == "N" or r.get("motion_content") is None:
            continue
        mc = r["motion_content"]
        log = [list(x) for x in mc.get("replace_log", [])]
        want = set(STARTS_W0 if r["group"] == "W0" else STARTS_TRIM)
        scope_ok = (bool(log) and all(len(x) == 2 and x[0] == x[1] for x in log) and {x[0] for x in log} == want
                    and mc.get("mode") == "swap_exact" and mc.get("donor") == f"{TASK}:{int(r['donor'])}")
        expected = record_counts(r["record_dir"], r["ep"], sorted(want))
        count_ok = expected is not None and int(mc.get("n_replaced", -1)) == expected == len(log)
        mismatch += 0 if count_ok else 1
        bad += 0 if scope_ok else 1
        details.append({"stage": r["stage"], "group": r["group"], "seed": r["seed"], "ep": r["ep"], "donor": r["donor"],
                        "starts": sorted({x[0] for x in log}), "n_replaced": mc.get("n_replaced"),
                        "expected_count": expected, "scope_ok": scope_ok, "count_ok": count_ok})
    n = len(details)
    verdict = "FAIL" if (mismatch or bad) else ("PASS" if n == planned and n > 0 else "PARTIAL")
    line = f"SWAP_SCOPE={verdict} model={model} runs={n} count_mismatch={mismatch} scope_bad={bad}"
    return line, {"runs": n, "planned": planned, "count_mismatch": mismatch, "scope_bad": bad, "details": details}


def judge_multieq(runs):
    by_pair = {}
    for r in runs:
        if r["stage"] == "multieq":
            by_pair.setdefault(r.get("pair_index"), {})[r.get("variant")] = r
    differ, complete, rows = [], 0, []
    for idx, v in sorted(by_pair.items(), key=lambda kv: kv[0]):
        a, b = v.get("single"), v.get("multi")
        if not (a and b and a["status"] == "ok" and b["status"] == "ok"):
            continue
        complete += 1
        fa = [a.get("reached"), a.get("swing_actual"), a.get("success")]
        fb = [b.get("reached"), b.get("swing_actual"), b.get("success")]
        rows.append({"pair_index": idx, "ep": a["ep"], "donor": a["donor"], "single": fa, "multi": fb, "same": fa == fb})
        if fa != fb:
            differ.append(idx)
    verdict = "FAIL" if differ else ("PASS" if complete == len(by_pair) and complete > 0 else "PARTIAL")
    return f"MULTI_SWAP_EQ={verdict} runs={complete} differ={len(differ)}", {"planned": len(by_pair), "rows": rows}


def seed_first_step(runs):
    base = {(r["group"], r["ep"], r["donor"]): r for r in runs
            if r["stage"] == "swap" and r["status"] == "ok" and r["group"] in PAIR_GROUPS}
    pairs = {}
    for r in runs:
        if r["stage"] == "seeds" and r["status"] == "ok":
            pairs.setdefault((r["group"], r["ep"], r["donor"]), {})[r["seed"]] = (r.get("reached"), r.get("swing_actual"))
    rows, agree = [], 0
    for key, seeds in sorted(pairs.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if key not in base:
            continue
        seeds = {MAIN_SEED: (base[key].get("reached"), base[key].get("swing_actual")), **seeds}
        same = len(set(seeds.values())) == 1
        agree += int(same)
        rows.append({"group": key[0], "ep": key[1], "donor": key[2],
                     "first_step": {str(s): list(v) for s, v in seeds.items()}, "agree": same})
    return f"SEED_FIRST_STEP pairs={len(rows)} agree={agree}", {"rows": rows}


def official_info(rows):
    known = [r for r in rows if r.get("official_step1") is not None]
    agree = sum(1 for r in known if r["official_step1"] == r["step1_success"])
    return (f"OFFICIAL_STEP1 runs={len(rows)} known={len(known)} official_ok={sum(r['official_step1'] for r in known)} "
            f"readout_ok={sum(r['step1_success'] for r in known)} agree={agree}")


def group_stats(rows):
    return {"n": len(rows), "success": sum(r["success"] for r in rows),
            "step1_success": sum(r.get("step1_success", 0) for r in rows),
            "official_step1": sum(1 for r in rows if r.get("official_step1") == 1),
            "official_known": sum(1 for r in rows if r.get("official_step1") is not None),
            "reach": {c: sum(1 for r in rows if r.get("category") == c) for c in REACH_CATS},
            "swing": {c: sum(1 for r in rows if r.get("swing_category") == c) for c in SWING_CATS},
            "success_ci95": bootstrap_ci([r["success"] for r in rows]),
            "step1_ci95": bootstrap_ci([r.get("step1_success", 0) for r in rows])}


def strata(rows):
    """配对组按 起点差（供体 − 接收）、左右（接收→供体）、绕向（接收→供体）分层；D2 按盘内/出盘；W0 按来源。"""
    tables = {"by_start_diff": {}, "by_side": {}, "by_swing": {}, "D2_by_stratum": {}, "W0_by_from": {}}
    for r in rows:
        if not r.get("donor_type") or r["group"] not in PAIR_GROUPS + ("W0",):
            continue
        rt, dt = r["recv_type"], r["donor_type"]
        tables["by_start_diff"].setdefault(f"{r['group']}:dstart={dt[0] - rt[0]:+d}", []).append(r)
        tables["by_side"].setdefault(f"{r['group']}:{rt[1]:+d}->{dt[1]:+d}", []).append(r)
        tables["by_swing"].setdefault(f"{r['group']}:{rt[2]}->{dt[2]}", []).append(r)
        if r["group"] == "D2":
            tables["D2_by_stratum"].setdefault(r.get("stratum") or "?", []).append(r)
        if r["group"] == "W0":
            tables["W0_by_from"].setdefault(r.get("w0_from") or "?", []).append(r)
    fold = lambda d: {k: {"n": len(v), "success": sum(x["success"] for x in v),
                          "step1_success": sum(x["step1_success"] for x in v),
                          "reach": {c: sum(1 for x in v if x.get("category") == c) for c in REACH_CATS},
                          "swing": {c: sum(1 for x in v if x.get("swing_category") == c) for c in SWING_CATS}}
                      for k, v in sorted(d.items())}
    return {k: fold(v) for k, v in tables.items()}


def main_rows(runs):
    return [r for r in runs if r["stage"] in MAIN_STAGES and (r["stage"] == "n" or r["seed"] == MAIN_SEED)]


def headline_fields(ok_rows):
    """RS_SWAP_FULL / RS_SWAP_INTERSECT 共用的读数字段（不含 N）。"""
    sel = lambda g, f=None: [r for r in ok_rows if r["group"] == g and (f is None or r.get("w0_from") == f)]
    frac = lambda rows, pred: f"{sum(1 for r in rows if pred(r))}/{len(rows)}"
    step1 = lambda r: r.get("step1_success") == 1
    reach = lambda c: (lambda r: r.get("category") == c)
    dsw = lambda r: r.get("swing_category") == "donor_swing"
    return (f"Z={frac(sel('Z'), step1)} Rnd={frac(sel('Rnd'), step1)} A={frac(sel('A'), step1)} "
            f"Bsw_swing={frac(sel('Bsw'), dsw)} Bside_dir={frac(sel('Bside'), reach('donor_dir'))} "
            f"C_own={frac(sel('C'), reach('own'))} D1_swing={frac(sel('D1'), dsw)} "
            f"D2_dir={frac(sel('D2'), reach('donor_dir'))} D2_abs={frac(sel('D2'), reach('donor_abs'))} "
            f"W0_swing={frac(sel('W0', 'Bsw'), dsw)} W0_dir={frac(sel('W0', 'Bside'), reach('donor_dir'))}")


def summarize(runs, plan_rows, model, gate=None):
    mains = main_rows(runs)
    planned = sum(1 for r in plan_rows if r["stage"] in MAIN_STAGES and (r["stage"] == "n" or r["seed"] == MAIN_SEED))
    ok_rows = [r for r in mains if r["status"] == "ok"]
    groups = {g: group_stats([r for r in ok_rows if r["group"] == g]) for g in GROUPS}
    valid = len(ok_rows)
    invalid_reads = sum(1 for r in mains if r["status"] == "invalid")
    retried = sum(1 for r in ok_rows if r.get("attempts", 1) > 1)
    invalid = invalid_reads + retried
    swap_planned = any(r["stage"] == "swap" for r in plan_rows)
    few = bool(gate and gate.get("few"))
    if invalid_reads:
        verdict = "FAIL"
    elif few:
        verdict = "PARTIAL"
    elif valid == planned and invalid == 0 and swap_planned and groups["N"]["n"] == N_EPS:
        verdict = "PASS"
    else:
        verdict = "PARTIAL"
    n_ok = groups["N"]["success"]
    if few:
        sel = lambda g: [r for r in ok_rows if r["group"] == g]
        f1 = lambda rows: f"{sum(r['step1_success'] for r in rows)}/{len(rows)}"
        line = (f"RS_SWAP_FULL={verdict} model={model} reason=few_success r_success={gate.get('r_success')} groups=N,Z,Rnd "
                f"runs={valid} invalid={invalid} N={n_ok}/{N_EPS} Z={f1(sel('Z'))} Rnd={f1(sel('Rnd'))}")
    else:
        line = (f"RS_SWAP_FULL={verdict} model={model} runs={valid} invalid={invalid} N={n_ok}/{N_EPS} "
                + headline_fields(ok_rows))
    return line, {"groups": groups, "strata": strata(ok_rows), "valid": valid, "invalid_reads": invalid_reads,
                  "retried": retried, "planned": planned, "gate": gate, "official": official_info(ok_rows)}


# ---------------------------------------------------------------- 输出
def group_table(summary, title):
    text = [f"## {title}（第一步成功与整局成功括号内为局级重采样 95% 区间，只作参考）", "",
            "| 组 | 有效局 | 第一步成功 | 整局成功 | 官方第一子目标 | " + " | ".join(REACH_CATS) + " | "
            + " | ".join(SWING_CATS) + " |", "|---" * (5 + len(REACH_CATS) + len(SWING_CATS)) + "|"]
    ci = lambda v, n, c: f"{v}/{n}" + (f" [{c[0]:.2f}, {c[1]:.2f}]" if c else "")
    for group in GROUPS:
        g = summary["groups"][group]
        text.append(f"| {group} | {g['n']} | {ci(g['step1_success'], g['n'], g['step1_ci95'])} | "
                    f"{ci(g['success'], g['n'], g['success_ci95'])} | {g['official_step1']}/{g['official_known']} | "
                    + " | ".join(str(g["reach"][c]) for c in REACH_CATS) + " | "
                    + " | ".join(str(g["swing"][c]) for c in SWING_CATS) + " |")
    return text + [""]


def strata_table(name, table):
    text = [f"## {name}", "", "| 分层 | 局数 | 第一步成功 | 整局成功 | " + " | ".join(REACH_CATS) + " | "
            + " | ".join(SWING_CATS) + " |", "|---" * (4 + len(REACH_CATS) + len(SWING_CATS)) + "|"]
    for key, v in table.items():
        text.append(f"| {key} | {v['n']} | {v['step1_success']} | {v['success']} | "
                    + " | ".join(str(v["reach"][c]) for c in REACH_CATS) + " | "
                    + " | ".join(str(v["swing"][c]) for c in SWING_CATS) + " |")
    return text + [""]


def write_md(path, lines, summary, title):
    text = [f"# {title}", "", "## 判定行", "", "```"] + lines + ["```", ""]
    text += group_table(summary, "各组（seed 42；W0 只换起点 0 窗口）")
    for name, key in (("按起点差（供体 − 接收）", "by_start_diff"), ("按左右（接收 → 供体，+1 = L）", "by_side"),
                      ("按绕向（接收 → 供体）", "by_swing"), ("D2 按供体方向在盘内 / 出盘", "D2_by_stratum"),
                      ("W0 按配对来源", "W0_by_from")):
        text += strata_table(name, summary["strata"][key])
    Path(path).write_text("\n".join(text) + "\n")


def readout_model(root, out=None):
    root = Path(root)
    plan = read_json(root / "plan.json")
    model = plan["model"]
    rows = plan["runs"]
    runs = [load_run(r) for r in rows]
    planned_scope = sum(1 for r in rows if r["stage"] in SCOPE_STAGES and r["group"] != "N")
    lines, info = [], {}
    for name, (line, data) in (("demo_layout", judge_layout(runs, model)),
                               ("swap_scope", judge_scope(runs, model, planned_scope)),
                               ("multieq", judge_multieq(runs)),
                               ("seed_first_step", seed_first_step(runs))):
        lines.append(line)
        info[name] = data
    swap_line, summary = summarize(runs, rows, model, plan.get("gate"))
    lines.append(summary["official"])
    lines.append(swap_line)
    out = Path(out) if out else root / "readout"
    out.mkdir(parents=True, exist_ok=True)
    (out / "readout.json").write_text(json.dumps(
        {"model": model, "lines": lines, **info, "summary": summary, "runs": runs,
         "missing": [f"{r['stage']}:{r['group']}:seed{r['seed']}:ep{r['ep']}<-{r['donor']}" for r in runs
                     if r["status"] == "missing"]}, ensure_ascii=False, indent=2) + "\n")
    write_md(out / "readout.md", lines, summary, f"RouteStick 第一子目标交换读数（{model}）")
    return lines, runs, plan


def readout_intersect(root, models, out=None):
    root = Path(root)
    per = {}
    for m in models:
        plan = read_json(root / m / "plan.json")
        per[m] = main_rows([load_run(r) for r in plan["runs"] if r["stage"] in MAIN_STAGES])
    ok_recv = None
    for m in models:
        good = {r["ep"] for r in per[m] if r["stage"] == "n" and r["status"] == "ok" and r["success"] == 1
                and r["ep"] < N_RECV}
        ok_recv = good if ok_recv is None else ok_recv & good
    ok_recv = ok_recv or set()
    tables, parts = {}, [f"RS_SWAP_INTERSECT models={','.join(models)} recv={len(ok_recv)}"]
    for m in models:
        rows = [r for r in per[m] if r["status"] == "ok" and r["ep"] in ok_recv and r["stage"] in ("swap", "w0")]
        tables[m] = {g: group_stats([r for r in rows if r["group"] == g]) for g in SWAP_GROUPS + ("W0",)}
        parts.append(f"{m}:" + headline_fields(rows).replace(" ", ","))
    line = " ".join(parts)
    out = Path(out) if out else root / "readout"
    out.mkdir(parents=True, exist_ok=True)
    (out / "intersect.json").write_text(json.dumps({"line": line, "recv": sorted(ok_recv), "tables": tables},
                                                   ensure_ascii=False, indent=2) + "\n")
    text = ["# 两模型都原样整局成功的接收局上的并排读数", "", "```", line, "```", "",
            f"接收局（{len(ok_recv)} 局）：{sorted(ok_recv)}", "",
            "| 组 | " + " | ".join(f"{m} 局数 | {m} 第一步成功 | {m} own | {m} donor_dir | {m} donor_abs | {m} donor_swing"
                                 for m in models) + " |", "|---" * (1 + 6 * len(models)) + "|"]
    for g in SWAP_GROUPS + ("W0",):
        cells = []
        for m in models:
            s = tables[m][g]
            cells += [str(s["n"]), str(s["step1_success"]), str(s["reach"]["own"]), str(s["reach"]["donor_dir"]),
                      str(s["reach"]["donor_abs"]), str(s["swing"]["donor_swing"])]
        text.append(f"| {g} | " + " | ".join(cells) + " |")
    (out / "intersect.md").write_text("\n".join(text) + "\n")
    return [line]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="<O>/<M>；与 --intersect 同用时为 <O>")
    parser.add_argument("--intersect", default=None, help="如 motion,wan")
    parser.add_argument("--out", default=None, help="输出目录，缺省 <root>/readout")
    args = parser.parse_args(argv)
    if args.intersect:
        lines = readout_intersect(args.root, [m.strip() for m in args.intersect.split(",") if m.strip()], args.out)
    else:
        lines, _, _ = readout_model(args.root, args.out)
    for line in lines:
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
