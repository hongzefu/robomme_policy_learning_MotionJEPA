#!/usr/bin/env python3
"""多步局逐子目标位置 motion 交换读数（计划 1109-subgoal-position-swap-all-lengths-plan.html 第一部分四、五、七节；
第二部分〇′ F6、F8、F9）。

两种入口（只用 CPU）：
  --root <O>/<M>                       单模型：读 <O>/<M>/plan.json（sp_swap_prep specs + donors 写出）列出的每局
  --root <O> --intersect motion,wan    两模型都有的 (接收局, j) 上的并排表
输出写 <root>/readout/（readout.json、readout.md、matrix_<task>_<group>.png；交集 intersect.json、intersect.md），--out 可改。

每局逐步复原（sp_swap_prep.reconstruct）：第 i 步完成 = step_trace 的 task_index 从 E_i 跳出（E_i = m+2+i），到达格 =
该行 tcp_after 最近格；未完成的那一步按几何取第一次到达的非起点格（PL 阈值 0.025、RS 0.03），RS 另算绕向（平均叉积）。
第 j 步归类：own / donor_dir（供体子目标从接收局第 j 步起点走出的格）/ donor_abs（供体绝对目标格；PL 只在同 grid 时）/
other / none / unreached（没走到第 j 步起点）；RS 另 own_swing / donor_swing / other_swing / none。
跟随 = 到达 donor_dir 且（RS）绕向 = 供体绕向。

矩阵 M[j][i]（每任务按 m 分块，再合并按 j 汇总；B 组与 Z 组各一张）：
  i<j：第 i 步结果（到达格、是否完成、RS 绕向）与 N 同一局相同的比率（保持率）
  i=j：B 组为跟随率；Z 组为第 j 步正确率
  i>j：只在第 j 步正确（完成）的局上，第 i 步正确率
主分析（〇′ F6，不按第 j 步是否正确二次筛选）：跟随率 − 同一 (局,j) 集合上 N 运行按同一供体判定的跟随率；
补充：只看 N 中第 j 步正确的 (局,j)。区间按接收局重采样 2000 次（95%）。每 (m,j) 报 n。

判定行：
  SP_REPLAY=PASS|FAIL model=<m> eps=<n> steps=<k> mismatches=<k>   N 运行中按跳变完成的每一步，到达格 = scan 路径
  SP_SEGMENT=PASS|FAIL model=<m> task=<t> eps=<n> bad=<k>           N 回执 demo_task_index 分段
  SWAP_SCOPE=PASS|FAIL model=<m> runs=<n> scope_bad=<k> count_mismatch=<k>   replace_log 每条 [s,s]、起点集合 = W_j
                                                                     （plan 行 w_starts，all/core 两版各自），供体键一致
  MULTI_SWAP_EQ=PASS|FAIL runs=<n> differ=<k>
  SP_MATRIX=PASS|FAIL model=<m> task=<t> runs=<n> invalid=<k> Bsame_follow=<a/b> Bsame_keep=<a/b> Bcross_follow=…
        Bcross_keep=… Z_correct=… Z_keep=…   PASS 只代表无效局为 0
另有信息行 SEED_STEPS（seed 1 与 seed 42 的逐步到达格是否相同）。比率只是读数，不设及格线。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from scripts.interpret import sp_swap_prep as S

BOOT_N, BOOT_SEED = 2000, 20261110
MAIN_STAGES = ("n", "swap", "cross")
CORE_STAGES = ("swapcore", "crosscore")
SCOPE_STAGES = ("smoke", "multieq", "swap", "cross", "seeds", "swapcore", "crosscore")
CELL_CATS = ("own", "donor_dir", "donor_abs", "other", "none", "unreached")
SWING_CATS = ("own_swing", "donor_swing", "other_swing", "none")


def read_json(path):
    return json.loads(Path(path).read_text())


def attempts_for(run_root, key):
    n = 0
    for path in sorted(Path(run_root).glob("attempts/*/commands.json")):
        try:
            if key in read_json(path).get("episode_keys", []):
                n += 1
        except (OSError, ValueError):
            continue
    return n


# ---------------------------------------------------------------- 归类
def step_targets(task, e, j, de=None, dj=None):
    """(own, donor_dir, donor_abs, own_swing, donor_swing)；无供体时后几项 None。"""
    rsg = S.subgoals(e)[j - 1]
    own, osw = rsg["target"], rsg.get("swing")
    if de is None:
        return own, None, None, osw, None
    dsg = S.subgoals(de)[dj - 1]
    ddir = S.donor_dir_target(task, e, j, dsg)
    dabs = dsg["target"]
    if task == "PatternLock" and de.get("grid") != e.get("grid"):
        dabs = None
    return own, ddir, dabs, osw, dsg.get("swing")


def classify_cell(cell, own, ddir=None, dabs=None):
    if cell is None:
        return "none"
    if cell == own:
        return "own"
    if ddir is not None and cell == ddir:
        return "donor_dir"
    if dabs is not None and cell == dabs:
        return "donor_abs"
    return "other"


def classify_swing(actual, own_swing, donor_swing=None):
    if actual is None:
        return "none"
    if actual == own_swing:
        return "own_swing"
    if donor_swing is not None and actual == donor_swing:
        return "donor_swing"
    return "other_swing"


def step_view(task, steps, e, j, de=None, dj=None):
    """一次运行第 j 步：{cell_cat, swing_cat, correct, follow, reached}。"""
    own, ddir, dabs, osw, dsw = step_targets(task, e, j, de, dj)
    if len(steps) < j:
        return {"cell_cat": "unreached", "swing_cat": "none", "correct": False, "follow": False, "reached": None}
    s = steps[j - 1]
    cc = classify_cell(s["reached"], own, ddir, dabs)
    sc = classify_swing(s["swing"], osw, dsw) if task == "RouteStick" else None
    follow = False
    if de is not None and ddir is not None and s["reached"] == ddir:
        follow = task != "RouteStick" or s["swing"] == dsw
    return {"cell_cat": cc, "swing_cat": sc, "correct": bool(s["completed"]), "follow": bool(follow),
            "reached": s["reached"]}


def same_step(a, b, i):
    """两次运行第 i 步结果相同（到达格、是否完成、绕向）；有一方没走到则只有双方都没走到才算相同。"""
    sa = a[i - 1] if len(a) >= i else None
    sb = b[i - 1] if len(b) >= i else None
    if sa is None or sb is None:
        return sa is None and sb is None
    return (sa["reached"], sa["completed"], sa["swing"]) == (sb["reached"], sb["completed"], sb["swing"])


# ---------------------------------------------------------------- 单局
def load_run(row, pool_by_key):
    key = S.episode_key(row["task"], row["ep"])
    out = {k: row.get(k) for k in ("stage", "worker", "site", "group", "seed", "task", "ep", "m", "j", "slot", "donor",
                                    "donor_src", "donor_j", "variant", "spec", "mvariant", "pair_index", "w_starts",
                                    "pad")}
    out["key"] = key
    out["attempts"] = attempts_for(row["run_root"], key)
    run = S.find_run(row["run_root"], row["attn_out"], key)
    if run["terminal"] is None:
        out["status"], out["reason"] = ("invalid" if out["attempts"] else "missing"), "no_terminal"
        return out
    term = run["terminal"]
    out.update(nonce=term.get("episode_nonce"), outcome=term.get("outcome"), motion_content=run["motion_content"],
               receipt=run["receipt"])
    if term.get("outcome") == "error":
        out["status"], out["reason"] = "invalid", "error_outcome"
        return out
    if run["trace"] is None:
        out["status"], out["reason"] = "invalid", "no_trace"
        return out
    e = pool_by_key[key]
    rows = S.read_step_rows(run["trace"])
    out["steps"] = S.reconstruct(row["task"], e, rows, e["m"], term.get("outcome") == "success")
    out["status"] = "ok"
    if row["group"] != "N" and run["motion_content"] is None:
        out["status"], out["reason"] = "invalid", "no_episode_json"
    return out


# ---------------------------------------------------------------- 判定
def judge_replay(runs, pool_by_key, model):
    eps = steps = 0
    bad = {}
    for r in runs:
        if r["stage"] != "n" or r["status"] != "ok":
            continue
        eps += 1
        cells = pool_by_key[r["key"]]["cells"]
        for s in r["steps"]:
            if s["how"] != "jump":
                continue
            steps += 1
            if s["reached"] != int(cells[s["i"]]):
                bad.setdefault(r["key"], []).append([s["i"], s["reached"], int(cells[s["i"]])])
    verdict = "PASS" if not bad and eps else "FAIL"
    return f"SP_REPLAY={verdict} model={model} eps={eps} steps={steps} mismatches={len(bad)}", {"mismatches": bad}


def judge_segment(runs, pool_by_key, model):
    lines, bad = [], {}
    for task in S.TASKS:
        rows = [r for r in runs if r["stage"] == "n" and r["status"] == "ok" and r["task"] == task]
        nbad = 0
        for r in rows:
            rc = r.get("receipt") or {}
            es = rc.get("exec_start_idx")
            segs, why = S.segments_from_labels(rc.get("demo_task_index") or [], es or 0, pool_by_key[r["key"]]["m"])
            why = why or S.segment_ok(task, segs, es)
            if why:
                nbad += 1
                bad[r["key"]] = why
        lines.append(f"SP_SEGMENT={'PASS' if rows and not nbad else 'FAIL'} model={model} task={task} eps={len(rows)} "
                     f"bad={nbad}")
    return lines, {"bad": bad}


def scope_check(r):
    mc = r["motion_content"]
    log = [list(x) for x in mc.get("replace_log", [])]
    starts = {int(x[0]) for x in log}
    same = all(len(x) == 2 and int(x[0]) == int(x[1]) for x in log)
    donor_ok = mc.get("mode") == "swap_exact" and mc.get("donor") == S.episode_key(r["task"], r["donor"])
    want = {int(s) for s in (r.get("w_starts") or [])}
    n_rep = int(mc.get("n_replaced", -1))
    return bool(log) and same and donor_ok and starts == want, n_rep == len(starts) == len(log)


def judge_scope(runs, model):
    details, bad, mismatch = [], 0, 0
    for r in runs:
        if r["stage"] not in SCOPE_STAGES or r["group"] == "N" or r["status"] != "ok":
            continue
        ok, cnt = scope_check(r)
        bad += 0 if ok else 1
        mismatch += 0 if cnt else 1
        details.append({"stage": r["stage"], "key": r["key"], "j": r["j"], "group": r["group"], "variant": r["variant"],
                        "scope_ok": ok, "count_ok": cnt, "w_starts": r.get("w_starts"),
                        "starts": sorted({int(x[0]) for x in r["motion_content"].get("replace_log", [])})})
    verdict = "FAIL" if (bad or mismatch or not details) else "PASS"
    return (f"SWAP_SCOPE={verdict} model={model} runs={len(details)} scope_bad={bad} count_mismatch={mismatch}",
            {"details": details})


def judge_multieq(runs):
    by = {}
    for r in runs:
        if r["stage"] == "multieq":
            by.setdefault(r.get("pair_index"), {})[r.get("mvariant")] = r
    differ, complete = [], 0
    for idx, v in by.items():
        a, b = v.get("single"), v.get("multi")
        if not (a and b and a["status"] == "ok" and b["status"] == "ok"):
            continue
        complete += 1
        fa = [[s["reached"], s["completed"], s["swing"]] for s in a["steps"]] + [a["outcome"]]
        fb = [[s["reached"], s["completed"], s["swing"]] for s in b["steps"]] + [b["outcome"]]
        if fa != fb:
            differ.append(idx)
    verdict = "FAIL" if differ or complete != len(by) or not by else "PASS"
    return f"MULTI_SWAP_EQ={verdict} runs={complete} differ={len(differ)}", {"differ": differ, "planned": len(by)}


def seed_steps(runs):
    def sig(r):
        return [[s["reached"], s["completed"], s["swing"]] for s in r["steps"]]
    base = {(r["group"], r["key"], r.get("j")): r for r in runs if r["stage"] in ("n", "swap") and r["status"] == "ok"
            and r["seed"] == S.MAIN_SEED}
    n = agree = 0
    for r in runs:
        if r["stage"] == "seeds" and r["status"] == "ok":
            b = base.get((r["group"], r["key"], r.get("j")))
            if b is None:
                continue
            n += 1
            agree += int(sig(b) == sig(r))
    return f"SEED_STEPS pairs={n} agree={agree}"


# ---------------------------------------------------------------- 矩阵与主分析
def bootstrap_diff(rows, a_key, b_key=None, n=BOOT_N, seed=BOOT_SEED):
    """rows：带 recv、a_key（及 b_key）布尔的 dict。按接收局重采样，返回 mean(a) − mean(b) 的 95% 区间。"""
    if not rows:
        return None
    by = {}
    for r in rows:
        by.setdefault(r["recv"], []).append(r)
    keys = sorted(by)
    sums = np.array([[sum(float(x[a_key]) for x in by[k]), sum(float(x[b_key]) for x in by[k]) if b_key else 0.0,
                      len(by[k])] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    tot = sums[idx].sum(axis=1)
    vals = (tot[:, 0] - tot[:, 1]) / tot[:, 2]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def position_rows(runs, pool_by_key, n_by_key, stages=MAIN_STAGES, eligibility=None):
    """干预局 → 每局一行 {recv, task, m, j, group, variant, follow, base_follow, correct, n_correct, keep[i], later[i], cats}。"""
    out = []
    for r in runs:
        if r["stage"] not in stages or r["group"] == "N" or r["seed"] != S.MAIN_SEED:
            continue
        nrun = n_by_key.get(r["key"])
        if r["status"] != "ok" or nrun is None or nrun["status"] != "ok":
            out.append({"recv": r["key"], "task": r["task"], "m": r["m"], "j": r["j"], "group": r["group"],
                        "variant": r["variant"], "valid": False, "reason": r.get("reason") or "no_n"})
            continue
        e = pool_by_key[r["key"]]
        de = pool_by_key.get(r["donor_src"]) if r.get("donor_src") else None
        j = int(r["j"])
        v = step_view(r["task"], r["steps"], e, j, de, r.get("donor_j"))
        nv = step_view(r["task"], nrun["steps"], e, j, de, r.get("donor_j"))
        keep = {i: same_step(r["steps"], nrun["steps"], i) for i in range(1, j)}
        later = {}
        if v["correct"]:
            for i in range(j + 1, int(e["m"]) + 1):
                later[i] = len(r["steps"]) >= i and bool(r["steps"][i - 1]["completed"])
        out.append({"recv": r["key"], "task": r["task"], "m": int(e["m"]), "j": j, "group": r["group"],
                    "variant": r["variant"], "valid": True, "follow": v["follow"], "base_follow": nv["follow"],
                    "correct": v["correct"], "n_correct": nv["correct"], "cell_cat": v["cell_cat"],
                    "swing_cat": v["swing_cat"], "keep": keep, "later": later,
                    "clean": window_clean((eligibility or {}).get(r["key"]), j)})
    return out


def window_clean(elig, j):
    """W_j 是否不含跨界窗（all 版 = core 版）；取不到返回 None。"""
    if not elig or not elig.get("W_all") or not elig.get("W_core") or len(elig["W_all"]) < j:
        return None
    return list(elig["W_all"][j - 1]) == list(elig["W_core"][j - 1])


def rate(flags):
    flags = list(flags)
    return [int(sum(flags)), len(flags)]


def matrix(rows, task, group, m=None):
    """M[j][i] = [分子, 分母]；i<j 保持率，i=j 跟随率（Z 组为正确率），i>j 第 j 步正确局上第 i 步正确率。"""
    rows = [r for r in rows if r["valid"] and r["task"] == task and r["group"] == group and (m is None or r["m"] == m)]
    mmax = max([r["m"] for r in rows], default=0)
    M = {}
    for j in range(1, mmax + 1):
        mine = [r for r in rows if r["j"] == j]
        if not mine:
            continue
        row = {}
        for i in range(1, mmax + 1):
            if i < j:
                row[i] = rate(r["keep"][i] for r in mine if i in r["keep"])
            elif i == j:
                row[i] = rate((r["correct"] if group == "Z" else r["follow"]) for r in mine)
            else:
                row[i] = rate(r["later"][i] for r in mine if i in r["later"])
        M[j] = row
    return M


def follow_stats(rows):
    """主分析与补充分析：跟随率、N 基线、差值与接收局重采样区间。"""
    n = len(rows)
    out = {"n": n, "recv": len({r["recv"] for r in rows})}
    if not n:
        return out
    f, b = np.mean([r["follow"] for r in rows]), np.mean([r["base_follow"] for r in rows])
    out.update(follow=float(f), base=float(b), diff=float(f - b), ci=bootstrap_diff(rows, "follow", "base_follow"),
               correct=float(np.mean([r["correct"] for r in rows])))
    sup = [r for r in rows if r["n_correct"]]
    out["supp"] = {"n": len(sup)}
    if sup:
        f2, b2 = np.mean([r["follow"] for r in sup]), np.mean([r["base_follow"] for r in sup])
        out["supp"].update(follow=float(f2), base=float(b2), diff=float(f2 - b2),
                           ci=bootstrap_diff(sup, "follow", "base_follow"))
    cats = {}
    for r in rows:
        cats[r["cell_cat"]] = cats.get(r["cell_cat"], 0) + 1
    out["cats"] = cats
    if rows[0]["task"] == "RouteStick":
        sw = {}
        for r in rows:
            sw[r["swing_cat"]] = sw.get(r["swing_cat"], 0) + 1
        out["swing_cats"] = sw
    return out


def task_summary(rows, task, variant="all"):
    rows = [r for r in rows if r["task"] == task and r.get("variant", "all") == variant]
    s = {"invalid": sum(1 for r in rows if not r["valid"]), "runs": len(rows), "groups": {}}
    for group in S.SWAP_GROUPS:
        g = [r for r in rows if r["valid"] and r["group"] == group]
        ms = sorted({r["m"] for r in g})
        s["groups"][group] = {
            "matrix": matrix(rows, task, group),
            "by_m": {m: matrix(rows, task, group, m) for m in ms},
            "by_mj": {f"{m},{j}": follow_stats([r for r in g if r["m"] == m and r["j"] == j])
                      for m in ms for j in range(1, m + 1) if any(r["m"] == m and r["j"] == j for r in g)},
            "by_j": {j: follow_stats([r for r in g if r["j"] == j]) for j in sorted({r["j"] for r in g})},
            "all": follow_stats(g),
            "all_clean": follow_stats([r for r in g if r.get("clean")]),
            "all_crossing": follow_stats([r for r in g if r.get("clean") is False]),
            "diag": rate((r["correct"] if group == "Z" else r["follow"]) for r in g),
            "keep": rate(v for r in g for v in r["keep"].values()),
        }
    return s


def matrix_line(model, task, s):
    g = s["groups"]
    verdict = "PASS" if s["invalid"] == 0 and s["runs"] else "FAIL"
    f = lambda x: f"{x[0]}/{x[1]}"
    return (f"SP_MATRIX={verdict} model={model} task={task} runs={s['runs']} invalid={s['invalid']} "
            f"Bsame_follow={f(g['Bsame']['diag'])} Bsame_keep={f(g['Bsame']['keep'])} "
            f"Bcross_follow={f(g['Bcross']['diag'])} Bcross_keep={f(g['Bcross']['keep'])} "
            f"Z_correct={f(g['Z']['diag'])} Z_keep={f(g['Z']['keep'])}")


def heatmap(path, M, cell=48):
    """M[j][i] → 彩色热力图 png；每格按最近邻放大（cv2.INTER_NEAREST，规则第 5 条），无数据格为灰。"""
    import cv2
    if not M:
        return False
    size = max(max(M), max(max(r) for r in M.values()))
    val = np.full((size, size), np.nan)
    for j, row in M.items():
        for i, (a, b) in row.items():
            if b:
                val[int(j) - 1, int(i) - 1] = a / b
    img = np.zeros((size, size, 3), np.uint8)
    u8 = np.nan_to_num(val * 255, nan=0).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
    img[:] = color
    img[np.isnan(val)] = (128, 128, 128)
    big = cv2.resize(img, (size * cell, size * cell), interpolation=cv2.INTER_NEAREST)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), big))


# ---------------------------------------------------------------- 汇总
def collect(root):
    root = Path(root)
    plan = read_json(root / "plan.json")
    pool, pool_by_key = S.load_pool(root.parent)
    runs = [load_run(row, pool_by_key) for row in plan["runs"]]
    return plan, pool_by_key, runs


def n_index(runs):
    return {r["key"]: r for r in runs if r["stage"] == "n" and r["group"] == "N" and r["seed"] == S.MAIN_SEED}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def fmt_stats(st):
    if not st.get("n"):
        return "n=0"
    ci = st.get("ci")
    txt = (f"n={st['n']}（{st['recv']} 局）跟随 {st['follow']:.2f} 基线 {st['base']:.2f} 差 {st['diff']:+.2f}"
           + (f" [{ci[0]:+.2f},{ci[1]:+.2f}]" if ci else ""))
    sup = st.get("supp") or {}
    if sup.get("n"):
        txt += f"；补充 n={sup['n']} 差 {sup['diff']:+.2f}"
    return txt


def write_md(path, title, lines, summaries):
    md = [f"# {title}", "", "## 判定行", ""] + [f"- `{l}`" for l in lines] + [""]
    for task, s in summaries.items():
        md += [f"## {task}", ""]
        for group in S.SWAP_GROUPS:
            g = s["groups"][group]
            md += [f"### {group}", "", f"- 全部：{fmt_stats(g['all'])}",
                   f"- W_j 无跨界窗：{fmt_stats(g['all_clean'])}", f"- W_j 含跨界窗：{fmt_stats(g['all_crossing'])}", ""]
            M = g["matrix"]
            if M:
                size = max(max(M), max(max(r) for r in M.values()))
                md += ["| j \\ i | " + " | ".join(str(i) for i in range(1, size + 1)) + " |",
                       "|---|" + "---|" * size]
                for j in sorted(M):
                    md.append(f"| {j} | " + " | ".join(
                        (f"{M[j][i][0]}/{M[j][i][1]}" if i in M[j] and M[j][i][1] else "—") for i in range(1, size + 1))
                        + " |")
                md.append("")
            md += ["| (m,j) | 读数 |", "|---|---|"]
            for k, st in g["by_mj"].items():
                md.append(f"| {k} | {fmt_stats(st)} |")
            md += ["", "| j（合并） | 读数 |", "|---|---|"]
            for j, st in g["by_j"].items():
                md.append(f"| {j} | {fmt_stats(st)} |")
            md.append("")
    Path(path).write_text("\n".join(md) + "\n")


def readout_model(root, out=None):
    root = Path(root)
    plan, pool_by_key, runs = collect(root)
    model = plan["model"]
    lines, extra = [], {}
    line, extra["replay"] = judge_replay(runs, pool_by_key, model)
    lines.append(line)
    seg_lines, extra["segment"] = judge_segment(runs, pool_by_key, model)
    lines += seg_lines
    line, extra["scope"] = judge_scope(runs, model)
    lines.append(line)
    line, extra["multieq"] = judge_multieq(runs)
    lines.append(line)
    lines.append(seed_steps(runs))
    nidx = n_index(runs)
    elig = plan.get("eligibility") or {}
    rows = position_rows(runs, pool_by_key, nidx, eligibility=elig)
    core_rows = position_rows(runs, pool_by_key, nidx, CORE_STAGES, eligibility=elig)
    out = Path(out) if out else root / "readout"
    out.mkdir(parents=True, exist_ok=True)
    summaries, core = {}, {}
    for task in S.TASKS:
        summaries[task] = task_summary(rows, task)
        lines.append(matrix_line(model, task, summaries[task]))
        if core_rows:
            core[task] = task_summary(core_rows, task, "core")
            lines.append(matrix_line(model, task, core[task]).replace("SP_MATRIX=", "SP_MATRIX_CORE="))
        for group in S.SWAP_GROUPS:
            heatmap(out / f"matrix_{task}_{group}.png", summaries[task]["groups"][group]["matrix"])
    payload = {"model": model, "lines": lines, "summaries": _jsonable(summaries), "core": _jsonable(core),
               "details": _jsonable(extra), "rows": _jsonable(rows + core_rows)}
    (out / "readout.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    write_md(out / "readout.md", f"逐子目标位置交换读数（{model}）", lines, summaries)
    for line in lines:
        print(line, flush=True)
    return 0 if all("=FAIL" not in l for l in lines) else 1


def readout_intersect(root, models, out=None):
    root = Path(root)
    per = {m: collect(root / m) for m in models}
    pos = {}
    for m, (plan, pool_by_key, runs) in per.items():
        pos[m] = position_rows(runs, pool_by_key, n_index(runs), eligibility=plan.get("eligibility") or {})
    common = set.intersection(*[{(r["recv"], r["j"]) for r in rows if r["valid"]} for rows in pos.values()])
    lines, summaries = [], {}
    for task in S.TASKS:
        parts = []
        for m in models:
            rows = [r for r in pos[m] if (r["recv"], r["j"]) in common]
            s = task_summary(rows, task)
            summaries.setdefault(task, {})[m] = s
            g = s["groups"]
            parts.append(f"{m}:" + ",".join(f"{k}={g[k]['diag'][0]}/{g[k]['diag'][1]}" for k in S.SWAP_GROUPS))
        n = sum(1 for k in common if S.split_key(k[0])[0] == task)
        lines.append(f"SP_MATRIX_INTERSECT models={','.join(models)} task={task} positions={n} " + " ".join(parts))
    out = Path(out) if out else root / "readout"
    out.mkdir(parents=True, exist_ok=True)
    (out / "intersect.json").write_text(json.dumps(_jsonable({"models": list(models), "positions": sorted(common),
                                                              "lines": lines, "summaries": summaries}),
                                                   ensure_ascii=False, indent=2, default=str) + "\n")
    md = [f"# 两模型交集读数（{','.join(models)}）", ""] + [f"- `{l}`" for l in lines]
    (out / "intersect.md").write_text("\n".join(md) + "\n")
    for line in lines:
        print(line, flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--intersect", default=None, help="逗号分隔的模型名，如 motion,wan")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.intersect:
        return readout_intersect(args.root, [m.strip() for m in args.intersect.split(",") if m.strip()], args.out)
    return readout_model(args.root, args.out)


if __name__ == "__main__":
    sys.exit(main())
