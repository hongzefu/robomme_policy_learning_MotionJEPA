#!/usr/bin/env python3
"""StopCube motion 速度替换读数（计划 1110-stopcube-motion-speed-swap-plan.html §3.1、§3.3、§5；第二部分〇′ B1–B5、
N1–N5 与 1.4 节）。只用 CPU。

用法：--root <O>/<M> [--smoke] [--out <目录>]；读 <根>/plan.json（sc_speed_prep specs/donors 写出）列出的 N、S、D（供体）、
R（干预）回合的终局、逐步轨迹、供体回执与服务端 episode.json，输出 <根>/readout/readout.{md,json}。

窗口：W(M′,N) 由 M′ 供体录制轨迹经 stopcube_windows.geometry_window（definition 的 move_interval = M′、target_pass = N
（取配置，不取行内改写后的 99）、cube_half_size 取回执）解出；W(M,N) 取 M′ == M 那份供体轨迹。p = 第一个
button_pressed_after 为真的 env_step_after。

判定行：
  SC_POOL=PASS|FAIL model=<m> eps=<n> mismatch=<k>          N 轨迹行内 move_interval、target_pass 与局池一致
  WINDOW_GEOM=PASS|FAIL model=<m> donors=<n> resolved=<k> formula_dev_max=<步>
  PRESS_LABEL=PASS|FAIL model=<m> runs=<n> mismatch=<k>     「p ∈ W(M,N)」与官方 task_success 逐局一致（N/S/R）
  SPEED_SCOPE=PASS|FAIL model=<m> runs=<n> unreplaced=<k> scope_bad=<k>
        每个 R 回合：最后一次推理时已齐的窗口（起点 s=32u、s+32 ≤ 该次推理前的帧号）都在 replace_log 里且为 [s,s]，
        另加 n_windows − n_replaced；scope_bad = 模式不是 swap_exact、供体键不符、缺 episode.json 或取不到推理帧号
  SEED_PRESS model=<m> pairs=<n> mad=<步> mean_abs=<步> press_change=<k>   N 与 S 都按下的局 |p_S − p_N| 的中位数
  SPEED_FOLLOW model=<m> beta=<b> ci=[..] beta0=<过原点> same_delta=<d> ci=[..] n=<n>
        Δ = p_R − p_N（两者都按下），Δ* = T(M′,N) − T(M,N)，β 为带截距 OLS 斜率（只用 M′ ≠ M），接收局重采样 2000 次
  SPEED_SUCCESS model=<m> group=<fast|slow|all> new_speed=<k>/<n> base=<k>/<n> diff=<d> ci=[..] true_speed=<k>/<n>
        neither=<k> nopress=<k> overlap=<k>
  SPEED_SAME model=<m> true_speed=<k>/<n> base_true=<k>/<n> nopress=<k>   R_same 只报真实成功（〇′ N2）
  SPEED_FOLLOW_SYM model=<m> beta=… ci=… beta0=… same_delta=… ci=… n=<n>
        对称截断口径：R 的按下步也只认 ≤ M·N（超出视为未按，与 N 回合受原超时截断一致），其余同 SPEED_FOLLOW
  SPEED_LATE model=<m> group=<fast|slow|same> runs=<n> late_after_MN=<k> after_cube_stop=<k> timeout_off=<k>/<n>
        R 回合按下步 > M·N 的局数、按下步 > 5·M（方块已静止）的局数；timeout_off = 有 meta/timeout_off_receipt.json
        （关超时跑到 horizon）的 R 回合数，兼容旧口径（未关超时）冒烟
  SPEED_NEWWIN_LATE model=<m> group=slow w_new_after_MN=<k>/<n>     slow 组 W(M′,N) 整体在 M·N 之后的局数
R 回合「未按」= horizon（max(M,M′)·N+32）内未按。
  SC_INVALID model=<m> …                                    各类无效单独计数，不静默丢弃
另写补充：按 (M→M′) 格与 (M→M′,N) 格的 Δ 均值、「减 R_same」版斜率、各条件按下率、新速度成功分格表。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from scripts.interpret import sc_speed_prep as C
from scripts.interpret.analysis.sp_swap_readout import bootstrap_diff
from scripts.interpret.hover_points import press_step

BOOT_N, BOOT_SEED = 2000, 20261110
GROUPS = ("fast", "slow")
DONOR_OUTCOME = "speed_donor_horizon"
CUBE_STOP_PASSES = 5          # 方块至多经过 5 次后静止：按下步 > 5·M 记 after_cube_stop


# ---------------------------------------------------------------- 纯计算
def in_window(p, w):
    return p is not None and w is not None and w[0] <= p <= w[1]


def windows_overlap(a, b):
    return a is not None and b is not None and max(a[0], b[0]) <= min(a[1], b[1])


def classify_press(p, w_true, w_new):
    """§3.3 四类：true / new / nopress / overlap / neither_early|neither_between|neither_late。"""
    if windows_overlap(w_true, w_new):
        return "overlap"
    if p is None:
        return "nopress"
    if in_window(p, w_true):
        return "true"
    if in_window(p, w_new):
        return "new"
    lo, hi = min(w_true[0], w_new[0]), max(w_true[1], w_new[1])
    if p < lo:
        return "neither_early"
    if p > hi:
        return "neither_late"
    return "neither_between"


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2 or float(np.var(x)) == 0.0:
        return None
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def origin_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = float(np.sum(x * x))
    return None if d == 0.0 else float(np.sum(x * y) / d)


def boot_by_recv(rows, stat, n=BOOT_N, seed=BOOT_SEED):
    """按接收局重采样：rows 带 recv；stat(rows) → 数或 None。返回 95% 区间或 None。"""
    by = {}
    for r in rows:
        by.setdefault(r["recv"], []).append(r)
    keys = sorted(by)
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for idx in rng.integers(0, len(keys), size=(n, len(keys))):
        v = stat([r for i in idx for r in by[keys[i]]])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def follow_stats(rows, ykey="delta"):
    """rows：{recv, dstar, delta}。返回 β（带截距）、β0（过原点）与 β 的重采样区间。"""
    rows = [r for r in rows if r.get(ykey) is not None]
    out = {"n": len(rows), "recv": len({r["recv"] for r in rows})}
    if not rows:
        return out
    x, y = [r["dstar"] for r in rows], [r[ykey] for r in rows]
    out.update(beta=ols_slope(x, y), beta0=origin_slope(x, y),
               ci=boot_by_recv(rows, lambda rs: ols_slope([r["dstar"] for r in rs], [r[ykey] for r in rs])))
    return out


def mean_stats(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    rs = [r for r in rows if r.get(key) is not None]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": float(np.mean(vals)),
            "ci": boot_by_recv(rs, lambda xs: float(np.mean([r[key] for r in xs])) if xs else None)}


def success_stats(rows):
    """rows：每个 R 回合 {recv, cat, base_new, true}（cat 来自 classify_press）。overlap 不进新速度成功分子分母。"""
    cats = Counter(r["cat"] for r in rows)
    usable = [dict(r, new=r["cat"] == "new", base=bool(r["base_new"])) for r in rows if r["cat"] != "overlap"]
    k_new, k_base = sum(r["new"] for r in usable), sum(r["base"] for r in usable)
    n = len(usable)
    return {"n_all": len(rows), "n": n, "new": k_new, "base": k_base,
            "diff": (k_new - k_base) / n if n else None,
            "ci": bootstrap_diff(usable, "new", "base", BOOT_N, BOOT_SEED) if n else None,
            "true": sum(1 for r in rows if r["true"]),
            "neither": sum(v for k, v in cats.items() if k.startswith("neither")),
            "neither_detail": {k: v for k, v in cats.items() if k.startswith("neither")},
            "nopress": cats["nopress"], "overlap": cats["overlap"]}


def fmt_ci(ci, nd=2):
    return "[na]" if not ci else f"[{ci[0]:.{nd}f},{ci[1]:.{nd}f}]"


def fmt_num(x, nd=2):
    return "na" if x is None else f"{x:.{nd}f}"


def follow_line(model, fs, same, name="SPEED_FOLLOW"):
    return (f"{name} model={model} beta={fmt_num(fs.get('beta'), 3)} ci={fmt_ci(fs.get('ci'), 3)} "
            f"beta0={fmt_num(fs.get('beta0'), 3)} same_delta={fmt_num(same.get('mean'), 1)} "
            f"ci={fmt_ci(same.get('ci'), 1)} n={fs['n']}")


def sym_press(p, M, N):
    """对称截断口径：按下步只认 ≤ M·N，超出视为未按。"""
    return p if p is not None and p <= int(M) * int(N) else None


def late_stats(rows):
    """rows：R 回合 {press, M, N, timeout_off}。返回 late_after_MN、after_cube_stop 与 timeout_off 计数。"""
    return {"runs": len(rows),
            "late_after_MN": sum(1 for r in rows if r["press"] is not None and r["press"] > r["M"] * r["N"]),
            "after_cube_stop": sum(1 for r in rows if r["press"] is not None and r["press"] > CUBE_STOP_PASSES * r["M"]),
            "timeout_off": sum(1 for r in rows if r.get("timeout_off"))}


def late_line(model, group, st):
    return (f"SPEED_LATE model={model} group={group} runs={st['runs']} late_after_MN={st['late_after_MN']} "
            f"after_cube_stop={st['after_cube_stop']} timeout_off={st['timeout_off']}/{st['runs']}")


def success_line(model, group, st):
    return (f"SPEED_SUCCESS model={model} group={group} new_speed={st['new']}/{st['n']} base={st['base']}/{st['n']} "
            f"diff={fmt_num(st['diff'], 3)} ci={fmt_ci(st['ci'], 3)} true_speed={st['true']}/{st['n_all']} "
            f"neither={st['neither']} nopress={st['nopress']} overlap={st['overlap']}")


# ---------------------------------------------------------------- 读回合
def load_run(row):
    """一回合 → {status, outcome, success, press, last_step, trace, receipt, donor_receipt, motion_content, last_frame}。"""
    out = {k: row.get(k) for k in ("stage", "group", "ep", "M", "N", "slot", "M_prime", "syn", "kind", "seed", "spec")}
    out["recv"] = C.episode_key(row["ep"])
    run = C.find_sc_run(row)
    if run["terminal"] is None:
        out["status"] = "no_terminal"
        return out
    out["outcome"] = run["terminal"].get("outcome")
    out["success"] = out["outcome"] == "success"
    out["motion_content"] = run["motion_content"]
    out["donor_receipt"] = run["donor_receipt"]
    out["timeout_receipt"] = run.get("timeout_receipt")
    out["timeout_off"] = out["timeout_receipt"] is not None
    if out["outcome"] == "error":
        out["status"] = "error_outcome"
        return out
    if run["trace"] is None:
        out["status"] = "no_trace"
        return out
    try:
        trace = C.load_trace(run["trace"])
    except ValueError as error:
        out.update(status="trace_bad", reason=str(error))
        return out
    rows = trace["rows"]
    if not rows:
        out["status"] = "empty_trace"
        return out
    press = C.first_press(rows)
    if row.get("group") == "R" and press is not None and press > C.horizon(row["M"], row["M_prime"], row["N"]):
        press = None                      # R 回合「未按」= horizon 内未按
    out.update(status="ok", trace=trace, press=press, last_step=rows[-1]["env_step_after"],
               first_env_step=rows[0]["env_step_before"], move_interval=rows[0].get("move_interval"),
               target_pass=rows[0].get("target_pass"))
    starts = [r.get("chunk_start_env_step") for r in rows if isinstance(r.get("chunk_start_env_step"), int)]
    out["last_frame"] = (max(starts) - out["first_env_step"]) if starts else None
    return out


def donor_window(d, N):
    """供体回合 → (窗口 [in,out] 或 None, 原因)。"""
    from scripts.interpret.stopcube_windows import geometry_window
    if d is None or d["status"] != "ok":
        return None, (d or {}).get("status", "missing")
    rc = d.get("donor_receipt") or {}
    if rc.get("button_ever_pressed") is not False:
        return None, "donor_pressed_or_no_receipt"
    geo = geometry_window([d["trace"]], {"move_interval": int(d["M_prime"]), "target_pass": int(N),
                                         "cube_half_size": rc.get("cube_half_size")})
    if geo["window_status"] != "resolved":
        return None, "unresolved"
    return (int(geo["target_visit_in"]), int(geo["target_visit_out"])), None


def scope_check(r):
    """返回 (unreplaced, scope_bad 原因或 None)。"""
    mc = r.get("motion_content")
    if mc is None:
        return 0, "no_episode_json"
    if mc.get("mode") != "swap_exact" or mc.get("donor") != C.episode_key(r["syn"]):
        return 0, "mode_or_donor"
    log = [list(x) for x in mc.get("replace_log", [])]
    bad = [x for x in log if len(x) != 2 or int(x[0]) != int(x[1])]
    replaced = {int(x[0]) for x in log if len(x) == 2 and int(x[0]) == int(x[1])}
    F = r.get("last_frame")
    if F is None:
        return 0, "no_last_frame"
    expected = {s for s in range(0, F + 1, C.MOTION_STRIDE) if s + C.MOTION_WINDOW - 1 <= F}
    extra = max(0, int(mc.get("n_windows", 0)) - int(mc.get("n_replaced", 0)))
    return len(expected - replaced) + extra, ("non_identity_log" if bad else None)


# ---------------------------------------------------------------- 汇总
def collect(root):
    root = Path(root)
    plan = C.P._read_json(root / "plan.json")
    _, by_ep = C.load_pool(Path(plan["pool_root"]))
    runs = [load_run(r) for r in plan["runs"]]
    return plan, by_ep, runs


def analyze(plan, by_ep, runs):
    model = plan["model"]
    lines, invalid = [], Counter()
    N_ = {r["ep"]: r for r in runs if r["group"] == "N"}
    S_ = {r["ep"]: r for r in runs if r["group"] == "S"}
    D_ = {(r["ep"], r["slot"]): r for r in runs if r["group"] == "D"}
    R_ = [r for r in runs if r["group"] == "R"]
    for r in runs:
        if r["status"] != "ok":
            invalid[f"{r['group']}:{r['status']}"] += 1
        elif r["group"] == "D" and r["outcome"] != DONOR_OUTCOME:
            invalid[f"D:outcome_{r['outcome']}"] += 1

    # SC_POOL 现场核对
    ok_n = [r for r in N_.values() if r["status"] == "ok"]
    mism = [r["ep"] for r in ok_n if r["move_interval"] != by_ep[r["ep"]]["M"] or r["target_pass"] != by_ep[r["ep"]]["N"]]
    lines.append(f"SC_POOL={'PASS' if ok_n and not mism else 'FAIL'} model={model} eps={len(ok_n)} mismatch={len(mism)}")

    # 窗口
    win, dev, unresolved = {}, [], {}
    for (ep, slot), d in D_.items():
        w, why = donor_window(d, by_ep[ep]["N"])
        win[(ep, slot)] = w
        if w is None:
            unresolved[f"{C.episode_key(ep)}:s{slot}"] = why
        else:
            dev.append(abs((w[0] + w[1]) / 2 - press_step(C.SLOTS[slot], by_ep[ep]["N"])))
    resolved = sum(1 for w in win.values() if w is not None)
    lines.append(f"WINDOW_GEOM={'PASS' if D_ and resolved == len(D_) else 'FAIL'} model={model} donors={len(D_)} "
                 f"resolved={resolved} formula_dev_max={fmt_num(max(dev) if dev else None, 1)}")

    def w_true(ep):
        return win.get((ep, C.slot_of(by_ep[ep]["M"])))

    # PRESS_LABEL
    label_runs = mismatch = 0
    label_bad = []
    for r in list(N_.values()) + list(S_.values()) + R_:
        if r["status"] != "ok" or w_true(r["ep"]) is None:
            continue
        label_runs += 1
        if in_window(r["press"], w_true(r["ep"])) != bool(r["success"]):
            mismatch += 1
            label_bad.append([r["group"], r["recv"], r.get("slot"), r["press"], list(w_true(r["ep"])), r["outcome"]])
    lines.append(f"PRESS_LABEL={'PASS' if label_runs and not mismatch else 'FAIL'} model={model} runs={label_runs} "
                 f"mismatch={mismatch}")

    # SPEED_SCOPE
    sc_runs = unrep = bad = 0
    scope_detail = []
    for r in R_:
        if r["status"] != "ok":
            continue
        sc_runs += 1
        u, why = scope_check(r)
        unrep += u
        bad += int(why is not None)
        if u or why:
            scope_detail.append({"recv": r["recv"], "slot": r["slot"], "unreplaced": u, "why": why})
    lines.append(f"SPEED_SCOPE={'PASS' if sc_runs and not unrep and not bad else 'FAIL'} model={model} runs={sc_runs} "
                 f"unreplaced={unrep} scope_bad={bad}")

    # SEED_PRESS
    diffs, change = [], 0
    for ep, n in N_.items():
        s = S_.get(ep)
        if n["status"] != "ok" or s is None or s["status"] != "ok":
            continue
        if n["press"] is not None and s["press"] is not None:
            diffs.append(abs(s["press"] - n["press"]))
        elif (n["press"] is None) != (s["press"] is None):
            change += 1
    lines.append(f"SEED_PRESS model={model} pairs={len(diffs)} mad={fmt_num(float(np.median(diffs)) if diffs else None, 1)} "
                 f"mean_abs={fmt_num(float(np.mean(diffs)) if diffs else None, 1)} press_change={change}")

    # Δ 行与成功分类
    same_press = {r["ep"]: r["press"] for r in R_ if r["status"] == "ok" and r["kind"] == "same"}
    drows, srows, same_rows = [], [], []
    for r in R_:
        n = N_.get(r["ep"])
        e = by_ep[r["ep"]]
        if r["status"] != "ok" or n is None or n["status"] != "ok":
            if r["status"] == "ok":
                invalid["R:no_valid_N"] += 1
            continue
        M, Mp, Nn = e["M"], int(r["M_prime"]), e["N"]
        dstar = C.delta_star(M, Mp, Nn)
        both = r["press"] is not None and n["press"] is not None
        sp = same_press.get(r["ep"])
        p_sym, n_sym = sym_press(r["press"], M, Nn), sym_press(n["press"], M, Nn)
        base = {"recv": r["recv"], "M": M, "Mp": Mp, "N": Nn, "kind": r["kind"], "dstar": dstar,
                "delta": (r["press"] - n["press"]) if both else None,
                "delta_sym": (p_sym - n_sym) if (p_sym is not None and n_sym is not None) else None,
                "timeout_off": bool(r.get("timeout_off")),
                "delta_same": (r["press"] - sp) if (r["press"] is not None and sp is not None) else None,
                "p_R": r["press"], "p_N": n["press"]}
        drows.append(base)
        wt, wn = w_true(r["ep"]), win.get((r["ep"], r["slot"]))
        if wt is None or wn is None:
            invalid["R:window_unresolved"] += 1
            continue
        if r["kind"] == "same":
            same_rows.append({"recv": r["recv"], "true": in_window(r["press"], wt), "nopress": r["press"] is None,
                              "base_true": in_window(n["press"], wt)})
            continue
        srows.append(dict(base, cat=classify_press(r["press"], wt, wn), true=in_window(r["press"], wt),
                          w_new_after_MN=wn[0] > M * Nn,
                          base_new=in_window(n["press"], wn), w_true=list(wt), w_new=list(wn)))

    diff_rows = [d for d in drows if d["kind"] != "same"]
    fs = follow_stats(diff_rows)
    same = mean_stats([d for d in drows if d["kind"] == "same"], "delta")
    lines.append(follow_line(model, fs, same))
    fs_sym = follow_stats(diff_rows, "delta_sym")
    same_sym = mean_stats([d for d in drows if d["kind"] == "same"], "delta_sym")
    lines.append(follow_line(model, fs_sym, same_sym, "SPEED_FOLLOW_SYM"))
    fs_adj = follow_stats(diff_rows, "delta_same")
    lines.append(f"SPEED_FOLLOW_ADJ model={model} beta={fmt_num(fs_adj.get('beta'), 3)} "
                 f"ci={fmt_ci(fs_adj.get('ci'), 3)} beta0={fmt_num(fs_adj.get('beta0'), 3)} n={fs_adj['n']}")

    succ = {}
    for group in GROUPS + ("all",):
        rows = [r for r in srows if group == "all" or r["kind"] == group]
        succ[group] = success_stats(rows)
        lines.append(success_line(model, group, succ[group]))
    slow_rows = [r for r in srows if r["kind"] == "slow"]
    lines.append(f"SPEED_NEWWIN_LATE model={model} group=slow "
                 f"w_new_after_MN={sum(r['w_new_after_MN'] for r in slow_rows)}/{len(slow_rows)}")
    late = {}
    for group in GROUPS + ("same",):
        late[group] = late_stats([dict(press=r["press"], M=by_ep[r["ep"]]["M"], N=by_ep[r["ep"]]["N"],
                                       timeout_off=r.get("timeout_off"))
                                  for r in R_ if r["status"] == "ok" and r["kind"] == group])
        lines.append(late_line(model, group, late[group]))
    lines.append(f"SPEED_SAME model={model} true_speed={sum(r['true'] for r in same_rows)}/{len(same_rows)} "
                 f"base_true={sum(r['base_true'] for r in same_rows)}/{len(same_rows)} "
                 f"nopress={sum(r['nopress'] for r in same_rows)}")
    lines.append(f"SC_INVALID model={model} total={sum(invalid.values())} "
                 + " ".join(f"{k}={v}" for k, v in sorted(invalid.items())))

    # 分格
    cells = {}
    for d in diff_rows + [d for d in drows if d["kind"] == "same"]:
        cells.setdefault(f"{d['M']}->{d['Mp']}", []).append(d)
    by_cell = {k: {"dstar_by_N": sorted({(x["N"], x["dstar"]) for x in v}), **mean_stats(v, "delta"),
                   "by_N": {str(n): mean_stats([x for x in v if x["N"] == n], "delta")
                            for n in sorted({x["N"] for x in v})}} for k, v in sorted(cells.items())}
    succ_cells = {}
    for r in srows:
        succ_cells.setdefault(f"{r['M']}->{r['Mp']}|N={r['N']}", []).append(r)
    succ_by_cell = {k: success_stats(v) for k, v in sorted(succ_cells.items())}
    rates = {}
    for name, rs in (("N", list(N_.values())), ("S", list(S_.values())),
                     *[(k, [r for r in R_ if r["kind"] == k]) for k in ("same", "fast", "slow")]):
        ok = [r for r in rs if r["status"] == "ok"]
        rates[name] = {"pressed": sum(r["press"] is not None for r in ok), "valid": len(ok),
                       "success": sum(bool(r["success"]) for r in ok), "planned": len(rs)}
    detail = {"pool_mismatch": mism, "unresolved": unresolved, "label_bad": label_bad, "scope": scope_detail,
              "invalid": dict(invalid), "follow": fs, "follow_adj": fs_adj, "follow_sym": fs_sym,
              "same_delta_sym": same_sym, "late": late,
              "r_timeout_off": {f"{r['recv']}:s{r['slot']}": bool(r.get("timeout_off")) for r in R_ if r["status"] == "ok"}, "same_delta": same, "by_cell": by_cell,
              "success": succ, "success_by_cell": succ_by_cell, "press_rates": rates, "delta_rows": drows,
              "success_rows": srows, "same_rows": same_rows,
              "windows": {f"{C.episode_key(k[0])}:s{k[1]}": (list(v) if v else None) for k, v in win.items()}}
    return lines, detail


def write_md(path, model, lines, d):
    md = [f"# StopCube motion 速度替换读数（{model}）", "", "## 判定行", ""] + [f"- `{l}`" for l in lines] + [""]
    md += ["## 各条件按下率与官方成功", "", "| 条件 | 计划 | 有效 | 按下 | 成功 |", "|---|---|---|---|---|"]
    for k, v in d["press_rates"].items():
        md.append(f"| {k} | {v['planned']} | {v['valid']} | {v['pressed']} | {v['success']} |")
    md += ["", "## Δ 按 (M→M′) 格（Δ = p_R − p_N，两者都按下）", "", "| 格 | Δ*（按 N） | n | 平均 Δ | 95% 区间 |",
           "|---|---|---|---|---|"]
    for k, v in d["by_cell"].items():
        ds = ", ".join(f"N{n}:{x:+d}" for n, x in v["dstar_by_N"])
        md.append(f"| {k} | {ds} | {v['n']} | {fmt_num(v.get('mean'), 1)} | {fmt_ci(v.get('ci'), 1)} |")
    md += ["", "## 新速度成功按 (M→M′, N) 分格", "",
           "| 格 | 新速度成功 | 基线 | 真实速度成功 | 都不 | 未按 | 重叠 |", "|---|---|---|---|---|---|---|"]
    for k, st in d["success_by_cell"].items():
        md.append(f"| {k} | {st['new']}/{st['n']} | {st['base']}/{st['n']} | {st['true']}/{st['n_all']} | "
                  f"{st['neither']} | {st['nopress']} | {st['overlap']} |")
    md += ["", "## 无效计数", ""] + [f"- {k}: {v}" for k, v in sorted(d["invalid"].items())] + [""]
    Path(path).write_text("\n".join(md) + "\n")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def readout(root, out=None):
    root = Path(root)
    plan, by_ep, runs = collect(root)
    lines, detail = analyze(plan, by_ep, runs)
    out = Path(out) if out else root / "readout"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model": plan["model"], "smoke": plan.get("smoke"), "lines": lines, "detail": _jsonable(detail)}
    (out / "readout.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    write_md(out / "readout.md", plan["model"], lines, detail)
    for line in lines:
        print(line, flush=True)
    return 0 if all("=FAIL" not in l for l in lines) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="<O>/<M>")
    parser.add_argument("--smoke", action="store_true", help="读 <O>/<M>/smoke")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    root, _ = C.layout(args.root, args.smoke)
    return readout(root, args.out)


if __name__ == "__main__":
    sys.exit(main())
