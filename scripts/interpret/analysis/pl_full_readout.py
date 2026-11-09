#!/usr/bin/env python3
"""PatternLock 单步 motion token 交换全因子全量的读数（计划 1109-patternlock-swap-full-factorial-wan-plan.html 第一部分第四节）。

两种入口（只用 CPU）：
  --root <O>/<M>                       单模型读数：读 <O>/<M>/plan.json（pl_full_prep specs + donors 写出）列出的每局
  --root <O> --intersect motion,wan    两模型都原样成功的接收局上的并排表
输出写 <root>/readout/（单模型 readout.json、readout.md；交集 intersect.json、intersect.md），可用 --out 改。

每局读法沿用先导轮 pl_swap_readout：终局 <run_root>/accepted/PatternLock:<id>.<nonce8>.terminal.json（多份取最新）、
局记录 <attn_out>/<nonce>/episode.json 的 motion_content、轨迹 attempts/*/client/traces/<nonce>/step_trace.jsonl 的 tcp_after；
到达格 = pl_swap_readout.first_reached（杆尖水平距离第一次 < 0.025 m 的非起点按钮）。
五类归类 classify5：own（自己目标）、donor_dir（供体方向从自己起点推出的格；与 own 重合记 own）、donor_abs（供体的绝对
目标格；与前两类重合时不单记）、other、none。

判定行：
  DEMO_WINDOWS=PASS|PARTIAL|FAIL model=<m> eps=<n> not_one=<k>          n 阶段终局 budget.demo_windows == 1
  SWAP_SCOPE=PASS|PARTIAL|FAIL model=<m> runs=<n> log_all_00=<0|1> count_mismatch=<k>
  MULTI_SWAP_EQ=PASS|PARTIAL|FAIL runs=<n> differ=<k>                   一局一块 vs 一块多局，到达格与成败逐局相同
  PL_SWAP_FULL=PASS|PARTIAL|FAIL model=<m> runs=<n> invalid=<k> N=<a>/80 Z=… Rnd=… A=… B_donor=… C_own=… D_dir=… D_abs=…
    （Wan 门槛触发时：PL_SWAP_FULL=PARTIAL model=wan reason=few_success r_success=<k> groups=N,Z,Rnd …）
另有信息行（不是验收项）：REPRO_RUNS、SEED_REACH。成功率与五类占比只是读数，不设及格线。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from scripts.interpret.analysis.pl_swap_readout import (attempts_for, bootstrap_ci, first_reached, read_json, read_trace,
                                                         record_start0)
from scripts.interpret.pl_swap_prep import cell_rc, direction

TASK = "PatternLock"
GRID = 3
CATS5 = ("own", "donor_dir", "donor_abs", "other", "none")
GROUPS = ("N", "Z", "Rnd", "A", "B", "C", "D")
SWAP_GROUPS = ("Z", "Rnd", "A", "B", "C", "D")
PAIR_GROUPS = ("A", "B", "C", "D")
MAIN_STAGES = ("n", "swap")
SCOPE_STAGES = ("smoke", "multieq", "swap", "seeds")
FULL_N = 80
MAIN_SEED = 42


# ---------------------------------------------------------------- 几何与归类
def step_cell(start, d):
    r, c = cell_rc(start)
    r2, c2 = r + int(d[0]), c + int(d[1])
    return r2 * GRID + c2 if 0 <= r2 < GRID and 0 <= c2 < GRID else None


def targets(path, donor_path):
    """返回 (own, donor_dir, donor_abs)；无供体（N、Z、Rnd）时后两者为 None；donor_dir 出盘为 None。"""
    own = int(path[1])
    if not donor_path:
        return own, None, None
    return own, step_cell(path[0], direction(donor_path)), int(donor_path[1])


def classify5(cell, own, donor_dir=None, donor_abs=None):
    """优先级 own > donor_dir > donor_abs > other；cell 为 None 记 none。"""
    if cell is None:
        return "none"
    if cell == own:
        return "own"
    if donor_dir is not None and cell == donor_dir:
        return "donor_dir"
    if donor_abs is not None and cell == donor_abs:
        return "donor_abs"
    return "other"


# ---------------------------------------------------------------- 读一局
def load_run(row):
    """一局读数；status ∈ ok / missing（未跑）/ invalid（跑了但读不全）。"""
    key = f"{TASK}:{int(row['ep'])}"
    run_root, attn_out = Path(row["run_root"]), Path(row["attn_out"])
    out = {k: row.get(k) for k in ("stage", "worker", "group", "seed", "ep", "donor", "spec", "variant", "pair_index",
                                    "record_dir")}
    out["attempts"] = attempts_for(run_root, key)
    terminals = sorted(run_root.glob(f"accepted/{key}.*.terminal.json"), key=lambda p: p.stat().st_mtime)
    if not terminals:
        out["status"] = "invalid" if out["attempts"] else "missing"
        out["reason"] = "no_terminal"
        return out
    terminal = read_json(terminals[-1])
    nonce = terminal["episode_nonce"]
    out.update(nonce=nonce, outcome=terminal.get("outcome"), success=int(terminal.get("outcome") == "success"),
               terminals=len(terminals), demo_windows=(terminal.get("budget") or {}).get("demo_windows"))
    if row.get("path") is None:   # repro：官方局，只看成败
        out["status"] = "ok"
        return out
    episode_path = attn_out / nonce / "episode.json"
    out["motion_content"] = read_json(episode_path).get("motion_content") if episode_path.is_file() else None
    traces = sorted(run_root.glob(f"attempts/*/client/traces/{nonce}/step_trace.jsonl"))
    if not traces:
        out["status"], out["reason"] = "invalid", "no_trace"
        return out
    tcps = read_trace(traces[0])
    own, ddir, dabs = targets(row["path"], row.get("donor_path"))
    cell = first_reached(tcps, row["path"][0])
    out.update(trace=str(traces[0]), steps=len(tcps), tcp_after=tcps, reached=cell, own=own, donor_dir_cell=ddir,
               donor_abs_cell=dabs, category=classify5(cell, own, ddir, dabs), status="ok",
               recv_start=int(row["path"][0]), recv_dir=list(direction(row["path"])),
               donor_start=int(row["donor_path"][0]) if row.get("donor_path") else None,
               donor_dir=list(direction(row["donor_path"])) if row.get("donor_path") else None)
    if row["group"] != "N" and out["motion_content"] is None:
        out["status"], out["reason"] = "invalid", "no_episode_json"
    return out


# ---------------------------------------------------------------- 判定
def judge_demo_windows(runs, model):
    rows = [r for r in runs if r["stage"] == "n" and r["status"] == "ok"]
    not_one = [r["ep"] for r in rows if r.get("demo_windows") != 1]
    if not_one:
        verdict = "FAIL"
    elif len(rows) == FULL_N:
        verdict = "PASS"
    else:
        verdict = "PARTIAL"
    return (f"DEMO_WINDOWS={verdict} model={model} eps={len(rows)} not_one={len(not_one)}",
            {"eps": len(rows), "not_one": not_one})


def judge_scope(runs, model, planned):
    details, all00, mismatch, bad = [], True, 0, 0
    for r in runs:
        if r["stage"] not in SCOPE_STAGES or r["group"] == "N" or r.get("motion_content") is None:
            continue
        mc = r["motion_content"]
        log = [list(x) for x in mc.get("replace_log", [])]
        log00 = bool(log) and all(x == [0, 0] for x in log)
        rec = record_start0(r["record_dir"], r["ep"])
        expected = rec[1] if rec is not None else None
        count_ok = expected is not None and int(mc.get("n_replaced", -1)) == expected
        donor_ok = mc.get("mode") == "swap_exact" and mc.get("donor") == f"{TASK}:{int(r['donor'])}"
        all00 = all00 and log00
        mismatch += 0 if count_ok else 1
        bad += 0 if donor_ok else 1
        details.append({"stage": r["stage"], "group": r["group"], "seed": r["seed"], "ep": r["ep"], "donor": r["donor"],
                        "log_all_00": log00, "n_replaced": mc.get("n_replaced"), "expected_count": expected,
                        "donor_seen": mc.get("donor"), "donor_ok": donor_ok})
    n = len(details)
    if not (all00 and mismatch == 0 and bad == 0):
        verdict = "FAIL"
    elif n == planned and n > 0:
        verdict = "PASS"
    else:
        verdict = "PARTIAL"
    line = f"SWAP_SCOPE={verdict} model={model} runs={n} log_all_00={int(all00 and n > 0)} count_mismatch={mismatch}"
    return line, {"runs": n, "planned": planned, "count_mismatch": mismatch, "donor_mismatch": bad, "details": details}


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
        same = a.get("reached") == b.get("reached") and a.get("success") == b.get("success")
        rows.append({"pair_index": idx, "ep": a["ep"], "donor": a["donor"], "single": [a.get("reached"), a.get("success")],
                     "multi": [b.get("reached"), b.get("success")], "same": same})
        if not same:
            differ.append(idx)
    if differ:
        verdict = "FAIL"
    elif complete == len(by_pair) and complete > 0:
        verdict = "PASS"
    else:
        verdict = "PARTIAL"
    return f"MULTI_SWAP_EQ={verdict} runs={complete} differ={len(differ)}", {"planned": len(by_pair), "rows": rows}


def repro_info(runs):
    rows = [r for r in runs if r["stage"] == "repro" and r["status"] == "ok"]
    per_ep = {str(r["ep"]): r["success"] for r in sorted(rows, key=lambda r: r["ep"])}
    return f"REPRO_RUNS eps={len(rows)} success={sum(per_ep.values())}", {"per_episode": per_ep}


def seed_reach(runs):
    base = {(r["group"], r["ep"], r["donor"]): r for r in runs
            if r["stage"] == "swap" and r["status"] == "ok" and r["group"] in PAIR_GROUPS}
    pairs = {}
    for r in runs:
        if r["stage"] == "seeds" and r["status"] == "ok":
            pairs.setdefault((r["group"], r["ep"], r["donor"]), {})[r["seed"]] = r.get("reached")
    rows, agree = [], 0
    for key, seeds in sorted(pairs.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if key not in base:
            continue
        seeds = {MAIN_SEED: base[key].get("reached"), **seeds}
        same = len(set(seeds.values())) == 1
        agree += int(same)
        rows.append({"group": key[0], "ep": key[1], "donor": key[2], "reached": {str(s): c for s, c in seeds.items()},
                     "agree": same})
    return f"SEED_REACH pairs={len(rows)} agree={agree}", {"rows": rows}


def group_stats(rows):
    cats = {c: sum(1 for r in rows if r.get("category") == c) for c in CATS5}
    return {"n": len(rows), "success": sum(r["success"] for r in rows), "categories": cats,
            "success_ci95": bootstrap_ci([r["success"] for r in rows]),
            "category_ci95": {c: bootstrap_ci([int(r.get("category") == c) for r in rows]) for c in CATS5}}


def strata(rows):
    """按位置差（供体起点 − 接收起点，(Δ行,Δ列)）、按方向（接收方向→供体方向）、D 按盘内/出盘分层。"""
    by_pos, by_dir, by_stratum = {}, {}, {}
    for r in rows:
        if r.get("donor_start") is None or r["group"] not in PAIR_GROUPS:
            continue
        (r0, c0), (r1, c1) = cell_rc(r["recv_start"]), cell_rc(r["donor_start"])
        pos = f"{r['group']}:dpos=({r1 - r0},{c1 - c0})"
        drc = f"{r['group']}:{tuple(r['recv_dir'])}->{tuple(r['donor_dir'])}"
        by_pos.setdefault(pos, []).append(r)
        by_dir.setdefault(drc, []).append(r)
        if r["group"] == "D":
            by_stratum.setdefault("in" if r.get("donor_dir_cell") is not None else "out", []).append(r)
    fold = lambda d: {k: {"n": len(v), "success": sum(x["success"] for x in v),
                          "categories": {c: sum(1 for x in v if x.get("category") == c) for c in CATS5}}
                      for k, v in sorted(d.items())}
    return {"by_position": fold(by_pos), "by_direction": fold(by_dir), "D_by_stratum": fold(by_stratum)}


def main_rows(runs):
    return [r for r in runs if r["stage"] in MAIN_STAGES and (r["stage"] == "n" or r["seed"] == MAIN_SEED)]


def summarize(runs, plan_rows, model, gate=None):
    mains = main_rows(runs)
    planned = sum(1 for r in plan_rows if r["stage"] in MAIN_STAGES)
    groups = {g: group_stats([r for r in mains if r["group"] == g and r["status"] == "ok"]) for g in GROUPS}
    valid = sum(1 for r in mains if r["status"] == "ok")
    invalid_reads = sum(1 for r in mains if r["status"] == "invalid")
    retried = sum(1 for r in mains if r["status"] == "ok" and r.get("attempts", 1) > 1)
    invalid = invalid_reads + retried
    swap_planned = any(r["stage"] == "swap" for r in plan_rows)
    few = bool(gate and gate.get("few"))
    if invalid_reads:
        verdict = "FAIL"
    elif few:
        verdict = "PARTIAL"
    elif valid == planned and invalid == 0 and swap_planned and groups["N"]["n"] == FULL_N:
        verdict = "PASS"
    else:
        verdict = "PARTIAL"
    g = groups
    frac = lambda grp, cat=None: (f"{g[grp]['success'] if cat is None else g[grp]['categories'][cat]}/{g[grp]['n']}")
    head = f"PL_SWAP_FULL={verdict} model={model}"
    if few:
        head += f" reason=few_success r_success={gate.get('r_success')} groups=N,Z,Rnd"
    line = (f"{head} runs={valid} invalid={invalid} N={g['N']['success']}/{FULL_N} Z={frac('Z')} Rnd={frac('Rnd')} "
            f"A={frac('A')} B_donor={frac('B', 'donor_dir')} C_own={frac('C', 'own')} D_dir={frac('D', 'donor_dir')} "
            f"D_abs={frac('D', 'donor_abs')}")
    return line, {"groups": groups, "strata": strata([r for r in mains if r["status"] == "ok"]), "valid": valid,
                  "invalid_reads": invalid_reads, "retried": retried, "planned": planned, "gate": gate}


# ---------------------------------------------------------------- 输出
def group_table(summary, title):
    text = [f"## {title}（括号内为局级重采样 95% 区间，只作参考）", "",
            "| 组 | 有效局 | 成功 | " + " | ".join(CATS5) + " |", "|---" * (3 + len(CATS5)) + "|"]
    for group in GROUPS:
        g = summary["groups"][group]
        def cell(v, ci):
            return f"{v}/{g['n']}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
        text.append(f"| {group} | {g['n']} | {cell(g['success'], g['success_ci95'])} | "
                    + " | ".join(cell(g["categories"][c], g["category_ci95"][c]) for c in CATS5) + " |")
    return text + [""]


def strata_table(name, table):
    text = [f"## {name}", "", "| 分层 | 局数 | 成功 | " + " | ".join(CATS5) + " |", "|---" * (3 + len(CATS5)) + "|"]
    for key, v in table.items():
        text.append(f"| {key} | {v['n']} | {v['success']} | " + " | ".join(str(v["categories"][c]) for c in CATS5) + " |")
    return text + [""]


def write_md(path, lines, summary, title):
    text = [f"# {title}", "", "## 判定行", "", "```"] + lines + ["```", ""]
    text += group_table(summary, "各组（seed 42）")
    for name, key in (("按位置差（供体起点 − 接收起点）", "by_position"), ("按方向（接收 → 供体）", "by_direction"),
                      ("D 组按供体方向在盘内 / 出盘", "D_by_stratum")):
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
    for name, (line, data) in (("demo_windows", judge_demo_windows(runs, model)),
                               ("swap_scope", judge_scope(runs, model, planned_scope)),
                               ("multieq", judge_multieq(runs)),
                               ("repro", repro_info(runs)),
                               ("seed_reach", seed_reach(runs))):
        lines.append(line)
        info[name] = data
    swap_line, summary = summarize(runs, rows, model, plan.get("gate"))
    lines.append(swap_line)
    out = Path(out) if out else root / "readout"
    out.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in r.items() if k != "tcp_after"} for r in runs]
    (out / "readout.json").write_text(json.dumps(
        {"model": model, "lines": lines, **info, "summary": summary, "runs": slim,
         "missing": [f"{r['stage']}:{r['group']}:seed{r['seed']}:ep{r['ep']}<-{r['donor']}" for r in runs
                     if r["status"] == "missing"]}, ensure_ascii=False, indent=2) + "\n")
    write_md(out / "readout.md", lines, summary, f"PatternLock 全因子交换读数（{model}）")
    return lines, runs, plan


def readout_intersect(root, models, out=None):
    root = Path(root)
    per = {}
    for m in models:
        plan = read_json(root / m / "plan.json")
        runs = [load_run(r) for r in plan["runs"] if r["stage"] in MAIN_STAGES]
        per[m] = main_rows(runs)
    ok_recv = None
    for m in models:
        good = {r["ep"] for r in per[m] if r["stage"] == "n" and r["status"] == "ok" and r["success"] == 1 and r["ep"] < 40}
        ok_recv = good if ok_recv is None else ok_recv & good
    ok_recv = ok_recv or set()
    tables = {}
    for m in models:
        rows = [r for r in per[m] if r["status"] == "ok" and r["ep"] in ok_recv and r["stage"] == "swap"]
        tables[m] = {g: group_stats([r for r in rows if r["group"] == g]) for g in SWAP_GROUPS}
    parts = [f"PL_SWAP_INTERSECT models={','.join(models)} recv={len(ok_recv)}"]
    for m in models:
        t = tables[m]
        parts.append(f"{m}:Z={t['Z']['success']}/{t['Z']['n']},Rnd={t['Rnd']['success']}/{t['Rnd']['n']},"
                     f"A={t['A']['success']}/{t['A']['n']},B_donor={t['B']['categories']['donor_dir']}/{t['B']['n']},"
                     f"C_own={t['C']['categories']['own']}/{t['C']['n']},D_dir={t['D']['categories']['donor_dir']}/{t['D']['n']},"
                     f"D_abs={t['D']['categories']['donor_abs']}/{t['D']['n']}")
    line = " ".join(parts)
    out = Path(out) if out else root / "readout"
    out.mkdir(parents=True, exist_ok=True)
    (out / "intersect.json").write_text(json.dumps({"line": line, "recv": sorted(ok_recv), "tables": tables},
                                                   ensure_ascii=False, indent=2) + "\n")
    text = ["# 两模型都原样成功的接收局上的并排读数", "", "```", line, "```", "",
            f"接收局（{len(ok_recv)} 局）：{sorted(ok_recv)}", "",
            "| 组 | " + " | ".join(f"{m} 局数 | {m} 成功 | {m} own | {m} donor_dir | {m} donor_abs | {m} other | {m} none"
                                 for m in models) + " |",
            "|---" * (1 + 7 * len(models)) + "|"]
    for g in SWAP_GROUPS:
        cells = []
        for m in models:
            s = tables[m][g]
            cells += [str(s["n"]), str(s["success"])] + [str(s["categories"][c]) for c in CATS5]
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
