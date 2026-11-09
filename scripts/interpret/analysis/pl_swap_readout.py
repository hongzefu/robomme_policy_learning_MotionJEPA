#!/usr/bin/env python3
"""PatternLock 单步 motion token 交换的读数（计划 1108-patternlock-single-step-motion-swap-plan.html 第一部分四、五节）。

入口：python -m scripts.interpret.analysis.pl_swap_readout --root <产物根 O> [--out <目录>] [--seeds 42,1]
读 <O>/plan.json（pl_swap_prep specs 写出）列出的每局：
  终局    <run_root>/accepted/PatternLock:<id>.<nonce8>.terminal.json（多份取最新，成功按 outcome）
  局记录  <attn_out>/<nonce>/episode.json 的 motion_content（mode、donor、n_windows、n_replaced、replace_log）
  轨迹    <run_root>/attempts/*/client/traces/<nonce>/step_trace.jsonl，event=step 行的 tcp_after
  录制    <O>/record/PatternLock_ep<id>.npz（seed 42 N 组），供 SWAP_SCOPE 次数与余弦距离
到达格：轨迹中杆尖水平位置与某个非起点按钮中心距离第一次 < 0.025 m 的那一格；分 own / donor_abs / other / none，
donor_abs 与 own 同格时记 own（A 组与 N、Z 天然如此）。局级重采样 2000 次（种子 20261108）给 95% 区间，只作参考。
数据不全时照样运行：判定行按已有数报，未齐标 PARTIAL；读到明显矛盾标 FAIL。只用 CPU。

判定行（名称照抄计划第五节）：
  RECORD_REPRO=PASS|PARTIAL|FAIL eps=<n> success=<k>
  SEED_EFFECT=PASS|PARTIAL|FAIL seeds=<n> eps_differ=<n>
  SWAP_SCOPE=PASS|PARTIAL|FAIL runs=<n> log_all_00=<0|1> count_mismatch=<n>
  PL_SWAP=PASS|PARTIAL|FAIL runs=<n> invalid=<n> N=<s/n> Z=<s/n> A=<s/n> B_donor=<k/n> C_own=<k/n> C_donor=<k/n>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

TASK = "PatternLock"
GRID = 3
GOAL_THRESH = 0.025          # PatternLock.goal_thresh
SEED_EPS = 1e-6              # SEED_EFFECT：max|Δ| 超过即算不同
BOOT_N, BOOT_SEED = 2000, 20261108
RECORD_SEED = 42
GROUPS = ("N", "Z", "A", "B", "C")
CATS = ("own", "donor_abs", "other", "none")
FULL = {"N": 40, "Z": 40, "A": 10, "B": 15, "C": 25}
FULL_TOTAL = sum(FULL.values())
FULL_SCOPE = FULL_TOTAL - FULL["N"]
FULL_RECORD = 8


# ---------------------------------------------------------------- 几何
def button_xy(cell):
    """easy 按钮中心：x = −0.1 + (行 − 1)·0.1，y = (列 − 1)·0.1，格号 = 行·3 + 列。"""
    r, c = divmod(int(cell), GRID)
    return (-0.1 + (r - 1) * 0.1, (c - 1) * 0.1)


def first_reached(tcps, start_cell, thresh=GOAL_THRESH):
    """tcps: 依序 [x,y,(z)]。返回第一次水平距离 < thresh 的非起点格（同步多格取最近），没有返回 None。"""
    centers = {cell: button_xy(cell) for cell in range(GRID * GRID) if cell != int(start_cell)}
    for tcp in tcps:
        x, y = float(tcp[0]), float(tcp[1])
        hits = [(math.hypot(x - cx, y - cy), cell) for cell, (cx, cy) in centers.items()]
        hits = [h for h in hits if h[0] < thresh]
        if hits:
            return min(hits)[1]
    return None


def classify(cell, own, donor_abs=None):
    if cell is None:
        return "none"
    if cell == own:
        return "own"
    if donor_abs is not None and cell == donor_abs:
        return "donor_abs"
    return "other"


# ---------------------------------------------------------------- 读产物
def read_json(path):
    return json.loads(Path(path).read_text())


def read_trace(path):
    tcps = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") == "step" and row.get("tcp_after") is not None:
                tcps.append([float(v) for v in row["tcp_after"]])
    return tcps


def attempts_for(run_root, key):
    """该局在 run_root 下的尝试目录数（按 commands.json 的 episode_keys）。"""
    n = 0
    for path in sorted(Path(run_root).glob("attempts/*/commands.json")):
        try:
            if key in read_json(path).get("episode_keys", []):
                n += 1
        except (OSError, ValueError):
            continue
    return n


def load_run(row):
    """返回一局的读数 dict；status ∈ ok / missing（未跑）/ invalid（跑了但读不全）。"""
    key = f"{TASK}:{int(row['ep'])}"
    run_root, attn_out = Path(row["run_root"]), Path(row["attn_out"])
    out = {k: row[k] for k in ("stage", "worker", "group", "seed", "ep", "donor")}
    out["attempts"] = attempts_for(run_root, key)
    terminals = sorted(run_root.glob(f"accepted/{key}.*.terminal.json"), key=lambda p: p.stat().st_mtime)
    if not terminals:
        out["status"] = "invalid" if out["attempts"] else "missing"
        out["reason"] = "no_terminal"
        return out
    terminal = read_json(terminals[-1])
    nonce = terminal["episode_nonce"]
    out.update(nonce=nonce, outcome=terminal.get("outcome"), success=int(terminal.get("outcome") == "success"),
               terminals=len(terminals))
    episode_path = attn_out / nonce / "episode.json"
    out["motion_content"] = read_json(episode_path).get("motion_content") if episode_path.is_file() else None
    traces = sorted(run_root.glob(f"attempts/*/client/traces/{nonce}/step_trace.jsonl"))
    if not traces:
        out["status"], out["reason"] = "invalid", "no_trace"
        return out
    tcps = read_trace(traces[0])
    path = row["path"]
    donor_target = row["donor_path"][1] if row.get("donor_path") else None
    cell = first_reached(tcps, path[0])
    out.update(trace=str(traces[0]), steps=len(tcps), tcp_after=tcps, reached=cell,
               category=classify(cell, path[1], donor_target), status="ok")
    if row["group"] != "N" and out["motion_content"] is None:
        out["status"], out["reason"] = "invalid", "no_episode_json"
    return out


def record_start0(record_dir, ep):
    """录制 npz 中 start_frame==0 的 (emb, count)；缺文件或无此行返回 None。"""
    path = Path(record_dir) / f"{TASK}_ep{int(ep)}.npz"
    if not path.is_file():
        return None
    with np.load(path) as z:
        starts = np.asarray(z["start_frames"]).reshape(-1)
        idx = np.flatnonzero(starts == 0)
        if idx.size != 1:
            return None
        i = int(idx[0])
        counts = np.asarray(z["counts"]).reshape(-1) if "counts" in z.files else np.ones_like(starts)
        return np.asarray(z["embs"], np.float64)[i], int(counts[i])


def cosine_distance(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(1.0 - np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------- 统计与判定
def bootstrap_ci(flags, n=BOOT_N, seed=BOOT_SEED):
    flags = np.asarray(flags, np.float64)
    if flags.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = flags[rng.integers(0, flags.size, size=(n, flags.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def sequences_differ(a, b, eps=SEED_EPS):
    if len(a) != len(b):
        return True
    if not a:
        return False
    return bool(np.max(np.abs(np.asarray(a) - np.asarray(b))) > eps)


def judge_record(runs):
    rows = [r for r in runs if r["group"] == "N" and r["seed"] == RECORD_SEED and r["status"] == "ok"]
    success = sum(r["success"] for r in rows)
    if len(rows) == FULL_RECORD and success == FULL_RECORD:
        verdict = "PASS"
    elif success < len(rows):
        verdict = "FAIL"
    else:
        verdict = "PARTIAL"
    return f"RECORD_REPRO={verdict} eps={len(rows)} success={success}", {"eps": len(rows), "success": success}


def judge_seed(runs, seeds_wanted):
    by_ep = {}
    for r in runs:
        if r["group"] == "N" and r["status"] == "ok":
            by_ep.setdefault(r["ep"], {})[r["seed"]] = r["tcp_after"]
    seeds_seen = sorted({s for d in by_ep.values() for s in d})
    differ, compared = [], []
    for ep, seqs in sorted(by_ep.items()):
        if len(seqs) < 2:
            continue
        compared.append(ep)
        items = list(seqs.values())
        if any(sequences_differ(items[0], other) for other in items[1:]):
            differ.append(ep)
    if differ:
        verdict = "PASS"
    elif not compared:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    line = f"SEED_EFFECT={verdict} seeds={len(seeds_seen)} eps_differ={len(differ)}"
    return line, {"seeds": seeds_seen, "eps_compared": compared, "eps_differ": differ,
                  "seeds_wanted": list(seeds_wanted)}


def judge_scope(runs, record_dir):
    details, all00, mismatch, bad = [], True, 0, 0
    for r in runs:
        if r["group"] == "N" or r.get("motion_content") is None:
            continue
        mc = r["motion_content"]
        log = [list(x) for x in mc.get("replace_log", [])]
        log00 = bool(log) and all(x == [0, 0] for x in log)
        rec = record_start0(record_dir, r["ep"])
        expected = rec[1] if rec is not None else None
        count_ok = expected is not None and int(mc.get("n_replaced", -1)) == expected
        donor_expected = f"{TASK}:{int(r['donor'])}"
        donor_ok = mc.get("mode") == "swap_exact" and mc.get("donor") == donor_expected
        all00 = all00 and log00
        mismatch += 0 if count_ok else 1
        bad += 0 if donor_ok else 1
        details.append({"group": r["group"], "seed": r["seed"], "ep": r["ep"], "log_all_00": log00,
                        "n_replaced": mc.get("n_replaced"), "expected_count": expected, "donor": mc.get("donor"),
                        "donor_ok": donor_ok})
    n = len(details)
    if not (all00 and mismatch == 0 and bad == 0):
        verdict = "FAIL"
    elif n == FULL_SCOPE:
        verdict = "PASS"
    else:
        verdict = "PARTIAL"
    line = f"SWAP_SCOPE={verdict} runs={n} log_all_00={int(all00 and n > 0)} count_mismatch={mismatch}"
    return line, {"runs": n, "count_mismatch": mismatch, "donor_mismatch": bad, "details": details}


def summarize(runs, planned_total):
    groups = {}
    for group in GROUPS:
        rows = [r for r in runs if r["group"] == group and r["status"] == "ok"]
        cats = {c: sum(1 for r in rows if r["category"] == c) for c in CATS}
        groups[group] = {
            "n": len(rows), "success": sum(r["success"] for r in rows), "categories": cats,
            "success_ci95": bootstrap_ci([r["success"] for r in rows]),
            "category_ci95": {c: bootstrap_ci([int(r["category"] == c) for r in rows]) for c in CATS},
        }
    pairs = {}
    for r in runs:
        if r["status"] != "ok" or r["group"] == "N":
            continue
        key = f"{r['group']}:{r['ep']}<-{r['donor']}"
        entry = pairs.setdefault(key, {"group": r["group"], "ep": r["ep"], "donor": r["donor"], "seeds": {},
                                       "categories": {c: 0 for c in CATS}, "success": 0})
        entry["seeds"][str(r["seed"])] = {"category": r["category"], "reached": r["reached"], "success": r["success"]}
        entry["categories"][r["category"]] += 1
        entry["success"] += r["success"]
    valid = sum(1 for r in runs if r["status"] == "ok")
    invalid_reads = sum(1 for r in runs if r["status"] == "invalid")
    retried = sum(1 for r in runs if r["status"] == "ok" and r.get("attempts", 1) > 1)
    invalid = invalid_reads + retried
    g = groups
    if invalid_reads:
        verdict = "FAIL"
    elif valid == FULL_TOTAL and planned_total == FULL_TOTAL:
        verdict = "PASS"
    else:
        verdict = "PARTIAL"
    line = (f"PL_SWAP={verdict} runs={valid} invalid={invalid} N={g['N']['success']}/{g['N']['n']} "
            f"Z={g['Z']['success']}/{g['Z']['n']} A={g['A']['success']}/{g['A']['n']} "
            f"B_donor={g['B']['categories']['donor_abs']}/{g['B']['n']} "
            f"C_own={g['C']['categories']['own']}/{g['C']['n']} C_donor={g['C']['categories']['donor_abs']}/{g['C']['n']}")
    return line, {"groups": groups, "pairs": pairs, "valid": valid, "invalid_reads": invalid_reads,
                  "retried": retried, "planned": planned_total}


def cosine_table(plan_rows, record_dir):
    out = {}
    for row in plan_rows:
        if row["group"] not in ("A", "B", "C"):
            continue
        key = f"{row['group']}:{row['ep']}<-{row['donor']}"
        if key in out:
            continue
        a, b = record_start0(record_dir, row["ep"]), record_start0(record_dir, row["donor"])
        out[key] = cosine_distance(a[0], b[0]) if a is not None and b is not None else None
    return out


def write_md(path, lines, summary, cos):
    text = ["# PatternLock 单步 motion token 交换读数", "", "## 判定行", "", "```"] + lines + ["```", ""]
    text += ["## 各组（局数 / 有效局数；括号内为局级重采样 95% 区间，只作参考）", "",
             "| 组 | 有效局 | 成功 | own | donor_abs | other | none |", "|---|---|---|---|---|---|---|"]
    for group in GROUPS:
        g = summary["groups"][group]
        def cell(k, v, ci):
            return f"{v}/{g['n']}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
        text.append(f"| {group} | {g['n']} | {cell('s', g['success'], g['success_ci95'])} | "
                    + " | ".join(cell(c, g["categories"][c], g["category_ci95"][c]) for c in CATS) + " |")
    text += ["", "## 逐对（5 个 seed 汇总）", "", "| 对 | 余弦距离 | 成功 | own | donor_abs | other | none | 各 seed 到达格 |",
             "|---|---|---|---|---|---|---|---|"]
    for key, entry in sorted(summary["pairs"].items()):
        n = len(entry["seeds"])
        d = cos.get(key)
        reached = ", ".join(f"s{s}:{v['reached']}" for s, v in sorted(entry["seeds"].items(), key=lambda kv: int(kv[0])))
        text.append(f"| {key} | {'—' if d is None else f'{d:.4f}'} | {entry['success']}/{n} | "
                    + " | ".join(str(entry["categories"][c]) for c in CATS) + f" | {reached} |")
    Path(path).write_text("\n".join(text) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None, help="输出目录，缺省 <root>/readout")
    parser.add_argument("--seeds", default=None, help="只读这些 seed，如 42,1（中间检查）")
    args = parser.parse_args(argv)
    root = Path(args.root)
    plan = read_json(root / "plan.json")
    record_dir = Path(plan.get("record_dir", root / "record"))
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(plan.get("seeds", []))
    rows = [r for r in plan["runs"] if not args.seeds or int(r["seed"]) in seeds]
    runs = [load_run(r) for r in rows]
    record_line, record = judge_record(runs)
    seed_line, seed_info = judge_seed(runs, seeds)
    scope_line, scope = judge_scope(runs, record_dir)
    swap_line, summary = summarize(runs, len(rows))
    cos = cosine_table(rows, record_dir)
    lines = [record_line, seed_line, scope_line, swap_line]
    out = Path(args.out) if args.out else root / "readout"
    out.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in r.items() if k != "tcp_after"} for r in runs]
    (out / "readout.json").write_text(json.dumps(
        {"lines": lines, "seeds": seeds, "record_repro": record, "seed_effect": seed_info, "swap_scope": scope,
         "summary": summary, "cosine_distance": cos, "runs": slim,
         "missing": [f"{r['group']}:seed{r['seed']}:ep{r['ep']}" for r in runs if r["status"] == "missing"]},
        ensure_ascii=False, indent=2) + "\n")
    write_md(out / "readout.md", lines, summary, cos)
    for line in lines:
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
