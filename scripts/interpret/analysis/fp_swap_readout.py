#!/usr/bin/env python3
"""三个交换任务「第一次拿起」motion 交换读数（计划 1109-swap-tasks-first-pick-motion-swap-plan.html 第一部分四、五、七节；
第二部分〇′ F1–F4、F11）。

两种入口（只用 CPU）：
  --root <O>/<M>                       单模型：读 <O>/<M>/plan.json（fp_swap_prep specs + donors 写出）列出的每局
  --root <O> --intersect motion,wan    两模型都是接收局的局上的并排表
输出写 <root>/readout/（readout.json、readout.md；交集 intersect.json、intersect.md），可用 --out 改。

每局归类（classify_pick，按回执 first_pick 与回执交换序列）：correct / swap_blind（只对 moved 接收局有意义，另报）/
other / none。回执、轨迹字段统一经 fp_swap_prep.receipt_fields、trace_row_fields 读。

判定行：
  FIRST_PICK_LABEL=PASS|FAIL model=<m> eps=<n> on_turn=<k> before_buttons=<k> wrong_but_success=<k> correct_but_fail=<k>
        outcome_mismatch=<k>   只在「拿起时子目标已轮到第一次拿起」的 N 局上比 is_target 与 failureflag；
        截断终局 outcome（F4）应为 is_target→success、否则 fail
  FP_REPLAY=PASS|FAIL model=<m> eps=<n> excluded_f3=<k> mismatches=<k>   扫描标签 vs N 阶段回执（swap_times、交换对与起止、
        目标下标、moved），排除 F3 不合格局
  FP_FRAMEMAP=PASS|FAIL model=<m> video=<n> button=<n> bad=<k>   Video：es−1 ≥ 最后交换 end 且差 ≤ 40；Button：逐行
        env_step == es + action_index + 1（视频帧号）
  SWAP_SCOPE=PASS|FAIL model=<m> runs=<n> scope_bad=<k> count_mismatch=<k>   replace_log 每条 [s,s]，起点集合 = plan 行
        w_starts；Zall：起点集合 = 本局编码窗全集且 n_replaced == n_windows
  MULTI_SWAP_EQ=PASS|FAIL runs=<n> differ=<k>          一局一服务端 vs 连跑：拿起物体、拿起步、终局 outcome 相同
  FP_SWAP=PASS|FAIL model=<m> task=<t> runs=<n> invalid=<k> N=a/b Z=… Rnd=… S0=… A=… B=… Zall=… S0_blind=… B_blind=…
        各组 correct/有效局；*_blind 为 moved 接收局上的 swap_blind/moved 数。PASS 只代表无效局为 0。
另有信息行 SEED_FIRST_PICK（seed 1、2 与 seed 42 拿起物体是否相同）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from scripts.interpret.analysis.pl_swap_readout import attempts_for, read_json
from scripts.interpret import fp_swap_prep as F

MAIN_STAGES = ("n", "swap")
SCOPE_STAGES = ("smoke", "multieq", "swap", "seeds")
FRAMEMAP_MAX_GAP = 40
CATS = ("correct", "swap_blind", "other", "none")


# ---------------------------------------------------------------- 单局
def load_run(row, pool_by_key):
    key = F.episode_key(row["task"], row["ep"])
    out = {k: row.get(k) for k in ("stage", "worker", "group", "seed", "task", "ep", "slot", "donor", "donor_src",
                                    "spec", "variant", "pair_index", "w_starts", "pad", "fallback")}
    out["key"] = key
    out["attempts"] = attempts_for(row["run_root"], key)
    run = F.find_run(row["run_root"], row["attn_out"], key)
    if run["terminal"] is None:
        out["status"], out["reason"] = ("invalid" if out["attempts"] else "missing"), "no_terminal"
        return out
    term = run["terminal"]
    out.update(nonce=term.get("episode_nonce"), outcome=term.get("outcome"),
               truncated=term.get("truncated_first_pick"), motion_content=run["motion_content"])
    if term.get("outcome") == "error":
        out["status"], out["reason"] = "invalid", "error_outcome"
        return out
    if run["receipt"] is None or run["trace"] is None:
        out["status"], out["reason"] = "invalid", "no_receipt" if run["trace"] else "no_trace"
        return out
    rf = F.receipt_fields(run["receipt"])
    if not rf["schedule_complete"] or rf["target_index"] is None:
        out["status"], out["reason"] = "invalid", "bad_schedule"
        return out
    rows =[F.trace_row_fields(r) for r in F.read_step_rows(run["trace"])]
    n_obj = len(rows[0]["objects"]) if rows and rows[0]["objects"] else None
    labels = F.episode_labels(row["task"], rf, n_obj)
    pick = F.pick_status(row["task"], rf, labels, rows[0] if rows else None)
    fp = rf["first_pick"]
    out.update(rf=rf, labels=labels, pick=pick, rows=rows, moved=labels["moved"],
               picked=None if fp is None else fp["index"], pick_step=None if fp is None else fp["env_step"],
               category=F.classify_pick(fp, rf["target_index"], labels["swap_blind_index"], labels["moved"]),
               scan=pool_by_key.get(key), status="ok")
    if row["group"] != "N" and run["motion_content"] is None:
        out["status"], out["reason"] = "invalid", "no_episode_json"
    return out


# ---------------------------------------------------------------- 判定
def judge_label(runs, model):
    rows = [r for r in runs if r["stage"] == "n" and r["status"] == "ok"]
    on_turn = before = wbs = cbf = omis = 0
    bad = []
    for r in rows:
        fp = r["rf"]["first_pick"]
        if fp is None:
            continue
        if r["pick"]["before_buttons"]:
            before += 1
            continue
        if not r["pick"]["on_turn"]:
            continue
        on_turn += 1
        if not fp["is_target"] and fp["failureflag"] is False:
            wbs += 1
            bad.append(r["key"])
        if fp["is_target"] and fp["failureflag"] is True:
            cbf += 1
            bad.append(r["key"])
        if (fp["index"] == r["rf"]["target_index"]) != fp["is_target"]:
            bad.append(r["key"])
            omis += 1
            continue
        want = "success" if fp["is_target"] else "fail"
        if (r.get("truncated") and r["outcome"] != want) or (fp.get("outcome") is not None and fp["outcome"] != want):
            omis += 1
            bad.append(r["key"])
    verdict = "FAIL" if (wbs or cbf or omis) else "PASS"
    return (f"FIRST_PICK_LABEL={verdict} model={model} eps={len(rows)} on_turn={on_turn} before_buttons={before} "
            f"wrong_but_success={wbs} correct_but_fail={cbf} outcome_mismatch={omis}"), {"bad": sorted(set(bad))}


def replay_mismatch(scan, rf, moved):
    """扫描标签 vs 回执；返回不符字段列表。"""
    diffs = []
    if scan is None:
        return ["no_scan"]
    if int(scan["swap_times"]) != rf["swap_times"]:
        diffs.append("swap_times")
    if [list(map(int, x)) for x in scan["swap_schedule"]] != rf["schedule"]:
        diffs.append("swap_schedule")
    if int(scan["target_index"]) != rf["target_index"]:
        diffs.append("target_index")
    if bool(scan["moved"]) != bool(moved):
        diffs.append("moved")
    return diffs


def judge_replay(runs, model):
    rows = [r for r in runs if r["stage"] == "n" and r["status"] == "ok"]
    excluded = [r for r in rows if not r["pick"]["f3_ok"]]
    checked = [r for r in rows if r["pick"]["f3_ok"]]
    bad = {}
    for r in checked:
        d = replay_mismatch(r["scan"], r["rf"], r["moved"])
        if d:
            bad[r["key"]] = d
    verdict = "FAIL" if bad else "PASS"
    return (f"FP_REPLAY={verdict} model={model} eps={len(checked)} excluded_f3={len(excluded)} mismatches={len(bad)}",
            {"mismatches": bad, "excluded_f3": [r["key"] for r in excluded]})


def framemap_problem(task, rf, rows):
    """FP_FRAMEMAP 单局：None 为通过，否则原因。"""
    if not rf["schedule"]:
        return "no_schedule"
    if task in F.VIDEO_TASKS:
        last_end = max(int(x[3]) for x in rf["schedule"])
        gap = rf["es"] - 1 - last_end
        return None if 0 <= gap <= FRAMEMAP_MAX_GAP else f"video_gap={gap}"
    es = rf["es"] or 0
    for row in rows:
        if row["action_index"] is None or row["env_step"] is None:
            return "no_action_index"
        if row["env_step"] != es + int(row["action_index"]) + 1:
            return f"button_step{row['env_step']}!=frame{es + int(row['action_index']) + 1}"
    return None


def judge_framemap(runs, model):
    rows = [r for r in runs if r["stage"] in ("smoke", "n") and r["group"] == "N" and r["status"] == "ok"]
    bad = {}
    for r in rows:
        why = framemap_problem(r["task"], r["rf"], r["rows"])
        if why:
            bad[r["key"]] = why
    video = sum(1 for r in rows if r["task"] in F.VIDEO_TASKS)
    verdict = "FAIL" if bad or not rows else "PASS"
    return (f"FP_FRAMEMAP={verdict} model={model} video={video} button={len(rows) - video} bad={len(bad)}",
            {"bad": bad})


def scope_check(r):
    """返回 (scope_ok, count_ok)。"""
    mc = r["motion_content"]
    log = [list(x) for x in mc.get("replace_log", [])]
    starts = {int(x[0]) for x in log}
    same = all(len(x) == 2 and int(x[0]) == int(x[1]) for x in log)
    donor_ok = mc.get("mode") == "swap_exact" and mc.get("donor") == F.episode_key(r["task"], r["donor"])
    n_rep = int(mc.get("n_replaced", -1))
    if r["group"] == "Zall":
        n_win = int(mc.get("n_windows", -2))
        return bool(log) and same and donor_ok and len(starts) == n_win, n_rep == n_win == len(log)
    want = {int(s) for s in (r.get("w_starts") or [])}
    return bool(log) and same and donor_ok and starts == want, n_rep == len(want) == len(log)


def judge_scope(runs, model):
    details, bad, mismatch = [], 0, 0
    for r in runs:
        if r["stage"] not in SCOPE_STAGES or r["group"] == "N" or r["status"] != "ok":
            continue
        scope_ok, count_ok = scope_check(r)
        bad += 0 if scope_ok else 1
        mismatch += 0 if count_ok else 1
        details.append({"stage": r["stage"], "group": r["group"], "key": r["key"], "donor": r["donor"],
                        "scope_ok": scope_ok, "count_ok": count_ok,
                        "starts": sorted({int(x[0]) for x in r["motion_content"].get("replace_log", [])}),
                        "w_starts": r.get("w_starts"), "n_replaced": r["motion_content"].get("n_replaced"),
                        "n_windows": r["motion_content"].get("n_windows")})
    verdict = "FAIL" if (bad or mismatch or not details) else "PASS"
    return (f"SWAP_SCOPE={verdict} model={model} runs={len(details)} scope_bad={bad} count_mismatch={mismatch}",
            {"details": details})


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
        fa = [a.get("picked"), a.get("pick_step"), a.get("outcome")]
        fb = [b.get("picked"), b.get("pick_step"), b.get("outcome")]
        rows.append({"pair_index": idx, "key": a["key"], "single": fa, "multi": fb, "same": fa == fb})
        if fa != fb:
            differ.append(idx)
    verdict = "FAIL" if differ or complete != len(by_pair) or not by_pair else "PASS"
    return f"MULTI_SWAP_EQ={verdict} runs={complete} differ={len(differ)}", {"planned": len(by_pair), "rows": rows}


def seed_first_pick(runs):
    base = {(r["group"], r["key"], r.get("slot")): r for r in runs
            if r["stage"] in MAIN_STAGES and r["status"] == "ok"}
    pairs = {}
    for r in runs:
        if r["stage"] == "seeds" and r["status"] == "ok":
            pairs.setdefault((r["group"], r["key"], r.get("slot")), {})[r["seed"]] = r.get("picked")
    rows, agree = [], 0
    for key, seeds in sorted(pairs.items(), key=lambda kv: str(kv[0])):
        if key not in base:
            continue
        seeds = {F.MAIN_SEED: base[key].get("picked"), **seeds}
        same = len(set(seeds.values())) == 1
        agree += int(same)
        rows.append({"group": key[0], "key": key[1], "slot": key[2], "picked": {str(s): v for s, v in seeds.items()},
                     "agree": same})
    return f"SEED_FIRST_PICK pairs={len(rows)} agree={agree}", {"rows": rows}


# ---------------------------------------------------------------- 汇总
def group_stats(rows):
    """rows：同组有效局。返回 correct、总数、moved 局 swap_blind 数与 moved 数、各类计数。"""
    cats = {c: sum(1 for r in rows if r["category"] == c) for c in CATS}
    moved = [r for r in rows if r["moved"]]
    return {"n": len(rows), "correct": cats["correct"], "cats": cats, "moved": len(moved),
            "blind": sum(1 for r in moved if r["category"] == "swap_blind")}


def task_summary(runs, task, recv=None):
    """一任务：N（全部 N 局）与各交换组（seed 42 swap 阶段）。recv 给出时只留这些接收局（交集用；N 也只留它们）。"""
    out, invalid = {}, 0
    for group in F.GROUPS:
        stage = "n" if group == "N" else "swap"
        mine = [r for r in runs if r["task"] == task and r["stage"] == stage and r["group"] == group]
        if recv is not None:
            mine = [r for r in mine if r["key"] in recv]
        invalid += sum(1 for r in mine if r["status"] != "ok")
        out[group] = group_stats([r for r in mine if r["status"] == "ok"])
    planned = sum(1 for r in runs if r["task"] == task and r["stage"] in MAIN_STAGES
                  and (recv is None or r["key"] in recv))
    return {"groups": out, "invalid": invalid, "runs": planned}


def frac(stats, key="correct", den="n"):
    return f"{stats[key]}/{stats[den]}"


def swap_line(model, task, s, gate=None):
    g = s["groups"]
    verdict = "PASS" if s["invalid"] == 0 else "FAIL"
    line = (f"FP_SWAP={verdict} model={model} task={task} runs={s['runs']} invalid={s['invalid']} "
            + " ".join(f"{k}={frac(g[k])}" for k in F.GROUPS)
            + f" S0_blind={frac(g['S0'], 'blind', 'moved')} B_blind={frac(g['B'], 'blind', 'moved')}")
    if gate and gate.get("few"):
        line += f" gate=few recv={gate['recv']} groups={','.join(gate['groups'])}"
    return line


def collect(root):
    plan = read_json(Path(root) / "plan.json")
    pool, pool_by_key = F.load_pool(Path(root).parent)
    return plan, [load_run(row, pool_by_key) for row in plan["runs"]]


def _strip(r):
    return {k: v for k, v in r.items() if k not in ("rows", "rf")}


def readout_model(root, out=None):
    root = Path(root)
    plan, runs = collect(root)
    model = plan["model"]
    lines, extra = [], {}
    for name, fn in (("label", judge_label), ("replay", judge_replay), ("framemap", judge_framemap),
                     ("scope", judge_scope)):
        line, detail = fn(runs, model)
        lines.append(line)
        extra[name] = detail
    line, extra["multieq"] = judge_multieq(runs)
    lines.append(line)
    line, extra["seeds"] = seed_first_pick(runs)
    lines.append(line)
    gates = (plan.get("gate") or {}).get("tasks") or {}
    summaries = {}
    for task in F.TASKS:
        summaries[task] = task_summary(runs, task)
        lines.append(swap_line(model, task, summaries[task], gates.get(task)))
    out = Path(out) if out else root / "readout"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "lines": lines, "summaries": summaries, "details": extra,
               "runs": [_strip(r) for r in runs]}
    (out / "readout.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    write_md(out / "readout.md", f"第一次拿起交换读数（{model}）", lines, summaries)
    for line in lines:
        print(line, flush=True)
    ok = all("=FAIL" not in l for l in lines)
    return 0 if ok else 1


def readout_intersect(root, models, out=None):
    root = Path(root)
    per = {m: collect(root / m) for m in models}
    recv = set.intersection(*[set(plan.get("recv") or []) for plan, _ in per.values()])
    lines, summaries = [], {}
    for task in F.TASKS:
        mine = {k for k in recv if F.split_key(k)[0] == task}
        parts = []
        for m in models:
            s = task_summary(per[m][1], task, recv=mine)
            summaries.setdefault(task, {})[m] = s
            g = s["groups"]
            parts.append(f"{m}:" + ",".join(f"{k}={frac(g[k])}" for k in F.SWAP_GROUPS)
                         + f",S0_blind={frac(g['S0'], 'blind', 'moved')},B_blind={frac(g['B'], 'blind', 'moved')}")
        lines.append(f"FP_SWAP_INTERSECT models={','.join(models)} task={task} recv={len(mine)} " + " ".join(parts))
    out = Path(out) if out else root / "readout"
    out.mkdir(parents=True, exist_ok=True)
    (out / "intersect.json").write_text(json.dumps({"models": list(models), "recv": sorted(recv), "lines": lines,
                                                    "summaries": summaries}, ensure_ascii=False, indent=2) + "\n")
    md = [f"# 两模型交集接收局读数（{','.join(models)}）", ""] + [f"- `{l}`" for l in lines]
    (out / "intersect.md").write_text("\n".join(md) + "\n")
    for line in lines:
        print(line, flush=True)
    return 0


def write_md(path, title, lines, summaries):
    md = [f"# {title}", "", "## 判定行", ""] + [f"- `{l}`" for l in lines] + [""]
    for task, s in summaries.items():
        md += [f"## {task}", "", "| 组 | 有效局 | correct | swap_blind | other | none | moved 局 swap_blind |",
               "|---|---|---|---|---|---|---|"]
        for group in F.GROUPS:
            g = s["groups"][group]
            md.append(f"| {group} | {g['n']} | {g['cats']['correct']} | {g['cats']['swap_blind']} | {g['cats']['other']} "
                      f"| {g['cats']['none']} | {g['blind']}/{g['moved']} |")
        md.append("")
    Path(path).write_text("\n".join(md) + "\n")


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
