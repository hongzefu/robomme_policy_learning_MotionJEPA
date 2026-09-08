#!/usr/bin/env python3
"""阶段 2 汇总：合并 32 批（4 条件 × 4 seed × 2 split × 8 worker）的分片结果，做 pooled 400 集配对统计（计划第二部分 §9）。

输入（每 (cond, split, seed) 一格，每格 8 个 worker）：
  v1-store/evaluation/mv-<cond>-s<seed>-<split>-w<k>/ckpt<id>/<seed<S>|val-seed<S>>/{progress.json, videos/*.mp4}
  v1-store/logs/mv-<t|v><seed>-<cond>-w<k>.{eval.log,probe.jsonl}（MV_EP_DONE / EVAL_EPISODE 行；逐 infer 记录）
核对（阻断）：MV_GRID / SPLIT_SEED_MATCH / MV_PAIRING / MV_STRUCT_IDENTICAL（normal/mask/swap 三臂首步 motion_mask·motion_pos·mem_order·motion_cols sha 全等）/ DONOR_CROSS_SEG
统计（主估计 pooled 400 = test+val；两 split 分别作辅助）：
  逐 seed 配对差 mean ± t(0.975,3)·sd/√4；episode bootstrap 10000 次按 split×task 分层、所有条件与 seed 共享索引；
  McNemar b/c 与精确二项 p（math.comb）；等价判定：t 与 bootstrap 区间都落在 ±2pp 内 → EQUIV_2PP；分层 task / difficulty / 启动时空窗 / donor 覆盖档 / steps 档。
判定行：MV_GRID=DONE|INCOMPLETE … / MV_PAIRED pool= pair= cell= dmean= t_ci= boot_ci= verdict= … / MV_SUMMARY=PASS|FAIL
跨卡校验：--xgpu A_RUN B_RUN A_LOGPREFIX B_LOGPREFIX GPU_A GPU_B → MV_XGPU=BITEXACT|OUTCOME_SAME|OUTCOME_DIFF …
用法：uv run --no-sync python scripts/motion-variance/summarize_mv.py --out-dir docs/training-doc/mv-matrix-40k/records [--partial]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import mv_common as C  # noqa: E402

_VIDEO_RE = re.compile(r"^(?P<task>[A-Za-z]+)_ep(?P<ep>\d+)_(?P<flag>[a-z]+)_")
_EPISODE_RE = re.compile(r"EVAL_EPISODE split=(?P<split>\S+) task=(?P<task>\S+) ep=(?P<ep>\d+) env_seed=(?P<seed>\S+) difficulty=(?P<diff>\S+)")
_DONE_RE = re.compile(r"^MV_EP_DONE split=(?P<split>\S+) task=(?P<task>\S+) ep=(?P<ep>\d+) flag=(?P<flag>\S+) steps=(?P<steps>\d+) "
                      r"infers=(?P<infers>\d+) exec_start_idx=(?P<es>\d+) wall_s=(?P<wall>[\d.]+) difficulty=(?P<diff>\S+)")
PAIRS = (("normal", "mask"), ("normal", "swap"), ("mask", "swap"), ("normal", "official"))
CKPT_ID = {"official": 79999, "normal": 39999, "mask": 39999, "swap": 39999}
STRUCT_KEYS = ("motion_mask", "motion_pos_sha", "mem_order_sha", "motion_cols_sha")


def seed_seg(split: str, seed: int) -> str:
    return f"seed{seed}" if split == "test" else f"{split}-seed{seed}"


def log_prefix(cond: str, split: str, seed: int, suffix: str = "") -> str:
    return f"mv-{'t' if split == 'test' else 'v'}{seed}-{cond}{suffix}"


def t_quantile(p: float, df: int) -> float:
    """t 分布分位数（数值积分 + 二分；不引 scipy）。df=3, p=0.975 → 3.182。"""
    def pdf(x):
        return math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2)) * (1 + x * x / df) ** (-(df + 1) / 2)

    def cdf(x):
        xs = np.linspace(0, abs(x), 20001)
        area = np.trapz([pdf(v) for v in xs], xs)
        return 0.5 + area if x >= 0 else 0.5 - area
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_meta(meta_dir: pathlib.Path, split: str, task: str) -> dict[int, dict]:
    p = meta_dir / split / f"record_dataset_{task}_metadata.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {int(r["episode"]): {"seed": r.get("seed"), "difficulty": r.get("difficulty")} for r in payload.get("records", [])}


def collect_cell(eval_root: pathlib.Path, logs_dir: pathlib.Path, cond: str, split: str, seed: int, workers: int,
                 suffix: str = "") -> dict:
    """返回 {"episodes": {(task, ep): rec}, "workers": [...], "missing_workers": [...]}"""
    eps: dict[tuple[str, int], dict] = {}
    winfo = []
    missing = []
    lp = log_prefix(cond, split, seed, suffix)
    for k in range(workers):
        d = eval_root / f"mv-{cond}-s{seed}-{split}{suffix}-w{k}" / f"ckpt{CKPT_ID[cond]}" / seed_seg(split, seed)
        pj = d / "progress.json"
        if not pj.is_file():
            missing.append(k)
            continue
        prog = json.loads(pj.read_text(encoding="utf-8"))
        flags = {}
        for mp4 in (d / "videos").glob("*.mp4") if (d / "videos").is_dir() else []:
            m = _VIDEO_RE.match(mp4.name)
            if m:
                flags[(m.group("task"), int(m.group("ep")))] = m.group("flag")
        done, logged = {}, {}
        el = logs_dir / f"{lp}-w{k}.eval.log"
        if el.is_file():
            for line in open(el, encoding="utf-8", errors="replace"):
                m = _DONE_RE.match(line.strip())
                if m:
                    done[(m.group("task"), int(m.group("ep")))] = {"flag": m.group("flag"), "steps": int(m.group("steps")),
                                                                    "infers": int(m.group("infers")), "es": int(m.group("es")),
                                                                    "wall_s": float(m.group("wall")), "difficulty_logged": m.group("diff")}
                m = _EPISODE_RE.search(line)
                if m:
                    logged[(m.group("task"), int(m.group("ep")))] = {"split": m.group("split"), "env_seed": m.group("seed"), "difficulty": m.group("diff")}
        first_infer: dict[tuple[str, int], dict] = {}
        cover: dict[tuple[str, int], dict] = {}
        n_infer: dict[tuple[str, int], int] = {}
        pf = logs_dir / f"{lp}-w{k}.probe.jsonl"
        if pf.is_file():
            for line in open(pf, encoding="utf-8", errors="replace"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r["task"], int(r["ep"]))
                n_infer[key] = n_infer.get(key, 0) + 1
                if key not in first_infer:
                    first_infer[key] = {sk: r.get(sk) for sk in STRUCT_KEYS} | {"k": int(r["k"]), "motion_emb_sha": r.get("motion_emb_sha"), "donor": r.get("donor")}
                cv = cover.setdefault(key, {"exact": 0, "fallback": 0, "cycle": 0, "cross_seg": 0})
                for h in cv:
                    cv[h] += int(r.get("cover", {}).get(h, 0))
        n_cell = 0
        for task, d_ in prog.items():
            for e, v in d_.items():
                key = (task, int(e))
                rec = {"cond": cond, "split": split, "seed": seed, "task": task, "ep": int(e), "success": v, "worker": k,
                       "flag": flags.get(key, "?")}
                rec.update(done.get(key, {}))
                rec["logged"] = logged.get(key)
                rec["first"] = first_infer.get(key)
                rec["cover"] = cover.get(key)
                rec["n_infer_probe"] = n_infer.get(key)
                eps[key] = rec
                n_cell += 1
        winfo.append({"worker": k, "episodes": n_cell, "has_log": el.is_file(), "has_probe": pf.is_file()})
    return {"episodes": eps, "workers": winfo, "missing_workers": missing}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", default=str(C.V1 / "evaluation"))
    ap.add_argument("--logs-dir", default=str(C.V1 / "logs"))
    ap.add_argument("--metadata-dir", default=str(C.REPO_ROOT / "third_party/robomme_benchmark/src/robomme/env_metadata"))
    ap.add_argument("--conds", default=",".join(C.CONDS))
    ap.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    ap.add_argument("--splits", default=",".join(C.SPLITS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--partial", action="store_true", help="只对已完整的 seed 做统计（seed-major 部分完成时用）")
    ap.add_argument("--xgpu", nargs=6, metavar=("A_RUN", "B_RUN", "A_LOGPREFIX", "B_LOGPREFIX", "GPU_A", "GPU_B"),
                    help="跨卡校验：比两次同条件同 seed 的 progress.json 成败与逐 infer act_sha")
    args = ap.parse_args()
    eval_root, logs_dir, meta_dir = pathlib.Path(args.eval_root), pathlib.Path(args.logs_dir), pathlib.Path(args.metadata_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    V = C.Verdict()

    if args.xgpu:
        return xgpu(args, eval_root, logs_dir, out_dir, V)

    conds = args.conds.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    splits = args.splits.split(",")
    meta = {(sp, t): load_meta(meta_dir, sp, t) for sp in splits for t in C.TASKS}
    cells: dict[tuple[str, str, int], dict] = {}
    for cond in conds:
        for sp in splits:
            for sd in seeds:
                cells[(cond, sp, sd)] = collect_cell(eval_root, logs_dir, cond, sp, sd, args.workers)
    N_PER = C.EPISODES_PER_TASK * len(C.TASKS)
    # ── MV_GRID ────────────────────────────────────────────────────────────────
    n_cells_ok = 0
    n_eps = n_err = n_missing = 0
    cell_ok: dict[tuple[str, str, int], bool] = {}
    for key, cell in cells.items():
        eps = cell["episodes"]
        errs = sum(1 for r in eps.values() if r["success"] == "error")
        missing = N_PER - len(eps)
        ok = len(eps) == N_PER and errs == 0 and not cell["missing_workers"]
        cell_ok[key] = ok
        n_cells_ok += int(ok)
        n_eps += len(eps); n_err += errs; n_missing += max(missing, 0)
    grid_ok = n_cells_ok == len(cells)
    V.block(grid_ok or args.partial, f"MV_GRID={'DONE' if grid_ok else 'INCOMPLETE'} cells={n_cells_ok}/{len(cells)} episodes={n_eps}/{N_PER * len(cells)} "
                                     f"errors={n_err} missing={n_missing} workers={args.workers}")
    complete_seeds = [sd for sd in seeds if all(cell_ok[(c, sp, sd)] for c in conds for sp in splits)]
    if args.partial:
        V.observe(f"MV_PARTIAL complete_seeds={complete_seeds} of {seeds}")
    # ── SPLIT_SEED_MATCH ───────────────────────────────────────────────────────
    for sp in splits:
        n_logged = n_mis = n_unlogged = 0
        for (cond, sp_, sd), cell in cells.items():
            if sp_ != sp:
                continue
            for (task, ep), r in cell["episodes"].items():
                lg = r.get("logged")
                if lg is None:
                    n_unlogged += 1
                    continue
                n_logged += 1
                want = meta[(sp, task)].get(ep, {}).get("seed")
                if lg["split"] != sp or want is None or str(want) != lg["env_seed"]:
                    n_mis += 1
        st = "SKIP" if n_logged == 0 else ("PASS" if n_mis == 0 else "FAIL")
        V.block(st != "FAIL", f"SPLIT_SEED_MATCH={st} split={sp} n={n_logged} mismatch={n_mis} unlogged={n_unlogged}")
    # ── MV_PAIRING / MV_STRUCT_IDENTICAL / DONOR_COVER_ACTUAL ─────────────────
    keys_all = {(sp, t, e) for sp in splits for t in C.TASKS for e in range(C.EPISODES_PER_TASK)}
    pair_missing = 0
    for sp in splits:
        for t in C.TASKS:
            for e in range(C.EPISODES_PER_TASK):
                for c in conds:
                    for sd in seeds:
                        if (t, e) not in cells[(c, sp, sd)]["episodes"]:
                            pair_missing += 1
    V.block(pair_missing == 0 or args.partial, f"MV_PAIRING={'PASS' if pair_missing == 0 else 'FAIL'} cells={len(cells)} episodes={len(keys_all)} missing_pairs={pair_missing}")
    motion_conds = [c for c in ("normal", "mask", "swap") if c in conds]
    if len(motion_conds) >= 2:
        for sp in splits:
            for sd in seeds:
                n_ep = n_match = n_emb_diff = n_first = 0
                for t in C.TASKS:
                    for e in range(C.EPISODES_PER_TASK):
                        firsts = [cells[(c, sp, sd)]["episodes"].get((t, e), {}).get("first") for c in motion_conds]
                        if any(f is None for f in firsts):
                            continue
                        n_first += 1
                        n_ep += 1
                        same = all(all(f[k_] == firsts[0][k_] for k_ in STRUCT_KEYS) for f in firsts[1:]) and all(f["k"] == firsts[0]["k"] for f in firsts)
                        n_match += int(same)
                        if "swap" in motion_conds and "normal" in motion_conds:
                            fn = cells[("normal", sp, sd)]["episodes"][(t, e)]["first"]
                            fs = cells[("swap", sp, sd)]["episodes"][(t, e)]["first"]
                            n_emb_diff += int(fn["motion_emb_sha"] != fs["motion_emb_sha"] or fn["k"] == 0)
                ok = n_ep > 0 and n_match == n_ep
                V.block(ok or (args.partial and sd not in complete_seeds),
                        f"MV_STRUCT_IDENTICAL={'PASS' if ok else 'FAIL'} split={sp} seed={sd} episodes={n_ep} first_infer_sha_match={n_match}/{n_ep} "
                        f"keys={list(STRUCT_KEYS)} conds={motion_conds} swap_emb_differs_or_k0={n_emb_diff}/{n_ep} "
                        f"note=第二个 infer 起轨迹发散，只核首步；motion_emb 的 sha 三臂必然不同（这正是干预本身）")
    if "swap" in conds:
        tot = {"exact": 0, "fallback": 0, "cycle": 0, "cross_seg": 0}
        by_task = {t: {h: 0 for h in tot} for t in C.TASKS}
        ep_class = {}
        for (cond, sp, sd), cell in cells.items():
            if cond != "swap":
                continue
            for (t, e), r in cell["episodes"].items():
                cv = r.get("cover")
                if not cv:
                    continue
                for h in tot:
                    tot[h] += cv[h]; by_task[t][h] += cv[h]
                cls = "cycle" if cv["cycle"] else ("fallback" if cv["fallback"] else "exact")
                ep_class.setdefault((sp, t, e), {})[sd] = cls
        n = sum(tot.values())
        V.observe(f"DONOR_COVER_ACTUAL windows={n} " + " ".join(f"{h}={tot[h]}/{n}({tot[h] / max(n, 1):.1%})" for h in tot)
                  + " by_task=[" + ", ".join(f"{t}: " + "/".join(f"{h[:2]}={by_task[t][h]}" for h in tot) for t in C.TASKS) + "]"
                  + f" episodes_all_exact={sum(1 for v in ep_class.values() if all(x == 'exact' for x in v.values()))}/{len(ep_class)}")
        V.block(tot["cross_seg"] == 0, f"DONOR_CROSS_SEG={'PASS' if tot['cross_seg'] == 0 else 'FAIL'} cross_seg={tot['cross_seg']}")
    else:
        ep_class = {}

    # ── 统计 ───────────────────────────────────────────────────────────────────
    stat_seeds = complete_seeds if args.partial else seeds
    T_Q = C.T975_DF3 if len(stat_seeds) == 4 else (t_quantile(0.975, len(stat_seeds) - 1) if len(stat_seeds) >= 2 else float("nan"))
    n_pairs = len(PAIRS)
    T_Q_BONF = t_quantile(1 - 0.05 / (2 * n_pairs), len(stat_seeds) - 1) if len(stat_seeds) >= 2 else float("nan")
    ep_keys = sorted(keys_all)
    ep_index = {k: i for i, k in enumerate(ep_keys)}
    # 成败矩阵 succ[cond][seed] -> float[N_ep]（nan = 缺）
    succ: dict[str, dict[int, np.ndarray]] = {}
    for c in conds:
        succ[c] = {}
        for sd in stat_seeds:
            arr = np.full(len(ep_keys), np.nan)
            for sp in splits:
                for (t, e), r in cells[(c, sp, sd)]["episodes"].items():
                    if r["success"] in (True, False):
                        arr[ep_index[(sp, t, e)]] = float(r["success"])
            succ[c][sd] = arr
    # 分层属性（按 episode 固定，不随条件变）
    attrs = {}
    normal_ref = "normal" if "normal" in conds else conds[0]
    steps_ref = {}
    for (sp, t, e) in ep_keys:
        r42 = cells[(normal_ref, sp, stat_seeds[0])]["episodes"].get((t, e), {}) if stat_seeds else {}
        first = r42.get("first") or {}
        attrs[(sp, t, e)] = {"split": sp, "task": t, "difficulty": meta[(sp, t)].get(e, {}).get("difficulty"),
                             "start_k": ("k=0" if first.get("k", None) == 0 else ("k>0" if first.get("k") is not None else "n/a")),
                             "donor": (ep_class.get((sp, t, e), {}).get(stat_seeds[0], "n/a") if stat_seeds else "n/a")}
        steps_ref[(sp, t, e)] = r42.get("steps")
    valid_steps = [v for v in steps_ref.values() if v is not None]
    if valid_steps:
        q = np.percentile(valid_steps, [25, 50, 75])
        for k_, v in steps_ref.items():
            attrs[k_]["steps_bin"] = "n/a" if v is None else ("Q1" if v <= q[0] else "Q2" if v <= q[1] else "Q3" if v <= q[2] else "Q4")
    else:
        for k_ in attrs:
            attrs[k_]["steps_bin"] = "n/a"
    # bootstrap 共享索引：按 split×task 分层，每层 50 集有放回
    rng = np.random.default_rng(C.BOOT_SEED)
    strata = {}
    for i, (sp, t, e) in enumerate(ep_keys):
        strata.setdefault((sp, t), []).append(i)
    boot_idx = {s: rng.integers(0, len(ix), size=(C.BOOT_N, len(ix))) for s, ix in strata.items()}

    def cell_stats(pair, sel_mask: np.ndarray, label: str, pool: str) -> dict | None:
        a, b = pair
        if a not in succ or b not in succ or not stat_seeds:
            return None
        diffs_seed, rates_a, rates_b, mc_b, mc_c, mc_p = [], [], [], [], [], []
        boot = np.zeros(C.BOOT_N)
        n_ep_sel = 0
        for sd in stat_seeds:
            xa, xb = succ[a][sd], succ[b][sd]
            ok = sel_mask & ~np.isnan(xa) & ~np.isnan(xb)
            n_ep_sel = int(ok.sum())
            if n_ep_sel == 0:
                return None
            d = xa[ok] - xb[ok]
            diffs_seed.append(float(d.mean()))
            rates_a.append(float(xa[ok].mean())); rates_b.append(float(xb[ok].mean()))
            bb = int(((xa == 1) & (xb == 0) & ok).sum()); cc = int(((xa == 0) & (xb == 1) & ok).sum())
            mc_b.append(bb); mc_c.append(cc); mc_p.append(mcnemar_p(bb, cc))
            # bootstrap：在被选中的层内重采样（分层 = split×task），只用落在 sel_mask 内的层
            dfull = xa - xb
            acc = np.zeros(C.BOOT_N); cnt = 0
            for s, ix in strata.items():
                ix_arr = np.asarray(ix)
                m = ok[ix_arr]
                if not m.any():
                    continue
                sub = ix_arr[m]
                # 层内重采样索引按 sub 长度重映射（共享随机流：取 boot_idx 的前 len(sub) 列并取模）
                bi = boot_idx[s][:, :len(sub)] % len(sub)
                acc += dfull[sub][bi].sum(axis=1); cnt += len(sub)
            boot += acc / max(cnt, 1)
        boot /= len(stat_seeds)
        dm = float(np.mean(diffs_seed))
        sd_ = float(np.std(diffs_seed, ddof=1)) if len(diffs_seed) > 1 else float("nan")
        half = T_Q * sd_ / math.sqrt(len(diffs_seed)) if len(diffs_seed) > 1 else float("nan")
        half_b = T_Q_BONF * sd_ / math.sqrt(len(diffs_seed)) if len(diffs_seed) > 1 else float("nan")
        lo_b, hi_b = np.percentile(boot, [2.5, 97.5])
        lo_bb, hi_bb = np.percentile(boot, [100 * 0.05 / (2 * n_pairs), 100 * (1 - 0.05 / (2 * n_pairs))])
        t_lo, t_hi = dm - half, dm + half
        pp = 100.0
        in_rope = (t_lo * pp >= -C.ROPE_PP and t_hi * pp <= C.ROPE_PP and lo_b * pp >= -C.ROPE_PP and hi_b * pp <= C.ROPE_PP)
        if in_rope:
            verdict = f"EQUIV_{int(C.ROPE_PP)}PP"
        elif t_lo > 0 and lo_b > 0:
            verdict = "POSITIVE"
        elif t_hi < 0 and hi_b < 0:
            verdict = "NEGATIVE"
        else:
            verdict = "NOT_DETECTED"
        return {"pair": f"{a}-{b}", "pool": pool, "cell": label, "n_seed": len(stat_seeds), "n_ep": n_ep_sel,
                "rate_a": rates_a, "rate_b": rates_b, "diff_seed": diffs_seed, "dmean_pp": dm * pp, "sd_pp": sd_ * pp,
                "t_ci_pp": [t_lo * pp, t_hi * pp], "t_ci_bonf_pp": [(dm - half_b) * pp, (dm + half_b) * pp],
                "boot_ci_pp": [lo_b * pp, hi_b * pp], "boot_ci_bonf_pp": [lo_bb * pp, hi_bb * pp],
                "p_gt0": float((boot > 0).mean()), "mcnemar_b": mc_b, "mcnemar_c": mc_c, "mcnemar_p": mc_p, "verdict": verdict}

    results = []
    lines_stats = []
    pools = [("pool400", np.ones(len(ep_keys), bool))] + [(f"split={sp}", np.array([k_[0] == sp for k_ in ep_keys])) for sp in splits]
    for pool, pmask in pools:
        for pair in PAIRS:
            if any(c not in conds for c in pair):
                continue
            r = cell_stats(pair, pmask, "overall", pool)
            if r:
                results.append(r)
                lines_stats.append(f"MV_PAIRED pool={pool} pair={r['pair']} cell=overall n_ep={r['n_ep']} n_seed={r['n_seed']} "
                                   f"rate_a={np.mean(r['rate_a']):.4f} rate_b={np.mean(r['rate_b']):.4f} dmean={r['dmean_pp']:+.2f}pp "
                                   f"t_ci=[{r['t_ci_pp'][0]:+.2f},{r['t_ci_pp'][1]:+.2f}] boot_ci=[{r['boot_ci_pp'][0]:+.2f},{r['boot_ci_pp'][1]:+.2f}] "
                                   f"bonf_t=[{r['t_ci_bonf_pp'][0]:+.2f},{r['t_ci_bonf_pp'][1]:+.2f}] p_gt0={r['p_gt0']:.3f} "
                                   f"mcnemar_b={sum(r['mcnemar_b'])} c={sum(r['mcnemar_c'])} p_by_seed={[round(x, 4) for x in r['mcnemar_p']]} verdict={r['verdict']}")
            if pool != "pool400":
                continue
            for attr, vals in (("task", C.TASKS), ("difficulty", ("easy", "medium", "hard")), ("start_k", ("k=0", "k>0")),
                               ("donor", ("exact", "fallback", "cycle")), ("steps_bin", ("Q1", "Q2", "Q3", "Q4"))):
                if attr == "donor" and "swap" not in pair:
                    continue
                for v in vals:
                    m = np.array([attrs[k_].get(attr) == v for k_ in ep_keys])
                    if m.sum() == 0:
                        continue
                    r = cell_stats(pair, m, f"{attr}={v}", pool)
                    if r:
                        results.append(r)
                        lines_stats.append(f"MV_PAIRED pool={pool} pair={r['pair']} cell={attr}={v} n_ep={r['n_ep']} dmean={r['dmean_pp']:+.2f}pp "
                                           f"t_ci=[{r['t_ci_pp'][0]:+.2f},{r['t_ci_pp'][1]:+.2f}] boot_ci=[{r['boot_ci_pp'][0]:+.2f},{r['boot_ci_pp'][1]:+.2f}] verdict={r['verdict']}")
    for l in lines_stats:
        V.observe(l)
    # 绝对成功率表
    rate_lines = []
    for c in conds:
        for pool, pmask in pools:
            rs = [float(np.nanmean(succ[c][sd][pmask])) for sd in stat_seeds] if stat_seeds else []
            if rs:
                rate_lines.append(f"MV_RATE cond={c} pool={pool} mean={np.mean(rs):.4f} sd={np.std(rs, ddof=1) if len(rs) > 1 else 0:.4f} by_seed={[round(x, 4) for x in rs]}")
    for l in rate_lines:
        V.observe(l)
    final = V.final("MV_SUMMARY", f"cells={len(cells)} stat_seeds={stat_seeds} pairs={len(PAIRS)} rope_pp={C.ROPE_PP} t_q={T_Q:.3f} t_q_bonf={T_Q_BONF:.3f}")
    lines = V.texts() + [final]
    print("\n".join(lines))
    # 产物
    per_ep = []
    for (c, sp, sd), cell in cells.items():
        for (t, e), r in cell["episodes"].items():
            per_ep.append({"cond": c, "split": sp, "seed": sd, "task": t, "ep": e, "success": r["success"], "flag": r["flag"],
                           "steps": r.get("steps"), "infers": r.get("infers"), "wall_s": r.get("wall_s"), "worker": r["worker"],
                           "difficulty": meta[(sp, t)].get(e, {}).get("difficulty"), "env_seed": meta[(sp, t)].get(e, {}).get("seed"),
                           "first_k": (r.get("first") or {}).get("k"), "donor": (r.get("first") or {}).get("donor"), "cover": r.get("cover"),
                           "attrs": attrs.get((sp, t, e))})
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "per_episode_by_cond.json").write_text(json.dumps({"lines": lines, "episodes": per_ep}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out_dir / "paired_stats.json").write_text(json.dumps({"lines": lines, "results": results, "stat_seeds": stat_seeds, "rope_pp": C.ROPE_PP,
                                                           "t_q": T_Q, "t_q_bonf": T_Q_BONF, "boot_n": C.BOOT_N, "boot_seed": C.BOOT_SEED,
                                                           "pairs": PAIRS, "conds": conds, "splits": splits, "seeds": seeds},
                                                          ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0 if V.all_ok else 1


def xgpu(args, eval_root, logs_dir, out_dir, V) -> int:
    a_run, b_run, a_lp, b_lp, gpu_a, gpu_b = args.xgpu
    def load(run, lp):
        d = eval_root / f"{run}-w0" / "ckpt39999" / "seed42"
        prog = json.loads((d / "progress.json").read_text(encoding="utf-8"))
        outcome = {(t, int(e)): v for t, dd in prog.items() for e, v in dd.items()}
        shas: dict[tuple[str, int], list[str]] = {}
        for line in open(logs_dir / f"{lp}-w0.probe.jsonl", encoding="utf-8"):
            r = json.loads(line)
            shas.setdefault((r["task"], int(r["ep"])), []).append(r["act_sha"])
        return outcome, shas
    oa, sa = load(a_run, a_lp)
    ob, sb = load(b_run, b_lp)
    keys = sorted(set(oa) & set(ob))
    n_out = sum(1 for k in keys if oa[k] == ob[k])
    n_inf = n_match = 0
    n_first_match = 0
    for k in keys:
        la, lb = sa.get(k, []), sb.get(k, [])
        n = min(len(la), len(lb))
        n_inf += max(len(la), len(lb))
        n_match += sum(1 for i in range(n) if la[i] == lb[i])
        n_first_match += int(bool(la and lb and la[0] == lb[0]))
    status = "BITEXACT" if n_match == n_inf and n_inf > 0 else ("OUTCOME_SAME" if n_out == len(keys) else "OUTCOME_DIFF")
    line = (f"MV_XGPU={status} episodes={len(keys)} infers={n_inf} act_sha_match={n_match}/{n_inf} first_infer_sha_match={n_first_match}/{len(keys)} "
            f"outcome_match={n_out}/{len(keys)} gpu_a={gpu_a} gpu_b={gpu_b} run_a={a_run} run_b={b_run} "
            f"note=观察行；矩阵 w_k→GPU k 固定映射使同一集的四条件永远同卡，跨卡差异不进主比较")
    V.observe(line)
    print(line)
    (out_dir / "xgpu.txt").write_text(line + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
