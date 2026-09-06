#!/usr/bin/env python3
"""合并 8 卡分片评估结果（eval_all_shards.sh / eval_shard.sh 口径；留档 docs/training-doc/eval-awsprod40k-b128-motion/）。

输入（--layout 两种分片布局，分片目录 / 日志名不同，其余同）：
  task   ：<eval-root>/<run>-<task>-<k>/ckpt<id>/seed<seed>/{progress.json,log.json,videos/*.mp4}（每任务 --shards 片，旧口径）
  stride ：<eval-root>/<run>-w<k>/ckpt<id>/seed<seed>/…（--shards 个 worker，每个跑全部任务、集号交错；留档 eval-official-framesamp-context/）
  <logs-dir>/<prefix>-<分片名>.log（EVAL_SHARD 起止行）与 <prefix>-<分片名>.server.log（TIMING add_buffer_ms / infer_ms 行）
  <metadata-dir>/test/record_dataset_<task>_metadata.json（seed、难度）
输出：
  合并 progress.json / log.json / shards.json → <eval-root>/<run>/ckpt<id>/seed<seed>/（与单进程 eval.py 布局一致；视频留各分片目录）
  summary.txt / per_episode.json → --out-dir
成功率公式与 eval.py::evaluate 相同：任务成功率 = 成功集数 / 该任务集数；总成功率 = 各任务成功率的算术平均。
逐集三分 success / fail / timeout 取自 eval.py 视频文件名 `<task>_ep<k>_<success_flag>_…mp4`。
判定行：<TAG>=DONE|INCOMPLETE tasks=<t> episodes=<n>/<N> errors=<e> <task>=<succ>/<n> … mean_rate=<r> timeout=<k>/<n> fail=<k>/<n>
用法：UV_LINK_MODE=copy uv run --no-sync python scripts/training/prod/merge_eval_shards.py --out-dir docs/training-doc/eval-awsprod40k-b128-motion/records [--allow-partial]
      stride 布局：… --layout stride --shards 8 --run-name official-framesamp-context --ckpt-id 79999 --log-prefix evoffctx --tag EVAL_OFFICIAL_CTX --out-dir …
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import statistics as st
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))
TASKS = "ButtonUnmask,VideoUnmask,ButtonUnmaskSwap,VideoUnmaskSwap"
_VIDEO_RE = re.compile(r"^(?P<task>[A-Za-z]+)_ep(?P<ep>\d+)_(?P<flag>[a-z]+)_")
_TS = "%Y-%m-%d %H:%M:%S"


def _q(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"n": len(xs), "mean": round(st.mean(xs), 1), "median": round(st.median(xs), 1),
            "p90": round(s[max(0, int(0.9 * len(s)) - 1)], 1), "max": round(s[-1], 1)}


def timing(log: pathlib.Path) -> dict:
    """同 scripts/training/tests/summarize_t3_eval_obs.py::timing（server 日志 TIMING 行，单位 ms）。"""
    if not log.is_file():
        return {"missing": str(log)}
    ab, inf = [], []
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"TIMING add_buffer_ms=([\d.]+) frames=(\d+)", line)
        if m:
            ab.append((float(m.group(1)), int(m.group(2))))
        m = re.search(r"TIMING infer_ms=([\d.]+)", line)
        if m:
            inf.append(float(m.group(1)))
    return {"add_buffer_le16": _q([a for a, f in ab if f <= 16]), "add_buffer_first_gt16": _q([a for a, f in ab if f > 16]),
            "infer_all": _q(inf), "infer_excl_first": _q(inf[1:])}


def shard_wall(log: pathlib.Path) -> dict:
    """eval_shard.sh 的 `=== EVAL_SHARD … start=… ===` 与 `EVAL_RC=<rc> end=…` / `EXIT_CODE=` 行。"""
    if not log.is_file():
        return {"missing": str(log)}
    start = end = rc = exit_code = None
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"=== EVAL_SHARD .* start=(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ===", line)
        if m:
            start = m.group(1)
        m = re.search(r"EVAL_RC=(\d+) end=(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
        if m:
            rc, end = int(m.group(1)), m.group(2)
        m = re.match(r"EXIT_CODE=(\d+)", line)
        if m:
            exit_code = int(m.group(1))
    wall = None
    if start and end:
        wall = (dt.datetime.strptime(end, _TS) - dt.datetime.strptime(start, _TS)).total_seconds()
    return {"start": start, "end": end, "wall_s": wall, "eval_rc": rc, "exit_code": exit_code}


def load_test_meta(meta_dir: pathlib.Path, task: str) -> dict[int, dict]:
    p = meta_dir / "test" / f"record_dataset_{task}_metadata.json"
    if not p.is_file():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {int(r["episode"]): {"seed": r.get("seed"), "difficulty": r.get("difficulty")} for r in payload.get("records", [])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="awsprod40k-b128-motion")
    ap.add_argument("--ckpt-id", type=int, default=39999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--layout", choices=("task", "stride"), default="task",
                    help="task=旧布局 <run>-<task>-<k>（每任务 --shards 片）；stride=新布局 <run>-w<k>（--shards 即 worker 数）")
    ap.add_argument("--shards", type=int, default=2, help="task 布局：每任务分片数（eval_all_shards.sh 为 2）；stride 布局：worker 数")
    ap.add_argument("--episodes-per-task", type=int, default=50)
    ap.add_argument("--log-prefix", default="ev40k")
    ap.add_argument("--tag", default="EVAL_40K_MOTION")
    ap.add_argument("--eval-root", default=str(_V1 / "evaluation"))
    ap.add_argument("--logs-dir", default=str(_V1 / "logs"))
    ap.add_argument("--metadata-dir", default=str(_REPO_ROOT / "third_party/robomme_benchmark/src/robomme/env_metadata"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-partial", action="store_true", help="中途看进度：只写 summary / per_episode，不写合并 progress / log.json，退出码 0")
    args = ap.parse_args()
    tasks = args.tasks.split(",")
    eval_root, logs_dir, meta_dir = pathlib.Path(args.eval_root), pathlib.Path(args.logs_dir), pathlib.Path(args.metadata_dir)
    N = args.episodes_per_task

    per_task: dict[str, dict[int, object]] = {t: {} for t in tasks}
    flags: dict[tuple[str, int], str] = {}
    shard_of: dict[tuple[str, int], str] = {}
    shards: list[dict] = []
    # 分片单元表 (label, 结果目录名, 日志名去后缀)：task 布局按任务 × 片枚举，stride 布局按 worker 枚举；后续统计只认 progress.json 内容
    if args.layout == "stride":
        units = [(f"w{k}", f"{args.run_name}-w{k}", f"{args.log_prefix}-w{k}") for k in range(args.shards)]
    else:
        units = [(f"{t}-{k}", f"{args.run_name}-{t}-{k}", f"{args.log_prefix}-{t}-{k}") for t in tasks for k in range(args.shards)]
    for label, dirname, logbase in units:
        d = eval_root / dirname / f"ckpt{args.ckpt_id}" / f"seed{args.seed}"
        pj = d / "progress.json"
        info = {"shard": label, "dir": str(d), "has_progress": pj.is_file(), "finished": (d / "log.json").is_file(), "episodes": 0}
        if pj.is_file():
            prog = json.loads(pj.read_text(encoding="utf-8"))
            for t, eps in prog.items():
                for e, v in eps.items():
                    per_task.setdefault(t, {})[int(e)] = v
                    shard_of[(t, int(e))] = label
                    info["episodes"] += 1
            for mp4 in (d / "videos").glob("*.mp4") if (d / "videos").is_dir() else []:
                m = _VIDEO_RE.match(mp4.name)
                if m:
                    flags[(m.group("task"), int(m.group("ep")))] = m.group("flag")
        info["wall"] = shard_wall(logs_dir / f"{logbase}.log")
        info["timing"] = timing(logs_dir / f"{logbase}.server.log")
        shards.append(info)

    rates: dict[str, dict] = {}
    per_episode: list[dict] = []
    n_total = n_succ = n_err = n_timeout = n_fail = 0
    complete = all(s["finished"] for s in shards)
    for task in tasks:
        meta = load_test_meta(meta_dir, task)
        eps = per_task.get(task, {})
        missing = sorted(set(range(N)) - set(eps))
        extra = sorted(set(eps) - set(range(N)))
        errs = sorted(e for e, v in eps.items() if v == "error")
        succ = sum(1 for v in eps.values() if v is True)
        by_diff: dict[str, list[int]] = {}
        for e in sorted(eps):
            v = eps[e]
            flag = flags.get((task, e), "?")
            diff = str(meta.get(e, {}).get("difficulty"))
            by_diff.setdefault(diff, [0, 0])
            by_diff[diff][1] += 1
            by_diff[diff][0] += 1 if v is True else 0
            if flag == "timeout":
                n_timeout += 1
            elif v is False:
                n_fail += 1
            per_episode.append({"task": task, "episode": e, "seed": meta.get(e, {}).get("seed"), "difficulty": diff,
                                "success": v, "flag": flag, "shard": shard_of.get((task, e))})
        n = len(eps)
        rate = (succ / n) if n else None
        rates[task] = {"episodes": n, "success": succ, "error": len(errs), "rate": rate, "missing": missing, "extra": extra,
                       "timeout": sum(1 for e in eps if flags.get((task, e)) == "timeout"),
                       "fail": sum(1 for e, v in eps.items() if v is False and flags.get((task, e)) != "timeout"),
                       "by_difficulty": {d: f"{a}/{b}" for d, (a, b) in sorted(by_diff.items())}}
        n_total += n
        n_succ += succ
        n_err += len(errs)
        if missing or extra or errs or n != N:
            complete = False
    valid_rates = [r["rate"] for r in rates.values() if r["rate"] is not None]
    mean_rate = st.mean(valid_rates) if valid_rates else None
    status = "DONE" if complete else "INCOMPLETE"
    head = (f"{args.tag}={status} tasks={len(tasks)} episodes={n_total}/{N * len(tasks)} errors={n_err} "
            + " ".join(f"{t}={rates[t]['success']}/{rates[t]['episodes']}" for t in tasks)
            + f" mean_rate={mean_rate:.4f} timeout={n_timeout}/{n_total} fail={n_fail}/{n_total}" if mean_rate is not None else
            f"{args.tag}={status} tasks={len(tasks)} episodes=0/{N * len(tasks)} errors=0 mean_rate=NA")

    lines = [head, ""]
    lines.append("| 任务 | 成功/集数 | 成功率 | timeout | fail | 按难度 成功/集数 | 缺集 | error |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in tasks:
        r = rates[t]
        rate_s = "—" if r["rate"] is None else f"{r['rate']:.2%}"
        lines.append(f"| {t} | {r['success']}/{r['episodes']} | {rate_s} | {r['timeout']} | {r['fail']} | "
                     f"{', '.join(f'{d} {v}' for d, v in r['by_difficulty'].items())} | {len(r['missing'])} | {r['error']} |")
    lines += ["", "| 分片 | 集数 | 完成 | 墙钟 | add_buffer≤16帧 ms（mean/median/p90） | 首批 ms（mean/max） | infer ms（除首次 mean/median/p90） |", "|---|---|---|---|---|---|---|"]
    for s in shards:
        w, tm = s["wall"], s["timing"]
        wall = f"{w['wall_s'] / 60:.1f} min" if w.get("wall_s") else "—"
        a, f_, i = tm.get("add_buffer_le16", {}), tm.get("add_buffer_first_gt16", {}), tm.get("infer_excl_first", {})
        fmt = lambda q, keys: "/".join(str(q.get(k, "—")) for k in keys) if q.get("n") else "—"
        lines.append(f"| {s['shard']} | {s['episodes']} | {'是' if s['finished'] else '否'} | {wall} | {fmt(a, ('mean', 'median', 'p90'))} | {fmt(f_, ('mean', 'max'))} | {fmt(i, ('mean', 'median', 'p90'))} |")
    lines += ["", "逐集（task ep seed difficulty result flag shard）："]
    for r in per_episode:
        lines.append(f"  {r['task']} {r['episode']:2d} {r['seed']} {r['difficulty']} {r['success']} {r['flag']} {r['shard']}")
    summary = "\n".join(lines) + "\n"
    print(summary)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    (out_dir / "per_episode.json").write_text(json.dumps(
        {"line": head, "rates": rates, "mean_rate": mean_rate, "shards": shards, "episodes": per_episode}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if complete:
        merged = eval_root / args.run_name / f"ckpt{args.ckpt_id}" / f"seed{args.seed}"
        merged.mkdir(parents=True, exist_ok=True)
        (merged / "progress.json").write_text(json.dumps({t: {str(e): per_task[t][e] for e in sorted(per_task[t])} for t in tasks}, indent=2) + "\n", encoding="utf-8")
        (merged / "log.json").write_text(json.dumps({"success_rate": {t: rates[t]["rate"] for t in tasks}, "total_success_rate": mean_rate}, indent=2) + "\n", encoding="utf-8")
        (merged / "shards.json").write_text(json.dumps([{"shard": s["shard"], "dir": s["dir"]} for s in shards], indent=2) + "\n", encoding="utf-8")
        print(f"合并结果写入 {merged}（视频留各分片目录，见 shards.json）")
        return 0
    return 0 if args.allow_partial else 1


if __name__ == "__main__":
    sys.exit(main())
