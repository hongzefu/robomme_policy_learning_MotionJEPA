#!/usr/bin/env python3
"""T3_EVAL_OBS 汇总（motion-memory-plan.md 四节表二）：合并每侧各分片的 progress.json / log.json 与 server 日志 TIMING 行。

输入约定（run_t3_eval_obs.sh 分片口径）：
  结果  v1-store/evaluation/motion-t3-<side><suffix>/ckpt999/seed<seed>/{progress.json,log.json}
  日志  v1-store/logs/motion-t3-<side><suffix>-eval.server.log（TIMING add_buffer_ms=… frames=… / TIMING infer_ms=…）
输出：判定行 `T3_EVAL_OBS open=<p> closed=<q> episodes=<n>/<n>`（无 PASS/FAIL）+ 逐任务成功率 + 两侧 TIMING 统计；JSON 落 --out。
用法：UV_LINK_MODE=copy uv run --no-sync python scripts/training/tests/summarize_t3_eval_obs.py --suffixes -a,-b --out v1-store/reports/motion/t3_eval_obs.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics as st
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))


def _q(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"n": len(xs), "mean": round(st.mean(xs), 1), "median": round(st.median(xs), 1),
            "p90": round(s[max(0, int(0.9 * len(s)) - 1)], 1), "max": round(s[-1], 1)}


def timing(log: pathlib.Path) -> dict:
    ab, inf = [], []
    if not log.is_file():
        return {"missing": str(log)}
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.search(r"TIMING add_buffer_ms=([\d.]+) frames=(\d+)", line)
        if m:
            ab.append((float(m.group(1)), int(m.group(2))))
        m = re.search(r"TIMING infer_ms=([\d.]+)", line)
        if m:
            inf.append(float(m.group(1)))
    return {"add_buffer_le16": _q([a for a, f in ab if f <= 16]), "add_buffer_first_gt16": _q([a for a, f in ab if f > 16]),
            "infer_all": _q(inf), "infer_excl_first": _q(inf[1:])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffixes", default="-a,-b")
    ap.add_argument("--run-prefix", default="motion-t3", help="与 run_t3_eval_obs.sh 的 RUN_PREFIX 同值（结果目录 <prefix>-<side><suffix>）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(_V1 / "reports/motion/t3_eval_obs.json"))
    args = ap.parse_args()
    suffixes = [s for s in args.suffixes.split(",")]
    rep: dict = {"sides": {}, "note": "描述性观察，无 PASS/FAIL；单 seed；两侧 checkpoint 均为 1000 步 × b8 的 T3 run；ep0–9 在 encoder 训练集内"}
    lines = []
    for side in ("closed", "open"):
        per_task: dict[str, dict] = {}
        tim = {}
        for suf in suffixes:
            d = _V1 / "evaluation" / f"{args.run_prefix}-{side}{suf}" / "ckpt999" / f"seed{args.seed}"
            pj = d / "progress.json"
            if not pj.is_file():
                print(f"  ✗ 缺 {pj}", file=sys.stderr)
                continue
            prog = json.loads(pj.read_text())
            for task, eps in prog.items():
                per_task.setdefault(task, {}).update({str(k): v for k, v in eps.items()})
            tim[suf] = {"log_json": (d / "log.json").is_file(), **timing(_V1 / "logs" / f"{args.run_prefix}-{side}{suf}-eval.server.log")}
        rates = {}
        n_total = n_succ = n_err = 0
        for task, eps in sorted(per_task.items()):
            vals = list(eps.values())
            succ = sum(1 for v in vals if v is True); err = sum(1 for v in vals if v == "error")
            rates[task] = {"episodes": len(vals), "success": succ, "error": err, "rate": (succ / len(vals)) if vals else None}
            n_total += len(vals); n_succ += succ; n_err += err
        rep["sides"][side] = {"per_task": rates, "episodes": n_total, "success": n_succ, "error": n_err,
                              "rate": (n_succ / n_total) if n_total else None, "timing": tim}
        lines.append(f"  {side}: {n_succ}/{n_total} 成功（error {n_err}）" + "".join(f" | {t} {r['success']}/{r['episodes']}" for t, r in rates.items()))
    o, c = rep["sides"]["open"], rep["sides"]["closed"]
    head = (f"T3_EVAL_OBS open={o['rate'] if o['rate'] is not None else 'NA'} closed={c['rate'] if c['rate'] is not None else 'NA'} "
            f"episodes={o['episodes']}/{c['episodes']}（单 seed，描述性；ep0–9 泄漏）")
    print(head); print("\n".join(lines))
    for side in ("closed", "open"):
        for suf, t in rep["sides"][side]["timing"].items():
            print(f"  TIMING {side}{suf}: add_buffer(≤16帧) {t.get('add_buffer_le16')} | 首批(>16帧) {t.get('add_buffer_first_gt16')} | infer(除首次) {t.get('infer_excl_first')}")
    rep["line"] = head
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
