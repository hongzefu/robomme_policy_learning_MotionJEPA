#!/usr/bin/env python3
"""多个评估 run 的逐集配对对比（留档 docs/training-doc/eval-orig80k-modul-vs-official/）。

输入：merge_eval_shards.py 产出的 per_episode.json（每集 task / episode / seed / difficulty / success）。
同一 split、同一 seed 下各 run 的同一 (task, ep) 用的是同一个 env seed，故可逐集配对；本脚本先核 seed 一致，
缺集或 seed 不一致一律报错退出（不静默取交集）。success 只有 True 算成功，False / "error" 都记失败（error 数另列）。

每对 (A, B) 在 overall / 每个 suite（与 scripts/training/compute_results.py::TASK_SUITES 同分组，只取本轮任务表里有的）/
每个难度档 / 每个任务 上输出：
  - A、B 成功率与差值 delta = A − B（百分点）
  - 配对 bootstrap 95% CI（按集有放回重采样 --n-boot 次，固定 --boot-seed）
  - 精确 McNemar 双侧 p（b = A 成 B 败，c = A 败 B 成，二项 0.5）
判定行：CMP_<A>_VS_<B> scope=<overall|suite:X|diff:X|task:X> n=<集数> a=<%> b=<%> delta=<pp> ci=[lo,hi] p=<…> verdict=<A_BETTER|B_BETTER|NOT_DETECTED>
verdict 只看 CI 是否跨 0：跨 0 写 NOT_DETECTED（未检出差异，不等于等价）。

用法：
  uv run --no-sync python scripts/training/legacy-eval/compare_eval_runs.py \\
    --run FULL16=<records>/full16/per_episode.json --run OFFICIAL=<records>/official/per_episode.json \\
    --pair FULL16:OFFICIAL --tasks <逗号任务表> --out <records>/compare_full16_vs_official.txt
  uv run --no-sync python scripts/training/legacy-eval/compare_eval_runs.py --self-test   # 已知答案单测
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys

TASK_SUITES = {
    "Counting": ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"],
    "Persistent": ["ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"],
    "Referential": ["PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"],
    "Behavior": ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"],
}


def load_run(path: pathlib.Path, tasks: list[str]) -> dict[tuple[str, int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    eps = {}
    for r in payload["episodes"]:
        if r["task"] in tasks:
            eps[(r["task"], int(r["episode"]))] = {"ok": r["success"] is True, "error": r["success"] == "error",
                                                    "seed": r.get("seed"), "difficulty": str(r.get("difficulty"))}
    return eps


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_stats(pairs: list[tuple[int, int]], n_boot: int, seed: int) -> dict:
    n = len(pairs)
    a = sum(x for x, _ in pairs) / n
    b = sum(y for _, y in pairs) / n
    diffs = [x - y for x, y in pairs]
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[min(n_boot - 1, int(0.975 * n_boot))]
    bb = sum(1 for x, y in pairs if x == 1 and y == 0)
    cc = sum(1 for x, y in pairs if x == 0 and y == 1)
    return {"n": n, "a": a, "b": b, "delta": a - b, "ci": (lo, hi), "p": mcnemar_exact(bb, cc), "a_only": bb, "b_only": cc}


def verdict(ci: tuple[float, float]) -> str:
    if ci[0] > 0:
        return "A_BETTER"
    if ci[1] < 0:
        return "B_BETTER"
    return "NOT_DETECTED"


def compare(runs: dict[str, dict], name_a: str, name_b: str, tasks: list[str], n_boot: int, seed: int) -> list[str]:
    ra, rb = runs[name_a], runs[name_b]
    keys = sorted(ra)
    if set(ra) != set(rb):
        raise SystemExit(f"错误: {name_a} 与 {name_b} 的集合不一致：仅 A {sorted(set(ra) - set(rb))[:5]} 仅 B {sorted(set(rb) - set(ra))[:5]}")
    bad = [k for k in keys if str(ra[k]["seed"]) != str(rb[k]["seed"])]
    if bad:
        raise SystemExit(f"错误: {len(bad)} 集 seed 不一致，例 {bad[:5]}")
    scopes: list[tuple[str, list]] = [("overall", keys)]
    for s, ts in TASK_SUITES.items():
        sel = [k for k in keys if k[0] in ts]
        if sel and len({k[0] for k in sel}) > 1:
            scopes.append((f"suite:{s}", sel))
    for d in sorted({ra[k]["difficulty"] for k in keys}):
        scopes.append((f"diff:{d}", [k for k in keys if ra[k]["difficulty"] == d]))
    for t in tasks:
        scopes.append((f"task:{t}", [k for k in keys if k[0] == t]))
    tag = f"CMP_{name_a}_VS_{name_b}"
    lines = []
    for scope, sel in scopes:
        st = paired_stats([(int(ra[k]["ok"]), int(rb[k]["ok"])) for k in sel], n_boot, seed)
        lines.append(f"{tag} scope={scope} n={st['n']} a={100 * st['a']:.2f} b={100 * st['b']:.2f} "
                     f"delta={100 * st['delta']:+.2f} ci=[{100 * st['ci'][0]:+.2f},{100 * st['ci'][1]:+.2f}] "
                     f"p={st['p']:.4g} a_only={st['a_only']} b_only={st['b_only']} verdict={verdict(st['ci'])}")
    ea = sum(1 for k in keys if ra[k]["error"])
    eb = sum(1 for k in keys if rb[k]["error"])
    lines.append(f"{tag}_ERRORS {name_a}={ea} {name_b}={eb}（error 集按失败计入上面各行）")
    return lines


def self_test() -> int:
    # 已知答案：40 集，A 成 30、B 成 20；A 独成 12、B 独成 2 → delta=+25pp，McNemar p = 2*P(X<=2 | n=14) = 2*(1+14+91)/16384
    pairs = [(1, 1)] * 18 + [(1, 0)] * 12 + [(0, 1)] * 2 + [(0, 0)] * 8
    st = paired_stats(pairs, 2000, 0)
    exp_p = 2 * (1 + 14 + 91) / 2 ** 14
    ok = (st["n"] == 40 and abs(st["a"] - 0.75) < 1e-12 and abs(st["b"] - 0.5) < 1e-12 and abs(st["delta"] - 0.25) < 1e-12
          and abs(st["p"] - exp_p) < 1e-12 and st["ci"][0] > 0 and st["ci"][0] <= 0.25 <= st["ci"][1])
    ok &= mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(5, 5) == 1.0
    # 全同 → delta 0、CI [0,0]、NOT_DETECTED
    st0 = paired_stats([(1, 1)] * 5 + [(0, 0)] * 5, 500, 0)
    ok &= st0["delta"] == 0 and st0["ci"] == (0, 0) and verdict(st0["ci"]) == "NOT_DETECTED"
    # compare 端到端：两任务、error 计失败；seed 不一致必须报错
    ra = {("BinFill", 0): {"ok": True, "error": False, "seed": 1, "difficulty": "easy"},
          ("StopCube", 0): {"ok": False, "error": True, "seed": 2, "difficulty": "hard"}}
    rb = {k: dict(v, ok=False, error=False) for k, v in ra.items()}
    out = compare({"A": ra, "B": rb}, "A", "B", ["BinFill", "StopCube"], 200, 0)
    ok &= out[0].startswith("CMP_A_VS_B scope=overall n=2 a=50.00 b=0.00 delta=+50.00") and out[-1].startswith("CMP_A_VS_B_ERRORS A=1 B=0")
    rb_bad = {k: dict(v, seed=99) for k, v in rb.items()}
    try:
        compare({"A": ra, "B": rb_bad}, "A", "B", ["BinFill", "StopCube"], 200, 0)
        ok = False
    except SystemExit:
        pass
    print(f"COMPARE_SELFTEST={'PASS' if ok else 'FAIL'} delta={st['delta']:+.4f} p={st['p']:.6g} exp_p={exp_p:.6g} ci=[{st['ci'][0]:+.4f},{st['ci'][1]:+.4f}]")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="名=per_episode.json，可多次")
    ap.add_argument("--pair", action="append", default=[], help="A:B（delta = A − B），可多次")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    tasks = [t for t in args.tasks.split(",") if t]
    if not (tasks and args.run and args.pair):
        ap.error("需要 --tasks、--run、--pair")
    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        runs[name] = load_run(pathlib.Path(path), tasks)
        n_exp = 50 * len(tasks)
        if len(runs[name]) != n_exp:
            raise SystemExit(f"错误: {name} 在任务表内有 {len(runs[name])} 集，期望 {n_exp}")
    lines = []
    for spec in args.pair:
        a, b = spec.split(":")
        lines += compare(runs, a, b, tasks, args.n_boot, args.boot_seed)
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
