#!/usr/bin/env python3
"""T3_SMOKE：两侧（open / closed）训练 records 的收尾汇总判定。

motion-memory-plan.md 第一部分五节的 T3_SMOKE 原是人工从 records 汇总，仓库里没有 emitter；环境 B 复刻（2026-09-04）
把它写成脚本，判据不变、全部零容差：

  1. metrics.jsonl 两侧都恰好 steps 行（步 0..steps-1），loss / grad_norm / param_norm 全部有限 → nan=0；
  2. param_checksums.jsonl 首末两条摘要：open 侧 per_leaf 比 closed 侧多出的叶子里，`params[` 开头的恰 4 个
     （motion 投影 / pos / …），且这 4 个的 sha256 首末不同（motion 参数确实被更新）；其 ema / opt 对应叶也全部变化；
     closed 侧没有任何 motion 叶；
  3. batch_digests.jsonl 键数 open 16 / closed 12；param_checksums 叶数 open 193 / closed 177。

判定行：T3_SMOKE=PASS steps=<n> nan=0 motion_params_updated=4 n_keys=16/12 n_leaves=193/177

用法：uv run --no-sync python scripts/training/tests/t3_smoke.py --open-records <dir> --closed-records <dir> [--steps 100]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

_EXPECT_OPEN_ONLY_PARAMS = 4
_EXPECT_N_KEYS = (16, 12)
_EXPECT_N_LEAVES = (193, 177)


def _jsonl(p: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _scalar(v):
    return float(v["dec"]) if isinstance(v, dict) else float(v)


def _metrics_check(rec: pathlib.Path, steps: int, fails: list[str], tag: str) -> int:
    rows = _jsonl(rec / "metrics.jsonl")
    got = sorted(int(r["step"]) for r in rows)
    if got != list(range(steps)):
        fails.append(f"{tag}: metrics.jsonl 步集 {got[:3]}…{got[-3:]}（{len(got)} 行）≠ 0..{steps - 1}")
    nan = 0
    for r in rows:
        for k in ("loss", "grad_norm", "param_norm"):
            if k in r and not math.isfinite(_scalar(r[k])):
                nan += 1
    if nan:
        fails.append(f"{tag}: {nan} 个非有限标量")
    return nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open-records", required=True)
    ap.add_argument("--closed-records", required=True)
    ap.add_argument("--steps", type=int, default=None, help="默认从 open 侧 env.json 的 steps 读")
    args = ap.parse_args()
    ro, rc = pathlib.Path(args.open_records), pathlib.Path(args.closed_records)
    fails: list[str] = []

    steps = args.steps
    if steps is None:
        env = json.loads((ro / "env.json").read_text())
        steps = int(env["steps"])
        env_c = json.loads((rc / "env.json").read_text())
        if int(env_c["steps"]) != steps:
            fails.append(f"两侧 env.json steps 不同: open={steps} closed={env_c['steps']}")
    nan = _metrics_check(ro, steps, fails, "open") + _metrics_check(rc, steps, fails, "closed")

    po, pc = _jsonl(ro / "param_checksums.jsonl"), _jsonl(rc / "param_checksums.jsonl")
    if len(po) < 2 or len(pc) < 2:
        fails.append(f"param_checksums 少于 2 条摘要: open={len(po)} closed={len(pc)}")
        n_updated = 0
    else:
        po.sort(key=lambda r: int(r["step"])); pc.sort(key=lambda r: int(r["step"]))
        ko, kc = set(po[0]["per_leaf"]), set(pc[0]["per_leaf"])
        if kc - ko:
            fails.append(f"closed 侧有 open 侧没有的叶子: {sorted(kc - ko)[:3]}")
        open_only = sorted(ko - kc)
        motion_like = [k for k in kc if "motion" in k.lower()]
        if motion_like:
            fails.append(f"closed 侧含 motion 叶: {motion_like[:3]}")
        only_params = [k for k in open_only if k.startswith("params[")]
        if len(only_params) != _EXPECT_OPEN_ONLY_PARAMS:
            fails.append(f"open-only 的 params 叶 {len(only_params)} 个 ≠ {_EXPECT_OPEN_ONLY_PARAMS}: {only_params}")
        first, last = po[0]["per_leaf"], po[-1]["per_leaf"]
        unchanged = [k for k in open_only if first[k] == last.get(k)]
        n_updated = sum(1 for k in only_params if first[k] != last.get(k))
        if unchanged:
            fails.append(f"open-only 叶首末摘要未变（step {po[0]['step']}→{po[-1]['step']}）: {unchanged[:4]}")
        n_leaves = (int(po[0]["n_leaves"]), int(pc[0]["n_leaves"]))
        if n_leaves != _EXPECT_N_LEAVES:
            fails.append(f"n_leaves {n_leaves} ≠ {_EXPECT_N_LEAVES}")

    bo, bc = _jsonl(ro / "batch_digests.jsonl"), _jsonl(rc / "batch_digests.jsonl")
    n_keys = (int(bo[0]["n_keys"]) if bo else -1, int(bc[0]["n_keys"]) if bc else -1)
    if n_keys != _EXPECT_N_KEYS:
        fails.append(f"batch_digests n_keys {n_keys} ≠ {_EXPECT_N_KEYS}")
    if any(int(r["n_keys"]) != n_keys[0] for r in bo) or any(int(r["n_keys"]) != n_keys[1] for r in bc):
        fails.append("batch_digests 键数随步摆动")

    ok = not fails
    for f in fails:
        print("  ✗", f)
    nl = f"{n_leaves[0]}/{n_leaves[1]}" if not (len(po) < 2 or len(pc) < 2) else "?/?"
    print(f"T3_SMOKE={'PASS' if ok else 'FAIL'} steps={steps} nan={nan} motion_params_updated={n_updated} "
          f"n_keys={n_keys[0]}/{n_keys[1]} n_leaves={nl}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
