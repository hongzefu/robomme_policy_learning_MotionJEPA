#!/usr/bin/env python3
"""两份 single_step_grad.py 产物（grad_summary.json）逐叶互核——A22 式「同机两侧」口径。

环境 B（AWS，2026-09-04）没有环境 A 的 `v1-dtype-p5-grad` 固化基线（Ada 卡产出，A100 上 bf16 归约不逐位），
A22 的「对历史基线逐叶 sha」改为：S2_BASE 旧码 worktree 与 HEAD 各跑一次 single_step_grad.py
（同 seed / 同 fixture / 同确定性 XLA_FLAGS / 同两张卡），本脚本逐 kind 比：

  * batch 索引（fixture 同源）逐项相同；
  * loss 的 float.hex() 相同；
  * 每叶梯度 sha256 相同、叶集合相同。

全部零容差。判定行：GRAD_EQ=PASS kinds=3 leaves=<每 kind 叶数> mismatches=0

用法：uv run --no-sync python scripts/training/tests/compare_grad_summaries.py <A/grad_summary.json> <B/grad_summary.json>
"""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    A = json.loads(pathlib.Path(args.a).read_text(encoding="utf-8"))
    B = json.loads(pathlib.Path(args.b).read_text(encoding="utf-8"))
    fails: list[str] = []
    for k in ("seed", "fsdp_devices", "xla_flags"):
        if A.get(k) != B.get(k):
            fails.append(f"口径不同 {k}: {A.get(k)!r} vs {B.get(k)!r}")
    ra, rb = A["results"], B["results"]
    if set(ra) != set(rb):
        fails.append(f"kind 集合不同: {sorted(ra)} vs {sorted(rb)}")
    leaves = set()
    for kind in sorted(set(ra) & set(rb)):
        x, y = ra[kind], rb[kind]
        if x["indices"] != y["indices"]:
            fails.append(f"{kind}: batch 索引不同")
        if x["loss_hex"] != y["loss_hex"]:
            fails.append(f"{kind}: loss 不同 {x['loss_hex']} vs {y['loss_hex']}")
        pa, pb = x["per_leaf"], y["per_leaf"]
        if set(pa) != set(pb):
            fails.append(f"{kind}: 叶集合不同（{len(pa)} vs {len(pb)}）")
        bad = [l for l in pa if l in pb and pa[l] != pb[l]]
        for l in bad[:5]:
            print(f"  ✗ {kind} {l} sha 不同")
        if bad:
            fails.append(f"{kind}: {len(bad)} 叶梯度 sha 不同")
        leaves.add(len(pa))
    ok = not fails
    for f in fails:
        print("  ✗", f)
    print(f"GRAD_EQ={'PASS' if ok else 'FAIL'} kinds={len(set(ra) & set(rb))} "
          f"leaves={'/'.join(str(n) for n in sorted(leaves))} mismatches={len(fails)}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
