#!/usr/bin/env python3
"""G2 的 K_DIST：1600ep 库全部训练样本 (episode, t) 的有效窗数分布（1117-e2e-enc-train-plan.html 第二部分一节
``scripts/dataset/e2e_k_dist.py`` 行、三节 G2、〇.2 P10）。

纯 CPU、只读：只读 motion 表的 ``meta/store_meta.json`` 与 ``meta/motion_index.json``（经 ``MotionMeta.load``，
现场重算 index sha256 并与 store_meta 比对），不读表体。768 维表与步长 16 bf16 新表 index 逐行相同，传哪一张都行。

训练样本集合 = 每个 episode 的 exec 段全部帧 ``t ∈ [exec_start_idx, num_timesteps)``（与
``framesamp_store.build_exec_lookup`` 的 ``_step_of = exec_start_idx + k`` 同一集合）；每个样本的有效窗数
k(t) = ``len(motion_store.visible_motion_rows(entry, t)[0])``，即训练取数实际调用的同一函数。

输出两行：
  ``K_DIST=<mean,p50,p95,max> samples=… episodes=… rows=… budget=… over_budget=… slot_util=…% index_sha256=…``
  ``K_DIST_VS_AUDIT=PASS|FAIL …``——与外部审计重算值（P10：43.906481／39／93／141、超 160 为 0）逐项比对，
  任一不符即 FAIL、退出码 1。mean 按 6 位小数比，p50／p95 用 ``np.percentile`` 默认（linear）插值后按浮点值比。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mme_vla_suite.datastore import motion_store as ms  # noqa: E402

AUDIT = {"mean": "43.906481", "p50": 39, "p95": 93, "max": 141, "over_budget": 0}


def k_values(entries) -> np.ndarray:
    """逐 episode、逐 exec 帧求有效窗数（训练取数同一函数 visible_motion_rows）。"""
    out = []
    for e in entries:
        for t in range(e.exec_start_idx, e.num_timesteps):
            out.append(len(ms.visible_motion_rows(e, t)[0]))
    return np.asarray(out, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", required=True, help="motion 表根（768 维表 <lib>/motion 或 s16 新表）")
    ap.add_argument("--budget", type=int, default=160)
    ap.add_argument("--expect-mean", default=AUDIT["mean"])
    ap.add_argument("--expect-p50", type=float, default=AUDIT["p50"])
    ap.add_argument("--expect-p95", type=float, default=AUDIT["p95"])
    ap.add_argument("--expect-max", type=int, default=AUDIT["max"])
    ap.add_argument("--expect-over-budget", type=int, default=AUDIT["over_budget"])
    args = ap.parse_args()

    meta = ms.MotionMeta.load(pathlib.Path(args.motion).resolve())
    k = k_values(meta.entries)
    if k.size == 0:
        print("K_DIST_VS_AUDIT=FAIL reason=没有任何训练样本", flush=True)
        raise SystemExit(1)
    mean = f"{k.mean():.6f}"
    p50, p95 = np.percentile(k, 50), np.percentile(k, 95)
    kmax = int(k.max())
    over = int((k > args.budget).sum())
    util = 100.0 * k.mean() / args.budget
    print(f"K_DIST={mean},{p50:g},{p95:g},{kmax} samples={k.size} episodes={len(meta.entries)} rows={meta.num_rows} "
          f"budget={args.budget} over_budget={over} slot_util={util:.2f}% index_sha256={meta.motion_index_sha256}",
          flush=True)
    diffs = [name for name, got, want in (("mean", mean, args.expect_mean), ("p50", p50, args.expect_p50),
                                          ("p95", p95, args.expect_p95), ("max", kmax, args.expect_max),
                                          ("over_budget", over, args.expect_over_budget)) if got != want]
    ok = not diffs
    print(f"K_DIST_VS_AUDIT={'PASS' if ok else 'FAIL'} expect={args.expect_mean},{args.expect_p50:g},"
          f"{args.expect_p95:g},{args.expect_max} over_budget_expect={args.expect_over_budget}"
          + ("" if ok else f" diff={','.join(diffs)}"), flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
