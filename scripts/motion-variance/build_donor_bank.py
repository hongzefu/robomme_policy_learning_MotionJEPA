#!/usr/bin/env python3
"""建 donor bank（本轮只实现 --source library）。判定行：
  DONOR_BANK=PASS|FAIL source=library rows= tasks=4 pool=100 self_loops=0 primary_pairs= chain_len=[…] exec_grid_max=[…] sha256=
  DONOR_COVER_EST windows= exact= fallback= cycle= cross_seg= by_task=[…]（接收方最坏 t=es+1300 的估计，观察行）
用法：MMEVLA_V1_STORE 可选；uv run --no-sync python scripts/motion-variance/build_donor_bank.py --source library --out v1-store/reports/motion-variance/bank-lib
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import mv_common as C  # noqa: E402
from mv_donor_bank import DonorBank, build_library_bank  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("library",), default="library")
    ap.add_argument("--lib", default=str(C.LIB))
    ap.add_argument("--out", default=str(C.REPORT_ROOT / "bank-lib"))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    meta = build_library_bank(pathlib.Path(args.lib), out)
    bank = DonorBank(out)
    V = C.Verdict()
    n_primary = sum(1 for k in meta["primary"] if not k.startswith("ol"))
    ok = meta["self_loops"] == 0 and all(v == 100 for v in meta["pool_size"].values())
    V.block(ok, f"DONOR_BANK={'PASS' if ok else 'FAIL'} source=library rows={meta['tokens_shape'][0]} tasks={len(C.TASKS)} "
                f"pool={meta['pool_size']} self_loops={meta['self_loops']} primary_pairs={n_primary} "
                f"open_loop_pairs={len(meta['primary']) - n_primary} chain_len={[len(meta['fallback_chain'][t]) for t in C.TASKS]} "
                f"exec_grid_max={meta['exec_grid_max']} sha256={meta['sha256'][:16]}")
    cov = bank.coverage_estimate()
    tot = cov["total"]; n = sum(tot.values())
    V.observe(f"DONOR_COVER_EST windows={n} " + " ".join(f"{h}={tot[h]}/{n}({tot[h] / n:.1%})" for h in tot) + " by_task=["
              + ", ".join(f"{t}: " + "/".join(f"{h[:2]}={cov['by_task'][t][h]}" for h in tot) for t in C.TASKS)
              + "] note=接收方按 t=es+1300 最坏估计；矩阵跑完由 summarize_mv.py 按实际推理次数重算 DONOR_COVER_ACTUAL")
    ok2 = tot["cross_seg"] == 0
    V.block(ok2, f"DONOR_CROSS_SEG={'PASS' if ok2 else 'FAIL'} cross_seg={tot['cross_seg']}")
    print("\n".join(V.texts() + [V.final("DONOR_BUILD", f"out={out}")]))
    return 0 if V.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
