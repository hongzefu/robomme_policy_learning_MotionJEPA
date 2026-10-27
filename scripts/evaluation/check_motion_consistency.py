"""同一运行时内的数组重复性检查，不签发跨硬件证书。"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scripts.evaluation.runtime_contract import numeric_metrics


def check_repeatability(reference, actual):
    metrics = numeric_metrics(reference, actual)
    if metrics["max_abs"] != 0 or reference.tobytes() != actual.tobytes():
        raise ValueError("同机固定输入/noise输出未逐位重复")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        check_repeatability(np.load(args.reference, allow_pickle=False), np.load(args.actual, allow_pickle=False))
    except Exception as error:
        print(f"LOCAL_REPEATABILITY=FAIL detail={error}")
        return 1
    print("LOCAL_REPEATABILITY=PASS cross_hardware_numeric=not_required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
