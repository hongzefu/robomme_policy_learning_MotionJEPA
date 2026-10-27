"""记录普通评估运行合同和实际硬件属性，不进行跨硬件数值认证。"""
from __future__ import annotations
import argparse
from pathlib import Path
from scripts.evaluation.runtime_contract import canonical, make_eval_runtime_contract, read_json, validate_eval_runtime_contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--actual-runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("禁止覆盖已有运行合同")
    provenance = read_json(args.provenance)
    contract = make_eval_runtime_contract(provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(contract) + b"\n")
    if args.actual_runtime is not None:
        validate_eval_runtime_contract(args.output, training_provenance=provenance,
                                      actual_runtime=read_json(args.actual_runtime))
    print("EVAL_RUNTIME_CONTRACT=RECORDED cross_hardware_numeric=not_required numeric_verdict=not_applicable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
