"""独立读取固定参考与目标数组，重算motion数值证据后签发目标合同。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.evaluation.runtime_contract import (canonical, descriptor_identity, identity_sha256,
    motion_descriptor, read_json, sha256_file, validate_target_contract, verifier_sha256, verify_measurement)


def sign_target_contract(*, provenance_path, fixture_path, measurement_path, eval_code_commit, output):
    provenance_path, fixture_path, measurement_path, output = map(Path,
        (provenance_path, fixture_path, measurement_path, output))
    # 所有引用由证书目录内实体提供，便于在异机完整读回校验。
    for path in (fixture_path, measurement_path):
        if not path.resolve().is_relative_to(output.parent.resolve()):
            raise ValueError("参考与测量报告须位于证书目录树内")
    if output.exists() or output.resolve() in (fixture_path.resolve(), measurement_path.resolve(), provenance_path.resolve()):
        raise ValueError("禁止覆盖已有证书或输入")
    provenance = read_json(provenance_path)
    descriptor = motion_descriptor(provenance, sha256_file(fixture_path))
    motion_identity = descriptor_identity("motion", descriptor)
    measured = read_json(measurement_path)
    runtime = descriptor["software_numerics"]
    verdict = verify_measurement(fixture_path, measurement_path, fixture_sha256=sha256_file(fixture_path),
        motion_identity_sha256=motion_identity, expected_runtime=runtime,
        target_gpu_family=measured.get("target_gpu_family"), target_driver=measured.get("target_driver"))
    certificate = {"schema": "eval-target-contract-v1", "schema_version": 1, "status": "validated",
        "training_provenance_sha256": identity_sha256(provenance),
        "training_provenance_file_sha256": sha256_file(provenance_path),
        "motion_identity_sha256": motion_identity, "weights": descriptor["weights"],
        "source_pin_sha256": descriptor["source_pin_sha256"], "thresholds": descriptor["thresholds"],
        "allowed_hardware_differences": ["gpu_name", "driver"], "training_runtime": runtime,
        "target_runtime_sha256": identity_sha256(measured["runtime"]),
        "target_gpu_family": measured["target_gpu_family"], "target_driver": measured["target_driver"],
        "fixture": {"path": str(fixture_path.resolve().relative_to(output.parent.resolve())), "sha256": sha256_file(fixture_path)},
        "measurement": {"path": str(measurement_path.resolve().relative_to(output.parent.resolve())), "sha256": sha256_file(measurement_path)},
        "eval_code_commit": eval_code_commit, "verifier_sha256": verifier_sha256(), "recomputed_verdict": verdict}
    output.parent.mkdir(parents=True, exist_ok=True)
    # 校验最终消费路径，失败不留下可被误用的validated文件。
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(canonical(certificate) + b"\n")
        validate_target_contract(temporary, training_provenance=provenance, actual_runtime=measured["runtime"],
                                 training_provenance_path=provenance_path)
        with output.open("xb") as stream:
            stream.write(temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)
    return certificate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("provenance", "fixture", "measurement", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--eval-code-commit", required=True)
    args = parser.parse_args(argv)
    try:
        certificate = sign_target_contract(provenance_path=args.provenance, fixture_path=args.fixture,
            measurement_path=args.measurement, eval_code_commit=args.eval_code_commit, output=args.output)
    except FileNotFoundError as error:
        print(f"A40_TARGET=PENDING reason=missing_reference detail={error}")
        return 2
    except (ValueError, KeyError, TypeError) as error:
        print(f"A40_TARGET=FAIL detail={error}")
        return 1
    print(f"A40_TARGET=PASS samples={certificate['recomputed_verdict']['samples']} max_abs={certificate['recomputed_verdict']['max_abs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
