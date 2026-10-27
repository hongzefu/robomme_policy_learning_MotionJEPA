"""用合成CPU数组验证签证流程与负控，不代表A40数值证据。"""

import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.check_motion_consistency import sign_target_contract
from scripts.evaluation.runtime_contract import (canonical, compare_runtime, descriptor_identity, motion_descriptor,
    numeric_metrics, read_json, sha256_file, validate_target_contract)


@pytest.fixture
def evidence(tmp_path):
    numeric = {"gpu_name": "NVIDIA A100-SXM4-80GB", "driver": "old", "torch": "fixed", "dtype": "float32"}
    provenance = {"encoder": {"checkpoint_sha256": "a" * 64, "state_key": "model"},
                  "encoder_state_sha256": {"sha256": "b" * 64}, "source_pin": {"sha": "c" * 64},
                  "vae": {"vae_state_sha256": "d" * 64}, "numerics": numeric,
                  "stage2_compat": {"online_vae": {"vae_state_sha256": "d" * 64, "numerics": numeric}}}
    prov_path = tmp_path / "provenance.json"
    prov_path.write_bytes(canonical(provenance))
    samples, actual_samples = [], []
    for index, layer in enumerate(("encoder", "e2e")):
        ref, out = tmp_path / f"ref{index}.npy", tmp_path / f"out{index}.npy"
        np.save(ref, np.arange(1024, dtype=np.float32) / 1024)
        np.save(out, np.arange(1024, dtype=np.float32) / 1024)
        samples.append({"id": str(index), "layer": layer, "coverage": ["demo_tail", "real16", "exec"],
                        "reference": {"path": ref.name, "sha256": sha256_file(ref)},
                        "input": {"path": ref.name, "sha256": sha256_file(ref)}})
        actual_samples.append({"id": str(index), "actual": {"path": out.name, "sha256": sha256_file(out)}})
    fixture = {"schema": "eval-motion-reference-v1", "status": "locked", "samples": samples,
               "producer_code_sha256": "e" * 64, "producer_gpu_family": "NVIDIA A100-SXM4-80GB",
               "reference_kind": "packed-reference"}
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_bytes(canonical(fixture))
    runtime = copy.deepcopy({"numerics": numeric, "online_vae": provenance["stage2_compat"]["online_vae"]})
    runtime["numerics"].update(gpu_name="NVIDIA A40", driver="new")
    runtime["online_vae"]["numerics"].update(gpu_name="NVIDIA A40", driver="new")
    measurement = {"schema": "eval-motion-measurement-v1", "status": "unvalidated_measurement",
        "fixture_manifest_sha256": sha256_file(fixture_path),
        "motion_identity_sha256": descriptor_identity("motion", motion_descriptor(provenance, sha256_file(fixture_path))),
        "measurement_code_sha256": sha256_file(Path(__file__).resolve().parents[2] / "scripts/evaluation/measure_target_runtime.py"),
        "runtime": runtime, "target_gpu_family": "NVIDIA A40", "target_driver": "new", "samples": actual_samples,
        "sidecar": {"pid": 42, "gpu_uuid": "GPU-test-only", "cuda_visible_devices": "GPU-test-only", "stub": False,
                    "encoder": provenance["encoder"], "encoder_state_sha256": provenance["encoder_state_sha256"],
                    "source_pin": provenance["source_pin"], **runtime}}
    measurement_path = tmp_path / "measurement.json"
    measurement_path.write_bytes(canonical(measurement))
    return {"provenance_path": prov_path, "fixture_path": fixture_path, "measurement_path": measurement_path,
            "output": tmp_path / "certificate.json", "eval_code_commit": "f" * 40}, provenance, runtime


def test_independent_recomputation_and_consumption(evidence):
    paths, provenance, runtime = evidence
    certificate = sign_target_contract(**paths)
    assert certificate["recomputed_verdict"]["samples"] == 2
    assert certificate["recomputed_verdict"]["max_abs"] == 0
    assert validate_target_contract(paths["output"], training_provenance=provenance, actual_runtime=runtime) == certificate
    changed = copy.deepcopy(runtime)
    changed["numerics"]["torch"] = "other"
    with pytest.raises(ValueError):
        validate_target_contract(paths["output"], training_provenance=provenance, actual_runtime=changed)


@pytest.mark.parametrize("mutation", ["self_pass", "missing_sample", "duplicate", "bad_source", "nan", "empty", "over_threshold", "wrong_dtype", "missing_layer", "missing_coverage"])
def test_bad_candidate_cannot_get_certificate(evidence, mutation):
    paths, _, _ = evidence
    measured = read_json(paths["measurement_path"])
    if mutation == "self_pass":
        measured.update(status="validated", metrics={"max_abs": 0}, verdict="PASS")
    elif mutation == "missing_sample":
        measured["samples"].pop()
    elif mutation == "duplicate":
        measured["samples"] = [measured["samples"][0]] * 2
    elif mutation == "bad_source":
        measured["sidecar"]["source_pin"] = {"sha": "0" * 64}
    elif mutation in ("nan", "empty", "over_threshold", "wrong_dtype"):
        array = np.zeros(0 if mutation == "empty" else 1024, dtype=np.float64 if mutation == "wrong_dtype" else np.float32)
        if mutation == "nan":
            array[0] = np.nan
        if mutation == "over_threshold":
            array += 2
        np.save(paths["output"].parent / "out0.npy", array)
        measured["samples"][0]["actual"]["sha256"] = sha256_file(paths["output"].parent / "out0.npy")
    else:
        fixture = read_json(paths["fixture_path"])
        if mutation == "missing_layer":
            fixture["samples"][1]["layer"] = "encoder"
        else:
            for sample in fixture["samples"]:
                sample["coverage"] = []
        paths["fixture_path"].write_bytes(canonical(fixture))
        measured["fixture_manifest_sha256"] = sha256_file(paths["fixture_path"])
        provenance = read_json(paths["provenance_path"])
        measured["motion_identity_sha256"] = descriptor_identity("motion", motion_descriptor(provenance, measured["fixture_manifest_sha256"]))
    paths["measurement_path"].write_bytes(canonical(measured))
    with pytest.raises(ValueError):
        sign_target_contract(**paths)
    assert not paths["output"].exists()


@pytest.mark.parametrize("key", ["torch", "dtype", "amp", "source_pin", "compute_cap"])
def test_only_two_hardware_fields_allowed(key):
    with pytest.raises(ValueError):
        compare_runtime({key: "old"}, {key: "new"}, {"target_gpu_family": "NVIDIA A40", "target_driver": "new"})


def test_candidate_is_not_certificate(evidence):
    paths, provenance, runtime = evidence
    with pytest.raises(ValueError, match="validated"):
        validate_target_contract(paths["measurement_path"], training_provenance=provenance, actual_runtime=runtime)


def test_identity_separates_scheduling_from_checkpoint():
    descriptor = {"training_run": "train", "target_step": 60000, "eval_code_commit": "a" * 40,
                  "benchmark_commit": "b" * 40, "job_spec": {"gpu": 1}, "motion_identity_sha256": "c" * 64}
    assert len(descriptor_identity("schedule", descriptor)) == 64
    with pytest.raises(ValueError):
        descriptor_identity("schedule", {**descriptor, "desired_jobs": 4})


def test_numeric_metrics_records_location_and_zero_denominator():
    result = numeric_metrics(np.zeros(3, np.float32), np.array([0, 1e-6, 0], np.float32))
    assert result["max_abs_index"] == [1]
    assert result["relative_zero_floor"] == 1e-12
    with pytest.raises(ValueError):
        numeric_metrics(np.array([], np.float32), np.array([], np.float32))
