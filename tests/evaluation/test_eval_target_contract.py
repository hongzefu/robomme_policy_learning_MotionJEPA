"""普通探索性运行合同：保留原子来源与软件精度，跨硬件数值不列准入。"""
import copy
import numpy as np
import pytest
from scripts.evaluation.runtime_contract import (
    canonical, compare_runtime, descriptor_identity, make_eval_runtime_contract,
    validate_eval_runtime_contract, numeric_metrics)
from scripts.evaluation.check_motion_consistency import check_repeatability


@pytest.fixture
def contract_inputs(tmp_path):
    numeric = {"gpu_name": "NVIDIA A100-SXM4-80GB", "driver": "old",
               "architecture": "sm80", "torch": "fixed", "dtype": "float32", "tf32": False}
    provenance = {"encoder": {"checkpoint_sha256": "a" * 64, "arch": "fixed_encoder"},
        "encoder_state_sha256": {"sha256": "b" * 64}, "source_pin": {"sha": "c" * 64},
        "vae": {"vae_state_sha256": "d" * 64}, "numerics": numeric,
        "stage2_compat": {"online_vae": {"vae_state_sha256": "d" * 64, "numerics": numeric}}}
    path = tmp_path / "runtime.json"
    record = make_eval_runtime_contract(provenance)
    path.write_bytes(canonical(record))
    actual = copy.deepcopy(record["training_runtime"])
    for section in (actual["numerics"], actual["online_vae"]["numerics"]):
        section.update(gpu_name="NVIDIA A40", driver="new", architecture="sm86")
    return path, provenance, actual


def test_runtime_hardware_observed_without_numeric_certificate(contract_inputs):
    path, provenance, actual = contract_inputs
    contract = validate_eval_runtime_contract(path, training_provenance=provenance, actual_runtime=actual)
    assert contract["cross_hardware_numeric"] == "not_required"
    assert contract["user_decision"] == "不需要做到跨硬件的数值性一致。"
    assert "status" not in contract and "fixture" not in contract
    assert validate_eval_runtime_contract(path, training_provenance=provenance) == contract


@pytest.mark.parametrize("key,value", [("torch", "changed"), ("dtype", "float16"), ("tf32", True)])
def test_software_and_precision_still_strict(contract_inputs, key, value):
    path, provenance, actual = contract_inputs
    actual["numerics"][key] = value
    with pytest.raises(ValueError):
        validate_eval_runtime_contract(path, training_provenance=provenance, actual_runtime=actual)


@pytest.mark.parametrize("section", ["encoder", "encoder_state_sha256", "source_pin", "vae"])
def test_atomic_sources_still_bound(contract_inputs, section):
    path, provenance, actual = contract_inputs
    provenance[section]["tampered"] = True
    with pytest.raises(ValueError):
        validate_eval_runtime_contract(path, training_provenance=provenance, actual_runtime=actual)


def test_old_validated_certificate_not_consumable(contract_inputs):
    path, provenance, actual = contract_inputs
    path.write_bytes(canonical({"schema": "eval-target-contract-v1", "status": "validated"}))
    with pytest.raises(ValueError):
        validate_eval_runtime_contract(path, training_provenance=provenance, actual_runtime=actual)


def test_default_strict_comparison_does_not_ignore_hardware():
    with pytest.raises(ValueError):
        compare_runtime({"gpu_name": "A100"}, {"gpu_name": "A40"})


def test_run_identity_binds_runtime_contract():
    descriptor = {"validation_model_identity_sha256": "a" * 64,
        "runtime_contract_sha256": "b" * 64, "benchmark_commit": "c" * 40,
        "metadata_sha256": "d" * 64, "model_seed": 42, "eval_parameters": {}}
    assert len(descriptor_identity("validation_run", descriptor)) == 64
    bad = dict(descriptor)
    bad["certificate_sha256"] = bad.pop("runtime_contract_sha256")
    with pytest.raises(ValueError):
        descriptor_identity("validation_run", bad)


def test_same_machine_arrays_only_check_repeatability():
    array = np.zeros((20, 8), np.float32)
    assert check_repeatability(array, array.copy())["max_abs"] == 0
    changed = array.copy()
    changed[1, 2] = 1e-6
    with pytest.raises(ValueError):
        check_repeatability(array, changed)
    changed[1, 2] = np.nan
    with pytest.raises(ValueError):
        numeric_metrics(array, changed)
