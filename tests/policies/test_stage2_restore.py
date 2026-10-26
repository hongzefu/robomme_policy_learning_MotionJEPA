"""恢复闸门 CPU 合同测试：拒绝伪造摘要、不完整交付及错误设备证据。"""

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/training/check_stage2_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("stage2_restore", SCRIPT)
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.write_bytes(restore.canonical(value))


def test_import_does_not_import_jax_or_launch_gpu():
    command = ("import importlib.util,sys; "
               f"s=importlib.util.spec_from_file_location('restore', {str(SCRIPT)!r}); "
               "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
               "assert 'jax' not in sys.modules; assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", command], check=True, timeout=15,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})


def test_fp32_difference_hidden_by_bfloat16_is_rejected():
    original = {"encoder": {"weight": np.array([1.00001], np.float32)},
                "frozen": np.array([2], dtype=ml_dtypes.bfloat16)}
    changed = copy.deepcopy(original)
    changed["encoder"]["weight"][0] = np.nextafter(original["encoder"]["weight"][0], np.float32(2))
    assert np.array_equal(original["encoder"]["weight"].astype(ml_dtypes.bfloat16),
                          changed["encoder"]["weight"].astype(ml_dtypes.bfloat16))
    summary, fp32 = restore.parameter_summary(original)
    assert set(fp32["leaves"]) == {"encoder/weight"}
    assert summary["leaves"]["frozen"]["dtype"] == "bfloat16"
    restore.verify_summary(fp32, restore.parameter_summary(original)[1], "原始 fp32")
    with pytest.raises(ValueError, match="原始字节不符"):
        restore.verify_summary(fp32, restore.parameter_summary(changed)[1], "原始 fp32")


@pytest.mark.parametrize("corrupt", ["missing", "extra", "shape", "dtype"])
def test_parameter_tree_rejects_each_structural_mismatch(corrupt):
    expected = {"a": np.ones((2, 3), np.float32)}
    actual = copy.deepcopy(expected)
    if corrupt == "missing":
        actual.clear()
    elif corrupt == "extra":
        actual["b"] = np.ones(1, np.float32)
    elif corrupt == "shape":
        actual["a"] = np.ones((3, 2), np.float32)
    else:
        actual["a"] = actual["a"].astype(np.float16)
    with pytest.raises(ValueError, match="完整参数树不符"):
        restore.verify_parameter_tree(expected, actual)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_original_ema_is_rejected(value):
    with pytest.raises(ValueError, match="非有限"):
        restore.parameter_summary({"a": np.array([value], np.float32)})


def reference_fixture(tmp_path):
    value = np.array([[1.5, -2]], dtype=ml_dtypes.bfloat16)
    path = tmp_path / "inputs.npz"
    np.savez(path, leaf_0000=value.view(np.uint16))
    record = {"path": path.name, "sha256": restore.sha256_file(path), "none_paths": [["optional"]],
              "leaves": {"leaf_0000": {"path": ["image", "base"], "shape": [1, 2], "dtype": "bfloat16",
                                        "sha256": digest(value.tobytes())}}}
    return value, record


def test_reference_bfloat16_bitstrings_and_none_roundtrip(tmp_path):
    wanted, record = reference_fixture(tmp_path)
    actual = restore.load_reference_tree(tmp_path, record)
    assert actual["optional"] is None
    assert actual["image"]["base"].dtype == ml_dtypes.bfloat16
    assert actual["image"]["base"].tobytes() == wanted.tobytes()


@pytest.mark.parametrize("corrupt", ["payload", "leaf_sha", "dtype", "shape", "duplicate", "prefix", "extra"])
def test_reference_tree_corruptions_fail(tmp_path, corrupt):
    _, record = reference_fixture(tmp_path)
    leaf = record["leaves"]["leaf_0000"]
    if corrupt == "payload":
        record["sha256"] = "0" * 64
    elif corrupt == "leaf_sha":
        leaf["sha256"] = "0" * 64
    elif corrupt == "dtype":
        leaf["dtype"] = "float32"
    elif corrupt == "shape":
        leaf["shape"] = [2, 1]
    elif corrupt == "duplicate":
        record["none_paths"].append(["image", "base"])
    elif corrupt == "prefix":
        record["none_paths"].append(["image"])
    else:
        np.savez(tmp_path / "inputs.npz", leaf_0000=np.ones((1, 2), np.uint16), extra=np.ones(1))
        record["sha256"] = restore.sha256_file(tmp_path / "inputs.npz")
    with pytest.raises(ValueError):
        restore.load_reference_tree(tmp_path, record)


@pytest.mark.parametrize("corrupt", ["bit", "negative_zero", "nan", "dtype"])
def test_output_comparison_requires_exact_bits_and_dtype(corrupt):
    expected = {"actions": np.array([0, 1], np.float32)}
    actual = copy.deepcopy(expected)
    if corrupt == "bit":
        actual["actions"][1] = np.nextafter(np.float32(1), np.float32(2))
    elif corrupt == "negative_zero":
        actual["actions"][0] = -0.0
    elif corrupt == "nan":
        actual["actions"][0] = np.nan
    else:
        actual["actions"] = actual["actions"].astype(np.float64)
    assert restore.compare_outputs(expected, expected) == 0
    with pytest.raises(ValueError):
        restore.compare_outputs(expected, actual)


def delivery_fixture(tmp_path):
    root = tmp_path / "delivery"
    root.mkdir()
    files = {}
    for name in restore.ROOT_FILES | {"motion_snapshot.ready.json", "60000/params/_METADATA", "60000/params/d/0",
                                      "60000/assets/robomme/norm_stats.json", "encoder_dependencies/run/encoder.pt"}:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = {"sha256": restore.sha256_file(path), "bytes": path.stat().st_size}
    manifest = {"schema": "stage2-delivery-manifest-v1", "files": files}
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    return root, path, manifest


def test_delivery_verifies_every_listed_file_including_dependencies(tmp_path):
    root, path, manifest = delivery_fixture(tmp_path)
    assert restore.verify_delivery_manifest(root, path, root / "60000") == manifest
    (root / "encoder_dependencies/run/encoder.pt").write_bytes("篡改".encode())
    with pytest.raises(ValueError):
        restore.verify_delivery_manifest(root, path, root / "60000")


@pytest.mark.parametrize("corrupt", ["missing_root", "missing_metadata", "extra_checkpoint", "size", "sha"])
def test_incomplete_delivery_is_rejected(tmp_path, corrupt):
    root, path, manifest = delivery_fixture(tmp_path)
    if corrupt == "missing_root":
        del manifest["files"]["online_restore_inputs.npz"]
    elif corrupt == "missing_metadata":
        del manifest["files"]["60000/params/_METADATA"]
    elif corrupt == "extra_checkpoint":
        (root / "60000/params/new").write_bytes(b"new")
    elif corrupt == "size":
        manifest["files"]["60000/params/d/0"]["bytes"] += 1
    else:
        manifest["files"]["60000/params/d/0"]["sha256"] = "0" * 64
    write_json(path, manifest)
    with pytest.raises(ValueError):
        restore.verify_delivery_manifest(root, path, root / "60000")


@pytest.mark.parametrize("name", ["../secret", "/tmp/secret", "a/../secret", "a//b", "./a", "a\\b"])
def test_delivery_rejects_path_escape_and_aliases(tmp_path, name):
    with pytest.raises(ValueError):
        restore.inside(tmp_path, name)


def test_delivery_rejects_file_and_directory_symlinks(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    (target / "file").write_bytes(b"real")
    (tmp_path / "link").symlink_to(target, target_is_directory=True)
    (tmp_path / "file_link").symlink_to(target / "file")
    for name in ("link/file", "file_link"):
        with pytest.raises(ValueError, match="符号链接"):
            restore.inside(tmp_path, name)


def online_fixture(tmp_path):
    es = 17
    arrays = {"images": np.zeros((es + 65, 256, 256, 3), np.uint8),
              "states": np.ones((es + 65, 8), np.float32), "wrist_image": np.zeros((256, 256, 3), np.uint8)}
    np.savez_compressed(tmp_path / "online_restore_inputs.npz", **arrays)
    record = {"schema": "stage2-online-restore-inputs-v1", "exec_start_idx": es, "exec_batch_lengths": [33, 32],
              "prompt": "移动方块", "sources": [{"path": "/diagnostic/source.pkl", "sha256": "1" * 64}],
              "payload": {"path": "online_restore_inputs.npz", "sha256": restore.sha256_file(tmp_path / "online_restore_inputs.npz")},
              "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": digest(value.tobytes())}
                         for name, value in arrays.items()},
              "tokenizer": {"uri": restore.TOKENIZER_URI, "sha256": restore.read_json(
                  restore.REPO_ROOT / "scripts/assets/ASSETS_LOCK.json")["assets"]["paligemma_tokenizer"]["sha256"]}}
    write_json(tmp_path / "online_restore_inputs.json", record)
    return record, arrays


def test_online_inputs_do_not_read_diagnostic_source_paths(tmp_path):
    record, arrays = online_fixture(tmp_path)
    actual_record, actual_arrays = restore.load_online_inputs(tmp_path)
    assert actual_record == record
    assert actual_arrays["states"].tobytes() == arrays["states"].tobytes()


@pytest.mark.parametrize("corrupt", ["batch", "tokenizer", "tokenizer_sha", "image_sha", "state_shape", "payload"])
def test_online_inputs_fail_closed(tmp_path, corrupt):
    record, _ = online_fixture(tmp_path)
    if corrupt == "batch":
        record["exec_batch_lengths"] = [32, 33]
    elif corrupt == "tokenizer":
        del record["tokenizer"]
    elif corrupt == "tokenizer_sha":
        record["tokenizer"]["sha256"] = "0" * 64
    elif corrupt == "image_sha":
        record["arrays"]["images"]["sha256"] = "0" * 64
    elif corrupt == "state_shape":
        record["arrays"]["states"]["shape"] = [82, 7]
    else:
        record["payload"]["sha256"] = "0" * 64
    write_json(tmp_path / "online_restore_inputs.json", record)
    with pytest.raises(ValueError):
        restore.load_online_inputs(tmp_path)


@pytest.mark.parametrize("output", ["10, GPU-A\n11, GPU-B\n", "10, GPU-B\n11, GPU-A\n",
                                    "10, GPU-A\n11, GPU-A\n", "10, GPU-A\n", "10, GPU-A\n10, GPU-C\n11, GPU-B\n"])
def test_physical_gpu_mapping_requires_actual_parent_and_sidecar_processes(output):
    inventory = {0: {"uuid": "GPU-A", "name": "A100"}, 1: {"uuid": "GPU-B", "name": "A100"}}
    if output == "10, GPU-A\n11, GPU-B\n":
        assert restore.verify_process_gpus(10, 11, inventory, 1, output)["sidecar"]["physical_index"] == 1
    else:
        with pytest.raises(ValueError, match="实际 GPU"):
            restore.verify_process_gpus(10, 11, inventory, 1, output)


def test_cpu_cli_records_failure_without_real_restore(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    report_path = tmp_path / "report.json"
    status = restore.main(["--mode", "smoke", "--checkpoint", str(tmp_path / "run/19"),
                           "--reference", str(tmp_path / "reference.json"), "--online-gpu", "1", "--out", str(report_path)])
    assert status == 1
    report = restore.read_json(report_path)
    assert report["status"] == "FAIL"
    assert "online" not in report
    assert "PASS" not in capsys.readouterr().out
    assert not list(tmp_path.glob("*.tmp"))


def test_restore_uses_verified_worker_instead_of_uv_launcher():
    client = SimpleNamespace(worker_pid=11, _proc=SimpleNamespace(pid=10),
                             provenance={"pid": 11, "cuda_visible_devices": "1", "gpu_uuid": "GPU-B"})
    inventory = {1: {"uuid": "GPU-B"}}
    assert restore.verified_sidecar_pid(client, 1, inventory) == 11
    client.provenance["pid"] = 10
    with pytest.raises(ValueError, match="PID"):
        restore.verified_sidecar_pid(client, 1, inventory)


def test_restore_propagates_worker_ownership_rejection():
    class Client:
        provenance = {"pid": 11, "cuda_visible_devices": "1", "gpu_uuid": "GPU-B"}
        @property
        def worker_pid(self):
            raise ValueError("实际 worker 不属于本次启动器")
    with pytest.raises(ValueError, match="不属于"):
        restore.verified_sidecar_pid(Client(), 1, {1: {"uuid": "GPU-B"}})


@pytest.mark.parametrize("mode,extra", [("smoke", []), ("delivery", ["--reference", "r"]),
                                       ("smoke", ["--reference", "r", "--expected-manifest", "m"])])
def test_cli_modes_cannot_mix_reference_and_delivery(tmp_path, mode, extra):
    with pytest.raises(SystemExit):
        restore.parse_args(["--mode", mode, "--checkpoint", str(tmp_path / "run/19"),
                            "--online-gpu", "1", "--out", str(tmp_path / "report"), *extra])


def test_explicit_runtime_cache_is_required(tmp_path, monkeypatch):
    monkeypatch.delenv("MMEVLA_V1_STORE", raising=False)
    with pytest.raises(ValueError, match="MMEVLA_V1_STORE"):
        restore.runtime_dependencies({}, {})


def test_runtime_cache_sources_and_hash_are_explicit(tmp_path, monkeypatch):
    v1, cache = tmp_path / "v1", tmp_path / "openpi"
    v1.mkdir()
    (cache / "big_vision").mkdir(parents=True)
    tokenizer = cache / "big_vision/paligemma_tokenizer.model"
    tokenizer.write_bytes(b"tokenizer")
    monkeypatch.setenv("MMEVLA_V1_STORE", str(v1))
    monkeypatch.setenv("OPENPI_DATA_HOME", str(cache))
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_ASSETS_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(name, raising=False)
    record = {"tokenizer": {"uri": restore.TOKENIZER_URI, "sha256": restore.sha256_file(tokenizer)}}
    provenance = {"stage2_compat": {"online_vae": {"vae_id": "Wan-AI/model", "vae_state_sha256": "1" * 64}}}
    result = restore.runtime_dependencies(record, provenance)
    assert result["tokenizer"]["actual_path"] == str(tokenizer)
    assert result["vae"]["uri"] == "https://huggingface.co/Wan-AI/model"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hidden"))
    with pytest.raises(ValueError, match="绕过显式 VAE 缓存"):
        restore.runtime_dependencies(record, provenance)


def test_orbax_preserves_original_dtype_before_ema_comparison(tmp_path):
    # 只恢复两个微型 CPU 数组，验证生产 restore_params 不会先隐式降为 BF16。
    import orbax.checkpoint as ocp
    from openpi.models.model import restore_params

    value = np.array([1.00001, 2.00002], np.float32)
    frozen = np.array([3, 4], dtype=ml_dtypes.bfloat16)
    path = tmp_path / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, {"params": {"trained": {"value": value}, "frozen": {"value": frozen}}})
    restored = restore_params(path, restore_type=np.ndarray, dtype=None)
    assert restored["trained"].dtype == np.float32
    assert restored["frozen"].dtype == ml_dtypes.bfloat16
    restore.verify_parameter_tree({"trained": value, "frozen": frozen}, restored)
    assert restore.parameter_summary(restored) == restore.parameter_summary({"trained": value, "frozen": frozen})


def test_nnx_expected_training_dtype_and_single_device_inference():
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.shared import nnx_utils

    class Tiny(nnx.Module):
        def __init__(self):
            self.frozen = nnx.Param(jnp.ones(2, jnp.float32))
            self.trained = nnx.Param(jnp.ones(3, jnp.float32))

    frozen_filter = nnx_utils.PathRegex("frozen")
    shape_model = nnx.eval_shape(Tiny)
    expected = nnx_utils.state_map(nnx.state(shape_model), frozen_filter,
        lambda item: item.replace(jax.ShapeDtypeStruct(item.value.shape, jnp.bfloat16))).to_pure_dict()
    assert str(expected["frozen"].dtype) == "bfloat16"
    assert str(expected["trained"].dtype) == "float32"
    model = Tiny()
    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    named = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = jax.tree.map(lambda value: jax.device_put(value.astype(jnp.bfloat16), named), nnx.state(model))
    state = jax.tree.map(lambda value: jax.device_put(value, jax.devices()[0]), state)
    nnx.update(model, state)
    assert all(isinstance(value.sharding, jax.sharding.SingleDeviceSharding) for value in jax.tree.leaves(state))
    assert str(model.frozen.value.dtype) == "bfloat16"


def test_json_duplicate_keys_and_nonfinite_metadata_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    for text in ('{"step":19,"step":60000}', '{"value":NaN}', '{"value":Infinity}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            restore.read_json(path)


@pytest.mark.parametrize("uuid", ["GPU-B", "B"])
def test_restore_uuid_prefix_is_format_only(uuid):
    client = SimpleNamespace(worker_pid=11,
                             provenance={"pid": 11, "cuda_visible_devices": "1", "gpu_uuid": uuid})
    inventory = {0: {"uuid": "GPU-A"}, 1: {"uuid": "GPU-B"}}
    assert restore.verified_sidecar_pid(client, 1, inventory) == 11
    result = restore.verify_process_gpus(10, 11, inventory, 1, f"10, A\n11, {uuid}\n")
    assert result["sidecar"]["pid"] == 11
    client.provenance["gpu_uuid"] = "GPU-C"
    with pytest.raises(ValueError, match="GPU/PID"):
        restore.verified_sidecar_pid(client, 1, inventory)
