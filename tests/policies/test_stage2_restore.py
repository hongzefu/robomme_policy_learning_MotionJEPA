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


def delivery_fixture(tmp_path, step=60000):
    root = tmp_path / "delivery"
    root.mkdir()
    files = {}
    for name in restore.ROOT_FILES | {"motion_snapshot.ready.json", f"{step}/params/_METADATA", f"{step}/params/d/0",
                                      f"{step}/assets/robomme/norm_stats.json", "encoder_dependencies/run/encoder.pt"}:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = {"sha256": restore.sha256_file(path), "bytes": path.stat().st_size}
    manifest = {"schema": "stage2-delivery-manifest-v1", "files": files}
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    return root, path, manifest


@pytest.mark.parametrize("step", [59999, 60000, 70000])
def test_delivery_verifies_every_listed_file_including_dependencies(tmp_path, step):
    root, path, manifest = delivery_fixture(tmp_path, step)
    assert restore.verify_delivery_manifest(root, path, root / str(step)) == manifest
    (root / "encoder_dependencies/run/encoder.pt").write_bytes("篡改".encode())
    with pytest.raises(ValueError):
        restore.verify_delivery_manifest(root, path, root / str(step))


@pytest.mark.parametrize("corrupt", ["missing_root", "missing_metadata", "extra_checkpoint", "size", "sha"])
@pytest.mark.parametrize("step", [59999, 60000, 70000])
def test_incomplete_delivery_is_rejected(tmp_path, corrupt, step):
    root, path, manifest = delivery_fixture(tmp_path, step)
    if corrupt == "missing_root":
        del manifest["files"]["online_restore_inputs.npz"]
    elif corrupt == "missing_metadata":
        del manifest["files"][f"{step}/params/_METADATA"]
    elif corrupt == "extra_checkpoint":
        (root / f"{step}/params/new").write_bytes(b"new")
    elif corrupt == "size":
        manifest["files"][f"{step}/params/d/0"]["bytes"] += 1
    else:
        manifest["files"][f"{step}/params/d/0"]["sha256"] = "0" * 64
    write_json(path, manifest)
    with pytest.raises(ValueError):
        restore.verify_delivery_manifest(root, path, root / str(step))


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


@pytest.mark.parametrize("extra,step", [([], 60000), (["--delivery-step", "60000"], 60000),
                                       (["--delivery-step", "70000"], 70000), (["--delivery-step", "59999"], 59999)])
def test_delivery_step_is_explicit_with_60k_default(tmp_path, extra, step):
    args = restore.parse_args(["--mode", "delivery", "--checkpoint", str(tmp_path / f"run/{step}"),
                               "--expected-manifest", str(tmp_path / "manifest.json"),
                               "--online-gpu", "1", "--out", str(tmp_path / "report"), *extra])
    assert args.delivery_step == step


@pytest.mark.parametrize("mode,step", [("smoke", "60000"), ("smoke", "70000"),
                                      ("delivery", "19"), ("delivery", "65000"), ("delivery", "80000")])
def test_cli_rejects_inapplicable_delivery_step(tmp_path, mode, step):
    source = ["--reference", "r"] if mode == "smoke" else ["--expected-manifest", "m"]
    with pytest.raises(SystemExit):
        restore.parse_args(["--mode", mode, "--checkpoint", str(tmp_path / "run/19"),
                            "--online-gpu", "1", "--out", str(tmp_path / "report"),
                            "--delivery-step", step, *source])


@pytest.mark.parametrize("step", [59999, 60000, 70000])
def test_delivery_selected_step_reaches_verified_manifest_before_runtime(tmp_path, monkeypatch, step):
    root, path, manifest = delivery_fixture(tmp_path, step)
    args = restore.parse_args(["--mode", "delivery", "--checkpoint", str(root / str(step)),
                               "--expected-manifest", str(path), "--delivery-step", str(step),
                               "--online-gpu", "1", "--out", str(tmp_path / "report")])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    def stop_before_runtime(actual_root, delivery):
        assert actual_root == root
        assert delivery == manifest
        raise RuntimeError("已核交付清单，在导入 GPU 运行时前停止")

    monkeypatch.setattr(restore, "verify_snapshot", stop_before_runtime)
    report = {}
    with pytest.raises(RuntimeError, match="在导入 GPU 运行时前停止"):
        restore.restore_and_check(args, report)
    assert report["step"] == step
    assert report["delivery"] == {"manifest_sha256": restore.sha256_file(path),
                                  "files": len(manifest["files"]), "sha256_mismatch": 0}


@pytest.mark.parametrize("mode,selected,actual", [("delivery", 60000, 70000), ("delivery", 70000, 60000),
                                                ("smoke", None, 70000)])
def test_checkpoint_directory_must_match_selected_step(tmp_path, monkeypatch, mode, selected, actual):
    checkpoint = tmp_path / f"run/{actual}"
    checkpoint.mkdir(parents=True)
    source = (["--reference", "r"] if mode == "smoke" else
              ["--expected-manifest", "m", "--delivery-step", str(selected)])
    args = restore.parse_args(["--mode", mode, "--checkpoint", str(checkpoint),
                               "--online-gpu", "1", "--out", str(tmp_path / "report"), *source])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    with pytest.raises(ValueError, match=f"只验收 step{19 if mode == 'smoke' else selected}"):
        restore.restore_and_check(args, {})


@pytest.mark.parametrize("step", [59999, 60000, 70000])
def test_delivery_failure_keeps_selected_step_label(tmp_path, monkeypatch, capsys, step):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    report_path = tmp_path / "report.json"
    status = restore.main(["--mode", "delivery", "--checkpoint", str(tmp_path / f"run/{step}"),
                           "--expected-manifest", str(tmp_path / "manifest.json"), "--delivery-step", str(step),
                           "--online-gpu", "1", "--out", str(report_path)])
    report = restore.read_json(report_path)
    assert status == 1
    assert report["status"] == "FAIL"
    assert report["verdict"] == ("CKPT_DELIVERY_READY=FAIL" if step == 59999 else f"CKPT_{step // 1000}K_READY=FAIL")
    assert capsys.readouterr().out.strip() == report["verdict"]
    assert "online" not in report


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


@pytest.mark.parametrize("variables", [("MMEVLA_JAX_CACHE_DIR",), ("JAX_COMPILATION_CACHE_DIR",),
                                       ("MMEVLA_JAX_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR"), ()])
def test_restore_activates_requested_training_compilation_cache(tmp_path, monkeypatch, variables):
    import jax

    previous = jax.config.jax_compilation_cache_dir
    for name in ("MMEVLA_JAX_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    for name in variables:
        monkeypatch.setenv(name, str(tmp_path))
    try:
        jax.config.update("jax_compilation_cache_dir", None)
        report = restore.configure_compilation_cache(jax.config)
        expected = str(tmp_path) if variables else None
        assert jax.config.jax_compilation_cache_dir == expected
        assert report["path"] == expected
        assert report["enabled"] == bool(variables)
        assert report["environment"] == {name: str(tmp_path) for name in variables}
        assert report["xla_caches"] == jax.config.jax_persistent_cache_enable_xla_caches
    finally:
        jax.config.update("jax_compilation_cache_dir", previous)


@pytest.mark.parametrize("invalid", ["conflict", "missing", "relative", "disabled"])
def test_restore_rejects_ambiguous_or_unusable_compilation_cache(tmp_path, monkeypatch, invalid):
    import jax

    previous = jax.config.jax_compilation_cache_dir
    enabled = jax.config.jax_enable_compilation_cache
    monkeypatch.setenv("MMEVLA_JAX_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    if invalid == "conflict":
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(other))
    elif invalid == "missing":
        monkeypatch.setenv("MMEVLA_JAX_CACHE_DIR", str(tmp_path / "missing"))
    elif invalid == "relative":
        monkeypatch.setenv("MMEVLA_JAX_CACHE_DIR", ".")
    else:
        jax.config.update("jax_enable_compilation_cache", False)
    try:
        with pytest.raises(ValueError):
            restore.configure_compilation_cache(jax.config)
        assert jax.config.jax_compilation_cache_dir == previous
    finally:
        jax.config.update("jax_enable_compilation_cache", enabled)


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


@pytest.mark.parametrize("fusion,dim", [("context", 2048), ("modulation", 1024)])
@pytest.mark.parametrize("schedule", [[33, 32], [1, 16, 16, 16, 16]])
def test_restore_online_real_policy_dplus1_split896(monkeypatch, fusion, dim, schedule):
    """真实pack_buffer/FrameSampMemory/policy组包；sidecar与动作网络用确定性CPU夹具。"""
    import functools
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from mme_vla_suite.policies import policy as module
    from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
    hc = OmegaConf.load(SCRIPT.parents[2] / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-orig16.yaml")
    hc.integration_type, hc.memory_token_dim = fusion, dim
    monkeypatch.setattr(module.nnx_utils, "module_jit", lambda fn: fn)
    monkeypatch.setattr(module, "FrameSampMemory", functools.partial(FrameSampMemory, max_steps=128))
    monkeypatch.setattr(restore, "verified_sidecar_pid", lambda *args: 11)
    monkeypatch.setattr(restore, "verify_process_gpus", lambda *args: {"scope": "CPU夹具"})

    class Client:
        stub = False
        alive = True
        n_calls = 0
        provenance = {"scope": "CPU夹具"}

        def __call__(self, frames, start):
            self.n_calls += 1
            return np.arange(1024, dtype=np.float32) + start

    model = SimpleNamespace(history_config=hc, action_horizon=20, sample_actions=lambda *a: None,
        vision_encode=lambda images: jnp.zeros((*images.shape[:2], 256, 2048), jnp.float32))
    policy = module.MME_VLA_Policy(model, motion_enc_fn=Client(),
        norm_stats={"state": SimpleNamespace(mean=np.zeros(8), std=np.ones(8))})
    seen = []
    original_add = policy.add_buffer

    def add(value):
        seen.append((value["images"].shape[0], value["exec_start_idx"], int(value["images"][0, 0, 0, 0, 0])))
        original_add(value)

    policy.add_buffer = add
    policy.infer = lambda observation: {"actions": np.zeros((20, 8), np.float32)}
    arrays = dict(images=np.broadcast_to(np.arange(82, dtype=np.uint8)[:, None, None, None], (82, 256, 256, 3)).copy(),
                  states=np.zeros((82, 8), np.float32), wrist_image=np.zeros((256, 256, 3), np.uint8))
    result = restore.run_online(policy, dict(exec_start_idx=17, exec_batch_lengths=schedule, prompt="夹具"), arrays, 1, {})
    assert seen == [(18, 17, 0), (16, 17, 18), (16, 17, 34), (16, 17, 50), (16, 17, 66)]
    assert result["memory_tokens"] == 896 and result["sidecar_calls"] == 3 and result["reset"] == 1
    assert result["batches"][2]["motion_keys"] == [0, 17]
    assert result["batches"][4]["motion_keys"] == [0, 17, 49]


@pytest.mark.parametrize("tokens,layout", [(16, "4x4"), (64, "8x8")])
def test_raw_h5_portable_state_source_and_layout(tmp_path, monkeypatch, tokens, layout):
    import h5py
    import pickle
    from mme_vla_suite.datastore import framesamp_store as fs
    from mme_vla_suite.datastore import manifest as manifests
    spec = importlib.util.spec_from_file_location("o16_online_fixture", SCRIPT.with_name("check_stage2_online.py"))
    online = importlib.util.module_from_spec(spec); spec.loader.exec_module(online)
    parent = tmp_path / "dataset"; source = parent / "source/data"
    frames = parent / f"framesamp-{layout}"; raw = tmp_path / "raw"
    for folder in (parent / "meta", frames / "meta", source, raw): folder.mkdir(parents=True)
    manifest = parent / "meta/episode_manifest.json"; manifest.write_text("{}")
    ep = dict(h5_file="record_dataset_Task.h5", raw_ep_idx=2, global_episode_idx=0,
              total_sample_offset=0, exec_sample_offset=0, exec_start_idx=17, num_timesteps=82)
    monkeypatch.setattr(manifests, "load_manifest", lambda path: dict(episodes=[ep]))
    states = np.arange(82 * 8, dtype=np.float32).reshape(82, 8)
    state_path = frames / fs.STATE_TABLE_RELPATH; state_path.write_bytes(states.tobytes())
    (frames / fs.META_RELPATH).write_text("{}")
    fm = SimpleNamespace(status="verified", manifest_scope="full", manifest_sha256="a" * 64, num_rows=82,
        source_dataset_root=str(source.parent), raw=dict(layout=f"framesamp-{layout}-v1", tables={fs.STATE_KEY:
            dict(sha256=online.V.sha_file(state_path), row_shape=[8], dtype="float32")}))
    monkeypatch.setattr(fs.StoreMeta, "load", lambda root: fm)
    h5 = raw / ep["h5_file"]
    with h5py.File(h5, "w") as handle:
        for index in range(17): handle.create_dataset(f"episode_2/timestep_{index}/obs/front_rgb", data=np.full((256, 256, 3), index, np.uint8))
    sha = online.V.sha_file(h5)
    inputs = parent / "meta/input_manifest.json"
    write_json(inputs, dict(files={ep["h5_file"]: dict(size=h5.stat().st_size, sha256=sha)}))
    for index in range(65):
        (source / f"{index}.pkl").write_bytes(pickle.dumps(dict(epis_idx=np.array([0], np.int32),
            step_idx=np.array([17 + index], np.int32), exec_start_idx=np.array([17], np.int32),
            is_demo=np.array([False], bool), image=np.full((256, 256, 3), 17 + index, np.uint8),
            state=states[17 + index], wrist_image=np.zeros((256, 256, 3), np.uint8), prompt="夹具")))
    tokenizer = tmp_path / "models/big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True); tokenizer.write_bytes(b"test")
    samples = dict(windows=[dict(key="Task_ep2_demo", segment_kind="demo",
        sources=[dict(kind="raw_h5", path=str(h5), sha256=sha, episode=2)])])
    kwargs = dict(framesamp_store=frames, h5_root=raw, input_manifest=inputs, history=dict(token_per_image=tokens))
    result = online.prepare_restore_inputs(SimpleNamespace(manifest_path=str(manifest), manifest_sha256="a" * 64),
        samples, tmp_path, tmp_path, **kwargs)
    record = online.V.read_json(result["path"])
    assert record["exec_batch_lengths"] == [1, 16, 16, 16, 16]
    assert record["demo_state_origin"]["path"] == f"framesamp-{layout}/{fs.STATE_TABLE_RELPATH}"
    with np.load(result["payload"]["path"]) as payload:
        np.testing.assert_array_equal(payload["states"], states)
        np.testing.assert_array_equal(payload["images"][:, 0, 0, 0], np.arange(82))
    fm.raw["layout"] = "framesamp-8x8-v1" if layout == "4x4" else "framesamp-4x4-v1"
    with pytest.raises(ValueError, match="layout"):
        online.prepare_restore_inputs(SimpleNamespace(manifest_path=str(manifest), manifest_sha256="a" * 64),
            samples, tmp_path, tmp_path, **kwargs)


@pytest.mark.parametrize("tokens", [16, 64])
@pytest.mark.parametrize("split", [False, True])
def test_g4_state_machine_both_layouts_and_readouts(tokens, split):
    from mme_vla_suite.policies import motion_protocol as protocol
    spec = importlib.util.spec_from_file_location("o16_g4_machine", SCRIPT.with_name("check_stage2_online.py"))
    online = importlib.util.module_from_spec(spec); spec.loader.exec_module(online)

    class Client:
        alive = True

        def __call__(self, frames, start):
            assert protocol.stub_decode(frames)[0] == start
            return np.full(1024, start, np.float32)

    motion = dict(stride=32, window_frames=33, budget=384 if split else 96, frame_size=256,
        pos_dim=256, dim=256 if split else 1024, demo_min_real_frames=17, demo_tail_pad="repeat_last",
        source_shape=[1, 1024], num_tokens=4 if split else 1,
        readout="stage2_pool4_split_v1" if split else "stage2_pool4_concat_v1")
    result = online.check_state_machine(Client(), motion, token_per_image=tokens)
    assert result["stub_windows"] == 6 and result["first_packet"] == "D+1" and result["max_followup_frames"] == 16


@pytest.mark.parametrize("cached", [False, True])
def test_g4_cached_verdict_does_not_claim_numeric_parity(tmp_path, monkeypatch, capsys, cached):
    spec = importlib.util.spec_from_file_location("o16_g4_verdict", SCRIPT.with_name("check_stage2_online.py"))
    online = importlib.util.module_from_spec(spec); spec.loader.exec_module(online)
    record = dict(metrics={"windows": 64}, kind="published-cache-with-local-fill" if cached else "local-restore")
    monkeypatch.setattr(online, "run", lambda args: record)
    assert online.main(["--store", "fixture", "--samples", "fixture", "--history", "fixture",
                        "--online-gpu", "1", "--out", str(tmp_path / "G4.json")]) == 0
    output = capsys.readouterr().out
    assert ("ONLINE_INPUT_IDENTITY=PASS" in output) == cached
    assert ("ONLINE_OFFLINE_PARITY=PASS" in output) != cached
