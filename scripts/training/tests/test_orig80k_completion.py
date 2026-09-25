"""完成器负例覆盖普通日志、尾窗、身份、保存终态与真实数组摘要。"""

from copy import deepcopy
import datetime
from pathlib import Path
import sys
import uuid

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_orig80k_completion as checker


def fixture(mode="prod"):
    steps = checker.MODE_STEPS[mode]
    run, head = "test-orig80k-" + mode, "a" * 40
    scalar = {"dec": 1.0, "hex": float(1).hex()}
    complete = {"train_config": {"fields": {
        "name": checker.CONFIG, "exp_name": run, "num_train_steps": steps, "batch_size": 64,
        "num_workers": 4, "fsdp_devices": 4, "seed": 42, "log_interval": 100,
        "save_interval": 10000, "keep_period": 10000, "ema_decay": .999, "wandb_enabled": True,
        "overwrite": False, "resume": False}}}
    start = {"run_name": run, "head": head, "budget": 512, "num_train_steps": steps,
             "log_interval": 100, "save_interval": 10000, "keep_period": 10000,
             "run_uuid": str(uuid.uuid4()), "wandb_run_id": "fixture-id", "complete": complete}
    tail = [{"step": step, "scalars": {key: {**scalar, "finite": True} for key in checker.SCALARS}}
            for step in range((steps - 1) // 100 * 100 + 1, steps)]
    arrays = {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3)}
    final = {**deepcopy(start), "state_step": steps, "loop_step": steps - 1, "checkpoint_step": steps - 1,
             "tail": tail, "tail_means": dict.fromkeys(checker.SCALARS, 1.0),
             "ema_leaves": checker.summarize_tree(arrays)}
    wait = {"head": head, "run_uuid": start["run_uuid"], "wait_until_finished": True}
    metrics = [{"step": step, "wall_time": float(step), **{key: deepcopy(scalar) for key in checker.SCALARS}}
               for step in range(0, steps, 100)]
    return start, final, wait, metrics, ["EXIT_CODE=0"], checker.expected_checkpoints(mode), arrays


def validate(records, mode="prod"):
    checker.validate_records(*records[:6], mode=mode, run="test-orig80k-" + mode, head="a" * 40)


@pytest.mark.parametrize("mode", ["prod", "perf", "smoke"])
def test_complete_mode_records_and_arrays(mode):
    records = fixture(mode)
    validate(records, mode)
    checker.validate_arrays(records[1], records[6])
    assert len(records[1]["tail"]) == (19 if mode == "smoke" else 99)
    assert records[5] == ([*range(10000, 80000, 10000), 79999] if mode == "prod" else [checker.MODE_STEPS[mode] - 1])


@pytest.mark.parametrize("fault", [
    "missing_checkpoint", "extra_checkpoint", "duplicate_log", "missing_log", "log_key", "log_nan", "log_hex",
    "tail_missing", "tail_duplicate", "tail_nan", "tail_inf", "tail_key", "tail_finite_missing", "tail_mean",
    "wrong_state_step", "wrong_run", "wrong_head", "wrong_uuid", "missing_wait", "wrong_wait_head",
    "duplicate_exit", "missing_exit", "failed_exit", "ema_nan", "ema_empty", "wrong_config", "changed_complete",
])
def test_invalid_completion_rejected(fault):
    records = list(fixture())
    start, final, wait, metrics, exits, names, _ = records
    if fault == "missing_checkpoint":
        names.pop()
    elif fault == "extra_checkpoint":
        names.append(80000)
    elif fault == "duplicate_log":
        metrics.append(deepcopy(metrics[-1]))
    elif fault == "missing_log":
        metrics.pop()
    elif fault == "log_key":
        metrics[-1].pop("loss")
    elif fault == "log_nan":
        metrics[-1]["loss"] = {"dec": float("nan"), "hex": "nan"}
    elif fault == "log_hex":
        metrics[-1]["loss"]["hex"] = float(2).hex()
    elif fault == "tail_missing":
        final["tail"].pop()
    elif fault == "tail_duplicate":
        final["tail"].append(deepcopy(final["tail"][-1]))
    elif fault in ("tail_nan", "tail_inf"):
        value = float("nan" if fault == "tail_nan" else "inf")
        final["tail"][-1]["scalars"]["loss"] = {"finite": True, "dec": value, "hex": value.hex()}
    elif fault == "tail_key":
        final["tail"][-1]["scalars"].pop("loss")
    elif fault == "tail_finite_missing":
        final["tail"][-1]["scalars"]["loss"].pop("finite")
    elif fault == "tail_mean":
        final["tail_means"]["loss"] = 2.0
    elif fault == "wrong_state_step":
        final["state_step"] -= 1
    elif fault == "wrong_run":
        final["run_name"] = "other"
    elif fault == "wrong_head":
        final["head"] = "b" * 40
    elif fault == "wrong_uuid":
        final["run_uuid"] = str(uuid.uuid4())
    elif fault == "missing_wait":
        wait.pop("wait_until_finished")
    elif fault == "wrong_wait_head":
        wait["head"] = "b" * 40
    elif fault == "duplicate_exit":
        exits.append("EXIT_CODE=0")
    elif fault == "missing_exit":
        exits.clear()
    elif fault == "failed_exit":
        exits[0] = "EXIT_CODE=1"
    elif fault == "ema_nan":
        final["ema_leaves"]["kernel"]["finite"] = False
    elif fault == "ema_empty":
        final["ema_leaves"] = {}
    elif fault == "wrong_config":
        start["complete"]["train_config"]["fields"]["batch_size"] = 128
        final["complete"] = deepcopy(start["complete"])
    elif fault == "changed_complete":
        final["complete"]["unexpected"] = True
    with pytest.raises((ValueError, KeyError)):
        validate(records)


@pytest.mark.parametrize("fault", ["dtype", "shape", "value", "nonfinite", "empty", "missing_leaf"])
def test_restored_arrays_must_equal_saving_site(fault):
    records = fixture()
    arrays = deepcopy(records[6])
    if fault == "dtype":
        arrays["kernel"] = arrays["kernel"].astype(np.float64)
    elif fault == "shape":
        arrays["kernel"] = arrays["kernel"].reshape(3, 2)
    elif fault == "value":
        arrays["kernel"][0, 0] += 1
    elif fault == "nonfinite":
        arrays["kernel"][0, 0] = np.inf
    elif fault in ("empty", "missing_leaf"):
        arrays.clear()
    with pytest.raises(ValueError, match="真实恢复权重"):
        checker.validate_arrays(records[1], arrays)


def test_restored_bfloat16_dtype_is_not_coerced():
    import ml_dtypes
    arrays = {"kernel": np.ones((2, 3), dtype=ml_dtypes.bfloat16)}
    final = {"ema_leaves": checker.summarize_tree(arrays)}
    assert checker.validate_arrays(final, arrays)["kernel"]["dtype"] == "bfloat16"
    with pytest.raises(ValueError, match="真实恢复权重"):
        checker.validate_arrays(final, {"kernel": arrays["kernel"].astype(np.float32)})


def test_real_orbax_save_restore_compares_original_ema_bytes(tmp_path):
    """真实 OCDBT 保存/恢复，包含 NNX value 外壳和混合精度，零模型训练。"""
    import ml_dtypes
    import orbax.checkpoint as ocp

    pure = {"layer": {"kernel": np.arange(6).reshape(2, 3).astype(ml_dtypes.bfloat16),
                       "bias": np.array([0.25, -1.25, 2.5], dtype=np.float32)}}
    wrapped = {"params": {"layer": {key: {"value": value} for key, value in pure["layer"].items()}}}
    path = tmp_path / "params"
    with ocp.PyTreeCheckpointer() as saver:
        saver.save(path, args=ocp.args.PyTreeSave(wrapped))
    final = {"ema_leaves": checker.summarize_tree(pure)}
    assert checker.restore_arrays(final, path) == final["ema_leaves"]
    final["ema_leaves"]["layer/kernel"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="真实恢复权重"):
        checker.restore_arrays(final, path)


def sampling_fixture(tmp_path):
    base = datetime.datetime(2026, 9, 25, tzinfo=datetime.UTC).timestamp()
    rows = []
    for index in range(13):
        timestamp = datetime.datetime.fromtimestamp(base + index * .5, datetime.UTC)
        rows.extend(timestamp.strftime("%Y/%m/%d %H:%M:%S.%f") + f", {gpu}, 50, 1000\n" for gpu in range(4))
    (tmp_path / "gpu.csv").write_text("".join(rows))
    (tmp_path / "gpu.csv.err").write_text("")
    start = {"started_at": datetime.datetime.fromtimestamp(base + 2, datetime.UTC).isoformat()}
    wait = {"completed_at": datetime.datetime.fromtimestamp(base + 5, datetime.UTC).isoformat()}
    lines = ["GPU_SAMPLER_PID=12345", "GPU_SAMPLER_STOP pid=12345 exit_code=143 reason=driver_stop",
             "GPU_SAMPLER_FINAL_SAMPLE exit_code=0"]
    return lines, start, wait, rows


def test_gpu_samples_cover_real_execution_window(tmp_path):
    lines, start, wait, _ = sampling_fixture(tmp_path)
    report = checker.validate_gpu_sampling(tmp_path, lines, [0, 1, 2, 3], start, wait)
    assert set(report["gpu"]) == {"0", "1", "2", "3"}
    assert all(value["max_gap_s"] == .5 for value in report["gpu"].values())


@pytest.mark.parametrize("fault", [
    "early_exit", "stop_failure", "wrong_pid", "duplicate_stop", "missing_endpoint", "missing_csv",
    "empty_csv", "missing_err", "stderr", "late_start", "early_end", "middle_gap", "wrong_gpu",
    "nan", "duplicate_timestamp", "missing_timezone",
])
def test_gpu_sampler_failure_or_incomplete_coverage_rejected(tmp_path, fault):
    lines, start, wait, rows = sampling_fixture(tmp_path)
    csv = tmp_path / "gpu.csv"
    if fault == "early_exit":
        lines[1] = lines[1].replace("driver_stop", "exited_before_stop")
    elif fault == "stop_failure":
        lines[1] = lines[1].replace("exit_code=143", "exit_code=1")
    elif fault == "wrong_pid":
        lines[1] = lines[1].replace("12345", "12346")
    elif fault == "duplicate_stop":
        lines.append(lines[1])
    elif fault == "missing_endpoint":
        lines.pop()
    elif fault == "missing_csv":
        csv.unlink()
    elif fault == "empty_csv":
        csv.write_text("")
    elif fault == "missing_err":
        (tmp_path / "gpu.csv.err").unlink()
    elif fault == "stderr":
        (tmp_path / "gpu.csv.err").write_text("模拟采样错误")
    elif fault == "late_start":
        csv.write_text("".join(rows[20:]))
    elif fault == "early_end":
        csv.write_text("".join(rows[:36]))
    elif fault == "middle_gap":
        csv.write_text("".join(rows[:8] + rows[-8:]))
    elif fault == "wrong_gpu":
        csv.write_text(csv.read_text().replace(", 3,", ", 4,"))
    elif fault == "nan":
        csv.write_text(csv.read_text().replace(", 50,", ", nan,"))
    elif fault == "duplicate_timestamp":
        csv.write_text(rows[0] + "".join(rows))
    elif fault == "missing_timezone":
        start["started_at"] = start["started_at"].removesuffix("+00:00")
    with pytest.raises((ValueError, OSError)):
        checker.validate_gpu_sampling(tmp_path, lines, [0, 1, 2, 3], start, wait)
