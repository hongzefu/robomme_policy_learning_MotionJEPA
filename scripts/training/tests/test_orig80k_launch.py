"""原版四卡启动契约的实际 CLI 解析与失败传播，不启动训练。"""

import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/training"))
import orig80k_contract as contract  # noqa: E402 -- 独立工具目录在上方明确加入


def argv(mode="prod", repo=ROOT):
    store = Path(repo) / "v1-store"
    run = contract.LIBRARIES["16task-pub-1600ep"]["run"] if mode == "prod" else "test-orig80k-" + mode
    return contract.make_train_args(mode, run, store / "datasets/16task-pub-1600ep",
                                    store / "train-assets/mme_vla_suite", repo)


@pytest.fixture(scope="module")
def cpu_environment():
    previous = dict(os.environ)
    store = ROOT / "v1-store"
    os.environ.update(OPENPI_DATA_HOME=str(store / "models"), UV_CACHE_DIR=str(store / "cache/uv"),
                      XDG_CACHE_HOME=str(store / "cache/xdg"), HF_HOME=str(store / "cache/hf"))
    yield
    os.environ.clear()
    os.environ.update(previous)


@pytest.mark.parametrize("mode", ["prod", "perf", "smoke"])
def test_real_cli_parses_only_approved_mode_difference(mode, cpu_environment):
    before = os.environ.get("JAX_PLATFORMS"), os.environ.get("CUDA_VISIBLE_DEVICES")
    parsed = contract.parse_config(argv(mode), mode)
    assert parsed["jax_enable_x64"] is False
    fields = parsed["complete"]["train_config"]["fields"]
    assert parsed["num_train_steps"] == contract.MODE_STEPS[mode]
    assert (fields["batch_size"], fields["num_workers"], fields["fsdp_devices"], fields["wandb_enabled"]) == (64, 4, 4, True)
    assert before == (os.environ.get("JAX_PLATFORMS"), os.environ.get("CUDA_VISIBLE_DEVICES"))


def test_inherited_x64_enabled_is_rejected_without_rewriting_environment(cpu_environment, monkeypatch):
    monkeypatch.setenv("JAX_ENABLE_X64", "true")
    with pytest.raises(ValueError, match="jax_enable_x64=True"):
        contract.parse_config(argv("smoke"), "smoke")
    assert os.environ["JAX_ENABLE_X64"] == "true"


def test_upstream_original_configuration_reference():
    assert contract.original_reference()["upstream_head"] == contract.UPSTREAM_HEAD


def test_manifest_canonical_matches_committed_real_dataset_and_unicode():
    path = ROOT / "docs/dataset-build-doc/16task-h5-scan/records/episode_manifest.json"
    value = json.loads(path.read_text())
    assert contract.json_sha({key: item for key, item in value.items() if key != "sha256"}) == value["sha256"]
    assert contract.json_sha({"task": "任务"}) == hashlib.sha256('{"task":"任务"}'.encode()).hexdigest()


@pytest.mark.parametrize("extra", [
    ["--num-train-steps", "20"], ["--batch-size", "128"], ["--num-workers", "8"],
    ["--lr-schedule.peak-lr", "0.01"], ["--overwrite"], ["--resume"],
    ["--no-wandb-enabled"], ["--exp-name", "duplicate"], ["extra"],
])
def test_prod_rejects_unapproved_or_duplicate_arguments(extra):
    with pytest.raises(ValueError, match="禁止|重复"):
        contract.validate_argv(argv() + extra, "prod")


@pytest.mark.parametrize("mode", ["perf", "smoke"])
def test_validation_steps_cannot_drift(mode):
    values = argv(mode)
    values[-1] = "80000"
    with pytest.raises(ValueError, match="步数"):
        contract.validate_argv(values, mode)


def path_fixture(tmp_path, mode="smoke"):
    store = tmp_path / "v1-store"
    lib = store / "datasets/16task-pub-1600ep"
    assets = store / "train-assets/mme_vla_suite"
    lib.mkdir(parents=True)
    (assets / "robomme").mkdir(parents=True)
    (assets / "robomme/norm_stats.json").write_text("{}")
    args = argv(mode, tmp_path)
    values = contract.validate_argv(args, mode)
    return lib, assets, values


def test_paths_and_nonexistent_output_accepted(tmp_path):
    lib, assets, values = path_fixture(tmp_path)
    output, selected = contract.validate_paths(values, "smoke", values["--exp-name"], lib, assets, tmp_path)
    assert output.name == "test-orig80k-smoke"
    assert selected["gpus"] == "0,1,2,3"


@pytest.mark.parametrize("fault", ["relative", "double_robomme", "existing", "dangling", "outside"])
def test_bad_assets_or_output_never_allowed(tmp_path, fault):
    lib, assets, values = path_fixture(tmp_path)
    output = tmp_path / "v1-store/train-runs/mme_vla_suite" / values["--exp-name"]
    if fault == "relative":
        assets = Path("v1-store/train-assets/mme_vla_suite")
    elif fault == "double_robomme":
        assets = assets / "robomme"
    elif fault == "existing":
        output.mkdir(parents=True)
    elif fault == "dangling":
        output.parent.mkdir(parents=True)
        output.symlink_to(tmp_path / "missing")
    else:
        assets = tmp_path
    with pytest.raises(ValueError, match="绝对路径|父目录|run-root|越界"):
        contract.validate_paths(values, "smoke", values["--exp-name"], lib, assets, tmp_path)


def test_wrong_norm_sha_rejected_before_parsing_dataset(tmp_path):
    norm = tmp_path / "norm.json"
    norm.write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        contract.validate_data(tmp_path, {}, norm, "0" * 64)


@pytest.mark.parametrize("gpus", ["0,1,2", "0,1,2,2", "0,1,2,4", "4,5,6,7", "0,1,2,3,"])
def test_invalid_or_overlapping_group_rejected(monkeypatch, gpus):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpus)
    with pytest.raises(ValueError, match="GPU|四张"):
        contract.validate_gpus(gpus, "0,1,2,3")


@pytest.mark.parametrize("busy", [False, True])
def test_gpu_query_checks_only_owned_group(monkeypatch, busy):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    rows = "\n".join(f"{index}, GPU-{index}, {1 if index == 7 or (busy and index == 0) else 0}" for index in range(8))
    monkeypatch.setattr(contract.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=rows))
    if busy:
        with pytest.raises(ValueError, match="空闲"):
            contract.validate_gpus("0,1,2,3", "0,1,2,3")
    else:
        assert len(contract.validate_gpus("0,1,2,3", "0,1,2,3")) == 4


def write_budget_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def perf_budget_source(repo, group, library):
    """构造完整小记录及真实小型checkpoint目录；不宣称执行了模型或真实perf。"""
    from check_orig80k_completion import SCALARS
    from check_orig80k_completion import validate_gpu_sampling
    import check_orig80k_speed as speed

    run, head = "test-perf-" + group, "a" * 40
    records = repo / "v1-store/bench/orig80k" / run
    run_root = repo / "v1-store/train-runs/mme_vla_suite" / run
    checkpoint = run_root / "299"
    checkpoint.mkdir(parents=True)
    write_budget_json(checkpoint / "_CHECKPOINT_METADATA", {
        "init_timestamp_nsecs": 1000300000000, "commit_timestamp_nsecs": 1000800000000})
    write_budget_json(checkpoint / "params/_METADATA", {"测试": "参数索引"})
    (checkpoint / "params/manifest.ocdbt").write_bytes(b"fixture")
    (checkpoint / "params/array.bin").write_bytes(b"\x00\x00\x80?")
    write_budget_json(checkpoint / "assets/robomme/norm_stats.json", {"测试": "统计量"})
    complete = {"train_config": {"fields": {
        "name": contract.CONFIG, "exp_name": run, "num_train_steps": 300, "batch_size": 64,
        "num_workers": 4, "fsdp_devices": 4, "seed": 42, "log_interval": 100,
        "save_interval": 10000, "keep_period": 10000, "ema_decay": .999,
        "wandb_enabled": True, "overwrite": False, "resume": False,
        "dataset_path": str(repo / "v1-store/datasets" / library / "framesamp")}}}
    gpus = contract.LIBRARIES[library]["gpus"]
    launch = {"mode": "perf", "run_name": run, "head": head, "checkpoint_dir": str(run_root),
              "environment": {"CUDA_VISIBLE_DEVICES": gpus},
              "argv": contract.make_train_args("perf", run, repo / "v1-store/datasets" / library,
                        repo / "v1-store/train-assets" / contract.LIBRARIES[library]["assets"], repo),
              "actual": {"num_train_steps": 300, "complete": complete, "jax_enable_x64": False}}
    write_budget_json(records / "launch.json", launch)
    write_budget_json(records / "runtime.json", {"device_count": 4, "fsdp_devices": 4, "batch_size": 64,
        "num_workers": 4, "exp_name": run, "config_name": contract.CONFIG, "seed": 42, "cuda_visible_devices": gpus})
    identity = {"run_name": run, "head": head, "run_uuid": str(uuid.uuid4()), "budget": 512,
                "num_train_steps": 300, "log_interval": 100, "save_interval": 10000, "keep_period": 10000,
                "wandb_run_id": "fixture", "complete": complete, "checkpoint_dir": str(run_root),
                "started_at": datetime.datetime.fromtimestamp(999.9, datetime.UTC).isoformat()}
    scalar = {"dec": 1.0, "hex": 1.0.hex(), "finite": True}
    leaves = {"kernel": {"dtype": "float32", "shape": [1], "bytes": 4, "sha256": "c" * 64, "finite": True}}
    final = {**identity, "state_step": 300, "loop_step": 299, "checkpoint_step": 299,
             "tail": [{"step": step, "scalars": {key: dict(scalar) for key in SCALARS}} for step in range(201, 300)],
             "tail_means": dict.fromkeys(SCALARS, 1.0), "ema_leaves": leaves}
    wait = {"head": head, "run_uuid": identity["run_uuid"], "wait_until_finished": True,
            "completed_at": datetime.datetime.fromtimestamp(1000.9, datetime.UTC).isoformat()}
    for name, payload in (("start.json", identity), ("final.json", final), ("checkpoint_wait_done.json", wait)):
        write_budget_json(records / "final" / name, payload)
    metrics = [{"step": step, "wall_time": float(step), **{key: {"dec": 1.0, "hex": 1.0.hex()} for key in SCALARS}}
               for step in (0, 100, 200)]
    (records / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    log = repo / "v1-store/logs" / f"{run}.driver.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("GPU_SAMPLER_PID=1234\nTRAIN_WRAPPER_PID=1235\n"
                   "GPU_SAMPLER_STOP pid=1234 exit_code=143 reason=driver_stop\n"
                   "GPU_SAMPLER_FINAL_SAMPLE exit_code=0\nTRAIN_PIPE_EXIT=0\nTEE_EXIT=0\nEXIT_CODE=0\n")
    (records / "gpu.csv").write_text("".join(
        f"{datetime.datetime.fromtimestamp(stamp, datetime.UTC).strftime('%Y/%m/%d %H:%M:%S.%f')}, {gpu}, 90, 1000\n"
        for stamp in (999.5, 1000., 1000.5, 1001.) for gpu in gpus.split(",")))
    (records / "gpu.csv.err").write_text("")
    completion_paths = [*[records / name for name in ("launch.json", "runtime.json", "metrics.jsonl", "gpu.csv", "gpu.csv.err")], log,
                        *[records / "final" / name for name in ("start.json", "final.json", "checkpoint_wait_done.json")],
                        checkpoint / "_CHECKPOINT_METADATA", checkpoint / "params/_METADATA"]
    completion = {"status": "PASS", "mode": "perf", "run_name": run, "head": head,
                  "run_uuid": identity["run_uuid"], "state_step": 300, "final": 299, "checkpoints": [299],
                  "restored_leaves": leaves,
                  "gpu_sampling": validate_gpu_sampling(records, log.read_text().splitlines(),
                                                        [int(value) for value in gpus.split(",")], identity, wait),
                  "files": [contract.file_reference(path) for path in completion_paths]}
    write_budget_json(records / "completion.json", completion)
    allocation = speed.checkpoint_allocation(run_root)
    disk_rows = []
    for index, (kind, stamp) in enumerate((("start", 1000.), ("periodic", 1000.5), ("final", 1001.))):
        disk_rows.append({"sequence": index, "kind": kind, "phase": "save_pending",
                          "wall_start": stamp, "wall_end": stamp + .01, "monotonic_start": stamp,
                          "scheduled_monotonic": stamp, "interval_s": .5 if index else None,
                          "lateness_s": 0., "duration_s": .01, "errors": [], "scratch_available_bytes": 999999,
                          "checkpoint": {**allocation, "allocated_bytes": allocation["allocated_bytes"] + (4096 if index == 1 else 0)}})
    disk_path = records / "disk_samples.jsonl"
    disk_path.write_text("".join(json.dumps(row) + "\n" for row in disk_rows))
    speed_run = {"head": head, "mode": "perf", "steps": 300, "success": True, "entry_calls": 1,
                 "sampler_stopped": True, "profiler": False,
                 "entry_start_wall": 1000.1, "end_wall": 1000.95,
                 "save_calls": [{"step": 299, "state_step": 300, "start_wall": 1000.25,
                                 "return_wall": 1000.4, "checkpoint_root": str(run_root)}],
                 "disk_sampling": {"schema": 1, "interval_s": .5, "stopped": True, "error": None,
                                   "checkpoint_root": str(run_root), "scratch_path": str(speed.SCRATCH),
                                   "samples_file": "disk_samples.jsonl", "samples": 3, "missed_ticks": 0,
                                   "sha256": contract.sha256_file(disk_path), "note": "测试离散采样，不是连续峰值"}}
    write_budget_json(records / "speed_run.json", speed_run)
    report = {"head": head, "batch_size": 64, "workers": 4, "steady_steps": [100, 299],
              "disk": speed.summarize_disk(records, speed_run)}
    report_path = records / "perf-report.json"
    write_budget_json(report_path, report)
    receipt_path = records / "checkpoint-measurement.json"
    contract.measure_checkpoint(SimpleNamespace(run=run, head=head, group=group, report=str(report_path),
                                                completion=str(records / "completion.json"), out=str(receipt_path)), repo)
    return {"run_name": run, "head": head,
            **{key: contract.file_reference(path) for key, path in {
                "launch": records / "launch.json", "report": report_path, "completion": records / "completion.json",
                "disk_samples": disk_path, "measurement": receipt_path}.items()}}


def budget_fixture(tmp_path):
    sources = {group: perf_budget_source(tmp_path, group, library)
               for group, library in (("full", "16task-pub-1600ep"), ("counting", "4task-counting-pub-400ep"))}
    receipts = [json.loads(Path(source["measurement"]["path"]).read_text()) for source in sources.values()]
    retained = 8 * sum(receipt["checkpoint"]["allocated_bytes"] for receipt in receipts)
    extra = sum(max(0, receipt["disk"]["sampled_max_allocated_bytes"] - receipt["disk"]["final_allocated_bytes"])
                for receipt in receipts)
    # 这些容量只用于边界测试，不能写入正式预算或冒称用户已确认的余量。
    margins = {"schema": 1, "checkpoint_growth_margin_bytes": 320 * 2**30 - retained,
               "save_sampling_margin_bytes": 12 * 2**30 - extra, "logs_cache_margin_bytes": 4 * 2**30,
               "basis": dict.fromkeys(contract.MARGIN_KEYS, "仅测试夹具的明确数值")}
    margin_path = tmp_path / "margin-source.json"
    write_budget_json(margin_path, margins)
    value = {"schema": 2, "checkpoint_total_bytes": 320 * 2**30, "save_temporary_peak_bytes": 12 * 2**30,
             "logs_cache_margin_bytes": 4 * 2**30, "source_perf": sources,
             "margin_source": contract.file_reference(margin_path)}
    path = tmp_path / "disk-budget.json"
    write_budget_json(path, value)
    return path, value


def test_explicit_production_disk_budget_and_perf_identities(tmp_path, monkeypatch):
    path, value = budget_fixture(tmp_path)
    monkeypatch.setattr(contract.os, "statvfs", lambda path: SimpleNamespace(f_bavail=340, f_frsize=2**30))
    report = contract.validate_disk_budget("prod", tmp_path, budget_path=str(path), budget_sha=contract.sha256_file(path))
    assert report["required_bytes"] == 336 * 2**30
    assert set(report["source_perf"]) == {"full", "counting"}


@pytest.mark.parametrize("fault", ["missing", "wrong_sha", "missing_margin", "negative", "too_small", "wrong_source"])
def test_production_disk_budget_fail_closed(tmp_path, monkeypatch, fault):
    path, value = budget_fixture(tmp_path)
    monkeypatch.setattr(contract.os, "statvfs", lambda path: SimpleNamespace(f_bavail=335 if fault == "too_small" else 340, f_frsize=2**30))
    if fault == "missing_margin":
        value.pop("logs_cache_margin_bytes")
    if fault == "negative":
        value["logs_cache_margin_bytes"] = -1
    if fault == "wrong_source":
        value["source_perf"]["full"]["head"] = "b" * 40
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, KeyError)):
        contract.validate_disk_budget("prod", tmp_path,
            budget_path=None if fault == "missing" else str(path),
            budget_sha="0" * 64 if fault == "wrong_sha" else contract.sha256_file(path))


def test_budget_launch_only_is_not_completed_perf(tmp_path):
    old = {"checkpoint_total_bytes": 1, "save_temporary_peak_bytes": 0, "logs_cache_margin_bytes": 0,
           "source_perf": {group: {"run_name": group, "head": "a" * 40} for group in ("full", "counting")}}
    path = tmp_path / "old-budget.json"
    write_budget_json(path, old)
    with pytest.raises(ValueError, match="schema2"):
        contract.validate_disk_budget("prod", tmp_path, budget_path=str(path), budget_sha=contract.sha256_file(path))


@pytest.mark.parametrize("fault", ["completion_fail", "smoke", "wrong_step", "wrong_uuid", "disk_changed",
                                   "report_changed", "stat_total", "stat_subtree", "eight_copies", "zero_margin", "no_basis"])
def test_budget_rejects_incomplete_or_inconsistent_proof(tmp_path, monkeypatch, fault):
    path, budget = budget_fixture(tmp_path)
    monkeypatch.setattr(contract.os, "statvfs", lambda _: SimpleNamespace(f_bavail=1000, f_frsize=2**30))
    source = budget["source_perf"]["full"]
    key = "completion" if fault in {"completion_fail", "smoke", "wrong_step", "wrong_uuid"} else "measurement"
    if fault in {"completion_fail", "smoke", "wrong_step", "wrong_uuid"}:
        target = Path(source[key]["path"])
        value = json.loads(target.read_text())
        field, update = {"completion_fail": ("status", "FAIL"), "smoke": ("mode", "smoke"),
                         "wrong_step": ("state_step", 20), "wrong_uuid": ("run_uuid", str(uuid.uuid4()))}[fault]
        value[field] = update
        write_budget_json(target, value)
        source[key] = contract.file_reference(target)
    elif fault in {"disk_changed", "report_changed"}:
        key = "disk_samples" if fault == "disk_changed" else "report"
        with Path(source[key]["path"]).open("a") as stream:
            stream.write("\n")
    elif fault in {"stat_total", "stat_subtree"}:
        target = Path(source["measurement"]["path"])
        value = json.loads(target.read_text())
        if fault == "stat_total":
            value["checkpoint"]["allocated_bytes"] = 1
        else:
            entries = value["checkpoint"]["entries"]
            for entry in entries:
                entry["blocks"] = 0
            value["checkpoint"] = contract.stat_receipt(entries)
        write_budget_json(target, value)
        source["measurement"] = contract.file_reference(target)
    elif fault == "eight_copies":
        budget["checkpoint_total_bytes"] -= 1
    else:
        target = Path(budget["margin_source"]["path"])
        value = json.loads(target.read_text())
        if fault == "zero_margin":
            value["save_sampling_margin_bytes"] = 0
        else:
            value.pop("basis")
        write_budget_json(target, value)
        budget["margin_source"] = contract.file_reference(target)
    write_budget_json(path, budget)
    with pytest.raises((ValueError, KeyError)):
        contract.validate_disk_budget("prod", tmp_path, budget_path=str(path), budget_sha=contract.sha256_file(path))


def test_budget_keeps_complete_measurement_after_checkpoint_cleanup(tmp_path, monkeypatch):
    path, budget = budget_fixture(tmp_path)
    for source in budget["source_perf"].values():
        receipt = json.loads(Path(source["measurement"]["path"]).read_text())
        shutil.rmtree(Path(receipt["checkpoint_path"]).parent)
    monkeypatch.setattr(contract.os, "statvfs", lambda _: SimpleNamespace(f_bavail=1000, f_frsize=2**30))
    report = contract.validate_disk_budget("prod", tmp_path, budget_path=str(path), budget_sha=contract.sha256_file(path))
    assert all(value["checkpoints"] == 8 for value in report["source_perf"].values())


@pytest.mark.parametrize("filename", ["runtime.json", "gpu.csv", "gpu.csv.err"])
def test_budget_requires_all_completion_evidence(tmp_path, filename):
    path, budget = budget_fixture(tmp_path)
    source = budget["source_perf"]["full"]
    target = Path(source["completion"]["path"])
    value = json.loads(target.read_text())
    value["files"] = [item for item in value["files"] if Path(item["path"]).name != filename]
    write_budget_json(target, value)
    source["completion"] = contract.file_reference(target)
    write_budget_json(path, budget)
    with pytest.raises(ValueError, match="证据文件链不完整"):
        contract.validate_disk_budget("prod", tmp_path, budget_path=str(path), budget_sha=contract.sha256_file(path))


@pytest.mark.parametrize(("field", "value"), [("device_count", 1), ("batch_size", 32), ("fsdp_devices", 1),
                                         ("num_workers", 0), ("seed", 1), ("exp_name", "other"),
                                         ("cuda_visible_devices", "4,5,6,7")])
def test_budget_rechecks_actual_runtime(tmp_path, field, value):
    path, budget = budget_fixture(tmp_path)
    source = budget["source_perf"]["full"]
    runtime_path = Path(source["launch"]["path"]).with_name("runtime.json")
    runtime = json.loads(runtime_path.read_text())
    runtime[field] = value
    write_budget_json(runtime_path, runtime)
    with pytest.raises(ValueError, match="真实设备|真实runtime|真实GPU"):
        contract.validate_disk_budget("prod", tmp_path, budget_path=str(path), budget_sha=contract.sha256_file(path))


def test_budget_recomputes_gpu_summary(tmp_path):
    path, budget = budget_fixture(tmp_path)
    source = budget["source_perf"]["full"]
    target = Path(source["completion"]["path"])
    value = json.loads(target.read_text())
    value["gpu_sampling"]["gpu"]["0"]["samples"] += 1
    write_budget_json(target, value)
    source["completion"] = contract.file_reference(target)
    write_budget_json(path, budget)
    with pytest.raises(ValueError, match="GPU采样摘要"):
        contract.validate_disk_budget("prod", tmp_path, budget_path=str(path), budget_sha=contract.sha256_file(path))


def test_cleaned_checkpoint_does_not_allow_lower_derived_disk_peak(tmp_path):
    path, budget = budget_fixture(tmp_path)
    source = budget["source_perf"]["full"]
    report_path, receipt_path = (Path(source[key]["path"]) for key in ("report", "measurement"))
    report, receipt = (json.loads(target.read_text()) for target in (report_path, receipt_path))
    old_extra = report["disk"]["sampled_max_allocated_bytes"] - report["disk"]["final_allocated_bytes"]
    assert old_extra > 0
    # 同步伪改全部派生摘要和它们的SHA，保留原始采样不变；必须由原始证据拒绝。
    report["disk"]["sampled_max_allocated_bytes"] = report["disk"]["final_allocated_bytes"]
    write_budget_json(report_path, report)
    source["report"] = contract.file_reference(report_path)
    receipt["disk"] = report["disk"]
    receipt["evidence"]["report"] = source["report"]
    write_budget_json(receipt_path, receipt)
    source["measurement"] = contract.file_reference(receipt_path)
    budget["save_temporary_peak_bytes"] -= old_extra
    write_budget_json(path, budget)
    for side in budget["source_perf"].values():
        measured = json.loads(Path(side["measurement"]["path"]).read_text())
        shutil.rmtree(Path(measured["checkpoint_path"]).parent)
    with pytest.raises(ValueError, match="原始采样"):
        contract.validate_disk_budget("prod", tmp_path, budget_path=str(path), budget_sha=contract.sha256_file(path))


def test_checkpoint_stat_deduplicates_hardlinks_and_rejects_symlinks(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "data").write_bytes(b"bytes" * 1024)
    os.link(root / "data", root / "alias")
    receipt = contract.checkpoint_stat(root)
    assert receipt["allocated_bytes"] == (root.stat().st_blocks + (root / "data").stat().st_blocks) * 512
    (root / "symlink").symlink_to(root / "data")
    with pytest.raises(ValueError, match="软链"):
        contract.checkpoint_stat(root)


def test_measurement_selects_own_side_of_combined_perf_report(tmp_path, monkeypatch):
    path, budget = budget_fixture(tmp_path)
    sections = [json.loads(Path(source["report"]["path"]).read_text()) for source in budget["source_perf"].values()]
    combined = tmp_path / "combined-perf.json"
    # 故意交换顺序，必须依据checkpoint_root归属，不依赖first标签。
    write_budget_json(combined, {"full_or_first": sections[1], "counting_or_second": sections[0]})
    for group, source in budget["source_perf"].items():
        output = Path(source["measurement"]["path"]).with_name("combined-measurement.json")
        contract.measure_checkpoint(SimpleNamespace(run=source["run_name"], head=source["head"], group=group,
            report=str(combined), completion=source["completion"]["path"], out=str(output)), tmp_path)
        source["report"] = contract.file_reference(combined)
        source["measurement"] = contract.file_reference(output)
    write_budget_json(path, budget)
    monkeypatch.setattr(contract.os, "statvfs", lambda _: SimpleNamespace(f_bavail=1000, f_frsize=2**30))
    assert contract.validate_disk_budget("prod", tmp_path, budget_path=str(path), budget_sha=contract.sha256_file(path))["required_bytes"] == 336 * 2**30


def test_measurement_rejects_missing_real_save_window(tmp_path):
    _, budget = budget_fixture(tmp_path)
    source = budget["source_perf"]["full"]
    records = Path(source["launch"]["path"]).parent
    disk_path = records / "disk_samples.jsonl"
    rows = [json.loads(line) for line in disk_path.read_text().splitlines()]
    rows[1].update(wall_start=1000.85, wall_end=1000.86, monotonic_start=1000.85, interval_s=.85)
    rows[1]["interval_s"] = rows[1]["monotonic_start"] - rows[0]["monotonic_start"]
    rows[2]["interval_s"] = rows[2]["monotonic_start"] - rows[1]["monotonic_start"]
    disk_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    speed_run = json.loads((records / "speed_run.json").read_text())
    speed_run["disk_sampling"]["sha256"] = contract.sha256_file(disk_path)
    write_budget_json(records / "speed_run.json", speed_run)
    report_path = Path(source["report"]["path"])
    report = json.loads(report_path.read_text())
    report["disk"]["samples_sha256"] = contract.sha256_file(disk_path)
    write_budget_json(report_path, report)
    with pytest.raises(ValueError, match="保存期间"):
        contract.measure_checkpoint(SimpleNamespace(run=source["run_name"], head=source["head"], group="full",
            report=str(report_path), completion=source["completion"]["path"], out=str(records / "invalid-measurement.json")), tmp_path)


def test_measurement_never_overwrites_existing_receipt(tmp_path):
    _, budget = budget_fixture(tmp_path)
    source = budget["source_perf"]["full"]
    output = Path(source["measurement"]["path"])
    before = output.read_bytes()
    with pytest.raises(ValueError, match="输出已存在"):
        contract.measure_checkpoint(SimpleNamespace(run=source["run_name"], head=source["head"], group="full",
            report=source["report"]["path"], completion=source["completion"]["path"], out=str(output)), tmp_path)
    assert output.read_bytes() == before


@pytest.mark.parametrize("mode", ["perf", "smoke"])
def test_validation_preserves_300_gib_without_inventing_budget(mode, tmp_path, monkeypatch):
    monkeypatch.setattr(contract.os, "statvfs", lambda path: SimpleNamespace(f_bavail=300, f_frsize=2**30))
    assert contract.validate_disk_budget(mode, tmp_path)["required_bytes"] == 300 * 2**30
    monkeypatch.setattr(contract.os, "statvfs", lambda path: SimpleNamespace(f_bavail=299, f_frsize=2**30))
    with pytest.raises(ValueError, match="磁盘空间不足"):
        contract.validate_disk_budget(mode, tmp_path)


@pytest.mark.parametrize("mode", ["prod", "perf", "smoke"])
@pytest.mark.parametrize("failure", ["none", "preflight", "contract", "train", "sampler"])
def test_real_shell_propagates_failures_and_keeps_cpu_scoped(tmp_path, mode, failure):
    """执行真实 shell 控制流，外部训练与设备调用换为计数桩。"""
    fixture = tmp_path / "repo"
    paths = fixture / "scripts/training/paths.sh"
    paths.parent.mkdir(parents=True)
    paths.write_text('V1_STORE="' + str(fixture / "v1-store") + '"\n')
    speed = fixture / "scripts/training/tests/check_orig80k_speed.py"
    speed.parent.mkdir()
    speed.write_text("# 测试桩，不执行 Python\n")
    (fixture / "v1-store/logs").mkdir(parents=True)
    runner = tmp_path / "runner.sh"
    source = (ROOT / "scripts/training/prod/run_orig80k.sh").read_text()
    runner.write_text(source.replace("MAIN=/scratch/hongze/robomme_policy_learning_MotionJEPA", "MAIN=" + str(fixture)))
    bins = tmp_path / "bin"
    bins.mkdir()
    (bins / "git").write_text('#!/bin/bash\nif [ "$1" = rev-parse ]; then printf "%s\\n" "' + "a" * 40 + '"; fi\n')
    (bins / "nvidia-smi").write_text("""#!/bin/bash
case "$*" in
  *-lms*)
    if [ "$TEST_FAILURE" = sampler ]; then exit 0; fi
    exec sleep 60 ;;
  *)
    for gpu in 0 1 2 3; do printf '2026/09/25 00:00:00.000, %s, 0, 0\n' "$gpu"; done ;;
esac
""")
    (bins / "uv").write_text("""#!/bin/bash
case "$*" in
  *preflight_train_launch.py*) phase=preflight ;;
  *orig80k_contract.py*) phase=contract ;;
  *scripts/training/train.py*|*check_orig80k_speed.py*) phase=train ;;
  *) exit 90 ;;
esac
printf '%s:%s\n' "$phase" "${JAX_PLATFORMS:-unset}" >> "$TEST_CALLS"
if [ "$phase" = "$TEST_FAILURE" ]; then exit 7; fi
if [ "$phase" = contract ]; then mkdir -p "$TRAIN_RECORD_DIR"; fi
if [ "$phase" = train ] && [ "$TEST_FAILURE" = sampler ]; then sleep .2; fi
exit 0
""")
    for binary in bins.iterdir():
        binary.chmod(0o700)
    calls = tmp_path / "calls"
    env = dict(os.environ, PATH=str(bins) + os.pathsep + os.environ["PATH"], TEST_CALLS=str(calls),
               TEST_FAILURE=failure, TRAIN_HEAD="a" * 40, HISTORY_CONFIG_SHA256="b" * 64, NORM_STATS_SHA256="c" * 64,
               JAX_PLATFORMS="old-cpu-value", XLA_FLAGS="old-validation-flags")
    result = subprocess.run(["bash", str(runner), mode, "test-launch", "0,1,2,3", str(tmp_path / "lib"), str(tmp_path / "assets")],
                            env=env, text=True, capture_output=True, timeout=20, check=False)
    expected = ["preflight:cpu"]
    if failure != "preflight":
        expected += ["contract:unset"]
    if failure in ("none", "train", "sampler"):
        expected += ["train:unset"]
    assert calls.read_text().splitlines() == expected
    assert result.returncode == (0 if failure == "none" else (1 if failure == "sampler" else 7)), result.stdout + result.stderr
    # 最终状态在footer检查之后直接落日志；独立核对真实进程返回及唯一持久化回执。
    log_lines = (fixture / "v1-store/logs/test-launch.driver.log").read_text().splitlines()
    assert [line for line in log_lines if line.startswith("EXIT_CODE=")] == [f"EXIT_CODE={result.returncode}"]
    for key in ("FOOTER_PRINTF_EXIT", "FOOTER_TEE_EXIT"):
        assert [line for line in log_lines if line.startswith(key + "=")] == [key + "=0"]
    assert "TEE_EXIT=0" in result.stdout
    if failure == "sampler":
        assert "reason=exited_before_stop" in result.stdout
        assert "GPU_SAMPLER_FINAL_SAMPLE" not in result.stdout
    elif failure in ("none", "train"):
        assert "reason=driver_stop" in result.stdout
        assert "GPU_SAMPLER_FINAL_SAMPLE exit_code=0" in result.stdout
    # 同名第二次启动连 preflight 都不能执行。
    repeated = subprocess.run(["bash", str(runner), mode, "test-launch", "0,1,2,3", str(tmp_path / "lib"), str(tmp_path / "assets")],
                              env=env, text=True, capture_output=True, timeout=20, check=False)
    assert repeated.returncode != 0
    assert calls.read_text().splitlines() == expected
