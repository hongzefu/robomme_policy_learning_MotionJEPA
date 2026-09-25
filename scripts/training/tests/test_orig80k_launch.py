"""原版四卡启动契约的实际 CLI 解析与失败传播，不启动训练。"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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


def budget_fixture(tmp_path):
    source = {}
    for group, library in (("full", "16task-pub-1600ep"), ("counting", "4task-counting-pub-400ep")):
        run = "test-perf-" + group
        source[group] = {"run_name": run, "head": "a" * 40}
        path = tmp_path / "v1-store/bench/orig80k" / run / "launch.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"mode": "perf", **source[group], "actual": {
            "num_train_steps": 300, "complete": {"train_config": {"fields": {
                "dataset_path": str(tmp_path / "v1-store/datasets" / library / "framesamp")}}}}}))
    value = {"checkpoint_total_bytes": 320 * 2**30, "save_temporary_peak_bytes": 12 * 2**30,
             "logs_cache_margin_bytes": 4 * 2**30, "source_perf": source}
    path = tmp_path / "disk-budget.json"
    path.write_text(json.dumps(value))
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
    with pytest.raises(ValueError, match="预算|磁盘空间不足"):
        contract.validate_disk_budget("prod", tmp_path,
            budget_path=None if fault == "missing" else str(path),
            budget_sha="0" * 64 if fault == "wrong_sha" else contract.sha256_file(path))


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
    assert f"EXIT_CODE={result.returncode}" in result.stdout
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
