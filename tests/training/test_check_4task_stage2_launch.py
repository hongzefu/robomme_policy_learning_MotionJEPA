"""4task 起跑合同（1114 计划 S1）：步数口径（裁决 2）、final record 50001 尾窗为空、attempt 白名单与真实 CLI 解析。

库数据部分（DATASET 组）依赖 S2 的 local-built 打包产物，本文件以桩替代；其真实闭环由合入后 L0／短测前 dry-run 验证。
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/check_4task_stage2_launch.py"
spec = importlib.util.spec_from_file_location("check_4task_stage2_launch", SCRIPT)
launch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launch
spec.loader.exec_module(launch)

SCALARS = {"loss", "grad_norm", "llm_grad_norm", "param_norm", "mem_enc_norm"}
HEAD = "a" * 40
MODUL = launch.HISTORIES[1]
CTX = launch.HISTORIES[0]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def final_fixture(run, *, steps):
    return dict(run_name=run, head=HEAD, num_train_steps=steps, state_step=steps,
                loop_step=steps - 1, checkpoint_step=steps - 1, log_interval=100,
                tail=[dict(step=i, scalars={name: dict(finite=True) for name in SCALARS})
                      for i in range((steps - 1) // 100 * 100 + 1, steps)],
                ema_leaves=dict(weight=dict(finite=True), bias=dict(finite=True)))


def run_final(tmp_path, record, *, run, steps):
    path = tmp_path / "final.json"
    dump(path, record)
    return subprocess.run([sys.executable, str(SCRIPT), "--check-final-record", str(path), "--run", run,
                           "--steps", str(steps), "--train-head", HEAD], capture_output=True, text=True, timeout=120)


# ── 步数与 final record ─────────────────────────────────────────────────────


def test_scalar_steps_and_schedule_constants():
    assert launch.STEPS_CHOICES == (20, 50001)
    assert launch.SCHEDULE == (5000, 5e-5, 80000, 5e-5)
    assert launch.CONFIG_NAME == "mme_vla_suite_4task_b128_50k"


def test_final_record_50001_empty_tail_passes(tmp_path):
    run = launch.RUNS["modulation"]
    record = final_fixture(run, steps=50001)
    assert record["tail"] == [] and record["loop_step"] == record["checkpoint_step"] == 50000
    result = run_final(tmp_path, record, run=run, steps=50001)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"TRAIN_FINAL_RECORD=PASS run={run} steps=50001 tail_steps=0 ema_leaves=2" in result.stdout


def test_final_record_50001_nan_ema_rejected(tmp_path):
    run = launch.RUNS["context"]
    record = final_fixture(run, steps=50001)
    record["ema_leaves"]["weight"]["finite"] = False
    result = run_final(tmp_path, record, run=run, steps=50001)
    assert result.returncode == 1 and "TRAIN_FINAL_RECORD=FAIL" in result.stdout


@pytest.mark.parametrize("fault", ["state_step", "loop_step", "nonempty_tail", "run", "head", "empty_ema"])
def test_final_record_50001_identity_faults(tmp_path, fault):
    run = launch.RUNS["modulation"]
    record = final_fixture(run, steps=50001)
    if fault == "state_step": record["state_step"] = 50000
    if fault == "loop_step": record["loop_step"] = record["checkpoint_step"] = 50001
    if fault == "nonempty_tail": record["tail"] = final_fixture(run, steps=50000)["tail"][-1:]
    if fault == "run": record["run_name"] = launch.RUNS["context"]
    if fault == "head": record["head"] = "b" * 40
    if fault == "empty_ema": record["ema_leaves"] = {}
    result = run_final(tmp_path, record, run=run, steps=50001)
    assert result.returncode == 1 and "TRAIN_FINAL_RECORD=FAIL" in result.stdout


def test_final_record_20_step_smoke_passes_with_19_tail_rows(tmp_path):
    record = final_fixture("v16-smoke-modul", steps=20)
    result = run_final(tmp_path, record, run="v16-smoke-modul", steps=20)
    assert result.returncode == 0 and "tail_steps=19" in result.stdout


@pytest.mark.parametrize("steps", [60000, 50000, 80000])
def test_final_record_other_steps_rejected_by_cli(tmp_path, steps):
    result = run_final(tmp_path, final_fixture("x", steps=steps), run="x", steps=steps)
    assert result.returncode == 2 and "TRAIN_FINAL_RECORD=PASS" not in result.stdout


@pytest.mark.parametrize("steps", [60000, 50000])
def test_launch_cli_rejects_other_steps(steps):
    result = subprocess.run([sys.executable, str(SCRIPT), "--train-head", HEAD, "--steps", str(steps)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 2 and "LAUNCH_CONTRACT=PASS" not in result.stdout


def test_print_config_and_library():
    for flag, want in (("--print-config", launch.CONFIG_NAME), ("--print-library", "4task-v2-1600ep-604f16da")):
        out = subprocess.run([sys.executable, str(SCRIPT), flag], capture_output=True, text=True, timeout=120)
        assert out.returncode == 0 and out.stdout.strip() == want


def argv_for(steps, *, extra=None, history=MODUL):
    argv = [launch.CONFIG_NAME, "--exp-name", "x", "--model.use-history", "--model.history-config", history]
    return argv + (extra or [])


@pytest.mark.parametrize("steps,extra,ok", [
    (20, ["--num-train-steps", "20"], True),
    (50001, None, True),
    (50001, ["--num-train-steps", "50001"], False),
    (20, None, False),
    (60000, None, False),
    (60000, ["--num-train-steps", "60000"], False),
])
def test_argv_steps(steps, extra, ok):
    rec = launch.Recorder()
    launch.check_argv(rec, argv_for(steps, extra=extra), steps)
    assert ("ARGV_STEPS" not in rec.failed) == ok


def synthetic_base(**override):
    """S3 的 T07 条目＝mme_vla_suite_b128_80k 只改 name 与 num_train_steps=50_001；S3 合入前用此合成。"""
    from mme_vla_suite.training import config
    base = dataclasses.replace(config.get_config("mme_vla_suite_b128_80k"), name=launch.CONFIG_NAME,
                               num_train_steps=50_001)
    return dataclasses.replace(base, **override)


@pytest.mark.parametrize("steps,base_steps,ok", [(20, 50001, True), (50001, 50001, True), (60000, 50001, False),
                                                 (20, 60000, False), (50001, 80000, False)])
def test_cfg_steps(steps, base_steps, ok):
    base = synthetic_base(num_train_steps=base_steps)
    cfg = dataclasses.replace(base, num_train_steps=steps)
    rec = launch.Recorder()
    launch.check_steps(rec, cfg, base, steps)
    assert ("CFG_STEPS" not in rec.failed) == ok
    assert "CFG_SCHEDULE" not in rec.failed


def test_cfg_schedule_mismatch_rejected():
    from openpi.training import optimizer
    base = synthetic_base(lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=5000, peak_lr=5e-5,
                                                                    decay_steps=50001, decay_lr=5e-5))
    rec = launch.Recorder()
    launch.check_steps(rec, base, base, 50001)
    assert rec.failed == ["CFG_SCHEDULE"]


# ── attempt ───────────────────────────────────────────────────────────────


def attempt_files(tmp_path, *, base=None, nn="02", allowed=True, history=MODUL, **override):
    base = base or launch.RUNS["modulation"]
    run = f"{base}-attempt{nn}"
    decision = tmp_path / "decisions/d.json"
    dump(decision, dict(allowed_runs=[run] if allowed else [], quote="工程故障自修"))
    manifest = dict(base_run=base, run=run, history=history, train_head=HEAD, decision=str(decision),
                    bucket=launch.BUCKETS[base] + f"-attempt{nn}")
    manifest.update(override)
    path = tmp_path / "attempt.json"
    dump(path, manifest)
    return path, run


def test_attempt_registered_passes(tmp_path):
    path, run = attempt_files(tmp_path)
    assert launch.check_attempt(path, history_name=MODUL, train_head=HEAD) == run
    path, run = attempt_files(tmp_path / "c", base=launch.RUNS["context"], history=CTX)
    assert launch.check_attempt(path, history_name=CTX, train_head=HEAD) == run


@pytest.mark.parametrize("fault", ["unregistered", "attempt03", "bucket", "history", "head", "fusion", "base",
                                   "extra_key"])
def test_attempt_faults_rejected(tmp_path, fault):
    kwargs = {}
    if fault == "unregistered": kwargs["allowed"] = False
    if fault == "attempt03": kwargs["nn"] = "03"
    if fault == "bucket": kwargs["bucket"] = launch.BUCKETS[launch.RUNS["modulation"]] + "-attempt03"
    if fault == "history": kwargs["history"] = CTX
    if fault == "head": kwargs["train_head"] = "c" * 40
    if fault == "base": kwargs["base"] = "orig100-m32x4x4-modul-motion-s2concat1x1024-b128-60k"
    path, _ = attempt_files(tmp_path, **kwargs) if fault != "base" else (None, None)
    if fault == "base":
        decision = tmp_path / "d.json"
        run = kwargs["base"] + "-attempt02"
        dump(decision, dict(allowed_runs=[run]))
        path = tmp_path / "attempt.json"
        dump(path, dict(base_run=kwargs["base"], run=run, history=MODUL, train_head=HEAD, decision=str(decision),
                        bucket="HongzeFu/x-attempt02"))
    if fault == "extra_key":
        value = json.loads(path.read_text()); value["note"] = "x"; dump(path, value)
    history = CTX if fault == "fusion" else MODUL
    with pytest.raises((ValueError, KeyError)):
        launch.check_attempt(path, history_name=history, train_head=HEAD)


# ── 真实 CLI 解析的核心合同（DATASET 组以桩替代） ──────────────────────────


CONTRACT = dataclasses.replace(launch.Contract(), motion_rows=36484, train_max=71, encoder_sha256="e" * 64)


@pytest.fixture
def fx(tmp_path, monkeypatch):
    """临时仓库只放 4task 两份 yaml 与 orig16 模板；store 根只含 v1-store 下合同读取的目录。"""
    repo = tmp_path / "repo"
    ydir = repo / launch.legacy.YAML_DIR
    ydir.mkdir(parents=True)
    for tag, template in launch.TEMPLATES.items():
        shutil.copyfile(REPO / launch.legacy.YAML_DIR / template, ydir / template)
    for name in launch.HISTORIES:
        tag = name.split("-")[2]
        (ydir / name).write_text(yaml.safe_dump(launch.expected_history(repo, tag, CONTRACT)))
    store = tmp_path / "store"
    (store / "v1-store/models").mkdir(parents=True)
    from mme_vla_suite.training import config
    monkeypatch.setitem(config._CONFIGS_DICT, launch.CONFIG_NAME, synthetic_base())
    monkeypatch.setattr(config, "OPENPI_DATA_HOME", str(store / "v1-store/models"))
    monkeypatch.setattr(launch, "check_dataset", lambda rec, store, contract, repo=None: rec("DATASET_STUB", True))
    return SimpleNamespace(repo=repo, store=store)


def contract_inputs(fx, name, *, steps, run=None):
    c, parent = CONTRACT, fx.store / CONTRACT.parent_rel
    run = run or ("v16-smoke-modul" if steps == 20 else launch.RUNS[launch.history_fusion(name)])
    argv = [launch.CONFIG_NAME, "--exp-name", run, "--dataset-path", str(parent / "framesamp-8x8"),
            "--assets-base-dir", str(fx.store / "v1-store/train-assets"),
            "--data.assets.assets-dir", str(fx.store / c.assets_rel), "--data.assets.asset-id", "robomme",
            "--checkpoint-base-dir", str(fx.store / "v1-store/train-runs"),
            "--weight-loader.params-path", str(fx.store / c.pi05_rel), "--model.use-history",
            "--model.history-config", name, "--no-wandb-enabled"]
    if steps == 20:
        argv += ["--num-train-steps", "20"]
    env = dict(OPENPI_DATA_HOME=str(fx.store / "v1-store/models"), CUDA_VISIBLE_DEVICES=c.gpus,
               XLA_PYTHON_CLIENT_PREALLOCATE="false", MMEVLA_FRAMESAMP_SOURCE=str(parent / "source"),
               MMEVLA_FRAMESAMP_MANIFEST=str(parent / "meta/episode_manifest.json"),
               TRAIN_FINAL_RECORD_DIR=str(fx.store / "v1-store/validation/x/final-record"),
               TRAIN_RECORD_DIR=str(fx.store / "v1-store/validation/x/train-record"))
    args = SimpleNamespace(repo=fx.repo, store_root=fx.store, steps=steps, train_head=HEAD)
    return args, argv, env


@pytest.mark.parametrize("name", launch.HISTORIES)
@pytest.mark.parametrize("steps", [20, 50001])
def test_core_contract_passes_for_both_runs_and_both_steps(fx, name, steps, capsys):
    args, argv, env = contract_inputs(fx, name, steps=steps)
    rec = launch.core(args, argv, contract=CONTRACT, env=env)
    assert not rec.failed, rec.failed
    out = capsys.readouterr().out
    assert "CONCAT_CONFIG=PASS" in out and "MOTION_LAYOUT=PASS readout=concat" in out


@pytest.mark.parametrize("fault", ["steps60000", "formal_with_flag", "smoke_name", "formal_wrong_run",
                                   "placeholder", "history_drift", "readout", "frame4x4"])
def test_core_contract_negatives(fx, fault):
    name, steps = MODUL, 50001
    args, argv, env = contract_inputs(fx, name, steps=steps)
    history = None
    contract = CONTRACT
    if fault == "steps60000": args.steps = 60000
    if fault == "formal_with_flag": argv += ["--num-train-steps", "50001"]
    if fault == "smoke_name":
        args, argv, env = contract_inputs(fx, name, steps=20, run="o16-smoke-x")
    if fault == "formal_wrong_run": argv[2] = launch.RUNS["context"]
    if fault == "placeholder": contract = dataclasses.replace(CONTRACT, train_max=launch.PLACEHOLDER)
    if fault in ("history_drift", "readout", "frame4x4"):
        history = launch.expected_history(fx.repo, "modul", CONTRACT)
        if fault == "history_drift": history["motion"]["stride"] = 16
        if fault == "readout": history["motion"]["num_tokens"] = 4
        if fault == "frame4x4": history["token_per_image"] = 16
    rec = launch.core(args, argv, contract=contract, env=env, history=history)
    assert rec.failed


def test_core_attempt_manifest_passes_and_unregistered_name_fails(fx, tmp_path, capsys):
    path, run = attempt_files(tmp_path)
    args, argv, env = contract_inputs(fx, MODUL, steps=50001, run=run)
    env["ATTEMPT_MANIFEST"] = str(path)
    assert not launch.core(args, argv, contract=CONTRACT, env=env).failed
    assert f"ATTEMPT=PASS run={run}" in capsys.readouterr().out
    args, argv, env = contract_inputs(fx, MODUL, steps=50001, run=run)  # 无清单：attempt 名不被放行
    assert "CFG_RUN" in launch.core(args, argv, contract=CONTRACT, env=env).failed


def test_default_contract_requires_backfill():
    rec = launch.Recorder()
    rec("BUILD_PINS", launch.PLACEHOLDER not in str(dataclasses.asdict(launch.Contract())))
    assert rec.failed == ["BUILD_PINS"]
