"""stage2 启动合同：真实 CLI／模型树／manifest，以及隔离入口退出码。"""

import copy
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
MAIN = Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")


@pytest.fixture(scope="module")
def launch():
    path = REPO / "scripts/training/check_mixed8_stage2_1024_launch.py"
    spec = importlib.util.spec_from_file_location("_test_stage2_launch", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def argv_for(launch, *, steps=80000):
    contract = launch.Contract()
    run = "mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k" if steps == 80000 else "cc-vla-s2-smoke-contract"
    args = [launch.CONFIG_NAME, "--exp-name", run,
            "--dataset-path", str(MAIN / contract.parent_rel / "framesamp-8x8"),
            "--assets-base-dir", str(MAIN / "v1-store/train-assets"),
            "--data.assets.assets-dir", str(MAIN / contract.assets_rel),
            "--data.assets.asset-id", "robomme",
            "--checkpoint-base-dir", str(MAIN / "v1-store/train-runs"),
            "--weight-loader.params-path", str(MAIN / contract.pi05_rel),
            "--model.use-history", "--model.history-config", launch.HISTORY_CONFIG,
            "--no-wandb-enabled"]
    return args if steps == 80000 else args + ["--num-train-steps", str(steps)]


@pytest.mark.parametrize("steps", [20, 80000])
def test_real_cli_contract(launch, monkeypatch, steps):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(MAIN / "v1-store/models"))
    args = argv_for(launch, steps=steps)
    cfg, base = launch.parse_cli(args)
    record = launch.Recorder()
    launch.check_argv(record, args, steps=steps)
    launch.check_config(record, cfg, base, store=MAIN, contract=launch.Contract(), steps=steps)
    assert not record.failed


@pytest.mark.parametrize("extra", [["--resume"], ["--overwrite"], ["--resum-ckpt-id", "19"],
                                  ["--batch-size", "256"], ["--exp-name", "another"]])
def test_argv_rejects_training_drift(launch, extra):
    record = launch.Recorder()
    launch.check_argv(record, argv_for(launch) + extra, steps=80000)
    assert record.failed


def test_short_run_keeps_formal_schedule(launch, monkeypatch):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(MAIN / "v1-store/models"))
    formal, _ = launch.parse_cli(argv_for(launch))
    smoke, _ = launch.parse_cli(argv_for(launch, steps=20))
    assert dataclasses.replace(smoke, exp_name=formal.exp_name, num_train_steps=80000) == formal


def test_history_bytes_and_sha_gate(launch):
    formal = launch.load_yaml(REPO, launch.HISTORY_CONFIG)
    old = launch.load_yaml(REPO, launch.FRAME_CONFIG)
    record = launch.Recorder()
    launch.check_frame_bytes(record, REPO)
    launch.check_history(record, formal, old)
    # G3 尚未现场填值时必须拒绝启动；填入64位后才能通过格式合同。
    if formal["motion"]["checkpoint_sha256"] == launch.PLACEHOLDER:
        assert record.failed == ["YAML_SHA_FORMAT"]
    else:
        assert not record.failed
    formal["motion"]["checkpoint_sha256"] = "a" * 64
    record = launch.Recorder()
    launch.check_history(record, formal, old)
    assert not record.failed
    formal["motion"]["stride"] = 16
    launch.check_history(record, formal, old)
    assert "YAML_MOTION" in record.failed


def test_manifest_boundaries(launch):
    episodes = [{"global_episode_idx": i, "exec_start_idx": demo, "num_timesteps": demo + execute}
                for i, (demo, execute) in enumerate([(16, 32), (17, 33), (48, 64), (49, 65)])]
    result = launch.manifest_bounds({"episodes": episodes})
    assert result == {"counts": {0: 0, 1: 2, 2: 2, 3: 4}, "rows": 8, "train_max": 4, "eval_max": 42}
    episodes[0]["exec_start_idx"] = True
    with pytest.raises(ValueError, match="长度非法"):
        launch.manifest_bounds({"episodes": episodes})


def test_real_manifest_bounds(launch):
    from mme_vla_suite.datastore.manifest import load_manifest
    path = MAIN / launch.Contract.parent_rel / "meta/episode_manifest.json"
    if not path.exists():
        pytest.skip("真实 mixed8 manifest 不在本机")
    manifest = load_manifest(path)
    assert launch.sha256_file(path) == launch.Contract.episode_manifest_sha256
    record = launch.Recorder()
    result = launch.check_budget(record, manifest, launch.Contract())
    assert not record.failed
    assert (result["rows"], result["train_max"], result["eval_max"]) == (65419, 62, 79)
    changed = copy.deepcopy(manifest)
    changed["episodes"][0]["num_timesteps"] += 32
    record = launch.Recorder()
    launch.check_budget(record, changed, launch.Contract())
    assert "BUDGET_ROWS" in record.failed


@pytest.fixture(scope="module")
def real_trees(launch):
    from omegaconf import OmegaConf
    os.environ["OPENPI_DATA_HOME"] = str(MAIN / "v1-store/models")
    cfg, base = launch.parse_cli(argv_for(launch))
    if not Path(cfg.weight_loader.params_path).exists():
        pytest.skip("真实 pi05_base 不在本机")
    actual_history = launch.load_yaml(REPO, launch.HISTORY_CONFIG)
    frame = launch.load_yaml(REPO, launch.FRAME_CONFIG)
    reference_history = {**frame, "motion": {**launch.MOTION_FIXED, "checkpoint_sha256": "a" * 64}}
    return (launch.model_leaves(cfg, OmegaConf.create(actual_history)),
            launch.model_leaves(base, OmegaConf.create(reference_history)),
            launch.base_leaves(cfg.weight_loader.params_path), launch.model_leaves(base, None))


def test_real_parameter_trees(launch, real_trees):
    record = launch.Recorder()
    launch.check_leaves(record, *real_trees)
    launch.check_pi05_asset(record, REPO, MAIN)
    assert not record.failed


@pytest.mark.parametrize("kind", ["shape", "dtype", "missing", "extra"])
def test_parameter_tree_rejects_drift(launch, real_trees, kind):
    model, reference, base, base_model = copy.deepcopy(real_trees)
    if kind == "shape":
        model[launch.MOTION_PROJ] = ([1280, 1024], "float32")
    elif kind == "dtype":
        key = next(iter(base))
        model[key] = (model[key][0], "float16")
    elif kind == "missing":
        del model[next(iter(base))]
    else:
        base["unexpected/weight"] = ([1], "float32")
    record = launch.Recorder()
    launch.check_leaves(record, model, reference, base, base_model)
    assert record.failed


@pytest.mark.parametrize("fault", [None, "missing", "table_bytes", "index_bytes", "table_sha",
                                  "index_sha", "checkpoint_sha", "scope", "schema", "source_grid",
                                  "missing_field", "compared"])
def test_bitexact_checker_uses_pack_script_and_raw_metadata(launch, tmp_path, fault):
    import numpy as np
    from mme_vla_suite.datastore import motion_store as ms

    pack = launch.load_script(REPO, "scripts/dataset/pack_motion_store.py")
    fixtures = launch.load_script(REPO, "tests/datastore/test_motion_store_stage2.py")
    manifest = fixtures.manifest_for([(165, 100), (82, 50), (32, 0)])
    provenance = fixtures.stage2_identity()
    spec = ms.LAYOUT_SPECS[ms.STAGE2_LAYOUT]
    entries = ms.build_index_entries(manifest, spec)
    totals = ms.index_totals(entries)
    assert totals["rows"] == 7
    source_grid = {key: list(range(0, num_grid * 2, 2))
                   for _, _, key, _, num_grid in pack.iter_rows_in_order(entries)}
    provenance["source_grid_sha256"] = ms.sha256_bytes(ms.canonical_json_bytes(source_grid))
    (tmp_path / "meta").mkdir()
    manifest_path = tmp_path / "meta/episode_manifest.json"
    manifest_path.write_bytes(ms.canonical_json_bytes(manifest))
    index_path = tmp_path / ms.INDEX_RELPATH
    index_path.write_bytes(ms.canonical_json_bytes(ms.index_payload(
        manifest, entries, spec=spec, layout=ms.STAGE2_LAYOUT,
        mj_repo_commit=provenance["source_pin"]["mj_repo_commit"])))
    table_path = tmp_path / ms.MOTION_TABLE_RELPATH
    table_path.write_bytes(np.arange(totals["rows"] * 1024, dtype="<f4").reshape(-1, 1, 1024).tobytes())
    raw = {"schema": ms.META_SCHEMA, "layout": ms.STAGE2_LAYOUT, "status": "packed",
           "manifest_scope": "full", "manifest_sha256": manifest["sha256"],
           "manifest_path": str(manifest_path), "motion_index_sha256": launch.sha256_file(index_path),
           "grid_stride": 32, "window_frames": 33, "grid_origin": "segment_start",
           "window_direction": "forward", "truncation_policy": "none", "frame_size": 256,
           "byte_order": "little", "array_order": "C", **spec.fields(),
           "num_rows": totals["rows"], "totals": totals, "provenance": provenance,
           "tables": {"motion_token": {"relpath": ms.MOTION_TABLE_RELPATH,
                       "row_shape": [1, 1024], "dtype": "float32", "row_bytes": 4096,
                       "num_rows": totals["rows"], "byte_count": table_path.stat().st_size,
                       "sha256": launch.sha256_file(table_path)}}}
    (tmp_path / ms.META_RELPATH).write_bytes(ms.canonical_json_bytes(raw))
    # 实际元数据读取与 fast 检查先通过，避免只凑认证器读取的几个字段。
    loaded = ms.MotionMeta.load(tmp_path)
    ms.run_fast_checks(loaded)
    assert loaded.row_shape == (1, 1024) and loaded.spec.grid_stride == 32
    record = {"schema": "stage2-encoder-bitexact-v1", "scope": "full", "layout": ms.STAGE2_LAYOUT,
              "status": "PASS", "compared": totals["rows"], "mismatches": 0,
              "store_table_sha256": raw["tables"]["motion_token"]["sha256"],
              "motion_index_sha256": raw["motion_index_sha256"],
              "checkpoint_sha256": provenance["encoder"]["checkpoint_sha256"],
              "source_grid_sha256": provenance["source_grid_sha256"]}
    path = tmp_path / pack.BITEXACT_RELPATH
    path.write_text(json.dumps(record))
    assert pack.check_bitexact_record(tmp_path, loaded.raw) == record
    if fault is None:
        return
    if fault == "missing":
        path.unlink()
    elif fault == "table_bytes":
        blob = bytearray(table_path.read_bytes())
        blob[0] ^= 1
        table_path.write_bytes(blob)
    elif fault == "index_bytes":
        index_path.write_bytes(index_path.read_bytes() + b"\n")
    else:
        if fault == "missing_field":
            del record["source_grid_sha256"]
        else:
            field, value = {"table_sha": ("store_table_sha256", "0" * 64),
                            "index_sha": ("motion_index_sha256", "0" * 64),
                            "checkpoint_sha": ("checkpoint_sha256", "0" * 64),
                            "scope": ("scope", "sample"), "schema": ("schema", "wrong-schema"),
                            "source_grid": ("source_grid_sha256", "0" * 64),
                            "compared": ("compared", totals["rows"] - 1)}[fault]
            record[field] = value
        path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="不符|比较记录"):
        pack.check_bitexact_record(tmp_path, loaded.raw)


def shell_fixture(tmp_path):
    """只替换固定根目录，用假 uv 观察入口参数与状态传递，不启动真实训练。"""
    main = tmp_path / "repo"
    (main / "scripts/training").mkdir(parents=True)
    (main / "v1-store/logs").mkdir(parents=True)
    (main / "scripts/training/paths.sh").write_text(
        'V1_STORE="$MAIN/v1-store"\nexport OPENPI_DATA_HOME="$V1_STORE/models"\n')
    source = (REPO / "scripts/training/run_mixed8_stage2_1024_80k.sh").read_text()
    script = main / "scripts/training/run.sh"
    script.write_text(source.replace(f"MAIN={MAIN}", f"MAIN={main}"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                  "check='check_mixed8_stage2_1024_launch.py' in ' '.join(sys.argv)\n"
                  "with open(os.environ['CAPTURE'], 'a') as f: "
                  "f.write(json.dumps({'argv':sys.argv[1:],'env':dict(os.environ)})+'\\n')\n"
                  "sys.exit(int(os.environ['CHECK_RC' if check else 'TRAIN_RC']))\n")
    uv.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "RUN_NAME": "cc-vla-s2-smoke-contract", "MAX_STEPS": "20", "TRAIN_HEAD": "a" * 40,
           "FINAL_RECORD_DIR": str(main / "v1-store/validation/smoke/final-record"),
           "CAPTURE": str(tmp_path / "capture.jsonl"), "CHECK_RC": "0", "TRAIN_RC": "0",
           "MMEVLA_TEST_ALLOW_UNVERIFIED": "1", "MMEVLA_MOTION_STORE": "/wrong", "XLA_FLAGS": "wrong"}
    return script, main, bin_dir, env


@pytest.mark.parametrize("check_rc,train_rc,tee_rc", [(0, 0, 0), (17, 0, 0), (0, 23, 0), (0, 0, 31), (0, 23, 31)])
def test_shell_exit_codes_and_same_argv(tmp_path, check_rc, train_rc, tee_rc):
    script, main, bin_dir, env = shell_fixture(tmp_path)
    env.update(CHECK_RC=str(check_rc), TRAIN_RC=str(train_rc))
    if tee_rc:
        tee = bin_dir / "tee"
        tee.write_text(f"#!/usr/bin/env bash\ncat >/dev/null\nexit {tee_rc}\n")
        tee.chmod(0o755)
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == (check_rc or train_rc or tee_rc), result.stderr
    calls = [json.loads(line) for line in Path(env["CAPTURE"]).read_text().splitlines()]
    assert len(calls) == (1 if check_rc else 2)
    checker_args = calls[0]["argv"]
    assert calls[0]["env"]["JAX_PLATFORMS"] == "cpu"
    if not check_rc:
        assert calls[1]["argv"][4:] == checker_args[checker_args.index("--") + 1:]
        assert "JAX_PLATFORMS" not in calls[1]["env"]
    for call in calls:
        assert call["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
        assert call["env"]["UV_PROJECT_ENVIRONMENT"] == str(main / ".venv")
        assert not any(key.endswith("_ALLOW_UNVERIFIED") for key in call["env"])
        assert "PYTHONPATH" not in call["env"]
        assert "MMEVLA_MOTION_STORE" not in call["env"]
    log = main / f"v1-store/logs/{env['RUN_NAME']}.log"
    assert f"EXIT_CODE={result.returncode}\n" in log.read_text()


@pytest.mark.parametrize("kind", ["extra_arg", "steps", "head", "existing_log"])
def test_shell_rejects_before_python(tmp_path, kind):
    script, main, _, env = shell_fixture(tmp_path)
    args = ["bash", str(script)]
    if kind == "extra_arg":
        args.append("--overwrite")
    elif kind == "steps":
        env["MAX_STEPS"] = "21"
    elif kind == "head":
        env["TRAIN_HEAD"] = "short"
    else:
        (main / f"v1-store/logs/{env['RUN_NAME']}.log").write_text("原日志\n")
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert not Path(env["CAPTURE"]).exists()
    if kind == "existing_log":
        assert (main / f"v1-store/logs/{env['RUN_NAME']}.log").read_text() == "原日志\n"
