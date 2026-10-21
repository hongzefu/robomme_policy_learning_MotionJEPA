"""四对 modulation 正式入口/完成检查的 CPU 合同测试；不启动训练、不占 GPU。"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/training"))
import check_mixed8_pair4_launch as L  # noqa: E402
import check_mixed8_pair4_complete as C  # noqa: E402

MAIN = L.MAIN
STORE = MAIN
PARENT = STORE / L.Contract.parent_rel
ASSETS = STORE / L.Contract.assets_rel
GOOD_SHA = "ab" * 32
GOOD_RUN = "wan-mixed8-200-v10-inseq256q8-b352x8-60ep-20261020-a-final-e60/checkpoint_epoch_60.pt#encoder"
GOOD_STORE = "v1-store/datasets/16task-mixed8-200-v10e60-pair4-20261020-a/motion"


def train_argv(run="pair4-contract-test-nonexistent", steps=None, **replace):
    args = [L.CONFIG_NAME, "--exp-name", run, "--dataset-path", str(PARENT / "framesamp-8x8"),
            "--assets-base-dir", str(STORE / "v1-store/train-assets"),
            "--data.assets.assets-dir", str(ASSETS), "--data.assets.asset-id", "robomme",
            "--checkpoint-base-dir", str(STORE / "v1-store/train-runs"),
            "--weight-loader.params-path", str(STORE / L.Contract.pi05_rel),
            "--model.use-history", "--model.history-config", L.HISTORY_CONFIG]
    for flag, value in replace.items():
        flag = "--" + flag.replace("_", "-").replace("--", ".")
        index = args.index(flag)
        args[index + 1] = value
    if steps is not None:
        args += ["--num-train-steps", str(steps)]
    return args


def rows(func, *args, **kwargs):
    rec = L.Recorder()
    func(rec, *args, **kwargs)
    return {name: ok for name, ok, _ in rec.rows}


@pytest.fixture(scope="module")
def cli_env():
    previous = dict(os.environ)
    os.environ["OPENPI_DATA_HOME"] = str(STORE / "v1-store/models")
    os.chdir(ROOT)
    yield
    os.environ.clear()
    os.environ.update(previous)


@pytest.fixture(scope="module")
def yamls():
    return L.load_yaml(ROOT, L.HISTORY_CONFIG), L.load_yaml(ROOT, L.SMOKE_CONFIG)


def filled(formal, **values):
    result = copy.deepcopy(formal)
    result["motion"].update({"store_path": GOOD_STORE, "source_run": GOOD_RUN, "checkpoint_sha256": GOOD_SHA})
    result["motion"].update(values)
    return result


# ── history YAML ────────────────────────────────────────────────────────────


def test_yaml_placeholder_rejected(yamls):
    placeholder = filled(yamls[0], **{k: L.PLACEHOLDER for k in L.SOURCE_FIELDS})
    got = rows(L.check_history, placeholder, yamls[1])
    assert got["YAML_EQUALS_SMOKE"] and not got["YAML_NO_PLACEHOLDER"]


def test_committed_yaml_formal_fields_pass(yamls):
    assert all(rows(L.check_history, *yamls).values())
    assert yamls[0]["motion"]["checkpoint_sha256"] == "06289905ba9ca7c5879aa32a6e1e88f32d45fc371d506e5a8e5735535a6a20c8"


def test_yaml_filled_passes(yamls):
    assert all(rows(L.check_history, filled(yamls[0]), yamls[1]).values())


def test_yaml_e6_sha_rejected(yamls):
    got = rows(L.check_history, filled(yamls[0], checkpoint_sha256=L.E6_SHA), yamls[1])
    assert not got["YAML_NOT_E6"]
    got = rows(L.check_history, filled(yamls[0], source_run=yamls[1]["motion"]["source_run"]), yamls[1])
    assert not got["YAML_NOT_E6"] and not got["YAML_SOURCE_RUN"]


def test_yaml_smoke_fixture_rejected(yamls):
    got = rows(L.check_history, filled(yamls[0], store_path=yamls[1]["motion"]["store_path"]), yamls[1])
    assert not got["YAML_NOT_SMOKE_FIXTURE"]


def test_yaml_other_field_drift_rejected(yamls):
    bad = filled(yamls[0])
    bad["motion"]["budget"] = 160
    assert not rows(L.check_history, bad, yamls[1])["YAML_EQUALS_SMOKE"]


# ── 参数数组 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("extra", [["--resume"], ["--overwrite"], ["--resum-ckpt-id", "5000"], ["--resum_ckpt_id=5"]])
def test_resume_args_rejected(extra):
    got = rows(L.check_argv, train_argv() + extra, steps=80000)
    assert not got["ARGV_NO_RESUME"] and not got["ARGV_ALLOWLIST"]


@pytest.mark.parametrize("extra", [["--batch-size", "64"], ["--num-workers", "8"], ["--fsdp-devices", "4"]])
def test_parallel_overrides_rejected_by_allowlist(extra):
    assert not rows(L.check_argv, train_argv() + extra, steps=80000)["ARGV_ALLOWLIST"]


def test_formal_steps_must_not_be_overridden():
    assert not rows(L.check_argv, train_argv(steps=300), steps=80000)["ARGV_STEPS"]
    assert all(rows(L.check_argv, train_argv(steps=300), steps=300).values())
    assert all(rows(L.check_argv, train_argv(), steps=80000).values())


# ── 真实 config.cli() ──────────────────────────────────────────────────────


def test_real_cli_formal_passes(cli_env):
    cfg, base = L.parse_cli(train_argv())
    got = rows(L.check_config, cfg, base, store=STORE, contract=L.Contract(), steps=80000)
    assert all(got.values()), got


@pytest.mark.parametrize("extra,key", [(["--batch-size", "64"], "CFG_BATCH_SIZE"),
                                       (["--num-workers", "8"], "CFG_NUM_WORKERS"),
                                       (["--fsdp-devices", "4"], "CFG_FSDP_DEVICES"),
                                       (["--seed", "1"], "CFG_SEED")])
def test_real_cli_parallel_mismatch_rejected(cli_env, extra, key):
    cfg, base = L.parse_cli(train_argv() + extra)
    assert not rows(L.check_config, cfg, base, store=STORE, contract=L.Contract(), steps=80000)[key]


def test_real_cli_old_4task_norm_rejected(cli_env):
    old = str(STORE / "v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da")
    cfg, base = L.parse_cli(train_argv(data__assets__assets_dir=old))
    assert not rows(L.check_config, cfg, base, store=STORE, contract=L.Contract(), steps=80000)["CFG_ASSETS"]
    argv = [t for t in train_argv() if t not in (str(ASSETS),)]
    argv.remove("--data.assets.assets-dir")
    cfg, base = L.parse_cli(argv)
    assert "4task" in cfg.data.assets.assets_dir
    assert not rows(L.check_config, cfg, base, store=STORE, contract=L.Contract(), steps=80000)["CFG_ASSETS"]


def test_real_cli_short_test_keeps_schedule(cli_env):
    cfg, base = L.parse_cli(train_argv(steps=300))
    got = rows(L.check_config, cfg, base, store=STORE, contract=L.Contract(), steps=300)
    assert all(got.values()) and cfg.lr_schedule.decay_steps == 80000


def test_real_cli_smoke_yaml_rejected(cli_env):
    cfg, base = L.parse_cli(train_argv(model__history_config=L.SMOKE_CONFIG))
    assert not rows(L.check_config, cfg, base, store=STORE, contract=L.Contract(), steps=80000)["CFG_MODEL"]


# ── 环境 ────────────────────────────────────────────────────────────────────


def good_env(tmp_path):
    final = tmp_path / "v1-store/diagnostics/x/final-record"
    return {"OPENPI_DATA_HOME": str(tmp_path / "v1-store/models"),
            "MMEVLA_FRAMESAMP_SOURCE": str(tmp_path / L.Contract.parent_rel / "source"),
            "MMEVLA_FRAMESAMP_MANIFEST": str(tmp_path / L.Contract.parent_rel / "meta/episode_manifest.json"),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "TRAIN_FINAL_RECORD_DIR": str(final),
            "TRAIN_RECORD_DIR": str(final.parent / "train-record")}


def test_env_good_passes(tmp_path):
    assert all(rows(L.check_env, good_env(tmp_path), tmp_path, L.Contract()).values())


@pytest.mark.parametrize("key,value,check", [
    ("MMEVLA_MOTION_STORE", "/tmp/x", "ENV_MOTION_STORE_UNSET"),
    ("MMEVLA_MOTION_ALLOW_UNVERIFIED", "1", "ENV_NO_ALLOW_OVERRIDES"),
    ("MMEVLA_FRAMESAMP_ALLOW_UNVERIFIED", "1", "ENV_NO_ALLOW_OVERRIDES"),
    ("MMEVLA_FRAMESAMP_ALLOW_SUBSET", "1", "ENV_NO_ALLOW_OVERRIDES"),
    ("PYTHONPATH", "/tmp", "ENV_PYTHONPATH_UNSET"),
    ("CUDA_VISIBLE_DEVICES", "0,1,2,3", "ENV_GPUS"),
    ("TRAIN_FINAL_RECORD_DIR", "relative/final-record", "ENV_FINAL_RECORD_DIR")])
def test_env_leftovers_rejected(tmp_path, key, value, check):
    env = good_env(tmp_path)
    env[key] = value
    assert not rows(L.check_env, env, tmp_path, L.Contract())[check]


def test_env_nonempty_final_record_rejected(tmp_path):
    env = good_env(tmp_path)
    Path(env["TRAIN_FINAL_RECORD_DIR"]).mkdir(parents=True)
    assert rows(L.check_env, env, tmp_path, L.Contract())["ENV_FINAL_RECORD_DIR"]
    (Path(env["TRAIN_FINAL_RECORD_DIR"]) / "start.json").write_text("{}")
    assert not rows(L.check_env, env, tmp_path, L.Contract())["ENV_FINAL_RECORD_DIR"]


# ── 父库与 norm（真实文件） ─────────────────────────────────────────────────


def test_real_parent_and_norm_pass():
    assert all(rows(L.check_parent, STORE, L.Contract()).values())


def test_old_norm_sha_rejected():
    contract = dataclasses.replace(L.Contract(), assets_rel="v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da")
    assert not rows(L.check_parent, STORE, contract)["NORM_SHA256"]


# ── 新 motion 表元数据（合成） ─────────────────────────────────────────────


def make_store(tmp_path, *, rows_=6, sha=GOOD_SHA, epoch=60):
    contract = dataclasses.replace(L.Contract(), motion_rows=rows_, motion_exec_rows=4, motion_demo_rows=2)
    root = tmp_path / GOOD_STORE
    (root / "meta").mkdir(parents=True)
    (root / "motion_token.f32.bin").write_bytes(b"\0" * (6 * 8192))
    meta = {"layout": contract.motion_layout, "status": "verified", "num_rows": 6,
            "totals": {"rows": 6, "exec_rows": 4, "demo_rows": 2},
            "manifest_sha256": contract.canonical_manifest_sha256,
            "tables": {"motion_token": {"row_shape": [8, 256], "dtype": "float32", "row_bytes": 8192,
                                        "num_rows": 6, "relpath": "motion_token.f32.bin", "byte_count": 6 * 8192}},
            "provenance": {"encoder": {"run_name": GOOD_RUN.split("/")[0], "checkpoint_name": f"checkpoint_epoch_{epoch}.pt",
                                       "epoch": epoch, "state_key": "encoder", "checkpoint_sha256": sha}}}
    (root / "meta/store_meta.json").write_text(json.dumps(meta))
    return contract


def motion_rows_(tmp_path, contract, fields):
    match = re.fullmatch(r"([^/#]+)/checkpoint_epoch_(\d+)\.pt#encoder", fields["source_run"])
    return rows(L.check_motion, tmp_path, tmp_path, fields, match, contract)


def test_motion_store_passes(tmp_path):
    contract = make_store(tmp_path)
    fields = {"store_path": GOOD_STORE, "source_run": GOOD_RUN, "checkpoint_sha256": GOOD_SHA}
    assert all(motion_rows_(tmp_path, contract, fields).values())


def test_motion_store_sha_mismatch_and_rows_rejected(tmp_path):
    contract = make_store(tmp_path, sha="cd" * 32)
    fields = {"store_path": GOOD_STORE, "source_run": GOOD_RUN, "checkpoint_sha256": GOOD_SHA}
    assert not motion_rows_(tmp_path, contract, fields)["MOTION_ENCODER_SHA"]
    contract = dataclasses.replace(contract, motion_rows=128342)
    assert not motion_rows_(tmp_path, contract, fields)["MOTION_ROWS"]


def test_parent_old_motion_store_rejected(tmp_path):
    contract = make_store(tmp_path)
    contract = dataclasses.replace(contract, parent_rel=str(Path(GOOD_STORE).parent))
    fields = {"store_path": GOOD_STORE, "source_run": GOOD_RUN, "checkpoint_sha256": GOOD_SHA}
    assert not motion_rows_(tmp_path, contract, fields)["MOTION_PATH"]


# ── 初始权重叶集合 ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def leaves(cli_env, yamls):
    from omegaconf import OmegaConf
    cfg, _ = L.parse_cli(train_argv())
    return (L.model_leaves(cfg, OmegaConf.create(yamls[0])), L.reference_leaves(ROOT, L.Contract()),
            L.base_leaves(cfg.weight_loader.params_path))


def test_real_leaves_pass(leaves):
    got = rows(L.check_leaves, *leaves)
    assert all(got.values()), got


def test_missing_set_not_memory_rejected(leaves):
    model, reference, base = leaves
    fewer = {k: v for k, v in base.items() if k != "action_in_proj/bias"}
    assert not rows(L.check_leaves, model, reference, fewer)["PI05_MISSING_EQ_MEMORY"]
    more = {**base, "mem_encoder/motion_pos_proj/bias": ([768], "float32")}
    assert not rows(L.check_leaves, model, reference, more)["PI05_MISSING_EQ_MEMORY"]


def test_motion_projection_shape_rejected(leaves):
    model, reference, base = leaves
    bad = {**model, L.MOTION_PROJ: ([1536, 1024], "float32")}
    got = rows(L.check_leaves, bad, reference, base)
    assert not got["MOTION_PROJ_SHAPE"] and not got["MODEL_65_LEAVES"]


# ── 完成检查（合成 final-record 与 checkpoint 目录） ──────────────────────


def fake_run(tmp_path, steps=80000, *, drop=None, wait_uuid=None):
    reference = L.reference_leaves(ROOT, L.Contract())
    leaves = {k: {"dtype": d, "shape": s, "bytes": 1, "sha256": "0" * 64, "finite": True} for k, (s, d) in reference.items()}
    run_root = tmp_path / "run"
    records = tmp_path / "final-record"
    records.mkdir()
    rid, head = str(uuid.uuid4()), "a" * 40
    identity = {"run_uuid": rid, "run_name": "r", "head": head, "budget": 512, "checkpoint_dir": str(run_root),
                "num_train_steps": steps, "save_interval": 5000, "keep_period": 5000, "complete": {"x": 1}, "pid": None}
    (records / "start.json").write_text(json.dumps(identity))
    (records / "final.json").write_text(json.dumps({**identity, "state_step": steps, "loop_step": steps - 1,
                                                    "checkpoint_step": steps - 1, "ema_leaves": leaves}))
    (records / "checkpoint_wait_done.json").write_text(json.dumps(
        {"run_uuid": wait_uuid or rid, "head": head, "wait_until_finished": True}))
    norm = (ASSETS / "robomme/norm_stats.json").read_bytes()
    for step in C.expected_steps_dirs(steps):
        if step == drop:
            continue
        path = run_root / str(step)
        (path / "params").mkdir(parents=True)
        (path / "assets/robomme").mkdir(parents=True)
        (path / "params/manifest.ocdbt").write_bytes(b"x")
        (path / "params/_METADATA").write_bytes(b"x")
        (path / "assets/robomme/norm_stats.json").write_bytes(norm)
        (path / "_CHECKPOINT_METADATA").write_text(json.dumps({"init_timestamp_nsecs": 1, "commit_timestamp_nsecs": 2}))
    args = SimpleNamespace(records=str(records), run_root=str(run_root), steps=steps, run="r", head=head,
                           repo=str(ROOT), mode="s7" if steps == 80000 else "s6", skip_pid=True)
    return args, leaves


def test_complete_synthetic_passes(tmp_path):
    args, leaves = fake_run(tmp_path)
    assert C.expected_steps_dirs(80000) == [*range(5000, 80000, 5000), 79999]
    assert not C.run(args, restore=lambda _: leaves).failed


def test_complete_s6_short_steps(tmp_path):
    args, leaves = fake_run(tmp_path, steps=300)
    assert C.expected_steps_dirs(300) == [299]
    assert not C.run(args, restore=lambda _: leaves).failed


def test_complete_run_uuid_mismatch_rejected(tmp_path):
    args, leaves = fake_run(tmp_path, wait_uuid=str(uuid.uuid4()))
    assert "RECORD_RUN_UUID" in C.run(args, restore=lambda _: leaves).failed


def test_complete_missing_79999_rejected(tmp_path):
    args, leaves = fake_run(tmp_path, drop=79999)
    failed = C.run(args, restore=lambda _: leaves).failed
    assert "CKPT_DIR_SET" in failed and "RESTORE_65_FINITE" in failed


def test_complete_restore_mismatch_rejected(tmp_path):
    args, leaves = fake_run(tmp_path)
    changed = dict(leaves)
    key = next(iter(changed))
    changed[key] = {**changed[key], "dtype": "float32" if changed[key]["dtype"] != "float32" else "bfloat16"}
    assert "RESTORE_EQUALS_FINAL" in C.run(args, restore=lambda _: changed).failed


def test_complete_state_step_rejected(tmp_path):
    args, leaves = fake_run(tmp_path)
    final = json.loads((Path(args.records) / "final.json").read_text())
    final["state_step"] = 79999
    (Path(args.records) / "final.json").write_text(json.dumps(final))
    assert "RECORD_STEPS" in C.run(args, restore=lambda _: leaves).failed


# ── 启动脚本的前置拒绝（不写日志、不起训练） ──────────────────────────────


@pytest.mark.parametrize("env,extra", [
    ({"RUN_NAME": "x", "TRAIN_HEAD": "a" * 40, "FINAL_RECORD_DIR": "/abs/final-record"}, ["--resume"]),
    ({"RUN_NAME": "x", "TRAIN_HEAD": "a" * 40, "FINAL_RECORD_DIR": "rel/final-record"}, []),
    ({"RUN_NAME": "x", "TRAIN_HEAD": "a" * 40, "FINAL_RECORD_DIR": "/abs/final-record", "MAX_STEPS": "80000"}, []),
    ({"RUN_NAME": "x", "TRAIN_HEAD": "short", "FINAL_RECORD_DIR": "/abs/final-record"}, []),
    ({"TRAIN_HEAD": "a" * 40, "FINAL_RECORD_DIR": "/abs/final-record"}, [])])
def test_launcher_rejects_before_launch(env, extra):
    clean = {k: v for k, v in os.environ.items() if k not in ("RUN_NAME", "TRAIN_HEAD", "FINAL_RECORD_DIR", "MAX_STEPS")}
    result = subprocess.run(["bash", str(ROOT / "scripts/training/run_mixed8_pair4_80k.sh"), *extra],
                            env={**clean, **env}, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert not (MAIN / "v1-store/logs/x.log").exists()
