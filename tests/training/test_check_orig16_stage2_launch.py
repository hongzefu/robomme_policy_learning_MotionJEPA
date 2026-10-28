"""新启动合同：完整缓存证据小库、真实CLI解析及实际shell边界。"""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests/dataset"))
sys.path.insert(0, str(REPO / "scripts/dataset"))
spec = importlib.util.spec_from_file_location("orig16_launch", REPO / "scripts/training/check_orig16_stage2_launch.py")
launch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launch
spec.loader.exec_module(launch)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_fixture(root):
    """全部读取都绑定临时根；不查主检出产物、真实模型或网络。"""
    from test_pack_motion_stage2_cached import cached_fx, pack, verify_args
    from mme_vla_suite.datastore import motion_store as ms
    with pytest.MonkeyPatch.context() as patch:
        source = cached_fx.__wrapped__(root, patch)
        pack.cmd_pack(source.args)
        pack.cmd_verify(verify_args(source))
    parent = root / "v1-store/datasets/fixture"
    (parent / "meta").mkdir(parents=True)
    shutil.copyfile(source.fx.manifest_path, parent / "meta/episode_manifest.json")
    shutil.copyfile(source.fx.manifest_path.parent / "input_manifest.json", parent / "meta/input_manifest.json")
    shutil.copytree(source.args.out, parent / "motion-stage2-1024")
    (parent / "source/data").mkdir(parents=True)
    for layout in ("4x4", "8x8"):
        dump(parent / f"framesamp-{layout}/meta/store_meta.json", dict(layout=f"framesamp-{layout}-v1",
            status="verified", manifest_scope="full", manifest_sha256=source.fx.manifest["sha256"]))
    norm = root / "v1-store/train-assets/mme_vla_suite/fixture/robomme/norm_stats.json"
    dump(norm, dict(norm_stats={}))
    inputs = json.loads((parent / "meta/input_manifest.json").read_text())["files"]
    encoder = root / "v1-store/external/motionjepa" / source.fx.enc_dir.name
    shutil.copytree(source.fx.enc_dir, encoder)
    training_source = root / "encoder-training-source"
    dump(training_source / "control/build_spec.json", dict(identity=dict(source_pin=dict(sha256="e" * 64,
        files=[dict(path=name, bytes=value["size"], sha256=value["sha256"]) for name, value in inputs.items()]))))
    (encoder / "config.yaml").write_text(f"data:\n  data_root: {training_source}\n")
    meta_path = parent / "motion-stage2-1024" / ms.META_RELPATH
    meta = json.loads(meta_path.read_text())
    meta["provenance"]["encoder"]["config_sha256"] = launch.sha256_file(encoder / "config.yaml")
    dump(meta_path, meta)
    mj = root / "mj-source.json"
    dump(mj, dict(files={name: dict(h5=dict(file=name, **value)) for name, value in inputs.items()}))
    contract = launch.Contract(library="fixture", encoder_run=source.fx.enc_dir.name,
        encoder_sha256=source.fx.ckpt_sha, norm_sha256=launch.sha256_file(norm),
        episode_manifest_sha256=launch.sha256_file(parent / "meta/episode_manifest.json"),
        input_manifest_sha256=launch.sha256_file(parent / "meta/input_manifest.json"),
        canonical_manifest_sha256=source.fx.manifest["sha256"], motion_rows=64, train_max=4,
        mj_source_manifest=str(mj), episodes=16, tasks=16)
    return SimpleNamespace(root=root, parent=parent, contract=contract)


def arguments(fx, name, *, steps=20):
    c = fx.contract
    h = launch.load_yaml(REPO, name)
    h["motion"].update(store_path=c.parent_rel + "/motion-stage2-1024", checkpoint_sha256=c.encoder_sha256,
                       source_run=f"{c.encoder_run}/checkpoint_epoch_105.pt#model")
    layout = "framesamp-4x4" if h["token_per_image"] == 16 else "framesamp-8x8"
    run = "o16-smoke-fixture" if steps == 20 else launch.RUNS[h["integration_type"]]
    argv = [launch.CONFIG_NAME, "--exp-name", run,
        "--dataset-path", str(fx.parent / layout), "--assets-base-dir", str(fx.root / "v1-store/train-assets"),
        "--data.assets.assets-dir", str(fx.root / c.assets_rel), "--data.assets.asset-id", "robomme",
        "--checkpoint-base-dir", str(fx.root / "v1-store/train-runs"),
        "--weight-loader.params-path", str(fx.root / c.pi05_rel), "--model.use-history",
        "--model.history-config", name, "--no-wandb-enabled"]
    if steps == 20:
        argv += ["--num-train-steps", "20"]
    env = dict(OPENPI_DATA_HOME=str(fx.root / "v1-store/models"), CUDA_VISIBLE_DEVICES=c.gpus,
        XLA_PYTHON_CLIENT_PREALLOCATE="false", MMEVLA_FRAMESAMP_SOURCE=str(fx.parent / "source"),
        MMEVLA_FRAMESAMP_MANIFEST=str(fx.parent / "meta/episode_manifest.json"),
        TRAIN_FINAL_RECORD_DIR=str(fx.root / "v1-store/validation/final-record"),
        TRAIN_RECORD_DIR=str(fx.root / "v1-store/validation/train-record"))
    args = SimpleNamespace(repo=REPO, store_root=fx.root, steps=steps)
    return args, argv, env, h


def run_contract(fx, name, *, fault=None):
    args, argv, env, h = arguments(fx, name)
    c = fx.contract
    if fault == "fusion": h["memory_token_dim"] = 17
    if fault == "readout": h["motion"]["num_tokens"] = 2
    if fault == "gpu": env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    if fault == "steps": args.steps = 60001
    if fault == "placeholder": c = dataclasses.replace(c, train_max=launch.PLACEHOLDER)
    from mme_vla_suite.training import config
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "OPENPI_DATA_HOME", env["OPENPI_DATA_HOME"])
        return launch.core(args, argv, contract=c, env=env, history=h)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return make_fixture(tmp_path_factory.mktemp("o16-contract"))


@pytest.mark.parametrize("name", launch.HISTORIES)
def test_all_eight_real_cli_and_fixture_contracts(fx, name):
    assert not run_contract(fx, name).failed


@pytest.mark.parametrize("fault", ["fusion", "readout", "gpu", "steps", "placeholder"])
def test_contract_negatives(fx, fault):
    assert run_contract(fx, launch.HISTORIES[0], fault=fault).failed


@pytest.mark.parametrize("fault", ["layout", "g2_missing", "lineage"])
def test_disk_identity_failures(fx, fault):
    if fault == "layout":
        path = fx.parent / "framesamp-4x4/meta/store_meta.json"
        before = path.read_bytes()
        value = json.loads(before); value["layout"] = "framesamp-8x8-v1"
        path.write_text(json.dumps(value))
    elif fault == "lineage":
        path = Path(fx.contract.mj_source_manifest); before = path.read_bytes()
        value = json.loads(before); next(iter(value["files"].values()))["h5"]["sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    else:
        meta = json.loads((fx.parent / "motion-stage2-1024/meta/store_meta.json").read_text())
        path = Path(meta["provenance"]["stage2_compat"]["evidence"]["G2"]["path"])
        before = path.read_bytes(); path.unlink()
    try:
        assert run_contract(fx, launch.HISTORIES[0]).failed
    finally:
        path.write_bytes(before)


def shell_tree(tmp_path):
    root = tmp_path / "repo"
    scripts = root / "scripts/training"
    scripts.mkdir(parents=True)
    for name in ("run_orig16_stage2_60k.sh", "run_orig16_ctx_then_modul_60k.sh"):
        shutil.copyfile(REPO / "scripts/training" / name, scripts / name)
    return root


@pytest.mark.parametrize("checker_exit", [0, 7])
def test_actual_shell_dry_run_creates_no_log_run_or_cache(tmp_path, checker_exit):
    root = shell_tree(tmp_path)
    py = root / ".venv/bin/python"; py.parent.mkdir(parents=True)
    py.write_text(f'''#!/usr/bin/env bash
if [[ "$*" == *--print-library* ]]; then echo fixture; exit 0; fi
[[ "$1" == scripts/training/check_orig16_stage2_launch.py ]] || exit 91
[[ "$JAX_PLATFORMS" == cpu && "$*" == *--num-train-steps* ]] || exit 92
echo CONTRACT_INVOKED=1
exit {checker_exit}
'''); py.chmod(0o755)
    before = {str(p.relative_to(root)) for p in root.rglob("*")}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", DRY_RUN="1", MAX_STEPS="20",
        RUN_NAME="o16-smoke-shell", HISTORY=launch.HISTORIES[0], TRAIN_HEAD="a" * 40,
        FINAL_RECORD_DIR=str(root / "v1-store/validation/final-record"))
    result = subprocess.run(["bash", str(root / "scripts/training/run_orig16_stage2_60k.sh")], env=env,
                            capture_output=True, text=True)
    assert result.returncode == checker_exit, result.stdout + result.stderr
    assert "TRAIN_CMD=" in result.stdout and "CONTRACT_INVOKED=1" in result.stdout
    assert {str(p.relative_to(root)) for p in root.rglob("*")} == before


@pytest.mark.parametrize("fault", ["exit", "checkpoint", "log", "nan", "grad_inf", "oom", None])
def test_serial_driver_requires_exit_log_and_59999(tmp_path, fault):
    root = shell_tree(tmp_path)
    child = root / "scripts/training/run_orig16_stage2_60k.sh"
    child.write_text(f'''#!/usr/bin/env bash
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
echo "$RUN_NAME" >> "$ROOT/calls"
[[ {fault!s} != exit ]] || exit 9
mkdir -p "$ROOT/v1-store/logs"
printf 'RUN=%s\\ninference nan_count=0 bloom NaN/Inf说明\\nStep 59999: loss=0.25, grad_norm=0.5\\nEXIT_CODE=%s\\n' "$RUN_NAME" {'3' if fault == 'log' else '0'} > "$ROOT/v1-store/logs/$RUN_NAME.log"
if [[ {fault!s} == nan ]]; then printf 'Step 59999: loss=nan\\nEXIT_CODE=0\\n' >> "$ROOT/v1-store/logs/$RUN_NAME.log"; fi
if [[ {fault!s} == grad_inf ]]; then printf 'Step 59999: loss=0.2, memory_grad_norm=inf\\nEXIT_CODE=0\\n' >> "$ROOT/v1-store/logs/$RUN_NAME.log"; fi
if [[ {fault!s} == oom ]]; then printf 'RESOURCE_EXHAUSTED: failed to allocate\\nEXIT_CODE=0\\n' >> "$ROOT/v1-store/logs/$RUN_NAME.log"; fi
[[ {fault!s} == checkpoint ]] || mkdir -p "$ROOT/v1-store/train-runs/{launch.CONFIG_NAME}/$RUN_NAME/59999"
exit 0
''')
    env = dict(os.environ, TRAIN_HEAD="a" * 40, FINAL_ROOT=str(root / "v1-store/validation"))
    env.pop("MAX_STEPS", None); env.pop("DRY_RUN", None)
    result = subprocess.run(["bash", str(root / "scripts/training/run_orig16_ctx_then_modul_60k.sh")],
                            env=env, capture_output=True, text=True)
    calls = (root / "calls").read_text().splitlines()
    assert calls == [launch.RUNS["context"]] + ([launch.RUNS["modulation"]] if fault is None else [])
    assert (result.returncode == 0) == (fault is None), result.stdout + result.stderr
