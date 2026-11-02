"""GreatLakes 复现入口与合同（1031 计划 S1、V5）：卡数 2/4 × batch 组合表、前缀放行、默认 8 卡 B128 不变。"""
import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_check_orig16_stage2_launch as base  # noqa: E402  复用完整缓存证据小库夹具

launch = base.launch
TIERS = tuple(launch.TIER_HISTORIES)


def flatten_evidence(fx):
    """按 runbook 布局把五份证据与在线输入平铺到 v1-store/validation/orig16-stage2。"""
    from mme_vla_suite.datastore import motion_store as ms
    motion = ms.MotionMeta.load(fx.parent / "motion-stage2-1024")
    root = fx.root / "v1-store/validation/orig16-stage2"
    root.mkdir(parents=True, exist_ok=True)
    for item in motion.provenance["stage2_compat"]["evidence"].values():
        shutil.copyfile(item["path"], root / Path(item["path"]).name)
    npz = root / "online_restore_inputs.npz"
    npz.write_bytes(b"synthetic-npz")
    (root / launch.ONLINE_INPUTS).write_text(json.dumps(
        {"payload": {"path": npz.name, "sha256": launch.sha256_file(npz)}}))
    return root, motion


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    fx = base.make_fixture(tmp_path_factory.mktemp("o16gl-contract"))
    fx.evidence_root, fx.motion = flatten_evidence(fx)
    return fx


def gl_arguments(fx, name, *, steps, run, fsdp, batch, visible=None, contract=None, config_name=launch.CONFIG_NAME):
    c = contract or fx.contract
    h = launch.load_yaml(REPO, name)
    if "motion" in h:
        h["motion"].update(store_path=c.parent_rel + "/" + c.store_dir, checkpoint_sha256=c.encoder_sha256,
                           source_run=f"{c.encoder_run}/checkpoint_epoch_105.pt#model")
    argv = [config_name, "--exp-name", run, "--dataset-path", str(fx.parent / "framesamp-4x4"),
            "--assets-base-dir", str(fx.root / "v1-store/train-assets"),
            "--data.assets.assets-dir", str(fx.root / c.assets_rel), "--data.assets.asset-id", "robomme",
            "--checkpoint-base-dir", str(fx.root / "v1-store/train-runs"),
            "--weight-loader.params-path", str(fx.root / c.pi05_rel), "--model.use-history",
            "--model.history-config", name, "--no-wandb-enabled"]
    if steps != (80000 if config_name == launch.ABLATION_CONFIG else 60000):
        argv += ["--num-train-steps", str(steps)]
    argv += ["--fsdp-devices", str(fsdp), "--num-workers", str(2 * fsdp), "--batch-size", str(batch)]
    visible = ",".join(str(i) for i in range(fsdp)) if visible is None else visible
    env = dict(OPENPI_DATA_HOME=str(fx.root / "v1-store/models"), CUDA_VISIBLE_DEVICES=visible,
               XLA_PYTHON_CLIENT_PREALLOCATE="false", MMEVLA_FRAMESAMP_SOURCE=str(fx.parent / "source"),
               MMEVLA_FRAMESAMP_MANIFEST=str(fx.parent / "meta/episode_manifest.json"),
               TRAIN_FINAL_RECORD_DIR=str(fx.root / "v1-store/validation/final-record"),
               TRAIN_RECORD_DIR=str(fx.root / "v1-store/validation/train-record"))
    if "motion" in h:
        env[launch.EVIDENCE_ENV] = str(fx.evidence_root)
    args = SimpleNamespace(repo=REPO, store_root=fx.root, steps=steps, fsdp_devices=fsdp, batch_size=batch,
                           gpus_visible=visible, config_name=config_name)
    return args, argv, env, h


def gl_core(fx, name=launch.GL_HISTORY, *, steps=60000, run=None, fsdp=4, batch=64, mutate=None, contract=None,
            visible=None):
    run = run or launch.gl_formal_run(batch)
    args, argv, env, h = gl_arguments(fx, name, steps=steps, run=run, fsdp=fsdp, batch=batch, visible=visible,
                                      contract=contract)
    if mutate:
        mutate(args, argv, env, h)
    from mme_vla_suite.training import config
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "OPENPI_DATA_HOME", env["OPENPI_DATA_HOME"])
        return launch.core(args, argv, contract=contract or fx.contract, env=env, history=h)


# ── 正式口径：GPUS 2/4 × batch 64/32 通过；128、16 与测时 yaml 被拒 ──────────────────

@pytest.mark.parametrize("fsdp", [2, 4])
@pytest.mark.parametrize("batch", [64, 32])
def test_gl_formal_passes(fx, fsdp, batch, capsys):
    rec = gl_core(fx, fsdp=fsdp, batch=batch)
    assert not rec.failed, rec.failed
    out = capsys.readouterr().out
    assert f"GPU_SCOPE=PASS visible={','.join(str(i) for i in range(fsdp))} fsdp={fsdp}" in out
    assert "EVIDENCE_REBASE=PASS files=7 sha_mismatch=0" in out
    assert "CHECK_CFG_BATCH_SIZE=PASS" in out and "CHECK_CFG_NUM_WORKERS=PASS" in out


@pytest.mark.parametrize("fsdp", [2, 4])
@pytest.mark.parametrize("batch", [128, 16, 8, 4])
def test_gl_formal_rejects_batch(fx, fsdp, batch):
    assert "GL_COMBO" in gl_core(fx, fsdp=fsdp, batch=batch).failed


@pytest.mark.parametrize("tier", TIERS)
def test_gl_formal_rejects_tier_yaml(fx, tier):
    rec = gl_core(fx, tier)
    assert {"GL_COMBO", "CFG_RUN"} <= set(rec.failed)


@pytest.mark.parametrize("name", [launch.pair_name("modul", "4x256"), launch.pair_name("ctx", "1x1024")])
def test_gl_formal_rejects_other_history(fx, name):
    assert "GL_COMBO" in gl_core(fx, name).failed


def test_gl_formal_run_name_must_match_batch(fx):
    rec = gl_core(fx, batch=64, run=launch.gl_formal_run(32))
    assert {"GL_COMBO", "CFG_RUN"} <= set(rec.failed)


@pytest.mark.parametrize("registered", [False, True])
def test_gl_attempt_requires_matching_manifest(fx, tmp_path, registered):
    path, run = base.attempt_files(tmp_path, base=launch.gl_formal_run(64), history=launch.GL_HISTORY)
    def mutate(args, argv, env, h):
        args.train_head = "a" * 40
        if registered:
            env["ATTEMPT_MANIFEST"] = str(path)
    rec = gl_core(fx, run=run, mutate=mutate)
    assert ("CFG_RUN" not in rec.failed) == registered
    if registered:
        assert not rec.failed, rec.failed


def test_gl_num_workers_must_be_twice_fsdp(fx):
    def mutate(args, argv, env, h):
        argv[argv.index("--num-workers") + 1] = "16"
    assert "CFG_NUM_WORKERS" in gl_core(fx, fsdp=4, mutate=mutate).failed


@pytest.mark.parametrize("visible", ["0,1", "0,1,2,3,4"])
def test_gl_visible_gpu_count_must_match(fx, visible):
    assert "GL_GPU_COUNT" in gl_core(fx, fsdp=4, visible=visible).failed


def test_gl_slurm_empty_visible_passes(fx, capsys):
    """Slurm 下入口不设 CUDA_VISIBLE_DEVICES：合同按传入空串核，卡数由入口 nvidia-smi 核。"""
    def mutate(args, argv, env, h):
        env.pop("CUDA_VISIBLE_DEVICES")
    assert not gl_core(fx, fsdp=4, visible="", mutate=mutate).failed
    assert "GPU_SCOPE=PASS visible=<slurm> fsdp=4" in capsys.readouterr().out


def test_gl_formal_requires_evidence_root(fx):
    def mutate(args, argv, env, h):
        env.pop(launch.EVIDENCE_ENV)
    rec = gl_core(fx, mutate=mutate)
    assert "ENV_EVIDENCE_ROOT" in rec.failed


def test_gl_rebased_evidence_sha_mismatch_fails(fx):
    path = fx.evidence_root / "G2.json" if (fx.evidence_root / "G2.json").exists() else next(
        p for p in fx.evidence_root.iterdir() if p.name != launch.ONLINE_INPUTS and p.suffix == ".json")
    before = path.read_bytes()
    path.write_bytes(before + b" ")
    try:
        assert "DATASET" in gl_core(fx).failed
    finally:
        path.write_bytes(before)


# ── 冒烟前缀：20 步 64/32/16/8/4、100 步 64/32 通过 ──────────────────────────────

@pytest.mark.parametrize("fsdp", [2, 4])
@pytest.mark.parametrize("steps,batch", [(20, 64), (20, 32), (20, 16), (20, 8), (20, 4), (100, 64), (100, 32)])
def test_gl_smoke_passes(fx, fsdp, steps, batch):
    rec = gl_core(fx, steps=steps, batch=batch, fsdp=fsdp, run="o16gl-smoke-fixture")
    assert not rec.failed, rec.failed


@pytest.mark.parametrize("steps,batch", [(100, 16), (100, 8), (100, 4), (20, 128), (60000, 64)])
def test_gl_smoke_rejects(fx, steps, batch):
    rec = gl_core(fx, steps=steps, batch=batch, run="o16gl-smoke-fixture")
    assert "GL_COMBO" in rec.failed


@pytest.mark.parametrize("tier", TIERS)
def test_gl_smoke_rejects_tier_yaml(fx, tier):
    assert "GL_COMBO" in gl_core(fx, tier, steps=100, batch=64, run="o16gl-smoke-fixture").failed


# ── 测时前缀：只有 100 步 64 + 三份测时 yaml 通过 ───────────────────────────────────

@pytest.mark.parametrize("tier", TIERS)
def test_gl_tier_passes(fx, tier, capsys):
    rec = gl_core(fx, tier, steps=100, batch=64, run="o16gl-tier-fixture")
    assert not rec.failed, rec.failed
    out = capsys.readouterr().out
    budget = launch.TIER_HISTORIES[tier]
    assert f"FRAME_LAYOUT=PASS layout=4x4 frames={budget // 16} tokens_per_frame=16 budget={budget}" in out
    assert "MOTION_LAYOUT=SKIP motion=0 reason=tier" in out
    assert "EVIDENCE_REBASE" not in out and "LINEAGE" not in out


@pytest.mark.parametrize("steps,batch", [(100, 32), (20, 64), (20, 32), (60000, 64)])
def test_gl_tier_rejects_combo(fx, steps, batch):
    assert "GL_COMBO" in gl_core(fx, TIERS[1], steps=steps, batch=batch, run="o16gl-tier-fixture").failed


def test_gl_tier_rejects_formal_yaml(fx):
    assert "GL_COMBO" in gl_core(fx, steps=100, batch=64, run="o16gl-tier-fixture").failed


def test_gl_tier_rejects_evidence_root(fx):
    def mutate(args, argv, env, h):
        env[launch.EVIDENCE_ENV] = str(fx.evidence_root)
    assert "ENV_EVIDENCE_ROOT" in gl_core(fx, TIERS[0], steps=100, batch=64, run="o16gl-tier-fixture",
                                          mutate=mutate).failed


def test_gl_tier_yaml_drift_fails(fx):
    def mutate(args, argv, env, h):
        h["pool_type"] = "max"
    rec = gl_core(fx, TIERS[0], steps=100, batch=64, run="o16gl-tier-fixture", mutate=mutate)
    assert "HISTORY_KEYS" in rec.failed


def test_tier_yamls_only_drop_motion_and_change_budget():
    formal = launch.load_yaml(REPO, launch.GL_HISTORY)
    for name, budget in launch.TIER_HISTORIES.items():
        tier = launch.load_yaml(REPO, name)
        assert {k: v for k, v in formal.items() if k != "motion"} | {"budget": budget} == tier
        assert "仅供 o16gl-tier 测时" in (REPO / launch.YAML_DIR / name).read_text().splitlines()[0]


# ── 默认参数（fsdp 8、batch 128）不变；非 GL 口径不认测时 yaml 与其他 batch ──────────

def test_default_contract_unchanged(fx, capsys):
    args, argv, env, h = base.arguments(fx, launch.GL_HISTORY, steps=60000)
    from mme_vla_suite.training import config
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "OPENPI_DATA_HOME", env["OPENPI_DATA_HOME"])
        rec = launch.core(args, argv, contract=fx.contract, env=env, history=h)
    assert not rec.failed
    out = capsys.readouterr().out
    assert "GPU_SCOPE=PASS allowed=0-7" in out and "GL_" not in out and "EVIDENCE_REBASE" not in out
    assert "LINEAGE_ROUTE" not in out


def test_default_fsdp8_rejects_gl_batch_and_tier(fx):
    def run(name, batch, steps):
        args, argv, env, h = gl_arguments(fx, name, steps=steps, run="o16-smoke-fixture", fsdp=8, batch=batch)
        args.gpus_visible = None
        env["CUDA_VISIBLE_DEVICES"] = fx.contract.gpus
        from mme_vla_suite.training import config
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(config, "OPENPI_DATA_HOME", env["OPENPI_DATA_HOME"])
            return launch.core(args, argv, contract=fx.contract, env=env, history=h)
    assert "GL_COMBO" in run(launch.GL_HISTORY, 64, 20).failed
    assert "HISTORY_NAME" in run(TIERS[0], 128, 20).failed
    assert "ARGV_STEPS" in run(launch.GL_HISTORY, 128, 100).failed


ABLATION_RUNS = tuple(run for run in launch.ENCODERS if run != launch.ENCODER_RUN)


def ablation_fixture(tmp_path, monkeypatch, encoder_run):
    """外部资产身份替换为完整证据小库，保留真实配置及 yaml。"""
    fx = base.make_fixture(tmp_path)
    fx.evidence_root, fx.motion = flatten_evidence(fx)
    identity = dict(launch.ENCODERS[encoder_run])
    encoder = fx.root / "v1-store/external/motionjepa"
    shutil.copytree(encoder / fx.contract.encoder_run, encoder / encoder_run)
    shutil.copytree(fx.parent / "motion-stage2-1024", fx.parent / identity["store_dir"])
    identity.update(ckpt_sha256=fx.contract.encoder_sha256,
                    config_sha256=launch.sha256_file(encoder / encoder_run / "config.yaml"))
    monkeypatch.setitem(launch.ENCODERS, encoder_run, identity)
    fx.contract = dataclasses.replace(fx.contract, encoder_run=encoder_run, store_dir=identity["store_dir"],
                                      evidence_dir=identity["evidence_dir"])
    # 缺源清单时必须走当前 encoder 的证据目录，旧 evidence/ 不参与。
    folder = fx.parent / identity["evidence_dir"]
    folder.mkdir()
    from mme_vla_suite.datastore import motion_store as ms
    for role in ms.stage2_evidence_roles(fx.motion.provenance):
        shutil.copyfile(fx.motion.provenance["stage2_compat"]["evidence"][role]["path"], folder / f"{role}.json")
    fx.contract = dataclasses.replace(fx.contract, mj_source_manifest=str(tmp_path / "absent.json"))
    return fx, identity


@pytest.mark.parametrize("encoder_run", ABLATION_RUNS)
@pytest.mark.parametrize("steps", [20, 100, 80000])
def test_ablation_full_contract(tmp_path, monkeypatch, encoder_run, steps):
    """真实 CLI 与真实新 yaml；仅外部资产身份替换为完整证据小库。"""
    fx, identity = ablation_fixture(tmp_path, monkeypatch, encoder_run)
    name = identity["yaml"]
    run = launch.gl_formal_run(64, encoder_run) if steps == 80000 else f"o16gl-smoke-{encoder_run}"
    args, argv, env, h = gl_arguments(fx, name, steps=steps, run=run, fsdp=4, batch=64,
                                     config_name=launch.ABLATION_CONFIG)
    args.encoder_run = encoder_run
    from mme_vla_suite.training import config
    monkeypatch.setattr(config, "OPENPI_DATA_HOME", env["OPENPI_DATA_HOME"])
    rec = launch.core(args, argv, contract=fx.contract, env=env, history=h)
    assert not rec.failed, rec.failed


@pytest.mark.parametrize("encoder_run", ABLATION_RUNS)
@pytest.mark.parametrize("steps", [20, 60000])
def test_ablation_core_rejects_legacy_eight_gpu_config(tmp_path, monkeypatch, encoder_run, steps):
    """完整核心拒绝新 encoder 借旧八卡 B128 配置绕过 GL 组合检查。"""
    fx, identity = ablation_fixture(tmp_path, monkeypatch, encoder_run)
    args, argv, env, h = base.arguments(fx, identity["yaml"], steps=steps)
    h["motion"]["store_path"] = fx.contract.parent_rel + "/" + identity["store_dir"]
    args.encoder_run = encoder_run
    from mme_vla_suite.training import config
    monkeypatch.setattr(config, "OPENPI_DATA_HOME", env["OPENPI_DATA_HOME"])
    rec = launch.core(args, argv, contract=fx.contract, env=env, history=h)
    assert rec.failed == ["ENCODER_COMBO"]


@pytest.mark.parametrize("encoder_run", ABLATION_RUNS)
@pytest.mark.parametrize("fault", ["old_yaml", "batch128", "steps60000", "old_encoder", "attempt01", "fsdp2"])
def test_ablation_combo_rejects(encoder_run, fault):
    values = dict(fsdp=4, batch=64, steps=80000, run=launch.gl_formal_run(64, encoder_run),
                  history_name=launch.ENCODERS[encoder_run]["yaml"], encoder_run=encoder_run,
                  config_name=launch.ABLATION_CONFIG)
    if fault == "old_yaml": values["history_name"] = launch.GL_HISTORY
    if fault == "batch128": values["batch"] = 128
    if fault == "steps60000": values["steps"] = 60000
    if fault == "old_encoder": values["encoder_run"] = launch.ENCODER_RUN
    if fault == "attempt01": values["run"] += "-attempt01"
    if fault == "fsdp2": values["fsdp"] = 2
    assert not launch.gl_combo_ok(**values)


def test_ablation_identity_sha_constants():
    assert launch.ENCODERS["aws-stage2-noxyz-1101-a"]["config_sha256"] == "583e48c0a6afa2a798eb14fc8ccdcfe8d9274cb13f1393e05859d6bca368bada"
    assert launch.ENCODERS["aws-stage2-nostate-1101-a"]["config_sha256"] == "869ac77872092a0d0b08e1bc3ac60b0a893946af02ae3de193aa5715e8aecc34"


def test_final_record_accepts_100_steps(tmp_path, capsys):
    path = tmp_path / "final.json"
    path.write_text(json.dumps(base.final_fixture("o16gl-tier-x", steps=100)))
    assert launch.main(["--check-final-record", str(path), "--run", "o16gl-tier-x", "--steps", "100",
                        "--train-head", "a" * 40]) == 0
    assert "TRAIN_FINAL_RECORD=PASS run=o16gl-tier-x tail_steps=99" in capsys.readouterr().out


# ── 血缘证据 SHA 路由：MJ 源清单缺失且库根有 evidence/ 五份 ─────────────────────────

@pytest.fixture
def evidence_route(fx):
    from mme_vla_suite.datastore import motion_store as ms
    folder = fx.parent / "evidence"
    folder.mkdir()
    evidence = fx.motion.provenance["stage2_compat"]["evidence"]
    for role in ms.stage2_evidence_roles(fx.motion.provenance):
        shutil.copyfile(evidence[role]["path"], folder / f"{role}.json")
    contract = dataclasses.replace(fx.contract, mj_source_manifest=str(fx.root / "absent/source_manifest.json"))
    yield folder, contract
    shutil.rmtree(folder)


def test_lineage_evidence_route_passes(fx, evidence_route, capsys):
    folder, contract = evidence_route
    rec = gl_core(fx, contract=contract)
    assert not rec.failed, rec.failed
    out = capsys.readouterr().out
    assert "LINEAGE_ROUTE=evidence" in out
    assert "DATA_LINEAGE=PASS route=evidence" in out and "evidence_files=5 mismatches=0" in out


def test_lineage_evidence_route_sha_mismatch_fails(fx, evidence_route):
    folder, contract = evidence_route
    path = next(folder.iterdir())
    path.write_bytes(path.read_bytes() + b" ")
    assert "ENCODER_DATA_LINEAGE" in gl_core(fx, contract=contract).failed


def test_lineage_without_manifest_or_evidence_fails(fx):
    contract = dataclasses.replace(fx.contract, mj_source_manifest=str(fx.root / "absent/source_manifest.json"))
    assert "DATASET" in gl_core(fx, contract=contract).failed


# ── 实际 shell 入口 DRY_RUN：组合表、GPU 可见范围、证据根与计时放行 ───────────────────

@pytest.fixture
def gl_tree():
    """paths.sh 只认本机/turbo/AWS 前缀，故临时树建在本仓库 v1-store/ 下（gitignore），用后删除。"""
    parent = REPO / "v1-store/tmp-o16gl-tests"
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(dir=parent))
    try:
        scripts = root / "scripts/training"
        scripts.mkdir(parents=True)
        for name in ("run_orig16_stage2_60k_gl.sh", "paths.sh", "check_orig16_stage2_launch.py",
                     "check_mixed8_stage2_1024_launch.py", "final_record.py"):
            shutil.copyfile(REPO / "scripts/training" / name, scripts / name)
        # 临时副本沿用当前工作树前缀，允许在独立工作树运行同一 shell 测试。
        paths = scripts / "paths.sh"
        paths.write_text(paths.read_text().replace('readonly LOCAL_WORK_PREFIX="/data/hongzefu/"',
                                                   f'readonly LOCAL_WORK_PREFIX="{REPO}/"'))
        (root / "pyproject.toml").write_text("")
        evidence = root / "v1-store/validation/orig16-stage2"
        evidence.mkdir(parents=True)
        py = root / ".venv/bin/python"
        py.parent.mkdir(parents=True)
        py.write_text('''#!/usr/bin/env bash
if [[ "$*" == *--print-library* ]]; then echo fixture; exit 0; fi
[[ "$1" == scripts/training/check_orig16_stage2_launch.py && "$JAX_PLATFORMS" == cpu ]] || exit 91
echo "CONTRACT_ARGS=$*"
echo "CONTRACT_ENV timing=${TRAIN_TIMING_STEPS-<unset>} evidence=${MMEVLA_STAGE2_EVIDENCE_ROOT-<unset>} cvd=${CUDA_VISIBLE_DEVICES-<unset>} jaxcache=$MMEVLA_JAX_CACHE_DIR uvcache=$UV_CACHE_DIR attempt=${ATTEMPT_MANIFEST-<unset>}"
exit 0
''')
        py.chmod(0o755)
        yield root
    finally:
        shutil.rmtree(root)


def run_entry(root, *, gpus="4", batch="64", run=None, history=launch.GL_HISTORY, extra=None, drop=()):
    env = {k: v for k, v in os.environ.items() if k not in ("SLURM_JOB_ID", "MAX_STEPS", "TRAIN_TIMING_STEPS",
                                                             launch.EVIDENCE_ENV, "CUDA_VISIBLE_DEVICES", "CONFIG",
                                                             "ENCODER_RUN", "ATTEMPT_MANIFEST")}
    env.update(DRY_RUN="1", GPUS=gpus, GLOBAL_BATCH=batch, HISTORY=history, TRAIN_HEAD="a" * 40,
               RUN_NAME=run or launch.gl_formal_run(batch), CUDA_VISIBLE_DEVICES=",".join(map(str, range(int(gpus)))),
               FINAL_RECORD_DIR=str(root / "v1-store/validation/o16gl/final-record"))
    env[launch.EVIDENCE_ENV] = str(root / "v1-store/validation/orig16-stage2")
    env.update(extra or {})
    for key in drop:
        env.pop(key, None)
    before = {str(p.relative_to(root)) for p in root.rglob("*")}
    result = subprocess.run(["bash", str(root / "scripts/training/run_orig16_stage2_60k_gl.sh")], env=env,
                            capture_output=True, text=True)
    assert {str(p.relative_to(root)) for p in root.rglob("*")} == before
    return result


@pytest.mark.parametrize("gpus", ["2", "4"])
@pytest.mark.parametrize("batch", ["64", "32"])
def test_entry_formal_dry_run(gl_tree, gpus, batch):
    result = run_entry(gl_tree, gpus=gpus, batch=batch)
    assert result.returncode == 0, result.stdout + result.stderr
    n = int(gpus)
    tail = f"--fsdp-devices {n} --num-workers {2 * n} --batch-size {batch}"
    cmd = next(line for line in result.stdout.splitlines() if line.startswith("TRAIN_CMD="))
    assert cmd.rstrip().endswith(tail) and "--num-train-steps" not in cmd
    contract = next(line for line in result.stdout.splitlines() if line.startswith("CONTRACT_ARGS="))
    assert f"--steps 60000 --fsdp-devices {n} --batch-size {batch} --gpus-visible {','.join(map(str, range(n)))}" in contract
    env_line = next(line for line in result.stdout.splitlines() if line.startswith("CONTRACT_ENV"))
    assert "timing=<unset>" in env_line and f"evidence={gl_tree}/v1-store/validation/orig16-stage2" in env_line
    assert f"jaxcache={gl_tree}/v1-store/cache/jax/" in env_line and f"uvcache={os.environ['HOME']}/.cache/uv" in env_line


@pytest.mark.parametrize("run,steps,batch,history,timing", [
    ("o16gl-smoke-a", "20", "64", launch.GL_HISTORY, None),
    ("o16gl-smoke-a", "20", "32", launch.GL_HISTORY, None),
    ("o16gl-smoke-a", "20", "16", launch.GL_HISTORY, "20"),
    ("o16gl-smoke-a", "100", "64", launch.GL_HISTORY, "100"),
    ("o16gl-smoke-a", "20", "8", launch.GL_HISTORY, None),
    ("o16gl-smoke-a", "20", "4", launch.GL_HISTORY, "20"),
    ("o16gl-smoke-a", "100", "32", launch.GL_HISTORY, None),
    *[("o16gl-tier-a", "100", "64", tier, "100") for tier in TIERS]])
def test_entry_prefix_dry_run(gl_tree, run, steps, batch, history, timing):
    extra = {"MAX_STEPS": steps, **({"TRAIN_TIMING_STEPS": timing} if timing else {})}
    result = run_entry(gl_tree, batch=batch, run=run, history=history, extra=extra)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"--num-train-steps {steps}" in result.stdout and f"--steps {steps} " in result.stdout
    env_line = next(line for line in result.stdout.splitlines() if line.startswith("CONTRACT_ENV"))
    assert f"timing={timing or '<unset>'}" in env_line
    assert ("evidence=<unset>" in env_line) == run.startswith("o16gl-tier-")
    if run.startswith("o16gl-tier-"):
        assert "--dataset-path" in result.stdout and "framesamp-4x4" in result.stdout


@pytest.mark.parametrize("encoder_run", ABLATION_RUNS)
@pytest.mark.parametrize("steps", [20, 100, 80000])
def test_entry_ablation_three_commands(gl_tree, encoder_run, steps):
    extra = dict(CONFIG=launch.ABLATION_CONFIG, ENCODER_RUN=encoder_run)
    if steps != 80000:
        extra["MAX_STEPS"] = str(steps)
    run = launch.gl_formal_run(64, encoder_run) if steps == 80000 else "o16gl-smoke-ablation"
    result = run_entry(gl_tree, run=run, history=launch.ENCODERS[encoder_run]["yaml"], extra=extra)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"--config-name {launch.ABLATION_CONFIG} --encoder-run {encoder_run}" in result.stdout
    assert f"--steps {steps}" in result.stdout
    cmd = next(line for line in result.stdout.splitlines() if line.startswith("TRAIN_CMD="))
    assert ("--num-train-steps" in cmd) == (steps != 80000)


@pytest.mark.parametrize("manifest", [None, "/absolute/attempt.json"])
def test_entry_attempt_explicit_environment(gl_tree, manifest):
    extra = {"ATTEMPT_MANIFEST": manifest} if manifest else {}
    result = run_entry(gl_tree, extra=extra)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"attempt={manifest or '<unset>'}" in result.stdout


@pytest.mark.parametrize("case", [
    dict(gpus="8"), dict(gpus="1"), dict(batch="128"), dict(batch="16"),
    dict(history=TIERS[1]), dict(history=launch.pair_name("modul", "4x256")),
    dict(extra={"MAX_STEPS": "20"}), dict(extra={"TRAIN_TIMING_STEPS": "100"}),
    dict(run="o16gl-smoke-a", batch="16", extra={"MAX_STEPS": "100"}),
    dict(run="o16gl-smoke-a", batch="8", extra={"MAX_STEPS": "100"}), dict(batch="8"), dict(batch="4"),
    dict(run="o16gl-smoke-a", batch="32", drop=("MAX_STEPS",)),
    dict(run="o16gl-smoke-a", batch="32", history=TIERS[0], extra={"MAX_STEPS": "20"}),
    dict(run="o16gl-smoke-a", batch="32", extra={"MAX_STEPS": "20", "TRAIN_TIMING_STEPS": "21"}),
    dict(run="o16gl-tier-a", batch="32", history=TIERS[0], extra={"MAX_STEPS": "100"}),
    dict(run="o16gl-tier-a", batch="64", history=TIERS[0], extra={"MAX_STEPS": "20"}),
    dict(run="o16gl-tier-a", batch="64", history=launch.GL_HISTORY, extra={"MAX_STEPS": "100"}),
    dict(extra={"CUDA_VISIBLE_DEVICES": "0,1"}), dict(drop=("CUDA_VISIBLE_DEVICES",)),
    dict(drop=(launch.EVIDENCE_ENV,)), dict(extra={launch.EVIDENCE_ENV: "relative/dir"})])
def test_entry_rejects_before_contract(gl_tree, case):
    result = run_entry(gl_tree, **case)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "CONTRACT_ARGS=" not in result.stdout


@pytest.mark.parametrize("seen,code", [(4, 0), (2, 2)])
def test_entry_slurm_counts_nvidia_smi(gl_tree, tmp_path, seen, code):
    fake = tmp_path / "bin/nvidia-smi"
    fake.parent.mkdir()
    fake.write_text("#!/usr/bin/env bash\n" + "".join(f"echo 'GPU {i}: NVIDIA A40 (UUID: x)'\n" for i in range(seen)))
    fake.chmod(0o755)
    result = run_entry(gl_tree, drop=("CUDA_VISIBLE_DEVICES",),
                       extra={"SLURM_JOB_ID": "123", "PATH": f"{fake.parent}:{os.environ['PATH']}"})
    assert result.returncode == code, result.stdout + result.stderr
    if code == 0:
        assert "--gpus-visible  --" in result.stdout or "--gpus-visible --" in result.stdout
        assert "cvd=<unset>" in result.stdout


# ── 加载器形制白名单：只给测时 (256,16,1)/(1024,16,1) + context + 无 motion 放行 ────────

def frame_dataset(history, tmp_path):
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    return FrameSampDataset(str(tmp_path / "absent-store"), None, history, 20)


@pytest.mark.parametrize("tier", TIERS)
def test_loader_accepts_tier_shapes(tmp_path, tier):
    from omegaconf import OmegaConf
    history = OmegaConf.create(launch.load_yaml(REPO, tier))
    with pytest.raises(Exception) as error:
        frame_dataset(history, tmp_path)  # 形制通过后才去读不存在的库
    assert "形制断言失败" not in str(error.value)


@pytest.mark.parametrize("change", ["modulation", "motion", "budget768"])
def test_loader_rejects_tier_shape_outside_context_nomotion(tmp_path, change):
    from omegaconf import OmegaConf
    value = launch.load_yaml(REPO, TIERS[0])
    if change == "modulation":
        value.update(integration_type="modulation", memory_token_dim=1024)
    elif change == "motion":
        value["motion"] = launch.load_yaml(REPO, launch.GL_HISTORY)["motion"]
    else:
        value["budget"] = 768
    with pytest.raises(ValueError, match="形制断言失败"):
        frame_dataset(OmegaConf.create(value), tmp_path)
