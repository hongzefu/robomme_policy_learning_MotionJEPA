"""1114 4task RGB+state VLA 训练接线（S3）：TrainConfig、两份 yaml、单条入口 dry-run、交付检查 50000 与训练诊断开关。"""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
YAML_DIR = REPO / "src/mme_vla_suite/models/config/robomme"
MOD_H = "perceptual-framesamp-modul-8frame-8x8-motion-stage2-1x1024-4task-rgbstate.yaml"
CTX_H = "perceptual-framesamp-ctx-8frame-8x8-motion-stage2-1x1024-4task-rgbstate.yaml"
CONFIG_NAME = "mme_vla_suite_4task_b128_50k"
CKPT_STEPS = tuple(range(5000, 50001, 5000))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


restore = load_module("s3_check_stage2_checkpoint", REPO / "scripts/training/check_stage2_checkpoint.py")


# ── TrainConfig（B 表 T07 继承表）────────────────────────────────────────────────
def test_4task_config_inherits_80k_and_changes_only_name_and_steps():
    from mme_vla_suite.training import config
    new, base = config.get_config(CONFIG_NAME), config.get_config("mme_vla_suite_b128_80k")
    assert dataclasses.replace(new, name=base.name, num_train_steps=base.num_train_steps) == base
    assert new.num_train_steps == 50_001
    lr = new.lr_schedule
    assert (type(lr).__name__, lr.warmup_steps, lr.peak_lr, lr.decay_steps, lr.decay_lr) == (
        "CosineDecaySchedule", 5_000, 5e-5, 80_000, 5e-5)
    assert (new.batch_size, new.fsdp_devices, new.num_workers, new.ema_decay) == (128, 8, 16, 0.999)
    assert (new.save_interval, new.keep_period, new.project_name) == (5_000, 5_000, "robomme-framesamp")
    assert new.data.assets.assets_dir == "v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da"
    assert (new.data.assets.asset_id, new.data.base_config.prompt_from_task) == ("robomme", True)
    assert (new.model.pi05, new.model.action_horizon, new.model.use_history, new.model.discrete_state_input) == (
        True, 20, True, False)
    assert new.optimizer.clip_gradient_norm == 1.0
    assert new.weight_loader.params_path.endswith("openpi-assets/checkpoints/pi05_base/params")


def test_checkpoint_steps_ten_dirs_and_existing_configs():
    from mme_vla_suite.training import config
    assert config.checkpoint_steps(config.get_config(CONFIG_NAME)) == CKPT_STEPS
    assert config.checkpoint_steps(config.get_config("mme_vla_suite_orig16_b128_60k")) == \
        tuple(range(5000, 60000, 5000)) + (59999,)
    short = dataclasses.replace(config.get_config(CONFIG_NAME), num_train_steps=20)
    assert config.checkpoint_steps(short) == (19,)


# ── 两份 yaml（CONCAT_CONFIG）──────────────────────────────────────────────────
def load_yaml(name):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(YAML_DIR / name))


def test_concat_config_pair_differs_only_in_fusion():
    mod, ctx = load_yaml(MOD_H), load_yaml(CTX_H)
    assert set(mod) == set(ctx)
    assert {key for key in mod if mod[key] != ctx[key]} == {"integration_type", "memory_token_dim"}
    assert (mod["integration_type"], mod["memory_token_dim"]) == ("modulation", 1024)
    assert (ctx["integration_type"], ctx["memory_token_dim"]) == ("context", 2048)
    print("CONCAT_CONFIG=PASS diff_keys=integration_type,memory_token_dim")


@pytest.mark.parametrize("name", [MOD_H, CTX_H])
def test_4task_yaml_layout_and_motion(name):
    from mme_vla_suite.datastore.motion_adapter import motion_policy_spec
    from mme_vla_suite.models.config.utils import get_history_config
    h = load_yaml(name)
    assert (h["budget"], h["token_per_image"], h["num_views"], h["streaming_obs_horizon"]) == (512, 64, 1, 16)
    assert h["perceptual_memory"]["type"] == "frame_sampling"
    m = h["motion"]
    spec = motion_policy_spec(m)
    assert (spec.readout, spec.num_tokens, spec.dim, spec.budget, spec.window_budget) == (
        "stage2_pool4_concat_v1", 1, 1024, 96, 96)
    assert (m["stride"], m["window_frames"], m["encoder_kind"], m["online_gpu"]) == (32, 33, "stage2_pool4_concat", 1)
    assert m["store_path"] == "v1-store/datasets/4task-v2-1600ep-604f16da/motion-stage2-1024-rgbstate-ep144"
    assert m["source_run"] == "stage2-4task-rgbstate-144ep-1113-a/checkpoint_epoch_144.pt#model"
    # ckpt sha 由 encoder 训满后的单独修订 commit 回填（计划文首）；之前任何正式合同必须失败。
    assert m["checkpoint_sha256"] == "__FILL_AFTER_BUILD__"
    # 与 16task 同布局 8×8 concat yaml 只差库路径、encoder 来源与 sha 三项（注入代码同一份）。
    orig = load_yaml(name.replace("-4task-rgbstate.yaml", "-orig16.yaml"))
    assert {k for k in h if h[k] != orig[k]} == {"motion"}
    assert {k for k in m if m[k] != orig["motion"][k]} == {"store_path", "source_run", "checkpoint_sha256"}
    assert get_history_config(name).motion.readout == "stage2_pool4_concat_v1"


# ── 单条入口 dry-run（真实 shell；合同脚本由夹具替身代答）─────────────────────────
def entry_tree(tmp_path, contract_exit=0):
    root = tmp_path / "repo"
    (root / "scripts/training").mkdir(parents=True)
    shutil.copyfile(REPO / "scripts/training/run_orig16_stage2_60k.sh", root / "scripts/training/run_orig16_stage2_60k.sh")
    (root / "scripts/training/check_4task_stage2_launch.py").write_text("# S1 合同脚本替身（只为存在性检查）\n")
    py = root / ".venv/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text(f'''#!/usr/bin/env bash
if [[ "$1" == scripts/training/check_4task_stage2_launch.py ]]; then
  shift
  case "$*" in
    --print-library) echo 4task-v2-1600ep-604f16da; exit 0 ;;
    --print-config) echo {CONFIG_NAME}; exit 0 ;;
  esac
  [[ "$JAX_PLATFORMS" == cpu ]] || exit 92
  echo "CONTRACT_ARGS=$*"
  echo "LAUNCH_CONTRACT=STUB exit={contract_exit}"
  exit {contract_exit}
fi
exec {sys.executable} "$@"
''')
    py.chmod(0o755)
    return root


def run_entry(root, **extra):
    env = {k: v for k, v in os.environ.items() if k not in ("MAX_STEPS", "NUM_TRAIN_STEPS", "ATTEMPT_MANIFEST")}
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", DRY_RUN="1", TRAIN_HEAD="a" * 40,
               FINAL_RECORD_DIR=str(root / "v1-store/validation/x/final-record"))
    env.update(extra)
    return subprocess.run(["bash", str(root / "scripts/training/run_orig16_stage2_60k.sh")], env=env,
                          capture_output=True, text=True)


@pytest.mark.parametrize("name,run", [(MOD_H, "v16-smoke-modul-s2rgbstate-concat1x1024"),
                                      (CTX_H, "v16-smoke-ctx-s2rgbstate-concat1x1024")])
def test_smoke_dry_run_prints_cmd_and_calls_4task_contract(tmp_path, name, run):
    root = entry_tree(tmp_path)
    before = {str(p.relative_to(root)) for p in root.rglob("*")}
    result = run_entry(root, HISTORY=name, RUN_NAME=run, MAX_STEPS="20")
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    cmd = next(line for line in out.splitlines() if line.startswith("TRAIN_CMD="))
    assert f" {CONFIG_NAME} " in cmd and "--num-train-steps 20" in cmd and name in cmd
    assert f"{root}/v1-store/datasets/4task-v2-1600ep-604f16da/framesamp-8x8" in cmd
    assert "/v1-store/train-assets/mme_vla_suite/4task-v2-1600ep-604f16da" in cmd
    contract = next(line for line in out.splitlines() if line.startswith("CONTRACT_ARGS="))
    assert "--steps 20 " in contract and "TRAIN_PARAM" not in out
    assert {str(p.relative_to(root)) for p in root.rglob("*")} == before


@pytest.mark.parametrize("name,run", [(MOD_H, "v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k"),
                                      (CTX_H, "v2-1600ep-m8x8-ctx-motion-s2rgbstate-concat1x1024-b128-50k")])
def test_formal_dry_run_prints_full_inherited_table_matching_config(tmp_path, name, run):
    from mme_vla_suite.training import config
    root = entry_tree(tmp_path)
    result = run_entry(root, HISTORY=name, RUN_NAME=run, NUM_TRAIN_STEPS="50001")
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout.splitlines()
    cmd = next(line for line in out if line.startswith("TRAIN_CMD="))
    assert "--num-train-steps" not in cmd
    params = dict(line.removeprefix("TRAIN_PARAM ").split("=", 1) for line in out if line.startswith("TRAIN_PARAM "))
    cfg = config.get_config(CONFIG_NAME)
    lr = cfg.lr_schedule
    expected = dict(config=CONFIG_NAME, exp_name=run, num_train_steps=50001, batch_size=cfg.batch_size,
                    fsdp_devices=cfg.fsdp_devices, num_workers=cfg.num_workers, seed=cfg.seed, ema_decay=cfg.ema_decay,
                    lr_schedule="CosineDecaySchedule", warmup_steps=lr.warmup_steps, peak_lr=lr.peak_lr,
                    decay_steps=lr.decay_steps, end_lr=lr.decay_lr, save_interval=cfg.save_interval,
                    keep_period=cfg.keep_period, log_interval=cfg.log_interval, action_horizon=20, pi05=True,
                    use_history=True, history_config=name, clip_gradient_norm=1.0)
    assert params == {key: str(value) for key, value in expected.items()}
    assert (params["num_workers"], params["ema_decay"], params["warmup_steps"], params["peak_lr"]) == (
        "16", "0.999", "5000", "5e-05")
    assert f"CKPT_STEPS={','.join(map(str, CKPT_STEPS))} count=10" in out
    contract = next(line for line in out if line.startswith("CONTRACT_ARGS="))
    assert "--steps 50001 " in contract


@pytest.mark.parametrize("extra", [
    dict(HISTORY=MOD_H, RUN_NAME="o16-smoke-x", MAX_STEPS="20"),                                  # 前缀不对
    dict(HISTORY=MOD_H, RUN_NAME="v16-smoke-ctx-s2rgbstate-concat1x1024", MAX_STEPS="20"),       # 融合不符
    dict(HISTORY=MOD_H, RUN_NAME="v16-smoke-modul-s2rgbstate-concat1x1024", MAX_STEPS="50"),     # 步数非 20
    dict(HISTORY=MOD_H, RUN_NAME="formal", NUM_TRAIN_STEPS="60000"),                              # 4task 正式步数
    dict(HISTORY=MOD_H, RUN_NAME="formal", NUM_TRAIN_STEPS="50000"),
    dict(HISTORY="perceptual-framesamp-modul-8frame-8x8-motion-stage2-1x1024-mixed8.yaml", RUN_NAME="x"),
])
def test_entry_rejects_bad_names_and_steps(tmp_path, extra):
    root = entry_tree(tmp_path)
    result = run_entry(root, **extra)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "TRAIN_CMD=" not in result.stdout


def test_entry_contract_failure_propagates(tmp_path):
    root = entry_tree(tmp_path, contract_exit=7)
    result = run_entry(root, HISTORY=CTX_H, RUN_NAME="v16-smoke-ctx-s2rgbstate-concat1x1024", MAX_STEPS="20")
    assert result.returncode == 7


# ── 交付检查 50000（check_stage2_checkpoint.py）──────────────────────────────────
def test_label_maps_4task_yaml_to_new_config():
    assert restore.config_name_for_label(MOD_H) == CONFIG_NAME
    assert restore.config_name_for_label(CTX_H) == CONFIG_NAME
    assert restore.config_name_for_label("x-orig16.yaml") == "mme_vla_suite_orig16_b128_60k"
    assert restore.config_name_for_label("perceptual-framesamp-modul-8frame-8x8-motion-stage2-mixed8.yaml") == \
        "mme_vla_suite_b128_80k"
    assert [restore.delivery_label(s) for s in (50000, 59999, 60000, 70000)] == [
        "CKPT_DELIVERY_READY", "CKPT_DELIVERY_READY", "CKPT_60K_READY", "CKPT_70K_READY"]


def stage_fixture(tmp_path, step=50000):
    """真实形状的 stage：根文件、step 目录（params/assets）、encoder 依赖实体与 delivery 清单。"""
    root = tmp_path / "exports/hf-ckpt-run/stage"
    root.mkdir(parents=True)
    files = {}
    encoder_files = {"encoder_dependencies/stage2-4task-rgbstate-144ep-1113-a/checkpoint_epoch_144.pt": b"ckpt",
                     "encoder_dependencies/stage2-4task-rgbstate-144ep-1113-a/config.yaml": b"cfg"}
    names = restore.ROOT_FILES - {"encoder_dependencies.json"} | {
        "motion_snapshot.ready.json", f"{step}/params/_METADATA", f"{step}/params/d/0",
        f"{step}/assets/robomme/norm_stats.json"}
    for name in sorted(names):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    for name, blob in encoder_files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    sha = {name: restore.sha256_file(root / name) for name in encoder_files}
    roles = dict(zip(("encoder_checkpoint", "encoder_config"), encoder_files))
    provenance = {"encoder": {"checkpoint_sha256": sha[roles["encoder_checkpoint"]],
                              "config_sha256": sha[roles["encoder_config"]]},
                  "source_pin": {"member_pin_sha256": "2c" * 32}, "vae": {"kind": "wan2.1"}}
    deps = {"schema": "stage2-encoder-dependencies-v1", "encoder": provenance["encoder"],
            "source_pin": provenance["source_pin"], "vae": provenance["vae"],
            "files": [{"role": role, "relative_path": name, "sha256": sha[name], "bytes": len(encoder_files[name])}
                      for role, name in roles.items()]}
    (root / "encoder_dependencies.json").write_text(json.dumps(deps))
    for path in root.rglob("*"):
        if path.is_file():
            name = path.relative_to(root).as_posix()
            files[name] = {"sha256": restore.sha256_file(path), "bytes": path.stat().st_size}
    manifest = root / "manifests" / f"step-{step}.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"schema": "stage2-delivery-manifest-v1", "files": files}))
    return SimpleNamespace(root=root, manifest=manifest, provenance=provenance, roles=roles)


def run_delivery(fx, tmp_path, monkeypatch, step=50000):
    """真实走 CLI、设备合同、交付清单与 encoder 依赖实体核对；只把根快照证据链（S2 范围）与 GPU 恢复段替换。"""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(restore, "verify_snapshot", lambda root, delivery: (
        restore.verify_encoder_dependencies(root, fx.provenance, delivery), fx.provenance)[1])
    monkeypatch.setattr(restore, "restore_runtime",
                        lambda args, report, root, checkpoint, step, provenance: restore.pass_line(args.mode, step))
    out = tmp_path / "report" / "delivery.json"
    code = restore.main(["--mode", "delivery", "--delivery-step", str(step), "--checkpoint", str(fx.root / str(step)),
                         "--expected-manifest", str(fx.manifest), "--online-gpu", "1", "--out", str(out)])
    return code, restore.read_json(out)


def test_delivery_50000_pass_on_synthetic_stage(tmp_path, monkeypatch, capsys):
    fx = stage_fixture(tmp_path)
    code, report = run_delivery(fx, tmp_path, monkeypatch)
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == 0 and report["status"] == "PASS"
    assert line.startswith("CKPT_DELIVERY_READY=PASS step=50000 ")
    assert report["delivery"]["sha256_mismatch"] == 0 and report["step"] == 50000


@pytest.mark.parametrize("fault", ["missing_encoder_ckpt", "encoder_sha", "manifest_drops_encoder", "wrong_dir"])
def test_delivery_50000_rejects(tmp_path, monkeypatch, capsys, fault):
    fx = stage_fixture(tmp_path)
    ckpt_rel = fx.roles["encoder_checkpoint"]
    if fault == "missing_encoder_ckpt":
        (fx.root / ckpt_rel).unlink()
    elif fault == "encoder_sha":
        (fx.root / ckpt_rel).write_bytes(b"ckpX")                       # 同长度改字节：sha 不符
    elif fault == "manifest_drops_encoder":
        manifest = json.loads(fx.manifest.read_text()); del manifest["files"][ckpt_rel]
        fx.manifest.write_text(json.dumps(manifest))
    else:
        shutil.move(fx.root / "50000", fx.root / "49999")
        (fx.root / "50000").mkdir()
    code, report = run_delivery(fx, tmp_path, monkeypatch)
    assert code == 1 and report["status"] == "FAIL"
    assert capsys.readouterr().out.strip().splitlines()[-1] == "CKPT_DELIVERY_READY=FAIL"


def test_delivery_requires_expected_manifest_and_known_step(tmp_path):
    base = ["--mode", "delivery", "--checkpoint", str(tmp_path / "s/50000"), "--online-gpu", "1",
            "--out", str(tmp_path / "o.json")]
    with pytest.raises(SystemExit):
        restore.parse_args(base + ["--delivery-step", "50000"])
    with pytest.raises(SystemExit):
        restore.parse_args(base + ["--delivery-step", "50001", "--expected-manifest", "m"])
    assert restore.parse_args(base + ["--delivery-step", "50000", "--expected-manifest", "m"]).delivery_step == 50000
    assert restore.parse_args(base + ["--delivery-step", "59999", "--expected-manifest", "m"]).delivery_step == 59999


def test_delivery_59999_still_passes(tmp_path, monkeypatch, capsys):
    fx = stage_fixture(tmp_path, step=59999)
    code, _ = run_delivery(fx, tmp_path, monkeypatch, step=59999)
    assert code == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("CKPT_DELIVERY_READY=PASS step=59999 ")


# ── train.py 诊断：关闭时与基线 4bea090 逐位一致 ────────────────────────────────
BASE_SHA = "4bea090f72632beba58c5055f4e190274e8e01ad"


def load_train(tmp_path, source=None):
    sys.path.insert(0, str(REPO / "scripts/training"))
    if source is None:
        return load_module("s3_train_new", REPO / "scripts/training/train.py")
    blob = subprocess.run(["git", "-C", str(REPO), "show", f"{source}:scripts/training/train.py"],
                          check=True, capture_output=True, text=True).stdout
    path = tmp_path / "train_base.py"
    path.write_text(blob)
    return load_module("s3_train_base", path)


def fake_setup(motion):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from omegaconf import OmegaConf
    import openpi.training.utils as training_utils

    class Mem(nnx.Module):
        def __init__(self, rngs):
            self.motion_enabled = True
            self.feature_encoder = nnx.Linear(4, 3, rngs=rngs)
            self.motion_pos_proj = nnx.Linear(2, 3, rngs=rngs)
            self.motion_encoder_static = nnx.Linear(5 + 3, 3, rngs=rngs)

    class PG(nnx.Module):
        def __init__(self, rngs):
            self.llm = nnx.Linear(3, 3, rngs=rngs)

    class Fake(nnx.Module):
        def __init__(self, rngs):
            self.PaliGemma = PG(rngs)
            self.mem_encoder = Mem(rngs)

        def compute_loss(self, rng, obs, actions, train=False):
            mem = self.mem_encoder
            pos_h = nnx.silu(mem.motion_pos_proj(obs.motion_pos))
            tokens = mem.motion_encoder_static(jnp.concatenate([obs.motion_emb, pos_h], axis=-1))
            hidden = self.PaliGemma.llm(tokens.mean(axis=1) + mem.feature_encoder(obs.x))
            return jnp.mean(jnp.square(hidden - actions), axis=-1)

    model = Fake(nnx.Rngs(0))
    params = nnx.state(model)
    tx = optax.adamw(1e-2)
    state = training_utils.TrainState(step=jnp.asarray(0), params=params, model_def=nnx.graphdef(model),
                                      opt_state=tx.init(params.filter(nnx.Param)), tx=tx, ema_decay=0.9,
                                      ema_params=params)
    key = jax.random.split(jax.random.key(3), 4)
    obs = SimpleNamespace(motion_emb=jax.random.normal(key[0], (2, 6, 5)),
                          motion_pos=jax.random.normal(key[1], (2, 6, 2)),
                          motion_mask=jnp.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], bool),
                          x=jax.random.normal(key[2], (2, 4)))
    actions = jax.random.normal(key[3], (2, 3))
    cfg = SimpleNamespace(trainable_filter=nnx.Param,
                          model=SimpleNamespace(use_history=True, history_config=OmegaConf.create({"motion": motion})))
    return cfg, state, (obs, actions)


def step(module, cfg, state, batch):
    import jax
    from openpi.shared import array_typing as at
    with at.disable_typechecking():
        new_state, info = module.train_step(cfg, jax.random.key(0), state, batch)
    return new_state, {k: np.asarray(v) for k, v in info.items()}


CONCAT = {"enabled": True, "readout": "stage2_pool4_concat_v1", "encoder_kind": "stage2_pool4_concat"}
SPLIT = {"enabled": True, "readout": "stage2_pool4_split_v1", "encoder_kind": "stage2_pool4_concat"}
DIAG_KEYS = {"motion_enc_kernel_norm", "motion_enc_grad_norm", "frame_enc_grad_norm", "motion_in_var", "motion_out_var"}


def test_diag_switch_truth_table(tmp_path):
    from omegaconf import OmegaConf
    train = load_train(tmp_path)

    def cfg(motion, hc=True):
        return SimpleNamespace(model=SimpleNamespace(use_history=True,
                                                     history_config=OmegaConf.create({"motion": motion}) if hc else "x.yaml"))
    assert train._motion_diag_enabled(cfg(CONCAT))
    assert train._motion_diag_enabled(cfg({"enabled": True, "encoder_kind": "wan_raw9"}))
    assert not train._motion_diag_enabled(cfg(SPLIT))
    assert not train._motion_diag_enabled(cfg({**CONCAT, "enabled": False}))
    assert not train._motion_diag_enabled(cfg(None))
    assert not train._motion_diag_enabled(cfg(CONCAT, hc=False))


def test_diag_off_is_bitwise_identical_to_base_and_on_adds_keys_only(tmp_path):
    import jax
    new, base = load_train(tmp_path), load_train(tmp_path, BASE_SHA)
    assert not hasattr(base, "_motion_diag_enabled")
    leaves = lambda s: [np.asarray(x) for x in jax.tree.leaves((s.params, s.ema_params, s.opt_state))]
    cfg_off, state, batch = fake_setup(SPLIT)
    ref_state, ref_info = step(base, cfg_off, state, batch)
    off_state, off_info = step(new, cfg_off, state, batch)
    assert set(off_info) == set(ref_info) == {"loss", "grad_norm", "param_norm", "llm_grad_norm", "mem_enc_norm"}
    assert all(off_info[k].tobytes() == ref_info[k].tobytes() for k in ref_info)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(leaves(off_state), leaves(ref_state)))
    cfg_on, state, batch = fake_setup(CONCAT)
    on_state, on_info = step(new, cfg_on, state, batch)
    assert set(on_info) == set(ref_info) | DIAG_KEYS
    assert all(on_info[k].tobytes() == ref_info[k].tobytes() for k in ref_info)        # 原五标量不受诊断影响
    assert all(a.tobytes() == b.tobytes() for a, b in zip(leaves(on_state), leaves(ref_state)))
    assert all(np.isfinite(on_info[k]) for k in DIAG_KEYS)
    # motion_in_var 只统计 mask 为真的窗：手算对照。
    obs = batch[0]
    valid = np.asarray(obs.motion_emb)[np.asarray(obs.motion_mask)]
    np.testing.assert_allclose(on_info["motion_in_var"], valid.var(), rtol=1e-5)


def test_final_record_filter_keeps_five_scalars():
    """train.py 交给 final_record.finish 的 info 只含五标量；诊断关闭时与原对象键值相同。"""
    source = (REPO / "scripts/training/train.py").read_text()
    assert "if key in final_record.SCALARS}" in source
    sys.path.insert(0, str(REPO / "scripts/training"))
    import final_record
    info = {k: 1.0 for k in final_record.SCALARS} | {k: 2.0 for k in DIAG_KEYS}
    filtered = {key: value for key, value in info.items() if key in final_record.SCALARS}
    assert set(filtered) == final_record.SCALARS
