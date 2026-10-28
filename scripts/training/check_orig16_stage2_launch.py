#!/usr/bin/env python3
"""官方16×100的只读启动合同；正式CLI没有夹具或跳过检查入口。"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
CONFIG_NAME = "mme_vla_suite_orig16_b128_60k"
PLACEHOLDER = "__FILL_AFTER_BUILD__"
YAML_DIR = "src/mme_vla_suite/models/config/robomme"
ENCODER_SHA = "27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035"
RUNS = {"context": "orig100-ctx32x4x4-motion-s2split4x256-b128-60k",
        "modulation": "orig100-m32x4x4-modul-motion-s2split4x256-b128-60k"}
HISTORIES = tuple(f"perceptual-framesamp-{fusion}-{frames}-motion-stage2-{readout}-orig16.yaml"
    for fusion in ("ctx", "modul") for frames in ("32frame-4x4", "8frame-8x8") for readout in ("1x1024", "4x256"))


def load_script(relative):
    spec = importlib.util.spec_from_file_location(f"_o16_{Path(relative).stem}", REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_script("scripts/training/check_mixed8_stage2_1024_launch.py")
Recorder, sha256_file = legacy.Recorder, legacy.sha256_file


@dataclasses.dataclass(frozen=True)
class Contract:
    """S7统一维护构建pin；阶段3回填前任何正式启动必须失败。"""
    library: str = "16task-orig100-1600ep-__FILL_AFTER_BUILD__"
    norm_sha256: str = PLACEHOLDER
    episode_manifest_sha256: str = PLACEHOLDER
    input_manifest_sha256: str = PLACEHOLDER
    canonical_manifest_sha256: str = PLACEHOLDER
    motion_rows: int | str = PLACEHOLDER
    train_max: int | str = PLACEHOLDER
    encoder_sha256: str = ENCODER_SHA
    encoder_run: str = "stage2-full144-1025-a"
    mj_source_manifest: str = "/scratch/hongze/dataset-original-robomme-16x100-wan21-v1/dataset/source_manifest.json"
    episodes: int = 1600
    tasks: int = 16
    batch_size: int = 128
    fsdp_devices: int = 8
    num_workers: int = 16
    seed: int = 42
    ema_decay: float = 0.999
    gpus: str = "0,1,2,3,4,5,6,7"
    pi05_rel: str = "v1-store/models/openpi-assets/checkpoints/pi05_base/params"

    @property
    def parent_rel(self):
        return f"v1-store/datasets/{self.library}"

    @property
    def assets_rel(self):
        return f"v1-store/train-assets/mme_vla_suite/{self.library}"


def load_yaml(repo, name):
    if name not in HISTORIES:
        raise ValueError(f"history不在八组合内: {name}")
    return legacy.load_yaml(repo, name)


def check_history(rec, history, contract=Contract()):
    from mme_vla_suite.datastore.motion_adapter import motion_policy_spec
    tokens = history.get("token_per_image")
    rec("FRAME_LAYOUT", tokens in (16, 64) and history.get("budget") == 512
        and history.get("num_views") == 1 and history.get("streaming_obs_horizon") == 16)
    dim = {"context": 2048, "modulation": 1024}.get(history.get("integration_type"))
    rec("FUSION", dim is not None and history.get("memory_token_dim") == dim)
    motion = history.get("motion", {})
    spec = motion_policy_spec(motion)
    rec("MOTION_READOUT", (spec.readout, spec.num_tokens, spec.dim, spec.budget, spec.window_budget) in (
        ("stage2_pool4_concat_v1", 1, 1024, 96, 96), ("stage2_pool4_split_v1", 4, 256, 384, 96)))
    expected = legacy.load_yaml(REPO, legacy.HISTORY_CONFIG)
    expected.update(token_per_image=tokens, integration_type=history.get("integration_type"), memory_token_dim=dim)
    expected["motion"].update(readout=spec.readout, num_tokens=spec.num_tokens, dim=spec.dim, budget=spec.budget,
        store_path=contract.parent_rel + "/motion-stage2-1024", online_gpu=1,
        source_run=f"{contract.encoder_run}/checkpoint_epoch_105.pt#model", checkpoint_sha256=contract.encoder_sha256)
    rec("HISTORY_KEYS", history == expected, "除三机制与已固定来源以外不接受配置漂移")
    if not rec.failed:
        print(f"FRAME_LAYOUT=PASS layout={'4x4' if tokens == 16 else '8x8'} frames={512 // tokens} tokens_per_frame={tokens} budget=512")
        print(f"MOTION_LAYOUT=PASS readout={'split' if spec.num_tokens == 4 else 'concat'} tokens_per_window={spec.num_tokens} budget={spec.budget} total_tokens={512 + spec.budget}")
        print(f"FUSION=PASS integration={history['integration_type']} memory_token_dim={dim}")
    return spec


def parse_cli(argv):
    from mme_vla_suite.training import config
    previous = sys.argv
    try:
        sys.argv = ["train.py", *argv]
        return config.cli(), config.get_config(CONFIG_NAME)
    finally:
        sys.argv = previous


def check_argv(rec, argv, steps):
    rec("ARGV_CONFIG_NAME", bool(argv) and argv[0] == CONFIG_NAME)
    allowed = legacy.VALUE_FLAGS | {"--model.history-config"}
    unknown, seen, i = [], set(), 1
    while i < len(argv):
        token = argv[i]
        key = token.split("=", 1)[0]
        if key in seen:
            unknown.append(f"重复:{key}")
        seen.add(key)
        if key in allowed:
            if "=" not in token and (i + 1 == len(argv) or argv[i + 1].startswith("--")):
                unknown.append(f"缺值:{key}")
            i += 1 if "=" in token else 2
        elif token in legacy.BOOL_FLAGS:
            i += 1
        else:
            unknown.append(token)
            i += 1
    rec("ARGV_ALLOWLIST", not unknown, str(unknown))
    values = legacy.flag_value(argv, "--num-train-steps")
    rec("ARGV_STEPS", (steps == 60000 and not values) or (steps == 20 and values == ["20"]))
    rec("ARGV_HISTORY", argv.count("--model.use-history") == 1)


def check_config(rec, cfg, base, *, store, contract, steps, history_name, history):
    from mme_vla_suite.training import config
    for key in ("batch_size", "fsdp_devices", "num_workers", "seed", "ema_decay"):
        rec(f"CFG_{key.upper()}", getattr(cfg, key) == getattr(base, key) == getattr(contract, key))
    rec("CFG_STEPS", cfg.num_train_steps == steps and base.num_train_steps == 60000 and steps in (20, 60000))
    rec("CFG_SCHEDULE", cfg.lr_schedule == base.lr_schedule and (cfg.lr_schedule.warmup_steps,
        cfg.lr_schedule.peak_lr, cfg.lr_schedule.decay_steps, cfg.lr_schedule.decay_lr) == (5000, 5e-5, 60000, 5e-5))
    rec("CFG_MODEL", cfg.model == dataclasses.replace(base.model, history_config=history_name) and cfg.model.use_history)
    layout = "framesamp-4x4" if history["token_per_image"] == 16 else "framesamp-8x8"
    expected = dataclasses.replace(base, exp_name=cfg.exp_name, num_train_steps=steps,
        model=dataclasses.replace(base.model, history_config=history_name),
        dataset_path=str(store / contract.parent_rel / layout), assets_base_dir=str(store / "v1-store/train-assets"),
        checkpoint_base_dir=str(store / "v1-store/train-runs"),
        data=dataclasses.replace(base.data, assets=dataclasses.replace(base.data.assets,
            assets_dir=str(store / contract.assets_rel), asset_id="robomme")),
        weight_loader=dataclasses.replace(base.weight_loader, params_path=str(store / contract.pi05_rel)),
        wandb_enabled=cfg.wandb_enabled)
    rec("CFG_EXACT", cfg == expected, "完整配置除已列CLI路径/名称/短测步数外须等于新60k配置")
    rec("CFG_IMPORT_ROOT", config.OPENPI_DATA_HOME == str(store / "v1-store/models"))
    formal = history["token_per_image"] == 16 and history["motion"]["readout"] == "stage2_pool4_split_v1"
    rec("CFG_RUN", (formal and cfg.exp_name == RUNS[history["integration_type"]]) if steps == 60000
        else cfg.exp_name.startswith("o16-smoke-"))
    rec("CFG_NEW_RUN", not cfg.checkpoint_dir.exists() and not cfg.checkpoint_dir.is_symlink())


def check_dataset(rec, store, contract, history):
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.datastore.manifest import load_manifest
    from mme_vla_suite.policies.motion_client import validate_stage2_evidence
    parent = store / contract.parent_rel
    for name, want in (("episode_manifest.json", contract.episode_manifest_sha256),
                       ("input_manifest.json", contract.input_manifest_sha256)):
        rec(f"{name.upper()}_SHA", sha256_file(parent / "meta" / name) == want)
    rec("NORM_SHA", sha256_file(store / contract.assets_rel / "robomme/norm_stats.json") == contract.norm_sha256)
    manifest = load_manifest(parent / "meta/episode_manifest.json")
    rec("MANIFEST_IDENTITY", manifest["sha256"] == contract.canonical_manifest_sha256
        and len(manifest["episodes"]) == contract.episodes and len(manifest["canonical_order"]) == contract.tasks)
    layout = "framesamp-4x4-v1" if history["token_per_image"] == 16 else "framesamp-8x8-v1"
    frame_path = parent / layout.removesuffix("-v1")
    meta = json.loads((frame_path / "meta/store_meta.json").read_text())
    rec("DATASET_LAYOUT", meta.get("layout") == layout and meta.get("status") == "verified"
        and meta.get("manifest_scope") == "full" and meta.get("manifest_sha256") == manifest["sha256"])
    bounds = legacy.manifest_bounds(manifest)
    rec("WINDOW_BUDGET", bounds["rows"] == contract.motion_rows and bounds["train_max"] == contract.train_max
        and type(contract.train_max) is int and contract.train_max <= 96)
    motion = ms.MotionMeta.load(parent / "motion-stage2-1024")
    ms.run_fast_checks(motion, manifest_path=parent / "meta/episode_manifest.json")
    ms.run_full_checks(motion)
    ms.check_same_source(manifest["sha256"], motion, manifest)
    rec("MOTION_VERIFIED", motion.status == "verified" and motion.raw["layout"] == ms.STAGE2_LAYOUT
        and motion.num_rows == contract.motion_rows and motion.row_shape == (1, 1024)
        and motion.provenance["encoder"]["checkpoint_sha256"] == contract.encoder_sha256)
    if not ms.stage2_uses_cache(motion.provenance):
        raise ValueError("原版16×100必须使用published-cache-with-local-fill来源合同")
    validate_stage2_evidence(motion.provenance, motion.root)
    rec("INPUT_EQUIVALENCE", True, "G0/G2/G3/缓存收据逐SHA回读并验证；输出数值按用户裁决不作同值准入")
    lineage = load_script("scripts/training/check_data_lineage.py")
    checkpoint = store / "v1-store/external/motionjepa" / contract.encoder_run / "checkpoint_epoch_105.pt"
    rec("ENCODER_CONFIG_SHA", sha256_file(checkpoint.parent / "config.yaml") == motion.provenance["encoder"]["config_sha256"])
    result = lineage.check_lineage(checkpoint, contract.encoder_sha256, parent / "meta/input_manifest.json",
                                   contract.mj_source_manifest)
    rec("ENCODER_DATA_LINEAGE", result["files"] == contract.tasks and result["mismatches"] == 0)
    print(f"DATA_LINEAGE=PASS encoder_ckpt={result['encoder_ckpt'][:8]} encoder_source_pin={result['encoder_source_pin']} vla_input_pin={result['vla_input_pin']} files={result['files']} mismatches=0")
    return result


def core(args, argv, *, contract=Contract(), env=None, history=None):
    """同一只读契约核心；CPU夹具仅由Python调用替换身份值，正式CLI无此选项。"""
    rec = Recorder()
    env = dict(os.environ if env is None else env)
    store = Path(args.store_root).resolve()
    pairs = [load_yaml(args.repo, f"perceptual-framesamp-{fusion}-32frame-4x4-motion-stage2-4x256-orig16.yaml")
             for fusion in ("ctx", "modul")]
    rec("PAIRED_CONFIG", {key for key in pairs[0] if pairs[0].get(key) != pairs[1].get(key)}
        == {"integration_type", "memory_token_dim"} and set(pairs[0]) == set(pairs[1]))
    rec("BUILD_PINS", PLACEHOLDER not in str(dataclasses.asdict(contract)), "构建pin必须全部回填")
    rec.guard("ARGV", check_argv, rec, argv, args.steps)
    rec.guard("ENV", legacy.check_env, rec, env, store, contract)
    if env.get("CUDA_VISIBLE_DEVICES") == contract.gpus:
        print("GPU_SCOPE=PASS allowed=0-7")
    names = legacy.flag_value(argv, "--model.history-config")
    if len(names) != 1 or names[0] not in HISTORIES:
        rec("HISTORY_NAME", False, str(names))
        return rec
    name = names[0]
    history = load_yaml(args.repo, name) if history is None else history
    rec.guard("HISTORY", check_history, rec, history, contract)
    rec("NO_PLACEHOLDER", PLACEHOLDER not in str(history) and PLACEHOLDER not in str(argv))
    if not any(x in rec.failed for x in ("BUILD_PINS", "NO_PLACEHOLDER", "HISTORY")):
        rec.guard("DATASET", check_dataset, rec, store, contract, history)
    parsed = rec.guard("CLI", parse_cli, argv)
    if parsed:
        rec.guard("CONFIG", check_config, rec, *parsed, store=store, contract=contract,
                  steps=args.steps, history_name=name, history=history)
    return rec


def check_final_record(path, *, run, steps, head):
    """消费既有最终记录：拒绝末段缺行、非有限标量或非有限EMA，不重算模型。"""
    record = json.loads(Path(path).read_text())
    expected = dict(run_name=run, head=head, num_train_steps=steps, state_step=steps,
                    loop_step=steps - 1, checkpoint_step=steps - 1, log_interval=100)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("最终记录的run、代码或末步身份不符")
    tail = record.get("tail")
    first = (steps - 1) // 100 * 100 + 1
    if not isinstance(tail, list) or [row.get("step") for row in tail] != list(range(first, steps)):
        raise ValueError("最终尾段标量证据缺失或步集合不完整")
    scalars = load_script("scripts/training/final_record.py").SCALARS
    for row in tail:
        values = row.get("scalars")
        if (not isinstance(values, dict) or set(values) != scalars
                or any(not isinstance(value, dict) or value.get("finite") is not True for value in values.values())):
            raise ValueError(f"最终尾段含非有限值或缺标量证据: step={row.get('step')}")
    leaves = record.get("ema_leaves")
    if (not isinstance(leaves, dict) or not leaves
            or any(not isinstance(value, dict) or value.get("finite") is not True for value in leaves.values())):
        raise ValueError("最终EMA含非有限值或缺有限性证据")
    return dict(tail_steps=len(tail), ema_leaves=len(leaves))


def main(raw=None):
    raw = list(sys.argv[1:] if raw is None else raw)
    if raw and raw[0] == "--check-final-record":
        parser = argparse.ArgumentParser(description="核验已有最终保存现场的有限性")
        parser.add_argument("--check-final-record", type=Path, required=True)
        parser.add_argument("--run", required=True)
        parser.add_argument("--steps", type=int, choices=(20, 60000), required=True)
        parser.add_argument("--train-head", required=True)
        args = parser.parse_args(raw)
        try:
            result = check_final_record(args.check_final_record, run=args.run, steps=args.steps, head=args.train_head)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            print(f"TRAIN_FINAL_RECORD=FAIL reason={error}")
            return 1
        print(f"TRAIN_FINAL_RECORD=PASS run={args.run} tail_steps={result['tail_steps']} ema_leaves={result['ema_leaves']}")
        return 0
    if raw == ["--print-library"]:
        print(Contract().library)
        return 0
    argv = []
    if "--" in raw:
        i = raw.index("--")
        raw, argv = raw[:i], raw[i + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--store-root", default=str(REPO))
    parser.add_argument("--train-head", required=True)
    parser.add_argument("--steps", type=int, choices=(20, 60000), default=60000)
    args = parser.parse_args(raw)
    rec = core(args, argv)
    rec.guard("REPO", legacy.check_repo, rec, Path(args.repo), args.train_head)
    # 基础pi05权重仍由已有资产锁核实际文件；夹具机制前向不读该12GB资产。
    rec.guard("PI05", legacy.check_pi05_asset, rec, REPO, Path(args.store_root))
    if not rec.failed:
        print("PAIRED_CONFIG=PASS diff_keys=integration_type,memory_token_dim")
        print(f"PAIRED_RUNS_SAME_LIB=PASS input_pin={Contract().input_manifest_sha256}")
    print(f"LAUNCH_CONTRACT={'FAIL' if rec.failed else 'PASS'} checks={len(rec.rows)} failed={','.join(rec.failed) or 0}")
    return int(bool(rec.failed))


if __name__ == "__main__":
    raise SystemExit(main())
