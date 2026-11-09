#!/usr/bin/env python3
"""官方16×100 Wan零参数池化的只读启动合同；除来源身份外与stage2入口同等严格，正式CLI没有夹具或跳过入口。"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
HISTORY_NAME = "perceptual-framesamp-ctx-32frame-4x4-wan-timepool1024-orig16.yaml"
FORMAL_RUN = "orig100-ctx32x4x4-wan-timepool8x8-b128-60k"
SMOKE_RUN = re.compile(r"o16-smoke-timepool-[a-zA-Z0-9_.-]+")
STORE_DIR = "motion-wan-timepool1024"
FRAME_DIR = "framesamp-4x4"
# 旧concat配置 → 新配置的规范化差分必须恰为这七个键；多一个少一个都不是“只换来源身份”。
TRAIN_CONFIG_ALLOWED = frozenset({
    "exp_name", "model.history_config", "history.motion.readout", "history.motion.encoder_kind",
    "history.motion.store_path", "history.motion.source_run", "history.motion.checkpoint_sha256"})
_ABSENT = "<absent>"


def _load_stage2_module():
    spec = importlib.util.spec_from_file_location("_o16_tp_stage2_launch", REPO / "scripts/training/check_orig16_stage2_launch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


s2 = _load_stage2_module()
legacy = s2.legacy
Recorder, Contract, sha256_file = s2.Recorder, s2.Contract, s2.sha256_file
CONFIG_NAME = s2.CONFIG_NAME
OLD_HISTORY = s2.pair_name("ctx", "1x1024")
OLD_RUN = s2.RUNS["context", s2.CONCAT]


def check_history(rec, repo, history, contract=Contract()):
    """新配置必须等于旧concat配置只换readout/kind/store、删来源两键；null也算漂移。"""
    from mme_vla_suite.datastore.motion_adapter import motion_policy_spec
    from mme_vla_suite.shared import wan_timepool as tp
    rec("FRAME_LAYOUT", history.get("token_per_image") == 16 and history.get("budget") == 512
        and history.get("num_views") == 1 and history.get("streaming_obs_horizon") == 16)
    rec("FUSION", history.get("integration_type") == "context" and history.get("memory_token_dim") == 2048)
    spec = motion_policy_spec(history.get("motion", {}))
    rec("MOTION_READOUT", (spec.readout, spec.num_tokens, spec.dim, spec.budget, spec.window_budget, spec.layouts)
        == (tp.TIMEPOOL_READOUT, 1, 1024, 96, 96, (tp.TIMEPOOL_LAYOUT,)))
    expected = legacy.load_yaml(repo, OLD_HISTORY)
    for key in ("source_run", "checkpoint_sha256"):
        del expected["motion"][key]
    expected["motion"].update(readout=tp.TIMEPOOL_READOUT, encoder_kind=tp.TIMEPOOL_ENCODER_KIND,
                              store_path=f"{contract.parent_rel}/{STORE_DIR}")
    rec("HISTORY_KEYS", history == expected, "除readout、encoder_kind、store_path与删除的来源两键外不接受配置漂移")
    if not rec.failed:
        print("FRAME_LAYOUT=PASS layout=4x4 frames=32 tokens_per_frame=16 budget=512")
        print("MOTION_LAYOUT=PASS readout=timepool tokens_per_window=1 budget=96 total_tokens=608")
        print("FUSION=PASS integration=context memory_token_dim=2048")
    return spec


def check_argv(rec, argv, steps):
    s2.check_argv(rec, argv, steps)
    bad = [token for token in argv if re.search(r"resum|overwrite", token, re.IGNORECASE)]
    rec("ARGV_NO_RESUME", not bad, f"禁止resume/overwrite/resum_ckpt_id 实际={bad}")


def check_config(rec, cfg, base, *, store, contract, steps):
    """同 s2.check_config，只把运行名换成本入口唯一允许的名字；学习率日程与保存间隔不因短测改变。"""
    from mme_vla_suite.training import config
    for key in ("batch_size", "fsdp_devices", "num_workers", "seed", "ema_decay"):
        rec(f"CFG_{key.upper()}", getattr(cfg, key) == getattr(base, key) == getattr(contract, key))
    rec("CFG_STEPS", cfg.num_train_steps == steps and base.num_train_steps == 60000 and steps in (20, 60000))
    rec("CFG_SCHEDULE", cfg.lr_schedule == base.lr_schedule and (cfg.lr_schedule.warmup_steps,
        cfg.lr_schedule.peak_lr, cfg.lr_schedule.decay_steps, cfg.lr_schedule.decay_lr) == (5000, 5e-5, 60000, 5e-5))
    rec("CFG_SAVE", (cfg.save_interval, cfg.keep_period) == (base.save_interval, base.keep_period) == (5000, 5000))
    rec("CFG_MODEL", cfg.model == dataclasses.replace(base.model, history_config=HISTORY_NAME) and cfg.model.use_history)
    rec("CFG_NO_RESUME", not cfg.resume and not cfg.overwrite and cfg.resum_ckpt_id is None)
    expected = dataclasses.replace(base, exp_name=cfg.exp_name, num_train_steps=steps,
        model=dataclasses.replace(base.model, history_config=HISTORY_NAME),
        dataset_path=str(store / contract.parent_rel / FRAME_DIR), assets_base_dir=str(store / "v1-store/train-assets"),
        checkpoint_base_dir=str(store / "v1-store/train-runs"),
        data=dataclasses.replace(base.data, assets=dataclasses.replace(base.data.assets,
            assets_dir=str(store / contract.assets_rel), asset_id="robomme")),
        weight_loader=dataclasses.replace(base.weight_loader, params_path=str(store / contract.pi05_rel)),
        wandb_enabled=cfg.wandb_enabled)
    rec("CFG_EXACT", cfg == expected, "完整配置除已列CLI路径/名称/短测步数外须等于新60k配置")
    rec("CFG_IMPORT_ROOT", config.OPENPI_DATA_HOME == str(store / "v1-store/models"))
    rec("CFG_RUN", cfg.exp_name == FORMAL_RUN if steps == 60000 else SMOKE_RUN.fullmatch(cfg.exp_name) is not None,
        f"exp_name={cfg.exp_name}")
    rec("CFG_NEW_RUN", not cfg.checkpoint_dir.exists() and not cfg.checkpoint_dir.is_symlink())


def check_store_head(rec, repo, store_head, train_head):
    """建库HEAD须是--train-head的祖先，且两者之间 src、scripts 零差异。"""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    pair = f"store={store_head} train={train_head}"
    rec("STORE_HEAD_ANCESTOR", all(re.fullmatch(r"[0-9a-f]{40}", str(x) or "") for x in (store_head, train_head))
        and git("merge-base", "--is-ancestor", store_head, train_head).returncode == 0, pair)
    diff = git("diff", "--name-only", str(store_head), str(train_head), "--", "src", "scripts")
    changed = diff.stdout.split()
    rec("STORE_HEAD_CODE_EQ", diff.returncode == 0 and not changed, f"{pair} changed={changed[:8]}")


def source_h5_mismatches(provenance, input_manifest):
    """库来源的 H5 与 input manifest 按 名→(字节,SHA) 集合比较；返回不一致的文件名。"""
    files = input_manifest["files"]
    declared = {name: (item["size"], item["sha256"]) for name, item in files.items()}
    recorded = {item["name"]: (item["bytes"], item["sha256"]) for item in provenance["source_h5"]}
    return sorted(name for name in declared.keys() | recorded.keys() if declared.get(name) != recorded.get(name))


def check_lineage(rec, provenance, parent, contract):
    """零参数池化没有encoder，但16 H5与input manifest的集合恒等不可豁免；逐H5重算由G0完成并绑定在记录里。"""
    inputs = json.loads((parent / "meta/input_manifest.json").read_text())
    mismatches = source_h5_mismatches(provenance, inputs)
    files = len(inputs["files"])
    rec("SOURCE_H5_SET", not mismatches and files == contract.tasks, f"files={files} mismatches={mismatches}")
    rec("NO_ENCODER", provenance["encoder_ckpt"] is None and provenance["parameter_count"] == 0
        and provenance["encoder_kind"] == "wan_fixedpool1024")
    for role, record in provenance["validation"].items():
        path = Path(record["path"])
        rec(f"RECORD_{role}", path.is_file() and sha256_file(path) == record["sha256"], str(path))
    verdict = provenance["validation"]["DATA_LINEAGE"]["verdict"]
    want = f"DATA_LINEAGE=PASS encoder_kind=wan_fixedpool1024 encoder_ckpt=none files={contract.tasks} mismatches=0"
    rec("RECORD_LINEAGE_VERDICT", verdict == want, verdict)
    if not mismatches and verdict == want:
        print(want)


def check_dataset(rec, repo, train_head, store, contract, history):
    from omegaconf import OmegaConf
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.datastore.manifest import load_manifest
    from mme_vla_suite.training.framesamp_dataset import check_motion_contract
    parent = store / contract.parent_rel
    for name, want in (("episode_manifest.json", contract.episode_manifest_sha256),
                       ("input_manifest.json", contract.input_manifest_sha256)):
        rec(f"{name.upper()}_SHA", sha256_file(parent / "meta" / name) == want)
    rec("NORM_SHA", sha256_file(store / contract.assets_rel / "robomme/norm_stats.json") == contract.norm_sha256)
    manifest = load_manifest(parent / "meta/episode_manifest.json")
    rec("MANIFEST_IDENTITY", manifest["sha256"] == contract.canonical_manifest_sha256
        and len(manifest["episodes"]) == contract.episodes and len(manifest["canonical_order"]) == contract.tasks)
    meta = json.loads((parent / FRAME_DIR / "meta/store_meta.json").read_text())
    rec("DATASET_LAYOUT", meta.get("layout") == "framesamp-4x4-v1" and meta.get("status") == "verified"
        and meta.get("manifest_scope") == "full" and meta.get("manifest_sha256") == manifest["sha256"])
    bounds = legacy.manifest_bounds(manifest)
    rec("WINDOW_BUDGET", bounds["rows"] == contract.motion_rows and bounds["train_max"] == contract.train_max
        and type(contract.train_max) is int and contract.train_max <= 96)
    root = parent / STORE_DIR
    ms.require_no_pack_lock(root)
    motion = ms.MotionMeta.load(root)
    rec("STORE_VERIFIED", motion.status == "verified", f"status={motion.status}")
    if motion.status != "verified":
        return
    ms.run_fast_checks(motion, manifest_path=parent / "meta/episode_manifest.json")
    ms.run_full_checks(motion)
    ms.check_same_source(manifest["sha256"], motion, manifest)
    check_motion_contract(OmegaConf.create(history).motion, motion)
    rec("MOTION_VERIFIED", motion.raw["layout"] == ms.TIMEPOOL_LAYOUT and motion.num_rows == contract.motion_rows
        and motion.row_shape == (1, 1024))
    rec("STORE_MANIFESTS", motion.provenance["manifests"]["episode_manifest_sha256"] == manifest["sha256"]
        and motion.provenance["manifests"]["episode_manifest_file_sha256"] == contract.episode_manifest_sha256
        and motion.provenance["manifests"]["input_manifest_sha256"] == contract.input_manifest_sha256)
    check_lineage(rec, motion.provenance, parent, contract)
    check_store_head(rec, repo, motion.provenance["train_head"], train_head)


def _flatten(prefix, value, out):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out[f"{prefix}.__type__".lstrip(".")] = type(value).__qualname__
        for field in dataclasses.fields(value):
            _flatten(f"{prefix}.{field.name}", getattr(value, field.name), out)
    elif isinstance(value, dict) and value:
        # 键集合由各子键是否出现体现，缺键在差分里即一侧缺失。
        for key in sorted(value, key=str):
            _flatten(f"{prefix}.{key}", value[key], out)
    elif isinstance(value, (list, tuple)):
        out[f"{prefix}.__len__".lstrip(".")] = len(value)
        for index, item in enumerate(value):
            _flatten(f"{prefix}.{index}", item, out)
    else:
        # repr 里的内存地址每次进程不同，不是配置差异。
        out[prefix.lstrip(".")] = re.sub(r" at 0x[0-9a-fA-F]+", "", repr(value))


def normalize_train_config(cfg, history):
    """TrainConfig 与解析后的 history YAML 展平成 {点分键: 规范字符串}；history 键带 history. 前缀。"""
    flat = {}
    _flatten("", cfg, flat)
    _flatten("history", history, flat)
    return flat


def train_config_diff(old_flat, new_flat):
    """两份规范化配置取值不同（含一侧缺失）的键。"""
    return sorted(key for key in old_flat.keys() | new_flat.keys()
                  if old_flat.get(key, _ABSENT) != new_flat.get(key, _ABSENT))


def formal_argv(store_root, *, run=FORMAL_RUN, history=HISTORY_NAME, steps=60000, library=None):
    """与 run_orig16_timepool_60k.sh 相同形态的训练参数（关闭W&B）；E 可不经入口直接取得。"""
    root = str(Path(store_root).resolve())
    lib = library or Contract().library
    argv = [CONFIG_NAME, "--exp-name", run, "--dataset-path", f"{root}/v1-store/datasets/{lib}/{FRAME_DIR}",
            "--assets-base-dir", f"{root}/v1-store/train-assets",
            "--data.assets.assets-dir", f"{root}/v1-store/train-assets/mme_vla_suite/{lib}",
            "--data.assets.asset-id", "robomme", "--checkpoint-base-dir", f"{root}/v1-store/train-runs",
            "--weight-loader.params-path", f"{root}/{Contract().pi05_rel}",
            "--model.use-history", "--model.history-config", history, "--no-wandb-enabled"]
    return argv + (["--num-train-steps", "20"] if steps == 20 else [])


def old_concat_argv(new_argv):
    """旧concat入口的规范对照：只从新参数取路径类值与短测/W&B形态，其余一概不带，新侧多传的漂移才看得见。"""
    argv = [CONFIG_NAME, "--exp-name", OLD_RUN, "--model.use-history", "--model.history-config", OLD_HISTORY]
    for flag in ("--dataset-path", "--assets-base-dir", "--data.assets.assets-dir", "--data.assets.asset-id",
                 "--checkpoint-base-dir", "--weight-loader.params-path"):
        for value in legacy.flag_value(new_argv, flag):
            argv += [flag, value]
    if legacy.flag_value(new_argv, "--num-train-steps") == ["20"]:
        argv += ["--num-train-steps", "20"]
    return argv + (["--no-wandb-enabled"] if "--no-wandb-enabled" in new_argv else [])


def _identity_errors(flat):
    """差分键集合只说明“变了”；这里再核新侧变成了什么：来源两键须完全不存在，null也不行。"""
    from mme_vla_suite.shared import wan_timepool as tp
    errors = [key for key in ("history.motion.source_run", "history.motion.checkpoint_sha256") if key in flat]
    for key, want in (("history.motion.readout", tp.TIMEPOOL_READOUT), ("history.motion.encoder_kind", tp.TIMEPOOL_ENCODER_KIND)):
        if flat.get(key) != repr(want):
            errors.append(key)
    if not str(flat.get("history.motion.store_path", "")).endswith(f"/{STORE_DIR}'"):
        errors.append("history.motion.store_path")
    return errors


def train_config_eq(new_argv, *, repo=REPO, new_history=None, old_argv=None, old_history=None):
    """规范化差分：旧concat对新配置。返回 dict(ok, verdict, diff_keys, unexpected, missing)，不打印不抛判定。"""
    new_cfg, _ = s2.parse_cli(new_argv)
    old_cfg, _ = s2.parse_cli(old_argv or old_concat_argv(new_argv))
    new_history = legacy.load_yaml(repo, HISTORY_NAME) if new_history is None else new_history
    old_history = legacy.load_yaml(repo, OLD_HISTORY) if old_history is None else old_history
    new_flat = normalize_train_config(new_cfg, new_history)
    diff = set(train_config_diff(normalize_train_config(old_cfg, old_history), new_flat))
    unexpected = sorted((diff - TRAIN_CONFIG_ALLOWED) | set(_identity_errors(new_flat)))
    missing = sorted(TRAIN_CONFIG_ALLOWED - diff)
    ok = not unexpected and not missing
    verdict = (f"TRAIN_CONFIG_EQ={'PASS' if ok else 'FAIL'} diff_keys={','.join(sorted(diff))}"
               + ("" if ok else f" unexpected={unexpected} missing={missing}"))
    return dict(ok=ok, verdict=verdict, diff_keys=sorted(diff), unexpected=unexpected, missing=missing)


def core(args, argv, *, contract=Contract(), env=None, history=None):
    """同一只读契约核心；CPU夹具仅由Python调用替换身份值，正式CLI无此选项。"""
    rec = Recorder()
    env = dict(os.environ if env is None else env)
    store = Path(args.store_root).resolve()
    rec("BUILD_PINS", s2.PLACEHOLDER not in str(dataclasses.asdict(contract)), "构建pin必须全部回填")
    rec.guard("ARGV", check_argv, rec, argv, args.steps)
    rec.guard("ENV", legacy.check_env, rec, env, store, contract)
    if env.get("CUDA_VISIBLE_DEVICES") == contract.gpus:
        print("GPU_SCOPE=PASS allowed=0-7")
    names = legacy.flag_value(argv, "--model.history-config")
    if names != [HISTORY_NAME]:
        rec("HISTORY_NAME", False, str(names))
        return rec
    history = rec.guard("HISTORY_LOAD", legacy.load_yaml, args.repo, HISTORY_NAME) if history is None else history
    if history is None:
        return rec
    rec.guard("HISTORY", check_history, rec, args.repo, history, contract)
    rec.guard("DATASET", check_dataset, rec, args.repo, args.train_head, store, contract, history)
    parsed = rec.guard("CLI", s2.parse_cli, argv)
    if parsed:
        rec.guard("CONFIG", check_config, rec, *parsed, store=store, contract=contract, steps=args.steps)
    eq = rec.guard("TRAIN_CONFIG_EQ", train_config_eq, argv, repo=args.repo, new_history=history)
    if eq:
        print(eq["verdict"])
        rec("TRAIN_CONFIG_EQ", eq["ok"])
    return rec


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
            result = s2.check_final_record(args.check_final_record, run=args.run, steps=args.steps, head=args.train_head)
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
    # 基础pi05权重仍由已有资产锁核实际文件。
    rec.guard("PI05", legacy.check_pi05_asset, rec, REPO, Path(args.store_root))
    print(f"LAUNCH_CONTRACT={'FAIL' if rec.failed else 'PASS'} checks={len(rec.rows)} failed={','.join(rec.failed) or 0}")
    return int(bool(rec.failed))


if __name__ == "__main__":
    raise SystemExit(main())
