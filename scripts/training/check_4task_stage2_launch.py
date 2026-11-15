#!/usr/bin/env python3
"""4task RGB+state encoder 下游 VLA（1114 计划）的只读启动合同；正式 CLI 没有夹具或跳过检查入口。

以 check_orig16_stage2_launch.py 为模板逐项改写：两条正式 run 只用 concat 1×1024、8 帧×8×8，
motion 表为 motion-stage2-1024-rgbstate-ep144（来源 local-built-latents），encoder 为 1113 ep144 rgb-state，
血缘改调 check_data_lineage.py 的 member-pin 四环链；步数口径（裁决 2）为正式 50001、短测 20。
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import re
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
CONFIG_NAME = "mme_vla_suite_4task_b128_50k"
PLACEHOLDER = "__FILL_AFTER_BUILD__"
CONCAT = "stage2_pool4_concat_v1"
FORMAL_STEPS, SMOKE_STEPS = 50001, 20
STEPS_CHOICES = (SMOKE_STEPS, FORMAL_STEPS)
SMOKE_PREFIX = "v16-smoke-"
# 期望学习率表（T07 继承 mme_vla_suite_b128_80k）：(warmup, peak, decay_steps, decay_lr)。
SCHEDULE = (5000, 5e-5, 80000, 5e-5)
# 以下两项是 S2 引入的字符串常量；本合同只做字符串比较，不依赖 S2 代码。
LOCAL_BUILT = "local-built-latents"
RGBSTATE_ARCH = "wan-latent-stage2-blockwise-pool4-rgb-state"
MOTION_SUBDIR = "motion-stage2-1024-rgbstate-ep144"
MERGE_RECORDS_REL = "docs/dataset-build-doc/4task-v2-1600ep-604f16da/records"
# 融合方式 → 正式 50k 运行名与公开 bucket（计划 §4）；attempt 的 bucket 只能是同 base 加相同后缀。
RUNS = {"modulation": "v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k",
        "context": "v2-1600ep-m8x8-ctx-motion-s2rgbstate-concat1x1024-b128-50k"}
BUCKETS = {RUNS["modulation"]: "HongzeFu/robomme-vla-4task-m8x8-modul-s2rgbstate-concat1x1024-50k-v1",
           RUNS["context"]: "HongzeFu/robomme-vla-4task-m8x8-ctx-s2rgbstate-concat1x1024-50k-v1"}
FUSIONS = {"ctx": "context", "modul": "modulation"}
ATTEMPT_KEYS = {"base_run", "run", "history", "train_head", "decision", "bucket"}
ATTEMPT_ALLOWED = ("02",)  # R5：attempt 只放行 NN=02，第三次一律等用户
HISTORIES = tuple(f"perceptual-framesamp-{fusion}-8frame-8x8-motion-stage2-1x1024-4task-rgbstate.yaml"
                  for fusion in ("ctx", "modul"))
# 期望配置的模板：同融合的 orig16 8×8 concat yaml，只替换 motion 来源三项（store_path、source_run、checkpoint_sha256）。
TEMPLATES = {fusion: f"perceptual-framesamp-{fusion}-8frame-8x8-motion-stage2-1x1024-orig16.yaml"
             for fusion in ("ctx", "modul")}


def load_script(relative):
    spec = importlib.util.spec_from_file_location(f"_4t_{Path(relative).stem}", REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_script("scripts/training/check_mixed8_stage2_1024_launch.py")
Recorder, sha256_file = legacy.Recorder, legacy.sha256_file


@dataclasses.dataclass(frozen=True)
class Contract:
    """库与 encoder 身份；motion 行数与 train_max 由 L0 回填，encoder sha 由 TRAIN_TERMINAL 后的修订 commit 回填。"""
    library: str = "4task-v2-1600ep-604f16da"
    norm_sha256: str = "856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173"
    episode_manifest_sha256: str = "df0ec8edd823b1415fa2bba6a51fa1c911dadc4d10364590d537a97526add482"
    input_manifest_sha256: str = "655e64b4fd033aea1ff79a1e5ad47f886a148592496759ed2a667fe167c57d86"
    canonical_manifest_sha256: str = "4cd5a170b0ed9718922bfd7c9287e80b3681a0ea7489dfdb07ddeb3a53dbb918"
    motion_rows: int | str = 36484
    train_max: int | str = 71
    encoder_sha256: str = "ece3465b57f9d1ddea130a631aff1d1006b2c8ba5c68a0e5e92b8215ae80b55c"
    encoder_run: str = "stage2-4task-rgbstate-144ep-1113-a"
    encoder_epoch: int = 144
    merge_records_rel: str = MERGE_RECORDS_REL
    episodes: int = 1600
    tasks: int = 4
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

    @property
    def checkpoint_name(self):
        return f"checkpoint_epoch_{self.encoder_epoch}.pt"


def history_fusion(name):
    """由两份 4task history 文件名取融合方式。"""
    if name not in HISTORIES:
        raise ValueError(f"history不在两份4task配置内: {name}")
    return FUSIONS[name.split("-")[2]]


def check_attempt(path, *, history_name, train_head):
    """读取主会话按 R5 登记的 attempt 清单；只放行 -attempt02，任一字段不符即抛错。"""
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict) or set(manifest) != ATTEMPT_KEYS:
        raise ValueError(f"attempt清单字段必须恰为{sorted(ATTEMPT_KEYS)}")
    base, run = manifest["base_run"], manifest["run"]
    match = re.fullmatch(re.escape(str(base)) + r"-attempt(\d{2})", str(run))
    if match is None or match.group(1) not in ATTEMPT_ALLOWED:
        raise ValueError("attempt run必须为base_run加-attempt02（第三次须等用户）")
    suffix = f"-attempt{match.group(1)}"
    fusions = [fusion for fusion, value in RUNS.items() if value == base]
    if not fusions:
        raise ValueError("attempt base_run不是两个正式运行名之一")
    if fusions[0] != history_fusion(history_name):
        raise ValueError("attempt base_run的融合方式与当前history不一致")
    if manifest["history"] != history_name:
        raise ValueError("attempt清单history与当前history不同")
    if manifest["train_head"] != train_head:
        raise ValueError("attempt清单train_head与--train-head不同")
    decision = Path(manifest["decision"])
    if decision.is_symlink() or not decision.is_file():
        raise ValueError("attempt决定记录不存在或不是实体文件")
    allowed = json.loads(decision.read_text()).get("allowed_runs")
    if not isinstance(allowed, list) or run not in allowed:
        raise ValueError("决定记录allowed_runs未登记该attempt")
    if manifest["bucket"] != BUCKETS[base] + suffix:
        raise ValueError("attempt bucket必须为base bucket加相同-attempt02后缀")
    return run


def load_yaml(repo, name):
    if name not in HISTORIES and name not in TEMPLATES.values():
        raise ValueError(f"history不在4task配置内: {name}")
    return legacy.load_yaml(repo, name)


def expected_history(repo, fusion_tag, contract):
    """同融合 orig16 8×8 concat yaml 只换 motion 来源三项，即 4task 期望配置。"""
    expected = load_yaml(repo, TEMPLATES[fusion_tag])
    expected["motion"].update(store_path=f"{contract.parent_rel}/{MOTION_SUBDIR}", online_gpu=1,
                              source_run=f"{contract.encoder_run}/{contract.checkpoint_name}#model",
                              checkpoint_sha256=contract.encoder_sha256)
    return expected


def check_concat_config(pairs):
    """CONCAT_CONFIG：两份 yaml 都是 concat 1×1024、96 窗、stride 32、8 帧×8×8。"""
    return all(h.get("token_per_image") == 64 and h.get("budget") == 512
               and (h["motion"].get("readout"), h["motion"].get("num_tokens"), h["motion"].get("dim"),
                    h["motion"].get("budget"), h["motion"].get("stride")) == (CONCAT, 1, 1024, 96, 32)
               for h in pairs)


def check_history(rec, history, fusion_tag, contract=Contract(), repo=REPO):
    from mme_vla_suite.datastore.motion_adapter import motion_policy_spec
    tokens = history.get("token_per_image")
    rec("FRAME_LAYOUT", tokens == 64 and history.get("budget") == 512
        and history.get("num_views") == 1 and history.get("streaming_obs_horizon") == 16)
    dim = {"context": 2048, "modulation": 1024}.get(history.get("integration_type"))
    rec("FUSION", dim is not None and history.get("integration_type") == FUSIONS[fusion_tag]
        and history.get("memory_token_dim") == dim)
    spec = motion_policy_spec(history.get("motion", {}))
    rec("MOTION_READOUT", (spec.readout, spec.num_tokens, spec.dim, spec.budget, spec.window_budget)
        == (CONCAT, 1, 1024, 96, 96))
    rec("HISTORY_KEYS", history == expected_history(repo, fusion_tag, contract),
        "除motion来源三项外须等于同融合orig16 8x8 concat配置")
    if not rec.failed:
        print("FRAME_LAYOUT=PASS layout=8x8 frames=8 tokens_per_frame=64 budget=512")
        print(f"MOTION_LAYOUT=PASS readout=concat tokens_per_window=1 budget={spec.budget} total_tokens={512 + spec.budget}")
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
    # 正式段（50001）argv 不带步数 flag，步数取自 TrainConfig；短测恰传 ["20"]。
    rec("ARGV_STEPS", (steps == FORMAL_STEPS and not values) or (steps == SMOKE_STEPS and values == [str(SMOKE_STEPS)]),
        f"steps={steps} argv={values}")
    rec("ARGV_HISTORY", argv.count("--model.use-history") == 1)


def check_steps(rec, cfg, base, steps):
    """裁决 2：CFG_STEPS 与 CFG_SCHEDULE。"""
    rec("CFG_STEPS", cfg.num_train_steps == steps and base.num_train_steps == FORMAL_STEPS and steps in STEPS_CHOICES,
        f"cfg={cfg.num_train_steps} base={base.num_train_steps} steps={steps}")
    schedule = cfg.lr_schedule
    rec("CFG_SCHEDULE", schedule == base.lr_schedule and (schedule.warmup_steps, schedule.peak_lr,
        schedule.decay_steps, schedule.decay_lr) == SCHEDULE)


def check_config(rec, cfg, base, *, store, contract, steps, history_name, history, attempt=None):
    from mme_vla_suite.training import config
    for key in ("batch_size", "fsdp_devices", "num_workers", "seed", "ema_decay"):
        rec(f"CFG_{key.upper()}", getattr(cfg, key) == getattr(base, key) == getattr(contract, key))
    check_steps(rec, cfg, base, steps)
    rec("CFG_MODEL", cfg.model == dataclasses.replace(base.model, history_config=history_name) and cfg.model.use_history)
    expected = dataclasses.replace(base, exp_name=cfg.exp_name, num_train_steps=steps,
        model=dataclasses.replace(base.model, history_config=history_name),
        dataset_path=str(store / contract.parent_rel / "framesamp-8x8"),
        assets_base_dir=str(store / "v1-store/train-assets"),
        checkpoint_base_dir=str(store / "v1-store/train-runs"),
        data=dataclasses.replace(base.data, assets=dataclasses.replace(base.data.assets,
            assets_dir=str(store / contract.assets_rel), asset_id="robomme")),
        weight_loader=dataclasses.replace(base.weight_loader, params_path=str(store / contract.pi05_rel)),
        wandb_enabled=cfg.wandb_enabled)
    rec("CFG_EXACT", cfg == expected, "完整配置除已列CLI路径/名称/短测步数外须等于4task 50k配置")
    rec("CFG_IMPORT_ROOT", config.OPENPI_DATA_HOME == str(store / "v1-store/models"))
    # 正式段只许融合方式映射的运行名；设置 attempt 时只放行其已登记的 run；短测只许 v16-smoke- 前缀。
    allowed = RUNS.get(history["integration_type"]) if attempt is None else attempt
    rec("CFG_RUN", cfg.exp_name == allowed if steps == FORMAL_STEPS else cfg.exp_name.startswith(SMOKE_PREFIX),
        f"exp_name={cfg.exp_name} allowed={allowed}")
    rec("CFG_NEW_RUN", not cfg.checkpoint_dir.exists() and not cfg.checkpoint_dir.is_symlink())


def check_dataset(rec, store, contract, *, repo=REPO):
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
    meta = json.loads((parent / "framesamp-8x8/meta/store_meta.json").read_text())
    rec("DATASET_LAYOUT", meta.get("layout") == "framesamp-8x8-v1" and meta.get("status") == "verified"
        and meta.get("manifest_scope") == "full" and meta.get("manifest_sha256") == manifest["sha256"])
    bounds = legacy.manifest_bounds(manifest)
    rec("WINDOW_BUDGET", bounds["rows"] == contract.motion_rows and bounds["train_max"] == contract.train_max
        and type(contract.train_max) is int and contract.train_max <= 96,
        f"rows={bounds['rows']} train_max={bounds['train_max']}")
    motion = ms.MotionMeta.load(parent / MOTION_SUBDIR)
    ms.run_fast_checks(motion, manifest_path=parent / "meta/episode_manifest.json")
    ms.run_full_checks(motion)
    ms.check_same_source(manifest["sha256"], motion, manifest)
    encoder = motion.provenance["encoder"]
    rec("MOTION_VERIFIED", motion.status == "verified" and motion.raw["layout"] == ms.STAGE2_LAYOUT
        and motion.num_rows == contract.motion_rows and motion.row_shape == (1, 1024)
        and encoder["checkpoint_sha256"] == contract.encoder_sha256 and encoder.get("arch") == RGBSTATE_ARCH)
    rec("MOTION_SOURCE", motion.provenance.get("stage2_compat", {}).get("kind") == LOCAL_BUILT,
        f"4task必须使用{LOCAL_BUILT}来源合同")
    validate_stage2_evidence(motion.provenance, motion.root)
    rec("INPUT_EQUIVALENCE", True, "G0/G2/G3与血缘收据逐SHA回读并验证")
    lineage = load_script("scripts/training/check_data_lineage.py")
    checkpoint = store / "v1-store/external/motionjepa" / contract.encoder_run / contract.checkpoint_name
    rec("ENCODER_CONFIG_SHA", sha256_file(checkpoint.parent / "config.yaml") == encoder["config_sha256"])
    records = Path(repo) / contract.merge_records_rel
    # 起跑时对 DATA_LINEAGE 再算一次（同一四环链，不是另一道比对）。
    result = lineage.check_member_pin(checkpoint, contract.encoder_sha256, parent / "meta/input_manifest.json",
                                      records / "MERGE_DONE.json", records)
    rec("ENCODER_DATA_LINEAGE", result["members"] == contract.episodes and result["h5_files"] == contract.tasks
        and result["mismatches"] == 0)
    print(lineage.member_pin_line(result))
    return result


def core(args, argv, *, contract=Contract(), env=None, history=None):
    """同一只读契约核心；CPU 夹具仅由 Python 调用替换身份值，正式 CLI 无此选项。"""
    rec = Recorder()
    env = dict(os.environ if env is None else env)
    store = Path(args.store_root).resolve()
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
    fusion_tag = name.split("-")[2]
    pairs = rec.guard("HISTORY_FILES", lambda: [load_yaml(args.repo, h) for h in HISTORIES])
    if pairs is None:
        return rec
    rec("PAIRED_CONFIG", {key for key in pairs[0] if pairs[0].get(key) != pairs[1].get(key)}
        == {"integration_type", "memory_token_dim"} and set(pairs[0]) == set(pairs[1]))
    if rec("CONCAT_CONFIG", check_concat_config(pairs)):
        print("CONCAT_CONFIG=PASS readout=stage2_pool4_concat_v1 tokens_per_window=1 dim=1024 budget=96 stride=32 diff_keys=integration_type,memory_token_dim")
    attempt = None
    if env.get("ATTEMPT_MANIFEST"):
        attempt = rec.guard("ATTEMPT", check_attempt, env["ATTEMPT_MANIFEST"], history_name=name,
                            train_head=getattr(args, "train_head", None))
        if attempt is not None:
            print(f"ATTEMPT=PASS run={attempt}")
    history = load_yaml(args.repo, name) if history is None else history
    rec.guard("HISTORY", check_history, rec, history, fusion_tag, contract, Path(args.repo))
    rec("NO_PLACEHOLDER", PLACEHOLDER not in str(history) and PLACEHOLDER not in str(argv))
    if not any(x in rec.failed for x in ("BUILD_PINS", "NO_PLACEHOLDER", "HISTORY")):
        rec.guard("DATASET", check_dataset, rec, store, contract, repo=Path(args.repo))
    parsed = rec.guard("CLI", parse_cli, argv)
    if parsed:
        rec.guard("CONFIG", check_config, rec, *parsed, store=store, contract=contract,
                  steps=args.steps, history_name=name, history=history, attempt=attempt)
    return rec


def check_final_record(path, *, run, steps, head):
    """消费既有最终记录：拒绝身份不符、非有限标量或非有限 EMA，不重算模型。

    steps=50001 时期望 state_step=50001、loop_step=checkpoint_step=50000，尾窗 first=50001、期望为空：
    train.py 在 step 50000（%100==0）先清空 infos 再调 final_record.finish，尾段有限性由
    「EMA 叶全部有限」加训练日志 DIAG／TRAIN_NUMERICS 行兜底（串行链 grep），判定行照常输出 tail_steps=0。
    """
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
        parser.add_argument("--steps", type=int, choices=STEPS_CHOICES, required=True)
        parser.add_argument("--train-head", required=True)
        args = parser.parse_args(raw)
        try:
            result = check_final_record(args.check_final_record, run=args.run, steps=args.steps, head=args.train_head)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            print(f"TRAIN_FINAL_RECORD=FAIL reason={error}")
            return 1
        print(f"TRAIN_FINAL_RECORD=PASS run={args.run} steps={args.steps} tail_steps={result['tail_steps']} "
              f"ema_leaves={result['ema_leaves']}")
        return 0
    if raw == ["--print-library"]:
        print(Contract().library)
        return 0
    if raw == ["--print-config"]:
        print(CONFIG_NAME)
        return 0
    argv = []
    if "--" in raw:
        i = raw.index("--")
        raw, argv = raw[:i], raw[i + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument("--store-root", default=str(REPO))
    parser.add_argument("--train-head", required=True)
    parser.add_argument("--steps", type=int, choices=STEPS_CHOICES, default=FORMAL_STEPS)
    args = parser.parse_args(raw)
    rec = core(args, argv)
    rec.guard("REPO", legacy.check_repo, rec, Path(args.repo), args.train_head)
    rec.guard("PI05", legacy.check_pi05_asset, rec, REPO, Path(args.store_root))
    if not rec.failed:
        print("PAIRED_CONFIG=PASS diff_keys=integration_type,memory_token_dim")
        print(f"PAIRED_RUNS_SAME_LIB=PASS input_pin={Contract().input_manifest_sha256}")
    print(f"LAUNCH_CONTRACT={'FAIL' if rec.failed else 'PASS'} checks={len(rec.rows)} failed={','.join(rec.failed) or 0}")
    return int(bool(rec.failed))


if __name__ == "__main__":
    raise SystemExit(main())
