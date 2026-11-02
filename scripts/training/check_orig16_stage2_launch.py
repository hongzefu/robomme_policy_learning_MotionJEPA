#!/usr/bin/env python3
"""官方16×100的只读启动合同；正式CLI没有夹具或跳过检查入口。"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib.util
import json
import os
import re
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
CONFIG_NAME = "mme_vla_suite_orig16_b128_60k"
PLACEHOLDER = "__FILL_AFTER_BUILD__"
YAML_DIR = "src/mme_vla_suite/models/config/robomme"
ENCODER_SHA = "27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035"
ENCODER_RUN = "stage2-full144-1025-a"
ABLATION_CONFIG = "mme_vla_suite_orig16_b64_80k"
# S5 对齐此身份表：run 为键，字段为 ckpt_sha256/config_sha256/store_dir/evidence_dir/yaml。
# 新配置指纹来自各公开 encoder bucket 的 SHA256SUMS.txt；旧值来自入库 G4-ctx.json。
ENCODERS = {
    ENCODER_RUN: dict(ckpt_sha256=ENCODER_SHA,
        config_sha256="879213d9bc47d3cb39b133acc7b6534333cc70faf0222cf566d52d28348262c6",
        store_dir="motion-stage2-1024", evidence_dir="evidence",
        yaml="perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-orig16.yaml"),
    "aws-stage2-noxyz-1101-a": dict(
        ckpt_sha256="6acc92af239e9b9a4c58ced4c5123bc25183188ec518b10734e917879240d00d",
        config_sha256="583e48c0a6afa2a798eb14fc8ccdcfe8d9274cb13f1393e05859d6bca368bada",
        store_dir="motion-stage2-1024-noxyz", evidence_dir="evidence-noxyz",
        yaml="perceptual-framesamp-ctx-32frame-4x4-noxyz-motion-stage2-4x256-orig16.yaml"),
    "aws-stage2-nostate-1101-a": dict(
        ckpt_sha256="be7cdec7d96c889193f27cd925977c7ff47423d4321d1fd7446799d38495e3d4",
        config_sha256="869ac77872092a0d0b08e1bc3ac60b0a893946af02ae3de193aa5715e8aecc34",
        store_dir="motion-stage2-1024-nostate", evidence_dir="evidence-nostate",
        yaml="perceptual-framesamp-ctx-32frame-4x4-nostate-motion-stage2-4x256-orig16.yaml"),
}
SPLIT, CONCAT = "stage2_pool4_split_v1", "stage2_pool4_concat_v1"
# (融合方式, motion读法) → 正式60k运行名；四项显式映射，不按字符串规则推导。
RUNS = {("context", SPLIT): "orig100-ctx32x4x4-motion-s2split4x256-b128-60k",
        ("modulation", SPLIT): "orig100-m32x4x4-modul-motion-s2split4x256-b128-60k",
        ("context", CONCAT): "orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k",
        ("modulation", CONCAT): "orig100-m32x4x4-modul-motion-s2concat1x1024-b128-60k"}
# 正式运行名 → 公开备份bucket；attempt的bucket只能是同base加相同后缀。
BUCKETS = {RUNS["context", SPLIT]: "HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2split-60k-v1",
           RUNS["modulation", SPLIT]: "HongzeFu/robomme-vla-orig100-m32x4x4-modul-motion-s2split-60k-v1",
           RUNS["context", CONCAT]: "HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2concat1x1024-60k-v1",
           RUNS["modulation", CONCAT]: "HongzeFu/robomme-vla-orig100-m32x4x4-modul-motion-s2concat1x1024-60k-v1"}
FUSIONS = {"ctx": "context", "modul": "modulation"}
READOUTS = {"4x256": SPLIT, "1x1024": CONCAT}
ATTEMPT_KEYS = {"base_run", "run", "history", "train_head", "decision", "bucket"}
HISTORIES = tuple(f"perceptual-framesamp-{fusion}-{frames}-motion-stage2-{readout}-orig16.yaml"
    for fusion in ("ctx", "modul") for frames in ("32frame-4x4", "8frame-8x8") for readout in ("1x1024", "4x256")) + tuple(
        item["yaml"] for run, item in ENCODERS.items() if run != ENCODER_RUN)
# ── GreatLakes 复现（1031 计划 K4/K6/K9/K11）：卡数可变口径，只对 fsdp 2/4 生效，默认 8 卡 B128 不变 ──
GL_FSDP = (2, 4)
GL_HISTORY = "perceptual-framesamp-ctx-32frame-4x4-motion-stage2-4x256-orig16.yaml"
# 三份测时 yaml（仅供 o16gl-tier- 测时）→ 帧路记忆总预算。
TIER_HISTORIES = {f"perceptual-framesamp-ctx-{frames}frame-4x4-nomotion-tier-orig16.yaml": frames * 16
                  for frames in (16, 32, 64)}
GL_SMOKE, GL_TIER = "o16gl-smoke-", "o16gl-tier-"
# (运行类别, 步数) → 允许的 global batch；表外组合一律 FAIL。
GL_COMBOS = {("formal", 60000): (64, 32), ("formal", 80000): (64,),
             ("smoke", 20): (64, 32, 16, 8, 4), ("smoke", 100): (64, 32), ("tier", 100): (64,)}
EVIDENCE_ENV = "MMEVLA_STAGE2_EVIDENCE_ROOT"
ONLINE_INPUTS = "online_restore_inputs.json"


def gl_formal_run(batch, encoder_run=ENCODER_RUN):
    if encoder_run != ENCODER_RUN:
        tag = ENCODERS[encoder_run]["store_dir"].removeprefix("motion-stage2-1024-")
        return f"gl-orig100-ctx32x4x4-{tag}-motion-s2split4x256-b{batch}-80k"
    return f"gl-orig100-ctx32x4x4-motion-s2split4x256-b{batch}-60k"


GL_RUNS = {gl_formal_run(batch): GL_HISTORY for batch in (64, 32)} | {
    gl_formal_run(64, run): item["yaml"] for run, item in ENCODERS.items() if run != ENCODER_RUN}
BUCKETS.update({gl_formal_run(batch): "HongzeFu/robomme-vla-gl-orig100-ctx32x4x4-motion-s2split-60k-v1"
                for batch in (64, 32)})
BUCKETS.update({gl_formal_run(64, f"aws-stage2-{tag}-1101-a"):
    f"HongzeFu/robomme-vla-gl-orig100-ctx-s2split-{tag}-b64-80k-v1" for tag in ("noxyz", "nostate")})


def gl_kind(run):
    """GL 运行类别：o16gl-tier- 测时、o16gl-smoke- 冒烟，其余按正式口径。"""
    return "tier" if run.startswith(GL_TIER) else "smoke" if run.startswith(GL_SMOKE) else "formal"


def gl_combo_ok(*, fsdp, batch, steps, run, history_name, encoder_run=ENCODER_RUN, config_name=CONFIG_NAME):
    """入口与合同共用的 GL 组合表；正式只认 ctx 4x256 与 gl-orig100-…-b<batch>-60k。"""
    kind = gl_kind(run)
    if fsdp not in GL_FSDP or batch not in GL_COMBOS.get((kind, steps), ()):
        return False
    if kind == "tier":
        return history_name in TIER_HISTORIES and encoder_run == ENCODER_RUN and config_name == CONFIG_NAME
    new = encoder_run != ENCODER_RUN
    if config_name != (ABLATION_CONFIG if new else CONFIG_NAME):
        return False
    if new and (fsdp != 4 or batch != 64 or steps not in (20, 100, 80000)):
        return False
    if not new and steps == 80000:
        return False
    formal = gl_formal_run(batch, encoder_run)
    return history_name == ENCODERS[encoder_run]["yaml"] and (kind != "formal" or
        re.fullmatch(re.escape(formal) + r"(?:-attempt(0[2-9]|[1-9][0-9]))?", run) is not None)


def rebase_provenance(provenance, root):
    """只把 stage2_compat.evidence.*.path 的父目录换成 root（不换文件名），其余逐字不动。"""
    prov = copy.deepcopy(provenance)
    for item in prov["stage2_compat"]["evidence"].values():
        item["path"] = str(Path(root) / Path(item["path"]).name)
    return prov


def verify_evidence_root(root, provenance):
    """逐文件核重定根目录：五份证据按 provenance 记录的 SHA，在线输入 NPZ 按其 JSON 内 payload SHA。"""
    from mme_vla_suite.datastore import motion_store as ms
    root = Path(root)
    evidence = provenance["stage2_compat"]["evidence"]
    bad = []
    for role in ms.stage2_evidence_roles(provenance):
        path = root / Path(evidence[role]["path"]).name
        if not path.is_file() or sha256_file(path) != evidence[role]["sha256"]:
            bad.append(role)
    portable = root / ONLINE_INPUTS
    payload = json.loads(portable.read_text()).get("payload", {}) if portable.is_file() else {}
    npz = root / "online_restore_inputs.npz"
    if not portable.is_file():
        bad.append(ONLINE_INPUTS)
    if payload.get("path") != npz.name or not npz.is_file() or sha256_file(npz) != payload.get("sha256"):
        bad.append(npz.name)
    return len(ms.stage2_evidence_roles(provenance)) + 2, bad


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
    library: str = "16task-orig100-1600ep-9e109419"
    norm_sha256: str = "c5d45b4bdb481ca46d18aeb32446bbedec34de435eef31d3212a179ee7868bcd"
    episode_manifest_sha256: str = "93d7c5a0c1f34fd955de5f8c37370369e54457019f75e56aea8908cba894d999"
    input_manifest_sha256: str = "5d1bfed54377c22e68bc1883ebd35fdf77372129ac1565dd6a78601c1d9965f9"
    canonical_manifest_sha256: str = "337c4fcde163974039b5c0f2ad1f46ebaacd272719f72818503e23c1360447e2"
    motion_rows: int | str = 23253
    train_max: int | str = 44
    encoder_sha256: str = ENCODER_SHA
    encoder_run: str = "stage2-full144-1025-a"
    store_dir: str = "motion-stage2-1024"
    evidence_dir: str = "evidence"
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


def history_key(name):
    """由八组合history文件名取 (融合方式, 读法)。"""
    if name not in HISTORIES:
        raise ValueError(f"history不在八组合内: {name}")
    parts = name.split("-")
    return FUSIONS[parts[2]], READOUTS[parts[-2]]


def pair_name(fusion, readout, encoder_run=ENCODER_RUN):
    """32×4×4正式配置文件名；fusion取ctx/modul，readout取4x256/1x1024。"""
    if encoder_run != ENCODER_RUN and (fusion, readout) == ("ctx", "4x256"):
        return ENCODERS[encoder_run]["yaml"]
    return f"perceptual-framesamp-{fusion}-32frame-4x4-motion-stage2-{readout}-orig16.yaml"


def check_attempt(path, *, history_name, train_head):
    """读取用户决定后登记的attempt清单；任一字段不符即抛错，不回退到正式名。"""
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict) or set(manifest) != ATTEMPT_KEYS:
        raise ValueError(f"attempt清单字段必须恰为{sorted(ATTEMPT_KEYS)}")
    base, run = manifest["base_run"], manifest["run"]
    match = re.fullmatch(re.escape(str(base)) + r"-attempt(\d{2})", str(run))
    if match is None or int(match.group(1)) < 2:
        raise ValueError("attempt run必须为base_run加-attemptNN（NN≥02两位）")
    suffix = f"-attempt{match.group(1)}"
    keys = [key for key, value in RUNS.items() if value == base]
    if not keys and base not in GL_RUNS:
        raise ValueError("attempt base_run不是已登记的正式运行名")
    if (keys and keys[0] != history_key(history_name)) or (base in GL_RUNS and GL_RUNS[base] != history_name):
        raise ValueError("attempt base_run的融合或读法与当前history不一致")
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
        raise ValueError("attempt bucket必须为base bucket加相同-attemptNN后缀")
    return run


def check_nosplit_config(repo, fusion):
    """同融合split/concat两份yaml只允许motion读法四项不同。"""
    split = legacy.load_yaml(repo, pair_name(fusion, "4x256"))
    concat = legacy.load_yaml(repo, pair_name(fusion, "1x1024"))
    top = {key for key in set(split) | set(concat) if split.get(key) != concat.get(key)}
    motion = {key for key in set(split["motion"]) | set(concat["motion"])
              if split["motion"].get(key) != concat["motion"].get(key)}
    return top <= {"motion"} and set(split["motion"]) == set(concat["motion"]) \
        and motion == {"readout", "num_tokens", "dim", "budget"}


def load_yaml(repo, name):
    if name not in HISTORIES and name not in TIER_HISTORIES:
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
        store_path=contract.parent_rel + "/" + contract.store_dir, online_gpu=1,
        source_run=f"{contract.encoder_run}/checkpoint_epoch_105.pt#model", checkpoint_sha256=contract.encoder_sha256)
    rec("HISTORY_KEYS", history == expected, "除三机制与已固定来源以外不接受配置漂移")
    if not rec.failed:
        print(f"FRAME_LAYOUT=PASS layout={'4x4' if tokens == 16 else '8x8'} frames={512 // tokens} tokens_per_frame={tokens} budget=512")
        print(f"MOTION_LAYOUT=PASS readout={'split' if spec.num_tokens == 4 else 'concat'} tokens_per_window={spec.num_tokens} budget={spec.budget} total_tokens={512 + spec.budget}")
        print(f"FUSION=PASS integration={history['integration_type']} memory_token_dim={dim}")
    return spec


def check_tier_history(rec, history, name):
    """测时 yaml：正式 ctx 4x256 删 motion 段、只改 budget，其余逐键相同。"""
    budget = TIER_HISTORIES[name]
    rec("FRAME_LAYOUT", history.get("token_per_image") == 16 and history.get("budget") == budget
        and history.get("num_views") == 1 and history.get("streaming_obs_horizon") == 16)
    rec("FUSION", history.get("integration_type") == "context" and history.get("memory_token_dim") == 2048)
    expected = {key: value for key, value in legacy.load_yaml(REPO, GL_HISTORY).items() if key != "motion"}
    expected["budget"] = budget
    rec("HISTORY_KEYS", history == expected, "测时yaml只允许删motion段与改budget")
    if not rec.failed:
        print(f"FRAME_LAYOUT=PASS layout=4x4 frames={budget // 16} tokens_per_frame=16 budget={budget}")
        print("MOTION_LAYOUT=SKIP motion=0 reason=tier")
        print("FUSION=PASS integration=context memory_token_dim=2048")
    return None


def parse_cli(argv, config_name=CONFIG_NAME):
    from mme_vla_suite.training import config
    previous = sys.argv
    try:
        sys.argv = ["train.py", *argv]
        return config.cli(), config.get_config(config_name)
    finally:
        sys.argv = previous


def check_argv(rec, argv, steps, gl=False, config_name=CONFIG_NAME):
    rec("ARGV_CONFIG_NAME", bool(argv) and argv[0] == config_name)
    allowed = legacy.VALUE_FLAGS | {"--model.history-config"}
    if gl:
        allowed |= {"--fsdp-devices", "--num-workers", "--batch-size"}
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
    formal_steps = 80000 if config_name == ABLATION_CONFIG else 60000
    rec("ARGV_STEPS", (steps == formal_steps and not values) or (steps == 20 and values == ["20"])
        or (gl and steps == 100 and values == ["100"]))
    rec("ARGV_HISTORY", argv.count("--model.use-history") == 1)


def check_config(rec, cfg, base, *, store, contract, steps, history_name, history, attempt=None,
                 fsdp_devices=None, batch_size=None, config_name=CONFIG_NAME):
    from mme_vla_suite.training import config
    # GL 口径只放开 batch 与 fsdp/num_workers 两项资源量；未传时期望值即合同默认（逐位同原检查）。
    fsdp = contract.fsdp_devices if fsdp_devices is None else fsdp_devices
    gl = fsdp in GL_FSDP
    ablation = config_name == ABLATION_CONFIG
    formal_steps = 80000 if ablation else 60000
    base_resources = dict(batch_size=64, fsdp_devices=4, num_workers=8) if ablation else {}
    want = dict(batch_size=contract.batch_size if batch_size is None else batch_size, fsdp_devices=fsdp,
                num_workers=2 * fsdp if gl else contract.num_workers, seed=contract.seed, ema_decay=contract.ema_decay)
    for key in ("batch_size", "fsdp_devices", "num_workers", "seed", "ema_decay"):
        rec(f"CFG_{key.upper()}", getattr(cfg, key) == want[key]
            and getattr(base, key) == base_resources.get(key, getattr(contract, key)))
    rec("CFG_STEPS", cfg.num_train_steps == steps and base.num_train_steps == formal_steps
        and steps in ((20, 100, formal_steps) if gl else (20, 60000)))
    rec("CFG_SCHEDULE", cfg.lr_schedule == base.lr_schedule and (cfg.lr_schedule.warmup_steps,
        cfg.lr_schedule.peak_lr, cfg.lr_schedule.decay_steps, cfg.lr_schedule.decay_lr)
        == (10000 if ablation else 5000, 5e-5, formal_steps, 5e-5))
    rec("CFG_MODEL", cfg.model == dataclasses.replace(base.model, history_config=history_name) and cfg.model.use_history)
    layout = "framesamp-4x4" if history["token_per_image"] == 16 else "framesamp-8x8"
    expected = dataclasses.replace(base, exp_name=cfg.exp_name, num_train_steps=steps,
        batch_size=want["batch_size"], fsdp_devices=want["fsdp_devices"], num_workers=want["num_workers"],
        model=dataclasses.replace(base.model, history_config=history_name),
        dataset_path=str(store / contract.parent_rel / layout), assets_base_dir=str(store / "v1-store/train-assets"),
        checkpoint_base_dir=str(store / "v1-store/train-runs"),
        data=dataclasses.replace(base.data, assets=dataclasses.replace(base.data.assets,
            assets_dir=str(store / contract.assets_rel), asset_id="robomme")),
        weight_loader=dataclasses.replace(base.weight_loader, params_path=str(store / contract.pi05_rel)),
        wandb_enabled=cfg.wandb_enabled)
    rec("CFG_EXACT", cfg == expected, "完整配置除已列CLI路径/名称/短测步数外须等于新60k配置")
    rec("CFG_IMPORT_ROOT", config.OPENPI_DATA_HOME == str(store / "v1-store/models"))
    # 正式60k只允许32×4×4与(融合,读法)映射的运行名；设置attempt时只放行其已登记的run。
    if gl:
        # GL：正式只认 gl-orig100-…-b<64|32>-60k；20/100 步只认 o16gl- 前缀，组合细则见 GL_COMBO。
        kind = gl_kind(cfg.exp_name)
        rec("CFG_RUN", gl_combo_ok(fsdp=fsdp, batch=want["batch_size"], steps=steps, run=cfg.exp_name,
            history_name=history_name, encoder_run=contract.encoder_run if contract.encoder_run in ENCODERS else ENCODER_RUN,
            config_name=config_name) and (kind == "formal") == (steps == formal_steps)
            and (kind != "formal" or "-attempt" not in cfg.exp_name or cfg.exp_name == attempt),
            f"exp_name={cfg.exp_name} kind={kind}")
    else:
        allowed = RUNS.get((history["integration_type"], history["motion"]["readout"])) if attempt is None else attempt
        rec("CFG_RUN", (history["token_per_image"] == 16 and cfg.exp_name == allowed) if steps == 60000
            else cfg.exp_name.startswith("o16-smoke-"), f"exp_name={cfg.exp_name} allowed={allowed}")
    rec("CFG_NEW_RUN", not cfg.checkpoint_dir.exists() and not cfg.checkpoint_dir.is_symlink())


def check_evidence_lineage(rec, evidence_dir, motion, checkpoint, contract):
    """MJ 源清单缺失时的证据 SHA 路由（K11/F4）：核库根 evidence/ 五份与 encoder，不调 check_lineage、不读 H5。"""
    from mme_vla_suite.datastore import motion_store as ms
    evidence = motion.provenance["stage2_compat"]["evidence"]
    roles = ms.stage2_evidence_roles(motion.provenance)
    bad = [role for role in roles if sha256_file(evidence_dir / f"{role}.json") != evidence[role]["sha256"]]
    ckpt = sha256_file(checkpoint)
    ok = rec("ENCODER_DATA_LINEAGE", not bad and ckpt == contract.encoder_sha256
             == motion.provenance["encoder"]["checkpoint_sha256"], f"route=evidence mismatches={bad}")
    print("LINEAGE_ROUTE=evidence")
    if ok:
        print(f"DATA_LINEAGE=PASS route=evidence encoder_ckpt={ckpt[:8]} evidence_files={len(roles)} mismatches=0")
    return {"route": "evidence", "encoder_ckpt": ckpt, "files": len(roles), "mismatches": len(bad)}


def check_dataset(rec, store, contract, history, env=None):
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
    if "motion" not in history:
        return None  # 测时 yaml 无 motion：跳过 motion 表、证据与血缘（计划 S1 契约行）
    bounds = legacy.manifest_bounds(manifest)
    rec("WINDOW_BUDGET", bounds["rows"] == contract.motion_rows and bounds["train_max"] == contract.train_max
        and type(contract.train_max) is int and contract.train_max <= 96)
    motion = ms.MotionMeta.load(parent / contract.store_dir)
    ms.run_fast_checks(motion, manifest_path=parent / "meta/episode_manifest.json")
    ms.run_full_checks(motion)
    ms.check_same_source(manifest["sha256"], motion, manifest)
    rec("MOTION_VERIFIED", motion.status == "verified" and motion.raw["layout"] == ms.STAGE2_LAYOUT
        and motion.num_rows == contract.motion_rows and motion.row_shape == (1, 1024)
        and motion.provenance["encoder"]["checkpoint_sha256"] == contract.encoder_sha256)
    if not ms.stage2_uses_cache(motion.provenance):
        raise ValueError("原版16×100必须使用published-cache-with-local-fill来源合同")
    evidence_root = (env or {}).get(EVIDENCE_ENV)
    if evidence_root:
        files, bad = verify_evidence_root(evidence_root, motion.provenance)
        if bad:
            raise ValueError(f"重定根证据SHA不符: {bad}")
        print(f"EVIDENCE_REBASE=PASS files={files} sha_mismatch=0")
        validate_stage2_evidence(rebase_provenance(motion.provenance, evidence_root), motion.root)
    else:
        validate_stage2_evidence(motion.provenance, motion.root)
    rec("INPUT_EQUIVALENCE", True, "G0/G2/G3/缓存收据逐SHA回读并验证；输出数值按用户裁决不作同值准入")
    lineage = load_script("scripts/training/check_data_lineage.py")
    checkpoint = store / "v1-store/external/motionjepa" / contract.encoder_run / "checkpoint_epoch_105.pt"
    config_sha = sha256_file(checkpoint.parent / "config.yaml")
    expected_sha = ENCODERS.get(contract.encoder_run, {}).get("config_sha256")
    rec("ENCODER_CONFIG_SHA", config_sha == motion.provenance["encoder"]["config_sha256"]
        and (contract.encoder_run == ENCODER_RUN or expected_sha is None or config_sha == expected_sha))
    evidence_dir = parent / contract.evidence_dir
    if not Path(contract.mj_source_manifest).exists() and all(
            (evidence_dir / f"{role}.json").is_file() for role in ms.stage2_evidence_roles(motion.provenance)):
        return check_evidence_lineage(rec, evidence_dir, motion, checkpoint, contract)
    result = lineage.check_lineage(checkpoint, contract.encoder_sha256, parent / "meta/input_manifest.json",
                                   contract.mj_source_manifest)
    rec("ENCODER_DATA_LINEAGE", result["files"] == contract.tasks and result["mismatches"] == 0)
    print(f"DATA_LINEAGE=PASS encoder_ckpt={result['encoder_ckpt'][:8]} encoder_source_pin={result['encoder_source_pin']} vla_input_pin={result['vla_input_pin']} files={result['files']} mismatches=0")
    return result


def core(args, argv, *, contract=Contract(), env=None, history=None):
    """同一只读契约核心；CPU夹具仅由Python调用替换身份值，正式CLI无此选项。"""
    rec = Recorder()
    env = dict(os.environ if env is None else env)
    encoder_run = getattr(args, "encoder_run", None)
    if encoder_run is not None:
        identity = ENCODERS[encoder_run]
        contract = dataclasses.replace(contract, encoder_run=encoder_run, encoder_sha256=identity["ckpt_sha256"],
            store_dir=identity["store_dir"], evidence_dir=identity["evidence_dir"])
    config_name = getattr(args, "config_name", CONFIG_NAME)
    store = Path(args.store_root).resolve()
    fsdp = getattr(args, "fsdp_devices", contract.fsdp_devices)
    batch = getattr(args, "batch_size", contract.batch_size)
    gpus_visible = getattr(args, "gpus_visible", None)
    gl = fsdp in GL_FSDP
    rec("BUILD_PINS", PLACEHOLDER not in str(dataclasses.asdict(contract)), "构建pin必须全部回填")
    rec.guard("ARGV", check_argv, rec, argv, args.steps, gl, config_name)
    rec.guard("ENV", legacy.check_env, rec, env, store, contract, gpus_visible)
    if not gl and (fsdp, batch) != (contract.fsdp_devices, contract.batch_size):
        rec("GL_COMBO", False, f"非GL口径只认fsdp={contract.fsdp_devices} batch={contract.batch_size}")
    if gl:
        count = len(gpus_visible.split(",")) if gpus_visible else None
        if rec("GL_GPU_COUNT", gpus_visible is not None and count in (None, fsdp), f"visible={gpus_visible} fsdp={fsdp}"):
            print(f"GPU_SCOPE=PASS visible={gpus_visible or '<slurm>'} fsdp={fsdp}")
    elif env.get("CUDA_VISIBLE_DEVICES") == contract.gpus:
        print("GPU_SCOPE=PASS allowed=0-7")
    names = legacy.flag_value(argv, "--model.history-config")
    if len(names) != 1 or names[0] not in (HISTORIES + tuple(TIER_HISTORIES) if gl else HISTORIES):
        rec("HISTORY_NAME", False, str(names))
        return rec
    name = names[0]
    if contract.encoder_run in ENCODERS and contract.encoder_run != ENCODER_RUN:
        rec("ENCODER_COMBO", config_name == ABLATION_CONFIG and (fsdp, batch) == (4, 64)
            and args.steps in (20, 100, 80000) and name == ENCODERS[contract.encoder_run]["yaml"],
            f"encoder={contract.encoder_run} config={config_name} fsdp={fsdp} batch={batch} steps={args.steps} history={name}")
    tier = name in TIER_HISTORIES
    if gl:
        runs = legacy.flag_value(argv, "--exp-name")
        rec("GL_COMBO", len(runs) == 1 and gl_combo_ok(fsdp=fsdp, batch=batch, steps=args.steps, run=runs[0],
            history_name=name, encoder_run=contract.encoder_run if contract.encoder_run in ENCODERS else ENCODER_RUN,
            config_name=config_name), f"fsdp={fsdp} batch={batch} steps={args.steps} run={runs} history={name}")
        root = env.get(EVIDENCE_ENV)
        if tier:
            rec("ENV_EVIDENCE_ROOT", not root, "测时无motion，不设证据重定根")
        else:
            rec("ENV_EVIDENCE_ROOT", bool(root) and Path(root).is_absolute() and Path(root).is_dir(),
                f"{EVIDENCE_ENV}={root or '<未设置>'}")
    # 按当前history读法选ctx/modul配对；两份只允许融合方式与记忆宽度不同。测时yaml按正式split配对核。
    fusion_tag, readout_tag = ("ctx", "4x256") if tier else (name.split("-")[2], name.split("-")[-2])
    pairs = [load_yaml(args.repo, pair_name(fusion, readout_tag)) for fusion in ("ctx", "modul")]
    rec("PAIRED_CONFIG", {key for key in pairs[0] if pairs[0].get(key) != pairs[1].get(key)}
        == {"integration_type", "memory_token_dim"} and set(pairs[0]) == set(pairs[1]), f"readout={readout_tag}")
    if rec("NOSPLIT_CONFIG", check_nosplit_config(args.repo, fusion_tag), f"fusion={fusion_tag}") and readout_tag == "1x1024":
        print(f"NOSPLIT_CONFIG=PASS fusion={fusion_tag} diff_keys=motion.readout,motion.num_tokens,motion.dim,motion.budget")
    attempt = None
    if env.get("ATTEMPT_MANIFEST"):
        attempt = rec.guard("ATTEMPT", check_attempt, env["ATTEMPT_MANIFEST"], history_name=name,
                            train_head=getattr(args, "train_head", None))
        if attempt is not None:
            print(f"ATTEMPT=PASS run={attempt}")
    history = load_yaml(args.repo, name) if history is None else history
    if tier:
        rec.guard("HISTORY", check_tier_history, rec, history, name)
    else:
        rec.guard("HISTORY", check_history, rec, history, contract)
    rec("NO_PLACEHOLDER", PLACEHOLDER not in str(history) and PLACEHOLDER not in str(argv))
    if not any(x in rec.failed for x in ("BUILD_PINS", "NO_PLACEHOLDER", "HISTORY")):
        rec.guard("DATASET", check_dataset, rec, store, contract, history, env)
    parsed = rec.guard("CLI", parse_cli, argv, config_name)
    if parsed:
        rec.guard("CONFIG", check_config, rec, *parsed, store=store, contract=contract,
                  steps=args.steps, history_name=name, history=history, attempt=attempt,
                  fsdp_devices=fsdp, batch_size=batch, config_name=config_name)
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
        parser.add_argument("--steps", type=int, choices=(20, 100, 60000, 80000), required=True)
        parser.add_argument("--train-head", required=True)
        args = parser.parse_args(raw)
        try:
            result = check_final_record(args.check_final_record, run=args.run, steps=args.steps, head=args.train_head)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            print(f"TRAIN_FINAL_RECORD=FAIL reason={error}")
            return 1
        print(f"TRAIN_FINAL_RECORD=PASS run={args.run} tail_steps={result['tail_steps']} ema_leaves={result['ema_leaves']}")
        return 0
    if raw and raw[0] == "--verify-evidence-root":
        parser = argparse.ArgumentParser(description="逐文件核证据重定根目录（K11）")
        parser.add_argument("--verify-evidence-root", type=Path, required=True)
        parser.add_argument("--store", type=Path, required=True, help="motion-stage2-1024 表根")
        args = parser.parse_args(raw)
        try:
            from mme_vla_suite.datastore import motion_store as ms
            from mme_vla_suite.policies.motion_client import validate_stage2_evidence
            provenance = ms.MotionMeta.load(args.store).provenance
            files, bad = verify_evidence_root(args.verify_evidence_root, provenance)
            if bad:
                print(f"EVIDENCE_REBASE=FAIL files={files} sha_mismatch={len(bad)} bad={','.join(bad)}")
                return 1
            validate_stage2_evidence(rebase_provenance(provenance, args.verify_evidence_root), args.store)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"EVIDENCE_REBASE=FAIL reason={error}")
            return 1
        print(f"EVIDENCE_REBASE=PASS files={files} sha_mismatch=0")
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
    parser.add_argument("--steps", type=int, choices=(20, 100, 60000, 80000), default=60000)
    parser.add_argument("--encoder-run", choices=tuple(ENCODERS), default=None)
    parser.add_argument("--config-name", choices=(CONFIG_NAME, ABLATION_CONFIG), default=CONFIG_NAME)
    parser.add_argument("--fsdp-devices", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--gpus-visible", default=None, help="GL 入口传入的期望可见卡列表；Slurm 下未设时为空串")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(raw)
    rec = core(args, argv)
    rec.guard("REPO", legacy.check_repo, rec, Path(args.repo), args.train_head)
    # 基础pi05权重仍由已有资产锁核实际文件；夹具机制前向不读该12GB资产。
    rec.guard("PI05", legacy.check_pi05_asset, rec, REPO, Path(args.store_root))
    if not rec.failed:
        print("PAIRED_CONFIG=PASS diff_keys=integration_type,memory_token_dim")
        print(f"PAIRED_RUNS_SAME_LIB=PASS input_pin={Contract().input_manifest_sha256}")
    print(f"LAUNCH_CONTRACT={'FAIL' if rec.failed else 'PASS'} checks={len(rec.rows)} failed={','.join(rec.failed) or 0}")
    if args.fsdp_devices in GL_FSDP:
        extra = f"steps={args.steps} encoder={args.encoder_run} " if args.encoder_run not in (None, ENCODER_RUN) else ""
        print(f"LAUNCH_CONTRACT_GL={'FAIL' if rec.failed else 'PASS'} gpus={args.fsdp_devices} fsdp={args.fsdp_devices} "
              f"batch={args.batch_size} {extra}checks={len(rec.rows)} failed={','.join(rec.failed) or 0}")
    return int(bool(rec.failed))


if __name__ == "__main__":
    raise SystemExit(main())
