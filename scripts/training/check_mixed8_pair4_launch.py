#!/usr/bin/env python
"""四对 modulation 正式 80k（S7）与同配置短测（S6）的专用启动合同检查。

用实际交给 train.py 的参数数组调用真实 ``config.cli()`` 得到解析对象，再逐项核对：
配置名、batch128、FSDP8、workers16、步数、学习率日程、EMA、seed、history YAML
（除三个来源字段外与已验四对冒烟配置逐项相同）、norm 文件 SHA、父 manifest 两个
文件 SHA、新 motion 表元数据（行数、行字节、encoder SHA 与 YAML 相同）、pi05_base
51 叶键/shape/dtype 与缺失集合恰为 14 个记忆叶、``motion_encoder_static`` 为
[1280,1024]、``TRAIN_FINAL_RECORD_DIR`` 为绝对且为空、残留环境变量为空。

不启动训练、不写任何文件；所有检查都执行完后再汇总，末行
``POLICY_LAUNCH_CONTRACT=PASS|FAIL``。

用法（trailing argv 必须与随后交给 train.py 的参数数组逐字相同）::

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/training/check_mixed8_pair4_launch.py \\
        --repo "$MAIN" --store-root "$MAIN" --train-head <40位sha> --steps 80000 -- "${TRAIN_ARGS[@]}"
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

MAIN = Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
CONFIG_NAME = "mme_vla_suite_b128_80k"
HISTORY_CONFIG = "perceptual-framesamp-modul-pair4-v10e60-mixed8.yaml"
SMOKE_CONFIG = "perceptual-framesamp-modul-pair4-epoch6-original-smoke.yaml"
YAML_DIR = "src/mme_vla_suite/models/config/robomme"
SOURCE_FIELDS = ("store_path", "source_run", "checkpoint_sha256")
PLACEHOLDER = "__FILL_AFTER_S5B__"
E6_SHA = "146efd425f7923428d630dfb7ee2a8f010f19b207d36519dba49454aaa373d30"
SMOKE_REFERENCE = "docs/training-doc/paired4-epoch6-original-smoke-20261020-a/records/exported_parameter_reference.json"
FORMAL_STEPS = 80000

# 14 个记忆叶：pi05_base 中不存在、由正常冷初始化生成；其余 51 叶必须从 pi05_base 完整恢复。
MEMORY_LEAVES = frozenset({
    "PaliGemma/llm/layers/mem_attn/kv_einsum_mem/w",
    "PaliGemma/llm/layers/mem_attn/mem_rms_norm/scale",
    "PaliGemma/llm/layers/mem_attn/out_einsum_mem/w",
    "PaliGemma/llm/layers/mem_attn/q_einsum_mem/w",
    "PaliGemma/llm/layers/mem_rms_norm_ffn/Dense_0/bias",
    "PaliGemma/llm/layers/mem_rms_norm_ffn/Dense_0/kernel",
    "mem_encoder/feature_encoder/encoder_static/bias",
    "mem_encoder/feature_encoder/encoder_static/kernel",
    "mem_encoder/feature_encoder/pos_proj/bias",
    "mem_encoder/feature_encoder/pos_proj/kernel",
    "mem_encoder/motion_encoder_static/bias",
    "mem_encoder/motion_encoder_static/kernel",
    "mem_encoder/motion_pos_proj/bias",
    "mem_encoder/motion_pos_proj/kernel",
})
MOTION_PROJ = "mem_encoder/motion_encoder_static/kernel"

# 允许出现在 train.py 参数数组里的选项；其余（含任何 resume/overwrite/resum_ckpt_id）一律拒绝。
VALUE_FLAGS = frozenset({"--exp-name", "--dataset-path", "--assets-base-dir", "--data.assets.assets-dir",
                         "--data.assets.asset-id", "--checkpoint-base-dir", "--weight-loader.params-path",
                         "--model.history-config", "--num-train-steps"})
BOOL_FLAGS = frozenset({"--model.use-history", "--no-wandb-enabled"})


@dataclasses.dataclass(frozen=True)
class Contract:
    """固定身份；测试可替换为合成值，正式入口只用默认值。"""
    parent_rel: str = "v1-store/datasets/16task-mixed8-200-3200ep-e3c7564f"
    assets_rel: str = "v1-store/train-assets/mme_vla_suite/16task-mixed8-200-3200ep-e3c7564f"
    pi05_rel: str = "v1-store/models/openpi-assets/checkpoints/pi05_base/params"
    norm_sha256: str = "36ee9acdea4c8383b50c7b07b45a740efb3fa073f6e60455977d5a6fb3204394"
    episode_manifest_sha256: str = "c6a063dfbfc6da404dd8f1d3ddc38e9ba6e25abca81417fe742872401e7f97df"
    input_manifest_sha256: str = "d7dd651edb8cb9652fa647a73a2d0d3c4be64d6876e1d7a186a5e8c75b02aafb"
    canonical_manifest_sha256: str = "d6a3ddb2b579b5cf50cd66fbefedd36478766e2b4cc07833a7efae29769742ad"
    smoke_reference_sha256: str = "1c175c8a8fecf0e615a0693c872bb1697f0d0b1b262ae1e47bac14a7fb4ddeb0"
    motion_layout: str = "motion-v10-8x256-grid16-demopad17-v1"
    motion_rows: int = 128342
    motion_exec_rows: int = 81611
    motion_demo_rows: int = 46731
    motion_row_bytes: int = 8192
    encoder_epoch: int = 60
    batch_size: int = 128
    fsdp_devices: int = 8
    num_workers: int = 16
    seed: int = 42
    ema_decay: float = 0.999
    gpus: str = "0,1,2,3,4,5,6,7"


class Recorder:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def __call__(self, name, ok, detail=""):
        ok = bool(ok)
        self.rows.append((name, ok, str(detail)))
        print(f"CHECK_{name}={'PASS' if ok else 'FAIL'} {detail}", flush=True)
        return ok

    def guard(self, name, func, *args, **kwargs):
        """执行一组检查；异常记为该组 FAIL，不中断其余检查。"""
        try:
            return func(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 -- 记账后继续，末行统一判定
            self(name, False, f"异常 {type(error).__name__}: {error}")
            return None

    @property
    def failed(self):
        return [name for name, ok, _ in self.rows if not ok]


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def flag_value(argv, flag):
    values = [argv[i + 1] for i, token in enumerate(argv[:-1]) if token == flag]
    values += [token[len(flag) + 1:] for token in argv if token.startswith(flag + "=")]
    return values


# ── 参数数组 ────────────────────────────────────────────────────────────────


def check_argv(rec, argv, *, steps):
    rec("ARGV_CONFIG_NAME", bool(argv) and argv[0] == CONFIG_NAME, f"期望首参={CONFIG_NAME} 实际={argv[:1]}")
    bad = [t for t in argv if re.search(r"resum|overwrite", t, re.IGNORECASE)]
    rec("ARGV_NO_RESUME", not bad, f"禁止resume/overwrite/resum_ckpt_id 实际={bad}")
    unknown, i = [], 1
    while i < len(argv):
        token = argv[i]
        name = token.split("=", 1)[0]
        if name in VALUE_FLAGS:
            i += 1 if "=" in token else 2
        elif token in BOOL_FLAGS:
            i += 1
        else:
            unknown.append(token)
            i += 1
    rec("ARGV_ALLOWLIST", not unknown, f"不允许的参数={unknown}")
    dup = sorted(f for f in VALUE_FLAGS if len(flag_value(argv, f)) > 1)
    rec("ARGV_NO_DUPLICATE", not dup, f"重复的参数={dup}")
    got_steps = flag_value(argv, "--num-train-steps")
    if steps == FORMAL_STEPS:
        rec("ARGV_STEPS", not got_steps, f"正式80k不得传--num-train-steps 实际={got_steps}")
    else:
        rec("ARGV_STEPS", got_steps == [str(steps)] and 0 < steps < FORMAL_STEPS,
            f"短测必须恰传--num-train-steps {steps} 实际={got_steps}")
    rec("ARGV_USE_HISTORY", argv.count("--model.use-history") == 1, "须恰有一次 --model.use-history")


def parse_cli(argv):
    """在真实 tyro 入口上解析（train.py 的 ``_config.cli()`` 读取 sys.argv[1:]）。"""
    import mme_vla_suite.training.config as config_module
    previous = sys.argv
    sys.argv = ["train.py", *argv]
    try:
        return config_module.cli(), config_module.get_config(CONFIG_NAME)
    finally:
        sys.argv = previous


def check_config(rec, cfg, base, *, store, contract, steps):
    import mme_vla_suite.training.config as config_module
    rec("CFG_NAME", cfg.name == CONFIG_NAME, f"期望={CONFIG_NAME} 实际={cfg.name}")
    for key in ("batch_size", "fsdp_devices", "num_workers", "seed", "ema_decay"):
        want = getattr(contract, key)
        rec(f"CFG_{key.upper()}", getattr(cfg, key) == want == getattr(base, key),
            f"期望={want} 实际={getattr(cfg, key)} 基础配置={getattr(base, key)}")
    rec("CFG_STEPS", cfg.num_train_steps == steps, f"期望={steps} 实际={cfg.num_train_steps}")
    rec("CFG_LR_SCHEDULE", cfg.lr_schedule == base.lr_schedule
        and (cfg.lr_schedule.warmup_steps, cfg.lr_schedule.peak_lr, cfg.lr_schedule.decay_steps,
             cfg.lr_schedule.decay_lr) == (5000, 5e-5, FORMAL_STEPS, 5e-5),
        f"实际={cfg.lr_schedule}（短测也不缩减日程）")
    rec("CFG_OPTIMIZER", cfg.optimizer == base.optimizer, f"实际={cfg.optimizer}")
    rec("CFG_SAVE", (cfg.save_interval, cfg.keep_period, cfg.log_interval) == (5000, 5000, 100)
        == (base.save_interval, base.keep_period, base.log_interval),
        f"save/keep/log 实际={(cfg.save_interval, cfg.keep_period, cfg.log_interval)}")
    rec("CFG_NO_RESUME", not cfg.resume and not cfg.overwrite and cfg.resum_ckpt_id is None
        and cfg.pytorch_weight_path is None,
        f"resume={cfg.resume} overwrite={cfg.overwrite} resum_ckpt_id={cfg.resum_ckpt_id}")
    want_model = dataclasses.replace(base.model, history_config=HISTORY_CONFIG)
    rec("CFG_MODEL", cfg.model == want_model and cfg.model.pi05 and cfg.model.use_history
        and cfg.model.action_horizon == 20 and not cfg.model.discrete_state_input,
        f"history_config={cfg.model.history_config} action_horizon={cfg.model.action_horizon}")
    rec("CFG_FREEZE_FILTER", repr(cfg.freeze_filter) == repr(base.freeze_filter), "冻结范围与基础配置相同")
    assets = str(store / contract.assets_rel)
    rec("CFG_ASSETS", cfg.data.assets.assets_dir == assets and cfg.data.assets.asset_id == "robomme"
        and cfg.data.repo_id == base.data.repo_id,
        f"期望={assets}/robomme 实际={cfg.data.assets}（缺省会静默落到旧4task norm）")
    rec("CFG_ASSETS_BASE", Path(cfg.assets_base_dir) == store / "v1-store/train-assets",
        f"实际={cfg.assets_base_dir}")
    rec("CFG_DATASET_PATH", Path(cfg.dataset_path) == store / contract.parent_rel / "framesamp-8x8",
        f"实际={cfg.dataset_path}")
    pi05 = str(store / contract.pi05_rel)
    rec("CFG_INIT_WEIGHTS", type(cfg.weight_loader).__name__ == "CheckpointWeightLoader"
        and cfg.weight_loader.params_path == pi05, f"期望={pi05} 实际={cfg.weight_loader}")
    rec("CFG_OPENPI_DATA_HOME", config_module.OPENPI_DATA_HOME == str(store / "v1-store/models"),
        f"导入时 OPENPI_DATA_HOME={config_module.OPENPI_DATA_HOME}")
    run_root = Path(cfg.checkpoint_dir)
    rec("CFG_RUN_ROOT", run_root.parent == store / "v1-store/train-runs" / CONFIG_NAME
        and not run_root.exists() and not run_root.is_symlink(),
        f"新run目录须不存在 实际={run_root} exists={run_root.exists()}")


# ── history YAML ────────────────────────────────────────────────────────────


def load_yaml(repo, name):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(Path(repo) / YAML_DIR / name), resolve=True)


def check_history(rec, formal, smoke):
    """除三个来源字段外逐项相同；三个字段必须是真实正式值。返回 motion 来源字段。"""
    strip = lambda d: {**d, "motion": {k: v for k, v in d["motion"].items() if k not in SOURCE_FIELDS}}
    rec("YAML_EQUALS_SMOKE", strip(formal) == strip(smoke), "除store_path/source_run/checkpoint_sha256外与四对冒烟配置逐项相同")
    motion = formal.get("motion", {})
    fields = {k: motion.get(k) for k in SOURCE_FIELDS}
    placeholders = [k for k, v in fields.items() if not isinstance(v, str) or PLACEHOLDER in v or "__FILL" in v]
    rec("YAML_NO_PLACEHOLDER", not placeholders, f"仍为占位或缺失={placeholders}")
    sha = str(fields["checkpoint_sha256"])
    rec("YAML_SHA_FORMAT", re.fullmatch(r"[0-9a-f]{64}", sha) is not None, f"checkpoint_sha256={sha}")
    rec("YAML_NOT_E6", sha != E6_SHA and fields["source_run"] != smoke["motion"]["source_run"],
        f"禁止沿用e6冒烟身份 source_run={fields['source_run']}")
    store_path = str(fields["store_path"])
    smoke_like = ("v1-store/smoke" in store_path or "fixture" in store_path
                  or store_path == smoke["motion"]["store_path"])
    rec("YAML_NOT_SMOKE_FIXTURE", not smoke_like, f"store_path={store_path}")
    run = str(fields["source_run"])
    match = re.fullmatch(r"([^/#]+)/checkpoint_epoch_(\d+)\.pt#encoder", run)
    rec("YAML_SOURCE_RUN", match is not None and int(match[2]) == Contract.encoder_epoch,
        f"须为 <run>/checkpoint_epoch_60.pt#encoder 实际={run}")
    return fields, match


# ── 数据、norm 与新 motion 表 ───────────────────────────────────────────────


def check_parent(rec, store, contract):
    parent = store / contract.parent_rel
    norm = store / contract.assets_rel / "robomme/norm_stats.json"
    got = sha256_file(norm) if norm.is_file() else "<缺文件>"
    rec("NORM_SHA256", got == contract.norm_sha256, f"期望={contract.norm_sha256} 实际={got} 路径={norm}")
    for key, name in (("episode_manifest_sha256", "episode_manifest.json"),
                      ("input_manifest_sha256", "input_manifest.json")):
        path = parent / "meta" / name
        got = sha256_file(path) if path.is_file() else "<缺文件>"
        rec(f"PARENT_{key.upper()}", got == getattr(contract, key), f"期望={getattr(contract, key)} 实际={got}")
    meta = json.loads((parent / "framesamp-8x8/meta/store_meta.json").read_text())
    rec("PARENT_FRAMESAMP", meta.get("status") == "verified" and meta.get("manifest_scope") == "full"
        and meta.get("manifest_sha256") == contract.canonical_manifest_sha256,
        f"status={meta.get('status')} scope={meta.get('manifest_scope')} manifest={meta.get('manifest_sha256')}")
    rec("PARENT_SOURCE_DIR", (parent / "source/data").is_dir(), f"{parent}/source/data")


def check_motion(rec, repo, store, fields, match, contract):
    """新库 motion 元数据；相对 store_path 按 v1-store 所在仓库根解析。

    dataloader 相对代码仓库根解析；正式入口 REPO_CWD/REPO_PACKAGE 已要求代码根即主检出，
    两者在正式起跑时相同。按 store 根解析使 worktree 演练也能核对真实新库。
    """
    raw = str(fields["store_path"])
    root = Path(raw) if Path(raw).is_absolute() else Path(store) / raw
    root = root.resolve()
    parent_motion = (store / contract.parent_rel / "motion").resolve()
    rec("MOTION_PATH", root.is_dir() and root != parent_motion
        and root.is_relative_to((store / "v1-store/datasets").resolve()),
        f"须为 v1-store/datasets 下的新库且不是父库旧768表 实际={root}")
    meta = json.loads((root / "meta/store_meta.json").read_text())
    table = meta.get("tables", {}).get("motion_token", {})
    rec("MOTION_LAYOUT", meta.get("layout") == contract.motion_layout and meta.get("status") == "verified",
        f"layout={meta.get('layout')} status={meta.get('status')}")
    rec("MOTION_ROWS", meta.get("num_rows") == table.get("num_rows") == contract.motion_rows
        and meta.get("totals", {}).get("exec_rows") == contract.motion_exec_rows
        and meta.get("totals", {}).get("demo_rows") == contract.motion_demo_rows,
        f"num_rows={meta.get('num_rows')} totals={meta.get('totals')}")
    expected_bytes = contract.motion_rows * contract.motion_row_bytes
    body = root / str(table.get("relpath", ""))
    rec("MOTION_ROW_BYTES", table.get("row_bytes") == contract.motion_row_bytes and table.get("row_shape") == [8, 256]
        and table.get("dtype") == "float32" and table.get("byte_count") == expected_bytes
        and body.is_file() and body.stat().st_size == expected_bytes,
        f"row_bytes={table.get('row_bytes')} row_shape={table.get('row_shape')} byte_count={table.get('byte_count')}")
    rec("MOTION_MANIFEST", meta.get("manifest_sha256") == contract.canonical_manifest_sha256,
        f"实际={meta.get('manifest_sha256')}")
    encoder = meta.get("provenance", {}).get("encoder", {})
    rec("MOTION_ENCODER_SHA", encoder.get("checkpoint_sha256") == fields["checkpoint_sha256"]
        and encoder.get("checkpoint_sha256") != E6_SHA,
        f"store={encoder.get('checkpoint_sha256')} yaml={fields['checkpoint_sha256']}")
    want = None if match is None else {"run_name": match[1], "checkpoint_name": f"checkpoint_epoch_{match[2]}.pt",
                                       "epoch": int(match[2]), "state_key": "encoder"}
    got = {k: encoder.get(k) for k in ("run_name", "checkpoint_name", "epoch", "state_key")}
    rec("MOTION_SOURCE_RUN", want is not None and got == want, f"store={got} yaml={want}")


# ── 环境与记录目录 ──────────────────────────────────────────────────────────


def check_env(rec, env, store, contract):
    rec("ENV_MOTION_STORE_UNSET", "MMEVLA_MOTION_STORE" not in env, f"MMEVLA_MOTION_STORE={env.get('MMEVLA_MOTION_STORE', '<未设置>')}")
    leftovers = sorted(k for k in env if k.endswith("_ALLOW_UNVERIFIED") or k == "MMEVLA_FRAMESAMP_ALLOW_SUBSET")
    rec("ENV_NO_ALLOW_OVERRIDES", not leftovers, f"残留放行变量={leftovers}")
    rec("ENV_VERIFY_LEVEL", env.get("MMEVLA_FRAMESAMP_VERIFY", "") in ("", "fast"),
        f"MMEVLA_FRAMESAMP_VERIFY={env.get('MMEVLA_FRAMESAMP_VERIFY', '<未设置>')}")
    rec("ENV_PYTHONPATH_UNSET", not env.get("PYTHONPATH"), f"PYTHONPATH={env.get('PYTHONPATH', '<未设置>')}")
    rec("ENV_OPENPI_DATA_HOME", env.get("OPENPI_DATA_HOME") == str(store / "v1-store/models"),
        f"OPENPI_DATA_HOME={env.get('OPENPI_DATA_HOME')}")
    parent = store / contract.parent_rel
    rec("ENV_FRAMESAMP_SOURCE", env.get("MMEVLA_FRAMESAMP_SOURCE") == str(parent / "source"),
        f"实际={env.get('MMEVLA_FRAMESAMP_SOURCE')}")
    rec("ENV_FRAMESAMP_MANIFEST", env.get("MMEVLA_FRAMESAMP_MANIFEST") == str(parent / "meta/episode_manifest.json"),
        f"实际={env.get('MMEVLA_FRAMESAMP_MANIFEST')}")
    rec("ENV_GPUS", env.get("CUDA_VISIBLE_DEVICES") == contract.gpus,
        f"单JAX进程8卡 CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
    final = env.get("TRAIN_FINAL_RECORD_DIR", "")
    path = Path(final)
    empty = path.is_absolute() and (not path.exists() or (path.is_dir() and not path.is_symlink() and not any(path.iterdir())))
    rec("ENV_FINAL_RECORD_DIR", bool(final) and path.is_absolute() and path.name == "final-record"
        and path.resolve().is_relative_to((store / "v1-store").resolve()) and empty,
        f"须为 v1-store 内绝对、名为final-record、不存在或为空 实际={final or '<未设置>'}")
    record = env.get("TRAIN_RECORD_DIR", "")
    rec("ENV_TRAIN_RECORD_DIR", bool(record) and Path(record).is_absolute() and bool(final)
        and Path(record) == path.parent / "train-record" and not (Path(record) / "metrics.jsonl").exists(),
        f"须为 final-record 同级 train-record 且无 metrics.jsonl 实际={record or '<未设置>'}")


def check_repo(rec, repo, head):
    cwd = Path.cwd().resolve()
    rec("REPO_CWD", cwd == Path(repo).resolve(), f"期望={repo} 实际={cwd}")
    import importlib.util
    spec = importlib.util.find_spec("mme_vla_suite")
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    rec("REPO_PACKAGE", origin is not None and origin.is_relative_to(Path(repo).resolve()), f"mme_vla_suite={origin}")
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=False)
    got = git("rev-parse", "HEAD").stdout.strip()
    rec("REPO_HEAD", re.fullmatch(r"[0-9a-f]{40}", head or "") is not None and got == head, f"期望={head} 实际={got}")
    porcelain = git("status", "--porcelain").stdout.strip()
    rec("REPO_CLEAN", porcelain == "", f"porcelain={porcelain.replace(chr(10), ' | ') or '空'}")


# ── 初始权重与参数树 ────────────────────────────────────────────────────────


def model_leaves(cfg, history):
    """与 train.py::init_train_state 同口径：真实模型抽象初始化后把冻结参数转 BF16。"""
    import jax
    import jax.numpy as jnp
    from flax import nnx, traverse_util
    import openpi.shared.nnx_utils as nnx_utils
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, history_config=history))

    def init(rng):
        params = nnx.state(cfg.model.create(rng))
        return nnx_utils.state_map(params, cfg.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

    shape = jax.eval_shape(init, jax.random.key(0)).to_pure_dict()
    return {"/".join(k): (list(v.shape), str(v.dtype)) for k, v in traverse_util.flatten_dict(shape).items()}


def reference_leaves(repo, contract):
    path = Path(repo) / SMOKE_REFERENCE
    if sha256_file(path) != contract.smoke_reference_sha256:
        raise ValueError(f"冒烟65叶参考文件SHA不符: {path}")
    rows = json.loads(path.read_text())["leaves"]
    convert = lambda key: "/".join(re.findall(r"\['([^']+)'\]", key))
    return {convert(k): (list(v["shape"]), v["dtype"]) for k, v in rows.items()}


def base_leaves(params_path):
    import orbax.checkpoint as ocp
    from flax import traverse_util
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(str(params_path))
    flat = traverse_util.flatten_dict(dict(metadata["params"]))
    return {"/".join(k): (list(v.shape), str(v.dtype)) for k, v in flat.items()}


def check_pi05_asset(rec, repo, store):
    sys.path.insert(0, str(Path(repo) / "scripts/assets"))
    import assets_lock
    fails = assets_lock.verify(assets_lock.load_lock(Path(repo) / "scripts/assets/ASSETS_LOCK.json"),
                               store, level="cheap", names=["pi05_base"])
    rec("PI05_ASSET_LOCK", not fails, f"ASSETS_LOCK cheap pi05_base 失败={fails}")


def check_leaves(rec, model, reference, base):
    rec("MODEL_65_LEAVES", len(model) == 65 and model == reference,
        f"模型叶数={len(model)} 与冒烟65叶参考键/shape/dtype相同={model == reference}")
    rec("MOTION_PROJ_SHAPE", model.get(MOTION_PROJ, ([], ""))[0] == [1280, 1024], f"{MOTION_PROJ}={model.get(MOTION_PROJ)}")
    rec("PI05_51_LEAVES", len(base) == 51 and all(dtype == "float32" for _, dtype in base.values()),
        f"pi05_base叶数={len(base)} dtype={sorted({d for _, d in base.values()})}")
    shape_bad = sorted(k for k in base if k not in model or model[k][0] != base[k][0])
    rec("PI05_SHAPES", not shape_bad, f"键或shape不符={shape_bad[:5]}")
    missing = set(model) - set(base)
    rec("PI05_MISSING_EQ_MEMORY", missing == MEMORY_LEAVES,
        f"缺失集合须恰为14个记忆叶 多={sorted(missing - MEMORY_LEAVES)[:5]} 少={sorted(MEMORY_LEAVES - missing)[:5]}")


# ── 主流程 ──────────────────────────────────────────────────────────────────


def run(args, argv, env=None, contract=Contract()):
    env = dict(os.environ if env is None else env)
    rec = Recorder()
    repo, store = Path(args.repo).resolve(), Path(args.store_root).resolve()
    rec.guard("ARGV", check_argv, rec, argv, steps=args.steps)
    rec.guard("ENV", check_env, rec, env, store, contract)
    if not args.skip_repo_state:
        rec.guard("REPO", check_repo, rec, repo, args.train_head)
    rec.guard("PARENT", check_parent, rec, store, contract)
    history = rec.guard("YAML", lambda: (load_yaml(repo, HISTORY_CONFIG), load_yaml(repo, SMOKE_CONFIG)))
    fields = match = None
    if history is not None:
        result = rec.guard("YAML", check_history, rec, *history)
        if result is not None:
            fields, match = result
    if fields is not None and not any(PLACEHOLDER in str(v) for v in fields.values()):
        rec.guard("MOTION", check_motion, rec, repo, store, fields, match, contract)
    else:
        rec("MOTION_STORE", False, "YAML来源字段仍为占位，无法核对新motion表")
    if env.get("OPENPI_DATA_HOME") != str(store / "v1-store/models"):
        rec("CLI_PARSE", False, "OPENPI_DATA_HOME 未固定，拒绝导入配置")
        parsed = None
    else:
        os.environ["OPENPI_DATA_HOME"] = env["OPENPI_DATA_HOME"]
        parsed = rec.guard("CLI_PARSE", parse_cli, argv)
    if parsed is not None:
        cfg, base = parsed
        rec("CLI_PARSE", True, f"真实 config.cli() 解析 exp_name={cfg.exp_name}")
        rec.guard("CFG", check_config, rec, cfg, base, store=store, contract=contract, steps=args.steps)
        rec.guard("PI05", check_pi05_asset, rec, repo, store)
        if history is not None and not args.skip_model:
            from omegaconf import OmegaConf
            leaves = rec.guard("LEAVES", lambda: (
                model_leaves(cfg, OmegaConf.create(history[0])), reference_leaves(repo, contract),
                base_leaves(cfg.weight_loader.params_path)))
            if leaves is not None:
                rec.guard("LEAVES", check_leaves, rec, *leaves)
    failed = rec.failed
    verdict = "FAIL" if failed else "PASS"
    print(f"POLICY_LAUNCH_CONTRACT={verdict} checks={len(rec.rows)} failed={','.join(failed) or 0}", flush=True)
    return rec


def main(raw=None):
    raw = list(sys.argv[1:] if raw is None else raw)
    argv = []
    if "--" in raw:
        index = raw.index("--")
        raw, argv = raw[:index], raw[index + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(MAIN), help="代码仓库根；正式起跑必须是主检出")
    parser.add_argument("--store-root", default=str(MAIN), help="v1-store 所在仓库根")
    parser.add_argument("--train-head", default="")
    parser.add_argument("--steps", type=int, default=FORMAL_STEPS)
    parser.add_argument("--skip-repo-state", action="store_true", help="仅供离线测试；正式入口不传")
    parser.add_argument("--skip-model", action="store_true", help="仅供离线测试；正式入口不传")
    args = parser.parse_args(raw)
    return 1 if run(args, argv).failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
