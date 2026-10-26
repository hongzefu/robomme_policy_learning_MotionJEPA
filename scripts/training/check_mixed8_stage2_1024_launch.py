#!/usr/bin/env python
"""stage2 1024 的实际训练 argv、数据身份和完整模型树启动合同。

只做 CPU 抽象模型初始化与只读元数据验证；不启动训练、不写产物。
正式入口不提供跳过检查选项，全部成功才输出 LAUNCH_CONTRACT=PASS。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

MAIN = Path("/scratch/hongze/robomme_policy_learning_MotionJEPA")
CONFIG_NAME = "mme_vla_suite_b128_80k"
HISTORY_CONFIG = "perceptual-framesamp-modul-8frame-8x8-motion-stage2-mixed8.yaml"
FRAME_CONFIG = "perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml"
YAML_DIR = "src/mme_vla_suite/models/config/robomme"
SOURCE_FIELDS = ("store_path", "source_run", "checkpoint_sha256")
PLACEHOLDER = "__FILL_AFTER_G3__"
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
    motion_layout: str = "motion-stage2-1x1024-grid32-demopad17-v1"
    motion_rows: int = 65419
    motion_row_bytes: int = 4096
    train_max: int = 62
    eval_max: int = 79
    eval_steps: int = 1300
    encoder_epoch: int = 105
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
        except (Exception, SystemExit) as error:  # noqa: BLE001 -- CLI 拒绝也记账，末行统一判定
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


def load_script(repo, relative):
    """按仓库内实际文件导入脚本，避免同名脚本或安装包抢占。"""
    path = Path(repo) / relative
    spec = importlib.util.spec_from_file_location(f"_stage2_launch_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        rec("ARGV_STEPS", got_steps == [str(steps)] and steps == 20,
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
    rec("CFG_RUN_NAME", cfg.exp_name == "mixed8-200-m8x8-modul-motion-stage2-1024-b128-80k"
        if steps == FORMAL_STEPS else cfg.exp_name.startswith("cc-vla-s2-smoke-"))
    run_root = Path(cfg.checkpoint_dir)
    rec("CFG_RUN_ROOT", run_root.parent == store / "v1-store/train-runs" / CONFIG_NAME
        and not run_root.exists() and not run_root.is_symlink(),
        f"新run目录须不存在 实际={run_root} exists={run_root.exists()}")


# ── history YAML ────────────────────────────────────────────────────────────


def load_yaml(repo, name):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(Path(repo) / YAML_DIR / name), resolve=True)


STAGE2_RUN = "stage2-full144-1025-a"
ARCH = "wan-latent-stage2-blockwise-pool4-three-decoder"
SOURCE_RUN = f"{STAGE2_RUN}/checkpoint_epoch_105.pt#model"
MOTION_FIXED = {
    "enabled": True, "source_shape": [1, 1024], "readout": "stage2_pool4_concat_v1",
    "num_tokens": 1, "dim": 1024, "budget": 96, "pos_dim": 256,
    "demo_min_real_frames": 17, "demo_tail_pad": "repeat_last", "stride": 32,
    "window_frames": 33, "window_direction": "forward", "grid_origin": "segment_start",
    "frame_size": 256, "store_path": Contract.parent_rel + "/motion-stage2-1024",
    "source_run": SOURCE_RUN, "encoder_kind": "stage2_pool4_concat", "online_gpu": 4,
}


def check_history(rec, formal, frame):
    rec("YAML_FRAME_UNCHANGED", {k: v for k, v in formal.items() if k != "motion"}
        == {k: v for k, v in frame.items() if k != "motion"}, "帧路完整结构与旧8帧配置相同")
    motion = formal.get("motion", {})
    rec("YAML_MOTION", {k: v for k, v in motion.items() if k != "checkpoint_sha256"}
        == MOTION_FIXED, "stage2全部配置键必须精确匹配")
    sha = motion.get("checkpoint_sha256", "")
    rec("YAML_SHA_FORMAT", isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None,
        "checkpoint_sha256须为G3已核验值")
    return {k: motion.get(k) for k in SOURCE_FIELDS}


def check_frame_bytes(rec, repo):
    """帧路不仅语义相同，也保留旧 YAML 的注释、键序与空白。"""
    paths = [Path(repo) / YAML_DIR / name for name in (HISTORY_CONFIG, FRAME_CONFIG)]
    prefixes = [path.read_bytes().split(b"\nmotion:\n", 1)[0] for path in paths]
    rec("YAML_FRAME_BYTES", prefixes[0] == prefixes[1], "motion 节之前逐字节相同")


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


def manifest_bounds(manifest, *, eval_steps=1300):
    """独立复算 stride32、demo 最少17帧、exec 完整33帧的可见窗数。"""
    counts, total, train_max, demo_max = {}, 0, 0, 0
    for episode in manifest["episodes"]:
        g, length, demo = (episode[k] for k in ("global_episode_idx", "num_timesteps", "exec_start_idx"))
        if any(type(v) is not int for v in (g, length, demo)) or not 0 <= demo < length or g in counts:
            raise ValueError(f"manifest episode 身份或长度非法: {episode}")
        # 最后一个 exec 样本看到全部合法窗；任一更早时刻只可能更少。
        nd = len(range(0, max(0, demo - 16), 32))
        ne = len(range(0, max(0, length - demo - 32), 32))
        counts[g] = nd + ne
        total += nd + ne
        train_max = max(train_max, nd + ne)
        demo_max = max(demo_max, nd)
    if not counts:
        raise ValueError("manifest 没有 episode")
    return {"counts": counts, "rows": total, "train_max": train_max,
            "eval_max": demo_max + len(range(0, max(0, eval_steps - 32), 32))}


def check_budget(rec, manifest, contract, entries=None):
    """精确公式与实际 loader 的可见窗枚举必须逐 episode 相等。"""
    bounds = manifest_bounds(manifest, eval_steps=contract.eval_steps)
    rec("BUDGET_ROWS", bounds["rows"] == contract.motion_rows, f"rows={bounds['rows']}")
    rec("BUDGET_TRAIN", bounds["train_max"] == contract.train_max <= MOTION_FIXED["budget"],
        f"train_max={bounds['train_max']} budget={MOTION_FIXED['budget']}")
    rec("BUDGET_EVAL", bounds["eval_max"] == contract.eval_max <= MOTION_FIXED["budget"],
        f"eval_max={bounds['eval_max']} eval_steps={contract.eval_steps} budget={MOTION_FIXED['budget']}")
    if entries is not None:
        from mme_vla_suite.datastore.motion_store import max_visible_count
        actual = {entry.g: max_visible_count(entry) for entry in entries}
        rec("BUDGET_LOADER", len(actual) == len(entries) and actual == bounds["counts"],
            "loader 末帧可见窗数与 manifest 独立公式逐 episode 比较")
        ok = not any(name.startswith("BUDGET_") for name in rec.failed)
        print(f"TRAIN_ES_BOUND={'PASS' if ok else 'FAIL'} train_max={bounds['train_max']} "
              f"eval_max={bounds['eval_max']} budget={MOTION_FIXED['budget']}", flush=True)
    return bounds


def check_motion(rec, repo, store, fields, contract):
    """验证实际库、来源、完整表摘要和注册的ep105实体副本。"""
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.datastore.manifest import load_manifest
    root = store / contract.parent_rel / "motion-stage2-1024"
    rec("MOTION_PATH", fields["store_path"] == contract.parent_rel + "/motion-stage2-1024"
        and root.is_dir() and not root.is_symlink(), str(root))
    meta = json.loads((root / "meta/store_meta.json").read_text())
    loaded = ms.MotionMeta.load(root)
    table = meta["tables"]["motion_token"]
    rec("MOTION_LAYOUT", meta.get("layout") == contract.motion_layout and meta.get("status") == "verified"
        and loaded.spec.grid_stride == 32, "已verified的stage2 grid32库")
    rec("MOTION_ROWS", meta.get("num_rows") == table.get("num_rows") == contract.motion_rows,
        f"期望65419，实际{meta.get('num_rows')}")
    body = root / table["relpath"]
    rec("MOTION_TABLE", table.get("row_bytes") == 4096 and table.get("row_shape") == [1, 1024]
        and table.get("dtype") == "float32" and table.get("byte_count") == 267956224
        and body.stat().st_size == 267956224 and sha256_file(body) == loaded.table_sha256,
        "逐字节摘要与元数据相同")
    rec("MOTION_MANIFEST", meta.get("manifest_sha256") == contract.canonical_manifest_sha256)
    provenance = meta["provenance"]
    ms.validate_stage2_provenance(provenance)
    encoder = provenance["encoder"]
    want = {"run_name": STAGE2_RUN, "checkpoint_name": "checkpoint_epoch_105.pt", "epoch": 105,
            "state_key": "model", "weight_type": "ema", "arch": ARCH,
            "checkpoint_sha256": fields["checkpoint_sha256"], "source_times": [0, 3, 5, 8],
            "output_shape": [1, 1024]}
    rec("MOTION_ENCODER", all(encoder.get(k) == v for k, v in want.items()), "ep105/model/EMA身份")
    registered = store / "v1-store/external/motionjepa" / STAGE2_RUN / "checkpoint_epoch_105.pt"
    rec("ENCODER_REGISTERED", registered.is_file() and not registered.is_symlink()
        and sha256_file(registered) == fields["checkpoint_sha256"], "注册件SHA与yaml及库一致")
    config_file = registered.parent / "config.yaml"
    rec("ENCODER_CONFIG_REGISTERED", config_file.is_file() and not config_file.is_symlink()
        and sha256_file(config_file) == encoder["config_sha256"], "注册配置字节与来源一致")
    pack = load_script(repo, "scripts/dataset/pack_motion_store.py")
    certification = pack.check_bitexact_record(root, meta)
    rec("MOTION_PROVENANCE", True, f"六组来源与CPU全表完整性已核验；encoder_scope={certification['scope']} "
        f"encoder_compared={certification['compared']} total_rows={loaded.num_rows}")
    manifest_path = store / contract.parent_rel / "meta/episode_manifest.json"
    manifest = load_manifest(manifest_path)
    ms.run_fast_checks(loaded, manifest_path=manifest_path)
    ms.check_same_source(contract.canonical_manifest_sha256, loaded, manifest)
    check_budget(rec, manifest, contract, loaded.entries)


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
    rec("ENV_PREALLOCATE", env.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false")
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
    if history is None:
        # 基础 pi05 用其真实模型类；HistoryPi0Config.inputs_spec 即使关闭 history 仍读取帧路字段。
        from openpi.models.pi0_config import Pi0Config
        model = Pi0Config(**{field.name: getattr(cfg.model, field.name) for field in dataclasses.fields(Pi0Config)})
    else:
        model = dataclasses.replace(cfg.model, history_config=history)
    cfg = dataclasses.replace(cfg, model=model)

    def init(rng):
        params = nnx.state(cfg.model.create(rng))
        return nnx_utils.state_map(params, cfg.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

    shape = jax.eval_shape(init, jax.random.key(0)).to_pure_dict()
    return {"/".join(k): (list(v.shape), str(v.dtype)) for k, v in traverse_util.flatten_dict(shape).items()}


def base_leaves(params_path):
    import orbax.checkpoint as ocp
    from flax import traverse_util
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(str(params_path))
    flat = traverse_util.flatten_dict(dict(metadata["params"]))
    return {"/".join(k): (list(v.shape), str(v.dtype)) for k, v in flat.items()}


def check_pi05_asset(rec, repo, store):
    assets_lock = load_script(repo, "scripts/assets/assets_lock.py")
    fails = assets_lock.verify(assets_lock.load_lock(Path(repo) / "scripts/assets/ASSETS_LOCK.json"),
                               store, level="cheap", names=["pi05_base"])
    rec("PI05_ASSET_LOCK", not fails, f"ASSETS_LOCK cheap pi05_base 失败={fails}")


def check_leaves(rec, model, reference, base, base_model):
    """新配置的真实模型树作参考；基础模型另行构造，允许冻结转BF16但不忽略dtype。"""
    rec("MODEL_65_LEAVES", len(model) == 65 and model == reference,
        f"模型键/shape/dtype与stage2独立参考相同，叶数={len(model)}")
    rec("MOTION_PROJ_SHAPE", model.get(MOTION_PROJ) == ([1792, 1024], "float32"),
        str(model.get(MOTION_PROJ)))
    rec("PI05_51_LEAVES", len(base) == 51 and all(dtype == "float32" for _, dtype in base.values()))
    rec("PI05_NO_EXTRA", not (set(base) - set(model)), "不允许基础权重中的未知叶被静默丢弃")
    shape_bad = sorted(k for k in base if k not in model or model[k][0] != base[k][0])
    rec("PI05_SHAPES", not shape_bad, str(shape_bad[:5]))
    missing = set(model) - set(base)
    rec("PI05_MISSING_EQ_MEMORY", missing == MEMORY_LEAVES, str(sorted(missing ^ MEMORY_LEAVES)))
    rec("PI05_MODEL_TREE", set(base_model) == set(base)
        and all(base_model[k][0] == base[k][0] for k in base_model), "真实无memory模型与51叶基础权重吻合")
    rec("PI05_TARGET_DTYPES", all(model.get(k) == v for k, v in base_model.items()),
        "所有基础叶的训练目标dtype与无memory模型相同")
    rec("MEMORY_DTYPES", all(model.get(k, (None, None))[1] == "float32" for k in MEMORY_LEAVES))


# ── 主流程 ──────────────────────────────────────────────────────────────────


def run(args, argv, env=None, contract=Contract()):
    env = dict(os.environ if env is None else env)
    rec = Recorder()
    repo, store = Path(args.repo).resolve(), Path(args.store_root).resolve()
    rec.guard("ARGV", check_argv, rec, argv, steps=args.steps)
    rec.guard("ENV", check_env, rec, env, store, contract)
    rec.guard("REPO", check_repo, rec, repo, args.train_head)
    rec.guard("PARENT", check_parent, rec, store, contract)
    from mme_vla_suite.datastore.manifest import load_manifest
    rec.guard("BUDGET", lambda: check_budget(rec,
        load_manifest(store / contract.parent_rel / "meta/episode_manifest.json"), contract))
    rec.guard("YAML_FRAME_BYTES", check_frame_bytes, rec, repo)
    history = rec.guard("YAML", lambda: (load_yaml(repo, HISTORY_CONFIG), load_yaml(repo, FRAME_CONFIG)))
    fields = None
    if history is not None:
        result = rec.guard("YAML", check_history, rec, *history)
        if result is not None:
            fields = result
    if fields is not None and not any(PLACEHOLDER in str(v) for v in fields.values()):
        rec.guard("MOTION", check_motion, rec, repo, store, fields, contract)
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
        if history is not None:
            from omegaconf import OmegaConf
            leaves = rec.guard("LEAVES", lambda: (
                model_leaves(cfg, OmegaConf.create(history[0])),
                model_leaves(base, OmegaConf.create({**history[1], "motion": {
                    **MOTION_FIXED, "checkpoint_sha256": fields["checkpoint_sha256"]}})),
                base_leaves(cfg.weight_loader.params_path),
                model_leaves(base, None)))
            if leaves is not None:
                rec.guard("LEAVES", check_leaves, rec, *leaves)
    failed = rec.failed
    verdict = "FAIL" if failed else "PASS"
    print(f"LAUNCH_CONTRACT={verdict} checks={len(rec.rows)} failed={','.join(failed) or 0}", flush=True)
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
    args = parser.parse_args(raw)
    return 1 if run(args, argv).failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
