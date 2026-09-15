#!/usr/bin/env python3
"""从现成位型容器重建 batch，执行 modulation 关闭态的单步梯度取证。

两侧共用主副本的本工具、train.py 与 _common.py；模型和配置从各自 cwd 的
源码树导入。数据读取不经过 Dataset，不生成或修改 fixture，不更新参数。
必设 DTYPE_GRAD_DIR、DTYPE_BATCH_FIXTURE_DIR、MMEVLA_JAX_CACHE_DIR；
以 DTYPE_SOURCE_COMMIT 固定源码 SHA。初态为全部 state.params 的逐叶摘要。
"""

from __future__ import annotations

import functools
import importlib.metadata
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if not (_REPO_ROOT / "pyproject.toml").is_file():
    raise SystemExit(f"工具根错误：{_REPO_ROOT}")
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_ROOT / "scripts/training"))

import _common as C  # noqa: E402
from flax import nnx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import train as _train  # noqa: E402
from mme_vla_suite.models.integration.history_observation import HistAugObservation  # noqa: E402
import mme_vla_suite.training.config as _config  # noqa: E402
from openpi.training import sharding  # noqa: E402

_KINDS = ("mixed1", "allshort", "allfull")
_CONFIGS = ("perceptual-framesamp-modul.yaml", "perceptual-framesamp-modul-8frame-8x8.yaml")
_FLAGS = ("--xla_gpu_deterministic_ops=true", "--xla_gpu_autotune_level=0")


def _guard_train_step_source():
    """与现有 single_step_grad 的护栏一致；模型始终由各自源码提供。"""
    src = inspect.getsource(_train.train_step)
    for needle in ("nnx.value_and_grad(", "nnx.DiffState(0, config.trainable_filter)",
                   "model.compute_loss(rng, observation, actions, train=True)",
                   "jax.random.fold_in(rng, state.step)", "return new_state, info"):
        if needle not in src:
            raise ValueError(f"训练梯度护栏缺失：{needle}")


def _grad_only(config, rng, state, batch):
    """与 single_step_grad._grad_only 数值语句相同，不执行参数更新。"""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, observation, actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jax.numpy.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )
    return loss, grads


def _leaf_items(tree, prefix):
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, leaf in flat:
        if leaf is not None:
            yield prefix + jax.tree_util.keystr(path), leaf


def _build_batches(fixture_dir, batch_size):
    """只加载数组键，并逐键复验 dtype、shape、raw/canonical 摘要。"""
    batches = {}
    for kind in _KINDS:
        bdir = fixture_dir / kind
        meta = json.loads((bdir / "batch_meta.json").read_text())
        if meta["kind"] != kind or len(meta["indices"]) != batch_size:
            raise ValueError(f"fixture 类型或 batch 大小不符：{bdir}")
        batch = {}
        expected = {}
        for key, desc in meta["keys"].items():
            if desc["kind"] == "none":
                if key not in {"['motion_emb']", "['motion_pos']", "['motion_mask']", "['mem_order']"}:
                    raise ValueError(f"未预期的 None 键：{key}")
                continue
            if desc["kind"] != "array":
                raise ValueError(f"fixture 出现非数组键：{key}")
            parts = re.findall(r"\['([^']+)'\]", key)
            if not parts or "".join(f"['{p}']" for p in parts) != key:
                raise ValueError(f"键路径非法：{key}")
            arr = C.load_array(bdir, key)
            if C.describe_leaf(key, arr) != desc or arr.shape[0] != batch_size:
                raise ValueError(f"fixture 内容摘要不符：{bdir} {key}")
            node = batch
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            if parts[-1] in node:
                raise ValueError(f"fixture 重复键：{key}")
            node[parts[-1]] = arr
            expected[key] = desc
        if len(expected) != 12 or C.describe_tree(batch) != expected:
            raise ValueError(f"重建 batch 键集不符：{bdir}")
        batches[kind] = {"batch": batch, "spec": meta, "summary": expected}
        print(f"FIXTURE=PASS kind={kind} keys={len(expected)} batch={batch_size}", flush=True)
    return batches


def _git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def _source_record(config):
    """核对实际导入来自当前源码树，工具来自主副本；仅允许额外的同字节 YAML。"""
    source = pathlib.Path.cwd().resolve()
    head = _git("rev-parse", "HEAD")
    if head != os.environ["DTYPE_SOURCE_COMMIT"]:
        raise ValueError(f"源码 SHA 不符：{head}")
    status = _git("status", "--porcelain")
    allowed = "?? src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8.yaml"
    if status not in ("", allowed):
        raise ValueError(f"源码工作区包含未允许改动：{status}")
    origins = {}
    for name in ("mme_vla_suite.training.config", "mme_vla_suite.models.integration.history_pi0",
                 "mme_vla_suite.models.integration.history_observation", "openpi.models.gemma"):
        mod = __import__(name, fromlist=["__file__"])
        path = pathlib.Path(mod.__file__).resolve()
        if not path.is_relative_to(source / "src"):
            raise ValueError(f"源码导入越界：{name} -> {path}")
        origins[name] = str(path)
    yrel = pathlib.Path("src/mme_vla_suite/models/config/robomme") / config.model.history_config
    if (source / yrel).read_bytes() != (_REPO_ROOT / yrel).read_bytes():
        raise ValueError("两侧 history YAML 字节不一致")
    return {"head": head, "cwd": str(source), "status": status, "origins": origins,
            "history_config_sha256": C.sha256_file(source / yrel), "tool_root": str(_REPO_ROOT)}


def main():
    config = _config.cli()
    if config.model.history_config not in _CONFIGS or not config.model.use_history:
        raise ValueError("只接受 modulation 关闭态两档配置")
    if any(f not in os.environ.get("XLA_FLAGS", "").split() for f in _FLAGS):
        raise ValueError("必须启用已约定的 XLA 确定性档")
    if os.environ.get("DTYPE_BASELINE_CHECKSUMS"):
        raise ValueError("modulation 没有黄金基线，禁止引用 context 的初态摘要")
    _guard_train_step_source()
    source = _source_record(config)
    rec_dir = pathlib.Path(os.environ["DTYPE_GRAD_DIR"])
    rec_dir.mkdir(parents=True, exist_ok=False)
    fixture_dir = pathlib.Path(os.environ["DTYPE_BATCH_FIXTURE_DIR"])
    jax.config.update("jax_compilation_cache_dir", os.environ["MMEVLA_JAX_CACHE_DIR"])
    batches = _build_batches(fixture_dir, config.batch_size)
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    start = time.time()
    state, state_sharding = _train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    initial = {}
    for key, leaf in _leaf_items(state.params, "params"):
        arr = np.asarray(jax.device_get(leaf))
        if not np.isfinite(arr).all():
            raise ValueError(f"初态参数非有限：{key}")
        initial[key] = C.describe_leaf(key, arr)
    if not initial:
        raise ValueError("初态参数树为空")
    (rec_dir / "init_params.json").write_text(json.dumps(initial, ensure_ascii=False, indent=2) + "\n")
    print(f"INIT_PARAMS=PASS leaves={len(initial)} seconds={time.time()-start:.1f}", flush=True)
    pgrad = jax.jit(functools.partial(_grad_only, config),
                   in_shardings=(replicated, state_sharding, data_sharding))
    results = {}
    for kind, item in batches.items():
        dev = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), item["batch"])
        obs_batch = (HistAugObservation.from_dict(dev), dev["actions"])
        t1 = time.time()
        with sharding.set_mesh(mesh):
            loss, grads = pgrad(train_rng, state, obs_batch)
        jax.block_until_ready(grads)
        if not np.isfinite(float(loss)):
            raise ValueError(f"loss 非有限：{kind}")
        per_leaf = {}
        for key, leaf in _leaf_items(grads, "grads"):
            arr = np.asarray(jax.device_get(leaf))
            if not np.isfinite(arr).all():
                raise ValueError(f"梯度非有限：{kind} {key}")
            per_leaf[key] = C.leaf_sha256(arr)
        if not per_leaf:
            raise ValueError("可训练梯度树为空")
        results[kind] = {"batch_id": item["spec"]["batch_id"], "indices": item["spec"]["indices"],
                         "loss_hex": float(loss).hex(), "loss": float(loss), "n_leaves": len(per_leaf),
                         "seconds": round(time.time()-t1, 2), "per_leaf": per_leaf, "finite": True,
                         "batch_keys": item["summary"]}
        print(f"GRAD_KIND=PASS kind={kind} loss={float(loss):.9g} leaves={len(per_leaf)} seconds={time.time()-t1:.1f}", flush=True)
    environment = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "uv_lock_sha256": C.sha256_file(pathlib.Path.cwd() / "uv.lock"),
                   "packages": {name: importlib.metadata.version(name)
                                for name in ("jax", "jaxlib", "flax", "numpy", "ml_dtypes", "optax")},
                   "jax_enable_x64": bool(jax.config.jax_enable_x64),
                   "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version",
                                                    "--format=csv,noheader"], text=True).splitlines()}
    summary = {"schema": "modul-fixed-grad-v1", "seed": config.seed, "fsdp_devices": config.fsdp_devices,
               "xla_flags": os.environ["XLA_FLAGS"], "initial_params": initial, "source": source,
               "fixture_dir": str(fixture_dir), "environment": environment,
               "results": results, "seconds": time.time()-start}
    (rec_dir / "grad_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"GRAD_DONE kinds={len(results)} out={rec_dir}", flush=True)


if __name__ == "__main__":
    main()
