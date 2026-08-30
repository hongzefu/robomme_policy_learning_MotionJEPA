#!/usr/bin/env python3
"""单步定点梯度取证（dtype 统一修复第二块的最便宜一档）。

**它做什么**：固定初始 TrainState + 固定 batch，只算一次前向反向，把梯度逐叶落盘。
修复前后各跑一次，逐元素对拍。比 1000 步轨迹便宜两个数量级，且能定位到具体是哪个
参数叶子先分叉。

**三个定点 batch 的分工**（取自 `_common.build_fixture_batches` 的定点计划）：
- `mixed1`（1 短 + 7 满长）：主判据——唯一存在 dtype 差异的典型场景；
- `allshort`（全短样本）：差异密度最大化；
- `allfull`（全满长）：**阴性对照**——两侧本就同为 bf16，若它不逐位相同，说明改动
  越界（与 dtype 无关），必须立刻停下排查。

**初始 state 怎么保证与 G0b 同源**：不加载本机那份 45.4 GiB 的 `state_step_0.bin`，
而是用同 seed / 同 config 现场 `init_train_state`，再把逐叶 sha256 与 G0b r1
`param_checksums.jsonl` 的 step 0 `per_leaf` 逐条比对——全等即同源（G0b 的步 0 摘要
本来就是 init 后立即记的）。校验失败即 fail-loud。

**梯度怎么取**：`train_step` 用 `nnx.value_and_grad` 算出 grads 但不返回，本脚本
复刻其前半段。复刻有漂移风险，因此照 bench 既有做法加源码指纹护栏：`train.train_step`
的源码不含预期子串即当场报错。

用法（cwd 仓库根；正确性口径须注入 D2 档 XLA_FLAGS）：

    DTYPE_GRAD_DIR=<records 目录> \\
    DTYPE_BATCH_FIXTURE_DIR=<batch fixture 目录> \\
    [DTYPE_GRAD_ARRAYS_DIR=<梯度数组目录，不设则只落摘要>] \\
    [DTYPE_BASELINE_CHECKSUMS=<G0b r1 param_checksums.jsonl>] \\
    XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" \\
    UV_LINK_MODE=copy uv run scripts/dtype-unify/single_step_grad.py mme_vla_suite ...
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))   # train.py 在 scripts/ 下，不是包

import _common as C  # noqa: E402
from flax import nnx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import train as _train  # noqa: E402

from mme_vla_suite.models.config.utils import get_history_config  # noqa: E402
from mme_vla_suite.models.integration.history_observation import HistAugObservation  # noqa: E402
import mme_vla_suite.training.config as _config  # noqa: E402
from mme_vla_suite.training.dataloader import _create_framesamp_dataset  # noqa: E402
from openpi.training import sharding  # noqa: E402
from openpi.training.data_loader import _collate_fn  # noqa: E402
from openpi.training.data_loader import transform_dataset  # noqa: E402

_EXPECTED_HISTORY_CONFIG = "perceptual-framesamp-context.yaml"

# 三个定点 batch：每种组成取该组第一个（batch_id 由 BATCH_PLAN 顺序决定，两侧一致）
_GRAD_BATCH_KINDS = ("mixed1", "allshort", "allfull")


def _guard_train_step_source() -> None:
    """源码指纹护栏：本脚本复刻了 train_step 的梯度段，train.py 变了必须立刻炸。"""
    src = inspect.getsource(_train.train_step)
    for needle in (
        "nnx.value_and_grad(",
        "nnx.DiffState(0, config.trainable_filter)",
        "model.compute_loss(rng, observation, actions, train=True)",
        "jax.random.fold_in(rng, state.step)",
        # commitV4.3 起 train_step 为二返回（stats 链整删）；stats/has_aux 回潮时此针立断
        "return new_state, info",
    ):
        if needle not in src:
            raise SystemExit(
                f"train.train_step 源码中找不到 {needle!r}——梯度复刻前提失效，"
                f"请检查 scripts/train.py 是否已改动"
            )


def _grad_only(config, rng, state, batch):
    """复刻 `train_step` 的梯度段（到 value_and_grad 为止），只返回 loss 与 grads。

    与 train_step 逐行对应：merge → train() → fold_in(rng, step) → DiffState 过滤
    frozen 参数 → value_and_grad。不做 optimizer update / EMA，因此不改变 state。
    """
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


def _leaf_items(tree, prefix: str):
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, leaf in flat:
        if leaf is None:
            continue
        yield prefix + jax.tree_util.keystr(path), leaf


def _verify_same_origin(state, baseline_checksums: pathlib.Path) -> dict:
    """现场 init 出的 TrainState 与 G0b 步 0 摘要逐叶比对。"""
    rows = [json.loads(x) for x in baseline_checksums.read_text().splitlines() if x.strip()]
    step0 = next((r for r in rows if int(r["step"]) == 0), None)
    if step0 is None:
        raise SystemExit(f"基线摘要里找不到 step 0: {baseline_checksums}")
    want = step0["per_leaf"]

    trees = {"params": state.params, "opt_state": state.opt_state, "step": state.step}
    if state.ema_params is not None:
        trees["ema_params"] = state.ema_params
    got = {}
    for name, tree in trees.items():
        for key, leaf in _leaf_items(tree, name):
            got[key] = C.leaf_sha256(np.asarray(jax.device_get(leaf)))

    only_want = sorted(set(want) - set(got))[:5]
    only_got = sorted(set(got) - set(want))[:5]
    bad = sorted(k for k in set(want) & set(got) if want[k] != got[k])
    if only_want or only_got or bad:
        raise SystemExit(
            f"初始 TrainState 与 G0b 步 0 不同源: 仅基线有 {only_want}，仅现场有 {only_got}，"
            f"{len(bad)} 个叶子摘要不同（首个: {bad[0] if bad else '-'}）"
        )
    return {"n_leaves": len(got), "baseline": str(baseline_checksums), "verdict": "PASS"}


def _build_batches(config, plan: list[dict], fixture_dir: pathlib.Path) -> dict[str, dict]:
    """按定点计划构造三个 batch（走完整 transform 链），并落 fixture 位型容器。"""
    history_config = get_history_config(config.model.history_config)
    data_config = config.data.create(config.assets_dirs, config.model)
    ds = _create_framesamp_dataset(
        dataset_path=str(config.dataset_path),
        data_config=data_config,
        history_config=history_config,
        action_horizon=config.model.action_horizon,
    )
    tds = transform_dataset(ds, data_config)

    out = {}
    only = os.environ.get("DTYPE_GRAD_KINDS", "")
    kinds = tuple(k for k in _GRAD_BATCH_KINDS if not only or k in only.split(","))
    for kind in kinds:
        spec = next(s for s in plan if s["kind"] == kind)
        batch = _collate_fn([tds[i] for i in spec["indices"]])
        bdir = fixture_dir / kind
        summary = C.describe_tree(batch)
        flat, _ = jax.tree_util.tree_flatten_with_path(batch)
        for path, leaf in flat:
            k = jax.tree_util.keystr(path)
            if summary[k]["kind"] == "array":
                C.save_array(bdir, k, np.asarray(leaf))
        (bdir / "batch_meta.json").write_text(json.dumps({
            "kind": kind, "batch_id": spec["batch_id"], "indices": spec["indices"],
            "keys": summary,
        }, ensure_ascii=False), encoding="utf-8")
        out[kind] = {"batch": batch, "spec": spec, "summary": summary}
        print(f"[fixture] {kind} batch 已构造并落盘（{len(summary)} 键）", flush=True)
    return out


def main() -> None:
    config = _config.cli()
    if config.model.history_config != _EXPECTED_HISTORY_CONFIG:
        raise SystemExit(f"本工具只支持 {_EXPECTED_HISTORY_CONFIG}")
    _guard_train_step_source()

    rec_dir = pathlib.Path(os.environ["DTYPE_GRAD_DIR"])
    rec_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = pathlib.Path(os.environ["DTYPE_BATCH_FIXTURE_DIR"])
    arrays_dir_raw = os.environ.get("DTYPE_GRAD_ARRAYS_DIR")
    arrays_dir = pathlib.Path(arrays_dir_raw) if arrays_dir_raw else None
    baseline_raw = os.environ.get("DTYPE_BASELINE_CHECKSUMS")

    jax.config.update(
        "jax_compilation_cache_dir",
        str(pathlib.Path(f"~/.cache/jax_{config.exp_name}").expanduser()),
    )

    manifest = C.load_manifest(C.REPO_ROOT / "v1-store" / "episode_manifest.json")
    groups = C.build_fixture_indices(manifest)
    plan = C.build_fixture_batches(groups)

    batches = _build_batches(config, plan, fixture_dir)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    t0 = time.time()
    train_state, train_state_sharding = _train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    print(f"[state] init_train_state 完成，用时 {time.time() - t0:.1f}s", flush=True)

    origin = None
    if baseline_raw:
        origin = _verify_same_origin(train_state, pathlib.Path(baseline_raw))
        print(f"[state] 同源校验 PASS（{origin['n_leaves']} 叶子）", flush=True)

    pgrad = jax.jit(
        functools.partial(_grad_only, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
    )

    results = {}
    for kind, item in batches.items():
        host = item["batch"]
        dev = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), host)
        obs_batch = (HistAugObservation.from_dict(dev), dev["actions"])
        t1 = time.time()
        # 必须与 train.py 一样在 set_mesh 上下文里调用：模型内部的
        # activation_sharding_constraint 依赖活动 mesh，缺它 HLO 就与训练不同
        with sharding.set_mesh(mesh):
            loss, grads = pgrad(train_rng, train_state, obs_batch)
        jax.block_until_ready(grads)
        per_leaf, stats = {}, {}
        adir = (arrays_dir / kind) if arrays_dir else None
        for key, leaf in _leaf_items(grads, "grads"):
            arr = np.asarray(jax.device_get(leaf))
            per_leaf[key] = C.leaf_sha256(arr)
            a64 = arr.astype(np.float64)
            stats[key] = {
                "dtype": str(arr.dtype), "shape": list(arr.shape),
                "max_abs": float(np.max(np.abs(a64))) if a64.size else 0.0,
                "l2": float(np.sqrt(np.sum(a64 * a64))) if a64.size else 0.0,
            }
            if adir is not None:
                C.save_array(adir, key, arr)
        results[kind] = {
            "batch_id": item["spec"]["batch_id"], "indices": item["spec"]["indices"],
            "loss_hex": float(loss).hex(), "loss": float(loss),
            "n_leaves": len(per_leaf), "seconds": round(time.time() - t1, 2),
            "per_leaf": per_leaf, "stats": stats,
            "batch_keys": item["summary"],
        }
        print(f"[grad] {kind}: loss={float(loss):.6f} 叶子 {len(per_leaf)} "
              f"用时 {time.time() - t1:.1f}s", flush=True)

    (rec_dir / "grad_summary.json").write_text(json.dumps({
        "schema": "dtype-unify-grad-v1",
        "seed": config.seed, "fsdp_devices": config.fsdp_devices,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "same_origin": origin,
        "arrays_dir": str(arrays_dir) if arrays_dir else None,
        "fixture_dir": str(fixture_dir),
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"GRAD_DONE kinds={len(results)} out={rec_dir}", flush=True)


if __name__ == "__main__":
    main()
