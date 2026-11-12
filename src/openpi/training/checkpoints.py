from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            # "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        # "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                # "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])



from flax import nnx

def save_state_trainable_only(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    trainable_filter: nnx.filterlib.Filter,
    variables_filter: nnx.filterlib.Filter,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    ema_params = state.ema_params
    params = state.params
    train_state = dataclasses.replace(state, ema_params=None, params=nnx.statelib.State({}))
    
    if ema_params is not None:
        params_to_save = ema_params
    else:
        params_to_save = params.filter(trainable_filter)
                
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {
            "params":  params_to_save,
            "variables": params.filter(variables_filter),
        },
    }
    checkpoint_manager.save(step, items)
    

def restore_state_trainable_only(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None,
    trainable_filter: nnx.filterlib.Filter,
    variables_filter: nnx.filterlib.Filter,
    partial_params: at.Params,
    no_sharding: bool = False
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        ema_params = state.ema_params
        params = state.params
        train_state = dataclasses.replace(state, ema_params=None, params=nnx.statelib.State({}))
        
        if no_sharding:
            import numpy as np
            from flax import traverse_util
            mesh = jax.sharding.Mesh(jax.devices(), ("x",))
            sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            path = checkpoint_manager.directory
            step = checkpoint_manager.latest_step()
            ckpt_path = path / str(step)

            with ocp.PyTreeCheckpointer() as ckptr:
                params_path = ckpt_path / "params"
                item = ckptr.metadata(params_path)
                restored_params = ckptr.restore(
                    params_path,
                    ocp.args.PyTreeRestore(
                        item=item,
                        restore_args=jax.tree.map(
                            lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=np.ndarray), item
                        ),
                    ),
                )
                
                flat_params = traverse_util.flatten_dict(restored_params)
                if all(kp[-1] == "value" for kp in flat_params):
                    flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
                restored_params = traverse_util.unflatten_dict(flat_params)

                partial_params = jax.tree.map(lambda x: x.astype(jax.numpy.bfloat16), partial_params)        
                params.replace_by_pure_dict(partial_params) # restore the frozen params (bfloat16)
                params.replace_by_pure_dict(restored_params["params"]) # restore the trainable params
                if "variables" in restored_params:
                    params.replace_by_pure_dict(restored_params["variables"]) # restore the variables params
                
            
        else:
            trainable_params = params.filter(trainable_filter)
            variables_params = params.filter(variables_filter)
            restored = checkpoint_manager.restore(
                step,
                items={
                    "train_state": train_state,
                    "params": {"params": trainable_params, "variables": variables_params},
                },
            )
            partial_params = jax.tree.map(lambda x: x.astype(jax.numpy.bfloat16), partial_params)        
            params.replace_by_pure_dict(partial_params) # restore the frozen params (bfloat16)
            params.replace_by_pure_dict(restored["params"]["params"].to_pure_dict()) # restore the trainable params
            params.replace_by_pure_dict(restored["params"]["variables"].to_pure_dict()) # restore the variables params
            
            
    if ema_params is not None:
        return dataclasses.replace(restored["train_state"], ema_params=restored["params"]["params"] , params=params)
    else:
        return dataclasses.replace(restored["train_state"], params=params)


# ── 完整续跑状态（--resume-full，G4C-S4）─────────────────────────────────────
# 与上面只存 EMA 的 save_state 并存、互不影响：把完整 train_state（raw params、
# AdamW opt_state、step、ema_params）按叶子逐个经 orbax 存进 <run_dir>/resume/，
# 训练用 rng 在 TrainState 之外，以 key_data 写进 manifest。
# 落盘流程：先写 <run_dir>/resume.tmp-<step>/（state/ + manifest.json 最后写），
# 再把旧 resume/ 改名为 resume.old-<step>、tmp 改名为 resume/、删旧份；只保留最新 1 份。
# 中途任一步失败时旧 resume/ 原样保留（最多残留 resume.tmp-*，下次保存时清掉）。

import hashlib
import json
import os

import numpy as np

RESUME_DIRNAME = "resume"
RESUME_SCHEMA = "resume_full/v1"
# manifest 与当前 config 必须逐项相等的关键字段
RESUME_CONFIG_KEYS = ("batch_size", "grad_accum_steps", "num_train_steps", "seed", "exp_name")


class ResumeStateError(RuntimeError):
    """resume/ 缺件、校验失败或与当前 config 不一致时抛出（fail-loud）。"""


def resume_config_fields(config) -> dict:
    """从 TrainConfig（或任意带这些属性的对象）取 manifest 需要比对的关键字段。"""
    return {k: getattr(config, k) for k in RESUME_CONFIG_KEYS}


def _leaf_records(tree) -> tuple[list, list[dict]]:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    leaves = [leaf for _, leaf in flat]
    records = [
        {"path": jax.tree_util.keystr(path), "shape": list(np.shape(leaf)), "dtype": str(np.dtype(leaf.dtype))}
        for path, leaf in flat
    ]
    return leaves, records


def _sha256_tree(root: str) -> dict[str, str]:
    """root 下全部文件（相对路径排序）的 sha256。"""
    out = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            h = hashlib.sha256()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                    h.update(chunk)
            out[os.path.relpath(full, root)] = h.hexdigest()
    return dict(sorted(out.items()))


def save_resume_state(
    run_dir,
    train_state,
    step: int,
    *,
    config_fields: dict,
    rng=None,
) -> str:
    """同步写完整续跑状态，返回最终 resume/ 路径。

    step 是 train_state.step（即续跑时第一个要跑的步号），必须与 state 内 step 一致。
    """
    run_dir = os.path.abspath(str(run_dir))
    state_step = int(jax.device_get(train_state.step))
    if state_step != int(step):
        raise ResumeStateError(f"save_resume_state: 传入 step={step} 与 train_state.step={state_step} 不一致")
    missing = [k for k in RESUME_CONFIG_KEYS if k not in config_fields]
    if missing:
        raise ResumeStateError(f"save_resume_state: config_fields 缺少关键字段 {missing}")

    leaves, records = _leaf_records(train_state)
    leaves = [x if isinstance(x, jax.Array) else jax.numpy.asarray(x) for x in leaves]
    items = {f"leaf_{i:05d}": x for i, x in enumerate(leaves)}

    final_dir = os.path.join(run_dir, RESUME_DIRNAME)
    tmp_dir = os.path.join(run_dir, f"{RESUME_DIRNAME}.tmp-{state_step}")
    old_dir = os.path.join(run_dir, f"{RESUME_DIRNAME}.old-{state_step}")
    # 清理上次中断留下的半份：tmp-* 一律删；old-* 只在 resume/ 仍在时删（否则它可能是唯一一份）
    os.makedirs(run_dir, exist_ok=True)
    for name in os.listdir(run_dir):
        stale = os.path.join(run_dir, name)
        if name.startswith(f"{RESUME_DIRNAME}.tmp-") or (
            name.startswith(f"{RESUME_DIRNAME}.old-") and os.path.isdir(final_dir)
        ):
            epath.Path(stale).rmtree()
    os.makedirs(tmp_dir)

    # 1) orbax 按各叶子自身 sharding 存（FSDP 下多卡分片），同步等写完
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(os.path.join(tmp_dir, "state"), items)
    ckptr.wait_until_finished()
    ckptr.close()

    # 2) manifest 最后写：存在即代表 state/ 已完整
    rng_record = None
    if rng is not None:
        key_data = np.asarray(jax.device_get(jax.random.key_data(rng)))
        rng_record = {"impl": str(jax.random.key_impl(rng)), "key_data": key_data.astype(np.uint32).tolist()}
    manifest = {
        "schema": RESUME_SCHEMA,
        "step": state_step,
        "config": {k: config_fields[k] for k in RESUME_CONFIG_KEYS},
        "rng": rng_record,
        "num_leaves": len(records),
        "leaves": records,
        "files_sha256": _sha256_tree(os.path.join(tmp_dir, "state")),
    }
    manifest_tmp = os.path.join(tmp_dir, "manifest.json.part")
    with open(manifest_tmp, "w") as f:
        json.dump(manifest, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(manifest_tmp, os.path.join(tmp_dir, "manifest.json"))

    # 3) 原子替换：旧 resume/ → resume.old-<step>，tmp → resume/，再删旧份
    if os.path.exists(final_dir):
        os.rename(final_dir, old_dir)
    os.rename(tmp_dir, final_dir)
    if os.path.exists(old_dir):
        epath.Path(old_dir).rmtree()
    logging.info(f"RESUME_SAVE step={state_step} dir={final_dir} leaves={len(records)}")
    return final_dir


def read_resume_manifest(run_dir, *, expected_config: dict, verify_files: bool = True) -> dict:
    """读并校验 <run_dir>/resume/manifest.json；任何缺件或不一致都 fail-loud。"""
    run_dir = os.path.abspath(str(run_dir))
    final_dir = os.path.join(run_dir, RESUME_DIRNAME)
    manifest_path = os.path.join(final_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        olds = (
            sorted(p for p in os.listdir(run_dir) if p.startswith(f"{RESUME_DIRNAME}.old-"))
            if os.path.isdir(run_dir)
            else []
        )
        hint = f"；发现 {olds}，可能上次替换中断，需人工核对后改名为 resume/" if olds else ""
        raise ResumeStateError(f"--resume-full 需要 {manifest_path}，但它不存在{hint}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("schema") != RESUME_SCHEMA:
        raise ResumeStateError(f"resume manifest schema={manifest.get('schema')!r}，期望 {RESUME_SCHEMA!r}")
    for key in ("step", "config", "leaves", "files_sha256"):
        if key not in manifest:
            raise ResumeStateError(f"resume manifest 缺少字段 {key!r}")
    saved = manifest["config"]
    diffs = [
        f"{k}: manifest={saved.get(k)!r} 当前={expected_config.get(k)!r}"
        for k in RESUME_CONFIG_KEYS
        if saved.get(k) != expected_config.get(k)
    ]
    if diffs:
        raise ResumeStateError("resume manifest 与当前 config 关键字段不一致：" + "；".join(diffs))
    if verify_files:
        state_dir = os.path.join(final_dir, "state")
        if not os.path.isdir(state_dir):
            raise ResumeStateError(f"resume 状态目录缺失：{state_dir}")
        actual = _sha256_tree(state_dir)
        expected = manifest["files_sha256"]
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            changed = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
            raise ResumeStateError(
                f"resume/state 文件 sha256 校验失败：缺失 {missing[:5]}，多出 {extra[:5]}，内容变化 {changed[:5]}"
            )
    return manifest


def restore_resume_state(run_dir, target_state, *, expected_config: dict, manifest: dict | None = None):
    """按 target_state 的结构/shape/dtype/sharding 从 resume/ 还原完整 train_state。

    target_state 可以是真实 train_state（取其 sharding）或 ShapeDtypeStruct 树（须带 sharding）。
    返回 (train_state, manifest)；manifest["rng"] 为保存时的训练 rng（key_data），可能为 None。
    """
    if manifest is None:
        manifest = read_resume_manifest(run_dir, expected_config=expected_config)
    flat, treedef = jax.tree_util.tree_flatten(target_state)
    _, records = _leaf_records(target_state)
    if records != manifest["leaves"]:
        n = min(len(records), len(manifest["leaves"]))
        first = next((i for i in range(n) if records[i] != manifest["leaves"][i]), n)
        raise ResumeStateError(
            f"resume 叶子结构与当前 train_state 不一致：当前 {len(records)} 叶、manifest {len(manifest['leaves'])} 叶，"
            f"首个差异 #{first}: 当前={records[first] if first < len(records) else None} "
            f"manifest={manifest['leaves'][first] if first < len(manifest['leaves']) else None}"
        )
    default_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def _abstract(x):
        sharding = getattr(x, "sharding", None) or default_sharding
        return jax.ShapeDtypeStruct(np.shape(x), x.dtype, sharding=sharding)

    abstract = {f"leaf_{i:05d}": _abstract(x) for i, x in enumerate(flat)}
    state_dir = os.path.join(os.path.abspath(str(run_dir)), RESUME_DIRNAME, "state")
    ckptr = ocp.StandardCheckpointer()
    restored = ckptr.restore(state_dir, abstract)
    ckptr.close()
    leaves = [restored[f"leaf_{i:05d}"] for i in range(len(flat))]
    state = jax.tree_util.tree_unflatten(treedef, leaves)
    step = int(jax.device_get(state.step))
    if step != int(manifest["step"]):
        raise ResumeStateError(f"还原后 train_state.step={step} 与 manifest step={manifest['step']} 不一致")
    return state, manifest
