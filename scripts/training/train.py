import dataclasses
import functools
import json
import logging
import os
import pathlib
import platform
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import time

from typing import Any
import tqdm_loggable.auto as tqdm
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


import mme_vla_suite.models.integration.history_pi0 as _model
from mme_vla_suite.models.integration.history_observation import (
    HistAugObservation,
)
import mme_vla_suite.training.config as _config
import mme_vla_suite.training.dataloader as _data_loader
from mme_vla_suite.models.config.utils import get_history_config


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            # 2026-08-30 实测：本组 API key 写 daiyp_umich 报 403，写
            # hongzefu-university-of-michigan PASS；经 WANDB_ENTITY 覆盖、默认保持上游值
            entity=os.environ.get("WANDB_ENTITY", "daiyp_umich"),
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        # train.py 位于 scripts/training/，归档仓库根须上跳三层（V4.6 搬移修正）
        wandb.run.log_code(epath.Path(__file__).parent.parent.parent)

def init_history_config(config: _config.TrainConfig):
    # this is for evaluation config checking
    if config.model.history_config is not None:
        with open(config.checkpoint_dir / "history_config.txt", "w") as f:
            f.write(config.model.history_config)

def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True
    )

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )


def params_split(params, trainable_filter):
    memory_filter = params.filter(trainable_filter).filter(
        nnx.All(nnx.Param, nnx_utils.PathRegex(".*mem.*"))
    )
    non_memory_filter = params.filter(trainable_filter).filter(
        nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*mem.*")))
    )
    return memory_filter, non_memory_filter


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )

    def init(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        logging.info(
            f"Total Model Size: {sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1024 / 1024} MB"
        )
        logging.info(
            f"Trainable Model Size: {sum(x.size for x in jax.tree_util.tree_leaves(params.filter(config.trainable_filter))) / 1024 / 1024} MB"
        )

        memory_params, non_memory_params = params_split(params, config.trainable_filter)
        logging.info(
            f"Memory-related  Size: {sum(x.size for x in jax.tree_util.tree_leaves(memory_params))/1024/1024} MB"
        )
        logging.info(
            f"Non-Memory Size: {sum(x.size for x in jax.tree_util.tree_leaves(non_memory_params))/1024/1024} MB"
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        # replace pi05_base with the checkpoint id
        ckpt_epath = config.checkpoint_dir / str(config.resum_ckpt_id) / "params"
        weight_loader = _weight_loaders.CheckpointWeightLoader(str(ckpt_epath))
        partial_params = _load_weights_and_validate(weight_loader, train_state_shape.params.to_pure_dict())
    else:
        partial_params = _load_weights_and_validate(
            config.weight_loader, train_state_shape.params.to_pure_dict()
        )
        
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[HistAugObservation, _model.Actions],
) -> tuple[
    training_utils.TrainState, dict[str, at.Array]
]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.HistoryPi0,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        actions: _model.Actions,
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(
        loss_fn, argnums=diff_state
    )(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state
    )

    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(
                nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")
            ),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "llm_grad_norm": optax.global_norm(grads.PaliGemma.llm),
    }
    if config.model.use_history and hasattr(grads, "mem_encoder"):
        info["mem_enc_norm"] = optax.global_norm(grads.mem_encoder)

    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"TrainConfig: {config}")

    # ── fail-loud 护栏（早于权重加载与 JIT）─────────────────────────────
    # checkpoint 只存 EMA 权重、不存优化器状态，续跑会丢 AdamW 动量并把 warmup
    # 从头再爬一遍，故一律禁用两个开关（沿袭 prod_train_once.py 的护栏，v5.0 迁入）。
    if config.overwrite or config.resume:
        raise ValueError("本入口禁用 --overwrite / --resume：续跑语义有损"
                         "（checkpoint 只存 EMA，丢 AdamW 动量与 warmup 计数）")
    # HistoryPi0.__init__ 的 else 分支（# safe setting）会静默训出不含记忆分支的
    # 模型且全链路零告警，这是唯一的「静默」失败模式，必须 fail-loud。
    if not config.model.use_history:
        raise ValueError("正式训练必须启用 --model.use-history")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path(f"~/.cache/jax_{config.exp_name}").expanduser()),
    )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    init_history_config(config)
    history_config = get_history_config(config.model.history_config)
    
    if history_config:
        if history_config.streaming_obs_horizon == 16:
            assert config.model.action_horizon == 20, "action_horizon must be 20 when streaming_obs_horizon is 16"
        else:
            raise ValueError(f"Unsupported streaming_obs_horizon: {history_config.streaming_obs_horizon}")

    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    data_loader = _data_loader.create_data_loader(
        config.dataset_path,
        data_config,
        history_config=config.model.history_config,
        sharding=data_sharding,
        shuffle=True,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(
        f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}"
    )

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(
            np.concatenate(
                [np.array(img[i]) for img in batch[0].images.values()], axis=1
            )
        )
        for i in range(min(1, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming
    )
    jax.block_until_ready(train_state)
    logging.info(
        f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params.filter(nnx.All(nnx.Param)))}"
    )

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)

    if config.resum_ckpt_id is not None:
        start_step += config.resum_ckpt_id

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        batch = next(data_iter)

        if (
            step % config.save_interval == 0 and step > start_step
        ) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


# ── 可选 metrics 记录器（仅 __main__ 路径、设 TRAIN_RECORD_DIR 时装载）───────────
# 供集群收尾判读：util/analyze_gpu_util.py 以 metrics.jsonl 为唯一硬依赖，
# prod_train_once.py 删除后由本记录器自产（v5.0）。未设 TRAIN_RECORD_DIR 时零行为差异。
# 记录目录经环境变量传入、不走命令行——_config.cli()（tyro）会吃掉整个 sys.argv[1:]
# 且对未知参数直接报错退出。


class _MetricsProxy:
    """替换本模块全局名 `wandb` 的代理：log 先记录再转发，其余属性透传。

    ⚠ 不能直接 patch `wandb.log`：main 里的 `wandb.init(mode="disabled")` 会把
    wandb 模块级的 `log` 重新赋值成 run 的 stub，把 patch 盖掉（2026-08-24 实测）。
    行 schema 与 g0/bench_train_steps.py::_WandbProxy 逐字段相同。
    """

    def __init__(self, real_wandb, metrics_path: pathlib.Path):
        self._real = real_wandb
        self._metrics_path = metrics_path

    def log(self, data, step=None, **kwargs):
        # main 两处调用：step 0 的 camera_views（wandb.Image 列表，非标量，跳过）
        # 与每个 log_interval 的 reduced_info（全标量，逐键记录）
        row: dict = {"step": int(step) if step is not None else None,
                     "wall_time": time.time()}
        n_scalar = 0
        for k, v in data.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            row[k] = {"dec": fv, "hex": fv.hex()}
            n_scalar += 1
        if n_scalar:
            with self._metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return self._real.log(data, step=step, **kwargs)   # disabled 模式下是 no-op

    def __getattr__(self, name):
        return getattr(self._real, name)


def _install_metrics_recorder(record_dir: pathlib.Path) -> None:
    # ⚠ 不能用「目录非空即拒」做护栏：驱动 sbatch 会先 mkdir、写 env.json、起五路
    # 采样器之后才起训练——那时目录里已有 6 个文件（2026-08-28 job 59092143 实测被
    # 自己的护栏打死）。「该目录是否已被某次训练用过」的正确信号是 metrics.jsonl。
    record_dir.mkdir(parents=True, exist_ok=True)
    metrics = record_dir / "metrics.jsonl"
    if metrics.exists():
        raise FileExistsError(f"记录目录已有 metrics.jsonl（该目录已被某次训练用过），"
                              f"拒绝覆盖: {metrics}")
    globals()["wandb"] = _MetricsProxy(wandb, metrics)


def _finalize_record(record_dir: pathlib.Path, config: _config.TrainConfig) -> None:
    meta = {
        "argv": list(sys.argv),
        "entry": "scripts/training/train.py",
        "checkpoint_dir": str(config.checkpoint_dir),
        "num_train_steps": config.num_train_steps,
        "log_interval": config.log_interval,
        "save_interval": config.save_interval,
    }
    with (record_dir / "run_meta.json").open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    try:
        wandb.finish()
    except Exception as e:  # 收尾失败不掩盖训练本身的结果
        print(f"[train] wandb.finish() 失败（忽略）: {e}", flush=True)


if __name__ == "__main__":
    _record_dir = os.environ.get("TRAIN_RECORD_DIR")
    _cfg = _config.cli()
    if _record_dir:
        _install_metrics_recorder(pathlib.Path(_record_dir))
    try:
        main(_cfg)
    finally:
        if _record_dir:
            _finalize_record(pathlib.Path(_record_dir), _cfg)
