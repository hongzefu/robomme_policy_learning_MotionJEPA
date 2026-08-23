"""只调用一次原版 train.main 的 12-step smoke 入口."""

from __future__ import annotations

import train as _train

import mme_vla_suite.training.config as _config

_EXPECTED_STEPS = 12
_EXPECTED_BATCH_SIZE = 2
_EXPECTED_FSDP_DEVICES = 2
_EXPECTED_HISTORY_CONFIG = "perceptual-framesamp-context.yaml"
_EXPECTED_EXP_NAME = "v1-original-framesamp-context-12step-smoke"


def main() -> None:
    """校验固定 smoke 参数后, 原样调用训练 main 一次."""
    config = _config.cli()
    if config.num_train_steps != _EXPECTED_STEPS:
        raise ValueError(f"smoke 必须固定为 {_EXPECTED_STEPS} steps")
    if config.batch_size != _EXPECTED_BATCH_SIZE:
        raise ValueError(f"smoke batch size 必须为 {_EXPECTED_BATCH_SIZE}")
    if config.fsdp_devices != _EXPECTED_FSDP_DEVICES:
        raise ValueError(f"smoke fsdp_devices 必须为 {_EXPECTED_FSDP_DEVICES}")
    if config.num_workers != 0:
        raise ValueError("smoke num_workers 必须为 0")
    if config.exp_name != _EXPECTED_EXP_NAME:
        raise ValueError(f"smoke exp_name 必须为 {_EXPECTED_EXP_NAME}")
    if not config.model.use_history:
        raise ValueError("smoke 必须启用 model.use_history")
    if config.model.history_config != _EXPECTED_HISTORY_CONFIG:
        raise ValueError(
            f"smoke history_config 必须为 {_EXPECTED_HISTORY_CONFIG}"
        )
    if config.wandb_enabled:
        raise ValueError("smoke 必须关闭 wandb")
    if config.overwrite or config.resume:
        raise ValueError("smoke 禁止 overwrite 和 resume")
    _train.main(config, tentative_run=True)


if __name__ == "__main__":
    main()
