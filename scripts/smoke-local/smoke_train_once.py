#!/usr/bin/env python3
"""只调用一次 train.main 的 smoke 入口。

**为什么必须有这个文件**：`scripts/train.py` 的 `__main__` 是

    main(_config.cli(), tentative_run=True)
    time.sleep(20)
    main(_config.cli())          # ← 紧接着又跑一次完整训练

直接执行它会在 tentative 的 11 步之后**接着起 80k step 的正式训练**。smoke 只要前者，
所以这里自己做入口，只调一次 `main(config, tentative_run=True)`，其余逻辑（dataloader、
模型、loss、优化器）一行不改、全部走原版。

同时做几道 fail-loud 护栏，防止这个入口被误当成正式训练启动器。
"""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))   # train.py 在 scripts/ 下，不是包

import train as _train  # noqa: E402

import mme_vla_suite.training.config as _config  # noqa: E402

_MAX_SMOKE_STEPS = 50
_EXPECTED_HISTORY_CONFIG = "perceptual-framesamp-context.yaml"


def main() -> None:
    config = _config.cli()
    if config.num_train_steps > _MAX_SMOKE_STEPS:
        raise ValueError(
            f"smoke 入口只允许 ≤{_MAX_SMOKE_STEPS} steps（当前 {config.num_train_steps}）；"
            f"正式训练请用 scripts/train.py"
        )
    if config.wandb_enabled:
        raise ValueError("smoke 必须关闭 wandb（--no-wandb-enabled）")
    if config.overwrite or config.resume:
        raise ValueError("smoke 禁止 overwrite / resume——避免误清已有 run 目录")
    if not config.model.use_history:
        raise ValueError("smoke 必须启用 --model.use-history")
    if config.model.history_config != _EXPECTED_HISTORY_CONFIG:
        raise ValueError(f"smoke 的 history_config 必须是 {_EXPECTED_HISTORY_CONFIG}")
    if config.batch_size % max(1, config.fsdp_devices):
        raise ValueError(
            f"batch_size {config.batch_size} 必须能被 fsdp_devices {config.fsdp_devices} 整除"
        )
    _train.main(config, tentative_run=True)


if __name__ == "__main__":
    main()
