#!/usr/bin/env python3
"""本地 2 卡吞吐基准 + 一致性检验记录入口（训练循环一行不改）。

**为什么必须有这个文件**：`scripts/train.py` 的 `__main__` 是

    main(_config.cli(), tentative_run=True)
    time.sleep(20)
    main(_config.cli())          # ← 紧接着又跑一次完整训练（默认 80k step）

直接执行它无法只跑一小段。本入口只调一次 `train.main(config)`（不带 tentative_run，
靠 `--num-train-steps` 截断），dataloader、模型、loss、优化器、lr schedule 全走原版。

在此之上通过两处 monkeypatch 完成记录（均为「只读观测」，不改变训练计算）：

1. `train.wandb.log` → 逐步标量记录器：wandb 关闭时它本来就是 no-op，这里替换成
   追加写 `metrics.jsonl`，每步一行，含 loss / grad_norm / llm_grad_norm /
   mem_enc_norm / param_norm 的十进制与 `float.hex()` 双精度（hex 供未来 bitwise
   对比——`Step N:` 打印的 4 位小数不够用），外加墙钟 `wall_time`（顺便当逐步计时）。
   需要 `--log-interval 1` 才是真正逐步（interval>1 时记录的是区间均值）。

2. `train._checkpoints.save_state` → 参数校验和记录器：不落任何权重文件（单个
   checkpoint 约 14 GB），改为遍历 `state.params` 与 `state.ema_params` 的 pytree，
   逐叶子 `sha256(device_get(leaf).tobytes())`，写 `param_checksums.jsonl`。
   `--save-interval` 因此成为校验和间隔。逐叶子摘要为将来「定位哪个模块开始分叉」
   铺路（见同目录 README.md 的三级比较协议）。

记录文件落在环境变量 `BENCH_RECORD_DIR` 指定的目录（由驱动脚本传入）。

同时做几道 fail-loud 护栏，防止这个入口被误当成正式训练启动器。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))   # train.py 在 scripts/ 下，不是包

import train as _train  # noqa: E402

import jax  # noqa: E402

import mme_vla_suite.training.config as _config  # noqa: E402

_MAX_BENCH_STEPS = 500
_EXPECTED_HISTORY_CONFIG = "perceptual-framesamp-context.yaml"


def _record_dir() -> pathlib.Path:
    raw = os.environ.get("BENCH_RECORD_DIR")
    if not raw:
        raise ValueError("必须设置 BENCH_RECORD_DIR 指定记录输出目录（由驱动脚本传入）")
    d = pathlib.Path(raw)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _install_metrics_recorder(record_dir: pathlib.Path) -> None:
    """把 train 模块里的 wandb.log 换成 metrics.jsonl 记录器。

    train.py 里对 wandb.log 有两处调用：step 0 的 camera_views（wandb.Image 列表，
    非标量，跳过）与每个 log_interval 的 reduced_info（全标量，逐键记录）。
    """
    metrics_path = record_dir / "metrics.jsonl"
    real_log = _train.wandb.log

    def recording_log(data, step=None, **kwargs):
        row: dict = {"step": int(step) if step is not None else None,
                     "wall_time": time.time()}
        n_scalar = 0
        for k, v in data.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue           # 跳过 camera_views 这类非标量条目
            row[k] = {"dec": fv, "hex": fv.hex()}
            n_scalar += 1
        if n_scalar:
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return real_log(data, step=step, **kwargs)   # disabled 模式下是 no-op

    _train.wandb.log = recording_log


def _install_checksum_recorder(record_dir: pathlib.Path) -> None:
    """把 train 模块里的 _checkpoints.save_state 换成参数校验和记录器（不落权重）。"""
    checksums_path = record_dir / "param_checksums.jsonl"

    def checksum_state(checkpoint_manager, state, data_loader, step):
        del checkpoint_manager, data_loader
        t0 = time.time()
        per_leaf: dict[str, str] = {}
        trees = {"params": state.params}
        if state.ema_params is not None:
            trees["ema_params"] = state.ema_params
        for tree_name, tree in trees.items():
            flat, _ = jax.tree_util.tree_flatten_with_path(tree)
            for path, leaf in flat:
                key = tree_name + jax.tree_util.keystr(path)
                arr = jax.device_get(leaf)
                h = hashlib.sha256()
                h.update(str(arr.dtype).encode())
                h.update(str(arr.shape).encode())
                h.update(arr.tobytes())
                per_leaf[key] = h.hexdigest()
        g = hashlib.sha256()
        for key in sorted(per_leaf):
            g.update(f"{key}:{per_leaf[key]}\n".encode())
        row = {
            "step": int(step),
            "wall_time": time.time(),
            "checksum_seconds": round(time.time() - t0, 3),
            "n_leaves": len(per_leaf),
            "global_digest": g.hexdigest(),
            "per_leaf": per_leaf,
        }
        with checksums_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"\n[bench] step {step}: 参数校验和 {row['global_digest'][:16]}… "
              f"({len(per_leaf)} 叶子, 耗时 {row['checksum_seconds']}s)")

    _train._checkpoints.save_state = checksum_state


def main() -> None:
    config = _config.cli()
    if config.num_train_steps > _MAX_BENCH_STEPS:
        raise ValueError(
            f"bench 入口只允许 ≤{_MAX_BENCH_STEPS} steps（当前 {config.num_train_steps}）；"
            f"正式训练请用 scripts/train.py"
        )
    if config.wandb_enabled:
        raise ValueError("bench 必须关闭 wandb（--no-wandb-enabled）")
    if config.overwrite or config.resume:
        raise ValueError("bench 禁止 overwrite / resume——避免误清已有 run 目录")
    if not config.model.use_history:
        raise ValueError("bench 必须启用 --model.use-history")
    if config.model.history_config != _EXPECTED_HISTORY_CONFIG:
        raise ValueError(f"bench 的 history_config 必须是 {_EXPECTED_HISTORY_CONFIG}")
    if config.batch_size % max(1, config.fsdp_devices):
        raise ValueError(
            f"batch_size {config.batch_size} 必须能被 fsdp_devices {config.fsdp_devices} 整除"
        )
    if config.log_interval != 1:
        raise ValueError("bench 必须 --log-interval 1，否则 metrics.jsonl 不是逐步记录")

    # fail-loud：monkeypatch 依赖 train.py 当前的调用点，train.py 变了这里要立刻炸
    src = inspect.getsource(_train.main)
    if "wandb.log(reduced_info" not in src or "_checkpoints.save_state(" not in src:
        raise RuntimeError("train.main 源码中找不到预期的 wandb.log/_checkpoints.save_state 调用点，"
                           "monkeypatch 前提失效，请检查 train.py 是否已改动")

    record_dir = _record_dir()
    _install_metrics_recorder(record_dir)
    _install_checksum_recorder(record_dir)
    _train.main(config)


if __name__ == "__main__":
    main()
