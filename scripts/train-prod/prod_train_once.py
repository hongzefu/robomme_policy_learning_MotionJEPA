#!/usr/bin/env python3
"""正式训练启动器：只调一次 `train.main()`，保留 checkpoint 保存（v1-prod-trend-10h）。

**为什么必须有这个文件**：`scripts/train.py` 的 `__main__` 是

    main(_config.cli(), tentative_run=True)   # 预热 12 个 optimizer step
    time.sleep(20)
    main(_config.cli())                        # 完整训练

第一次调用里 `initialize_checkpoint_dir` 已经把 checkpoint 目录建出来了，第二次进来
在默认 `overwrite=False` / `resume=False` 下必然 `FileExistsError`——**照现状直接跑
`scripts/train.py` 必崩**（仓库已登记为「正式入口的可运行性阻断」；v1 全部 e2e 走 bench
入口，所以从没撞上过）。本入口只调一次 `train.main(config)`、不带 `tentative_run`，
既绕开该阻断，也满足用户「不要 tentative 预热轮」的要求。**`scripts/train.py` 一字不改。**

**与 `scripts/smoke-local/bench_train_steps.py` 的关键区别**：bench 把
`train._checkpoints.save_state` 换成了摘要记录器（`BENCH_CHECKSUM=0` 时是纯 no-op），
**全程一个权重都不落**——拿它跑正式训练会白跑几小时。本入口**绝不 patch save_state**，
checkpoint 照常保存。另外 bench 有 `_MAX_BENCH_STEPS=1200` 上限、强制关 wandb、
禁 overwrite/resume、锁死 history_config 等一组只对 bench 成立的护栏，本入口不照抄。

唯一的 monkeypatch 是**只读观测**的指标记录器：把 `train.wandb` 换成 `_WandbProxy`
（复用 bench 的实现），每 `log_interval` 步把标量追加写 `metrics.jsonl`，再原样转发给
真 wandb。`train.py` 里 `wandb` 是模块级全局名，替换 `train.wandb` 即可生效；
**不能直接 patch `wandb.log`**——`wandb.init()` 会重新赋值模块级 `log` 把 patch 盖掉
（bench 文件头记着这个 2026-08-24 实测踩过的坑）。该文件供 `analyze_gpu_util.py` 判读，
`train.py` 自身不写任何指标文件。

记录目录经环境变量 `PROD_RECORD_DIR` 传入——**不能走命令行**：`_config.cli()`（tyro）
会吃掉整个 `sys.argv[1:]` 且对未知参数直接报错退出，wrapper 自有参数混进 argv 必挂。
"""

from __future__ import annotations

import inspect
import json
import os
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))                  # train.py 在 scripts/ 下，不是包
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "smoke-local"))  # 复用 _WandbProxy / _CacheEventCounter

import train as _train  # noqa: E402

import mme_vla_suite.training.config as _config  # noqa: E402

# bench 模块的 module-level 只有 import 与常量、无副作用，`main()` 不会自动执行
from bench_train_steps import _CacheEventCounter, _WandbProxy  # noqa: E402


def _record_dir() -> pathlib.Path:
    raw = os.environ.get("PROD_RECORD_DIR")
    if not raw:
        raise ValueError("必须设置 PROD_RECORD_DIR 指定记录输出目录（由驱动脚本传入）")
    d = pathlib.Path(raw)
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"记录目录已存在且非空，拒绝覆盖: {d}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    config = _config.cli()

    # ── 护栏（fail-loud）──────────────────────────────────────────────
    # 本轮不做续跑（用户 2026-08-28 拍板）；且 checkpoint 只存 EMA 权重、不存优化器状态，
    # 续跑会丢 AdamW 动量并把 10000 步 warmup 从头再爬一遍，故一律禁用两个开关。
    if config.overwrite or config.resume:
        raise ValueError("本入口禁用 --overwrite / --resume：本轮不做续跑，"
                         "且续跑语义有损（丢优化器状态与 LR schedule 计数器）")
    if not config.model.use_history:
        raise ValueError("正式训练必须启用 --model.use-history")
    if config.batch_size % max(1, config.fsdp_devices):
        raise ValueError(
            f"batch_size {config.batch_size} 必须能被 fsdp_devices {config.fsdp_devices} 整除")
    # run 目录 fail-loud：initialize_checkpoint_dir 对已存在目录会 raise，但那时
    # 权重加载与 JIT 编译已经白跑了几分钟，提前拦掉（同 gl_e2e_fix.sbatch 的做法）
    ckpt_dir = config.checkpoint_dir
    if ckpt_dir.exists():
        raise FileExistsError(f"run 目录已存在，请换 --exp-name（AGENTS 6 禁止复用）: {ckpt_dir}")
    # monkeypatch 依赖 train.py 当前的调用点，train.py 变了这里要立刻炸
    src = inspect.getsource(_train.main)
    if "wandb.log(reduced_info" not in src:
        raise RuntimeError("train.main 源码中找不到预期的 wandb.log 调用点，"
                           "指标记录器前提失效，请检查 train.py 是否已改动")
    if "_checkpoints.save_state(" not in src:
        raise RuntimeError("train.main 源码中找不到 _checkpoints.save_state 调用点，"
                           "checkpoint 保存前提失效，请检查 train.py 是否已改动")

    record_dir = _record_dir()
    cache_counter = _CacheEventCounter()
    _train.wandb = _WandbProxy(_train.wandb, record_dir / "metrics.jsonl")

    print(f"[prod] run={config.exp_name} steps={config.num_train_steps} "
          f"batch={config.batch_size} workers={config.num_workers} "
          f"fsdp={config.fsdp_devices} log_interval={config.log_interval} "
          f"save_interval={config.save_interval} ckpt_dir={ckpt_dir}", flush=True)

    try:
        _train.main(config)   # 只调一次，不带 tentative_run
    finally:
        meta = {
            "argv": list(sys.argv),
            "monitoring_event_counts": cache_counter.counts,
            "entry": "prod_train_once",
            "tentative_run": False,
            "checkpoint_dir": str(ckpt_dir),
            "num_train_steps": config.num_train_steps,
            "log_interval": config.log_interval,
            "save_interval": config.save_interval,
        }
        with (record_dir / "run_meta.json").open("w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        try:
            _train.wandb.finish()
        except Exception as e:  # noqa: BLE001 —— 收尾失败不掩盖训练本身的结果
            print(f"[prod] wandb.finish() 失败（忽略）: {e}", flush=True)


if __name__ == "__main__":
    main()
