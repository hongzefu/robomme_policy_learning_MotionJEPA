#!/usr/bin/env python3
"""实验 3：compute-only 步时基准入口（喂同一个缓存 batch 反复训练）。

目的：测 4×A40、batch 64 下训练步的**纯计算耗时**（把 NFS/dataloader 从流水线里
摘掉）。它决定了 NFS 的供给需求：需求 MB/s = 每步字节（b64 ≈ 1.20 GB）÷ 步时。
与实验 1（dataloader-only 供给侧）合起来即可判定「4 卡 b64 是否有 NFS 瓶颈」。

结构复制 scripts/smoke-local/bench_train_steps.py（训练循环一行不改），三处 monkeypatch：
1. `train.wandb` → _WandbProxy：逐步写 metrics.jsonl（wall_time 差即步时；
   不能直接 patch wandb.log——wandb.init(mode="disabled") 会盖掉，见 smoke-local 实测）。
2. `train._data_loader.create_data_loader` → 包装器：调原函数拿真 loader，取第一个
   batch（真读一次 NFS ≈1.2 GB，稳态统计会剔除），之后无限 yield 同一个 batch
   （该 batch 已按 data_sharding 放置，直接可喂 ptrain_step）。
3. `train._checkpoints.save_state` → no-op（只写一行标记；本实验只测步时，
   不做 14 GB 参数校验和，也不落任何权重）。
"""

from __future__ import annotations

import inspect
import json
import os
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import train as _train  # noqa: E402

import mme_vla_suite.training.config as _config  # noqa: E402

_MAX_BENCH_STEPS = 200
_EXPECTED_HISTORY_CONFIG = "perceptual-framesamp-context.yaml"


class _WandbProxy:
    """同 smoke-local/bench_train_steps.py：log 先记录再转发，其余属性透传。"""

    def __init__(self, real_wandb, metrics_path: pathlib.Path):
        self._real = real_wandb
        self._metrics_path = metrics_path

    def log(self, data, step=None, **kwargs):
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
        return self._real.log(data, step=step, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _RepeatFirstBatchLoader:
    """取真 loader 的第一个 batch 后无限重复；data_config 透传。"""

    def __init__(self, real_loader):
        self._real_loader = real_loader
        self._batch = None

    def data_config(self):
        return self._real_loader.data_config()

    def __iter__(self):
        if self._batch is None:
            t0 = time.time()
            self._batch = next(iter(self._real_loader))
            print(f"\n[bench] 首个 batch 已从 NFS 读入并缓存（耗时 {time.time()-t0:.1f}s），"
                  f"之后所有步复用同一 batch，不再碰 dataloader/NFS")
        while True:
            yield self._batch


def main() -> None:
    config = _config.cli()
    if config.num_train_steps > _MAX_BENCH_STEPS:
        raise ValueError(f"compute-only 入口只允许 ≤{_MAX_BENCH_STEPS} steps"
                         f"（当前 {config.num_train_steps}）")
    if config.wandb_enabled:
        raise ValueError("必须关闭 wandb（--no-wandb-enabled）")
    if config.overwrite or config.resume:
        raise ValueError("禁止 overwrite / resume")
    if not config.model.use_history:
        raise ValueError("必须启用 --model.use-history")
    if config.model.history_config != _EXPECTED_HISTORY_CONFIG:
        raise ValueError(f"history_config 必须是 {_EXPECTED_HISTORY_CONFIG}")
    if config.log_interval != 1:
        raise ValueError("必须 --log-interval 1，否则步时不是逐步记录")

    src = inspect.getsource(_train.main)
    for anchor in ("wandb.log(reduced_info", "_checkpoints.save_state(",
                   "_data_loader.create_data_loader("):
        if anchor not in src:
            raise RuntimeError(f"train.main 源码缺少预期锚点 {anchor!r}，monkeypatch 前提失效")

    record_dir = pathlib.Path(os.environ["BENCH_RECORD_DIR"])
    record_dir.mkdir(parents=True, exist_ok=True)

    _train.wandb = _WandbProxy(_train.wandb, record_dir / "metrics.jsonl")

    real_create = _train._data_loader.create_data_loader

    def create_repeat_loader(*args, **kwargs):
        return _RepeatFirstBatchLoader(real_create(*args, **kwargs))

    _train._data_loader.create_data_loader = create_repeat_loader

    def noop_save_state(checkpoint_manager, state, data_loader, step):
        del checkpoint_manager, state, data_loader
        with (record_dir / "save_state_calls.jsonl").open("a") as f:
            f.write(json.dumps({"step": int(step), "wall_time": time.time(),
                                "action": "noop"}) + "\n")

    _train._checkpoints.save_state = noop_save_state

    _train.main(config)


if __name__ == "__main__":
    main()
