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
   checkpoint 约 14 GB），改为遍历**完整 TrainState**（params / ema_params /
   opt_state / step）的 pytree，逐叶子 `sha256(device_get(leaf).tobytes())`，写
   `param_checksums.jsonl`。`--save-interval` 因此成为校验和间隔。`global_digest`
   只覆盖 params+ema（与 2026-08-24 之前的旧记录同口径），`state_digest` 覆盖全部
   叶子（Adam 动量是「两条轨迹是否同一条」最灵敏的累积量，见 v1-gradient-baseline.md）。
   逐叶子摘要为将来「定位哪个模块开始分叉」铺路（见同目录 README.md 的三级比较协议）。

3. `train.init_train_state` → 包一层：初始化完成后立即对初始 TrainState 记一次
   step 0 摘要（「步 0 必记」，v1-gradient-baseline.md 产物清单）；train.main 的
   save 触发条件 `step % save_interval == 0 and step > start_step` 永远轮不到步 0。

4. `openpi.training.data_loader.TorchDataLoader.__iter__` → 输入摘要记录器：对
   collate 后、device_put 前的 host 侧 batch 逐键 `sha256(dtype‖shape‖bytes)`，写
   `batch_digests.jsonl`（步 0/1/2 + 每 SAVE_INTERVAL + 末步）。输出摘要把「输入变了」
   与「计算变了」混在一起且跨计算图失比；输入摘要与 XLA/缓存/驱动无关，跨 HLO 永远
   逐位可比，是 roadmap 改输入签名场景对拍 G0 的主判据。

另注册 jax.monitoring 事件监听（编译缓存命中/编译计数），训练结束时连同真实
`sys.argv` 写 `run_meta.json`——「这轮是热缓存还是冷编译」从口头猜测变成留档事实。

记录文件落在环境变量 `BENCH_RECORD_DIR` 指定的目录（由驱动脚本传入）。
性能口径开关（供 speed 链 run 使用，默认全开、保持现行为）：
`BENCH_CHECKSUM=0` 禁 TrainState 摘要（连同步 0），`BENCH_BATCH_DIGESTS=0` 禁输入摘要。

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
import jax.monitoring  # noqa: E402
import numpy as np  # noqa: E402

import openpi.training.data_loader as _openpi_dl  # noqa: E402

import mme_vla_suite.training.config as _config  # noqa: E402

_MAX_BENCH_STEPS = 600  # bottleneck-bench v2 修复验证需 600 步（用户 2026-08-24 指定）；上限仍远低于正式训练量级
_EXPECTED_HISTORY_CONFIG = "perceptual-framesamp-context.yaml"


def _record_dir() -> pathlib.Path:
    raw = os.environ.get("BENCH_RECORD_DIR")
    if not raw:
        raise ValueError("必须设置 BENCH_RECORD_DIR 指定记录输出目录（由驱动脚本传入）")
    d = pathlib.Path(raw)
    d.mkdir(parents=True, exist_ok=True)
    return d


class _WandbProxy:
    """替换 train 模块全局名 `wandb` 的代理：log 先记录再转发，其余属性透传。

    ⚠ 不能直接 patch `wandb.log`：train.main 里的 `wandb.init(mode="disabled")`
    会把 wandb 模块级的 `log` 重新赋值成 run 的 stub，把 patch 盖掉（2026-08-24
    b8 首跑实测踩过：训练 300 步全部正常、校验和 12 次齐全，metrics.jsonl 却一行
    没写）。代理对象让 train.py 的 `wandb.log` 查找永远先经过记录器，真 wandb
    模块随便改自己的属性都影响不到。
    """

    def __init__(self, real_wandb, metrics_path: pathlib.Path):
        self._real = real_wandb
        self._metrics_path = metrics_path

    def log(self, data, step=None, **kwargs):
        # train.py 两处调用：step 0 的 camera_views（wandb.Image 列表，非标量，
        # 跳过）与每个 log_interval 的 reduced_info（全标量，逐键记录）
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
    _train.wandb = _WandbProxy(_train.wandb, record_dir / "metrics.jsonl")


def _leaf_sha256(arr) -> str:
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _checksum_full_state(checksums_path: pathlib.Path, state, step: int) -> None:
    """完整 TrainState 摘要：params / ema_params / opt_state / step 全部叶子逐个 sha256。

    `global_digest` 只覆盖 params+ema（与旧记录同口径可续比），`state_digest`
    覆盖全部叶子——Adam 动量（opt_state）是最灵敏的累积量，缺了它基线就有永久盲区。
    """
    t0 = time.time()
    per_leaf: dict[str, str] = {}
    trees = {"params": state.params, "opt_state": state.opt_state, "step": state.step}
    if state.ema_params is not None:
        trees["ema_params"] = state.ema_params
    for tree_name, tree in trees.items():
        flat, _ = jax.tree_util.tree_flatten_with_path(tree)
        for path, leaf in flat:
            if leaf is None:
                continue
            key = tree_name + jax.tree_util.keystr(path)
            per_leaf[key] = _leaf_sha256(np.asarray(jax.device_get(leaf)))
    g = hashlib.sha256()   # 旧口径：仅 params+ema
    s = hashlib.sha256()   # 全量口径：全部叶子
    for key in sorted(per_leaf):
        line = f"{key}:{per_leaf[key]}\n".encode()
        s.update(line)
        if key.startswith(("params", "ema_params")):
            g.update(line)
    row = {
        "step": int(step),
        "wall_time": time.time(),
        "checksum_seconds": round(time.time() - t0, 3),
        "n_leaves": len(per_leaf),
        "global_digest": g.hexdigest(),
        "state_digest": s.hexdigest(),
        "per_leaf": per_leaf,
    }
    with checksums_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\n[bench] step {step}: TrainState 摘要 state={row['state_digest'][:16]}… "
          f"params+ema={row['global_digest'][:16]}… "
          f"({len(per_leaf)} 叶子, 耗时 {row['checksum_seconds']}s)")


def _install_checksum_recorder(record_dir: pathlib.Path, enabled: bool) -> None:
    """把 train 模块里的 _checkpoints.save_state 换成完整 TrainState 摘要记录器（不落权重）。

    enabled=False（BENCH_CHECKSUM=0，speed 链口径）时替换为纯 no-op：既不落 14 GB
    checkpoint，也不做任何 device_get 停顿。
    """
    checksums_path = record_dir / "param_checksums.jsonl"

    def checksum_state(checkpoint_manager, state, data_loader, step):
        del checkpoint_manager, data_loader
        if enabled:
            _checksum_full_state(checksums_path, state, int(step))

    _train._checkpoints.save_state = checksum_state


def _install_step0_checksum(record_dir: pathlib.Path) -> None:
    """包 train.init_train_state：初始化完成后立即记 step 0 完整 TrainState 摘要。

    train.main 的 save 触发条件（step % save_interval == 0 and step > start_step）
    永远轮不到步 0，而「步 0 必记」是 G0 产物清单的硬要求——它锚定两条轨迹的起点。
    """
    checksums_path = record_dir / "param_checksums.jsonl"
    orig_init = _train.init_train_state

    def init_and_checksum(config, init_rng, mesh, *, resume):
        train_state, state_sharding = orig_init(config, init_rng, mesh, resume=resume)
        jax.block_until_ready(train_state)
        _checksum_full_state(checksums_path, train_state, step=0)
        return train_state, state_sharding

    _train.init_train_state = init_and_checksum


def _install_batch_digest_recorder(record_dir: pathlib.Path, interval: int,
                                   max_step: int) -> None:
    """把 TorchDataLoader.__iter__ 换成带输入摘要的版本（host 侧、device_put 前）。

    重实现原 __iter__ 的循环（原版在 yield 前就把 batch 转成了 device array，包不进
    去），插入点是 collate 后、`make_array_from_process_local_data` 前的 numpy batch。
    记录步：0/1/2、每 interval 步（interval>0 时）、末步 max_step；train.main 在末步
    之后还会多取一个 batch（idx == max_step+1，取而不用），不记录。
    """
    digests_path = record_dir / "batch_digests.jsonl"

    def record(idx: int, batch) -> None:
        t0 = time.time()
        flat, _ = jax.tree_util.tree_flatten_with_path(batch)
        per_key = {}
        for path, leaf in flat:
            per_key[jax.tree_util.keystr(path)] = _leaf_sha256(np.asarray(leaf))
        g = hashlib.sha256()
        for key in sorted(per_key):
            g.update(f"{key}:{per_key[key]}\n".encode())
        row = {
            "step": idx,
            "wall_time": time.time(),
            "digest_seconds": round(time.time() - t0, 3),
            "n_keys": len(per_key),
            "batch_digest": g.hexdigest(),
            "per_key": per_key,
        }
        with digests_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def iter_with_digest(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                idx = num_items
                num_items += 1
                if idx <= max_step and (
                    idx in (0, 1, 2) or idx == max_step
                    or (interval > 0 and idx % interval == 0)
                ):
                    record(idx, batch)
                if self._sharding is not None:
                    yield jax.tree.map(
                        lambda x: jax.make_array_from_process_local_data(self._sharding, x),
                        batch)
                else:
                    yield jax.tree.map(_openpi_dl.torch.as_tensor, batch)

    _openpi_dl.TorchDataLoader.__iter__ = iter_with_digest


class _CacheEventCounter:
    """jax.monitoring 事件计数器：编译缓存命中/未中/编译请求从口头猜测变成留档事实。"""

    def __init__(self):
        self.counts: dict[str, int] = {}
        jax.monitoring.register_event_listener(self._on_event)

    def _on_event(self, event: str, **kwargs) -> None:
        del kwargs
        self.counts[event] = self.counts.get(event, 0) + 1


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
    if "init_train_state(" not in src:
        raise RuntimeError("train.main 源码中找不到 init_train_state 调用点，步 0 摘要前提失效")
    # 输入摘要重实现了 TorchDataLoader.__iter__ 的循环，原实现变了必须立刻炸
    dl_src = inspect.getsource(_openpi_dl.TorchDataLoader.__iter__)
    if ("make_array_from_process_local_data" not in dl_src
            or "num_items" not in dl_src or "StopIteration" not in dl_src):
        raise RuntimeError("TorchDataLoader.__iter__ 源码与输入摘要记录器的重实现假设不符，"
                           "请检查 openpi/training/data_loader.py 是否已改动")

    checksum_on = os.environ.get("BENCH_CHECKSUM", "1") != "0"
    digests_on = os.environ.get("BENCH_BATCH_DIGESTS", "1") != "0"

    record_dir = _record_dir()
    cache_counter = _CacheEventCounter()
    _install_metrics_recorder(record_dir)
    _install_checksum_recorder(record_dir, enabled=checksum_on)
    if checksum_on:
        _install_step0_checksum(record_dir)
    if digests_on:
        _install_batch_digest_recorder(record_dir, interval=config.save_interval,
                                       max_step=config.num_train_steps - 1)
    try:
        _train.main(config)
    finally:
        # 真实 argv 与编译缓存事件计数——驱动脚本收官时并进 env.json
        meta = {
            "argv": list(sys.argv),
            "monitoring_event_counts": cache_counter.counts,
            "bench_checksum_enabled": checksum_on,
            "bench_batch_digests_enabled": digests_on,
        }
        with (record_dir / "run_meta.json").open("w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
