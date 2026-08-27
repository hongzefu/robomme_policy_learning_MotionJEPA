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
   collate 后、device_put 前的 host 侧 batch 逐键记**双口径**摘要（schema 2，P1b）：
   - raw：`sha256(dtype‖shape‖bytes)`——「输入应逐字节不变」场景的主判据；
   - canonical：浮点键升到 float32 后 `sha256("f32"‖shape‖bytes)`（dtype 不入摘要域），
     非浮点键与 raw 同——跨 dtype 对拍（G1 vs G0）的输入侧判据：类型变化被抹平、
     数值变化仍逃不掉（无谓升档的 f64←f32 降回 f32 逐位还原，canonical 相等）。
   写 `batch_digests.jsonl`（步 0/1/2 + 每摘要间隔 + 附加步 + 末步），每行附该步
   样本 index（见下）。输出摘要把「输入变了」与「计算变了」混在一起且跨计算图失比；
   输入摘要与 XLA/缓存/驱动无关，跨 HLO 永远逐位可比。

5. `BatchSampler.sampler` → index 序列记录器（P1b）：包一层记录 torch DataLoader
   主进程侧抽出的全部样本 index（抽取顺序与 batch 交付顺序一致，prefetch 只提前
   不重排），收官写 `index_sequence.json`（全序列 + sha256）。「同一批样本、同一
   顺序」由此独立可证，是跨 dtype 对拍的另一半输入侧判据。

6. `BENCH_DUMP_IDX=1` → batch_sampler 层 index 记录器（S0'，v2 计划 C.1 端到端旁证）：
   包 `mme_vla_suite.training.dataloader.create_data_loader`，拿到 loader 后、首次
   `iter()` 之前把 torch DataLoader 的 `batch_sampler` 换成 `_IdxProbe`（必须
   `object.__setattr__` 绕过 DataLoader 初始化后的 `__setattr__` 赋值守卫），每个
   batch 的 index 列表追加写 `idx_seq.jsonl` 后原样 yield。与第 5 条互补：本层记录
   batch_sampler 的真实产出（含 prefetch 超前），比对时取前 N 条（N=实际消费步数），
   尾部允许至多 prefetch_factor × num_workers 条超前记录。默认 0（关闭，现行为不变）。

另注册 jax.monitoring 事件监听（编译缓存命中/编译计数），训练结束时连同真实
`sys.argv` 写 `run_meta.json`——「这轮是热缓存还是冷编译」从口头猜测变成留档事实。

记录文件落在环境变量 `BENCH_RECORD_DIR` 指定的目录（由驱动脚本传入）。
性能口径开关（供 speed 链 run 使用，默认全开、保持现行为）：
`BENCH_CHECKSUM=0` 禁 TrainState 摘要（连同步 0），`BENCH_BATCH_DIGESTS=0` 禁输入摘要。

摘要步集合的记录器侧自选（P1b，供「附加摘要步」与 TrainState 数组落盘使用）：
- `BENCH_DIGEST_INTERVAL`：有效摘要间隔；设了它时驱动脚本会给 train 传
  `--save-interval 1`（train 的 save 分支只调 save_state 一个函数，本入口已把它换成
  记录器，每步空调用零开销），由记录器按本间隔 + 附加步自选；未设时沿用
  config.save_interval（现行为，train 侧自身过滤）。
- `BENCH_EXTRA_DIGEST_STEPS`：逗号分隔附加摘要步（如 `299`——对齐旧 300 步基线的
  末步摘要，使前缀对拍每个旧摘要步都有逐位对应点）。
- `BENCH_STATE_DUMP_STEPS` + `BENCH_STATE_DUMP_DIR`：在指定摘要步把完整 TrainState
  数组按位落盘（每步一对 `state_step_<N>.json`/`.bin`，逐叶 dtype/shape/offset/sha256
  进 meta；npy/npz 会丢 bf16 类型故用裸字节容器）——G1 对拍逐叶数值裁决的参照
  （compare_baseline.py 的 --state-arrays-*）。单步全量约 14 GB，只在明确指定的步落。

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

_MAX_BENCH_STEPS = 1200  # G0b 基线升级为 1000 步（用户 2026-08-26 指定）；上限仍远低于正式训练量级
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


def _canonical_sha256(arr: np.ndarray) -> str:
    """canonical 数值口径（P1b）：浮点叶子升到 float32 后按位哈希，dtype 不入摘要域。

    跨 dtype 对拍（G1 vs G0）用：类型变化被抹平、数值变化仍逃不掉——无谓升档的
    f64（原值是 f32 的精确升档）降回 f32 逐位还原，canonical 相等；真数值差异则
    在 f32 分辨率下必然失配。非浮点叶子（int/bool/uint）保持 raw 口径（dtype 入域）
    ——它们的 dtype 变化本身就是 bug，不该被抹平。
    kind 'f' 是标准浮点；'V' 覆盖 ml_dtypes 自定义浮点（bfloat16 等），若真是
    structured dtype 则 astype 会当场抛错（fail-loud，不静默算错口径）。
    """
    h = hashlib.sha256()
    if arr.dtype.kind in "fV":
        a32 = arr.astype(np.float32)
        h.update(b"f32")
        h.update(str(a32.shape).encode())
        h.update(a32.tobytes())
    else:
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _parse_step_set(env_name: str) -> set[int]:
    raw = os.environ.get(env_name, "")
    return {int(s) for s in raw.split(",") if s.strip()}


def _make_digest_gate(config) -> "callable":
    """摘要步集合的记录器侧判定（P1b）。

    BENCH_DIGEST_INTERVAL 未设时恒真（train 侧按 config.save_interval 自身过滤，
    现行为不变）；设了时按「步 0 / 末步 / 间隔倍数 / 附加步」自选（此时驱动脚本
    给 train 传 --save-interval 1，让每步都轮到本记录器判定）。
    """
    interval_env = os.environ.get("BENCH_DIGEST_INTERVAL") or None  # 空串视同未设
    extra = _parse_step_set("BENCH_EXTRA_DIGEST_STEPS")
    last_step = config.num_train_steps - 1
    bad = {s for s in extra if not 0 <= s <= last_step}
    if bad:
        raise ValueError(f"BENCH_EXTRA_DIGEST_STEPS 越界（须在 0..{last_step}）: {sorted(bad)}")
    if interval_env is None:
        if extra:
            raise ValueError("BENCH_EXTRA_DIGEST_STEPS 需要配合 BENCH_DIGEST_INTERVAL"
                             "（驱动脚本 EXTRA_DIGEST_STEPS 路径会一并设置）")
        return lambda step: True
    interval = int(interval_env)

    def gate(step: int) -> bool:
        return (step == 0 or step == last_step or step in extra
                or (interval > 0 and step > 0 and step % interval == 0))
    return gate


def _state_dump_config(config) -> tuple[set[int], pathlib.Path | None]:
    """TrainState 数组落盘配置（P1b）：只在明确指定的摘要步落，单步全量约 14 GB。"""
    steps = _parse_step_set("BENCH_STATE_DUMP_STEPS")
    if not steps:
        return set(), None
    raw = os.environ.get("BENCH_STATE_DUMP_DIR")
    if not raw:
        raise ValueError("设置了 BENCH_STATE_DUMP_STEPS 就必须设置 BENCH_STATE_DUMP_DIR")
    last_step = config.num_train_steps - 1
    bad = {s for s in steps if not 0 <= s <= last_step}
    if bad:
        raise ValueError(f"BENCH_STATE_DUMP_STEPS 越界（须在 0..{last_step}）: {sorted(bad)}")
    d = pathlib.Path(raw)
    d.mkdir(parents=True, exist_ok=True)
    return steps, d


def _checksum_full_state(checksums_path: pathlib.Path, state, step: int,
                         dump_dir: pathlib.Path | None = None) -> None:
    """完整 TrainState 摘要：params / ema_params / opt_state / step 全部叶子逐个 sha256。

    `global_digest` 只覆盖 params+ema（与旧记录同口径可续比），`state_digest`
    覆盖全部叶子——Adam 动量（opt_state）是最灵敏的累积量，缺了它基线就有永久盲区。

    dump_dir 非 None 时（P1b）同趟把每个叶子的原始字节顺序写进
    `state_step_<step>.bin`，逐叶 dtype/shape/offset/nbytes/sha256 进
    `state_step_<step>.json`——与摘要同一次 device_get，不加第二趟停顿；meta 里的
    sha 与 per_leaf 完全同源，落盘产物可独立防腐。
    """
    t0 = time.time()
    per_leaf: dict[str, str] = {}
    dump_meta: dict[str, dict] = {}
    dump_f = None
    dump_offset = 0
    if dump_dir is not None:
        bin_path = dump_dir / f"state_step_{step}.bin"
        meta_path = dump_dir / f"state_step_{step}.json"
        if bin_path.exists() or meta_path.exists():
            raise FileExistsError(f"TrainState 落盘目标已存在, 拒绝覆盖: {bin_path}")
        dump_f = bin_path.open("wb")
    trees = {"params": state.params, "opt_state": state.opt_state, "step": state.step}
    if state.ema_params is not None:
        trees["ema_params"] = state.ema_params
    for tree_name, tree in trees.items():
        flat, _ = jax.tree_util.tree_flatten_with_path(tree)
        for path, leaf in flat:
            if leaf is None:
                continue
            key = tree_name + jax.tree_util.keystr(path)
            arr = np.asarray(jax.device_get(leaf))
            per_leaf[key] = _leaf_sha256(arr)
            if dump_f is not None:
                data = arr.tobytes()
                dump_f.write(data)
                dump_meta[key] = {"dtype": str(arr.dtype), "shape": list(arr.shape),
                                  "offset": dump_offset, "nbytes": len(data),
                                  "sha256": per_leaf[key]}
                dump_offset += len(data)
    if dump_f is not None:
        dump_f.close()
        with meta_path.open("w") as f:
            json.dump({"step": int(step), "schema": 1, "total_bytes": dump_offset,
                       "leaves": dump_meta}, f, indent=1, ensure_ascii=False)
        print(f"\n[bench] step {step}: TrainState 数组落盘 {dump_offset/2**30:.1f} GiB "
              f"→ {bin_path.name} ({len(dump_meta)} 叶子)")
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


def _install_checksum_recorder(record_dir: pathlib.Path, enabled: bool, gate,
                               dump_steps: set[int],
                               dump_dir: pathlib.Path | None) -> None:
    """把 train 模块里的 _checkpoints.save_state 换成完整 TrainState 摘要记录器（不落权重）。

    enabled=False（BENCH_CHECKSUM=0，speed 链口径）时替换为纯 no-op：既不落 14 GB
    checkpoint，也不做任何 device_get 停顿。gate 决定哪些步真正记摘要（P1b：驱动
    脚本走附加摘要步路径时给 train 传 --save-interval 1，每步都进到这里，由 gate
    自选；空调用零开销）。
    """
    checksums_path = record_dir / "param_checksums.jsonl"

    def checksum_state(checkpoint_manager, state, data_loader, step):
        del checkpoint_manager, data_loader
        step = int(step)
        if enabled and gate(step):
            _checksum_full_state(checksums_path, state, step,
                                 dump_dir=dump_dir if step in dump_steps else None)

    _train._checkpoints.save_state = checksum_state


def _install_step0_checksum(record_dir: pathlib.Path, dump_steps: set[int],
                            dump_dir: pathlib.Path | None) -> None:
    """包 train.init_train_state：初始化完成后立即记 step 0 完整 TrainState 摘要。

    train.main 的 save 触发条件（step % save_interval == 0 and step > start_step）
    永远轮不到步 0，而「步 0 必记」是 G0 产物清单的硬要求——它锚定两条轨迹的起点。
    """
    checksums_path = record_dir / "param_checksums.jsonl"
    orig_init = _train.init_train_state

    def init_and_checksum(config, init_rng, mesh, *, resume):
        train_state, state_sharding = orig_init(config, init_rng, mesh, resume=resume)
        jax.block_until_ready(train_state)
        _checksum_full_state(checksums_path, train_state, step=0,
                             dump_dir=dump_dir if 0 in dump_steps else None)
        return train_state, state_sharding

    _train.init_train_state = init_and_checksum


class _LoggingSampler:
    """包 BatchSampler.sampler：记录主进程侧抽出的样本 index 全序列（P1b）。

    torch DataLoader 的 index 抽取发生在主进程（worker 只按派发的 index 取数），
    且 _MultiProcessingDataLoaderIter 按抽取顺序交付 batch——prefetch 只提前不重排，
    故本序列第 k 个 batch_size 段就是交付的第 k 个 batch 的样本 index。
    序列尾部可能含已抽取未交付的 prefetch 余量，比对时取前 steps×batch 个即可。
    """

    def __init__(self, inner, log: list):
        self._inner = inner
        self._log = log

    def __iter__(self):
        for i in self._inner:
            self._log.append(int(i))
            yield i

    def __len__(self):
        return len(self._inner)


def _install_batch_digest_recorder(record_dir: pathlib.Path, interval: int,
                                   extra_steps: set[int], max_step: int):
    """把 TorchDataLoader.__iter__ 换成带输入摘要的版本（host 侧、device_put 前）。

    重实现原 __iter__ 的循环（原版在 yield 前就把 batch 转成了 device array，包不进
    去），插入点是 collate 后、`make_array_from_process_local_data` 前的 numpy batch。
    记录步：0/1/2、每 interval 步（interval>0 时）、附加步（P1b）、末步 max_step；
    train.main 在末步之后还会多取一个 batch（idx == max_step+1，取而不用），不记录。
    每行 raw + canonical 双口径（schema 2）并附该步样本 index。

    返回收官回调：把 index 全序列写 index_sequence.json（main 的 finally 调）。
    """
    digests_path = record_dir / "batch_digests.jsonl"
    index_log: list[int] = []

    def record(idx: int, batch) -> None:
        t0 = time.time()
        flat, _ = jax.tree_util.tree_flatten_with_path(batch)
        per_key = {}
        per_key_canonical = {}
        batch_size = None
        for path, leaf in flat:
            arr = np.asarray(leaf)
            key = jax.tree_util.keystr(path)
            per_key[key] = _leaf_sha256(arr)
            per_key_canonical[key] = _canonical_sha256(arr)
            if batch_size is None and arr.ndim > 0:
                batch_size = int(arr.shape[0])
        g = hashlib.sha256()
        c = hashlib.sha256()
        for key in sorted(per_key):
            g.update(f"{key}:{per_key[key]}\n".encode())
            c.update(f"{key}:{per_key_canonical[key]}\n".encode())
        # 本 batch 的样本 index：抽取顺序 == 交付顺序，第 idx 段即本步样本；
        # 能走到这里说明该段必已抽出（没有 index 就产不出 batch），断言防口径漂移
        sample_indices = None
        if batch_size is not None and len(index_log) >= (idx + 1) * batch_size:
            sample_indices = index_log[idx * batch_size:(idx + 1) * batch_size]
        row = {
            "schema": 2,
            "step": idx,
            "wall_time": time.time(),
            "digest_seconds": round(time.time() - t0, 3),
            "n_keys": len(per_key),
            "batch_digest": g.hexdigest(),
            "batch_digest_canonical": c.hexdigest(),
            "sample_indices": sample_indices,
            "per_key": per_key,
            "per_key_canonical": per_key_canonical,
        }
        with digests_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def iter_with_digest(self):
        # 包 index 记录器：BatchSampler 每次 __iter__ 都 iter(self.sampler)，普通
        # 属性替换即可生效（DataLoader 本体的 batch_sampler 属性有 __setattr__ 护栏，
        # 不动它）；guard 防重复包（train.main 只建一次 loader，防御式仍加）
        bs = getattr(self._data_loader, "batch_sampler", None)
        if bs is None or not hasattr(bs, "sampler"):
            raise RuntimeError("torch DataLoader 无 batch_sampler.sampler，"
                               "index 序列记录前提失效，请检查 DataLoader 构造")
        if not isinstance(bs.sampler, _LoggingSampler):
            bs.sampler = _LoggingSampler(bs.sampler, index_log)
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
                    idx in (0, 1, 2) or idx == max_step or idx in extra_steps
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

    def finalize() -> None:
        if not index_log:
            return
        seq_sha = hashlib.sha256(json.dumps(index_log).encode()).hexdigest()
        with (record_dir / "index_sequence.json").open("w") as f:
            json.dump({"schema": 1, "n": len(index_log),
                       "note": "主进程抽取顺序==交付顺序; 尾部可含 prefetch 余量, "
                               "比对取前 steps×batch 个",
                       "indices_sha256": seq_sha, "indices": index_log}, f)
        print(f"[bench] index 序列 {len(index_log)} 个已写 index_sequence.json "
              f"(sha256 {seq_sha[:16]}…)")

    return finalize


class _IdxProbe:
    """batch_sampler 层 index 记录器（S0'，v2 计划 C.1 端到端旁证；BENCH_DUMP_IDX=1 启用）。

    包 torch DataLoader 的 batch_sampler：每个 batch 的 index 列表追加写
    idx_seq.jsonl 后原样 yield。`sampler` 属性读写转发给内层 BatchSampler——
    输入摘要记录器（_install_batch_digest_recorder）要在 bs.sampler 上包
    _LoggingSampler，不转发的话赋值会落在本包装器上、内层 BatchSampler 照旧
    迭代旧 sampler，index_sequence.json 就静默记不到——两个记录器必须可叠加。
    """

    def __init__(self, inner, path: pathlib.Path):
        self._inner = inner
        self._path = path
        self._n = 0
        self._f = None

    @property
    def sampler(self):
        return self._inner.sampler

    @sampler.setter
    def sampler(self, value):
        self._inner.sampler = value

    def __iter__(self):
        if self._f is None:
            self._f = self._path.open("a")
        for batch_indices in self._inner:
            self._f.write(json.dumps(
                {"batch": self._n, "indices": [int(i) for i in batch_indices]}) + "\n")
            self._f.flush()
            self._n += 1
            yield batch_indices

    def __len__(self):
        return len(self._inner)


def _install_idx_probe(record_dir: pathlib.Path) -> None:
    """包 create_data_loader：返回 loader 前把 torch DataLoader 的 batch_sampler 换成 _IdxProbe。

    安装点天然在首次 iter() 之前（train.main 先 create 后 iter）。替换必须走
    object.__setattr__——torch DataLoader 初始化后直接赋值 batch_sampler 会被
    __setattr__ 守卫拒绝（ValueError，torch 2.7.1 实测）；绕道后 _index_sampler
    正确返回包装器，persistent_workers 下跨 epoch 持续生效（v2 计划 C.1）。
    """
    dl_mod = _train._data_loader
    orig_create = dl_mod.create_data_loader

    def create_and_probe(*args, **kwargs):
        loader = orig_create(*args, **kwargs)
        torch_dl = getattr(getattr(loader, "_data_loader", None), "torch_loader", None)
        if torch_dl is None or getattr(torch_dl, "batch_sampler", None) is None:
            raise RuntimeError(
                "BENCH_DUMP_IDX: loader._data_loader.torch_loader.batch_sampler 结构与"
                "预期不符，请检查 mme_vla_suite/training/dataloader.py 是否已改动")
        object.__setattr__(torch_dl, "batch_sampler",
                           _IdxProbe(torch_dl.batch_sampler,
                                     record_dir / "idx_seq.jsonl"))
        return loader

    dl_mod.create_data_loader = create_and_probe


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

    # P1b：摘要步集合的记录器侧自选 + TrainState 数组落盘配置
    gate = _make_digest_gate(config)
    extra_steps = _parse_step_set("BENCH_EXTRA_DIGEST_STEPS")
    digest_interval_env = os.environ.get("BENCH_DIGEST_INTERVAL") or None  # 空串视同未设
    digest_interval = (int(digest_interval_env) if digest_interval_env
                       else config.save_interval)
    dump_steps, dump_dir = _state_dump_config(config)
    if dump_steps and not checksum_on:
        raise ValueError("BENCH_STATE_DUMP_STEPS 需要 BENCH_CHECKSUM=1（落盘与摘要同趟）")
    not_digested = {s for s in dump_steps if not gate(s)}
    if not_digested:
        raise ValueError(f"BENCH_STATE_DUMP_STEPS 含非摘要步 {sorted(not_digested)}——"
                         f"落盘只在摘要步进行（同趟 device_get、sha 同源可防腐）")
    if dump_steps and digest_interval_env is None:
        # 记录器侧 gate 恒真、真实过滤在 train 侧（step%save_interval），落盘步不在
        # 该调度上会被静默跳过——必须 fail-loud
        last = config.num_train_steps - 1
        iv = config.save_interval
        missed = {s for s in dump_steps
                  if s not in (0, last) and not (iv > 0 and s > 0 and s % iv == 0)}
        if missed:
            raise ValueError(f"BENCH_STATE_DUMP_STEPS {sorted(missed)} 不在 train 侧 "
                             f"save 调度（间隔 {iv}）上且未设 BENCH_DIGEST_INTERVAL，"
                             f"会被静默跳过；请经驱动脚本 EXTRA_DIGEST_STEPS 路径启用")

    record_dir = _record_dir()
    cache_counter = _CacheEventCounter()
    _install_metrics_recorder(record_dir)
    _install_checksum_recorder(record_dir, enabled=checksum_on, gate=gate,
                               dump_steps=dump_steps, dump_dir=dump_dir)
    if checksum_on:
        _install_step0_checksum(record_dir, dump_steps=dump_steps, dump_dir=dump_dir)
    finalize_digests = None
    if digests_on:
        finalize_digests = _install_batch_digest_recorder(
            record_dir, interval=digest_interval, extra_steps=extra_steps,
            max_step=config.num_train_steps - 1)
    dump_idx_on = os.environ.get("BENCH_DUMP_IDX", "0") == "1"
    if dump_idx_on:
        # fail-loud：安装点依赖 train.main 里的 create_data_loader 调用，变了立刻炸
        if "create_data_loader(" not in src:
            raise RuntimeError("train.main 源码中找不到 create_data_loader 调用点，"
                               "BENCH_DUMP_IDX 安装前提失效，请检查 train.py 是否已改动")
        _install_idx_probe(record_dir)
    try:
        _train.main(config)
    finally:
        if finalize_digests is not None:
            finalize_digests()
        # 真实 argv 与编译缓存事件计数——驱动脚本收官时并进 env.json
        meta = {
            "argv": list(sys.argv),
            "monitoring_event_counts": cache_counter.counts,
            "bench_checksum_enabled": checksum_on,
            "bench_batch_digests_enabled": digests_on,
            "bench_dump_idx_enabled": dump_idx_on,
            "digest_interval_effective": digest_interval,
            "extra_digest_steps": sorted(extra_steps),
            "state_dump_steps": sorted(dump_steps),
        }
        with (record_dir / "run_meta.json").open("w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
