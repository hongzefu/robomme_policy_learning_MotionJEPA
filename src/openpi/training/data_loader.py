from collections.abc import Iterator, Sequence
import atexit
import ctypes
import logging
import multiprocessing
import os
import queue
import sys
import threading
import typing
import weakref
from typing import Literal, Protocol, SupportsIndex, TypeVar
import time

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms
import time

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __getitems__(self, indices):
        """torch DataLoader 的批取数钩子（_MapDatasetFetcher 优先调用它，再把返回值交给 collate_fn）。

        普通数据集：逐样本 ``self[i]``，与 torch 默认 fetcher 完全相同。
        wan_raw9（底层数据集带 ``raw9_motion_shape``，S5-L1）：整批 motion_emb (B,720,16384) f32（每批约 6 GB）
        先在共享内存里一次分配，逐样本 ``getitem_into`` 把表行直接写进第 j 行，transforms 照常逐样本跑；
        返回 ``_PreCollated``，``_collate_fn_shm`` 只 stack 其余键、motion_emb 直接用这块缓冲——省掉逐样本 47 MB
        分配、np.stack 整批拷贝与送进 DataLoader 队列时再拷进共享内存这三遍 6 GB 内存流量，字节不变。
        """
        shape = getattr(self._dataset, "raw9_motion_shape", None)
        if shape is None:
            return [self[i] for i in indices]
        tensor, buf = _shared_empty((len(indices),) + tuple(shape), np.dtype(np.float32))
        items = []
        for j, i in enumerate(indices):
            item = self._transform(self._dataset.getitem_into(i, buf[j]))
            m = item["motion_emb"]
            if not (isinstance(m, np.ndarray) and m.dtype == buf.dtype and m.shape == buf.shape[1:]
                    and m.__array_interface__["data"][0] == buf[j].__array_interface__["data"][0]):
                # transforms 若换了对象（当前链路不会），按 dtype/shape 严格一致拷回槽位，否则显式报错
                np.copyto(buf[j], m, casting="no")
            item["motion_emb"] = None            # 占位：其余键照常 collate，motion_emb 由 _PreCollated 交付
            items.append(item)
        return _PreCollated(items, "motion_emb", tensor)

    def __len__(self) -> int:
        return len(self._dataset)


class _PreCollated:
    """``TransformedDataset.__getitems__`` 交给 ``_collate_fn_shm`` 的批：items（某键已置 None）+ 该键已拼好的共享 tensor。"""

    def __init__(self, items, key, tensor):
        self.items = items
        self.key = key
        self.tensor = tensor


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        prefetch_factor: int | None = None,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
            prefetch_factor: 每 worker 预取批数；None 时沿用 torch 默认（2），只在 num_workers > 0 时传入。
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn_shm,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            **({"prefetch_factor": prefetch_factor} if (prefetch_factor is not None and num_workers > 0) else {}),
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        # S5-L1：取批（收 worker 的共享内存批、对新映射页预建页表 MADV_POPULATE_READ）放到后台取批线程，与训练主循环重叠；
        # 主线程只做 make_array（H2D）。用完的主机批交给释放线程丢掉最后一个引用（6 GB 共享内存批的 munmap + 归还
        # tmpfs 页实测约 0.6–0.8 s，原先落在主线程下一次 next() 里）。批的顺序与字节和原来完全相同。
        # 收尾：迭代器 close()（耗尽 / 被丢弃 / 进程退出 atexit）时停两个线程、先关底层 torch 迭代器再 join，
        # 不在解释器退出阶段留下仍在调 torch / jax 的线程。
        return _PrefetchIter(self)


_BG_END = object()
_JOIN_TIMEOUT_S = 60.0


class _PrefetchIter:
    """``TorchDataLoader.__iter__`` 的迭代器：后台取批线程 + 主线程 make_array + 释放线程。

    取批线程（_run，原 __iter__ 的 epoch 循环）：``next(torch 迭代器)`` → ``_from_shared_torch`` → 预建页表 → 容量 1 的队列。
    ``close()``：置停止标志 → 关底层 torch 迭代器（多进程时 ``_shutdown_workers``，让阻塞在 ``next`` 里的取批线程醒来）
    → join 取批线程 → 关并 join 释放线程。耗尽、异常、``__del__`` 与创建时登记的 atexit 都走 ``close()``。
    """

    def __init__(self, loader: "TorchDataLoader"):
        self._loader = loader
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._torch_iter = None
        self._closed = False
        self._releaser = _Releaser()
        self._thread = threading.Thread(target=self._run, name="TorchDataLoader-host-prefetch", daemon=True)
        self._thread.start()
        # atexit 在创建时登记（晚于 jax 后端初始化登记的清理 → LIFO 下先于它执行）；弱引用，不延长迭代器寿命
        ref = weakref.ref(self)
        self._atexit = lambda: (lambda it: it.close() if it is not None else None)(ref())
        atexit.register(self._atexit)

    # ── 取批线程 ──
    def _put(self, x) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(x, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self):
        loader = self._loader
        num_items = 0
        try:
            while not self._stop.is_set():
                self._torch_iter = iter(loader._data_loader)
                while not self._stop.is_set():
                    if loader._num_batches is not None and num_items >= loader._num_batches:
                        self._put(_BG_END)
                        return
                    try:
                        batch = next(self._torch_iter)
                    except StopIteration:
                        break  # We've exhausted the dataset. Create a new iterator and start over.
                    batch = _from_shared_torch(batch)
                    num_items += 1
                    _populate_tree(batch)
                    if not self._put(batch):
                        return
                    del batch
        except BaseException as e:  # noqa: BLE001 —— 交给消费侧抛出；close() 引发的收尾异常不再上抛
            if not self._stop.is_set():
                self._put(e)

    # ── 消费侧（主线程）──
    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        x = self._q.get()
        if x is _BG_END:
            self.close()
            raise StopIteration
        if isinstance(x, BaseException):
            self.close()
            raise x
        sharding = self._loader._sharding
        # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
        if sharding is not None:
            out = jax.tree.map(lambda a: jax.make_array_from_process_local_data(sharding, a), x)
        else:
            out = jax.tree.map(torch.as_tensor, x)
        self._releaser.hand_over(x)
        del x
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        ti = self._torch_iter
        if ti is not None and hasattr(ti, "_shutdown_workers"):
            ti._shutdown_workers()                      # 唤醒阻塞在 next(torch 迭代器) 的取批线程，并 join 全部 worker
            dl = self._loader._data_loader
            if getattr(dl, "_iterator", None) is ti:
                dl._iterator = None                     # persistent_workers：下次 iter() 重建 worker，不复用已关的迭代器
        if self._thread is not threading.current_thread():
            self._thread.join(_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logging.warning("TorchDataLoader 取批线程 %.0f s 内未退出", _JOIN_TIMEOUT_S)
        while True:                                     # 丢掉队列里残留的批
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._releaser.close()
        atexit.unregister(self._atexit)

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 —— 析构期不抛
            pass


class _Releaser:
    """释放线程：收到第 N 批时才丢掉第 N−1 批的引用。

    主线程交出第 N 批时，第 N−1 批在主线程（循环变量、取批生成器局部变量）里的引用都已解除，所以最后一个引用
    落在本线程，释放开销（munmap、归还共享内存页）不再占主线程。``close()`` 投递结束标记并 join。

    FIX-SHM：队列必须有界（容量 1）。原先用无界 ``SimpleQueue``，释放一批（6 GB：munmap + 归还 tmpfs 页，且要抢 GIL）
    一旦慢于取批节奏，待释放的批就在队列里无限堆积、全部仍占着 /dev/shm（CPU 复现：B32×16 worker 240 批后队列积压
    178 批、/dev/shm 涨到 325 GB；正式训练每步只慢一点点，约 7560 步撑满 561 G）。有界后释放跟不上时 ``hand_over``
    阻塞主线程等释放线程（反压），本线程持有 + 队列内最多 2 批。
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._t = threading.Thread(target=self._run, name="TorchDataLoader-release", daemon=True)
        self._t.start()

    def _run(self):
        held = None
        while True:
            nxt = self._q.get()
            held = nxt              # 旧 held 在本线程释放
            del nxt
            if held is _BG_END:
                return

    def hand_over(self, batch) -> None:
        self._q.put(batch)

    def close(self) -> None:
        self._q.put(_BG_END)
        if self._t is not threading.current_thread():
            self._t.join(_JOIN_TIMEOUT_S)


_MADV_POPULATE_READ = 22      # Linux ≥ 5.14
_MADV_POPULATE_WRITE = 23
_POPULATE_MIN_BYTES = 1 << 20
_LIBC = None


def _madvise(a: np.ndarray, advice: int) -> None:
    """对数组所在页整段 madvise（只建页表 / 分配页，不改字节）；失败显式报错。"""
    global _LIBC
    if a.nbytes == 0:
        return
    if _LIBC is None:
        _LIBC = ctypes.CDLL(None, use_errno=True)
    page = os.sysconf("SC_PAGE_SIZE")
    addr = a.__array_interface__["data"][0]
    start = addr - addr % page
    if _LIBC.madvise(ctypes.c_void_p(start), ctypes.c_size_t(addr + a.nbytes - start), advice) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"madvise({advice}) 失败: {os.strerror(err)}")


def _populate_tree(batch) -> None:
    """主进程对收到的批预建页表（MADV_POPULATE_READ）：worker 写好的共享内存页在本进程首次触碰时逐 4 KiB 缺页，
    6 GB 批实测约 0.5 s；在后台取批线程里一次做掉，H2D 时不再缺页。只读、不改字节。"""
    for x in jax.tree.leaves(batch):
        if isinstance(x, np.ndarray) and x.nbytes >= _POPULATE_MIN_BYTES:
            _madvise(x, _MADV_POPULATE_READ)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _to_shared_torch(batch):
    """worker 侧转为零拷贝 tensor 视图，让 DataLoader 队列通过共享内存传递。

    numpy 的 bf16 先按 uint16 查看，再还原 torch bf16；全程保持位模式。
    """
    def convert(x):
        if x is None:
            return None
        x = np.asarray(x)
        if x.dtype == ml_dtypes.bfloat16:
            return torch.from_numpy(x.view(np.uint16)).view(torch.bfloat16)
        return torch.from_numpy(x)

    return jax.tree.map(convert, batch, is_leaf=lambda x: x is None)


def _from_shared_torch(batch):
    """主进程还原共享内存上的 numpy 视图，bf16 反向查看且不改字节。"""
    def convert(t):
        if t is None:
            return None
        if t.dtype == torch.bfloat16:
            return t.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
        return t.numpy()

    return jax.tree.map(convert, batch, is_leaf=lambda x: x is None)


_NP_TO_TORCH = {np.dtype(k): v for k, v in {
    np.bool_: torch.bool, np.uint8: torch.uint8, np.int8: torch.int8, np.int16: torch.int16, np.uint16: torch.uint16,
    np.int32: torch.int32, np.int64: torch.int64, np.float16: torch.float16, np.float32: torch.float32,
    np.float64: torch.float64}.items()}


def _shared_empty(shape, dtype: np.dtype):
    """在共享内存里直接分配（torch 当前 sharing strategy 的新 storage），返回 (torch tensor, 同一内存的 numpy 视图)。

    送进 DataLoader 队列时 storage 已共享，不再整块拷进共享内存。bf16 用 uint16 存、torch 侧再 view 回 bf16。
    """
    store_dt = np.dtype(np.uint16) if dtype == ml_dtypes.bfloat16 else np.dtype(dtype)
    tdt = _NP_TO_TORCH[store_dt]
    n = int(np.prod(shape, dtype=np.int64))
    storage = torch.UntypedStorage._new_shared(max(n * store_dt.itemsize, 1))
    t = torch.empty((0,), dtype=tdt).set_(storage, 0, tuple(shape))
    view = t.numpy()
    if view.nbytes >= _POPULATE_MIN_BYTES:
        # 新 tmpfs 页逐 4 KiB 缺页分配很慢（6 GB 批实测写入 4.8 s）；先一次 MADV_POPULATE_WRITE 批量建页（2.7 s + 写 1.1 s）
        _madvise(view, _MADV_POPULATE_WRITE)
    if dtype == ml_dtypes.bfloat16:
        t = t.view(torch.bfloat16)
        view = view.view(ml_dtypes.bfloat16)
    return t, view


def _stack_shared(*xs):
    """与 ``np.stack([np.asarray(x) for x in xs])`` 逐字节相同，但直接写进共享内存 storage。

    只在各元素 dtype / shape 一致且 dtype 可由 torch 承载时走直写；否则回到原 np.stack + 共享化（与 S5-L1 之前同式）。
    """
    arrs = [np.asarray(x) for x in xs]
    a0 = arrs[0]
    dt = a0.dtype
    ok = (dt == ml_dtypes.bfloat16 or dt in _NP_TO_TORCH) and a0.size > 0 and all(
        a.dtype == dt and a.shape == a0.shape for a in arrs)
    if not ok:
        return _to_shared_torch(np.stack(arrs, axis=0))
    t, view = _shared_empty((len(arrs),) + a0.shape, dt)
    np.stack(arrs, axis=0, out=view)
    return t


def _collate_fn_shm(items):
    """worker 侧 collate：各键直接 stack 进共享内存（S5-L1），字节与 ``_to_shared_torch(_collate_fn(items))`` 相同。

    ``items`` 为 ``_PreCollated``（wan_raw9 的 ``TransformedDataset.__getitems__``）时，其 key 已是拼好的共享 tensor。
    """
    if isinstance(items, _PreCollated):
        batch = jax.tree.map(_stack_shared, *items.items)
        if batch.get(items.key, None) is not None:
            raise RuntimeError(f"_PreCollated 键 {items.key!r} 未置空")
        batch[items.key] = items.tensor
        return batch
    return jax.tree.map(_stack_shared, *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    # S5-L1：worker 正常收尾（_shutdown_workers 发结束信号、worker 循环 return）后，解释器退出的 C 层静态析构里有
    # 仍可 join 的 std::thread，触发 `terminate called without an active exception` → SIGABRT（BASE 代码同样复现，
    # 原先训练退出时 worker 被 SIGTERM 直接杀掉所以看不到）。worker 循环已把结果队列 cancel_join_thread + close，
    # 此时无待交付数据：在 worker 的 atexit 里刷完标准流后 os._exit(0)，跳过 C 层析构。
    atexit.register(_worker_exit_now)


def _worker_exit_now() -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(0)


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
