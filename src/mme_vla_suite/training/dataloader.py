"""
We implemented our own data loader,
which can be 5-10x faster than LeRobot dataloader and can avoid memory explosion issue
"""

import jax
import logging
import os
from omegaconf import DictConfig
from openpi.models import model as _model
from openpi.training.data_loader import DataLoader, TorchDataLoader,transform_dataset
import openpi.training.config as _config

from mme_vla_suite.datastore import StoreMeta, require_no_pack_lock, require_verified
from mme_vla_suite.training.dataset import RoboMMEDataset
from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.config.utils import get_history_config


def _resolve_backend(dataset_path: str) -> str:
    """MMEVLA_DATA_BACKEND ∈ {packed, legacy, auto}（v2 计划 B.4）。

    未设置默认 legacy——与现状行为逐字节相同，零静默切换；auto 必须显式设置才生效
    （仅限本机探索）且必打 WARNING；正式 launcher 一律显式 packed|legacy（R16）。
    """
    backend = os.environ.get("MMEVLA_DATA_BACKEND", "") or "legacy"
    if backend not in ("packed", "legacy", "auto"):
        raise ValueError(f"MMEVLA_DATA_BACKEND 非法: {backend!r}（∈ packed|legacy|auto）")
    if backend == "auto":
        is_packed = os.path.isfile(os.path.join(dataset_path, "meta", "store_meta.json"))
        backend = "packed" if is_packed else "legacy"
        logging.warning(
            "MMEVLA_DATA_BACKEND=auto 按 store_meta.json 存在性分派为 %s"
            "（仅限本机探索；正式 launcher 必须显式 packed|legacy，R16）", backend)
    return backend


def _create_framesamp_dataset(dataset_path, data_config, history_config, action_horizon):
    """packed 分派（B.4）：闸全部 fail-loud，任何不过直接 raise，绝不回退散 npy。"""
    require_no_pack_lock(dataset_path)                     # 打包/verify 进行中即拒
    meta = StoreMeta.load(dataset_path)                    # 缺失/损坏/契约不符即拒
    require_verified(meta)                                 # status != verified 即拒（G14）
    if meta.manifest_scope == "subset":
        # subset 迷你库禁止用于 S5 及以上任何判据（A.1）；S3 开发期矩阵须显式放行
        if os.environ.get("MMEVLA_FRAMESAMP_ALLOW_SUBSET") != "1":
            raise RuntimeError(
                f"manifest_scope=subset 的迷你库禁止用于 S5 及以上判据: {dataset_path}；"
                f"仅开发期可显式设 MMEVLA_FRAMESAMP_ALLOW_SUBSET=1 放行")
        logging.warning("packed 库为 subset 迷你库，凭 MMEVLA_FRAMESAMP_ALLOW_SUBSET=1 "
                        "放行——仅限开发期，一切判据 run 无效: %s", dataset_path)
    return FrameSampDataset(
        dataset_path,
        data_config=data_config,
        history_config=history_config,
        action_horizon=action_horizon,
        # 双根契约（B.4）：env 覆盖优先，未设取 store_meta 记录的绝对路径
        source_root=os.environ.get("MMEVLA_FRAMESAMP_SOURCE") or None,
        manifest_path=os.environ.get("MMEVLA_FRAMESAMP_MANIFEST") or None,
        verify_level=os.environ.get("MMEVLA_FRAMESAMP_VERIFY", "") or "fast",
    )



class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader
        self._total_samples = len(data_loader._data_loader.dataset)

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield HistAugObservation.from_dict(batch), batch["actions"]
            

def create_data_loader(
    dataset_path: str,
    data_config: _config.DataConfig,
    history_config: str | DictConfig | None,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[HistAugObservation, _model.Actions]]:
    
    history_config = get_history_config(history_config)

    # backend 三态分派（v2 计划 B.4）：packed=新链路 fail-loud；legacy=旧链路逐字不动
    if _resolve_backend(dataset_path) == "packed":
        dataset = _create_framesamp_dataset(
            dataset_path, data_config, history_config, action_horizon)
    else:
        dataset = RoboMMEDataset(
            dataset_path=dataset_path,
            data_config=data_config,
            history_config=history_config,
            action_horizon=action_horizon
        )
    
    dataset = transform_dataset(
        dataset, data_config, skip_norm_stats=skip_norm_stats)

    local_batch_size = batch_size // jax.process_count()
    logging.info(f"local_batch_size: {local_batch_size}")
    
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework="jax",
    )

    return DataLoaderImpl(data_config, data_loader)