"""
We implemented our own data loader,
which can be 5-10x faster than LeRobot dataloader and can avoid memory explosion issue
"""

import jax
import logging
import os
import pathlib
import re
from omegaconf import DictConfig
from openpi.models import model as _model
from openpi.training.data_loader import DataLoader, TorchDataLoader,transform_dataset
import openpi.training.config as _config

from mme_vla_suite.datastore import StoreMeta, load_manifest, require_no_pack_lock, require_verified
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.config.utils import get_history_config


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _parse_source_run(spec: str) -> dict:
    """`<run_name>/<checkpoint_name>#<state_key>` → {run_name, checkpoint_name, epoch, state_key}（禁止只当注释串）。"""
    if "#" not in spec or "/" not in spec:
        raise ValueError(f"motion.source_run 格式须为 <run>/<ckpt>#<state_key>: {spec!r}")
    path, state_key = spec.rsplit("#", 1)
    run_name, ckpt = path.rsplit("/", 1)
    m = re.fullmatch(r"checkpoint_epoch_(\d+)\.pt", ckpt)
    if m is None:
        raise ValueError(f"motion.source_run 的 checkpoint 名解析不出 epoch: {ckpt!r}")
    return {"run_name": run_name, "checkpoint_name": ckpt, "epoch": int(m[1]), "state_key": state_key}


def _pool_motion_gate(mcfg, mmeta, kind: str) -> None:
    """池化分支（1108 计划 1.1 第 4 行）：不解析 source_run；YAML readout 与 store provenance 的 kind、readout、
    池化源码 sha256 逐项相等，encoder_ckpt 为 null（provenance 自身的形制已由 MotionMeta.load 核过）。"""
    from mme_vla_suite.shared import wan_pool768 as wp

    if kind != wp.POOL768_ENCODER_KIND:
        raise ValueError(f"motion.encoder_kind={kind!r} 未知（只支持 {wp.POOL768_ENCODER_KIND!r}）")
    if mcfg.get("source_run") is not None:
        raise ValueError("池化分支（motion.encoder_kind 已写）不得同时写 motion.source_run")
    prov = mmeta.provenance
    if prov.get("encoder_kind") != kind:
        raise ValueError(f"motion.encoder_kind={kind!r} != motion store provenance.encoder_kind {prov.get('encoder_kind')!r}")
    readout = mcfg.get("readout")
    if readout != wp.POOL768_READOUT or prov.get("readout") != readout:
        raise ValueError(f"motion.readout={readout!r} / store provenance.readout={prov.get('readout')!r} "
                         f"与 {wp.POOL768_READOUT!r} 不全相等")
    if prov.get("pool_source_sha256") != wp.pool_source_sha256():
        raise ValueError("motion store provenance.pool_source_sha256 与现行 wan_pool768.py 不符")
    if "encoder_ckpt" not in prov or prov["encoder_ckpt"] is not None:
        raise ValueError("池化表 provenance.encoder_ckpt 必须显式为 null")


def _raw9_motion_gate(mcfg, mmeta) -> None:
    """raw9 直读分支（1112 计划 1.1 第 8 行）：不解析 source_run；YAML 与 store provenance 的 kind / readout 相等、
    raw9 源码 sha256 同现行 wan_raw9.py、encoder_ckpt 为 null；YAML 几何（dim / tokens_per_window / stride）与表布局一致。"""
    from mme_vla_suite.shared import wan_raw9 as wr

    kind = mcfg.get("encoder_kind")
    if mcfg.get("source_run") is not None:
        raise ValueError("raw9 分支（motion.encoder_kind=wan_raw9）不得同时写 motion.source_run")
    prov = mmeta.provenance
    if prov.get("encoder_kind") != kind:
        raise ValueError(f"motion.encoder_kind={kind!r} != motion store provenance.encoder_kind {prov.get('encoder_kind')!r}")
    readout = mcfg.get("readout")
    if readout != wr.RAW9_READOUT or prov.get("readout") != readout:
        raise ValueError(f"motion.readout={readout!r} / store provenance.readout={prov.get('readout')!r} "
                         f"与 {wr.RAW9_READOUT!r} 不全相等")
    if prov.get("raw9_source_sha256") != wr.source_sha256():
        raise ValueError("motion store provenance.raw9_source_sha256 与现行 wan_raw9.py 不符")
    if "encoder_ckpt" not in prov or prov["encoder_ckpt"] is not None:
        raise ValueError("raw9 表 provenance.encoder_ckpt 必须显式为 null")
    geom = {"dim": (mcfg.get("dim"), wr.RAW9_TOKEN_DIM),
            "tokens_per_window": (mcfg.get("tokens_per_window"), wr.RAW9_TOKENS_PER_WINDOW),
            "stride": (mcfg.get("stride"), wr.RAW9_STRIDE)}
    for key, (got, want) in geom.items():
        if got is None or int(got) != want:
            raise ValueError(f"raw9 分支 motion.{key}={got!r} != {want}")
    if mmeta.raw.get("layout") != ms.LAYOUT_RAW9:
        raise ValueError(f"raw9 分支的 motion store layout={mmeta.raw.get('layout')!r} != {ms.LAYOUT_RAW9!r}")


def _motion_gates(history_config, frame_meta: StoreMeta) -> str | None:
    """motion.enabled=true 时的 fail-loud 闸：锁 / MotionMeta / verified / stride / 同源 / source_run 绑定；关闭态返回 None 且什么都不读。"""
    mcfg = getattr(history_config, "motion", None) if history_config is not None else None
    if not (mcfg is not None and mcfg.get("enabled", False)):
        return None
    root = os.environ.get("MMEVLA_MOTION_STORE") or str(mcfg.store_path)
    motion_root = pathlib.Path(root)
    if not motion_root.is_absolute():
        motion_root = _REPO_ROOT / motion_root
    ms.require_no_pack_lock(motion_root)
    mmeta = ms.MotionMeta.load(motion_root)                # 不得拿 StoreMeta.load 解析 motion layout
    ms.require_verified(mmeta)
    if int(mcfg.stride) != mmeta.spec.grid_stride:         # 步长由表布局给出（旧两布局恒为 16）
        raise ValueError(f"motion.stride={mcfg.stride} != motion store GRID_STRIDE {mmeta.spec.grid_stride}")
    manifest = load_manifest(os.environ.get("MMEVLA_FRAMESAMP_MANIFEST") or frame_meta.manifest_path)
    ms.check_same_source(frame_meta.manifest_sha256, mmeta, manifest)
    kind = mcfg.get("encoder_kind")
    if kind is not None:
        from mme_vla_suite.shared import wan_raw9 as _wr
        if kind == _wr.RAW9_ENCODER_KIND:
            _raw9_motion_gate(mcfg, mmeta)
        else:
            _pool_motion_gate(mcfg, mmeta, kind)
        logging.info("motion memory 开启（%s）：store=%s rows=%d manifest=%s…", kind, motion_root, mmeta.num_rows,
                     mmeta.manifest_sha256[:16])
        return str(motion_root)
    if "encoder_kind" in mmeta.provenance:
        raise ValueError(f"YAML 未写 motion.encoder_kind（encoder 路径），但 motion store 是 "
                         f"{mmeta.provenance['encoder_kind']!r} 池化表")
    want = _parse_source_run(str(mcfg.source_run))
    enc = mmeta.provenance.get("encoder", {})
    got = {k: enc.get(k) for k in ("run_name", "checkpoint_name", "epoch", "state_key")}
    if got != want:
        raise ValueError(f"motion.source_run {want} != motion store provenance.encoder {got}")
    m = re.fullmatch(r"checkpoint_epoch_(\d+)\.pt", str(enc.get("checkpoint_name", "")))
    if m is None or int(m[1]) != int(enc.get("epoch", -1)):
        raise ValueError("motion store provenance.encoder 的 checkpoint_name 解析出的 epoch 与显式 epoch 不符")
    logging.info("motion memory 开启：store=%s rows=%d manifest=%s…", motion_root, mmeta.num_rows, mmeta.manifest_sha256[:16])
    return str(motion_root)


def _create_framesamp_dataset(dataset_path, data_config, history_config, action_horizon):
    """packed 分派（B.4）：闸全部 fail-loud，任何不过直接 raise，绝不回退散 npy。"""
    require_no_pack_lock(dataset_path)                     # 打包/verify 进行中即拒
    meta = StoreMeta.load(dataset_path)                    # 缺失/损坏/契约不符即拒
    require_verified(meta)                                 # status != verified 即拒（G14）
    motion_root = _motion_gates(history_config, meta)      # 关闭态不执行 motion 闸
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
        motion_root=motion_root,
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

    # packed FrameSampDataset 是唯一训练数据路径（commitV4.1 数据链单一化，
    # legacy RoboMMEDataset 与 MMEVLA_DATA_BACKEND 三态分派已删除）
    dataset = _create_framesamp_dataset(
        dataset_path, data_config, history_config, action_horizon)

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