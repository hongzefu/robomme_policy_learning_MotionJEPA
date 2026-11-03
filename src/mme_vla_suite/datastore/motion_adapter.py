"""训练与在线策略的原始 motion 合同及无损配对／拆分边界。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import numpy as np

RAW_SHAPE = (8, 256)
PAIR_SHAPE = (4, 512)
PAIRING = "adjacent_pair_v1"
STAGE2_READOUT = "stage2_pool4_concat_v1"
STAGE2_SHAPE = (1, 1024)
STAGE2_SPLIT_READOUT = "stage2_pool4_split_v1"
STAGE2_SPLIT_SHAPE = (4, 256)
STAGE2_LAYOUT = "motion-stage2-1x1024-grid32-demopad17-v1"
RAW_LAYOUT = "motion-v10-8x256-grid16-demopad17-v1"
LEGACY_LAYOUTS = ("motion-768-grid16-v1", "motion-768-grid16-demopad17-v1")


def pair_adjacent_tokens(raw: np.ndarray) -> np.ndarray:
    """保持全部 fp32 位串，依次合并槽 (0,1)、(2,3)、(4,5)、(6,7)。"""
    if not isinstance(raw, np.ndarray):
        raise TypeError("原始 motion 必须是 numpy.ndarray")
    if raw.dtype != np.dtype(np.float32):
        raise ValueError(f"原始 motion 必须为 float32，得到 {raw.dtype}")
    if raw.ndim < 2 or raw.shape[-2:] != RAW_SHAPE:
        raise ValueError(f"原始 motion 末两维必须为 {RAW_SHAPE}，得到 {raw.shape}；禁止重复配对")
    return np.ascontiguousarray(raw).reshape(raw.shape[:-2] + PAIR_SHAPE)


def split_stage2_tokens(raw: np.ndarray) -> np.ndarray:
    """保持全部 fp32 位串，按原 pool 组顺序拆成四个 256 维槽。"""
    if not isinstance(raw, np.ndarray):
        raise TypeError("原始 stage2 motion 必须是 numpy.ndarray")
    if raw.dtype != np.dtype(np.float32):
        raise ValueError(f"原始 stage2 motion 必须为 float32，得到 {raw.dtype}")
    if raw.ndim < 2 or raw.shape[-2:] != STAGE2_SHAPE:
        raise ValueError(f"原始 stage2 motion 末两维必须为 {STAGE2_SHAPE}，得到 {raw.shape}；禁止重复拆分")
    return np.ascontiguousarray(raw).reshape(raw.shape[:-2] + STAGE2_SPLIT_SHAPE)


@dataclass(frozen=True)
class MotionPolicySpec:
    """分别描述落盘形状、策略形状与以完整窗口计的容量。"""

    source_shape: tuple[int, ...]
    readout: str
    num_tokens: int
    dim: int
    budget: int
    window_budget: int
    layouts: tuple[str, ...]


def _integer(cfg: Mapping, key: str, default: int) -> int:
    value = cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"motion.{key} 必须为整数，得到 {value!r}")
    return int(value)


def motion_policy_spec(cfg: Mapping) -> MotionPolicySpec:
    """解析 motion 节；新读法显式声明形状键，旧配置保持单个 768 维槽。"""
    if not isinstance(cfg, Mapping):
        raise TypeError("motion 配置必须为映射")
    if cfg.get("readout") == STAGE2_SPLIT_READOUT:
        required = {"source_shape", "num_tokens", "dim", "budget"}
        if required - cfg.keys():
            raise ValueError(f"stage2 split motion 缺字段 {sorted(required - cfg.keys())}")
        try:
            shape = tuple(cfg["source_shape"])
        except TypeError as exc:
            raise ValueError("stage2 split motion.source_shape 必须为 [1,1024]") from exc
        budget = _integer(cfg, "budget", 384)
        if (any(isinstance(v, bool) or not isinstance(v, Integral) for v in shape)
                or shape != STAGE2_SHAPE or _integer(cfg, "num_tokens", 4) != 4
                or _integer(cfg, "dim", 256) != 256):
            raise ValueError("stage2 split motion 必须为 source_shape=[1,1024]、num_tokens=4、dim=256")
        if budget <= 0 or budget % 4:
            raise ValueError("stage2 split motion.budget 必须为正数且能被 num_tokens=4 整除")
        return MotionPolicySpec(STAGE2_SHAPE, STAGE2_SPLIT_READOUT, 4, 256, budget,
                                budget // 4, (STAGE2_LAYOUT,))
    if cfg.get("readout") == STAGE2_READOUT:
        required = {"source_shape", "num_tokens", "dim", "budget"}
        if required - cfg.keys():
            raise ValueError(f"stage2 motion 缺字段 {sorted(required - cfg.keys())}")
        shape = tuple(cfg["source_shape"])
        budget = _integer(cfg, "budget", 96)
        if (any(isinstance(v, bool) or not isinstance(v, Integral) for v in shape)
                or shape != STAGE2_SHAPE or _integer(cfg, "num_tokens", 1) != 1
                or _integer(cfg, "dim", 1024) != 1024 or budget <= 0):
            raise ValueError("stage2 motion 必须为 source_shape=[1,1024]、num_tokens=1、dim=1024、budget>0")
        return MotionPolicySpec(STAGE2_SHAPE, STAGE2_READOUT, 1, 1024, budget, budget, (STAGE2_LAYOUT,))
    paired = "source_shape" in cfg or "readout" in cfg
    if paired:
        budget = _integer(cfg, "budget", 160)
        num_tokens = _integer(cfg, "num_tokens", 1)
        dim = _integer(cfg, "dim", 768)
        required = {"source_shape", "readout", "num_tokens", "dim"}
        missing = required - cfg.keys()
        if missing:
            raise ValueError(f"新 motion 合同缺字段 {sorted(missing)}")
        try:
            shape = tuple(cfg["source_shape"])
        except TypeError as exc:
            raise ValueError("motion.source_shape 必须为 [8,256]") from exc
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in shape) or shape != RAW_SHAPE:
            raise ValueError(f"motion.source_shape 必须为 {RAW_SHAPE}，得到 {shape}")
        if cfg["readout"] != PAIRING or (num_tokens, dim) != PAIR_SHAPE:
            raise ValueError("新 motion 必须使用 adjacent_pair_v1、num_tokens=4、dim=512")
        if budget <= 0 or budget % num_tokens:
            raise ValueError("新 motion.budget 必须为正数且能被 num_tokens=4 整除")
        return MotionPolicySpec(RAW_SHAPE, PAIRING, num_tokens, dim, budget,
                                budget // num_tokens, (RAW_LAYOUT,))
    # 旧 loader 对 dim/budget 使用 int 转换，保留其既有配置接受范围。
    budget = int(cfg.get("budget", 160))
    dim = int(cfg.get("dim", 768))
    num_tokens = _integer(cfg, "num_tokens", 1)
    if (num_tokens, dim) != (1, 768):
        raise ValueError("旧 motion 配置仅支持 num_tokens=1、dim=768；新配置须显式声明 source_shape/readout")
    if budget < 16 or budget % 16:
        raise ValueError("旧 motion.budget 必须至少为 16 且能被 16 整除")
    return MotionPolicySpec((768,), "identity", 1, 768, budget, budget, LEGACY_LAYOUTS)
