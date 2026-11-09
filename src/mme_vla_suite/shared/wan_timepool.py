"""Wan raw latent 零参数时间优先池化：离线建库与在线 sidecar 共用的唯一实现。

输入是 ``wan_motion_infer.py::encode_chunk`` 的 raw 输出，float32 (9,16,32,32)，轴 (T,C,H,W)：

    A[c,h,w]       = (1/8)  × Σ_{t=1..8} Z[t,c,h,w]            丢第 0 个 latent，后 8 个取时间均值
    P[c,i,j]       = (1/16) × Σ_{a=0..3} Σ_{b=0..3} A[c,4i+a,4j+b]  不重叠 4×4 块
    F[0,64c+8i+j]  = P[c,i,j]                                  C/H/W 展平，输出 float32 (1,1024)

数值合同：全程 float32，归约次序固定为先 T（t=1→8）、再块内 W（b=0→3）、再块内 H（a=0→3），
每一级都是逐元素的 IEEE 顺序加法，不走 ``ndarray.sum`` / ``mean``，因此结果不随 NumPy 的成对
求和策略或轴迭代顺序变化。实数域均值可交换，浮点不可交换，禁止调换上述次序。

本模块只依赖标准库与 NumPy，可在没有安装 ``mme_vla_suite`` 的环境里按文件路径加载（sidecar
即如此），不得引入 torch / jax，也不得引入 affine、LayerNorm、L2、MLP 或任何可训练参数。
"""

from __future__ import annotations

import hashlib
import pathlib
import types

import numpy as np

TIMEPOOL_READOUT = "wan_raw_timepool8_spatial8x8_v1"
TIMEPOOL_LAYOUT = "motion-wan-timepool1024-grid32-demopad17-v1"
TIMEPOOL_ENCODER_KIND = "wan_fixedpool1024"
TIMEPOOL_INPUT_SHAPE = (9, 16, 32, 32)
TIMEPOOL_SHAPE = (1, 1024)
TIMEPOOL_OUTPUT_BYTES = 4096
TIMEPOOL_PARAMETER_COUNT = 0
POOL_VERSION = "wan-timepool-pool-v1"

# 来源记录的 pool.contract 必须与 dict(POOL_CONTRACT) 逐键相等；值全部可直接 JSON 序列化。
POOL_CONTRACT = types.MappingProxyType({
    "version": POOL_VERSION,
    "raw": True,
    "drop_first": True,
    "temporal_mean8": True,
    "spatial_block4": True,
    "flatten_CHW": True,
    "float32": True,
    "reduction_order": "T>block_W>block_H",
    "backend": "numpy-cpu",
})


def pool_raw_wan_latent(z: np.ndarray) -> np.ndarray:
    """把单窗 raw latent 池化成 C 连续 float32 (1,1024)；不改写输入，不做隐式类型转换。"""
    if not isinstance(z, np.ndarray):
        raise TypeError(f"raw Wan latent 必须是 numpy.ndarray，得到 {type(z).__name__}")
    if z.dtype != np.dtype(np.float32):
        raise TypeError(f"raw Wan latent 必须为原生字节序 float32，得到 {z.dtype}")
    if z.shape != TIMEPOOL_INPUT_SHAPE:
        raise ValueError(f"raw Wan latent 形状必须为 {TIMEPOOL_INPUT_SHAPE}，得到 {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("raw Wan latent 含非有限值")
    # 溢出统一由末尾的有限性检查报错，这里只屏蔽 NumPy 的告警。
    with np.errstate(over="ignore", invalid="ignore"):
        temporal = np.array(z[1], dtype=np.float32, copy=True, order="C")
        for t in range(2, 9):
            temporal += z[t]
        temporal = temporal / np.float32(8)
        blocks = temporal.reshape(16, 8, 4, 8, 4)  # (C, i, a, j, b)
        block_w = blocks[..., 0].copy()
        for b in range(1, 4):
            block_w += blocks[..., b]
        block_h = block_w[:, :, 0].copy()
        for a in range(1, 4):
            block_h += block_w[:, :, a]
        pooled = block_h / np.float32(16)
    if not np.isfinite(pooled).all():
        raise ValueError("池化累加溢出为非有限值")
    return pooled.reshape(TIMEPOOL_SHAPE)


def pool_source_sha256() -> str:
    """返回本文件字节的 SHA256，供来源记录绑定池化源码身份。"""
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
