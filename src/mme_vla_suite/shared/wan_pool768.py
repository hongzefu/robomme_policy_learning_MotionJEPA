"""Wan raw latent 零参数池化（768 维）：离线建表与在线 sidecar 共用的唯一实现。

计划 1108-4task-modul-wantp768-50k-plan.html §3.1。输入是 ``wan_motion_infer.py::encode_chunk`` 的 raw 输出，
float32 (9,16,32,32)，轴 (T,C,H,W)：

    A[c,h,w]        = (1/8)  × Σ_{t=1..8} Z[t,c,h,w]                   丢第 0 个 latent，后 8 个取时间均值
    H 箱 i=0..5     : [s_i, e_i) = [⌊32i/6⌋, ⌈32(i+1)/6⌉) = [0,6) [5,11) [10,16) [16,22) [21,27) [26,32)
    W 块 j=0..7     : [4j, 4j+4)
    P[c,i,j]        = (1/24) × Σ_{h∈[s_i,e_i)} Σ_{w∈[4j,4j+4)} A[c,h,w]     每格 6×4 = 24 个值
    F[0, 48c+8i+j]  = P[c,i,j]                                          C→H→W 展平，float32 (1,768)

分箱语义与 ``torch.nn.functional.adaptive_avg_pool2d(A, (6,8))`` 相同（起点向下取整、终点向上取整）。
H 方向相邻箱共用第 5、10、21、26 行（各被计入两格），这是 32 行切 6 箱的必然结果，不是缺陷；W 方向不重叠。

数值合同：全程 float32，归约次序固定为先 T（t=1→8 逐个累加后 ÷8）、再块内 W（b=0→3）、再箱内 H
（h=s_i→e_i−1），每一级都是逐元素的 IEEE 顺序加法，最后 ÷24；不走 ``ndarray.sum`` / ``mean``，
结果不随 NumPy 的成对求和策略变化。实数域均值可交换，浮点不可交换，禁止调换上述次序。

本模块只依赖标准库与 NumPy，可在没有安装 ``mme_vla_suite`` 的环境里按文件路径加载（sidecar 即如此），
不得引入 torch / jax，也不得引入任何可训练参数。
"""

from __future__ import annotations

import hashlib
import math
import pathlib

import numpy as np

POOL768_READOUT = "wan_raw_timepool8_adaptive6x8_v1"
POOL768_ENCODER_KIND = "wan_fixedpool768"
POOL768_INPUT_SHAPE = (9, 16, 32, 32)
POOL768_SHAPE = (1, 768)
POOL768_OUTPUT_BYTES = 768 * 4
POOL768_PARAMETER_COUNT = 0

_C, _H, _W = POOL768_INPUT_SHAPE[1:]
_OUT_H, _OUT_W = 6, 8
POOL768_W_BLOCK = _W // _OUT_W                                     # 4
POOL768_H_BINS = tuple((math.floor(_H * i / _OUT_H), math.ceil(_H * (i + 1) / _OUT_H)) for i in range(_OUT_H))
if POOL768_H_BINS != ((0, 6), (5, 11), (10, 16), (16, 22), (21, 27), (26, 32)):
    raise RuntimeError(f"H 分箱与计划 §3.1 不符: {POOL768_H_BINS}")
if any(e - s != 6 for s, e in POOL768_H_BINS) or POOL768_W_BLOCK * _OUT_W != _W:
    raise RuntimeError("每格须恰为 6 行 × 4 列 = 24 个值")
_CELL = np.float32(24)


def pool_raw_wan_latent_768(z: np.ndarray) -> np.ndarray:
    """把单窗 raw latent 池化成 C 连续 float32 (1,768)；不改写输入，不做隐式类型转换。"""
    if not isinstance(z, np.ndarray):
        raise TypeError(f"raw Wan latent 必须是 numpy.ndarray，得到 {type(z).__name__}")
    if z.dtype != np.dtype(np.float32):
        raise TypeError(f"raw Wan latent 必须为原生字节序 float32，得到 {z.dtype}")
    if z.shape != POOL768_INPUT_SHAPE:
        raise ValueError(f"raw Wan latent 形状必须为 {POOL768_INPUT_SHAPE}，得到 {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("raw Wan latent 含非有限值")
    # 溢出统一由末尾的有限性检查报错，这里只屏蔽 NumPy 的告警。
    with np.errstate(over="ignore", invalid="ignore"):
        temporal = np.array(z[1], dtype=np.float32, copy=True, order="C")
        for t in range(2, 9):
            temporal += z[t]
        temporal = temporal / np.float32(8)                         # (C, H, W)
        blocks = temporal.reshape(_C, _H, _OUT_W, POOL768_W_BLOCK)  # (C, h, j, b)
        block_w = blocks[..., 0].copy()                             # (C, H, 8)
        for b in range(1, POOL768_W_BLOCK):
            block_w += blocks[..., b]
        pooled = np.empty((_C, _OUT_H, _OUT_W), dtype=np.float32)
        for i, (s, e) in enumerate(POOL768_H_BINS):
            acc = block_w[:, s].copy()                              # (C, 8)
            for h in range(s + 1, e):
                acc += block_w[:, h]
            pooled[:, i] = acc / _CELL
    if not np.isfinite(pooled).all():
        raise ValueError("池化累加溢出为非有限值")
    return pooled.reshape(POOL768_SHAPE)


def pool_source_sha256() -> str:
    """返回本文件字节的 SHA256，供来源记录绑定池化源码身份。"""
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
