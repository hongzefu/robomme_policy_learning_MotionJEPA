"""fp32 → bf16 的唯一转换函数（1117-e2e-enc-train-plan.html 〇节 R9、D5）。

全链路只有两处做 fp32→bf16，且都调本文件的 ``raw9_f32_to_bf16``：离线导出步长 16 新表
（``scripts/dataset/export_wan_raw9_s32.py --out-dtype bfloat16``），以及在线 ``motion_client`` 收到 sidecar 返回的
fp32 latent 之后。数据集、collate、预分配与模型只搬运 bf16 字节，不再转换；这样训练与评估喂给 encoder 的是同一套舍入。

舍入：``np.asarray(x, np.float32).astype(ml_dtypes.bfloat16)``，round-to-nearest-even（平局取尾数为偶者），
次正规数照常舍入（bf16 与 fp32 指数域相同）。输入须全有限，形状不变。

不放进 ``shared/wan_raw9.py``：那个文件的 ``source_sha256()`` 进 s32 表与 Wan-9 run 的 provenance 握手，改它会让既有校验失配。
本模块只依赖 NumPy 与 ml_dtypes（随 jax 安装），可按文件路径加载，不引入 torch / jax，不含任何可训练参数。
"""

from __future__ import annotations

import hashlib
import pathlib

import ml_dtypes
import numpy as np

BF16 = np.dtype(ml_dtypes.bfloat16)


def raw9_f32_to_bf16(x) -> np.ndarray:
    """fp32 数组 → 同形状 bf16 数组（RNE）。输入含 NaN / ±Inf 即 raise ValueError。"""
    a = np.asarray(x, np.float32)
    if not np.isfinite(a).all():
        raise ValueError("raw9_f32_to_bf16: 输入含非有限值（NaN / Inf）")
    out = a.astype(BF16)
    if out.shape != a.shape:                        # astype 恒保形；不是即说明实现被改坏
        raise RuntimeError("raw9_f32_to_bf16: 输出形状与输入不同")
    return out


def source_sha256() -> str:
    """本文件字节的 SHA256（导出 provenance 里只作记录，不参与校验）。"""
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
