"""Wan raw latent 零参数直读（每窗 9 个 16384 维 token，窗步长 32）：离线建表、训练取数与在线 sidecar 共用的唯一实现。

计划 1112-4task-modul-wanraw9-s32-50k-plan.html 第二部分 1.0（契约 C-B / C-τ）与 1.1 第 1 行。输入是
``wan_motion_infer.py::encode_chunk`` 的 raw 输出，float32 (9,16,32,32)，轴 (T,C,H,W)：

    token[j, 1024c + 32h + w] = Z[j, c, h, w]        j = 0..8，C→H→W 展平，不做任何数值处理

- 零额外处理：不做 z-score、LayerNorm、裁剪或 dtype 转换，展平只是 reshape 视图（零拷贝，不改字节）。
- 窗步长 32：stride-16 源网格里取偶数 m（新行号 m' = m/2），demo / exec 两段各自从段起点起算。
- token 时刻（C-τ）：起点全域帧号 f 的窗，第 j 个 latent 的时刻 ``τ_j = min(f + 4j, last_real)``；
  ``last_real`` 对 demo 窗取 ``es − 1``（尾补窗的重复帧不算新时刻），对 exec 窗取 ``f + 32``（截断恒不生效）。
- budget 语义（C-B）：YAML ``motion.budget`` 是**窗数**；token 数 = ``budget × tokens_per_window``，
  全链路只经 ``motion_token_budget`` 换算；旧 YAML 缺 ``tokens_per_window`` 时按 1，返回值即 ``budget``。

本模块只依赖标准库与 NumPy，可在没有安装 ``mme_vla_suite`` 的环境里按文件路径加载（sidecar 即如此），
不得引入 torch / jax，也不得引入任何可训练参数。
"""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Mapping

import numpy as np

RAW9_ENCODER_KIND = "wan_raw9"
RAW9_READOUT = "wan_raw9_flat16384_s32_v1"
RAW9_INPUT_SHAPE = (9, 16, 32, 32)
RAW9_TOKEN_DIM = 16 * 32 * 32                       # 16384
RAW9_TOKENS_PER_WINDOW = 9
RAW9_STRIDE = 32
RAW9_ROW_BYTES = 9 * 16 * 32 * 32 * 4               # 589,824
RAW9_PARAMETER_COUNT = 0
RAW9_LATENT_FRAME_STEP = 4                          # 相邻 latent 的帧距：τ_j = f + 4j
RAW9_WINDOW_LAST_OFFSET = 32                        # 33 帧前视窗的末帧偏移（exec 窗 last_real = f + 32）

if RAW9_TOKEN_DIM != 16384 or RAW9_ROW_BYTES != 589824 or RAW9_TOKENS_PER_WINDOW != RAW9_INPUT_SHAPE[0]:
    raise RuntimeError("raw9 常量与计划 1.1 第 1 行不符")


def flatten_raw9(z: np.ndarray) -> np.ndarray:
    """单窗 raw latent (9,16,32,32) → (9,16384) 的 reshape 视图：C→H→W 展平，零拷贝、不改字节。

    输入须为原生字节序 float32、C 连续、全有限，否则 raise；不做隐式类型转换。
    """
    if not isinstance(z, np.ndarray):
        raise TypeError(f"raw Wan latent 必须是 numpy.ndarray，得到 {type(z).__name__}")
    if z.dtype != np.dtype(np.float32):
        raise TypeError(f"raw Wan latent 必须为原生字节序 float32，得到 {z.dtype}")
    if z.shape != RAW9_INPUT_SHAPE:
        raise ValueError(f"raw Wan latent 形状必须为 {RAW9_INPUT_SHAPE}，得到 {z.shape}")
    if not z.flags["C_CONTIGUOUS"]:
        raise ValueError("raw Wan latent 必须 C 连续（展平须是零拷贝视图）")
    if not np.isfinite(z).all():
        raise ValueError("raw Wan latent 含非有限值")
    out = z.reshape(RAW9_TOKENS_PER_WINDOW, RAW9_TOKEN_DIM)
    if not np.shares_memory(out, z):                 # C 连续输入下 reshape 恒为视图；不是即说明实现被改坏
        raise RuntimeError("flatten_raw9 未得到零拷贝视图")
    return out


def token_times(f: int, seg_last_real: int) -> np.ndarray:
    """起点全域帧号 f 的窗的 9 个 token 时刻 ``τ_j = min(f + 4j, seg_last_real)``，(9,) int64。

    demo 窗传 ``es − 1``，exec 窗传 ``f + 32``；``seg_last_real < f`` 说明调用方传错，raise。
    """
    f = int(f)
    last = int(seg_last_real)
    if last < f:
        raise ValueError(f"seg_last_real={last} < 窗起点 f={f}")
    tau = f + RAW9_LATENT_FRAME_STEP * np.arange(RAW9_TOKENS_PER_WINDOW, dtype=np.int64)
    return np.minimum(tau, np.int64(last))


def motion_token_budget(mcfg) -> int:
    """token 数 = ``budget × tokens_per_window``（C-B 唯一换算处）；缺 ``tokens_per_window`` 按 1。

    ``mcfg`` 可以是 OmegaConf DictConfig、dict 或带属性的对象。
    """
    if isinstance(mcfg, Mapping):
        budget = mcfg["budget"]
        tpw = mcfg.get("tokens_per_window", 1)
    else:
        budget = getattr(mcfg, "budget")
        tpw = getattr(mcfg, "tokens_per_window", 1)
    if tpw is None:
        tpw = 1
    return int(budget) * int(tpw)


def source_sha256() -> str:
    """返回本文件字节的 SHA256，供 store provenance（``raw9_source_sha256``）与在线握手绑定实现身份。"""
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
