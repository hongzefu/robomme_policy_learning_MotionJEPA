#!/usr/bin/env python
"""`wan_motion_infer.py` 的快速数值模式（steer-speedup-1116 计划 F3，`1116-steer-speedup-fix-plan.html` 第二部分 1 节 S-A）。

为什么单独成文件而不改 `wan_motion_infer.py`：后者是上游逐字节复制件，其 sha256 被 `SOURCE_PIN.json::source_sha256`、
sidecar 启动校验、离线库 provenance 的 `module_sha256` 与阶段二 / 固定池化 `vae_adapter_sha256` 多处钉死；改一个字节，
缺省模式的握手就会失败。本文件只在 `MMEVLA_MOTION_FAST=1` 时被调用，缺省路径一行不经过这里，缺省行为逐字节不变。

快速模式（用户 2026-11-16 放宽精度口径，误差只由 ENC_TOL 约束：每窗 cos ≥ 0.999、maxabs ≤ 0.1）：
  * VAE 前向 bf16 autocast（权重仍 fp32），matmul / cuDNN TF32 开，`cudnn.benchmark=True`；
  * encoder 保持 bf16 autocast（与训练、缺省模式相同，直接复用 `wan_motion_infer.motion_token`）；
  * 允许 B>1 批量（离线批量编码用；在线 sidecar 仍一次一窗）。
  * 回退：`MMEVLA_MOTION_FAST_VAE_FP32=1` 时 VAE 回 fp32、TF32 关（benchmark 仍开），供 ENC_TOL 失败时使用。

环境变量检查（`check_env`）与版本检查（`check_versions`）在快速模式下的期望值与缺省相同：
`NVIDIA_TF32_OVERRIDE` 等会绕过代码开关（含回退档的 TF32 关），快速模式同样禁止；软件版本同样钉 PINNED。
开关读回值的期望见 `expected_flags`，`pin_numerics_fast` 末尾按它断言。

本模块 import 零副作用（不 import torch，直到调用函数）；按文件路径加载 `wan_motion_infer.py`，不改它。
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import warnings

import numpy as np

FAST_ENV = "MMEVLA_MOTION_FAST"
FAST_VAE_FP32_ENV = "MMEVLA_MOTION_FAST_VAE_FP32"
FAST_SCHEMA = "wan-motion-fast-v1"

_HERE = pathlib.Path(__file__).resolve().parent
_W = None


def infer_module():
    """按文件路径加载同目录的 `wan_motion_infer.py`（缓存，同一进程只加载一次）。"""
    global _W
    if _W is None:
        spec = importlib.util.spec_from_file_location("wan_motion_infer_for_fast", str(_HERE / "wan_motion_infer.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _W = mod
    return _W


def fast_enabled(env=None) -> bool:
    """`MMEVLA_MOTION_FAST=1` 才算开启；其它值（含空串、"0"、"true"）一律视为关闭。"""
    return (os.environ if env is None else env).get(FAST_ENV) == "1"


def vae_fp32(env=None) -> bool:
    """快速模式下的 VAE 回退档：`MMEVLA_MOTION_FAST_VAE_FP32=1` 时 VAE fp32、TF32 关。"""
    return (os.environ if env is None else env).get(FAST_VAE_FP32_ENV) == "1"


def expected_flags(fp32_vae: bool) -> dict:
    """快速模式 pin 之后各开关的读回期望；在缺省 `EXPECTED_FLAGS` 上只改 TF32、精度档与 benchmark。"""
    flags = dict(infer_module().EXPECTED_FLAGS)
    flags["cudnn_benchmark"] = True
    if not fp32_vae:
        flags.update(matmul_allow_tf32=True, cudnn_allow_tf32=True, float32_matmul_precision="high")
    return flags


def pin_numerics_fast(fp32_vae: bool | None = None) -> dict:
    """快速模式的全局数值开关；末尾断言 SDPA 四后端全开且读回值等于 `expected_flags(fp32_vae)`。"""
    import torch
    W = infer_module()
    if fp32_vae is None:
        fp32_vae = vae_fp32()
    tf32 = not fp32_vae
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    snap = W.numerics_snapshot()
    sdp = {k: v for k, v in snap.items() if k.startswith("sdp_")}
    assert all(sdp.values()), f"SDPA 后端被关闭 {sdp}"
    want = expected_flags(fp32_vae)
    assert snap == want, f"快速模式数值开关读回值与预期不符：{snap} ≠ {want}"
    return snap


def check_env(strict=True):
    """快速模式的环境变量期望与缺省相同（禁止会绕过代码开关的变量）。"""
    return infer_module().check_env(strict=strict)


def check_versions(strict=True, with_diffusers=False):
    """快速模式的版本期望与缺省相同（PINNED）。"""
    return infer_module().check_versions(strict=strict, with_diffusers=with_diffusers)


def module_sha256() -> str:
    return hashlib.sha256((_HERE / "wan_motion_fast.py").read_bytes()).hexdigest()


def fast_info(fp32_vae: bool | None = None, flags: dict | None = None, batch: int = 1) -> dict:
    """握手 / 离线清单里的快速标记；客户端凭 `enabled is True` 判定 sidecar 处于快速模式。"""
    if fp32_vae is None:
        fp32_vae = vae_fp32()
    return {"schema": FAST_SCHEMA, "enabled": True, "vae_amp": "off" if fp32_vae else "bf16",
            "vae_fp32_fallback": bool(fp32_vae), "tf32": "off" if fp32_vae else "on", "cudnn_benchmark": True,
            "encoder_amp": "bf16", "batch": int(batch), "flags": flags,
            "fast_module_sha256": module_sha256()}


def fast_vae_info(vinfo: dict, flags: dict, fp32_vae: bool) -> dict:
    """把 `load_vae` 的 info 改成快速模式的如实数值字段（权重 dtype 仍 float32）；身份键不动。"""
    out = dict(vinfo)
    out.update(amp="off" if fp32_vae else "bf16", tf32="off" if fp32_vae else "on", batch="any", flags=dict(flags))
    return out


def fast_encoder_info(einfo: dict, flags: dict, fp32_vae: bool) -> dict:
    """encoder 段 amp 不变（bf16），只改 TF32 / batch / flags 为如实值；身份键不动。"""
    out = dict(einfo)
    out.update(tf32="off" if fp32_vae else "on", batch="any", flags=dict(flags))
    return out


def encode_chunks(vae, windows, device, fp32_vae: bool | None = None):
    """(B,33,256,256,3) uint8 → 组优先 latent (B,9,16,32,32) fp32 torch（留在 device）。

    预处理与 `wan_motion_infer.encode_chunk` 逐句相同（/127.5−1、通道轴调整、`.mode()`），只把 VAE 前向
    放进 bf16 autocast（回退档不开），并允许 B>1。调用前须已 `pin_numerics_fast()`。
    """
    import torch
    W = infer_module()
    if fp32_vae is None:
        fp32_vae = vae_fp32()
    windows = np.ascontiguousarray(windows)
    if windows.ndim == 4:
        windows = windows[None]
    assert windows.ndim == 5 and windows.shape[1:] == (W.WINDOW, W.FRAME_SIZE, W.FRAME_SIZE, 3), \
        f"期望 (B,33,256,256,3)，得到 {windows.shape}"
    assert windows.dtype == np.uint8, f"期望 uint8，得到 {windows.dtype}"
    device = torch.device(device)
    with torch.no_grad():
        f = torch.from_numpy(windows).permute(0, 1, 4, 2, 3).float().to(device)         # (B,33,3,256,256)
        x = (f / 127.5 - 1.0).permute(0, 2, 1, 3, 4).contiguous()                       # (B,3,33,256,256)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=not fp32_vae):
            z = vae.encode(x).latent_dist.mode()
        B = windows.shape[0]
        assert z.shape == (B, W.LAT_C, W.LAT_G, W.LAT_H, W.LAT_W), f"VAE 返回形状异常 {tuple(z.shape)}"
        lat = z.float().permute(0, 2, 1, 3, 4).contiguous()
    assert torch.isfinite(lat).all()
    return lat


def motion_tokens(encoder, latents, use_amp, device):
    """(B,9,16,32,32) fp32 latent → (B,M*D) fp32 numpy；bf16 autocast 与 `wan_motion_infer.motion_token` 相同。"""
    import torch
    W = infer_module()
    device = torch.device(device)
    assert latents.ndim == 5 and tuple(latents.shape[1:]) == (W.LAT_G, W.LAT_C, W.LAT_H, W.LAT_W), \
        f"期望 (B,9,16,32,32)，得到 {tuple(latents.shape)}"
    assert latents.dtype == torch.float32
    with torch.no_grad():
        x = latents.contiguous().to(device)
        torch.clear_autocast_cache()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            m = encoder(x)
        out = m.reshape(m.shape[0], -1).float().cpu().numpy()
    assert np.isfinite(out).all()
    return out
