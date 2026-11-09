#!/usr/bin/env python3
"""Wan 零参数时间优先池化的在线进程：33 帧经旧 VAE 适配件和共享池化返回 4096 字节。

运行环境沿用 ``v1-store/venvs/wan-stage2``（由 ``scripts/dataset/wan_stage2`` 项目锁定），本目录不新建
pyproject。协议与 ``stage2_sidecar.py`` 相同；本进程不加载任何 encoder 权重，握手也不带 encoder 字段。

模块顶层只依赖标准库与 NumPy：torch 只在真实模式里由适配件导入，所以 ``token_from_latent`` 可以在任何
环境按文件路径加载后直接调用，用于离线与在线的同 latent 对拍（G6）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ADAPTER_PATH = ROOT / "scripts/dataset/wan/wan_motion_infer.py"
LOCK_PATH = ROOT / "scripts/dataset/wan_stage2/uv.lock"
PROTOCOL_PATH = ROOT / "src/mme_vla_suite/policies/motion_protocol.py"
POOL_PATH = ROOT / "src/mme_vla_suite/shared/wan_timepool.py"
VAE_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
TOKEN_DIM = 1024
ENV_KEYS = ("python", "torch", "cuda", "cudnn", "cublas_pkg", "cudnn_pkg", "diffusers", "hostname",
            "gpu_name", "compute_cap", "sm_count", "driver", "env")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pool = _load(POOL_PATH, "wan_timepool_shared")
RESPONSE_BYTES = _pool.TIMEPOOL_OUTPUT_BYTES


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_from_latent(latent: np.ndarray) -> np.ndarray:
    """raw latent 经共享池化得到 (1024,) float32；离线与在线逐位相等靠的就是这唯一入口。"""
    token = _pool.pool_raw_wan_latent(latent).reshape(TOKEN_DIM)
    if token.dtype != np.float32 or not np.isfinite(token).all():
        raise ValueError("池化响应必须为有限 (1024,) float32")
    return token


def encode_raw_latent(adapter, vae, window: np.ndarray, device) -> np.ndarray:
    """单窗独立 encode 成 numpy raw latent；调用前后都清 VAE 内部缓存，不让状态跨窗续接。"""
    adapter.pin_numerics()
    vae.clear_cache()
    try:
        return adapter.encode_chunk(vae, window, device).detach().cpu().numpy()
    finally:
        vae.clear_cache()


def encode_window(adapter, vae, window: np.ndarray, device) -> np.ndarray:
    return token_from_latent(encode_raw_latent(adapter, vae, window, device))


def load_online_stack(expected_vae_sha256: str, device_name: str = "cuda:0"):
    """按生产顺序钉数值、核版本并加载 VAE，返回 (adapter, vae, vinfo, device)；真实在线检查复用同一入口。"""
    import torch
    adapter = _load(ADAPTER_PATH, "wan_timepool_vae_adapter")
    adapter.check_env()
    adapter.pin_numerics()
    adapter.check_versions(strict=True, with_diffusers=True)
    torch.manual_seed(0)
    device = torch.device(device_name)
    vae, vinfo = adapter.load_vae(VAE_ID, device, expected_state_sha256=expected_vae_sha256)
    return adapter, vae, vinfo, device


def base_handshake(protocol, *, stub: bool) -> dict:
    return {"encoder_kind": _pool.TIMEPOOL_ENCODER_KIND, "dtype": "float32", "shape": [TOKEN_DIM],
            "response_bytes": RESPONSE_BYTES, "stub": stub, "protocol_sha256": protocol.protocol_sha256(),
            "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pool": {"version": _pool.POOL_VERSION, "source_sha256": _pool.pool_source_sha256()}}


def real_handshake_fields(vinfo: dict, flags: dict) -> dict:
    """从适配件信息里只挑白名单字段；其中的 encoder 源码摘要不外传，也不提供任何 encoder 权重身份。"""
    return {"wan": {"vae_id": vinfo["vae_id"], "vae_state_sha256": vinfo["vae_state_sha256"],
                    "adapter_sha256": _sha256(ADAPTER_PATH), "dependency_lock_sha256": _sha256(LOCK_PATH)},
            "numerics": {"dtype": "float32", "autocast": False, "batch": 1, "flags": flags},
            "environment": {**{key: vinfo.get(key) for key in ENV_KEYS}, "numpy": np.__version__}}


def stub_encoder(protocol):
    def encode(start, window):
        ids = protocol.stub_decode(window)
        if not any(ids[:r] == list(range(start, start + r))
                   and ids[r:] == [ids[r - 1]] * (33 - r) for r in range(1, 34)):
            raise ValueError("stub 帧号须为连续前缀加末帧重复")
        return np.full(TOKEN_DIM, start, dtype=np.float32)
    return encode


def serve(sock, protocol, encode) -> int:
    """循环处理请求；关闭请求返回 0，任何异常回错误包后返回 3。"""
    try:
        while True:
            request = protocol.recv_request(sock, time.monotonic() + protocol.REQUEST_TIMEOUT_S * 100)
            if request is None:
                return 0
            start, payload = request
            window = np.frombuffer(payload, dtype=np.uint8).reshape(33, 256, 256, 3)
            token = encode(start, np.ascontiguousarray(window))
            if token.shape != (TOKEN_DIM,) or token.dtype != np.float32 or not np.isfinite(token).all():
                raise ValueError("池化响应必须为有限 (1024,) float32")
            protocol.send_all(sock, protocol.encode_response(token.astype("<f4", copy=False).tobytes(),
                                                            response_bytes=RESPONSE_BYTES))
    except Exception:
        try:
            protocol.send_all(sock, protocol.encode_error(traceback.format_exc()))
        except OSError:
            pass
        return 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", required=True, type=int)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--expected-vae-sha256")
    args = parser.parse_args()
    protocol = _load(PROTOCOL_PATH, "motion_protocol")
    sock = socket.socket(fileno=args.fd)
    try:
        info = base_handshake(protocol, stub=args.stub)
        if args.stub:
            encode = stub_encoder(protocol)
        else:
            if not args.expected_vae_sha256:
                raise ValueError("真实模式必须指定 VAE 权重摘要")
            environment = os.environ.get("UV_PROJECT_ENVIRONMENT")
            if not environment or Path(sys.prefix).resolve() != Path(environment).resolve():
                raise RuntimeError("真实 sidecar 未使用指定的独立环境")
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("真实池化 sidecar 需要 CUDA")
            adapter, vae, vinfo, device = load_online_stack(args.expected_vae_sha256)
            info.update(real_handshake_fields(vinfo, adapter.pin_numerics()))
            info["gpu_uuid"] = str(torch.cuda.get_device_properties(device).uuid)

            def encode(start, window):
                return encode_window(adapter, vae, window, device)

            encode(0, np.zeros((33, 256, 256, 3), dtype=np.uint8))
            torch.cuda.synchronize(device)
        protocol.send_handshake(sock, info)
    except Exception:
        protocol.send_handshake(sock, {"error": traceback.format_exc()})
        sock.close()
        return 2
    try:
        return serve(sock, protocol, encode)
    finally:
        sock.close()


if __name__ == "__main__":
    sys.exit(main())
