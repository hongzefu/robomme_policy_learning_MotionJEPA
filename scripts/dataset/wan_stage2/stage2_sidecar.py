#!/usr/bin/env python3
"""阶段二独立在线进程：33 帧经旧 VAE 适配件和共享 EMA encoder 返回 4096 字节。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENCODER_KIND = "stage2_pool4_concat"
RESPONSE_BYTES = 4096


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_source_pin() -> dict:
    """先核本机适配件和独立锁文件；实际 MJ 导入身份在模型加载前另核。"""
    pin = json.loads((HERE / "SOURCE_PIN.json").read_text())
    for key, path in (("vae_adapter_sha256", HERE.parent / "wan/wan_motion_infer.py"),
                      ("dependency_lock_sha256", HERE / "uv.lock")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin[key]:
            raise ValueError(f"阶段二来源锁不符: {key}")
    return pin


def encode_latent(encoder, latent) -> np.ndarray:
    """生产与同 latent 对拍共用入口；一次只编码一个完整窗口。"""
    import torch
    from motion_jepa.export.stage2_memory import encode_windows
    if tuple(latent.shape) != (9, 16, 32, 32) or latent.dtype != torch.float32:
        raise ValueError("阶段二 latent 必须为 (9,16,32,32) float32")
    return encode_windows(encoder, latent.contiguous().unsqueeze(0))[0].detach().cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", required=True, type=int)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--ckpt")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-vae-sha256")
    args = parser.parse_args()
    protocol = _load(ROOT / "src/mme_vla_suite/policies/motion_protocol.py", "motion_protocol")
    sock = socket.socket(fileno=args.fd)
    try:
        info = {"encoder_kind": ENCODER_KIND, "dtype": "float32", "shape": [1024],
                "response_bytes": RESPONSE_BYTES, "stub": args.stub,
                "protocol_sha256": protocol.protocol_sha256(), "pid": os.getpid(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        if args.stub:
            def encode(start, window):
                ids = protocol.stub_decode(window)
                if not any(ids[:r] == list(range(start, start + r))
                           and ids[r:] == [ids[r - 1]] * (33 - r) for r in range(1, 34)):
                    raise ValueError("stub 帧号须为连续前缀加末帧重复")
                return np.full(1024, start, dtype=np.float32)
        else:
            if not args.ckpt or not args.expected_sha256 or not args.expected_vae_sha256:
                raise ValueError("真实模式必须指定 checkpoint 与两份权重摘要")
            environment = os.environ.get("UV_PROJECT_ENVIRONMENT")
            if not environment or Path(sys.prefix).resolve() != Path(environment).resolve():
                raise RuntimeError("真实 sidecar 未使用指定的独立阶段二环境")
            pin = checked_source_pin()
            adapter = _load(HERE.parent / "wan/wan_motion_infer.py", "wan_stage2_vae_adapter")
            adapter.check_env()
            adapter.pin_numerics()
            adapter.check_versions(strict=True, with_diffusers=True)
            import torch
            from motion_jepa.export import stage2_memory as shared
            actual_pin = shared.source_pin()
            if actual_pin != {key: pin[key] for key in actual_pin} or len(actual_pin) != 3:
                raise ValueError("实际导入的阶段二共享模块与 SOURCE_PIN 不同源")
            if not torch.cuda.is_available():
                raise RuntimeError("真实阶段二 sidecar 需要 CUDA")
            torch.manual_seed(0)
            device = torch.device("cuda:0")
            encoder, einfo = shared.load_ema_encoder(args.ckpt, device=str(device),
                                                    expected_sha256=args.expected_sha256)
            vae, vinfo = adapter.load_vae("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", device,
                                         expected_state_sha256=args.expected_vae_sha256)
            online_vae = {key: vinfo[key] for key in
                          ("vae_id", "vae_state_sha256", "vae_dtype", "latent_mode", "batch", "tf32", "amp")}
            online_vae.update(adapter_sha256=pin["vae_adapter_sha256"],
                              preprocess={"frame_size": 256, "input_dtype": "uint8",
                                          "input_layout": "THWC", "window_frames": 33},
                              numerics=einfo["numerics"])
            info.update({key: einfo[key] for key in ("encoder", "encoder_state_sha256", "numerics")})
            info.update(source_pin=pin, online_vae=online_vae, vae=vinfo,
                        gpu_uuid=str(torch.cuda.get_device_properties(device).uuid))

            def encode(start, window):
                adapter.pin_numerics()
                return encode_latent(encoder, adapter.encode_chunk(vae, window, device))

            encode(0, np.zeros((33, 256, 256, 3), dtype=np.uint8))
            torch.cuda.synchronize(device)
        protocol.send_handshake(sock, info)
    except Exception:
        protocol.send_handshake(sock, {"error": traceback.format_exc()})
        sock.close()
        return 2
    try:
        while True:
            request = protocol.recv_request(sock, time.monotonic() + protocol.REQUEST_TIMEOUT_S * 100)
            if request is None:
                return 0
            start, payload = request
            window = np.frombuffer(payload, dtype=np.uint8).reshape(33, 256, 256, 3)
            token = encode(start, np.ascontiguousarray(window))
            if token.shape != (1024,) or token.dtype != np.float32 or not np.isfinite(token).all():
                raise ValueError("阶段二响应必须为有限 (1024,) float32")
            protocol.send_all(sock, protocol.encode_response(token.astype("<f4", copy=False).tobytes(),
                                                            response_bytes=RESPONSE_BYTES))
    except Exception:
        try:
            protocol.send_all(sock, protocol.encode_error(traceback.format_exc()))
        except OSError:
            pass
        return 3
    finally:
        sock.close()


if __name__ == "__main__":
    sys.exit(main())
