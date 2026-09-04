#!/usr/bin/env python3
"""在线 motion 编码进程（sidecar，motion-memory-plan.md 第二部分三节）：收 33 帧、还 768 个数，一次一窗、B=1。

由 `MotionEncoderClient` 用 `subprocess.Popen` 起（argv 从 `uv run --project scripts/dataset/wan --no-sync python motion_sidecar.py --fd N` 开始，
`socketpair` 的一端经 `pass_fds` 交给本进程）。起手 `check_env()` + `pin_numerics()` + `check_versions()`，`load_vae(expected_state_sha256)`、
`load_encoder(expected_sha256)`，把 `provenance()` + 协议文件 sha256 发回；用一窗全零帧预热（结果丢弃）；之后循环收请求 → 复制件
`encode_chunk` + `motion_token` → 回 3,072 B。stdout 不用于协议，日志全走 stderr。

`--stub`：不加载模型、不 import torch；收到帧后按 P 系列约定解出 33 个全域帧号（通道 0 低 8 位、通道 1 高 8 位、通道 2 更高位），
校验连续且首帧 == 请求里的起点，返回 `np.full(768, 起点, float32)`；P1–P4 用它走完整 IPC 路径。

协议公共件从 `<repo>/src/mme_vla_suite/policies/motion_protocol.py`（由 `Path(__file__).resolve().parents[3]` 定位）按文件加载，不 import 整个包。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import socket
import sys
import time
import traceback

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_PROTO_PATH = _REPO_ROOT / "src" / "mme_vla_suite" / "policies" / "motion_protocol.py"
VAE_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"[sidecar pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fd", type=int, required=True, help="父进程经 pass_fds 传入的 socket fd")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--encoder-run-dir", default=str(_REPO_ROOT / "v1-store/external/motionjepa/wan-v8-filter10-72ep-a"))
    ap.add_argument("--checkpoint", default="checkpoint_epoch_72.pt")
    ap.add_argument("--expected-ckpt-sha256", default="")
    ap.add_argument("--expected-vae-sha256", default="")
    args = ap.parse_args()

    P = _load(_PROTO_PATH, "motion_protocol")
    sock = socket.socket(fileno=args.fd)
    sock.setblocking(True)
    deadline = time.monotonic() + P.HANDSHAKE_TIMEOUT_S
    encode = None
    try:
        if args.stub:
            info = {"stub": True, "protocol_sha256": P.protocol_sha256(), "protocol_path": str(_PROTO_PATH), "pid": os.getpid()}

            def encode(start: int, window: np.ndarray) -> np.ndarray:
                ids = P.stub_decode(window)
                if ids != list(range(start, start + P.WINDOW_FRAMES)):
                    raise RuntimeError(f"stub: 33 帧编号不连续或与起点 {start} 不符: {ids[:4]}…{ids[-2:]}")
                return np.full(P.TOKEN_DIM, float(start), np.float32)
        else:
            import torch
            W = _load(_HERE / "wan_motion_infer.py", "wan_motion_infer")
            pin = json.loads((_HERE / "SOURCE_PIN.json").read_text(encoding="utf-8"))
            if W.sha256_file(str(_HERE / "wan_motion_infer.py")) != pin["source_sha256"]:
                raise RuntimeError("复制件 wan_motion_infer.py 与 SOURCE_PIN 不符")
            W.check_env()
            flags = W.pin_numerics()
            W.check_versions(strict=True, with_diffusers=True)
            torch.manual_seed(0)
            if not torch.cuda.is_available():
                raise RuntimeError("sidecar 需要 CUDA")
            device = torch.device("cuda")
            vae, vinfo = W.load_vae(VAE_ID, device, expected_state_sha256=(args.expected_vae_sha256 or W.VAE_STATE_SHA256_EXPECTED))
            # ckpt 期望值与 VAE 侧对称（此前 VAE 有兜底硬钉、ckpt 却 `or None` 静默跳过）。
            # 调用方传的是数据集自报的 provenance.encoder.checkpoint_sha256；与仓库锁不符即说明该库是用
            # 另一份 encoder 建的 —— 在线推理与离线建库不同源，必须当场拒绝而不是各校各的。
            lock_mod = _load(_REPO_ROOT / "scripts" / "assets" / "assets_lock.py", "assets_lock")
            lock_ckpt = lock_mod.expected_sha256("motionjepa_ckpt")
            if args.expected_ckpt_sha256 and args.expected_ckpt_sha256 != lock_ckpt:
                raise SystemExit(f"数据集 provenance 记的 ckpt {args.expected_ckpt_sha256} 与仓库 "
                                 f"ASSETS_LOCK {lock_ckpt} 不符：该数据集是用另一份 encoder 建的，"
                                 f"在线推理与离线建库不同源")
            encoder, einfo, use_amp = W.load_encoder(args.encoder_run_dir, args.checkpoint, device,
                                                     expected_sha256=(args.expected_ckpt_sha256 or lock_ckpt))
            info = {"stub": False, "protocol_sha256": P.protocol_sha256(), "protocol_path": str(_PROTO_PATH), "pid": os.getpid(),
                    "vae": vinfo, "encoder": einfo, "flags": flags, "source_pin": pin,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

            def encode(start: int, window: np.ndarray) -> np.ndarray:
                W.pin_numerics()
                lat = W.encode_chunk(vae, window, device)        # 🔒 33 帧一次喂、B=1、fp32、关 TF32
                return W.motion_token(encoder, lat, use_amp, device).astype(np.float32)

            # 预热：一窗全零帧，结果丢弃（pin_numerics 已关 benchmark，预热不改后续数值）
            t0 = time.perf_counter()
            encode(0, np.zeros((P.WINDOW_FRAMES, P.FRAME_SIZE, P.FRAME_SIZE, 3), np.uint8))
            torch.cuda.synchronize(device)
            info["warmup_s"] = time.perf_counter() - t0
            log(f"模型就绪 warmup={info['warmup_s']:.2f}s vae={vinfo['vae_state_sha256'][:16]}… ckpt={einfo['checkpoint_sha256'][:16]}…")
        P.send_handshake(sock, info)
    except Exception:
        log("握手前失败:\n" + traceback.format_exc())
        try:
            P.send_handshake(sock, {"error": traceback.format_exc()})
        except Exception:
            pass
        return 2

    n = 0
    t_serve = time.perf_counter()
    while True:
        try:
            req = P.recv_request(sock, time.monotonic() + P.REQUEST_TIMEOUT_S * 100)   # 空闲等待不计超时；单次请求内 recv 由 deadline 兜底
        except P.ProtocolError as e:
            log(f"请求接收失败，退出: {e}")
            return 3
        if req is None:
            log(f"收到关闭请求，已服务 {n} 窗，退出")
            return 0
        start, payload = req
        try:
            window = np.frombuffer(payload, dtype=np.uint8).reshape(P.WINDOW_FRAMES, P.FRAME_SIZE, P.FRAME_SIZE, 3)
            window = np.ascontiguousarray(window)
            t0 = time.perf_counter()
            tok = encode(start, window)
            dt = time.perf_counter() - t0
            if tok.shape != (P.TOKEN_DIM,) or tok.dtype != np.float32:
                raise RuntimeError(f"token 形制 {tok.shape} {tok.dtype}")
            P.send_all(sock, P.encode_response(np.ascontiguousarray(tok).tobytes()))
            n += 1
            if n % 50 == 1:
                log(f"窗 #{n} 起点 {start} 耗时 {dt * 1e3:.0f} ms（累计 {time.perf_counter() - t_serve:.0f}s）")
        except Exception:
            err = traceback.format_exc()
            log(f"编码失败（起点 {start}）:\n{err}")
            try:
                P.send_all(sock, P.encode_error(err))
            except Exception:
                pass
            return 4


if __name__ == "__main__":
    sys.exit(main())
