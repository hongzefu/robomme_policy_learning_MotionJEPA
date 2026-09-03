"""`MotionEncoderClient`：policy 进程内的 sidecar 客户端（motion-memory-plan.md 第二部分三节）。

- 进程：复制 `os.environ` 为 child env，设 `UV_LINK_MODE=copy`、`UV_PROJECT_ENVIRONMENT=$V1_STORE/venvs/wan`、`HF_HOME`、`HF_HUB_OFFLINE=1`、
  `CUDA_VISIBLE_DEVICES=<motion.online_gpu>`；argv 固定 `["uv","run","--project",scripts/dataset/wan,"--no-sync","python",motion_sidecar.py,"--fd",N,...]`
  （禁止把 `KEY=value` 塞进 argv、禁直调 venv 内 Python；禁 fork）。一个 policy 配一个子进程，跨 episode 常驻，`reset()` 不动它。
- 通道：`socket.socketpair(AF_UNIX, SOCK_STREAM)`，另一端经 `pass_fds` 交给子进程；Popen 成功后父进程立即关闭 child 副本，保证父退出时子读到 EOF。
- 握手：子进程上报 provenance（VAE / encoder / flags / 协议 sha）；与离线库 `store_meta.provenance` 逐键比对（排除 hostname / pid / 路径），任一不等 raise。
- 接口契约：`motion_enc_fn(frames: (33,256,256,3) uint8 C 连续) -> (768,) float32`；请求同步、一次一窗、`threading.Lock` 互斥。
- 只依赖 numpy 与标准库。
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time

import numpy as np

from mme_vla_suite.policies import motion_protocol as P

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# 与离线库 store_meta.provenance 必须逐键相等的键——与打包器 pack_motion_store.gather_provenance 的 same_keys_* 同一份清单
# （含同机同型号卡的 gpu_name / compute_cap / sm_count / driver；排除 hostname / pid / run_dir 等进程与路径键）。
# store 侧 encoder.flags 由打包器从 encoder_flags 并入，sidecar 侧 einfo 自带 flags，两侧同键比较。
PROV_SAME_VAE = ("vae_id", "vae_state_sha256", "vae_dtype", "latent_mode", "batch", "tf32", "amp",
                 "torch", "cuda", "cudnn", "diffusers", "cublas_pkg", "cudnn_pkg", "module_sha256",
                 "encoder_src_sha256", "flags", "env", "gpu_name", "compute_cap", "sm_count", "driver")
PROV_SAME_ENC = ("checkpoint", "checkpoint_sha256", "checkpoint_epoch", "arch", "state_key", "precision",
                 "amp", "tf32", "batch", "vae_id", "motion_dims", "torch", "cuda", "cudnn", "diffusers",
                 "cublas_pkg", "cudnn_pkg", "module_sha256", "encoder_src_sha256", "env", "gpu_name",
                 "compute_cap", "sm_count", "driver", "flags")


class MotionEncoderClient:
    def __init__(self, *, online_gpu: int | str, encoder_run_dir: str | None = None, checkpoint: str = "checkpoint_epoch_72.pt",
                 expected_ckpt_sha256: str | None = None, store_provenance: dict | None = None, stub: bool = False,
                 v1_store: str | pathlib.Path | None = None, handshake_timeout: float = P.HANDSHAKE_TIMEOUT_S):
        self._lock = threading.Lock()
        self.stub = bool(stub)
        # v1-store 根：显式参数 > MMEVLA_V1_STORE（worktree 开发时指到主树）> 仓库内 v1-store
        v1 = pathlib.Path(v1_store or os.environ.get("MMEVLA_V1_STORE") or (_REPO_ROOT / "v1-store"))
        wan_dir = _REPO_ROOT / "scripts" / "dataset" / "wan"
        child_env = dict(os.environ)
        child_env["UV_LINK_MODE"] = "copy"
        child_env["UV_PROJECT_ENVIRONMENT"] = str(v1 / "venvs" / "wan")
        child_env["HF_HOME"] = str(v1 / "cache" / "hf")
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["CUDA_VISIBLE_DEVICES"] = str(online_gpu)
        child_env["PYTHONUNBUFFERED"] = "1"
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        argv = ["uv", "run", "--project", str(wan_dir), "--no-sync", "python", str(wan_dir / "motion_sidecar.py"),
                "--fd", str(child_sock.fileno())]
        if self.stub:
            argv.append("--stub")
        else:
            argv += ["--encoder-run-dir", str(encoder_run_dir or v1 / "external/motionjepa/wan-v8-filter10-72ep-a"),
                     "--checkpoint", checkpoint]
            if expected_ckpt_sha256:
                argv += ["--expected-ckpt-sha256", expected_ckpt_sha256]
        self.argv = argv
        self._proc = subprocess.Popen(argv, env=child_env, pass_fds=[child_sock.fileno()], cwd=str(_REPO_ROOT),
                                      stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr)
        child_sock.close()                                   # 父进程立即关闭 child 副本
        self._sock = parent_sock
        self._sock.setblocking(True)
        deadline = time.monotonic() + handshake_timeout
        try:
            self.provenance = P.recv_handshake(self._sock, deadline)
        except Exception as e:
            self.close()
            raise RuntimeError(f"sidecar 握手失败（argv={argv}）: {e}") from e
        if "error" in self.provenance:
            self.close()
            raise RuntimeError("sidecar 启动失败:\n" + str(self.provenance["error"]))
        if self.provenance.get("protocol_sha256") != P.protocol_sha256():
            self.close()
            raise RuntimeError("sidecar 加载的 motion_protocol.py 与主侧不是同一份（sha256 不同）")
        if bool(self.provenance.get("stub")) != self.stub:
            self.close()
            raise RuntimeError("sidecar 的 stub 标记与客户端不一致")
        if not self.stub:
            if store_provenance is None:
                self.close()
                raise RuntimeError("真编码器模式必须提供离线库 store_provenance 以核对同源")
            self.check_provenance(store_provenance)
        self.n_calls = 0
        self.total_s = 0.0

    def check_provenance(self, store_prov: dict) -> None:
        diffs = []
        for section, keys in (("vae", PROV_SAME_VAE), ("encoder", PROV_SAME_ENC)):
            a, b = self.provenance.get(section, {}), store_prov.get(section, {})
            for k in keys:
                if a.get(k) != b.get(k):
                    diffs.append(f"{section}.{k}: sidecar={a.get(k)!r} store={b.get(k)!r}")
        if diffs:
            self.close()
            raise RuntimeError("sidecar 与离线库 provenance 不同源:\n  " + "\n  ".join(diffs))

    def __call__(self, frames: np.ndarray, start_frame: int = 0) -> np.ndarray:
        frames = np.asarray(frames)
        if frames.shape != (P.WINDOW_FRAMES, P.FRAME_SIZE, P.FRAME_SIZE, 3) or frames.dtype != np.uint8:
            raise ValueError(f"motion_enc_fn 输入须为 (33,256,256,3) uint8，得到 {frames.shape} {frames.dtype}")
        payload = np.ascontiguousarray(frames).tobytes()
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError(f"sidecar 已退出（rc={self._proc.returncode}）")
            deadline = time.monotonic() + P.REQUEST_TIMEOUT_S
            t0 = time.perf_counter()
            P.send_all(self._sock, P.encode_request(start_frame, payload))
            raw = P.recv_response(self._sock, deadline)
            self.total_s += time.perf_counter() - t0
            self.n_calls += 1
        return np.frombuffer(raw, dtype=np.float32).copy()

    def close(self) -> None:
        try:
            if getattr(self, "_sock", None) is not None:
                try:
                    P.send_all(self._sock, P.encode_shutdown())
                except Exception:
                    pass
                self._sock.close()
                self._sock = None
        finally:
            p = getattr(self, "_proc", None)
            if p is not None and p.poll() is None:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
