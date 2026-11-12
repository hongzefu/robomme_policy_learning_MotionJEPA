"""`MotionEncoderClient`：policy 进程内的 sidecar 客户端（0901-motion-memory-plan.md 第二部分三节）。

- 进程：复制 `os.environ` 为 child env，设 `UV_LINK_MODE=copy`、`UV_PROJECT_ENVIRONMENT=$V1_STORE/venvs/wan`、`HF_HOME`、`HF_HUB_OFFLINE=1`、
  `CUDA_VISIBLE_DEVICES=<motion.online_gpu>`；argv 固定 `["uv","run","--project",scripts/dataset/wan,"--no-sync","python",motion_sidecar.py,"--fd",N,...]`
  （禁止把 `KEY=value` 塞进 argv、禁直调 venv 内 Python；禁 fork）。一个 policy 配一个子进程，跨 episode 常驻，`reset()` 不动它。
- 通道：`socket.socketpair(AF_UNIX, SOCK_STREAM)`，另一端经 `pass_fds` 交给子进程；Popen 成功后父进程立即关闭 child 副本，保证父退出时子读到 EOF。
- 握手：子进程上报 provenance（VAE / encoder / flags / 协议 sha）；与离线库 `store_meta.provenance` 逐键比对（排除 hostname / pid / 路径），任一不等 raise。
- 接口契约：`motion_enc_fn(frames: (33,256,256,3) uint8 C 连续) -> (768,) float32`；请求同步、一次一窗、`threading.Lock` 互斥。
- 只依赖 numpy 与标准库。
- 池化 kind（`encoder_kind="wan_fixedpool768"`）：argv 加 `--encoder-kind`，不传 encoder 参数；握手比 encoder_kind / readout /
  pool_source_sha256 与 vae 段，不比 encoder 段。未传 `encoder_kind` 时启动参数与握手比对逐字不变。
- raw9 kind（`encoder_kind="wan_raw9"`，计划 1112-4task-modul-wanraw9-s32-50k-plan.html 第二部分 1.1 第 17 行）：argv 同池化 kind；
  握手比 encoder_kind / readout / raw9_source_sha256 与 vae 段；`__call__` 收 589,824 B（`P.RAW9_RESPONSE_BYTES`）→
  (9,16,32,32) f32 → `wan_raw9.flatten_raw9` → 返回 (9,16384) f32 副本（每窗 9 个 token）。
- 跨硬件放行：`MMEVLA_MOTION_PROV_RELAX=gpu_name,compute_cap,sm_count` 时这三个硬件键不等只记 WARN（见 `check_provenance`）。
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
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

# 零参数池化 kind（计划 1108-4task-modul-wantp768-50k-plan.html 第二部分 1.1 第 8 行）：sidecar 只跑 Wan VAE 再池化，
# 不加载 encoder。握手只比 vae 段、池化源码 sha256 与 readout，不比 encoder 字段。值与 shared/wan_pool768.py 的
# POOL768_ENCODER_KIND 相同（这里写字面量，避免本模块在客户端侧多一层导入）。
POOL_ENCODER_KINDS = ("wan_fixedpool768",)
# 不池化 kind（计划 1112 第二部分 1.1 第 17 行）：值与 shared/wan_raw9.py 的 RAW9_ENCODER_KIND 相同；kind 白名单 = 两者并集。
RAW9_ENCODER_KINDS = ("wan_raw9",)


class MotionEncoderClient:
    def __init__(self, *, online_gpu: int | str, encoder_run_dir: str | None = None, checkpoint: str = "checkpoint_epoch_72.pt",
                 expected_ckpt_sha256: str | None = None, store_provenance: dict | None = None, stub: bool = False,
                 v1_store: str | pathlib.Path | None = None, handshake_timeout: float = P.HANDSHAKE_TIMEOUT_S,
                 source_run: str | None = None, encoder_kind: str | None = None, readout: str | None = None,
                 expected_pool_sha256: str | None = None, expected_raw9_sha256: str | None = None):
        self._lock = threading.Lock()
        self.stub = bool(stub)
        # encoder_kind=None：旧 encoder 路径，启动参数与握手比对逐字不变；池化 kind 不接受 encoder 侧参数
        if encoder_kind is not None:
            if encoder_kind not in POOL_ENCODER_KINDS + RAW9_ENCODER_KINDS:
                raise ValueError(f"未知 motion encoder_kind: {encoder_kind!r}")
            if source_run is not None or encoder_run_dir is not None or expected_ckpt_sha256 is not None:
                raise ValueError("池化 kind 不使用 source_run / encoder_run_dir / expected_ckpt_sha256")
            if encoder_kind in RAW9_ENCODER_KINDS:
                if not readout or not expected_raw9_sha256:
                    raise ValueError("raw9 kind 必须给出 readout 与 expected_raw9_sha256")
            elif not readout or not expected_pool_sha256:
                raise ValueError("池化 kind 必须给出 readout 与 expected_pool_sha256")
        self.encoder_kind = encoder_kind
        self.readout = readout
        self.expected_pool_sha256 = expected_pool_sha256
        self.expected_raw9_sha256 = expected_raw9_sha256
        # v1-store 根：显式参数 > MMEVLA_V1_STORE（worktree 开发时指到主树）> 仓库内 v1-store
        v1 = pathlib.Path(v1_store or os.environ.get("MMEVLA_V1_STORE") or (_REPO_ROOT / "v1-store"))
        if source_run is not None:
            match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)/(checkpoint_epoch_\d+\.pt)#encoder", source_run)
            if match is None:
                raise ValueError(f"motion.source_run 格式不合法: {source_run!r}")
            expected_dir = v1 / "external" / "motionjepa" / match.group(1)
            if encoder_run_dir is not None and pathlib.Path(encoder_run_dir).resolve() != expected_dir.resolve():
                raise ValueError("encoder_run_dir 与 motion.source_run 指向不同目录")
            encoder_run_dir, checkpoint = str(expected_dir), match.group(2)
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
        if encoder_kind is not None:
            argv += ["--encoder-kind", encoder_kind]
        elif not self.stub:
            argv += ["--encoder-run-dir", str(encoder_run_dir or v1 / "external/motionjepa/wan-full1600-filter2-b176x4-72ep-a"),
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
        if self.encoder_kind is not None:
            self.check_pool_handshake()
        if not self.stub:
            if store_provenance is None:
                self.close()
                raise RuntimeError("真编码器模式必须提供离线库 store_provenance 以核对同源")
            self.check_provenance(store_provenance)
        self.n_calls = 0
        self.total_s = 0.0

    def check_pool_handshake(self) -> None:
        """池化 / raw9 kind 的握手比对：sidecar 上报的 encoder_kind / readout / 源码 sha256 与 run 快照逐值相等
        （池化 kind 比 pool_source_sha256，raw9 kind 比 raw9_source_sha256）。"""
        diffs = []
        sha_item = (("raw9_source_sha256", self.expected_raw9_sha256) if self.encoder_kind in RAW9_ENCODER_KINDS
                    else ("pool_source_sha256", self.expected_pool_sha256))
        for key, want in (("encoder_kind", self.encoder_kind), ("readout", self.readout), sha_item):
            got = self.provenance.get(key)
            if got != want:
                diffs.append(f"{key}: sidecar={got!r} 期望={want!r}")
        if diffs:
            self.close()
            raise RuntimeError("sidecar 池化身份与 run 快照不符:\n  " + "\n  ".join(diffs))

    def check_provenance(self, store_prov: dict) -> None:
        """逐键比对 sidecar 与离线库 provenance。
        环境变量 `MMEVLA_MOTION_PROV_RELAX`（逗号分隔键名，如 `gpu_name,compute_cap,sm_count`）里列出的键
        不等时只打印 `MOTION_PROV_RELAXED …` 到 stderr 并记入 `self.relaxed_diffs`，不 raise；其余键不等仍 raise。
        用途：评估侧跨硬件复用训练侧 run（训练在 A100、评估在 A40 / RTX 6000 Ada 时这三个硬件键必然不等，
        driver / torch / cuda / cudnn / diffusers 等软件键仍必须逐值相同）。默认不设即全键硬比对。
        池化 kind 只比 vae 段（没有 encoder），且 vae 段排除 `encoder_src_sha256`：该键是学习型 encoder 模块
        `motion_jepa.models.wan_latent_encoder` 源文件的 sha256，池化分支不使用它。"""
        relax = {k.strip() for k in os.environ.get("MMEVLA_MOTION_PROV_RELAX", "").split(",") if k.strip()}
        diffs, relaxed = [], []
        sections = (("vae", tuple(k for k in PROV_SAME_VAE if k != "encoder_src_sha256")),) \
            if getattr(self, "encoder_kind", None) is not None \
            else (("vae", PROV_SAME_VAE), ("encoder", PROV_SAME_ENC))
        for section, keys in sections:
            a, b = self.provenance.get(section, {}), store_prov.get(section, {})
            for k in keys:
                if a.get(k) != b.get(k):
                    line = f"{section}.{k}: sidecar={a.get(k)!r} store={b.get(k)!r}"
                    (relaxed if k in relax else diffs).append(line)
        for line in relaxed:
            print("MOTION_PROV_RELAXED", line, file=sys.stderr, flush=True)
        self.relaxed_diffs = relaxed
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
            raw9 = getattr(self, "encoder_kind", None) in RAW9_ENCODER_KINDS
            raw = P.recv_response(self._sock, deadline, P.RAW9_RESPONSE_BYTES) if raw9 else P.recv_response(self._sock, deadline)
            self.total_s += time.perf_counter() - t0
            self.n_calls += 1
        if raw9:
            # 589,824 B → (9,16,32,32) f32 副本 → C→H→W 展平为 (9,16384)（零拷贝视图，不改字节）
            from mme_vla_suite.shared import wan_raw9
            z = np.frombuffer(raw, dtype=np.float32).reshape(wan_raw9.RAW9_INPUT_SHAPE).copy()
            return wan_raw9.flatten_raw9(z)
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
