"""wan 子项目（torch 侧）三个 worker 脚本的公共件：清单读取、段工作项、网格公式、claim 领任务、原子落盘、provenance。

只依赖 stdlib + numpy + h5py：本目录运行在 `v1-store/venvs/wan`（torch 2.9 栈），装不下主 venv 的
`mme_vla_suite`（其包入口拉 jax / ml_dtypes），所以网格公式在此**独立实现**；与
`src/mme_vla_suite/datastore/motion_store.py` 的常量 / 公式必须逐项同值，由
`scripts/dataset/test_guards.py` 在主 venv 里断言（两份实现互为对照，不是复制粘贴的借口）。

段工作项键：``<Task>_ep<j>_<exec|demo>``，demo = 全域 ``[0, es)``、exec = ``[es, T)``；
每段网格起点 ``0, 16, 32, …``（段内绝对位置），起点数 ``len(range(0, max(0, L-32), 16))``。
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import platform
import socket
import subprocess
import time

import numpy as np

GRID_STRIDE = 16
WINDOW_FRAMES = 33
GRID_ORIGIN = "segment_start"
WINDOW_DIRECTION = "forward"
TRUNCATION_POLICY = "none"
FRAME_SIZE = 256
LAT_SHAPE = (9, 16, 32, 32)                 # 组优先 latent，f32
CHUNK_BYTES = int(np.prod(LAT_SHAPE)) * 4    # 589,824
TOKEN_DIM = 768
TOKEN_BYTES = TOKEN_DIM * 4                 # 3,072
SEGMENTS = ("demo", "exec")


# ── 清单（与 mme_vla_suite.datastore.manifest 同口径，独立实现）────────────────


def manifest_sha256(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != "sha256"}
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_manifest(path: str | pathlib.Path) -> dict:
    p = pathlib.Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    expect = manifest_sha256(payload)
    if payload.get("sha256") != expect:
        raise ValueError(f"清单 sha256 不符（已被改动？）: {p}: 记录 {payload.get('sha256')} != 现算 {expect}")
    return payload


# ── 网格公式 ──────────────────────────────────────────────────────────────────


def seg_num_chunks(seg_len: int) -> int:
    return max(0, int(seg_len) - (WINDOW_FRAMES - 1))


def seg_num_grid(seg_len: int) -> int:
    return len(range(0, seg_num_chunks(seg_len), GRID_STRIDE))


def task_of_h5(h5_file: str) -> str:
    name = os.path.basename(h5_file)
    if not (name.startswith("record_dataset_") and name.endswith(".h5")):
        raise ValueError(f"h5 文件名不符合 record_dataset_<Task>.h5: {h5_file}")
    return name[len("record_dataset_"):-len(".h5")]


def list_segments(manifest: dict) -> list[dict]:
    """全部 num_grid > 0 的段工作项（清单序）。每项：key / g / h5_file / raw_ep_idx / segment /
    seg_start（全域帧号）/ seg_len / num_grid / num_chunks。"""
    items: list[dict] = []
    for ep in manifest["episodes"]:
        nt, es = int(ep["num_timesteps"]), int(ep["exec_start_idx"])
        for seg, start, L in (("demo", 0, es), ("exec", es, nt - es)):
            ng = seg_num_grid(L)
            if ng == 0:
                continue
            items.append({
                "key": f"{task_of_h5(ep['h5_file'])}_ep{int(ep['raw_ep_idx'])}_{seg}",
                "g": int(ep["global_episode_idx"]), "h5_file": str(ep["h5_file"]),
                "raw_ep_idx": int(ep["raw_ep_idx"]), "segment": seg,
                "seg_start": start, "seg_len": L, "num_grid": ng, "num_chunks": seg_num_chunks(L),
            })
    return items


def lpt_order(items: list[dict], weight_key: str = "num_grid") -> list[dict]:
    """按权重降序排队（同权重按 key 稳定）；动态领任务下与静态 LPT 装箱等效。"""
    return sorted(items, key=lambda it: (-int(it[weight_key]), it["key"]))


# ── h5 帧读取 ─────────────────────────────────────────────────────────────────


def read_segment_frames(h5_path: str, raw_ep_idx: int, seg_start: int, seg_len: int) -> np.ndarray:
    """一次读入整段 ``obs/front_rgb`` → (L,256,256,3) uint8 C 连续。"""
    import h5py
    out = np.empty((seg_len, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    with h5py.File(h5_path, "r") as f:
        ep = f[f"episode_{raw_ep_idx}"]
        for i in range(seg_len):
            ds = ep[f"timestep_{seg_start + i}/obs/front_rgb"]
            if ds.shape != (FRAME_SIZE, FRAME_SIZE, 3) or ds.dtype != np.uint8:
                raise ValueError(f"{h5_path} episode_{raw_ep_idx} t={seg_start + i} 帧形制异常 "
                                 f"{ds.shape} {ds.dtype}")
            ds.read_direct(out[i])
    return out


def sha256_bytes(arr_or_bytes) -> str:
    if isinstance(arr_or_bytes, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(arr_or_bytes).tobytes()).hexdigest()
    return hashlib.sha256(arr_or_bytes).hexdigest()


def sha256_file(p: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── claim 领任务 / 原子落盘 ───────────────────────────────────────────────────


def try_claim(claims_dir: pathlib.Path, key: str, worker: str) -> bool:
    """``os.open(O_CREAT|O_EXCL)`` 领一项；已被领走返回 False。"""
    claims_dir.mkdir(parents=True, exist_ok=True)
    p = claims_dir / f"_claim_{key}"
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, f"{worker} pid={os.getpid()} {datetime.datetime.now().isoformat(timespec='seconds')}\n".encode())
    finally:
        os.close(fd)
    return True


def release_claim(claims_dir: pathlib.Path, key: str) -> None:
    (claims_dir / f"_claim_{key}").unlink(missing_ok=True)


def atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def write_json(path: pathlib.Path, obj) -> None:
    atomic_write_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def segment_outputs_complete(out_dir: pathlib.Path, key: str, expect_bytes: int) -> bool:
    """续跑判据：``.bin`` 存在且字节数相符 + ``.sha256`` sidecar 相符 + metadata 可解析。"""
    b = out_dir / f"{key}.bin"
    s = out_dir / f"{key}.bin.sha256"
    m = out_dir / f"{key}.metadata.json"
    if not (b.is_file() and s.is_file() and m.is_file()):
        return False
    if b.stat().st_size != expect_bytes:
        return False
    try:
        want = s.read_text().split()[0]
        json.loads(m.read_text(encoding="utf-8"))
    except (OSError, ValueError, IndexError):
        return False
    return sha256_file(b) == want


def purge_segment_outputs(out_dir: pathlib.Path, key: str) -> None:
    for suffix in (".bin", ".bin.sha256", ".metadata.json"):
        (out_dir / f"{key}{suffix}").unlink(missing_ok=True)
    for tmp in out_dir.glob(f"{key}.bin.tmp.*"):
        tmp.unlink(missing_ok=True)


# ── provenance ────────────────────────────────────────────────────────────────


def git_commit(repo: pathlib.Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30, check=False).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def gpu_uuid_of_visible() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if vis.strip():
            first = vis.split(",")[0].strip()
            for line in out:
                i, u = [s.strip() for s in line.split(",")]
                if i == first or u == first:
                    return u
        return out[0].split(",")[1].strip() if out else "unknown"
    except Exception:
        return "unknown"


def worker_fingerprint(repo_root: pathlib.Path, mj_commit: str, extra: dict | None = None) -> dict:
    """逐 worker 指纹（1.3）：gpu_name / compute_cap / gpu_uuid / driver / torch / cudnn / hostname /
    cuda_visible_devices / git_commit / mj_commit / pid。"""
    import torch
    props = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        driver = "unknown"
    fp = {
        "gpu_name": props.name, "compute_cap": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count, "gpu_uuid": gpu_uuid_of_visible(),
        "driver_version": driver, "torch": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(), "python": platform.python_version(),
        "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": git_commit(repo_root), "mj_commit": mj_commit, "pid": os.getpid(),
    }
    if extra:
        fp.update(extra)
    return fp


def load_source_pin(here: pathlib.Path) -> dict:
    pin = json.loads((here / "SOURCE_PIN.json").read_text(encoding="utf-8"))
    got = sha256_file(here / "wan_motion_infer.py")
    if got != pin["source_sha256"]:
        raise SystemExit(f"复制件 wan_motion_infer.py sha256 {got[:16]}… != SOURCE_PIN.source_sha256 "
                         f"{pin['source_sha256'][:16]}…（复制件被改动）")
    return pin


class Progress:
    def __init__(self, tag: str):
        self.tag = tag
        self.t0 = time.perf_counter()
        self.items = 0
        self.windows = 0

    def done(self, key: str, n_windows: int, elapsed: float) -> None:
        self.items += 1
        self.windows += n_windows
        print(f"[{self.tag}] {key} windows={n_windows} took={elapsed:.1f}s "
              f"({elapsed / max(1, n_windows) * 1e3:.0f} ms/窗) 累计 items={self.items} windows={self.windows} "
              f"elapsed={time.perf_counter() - self.t0:.0f}s", flush=True)
