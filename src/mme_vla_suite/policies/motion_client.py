"""`MotionEncoderClient`：policy 进程内的 sidecar 客户端（docs/root-archive/0901-motion-memory-plan.md 第二部分三节）。

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
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np

from mme_vla_suite.policies import motion_protocol as P
from mme_vla_suite.shared import wan_timepool as _timepool

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

STAGE2_KIND = "stage2_pool4_concat"
STAGE2_GROUPS = ("encoder", "encoder_state_sha256", "source_pin", "numerics", "stage2_compat", "vae")
TIMEPOOL_KIND = _timepool.TIMEPOOL_ENCODER_KIND
# 独立会话、进程归属、4096 字节响应这一套 worker 管理由阶段二与固定池化共用。
SESSION_KINDS = (STAGE2_KIND, TIMEPOOL_KIND)
TIMEPOOL_WAN_KEYS = ("vae_id", "vae_state_sha256", "adapter_sha256", "dependency_lock_sha256")
TIMEPOOL_SNAPSHOT_SCHEMA = "wan-timepool-motion-snapshot-v1"
TIMEPOOL_WINDOW_MAP_COPY = "timepool_window_map.json"
TIMEPOOL_ONLINE_RECORD = "online_restore_inputs.json"
TIMEPOOL_ONLINE_PAYLOAD = "online_restore_inputs.npz"
TIMEPOOL_ONLINE_SCHEMA = "stage2-online-restore-inputs-v1"
# run 根 motion_provenance.json 的完整键集合（init_history_config 基础键 + timepool 段）。
TIMEPOOL_SNAPSHOT_PROV_KEYS = frozenset((
    "history_config_name", "motion_enabled", "resolved_sha256", "framesamp_root", "framesamp_manifest_sha256",
    "framesamp_store_meta_sha256", "motion_root", "motion_manifest_sha256", "motion_index_sha256",
    "motion_store_meta_sha256", "motion_table_sha256", "vae", "encoder", "timepool"))
_ENCODER_FIELD = re.compile(r"(^|_)(encoder|ckpt|checkpoint)(_|$)")


def _reject_encoder_fields(value, path: str = "握手") -> None:
    """固定池化没有 encoder：除身份键 encoder_kind 外，任意层级不得出现 encoder 或权重字段。"""
    if isinstance(value, dict):
        for key, item in value.items():
            if key != "encoder_kind" and _ENCODER_FIELD.search(str(key)):
                raise ValueError(f"固定池化{path}不得带 encoder 字段: {key}")
            _reject_encoder_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for item in value:
            _reject_encoder_fields(item, path)


def _live_process(pid: int) -> tuple[int, int]:
    """读取 Linux 进程父 PID 与启动时钟；退出、僵尸和非法 PID 均不算存活。"""
    if type(pid) is not int or pid <= 0:
        raise ValueError("阶段二 worker 进程身份必须为正整数 PID")
    try:
        fields = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X", "x"):
            raise ValueError("阶段二 worker 进程身份已退出")
        return int(fields[1]), int(fields[19])
    except (OSError, IndexError) as exc:
        raise ValueError("阶段二 worker 进程身份不存在或不可读") from exc


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_path(value: str, root: pathlib.Path | None) -> pathlib.Path:
    """绝对路径仅供本地建库；交付引用必须在根目录内解析。"""
    path = pathlib.Path(value)
    if ".." in path.parts:
        raise ValueError("阶段二证据路径禁止 ..")
    if path.is_absolute():
        return path.resolve(strict=True)
    if root is None:
        raise ValueError("相对证据路径必须指定 provenance_root")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("阶段二证据路径越过 provenance_root")
    return resolved


def validate_stage2_evidence(provenance: dict, root: pathlib.Path | None) -> None:
    """实际读取引用证据并核内容身份；历史 VAE 身份不改写成当前源码。"""
    compat = provenance["stage2_compat"]
    evidence = compat["evidence"]
    from mme_vla_suite.datastore import motion_store as ms
    records = {}
    for key in ms.stage2_evidence_roles(provenance):
        ref = evidence[key]
        path = _evidence_path(ref["path"], root)
        if _sha256(path) != ref["sha256"]:
            raise ValueError(f"阶段二证据 SHA 不符: {key}")
        records[key] = json.loads(path.read_text(encoding="utf-8"))
    if ms.stage2_uses_cache(provenance):
        ms.validate_stage2_cache_evidence(provenance, records)
        return
    if records["SOURCE_PIN"] != provenance["source_pin"]:
        raise ValueError("证据 SOURCE_PIN 与来源不一致")
    if evidence["restore_receipt"]["sha256"] != compat["restore_receipt_sha256"]:
        raise ValueError("恢复 receipt 与兼容合同不一致")
    receipt = records["restore_receipt"]
    historical = receipt.get("vae", {})
    if any(provenance["vae"].get(key) != value for key, value in compat["historical_vae"].items()):
        raise ValueError("阶段二 vae 来源与历史 receipt 身份不一致")
    if (receipt.get("schema_version") != "mixed8-vla-latent-restore/v1"
            or receipt.get("status") != "PASS" or receipt.get("new_encoder_identity") is not None
            or historical.get("identity") != compat["historical_vae"]
            or historical.get("vae_state_sha256") != compat["historical_vae"]["vae_state_sha256"]
            or historical.get("historical_module_sha256") != compat["historical_vae"].get("module_sha256")
            or historical.get("historical_encoder_src_sha256") != compat["historical_vae"].get("encoder_src_sha256")):
        raise ValueError("恢复 receipt 未绑定原始历史 VAE 身份")
    g0 = records["G0"]
    counts = g0.get("counts", {})
    windows = g0.get("windows", [])
    if (g0.get("schema") != "stage2-validation-inputs-v1" or g0.get("gate") != "G0"
            or g0.get("status") != "PASS" or g0.get("seed") != 0
            or counts.get("windows") != 64 or len(windows) != 64
            or counts.get("tasks") != 16 or counts.get("sha_mismatches") != 0
            or counts.get("demo_tail", 0) < 16):
        raise ValueError("阶段二 G0 证据不完整")
    if (g0.get("latent_metadata_sha256") != receipt.get("latents", {}).get("total_metadata_sha256")
            or g0.get("manifest_sha256") != receipt.get("parent", {}).get("manifest_canonical_sha256")):
        raise ValueError("阶段二 G0 与恢复 receipt 的输入身份不符")
    tasks = {}
    for window in windows:
        tasks[window["task"]] = tasks.get(window["task"], 0) + 1
        if (len(window.get("frame_ids", [])) != 33
                or len(window.get("source_frame_indices", [])) != 33
                or not window.get("sources") or type(window.get("source_grid")) is not int
                or window["source_grid"] % 2):
            raise ValueError("阶段二 G0 窗口来源不完整")
    if len(tasks) != 16 or set(tasks.values()) != {4}:
        raise ValueError("阶段二 G0 必须每任务四窗")
    for prov_key, record_key in (("manifest_sha256", "manifest_sha256"),
                                 ("latents_metadata_sha256", "latent_metadata_sha256")):
        if prov_key in provenance and provenance[prov_key] != g0.get(record_key):
            raise ValueError(f"阶段二 G0 输入身份不符: {prov_key}")
    from mme_vla_suite.datastore.motion_store import validate_stage2_g2_evidence
    for gate in ("G2", "G3"):
        record = records[gate]
        if gate == "G2":
            validate_stage2_g2_evidence(
                record, checkpoint_sha256=provenance["encoder"]["checkpoint_sha256"],
                samples_sha256=evidence["G0"]["sha256"], source_pin=provenance["source_pin"],
                numerics=provenance["numerics"], online_vae=compat["online_vae"], demo_tail=counts["demo_tail"])
            continue
        if (record.get("schema") != "stage2-validation-gate-v1" or record.get("gate") != gate
                or record.get("status") != "PASS"
                or record.get("checkpoint_sha256") != provenance["encoder"]["checkpoint_sha256"]
                or record.get("samples_sha256") != evidence["G0"]["sha256"]):
            raise ValueError(f"阶段二 {gate} 身份不符或未通过")
        metrics = record.get("metrics", {})
        for group in ("source_pin", "numerics"):
            if record.get(group) != provenance[group]:
                raise ValueError(f"阶段二 {gate} 验证环境的 {group} 与来源不符")
        if any(metrics.get(k) != v for k, v in dict(
                windows=32, seed=0, max_abs=0, geometry_all_true=1, leaves=65,
                state_key="model", exact=1, live_negative_rejected=1).items()):
            raise ValueError("阶段二 G3 证据未覆盖完整 EMA 核验")


def verify_timepool_snapshot(run_root: str | pathlib.Path) -> dict:
    """核 run 根 ``motion_snapshot.ready.json`` 的每个文件 SHA 与身份绑定，返回 ``motion_provenance.json`` 内容。

    只读主会话 ``write_timepool_motion_snapshot`` 发布的快照；文件集合必须与合同恰好相等，因此 stage2 快照
    （带 ``encoder_dependencies.json``）不可能通过。不读取任何 stage2 encoder 路径或 checkpoint。
    """
    from mme_vla_suite.datastore import motion_store as ms
    root = pathlib.Path(run_root)
    ready_path = root / "motion_snapshot.ready.json"
    if not ready_path.is_file():
        raise ValueError(f"timepool 恢复缺 motion_snapshot.ready.json: {root}")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    files = ready.get("files") if isinstance(ready, dict) else None
    if not isinstance(ready, dict) or ready.get("schema") != TIMEPOOL_SNAPSHOT_SCHEMA or not isinstance(files, dict):
        raise ValueError("timepool 快照 ready schema 不符")
    evidence = {role: f"motion_evidence/{role}.json" for role in ms.TIMEPOOL_VALIDATION_ROLES}
    expected = {"history_config.txt", "history_config.resolved.yaml", "history_config.resolved.sha256",
                "motion_provenance.json", TIMEPOOL_WINDOW_MAP_COPY, TIMEPOOL_ONLINE_RECORD, TIMEPOOL_ONLINE_PAYLOAD,
                *evidence.values()}
    if set(files) != expected:
        raise ValueError(f"timepool 快照文件清单不符: 缺 {sorted(expected - set(files))}，多 {sorted(set(files) - expected)}")
    for relative, sha in files.items():
        path = root / relative
        if not path.is_file() or _sha256(path) != sha:
            raise ValueError(f"timepool 快照文件 SHA 不符: {relative}")
    prov = json.loads((root / "motion_provenance.json").read_text(encoding="utf-8"))
    if not isinstance(prov, dict) or set(prov) != TIMEPOOL_SNAPSHOT_PROV_KEYS:
        raise ValueError("timepool motion_provenance.json 键集合不符")
    if prov["motion_enabled"] is not True or prov["vae"] is not None or prov["encoder"] is not None:
        raise ValueError("timepool motion_provenance.json 必须 motion_enabled=true 且 vae/encoder 为 null")
    if (prov["resolved_sha256"] != files["history_config.resolved.yaml"]
            or (root / "history_config.resolved.sha256").read_text().split()[:1] != [prov["resolved_sha256"]]):
        raise ValueError("timepool 来源记录与解析配置不同步")
    timepool = prov["timepool"]
    ms.validate_timepool_provenance(timepool, status="verified")
    for key, expected_sha in (("motion_manifest_sha256", timepool["manifests"]["episode_manifest_sha256"]),
                              ("motion_table_sha256", timepool["output"]["motion_table_sha256"]),
                              ("motion_index_sha256", timepool["output"]["motion_index_sha256"])):
        if prov[key] != expected_sha:
            raise ValueError(f"timepool 快照 {key} 与来源记录不同")
    if files[TIMEPOOL_WINDOW_MAP_COPY] != timepool["window_map"]["sha256"]:
        raise ValueError("timepool 逐窗映射副本与来源记录不同")
    for role, relative in evidence.items():
        if files[relative] != timepool["validation"][role]["sha256"]:
            raise ValueError(f"timepool 证据副本与来源记录不同: {role}")
    online = json.loads((root / TIMEPOOL_ONLINE_RECORD).read_text(encoding="utf-8"))
    payload = online.get("payload") if isinstance(online, dict) else None
    if (not isinstance(online, dict) or online.get("schema") != TIMEPOOL_ONLINE_SCHEMA or not isinstance(payload, dict)
            or payload.get("path") != TIMEPOOL_ONLINE_PAYLOAD or payload.get("sha256") != files[TIMEPOOL_ONLINE_PAYLOAD]):
        raise ValueError("timepool 在线恢复输入记录与 NPZ 不同步")
    return prov


def resolve_stage2_dependencies(run_root: pathlib.Path, provenance: dict) -> pathlib.Path | None:
    """交付包只凭相对路径定位 encoder 两实体，拒绝越界与身份漂移。"""
    dependency_path = run_root / "encoder_dependencies.json"
    if not dependency_path.exists():
        return None
    record = json.loads(dependency_path.read_text(encoding="utf-8"))
    if record.get("schema") != "stage2-encoder-dependencies-v1":
        raise ValueError("encoder_dependencies schema 不符")
    for group in ("encoder", "source_pin", "vae"):
        expected = {k: v for k, v in provenance[group].items() if k != "run_dir"}
        actual = {k: v for k, v in record.get(group, {}).items() if k != "run_dir"}
        if actual != expected:
            raise ValueError(f"encoder_dependencies 的 {group} 不同源")
    paths = {}
    # 本机 run 只写依赖清单，实体在标准注册目录；交付包则带完整实体子目录。
    delivered = (run_root / "encoder_dependencies").exists()
    for item in record.get("files", []):
        role = item.get("role")
        if role not in ("encoder_checkpoint", "encoder_config") or role in paths:
            raise ValueError("encoder_dependencies 的文件角色非法或重复")
        relative = pathlib.Path(item["relative_path"])
        enc = provenance["encoder"]
        expected_name = enc["checkpoint_name"] if role == "encoder_checkpoint" else "config.yaml"
        if relative != pathlib.Path("encoder_dependencies") / enc["run_name"] / expected_name:
            raise ValueError("encoder_dependencies 必须使用交付相对路径")
        expected_sha = provenance["encoder"]["checkpoint_sha256" if role == "encoder_checkpoint" else "config_sha256"]
        if item.get("sha256") != expected_sha or type(item.get("bytes")) is not int or item["bytes"] <= 0:
            raise ValueError(f"encoder_dependencies 清单不符: {role}")
        path = _evidence_path(str(relative), run_root) if delivered else run_root / relative
        if delivered and (_sha256(path) != expected_sha or item["bytes"] != path.stat().st_size):
            raise ValueError(f"encoder_dependencies 实体不符: {role}")
        paths[role] = path
    if set(paths) != {"encoder_checkpoint", "encoder_config"}:
        raise ValueError("encoder_dependencies 缺 encoder 两实体")
    checkpoint, config = paths["encoder_checkpoint"], paths["encoder_config"]
    if (checkpoint.name != provenance["encoder"]["checkpoint_name"]
            or config.name != "config.yaml" or config.parent != checkpoint.parent):
        raise ValueError("encoder_dependencies 两实体名称或同目录关系不符")
    return checkpoint.parent if delivered else None


class MotionEncoderClient:
    def __init__(self, *, online_gpu: int | str, encoder_run_dir: str | None = None, checkpoint: str = "checkpoint_epoch_72.pt",
                 expected_ckpt_sha256: str | None = None, store_provenance: dict | None = None, stub: bool = False,
                 v1_store: str | pathlib.Path | None = None, handshake_timeout: float = P.HANDSHAKE_TIMEOUT_S,
                 source_run: str | None = None, encoder_kind: str = "v10",
                 provenance_root: str | pathlib.Path | None = None,
                 stage2_python: str | None = None, eval_target_contract: str | pathlib.Path | None = None):
        self.encoder_kind = encoder_kind
        self._eval_target_contract = eval_target_contract
        # stage2_python 名字不改以免破坏旧调用方，在固定池化下同样表示 sidecar 解释器。
        if encoder_kind not in SESSION_KINDS and stage2_python or encoder_kind != STAGE2_KIND and eval_target_contract:
            raise ValueError("阶段二评估选项不能用于其他encoder")
        if encoder_kind == TIMEPOOL_KIND:
            if encoder_run_dir is not None or source_run is not None or expected_ckpt_sha256 is not None:
                raise ValueError("固定池化没有 encoder：不得传 encoder_run_dir、source_run 或 expected_ckpt_sha256")
            self._init_timepool(online_gpu=online_gpu, store_provenance=store_provenance, stub=stub, v1_store=v1_store,
                                handshake_timeout=handshake_timeout, stage2_python=stage2_python)
            return
        if encoder_kind == STAGE2_KIND:
            self._init_stage2(online_gpu=online_gpu, encoder_run_dir=encoder_run_dir,
                             expected_ckpt_sha256=expected_ckpt_sha256, store_provenance=store_provenance,
                             stub=stub, v1_store=v1_store, handshake_timeout=handshake_timeout,
                             source_run=source_run, provenance_root=provenance_root, stage2_python=stage2_python)
            return
        if encoder_kind != "v10":
            raise ValueError(f"未知 motion.encoder_kind: {encoder_kind}")
        self._lock = threading.Lock()
        self.stub = bool(stub)
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
        else:
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
        if not self.stub:
            if store_provenance is None:
                self.close()
                raise RuntimeError("真编码器模式必须提供离线库 store_provenance 以核对同源")
            self.check_provenance(store_provenance)
        self.n_calls = 0
        self.total_s = 0.0

    def _init_stage2(self, *, online_gpu, encoder_run_dir, expected_ckpt_sha256, store_provenance,
                     stub, v1_store, handshake_timeout, source_run, provenance_root, stage2_python=None) -> None:
        self._lock = threading.Lock()
        self.stub = bool(stub)
        self._online_gpu = str(online_gpu)
        self._provenance_root = pathlib.Path(provenance_root) if provenance_root is not None else None
        if not self.stub:
            if store_provenance is None:
                raise ValueError("真实阶段二编码必须提供完整 store_provenance")
            from mme_vla_suite.datastore.motion_store import validate_stage2_provenance
            validate_stage2_provenance(store_provenance)
            validate_stage2_evidence(store_provenance, self._provenance_root)
            if expected_ckpt_sha256 != store_provenance["encoder"]["checkpoint_sha256"]:
                raise ValueError("阶段二 expected_ckpt_sha256 与来源不一致")
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)/(checkpoint_epoch_\d+\.pt)#model", source_run or "")
        if match is None:
            raise ValueError("阶段二 source_run 必须为 <run>/checkpoint_epoch_<n>.pt#model")
        run, checkpoint = match.groups()
        if not self.stub and (run != store_provenance["encoder"]["run_name"]
                              or checkpoint != store_provenance["encoder"]["checkpoint_name"]):
            raise ValueError("阶段二 source_run 与完整来源不同")
        v1 = pathlib.Path(v1_store or os.environ.get("MMEVLA_V1_STORE") or (_REPO_ROOT / "v1-store")).resolve()
        registered = v1 / "external/motionjepa" / run
        delivered = (resolve_stage2_dependencies(self._provenance_root, store_provenance)
                     if not self.stub and self._provenance_root is not None else None)
        expected_dir = delivered or registered
        if encoder_run_dir is not None and pathlib.Path(encoder_run_dir).resolve() != expected_dir.resolve():
            raise ValueError("阶段二 encoder_run_dir 与注册或交付身份不同")
        project = _REPO_ROOT / "scripts/dataset/wan_stage2"
        local_pin = json.loads((project / "SOURCE_PIN.json").read_text())
        for key, path in (("dependency_lock_sha256", project / "uv.lock"),
                          ("vae_adapter_sha256", project.parent / "wan/wan_motion_infer.py")):
            if _sha256(path) != local_pin[key]:
                raise ValueError(f"阶段二本地文件不符合 SOURCE_PIN: {key}")
        if not self.stub and store_provenance["source_pin"] != local_pin:
            raise ValueError("阶段二 store 与当前 SOURCE_PIN 不同源")
        script_args = ["--stub"] if self.stub else [
            "--ckpt", str(expected_dir / checkpoint), "--expected-sha256", expected_ckpt_sha256,
            "--expected-vae-sha256", store_provenance["vae"]["vae_state_sha256"]]
        self._spawn_session_worker(project=project, script=project / "stage2_sidecar.py", script_args=script_args,
                                   v1=v1, online_gpu=online_gpu, stage2_python=stage2_python)
        self._complete_session_handshake(label="阶段二", expected={"encoder_kind": STAGE2_KIND},
                                         store_provenance=store_provenance, handshake_timeout=handshake_timeout)

    def _init_timepool(self, *, online_gpu, store_provenance, stub, v1_store, handshake_timeout, stage2_python) -> None:
        """固定池化：不依赖任何 encoder 路径或 checkpoint，进程管理与阶段二共用。"""
        self._lock = threading.Lock()
        self.stub = bool(stub)
        self._online_gpu = str(online_gpu)
        if not self.stub:
            if store_provenance is None:
                raise ValueError("真实固定池化必须提供完整 store_provenance")
            from mme_vla_suite.datastore.motion_store import validate_timepool_provenance
            validate_timepool_provenance(store_provenance, status="verified")
        v1 = pathlib.Path(v1_store or os.environ.get("MMEVLA_V1_STORE") or (_REPO_ROOT / "v1-store")).resolve()
        script_args = ["--stub"] if self.stub else [
            "--expected-vae-sha256", store_provenance["wan"]["vae_state_sha256"]]
        self._spawn_session_worker(project=_REPO_ROOT / "scripts/dataset/wan_stage2",
                                   script=_REPO_ROOT / "scripts/dataset/wan_timepool/timepool_sidecar.py",
                                   script_args=script_args, v1=v1, online_gpu=online_gpu, stage2_python=stage2_python)
        self._complete_session_handshake(
            label="固定池化", store_provenance=store_provenance, handshake_timeout=handshake_timeout,
            expected={"encoder_kind": TIMEPOOL_KIND,
                      "pool": {"version": _timepool.POOL_VERSION, "source_sha256": _timepool.pool_source_sha256()}})

    def _spawn_session_worker(self, *, project, script, script_args, v1, online_gpu, stage2_python) -> None:
        """在独立会话里启动 sidecar，argv 与 socket 身份留给后续进程归属核验。"""
        child_env = dict(os.environ)
        child_env.update(UV_LINK_MODE="copy", UV_PROJECT_ENVIRONMENT=str(v1 / "venvs/wan-stage2"),
                         UV_CACHE_DIR=str(v1 / "cache/uv"),
                         HF_HOME=str(v1 / "cache/hf"), HF_HUB_OFFLINE="1",
                         CUDA_VISIBLE_DEVICES=str(online_gpu), PYTHONUNBUFFERED="1")
        # 子环境必须从锁定安装包 import，不能继承主环境的 Python 注入路径。
        child_env.pop("PYTHONPATH", None)
        child_env.pop("PYTHONHOME", None)
        argv = ["uv", "run", "--project", str(project), "--no-sync"]
        if stage2_python is not None:
            interpreter = pathlib.Path(stage2_python).absolute()
            if not interpreter.is_file() or interpreter.parent.name != "bin":
                raise ValueError("stage2_python必须为现存虚拟环境bin内解释器")
            child_env["UV_PROJECT_ENVIRONMENT"] = str(interpreter.parent.parent)
            # 直接保留uv管理venv入口，使Popen PID就是握手进程，且不丢失venv前缀。
            argv = [str(interpreter)]
        else:
            argv += ["python"]
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock = parent_sock
        self._stage2_socket_identity = (child_sock.fileno(), os.fstat(child_sock.fileno()).st_ino)
        self._stage2_script_argv_offset = len(argv)
        argv += [str(script), "--fd", str(child_sock.fileno()), *script_args]
        self.argv = argv
        try:
            self._proc = subprocess.Popen(argv, env=child_env, pass_fds=[child_sock.fileno()], cwd=str(_REPO_ROOT),
                                          stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr,
                                          start_new_session=True)
            self._stage2_owns_session = True
            self._capture_stage2_session()
        except Exception:
            parent_sock.close()
            self._sock = None
            self.close()
            raise
        finally:
            child_sock.close()

    def _complete_session_handshake(self, *, label, expected, store_provenance, handshake_timeout) -> None:
        """核握手固定字段与 worker 归属；真实模式再核离线来源，任一失败都清理进程。"""
        try:
            self.provenance = P.recv_handshake(self._sock, time.monotonic() + handshake_timeout)
            if "error" in self.provenance:
                raise RuntimeError(f"{label} sidecar 启动失败: {self.provenance['error']}")
            fixed = {"dtype": "float32", "shape": [1024], "response_bytes": 4096, "stub": self.stub,
                     "protocol_sha256": P.protocol_sha256(), **expected}
            for key, value in fixed.items():
                if type(self.provenance.get(key)) is not type(value) or self.provenance[key] != value:
                    raise RuntimeError(f"{label}握手 {key} 不匹配")
            if self.encoder_kind == TIMEPOOL_KIND:
                _reject_encoder_fields(self.provenance)
            self._worker_identity = self._stage2_worker_identity()
            if not self.stub:
                self.check_provenance(store_provenance)
        except Exception:
            self.close()
            raise
        self.n_calls = 0
        self.total_s = 0.0

    def _capture_stage2_session(self) -> None:
        """首次 poll/reap 前记录自有会话与启动时钟；已退出启动器的 stat 仍可核验。"""
        pid = self._proc.pid
        fields = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if (int(fields[2]), int(fields[3])) != (pid, pid):
            raise ValueError("阶段二启动器未建立本次独立会话与进程组")
        self._stage2_session_identity = (pid, int(fields[19]))

    def _stage2_script_matches(self, pid: int) -> bool:
        command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        executable = pathlib.Path(f"/proc/{pid}/exe").resolve(strict=True).name
        return (re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable) is not None
                and [os.fsdecode(arg) for arg in command[1:]] == self.argv[self._stage2_script_argv_offset:])

    def _stage2_worker_identity(self) -> tuple[int, int]:
        """核上报 PID 属于启动器的存活后代，且运行本次指定脚本与完整参数。"""
        pid = self.provenance.get("pid")
        if self._proc.poll() is not None:
            raise ValueError("阶段二 worker 进程身份的启动器已退出")
        snapshot = _live_process(pid)
        parent, started = snapshot
        identity = (pid, started)
        if getattr(self, "_worker_identity", identity) != identity:
            raise ValueError("阶段二 worker 进程身份已被替换")
        try:
            matches = self._stage2_script_matches(pid)
        except OSError as exc:
            raise ValueError("阶段二 worker 脚本身份不可读") from exc
        if not matches:
            raise ValueError("阶段二 worker 脚本身份与启动参数不符")
        ancestors = {pid}
        current = pid
        while current != self._proc.pid:
            current = parent
            if current in ancestors or current <= 1 or len(ancestors) >= 128:
                raise ValueError("阶段二 worker 进程身份不属于本次启动器")
            ancestors.add(current)
            parent, _ = _live_process(current)
        # 前后复核避免读取归属链期间退出或被替换。
        if _live_process(pid) != snapshot:
            raise ValueError("阶段二 worker 进程身份在校验中变化")
        if self._proc.poll() is not None:
            raise ValueError("阶段二 worker 进程身份的启动器已退出")
        return identity

    @property
    def worker_pid(self) -> int:
        """返回经归属、脚本与启动时钟复核的实际阶段二 worker PID。"""
        if self.encoder_kind not in SESSION_KINDS:
            raise ValueError("worker_pid 仅用于阶段二客户端")
        return self._stage2_worker_identity()[0]

    def _check_stage2_provenance(self, store_prov: dict) -> None:
        from mme_vla_suite.datastore.motion_store import validate_stage2_provenance
        validate_stage2_provenance(store_prov)
        validate_stage2_evidence(store_prov, self._provenance_root)
        target = None
        if getattr(self, "_eval_target_contract", None) is not None:
            from scripts.evaluation.runtime_contract import validate_eval_runtime_contract
            target = validate_eval_runtime_contract(self._eval_target_contract, training_provenance=store_prov,
                actual_runtime={"numerics": self.provenance.get("numerics"),
                                "online_vae": self.provenance.get("online_vae")})
        for group in ("encoder", "encoder_state_sha256", "source_pin", "numerics"):
            actual, expected = self.provenance.get(group), store_prov[group]
            if group == "encoder" and isinstance(actual, dict):
                actual = {key: value for key, value in actual.items() if key != "run_dir"}
                expected = {key: value for key, value in expected.items() if key != "run_dir"}
            if target is not None and group == "numerics":
                from scripts.evaluation.runtime_contract import compare_runtime, software_runtime
                compare_runtime(software_runtime(expected), software_runtime(actual))
            elif actual != expected:
                raise ValueError(f"阶段二真实握手的 {group} 与离线来源不同")
        if target is not None:
            from scripts.evaluation.runtime_contract import compare_runtime, software_runtime
            compare_runtime(software_runtime(store_prov["stage2_compat"]["online_vae"]), software_runtime(self.provenance.get("online_vae")))
        elif self.provenance.get("online_vae") != store_prov["stage2_compat"]["online_vae"]:
            raise ValueError("阶段二真实 VAE 与兼容合同的 online_vae 不同")
        self._check_session_worker()

    def _check_session_worker(self) -> None:
        worker_pid = self.worker_pid
        if (worker_pid != self.provenance.get("pid")
                or not self.provenance.get("gpu_uuid") or not self.provenance.get("cuda_visible_devices")
                or self.provenance["cuda_visible_devices"] != self._online_gpu):
            raise ValueError("阶段二真实握手缺实际进程或 GPU 身份")

    def _check_timepool_provenance(self, store_prov: dict) -> None:
        """握手与离线来源逐项比对；stage2 来源因字段集合不同，过不了 validate_timepool_provenance。"""
        from mme_vla_suite.datastore.motion_store import validate_timepool_provenance
        validate_timepool_provenance(store_prov, status="verified")
        identity = self.provenance
        _reject_encoder_fields(identity)
        pool = {"version": store_prov["pool"]["version"], "source_sha256": store_prov["pool"]["source_sha256"]}
        if identity.get("pool") != pool:
            raise ValueError("固定池化握手的 pool 身份与离线来源不同")
        wan = identity.get("wan")
        if not isinstance(wan, dict) or set(wan) != set(TIMEPOOL_WAN_KEYS):
            raise ValueError("固定池化握手的 wan 字段不完整")
        for key in TIMEPOOL_WAN_KEYS:
            if wan[key] != store_prov["wan"][key]:
                raise ValueError(f"固定池化握手的 wan.{key} 与离线来源不同")
        numerics = identity.get("numerics")
        flags = numerics.get("flags") if isinstance(numerics, dict) else None
        if (not isinstance(flags, dict) or numerics.get("dtype") != "float32" or numerics.get("autocast") is not False
                or numerics.get("batch") != 1 or flags.get("matmul_allow_tf32") is not False
                or flags.get("cudnn_allow_tf32") is not False):
            raise ValueError("固定池化握手的数值开关不是 fp32、B=1、TF32=false")
        if not isinstance(identity.get("environment"), dict) or not identity["environment"]:
            raise ValueError("固定池化握手缺环境指纹")
        self._check_session_worker()

    def check_provenance(self, store_prov: dict) -> None:
        if self.encoder_kind in SESSION_KINDS:
            try:
                if self.encoder_kind == STAGE2_KIND:
                    self._check_stage2_provenance(store_prov)
                else:
                    self._check_timepool_provenance(store_prov)
            except Exception:
                self.close()
                raise
            return
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
            if self.encoder_kind in SESSION_KINDS:
                try:
                    P.send_all(self._sock, P.encode_request(start_frame, payload))
                    raw = P.recv_response(self._sock, deadline, response_bytes=4096)
                except (P.ProtocolError, OSError):
                    self.close()
                    raise
            else:
                P.send_all(self._sock, P.encode_request(start_frame, payload))
                raw = P.recv_response(self._sock, deadline)
            self.total_s += time.perf_counter() - t0
            self.n_calls += 1
        return np.frombuffer(raw, dtype=np.float32).copy()

    def close(self) -> None:
        if getattr(self, "encoder_kind", None) in SESSION_KINDS:
            self._close_stage2()
            return
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

    def _close_stage2(self) -> None:
        """先按协议结束 worker；超时只清理本客户端独立会话中的进程组。"""
        sock = getattr(self, "_sock", None)
        if sock is not None:
            try:
                P.send_all(sock, P.encode_shutdown())
            except Exception:
                pass
            finally:
                sock.close()
                self._sock = None
        process = getattr(self, "_proc", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    self._signal_stage2_group(sig)
                    try:
                        process.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        self._signal_stage2_group(signal.SIGKILL)

    def _signal_stage2_group(self, sig: int) -> None:
        """只向本次创建且现场复核归属的会话组发信号，不使用未验证的上报 PID。"""
        if not getattr(self, "_stage2_owns_session", False):
            return
        process = self._proc
        identity = getattr(self, "_worker_identity", None)
        candidates = [(process.pid, None)] if process.poll() is None else []
        if identity is not None:
            candidates.append(identity)
        elif process.poll() is not None:
            candidates.extend(self._unhandshaken_stage2_workers())
        for pid, started in candidates:
            try:
                if started is not None and _live_process(pid)[1] != started:
                    continue
                if os.getpgid(pid) == process.pid and os.getsid(pid) == process.pid:
                    os.killpg(process.pid, sig)
                    return
            except (ValueError, ProcessLookupError):
                continue

    def _unhandshaken_stage2_workers(self) -> list[tuple[int, int]]:
        """启动器先退出时，凭自有会话、时钟、脚本与继承 socket 重新找待清理进程。"""
        session = getattr(self, "_stage2_session_identity", None)
        endpoint = getattr(self, "_stage2_socket_identity", None)
        if session is None or endpoint is None:
            return []
        group, launched = session
        descriptor, inode = endpoint
        workers = []
        for directory in pathlib.Path("/proc").iterdir():
            if not directory.name.isdecimal():
                continue
            pid = int(directory.name)
            try:
                snapshot = _live_process(pid)
                if (snapshot[1] < launched or os.getpgid(pid) != group or os.getsid(pid) != group
                        or not self._stage2_script_matches(pid)
                        or os.readlink(directory / "fd" / str(descriptor)) != f"socket:[{inode}]"
                        or _live_process(pid) != snapshot):
                    continue
                workers.append((pid, snapshot[1]))
            except (ValueError, OSError):
                continue
        return workers

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
