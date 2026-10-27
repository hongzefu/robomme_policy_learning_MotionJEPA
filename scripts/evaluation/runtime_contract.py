"""官方评估的身份、数组证据与有序 RPC 合同；本模块不导入模型后端。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def identity_sha256(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("必须提供完整 SHA256")
    return value


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON 重复键: {key}")
            result[key] = value
        return result

    result = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(result, dict):
        raise ValueError("JSON 根必须为对象")
    return result


def payload_sha256(payload):
    """顶层剔除 eval_rpc；数组保留 dtype/shape/C顺序字节，容器保留类型与次序。"""
    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("RPC 数组不允许 object dtype")
            array = np.ascontiguousarray(value)
            return ["ndarray", array.dtype.str, list(array.shape), hashlib.sha256(array.tobytes()).hexdigest()]
        if isinstance(value, np.generic):
            return encode(np.asarray(value))
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ValueError("RPC 字典键必须为字符串")
            return ["dict", [[key, encode(value[key])] for key in sorted(value)]]
        if isinstance(value, (tuple, list)):
            # msgpack 将 tuple 解码为 list，两端按同一种序列编码。
            return ["sequence", [encode(item) for item in value]]
        if isinstance(value, bytes):
            return ["bytes", value.hex()]
        if value is None or type(value) in (str, bool, int, float):
            return [type(value).__name__, value]
        raise ValueError(f"不支持的 RPC 数据类型: {type(value).__name__}")

    return identity_sha256(encode({key: value for key, value in payload.items() if key != "eval_rpc"}))


IDENTITY_KEYS = {
    "motion": {"weights", "source_pin_sha256", "software_numerics", "cross_hardware_numeric"},
    "model": {"checkpoint_files", "resolved_sha256", "normstats_sha256", "motion_identity_sha256"},
    "validation_model": {"checkpoint_files", "resolved_sha256", "normstats_sha256", "motion_identity_sha256"},
    "run": {"model_identity_sha256", "runtime_contract_sha256", "benchmark_commit", "metadata_sha256", "model_seed", "eval_parameters"},
    "validation_run": {"validation_model_identity_sha256", "runtime_contract_sha256", "benchmark_commit", "metadata_sha256", "model_seed", "eval_parameters"},
    "schedule": {"training_run", "target_step", "eval_code_commit", "benchmark_commit", "job_spec", "motion_identity_sha256"},
}


def descriptor_identity(kind, descriptor):
    if kind not in IDENTITY_KEYS or IDENTITY_KEYS[kind] - descriptor.keys():
        raise ValueError(f"身份字段不完整: {kind}")
    if {"desired_jobs", "pool_revision", "worker_id", "worker_count"} & descriptor.keys():
        raise ValueError("可变池成员不进入评估身份")
    if kind == "schedule" and descriptor["target_step"] != 60000:
        raise ValueError("调度目标必须为60000")
    return identity_sha256({"kind": kind, "descriptor": descriptor})


def compare_runtime(expected, actual):
    """严格比较来源、软件与数值字段；默认训练通路仍比较全部字段。"""
    if type(expected) is not type(actual) or expected != actual:
        raise ValueError("严格来源、软件或数值配置不符")


def verified_ref(root, ref):
    path = Path(ref["path"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("证据引用必须为根内相对路径")
    path = (Path(root) / path).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("证据引用越界")
    if sha256_file(path) != require_sha256(ref["sha256"]):
        raise ValueError("证据文件 SHA256 不符")
    return path


def numeric_metrics(reference, actual):
    reference, actual = np.asarray(reference), np.asarray(actual)
    if (reference.size == 0 or reference.shape != actual.shape or reference.dtype != actual.dtype
            or reference.dtype.hasobject or not np.issubdtype(reference.dtype, np.number)
            or np.issubdtype(reference.dtype, np.complexfloating)
            or not np.isfinite(reference).all() or not np.isfinite(actual).all()):
        raise ValueError("参考/实际数组为空、shape/dtype不符或存在非有限值")
    left, right = reference.astype(np.float64), actual.astype(np.float64)
    difference = np.abs(left - right)
    maximum = int(np.argmax(difference))
    denominator = np.maximum(np.abs(left), 1e-12)
    norms = float(np.linalg.norm(left.ravel()) * np.linalg.norm(right.ravel()))
    cosine = float(np.dot(left.ravel(), right.ravel()) / norms) if norms else float(np.array_equal(left, right))
    return {"max_abs": float(difference.flat[maximum]), "max_abs_index": [int(i) for i in np.unravel_index(maximum, left.shape)],
            "max_rel": float(np.max(difference / denominator)), "relative_zero_floor": 1e-12,
            "cosine": cosine, "elements": int(left.size), "nonfinite": 0}


HARDWARE_FIELDS = frozenset({"gpu_name", "driver", "compute_cap", "sm_count", "gpu_uuid",
    "device_name", "device_arch", "gpu_arch", "arch", "architecture", "compute_capability"})
USER_DECISION = "不需要做到跨硬件的数值性一致。"


def software_runtime(value):
    """仅从数值运行时描述中剔除实际硬件属性；模型encoder架构仍由原来源严格核验。"""
    if isinstance(value, dict):
        return {key: software_runtime(item) for key, item in value.items() if key not in HARDWARE_FIELDS}
    if isinstance(value, list):
        return [software_runtime(item) for item in value]
    return value


def motion_descriptor(provenance):
    return {"weights": {"ema": provenance["encoder_state_sha256"]["sha256"],
                        "vae": provenance["vae"]["vae_state_sha256"]},
            "source_pin_sha256": identity_sha256(provenance["source_pin"]),
            "software_numerics": software_runtime({"numerics": provenance["numerics"],
                "online_vae": provenance["stage2_compat"]["online_vae"]}),
            "cross_hardware_numeric": "not_required"}


def make_eval_runtime_contract(training_provenance):
    """按本次用户裁决记录运行合同，不产生数值通过或validated证书。"""
    descriptor = motion_descriptor(training_provenance)
    return {"schema": "eval-runtime-contract-v1", "cross_hardware_numeric": "not_required",
            "user_decision": USER_DECISION,
            "training_provenance_sha256": identity_sha256(training_provenance),
            "motion_identity_sha256": descriptor_identity("motion", descriptor),
            "weights": descriptor["weights"], "source_pin_sha256": descriptor["source_pin_sha256"],
            "training_runtime": {"numerics": training_provenance["numerics"],
                "online_vae": training_provenance["stage2_compat"]["online_vae"]}}


def validate_eval_runtime_contract(path, *, training_provenance, actual_runtime=None):
    """训练来源与软件/精度仍严格；硬件差异不再要求跨卡数组或数值证明。"""
    contract = read_json(path)
    if contract != make_eval_runtime_contract(training_provenance):
        raise ValueError("评估运行合同与用户裁决、原子来源或原训练软件身份不符")
    if actual_runtime is not None:
        compare_runtime(software_runtime(contract["training_runtime"]), software_runtime(actual_runtime))
    return contract


class RpcSession:
    """在调用策略前检查身份和次序；异常后由worker重启resident，不做RPC重放。"""
    def __init__(self, metadata):
        self.metadata = metadata
        self.episode_nonce = None
        self.used_nonces = set()
        self.batch_seq = self.replan_seq = self.next_frame = 0
        self.last_inferred_frame = -1

    def validate(self, obs):
        if not isinstance(obs, dict) or sum(bool(obs.get(key)) for key in ("reset", "add_buffer", "release_attempt")) > 1:
            raise ValueError("每个RPC只能有一个操作")
        rpc = obs.get("eval_rpc")
        if not isinstance(rpc, dict):
            raise ValueError("缺eval_rpc")
        nonce = rpc.get("episode_nonce")
        if not isinstance(nonce, str) or not nonce:
            raise ValueError("缺episode_nonce")
        if obs.get("reset"):
            if nonce in self.used_nonces:
                raise ValueError("episode_nonce重复")
            for key in ("run_identity_sha256", "server_nonce"):
                if rpc.get(f"expected_{key}") != self.metadata[key]:
                    raise ValueError(f"reset身份不符: {key}")
            return "reset"
        if obs.get("release_attempt"):
            if self.episode_nonce is not None and nonce != self.episode_nonce:
                raise ValueError("release身份不符")
            return "release"
        if nonce != self.episode_nonce:
            raise ValueError("旧局或未reset")
        if obs.get("add_buffer"):
            count = len(obs.get("images", []))
            for key, value in (("batch_seq", self.batch_seq), ("first_global_frame", self.next_frame), ("frame_count", count)):
                if type(rpc.get(key)) is not int or rpc[key] != value:
                    raise ValueError(f"历史顺序不符: {key}")
            if count <= 0 or len(obs.get("state", [])) != count or rpc.get("payload_sha256") != payload_sha256(obs):
                raise ValueError("历史payload不符")
            boundary = obs.get("exec_start_idx", 0)
            if type(boundary) is not int or boundary < 0 or (self.batch_seq == 0 and count != boundary + 1):
                raise ValueError("首包帧数必须严格为D+1")
            if self.batch_seq > 0 and count != 16:
                raise ValueError("官方续包帧数必须严格为16")
            return "add_buffer"
        if (type(rpc.get("replan_seq")) is not int or rpc["replan_seq"] != self.replan_seq
                or type(rpc.get("last_global_frame")) is not int or rpc["last_global_frame"] != self.next_frame - 1
                or self.next_frame == 0 or self.last_inferred_frame == self.next_frame - 1):
            raise ValueError("infer序号或末帧不符")
        return "infer"

    def commit(self, operation, rpc):
        if operation == "reset":
            self.episode_nonce = rpc["episode_nonce"]
            self.used_nonces.add(self.episode_nonce)
            self.batch_seq = self.replan_seq = self.next_frame = 0
            self.last_inferred_frame = -1
        elif operation == "add_buffer":
            self.batch_seq += 1
            self.next_frame += rpc["frame_count"]
        elif operation == "infer":
            self.replan_seq += 1
            self.last_inferred_frame = self.next_frame - 1
