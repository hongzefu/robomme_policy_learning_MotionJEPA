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
    "motion": {"weights", "source_pin_sha256", "software_numerics", "fixture_manifest_sha256", "thresholds"},
    "model": {"checkpoint_files", "resolved_sha256", "normstats_sha256", "motion_identity_sha256"},
    "validation_model": {"checkpoint_files", "resolved_sha256", "normstats_sha256", "motion_identity_sha256"},
    "run": {"model_identity_sha256", "certificate_sha256", "benchmark_commit", "metadata_sha256", "model_seed", "eval_parameters"},
    "validation_run": {"validation_model_identity_sha256", "certificate_sha256", "benchmark_commit", "metadata_sha256", "model_seed", "eval_parameters"},
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


def compare_runtime(expected, actual, target=None):
    """只允许目标证书指定的 GPU 名称和驱动差异；其他叶逐项严格比较。"""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise ValueError("运行时字段集合不符")
        for key in expected:
            if target is not None and key in ("gpu_name", "driver"):
                wanted = target["target_gpu_family" if key == "gpu_name" else "target_driver"]
                if not isinstance(actual[key], str) or actual[key] != wanted:
                    raise ValueError(f"目标硬件不符: {key}")
            else:
                compare_runtime(expected[key], actual[key], target)
    elif type(expected) is not type(actual) or expected != actual:
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


def verify_measurement(fixture_path, measurement_path, *, fixture_sha256, motion_identity_sha256,
                       expected_runtime, target_gpu_family, target_driver):
    """从锁定参考及实际数组独立重算，不消费候选报告的 PASS/metrics。"""
    if sha256_file(fixture_path) != require_sha256(fixture_sha256):
        raise ValueError("fixture 清单身份不符")
    fixture, measured = read_json(fixture_path), read_json(measurement_path)
    if fixture.get("schema") != "eval-motion-reference-v1" or fixture.get("status") != "locked":
        raise ValueError("缺锁定参考包")
    require_sha256(fixture.get("producer_code_sha256"))
    if (not isinstance(fixture.get("producer_gpu_family"), str) or "A100" not in fixture["producer_gpu_family"]
            or fixture.get("reference_kind") not in ("a100-online", "packed-reference")):
        raise ValueError("参考必须明确标记原A100在线或认证packed来源")
    if measured.get("schema") != "eval-motion-measurement-v1" or measured.get("status") != "unvalidated_measurement":
        raise ValueError("需要隔离测量报告")
    if measured.get("measurement_code_sha256") != sha256_file(Path(__file__).with_name("measure_target_runtime.py")):
        raise ValueError("测量代码身份不符")
    if measured.get("fixture_manifest_sha256") != fixture_sha256 or measured.get("motion_identity_sha256") != motion_identity_sha256:
        raise ValueError("测量身份不符")
    target = {"target_gpu_family": target_gpu_family, "target_driver": target_driver}
    if not target_gpu_family or not target_driver:
        raise ValueError("缺实际GPU名称或driver")
    compare_runtime(expected_runtime, measured.get("runtime"), target)
    if measured.get("target_gpu_family") != target_gpu_family or measured.get("target_driver") != target_driver:
        raise ValueError("测量硬件不符")
    sidecar = measured.get("sidecar", {})
    if (type(sidecar.get("pid")) is not int or sidecar["pid"] <= 0 or not sidecar.get("gpu_uuid")
            or not sidecar.get("cuda_visible_devices") or sidecar.get("stub") is not False
            or {"numerics": sidecar.get("numerics"), "online_vae": sidecar.get("online_vae")} != measured["runtime"]):
        raise ValueError("测量缺实际sidecar进程/硬件/严格运行时身份")
    samples, actual_samples = fixture.get("samples"), measured.get("samples")
    if not isinstance(samples, list) or not samples or not isinstance(actual_samples, list):
        raise ValueError("样本集合为空")
    expected_ids = [sample["id"] for sample in samples]
    actual_ids = [sample["id"] for sample in actual_samples]
    if len(set(expected_ids)) != len(expected_ids) or len(set(actual_ids)) != len(actual_ids) or set(expected_ids) != set(actual_ids):
        raise ValueError("测量样本缺失或重复")
    if {sample.get("layer") for sample in samples} != {"encoder", "e2e"}:
        raise ValueError("必须覆盖encoder与原像素端到端两层")
    if not {"demo_tail", "real16", "exec"} <= {tag for sample in samples for tag in sample.get("coverage", [])}:
        raise ValueError("缺demo尾补、真实16节奏或exec覆盖")
    actual_by_id = {sample["id"]: sample for sample in actual_samples}
    metrics = []
    for sample in samples:
        actual_sample = actual_by_id[sample["id"]]
        ref_path = verified_ref(Path(fixture_path).parent, sample["reference"])
        out_path = verified_ref(Path(measurement_path).parent, actual_sample["actual"])
        ref = np.load(ref_path, allow_pickle=False)
        out = np.load(out_path, allow_pickle=False)
        if ref.shape != (1024,) or ref.dtype != np.float32:
            raise ValueError("motion参考必须为1024维float32")
        metric = numeric_metrics(ref, out)
        if metric["max_abs"] > 1e-5:
            raise ValueError(f"数值阈值失败: {sample['id']} max_abs={metric['max_abs']}")
        metrics.append({"id": sample["id"], "layer": sample["layer"], **metric})
    return {"samples": len(samples), "metrics": metrics, "max_abs": max(item["max_abs"] for item in metrics),
            "reference_kind": fixture["reference_kind"]}


def verifier_sha256():
    """将独立校验器源码直接绑定到证书，避免自填版本号。"""
    return sha256_file(__file__)


def motion_descriptor(provenance, fixture_sha256):
    return {"weights": {"ema": provenance["encoder_state_sha256"]["sha256"], "vae": provenance["vae"]["vae_state_sha256"]},
            "source_pin_sha256": identity_sha256(provenance["source_pin"]),
            "software_numerics": {"numerics": provenance["numerics"],
                                  "online_vae": provenance["stage2_compat"]["online_vae"]},
            "fixture_manifest_sha256": require_sha256(fixture_sha256),
            "thresholds": {"encoder_max_abs": 1e-5, "e2e_max_abs": 1e-5}}


def validate_target_contract(path, *, training_provenance, actual_runtime, training_provenance_path=None):
    path = Path(path)
    certificate = read_json(path)
    if certificate.get("schema") != "eval-target-contract-v1" or certificate.get("status") != "validated":
        raise ValueError("候选报告不能充当validated目标证书")
    if certificate.get("allowed_hardware_differences") != ["gpu_name", "driver"]:
        raise ValueError("目标合同只能放行gpu_name/driver")
    if certificate.get("target_gpu_family") != "NVIDIA A40":
        raise ValueError("本评估目标证书只接受NVIDIA A40")
    if certificate.get("training_provenance_sha256") != identity_sha256(training_provenance):
        raise ValueError("目标合同绑定的训练来源不符")
    if (training_provenance_path is not None
            and certificate.get("training_provenance_file_sha256") != sha256_file(training_provenance_path)):
        raise ValueError("目标合同绑定的训练来源原件SHA不符")
    if certificate.get("verifier_sha256") != verifier_sha256():
        raise ValueError("目标合同校验代码已变化")
    if certificate.get("thresholds") != {"encoder_max_abs": 1e-5, "e2e_max_abs": 1e-5}:
        raise ValueError("不允许放宽数值阈值")
    if certificate.get("weights") != {"ema": training_provenance["encoder_state_sha256"]["sha256"],
                                       "vae": training_provenance["vae"]["vae_state_sha256"]}:
        raise ValueError("目标合同权重不符")
    if certificate.get("source_pin_sha256") != identity_sha256(training_provenance["source_pin"]):
        raise ValueError("目标合同source_pin不符")
    runtime = certificate["training_runtime"]
    descriptor = motion_descriptor(training_provenance, certificate["fixture"]["sha256"])
    if certificate.get("motion_identity_sha256") != descriptor_identity("motion", descriptor):
        raise ValueError("motion分层身份不符")
    if not isinstance(certificate.get("eval_code_commit"), str) or re.fullmatch("[0-9a-f]{40}", certificate["eval_code_commit"]) is None:
        raise ValueError("证书缺锁定的验证代码commit")
    if runtime != {"numerics": training_provenance["numerics"],
                   "online_vae": training_provenance["stage2_compat"]["online_vae"]}:
        raise ValueError("目标合同软件来源不符")
    compare_runtime(runtime, actual_runtime, certificate)
    if certificate.get("target_runtime_sha256") != identity_sha256(actual_runtime):
        raise ValueError("目标实际软件配置与测量不符")
    fixture = verified_ref(path.parent, certificate["fixture"])
    measurement = verified_ref(path.parent, certificate["measurement"])
    measured = read_json(measurement)
    sidecar = measured.get("sidecar", {})
    for key in ("encoder", "encoder_state_sha256", "source_pin"):
        expected, actual = training_provenance[key], sidecar.get(key)
        if key == "encoder" and isinstance(actual, dict):
            expected, actual = ({k: v for k, v in item.items() if k != "run_dir"} for item in (expected, actual))
        if expected != actual:
            raise ValueError(f"目标测量严格来源不符: {key}")
    verdict = verify_measurement(fixture, measurement,
        fixture_sha256=certificate["fixture"]["sha256"], motion_identity_sha256=certificate["motion_identity_sha256"],
        expected_runtime=runtime, target_gpu_family=certificate["target_gpu_family"], target_driver=certificate["target_driver"])
    if certificate.get("recomputed_verdict") != verdict:
        raise ValueError("证书重算结果不符")
    return certificate


def validate_target_evidence(path, *, training_provenance, training_provenance_path=None):
    """起服务前核完整离线证据；实际节点仍须在sidecar握手时重新核现场运行时。"""
    path = Path(path)
    certificate = read_json(path)
    if certificate.get("schema") != "eval-target-contract-v1" or certificate.get("status") != "validated":
        raise ValueError("候选报告不是validated证书")
    measurement = verified_ref(path.parent, certificate["measurement"])
    return validate_target_contract(path, training_provenance=training_provenance,
                                    actual_runtime=read_json(measurement)["runtime"],
                                    training_provenance_path=training_provenance_path)


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
            if type(boundary) is not int or boundary < 0 or (self.batch_seq == 0 and boundary >= count):
                raise ValueError("首包必须包含完整demo与exec0")
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
