#!/usr/bin/env python3
"""G7 保存前参考对拍与 G12 交付恢复；两者成功均须真实在线推理。

CPU 合同测试只导入本模块，不导入 JAX、不加载大模型。真实恢复前显式设置
CUDA_VISIBLE_DEVICES=0、MMEVLA_V1_STORE、OPENPI_DATA_HOME，并传 --online-gpu 1。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_URI = "gs://big_vision/paligemma_tokenizer.model"
ROOT_FILES = {"history_config.txt", "history_config.resolved.yaml", "history_config.resolved.sha256",
              "motion_provenance.json", "encoder_dependencies.json", "online_restore_inputs.json",
              "online_restore_inputs.npz"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"非法 SHA256: {value!r}")
    return value


def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"JSON 重复键: {key}")
            value[key] = item
        return value

    def invalid(value):
        raise ValueError(f"JSON 非有限常数: {value}")

    result = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError(f"JSON 根必须为对象: {path}")
    return result


def inside(root, relative):
    """交付引用只接受根内实体文件，拒绝绝对路径、别名和全部符号链接。"""
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"非法相对路径: {relative!r}")
    parts = relative.split("/")
    if Path(relative).is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"交付路径越界或存在别名: {relative}")
    path = root
    for part in parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"交付路径不能使用符号链接: {path}")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"交付实体文件缺失或越界: {relative}")
    return path


def verify_file(root, relative, expected, size=None):
    path = inside(root, relative)
    if size is not None and (type(size) is not int or size < 0 or path.stat().st_size != size):
        raise ValueError(f"文件字节数不符: {relative}")
    if sha256_file(path) != _sha(expected):
        raise ValueError(f"文件 SHA256 不符: {relative}")
    return path


def verify_delivery_manifest(root, manifest_path, checkpoint):
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "stage2-delivery-manifest-v1":
        raise ValueError("发布清单 schema 不符")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("发布清单 files 必须非空")
    for name, item in files.items():
        if not isinstance(item, dict) or not {"sha256", "bytes"} <= item.keys():
            raise ValueError(f"发布文件记录不完整: {name}")
        if type(item["bytes"]) is not int:
            raise ValueError(f"发布文件字节数不合法: {name}")
        verify_file(root, name, item["sha256"], item["bytes"])
    required = ROOT_FILES | {"motion_snapshot.ready.json"}
    if required - files.keys():
        raise ValueError(f"发布清单缺根文件: {sorted(required - files.keys())}")
    # Orbax 元数据与 assets 同样必须被清单覆盖，不能只核数值分片。
    checkpoint_files = set()
    for path in Path(checkpoint).rglob("*"):
        if path.is_symlink():
            raise ValueError(f"checkpoint 内有符号链接: {path}")
        if path.is_file():
            checkpoint_files.add(path.relative_to(root).as_posix())
    if not checkpoint_files or checkpoint_files - files.keys():
        raise ValueError(f"发布清单未覆盖完整 checkpoint: {sorted(checkpoint_files - files.keys())[:8]}")
    for section in ("params", "assets"):
        if not any(name.startswith(f"{Path(checkpoint).name}/{section}/") for name in checkpoint_files):
            raise ValueError(f"checkpoint 缺 {section} 文件")
    return manifest


def verify_snapshot(root, delivery=None):
    from mme_vla_suite.datastore import motion_store

    ready = read_json(inside(root, "motion_snapshot.ready.json"))
    files = ready.get("files")
    if ready.get("schema") != "stage2-motion-snapshot-v1" or not isinstance(files, dict):
        raise ValueError("根快照 ready schema 不符")
    if ROOT_FILES - files.keys():
        raise ValueError(f"根快照缺必需文件: {sorted(ROOT_FILES - files.keys())}")
    for name, digest in files.items():
        verify_file(root, name, digest)
        if delivery is not None and delivery["files"].get(name, {}).get("sha256") != digest:
            raise ValueError(f"根快照文件不在发布清单中: {name}")
    provenance = read_json(inside(root, "motion_provenance.json"))
    motion_store.validate_stage2_provenance(provenance)
    from mme_vla_suite.policies.motion_client import validate_stage2_evidence
    validate_stage2_evidence(provenance, Path(root))
    yaml_sha = files["history_config.resolved.yaml"]
    if (inside(root, "history_config.resolved.sha256").read_text().strip() != yaml_sha
            or provenance.get("resolved_sha256") != yaml_sha or provenance.get("motion_enabled") is not True):
        raise ValueError("配置快照、来源记录与 SHA 文件不一致")
    evidence = provenance["stage2_compat"]["evidence"]
    for role, item in evidence.items():
        path = verify_file(root, item["path"], item["sha256"])
        if files.get(item["path"]) != item["sha256"]:
            raise ValueError(f"来源证据未被根快照绑定: {role}")
        record = read_json(path)
        if role == "G2":
            motion_store.validate_stage2_g2_evidence(
                record, checkpoint_sha256=provenance["encoder"]["checkpoint_sha256"],
                samples_sha256=evidence["G0"]["sha256"], source_pin=provenance["source_pin"],
                numerics=provenance["numerics"], online_vae=provenance["stage2_compat"]["online_vae"])
        if role in ("G0", "G3") and (record.get("gate") != role or record.get("status") != "PASS"):
            raise ValueError(f"来源证据未通过: {role}")
        if role in ("G2", "G3") and (
                record.get("checkpoint_sha256") != provenance["encoder"]["checkpoint_sha256"]
                or record.get("samples_sha256") != evidence["G0"]["sha256"]):
            raise ValueError(f"{role} 的 checkpoint 或 G0 身份不符")
        if role == "SOURCE_PIN" and record != provenance["source_pin"]:
            raise ValueError("SOURCE_PIN 证据与来源记录不一致")
    dependencies = read_json(inside(root, "encoder_dependencies.json"))
    if dependencies.get("schema") != "stage2-encoder-dependencies-v1":
        raise ValueError("encoder 依赖 schema 不符")
    for group in ("encoder", "source_pin", "vae"):
        if dependencies.get(group) != provenance[group]:
            raise ValueError(f"依赖清单未绑定完整 {group}")
    roles = {}
    for item in dependencies.get("files", []):
        role = item.get("role")
        if role in roles:
            raise ValueError(f"依赖角色重复: {role}")
        roles[role] = item
        if delivery is not None:
            # source_path 仅为上传诊断信息，交付恢复绝不使用它。
            verify_file(root, item["relative_path"], item["sha256"], item["bytes"])
            if delivery["files"].get(item["relative_path"]) != {"sha256": item["sha256"], "bytes": item["bytes"]}:
                raise ValueError(f"依赖实体不在发布清单中: {role}")
    for role, field in (("encoder_checkpoint", "checkpoint_sha256"), ("encoder_config", "config_sha256")):
        if roles.get(role, {}).get("sha256") != provenance["encoder"][field]:
            raise ValueError(f"缺少同源依赖实体: {role}")
    return provenance


def flatten(tree, prefix=()):
    if not isinstance(tree, dict):
        if not prefix:
            raise ValueError("参数树必须为字典")
        return {"/".join(map(str, prefix)): tree}
    result = {}
    for key, value in tree.items():
        if not isinstance(key, (str, int)) or "/" in str(key):
            raise ValueError(f"参数树路径歧义: {key!r}")
        leaves = flatten(value, (*prefix, key))
        if result.keys() & leaves.keys():
            raise ValueError("参数树路径冲突")
        result.update(leaves)
    return result


def parameter_summary(tree):
    leaves = {}
    for name, value in flatten(tree).items():
        array = np.asarray(value)
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError(f"参数含对象或非有限值: {name}")
        leaves[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    if not leaves:
        raise ValueError("参数树为空")

    def summary(items):
        return {"algorithm": "named-arrays-v1", "leaves": items, "sha256": hashlib.sha256(canonical(items)).hexdigest()}

    return summary(leaves), summary({name: item for name, item in leaves.items() if item["dtype"] == "float32"})


def verify_parameter_tree(expected, actual):
    expected, actual = flatten(expected), flatten(actual)
    result = {"missing": sorted(expected.keys() - actual.keys()), "extra": sorted(actual.keys() - expected.keys()),
              "shape_mismatch": [], "dtype_mismatch": []}
    for name in expected.keys() & actual.keys():
        if tuple(expected[name].shape) != tuple(actual[name].shape):
            result["shape_mismatch"].append(name)
        if str(expected[name].dtype) != str(actual[name].dtype):
            result["dtype_mismatch"].append(name)
    if any(result.values()):
        raise ValueError(f"完整参数树不符: {result}")
    return {key: 0 for key in result} | {"leaves": len(actual)}


def verify_summary(expected, actual, label):
    if (not isinstance(expected, dict) or expected.get("algorithm") != "named-arrays-v1"
            or expected.get("sha256") != hashlib.sha256(canonical(expected.get("leaves"))).hexdigest()):
        raise ValueError(f"{label} 保存前摘要不自洽")
    if expected != actual:
        wanted, got = expected["leaves"], actual["leaves"]
        changed = sorted(k for k in wanted.keys() | got.keys() if wanted.get(k) != got.get(k))
        raise ValueError(f"{label} 保存前原始字节不符: {changed[:8]}（共 {len(changed)} 叶）")


def load_reference_tree(root, record):
    path = verify_file(root, record["path"], record["sha256"])
    leaves, none_paths = record.get("leaves"), record.get("none_paths")
    if not isinstance(leaves, dict) or not leaves or not isinstance(none_paths, list):
        raise ValueError("参考树缺 leaves / none_paths")
    tree, seen = {}, set()

    def insert(keys, value):
        if not isinstance(keys, list) or not keys or any(not isinstance(k, (str, int)) for k in keys):
            raise ValueError("参考树路径不合法")
        keys = tuple(keys)
        if keys in seen or any(keys[:len(other)] == other or other[:len(keys)] == keys for other in seen):
            raise ValueError("参考树路径重复或父子冲突")
        seen.add(keys)
        current = tree
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value

    with np.load(path, allow_pickle=False) as arrays:
        if len(arrays.files) != len(set(arrays.files)) or set(arrays.files) != set(leaves):
            raise ValueError("参考 NPZ 键与清单不一致")
        for name, meta in leaves.items():
            array = arrays[name]
            if meta["dtype"] == "bfloat16":
                import ml_dtypes
                if array.dtype != np.uint16:
                    raise ValueError("BF16 参考必须以 uint16 位串存储")
                array = array.view(ml_dtypes.bfloat16)
            if list(array.shape) != meta["shape"] or str(array.dtype) != meta["dtype"]:
                raise ValueError(f"参考数组 shape/dtype 不符: {name}")
            if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() != _sha(meta["sha256"]):
                raise ValueError(f"参考数组 SHA 不符: {name}")
            if array.dtype.hasobject or not np.isfinite(array).all():
                raise ValueError(f"参考数组非有限: {name}")
            insert(meta["path"], array)
    for keys in none_paths:
        insert(keys, None)
    return tree


def compare_outputs(expected, actual):
    verify_parameter_tree(expected, actual)
    actual = flatten(actual)
    max_abs = 0.0
    for name, wanted in flatten(expected).items():
        got, wanted = np.asarray(actual[name]), np.asarray(wanted)
        if not np.isfinite(got).all() or not np.isfinite(wanted).all():
            raise ValueError(f"参考输出非有限: {name}")
        difference = float(np.max(np.abs(got.astype(np.float64) - wanted.astype(np.float64)), initial=0))
        max_abs = max(max_abs, difference)
        if difference != 0 or got.tobytes() != wanted.tobytes():
            raise ValueError(f"参考输出不逐位一致: {name} max_abs={difference}")
    return max_abs


def load_online_inputs(root):
    record = read_json(inside(root, "online_restore_inputs.json"))
    if record.get("schema") != "stage2-online-restore-inputs-v1":
        raise ValueError("在线恢复素材 schema 不符")
    if record.get("exec_batch_lengths") != [33, 32]:
        raise ValueError("在线恢复必须覆盖连续 33/32 帧的两批 exec")
    es = record.get("exec_start_idx")
    if type(es) is not int or not 17 <= es < 4031:
        raise ValueError("在线恢复 demo 必须至少17帧且总帧数不越4096")
    if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
        raise ValueError("在线恢复缺真实任务提示词")
    if not isinstance(record.get("sources"), list) or not record["sources"]:
        raise ValueError("在线恢复素材缺真实来源记录")
    for source in record["sources"]:
        if not isinstance(source.get("path"), str) or not source["path"]:
            raise ValueError("在线素材来源路径缺失")
        _sha(source.get("sha256"))
    payload = record.get("payload", {})
    if payload.get("path") != "online_restore_inputs.npz":
        raise ValueError("在线素材 NPZ 文件名不符")
    path = verify_file(root, payload["path"], payload["sha256"])
    shapes = {"images": ([es + 65, 256, 256, 3], "uint8"),
              "states": ([es + 65, 8], "float32"), "wrist_image": ([256, 256, 3], "uint8")}
    arrays = {}
    if set(record.get("arrays", {})) != set(shapes):
        raise ValueError("在线素材数组清单不完整")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(shapes) or len(data.files) != len(shapes):
            raise ValueError("在线素材 NPZ 键不符")
        for name, (shape, dtype) in shapes.items():
            value, meta = data[name], record["arrays"][name]
            if (list(value.shape) != shape or str(value.dtype) != dtype
                    or meta.get("shape") != shape or meta.get("dtype") != dtype):
                raise ValueError(f"在线素材 shape/dtype 不符: {name}")
            if hashlib.sha256(value.tobytes()).hexdigest() != _sha(meta.get("sha256")) or not np.isfinite(value).all():
                raise ValueError(f"在线素材 SHA 或有限值不符: {name}")
            arrays[name] = value
    if record.get("tokenizer", {}).get("uri") != TOKENIZER_URI:
        raise ValueError("在线恢复缺可取得的 tokenizer 来源")
    tokenizer_lock = read_json(REPO_ROOT / "scripts/assets/ASSETS_LOCK.json")["assets"]["paligemma_tokenizer"]
    if _sha(record["tokenizer"].get("sha256")) != tokenizer_lock["sha256"]:
        raise ValueError("在线恢复 tokenizer SHA 与仓库资产锁不符")
    return record, arrays


def runtime_dependencies(record, provenance):
    """读取显式缓存并绑定可取得的公开来源，不回退到主机默认隐藏目录。"""
    roots = {}
    for name in ("MMEVLA_V1_STORE", "OPENPI_DATA_HOME"):
        value = os.environ.get(name)
        if not value or not Path(value).is_absolute() or not Path(value).is_dir():
            raise ValueError(f"恢复前必须显式设置现存绝对目录 {name}")
        roots[name] = str(Path(value).resolve())
    hf_cache = Path(roots["MMEVLA_V1_STORE"]) / "cache/hf"
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_ASSETS_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(name)
        if value and not Path(value).expanduser().resolve().is_relative_to(hf_cache.resolve()):
            raise ValueError(f"{name} 绕过显式 VAE 缓存；恢复前须清除或改到 {hf_cache}")
    tokenizer = Path(roots["OPENPI_DATA_HOME"]) / "big_vision/paligemma_tokenizer.model"
    if not tokenizer.is_file() or sha256_file(tokenizer) != record["tokenizer"]["sha256"]:
        raise ValueError("显式缓存缺已固定 tokenizer；须先按记录 URI 获取并核 SHA")
    vae = provenance["stage2_compat"]["online_vae"]
    if not isinstance(vae.get("vae_id"), str) or re.fullmatch(r"[\w.-]+/[\w.-]+", vae["vae_id"]) is None:
        raise ValueError("VAE 必须给可取得的 Hugging Face 仓库标识，不能用原机器路径")
    return {"roots": roots, "tokenizer": {**record["tokenizer"], "actual_path": str(tokenizer.resolve())},
            "vae": {"uri": f"https://huggingface.co/{vae['vae_id']}", "subfolder": "vae",
                    "vae_state_sha256": vae["vae_state_sha256"],
                    "cache": str(hf_cache),
                    "verification": "sidecar 握手严格核对实际 VAE 状态摘要"}}


def gpu_inventory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"], text=True, timeout=15)
    result = {}
    for line in output.splitlines():
        index, uuid, name = [item.strip() for item in line.split(",", 2)]
        result[int(index)] = {"uuid": uuid, "name": name}
    return result


def verify_process_gpus(parent_pid, sidecar_pid, inventory, online_gpu, output=None):
    if output is None:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
    observed = {parent_pid: set(), sidecar_pid: set()}
    for line in output.splitlines():
        pid, uuid = [item.strip() for item in line.split(",", 1)]
        if int(pid) in observed:
            observed[int(pid)].add(uuid)
    expected = {parent_pid: {inventory[0]["uuid"]}, sidecar_pid: {inventory[online_gpu]["uuid"]}}
    if parent_pid == sidecar_pid or observed != expected:
        raise ValueError(f"实际 GPU 进程映射不符: expected={expected}, actual={observed}")
    return {"parent": {"pid": parent_pid, "physical_index": 0, **inventory[0]},
            "sidecar": {"pid": sidecar_pid, "physical_index": online_gpu, **inventory[online_gpu]}}


def run_online(policy, record, arrays, online_gpu, inventory):
    """必须使用客户端原 pack_buffer 和原 policy 状态机，不另写窗口实现。"""
    spec = importlib.util.spec_from_file_location("stage2_original_robomme_utils", REPO_ROOT / "examples/robomme/utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    client = policy._motion_client
    if client is None or client.stub or not client.alive:
        raise ValueError("恢复验收必须使用存活的真实 sidecar")
    identity = client.provenance
    sidecar_pid = identity.get("pid")
    if (type(sidecar_pid) is not int or sidecar_pid != client._proc.pid
            or str(identity.get("cuda_visible_devices")) != str(online_gpu)
            or identity.get("gpu_uuid") != inventory[online_gpu]["uuid"]):
        raise ValueError("sidecar 握手物理 GPU/PID 与指定分配不一致")
    policy.reset()
    es = record["exec_start_idx"]
    demo_starts = list(range(0, es - 16, 32))
    counts = []
    cursor = 0
    for count, boundary in ((es, es), (33, 0), (32, 0)):
        end = cursor + count
        policy.add_buffer(utils.pack_buffer(arrays["images"][cursor:end], arrays["states"][cursor:end], boundary))
        counts.append({"frames": end, "motion_keys": sorted(policy.mem_buffer._history_feats_motion),
                       "raw_frame_keys": sorted(policy.mem_buffer._raw_frames)})
        expected = demo_starts + ([] if cursor == 0 else [es] if cursor == es else [es, es + 32])
        if counts[-1]["motion_keys"] != expected or policy.exec_start_idx != es or policy.step_idx != end - 1:
            raise ValueError(f"原 pack_buffer 在线状态机不符: {counts[-1]}")
        cursor = end
    if counts[1]["raw_frame_keys"] != [es + 32] or counts[2]["raw_frame_keys"] != [es + 64]:
        raise ValueError("连续 exec 批次未正确保留共享边界帧")
    outputs = policy.infer({"observation/image": arrays["images"][-1],
                            "observation/wrist_image": arrays["wrist_image"],
                            "observation/state": arrays["states"][-1], "prompt": record["prompt"]})
    actions = np.asarray(outputs["actions"])
    if actions.shape != (policy._model.action_horizon, 8) or not np.isfinite(actions).all():
        raise ValueError(f"真实 infer 输出形状或有限值错误: {actions.shape}")
    gpu = verify_process_gpus(os.getpid(), sidecar_pid, inventory, online_gpu)
    calls = int(client.n_calls)
    if calls != len(demo_starts) + 2:
        raise ValueError(f"真实 sidecar 编码调用数不符: {calls}")
    policy.reset()
    if (policy._motion_client is not client or not client.alive or policy.mem_buffer.n_steps != 0
            or policy.mem_buffer._history_feats_motion or policy.mem_buffer._raw_frames
            or policy.mem_buffer.exec_start_idx is not None or policy.step_idx != -1 or policy.exec_start_idx != 0):
        raise ValueError("reset 未清空 episode 状态或重启了 sidecar")
    return {"online_infer": 1, "pack_buffer_source": "examples/robomme/utils.py", "batches": counts,
            "reset": 1, "sidecar_calls": calls, "gpu": gpu, "sidecar_provenance": identity,
            "actions": {"shape": list(actions.shape), "dtype": str(actions.dtype),
                        "sha256": hashlib.sha256(actions.tobytes()).hexdigest()}}


def restore_and_check(args, report):
    # 必须在导入会初始化后端的模块之前核对资源分配。
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or args.online_gpu != 1:
        raise ValueError("本闸门要求父进程 CUDA_VISIBLE_DEVICES=0，显式 --online-gpu 1")
    if os.environ.get("JAX_PLATFORMS") == "cpu":
        raise ValueError("CPU 合同测试不可启动真实恢复；真实闸门必须使用 GPU")
    checkpoint = Path(args.checkpoint).absolute()
    if checkpoint.is_symlink() or checkpoint.resolve() != checkpoint or not checkpoint.is_dir():
        raise ValueError("checkpoint 必须为无符号链接的现存绝对实体目录")
    root = checkpoint.parent
    step = 19 if args.mode == "smoke" else 60000
    if checkpoint.name != str(step):
        raise ValueError(f"{args.mode} 只验收 step{step}，实际 {checkpoint.name}")
    report.update(checkpoint=str(checkpoint), step=step)
    manifest = verify_delivery_manifest(root, args.expected_manifest, checkpoint) if args.mode == "delivery" else None
    if manifest is not None:
        report["delivery"] = {"manifest_sha256": sha256_file(args.expected_manifest),
                              "files": len(manifest["files"]), "sha256_mismatch": 0}
    provenance = verify_snapshot(root, manifest)
    online_record, online_arrays = load_online_inputs(root)
    report["runtime_dependencies"] = runtime_dependencies(online_record, provenance)
    reference = None
    if args.mode == "smoke":
        reference = read_json(args.reference)
        if (reference.get("schema") != "stage2-pre-save-reference-v1" or reference.get("step") != 19
                or reference.get("state_step") != 20 or reference.get("seed") != 42
                or reference.get("weight_type") != "ema"):
            raise ValueError("保存前参考必须来自 step19/state_step20/seed42 的内存 EMA")
        inference = reference.get("inference", {})
        if (inference.get("parameter_dtype") != "bfloat16" or inference.get("num_steps") != 10
                or inference.get("rng") != "jax.random.split(jax.random.key(seed))[1]"
                or "jax_default_matmul_precision" not in inference
                or inference.get("sharding", {}).get("kind") != "SingleDeviceSharding"
                or inference.get("sharding", {}).get("batch_size") != 1):
            raise ValueError("保存前参考精度、seed 或单卡分片合同不符")
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config
    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils

    for module in (policy_config, config, model_lib):
        if not Path(module.__file__).resolve().is_relative_to(REPO_ROOT / "src"):
            raise ValueError(f"包来源不是本工作树: {module.__file__}")
    if reference is not None:
        jax.config.update("jax_default_matmul_precision", reference["inference"]["jax_default_matmul_precision"])
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError(f"父进程必须只看到一个 GPU: {devices}")
    if reference is not None:
        saved_device = reference["inference"].get("device", {})
        saved_sharding = reference["inference"]["sharding"]
        if (saved_device.get("platform") != "gpu" or saved_device.get("id") != 0
                or saved_device.get("device_kind") != devices[0].device_kind
                or saved_sharding.get("parameter_devices") != [str(devices[0])]
                or saved_sharding.get("input_devices") != [str(devices[0])]):
            raise ValueError("保存前参考与恢复的单卡设备型号或首卡身份不一致")
    inventory = gpu_inventory()
    if 0 not in inventory or 1 not in inventory:
        raise ValueError("缺物理 GPU0/1")
    train_config = config.get_config("mme_vla_suite_b128_80k")
    resolved, enabled = policy_config._load_resolved_snapshot(root)
    if not enabled or resolved.motion.get("encoder_kind") != "stage2_pool4_concat":
        raise ValueError("恢复快照不是已开启的 stage2 encoder")
    train_config = dataclasses.replace(train_config, model=dataclasses.replace(
        train_config.model, history_config=resolved, use_history=True))
    shape_model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    expected = nnx_utils.state_map(nnx.state(shape_model), train_config.freeze_filter,
        lambda item: item.replace(jax.ShapeDtypeStruct(item.value.shape, jnp.bfloat16))).to_pure_dict()
    raw_params = model_lib.restore_params(checkpoint / "params", restore_type=np.ndarray, dtype=None)
    report["strict_restore"] = verify_parameter_tree(expected, raw_params)
    summary, fp32 = parameter_summary(raw_params)
    report["strict_restore"]["finite"] = 1
    report["parameters"] = {"sha256": summary["sha256"], "leaves": len(summary["leaves"])}
    report["fp32_ema"] = fp32
    if reference is not None:
        verify_summary(reference.get("parameters"), summary, "完整 EMA")
        verify_summary(reference.get("fp32_ema"), fp32, "原始 fp32 EMA")
        if not fp32["leaves"]:
            raise ValueError("保存前参考没有原始 fp32 EMA 叶")
        report["ema_mismatch"] = 0
    del raw_params, shape_model, expected, summary
    policy = None
    try:
        policy = policy_config.create_trained_policy(train_config, checkpoint, seed=42,
            sample_kwargs={"num_steps": 10}, motion_online_gpu=args.online_gpu)
        # restore_params 默认生成单卡 NamedSharding；对齐保存前参考的无分片单设备副本。
        policy_state = jax.tree.map(lambda value: jax.device_put(value, devices[0]), nnx.state(policy._model))
        state_leaves = jax.tree.leaves(policy_state)
        if any(str(value.dtype) != "bfloat16" for value in state_leaves):
            raise ValueError("恢复 policy 的实际参数精度不是统一 bfloat16")
        if any(not isinstance(value.sharding, jax.sharding.SingleDeviceSharding)
               or value.devices() != {devices[0]} for value in state_leaves):
            raise ValueError("恢复 policy 参数不是实际单卡 SingleDeviceSharding")
        nnx.update(policy._model, policy_state)
        # module_jit 在创建时冻结 NNX state，换分片后必须重新绑定两个原模型方法。
        policy._sample_actions = nnx_utils.module_jit(policy._model.sample_actions)
        policy._vision_encode = nnx_utils.module_jit(policy._model.vision_encode)
        report["inference"] = {"parameter_dtype": "bfloat16", "num_steps": 10, "seed": 42,
                               "sharding": "SingleDeviceSharding",
                               "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
                               "device_kind": devices[0].device_kind}
        del policy_state, state_leaves
        if reference is not None:
            reference_root = Path(args.reference).resolve().parent
            inputs = load_reference_tree(reference_root, reference["inputs"])
            wanted = load_reference_tree(reference_root, reference["outputs"])
            observation = HistAugObservation.from_dict(jax.tree.map(
                lambda value: jax.device_put(value, devices[0]), inputs))
            output = policy._sample_actions(jax.random.split(jax.random.key(42))[1], observation, num_steps=10)
            jax.block_until_ready(output)
            report["reference_output_max_abs"] = compare_outputs(wanted, {"actions": np.asarray(output)})
            report["reference_sha256"] = sha256_file(args.reference)
        report["online"] = run_online(policy, online_record, online_arrays, args.online_gpu, inventory)
    finally:
        if policy is not None and policy._motion_client is not None:
            policy._motion_client.close()
    report["root_complete"] = 1
    # delivery 不填 reference_output_max_abs / ema_mismatch，避免伪造保存前对拍。
    if args.mode == "smoke":
        return ("SAVE_LOAD=PASS missing=0 extra=0 shape_mismatch=0 dtype_mismatch=0 "
                "ema_mismatch=0 reference_output_max_abs=0 online_infer=1")
    return "CKPT_60K_READY=PASS step=60000 root_complete=1 sha256_mismatch=0 strict_restore=1 online_infer=1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "delivery"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--online-gpu", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.mode == "smoke" and (args.reference is None or args.expected_manifest is not None):
        parser.error("smoke 必须仅给 --reference")
    if args.mode == "delivery" and (args.expected_manifest is None or args.reference is not None):
        parser.error("delivery 必须仅给 --expected-manifest")
    if args.out.resolve().is_relative_to(args.checkpoint.resolve().parent):
        parser.error("报告必须写在 checkpoint/run 根之外，避免污染交付文件")
    if any(path is not None and args.out.resolve() == path.resolve()
           for path in (args.reference, args.expected_manifest)):
        parser.error("报告不能覆盖输入清单")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = {"schema": "stage2-checkpoint-check-v1", "mode": args.mode, "status": "FAIL"}
    started = time.monotonic()
    try:
        line = restore_and_check(args, report)
        report.update(status="PASS", verdict=line)
        code = 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        line = f"{'SAVE_LOAD' if args.mode == 'smoke' else 'CKPT_60K_READY'}=FAIL"
        report["verdict"] = line
        print(report["error"], file=sys.stderr)
        code = 1
    report["elapsed_seconds"] = time.monotonic() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(canonical(report) + b"\n")
        os.replace(temporary, args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(line, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
