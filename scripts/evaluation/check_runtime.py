"""评估专用单分配GPU恢复；CPU树检查与真实模型验证明确分开。"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.evaluation.runtime_contract import (canonical, numeric_metrics, read_json, require_sha256,
                                               sha256_file, verified_ref, identity_sha256)
from scripts.training.check_stage2_checkpoint import (flatten, load_reference_tree, parameter_summary,
    verify_delivery_manifest, verify_parameter_tree, verify_snapshot)

ROOT = Path(__file__).resolve().parents[2]


def check_single_gpu_token(token, *, environ=None):
    environ = os.environ if environ is None else environ
    if not isinstance(token, str) or not token or "," in token or token.strip() != token:
        raise ValueError("必须明确单个Slurm分配token，可为物理序号或UUID")
    if environ.get("CUDA_VISIBLE_DEVICES") != token:
        raise ValueError("禁止重写Slurm CUDA_VISIBLE_DEVICES")
    if not environ.get("SLURM_JOB_ID") or not environ.get("SLURM_STEP_ID"):
        raise ValueError("真实恢复须在明确Slurm job/step内执行")
    return token


def check_restore_tree(expected, actual):
    """无需模型/GPU即可测的纯严格恢复核心；拒绝空、额外、dtype/shape/finite错误。"""
    if not flatten(expected) or not flatten(actual):
        raise ValueError("参数树不可为空")
    result = verify_parameter_tree(expected, actual)
    summary, fp32 = parameter_summary(actual)
    return {"strict_restore": result, "parameters": summary, "fp32_ema": fp32}


def bf16_parameter_summary(tree):
    """逐叶转换以控制CPU内存峰值，完整记录送进模型的bf16字节。"""
    import ml_dtypes
    leaves = {}
    for name, value in flatten(tree).items():
        array = np.asarray(value).astype(ml_dtypes.bfloat16)
        if not np.isfinite(array).all():
            raise ValueError(f"bf16参数含非有限值: {name}")
        leaves[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    if not leaves:
        raise ValueError("bf16参数树为空")
    return {"algorithm": "named-arrays-v1", "leaves": leaves, "sha256": identity_sha256(leaves)}


def validate_restore_purpose(checkpoint, step, purpose):
    checkpoint = Path(checkpoint)
    if type(step) is not int or step < 0 or checkpoint.name != str(step):
        raise ValueError("显式step必须匹配checkpoint目录名")
    if purpose not in ("validation", "production") or (purpose == "production" and step != 60000):
        raise ValueError("正式评估只接受60000，验证checkpoint不能冒充正式")


def process_gpu_binding(sidecar, *, output=None):
    if output is None:
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits"], text=True, timeout=15)
    wanted = {os.getpid(), sidecar["pid"]}
    observed = {pid: set() for pid in wanted}
    for line in output.splitlines():
        pid, gpu = (item.strip() for item in line.split(",", 1))
        if int(pid) in observed:
            observed[int(pid)].add(gpu)
    if len(wanted) != 2 or any(value != {sidecar["gpu_uuid"]} for value in observed.values()):
        raise ValueError(f"父进程/sidecar没有同用唯一分配GPU: {observed}")
    return {"parent_pid": os.getpid(), "sidecar_pid": sidecar["pid"], "gpu_uuid": sidecar["gpu_uuid"]}


def check_restore(args, report=None):
    """真实入口不沿用训练检查器的物理GPU0/1和固定step19限制。"""
    report = {} if report is None else report
    validate_restore_purpose(args.checkpoint, args.step, args.purpose)
    check_single_gpu_token(args.motion_online_gpu)
    if os.environ.get("JAX_PLATFORMS") == "cpu":
        raise ValueError("CPU测试不能冒充真实恢复")
    checkpoint = Path(args.checkpoint).absolute()
    if checkpoint.is_symlink() or not checkpoint.is_dir() or checkpoint.resolve() != checkpoint:
        raise ValueError("checkpoint必须为无别名实体目录")
    if args.reference is None or args.reference_sha256 is None:
        raise FileNotFoundError("缺锁定的完整输入/noise/动作reference；恢复与重复性保持pending")
    if sha256_file(args.reference) != require_sha256(args.reference_sha256):
        raise ValueError("恢复reference身份不符")
    reference = read_json(args.reference)
    if (reference.get("schema") != "eval-vla-restore-reference-v1" or reference.get("status") != "locked"
            or reference.get("model_seed") != 42 or reference.get("num_steps") != 10
            or reference.get("parameter_dtype") != "bfloat16" or not reference.get("sources")):
        raise ValueError("reference必须锁定真实来源、seed42、bf16与10步采样")
    if args.purpose == "production" and args.expected_manifest is None:
        raise ValueError("正式60000须给完整交付manifest")
    root = checkpoint.parent
    delivery = verify_delivery_manifest(root, args.expected_manifest, checkpoint) if args.expected_manifest else None
    provenance = verify_snapshot(root, delivery)
    # 此后才允许导入会初始化后端的模块。
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("父进程必须恰好看见一张GPU")
    resolved, enabled = policy_config._load_resolved_snapshot(root)
    if not enabled or resolved.motion.get("encoder_kind") != "stage2_pool4_concat":
        raise ValueError("恢复必须为原快照stage2 motion模型")
    train = config.get_config(args.config)
    train = dataclasses.replace(train, model=dataclasses.replace(train.model, history_config=resolved, use_history=True))
    shape_model = nnx.eval_shape(train.model.create, jax.random.key(0))
    expected = nnx_utils.state_map(nnx.state(shape_model), train.freeze_filter,
        lambda item: item.replace(jax.ShapeDtypeStruct(item.value.shape, jnp.bfloat16))).to_pure_dict()
    raw = model_lib.restore_params(checkpoint / "params", restore_type=np.ndarray, dtype=None)
    report.update(check_restore_tree(expected, raw))
    expected_bf16 = bf16_parameter_summary(raw)
    checkpoint_files = {path.relative_to(checkpoint).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                        for path in sorted(checkpoint.rglob("*")) if path.is_file()}
    if not checkpoint_files:
        raise ValueError("checkpoint文件清单为空")
    checkpoint_digest = identity_sha256(checkpoint_files)
    report["checkpoint_files_sha256"] = checkpoint_digest
    del raw, expected, shape_model
    policy = None
    try:
        policy = policy_config.create_trained_policy(train, checkpoint, seed=42,
            motion_online_gpu=args.motion_online_gpu, stage2_python=str(args.stage2_python),
            eval_target_contract=args.eval_target_contract, sample_kwargs={"num_steps": 10})
        if policy._seed != 42 or policy.metadata["model_seed"] != 42:
            raise ValueError("实际模型seed不符")
        parameter_state = jax.tree.map(lambda value: jax.device_put(value, devices[0]), nnx.state(policy._model))
        if any(str(value.dtype) != "bfloat16" for value in jax.tree.leaves(parameter_state)):
            raise ValueError("实际policy参数未统一恢复为bf16")
        report["bf16_parameters"] = bf16_parameter_summary(nnx.state(policy._model, nnx.Param).to_pure_dict())
        if report["bf16_parameters"] != expected_bf16:
            raise ValueError("实际policy bf16参数与checkpoint逐叶摘要不符")
        nnx.update(policy._model, parameter_state)
        policy._sample_actions = nnx_utils.module_jit(policy._model.sample_actions)
        fingerprint = {"jax": jax.__version__, "jaxlib": importlib.metadata.version("jaxlib"),
            "flax": importlib.metadata.version("flax"), "device_kind": devices[0].device_kind,
            "parameter_dtype": "bfloat16", "num_steps": 10,
            "matmul_precision": str(jax.config.jax_default_matmul_precision),
            "model_source_sha256": sha256_file(sys.modules[type(policy._model).__module__].__file__),
            "resolved_sha256": sha256_file(root / "history_config.resolved.yaml"),
            "checkpoint_files_sha256": checkpoint_digest, "inputs_descriptor_sha256": identity_sha256(reference["inputs"]),
            "noise_sha256": reference["noise"]["sha256"], "dependency_lock_sha256": sha256_file(ROOT / "uv.lock"),
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "gpu_uuid": policy._motion_client.provenance["gpu_uuid"],
            "driver": policy._motion_client.provenance["numerics"]["driver"]}
        report["environment_fingerprint"] = fingerprint
        inputs = load_reference_tree(args.reference.parent, reference["inputs"])
        # noise为完整32动作维，不把执行8维切片补零当完整noise。
        noise = np.load(verified_ref(args.reference.parent, reference["noise"]), allow_pickle=False)
        if noise.shape != (1, 20, policy._model.action_dim) or noise.dtype != np.float32 or not np.isfinite(noise).all():
            raise ValueError("完整noise必须为(1,20,action_dim) float32有限数组")
        if policy._model.action_horizon != 20:
            raise ValueError("动作horizon必须为20")
        observation = HistAugObservation.from_dict(jax.tree.map(lambda value: jax.device_put(value, devices[0]), inputs))
        outputs = []
        for _ in range(2):
            policy.reset()
            # 与policy.infer相同split顺序，并显式使用固定完整noise。
            policy._rng, sample_rng = jax.random.split(policy._rng)
            actions = policy._sample_actions(sample_rng, observation, num_steps=10,
                                             noise=jax.device_put(noise, devices[0]))
            jax.block_until_ready(actions)
            transformed = policy._output_transform({"state": np.asarray(observation.state[0]),
                                                    "actions": np.asarray(actions[0])})
            outputs.append(np.asarray(transformed["actions"]))
        if any(value.shape != (20, 8) or not np.isfinite(value).all() for value in outputs):
            raise ValueError("完整动作必须为20×8且有限")
        if outputs[0].tobytes() != outputs[1].tobytes():
            raise ValueError("相同输入/noise/reset后的动作不逐位重复")
        report["repeatability"] = {"max_abs": 0, "actions_sha256": hashlib.sha256(outputs[0].tobytes()).hexdigest(),
                                   "full_shape": [20, 8], "execution_shape": [16, 8], "num_steps": 10, "seed": 42}
        # 跨卡动作阈值必须预先冻结；缺同checkpoint真值时如实标注，不伪造数值通过。
        if "actions" in reference:
            wanted = np.load(verified_ref(args.reference.parent, reference["actions"]), allow_pickle=False)
            if reference.get("checkpoint_files_sha256") != checkpoint_digest:
                raise ValueError("动作参考未绑定同一checkpoint")
            if reference.get("comparison") == "same_environment_bitwise":
                if reference.get("environment_fingerprint") != fingerprint:
                    raise ValueError("保存参考与现场环境指纹不符，不能复用旧baseline")
                metric = numeric_metrics(wanted, outputs[0])
                if metric["max_abs"] != 0 or wanted.tobytes() != outputs[0].tobytes():
                    raise ValueError("同环境动作不逐位一致")
                report["same_checkpoint_actions"] = metric
            else:
                # 跨卡阈值未由本计划批准，不在运行后自行选择。
                report["cross_gpu_same_checkpoint"] = "NOT_AVAILABLE"
        else:
            report["cross_gpu_same_checkpoint"] = "NOT_AVAILABLE"
        report["gpu"] = process_gpu_binding(policy._motion_client.provenance)
        report.update(step=args.step, purpose=args.purpose, reference_sha256=args.reference_sha256,
                      snapshot_provenance_sha256=sha256_file(root / "motion_provenance.json"),
                      eval_target_contract_sha256=sha256_file(args.eval_target_contract), parameter_dtype="bfloat16")
    finally:
        if policy is not None and policy._motion_client is not None:
            policy._motion_client.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--purpose", choices=("validation", "production"), required=True)
    parser.add_argument("--motion-online-gpu", required=True)
    parser.add_argument("--stage2-python", type=Path, required=True)
    parser.add_argument("--eval-target-contract", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--config", default="mme_vla_suite_b128_80k")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve().is_relative_to(args.checkpoint.resolve().parent):
        parser.error("报告须写全新、独立于checkpoint的产物路径")
    report = {"schema": "eval-restore-check-v1", "purpose": args.purpose, "status": "FAIL"}
    try:
        check_restore(args, report)
        report["status"], code = "PASS", 0
    except FileNotFoundError as error:
        report.update(status="pending", error=str(error))
        code = 2
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        code = 1
    line = f"{'CKPT60K_RESTORE' if args.purpose == 'production' else 'VALIDATION_RESTORE'}={report['status'].upper()}"
    report["verdict"] = line
    repeatability_line = f"{'CKPT60K_REPEATABILITY' if args.purpose == 'production' else 'VALIDATION_REPEATABILITY'}={report['status'].upper()}"
    report["repeatability_verdict"] = repeatability_line
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(report) + b"\n")
    print(line)
    print(repeatability_line)
    if report.get("error"):
        print(report["error"], file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
