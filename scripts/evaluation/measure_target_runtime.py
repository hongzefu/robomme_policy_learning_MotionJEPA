"""隔离的首证测量入口：只消费锁定fixture，不创建服务、任务或成绩。"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.evaluation.runtime_contract import (canonical, compare_runtime, descriptor_identity,
    motion_descriptor, read_json, require_sha256, sha256_file, verified_ref)

ROOT = Path(__file__).resolve().parents[2]


def validate_bootstrap(admission, provenance, actual):
    if admission.get("purpose") != "measurement_bootstrap":
        raise ValueError("测量必须使用隔离purpose")
    fixture_path = Path(admission["fixture"])
    if sha256_file(fixture_path) != require_sha256(admission["fixture_sha256"]):
        raise ValueError("bootstrap fixture SHA不符")
    fixture = read_json(fixture_path)
    if fixture.get("schema") != "eval-motion-reference-v1" or fixture.get("status") != "locked" or not fixture.get("samples"):
        raise ValueError("bootstrap缺固定参考fixture")
    # 测量与引用必须完整，不能拿缺失素材启动GPU路径。
    for sample in fixture["samples"]:
        verified_ref(fixture_path.parent, sample["input"])
        verified_ref(fixture_path.parent, sample["reference"])
    target = {"target_gpu_family": admission["target_gpu_family"], "target_driver": admission["target_driver"]}
    if not target["target_gpu_family"] or not target["target_driver"]:
        raise ValueError("缺显式待测目标GPU/driver")
    for group in ("encoder", "encoder_state_sha256", "source_pin"):
        left, right = provenance[group], actual.get(group)
        if group == "encoder" and isinstance(right, dict):
            left, right = ({k: v for k, v in item.items() if k != "run_dir"} for item in (left, right))
        if left != right:
            raise ValueError(f"bootstrap严格来源不符: {group}")
    compare_runtime(provenance["numerics"], actual.get("numerics"), target)
    compare_runtime(provenance["stage2_compat"]["online_vae"], actual.get("online_vae"), target)
    return target


def encoder_worker(args):
    """在明确的阶段二解释器内读取固定latent；不接受临时更换源码或权重。"""
    if Path(sys.prefix).resolve() != Path(os.environ.get("UV_PROJECT_ENVIRONMENT", "")).resolve():
        raise ValueError("encoder测量解释器与指定环境不符")
    provenance = read_json(args.provenance)
    fixture = read_json(args.fixture)
    if sha256_file(args.fixture) != args.fixture_sha256:
        raise ValueError("encoder fixture SHA不符")
    spec = importlib.util.spec_from_file_location("eval_stage2_sidecar", ROOT / "scripts/dataset/wan_stage2/stage2_sidecar.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pin = module.checked_source_pin()
    if pin != provenance["source_pin"]:
        raise ValueError("encoder测量source pin不符")
    adapter = module._load(ROOT / "scripts/dataset/wan/wan_motion_infer.py", "eval_vae_adapter")
    adapter.check_env()
    adapter.pin_numerics()
    adapter.check_versions(strict=True, with_diffusers=True)
    import torch
    from motion_jepa.export import stage2_memory as shared
    if shared.source_pin() != {key: pin[key] for key in shared.source_pin()} or len(shared.source_pin()) != 3:
        raise ValueError("实际导入的共享encoder源码与锁不符")
    from motion_jepa.export.stage2_memory import load_ema_encoder
    encoder, info = load_ema_encoder(str(args.encoder_checkpoint), device="cuda:0",
                                    expected_sha256=provenance["encoder"]["checkpoint_sha256"])
    compare_runtime(provenance["numerics"], info["numerics"],
                    {"target_gpu_family": args.target_gpu_family, "target_driver": args.target_driver})
    if info["encoder_state_sha256"] != provenance["encoder_state_sha256"]:
        raise ValueError("实际encoder EMA摘要不符")
    for index, sample in enumerate(fixture["samples"]):
        if sample["layer"] != "encoder":
            continue
        latent = np.load(verified_ref(args.fixture.parent, sample["input"]), allow_pickle=False)
        if latent.shape != (9, 16, 32, 32) or latent.dtype != np.float32 or not np.isfinite(latent).all():
            raise ValueError("encoder fixture latent shape/dtype/finite不符")
        value = module.encode_latent(encoder, torch.from_numpy(latent).to("cuda:0"))
        np.save(args.output / f"sample-{index}.npy", value, allow_pickle=False)


def measure(args):
    if not args.fixture.is_file():
        raise FileNotFoundError("锁定reference未交付")
    if args.output.exists():
        raise ValueError("测量目录必须全新")
    provenance = read_json(args.provenance)
    admission = {"purpose": "measurement_bootstrap", "fixture": str(args.fixture),
                 "fixture_sha256": args.fixture_sha256, "target_gpu_family": args.target_gpu_family,
                 "target_driver": args.target_driver}
    # CPU预检全部素材，不先起sidecar再发现参考缺失。
    descriptor = motion_descriptor(provenance, args.fixture_sha256)
    simulated = {"encoder": provenance["encoder"], "encoder_state_sha256": provenance["encoder_state_sha256"],
                 "source_pin": provenance["source_pin"], "numerics": provenance["numerics"],
                 "online_vae": provenance["stage2_compat"]["online_vae"]}
    def target_hardware(value):
        if isinstance(value, dict):
            return {k: args.target_gpu_family if k == "gpu_name" else args.target_driver if k == "driver"
                    else target_hardware(v) for k, v in value.items()}
        return value
    validate_bootstrap(admission, provenance, target_hardware(simulated))
    fixture = read_json(args.fixture)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.motion_online_gpu or "," in args.motion_online_gpu:
        raise ValueError("测量必须沿用显式单Slurm GPU token")
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    client = MotionEncoderClient(online_gpu=args.motion_online_gpu, source_run=args.source_run,
        encoder_kind="stage2_pool4_concat", expected_ckpt_sha256=provenance["encoder"]["checkpoint_sha256"],
        store_provenance=provenance, provenance_root=args.provenance.parent, stage2_python=str(args.stage2_python),
        measurement_bootstrap=admission)
    try:
        # 报告和数组只落用户显式指定的独立测量目录。
        args.output.mkdir(parents=True, exist_ok=False)
        for index, sample in enumerate(fixture["samples"]):
            if sample["layer"] != "e2e":
                continue
            pixels = np.load(verified_ref(args.fixture.parent, sample["input"]), allow_pickle=False)
            value = client(pixels, start_frame=sample["start_frame"])
            np.save(args.output / f"sample-{index}.npy", value, allow_pickle=False)
        # encoder同latent层在同一个明确软件环境另起短进程，不能用VAE结果自比。
        from mme_vla_suite.policies.motion_client import resolve_stage2_dependencies
        dependency = resolve_stage2_dependencies(args.provenance.parent, provenance)
        if dependency is None:
            from mme_vla_suite.policies.motion_client import _REPO_ROOT
            dependency = Path(os.environ.get("MMEVLA_V1_STORE", _REPO_ROOT / "v1-store")) / "external/motionjepa" / provenance["encoder"]["run_name"]
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(args.stage2_python.absolute().parent.parent),
                   UV_LINK_MODE="copy", UV_CACHE_DIR=os.environ.get("UV_CACHE_DIR", str(ROOT / "v1-store/cache/uv")))
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        command = ["uv", "run", "--project", str(ROOT / "scripts/dataset/wan_stage2"), "--no-sync",
                   "--python", str(args.stage2_python), str(args.stage2_python), "-m", "scripts.evaluation.measure_target_runtime",
                   "--encoder-worker", "--fixture", str(args.fixture.absolute()), "--fixture-sha256", args.fixture_sha256,
                   "--provenance", str(args.provenance.absolute()), "--output", str(args.output.absolute()),
                   "--target-gpu-family", args.target_gpu_family, "--target-driver", args.target_driver,
                   "--encoder-checkpoint", str(dependency / provenance["encoder"]["checkpoint_name"])]
        subprocess.run(command, env=env, cwd=ROOT, check=True, timeout=args.timeout)
        runtime = {"numerics": client.provenance["numerics"], "online_vae": client.provenance["online_vae"]}
        record = {"schema": "eval-motion-measurement-v1", "status": "unvalidated_measurement",
            "purpose": "measurement_bootstrap", "fixture_manifest_sha256": args.fixture_sha256,
            "motion_identity_sha256": descriptor_identity("motion", descriptor),
            "measurement_code_sha256": sha256_file(__file__), "runtime": runtime,
            "target_gpu_family": args.target_gpu_family, "target_driver": args.target_driver,
            "sidecar": client.provenance, "samples": [
                {"id": sample["id"], "actual": {"path": f"sample-{i}.npy", "sha256": sha256_file(args.output / f"sample-{i}.npy")}}
                for i, sample in enumerate(fixture["samples"])]}
        (args.output / "measurement.json").write_bytes(canonical(record) + b"\n")
        return record
    finally:
        client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-worker", action="store_true", help=argparse.SUPPRESS)
    for name in ("fixture", "provenance", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--target-gpu-family", required=True)
    parser.add_argument("--target-driver", required=True)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--source-run")
    parser.add_argument("--motion-online-gpu")
    parser.add_argument("--stage2-python", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    if not args.encoder_worker and any(value is None for value in (args.source_run, args.motion_online_gpu, args.stage2_python)):
        parser.error("隔离测量须明确source_run、单GPU token与stage2解释器")
    try:
        if args.encoder_worker:
            encoder_worker(args)
        else:
            measure(args)
    except FileNotFoundError as error:
        print(f"TARGET_MEASUREMENT=PENDING detail={error}")
        return 2
    except Exception as error:
        print(f"TARGET_MEASUREMENT=FAIL detail={type(error).__name__}: {error}")
        return 1
    print("TARGET_MEASUREMENT=UNVALIDATED production_authorized=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
