"""显式给定已有JobID列表，逐个启动可直接加入队列的worker。"""
from __future__ import annotations

import argparse
import shlex

try:
    from .gl_official_pool import srun_command, start_worker
    from .queue_store import read_json
except ImportError:
    from gl_official_pool import srun_command, start_worker
    from queue_store import read_json


def validate_spec(spec):
    required = {"repo", "policy_python", "stage2_python", "sim_python", "checkpoint",
                "policy_config", "run_identity_sha256", "model_kind"}
    if any(not isinstance(spec.get(k), str) or not spec[k] for k in required):
        raise ValueError("spec缺少明确的模型、解释器、代码或运行身份")
    if spec["model_kind"] != "stage2_pool4_concat":
        raise ValueError("当前执行对象必须为stage2_pool4_concat")
    value = spec["run_identity_sha256"]
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("运行身份必须为SHA256")
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--job-id", action="append", required=True, help="只使用逐个明确给出的已有hold")
    parser.add_argument("--worker-prefix", default="worker")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--ssh-host", default="greatlakes")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args(argv)
    validate_spec(read_json(args.spec))
    if len(set(args.job_id)) != len(args.job_id):
        raise ValueError("同一启动批次不可重复JobID")
    for i, job in enumerate(args.job_id):
        worker = f"{args.worker_prefix}-{i + 1}"
        if args.start:
            print(start_worker(args.run_root, args.spec, worker, job, cpus=args.cpus, ssh_host=args.ssh_host), flush=True)
        else:
            print(shlex.join(srun_command(args.run_root, args.spec, worker, job, cpus=args.cpus, ssh_host=args.ssh_host)))


if __name__ == "__main__":
    main()
