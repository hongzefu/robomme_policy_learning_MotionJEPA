"""仅构造已有明确JobID的srun命令和记录启动；不提交或取消占位作业。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import uuid

try:
    from .queue_store import publish, read_json
except ImportError:
    from queue_store import publish, read_json


def srun_command(run_root, spec_path, worker_id, job_id, *, cpus=4, ssh_host="greatlakes"):
    spec = read_json(spec_path)
    if not str(job_id).isdigit() or cpus < 1:
        raise ValueError("需明确已有JobID和正数CPU数量")
    command = ["srun", f"--jobid={job_id}", "--overlap", "--exact", "--ntasks=1",
               f"--cpus-per-task={cpus}", "--gpu_cmode=shared", "env", "PYTHONUNBUFFERED=1", spec["policy_python"],
               str(Path(spec["repo"]) / "scripts/evaluation/resident_worker.py"),
               "--run-root", str(run_root), "--spec", str(spec_path), "--worker-id", worker_id, "--job-id", str(job_id)]
    return ["ssh", ssh_host, shlex.join(command)] if ssh_host else command


def start_worker(run_root, spec_path, worker_id, job_id, *, cpus=4, ssh_host="greatlakes", session=None):
    root = Path(run_root).resolve()
    session = session or f"eval-simple-{uuid.uuid4().hex[:12]}"
    log = root / "logs" / f"{session}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = srun_command(root, Path(spec_path).resolve(), worker_id, job_id, cpus=cpus, ssh_host=ssh_host)
    # tmux与tee只拥有本次ssh/srun；不会取消hold或触碰其他会话。
    body = f"set -o pipefail; {shlex.join(command)} 2>&1 | tee {shlex.quote(str(log))}; code=$?; echo EXIT_CODE=$code | tee -a {shlex.quote(str(log))}; exit $code"
    record = {"worker_id": worker_id, "job_id": str(job_id), "tmux": session,
              "log": str(log), "command": command}
    publish(root / "launches" / f"{session}.json", record)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", "-lc", body], check=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--ssh-host", default="greatlakes")
    parser.add_argument("--start", action="store_true", help="实际启动；省略时只打印命令")
    args = parser.parse_args(argv)
    if args.start:
        print(start_worker(args.run_root, args.spec, args.worker_id, args.job_id, cpus=args.cpus, ssh_host=args.ssh_host))
    else:
        print(shlex.join(srun_command(args.run_root, args.spec, args.worker_id, args.job_id, cpus=args.cpus, ssh_host=args.ssh_host)))


if __name__ == "__main__":
    main()
