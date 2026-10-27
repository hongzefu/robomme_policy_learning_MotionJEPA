"""每卡常驻策略服务，逐块调用现有官方客户端，不操作其他进程或JobID。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from urllib.request import urlopen
import uuid

try:
    from .queue_store import QueueStore, publish, read_json
except ImportError:
    from queue_store import QueueStore, publish, read_json


def server_command(spec, nonce, port):
    command = [spec["policy_python"], "-m", "scripts.training.serve_policy", "--port", str(port),
               "--seed", str(spec.get("model_seed", 42)), "--motion-online-gpu", os.environ["CUDA_VISIBLE_DEVICES"],
               "--stage2-python", spec["stage2_python"],
               "--run-identity-sha256", spec["run_identity_sha256"], "--server-nonce", nonce,
               "--eval-purpose", spec.get("purpose", "validation")]
    if spec.get("target_contract"):
        command += ["--eval-target-contract", spec["target_contract"]]
    return command + ["policy:checkpoint", "--policy.config", spec["policy_config"],
                      "--policy.dir", spec["checkpoint"]]


def client_command(spec, grant_path, output, nonce, port, drain_file):
    command = [spec["sim_python"], str(Path(spec["repo"]) / "examples/robomme/eval.py"),
               "--episode-manifest", str(grant_path), "--chunk", read_json(grant_path)["chunk_id"],
               "--attempt-output", str(output), "--run-identity-sha256", spec["run_identity_sha256"],
               "--expected-server-nonce", nonce, "--model-kind", spec["model_kind"],
               "--model-seed", str(spec.get("model_seed", 42)), "--host", "127.0.0.1", "--port", str(port),
               "--rpc-timeout-seconds", str(spec.get("rpc_timeout_seconds", 600)),
               "--drain-file", str(drain_file)]
    # 只有testx口径追加；官方缺省命令行与改动前逐字相同。
    if spec.get("profile", "official") == "testx":
        command += ["--profile", "testx", "--max-steps", str(spec["max_steps"])]
    return command


def worker_environment(spec):
    # 缓存和CUDA直接继承运行脚本；只把执行副本放到模块检索路径。
    repo = Path(spec["repo"])
    paths = [str(repo), str(repo / "src"), str(repo / "packages/openpi-client/src"),
             str(repo / "third_party/robomme_benchmark/src")]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(paths), PYTHONUNBUFFERED="1")
    # 服务端与客户端都以本环境启动；官方缺省两键都不设，环境与改动前相同。
    if spec.get("xhard_metadata_dir"):
        environment["XHARD_METADATA_DIR"] = spec["xhard_metadata_dir"]
    if spec.get("motion_overflow", "raise") == "resample":
        environment["MMEVLA_MOTION_OVERFLOW"] = "resample"
    return environment


def stop_process_tree(process, grace_seconds=10):
    """只停止本worker用start_new_session创建的子进程组。"""
    if process is None:
        return True
    for sig, delay in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.poll()
            return True
        end = time.monotonic() + delay
        while time.monotonic() < end:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return True
            threading.Event().wait(0.05)
    return False


class Resident:
    def __init__(self, root, spec, worker_id, job_id, *, popen=subprocess.Popen):
        self.root, self.spec = Path(root).resolve(), spec
        self.worker_id, self.job_id = worker_id, str(job_id)
        self.step_id = os.environ.get("SLURM_STEP_ID", "")
        self.store = QueueStore(self.root)
        self.popen = popen
        self.server = self.sim = None
        self.stop = threading.Event()
        self.nonce, self.port = uuid.uuid4().hex, None
        self.drain_file = self.store.drain_path(worker_id)

    def rpc(self, operation, payload, request_id=None):
        request_id = request_id or f"{self.worker_id}-{uuid.uuid4().hex}"
        self.store.request(request_id, operation, {"worker_id": self.worker_id, **payload})
        end = time.monotonic() + float(self.spec.get("controller_timeout_seconds", 120))
        while time.monotonic() < end:
            response = self.store.query(request_id)
            if response is not None:
                if response.get("status") == "REJECTED":
                    raise RuntimeError(str(response))
                return response
            if self.server is not None and self.server.poll() is not None:
                raise RuntimeError("策略服务已退出")
            if operation == "claim" and self.stop.is_set():
                return {"status": "DRAIN"}
            threading.Event().wait(0.1)
        raise TimeoutError(f"控制器响应未确认，保留请求编号：{request_id}")

    def run_chunk(self, grant):
        attempt = Path(grant["attempt_dir"])
        grant_path, output = attempt / "grant.json", attempt / "client"
        publish(grant_path, grant)
        command = client_command(self.spec, grant_path, output, self.nonce, self.port, self.drain_file)
        imported = set()
        with (attempt / "sim.log").open("ab", buffering=0) as log:
            self.sim = self.popen(command, cwd=self.spec["repo"], env=worker_environment(self.spec),
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while self.sim.poll() is None:
                terminals = set(output.glob("*.terminal.json"))
                if terminals - imported:
                    self.rpc("import", {"attempt_id": grant["attempt_id"]})
                    imported = terminals
                if self.stop.is_set():
                    self.drain_file.parent.mkdir(parents=True, exist_ok=True)
                    self.drain_file.touch(exist_ok=True)
                if self.server.poll() is not None:
                    raise RuntimeError("策略服务在仿真期间退出")
                threading.Event().wait(0.25)
            code = self.sim.returncode
            if not stop_process_tree(self.sim):
                raise RuntimeError("无法确认仿真子进程组停止")
            self.sim = None
            self.rpc("import", {"attempt_id": grant["attempt_id"]})
            if code not in (0, 3):
                raise RuntimeError(f"客户端异常退出code={code}，等待明确step终态后恢复")
            self.rpc("finish", {"attempt_id": grant["attempt_id"], "sim_stopped": True})
            return code

    def run(self):
        if (not self.step_id or str(os.environ.get("SLURM_JOB_ID")) != self.job_id
                or not os.environ.get("CUDA_VISIBLE_DEVICES")):
            raise ValueError("worker必须在指定JobID的GPU srun step内执行")
        self.rpc("register", {"job_id": self.job_id, "step_id": self.step_id})
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", int(self.spec.get("port", 0))))
            self.port = probe.getsockname()[1]
        logs = self.root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        try:
            with (logs / f"{self.store.name(self.worker_id)}-server.log").open("ab", buffering=0) as log:
                self.server = self.popen(server_command(self.spec, self.nonce, self.port), cwd=self.spec["repo"],
                    env=worker_environment(self.spec), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                end = time.monotonic() + float(self.spec.get("startup_timeout_seconds", 1800))
                while True:
                    if self.server.poll() is not None:
                        raise RuntimeError("策略服务启动退出")
                    try:
                        with urlopen(f"http://127.0.0.1:{self.port}/healthz", timeout=2) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() >= end or self.stop.wait(0.25):
                        raise TimeoutError("策略服务启动等待结束")
                # 客户端首包另核run/nonce/seed/model_kind，健康检查仅说明监听。
                while not self.stop.is_set() and not self.drain_file.exists():
                    response = self.rpc("claim", {})
                    if response["status"] in ("COMPUTE_DONE", "STOPPED_ERROR", "DRAIN"):
                        break
                    if response["status"] == "WAIT":
                        self.stop.wait(0.5)
                        continue
                    if self.run_chunk(response["grant"]) == 3:
                        break
            return 0
        except BaseException as exc:
            self.store.request(f"error-{self.worker_id}-{uuid.uuid4().hex}", "error",
                               {"worker_id": self.worker_id, "error": str(exc)})
            raise
        finally:
            stop_process_tree(self.sim)
            stop_process_tree(self.server)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "spec", "worker-id", "job-id"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    worker = Resident(args.run_root, read_json(args.spec), args.worker_id, args.job_id)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: worker.stop.set())
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(main())
