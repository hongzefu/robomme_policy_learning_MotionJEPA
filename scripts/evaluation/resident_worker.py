"""GL step内的常驻服务与独立逐块仿真进程；不直接领取或写canonical。"""

from __future__ import annotations

import argparse
import hashlib
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
    from .gl_official_pool import atomic_json, read_json
    from .launch_official_pool import validate_spec, verify_launch_gates
except ImportError:
    from gl_official_pool import atomic_json, read_json
    from launch_official_pool import validate_spec, verify_launch_gates


def pick_port(host="127.0.0.1", requested=0):
    """启动前占用探测；metadata中的nonce继续防止误连其他服务。"""
    with socket.socket() as probe:
        probe.bind((host, requested))
        return probe.getsockname()[1]


def worker_environment(spec):
    """缓存仅由共享spec的明确字段设置，CUDA与Slurm身份仍继承计算step。"""
    cache = spec.get("cache_env", {})
    allowed = {"UV_CACHE_DIR", "UV_LINK_MODE", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME",
               "WANDB_DIR", "WANDB_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR", "TRANSFORMERS_CACHE"}
    if set(cache) - allowed or any(not isinstance(value, str) for value in cache.values()):
        raise ValueError("cache_env仅接受明确的缓存字段，不接收控制主机整份环境")
    return dict(os.environ, **cache, PYTHONUNBUFFERED="1")


def stop_process_tree(process, grace_seconds):
    """只操作本worker以新会话创建的进程组，父进程退出也检查残余子进程。"""
    if process is None:
        return True
    group = process.pid
    for sig, wait_seconds in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            process.poll()
            return True
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                return True
            threading.Event().wait(min(0.05, max(0, deadline-time.monotonic())))
    process.poll()
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return True
    return False


def server_command(spec, nonce, port):
    return [spec["policy_python"], "-m", "scripts.training.serve_policy", "--port", str(port),
            "--seed", "42", "--motion-online-gpu", os.environ["CUDA_VISIBLE_DEVICES"],
            "--stage2-python", spec["stage2_python"], "--eval-target-contract", spec["target_contract"],
            "--model-binding", spec["model_binding"],
            "--run-identity-sha256", spec["run_identity_sha256"], "--server-nonce", nonce,
            "--eval-purpose", spec["purpose"],
            "--checkpoint-step", str(spec["checkpoint_step"]),
            "--worker-uuid", spec["worker_uuid"],
            "policy:checkpoint", "--policy.config", spec["policy_config"], "--policy.dir", spec["checkpoint"]]


def client_command(spec, grant_path, output, nonce, port, drain_file):
    return [spec["sim_python"], str(Path(spec["repo"])/"examples/robomme/eval.py"),
            "--episode-manifest", str(grant_path), "--chunk", read_json(grant_path)["chunk_id"],
            "--attempt-output", str(output), "--run-identity-sha256", spec["run_identity_sha256"],
            "--expected-server-nonce", nonce, "--model-kind", spec["model_kind"], "--model-seed", "42",
            "--host", "127.0.0.1", "--port", str(port), "--rpc-timeout-seconds",
            str(spec.get("rpc_timeout_seconds", 600)), "--drain-file", str(drain_file)]


def validate_permit(root, spec, environment):
    if not environment.get("SLURM_JOB_ID") or not environment.get("SLURM_STEP_ID"):
        raise ValueError("resident只能在srun计算step内执行")
    if str(environment["SLURM_JOB_ID"]) != str(spec["job_id"]) or not environment.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Slurm/GPU身份不符")
    ledger = read_json(Path(root)/"pool/ledger.json")
    slot = ledger["slots"][spec["slot_instance"]]
    if ledger["state"] != "ACTIVE" or ledger["pool_revision"] != spec["pool_revision"]:
        raise ValueError("启动许可已冻结或revision变化")
    if slot["state"] != "STARTING" or any(slot.get(key) != spec.get(key) for key in
        ("launch_id", "worker_uuid", "job_id", "job_incarnation", "run_identity_sha256")):
        raise ValueError("迟到或重复的resident启动")
    return dict(slot, step_id=environment["SLURM_STEP_ID"], state="RUNNING")


def validate_server_metadata(metadata, spec, nonce, server_pid, environment):
    """只比对已签封的摘要；12GB实体由刚启动的确定版server现场核验一次。"""
    try:
        from .manifest import digest, verify_seal
    except ImportError:
        from manifest import digest, verify_seal
    binding = verify_seal(read_json(spec["model_binding"]))
    identity_key = "validation_model_identity_sha256" if spec["purpose"] == "validation" else "model_identity_sha256"
    expected = {"run_identity_sha256": binding["run_identity_sha256"], "server_nonce": nonce,
                "model_seed": 42, "model_kind": spec["model_kind"], "server_pid": server_pid,
                "eval_purpose": spec["purpose"], "checkpoint_step": spec["checkpoint_step"],
                "worker_uuid": spec["worker_uuid"], "model_binding_sha256": binding["sha256"],
                "model_identity_sha256": binding[identity_key],
                "checkpoint_files_sha256": digest(binding["model_descriptor"]["checkpoint_files"]),
                "slurm_job_id": environment["SLURM_JOB_ID"], "slurm_step_id": environment["SLURM_STEP_ID"],
                "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"], "cross_hardware_numeric": "not_required"}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"刚启动服务的现场身份不符：{key}")
    runtime = metadata.get("motion_runtime", {})
    if (not runtime.get("gpu_uuid") or type(runtime.get("pid")) is not int
            or runtime["pid"] <= 0 or runtime.get("cuda_visible_devices") != environment["CUDA_VISIBLE_DEVICES"]):
        raise ValueError("缺真实sidecar进程/GPU UUID/分配token")
    if spec.get("expected_gpu_uuid") is not None and runtime["gpu_uuid"] != spec["expected_gpu_uuid"]:
        raise ValueError("sidecar实际GPU UUID与本hold已核分配不符")
    return metadata


def read_and_release_metadata(spec, nonce, server_pid, port, *, connector=None, codec=None, environment=None):
    """读取首包后使用既有release协议释放会话；不reset、不推理、不多占仿真局。"""
    if connector is None:
        from websockets.sync.client import connect
        connector = connect
    if codec is None:
        from openpi_client import msgpack_numpy
        codec = msgpack_numpy
    timeout = float(spec.get("metadata_timeout_seconds", 30))
    with connector(f"ws://127.0.0.1:{port}", compression=None, max_size=None,
                   open_timeout=timeout, close_timeout=min(10, timeout)) as websocket:
        response = websocket.recv(timeout=timeout)
        if not isinstance(response, bytes):
            raise RuntimeError("metadata不是有类型的协议首包")
        metadata = codec.unpackb(response)
        validate_server_metadata(metadata, spec, nonce, server_pid, os.environ if environment is None else environment)
        rpc = {"episode_nonce": uuid.uuid4().hex}
        websocket.send(codec.Packer().pack({"release_attempt": True, "eval_rpc": rpc}))
        reply = websocket.recv(timeout=timeout)
        if not isinstance(reply, bytes):
            raise RuntimeError("metadata探针的会话释放响应不确定")
        ack = codec.unpackb(reply)
        if ack.get("release_attempt_finished") is not True or ack.get("quiescent") is not True or ack.get("eval_rpc") != rpc:
            raise RuntimeError("metadata探针未确认静止释放，停止resident")
        return metadata


class Resident:
    def __init__(self, root, spec, store, *, popen=subprocess.Popen):
        self.root, self.spec, self.store = Path(root), spec, store
        self.popen = popen
        self.server = self.sim = None
        self.stop = threading.Event()
        self.forwarded = set()
        self.nonce = uuid.uuid4().hex
        self.port = None
        self.last_heartbeat = 0.0
        self.drain_file = self.root/"pool"/"drain"/(spec["slot_instance"] + ".json")
        self.request_timeout = float(spec.get("controller_timeout_seconds", 60))
        self.grace = float(spec.get("term_grace_seconds", 10))
        if self.grace <= 0 or self.grace > 120:
            raise ValueError("TERM宽限必须在(0,120]秒")

    def rpc(self, operation, payload, request_id=None, *, timeout=None):
        request_id = request_id or self.spec["launch_id"] + "-" + uuid.uuid4().hex
        self.store.request(request_id, operation, payload)
        end = time.monotonic() + (self.request_timeout if timeout is None else timeout)
        while time.monotonic() < end:
            response = self.store.query(request_id)
            if response is not None:
                if response.get("status") in {"ERROR", "REJECTED", "UNKNOWN"} and operation != "artifact":
                    raise RuntimeError(f"controller拒绝{operation}：{response}")
                return response
            if operation == "claim":
                if self.drain_file.exists() or time.time() >= min(float(self.spec["hold_end_time"]), float(self.spec["workload_deadline"])):
                    return {"status": "DRAIN"}
                if self.server is not None and self.server.poll() is not None:
                    raise RuntimeError("等待领取时server死亡")
            if self.stop.wait(0.1):
                raise RuntimeError("停止请求中断controller等待")
        raise TimeoutError(f"controller响应不确定：{request_id}")

    def heartbeat(self, stage, **extra):
        if stage == "chunk" and time.monotonic()-self.last_heartbeat < 30:
            return
        self.last_heartbeat = time.monotonic()
        # 不启动nvidia-smi；GPU监控影响尚无实证，明确保留pending。
        rss = {}
        for name, proc in (("server", self.server), ("sim", self.sim)):
            if proc is not None:
                try:
                    fields = Path(f"/proc/{proc.pid}/status").read_text().splitlines()
                    rss[name] = next((line.split(":", 1)[1].strip() for line in fields if line.startswith("VmRSS:")), None)
                except FileNotFoundError:
                    rss[name] = None
        atomic_json(self.root/"pool"/"heartbeats"/(self.spec["worker_uuid"] + ".json"),
                    {"worker_uuid": self.spec["worker_uuid"], "slot_instance": self.spec["slot_instance"],
                     "job_id": self.spec["job_id"], "step_id": os.environ.get("SLURM_STEP_ID"),
                     "time": time.time(), "stage": stage, "rss": rss,
                     "gpu_monitor_overhead": "pending", "server_pid": getattr(self.server, "pid", None),
                     "sim_pid": getattr(self.sim, "pid", None), "port": self.port, "server_nonce": self.nonce, **extra})

    def receipt(self, grant, kind, payload, receipt_id):
        path = self.store.write_attempt(grant, kind, payload, receipt_id)
        if kind in {"terminal", "artifact", "infra"}:
            self.rpc(kind, {"path": str(path)}, "receipt-" + grant["attempt_id"] + "-" + receipt_id)

    def forward_receipts(self, grant, client_output):
        """终局文件出现即接受，不等待独立的视频封口。"""
        for kind in ("started", "terminal", "infra", "artifact"):
            paths = list(client_output.glob(f"*.{kind}.json"))
            if kind == "artifact":
                paths.extend(client_output.glob("*.close-error.json"))
                paths.extend(client_output.glob("*.artifact-error.json"))
            for path in sorted(paths):
                key = str(path)
                if key in self.forwarded:
                    continue
                payload = read_json(path)
                if kind == "infra" and not payload.get("episode_key"):
                    retryable = bool(payload.get("retryable", True))
                    self.rpc("worker_infra" if retryable else "worker_error", {"slot_instance": self.spec["slot_instance"],
                             "error": payload.get("error", "client全局握手失败"),
                             "retryable": retryable,
                             "uncertain_terminal": bool(payload.get("uncertain_terminal", retryable))})
                    self.forwarded.add(key)
                    continue
                if kind == "artifact":
                    files = []
                    for item in payload.get("files", []):
                        asset = (client_output/item["path"]).resolve()
                        asset.relative_to(client_output.resolve())
                        relative = str(asset.relative_to(self.root.resolve()))
                        files.append(dict(item, path=relative))
                    payload = dict(payload, files=files)
                receipt_id = hashlib.sha256(path.name.encode()).hexdigest()
                self.receipt(grant, kind, payload, receipt_id)
                self.forwarded.add(key)

    def run_chunk(self, grant):
        if not 1 <= len(grant["episodes"]) <= 5 or grant["run_identity_sha256"] != self.spec["run_identity_sha256"]:
            raise ValueError("chunk规模或身份不符")
        attempt = Path(grant["attempt_dir"])
        attempt.resolve().relative_to(self.root.resolve())
        grant_path = attempt/"client-grant.json"
        atomic_json(grant_path, grant)
        output = attempt/"client"
        command = client_command(self.spec, grant_path, output, self.nonce, self.port, self.drain_file)
        env = worker_environment(self.spec)
        repo = Path(self.spec["repo"])
        env["PYTHONPATH"] = os.pathsep.join(str(repo/p) for p in (".", "src", "packages/openpi-client/src", "third_party/robomme_benchmark/src"))
        chunk_seconds = len(grant["episodes"])*float(self.spec.get("episode_wall_seconds", 2400)) + float(self.spec.get("startup_cleanup_seconds", 1800))
        deadline = min(time.time()+chunk_seconds, float(self.spec["hold_end_time"]), float(self.spec["workload_deadline"]))
        with (attempt/"sim.log").open("ab", buffering=0) as log:
            self.sim = self.popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while self.sim.poll() is None:
                self.forward_receipts(grant, output)
                if self.stop.is_set() or time.time() >= deadline or self.server.poll() is not None:
                    stop_process_tree(self.sim, self.grace)
                    self.forward_receipts(grant, output)
                    raise RuntimeError("仿真停止或服务死亡，整个resident退出等待controller恢复")
                self.heartbeat("chunk", attempt_id=grant["attempt_id"], chunk_id=grant["chunk_id"])
                self.stop.wait(0.25)
            code = self.sim.returncode
            if not stop_process_tree(self.sim, self.grace):
                raise RuntimeError("仿真进程树无法确认静止")
            self.forward_receipts(grant, output)
            if code not in {0, 3}:
                raise RuntimeError(f"client基础设施或RPC不确定退出：{code}")
            completion = read_json(output/"attempt.json")
            expected = {item["episode_key"] for item in grant["episodes"]}
            completed = set(completion["completed_episode_keys"])
            unstarted = set(completion["unstarted_episode_keys"])
            if completed & unstarted or completed | unstarted != expected or (code == 0 and completed != expected):
                raise RuntimeError("client完成清单不闭合")
            terminals = {read_json(path)["episode_key"] for path in output.glob("*.terminal.json")}
            if completed != terminals:
                raise RuntimeError("client完成标记缺少独立terminal收据")
            if code == 3:
                for episode in grant["episodes"]:
                    if episode["episode_key"] in unstarted:
                        self.receipt(grant, "handback", dict(episode, reason="drain"), "handback-"+episode["episode_key"])
            self.rpc("close", {"attempt_id": grant["attempt_id"], "sim_stopped": True})
            self.sim = None
            return code

    def run(self):
        slot = validate_permit(self.root, self.spec, os.environ)
        # 计算节点再次核门；来源与软件精度由server真实恢复和sidecar握手校验。
        verify_launch_gates(self.spec["gates"], self.spec.get("purpose"), self.spec.get("validation_stage"))
        validate_spec(self.spec)
        self.rpc("register_slot", {"slot": {key: slot[key] for key in
                 ("slot_instance", "worker_uuid", "job_id", "job_incarnation", "step_id", "state")}})
        self.port = pick_port(requested=int(self.spec.get("port", 0)))
        log_path = self.root/"logs"/(self.spec["worker_uuid"] + "-server.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log:
            self.server = self.popen(server_command(self.spec, self.nonce, self.port), cwd=self.spec["repo"],
                                     env=worker_environment(self.spec), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            health_until = time.monotonic() + float(self.spec.get("server_load_seconds", 1800))
            while True:
                if self.server.poll() is not None or self.stop.is_set():
                    raise RuntimeError("server加载失败或收到停止信号")
                try:
                    with urlopen(f"http://127.0.0.1:{self.port}/healthz", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= health_until:
                    raise TimeoutError("server健康检查超时")
                self.stop.wait(0.25)
            try:
                metadata = read_and_release_metadata(self.spec, self.nonce, self.server.pid, self.port)
            except ValueError as error:
                self.rpc("worker_error", {"slot_instance": self.spec["slot_instance"], "error": str(error),
                                          "retryable": False, "uncertain_terminal": False})
                raise
            if self.server.poll() is not None:
                raise RuntimeError("metadata核验后服务已退出")
            atomic_json(self.root/"pool/server-metadata"/(self.spec["worker_uuid"]+".json"), metadata)
            self.heartbeat("loaded", first_inference_ready="pending", gpu_uuid=metadata["motion_runtime"]["gpu_uuid"],
                           sidecar_pid=metadata["motion_runtime"]["pid"], cross_hardware_numeric="not_required")
            while not self.stop.is_set():
                # 每个hold独立计时，不把chunk deadline当resident总寿命。
                remaining = min(float(self.spec["hold_end_time"]), float(self.spec["workload_deadline"])) - time.time()
                need = 5*float(self.spec.get("episode_wall_seconds", 2400)) + float(self.spec.get("startup_cleanup_seconds", 1800))
                if self.drain_file.exists() or remaining < need:
                    return 3
                response = self.rpc("claim", {"slot_instance": self.spec["slot_instance"]}, timeout=remaining)
                status = response["status"]
                if status in {"COMPUTE_DONE", "DRAIN"}:
                    return 0 if status == "COMPUTE_DONE" else 3
                if status == "STOPPED_ERROR":
                    return 2
                if status == "READY":
                    self.stop.wait(0.25)
                    continue
                if status != "GRANTED":
                    raise RuntimeError(f"领取状态不确定：{response}")
                if self.run_chunk(response["grant"]) == 3:
                    return 3
        return 3

    def close(self):
        sim_stopped = stop_process_tree(self.sim, self.grace)
        server_stopped = stop_process_tree(self.server, self.grace)
        self.heartbeat("stopped", sim_stopped=sim_stopped, server_group_stopped=server_stopped,
                       rpc_quiescent=server_stopped, sidecar_quiescent=server_stopped)
        return sim_stopped and server_stopped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--launch-spec", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from .queue_store import QueueStore
    except ImportError:
        from queue_store import QueueStore
    spec = read_json(args.launch_spec)
    if args.worker_id != spec["worker_uuid"]:
        raise ValueError("worker身份不符")
    worker = Resident(args.run_root, spec, QueueStore(args.run_root))
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, lambda *_: worker.stop.set())
    code = 2
    try:
        code = worker.run()
    except Exception as error:
        print(f"RESIDENT=FAIL error={error}", flush=True)
    finally:
        if not worker.close():
            code = 2
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(f"RESIDENT_EXIT_CODE={code}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
