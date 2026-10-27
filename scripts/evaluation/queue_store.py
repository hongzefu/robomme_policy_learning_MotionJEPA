"""单控制器文件队列：分配、逐局终局、正常归还与明确step停止后的恢复。"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import uuid

try:
    from .manifest import canonical_bytes, digest
except ImportError:
    from manifest import canonical_bytes, digest


def read_json(path):
    return json.loads(Path(path).read_bytes())


def _write(path, value, *, replace):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def publish(path, value):
    _write(path, value, replace=False)


def replace_json(path, value):
    _write(path, value, replace=True)


def confirm_step_terminal(job_id, step_id, slurm_host="greatlakes"):
    """只接受sacct中完全匹配的数值step终态，不使用mtime或heartbeat。"""
    if not str(job_id).isdigit() or not str(step_id).isdigit():
        raise ValueError("JobID与StepID必须为明确数字")
    exact = f"{job_id}.{step_id}"
    command = ["sacct", "-j", exact, "--noheader", "--parsable2", "--format=JobIDRaw,State,End"]
    if slurm_host:
        command = ["ssh", slurm_host, *command]
    raw = subprocess.check_output(command, text=True, timeout=30)
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
    rows = [line.split("|") for line in raw.splitlines() if line.split("|")[0] == exact]
    if (len(rows) != 1 or len(rows[0]) < 3 or rows[0][1].split()[0].rstrip("+") not in terminal
            or rows[0][2] in ("", "Unknown", "None")):
        raise ValueError(f"尚未明确确认旧step终态：{exact}；{raw.strip()}")
    return {"job_id": str(job_id), "step_id": str(step_id), "state": rows[0][1], "end": rows[0][2], "raw": raw}


class QueueStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    @staticmethod
    def name(value):
        return digest(value)

    def drain_path(self, worker_id):
        return self.root / "queue/drain" / self.name(worker_id)

    def request(self, request_id, operation, payload):
        record = {"request_id": request_id, "operation": operation, "payload": payload}
        path = self.root / "queue/requests" / f"{self.name(request_id)}.json"
        try:
            publish(path, record)
        except FileExistsError:
            if read_json(path) != record:
                raise ValueError("同一请求编号不得改变内容") from None
        return path

    def query(self, request_id):
        path = self.root / "queue/responses" / f"{self.name(request_id)}.json"
        return read_json(path) if path.exists() else None

    def initialize(self, manifest=None, spec=None):
        self.root.mkdir(parents=True, exist_ok=True)
        for name, value in (("manifest.json", manifest), ("spec.json", spec)):
            if value is not None:
                publish(self.root / name, value)
        for directory in ("queue/requests", "queue/responses", "queue/drain", "attempts", "accepted", "logs"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def accepted(self):
        return {record["episode_key"]: record for record in
                (read_json(path) for path in (self.root / "accepted").glob("*.json"))}


class Controller:
    def __init__(self, store, *, terminal_lookup=confirm_step_terminal):
        self.store, self.root = store, store.root
        self.terminal_lookup = terminal_lookup
        self.lock = None

    def __enter__(self):
        self.store.initialize()
        self.lock = (self.root / "queue/controller.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            raise RuntimeError("已有控制器持锁") from None
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            manifest_path = self.root / "inputs/episodes.json"
        manifest, self.spec = read_json(manifest_path), read_json(self.root / "spec.json")
        episodes = manifest.get("episodes") or [e for chunk in manifest["chunks"] for e in chunk["episodes"]]
        self.episodes = {e["episode_key"]: e for e in episodes}
        if not episodes or len(self.episodes) != len(episodes):
            raise ValueError("任务清单为空或含重复局")
        self.chunks = []
        for chunk in manifest["chunks"]:
            keys = chunk.get("episode_keys") or [e["episode_key"] for e in chunk["episodes"]]
            self.chunks.append({"chunk_id": chunk.get("chunk_id", digest(keys)), "episode_keys": keys})
        keys = [e["episode_key"] for e in episodes]
        if [c["episode_keys"] for c in self.chunks] != [keys[i:i + 5] for i in range(0, len(keys), 5)]:
            raise ValueError("清单必须按连续每五局切块，末块可不足五局")
        path = self.root / "queue/state.json"
        self.state = read_json(path) if path.exists() else {"workers": {}, "assignments": {},
                     "grants": {}, "retries": {}, "errors": {}, "uncertain": []}
        # 分配已落盘但快照未落盘时，从控制器自己生成的分配记录恢复。
        for path in (self.root / "attempts").glob("*/assignment.json"):
            grant = read_json(path)
            self.state["assignments"].setdefault(grant["attempt_id"], {"grant": grant, "closed": False})
            self.state["grants"].setdefault(grant["request_id"], grant)
        self.save()
        return self

    def __exit__(self, *unused):
        if self.lock:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None

    def save(self):
        if self.lock is None:
            raise RuntimeError("控制器未持锁")
        replace_json(self.root / "queue/state.json", self.state)

    def register(self, worker_id, job_id, step_id):
        worker = {"job_id": str(job_id), "step_id": str(step_id), "state": "RUNNING"}
        old = self.state["workers"].get(worker_id)
        if old and (old["job_id"], old["step_id"]) != (worker["job_id"], worker["step_id"]):
            raise ValueError("新执行必须使用新的worker-id")
        if old and old["state"] == "STOPPED":
            raise ValueError("已停止worker不得重新注册")
        self.state["workers"][worker_id] = worker
        self.save()

    def claim(self, request_id, worker_id):
        if request_id in self.state["grants"]:
            grant = self.state["grants"][request_id]
            if grant["worker_id"] != worker_id:
                raise ValueError("请求编号属于其他worker")
            return {"status": "GRANTED", "grant": deepcopy(grant)}
        if self.store.drain_path(worker_id).exists() or self.state["workers"][worker_id]["state"] != "RUNNING":
            return {"status": "DRAIN"}
        active = [a["grant"] for a in self.state["assignments"].values() if not a["closed"]]
        if any(g["worker_id"] == worker_id for g in active):
            raise ValueError("worker已有一块；回执丢失时须query原请求")
        occupied = {g["chunk_id"] for g in active}
        completed = self.store.accepted()
        for chunk in self.chunks:
            remaining = [self.episodes[k] for k in chunk["episode_keys"]
                         if k not in completed and k not in self.state["errors"]]
            if chunk["chunk_id"] in occupied or not remaining:
                continue
            attempt_id = uuid.uuid4().hex
            grant = {"chunk_id": chunk["chunk_id"], "attempt_id": attempt_id,
                     "request_id": request_id, "worker_id": worker_id, **self.state["workers"][worker_id],
                     "run_identity_sha256": self.spec["run_identity_sha256"], "episodes": remaining,
                     "attempt_dir": str(self.root / "attempts" / attempt_id)}
            publish(Path(grant["attempt_dir"]) / "assignment.json", grant)
            self.state["assignments"][attempt_id] = {"grant": grant, "closed": False}
            self.state["grants"][request_id] = grant
            self.save()
            return {"status": "GRANTED", "grant": deepcopy(grant)}
        status = self.summary()["compute_status"]
        return {"status": "WAIT" if status == "READY" else status}

    def _record(self, grant, path):
        record = read_json(path)
        members = {e["episode_key"]: e for e in grant["episodes"]}
        key = record["episode_key"]
        if key not in members or any(record.get(k) != v for k, v in members[key].items()):
            raise ValueError("客户端局身份与分配不符")
        if any(record.get(k) != grant[k] for k in ("run_identity_sha256", "chunk_id")):
            raise ValueError("客户端run或chunk身份不符")
        return record

    def import_results(self, attempt_id):
        assignment = self.state["assignments"][attempt_id]
        grant = assignment["grant"]
        if assignment["closed"]:
            return
        for path in sorted((Path(grant["attempt_dir"]) / "client").glob("*.terminal.json")):
            record = self._record(grant, path)
            if record.get("record_type") != "terminal" or record.get("outcome") not in {"success", "fail", "timeout"}:
                raise ValueError("不是正常终局记录")
            target = self.root / "accepted" / f"{digest(record['episode_key'])}.json"
            descriptor = {**deepcopy(record), "attempt_id": attempt_id, "worker_id": grant["worker_id"]}
            if target.exists():
                if read_json(target) != descriptor:
                    raise ValueError("同一局已接受另一终局，拒绝覆盖")
            else:
                publish(target, descriptor)

    def finish(self, attempt_id, *, sim_stopped):
        assignment = self.state["assignments"][attempt_id]
        if assignment["closed"]:
            return
        if sim_stopped is not True:
            raise ValueError("未确认仿真退出不得归还")
        self.import_results(attempt_id)
        grant = assignment["grant"]
        directory = Path(grant["attempt_dir"]) / "client"
        completion = read_json(directory / "attempt.json")
        accepted = self.store.accepted()
        remaining = {e["episode_key"] for e in grant["episodes"] if e["episode_key"] not in accepted}
        begun = {self._record(grant, p)["episode_key"] for p in directory.glob("*.started.json")}
        completed = {e["episode_key"] for e in grant["episodes"] if e["episode_key"] in accepted}
        if (completion.get("exit_code") not in (0, 3) or set(completion["completed_episode_keys"]) != completed
                or set(completion["unstarted_episode_keys"]) != remaining or remaining & begun):
            raise ValueError("归还记录不闭合，或归还了已开始局")
        assignment["closed"] = True
        assignment["returned"] = sorted(remaining)
        self.save()

    def recover(self, worker_id, job_id, step_id):
        worker = self.state["workers"][worker_id]
        if (str(job_id), str(step_id)) != (worker["job_id"], worker["step_id"]):
            raise ValueError("待恢复的JobID/StepID与原worker不符")
        proof = self.terminal_lookup(job_id, step_id)
        if worker["state"] == "STOPPED":
            return self.summary()
        for attempt_id, assignment in self.state["assignments"].items():
            grant = assignment["grant"]
            if assignment["closed"] or grant["worker_id"] != worker_id:
                continue
            self.import_results(attempt_id)
            directory = Path(grant["attempt_dir"]) / "client"
            begun = {self._record(grant, p)["episode_key"] for p in directory.glob("*.started.json")}
            errors = {self._record(grant, p)["episode_key"]: self._record(grant, p)
                      for p in directory.glob("*.infra.json") if read_json(p).get("episode_key")}
            accepted = self.store.accepted()
            for e in grant["episodes"]:
                key = e["episode_key"]
                if key in accepted or key not in begun | set(errors):
                    continue
                error = errors.get(key, {})
                if error.get("uncertain_terminal", True):
                    self.state["uncertain"] = sorted(set(self.state["uncertain"]) | {key})
                if error.get("retryable") is False or self.state["retries"].get(key, 0) >= 1:
                    self.state["errors"][key] = error or {"error": "重试耗尽后step异常退出"}
                else:
                    self.state["retries"][key] = 1
            assignment["closed"] = True
            assignment["recovery"] = proof
        worker["state"] = "STOPPED"
        self.save()
        return self.summary()

    def process_requests(self):
        for path in sorted((self.root / "queue/requests").glob("*.json")):
            target = self.root / "queue/responses" / path.name
            if target.exists():
                continue
            try:
                request = read_json(path)
                operation, payload = request["operation"], request["payload"]
                worker = payload["worker_id"]
                if operation == "register":
                    self.register(worker, payload["job_id"], payload["step_id"])
                    response = {"status": "REGISTERED"}
                elif operation == "claim":
                    response = self.claim(request["request_id"], worker)
                elif operation in {"import", "finish"}:
                    attempt = payload["attempt_id"]
                    if self.state["assignments"][attempt]["grant"]["worker_id"] != worker:
                        raise ValueError("不能提交其他worker的块")
                    self.import_results(attempt) if operation == "import" else self.finish(attempt, sim_stopped=payload["sim_stopped"])
                    response = {"status": "IMPORTED" if operation == "import" else "FINISHED"}
                elif operation == "recover":
                    response = {"status": "RECOVERED", "summary": self.recover(worker, payload["job_id"], payload["step_id"])}
                elif operation == "drain":
                    self.store.drain_path(worker).touch()
                    response = {"status": "DRAINING"}
                elif operation == "error":
                    self.state["workers"][worker]["error"] = payload["error"]
                    self.save()
                    response = {"status": "RECOVERY_REQUIRED"}
                else:
                    raise ValueError("未知队列操作")
            except (ValueError, KeyError, OSError, TypeError, subprocess.SubprocessError) as exc:
                response = {"status": "REJECTED", "error": str(exc)}
            publish(target, response)

    def summary(self):
        return summarize(self.store, self.state, self.episodes)


def summarize(store, state, episodes):
    accepted = store.accepted()
    missing = set(episodes) - set(accepted)
    active = sum(not a["closed"] for a in state["assignments"].values())
    ready = missing - set(state["errors"])
    status = "WAIT" if active else "READY" if ready else "STOPPED_ERROR" if missing else "COMPUTE_DONE"
    outcomes = {name: sum(r["outcome"] == name for r in accepted.values()) for name in ("success", "fail", "timeout")}
    return {"compute_status": status, "episodes": len(episodes), "completed": len(accepted),
            "outcomes": outcomes, "success_rate": outcomes["success"] / len(episodes),
            "missing": sorted(missing), "errors": deepcopy(state["errors"]),
            "retries": deepcopy(state["retries"]), "uncertain": list(state["uncertain"]), "active_chunks": active,
            # testx resample 口径：终局记录带 motion_downsample_steps（缺失按 0）
            "downsampled_episodes": sum(r.get("motion_downsample_steps", 0) > 0 for r in accepted.values())}
