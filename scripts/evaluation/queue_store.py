"""单控制器文件队列；worker文件仅为输入，正式记录由控制器复制发布。"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import uuid

try:
    from .manifest import canonical_bytes, digest, validate_episodes
except ImportError:
    from manifest import canonical_bytes, digest, validate_episodes


TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "TERMINATED", "RETIRED", "TERMINAL"}
SLOT_FIELDS = ("slot_instance", "worker_uuid", "job_id", "job_incarnation", "step_id")


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON重复键")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def publish(path, value):
    """同盘同步临时文件后link创建，不覆盖目标，也不链接worker原文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def replace_json(path, value):
    path = Path(path)
    temporary = path.parent / f".state-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class QueueStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def initialize(self, manifest, run_identity_sha256, *, controller_host=None,
                   mode="production", official=True):
        if mode not in {"production", "validation", "measurement_bootstrap"}:
            raise ValueError("未知队列用途")
        validate_episodes(manifest, official=official)
        if len(run_identity_sha256) != 64 or any(c not in "0123456789abcdef" for c in run_identity_sha256):
            raise ValueError("run identity必须为SHA256")
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ("inputs", "queue/claims", "queue/requests", "queue/responses",
                          "attempts", "accepted", "artifacts", "history", "quarantine", "reports"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        publish(self.root / "inputs/episodes.json", manifest)
        publish(self.root / "inputs/queue.json", {"schema_version": 1,
                "run_identity_sha256": run_identity_sha256,
                "controller_host": controller_host or socket.gethostname(), "mode": mode})
        publish(self.root / "queue/state.json", {"revision": 0, "phase": "RUNNING", "slots": {},
                "assignments": {}, "generations": {}, "retries": {}, "exhausted": [],
                "uncertain": [], "requests": {}, "freeze": None, "resize_requests": []})

    @staticmethod
    def request_name(request_id):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("请求编号不能为空")
        return digest(request_id)

    def request(self, request_id, operation, payload):
        record = {"request_id": request_id, "operation": operation, "payload": payload}
        path = self.root / "queue/requests" / f"{self.request_name(request_id)}.json"
        try:
            publish(path, record)
        except FileExistsError:
            if read_json(path) != record:
                raise ValueError("同请求编号的内容发生变化") from None
        return path

    def query(self, request_id):
        path = self.root / "queue/responses" / f"{self.request_name(request_id)}.json"
        return read_json(path) if path.exists() else None

    def write_attempt(self, grant, kind, payload, receipt_id):
        if kind not in {"terminal", "infra", "artifact", "started", "handback"}:
            raise ValueError("未知attempt记录类型")
        for field in ("attempt_id", "chunk_id"):
            value = grant[field]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("attempt/chunk身份必须为SHA256，禁止路径注入")
        if grant["run_identity_sha256"] != read_json(self.root / "inputs/queue.json")["run_identity_sha256"]:
            raise ValueError("grant属于其他run")
        path = self.root / "attempts" / grant["attempt_id"] / kind / f"{digest(receipt_id)}.json"
        record = {"schema_version": 1, "kind": kind, "attempt_id": grant["attempt_id"],
                  "run_identity_sha256": grant["run_identity_sha256"],
                  "generation": grant["generation"], "chunk_id": grant["chunk_id"],
                  "slot": grant["slot"], "payload": payload}
        try:
            publish(path, record)
        except FileExistsError:
            if read_json(path) != record:
                raise ValueError("receipt已存在且内容不符") from None
        return path


class Controller:
    """唯一宿主的进程生命周期锁；恢复证明由资源owner提供，TTL无效。"""
    def __init__(self, store, *, readonly=False):
        self.store = store
        self.root = store.root
        self.lock = None
        self.readonly = readonly

    def __enter__(self):
        self.config = read_json(self.root / "inputs/queue.json")
        if self.config["controller_host"] != socket.gethostname():
            raise RuntimeError("控制器宿主不符，换主必须先确认旧主退出")
        self.lock = (self.root / "queue/controller.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            raise RuntimeError("已有控制器持锁") from None
        self.manifest = read_json(self.root / "inputs/episodes.json")
        self.episodes = {e["episode_key"]: e for e in self.manifest["episodes"]}
        self.state = read_json(self.root / "queue/state.json")
        # owner完整而快照未写的窗口可从controller原始grant重建。
        ownerless = False
        for directory in sorted((self.root / "queue/claims").iterdir()):
            owner = directory / "owner.json"
            if not owner.exists():
                ownerless = directory.name not in self.state.get("recovered_ownerless", []) or ownerless
                continue
            grant = read_json(owner)
            self.state["assignments"].setdefault(grant["attempt_id"], {"grant": grant, "closed": False})
            self.state["requests"].setdefault(grant["claim_request_id"], {"status": "GRANTED", "grant": grant})
        self._rebuild_history()
        if self.state["phase"] == "RECOVERING" and self.state["freeze"].get("pending") and not self.readonly:
            self._finish_freeze()
        if ownerless and self.state["phase"] != "RECOVERING" and not self.readonly:
            self.begin_recovery("ownerless")
        if not self.readonly:
            self._save()
        return self

    def __exit__(self, *unused):
        if self.lock is not None:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None

    def _assert_lock(self):
        if self.lock is None:
            raise RuntimeError("必须在Controller上下文中操作")

    def _save(self):
        self._assert_lock()
        if self.readonly:
            raise RuntimeError("只读控制器禁止写入")
        replace_json(self.root / "queue/state.json", self.state)

    def authorize_slot(self, slot):
        self._assert_lock()
        if self.state["phase"] == "RECOVERING":
            raise RuntimeError("恢复期间禁止launch")
        fields = SLOT_FIELDS[:-1]
        if any(not slot.get(f) for f in fields):
            raise ValueError("launch permit身份不完整")
        target = self.root / "inputs/slot_permits" / f"{digest(slot['slot_instance'])}.json"
        if target.exists():
            if read_json(target) != slot:
                raise ValueError("launch permit不可覆盖")
        else:
            publish(target, slot)

    def _rebuild_history(self):
        # U以控制器不可变历史为权威，重试success不会清除。
        uncertain = set(self.state["uncertain"])
        exhausted = set(self.state["exhausted"])
        retries = dict(self.state["retries"])
        for path in (self.root / "history").glob("*.json"):
            event = read_json(path)
            uncertain.update(event.get("uncertain", []))
            exhausted.update(event.get("exhausted", []))
            for key, count in event.get("retries", {}).items():
                retries[key] = max(retries.get(key, 0), count)
            if event.get("kind") == "handback_close":
                if event["attempt_id"] in self.state["assignments"]:
                    self.state["assignments"][event["attempt_id"]]["closed"] = True
                chunk_id = event["chunk_id"]
                self.state["generations"][chunk_id] = max(self.state["generations"].get(chunk_id, 0), event["generation"])
        self.state["uncertain"] = sorted(uncertain)
        self.state["retries"] = retries
        self.state["exhausted"] = sorted(exhausted)

    def register_slot(self, slot):
        self._assert_lock()
        if self.state["phase"] == "RECOVERING":
            raise RuntimeError("恢复期间membership冻结")
        if any(not slot.get(field) for field in SLOT_FIELDS):
            raise ValueError("slot身份不完整")
        key = slot["slot_instance"]
        if key in self.state["slots"]:
            if any(self.state["slots"][key][f] != slot[f] for f in SLOT_FIELDS):
                raise ValueError("slot incarnation不可复用")
            return
        self.state["slots"][key] = deepcopy(slot)
        self.state["revision"] += 1
        self._save()

    def set_slot_state(self, slot_instance, state):
        self._assert_lock()
        if self.state["phase"] == "RECOVERING":
            raise RuntimeError("恢复期间membership冻结")
        self.state["slots"][slot_instance]["state"] = state
        self.state["revision"] += 1
        self._save()

    def request_resize(self, desired_jobs, expected_revision):
        if expected_revision != self.state["revision"] or type(desired_jobs) is not int or desired_jobs < 0:
            raise ValueError("pool revision或目标容量不符")
        if self.state["phase"] == "RECOVERING":
            self.state["resize_requests"].append({"desired_jobs": desired_jobs, "expected_revision": expected_revision})
            self._save()
            return {"status": "QUEUED"}
        self.state["desired_jobs"] = desired_jobs
        self.state["revision"] += 1
        self._save()
        return {"status": "APPLIED", "revision": self.state["revision"]}

    def _accepted(self):
        return {record["episode_key"]: record for record in
                (read_json(p) for p in (self.root / "accepted").glob("*.json"))}

    def claim(self, claim_request_id, slot_instance):
        self._assert_lock()
        if self.config["mode"] not in {"production", "validation"}:
            raise RuntimeError("bootstrap禁止claim与canonical")
        if claim_request_id in self.state["requests"]:
            return deepcopy(self.state["requests"][claim_request_id])
        slot = self.state["slots"][slot_instance]
        if self.state["phase"] == "RECOVERING":
            return {"status": "WAIT", "reason": "RECOVERING"}
        if self.state.get("blocked_reason"):
            return {"status": "STOPPED_ERROR", "reason": self.state["blocked_reason"]}
        ledger_path = self.root / "pool/ledger.json"
        if ledger_path.exists():
            ledger = read_json(ledger_path)
            member = ledger["slots"].get(slot_instance)
            if (ledger["state"] != "ACTIVE" or member is None
                    or member.get("state") not in {"STARTING", "RUNNING"} or member.get("retiring")
                    or any(member.get(f) != slot[f] for f in SLOT_FIELDS[:-1])):
                return {"status": "DRAIN"}
        if slot.get("state") != "RUNNING" or self.state.get("desired_jobs", 1) == 0:
            return {"status": "DRAIN"}
        for assignment in self.state["assignments"].values():
            if not assignment["closed"] and assignment["grant"]["slot"]["slot_instance"] == slot_instance:
                raise RuntimeError("slot已有未关闭assignment，必须query原grant")
        accepted = self._accepted()
        occupied = {g["chunk_id"] for a in self.state["assignments"].values()
                    if not a["closed"] for g in [a["grant"]]}
        for chunk in self.manifest["chunks"]:
            chunk_id = chunk["chunk_id"]
            members = [self.episodes[k] for k in chunk["episode_keys"]
                       if k not in accepted and k not in self.state["exhausted"]]
            if not members or chunk_id in occupied:
                continue
            generation = self.state["generations"].get(chunk_id, 0)
            directory = self.root / "queue/claims" / f"{chunk_id}-{generation}"
            try:
                directory.mkdir()
            except FileExistsError:
                self.begin_recovery("既有claim未闭合")
                return {"status": "WAIT", "reason": "RECOVERING"}
            attempt_id = digest([chunk_id, generation, claim_request_id, slot])
            grant = {"schema_version": 1, "run_identity_sha256": self.config["run_identity_sha256"],
                     "chunk_id": chunk_id, "attempt_id": attempt_id, "generation": generation,
                     "claim_request_id": claim_request_id, "slot": deepcopy(slot),
                     "episodes": deepcopy(members), "attempt_dir": str(self.root / "attempts" / attempt_id)}
            publish(directory / "owner.json", grant)
            (self.root / "attempts" / attempt_id).mkdir(parents=True, exist_ok=True)
            self.state["assignments"][attempt_id] = {"grant": grant, "closed": False}
            response = {"status": "GRANTED", "grant": grant}
            self.state["requests"][claim_request_id] = response
            self._save()
            return deepcopy(response)
        return {"status": self.summary()["compute_status"]}

    def _quarantine(self, path, reason):
        publish(self.root / "quarantine" / f"{uuid.uuid4().hex}.json",
                {"source": str(path), "reason": reason})
        return False

    def _record(self, path, *, allow_stale=False):
        original = Path(path)
        path = original.resolve()
        if not path.is_relative_to(self.root / "attempts") or original.is_symlink():
            raise ValueError("receipt越界")
        record = read_json(path)
        if not isinstance(record, dict) or record.get("schema_version") != 1 or not isinstance(record.get("payload"), dict):
            raise ValueError("receipt schema不符")
        assignment = self.state["assignments"].get(record["attempt_id"])
        if assignment is None:
            raise ValueError("未知attempt")
        grant = assignment["grant"]
        expected_path = self.root / "attempts" / grant["attempt_id"]
        if path.parent.parent != expected_path or record["kind"] != path.parent.name:
            raise ValueError("receipt不属于attempt")
        for field in ("run_identity_sha256", "generation", "chunk_id", "slot"):
            if record[field] != grant[field]:
                raise ValueError(f"receipt身份错误:{field}")
        if not allow_stale and record["generation"] != self.state["generations"].get(grant["chunk_id"], 0):
            raise ValueError("旧generation迟到")
        payload = record["payload"]
        key = payload["episode_key"]
        matches = [e for e in grant["episodes"] if e["episode_key"] == key]
        if len(matches) != 1:
            raise ValueError("episode不在assignment")
        episode = matches[0]
        for field in ("task_name", "episode_id", "env_seed", "difficulty"):
            if payload.get(field) != episode[field]:
                raise ValueError(f"episode身份错误:{field}")
        return record, grant, episode

    def accept_terminal(self, path, *, recovering=False):
        self._assert_lock()
        if self.config["mode"] not in {"production", "validation"}:
            raise RuntimeError("bootstrap禁止canonical")
        if self.state["phase"] == "RECOVERING" and not recovering:
            return False
        try:
            record, grant, episode = self._record(path)
            payload = record["payload"]
            if record["kind"] != "terminal" or payload.get("outcome") not in {"success", "fail", "timeout"}:
                raise ValueError("非正常终局")
            if type(payload.get("actions_executed")) is not int or payload["actions_executed"] < 0:
                raise ValueError("终局缺动作计数")
            if not payload.get("episode_nonce") or type(payload.get("rpc_seq")) is not int or payload["rpc_seq"] < 0:
                raise ValueError("终局缺RPC身份")
            descriptor = {**deepcopy(record), **deepcopy(episode), "source_receipt_sha256": digest(record)}
            target = self.root / "accepted" / f"{digest(episode['episode_key'])}.json"
            if target.exists():
                if read_json(target) == descriptor:
                    return True
                raise ValueError("重复episode接受被拒")
            publish(target, descriptor)
            return True
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            return self._quarantine(path, str(exc))

    def accept_artifact(self, path):
        self._assert_lock()
        try:
            record, grant, episode = self._record(path, allow_stale=True)
            accepted = self._accepted().get(episode["episode_key"])
            if record["kind"] != "artifact" or not accepted or accepted["attempt_id"] != grant["attempt_id"]:
                raise ValueError("产物没有已接受终局")
            receipt = deepcopy(record)
            errors = []
            if not record["payload"].get("files") and not record["payload"].get("artifact_error"):
                raise ValueError("产物清单不能为空")
            for asset in record["payload"].get("files", []):
                target = (self.root / asset["path"]).resolve()
                if not target.is_relative_to(self.root / "attempts" / grant["attempt_id"]):
                    raise ValueError("产物路径越界")
                if target.stat().st_size != asset["bytes"] or hashlib.sha256(target.read_bytes()).hexdigest() != asset["sha256"]:
                    errors.append(asset["path"])
            receipt["artifact_error"] = bool(record["payload"].get("artifact_error") or errors)
            receipt["mismatches"] = errors
            target = self.root / "artifacts" / f"{digest(record)}.json"
            if not target.exists():
                publish(target, receipt)
            return not receipt["artifact_error"]
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            return self._quarantine(path, str(exc))

    def close_assignment(self, attempt_id, *, sim_stopped):
        self._assert_lock()
        if not sim_stopped:
            raise ValueError("必须确认sim停止")
        if self.state["phase"] == "RECOVERING":
            raise ValueError("恢复中须先确认全部冻结实体静止")
        assignment = self.state["assignments"][attempt_id]
        if assignment["closed"]:
            return
        grant = assignment["grant"]
        self._import_client_records(grant)
        directory = self.root / "attempts" / attempt_id
        for path in sorted((directory / "terminal").glob("*.json")):
            self.accept_terminal(path)
        for path in sorted((directory / "artifact").glob("*.json")):
            self.accept_artifact(path)
        accepted = self._accepted()
        missing = {e["episode_key"] for e in grant["episodes"]
                   if e["episode_key"] not in accepted and e["episode_key"] not in self.state["exhausted"]}
        if missing:
            returned, started = set(), set()
            for kind in ("handback", "started", "infra"):
                for path in sorted((directory / kind).glob("*.json")):
                    record, _, episode = self._record(path)
                    if kind == "handback":
                        if record["payload"].get("reason") != "drain":
                            raise ValueError("归还必须明确是未开始drain成员")
                        returned.add(episode["episode_key"])
                    else:
                        started.add(episode["episode_key"])
            if not missing <= returned or missing & started or returned - missing:
                raise ValueError("assignment尚有未封口成员，或归还成员已开始")
            event = {"kind": "handback_close", "attempt_id": attempt_id, "chunk_id": grant["chunk_id"],
                     "generation": grant["generation"] + 1, "returned": sorted(returned)}
            event_path = self.root / "history" / f"close-{attempt_id}.json"
            if not event_path.exists():
                publish(event_path, event)
            self.state["generations"][grant["chunk_id"]] = event["generation"]
        assignment["closed"] = True
        self._save()

    def _import_client_records(self, grant):
        """仅在sim/step已确证静止后补收A原始文件，覆盖G轮询前骤死窗口。"""
        directory = self.root / "attempts" / grant["attempt_id"] / "client"
        if not directory.exists():
            return
        if directory.is_symlink() or directory.resolve() != directory:
            raise ValueError("原始client目录不是本attempt实体目录")
        members = {e["episode_key"]: e for e in grant["episodes"]}
        nonces = {}
        for kind in ("started", "terminal", "infra", "artifact"):
            paths = list(directory.glob(f"*.{kind}.json"))
            if kind == "artifact":
                paths += list(directory.glob("*.close-error.json")) + list(directory.glob("*.artifact-error.json"))
            for path in sorted(paths):
                try:
                    if path.is_symlink() or path.resolve().parent != directory:
                        raise ValueError("原始client文件越界")
                    payload = read_json(path)
                    key = payload.get("episode_key")
                    if kind == "infra" and key is None:
                        # 无episode的启动错误由G正向核step后提交worker_error/worker_infra。
                        continue
                    if key not in members or payload.get("record_type") != kind:
                        raise ValueError("原始client记录类型或episode身份错误")
                    if payload.get("run_identity_sha256") != grant["run_identity_sha256"] or payload.get("chunk_id") != grant["chunk_id"]:
                        raise ValueError("原始client记录跨run/chunk")
                    if any(payload.get(f) != members[key][f] for f in ("task_name", "episode_id", "env_seed", "difficulty")):
                        raise ValueError("原始client记录seed/difficulty错误")
                    nonce = payload.get("episode_nonce")
                    if not isinstance(nonce, str) or not nonce or key in nonces and nonces[key] != nonce:
                        raise ValueError("原始client episode_nonce不一致")
                    nonces[key] = nonce
                    if kind == "artifact":
                        files = []
                        for item in payload.get("files", []):
                            relative = Path(item["path"])
                            if relative.is_absolute() or ".." in relative.parts:
                                raise ValueError("原始artifact路径越界")
                            asset = (directory / relative).resolve()
                            if not asset.is_relative_to(directory):
                                raise ValueError("原始artifact路径越界")
                            files.append({**item, "path": str(asset.relative_to(self.root))})
                        payload = {**payload, "files": files}
                    receipt_id = hashlib.sha256(path.name.encode()).hexdigest()
                    self.store.write_attempt(grant, kind, payload, receipt_id)
                except (ValueError, KeyError, OSError, TypeError, AttributeError) as exc:
                    self._quarantine(path, str(exc))

    def begin_recovery(self, reason):
        self._assert_lock()
        if self.state["phase"] != "RECOVERING":
            self.state["phase"] = "RECOVERING"
            self.state["freeze"] = {"recovery_id": uuid.uuid4().hex, "revision": self.state["revision"], "reason": reason,
                                    "pending": True}
            # 先冻结新事务，再等已经进入的pool事务落盘，最后取完整membership。
            self._save()
            self._finish_freeze()
        return deepcopy(self.state["freeze"])

    def _finish_freeze(self):
        members = deepcopy({k: s for k, s in self.state["slots"].items()
                            if s.get("state") not in TERMINAL_STATES})
        pool_revision = None
        ledger_path = self.root / "pool/ledger.json"
        if ledger_path.exists():
            lock_path = self.root / "pool/controller.lock"
            with lock_path.open("a+") as pool_lock:
                fcntl.flock(pool_lock, fcntl.LOCK_EX)
                ledger = read_json(ledger_path)
                pool_revision = ledger["pool_revision"]
                for key, member in ledger["slots"].items():
                    if member.get("state") not in TERMINAL_STATES:
                        members[key] = {**member, **members.get(key, {})}
                self.state["freeze"].update({"pending": False, "pool_revision": pool_revision, "slots": members})
                self._save()
        else:
            self.state["freeze"].update({"pending": False, "pool_revision": None, "slots": members})
            self._save()

    def recover(self, proofs):
        self._assert_lock()
        freeze = self.state["freeze"]
        if self.state["phase"] != "RECOVERING" or freeze["revision"] != self.state["revision"]:
            raise ValueError("恢复snapshot或revision不符")
        if freeze.get("pool_revision") is not None and read_json(self.root / "pool/ledger.json")["pool_revision"] != freeze["pool_revision"]:
            raise ValueError("恢复期间pool revision发生变化")
        proof_map = {p["slot_instance"]: p for p in proofs}
        if set(proof_map) != set(freeze["slots"]) or len(proof_map) != len(proofs):
            raise ValueError("必须证明全部冻结membership静止")
        for key, slot in freeze["slots"].items():
            proof = proof_map[key]
            if slot.get("step_id") is None and proof.get("step_id") is not None:
                # 尚未注册的launch只可凭精确launch名的正向Slurm事实补绑一次。
                launch = slot.get("launch_id")
                if (key in self.state["slots"] or not launch or proof.get("launch_id") != launch
                        or proof.get("slurm_job_name") != f"official-resident-{launch}"
                        or not str(proof["step_id"]).isdigit()
                        or any(proof.get(f) != slot.get(f) for f in SLOT_FIELDS[:-1])
                        or any(proof.get(f) is not True for f in ("terminal", "rpc_quiescent", "sidecar_quiescent"))):
                    raise ValueError("未注册step缺少精确launch名的正向绑定证明")
                slot["step_id"] = proof["step_id"]
                slot["step_binding"] = {"launch_id": launch, "slurm_job_name": proof["slurm_job_name"],
                                        "step_id": proof["step_id"]}
                self._save()
            if any(proof.get(f) != slot.get(f) for f in SLOT_FIELDS) or any(proof.get(f) is not True for f in
                    ("terminal", "rpc_quiescent", "sidecar_quiescent")):
                raise ValueError("旧step incarnation或执行实体仍不确定")
        transaction_path = self.root / "history" / f"{freeze['recovery_id']}.json"
        if transaction_path.exists():
            return self._apply_recovery(read_json(transaction_path))
        # 所有静止证明齐全之后，先接收尚可接受的旧代正常终局。
        for assignment in self.state["assignments"].values():
            if not assignment["closed"]:
                self._import_client_records(assignment["grant"])
                directory = self.root / "attempts" / assignment["grant"]["attempt_id"]
                for path in sorted((directory / "terminal").glob("*.json")):
                    self.accept_terminal(path, recovering=True)
                for path in sorted((directory / "artifact").glob("*.json")):
                    self.accept_artifact(path)
        accepted = self._accepted()
        event = {"freeze": deepcopy(freeze), "uncertain": [], "retries": {}, "exhausted": [],
                 "closed_assignments": [], "generations": {}, "ownerless": []}
        chunks = set()
        for assignment in self.state["assignments"].values():
            if assignment["closed"]:
                continue
            grant = assignment["grant"]
            directory = self.root / "attempts" / grant["attempt_id"]
            started, errors, handback = set(), {}, set()
            for kind in ("started", "infra", "handback"):
                for path in sorted((directory / kind).glob("*.json")):
                    try:
                        record, _, episode = self._record(path)
                        key = episode["episode_key"]
                        if kind == "started":
                            started.add(key)
                        elif kind == "infra":
                            errors[key] = record["payload"]
                        else:
                            handback.add(key)
                    except (ValueError, KeyError, OSError) as exc:
                        self._quarantine(path, str(exc))
            for episode in grant["episodes"]:
                key = episode["episode_key"]
                if key in accepted:
                    continue
                error = errors.get(key)
                if key in started or error:
                    uncertain = error.get("uncertain_terminal", True) if error else True
                    if uncertain:
                        event["uncertain"].append(key)
                    if (error and error.get("retryable") is False) or self.state["retries"].get(key, 0) >= 1:
                        event["exhausted"].append(key)
                    else:
                        event["retries"][key] = 1
                elif key not in handback:
                    # 无started持久记录的成员按未开始归还；worker必须先记started再执行。
                    pass
            event["closed_assignments"].append(grant["attempt_id"])
            chunks.add(grant["chunk_id"])
        if self.state.get("blocked_reason"):
            event["exhausted"] = sorted(set(self.episodes) - set(accepted))
        # 先持久历史，再换generation；崩溃重放不会洗掉U或retry。
        # ownerless claim也必须换代，不删除原claim审计。
        for path in (self.root / "queue/claims").iterdir():
            if not (path / "owner.json").exists() and path.name not in self.state.get("recovered_ownerless", []):
                chunks.add(path.name.rsplit("-", 1)[0])
                event["ownerless"].append(path.name)
        for chunk in chunks:
            event["generations"][chunk] = self.state["generations"].get(chunk, 0) + 1
        publish(transaction_path, event)
        return self._apply_recovery(event)

    def _apply_recovery(self, event):
        self.state["uncertain"] = sorted(set(self.state["uncertain"]) | set(event["uncertain"]))
        self.state["retries"].update(event["retries"])
        self.state["exhausted"] = sorted(set(self.state["exhausted"]) | set(event["exhausted"]))
        for attempt_id in event["closed_assignments"]:
            self.state["assignments"][attempt_id]["closed"] = True
        self.state["generations"].update(event["generations"])
        self.state["recovered_ownerless"] = sorted(set(self.state.get("recovered_ownerless", [])) | set(event["ownerless"]))
        for key in event["freeze"]["slots"]:
            self.state["slots"][key] = {**event["freeze"]["slots"][key], "state": "TERMINATED"}
        self.state["phase"] = "RUNNING"
        self.state["freeze"] = None
        self.state["revision"] += 1
        self._save()
        return self.summary()

    def process_requests(self):
        self._assert_lock()
        responses = []
        for path in sorted((self.root / "queue/requests").glob("*.json")):
            target_response = self.root / "queue/responses" / path.name
            if target_response.exists():
                continue
            try:
                request = read_json(path)
                rid, operation, payload = request["request_id"], request["operation"], request["payload"]
                if path.stem != self.store.request_name(rid):
                    raise ValueError("请求文件名与编号不符")
            except (ValueError, KeyError, OSError, TypeError) as exc:
                self._quarantine(path, str(exc))
                publish(target_response, {"status": "REJECTED", "error": str(exc)})
                continue
            if self.store.query(rid) is not None:
                continue
            try:
                if operation == "authorize_slot":
                    ledger = read_json(self.root / "pool/ledger.json")
                    slot = payload["slot"]
                    member = ledger["slots"][slot["slot_instance"]]
                    if (ledger["state"] != "ACTIVE" or member["state"] != "STARTING"
                            or member["launch_id"] != payload["launch_id"]
                            or member["run_identity_sha256"] != self.config["run_identity_sha256"]
                            or any(member.get(f) != slot.get(f) for f in SLOT_FIELDS[:-1])):
                        raise ValueError("launch请求与pool账本不符")
                    self.authorize_slot({**slot, "launch_id": payload["launch_id"],
                                         "pool_revision": ledger["pool_revision"]})
                    response = {"status": "AUTHORIZED", "pool_revision": ledger["pool_revision"]}
                elif operation == "register_slot":
                    slot = payload["slot"]
                    permit = read_json(self.root / "inputs/slot_permits" / f"{digest(slot['slot_instance'])}.json")
                    if any(slot.get(f) != permit[f] for f in SLOT_FIELDS[:-1]) or (
                            "step_id" in permit and slot.get("step_id") != permit["step_id"]):
                        raise ValueError("slot不匹配launch permit")
                    if "pool_revision" in permit:
                        ledger = read_json(self.root / "pool/ledger.json")
                        member = ledger["slots"][slot["slot_instance"]]
                        if (ledger["state"] != "ACTIVE" or ledger["pool_revision"] != permit["pool_revision"]
                                or member["launch_id"] != permit["launch_id"]):
                            raise ValueError("launch许可已过期")
                    if slot.get("state") != "RUNNING":
                        raise ValueError("注册时须为RUNNING")
                    self.register_slot(slot)
                    response = {"status": "REGISTERED"}
                elif operation == "claim":
                    response = self.claim(rid, payload["slot_instance"])
                    if response["status"] == "WAIT":
                        continue
                elif operation == "query":
                    response = self.state["requests"].get(payload["claim_request_id"], {"status": "UNKNOWN"})
                elif operation == "terminal":
                    if self.state["phase"] == "RECOVERING":
                        continue
                    response = {"status": "ACCEPTED" if self.accept_terminal(payload["path"]) else "REJECTED"}
                elif operation == "artifact":
                    response = {"status": "ACCEPTED" if self.accept_artifact(payload["path"]) else "REJECTED"}
                elif operation == "worker_infra":
                    known = self.state["slots"].get(payload["slot_instance"])
                    if known and known.get("state") in TERMINAL_STATES:
                        response = {"status": "ALREADY_TERMINATED"}
                        publish(target_response, response)
                        responses.append((rid, response))
                        continue
                    if payload["slot_instance"] not in self.state["slots"]:
                        ledger = read_json(self.root / "pool/ledger.json")
                        member = ledger["slots"].get(payload["slot_instance"])
                        if not member or not member.get("launch_id") or member.get("step_id") is None:
                            raise ValueError("未知或尚未核定精确step的worker错误来源")
                    self.begin_recovery("worker全局基础设施错误:" + str(payload["error"]))
                    response = {"status": "RECOVERING"}
                elif operation == "recover":
                    if self.state["phase"] != "RECOVERING":
                        self.begin_recovery("资源owner核定旧step终态")
                    response = {"status": "RECOVERED", "summary": self.recover(payload["proofs"])}
                elif operation == "worker_error":
                    if payload.get("retryable") is not False or payload.get("uncertain_terminal") is not False:
                        raise ValueError("全局worker_error仅接受明确不可重试且无终局不确定的错误")
                    if payload["slot_instance"] not in self.state["slots"]:
                        raise ValueError("未知worker错误来源")
                    self.state["blocked_reason"] = str(payload["error"])
                    self.begin_recovery("worker全局合同错误")
                    response = {"status": "RECOVERING"}
                elif operation == "infra":
                    self._record(payload["path"])
                    self.begin_recovery("worker基础设施错误")
                    response = {"status": "RECOVERING"}
                elif operation == "close":
                    self.close_assignment(payload["attempt_id"], sim_stopped=payload["sim_stopped"])
                    response = {"status": "CLOSED"}
                else:
                    raise ValueError("未知队列操作")
            except (ValueError, KeyError, RuntimeError, OSError) as exc:
                response = {"status": "REJECTED", "error": str(exc)}
            publish(self.root / "queue/responses" / f"{self.store.request_name(rid)}.json", response)
            responses.append((rid, response))
        return responses

    def summary(self):
        self._assert_lock()
        self._rebuild_history()
        accepted = self._accepted()
        uncertain = set(self.state["uncertain"])
        missing = set(self.episodes) - set(accepted)
        unknown = uncertain | missing
        active = any(not a["closed"] for a in self.state["assignments"].values())
        ready = missing - set(self.state["exhausted"])
        unknown_slots = any(s.get("state") in {"UNKNOWN", "SUBMITTING", "STARTING", "DRAINING", "CANCEL_REQUESTED"}
                            for s in self.state["slots"].values())
        if self.state["phase"] == "RECOVERING" or active or unknown_slots:
            compute = "WAIT"
        elif ready:
            compute = "READY"
        else:
            compute = "STOPPED_ERROR" if missing else "COMPUTE_DONE"
        score = "INVALID" if missing else "QUALIFIED" if uncertain else "PASS"
        certain_success = sum(r["payload"]["outcome"] == "success" for k, r in accepted.items() if k not in unknown)
        good_artifacts = set()
        artifact_errors = {}
        resolved_errors = set()
        for path in (self.root / "artifacts").glob("*.json"):
            record = read_json(path)
            key = record["payload"]["episode_key"]
            valid = not record["artifact_error"]
            for asset in record["payload"].get("files", []):
                target = (self.root / asset["path"]).resolve()
                try:
                    valid = valid and target.stat().st_size == asset["bytes"] and hashlib.sha256(target.read_bytes()).hexdigest() == asset["sha256"]
                except OSError:
                    valid = False
            if valid:
                good_artifacts.add(key)
                for previous in record["payload"].get("resolves", []):
                    old_path = self.root / "artifacts" / f"{previous}.json"
                    if (len(previous) == 64 and all(c in "0123456789abcdef" for c in previous)
                            and old_path.exists() and read_json(old_path)["payload"]["episode_key"] == key):
                        resolved_errors.add(previous)
            else:
                artifact_errors[path.stem] = key
        unresolved_artifacts = set(artifact_errors) - resolved_errors
        bad_episodes = {artifact_errors[key] for key in unresolved_artifacts}
        good_artifacts -= bad_episodes
        total = len(self.episodes)
        return {"purpose": self.config["mode"], "compute_status": compute, "score_status": score,
                "artifact_status": "PASS" if len(good_artifacts) == total else "BLOCKED_ARTIFACTS",
                "execution_coverage": len(accepted), "artifact_coverage": len(good_artifacts),
                "artifact_errors": sorted(unresolved_artifacts),
                "execution_errors": len(missing & set(self.state["exhausted"])),
                "missing": sorted(missing), "uncertain_episodes": sorted(uncertain), "U": len(uncertain),
                "unknown_episodes": sorted(unknown), "K": len(unknown),
                "success_certain": certain_success,
                "diagnostic_interval": [certain_success / total, (certain_success + len(unknown)) / total],
                "episodes": [{**deepcopy(e), "execution": deepcopy(accepted.get(k)),
                              "had_uncertain_terminal": k in uncertain,
                              "exhausted_error": k in self.state["exhausted"]}
                             for k, e in self.episodes.items()]}
