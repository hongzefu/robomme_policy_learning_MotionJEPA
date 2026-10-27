"""持久占位资源账本；默认只展示，提交与计算入口分离。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import re
import shlex
from zoneinfo import ZoneInfo

ACTIVE = {"SUBMITTING", "UNKNOWN", "SUBMITTED", "STARTING", "RUNNING", "DRAINING", "CANCEL_REQUESTED"}
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
SUBMIT_GATES = {"go", "environment_a", "ssh_controlmaster", "nfs_paths", "nfs_python",
                "resources_approved", "schedule_identity", "eta_ready_due"}


class SlurmTransport:
    """显式选择GL本地或既有SSH连接，不回退、不转发控制主机环境。"""

    def __init__(self, ssh_host=None, *, runner=None, popen=None):
        if ssh_host is not None and (not re.fullmatch(r"[A-Za-z0-9_@.-]+", ssh_host) or ssh_host.startswith("-")):
            raise ValueError("SSH宿主必须是明确的已有别名或域名")
        self.ssh_host = ssh_host
        self.runner = subprocess.run if runner is None else runner
        self.process = subprocess.Popen if popen is None else popen

    def command(self, argv):
        if self.ssh_host is None:
            return list(argv)
        if not argv or argv[0] not in {"sbatch", "srun", "sacct", "scancel", "squeue"}:
            raise ValueError("SSH传输仅执行明确的Slurm argv")
        remote = ["/usr/bin/env", "TZ=America/New_York", *argv] if argv[0] == "sacct" else list(argv)
        return ["ssh", "-o", "BatchMode=yes", self.ssh_host, shlex.join(remote)]

    def _kwargs(self, kwargs):
        kwargs = dict(kwargs)
        if self.ssh_host is not None:
            kwargs.pop("env", None)  # sacct时区已在远端命令显式设置，其余不发整份Aspen环境。
        return kwargs

    def run(self, argv, **kwargs):
        return self.runner(self.command(argv), **self._kwargs(kwargs))

    def popen(self, argv, **kwargs):
        return self.process(self.command(argv), **self._kwargs(kwargs))


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, payload):
    """先持久化，再替换；预留不能落盘时不允许执行sbatch。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("x") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def require_gates(report, group, minimum=()):
    """消费分组证据，不把成绩或产物要求混入资源门。"""
    required = set(report.get("required_gates", {}).get(group, [])) | set(minimum)
    if not required:
        raise ValueError(f"缺少{group}门定义")
    gates = report.get("gates", {})
    missing = [name for name in sorted(required)
               if not isinstance(gates.get(name), dict) or gates[name].get("status") != "PASS"
               or gates[name].get("mock", False) or not gates[name].get("evidence")]
    if missing:
        raise ValueError(f"{group}门未通过：{','.join(missing)}")


def hold_command(root, ledger, slot):
    env = ledger["envelope"]
    return ["sbatch", "--parsable", f"--account={env['account']}", f"--partition={env['partition']}",
            "--nodes=1", "--ntasks-per-node=1", "--gres=gpu:1", "--gpu_cmode=shared",
            f"--cpus-per-task={env['cpus']}", f"--mem={env['mem']}", "--time=48:00:00",
            f"--job-name=official-stage2-{slot['slot_instance']}",
            f"--comment=pool={ledger['pool_id']},slot={slot['slot_instance']},submission={slot['submission_id']}",
            f"--output={Path(root)/'logs'/'%x-%j.log'}", "--wrap=sleep infinity"]


def committed_gpu_hours(ledger):
    if "holds" in ledger:
        return sum(float(hold.get("actual_gpu_hours", 0)) if hold.get("settled") else 48.0
                   for hold in ledger["holds"].values())
    return sum(float(s.get("actual_gpu_hours", 0)) if s.get("settled") else 48.0
               for s in ledger["slots"].values())


def active_holds(ledger):
    return {slot.get("hold_id", slot["slot_instance"]) for slot in ledger["slots"].values() if slot["state"] in ACTIVE}


def covered_holds(ledger):
    # 自然到期仍占本批已创建容量，只有明确缩容且确认终态的hold可在再次扩容时替换。
    return {slot.get("hold_id", slot["slot_instance"]) for slot in ledger["slots"].values()
            if slot["state"] != "RETIRED" and not (slot.get("retiring") and slot.get("terminal_verified"))}


def slurm_timestamp(value):
    if isinstance(value, (int, float)):
        return float(value)
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=ZoneInfo("America/New_York")).timestamp() if parsed.tzinfo is None else parsed.timestamp()


def slurm_records(job_id, runner=subprocess.run):
    """只在状态事件或低频对账时查指定JobID；查询失败保持未知。"""
    fields = ["JobIDRaw", "Cluster", "User", "Account", "Comment", "Submit", "Start", "End", "State", "JobName", "ElapsedRaw", "ExitCode"]
    command = ["sacct", "--noheader", "--parsable2", f"--jobs={job_id}", "--format=" + ",".join(fields)]
    try:
        result = runner(command, capture_output=True, text=True, timeout=20, check=False,
                        env=dict(os.environ, TZ="America/New_York"))
        if result.returncode != 0:
            return []
        records = []
        for line in result.stdout.splitlines():
            values = line.split("|")
            if len(values) == len(fields) + 1 and values[-1] == "":
                values.pop()
            if len(values) != len(fields):
                return []
            records.append(dict(zip(fields, values)))
        return records
    except (OSError, subprocess.TimeoutExpired):
        return []


def verify_job_record(ledger, slot, records):
    rows = [row for row in records if row["JobIDRaw"] == str(slot["job_id"])]
    if len(rows) != 1:
        return None
    row = rows[0]
    origin = slot.get("adoption_proof", slot)
    try:
        if (row["Cluster"] != ledger["envelope"]["cluster"] or row["User"] != ledger["envelope"]["user"]
                or row["Account"] != ledger["envelope"]["account"]
                or slurm_timestamp(row["Submit"]) != slurm_timestamp(origin["submit_time"])):
            return None
        expected_start = slot.get("start_time") or origin.get("start_time")
        if expected_start not in {None, "Unknown", "N/A", ""}:
            if slurm_timestamp(row["Start"]) != slurm_timestamp(expected_start):
                return None
        # 从未开始的提交以Submit及标签识别；Start=Unknown不转换为虚构时刻。
        # 一旦账本已记录真实Start，上面的严格比对永不放宽。
    except (KeyError, TypeError, ValueError):
        return None
    if not slot.get("adoption_proof"):
        expected = f"pool={ledger['pool_id']},slot={slot.get('original_slot', slot['slot_instance'])},submission={slot['submission_id']}"
        if row["Comment"] != expected:
            return None
    return row


def terminal_step_proof(ledger, slot, queue_slot=None, *, runner=subprocess.run):
    records = slurm_records(slot["job_id"], runner)
    if verify_job_record(ledger, slot, records) is None:
        return None
    step_id = (queue_slot or {}).get("step_id") or slot.get("step_id")
    if step_id is not None:
        rows = [row for row in records if row["JobIDRaw"] == f"{slot['job_id']}.{step_id}"]
    else:
        rows = [row for row in records if row["JobName"] == f"official-resident-{slot.get('launch_id')}"
                and row["JobIDRaw"].startswith(str(slot["job_id"]) + ".")]
    if len(rows) != 1 or rows[0]["State"].split()[0].rstrip("+") not in TERMINAL:
        return None
    step_id = rows[0]["JobIDRaw"].split(".", 1)[1]
    return {**{key: slot[key] for key in ("slot_instance", "worker_uuid", "job_id", "job_incarnation")},
            "step_id": step_id, "terminal": True, "rpc_quiescent": True, "sidecar_quiescent": True,
            "launch_id": slot.get("launch_id"), "slurm_job_name": rows[0]["JobName"],
            "slurm_exit_code": rows[0]["ExitCode"],
            "slurm_state": rows[0]["State"], "slurm_start": rows[0]["Start"], "slurm_end": rows[0]["End"]}


def matches_proof(ledger, slot, proof):
    """JobID不是身份；取消和对账必须同时核提交标签与incarnation。"""
    keys = ("slot_instance", "submission_id", "job_id")
    return (all(str(proof.get(k, "")) == str(slot.get(k, "")) and slot.get(k) for k in keys)
            and str(slot.get("job_id", "")).isdecimal()
            and proof.get("pool_id") == ledger["pool_id"]
            and proof.get("account") == ledger["envelope"]["account"]
            and proof.get("cluster") == ledger["envelope"]["cluster"]
            and proof.get("user") == ledger["envelope"]["user"]
            and bool(proof.get("submit_time")) and bool(proof.get("job_incarnation"))
            and (not slot.get("job_incarnation") or proof["job_incarnation"] == slot["job_incarnation"]))


class Pool:
    """控制器调用的串行资源事务；worker不接触本类。"""

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "pool" / "ledger.json"

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with (self.path.parent / "controller.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = read_json(self.path)
            yield ledger
            atomic_json(self.path, ledger)

    def initialize(self, *, pool_id, schedule_id, schedule_identity_sha256, envelope, count):
        if not pool_id or not schedule_id or not re.fullmatch(r"[0-9a-f]{64}", schedule_identity_sha256):
            raise ValueError("pool、schedule及其身份必须固定")
        self.validate_envelope(envelope, count)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with (self.path.parent / "controller.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                raise ValueError("既有pool不得重新初始化")
            ledger = {"schema_version": 1, "pool_id": pool_id, "schedule_id": schedule_id,
                      "schedule_identity_sha256": schedule_identity_sha256,
                      "descriptors": [{"revision": 0, "schedule_identity_sha256": schedule_identity_sha256}],
                      "pool_revision": 0, "desired_jobs": count, "state": "ACTIVE", "envelope": envelope,
                      "slots": {}, "holds": {}, "history": [], "no_auto_renew": True}
            atomic_json(self.path, ledger)
        return ledger

    @staticmethod
    def validate_envelope(envelope, count):
        if type(count) is not int or count < 0 or count > int(envelope["max_jobs"]):
            raise ValueError("目标N越出批准包络")
        if int(envelope.get("hours", 48)) != 48 or int(envelope["cpus"]) < 1:
            raise ValueError("仅允许固定48h单GPU规格")
        if (not math.isfinite(float(envelope["gpu_hour_limit"])) or float(envelope["gpu_hour_limit"]) < 0
                or not math.isfinite(float(envelope["workload_deadline"]))):
            raise ValueError("须明确GPUh和时间包络")
        if any(not envelope.get(key) for key in ("account", "partition", "cluster", "user", "mem")):
            raise ValueError("资源身份与规格不完整")
        if envelope.get("hard_release_deadline"):
            raise ValueError("本入口未实现经现场核验的站点硬截止机制，不能承诺硬释放")

    @staticmethod
    def writable(ledger, expected_revision):
        if ledger["state"] != "ACTIVE" or ledger["pool_revision"] != expected_revision:
            raise ValueError("pool冻结或revision冲突")

    def guard_queue(self):
        state = self.root/"queue/state.json"
        if state.exists() and read_json(state).get("phase") == "RECOVERING":
            raise ValueError("队列恢复冻结期间禁止提交/伸缩/启动")

    def revise(self, schedule_sha, expected_revision):
        with self.transaction() as ledger:
            self.guard_queue()
            self.writable(ledger, expected_revision)
            if not re.fullmatch(r"[0-9a-f]{64}", schedule_sha):
                raise ValueError("schedule身份无效")
            ledger["pool_revision"] += 1
            ledger["schedule_identity_sha256"] = schedule_sha
            ledger["descriptors"].append({"revision": ledger["pool_revision"], "schedule_identity_sha256": schedule_sha})
        return ledger

    def scale(self, target, expected_revision, reason):
        with self.transaction() as ledger:
            self.guard_queue()
            self.writable(ledger, expected_revision)
            self.validate_envelope(ledger["envelope"], target)
            old = ledger["desired_jobs"]
            ledger["desired_jobs"] = target
            ledger["pool_revision"] += 1
            candidates = [s for s in ledger["slots"].values() if s["state"] in ACTIVE and not s.get("retiring")]
            for slot in sorted(candidates, key=lambda s: s["slot_instance"], reverse=True)[:max(0, len(candidates)-target)]:
                slot["retiring"] = True
                slot["state_before_drain"] = slot["state"]
                slot["state"] = "DRAINING"
                atomic_json(self.root / "pool" / "drain" / (slot["slot_instance"] + ".json"),
                            {"reason": reason, "pool_revision": ledger["pool_revision"]})
            ledger["history"].append({"operation": "scale", "before": old, "after": target,
                                      "reason": reason, "revision": ledger["pool_revision"]})
        return ledger

    def submit(self, gates, expected_revision, *, execute=False, runner=subprocess.run, now=None):
        now = time.time() if now is None else now
        commands = []
        with self.transaction() as ledger:
            self.guard_queue()
            self.writable(ledger, expected_revision)
            require_gates(gates, "submit", SUBMIT_GATES)
            if now >= float(ledger["envelope"]["workload_deadline"]):
                raise ValueError("工作负载截止已到，不自动续批")
            # 终态席位保留为本campaign已消耗容量，TIMEOUT不能触发自动补交。
            assigned = len(covered_holds(ledger))
            needed = max(0, ledger["desired_jobs"] - assigned)
            active = len(active_holds(ledger))
            if active + needed > ledger["envelope"]["max_jobs"]:
                raise ValueError("潜在占用超过批准席位")
            if committed_gpu_hours(ledger) + 48 * needed > ledger["envelope"]["gpu_hour_limit"]:
                raise ValueError("GPUh承诺余额不足")
            for _ in range(needed):
                slot_id = uuid.uuid4().hex
                slot = {"slot_instance": slot_id, "submission_id": uuid.uuid4().hex,
                        "hold_id": slot_id,
                        "state": "SUBMITTING", "reserved_gpu_hours": 48, "settled": False,
                        "created_at": now, "pool_revision": expected_revision, "job_id": None}
                command = hold_command(self.root, ledger, slot)
                commands.append(command)
                if not execute:
                    continue
                ledger["slots"][slot_id] = slot
                ledger["holds"][slot_id] = {"hold_id": slot_id, "settled": False, "reserved_gpu_hours": 48}
                atomic_json(self.path, ledger)
                try:
                    result = runner(command, capture_output=True, text=True, timeout=30, check=False)
                    output = result.stdout.strip().split(";", 1)[0]
                    if result.returncode != 0 or not output.isdecimal():
                        raise RuntimeError(f"sbatch结果不明：{result.stdout} {result.stderr}")
                    slot["job_id"] = output
                    slot["state"] = "SUBMITTED"
                except Exception as error:
                    slot["state"] = "UNKNOWN"
                    slot["submission_error"] = str(error)
                    atomic_json(self.path, ledger)
                    break
                atomic_json(self.path, ledger)
        return commands

    def reconcile(self, slot_instance, proof):
        with self.transaction() as ledger:
            slot = ledger["slots"][slot_instance]
            # 查询零条不构成未提交证据；仅接受完整的正向Slurm身份。
            candidate = dict(slot)
            if not candidate.get("job_id"):
                candidate["job_id"] = str(proof.get("job_id", ""))
            if not matches_proof(ledger, candidate, proof):
                raise ValueError("Slurm提交对账身份不完整或不匹配")
            slot.update({key: proof[key] for key in ("job_id", "job_incarnation", "cluster", "user", "submit_time")})
            slot["slurm_state"] = proof["state"]
            slot["step_id"] = proof.get("step_id")
            slot["start_time"] = slurm_timestamp(proof["start_time"]) if proof.get("start_time") not in {None, "Unknown", "N/A", ""} else None
            slot["end_time"] = slurm_timestamp(proof["end_time"]) if proof.get("end_time") not in {None, "Unknown", "N/A", ""} else None
            slot["submission_comment"] = proof.get("submission_comment")
            if proof["state"] in TERMINAL:
                slot["state"] = "TERMINAL"
                slot["terminal_verified"] = True
            elif not slot.get("retiring"):
                slot["state"] = "RUNNING" if proof["state"] == "RUNNING" else "SUBMITTED"
        return ledger

    def settle(self, slot_instance, actual_gpu_hours):
        with self.transaction() as ledger:
            slot = ledger["slots"][slot_instance]
            if not slot.get("terminal_verified") or not math.isfinite(actual_gpu_hours) or actual_gpu_hours < 0 or actual_gpu_hours > 48:
                raise ValueError("仅终态与实际占用确认后可结算")
            slot["settled"] = True
            slot["actual_gpu_hours"] = actual_gpu_hours
            ledger["holds"][slot.get("hold_id", slot_instance)].update(settled=True, actual_gpu_hours=actual_gpu_hours)
        return ledger

    def adopt(self, proof, expected_revision, authorization):
        """复用用户逐个批准的既有hold；不提交新job，也不重置预算。"""
        if not authorization or proof.get("state") not in {"RUNNING", "PENDING"}:
            raise ValueError("复用须有明确授权与现场Slurm状态")
        with self.transaction() as ledger:
            self.guard_queue()
            self.writable(ledger, expected_revision)
            if any(str(slot.get("job_id")) == str(proof.get("job_id")) for slot in ledger["slots"].values()):
                raise ValueError("同一JobID已在本pool登记，不能重复计为新席位")
            if (len(active_holds(ledger)) + 1 > ledger["envelope"]["max_jobs"]
                    or committed_gpu_hours(ledger) + 48 > ledger["envelope"]["gpu_hour_limit"]):
                raise ValueError("复用仍受席位与48GPUh承诺上限约束")
            slot = {key: proof[key] for key in ("slot_instance", "submission_id", "job_id", "job_incarnation")}
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", slot["slot_instance"]):
                raise ValueError("slot_instance必须是单个安全标识符")
            if not matches_proof(ledger, slot, proof) or proof.get("hours") != 48 or proof.get("gpus") != 1:
                raise ValueError("既有hold身份或1GPU/48h规格不符")
            if proof["state"] == "RUNNING" and (not proof.get("start_time") or not proof.get("end_time")):
                raise ValueError("复用必须现场核独立StartTime/EndTime")
            slot.update({"state": "RUNNING" if proof["state"] == "RUNNING" else "SUBMITTED",
                         "hold_id": slot["slot_instance"],
                         "adoption_authorization": authorization, "adoption_proof": proof,
                         "reserved_gpu_hours": 48, "settled": False,
                         "start_time": slurm_timestamp(proof["start_time"]) if proof.get("start_time") not in {None, "Unknown", "N/A", ""} else None,
                         "end_time": slurm_timestamp(proof["end_time"]) if proof.get("end_time") not in {None, "Unknown", "N/A", ""} else None,
                         "pool_revision": expected_revision})
            ledger["slots"][slot["slot_instance"]] = slot
            ledger["holds"][slot["hold_id"]] = {"hold_id": slot["hold_id"], "settled": False, "reserved_gpu_hours": 48}
        return ledger

    def restart(self, slot_instance, proof, expected_revision):
        """只在C已确认旧执行静止后，在同一hold中创建全新的执行slot。"""
        with self.transaction() as ledger:
            self.guard_queue()
            self.writable(ledger, expected_revision)
            old = ledger["slots"][slot_instance]
            queue_slot = read_json(self.root/"queue/state.json")["slots"][slot_instance]
            fields = ("slot_instance", "worker_uuid", "job_id", "job_incarnation", "step_id")
            if (queue_slot.get("state") != "TERMINATED" or any(proof.get(key) != queue_slot.get(key) for key in fields)
                    or any(proof.get(key) is not True for key in ("terminal", "rpc_quiescent", "sidecar_quiescent"))):
                raise ValueError("旧step/RPC/sidecar静止证明未由controller接受")
            if old.get("retiring") or old["state"] == "TERMINAL" or old.get("replacement_slot"):
                raise ValueError("已释放/缩容/已替换的执行不得重启")
            if time.time() >= min(float(old["end_time"]), float(ledger["envelope"]["workload_deadline"])):
                raise ValueError("原hold预算已到，不自动续48h")
            new_id = uuid.uuid4().hex
            replacement = {key: old[key] for key in ("hold_id", "submission_id", "job_id", "job_incarnation", "start_time", "end_time")}
            for key in ("adoption_proof", "submit_time", "cluster", "user", "submission_comment"):
                if key in old:
                    replacement[key] = old[key]
            replacement.update(slot_instance=new_id, state="RUNNING", previous_slot=slot_instance,
                               original_slot=old.get("original_slot", slot_instance), settled=False, pool_revision=expected_revision)
            old["state"] = "RETIRED"
            old["replacement_slot"] = new_id
            ledger["slots"][new_id] = replacement
        return replacement

    def note_terminal_step(self, slot_instance, proof):
        with self.transaction() as ledger:
            slot = ledger["slots"][slot_instance]
            if any(proof.get(key) != slot.get(key) for key in ("slot_instance", "worker_uuid", "job_id", "job_incarnation")):
                raise ValueError("step事实与已登记slot身份不符")
            if not all(proof.get(key) is True for key in ("terminal", "rpc_quiescent", "sidecar_quiescent")):
                raise ValueError("step未静止")
            slot["step_id"] = proof["step_id"]
            slot["step_terminal_proof"] = proof

    def handoff_smoke(self, proofs, *, parent_run_name, scope_manifest_sha256):
        """同一campaign的阶段移交：保留hold及剩余原walltime，不新提交或续期。"""
        with self.transaction() as ledger:
            by_slot = {item["slot_instance"]: item for item in proofs}
            launched = {key: slot for key, slot in ledger["slots"].items() if slot["state"] in ACTIVE and slot.get("launch_id")}
            if set(launched) != set(by_slot):
                raise ValueError("smoke移交必须核全部已启动resident静止")
            for key, slot in launched.items():
                proof = by_slot[key]
                if (any(proof.get(name) != slot.get(name) for name in ("slot_instance", "worker_uuid", "job_id", "job_incarnation"))
                        or any(proof.get(name) is not True for name in ("terminal", "rpc_quiescent", "sidecar_quiescent"))):
                    raise ValueError("smoke移交的step/RPC/sidecar未确认静止")
                new_id = uuid.uuid4().hex
                member = {name: slot[name] for name in ("hold_id", "submission_id", "job_id", "job_incarnation", "start_time", "end_time")}
                for name in ("adoption_proof", "submit_time", "cluster", "user", "gpu_uuid"):
                    if name in slot:
                        member[name] = slot[name]
                member.update(slot_instance=new_id, state="RUNNING", previous_slot=key,
                              original_slot=slot.get("original_slot", key), pool_revision=ledger["pool_revision"], settled=False)
                slot.update(state="RETIRED", replacement_slot=new_id, step_terminal_proof=proof)
                ledger["slots"][new_id] = member
            ledger["campaign"] = {"parent_run_name": parent_run_name, "current_stage": "smoke1",
                                  "stage_state": "HANDOFF_READY", "next_stage": "shards10",
                                  "scope_manifest_sha256": scope_manifest_sha256, "proofs": proofs}
        return ledger

    def cancel(self, proofs, *, confirmed=False, runner=subprocess.run, audit_path=None, verified_snapshot=None,
               verify_current=True):
        """先取核验后的本地快照，取消不依赖成绩、产物或NFS后续写成功。"""
        ledger = verified_snapshot if verified_snapshot is not None else read_json(self.path)
        verified = []
        for proof in proofs:
            slot = ledger["slots"].get(proof.get("slot_instance"))
            if slot is None or not matches_proof(ledger, slot, proof):
                raise ValueError("禁止取消身份不明或其他pool的JobID")
            if slot["state"] != "TERMINAL":
                verified.append(copy.deepcopy(slot))
        commands = [["scancel", job_id] for job_id in dict.fromkeys(str(slot["job_id"]) for slot in verified)]
        if not confirmed:
            return commands
        if audit_path is None:
            raise ValueError("取消前必须指定本机审计副本")
        atomic_json(audit_path, {"pool_id": ledger["pool_id"], "slots": verified, "commands": commands})
        results = []
        for command in commands:
            if verify_current:
                slot = next(item for item in verified if str(item["job_id"]) == command[1])
                if verify_job_record(ledger, slot, slurm_records(command[1], runner)) is None:
                    raise RuntimeError(f"取消前Slurm身份无法现场复核：JobID={command[1]}")
            result = runner(command, capture_output=True, text=True, timeout=30, check=False)
            results.append({"command": command, "returncode": result.returncode, "stderr": result.stderr})
        atomic_json(audit_path, {"pool_id": ledger["pool_id"], "slots": verified, "results": results,
                                 "resource_release": "PENDING_TERMINAL_CONFIRMATION"})
        try:
            with self.transaction() as current:
                for slot in verified:
                    current["slots"][slot["slot_instance"]]["state"] = "CANCEL_REQUESTED"
        except OSError:
            pass  # 本机审计已有记录；恢复NFS后补写，不阻塞GPU释放请求。
        return results

    def confirm_release(self, snapshot, audit_path, *, runner=subprocess.run, timeout_seconds=120, poll_seconds=5):
        """取消请求后以Slurm正向终态确认；无法核实就保留承诺并报告pending。"""
        end = time.monotonic()+timeout_seconds
        finished = {}
        slots = {str(slot["job_id"]): slot for slot in snapshot["slots"].values()
                 if slot.get("job_id") and slot["state"] in ACTIVE}
        while True:
            for job_id, slot in slots.items():
                if job_id in finished:
                    continue
                row = verify_job_record(snapshot, slot, slurm_records(job_id, runner))
                if row is None or row["State"].split()[0].rstrip("+") not in TERMINAL:
                    continue
                try:
                    seconds = float(row["ElapsedRaw"])
                    if not math.isfinite(seconds) or not 0 <= seconds <= 172800:
                        continue
                except ValueError:
                    continue
                finished[job_id] = {"state": row["State"], "actual_gpu_hours": seconds/3600,
                                    "job_incarnation": slot["job_incarnation"]}
            passed = len(finished) == len(slots)
            audit = read_json(audit_path)
            audit.update(resource_release="PASS" if passed else "PENDING_TERMINAL_CONFIRMATION", terminal_jobs=finished)
            atomic_json(audit_path, audit)
            if passed or time.monotonic() >= end:
                break
            threading_delay = min(poll_seconds, max(0, end-time.monotonic()))
            time.sleep(threading_delay)
        if passed:
            try:
                with self.transaction() as current:
                    for slot in current["slots"].values():
                        fact = finished.get(str(slot.get("job_id")))
                        if fact is not None:
                            slot.update(state="TERMINAL", terminal_verified=True, settled=True,
                                        actual_gpu_hours=fact["actual_gpu_hours"])
                            current["holds"][slot["hold_id"]].update(settled=True, actual_gpu_hours=fact["actual_gpu_hours"])
            except OSError:
                pass  # 终态与结算仍保存在本机审计；NFS恢复后可补写。
        return passed


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("--") and argv[0] != "--help":
        argv.insert(0, "submit")
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="operation", required=True)
    for name in ("init", "submit", "scale", "status", "reconcile", "settle", "revise", "cancel", "adopt", "restart"):
        p = subs.add_parser(name)
        p.add_argument("--run-root", type=Path, required=True)
        p.add_argument("--ssh-host", help="显式通过既有SSH宿主执行Slurm；省略时在GL本地执行")
        if name == "init":
            p.add_argument("--pool-id", required=True)
            p.add_argument("--schedule-id", required=True)
            p.add_argument("--schedule-identity-sha256", required=True)
            p.add_argument("--pool-policy", type=Path, required=True)
            p.add_argument("--count", type=int, required=True)
        if name in {"submit", "scale", "revise", "adopt", "restart"}:
            p.add_argument("--expected-pool-revision", type=int, required=True)
        if name == "submit":
            p.add_argument("--gates", type=Path, required=True)
            p.add_argument("--execute", action="store_true")
            p.add_argument("--dry-run", action="store_true")
        if name == "scale":
            p.add_argument("--target", type=int, required=True)
            p.add_argument("--reason", required=True)
            p.add_argument("--dry-run", action="store_true")
        if name in {"reconcile", "cancel", "adopt", "restart"}:
            p.add_argument("--proof", type=Path, required=True)
        if name in {"reconcile", "settle", "restart"}:
            p.add_argument("--slot-instance", required=True)
        if name == "settle":
            p.add_argument("--actual-gpu-hours", type=float, required=True)
        if name == "revise":
            p.add_argument("--schedule-identity-sha256", required=True)
        if name == "cancel":
            p.add_argument("--confirm-pool-cancel", action="store_true")
            p.add_argument("--local-audit", type=Path)
        if name == "adopt":
            p.add_argument("--authorization", required=True)
    args = parser.parse_args(argv)
    pool = Pool(args.run_root)
    transport = SlurmTransport(args.ssh_host)
    if args.operation == "init":
        result = pool.initialize(pool_id=args.pool_id, schedule_id=args.schedule_id,
                                 schedule_identity_sha256=args.schedule_identity_sha256,
                                 envelope=read_json(args.pool_policy), count=args.count)
    elif args.operation == "submit":
        result = pool.submit(read_json(args.gates), args.expected_pool_revision,
                             execute=args.execute and not args.dry_run, runner=transport.run)
    elif args.operation == "scale":
        if args.dry_run:
            result = read_json(pool.path)
            Pool.writable(result, args.expected_pool_revision)
            Pool.validate_envelope(result["envelope"], args.target)
            result = {"dry_run": True, "target": args.target}
        else:
            result = pool.scale(args.target, args.expected_pool_revision, args.reason)
    elif args.operation == "status":
        result = read_json(pool.path)
    elif args.operation == "reconcile":
        result = pool.reconcile(args.slot_instance, read_json(args.proof))
    elif args.operation == "settle":
        result = pool.settle(args.slot_instance, args.actual_gpu_hours)
    elif args.operation == "revise":
        result = pool.revise(args.schedule_identity_sha256, args.expected_pool_revision)
    elif args.operation == "adopt":
        result = pool.adopt(read_json(args.proof), args.expected_pool_revision, args.authorization)
    elif args.operation == "restart":
        result = pool.restart(args.slot_instance, read_json(args.proof), args.expected_pool_revision)
    else:
        result = pool.cancel(read_json(args.proof), confirmed=args.confirm_pool_cancel, audit_path=args.local_audit, runner=transport.run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
