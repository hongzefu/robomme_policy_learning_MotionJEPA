"""在已登记hold内启动正式resident；不在控制主机加载模型。"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

try:
    from .gl_official_pool import ACTIVE, Pool, SlurmTransport, atomic_json, read_json, require_gates, terminal_step_proof, slurm_records, verify_job_record, TERMINAL
except ImportError:
    from gl_official_pool import ACTIVE, Pool, SlurmTransport, atomic_json, read_json, require_gates, terminal_step_proof, slurm_records, verify_job_record, TERMINAL

LAUNCH_GATES = {"source_pin", "ssh_controlmaster", "nfs_paths", "nfs_python", "nfs_atomicity",
                "eval_runtime_contract",
                "resource_fit", "official_client_parity", "elastic_worker_queue", "source_split_audit",
                "checkpoint_60000", "checkpoint_manifest", "strict_restore", "fixed_seed", "final_run_identity"}
VALIDATION_BASE_GATES = {"source_pin", "checkpoint_manifest", "eval_runtime_contract", "fixed_seed", "checkpoint_20000",
                         "final_run_identity", "ssh_controlmaster", "nfs_paths", "nfs_python", "slurm_gpu_identity",
                         "official_client_parity", "static_budget", "resource_envelope", "elastic_worker_queue",
                         "strict_restore", "finite", "repeatability"}


def verify_launch_gates(report, purpose="production", validation_stage=None):
    if report.get("cross_hardware_numeric") != "not_required":
        raise ValueError("本次用户裁决必须如实记录跨硬件数值一致性不要求")
    numeric = {"a40_motion_numeric", "vla_stack_numeric", "a100_reference"}
    if numeric & set(report.get("required_gates", {}).get("launch", [])):
        raise ValueError("已取消的跨硬件数值门不得伪装成PASS或继续作为前置")
    required = set(LAUNCH_GATES)
    if purpose == "validation":
        required = set(VALIDATION_BASE_GATES)
        if validation_stage == "shards10":
            required.add("validation_smoke1")
        elif validation_stage != "smoke1":
            raise ValueError("validation阶段必须明确smoke1或shards10")
    elif purpose != "production":
        raise ValueError("bootstrap不通过正式或validation服务入口")
    require_gates(report, "launch", required)
    for name in set(report["required_gates"]["launch"]) | required:
        evidence = report["gates"][name]["evidence"]
        if not isinstance(evidence, dict) or not {"path", "sha256", "verdict_line"} <= evidence.keys():
            raise ValueError(f"{name}缺少可核验证据路径/哈希/判定行")
        data = Path(evidence["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != evidence["sha256"]:
            raise ValueError(f"{name}证据SHA不符")
        if evidence["verdict_line"] not in data.decode("utf-8") or "=PASS" not in evidence["verdict_line"]:
            raise ValueError(f"{name}证据判定行缺失")


def validate_spec(spec):
    purpose = spec.get("purpose")
    expected_step = {"production": 60000, "validation": 20000}.get(purpose)
    if expected_step is None or spec.get("model_seed") != 42 or spec.get("checkpoint_step") != expected_step:
        raise ValueError("正式仅60000、独立validation仅20000，均固定seed42")
    for key in ("policy_python", "stage2_python", "sim_python"):
        value = Path(spec[key])
        if not value.is_absolute() or not value.is_file():
            raise ValueError(f"显式解释器不可用：{key}")
    if len(spec.get("run_identity_sha256", "")) != 64 or not Path(spec["target_contract"]).is_file():
        raise ValueError("缺final run身份或普通评估运行合同")
    for key in ("repo", "checkpoint", "episode_manifest"):
        if not Path(spec[key]).is_absolute() or not Path(spec[key]).exists():
            raise ValueError(f"路径不可用：{key}")
    try:
        from .manifest import digest, validate_episodes, verify_seal
    except ImportError:
        from manifest import digest, validate_episodes, verify_seal
    binding = verify_seal(read_json(spec["model_binding"]))
    runtime_contract = read_json(spec["target_contract"])
    if (runtime_contract.get("schema") != "eval-runtime-contract-v1"
            or runtime_contract.get("cross_hardware_numeric") != "not_required"
            or binding.get("cross_hardware_numeric") != "not_required"
            or binding["run_descriptor"].get("cross_hardware_numeric") != "not_required"):
        raise ValueError("本用途须消费普通运行合同并如实记录跨硬件数值不要求")
    kind = "validation_run" if purpose == "validation" else "run"
    if (binding["purpose"] != purpose or binding["run_descriptor"]["purpose"] != purpose
            or binding["model_descriptor"]["purpose"] != purpose
            or binding["model_descriptor"]["checkpoint_step"] != expected_step
            or binding["run_identity_sha256"] != spec["run_identity_sha256"]
            or digest({"kind": kind, "descriptor": binding["run_descriptor"]}) != spec["run_identity_sha256"]
            or Path(binding["checkpoint_root"]).resolve() != Path(spec["checkpoint"]).resolve().parent
            or Path(spec["checkpoint"]).name != str(expected_step)
            or binding["run_descriptor"]["runtime_contract_sha256"] != hashlib.sha256(Path(spec["target_contract"]).read_bytes()).hexdigest()):
        raise ValueError("用途/模型/运行合同与C签封的run绑定不符")
    manifest = read_json(spec["episode_manifest"])
    validate_episodes(manifest, official=purpose == "production")
    if purpose == "validation":
        if not spec.get("parent_run_name"):
            raise ValueError("两个validation阶段须明确同一parent run")
        selected = spec.get("validation_episode_keys", [])
        keys = [item["episode_key"] for item in manifest["episodes"]]
        expected_count = {"smoke1": 1, "shards10": 10}.get(spec.get("validation_stage"))
        if len(selected) != expected_count or len(set(selected)) != len(selected) or set(selected) != set(keys):
            raise ValueError("validation两个阶段必须分别锁定独立1局或10局清单")
        original = read_json(spec["official_episode_manifest"])
        validate_episodes(original)
        source = {item["episode_key"]: item for item in original["episodes"]}
        if (binding["episode_manifest_sha256"] != original["sha256"]
                or spec.get("scope_manifest_sha256") != manifest["sha256"]
                or any(item != source.get(item["episode_key"]) for item in manifest["episodes"])):
            raise ValueError("validation范围不是已绑定官方800身份的不可变子集")
        bound_scope = verify_seal(binding["validation_scope"])
        stage_scope = binding["run_descriptor"]["validation_scope"]
        if (binding["validation_stage"] != spec["validation_stage"]
                or binding["run_descriptor"]["validation_stage"] != spec["validation_stage"]
                or bound_scope["sha256"] != manifest["sha256"]
                or stage_scope != {"source_episode_manifest_sha256": original["sha256"],
                                   "scope_manifest_sha256": manifest["sha256"], "episode_keys": keys}):
            raise ValueError("阶段或scope与C签封的独立validation run身份不符")
    elif binding["episode_manifest_sha256"] != manifest["sha256"]:
        raise ValueError("production必须消费完整已绑定800局清单")


def srun_command(root, ledger, slot, spec_path, policy_python, repo):
    return ["srun", f"--jobid={slot['job_id']}", "--overlap", "--exact", "--ntasks=1",
            f"--chdir={repo}",
            f"--job-name=official-resident-{slot['launch_id']}",
            f"--cpus-per-task={ledger['envelope']['cpus']}", "--gpu_cmode=shared",
            policy_python, str(Path(repo)/"scripts/evaluation/resident_worker.py"),
            "--run-root", str(root), "--worker-id", slot["worker_uuid"], "--launch-spec", str(spec_path)]


def launch_candidates(ledger, spec):
    """smoke只起一份模型，闲置第二hold保留给同campaign的十局阶段。"""
    width = ledger["desired_jobs"]
    if spec.get("purpose") == "validation":
        width = min(width, {"smoke1": 1, "shards10": 2}[spec["validation_stage"]])
    occupied = sum(slot["state"] in ACTIVE and bool(slot.get("launch_id")) and not slot.get("retiring")
                   for slot in ledger["slots"].values())
    candidates = [slot for slot in ledger["slots"].values() if slot["state"] == "RUNNING"
                  and not slot.get("retiring") and not slot.get("launch_id")]
    return sorted(candidates, key=lambda slot: slot["slot_instance"])[:max(0, width-occupied)]


def launch(pool, spec, gates, expected_revision, *, execute=False, popen=subprocess.Popen):
    verify_launch_gates(gates, spec.get("purpose"), spec.get("validation_stage"))
    validate_spec(spec)
    commands, children, reserved = [], [], []
    with pool.transaction() as ledger:
        pool.guard_queue()
        Pool.writable(ledger, expected_revision)
        if time.time() >= float(ledger["envelope"]["workload_deadline"]):
            raise ValueError("工作负载截止已到")
        if spec["purpose"] == "validation":
            campaign = ledger.get("campaign")
            if campaign is not None and campaign["parent_run_name"] != spec["parent_run_name"]:
                raise ValueError("不能把既有资源账本绑定到另一parent run")
            if campaign and campaign.get("stage_state") == "HANDOFF_READY" and spec["validation_stage"] != campaign["next_stage"]:
                raise ValueError("smoke已封口，不为取得成功重跑该阶段")
            if spec["validation_stage"] == "shards10" and (campaign is None or
                    (campaign.get("stage_state") != "HANDOFF_READY" and campaign.get("current_stage") != "shards10")):
                raise ValueError("shards10须消费同parent pool的smoke静止移交")
            if execute:
                ledger["campaign"] = {"parent_run_name": spec["parent_run_name"], "current_stage": spec["validation_stage"],
                                      "stage_state": "RUNNING", "scope_manifest_sha256": spec["scope_manifest_sha256"]}
        for slot in launch_candidates(ledger, spec):
            if not slot.get("job_incarnation") or not slot.get("end_time"):
                raise ValueError("launch前须现场对账JobID incarnation与EndTime")
            current = dict(slot)
            current.update(launch_id=uuid.uuid4().hex, worker_uuid=uuid.uuid4().hex,
                           state="STARTING", run_identity_sha256=spec["run_identity_sha256"])
            spec_path = pool.root/"pool"/"launches"/(current["launch_id"] + ".json")
            command = srun_command(pool.root, ledger, current, spec_path, spec["policy_python"], spec["repo"])
            commands.append(command)
            if not execute:
                continue
            slot.update(current)
            worker_spec = dict(spec, slot_instance=slot["slot_instance"], worker_uuid=slot["worker_uuid"],
                               job_id=slot["job_id"], job_incarnation=slot["job_incarnation"],
                               launch_id=slot["launch_id"], pool_revision=expected_revision,
                               hold_end_time=slot["end_time"], workload_deadline=ledger["envelope"]["workload_deadline"],
                               expected_gpu_uuid=slot.get("gpu_uuid", spec.get("gpu_uuid_by_job", {}).get(str(slot["job_id"]))),
                               gates=gates)
            atomic_json(spec_path, worker_spec)
            atomic_json(pool.path, ledger)  # GPU初始化前启动许可已持久。
            reserved.append((dict(slot), command))
    # 等待queue controller时不持pool锁；恢复按queue→pool顺序冻结，避免锁环。
    for slot, command in reserved:
        try:
            try:
                from .queue_store import QueueStore
            except ImportError:
                from queue_store import QueueStore
            store = QueueStore(pool.root)
            request_id = "authorize-" + slot["launch_id"]
            store.request(request_id, "authorize_slot", {
                "slot": {key: slot[key] for key in ("slot_instance", "worker_uuid", "job_id", "job_incarnation")},
                "launch_id": slot["launch_id"]})
            until = time.monotonic() + float(spec.get("controller_timeout_seconds", 60))
            response = None
            while response is None and time.monotonic() < until:
                response = store.query(request_id)
                if response is None:
                    time.sleep(0.1)
            if response is None or response.get("status") != "AUTHORIZED":
                raise OSError(f"controller未授权启动：{response}")
            with pool.transaction() as current:
                pool.guard_queue()
                Pool.writable(current, expected_revision)
                live = current["slots"][slot["slot_instance"]]
                if live["state"] != "STARTING" or live["launch_id"] != slot["launch_id"]:
                    raise ValueError("启动许可在ACK后发生变化")
                child = popen(command, cwd=spec["repo"], start_new_session=True)
                live["srun_pid"] = child.pid
                children.append(child)
        except (OSError, ValueError) as error:
            with pool.transaction() as current:
                current["slots"][slot["slot_instance"]]["state"] = "UNKNOWN"
                current["slots"][slot["slot_instance"]]["launch_error"] = str(error)
            break
    return commands, children


def controller_request(store, operation, payload, *, timeout=60):
    request_id = operation + "-" + uuid.uuid4().hex
    store.request(request_id, operation, payload)
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        response = store.query(request_id)
        if response is not None:
            if response.get("status") in {"REJECTED", "ERROR"}:
                raise RuntimeError(f"controller拒绝{operation}：{response}")
            return response
        threading.Event().wait(.1)
    raise TimeoutError(f"controller未确认{operation}，不重发为新请求")


def stopped_proof(ledger, slot, queue_slot, runner):
    if slot.get("launch_id"):
        return terminal_step_proof(ledger, slot, queue_slot, runner=runner)
    # 未起过resident的hold仍核正向Slurm身份和所有数值step，不凭空目录猜静止。
    records = slurm_records(slot["job_id"], runner)
    if verify_job_record(ledger, slot, records) is None:
        return None
    live = [row for row in records if row["JobIDRaw"].startswith(str(slot["job_id"])+".")
            and row["JobIDRaw"].split(".", 1)[1].isdigit()
            and row["State"].split()[0].rstrip("+") not in TERMINAL]
    if live:
        return None
    return {key: slot.get(key) for key in ("slot_instance", "worker_uuid", "job_id", "job_incarnation", "step_id")} | {
            "terminal": True, "rpc_quiescent": True, "sidecar_quiescent": True}


def refresh_release_snapshot(pool, release_state, audit_path, ledger=None, *, runner=subprocess.run):
    """新hold启动前先持久化本机释放身份；不会因刷新另起资源账本。"""
    ledger = read_json(pool.path) if ledger is None else ledger
    proofs, unresolved = [], []
    for key, slot in ledger["slots"].items():
        if slot["state"] not in ACTIVE:
            continue
        origin = slot.get("adoption_proof", slot)
        if not slot.get("job_id") or not slot.get("job_incarnation") or not origin.get("submit_time"):
            unresolved.append(key)
            continue
        proofs.append({**slot, "pool_id": ledger["pool_id"], "account": ledger["envelope"]["account"],
                       "cluster": ledger["envelope"]["cluster"], "user": ledger["envelope"]["user"],
                       "submit_time": origin["submit_time"]})
    pool.cancel(proofs, verified_snapshot=ledger, runner=runner)  # 只核身份，不执行取消。
    payload = {"verified_snapshot": ledger, "proofs": proofs, "unresolved": unresolved}
    atomic_json(audit_path, payload)
    release_state.update(snapshot=ledger, proofs=proofs, unresolved=unresolved)


class ExistingStep:
    """launcher重启后跟踪已登记的Slurm step，不向可能复用的旧本机PID发信号。"""
    attached = True

    def __init__(self, pool, slot, runner=subprocess.run):
        self.pool, self.slot_instance, self.runner = pool, slot["slot_instance"], runner
        self.pid = slot.get("srun_pid") or "attached-"+self.slot_instance
        self.returncode, self.proof, self.next_query = None, None, 0

    def poll(self):
        if self.returncode is not None or time.monotonic() < self.next_query:
            return self.returncode
        self.next_query = time.monotonic()+5
        ledger = read_json(self.pool.path)
        state = read_json(self.pool.root/"queue/state.json")
        slot = ledger["slots"][self.slot_instance]
        self.proof = terminal_step_proof(ledger, slot, state["slots"].get(self.slot_instance), runner=self.runner)
        if self.proof is not None:
            pieces = self.proof.get("slurm_exit_code", "").split(":")
            if len(pieces) == 2 and all(value.isdecimal() for value in pieces):
                self.returncode = int(pieces[0]) if int(pieces[1]) == 0 else 128+int(pieces[1])
            else:
                self.returncode = 0 if self.proof["slurm_state"].split()[0] == "COMPLETED" else 2
        return self.returncode


def supervise(pool, spec, gates, children, *, runner=subprocess.run, popen=subprocess.Popen,
              poll_seconds=1, max_iterations=None, drain_audit_dir=None,
              release_state=None, release_audit_path=None):
    """消费真实srun句柄与精确sacct终态，冻结后静止恢复；不以心跳判死。"""
    try:
        from .queue_store import QueueStore
    except ImportError:
        from queue_store import QueueStore
    store = QueueStore(pool.root)
    seen, evidence, checks = set(), {}, {}
    iteration = 0
    while children:
        iteration += 1
        if max_iterations is not None and iteration > max_iterations:
            return [3]
        ledger = read_json(pool.path)
        if time.time() >= float(ledger["envelope"]["workload_deadline"]):
            return [3]
        state = read_json(pool.root/"queue/state.json")
        summary_path = pool.root/"queue/summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        if summary.get("compute_status") in {"COMPUTE_DONE", "STOPPED_ERROR"}:
            return [0 if summary["compute_status"] == "COMPUTE_DONE" else 2]
        eligible = bool(launch_candidates(ledger, spec))
        if state["phase"] != "RECOVERING" and eligible:
            if release_state is None or release_audit_path is None:
                raise ValueError("接纳新hold前必须有本机释放快照")
            refresh_release_snapshot(pool, release_state, release_audit_path, ledger, runner=runner)
            _, joined = launch(pool, spec, gates, ledger["pool_revision"], execute=True, popen=popen)
            children.extend(joined)
            ledger = read_json(pool.path)
            if any(slot.get("launch_error") for slot in ledger["slots"].values() if slot["state"] in ACTIVE):
                return [2]
        for child in children:
            code = child.poll()
            if code is None or child.pid in seen or code == 0:
                continue
            candidates = [slot for slot in ledger["slots"].values() if slot["state"] in ACTIVE and
                          (slot["slot_instance"] == getattr(child, "slot_instance", None) or slot.get("srun_pid") == child.pid)]
            if len(candidates) != 1:
                raise RuntimeError("退出srun不能唯一关联本pool slot")
            slot = candidates[0]
            if time.monotonic()-checks.get(slot["slot_instance"], -100) < 5:
                continue
            checks[slot["slot_instance"]] = time.monotonic()
            proof = getattr(child, "proof", None) or terminal_step_proof(ledger, slot, state["slots"].get(slot["slot_instance"]), runner=runner)
            if proof is None:
                continue
            pool.note_terminal_step(slot["slot_instance"], proof)
            evidence[slot["slot_instance"]] = proof
            seen.add(child.pid)
            if code == 3:
                if read_json(pool.root/"queue/state.json")["phase"] != "RECOVERING":
                    if drain_audit_dir is None:
                        raise ValueError("正常drain释放须指定本机审计目录")
                    snapshot = read_json(pool.path)
                    origin = slot.get("adoption_proof", slot)
                    cancel_proof = {**slot, "pool_id": ledger["pool_id"], "account": ledger["envelope"]["account"],
                                    "cluster": ledger["envelope"]["cluster"], "user": ledger["envelope"]["user"],
                                    "submit_time": origin["submit_time"]}
                    audit = Path(drain_audit_dir)/(slot["slot_instance"]+"-drain-release.json")
                    pool.cancel([cancel_proof], confirmed=True, audit_path=audit, verified_snapshot=snapshot, runner=runner)
                    one_hold = dict(snapshot, slots={slot["slot_instance"]: snapshot["slots"][slot["slot_instance"]]})
                    pool.confirm_release(one_hold, audit, runner=runner)
                continue
            if state["phase"] != "RECOVERING":
                controller_request(store, "worker_infra", {"slot_instance": slot["slot_instance"],
                                  "error": "已核sacct精确step终态的resident异常退出", "uncertain_terminal": True})
                state = read_json(pool.root/"queue/state.json")
        if state["phase"] == "RECOVERING" and not state["freeze"].get("pending"):
            frozen = state["freeze"]["slots"]
            for key in frozen:
                atomic_json(pool.root/"pool/drain"/(key+".json"), {"reason": "controller恢复冻结"})
            ledger = read_json(pool.path)
            proofs = []
            for key in frozen:
                slot = ledger["slots"].get(key, frozen[key])
                proof = evidence.get(key)
                if proof is None and time.monotonic()-checks.get(key, -100) >= 5:
                    checks[key] = time.monotonic()
                    proof = stopped_proof(ledger, slot, state["slots"].get(key), runner)
                    if proof is not None:
                        evidence[key] = proof
                if proof is None:
                    break
                proofs.append(proof)
            if len(proofs) == len(frozen):
                # 无episode曾开始的启动失败也只有一次自动恢复，防止空转。
                startup_failures = []
                for proof in proofs:
                    key = proof["slot_instance"]
                    has_assignment = any(a["grant"]["slot"]["slot_instance"] == key for a in state["assignments"].values())
                    if not has_assignment and ledger["holds"][ledger["slots"][key]["hold_id"]].get("startup_restarts", 0) >= 1:
                        startup_failures.append(key)
                if startup_failures:
                    controller_request(store, "worker_error", {"slot_instance": startup_failures[0],
                                       "error": "同hold全局启动恢复次数已达1", "retryable": False, "uncertain_terminal": False})
                recovered = controller_request(store, "recover", {"proofs": proofs})
                result = recovered.get("summary", {})
                if result.get("compute_status") in {"COMPUTE_DONE", "STOPPED_ERROR"}:
                    return [0 if result["compute_status"] == "COMPUTE_DONE" else 2]
                for proof in proofs:
                    key = proof["slot_instance"]
                    old = ledger["slots"][key]
                    if old.get("retiring") or old.get("launch_id") is None:
                        continue
                    new_slot = pool.restart(key, proof, ledger["pool_revision"])
                    has_assignment = any(a["grant"]["slot"]["slot_instance"] == key for a in state["assignments"].values())
                    if not has_assignment:
                        with pool.transaction() as current:
                            hold = current["holds"][new_slot["hold_id"]]
                            hold["startup_restarts"] = hold.get("startup_restarts", 0)+1
                if release_state is not None and release_audit_path is not None:
                    refresh_release_snapshot(pool, release_state, release_audit_path, runner=runner)
                _, resumed = launch(pool, spec, gates, ledger["pool_revision"], execute=True, popen=popen)
                children.extend(resumed)
                if not resumed:
                    return [3]
                continue
        if all(child.poll() is not None for child in children) and state["phase"] != "RECOVERING":
            if any(child.returncode not in {0, 3} and child.pid not in seen for child in children):
                # sacct尚未给正向终态，保留未知，不提前换代。
                threading.Event().wait(poll_seconds)
                continue
            codes = [child.returncode for child in children]
            # 子进程退出0不能替代全局队列已封口的证据。
            return [2] if all(code == 0 for code in codes) else codes
        threading.Event().wait(poll_seconds)
    return []


def handoff_smoke(pool, spec, *, runner=subprocess.run, timeout_seconds=120):
    """封口后的resident静止证明到齐，才把原hold移交下一独立scope。"""
    end = time.monotonic()+timeout_seconds
    proofs = {}
    while True:
        ledger = read_json(pool.path)
        state = read_json(pool.root/"queue/state.json")
        launched = {key: slot for key, slot in ledger["slots"].items() if slot["state"] in ACTIVE and slot.get("launch_id")}
        for key, slot in launched.items():
            atomic_json(pool.root/"pool/drain"/(key+".json"), {"reason": "smoke1完成，移交shards10"})
            proof = terminal_step_proof(ledger, slot, state["slots"].get(key), runner=runner)
            if proof is not None:
                proofs[key] = proof
        if set(proofs) == set(launched):
            pool.handoff_smoke(list(proofs.values()), parent_run_name=spec["parent_run_name"],
                               scope_manifest_sha256=spec["scope_manifest_sha256"])
            return True
        if time.monotonic() >= end:
            return False
        threading.Event().wait(min(5, max(0, end-time.monotonic())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--launch-spec", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--expected-pool-revision", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--release-proofs", type=Path)
    parser.add_argument("--local-release-audit", type=Path)
    parser.add_argument("--ssh-host", help="显式通过既有SSH宿主执行Slurm；省略时在GL本地执行")
    args = parser.parse_args(argv)
    execute = args.execute and not args.dry_run
    if execute and (args.release_proofs is None or args.local_release_audit is None):
        parser.error("正式运行须先指定本pool已核Slurm身份与本机释放审计路径")
    pool = Pool(args.run_root)
    transport = SlurmTransport(args.ssh_host)
    supervisor_lock = None
    if execute:
        supervisor_lock = (pool.path.parent/"supervisor.lock").open("a+")
        try:
            fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("已有launcher监督本pool，不能重复接管") from None
    snapshot = read_json(pool.path)
    proofs = read_json(args.release_proofs) if args.release_proofs else []
    if execute:
        # 起跑前将取消所需身份固定在本机；NFS丢失不能拖住GPU释放。
        pool.cancel(proofs, verified_snapshot=snapshot, runner=transport.run)
        active = {key for key, slot in snapshot["slots"].items() if slot["state"] in ACTIVE}
        if active != {proof["slot_instance"] for proof in proofs}:
            parser.error("释放身份必须覆盖本pool全部未终态席位，UNKNOWN须先对账")
        atomic_json(args.local_release_audit, {"verified_snapshot": snapshot, "proofs": proofs})
    release_state = {"snapshot": snapshot, "proofs": proofs, "unresolved": []}
    commands, children = launch(pool, read_json(args.launch_spec), read_json(args.gates),
                                args.expected_pool_revision, execute=args.execute and not args.dry_run, popen=transport.popen)
    if execute:
        ledger = read_json(pool.path)
        created = {child.pid for child in children}
        children.extend(ExistingStep(pool, slot, runner=transport.run) for slot in ledger["slots"].values()
                        if slot["state"] in ACTIVE and slot.get("launch_id") and slot.get("srun_pid") not in created
                        and not slot.get("launch_error"))
    print(json.dumps({"commands": commands, "started": len(children)}, ensure_ascii=False), flush=True)
    # 由外层受管tmux运行；独立srun同时等待，不在登录节点执行模型。
    previous = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, interrupted)
    codes = [2]
    spec = read_json(args.launch_spec)
    try:
        failed_launch = execute and any(slot.get("launch_error") for slot in read_json(pool.path)["slots"].values())
        codes = ([2] if failed_launch or not children else supervise(pool, read_json(args.launch_spec), read_json(args.gates), children,
                        drain_audit_dir=args.local_release_audit.parent, release_state=release_state,
                        release_audit_path=args.local_release_audit, runner=transport.run, popen=transport.popen)) if execute else []
    except KeyboardInterrupt:
        codes = [3]
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if execute:
                try:
                    from .resident_worker import stop_process_tree
                except ImportError:
                    from resident_worker import stop_process_tree
                for child in children:
                    if not getattr(child, "attached", False) and child.poll() is None:
                        stop_process_tree(child, 10)
                transferred = (codes == [0] and spec.get("purpose") == "validation" and spec.get("validation_stage") == "smoke1"
                               and handoff_smoke(pool, spec, runner=transport.run))
                if transferred:
                    print("PHASE_HANDOFF=PASS from=smoke1 to=shards10 holds=reused campaign_complete=0", flush=True)
                else:
                    if codes == [0] and spec.get("validation_stage") == "smoke1":
                        codes = [2]
                    snapshot, proofs = release_state["snapshot"], release_state["proofs"]
                    pool.cancel(proofs, confirmed=True, audit_path=args.local_release_audit, verified_snapshot=snapshot, runner=transport.run)
                    released = pool.confirm_release(snapshot, args.local_release_audit, runner=transport.run) and not release_state["unresolved"]
                    print(f"RESOURCE_RELEASE={'PASS' if released else 'PENDING'} jobs={len(proofs)}", flush=True)
                    if not released:
                        codes = [2]
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            if supervisor_lock is not None:
                supervisor_lock.close()
    return 0 if all(code == 0 for code in codes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
