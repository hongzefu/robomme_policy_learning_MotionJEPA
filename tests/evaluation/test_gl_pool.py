"""真实文件锁与原子账本，Slurm命令全部由mock接管。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import types
from unittest.mock import patch

from scripts.evaluation.gl_official_pool import Pool, SUBMIT_GATES, atomic_json, committed_gpu_hours, read_json, terminal_step_proof
from scripts.evaluation.launch_official_pool import LAUNCH_GATES, srun_command, supervise, validate_spec, verify_launch_gates

try:
    from scripts.evaluation.manifest import make_manifest
    from scripts.evaluation.queue_store import Controller, QueueStore
except ImportError:
    QueueStore = None


class PoolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pool = Pool(self.root)
        self.envelope = {"max_jobs": 4, "gpu_hour_limit": 96, "workload_deadline": time.time()+100000,
                         "account": "test", "partition": "mock", "cluster": "gl", "user": "me",
                         "cpus": 4, "mem": "24G", "hours": 48}
        self.pool.initialize(pool_id="fixed-pool", schedule_id="fixed-schedule", schedule_identity_sha256="a"*64,
                             envelope=self.envelope, count=2)
        self.gates = {"required_gates": {"submit": list(SUBMIT_GATES)},
                      "gates": {key: {"status": "PASS", "evidence": "mock-input-only"} for key in SUBMIT_GATES}}

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def success(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "101;gl\n", "")

    def test_default_dry_run_no_reservation_and_shared_hold(self):
        commands = self.pool.submit(self.gates, 0)
        self.assertEqual(len(commands), 2)
        self.assertEqual(read_json(self.pool.path)["slots"], {})
        self.assertIn("--wrap=sleep infinity", commands[0])
        self.assertIn("--gpu_cmode=shared", commands[0])
        self.assertIn("--time=48:00:00", commands[0])

    def test_unknown_submission_keeps_full_commitment_and_no_retry(self):
        def unknown(*args, **kwargs):
            raise subprocess.TimeoutExpired("sbatch", 30)
        self.pool.submit(self.gates, 0, execute=True, runner=unknown)
        ledger = read_json(self.pool.path)
        self.assertEqual(len(ledger["slots"]), 1)
        self.assertEqual(committed_gpu_hours(ledger), 48)
        self.assertEqual(next(iter(ledger["slots"].values()))["state"], "UNKNOWN")
        self.pool.submit(self.gates, 0, execute=True, runner=self.success)
        self.assertEqual(len(read_json(self.pool.path)["slots"]), 2)
        self.assertEqual(self.pool.submit(self.gates, 0, execute=True, runner=self.success), [])

    def test_47_gpu_hours_rejects_and_concurrent_submit_reserves_once(self):
        with self.pool.transaction() as ledger:
            ledger["envelope"]["gpu_hour_limit"] = 47
            ledger["desired_jobs"] = 1
        with self.assertRaises(ValueError):
            self.pool.submit(self.gates, 0, execute=True, runner=self.success)
        with self.pool.transaction() as ledger:
            ledger["envelope"]["gpu_hour_limit"] = 48
        with ThreadPoolExecutor(2) as executor:
            results = list(executor.map(lambda _: self.pool.submit(self.gates, 0, execute=True, runner=self.success), range(2)))
        self.assertEqual(sum(map(len, results)), 1)
        self.assertEqual(committed_gpu_hours(read_json(self.pool.path)), 48)

    def test_scale_revision_and_drain_no_budget_refund(self):
        self.pool.submit(self.gates, 0, execute=True, runner=self.success)
        self.pool.scale(0, 0, "用户缩容")
        ledger = read_json(self.pool.path)
        self.assertEqual(ledger["pool_revision"], 1)
        self.assertEqual(committed_gpu_hours(ledger), 96)
        self.assertEqual(len(list((self.root/"pool/drain").glob("*.json"))), 2)
        with self.assertRaises(ValueError):
            self.pool.scale(1, 0, "旧revision")
        self.pool.revise("b"*64, 1)
        revised = read_json(self.pool.path)
        self.assertEqual(revised["pool_id"], "fixed-pool")
        self.assertEqual(len(revised["slots"]), 2)

    def test_recovery_freezes_submit_scale_launch_intent(self):
        atomic_json(self.root/"queue/state.json", {"phase": "RECOVERING"})
        with self.assertRaises(ValueError):
            self.pool.submit(self.gates, 0)
        with self.assertRaises(ValueError):
            self.pool.scale(0, 0, "冻结")

    def test_zero_query_never_clears_unknown_and_foreign_cancel_rejected(self):
        self.pool.submit(self.gates, 0, execute=True, runner=self.success)
        slot = next(iter(read_json(self.pool.path)["slots"].values()))
        with self.assertRaises(ValueError):
            self.pool.reconcile(slot["slot_instance"], {})
        proof = dict(slot, pool_id="other-pool", account="test", cluster="gl", user="me", submit_time="t", job_incarnation="inc")
        with self.assertRaises(ValueError):
            self.pool.cancel([proof], confirmed=True, runner=self.success, audit_path=self.root/"audit.json")
        proof["pool_id"] = "fixed-pool"
        calls = []
        def cancel(command, **kwargs):
            calls.append(command)
            return self.success(command)
        self.pool.cancel([proof], confirmed=True, runner=cancel, audit_path=self.root/"audit.json", verify_current=False)
        self.assertEqual(calls, [["scancel", "101"]])
        self.assertEqual(read_json(self.root/"audit.json")["resource_release"], "PENDING_TERMINAL_CONFIRMATION")

    def test_terminal_timeout_not_automatically_replaced(self):
        self.pool.submit(self.gates, 0, execute=True, runner=self.success)
        with self.pool.transaction() as ledger:
            for slot in ledger["slots"].values():
                slot.update(state="TERMINAL", terminal_verified=True, slurm_state="TIMEOUT")
        self.assertEqual(self.pool.submit(self.gates, 0, execute=True, runner=self.success), [])

    def test_cancel_uses_preverified_snapshot_when_nfs_fails(self):
        self.pool.submit(self.gates, 0, execute=True, runner=self.success)
        snapshot = read_json(self.pool.path)
        slot = next(iter(snapshot["slots"].values()))
        proof = dict(slot, pool_id="fixed-pool", account="test", cluster="gl", user="me", submit_time="t", job_incarnation="inc")
        calls = []
        def cancel(command, **kwargs):
            calls.append(command)
            return self.success(command)
        with patch.object(self.pool, "transaction", side_effect=OSError("NFS不可用")):
            self.pool.cancel([proof], confirmed=True, runner=cancel, audit_path=self.root/"local-audit.json",
                             verified_snapshot=snapshot, verify_current=False)
        self.assertEqual(calls, [["scancel", "101"]])

    def test_existing_authorized_hold_adopts_without_sbatch(self):
        proof = {"slot_instance": "adopted", "submission_id": "original-submit", "job_id": "123",
                 "job_incarnation": "original-start", "pool_id": "fixed-pool", "account": "test", "cluster": "gl",
                 "user": "me", "submit_time": 10, "start_time": 20, "end_time": 172820,
                 "state": "RUNNING", "hours": 48, "gpus": 1}
        self.pool.adopt(proof, 0, "用户已逐个批准JobID123")
        ledger = read_json(self.pool.path)
        self.assertEqual(committed_gpu_hours(ledger), 48)
        with self.assertRaises(ValueError):
            self.pool.adopt(proof, 0, "不可重复")

    def test_validation_cannot_acquire_production_step(self):
        for purpose, step in (("production", 20000), ("validation", 60000), ("measurement_bootstrap", 20000)):
            with self.assertRaises(ValueError):
                validate_spec({"purpose": purpose, "checkpoint_step": step, "model_seed": 42})

    def test_budget_nan_rejected(self):
        envelope = dict(self.envelope, gpu_hour_limit=float("nan"))
        with self.assertRaises(ValueError):
            Pool.validate_envelope(envelope, 1)

    def test_launch_gates_distinct_and_evidence_tamper_rejected(self):
        data = b"REAL_REPORT=PASS\n"
        evidence = self.root/"real-report.txt"
        evidence.write_bytes(data)
        report = {"required_gates": {"launch": list(LAUNCH_GATES)}, "gates": {
            key: {"status": "PASS", "evidence": {"path": str(evidence), "sha256": hashlib.sha256(data).hexdigest(),
                                                   "verdict_line": "REAL_REPORT=PASS"}} for key in LAUNCH_GATES}}
        verify_launch_gates(report)
        report["gates"]["resource_fit"]["mock"] = True
        with self.assertRaises(ValueError):
            verify_launch_gates(report)
        report["gates"]["resource_fit"].pop("mock")
        evidence.write_text("REAL_REPORT=FAIL")
        with self.assertRaises(ValueError):
            verify_launch_gates(report)
        ledger = read_json(self.pool.path)
        command = srun_command(self.root, ledger, {"job_id": "101", "worker_uuid": "worker", "launch_id": "launch"}, "/spec", "/policy", "/repo")
        self.assertEqual(command[0], "srun")
        self.assertIn("--gpu_cmode=shared", command)

    def test_exact_step_terminal_requires_positive_job_incarnation(self):
        ledger = read_json(self.pool.path)
        origin = {"submit_time": "2026-10-26T10:00:00", "start_time": "2026-10-26T10:01:00"}
        slot = {"slot_instance": "slot", "job_id": "123", "worker_uuid": "worker", "job_incarnation": "inc",
                "launch_id": "launch", "adoption_proof": origin, "step_id": "0"}
        rows = ("123|gl|me|test|original|2026-10-26T10:00:00|2026-10-26T10:01:00|Unknown|RUNNING|hold|100\n"
                "123.0||||||2026-10-26T10:02:00|2026-10-26T10:03:00|FAILED|official-resident-launch|60\n")
        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, rows, "")
        proof = terminal_step_proof(ledger, slot, runner=runner)
        self.assertEqual(proof["step_id"], "0")
        self.assertTrue(proof["sidecar_quiescent"])
        changed = dict(slot, adoption_proof=dict(origin, start_time="2026-10-26T10:02:00"))
        self.assertIsNone(terminal_step_proof(ledger, changed, runner=runner))
        self.assertIsNone(terminal_step_proof(ledger, slot, runner=lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "", "")))

    def test_hold_release_confirmation_settles_actual_usage(self):
        proof = {"slot_instance": "adopted", "submission_id": "original-submit", "job_id": "123",
                 "job_incarnation": "original-start", "pool_id": "fixed-pool", "account": "test", "cluster": "gl",
                 "user": "me", "submit_time": "2026-10-26T10:00:00", "start_time": "2026-10-26T10:01:00",
                 "end_time": 9999999999, "state": "RUNNING", "hours": 48, "gpus": 1}
        self.pool.adopt(proof, 0, "已批准")
        snapshot = read_json(self.pool.path)
        audit = self.root/"release.json"
        atomic_json(audit, {})
        output = "123|gl|me|test|original|2026-10-26T10:00:00|2026-10-26T10:01:00|2026-10-26T11:01:00|CANCELLED|hold|3600\n"
        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, output, "")
        self.assertTrue(self.pool.confirm_release(snapshot, audit, runner=runner, timeout_seconds=0))
        self.assertEqual(committed_gpu_hours(read_json(self.pool.path)), 1)
        self.assertEqual(read_json(audit)["resource_release"], "PASS")

    @unittest.skipIf(QueueStore is None, "C合并后运行真实跨owner协议测试")
    def test_supervisor_imports_orphan_raw_terminal_recovers_and_reuses_hold(self):
        proof = {"slot_instance": "slot", "submission_id": "original-submit", "job_id": "123",
                 "job_incarnation": "inc", "pool_id": "fixed-pool", "account": "test", "cluster": "gl",
                 "user": "me", "submit_time": "2026-10-26T10:00:00", "start_time": "2026-10-26T10:01:00",
                 "end_time": time.time()+20000, "state": "RUNNING", "hours": 48, "gpus": 1}
        self.pool.adopt(proof, 0, "已批准")
        with self.pool.transaction() as ledger:
            ledger["slots"]["slot"].update(worker_uuid="worker", launch_id="launch", srun_pid=111, step_id="0")
        manifest = make_manifest([{"task_name": "Task", "episode_id": i, "env_seed": i+100,
                                   "difficulty": "easy"} for i in range(5)], official=False)
        store = QueueStore(self.root)
        store.initialize(manifest, "b"*64, official=False)
        original_request = QueueStore.request
        with Controller(store) as controller:
            member = {"slot_instance": "slot", "worker_uuid": "worker", "job_id": "123",
                      "job_incarnation": "inc", "step_id": "0", "state": "RUNNING"}
            controller.register_slot(member)
            grant = controller.claim("claim-before-crash", "slot")["grant"]
            payload = dict(grant["episodes"][0], run_identity_sha256="b"*64, chunk_id=grant["chunk_id"],
                           episode_nonce="nonce", outcome="success", actions_executed=1, rpc_seq=1)
            atomic_json(Path(grant["attempt_dir"])/"client/Task:0.started.json", dict(payload, record_type="started"))
            atomic_json(Path(grant["attempt_dir"])/"client/Task:0.terminal.json", dict(payload, record_type="terminal"))
            def request(self, *args):
                result = original_request(self, *args)
                controller.process_requests()
                atomic_json(self.root/"queue/summary.json", controller.summary())
                return result
            rows = ("123|gl|me|test|original|2026-10-26T10:00:00|2026-10-26T10:01:00|Unknown|RUNNING|hold|100\n"
                    "123.0||||||2026-10-26T10:02:00|2026-10-26T10:03:00|FAILED|official-resident-launch|60\n")
            def runner(command, **kwargs):
                return subprocess.CompletedProcess(command, 0, rows, "")
            failed = types.SimpleNamespace(pid=111, returncode=2, poll=lambda: 2)
            resumed = types.SimpleNamespace(pid=222, returncode=None, poll=lambda: None)
            spec = {"purpose": "production", "run_identity_sha256": "b"*64, "policy_python": "/mock/policy",
                    "stage2_python": "/mock/stage2", "sim_python": "/mock/sim", "repo": str(self.root),
                    "controller_timeout_seconds": 1}
            with patch.object(QueueStore, "request", request), patch("scripts.evaluation.launch_official_pool.verify_launch_gates"), \
                 patch("scripts.evaluation.launch_official_pool.validate_spec"):
                supervise(self.pool, spec, {}, [failed], runner=runner, popen=lambda *a, **k: resumed,
                          poll_seconds=.001, max_iterations=2)
            result = controller.summary()
            self.assertEqual(result["execution_coverage"], 1)
            ledger = read_json(self.pool.path)
            self.assertEqual(ledger["slots"]["slot"]["state"], "RETIRED")
            self.assertEqual(len(ledger["slots"]), 2)
            self.assertEqual(committed_gpu_hours(ledger), 48)


if __name__ == "__main__":
    unittest.main()
