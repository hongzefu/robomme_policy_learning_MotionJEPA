"""进程生命周期、终局先于产物、Slurm启动许可负控。"""
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluation.gl_official_pool import atomic_json
from scripts.evaluation.resident_worker import Resident, pick_port, stop_process_tree, validate_permit


class Store:
    def __init__(self, root):
        self.root, self.requests, self.receipts = root, [], []

    def request(self, rid, operation, payload):
        self.requests.append((operation, payload))

    def query(self, rid):
        return {"status": "ACCEPTED"}

    def write_attempt(self, grant, kind, payload, rid):
        path = self.root/(rid + ".json")
        atomic_json(path, payload)
        self.receipts.append((kind, payload))
        return path


class ResidentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.spec = {"slot_instance": "slot", "worker_uuid": "worker", "launch_id": "launch",
                     "job_id": "1", "job_incarnation": "inc", "pool_revision": 0,
                     "run_identity_sha256": "a"*64, "hold_end_time": 99999999999,
                     "workload_deadline": 99999999999, "repo": str(self.root), "sim_python": sys.executable,
                     "model_kind": "stage2", "term_grace_seconds": .1}
        self.store = Store(self.root)
        self.worker = Resident(self.root, self.spec, self.store)

    def tearDown(self):
        self.temp.cleanup()

    def grant(self):
        return {"attempt_id": "attempt", "chunk_id": "chunk", "attempt_dir": str(self.root/"attempt"),
                "run_identity_sha256": "a"*64, "episodes": [{"episode_key": "Task:0", "task_name": "Task",
                  "episode_id": 0, "env_seed": 123, "difficulty": "easy"}]}

    def test_port_probe_rejects_busy_port(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            with self.assertRaises(OSError):
                pick_port(requested=sock.getsockname()[1])

    def test_real_owned_process_group_cleanup(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        self.assertTrue(stop_process_tree(child, .1))
        self.assertIsNotNone(child.poll())

    def test_terminal_forwarded_without_artifact(self):
        grant = self.grant()
        output = self.root/"attempt/client"
        terminal = dict(grant["episodes"][0], outcome="success", actions_executed=1, episode_nonce="nonce", rpc_seq=1)
        atomic_json(output/"Task:0.terminal.json", terminal)
        self.worker.forward_receipts(grant, output)
        self.assertEqual([item[0] for item in self.store.receipts], ["terminal"])
        self.assertEqual(self.store.requests[0][0], "terminal")
        self.worker.forward_receipts(grant, output)
        self.assertEqual(len(self.store.receipts), 1)

    def test_artifact_paths_translated_and_escape_rejected(self):
        grant = self.grant()
        output = self.root/"attempt/client"
        artifact = dict(grant["episodes"][0], files=[{"path": "movie.mp4", "bytes": 1, "sha256": "b"*64}], artifact_error=False)
        atomic_json(output/"Task:0.artifact.json", artifact)
        self.worker.forward_receipts(grant, output)
        self.assertEqual(self.store.receipts[0][1]["files"][0]["path"], "attempt/client/movie.mp4")
        self.worker.forwarded.clear()
        artifact["files"][0]["path"] = "../../../../escape"
        atomic_json(output/"Task:0.artifact.json", artifact)
        with self.assertRaises(ValueError):
            self.worker.forward_receipts(grant, output)

    def test_srun_required_and_stale_launch_fenced(self):
        atomic_json(self.root/"pool/ledger.json", {"state": "ACTIVE", "pool_revision": 0,
                    "slots": {"slot": dict(self.spec, state="STARTING")}})
        with self.assertRaises(ValueError):
            validate_permit(self.root, self.spec, {})
        env = {"SLURM_JOB_ID": "1", "SLURM_STEP_ID": "2", "CUDA_VISIBLE_DEVICES": "0"}
        self.assertEqual(validate_permit(self.root, self.spec, env)["step_id"], "2")
        atomic_json(self.root/"pool/ledger.json", {"state": "RECOVERING", "pool_revision": 0,
                    "slots": {"slot": dict(self.spec, state="STARTING")}})
        with self.assertRaises(ValueError):
            validate_permit(self.root, self.spec, env)

    def test_six_episode_chunk_rejected_before_start(self):
        grant = self.grant()
        grant["episodes"] *= 6
        with self.assertRaises(ValueError):
            self.worker.run_chunk(grant)

    def test_two_real_sim_lifecycles_reuse_same_resident_server(self):
        server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        self.worker.server = server
        self.worker.port = 12345
        script = (
            "import json,pathlib,sys; o=pathlib.Path(sys.argv[1]);o.mkdir();"
            "g=json.loads(pathlib.Path(sys.argv[2]).read_text());e=g['episodes'][0];"
            "e.update(outcome='success',actions_executed=1,episode_nonce='n',rpc_seq=1);"
            "(o/'Task:0.terminal.json').write_text(json.dumps(e));"
            "(o/'attempt.json').write_text(json.dumps({'completed_episode_keys':['Task:0'],"
            "'unstarted_episode_keys':[]}))"
        )
        commands = []
        def client(spec, grant_path, output, nonce, port, drain_file):
            commands.append(str(output))
            return [sys.executable, "-c", script, str(output), str(grant_path)]
        try:
            with patch("scripts.evaluation.resident_worker.client_command", side_effect=client):
                for index in range(2):
                    grant = self.grant()
                    grant.update(attempt_id=f"attempt-{index}", attempt_dir=str(self.root/f"attempt-{index}"))
                    self.assertEqual(self.worker.run_chunk(grant), 0)
                    self.assertIsNone(server.poll())
            self.assertEqual(len(set(commands)), 2)
            self.assertEqual([op for op, _ in self.store.requests].count("close"), 2)
        finally:
            stop_process_tree(server, .1)

    def test_artifact_rejection_does_not_retry_policy(self):
        with patch.object(self.store, "query", return_value={"status": "REJECTED"}):
            self.assertEqual(self.worker.rpc("artifact", {"path": "missing"})["status"], "REJECTED")
            with self.assertRaises(RuntimeError):
                self.worker.rpc("terminal", {"path": "missing"})

    def test_global_contract_error_uses_deterministic_worker_error(self):
        output = self.root/"attempt/client"
        atomic_json(output/"client.infra.json", {"retryable": False, "error": "合同错"})
        self.worker.forward_receipts(self.grant(), output)
        operation, payload = self.store.requests[-1]
        self.assertEqual(operation, "worker_error")
        self.assertFalse(payload["uncertain_terminal"])


if __name__ == "__main__":
    unittest.main()
