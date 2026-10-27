"""进程生命周期、终局先于产物、Slurm启动许可负控。"""
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.evaluation.gl_official_pool import atomic_json
from scripts.evaluation.resident_worker import Resident, pick_port, read_and_release_metadata, stop_process_tree, validate_permit, validate_server_metadata, worker_environment

try:
    from scripts.evaluation.manifest import digest, seal
except ImportError:
    seal = None


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
                     "model_kind": "stage2", "term_grace_seconds": .1,
                     "cache_env": {"MMEVLA_V1_STORE": "/nfs/approved/v1-store", "OPENPI_DATA_HOME": "/nfs/approved/tokenizer",
                         "MS_ASSET_DIR": "/nfs/approved/cache/maniskill",
                         "UV_CACHE_DIR": "/nfs/approved/cache/uv", "UV_LINK_MODE": "copy", "HF_HOME": "/nfs/approved/cache/hf",
                         "HF_HUB_CACHE": "/nfs/approved/cache/hf/hub", "XDG_CACHE_HOME": "/nfs/approved/cache",
                         "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "UV_OFFLINE": "1",
                         "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}}
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
            "import json,os,pathlib,sys; assert os.environ['MMEVLA_V1_STORE']=='/nfs/approved/v1-store';"
            "assert os.environ['HF_HUB_CACHE']=='/nfs/approved/cache/hf/hub';"
            "assert os.environ['MS_ASSET_DIR']=='/nfs/approved/cache/maniskill';"
            "assert os.environ['UV_OFFLINE']=='1';assert os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']=='false';"
            "o=pathlib.Path(sys.argv[1]);o.mkdir();"
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

    def test_worker_cache_comes_from_spec_and_cannot_replace_slurm_or_home(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1", "SLURM_JOB_ID": "123"}, clear=True):
            env = worker_environment(self.spec)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["UV_CACHE_DIR"], "/nfs/approved/cache/uv")
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")
        with self.assertRaises(ValueError):
            worker_environment({"cache_env": {"HOME": "/wrong"}})
        for key in ("MMEVLA_V1_STORE", "OPENPI_DATA_HOME", "MS_ASSET_DIR", "HF_HUB_CACHE"):
            cache = dict(self.spec["cache_env"])
            cache.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                worker_environment(dict(self.spec, cache_env=cache))
        for key, value in (("HF_HUB_OFFLINE", "0"), ("UV_OFFLINE", "0"), ("XLA_PYTHON_CLIENT_PREALLOCATE", "true")):
            with self.subTest(changed=key), self.assertRaises(ValueError):
                worker_environment(dict(self.spec, cache_env=dict(self.spec["cache_env"], **{key: value})))

    def test_real_server_subprocess_receives_explicit_store_and_cache_environment(self):
        output = self.root/"server-env.json"
        script = ("import json,os,pathlib,time;pathlib.Path("+repr(str(output))+").write_text(json.dumps(dict(os.environ)));time.sleep(30)")
        self.worker.spec["gates"] = {}
        member = dict(self.spec, step_id="0", state="RUNNING")
        class Health:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        def metadata(*args, **kwargs):
            until = time.monotonic()+2
            while not output.exists() and time.monotonic() < until:
                time.sleep(.01)
            return {"motion_runtime": {"gpu_uuid": "GPU-test", "pid": 222}}
        try:
            with patch("scripts.evaluation.resident_worker.validate_permit", return_value=member), \
                 patch("scripts.evaluation.resident_worker.verify_launch_gates"), \
                 patch("scripts.evaluation.resident_worker.validate_spec"), \
                 patch("scripts.evaluation.resident_worker.server_command", return_value=[sys.executable, "-c", script]), \
                 patch("scripts.evaluation.resident_worker.urlopen", return_value=Health()), \
                 patch("scripts.evaluation.resident_worker.read_and_release_metadata", side_effect=metadata), \
                 patch.object(self.worker, "rpc", side_effect=[{"status": "REGISTERED"}, {"status": "COMPUTE_DONE"}]):
                self.assertEqual(self.worker.run(), 0)
            actual = json.loads(output.read_text())
            for key, value in self.spec["cache_env"].items():
                self.assertEqual(actual[key], value)
        finally:
            self.worker.close()

    @unittest.skipIf(seal is None, "C合并后运行模型binding摘要核验")
    def test_metadata_checks_actual_binding_process_and_gpu_then_releases_session(self):
        binding = seal({"run_identity_sha256": "a"*64, "validation_model_identity_sha256": "b"*64,
                        "model_descriptor": {"checkpoint_files": [{"path": "20000/params", "sha256": "c"*64, "bytes": 10}]}})
        path = self.root/"binding.json"
        atomic_json(path, binding)
        spec = dict(self.spec, model_binding=str(path), purpose="validation", checkpoint_step=20000,
                    model_kind="stage2_pool4_concat", expected_gpu_uuid="GPU-owned")
        env = {"SLURM_JOB_ID": "1", "SLURM_STEP_ID": "0", "CUDA_VISIBLE_DEVICES": "0"}
        metadata = {"run_identity_sha256": "a"*64, "server_nonce": "nonce", "model_seed": 42,
                    "model_kind": spec["model_kind"], "server_pid": 111, "eval_purpose": "validation",
                    "checkpoint_step": 20000, "worker_uuid": "worker", "model_binding_sha256": binding["sha256"],
                    "model_identity_sha256": "b"*64, "checkpoint_files_sha256": digest(binding["model_descriptor"]["checkpoint_files"]),
                    "slurm_job_id": "1", "slurm_step_id": "0", "cuda_visible_devices": "0",
                    "cross_hardware_numeric": "not_required", "motion_runtime": {"gpu_uuid": "GPU-owned", "pid": 222, "cuda_visible_devices": "0"}}
        for change in ({"model_binding_sha256": "d"*64}, {"server_pid": 333}, {"server_nonce": "other"},
                       {"motion_runtime": {"gpu_uuid": "GPU-other", "pid": 222, "cuda_visible_devices": "0"}}):
            with self.assertRaises(ValueError):
                validate_server_metadata(dict(metadata, **change), spec, "nonce", 111, env)
        class Codec:
            unpackb = staticmethod(json.loads)
            class Packer:
                def pack(self, data):
                    return json.dumps(data).encode()
        class Connection:
            def __enter__(self):
                self.count, self.sent, self.closed = 0, None, False
                return self
            def __exit__(self, *args):
                self.closed = True
            def send(self, data):
                self.sent = json.loads(data)
            def recv(self, **kwargs):
                self.count += 1
                if self.count == 1:
                    return json.dumps(metadata).encode()
                return json.dumps({"release_attempt_finished": True, "quiescent": True, "eval_rpc": self.sent["eval_rpc"]}).encode()
        connection = Connection()
        result = read_and_release_metadata(spec, "nonce", 111, 1234, connector=lambda *a, **k: connection, codec=Codec, environment=env)
        self.assertEqual(result["model_binding_sha256"], binding["sha256"])
        self.assertTrue(connection.sent["release_attempt"])
        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
