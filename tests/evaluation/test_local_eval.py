"""本机弹性队列：真实控制器+真实worker+假策略／仿真进程；不用tmux、不用GPU。"""
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import time

import pytest

from examples.robomme import eval as candidate
from scripts.evaluation import local_eval
from scripts.evaluation.manifest import make_manifest
from scripts.evaluation.queue_store import QueueStore, confirm_local_terminal, read_json
from scripts.evaluation.resident_worker import server_command

REPO = Path(__file__).resolve().parents[2]
QUEUE = REPO / "scripts/evaluation/official_queue.py"
WORKER = REPO / "scripts/evaluation/resident_worker.py"

FAKE_POLICY = f"""#!{sys.executable}
import http.server, os, sys, pathlib
port = int(sys.argv[sys.argv.index("--port") + 1])
parent = os.getppid()
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
pathlib.Path(f"server-{{port}}").touch()
server = http.server.HTTPServer(("127.0.0.1", port), Health)
server.timeout = 0.1
while os.getppid() == parent:
    server.handle_request()
"""

# 卡住的那次先写started再等父进程死亡，模拟局中崩溃。
FAKE_SIM = f"""#!{sys.executable}
import json, os, pathlib, sys, time
value = lambda name: sys.argv[sys.argv.index(name) + 1]
grant = json.loads(pathlib.Path(value("--episode-manifest")).read_text())
output = pathlib.Path(value("--attempt-output"))
output.mkdir(parents=True)
parent = os.getppid()
keys = []
for e in grant["episodes"]:
    identity = dict(e, run_identity_sha256=grant["run_identity_sha256"], chunk_id=grant["chunk_id"], episode_nonce="n")
    key = e["episode_key"]
    (output / f"{{key}}.started.json").write_text(json.dumps(dict(identity, record_type="started")))
    if os.path.exists("hang-once"):
        os.unlink("hang-once")
        pathlib.Path("hanging").touch()
        while os.getppid() == parent:
            time.sleep(0.05)
        sys.exit(1)
    (output / f"{{key}}.terminal.json").write_text(json.dumps(dict(identity, record_type="terminal", outcome="success")))
    keys.append(key)
(output / "attempt.json").write_text(json.dumps({{"exit_code": 0, "completed_episode_keys": keys,
    "started_episode_keys": keys, "unstarted_episode_keys": []}}))
"""


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def executable(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture
def queue(tmp_path):
    """初始化7局（两块）的run根并起真实控制器。"""
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "run"
    spec = {"repo": str(home), "policy_python": executable(tmp_path / "policy", FAKE_POLICY),
            "sim_python": executable(tmp_path / "sim", FAKE_SIM), "stage2_python": "/none", "checkpoint": "/none",
            "policy_config": "c", "model_kind": "wan_fixedpool1024", "model_seed": 7, "purpose": "production", "run_identity_sha256": "a" * 64}
    episodes = [{"task_name": "BinFill", "episode_id": i, "env_seed": 100 + i, "difficulty": "easy"} for i in range(7)]
    QueueStore(root).initialize(make_manifest(episodes, official=False), spec)
    controller = subprocess.Popen([sys.executable, str(QUEUE), "controller", "--run-root", str(root), "--local"])
    yield root, home
    controller.terminate()
    controller.wait(timeout=10)


def start_worker(root, name, *extra):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="fake-gpu")
    return subprocess.Popen([sys.executable, str(WORKER), "--run-root", str(root), "--spec", str(root / "spec.json"),
                             "--worker-id", name, "--local", *extra], env=environment)


def recover(root, name, step):
    return subprocess.run([sys.executable, str(QUEUE), "recover", "--run-root", str(root), "--worker-id", name,
                           "--job-id", "local", "--step-id", str(step)], capture_output=True, text=True)


def wait_for(check, seconds=30):
    end = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < end
        time.sleep(0.05)


def started_counts(root):
    counts = {}
    for path in root.glob("attempts/*/client/*.started.json"):
        key = read_json(path)["episode_key"]
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_two_workers_finish_without_repeat_or_loss(queue):
    root, home = queue
    port = free_port()
    workers = [start_worker(root, "a", "--port", str(port)), start_worker(root, "b")]
    assert [w.wait(timeout=60) for w in workers] == [0, 0]
    summary = read_json(root / "queue/summary.json")
    assert summary["compute_status"] == "COMPUTE_DONE"
    assert summary["outcomes"]["success"] == 7 and summary["missing"] == []
    assert started_counts(root) == {f"BinFill:{i}": 1 for i in range(7)}
    assert (home / f"server-{port}").exists()
    state = read_json(root / "queue/state.json")
    assert state["workers"]["a"]["job_id"] == "local"


def test_killed_worker_is_recovered_and_other_worker_finishes(queue):
    root, home = queue
    (home / "hang-once").touch()
    first = start_worker(root, "a")
    wait_for((home / "hanging").exists)
    # 还活着时不许恢复。
    assert recover(root, "a", first.pid).returncode == 2
    first.kill()
    first.wait(timeout=10)
    # worker本身就是step记录，用它写入的PID恢复。
    assert read_json(root / "queue/state.json")["workers"]["a"]["step_id"] == str(first.pid)
    result = recover(root, "a", first.pid)
    assert result.returncode == 0, result.stdout
    second = start_worker(root, "b")
    assert second.wait(timeout=60) == 0
    summary = read_json(root / "queue/summary.json")
    assert summary["compute_status"] == "COMPUTE_DONE" and summary["completed"] == 7
    assert summary["retries"] == {"BinFill:0": 1}


def test_confirm_local_terminal():
    with pytest.raises(ValueError):
        confirm_local_terminal("local", str(os.getpid()))
    with pytest.raises(ValueError):
        confirm_local_terminal("123", "999999999")
    assert confirm_local_terminal("local", "999999999")["state"] == "LOCAL_EXITED"


def test_server_command_uses_motion_gpu_and_stage2(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    spec = {"policy_python": "/p", "stage2_python": "/s", "run_identity_sha256": "a" * 64, "policy_config": "c",
            "checkpoint": "/k", "model_kind": local_eval.MODEL_KIND, "model_seed": 7, "purpose": "production"}
    assert server_command(spec, "n", 9) == [
        "/p", "-m", "scripts.training.serve_policy", "--port", "9", "--seed", "7", "--motion-online-gpu", "3",
        "--stage2-python", "/s", "--run-identity-sha256", "a" * 64, "--server-nonce", "n",
        "--eval-purpose", "production", "policy:checkpoint", "--policy.config", "c", "--policy.dir", "/k"]


def official_error(seed):
    """seed闸门之后缺身份参数同样抛错，用文案区分是否被闸门拦下；不建仿真、不起模型。"""
    with pytest.raises(Exception) as info:
        candidate.evaluate(candidate.Args(profile="official", model_seed=seed))
    return info.value


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_official_branch_accepts_three_seeds(seed):
    assert candidate.OFFICIAL_SEEDS == (0, 7, 42)
    assert "seed 须为官方" not in str(official_error(seed))


def test_official_branch_rejects_other_seed():
    error = official_error(1)
    assert isinstance(error, candidate.ContractError) and "seed 须为官方" in str(error)


def run_seed_with_fake_tmux(monkeypatch, tmp_path, marker_exists):
    """只替换tmux与等待，真实run_seed循环：一轮RUNNING后COMPUTE_DONE，返回起过的会话。"""
    marker = tmp_path / "RESTORE_POOL.json"
    if marker_exists:
        marker.touch()
    summary = tmp_path / "out/seed42/run800/queue/summary.json"
    started, running = [], [True]

    def start_session(name, argv, log, env=None):
        started.append((name, env))
        summary.touch()

    def read(path):
        if Path(path) == summary:
            if read.first:
                read.first = False
                return {"compute_status": "RUNNING"}
            running[0] = False
            return {"compute_status": "COMPUTE_DONE", "completed": 7,
                    "outcomes": {"success": 7, "fail": 0, "timeout": 0}}
        return {"assignments": {}, "workers": {}}

    read.first = True
    for name, value in (("OUT", tmp_path / "out"), ("LOGS", tmp_path / "out/logs"), ("LATE_MARKER", marker),
                        ("GPUS", [2]), ("START_GAP", 0), ("POLL", 0), ("start_session", start_session),
                        ("read_json", read), ("alive", lambda name: running[0])):
        monkeypatch.setattr(local_eval, name, value)
    monkeypatch.setattr(local_eval.subprocess, "run", lambda *args, **kwargs: None)
    episodes = [{"task_name": "BinFill", "episode_id": i, "env_seed": 100 + i, "difficulty": "easy"} for i in range(7)]
    assert local_eval.run_seed(42, make_manifest(episodes, official=False), "c" * 40) is True
    return started


def test_late_gpus_join_when_marker_exists(monkeypatch, tmp_path):
    started = run_seed_with_fake_tmux(monkeypatch, tmp_path, True)
    assert [name for name, _ in started] == ["evl-tp-ctl-s42"] + [
        f"evl-tp-s42-g{g}k{k}" for g in (2, 0, 1) for k in range(4)]
    assert [env["CUDA_VISIBLE_DEVICES"] for _, env in started[1:]] == [2] * 4 + [0] * 4 + [1] * 4


def test_late_gpus_wait_without_marker(monkeypatch, tmp_path):
    started = run_seed_with_fake_tmux(monkeypatch, tmp_path, False)
    assert [name for name, _ in started] == ["evl-tp-ctl-s42"] + [f"evl-tp-s42-g2k{k}" for k in range(4)]


def write_accepted(out, seed, successes, total=800):
    directory = out / f"seed{seed}/run800/accepted"
    directory.mkdir(parents=True)
    tasks = ("BinFill", "StopCube")
    for i in range(total):
        outcome = "success" if i < successes else "timeout" if i % 2 else "fail"
        record = {"episode_key": f"{tasks[i % 2]}:{i}", "task_name": tasks[i % 2],
                  "difficulty": "easy" if i % 4 < 2 else "hard", "outcome": outcome}
        (directory / f"{i}.json").write_text(json.dumps(record))


def test_three_seed_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(local_eval, "OUT", tmp_path)
    for seed, successes in ((42, 346), (7, 344), (0, 352)):
        write_accepted(tmp_path, seed, successes)
    assert local_eval.three_seed_summary() == 0
    assert capsys.readouterr().out.strip().endswith("THREE_SEED=PASS seeds=7,42,0 overall=43.42±0.52")
    rows = json.loads((tmp_path / "three_seed_summary.json").read_text())
    assert {(r["group"], r["name"]) for r in rows} == {("overall", "overall"), ("difficulty", "easy"),
        ("difficulty", "hard"), ("task", "BinFill"), ("task", "StopCube")}
    assert rows[0]["seed42"] == pytest.approx(43.25) and rows[0]["std"] == pytest.approx(0.5204, abs=1e-4)
    assert "43.42±0.52" in (tmp_path / "three_seed_summary.txt").read_text()


def test_three_seed_table_requires_800(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(local_eval, "OUT", tmp_path)
    write_accepted(tmp_path, 42, 346)
    write_accepted(tmp_path, 7, 344)
    write_accepted(tmp_path, 0, 352, total=799)
    assert local_eval.three_seed_summary() == 1
    assert "THREE_SEED=FAIL" in capsys.readouterr().out
