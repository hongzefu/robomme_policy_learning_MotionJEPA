"""冻结观测与独立官方循环 oracle；不创建仿真环境、不启动模型。"""

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from examples.robomme import eval as candidate
from examples.robomme.utils import EpisodeState, TASK_NAME_LIST, pack_buffer


IMAGE = np.full((256, 256, 3), 7, np.uint8)
WRIST = np.full((256, 256, 3), 11, np.uint8)
STATE = np.arange(8, dtype=np.float32)


class Recorder:
    def __init__(self, save_dir, task_goal, fps):
        self.save_dir = Path(save_dir)
        self.frames = []

    def record(self, **frame):
        self.frames.append((int(frame["image"][0, 0, 0]), "action" in frame))

    def save_video(self, filename):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / filename).write_bytes(b"fixture-video")


class FrozenRunner:
    stop_at = 17
    demo = 33
    failure = None
    created = []

    def __init__(self, env_id="BinFill", video_save_dir=None, **kwargs):
        self.env_id = env_id
        self.episode_id = 0
        self.env_builder = SimpleNamespace(resolve_episode=lambda episode_id: (episode_id + 100, "hard"))
        self.num_episodes = 50
        self.trace = []
        self.closed = False
        self.count = 0
        self.created.append(self)

    def make_env(self, episode_id):
        self.episode_id = episode_id
        if self.failure == "make":
            raise RuntimeError("original-make-stack")

    def get_init_obs(self):
        self.trace.append(("reset", self.demo + 1))
        if self.failure == "reset":
            raise RuntimeError("original-reset-stack")
        return {"images": [IMAGE] * (self.demo + 1), "wrist_images": [WRIST] * (self.demo + 1),
                "states": [STATE] * (self.demo + 1), "task_goal": "冻结的官方 prompt"}

    def step(self, action):
        self.count += 1
        self.trace.append(("step", int(action[0])))
        if self.failure == "step":
            raise RuntimeError("original-step-stack")
        if self.failure == "none":
            return None, False, "error"
        return (IMAGE, WRIST, STATE), self.count == self.stop_at, "success" if self.count == self.stop_at else "unknown"

    def close_env(self):
        self.closed = True
        if self.failure == "close":
            raise RuntimeError("original-close-stack")


class FakeClient:
    def __init__(self, args=None):
        self.metadata = {"run_identity_sha256": "a" * 64, "server_nonce": "server", "model_seed": 42, "model_kind": "stage2"}
        self.requests = []
        self.closed = False
        self.drop = None

    def request(self, payload):
        self.requests.append(payload)
        rpc = payload["eval_rpc"]
        operation = "reset" if payload.get("reset") else "add_buffer" if payload.get("add_buffer") else "release_attempt" if payload.get("release_attempt") else "infer"
        if self.drop == operation:
            raise candidate.UncertainRPC("已执行但 ACK 丢失")
        if operation == "infer":
            actions = np.repeat(np.arange(20, dtype=np.float32)[:, None], 8, axis=1)
            return {"actions": actions, "eval_rpc": rpc}
        return {f"{operation}_finished": True, "eval_rpc": rpc, "quiescent": True}

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(candidate, "RolloutRecorder", Recorder)
    FrozenRunner.created = []
    monkeypatch.setattr(FrozenRunner, "failure", None)
    monkeypatch.setattr(FrozenRunner, "stop_at", 17)
    monkeypatch.setattr(FrozenRunner, "demo", 33)


def args_for(tmp_path, episodes=1):
    members = [{"episode_key": f"BinFill:{index}", "task_name": "BinFill", "episode_id": index,
                "env_seed": index + 100, "difficulty": "hard"} for index in range(episodes)]
    manifest = tmp_path / "grant.json"
    manifest.write_text(json.dumps({"chunk_id": "chunk", "run_identity_sha256": "a" * 64, "episodes": members}))
    return candidate.Args(episode_manifest=str(manifest), chunk="chunk", attempt_output=str(tmp_path / "client"),
                          run_identity_sha256="a" * 64, expected_server_nonce="server", model_kind="stage2")


def oracle(stop_at):
    # 独立直译官方顺序；期望值不调用候选计数或动作切片函数。
    count, trace = 0, []
    while True:
        action = count % 16
        trace.append(("step", action))
        count += 1
        outcome = "success" if count == stop_at else "unknown"
        if count > 1300:
            return "timeout", count, trace
        if count == stop_at:
            return outcome, count, trace


@pytest.mark.parametrize("stop_at", [1, 16, 17, 1299, 1300, 1301, 1400])
def test_official_loop_trace(tmp_path, monkeypatch, stop_at):
    monkeypatch.setattr(FrozenRunner, "stop_at", stop_at)
    runner, client = FrozenRunner(), FakeClient()
    evaluator = candidate.EpisodeEvaluator(candidate.Args(), tmp_path)
    expected, count, trace = oracle(stop_at)
    assert evaluator.eval_each_episode(runner, tmp_path, client, "episode") == expected
    assert evaluator.actions_executed == count
    assert runner.trace[1:] == trace
    assert runner.trace[0] == ("reset", 34)
    assert evaluator.recorder.frames[0] == (7, False)
    assert len(evaluator.recorder.frames) == 34 + min(count, 1300)


def test_full_demo_then_16_and_stable_global_domain(tmp_path):
    runner, client = FrozenRunner(), FakeClient()
    evaluator = candidate.EpisodeEvaluator(candidate.Args(), tmp_path)
    evaluator.eval_each_episode(runner, tmp_path, client, "episode")
    assert [next(key for key in ("reset", "add_buffer", "release_attempt") if request.get(key))
            if "observation/image" not in request else "infer" for request in client.requests] == ["reset", "add_buffer", "infer", "add_buffer", "infer"]
    buffers = [request for request in client.requests if request.get("add_buffer")]
    assert [request["images"].shape for request in buffers] == [(34, 1, 256, 256, 3), (16, 1, 256, 256, 3)]
    assert [request["exec_start_idx"] for request in buffers] == [33, 33]
    assert [request["eval_rpc"]["first_global_frame"] for request in buffers] == [0, 34]
    assert [request["eval_rpc"]["batch_seq"] for request in buffers] == [0, 1]
    for request in buffers:
        assert request["images"].dtype == np.uint8
        assert request["state"].dtype == np.float32
        np.testing.assert_array_equal(request["images"][0, 0], IMAGE)
        np.testing.assert_array_equal(request["state"][0], STATE)
    inferences = [request for request in client.requests if "observation/image" in request]
    assert [request["eval_rpc"]["last_global_frame"] for request in inferences] == [33, 49]
    assert [request["eval_rpc"]["replan_seq"] for request in inferences] == [0, 1]
    assert all(request["prompt"] == "冻结的官方 prompt" for request in inferences)


def test_zero_actions_when_reset_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(FrozenRunner, "failure", "reset")
    runner = FrozenRunner()
    evaluator = candidate.EpisodeEvaluator(candidate.Args(), tmp_path)
    with pytest.raises(RuntimeError, match="original-reset-stack"):
        evaluator.eval_each_episode(runner, tmp_path, FakeClient(), "episode")
    assert evaluator.actions_executed == 0
    assert not any(event[0] == "step" for event in runner.trace)


@pytest.mark.parametrize("failure", ["make", "reset", "step", "none"])
def test_episode_boundary_preserves_failure_and_closes(tmp_path, monkeypatch, failure):
    args = args_for(tmp_path)
    monkeypatch.setattr(FrozenRunner, "failure", failure)
    client = FakeClient()
    assert candidate.evaluate(args, runner_factory=FrozenRunner, client_factory=lambda args: client) == 2
    output = Path(args.attempt_output)
    assert (output / "BinFill:0.started.json").exists()
    assert not (output / "BinFill:0.terminal.json").exists()
    error = json.loads((output / "BinFill:0.infra.json").read_text())
    assert (f"original-{failure}-stack" in error["error"]) if failure != "none" else "完整观测" in error["error"]
    assert error["retryable"] is True
    assert error["uncertain_terminal"] is (failure in ("step", "none"))
    assert FrozenRunner.created[0].closed and client.closed
    assert not any(request.get("release_attempt") for request in client.requests)


@pytest.mark.parametrize("operation", ["reset", "add_buffer", "infer"])
def test_lost_response_never_retries_original_rpc(tmp_path, operation):
    args = args_for(tmp_path)
    client = FakeClient()
    client.drop = operation
    assert candidate.evaluate(args, runner_factory=FrozenRunner, client_factory=lambda args: client) == 2
    assert sum(bool(request.get(operation)) if operation != "infer" else "observation/image" in request for request in client.requests) == 1
    assert not any(request.get("release_attempt") for request in client.requests)
    assert json.loads((Path(args.attempt_output) / "BinFill:0.infra.json").read_text())["uncertain_rpc"]


def test_stale_ack_is_rejected_without_polling():
    with pytest.raises(candidate.UncertainRPC):
        candidate.require_ack({"reset_finished": False}, "reset_finished", {"episode_nonce": "x"})


@pytest.mark.parametrize("key", ["model_seed", "model_kind", "run_identity_sha256", "server_nonce"])
def test_metadata_wrong_before_environment(tmp_path, key):
    args = args_for(tmp_path)
    client = FakeClient()
    client.metadata[key] = "wrong"
    assert candidate.evaluate(args, runner_factory=FrozenRunner, client_factory=lambda args: client) == 2
    assert FrozenRunner.created == []
    assert client.requests == [] and client.closed


def test_video_error_keeps_terminal_and_next_episode(tmp_path, monkeypatch):
    args = args_for(tmp_path, 2)
    def fail_video(self, filename):
        assert list(Path(args.attempt_output).glob("*.terminal.json"))
        raise RuntimeError("video-only")
    monkeypatch.setattr(Recorder, "save_video", fail_video)
    client = FakeClient()
    assert candidate.evaluate(args, runner_factory=FrozenRunner, client_factory=lambda args: client) == 0
    output = Path(args.attempt_output)
    assert len(list(output.glob("*.terminal.json"))) == 2
    assert all(json.loads(path.read_text())["artifact_error"] for path in output.glob("*.artifact.json"))
    assert sum(bool(request.get("reset")) for request in client.requests) == 2


def test_close_failure_keeps_terminal(tmp_path, monkeypatch):
    args = args_for(tmp_path, 2)
    monkeypatch.setattr(FrozenRunner, "failure", "close")
    assert candidate.evaluate(args, runner_factory=FrozenRunner, client_factory=FakeClient) == 2
    assert (Path(args.attempt_output) / "BinFill:0.terminal.json").exists()
    assert not (Path(args.attempt_output) / "BinFill:1.started.json").exists()


def test_drain_only_between_episodes(tmp_path, monkeypatch):
    args = args_for(tmp_path, 2)
    args.drain_file = str(tmp_path / "drain")
    original = FrozenRunner.step
    def draining_step(self, action):
        Path(args.drain_file).touch()
        return original(self, action)
    monkeypatch.setattr(FrozenRunner, "step", draining_step)
    assert candidate.evaluate(args, runner_factory=FrozenRunner, client_factory=FakeClient) == 3
    summary = json.loads((Path(args.attempt_output) / "attempt.json").read_text())
    assert summary["completed_episode_keys"] == ["BinFill:0"]
    assert summary["unstarted_episode_keys"] == ["BinFill:1"]


def test_budget_before_first_action(tmp_path, monkeypatch):
    monkeypatch.setattr(FrozenRunner, "demo", 1841)
    runner, client = FrozenRunner(), FakeClient()
    with pytest.raises(candidate.ContractError, match="预算"):
        candidate.EpisodeEvaluator(candidate.Args(), tmp_path).eval_each_episode(runner, tmp_path, client, "episode")
    assert runner.count == 0
    assert len(client.requests) == 1


def test_fixed_official_tasks_and_payload_dtypes():
    assert TASK_NAME_LIST == ["BinFill", "StopCube", "PickXtimes", "SwingXtimes", "ButtonUnmask", "VideoUnmask",
                              "VideoUnmaskSwap", "ButtonUnmaskSwap", "PickHighlight", "VideoRepick", "VideoPlaceButton",
                              "VideoPlaceOrder", "MoveCube", "InsertPeg", "PatternLock", "RouteStick"]
    with pytest.raises(ValueError, match="uint8"):
        pack_buffer([IMAGE.astype(np.float32)], [STATE])
    with pytest.raises(ValueError, match="float32"):
        pack_buffer([IMAGE], [STATE.astype(np.float64)])
    state = EpisodeState()
    state.exec_start_idx = 33
    state.clear_buffers()
    assert state.exec_start_idx == 33


def test_atomic_receipt_rejects_overwrite(tmp_path):
    target = tmp_path / "terminal.json"
    candidate.write_record(target, {"outcome": "success"})
    with pytest.raises(FileExistsError):
        candidate.write_record(target, {"outcome": "fail"})
    assert json.loads(target.read_text()) == {"outcome": "success"}


def test_environment_step_original_exception_and_joint_state():
    from examples.robomme.env_runner import EnvRunner, pack_state
    runner = object.__new__(EnvRunner)
    def raise_original(action):
        raise OSError("original-driver-stack")
    runner.env = SimpleNamespace(step=raise_original)
    with pytest.raises(OSError, match="original-driver-stack"):
        runner.step(np.zeros(8))
    actual = pack_state(np.arange(7, dtype=np.float64), np.array([0.1, 0.2]))
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.array([0, 1, 2, 3, 4, 5, 6, 0.1], np.float32))


def test_transport_receive_timeout_is_bounded_and_single_send():
    client = object.__new__(candidate.EvaluationClient)
    sent, received = [], []
    def receive(*, timeout):
        received.append(timeout)
        raise TimeoutError("lost")
    client.timeout = 0.01
    client.ws = SimpleNamespace(send=sent.append, recv=receive)
    client.packer = SimpleNamespace(pack=lambda payload: b"packed")
    with pytest.raises(candidate.UncertainRPC):
        client.request({"reset": True})
    assert sent == [b"packed"] and received == [0.01]


@pytest.mark.parametrize("error,expected", [("contract_violation", candidate.ContractError),
                                           ("active_attempt", candidate.ContractError),
                                           ("runtime_failure", candidate.UncertainRPC)])
def test_server_error_classification(error, expected):
    client = object.__new__(candidate.EvaluationClient)
    client.timeout = 1
    client.ws = SimpleNamespace(recv=lambda timeout: b"error")
    client.codec = SimpleNamespace(unpackb=lambda data: {"error": error})
    with pytest.raises(expected):
        client._receive()


def test_real_cpu_websocket_roundtrip(tmp_path):
    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve

    args = args_for(tmp_path)
    server_client = FakeClient()

    def handler(websocket):
        websocket.send(msgpack_numpy.packb(server_client.metadata))
        for raw in websocket:
            request = msgpack_numpy.unpackb(raw)
            websocket.send(msgpack_numpy.packb(server_client.request(request)))
            if request.get("release_attempt"):
                return

    with serve(handler, "127.0.0.1", 0, compression=None, max_size=None) as server:
        args.port = server.socket.getsockname()[1]
        args.rpc_timeout_seconds = 2
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            assert candidate.evaluate(args, runner_factory=FrozenRunner) == 0
        finally:
            server.shutdown()
            thread.join(timeout=3)
    assert not thread.is_alive()
    assert sum(bool(request.get("reset")) for request in server_client.requests) == 1
    assert server_client.requests[-1]["release_attempt"] is True


def test_wrapper_returned_frame_domain_ignores_internal_physics(tmp_path):
    from examples.robomme.env_runner import EnvRunner

    physical_trace = []
    def observation(front_values):
        return {"front_rgb_list": [np.full_like(IMAGE, value) for value in front_values],
                "wrist_rgb_list": [WRIST] * len(front_values),
                "joint_state_list": [STATE[:7]] * len(front_values),
                "gripper_state_list": [STATE[7:]] * len(front_values)}

    def reset():
        # 固化 wrapper 返回记录：内部初始化和过滤不会变成额外历史帧。
        physical_trace.extend(["init_action", "NO RECORD", "demo_frame", "exec0"])
        return observation([5, 6]), {"task_goal": ["冻结的官方 prompt"]}

    def step(action):
        physical_trace.extend(["physical_step", "NO RECORD", "terminal_recursive_step"])
        return observation([12]), 0, True, False, {"status": "success"}

    runner = object.__new__(EnvRunner)
    runner.env_id = "MoveCube"
    runner.env = SimpleNamespace(reset=reset, step=step)
    client = FakeClient()
    evaluator = candidate.EpisodeEvaluator(candidate.Args(), tmp_path)
    assert evaluator.eval_each_episode(runner, tmp_path, client, "episode") == "success"
    history = client.requests[1]
    assert history["exec_start_idx"] == 1
    assert history["images"][:, 0, 0, 0, 0].tolist() == [5, 6]
    assert physical_trace == ["init_action", "NO RECORD", "demo_frame", "exec0",
                              "physical_step", "NO RECORD", "terminal_recursive_step"]
    assert evaluator.actions_executed == 1
    assert evaluator.recorder.frames == [(5, False), (6, False), (12, True)]
