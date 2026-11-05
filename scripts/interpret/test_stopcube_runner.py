"""StopCube 动作接管、真实续包、记录与本机命令身份的 CPU 定向测试。"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.interpret import run_local as local
from scripts.interpret import stopcube_runner as runner


class Recorder:
    def __init__(self, *args, fps=30):
        self.fps = fps

    def record(self, **kwargs):
        pass


class Client:
    def __init__(self):
        self.frames = []
        self.payloads = []

    def request(self, payload):
        rpc = payload["eval_rpc"]
        if payload.get("reset"):
            return {"reset_finished": True, "eval_rpc": rpc}
        if payload.get("add_buffer"):
            self.frames.append(rpc["frame_count"])
            self.payloads.append((payload["images"].copy(), payload["state"].copy(), dict(rpc)))
            return {"add_buffer_finished": True, "eval_rpc": rpc}
        # 每次边界给出可识别的新动作，确保恢复没有执行积压动作。
        actions = np.repeat((100 * rpc["replan_seq"] + np.arange(20))[:, None], 8, axis=1).astype(np.float32)
        return {"actions": actions, "eval_rpc": rpc}


def setup_sim(monkeypatch):
    def initial(self):
        self.env.elapsed = 7
        self.env.actions = []
        self.env.state = np.arange(8, dtype=np.float32)
        images = [np.full((256, 256, 3), index, dtype=np.uint8) for index in range(4)]
        return {"images": images, "wrist_images": images,
                "states": [self.env.state.copy() for _ in images], "task_goal": "按下按钮"}

    def step(self, action):
        self.env.actions.append(np.asarray(action).copy())
        self.env.state = np.asarray(action, dtype=np.float32).copy()
        self.env.elapsed += 1
        image = np.full((256, 256, 3), self.env.elapsed, dtype=np.uint8)
        return (image, image, self.env.state.copy()), len(self.env.actions) == 35, "fail"

    def snapshot(env):
        return {"env_step": env.elapsed, "tcp": [0.1, 0.2, float(env.elapsed)],
                "button_pressed": False, "cube_position": [0, 0, 0], "target_position": [1, 0, 0],
                "stop_timestep": None, "move_interval": 80, "target_pass": 3}

    monkeypatch.setattr(runner.EnvRunner, "get_init_obs", initial)
    monkeypatch.setattr(runner.EnvRunner, "step", step)
    monkeypatch.setattr(runner, "snapshot", snapshot)
    monkeypatch.setattr(runner.official, "RolloutRecorder", Recorder)


def make_runner():
    result = runner.TraceRunner.__new__(runner.TraceRunner)
    result.env_id, result.episode_id = "StopCube", 0
    result.env_seed, result.difficulty = 105, "easy"
    result.env = SimpleNamespace(elapsed=7)
    result.env.unwrapped = SimpleNamespace()
    return result


def run_episode(tmp_path, monkeypatch, mode, schedule=None):
    setup_sim(monkeypatch)
    env = make_runner()
    args = runner.Args(stopcube_mode=mode)
    evaluator = runner.TraceEvaluator(args, tmp_path)
    evaluator.config = {"StopCube:0": schedule} if schedule else {}
    client = Client()
    outcome = evaluator.eval_each_episode(env, tmp_path / "videos", client, "nonce")
    directory = tmp_path / "traces" / "nonce"
    records = [json.loads(line) for line in (directory / "step_trace.jsonl").read_text().splitlines()]
    receipt = json.loads((directory / "meta" / "hold_receipt.json").read_text())
    return env, evaluator, client, outcome, records, receipt


def test_hold_replays_prefix_and_resumes_latest_chunk(tmp_path, monkeypatch):
    hold = {"takeover_step": 12, "release_step": 39, "hold_qpos": [-1] * 8}
    env, evaluator, client, outcome, rows, receipt = run_episode(tmp_path, monkeypatch, "hold_extended", hold)
    assert outcome == "fail" and evaluator.actions_executed == 35
    assert client.frames == [4, 16, 16]
    assert np.array_equal(env.env.actions[:5], np.repeat(np.arange(5)[:, None], 8, axis=1))
    assert np.array_equal(env.env.actions[5:32], np.full((27, 8), -1))
    assert np.array_equal(env.env.actions[32:], np.repeat(np.arange(200, 203)[:, None], 8, axis=1))
    # 续包逐帧来自实际执行后的观测，接管期间的真实状态照常交付。
    assert client.payloads[1][0][:, 0, 0, 0, 0].tolist() == list(range(8, 24))
    assert np.array_equal(client.payloads[2][1], np.full((16, 8), -1))
    assert rows[32]["infer_seq"] == 2 and rows[32]["row_in_chunk"] == 0
    assert rows[32]["command_action"] == [200] * 8 == rows[32]["executed_action"]
    assert rows[6]["command_action"] == [6] * 8 and rows[6]["executed_action"] == [-1] * 8
    assert rows[6]["tcp_before"] == [0.1, 0.2, 13.0]
    assert rows[6]["tcp_after"] == [0.1, 0.2, 14.0]
    assert receipt["D"] == 3 and receipt["initial_env_step"] == 7
    assert receipt["episode_id"] == 0 and receipt["env_seed"] == 105 and receipt["difficulty"] == "easy"
    assert receipt["actual_takeover_step"] == 12 and receipt["actual_release_step"] == 39
    assert receipt["video_fps"] == 30 and receipt["video_exec0_frame_index"] == 3
    assert receipt["task_success"] == 0 and receipt["extra"]["button_position"] is None
    required = {"event", "episode_key", "trajectory_kind", "infer_seq", "row_in_chunk", "action_index",
                "env_step_before", "env_step_after", "command_action", "executed_action", "state_before",
                "state_after", "tcp_before", "tcp_after", "button_pressed_before", "button_pressed_after",
                "cube_position_before", "cube_position_after", "target_position_before", "target_position_after",
                "stop_timestep", "move_interval", "target_pass"}
    assert all(required <= row.keys() for row in rows)


def test_normal_matches_official_loop(tmp_path, monkeypatch):
    env, evaluator, client, outcome, rows, receipt = run_episode(tmp_path, monkeypatch, "normal")
    baseline = runner.EnvRunner.__new__(runner.EnvRunner)
    baseline.env_id, baseline.env = "StopCube", SimpleNamespace(elapsed=7)
    baseline_client = Client()
    official = runner.official.EpisodeEvaluator(runner.Args(), tmp_path)
    expected = official.eval_each_episode(baseline, tmp_path / "videos", baseline_client, "nonce")
    assert outcome == expected and evaluator.actions_executed == official.actions_executed
    assert np.array_equal(env.env.actions, baseline.env.actions)
    for actual, reference in zip(client.payloads, baseline_client.payloads, strict=True):
        assert np.array_equal(actual[0], reference[0]) and np.array_equal(actual[1], reference[1])
        assert actual[2] == reference[2]
    assert all(row["command_action"] == row["executed_action"] for row in rows)
    assert receipt["trajectory_kind"] == "normal" and receipt["hold_qpos"] is None


@pytest.mark.parametrize("release", [38, 12])
def test_invalid_boundary_rejected(tmp_path, monkeypatch, release):
    setup_sim(monkeypatch)
    env = make_runner()
    env.begin_trace(runner.Args(stopcube_mode="hold_extended"), "bad", tmp_path,
                    {"StopCube:0": {"takeover_step": 12, "release_step": release, "hold_qpos": [0] * 8}})
    try:
        with pytest.raises(runner.official.ContractError):
            env.get_init_obs()
    finally:
        env.trace_stream.close()


def test_client_command_and_content_identity(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema_version": 1, "episodes": {}}))
    spec = {"repo": "/repo", "sim_python": "/sim/python", "run_identity_sha256": "a" * 64,
            "model_kind": "stage2_pool4_concat", "checkpoint": "/ckpt", "policy_config": "cfg", "model_seed": 42}
    grant = tmp_path / "grant.json"
    grant.write_text(json.dumps({"chunk_id": "chunk"}))
    arguments = (grant, tmp_path / "out", "nonce", 9000, tmp_path / "drain")
    assert local.build_client_command(spec, *arguments) == local.RW.client_command(spec, *arguments)
    baseline = local.run_identity(spec, {"sha256": "b" * 64})
    changed = dict(spec, stopcube_mode="normal", stopcube_config=str(config))
    command = local.build_client_command(changed, *arguments)
    assert command[1] == "/repo/scripts/interpret/stopcube_runner.py"
    assert command[-4:] == ["--stopcube-mode", "normal", "--stopcube-config", str(config)]
    first = local.run_identity(changed, {"sha256": "b" * 64})
    assert first != baseline
    config.write_text(json.dumps({"schema_version": 1, "episodes": {}, "note": "改变内容"}))
    assert first != local.run_identity(changed, {"sha256": "b" * 64})
    assert local.run_identity(dict(changed, stopcube_mode="hold_extended"), {"sha256": "b" * 64}) != first


def test_load_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 1, "episodes": {"StopCube:0": {}}}))
    assert runner.load_config(path) == {"StopCube:0": {}}
