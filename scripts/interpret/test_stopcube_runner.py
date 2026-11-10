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


def test_hold_preserves_json_command_precision(tmp_path, monkeypatch):
    # 前七维是物理 float32 状态的精确升格；末维保留实际正常命令的 float64 数值。
    joints = np.asarray([0.1234567, -0.2345678, 0.3456789, -1.456789, 0.5678912, 1.6789123,
                         -0.7891234], dtype=np.float32).astype(np.float64).tolist()
    qpos = joints + [-1.0039340752248764]
    hold = {"takeover_step": 12, "release_step": 39, "hold_qpos": qpos}
    env, _, _, _, rows, receipt = run_episode(tmp_path, monkeypatch, "hold_extended", hold)
    assert receipt["hold_qpos"] == qpos
    assert all(action.dtype == np.float64 and action.tolist() == qpos for action in env.env.actions[5:32])
    assert all(row["executed_action"] == qpos for row in rows[5:32])


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


class OracleClient(Client):
    def __init__(self):
        super().__init__()
        self.calls = []

    def request(self, payload):
        self.calls.append(payload)
        if payload.get("add_buffer"):
            from scripts.evaluation.runtime_contract import payload_sha256
            assert payload["eval_rpc"]["payload_sha256"] == payload_sha256(payload)
        result = super().request(payload)
        if not payload.get("reset") and not payload.get("add_buffer"):
            if payload["oracle_context"]["mode"] == "oracle_query":
                result.update(query_actions=np.repeat(result["actions"][None], 10, axis=0),
                              query_record=f"query_{payload['eval_rpc']['replan_seq']}.npz",
                              query_seq=payload["eval_rpc"]["replan_seq"])
        return result


def oracle_episode(tmp_path, monkeypatch, mode="oracle_query", *, interval=4, release=9,
                   kind="high_probability", terminal_steps=80, trigger=None, table_steps=100, horizon=80):
    setup_sim(monkeypatch)
    old_initial = runner.EnvRunner.get_init_obs

    def initial(self):
        result = old_initial(self)
        self.env.elapsed = 1
        return result

    def step(self, action):
        self.env.actions.append(np.asarray(action).copy())
        self.env.state = np.asarray(action, dtype=np.float32).copy()
        self.env.elapsed += 1
        image = np.full((256, 256, 3), self.env.elapsed, dtype=np.uint8)
        return (image, image, self.env.state.copy()), len(self.env.actions) == terminal_steps, "fail"

    def snapshot(env):
        return {"env_step": env.elapsed, "tcp": [0, 0, 0.15],
                "button_pressed": trigger is not None and env.elapsed >= trigger,
                "cube_position": [0, 0, 0], "target_position": [1, 0, 0], "stop_timestep": None,
                "move_interval": horizon, "target_pass": 1}

    monkeypatch.setattr(runner.EnvRunner, "get_init_obs", initial)
    monkeypatch.setattr(runner.EnvRunner, "step", step)
    monkeypatch.setattr(runner, "snapshot", snapshot)
    env = make_runner()
    table = {"env_seed": 105, "first_env_step": 1, "last_env_step": table_steps,
             "terminal_env_step": table_steps + 1, "move_interval": horizon, "target_pass": 1,
             "reach_step": 5, "z_hover": 0.15,
             "actions": np.repeat(np.arange(1, table_steps + 1)[:, None], 8, axis=1).tolist()}
    args = runner.Args(stopcube_mode=mode, query_interval=interval, calibration_kind=kind,
                       calibration_model_id="motion")
    evaluator = runner.TraceEvaluator(args, tmp_path)
    evaluator.config = {"StopCube:0": table}
    evaluator.calibration = {"StopCube:0": {kind: {"status": "selected", "release_step": release,
                             "stop_after_release": None if kind == "fixed_release" else 48}}}
    client = OracleClient()
    outcome = evaluator.eval_each_episode(env, tmp_path / "videos", client, "oracle")
    receipt = json.loads((tmp_path / "traces/oracle/meta/hold_receipt.json").read_text())
    rows = [json.loads(line) for line in (tmp_path / "traces/oracle/step_trace.jsonl").read_text().splitlines()]
    return env, evaluator, client, outcome, receipt, rows


@pytest.mark.parametrize("interval", [4, 16])
def test_oracle_query_real_frames_and_actions_to_terminal(tmp_path, monkeypatch, interval):
    env, evaluator, client, outcome, receipt, rows = oracle_episode(
        tmp_path, monkeypatch, interval=interval, terminal_steps=35, horizon=28)
    assert outcome == "fail" and evaluator.actions_executed == 35
    assert np.array_equal(env.env.actions, np.repeat(np.arange(1, 36)[:, None], 8, axis=1))
    assert receipt["query_steps"] == list(range(1, 29, interval))
    assert receipt["query_points"] == len(receipt["query_steps"]) and receipt["execution_infers"] == 0
    assert client.frames == [4] + [interval] * (receipt["query_points"] - 1)
    if len(client.payloads) > 1:
        assert client.payloads[1][0][:, 0, 0, 0, 0].tolist() == list(range(2, interval + 2))
    infers = [c for c in client.calls if "observation/image" in c]
    assert [p["oracle_context"]["pre_reach"] for p in infers] == [t < 5 for t in receipt["query_steps"]]
    assert all(row["control_source"] == "oracle" for row in rows)
    assert all(row["command_action"] == row["executed_action"] for row in rows)


def test_oracle_table_exhaustion_rejects_without_repeating(tmp_path, monkeypatch):
    with pytest.raises(runner.official.ContractError, match="越界"):
        oracle_episode(tmp_path, monkeypatch, terminal_steps=35, table_steps=10)
    rows = (tmp_path / "traces/oracle/step_trace.jsonl").read_text().splitlines()
    assert len(rows) == 10


def test_calibration_continuous_history_new_infer_and_48_steps(tmp_path, monkeypatch):
    env, evaluator, client, outcome, receipt, rows = oracle_episode(tmp_path, monkeypatch, mode="oracle_calibrate", release=9)
    assert outcome == "calibration_stop" and evaluator.terminal_observed
    assert evaluator.actions_executed == 8 + 48
    assert receipt["actual_release_step"] == 9 and receipt["calibration_stop"] == "release_48_steps"
    assert receipt["history_batches"] == 5 and receipt["execution_infers"] == 3
    assert client.frames == [4, 4, 4, 16, 16]
    assert [c["oracle_context"]["env_step"] for c in client.calls if "observation/image" in c] == [9, 25, 41]
    assert [c.get("add_buffer", False) for c in client.calls[1:4]] == [True, True, True]
    assert np.array_equal(env.env.actions[:8], np.repeat(np.arange(1, 9)[:, None], 8, axis=1))
    assert np.array_equal(env.env.actions[8:24], np.repeat(np.arange(16)[:, None], 8, axis=1))
    assert rows[8]["control_source"] == "model" and rows[8]["row_in_chunk"] == 0
    assert receipt["first_release_prediction"][0] == [0] * 8


def test_calibration_button_and_fixed_release_terminal(tmp_path, monkeypatch):
    _, _, _, outcome, receipt, _ = oracle_episode(tmp_path, monkeypatch, mode="oracle_calibrate", trigger=14)
    assert outcome == "calibration_stop" and receipt["button_trigger_step"] == 14
    assert receipt["calibration_stop"] == "button_trigger"
    fixed = tmp_path / "fixed"
    _, evaluator, _, outcome, receipt, _ = oracle_episode(fixed, monkeypatch, mode="oracle_calibrate", kind="fixed_release", trigger=14)
    assert outcome == "fail" and evaluator.actions_executed == 80
    assert receipt["calibration_stop"] == "official_terminal"


def test_oracle_content_identity_and_calibration_cli(tmp_path):
    config, plan = tmp_path / "table.json", tmp_path / "calibration.json"
    config.write_text(json.dumps({"schema_version": 1, "episodes": {}}))
    plan.write_text(json.dumps({"schema_version": 1, "episodes": {}}))
    spec = {"repo": "/repo", "sim_python": "/sim/python", "run_identity_sha256": "a" * 64,
            "model_kind": "history", "checkpoint": "/ckpt", "policy_config": "cfg", "model_seed": 42,
            "stopcube_mode": "oracle_calibrate", "stopcube_config": str(config),
            "calibration_plan": str(plan), "calibration_kind": "fixed_release", "model_id": "ctx"}
    grant = tmp_path / "grant.json"
    grant.write_text(json.dumps({"chunk_id": "chunk"}))
    command = local.build_client_command(spec, grant, tmp_path / "out", "nonce", 9000, tmp_path / "drain")
    assert command[-6:] == ["--calibration-plan", str(plan), "--calibration-kind", "fixed_release", "--calibration-model-id", "ctx"]
    first = local.run_identity(spec, {"sha256": "b" * 64})
    plan.write_text(json.dumps({"schema_version": 1, "episodes": {}, "note": "选点已变"}))
    assert first != local.run_identity(spec, {"sha256": "b" * 64})


def test_oracle_collect_separates_rpc_counts(tmp_path):
    output = tmp_path / "client"
    receipt_dir = output / "traces/nonce/meta"
    receipt_dir.mkdir(parents=True)
    attn = tmp_path / "attn/nonce"
    attn.mkdir(parents=True)
    table = tmp_path / "table.json"
    table.write_text(json.dumps({"episodes": {"StopCube:0": {"first_env_step": 1, "move_interval": 8, "target_pass": 1}}}))
    (receipt_dir / "hold_receipt.json").write_text(json.dumps(
        {"history_batches": 2, "query_points": 2, "execution_infers": 0, "query_steps": [1, 5]}))
    (output / "StopCube:0.terminal.json").write_text(json.dumps(
        {"episode_key": "StopCube:0", "episode_nonce": "nonce", "rpc_seq": 5, "outcome": "fail"}))
    for index in range(2):
        (attn / f"query_{index}.npz").touch()
    spec = {"run_root": str(tmp_path / "run"), "attn_out": str(tmp_path / "attn"),
            "stopcube_mode": "oracle_query", "stopcube_config": str(table)}
    results = local.collect(spec, tmp_path, output, [{"episode_key": "StopCube:0", "task_name": "StopCube", "episode_id": 0}])
    assert results[0]["cadence_ok"] and results[0]["npz"] == 0 and results[0]["query_npz"] == 2


def test_oracle_service_reused_and_clients_bounded(tmp_path, monkeypatch):
    calls, stopped = [], []

    class Process:
        def __init__(self, command):
            self.command = command
            self.server = command[0] == "server"
            self.returncode = None if self.server else 0

        def poll(self):
            return self.returncode

    def popen(command, **kwargs):
        process = Process(command)
        calls.append(process)
        return process

    monkeypatch.setattr(local, "pick_port", lambda value: 9001)
    monkeypatch.setattr(local, "build_environment", lambda spec: {})
    monkeypatch.setattr(local, "wait_ready", lambda *args: None)
    monkeypatch.setattr(local, "build_server_command", lambda spec, nonce, port: ["server", nonce])
    monkeypatch.setattr(local, "build_client_command", lambda spec, grant, output, nonce, port, drain:
                        ["client", nonce, str(grant), str(output)])
    monkeypatch.setattr(local.RW, "stop_process_tree", lambda process: stopped.append(process) if process else None)
    episodes = [{"episode_key": f"StopCube:{index}"} for index in range(12)]
    results = list(local.run_oracle_chunks({"run_root": str(tmp_path), "repo": "/repo", "run_identity_sha256": "a" * 64},
                                         [("chunk", episodes)], popen=popen))
    assert len(calls) == 4 and sum(p.server for p in calls) == 1
    assert [len(row[3]) for row in results] == [5, 5, 2]
    assert len({p.command[1] for p in calls}) == 1
    assert len({p.command[3] for p in calls if not p.server}) == 3
    assert all(len(json.loads(Path(p.command[2]).read_text())["episodes"]) <= 5 for p in calls if not p.server)
    assert set(stopped) == set(calls)


def test_oracle_service_cleanup_on_consumer_close(tmp_path, monkeypatch):
    stopped = []
    process = SimpleNamespace(returncode=0, poll=lambda: 0)
    monkeypatch.setattr(local, "pick_port", lambda value: 9001)
    monkeypatch.setattr(local, "build_environment", lambda spec: {})
    monkeypatch.setattr(local, "wait_ready", lambda *args: None)
    monkeypatch.setattr(local, "build_server_command", lambda *args: ["server"])
    monkeypatch.setattr(local, "build_client_command", lambda *args: ["client"])
    monkeypatch.setattr(local.RW, "stop_process_tree", lambda proc: stopped.append(proc) if proc else None)
    # 服务应存活到客户端退出；消费方提前关闭生成器仍必须收掉该服务。
    server = SimpleNamespace(returncode=None, poll=lambda: None)
    processes = iter([server, process])
    stream = local.run_oracle_chunks({"run_root": str(tmp_path), "repo": "/repo", "run_identity_sha256": "a" * 64},
                                    [("chunk", [{"episode_key": "StopCube:0"}])], popen=lambda *a, **k: next(processes))
    assert next(stream)[2] == 0
    stream.close()
    assert stopped == [process, server]


def test_query_batch_size_command_and_identity(tmp_path, monkeypatch):
    table = tmp_path / "table.json"
    table.write_text(json.dumps({"schema_version": 1, "episodes": {}}))
    spec = {"repo": "/repo", "policy_python": "/policy/python", "sim_python": "/sim/python",
            "stage2_python": "/stage2/python", "run_identity_sha256": "a" * 64, "cuda": "0",
            "model_kind": "history", "checkpoint": "/ckpt", "policy_config": "cfg", "model_seed": 42,
            "stopcube_mode": "oracle_query", "stopcube_config": str(table), "attn_out": "/out",
            "run_root": "/run", "manifest": "/manifest.json", "xla_mem_fraction": 0.55}
    implicit = local.run_identity(spec, {"sha256": "b" * 64})
    explicit = dict(spec, query_batch_size=10)
    assert implicit == local.run_identity(explicit, {"sha256": "b" * 64})
    small = dict(spec, query_batch_size=2)
    assert implicit != local.run_identity(small, {"sha256": "b" * 64})
    command = local.build_server_command(small, "nonce", 9000)
    assert command[command.index("--query-batch-size") + 1] == "2"
    assert command[command.index("--query-record") + 1] == "10"
    assert "--query-batch-size" not in local.build_server_command(spec, "nonce", 9000)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(small))
    assert local.load_spec(spec_path)["query_batch_size"] == 2
    spec_path.write_text(json.dumps(dict(small, query_batch_size=3)))
    with pytest.raises(SystemExit, match="query_batch_size"):
        local.load_spec(spec_path)


# ---------------- speed_donor：改速度、回放后悬停、horizon 收尾 ----------------

def donor_table(t_hold=10, horizon=40, **changes):
    actions = [[1000.0 + j] * 8 for j in range(1, t_hold)]
    table = {"env_seed": 105, "move_interval_override": 120, "stop_time_override": 99, "first_env_step": 1,
             "actions": actions, "t_hold": t_hold, "hold_qpos": actions[-1], "horizon": horizon,
             "orig_move_interval": 80, "target_pass": 3}
    return table | changes


def donor_sim(monkeypatch, *, terminal_steps=10_000, trigger=None):
    """假环境：步前值从 1 起，逐步记下当时的 move_interval／stop_time，供检查改写发生在第一步之前。"""
    setup_sim(monkeypatch)
    old_initial = runner.EnvRunner.get_init_obs

    def initial(self):
        result = old_initial(self)
        self.env.elapsed = 1
        self.env.seen = []
        return result

    def step(self, action):
        u = self.env.unwrapped
        self.env.seen.append((u.move_interval, u.stop_time))
        self.env.actions.append(np.asarray(action).copy())
        self.env.state = np.asarray(action, dtype=np.float32).copy()
        self.env.elapsed += 1
        image = np.full((256, 256, 3), self.env.elapsed % 256, dtype=np.uint8)
        return (image, image, self.env.state.copy()), len(self.env.actions) == terminal_steps, "fail"

    def snapshot(env):
        u = env.unwrapped
        return {"env_step": env.elapsed, "tcp": [0, 0, 0.15],
                "button_pressed": trigger is not None and env.elapsed >= trigger,
                "cube_position": [float(env.elapsed), 0, 0], "target_position": [1, 0, 0], "stop_timestep": None,
                "move_interval": u.move_interval, "target_pass": u.stop_time}

    monkeypatch.setattr(runner.EnvRunner, "get_init_obs", initial)
    monkeypatch.setattr(runner.EnvRunner, "step", step)
    monkeypatch.setattr(runner, "snapshot", snapshot)


def donor_runner():
    env = make_runner()
    env.env.unwrapped = SimpleNamespace(move_interval=80, stop_time=3, cube_half_size=0.02)
    return env


def donor_episode(tmp_path, monkeypatch, table=None, **sim):
    donor_sim(monkeypatch, **sim)
    env = donor_runner()
    evaluator = runner.TraceEvaluator(runner.Args(stopcube_mode="speed_donor"), tmp_path)
    evaluator.config = {"StopCube:0": table or donor_table()}
    client = Client()
    outcome = evaluator.eval_each_episode(env, tmp_path / "videos", client, "donor")
    directory = tmp_path / "traces" / "donor"
    rows = [json.loads(line) for line in (directory / "step_trace.jsonl").read_text().splitlines()]
    receipt = json.loads((directory / "meta" / "hold_receipt.json").read_text())
    donor = json.loads((directory / "meta" / "speed_donor_receipt.json").read_text())
    return env, evaluator, client, outcome, rows, receipt, donor


def test_speed_donor_replay_hold_and_horizon(tmp_path, monkeypatch):
    env, evaluator, client, outcome, rows, receipt, donor = donor_episode(tmp_path, monkeypatch)
    # 步前值 1..39 共 39 步，env_step_after 达到 horizon=40 即收尾，不算 infra。
    assert outcome == runner.SPEED_DONOR_HORIZON and evaluator.terminal_observed
    assert evaluator.actions_executed == 39 and rows[-1]["env_step_after"] == 40
    # 推理照常按 16 步节奏请求（服务端靠它编码 motion 窗口）。
    assert client.frames == [4, 16, 16]
    # 改写发生在第一步之前，全程生效。
    assert env.env.seen == [(120, 99)] * 39
    # 下标对齐：步前值 j（1..t_hold-1）执行 actions[j-1]，j>=t_hold 执行 hold_qpos。
    for row in rows:
        j = row["env_step_before"]
        expected = [1000.0 + j] * 8 if j < 10 else [1009.0] * 8
        assert row["executed_action"] == expected
    assert np.array_equal(env.env.actions[:9], np.repeat((1000.0 + np.arange(1, 10))[:, None], 8, axis=1))
    assert np.array_equal(env.env.actions[9:], np.full((30, 8), 1009.0))
    # 模型命令照常记录，与执行动作分列。
    assert rows[0]["command_action"] == [0] * 8 and rows[16]["command_action"] == [100] * 8
    assert rows[5]["move_interval"] == 120 and rows[5]["target_pass"] == 99
    assert {"cube_position_before", "cube_position_after", "target_position_after", "button_pressed_after",
            "env_step_before"} <= rows[0].keys()
    assert receipt["trajectory_kind"] == "speed_donor" and receipt["outcome"] == runner.SPEED_DONOR_HORIZON
    assert donor == donor | {"orig_move_interval": 80, "move_interval_applied": 120, "orig_stop_time": 3,
                             "stop_time_applied": 99, "t_hold": 10, "horizon": 40, "cube_half_size": 0.02,
                             "button_ever_pressed": False, "first_button_step": None, "last_env_step": 40,
                             "termination": "horizon", "outcome": runner.SPEED_DONOR_HORIZON,
                             "actions_executed": 39, "first_env_step": 1, "target_pass": 3, "env_seed": 105}


def test_speed_donor_button_and_env_terminal(tmp_path, monkeypatch):
    _, _, _, outcome, _, _, donor = donor_episode(tmp_path, monkeypatch, trigger=15, terminal_steps=20)
    assert outcome == "fail"
    assert donor["button_ever_pressed"] is True and donor["first_button_step"] == 15
    assert donor["termination"] == "env_terminal:fail" and donor["last_env_step"] == 21


@pytest.mark.parametrize("changes", [
    {"env_seed": 106},
    {"orig_move_interval": 60},
    {"target_pass": 4},
    {"first_env_step": 2},
    {"actions": [[0.0] * 7] * 9},
    {"actions": [[0.0] * 8] * 8},
    {"hold_qpos": [0.0] * 8},
    {"move_interval_override": 70},
    {"stop_time_override": 0},
    {"horizon": 200, "stop_time_override": 1},
    {"t_hold": 1, "actions": [], "hold_qpos": [0.0] * 8},
    {"actions": [[float("nan")] * 8] * 9},
    {"horizon": 40.0},
])
def test_speed_donor_rejects_bad_table(tmp_path, monkeypatch, changes):
    donor_sim(monkeypatch)
    env = donor_runner()
    env.begin_trace(runner.Args(stopcube_mode="speed_donor"), "bad", tmp_path, {"StopCube:0": donor_table() | changes})
    try:
        with pytest.raises(runner.official.ContractError):
            env.get_init_obs()
        # 校验失败时不得已改写环境。
        assert (env.env.unwrapped.move_interval, env.env.unwrapped.stop_time) == (80, 3)
    finally:
        env.trace_stream.close()


def test_speed_donor_missing_field_or_episode(tmp_path, monkeypatch):
    donor_sim(monkeypatch)
    table = donor_table()
    del table["horizon"]
    env = donor_runner()
    env.begin_trace(runner.Args(stopcube_mode="speed_donor"), "a", tmp_path, {"StopCube:0": table})
    try:
        with pytest.raises(runner.official.ContractError, match="缺字段"):
            env.get_init_obs()
    finally:
        env.trace_stream.close()
    with pytest.raises(runner.official.ContractError, match="缺局"):
        donor_runner().begin_trace(runner.Args(stopcube_mode="speed_donor"), "b", tmp_path, {})


def test_stopcube_modes_match_driver():
    assert local.STOPCUBE_MODES == runner.STOPCUBE_MODES


def stopcube_plfull_spec(tmp_path, **changes):
    from scripts.evaluation import manifest as M
    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    (meta / "record_dataset_StopCube_metadata.json").write_text("[]")
    rows = [{"task_name": "StopCube", "episode_id": ep, "env_seed": 2500000 + ep, "difficulty": "easy"}
            for ep in (0, 71)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(M.make_manifest(rows, metadata_sha256={"m": "0" * 64}, chunk_size=5,
                                                   official=False)))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema_version": 1, "episodes": {}}))
    values = dict(repo="/repo", policy_python="/policy/python", sim_python="/sim/python",
                  stage2_python="/stage2/python", checkpoint="/ckpt", policy_config="cfg",
                  model_kind="stage2_pool4_concat", cuda="0", attn_out=str(tmp_path / "attn"),
                  run_root=str(tmp_path / "run"), manifest=str(manifest), xla_mem_fraction=0.5, model_seed=42,
                  profile="plfull", plfull_metadata_dir=str(meta), save_attn=False,
                  stopcube_mode="normal", stopcube_config=str(config), run_identity_sha256="a" * 64)
    return values | changes


def load_stopcube(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return local.load_spec(path)


@pytest.mark.parametrize("mode", ["normal", "speed_donor"])
def test_plfull_accepts_normal_and_speed_donor(tmp_path, mode):
    spec = load_stopcube(tmp_path, stopcube_plfull_spec(
        tmp_path, stopcube_mode=mode, record_motion=str(tmp_path / "record"),
        ablate="motion_swap_exact:/donors", swap_map={"StopCube:0": "StopCube:20000000"}))
    grant = tmp_path / "grant.json"
    grant.write_text(json.dumps({"chunk_id": "c"}))
    command = local.build_client_command(spec, grant, tmp_path / "out", "nonce", 9000, tmp_path / "drain")
    assert command[1] == "/repo/scripts/interpret/stopcube_runner.py"
    i = command.index("--profile")
    assert command[i:i + 4] == ["--profile", "plfull", "--plfull-metadata-dir", spec["plfull_metadata_dir"]]
    j = command.index("--stopcube-mode")
    assert command[j:j + 4] == ["--stopcube-mode", mode, "--stopcube-config", spec["stopcube_config"]]
    server = local.build_server_command(spec, "nonce", 9000, episodes=[{"episode_key": "StopCube:0"}])
    assert server[server.index("--record-motion") + 1] == str(tmp_path / "record")
    assert server[server.index("--swap-map") + 1] == "StopCube:0=StopCube:20000000"
    assert "--no-save-attn" in server
    # 模式进入身份
    other = "speed_donor" if mode == "normal" else "normal"
    assert local.run_identity(spec, {"sha256": "b" * 64}) != local.run_identity(
        dict(spec, stopcube_mode=other), {"sha256": "b" * 64})


@pytest.mark.parametrize("changes", [
    {"stopcube_mode": "hold_extended"}, {"stopcube_mode": "oracle_query"}, {"stopcube_mode": "oracle_calibrate"},
    {"stopcube_mode": "speed_donor", "stopcube_config": None}, {"stopcube_mode": "fast"},
])
def test_plfull_rejects_other_stopcube_modes(tmp_path, changes):
    with pytest.raises(SystemExit):
        load_stopcube(tmp_path, stopcube_plfull_spec(tmp_path, **changes))


def test_speed_donor_without_plfull(tmp_path):
    spec = stopcube_plfull_spec(tmp_path, stopcube_mode="speed_donor")
    del spec["profile"], spec["plfull_metadata_dir"]
    loaded = load_stopcube(tmp_path, spec)
    grant = tmp_path / "grant.json"
    grant.write_text(json.dumps({"chunk_id": "c"}))
    command = local.build_client_command(loaded, grant, tmp_path / "out", "nonce", 9000, tmp_path / "drain")
    assert "--profile" not in command and command[-4:-2] == ["--stopcube-mode", "speed_donor"]


# ---------------- normal 关超时：只改 stop_time、跑到 horizon、无条目逐位不变 ----------------

def timeout_table(**changes):
    table = {"env_seed": 105, "orig_move_interval": 80, "target_pass": 3, "stop_time_override": 99, "horizon": 40}
    return table | changes


def timeout_episode(tmp_path, monkeypatch, config, **sim):
    donor_sim(monkeypatch, **sim)
    env = donor_runner()
    evaluator = runner.TraceEvaluator(runner.Args(stopcube_mode="normal"), tmp_path)
    evaluator.config = config
    client = Client()
    outcome = evaluator.eval_each_episode(env, tmp_path / "videos", client, "toff")
    directory = tmp_path / "traces" / "toff"
    rows = [json.loads(line) for line in (directory / "step_trace.jsonl").read_text().splitlines()]
    path = directory / "meta" / "timeout_off_receipt.json"
    receipt = json.loads(path.read_text()) if path.exists() else None
    return env, evaluator, client, outcome, rows, receipt


def test_timeout_off_only_stop_time_and_horizon(tmp_path, monkeypatch):
    env, evaluator, client, outcome, rows, receipt = timeout_episode(
        tmp_path, monkeypatch, {"StopCube:0": timeout_table()})
    # env_step_after 达到 horizon=40 即正常收尾，终局值可辨识、不算 infra。
    assert outcome == runner.TIMEOUT_OFF_HORIZON and evaluator.terminal_observed
    assert evaluator.actions_executed == 39 and rows[-1]["env_step_after"] == 40
    # 第一步之前只改 stop_time，move_interval 保持原值。
    assert env.env.seen == [(80, 99)] * 39
    # 动作不替换：执行动作即模型命令，推理节奏同官方。
    assert all(row["executed_action"] == row["command_action"] for row in rows)
    assert client.frames == [4, 16, 16]
    assert not (tmp_path / "traces" / "toff" / "meta" / "speed_donor_receipt.json").exists()
    assert receipt == receipt | {"orig_stop_time": 3, "stop_time_applied": 99, "move_interval": 80, "horizon": 40,
                                 "termination": "horizon", "first_button_step": None, "last_env_step": 40,
                                 "outcome": runner.TIMEOUT_OFF_HORIZON, "actions_executed": 39, "env_seed": 105}


def test_timeout_off_env_terminal(tmp_path, monkeypatch):
    _, _, _, outcome, _, receipt = timeout_episode(tmp_path, monkeypatch, {"StopCube:0": timeout_table()},
                                                   trigger=15, terminal_steps=20)
    assert outcome == "fail"
    assert receipt["termination"] == "env_terminal:fail" and receipt["first_button_step"] == 15
    assert receipt["button_ever_pressed"] is True and receipt["last_env_step"] == 21


def test_normal_without_entry_unchanged(tmp_path, monkeypatch):
    # 表里只有别的局：本局不改写、不按 horizon 收尾、不写关超时回执。
    env, evaluator, _, outcome, rows, receipt = timeout_episode(
        tmp_path, monkeypatch, {"StopCube:5": timeout_table()}, terminal_steps=50)
    assert outcome == "fail" and evaluator.actions_executed == 50 and receipt is None
    assert env.env.seen == [(80, 3)] * 50 and env.timeout_off is None


@pytest.mark.parametrize("changes", [
    {"env_seed": 106}, {"orig_move_interval": 60}, {"target_pass": 4}, {"stop_time_override": 0},
    {"horizon": 400, "stop_time_override": 4}, {"horizon": 40.0}, {"horizon": 1},
])
def test_timeout_off_rejects_bad_entry(tmp_path, monkeypatch, changes):
    donor_sim(monkeypatch)
    env = donor_runner()
    env.begin_trace(runner.Args(stopcube_mode="normal"), "bad", tmp_path, {"StopCube:0": timeout_table(**changes)})
    try:
        with pytest.raises(runner.official.ContractError):
            env.get_init_obs()
        assert (env.env.unwrapped.move_interval, env.env.unwrapped.stop_time) == (80, 3)
    finally:
        env.trace_stream.close()


def test_timeout_off_missing_field(tmp_path, monkeypatch):
    donor_sim(monkeypatch)
    table = timeout_table()
    del table["horizon"]
    env = donor_runner()
    env.begin_trace(runner.Args(stopcube_mode="normal"), "a", tmp_path, {"StopCube:0": table})
    try:
        with pytest.raises(runner.official.ContractError, match="缺字段"):
            env.get_init_obs()
    finally:
        env.trace_stream.close()
