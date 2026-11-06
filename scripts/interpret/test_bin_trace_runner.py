"""Swap 轨迹客户端与 run_local 开关的 CPU 测试：假环境、假客户端，不导入 robomme。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import manifest as M
from scripts.interpret import bin_trace_runner as runner
from scripts.interpret import rollout_offline
from scripts.interpret import run_local as local

TERMINAL_STEPS = 35


class Client:
    def __init__(self):
        self.payloads = []

    def request(self, payload):
        rpc = payload["eval_rpc"]
        if payload.get("reset"):
            return {"reset_finished": True, "eval_rpc": rpc}
        if payload.get("add_buffer"):
            self.payloads.append((payload["images"].copy(), payload["state"].copy(), dict(rpc)))
            return {"add_buffer_finished": True, "eval_rpc": rpc}
        actions = np.repeat((100 * rpc["replan_seq"] + np.arange(20))[:, None], 8, axis=1).astype(np.float32)
        return {"actions": actions, "eval_rpc": rpc}


class Recorder:
    def __init__(self, *args, fps=30):
        self.fps = fps

    def record(self, **kwargs):
        pass


def actor(*xyz):
    return SimpleNamespace(pose=SimpleNamespace(p=np.asarray([xyz], dtype=np.float32)))


def make_unwrapped():
    bins = [actor(0.1, 0.0, 0.05), actor(0.2, 0.0, 0.05), actor(0.3, 0.0, 0.05)]
    # 第一目标不是 spawned_bins 的第 0 个，才能测出下标确实按对象身份取。
    return SimpleNamespace(elapsed_steps=None, agent=SimpleNamespace(tcp=actor(0.0, 0.0, 0.3)), spawned_bins=bins,
                           selected_bins=[bins[2], bins[0], bins[1]], swap_times=2, pick_times=1,
                           swap_schedule=[(bins[0], bins[1], 64, 114), (bins[1], bins[2], 114, 164)],
                           color_names=["red", "green", "blue"])


def install_sim(monkeypatch):
    def initial(self):
        self.env.actions = []
        self.env.unwrapped.elapsed_steps = 7
        state = np.arange(8, dtype=np.float32)
        images = [np.full((256, 256, 3), index, dtype=np.uint8) for index in range(4)]
        return {"images": images, "wrist_images": images, "states": [state.copy() for _ in images],
                "task_goal": "pick up the container that hides the red cube"}

    def step(self, action):
        u = self.env.unwrapped
        self.env.actions.append(np.asarray(action).copy())
        u.elapsed_steps += 1
        u.spawned_bins[2].pose.p = u.spawned_bins[2].pose.p + np.asarray([[0, 0, 0.01]], dtype=np.float32)
        u.agent.tcp.pose.p = u.agent.tcp.pose.p - np.asarray([[0, 0, 0.001]], dtype=np.float32)
        u.timestep = 2 if u.elapsed_steps >= 20 else 1
        image = np.full((256, 256, 3), u.elapsed_steps, dtype=np.uint8)
        state = np.asarray(action, dtype=np.float32).copy()
        return (image, image, state), len(self.env.actions) == TERMINAL_STEPS, "fail"

    monkeypatch.setattr(runner.EnvRunner, "get_init_obs", initial)
    monkeypatch.setattr(runner.EnvRunner, "step", step)
    monkeypatch.setattr(runner.official, "RolloutRecorder", Recorder)


def make_runner(task="ButtonUnmaskSwap"):
    result = runner.TraceRunner.__new__(runner.TraceRunner)
    result.env_id, result.episode_id = task, 8
    result.env_seed, result.difficulty = 105, "easy"
    result.env = SimpleNamespace(unwrapped=make_unwrapped())
    return result


def run_episode(tmp_path, monkeypatch, task="ButtonUnmaskSwap", nonce="nonce"):
    install_sim(monkeypatch)
    env = make_runner(task)
    evaluator = runner.TraceEvaluator(runner.official.Args(), tmp_path)
    client = Client()
    outcome = evaluator.eval_each_episode(env, tmp_path / "videos", client, nonce)
    directory = tmp_path / "traces" / nonce
    rows = [json.loads(line) for line in (directory / "step_trace.jsonl").read_text().splitlines()]
    receipt = json.loads((directory / "meta" / "hold_receipt.json").read_text())
    return env, evaluator, client, outcome, rows, receipt


@pytest.mark.parametrize("task", runner.TASKS)
def test_trace_rows_and_receipt(tmp_path, monkeypatch, task):
    _, evaluator, _, outcome, rows, receipt = run_episode(tmp_path, monkeypatch, task)
    assert outcome == "fail" and evaluator.actions_executed == len(rows) == TERMINAL_STEPS
    required = {"event", "episode_key", "infer_seq", "row_in_chunk", "chunk_start_env_step", "state_before",
                "action_index", "exec_start_idx", "command_action", "executed_action", "state_after",
                "env_step_before", "env_step_after", "tcp_before", "tcp_after", "bins_before", "bins_after",
                "subgoal_index"}
    assert all(set(row) == required for row in rows)
    assert all(row["event"] == "step" and row["episode_key"] == f"{task}:8" for row in rows)
    assert [row["action_index"] for row in rows] == list(range(TERMINAL_STEPS))
    assert [(row["infer_seq"], row["row_in_chunk"]) for row in rows[15:17]] == [(0, 15), (1, 0)]
    assert rows[16]["chunk_start_env_step"] == 7 + 16 == rows[16]["env_step_before"]
    assert rows[16]["command_action"] == [100.0] * 8 == rows[16]["executed_action"]
    assert rows[0]["env_step_before"] == 7 and rows[0]["env_step_after"] == 8
    # 状态接续：步前状态是上一行步后状态，首行来自 reset 的最后一帧。
    assert rows[0]["state_before"] == list(range(8))
    assert all(a["state_after"] == b["state_before"] for a, b in zip(rows, rows[1:]))
    # bin 按 spawned_bins 顺序，只有第 3 个在抬高；步前后都读真实位姿。
    assert [len(row["bins_before"]) for row in rows[:2]] == [3, 3]
    assert rows[0]["bins_before"][2][2] == pytest.approx(0.05)
    assert rows[0]["bins_after"][2][2] == pytest.approx(0.06)
    assert rows[0]["bins_after"][0] == rows[0]["bins_before"][0]
    assert rows[0]["tcp_after"][2] == pytest.approx(0.299)
    # 子目标序号记步后值。
    assert rows[11]["env_step_after"] == 19 and rows[11]["subgoal_index"] == 1
    assert rows[12]["env_step_after"] == 20 and rows[12]["subgoal_index"] == 2
    assert receipt["difficulty"] == "easy" and receipt["swap_times"] == 2 and receipt["pick_times"] == 1
    assert receipt["swap_end"] == 164 and receipt["D"] == 3 == receipt["exec_start_idx"]
    assert receipt["color_names"] == ["red", "green", "blue"]
    assert receipt["prompt"] == "pick up the container that hides the red cube"
    assert receipt["first_target_bin_index"] == 2
    assert receipt["task_name"] == task and receipt["env_seed"] == 105 and receipt["episode_id"] == 8
    assert receipt["outcome"] == "fail" and receipt["task_success"] == 0 and receipt["actions_executed"] == TERMINAL_STEPS
    assert receipt["initial_env_step"] == 7 and receipt["video_exec0_frame_index"] == 3
    assert receipt["video_path"].endswith(f"{task}_ep8_fail.mp4")


def test_actions_and_payloads_match_official_loop(tmp_path, monkeypatch):
    env, evaluator, client, outcome, _, _ = run_episode(tmp_path, monkeypatch)
    baseline = runner.EnvRunner.__new__(runner.EnvRunner)
    baseline.env_id, baseline.env = "ButtonUnmaskSwap", SimpleNamespace(unwrapped=make_unwrapped())
    baseline_client = Client()
    official = runner.official.EpisodeEvaluator(runner.official.Args(), tmp_path)
    expected = official.eval_each_episode(baseline, tmp_path / "videos", baseline_client, "nonce")
    assert outcome == expected and evaluator.actions_executed == official.actions_executed
    assert np.array_equal(env.env.actions, baseline.env.actions)
    assert all(np.asarray(a).dtype == np.float32 for a in env.env.actions)
    for actual, reference in zip(client.payloads, baseline_client.payloads, strict=True):
        assert np.array_equal(actual[0], reference[0]) and np.array_equal(actual[1], reference[1])
        assert actual[2] == reference[2]


def test_trace_is_consumed_by_load_trace(tmp_path, monkeypatch):
    client_root = tmp_path / "attempts" / "000" / "client"
    client_root.mkdir(parents=True)
    run_episode(client_root, monkeypatch)
    rows = rollout_offline.load_trace(tmp_path, "nonce")
    assert sorted(rows) == [0, 1, 2]
    assert [rows[seq]["chunk_start_env_step"] for seq in rows] == [7, 23, 39]
    assert all(np.asarray(row["state_before"], dtype=np.float32).shape == (8,) for row in rows.values())
    assert rows[1]["episode_key"] == "ButtonUnmaskSwap:8"


def test_snapshot_does_not_require_subgoal_before_first_evaluate():
    env = SimpleNamespace(unwrapped=make_unwrapped())
    env.unwrapped.elapsed_steps = 0
    assert runner.snapshot(env)["subgoal_index"] is None


def test_receipt_written_when_episode_raises(tmp_path, monkeypatch):
    install_sim(monkeypatch)

    def broken(self, action):
        raise RuntimeError("仿真异常")

    monkeypatch.setattr(runner.EnvRunner, "step", broken)
    env = make_runner()
    with pytest.raises(RuntimeError, match="仿真异常"):
        runner.TraceEvaluator(runner.official.Args(), tmp_path).eval_each_episode(env, tmp_path / "videos", Client(), "nonce")
    receipt = json.loads((tmp_path / "traces/nonce/meta/hold_receipt.json").read_text())
    assert receipt["outcome"] is None and receipt["video_path"] is None and env.trace_stream.closed


def test_first_target_missing_is_rejected():
    u = make_unwrapped()
    u.selected_bins = [actor(0, 0, 0)]
    with pytest.raises(runner.official.ContractError, match="selected_bins"):
        runner.first_target_bin_index(u)


def test_runner_rejects_other_task(tmp_path):
    env = make_runner("StopCube")
    with pytest.raises(runner.official.ContractError, match="轨迹入口"):
        env.begin_trace("nonce", tmp_path)
    assert not (tmp_path / "traces").exists()


def grant(tmp_path, task):
    key = f"{task}:0"
    path = tmp_path / "grant.json"
    path.write_text(json.dumps({"chunk_id": "c", "run_identity_sha256": "a" * 64, "episodes": [
        {"task_name": task, "episode_id": 0, "episode_key": key, "env_seed": 1, "difficulty": "easy"}]}))
    return runner.official.Args(episode_manifest=str(path), chunk="c", run_identity_sha256="a" * 64)


def test_evaluate_rejects_other_task_before_official(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.official, "evaluate", lambda *a, **k: pytest.fail("不应进入官方入口"))
    with pytest.raises(runner.official.ContractError, match="轨迹入口"):
        runner.evaluate(grant(tmp_path, "StopCube"))


def test_evaluate_installs_and_restores_evaluator(tmp_path, monkeypatch):
    seen = {}

    def official(args, runner_factory):
        seen.update(evaluator=runner.official.EpisodeEvaluator, factory=runner_factory)
        return 0

    monkeypatch.setattr(runner.official, "evaluate", official)
    previous = runner.official.EpisodeEvaluator
    assert runner.evaluate(grant(tmp_path, "VideoUnmaskSwap")) == 0
    assert seen == {"evaluator": runner.TraceEvaluator, "factory": runner.TraceRunner}
    assert runner.official.EpisodeEvaluator is previous


def base_spec(tmp_path, tasks=runner.TASKS, **changes):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": [{"task_name": task} for task in tasks]}))
    values = dict(repo="/repo", policy_python="/policy/python", sim_python="/sim/python", stage2_python="/stage2/python",
                  checkpoint="/ckpt", policy_config="cfg", model_kind="stage2_pool4_concat", cuda="0",
                  attn_out="/out/attn", run_root="/out/run", manifest=str(manifest), xla_mem_fraction=0.5,
                  model_seed=42, run_identity_sha256="a" * 64)
    return values | changes


def load(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return local.load_spec(path)


def test_task_lists_stay_equal():
    assert local.BIN_TRACE_TASKS == runner.TASKS


def test_spec_accepts_bin_trace_with_rollout_record(tmp_path):
    spec = load(tmp_path, base_spec(tmp_path, bin_trace=True, rollout_record=True))
    assert spec["bin_trace"] is True and spec["rollout_record"] is True


def test_spec_rollout_record_still_needs_a_trace_client(tmp_path):
    with pytest.raises(SystemExit, match="rollout_record"):
        load(tmp_path, base_spec(tmp_path, rollout_record=True))
    with pytest.raises(SystemExit, match="rollout_record"):
        load(tmp_path, base_spec(tmp_path, rollout_record=True, bin_trace=False))


@pytest.mark.parametrize("value", ["true", 1, None])
def test_spec_bin_trace_must_be_bool(tmp_path, value):
    with pytest.raises(SystemExit, match="bin_trace 必须为 bool"):
        load(tmp_path, base_spec(tmp_path, bin_trace=value))


def test_spec_bin_trace_excludes_stopcube_mode(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema_version": 1, "episodes": {}}))
    with pytest.raises(SystemExit, match="互斥"):
        load(tmp_path, base_spec(tmp_path, bin_trace=True, rollout_record=True, stopcube_mode="normal",
                                 stopcube_config=str(config)))


def test_spec_bin_trace_rejects_other_tasks(tmp_path):
    spec = base_spec(tmp_path, tasks=("ButtonUnmaskSwap", "StopCube"), bin_trace=True, rollout_record=True)
    with pytest.raises(SystemExit, match="StopCube"):
        load(tmp_path, spec)
    assert load(tmp_path, base_spec(tmp_path, tasks=("ButtonUnmaskSwap", "StopCube"), bin_trace=False))["bin_trace"] is False


def client_arguments(tmp_path):
    grant_path = tmp_path / "grant.json"
    grant_path.write_text(json.dumps({"chunk_id": "chunk"}))
    return grant_path, tmp_path / "out", "nonce", 9000, tmp_path / "drain"


def test_client_command_replaces_only_entry(tmp_path):
    spec, arguments = base_spec(tmp_path), client_arguments(tmp_path)
    default = local.RW.client_command(spec, *arguments)
    assert local.build_client_command(spec, *arguments) == default
    assert local.build_client_command(dict(spec, bin_trace=False), *arguments) == default
    command = local.build_client_command(dict(spec, bin_trace=True), *arguments)
    assert command[1] == "/repo/scripts/interpret/bin_trace_runner.py"
    assert command[:1] + command[2:] == default[:1] + default[2:]


def test_server_command_gets_rollout_flag_only(tmp_path):
    spec = base_spec(tmp_path)
    off = local.build_server_command(spec, "nonce", 9000)
    assert local.build_server_command(dict(spec, bin_trace=True), "nonce", 9000) == off
    on = local.build_server_command(dict(spec, bin_trace=True, rollout_record=True), "nonce", 9000)
    index = on.index("--rollout-record")
    assert index < on.index("policy:checkpoint") and on[:index] + on[index + 1:] == off


def test_identity_unchanged_without_key(tmp_path):
    spec, manifest = base_spec(tmp_path), {"sha256": "b" * 64}
    identity = local.run_identity(spec, manifest)
    assert identity == M.digest({"manifest_seal": "b" * 64, "checkpoint": "/ckpt", "policy_config": "cfg",
                                 "model_kind": "stage2_pool4_concat", "model_seed": 42})
    assert local.run_identity(dict(spec, bin_trace=False), manifest) == identity
    assert local.run_identity(dict(spec, bin_trace=True), manifest) != identity


def test_collect_without_stopcube_keeps_rollout_count(tmp_path):
    attn, output = tmp_path / "attn", tmp_path / "client"
    (attn / "n").mkdir(parents=True)
    (attn / "n" / "rollout_0000.npz").write_bytes(b"")
    (attn / "n" / "episode.json").write_text("{}")
    terminal = dict(episode_key="ButtonUnmaskSwap:8", episode_nonce="n", task_name="ButtonUnmaskSwap", episode_id=8,
                    env_seed=1, difficulty="easy", chunk_id="c", run_identity_sha256="r", outcome="fail", rpc_seq=3)
    output.mkdir()
    (output / "x.terminal.json").write_text(json.dumps(terminal))
    spec = base_spec(tmp_path, attn_out=str(attn), run_root=str(tmp_path / "run"), bin_trace=True, rollout_record=True)
    row = local.collect(spec, tmp_path, output, [{"episode_key": "ButtonUnmaskSwap:8", "task_name": "ButtonUnmaskSwap",
                                                  "episode_id": 8}])[0]
    assert row["npz"] == 1 == row["infers"] and row["terminal"]
    assert list((tmp_path / "run" / "accepted").glob("*.terminal.json"))
