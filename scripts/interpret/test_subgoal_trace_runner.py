"""PatternLock、RouteStick 轨迹客户端与 run_local 开关的 CPU 测试：假环境、假客户端，不开仿真。"""
from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import manifest as M
from scripts.interpret import rollout_offline
from scripts.interpret import run_local as local
from scripts.interpret import subgoal_trace_runner as runner

TERMINAL_STEPS = 35
DEMO_INDICES = [1, 1, 2, 3, 3, 5]   # m=2：序号 1..m，末帧 m+3
MODULE = "robomme.env_record_wrapper.DemonstrationWrapper"


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


class FakeDemonstrationWrapper:
    """只保留被用到的形状：类方法 _augment_obs_and_info、reset 里逐帧经 self 调用它、结果存 demonstration_data。"""

    def __init__(self, unwrapped, indices=DEMO_INDICES, extra_frames=0, reset_error=None):
        self.unwrapped, self.indices, self.extra_frames, self.reset_error = unwrapped, indices, extra_frames, reset_error
        self.demonstration_data = None

    def _augment_obs_and_info(self, obs, info, action):
        return obs, dict(info, frame=obs)

    def reset(self):
        columns = {}
        for frame, index in enumerate(self.indices):
            self.unwrapped.current_task_index = index
            if self.reset_error is not None and frame == 2:
                raise self.reset_error
            _, info = self._augment_obs_and_info(frame, {}, None)
            for key, value in info.items():
                columns.setdefault(key, []).append(value)
        self.demonstration_data = (None, None, None, None, columns)
        self.unwrapped.current_task_index = self.indices[-1]
        self.unwrapped.timestep = self.indices[-1]
        self.unwrapped.elapsed_steps = 7
        return len(self.indices) + self.extra_frames


class OuterWrapper:
    """外层 gym wrapper 的行为：转发公开属性，不转发私有属性。"""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)


@pytest.fixture(autouse=True)
def fake_robomme(monkeypatch):
    for name, attrs in (("robomme", {}), ("robomme.env_record_wrapper", {}),
                        (MODULE, {"DemonstrationWrapper": FakeDemonstrationWrapper})):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)


def make_unwrapped():
    return SimpleNamespace(elapsed_steps=None, agent=SimpleNamespace(tcp=actor(0.0, 0.0, 0.3)),
                           selected_buttons=[object(), object(), object()], current_task_index=-1, timestep=0)


def install_sim(monkeypatch, step=None):
    def initial(self):
        self.env.actions = []
        frames = self.env.reset()
        state = np.arange(8, dtype=np.float32)
        images = [np.full((256, 256, 3), index, dtype=np.uint8) for index in range(frames)]
        return {"images": images, "wrist_images": images, "states": [state.copy() for _ in images],
                "task_goal": "press the buttons in order"}

    def default_step(self, action):
        u = self.env.unwrapped
        self.env.actions.append(np.asarray(action).copy())
        u.elapsed_steps += 1
        u.agent.tcp.pose.p = u.agent.tcp.pose.p - np.asarray([[0, 0, 0.001]], dtype=np.float32)
        u.timestep = 2 if u.elapsed_steps >= 15 else 1
        u.current_task_index = 2 if u.elapsed_steps >= 20 else 1
        image = np.full((256, 256, 3), u.elapsed_steps, dtype=np.uint8)
        state = np.asarray(action, dtype=np.float32).copy()
        return (image, image, state), len(self.env.actions) == TERMINAL_STEPS, "fail"

    monkeypatch.setattr(runner.EnvRunner, "get_init_obs", initial)
    monkeypatch.setattr(runner.EnvRunner, "step", step or default_step)
    monkeypatch.setattr(runner.official, "RolloutRecorder", Recorder)


def make_runner(task="PatternLock", **wrapper_kwargs):
    result = runner.TraceRunner.__new__(runner.TraceRunner)
    result.env_id, result.episode_id = task, 8
    result.env_seed, result.difficulty = 105, "easy"
    demo = FakeDemonstrationWrapper(make_unwrapped(), **wrapper_kwargs)
    result.env = OuterWrapper(OuterWrapper(demo))
    result.demo = demo
    return result


def run_episode(tmp_path, monkeypatch, task="PatternLock", nonce="nonce"):
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
                "env_step_before", "env_step_after", "tcp_before", "tcp_after",
                "task_index_before", "task_index_after", "timestep_before", "timestep_after"}
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
    assert rows[0]["tcp_after"][2] == pytest.approx(0.299)
    # 首步步前是示范末帧的序号 m+3，步后是环境写入的执行段序号。
    assert rows[0]["task_index_before"] == 5 == rows[0]["timestep_before"]
    assert rows[0]["task_index_after"] == 1 == rows[0]["timestep_after"]
    # timestep 早于 current_task_index 变化，两者分别记录。
    assert rows[7]["env_step_after"] == 15 and (rows[7]["timestep_after"], rows[7]["task_index_after"]) == (2, 1)
    assert rows[12]["env_step_after"] == 20 and (rows[12]["timestep_after"], rows[12]["task_index_after"]) == (2, 2)
    assert rows[12]["task_index_before"] == 1 and rows[12]["timestep_before"] == 2
    assert receipt["m"] == 2 and receipt["demo_task_index"] == DEMO_INDICES
    assert receipt["D"] == len(DEMO_INDICES) - 1 == receipt["exec_start_idx"]
    assert receipt["prompt"] == "press the buttons in order"
    assert receipt["task_name"] == task and receipt["env_seed"] == 105 and receipt["episode_id"] == 8
    assert receipt["outcome"] == "fail" and receipt["task_success"] == 0 and receipt["actions_executed"] == TERMINAL_STEPS
    assert receipt["initial_env_step"] == 7 and receipt["video_exec0_frame_index"] == len(DEMO_INDICES) - 1
    assert receipt["video_path"].endswith(f"{task}_ep8_fail.mp4")
    assert not {"swap_times", "pick_times", "swap_end", "color_names", "first_target_bin_index"} & set(receipt)


def test_actions_and_payloads_match_official_loop(tmp_path, monkeypatch):
    env, evaluator, client, outcome, _, _ = run_episode(tmp_path, monkeypatch)
    baseline = runner.EnvRunner.__new__(runner.EnvRunner)
    baseline.env_id = "PatternLock"
    baseline.env = OuterWrapper(FakeDemonstrationWrapper(make_unwrapped()))
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
    assert rows[1]["episode_key"] == "PatternLock:8"


def test_hook_only_exists_during_reset(tmp_path, monkeypatch):
    install_sim(monkeypatch)
    env = make_runner()
    seen = {}
    reset = env.demo.reset

    def spying_reset():
        seen["during"] = "_augment_obs_and_info" in vars(env.demo)
        return reset()

    env.demo.reset = spying_reset
    env.begin_trace("nonce", tmp_path)
    env.get_init_obs()
    assert seen["during"] is True
    assert "_augment_obs_and_info" not in vars(env.demo)
    assert env.demo._augment_obs_and_info.__func__ is FakeDemonstrationWrapper._augment_obs_and_info
    # 原函数的输出照旧进入 info，只多一列序号。
    assert set(env.demo.demonstration_data[4]) == {"frame", runner.DEMO_KEY}
    assert env.demo.demonstration_data[4]["frame"] == list(range(len(DEMO_INDICES)))


def test_hook_removed_when_reset_raises(tmp_path, monkeypatch):
    install_sim(monkeypatch)
    env = make_runner(reset_error=RuntimeError("示范失败"))
    env.begin_trace("nonce", tmp_path)
    with pytest.raises(RuntimeError, match="示范失败"):
        env.get_init_obs()
    assert "_augment_obs_and_info" not in vars(env.demo)
    assert env.receipt["demo_task_index"] is None


def test_length_mismatch_is_rejected(tmp_path, monkeypatch):
    install_sim(monkeypatch)
    env = make_runner(extra_frames=1)
    env.begin_trace("nonce", tmp_path)
    with pytest.raises(runner.official.ContractError, match="demo_task_index"):
        env.get_init_obs()
    assert "_augment_obs_and_info" not in vars(env.demo)


def test_snapshot_only_reads():
    u = make_unwrapped()
    u.elapsed_steps, u.current_task_index, u.timestep = 3, 1, 2
    before = dict(vars(u))
    result = runner.snapshot(SimpleNamespace(unwrapped=u))
    assert result["env_step"] == 3 and result["task_index"] == 1 and result["timestep"] == 2
    assert vars(u) == before


def test_receipt_written_when_episode_raises(tmp_path, monkeypatch):
    def broken(self, action):
        raise RuntimeError("仿真异常")

    install_sim(monkeypatch, step=broken)
    env = make_runner()
    with pytest.raises(RuntimeError, match="仿真异常"):
        runner.TraceEvaluator(runner.official.Args(), tmp_path).eval_each_episode(env, tmp_path / "videos", Client(), "nonce")
    receipt = json.loads((tmp_path / "traces/nonce/meta/hold_receipt.json").read_text())
    assert receipt["outcome"] is None and receipt["video_path"] is None and env.trace_stream.closed


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
        runner.evaluate(grant(tmp_path, "ButtonUnmaskSwap"))


def test_evaluate_installs_and_restores_evaluator(tmp_path, monkeypatch):
    seen = {}

    def official(args, runner_factory):
        seen.update(evaluator=runner.official.EpisodeEvaluator, factory=runner_factory)
        return 0

    monkeypatch.setattr(runner.official, "evaluate", official)
    previous = runner.official.EpisodeEvaluator
    assert runner.evaluate(grant(tmp_path, "RouteStick")) == 0
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
    assert local.SUBGOAL_TRACE_TASKS == runner.TASKS


def test_spec_accepts_subgoal_trace_with_rollout_record(tmp_path):
    spec = load(tmp_path, base_spec(tmp_path, subgoal_trace=True, rollout_record=True))
    assert spec["subgoal_trace"] is True and spec["rollout_record"] is True


def test_spec_rollout_record_still_needs_a_trace_client(tmp_path):
    with pytest.raises(SystemExit, match="rollout_record"):
        load(tmp_path, base_spec(tmp_path, rollout_record=True))
    with pytest.raises(SystemExit, match="rollout_record"):
        load(tmp_path, base_spec(tmp_path, rollout_record=True, subgoal_trace=False))


@pytest.mark.parametrize("value", ["true", 1, None])
def test_spec_subgoal_trace_must_be_bool(tmp_path, value):
    with pytest.raises(SystemExit, match="subgoal_trace 必须为 bool"):
        load(tmp_path, base_spec(tmp_path, subgoal_trace=value))


def test_spec_subgoal_trace_excludes_stopcube_mode(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema_version": 1, "episodes": {}}))
    with pytest.raises(SystemExit, match="互斥"):
        load(tmp_path, base_spec(tmp_path, subgoal_trace=True, stopcube_mode="normal", stopcube_config=str(config)))


def test_spec_subgoal_trace_excludes_bin_trace(tmp_path):
    with pytest.raises(SystemExit, match="互斥"):
        load(tmp_path, base_spec(tmp_path, tasks=local.BIN_TRACE_TASKS, subgoal_trace=True, bin_trace=True))


def test_spec_subgoal_trace_rejects_other_tasks(tmp_path):
    spec = base_spec(tmp_path, tasks=("PatternLock", "StopCube"), subgoal_trace=True, rollout_record=True)
    with pytest.raises(SystemExit, match="StopCube"):
        load(tmp_path, spec)
    assert load(tmp_path, base_spec(tmp_path, tasks=("PatternLock", "StopCube"), subgoal_trace=False))["subgoal_trace"] is False


def client_arguments(tmp_path):
    grant_path = tmp_path / "grant.json"
    grant_path.write_text(json.dumps({"chunk_id": "chunk"}))
    return grant_path, tmp_path / "out", "nonce", 9000, tmp_path / "drain"


def test_client_command_replaces_only_entry(tmp_path):
    spec, arguments = base_spec(tmp_path), client_arguments(tmp_path)
    default = local.RW.client_command(spec, *arguments)
    assert local.build_client_command(spec, *arguments) == default
    assert local.build_client_command(dict(spec, subgoal_trace=False), *arguments) == default
    command = local.build_client_command(dict(spec, subgoal_trace=True), *arguments)
    assert command[1] == "/repo/scripts/interpret/subgoal_trace_runner.py"
    assert command[:1] + command[2:] == default[:1] + default[2:]


def test_server_command_gets_rollout_flag_only(tmp_path):
    spec = base_spec(tmp_path)
    off = local.build_server_command(spec, "nonce", 9000)
    assert local.build_server_command(dict(spec, subgoal_trace=True), "nonce", 9000) == off
    on = local.build_server_command(dict(spec, subgoal_trace=True, rollout_record=True), "nonce", 9000)
    index = on.index("--rollout-record")
    assert index < on.index("policy:checkpoint") and on[:index] + on[index + 1:] == off


def test_identity_unchanged_without_key(tmp_path):
    spec, manifest = base_spec(tmp_path), {"sha256": "b" * 64}
    identity = local.run_identity(spec, manifest)
    assert identity == M.digest({"manifest_seal": "b" * 64, "checkpoint": "/ckpt", "policy_config": "cfg",
                                 "model_kind": "stage2_pool4_concat", "model_seed": 42})
    assert local.run_identity(dict(spec, subgoal_trace=False), manifest) == identity
    assert local.run_identity(dict(spec, subgoal_trace=True), manifest) != identity
