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
SWAP_TASKS = ("ButtonUnmaskSwap", "VideoUnmaskSwap")
TASK_NAMES = {
    "VideoUnmaskSwap": ["static", "pick up the container that hides the red cube", "put down the container",
                        "pick up the container that hides the green cube"],
    "ButtonUnmaskSwap": ["press the first button", "press the second button",
                         "pick up the container that hides the red cube", "put down the container"],
    # swap_times=2：示范的拿起、放下、static ×3、NO RECORD 之后才是第一次正式拿起，下标 6。
    "VideoRepick": ["pick up the cube", "drop the cube on the table", "static", "static", "static", "NO RECORD",
                    "pick up the correct cube for the first time", "put it down", "press the button to finish"],
}


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


def make_unwrapped(task="ButtonUnmaskSwap"):
    if task == "VideoRepick":
        return make_cube_unwrapped()
    bins = [actor(0.1, 0.0, 0.05), actor(0.2, 0.0, 0.05), actor(0.3, 0.0, 0.05)]
    # 第一目标不是 spawned_bins 的第 0 个，才能测出下标确实按对象身份取。
    u = SimpleNamespace(elapsed_steps=None, spawned_bins=bins,
                        selected_bins=[bins[2], bins[0], bins[1]], swap_times=2, pick_times=1,
                        swap_schedule=[(bins[0], bins[1], 64, 114), (bins[1], bins[2], 114, 164)],
                        color_names=["red", "green", "blue"], task_list=[{"name": n} for n in TASK_NAMES.get(task, [])],
                        successflag=np.asarray([False]), failureflag=np.asarray([False]), grasped=[])
    u.agent = SimpleNamespace(tcp=actor(0.0, 0.0, 0.3),
                              is_grasping=lambda obj: np.asarray([any(obj is g for g in u.grasped)]))
    return u


def make_cube_unwrapped():
    """VideoRepick 假环境：没有 spawned_bins／selected_bins／pick_times／color_names；夹住与否看 grasped。"""
    cubes = [actor(0.1, 0.1, 0.02), actor(0.2, 0.1, 0.02), actor(0.3, 0.1, 0.02)]
    u = SimpleNamespace(elapsed_steps=None, spawned_cubes=cubes, target_cube_1=cubes[1], swap_times=2, num_repeats=2,
                        swap_schedule=[(cubes[1], cubes[0], 400, 450), (cubes[2], None, 450, 500)],
                        task_list=[{"name": n} for n in TASK_NAMES["VideoRepick"]],
                        successflag=np.asarray([False]), failureflag=np.asarray([False]), grasped=[])
    u.agent = SimpleNamespace(tcp=actor(0.0, 0.0, 0.3),
                              is_grasping=lambda obj: np.asarray([any(obj is g for g in u.grasped)]))
    return u


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
    result.env = SimpleNamespace(unwrapped=make_unwrapped(task))
    return result


def run_episode(tmp_path, monkeypatch, task="ButtonUnmaskSwap", nonce="nonce", args=None, sim=None):
    (sim or install_sim)(monkeypatch)
    env = make_runner(task)
    evaluator = runner.TraceEvaluator(args or runner.official.Args(), tmp_path)
    client = Client()
    outcome = evaluator.eval_each_episode(env, tmp_path / "videos", client, nonce)
    directory = tmp_path / "traces" / nonce
    rows = [json.loads(line) for line in (directory / "step_trace.jsonl").read_text().splitlines()]
    receipt = json.loads((directory / "meta" / "hold_receipt.json").read_text())
    return env, evaluator, client, outcome, rows, receipt


@pytest.mark.parametrize("task", SWAP_TASKS)
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
    # 开关关闭时 Swap 任务不写任何扩展字段（行字段集合已在上面钉死）。
    assert not {"target_index", "swap_schedule", "first_pick", "truncated_first_pick", "stop_after_first_pick",
                "first_pick_subgoal_index", "num_repeats"} & set(receipt)


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


# ---------------- 三任务化与拿起即停 ----------------

def install_plan_sim(plan, terminal=TERMINAL_STEPS):
    """plan(u, n) 在第 n 步（从 1 起）物理推进后改写物体位姿、子目标与判定标志。"""
    def install(monkeypatch):
        def initial(self):
            self.env.actions = []
            self.env.unwrapped.elapsed_steps = 7
            self.env.unwrapped.timestep = 0
            state = np.arange(8, dtype=np.float32)
            images = [np.full((256, 256, 3), index, dtype=np.uint8) for index in range(4)]
            return {"images": images, "wrist_images": images, "states": [state.copy() for _ in images],
                    "task_goal": "goal"}

        def step(self, action):
            u = self.env.unwrapped
            self.env.actions.append(np.asarray(action).copy())
            u.elapsed_steps += 1
            plan(u, len(self.env.actions))
            image = np.full((256, 256, 3), u.elapsed_steps % 256, dtype=np.uint8)
            state = np.asarray(action, dtype=np.float32).copy()
            return (image, image, state), len(self.env.actions) == terminal, "fail"

        monkeypatch.setattr(runner.EnvRunner, "get_init_obs", initial)
        monkeypatch.setattr(runner.EnvRunner, "step", step)
        monkeypatch.setattr(runner.official, "RolloutRecorder", Recorder)
    return install


def lift(obj, z):
    obj.pose.p = np.asarray([[obj.pose.p[0, 0], obj.pose.p[0, 1], z]], dtype=np.float32)


STOP = runner.Args(stop_after_first_pick=True)


@pytest.mark.parametrize("task, objects, target, subgoal", [("ButtonUnmaskSwap", "spawned_bins", 2, 2),
                                                             ("VideoUnmaskSwap", "spawned_bins", 2, 1),
                                                             ("VideoRepick", "spawned_cubes", 1, 6)])
def test_objects_and_target_by_task(task, objects, target, subgoal):
    u = make_unwrapped(task)
    assert runner.objects_of(u, task) == getattr(u, objects)
    assert runner.first_target_bin_index(u, task) == target
    expected = [np.asarray(a.pose.p).reshape(-1).tolist() for a in getattr(u, objects)]
    assert runner.snapshot(SimpleNamespace(unwrapped=u), task)["bins"] == expected
    assert runner.first_pick_subgoal_index(u, task) == subgoal


def test_cube_target_missing_is_rejected():
    u = make_cube_unwrapped()
    u.target_cube_1 = actor(0, 0, 0)
    with pytest.raises(runner.official.ContractError, match="target_cube_1"):
        runner.first_target_bin_index(u, "VideoRepick")


def test_first_pick_subgoal_index_checked_for_swap_tasks():
    u = make_unwrapped("ButtonUnmaskSwap")
    u.task_list = [{"name": "press the first button"}, {"name": "pick up the container that hides x"}]
    with pytest.raises(runner.official.ContractError, match="第一次拿起"):
        runner.first_pick_subgoal_index(u, "ButtonUnmaskSwap")


def test_swap_schedule_maps_actors_to_indices():
    assert runner.swap_schedule_of(make_cube_unwrapped(), "VideoRepick") == [[1, 0, 400, 450], [2, None, 450, 500]]
    bins = make_unwrapped()
    assert runner.swap_schedule_of(bins, "ButtonUnmaskSwap") == [[0, 1, 64, 114], [1, 2, 114, 164]]
    bins.swap_schedule = None
    assert runner.swap_schedule_of(bins, "ButtonUnmaskSwap") == []


def test_videorepick_without_switch_runs_to_end_with_objects(tmp_path, monkeypatch):
    _, _, _, outcome, rows, receipt = run_episode(tmp_path, monkeypatch, "VideoRepick",
                                                  sim=install_plan_sim(lambda u, n: None))
    assert outcome == "fail" and len(rows) == TERMINAL_STEPS
    assert all(row["objects"] == row["bins_after"] and len(row["objects"]) == 3 for row in rows)
    assert rows[0]["objects"][1] == pytest.approx([0.2, 0.1, 0.02])
    assert "first_pick" not in receipt and receipt["stop_after_first_pick"] is False
    assert receipt["target_index"] == 1 == receipt["first_target_bin_index"]
    assert receipt["swap_schedule"] == [[1, 0, 400, 450], [2, None, 450, 500]]
    assert receipt["pick_times"] is None and receipt["color_names"] is None and receipt["num_repeats"] == 2
    assert receipt["swap_end"] == 500 and receipt["first_pick_subgoal_index"] == 6


def test_bin_stop_uses_container_threshold(tmp_path, monkeypatch):
    def plan(u, n):
        u.timestep = 2 if n >= 3 else 0
        lift(u.spawned_bins[2], {5: 0.15, 6: 0.151}.get(n, 0.05))  # 恰好 0.15 不算，超过才算
        u.grasped = [u.spawned_bins[2]] if n >= 5 else []
        if n >= 6:
            u.timestep = 3  # 拿对时同一步 evaluate 里子目标已前进
    _, evaluator, _, outcome, rows, receipt = run_episode(tmp_path, monkeypatch, "ButtonUnmaskSwap", args=STOP,
                                                          sim=install_plan_sim(plan))
    assert outcome == "success" and len(rows) == 6 == evaluator.actions_executed
    pick = receipt["first_pick"]
    assert pick == {"index": 2, "env_step": 13, "action_index": 5, "is_target": True, "subgoal_index_at_pick": 2,
                    "subgoal_index_after_pick": 3, "successflag": False, "failureflag": False, "outcome": "success"}
    assert rows[-1]["first_pick"] == pick and all("first_pick" not in row for row in rows[:-1])
    assert all("objects" in row for row in rows)
    assert receipt["truncated_first_pick"] is True and receipt["outcome"] == "success" and receipt["task_success"] == 1
    assert receipt["first_pick_subgoal_index"] == 2 and receipt["target_index"] == 2
    assert receipt["stop_after_first_pick"] is True


def test_bin_stop_wrong_container_is_fail(tmp_path, monkeypatch):
    def plan(u, n):
        u.timestep = 1
        if n == 4:
            lift(u.spawned_bins[0], 0.2)
            u.grasped = [u.spawned_bins[0]]
            u.failureflag = np.asarray([True])
    _, _, _, outcome, rows, receipt = run_episode(tmp_path, monkeypatch, "VideoUnmaskSwap", args=STOP,
                                                  sim=install_plan_sim(plan))
    assert outcome == "fail" and len(rows) == 4
    assert receipt["first_pick"]["index"] == 0 and receipt["first_pick"]["is_target"] is False
    assert receipt["first_pick"]["failureflag"] is True and receipt["truncated_first_pick"] is True


def test_button_pick_before_buttons_is_fail(tmp_path, monkeypatch):
    def plan(u, n):
        u.timestep = 1  # 第二个按钮还没按
        if n == 2:
            lift(u.spawned_bins[2], 0.3)
            u.grasped = [u.spawned_bins[2]]
    _, _, _, outcome, _, receipt = run_episode(tmp_path, monkeypatch, "ButtonUnmaskSwap", args=STOP,
                                               sim=install_plan_sim(plan))
    assert outcome == "fail" and receipt["first_pick"]["is_target"] is True
    assert receipt["first_pick"]["subgoal_index_at_pick"] == 1


def test_cube_stop_needs_grasp(tmp_path, monkeypatch):
    def plan(u, n):
        u.timestep = 6
        if n >= 3:
            lift(u.spawned_cubes[0], 0.08)  # 抬高但没夹住：不算拿起
        if n >= 5:
            lift(u.spawned_cubes[1], 0.06)
            u.grasped = [u.spawned_cubes[1]]
            u.timestep = 7
    _, _, _, outcome, rows, receipt = run_episode(tmp_path, monkeypatch, "VideoRepick", args=STOP,
                                                  sim=install_plan_sim(plan))
    assert outcome == "success" and len(rows) == 5
    assert receipt["first_pick"]["index"] == 1 and receipt["first_pick"]["subgoal_index_at_pick"] == 6
    assert receipt["first_pick"]["subgoal_index_after_pick"] == 7


def test_cube_grasped_but_low_is_not_pick():
    u = make_cube_unwrapped()
    u.grasped = [u.spawned_cubes[0]]
    lift(u.spawned_cubes[0], 0.05)
    assert runner.picked_index(u, "VideoRepick") is None
    lift(u.spawned_cubes[0], 0.0501)
    assert runner.picked_index(u, "VideoRepick") == 0


def test_bin_lifted_by_env_without_grasp_is_not_pick(tmp_path, monkeypatch):
    """ButtonUnmaskSwap 开局环境抬起全部容器（未抓取）：不算拿起，开关打开也跑到底。"""
    def plan(u, n):
        u.timestep = 0
        for b in u.spawned_bins:
            lift(b, 0.3 if n <= 4 else 0.05)
    _, _, _, outcome, rows, receipt = run_episode(tmp_path, monkeypatch, "ButtonUnmaskSwap", args=STOP,
                                                  sim=install_plan_sim(plan))
    assert outcome == "fail" and len(rows) == TERMINAL_STEPS
    assert receipt["first_pick"] is None and receipt["truncated_first_pick"] is False


def test_bin_pick_needs_grasp_and_height():
    u = make_unwrapped("ButtonUnmaskSwap")
    lift(u.spawned_bins[1], 0.3)
    assert runner.picked_index(u, "ButtonUnmaskSwap") is None  # 抬高未抓取
    u.grasped = [u.spawned_bins[1]]
    assert runner.picked_index(u, "ButtonUnmaskSwap") == 1
    lift(u.spawned_bins[1], 0.15)
    assert runner.picked_index(u, "ButtonUnmaskSwap") is None  # 抓取但恰好 0.15 不算


def test_stop_switch_without_pick_runs_to_end(tmp_path, monkeypatch):
    _, _, _, outcome, rows, receipt = run_episode(tmp_path, monkeypatch, "VideoUnmaskSwap", args=STOP,
                                                  sim=install_plan_sim(lambda u, n: None))
    assert outcome == "fail" and len(rows) == TERMINAL_STEPS
    assert receipt["first_pick"] is None and receipt["truncated_first_pick"] is False


def test_button_swap_schedule_reread_at_finish(tmp_path, monkeypatch):
    def plan(u, n):
        u.timestep = 2
        if n == 3:
            # 执行中交换窗开始，第二个物体才确定，元组被重建。
            u.swap_schedule = [(u.spawned_bins[0], u.spawned_bins[2], 64, 114)]
        if n == 5:
            lift(u.spawned_bins[1], 0.2)
    install_plan_sim(plan)(monkeypatch)
    env = make_runner("ButtonUnmaskSwap")
    env.env.unwrapped.swap_schedule = [(env.env.unwrapped.spawned_bins[0], None, 64, 114)]
    evaluator = runner.TraceEvaluator(STOP, tmp_path)
    assert evaluator.eval_each_episode(env, tmp_path / "videos", Client(), "n2") == "fail"
    receipt = json.loads((tmp_path / "traces/n2/meta/hold_receipt.json").read_text())
    assert receipt["swap_schedule"] == [[0, 2, 64, 114]] and receipt["swap_end"] == 114


def test_truncated_terminal_record_is_marked(tmp_path):
    seen = []
    wrapped = runner.write_record_marking_truncation(lambda path, record: seen.append(record))
    runner.TRUNCATED_NONCES.add("cut")
    wrapped(tmp_path / "a", {"record_type": "terminal", "episode_nonce": "cut", "outcome": "fail"})
    wrapped(tmp_path / "b", {"record_type": "terminal", "episode_nonce": "full", "outcome": "fail"})
    wrapped(tmp_path / "c", {"record_type": "started", "episode_nonce": "cut"})
    assert seen == [{"record_type": "terminal", "episode_nonce": "cut", "outcome": "fail", "truncated_first_pick": True},
                    {"record_type": "terminal", "episode_nonce": "full", "outcome": "fail"},
                    {"record_type": "started", "episode_nonce": "cut"}]


def test_truncation_reaches_terminal_record(tmp_path, monkeypatch):
    """截断局经官方 write_record 写 terminal 时带上标记；未截断局不带。"""
    def plan(u, n):
        u.timestep = 1
        if n == 2:
            lift(u.spawned_bins[2], 0.3)
            u.grasped = [u.spawned_bins[2]]
    _, _, _, outcome, _, _ = run_episode(tmp_path, monkeypatch, "VideoUnmaskSwap", nonce="cutnonce", args=STOP,
                                         sim=install_plan_sim(plan))
    assert outcome == "success"
    write = runner.write_record_marking_truncation(runner.official.write_record)
    write(tmp_path / "x.terminal.json", {"record_type": "terminal", "episode_nonce": "cutnonce", "outcome": outcome})
    assert json.loads((tmp_path / "x.terminal.json").read_text())["truncated_first_pick"] is True


@pytest.mark.parametrize("stop", [False, True])
def test_evaluate_installs_and_restores_write_record(tmp_path, monkeypatch, stop):
    seen = {}

    def official(args, runner_factory):
        seen["write"] = runner.official.write_record
        return 0

    monkeypatch.setattr(runner.official, "evaluate", official)
    previous = runner.official.write_record
    base = grant(tmp_path, "VideoRepick")
    args = runner.Args(episode_manifest=base.episode_manifest, chunk=base.chunk,
                       run_identity_sha256=base.run_identity_sha256, stop_after_first_pick=stop)
    assert runner.evaluate(args) == 0
    assert (seen["write"] is previous) is (not stop)
    assert runner.official.write_record is previous


def test_cli_flag_parses():
    tyro = pytest.importorskip("tyro")
    assert tyro.cli(runner.Args, args=["--stop-after-first-pick"]).stop_after_first_pick is True
    assert tyro.cli(runner.Args, args=[]).stop_after_first_pick is False


def test_spec_stop_after_first_pick(tmp_path):
    spec = load(tmp_path, base_spec(tmp_path, bin_trace=True, stop_after_first_pick=True))
    assert spec["stop_after_first_pick"] is True
    with pytest.raises(SystemExit, match="stop_after_first_pick 只许"):
        load(tmp_path, base_spec(tmp_path, stop_after_first_pick=True))
    with pytest.raises(SystemExit, match="stop_after_first_pick 只许"):
        load(tmp_path, base_spec(tmp_path, bin_trace=False, stop_after_first_pick=True))
    with pytest.raises(SystemExit, match="stop_after_first_pick 必须为 bool"):
        load(tmp_path, base_spec(tmp_path, bin_trace=True, stop_after_first_pick="true"))
    assert load(tmp_path, base_spec(tmp_path, stop_after_first_pick=False))["stop_after_first_pick"] is False


def test_spec_bin_trace_accepts_videorepick(tmp_path):
    assert load(tmp_path, base_spec(tmp_path, tasks=("VideoRepick",), bin_trace=True))["bin_trace"] is True


def test_client_command_and_identity_with_stop(tmp_path):
    spec, arguments = base_spec(tmp_path, bin_trace=True), client_arguments(tmp_path)
    plain = local.build_client_command(spec, *arguments)
    assert local.build_client_command(dict(spec, stop_after_first_pick=False), *arguments) == plain
    with_stop = local.build_client_command(dict(spec, stop_after_first_pick=True), *arguments)
    assert with_stop == plain + ["--stop-after-first-pick"]
    manifest = {"sha256": "b" * 64}
    identity = local.run_identity(spec, manifest)
    assert local.run_identity(dict(spec, stop_after_first_pick=False), manifest) == identity
    assert local.run_identity(dict(spec, stop_after_first_pick=True), manifest) != identity
