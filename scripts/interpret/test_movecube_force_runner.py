"""MoveCube 强推动作表、客户端与 run_local 开关的 CPU 测试：替身环境与替身客户端，不依赖 GPU 与 sapien。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import manifest as M
from scripts.interpret import movecube_force_runner as runner
from scripts.interpret import movecube_force_table as force_table
from scripts.interpret import run_local as local

ORIGIN = 7
PEG, CUBE, HEAD, GOAL = [0.1, 0.2, 0.03], [0.3, -0.1, 0.02], [0.15, 0.2, 0.03], [0.4, 0.1, 0.0]


class Recorder:
    def __init__(self, *args, fps=30):
        self.fps = fps

    def record(self, **kwargs):
        pass


class Client:
    def __init__(self):
        self.frames = []

    def request(self, payload):
        rpc = payload["eval_rpc"]
        if payload.get("reset"):
            return {"reset_finished": True, "eval_rpc": rpc}
        if payload.get("add_buffer"):
            self.frames.append(rpc["frame_count"])
            return {"add_buffer_finished": True, "eval_rpc": rpc}
        # 每个边界给可识别的新动作，放手步能看出执行的是最新推理。
        actions = np.repeat((100 * rpc["replan_seq"] + np.arange(20))[:, None], 8, axis=1).astype(np.float32)
        return {"actions": actions, "eval_rpc": rpc}


def actor(xyz):
    return SimpleNamespace(pose=SimpleNamespace(p=np.asarray([xyz], dtype=np.float32)))


def make_unwrapped(way="grasp_putdown", cube=CUBE):
    return SimpleNamespace(elapsed_steps=None, way=way, agent=SimpleNamespace(tcp=actor([0.0, 0.0, 0.3])),
                           cube=actor(cube), peg_tail=actor(PEG), peg_head=actor(HEAD), goal_site=actor(GOAL))


def table_entry(way="grasp_putdown", **changes):
    actions = -np.ones((80, 8)) * (np.arange(1, 81)[:, None])
    entry = {"first_env_step": ORIGIN, "env_seed": 105, "way": way, "direction": force_table.WAYS[way][0],
             "peg_tail_xyz": np.float32(PEG).tolist(), "cube_xyz": np.float32(CUBE).tolist(),
             "goal_xyz": np.float32(GOAL).tolist(), "force_target": force_table.WAYS[way][1],
             "actions": actions.tolist()}
    entry.update(changes)
    return entry


def install_sim(monkeypatch, *, terminal_at=None, terminal_status="fail", picked=()):
    """terminal_at：第几个动作后环境终局；picked：{(物体属性名, 步后 elapsed)} 判定为抓起。"""
    def initial(self):
        self.env.actions = []
        self.env.unwrapped.elapsed_steps = ORIGIN
        state = np.arange(8, dtype=np.float32)
        images = [np.full((256, 256, 3), index, dtype=np.uint8) for index in range(4)]
        return {"images": images, "wrist_images": images, "states": [state.copy() for _ in images], "task_goal": "移动方块"}

    def step(self, action):
        u = self.env.unwrapped
        self.env.actions.append(np.asarray(action).copy())
        u.elapsed_steps += 1
        image = np.full((256, 256, 3), u.elapsed_steps % 255, dtype=np.uint8)
        done = len(self.env.actions) == terminal_at
        return (image, image, np.asarray(action, dtype=np.float32).copy()), done, terminal_status if done else "ongoing"

    def is_pickup(u, obj):
        return any(obj is getattr(u, name) and u.elapsed_steps == step for name, step in picked)

    monkeypatch.setattr(runner.EnvRunner, "get_init_obs", initial)
    monkeypatch.setattr(runner.EnvRunner, "step", step)
    monkeypatch.setattr(runner, "is_pickup", is_pickup)
    monkeypatch.setattr(runner.official, "RolloutRecorder", Recorder)


def make_runner(way="grasp_putdown", cube=CUBE, env_seed=105):
    result = runner.TraceRunner.__new__(runner.TraceRunner)
    result.env_id, result.episode_id = "MoveCube", 3
    result.env_seed, result.difficulty = env_seed, "easy"
    result.env = SimpleNamespace(unwrapped=make_unwrapped(way, cube))
    return result


def run_episode(root, *, k, way="grasp_putdown", entry=None, max_steps=1300, **env_changes):
    env = make_runner(way, **env_changes)
    args = runner.Args(force_k=k, max_steps=max_steps)
    evaluator = runner.TraceEvaluator(args, root)
    evaluator.table = {"MoveCube:3": entry or table_entry(way)}
    client = Client()
    outcome = evaluator.eval_each_episode(env, root / "videos", client, "nonce")
    rows = [json.loads(line) for line in (root / "traces/nonce/step_trace.jsonl").read_text().splitlines()]
    receipt = json.loads((root / "traces/nonce/meta/force_receipt.json").read_text())
    return env, evaluator, client, outcome, rows, receipt


@pytest.mark.parametrize("k", [0, 16, 80])
def test_replacement_and_release_boundary(tmp_path, monkeypatch, k):
    install_sim(monkeypatch, terminal_at=k + 20)
    env, evaluator, _, outcome, rows, receipt = run_episode(tmp_path, k=k)
    assert outcome == "fail" and evaluator.actions_executed == k + 20
    table = np.asarray(table_entry()["actions"])
    assert np.array_equal(env.env.actions[:k], table[:k]) if k else True
    # 放手步执行的是当前边界的新推理动作，不是积压动作。
    replan = k // 16
    assert np.array_equal(env.env.actions[k], np.full(8, 100 * replan))
    assert [row["control_source"] for row in rows] == ["force"] * k + ["policy"] * 20
    first_policy = rows[k]
    assert first_policy["env_step"] == ORIGIN + k and first_policy["row_in_chunk"] == 0 and first_policy["infer_seq"] == replan
    assert first_policy["command_action"] == first_policy["executed_action"]
    assert receipt["reached_release"] is True and receipt["actual_release_step"] == ORIGIN + k
    assert receipt["origin_env_step"] == ORIGIN and receipt["force_k"] == k and receipt["direction"] == "A"
    assert receipt["way"] == "grasp_putdown" and receipt["outcome"] == "fail" and receipt["env_status"] == "fail"
    assert receipt["grasp_success"] == 0 and receipt["grasp_step"] is None and receipt["actions_executed"] == k + 20
    required = {"event", "episode_key", "action_index", "env_step", "env_step_after", "infer_seq", "row_in_chunk",
                "chunk_start_env_step", "control_source", "command_action", "executed_action", "env_status",
                "tcp", "cube", "peg_tail", "peg_head"}
    assert all(required <= row.keys() for row in rows)


def test_policy_still_receives_real_observations_while_forced(tmp_path, monkeypatch):
    install_sim(monkeypatch, terminal_at=40)
    _, _, client, _, rows, _ = run_episode(tmp_path, k=32)
    # 强推期间官方循环照常续包、照常推理，动作被替换。
    assert client.frames == [4, 16, 16]
    assert rows[5]["command_action"] == [5] * 8 and rows[5]["executed_action"] == [-6] * 8


def test_grasp_ends_episode_as_success(tmp_path, monkeypatch):
    install_sim(monkeypatch, picked={("cube", ORIGIN + 16 + 5)})
    env, evaluator, _, outcome, rows, receipt = run_episode(tmp_path, k=16)
    assert outcome == "success" and evaluator.terminal_observed is True
    assert evaluator.actions_executed == 21 and len(env.env.actions) == 21
    assert receipt["grasp_success"] == 1 and receipt["grasp_step"] == ORIGIN + 21 and receipt["grasped_object"] == "cube"
    assert receipt["outcome"] == "success" and receipt["env_status"] == "ongoing"
    assert receipt["video_path"].endswith("MoveCube_ep3_success.mp4")
    assert rows[-1]["env_status"] == "ongoing"


def test_grasp_on_first_step_with_k_zero(tmp_path, monkeypatch):
    install_sim(monkeypatch, picked={("cube", ORIGIN + 1)})
    _, evaluator, _, outcome, _, receipt = run_episode(tmp_path, k=0)
    assert outcome == "success" and evaluator.actions_executed == 1 and receipt["grasp_step"] == ORIGIN + 1


def test_direction_objects(tmp_path, monkeypatch):
    # 方向 A 抓 peg 不算；方向 B 抓 peg 头或尾都算，抓 cube 不算。
    install_sim(monkeypatch, terminal_at=20, picked={("peg_tail", ORIGIN + 3), ("peg_head", ORIGIN + 3)})
    _, _, _, outcome, _, receipt = run_episode(tmp_path / "a", k=0)
    assert outcome == "fail" and receipt["grasp_success"] == 0
    install_sim(monkeypatch, terminal_at=20, picked={("cube", ORIGIN + 3)})
    _, _, _, outcome, _, receipt = run_episode(tmp_path / "b0", k=0, way="peg_push")
    assert outcome == "fail" and receipt["grasp_success"] == 0
    for name in ("peg_tail", "peg_head"):
        install_sim(monkeypatch, terminal_at=20, picked={(name, ORIGIN + 3)})
        _, _, _, outcome, _, receipt = run_episode(tmp_path / name, k=0, way="peg_push")
        assert outcome == "success" and receipt["grasped_object"] == name and receipt["direction"] == "B"


def test_env_fail_while_forced_is_recorded_without_raising(tmp_path, monkeypatch):
    install_sim(monkeypatch, terminal_at=10, picked={("cube", ORIGIN + 10)})
    env, _, _, outcome, rows, receipt = run_episode(tmp_path, k=48)
    assert outcome == "fail" and len(env.env.actions) == 10
    assert receipt["reached_release"] is False and receipt["actual_release_step"] is None
    assert receipt["grasp_success"] == 0 and receipt["env_status"] == "fail"
    assert all(row["control_source"] == "force" for row in rows)


def test_pickup_while_forced_does_not_count(tmp_path, monkeypatch):
    install_sim(monkeypatch, terminal_at=20, picked={("cube", ORIGIN + 5)})
    _, _, _, outcome, _, receipt = run_episode(tmp_path, k=16)
    assert outcome == "fail" and receipt["grasp_success"] == 0


def test_fail_on_grasp_step_stays_fail(tmp_path, monkeypatch):
    install_sim(monkeypatch, terminal_at=20, picked={("cube", ORIGIN + 20)})
    _, _, _, outcome, _, receipt = run_episode(tmp_path, k=0)
    assert outcome == "fail" and receipt["grasp_success"] == 0


def test_step_budget_times_out_through_official_loop(tmp_path, monkeypatch):
    install_sim(monkeypatch)
    _, evaluator, _, outcome, _, receipt = run_episode(tmp_path, k=16, max_steps=40)
    assert outcome == "timeout" and evaluator.actions_executed == 41
    assert receipt["outcome"] == "timeout" and receipt["grasp_success"] == 0 and receipt["reached_release"] is True


@pytest.mark.parametrize("changes", [
    {"entry": table_entry(first_env_step=ORIGIN + 1)}, {"env_seed": 106}, {"cube": [0.3, -0.1, 0.03]},
    {"entry": table_entry(way="peg_push")}, {"entry": table_entry(peg_tail_xyz=[0.1, 0.2, 0.04])},
    {"entry": table_entry(goal_xyz=[0.0, 0.0, 0.0])}, {"entry": table_entry(actions=[[0.0] * 8] * 79)},
])
def test_table_identity_mismatch_raises_and_keeps_receipt(tmp_path, monkeypatch, changes):
    install_sim(monkeypatch)
    with pytest.raises(runner.official.ContractError):
        run_episode(tmp_path, k=16, **changes)
    receipt = json.loads((tmp_path / "traces/nonce/meta/force_receipt.json").read_text())
    assert receipt["outcome"] is None and receipt["actions_executed"] == 0


def test_episode_missing_from_table_raises(tmp_path, monkeypatch):
    install_sim(monkeypatch)
    evaluator = runner.TraceEvaluator(runner.Args(force_k=16), tmp_path)
    evaluator.table = {}
    with pytest.raises(runner.official.ContractError, match="缺局"):
        evaluator.eval_each_episode(make_runner(), tmp_path / "videos", Client(), "nonce")


def test_evaluate_rejects_bad_k_and_other_tasks(tmp_path):
    with pytest.raises(runner.official.ContractError, match="force_k"):
        runner.evaluate(runner.Args(force_k=8, force_table="/table.json"))
    grant = tmp_path / "grant.json"
    grant.write_text(json.dumps({"chunk_id": "c", "run_identity_sha256": "a" * 64, "episodes": [
        {"task_name": "StopCube", "episode_id": 0, "episode_key": "StopCube:0", "env_seed": 1, "difficulty": "easy"}]}))
    with pytest.raises(runner.official.ContractError, match="只接受"):
        runner.evaluate(runner.Args(force_k=16, force_table="/table.json", episode_manifest=str(grant), chunk="c",
                                    run_identity_sha256="a" * 64))


# ---------- 动作表 ----------

def test_pad_actions_pad_truncate_and_gripper_open():
    path = np.arange(70, dtype=np.float64).reshape(10, 7)
    actions, n_path, padded, truncated = force_table.pad_actions(path)
    assert actions.shape == (80, 8) and n_path == 10 and padded and not truncated
    assert np.array_equal(actions[9:, :7], np.repeat(path[-1:], 71, axis=0)) and (actions[:, 7] == 1.0).all()
    long = np.zeros((89, 7))
    long[79], long[80] = 1.0, 2.0
    actions, n_path, padded, truncated = force_table.pad_actions(long)
    assert actions.shape == (80, 8) and n_path == 89 and truncated and not padded and (actions[-1, :7] == 1.0).all()
    assert not any(force_table.pad_actions(np.zeros((80, 7)))[2:])
    with pytest.raises(ValueError):
        force_table.pad_actions(np.zeros((0, 7)))


class TableEnv:
    def __init__(self, way, terminal_at=None):
        self.unwrapped = make_unwrapped(way)
        self.unwrapped.elapsed_steps = None
        self.way = way
        self.terminal_at = terminal_at
        self.steps = 0
        self.closed = False

    def reset(self):
        self.unwrapped.elapsed_steps = ORIGIN
        return {}, {}

    def step(self, action):
        self.steps += 1
        self.unwrapped.elapsed_steps += 1
        done = self.steps == self.terminal_at
        return {}, 0.0, done, False, {"status": "fail" if done else "ongoing"}

    def close(self):
        self.closed = True


class Builder:
    def __init__(self, ways, terminal_at=None):
        self.ways, self.terminal_at = ways, terminal_at
        self.envs = []

    def resolve_episode(self, episode_id):
        return 100 + episode_id, "easy"

    def make_env_for_episode(self, episode_id):
        self.envs.append(TableEnv(self.ways[episode_id], self.terminal_at))
        return self.envs[-1]


def write_table_manifest(path, ids, task="MoveCube"):
    path.write_text(json.dumps({"episodes": [{"task_name": task, "episode_id": i, "episode_key": f"{task}:{i}"} for i in ids]}))
    return path


def run_table(tmp_path, ways, capsys, planner=None, terminal_at=None, task="MoveCube"):
    manifest = write_table_manifest(tmp_path / "manifest.json", sorted(ways), task)
    out = tmp_path / "table.json"
    seen = []

    def default_planner(env, target):
        seen.append(target)
        return np.zeros((30, 7))

    builder = Builder(ways, terminal_at)
    code = force_table.main(["--manifest", str(manifest), "--out", str(out)], builder_factory=lambda: builder,
                            planner=planner or default_planner)
    return code, out, capsys.readouterr().out.strip().splitlines()[-1], seen, builder


def test_table_pass_routes_targets_and_loads(tmp_path, capsys):
    code, out, last, seen, builder = run_table(tmp_path, {1: "grasp_putdown", 2: "peg_push"}, capsys)
    assert code == 0 and last == "FORCE_TABLE=PASS episodes=2 plan_fail=0"
    assert seen[0] is builder.envs[0].unwrapped.peg_tail and seen[1] is builder.envs[1].unwrapped.cube
    assert all(env.closed and env.steps == 80 for env in builder.envs)
    table = runner.load_table(out)
    entry = table["MoveCube:2"]
    assert entry["direction"] == "B" and entry["force_target"] == "cube" and entry["env_seed"] == 102
    assert entry["first_env_step"] == ORIGIN and entry["n_path"] == 30 and entry["padded"] is True
    assert np.asarray(entry["actions"]).shape == (80, 8) and len(entry["open_loop"]["samples"]) == 5
    assert table["MoveCube:1"]["direction"] == "A" and table["MoveCube:1"]["force_target"] == "peg_tail"


def test_table_consumed_by_runner_identity_check(tmp_path, capsys, monkeypatch):
    _, out, _, _, _ = run_table(tmp_path, {3: "grasp_putdown"}, capsys)
    install_sim(monkeypatch, terminal_at=20)
    entry = runner.load_table(out)["MoveCube:3"]
    _, _, _, outcome, _, receipt = run_episode(tmp_path / "run", k=16, entry=dict(entry, env_seed=105))
    assert outcome == "fail" and receipt["reached_release"] is True


def test_table_rejects_gripper_push_and_other_task(tmp_path, capsys):
    code, out, last, _, _ = run_table(tmp_path, {1: "grasp_putdown", 2: "gripper_push"}, capsys)
    assert code == 1 and last == "FORCE_TABLE=FAIL episodes=2 plan_fail=0" and not out.exists()
    code, out, last, _, _ = run_table(tmp_path, {1: "grasp_putdown"}, capsys, task="StopCube")
    assert code == 1 and last == "FORCE_TABLE=FAIL episodes=1 plan_fail=0" and not out.exists()


def test_table_plan_failure_is_fail(tmp_path, capsys):
    code, out, last, _, _ = run_table(tmp_path, {1: "grasp_putdown"}, capsys, planner=lambda env, target: None)
    assert code == 1 and last == "FORCE_TABLE=FAIL episodes=1 plan_fail=1" and not out.exists()


def test_table_early_terminal_is_recorded_not_gated(tmp_path, capsys):
    code, out, last, _, _ = run_table(tmp_path, {1: "peg_push"}, capsys, terminal_at=20)
    assert code == 0 and last == "FORCE_TABLE=PASS episodes=1 plan_fail=0"
    loop = runner.load_table(out)["MoveCube:1"]["open_loop"]
    assert loop["early_terminal"] == {"step": 20, "status": "fail"} and [s["step"] for s in loop["samples"]] == [16, 20]


def test_load_table_rejects_bad_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 1, "force_steps": 16, "episodes": {}}))
    with pytest.raises(runner.official.ContractError):
        runner.load_table(path)


# ---------- run_local ----------

def base_spec(tmp_path, tasks=("MoveCube",), **changes):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": [{"task_name": task} for task in tasks]}))
    table = tmp_path / "force-table.json"
    table.write_text(json.dumps({"schema_version": 1, "force_steps": 80, "episodes": {}}))
    spec = dict(repo="/repo", policy_python="/policy/python", sim_python="/sim/python", stage2_python="/stage2/python",
                checkpoint="/ckpt", policy_config="cfg", model_kind="stage2_pool4_concat", cuda="1", run_root=str(tmp_path / "run"),
                manifest=str(manifest), xla_mem_fraction=0.3, model_seed=42, run_identity_sha256="a" * 64,
                movecube_force=True, force_table=str(table), force_k=16)
    spec.update(changes)
    return spec


def load(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return local.load_spec(path)


def test_force_ks_match_runner():
    assert local.FORCE_KS == runner.FORCE_KS


def test_load_spec_allows_missing_attn_out_only_in_force_mode(tmp_path):
    spec = base_spec(tmp_path)
    assert "attn_out" not in spec and load(tmp_path, spec)["force_k"] == 16
    with pytest.raises(SystemExit, match="attn_out"):
        load(tmp_path, dict(spec, movecube_force=False))


@pytest.mark.parametrize("changes", [
    {"stopcube_mode": "normal", "stopcube_config": "/x.json"}, {"bin_trace": True}, {"subgoal_trace": True},
    {"rollout_record": True}, {"force_k": 8}, {"force_k": "16"}, {"force_table": ""}, {"movecube_force": 1},
    {"ablate": "mask_motion"}, {"save_prefix": True}, {"save_attn": True},
])
def test_load_spec_rejects_invalid_force_spec(tmp_path, changes):
    with pytest.raises(SystemExit):
        load(tmp_path, base_spec(tmp_path, **changes))


def test_load_spec_rejects_non_movecube_manifest(tmp_path):
    with pytest.raises(SystemExit, match="MoveCube"):
        load(tmp_path, base_spec(tmp_path, tasks=("MoveCube", "StopCube")))


def test_server_command_uses_official_module(tmp_path):
    spec = base_spec(tmp_path)
    command = local.build_server_command(spec, "nonce", 9000)
    assert local.SERVER_MODULE in command and local.ATTN_MODULE not in command and "--attn-out" not in command
    assert command[command.index("--motion-online-gpu") + 1] == "1"
    baseline = base_spec(tmp_path, model_kind="history")
    command = local.build_server_command(baseline, "nonce", 9000)
    assert "--motion-online-gpu" not in command and local.ATTN_MODULE not in command and "--attn-out" not in command


def test_other_modes_commands_unchanged(tmp_path):
    spec = base_spec(tmp_path, attn_out="/out/attn", movecube_force=False)
    command = local.build_server_command(spec, "nonce", 9000)
    previous = dict(local.os.environ)
    local.os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    try:
        original = list(local.RW.server_command(spec, "nonce", 9000))
    finally:
        local.os.environ.clear()
        local.os.environ.update(previous)
    original[original.index(local.SERVER_MODULE)] = local.ATTN_MODULE
    index = original.index("policy:checkpoint")
    assert command == original[:index] + ["--attn-out", "/out/attn"] + original[index:]
    grant = tmp_path / "grant.json"
    grant.write_text(json.dumps({"chunk_id": "chunk"}))
    arguments = (grant, tmp_path / "out", "nonce", 9000, tmp_path / "drain")
    assert local.build_client_command(spec, *arguments) == local.RW.client_command(spec, *arguments)


def test_client_command_and_identity(tmp_path):
    spec = load(tmp_path, base_spec(tmp_path))
    grant = tmp_path / "grant.json"
    grant.write_text(json.dumps({"chunk_id": "chunk"}))
    command = local.build_client_command(spec, grant, tmp_path / "out", "nonce", 9000, tmp_path / "drain")
    assert command[1] == "/repo/scripts/interpret/movecube_force_runner.py"
    assert command[-4:] == ["--force-table", spec["force_table"], "--force-k", "16"]
    manifest = {"sha256": "b" * 64}
    first = local.run_identity(spec, manifest)
    assert first != local.run_identity(dict(spec, force_k=32), manifest)
    assert first != local.run_identity(dict(spec, movecube_force=False), manifest)
    Path(spec["force_table"]).write_text("{}")
    assert first != local.run_identity(spec, manifest)
    plain = {key: value for key, value in spec.items() if key not in ("movecube_force", "force_table", "force_k")}
    assert local.run_identity(plain, manifest) == local.run_identity(dict(plain, movecube_force=False), manifest)


@pytest.mark.parametrize("row,expected", [
    ({"force_k": 16, "terminal": True, "infers": 3, "npz": 0, "attn_saved": False}, True),
    ({"force_k": 16, "terminal": True, "infers": 0, "npz": 0, "attn_saved": False}, False),
    ({"force_k": 16, "terminal": False, "infers": None, "npz": 0, "attn_saved": False}, False),
])
def test_row_ok(row, expected):
    assert bool(local.row_ok(row)) is expected


def test_row_ok_other_modes_unchanged():
    assert local.row_ok({"attn_saved": False, "terminal": True, "episode_infers": 1, "npz": 0})
    assert not local.row_ok({"attn_saved": False, "terminal": True, "episode_infers": 0, "npz": 0})
    assert local.row_ok({"npz": 2, "infers": 2}) and not local.row_ok({"npz": 2, "infers": 3})


class Process:
    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def fake_service(monkeypatch, stopped=None, *, write_receipt=True, rpc_seq=7):
    """替身 popen：服务常驻；客户端按 grant 写 terminal 与回执后立刻退出。"""
    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        if command[0] == "/policy/python":
            return Process(None)
        grant = json.loads(Path(command[command.index("--episode-manifest") + 1]).read_text())
        output = Path(command[command.index("--attempt-output") + 1])
        output.mkdir(parents=True)
        for episode in grant["episodes"]:
            key, nonce = episode["episode_key"], f"n{episode['episode_id']:07d}"
            record = dict(episode, episode_nonce=nonce, record_type="started")
            (output / f"{key}.started.json").write_text(json.dumps(record))
            (output / f"{key}.terminal.json").write_text(json.dumps(dict(record, record_type="terminal", outcome="success", rpc_seq=rpc_seq)))
            if write_receipt:
                meta = output / "traces" / nonce / "meta"
                meta.mkdir(parents=True)
                (meta / "force_receipt.json").write_text(json.dumps({"grasp_success": 1, "env_status": "ongoing", "reached_release": True}))
        return Process(0)

    monkeypatch.setattr(local, "pick_port", lambda value: 9001)
    monkeypatch.setattr(local, "wait_ready", lambda *args: None)
    monkeypatch.setattr(local.RW, "stop_process_tree", lambda process: stopped.append(process) if stopped is not None and process else None)
    return calls, popen


def sealed_manifest(path, count, task="MoveCube"):
    episodes = [{"task_name": task, "episode_id": i, "env_seed": 100 + i, "difficulty": "easy"} for i in range(count)]
    path.write_text(json.dumps(M.make_manifest(episodes, official=False, chunk_size=count)))


def test_main_rejects_non_movecube_manifest(tmp_path):
    spec = base_spec(tmp_path)
    sealed_manifest(Path(spec["manifest"]), 2, task="StopCube")
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(SystemExit, match="MoveCube"):
        local.main(["--spec", str(path)])


def test_main_shared_service_batches_and_no_attention(tmp_path, monkeypatch, capsys):
    attn = tmp_path / "attn"
    spec = base_spec(tmp_path, attn_out=str(attn))
    sealed_manifest(Path(spec["manifest"]), 12)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    stopped = []
    calls, popen = fake_service(monkeypatch, stopped)
    assert local.main(["--spec", str(path)], popen=popen) == 0
    servers = [c for c in calls if c[0] == "/policy/python"]
    clients = [c for c in calls if c[0] == "/sim/python"]
    assert len(servers) == 1 and local.SERVER_MODULE in servers[0] and "--attn-out" not in servers[0]
    assert len(clients) == 3 and all(c[1].endswith("movecube_force_runner.py") for c in clients)
    sizes = [len(json.loads(Path(c[c.index("--episode-manifest") + 1]).read_text())["episodes"]) for c in clients]
    assert sizes == [5, 5, 2]
    assert not attn.exists(), "不得创建 attn_out"
    assert len(list((tmp_path / "run/attempts").glob("force-service-*"))) == 1
    assert len(list((tmp_path / "run/accepted").glob("*.terminal.json"))) == 12
    assert sum(process.returncode is None for process in stopped) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "RUN_LOCAL=PASS model=stage2_pool4_concat episodes=12 success=12 npz=0"
    summary = json.loads(next((tmp_path / "run").glob("run_local-*.json")).read_text())
    assert summary["results"][0]["grasp_success"] == 1 and summary["results"][0]["infers"] == 3


def test_main_fails_when_terminal_reports_no_inference(tmp_path, monkeypatch, capsys):
    spec = base_spec(tmp_path)
    sealed_manifest(Path(spec["manifest"]), 1)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    _, popen = fake_service(monkeypatch, rpc_seq=1)
    assert local.main(["--spec", str(path)], popen=popen) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("RUN_LOCAL=FAIL")
