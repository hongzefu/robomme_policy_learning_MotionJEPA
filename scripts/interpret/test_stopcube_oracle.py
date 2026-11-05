"""不建仿真，覆盖物理步记录、官方终局闭合和原 Oracle 调用顺序。"""
from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts.interpret.stopcube_oracle import (
    EpisodeFinished, OracleError, PhysicalRecorder, TerminalBoundary,
    drive_episode, hover_measurement, install_recorder, main, make_table_entry, press_result,
)


def state(step, z=0.15, pressed=False):
    return {"env_step": step, "qpos": [0.0] * 9, "state": [0.0] * 8,
            "tcp": [0.0, 0.0, z], "button_hover": True,
            "button_pressed": pressed, "cube_position": [0.0] * 3,
            "target_position": [0.0] * 3, "stop": pressed,
            "stop_timestep": step if pressed else None}


class RawEnv:
    def __init__(self):
        self.elapsed_steps = np.array([1])
        self.unwrapped = self
        self.agent = SimpleNamespace(robot=SimpleNamespace(get_qpos=lambda: np.zeros((1, 9))))
        self.button = object()

    def step(self, action):
        self.elapsed_steps += 1
        terminal = int(self.elapsed_steps[0]) >= 6
        return {}, 0, terminal, False, {"success": False}


class DemonstrationWrapper:
    """仿真官方递归额外一步的最小替身。"""

    def __init__(self, env):
        self.env = env

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def step(self, action):
        result = self.env.step(action)
        if result[2]:
            self.env.step(action)
        return (*result[:4], {**result[4], "status": "fail" if result[2] else "ongoing"})


def snapshot(env):
    return state(int(env.unwrapped.elapsed_steps[0]))


def recorded_run():
    raw = RawEnv()
    env = DemonstrationWrapper(raw)
    stream = io.StringIO()
    recorder = install_recorder(env, stream, "StopCube:0", snapshot)
    boundary = TerminalBoundary(env)
    with pytest.raises(EpisodeFinished):
        while True:
            boundary.step(np.zeros(8))
    return raw, recorder, boundary, stream


def definition():
    return {"env_seed": 42, "move_interval": 2, "target_pass": 2}


def geometry():
    return {"window_status": "resolved", "target_visit_in": 2,
            "target_visit_out": 4, "visits": [{"target_pass": 2, "in": 2, "out": 4}]}


def test_physical_recorder_includes_wrapper_extra_step_and_table_closes(tmp_path):
    raw, recorder, boundary, stream = recorded_run()
    assert int(raw.elapsed_steps[0]) == 7
    assert [r["env_step_before"] for r in recorder.rows] == list(range(1, 7))
    assert recorder.rows[-1]["terminated"]
    assert len([json.loads(line) for line in stream.getvalue().splitlines()]) == 6
    entry = make_table_entry(recorder.rows, definition(), geometry(), boundary.terminal, tmp_path / "window_probe.jsonl")
    assert entry["first_env_step"] == 1
    assert entry["last_env_step"] == 6
    assert entry["terminal_env_step"] == 7
    assert entry["terminal_env_step"] == entry["first_env_step"] + len(entry["actions"])
    assert entry["last_env_step"] > definition()["move_interval"] * definition()["target_pass"]
    np.testing.assert_array_equal(entry["actions"], np.zeros((6, 8)))


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows.pop(2), "不连续"),
    (lambda rows: rows[0].update(env_step_before=0), "一个物理步"),
    (lambda rows: rows[-1].update(button_pressed_after=True), "意外按下"),
])
def test_invalid_hold_table_rejected(tmp_path, mutation, message):
    _, recorder, boundary, _ = recorded_run()
    mutation(recorder.rows)
    with pytest.raises(OracleError, match=message):
        make_table_entry(recorder.rows, definition(), geometry(), boundary.terminal, tmp_path / "trace")


def test_missing_terminal_or_geometry_rejected(tmp_path):
    _, recorder, boundary, _ = recorded_run()
    with pytest.raises(OracleError, match="官方终局"):
        make_table_entry(recorder.rows, definition(), geometry(), None, tmp_path / "trace")
    with pytest.raises(OracleError, match="几何窗口"):
        make_table_entry(recorder.rows, definition(), {"window_status": "unresolved"}, boundary.terminal, tmp_path / "trace")


def test_hover_requires_three_stable_steps_and_reports_measured_median():
    _, recorder, _, _ = recorded_run()
    for row, z in zip(recorder.rows, [0.19, 0.17, 0.15, 0.1502, 0.1501, 0.15]):
        row["tcp_after"][2] = z
    reach, z_hover = hover_measurement(recorder.rows)
    assert reach == 4
    assert z_hover == pytest.approx(0.15005)
    assert hover_measurement(recorder.rows[:3]) == (None, None)


def test_hold_calls_only_ready_and_stays_to_official_terminal():
    raw = RawEnv()
    env = DemonstrationWrapper(raw)
    recorder = install_recorder(env, io.StringIO(), "StopCube:0", snapshot)
    boundary = TerminalBoundary(env)
    planner = SimpleNamespace(gripper_state=-1)
    solve_ready = Mock()
    assert drive_episode(env, planner, recorder, boundary, "hold", solve_ready) is None
    solve_ready.assert_called_once_with(raw, planner, obj=raw.button)
    assert boundary.terminal["status"] == "fail"
    assert all(row["phase"] == "hold" for row in recorder.rows)


def test_press_uses_original_task_solvers_in_order_and_stops_at_terminal():
    raw = RawEnv()
    env = DemonstrationWrapper(raw)
    recorder = install_recorder(env, io.StringIO(), "StopCube:0", snapshot)
    boundary = TerminalBoundary(env)
    calls = []
    def solve(name):
        def run(u, planner):
            assert u is raw
            calls.append(name)
            boundary.step(np.zeros(8))
        return run
    raw.task_list = [{"solve": solve(name)} for name in ("ready", "hold", "press")]
    command = drive_episode(env, SimpleNamespace(gripper_state=-1), recorder, boundary, "press", Mock())
    assert calls == ["ready", "hold", "press"]
    assert command == 3
    assert boundary.terminal is not None
    assert [r["phase"] for r in recorder.rows[:3]] == ["ready", "hold", "press"]


def test_press_events_keep_trigger_stop_success_separate():
    _, recorder, _, _ = recorded_run()
    for row in recorder.rows[3:]:
        row["phase"] = "press"
        row["tcp_before"][2] = 0.15
        row["tcp_after"][2] = 0.12
    recorder.rows[4].update(button_pressed_after=True, stop_timestep=None)
    recorder.rows[5].update(button_pressed_before=True, button_pressed_after=True, stop_timestep=6)
    result = press_result(recorder.rows, definition(), {"task_success": True}, 4, geometry())
    assert result["press_command_step"] == 4
    assert result["actual_descent_step"] == 5
    assert result["actual_press_onset"] == 5
    assert result["button_trigger_step"] == 6
    assert result["stop_timestep"] == 6
    assert result["trigger_in_window"] is False
    assert result["task_success"] is True


def test_recorder_rejects_nonfinite_action_and_env_errors():
    raw = RawEnv()
    recorder = PhysicalRecorder(raw, io.StringIO(), "StopCube:0", snapshot)
    with pytest.raises(OracleError, match="有限 8 维"):
        recorder.step([float("nan")] * 8)
    fake = SimpleNamespace(step=lambda _: ({}, 0, True, False, {"status": "error", "error_message": "错误证据"}))
    with pytest.raises(OracleError, match="错误证据"):
        TerminalBoundary(fake).step(np.zeros(8))


def test_cli_requires_single_official_episode_and_hold_table(tmp_path):
    common = ["--output-dir", str(tmp_path), "--benchmark-root", str(tmp_path)]
    with pytest.raises(SystemExit) as exc:
        main(["--episode", "50", "--mode", "hold", *common])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["--episode", "16", "--mode", "press", *common])
    assert exc.value.code == 2
