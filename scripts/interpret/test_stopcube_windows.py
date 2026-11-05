"""窗口与运动事件的 CPU 定向测试，不启动模型或仿真。"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.interpret.stopcube_windows import analyze, extract_events, geometry_window, main, read_trace


def definition():
    return {"env_seed": 7, "move_interval": 20, "target_pass": 2, "cube_half_size": 0.02,
            "hover_z": 0.15, "hover_source": {"symbol": "solve_button_ready::ready_position[2]",
            "git_commit": "合成测试"}, "button_position": [0, 0, 0]}


def trace(kind="normal", final=12):
    meta = {"episode_key": "StopCube/7", "trajectory_kind": kind, "task_success": False, **definition()}
    rows = []
    for before in range(final):
        after = before + 1
        row = {"event": "step", "episode_key": "StopCube/7", "trajectory_kind": kind,
            "env_step_before": before, "env_step_after": after, "infer_seq": before // 4,
            "executed_target_tcp_z": 0.12 if before >= 6 else 0.15,
            "command_action": [0.1] * 8, "executed_action": [0.1] * 8,
            "state_before": [before / 100] * 8, "state_after": [after / 100] * 8,
            "tcp_before": [0, 0, 0.12 if before >= 8 else 0.15],
            "tcp_after": [0, 0, 0.12 if after >= 8 else 0.15],
            "button_pressed_before": before >= 9, "button_pressed_after": after >= 9,
            "stop_timestep": 10 if after >= 10 else None}
        if kind == "window_probe":
            row.update(button_pressed_before=False, button_pressed_after=False, stop_timestep=None)
        for suffix, step in (("before", before), ("after", after)):
            # 合成几何故意与线性公式不同，验证输出确实消费位置。
            row[f"cube_position_{suffix}"] = [abs(step % 20 - 10) / 100, 0, 0]
            row[f"target_position_{suffix}"] = [0, 0, 0]
        rows.append(row)
    return {"meta": meta, "rows": rows, "source": {"path": f"合成/{kind}", "sha256": "测试"}}


class StopCubeWindowsTests(unittest.TestCase):
    def test_distinct_command_tcp_button_and_stop_steps(self):
        result = extract_events(trace(), definition())
        self.assertEqual(result["events"]["command_press_step"], 6)
        self.assertEqual(result["events"]["actual_press_onset"], 8)
        self.assertEqual(result["events"]["button_trigger_step"], 9)
        self.assertEqual(result["events"]["stop_timestep"], 10)
        self.assertEqual(result["measured_sample"]["motion_delay"], 3)

    def test_real_geometry_and_same_episode_failed_measurement(self):
        windows, calibration = analyze([trace(), trace("window_probe", 40)])
        window, cal = windows["episodes"]["StopCube/7"], calibration["episodes"]["StopCube/7"]
        self.assertEqual(window["target_visit_in"], 24)
        self.assertEqual(window["target_visit_out"], 36)
        self.assertEqual(window["start_range"], [21, 33])
        self.assertNotEqual(window["formula_crosscheck"]["linear_range"], [24, 36])
        self.assertEqual(cal["status"], "resolved")
        self.assertEqual(cal["calibration_source"], "same_episode_measured")
        self.assertIs(cal["calibration_evidence"]["task_success"], False)
        self.assertIn("window_probe", cal["trajectories"])

    def test_early_termination_cannot_supply_window(self):
        windows, calibration = analyze([trace()])
        self.assertEqual(windows["episodes"]["StopCube/7"]["start_range"], [None, None])
        self.assertEqual(windows["episodes"]["StopCube/7"]["status"], "unresolved")
        self.assertEqual(calibration["episodes"]["StopCube/7"]["motion_delay"], 3)

    def test_no_fk_keeps_command_unresolved_not_actual_onset(self):
        item = trace()
        for row in item["rows"]:
            del row["executed_target_tcp_z"]
        _, calibration = analyze([item])
        cal = calibration["episodes"]["StopCube/7"]
        self.assertIsNone(cal["events"]["command_press_step"])
        self.assertEqual(cal["events"]["actual_press_onset"], 8)
        self.assertIsNone(cal["motion_delay"])

    def test_no_button_never_uses_terminal_or_zero(self):
        item = trace()
        for row in item["rows"]:
            row.update(button_pressed_before=False, button_pressed_after=False)
        _, calibration = analyze([item])
        cal = calibration["episodes"]["StopCube/7"]
        self.assertIsNone(cal["motion_delay"])
        self.assertEqual(cal["status"], "unresolved")

    def test_initial_press_without_hover_is_explicit(self):
        item = trace()
        for row in item["rows"]:
            row["executed_target_tcp_z"] = 0.12
        _, calibration = analyze([item])
        cal = calibration["episodes"]["StopCube/7"]
        self.assertEqual(cal["events"]["command_press_step"], 0)
        self.assertEqual(cal["status"], "unresolved")
        self.assertTrue(any("悬停" in reason for reason in cal["reasons"]))

    def test_hover_reference_requires_provenance(self):
        item = trace()
        del item["meta"]["hover_source"]
        _, calibration = analyze([item])
        self.assertEqual(calibration["episodes"]["StopCube/7"]["status"], "unresolved")

    def test_stopped_cube_never_extends_geometric_window(self):
        probe = trace("window_probe", 40)
        for row in probe["rows"]:
            if row["env_step_after"] >= 30:
                row["button_pressed_after"] = True
            if row["env_step_before"] >= 30:
                row["button_pressed_before"] = True
        self.assertEqual(geometry_window([probe], definition())["window_status"], "unresolved")

    def test_seed_mismatch_and_duplicate_kind_reject(self):
        other = trace("window_probe", 40)
        other["meta"]["env_seed"] = 8
        with self.assertRaisesRegex(ValueError, "env_seed"):
            analyze([trace(), other])
        with self.assertRaisesRegex(ValueError, "重复轨迹"):
            analyze([trace(), copy.deepcopy(trace())])

    def test_trajectory_events_stay_separate(self):
        extended = trace("hold_extended")
        for row in extended["rows"]:
            row.pop("executed_target_tcp_z")
        _, calibration = analyze([trace(), extended])
        cal = calibration["episodes"]["StopCube/7"]
        self.assertEqual(cal["events"]["command_press_step"], 6)
        self.assertIsNone(cal["trajectories"]["hold_extended"]["events"]["command_press_step"])

    def test_terminal_extra_step_is_interval_not_exact_calibration(self):
        item = trace()
        row = item["rows"][8]
        row["env_step_after"] = 10
        item["rows"] = item["rows"][:9]
        _, calibration = analyze([item])
        cal = calibration["episodes"]["StopCube/7"]
        self.assertEqual(cal["events"]["button_trigger_bounds"], [9, 10])
        self.assertEqual(cal["status"], "unresolved")
        self.assertIsNone(cal["motion_delay"])

    def test_jsonl_cli_success_and_controlled_rejection(self):
        root = os.environ.get("STOPCUBE_TEST_ROOT")
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            folder = Path(tmp)
            paths = []
            for kind, final in (("normal", 12), ("window_probe", 40)):
                item = trace(kind, final)
                path = folder / f"{kind}.jsonl"
                path.write_text("\n".join(json.dumps(row) for row in [{"event": "meta", **item["meta"]}, *item["rows"]]))
                paths.extend(["--trace", str(path)])
            out = folder / "output"
            self.assertEqual(main([*paths, "--output-dir", str(out)]), 0)
            self.assertTrue((out / "windows.json").is_file())
            self.assertEqual(main([*paths[:2], "--output-dir", str(out)]), 1)
            item = trace()
            item["rows"][3]["env_step_before"] = 4
            broken = folder / "broken.jsonl"
            broken.write_text("\n".join(json.dumps(row) for row in [{"event": "meta", **item["meta"]}, *item["rows"]]))
            with self.assertRaises(ValueError):
                read_trace(broken)


if __name__ == "__main__":
    unittest.main()
