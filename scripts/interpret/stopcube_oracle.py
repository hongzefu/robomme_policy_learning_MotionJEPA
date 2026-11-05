"""每进程一局，生成 StopCube 官方 Oracle 悬停动作表或真下压记录。

仿真依赖只在 CLI 建环境时导入；动作表中的一行严格对应一个物理步，
包括 DemonstrationWrapper 在终止时追加的那一步。无模型、无自动重试。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

from scripts.interpret.stopcube_windows import BENCHMARK_PIN, geometry_window, read_trace, task_definition


class OracleError(ValueError):
    """Oracle 数据或现场步制不满足约定。"""


class EpisodeFinished(Exception):
    """外层官方终局已返回，停止规划器余下动作。"""


def plain(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1).tolist()


def scalar(value):
    values = plain(value)
    return values[0] if values else None


def physical_snapshot(env):
    """只读物理状态，避免 evaluate 写入 stop_timestep。"""
    from robomme.robomme_env.utils.subgoal_evaluate_func import button_hover, is_button_pressed

    u = env.unwrapped
    qpos = plain(u.agent.robot.get_qpos())
    button = u.button
    return {
        "env_step": int(scalar(u.elapsed_steps)),
        "qpos": qpos,
        "qvel": plain(u.agent.robot.get_qvel()),
        "state": qpos[:7] + qpos[7:8],
        "tcp": plain(u.agent.tcp.pose.p),
        "tcp_quaternion": plain(u.agent.tcp.pose.q),
        "cube_position": plain(u.cube.pose.p),
        "cube_quaternion": plain(u.cube.pose.q),
        "target_position": plain(u.target.pose.p),
        "button_position": plain(button.pose.p),
        "button_pressed": bool(is_button_pressed(u, obj=button)),
        "button_hover": bool(button_hover(u, button=button)),
        "stop": bool(u.stop),
        "stop_timestep": scalar(u.stop_timestep),
        "task_index": int(getattr(u, "timestep", 0)),
    }


class EnvProxy:
    """组合式转发；不替换第三方类、方法或任务判定。"""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)


class PhysicalRecorder(EnvProxy):
    """置于 DemonstrationWrapper 下层，记录包括其递归额外步的实际动作。"""

    def __init__(self, env, stream, episode_key, snapshot_fn=physical_snapshot):
        super().__init__(env)
        self.stream, self.episode_key, self.snapshot_fn = stream, episode_key, snapshot_fn
        self.rows = []
        self.phase = "ready"
        self.trajectory_kind = "window_probe"

    def step(self, action):
        before = self.snapshot_fn(self.env)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise OracleError("实际动作必须为有限 8 维关节动作")
        result = self.env.step(action)
        after = self.snapshot_fn(self.env)
        if after["env_step"] != before["env_step"] + 1:
            raise OracleError("记录层必须一次只推进一个物理步")
        if self.rows and before["env_step"] != self.rows[-1]["env_step_after"]:
            raise OracleError("物理记录存在遗漏或重复步")
        row = {"event": "step", "episode_key": self.episode_key,
               "trajectory_kind": self.trajectory_kind, "phase": self.phase,
               "executed_action": action.tolist(), "command_action": action.tolist(),
               "stop_timestep": after["stop_timestep"],
               "terminated": bool(scalar(result[2])), "truncated": bool(scalar(result[3])),
               "task_success": scalar(result[4].get("success"))}
        for key in before:
            row[f"{key}_before"], row[f"{key}_after"] = before[key], after[key]
        self.rows.append(row)
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        return result


class TerminalBoundary(EnvProxy):
    """终局在官方包装器完成额外步并返回后才停止。"""

    terminal = None

    def step(self, action):
        result = self.env.step(action)
        info = result[4]
        if info.get("status") == "error":
            raise OracleError(info.get("error_message", "官方环境报告错误"))
        if bool(scalar(result[2])) or bool(scalar(result[3])):
            self.terminal = {"status": info.get("status"),
                             "task_success": scalar(info.get("success")),
                             "terminated": bool(scalar(result[2])),
                             "truncated": bool(scalar(result[3]))}
            raise EpisodeFinished
        return result


def install_recorder(env, stream, episode_key, snapshot_fn=physical_snapshot):
    """仅在本局包装器链中增加记录层，保持原来的官方控制路径。"""
    current = env
    while type(current).__name__ != "DemonstrationWrapper":
        if "env" not in vars(current):
            raise OracleError("环境链缺少官方 DemonstrationWrapper")
        current = current.env
    recorder = PhysicalRecorder(current.env, stream, episode_key, snapshot_fn)
    current.env = recorder
    return recorder


def hover_measurement(rows):
    """三个连续实测步满足官方 hover 且高度极差不超过 1 毫米才算到位。"""
    candidates = []
    for row in rows:
        if row["button_hover_after"] and not row["button_pressed_after"]:
            candidates.append((row["env_step_after"], row["tcp_after"][2]))
        else:
            candidates = []
        recent = candidates[-3:]
        if (len(recent) == 3 and recent[-1][0] - recent[0][0] == 2
                and max(z for _, z in recent) - min(z for _, z in recent) <= 0.001):
            reach = recent[0][0]
            heights = [r["tcp_after"][2] for r in rows
                       if r["env_step_after"] >= reach and r["phase"] != "press"
                       and r["button_hover_after"] and not r["button_pressed_after"]]
            return reach, statistics.median(heights)
    return None, None


def make_table_entry(rows, definition, geometry, terminal, trace_path):
    """验证 4.1 步制闭合；失败仍由调用方保留原始 trace 和回执。"""
    if not rows or terminal is None:
        raise OracleError("动作表缺少实际动作或官方终局")
    for index, row in enumerate(rows):
        if row["env_step_before"] != rows[0]["env_step_before"] + index:
            raise OracleError("动作表物理步不连续")
        if row["env_step_after"] != row["env_step_before"] + 1:
            raise OracleError("动作表每行必须对应一个物理步")
    if rows[0]["env_step_before"] != 1:
        raise OracleError("官方初始化后首个物理步必须为 1")
    if any(r["button_pressed_before"] or r["button_pressed_after"] or r["stop_after"] for r in rows):
        raise OracleError("只悬停轨迹意外按下按钮")
    if geometry["window_status"] != "resolved":
        raise OracleError("真实几何窗口尚未完整解析")
    reach, z_hover = hover_measurement(rows)
    if reach is None:
        raise OracleError("未观察到高度稳定的官方按钮悬停姿态")
    center = definition["move_interval"] * (definition["target_pass"] - 0.5)
    return {**definition,
            "first_env_step": rows[0]["env_step_before"],
            "last_env_step": rows[-1]["env_step_before"],
            "terminal_env_step": rows[-1]["env_step_after"],
            "actions": [r["executed_action"] for r in rows],
            "reach_step": reach, "z_hover": z_hover, "late_reach": reach > center - 30,
            "target_visit_in": geometry["target_visit_in"],
            "target_visit_out": geometry["target_visit_out"], "visits": geometry["visits"],
            "trace_path": str(Path(trace_path).resolve()), "terminal": terminal}


def press_result(rows, definition, terminal, command_step, hold_entry):
    """真实下压、按钮触发、官方停步和成功分别报告，不设成功数阈值。"""
    reach, z_hover = hover_measurement(rows)
    press_rows = [r for r in rows if r["phase"] == "press"]
    descent = next((r["env_step_after"] for r in press_rows
                    if r["tcp_after"][2] < r["tcp_before"][2] - 0.0001), None)
    onset = next((r["env_step_after"] for r in press_rows
                  if z_hover is not None and r["tcp_after"][2] < z_hover - 0.02), None)
    trigger = next((r["env_step_after"] for r in rows
                    if not r["button_pressed_before"] and r["button_pressed_after"]), None)
    stop = next((r["stop_timestep"] for r in rows if r["stop_timestep"] is not None), None)
    inside = None if trigger is None else hold_entry["target_visit_in"] <= trigger <= hold_entry["target_visit_out"]
    return {**definition, "terminal": terminal, "reach_step": reach, "z_hover": z_hover,
            "press_command_step": command_step, "actual_descent_step": descent,
            "actual_press_onset": onset, "button_trigger_step": trigger, "stop_timestep": stop,
            "target_visit_in": hold_entry["target_visit_in"], "target_visit_out": hold_entry["target_visit_out"],
            "trigger_in_window": inside,
            "stop_in_window": None if stop is None else hold_entry["target_visit_in"] <= stop <= hold_entry["target_visit_out"],
            "task_success": None if terminal is None else terminal["task_success"]}


def drive_episode(env, planner, recorder, boundary, mode, solve_ready):
    """悬停只求一次 ready；下压严格依次调用该局原 task_list 的 solve。"""
    u, command_step = env.unwrapped, None
    try:
        if mode == "hold":
            solve_ready(u, planner, obj=u.button)
            recorder.phase = "hold"
        else:
            for index, task in enumerate(u.task_list):
                recorder.phase = "press" if index == len(u.task_list) - 1 else ("ready" if index == 0 else "hold")
                if recorder.phase == "press":
                    command_step = int(scalar(u.elapsed_steps))
                task["solve"](u, planner)
        # Oracle 规划或下压回程返回后保持当前实测关节，直至官方终局。
        action = np.asarray(plain(u.agent.robot.get_qpos())[:7] + [planner.gripper_state], dtype=np.float64)
        while boundary.terminal is None:
            boundary.step(action)
    except EpisodeFinished:
        pass
    return command_step


def generate_episode(episode_id, mode, output_dir, benchmark_root, hold_table=None):
    """真实入口；调用者负责 GPU、时限、目录归属与进程调度。"""
    benchmark_root = Path(benchmark_root).resolve()
    sys.path.insert(0, str(benchmark_root / "src"))
    from scripts.interpret.read_sim_eef_pose import build_env
    from robomme.robomme_env.utils.planner_fail_safe import FailAwarePandaArmMotionPlanningSolver
    from robomme.robomme_env.utils.subgoal_planner_func import solve_button_ready

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("window_probe.jsonl", "oracle_receipt.json", "oracle_table.json", "oracle_press.json"):
        if (output_dir / name).exists():
            raise OracleError(f"不覆盖已有 Oracle 产物：{output_dir / name}")
    key = f"StopCube:{episode_id}"
    hold_entry = None
    if mode == "press":
        table = json.loads(Path(hold_table).read_text(encoding="utf-8"))
        if table.get("schema_version") != 1 or table.get("benchmark_commit") != BENCHMARK_PIN:
            raise OracleError("真下压须读取同一锁定 benchmark 的悬停表")
        hold_entry = table["episodes"][key]
    env, seed, difficulty = build_env("StopCube", episode_id)
    receipt = {"schema_version": 1, "episode_key": key, "mode": mode, "status": "failed"}
    try:
        env.reset(seed=seed)
        definition = task_definition(env, benchmark_root)
        definition.update(episode_key=key, episode_id=episode_id, env_seed=seed, difficulty=difficulty)
        if hold_entry is not None:
            for field in ("env_seed", "difficulty", "move_interval", "target_pass"):
                if hold_entry[field] != definition[field]:
                    raise OracleError(f"真下压与悬停表的 {field} 不一致")
        trace_path = output_dir / "window_probe.jsonl"
        with trace_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "meta", "trajectory_kind": "window_probe" if mode == "hold" else "oracle_press",
                                     **definition}, ensure_ascii=False) + "\n")
            recorder = install_recorder(env, stream, key)
            recorder.trajectory_kind = "window_probe" if mode == "hold" else "oracle_press"
            boundary = TerminalBoundary(env)
            planner = FailAwarePandaArmMotionPlanningSolver(boundary, debug=False, vis=False,
                base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=False, print_env_info=False)
            command_step = drive_episode(env, planner, recorder, boundary, mode, solve_button_ready)
            stream.write(json.dumps({"event": "terminal", **(boundary.terminal or {})}, ensure_ascii=False) + "\n")
        if mode == "hold":
            geometry = geometry_window([read_trace(trace_path)], definition)
            entry = make_table_entry(recorder.rows, definition, geometry, boundary.terminal, trace_path)
            table_path = output_dir / "oracle_table.json"
            with table_path.open("x", encoding="utf-8") as stream:
                json.dump({"schema_version": 1, "benchmark_commit": BENCHMARK_PIN,
                    "episodes": {key: entry}}, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            receipt.update(status="complete", table_path=str(table_path.resolve()), late_reach=entry["late_reach"])
        else:
            if boundary.terminal is None:
                raise OracleError("真下压缺少官方终局")
            result = press_result(recorder.rows, definition, boundary.terminal, command_step, hold_entry)
            with (output_dir / "oracle_press.json").open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            receipt.update(status="complete", **result)
        receipt.update(trace_path=str(trace_path.resolve()), actions_executed=len(recorder.rows))
        return receipt
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            with (output_dir / "oracle_receipt.json").open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        finally:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--episode-key", help="官方局身份，例如 StopCube:16")
    identity.add_argument("--episode", type=int, help="局号别名，例如 16")
    parser.add_argument("--mode", choices=("hold", "press"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--hold-table", type=Path)
    args = parser.parse_args(argv)
    episode_id = args.episode
    if args.episode_key is not None:
        task, separator, number = args.episode_key.partition(":")
        if task != "StopCube" or not separator or not number or any(c not in "0123456789" for c in number):
            parser.error("--episode-key 必须为 StopCube:<官方局号>，例如 StopCube:16")
        episode_id = int(number)
    if not 0 <= episode_id < 50:
        parser.error("官方测试局号范围为 0～49")
    if args.mode == "press" and args.hold_table is None:
        parser.error("press 模式必须指定 --hold-table")
    receipt = generate_episode(episode_id, args.mode, args.output_dir, args.benchmark_root, args.hold_table)
    if args.mode == "hold":
        print(f"ORACLE_TABLE=PASS episodes=1 failed=0 late_reach={int(receipt['late_reach'])}", flush=True)
    else:
        print(f"ORACLE_PRESS=PASS episodes=1 success={int(receipt['task_success'] == 1)} "
              f"trigger_outside={int(receipt['trigger_in_window'] is False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
