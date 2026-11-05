"""从真实逐步记录生成 StopCube 窗口和机械臂提前量，缺证据时保持待确认。

普通分析只用标准库。--fk 显式建同局仿真计算已执行动作的目标 TCP，
--probe-episode 显式运行不按按钮的窗口仿真；两者由主会话分配资源后运行。
公式只列为交叉检查，窗口端点来自真实采样位置的官方水平距离判据。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import inspect
from pathlib import Path
import subprocess


SCHEMA_VERSION = 1
BENCHMARK_PIN = "856bc3a189d4172f3f47dbee4424d585f8d78db3"


def fingerprint(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value, length):
    return isinstance(value, (list, tuple)) and len(value) == length and all(finite(x) for x in value)


def is_task_success(value):
    """仅 JSON 布尔值与整数 0/1 是合法成功字段，保留原始值供审查。"""
    return type(value) in (bool, int) and value == 1


def _plain(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "reshape"):
        return value.reshape(-1).tolist()
    return value


def read_trace(path):
    """读取一条轨迹及旁边的回执；不把不同轨迹的时刻拼成一条。"""
    path = Path(path)
    meta, rows, metadata_sources = {}, [], []
    for name in ("meta.json", "hold_receipt.json", "meta/hold_receipt.json"):
        sidecar = path.parent / name
        if sidecar.is_file():
            meta.update(json.loads(sidecar.read_text()))
            metadata_sources.append(fingerprint(sidecar))
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "step":
            if not all(isinstance(row.get(k), int) for k in ("env_step_before", "env_step_after")):
                raise ValueError(f"{path} 第 {number} 行缺少整数步制")
            if row["env_step_after"] <= row["env_step_before"]:
                raise ValueError(f"{path} 第 {number} 行环境步没有递增")
            if rows and row["env_step_before"] != rows[-1]["env_step_after"]:
                raise ValueError(f"{path} 第 {number} 行存在重复或遗漏环境步")
            rows.append(row)
        else:
            meta.update(row)
    if rows:
        for key in ("episode_key", "trajectory_kind", "env_seed", "move_interval", "target_pass", "episode_id"):
            if key not in meta and key in rows[0]:
                meta[key] = rows[0][key]
    if not meta.get("episode_key"):
        raise ValueError(f"{path} 缺少 episode_key")
    meta.setdefault("trajectory_kind", "normal")
    extra = meta.get("extra") or {}
    for key, value in extra.items():
        meta.setdefault(key, value)
    return {"meta": meta, "rows": rows, "source": fingerprint(path), "metadata_sources": metadata_sources}


def task_definition(env, benchmark_root):
    """核对锁定源码，读取该 seed 实际任务参数及官方悬停目标定义。"""
    root = Path(benchmark_root)
    pin = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if pin != BENCHMARK_PIN:
        raise ValueError(f"benchmark 提交不符：{pin}")
    actual_module = Path(inspect.getfile(type(env.unwrapped))).resolve()
    expected_module = (root / "src/robomme/robomme_env/StopCube.py").resolve()
    if actual_module != expected_module:
        raise ValueError(f"实际任务导入不来自指定 benchmark：{actual_module}")
    relative = "src/robomme/robomme_env/utils/subgoal_planner_func.py"
    source = root / relative
    committed = subprocess.check_output(["git", "-C", str(root), "show", f"{BENCHMARK_PIN}:{relative}"])
    if source.read_bytes() != committed:
        raise ValueError("按钮悬停定义存在未提交修改")
    hover_z = None
    for function in ast.parse(committed).body:
        if isinstance(function, ast.FunctionDef) and function.name == "solve_button_ready":
            for node in ast.walk(function):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    if any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                           and t.value.id == "ready_position" and isinstance(t.slice, ast.Constant)
                           and t.slice.value == 2 for t in node.targets):
                        hover_z = node.value.value
    if not finite(hover_z):
        raise ValueError("不能从锁定 solve_button_ready 读取悬停高度")
    u = env.unwrapped
    return {"move_interval": int(u.move_interval), "target_pass": int(u.stop_time),
            "cube_half_size": float(u.cube_half_size), "button_position": _plain(u.button.pose.p),
            "hover_z": float(hover_z), "hover_source": {**fingerprint(source), "git_commit": pin,
            "symbol": "solve_button_ready::ready_position[2]"}}


def geometry_window(traces, definition):
    """仅接受未按按钮的连续完整经过，端点是满足官方几何判据的整数步。"""
    interval, target_pass = definition.get("move_interval"), definition.get("target_pass")
    half = definition.get("cube_half_size")
    result = {"target_visit_in": None, "target_visit_out": None, "visits": [],
              "window_source": None, "window_status": "unresolved", "window_reasons": []}
    if not (isinstance(interval, int) and interval > 0 and isinstance(target_pass, int)
            and target_pass > 0 and finite(half) and half > 0):
        result["window_reasons"].append("缺少每局运动周期、目标经过次数或方块尺寸")
        return result
    lo, hi = (target_pass - 1) * interval, target_pass * interval
    for trace in traces:
        samples = {}
        pressed = False
        for row in trace["rows"]:
            if row.get("button_pressed_before") is True:
                pressed = True
            if pressed:
                break
            for suffix in ("before", "after"):
                if suffix == "after" and row.get("button_pressed_after") is True:
                    pressed = True
                    continue
                cube, target = row.get(f"cube_position_{suffix}"), row.get(f"target_position_{suffix}")
                if vector(cube, 3) and vector(target, 3):
                    samples[row[f"env_step_{suffix}"]] = math.hypot(cube[0] - target[0], cube[1] - target[1])
        # 经过两侧必须都有真实样本；提前终止、已停方块均不能补成完整窗口。
        complete = all(t in samples for t in range(lo, hi + 1))
        inside = [t for t in range(lo, hi + 1) if t in samples and samples[t] <= 3 * half]
        if not complete or not inside:
            continue
        begin, end = min(inside), max(inside)
        if inside != list(range(begin, end + 1)):
            continue
        result.update(target_visit_in=begin, target_visit_out=end, window_source=trace["source"],
                      window_status="resolved")
        # 旧的线性公式只供人工发现不一致，绝不决定窗口。
        center = interval * (target_pass - 0.5)
        result["formula_crosscheck"] = {"linear_center": center,
            "linear_range": [center - interval * 3 * half / 0.6, center + interval * 3 * half / 0.6],
            "role": "crosscheck_only"}
        for visit in range(1, hi // interval + 1):
            points = [t for t in range((visit - 1) * interval, visit * interval + 1)
                      if t in samples and samples[t] <= 3 * half]
            if points:
                result["visits"].append({"target_pass": visit, "in": min(points), "out": max(points)})
        return result
    result["window_reasons"].append("没有覆盖目标经过全周期且未按按钮的真实几何记录")
    return result


def extract_events(trace, definition, delta_z=0.02):
    """动作命令在 before 发出，TCP／按钮实测在 after，官方停步单列。"""
    hover_z = definition.get("hover_z")
    button = definition.get("button_position")
    threshold = hover_z - delta_z if finite(hover_z) else None
    events = {"hover_steps": [], "command_press_step": None, "press_infer": None,
              "actual_press_onset": None, "actual_press_onset_bounds": None,
              "button_trigger_step": None, "button_trigger_bounds": None, "stop_timestep": None}
    reasons = []
    has_command_tcp = False
    for row in trace["rows"]:
        before, after = row["env_step_before"], row["env_step_after"]
        tcp0, tcp1 = row.get("tcp_before"), row.get("tcp_after")
        target_z = row.get("executed_target_tcp_z")
        if not finite(target_z) and vector(row.get("executed_target_tcp"), 3):
            target_z = row["executed_target_tcp"][2]
        if finite(target_z):
            has_command_tcp = True
            if threshold is not None and target_z < threshold and events["command_press_step"] is None:
                events["command_press_step"] = before
                events["press_infer"] = row.get("infer_seq")
        if threshold is not None and vector(tcp0, 3) and vector(tcp1, 3):
            if tcp0[2] >= threshold and tcp1[2] < threshold and events["actual_press_onset"] is None:
                events["actual_press_onset"] = after
                events["actual_press_onset_bounds"] = [before + 1, after]
            if (abs(tcp1[2] - hover_z) <= delta_z and vector(button, 3)
                    and math.hypot(tcp1[0] - button[0], tcp1[1] - button[1]) <= 0.03
                    and row.get("button_pressed_after") is False):
                events["hover_steps"].append(after)
        if row.get("button_pressed_before") is False and row.get("button_pressed_after") is True:
            if events["button_trigger_step"] is None:
                events["button_trigger_step"] = after
                events["button_trigger_bounds"] = [before + 1, after]
        stop = row.get("stop_timestep")
        if finite(stop) and events["stop_timestep"] is None:
            events["stop_timestep"] = stop
    if not finite(hover_z) or not definition.get("hover_source"):
        reasons.append("悬停参考缺少独立来源")
    if not has_command_tcp:
        reasons.append("缺少已执行动作的目标 TCP；需显式 --fk")
    for field, reason in (("command_press_step", "未观察到实际执行下压命令"),
                          ("button_trigger_step", "未观察到按钮由未按到按下")):
        if events[field] is None:
            reasons.append(reason)
    command, trigger = events["command_press_step"], events["button_trigger_step"]
    pre_hover = [t for t in events["hover_steps"] if command is not None and t < command]
    delay = trigger - command if command is not None and trigger is not None and trigger > command else None
    if not pre_hover:
        reasons.append("首次下压命令前没有可追溯的安全悬停姿态")
    if delay is None:
        reasons.append("命令到按钮的正耗时尚未建立")
    if events["button_trigger_bounds"] is not None and len(set(events["button_trigger_bounds"])) > 1:
        reasons.append("按钮变化发生在多物理步调用中，触发精确步尚未观测")
    sample = None
    if delay is not None and pre_hover:
        first = next(r for r in trace["rows"] if r["env_step_before"] == command)
        if not vector(first.get("state_before"), 8) or not vector(first.get("executed_action"), 8):
            reasons.append("命令起始姿态或实际动作不是有限 8 维向量")
        sample = {"motion_delay": delay, "command_press_step": command, "button_trigger_step": trigger,
                  "initial_state": first.get("state_before"), "command_action": first.get("command_action"),
                  "executed_action": first.get("executed_action"), "task_success": trace["meta"].get("task_success"),
                  "kinematics_source": trace["meta"].get("fk_source", trace["source"]),
                  "source": trace["source"], "scope": "same_episode_measured"}
    return {"events": events, "measured_sample": sample, "reasons": reasons,
            "trajectory_kind": trace["meta"]["trajectory_kind"]}


def validate_success_window(trace, events, geometry, definition):
    """成功字段独立于运动耗时；用真实成功轨迹核对按钮、停步与几何窗口。"""
    result = {"trajectory_kind": trace["meta"]["trajectory_kind"],
              "task_success": trace["meta"].get("task_success"), "status": "not_applicable", "reasons": []}
    if not is_task_success(result["task_success"]):
        return result
    reasons = result["reasons"]
    begin, end = geometry["target_visit_in"], geometry["target_visit_out"]
    trigger, stop = events["button_trigger_step"], events["stop_timestep"]
    if begin is None or end is None:
        reasons.append("成功轨迹缺少同局不按按钮的完整几何窗口")
    bounds = events["button_trigger_bounds"]
    if trigger is None or bounds is None or bounds[0] != bounds[1]:
        reasons.append("成功轨迹没有精确按钮触发步")
    if not finite(stop):
        reasons.append("成功轨迹缺少官方 stop_timestep")
    if begin is not None and end is not None:
        if trigger is not None and not begin <= trigger <= end:
            reasons.append("按钮触发步不在目标经过几何区间")
        if finite(stop) and not begin <= stop <= end:
            reasons.append("官方 stop_timestep 不在目标经过几何区间")
    if finite(stop) and trigger is not None and stop < trigger:
        reasons.append("官方停步早于首次按钮触发")
    stopped = next((row for row in trace["rows"] if row["env_step_after"] == stop), None)
    cube = stopped.get("cube_position_after") if stopped else None
    target = stopped.get("target_position_after") if stopped else None
    half = definition.get("cube_half_size")
    if not (vector(cube, 3) and vector(target, 3) and finite(half) and half > 0):
        reasons.append("官方停步缺少可核对的物体和目标实测位置")
    elif math.hypot(cube[0] - target[0], cube[1] - target[1]) > 3 * half:
        reasons.append("官方停步时物体超出 3*cube_half_size")
    result.update(status="fail" if reasons else "pass", button_trigger_step=trigger, stop_timestep=stop,
                  source=trace["source"])
    return result


def analyze(traces, definitions=None, delta_z=0.02):
    if not finite(delta_z) or delta_z <= 0:
        raise ValueError("delta_z 必须为正有限数")
    grouped = {}
    for trace in traces:
        grouped.setdefault(trace["meta"]["episode_key"], []).append(trace)
    windows, calibrations, groups = {}, {}, {}
    for key, items in grouped.items():
        primary = next((t for t in items if t["meta"]["trajectory_kind"] == "normal"), items[0])
        definition = dict(primary["meta"])
        definition.update((definitions or {}).get(key, {}))
        for trace in items:
            for field in ("episode_id", "env_seed", "difficulty", "move_interval", "target_pass"):
                if field in trace["meta"] and field in definition and trace["meta"][field] != definition[field]:
                    raise ValueError(f"{key} 的 {field} 在轨迹与任务定义间不一致")
        geometry = geometry_window(items, definition)
        trajectories = {}
        for trace in items:
            kind = trace["meta"]["trajectory_kind"]
            if kind in trajectories:
                raise ValueError(f"{key}/{kind} 重复轨迹，请明确选择一次运行")
            trajectories[kind] = extract_events(trace, definition, delta_z)
        normal = trajectories.get("normal", {"events": {}, "measured_sample": None, "reasons": ["缺少正常轨迹"]})
        sample = normal["measured_sample"]
        reasons = list(normal["reasons"])
        # 同局实际运动耗时独立于任务成败，不外推到其他姿态或恢复后的新动作。
        accepted = sample is not None and not reasons
        delay = sample["motion_delay"] if accepted else None
        calibration = {"episode_key": key, "env_seed": definition.get("env_seed"),
            "episode_id": definition.get("episode_id"), "difficulty": definition.get("difficulty"),
            "metadata_sources": primary.get("metadata_sources", []),
            "move_interval": definition.get("move_interval"), "target_pass": definition.get("target_pass"),
            "hover_z": definition.get("hover_z"), "delta_z": delta_z, "motion_delay": delay,
            "hover_source": definition.get("hover_source"), "calibration_source": "same_episode_measured" if accepted else None,
            "calibration_evidence": sample if accepted else None,
            "applicability": "仅该局实测起始姿态和动作；不保证恢复后新动作的延迟相同",
            "status": "resolved" if accepted else "unresolved", "reasons": reasons,
            "events": normal["events"], "samples": [sample] if sample else [], "trajectories": trajectories}
        begin, end = geometry["target_visit_in"], geometry["target_visit_out"]
        start = [begin - delay, end - delay] if delay is not None and begin is not None else [None, None]
        shared = {field: calibration[field] for field in ("episode_key", "episode_id", "env_seed", "difficulty", "move_interval", "target_pass",
                  "hover_z", "delta_z", "motion_delay", "calibration_source")}
        window = {**shared, **geometry, "start_range": start,
                  "status": "resolved" if accepted and geometry["window_status"] == "resolved" else "unresolved",
                  "reasons": reasons + geometry["window_reasons"]}
        calibration.update(target_visit_in=begin, target_visit_out=end, start_range=start)
        validation = [validate_success_window(trace, trajectories[trace["meta"]["trajectory_kind"]]["events"],
                                              geometry, definition) for trace in items]
        window["success_validation"] = validation
        group = groups.setdefault(str(definition.get("move_interval")), {"episodes": 0, "successful": 0, "validated": 0, "failed": 0})
        group["episodes"] += 1
        group["successful"] += sum(is_task_success(v["task_success"]) for v in validation)
        group["validated"] += sum(v["status"] == "pass" for v in validation)
        group["failed"] += sum(v["status"] == "fail" for v in validation)
        windows[key], calibrations[key] = window, calibration
    missing_groups = sorted(group for group, count in groups.items() if not count["validated"])
    valid = bool(groups) and not missing_groups and not any(g["failed"] for g in groups.values())
    verification = {"status": "pass" if valid else "fail", "groups": groups,
                    "missing_success_groups": missing_groups,
                    "successful": sum(g["successful"] for g in groups.values()),
                    "validated": sum(g["validated"] for g in groups.values())}
    return ({"schema_version": SCHEMA_VERSION, "episodes": windows, "window_sim": verification},
            {"schema_version": SCHEMA_VERSION, "episodes": calibrations})


def add_fk(trace, benchmark_root):
    """使用独立环境做目标 TCP 换算，绝不改运行器所持有的环境。"""
    try:
        from .read_sim_eef_pose import build_env, TcpReader
    except ImportError:
        from read_sim_eef_pose import build_env, TcpReader
    meta = trace["meta"]
    episode_id = meta.get("episode_id")
    if episode_id is None:
        raise ValueError(f"{meta['episode_key']} 缺 episode_id，不能构造同局 FK 环境")
    env, seed, difficulty = build_env("StopCube", int(episode_id))
    try:
        if meta.get("env_seed") != seed:
            raise ValueError("FK 环境 seed 与记录不符")
        if meta.get("difficulty") != difficulty:
            raise ValueError("FK 环境 difficulty 与记录不符")
        env.reset(seed=seed)
        definition = task_definition(env, benchmark_root)
        definition["env_seed"] = seed
        reader = TcpReader(env)
        for row in trace["rows"]:
            action = row.get("executed_action")
            if not vector(action, 8):
                raise ValueError("FK 需要每步实际执行的 8 维动作")
            row["executed_target_tcp"] = reader.set_arm(action[:7]).tolist()
        definition["fk_source"] = {"symbol": "TcpReader.set_arm", "trace": trace["source"],
            "reader": fingerprint(Path(__file__).with_name("read_sim_eef_pose.py")), "benchmark_commit": BENCHMARK_PIN}
        trace["meta"]["fk_source"] = definition["fk_source"]
        return definition
    finally:
        env.close()


def probe_episode(episode_id, episode_key, output_dir, benchmark_root):
    """独立同 seed 仿真保持初始关节位置，只采几何，不把它算作策略成功局。"""
    try:
        from .read_sim_eef_pose import build_env
    except ImportError:
        from read_sim_eef_pose import build_env
    import numpy as np
    from robomme.robomme_env.utils.subgoal_evaluate_func import is_button_pressed
    env, seed, difficulty = build_env("StopCube", episode_id)
    try:
        env.reset(seed=seed)
        definition = task_definition(env, benchmark_root)
        definition.update(episode_key=episode_key, episode_id=episode_id, env_seed=seed,
                          difficulty=difficulty, trajectory_kind="window_probe")
        u = env.unwrapped
        q = _plain(u.agent.robot.get_qpos())
        action = np.asarray(q[:7] + [q[7]], dtype=np.float32)
        rows = [{"event": "meta", **definition}]
        for _ in range(definition["move_interval"] * definition["target_pass"]):
            before = int(_plain(u.elapsed_steps)[0]) if hasattr(u.elapsed_steps, "reshape") else int(u.elapsed_steps)
            if before >= definition["move_interval"] * definition["target_pass"]:
                break
            cube, target = _plain(u.cube.pose.p), _plain(u.target.pose.p)
            pressed_before = bool(is_button_pressed(u, obj=u.button))
            _, _, terminated, truncated, _ = env.step(action)
            after = int(_plain(u.elapsed_steps)[0]) if hasattr(u.elapsed_steps, "reshape") else int(u.elapsed_steps)
            pressed_after = bool(is_button_pressed(u, obj=u.button))
            if pressed_before or pressed_after or u.stop:
                raise ValueError("窗口探针意外按下按钮，不能继续作为不按按钮几何证据")
            rows.append({"event": "step", "episode_key": episode_key, "trajectory_kind": "window_probe",
                "env_step_before": before, "env_step_after": after,
                "cube_position_before": cube, "cube_position_after": _plain(u.cube.pose.p),
                "target_position_before": target, "target_position_after": _plain(u.target.pose.p),
                "button_pressed_before": pressed_before, "button_pressed_after": pressed_after})
            if bool(terminated) or bool(truncated):
                break
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "window_probe.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        definitions = {"schema_version": 1, "episodes": {episode_key: definition}}
        (output_dir / "task_definitions.json").write_text(json.dumps(definitions, ensure_ascii=False, indent=2) + "\n")
        result = geometry_window([read_trace(path)], definition)
        ok = result["window_status"] == "resolved"
        print(f"WINDOW_GEOMETRY={'PASS' if ok else 'FAIL'} episode={episode_key} source={path}")
        return 0 if ok else 1
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", default=[], help="文件或递归包含 step_trace.jsonl 的目录，可重复")
    parser.add_argument("--task-definitions", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delta-z", type=float, default=0.02)
    parser.add_argument("--fk", action="store_true", help="显式建环境，换算已执行动作的 TCP 目标")
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--probe-episode", type=int)
    parser.add_argument("--episode-key")
    args = parser.parse_args(argv)
    if (args.fk or args.probe_episode is not None) and args.benchmark_root is None:
        parser.error("仿真操作需显式 --benchmark-root")
    if args.probe_episode is not None:
        if not args.episode_key:
            parser.error("--probe-episode 需 --episode-key")
        return probe_episode(args.probe_episode, args.episode_key, args.output_dir, args.benchmark_root)
    definitions = {}
    for path in args.task_definitions:
        definitions.update(json.loads(path.read_text())["episodes"])
    paths = []
    for path in args.trace:
        paths.extend(sorted(path.rglob("step_trace.jsonl")) if path.is_dir() else [path])
    traces = [read_trace(path) for path in dict.fromkeys(paths)]
    if not traces:
        parser.error("至少提供一条真实轨迹")
    if args.fk:
        for trace in traces:
            if trace["meta"]["trajectory_kind"] == "window_probe":
                continue
            definitions.setdefault(trace["meta"]["episode_key"], {}).update(add_fk(trace, args.benchmark_root))
    windows, calibration = analyze(traces, definitions, args.delta_z)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("windows.json", windows), ("timing_calibration.json", calibration)):
        (args.output_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    unresolved = sum(e["status"] != "resolved" for e in calibration["episodes"].values())
    missing_windows = sum(e["status"] != "resolved" for e in windows["episodes"].values())
    missing_geometry = sum(e["window_status"] != "resolved" for e in windows["episodes"].values())
    verification = windows["window_sim"]
    simulation_ok = verification["status"] == "pass" and not missing_windows
    print(f"TIMING_CALIBRATION={'PASS' if not unresolved else 'FAIL'} sources={len(traces)} unresolved={unresolved}")
    print(f"WINDOW_GEOMETRY={'PASS' if not missing_geometry else 'FAIL'} episodes={len(windows['episodes'])} unresolved={missing_geometry}")
    print(f"WINDOW_SIM={'PASS' if simulation_ok else 'FAIL'} successful={verification['successful']} "
          f"validated={verification['validated']} missing_success_groups={','.join(verification['missing_success_groups']) or 'none'}")
    return 0 if not unresolved and simulation_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
