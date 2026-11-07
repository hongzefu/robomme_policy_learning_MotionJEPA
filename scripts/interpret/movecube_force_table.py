#!/usr/bin/env python3
"""MoveCube 强推动作表：每局用环境自带规划器走到错误物体上方 z=0.2，夹爪张开，不下降、不抓。

方向 A（way=grasp_putdown）强推到 u.peg_tail 上方，方向 B（way=peg_push）强推到 u.cube 上方。
清单只应含这两类局；way=gripper_push 或非 MoveCube 的局算错误，判 FAIL。
不足 80 步补末行（原地悬停），超过 80 步截断，并如实记录 n_path、padded、truncated。
同进程开环执行一遍做校验：每 16 步记录 TCP 与物体位移，途中终局只记录，不作闸门。
规划失败不回退 RRT。只有 PASS 才写表，末行 FORCE_TABLE=PASS|FAIL episodes=<n> plan_fail=<n>。

用 official-sim 解释器运行，不加载模型：
  python scripts/interpret/movecube_force_table.py --manifest <清单> --out <动作表>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.interpret.stopcube_runner import position, scalar_value  # noqa: E402

TASK = "MoveCube"
FORCE_STEPS = 80
HOVER_Z = 0.2
GRIPPER_OPEN = 1.0
CHECK_EVERY = 16
# way -> (方向, 强推目标属性名)；gripper_push 全程不抓取，不入表。
WAYS = {"grasp_putdown": ("A", "peg_tail"), "peg_push": ("B", "cube")}


def pad_actions(path):
    """N×7 关节路径 → 80×8 动作：夹爪张开，不足补末行，超过截断。"""
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 7 or not len(path) or not np.isfinite(path).all():
        raise ValueError(f"规划路径须为非空有限 N×7，得到形状 {path.shape}")
    n_path = len(path)
    rows = path[:FORCE_STEPS]
    if n_path < FORCE_STEPS:
        rows = np.vstack([rows, np.repeat(rows[-1:], FORCE_STEPS - n_path, axis=0)])
    actions = np.hstack([rows, np.full((FORCE_STEPS, 1), GRIPPER_OPEN)])
    return actions, n_path, n_path < FORCE_STEPS, n_path > FORCE_STEPS


def plan_hover_path(env, target):
    """在当前场景规划到 target 正上方 z=0.2；成功返回 N×7 路径，失败返回 None。"""
    import torch
    from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix, matrix_to_quaternion, quaternion_multiply
    from robomme.robomme_env.utils.planner_fail_safe import FailAwarePandaArmMotionPlanningSolver as Solver

    u = env.unwrapped
    solver = Solver(env, debug=False, vis=False, base_pose=u.agent.robot.pose,
                    visualize_target_grasp_pose=False, print_env_info=False)
    flip = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([[np.pi, 0, 0]], dtype=torch.float32), convention="XYZ"))[0]
    quaternion = quaternion_multiply(target.pose.q.cpu()[0], flip)
    quaternion = (quaternion / torch.linalg.norm(quaternion)).numpy()
    point = target.pose.p.cpu().numpy()[0].copy()
    point[2] = HOVER_Z
    result = solver.planner.plan_screw(np.concatenate([point, quaternion]), u.agent.robot.get_qpos().cpu().numpy()[0],
                                       time_step=u.control_timestep, use_point_cloud=False)
    if not isinstance(result, dict) or result.get("status") != "Success":
        return None
    return result["position"]


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def open_loop_check(env, actions):
    """按表执行校验；途中终局只记录后停下，不作闸门。"""
    u = env.unwrapped
    cube0, peg0 = position(u.cube), position(u.peg_tail)
    samples, early_terminal = [], None
    for index, action in enumerate(actions, start=1):
        _, _, terminated, truncated, info = env.step(action)
        stop = bool(terminated or truncated)
        if index % CHECK_EVERY == 0 or stop:
            samples.append({"step": index, "tcp": position(u.agent.tcp),
                            "cube_displacement": distance(position(u.cube), cube0),
                            "peg_tail_displacement": distance(position(u.peg_tail), peg0)})
        if stop:
            early_terminal = {"step": index, "status": info.get("status")}
            break
    return {"samples": samples, "early_terminal": early_terminal}


def build_entry(env, env_seed, planner=plan_hover_path):
    """env 已 reset。规划失败返回 None；way 不合法由调用方先拦。"""
    u = env.unwrapped
    way = u.way
    direction, target_name = WAYS[way]
    target = getattr(u, target_name)
    identity = {"first_env_step": scalar_value(u.elapsed_steps), "env_seed": env_seed, "way": way, "direction": direction,
                "peg_tail_xyz": position(u.peg_tail), "cube_xyz": position(u.cube), "goal_xyz": position(u.goal_site),
                "force_target": target_name}
    path = planner(env, target)
    if path is None:
        return None
    actions, n_path, padded, truncated = pad_actions(path)
    return {**identity, "n_path": n_path, "padded": padded, "truncated": truncated,
            "actions": actions.tolist(), "open_loop": open_loop_check(env, actions)}


def build_table(episodes, builder, planner=plan_hover_path):
    """逐局生成；返回 (局表, plan_fail, way_errors)。"""
    table, plan_fail, way_errors = {}, 0, 0
    for episode in episodes:
        key = episode["episode_key"]
        if episode["task_name"] != TASK:
            way_errors += 1
            print(f"FORCE_TABLE_ERROR episode={key} reason=not_movecube", flush=True)
            continue
        env_seed, _ = builder.resolve_episode(episode["episode_id"])
        env = builder.make_env_for_episode(episode["episode_id"])
        try:
            env.reset()
            way = env.unwrapped.way
            if way not in WAYS:
                way_errors += 1
                print(f"FORCE_TABLE_ERROR episode={key} reason=way_{way}", flush=True)
                continue
            entry = build_entry(env, env_seed, planner)
        finally:
            env.close()
        if entry is None:
            plan_fail += 1
            print(f"FORCE_TABLE_PLAN_FAIL episode={key} way={way}", flush=True)
            continue
        table[key] = entry
        print(f"FORCE_TABLE_EPISODE episode={key} way={way} direction={entry['direction']} n_path={entry['n_path']} "
              f"padded={entry['padded']} truncated={entry['truncated']} early_terminal={entry['open_loop']['early_terminal']}",
              flush=True)
    return table, plan_fail, way_errors


def print_stats(table):
    for direction in sorted({entry["direction"] for entry in table.values()}):
        entries = [entry for entry in table.values() if entry["direction"] == direction]
        steps = [entry["n_path"] for entry in entries]
        print(f"FORCE_TABLE_STATS direction={direction} episodes={len(entries)} n_path_median={statistics.median(steps)} "
              f"n_path_min={min(steps)} n_path_max={max(steps)} truncated={sum(e['truncated'] for e in entries)} "
              f"early_terminal={sum(e['open_loop']['early_terminal'] is not None for e in entries)}", flush=True)


def make_builder():
    import robomme.robomme_env  # noqa: F401
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    return BenchmarkEnvBuilder(env_id=TASK, dataset="test", action_space="joint_angle", gui_render=False, max_steps=1300)


def main(argv=None, builder_factory=make_builder, planner=plan_hover_path):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    episodes = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["episodes"]
    table, plan_fail, way_errors = build_table(episodes, builder_factory(), planner)
    print_stats(table)
    ok = plan_fail == 0 and way_errors == 0 and len(table) == len(episodes) > 0
    if ok:
        out = Path(args.out)
        temporary = out.with_name(f".{out.name}.tmp")
        temporary.write_text(json.dumps({"schema_version": 1, "force_steps": FORCE_STEPS, "episodes": table},
                                        ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(out)
    print(f"FORCE_TABLE={'PASS' if ok else 'FAIL'} episodes={len(episodes)} plan_fail={plan_fail}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
