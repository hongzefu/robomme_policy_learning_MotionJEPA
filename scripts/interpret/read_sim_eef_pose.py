"""在仿真里设关节角后读环境给出的 TCP 位姿（计划 1103-interpret-motion-attention-plan.html §1.6、K12）。

这不是解析式运动学换算：脚本建与评估同 seed／同难度的 RoboMME 环境，env.reset 后对每次推理 npz 里的
state[:7] 与 actions 每行 [:7] 关节目标，拼成 9 维 qpos（7 臂关节 + 2 指关节保持 reset 后
robot.get_qpos()[7:] 原值），逐个 agent.robot.set_qpos，再读 env.unwrapped.agent.tcp.pose.p 的 z
（与 benchmark MultiStepDemonstrationWrapper._current_tcp_p 同一路径）。不跑 step、不改环境、不改 eval.py。

必须用 official-sim 解释器运行（robomme 可导入），例如
  $E/v1-store/envs/official-sim/bin/python scripts/interpret/read_sim_eef_pose.py --attn <局目录或其上级>
每个局目录写 <局目录>/sim_eef_pose.json（infer_seq → step_idx、state_z、action_z[20]），末行 SIM_EEF_POSE=PASS|FAIL。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# 建环境参数照抄 examples/robomme/env_runner.py::EnvRunner.__init__ 的 base 分支（eval.py 所用）。
DATASET = "test"
MAX_STEPS = 1300


def build_env(task: str, episode_id: int):
    """同 EnvRunner.make_env：BenchmarkEnvBuilder 按元数据解出 seed／difficulty 并建环境。"""
    import robomme.robomme_env  # noqa: F401  注册任务
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(env_id=task, dataset=DATASET, action_space="joint_angle",
                                  gui_render=False, max_steps=MAX_STEPS)
    seed, difficulty = builder.resolve_episode(episode_id)
    env = builder.make_env_for_episode(episode_id)
    return env, seed, difficulty


class TcpReader:
    """持有一个已 reset 的环境；set_arm(q7) 后返回 TCP 位置。"""

    def __init__(self, env):
        self.env = env
        self.u = env.unwrapped
        self.robot = self.u.agent.robot
        self.q0 = self.robot.get_qpos().clone()  # (1, 9) torch
        self.gpu = bool(getattr(self.u.scene, "gpu_sim_enabled", False))

    def tcp_p(self) -> np.ndarray:
        p = self.u.agent.tcp.pose.p
        if hasattr(p, "cpu"):
            p = p.cpu().numpy()
        return np.asarray(p, dtype=np.float64).reshape(-1)

    def set_arm(self, q7) -> np.ndarray:
        import torch

        q = self.q0.clone()
        q[..., :7] = torch.as_tensor(np.asarray(q7, dtype=np.float32), dtype=q.dtype, device=q.device)
        self.robot.set_qpos(q)
        if self.gpu:
            # GPU 物理后端：把 qpos 推到 physx 并刷新连杆位姿后再读。
            self.u.scene._gpu_apply_all()
            self.u.scene.px.gpu_update_articulation_kinematics()
            self.u.scene._gpu_fetch_all()
        return self.tcp_p()


def _episode_dirs(root: Path) -> list[Path]:
    if any(root.glob("infer_*.npz")):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and any(d.glob("infer_*.npz")))


def read_episode(d: Path, args) -> tuple[dict, bool]:
    meta = json.loads((d / "episode.json").read_text()) if (d / "episode.json").is_file() else {}
    task = args.task or meta.get("task_name") or meta.get("task") or "ButtonUnmaskSwap"
    ep = args.episode if args.episode is not None else meta.get("episode_id")
    if ep is None:
        raise SystemExit(f"局 {d.name}：episode.json 无 episode_id，需 --episode")
    env, seed, difficulty = build_env(task, int(ep))
    if args.env_seed is not None and args.env_seed != seed:
        raise SystemExit(f"--env-seed {args.env_seed} 与元数据 seed {seed} 不符")
    if args.difficulty is not None and args.difficulty != difficulty:
        raise SystemExit(f"--difficulty {args.difficulty} 与元数据 difficulty {difficulty} 不符")
    print(f"SIM_ENV task={task} ep={ep} env_seed={seed} difficulty={difficulty}", flush=True)
    try:
        env.reset(seed=seed)
        reader = TcpReader(env)
        reset_z = float(reader.tcp_p()[2])
        q0 = reader.q0.cpu().numpy().reshape(-1)
        print(f"SIM_RESET qpos={np.round(q0, 4).tolist()} tcp_z={reset_z:.4f} gpu={int(reader.gpu)}", flush=True)
        infers = {}
        ok = True
        for p in sorted(d.glob("infer_*.npz")):
            with np.load(p) as z:
                state = np.asarray(z["state"], dtype=np.float64)
                actions = np.asarray(z["actions"], dtype=np.float64)
                seq = int(z["infer_seq"]) if "infer_seq" in z.files else int(p.stem.split("_")[1])
                step_idx = int(z["step_idx"]) if "step_idx" in z.files else -1
            state_z = float(reader.set_arm(state[:7])[2])
            action_z = [float(reader.set_arm(row[:7])[2]) for row in actions]
            ok = ok and all(math.isfinite(v) for v in [state_z, *action_z])
            infers[str(seq)] = {"step_idx": step_idx, "state_z": state_z, "action_z": action_z}
        rec = {"task_name": task, "episode_id": int(ep), "env_seed": int(seed), "difficulty": difficulty,
               "swap_times": int(getattr(env.unwrapped, "swap_times")) if hasattr(env.unwrapped, "swap_times") else None,
               "finger_qpos": q0[7:].tolist(), "reset_tcp_z": reset_z, "gpu_sim": reader.gpu, "infers": infers}
    finally:
        env.close()
    return rec, ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attn", type=Path, required=True, help="局目录，或其上级（每个局目录各写一份）")
    ap.add_argument("--out", type=Path, default=None, help="只在 --attn 是单个局目录时可指定，默认 <局目录>/sim_eef_pose.json")
    ap.add_argument("--task", default=None)
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--env-seed", type=int, default=None, help="核对用：须与元数据 seed 一致")
    ap.add_argument("--difficulty", default=None, help="核对用：须与元数据 difficulty 一致")
    args = ap.parse_args(argv)

    dirs = _episode_dirs(args.attn)
    if args.out is not None and len(dirs) != 1:
        raise SystemExit("--out 只能配单个局目录")
    total = 0
    ok_all = bool(dirs)
    for d in dirs:
        rec, ok = read_episode(d, args)
        out = args.out or (d / "sim_eef_pose.json")
        out.write_text(json.dumps(rec, indent=1))
        n = len(rec["infers"])
        ok = ok and n == len(list(d.glob("infer_*.npz")))
        print(f"写出 {out} infers={n} ok={int(ok)}", flush=True)
        total += n
        ok_all = ok_all and ok
    print(f"SIM_EEF_POSE={'PASS' if ok_all else 'FAIL'} infers={total}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
