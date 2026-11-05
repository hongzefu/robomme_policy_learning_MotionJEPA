"""StopCube 逐步记录与固定关节悬停；复用官方评估循环、RPC 和终局。

配置为 {"schema_version": 1, "episodes": {"StopCube:0": {
"takeover_step": 90, "release_step": 160, "hold_qpos": [8维关节/夹爪]}}}。
两个时刻均为官方 elapsed_steps 的步前值；恢复必须落在现场首个执行步加 16k。
normal 仅记录，hold_extended 仅替换接管区间内送入环境的动作。
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from examples.robomme import eval as official
from examples.robomme.env_runner import EnvRunner


@dataclasses.dataclass
class Args(official.Args):
    stopcube_mode: str = "normal"
    stopcube_config: str = ""


def load_config(path):
    """读取固定接管表；具体局的参数在初始化观测后核对。"""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not isinstance(config.get("episodes"), dict):
        raise official.ContractError("StopCube 配置须含 schema_version=1 与 episodes 对象")
    return config["episodes"]


def array_value(value):
    """把真实仿真张量转为 JSON；不可得字段明确为空。"""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1).tolist()


def scalar_value(value):
    values = array_value(value)
    return values[0] if values else None


def position(actor):
    return array_value(actor.pose.p) if actor is not None else None


def snapshot(env):
    """只读当前物理状态，不调用会写 stop_timestep 的任务判定函数。"""
    from robomme.robomme_env.utils.subgoal_evaluate_func import is_button_pressed

    u = env.unwrapped
    button = getattr(u, "button", None)
    return {
        "env_step": scalar_value(getattr(u, "elapsed_steps", None)),
        "tcp": position(getattr(u.agent, "tcp", None)),
        "button_pressed": bool(is_button_pressed(u, obj=button)) if button is not None else None,
        "cube_position": position(getattr(u, "cube", None)),
        "target_position": position(getattr(u, "target", None)),
        "stop_timestep": scalar_value(getattr(u, "stop_timestep", None)),
        "move_interval": scalar_value(getattr(u, "move_interval", None)),
        "target_pass": scalar_value(getattr(u, "stop_time", None)),
    }


class TraceRunner(EnvRunner):
    """官方环境实例的 interpret 包装，不覆盖任务类或物理状态。"""

    def begin_trace(self, args, nonce, root, config):
        if self.env_id != "StopCube":
            raise official.ContractError("StopCube 记录入口只接受 StopCube")
        self.mode = args.stopcube_mode
        self.key = f"{self.env_id}:{self.episode_id}"
        self.schedule = config.get(self.key) if self.mode == "hold_extended" else None
        if self.mode == "hold_extended" and self.schedule is None:
            raise official.ContractError(f"接管表缺局 {self.key}")
        self.trace_dir = Path(root) / "traces" / nonce
        (self.trace_dir / "meta").mkdir(parents=True, exist_ok=False)
        self.trace_path = self.trace_dir / "step_trace.jsonl"
        self.trace_stream = self.trace_path.open("x", encoding="utf-8")
        self.receipt = {"schema_version": 1, "episode_key": self.key, "episode_nonce": nonce,
                        "episode_id": self.episode_id, "env_seed": self.env_seed, "difficulty": self.difficulty,
                        "trajectory_kind": self.mode, "trace_path": str(self.trace_path.resolve()),
                        "terminal_path": str((Path(root) / f"{self.key}.terminal.json").resolve()),
                        "video_path": None, "outcome": None, "task_success": None,
                        "D": None, "exec_start_idx": None, "initial_env_step": None,
                        "takeover_step": None, "release_step": None, "hold_qpos": None,
                        "actual_takeover_step": None, "actual_release_step": None}
        self.action_index = self.row_in_chunk = 0
        self.infer_seq = self.chunk_start_env_step = None

    def get_init_obs(self):
        result = super().get_init_obs()
        self.last_state = np.asarray(result["states"][-1]).copy()
        first = snapshot(self.env)
        self.receipt.update(D=len(result["images"]) - 1, exec_start_idx=len(result["images"]) - 1,
                            initial_env_step=first["env_step"])
        u = self.env.unwrapped
        self.receipt["extra"] = {"cube_half_size": scalar_value(getattr(u, "cube_half_size", None)),
                                 "button_position": position(getattr(u, "button", None)),
                                 "hover_z": None, "hover_source": None}
        if self.schedule is not None:
            start, release = self.schedule["takeover_step"], self.schedule["release_step"]
            qpos = np.asarray(self.schedule["hold_qpos"], dtype=np.float64)
            origin = first["env_step"]
            if (type(start) is not int or type(release) is not int or origin is None or start < origin
                    or release <= start or (release - origin) % 16 != 0
                    or qpos.shape != (8,) or not np.isfinite(qpos).all()):
                raise official.ContractError("接管参数须为有限 8 维动作，恢复晚于接管且符合现场首步+16k")
            self.hold_action = qpos
            self.receipt.update(takeover_step=start, release_step=release, hold_qpos=qpos.tolist())
        return result

    def begin_chunk(self, infer_seq):
        self.infer_seq = infer_seq
        self.row_in_chunk = 0
        self.chunk_start_env_step = snapshot(self.env)["env_step"]

    def step(self, action):
        before = snapshot(self.env)
        state_before = self.last_state.copy()
        command = np.asarray(action).copy()
        executed = command
        elapsed = before["env_step"]
        if self.schedule is not None:
            if self.schedule["takeover_step"] <= elapsed < self.schedule["release_step"]:
                executed = self.hold_action.copy()
                if self.receipt["actual_takeover_step"] is None:
                    self.receipt["actual_takeover_step"] = elapsed
            elif elapsed == self.schedule["release_step"]:
                if self.row_in_chunk != 0 or self.chunk_start_env_step != elapsed:
                    raise official.ContractError("恢复步必须执行当前边界的新推理动作")
                self.receipt["actual_release_step"] = elapsed
        result = super().step(executed)
        after = snapshot(self.env)
        self.last_state = np.asarray(result[0][2]).copy()
        record = {"event": "step", "episode_key": self.key, "trajectory_kind": self.mode,
                  "infer_seq": self.infer_seq, "row_in_chunk": self.row_in_chunk,
                  "action_index": self.action_index, "chunk_start_env_step": self.chunk_start_env_step,
                  "exec_start_idx": self.receipt["exec_start_idx"],
                  "command_action": array_value(command), "executed_action": array_value(executed),
                  "state_before": array_value(state_before), "state_after": array_value(self.last_state),
                  "stop_timestep": after["stop_timestep"], "move_interval": after["move_interval"],
                  "target_pass": after["target_pass"]}
        for key in ("env_step", "tcp", "button_pressed", "cube_position", "target_position"):
            record[f"{key}_before"] = before[key]
            record[f"{key}_after"] = after[key]
        self.trace_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.trace_stream.flush()
        self.row_in_chunk += 1
        self.action_index += 1
        return result

    def finish_trace(self, outcome, video_dir):
        self.trace_stream.close()
        self.receipt.update(outcome=outcome, task_success=int(outcome == "success") if outcome else None,
                            actions_executed=self.action_index,
                            video_path=str((Path(video_dir) / f"StopCube_ep{self.episode_id}_{outcome}.mp4").resolve())
                            if outcome else None)
        official.write_record(self.trace_dir / "meta" / "hold_receipt.json", self.receipt)


class TraceEvaluator(official.EpisodeEvaluator):
    """仅给官方循环安装记录上下文；历史、动作队列与终局原样复用。"""

    config = None

    def init_episode(self, env_runner, epstate, video_save_dir):
        result = super().init_episode(env_runner, epstate, video_save_dir)
        env_runner.receipt.update(video_fps=getattr(self.recorder, "fps", None),
                                  video_start_env_step=env_runner.receipt["initial_env_step"],
                                  video_exec0_frame_index=epstate.exec_start_idx)
        return result

    def get_action_chunk(self, *args, **kwargs):
        actions = super().get_action_chunk(*args, **kwargs)
        self.trace_runner.begin_chunk(self.replan_seq - 1)
        return actions

    def eval_each_episode(self, env_runner, video_save_dir, client, episode_nonce):
        self.trace_runner = env_runner
        env_runner.begin_trace(self.args, episode_nonce, self.save_dir, self.config)
        outcome = None
        try:
            outcome = super().eval_each_episode(env_runner, video_save_dir, client, episode_nonce)
            return outcome
        finally:
            env_runner.finish_trace(outcome, video_save_dir)


def evaluate(args):
    """在独立客户端进程里复用官方主入口，结束后恢复模块内的评估器绑定。"""
    if args.stopcube_mode not in ("normal", "hold_extended") or not args.stopcube_config:
        raise official.ContractError("必须提供 StopCube 模式与配置文件")
    episodes = official.load_chunk(args)
    if any(episode["task_name"] != "StopCube" for episode in episodes):
        raise official.ContractError("StopCube 入口不能执行其他任务")
    TraceEvaluator.config = load_config(args.stopcube_config)
    previous = official.EpisodeEvaluator
    official.EpisodeEvaluator = TraceEvaluator
    try:
        return official.evaluate(args, runner_factory=TraceRunner)
    finally:
        official.EpisodeEvaluator = previous


if __name__ == "__main__":
    import tyro
    raise SystemExit(evaluate(tyro.cli(Args)))
