"""两个 Swap 任务的逐步轨迹客户端；官方评估循环、RPC、动作与终局原样复用，只加只读记录。

结构对应 stopcube_runner 的 normal 路径，但只接受 ButtonUnmaskSwap、VideoUnmaskSwap，没有悬停与 Oracle。
每局写 traces/<nonce>/step_trace.jsonl（可被 rollout_offline.load_trace 消费）与 traces/<nonce>/meta/hold_receipt.json
（文件名沿用 stopcube_runner，便于同样按 nonce 定位）。

subgoal_index 读环境属性 timestep：它只由 sequential_task_check 写入，是 task_list 的当前下标；
ButtonUnmaskSwap 前两项是两个按钮，所以按完后不小于 2。每行记步后值。
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from examples.robomme import eval as official
from examples.robomme.env_runner import EnvRunner
from scripts.interpret.stopcube_runner import array_value, position, scalar_value

TASKS = ("ButtonUnmaskSwap", "VideoUnmaskSwap")
KIND = "normal"


def snapshot(env):
    """只读当前物理状态与子目标序号；不调用任何判定函数，也不写环境。"""
    u = env.unwrapped
    return {"env_step": scalar_value(getattr(u, "elapsed_steps", None)),
            "tcp": position(getattr(u.agent, "tcp", None)),
            "bins": [position(actor) for actor in u.spawned_bins],
            "subgoal_index": scalar_value(getattr(u, "timestep", None))}


def first_target_bin_index(u):
    """selected_bins[0] 在 spawned_bins 中的下标；两者是同一批 actor 对象，按对象身份比较。"""
    for index, actor in enumerate(u.spawned_bins):
        if actor is u.selected_bins[0]:
            return index
    raise official.ContractError("selected_bins[0] 不在 spawned_bins 中")


class TraceRunner(EnvRunner):
    """官方环境实例的记录包装：送入环境的动作与 EnvRunner 完全相同。"""

    def begin_trace(self, nonce, root):
        if self.env_id not in TASKS:
            raise official.ContractError(f"轨迹入口只接受 {TASKS}")
        self.key = f"{self.env_id}:{self.episode_id}"
        self.trace_dir = Path(root) / "traces" / nonce
        (self.trace_dir / "meta").mkdir(parents=True, exist_ok=False)
        self.trace_path = self.trace_dir / "step_trace.jsonl"
        self.trace_stream = self.trace_path.open("x", encoding="utf-8")
        self.receipt = {"schema_version": 1, "episode_key": self.key, "episode_nonce": nonce,
                        "task_name": self.env_id, "episode_id": self.episode_id, "env_seed": self.env_seed,
                        "difficulty": self.difficulty, "trajectory_kind": KIND,
                        "trace_path": str(self.trace_path.resolve()),
                        "terminal_path": str((Path(root) / f"{self.key}.terminal.json").resolve()),
                        "video_path": None, "outcome": None, "task_success": None,
                        "swap_times": None, "pick_times": None, "swap_end": None, "D": None,
                        "exec_start_idx": None, "initial_env_step": None, "color_names": None,
                        "prompt": None, "first_target_bin_index": None}
        self.action_index = self.row_in_chunk = 0
        self.infer_seq = self.chunk_start_env_step = None

    def get_init_obs(self):
        result = super().get_init_obs()
        self.last_state = np.asarray(result["states"][-1]).copy()
        u = self.env.unwrapped
        # 场景在 reset 内建好，这些量只能在 reset 之后读。
        self.receipt.update(D=len(result["images"]) - 1, exec_start_idx=len(result["images"]) - 1,
                            initial_env_step=snapshot(self.env)["env_step"], swap_times=int(u.swap_times),
                            pick_times=int(u.pick_times), swap_end=int(u.swap_schedule[-1][3]),
                            color_names=list(u.color_names), prompt=result["task_goal"],
                            first_target_bin_index=first_target_bin_index(u))
        return result

    def begin_chunk(self, infer_seq):
        self.infer_seq = infer_seq
        self.row_in_chunk = 0
        self.chunk_start_env_step = snapshot(self.env)["env_step"]

    def step(self, action):
        before = snapshot(self.env)
        state_before = self.last_state.copy()
        command = array_value(action)
        result = super().step(action)
        after = snapshot(self.env)
        self.last_state = np.asarray(result[0][2]).copy()
        record = {"event": "step", "episode_key": self.key, "infer_seq": self.infer_seq,
                  "row_in_chunk": self.row_in_chunk, "action_index": self.action_index,
                  "chunk_start_env_step": self.chunk_start_env_step,
                  "exec_start_idx": self.receipt["exec_start_idx"],
                  "command_action": command, "executed_action": command,
                  "state_before": array_value(state_before), "state_after": array_value(self.last_state),
                  "env_step_before": before["env_step"], "env_step_after": after["env_step"],
                  "tcp_before": before["tcp"], "tcp_after": after["tcp"],
                  "bins_before": before["bins"], "bins_after": after["bins"],
                  "subgoal_index": after["subgoal_index"]}
        self.trace_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.trace_stream.flush()
        self.row_in_chunk += 1
        self.action_index += 1
        return result

    def finish_trace(self, outcome, video_dir):
        self.trace_stream.close()
        video = Path(video_dir) / f"{self.env_id}_ep{self.episode_id}_{outcome}.mp4"
        self.receipt.update(outcome=outcome, task_success=int(outcome == "success") if outcome else None,
                            actions_executed=self.action_index,
                            video_path=str(video.resolve()) if outcome else None)
        official.write_record(self.trace_dir / "meta" / "hold_receipt.json", self.receipt)


class TraceEvaluator(official.EpisodeEvaluator):
    """只安装记录上下文；历史、动作队列与终局全部走官方 eval_each_episode。"""

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
        env_runner.begin_trace(episode_nonce, self.save_dir)
        outcome = None
        try:
            outcome = super().eval_each_episode(env_runner, video_save_dir, client, episode_nonce)
            return outcome
        finally:
            env_runner.finish_trace(outcome, video_save_dir)


def evaluate(args):
    """在独立客户端进程里复用官方主入口，结束后恢复模块内的评估器绑定。"""
    if any(episode["task_name"] not in TASKS for episode in official.load_chunk(args)):
        raise official.ContractError(f"轨迹入口只接受 {TASKS}")
    previous = official.EpisodeEvaluator
    official.EpisodeEvaluator = TraceEvaluator
    try:
        return official.evaluate(args, runner_factory=TraceRunner)
    finally:
        official.EpisodeEvaluator = previous


if __name__ == "__main__":
    import tyro
    raise SystemExit(evaluate(tyro.cli(official.Args)))
