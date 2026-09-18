"""
RoboMME environment runing wrapper: build envs, get observations, and step with a uniform API.

两个 runner 共用同一套观测/步进接口，只有「环境从哪来」不同：
* ``EnvRunner``——官方 ``env_metadata/<split>/`` 的 episode（seed 与 difficulty 由 metadata 给）；
* ``SpecEnvRunner``——fork 注入流水线的 test/primary 候选（seed / difficulty / episode_spec /
  sampling_config 全部来自 ``candidates.jsonl``，这些候选不在 metadata 里，官方 builder 走不到）。
"""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np

from robomme.robomme_env import *  # noqa: F401, F403 - env registration
from robomme.env_record_wrapper import BenchmarkEnvBuilder, make_env_for_spec

from utils import TASK_NAME_LIST

np.set_printoptions(precision=4, suppress=True)

#: 注入候选只覆盖这四个任务（fork 的 14 个「任务/难度」组都落在其中）
SPEC_TASKS = ("BinFill", "RouteStick", "VideoUnmaskSwap", "VideoRepick")


def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    # pack into 8-dim state, same as the joint action space
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)


class BaseEnvRunner:
    """观测与步进的公共实现；子类只负责把 ``self.env`` 建出来。"""

    def __init__(self, env_id: str, video_save_dir: str) -> None:
        self.env_id = env_id
        self.video_save_dir = video_save_dir

        # Set after make_env()
        self.env: Any = None
        self.episode_id: int | None = None
        self.task_goal: str = ""
        self.difficulty: str | None = None

    def get_init_obs(self) -> dict[str, Any]:
        """Reset env and return initial observation dict (images, wrist_images, states, task_goal)."""
        obs, self.info = self.env.reset()
        if isinstance(self.info["task_goal"], list):
            self.task_goal = self.info["task_goal"][0]
        else:
            self.task_goal = self.info["task_goal"]
        images = obs["front_rgb_list"]
        wrist_images = obs["wrist_rgb_list"]
        states = [pack_state(joint_state, gripper_state) for joint_state, gripper_state in 
                  zip(obs["joint_state_list"], obs["gripper_state_list"])]

        return {
            "images": images,
            "wrist_images": wrist_images,
            "states": states,
            "task_goal": self.task_goal,
        }

    def step(self, action: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], bool, str]:
        """
        Execute one step.
        Returns ( (img, wrist_img, state), stop_flag, success_flag ).
        success_flag is one of "success", "fail", "timeout", "unknown".
        """
        try:
            obs, _, terminated, truncated, self.info = self.env.step(action)
        except Exception as e:
            print(f"Error: {e}")
            return (None, None, None), True, "error"

        img = obs["front_rgb_list"][-1]
        wrist_img = obs["wrist_rgb_list"][-1]
        joint_state = obs["joint_state_list"][-1]
        gripper_state = obs["gripper_state_list"][-1]
        state = pack_state(joint_state, gripper_state)

        outcome = self.info.get("status", "unknown")
        stop = terminated or truncated
                
        return (img, wrist_img, state), stop, outcome

    def close_env(self) -> None:
        """Close and clear the current env."""
        if self.env is not None:
            self.env.close()
            del self.env
            self.env = None
            self.episode_id = None
            self.task_goal = None


class EnvRunner(BaseEnvRunner):
    """
    Wraps RoboMME BenchmarkEnvBuilder for a single task: create env per episode,
    expose initial observation and step API.
    """

    def __init__(self, env_id: str, video_save_dir: str, max_steps: int = 1300) -> None:
        if env_id not in TASK_NAME_LIST:
            raise ValueError(f"Environment ID {env_id} not in {TASK_NAME_LIST}")
        super().__init__(env_id, video_save_dir)

        self.env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="test",
            action_space="joint_angle",
            gui_render=False,
            max_steps=max_steps,
        )

    @property
    def num_episodes(self) -> int:
        return self.env_builder.get_episode_num()

    def make_env(self, episode_id: int) -> None:
        """Build and set the active env for the given episode."""
        self.env = self.env_builder.make_env_for_episode(episode_id)
        self.episode_id = episode_id
        self.difficulty = self.env.unwrapped.difficulty


class SpecEnvRunner(BaseEnvRunner):
    """按 ``candidates.jsonl`` 的一条候选建环境。

    与 ``EnvRunner`` 的区别只在 ``make_env``：这里的 seed / difficulty / episode_spec 全部来自
    候选行本身，不查 ``env_metadata``——注入候选的 episode 号落在官方实跑区间之后（ep115+/153+/155+），
    难度还可能是 ``xhard``（用户决定不纳入 env_metadata 评测链），官方 builder 取不到它们。

    一个 runner 绑定一个「任务/难度」组，``sampling_config`` 是候选 header 里该任务那一份。
    """

    def __init__(self, task: str, difficulty: str, sampling_config: Mapping[str, Any],
                 video_save_dir: str, max_steps: int = 2000) -> None:
        if task not in SPEC_TASKS:
            raise ValueError(f"注入候选不覆盖任务 {task}，只有 {SPEC_TASKS}")
        super().__init__(task, video_save_dir)
        self.group_difficulty = difficulty
        self.sampling_config = sampling_config
        self.max_steps = max_steps

    def make_env(self, candidate: Mapping[str, Any]) -> None:
        """``candidate`` 是候选行（已经过 load_candidates 校验）。"""
        if candidate["task"] != self.env_id or candidate["difficulty"] != self.group_difficulty:
            raise ValueError(
                f"候选 {candidate['task']}/{candidate['difficulty']} 不属于本 runner "
                f"{self.env_id}/{self.group_difficulty}"
            )
        self.env = make_env_for_spec(
            self.env_id,
            candidate["seed"],
            candidate["difficulty"],
            candidate["spec"],
            self.sampling_config,
            max_steps=self.max_steps,
        )
        self.episode_id = candidate["episode"]
        self.difficulty = self.env.unwrapped.difficulty
        # 难度必须原样传到环境里：xhard 走错分支会静默退化成 easy（RouteStick 的兜底分支有这个老 bug）
        if self.difficulty != self.group_difficulty:
            raise ValueError(
                f"难度未按规格生效：期望 {self.group_difficulty}，实际 {self.difficulty}"
            )
