"""
RoboMME environment runing wrapper: build envs, get observations, and step with a uniform API.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import numpy as np

try:
    from .utils import TASK_NAME_LIST
except ImportError:
    from utils import TASK_NAME_LIST

np.set_printoptions(precision=4, suppress=True)


def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    # pack into 8-dim state, same as the joint action space
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)

class EnvRunner:
    """
    Wraps RoboMME BenchmarkEnvBuilder for a single task: create env per episode,
    expose initial observation and step API.
    """

    def __init__(self, env_id: str, video_save_dir: str, max_steps: int = 1300, dataset: str = "test",
                 difficulty: str | None = None) -> None:
        # 延迟加载仿真依赖，使纯 CPU 契约测试无需创建图形环境。
        import robomme.robomme_env  # noqa: F401
        from robomme.env_record_wrapper import BenchmarkEnvBuilder
        if env_id not in TASK_NAME_LIST:
            raise ValueError(f"Environment ID {env_id} not in {TASK_NAME_LIST}")
        # env_id 始终是原任务名：视频命名与 TASK_WITH_VIDEO_DEMO 判断都依赖它。
        self.env_id = env_id
        self.video_save_dir = video_save_dir
        # benchmark split：train / val / test，决定 env_metadata/<split>/ 下读哪份 per-episode seed 与难度
        # （episode_config_resolver.py 的 _ALLOWED_DATASETS）；默认 test，行为与历史一致
        self.dataset = dataset
        # testx 口径传入清单难度；official 不传，走原路径。
        self.expected_difficulty = difficulty

        if difficulty is not None and difficulty.startswith("xhard"):
            # 注册 <Task>XHard 子类；必须在 import robomme.robomme_env 之后。
            import robomme_xhard  # noqa: F401
            xhard_dir = os.environ.get("XHARD_METADATA_DIR", "")
            if not xhard_dir or not Path(xhard_dir).is_dir():
                raise ValueError(f"xhard 局 {env_id}/{difficulty} 需要 XHARD_METADATA_DIR 指向 XHard metadata 目录，"
                                 f"得到 {xhard_dir!r}")
            self.route = "xhard"
            self.build_id = f"{env_id}XHard"
            self.env_builder = BenchmarkEnvBuilder(
                env_id=self.build_id,
                dataset=dataset,
                action_space="joint_angle",
                gui_render=False,
                override_metadata_path=Path(xhard_dir),
                max_steps=max_steps,
            )
        else:
            self.route = "base"
            self.build_id = env_id
            self.env_builder = BenchmarkEnvBuilder(
                env_id=env_id,
                dataset=dataset,
                action_space="joint_angle",
                gui_render=False,
                max_steps=max_steps,
            )

        # Set after make_env()
        self.env: Any = None
        self.episode_id: int | None = None
        self.task_goal: str = ""

    @property
    def num_episodes(self) -> int:
        return self.env_builder.get_episode_num()

    def make_env(self, episode_id: int) -> None:
        """Build and set the active env for the given episode."""
        # 逐集自证跑的是哪个 split：seed / difficulty 取自 resolve_episode（纯查表、无副作用），
        # 与 make_env_for_episode 内部写进 gym.make(env_kwargs) 的是同一组值。flush 必须有——
        # 驱动日志经 tmux + tee -a，无 flush 会卡缓冲（AGENTS 7）。
        _seed, _diff = self.env_builder.resolve_episode(episode_id)
        self.env_seed = _seed
        print(f"EVAL_EPISODE split={self.dataset} task={self.env_id} ep={episode_id} "
              f"env_seed={_seed} difficulty={_diff}", flush=True)
        self.env = self.env_builder.make_env_for_episode(episode_id)
        self.episode_id = episode_id
        self.difficulty = self.env.unwrapped.difficulty
        if self.expected_difficulty is not None:
            # 难度必须原样生效：xhard 走错分支会静默退化成别的档。
            if self.difficulty != self.expected_difficulty:
                raise ValueError(f"难度未按清单生效：期望 {self.expected_difficulty}，实际 {self.difficulty}")
            print(f"ENV_ROUTE route={self.route} env={self.build_id} task={self.env_id} episode={episode_id} "
                  f"difficulty={self.difficulty}", flush=True)

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
        # 原始异常由单局边界记录，不能用空观测覆盖首个异常栈。
        obs, _, terminated, truncated, self.info = self.env.step(action)

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
