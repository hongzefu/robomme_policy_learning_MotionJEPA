"""
RoboMME environment runing wrapper: build envs, get observations, and step with a uniform API.

两个 runner 共用同一套观测/步进接口，只有「环境从哪来」不同：
* ``EnvRunner``——官方 ``env_metadata/<split>/`` 的 episode（seed 与 difficulty 由 metadata 给）；
* ``SpecEnvRunner``——1200 局清单（1017-test-xhard5-12per）的一个「任务/难度」组，按难度分流建环境：
  xhard 行建 ``<Task>XHard`` 并读 ``XHARD_METADATA_DIR`` 里的 XHard metadata 副本，
  easy / medium / hard 行（含 MoveCube / InsertPeg）建原任务、读官方 test metadata 并断言 episode < 50。
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Mapping
import numpy as np

from robomme.robomme_env import *  # noqa: F401, F403 - env registration
from robomme.env_record_wrapper import BenchmarkEnvBuilder

from utils import TASK_NAME_LIST
from demo_prefix import DemoPrefixStore

np.set_printoptions(precision=4, suppress=True)

#: 官方 test metadata 里 base 档（easy / medium / hard）的 episode 上界；本次追加的 xhard 从 ep50 起
BASE_EPISODE_LIMIT = 50


def is_xhard(difficulty: str) -> bool:
    return str(difficulty).startswith("xhard")


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
    """按 1200 局清单的一行建环境；一个 runner 绑定一个「任务/难度」组。

    分流（计划 4.3 第 4 项）：
    * xhard 行：``BenchmarkEnvBuilder(f"{task}XHard", override_metadata_path=XHARD_METADATA_DIR)``——
      追加后的 test metadata 里 ep≥50 的难度是 ``xhard14`` 这类值，只有 ``<Task>XHard`` 子类接受；
      XHard 副本由 ``make_manifest.py export`` 从子模块 test metadata 导出（记录的 task 改成 ``<Task>XHard``）。
    * base 行（含 MoveCube / InsertPeg）：原任务名 + 官方 test metadata，断言 episode < 50。
    建成后断言 metadata 给出的 seed 与清单行相同、``env.unwrapped.difficulty`` 与清单难度相同，打一行 ``ENV_ROUTE``。
    ``env_id`` 始终是原任务名（视频名、``TASK_WITH_VIDEO_DEMO`` 判断都按原任务名）。
    """

    def __init__(self, task: str, difficulty: str, video_save_dir: str, max_steps: int = 2000,
                 demo_prefix_store: str = "") -> None:
        if task not in TASK_NAME_LIST:
            raise ValueError(f"未知任务 {task}，只有 {TASK_NAME_LIST}")
        super().__init__(task, video_save_dir)
        self.group_difficulty = difficulty
        self.max_steps = max_steps
        if is_xhard(difficulty):
            import robomme_xhard  # noqa: F401 - 注册 <Task>XHard 子类；必须在 import robomme.robomme_env 之后
            xhard_dir = os.environ.get("XHARD_METADATA_DIR", "")
            if not xhard_dir or not Path(xhard_dir).is_dir():
                raise ValueError(f"xhard 组 {task}/{difficulty} 需要 XHARD_METADATA_DIR 指向 XHard metadata 副本，得到 {xhard_dir!r}")
            self.route = "xhard"
            self.build_id = f"{task}XHard"
            self.env_builder = BenchmarkEnvBuilder(
                env_id=self.build_id, dataset="test", action_space="joint_angle", gui_render=False,
                override_metadata_path=Path(xhard_dir), max_steps=max_steps)
        else:
            self.route = "base"
            self.build_id = task
            self.env_builder = BenchmarkEnvBuilder(
                env_id=task, dataset="test", action_space="joint_angle", gui_render=False,
                max_steps=max_steps)
        # 守卫①：注入只对 BinFill 有定义——其余任务的 reset() 本来就返回 planner 演示帧，
        # 再拼一段会双重叠加。本轮 EXPECT_DEMO_INJECT=0、不传 store；保留开关只为与 0918 客户端同构。
        if demo_prefix_store and task != "BinFill":
            print(f"DEMO_PREFIX_DISABLED task={task} difficulty={difficulty} reason=not-BinFill", flush=True)
            demo_prefix_store = ""
        self._demo_store = DemoPrefixStore(demo_prefix_store) if demo_prefix_store else None
        self._demo_key: tuple[str, str, int] | None = None
        self.demo_prefix_len = 0

    def make_env(self, candidate: Mapping[str, Any]) -> None:
        """``candidate`` 是清单行 ``{"task","episode","seed","difficulty"}``。"""
        if candidate["task"] != self.env_id or candidate["difficulty"] != self.group_difficulty:
            raise ValueError(
                f"清单行 {candidate['task']}/{candidate['difficulty']} 不属于本 runner "
                f"{self.env_id}/{self.group_difficulty}"
            )
        episode = int(candidate["episode"])
        if self.route == "base" and episode >= BASE_EPISODE_LIMIT:
            raise ValueError(f"base 档 {self.env_id}/{self.group_difficulty} 的 episode {episode} 不在官方 ep0–49 内")
        seed, difficulty_hint = self.env_builder.resolve_episode(episode)
        if seed != int(candidate["seed"]) or difficulty_hint != candidate["difficulty"]:
            raise ValueError(
                f"metadata 与清单不符：{self.build_id} ep{episode} metadata=({seed}, {difficulty_hint}) "
                f"清单=({candidate['seed']}, {candidate['difficulty']})")
        self.env = self.env_builder.make_env_for_episode(episode)
        self.episode_id = episode
        self._demo_key = (candidate["task"], candidate["difficulty"], episode)
        self.demo_prefix_len = 0
        self.difficulty = self.env.unwrapped.difficulty
        # 难度必须原样传到环境里：xhard 走错分支会静默退化成别的档
        if self.difficulty != self.group_difficulty:
            raise ValueError(
                f"难度未按清单生效：期望 {self.group_difficulty}，实际 {self.difficulty}"
            )
        print(f"ENV_ROUTE route={self.route} env={self.build_id} task={self.env_id} episode={episode} "
              f"seed={seed} difficulty={self.difficulty}", flush=True)

    def get_init_obs(self) -> dict[str, Any]:
        """在干净 env 的 reset 观测前面拼上离线生成的 demo 前缀（0922-binfill-demo-prefix-plan.md B 节）。

        注入点选在这里而不是 ``eval.py::init_episode``：``eval.py`` 下游的三个 buffer、recorder 循环、
        ``exec_start_idx = len(image_buffer) - 1`` 与 ``is_video_demo`` 五处全部自动正确，逻辑一行不用改。

        off-by-one：注入后 ``image_buffer = [demo 0..D-1] + [干净 env 的 reset 帧]``，长度 D+1，
        故 ``exec_start_idx = D``。这与训练侧「demo 段 [0, es)、exec 起点 es」一致，
        也与原生 demo 任务同构（VideoRepick 的 reset 返回 [D 帧 planner] + [1 帧复位步]，es = D）。
        """
        obs = super().get_init_obs()
        if self._demo_store is None:
            return obs                       # 关闭时与改动前逐字等价
        if self._demo_key is None:
            raise RuntimeError("尚未 make_env，取不到候选键")
        prefix = self._demo_store.get(*self._demo_key)
        if prefix["task_goal"] != obs["task_goal"]:
            raise ValueError(
                f"{self._demo_key} 的 demo 前缀 task_goal 与本次 reset 不符：\n"
                f"  前缀：{prefix['task_goal']!r}\n  本次：{obs['task_goal']!r}")
        d = int(prefix["D"])
        if not (len(prefix["images"]) == len(prefix["wrist_images"]) == len(prefix["states"]) == d):
            raise ValueError(f"{self._demo_key} 的前缀三个缓冲长度不一致")
        if len(obs["images"]) != 1:
            # BinFill 的 task_list 里 demonstration 三处全 False，干净 env 的 reset 只返回 1 帧复位观测。
            # 不是 1 就说明 benchmark 侧行为变了，拼接语义随之失效，必须停。
            raise ValueError(
                f"{self._demo_key} 的干净 env reset 返回 {len(obs['images'])} 帧（期望 1 帧）")
        for key in ("images", "wrist_images", "states"):
            if obs[key][0].shape != prefix[key][0].shape or obs[key][0].dtype != prefix[key][0].dtype:
                raise ValueError(
                    f"{self._demo_key} 的 {key} 前缀与在线形制不符："
                    f"{prefix[key][0].shape}/{prefix[key][0].dtype} vs {obs[key][0].shape}/{obs[key][0].dtype}")
            obs[key] = list(prefix[key]) + list(obs[key])
        self.demo_prefix_len = d
        print(f"DEMO_PREFIX_INJECTED key={'/'.join(map(str, self._demo_key))} D={d} "
              f"total_frames={len(obs['images'])} exec_start_idx={len(obs['images']) - 1}", flush=True)
        return obs
