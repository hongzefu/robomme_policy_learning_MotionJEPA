"""
RoboMME environment runing wrapper: build envs, get observations, and step with a uniform API.

两个 runner 共用同一套观测/步进接口（``BaseEnvRunner``），只有「环境从哪来」不同：
* ``EnvRunner``——官方 ``env_metadata/<split>/`` 或自建 metadata 目录的 episode（seed 与 difficulty 由 metadata 给）；
* ``SpecEnvRunner``——（合入 v17）fork 注入流水线的 test/primary 候选（seed / difficulty / episode_spec /
  sampling_config 全部来自 ``candidates.jsonl``，这些候选不在 metadata 里，官方 builder 走不到）。
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Mapping
import numpy as np

try:
    from .utils import TASK_NAME_LIST
except ImportError:
    from utils import TASK_NAME_LIST

np.set_printoptions(precision=4, suppress=True)

#: 注入候选只覆盖这四个任务（fork 的 14 个「任务/难度」组都落在其中）
SPEC_TASKS = ("BinFill", "RouteStick", "VideoUnmaskSwap", "VideoRepick")


def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    # pack into 8-dim state, same as the joint action space
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)

class BaseEnvRunner:
    """观测与步进的公共实现；子类只负责把 ``self.env`` 建出来（合入 v17 的拆分）。"""

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


class EnvRunner(BaseEnvRunner):
    """
    Wraps RoboMME BenchmarkEnvBuilder for a single task: create env per episode,
    expose initial observation and step API.
    """

    def __init__(self, env_id: str, video_save_dir: str, max_steps: int = 1300, dataset: str = "test",
                 difficulty: str | None = None, metadata_dir: str | None = None) -> None:
        # 延迟加载仿真依赖，使纯 CPU 契约测试无需创建图形环境。
        import robomme.robomme_env  # noqa: F401
        from robomme.env_record_wrapper import BenchmarkEnvBuilder
        if env_id not in TASK_NAME_LIST:
            raise ValueError(f"Environment ID {env_id} not in {TASK_NAME_LIST}")
        # env_id 始终是原任务名：视频命名与 TASK_WITH_VIDEO_DEMO 判断都依赖它。
        super().__init__(env_id, video_save_dir)
        # benchmark split：train / val / test，决定 env_metadata/<split>/ 下读哪份 per-episode seed 与难度
        # （episode_config_resolver.py 的 _ALLOWED_DATASETS）；默认 test，行为与历史一致
        self.dataset = dataset
        # testx 口径传入清单难度；official 不传，走原路径。
        self.expected_difficulty = difficulty

        if metadata_dir is not None:
            # plfull 口径：照 XHARD_METADATA_DIR 的做法改读自建 metadata 目录，任务类与难度仍按清单原样。
            # 合入 v17 后 xhard 也走本路径（build_id=env_id，原任务类），不依赖 robomme_xhard / XHARD_METADATA_DIR；
            # 难度由自建 metadata 给出，make_env 仍逐集核对难度原样生效。
            if difficulty is None:
                raise ValueError("metadata_dir 需要同时给出清单难度 difficulty")
            if not Path(metadata_dir).is_dir():
                raise ValueError(f"metadata_dir 不是目录：{metadata_dir!r}")
            self.route = "plfull"
            self.build_id = env_id
            self.env_builder = BenchmarkEnvBuilder(
                env_id=env_id,
                dataset=dataset,
                action_space="joint_angle",
                gui_render=False,
                override_metadata_path=Path(metadata_dir),
                max_steps=max_steps,
            )
        elif difficulty is not None and difficulty.startswith("xhard"):
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


class SpecEnvRunner(BaseEnvRunner):
    """按 ``candidates.jsonl`` 的一条候选建环境。

    与 ``EnvRunner`` 的区别只在 ``make_env``：这里的 seed / difficulty / episode_spec 全部来自
    候选行本身，不查 ``env_metadata``——注入候选的 episode 号落在官方实跑区间之后（ep115+/153+/155+），
    难度还可能是 ``xhard``（用户决定不纳入 env_metadata 评测链），官方 builder 取不到它们。

    一个 runner 绑定一个「任务/难度」组，``sampling_config`` 是候选 header 里该任务那一份。
    """

    def __init__(self, task: str, difficulty: str, sampling_config: Mapping[str, Any],
                 video_save_dir: str, max_steps: int = 2000,
                 demo_prefix_store: str = "") -> None:
        if task not in SPEC_TASKS:
            raise ValueError(f"注入候选不覆盖任务 {task}，只有 {SPEC_TASKS}")
        super().__init__(task, video_save_dir)
        self.group_difficulty = difficulty
        self.sampling_config = sampling_config
        self.max_steps = max_steps
        # 守卫①：注入只对 BinFill 有定义——其余三任务的 reset() 本来就返回 planner 演示帧，
        # 再拼一段会双重叠加。692 条混合计划里非 BinFill 的 runner 也会收到 store 路径，
        # 这里**显式关掉并打一行日志**，不是静默忽略；eval.py 侧另有「BinFill 集必须全部命中 store」的硬校验，
        # check_shard.py 再核一遍「BinFill 集有 DEMO_PREFIX_INJECTED、非 BinFill 集没有」。
        if demo_prefix_store and task != "BinFill":
            print(f"DEMO_PREFIX_DISABLED task={task} difficulty={difficulty} reason=not-BinFill", flush=True)
            demo_prefix_store = ""
        if demo_prefix_store:
            # 延迟加载（cv2 / h5py），使纯 CPU 契约测试不必导入。
            try:
                from .demo_prefix import DemoPrefixStore
            except ImportError:
                from demo_prefix import DemoPrefixStore
            self._demo_store = DemoPrefixStore(demo_prefix_store)
        else:
            self._demo_store = None
        self._demo_key: tuple[str, str, int] | None = None
        self.demo_prefix_len = 0

    def make_env(self, candidate: Mapping[str, Any]) -> None:
        """``candidate`` 是候选行（已经过 load_candidates 校验）。"""
        if candidate["task"] != self.env_id or candidate["difficulty"] != self.group_difficulty:
            raise ValueError(
                f"候选 {candidate['task']}/{candidate['difficulty']} 不属于本 runner "
                f"{self.env_id}/{self.group_difficulty}"
            )
        # 延迟加载仿真依赖，与 EnvRunner 同口径。
        import robomme.robomme_env  # noqa: F401
        from robomme.env_record_wrapper import make_env_for_spec
        self.env = make_env_for_spec(
            self.env_id,
            candidate["seed"],
            candidate["difficulty"],
            candidate["spec"],
            self.sampling_config,
            max_steps=self.max_steps,
        )
        self.episode_id = candidate["episode"]
        self._demo_key = (candidate["task"], candidate["difficulty"], int(candidate["episode"]))
        self.demo_prefix_len = 0
        self.difficulty = self.env.unwrapped.difficulty
        # 难度必须原样传到环境里：xhard 走错分支会静默退化成 easy（RouteStick 的兜底分支有这个老 bug）
        if self.difficulty != self.group_difficulty:
            raise ValueError(
                f"难度未按规格生效：期望 {self.group_difficulty}，实际 {self.difficulty}"
            )

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
