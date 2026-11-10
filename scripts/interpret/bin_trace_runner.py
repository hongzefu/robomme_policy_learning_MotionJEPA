"""三个交换任务的逐步轨迹客户端；官方评估循环、RPC、动作与终局原样复用，只加只读记录。

结构对应 stopcube_runner 的 normal 路径，只接受 ButtonUnmaskSwap、VideoUnmaskSwap、VideoRepick，没有悬停与 Oracle。
每局写 traces/<nonce>/step_trace.jsonl（可被 rollout_offline.load_trace 消费）与 traces/<nonce>/meta/hold_receipt.json
（文件名沿用 stopcube_runner，便于同样按 nonce 定位）。

subgoal_index 读环境属性 timestep：它只由 sequential_task_check 写入，是 task_list 的当前下标；
ButtonUnmaskSwap 前两项是两个按钮，所以按完后不小于 2。每行记步后值。

物体列表与目标按任务取：两个 Swap 任务是 spawned_bins 与 selected_bins[0]，VideoRepick 是 spawned_cubes 与
target_cube_1；行字段 bins_before／bins_after 的物体顺序即该列表顺序（VideoRepick 里装的是方块）。

扩展输出（--stop-after-first-pick 打开，或任务是 VideoRepick 时）：每行加 objects（步后全部物体位置），receipt 加
target_index、swap_schedule（[[a,b,start,end],...]，a/b 为物体下标）、first_pick_subgoal_index、stop_after_first_pick，
VideoRepick 另加 num_repeats。两个 Swap 任务在开关关闭时，行与 receipt 与改动前逐字节一致。

拿起即停（--stop-after-first-pick）：只看执行段每步步后状态（示范在 reset 内，不经过 step），容器 z>0.15
且 agent.is_grasping（ButtonUnmaskSwap 开局环境自己抬起全部容器，只看高度会误判），方块 z>0.05 且
agent.is_grasping（同 is_obj_pickup）。第一次检测到即写 first_pick，并以
stop=True 交回官方循环结束该局：步前子目标是「第一次拿起」且拿的是目标 → success，否则 fail；
终局记录另带 truncated_first_pick=true。
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
from scripts.interpret.stopcube_runner import array_value, position, scalar_value

TASKS = ("ButtonUnmaskSwap", "VideoUnmaskSwap", "VideoRepick")
KIND = "normal"
CUBE_TASKS = ("VideoRepick",)
# 拿起阈值：容器同 subgoal_evaluate_func.is_bin_pickup，方块同 is_obj_pickup（另须夹住）。
BIN_PICK_Z = 0.15
CUBE_PICK_Z = 0.05
# 「第一次拿起」子目标按 task_list 名字前缀找；两个 Swap 任务的下标另行核对。
FIRST_PICK_PREFIX = {"VideoUnmaskSwap": "pick up the container", "ButtonUnmaskSwap": "pick up the container",
                     "VideoRepick": "pick up the correct cube"}
FIRST_PICK_EXPECTED = {"VideoUnmaskSwap": 1, "ButtonUnmaskSwap": 2}
# 本进程内被拿起即停截断的局 nonce；终局记录据此另带 truncated_first_pick=true。
TRUNCATED_NONCES = set()


@dataclasses.dataclass
class Args(official.Args):
    # 缺省关闭：行为与原入口相同。
    stop_after_first_pick: bool = False


def objects_of(u, task=None):
    """按任务取物体列表；task 缺省时按两个 Swap 任务处理（原行为）。"""
    return list(u.spawned_cubes) if task in CUBE_TASKS else list(u.spawned_bins)


def snapshot(env, task=None):
    """只读当前物理状态与子目标序号；不调用任何判定函数，也不写环境。"""
    u = env.unwrapped
    return {"env_step": scalar_value(getattr(u, "elapsed_steps", None)),
            "tcp": position(getattr(u.agent, "tcp", None)),
            "bins": [position(actor) for actor in objects_of(u, task)],
            "subgoal_index": scalar_value(getattr(u, "timestep", None))}


def first_target_bin_index(u, task=None):
    """目标物体在物体列表中的下标：Swap 任务是 selected_bins[0]，VideoRepick 是 target_cube_1；按对象身份比较。"""
    if task in CUBE_TASKS:
        target, message = u.target_cube_1, "target_cube_1 不在 spawned_cubes 中"
    else:
        target, message = u.selected_bins[0], "selected_bins[0] 不在 spawned_bins 中"
    for index, actor in enumerate(objects_of(u, task)):
        if actor is target:
            return index
    raise official.ContractError(message)


def object_index(objects, actor):
    """按对象身份找下标；actor 为空（交换对尚未确定）或不在列表中时返回 None。"""
    for index, candidate in enumerate(objects):
        if actor is not None and candidate is actor:
            return index
    return None


def swap_schedule_of(u, task):
    """swap_schedule 元组 (actor_a, actor_b, start, end) → [[a, b, start, end], ...]；无交换（hard 档）时为空表。
    元组里的 actor 就是 swap_pair{i}_idx1／idx2，start／end 是 elapsed step。"""
    objects = objects_of(u, task)
    return [[object_index(objects, a), object_index(objects, b), int(start), int(end)]
            for a, b, start, end in (getattr(u, "swap_schedule", None) or [])]


def first_pick_subgoal_index(u, task):
    """task_list 里第一个「第一次拿起」子目标的下标（VideoRepick 的下标随 swap_times 变，只能按名字找）。"""
    prefix = FIRST_PICK_PREFIX[task]
    found = next((i for i, entry in enumerate(u.task_list) if str(entry.get("name", "")).startswith(prefix)), None)
    if found is None or FIRST_PICK_EXPECTED.get(task, found) != found:
        raise official.ContractError(f"{task} 的第一次拿起子目标下标不符：{found}")
    return found


def flag_value(value):
    """环境的 successflag／failureflag（单元素张量）→ bool；属性不存在时为 None。"""
    value = scalar_value(value)
    return None if value is None else bool(value)


def picked_index(u, task):
    """步后状态里第一个被拿起的物体下标（按物体列表顺序），没有则 None。
    位姿是 float32 张量，官方判定里阈值也按 float32 比较，这里同样转 float32 再比，边界值才一致。"""
    for index, actor in enumerate(objects_of(u, task)):
        z = np.float32(position(actor)[2])
        threshold = CUBE_PICK_Z if task in CUBE_TASKS else BIN_PICK_Z
        # 容器也要求夹爪抓着：ButtonUnmaskSwap 开局 0–64 步环境自己把全部容器抬起再放下，z 会超过 0.15。
        if z > np.float32(threshold) and bool(scalar_value(u.agent.is_grasping(actor))):
            return index
    return None


class TraceRunner(EnvRunner):
    """官方环境实例的记录包装：送入环境的动作与 EnvRunner 完全相同。"""

    def begin_trace(self, nonce, root, stop_after_first_pick=False):
        if self.env_id not in TASKS:
            raise official.ContractError(f"轨迹入口只接受 {TASKS}")
        self.stop_after_first_pick = bool(stop_after_first_pick)
        # 扩展输出只在新开关或 VideoRepick 时写，Swap 任务关闭开关时与原输出逐字节一致。
        self.extended = self.stop_after_first_pick or self.env_id in CUBE_TASKS
        self.first_pick = None
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
        if self.extended:
            self.receipt.update(target_index=None, swap_schedule=None, first_pick_subgoal_index=None,
                                stop_after_first_pick=self.stop_after_first_pick)
            if self.stop_after_first_pick:
                self.receipt.update(first_pick=None, truncated_first_pick=False)
        self.action_index = self.row_in_chunk = 0
        self.infer_seq = self.chunk_start_env_step = None

    def get_init_obs(self):
        result = super().get_init_obs()
        self.last_state = np.asarray(result["states"][-1]).copy()
        u = self.env.unwrapped
        task = self.env_id
        # 场景在 reset 内建好，这些量只能在 reset 之后读。VideoRepick 没有 pick_times／color_names，hard 档没有交换。
        schedule = getattr(u, "swap_schedule", None) or []
        target = first_target_bin_index(u, task)
        self.receipt.update(D=len(result["images"]) - 1, exec_start_idx=len(result["images"]) - 1,
                            initial_env_step=snapshot(self.env, task)["env_step"], swap_times=int(u.swap_times),
                            pick_times=None if task in CUBE_TASKS else int(u.pick_times),
                            swap_end=int(schedule[-1][3]) if schedule else None,
                            color_names=None if task in CUBE_TASKS else list(u.color_names), prompt=result["task_goal"],
                            first_target_bin_index=target)
        if self.extended:
            # Video 类的交换在 reset 内的示范里完成，此处已可读；ButtonUnmaskSwap 在 finish_trace 再读一次。
            self.receipt.update(target_index=target, swap_schedule=swap_schedule_of(u, task),
                                first_pick_subgoal_index=first_pick_subgoal_index(u, task))
            if task in CUBE_TASKS:
                self.receipt["num_repeats"] = int(u.num_repeats)
        return result

    def begin_chunk(self, infer_seq):
        self.infer_seq = infer_seq
        self.row_in_chunk = 0
        self.chunk_start_env_step = snapshot(self.env, self.env_id)["env_step"]

    def step(self, action):
        before = snapshot(self.env, self.env_id)
        state_before = self.last_state.copy()
        command = array_value(action)
        result = super().step(action)
        after = snapshot(self.env, self.env_id)
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
        if self.extended:
            record["objects"] = after["bins"]
        if self.stop_after_first_pick and self.first_pick is None:
            result = self.check_first_pick(result, before, after, record)
        self.trace_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.trace_stream.flush()
        self.row_in_chunk += 1
        self.action_index += 1
        return result

    def check_first_pick(self, result, before, after, record):
        """第一次有物体被拿起：记 first_pick，并把该步改成终局交回官方循环；否则原样返回。"""
        u = self.env.unwrapped
        index = picked_index(u, self.env_id)
        if index is None:
            return result
        is_target = index == self.receipt["target_index"]
        # 子目标在同一步的 evaluate 里就会前进（拿对）或停在原处并置 failureflag（拿错），
        # 所以「拿起时的子目标」取步前的 timestep，步后值另记。
        on_time = before["subgoal_index"] == self.receipt["first_pick_subgoal_index"]
        self.first_pick = {"index": index, "env_step": after["env_step"], "action_index": self.action_index,
                           "is_target": is_target, "subgoal_index_at_pick": before["subgoal_index"],
                           "subgoal_index_after_pick": after["subgoal_index"],
                           "successflag": flag_value(getattr(u, "successflag", None)),
                           "failureflag": flag_value(getattr(u, "failureflag", None)),
                           "outcome": "success" if on_time and is_target else "fail"}
        record["first_pick"] = self.first_pick
        self.receipt.update(first_pick=self.first_pick, truncated_first_pick=True)
        TRUNCATED_NONCES.add(self.receipt["episode_nonce"])
        return result[0], True, self.first_pick["outcome"]

    def finish_trace(self, outcome, video_dir):
        self.trace_stream.close()
        if self.extended and self.receipt["exec_start_idx"] is not None:
            # ButtonUnmaskSwap 的第二个物体在交换窗开始时才确定，结束（含截断）时再读一次。
            self.receipt["swap_schedule"] = swap_schedule_of(self.env.unwrapped, self.env_id)
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
        env_runner.begin_trace(episode_nonce, self.save_dir, getattr(self.args, "stop_after_first_pick", False))
        outcome = None
        try:
            outcome = super().eval_each_episode(env_runner, video_save_dir, client, episode_nonce)
            return outcome
        finally:
            env_runner.finish_trace(outcome, video_save_dir)


def write_record_marking_truncation(write):
    """包住官方 write_record：只给被截断局的 terminal 记录加 truncated_first_pick=true，其余原样。"""
    def wrapped(path, record):
        if record.get("record_type") == "terminal" and record.get("episode_nonce") in TRUNCATED_NONCES:
            record = dict(record, truncated_first_pick=True)
        return write(path, record)
    return wrapped


def evaluate(args):
    """在独立客户端进程里复用官方主入口，结束后恢复模块内的评估器与写记录函数绑定。"""
    if any(episode["task_name"] not in TASKS for episode in official.load_chunk(args)):
        raise official.ContractError(f"轨迹入口只接受 {TASKS}")
    previous, previous_write = official.EpisodeEvaluator, official.write_record
    official.EpisodeEvaluator = TraceEvaluator
    if getattr(args, "stop_after_first_pick", False):
        official.write_record = write_record_marking_truncation(previous_write)
    try:
        return official.evaluate(args, runner_factory=TraceRunner)
    finally:
        official.EpisodeEvaluator, official.write_record = previous, previous_write


if __name__ == "__main__":
    import tyro
    raise SystemExit(evaluate(tyro.cli(Args)))
