"""PatternLock、RouteStick 的逐步轨迹客户端；官方评估循环、RPC、动作与终局原样复用，只加只读记录。

结构对应 bin_trace_runner，文件名与路径一致：每局写 traces/<nonce>/step_trace.jsonl（可被 rollout_offline.load_trace 消费）
与 traces/<nonce>/meta/hold_receipt.json。

子目标序号取 current_task_index，timestep 一并记录但不作标签（它在到位那一刻就变，比前者早若干帧）。
执行段逐步只读这两个属性。示范帧在 reset 内生成且被过滤过，读不到逐帧序号，所以只在 get_init_obs 期间给
DemonstrationWrapper 实例包一层 _augment_obs_and_info，往 info 里多放一个序号，返回前撤销；序号列表写入 receipt 的 demo_task_index。

计划 1115（steer-4task-1115）§4.2 新增三个开关，缺省全关，关闭时行为、行与 receipt 与改动前逐字节一致：
  --episode-specs <json>：{"<任务>:<局号>": {"seed", "difficulty", "episode_spec", "sampling_config"}}；清单里出现在表中的局
      改用 benchmark 的 make_env_for_spec 建环境（gym.make 的 episode_spec / sampling_config 参数，max_steps 与 builder 同值），
      seed / 难度须与 metadata 解析值一致；不在表中的局照旧走 metadata。receipt 记 episode_spec_sha256。
  --stop-after-step j：执行段每步步后读 current_task_index，越过第 j 个执行子目标（E_j = m+2+j）即以 success 交回官方循环结束该局
      （环境自己判定的终局优先，整局失败照常 fail）；receipt 写 truncated_step=j（未越过为 null），terminal 记录另带 truncated_step。
  --stop-after-first-infer：首次推理返回后执行第一步即以 fail 交回官方循环结束（只为让服务端在线编码示范窗口并 --record-motion），
      receipt 与 terminal 另带 truncated_first_infer=true。两个截断开关互斥。
"""
from __future__ import annotations

import dataclasses
import hashlib
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

TASKS = ("PatternLock", "RouteStick")
KIND = "normal"
DEMO_KEY = "subgoal_task_index"
# 本进程内被截断的局 nonce → 终局记录要附加的字段。
TRUNCATED = {}
# --episode-specs 载入后的表（"<任务>:<局号>" → 条目）；evaluate 里设置。
EPISODE_SPECS = {}


@dataclasses.dataclass
class Args(official.Args):
    # 缺省全关：行为与原入口相同。
    episode_specs: str = ""
    stop_after_step: int = 0
    stop_after_first_infer: bool = False


def load_episode_specs(path):
    """读 --episode-specs 表并校验条目形状；空路径返回 {}。"""
    if not path:
        return {}
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(table, dict):
        raise official.ContractError("episode_specs 必须是 {局键: 条目} 字典")
    for key, entry in table.items():
        task, _, ep = str(key).rpartition(":")
        if task not in TASKS or not ep.isdigit():
            raise official.ContractError(f"episode_specs 局键非法：{key!r}")
        if not isinstance(entry, dict) or not {"seed", "difficulty", "episode_spec", "sampling_config"} <= set(entry):
            raise official.ContractError(f"episode_specs[{key}] 缺 seed / difficulty / episode_spec / sampling_config")
        if entry["episode_spec"].get("task") != task:
            raise official.ContractError(f"episode_specs[{key}] 的 episode_spec.task 与局键任务不符")
    return table


def spec_sha256(entry):
    return hashlib.sha256(json.dumps({k: entry[k] for k in ("seed", "difficulty", "episode_spec", "sampling_config")},
                                     sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_truncation_args(stop_after_step, stop_after_first_infer):
    if type(stop_after_step) is not int or stop_after_step < 0:
        raise official.ContractError("stop_after_step 必须是非负整数（0 = 关闭）")
    if stop_after_step and stop_after_first_infer:
        raise official.ContractError("stop_after_step 与 stop_after_first_infer 互斥")


def crossed_step(task_index_after, m, j):
    """步后子目标序号越过第 j 个执行子目标（E_j = m+2+j）。"""
    return task_index_after is not None and int(task_index_after) > int(m) + 2 + int(j)


def snapshot(env):
    """只读当前物理状态与子目标序号；不调用任何判定函数，也不写环境。"""
    u = env.unwrapped
    return {"env_step": scalar_value(getattr(u, "elapsed_steps", None)),
            "tcp": position(getattr(u.agent, "tcp", None)),
            "task_index": int(u.current_task_index),
            "timestep": int(u.timestep)}


def find_demonstration_wrapper(env):
    """沿 .env 链找 DemonstrationWrapper 实例：外层 gym wrapper 不转发私有属性，钩子只能装在它本身上。"""
    from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
    wrapper = env
    while not isinstance(wrapper, DemonstrationWrapper):
        wrapper = wrapper.env
    return wrapper


class TraceRunner(EnvRunner):
    """官方环境实例的记录包装：送入环境的动作与 EnvRunner 完全相同。"""

    def make_env(self, episode_id):
        """局在 --episode-specs 表中时按 spec 建环境，否则走原路径。"""
        entry = EPISODE_SPECS.get(f"{self.env_id}:{int(episode_id)}")
        self.spec_entry = entry
        if entry is None:
            return super().make_env(episode_id)
        from robomme.env_record_wrapper import make_env_for_spec
        seed, difficulty = self.env_builder.resolve_episode(episode_id)
        if int(seed) != int(entry["seed"]) or str(difficulty) != str(entry["difficulty"]):
            raise official.ContractError(f"episode_specs[{self.env_id}:{episode_id}] 的 seed/难度与 metadata 不符："
                                         f"{entry['seed']}/{entry['difficulty']} vs {seed}/{difficulty}")
        self.env_seed = int(seed)
        print(f"EVAL_EPISODE_SPEC task={self.env_id} ep={episode_id} env_seed={seed} difficulty={difficulty} "
              f"spec_sha256={spec_sha256(entry)}", flush=True)
        max_steps = int(self.env_builder.max_steps_without_demonstration) - 2
        self.env = make_env_for_spec(self.env_id, int(seed), str(difficulty), entry["episode_spec"],
                                     entry["sampling_config"], max_steps=max_steps)
        self.episode_id = episode_id
        self.difficulty = self.env.unwrapped.difficulty
        if self.difficulty != entry["difficulty"]:
            raise ValueError(f"难度未按 spec 生效：期望 {entry['difficulty']}，实际 {self.difficulty}")

    def begin_trace(self, nonce, root, stop_after_step=0, stop_after_first_infer=False):
        check_truncation_args(stop_after_step, stop_after_first_infer)
        self.stop_after_step = int(stop_after_step)
        self.stop_after_first_infer = bool(stop_after_first_infer)
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
                        "m": None, "demo_task_index": None, "D": None,
                        "exec_start_idx": None, "initial_env_step": None, "prompt": None}
        # 新开关全关时 receipt 与改动前逐字节一致。
        if getattr(self, "spec_entry", None) is not None:
            self.receipt["episode_spec_sha256"] = spec_sha256(self.spec_entry)
        if self.stop_after_step:
            self.receipt.update(stop_after_step=self.stop_after_step, truncated_step=None)
        if self.stop_after_first_infer:
            self.receipt["truncated_first_infer"] = False
        self.action_index = self.row_in_chunk = 0
        self.infer_seq = self.chunk_start_env_step = None

    def get_init_obs(self):
        wrapper = find_demonstration_wrapper(self.env)
        original = wrapper._augment_obs_and_info
        u = self.env.unwrapped

        def augment(obs, info, action):
            obs, info = original(obs, info, action)
            info[DEMO_KEY] = int(u.current_task_index)
            return obs, info

        wrapper._augment_obs_and_info = augment   # 实例级遮蔽类方法，del 后恢复原方法
        try:
            result = super().get_init_obs()
        finally:
            del wrapper._augment_obs_and_info
        demo_task_index = [int(value) for value in self.env.demonstration_data[4][DEMO_KEY]]
        if len(demo_task_index) != len(result["images"]):
            raise official.ContractError(
                f"demo_task_index 长度 {len(demo_task_index)} 不等于示范帧数 {len(result['images'])}")
        self.last_state = np.asarray(result["states"][-1]).copy()
        # 场景在 reset 内建好，这些量只能在 reset 之后读。
        self.receipt.update(m=len(u.selected_buttons) - 1, demo_task_index=demo_task_index,
                            D=len(result["images"]) - 1, exec_start_idx=len(result["images"]) - 1,
                            initial_env_step=snapshot(self.env)["env_step"], prompt=result["task_goal"])
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
                  "task_index_before": before["task_index"], "task_index_after": after["task_index"],
                  "timestep_before": before["timestep"], "timestep_after": after["timestep"]}
        result = self.check_truncation(result, after)
        self.trace_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.trace_stream.flush()
        self.row_in_chunk += 1
        self.action_index += 1
        return result

    def check_truncation(self, result, after):
        """截断开关：环境已给出终局时原样返回；否则按开关把这一步改成终局交回官方循环。"""
        if result[1]:
            return result
        nonce = self.receipt["episode_nonce"]
        if self.stop_after_first_infer:
            self.receipt["truncated_first_infer"] = True
            TRUNCATED[nonce] = {"truncated_first_infer": True}
            return result[0], True, "fail"
        if self.stop_after_step and crossed_step(after["task_index"], self.receipt["m"], self.stop_after_step):
            self.receipt["truncated_step"] = self.stop_after_step
            TRUNCATED[nonce] = {"truncated_step": self.stop_after_step}
            return result[0], True, "success"
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
        env_runner.begin_trace(episode_nonce, self.save_dir, getattr(self.args, "stop_after_step", 0),
                               getattr(self.args, "stop_after_first_infer", False))
        outcome = None
        try:
            outcome = super().eval_each_episode(env_runner, video_save_dir, client, episode_nonce)
            return outcome
        finally:
            env_runner.finish_trace(outcome, video_save_dir)


def write_record_marking_truncation(write):
    """包住官方 write_record：只给被截断局的 terminal 记录加截断字段，其余原样。"""
    def wrapped(path, record):
        extra = TRUNCATED.get(record.get("episode_nonce")) if record.get("record_type") == "terminal" else None
        if extra:
            record = dict(record, **extra)
        return write(path, record)
    return wrapped


def evaluate(args):
    """在独立客户端进程里复用官方主入口，结束后恢复模块内的评估器、写记录函数与 spec 表绑定。"""
    if any(episode["task_name"] not in TASKS for episode in official.load_chunk(args)):
        raise official.ContractError(f"轨迹入口只接受 {TASKS}")
    stop_step = getattr(args, "stop_after_step", 0)
    first_infer = getattr(args, "stop_after_first_infer", False)
    check_truncation_args(stop_step, first_infer)
    specs = load_episode_specs(getattr(args, "episode_specs", ""))
    previous, previous_write = official.EpisodeEvaluator, official.write_record
    official.EpisodeEvaluator = TraceEvaluator
    if stop_step or first_infer:
        official.write_record = write_record_marking_truncation(previous_write)
    EPISODE_SPECS.clear()
    EPISODE_SPECS.update(specs)
    try:
        return official.evaluate(args, runner_factory=TraceRunner)
    finally:
        official.EpisodeEvaluator, official.write_record = previous, previous_write
        EPISODE_SPECS.clear()


if __name__ == "__main__":
    import tyro
    raise SystemExit(evaluate(tyro.cli(Args)))
