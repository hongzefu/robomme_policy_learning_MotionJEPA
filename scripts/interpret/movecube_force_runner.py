"""MoveCube 强推恢复客户端：开局按表强推 k 步后放手，放手后抓起正确物体即结束本局。

官方评估循环、RPC 与续包原样复用：强推期间策略照常收观测、照常推理，只替换送入环境的动作；
放手点是 16 的倍数，放手步执行当前边界的新推理动作。不录任何注意力或隐状态。

抓取判定只调只读的 is_obj_pickup，从放手步起每步环境执行后查一次（方向 A 查 u.cube，方向 B 查 u.peg_tail、
u.peg_head）；不调会写 currentpickup 的判定函数。本步环境已判 fail 则不算抓起。
抓起时返回 stop=True、outcome="success" 走官方正常终局路径，这里的 success 不是任务成功：
分析只读回执的 grasp_success，环境原始状态另写在 env_status。强推途中环境终局不特判，照实结束。

每局写 traces/<nonce>/step_trace.jsonl 与 traces/<nonce>/meta/force_receipt.json（异常退出也写）。
step_trace 每行：event、episode_key、action_index、env_step（步前值）、env_step_after、infer_seq、row_in_chunk、
chunk_start_env_step、control_source（force|policy）、command_action、executed_action、env_status，
以及步后位置 tcp、cube、peg_tail、peg_head。
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
from scripts.interpret.movecube_force_table import FORCE_STEPS, TASK, WAYS
from scripts.interpret.stopcube_runner import array_value, position, scalar_value

FORCE_KS = (0, 16, 32, 48, 64, 80)
# 方向 -> 放手后算抓取成功的物体属性名。
GRASP_OBJECTS = {"A": ("cube",), "B": ("peg_tail", "peg_head")}
XYZ_TOLERANCE = 1e-5


@dataclasses.dataclass
class Args(official.Args):
    force_table: str = ""
    force_k: int = -1


def load_table(path):
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if table.get("schema_version") != 1 or table.get("force_steps") != FORCE_STEPS or not isinstance(table.get("episodes"), dict):
        raise official.ContractError("动作表须含 schema_version=1、force_steps=80 与 episodes 对象")
    return table["episodes"]


def is_pickup(u, obj):
    """只读判定：物体 z>0.05 且确实夹住。"""
    from robomme.robomme_env.utils.subgoal_evaluate_func import is_obj_pickup

    return bool(is_obj_pickup(u, obj=obj))


def snapshot(env):
    u = env.unwrapped
    return {"env_step": scalar_value(u.elapsed_steps), "tcp": position(u.agent.tcp), "cube": position(u.cube),
            "peg_tail": position(u.peg_tail), "peg_head": position(u.peg_head)}


def xyz_matches(actual, expected):
    return (actual is not None and expected is not None and len(actual) == len(expected)
            and bool(np.all(np.abs(np.asarray(actual) - np.asarray(expected)) <= XYZ_TOLERANCE)))


class TraceRunner(EnvRunner):
    """官方环境实例的包装：放手前替换动作，放手后只读判定抓取。"""

    def begin_trace(self, args, nonce, root, table):
        if self.env_id != TASK:
            raise official.ContractError(f"强推入口只接受 {TASK}")
        self.key = f"{self.env_id}:{self.episode_id}"
        self.force_k = args.force_k
        self.entry = table.get(self.key)
        if self.entry is None:
            raise official.ContractError(f"动作表缺局 {self.key}")
        self.trace_dir = Path(root) / "traces" / nonce
        (self.trace_dir / "meta").mkdir(parents=True, exist_ok=False)
        self.trace_path = self.trace_dir / "step_trace.jsonl"
        self.trace_stream = self.trace_path.open("x", encoding="utf-8")
        self.receipt = {"schema_version": 1, "episode_key": self.key, "episode_nonce": nonce, "episode_id": self.episode_id,
                        "env_seed": self.env_seed, "difficulty": self.difficulty, "force_k": self.force_k,
                        "way": None, "direction": None, "origin_env_step": None,
                        "grasp_success": 0, "grasp_step": None, "grasped_object": None,
                        "outcome": None, "env_status": None, "reached_release": False, "actual_release_step": None,
                        "actions_executed": 0, "trace_path": str(self.trace_path.resolve()),
                        "terminal_path": str((Path(root) / f"{self.key}.terminal.json").resolve()), "video_path": None}
        self.action_index = self.row_in_chunk = 0
        self.infer_seq = self.chunk_start_env_step = None

    def get_init_obs(self):
        result = super().get_init_obs()
        u = self.env.unwrapped
        entry = self.entry
        origin = snapshot(self.env)["env_step"]
        actions = np.asarray(entry["actions"], dtype=np.float64)
        if (origin != entry["first_env_step"] or self.env_seed != entry["env_seed"] or u.way != entry["way"]
                or u.way not in WAYS or WAYS[u.way][0] != entry["direction"]
                or not xyz_matches(position(u.peg_tail), entry["peg_tail_xyz"])
                or not xyz_matches(position(u.cube), entry["cube_xyz"])
                or not xyz_matches(position(u.goal_site), entry["goal_xyz"])
                or actions.shape != (FORCE_STEPS, 8) or not np.isfinite(actions).all()):
            raise official.ContractError(f"动作表与现场身份不符 {self.key}")
        self.origin = origin
        self.force_actions = actions
        self.grasp_objects = GRASP_OBJECTS[entry["direction"]]
        self.receipt.update(way=u.way, direction=entry["direction"], origin_env_step=origin)
        return result

    def begin_chunk(self, infer_seq):
        self.infer_seq = infer_seq
        self.row_in_chunk = 0
        self.chunk_start_env_step = snapshot(self.env)["env_step"]

    def picked_object(self):
        u = self.env.unwrapped
        return next((name for name in self.grasp_objects if is_pickup(u, getattr(u, name))), None)

    def step(self, action):
        elapsed = snapshot(self.env)["env_step"]
        offset = elapsed - self.origin
        command = np.asarray(action).copy()
        forced = offset < self.force_k
        if forced:
            executed = self.force_actions[int(offset)].copy()
        else:
            executed = command
            if offset == self.force_k:
                if self.row_in_chunk != 0 or self.chunk_start_env_step != elapsed:
                    raise official.ContractError("放手步必须执行当前边界的新推理动作")
                self.receipt.update(reached_release=True, actual_release_step=elapsed)
        obs, stop, outcome = super().step(executed)
        after = snapshot(self.env)
        self.receipt["env_status"] = outcome
        if not forced and outcome not in ("fail", "error"):
            grasped = self.picked_object()
            if grasped is not None:
                self.receipt.update(grasp_success=1, grasp_step=after["env_step"], grasped_object=grasped)
                stop, outcome = True, "success"
        record = {"event": "step", "episode_key": self.key, "action_index": self.action_index,
                  "env_step": elapsed, "env_step_after": after["env_step"], "infer_seq": self.infer_seq,
                  "row_in_chunk": self.row_in_chunk, "chunk_start_env_step": self.chunk_start_env_step,
                  "control_source": "force" if forced else "policy", "command_action": array_value(command),
                  "executed_action": array_value(executed), "env_status": self.receipt["env_status"],
                  "tcp": after["tcp"], "cube": after["cube"], "peg_tail": after["peg_tail"], "peg_head": after["peg_head"]}
        self.trace_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.trace_stream.flush()
        self.row_in_chunk += 1
        self.action_index += 1
        return obs, stop, outcome

    def finish_trace(self, outcome, video_dir):
        self.trace_stream.close()
        video = Path(video_dir) / f"{self.env_id}_ep{self.episode_id}_{outcome}.mp4"
        self.receipt.update(outcome=outcome, actions_executed=self.action_index,
                            video_path=str(video.resolve()) if outcome else None)
        official.write_record(self.trace_dir / "meta" / "force_receipt.json", self.receipt)


class TraceEvaluator(official.EpisodeEvaluator):
    """只安装强推上下文；历史、动作队列与终局全部走官方 eval_each_episode。"""

    table = None

    def get_action_chunk(self, *args, **kwargs):
        actions = super().get_action_chunk(*args, **kwargs)
        self.trace_runner.begin_chunk(self.replan_seq - 1)
        return actions

    def eval_each_episode(self, env_runner, video_save_dir, client, episode_nonce):
        self.trace_runner = env_runner
        env_runner.begin_trace(self.args, episode_nonce, self.save_dir, self.table)
        outcome = None
        try:
            outcome = super().eval_each_episode(env_runner, video_save_dir, client, episode_nonce)
            return outcome
        finally:
            env_runner.finish_trace(outcome, video_save_dir)


def evaluate(args):
    """在独立客户端进程里复用官方主入口，结束后恢复模块内的评估器绑定。"""
    if args.force_k not in FORCE_KS or not args.force_table:
        raise official.ContractError(f"必须提供动作表，force_k 只能取 {FORCE_KS}")
    if any(episode["task_name"] != TASK for episode in official.load_chunk(args)):
        raise official.ContractError(f"强推入口只接受 {TASK}")
    TraceEvaluator.table = load_table(args.force_table)
    previous = official.EpisodeEvaluator
    official.EpisodeEvaluator = TraceEvaluator
    try:
        return official.evaluate(args, runner_factory=TraceRunner)
    finally:
        official.EpisodeEvaluator = previous


if __name__ == "__main__":
    import tyro
    raise SystemExit(evaluate(tyro.cli(Args)))
