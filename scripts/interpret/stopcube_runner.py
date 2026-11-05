"""StopCube 逐步记录与固定关节悬停；复用官方评估循环、RPC 和终局。

配置为 {"schema_version": 1, "episodes": {"StopCube:0": {
"takeover_step": 90, "release_step": 160, "hold_qpos": [8维关节/夹爪]}}}。
两个时刻均为官方 elapsed_steps 的步前值；恢复必须落在现场首个执行步加 16k。
normal 仅记录，hold_extended 仅替换接管区间内送入环境的动作。
oracle_query 消费 Oracle 动作表，独立按4步询问并丢弃模型执行权；
oracle_calibrate 消费固定选点，交还前仅提交真实历史，交还后按16步执行新推理。
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
    query_interval: int = 4
    query_record: int = 10
    calibration_plan: str = ""
    calibration_kind: str = "fixed_release"
    calibration_model_id: str = ""


def load_config(path):
    """读取固定接管表；具体局的参数在初始化观测后核对。"""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not isinstance(config.get("episodes"), dict):
        raise official.ContractError("StopCube 配置须含 schema_version=1 与 episodes 对象")
    return config["episodes"]


ORACLE_MODES = ("oracle_query", "oracle_calibrate")
CALIBRATION_KINDS = ("fixed_release", "high_probability", "low_probability", "early_press")


def load_calibration(path, model_id=""):
    """消费已固定的单模型选点，或按模型身份选择多模型选点。"""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise official.ContractError("校准选点须为 schema_version=1")
    if "models" in config:
        config = config["models"].get(model_id, {})
    if not isinstance(config.get("episodes"), dict):
        raise official.ContractError("校准选点缺少所选模型的 episodes")
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
        self.oracle = config.get(self.key) if self.mode in ORACLE_MODES else None
        if self.mode in ORACLE_MODES and self.oracle is None:
            raise official.ContractError(f"Oracle 动作表缺局 {self.key}")
        self.control_source = None
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
        if self.oracle is not None:
            table = self.oracle
            actions = np.asarray(table["actions"])
            first_step, last_step = table["first_env_step"], table["last_env_step"]
            if (first_step != first["env_step"] or table["env_seed"] != self.env_seed
                    or table["move_interval"] != first["move_interval"]
                    or table["target_pass"] != first["target_pass"]
                    or actions.ndim != 2 or actions.shape[1:] != (8,)
                    or not np.issubdtype(actions.dtype, np.number) or not np.isfinite(actions).all()
                    or last_step != first_step + len(actions) - 1
                    or table["terminal_env_step"] <= last_step):
                raise official.ContractError("Oracle 动作表身份、连续步号、终局或有限动作不符")
            self.oracle_actions = actions
            self.receipt.update(oracle_first_env_step=first_step, oracle_last_env_step=last_step,
                                oracle_terminal_env_step=table["terminal_env_step"],
                                oracle_trace_path=table.get("trace_path"), reach_step=table["reach_step"],
                                z_hover=table["z_hover"], history_batches=0, query_points=0,
                                execution_infers=0, query_steps=[], query_records=[],
                                button_trigger_step=None, calibration_stop=None)
            self.receipt["extra"].update(hover_z=table["z_hover"], hover_source="oracle_table")
        return result

    def oracle_action(self):
        """按真实步前值查表，绝不复用末行填补越界。"""
        elapsed = snapshot(self.env)["env_step"]
        index = elapsed - self.oracle["first_env_step"]
        if int(index) != index or not 0 <= index < len(self.oracle_actions):
            raise official.ContractError(f"Oracle 动作表越界 env_step={elapsed}")
        return self.oracle_actions[int(index)].copy()

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
        if self.mode in ORACLE_MODES:
            record["control_source"] = self.control_source
            if after["button_pressed"] and self.receipt["button_trigger_step"] is None:
                self.receipt["button_trigger_step"] = after["env_step"]
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
    calibration = None

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
            if self.args.stopcube_mode in ORACLE_MODES:
                outcome = self.eval_oracle(env_runner, video_save_dir, client, episode_nonce)
            else:
                outcome = super().eval_each_episode(env_runner, video_save_dir, client, episode_nonce)
            return outcome
        finally:
            env_runner.finish_trace(outcome, video_save_dir)

    def oracle_context(self):
        runner = self.trace_runner
        elapsed = snapshot(runner.env)["env_step"]
        return {"mode": runner.mode, "episode_key": runner.key, "env_step": int(elapsed),
                "pre_reach": elapsed < runner.oracle["reach_step"],
                "raw_state": np.asarray(runner.last_state, dtype=np.float32).copy()}

    def submit_history(self, client, state):
        from scripts.evaluation.runtime_contract import payload_sha256

        payload = official.pack_buffer(state.image_buffer, state.state_buffer, state.exec_start_idx)
        count = len(state.image_buffer)
        if self.batch_seq == 0 and count != state.exec_start_idx + 1:
            raise official.ContractError("Oracle 首包必须完整包含 D+1 帧")
        payload["oracle_context"] = self.oracle_context()
        rpc = {"episode_nonce": self.episode_nonce, "batch_seq": self.batch_seq,
               "first_global_frame": self.next_global_frame, "frame_count": count,
               "payload_sha256": payload_sha256(payload)}
        payload["eval_rpc"] = rpc
        official.require_ack(client.request(payload), "add_buffer_finished", rpc)
        self.batch_seq += 1
        self.next_global_frame += count
        self.trace_runner.receipt["history_batches"] += 1
        state.clear_buffers()

    def oracle_infer(self, client, obs, prompt, *, query):
        img, wrist, raw_state = obs
        rpc = {"episode_nonce": self.episode_nonce, "replan_seq": self.replan_seq,
               "last_global_frame": self.next_global_frame - 1}
        response = client.request({"observation/image": img, "observation/wrist_image": wrist,
                                   "observation/state": raw_state, "prompt": prompt, "eval_rpc": rpc,
                                   "oracle_context": self.oracle_context()})
        if response.get("eval_rpc") != rpc:
            raise official.UncertainRPC("Oracle 推理响应身份或序号不符")
        actions = np.asarray(response.get("actions"))
        if actions.shape != (20, 8) or not np.issubdtype(actions.dtype, np.number) or not np.isfinite(actions).all():
            raise official.ContractError("Oracle 推理必须返回有限 (20,8) 动作")
        receipt = self.trace_runner.receipt
        if query:
            predictions = np.asarray(response.get("query_actions"))
            if (predictions.shape != (self.args.query_record, 20, 8)
                    or not np.issubdtype(predictions.dtype, np.number) or not np.isfinite(predictions).all()
                    or response.get("query_seq") != self.replan_seq or not response.get("query_record")):
                raise official.ContractError("询问缺少匹配序号的完整预测与存盘回执")
            receipt["query_points"] += 1
            receipt["query_steps"].append(self.oracle_context()["env_step"])
            receipt["query_records"].append(response["query_record"])
        else:
            receipt["execution_infers"] += 1
            if receipt["execution_infers"] == 1:
                receipt["first_release_prediction"] = actions.tolist()
        self.trace_runner.begin_chunk(self.replan_seq)
        self.replan_seq += 1
        if "motion_downsample_steps" in response:
            self.motion_downsample_steps = response["motion_downsample_steps"]
        return actions[:16]

    def eval_oracle(self, runner, video_save_dir, client, episode_nonce):
        """询问和校准独立调度；全部历史均来自实际执行后的观测。"""
        self.episode_nonce = episode_nonce
        self.batch_seq = self.replan_seq = self.next_global_frame = 0
        rpc = {"episode_nonce": episode_nonce, "expected_run_identity_sha256": self.args.run_identity_sha256,
               "expected_server_nonce": self.args.expected_server_nonce}
        official.require_ack(client.request({"reset": True, "eval_rpc": rpc,
                                            "oracle_context": {"mode": runner.mode, "episode_key": runner.key}}),
                             "reset_finished", rpc)
        state = official.EpisodeState()
        prompt, self.recorder = self.init_episode(runner, state, video_save_dir)
        obs = state.get_current_obs()
        origin = runner.receipt["initial_env_step"]
        limit = runner.oracle["move_interval"] * runner.oracle["target_pass"]
        selection = None
        release = None
        if runner.mode == "oracle_calibrate":
            selection = (self.calibration or {}).get(runner.key, {}).get(self.args.calibration_kind)
            if not selection or selection.get("status", "selected") != "selected":
                raise official.ContractError("校准运行只接受已选定的非跳过选点")
            release = selection["release_step"]
            duration = selection.get("stop_after_release")
            expected_duration = None if self.args.calibration_kind == "fixed_release" else 48
            if (type(release) is not int or release < origin or (release - origin) % 4
                    or release < runner.oracle["reach_step"] or release > limit
                    or duration != expected_duration):
                raise official.ContractError("交还点须到位后、首步+4k，固定放手到终局，其余恰好48步")
            runner.receipt.update(calibration_kind=self.args.calibration_kind, calibration=dict(selection),
                                  release_step=release, actual_release_step=None,
                                  model_id=self.args.calibration_model_id or self.args.model_kind)
        interval = self.args.query_interval if runner.mode == "oracle_query" else 4
        released = False
        released_actions = 0
        while True:
            elapsed = snapshot(runner.env)["env_step"]
            at_boundary = (elapsed - origin) % interval == 0
            if runner.mode == "oracle_query":
                if at_boundary and elapsed <= limit:
                    self.submit_history(client, state)
                    self.oracle_infer(client, obs, prompt, query=True)
                runner.control_source = "oracle"
                action = runner.oracle_action()
            else:
                if not released:
                    if elapsed > release:
                        raise official.ContractError("真实环境跨过了固定交还点")
                    if at_boundary:
                        self.submit_history(client, state)
                    if elapsed == release:
                        released = True
                        runner.receipt["actual_release_step"] = elapsed
                if released:
                    if not state.action_plan:
                        if released_actions:
                            self.submit_history(client, state)
                        state.action_plan.extend(self.oracle_infer(client, obs, prompt, query=False))
                    runner.control_source = "model"
                    action = state.action_plan.popleft()
                else:
                    runner.control_source = "oracle"
                    action = runner.oracle_action()
            self.step_in_flight = True
            new_obs, stop, outcome = runner.step(action)
            state.count += 1
            self.actions_executed = state.count
            self.terminal_observed = stop and outcome in ("success", "fail", "timeout")
            if outcome == "error" or new_obs is None or any(item is None for item in new_obs):
                raise RuntimeError("环境 step 未返回完整观测")
            self.step_in_flight = False
            if state.count > self.args.max_steps:
                self.terminal_observed = True
                return "timeout"
            obs = new_obs
            img, wrist, raw_state = obs
            state.add_observation(img, wrist, raw_state)
            self.record_frame(image=img.copy(), wrist_image=wrist.copy(), state=raw_state.copy(), action=action.copy())
            if released:
                released_actions += 1
            if stop:
                if not self.terminal_observed:
                    raise official.ContractError(f"环境终止但没有正常终局状态：{outcome}")
                if released:
                    runner.receipt["calibration_stop"] = "official_terminal"
                return outcome
            if released and selection["stop_after_release"] is not None:
                if runner.receipt["button_trigger_step"] is not None or released_actions >= selection["stop_after_release"]:
                    runner.receipt["calibration_stop"] = ("button_trigger" if runner.receipt["button_trigger_step"]
                                                            is not None else "release_48_steps")
                    self.terminal_observed = True
                    return "calibration_stop"


def evaluate(args):
    """在独立客户端进程里复用官方主入口，结束后恢复模块内的评估器绑定。"""
    if args.stopcube_mode not in ("normal", "hold_extended", *ORACLE_MODES) or not args.stopcube_config:
        raise official.ContractError("必须提供 StopCube 模式与配置文件")
    episodes = official.load_chunk(args)
    if any(episode["task_name"] != "StopCube" for episode in episodes):
        raise official.ContractError("StopCube 入口不能执行其他任务")
    TraceEvaluator.config = load_config(args.stopcube_config)
    if args.stopcube_mode in ORACLE_MODES:
        if args.query_interval not in (4, 16) or args.query_record != 10:
            raise official.ContractError("Oracle 询问固定10预测，间隔只允许4或节奏诊断16")
    if args.stopcube_mode == "oracle_calibrate":
        if not args.calibration_plan or args.calibration_kind not in CALIBRATION_KINDS:
            raise official.ContractError("校准需要固定选点文件与四类运行之一")
        TraceEvaluator.calibration = load_calibration(args.calibration_plan, args.calibration_model_id)
        chosen = TraceEvaluator.calibration
        skipped = [ep for ep in episodes if chosen.get(ep["episode_key"], {}).get(args.calibration_kind, {}).get("status") == "skipped"]
        if skipped:
            if len(skipped) != len(episodes):
                raise official.ContractError("校准 grant 不得混合跳过局与已选局；请在驱动分块前筛选")
            for ep in skipped:
                print(f"CALIBRATION_SKIP episode={ep['episode_key']} kind={args.calibration_kind} "
                      f"reason={chosen[ep['episode_key']][args.calibration_kind].get('reason')}", flush=True)
            return 0
    previous = official.EpisodeEvaluator
    official.EpisodeEvaluator = TraceEvaluator
    try:
        return official.evaluate(args, runner_factory=TraceRunner)
    finally:
        official.EpisodeEvaluator = previous


if __name__ == "__main__":
    import tyro
    raise SystemExit(evaluate(tyro.cli(Args)))
