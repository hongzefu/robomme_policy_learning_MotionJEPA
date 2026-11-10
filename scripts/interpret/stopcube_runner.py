"""StopCube 逐步记录与固定关节悬停；复用官方评估循环、RPC 和终局。

配置为 {"schema_version": 1, "episodes": {"StopCube:0": {
"takeover_step": 90, "release_step": 160, "hold_qpos": [8维关节/夹爪]}}}。
两个时刻均为官方 elapsed_steps 的步前值；恢复必须落在现场首个执行步加 16k。
normal 仅记录，hold_extended 仅替换接管区间内送入环境的动作。
oracle_query 消费 Oracle 动作表，独立按4步询问并丢弃模型执行权；
oracle_calibrate 消费固定选点，交还前仅提交真实历史，交还后按16步执行新推理。
speed_donor 录速度供体：reset 后、第一步前核对身份并改写方块单程步数 move_interval 与超时用 stop_time，
推理照常按16步节奏请求（服务端照常编码 motion 窗口），送入环境的动作换成接收局回放表，
从 t_hold 起固定为 hold_qpos 悬停，env_step_after 达到 horizon 时由运行器正常收尾。配置每局为
{"env_seed", "move_interval_override", "stop_time_override", "first_env_step", "actions", "t_hold",
"hold_qpos", "horizon", "orig_move_interval", "target_pass"}；actions[i] 是步前值 first_env_step+i 那一步的动作，
共 t_hold-first_env_step 行，hold_qpos 等于末行。
normal 可选逐局「关超时」条目 {"env_seed", "orig_move_interval", "target_pass", "stop_time_override", "horizon"}：
reset 后、第一步前核对种子与原始 move_interval／stop_time，只把超时用 stop_time 改写为 stop_time_override
（move_interval 不改，官方成功窗口 stop_time_range 在初始化时已算好、不受影响），env_step_after 达到 horizon 时
由运行器正常收尾（终局值 timeout_off_horizon），环境自行终局照常结束；回执写 meta/timeout_off_receipt.json。
normal 表里没有该局条目时与官方循环逐位一致。
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
STOPCUBE_MODES = ("normal", "hold_extended", *ORACLE_MODES, "speed_donor")
# 速度供体只允许官方三种单程步数
SPEED_INTERVALS = (60, 80, 120)
SPEED_DONOR_KEYS = ("env_seed", "move_interval_override", "stop_time_override", "first_env_step", "actions",
                    "t_hold", "hold_qpos", "horizon", "orig_move_interval", "target_pass")
# 运行器在 horizon 主动收尾时的终局值，不属于官方 success/fail/timeout，也不是 infra 故障
SPEED_DONOR_HORIZON = "speed_donor_horizon"
# normal 关超时条目（干预回合按新速度按下不被原超时截断）
TIMEOUT_OFF_KEYS = ("env_seed", "orig_move_interval", "target_pass", "stop_time_override", "horizon")
TIMEOUT_OFF_HORIZON = "timeout_off_horizon"
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
        self.donor = config.get(self.key) if self.mode == "speed_donor" else None
        if self.mode == "speed_donor" and self.donor is None:
            raise official.ContractError(f"速度供体表缺局 {self.key}")
        self.donor_receipt = None
        # normal 模式下表里有该局条目才关超时；没有条目（含空表）时与官方循环逐位一致
        self.timeout_off = config.get(self.key) if self.mode == "normal" else None
        self.timeout_receipt = None
        self.horizon_reached = False
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
        if self.donor is not None:
            self.apply_speed_donor(first)
        if self.timeout_off is not None:
            self.apply_timeout_off(first)
        return result

    def apply_timeout_off(self, first):
        """第一步之前核对接收局身份与原始参数，只改写超时用 stop_time；move_interval 与其他物理状态不动。"""
        table = self.timeout_off
        missing = [key for key in TIMEOUT_OFF_KEYS if key not in table]
        if missing:
            raise official.ContractError(f"关超时条目缺字段 {missing}")
        if any(type(table[key]) is not int for key in TIMEOUT_OFF_KEYS):
            raise official.ContractError("关超时条目的种子、步数与次数字段须为整数")
        u = self.env.unwrapped
        if table["env_seed"] != self.env_seed:
            raise official.ContractError(f"关超时条目种子 {table['env_seed']} 与现场 {self.env_seed} 不符")
        move_interval, stop_time = scalar_value(u.move_interval), scalar_value(u.stop_time)
        if table["orig_move_interval"] != move_interval or table["target_pass"] != stop_time:
            raise official.ContractError("关超时条目的原始单程步数或目标经过次数与现场不符")
        override, horizon = table["stop_time_override"], table["horizon"]
        if (first["env_step"] is None or horizon <= first["env_step"] or override <= 0
                or override * move_interval < horizon):
            raise official.ContractError("关超时条目须 horizon 晚于现场首步，且 stop_time_override×M 不早于 horizon")
        u.stop_time = override
        self.timeout_receipt = {
            "schema_version": 1, "episode_key": self.key, "env_seed": self.env_seed,
            "move_interval": move_interval, "orig_stop_time": stop_time, "stop_time_applied": scalar_value(u.stop_time),
            "target_pass": table["target_pass"], "horizon": horizon, "first_env_step": first["env_step"],
            "button_ever_pressed": False, "first_button_step": None, "last_env_step": first["env_step"],
            "termination": None}

    def horizon_receipt(self):
        """horizon 收尾所用回执：速度供体或关超时，二者互斥。"""
        return self.donor_receipt if self.donor_receipt is not None else self.timeout_receipt

    def horizon_limit(self):
        return (self.donor or self.timeout_off)["horizon"]

    def apply_speed_donor(self, first):
        """第一步之前核对接收局身份与原始参数，再改写方块速度与超时；不改任务类与其他物理状态。"""
        table = self.donor
        missing = [key for key in SPEED_DONOR_KEYS if key not in table]
        if missing:
            raise official.ContractError(f"速度供体表缺字段 {missing}")
        u = self.env.unwrapped
        ints = [table[key] for key in SPEED_DONOR_KEYS if key not in ("actions", "hold_qpos")]
        if any(type(value) is not int for value in ints):
            raise official.ContractError("速度供体表的种子、步数与次数字段须为整数")
        if table["env_seed"] != self.env_seed:
            raise official.ContractError(f"速度供体表种子 {table['env_seed']} 与现场 {self.env_seed} 不符")
        if table["orig_move_interval"] != u.move_interval or table["target_pass"] != u.stop_time:
            raise official.ContractError("速度供体表的原始单程步数或目标经过次数与现场不符")
        actions = np.asarray(table["actions"], dtype=np.float64)
        qpos = np.asarray(table["hold_qpos"], dtype=np.float64)
        first_step, t_hold, horizon = table["first_env_step"], table["t_hold"], table["horizon"]
        override, stop_time = table["move_interval_override"], table["stop_time_override"]
        if (first_step != first["env_step"] or t_hold < 2 or t_hold <= first_step
                or actions.ndim != 2 or actions.shape != (t_hold - first_step, 8) or not np.isfinite(actions).all()
                or qpos.shape != (8,) or not np.isfinite(qpos).all() or not np.array_equal(qpos, actions[-1])
                or horizon <= first_step or override not in SPEED_INTERVALS
                or stop_time <= 0 or override * stop_time < horizon):
            raise official.ContractError("速度供体表须从现场首步起连续到 t_hold-1、有限 8 维且 hold_qpos 为末行，"
                                         "改写后的超时不得早于 horizon")
        orig_stop_time = scalar_value(u.stop_time)
        u.move_interval = override
        u.stop_time = stop_time
        self.donor_actions, self.donor_hold = actions, qpos
        self.donor_receipt = {
            "schema_version": 1, "episode_key": self.key, "env_seed": self.env_seed,
            "orig_move_interval": table["orig_move_interval"], "move_interval_applied": scalar_value(u.move_interval),
            "orig_stop_time": orig_stop_time, "stop_time_applied": scalar_value(u.stop_time),
            "target_pass": table["target_pass"], "first_env_step": first_step, "t_hold": t_hold, "horizon": horizon,
            "cube_half_size": self.receipt["extra"]["cube_half_size"], "button_ever_pressed": False,
            "first_button_step": None, "last_env_step": first["env_step"], "termination": None}

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
        if self.donor is not None:
            index = elapsed - self.donor["first_env_step"]
            if int(index) != index or index < 0:
                raise official.ContractError(f"速度供体回放步号越界 env_step={elapsed}")
            index = int(index)
            executed = (self.donor_actions[index] if index < len(self.donor_actions) else self.donor_hold).copy()
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
        tracked = self.horizon_receipt()
        if tracked is not None:
            tracked["last_env_step"] = after["env_step"]
            if after["button_pressed"] and not tracked["button_ever_pressed"]:
                tracked.update(button_ever_pressed=True, first_button_step=after["env_step"])
            self.horizon_reached = after["env_step"] >= self.horizon_limit()
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
        if self.donor_receipt is not None:
            self.donor_receipt.update(outcome=outcome, actions_executed=self.action_index)
            official.write_record(self.trace_dir / "meta" / "speed_donor_receipt.json", self.donor_receipt)
        if self.timeout_receipt is not None:
            self.timeout_receipt.update(outcome=outcome, actions_executed=self.action_index)
            official.write_record(self.trace_dir / "meta" / "timeout_off_receipt.json", self.timeout_receipt)


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
            elif self.args.stopcube_mode == "speed_donor":
                outcome = self.eval_speed_donor(env_runner, video_save_dir, client, episode_nonce)
            elif env_runner.timeout_off is not None:
                outcome = self.eval_speed_donor(env_runner, video_save_dir, client, episode_nonce,
                                                horizon_outcome=TIMEOUT_OFF_HORIZON)
            else:
                outcome = super().eval_each_episode(env_runner, video_save_dir, client, episode_nonce)
            return outcome
        finally:
            env_runner.finish_trace(outcome, video_save_dir)

    def eval_speed_donor(self, env_runner, video_save_dir, client, episode_nonce, horizon_outcome=SPEED_DONOR_HORIZON):
        """官方循环逐句照搬（推理节奏、续包、终局优先级不变），只多一个 horizon 收尾；
        速度供体送入环境的动作由 TraceRunner.step 替换；normal 关超时条目共用本循环、动作不替换。"""
        self.episode_nonce = episode_nonce
        self.batch_seq = self.replan_seq = self.next_global_frame = 0
        rpc = {"episode_nonce": episode_nonce, "expected_run_identity_sha256": self.args.run_identity_sha256,
               "expected_server_nonce": self.args.expected_server_nonce}
        if self.args.profile == "plfull":
            rpc["episode_key"] = self.episode_key
        official.require_ack(client.request({"reset": True, "eval_rpc": rpc}), "reset_finished", rpc)
        state = official.EpisodeState()
        prompt, self.recorder = self.init_episode(env_runner, state, video_save_dir)
        img, wrist, robot_state = state.get_current_obs()
        receipt = env_runner.horizon_receipt()
        while True:
            if not state.action_plan:
                state.action_plan.extend(self.get_action_chunk(client, state, img, wrist, robot_state, prompt, 16))
                state.clear_buffers()
            action = state.action_plan.popleft()
            self.step_in_flight = True
            obs, stop, outcome = env_runner.step(action)
            state.count += 1
            self.actions_executed = state.count
            self.terminal_observed = stop and outcome in ("success", "fail", "timeout")
            if outcome == "error" or obs is None or any(item is None for item in obs):
                raise RuntimeError("环境 step 未返回完整观测")
            self.step_in_flight = False
            if state.count > self.args.max_steps:
                self.terminal_observed = True
                receipt["termination"] = "max_steps"
                return "timeout"
            if stop:
                self.terminal_observed = outcome in ("success", "fail", "timeout")
            img, wrist, robot_state = obs
            state.add_observation(img, wrist, robot_state)
            self.record_frame(image=img.copy(), wrist_image=wrist.copy(), state=robot_state.copy(),
                              action=np.asarray(action).copy())
            if stop:
                if not self.terminal_observed:
                    raise official.ContractError(f"环境终止但没有正常终局状态：{outcome}")
                receipt["termination"] = f"env_terminal:{outcome}"
                return outcome
            if env_runner.horizon_reached:
                self.terminal_observed = True
                receipt["termination"] = "horizon"
                return horizon_outcome

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
    if args.stopcube_mode not in STOPCUBE_MODES or not args.stopcube_config:
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
