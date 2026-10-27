"""官方单局语义与有界、不可重发的评估协议客户端。"""

import dataclasses
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

try:
    from .utils import EpisodeState, RolloutRecorder, TASK_NAME_LIST, TASK_WITH_VIDEO_DEMO, pack_buffer
    from .env_runner import EnvRunner
except ImportError:
    from utils import EpisodeState, RolloutRecorder, TASK_NAME_LIST, TASK_WITH_VIDEO_DEMO, pack_buffer
    from env_runner import EnvRunner


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8011
    episode_manifest: str = ""
    chunk: str = ""
    attempt_output: str = ""
    run_identity_sha256: str = ""
    expected_server_nonce: str = ""
    model_kind: str = ""
    model_seed: int = 42
    rpc_timeout_seconds: float = 600.0
    drain_file: str = ""
    obs_horizon: int = 16
    max_steps: int = 1300
    dataset: str = "test"
    use_history: bool = True


class UncertainRPC(RuntimeError):
    """请求可能已改变服务端状态，整局必须中止且禁止原局重发。"""


class ContractError(ValueError):
    """确定性合同错误，由控制器阻断而非自动重试。"""


class EvaluationClient:
    """一个 chunk 共用一条同步连接，每次发送只等待一次有界响应。"""

    def __init__(self, args: Args):
        import websockets.sync.client
        from openpi_client import msgpack_numpy

        self.timeout = args.rpc_timeout_seconds
        self.codec = msgpack_numpy
        self.packer = msgpack_numpy.Packer()
        self.connection = websockets.sync.client.connect(
            f"ws://{args.host}:{args.port}", compression=None, max_size=None,
            ping_interval=None,
            open_timeout=self.timeout, close_timeout=min(self.timeout, 10),
        )
        self.ws = self.connection.__enter__()
        try:
            self.metadata = self._receive()
            validate_metadata(self.metadata, args)
        except BaseException:
            self.close()
            raise

    def _receive(self):
        response = self.ws.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise UncertainRPC(f"服务端返回未分类错误：{response}")
        result = self.codec.unpackb(response)
        if not isinstance(result, dict):
            raise ContractError("服务端响应必须是字典")
        if result.get("error") in ("contract_violation", "active_attempt"):
            raise ContractError(f"服务端确定性拒绝：{result}")
        if result.get("error"):
            raise UncertainRPC(f"服务端执行失败或响应不确定：{result}")
        return result

    def request(self, payload):
        try:
            self.ws.send(self.packer.pack(payload))
            return self._receive()
        except ContractError:
            raise
        except Exception as exc:
            raise UncertainRPC("RPC 响应不确定，禁止在原局重发") from exc

    def close(self):
        self.connection.__exit__(None, None, None)


def validate_metadata(metadata, args):
    expected = {"run_identity_sha256": args.run_identity_sha256, "server_nonce": args.expected_server_nonce,
                "model_seed": args.model_seed, "model_kind": args.model_kind}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ContractError(f"服务端身份不符：{key}")


def validate_budget(demo_frames):
    k_demo = max(0, (demo_frames - 17) // 32 + 1)
    if demo_frames + 1301 >= 4096 or k_demo + 40 > 96:
        raise ContractError(f"零截断预算越界：D={demo_frames} k_demo={k_demo} k_exec=40")
    return {"demo_frames": demo_frames, "demo_windows": k_demo, "conservative_exec_windows": 40}


def require_ack(response, field, rpc):
    if response.get(field) is not True or response.get("eval_rpc") != rpc:
        raise UncertainRPC(f"{field} 未获得匹配的确定 ACK")


class EpisodeEvaluator:
    def __init__(self, args: Args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir
        self.actions_executed = 0
        self.terminal_observed = False
        self.step_in_flight = False
        self.recorder = None
        self.budget = None
        self.artifact_errors = []

    def record_frame(self, **frame):
        if self.recorder is not None:
            try:
                self.recorder.record(**frame)
            except Exception:
                self.artifact_errors.append(traceback.format_exc())
                self.recorder = None

    def init_episode(self, env_runner, epstate, video_save_dir):
        pre_traj = env_runner.get_init_obs()
        images, wrists, states = (pre_traj[key] for key in ("images", "wrist_images", "states"))
        if not images or len(images) != len(wrists) or len(images) != len(states):
            raise ContractError("reset 观测必须含完整且对齐的 demo 与 exec0")
        epstate.image_buffer.extend(images)
        epstate.wrist_image_buffer.extend(wrists)
        epstate.state_buffer.extend(states)
        epstate.exec_start_idx = len(images) - 1
        self.budget = validate_budget(epstate.exec_start_idx)
        try:
            pack_buffer(images, states, epstate.exec_start_idx)
        except ValueError as exc:
            raise ContractError("reset 历史图像或状态不符合输入合同") from exc
        try:
            self.recorder = RolloutRecorder(video_save_dir, pre_traj["task_goal"], fps=30)
        except Exception:
            self.artifact_errors.append(traceback.format_exc())
        for index, (image, wrist, state) in enumerate(zip(images, wrists, states)):
            self.record_frame(image=image.copy(), wrist_image=wrist.copy(), state=state.copy(),
                            is_video_demo=env_runner.env_id in TASK_WITH_VIDEO_DEMO and index < len(images) - 1)
        return pre_traj["task_goal"], self.recorder

    def get_action_chunk(self, client, state, img, wrist_img, robot_state, prompt, exec_horizon):
        from scripts.evaluation.runtime_contract import payload_sha256

        try:
            payload = pack_buffer(state.image_buffer, state.state_buffer, state.exec_start_idx)
        except ValueError as exc:
            raise ContractError("历史续包图像或状态不符合输入合同") from exc
        frame_count = len(state.image_buffer)
        if self.batch_seq == 0 and frame_count != state.exec_start_idx + 1:
            raise ContractError("首包必须完整包含 D+1 帧")
        if self.batch_seq > 0 and frame_count != 16:
            raise ContractError("官方续包必须恰好包含 16 帧")
        rpc = {"episode_nonce": self.episode_nonce, "batch_seq": self.batch_seq,
               "first_global_frame": self.next_global_frame, "frame_count": frame_count,
               "payload_sha256": payload_sha256(payload)}
        payload["eval_rpc"] = rpc
        require_ack(client.request(payload), "add_buffer_finished", rpc)
        self.batch_seq += 1
        self.next_global_frame += frame_count
        infer_rpc = {"episode_nonce": self.episode_nonce, "replan_seq": self.replan_seq,
                     "last_global_frame": self.next_global_frame - 1}
        response = client.request({"observation/image": img, "observation/wrist_image": wrist_img,
                                   "observation/state": robot_state, "prompt": prompt, "eval_rpc": infer_rpc})
        if response.get("eval_rpc") != infer_rpc:
            raise UncertainRPC("infer 响应身份或序号不符")
        actions = np.asarray(response.get("actions"))
        if actions.shape != (20, 8) or not np.issubdtype(actions.dtype, np.number) or not np.isfinite(actions).all():
            raise ContractError("policy 必须返回有限的 (20,8) 动作")
        self.replan_seq += 1
        return actions[:exec_horizon]

    def eval_each_episode(self, env_runner, video_save_dir, client, episode_nonce):
        self.episode_nonce = episode_nonce
        self.batch_seq = self.replan_seq = self.next_global_frame = 0
        rpc = {"episode_nonce": episode_nonce, "expected_run_identity_sha256": self.args.run_identity_sha256,
               "expected_server_nonce": self.args.expected_server_nonce}
        require_ack(client.request({"reset": True, "eval_rpc": rpc}), "reset_finished", rpc)
        state = EpisodeState()
        prompt, self.recorder = self.init_episode(env_runner, state, video_save_dir)
        img, wrist, robot_state = state.get_current_obs()
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
            # 错误先保留；正常路径严格保持官方 count++ 后 >1300 的优先级。
            if outcome == "error" or obs is None or any(item is None for item in obs):
                raise RuntimeError("环境 step 未返回完整观测")
            self.step_in_flight = False
            if state.count > self.args.max_steps:
                self.terminal_observed = True
                return "timeout"
            if stop:
                self.terminal_observed = outcome in ("success", "fail", "timeout")
            img, wrist, robot_state = obs
            state.add_observation(img, wrist, robot_state)
            self.record_frame(image=img.copy(), wrist_image=wrist.copy(), state=robot_state.copy(), action=action.copy())
            if stop:
                if not self.terminal_observed:
                    raise ContractError(f"环境终止但没有正常终局状态：{outcome}")
                return outcome


def write_record(path, record):
    """同目录临时文件同步后发布；既有收据一律拒绝覆盖。"""
    path = Path(path)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)


def load_chunk(args):
    with Path(args.episode_manifest).open() as stream:
        manifest = json.load(stream)
    if "chunks" in manifest:
        from scripts.evaluation.manifest import validate_episodes
        validate_episodes(manifest)
        matches = [chunk for chunk in manifest["chunks"] if str(chunk["chunk_id"]) == args.chunk]
        if len(matches) != 1:
            raise ContractError("chunk 必须在清单中唯一存在")
        by_key = {episode["episode_key"]: episode for episode in manifest["episodes"]}
        episodes = [by_key[key] for key in matches[0]["episode_keys"]]
    else:
        if manifest.get("chunk_id") != args.chunk or manifest.get("run_identity_sha256") != args.run_identity_sha256:
            raise ContractError("grant 的 chunk／run 身份不符")
        episodes = manifest["episodes"]
    if not 1 <= len(episodes) <= 5:
        raise ContractError("每个 chunk 必须包含 1 至 5 局")
    keys = set()
    for episode in episodes:
        key = f"{episode['task_name']}:{episode['episode_id']}"
        if (episode["task_name"] not in TASK_NAME_LIST or type(episode["episode_id"]) is not int
                or not 0 <= episode["episode_id"] < 50 or key in keys or episode.get("episode_key") != key
                or type(episode.get("env_seed")) is not int or not isinstance(episode.get("difficulty"), str)
                or not episode["difficulty"]):
            raise ContractError("chunk 含非法或重复官方 episode")
        keys.add(key)
    return episodes


def evaluate(args: Args, *, runner_factory=EnvRunner, client_factory=EvaluationClient):
    from scripts.evaluation.runtime_contract import require_sha256, sha256_file

    if (args.dataset != "test" or args.max_steps != 1300 or args.obs_horizon != 16 or not args.use_history
            or args.rpc_timeout_seconds <= 0 or args.model_seed != 42):
        raise ContractError("官方评估固定 test、1300、16 步和历史模式，RPC 超时须为正")
    require_sha256(args.run_identity_sha256)
    if not all((args.episode_manifest, args.chunk, args.attempt_output, args.run_identity_sha256,
                args.expected_server_nonce, args.model_kind)):
        raise ContractError("必须显式提供清单、chunk、独立 attempt 输出及服务身份")
    episodes = load_chunk(args)
    output = Path(args.attempt_output)
    output.mkdir(parents=True, exist_ok=False)
    video_dir = output / "videos"
    client = None
    evaluator = None
    completed, started = [], []
    exit_code = 0
    connection_certain = True
    release_nonce = uuid.uuid4().hex
    try:
        client = client_factory(args)
        validate_metadata(client.metadata, args)
        for episode in episodes:
            if args.drain_file and Path(args.drain_file).exists():
                exit_code = 3
                break
            key = f"{episode['task_name']}:{episode['episode_id']}"
            started.append(key)
            nonce = uuid.uuid4().hex
            identity = dict(episode, episode_key=key, episode_nonce=nonce,
                            run_identity_sha256=args.run_identity_sha256, chunk_id=args.chunk)
            runner = None
            evaluator = EpisodeEvaluator(args, output)
            began = time.monotonic()
            terminal_saved = False
            try:
                write_record(output / f"{key}.started.json", dict(identity, record_type="started"))
                runner = runner_factory(episode["task_name"], video_dir, max_steps=1300, dataset="test")
                if runner.num_episodes != 50:
                    raise ContractError("官方 test builder 必须返回 50 局")
                seed, difficulty = runner.env_builder.resolve_episode(episode["episode_id"])
                if seed != episode["env_seed"] or difficulty != episode["difficulty"]:
                    raise ContractError("builder seed／difficulty 与不可变清单不符")
                runner.make_env(episode["episode_id"])
                outcome = evaluator.eval_each_episode(runner, video_dir, client, nonce)
                terminal = dict(identity, record_type="terminal", outcome=outcome,
                                actions_executed=evaluator.actions_executed, wall_seconds=time.monotonic() - began,
                                rpc_seq=1 + evaluator.batch_seq + evaluator.replan_seq, budget=evaluator.budget)
                write_record(output / f"{key}.terminal.json", terminal)
                terminal_saved = True
                completed.append(key)
                try:
                    if evaluator.artifact_errors or evaluator.recorder is None:
                        raise RuntimeError("录制期间产物失败：" + "\n".join(evaluator.artifact_errors))
                    video_name = f"{episode['task_name']}_ep{episode['episode_id']}_{outcome}.mp4"
                    evaluator.recorder.save_video(video_name)
                    video = video_dir / video_name
                    artifact = dict(identity, record_type="artifact", artifact_error=False,
                                    files=[{"path": str(video.relative_to(output)), "bytes": video.stat().st_size,
                                            "sha256": sha256_file(video)}])
                except Exception:
                    artifact = dict(identity, record_type="artifact", artifact_error=True, files=[], error=traceback.format_exc())
                write_record(output / f"{key}.artifact.json", artifact)
            except Exception as exc:
                connection_certain = False
                if terminal_saved:
                    record = dict(identity, record_type="artifact", artifact_error=True, files=[], error=traceback.format_exc())
                    write_record(output / f"{key}.artifact-error.json", record)
                else:
                    record = dict(identity, record_type="infra", error_type=type(exc).__name__,
                                  retryable=not isinstance(exc, ContractError),
                                  uncertain_terminal=evaluator.terminal_observed or evaluator.step_in_flight,
                                  uncertain_rpc=isinstance(exc, UncertainRPC), actions_executed=evaluator.actions_executed,
                                  error=traceback.format_exc())
                    write_record(output / f"{key}.infra.json", record)
                exit_code = 2
            finally:
                if runner is not None:
                    try:
                        runner.close_env()
                    except Exception:
                        write_record(output / f"{key}.close-error.json", dict(identity, record_type="artifact", artifact_error=True, files=[], error=traceback.format_exc()))
                        exit_code = 2
            if exit_code == 2:
                break
    except Exception as exc:
        connection_certain = False
        exit_code = 2
        write_record(output / "client.infra.json", {"record_type": "infra", "error_type": type(exc).__name__,
                     "retryable": not isinstance(exc, ContractError), "error": traceback.format_exc()})
    finally:
        if client is not None:
            try:
                if connection_certain:
                    rpc = {"episode_nonce": getattr(evaluator, "episode_nonce", release_nonce)}
                    response = client.request({"release_attempt": True, "eval_rpc": rpc})
                    require_ack(response, "release_attempt_finished", rpc)
                    if response.get("quiescent") is not True:
                        raise UncertainRPC("服务端未确认静止释放")
            except Exception:
                exit_code = 2
                write_record(output / "release.infra.json", {"record_type": "infra", "traceback": traceback.format_exc()})
            finally:
                try:
                    client.close()
                except Exception:
                    exit_code = 2
                    write_record(output / "client-close.infra.json", {"record_type": "infra", "error": traceback.format_exc()})
        all_keys = [f"{ep['task_name']}:{ep['episode_id']}" for ep in episodes]
        write_record(output / "attempt.json", {"status": {0: "complete", 2: "infra", 3: "drained"}[exit_code],
                     "exit_code": exit_code, "started_episode_keys": started, "completed_episode_keys": completed,
                     "unstarted_episode_keys": [key for key in all_keys if key not in started]})
    return exit_code


if __name__ == "__main__":
    import tyro
    raise SystemExit(evaluate(tyro.cli(Args)))
