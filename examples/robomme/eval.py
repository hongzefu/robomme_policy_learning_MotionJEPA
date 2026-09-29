import dataclasses
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Any, Tuple

import numpy as np

from openpi_client import websocket_client_policy as _websocket_client_policy
from utils import (
    pack_buffer,
    check_args,
    TASK_NAME_LIST,
    TASK_WITH_VIDEO_DEMO,
    SUBGOAL_TYPES,
    EpisodeState,
)
from utils import RolloutRecorder
from env_runner import EnvRunner
import xhard0_manifest
from subgoal_predictor import build_subgoal_predictor, SubgoalPredictorBase

# qwen3-vl environment variables
os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '64'
os.environ['FPS_MAX_FRAMES'] = '10'



@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8011

    obs_horizon: int = 16
    max_steps: int = 1300
    save_dir: str = "runs/evaluation"
    overwrite: bool = False

    use_history: bool = True
    policy_name: str = "dummy_test"
    model_seed: int = 42
    model_ckpt_id: int = 80000

    # 官方路线 xhard0：给了清单就只评本片 (task, source_episode)，逐局写 episodes.jsonl；官方全量循环不动
    episode_manifest: str = ""  # jsonl，每行 {task, source_episode, seed, shard}
    shard: int = -1
    video_dir: str = ""  # 视频写到这里，文件名 {task}_xhard0_{source_episode}_{seed}.mp4

    # task control
    re_eval_tasks: str = "" # tasks split by comma
    only_tasks: str = "" # tasks split by comma
    exclude_tasks: str = "" # tasks split by comma

    # VLM subgoal predictor
    use_oracle: bool = False
    use_qwenvl: bool = False
    use_memer: bool = False
    use_gemini: bool = False
    subgoal_type: Optional[str] = None  # [simple_subgoal, grounded_subgoal]
    gemini_model_name: str = "gemini-2.5-pro"
    qwenvl_simpleSG_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/qwenvl/simple_subgoal/checkpoint-1400"
    qwenvl_groundSG_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200"
    memer_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/memer/grounded_subgoal/checkpoint-1300"
    subgoal_keep_period: int = 1 # ever subgoal should be kept for this many steps
    # this can accelerate the evaluation process for symbolic memory
    # In our experiments, we just set this to 1



class EpisodeEvaluator:
    def __init__(self, args: Args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir

    def eval_each_episode(
        self,
        env_runner: EnvRunner,
        subgoal_predictor: SubgoalPredictorBase,
        video_save_dir: Path,
    ) -> str:
        client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(
            self.args.host, self.args.port
        )
        resp = client.reset()
        while not resp.get("reset_finished", False):
            time.sleep(0.1)

        epstate = EpisodeState()
        task_goal, recorder = self.init_episode(env_runner, epstate, video_save_dir)
        subgoal_predictor.start_episode(epstate, env_runner)        

        img, wrist_img, robot_state = epstate.get_current_obs()
        prompt = task_goal
        success_flag = "unknown"
        subgoal = None
        last_subgoal = None

        while True:
            subgoal_predictor.step(epstate)

            if not epstate.action_plan:
                if epstate.count % self.args.subgoal_keep_period == 0 or last_subgoal is None:
                    subgoal, has_api_error = subgoal_predictor.get_subgoal(
                        epstate.count,
                        subgoal,
                        last_subgoal,
                    )
                else:
                    subgoal = last_subgoal
                    has_api_error = False

                if has_api_error:
                    break

                action_chunk = self.get_action_chunk(
                    client, epstate, img, wrist_img, robot_state, prompt, subgoal, 
                    exec_horizon=self.args.obs_horizon
                )

                epstate.action_plan.extend(action_chunk)
                epstate.clear_buffers()

                last_subgoal = subgoal

            action = epstate.action_plan.popleft()
            obs, stop_flag, success_flag = env_runner.step(action)
            epstate.count += 1

            self.last_steps = epstate.count  # 清单模式逐局记录用
            if epstate.count > self.args.max_steps:
                success_flag = "timeout"
                break

            img, wrist_img, robot_state = obs

            epstate.add_observation(img, wrist_img, robot_state)
            recorder.record(
                image=img.copy(),
                wrist_image=wrist_img.copy(),
                state=robot_state.copy(),
                action=action.copy(),
                subgoal=subgoal,
            )

            if stop_flag:
                break

        if success_flag == "unknown":
            return "unknown"

        if self.args.episode_manifest:  # 清单模式：视频由 evaluate_manifest() 在记终态后写
            subgoal_predictor.end_episode(epstate, success_flag)
            return success_flag

        video_filename = f"{env_runner.env_id}_ep{env_runner.episode_id}_{success_flag}_{task_goal}_{env_runner.difficulty}.mp4"
        recorder.save_video(video_filename)

        subgoal_predictor.end_episode(epstate, success_flag)
        return success_flag


    def init_episode(
        self,
        env_runner: EnvRunner,
        epstate: EpisodeState,
        video_save_dir: Path,
    ) -> Tuple[str, RolloutRecorder]:
        pre_traj = env_runner.get_init_obs()
        task_goal = pre_traj["task_goal"]

        recorder = RolloutRecorder(video_save_dir, task_goal, fps=30)
        self.last_recorder = recorder

        print(f"task_goal: {task_goal}")

        epstate.image_buffer.extend(pre_traj["images"])
        epstate.wrist_image_buffer.extend(pre_traj["wrist_images"])
        epstate.state_buffer.extend(pre_traj["states"])

        for i in range(len(pre_traj["images"])):
            recorder.record(
                image=pre_traj["images"][i].copy(),
                wrist_image=pre_traj["wrist_images"][i].copy(),
                state=pre_traj["states"][i].copy(),
                is_video_demo=env_runner.env_id in TASK_WITH_VIDEO_DEMO and i < len(pre_traj["images"]) - 1,
                subgoal=None if self.args.subgoal_type is None else "[initializing...]",
            )

        epstate.exec_start_idx = len(epstate.image_buffer) - 1
        print(f"exec_start_idx: {epstate.exec_start_idx}")
        return task_goal, recorder

    def get_action_chunk(
        self,
        client,
        state: EpisodeState,
        img: np.ndarray,
        wrist_img: np.ndarray,
        robot_state: np.ndarray,
        prompt: str,
        subgoal: Optional[str],
        exec_horizon: int,
    ) -> list:
        if self.args.use_history:
            resp = client.add_buffer(pack_buffer(
                state.image_buffer,
                state.state_buffer,
                state.exec_start_idx,
            ))
            while not resp.get("add_buffer_finished", False):
                time.sleep(0.1)

        element = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": robot_state,
            "prompt": prompt,
        }

        if subgoal is not None:
            element['simple_subgoal'] = subgoal
            element['grounded_subgoal'] = subgoal

        action_chunk = client.infer(element)["actions"]
        return action_chunk[:exec_horizon]


def setup_save_directory(args: Args) -> Path:
    """Set up and validate save directories."""
    save_dir = (
        Path(args.save_dir)
        / args.policy_name
        / f"ckpt{args.model_ckpt_id}"
        / f"seed{args.model_seed}"
    )

    if args.subgoal_type in SUBGOAL_TYPES:
        if args.use_gemini:
            save_dir = save_dir / "gemini"
        elif args.use_qwenvl:
            save_dir = save_dir / "qwenvl"
        elif args.use_memer:
            save_dir = save_dir / "memer"
        else:
            save_dir = save_dir / "oracle"

    if save_dir.exists():
        if args.overwrite:
            shutil.rmtree(save_dir)
            print(f"we will overwrite the evaluation at {save_dir}")
        else:
            print("we will resume the evaluation")

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def setup_log_dict(save_dir: Path, args: Args) -> dict:
    if os.path.exists(save_dir / "progress.json"):
        with open(save_dir / "progress.json", "r") as f:
            log_dict = json.load(f)

    elif os.path.exists(save_dir / "log.json"):
        with open(save_dir / "log.json", "r") as f:
            log_dict = json.load(f)
        log_dict.pop("success_rate", None)
        log_dict.pop("total_success_rate", None)
    else:
        log_dict = {}

    for task_name in log_dict:
        error_list = []
        for k, v in log_dict[task_name].items():
            if v == "error":
                error_list.append(k)
        for k in error_list:
            log_dict[task_name].pop(k)

    if args.re_eval_tasks:
        for task_name in args.re_eval_tasks.split(","):
            if task_name in log_dict:
                del log_dict[task_name]
                os.system(f"rm -f {save_dir / 'videos' / f'{task_name}_ep*.mp4'}")

    with open(save_dir / "progress.json", "w") as f:
        json.dump(log_dict, f, indent=2)

    return log_dict


def evaluate_manifest(args: Args):
    """官方路线 xhard0 清单评估：只评 --args.shard 选中的 (task, source_episode)，每局一条 episodes.jsonl 记录。"""
    import sys
    import robomme
    check_args(args)
    # 官方路线不得引入 robomme_hard（test-hard 覆盖层）
    assert "robomme_hard" not in sys.modules, "官方路线不得 import robomme_hard"
    print(f"[robomme] robomme={robomme.__file__}")
    if args.shard < 0:
        raise ValueError("--args.episode_manifest 需同时给 --args.shard (>=0)")
    if not args.video_dir:
        raise ValueError("--args.episode_manifest 需同时给 --args.video_dir")
    save_dir = setup_save_directory(args)
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    by_task = xhard0_manifest.load_rows(args.episode_manifest, args.shard, TASK_NAME_LIST)
    if args.only_tasks:
        by_task = {t: v for t, v in by_task.items() if t in args.only_tasks.split(",")}
    print(f"[robomme] manifest={args.episode_manifest} shard={args.shard} episodes={sum(map(len, by_task.values()))}")
    subgoal_predictor = build_subgoal_predictor(args, save_dir)
    evaluator = EpisodeEvaluator(args, save_dir)
    ep_log = save_dir / "episodes.jsonl"
    done, rows = xhard0_manifest.done_keys(ep_log)
    for task_name, task_rows in by_task.items():
        env_runner = EnvRunner(task_name, save_dir / "videos", max_steps=args.max_steps)
        for row in task_rows:
            src_ep, seed = int(row["source_episode"]), int(row["seed"])
            if (task_name, src_ep) in done:
                print(f"[robomme] {task_name} source_episode {src_ep} already evaluated, skipping...")
                continue
            builder_seed, difficulty = env_runner.env_builder.resolve_episode(src_ep)
            xhard0_manifest.check_seed(row, builder_seed, difficulty)  # 不符即中止整片
            evaluator.last_recorder, evaluator.last_steps, error = None, 0, None
            env_runner.make_env(src_ep)
            print(f"\n[robomme] env for task {task_name} source_episode {src_ep} seed {seed} setup finished")
            try:
                success_flag = evaluator.eval_each_episode(env_runner, subgoal_predictor, save_dir / "videos")
            except Exception as e:
                print(f"Error evaluating source_episode {src_ep} for task {task_name}: {e}")
                success_flag, error = "error", f"{type(e).__name__}: {e}"  # 不沿用上一局的终态（同 7fc7128）
            status = success_flag if success_flag in xhard0_manifest.NORMAL else "error"
            if status == "error" and error is None:
                error = f"success_flag={success_flag}"
            attempt = sum((r["task"], int(r["source_episode"])) == (task_name, src_ep) for r in rows) + 1
            video = None
            recorder = evaluator.last_recorder
            if recorder is not None and recorder.total_images:
                # 终态之后写视频；error 局加后缀保留，不覆盖、不删除任何视频
                name = xhard0_manifest.video_name(task_name, src_ep, seed)
                if status == "error":
                    name = name[:-len(".mp4")] + f".error{attempt}.mp4"
                try:
                    recorder.save_dir = video_dir
                    recorder.save_video(name)
                    video = str(video_dir / name)
                except Exception as e:
                    print(f"Error saving video for {task_name} source_episode {src_ep}: {e}")
            rec = dict(task=task_name, source_episode=src_ep, seed=seed, status=status, task_success=status == "success",
                       steps=evaluator.last_steps, error=error, max_steps=args.max_steps, attempt=attempt,
                       shard=args.shard, video=video)
            with open(ep_log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            env_runner.close_env()
        del env_runner
        time.sleep(1)


def evaluate(args: Args):
    """Main evaluation function."""
    if args.episode_manifest:
        return evaluate_manifest(args)
    check_args(args)

    save_dir = setup_save_directory(args)
    video_save_dir = save_dir / "videos"

    log_dict = setup_log_dict(save_dir, args)

    if args.only_tasks:
        task_names = args.only_tasks.split(",")
    else:
        task_names = TASK_NAME_LIST

    if args.exclude_tasks:
        task_names = [task_name for task_name in task_names if task_name not in args.exclude_tasks.split(",")]
        for task in args.exclude_tasks.split(","):
            log_dict[task] = {str(i): False for i in range(50)}

    subgoal_predictor = build_subgoal_predictor(args, save_dir)
    evaluator = EpisodeEvaluator(args, save_dir)

    while not os.path.exists(save_dir / "log.json"):
        for task_name in task_names:
            if task_name not in log_dict:
                log_dict[task_name] = {}

            env_runner = EnvRunner(task_name, video_save_dir, max_steps=args.max_steps)
            num_episodes = env_runner.num_episodes

            success_flag = "unknown"

            for episode_id in range(num_episodes):
                if str(episode_id) in log_dict[task_name]:
                    print(f"[robomme] episode {episode_id} already evaluated, skipping...")
                    continue

                env_runner.make_env(episode_id)
                print(f"\n[robomme] env for task {task_name} episode {episode_id} setup finished")

                try:
                    success_flag = evaluator.eval_each_episode(env_runner, subgoal_predictor, video_save_dir)
                    if success_flag == "unknown":
                        log_dict[task_name][episode_id] = "error"
                    else:
                        log_dict[task_name][episode_id] = success_flag == "success"
                except Exception as e:
                    print(f"Error evaluating episode {episode_id} for task {task_name}: {e}")
                    log_dict[task_name][episode_id] = "error"

                env_runner.close_env()
                with open(save_dir / "progress.json", "w") as f:
                    json.dump(log_dict, f, indent=2)

                if success_flag == "unknown":
                    print("API calling error, aborting...")
                    return

            del env_runner
            time.sleep(1)

        try:
            final_results = {}
            final_results["success_rate"] = {
                task_name: sum(log_dict[task_name].values()) / len(log_dict[task_name].values())
                for task_name in log_dict.keys()
            }
            final_results["total_success_rate"] = (
                sum(final_results["success_rate"].values()) / len(final_results["success_rate"].values())
            )
            with open(save_dir / "log.json", "w") as f:
                json.dump(final_results, f, indent=2)
        except Exception as e:
            print(f"Error saving final results: {e}")
            time.sleep(1)


if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate)
