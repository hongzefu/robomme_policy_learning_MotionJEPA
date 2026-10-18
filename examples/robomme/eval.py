import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import signal
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
    EpisodeState,
)
from utils import RolloutRecorder, finalize_all
from env_runner import EnvRunner, SpecEnvRunner


class EpisodeWallClockTimeout(RuntimeError):
    """单集墙钟超时；只兜死锁，不参与正常的 max_steps 判定。"""


@contextlib.contextmanager
def episode_deadline(seconds: int):
    """给整集设墙钟，覆盖 make_env 与 reset，不只是 rollout 循环。

    原先只在 rollout 里比时间，管不到建环境与 reset。实测 VideoRepick/medium 有若干集
    卡在 reset 期（s9 的 ep172 静默 36 分钟、s2 的 ep155 静默 12 分钟，都停在
    「setup finished」之后、`exec_start_idx` 之前），只能靠 job 级 EVAL_TIMEOUT 兜底：
    一次浪费两小时，而且该集不会被记成 error，下一块续跑时还会再撞同一条。

    SIGALRM 能打断停在 Python 层的重试循环；若卡在 native 调用内部，信号要等调用返回
    才会递达，那种情况仍由 EVAL_TIMEOUT 兜底。超时抛 EpisodeWallClockTimeout，
    由调用方的 except 记成 "error" 并继续下一集。
    """
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise EpisodeWallClockTimeout(f"单集超过 {seconds}s 墙钟（含建环境与 reset）")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8011

    obs_horizon: int = 16
    # 默认保持 1300：官方 env_metadata 路径（run.sh 的单回合验收）不传这个参数，
    # 改默认值会把 docs/training-doc/vail-eval-gl-20260918T0400Z 的口径悄悄换掉。
    # 清单路径由 run_shard.sh 显式传 --args.max_steps=2000（写死 MAX_STEPS=2000）。
    max_steps: int = 1300
    save_dir: str = "v1-store/evaluation"
    overwrite: bool = False

    use_history: bool = True
    policy_name: str = "dummy_test"
    model_seed: int = 42
    model_ckpt_id: int = 80000

    # task control（官方 env_metadata 路径）
    re_eval_tasks: str = "" # tasks split by comma
    only_tasks: str = "" # tasks split by comma
    exclude_tasks: str = "" # tasks split by comma
    max_episodes: int = 0 # 每任务最多评几集（0 = 环境提供的全部；T3_EVAL_OBS 用 10）

    # 清单路径（1017-test-xhard5-12per 的 1200 局清单）：给了 episode_plan 就走这条，与上面的 task control 互斥
    episode_plan: str = ""      # 分片计划 json；{"shard_tag":..., "manifest":..., "manifest_sha256":..., "groups":{"任务/难度":[episode...]}}
    manifest: str = ""          # 清单 jsonl 路径；留空则用 plan 里记的那个
    max_new_episodes: int = 0   # 本进程最多「新评」几集（0 = 不限）；用于把 make_env 次数压在 Vulkan TLS 红线下
    shard_tag: str = ""         # 分片标识，只写进 plan.json 便于追溯
    episode_wall_s: int = 900   # 单集墙钟上限，超时记 "error"

    # BinFill demo 前缀注入（0922-binfill-demo-prefix-plan.md B 节）：给了路径才启用，默认关闭 ⇒ 旧 run 逐字等价。
    # 只支持 BinFill——其余三任务的 reset() 本来就返回 planner 演示帧，再拼一段会双重叠加。
    demo_prefix_store: str = ""


def motion_stat_of(epstate) -> Optional[dict]:
    """把一集的 motion 窗统计整理成一条记录；该集没跑过 motion 推理则返回 None。

    ``motion_infers == 0`` 有两种成因：非 motion 模型（server 不回传 motion_k），
    或该集在第一次 infer 之前就挂了。两种都不该产出记录——
    check_shard.py 的覆盖判据是「非 error 集都必须有记录」，能把后者揪出来。
    """
    if epstate is None or epstate.motion_infers == 0:
        return None
    return {
        "exec_start_idx": int(epstate.exec_start_idx_initial),
        "steps": int(epstate.count),
        "motion_infers": int(epstate.motion_infers),
        "motion_k_max": int(epstate.motion_k_max),
        "motion_downsample_steps": int(epstate.motion_downsample_steps),
        "motion_budget": int(epstate.motion_budget),
        "motion_overflow": str(epstate.motion_overflow),
    }


class EpisodeEvaluator:
    def __init__(self, args: Args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir
        self._epstate = None          # 当前集的状态，供收尾取 motion 统计（异常路径也取得到）
        self.last_motion_stat = None

    def eval_each_episode(
        self,
        env_runner,
        video_save_dir: Path,
    ) -> str:
        self._epstate = None
        self._recorder = None
        self.last_motion_stat = None
        client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(
            self.args.host, self.args.port
        )
        try:
            return self._rollout(client, env_runner, video_save_dir)
        finally:
            # error / unknown 路径没有 save_video：关管道、丢临时文件、关动作日志；已 save_video 的局这里不动
            if self._recorder is not None:
                self._recorder.discard()
            # 统计必须在 finally 里取：墙钟超时 / step 异常那两条路径同样要留下已跑部分的窗数
            self.last_motion_stat = motion_stat_of(self._epstate)
            # 每集一个新 client，不关会在 server 侧攒下同样多的常驻 handler
            try:
                client._ws.close()
            except Exception as e:
                print(f"关闭 websocket 失败（不影响本集结果）：{e}")

    def _rollout(self, client, env_runner, video_save_dir: Path) -> str:
        resp = client.reset()
        while not resp.get("reset_finished", False):
            time.sleep(0.1)

        epstate = EpisodeState()
        self._epstate = epstate
        task_goal, recorder = self.init_episode(env_runner, epstate, video_save_dir)
        self._recorder = recorder

        img, wrist_img, robot_state = epstate.get_current_obs()
        prompt = task_goal
        success_flag = "unknown"
        deadline = time.monotonic() + self.args.episode_wall_s

        while True:
            if time.monotonic() > deadline:
                raise EpisodeWallClockTimeout(
                    f"单集超过 {self.args.episode_wall_s}s 墙钟（已走 {epstate.count} 步）"
                )
            if not epstate.action_plan:
                action_chunk = self.get_action_chunk(
                    client, epstate, img, wrist_img, robot_state, prompt,
                    exec_horizon=self.args.obs_horizon
                )

                epstate.action_plan.extend(action_chunk)
                epstate.clear_buffers()

            action = epstate.action_plan.popleft()
            obs, stop_flag, success_flag = env_runner.step(action)
            epstate.count += 1

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
                motion_k=epstate.motion_k_last,
            )

            if stop_flag:
                break

        if success_flag == "unknown":
            return "unknown"

        # 计划格式 <Task>_ep<n>_<终态>_<难度>.mp4；不拼 task goal——xhard 的 goal 很长，拼进去单个文件名超 255 字节，
        # finalize_all 的 os.replace 抛 ENAMETOOLONG（goal 本身已烧进视频文字条）
        video_filename = f"{env_runner.env_id}_ep{env_runner.episode_id}_{success_flag}_{env_runner.difficulty}.mp4"
        recorder.save_video(video_filename)

        return success_flag


    def init_episode(
        self,
        env_runner,
        epstate: EpisodeState,
        video_save_dir: Path,
    ) -> Tuple[str, RolloutRecorder]:
        pre_traj = env_runner.get_init_obs()
        task_goal = pre_traj["task_goal"]

        # 逐步动作日志：每局一份 steps/<Task>_ep<n>.jsonl，行数 = record() 次数（含 demo 前缀帧）
        steps_path = self.save_dir / "steps" / f"{env_runner.env_id}_ep{env_runner.episode_id}.jsonl"
        recorder = RolloutRecorder(video_save_dir, task_goal, fps=30, steps_path=steps_path)

        print(f"task_goal: {task_goal}")

        epstate.image_buffer.extend(pre_traj["images"])
        epstate.wrist_image_buffer.extend(pre_traj["wrist_images"])
        epstate.state_buffer.extend(pre_traj["states"])

        # 注入 demo 前缀时走显式长度，不动 TASK_WITH_VIDEO_DEMO：那张表表达的是「该任务恒有 video demo」，
        # 而 BinFill 只在本轮注入时才有——把它加进表里会让 baseline 700 条的 BinFill 变成假命题。
        demo_len = getattr(env_runner, "demo_prefix_len", 0)
        for i in range(len(pre_traj["images"])):
            recorder.record(
                image=pre_traj["images"][i].copy(),
                wrist_image=pre_traj["wrist_images"][i].copy(),
                state=pre_traj["states"][i].copy(),
                is_video_demo=(i < demo_len) if demo_len else
                              (env_runner.env_id in TASK_WITH_VIDEO_DEMO and i < len(pre_traj["images"]) - 1),
            )

        epstate.exec_start_idx = len(epstate.image_buffer) - 1
        epstate.exec_start_idx_initial = epstate.exec_start_idx
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

        # 留住整个响应：motion 模型会随 actions 一并回传 motion_k / motion_downsample_steps
        resp = client.infer(element)
        if "motion_k" in resp:
            state.motion_infers += 1
            state.motion_k_max = max(state.motion_k_max, int(resp["motion_k"]))
            state.motion_k_last = int(resp["motion_k"])
            # 服务端侧是 episode 内累计值（FrameSampMemory 每 episode 随 reset 重建），取最大即本集总数
            state.motion_downsample_steps = max(
                state.motion_downsample_steps, int(resp["motion_downsample_steps"]))
            state.motion_budget = int(resp["motion_budget"])
            state.motion_overflow = str(resp["motion_overflow"])
        action_chunk = resp["actions"]
        return action_chunk[:exec_horizon]


def setup_save_directory(args: Args) -> Path:
    """Set up and validate save directories."""
    save_dir = (
        Path(args.save_dir)
        / args.policy_name
        / f"ckpt{args.model_ckpt_id}"
        / f"seed{args.model_seed}"
    )

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

    # 官方路径保持原语义：重启即重试 error 集。
    # 注入候选路径不能这么做——run_shard.sh 把客户端分块跑（每块新评 <= max_new_episodes 集），
    # 块数按「计划集数 / 每块集数」固定算；若在这里删掉 error，下一块会重跑它而不是推进，
    # 计划末尾的集就永远轮不到，check_shard.py 最后报「计划与结果不符」。
    # 所以那条路径下 error 是终态，补跑由 merge_shards.py 产出的 retry_plan.json + 独立 job 承担。
    if not args.episode_plan:
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


def summarize(log_dict: dict, save_dir: Path) -> None:
    """按组算成功率。组键在官方路径下是任务名，在注入候选路径下是「任务/难度」。"""
    try:
        rates = {
            group: sum(value is True for value in entries.values()) / len(entries)
            for group, entries in log_dict.items() if entries
        }
        if not rates:
            print("[robomme] 没有已完成的组，跳过汇总")
            return
        final_results = {
            "success_rate": rates,
            "total_success_rate": sum(rates.values()) / len(rates),
        }
        with open(save_dir / "log.json", "w") as f:
            json.dump(final_results, f, indent=2)
    except Exception as e:
        print(f"Error saving final results: {e}")
        raise


def load_plan(args: Args):
    """读分片计划与 1200 局清单，返回 (plan, header, 组 -> [清单行...])。

    清单 jsonl 每行 ``{"task","episode","seed","difficulty"}``，唯一键 (task, difficulty, episode)；
    plan 里记了清单的 sha256，读到的字节不符即拒绝起跑。
    """
    plan = json.loads(Path(args.episode_plan).read_text(encoding="utf-8"))
    manifest_path = Path(args.manifest or plan["manifest"])
    raw = manifest_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if plan.get("manifest_sha256") and plan["manifest_sha256"] != sha:
        raise ValueError(f"清单 sha256 {sha[:12]} 与计划记录 {plan['manifest_sha256'][:12]} 不符：{manifest_path}")
    index = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["task"], row["difficulty"], int(row["episode"]))
        if key in index:
            raise ValueError(f"清单里有重复行：{key}")
        index[key] = row

    groups = {}
    for group, episodes in plan["groups"].items():
        task, difficulty = group.split("/")
        picked = []
        for episode in episodes:
            key = (task, difficulty, int(episode))
            row = index.get(key)
            if row is None:
                raise ValueError(f"计划里的局不在清单中：{key}")
            picked.append(row)
        groups[group] = picked
    header = {"manifest": str(manifest_path), "manifest_sha256": sha}
    return plan, header, groups


def report_video_encodes() -> None:
    """回收本进程全部 ffmpeg 子进程（wait、rc=0 才改成最终名），打 VIDEO_ENCODE 判定行。"""
    finalized, failed = finalize_all()
    print(f"VIDEO_ENCODE={'PASS' if failed == 0 else 'FAIL'} pending=0 finalized={finalized} failed={failed}",
          flush=True)


def evaluate(args: Args):
    """Main evaluation function."""
    check_args(args)
    if args.max_episodes < 0:
        raise ValueError("max_episodes 必须为非负整数")
    if args.episode_plan and (args.only_tasks or args.exclude_tasks or args.re_eval_tasks or args.max_episodes):
        raise ValueError("episode_plan 与 only_tasks/exclude_tasks/re_eval_tasks/max_episodes 互斥")

    save_dir = setup_save_directory(args)
    video_save_dir = save_dir / "videos"

    log_dict = setup_log_dict(save_dir, args)
    evaluator = EpisodeEvaluator(args, save_dir)
    evaluated = 0
    # motion 窗统计与 progress.json 同样跨块续写（run_shard.sh 把一个分片切成多个客户端进程）；
    # 不能塞进 progress.json——check_shard.py 对那份硬断言 `type(value) is bool`
    motion_stats_path = save_dir / "motion_stats.json"
    motion_stats = json.loads(motion_stats_path.read_text(encoding="utf-8")) \
        if motion_stats_path.is_file() else {}

    if args.episode_plan:
        plan, header, groups = load_plan(args)
        if args.demo_prefix_store:
            # 守卫②：允许混合计划（692 条那种），但**计划里的每一条 BinFill 都必须在 store 里有条目**。
            # 非 BinFill 组由 SpecEnvRunner 显式关掉注入（打 DEMO_PREFIX_DISABLED），不与其原生 demo 叠加。
            # 缺条目直接拒绝起跑：静默跳过会让一部分 BinFill 集按旧的错口径跑，事后无从分辨。
            from demo_prefix import DemoPrefixStore
            store_probe = DemoPrefixStore(args.demo_prefix_store)
            missing = [f"BinFill/{g.split('/')[1]}/{row['episode']}"
                       for g, rows_ in groups.items() if g.startswith("BinFill/")
                       for row in rows_ if not store_probe.has("BinFill", g.split("/")[1], row["episode"])]
            if missing:
                raise ValueError(
                    f"计划里有 {len(missing)} 条 BinFill 不在 demo 前缀库中：{missing[:5]}"
                    f"（库根 {args.demo_prefix_store}）")
            n_binfill = sum(len(rows_) for g, rows_ in groups.items() if g.startswith("BinFill/"))
            n_other = sum(len(rows_) for g, rows_ in groups.items() if not g.startswith("BinFill/"))
            print(f"DEMO_PREFIX_PLAN binfill={n_binfill}（全部命中 store）other={n_other}（不注入）", flush=True)
        (save_dir / "plan.json").write_text(json.dumps(
            {"shard_tag": args.shard_tag or plan.get("shard_tag", ""),
             "manifest": header["manifest"],
             "manifest_sha256": header["manifest_sha256"],
             "identity_sha256": header["manifest_sha256"],
             "max_steps": args.max_steps,
             "groups": {g: [row["episode"] for row in rows] for g, rows in groups.items()}},
            indent=2, ensure_ascii=False), encoding="utf-8")

        for group, rows in groups.items():
            log_dict.setdefault(group, {})
            pending = [row for row in rows if str(row["episode"]) not in log_dict[group]]
            if not pending:
                continue
            task, difficulty = group.split("/")
            env_runner = SpecEnvRunner(task, difficulty, video_save_dir,
                                       max_steps=args.max_steps,
                                       demo_prefix_store=args.demo_prefix_store)
            for row in pending:
                if args.max_new_episodes and evaluated >= args.max_new_episodes:
                    break
                episode_id = row["episode"]
                try:
                    with episode_deadline(args.episode_wall_s):
                        env_runner.make_env(row)
                        print(f"\n[robomme] env for {group} episode {episode_id} setup finished")
                        success_flag = evaluator.eval_each_episode(env_runner, video_save_dir)
                    if success_flag in ("unknown", "error"):
                        log_dict[group][str(episode_id)] = "error"
                    else:
                        log_dict[group][str(episode_id)] = success_flag == "success"
                except Exception as e:
                    print(f"Error evaluating {group} episode {episode_id}: {e}")
                    log_dict[group][str(episode_id)] = "error"
                finally:
                    try:
                        env_runner.close_env()
                    except Exception as e:
                        print(f"关闭环境失败：{group}/{episode_id}: {e}")
                        log_dict[group][str(episode_id)] = "error"
                evaluated += 1
                stat = evaluator.last_motion_stat
                if stat is not None:
                    motion_stats[f"{group}/{episode_id}"] = stat
                with open(save_dir / "progress.json", "w") as f:
                    json.dump(log_dict, f, indent=2)
                if motion_stats:
                    with open(motion_stats_path, "w") as f:
                        json.dump(motion_stats, f, indent=2)
            del env_runner
            time.sleep(1)
            if args.max_new_episodes and evaluated >= args.max_new_episodes:
                print(f"[robomme] 本进程已新评 {evaluated} 集，达到 max_new_episodes，正常退出")
                report_video_encodes()
                return

        report_video_encodes()
        summarize(log_dict, save_dir)
        return

    # ---- 官方 env_metadata 路径（保持原口径，run.sh 的单回合验收依赖它）----
    if args.only_tasks:
        task_names = args.only_tasks.split(",")
    else:
        task_names = TASK_NAME_LIST

    if args.exclude_tasks:
        task_names = [task_name for task_name in task_names if task_name not in args.exclude_tasks.split(",")]
        for task in args.exclude_tasks.split(","):
            log_dict[task] = {str(i): False for i in range(50)}

    for task_name in task_names:
        if task_name not in log_dict:
            log_dict[task_name] = {}

        env_runner = EnvRunner(task_name, video_save_dir, max_steps=args.max_steps)
        num_episodes = env_runner.num_episodes

        success_flag = "unknown"

        if args.max_episodes > 0:
            num_episodes = min(num_episodes, args.max_episodes)
        for episode_id in range(num_episodes):
            if str(episode_id) in log_dict[task_name]:
                print(f"[robomme] episode {episode_id} already evaluated, skipping...")
                continue

            try:
                env_runner.make_env(episode_id)
                print(f"\n[robomme] env for task {task_name} episode {episode_id} setup finished")
                success_flag = evaluator.eval_each_episode(env_runner, video_save_dir)
                if success_flag in ("unknown", "error"):
                    log_dict[task_name][str(episode_id)] = "error"
                else:
                    log_dict[task_name][str(episode_id)] = success_flag == "success"
            except Exception as e:
                print(f"Error evaluating episode {episode_id} for task {task_name}: {e}")
                success_flag = "error"
                log_dict[task_name][str(episode_id)] = "error"
            finally:
                try:
                    env_runner.close_env()
                except Exception as e:
                    print(f"关闭环境失败：{task_name}/{episode_id}: {e}")
                    log_dict[task_name][str(episode_id)] = "error"
            with open(save_dir / "progress.json", "w") as f:
                json.dump(log_dict, f, indent=2)

        del env_runner
        time.sleep(1)

    report_video_encodes()
    summarize(log_dict, save_dir)


if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate)
