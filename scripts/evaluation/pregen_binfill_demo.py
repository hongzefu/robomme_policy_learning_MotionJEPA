"""离线生成 BinFill 的 demo 前缀（0922-binfill-demo-prefix-plan.md A 节）。

**为什么要这么绕**：BinFill 是四个评测任务里唯一「训练有 demo 段、评测没有」的。训练侧
``generate_dataset_newseed.py::_binfill_demo_deliverable`` 把整条成功轨迹复制一遍接在自己前面
（400/400 满足 ``num_timesteps == 2 x exec_start_idx``）；评测侧 ``BinFill.py::_initialize_episode``
的 ``task_list`` 里 ``demonstration`` 三处全是 ``False``，``reset()`` 只返回 1 帧，``exec_start_idx = 0``。

**为什么不能直接翻 benchmark 的标志**（已确证，不要再试）：方块入 bin 是
``is_any_obj_dropped_onto_delete`` 里的物理删除（``set_pose(p=[10,10,0])``），``*_cubes_in_bin``
单调累加且无复位路径；demo 跑完后 ``sequential_task_check`` 的任务指针已走到 "All tasks completed"，
策略第一步就白拿 success。BinFill 也没有 VideoRepick/RouteStick 那种 ``"NO RECORD"`` + ``solve_strong_reset``
的收尾任务，spawn 余量更不够演两遍（easy spawn 5 / target 3）。

**绕开的办法**：同一条候选（同 seed、同 spec）起**两个** env——这里这个把 ``demonstration`` 全翻成 True，
让 planner 演完整条、只取画面，演完即销毁；正式评测再起一个全新的干净 env（``demonstration`` 仍为 False）
交给策略。两者同 seed 同 spec，故布局逐字一致。monkey-patch 只在本脚本里做，
**不动 benchmark 源码、不换 gitlink**。

判定行：
  ``DEMO_OK task=BinFill diff=<d> ep=<n> D=<D> k_demo=<k> status=<s> ...``（每集一条）
  ``DEMO_FAIL ...``（每集一条，失败时）
  ``SHARD_PREGEN_DONE shard=<K> attempted=<n> ok=<n> failed=<n> too_long=<n>``
  ``PREGEN_OK episodes=<n> ok=<n> failed=<n> used_rrt=<n>``（finalize）
  ``DEMO_WINDOW_GUARD max_D=<n> max_k_demo=<n> cap=80 train_observed_max=71 over_train=<n>``（finalize）
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples/robomme"))

# ── 评测侧口径（run_shard.sh 写死 MAX_STEPS=2000，motion 配置取自 checkpoint 桶的 resolved 快照）──
MAX_STEPS = 2000
POS_TABLE_ROWS = 4096          # FrameSampMemory.__init__ 的 max_steps，pos_emb 只有这么多行，无人覆盖
MOTION_STRIDE = 16
DEMO_MIN_REAL = 17             # motion.demo_min_real_frames
MOTION_BUDGET = 160            # motion.budget
DEMO_WINDOW_CAP = MOTION_BUDGET // 2   # _motion_quota 的 demo 段上限；超过它降级才会真的丢 demo 窗
TRAIN_OBSERVED_MAX_K_DEMO = 71         # 训练 1600 集实测的 demo 窗数上限，只作对照、不作判据


def k_demo_of(d: int) -> int:
    """演示长度 D 能切出几个 motion 窗——与 ``FrameSampMemory.visible_motion_frames`` 的 demo 段同式：
    合法起点 s 满足 ``s + (demo_min_real - 1) <= es - 1``，其中 es = D。"""
    if d < DEMO_MIN_REAL:
        return 0
    return (d - DEMO_MIN_REAL) // MOTION_STRIDE + 1


def max_d_for_cap(cap: int) -> int:
    """反解：让 ``k_demo(D) <= cap`` 成立的最大 D。cap=80 时是 1296。

    计划正文写的 ``D <= 1281`` 是个更紧的保守值（1281 与 1296 的 k_demo 同为 80），
    这里直接用充要判据 ``k_demo(D) <= cap``，既精确又自解释；训练 D 上限 1152 ⇒ k_demo=71，
    两种写法在实际样本上都够不着。
    """
    return DEMO_MIN_REAL + cap * MOTION_STRIDE - 1


#: pos 表闸：注入后全域帧号最大到 D + MAX_STEPS，pos_emb 只有 POS_TABLE_ROWS 行
MAX_D_POS_TABLE = POS_TABLE_ROWS - MAX_STEPS - 1        # 2095
#: demo 窗闸：让「降级时 demo 一窗不丢」成为可执行约束，而不是「与训练同分布」的分布推断
MAX_D_DEMO_WINDOW = max_d_for_cap(DEMO_WINDOW_CAP)      # 1296
MAX_D = min(MAX_D_POS_TABLE, MAX_D_DEMO_WINDOW)         # 生效的是更紧的那个

DEFAULT_CANDIDATES = REPO / "third_party/robomme_benchmark/artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl"

#: planner 是否回退过 RRT*（screw 规划失败才会走到）。闭包里取的是 bound method，
#: 在子类上赋值即可拦截；用于「重试有效性」实测的病例筛选。
_RRT_USED = {"flag": False}


def install_patches() -> None:
    """装两个 monkey-patch：demo 开关翻转 + RRT* 打点。只在本进程生效，不改 benchmark 源码。"""
    from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
    from robomme.robomme_env.utils import planner_fail_safe as pfs

    original_get_demo = DemonstrationWrapper.get_demonstration_trajectory

    def patched_get_demo(self):
        """只对 BinFill：在原实现扫描 task_list 之前把全部 2N+1 条的 demonstration 翻成 True。

        翻的是 ``self.unwrapped.task_list``——原实现第一行 ``getattr(self, 'task_list', [])``
        经 ``gym.Wrapper.__getattr__`` 取到的就是同一个 list 对象。

        下手点选在这里而**不是** patch ``BinFill._initialize_episode``：后者是 ManiSkill
        ``BaseEnv.reset`` 的内部回调、可能因 reconfigure 被调多次，且其后紧跟 ``task4recovery`` /
        ``inject_fail_grasp``，插进去会打乱它们看到的 task_list。

        最后一条 "press the button" 也翻：训练侧的 demo 是含按钮的整条成功轨迹，
        不按按钮的 demo 与训练分布不符。副作用（方块被物理删除、任务指针走到
        "All tasks completed"）全部无所谓——这个 env 演完就扔。
        """
        if self.unwrapped.spec.id == "BinFill":
            task_list = self.unwrapped.task_list
            if len(task_list) % 2 != 1:
                raise RuntimeError(f"BinFill task_list 长度 {len(task_list)} 不是 2N+1")
            for entry in task_list:
                entry["demonstration"] = True
        return original_get_demo(self)

    DemonstrationWrapper.get_demonstration_trajectory = patched_get_demo

    original_rrt = pfs.FailAwarePandaArmMotionPlanningSolver.move_to_pose_with_RRTStar

    def patched_rrt(self, *args, **kwargs):
        _RRT_USED["flag"] = True
        return original_rrt(self, *args, **kwargs)

    pfs.FailAwarePandaArmMotionPlanningSolver.move_to_pose_with_RRTStar = patched_rrt


def find_demo_wrapper(env):
    """在 wrapper 链里找 DemonstrationWrapper（joint_angle 下外面只套了一层 FailAwareWrapper）。"""
    from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
    node = env
    while node is not None:
        if isinstance(node, DemonstrationWrapper):
            return node
        node = getattr(node, "env", None)
    raise RuntimeError("wrapper 链里找不到 DemonstrationWrapper")


def load_binfill_rows(candidates: Path):
    """取 test/primary 的全部 BinFill 候选，按 (难度, episode) 排序。"""
    import robomme
    from robomme.injection_candidates import load_candidates, candidate_key

    benchmark_root = Path(robomme.__file__).resolve().parents[2]
    header, rows = load_candidates(candidates, repo_root=benchmark_root)
    binfill = [r for r in rows
               if r["task"] == "BinFill" and r["split"] == "test" and r["role"] == "primary"]
    binfill.sort(key=candidate_key)
    per_group = collections.Counter(r["difficulty"] for r in binfill)
    if len(binfill) != 150 or set(per_group.values()) != {50}:
        raise SystemExit(f"BinFill test/primary 应为 3 难度 x 50，实际 {dict(per_group)}")
    return header, binfill


def check_episode(base, demo_wrapper, info, spec) -> tuple[bool, dict]:
    """四道正确性校验——``get_demonstration_trajectory`` 会吞掉 screw→RRT* 双重失败并 ``continue``，
    失败的集会安静产出「少放一个方块」的 demo，下面就是拦它的。

    ⚠ 按钮这一项绝不能写成 ``is_button_pressed(base, obj=base.button)``：
    ``solve_button`` 按下按钮后会**抬手**，按钮弹簧随即回位，而成功状态早已在
    ``sequential_task_check`` 里锁存。于是 ``status=="success"`` + 计数正确 + ``episode_success=True``
    与 ``is_button_pressed()==False`` 可以同时成立——按结束瞬时状态判会把正常样本当失败剔除。
    正确的锁存量是任务指针：``task_list`` 最后一条 "press the button" 的 ``func`` 就是
    ``is_button_pressed``，``sequential_task_check`` 里 ``self.timestep`` 单调递增不回退、
    走完才置 ``num_tasks``，所以 ``timestep == len(task_list)`` 等价于「某个 evaluate 时刻按钮确实被按下过」。
    """
    in_bin = {"red": int(base.red_cubes_in_bin),
              "blue": int(base.blue_cubes_in_bin),
              "green": int(base.green_cubes_in_bin)}
    target = {k: int(v) for k, v in spec["objects"]["target_count"].items()}
    put_in_total = int(spec["objects"]["put_in_total"])
    timestep = int(base.timestep)
    num_tasks = len(base.task_list)
    try:
        from robomme.robomme_env.utils.subgoal_evaluate_func import is_button_pressed
        button_final = bool(is_button_pressed(base, obj=base.button))
    except Exception:
        button_final = None          # 只作参考、不进判据，取不到就记 None

    checks = {
        "status_success": info.get("status") == "success",
        "in_bin_matches_target": all(in_bin.get(c, 0) == n for c, n in target.items()),
        "in_bin_total_matches": sum(in_bin.values()) == put_in_total,
        "task_pointer_finished": timestep == num_tasks,
        "episode_success": bool(demo_wrapper.episode_success),
    }
    detail = {
        "checks": checks,
        "status": str(info.get("status")),
        "in_bin": in_bin,
        "target_count": target,
        "put_in_total": put_in_total,
        "timestep_final": timestep,
        "task_list_len": num_tasks,
        "button_pressed_final": button_final,     # 参考量，见上面的说明
    }
    return all(checks.values()), detail


def write_episode_h5(path: Path, row, front, wrist, states, task_goal, detail, extra) -> dict:
    """front 走 PNG **无损**（进 SigLIP 与 Wan VAE），wrist 走 JPEG q90（只进 mp4）。

    不用 h5 的 gzip filter：RGB 交错没有预测器，实测压缩比只有 0.62–0.75，
    而 PNG 对 256x256x3 的 sim 渲染帧能到约 0.53（约 104 KB/帧）。
    """
    from demo_prefix import frames_sha256

    path.parent.mkdir(parents=True, exist_ok=True)
    d = len(front)
    front_png, wrist_jpg = [], []
    for i in range(d):
        ok, buf = cv2.imencode(".png", front[i][:, :, ::-1])
        if not ok:
            raise RuntimeError(f"front 第 {i} 帧 PNG 编码失败")
        front_png.append(np.frombuffer(buf.tobytes(), dtype=np.uint8))
        ok, buf = cv2.imencode(".jpg", wrist[i][:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError(f"wrist 第 {i} 帧 JPEG 编码失败")
        wrist_jpg.append(np.frombuffer(buf.tobytes(), dtype=np.uint8))

    state_arr = np.stack(states).astype(np.float32)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as h5:
        vlen = h5py.vlen_dtype(np.dtype("uint8"))
        h5.create_dataset("front_png", data=np.array(front_png, dtype=object), dtype=vlen)
        h5.create_dataset("wrist_jpg", data=np.array(wrist_jpg, dtype=object), dtype=vlen)
        h5.create_dataset("state", data=state_arr)
        h5.attrs["task"] = row["task"]
        h5.attrs["difficulty"] = row["difficulty"]
        h5.attrs["episode"] = int(row["episode"])
        h5.attrs["seed"] = int(row["seed"])
        h5.attrs["spec_sha256"] = row["spec_sha256"]
        h5.attrs["D"] = d
        h5.attrs["k_demo"] = k_demo_of(d)
        h5.attrs["task_goal"] = task_goal
        # 解码端逐位核对用：换了 cv2 / libpng 版本导致像素变化会当场炸
        h5.attrs["front_raw_sha256"] = frames_sha256(front)
        h5.attrs["wrist_raw_sha256"] = frames_sha256(wrist)
        h5.attrs["state_sha256"] = frames_sha256([state_arr])
        for key, value in detail.items():
            h5.attrs[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else (
                "" if value is None else value)
        for key, value in extra.items():
            h5.attrs[key] = "" if value is None else value
    tmp.replace(path)       # 原子落盘：中途被 Slurm 砍掉不会留下半截 h5 被后续误读
    return {"bytes": path.stat().st_size}


def gen_one(row, sampling, out_root: Path, video_dir: Path, retries: int) -> dict:
    """跑一条候选的 demo 前缀。返回一条记录（含 demo_status）。"""
    from env_runner import SpecEnvRunner

    task, difficulty, episode = row["task"], row["difficulty"], int(row["episode"])
    per_task = {"parameters": sampling["parameters"][task], "positions": sampling["positions"][task]}
    rel = f"{task}/{difficulty}/ep{episode}.h5"
    record = {"task": task, "difficulty": difficulty, "episode": episode,
              "seed": int(row["seed"]), "spec_sha256": row["spec_sha256"], "path": rel}

    last_error = ""
    for attempt in range(1, retries + 2):
        _RRT_USED["flag"] = False
        runner = SpecEnvRunner(task, difficulty, per_task, str(video_dir), max_steps=MAX_STEPS)
        t0 = time.perf_counter()
        try:
            runner.make_env(row)
            obs = runner.get_init_obs()
            used_rrt = bool(_RRT_USED["flag"])
            env = runner.env
            base = env.unwrapped
            demo_wrapper = find_demo_wrapper(env)
            ok, detail = check_episode(base, demo_wrapper, runner.info, row["spec"])
            # 丢掉最后一帧：那是 demo 跑完后的复位步；注入场景下这个角色由干净 env 自己的 reset 帧担任
            d = len(obs["images"]) - 1
            front = [np.ascontiguousarray(f, dtype=np.uint8) for f in obs["images"][:d]]
            wrist = [np.ascontiguousarray(f, dtype=np.uint8) for f in obs["wrist_images"][:d]]
            states = [np.ascontiguousarray(s, dtype=np.float32) for s in obs["states"][:d]]
            record.update(D=d, k_demo=k_demo_of(d), used_rrt=used_rrt, attempts=attempt,
                          seconds=round(time.perf_counter() - t0, 2), **detail)
            if d < 1:
                ok, last_error = False, f"demo 帧数 {d} < 1"
            elif any(f.shape != (256, 256, 3) for f in front):
                ok, last_error = False, "front 帧形状不是 (256,256,3)"
            elif len(front) != len(wrist) or len(front) != len(states):
                ok, last_error = False, "三个缓冲长度不一致"
            if not ok:
                last_error = last_error or f"校验未通过 {detail['checks']}"
                print(f"DEMO_FAIL task={task} diff={difficulty} ep={episode} attempt={attempt} "
                      f"D={d} used_rrt={used_rrt} reason={last_error}", flush=True)
                continue
            if d > MAX_D:
                # 两道硬闸取交集；超出即剔除并计数，不做任何截断（截断会让 demo 段与训练不同形）
                record["demo_status"] = "too_long"
                record["reason"] = (f"D={d} > MAX_D={MAX_D}"
                                    f"（pos 表闸 {MAX_D_POS_TABLE} / demo 窗闸 {MAX_D_DEMO_WINDOW}）")
                print(f"DEMO_FAIL task={task} diff={difficulty} ep={episode} D={d} "
                      f"reason=too_long {record['reason']}", flush=True)
                return record
            info = write_episode_h5(out_root / rel, row, front, wrist, states,
                                    runner.task_goal, detail,
                                    {"used_rrt": used_rrt, "attempts": attempt})
            record.update(demo_status="ok", bytes=info["bytes"])
            print(f"DEMO_OK task={task} diff={difficulty} ep={episode} D={d} k_demo={k_demo_of(d)} "
                  f"status={detail['status']} in_bin={detail['in_bin']} target={detail['target_count']} "
                  f"button_final={detail['button_pressed_final']} used_rrt={used_rrt} "
                  f"attempts={attempt} bytes={info['bytes']} secs={record['seconds']}", flush=True)
            return record
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"DEMO_FAIL task={task} diff={difficulty} ep={episode} attempt={attempt} "
                  f"reason={last_error}", flush=True)
        finally:
            try:
                runner.close_env()
            except Exception as exc:
                print(f"关闭环境失败 {task}/{difficulty}/{episode}: {exc}", flush=True)
            del runner
    record["demo_status"] = "failed"
    record["reason"] = last_error
    return record


def cmd_gen(args):
    install_patches()
    out_root = Path(args.out).resolve()
    header, rows = load_binfill_rows(Path(args.candidates).resolve())
    shard_rows = rows[args.shard::args.shards]
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"shard{args.shard}.json"
    done = {}
    if log_path.is_file():
        done = {f"{r['difficulty']}/{r['episode']}": r
                for r in json.loads(log_path.read_text(encoding="utf-8"))["episodes"]}
    video_dir = out_root / "smoke-videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    for row in shard_rows:
        key = f"{row['difficulty']}/{row['episode']}"
        if key in done and done[key].get("demo_status") in ("ok", "too_long"):
            continue
        if args.max_new_envs and made >= args.max_new_envs:
            print(f"[pregen] 本进程已建 {made} 个 env，达到 max_new_envs，正常退出（外层循环续跑）", flush=True)
            break
        done[key] = gen_one(row, header["sampling_config"], out_root, video_dir, args.retries)
        made += 1
        log_path.write_text(json.dumps(
            {"shard": args.shard, "shards": args.shards,
             "identity_sha256": header["identity_sha256"],
             "episodes": [done[k] for k in sorted(done, key=lambda s: (s.split("/")[0], int(s.split("/")[1])))]},
            indent=2, ensure_ascii=False), encoding="utf-8")

    stats = collections.Counter(r.get("demo_status", "pending") for r in done.values())
    print(f"SHARD_PREGEN_DONE shard={args.shard} planned={len(shard_rows)} recorded={len(done)} "
          f"ok={stats['ok']} failed={stats['failed']} too_long={stats['too_long']} new_envs={made}", flush=True)
    # 本片还没跑完就以 3 退出，外层 while 据此续跑（0 = 本片全部落定）
    return 0 if len(done) >= len(shard_rows) else 3


def cmd_finalize(args):
    out_root = Path(args.out).resolve()
    header, rows = load_binfill_rows(Path(args.candidates).resolve())
    records = {}
    for shard in range(args.shards):
        path = out_root / "logs" / f"shard{shard}.json"
        if not path.is_file():
            raise SystemExit(f"缺分片日志：{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["identity_sha256"] != header["identity_sha256"]:
            raise SystemExit(f"{path} 的候选库身份与当前不符")
        for record in payload["episodes"]:
            key = f"{record['task']}/{record['difficulty']}/{record['episode']}"
            if key in records:
                raise SystemExit(f"{key} 在多个分片里都有记录")
            records[key] = record
    expected = {f"{r['task']}/{r['difficulty']}/{r['episode']}" for r in rows}
    missing = sorted(expected - set(records))
    if missing:
        raise SystemExit(f"还有 {len(missing)} 条没跑：{missing[:5]}")

    ok = {k: v for k, v in records.items() if v.get("demo_status") == "ok"}
    for key, record in ok.items():
        path = out_root / record["path"]
        if not path.is_file():
            raise SystemExit(f"{key} 记为 ok 但 h5 不存在：{path}")

    index = {k: {"path": v["path"], "D": v["D"], "k_demo": v["k_demo"],
                 "demo_status": v["demo_status"], "used_rrt": v.get("used_rrt"),
                 "attempts": v.get("attempts")}
             for k, v in sorted(ok.items())}
    (out_root / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    d_values = [v["D"] for v in ok.values()]
    k_values = [v["k_demo"] for v in ok.values()]
    used_rrt = sum(1 for v in ok.values() if v.get("used_rrt"))
    failed = [k for k, v in records.items() if v.get("demo_status") != "ok"]
    by_difficulty = collections.Counter(v["difficulty"] for v in ok.values())
    manifest = {
        "task": "BinFill",
        "identity_sha256": header["identity_sha256"],
        "candidates": str(Path(args.candidates).resolve()),
        "repo_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "benchmark_head": subprocess.check_output(
            ["git", "-C", str(REPO / "third_party/robomme_benchmark"), "rev-parse", "HEAD"], text=True).strip(),
        "max_steps": MAX_STEPS,
        "gates": {"max_d": MAX_D, "max_d_pos_table": MAX_D_POS_TABLE,
                  "max_d_demo_window": MAX_D_DEMO_WINDOW, "demo_window_cap": DEMO_WINDOW_CAP},
        "planned": len(expected), "ok": len(ok), "failed": len(failed),
        "failed_keys": sorted(failed),
        "used_rrt": used_rrt,
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "D": {"min": min(d_values), "mean": round(sum(d_values) / len(d_values), 1), "max": max(d_values)},
        "k_demo": {"min": min(k_values), "mean": round(sum(k_values) / len(k_values), 1), "max": max(k_values)},
        "bytes": sum(v.get("bytes", 0) for v in ok.values()),
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"PREGEN_OK episodes={len(expected)} ok={len(ok)} failed={len(failed)} used_rrt={used_rrt} "
          f"D_min={manifest['D']['min']} D_mean={manifest['D']['mean']} D_max={manifest['D']['max']} "
          f"bytes={manifest['bytes']}", flush=True)
    over_train = sum(1 for k in k_values if k > TRAIN_OBSERVED_MAX_K_DEMO)
    print(f"DEMO_WINDOW_GUARD max_D={max(d_values)} max_k_demo={max(k_values)} cap={DEMO_WINDOW_CAP} "
          f"train_observed_max={TRAIN_OBSERVED_MAX_K_DEMO} over_train={over_train}", flush=True)
    if max(k_values) > DEMO_WINDOW_CAP:
        # 理论上被硬闸挡住；真出现说明闸写错了，必须停
        raise SystemExit(f"max_k_demo={max(k_values)} > cap={DEMO_WINDOW_CAP}，降级会丢 demo 窗，停止")
    if over_train:
        keys = sorted(k for k, v in ok.items() if v["k_demo"] > TRAIN_OBSERVED_MAX_K_DEMO)
        print(f"[warn] 有 {over_train} 条超出训练见过的最大 demo 窗数 {TRAIN_OBSERVED_MAX_K_DEMO}："
              f"{keys}（不停跑，但必须写进留档）", flush=True)
    if failed:
        rate = len(failed) / len(expected)
        print(f"[warn] {len(failed)} 条未产出（{rate:.1%}）：{sorted(failed)}", flush=True)
        if rate > 0.05:
            raise SystemExit(f"失败率 {rate:.1%} > 5%，按计划停下来找用户裁决")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("gen", "finalize"))
    parser.add_argument("--out", required=True, help="demo 前缀库根目录")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--max-new-envs", type=int, default=20,
                        help="本进程最多新建几个 env（Vulkan 静态 TLS 红线约 27 次/进程）")
    parser.add_argument("--retries", type=int, default=1,
                        help="单集失败后额外重试几次；重试有效性未实测前按计划保持 1，不设 0")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    args = parser.parse_args()
    if not (0 <= args.shard < args.shards):
        raise SystemExit(f"--shard 必须落在 [0, {args.shards})")
    if args.retries < 0:
        raise SystemExit("--retries 必须为非负整数")
    sys.exit(cmd_gen(args) if args.mode == "gen" else cmd_finalize(args))


if __name__ == "__main__":
    main()
