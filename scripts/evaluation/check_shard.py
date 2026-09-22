"""分片结果验收：计划全覆盖、取值合法、视频与非 error 集一一对应且可完整解码。

与单回合的 check_result.py 分开，不动那份——它承载着 vail-eval-gl-20260918T0400Z 已绿留档的
可复现口径。本验收的三点不同：
1. 组键是「任务/难度」而不是任务名（注入候选的唯一键是 (task, difficulty, episode)）；
2. episode 号来自计划、不是 range(N)；
3. **视频数 == 非 error 集数**，不是 == 总集数——eval.py 在 success_flag=="unknown" 时提前
   return 不落视频，抛异常那条路同样没有视频，这两种都记 "error"。
"""
import json
import os
from pathlib import Path
import sys

import imageio.v2 as imageio

# ── motion 窗数的闭式公式（0922-binfill-demo-prefix-plan.md C 节）────────────────────────────
# ⚠ 必须在**本文件本地重写**，不得 import mme_vla_suite：CLIENT_UV 的 client/pyproject.toml
# 没有那个包；更要紧的是共用代码会让这道闸失效——服务端是 while 循环逐窗 append 的枚举实现，
# 这里是闭式计数，两个独立实现互相复核才有意义。体例照抄
# scripts/training/tests/eval_es_bound.py::predicted_windows。
MOTION_STRIDE = 16
MOTION_WINDOW = 33          # motion.window_frames
DEMO_MIN_REAL = 17          # motion.demo_min_real_frames
EXEC_HORIZON = 16           # eval.py 的 obs_horizon，一次推理执行这么多步


def k_demo_closed(es: int) -> int:
    """demo 段窗数：合法起点 s = 16m 满足 s + (demo_min_real - 1) <= es - 1。"""
    if es < DEMO_MIN_REAL:
        return 0
    return (es - DEMO_MIN_REAL) // MOTION_STRIDE + 1


def k_exec_closed(t_rel: int) -> int:
    """exec 段窗数：合法起点 u = 16m 满足 u + (window_frames - 1) <= t_rel，t_rel = step_idx - es。"""
    if t_rel < MOTION_WINDOW - 1:
        return 0
    return (t_rel - (MOTION_WINDOW - 1)) // MOTION_STRIDE + 1


def k_total_closed(es: int, infer_index: int) -> int:
    """第 ``infer_index`` 次推理（1 起算）时的合法起点总数。

    第 j 次推理发生在已走 ``16(j-1)`` 步之后，故 ``t_rel = 16(j-1)``。
    """
    return k_demo_closed(es) + k_exec_closed(EXEC_HORIZON * (infer_index - 1))


def check_motion_windows(root: Path, progress: dict) -> dict:
    """MOTION_WINDOWS 闸：五条判据。

    ``motion_stats.json`` 不存在时按 ``EXPECT_MOTION_STATS`` 分叉：期望为真（本轮 motion run）
    → **FAIL**；否则 → SKIP。SKIP 只对「非 motion 模型」与「复核旧 run 的 check.json」有效，
    不能拿来掩盖本轮 motion run 漏写统计。
    """
    expect = os.environ.get("EXPECT_MOTION_STATS", "0") == "1"
    path = root / "motion_stats.json"
    if not path.is_file():
        if expect:
            print("MOTION_WINDOWS=FAIL reason=missing-motion-stats", flush=True)
            raise AssertionError(f"EXPECT_MOTION_STATS=1 但 {path} 不存在")
        print("MOTION_WINDOWS=SKIP reason=no-motion-stats", flush=True)
        return {"verdict": "SKIP", "reason": "no-motion-stats"}

    stats = json.loads(path.read_text(encoding="utf-8"))
    # ① 覆盖：每个非 error 集都必须有记录（缺了说明该集第一次 infer 之前就挂了却没记 error）
    non_error = {f"{g}/{e}" for g, entries in progress.items()
                 for e, v in entries.items() if v != "error"}
    missing = sorted(non_error - set(stats))
    assert not missing, f"MOTION_WINDOWS=FAIL 非 error 集缺 motion 记录：{missing[:5]}"

    budgets = {int(s["motion_budget"]) for s in stats.values()}
    overflows = {str(s["motion_overflow"]) for s in stats.values()}
    # ⑤ budget 自洽：全片同一个 budget，且与期望一致
    assert len(budgets) == 1, f"MOTION_WINDOWS=FAIL 同一分片里出现多个 budget：{sorted(budgets)}"
    budget = budgets.pop()
    expect_budget = int(os.environ.get("EXPECT_MOTION_BUDGET", "160"))
    assert budget == expect_budget, f"MOTION_WINDOWS=FAIL budget={budget} != 期望 {expect_budget}"
    assert len(overflows) == 1, f"MOTION_WINDOWS=FAIL 同一分片里出现多个 overflow 口径：{sorted(overflows)}"
    overflow = overflows.pop()
    assert overflow in ("raise", "resample"), f"MOTION_WINDOWS=FAIL 非法 overflow={overflow!r}"

    downsampled, over_budget_infers, k_max_seen = 0, 0, 0
    for key, stat in sorted(stats.items()):
        es, infers = int(stat["exec_start_idx"]), int(stat["motion_infers"])
        assert infers >= 1, f"MOTION_WINDOWS=FAIL {key} 的 motion_infers={infers}"
        # ② 公式：服务端枚举出的最大窗数必须等于客户端闭式预测
        predicted = k_total_closed(es, infers)
        actual = int(stat["motion_k_max"])
        assert predicted == actual, (
            f"MOTION_WINDOWS=FAIL {key} 窗数不符：闭式 {predicted} != 服务端枚举 {actual} "
            f"(es={es} infers={infers})")
        # ③ 降级触发与超预算严格互为充要：逐次推理闭式数一遍，必须与服务端计数逐条对上
        expected_downsample = sum(1 for j in range(1, infers + 1) if k_total_closed(es, j) > budget)
        actual_downsample = int(stat["motion_downsample_steps"])
        if overflow == "resample":
            assert expected_downsample == actual_downsample, (
                f"MOTION_WINDOWS=FAIL {key} 降级次数不符：闭式 {expected_downsample} != "
                f"实测 {actual_downsample} (es={es} infers={infers} budget={budget})")
        else:
            # ④ 未开 resample 时一次都不该降级——真超预算会 raise，该集记 error 而不是悄悄裁剪
            assert actual_downsample == 0, (
                f"MOTION_WINDOWS=FAIL overflow=raise 下 {key} 却降级了 {actual_downsample} 次")
            assert actual <= budget, f"MOTION_WINDOWS=FAIL overflow=raise 下 {key} 的 k_max={actual} > {budget}"
        downsampled += actual_downsample
        over_budget_infers += expected_downsample
        k_max_seen = max(k_max_seen, actual)

    summary = {"verdict": "PASS", "episodes": len(stats), "budget": budget, "overflow": overflow,
               "k_max": k_max_seen, "downsample_steps": downsampled,
               "episodes_downsampled": sum(1 for s in stats.values()
                                           if int(s["motion_downsample_steps"]) > 0),
               "over_budget_infers_closed": over_budget_infers}
    print("MOTION_WINDOWS=PASS " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def check_shard(run_dir, policy, ckpt, seed, plan_path):
    root = Path(run_dir) / policy / f"ckpt{ckpt}" / f"seed{seed}"
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))

    planned = {(g, int(e)) for g, eps in plan["groups"].items() for e in eps}
    done = {(g, int(e)) for g, entries in progress.items() for e in entries}
    assert done == planned, (
        f"计划与结果不符：缺 {sorted(planned - done)[:5]}，多 {sorted(done - planned)[:5]}"
    )

    errors, successes, videos_expected = [], 0, []
    for group, entries in progress.items():
        task, difficulty = group.split("/")
        for episode, value in entries.items():
            if value == "error":
                errors.append({"group": group, "episode": int(episode)})
                continue
            assert type(value) is bool, f"非法取值 {group}/{episode}={value!r}"
            successes += value
            videos_expected.append((task, int(episode), difficulty))

    videos = sorted((root / "videos").glob("*.mp4"))
    assert len(videos) == len(videos_expected), (
        f"视频数 {len(videos)} != 非 error 集数 {len(videos_expected)}"
    )

    records = []
    for task, episode, difficulty in videos_expected:
        prefix, suffix = f"{task}_ep{episode}_", f"_{difficulty}.mp4"
        matched = [v for v in videos if v.name.startswith(prefix) and v.name.endswith(suffix)]
        assert len(matched) == 1, f"{task}/{difficulty} ep{episode} 匹配到 {len(matched)} 个视频"
        video = matched[0]
        count = 0
        with imageio.get_reader(video) as reader:
            for frame in reader:
                assert frame.ndim == 3 and frame.shape[-1] == 3
                count += 1
        assert count > 1, video
        records.append({"group": f"{task}/{difficulty}", "episode": episode,
                        "path": str(video), "frames": count, "bytes": video.stat().st_size})

    motion = check_motion_windows(root, progress)

    total = len(planned)
    summary = {
        "pipeline_pass": True,
        "motion_windows": motion,
        "shard_tag": plan.get("shard_tag", ""),
        "identity_sha256": plan.get("identity_sha256", ""),
        "episodes": total,
        "evaluated": total - len(errors),
        "successes": successes,
        "errors": errors,
        "groups": {g: {"episodes": len(entries),
                       "successes": sum(v is True for v in entries.values()),
                       "errors": sum(v == "error" for v in entries.values())}
                   for g, entries in sorted(progress.items())},
        "videos": records,
    }
    (root / "check.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SHARD_PASS", json.dumps({k: v for k, v in summary.items() if k != "videos"},
                                   ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    run_dir, policy, ckpt, seed, plan_path = sys.argv[1:]
    check_shard(run_dir, policy, ckpt, seed, plan_path)
