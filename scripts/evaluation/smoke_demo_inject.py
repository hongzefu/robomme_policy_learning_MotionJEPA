"""注入侧冒烟核对（0922-binfill-demo-prefix-plan.md 冒烟节）。

评测链路是 client/server 分离的，客户端看不到 ``FrameSampMemory`` 内部状态，所以计划列的那批断言
（``exec_start_idx``、``_history_feats`` 键集合、motion 窗数、``_raw_frames`` 淘汰、``mem_order`` 置换）
没法从评测链路里验，只能直接建 policy 来看。

**为什么分两个子命令**：``robomme``（建 env）只装在 client 环境、``mme_vla_suite`` + jax（建 policy）
只装在 server 环境，两套 venv 互斥，一个进程里做不完。于是拆成：

* ``export-reset``（**client** 环境）：建一个干净 env（``demonstration`` 仍为 False、不装任何 monkey-patch），
  取它的 reset 帧存成 npy，并与前缀第 0 帧比一眼；
* ``verify``（**server** 环境，需 GPU）：读前缀 h5 + 那张 npy，建 policy，把注入后的首批喂进去逐项断言。
  顺带回答两个只能实测的问题：**D+1 帧一次性过 SigLIP 会不会 OOM、耗时多少**。

``export-reset`` 存下的 npy 同时供 ``enc_chunk_cmp.py --reset-frame-npy`` 用——那一帧是 exec 段第 0 帧、
有历史口径，必须进 SigLIP 分批对照的覆盖点。

**motion 侧必须走真 sidecar，不能用 stub**：stub sidecar 靠 ``P.stub_decode(window)`` 从**像素**
解出帧编号，只认 ``P.stub_frame(j)`` 那种合成的带编号帧；喂真实渲染帧解出来是乱的，必报
「33 帧编号不符合连续前缀与重复末帧规则」。正式评测本来也走真 sidecar，所以这里同源。
代价是首批要真编 ``k_demo`` 个窗（约 1.7 s/窗），顺带把这个耗时实测出来。

用法：
    # client 环境
    uv run scripts/evaluation/smoke_demo_inject.py export-reset \
        --store <前缀库根> --task BinFill --difficulty easy --episode 156
    # server 环境（需 GPU；motion run 另需 MMEVLA_MOTION_PROV_RELAX）
    uv run scripts/evaluation/smoke_demo_inject.py verify \
        --store <前缀库根> --task BinFill --difficulty easy --episode 156 \
        --ckpt <绝对路径>/50000 --config mme_vla_suite_b128_80k
判定行：``SMOKE_RESET=PASS {...}`` / ``SMOKE_INJECT=PASS {...}``
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples/robomme"))


def k_demo_expected(es: int, stride: int, demo_min_real: int) -> int:
    if es < demo_min_real:
        return 0
    return (es - demo_min_real) // stride + 1


def reset_npy_path(out_dir: pathlib.Path, task: str, difficulty: str, episode: int) -> pathlib.Path:
    return out_dir / f"reset_frame_{task}_{difficulty}_{episode}.npy"


def cmd_export_reset(args) -> int:
    """client 环境：建干净 env，取 reset 帧，与前缀第 0 帧比一眼，存 npy。"""
    from demo_prefix import DemoPrefixStore
    from env_runner import SpecEnvRunner
    from robomme.injection_candidates import load_candidates, candidate_key
    import robomme

    store = DemoPrefixStore(args.store)
    prefix = store.get(args.task, args.difficulty, args.episode)
    d = int(prefix["D"])

    benchmark_root = pathlib.Path(robomme.__file__).resolve().parents[2]
    header, rows = load_candidates(pathlib.Path(args.candidates), repo_root=benchmark_root)
    row = {candidate_key(r): r for r in rows}[(args.task, args.difficulty, args.episode)]
    sampling = header["sampling_config"]
    per_task = {"parameters": sampling["parameters"][args.task],
                "positions": sampling["positions"][args.task]}

    runner = SpecEnvRunner(args.task, args.difficulty, per_task,
                           str(REPO / "v1-store/tmp-smoke"), max_steps=2000)
    t0 = time.perf_counter()
    try:
        runner.make_env(row)
        clean = runner.get_init_obs()
        seconds = time.perf_counter() - t0
        # BinFill 的 task_list 里 demonstration 三处全 False ⇒ 干净 env 的 reset 只返回 1 帧复位观测
        assert len(clean["images"]) == 1, f"干净 env reset 返回 {len(clean['images'])} 帧，期望 1"
        assert clean["task_goal"] == prefix["task_goal"], (
            f"task_goal 与前缀不符：\n  前缀 {prefix['task_goal']!r}\n  本次 {clean['task_goal']!r}")
        reset_frame = np.ascontiguousarray(clean["images"][0], dtype=np.uint8)
        reset_state = np.ascontiguousarray(clean["states"][0], dtype=np.float32)
    finally:
        try:
            runner.close_env()
        except Exception as exc:
            print(f"关闭环境失败：{exc}", flush=True)

    # demo 段第 0 帧 vs exec 段第 0 帧：训练侧这两者逐位恒等（整条轨迹复制而来）；
    # 评测侧是两个 env 各渲染一次、不可能逐位相同，但应当极接近。计划点名「最值得先看一眼」的证据。
    diff = np.abs(prefix["images"][0].astype(np.int32) - reset_frame.astype(np.int32))
    report = {
        "task": args.task, "difficulty": args.difficulty, "episode": args.episode, "D": d,
        "clean_reset_frames": 1, "clean_reset_seconds": round(seconds, 1),
        "task_goal": prefix["task_goal"],
        "frame0_vs_reset": {"mean_abs": float(diff.mean()), "max_abs": int(diff.max()),
                            "pct_nonzero": float((diff > 0).mean() * 100)},
    }
    out_dir = pathlib.Path(args.out or (REPO / "v1-store/demo-prefix/smoke-inject"))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(reset_npy_path(out_dir, args.task, args.difficulty, args.episode), reset_frame)
    np.save(out_dir / f"reset_state_{args.task}_{args.difficulty}_{args.episode}.npy", reset_state)
    (out_dir / f"reset_{args.task}_{args.difficulty}_{args.episode}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SMOKE_RESET=PASS " + json.dumps(report, ensure_ascii=False), flush=True)
    print(f"[smoke] reset 帧与报告已写入 {out_dir}", flush=True)
    return 0


def cmd_verify(args) -> int:
    """server 环境：读前缀 + reset 帧，建 policy，喂注入后的首批并逐项断言。"""
    import jax
    from demo_prefix import DemoPrefixStore
    from mme_vla_suite.training import config as _config
    from mme_vla_suite.policies import policy_config as _policy_config
    from mme_vla_suite.shared.sampling import memory_order, pad_times

    if jax.default_backend() != "gpu":
        raise SystemExit(f"错误: jax 后端 {jax.default_backend()} != gpu")
    # sidecar 卡号与 run_shard.sh 同一口径：跟随本进程可见的第一张卡
    os.environ.setdefault("MMEVLA_MOTION_ONLINE_GPU",
                          (os.environ.get("CUDA_VISIBLE_DEVICES", "") or "0").split(",")[0] or "0")

    out_dir = pathlib.Path(args.out or (REPO / "v1-store/demo-prefix/smoke-inject"))
    npy = reset_npy_path(out_dir, args.task, args.difficulty, args.episode)
    if not npy.is_file():
        raise SystemExit(f"缺 reset 帧 {npy}——先在 client 环境跑 export-reset")
    reset_frame = np.ascontiguousarray(np.load(npy), dtype=np.uint8)
    state_npy = out_dir / f"reset_state_{args.task}_{args.difficulty}_{args.episode}.npy"
    reset_state = np.ascontiguousarray(np.load(state_npy), dtype=np.float32)

    store = DemoPrefixStore(args.store)
    prefix = store.get(args.task, args.difficulty, args.episode)
    d = int(prefix["D"])

    # 注入后的首批：D 帧前缀 + 1 帧干净 env 的 reset 观测，长度 D+1 ⇒ exec_start_idx = D
    images = np.stack([*prefix["images"], reset_frame])[:, None]
    states = np.stack([*prefix["states"], reset_state]).astype(np.float32)
    es = images.shape[0] - 1
    assert es == d, f"exec_start_idx {es} != D {d}"

    report: dict = {"task": args.task, "difficulty": args.difficulty, "episode": args.episode,
                    "D": d, "exec_start_idx": es, "first_batch_frames": int(images.shape[0]),
                    "enc_chunk": os.environ.get("MMEVLA_ENC_CHUNK", "0"),
                    "motion_overflow_env": os.environ.get("MMEVLA_MOTION_OVERFLOW", "")}

    # ⚠ 绝不能用 motion_stub=True：stub sidecar 靠 P.stub_decode(window) 从**像素**解帧编号，
    # 只认 P.stub_frame(j) 那种合成的带编号帧；真实渲染帧解出来是乱的，必报
    # 「33 帧编号不符合连续前缀与重复末帧规则」。真实帧只能走真 sidecar——正式评测本来也是这么跑的。
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), pathlib.Path(args.ckpt), motion_stub=args.motion_stub)
    mem = policy.mem_buffer
    report["motion_stub"] = bool(args.motion_stub)
    report["motion_enabled"] = bool(policy.motion_enabled)
    report["motion_overflow_effective"] = getattr(mem, "motion_overflow", None)

    t0 = time.perf_counter()
    policy.add_buffer({"images": images, "state": states, "exec_start_idx": es})
    report["first_add_buffer_seconds"] = round(time.perf_counter() - t0, 1)

    # 帧路：D+1 条特征，键恰好 0..D
    assert set(mem._history_feats) == set(range(es + 1)), "帧路键集合不是 0..D"
    report["history_feats"] = len(mem._history_feats)

    if policy.motion_enabled:
        expected = k_demo_expected(es, mem.motion_stride, mem.demo_min_real)
        report["k_demo_expected"], report["k_demo_actual"] = expected, len(mem._history_feats_motion)
        assert len(mem._history_feats_motion) == expected, "demo 窗数与公式不符"
        assert sorted(mem._history_feats_motion) == [mem.motion_stride * m for m in range(expected)], \
            "demo 窗起点不是 0,16,32,…"
        assert len(mem.visible_motion_frames(es)) == expected, "首次 infer 的可见窗数与公式不符"
        # 首批后原始帧只该留下索引 D 那一条（demo 段编完即可丢，exec 段游标还在 0）
        report["raw_frames_after_first"] = sorted(mem._raw_frames)
        assert sorted(mem._raw_frames) == [es], \
            f"_raw_frames 首批后剩 {sorted(mem._raw_frames)}，期望 [{es}]"

        # mem_order 在首批之后就能验，不必喂 exec 批：此刻 step_idx == es，帧路与运动路都已就绪
        _, _, m_mask, m_times = mem._prepare_motion(policy.step_idx)
        frames = mem.get_frame_sampling_indices(
            policy.step_idx, policy.config.budget, policy.config.token_per_image)
        max_frames = policy.config.budget // (policy.config.token_per_image * policy.config.num_views)
        order = memory_order(pad_times(frames, max_frames),
                             policy.config.token_per_image * policy.config.num_views, m_times)
        expected_order = policy.config.budget + mem.motion_budget
        assert len(order) == expected_order, f"mem_order 长度 {len(order)} != {expected_order}"
        report["mem_order_len"] = int(len(order))
        report["frame_path_indices"] = len(frames)
        assert int(m_mask.sum()) == expected, \
            f"motion mask 真实位 {int(m_mask.sum())} != demo 窗数 {expected}"

    # 再喂若干 16 帧的 exec 批，验窗数随步数增长、mem_order 是合法置换、降级按闭式触发
    downsample_first_at = None
    for _ in range(args.steps):
        batch = np.stack([reset_frame] * 16)[:, None]
        batch_states = np.repeat(states[-1:], 16, axis=0)
        before = getattr(mem, "motion_downsample_steps", 0)
        policy.add_buffer({"images": batch, "state": batch_states, "exec_start_idx": 0})
        if policy.motion_enabled:
            _, _, m_mask, m_times = mem._prepare_motion(policy.step_idx)
            frames = mem.get_frame_sampling_indices(
                policy.step_idx, policy.config.budget, policy.config.token_per_image)
            max_frames = policy.config.budget // (policy.config.token_per_image * policy.config.num_views)
            order = memory_order(pad_times(frames, max_frames),
                                 policy.config.token_per_image * policy.config.num_views, m_times)
            expected_order = policy.config.budget + mem.motion_budget
            assert len(order) == expected_order, f"mem_order 长度 {len(order)} != {expected_order}"
            k_visible = len(mem.visible_motion_frames(policy.step_idx))
            expected_true = min(k_visible, mem.motion_budget)
            assert int(m_mask.sum()) == expected_true, \
                f"motion mask 真实位 {int(m_mask.sum())} != {expected_true}（可见 {k_visible}）"
            if downsample_first_at is None and mem.motion_downsample_steps > before:
                downsample_first_at = int(policy.step_idx)
            report["mem_order_len"] = int(len(order))
    report["steps_fed"] = args.steps
    report["step_idx_final"] = int(policy.step_idx)
    if policy.motion_enabled:
        report["k_total_final"] = len(mem.visible_motion_frames(policy.step_idx))
        report["motion_downsample_steps"] = int(mem.motion_downsample_steps)
        report["downsample_first_at_step_idx"] = downsample_first_at

    print("SMOKE_INJECT=PASS " + json.dumps(report, ensure_ascii=False), flush=True)
    (out_dir / f"inject_{args.task}_{args.difficulty}_{args.episode}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("export-reset", "verify"))
    parser.add_argument("--store", required=True)
    parser.add_argument("--task", default="BinFill")
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--ckpt", default="", help="verify 模式必填")
    parser.add_argument("--config", default="mme_vla_suite_b128_80k")
    parser.add_argument("--candidates", default=str(
        REPO / "third_party/robomme_benchmark/artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl"))
    # 默认 0：motion 的 stub 编码器会校验每个 33 帧窗是「连续前缀 + 重复末帧」，
    # 而这里能合成的 exec 帧只有重复的同一张，33 帧编号全同、必被 stub 拒绝。
    # 首批之后 mem_order / demo 窗数 / 缓冲淘汰就都验得到了，不需要 exec 批；
    # 要喂 exec 批得起真 sidecar，或改用不开 motion 的 checkpoint。
    parser.add_argument("--steps", type=int, default=0,
                        help="verify 模式再喂几个 16 帧的 exec 批（stub sidecar 下必须为 0，见代码注释）")
    parser.add_argument("--motion-stub", action="store_true",
                        help="用 stub sidecar。只对合成的带编号帧有效，喂真实帧必报错，"
                             "所以本脚本默认关闭、走与正式评测同一条的真 sidecar")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.mode == "export-reset":
        return cmd_export_reset(args)
    if not args.ckpt:
        raise SystemExit("verify 模式需要 --ckpt")
    return cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
