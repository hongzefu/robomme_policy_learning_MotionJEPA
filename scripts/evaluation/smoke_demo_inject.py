"""注入侧冒烟核对（0922-binfill-demo-prefix-plan.md 冒烟节）。

评测是 client/server 分离的，客户端看不到 ``FrameSampMemory`` 内部状态，所以计划列的那批断言
（``exec_start_idx``、``_history_feats`` 键集合、motion 窗数、``_raw_frames`` 淘汰、``mem_order`` 置换）
只能在**一个进程里**直接建 policy 来验。本脚本因此不连 server，而是自己 ``create_trained_policy``。

顺带回答两个只能实测的问题：**D+1 帧一次性过 SigLIP 会不会 OOM、耗时多少**。
motion 侧用 stub 编码器（``motion_stub=True``）——窗数、键集合、缓冲淘汰这些结构性的东西与真编码器
逐一致，而 stub 省掉起 sidecar 的开销；本脚本不验 motion token 的数值。

还会把干净 env 的 reset 帧存成 npy，供 ``enc_chunk_cmp.py --reset-frame-npy`` 用——
那一帧是 exec 段第 0 帧、有历史口径，必须进分批对照的覆盖点。

用法（server 环境，需 GPU）：
    uv run scripts/evaluation/smoke_demo_inject.py \
        --store <demo 前缀库根> --task BinFill --difficulty easy --episode 156 \
        --ckpt <绝对路径>/50000 --config mme_vla_suite_b128_80k --steps 3
判定行：SMOKE_INJECT=PASS {...}
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True)
    parser.add_argument("--task", default="BinFill")
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="mme_vla_suite_b128_80k")
    parser.add_argument("--candidates", default=str(
        REPO / "third_party/robomme_benchmark/artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl"))
    parser.add_argument("--steps", type=int, default=3, help="注入后再喂几个 16 帧的 exec 批")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import jax
    from unittest import mock
    from demo_prefix import DemoPrefixStore
    from env_runner import SpecEnvRunner
    from mme_vla_suite.training import config as _config
    from mme_vla_suite.policies import policy_config as _policy_config
    from mme_vla_suite.policies import motion_client as mc
    from robomme.injection_candidates import load_candidates, candidate_key
    import robomme

    if jax.default_backend() != "gpu":
        raise SystemExit(f"错误: jax 后端 {jax.default_backend()} != gpu")

    report: dict = {"task": args.task, "difficulty": args.difficulty, "episode": args.episode}

    # ── ① 读前缀 ─────────────────────────────────────────────────────────────
    store = DemoPrefixStore(args.store)
    prefix = store.get(args.task, args.difficulty, args.episode)
    d = int(prefix["D"])
    report["D"] = d

    # ── ② 建一个干净 env（demonstration 仍为 False，不装 monkey-patch），取它的 reset 帧 ──
    benchmark_root = pathlib.Path(robomme.__file__).resolve().parents[2]
    header, rows = load_candidates(pathlib.Path(args.candidates), repo_root=benchmark_root)
    index = {candidate_key(r): r for r in rows}
    row = index[(args.task, args.difficulty, args.episode)]
    sampling = header["sampling_config"]
    per_task = {"parameters": sampling["parameters"][args.task],
                "positions": sampling["positions"][args.task]}
    runner = SpecEnvRunner(args.task, args.difficulty, per_task, str(REPO / "v1-store/tmp-smoke"),
                           max_steps=2000)
    t0 = time.perf_counter()
    runner.make_env(row)
    clean = runner.get_init_obs()
    report["clean_reset_frames"] = len(clean["images"])
    report["clean_reset_seconds"] = round(time.perf_counter() - t0, 1)
    # 干净 env 的 reset 只返回 1 帧复位观测——BinFill 的 demonstration 三处全 False
    assert len(clean["images"]) == 1, f"干净 env reset 返回 {len(clean['images'])} 帧，期望 1"
    assert clean["task_goal"] == prefix["task_goal"], "task_goal 与前缀不符"
    reset_frame = np.ascontiguousarray(clean["images"][0], dtype=np.uint8)

    # ── ③ demo 段第 0 帧 vs exec 段第 0 帧：同一场景初始态的两次渲染 ──────────────
    # 训练侧「demo 段第 0 帧 == exec 段第 0 帧」是逐位恒等（整条轨迹复制而来）；
    # 评测侧是两个 env 各渲染一次，不可能逐位相同，但应当极接近。这是最值得先看一眼的一致性证据。
    diff = np.abs(prefix["images"][0].astype(np.int32) - reset_frame.astype(np.int32))
    report["frame0_vs_reset"] = {"mean_abs": float(diff.mean()), "max_abs": int(diff.max()),
                                 "pct_nonzero": float((diff > 0).mean() * 100)}
    runner.close_env()
    del runner

    # ── ④ 注入后的首批：D 帧前缀 + 1 帧 reset 观测 ────────────────────────────
    images = np.stack([*prefix["images"], reset_frame])[:, None]        # (D+1,1,256,256,3)
    states = np.stack([*prefix["states"], clean["states"][0]]).astype(np.float32)
    es = images.shape[0] - 1
    report["exec_start_idx"] = es
    assert es == d, f"exec_start_idx {es} != D {d}"

    original = mc.MotionEncoderClient
    with mock.patch.object(mc, "MotionEncoderClient",
                           lambda **kw: original(**(kw | {"online_gpu": ""}))):
        policy = _policy_config.create_trained_policy(
            _config.get_config(args.config), pathlib.Path(args.ckpt), motion_stub=True)
    mem = policy.mem_buffer
    report["motion_enabled"] = bool(policy.motion_enabled)
    report["enc_chunk"] = os.environ.get("MMEVLA_ENC_CHUNK", "0")
    report["motion_overflow"] = getattr(mem, "motion_overflow", None)

    t0 = time.perf_counter()
    policy.add_buffer({"images": images, "state": states, "exec_start_idx": es})
    report["first_add_buffer_seconds"] = round(time.perf_counter() - t0, 1)
    report["first_batch_frames"] = int(images.shape[0])

    # 帧路：D+1 条特征，键恰好 0..D
    assert set(mem._history_feats) == set(range(es + 1)), "帧路键集合不是 0..D"
    report["history_feats"] = len(mem._history_feats)

    if policy.motion_enabled:
        expected = k_demo_expected(es, mem.motion_stride, mem.demo_min_real)
        report["k_demo_expected"] = expected
        report["k_demo_actual"] = len(mem._history_feats_motion)
        assert len(mem._history_feats_motion) == expected, "demo 窗数与公式不符"
        assert sorted(mem._history_feats_motion) == [mem.motion_stride * m for m in range(expected)], \
            "demo 窗起点不是 0,16,32,…"
        assert len(mem.visible_motion_frames(es)) == expected, "首次 infer 的可见窗数与公式不符"
        # 首批后原始帧只该留下索引 D 那一条（demo 段编完即可丢，exec 段游标还在 0）
        report["raw_frames_after_first"] = sorted(mem._raw_frames)
        assert sorted(mem._raw_frames) == [es], f"_raw_frames 首批后剩 {sorted(mem._raw_frames)}，期望 [{es}]"

    # ── ⑤ 再喂几个 exec 批，验窗数随步数增长、mem_order 是合法置换 ──────────────
    from mme_vla_suite.shared.sampling import memory_order, pad_times
    rng = np.random.default_rng(0)
    orders = []
    for step in range(args.steps):
        batch = np.stack([reset_frame] * 16)[:, None]
        batch_states = np.repeat(states[-1:], 16, axis=0)
        policy.add_buffer({"images": batch, "state": batch_states, "exec_start_idx": 0})
        if policy.motion_enabled:
            m_emb, m_pos, m_mask, m_times = mem._prepare_motion(policy.step_idx)
            frames = mem.get_frame_sampling_indices(
                policy.step_idx, policy.config.budget, policy.config.token_per_image)
            max_frames = policy.config.budget // (policy.config.token_per_image * policy.config.num_views)
            order = memory_order(pad_times(frames, max_frames),
                                 policy.config.token_per_image * policy.config.num_views, m_times)
            orders.append(len(order))
            assert int(m_mask.sum()) == len(mem.visible_motion_frames(policy.step_idx)), \
                "motion mask 的真实位数与可见窗数不符"
    report["steps_fed"] = args.steps
    report["step_idx_final"] = int(policy.step_idx)
    if orders:
        expected_order = policy.config.budget + mem.motion_budget
        report["mem_order_len"] = orders[-1]
        assert all(n == expected_order for n in orders), \
            f"mem_order 长度 {orders} != budget+motion_budget = {expected_order}"
    if policy.motion_enabled:
        report["k_total_final"] = len(mem.visible_motion_frames(policy.step_idx))
        report["motion_downsample_steps"] = int(mem.motion_downsample_steps)

    print("SMOKE_INJECT=PASS " + json.dumps(report, ensure_ascii=False), flush=True)
    out = args.out or str(REPO / "v1-store/demo-prefix/smoke-inject")
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    np.save(pathlib.Path(out) / f"reset_frame_{args.task}_{args.difficulty}_{args.episode}.npy", reset_frame)
    (pathlib.Path(out) / f"report_{args.task}_{args.difficulty}_{args.episode}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[smoke] reset 帧与报告已写入 {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
