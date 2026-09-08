#!/usr/bin/env python3
"""motion-variance 三条件 policy server（normal / mask / swap；official 走主线 scripts/training/serve_policy.py）。

骨架抄 scripts/training/g0/serve_policy_probe.py：起同一个 MME_VLA_Policy，在**实例属性**上遮蔽，主线 src/ 零改动。
  - 构造：mv_model_adapter.build_policy 把 MotionEncoderClient 换成工厂——normal 起真 sidecar 在 --motion-gpu；
    mask / swap 用零向量句柄（不起任何子进程；mask 由 MV_MASK_CONTENT_NULL 证明内容无关，swap 的内容整体被 donor 覆盖）。
  - 采样：三条件一律走 MVAdapter.f_sample（apply_gate=True 的同一份编译产物）；mask 给 motion 列门控，其余 gate=全 True。
  - swap：遮蔽 policy._prepare_history，在八键装好之后只覆盖 motion_emb 的有效行；motion_pos / motion_mask / mem_order 断言 sha 不变。
  - 自证：MV_SERVE_ENV（启动）/ MV_EPISODE_BEGIN expect=<task>:<ep>（reset；来自 mv_common.episode_plan）/ MV_INFER（每次推理，含 act_sha）/ MV_EPISODE_END。
    首次 MV_INFER 的 mv_task/mv_episode 与 expect 不符即 raise（抓续评错位）。
  - --probe-out：每次 infer 一行 jsonl；--dump-obs-dir：按 infer_seq 抽样落盘八键 + 原图（阶段 1 test 分布核对用）。

用法：
  CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
    UV_LINK_MODE=copy PYTHONUNBUFFERED=1 uv run --no-sync python scripts/motion-variance/serve_policy_mv.py \
      --cond swap --split test --seed 42 --port 9300 --tasks ButtonUnmask,VideoUnmask,ButtonUnmaskSwap,VideoUnmaskSwap \
      --ep-start 3 --ep-stride 8 --swap-bank v1-store/reports/motion-variance/bank-lib --probe-out v1-store/logs/x.probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import socket
import subprocess
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import mv_common as C  # noqa: E402
from mv_model_adapter import MVAdapter, ZeroMotion, build_policy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(C.CKPT_MOTION))
    ap.add_argument("--config", default=C.TRAIN_CONFIG)
    ap.add_argument("--cond", choices=("normal", "mask", "swap"), required=True)
    ap.add_argument("--split", choices=C.SPLITS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--ep-start", type=int, required=True)
    ap.add_argument("--ep-stride", type=int, required=True)
    ap.add_argument("--ep-count", type=int, default=0)
    ap.add_argument("--motion-gpu", default=None, help="cond=normal：真 sidecar 的绝对卡号")
    ap.add_argument("--swap-bank", default=None, help="cond=swap：donor bank 目录")
    ap.add_argument("--probe-out", default=None)
    ap.add_argument("--dump-obs-dir", default=None)
    ap.add_argument("--dump-obs-every", type=int, default=0)
    ap.add_argument("--dump-obs-max", type=int, default=200)
    args = ap.parse_args()

    import jax
    from mme_vla_suite.serving import websocket_policy_server
    from mme_vla_suite.training import config as _config
    import mme_vla_suite.policies.motion_client as mc_mod

    C.require_gpu()
    C.require_det_flags()
    ckpt_dir = pathlib.Path(args.ckpt).resolve()
    plan = C.episode_plan(args.tasks, args.ep_start, args.ep_stride, C.EPISODES_PER_TASK, args.ep_count)
    bank = None
    if args.cond == "swap":
        if not args.swap_bank:
            raise SystemExit("错误: cond=swap 必须给 --swap-bank")
        from mv_donor_bank import DonorBank
        bank = DonorBank(pathlib.Path(args.swap_bank))
        if bank.source != "library":
            raise SystemExit(f"错误: bank source={bank.source}，本轮只接受 library")
    if args.cond == "normal" and args.motion_gpu is None:
        raise SystemExit("错误: cond=normal 必须给 --motion-gpu")

    real_client = mc_mod.MotionEncoderClient

    def motion_factory(**kw):
        if args.cond == "normal":
            return real_client(online_gpu=args.motion_gpu, stub=False, store_provenance=kw.get("store_provenance"),
                               expected_ckpt_sha256=kw.get("expected_ckpt_sha256"))
        return ZeroMotion()

    t0 = time.perf_counter()
    train_config = _config.get_config(args.config)
    policy = build_policy(train_config, ckpt_dir, seed=args.seed, motion_factory=motion_factory)
    if not policy.motion_enabled:
        raise SystemExit("错误: 被评 checkpoint 不是 motion 开启态")
    adapter = MVAdapter(policy)
    fns = adapter.make_sample_fns()
    inner_sample = fns["mask"] if args.cond == "mask" else fns["none"]
    gate_name = "mask_all" if args.cond == "mask" else "none"
    head = subprocess.run(["git", "-C", str(C.REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"MV_SERVE_ENV cond={args.cond} split={args.split} seed={args.seed} ckpt={ckpt_dir} HEAD={head} "
          f"sidecar={'on' if args.cond == 'normal' else 'off'} motion_gpu={args.motion_gpu} "
          f"bank_sha={bank.sha256[:16] if bank else 'none'} bank_source={bank.source if bank else 'none'} "
          f"xla_flags={os.environ.get('XLA_FLAGS', '')!r} backend={jax.default_backend()} "
          f"tasks={args.tasks} ep_start={args.ep_start} ep_stride={args.ep_stride} ep_count={args.ep_count} plan_len={len(plan)} "
          f"sample_fn={inner_sample.__name__} gate={gate_name} num_steps=10 build_s={time.perf_counter() - t0:.1f}", flush=True)

    fh = open(args.probe_out, "w", encoding="utf-8") if args.probe_out else None
    dump_dir = pathlib.Path(args.dump_obs_dir) if args.dump_obs_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
    st = {"ep_seq": 0, "infer_seq": 0, "pending": None, "act_sha": None, "expect": None, "dumped": 0,
          "ep_infers": 0, "ep_windows": 0, "ep_t0": 0.0}
    _orig_reset = policy.reset
    _orig_prepare = policy._prepare_history
    _orig_infer = policy.infer

    def mv_reset() -> None:
        if st["ep_seq"] > 0:
            print(f"MV_EPISODE_END ep_seq={st['ep_seq']} infers={st['ep_infers']} windows={st['ep_windows']} "
                  f"wall_s={time.perf_counter() - st['ep_t0']:.1f}", flush=True)
        _orig_reset()
        st["ep_seq"] += 1
        st["infer_seq"] = 0
        st["ep_infers"] = 0
        st["ep_windows"] = 0
        st["ep_t0"] = time.perf_counter()
        i = st["ep_seq"] - 1
        st["expect"] = plan[i] if i < len(plan) else None
        exp = f"{st['expect'][0]}:{st['expect'][1]}" if st["expect"] else "BEYOND_PLAN"
        print(f"MV_EPISODE_BEGIN cond={args.cond} split={args.split} seed={args.seed} ep_seq={st['ep_seq']} expect={exp}", flush=True)
        if st["expect"] is None:
            raise RuntimeError(f"reset 次数 {st['ep_seq']} 超出 episode_plan 长度 {len(plan)}（续评 / 分片参数与 serve 端不一致）")

    def mv_prepare_history(inputs: dict) -> dict:
        task = inputs.pop("mv_task", None)
        ep = inputs.pop("mv_episode", None)
        if task is None or ep is None:
            raise RuntimeError("element 缺 mv_task / mv_episode（评估客户端不是 scripts/motion-variance/robomme/eval.py？）")
        ep = int(ep)
        if st["infer_seq"] == 0 and (task, ep) != tuple(st["expect"]):
            raise RuntimeError(f"ep_seq={st['ep_seq']} 期望 {st['expect']}，客户端实际评的是 ({task}, {ep})——分片参数或续评错位")
        out = _orig_prepare(inputs)
        t = int(policy.step_idx)
        es = int(policy.exec_start_idx)
        frames = [int(x) for x in policy.mem_buffer.visible_motion_frames(t)]
        k = len(frames)
        cover = {"exact": 0, "fallback": 0, "cycle": 0, "cross_seg": 0}
        donor = None
        pos_sha, mask_sha, order_sha = (C.leaf_sha256(np.asarray(out[key])) for key in ("motion_pos", "motion_mask", "mem_order"))
        if args.cond == "swap":
            emb = np.array(out["motion_emb"], dtype=np.float32, copy=True)
            for i, f in enumerate(frames):
                seg, m = ("demo", f // C.GRID_STRIDE) if f < es else ("exec", (f - es) // C.GRID_STRIDE)
                tok, how, g = bank.lookup(args.split, task, ep, seg, m)
                emb[i] = tok
                cover[how] += 1
                donor = g if donor is None else donor
            out["motion_emb"] = emb
            for key, sha in (("motion_pos", pos_sha), ("motion_mask", mask_sha), ("mem_order", order_sha)):
                if C.leaf_sha256(np.asarray(out[key])) != sha:
                    raise RuntimeError(f"swap 只准改 motion_emb，{key} 变了")
        mo = np.asarray(out["mem_order"])
        st["infer_seq"] += 1
        st["ep_infers"] += 1
        st["ep_windows"] = max(st["ep_windows"], k)
        st["pending"] = {
            "cond": args.cond, "split": args.split, "seed": args.seed, "task": task, "ep": ep,
            "ep_seq": st["ep_seq"], "infer_seq": st["infer_seq"], "t": t, "es": es, "k": k, "motion_frames": frames,
            "motion_emb_sha": C.leaf_sha256(np.asarray(out["motion_emb"])),
            "motion_pos_sha": pos_sha, "motion_mask": C.mask_bits(out["motion_mask"]), "mem_order_sha": order_sha,
            "motion_cols_sha": C.leaf_sha256(C.motion_col_mask_np(mo)),
            "static_mask_sha": C.leaf_sha256(np.asarray(out["static_mask"])),
            "donor": donor, "cover": cover, "gate": gate_name, "num_steps": 10, "prompt": inputs.get("prompt")}
        if dump_dir and args.dump_obs_every > 0 and st["infer_seq"] % args.dump_obs_every == 0 and st["dumped"] < args.dump_obs_max:
            arrs = {key: np.asarray(out[key]) for key in C.KEYS8}
            arrs.update({"image": np.asarray(inputs["observation/image"]), "wrist_image": np.asarray(inputs["observation/wrist_image"]),
                         "state": np.asarray(inputs["observation/state"]), "t": np.int64(t), "es": np.int64(es), "k": np.int64(k),
                         "frames": np.asarray(frames, np.int64), "prompt": np.array(str(inputs.get("prompt")))})
            np.savez(dump_dir / f"obs_{args.split}_{task}_ep{ep}_t{t}.npz", **arrs)
            st["dumped"] += 1
        return out

    def mv_sample(rng, observation, **kw):
        a = inner_sample(rng, observation, **kw)
        jax.block_until_ready(a)
        st["act_sha"] = C.leaf_sha256(np.asarray(a))
        return a

    def mv_infer(obs: dict) -> dict:
        out = _orig_infer(obs)
        rec = st["pending"]
        st["pending"] = None
        if rec is None:
            raise RuntimeError("infer 结束时没有 pending 记录（_prepare_history 未被调用？）")
        rec["act_sha"] = st["act_sha"]
        rec["infer_ms"] = float(out.get("infer_time_ms", 0.0))
        cov = rec["cover"]
        print(f"MV_INFER cond={rec['cond']} split={rec['split']} task={rec['task']} ep={rec['ep']} ep_seq={rec['ep_seq']} "
              f"infer_seq={rec['infer_seq']} t={rec['t']} es={rec['es']} k={rec['k']} "
              f"motion_frames={','.join(map(str, rec['motion_frames'])) or '-'} motion_emb_sha={rec['motion_emb_sha'][:16]} "
              f"motion_pos_sha={rec['motion_pos_sha'][:16]} motion_mask={rec['motion_mask']} mem_order_sha={rec['mem_order_sha'][:16]} "
              f"motion_cols_sha={rec['motion_cols_sha'][:16]} donor={rec['donor']} "
              f"cover=exact:{cov['exact']},fallback:{cov['fallback']},cycle:{cov['cycle']},cross_seg:{cov['cross_seg']} "
              f"gate={rec['gate']} num_steps=10 act_sha={rec['act_sha'][:16]} infer_ms={rec['infer_ms']:.1f}", flush=True)
        if fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        return out

    policy.reset = mv_reset
    policy._prepare_history = mv_prepare_history
    policy._sample_actions = mv_sample
    policy.infer = mv_infer

    hostname = socket.gethostname()
    logging.info("Creating motion-variance server (host: %s, ip: %s)", hostname, socket.gethostbyname(hostname))
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata)
    try:
        server.serve_forever()
    finally:
        if fh:
            fh.close()
        policy._motion_client.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    sys.exit(main())
