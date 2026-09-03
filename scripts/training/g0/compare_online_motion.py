#!/usr/bin/env python3
"""P5：真编码器在线链 vs 离线 motion 表（motion-memory-plan.md 四节表一 P5；run `motion-p5-online`）。

驱动：从录制 h5 按 `episode_manifest.json` 逐 episode 读全部 `front_rgb` (256,256,3) uint8 帧，沿 eval.py 真实节奏喂
`FrameSampMemory.add_buffer`（首批 pre_traj = demo [0, es) + exec 首帧、之后每批 16 帧、`exec_start_idx` 下传），运动路编码走真
`motion_sidecar.py`（wan 子 venv、fp32 / 关 TF32 / B=1 / 33 帧一次喂，独占 `--gpu` 那张卡）。40 条 episode 全跑。

五条判据 + provenance + 三笔耗时：
  ONLINE_ENC_BITEXACT=PASS compared=772 mismatches=0   每窗在线 (768,) 与离线表对应行 `np.array_equal`，772 行全覆盖
  ONLINE_START_SET=PASS steps=N                        每个推理时刻在线起点集合 == `motion_store.visible_motion_rows`
  ONLINE_POS=PASS                                      `motion_pos` 行 == `FrameSampStore.pos_rows([f])[0, 0, :256]` 逐位
  ONLINE_ORDER=PASS steps=N                            `mem_order`（含 motion_emb/pos/mask 四键）== 训练侧 `FrameSampDataset.__getitem__` 逐位，且为合法置换
  PROVENANCE=PASS                                      sidecar 握手 provenance 与 store_meta.provenance 逐键相等（客户端构造时已 raise 兜底）
  耗时：每窗（客户端夹 send/recv）、首批 demo（首次 add_buffer 挂钟）、每次推理前固定开销（后续每批 add_buffer 挂钟，含帧路 SigLIP 编码）
`--stub` 档：帧换成编号合成帧、只验起点集合 / pos / order 三条（token 判据标 SKIP），供 CPU 上先验证脚本本身。

用法（主树）：
  UV_LINK_MODE=copy uv run --no-sync python scripts/training/g0/compare_online_motion.py --gpu 1 --out v1-store/reports/motion/p5_online.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "training" / "tests"))
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))

from mme_vla_suite.datastore import motion_store as ms          # noqa: E402
from mme_vla_suite.datastore.framesamp_store import StoreMeta   # noqa: E402
from mme_vla_suite.policies.framesamp_memory import FrameSampMemory  # noqa: E402
from mme_vla_suite.policies.motion_client import MotionEncoderClient  # noqa: E402
from mme_vla_suite.shared.sampling import even_sampling_indices  # noqa: E402

HIST_BUDGET, TOKEN_PER_IMAGE, NUM_VIEWS, POS_DIM = 512, 16, 1, 256


def load_episode(raw_dir: pathlib.Path, h5_file: str, raw_ep_idx: int, T: int):
    import h5py
    frames, states = [], []
    with h5py.File(raw_dir / h5_file, "r") as f:
        g = f[f"episode_{raw_ep_idx}"]
        ts_ids = sorted(int(k.split("_")[-1]) for k in g.keys() if k.startswith("timestep_"))
        if len(ts_ids) != T or ts_ids != list(range(T)):
            raise SystemExit(f"错误: {h5_file} episode_{raw_ep_idx} timesteps {len(ts_ids)} != 清单 {T}")
        for t in ts_ids:
            ts = g[f"timestep_{t}"]
            img = ts["obs"]["front_rgb"][()]
            if img.shape != (256, 256, 3) or img.dtype != np.uint8:
                raise SystemExit(f"错误: 帧形制 {img.shape} {img.dtype}")
            frames.append(img)
            joint = ts["obs"]["joint_state"][()]
            grip = ts["obs"]["gripper_state"][()]
            states.append(np.concatenate([joint, grip[:1]], axis=0, dtype=np.float32))
    return np.stack(frames)[:, None], np.stack(states)   # (T,1,256,256,3) u8, (T,8) f32


class _Cfg:
    budget = HIST_BUDGET; token_per_image = TOKEN_PER_IMAGE; num_views = NUM_VIEWS


def _bare_policy(mem, motion_cfg, norm):
    from mme_vla_suite.policies.policy import MME_VLA_Policy
    pol = MME_VLA_Policy.__new__(MME_VLA_Policy)
    pol.config = _Cfg(); pol.mem_buffer = mem; pol.motion_enabled = True; pol._motion_cfg = motion_cfg
    pol.step_idx = -1; pol.exec_start_idx = 0; pol.use_quantiles = False; pol.state_norm_stats = norm
    return pol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", default=str(_V1 / "datasets/4task-motion-40ep"))
    ap.add_argument("--yaml", default="perceptual-framesamp-context-motion.yaml")
    ap.add_argument("--gpu", default="1", help="sidecar 的 CUDA_VISIBLE_DEVICES")
    ap.add_argument("--episodes", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--out", default=str(_V1 / "reports/motion/p5_online.json"))
    args = ap.parse_args()

    from motion_gates_model import _load_yaml, _make_dataset               # 与 M 系列同一 dataset 构造口径（含 MMEVLA_MOTION_STORE）
    from mme_vla_suite.training.dataloader import _motion_gates
    lib = pathlib.Path(args.lib)
    hc = _load_yaml(args.yaml)
    mcfg = hc.motion
    motion_cfg = {k: mcfg[k] for k in ("stride", "window_frames", "budget", "frame_size", "pos_dim", "dim", "window_direction", "grid_origin")}
    ds = _make_dataset(lib, args.yaml)
    fmeta = StoreMeta.load(str(lib / "framesamp"))
    motion_root = _motion_gates(hc, fmeta)
    mmeta = ms.MotionMeta.load(motion_root)
    mstore = ms.MotionStore(motion_root, meta=mmeta)
    fstore = ds._ensure_store()
    entries = ds._motion_entries
    manifest = json.load(open(lib / "meta" / "episode_manifest.json"))
    raw_dir = pathlib.Path(manifest["raw_dir"])
    norm = ds.state_norm_stats

    # 帧路用零特征（帧路数值一致性由 compare_online_memory.py 负责；本脚本只测运动路与交错次序，主进程不占 GPU）
    import jax.numpy as jnp

    def vision_enc(x):
        t, v = x.shape[:2]
        return jnp.zeros((t, v, 64, 2048), jnp.bfloat16)

    t0 = time.perf_counter()
    client = MotionEncoderClient(online_gpu=args.gpu, stub=args.stub,
                                 store_provenance={"vae": mmeta.provenance.get("vae"), "encoder": mmeta.provenance.get("encoder")},
                                 expected_ckpt_sha256=None if args.stub else mmeta.provenance["encoder"]["checkpoint_sha256"])
    startup_s = time.perf_counter() - t0
    print(f"[p5] sidecar 就绪 {startup_s:.1f}s stub={args.stub} warmup={client.provenance.get('warmup_s')}")

    rows_seen: set[int] = set()
    mismatches: list[dict] = []
    n_compared = 0
    n_steps = 0
    start_ok = pos_ok = order_ok = True
    per_ep = []
    eps = manifest["episodes"][: args.episodes or None]
    try:
        for ep in eps:
            g = int(ep["global_episode_idx"]); es = int(ep["exec_start_idx"]); T = int(ep["num_timesteps"])
            entry = entries[g]
            frames, states = load_episode(raw_dir, ep["h5_file"], int(ep["raw_ep_idx"]), T)
            if args.stub:
                # stub 档：帧换成把全域帧号写进像素的合成帧（sidecar --stub 解码校验连续性），状态与节奏保持真实
                from mme_vla_suite.policies import motion_protocol as P
                frames = np.stack([P.stub_frame(t) for t in range(T)])[:, None]
            mem = FrameSampMemory(vision_enc_fn=vision_enc, motion_enc_fn=client, motion_cfg=motion_cfg)
            pol = _bare_policy(mem, motion_cfg, norm)
            calls0, enc_s0 = client.n_calls, client.total_s
            batch_ms = []
            checked_rows: set[int] = set()

            def feed(lo, hi):
                tb = time.perf_counter()
                pol.add_buffer({"images": frames[lo:hi], "state": states[lo:hi], "exec_start_idx": es if lo == 0 else 0})
                batch_ms.append((time.perf_counter() - tb) * 1e3)

            def check(t):
                nonlocal n_compared, n_steps, start_ok, pos_ok, order_ok
                n_steps += 1
                rows_ref, f_ref = ms.visible_motion_rows(entry, t)
                got = mem.visible_motion_frames(t)
                if got != f_ref.tolist():
                    start_ok = False; mismatches.append({"g": g, "t": t, "kind": "start_set", "online": got[:8], "offline": f_ref[:8].tolist()})
                for r, f in zip(rows_ref.tolist(), f_ref.tolist()):
                    if r in checked_rows:
                        continue
                    checked_rows.add(r); rows_seen.add(r); n_compared += 1
                    if not args.stub:
                        on = mem._history_feats_motion[f]; off = mstore.rows(np.asarray([r]))[0]
                        if not np.array_equal(on, off):
                            mismatches.append({"g": g, "t": t, "kind": "token", "row": r, "f": f,
                                               "max_abs": float(np.max(np.abs(on - off))), "cos": float(np.dot(on, off) / (np.linalg.norm(on) * np.linalg.norm(off) + 1e-12))})
                inputs = pol._prepare_history({})
                k = len(f_ref)
                pos_ref = fstore.pos_rows(np.asarray(f_ref, np.int64))[:, 0, :POS_DIM] if k else np.zeros((0, POS_DIM), np.float32)
                if not (np.array_equal(inputs["motion_pos"][:k], pos_ref) and np.all(inputs["motion_pos"][k:] == 0)):
                    pos_ok = False; mismatches.append({"g": g, "t": t, "kind": "pos"})
                # 训练侧同一样本
                idx = np.flatnonzero((ds._epis_of == g) & (ds._step_of == t))
                if len(idx) != 1:
                    raise SystemExit(f"错误: dataset 中 (g={g}, t={t}) 样本数 {len(idx)}")
                d = ds[int(idx[0])]
                for key in ("motion_pos", "motion_mask", "mem_order"):
                    if not np.array_equal(np.asarray(inputs[key]), np.asarray(d[key])):
                        order_ok = False; mismatches.append({"g": g, "t": t, "kind": key})
                if not args.stub and not np.array_equal(inputs["motion_emb"], d["motion_emb"]):
                    order_ok = False; mismatches.append({"g": g, "t": t, "kind": "motion_emb_vs_dataset"})
                mo = np.asarray(inputs["mem_order"])
                if mo.dtype != np.int32 or not np.array_equal(np.sort(mo), np.arange(len(mo))):
                    order_ok = False; mismatches.append({"g": g, "t": t, "kind": "perm"})
                if args.stub:
                    for f in f_ref.tolist():
                        if not np.all(mem._history_feats_motion[f] == float(f)):
                            mismatches.append({"g": g, "t": t, "kind": "stub_token", "f": f})

            feed(0, es + 1); check(es)
            t = es
            while t + 16 < T:
                feed(t + 1, t + 17); t += 16; check(t)
            n_calls = client.n_calls - calls0
            per_ep.append({"g": g, "h5": ep["h5_file"], "raw_ep_idx": ep["raw_ep_idx"], "es": es, "T": T,
                           "windows": n_calls, "rows_checked": len(checked_rows),
                           "first_batch_ms": batch_ms[0], "first_batch_windows": ms.seg_num_grid(es),
                           "later_batch_ms_mean": float(np.mean(batch_ms[1:])) if len(batch_ms) > 1 else None,
                           "later_batch_ms_max": float(np.max(batch_ms[1:])) if len(batch_ms) > 1 else None,
                           "enc_ms_per_window": (client.total_s - enc_s0) / max(n_calls, 1) * 1e3})
            e = per_ep[-1]
            print(f"[p5] g={g:2d} es={es:3d} T={T:3d} 窗 {n_calls:3d} 行核 {len(checked_rows):3d} 首批 {e['first_batch_ms']:.0f} ms "
                  f"后续批均 {e['later_batch_ms_mean'] or 0:.0f} ms 每窗 {e['enc_ms_per_window']:.0f} ms 失配累计 {len(mismatches)}")
    finally:
        client.close()

    total_rows = mmeta.num_rows
    all_rows = len(rows_seen) == total_rows
    tok_mism = [m for m in mismatches if m["kind"] in ("token", "stub_token")]
    enc_line = ("ONLINE_ENC_BITEXACT=SKIP(stub)" if args.stub else
                f"ONLINE_ENC_BITEXACT={'PASS' if (not tok_mism and all_rows) else 'FAIL'}") + f" compared={n_compared} mismatches={len(tok_mism)} rows_total={total_rows} covered={len(rows_seen)}"
    lines = [enc_line,
             f"ONLINE_START_SET={'PASS' if start_ok else 'FAIL'} steps={n_steps}",
             f"ONLINE_POS={'PASS' if pos_ok else 'FAIL'}",
             f"ONLINE_ORDER={'PASS' if order_ok else 'FAIL'} steps={n_steps}",
             f"PROVENANCE={'SKIP(stub)' if args.stub else 'PASS'}",
             f"ENC_MS_PER_WINDOW mean={np.mean([e['enc_ms_per_window'] for e in per_ep]):.1f} "
             f"FIRST_BATCH_MS max={max(e['first_batch_ms'] for e in per_ep):.0f} "
             f"LATER_BATCH_MS mean={np.mean([e['later_batch_ms_mean'] for e in per_ep if e['later_batch_ms_mean'] is not None]):.1f}"]
    verdict = start_ok and pos_ok and order_ok and (args.stub or (not tok_mism and all_rows))
    lines.append(f"P5_ONLINE={'PASS' if verdict else 'FAIL'} episodes={len(per_ep)} stub={args.stub}")
    print("\n".join(lines))
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lines": lines, "per_episode": per_ep, "mismatches": mismatches[:200], "provenance": client.provenance,
                               "sidecar_startup_s": startup_s, "argv": sys.argv}, ensure_ascii=False, indent=1, default=str) + "\n")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
