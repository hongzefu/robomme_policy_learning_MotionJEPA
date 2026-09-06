#!/usr/bin/env python3
"""单 episode 开环重放：记忆帧 SigLIP 编码器多臂逐时刻对拍（审计 D-A1 / D-A2 的量化；1 张 GPU，不起 sidecar）。

五臂共用同一套预处理（`FrameSampMemory.add_buffer`：/255*2-1 → resize_with_pad(224) → enc → pool_tokens_to_size(16)）：
  S  = 训练库 framesamp store 的 image_emb_4x4 行（训练真值；离线建库 = 离线 f32 tokenizer 逐帧 batch=1 算出）
  A1 = 离线 `jax.jit(SigLipTokenizer().__call__)`（f32 pkl 权重），**逐帧 batch=1 喂**——应与 S 逐位（复现建库）；只走帧路，不参与装配/动作
  A  = 同一 f32 编码器，按 eval.py 节奏成批喂（首批 es+1 帧、之后每批 16）——与 S 之差 = 纯批形状效应（D-A2）
  B  = 在线 `policy._vision_encode`（checkpoint 内 bf16 `PaliGemma.img`），成批喂——生产推理用的；S vs B = 训练/推理真实差距
  C  = A 的权重 astype(bf16) 后成批喂——诊断臂：C == B 逐位 ⇒ A/B 之差完全由权重 dtype 解释
运动路各臂固定为训练库该 episode 的真 motion 行（`MotionStore.rows` 按起点帧号查表；P5 已证与真 sidecar 逐位相同）。

节奏复刻 `examples/robomme/eval.py`：首批 [0, es] 传 exec_start_idx=es，之后每批 16 帧传 0；每批后即一个 infer 时刻 t。
每个 t：
  帧级   逐位：A1 vs S、C vs B；数值：S vs B（真实差距）、S vs A（批形状）、A vs B（dtype）
  装配级 S 侧 `_prepare_history` 八键 vs `FrameSampDataset[idx(g,t)]`（逐位，证明 S 臂就是训练样本）
  动作级 `_sample_actions(key(0), obs, noise=固定)` 各臂 × 3 个 noise seed；S vs B / S vs A / A vs B 的 RMS 与「S 臂换 noise」RMS 作参照；重跑逐位（确定性）
判定行：
  MEM_A1_VS_STORE=PASS|FAIL frames mismatches      MEM_S_VS_TRAINSET=PASS|FAIL points key_mismatches
  MEM_C_VS_B=PASS|FAIL frames mismatches           MEM_A_VS_STORE_BITEXACT=<n>/<frames>（描述性）
  MEM_S_VS_B / MEM_S_VS_A / MEM_A_VS_B  frames max_abs mean_abs rel_fro cos_min cos_mean ulp_p50 ulp_p99 ulp_max frac_nonzero
  ACT_S_VS_B / ACT_S_VS_A / ACT_A_VS_B  points rms_norm max_abs_norm rms_unnorm | NOISE_S … | ratio | act_std（A1 不参与动作）
  ACT_C_VS_B_BITEXACT / ACT_DETERMINISM / SIGLIP_AB_REPLAY=DONE

用法（主进程 jax 必须在 GPU 上——P5 留档记过 CPU pos 表不逐位）：
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 UV_LINK_MODE=copy uv run --no-sync python \
    scripts/training/g0/compare_siglip_replay.py --lib v1-store/datasets/4task-motion-400ep \
    --ckpt v1-store/train-runs/mme_vla_suite_b128/awsprod40k-b128-motion/39999 --episodes VideoUnmask:0 --out <records>/
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import types

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "training" / "tests"))
_V1 = pathlib.Path(os.environ.get("MMEVLA_V1_STORE", str(_REPO_ROOT / "v1-store")))
os.environ.setdefault("OPENPI_DATA_HOME", str(_V1 / "models"))   # SigLipTokenizer 要求绝对路径（不 expanduser）

import _common as C  # noqa: E402

KEYS8 = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask",
         "motion_emb", "motion_pos", "motion_mask", "mem_order")
ARMS = ("S", "A", "B", "C")                       # 参与装配与动作对拍的臂
FRAME_ARMS = ARMS + ("A1",)                         # A1 只走帧路（逐帧 batch=1 复现建库），不开运动路、不参与装配/动作
MEM_PAIRS = (("S", "B"), ("S", "A"), ("A", "B"), ("S", "A1"), ("C", "B"))
ACT_PAIRS = (("S", "B"), ("S", "A"), ("A", "B"))


def load_episode_full(raw_dir: pathlib.Path, h5_file: str, raw_ep_idx: int, T: int):
    """与 compare_online_motion.load_episode 同式，另取 wrist_rgb 与 task_goal（动作对拍需要真 wrist 与 prompt）。"""
    import h5py
    frames, wrists, states = [], [], []
    with h5py.File(raw_dir / h5_file, "r") as f:
        g = f[f"episode_{raw_ep_idx}"]
        ts_ids = sorted(int(k.split("_")[-1]) for k in g.keys() if k.startswith("timestep_"))
        if len(ts_ids) != T or ts_ids != list(range(T)):
            raise SystemExit(f"错误: {h5_file} episode_{raw_ep_idx} timesteps {len(ts_ids)} != 清单 {T}")
        tg = g["setup"]["task_goal"][()]
        task_goal = (tg[0] if getattr(tg, "shape", ()) else tg)
        task_goal = task_goal.decode() if isinstance(task_goal, bytes) else str(task_goal)
        for t in ts_ids:
            ts = g[f"timestep_{t}"]
            img = ts["obs"]["front_rgb"][()]
            wr = ts["obs"]["wrist_rgb"][()]
            if img.shape != (256, 256, 3) or img.dtype != np.uint8 or wr.shape != (256, 256, 3) or wr.dtype != np.uint8:
                raise SystemExit(f"错误: 帧形制 front={img.shape} {img.dtype} wrist={wr.shape} {wr.dtype}")
            frames.append(img); wrists.append(wr)
            joint = ts["obs"]["joint_state"][()]
            grip = ts["obs"]["gripper_state"][()]
            states.append(np.concatenate([joint, grip[:1]], axis=0, dtype=np.float32))
    return np.stack(frames)[:, None], np.stack(wrists), np.stack(states), task_goal


def _bytes_equal(a, b) -> bool:
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def _bf16_ulp(x32: np.ndarray) -> np.ndarray:
    """bf16 在 |x| 处的 ULP：2^(floor(log2|x|) - 7)。"""
    ax = np.abs(x32).astype(np.float64)
    e = np.floor(np.log2(np.where(ax > 0, ax, 1.0)))
    return np.ldexp(1.0, (e - 7).astype(np.int32))


class DiffAcc:
    """两臂帧级 image_emb_4x4（(16,2048) bf16）数值差累计。ULP 以两侧绝对值较大者为基（零元素不计入 ULP 分位）。"""

    def __init__(self):
        self.n = 0; self.max_abs = 0.0; self.sum_abs = 0.0; self.cnt = 0
        self.sum_d2 = 0.0; self.sum_a2 = 0.0; self.nonzero = 0
        self.cos = []; self.ulp = []

    def add(self, a, b):
        a32 = np.asarray(a).astype(np.float32); b32 = np.asarray(b).astype(np.float32)
        d = b32 - a32
        self.n += 1
        self.max_abs = max(self.max_abs, float(np.abs(d).max()))
        self.sum_abs += float(np.abs(d).sum()); self.cnt += d.size; self.nonzero += int((d != 0).sum())
        self.sum_d2 += float((d.astype(np.float64) ** 2).sum()); self.sum_a2 += float((a32.astype(np.float64) ** 2).sum())
        na = np.linalg.norm(a32, axis=-1); nb = np.linalg.norm(b32, axis=-1)
        self.cos.extend(((a32 * b32).sum(-1) / (na * nb + 1e-12)).tolist())
        base = np.maximum(np.abs(a32), np.abs(b32))
        sel = base > 0
        if np.any(sel):
            self.ulp.append((np.abs(d)[sel] / _bf16_ulp(base[sel])).ravel())

    def summary(self) -> dict:
        ulp = np.concatenate(self.ulp) if self.ulp else np.zeros(1)
        return {"frames": self.n, "max_abs": self.max_abs, "mean_abs": self.sum_abs / max(self.cnt, 1),
                "rel_fro": float(np.sqrt(self.sum_d2 / max(self.sum_a2, 1e-30))),
                "cos_min": float(min(self.cos)) if self.cos else None, "cos_mean": float(np.mean(self.cos)) if self.cos else None,
                "ulp_p50": float(np.percentile(ulp, 50)), "ulp_p99": float(np.percentile(ulp, 99)), "ulp_max": float(ulp.max()),
                "frac_nonzero": self.nonzero / max(self.cnt, 1)}

    def line(self, tag: str) -> str:
        s = self.summary()
        return (f"{tag} frames={s['frames']} max_abs={s['max_abs']:.4g} mean_abs={s['mean_abs']:.4g} rel_fro={s['rel_fro']:.4g} "
                f"cos_min={s['cos_min']:.6f} cos_mean={s['cos_mean']:.6f} ulp_p50={s['ulp_p50']:.3g} ulp_p99={s['ulp_p99']:.3g} "
                f"ulp_max={s['ulp_max']:.3g} frac_nonzero={s['frac_nonzero']:.3f}")


def _rms(x) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", default=str(_V1 / "datasets/4task-motion-400ep"))
    ap.add_argument("--ckpt", required=True, help="checkpoint 目录（含 params/ 与 assets/，如 .../39999）")
    ap.add_argument("--config", default="mme_vla_suite", help="推理侧 config 条目（生产 serve_policy 用 mme_vla_suite）")
    ap.add_argument("--episodes", default="VideoUnmask:0", help="逗号分隔的 <task>:<raw_ep_idx>")
    ap.add_argument("--noise-seeds", default="0,1,2")
    ap.add_argument("--out", required=True, help="输出目录")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    import omegaconf
    from mme_vla_suite.training import config as _config
    from mme_vla_suite.policies import policy_config as _policy_config
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.datastore.framesamp_store import StoreMeta
    from mme_vla_suite.training.dataloader import _create_framesamp_dataset, _motion_gates
    from mme_vla_suite.dataset_builder.siglip_tokenizer import SigLipTokenizer
    from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
    from mme_vla_suite.models.integration.history_observation import HistAugObservation

    if jax.default_backend() != "gpu":
        raise SystemExit(f"错误: 主进程 jax 后端 {jax.default_backend()} != gpu（PosEmb3D 4x4 表 CPU/GPU 不逐位，P5 留档）")

    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    lib = pathlib.Path(args.lib)
    ckpt_dir = pathlib.Path(args.ckpt)
    run_root = ckpt_dir.parent
    noise_seeds = [int(s) for s in args.noise_seeds.split(",")]

    # ── 生产口径的 policy（B 臂编码器 + transforms + norm_stats 全部来自 checkpoint）；motion 先用 stub 过构造，随后换成查表 ──
    t0 = time.perf_counter()
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, ckpt_dir, motion_stub=True)
    stub = policy._motion_client
    stub.close()
    enc_B = policy._vision_encode
    print(f"[sg] policy 构造 {time.perf_counter() - t0:.1f}s config={args.config} ckpt={ckpt_dir} use_quantiles={policy.use_quantiles} "
          f"action_horizon={policy._model.action_horizon} action_dim={policy._model.action_dim}")

    # ── A / C 臂 ──
    t0 = time.perf_counter()
    tokA = SigLipTokenizer()
    enc_A = jax.jit(tokA.__call__)
    tokC = SigLipTokenizer()
    gd, st = nnx.split(tokC.img)
    n_cast = [0]

    def _cast(x):
        if isinstance(x, jax.Array) and x.dtype == jnp.float32:
            n_cast[0] += 1
            return x.astype(jnp.bfloat16)
        return x
    st = jax.tree.map(_cast, st)
    tokC.img = nnx.merge(gd, st)
    enc_C = jax.jit(tokC.__call__)
    leafA = jax.tree.leaves(nnx.state(tokA.img))
    leafC = jax.tree.leaves(nnx.state(tokC.img))
    print(f"[sg] A/C 臂构造 {time.perf_counter() - t0:.1f}s A_dtypes={sorted({str(x.dtype) for x in leafA})} "
          f"C_dtypes={sorted({str(x.dtype) for x in leafC})} cast_leaves={n_cast[0]}")

    # ── 训练库：dataset（与 policy 同一份 norm_stats / use_quantiles）、framesamp store、motion store ──
    os.environ["MMEVLA_MOTION_STORE"] = str(lib / "motion")
    hc = omegaconf.OmegaConf.load(run_root / "history_config.resolved.yaml")
    data_config = types.SimpleNamespace(norm_stats={"state": policy.state_norm_stats}, use_quantile_norm=policy.use_quantiles)
    ds = _create_framesamp_dataset(str(lib / "framesamp"), data_config, hc, int(policy._model.action_horizon))
    fmeta = StoreMeta.load(str(lib / "framesamp"))
    motion_root = _motion_gates(hc, fmeta)
    mmeta = ms.MotionMeta.load(motion_root)
    mstore = ms.MotionStore(motion_root, meta=mmeta)
    fstore = ds._ensure_store()
    manifest = json.load(open(lib / "meta" / "episode_manifest.json", encoding="utf-8"))
    raw_dir = pathlib.Path(manifest["raw_dir"])
    ns_path = sorted((ckpt_dir / "assets").glob("*/norm_stats.json"))
    assert len(ns_path) == 1, f"checkpoint assets 下 norm_stats.json 不唯一: {ns_path}"
    ns_all = json.load(open(ns_path[0], encoding="utf-8"))["norm_stats"]
    act_std_mean = float(np.mean(ns_all["actions"]["std"]))

    AH, AD = int(policy._model.action_horizon), int(policy._model.action_dim)
    NOISE = {s: jax.random.normal(jax.random.key(s), (1, AH, AD), dtype=jnp.float32) for s in noise_seeds}

    acc = {p: DiffAcc() for p in MEM_PAIRS}
    bit = {"A1_vs_store": [0, 0], "A_vs_store": [0, 0], "C_vs_B": [0, 0]}     # [n, mismatches]
    trainset_points = 0; trainset_key_mis: dict[str, int] = {k: 0 for k in KEYS8}
    a_trainset_img_mis = 0
    det_ok = True
    cb_act_ok = True
    per_point: list[dict] = []
    per_episode: list[dict] = []
    act_rms = {p: {"norm": [], "unnorm": []} for p in ACT_PAIRS}
    act_max = {p: 0.0 for p in ACT_PAIRS}
    noise_ref = {"norm": [], "unnorm": []}

    for spec in args.episodes.split(","):
        task, raw_ep = spec.split(":"); raw_ep = int(raw_ep)
        ep = next(e for e in manifest["episodes"] if e["h5_file"] == f"record_dataset_{task}.h5" and int(e["raw_ep_idx"]) == raw_ep)
        g = int(ep["global_episode_idx"]); es = int(ep["exec_start_idx"]); T = int(ep["num_timesteps"]); row_base = int(ep["total_sample_offset"])
        entry = ds._motion_entries[g]
        assert entry.g == g and entry.exec_start_idx == es and entry.num_timesteps == T, "motion index 与清单不符"
        rows_all, f_all = ms.visible_motion_rows(entry, T - 1)
        f2row = {int(f): int(r) for r, f in zip(rows_all.tolist(), f_all.tolist())}

        def motion_lookup(window, start_frame, _f2row=f2row):
            row = _f2row[int(start_frame)]
            return np.asarray(mstore.rows(np.asarray([row], dtype=np.int64))[0], dtype=np.float32)

        frames, wrists, states, task_goal = load_episode_full(raw_dir, ep["h5_file"], raw_ep, T)
        print(f"[sg] episode {task}:{raw_ep} g={g} es={es} T={T} row_base={row_base} motion_rows={len(f2row)} prompt={task_goal!r}")

        def make_mem(enc):
            policy._vision_encode = enc; policy._motion_client = motion_lookup; policy._prepare_mem_buffer(); return policy.mem_buffer
        mems = {"S": make_mem(enc_A), "A": make_mem(enc_A), "C": make_mem(enc_C), "B": make_mem(enc_B)}   # B 最后
        assert policy._vision_encode is enc_B
        cfgm = policy.config
        memA1 = FrameSampMemory(num_views=cfgm.num_views, img_emb_dim=cfgm.memory_feature.img.input_dim,
                                pos_emb_dim=cfgm.memory_feature.pos.input_dim, state_emb_dim=cfgm.memory_feature.state.input_dim,
                                vision_enc_fn=enc_A)                                   # 帧路与其余臂逐字同一 add_buffer，只是不开运动路
        policy.mem_buffer = mems["B"]; policy.step_idx = -1; policy.exec_start_idx = 0

        def feed(lo, hi):
            prev = policy.step_idx
            policy.add_buffer({"images": frames[lo:hi], "state": states[lo:hi], "exec_start_idx": es if lo == 0 else 0})   # B 臂，成批
            sl = list(range(prev + 1, prev + 1 + (hi - lo)))
            esx = policy.exec_start_idx
            for name in ("A", "C", "S"):                                                                              # 成批
                mems[name].add_buffer(frames[lo:hi], states[lo:hi], sl, exec_start_idx=esx)
            for i, s in enumerate(sl):                                                                                # A1 逐帧 batch=1（复现建库）
                memA1.add_buffer(frames[lo + i:lo + i + 1], states[lo + i:lo + i + 1], [s])
            rows = fstore.read_image_rows(np.asarray([row_base + s for s in sl], dtype=np.int64))                     # S 臂改写为训练库行
            for i, s in enumerate(sl):
                mems["S"]._history_feats[s]["image_emb_4x4"] = np.ascontiguousarray(rows[i][None])
            return sl

        ep_points = 0

        def check(t, new_steps):
            nonlocal trainset_points, a_trainset_img_mis, det_ok, cb_act_ok, ep_points
            assert policy.step_idx == t
            rec = {"task": task, "raw_ep": raw_ep, "g": g, "t": t, "new_frames": len(new_steps)}
            # 帧级
            fa = {p: DiffAcc() for p in MEM_PAIRS}
            for s in new_steps:
                e = {n: np.asarray(mems[n]._history_feats[s]["image_emb_4x4"])[0] for n in ARMS}
                e["A1"] = np.asarray(memA1._history_feats[s]["image_emb_4x4"])[0]
                bit["A1_vs_store"][0] += 1; bit["A1_vs_store"][1] += 0 if _bytes_equal(e["A1"], e["S"]) else 1
                bit["A_vs_store"][0] += 1; bit["A_vs_store"][1] += 0 if _bytes_equal(e["A"], e["S"]) else 1
                bit["C_vs_B"][0] += 1; bit["C_vs_B"][1] += 0 if _bytes_equal(e["C"], e["B"]) else 1
                for p in MEM_PAIRS:
                    acc[p].add(e[p[0]], e[p[1]]); fa[p].add(e[p[0]], e[p[1]])
            rec["frame"] = {f"{p[0]}_vs_{p[1]}": fa[p].summary() for p in MEM_PAIRS}
            # 装配级 + 动作级
            element = {"observation/image": frames[t, 0], "observation/wrist_image": wrists[t],
                       "observation/state": states[t], "prompt": task_goal}
            assembled, acts_norm, acts_unnorm = {}, {}, {}
            for name in ARMS:
                policy.mem_buffer = mems[name]
                inputs = policy._prepare_history(dict(element))
                assembled[name] = {k: np.asarray(inputs[k]).copy() for k in KEYS8}
                inputs = policy._input_transform(inputs)
                obs = HistAugObservation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs))
                acts_norm[name], acts_unnorm[name] = {}, {}
                for sd in noise_seeds:
                    a = policy._sample_actions(jax.random.key(0), obs, noise=NOISE[sd], **policy._sample_kwargs)
                    a = np.asarray(jax.block_until_ready(a))[0]
                    acts_norm[name][sd] = a
                    acts_unnorm[name][sd] = policy._output_transform({"state": np.asarray(obs.state[0]), "actions": a})["actions"]
                if name == "S":   # 确定性自检：同臂同噪声重跑逐位
                    a2 = np.asarray(jax.block_until_ready(policy._sample_actions(jax.random.key(0), obs, noise=NOISE[noise_seeds[0]], **policy._sample_kwargs)))[0]
                    if not _bytes_equal(a2, acts_norm["S"][noise_seeds[0]]):
                        det_ok = False
            policy.mem_buffer = mems["B"]
            # S 侧装配 vs 训练样本（应逐位）；A 侧 static_image_emb vs 训练样本（描述性）
            idx = np.flatnonzero((ds._epis_of == g) & (ds._step_of == t))
            assert len(idx) == 1, f"训练样本定位失败 g={g} t={t}: {idx}"
            sample = ds[int(idx[0])]
            key_ok = {}
            for k in KEYS8:
                ok = _bytes_equal(assembled["S"][k], np.asarray(sample[k]))
                key_ok[k] = ok
                trainset_key_mis[k] += 0 if ok else 1
            trainset_points += 1
            a_img_ok = _bytes_equal(assembled["A"]["static_image_emb"], np.asarray(sample["static_image_emb"]))
            a_trainset_img_mis += 0 if a_img_ok else 1
            rec["trainset_key_ok_S"] = key_ok; rec["trainset_static_image_emb_ok_A"] = a_img_ok
            mask = assembled["S"]["static_mask"]
            rec["static_image_emb"] = {}
            for p in MEM_PAIRS:
                ia = assembled[p[0]]["static_image_emb"].astype(np.float32)[mask]; ib = assembled[p[1]]["static_image_emb"].astype(np.float32)[mask]
                rec["static_image_emb"][f"{p[0]}_vs_{p[1]}"] = {"tokens": int(mask.sum()), "max_abs": float(np.abs(ib - ia).max()),
                                                                 "rel_fro": float(np.linalg.norm(ib - ia) / (np.linalg.norm(ia) + 1e-30))}
            # 动作
            rec["act"] = {}
            for p in ACT_PAIRS:
                dn = [acts_norm[p[1]][sd] - acts_norm[p[0]][sd] for sd in noise_seeds]
                du = [acts_unnorm[p[1]][sd] - acts_unnorm[p[0]][sd] for sd in noise_seeds]
                rec["act"][f"{p[0]}_vs_{p[1]}"] = {"rms_norm": [_rms(x) for x in dn], "max_abs_norm": [float(np.abs(x).max()) for x in dn],
                                                   "rms_unnorm": [_rms(x) for x in du], "max_abs_unnorm": [float(np.abs(x).max()) for x in du]}
                act_rms[p]["norm"].extend(_rms(x) for x in dn); act_rms[p]["unnorm"].extend(_rms(x) for x in du)
                act_max[p] = max(act_max[p], max(float(np.abs(x).max()) for x in dn))
            pairs = [(i, j) for i in range(len(noise_seeds)) for j in range(i + 1, len(noise_seeds))]
            nn_ = [acts_norm["S"][noise_seeds[j]] - acts_norm["S"][noise_seeds[i]] for i, j in pairs]
            nu_ = [acts_unnorm["S"][noise_seeds[j]] - acts_unnorm["S"][noise_seeds[i]] for i, j in pairs]
            rec["act"]["noise_S"] = {"rms_norm": [_rms(x) for x in nn_], "rms_unnorm": [_rms(x) for x in nu_]}
            noise_ref["norm"].extend(_rms(x) for x in nn_); noise_ref["unnorm"].extend(_rms(x) for x in nu_)
            cb = all(_bytes_equal(acts_norm["C"][sd], acts_norm["B"][sd]) for sd in noise_seeds)
            cb_act_ok = cb_act_ok and cb
            rec["act"]["C_vs_B_bitexact"] = cb
            rec["act"]["S_unnorm_seed0"] = acts_unnorm["S"][noise_seeds[0]].tolist()
            rec["act"]["B_unnorm_seed0"] = acts_unnorm["B"][noise_seeds[0]].tolist()
            per_point.append(rec); ep_points += 1
            sb = fa[("S", "B")].summary(); sa = fa[("S", "A")].summary(); ab = fa[("A", "B")].summary()
            print(f"[sg] t={t:4d} 新帧 {len(new_steps):3d} | 帧 rel_fro S-B={sb['rel_fro']:.3g} S-A={sa['rel_fro']:.3g} A-B={ab['rel_fro']:.3g} "
                  f"| A1==S {bit['A1_vs_store'][0] - bit['A1_vs_store'][1]}/{bit['A1_vs_store'][0]} C==B {bit['C_vs_B'][0] - bit['C_vs_B'][1]}/{bit['C_vs_B'][0]} "
                  f"trainset_S_ok={all(key_ok.values())} | act rms_unnorm S-B={np.mean(rec['act']['S_vs_B']['rms_unnorm']):.4g} "
                  f"S-A={np.mean(rec['act']['S_vs_A']['rms_unnorm']):.4g} A-B={np.mean(rec['act']['A_vs_B']['rms_unnorm']):.4g} "
                  f"noise={np.mean(rec['act']['noise_S']['rms_unnorm']):.4g} C==B={cb}")

        tb = time.perf_counter()
        sl = feed(0, es + 1); check(es, sl)
        t = es
        while t + 16 < T:
            sl = feed(t + 1, t + 17); t += 16; check(t, sl)
        per_episode.append({"task": task, "raw_ep": raw_ep, "g": g, "es": es, "T": T, "points": ep_points, "wall_s": time.perf_counter() - tb})

    mstore.close(); ds.close()
    key_mis = {k: v for k, v in trainset_key_mis.items() if v}
    n_points = len(per_point)

    def act_line(p):
        r = act_rms[p]; tag = f"ACT_{p[0]}_VS_{p[1]}"
        return (f"{tag} points={n_points} rms_norm={np.mean(r['norm']):.4g} max_abs_norm={act_max[p]:.4g} rms_unnorm={np.mean(r['unnorm']):.4g} "
                f"| NOISE_S rms_norm={np.mean(noise_ref['norm']):.4g} rms_unnorm={np.mean(noise_ref['unnorm']):.4g} "
                f"| ratio_norm={np.mean(r['norm']) / max(np.mean(noise_ref['norm']), 1e-30):.4g} ratio_unnorm={np.mean(r['unnorm']) / max(np.mean(noise_ref['unnorm']), 1e-30):.4g} "
                f"| act_std_mean={act_std_mean:.4g} rms_unnorm/act_std={np.mean(r['unnorm']) / act_std_mean:.4g}")

    lines = [
        f"MEM_A1_VS_STORE={'PASS' if bit['A1_vs_store'][1] == 0 else 'FAIL'} frames={bit['A1_vs_store'][0]} mismatches={bit['A1_vs_store'][1]}",
        f"MEM_S_VS_TRAINSET={'PASS' if not key_mis else 'FAIL'} points={trainset_points} key_mismatches={key_mis or 'none'}",
        f"MEM_C_VS_B={'PASS' if bit['C_vs_B'][1] == 0 else 'FAIL'} frames={bit['C_vs_B'][0]} mismatches={bit['C_vs_B'][1]}",
        f"MEM_A_VS_STORE_BITEXACT={bit['A_vs_store'][0] - bit['A_vs_store'][1]}/{bit['A_vs_store'][0]} "
        f"A_static_image_emb_vs_trainset_bitexact_points={trainset_points - a_trainset_img_mis}/{trainset_points}",
        acc[("S", "B")].line("MEM_S_VS_B"), acc[("S", "A")].line("MEM_S_VS_A"), acc[("A", "B")].line("MEM_A_VS_B"),
        acc[("S", "A1")].line("MEM_S_VS_A1"), acc[("C", "B")].line("MEM_C_VS_B_NUM"),
        act_line(("S", "B")), act_line(("S", "A")), act_line(("A", "B")),
        f"ACT_C_VS_B_BITEXACT={'PASS' if cb_act_ok else 'FAIL'}",
        f"ACT_DETERMINISM={'PASS' if det_ok else 'FAIL'}",
        f"SIGLIP_AB_REPLAY=DONE episodes={len(per_episode)} points={n_points}",
    ]
    print("\n".join(lines))
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "per_point.json").write_text(json.dumps({"lines": lines, "per_episode": per_episode, "per_point": per_point,
                                                          "argv": sys.argv, "ckpt": str(ckpt_dir), "lib": str(lib), "config": args.config,
                                                          "noise_seeds": noise_seeds}, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    ok = bit["A1_vs_store"][1] == 0 and not key_mis and bit["C_vs_B"][1] == 0 and det_ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
