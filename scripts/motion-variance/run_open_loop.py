#!/usr/bin/env python3
"""阶段 0（开环动作差）+ 阶段 1（18 层机制），零 rollout（计划第二部分 §8）。

阶段 0：每个决策点 × N 个 noise seed，同一 obs 同一 noise 下比较 none / mask_all / swap 三种输入的最终动作：
  d_mask = mean_s rms(a_mask − a_none)，d_swap 同理，标尺 d_noise = noise seed 两两差的 rms 均值；归一化与反归一化各一份。
  OL_ZEROK_NULL（阻断）：k=0 的点屏蔽 / 替换必须逐位无影响——内建伪阳性否证。
阶段 1（每集 cold / early / mid / late 4 点）：全部在 18 层展开路径（MVAdapter.f_sample_layered）内自洽比较：
  (i) LAYER_ATTN：去噪 action query 与 prefill frame/text query 的 motion 份额与逐 query 合法-key 均匀基准之比；
  (ii) LAYER_ACT_DELTA：step_only(l) / both(l) / kv_vzero(l) / kv_donor(l) 四种逐层干预的**最终动作**差（完整 10 步）；
      kv_donor 的 K/V 来自「接收方同一 obs、motion_emb 换成 donor 内容后重新 prefill」的同层同列；
  (iii) LAYER_GRAD：固定 (obs, x_t, t, u_t) 的 loss 对第 l 层入口隐状态的梯度（按 motion / frame / img / txt 列分组），t ∈ {0.1, 0.5, 0.9}。
swap 的 donor 走 library bank 的开环映射 `ol<noise_seed>|task|recv_g`（排除自身），多 donor 一并估 donor 方差。

用法：
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0' \
    UV_LINK_MODE=copy PYTHONUNBUFFERED=1 uv run --no-sync python scripts/motion-variance/run_open_loop.py --stage 0,1 \
      --donor-bank v1-store/reports/motion-variance/bank-lib --out v1-store/reports/motion-variance/open_loop.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import mv_common as C  # noqa: E402
from mv_donor_bank import DonorBank  # noqa: E402
from mv_model_adapter import MVAdapter  # noqa: E402
from mv_points import DEFAULT_EPISODES_OPEN_LOOP, PointSource  # noqa: E402

K_BINS = ((0, 0, "k=0"), (1, 4, "1-4"), (5, 12, "5-12"), (13, 10 ** 9, ">12"))
T_GRID = (0.1, 0.5, 0.9)
SCOPES = ("step_only", "both", "kv_vzero", "kv_donor")


def _fmt_list(xs, fmt=".4g") -> str:
    return "[" + ",".join(format(float(x), fmt) for x in xs) + "]"


def _kbin(k: int) -> str:
    for lo, hi, name in K_BINS:
        if lo <= k <= hi:
            return name
    return "?"


def _select_layer_points(n_points: int, taus: list[int], es: int) -> dict[int, str]:
    """每集 4 点：cold=首点；early=首个 exec 窗数≥3 的点（τ−es ≥ 64）；mid=50% 处；late=90% 处。去重后返回 {点序号: 标签}。"""
    sel: dict[int, str] = {}
    sel[0] = "cold"
    for j, t in enumerate(taus):
        if t - es >= 64:
            sel.setdefault(j, "early")
            break
    sel.setdefault(int(round(0.5 * (n_points - 1))), "mid")
    sel.setdefault(int(round(0.9 * (n_points - 1))), "late")
    return sel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", default=str(C.LIB))
    ap.add_argument("--ckpt", default=str(C.CKPT_MOTION))
    ap.add_argument("--train-config", default=C.TRAIN_CONFIG)
    ap.add_argument("--episodes", default=DEFAULT_EPISODES_OPEN_LOOP)
    ap.add_argument("--noise-seeds", default="0,1,2,3,4")
    ap.add_argument("--donor-bank", default=str(C.REPORT_ROOT / "bank-lib"))
    ap.add_argument("--max-points", type=int, default=0, help="每集前 N 个决策点（0=全部）")
    ap.add_argument("--stage", default="0,1")
    ap.add_argument("--out", default=str(C.REPORT_ROOT / "open_loop.json"))
    args = ap.parse_args()
    stages = {int(s) for s in args.stage.split(",")}
    seeds = [int(s) for s in args.noise_seeds.split(",")]
    if len(seeds) < 2:
        raise SystemExit("错误: 至少 2 个 noise seed（标尺需要两两差）")

    import jax
    import jax.numpy as jnp

    C.require_gpu()
    C.require_det_flags()
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    V = C.Verdict()
    t_start = time.perf_counter()
    bank = DonorBank(pathlib.Path(args.donor_bank))
    src = PointSource(lib=pathlib.Path(args.lib), ckpt_dir=pathlib.Path(args.ckpt), train_config_name=args.train_config,
                      episodes_spec=args.episodes, max_points=args.max_points)
    ad = MVAdapter(src.policy, unroll=(1 in stages))
    if 1 in stages:
        ad.build_grads()
    fns = ad.make_sample_fns()
    policy = src.policy
    NOISE = {s: jax.random.normal(jax.random.key(s), (1, ad.AH, ad.AD), dtype=jnp.float32) for s in seeds}
    key0 = jax.random.key(0)
    L = C.N_LAYERS
    pairs = [(seeds[i], seeds[j]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]

    def unnorm(obs, a):
        return np.asarray(policy._output_transform({"state": np.asarray(obs.state[0]), "actions": np.asarray(a)[0]})["actions"])

    def swap_obs(pt, s: int):
        me = np.array(np.asarray(pt["obs"].motion_emb), dtype=np.float32, copy=True)
        cov = {"exact": 0, "fallback": 0, "cycle": 0, "cross_seg": 0}
        donor = None
        for i, f in enumerate(pt["frames"]):
            es = pt["es"]
            seg, m = ("demo", f // C.GRID_STRIDE) if f < es else ("exec", (f - es) // C.GRID_STRIDE)
            tok, how, g = bank.lookup(f"ol{s}", pt["task"], pt["g"], seg, m)
            me[0, i] = tok
            cov[how] += 1
            donor = g if donor is None else donor
        return dataclasses.replace(pt["obs"], motion_emb=jnp.asarray(me)), cov, donor

    s0_rows: list[dict] = []
    s1_rows: list[dict] = []
    zerok = {"n": 0, "mask_ok": 0, "swap_ok": 0}
    cover_tot = {"exact": 0, "fallback": 0, "cycle": 0, "cross_seg": 0}
    layer_sel_cache: dict[int, dict[int, str]] = {}
    n_pts = 0
    try:
        for pt in src.iter_points():
            obs = pt["obs"]
            k = int(pt["k"])
            n_exec = sum(1 for f in pt["frames"] if f >= pt["es"])
            phase = "cold" if n_exec == 0 else "steady"
            n_pts += 1
            # ── 阶段 0 ─────────────────────────────────────────────────────────────
            a_n, a_m, a_s, u_n, u_m, u_s = {}, {}, {}, {}, {}, {}
            covs = []
            for s in seeds:
                a_n[s] = np.asarray(jax.block_until_ready(fns["none"](key0, obs, noise=NOISE[s])))
                a_m[s] = np.asarray(jax.block_until_ready(fns["mask"](key0, obs, noise=NOISE[s])))
                o_s, cov, donor = swap_obs(pt, s)
                covs.append(cov)
                a_s[s] = np.asarray(jax.block_until_ready(fns["none"](key0, o_s, noise=NOISE[s])))
                u_n[s], u_m[s], u_s[s] = unnorm(obs, a_n[s]), unnorm(obs, a_m[s]), unnorm(obs, a_s[s])
            for cov in covs:
                for h in cover_tot:
                    cover_tot[h] += cov[h]
            row = {"g": pt["g"], "task": pt["task"], "es": pt["es"], "t": pt["t"], "point_idx": pt["point_idx"],
                   "n_points": pt["n_points"], "k": k, "n_exec": n_exec, "phase": phase, "kbin": _kbin(k),
                   "d_mask": float(np.mean([C.rms(a_m[s] - a_n[s]) for s in seeds])),
                   "d_swap": float(np.mean([C.rms(a_s[s] - a_n[s]) for s in seeds])),
                   "d_noise": float(np.mean([C.rms(a_n[b] - a_n[a]) for a, b in pairs])),
                   "d_mask_un": float(np.mean([C.rms(u_m[s] - u_n[s]) for s in seeds])),
                   "d_swap_un": float(np.mean([C.rms(u_s[s] - u_n[s]) for s in seeds])),
                   "d_noise_un": float(np.mean([C.rms(u_n[b] - u_n[a]) for a, b in pairs])),
                   "d_swap_by_seed": [float(C.rms(a_s[s] - a_n[s])) for s in seeds],
                   "cover": covs[0]}
            if k == 0:
                zerok["n"] += 1
                zerok["mask_ok"] += int(all(C.bytes_equal(a_m[s], a_n[s]) for s in seeds))
                zerok["swap_ok"] += int(all(C.bytes_equal(a_s[s], a_n[s]) for s in seeds))
            s0_rows.append(row)
            msg = (f"[ol] pt {n_pts} {pt['task']} g={pt['g']} t={pt['t']} k={k} {phase} "
                   f"mask/noise={row['d_mask'] / max(row['d_noise'], 1e-12):.3f} swap/noise={row['d_swap'] / max(row['d_noise'], 1e-12):.3f}")
            # ── 阶段 1 ─────────────────────────────────────────────────────────────
            if 1 in stages:
                sel = layer_sel_cache.setdefault(pt["g"], _select_layer_points(pt["n_points"], C.expected_taus(pt["es"], pt["T"])[:pt["n_points"]], pt["es"]))
                if pt["point_idx"] in sel:
                    tag = sel[pt["point_idx"]]
                    t1 = time.perf_counter()
                    s_a, s_b = seeds[0], seeds[1]
                    gp0, gs0 = ad.layered_gates(obs, "none", "none")
                    base = jax.block_until_ready(ad.run_layered(obs, gate_prefill_layers=gp0, gate_step_layers=gs0, noise=NOISE[s_a]))
                    base_b = jax.block_until_ready(ad.run_layered(obs, gate_prefill_layers=gp0, gate_step_layers=gs0, noise=NOISE[s_b]))
                    x0 = np.asarray(base["x_0"])
                    d_noise_l = C.rms(np.asarray(base_b["x_0"]) - x0)
                    gpm, gsm = ad.layered_gates(obs, "mask", "mask")
                    allmask = jax.block_until_ready(ad.run_layered(obs, gate_prefill_layers=gpm, gate_step_layers=gsm, noise=NOISE[s_a]))
                    d_allmask = C.rms(np.asarray(allmask["x_0"]) - x0)
                    # donor KV：接收方 obs 换内容后重新 prefill（同 mem_order，列一一对应）
                    o_s, _, donor_g = swap_obs(pt, s_a)
                    swapkv = jax.block_until_ready(ad.run_layered(o_s, gate_prefill_layers=gp0, gate_step_layers=gs0, noise=NOISE[s_a]))
                    kv_ov = (swapkv["k"], swapkv["v"])
                    d_swap_l = C.rms(np.asarray(swapkv["x_0"]) - x0)
                    deltas = {sc: np.zeros(L) for sc in SCOPES}
                    gp_np = np.asarray(gp0); gs_np = np.asarray(gs0)
                    mc_row = ~np.asarray(ad.gates(obs, "mask")[0])          # bool[b,P] motion 列
                    for l in range(L):
                        for sc in SCOPES:
                            gp_l, gs_l = gp_np.copy(), gs_np.copy()
                            kw = {}
                            if sc in ("step_only", "both"):
                                gs_l[l] = ~mc_row
                            if sc == "both":
                                gp_l[l] = ~mc_row
                            if sc == "kv_vzero":
                                vz = np.zeros(L, bool); vz[l] = True
                                kw["v_zero_layers"] = jnp.asarray(vz)
                            if sc == "kv_donor":
                                ov = np.zeros(L, bool); ov[l] = True
                                kw["kv_override"] = kv_ov
                                kw["override_layers"] = jnp.asarray(ov)
                            r = jax.block_until_ready(ad.run_layered(obs, gate_prefill_layers=jnp.asarray(gp_l), gate_step_layers=jnp.asarray(gs_l),
                                                                     noise=NOISE[s_a], **kw))
                            deltas[sc][l] = C.rms(np.asarray(r["x_0"]) - x0)
                    # attention 份额（none 基线）
                    share_step = np.asarray(base["share_step"])[:, 0]              # (L,G,T)
                    U_step = float(np.asarray(base["U_step"])[0])
                    share_layer = share_step.mean(axis=(1, 2))                     # (L,)
                    head_max = share_step.mean(axis=2).max(axis=1)                 # (L,)
                    head_arg = share_step.mean(axis=2).argmax(axis=1)
                    sh_f = np.asarray(base["share_prefill_frame"])[:, 0]           # (L,G)
                    sh_t = np.asarray(base["share_prefill_txt"])[:, 0]
                    U_f = float(np.asarray(base["U_prefill_frame"])[0]); U_t = float(np.asarray(base["U_prefill_txt"])[0])
                    # 梯度
                    acts = jnp.asarray(pt["actions"], jnp.float32)[None]
                    grads = {}
                    for tt in T_GRID:
                        x_t = tt * NOISE[s_a] + (1.0 - tt) * acts
                        u_t = NOISE[s_a] - acts
                        tstep = jnp.asarray([tt], jnp.float32)
                        g = jax.block_until_ready(ad.f_layer_grads(ad.gstate, obs, x_t, tstep, u_t))
                        gi = jax.block_until_ready(ad.f_input_grads(ad.gstate, obs, x_t, tstep, u_t))
                        cn = np.asarray(g["col_norm"])[:, 0]                        # (L,P)
                        mc = np.asarray(g["mc"])[0]; pm = np.asarray(g["pm"])[0]
                        idx = np.arange(C.PREFIX_LEN)
                        groups = {"motion": mc & pm, "frame": pm & (idx < C.MEM_LEN) & ~mc,
                                  "img": (idx >= C.MEM_LEN) & (idx < C.MEM_LEN + C.IMG_LEN), "txt": pm & (idx >= C.MEM_LEN + C.IMG_LEN)}
                        gme = np.asarray(gi["g_motion_emb"])[0]; gmp = np.asarray(gi["g_motion_pos"])[0]
                        mm = np.asarray(obs.motion_mask)[0]
                        grads[str(tt)] = {
                            "loss": float(g["loss"]),
                            "gnorm": {name: np.sqrt((cn[:, sel_] ** 2).sum(axis=1)).tolist() for name, sel_ in groups.items()},
                            "per_token": {name: (np.sqrt((cn[:, sel_] ** 2).mean(axis=1)) if sel_.sum() else np.zeros(L)).tolist() for name, sel_ in groups.items()},
                            "n_cols": {name: int(sel_.sum()) for name, sel_ in groups.items()},
                            "g_motion_emb_valid_rms": float(C.rms(gme[mm])) if k else 0.0,
                            "g_motion_pos_valid_rms": float(C.rms(gmp[mm])) if k else 0.0,
                            "pad_zero": bool(np.all(gme[~mm] == 0) and np.all(gmp[~mm] == 0))}
                    s1_rows.append({"g": pt["g"], "task": pt["task"], "t": pt["t"], "es": pt["es"], "k": k, "phase": phase, "tag": tag,
                                    "point_idx": pt["point_idx"], "d_noise_layered": d_noise_l, "d_allmask_layered": d_allmask,
                                    "d_swapkv_layered": d_swap_l, "d_mask_stage0": row["d_mask"], "donor_g": donor_g,
                                    "deltas": {sc: deltas[sc].tolist() for sc in SCOPES},
                                    "share_layer": share_layer.tolist(), "head_max": head_max.tolist(), "head_arg": head_arg.tolist(),
                                    "U_step": U_step, "share_prefill_frame": sh_f.mean(axis=1).tolist(), "U_prefill_frame": U_f,
                                    "share_prefill_txt": sh_t.mean(axis=1).tolist(), "U_prefill_txt": U_t,
                                    "share_step_full": share_step.tolist(), "grads": grads, "wall_s": time.perf_counter() - t1})
                    msg += f" | L1[{tag}] allmask/noise={d_allmask / max(d_noise_l, 1e-12):.3f} {time.perf_counter() - t1:.0f}s"
            print(msg, flush=True)
    finally:
        src.close()

    # ── 阶段 0 判定行 ──────────────────────────────────────────────────────────────
    def agg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    def ratio(rows, num, den):
        return float(np.mean([r[num] for r in rows]) / max(np.mean([r[den] for r in rows]), 1e-12)) if rows else float("nan")

    def med_ratio(rows, num, den):
        return float(np.median([r[num] / max(r[den], 1e-12) for r in rows])) if rows else float("nan")

    ks = [r["k"] for r in s0_rows]
    by_k = {name: [r for r in s0_rows if r["kbin"] == name] for _, _, name in K_BINS}
    by_task = {t: [r for r in s0_rows if r["task"] == t] for t in C.TASKS}
    by_phase = {p: [r for r in s0_rows if r["phase"] == p] for p in ("cold", "steady")}
    std = src.ns_actions_std
    V.observe(f"OL_ACT_DELTA points={n_pts} episodes={len(src.eps)} noise_seeds={len(seeds)} k_mean={np.mean(ks):.2f} k_range={min(ks)}..{max(ks)} "
              f"mask_rms_norm={agg(s0_rows, 'd_mask'):.4g} swap_rms_norm={agg(s0_rows, 'd_swap'):.4g} noise_rms_norm={agg(s0_rows, 'd_noise'):.4g} "
              f"mask/noise={ratio(s0_rows, 'd_mask', 'd_noise'):.3f} swap/noise={ratio(s0_rows, 'd_swap', 'd_noise'):.3f} "
              f"mask/noise_median={med_ratio(s0_rows, 'd_mask', 'd_noise'):.3f} swap/noise_median={med_ratio(s0_rows, 'd_swap', 'd_noise'):.3f} "
              f"mask_unnorm/act_std={agg(s0_rows, 'd_mask_un') / std:.4g} swap_unnorm/act_std={agg(s0_rows, 'd_swap_un') / std:.4g} act_std_mean={std:.4g} "
              f"by_k=[" + ", ".join(f"{n}:n={len(r)},mask/noise={ratio(r, 'd_mask', 'd_noise'):.3f},swap/noise={ratio(r, 'd_swap', 'd_noise'):.3f}" for n, r in by_k.items()) + "] "
              f"by_task=[" + ", ".join(f"{t}:n={len(r)},mask/noise={ratio(r, 'd_mask', 'd_noise'):.3f},swap/noise={ratio(r, 'd_swap', 'd_noise'):.3f}" for t, r in by_task.items()) + "] "
              f"by_phase=[" + ", ".join(f"{p}:n={len(r)},mask/noise={ratio(r, 'd_mask', 'd_noise'):.3f},swap/noise={ratio(r, 'd_swap', 'd_noise'):.3f}" for p, r in by_phase.items()) + "]")
    nw = sum(cover_tot.values())
    V.observe(f"OL_DONOR_COVER windows={nw} " + " ".join(f"{h}={cover_tot[h]}/{nw}" for h in cover_tot) + f" bank_sha={bank.sha256[:16]}")
    zk_ok = zerok["mask_ok"] == zerok["n"] and zerok["swap_ok"] == zerok["n"]
    V.block(zk_ok, f"OL_ZEROK_NULL={'PASS' if zk_ok else 'FAIL'} zerok_points={zerok['n']} mask_bitexact={zerok['mask_ok']}/{zerok['n']} "
                   f"swap_bitexact={zerok['swap_ok']}/{zerok['n']} note=k=0 的点没有有效 motion 槽，屏蔽/替换必须逐位无影响")
    # ── 阶段 1 判定行 ──────────────────────────────────────────────────────────────
    if 1 in stages and s1_rows:
        n1 = len(s1_rows)
        share = np.mean([r["share_layer"] for r in s1_rows], axis=0)
        U = float(np.mean([r["U_step"] for r in s1_rows]))
        enrich = np.mean([np.asarray(r["share_layer"]) / max(r["U_step"], 1e-12) for r in s1_rows], axis=0)
        enr_f = np.mean([np.asarray(r["share_prefill_frame"]) / max(r["U_prefill_frame"], 1e-12) for r in s1_rows], axis=0)
        enr_t = np.mean([np.asarray(r["share_prefill_txt"]) / max(r["U_prefill_txt"], 1e-12) for r in s1_rows], axis=0)
        head_max = np.mean([r["head_max"] for r in s1_rows], axis=0)
        top3 = np.argsort(-enrich)[:3].tolist()
        by_phase1 = {}
        for p in ("cold", "steady"):
            rr = [r for r in s1_rows if r["phase"] == p]
            if rr:
                by_phase1[p] = np.mean([np.asarray(r["share_layer"]) / max(r["U_step"], 1e-12) for r in rr], axis=0)
        V.observe(f"LAYER_ATTN points={n1} layers={L} heads={C.N_QHEAD} U_step_mean={U:.4g} share={_fmt_list(share)} enrich={_fmt_list(enrich, '.3g')} "
                  f"prefill_frame_enrich={_fmt_list(enr_f, '.3g')} prefill_txt_enrich={_fmt_list(enr_t, '.3g')} head_max_share={_fmt_list(head_max)} "
                  f"top3_enrich_layers={top3} by_phase=[" + ", ".join(f"{p}:{_fmt_list(v, '.3g')}" for p, v in by_phase1.items()) + "]")
        dn = float(np.mean([r["d_noise_layered"] for r in s1_rows]))
        for sc in SCOPES:
            arr = np.mean([r["deltas"][sc] for r in s1_rows], axis=0)
            V.observe(f"LAYER_ACT_DELTA scope={sc} points={n1} rms_by_layer={_fmt_list(arr)} over_noise={_fmt_list(arr / max(dn, 1e-12), '.3g')} "
                      f"noise_layered={dn:.4g} top3_layers={np.argsort(-arr)[:3].tolist()}")
        allm = float(np.mean([r["d_allmask_layered"] for r in s1_rows]))
        s0m = float(np.mean([r["d_mask_stage0"] for r in s1_rows]))
        V.observe(f"LAYER_ALLMASK_VS_STAGE0 points={n1} allmask_layered_rms={allm:.4g} stage0_mask_rms={s0m:.4g} ratio={allm / max(s0m, 1e-12):.3f} "
                  f"swapkv_layered_rms={float(np.mean([r['d_swapkv_layered'] for r in s1_rows])):.4g} "
                  f"note=展开路径与 scan 路径 bf16 不逐位（见 adapter selftest），此处只核量级一致")
        for tt in T_GRID:
            key = str(tt)
            gm = np.mean([r["grads"][key]["gnorm"]["motion"] for r in s1_rows if r["k"] > 0], axis=0) if any(r["k"] > 0 for r in s1_rows) else np.zeros(L)
            gf = np.mean([r["grads"][key]["gnorm"]["frame"] for r in s1_rows], axis=0)
            gi_ = np.mean([r["grads"][key]["gnorm"]["img"] for r in s1_rows], axis=0)
            gt = np.mean([r["grads"][key]["gnorm"]["txt"] for r in s1_rows], axis=0)
            ptm = np.mean([r["grads"][key]["per_token"]["motion"] for r in s1_rows if r["k"] > 0], axis=0) if any(r["k"] > 0 for r in s1_rows) else np.zeros(L)
            ptf = np.mean([r["grads"][key]["per_token"]["frame"] for r in s1_rows], axis=0)
            pad0 = all(r["grads"][key]["pad_zero"] for r in s1_rows)
            V.observe(f"LAYER_GRAD t={tt} points={n1} loss_mean={np.mean([r['grads'][key]['loss'] for r in s1_rows]):.4g} "
                      f"gnorm_motion={_fmt_list(gm)} gnorm_frame={_fmt_list(gf)} gnorm_img={_fmt_list(gi_)} gnorm_txt={_fmt_list(gt)} "
                      f"per_token_motion_over_frame={_fmt_list(ptm / np.maximum(ptf, 1e-30), '.3g')} "
                      f"g_motion_emb_valid_rms={np.mean([r['grads'][key]['g_motion_emb_valid_rms'] for r in s1_rows]):.4g} "
                      f"g_motion_pos_valid_rms={np.mean([r['grads'][key]['g_motion_pos_valid_rms'] for r in s1_rows]):.4g} pad_zero={'PASS' if pad0 else 'FAIL'}")
            V.block(pad0, f"LAYER_GRAD_PAD_ZERO t={tt} {'PASS' if pad0 else 'FAIL'}")
    final = V.final("MV_OPEN_LOOP", f"points={n_pts} layer_points={len(s1_rows)} stages={sorted(stages)} wall_s={time.perf_counter() - t_start:.0f}")
    lines = V.texts() + [final]
    print("\n".join(lines))
    rec = {"argv": sys.argv, "lib": str(src.lib), "ckpt": str(src.ckpt_dir), "noise_seeds": seeds, "bank_sha": bank.sha256,
           "episodes": src.eps, "stage0": s0_rows, "stage1": s1_rows, "zerok": zerok, "cover": cover_tot,
           "act_std_mean": src.ns_actions_std, "lines": lines, "blocking": [V.block_pass, V.block_total]}
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    (out_path.parent / (out_path.stem + ".summary.txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if s1_rows:
        np.savez(out_path.parent / (out_path.stem + ".npz"),
                 share_step=np.asarray([r["share_step_full"] for r in s1_rows]),
                 deltas=np.asarray([[r["deltas"][sc] for sc in SCOPES] for r in s1_rows]),
                 k=np.asarray([r["k"] for r in s1_rows]), task=np.asarray([r["task"] for r in s1_rows]),
                 stage0_d=np.asarray([[r["d_mask"], r["d_swap"], r["d_noise"]] for r in s0_rows]),
                 stage0_k=np.asarray([r["k"] for r in s0_rows]))
    return 0 if V.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
