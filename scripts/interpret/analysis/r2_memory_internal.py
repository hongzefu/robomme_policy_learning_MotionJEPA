"""R2：记忆区内部谁读谁（attn-ctx512-vs-motion-1103 第二轮可解释性分析）。

问题：prefill 阶段 prefix 内部完整 attention（prefix_<seq>.npz 的 probs_prefix，f16，
(18 层, 8 头, P query, P key)）里，query 与 key 都在记忆区时，谁在读 VIS／SWAP？

分组（每次推理按该次的 frame_times／motion_times 现算，记忆位 s 的身份见计划 §2.3）：
- 帧 token（mem_order[s] < 512，帧 f = mem_order[s] // 16，全域帧号 frame_times[f]），
  一帧只进一个组，优先级 最近 > VIS > SWAP1 > SWAP2 > 中段：
  recent = frame_times 最大的 4 帧；VIS = [0,32)；SWAP1 = [64,114)；SWAP2 = [114,164)
  （ep4 只交换 1 次，没有 SWAP2，[114,164) 的帧记入中段）；mid = 其余帧。
- motion token（mem_order[s] ≥ 512，窗 w = mem_order[s] − 512，窗口 [motion_times[w], +33)）
  按重叠帧数／33 分到 motVIS／motSWAP／motOther 三组（分数权重，三者和为 1）。
- 聚合列：SWAP = SWAP1 + SWAP2；frames = 全部帧 token；motion = 全部 motion token。

度量：对每层每头，质量矩阵 M[gq, gk] = Σ_q Σ_k wq[q,gq]·A[q,k]·wk[k,gk] / Σ_q wq[q,gq]
（该 query 组平均一行落在该 key 组的质量）；均匀期望 U[gk] = Σ_k wk[k,gk] / 合法 key 数
（合法 key = 有效记忆位；自检证实记忆 query 对当前图像／文本 key 的质量恒为 0，即记忆区只看记忆区）；
富集倍数 E = M / U。整局汇总按 query 权重对 20 次推理加权平均 M 与 U 后再相除。

两阶段：`compute`（逐次推理流式读 prefix npz、f32、用完释放，结果缓存到 --cache）与
`report`（读缓存出图、写 r2_summary.json、末行判定）。可只算某一局（--episode）。
分流复用 scripts/interpret/aggregate_attn.py 的常量与 _mem_layout（只 import 不改）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "interpret"))
from aggregate_attn import FRAME_TOKENS, MOTION_WINDOW, TOKENS_PER_FRAME, _mem_layout  # noqa: E402

STAGE1B = Path("/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/interpret-1103/stage1b")
RUNS = {
    # 名称: (attn 目录, 图注, 交换次数)
    "ctx_ep8": ("ctx/attn/0621439bab554595b3ee0f0f9414ba84", "基线 ep8（成功）", 2),
    "ctx_ep4": ("ctx/attn/3a79ca2d563c4a679d51013e15394130", "基线 ep4（失败，交换 1 次）", 1),
    "mot_ep8": ("mot/attn/afb7438311e44dfab40a03e709c2ab16", "motion ep8（成功）", 2),
}
N_INFER = 20
KEY = [15, 16, 17, 18, 19]
KEY_MAIN = 18
N_RECENT = 4
VIS = (0, 32)
SWAPS = [(64, 114), (114, 164)]
BASE_GROUPS = ["recent", "mid", "VIS", "SWAP1", "SWAP2", "motVIS", "motSWAP", "motOther"]
AGG_GROUPS = ["SWAP", "frames", "motion"]
GROUPS = BASE_GROUPS + AGG_GROUPS
GI = {g: i for i, g in enumerate(GROUPS)}
NAME_ZH = {"recent": "最近4帧", "mid": "中段帧", "VIS": "VIS帧", "SWAP1": "SWAP1帧", "SWAP2": "SWAP2帧",
           "motVIS": "motion·VIS", "motSWAP": "motion·SWAP", "motOther": "motion·其余",
           "SWAP": "SWAP帧", "frames": "全部帧", "motion": "全部motion"}
# 热图坐标轴用的短名
SHORT = {"recent": "最近", "mid": "中段", "VIS": "VIS", "SWAP1": "S1", "SWAP2": "S2",
         "motVIS": "mV", "motSWAP": "mS", "motOther": "m其"}
# 关键通路 (query 组, key 组, 中文名)
PATHS = [("recent", "VIS", "最近帧→VIS"), ("recent", "SWAP", "最近帧→SWAP"),
         ("motSWAP", "SWAP", "motion(SWAP)→SWAP帧"), ("frames", "motion", "帧→motion")]
ROWSUM_TOL = 5e-4
OUTSIDE_TOL = 1e-6


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def group_weights(z, n_swap: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 (有效记忆位下标, 权重矩阵 W (n_valid, len(GROUPS)), 帧分组信息)。"""
    P, MEM, mem_order, mem_mask = _mem_layout(z)
    ft = np.asarray(z["frame_times"], np.int64)
    mt = np.asarray(z["motion_times"], np.int64).reshape(-1)
    swaps = SWAPS[:n_swap]
    valid_ft = np.where(ft >= 0)[0]
    order = valid_ft[np.argsort(ft[valid_ft])]
    recent = set(order[-N_RECENT:].tolist())
    fgroup = {}
    for f in valid_ft:
        t = ft[f]
        if f in recent:
            g = "recent"
        elif VIS[0] <= t < VIS[1]:
            g = "VIS"
        elif n_swap >= 1 and swaps[0][0] <= t < swaps[0][1]:
            g = "SWAP1"
        elif n_swap >= 2 and swaps[1][0] <= t < swaps[1][1]:
            g = "SWAP2"
        else:
            g = "mid"
        fgroup[int(f)] = g
    idx = np.where(mem_mask)[0]
    W = np.zeros((idx.size, len(GROUPS)), np.float64)
    for r, s in enumerate(idx):
        o = int(mem_order[s])
        if o < FRAME_TOKENS:
            f = o // TOKENS_PER_FRAME
            if f not in fgroup:
                raise ValueError(f"有效帧 token 指向无效帧 f={f}（frame_times={ft[f]}）")
            W[r, GI[fgroup[f]]] = 1.0
            W[r, GI["frames"]] = 1.0
        else:
            w = o - FRAME_TOKENS
            t0 = int(mt[w])
            if t0 < 0:
                raise ValueError(f"有效 motion token 指向无效窗 w={w}")
            fv = _overlap(t0, t0 + MOTION_WINDOW, *VIS) / MOTION_WINDOW
            fs = sum(_overlap(t0, t0 + MOTION_WINDOW, a, b) for a, b in swaps) / MOTION_WINDOW
            W[r, GI["motVIS"]] = fv
            W[r, GI["motSWAP"]] = fs
            W[r, GI["motOther"]] = 1.0 - fv - fs
            W[r, GI["motion"]] = 1.0
    W[:, GI["SWAP"]] = W[:, GI["SWAP1"]] + W[:, GI["SWAP2"]]
    info = {"frame_times": ft.tolist(),
            "groups": {g: sorted(int(ft[f]) for f, gg in fgroup.items() if gg == g)
                       for g in ["recent", "VIS", "SWAP1", "SWAP2", "mid"]},
            "motion_times": [int(x) for x in mt if x >= 0]}
    return idx, W, info


def compute_one(d: Path, seq: int, n_swap: int) -> dict:
    z = np.load(d / f"infer_{seq:04d}.npz")
    P, MEM, _, mem_mask = _mem_layout(z)
    idx, W, info = group_weights(z, n_swap)
    with np.load(d / f"prefix_{seq:04d}.npz") as pz:
        if int(pz["infer_seq"]) != seq or int(pz["P"]) != P:
            raise ValueError("prefix 与 infer 元数据不符")
        pp = pz["probs_prefix"]  # (18, 8, P, P) f16
    L, H = pp.shape[:2]
    rows = pp[:, :, idx, :].astype(np.float32)  # 有效记忆 query 行，全部 prefix key
    del pp
    rowsum = rows.sum(-1, dtype=np.float64)
    rowsum_err = float(np.abs(rowsum - 1.0).max())
    in_mem = rows[..., idx].sum(-1, dtype=np.float64)
    outside = float((rowsum - in_mem).max())  # 落在当前图像／文本／记忆 padding 的质量
    img_txt = float(rows[..., MEM:].sum(-1, dtype=np.float64).max())
    A = rows[..., idx]  # (L, H, nq, nk)
    del rows
    Wf = W.astype(np.float32)
    AK = A @ Wf  # (L, H, nq, G)
    num = np.einsum("qg,lhqk->lhgk", Wf, AK, dtype=np.float64)  # (L,H,Gq,Gk)
    qw = W.sum(0)  # (Gq,)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = num / qw[None, None, :, None]
    U = W.sum(0) / idx.size
    # 头内 top 读者：最近帧→VIS 的逐 query token 不必存，存逐层逐头 M 即可
    out = {"M": M.astype(np.float32), "qw": qw, "U": U, "rowsum_err": rowsum_err,
           "outside": outside, "img_txt": img_txt, "n_valid": int(idx.size),
           "step_idx": int(z["step_idx"]), "info": info, "L": L, "H": H}
    del A, AK
    return out


def cmd_compute(args) -> int:
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    names = [args.episode] if args.episode else list(RUNS)
    for name in names:
        rel, _, n_swap = RUNS[name]
        d = STAGE1B / rel
        Ms, qws, Us, meta = [], [], [], []
        for seq in range(N_INFER):
            r = compute_one(d, seq, n_swap)
            Ms.append(r["M"]); qws.append(r["qw"]); Us.append(r["U"])
            meta.append({k: r[k] for k in ["rowsum_err", "outside", "img_txt", "n_valid", "step_idx", "info"]})
            print(f"[{name}] #{seq} step={r['step_idx']} rowsum_err={r['rowsum_err']:.2e} "
                  f"outside={r['outside']:.2e} recent→VIS E(L均)="
                  f"{np.nanmean(r['M'][:, :, GI['recent'], GI['VIS']].mean(1)) / max(r['U'][GI['VIS']], 1e-12):.2f}",
                  flush=True)
        np.savez(cache / f"{name}.npz", M=np.stack(Ms), qw=np.stack(qws), U=np.stack(Us))
        (cache / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False))
    print("R2_COMPUTE=PASS episodes=%d" % len(names))
    return 0


# ---------------------------------------------------------------- 汇总

def load_cache(cache: Path, name: str):
    z = np.load(cache / f"{name}.npz")
    meta = json.loads((cache / f"{name}.json").read_text())
    return z["M"], z["qw"], z["U"], meta  # M (I,L,H,G,G) qw (I,G) U (I,G)


def enrich(M, U):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(U > 0, M / np.where(U > 0, U, 1), np.nan)


def episode_agg(M, qw, U, infers):
    """按 query 权重对推理加权平均：返回 (Mbar (L,H,G,G), Ubar (G,G))。"""
    M = M[infers]; qw = qw[infers]; U = U[infers]
    wq = qw[:, None, None, :, None]
    Mn = np.nan_to_num(M) * wq
    den = qw.sum(0)  # (G,)
    with np.errstate(invalid="ignore", divide="ignore"):
        Mbar = Mn.sum(0) / den[None, None, :, None]
        Ubar = (U[:, None, :] * qw[:, :, None]).sum(0) / den[:, None]  # (Gq, Gk)
    return Mbar, Ubar


def _r(x, n=4):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


def cmd_report(args) -> int:
    from plot_attn import CJK  # noqa: F401  导入即设置中文字体
    import matplotlib.pyplot as plt

    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {n: load_cache(cache, n) for n in RUNS}
    summary = {"question": "记忆区内部（prefill，query 与 key 都在记忆区）谁在读 VIS／SWAP",
               "groups": {g: NAME_ZH[g] for g in GROUPS}, "n_recent": N_RECENT,
               "intervals": {"VIS": VIS, "SWAP1": SWAPS[0], "SWAP2": SWAPS[1]},
               "checks": {}, "episodes": {}}
    ok = True
    for name, (M, qw, U, meta) in data.items():
        rs = max(m["rowsum_err"] for m in meta)
        outside = max(m["outside"] for m in meta)
        img_txt = max(m["img_txt"] for m in meta)
        good = rs <= ROWSUM_TOL and outside <= OUTSIDE_TOL
        ok &= good
        summary["checks"][name] = {"rowsum_err_max": rs, "outside_mem_mass_max": outside,
                                   "image_text_mass_max": img_txt, "pass": bool(good)}
        ep = {"label": RUNS[name][1], "infers": []}
        Ma = M.mean(2)  # 头平均 (I,L,G,G)
        for i in range(N_INFER):
            E = enrich(Ma[i], U[i][None, None, :])
            rec = {"seq": i, "step_idx": meta[i]["step_idx"], "frame_groups": meta[i]["info"]["groups"],
                   "paths": {}}
            for gq, gk, zh in PATHS:
                m_l = Ma[i, :, GI[gq], GI[gk]]
                rec["paths"][f"{gq}->{gk}"] = {
                    "mass_layer_mean": _r(np.nanmean(m_l)) if qw[i, GI[gq]] > 0 else None,
                    "uniform": _r(U[i, GI[gk]]),
                    "enrich_layer_mean": _r(np.nanmean(E[:, GI[gq], GI[gk]])) if qw[i, GI[gq]] > 0 else None,
                    "mass_by_layer": [_r(x) for x in m_l] if qw[i, GI[gq]] > 0 else None,
                }
            ep["infers"].append(rec)
        for tag, infers in [("key15_19", KEY), ("all", list(range(N_INFER)))]:
            Mbar, Ubar = episode_agg(M, qw, U, infers)
            Mh = Mbar.mean(1)  # (L,G,G)
            E = enrich(Mh, Ubar[None])
            Eh = enrich(Mbar, Ubar[None, None])
            blk = {"uniform": {f"{g}": _r(Ubar[GI['recent'], GI[g]]) for g in GROUPS}, "paths": {}}
            for gq, gk, zh in PATHS:
                e_l = E[:, GI[gq], GI[gk]]
                if not np.all(np.isfinite(e_l)):
                    blk["paths"][f"{gq}->{gk}"] = None
                    continue
                eh = Eh[:, :, GI[gq], GI[gk]]
                li, hi = np.unravel_index(np.nanargmax(eh), eh.shape)
                blk["paths"][f"{gq}->{gk}"] = {
                    "name": zh, "mass_layer_mean": _r(Mh[:, GI[gq], GI[gk]].mean()),
                    "uniform": _r(Ubar[GI[gq], GI[gk]]), "enrich_layer_mean": _r(np.nanmean(e_l)),
                    "enrich_by_layer": [_r(x, 3) for x in e_l],
                    "mass_by_layer": [_r(x) for x in Mh[:, GI[gq], GI[gk]]],
                    "max_layer": int(np.nanargmax(e_l)), "max_layer_enrich": _r(np.nanmax(e_l), 3),
                    "max_head": f"L{li}H{hi}", "max_head_enrich": _r(eh[li, hi], 3),
                    "max_head_mass": _r(Mbar[li, hi, GI[gq], GI[gk]]),
                }
            # 最近帧行在各 key 组的层平均质量与富集
            rr = {}
            for g in GROUPS:
                if Ubar[GI["recent"], GI[g]] > 0:
                    rr[g] = {"mass": _r(Mh[:, GI["recent"], GI[g]].mean()),
                             "enrich": _r(np.nanmean(E[:, GI["recent"], GI[g]]), 3)}
            blk["recent_row"] = rr
            if name.startswith("mot"):
                mrow = {}
                for gq in ["motVIS", "motSWAP", "motOther", "motion"]:
                    mrow[gq] = {g: {"mass": _r(Mh[:, GI[gq], GI[g]].mean()),
                                    "enrich": _r(np.nanmean(E[:, GI[gq], GI[g]]), 3)}
                                for g in GROUPS if Ubar[GI[gq], GI[g]] > 0}
                blk["motion_rows"] = mrow
            ep[tag] = blk
        summary["episodes"][name] = ep

    figs = []
    # 图 1～3：#18 每层 query 组×key 组富集热图（log2，nearest）
    for name, (M, qw, U, meta) in data.items():
        groups = [g for g in BASE_GROUPS if qw[KEY_MAIN, GI[g]] > 0]
        gi = [GI[g] for g in groups]
        Ma = M[KEY_MAIN].mean(1)  # (L,G,G)
        E = enrich(Ma, U[KEY_MAIN][None, None, :])[:, gi][:, :, gi]
        L = E.shape[0]
        fig, axes = plt.subplots(3, 6, figsize=(22, 12.5))
        fig.subplots_adjust(wspace=0.25, hspace=0.3, top=0.9)
        for l in range(L):
            ax = axes.flat[l]
            im = ax.imshow(np.log2(np.clip(E[l], 1 / 64, 64)), cmap="RdBu_r", vmin=-6, vmax=6,
                           interpolation="nearest")
            ax.set_title(f"L{l}", fontsize=11)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([SHORT[g] for g in groups], fontsize=8)
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels([SHORT[g] for g in groups], fontsize=8)
            for a in range(len(groups)):
                for b in range(len(groups)):
                    v = E[l, a, b]
                    if np.isfinite(v):
                        ax.text(b, a, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center",
                                fontsize=5.5, color="k" if abs(np.log2(max(v, 1e-9))) < 4 else "w")
        fig.colorbar(im, ax=axes, shrink=0.6, label="log2 富集倍数（行=query 组，列=key 组）")
        fig.suptitle(f"{RUNS[name][1]}：推理 #{KEY_MAIN}（step {meta[KEY_MAIN]['step_idx']}）记忆区内部"
                     f" query 组→key 组富集倍数（头平均；1=均匀）\n短名：最近=最近4帧，中段=中段帧，VIS／S1／S2=VIS／SWAP1／SWAP2 帧，"
                     f"mV／mS／m其=motion 窗按重叠分到 VIS／SWAP／其余", fontsize=13)
        p = out / f"fig_r2_1_heatmap_{name}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig); figs.append(p)

    colors = {"ctx_ep8": "tab:blue", "ctx_ep4": "tab:red", "mot_ep8": "tab:green"}
    # 图 4：关键通路随层（#18 与整局）
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True)
    for c, (gq, gk, zh) in enumerate(PATHS):
        for r, tag in enumerate(["k18", "all"]):
            ax = axes[r, c]
            for name, (M, qw, U, meta) in data.items():
                infers = [KEY_MAIN] if tag == "k18" else list(range(N_INFER))
                Mbar, Ubar = episode_agg(M, qw, U, infers)
                if Ubar[GI[gq], GI[gk]] <= 0 or not np.isfinite(Ubar[GI[gq], GI[gk]]):
                    continue
                e = Mbar.mean(1)[:, GI[gq], GI[gk]] / Ubar[GI[gq], GI[gk]]
                if np.all(np.isfinite(e)):
                    ax.plot(range(len(e)), np.clip(e, 1e-2, None), "o-", color=colors[name],
                            label=RUNS[name][1], ms=4)
            ax.axhline(1, color="0.5", ls="--", lw=1)
            ax.set_yscale("log")
            ax.set_xticks(range(0, 18, 2))
            ax.set_title(f"{zh}（{'#18' if tag == 'k18' else '整局 20 次'}）", fontsize=11)
            if r == 1:
                ax.set_xlabel("层")
            if c == 0:
                ax.set_ylabel("富集倍数（头平均，log 轴，<0.01 记为 0.01）")
            ax.legend(fontsize=8)
    fig.suptitle("记忆区内部关键通路富集倍数随层变化（虚线=均匀）", fontsize=14)
    p = out / "fig_r2_2_paths_by_layer.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig); figs.append(p)

    # 图 5：关键通路随推理（层平均与头平均）
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True)
    for c, (gq, gk, zh) in enumerate(PATHS):
        for name, (M, qw, U, meta) in data.items():
            Ma = M.mean(2)
            xs, em, mm = [], [], []
            for i in range(N_INFER):
                if qw[i, GI[gq]] <= 0 or U[i, GI[gk]] <= 0:
                    continue
                m = float(np.nanmean(Ma[i, :, GI[gq], GI[gk]]))
                xs.append(meta[i]["step_idx"]); mm.append(m); em.append(m / U[i, GI[gk]])
            if xs:
                axes[0, c].plot(xs, mm, "o-", color=colors[name], label=RUNS[name][1], ms=4)
                axes[1, c].plot(xs, em, "o-", color=colors[name], label=RUNS[name][1], ms=4)
        axes[0, c].set_title(f"{zh}：质量（层均）", fontsize=11)
        axes[1, c].set_title(f"{zh}：富集倍数（层均质量／均匀）", fontsize=11)
        axes[1, c].axhline(1, color="0.5", ls="--", lw=1)
        axes[1, c].set_xlabel("推理 step_idx")
        for ax in axes[:, c]:
            for s in [240, 256, 272, 288, 304]:
                ax.axvline(s, color="0.85", lw=0.8, zorder=0)
            ax.legend(fontsize=8)
    fig.suptitle("记忆区内部关键通路随推理变化（灰竖线=#15～#19）", fontsize=14)
    p = out / "fig_r2_3_paths_by_infer.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig); figs.append(p)

    # 图 6：#18 逐层逐头富集（最近帧→VIS／最近帧→SWAP，三局 + motion 两条）
    panels = [(n, gq, gk, zh) for n in RUNS for gq, gk, zh in PATHS[:2]] + \
             [("mot_ep8", gq, gk, zh) for gq, gk, zh in PATHS[2:]]
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    for ax, (name, gq, gk, zh) in zip(axes.flat, panels):
        M, qw, U, meta = data[name]
        Mbar, Ubar = episode_agg(M, qw, U, [KEY_MAIN])
        e = Mbar[:, :, GI[gq], GI[gk]] / Ubar[GI[gq], GI[gk]]
        im = ax.imshow(np.log2(np.clip(e, 1 / 64, 64)), cmap="RdBu_r", vmin=-6, vmax=6,
                       interpolation="nearest", aspect="auto")
        ax.set_title(f"{RUNS[name][1]}\n{zh}", fontsize=10)
        ax.set_xlabel("头"); ax.set_ylabel("层")
        ax.set_xticks(range(8)); ax.set_yticks(range(0, 18, 2))
        for l in range(e.shape[0]):
            for h in range(e.shape[1]):
                ax.text(h, l, f"{e[l, h]:.1f}" if e[l, h] < 10 else f"{e[l, h]:.0f}",
                        ha="center", va="center", fontsize=5.5)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="log2 富集倍数")
    fig.suptitle(f"推理 #{KEY_MAIN} 逐层逐头富集倍数", fontsize=14)
    p = out / "fig_r2_4_heads_k18.png"
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig); figs.append(p)

    (out / "r2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=False))
    json.loads((out / "r2_summary.json").read_text())
    for n, c in summary["checks"].items():
        print(f"CHECK {n}: rowsum_err_max={c['rowsum_err_max']:.2e} outside={c['outside_mem_mass_max']:.2e} "
              f"img_txt={c['image_text_mass_max']:.2e}")
    print(f"R2_MEMORY_INTERNAL={'PASS' if ok else 'FAIL'} episodes={len(data)} figs={len(figs)}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute"); c.add_argument("--cache", required=True)
    c.add_argument("--episode", choices=list(RUNS))
    r = sub.add_parser("report"); r.add_argument("--cache", required=True); r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return cmd_compute(args) if args.cmd == "compute" else cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
