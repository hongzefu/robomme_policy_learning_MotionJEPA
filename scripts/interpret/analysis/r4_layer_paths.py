"""R4：逐层通路分解——动作读取 VIS／SWAP 信息是「直接」还是「经由记忆／文本等中转 token 间接」。

数据（只读）：每局目录 infer_<seq>.npz（probs_action (10 去噪步, 18 层, 8 头, 20 动作 query, P+20 key)）
与 prefix_<seq>.npz（probs_prefix (18 层, 8 头, P, P)，prefill 时 prefix 内部完整注意力，padding query 行为 0）。
prefix = [记忆区 MEM | 当前图像 512 | 文本 64]；记忆区内部双向，图像看不到记忆，文本看记忆与图像，动作看全部。

定义（全部为 attention-only 近似，忽略 MLP、残差强度与 value 向量范数）：
- 目标 T ∈ {VIS=[0,32), SWAP1=[64,114), SWAP2=[114,164), SWAP=[64,64+50k)}。每个 prefix token 一个目标权重 g_T(s)：
  帧 token：帧时刻 t∈T 记 1，否则 0；motion token（窗 w 起点 t_w、长 33）：|[t_w,t_w+33)∩T|/33；其余 token 为 0。
- 第 l 层动作读取 a_l：probs_action 在全部去噪步、全部 20 个动作 query、8 头上平均，长度 P+20（与
  aggregate_attn.aggregate(layers=l) 完全同口径，并逐层用 split_streams 交叉核对帧流 VIS 质量）。
- 直接 D_l(T) = Σ_s a_l[s]·g_T(s)，再拆成帧流部分与 motion 流部分。
- 一跳间接 I_l(T) = Σ_k a_l[k]·(1−g_T(k))·R_{<l}[k,T]，其中中转 token k 的「已携带 T 信息量」
  R_{<l}[k,T] = (1/l)·Σ_{m=0}^{l−1} B_m[k,:]·g_T，B_m 是第 m 层 prefix 注意力的头平均（对动作 token k，
  B_m 换成同一去噪步第 m 层动作 query k 的读取，按步、按 query 配对后再平均）。
  选「第 0～l−1 层均值」而不是只取第 l−1 层：残差流把 k 在任意更早层读到的内容都保留到第 l 层输入，
  只取 l−1 层会漏掉「浅层读入、深层被读出」的通路（正是 motion 模型 L3→L16 这类）；取均值而不是求和，
  保证 R 仍在 [0,1]，与直接质量同量纲可比。只取 l−1 层的口径作为敏感性对照一并输出（ind_adj）。
- 两跳 I2_l(T) = Σ_k Σ_j a_l[k](1−g(k))·B_{l−1}[k,j](1−g(j))·B_{l−2}[j,T]（相邻层口径，只给总量）。
- 中转类别：最近帧（时刻最晚的 4 个有效帧）、中段帧（其余非目标帧，含另一区间的帧）、motion 窗、当前图像、文本、动作 token。
  当前图像按掩码看不到记忆，它的一跳间接恒为 0（作为掩码自检）。
- 均匀期望：同一行注意力若在帧流内均匀分配、在 motion 流内均匀分配时落在 T 上的质量：
  E = 帧流质量·n_T帧/n帧 + motion 流质量·mean_w g_T(w)。直接与间接各自算，倍数 = 实测 / 期望。

输出 <out>/r4_summary.json（严格 JSON，NaN 写 null）与若干 PNG；末行 R4_LAYER_PATHS=PASS episodes=<n> figs=<n>。
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from aggregate_attn import (FRAME_TOKENS, IMAGE_BASE, IMAGE_WRIST, MOTION_WINDOW, SWAP_LEN,  # noqa: E402
                            SWAP_START, TOKENS_PER_FRAME, VISIBLE, key_step_shares, split_streams)
import plot_attn  # noqa: E402  导入即设置中文字体

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

N_LAYERS = 18
KEY = 18
NEIGH = (15, 16, 17, 18, 19)
RECENT_N = 4
CATS = ["recent", "mid", "motion", "image", "text", "action"]
CAT_ZH = {"recent": "最近帧", "mid": "中段帧", "motion": "motion 窗", "image": "当前图像",
          "text": "文本", "action": "动作 token"}
CAT_COLOR = {"recent": "#4C78A8", "mid": "#9ECAE9", "motion": "#F58518", "image": "#B279A2",
             "text": "#54A24B", "action": "#BAB0AC"}
D_COLOR = {"frame": "#222222", "motion": "#E45756"}
KEY_LAYERS = (3, 16)


def targets_for(k: int) -> dict[str, tuple[int, int]]:
    t = {"VIS": VISIBLE}
    for j in range(1, k + 1):
        t[f"SWAP{j}"] = (SWAP_START + SWAP_LEN * (j - 1), SWAP_START + SWAP_LEN * j)
    t["SWAP"] = (SWAP_START, SWAP_START + SWAP_LEN * k)
    return t


def _ov(t0: int, length: int, lo: int, hi: int) -> int:
    return max(0, min(t0 + length, hi) - max(t0, lo))


def layout(z) -> dict:
    """逐 token 的类别与属性（长度 P+20）。"""
    P, MEM = int(z["P"]), int(z["MEM"])
    keys = set(z.files)
    mem_order = np.asarray(z["mem_order"], np.int64) if "mem_order" in keys else np.arange(MEM)
    mem_mask = np.asarray(z["mem_mask"], bool) if "mem_mask" in keys else np.ones(MEM, bool)
    ft = np.asarray(z["frame_times"], np.int64)
    mt = np.asarray(z["motion_times"], np.int64).reshape(-1)
    txt_len = int(z["txt_len"])
    n = P + 20
    kind = np.full(n, "pad", dtype=object)
    tt = np.full(n, -1, np.int64)      # 帧时刻或 motion 窗起点
    for s in range(MEM):
        if not mem_mask[s]:
            continue
        o = mem_order[s]
        if o < FRAME_TOKENS:
            kind[s] = "frame"; tt[s] = ft[o // TOKENS_PER_FRAME]
        else:
            kind[s] = "motion"; tt[s] = mt[o - FRAME_TOKENS]
    kind[MEM:MEM + IMAGE_BASE + IMAGE_WRIST] = "image"
    t0 = MEM + IMAGE_BASE + IMAGE_WRIST
    kind[t0:t0 + txt_len] = "text"
    kind[P:] = "action"
    valid_ft = sorted({int(x) for x in ft if x >= 0})
    recent_t = set(valid_ft[-RECENT_N:])
    return {"P": P, "MEM": MEM, "kind": kind, "t": tt, "recent_t": recent_t,
            "n_frames": len(valid_ft), "frame_t": valid_ft,
            "motion_t": [int(x) for x, m in zip(mt, np.asarray(z["motion_mask"], bool).reshape(-1)) if m]}


def target_weights(L: dict, lo: int, hi: int) -> dict:
    kind, tt = L["kind"], L["t"]
    g = np.zeros(kind.shape[0])
    fr = kind == "frame"
    mo = kind == "motion"
    g[fr] = ((tt[fr] >= lo) & (tt[fr] < hi)).astype(float)
    g[mo] = np.array([_ov(int(x), MOTION_WINDOW, lo, hi) / MOTION_WINDOW for x in tt[mo]])
    n_t = sum(1 for x in L["frame_t"] if lo <= x < hi)
    u_f = n_t / L["n_frames"] if L["n_frames"] else float("nan")
    u_m = (float(np.mean([_ov(x, MOTION_WINDOW, lo, hi) / MOTION_WINDOW for x in L["motion_t"]]))
           if L["motion_t"] else 0.0)
    # 中转类别（对该目标而言）：目标帧不作中转；motion 窗以 (1−g) 作中转权重。
    cat = np.full(kind.shape[0], "", dtype=object)
    for s in np.where(fr)[0]:
        if g[s] > 0:
            continue
        cat[s] = "recent" if int(tt[s]) in L["recent_t"] else "mid"
    cat[mo] = "motion"
    for c in ("image", "text", "action"):
        cat[kind == c] = c
    return {"g": g, "u_f": u_f, "u_m": u_m, "n_t": n_t, "cat": cat, "fr": fr, "mo": mo}


def _nan(x):
    return None if (x is None or not np.isfinite(x)) else round(float(x), 7)


def analyse_infer(z, zp, k: int) -> dict:
    L = layout(z)
    P = L["P"]
    A = np.asarray(z["probs_action"], np.float32).mean(axis=2)          # (10,18,20,P+20) 头平均
    a = A.astype(np.float64).mean(axis=(0, 2))                            # (18,P+20)
    B = np.asarray(zp["probs_prefix"], np.float32).mean(axis=1).astype(np.float64)  # (18,P,P)
    out = {"step_idx": int(z["step_idx"]), "infer_seq": int(z["infer_seq"]), "targets": {}}
    # 自检 1：逐层 split_streams 帧流质量与本脚本一致；图像行看不到记忆。
    base = {kk: z[kk] for kk in z.files if kk not in ("probs_action", "probs_text")}
    fr_mask = L["kind"] == "frame"
    for li in range(N_LAYERS):
        fake = dict(base); fake["probs_action"] = a[li][None, None, None, None, :]
        ss = split_streams(fake)
        assert abs(ss["mass"]["frame"] - a[li][fr_mask].sum()) < 1e-6
    img = L["kind"][:P] == "image"
    memv = np.isin(L["kind"][:P], ["frame", "motion"])
    img_to_mem = float(B[:, img][:, :, memv].sum(axis=2).max())
    out["check_image_to_mem_max"] = img_to_mem
    # 全层平均直接帧流 VIS 占比（与上一轮 KEY_STEP 同口径）
    fake = dict(base); fake["probs_action"] = a.mean(axis=0)[None, None, None, None, :]
    out["fullmean_frame_vis_share"] = key_step_shares(split_streams(fake), k)["vis_share"]

    for name, (lo, hi) in targets_for(k).items():
        W = target_weights(L, lo, hi)
        g, fr, mo = W["g"], W["fr"], W["mo"]
        gP = g[:P]
        if W["n_t"] == 0 and not np.any(g[mo] > 0):
            out["targets"][name] = None
            continue
        frP, moP = fr[:P], mo[:P]
        # 每层 prefix 行对 T、对帧流、对 motion 流的质量 (18,P)
        rT = B @ gP
        rF = B[:, :, frP].sum(axis=2)
        rM = B[:, :, moP].sum(axis=2)
        # 动作 token 行：(10,18,20)
        aT = A[..., :P].astype(np.float64) @ gP
        aF = A[..., :P][..., frP].sum(axis=-1).astype(np.float64)
        aM = A[..., :P][..., moP].sum(axis=-1).astype(np.float64)
        res = {k_: [] for k_ in ["direct", "direct_frame", "direct_motion", "exp_direct", "ind_total",
                                 "exp_ind", "ind_adj", "twohop"] + [f"ind_{c}" for c in CATS]}
        relay_w = 1.0 - g          # 中转权重
        for li in range(N_LAYERS):
            al = a[li]
            d_f = float((al * g)[fr].sum()); d_m = float((al * g)[mo].sum())
            res["direct"].append(d_f + d_m); res["direct_frame"].append(d_f); res["direct_motion"].append(d_m)
            res["exp_direct"].append(float(al[fr].sum() * W["u_f"] + al[mo].sum() * W["u_m"]))
            if li == 0:
                for c in CATS:
                    res[f"ind_{c}"].append(0.0)
                for kk in ("ind_total", "exp_ind", "ind_adj", "twohop"):
                    res[kk].append(0.0)
                continue
            R = rT[:li].mean(axis=0); RF = rF[:li].mean(axis=0); RM = rM[:li].mean(axis=0)
            w = al[:P] * relay_w[:P]
            contrib = w * R
            expc = w * (RF * W["u_f"] + RM * W["u_m"])
            # 动作中转：按去噪步、按 query 配对
            Aq = A[:, li, :, P:].astype(np.float64)                         # (10,20q,20j)
            Ract = aT[:, :li].mean(axis=1)                                  # (10,20j)
            RactE = aF[:, :li].mean(axis=1) * W["u_f"] + aM[:, :li].mean(axis=1) * W["u_m"]
            act = float(np.einsum("sqj,sj->", Aq, Ract) / (Aq.shape[0] * Aq.shape[1]))
            actE = float(np.einsum("sqj,sj->", Aq, RactE) / (Aq.shape[0] * Aq.shape[1]))
            catP = W["cat"][:P]
            tot = 0.0
            for c in CATS:
                v = act if c == "action" else float(contrib[catP == c].sum())
                res[f"ind_{c}"].append(v); tot += v
            res["ind_total"].append(tot)
            res["exp_ind"].append(float(expc.sum()) + actE)
            Radj = rT[li - 1]
            res["ind_adj"].append(float((w * Radj).sum()) + float(
                np.einsum("sqj,sj->", Aq, aT[:, li - 1]) / (Aq.shape[0] * Aq.shape[1])))
            if li >= 2:
                M1 = B[li - 1] * relay_w[None, :P]                            # k→j（j 非目标）
                res["twohop"].append(float(w @ (M1 @ rT[li - 2])))
            else:
                res["twohop"].append(0.0)
        out["targets"][name] = {kk: [_nan(x) for x in v] for kk, v in res.items()}
        out["targets"][name]["u_f"] = _nan(W["u_f"]); out["targets"][name]["u_m"] = _nan(W["u_m"])
    del A, B
    return out


def _arr(ep, tgt, key):
    """(20 推理, 18 层) 数组，缺失为 NaN。"""
    rows = []
    for r in ep["infers"]:
        t = r["targets"].get(tgt)
        rows.append([np.nan] * N_LAYERS if t is None else [np.nan if x is None else x for x in t[key]])
    return np.array(rows, float)


def _ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 1e-12, num / den, np.nan)


# ---------------------------------------------------------------- 绘图

def fig_key_stack(eps, labels, path):
    tg = ["VIS", "SWAP1", "SWAP2"]
    fig, axes = plt.subplots(len(tg), len(labels), figsize=(5.2 * len(labels), 3.3 * len(tg)), squeeze=False)
    x = np.arange(N_LAYERS)
    for ci, lab in enumerate(labels):
        ep = eps[lab]
        r = ep["infers"][KEY]
        for ri, t in enumerate(tg):
            ax = axes[ri, ci]
            d = r["targets"].get(t)
            if d is None:
                ax.text(0.5, 0.5, f"本局无 {t}（k={ep['k']}）", ha="center", va="center", transform=ax.transAxes,
                        color="0.4", fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title(ep["title"], fontsize=10)
                continue
            f = lambda key: np.array([0 if v is None else v for v in d[key]])  # noqa: E731
            w = 0.4
            ax.bar(x - w / 2, f("direct_frame"), w, color=D_COLOR["frame"], label="直接·帧流")
            ax.bar(x - w / 2, f("direct_motion"), w, bottom=f("direct_frame"), color=D_COLOR["motion"],
                   label="直接·motion 流")
            bot = np.zeros(N_LAYERS)
            for c in CATS:
                v = f(f"ind_{c}")
                ax.bar(x + w / 2, v, w, bottom=bot, color=CAT_COLOR[c], label=f"间接·经{CAT_ZH[c]}")
                bot += v
            ax.plot(x - w / 2, f("exp_direct"), "_", color="#D62728", ms=9, mew=2, label="直接均匀期望")
            ax.plot(x + w / 2, f("exp_ind"), "_", color="#2CA02C", ms=9, mew=2, label="间接均匀期望")
            for kl in KEY_LAYERS:
                ax.axvline(kl, color="0.7", lw=0.6, ls=":")
            ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x], fontsize=7)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(f"{ep['title']} · {t}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"落到 {t} 的注意力质量", fontsize=9)
            if ri == len(tg) - 1:
                ax.set_xlabel("层（左柱=直接，右柱=一跳间接）", fontsize=9)
    h, lab_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"推理 #{KEY}（step 288）逐层：直接 vs 一跳间接（按中转类别堆叠），点线为 L3／L16", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=130); plt.close(fig)


def fig_curves(eps, labels, path):
    colors = {"base_ep8": "#4C78A8", "base_ep4": "#9D755D", "mot_ep8": "#E45756"}
    rows = [("VIS", "direct", "VIS 直接"), ("VIS", "ind_total", "VIS 一跳间接"),
            ("SWAP", "direct", "SWAP 并集 直接"), ("SWAP", "ind_total", "SWAP 并集 一跳间接")]
    cols = [("L3", 3), ("L16", 16), ("全层均值", None)]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(15, 12), squeeze=False)
    for ri, (t, key, rl) in enumerate(rows):
        ekey = "exp_direct" if key == "direct" else "exp_ind"
        for ci, (cl, li) in enumerate(cols):
            ax = axes[ri, ci]
            for lab in labels:
                ep = eps[lab]
                m = _arr(ep, t, key); e = _arr(ep, t, ekey)
                if li is None:
                    m = np.nanmean(m, axis=1); e = np.nanmean(e, axis=1)
                else:
                    m = m[:, li]; e = e[:, li]
                ax.plot(np.arange(len(m)), _ratio(m, e), "-o", ms=3, color=colors.get(lab), label=ep["title"])
            ax.axhline(1, color="0.5", lw=0.8, ls="--")
            ax.axvspan(14.5, 19.5, color="#fbd38d", alpha=0.35, lw=0)
            ax.axvline(KEY, color="0.3", lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.set_title(f"{rl} · {cl}", fontsize=10)
            ax.set_xticks(range(0, 20, 2))
            if ci == 0:
                ax.set_ylabel("相对均匀期望的倍数（log）", fontsize=9)
            if ri == len(rows) - 1:
                ax.set_xlabel("推理序号", fontsize=9)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("关键层随推理变化：实测 / 均匀期望（橙底 #15～#19，点线 #18）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120); plt.close(fig)


def fig_heat(eps, labels, path):
    fig, axes = plt.subplots(2, len(labels), figsize=(5 * len(labels), 7.5), squeeze=False)
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    im = None
    for ci, lab in enumerate(labels):
        ep = eps[lab]
        for ri, (key, ekey, nm) in enumerate((("direct", "exp_direct", "直接"), ("ind_total", "exp_ind", "一跳间接"))):
            r = np.log2(np.clip(_ratio(_arr(ep, "VIS", key), _arr(ep, "VIS", ekey)), 1 / 8, 8)).T
            ax = axes[ri, ci]
            im = ax.imshow(r, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest", origin="lower")
            ax.set_title(f"{ep['title']} · VIS {nm}", fontsize=10)
            ax.set_xlabel("推理序号", fontsize=9); ax.set_ylabel("层", fontsize=9)
            ax.set_yticks(range(0, 18, 3))
            ax.axvline(KEY, color="k", lw=0.6, ls=":")
    fig.subplots_adjust(hspace=0.38, top=0.9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="log2(实测/均匀期望)，截断 ±3")
    fig.suptitle("VIS：层 × 推理 的富集倍数（最近邻显示）", fontsize=12)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def fig_ep4_vs_ep8(eps, path):
    sel = list(range(14, 18))
    panels = [("VIS", "direct", "VIS 直接"), ("VIS", "ind_total", "VIS 一跳间接"),
              ("SWAP1", "direct", "SWAP1 直接"), ("SWAP1", "ind_total", "SWAP1 一跳间接")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for i, (t, key, nm) in enumerate(panels):
        ax = axes[i // 2, i % 2]
        ekey = "exp_direct" if key == "direct" else "exp_ind"
        for lab, col in (("base_ep8", "#4C78A8"), ("base_ep4", "#9D755D")):
            m = _arr(eps[lab], t, key)[sel]; e = _arr(eps[lab], t, ekey)[sel]
            ax.plot(range(N_LAYERS), np.nanmean(m, 0), "-o", ms=3, color=col, label=f"{eps[lab]['title']} 实测")
            ax.plot(range(N_LAYERS), np.nanmean(e, 0), "--", color=col, lw=1, label=f"{eps[lab]['title']} 均匀期望")
        ax.set_title(f"{nm}（#14～#17 平均）", fontsize=10)
        ax.set_xlabel("层", fontsize=9); ax.set_ylabel("注意力质量", fontsize=9)
        ax.set_xticks(range(N_LAYERS))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("基线 ctx512：成功 ep8 vs 失败 ep4，#14～#17 逐层直接／间接质量", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------- 汇总

def summarize(ep) -> dict:
    s = {"k": ep["k"], "title": ep["title"],
         "fullmean_frame_vis_share_infer18": _nan(ep["infers"][KEY]["fullmean_frame_vis_share"]),
         "check_image_to_mem_max": max(r["check_image_to_mem_max"] for r in ep["infers"]), "by_target": {}}
    for t in targets_for(ep["k"]):
        blk = {}
        for scope, idx in (("infer18", [KEY]), ("infer15_19", list(NEIGH)), ("episode", list(range(20))),
                           ("infer14_17", list(range(14, 18)))):
            row = {}
            for key in ["direct", "direct_frame", "direct_motion", "exp_direct", "ind_total", "exp_ind",
                        "ind_adj", "twohop"] + [f"ind_{c}" for c in CATS]:
                m = _arr(ep, t, key)[idx]
                with np.errstate(all="ignore"):
                    per_layer = np.nanmean(m, axis=0) if np.isfinite(m).any() else np.full(N_LAYERS, np.nan)
                row[key] = [_nan(x) for x in per_layer]
            d = np.array([np.nan if x is None else x for x in row["direct"]])
            ed = np.array([np.nan if x is None else x for x in row["exp_direct"]])
            it = np.array([np.nan if x is None else x for x in row["ind_total"]])
            ei = np.array([np.nan if x is None else x for x in row["exp_ind"]])
            row["enrich_direct"] = [_nan(x) for x in _ratio(d, ed)]
            row["enrich_ind"] = [_nan(x) for x in _ratio(it, ei)]
            with np.errstate(all="ignore"):
                row["layer_mean"] = {"direct": _nan(np.nanmean(d)), "exp_direct": _nan(np.nanmean(ed)),
                                     "ind_total": _nan(np.nanmean(it)), "exp_ind": _nan(np.nanmean(ei)),
                                     "enrich_direct": _nan(np.nanmean(d) / np.nanmean(ed)),
                                     "enrich_ind": _nan(np.nanmean(it) / np.nanmean(ei))}
                cat_mean = {c: _nan(np.nanmean([np.nan if x is None else x for x in row[f"ind_{c}"]])) for c in CATS}
            row["layer_mean"]["ind_by_cat"] = cat_mean
            blk[scope] = row
        s["by_target"][t] = blk
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ep", action="append", required=True, help="label=目录:k:标题")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    eps, labels = {}, []
    for spec in a.ep:
        lab, rest = spec.split("=", 1)
        d, k, title = rest.split(":", 2)
        k = int(k)
        infers = []
        for i in range(20):
            with np.load(Path(d) / f"infer_{i:04d}.npz") as z, np.load(Path(d) / f"prefix_{i:04d}.npz") as zp:
                assert int(zp["infer_seq"]) == int(z["infer_seq"]) == i and int(zp["P"]) == int(z["P"])
                infers.append(analyse_infer(z, zp, k))
        eps[lab] = {"k": k, "title": title, "dir": d, "infers": infers}
        labels.append(lab)
        print(f"已处理 {lab}：20 次推理，#18 全层平均帧流 VIS 占比 = {infers[KEY]['fullmean_frame_vis_share']:.4f}，"
              f"图像→记忆最大质量 = {max(r['check_image_to_mem_max'] for r in infers):.2e}", flush=True)
    figs = []
    p = a.out / "fig_r4_infer18_layer_paths.png"; fig_key_stack(eps, labels, p); figs.append(p)
    p = a.out / "fig_r4_key_layers_curves.png"; fig_curves(eps, labels, p); figs.append(p)
    p = a.out / "fig_r4_vis_heat_layer_infer.png"; fig_heat(eps, labels, p); figs.append(p)
    if "base_ep8" in eps and "base_ep4" in eps:
        p = a.out / "fig_r4_base_ep8_vs_ep4_infer14_17.png"; fig_ep4_vs_ep8(eps, p); figs.append(p)
    summary = {"definitions": {
        "direct": "a_l·g_T，a_l = probs_action 全去噪步／全 query／头平均",
        "indirect_1hop": "Σ_k a_l[k](1−g_T(k))·mean_{m<l}(B_m[k,:]·g_T)；动作 token 中转按去噪步与 query 配对",
        "ind_adj": "同上但只用 B_{l−1}（敏感性对照）",
        "twohop": "Σ_k Σ_j a_l[k](1−g(k))B_{l−1}[k,j](1−g(j))B_{l−2}[j,T]",
        "uniform": "帧流质量·n_T/n帧 + motion 流质量·mean_w g_T(w)",
        "recent_frames": RECENT_N, "approximation": "attention-only，忽略 MLP、残差强度与 value 范数"},
        "episodes": {lab: summarize(eps[lab]) for lab in labels},
        "per_infer": {lab: [{"infer_seq": r["infer_seq"], "step_idx": r["step_idx"],
                             "fullmean_frame_vis_share": _nan(r["fullmean_frame_vis_share"]),
                             "targets": r["targets"]} for r in eps[lab]["infers"]] for lab in labels}}
    txt = json.dumps(summary, ensure_ascii=False, allow_nan=False)
    (a.out / "r4_summary.json").write_text(txt)
    json.loads(txt)
    print(f"R4_LAYER_PATHS=PASS episodes={len(labels)} figs={len(figs)}")
    return 0


warnings.filterwarnings("ignore", message="Mean of empty slice")


if __name__ == "__main__":
    raise SystemExit(main())
