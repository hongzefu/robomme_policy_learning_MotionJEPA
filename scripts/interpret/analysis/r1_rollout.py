"""R1：attention rollout——基线「只看最近帧」是否借记忆区内部的 prefill attention 间接读到 VIS／SWAP。

只读输入：stage1b 三局的 infer_*.npz（动作 query 的 attention）与 prefix_*.npz（prefill 阶段 prefix 内部
完整 attention，(18 层, 8 头, P, P) f16）；不跑仿真、不用 GPU，逐次推理流式读、f64 计算、用完即释放。

口径（Abnar & Zuidema 2020 attention rollout 的 prefix 版）：
- 有效 prefix 位置 V = 记忆区 mem_mask 为真的位置 + 当前图像 512 + 前 txt_len 个文本位；padding 行列剔除。
- 每层 Ã_l = 0.5·(8 头平均 probs_prefix_l[V,V]，行先归一) + 0.5·I，再行归一；R_0 = I，R_{l+1} = Ã_l·R_l。
  R_l[k, j] = 第 l 层输入（＝ prefill 第 l 层 KV 的来源）在位置 k 上对原始输入 token j 的近似贡献。
- 动作在第 l 层读到的是 prefill 第 l 层的 KV，即第 l 层输入表示，故有效贡献 c_l[j] = Σ_k a_l[k]·R_l[k, j]，
  a_l = 第 l 层动作 query（10 去噪步 × 20 动作 query × 8 头平均）对 prefix 的 attention；c = 18 层平均。
  动作对自身（suffix）key 的质量原样保留在 action 流，使 c 总和仍为 1，可直接复用 split_streams 分流。
- 直接版 d = 18 层平均 a_l（与 split_streams 全层全头平均一致）。
- 联合 rollout（补充口径）：把 20 个动作 token 也放进图里，第 l 层动作行 = [a_l 对 prefix, a_l 对动作]，
  同样 0.5·A + 0.5·I；取最后一层后 20 个动作行的平均，含「动作→动作→prefix」路径。
- 中介分解（对照）：c_l[j] 按中介 k 的类别拆成「经记忆区 k」「经当前图像 k」「经文本 k」三份；
  「经记忆区」一份恰是「只用记忆区子块的 rollout」（记忆区行只看记忆区，子块对乘法封闭）。

分流复用 scripts/interpret/aggregate_attn.py::split_streams（只 import 不改）；区间与 motion 窗计权同 X3：
VIS=[0,32)，SWAP1=[64,114)，SWAP2=[114,164)（基线 ep4 只有 SWAP1），motion 窗按与区间重叠帧数／33 计权。
输出到 --out（PNG、r1_summary.json），末行 R1_ROLLOUT=PASS|FAIL。
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "interpret"))
from aggregate_attn import MOTION_WINDOW, split_streams  # noqa: E402
from plot_attn import CJK  # noqa: E402  （导入即设置中文字体）

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

STAGE1B = Path("/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/interpret-1103/stage1b")
RUNS = {
    # 名称: (attn 目录, 图注, 交换区间)
    "ctx_ep8": ("ctx/attn/0621439bab554595b3ee0f0f9414ba84", "基线 ep8（成功）", [(64, 114), (114, 164)]),
    "ctx_ep4": ("ctx/attn/3a79ca2d563c4a679d51013e15394130", "基线 ep4（失败，交换 1 次）", [(64, 114)]),
    "mot_ep8": ("mot/attn/afb7438311e44dfab40a03e709c2ab16", "motion ep8（成功）", [(64, 114), (114, 164)]),
}
VIS = (0, 32)
KEY = [15, 16, 17, 18, 19]
FOCUS = 18
N_INFER = 20
IMG = 512
ROW_TOL = 1e-4
COLORS = {"direct": "#1c7ed6", "rollout": "#e8590c", "joint": "#2f9e44", "uniform": "#868e96"}


# ---------------------------------------------------------------- 计算

def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def shares(st: dict, swaps) -> dict:
    """帧流：VIS／SWAPj 质量占帧流比例与均匀期望；motion 流：按重叠帧数／33 计权（X3 修正口径）。"""
    fr = st["frame"]
    tot_f = sum(x["mass"] for x in fr)
    m = st["mass"]
    res = {"frame_total": tot_f, "motion_total": m["motion"], "image": m["image_base"] + m["image_wrist"],
           "text": m["text"], "action": m["action"]}

    def fshare(lo, hi):
        hit = [x for x in fr if lo <= x["t"] < hi]
        return (sum(x["mass"] for x in hit) / tot_f if tot_f > 0 else None, len(hit) / len(fr) if fr else None)

    segs = {"vis": [VIS], **{f"swap{j + 1}": [sw] for j, sw in enumerate(swaps)}, "swap": list(swaps)}
    for key, ss in segs.items():
        s = u = 0.0
        for a, b in ss:
            si, ui = fshare(a, b)
            s, u = s + si, u + ui
        res[f"{key}_frame_share"], res[f"{key}_frame_uniform"] = s, u
    mo = st["motion"]
    tot_m = sum(x["mass"] for x in mo)
    for key, ss in segs.items():
        if mo and tot_m > 0:
            w = [sum(overlap(x["t"], x["t"] + MOTION_WINDOW, a, b) for a, b in ss) / MOTION_WINDOW for x in mo]
            res[f"{key}_motion_share"] = sum(wi * x["mass"] for wi, x in zip(w, mo)) / tot_m
            res[f"{key}_motion_uniform"] = sum(w) / len(mo)
        else:
            res[f"{key}_motion_share"] = res[f"{key}_motion_uniform"] = None
    return res


def streams_from_vec(meta: dict, vec: np.ndarray) -> dict:
    fake = dict(meta)
    fake["probs_action"] = vec.reshape(1, 1, 1, 1, -1)
    return split_streams(fake, query="action")


def desink(meta: dict, vec: np.ndarray, swaps) -> dict:
    """X4 同口径的去 sink 对照：每帧剔除帧内最大的单个 token 后重算帧流 VIS／SWAP 占比；
    另给出帧流最大帧及其单 token 占该帧比例（判断是否是单 token sink）。"""
    mo = np.asarray(meta["mem_order"], np.int64)
    mm = np.asarray(meta["mem_mask"], bool)
    ft = np.asarray(meta["frame_times"], np.int64)
    MEM = int(meta["MEM"])
    mem = vec[:MEM]
    sel = mm & (mo < 512)
    f = mo[sel] // 16
    m = mem[sel]
    tot = np.zeros(32)
    mx = np.zeros(32)
    np.add.at(tot, f, m)
    np.maximum.at(mx, f, m)
    present = np.unique(f)
    rest = tot - mx
    t = ft[present]

    def share(arr, lo, hi):
        s = arr[present].sum()
        return float(arr[present][(t >= lo) & (t < hi)].sum() / s) if s > 0 else None

    top = int(present[np.argmax(tot[present])])
    out = {"vis_frame_share_desink": share(rest, *VIS),
           "swap_frame_share_desink": sum(share(rest, a, b) or 0.0 for a, b in swaps),
           "top_frame_t": int(ft[top]), "top_frame_share": float(tot[top] / tot[present].sum()),
           "top_frame_single_token": float(mx[top] / tot[top]) if tot[top] > 0 else None}
    return out


def process_infer(d: Path, i: int, swaps) -> dict:
    z = np.load(d / f"infer_{i:04d}.npz")
    meta = {k: z[k] for k in z.files if k not in ("probs_action", "probs_text", "top_frames")}
    P, MEM, txt_len = int(z["P"]), int(z["MEM"]), int(z["txt_len"])
    pa = z["probs_action"].astype(np.float64).mean(axis=(0, 2))  # (18, 20, P+20)：去噪步、头平均
    del z
    a_l = pa.mean(axis=1)  # (18, P+20)
    zp = np.load(d / f"prefix_{i:04d}.npz")
    assert int(zp["P"]) == P and int(zp["MEM"]) == MEM and int(zp["infer_seq"]) == i
    pp = zp["probs_prefix"]  # (18, 8, P, P) f16
    L = pp.shape[0]
    mem_mask = np.asarray(meta["mem_mask"], bool)
    valid = np.concatenate([mem_mask, np.ones(IMG, bool), np.arange(64) < txt_len])
    V = np.where(valid)[0]
    nV = V.size
    cat = np.zeros(P, np.int8)  # 0 记忆 1 图像 2 文本
    cat[MEM:MEM + IMG] = 1
    cat[MEM + IMG:] = 2
    catV = cat[V]

    R = np.eye(nV)
    Rj = np.eye(nV + 20)  # 联合图：prefix 有效位 + 20 动作
    c_layers = np.zeros((L, P + 20))
    via = np.zeros((L, 3, P))  # 按中介类别拆的 c_l（prefix 部分）
    row_err = 0.0
    eyeV = np.eye(nV)
    for l in range(L):
        # c_l 用第 l 层输入的 R_l（R_0 = I）
        al = a_l[l]
        alV = al[V]
        c = np.zeros(P + 20)
        c[V] = alV @ R
        c[P:] = al[P:]
        c_layers[l] = c
        for t in range(3):
            via[l, t, V] = (alV * (catV == t)) @ R
        A = pp[l].astype(np.float32).mean(axis=0).astype(np.float64)[np.ix_(V, V)]
        A /= A.sum(axis=1, keepdims=True)
        At = 0.5 * A + 0.5 * eyeV
        At /= At.sum(axis=1, keepdims=True)
        R = At @ R
        row_err = max(row_err, float(np.abs(R.sum(axis=1) - 1).max()))
        # 联合图第 l 层
        J = np.zeros((nV + 20, nV + 20))
        J[:nV, :nV] = At
        qa = pa[l]  # (20, P+20)
        J[nV:, :nV] = 0.5 * qa[:, V]
        J[nV:, nV:] = 0.5 * qa[:, P:] + 0.5 * np.eye(20)
        J[nV:] /= J[nV:].sum(axis=1, keepdims=True)
        Rj = J @ Rj
        row_err = max(row_err, float(np.abs(Rj.sum(axis=1) - 1).max()))
    del pp, zp
    joint = np.zeros(P + 20)
    jr = Rj[nV:].mean(axis=0)
    joint[V] = jr[:nV]
    joint[P:] = jr[nV:].sum() / 20.0  # 动作输入自身质量均摊到 20 列（只用于 action 流计数）

    direct = a_l.mean(axis=0)
    roll = c_layers.mean(axis=0)
    out = {"infer_seq": i, "step_idx": int(meta["step_idx"]), "row_err": row_err}
    st_d, st_r, st_j = streams_from_vec(meta, direct), streams_from_vec(meta, roll), streams_from_vec(meta, joint)
    out["direct"], out["rollout"], out["joint"] = shares(st_d, swaps), shares(st_r, swaps), shares(st_j, swaps)
    for kind, vec in (("direct", direct), ("rollout", roll), ("joint", joint)):
        out[kind].update(desink(meta, vec, swaps))
    # 中介分解：只看记忆区目标 j 的质量与 VIS／SWAP 占比
    via_m = via.mean(axis=0)
    out["via"] = {}
    for t, name in enumerate(["mem", "image", "text"]):
        vec = np.zeros(P + 20)
        vec[:P] = via_m[t]
        vec[P:] = (1 - vec.sum()) / 20.0  # 填平总和，只为通过 split_streams 的和检查；不解读 action 流
        st = streams_from_vec(meta, vec)
        sh = shares(st, swaps)
        out["via"][name] = {"mem_mass": float(via_m[t][:MEM].sum()), "frame_total": sh["frame_total"],
                            "motion_total": sh["motion_total"], "vis_frame_share": sh["vis_frame_share"],
                            "swap_frame_share": sh["swap_frame_share"], "vis_motion_share": sh["vis_motion_share"],
                            "swap_motion_share": sh["swap_motion_share"]}
    # 逐层
    out["layers"] = []
    for l in range(L):
        sd = shares(streams_from_vec(meta, a_l[l]), swaps)
        sr = shares(streams_from_vec(meta, c_layers[l]), swaps)
        out["layers"].append({"direct_vis": sd["vis_frame_share"], "rollout_vis": sr["vis_frame_share"],
                              "direct_swap": sd["swap_frame_share"], "rollout_swap": sr["swap_frame_share"],
                              "direct_frame_total": sd["frame_total"], "rollout_frame_total": sr["frame_total"]})
    if i == FOCUS:
        out["frames"] = {"direct": [(x["t"], x["mass"]) for x in st_d["frame"]],
                         "rollout": [(x["t"], x["mass"]) for x in st_r["frame"]],
                         "joint": [(x["t"], x["mass"]) for x in st_j["frame"]]}
        out["motion_windows"] = {"direct": [(x["t"], x["mass"]) for x in st_d["motion"]],
                                 "rollout": [(x["t"], x["mass"]) for x in st_r["motion"]]}
        # 记忆区内部最终 R（第 18 层输出）按帧聚合：行=帧（16 token 平均），列=帧（16 token 求和）
        mo = np.asarray(meta["mem_order"], np.int64)
        ft = np.asarray(meta["frame_times"], np.int64)
        memV = np.where(catV == 0)[0]
        orig = mo[V[memV]]
        isf = orig < 512
        fidx = orig[isf] // 16
        Rm = R[np.ix_(memV[isf], memV[isf])]
        nf = 32
        M = np.zeros((nf, nf))
        cnt = np.zeros(nf)
        for r_, fr_ in enumerate(fidx):
            np.add.at(M[fr_], fidx, Rm[r_])
            cnt[fr_] += 1
        M = M / np.maximum(cnt, 1)[:, None]
        order = np.argsort(ft)
        to_motion = float(R[np.ix_(memV[isf], memV[~isf])].sum(axis=1).mean()) if (~isf).any() else 0.0
        out["mem_R"] = {"times": ft[order].tolist(), "M": M[np.ix_(order, order)].tolist(),
                        "self_frame": float(np.mean(np.diag(M))), "frame_rows_to_motion": to_motion,
                        "n_motion_tokens": int((~isf).sum())}
    return out


def compute(stride: int) -> dict:
    res = {}
    for name, (rel, _, swaps) in RUNS.items():
        d = STAGE1B / rel
        seqs = sorted(set(list(range(0, N_INFER, stride)) + KEY))
        res[name] = {}
        for i in seqs:
            res[name][i] = process_infer(d, i, swaps)
            r = res[name][i]
            print(f"[{name}] #{i:02d} step={r['step_idx']} VIS 帧流 直接={r['direct']['vis_frame_share']:.3f} "
                  f"rollout={r['rollout']['vis_frame_share']:.3f} 联合={r['joint']['vis_frame_share']:.3f} "
                  f"均匀={r['direct']['vis_frame_uniform']:.3f} row_err={r['row_err']:.1e}", flush=True)
    return res


# ---------------------------------------------------------------- 画图

def _shade(ax, swaps):
    ax.axvspan(*VIS, color="#40c057", alpha=0.12)
    for a, b in swaps:
        ax.axvspan(a, b, color="#fab005", alpha=0.12)


def fig_curves(res, out: Path) -> str:
    cols = [("vis_frame", "帧流 VIS 占比"), ("swap_frame", "帧流 SWAP（全部交换）占比"),
            ("motion", "motion 流 VIS／SWAP（窗计权）")]
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    for r, (name, (_, label, _)) in enumerate(RUNS.items()):
        seqs = sorted(res[name])
        for c, (key, title) in enumerate(cols):
            ax = axes[r, c]
            if key == "motion":
                if all(res[name][i]["direct"]["vis_motion_share"] is None for i in seqs):
                    ax.text(0.5, 0.5, "无 motion 流", ha="center", va="center", transform=ax.transAxes, fontsize=13)
                    ax.set_yticks([])
                else:
                    nn = lambda v: np.nan if v is None else v  # noqa: E731
                    for seg, ls in (("vis", "-"), ("swap", "--")):
                        for kind in ("direct", "rollout"):
                            ax.plot(seqs, [nn(res[name][i][kind][f"{seg}_motion_share"]) for i in seqs], ls,
                                    marker="o", ms=3, color=COLORS[kind],
                                    label=f"{seg.upper()} {'直接' if kind == 'direct' else 'rollout'}")
                        ax.plot(seqs, [nn(res[name][i]["direct"][f"{seg}_motion_uniform"]) for i in seqs], ls,
                                color=COLORS["uniform"], label=f"{seg.upper()} 均匀")
                    ax.set_ylim(-0.02, 1.38)
                    ax.legend(fontsize=8, ncol=3, loc="upper center")
            else:
                for kind, lab in (("direct", "直接 attention"), ("rollout", "rollout（层平均）"),
                                  ("joint", "联合 rollout")):
                    ax.plot(seqs, [res[name][i][kind][f"{key}_share"] for i in seqs], "o-", ms=3,
                            color=COLORS[kind], label=lab)
                ax.plot(seqs, [res[name][i]["rollout"][f"{key}_share_desink"] for i in seqs], ":", marker="x",
                        ms=4, color=COLORS["rollout"], label="rollout 去 sink（每帧剔最大 token）")
                ax.plot(seqs, [res[name][i]["direct"][f"{key}_uniform"] for i in seqs], "--",
                        color=COLORS["uniform"], label="均匀期望")
                if r == 0 and c == 0:
                    ax.legend(fontsize=9)
            ax.axvspan(14.5, 19.5, color="#e9ecef", zorder=0)
            ax.set_title(f"{label}：{title}", fontsize=11)
            ax.grid(alpha=0.3)
            if r == 2:
                ax.set_xlabel("推理序号（step = 16×序号；灰底 #15～#19）")
    fig.tight_layout()
    p = out / "fig_r1_1_share_curves.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


def fig_focus_frames(res, out: Path) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(15, 11))
    for ax, (name, (_, label, swaps)) in zip(axes, RUNS.items()):
        r = res[name][FOCUS]
        _shade(ax, swaps)
        w = 2.2
        for kind, off in (("direct", -w / 2), ("rollout", w / 2)):
            ts = np.array([t for t, _ in r["frames"][kind]])
            ms = np.array([m for _, m in r["frames"][kind]])
            ax.bar(ts + off, ms / ms.sum() * 100, width=w, color=COLORS[kind],
                   label=("直接 attention" if kind == "direct" else "rollout 有效贡献") + "（帧流内 %）")
        u = 100 / len(r["frames"]["direct"])
        ax.axhline(u, color=COLORS["uniform"], ls="--", lw=1, label=f"均匀 {u:.1f}%")
        ax.set_title(f"{label}：推理 #{FOCUS}（step {r['step_idx']}）帧流按全域帧号；绿底 VIS，黄底 SWAP", fontsize=11)
        ax.set_xlim(-5, r["step_idx"] + 6)
        ax.set_ylabel("占帧流 %")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9, loc="upper left")
    axes[-1].set_xlabel("全域帧号")
    fig.tight_layout()
    p = out / "fig_r1_2_focus18_frames.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


def fig_routing(res, out: Path) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    names = {"mem": "经记忆区中介", "image": "经当前图像中介", "text": "经文本中介"}
    colors = {"mem": "#7048e8", "image": "#1098ad", "text": "#f08c00"}
    for c, (name, (_, label, _)) in enumerate(RUNS.items()):
        seqs = sorted(res[name])
        ax = axes[0, c]
        bottom = np.zeros(len(seqs))
        for t in ("mem", "text", "image"):
            v = np.array([res[name][i]["via"][t]["mem_mass"] for i in seqs]) * 100
            ax.bar(seqs, v, bottom=bottom, color=colors[t], label=names[t])
            bottom += v
        ax.plot(seqs, [(res[name][i]["direct"]["frame_total"] + res[name][i]["direct"]["motion_total"]) * 100
                       for i in seqs], "k.-", label="直接 attention 记忆区总质量")
        ax.set_title(f"{label}：rollout 落在记忆区的质量（% of 动作 attention）", fontsize=10)
        ax.set_xlabel("推理序号")
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, max(bottom.max(), 1e-9) * 1.45)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax = axes[1, c]
        for t, ls in (("mem", "-"), ("text", "--")):
            ax.plot(seqs, [res[name][i]["via"][t]["vis_frame_share"] for i in seqs], ls, marker="o", ms=3,
                    color=colors[t], label=f"{names[t]}：帧流 VIS 占比")
        ax.plot(seqs, [res[name][i]["direct"]["vis_frame_share"] for i in seqs], ":", marker="s", ms=3,
                color=COLORS["direct"], label="直接 attention：帧流 VIS 占比")
        ax.plot(seqs, [res[name][i]["direct"]["vis_frame_uniform"] for i in seqs], "--", color=COLORS["uniform"],
                label="均匀期望")
        ax.set_title(f"{label}：各中介通路的 VIS 占比", fontsize=10)
        ax.set_xlabel("推理序号")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = out / "fig_r1_3_routing.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


def fig_layers(res, out: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (name, (_, label, _)) in zip(axes, RUNS.items()):
        ly = res[name][FOCUS]["layers"]
        x = np.arange(len(ly))
        ax.plot(x, [v["direct_vis"] for v in ly], "o-", color=COLORS["direct"], label="直接 a_l：帧流 VIS 占比")
        ax.plot(x, [v["rollout_vis"] for v in ly], "o-", color=COLORS["rollout"], label="rollout c_l：帧流 VIS 占比")
        ax.axhline(res[name][FOCUS]["direct"]["vis_frame_uniform"], color=COLORS["uniform"], ls="--", label="均匀")
        ax2 = ax.twinx()
        ax2.bar(x, [v["direct_frame_total"] * 100 for v in ly], color="#adb5bd", alpha=0.35, width=0.6)
        ax2.set_ylabel("直接 a_l 帧流总质量 %（灰柱）", fontsize=9)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_title(f"{label}：#{FOCUS} 逐层", fontsize=11)
        ax.set_xlabel("层 l")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("占帧流比例")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p = out / "fig_r1_4_layers_focus18.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


def fig_mem_R(res, out: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, (name, (_, label, _)) in zip(axes, RUNS.items()):
        m = res[name][FOCUS]["mem_R"]
        M = np.asarray(m["M"])
        im = ax.imshow(M, interpolation="nearest", cmap="magma", vmin=0, vmax=min(1.0, float(np.percentile(M, 99.5))))
        ticks = list(range(0, len(m["times"]), 4))
        ax.set_xticks(ticks)
        ax.set_xticklabels([m["times"][t] for t in ticks], fontsize=8, rotation=45)
        ax.set_yticks(ticks)
        ax.set_yticklabels([m["times"][t] for t in ticks], fontsize=8)
        ax.set_xlabel("来源帧（全域帧号，16 token 求和）")
        ax.set_ylabel("记忆帧位置（16 token 平均）")
        extra = (f"；帧行落到 {m['n_motion_tokens']} 个 motion token 的质量 {m['frame_rows_to_motion'] * 100:.1f}%"
                 if m['n_motion_tokens'] else "")
        ax.set_title(f"{label}：#{FOCUS} 末层 R 记忆区帧×帧\n对角均值 {m['self_frame']:.3f}{extra}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = out / "fig_r1_5_mem_R_focus18.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


# ---------------------------------------------------------------- 主程序

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1, help="整局曲线每隔几次推理算一次（#15～#19 总算）")
    ap.add_argument("--cache", type=Path, default=None, help="可选：计算结果缓存（pickle，放 scratchpad）")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.cache and args.cache.is_file():
        res = pickle.loads(args.cache.read_bytes())
    else:
        res = compute(args.stride)
        if args.cache:
            args.cache.write_bytes(pickle.dumps(res))
    row_err = max(r["row_err"] for e in res.values() for r in e.values())
    figs = [fig_curves(res, out), fig_focus_frames(res, out), fig_routing(res, out), fig_layers(res, out),
            fig_mem_R(res, out)]
    summary = {"口径": "Ã_l=0.5·头平均A_l+0.5·I 行归一；R_0=I；c_l=a_l·R_l；c=层平均；联合 rollout 含动作行",
               "row_err_max": row_err, "row_tol": ROW_TOL, "figs": figs,
               "episodes": {n: {"label": RUNS[n][1], "swaps": RUNS[n][2], "infers": res[n]} for n in RUNS}}
    (out / "r1_summary.json").write_text(json.dumps(_clean(summary), ensure_ascii=False, indent=1, allow_nan=False))
    ok = row_err <= ROW_TOL and len(res) == 3
    print(f"ROW_SUM_CHECK max_err={row_err:.2e} tol={ROW_TOL}")
    print(f"R1_ROLLOUT={'PASS' if ok else 'FAIL'} episodes={len(res)} figs={len(figs)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
