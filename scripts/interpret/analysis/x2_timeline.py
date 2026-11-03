"""X2 整局时间线：20 次推理里模型何时开始、何时停止回看裸露窗与交换区间。

计划 1103-interpret-motion-attention-plan.html 第二部分 §1.3（npz 键）、§2.3（分流）。
分流一律复用 scripts/interpret/aggregate_attn.py::split_streams（只 import，不改），
动作 query、全部去噪步／层／头平均（split_streams 默认口径）。

输入三局（只读）：基线 ctx512 ep4（失败，k=1）、基线 ctx512 ep8（成功，k=2）、ctx512+motion ep8（成功，k=2）。
输出（--out 目录）：
- x2_heatmap_frame_raw.png   推理序号 × 全域帧号，帧流每帧 16 token 注意力之和（对数色标，三局共用色标）
- x2_heatmap_frame_rank.png  同上，行内排名归一化（论文 arXiv 2610.14406 Figure 9 呈现法，紫→红）
- x2_heatmap_motion.png      motion 模型 motion 流同型热图（窗口按起点画、宽 33），真实质量与行内排名两版
- x2_share_curves.png        每次推理 VIS／SWAP 占比与均匀期望 + 阶段标注 + 夹爪／TCP z
- x2_stream_stack.png        五流（帧／motion／base 图／wrist 图／文本／动作）质量随推理堆叠
- x2_summary.json            逐推理数字与按阶段汇总
末行 X2_TIMELINE=PASS episodes=3 infers=60 figs=<n>。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
import plot_attn  # noqa: E402,F401  导入即按本机字体设置 CJK 中文 rcParams
from aggregate_attn import MOTION_WINDOW, VISIBLE, split_streams, swap_interval  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CJK = plot_attn.CJK
ROOT = Path("/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/interpret-1103/stage1")
EPISODES = [  # (键, 标签, 相对目录, 交换次数 k)
    ("ctx_ep4", "基线 ctx512 ep4（失败，k=1）", "ctx/attn/1d677dadd3f34872a5779dbe8ab815e4", 1),
    ("ctx_ep8", "基线 ctx512 ep8（成功，k=2）", "ctx/attn/3625db1fb81a48af8aa5a136b3a7aa68", 2),
    ("mot_ep8", "ctx512+motion ep8（成功，k=2）", "mot/attn/af10850c1f7d47549fcd8934178a7497", 2),
]
KEY_INFER = 18          # ep8 关键推理（step 288，抬起容器）
GRIP_CLOSED = 0.03      # state[7] 手指开度 < 3 cm 视作闭合
N_COLS = 320            # 热图列数（全域帧号 0…319，局末 step 304 后留白）
PHASES = ["靠近按钮", "按按钮", "张开靠近", "下降闭合", "抬起"]
PHASE_COLOR = {"靠近按钮": "#d9d9d9", "按按钮": "#9ecae1", "张开靠近": "#c7e9c0",
               "下降闭合": "#fff176", "抬起": "#e377c2"}
VIS_C, SWAP1_C, SWAP2_C = "#2ca02c", "#ff7f0e", "#d62728"


def swap_segments(k: int) -> list[tuple[int, int]]:
    lo, _ = swap_interval(k)
    return [(lo + 50 * j, lo + 50 * (j + 1)) for j in range(k)]


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def phase_of(npzs: list) -> list[str]:
    """由 state[7]（手指开度）与 actions[:,7]（1 开 −1 闭）推阶段。"""
    out, seen_close, reopened = [], False, False
    for z in npzs:
        g = float(z["state"][7])
        cmd = np.asarray(z["actions"])[:, 7]
        if g < GRIP_CLOSED:
            seen_close = True
            out.append("抬起" if reopened else "按按钮")
        else:
            if seen_close:
                reopened = True
            if not seen_close:
                out.append("靠近按钮")
            else:
                out.append("下降闭合" if np.any(cmd < 0) else "张开靠近")
    return out


def frame_cells(frames: list[dict], step: int) -> list[tuple[int, int, float]]:
    """每个采样帧填到下一采样帧之前；最后一帧填宽取该行中位间距。"""
    fr = sorted(frames, key=lambda x: x["t"])
    ts = [x["t"] for x in fr]
    gaps = np.diff(ts) if len(ts) > 1 else np.array([1])
    last_w = max(1, int(np.median(gaps)))
    cells = []
    for i, x in enumerate(fr):
        t1 = ts[i + 1] if i + 1 < len(ts) else x["t"] + last_w
        cells.append((x["t"], max(t1, x["t"] + 1), x["mass"]))
    return cells


def rank_norm(vals: np.ndarray) -> np.ndarray:
    if vals.size == 1:
        return np.ones(1)
    r = np.argsort(np.argsort(vals))
    return r / (vals.size - 1)


def build_grid(rows: list[list[tuple[int, int, float]]], rank: bool) -> np.ndarray:
    g = np.full((len(rows), N_COLS), np.nan)
    for i, cells in enumerate(rows):
        if not cells:
            continue
        v = np.array([c[2] for c in cells])
        v = rank_norm(v) if rank else v
        for (t0, t1, _), val in zip(cells, v):
            g[i, t0:min(t1, N_COLS)] = val
    return g


def episode_record(key: str, label: str, rel: str, k: int) -> dict:
    d = ROOT / rel
    files = sorted(d.glob("infer_*.npz"))
    npzs = [np.load(f) for f in files]
    streams = [split_streams(z) for z in npzs]
    meta = json.loads((d / "episode.json").read_text())
    pose_p = d / "sim_eef_pose.json"
    pose = json.loads(pose_p.read_text()) if pose_p.exists() else None
    phases = phase_of(npzs)
    segs = swap_segments(k)
    infers = []
    for i, (z, s) in enumerate(zip(npzs, streams)):
        fr, mo = s["frame"], s["motion"]
        tot_f = sum(x["mass"] for x in fr)
        vis = [x for x in fr if VISIBLE[0] <= x["t"] < VISIBLE[1]]
        rec = {
            "infer": i, "step_idx": s["step_idx"], "phase": phases[i],
            "grip_state": float(z["state"][7]),
            "tcp_z": float(pose["infers"][str(i)]["state_z"]) if pose else None,
            "mass": s["mass"], "n_frames": len(fr), "n_motion": len(mo),
            "vis_share": sum(x["mass"] for x in vis) / tot_f, "vis_uniform": len(vis) / len(fr),
            "swap_frame": [], "swap_motion": None,
        }
        for (a, b) in segs:
            hit = [x for x in fr if a <= x["t"] < b]
            rec["swap_frame"].append({"seg": [a, b], "share": sum(x["mass"] for x in hit) / tot_f,
                                      "uniform": len(hit) / len(fr), "n": len(hit)})
        if s["has_motion"] and mo:
            tot_m = sum(x["mass"] for x in mo)
            per = []
            for (a, b) in segs:
                w = [overlap(x["t"], x["t"] + MOTION_WINDOW, a, b) / MOTION_WINDOW for x in mo]
                per.append({"seg": [a, b], "share": sum(x["mass"] * wi for x, wi in zip(mo, w)) / tot_m,
                            "uniform": sum(w) / len(mo)})
            lo, hi = swap_interval(k)
            w = [overlap(x["t"], x["t"] + MOTION_WINDOW, lo, hi) / MOTION_WINDOW for x in mo]
            rec["swap_motion"] = {"union": {"share": sum(x["mass"] * wi for x, wi in zip(mo, w)) / tot_m,
                                            "uniform": sum(w) / len(mo)}, "per_seg": per}
        infers.append(rec)
    return {"key": key, "label": label, "dir": str(d), "k": k, "outcome": meta["outcome"],
            "episode_id": meta["episode_id"], "has_pose": pose is not None, "segs": segs,
            "infers": infers, "streams": streams}


def phase_summary(ep: dict) -> dict:
    out = {}
    for ph in PHASES:
        rs = [r for r in ep["infers"] if r["phase"] == ph and r["n_frames"] > 1]
        if not rs:
            continue
        e = {"infers": [r["infer"] for r in rs],
             "vis_share": float(np.mean([r["vis_share"] for r in rs])),
             "vis_uniform": float(np.mean([r["vis_uniform"] for r in rs]))}
        e["vis_ratio"] = e["vis_share"] / e["vis_uniform"]
        segs = []
        for j in range(ep["k"]):
            ss = [r["swap_frame"][j] for r in rs if r["swap_frame"][j]["n"] > 0]
            if ss:
                sh, un = float(np.mean([x["share"] for x in ss])), float(np.mean([x["uniform"] for x in ss]))
                segs.append({"seg": ss[0]["seg"], "share": sh, "uniform": un, "ratio": sh / un})
        e["swap_frame"] = segs
        ms = [r["swap_motion"]["union"] for r in rs if r["swap_motion"] and r["swap_motion"]["union"]["uniform"] > 0]
        if ms:
            sh, un = float(np.mean([x["share"] for x in ms])), float(np.mean([x["uniform"] for x in ms]))
            e["swap_motion"] = {"share": sh, "uniform": un, "ratio": sh / un}
        for m in ["frame", "motion", "image_base", "image_wrist", "text", "action"]:
            e[f"mass_{m}"] = float(np.mean([r["mass"][m] for r in rs]))
        out[ph] = e
    return out


def above_uniform_span(vals: list[tuple[int, float, float]], thr: float = 1.0) -> dict:
    """返回占比 / 均匀期望 > thr 的推理序号、首末次。"""
    hits = [i for i, s, u in vals if u > 0 and s / u > thr]
    return {"above": hits, "first": hits[0] if hits else None, "last": hits[-1] if hits else None}


# ---------------------------------------------------------------- 画图

def _bands(ax, k: int, y0: float, y1: float) -> None:
    ax.axvspan(*VISIBLE, ymin=y0, ymax=y1, color=VIS_C, alpha=0.9, lw=0)
    for j, (a, b) in enumerate(swap_segments(k)):
        ax.axvspan(a, b, ymin=y0, ymax=y1, color=[SWAP1_C, SWAP2_C][j], alpha=0.9, lw=0)


def _heat(ax, grid: np.ndarray, k: int, title: str, cmap, norm=None, vmin=None, vmax=None, key=False):
    n = grid.shape[0]
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white")
    im = ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=cm, norm=norm, vmin=vmin, vmax=vmax,
                   extent=(0, N_COLS, n - 0.5, -0.5), origin="upper")
    for x in [VISIBLE[1]] + [b for _, b in swap_segments(k)] + [swap_segments(k)[0][0]]:
        ax.axvline(x, color="k", lw=0.6, ls="--")
    # 色带画在热图上方 -1.4…-0.6 的位置
    ax.set_ylim(n - 0.5, -1.6)
    ax.fill_between(VISIBLE, -1.5, -0.6, color=VIS_C, lw=0)
    for j, (a, b) in enumerate(swap_segments(k)):
        ax.fill_between([a, b], -1.5, -0.6, color=[SWAP1_C, SWAP2_C][j], lw=0)
    ax.set_yticks(range(0, n, 2))
    ax.set_ylabel("推理序号")
    ax.set_title(title, fontsize=10, loc="left")
    if key:
        ax.annotate("#18", xy=(N_COLS, KEY_INFER), xytext=(N_COLS + 4, KEY_INFER), fontsize=8,
                    va="center", annotation_clip=False, color="k")
        ax.plot([N_COLS - 6, N_COLS], [KEY_INFER, KEY_INFER], color="k", lw=2, clip_on=False)
    return im


def _band_legend(fig, k_any: int = 2, y: float = 0.005):
    h = [Patch(color=VIS_C, label="裸露窗 VIS [0,32)"), Patch(color=SWAP1_C, label="SWAP 第 1 段 [64,114)"),
         Patch(color=SWAP2_C, label="SWAP 第 2 段 [114,164)（仅 k=2）")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, y))


def fig_heat_frame(eps: list[dict], out: Path, rank: bool) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    grids = [build_grid([frame_cells(s["frame"], s["step_idx"]) for s in ep["streams"]], rank) for ep in eps]
    if rank:
        kw = {"cmap": "rainbow", "vmin": 0, "vmax": 1}
    else:
        allv = np.concatenate([g[np.isfinite(g)] for g in grids])
        kw = {"cmap": "viridis", "norm": LogNorm(vmin=max(allv[allv > 0].min(), 1e-5), vmax=allv.max())}
    for ax, ep, g in zip(axes, eps, grids):
        im = _heat(ax, g, ep["k"], ep["label"], key=ep["key"] != "ctx_ep4", **kw)
    axes[-1].set_xlabel("全域帧号（每个采样帧填到下一采样帧之前）")
    fig.subplots_adjust(left=0.07, right=0.88, top=0.93, bottom=0.08, hspace=0.25)
    cax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label="行内排名（0 最低 → 1 最高）" if rank else "帧流每帧注意力质量（对数）")
    fig.suptitle("推理序号 × 全域帧号：帧流注意力" + ("（行内排名归一化，论文 Fig.9 法）" if rank else "（真实质量）"),
                 fontsize=12)
    _band_legend(fig)
    p = out / ("x2_heatmap_frame_rank.png" if rank else "x2_heatmap_frame_raw.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_heat_motion(ep: dict, out: Path) -> Path:
    rows = []
    for s in ep["streams"]:
        rows.append([(x["t"], x["t"] + MOTION_WINDOW, x["mass"]) for x in sorted(s["motion"], key=lambda x: x["t"])])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    g_raw, g_rank = build_grid(rows, False), build_grid(rows, True)
    v = g_raw[np.isfinite(g_raw)]
    im0 = _heat(axes[0], g_raw, ep["k"], ep["label"] + "：motion 流真实质量", "viridis",
                norm=LogNorm(vmin=max(v[v > 0].min(), 1e-5), vmax=v.max()), key=True)
    im1 = _heat(axes[1], g_rank, ep["k"], ep["label"] + "：motion 流行内排名", "rainbow", vmin=0, vmax=1, key=True)
    axes[1].set_xlabel("全域帧号（每个 motion 窗按起点画、宽 33；#0–#1 无 motion 窗）")
    fig.subplots_adjust(left=0.07, right=0.88, top=0.92, bottom=0.11, hspace=0.25)
    fig.colorbar(im0, cax=fig.add_axes([0.91, 0.55, 0.015, 0.35]), label="每窗质量（对数）")
    fig.colorbar(im1, cax=fig.add_axes([0.91, 0.13, 0.015, 0.35]), label="行内排名")
    fig.suptitle("推理序号 × 全域帧号：motion 流注意力", fontsize=12)
    _band_legend(fig)
    p = out / "x2_heatmap_motion.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def _phase_strip(ax, ep: dict) -> None:
    for r in ep["infers"]:
        ax.axvspan(r["infer"] - 0.5, r["infer"] + 0.5, color=PHASE_COLOR[r["phase"]], alpha=0.45, lw=0)


def fig_curves(eps: list[dict], out: Path) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.7]})
    x = np.arange(20)
    for c, ep in enumerate(eps):
        rs = ep["infers"]
        ax = axes[0, c]
        _phase_strip(ax, ep)
        ax.plot(x, [r["vis_share"] for r in rs], "-o", color=VIS_C, ms=4, label="帧流 VIS 占比")
        ax.plot(x, [r["vis_uniform"] for r in rs], "--", color=VIS_C, lw=1.2, label="VIS 均匀期望")
        ax.set_ylim(0, 1.05)
        ax.set_title(ep["label"], fontsize=10)
        if c == 0:
            ax.set_ylabel("VIS 占帧流比例")
        ax.legend(fontsize=8, loc="upper right")
        ax = axes[1, c]
        _phase_strip(ax, ep)
        for j in range(ep["k"]):
            col = [SWAP1_C, SWAP2_C][j]
            a, b = ep["segs"][j]
            sh = [r["swap_frame"][j]["share"] if r["swap_frame"][j]["n"] else np.nan for r in rs]
            un = [r["swap_frame"][j]["uniform"] if r["swap_frame"][j]["n"] else np.nan for r in rs]
            ax.plot(x, sh, "-o", color=col, ms=3, label=f"帧流 SWAP[{a},{b}) 占比")
            ax.plot(x, un, "--", color=col, lw=1.0, label="均匀期望")
        if any(r["swap_motion"] for r in rs):
            sh = [r["swap_motion"]["union"]["share"] if r["swap_motion"] else np.nan for r in rs]
            un = [r["swap_motion"]["union"]["uniform"] if r["swap_motion"] else np.nan for r in rs]
            ax.plot(x, sh, "-s", color="#6a3d9a", ms=4, label="motion 流 SWAP 并集占比（重叠/33 加权）")
            ax.plot(x, un, ":", color="#6a3d9a", lw=1.5, label="motion 均匀期望")
        ax.set_ylim(0, 1.3)
        if c == 0:
            ax.set_ylabel("SWAP 占比")
        ax.legend(fontsize=7, loc="upper center", ncol=2)
        ax = axes[2, c]
        _phase_strip(ax, ep)
        ax.plot(x, [r["grip_state"] * 100 for r in rs], "-o", color="k", ms=3, label="手指开度 state[7]（cm）")
        ax.set_ylim(-0.3, 5.5)
        if c == 0:
            ax.set_ylabel("开度 cm")
        if ep["has_pose"]:
            ax2 = ax.twinx()
            ax2.plot(x, [r["tcp_z"] for r in rs], "-^", color="#1f77b4", ms=3, label="TCP z（m）")
            ax2.set_ylim(0, 0.6)
            ax2.tick_params(labelsize=8)
            if c == 2:
                ax2.set_ylabel("TCP z（m）")
            ax2.legend(fontsize=7, loc="upper right")
        else:
            ax.text(0.98, 0.9, "本局无 sim_eef_pose", transform=ax.transAxes, ha="right", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlabel("推理序号（step = 16×序号）")
        ax.set_xticks(range(0, 20, 2))
        if ep["key"] != "ctx_ep4":
            for rr in range(3):
                axes[rr, c].axvline(KEY_INFER, color="k", lw=0.8, ls="-.")
    h = [Patch(color=PHASE_COLOR[p], alpha=0.45, label=p) for p in PHASES]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0),
               title="阶段底色（由夹爪开度与指令推出；点划线 = ep8 关键推理 #18）", title_fontsize=9)
    fig.suptitle("每次推理的回看占比与均匀期望（VIS 上排，SWAP 中排）", fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.11, hspace=0.12, wspace=0.18)
    p = out / "x2_share_curves.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def fig_stack(eps: list[dict], out: Path) -> Path:
    keys = ["frame", "motion", "image_base", "image_wrist", "text", "action"]
    names = ["帧流", "motion 流", "当前 base 图", "当前 wrist 图", "文本", "动作"]
    cols = ["#2ca02c", "#6a3d9a", "#1f77b4", "#17becf", "#ff7f0e", "#7f7f7f"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    x = np.arange(20)
    for ax, ep in zip(axes, eps):
        bottom = np.zeros(20)
        for k_, nm, cl in zip(keys, names, cols):
            v = np.array([r["mass"][k_] for r in ep["infers"]])
            ax.bar(x, v, bottom=bottom, color=cl, width=0.85, label=nm)
            bottom += v
        ax.set_title(ep["label"], fontsize=10, pad=16)
        ax.set_xlabel("推理序号")
        ax.set_xticks(range(0, 20, 2))
        ax.set_ylim(0, 1.0)
        for r in ep["infers"]:
            ax.add_patch(plt.Rectangle((r["infer"] - 0.5, 1.0), 1, 0.04, color=PHASE_COLOR[r["phase"]],
                                       clip_on=False, lw=0))
    axes[0].set_ylabel("注意力质量（动作 query，全层／头／去噪步平均）")
    h1, l1 = axes[2].get_legend_handles_labels()
    h2 = [Patch(color=PHASE_COLOR[p], label=p) for p in PHASES]
    fig.legend(h1 + h2, l1 + PHASES, loc="lower center", ncol=11, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("五流质量随推理堆叠（顶部色条 = 阶段）", fontsize=12, y=0.99)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.2, wspace=0.06)
    p = out / "x2_stream_stack.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/evaluation-doc/attn-ctx512-vs-motion-1103/analysis/x2")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = [episode_record(*e) for e in EPISODES]
    n_inf = sum(len(e["infers"]) for e in eps)
    figs = [fig_heat_frame(eps, out, False), fig_heat_frame(eps, out, True),
            fig_heat_motion(eps[2], out), fig_curves(eps, out), fig_stack(eps, out)]
    summ = {"note": "动作 query、全部去噪步／层／头平均；VIS=[0,32)，SWAP=[64,64+50k)；motion SWAP 按重叠帧数/33 加权",
            "episodes": []}
    for ep in eps:
        rs = ep["infers"]
        spans = {"vis_frame": above_uniform_span([(r["infer"], r["vis_share"], r["vis_uniform"]) for r in rs if r["n_frames"] > 1])}
        for j in range(ep["k"]):
            spans[f"swap{j + 1}_frame"] = above_uniform_span(
                [(r["infer"], r["swap_frame"][j]["share"], r["swap_frame"][j]["uniform"]) for r in rs])
        if any(r["swap_motion"] for r in rs):
            spans["swap_motion"] = above_uniform_span(
                [(r["infer"], r["swap_motion"]["union"]["share"], r["swap_motion"]["union"]["uniform"])
                 for r in rs if r["swap_motion"]])
        summ["episodes"].append({k_: ep[k_] for k_ in ["key", "label", "dir", "k", "outcome", "episode_id", "segs"]}
                                | {"infers": rs, "phase_summary": phase_summary(ep), "above_uniform": spans})
        for r in rs:
            print(ep["key"], r["infer"], r["phase"], f"vis={r['vis_share']:.3f}/{r['vis_uniform']:.3f}",
                  " ".join(f"sw{j + 1}={s['share']:.3f}/{s['uniform']:.3f}" for j, s in enumerate(r["swap_frame"])),
                  (f"mot={r['swap_motion']['union']['share']:.3f}/{r['swap_motion']['union']['uniform']:.3f}"
                   if r["swap_motion"] else ""),
                  f"F={r['mass']['frame']:.3f} M={r['mass']['motion']:.3f} B={r['mass']['image_base']:.3f} "
                  f"W={r['mass']['image_wrist']:.3f} T={r['mass']['text']:.3f} A={r['mass']['action']:.3f}")
    (out / "x2_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    ok = len(eps) == 3 and n_inf == 60 and all(p.exists() for p in figs)
    print(f"X2_TIMELINE={'PASS' if ok else 'FAIL'} episodes={len(eps)} infers={n_inf} figs={len(figs)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
