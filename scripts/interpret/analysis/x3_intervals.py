"""X3：区间核实与区间内细分（attn-ctx512-vs-motion-1103 第一阶段后续分析）。

只读输入：stage1 的原始 attention npz（每局 20 次推理）与客户端视频；不跑仿真、不用 GPU。
做四件事：
1. 区间核实：从视频 base 视角量绿色方块像素数（裸露窗）与容器白色掩码逐帧变化量（交换），
   得到本局真实的 VIS／SWAP 边界，与环境源码推得的 [0,32)／[64,114)／[114,164) 对照；截帧拼图作证据。
   视频帧号 i ＝ 全域帧号（step_idx 坐标）：视频 = reset 观测 1 帧 + 每执行 1 个动作 1 帧
   （examples/robomme/eval.py::eval_each_episode 的 record_frame），帧上「Frame: i」文字可核对。
2. 裸露窗内逐帧：每次推理被采样进记忆的 32 帧里落在 VIS 的帧及其注意力。
3. 交换区间细分：motion 窗按与各段重叠帧数／33 加权（修正），对照旧口径「沾边整窗计入」。
4. 去噪步与动作 query：#15～#19 的 VIS／SWAP 占比按 10 个去噪步、20 个动作 query 拆开。

分流复用 scripts/interpret/aggregate_attn.py::split_streams（只 import 不改）。
输出到 --out（图 PNG、x3_summary.json），末行 X3_INTERVALS=PASS|FAIL。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "interpret"))
from aggregate_attn import MOTION_WINDOW, split_streams  # noqa: E402
from plot_attn import CJK  # noqa: E402  （导入即设置中文字体）

import matplotlib.pyplot as plt  # noqa: E402

STAGE1 = Path("/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/interpret-1103/stage1")
RUNS = {
    # 名称: (attn 目录, 视频, 图注)
    "ctx_ep8": ("ctx/attn/3625db1fb81a48af8aa5a136b3a7aa68",
                "ctx/run/attempts/000-f94d3288a905-a517c3db/client/videos/ButtonUnmaskSwap_ep8_success.mp4",
                "基线 ep8（成功）"),
    "mot_ep8": ("mot/attn/af10850c1f7d47549fcd8934178a7497",
                "mot/run/attempts/000-f94d3288a905-bec0ab84/client/videos/ButtonUnmaskSwap_ep8_success.mp4",
                "motion ep8（成功）"),
    "ctx_ep4": ("ctx/attn/1d677dadd3f34872a5779dbe8ab815e4",
                "ctx/run/attempts/000-1f83580c242d-3d5643d9/client/videos/ButtonUnmaskSwap_ep4_fail.mp4",
                "基线 ep4（失败）"),
}
# 环境源码推得的名义区间（ButtonUnmaskSwap.step：lift_and_drop 0..64 中点 32 落回；swap_schedule 64+50j）。
NOMINAL = {"vis": (0, 32), "swap": [(64, 114), (114, 164)]}
KEY = [15, 16, 17, 18, 19]
# 视频版式：上 200 行文字，下半左 base（256 宽）右 wrist。
BASE_Y0, BASE_X1 = 200, 256
# 容器所在区域（base 视角内，避开上方按钮与机械臂主体）。
BIN_ROI = (85, 170, 40, 230)
MOVE_THR = 60  # 相邻帧容器掩码变化像素数 > 此值视为容器在动（交换段中位数约 250，静止 <30）


# ---------------------------------------------------------------- 1. 视频核实

def read_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return np.asarray(frames)


def video_signals(fr: np.ndarray) -> dict:
    base = fr[:, BASE_Y0:, :BASE_X1]
    hsv = np.asarray([cv2.cvtColor(x, cv2.COLOR_BGR2HSV) for x in base])
    green = ((hsv[..., 0] > 45) & (hsv[..., 0] < 80) & (hsv[..., 1] > 120) & (hsv[..., 2] > 80)).sum((1, 2))
    y0, y1, x0, x1 = BIN_ROI
    h = hsv[:, y0:y1, x0:x1]
    m = (h[..., 1] < 40) & (h[..., 2] > 150)
    change = np.concatenate([[0], (m[1:] ^ m[:-1]).sum((1, 2))])  # change[i] = 帧 i-1→i
    return {"green": green.astype(int), "bin_change": change.astype(int)}


def detect_intervals(sig: dict) -> dict:
    """VIS = 绿色方块可见的连续帧；SWAP = 帧 33 之后容器连续移动段，按变化量谷底拆成各次交换。"""
    g = sig["green"]
    vis_hi = int(np.argmax(g == 0))  # 第一帧不可见
    vis = (0, vis_hi) if g[0] > 0 else None
    ch = sig["bin_change"]
    moving = np.where(ch > MOVE_THR)[0]
    moving = moving[moving > vis_hi + 8]  # 去掉容器落回瞬间的跳变
    # 连续段（间隙 ≤3 帧合并）
    runs: list[list[int]] = []
    for i in moving:
        if runs and i - runs[-1][1] <= 3:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    main = max(runs, key=lambda r: r[1] - r[0])
    # 帧 lo 是第一帧离开原位；帧 hi 是最后一帧仍有变化，即到达终点位姿的那一帧。
    # 区间记为「在途帧」[lo, hi)：与源码 swap_schedule (start,end) 同一写法。
    lo, hi = main[0], main[1]
    # 交换段内 smoothstep 速度在两端为 0，相邻两次交换之间出现变化量谷底；谷底宽 2～3 帧。
    seg = ch[lo:hi + 1]
    med = float(np.median(seg))
    valleys = [lo + j for j in range(10, len(seg) - 10)
               if seg[j] < 0.35 * med and seg[j] == seg[j - 5:j + 6].min()]
    k = len(valleys) + 1
    eq = [lo + round((hi - lo) * j / k) for j in range(1, k)]  # 等长切分点
    # 谷底与等长切分点相差 ≤3 帧时取等长切分（谷底只用来数交换次数、作证据）
    cuts = eq if all(abs(v - e) <= 3 for v, e in zip(valleys, eq)) else valleys
    bounds = [lo] + cuts + [hi]
    swaps = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]
    return {"vis": vis, "swap": swaps, "move_run": (int(lo), int(hi)), "valleys": [int(v) for v in valleys],
            "valley_vals": [int(ch[v]) for v in valleys], "change_at_bounds": {int(b): int(ch[b]) for b in bounds}}


def evidence_mosaic(videos: dict, out: Path) -> None:
    picks = {
        "ctx_ep8": [0, 31, 32, 63, 64, 89, 113, 114, 115, 139, 163, 164, 165],
        "mot_ep8": [0, 31, 32, 63, 64, 89, 113, 114, 115, 139, 163, 164, 165],
        "ctx_ep4": [0, 31, 32, 63, 64, 89, 113, 114, 115, 139, 163, 164, 165],
    }
    rows = []
    for name, idx in picks.items():
        fr = videos[name]
        tiles = []
        for i in idx:
            tile = cv2.resize(fr[i][BASE_Y0:, :BASE_X1], (192, 198), interpolation=cv2.INTER_AREA).copy()
            cv2.putText(tile, str(i), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    img = np.vstack(rows)[:, :, ::-1]
    fig, ax = plt.subplots(figsize=(img.shape[1] / 120, img.shape[0] / 120 + 0.9), dpi=120)
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([(k + 0.5) * 198 for k in range(3)])
    ax.set_yticklabels([RUNS[n][2] for n in picks], fontsize=9)
    ax.set_title("区间核实截帧（base 视角，角标为视频帧号＝全域帧号）：31→32 容器落回盖住；"
                 "63→64 第 1 次交换开始；114 两次交换分界；164 到达终点、165 起静止", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_video_curves(sigs: dict, det: dict, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for ax, (name, s) in zip(axes, sigs.items()):
        n = len(s["green"])
        ax.plot(np.arange(n), s["bin_change"], color="#3b6ea8", lw=1, label="容器掩码逐帧变化像素数")
        ax2 = ax.twinx()
        ax2.plot(np.arange(n), s["green"], color="#2f9e44", lw=1.2, label="绿色方块像素数")
        ax2.set_ylabel("绿色像素", color="#2f9e44", fontsize=8)
        d = det[name]
        if d["vis"]:
            ax.axvspan(*d["vis"], color="#b7e4c7", alpha=0.5)
        cols = ["#fbd38d", "#f6ad55", "#ed8936"]
        for j, (a, b) in enumerate(d["swap"]):
            ax.axvspan(a, b, color=cols[j % 3], alpha=0.35)
            ax.text((a + b) / 2, ax.get_ylim()[1] * 0.92, f"交换{j + 1} [{a},{b})", ha="center", fontsize=8)
        ax.text(1, ax.get_ylim()[1] * 0.92, f"裸露 [{d['vis'][0]},{d['vis'][1]})", fontsize=8)
        ax.set_ylabel("变化像素", fontsize=8)
        ax.set_title(RUNS[name][2], fontsize=9, loc="left")
        ax.set_xlim(0, 320)
    axes[-1].set_xlabel("全域帧号（＝视频帧号）")
    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1 + [plt.Line2D([], [], color="#2f9e44")], l1 + ["绿色方块像素数（右轴）"],
               loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("视频量化信号：方块可见帧与容器移动段", fontsize=11, x=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- 2–4. attention

def load_infer(d: Path, i: int) -> dict:
    z = np.load(d / f"infer_{i:04d}.npz")
    out = {k: z[k] for k in z.files if k not in ("probs_action", "probs_text", "top_frames")}
    pa = z["probs_action"].astype(np.float32)  # (10,18,8,20,K)
    out["pa_sq"] = pa.mean(axis=(1, 2))  # (10,20,K)：层、头已平均（等价于 split_streams 的全层全头平均）
    return out


def streams_of(rec: dict, steps=None, query=None) -> dict:
    """在选定去噪步／动作 query 上平均后调用 split_streams（构造一个只含所需键的 npz 字典）。"""
    x = rec["pa_sq"]
    if steps is not None:
        x = x[steps:steps + 1]
    if query is not None:
        x = x[:, query:query + 1]
    fake = {k: v for k, v in rec.items() if k != "pa_sq"}
    fake["probs_action"] = x[:, None, None]  # (s,1,1,q,K)
    return split_streams(fake, query="action")


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def shares(st: dict, vis, swaps) -> dict:
    fr = st["frame"]
    tot_f = sum(x["mass"] for x in fr)
    res = {"frame_total": tot_f, "motion_total": sum(x["mass"] for x in st["motion"]),
           "n_mem_frames": len(fr)}
    def fshare(lo, hi):
        hit = [x for x in fr if lo <= x["t"] < hi]
        return (sum(x["mass"] for x in hit) / tot_f if tot_f > 0 else float("nan"),
                len(hit) / len(fr) if fr else float("nan"), [x["t"] for x in hit])
    res["vis_frame_share"], res["vis_frame_uniform"], res["vis_frames"] = fshare(*vis)
    for j, (a, b) in enumerate(swaps):
        s, u, ts = fshare(a, b)
        res[f"swap{j + 1}_frame_share"], res[f"swap{j + 1}_frame_uniform"], res[f"swap{j + 1}_frames"] = s, u, ts
    mo = st["motion"]
    tot_m = res["motion_total"]
    if mo and tot_m > 0:
        n = len(mo)
        segs = {"vis": [vis], "swap": swaps, **{f"swap{j + 1}": [sw] for j, sw in enumerate(swaps)}}
        for key, ss in segs.items():
            w_new = [sum(overlap(x["t"], x["t"] + MOTION_WINDOW, a, b) for a, b in ss) / MOTION_WINDOW for x in mo]
            w_old = [1.0 if any(overlap(x["t"], x["t"] + MOTION_WINDOW, a, b) > 0 for a, b in ss) else 0.0 for x in mo]
            res[f"{key}_motion_share_new"] = sum(w * x["mass"] for w, x in zip(w_new, mo)) / tot_m
            res[f"{key}_motion_uniform_new"] = sum(w_new) / n
            res[f"{key}_motion_share_old"] = sum(w * x["mass"] for w, x in zip(w_old, mo)) / tot_m
            res[f"{key}_motion_uniform_old"] = sum(w_old) / n
        res["motion_windows"] = [{"t": x["t"], "mass": x["mass"]} for x in mo]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    figs: list[str] = []

    # ---- 1. 视频核实
    videos, sigs, det = {}, {}, {}
    for name, (_, vid, _) in RUNS.items():
        videos[name] = read_video(STAGE1 / vid)
        sigs[name] = video_signals(videos[name])
        det[name] = detect_intervals(sigs[name])
        print(name, "frames", len(videos[name]), det[name], flush=True)
    evidence_mosaic(videos, out / "fig_x3_1_interval_frames.png")
    figs.append("fig_x3_1_interval_frames.png")
    plot_video_curves(sigs, det, out / "fig_x3_2_video_signals.png")
    figs.append("fig_x3_2_video_signals.png")
    assert det["ctx_ep8"]["vis"] == det["mot_ep8"]["vis"] and det["ctx_ep8"]["swap"] == det["mot_ep8"]["swap"], \
        "ep8 两份视频的区间不一致"
    vis = tuple(det["ctx_ep8"]["vis"])
    swaps = [tuple(s) for s in det["ctx_ep8"]["swap"]]
    vis4 = tuple(det["ctx_ep4"]["vis"])
    swaps4 = [tuple(s) for s in det["ctx_ep4"]["swap"]]
    del videos

    # ---- 2–4. attention
    summary = {"intervals": {"nominal": NOMINAL, "verified": det,
                             "frame_index_mapping": "视频帧号 i = 全域帧号 i（reset 观测 1 帧 + 每动作 1 帧）"},
               "runs": {}}
    per_frame = {}   # name -> list over infers of (t list, mass list)
    key_steps = {}   # name -> {i: (10,) vis/swap}
    key_query = {}
    for name, (adir, _, _) in RUNS.items():
        d = STAGE1 / adir
        iv, sw = (vis, swaps) if name != "ctx_ep4" else (vis4, swaps4)
        rows = []
        per_frame[name] = []
        key_steps[name], key_query[name] = {}, {}
        for i in range(20):
            rec = load_infer(d, i)
            st = streams_of(rec)
            sh = shares(st, iv, sw)
            sh.update({"infer": i, "step_idx": int(rec["step_idx"]),
                       "vis_frame_mass": [{"t": x["t"], "mass": x["mass"]} for x in st["frame"]
                                          if iv[0] <= x["t"] < iv[1]]})
            rows.append(sh)
            per_frame[name].append(([x["t"] for x in st["frame"]], [x["mass"] for x in st["frame"]]))
            if i in KEY:
                ks = [shares(streams_of(rec, steps=s), iv, sw) for s in range(10)]
                kq = [shares(streams_of(rec, query=q), iv, sw) for q in range(20)]
                key_steps[name][i] = ks
                key_query[name][i] = kq
                rows[-1]["by_denoise_step"] = [{k: v for k, v in x.items() if k.endswith(("share", "share_new", "total"))}
                                              for x in ks]
                rows[-1]["by_action_query"] = [{k: v for k, v in x.items() if k.endswith(("share", "share_new", "total"))}
                                              for x in kq]
        summary["runs"][name] = {"vis": iv, "swap": sw, "infers": rows}
        print(name, "done", flush=True)

    # 名义边界下的对照（只对 ep8，名义与核实若一致则相同）
    summary["nominal_equals_verified_ep8"] = bool(vis == NOMINAL["vis"] and swaps == NOMINAL["swap"])

    # ---- 图 3：逐推理 × 全域帧号 的帧流注意力（占帧流比例）
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, name in zip(axes, RUNS):
        M = np.full((20, 320), np.nan)
        for i, (ts, ms) in enumerate(per_frame[name]):
            tot = sum(ms)
            for t_, m_ in zip(ts, ms):
                M[i, t_] = m_ / tot if tot > 0 else np.nan
        # 每个采样帧画成 3 帧宽方便看清（只影响显示；数值不插值）
        Mw = M.copy()
        for c in (1, 2):
            sh = np.roll(M, c, axis=1)
            Mw = np.where(np.isnan(Mw), sh, Mw)
        im = ax.imshow(Mw, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=0.25,
                       extent=(-0.5, 319.5, 19.5, -0.5))
        iv, sw = summary["runs"][name]["vis"], summary["runs"][name]["swap"]
        ax.axvline(iv[1], color="#40c057", lw=1.2, ls="--")
        for a, b in sw:
            ax.axvline(a, color="#fd7e14", lw=1, ls=":")
            ax.axvline(b, color="#fd7e14", lw=1, ls=":")
        for k in KEY:
            ax.axhline(k, color="white", lw=0.3, alpha=0.4)
        ax.set_yticks(range(0, 20, 2))
        ax.set_ylabel("推理序号")
        ax.set_title(f"{RUNS[name][2]}：帧流内归一化注意力（绿虚线＝裸露窗止 {iv[1]}，橙点线＝交换边界 "
                     + "、".join(f"[{a},{b})" for a, b in sw) + "）", fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="占帧流比例")
    axes[-1].set_xlabel("记忆帧的全域帧号（每个采样帧显示 3 帧宽，最右亮点为当前帧）")
    fig.tight_layout()
    fig.savefig(out / "fig_x3_3_frame_attn_heatmap.png", dpi=120)
    plt.close(fig)
    figs.append("fig_x3_3_frame_attn_heatmap.png")

    # ---- 图 4：#15～#19 裸露窗内逐帧注意力（占帧流比例）
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharey=True)
    colors = {"ctx_ep8": "#4c6ef5", "mot_ep8": "#e8590c", "ctx_ep4": "#868e96"}
    for ax, k in zip(axes, KEY):
        for off, name in zip((-0.9, 0, 0.9), RUNS):
            r = summary["runs"][name]["infers"][k]
            tot = r["frame_total"]
            ts = [x["t"] for x in r["vis_frame_mass"]]
            ms = [x["mass"] / tot * 100 for x in r["vis_frame_mass"]]
            ax.bar(np.asarray(ts) + off, ms, width=0.9, color=colors[name], label=RUNS[name][2])
        r8 = summary["runs"]["ctx_ep8"]["infers"][k]
        ax.set_title(f"#{k}（step {r8['step_idx']}）", fontsize=9)
        ax.set_xlim(-2, 33)
        ax.set_xlabel("裸露窗内帧号")
    axes[0].set_ylabel("该帧占帧流比例（%）")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("裸露窗 [0,32) 内被采样进记忆的帧及其注意力", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_x3_4_vis_frames_key.png", dpi=120)
    plt.close(fig)
    figs.append("fig_x3_4_vis_frames_key.png")

    # ---- 图 5：交换区间细分（motion 流修正前后 + 帧流）
    mo_rows = summary["runs"]["mot_ep8"]["infers"]
    xs = [r["infer"] for r in mo_rows if "swap_motion_share_new" in r]
    get = lambda key: [r[key] * 100 for r in mo_rows if "swap_motion_share_new" in r]  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    ax.plot(xs, get("swap_motion_share_old"), "o--", color="#adb5bd", label="旧口径：沾边整窗计入（SWAP 并集）")
    ax.plot(xs, get("swap_motion_share_new"), "o-", color="#e8590c", label="修正：按重叠帧数/33 加权（SWAP 并集）")
    ax.plot(xs, get("swap_motion_uniform_new"), ":", color="#e8590c", label="修正口径均匀期望")
    ax.plot(xs, get("swap1_motion_share_new"), "s-", color="#f59f00", ms=4, label="修正：第 1 次交换")
    ax.plot(xs, get("swap2_motion_share_new"), "^-", color="#c92a2a", ms=4, label="修正：第 2 次交换")
    ax.plot(xs, get("vis_motion_share_new"), "d-", color="#2f9e44", ms=4, label="修正：裸露窗")
    for k in KEY:
        ax.axvspan(k - 0.4, k + 0.4, color="#fff3bf", alpha=0.5, zorder=0)
    ax.set_xlabel("推理序号（黄底 #15～#19）")
    ax.set_ylabel("占 motion 流比例（%）")
    ax.set_title("motion ep8：motion 流落在交换区间的占比", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    ax.set_xticks(range(0, 20))
    ax = axes[1]
    for name, mk in zip(RUNS, ("o", "s", "^")):
        rows = summary["runs"][name]["infers"]
        nsw = len(summary["runs"][name]["swap"])
        for j in range(nsw):
            ax.plot(range(20), [r[f"swap{j + 1}_frame_share"] * 100 for r in rows], marker=mk, ms=3,
                    ls="-" if j == 0 else "--", color=colors[name], label=f"{RUNS[name][2]} 交换{j + 1}")
    ax.set_xlabel("推理序号")
    ax.set_ylabel("占帧流比例（%）")
    ax.set_title("帧流落在各次交换区间的占比", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    ax.set_xticks(range(0, 20))
    fig.tight_layout()
    fig.savefig(out / "fig_x3_5_swap_split.png", dpi=120)
    plt.close(fig)
    figs.append("fig_x3_5_swap_split.png")

    # ---- 图 6：去噪步 × 推理、动作 query × 推理 热图
    panels = [("ctx_ep8", "vis_frame_share", "基线 ep8 帧流裸露窗占比"),
              ("mot_ep8", "vis_frame_share", "motion ep8 帧流裸露窗占比"),
              ("ctx_ep4", "vis_frame_share", "基线 ep4 帧流裸露窗占比"),
              ("mot_ep8", "swap_motion_share_new", "motion ep8 motion 流交换占比（修正）")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.2))
    for c, (name, key, title) in enumerate(panels):
        for r_, (src, nx, xl) in enumerate(((key_steps, 10, "去噪步（0＝第一步，纯噪声起点）"),
                                            (key_query, 20, "动作 query（0＝最近的动作）"))):
            M = np.array([[x[key] * 100 for x in src[name][k]] for k in KEY])
            ax = axes[r_, c]
            im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap="viridis")
            for i_ in range(M.shape[0]):
                for j_ in range(M.shape[1]):
                    ax.text(j_, i_, f"{M[i_, j_]:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if M[i_, j_] < np.nanmean(M) else "black")
            ax.set_yticks(range(5))
            ax.set_yticklabels([f"#{k}" for k in KEY])
            ax.set_xticks(range(0, nx, 1 if nx == 10 else 2))
            ax.set_xlabel(xl, fontsize=8)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="%")
    fig.suptitle("#15～#19 按去噪步（上）与动作 query（下）拆开的占比（格内数字为 %）", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_x3_6_denoise_query.png", dpi=120)
    plt.close(fig)
    figs.append("fig_x3_6_denoise_query.png")

    summary["figs"] = figs
    (out / "x3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    ok = len(swaps) == 2 and vis[0] == 0
    print(f"X3_INTERVALS={'PASS' if ok else 'FAIL'} vis=[{vis[0]},{vis[1]}) swap1=[{swaps[0][0]},{swaps[0][1]}) "
          f"swap2=[{swaps[1][0]},{swaps[1][1]}) figs={len(figs)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
