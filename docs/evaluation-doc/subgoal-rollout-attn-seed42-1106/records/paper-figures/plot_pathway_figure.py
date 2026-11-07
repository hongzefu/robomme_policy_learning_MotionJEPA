"""论文主图（三面板）：(a) 序列布局、注意力掩码与分层通路；(b) 全局注意力矩阵；(c) 子目标对位矩阵。

只读仓库已归档或已落盘的数据，不跑模型。(a)(b) 的数字全部由 PatternLock、RouteStick 的 motion 模型完整注意力
（`pathway_flow.py` 的逐局 npz）算出；(c) 读 summary.json / success-only.json。
用法：plot_pathway_figure.py [仓库根]（默认 /data/hongzefu/robomme_policy_learning_MotionJEPA）。输出到脚本所在目录。

数据来源：
- 逐局通路：v1-store/evaluation/subgoal-rollout-attn-seed42-1106/mot/pathway/<nonce>.npz
  （direct 形状 (18 层, 8 头, 8 读者组, 8 来源组)，组序 pathway_flow.py GROUPS；route_<α> 形状 (18, 7 中转组, 8 来源组)）
- 任务归属：v1-store/evaluation/subgoal-rollout-attn-seed42-1106/mot/*/run/accepted/*.terminal.json
- 序列长度：任一 full-attn/<nonce>/infer_0000.npz 的 MEM、txt_len、P、probs_action 形状
- (c)：docs/evaluation-doc/subgoal-rollout-attn-seed42-1106/records/summary.json、success-only.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/hongzefu/robomme_policy_learning_MotionJEPA")
OUT = Path(__file__).resolve().parent
D1106 = REPO / "docs/evaluation-doc/subgoal-rollout-attn-seed42-1106/records"
STORE = REPO / "v1-store/evaluation/subgoal-rollout-attn-seed42-1106/mot"

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
DIVERGE = LinearSegmentedColormap.from_list("diverge", ["#2a78d6", "#f0efec", "#e34948"])
DIVERGE.set_bad("#ffffff")
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f7fb", "#2a78d6", "#0b2f63"])
INK, INK2, INK3 = "#1f1f1f", "#555555", "#8a8a8a"
GRID = "#d9d8d2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": INK3, "axes.linewidth": 0.5, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ----------------------------------------------------------------------------- 读通路 npz，按任务分组
GROUPS = ("frame_demo", "frame_exec", "motion_demo", "motion_exec", "image_base", "image_wrist", "text", "action")
FRAME, MOTION, IMAGE, TEXT, ACTION = [0, 1], [2, 3], [4, 5], 6, 7
task_of = {}
for path in glob.glob(str(STORE / "*" / "run" / "accepted" / "*.terminal.json")):
    record = json.load(open(path, encoding="utf-8"))
    task_of[record["episode_nonce"]] = record["episode_key"].split(":")[0]
episodes = {}
for path in sorted((STORE / "pathway").glob("*.npz")):
    with np.load(path) as z:
        episodes.setdefault(task_of[path.stem], []).append(
            {"direct": z["direct"].mean(1), "direct_max": z["direct"].max(1), "counts": z["counts"],
             **{f"route_{a}": z[f"route_{a}"] for a in ("0.2", "0.5", "0.8")}, "carry": z["rollout_0.5"]})
TASKS = ("PatternLock", "RouteStick")
assert set(episodes) == set(TASKS), set(episodes)
inferences = sum(json.load(open(STORE / "pathway" / "index.json"))["episodes"][i]["inferences"]
                 for i in range(len(json.load(open(STORE / "pathway" / "index.json"))["episodes"])))

def fold_frames(mat_by_layer, counts, source):
    """帧读者按 token 数加权成一行：(层, 来源) 。"""
    w = counts[FRAME] / counts[FRAME].sum()
    return np.array([(w[:, None] * m[FRAME][:, source]).sum() for m in mat_by_layer])

stat = {}
for task in TASKS:
    items = episodes[task]
    direct = np.mean([i["direct"] for i in items], 0) * 100            # (18, 8, 8)
    counts = np.mean([i["counts"] for i in items], 0)
    carry = np.mean([i["carry"] for i in items], 0) * 100              # (19, 8, 8)
    s = {"episodes": len(items), "motion_tokens": counts[MOTION].sum(), "direct": direct,
         "frame_reads_motion": fold_frames(direct, counts, MOTION),
         "frame_carry_motion": fold_frames(carry, counts, MOTION),
         "text_reads_frame": direct[:, TEXT][:, FRAME].sum(1), "text_reads_motion": direct[:, TEXT][:, MOTION].sum(1),
         "action_reads_frame": direct[:, ACTION][:, FRAME].sum(1), "action_reads_text": direct[:, ACTION, TEXT],
         "action_reads_image": direct[:, ACTION][:, IMAGE].sum(1), "action_reads_motion": direct[:, ACTION][:, MOTION].sum(1),
         "image_reads_text": direct[:, IMAGE][:, :, TEXT]}
    for a in ("0.2", "0.5", "0.8"):
        via = np.mean([i[f"route_{a}"].sum(0) for i in items], 0)[:, MOTION].sum(1) * 100
        s[f"via_{a}"] = {"total": via.sum(), "frame": via[FRAME].sum() / via.sum(), "text": via[TEXT] / via.sum(),
                         "image": via[IMAGE].sum() / via.sum(), "motion": via[MOTION].sum() / via.sum()}
    stat[task] = s
# 掩码：某格在全部层、头、局上都恰为 0 才算被掩
mask_zero = np.max([i["direct_max"].max(0) for t in TASKS for i in episodes[t]], 0) == 0

def pair(key, layer=None, nd=0):
    vals = [stat[t][key] if layer is None else stat[t][key][layer] for t in TASKS]
    return "|".join(f"{v:.{nd}f}" for v in vals) + "%"

def rng(values, nd=0):
    return f"{min(values):.{nd}f}–{max(values):.{nd}f}%"

# 序列长度
with np.load(next((STORE / "full-attn").glob("*/infer_0000.npz"))) as z:
    MEM, TXT, P, HORIZON = int(z["MEM"]), int(z["txt_len"]), int(z["P"]), int(z["probs_action"].shape[3])
    N_IMG = (P - MEM - int(z["probs_text"].shape[2])) // 2
    FRAME_TOKENS = int((np.asarray(z["mem_order"]) < 512).sum())
assert P == MEM + 2 * N_IMG + 64 and TXT <= 64

# ----------------------------------------------------------------------------- (c) 数据
summary = json.load(open(D1106 / "summary.json", encoding="utf-8"))
succ = json.load(open(D1106 / "success-only.json", encoding="utf-8"))
WEIGHT = {"frame": "single", "motion": "overlap"}
def matrix(task, model, reader, stream):
    c = summary["A2"][task][model][reader][stream][WEIGHT[stream]]
    E = np.array([[np.nan if v is None else float(v) for v in row] for row in c["matrix"]["enrichment"]])
    N = np.array(c["matrix"]["n_episodes_per_cell"])
    rows = np.where(np.isfinite(E).any(1))[0].max() + 1
    cols = np.where(np.isfinite(E).any(0))[0].max() + 1
    return E[:rows, :cols], N[:rows, :cols], c

# ----------------------------------------------------------------------------- 画布
W, H = 7.2, 8.0
fig = plt.figure(figsize=(W, H))
def ax_in(x, y, w, h, **kw):
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)
import textwrap
def header(y, text, bold=True, color=INK, x=0.06, width=138):
    lines = [text] if bold else textwrap.wrap(text, width)
    for i, line in enumerate(lines):
        fig.text(x / W, (y - i * 0.105) / H, line, fontsize=7.5 if bold else 6.5, fontweight="bold" if bold else "normal",
                 va="bottom", ha="left", color=color)
def panel_tag(ax, tag, dx, dy):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")

n_ep = sum(stat[t]["episodes"] for t in TASKS)
header(7.84, "I.  Two attention pathways carry motion-token content into the action tokens")
header(7.72, f"Motion model, PatternLock | RouteStick, seed 42, {n_ep} episodes / {inferences} inferences with full attention stored; "
             "numbers are PatternLock | RouteStick, layer-mean over heads, episode-equal. (a) token layout in sequence order, the attention mask "
             "(grey), and the strong edges by layer (colour); (b) the same edges layer by layer, both tasks pooled.",
       bold=False, color=INK2)

# ============================================================================ (a) 序列布局 + 掩码 + 分层通路
ax = ax_in(0.25, 3.75, 6.80, 3.70)
ax.set_xlim(0, 100); ax.set_ylim(-5, 100); ax.axis("off")
panel_tag(ax, "(a)", -0.03, 0.97)

def rbox(x0, y0, x1, y1, fc, ec=INK3, lw=0.6, z=2, r=1.2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))
def orth(points, color, ls="-", lw=1.0, z=4):
    pts = np.array(points, float)
    if len(pts) > 2:
        ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=z)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                shrinkA=0, shrinkB=0, mutation_scale=6), zorder=z)
def label(x, y, text, color=INK, ha="left", va="center", fs=6.3, z=6):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, linespacing=1.1, zorder=z,
            bbox=dict(boxstyle="square,pad=0.1", fc="#ffffff", ec="none", alpha=0.9))

# 序列行（prefix → suffix）
Y0, Y1 = 55, 66
MEMX = (1.0, 42.0); FR_SPLIT = MEMX[0] + (MEMX[1] - MEMX[0]) * 0.86
IMB = (45.5, 57.0); IMW = (58.5, 70.0); TXTX = (72.5, 80.5); DIV = 83.2; ACT = (86.0, 99.0)
rbox(*MEMX[:1], Y0, MEMX[1], Y1, "#eef2f8")
ax.plot([FR_SPLIT, FR_SPLIT], [Y0, Y1], color=INK3, lw=0.5, ls=":", zorder=3)
ax.text((MEMX[0] + FR_SPLIT) / 2, Y1 - 1.6, "Frame memory · demonstration", ha="center", va="top", fontsize=6.8, fontweight="bold", zorder=3)
ax.text((FR_SPLIT + MEMX[1]) / 2, Y1 - 1.6, "exec.", ha="center", va="top", fontsize=6.3, fontweight="bold", zorder=3)
# motion token：按时间交错，画成橙色细条
motion_xs = [6.5, 13.0, 19.5, 26.0, 32.5, 39.5]
for mx in motion_xs:
    ax.add_patch(Rectangle((mx - 0.35, Y0 + 3.8), 0.7, Y1 - Y0 - 6.6, fc=C2, ec="none", zorder=3))
ax.text((MEMX[0] + MEMX[1]) / 2, Y0 + 0.9, f"{FRAME_TOKENS} frame tok. + ≈{stat['PatternLock']['motion_tokens']:.1f} | "
        f"{stat['RouteStick']['motion_tokens']:.1f} motion tok. (orange), time-interleaved", ha="center", va="bottom", fontsize=5.6, color=INK2, zorder=3)
rbox(*IMB[:1], Y0, IMB[1], Y1, "#f5f5f3"); ax.text(sum(IMB) / 2, (Y0 + Y1) / 2 + 1.2, "Current image\n3rd person", ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=3)
ax.text(sum(IMB) / 2, Y0 + 1.0, f"{N_IMG} tokens", ha="center", va="bottom", fontsize=5.9, color=INK2, zorder=3)
rbox(*IMW[:1], Y0, IMW[1], Y1, "#f5f5f3"); ax.text(sum(IMW) / 2, (Y0 + Y1) / 2 + 1.2, "Current image\nwrist", ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=3)
ax.text(sum(IMW) / 2, Y0 + 1.0, f"{N_IMG} tokens", ha="center", va="bottom", fontsize=5.9, color=INK2, zorder=3)
rbox(*TXTX[:1], Y0, TXTX[1], Y1, "#fdf1ec"); ax.text(sum(TXTX) / 2, (Y0 + Y1) / 2 + 1.2, "Language", ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=3)
ax.text(sum(TXTX) / 2, Y0 + 1.0, f"{TXT} tokens", ha="center", va="bottom", fontsize=5.9, color=INK2, zorder=3)
ax.plot([DIV, DIV], [Y0 - 3, Y1 + 3], color=INK3, lw=0.6, ls="--", zorder=3)
ax.text(DIV - 0.8, Y1 + 3.2, "prefix", ha="right", va="bottom", fontsize=6, color=INK3)
ax.text(DIV + 0.8, Y1 + 3.2, "suffix", ha="left", va="bottom", fontsize=6, color=INK3)
rbox(*ACT[:1], Y0, ACT[1], Y1, "#f1f1f1"); ax.text(sum(ACT) / 2, (Y0 + Y1) / 2 + 1.2, "Action", ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=3)
ax.text(sum(ACT) / 2, Y0 + 1.0, f"{HORIZON} tokens", ha="center", va="bottom", fontsize=5.9, color=INK2, zorder=3)

# 掩码（灰色，画在序列行上方）：谁可以读谁
def mask_arc(x_from, x_to, y, text, both=False, text_x=None, color=INK3):
    """从 x_from 顶部出发，经高度 y，到 x_to 顶部（箭头朝向被读的一方）。"""
    pts = [(x_from, Y1), (x_from, y), (x_to, y), (x_to, Y1)]
    if both:
        ax.plot([x_from, x_from, x_to], [Y1, y, y], color=color, lw=0.7, zorder=4)
        ax.annotate("", xy=(x_to, Y1), xytext=(x_to, y), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.7, shrinkA=0, shrinkB=0, mutation_scale=5))
        ax.annotate("", xy=(x_from, Y1), xytext=(x_from, y), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.7, shrinkA=0, shrinkB=0, mutation_scale=5))
    else:
        orth(pts, color, lw=0.7)
    ax.text(text_x if text_x is not None else (x_from + x_to) / 2, y + 0.9, text, ha="center", va="bottom", fontsize=5.9, color=INK2, zorder=5)
mask_arc(4.0, 30.0, 72.5, "memory ↔ memory: all-to-all (frames ↔ motion, demo ↔ exec)", both=True)
mask_arc(76.0, 36.0, 84.0, "language → memory (one way)", text_x=56)
mask_arc(78.5, 63.0, 75.0, "images ↔ language ↔ images: all-to-all", both=True, text_x=63)
mask_arc(92.0, 50.0, 93.0, "action → every prefix token (causal within action)", text_x=71)
ax.text(1.0, 96.5, "Attention mask (grey arrows = allowed):", fontsize=6.3, fontweight="bold", va="bottom")
ax.text(1.0, 86.5, "masked (exactly 0 in all layers):\nmemory ✕ images / language / action\nimages ✕ memory", fontsize=6.0, color=INK2, va="bottom", linespacing=1.15)

# 分层强边（彩色，画在序列行下方）：竖直落到各自的横向通道，再上到目标
def edge(x_from, x_to, y, color, ls="-", lw=1.1):
    orth([(x_from, Y0), (x_from, y), (x_to, y), (x_to, Y0)], color, ls=ls, lw=lw)
fc = stat["PatternLock"]["frame_carry_motion"], stat["RouteStick"]["frame_carry_motion"]
# 1 motion → frames（记忆区内部）
edge(26.0, 10.0, 44.0, C2, lw=1.1)
label(27.5, 44.0, f"frames read motion  L0 {pair('frame_reads_motion', 0)}  L1 {pair('frame_reads_motion', 1)}\n→ frames carry "
      f"{fc[0][3]:.0f}|{fc[1][3]:.0f}% motion content by L3 (final {fc[0][18]:.0f}|{fc[1][18]:.0f}%, α = 0.5)", color=C2)
# 2 motion → language
edge(32.5, 75.5, 37.0, C2)
label(33.5, 37.0, f"language reads motion  L0 {pair('text_reads_motion', 0)}  L1 {pair('text_reads_motion', 1)}, "
      f"≤{max(stat[t]['text_reads_motion'][2:].max() for t in TASKS):.0f}% from L2", color=C2)
# 3 frames → language（弱，时序错开）
edge(16.0, 73.5, 30.5, INK3, ls="--", lw=0.9)
label(17.0, 30.5, f"language reads frames  L0 {pair('text_reads_frame', 0)}, ≤{max(stat[t]['text_reads_frame'][1:].max() for t in TASKS):.0f}% after; "
      f"frames carry 0% motion at L0", color=INK2)
# 4 frames → action
arf = [stat[t]["action_reads_frame"] for t in TASKS]
edge(6.0, 89.0, 24.0, C1)
label(7.0, 24.0, f"action reads frames  L14 {pair('action_reads_frame', 14)}  L15 {pair('action_reads_frame', 15)}  L16 {pair('action_reads_frame', 16)}, "
      f"other layers ≤{max(np.delete(a, [14, 15, 16]).max() for a in arf):.0f}%", color=C1)
# 5 language → action
art = np.concatenate([stat[t]["action_reads_text"] for t in TASKS])
edge(78.0, 93.0, 17.0, C2)
label(20.0, 17.0, f"action reads language  every layer {rng(art)} (mean {art.mean():.0f}%)", color=C2)
# 6 language → images, images → action（灰）
irt = np.concatenate([stat[t]["image_reads_text"][14:17].ravel() for t in TASKS])
ax.annotate("", xy=(IMW[1], (Y0 + Y1) / 2), xytext=(TXTX[0], (Y0 + Y1) / 2), arrowprops=dict(arrowstyle="-|>", color=INK3, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=6), zorder=4)
ax.text((IMW[1] + TXTX[0]) / 2, Y1 + 0.8, f"imgs read lang.\nL14–16 {rng(irt)}", ha="center", va="bottom", fontsize=5.6, color=INK2, linespacing=1.1, zorder=5)
ari = np.concatenate([stat[t]["action_reads_image"] for t in TASKS])
edge(60.0, 97.0, 10.0, INK3, lw=0.8)
label(61.0, 10.0, f"action reads images  L0–17 {rng(ari)}", color=INK2)
# 动作里 motion 含量的路径分解
v = {t: stat[t]["via_0.5"] for t in TASKS}
label(1.0, 13.0, f"motion content reaching action (α = 0.5)\n{v['PatternLock']['total']:.1f} | {v['RouteStick']['total']:.1f}% of its attention mass\n"
      f"last hop via frames {v['PatternLock']['frame'] * 100:.0f} | {v['RouteStick']['frame'] * 100:.0f}%\n"
      f"via language {v['PatternLock']['text'] * 100:.0f} | {v['RouteStick']['text'] * 100:.0f}%\n"
      f"via images {v['PatternLock']['image'] * 100:.0f} | {v['RouteStick']['image'] * 100:.0f}%", ha="left", va="top", fs=6.0)
ax.add_line(Line2D([1, 5], [-2.5, -2.5], color=C1, lw=1.2)); ax.text(6, -2.5, "Route A: motion → frames → action", va="center")
ax.add_line(Line2D([33, 37], [-2.5, -2.5], color=C2, lw=1.2)); ax.text(38, -2.5, "Route B: motion → language → action", va="center")
ax.add_line(Line2D([66, 70], [-2.5, -2.5], color=INK3, lw=0.9, ls="--")); ax.text(71, -2.5, "weak chain", va="center", color=INK2)
ax.add_line(Line2D([82, 86], [-2.5, -2.5], color=INK3, lw=0.8)); ax.text(87, -2.5, "non-memory", va="center", color=INK2)

# ============================================================================ (b) 逐层折线（两任务合并，按局等权）
def pooled(key, layers):
    return np.mean([stat[t][key][:layers] for t in TASKS], 0)   # 两任务各 50 局，等权平均即合并均值
axb = ax_in(0.70, 0.95, 2.25, 2.10)
panel_tag(axb, "(b)", -0.30, 1.14)
Lx = np.arange(18)
series_b = [("frame_carry_motion", 19, C1, "frame memory: cumulative motion content (α = 0.5)"),
            ("text_reads_motion", 18, C2, "language reads motion tokens (direct)"),
            ("text_reads_frame", 18, C3, "language reads frame memory (direct)"),
            ("action_reads_frame", 18, C4, "action reads frame memory (direct)")]
for key, n, color, _ in series_b:
    axb.plot(np.arange(n), pooled(key, n), color=color, lw=1.6, zorder=3)
axb.axvline(3, color=GRID, lw=0.8, zorder=1); axb.text(3.4, 69, "L3", fontsize=6.5, color=INK3, va="top")
axb.axvspan(13.5, 16.5, color="#f1f1f1", lw=0, zorder=0); axb.text(15, 69, "L14–16", fontsize=6.5, color=INK3, va="top", ha="center")
axb.set_xlim(-0.5, 18.5); axb.set_ylim(0, 70); axb.set_xticks([0, 3, 6, 9, 12, 15, 18]); axb.set_yticks([0, 20, 40, 60])
axb.grid(axis="y", color=GRID, lw=0.5); axb.set_axisbelow(True)
for sp in ("top", "right"):
    axb.spines[sp].set_visible(False)
axb.set_xlabel("layer", labelpad=1); axb.set_ylabel("% of attention / content", labelpad=2)
axb.set_title(f"motion model, PatternLock + RouteStick pooled\n({n_ep} episodes, {inferences} inferences, episode-equal)", fontsize=6.5, pad=3, loc="left")
fcm = pooled("frame_carry_motion", 19)
axb.text(18.3, fcm[18] + 1.5, f"{fcm[18]:.0f}%", ha="right", va="bottom", fontsize=6.5, color=INK2)
arf_p = pooled("action_reads_frame", 18); k = 14 + int(np.argmax(arf_p[14:17]))
axb.text(k, arf_p[k] + 2, f"{arf_p[k]:.0f}%", ha="center", va="bottom", fontsize=6.5, color=INK2)
trm = pooled("text_reads_motion", 18); k2 = int(np.argmax(trm))
axb.text(k2, trm[k2] + 3.5, f"{trm[k2]:.0f}%", ha="center", va="bottom", fontsize=6.5, color=INK2)
fig.legend([Line2D([], [], color=c, lw=1.6) for _, _, c, _ in series_b], [lab for _, _, _, lab in series_b], loc="upper left",
           bbox_to_anchor=(0.30 / W, 0.72 / H), ncol=1, frameon=False, handlelength=1.6, handletextpad=0.6, borderaxespad=0, labelspacing=0.35)

# ============================================================================ (c) 子目标对位矩阵（RouteStick）
header(3.50, "II.  The pathways exist, but which demo segment", x=3.55)
header(3.38, "is read shows no sub-goal alignment", x=3.55)
header(3.26, "RouteStick, motion model unless noted, 50 episodes per model, all inferences; rows pool episodes with different numbers of sub-goals m "
             "(n = episodes per row); frame stream weighted 'single', motion stream 'overlap'; D = diagonal enrichment, episode-level 95% CI.",
       bold=False, color=INK2, x=3.55, width=66)
panels_c = [("mot", "action", "frame", "action reads frames"),
            ("mot", "text", "frame", "language reads frames"),
            ("ctx", "text", "frame", "baseline · language reads frames"),
            ("mot", "text", "motion", "language reads motion (exception)")]
mats = [matrix("RouteStick", m, r, s) for m, r, s, _ in panels_c]
vmax = max(np.nanmax(E) for E, _, _ in mats)
norm = TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=vmax)
cax_list = []
for k, ((m, r, s, title), (E, N, c)) in enumerate(zip(panels_c, mats)):
    col, row = k % 2, k // 2
    a = ax_in(3.85 + col * 1.70, 1.78 - row * 1.12, 1.10, 0.68)
    cax_list.append(a)
    a.imshow(np.ma.masked_invalid(E), cmap=DIVERGE, norm=norm, interpolation="nearest", aspect="auto")
    for (j, jp), val in np.ndenumerate(E):
        if np.isfinite(val):
            a.text(jp, j, f"{val:.1f}", ha="center", va="center", fontsize=6.0, color="#ffffff" if (val > 2.2 or val < 0.35) else INK, zorder=3)
    a.set_xticks(range(E.shape[1])); a.set_xticklabels([str(i + 1) for i in range(E.shape[1])])
    a.set_yticks(range(E.shape[0])); a.set_yticklabels([str(i + 1) for i in range(E.shape[0])])
    a.tick_params(length=0, pad=1.5)
    for sp in a.spines.values():
        sp.set_visible(False)
    for j in range(E.shape[0]):
        a.text(E.shape[1] - 0.42, j, f"n={N[j, 0]}", ha="left", va="center", fontsize=6.0, color=INK3, clip_on=False)
    D = c["D"]; sl = c["slope"]
    lines = [title, f"D {D['mean']:.2f} [{D['ci95'][0]:.2f}, {D['ci95'][1]:.2f}], slope {sl['mean']:+.2f}"]
    if s == "motion" and r == "text":
        sc = succ["cells"][f"RouteStick/{m}/{r}/{s}"]["D"]
        lines.append(f"success only, n=16: D {sc['mean']:.2f} [{sc['ci95'][0]:.2f}, {sc['ci95'][1]:.2f}]")
    a.set_title("\n".join(lines), fontsize=6.0, pad=2, loc="left")
    if row == 1:
        a.set_xlabel("demo segment j′ read", labelpad=1)
    if col == 0:
        a.set_ylabel("executing sub-goal j", labelpad=1)
panel_tag(cax_list[0], "(c)", -0.26, 1.50)
cb_ax = ax_in(3.85, 0.30, 2.80, 0.07)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGE), cax=cb_ax, orientation="horizontal")
cb.set_ticks([0, 0.5, 1, 2, 3, round(float(vmax), 1)])
cb.ax.tick_params(length=2, pad=1.5); cb.outline.set_linewidth(0.4)
cb.set_label("attention share ÷ token-uniform share (1 = uniform)", labelpad=2)

fig.savefig(OUT / "fig_pathway.pdf")
fig.savefig(OUT / "fig_pathway.png", dpi=300)
print("FIG_PATHWAY=DONE", OUT / "fig_pathway.pdf", OUT / "fig_pathway.png")
