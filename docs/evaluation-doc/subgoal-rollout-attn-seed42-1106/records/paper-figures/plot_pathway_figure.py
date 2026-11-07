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
W, H = 7.2, 8.85
YOFF = 0.15   # 所有面板统一上移的英寸数
fig = plt.figure(figsize=(W, H))
def ax_in(x, y, w, h, **kw):
    return fig.add_axes([x / W, (y + YOFF) / H, w / W, h / H], **kw)
import textwrap
def header(y, text, bold=True, color=INK, x=0.06, width=138):
    lines = [text] if bold else textwrap.wrap(text, width)
    for i, line in enumerate(lines):
        fig.text(x / W, (y + YOFF - i * 0.105) / H, line, fontsize=7.5 if bold else 6.5, fontweight="bold" if bold else "normal",
                 va="bottom", ha="left", color=color)
def panel_tag(ax, tag, dx, dy):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")

n_ep = sum(stat[t]["episodes"] for t in TASKS)
header(8.54, "I.  Two attention pathways carry motion-token content into the action tokens")
header(8.42, f"Motion model, PatternLock | RouteStick, seed 42, {n_ep} episodes / {inferences} inferences, head-averaged, episode-equal. (a) the attention mask "
             "as a block matrix and, per layer, where each block mainly reads; (b) the pathway edges per layer, both tasks pooled.",
       bold=False, color=INK2)

# ============================================================================ (a) 块状掩码矩阵 + 逐层水平箭头
BLOCKS = [("frames", [0, 1], C1), ("motion", [2, 3], C2), ("image\n3rd", [4], C4), ("image\nwrist", [5], C4), ("language", [6], C3), ("action", [7], INK2)]
def fold6(direct_by_layer, counts):
    """(层, 8, 8) → (层, 6, 6)，读者组内按 token 数加权（单组读者权重为 1）。"""
    M = np.zeros((direct_by_layer.shape[0], 6, 6))
    for a, (_, q, _) in enumerate(BLOCKS):
        w = counts[q] / max(counts[q].sum(), 1e-9) if len(q) > 1 else np.ones(1)
        for b, (_, k, _) in enumerate(BLOCKS):
            M[:, a, b] = (w[None, :, None] * direct_by_layer[:, q][:, :, k]).sum((1, 2))
    return M
M6 = np.mean([np.mean([fold6(i["direct"], i["counts"]) for i in episodes[t]], 0) for t in TASKS], 0) * 100   # 两任务等权
mask6 = np.array([[all(mask_zero[q, k] for q in qs for k in ks) for _, ks, _ in BLOCKS] for _, qs, _ in BLOCKS])
sizes = [f"{FRAME_TOKENS}", f"≈{stat['PatternLock']['motion_tokens']:.1f}|{stat['RouteStick']['motion_tokens']:.1f}", f"{N_IMG}", f"{N_IMG}", f"{TXT}", f"{HORIZON}"]

# --- (a) 左：块状注意力掩码（行 = query 块，列 = key 块，实心 = 可见）
axm = ax_in(1.00, 5.95, 1.50, 1.50)
panel_tag(axm, "(a)", -0.62, 1.30)
axm.set_xlim(-0.5, 5.5); axm.set_ylim(5.5, -0.5); axm.set_aspect("equal")
for r in range(6):
    for c in range(6):
        allowed = not mask6[r, c]
        fc = "#3a3a3a" if allowed else "#ffffff"
        if allowed and r == 5 and c == 5:
            fc = "#9a9a9a"   # 动作内部因果
        axm.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc=fc, ec="#ffffff", lw=1.0))
axm.text(5, 5, "causal", ha="center", va="center", fontsize=4.6, color="#ffffff")
labels6 = [n.replace("\n", " ") for n, _, _ in BLOCKS]
axm.set_xticks(range(6)); axm.set_xticklabels([f"{n}\n{sz}" for n, sz in zip(labels6, sizes)], fontsize=5.0, rotation=90)
axm.set_yticks(range(6)); axm.set_yticklabels(labels6, fontsize=5.0)
axm.tick_params(length=0, pad=2)
axm.xaxis.set_ticks_position("top"); axm.xaxis.set_label_position("top")
axm.set_xlabel("key (attended) block · tokens", labelpad=3, fontsize=5.8); axm.set_ylabel("query (reading) block", labelpad=2, fontsize=5.8)
for sp in axm.spines.values():
    sp.set_visible(False)
axm.add_patch(Rectangle((-0.5, -0.5), 6, 6, fc="none", ec=INK3, lw=0.5))
axm.plot([4.5, 4.5], [-0.5, 5.5], color="#ffffff", lw=2.2); axm.plot([-0.5, 5.5], [4.5, 4.5], color="#ffffff", lw=2.2)
fig.text(0.40 / W, (5.68 + YOFF) / H, "Block attention mask (dark = may attend; verified\nas exactly 0 elsewhere in every layer, head, episode).\nprefix = memory → images → language; suffix = action.",
         fontsize=5.6, color=INK2, va="top", ha="left", linespacing=1.2)

# --- (a) 右：逐层，每个读者块一条水平箭头指向它的主要外部来源
axl = ax_in(2.80, 4.00, 4.20, 3.95)
X = {name: x for x, name in enumerate(("frames", "motion", "images", "language", "action"))}
READERS = [("frames", 0, [1, 2, 3, 4], +0.34), ("motion", 1, [0, 2, 3, 4], +0.17), ("images", None, [0, 1, 4], 0.0), ("language", 4, [0, 1, 2, 3], -0.17), ("action", 5, [0, 1, 2, 3, 4], -0.34)]
LABEL_X = {"frames": (-0.10, "right"), "motion": (1.10, "left"), "images": (1.90, "right"), "language": (3.10, "left"), "action": (4.10, "left")}
SRC_OF = {0: "frames", 1: "motion", 2: "images", 3: "images", 4: "language"}
SRC_COLOR = {"frames": C1, "motion": C2, "images": C4, "language": C3}
cnt6 = np.mean([np.mean([i["counts"] for i in episodes[t]], 0) for t in TASKS], 0)
def reader_row(l, idx):
    if idx is None:   # 两路图像按 token 数合并
        w = cnt6[[4, 5]] / cnt6[[4, 5]].sum()
        return (w[:, None] * M6[l, 2:4]).sum(0)
    return M6[l, idx]
def pool_sources(row):
    out = {"frames": row[0], "motion": row[1], "images": row[2] + row[3], "language": row[4]}
    return out
def lw_of(v):
    return 0.3 + v / 100 * 2.2
drawn = []
for l in range(18):
    for name, idx, allowed, off in READERS:
        row = pool_sources(reader_row(l, idx))
        cands = sorted(((v, src) for src, v in row.items() if src != name and SRC_OF and X[src] != X[name]), reverse=True)
        picks = [cands[0]] if cands[0][0] >= 5 else []
        if name == "action" and len(cands) > 1 and cands[1][0] >= 4:
            picks.append(cands[1])
        for k, (v, src) in enumerate(picks):
            y = l + off + (0.0 if k == 0 else -0.17)
            xs, xr = X[src], X[name]
            axl.annotate("", xy=(xr - (0.06 if xr > xs else -0.06), y), xytext=(xs + (0.06 if xr > xs else -0.06), y),
                         arrowprops=dict(arrowstyle="-|>", color=SRC_COLOR[src], lw=lw_of(v), shrinkA=0, shrinkB=0, mutation_scale=5 + v / 25), zorder=3)
            tx, ha = LABEL_X[name]
            if k == 1:
                tx += 0.33
            axl.text(tx, y, f"{v:.0f}", ha=ha, va="center", fontsize=4.6, color=SRC_COLOR[src], zorder=4)
            drawn.append((l, name, src, round(float(v), 1)))
for x in X.values():
    axl.axvline(x, color=GRID, lw=0.5, zorder=1)
for l in range(0, 18, 2):
    axl.axhspan(l - 0.5, l + 0.5, color="#f6f6f4", lw=0, zorder=0)
axl.set_xlim(-0.6, 4.75); axl.set_ylim(17.65, -0.85)
axl.set_xticks(list(X.values())); axl.set_xticklabels(["frame\nmemory", "motion\ntokens", "current\nimages", "language", "action"], fontsize=6.0)
axl.xaxis.set_ticks_position("top")
axl.set_yticks(range(18)); axl.set_yticklabels([f"L{l}" for l in range(18)], fontsize=5.6)
axl.tick_params(length=0, pad=2)
for sp in axl.spines.values():
    sp.set_visible(False)
axl.set_ylabel("layer", labelpad=2)
# 图例：颜色 = 来源，宽度 = 读者注意力占比
hx, hy = 0.78, 17.55
for k, (src, col) in enumerate(SRC_COLOR.items()):
    pass
fig.text(2.80 / W, (3.90 + YOFF) / H, "One row per layer. Each horizontal arrow runs from a source block to the block that reads it: the reader's largest external source "
         "at that layer (action: top two). Width and number = % of the reader's attention, printed beside the reader's column; "
         "colour = source. Two tasks pooled.", fontsize=5.6, color=INK2, va="top", ha="left", wrap=True)
hdl = [Line2D([], [], color=c, lw=1.6) for c in SRC_COLOR.values()]
fig.legend(hdl, ["source: frame memory", "source: motion tokens", "source: current images", "source: language"], loc="upper left",
           bbox_to_anchor=(2.80 / W, (3.52 + YOFF) / H), ncol=4, frameon=False, fontsize=5.4, handlelength=1.4, handletextpad=0.5, columnspacing=1.0, borderaxespad=0)
json.dump({"groups": [n for n, _, _ in BLOCKS], "layer_matrix_pct": M6.round(2).tolist(), "masked": mask6.tolist(), "arrows": drawn},
          open(OUT / "fig_pathway-layers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)

# ============================================================================ (b) 逐层折线（两任务合并，按局等权）
def pooled(key, layers):
    return np.mean([stat[t][key][:layers] for t in TASKS], 0)   # 两任务各 50 局，等权平均即合并均值
axb = ax_in(0.70, 0.85, 2.25, 2.10)
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
           bbox_to_anchor=(0.30 / W, (0.62 + YOFF) / H), ncol=1, frameon=False, handlelength=1.6, handletextpad=0.6, borderaxespad=0, labelspacing=0.35)

# ============================================================================ (c) 子目标对位矩阵（RouteStick）
header(3.26, "II.  The pathways exist, but which demo segment", x=3.55)
header(3.14, "is read shows no sub-goal alignment", x=3.55)
header(3.02, "RouteStick, motion model unless noted, 50 episodes per model, all inferences; rows pool episodes with different numbers of sub-goals m "
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
    a = ax_in(3.85 + col * 1.70, 1.66 - row * 1.12, 1.10, 0.68)
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
panel_tag(cax_list[0], "(c)", -0.26, 1.22)
cb_ax = ax_in(3.85, 0.20, 2.80, 0.07)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGE), cax=cb_ax, orientation="horizontal")
cb.set_ticks([0, 0.5, 1, 2, 3, round(float(vmax), 1)])
cb.ax.tick_params(length=2, pad=1.5); cb.outline.set_linewidth(0.4)
cb.set_label("attention share ÷ token-uniform share (1 = uniform)", labelpad=2)

fig.savefig(OUT / "fig_pathway.pdf")
fig.savefig(OUT / "fig_pathway.png", dpi=300)
print("FIG_PATHWAY=DONE", OUT / "fig_pathway.pdf", OUT / "fig_pathway.png")
