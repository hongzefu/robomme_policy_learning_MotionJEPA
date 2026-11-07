"""论文主图：motion token 到动作的两条注意力通路（第一部分）与「读哪段记忆不随子目标调整」（第二部分）。

只读仓库已归档的 JSON，不跑模型。全部数字由数据文件算出；面板 (a) 的示意图标签也从数据格式化，不手抄。
用法：plot_pathway_figure.py [仓库根]（默认 /data/hongzefu/robomme_policy_learning_MotionJEPA）。输出到脚本所在目录。

数据来源：
- 第一部分 Swap 两任务闭合前 4 次推理（motion 模型，seed 42）：
  docs/evaluation-doc/swap-first-bin-attn-seed42-1105/records/action-flow-summary.json
  （direct_layer_pct 形状 (层, 读者组, 来源组)；alpha_*/carry_by_layer 形状 (层+1, token 组, 来源组)；
    alpha_*/route_via_source 形状 (中转组, 来源组)；组序见 chain_hops.py）
- 第一部分 PatternLock / RouteStick 的逐层读取：docs/evaluation-doc/subgoal-rollout-attn-seed42-1106/records/layer-flow.json
- 第二部分：同目录 summary.json（A2 矩阵）、decompose.json（首段 / 真对位 / 末段）、success-only.json（例外的成功局读数）
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/hongzefu/robomme_policy_learning_MotionJEPA")
OUT = Path(__file__).resolve().parent
D1106 = REPO / "docs/evaluation-doc/subgoal-rollout-attn-seed42-1106/records"
D1105 = REPO / "docs/evaluation-doc/swap-first-bin-attn-seed42-1105/records"

# 固定分类色与发散色（用户指定）
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
DIVERGE = LinearSegmentedColormap.from_list("diverge", ["#2a78d6", "#f0efec", "#e34948"])
DIVERGE.set_bad("#ffffff")
INK, INK2, INK3 = "#1f1f1f", "#555555", "#8a8a8a"
GRID = "#d9d8d2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": INK3, "axes.linewidth": 0.5, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ----------------------------------------------------------------------------- 第一部分：读 Swap 的 action-flow
FRAME, MOTION, IMAGE, TEXT, ACTION = list(range(4)), list(range(4, 8)), [8, 9], 10, 11
flow = json.load(open(D1105 / "action-flow-summary.json", encoding="utf-8"))
swap = {}
for short, key in (("Button", "motion|ButtonUnmaskSwap"), ("Video", "motion|VideoUnmaskSwap")):
    cell = flow["cells"][key]
    direct, counts = np.array(cell["direct_layer_pct"]), np.array(cell["token_counts"], float)
    weight = counts[FRAME] / counts[FRAME].sum()
    carry = np.array(cell["alpha_0.5"]["carry_by_layer"])
    s = {
        "episodes": cell["episodes"],
        "frame_reads_motion": np.array([(weight[:, None] * direct[l][FRAME][:, MOTION]).sum() for l in range(18)]),
        "frame_carry_motion": np.array([(weight[:, None] * carry[l][FRAME][:, MOTION]).sum() for l in range(19)]),  # 第 l 层输入
        "text_reads_frame": direct[:, TEXT][:, FRAME].sum(1),
        "text_reads_motion": direct[:, TEXT][:, MOTION].sum(1),
        "action_reads_frame": direct[:, ACTION][:, FRAME].sum(1),
        "action_reads_text": direct[:, ACTION][:, TEXT],
        "action_reads_image": direct[:, ACTION][:, IMAGE].sum(1),
        "image_reads_text": direct[:, IMAGE][:, :, TEXT],              # (层, 2 路图像)
        "image_reads_memory_max": float(direct[:, IMAGE][:, :, FRAME + MOTION].max()),
    }
    for alpha in ("0.2", "0.5", "0.8"):
        via = np.array(cell[f"alpha_{alpha}"]["route_via_source"])[:, MOTION].sum(1)
        s[f"via_{alpha}"] = {"total": via.sum(), "frame": via[FRAME].sum() / via.sum(), "text": via[TEXT] / via.sum(), "image": via[IMAGE].sum() / via.sum()}
    swap[short] = s

layer_flow = json.load(open(D1106 / "layer-flow.json", encoding="utf-8"))
def lf(task, model, reader, block, layer):
    rec = layer_flow[f"{task}/{model}/{reader}"]
    return rec["by_layer_pct"][layer][rec["blocks"].index(block)]

def pair(key, layer, nd=0):
    """Button | Video 两个数的字符串。"""
    return f"{swap['Button'][key][layer]:.{nd}f}|{swap['Video'][key][layer]:.{nd}f}%"

def rng(values, nd=0):
    return f"{min(values):.{nd}f}–{max(values):.{nd}f}%"

via_frame = [swap[t][f"via_{a}"]["frame"] * 100 for t in swap for a in ("0.2", "0.5", "0.8")]
via_text = [swap[t][f"via_{a}"]["text"] * 100 for t in swap for a in ("0.2", "0.5", "0.8")]
via_image = [swap[t][f"via_{a}"]["image"] * 100 for t in swap for a in ("0.2", "0.5", "0.8")]
assert swap["Button"]["image_reads_memory_max"] == 0 and swap["Video"]["image_reads_memory_max"] == 0
image_text_1416 = np.concatenate([swap[t]["image_reads_text"][14:17].ravel() for t in swap])
action_text_all = np.concatenate([swap[t]["action_reads_text"] for t in swap])
action_image_all = np.concatenate([swap[t]["action_reads_image"] for t in swap])

# ----------------------------------------------------------------------------- 第二部分：A2 矩阵、拆解、成功局
summary = json.load(open(D1106 / "summary.json", encoding="utf-8"))
decomp = json.load(open(D1106 / "decompose.json", encoding="utf-8"))
succ = json.load(open(D1106 / "success-only.json", encoding="utf-8"))
WEIGHT = {"frame": "single", "motion": "overlap"}

def a2(task, model, reader, stream):
    return summary["A2"][task][model][reader][stream][WEIGHT[stream]]

def matrix(task, model, reader, stream):
    c = a2(task, model, reader, stream)
    E = np.array([[np.nan if v is None else float(v) for v in row] for row in c["matrix"]["enrichment"]])
    N = np.array(c["matrix"]["n_episodes_per_cell"])
    rows = np.where(np.isfinite(E).any(1))[0].max() + 1
    cols = np.where(np.isfinite(E).any(0))[0].max() + 1
    return E[:rows, :cols], N[:rows, :cols], c

# ----------------------------------------------------------------------------- 画布
W, H = 7.2, 8.0
fig = plt.figure(figsize=(W, H))
def ax_in(x, y, w, h, **kw):  # 英寸坐标（左下角）→ 轴
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)

import textwrap
def header(y, text, bold=True, color=INK):
    """标题行；灰色副标题按约 138 字符折行，行距 0.105 英寸。返回最后一行的 y。"""
    lines = [text] if bold else textwrap.wrap(text, 138)
    for i, line in enumerate(lines):
        fig.text(0.06 / W, (y - i * 0.105) / H, line, fontsize=7.5 if bold else 6.5, fontweight="bold" if bold else "normal",
                 va="bottom", ha="left", color=color)
    return y - (len(lines) - 1) * 0.105

def panel_tag(ax, tag, dx, dy):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")

header(7.84, "I.  Two attention pathways carry motion-token content into the action tokens")
header(7.72, "Swap tasks (ButtonUnmaskSwap | VideoUnmaskSwap), motion model, seed 42, 4 inferences up to first gripper closure; "
             "numbers are Button | Video.  Layer order: L0–1 language reads memory  →  by ≈L3 frames hold most of their final motion content  →  "
             "L14–16 action reads frames.", bold=False, color=INK2)
header(4.32, "II.  The pathways exist, but which demo segment is read shows no sub-goal alignment")
header(4.20, "PatternLock / RouteStick, all 922 inferences, seed 42.  (c) RouteStick, 50 episodes per model; rows pool episodes with different "
             "numbers of sub-goals m (n = episodes per row); frame stream weighted 'single', motion stream 'overlap'; D = diagonal enrichment "
             "with episode-level 95% CI.  (d) n varies by series (see caption); RS = RouteStick, PL = PatternLock; post-hoc decomposition of D, "
             "not corrected for multiple comparisons.", bold=False, color=INK2)

# ============================================================================ (a) 通路示意图
ax = ax_in(0.28, 4.50, 3.50, 3.05)
ax.set_xlim(0, 100); ax.set_ylim(-11, 100); ax.axis("off")
panel_tag(ax, "(a)", -0.06, 0.95)

def box(x0, y0, x1, y1, title, body=None, fc="#f5f5f3", ec=INK3, title_fs=7):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=fc, ec=ec, lw=0.6, zorder=2))
    cx = (x0 + x1) / 2
    if body is None:
        ax.text(cx, (y0 + y1) / 2, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", zorder=3)
    else:
        ax.text(cx, y1 - 1.8, title, ha="center", va="top", fontsize=title_fs, fontweight="bold", zorder=3)
        ax.text(cx, y0 + 1.4, body, ha="center", va="bottom", fontsize=6.5, color=INK2, zorder=3, linespacing=1.15)

def orth(points, color, ls="-", lw=1.0, zorder=4):
    """正交折线箭头：除最后一段外用线段，最后一段带箭头。"""
    pts = np.array(points, float)
    if len(pts) > 2:
        ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=zorder)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=7), zorder=zorder)

def elabel(x, y, text, color=INK, ha="left", va="center", fs=6.5):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, linespacing=1.12, zorder=5)

# 节点
box(4, 87, 46, 97, "Motion tokens", fc="#eef2f8")
box(2, 57, 40, 77, "Frame memory",
    body=f"motion content after layer\nL0 {pair('frame_carry_motion', 1)}   L2 {pair('frame_carry_motion', 3)}\n"
         f"L17 {pair('frame_carry_motion', 18)}", fc="#eef2f8")
box(48, 60, 76, 74, "Language tokens", fc="#fdf1ec")
box(72, 32, 98, 44, "Current image", body=f"reads memory {swap['Button']['image_reads_memory_max']:.0f}%")
box(2, 1, 62, 19, "Action tokens",
    body=f"motion content (α=0.5): {swap['Button']['via_0.5']['total']:.1f}% | {swap['Video']['via_0.5']['total']:.1f}%\n"
         f"via frames {rng(via_frame)}, language {rng(via_text)},\nimage {rng(via_image)} (last hop, α 0.2–0.8)", fc="#f1f1f1")

# 路线 A：motion → 帧记忆 → 动作
orth([(12, 87), (12, 77)], C1)
elabel(14, 82, f"frames read motion\nL0 {pair('frame_reads_motion', 0)}   L1 {pair('frame_reads_motion', 1)}")
orth([(10, 57), (10, 19)], C1)
elabel(12, 30, "action reads frames\n"
               f"L14 {pair('action_reads_frame', 14, 1)}\nL15 {pair('action_reads_frame', 15, 1)}\nL16 {pair('action_reads_frame', 16, 1)}\n"
               f"PL/RS L16 {lf('PatternLock', 'mot', 'action', '历史视觉', 16):.0f}|{lf('RouteStick', 'mot', 'action', '历史视觉', 16):.0f}% "
               f"(base {lf('PatternLock', 'ctx', 'action', '历史视觉', 16):.0f}|{lf('RouteStick', 'ctx', 'action', '历史视觉', 16):.0f}%)")
# 路线 B：motion → 语言 → 动作
orth([(46, 92), (62, 92), (62, 74)], C2)
elabel(64, 83, f"language reads motion\nL0 {pair('text_reads_motion', 0)}   L1 {pair('text_reads_motion', 1)}")
orth([(58, 60), (58, 19)], C2)
elabel(60, 26, f"action reads\nlanguage {rng(action_text_all)}\n(mean {action_text_all.mean():.0f}%)")
# 不成立的串联：帧 → 语言（虚线灰）
orth([(40, 67), (48, 67)], INK3, ls="--", lw=0.9)
elabel(40, 55, f"language reads frames\nmostly at L0 ({pair('text_reads_frame', 0)}),\nwhen frames hold 0%\nmotion; L1 {pair('text_reads_frame', 1, 1)}",
       color=INK2, ha="center", va="top")
# 图像只能经语言拿记忆；动作读图像
orth([(76, 67), (85, 67), (85, 44)], INK3, lw=0.8)
elabel(83, 55, f"image reads\nlanguage\nL14–16 {rng(image_text_1416)}", color=INK2, ha="right")
orth([(85, 32), (85, 10), (62, 10)], INK3, lw=0.8)
elabel(76, 8, f"action reads image\n{rng(action_image_all)} (L0–17)", color=INK2, ha="center", va="top")
# 图例
ax.add_line(Line2D([2, 8], [-5, -5], color=C1, lw=1.2)); ax.text(9.5, -5, "Route A: motion → frames → action", va="center")
ax.add_line(Line2D([2, 8], [-9.5, -9.5], color=C2, lw=1.2)); ax.text(9.5, -9.5, "Route B: motion → language → action", va="center")
ax.add_line(Line2D([63, 68], [-5, -5], color=INK3, lw=0.9, ls="--")); ax.text(69.5, -5, "weak chain (timing)", va="center", color=INK2)
ax.add_line(Line2D([63, 68], [-9.5, -9.5], color=INK3, lw=0.8)); ax.text(69.5, -9.5, "non-memory edges", va="center", color=INK2)

# ============================================================================ (b) 逐层折线（Swap 两任务）
layers = np.arange(18)
bx = {}
for i, task in enumerate(("Button", "Video")):
    a = ax_in(4.42 + i * 1.46, 5.20, 1.26, 2.10)
    bx[task] = a
    s = swap[task]
    a.plot(np.arange(19), s["frame_carry_motion"], color=C1, lw=1.6, zorder=3)
    a.plot(layers, s["text_reads_motion"], color=C2, lw=1.6, zorder=3)
    a.plot(layers, s["text_reads_frame"], color=C3, lw=1.6, zorder=3)
    a.plot(layers, s["action_reads_frame"], color=C4, lw=1.6, zorder=3)
    a.axvline(3, color=GRID, lw=0.8, zorder=1)
    a.text(3.4, 79, "≈L3", fontsize=6.5, color=INK3, va="top")
    a.axvspan(13.5, 16.5, color="#f1f1f1", lw=0, zorder=0)
    a.text(15, 79, "L14–16", fontsize=6.5, color=INK3, va="top", ha="center")
    a.set_xlim(-0.5, 18.5); a.set_ylim(0, 80)
    a.set_xticks([0, 3, 6, 9, 12, 15, 18]); a.set_yticks([0, 20, 40, 60, 80])
    a.grid(axis="y", color=GRID, lw=0.5); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.set_title(f"{task}UnmaskSwap\n(n = {s['episodes']} episodes)", fontsize=6.5, pad=3)
    a.set_xlabel("layer", labelpad=1)
    a.text(18.3, s["frame_carry_motion"][18] + 2, f"{s['frame_carry_motion'][18]:.0f}%", ha="right", va="bottom", fontsize=6.5, color=INK2)
    k = 14 + int(np.argmax(s["action_reads_frame"][14:17]))
    a.text(k, s["action_reads_frame"][k] + 2.5, f"{s['action_reads_frame'][k]:.1f}%", ha="center", va="bottom", fontsize=6.5, color=INK2)
    if i == 0:
        a.set_ylabel("% of attention / content", labelpad=2)
    else:
        a.set_yticklabels([])
panel_tag(bx["Button"], "(b)", -0.40, 0.98)
handles = [Line2D([], [], color=C1, lw=1.6), Line2D([], [], color=C2, lw=1.6), Line2D([], [], color=C3, lw=1.6), Line2D([], [], color=C4, lw=1.6)]
labels = ["frame memory: motion content (cum., α=0.5, layer-l input)", "language reads motion tokens (direct)",
          "language reads frame memory (direct)", "action reads frame memory (direct)"]
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(4.12 / W, 4.92 / H), ncol=1, frameon=False,
           handlelength=1.6, handletextpad=0.6, borderaxespad=0, labelspacing=0.35)

# ============================================================================ (c) 子目标对位矩阵（RouteStick）
panels_c = [("mot", "action", "frame", "motion · action reads frames"),
            ("mot", "text", "frame", "motion · language reads frames"),
            ("ctx", "text", "frame", "baseline · language reads frames"),
            ("mot", "text", "motion", "motion · language reads motion (exception)")]
mats = [matrix("RouteStick", m, r, s) for m, r, s, _ in panels_c]
vmax = max(np.nanmax(E) for E, _, _ in mats)
norm = TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=vmax)
cax_list = []
for k, ((m, r, s, title), (E, N, c)) in enumerate(zip(panels_c, mats)):
    col, row = k % 2, k // 2
    a = ax_in(0.40 + col * 1.80, 2.45 - row * 1.60, 1.32, 0.95)
    cax_list.append(a)
    a.imshow(np.ma.masked_invalid(E), cmap=DIVERGE, norm=norm, interpolation="nearest", aspect="auto")
    for (j, jp), v in np.ndenumerate(E):
        if np.isfinite(v):
            a.text(jp, j, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                   color="#ffffff" if (v > 2.2 or v < 0.35) else INK, zorder=3)
    a.set_xticks(range(E.shape[1])); a.set_xticklabels([str(i + 1) for i in range(E.shape[1])])
    a.set_yticks(range(E.shape[0])); a.set_yticklabels([str(i + 1) for i in range(E.shape[0])])
    a.tick_params(length=0, pad=1.5)
    for sp in a.spines.values():
        sp.set_visible(False)
    for j in range(E.shape[0]):
        a.text(E.shape[1] - 0.42, j, f"n={N[j, 0]}", ha="left", va="center", fontsize=6.5, color=INK3, clip_on=False)
    D = c["D"]; sl = c["slope"]
    lines = [title, f"D {D['mean']:.2f} [{D['ci95'][0]:.2f}, {D['ci95'][1]:.2f}], slope {sl['mean']:+.2f}"]
    if s == "motion" and r == "text":
        sc = succ["cells"][f"RouteStick/{m}/{r}/{s}"]["D"]
        lines.append(f"success-only (n=16) D {sc['mean']:.2f} [{sc['ci95'][0]:.2f}, {sc['ci95'][1]:.2f}]")
    a.set_title("\n".join(lines), fontsize=6.5, pad=2, loc="left")
    if row == 1:
        a.set_xlabel("demo segment j′ read", labelpad=1)
    if col == 0:
        a.set_ylabel("executing sub-goal j", labelpad=1)
panel_tag(cax_list[0], "(c)", -0.22, 1.32)
cb_ax = ax_in(0.40, 0.36, 2.66, 0.08)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGE), cax=cb_ax, orientation="horizontal")
cb.set_ticks([0, 0.5, 1, 2, 3, round(float(vmax), 1)])
cb.ax.tick_params(length=2, pad=1.5)
cb.outline.set_linewidth(0.4)
cb.set_label("attention share ÷ token-uniform share (1 = uniform)", labelpad=2)

# ============================================================================ (d) 首段 / 真对位 / 末段
rows_d = [("mot", "action", "frame", "motion · action reads frames"),
          ("mot", "action", "motion", "motion · action reads motion"),
          ("mot", "text", "frame", "motion · language reads frames"),
          ("mot", "text", "motion", "motion · language reads motion (exception)"),
          ("ctx", "action", "frame", "baseline · action reads frames"),
          ("ctx", "text", "frame", "baseline · language reads frames")]
TASKS = (("RouteStick", "RS"), ("PatternLock", "PL"))
a = ax_in(4.95, 0.60, 2.10, 2.82)
panel_tag(a, "(d)", -0.20, 1.03)
series = [("first_later", "first segment (j ≥ 2 reads seg. 1)", C1, "o"),
          ("diag_middle", "true alignment (1 < j < m reads seg. j)", C2, "D"),
          ("last_earlier", "last segment (j < m reads seg. m)", C3, "s")]
ticks, ticklabels = [], []
y = 0.0
for gi, (m, r, s, name) in enumerate(rows_d):
    a.text(0.02, y, name, ha="left", va="center", fontsize=6.5, color=INK, zorder=5,
           bbox=dict(boxstyle="square,pad=0.15", fc="#ffffff", ec="none"))
    for task, short in TASKS:
        y += 1.0
        cell = decomp["cells"][f"{task}/{m}/{r}/{s}"]
        for si, (key, _, color, marker) in enumerate(series):
            v = cell[key]
            off = (si - 1) * 0.27
            a.plot(v["ci95"], [y + off, y + off], color=color, lw=0.9, solid_capstyle="butt", zorder=2)
            a.plot(v["mean"], y + off, marker=marker, ms=4, mfc=color, mec="#ffffff", mew=0.6, color=color, lw=0, zorder=3)
        ticks.append(y); ticklabels.append(short)
    y += 1.6
a.axvline(1, color=INK3, lw=0.7, zorder=1)
a.set_yticks(ticks); a.set_yticklabels(ticklabels)
a.set_ylim(y - 1.6 + 0.6, -0.6)
a.set_xlim(0, 2.75); a.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
a.set_xlabel("enrichment (share ÷ uniform), mean & 95% CI", labelpad=1)
a.grid(axis="x", color=GRID, lw=0.5); a.set_axisbelow(True)
a.tick_params(axis="y", length=0, pad=2)
for sp in ("top", "right", "left"):
    a.spines[sp].set_visible(False)
h = [Line2D([], [], color=c, marker=mk, ms=4, mfc=c, mec="#ffffff", lw=0.9) for _, _, c, mk in series]
a.legend(h, [lab for _, lab, _, _ in series], loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False,
         handlelength=1.4, handletextpad=0.5, borderaxespad=0, labelspacing=0.3, ncol=1)

fig.savefig(OUT / "fig_pathway.pdf")
fig.savefig(OUT / "fig_pathway.png", dpi=300)
print("FIG_PATHWAY=DONE", OUT / "fig_pathway.pdf", OUT / "fig_pathway.png")
