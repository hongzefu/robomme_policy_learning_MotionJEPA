#!/usr/bin/env python
"""论文用四张英文图（StopCube rollout 线性探针，预测「到正确窗口还有几步」）。

fig-r2-layers-baseline / fig-r2-layers-motion：逐层 R²（单分面，单栏尺寸；参考线只留 Time only 与 R²=0）。
fig-r2-fit：第 17 层五组 × 两模型的拟合直线整合在一个坐标轴里（不画散点），右侧文字表列 R² 与 slope。
fig-point-accuracy：逐点准确率逐层图（2×2，双栏尺寸，无参考线）。
四张图都不含 "Time + target count"。
每张同时输出 PDF（fonttype 42）与 PNG（dpi 300）。只读数据，只写本目录。
数据键沿用同仓库 records/diagnostics/plot_q4_compare.py 与 plot_r2_complete.py。
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

REPO = "/data/hongzefu/robomme_policy_learning_MotionJEPA"
SRC_PROBE = f"{REPO}/docs/evaluation-doc/stopcube-rollout-1105/records/probe-summary.json"
FIT_DIR = f"{REPO}/v1-store/evaluation/stopcube-rollout-1105/analysis/fit-data"
SRC_FIT = f"{FIT_DIR}/fit-summary.json"
SRC_NPZ = f"{FIT_DIR}/fit-predictions.npz"
SRC_CLS = f"{FIT_DIR}/class-layers.json"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

LAB = "q4"
N_LAYERS = 18
X_INPUT = 0
X_LAYERS = np.arange(1, N_LAYERS + 1)        # 第 k 层画在 x=k+1
TICK_LAYERS = [2, 5, 8, 11, 14, 17]
X_TICKS = [X_INPUT] + [k + 1 for k in TICK_LAYERS]
X_TICKLABELS = ["Input"] + [str(k) for k in TICK_LAYERS]

# ---------- 字体与全局样式 ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
})
INK = "#111111"
INK2 = "#444444"
MUTED = "#7a7a7a"
GRID = "#e4e4e4"
SPINE = "#b8b8b8"
ZERO = "#a8a8a8"

# 组名 → (颜色, 线宽, probe-summary / class-layers / npz 里的读数键)
GROUPS = [
    ("Action token",   "#2a78d6", 1.6, "action"),
    ("Text",           "#eb6834", 1.6, "文字"),
    ("Current frame",  "#1baf7a", 1.3, "当前视觉"),
    ("History frames", "#eda100", 1.1, "历史视觉"),
    ("Motion",         "#e87ba4", 1.1, "motion"),
]
GROUP_COLOR = {g: c for g, c, _, _ in GROUPS}
TIME_NAME = "Time only"
MODELS = [("ctx", "Baseline"), ("mot", "Motion model")]
# 图 3 线型按模型：Motion model 实线、Baseline 虚线
MODEL_LS = {"ctx": (0, (4.5, 2.2)), "mot": "-"}


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(colors=INK2, labelcolor=INK)


# ---------- 线尾标签防重叠 ----------
def spread_labels(ends, ylim, min_gap):
    """ends: [(name, y)]，返回 name → 标签 y。按 y 排序后两两推开 min_gap，整体不出 ylim。"""
    order = sorted(range(len(ends)), key=lambda i: ends[i][1])
    ys = [ends[i][1] for i in order]
    lo, hi = ylim
    pad = min_gap * 0.6
    ys = [min(max(y, lo + pad), hi - pad) for y in ys]
    for _ in range(8):
        for k in range(1, len(ys)):
            if ys[k] - ys[k - 1] < min_gap:
                ys[k] = ys[k - 1] + min_gap
        over = ys[-1] - (hi - pad)
        if over > 0:
            ys = [y - over for y in ys]
        for k in range(len(ys) - 2, -1, -1):
            if ys[k + 1] - ys[k] < min_gap:
                ys[k] = ys[k + 1] - min_gap
        under = (lo + pad) - ys[0]
        if under > 0:
            ys = [y + under for y in ys]
    return {ends[i][0]: ys[k] for k, i in enumerate(order)}


X_DATA_LO = -0.7
X_DATA_HI = N_LAYERS + 0.45


def hline(ax, y, **kw):
    """参考横线只画在数据区（不伸进右侧标签区）。"""
    ax.plot([X_DATA_LO, X_DATA_HI], [y, y], **kw)


def end_labels(ax, ends, ylim, min_gap, x_line_end, x_text, fontsize=7):
    pos = spread_labels(ends, ylim, min_gap)
    for name, y_end, color, bold in ends_full(ends):
        y_lab = pos[name]
        if abs(y_lab - y_end) > 1e-9:
            ax.plot([x_line_end + 0.15, x_text - 0.15], [y_end, y_lab],
                    color=color, lw=0.5, alpha=0.9, zorder=2)
        ax.text(x_text, y_lab, name, color=color, fontsize=fontsize,
                fontweight="bold" if bold else "normal", va="center", ha="left", zorder=5)


def ends_full(ends):
    return [(e[0], e[1], e[2], e[3]) for e in ends]


# =====================================================================
# 图 1 / 图 2：逐层 R²
# =====================================================================
YLIM_R2 = (-0.25, 1.0)
X_TEXT = N_LAYERS + 0.9
XLIM_R2 = (-0.7, N_LAYERS + 9.6)


def series_r2(L, key):
    """返回 (x, y)；action 没有输入点，其余组输入点来自 prefix_layer0。"""
    xs, ys = [], []
    if key == "action":
        items = L["layers"]["action"]
    else:
        items = L["layers"]["prefix"].get(key)
        if not items:
            return None
        l0 = L["layers"]["prefix_layer0"].get(key)
        if l0:
            xs.append(X_INPUT)
            ys.append(l0["r2"])
    for i, it in enumerate(items):
        xs.append(X_LAYERS[i])
        ys.append(it["r2"])
    return np.array(xs, float), np.array(ys, float)


def fig_r2_layers(p, fit, m, m_name, out_stem):
    L = p["models"][m]["labels"][LAB]
    tb_t = fit[m][LAB]["time"]["r2"]          # Time only
    n_ep = L["n_episodes"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    style_axes(ax)
    hline(ax, 0, color=ZERO, lw=0.7, zorder=1)
    hline(ax, tb_t, color=INK, lw=0.8, ls=(0, (1.2, 2.0)), zorder=2)

    ends = [(TIME_NAME, tb_t, INK, False)]
    r2_at_17 = {}
    for name, color, lw, key in GROUPS:
        sv = series_r2(L, key)
        if sv is None:
            continue
        x, y = sv
        ax.plot(x, y, color=color, lw=lw, marker="o", ms=2.6, mfc=color, mec="white",
                mew=0.4, solid_capstyle="round", zorder=3)
        ends.append((name, y[-1], color, lw > 1.5))
        r2_at_17[name] = y[-1]
    end_labels(ax, ends, YLIM_R2, (YLIM_R2[1] - YLIM_R2[0]) / 13.5,
               X_LAYERS[-1], X_TEXT)

    ax.set_xlim(*XLIM_R2)
    ax.set_ylim(*YLIM_R2)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_TICKLABELS)
    ax.set_yticks(np.arange(-0.2, 1.01, 0.2))
    ax.set_xlabel("Model layer")
    ax.set_ylabel("R²")
    ax.set_title(f"{m_name} ({n_ep} episodes)", loc="left", fontweight="bold", pad=4)
    fig.tight_layout(pad=0.3)
    save(fig, out_stem)
    return r2_at_17, tb_t


# =====================================================================
# 图 3：第 17 层拟合直线，五组 × 两模型整合在一个坐标轴
# =====================================================================
PANELS = [
    ("Action token", "action"),
    ("Text", "文字"),
    ("Current frame", "当前视觉"),
    ("History frames", "历史视觉"),
    ("Motion", "motion"),
]


def wls_line(x, yhat, groups):
    """yhat 对 x 的加权最小二乘直线，权重 = 1/该局样本数（每局等权）。返回 (截距, 斜率)。"""
    _, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv)
    w = 1.0 / counts[inv]
    X = np.column_stack([np.ones_like(x), x])
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X, XtW @ yhat)
    return beta[0], beta[1]


def fig_r2_fit(fit, npz, out_stem):
    """单坐标轴：9 条拟合直线（每局等权 WLS），右侧文字表列 R²（fit-summary）与 slope（现算）。"""
    # ---- 先算所有直线 ----
    table = []                      # (组名, 模型名, R², slope, n points)
    segs = []                       # (组名, 模型键, x0, x1, 截距, 斜率)
    counts = {}                     # 模型键 → (局数, 点数)，从 npz 现数
    for m, m_name in MODELS:
        y = npz[f"{m}/{LAB}/y"].astype(float)
        g = npz[f"{m}/{LAB}/groups"]
        counts[m] = (len(np.unique(g)), len(y))
        assert counts[m] == (fit[m][LAB]["n_episodes"], fit[m][LAB]["n_samples"])
        for name, key in PANELS:
            k = f"{m}/{LAB}/{key}"
            if k not in npz.files:
                table.append((name, m_name, None, None, 0))
                continue
            pred = npz[k].astype(float)
            ok = np.isfinite(pred) & np.isfinite(y)
            xx, yy, gg = y[ok], pred[ok], g[ok]
            a, b = wls_line(xx, yy, gg)
            segs.append((name, m, xx.min(), xx.max(), a, b))
            table.append((name, m_name, fit[m][LAB][key]["r2"], b, int(ok.sum())))

    # 坐标范围：只需包住 9 条线段的端点（x、y 共用同一范围，等比例）
    ends = []
    for _, _, x0, x1, a, b in segs:
        ends += [x0, x1, a + b * x0, a + b * x1]
    lo, hi = min(ends), max(ends)
    pad = (hi - lo) * 0.05
    lim = (lo - pad, hi + pad)

    # ---- 版式：左方形坐标轴，右文字表 + 图例 ----
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_axes([0.085, 0.14, 0.40, 0.78])
    ax.set_facecolor("white")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("W")
    ax.grid(color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(colors=INK2, labelcolor=INK)
    ax.plot(lim, lim, color=ZERO, lw=0.8, zorder=1)

    # 不加白边（白边会盖住贴着的另一条线）；实线在下、虚线在上让虚线缝隙里透出实线。
    # Motion 与 History frames 的 Motion model 直线几乎重合（slope 0.128 vs 0.126），
    # 故 Motion 画粗、History frames 画细且压在最上，两条都能看见。
    SOLID_LW = {"Action token": 2.0, "Text": 2.0, "Current frame": 2.0,
                "History frames": 1.1, "Motion": 2.8}
    SOLID_Z = {"Action token": 3, "Text": 3, "Current frame": 3, "Motion": 3.2,
               "History frames": 3.4}
    for name, m, x0, x1, a, b in segs:
        color = GROUP_COLOR[name]
        xs = np.array([x0, x1])
        if m == "mot":
            ax.plot(xs, a + b * xs, color=color, lw=SOLID_LW[name], ls="-",
                    solid_capstyle="butt", zorder=SOLID_Z[name])
        else:
            ax.plot(xs, a + b * xs, color=color, lw=1.5, ls=MODEL_LS[m],
                    dash_capstyle="butt", zorder=4)

    ticks = np.arange(0, lim[1] + 1, 100)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("True steps to correct window")
    ax.set_ylabel("Predicted steps")
    ax.set_title("Layer 17", loc="left", fontweight="bold", pad=4)

    # ---- 右侧文字表（坐标 0–1 的隐形轴）----
    axt = fig.add_axes([0.54, 0.05, 0.45, 0.90])
    axt.set_axis_off()
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)
    FS = 7.5
    x_grp, x_br, x_bs, x_mr, x_ms = 0.00, 0.44, 0.60, 0.80, 0.96   # 列右端（数值右对齐）
    y_top, dy = 0.94, 0.082
    # 表头两行
    axt.text((x_br + x_bs) / 2 - 0.045, y_top, "Baseline", ha="center", va="center",
             fontsize=FS, fontweight="bold", color=INK)
    axt.text((x_mr + x_ms) / 2 - 0.045, y_top, "Motion model", ha="center", va="center",
             fontsize=FS, fontweight="bold", color=INK)
    y_sub = y_top - dy * 0.85
    for xr, xs_ in ((x_br, x_bs), (x_mr, x_ms)):
        axt.text(xr, y_sub, "R²", ha="right", va="center", fontsize=FS, color=INK2)
        axt.text(xs_, y_sub, "slope", ha="right", va="center", fontsize=FS, color=INK2)
    y_rule = y_sub - dy * 0.55
    axt.plot([x_grp, x_ms], [y_rule, y_rule], color=SPINE, lw=0.6)
    # 数据行
    by_cell = {(n, mn): (r2, b) for n, mn, r2, b, _ in table}
    for i, (name, _) in enumerate(PANELS):
        yy = y_rule - dy * 0.6 - dy * i
        axt.text(x_grp, yy, name, ha="left", va="center", fontsize=FS,
                 color=GROUP_COLOR[name], fontweight="bold")
        for mn, (xr, xs_) in (("Baseline", (x_br, x_bs)), ("Motion model", (x_mr, x_ms))):
            r2, b = by_cell[(name, mn)]
            if r2 is None:
                axt.text((xr + xs_) / 2 - 0.045, yy, "—", ha="center", va="center",
                         fontsize=FS, color=MUTED)
            else:
                axt.text(xr, yy, f"{r2:.2f}", ha="right", va="center", fontsize=FS, color=INK)
                axt.text(xs_, yy, f"{b:.2f}", ha="right", va="center", fontsize=FS, color=INK)
    y_bot = y_rule - dy * 0.6 - dy * (len(PANELS) - 1) - dy * 0.55
    axt.plot([x_grp, x_ms], [y_bot, y_bot], color=SPINE, lw=0.6)

    # 表下：线型含义与局数（从数据算）
    n_c, n_m = counts["ctx"], counts["mot"]
    handles = [
        Line2D([], [], color=INK, lw=1.9, ls=MODEL_LS["mot"],
               label=f"Motion model ({n_m[0]} episodes, {n_m[1]} points)"),
        Line2D([], [], color=INK, lw=1.5, ls=MODEL_LS["ctx"],
               label=f"Baseline ({n_c[0]} episodes, {n_c[1]} points)"),
        Line2D([], [], color=ZERO, lw=0.8, label="y = x"),
    ]
    axt.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, y_bot - 0.03),
               frameon=False, fontsize=FS, handlelength=2.6, handletextpad=0.6,
               labelspacing=0.35, borderaxespad=0)
    axt.text(0.0, 0.0, "Each line: episode-weighted least-squares fit of predicted vs. true\n"
             "steps, drawn over that model's range of true values.",
             ha="left", va="bottom", fontsize=7, color=INK2, linespacing=1.3)
    save(fig, out_stem)
    return table


# =====================================================================
# 图 4：逐点准确率逐层图
# =====================================================================
ROWS_ACC = [("press", "Press now?"), ("c", "Passes elapsed")]
YLIM_ACC = (0.0, 1.0)


def series_acc(reader, task):
    """class-layers 里的逐层逐点准确率；层号 '-1' 为输入。"""
    xs, ys = [], []
    for k in sorted(reader.keys(), key=int):
        li = int(k)
        xs.append(X_INPUT if li < 0 else li + 1)
        ys.append(reader[k][task]["point"]["mean"])
    return np.array(xs, float), np.array(ys, float)


def fig_point_accuracy(cls, p, out_stem):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.095,
                        hspace=0.38, wspace=0.12)
    vals = {}
    for r, (task, task_name) in enumerate(ROWS_ACC):
        for c, (m, m_name) in enumerate(MODELS):
            ax = axes[r][c]
            style_axes(ax)
            ends = []
            for name, color, lw, key in GROUPS:
                rd = cls[m]["readers"].get(key)
                if rd is None:
                    continue
                x, y = series_acc(rd, task)
                ax.plot(x, y, color=color, lw=lw, marker="o", ms=2.6, mfc=color,
                        mec="white", mew=0.4, solid_capstyle="round", zorder=3)
                ends.append((name, y[-1], color, lw > 1.5))
                vals[(task, m, name)] = y[-1]
            end_labels(ax, ends, YLIM_ACC, (YLIM_ACC[1] - YLIM_ACC[0]) / 13.5,
                       X_LAYERS[-1], X_TEXT)
            ax.set_xlim(-0.7, N_LAYERS + 7.4)
            ax.set_ylim(*YLIM_ACC)
            ax.set_xticks(X_TICKS)
            ax.set_xticklabels(X_TICKLABELS)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            n_ep = p["models"][m]["labels"][LAB]["n_episodes"]
            ax.set_title(f"{task_name} — {m_name} ({n_ep} episodes)", loc="left",
                         fontweight="bold", pad=4)
            if r == 1:
                ax.set_xlabel("Model layer")
            if c == 0:
                ax.set_ylabel("Point accuracy")
    save(fig, out_stem)
    return vals


def save(fig, stem):
    fig.savefig(f"{OUT_DIR}/{stem}.pdf")
    fig.savefig(f"{OUT_DIR}/{stem}.png", dpi=300)
    plt.close(fig)
    print("written:", f"{OUT_DIR}/{stem}.pdf", f"{OUT_DIR}/{stem}.png")


def main():
    p = load_json(SRC_PROBE)
    fit = load_json(SRC_FIT)
    npz = np.load(SRC_NPZ, allow_pickle=False)
    cls = load_json(SRC_CLS)
    from matplotlib import font_manager
    print("font:", font_manager.findfont(font_manager.FontProperties(family="Liberation Sans")))

    print("\n== Fig 1/2: layer-17 R² per group, and Time-only baseline ==")
    for m, m_name in MODELS:
        stem = "fig-r2-layers-baseline" if m == "ctx" else "fig-r2-layers-motion"
        r2_17, tb_t = fig_r2_layers(p, fit, m, m_name, stem)
        print(f"[{m_name}] episodes={p['models'][m]['labels'][LAB]['n_episodes']} "
              f"samples={p['models'][m]['labels'][LAB]['n_samples']}")
        for name, v in r2_17.items():
            print(f"  layer 17 {name:16s} R² = {v:.4f}")
        print(f"  {TIME_NAME:16s} R² = {tb_t:.4f}")

    print("\n== Fig 3: layer 17, 9 lines (group | model | R² | slope | n points) ==")
    table = fig_r2_fit(fit, npz, "fig-r2-fit")
    for name, m_name, r2, b, n in table:
        if r2 is None:
            print(f"{name:20s} | {m_name:13s} | n/a   | n/a   | -")
        else:
            print(f"{name:20s} | {m_name:13s} | {r2:.4f} | {b:.4f} | {n}")

    print("\n== Fig 4: layer-17 point accuracy ==")
    vals = fig_point_accuracy(cls, p, "fig-point-accuracy")
    for (task, m, name), v in sorted(vals.items()):
        print(f"{task:5s} | {m:3s} | {name:20s} | {v:.4f}")


if __name__ == "__main__":
    main()
