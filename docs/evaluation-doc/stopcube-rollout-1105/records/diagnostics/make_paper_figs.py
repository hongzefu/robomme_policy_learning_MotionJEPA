#!/usr/bin/env python
"""论文用四张英文图（StopCube rollout 线性探针，预测「到正确窗口还有几步」）。

fig-r2-layers-baseline / fig-r2-layers-motion：逐层 R²（单分面，单栏尺寸）。
fig-r2-fit：第 17 层两模型叠画的拟合直线对比（2×3，双栏尺寸）。
fig-point-accuracy：逐点准确率逐层图（2×2，双栏尺寸）。
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
TIME_N_NAME = "Time + target count"
TIME_NAME = "Time only"
MODELS = [("ctx", "Baseline"), ("mot", "Motion model")]
# 图 3 两模型的颜色与标记（与五组配色错开）
MODEL_STYLE = {"ctx": ("#5a5a5a", "o"), "mot": ("#8e44ad", "^")}


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
    tb_n = L["time_baseline"]["r2"]          # Time + target count
    tb_t = fit[m][LAB]["time"]["r2"]          # Time only
    assert abs(tb_n - fit[m][LAB]["time_N"]["r2"]) < 1e-9
    n_ep = L["n_episodes"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    style_axes(ax)
    hline(ax, 0, color=ZERO, lw=0.7, zorder=1)
    hline(ax, tb_n, color=INK, lw=1.4, ls=(0, (5, 2.5)), zorder=2)
    hline(ax, tb_t, color=INK, lw=0.8, ls=(0, (1.2, 2.0)), zorder=2)

    ends = [(TIME_N_NAME, tb_n, INK, False), (TIME_NAME, tb_t, INK, False)]
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
    return r2_at_17, tb_n, tb_t


# =====================================================================
# 图 3：第 17 层拟合直线，两模型叠画
# =====================================================================
PANELS = [
    ("Action token", "action"),
    ("Text", "文字"),
    ("Current frame", "当前视觉"),
    ("History frames", "历史视觉"),
    ("Motion", "motion"),
    (TIME_N_NAME, "time_N"),
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
    table = []
    data, vals = {}, []
    for m, _ in MODELS:
        y = npz[f"{m}/{LAB}/y"].astype(float)
        g = npz[f"{m}/{LAB}/groups"]
        data[m] = (y, g)
        vals.append(y)
        for _, key in PANELS:
            k = f"{m}/{LAB}/{key}"
            if k in npz.files:
                v = npz[k].astype(float)
                vals.append(v[np.isfinite(v)])
    allv = np.concatenate(vals)
    lo, hi = allv.min(), allv.max()
    pad = (hi - lo) * 0.04
    lim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.8))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.845, bottom=0.075,
                        hspace=0.52, wspace=0.12)
    for idx, (name, key) in enumerate(PANELS):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor("white")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("N")
        ax.grid(color=GRID, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(SPINE)
        ax.tick_params(colors=INK2, labelcolor=INK)
        ax.plot(lim, lim, color=ZERO, lw=0.7, zorder=1)

        lines = []
        for m, m_name in MODELS:
            k = f"{m}/{LAB}/{key}"
            if k not in npz.files:
                continue
            y, g = data[m]
            pred = npz[k].astype(float)
            ok = np.isfinite(pred) & np.isfinite(y)
            xx, yy, gg = y[ok], pred[ok], g[ok]
            color, marker = MODEL_STYLE[m]
            ax.scatter(xx, yy, s=7, marker=marker, color=color, alpha=0.28,
                       linewidths=0, zorder=2)
            a, b = wls_line(xx, yy, gg)
            xs = np.array([xx.min(), xx.max()])
            ax.plot(xs, a + b * xs, color=color, lw=1.8, solid_capstyle="round", zorder=4,
                    path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
            r2 = fit[m][LAB][key]["r2"]
            lines.append((f"{m_name}: R² = {r2:.2f}, slope = {b:.2f}", color))
            table.append((name, m_name, r2, b, int(ok.sum())))
        if key == "motion":
            lines.append(("Baseline: no motion tokens", MUTED))
            table.append((name, "Baseline", None, None, 0))
        # 小标题（黑、粗）在最上，其下按模型颜色各写一行统计量，都放在轴外、不压数据
        ax.set_title(name, fontweight="bold", pad=3 + 9.5 * len(lines))
        for i, (txt, color) in enumerate(lines):
            ax.text(0.5, 1.015 + 0.062 * (len(lines) - 1 - i), txt, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7, color=color)
        if idx // 3 == 1:
            ax.set_xlabel("True steps to correct window")
        if idx % 3 == 0:
            ax.set_ylabel("Predicted steps")
        else:
            ax.set_yticklabels([])
    n_c, n_m = fit["ctx"][LAB], fit["mot"][LAB]
    handles = [
        Line2D([], [], color=MODEL_STYLE["ctx"][0], marker="o", ms=5, lw=0, alpha=0.8,
               label=f"Baseline ({n_c['n_episodes']} episodes, {n_c['n_samples']} decision points)"),
        Line2D([], [], color=MODEL_STYLE["mot"][0], marker="^", ms=5, lw=0, alpha=0.8,
               label=f"Motion model ({n_m['n_episodes']} episodes, {n_m['n_samples']} decision points)"),
        Line2D([], [], color=INK, lw=1.8, label="Episode-weighted least-squares fit"),
        Line2D([], [], color=ZERO, lw=0.9, label="y = x"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.995), handlelength=2.0, columnspacing=1.6,
               handletextpad=0.5, labelspacing=0.3)
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
            tb = cls[m]["time_N"][task]["point"]["mean"]
            hline(ax, tb, color=INK, lw=1.4, ls=(0, (5, 2.5)), zorder=2)
            ends = [(TIME_N_NAME, tb, INK, False)]
            for name, color, lw, key in GROUPS:
                rd = cls[m]["readers"].get(key)
                if rd is None:
                    continue
                x, y = series_acc(rd, task)
                ax.plot(x, y, color=color, lw=lw, marker="o", ms=2.6, mfc=color,
                        mec="white", mew=0.4, solid_capstyle="round", zorder=3)
                ends.append((name, y[-1], color, lw > 1.5))
                vals[(task, m, name)] = y[-1]
            vals[(task, m, TIME_N_NAME)] = tb
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

    print("\n== Fig 1/2: layer-17 R² per group, and time baselines ==")
    for m, m_name in MODELS:
        stem = "fig-r2-layers-baseline" if m == "ctx" else "fig-r2-layers-motion"
        r2_17, tb_n, tb_t = fig_r2_layers(p, fit, m, m_name, stem)
        print(f"[{m_name}] episodes={p['models'][m]['labels'][LAB]['n_episodes']} "
              f"samples={p['models'][m]['labels'][LAB]['n_samples']}")
        for name, v in r2_17.items():
            print(f"  layer 17 {name:16s} R² = {v:.4f}")
        print(f"  {TIME_N_NAME:16s} R² = {tb_n:.4f}")
        print(f"  {TIME_NAME:16s} R² = {tb_t:.4f}")

    print("\n== Fig 3: layer 17, per panel (group | model | R² | slope | n points) ==")
    table = fig_r2_fit(fit, npz, "fig-r2-fit")
    for name, m_name, r2, b, n in table:
        if r2 is None:
            print(f"{name:20s} | {m_name:13s} | n/a   | n/a   | -")
        else:
            print(f"{name:20s} | {m_name:13s} | {r2:.4f} | {b:.4f} | {n}")

    print("\n== Fig 4: layer-17 point accuracy and time baseline ==")
    vals = fig_point_accuracy(cls, p, "fig-point-accuracy")
    for (task, m, name), v in sorted(vals.items()):
        print(f"{task:5s} | {m:3s} | {name:20s} | {v:.4f}")


if __name__ == "__main__":
    main()
