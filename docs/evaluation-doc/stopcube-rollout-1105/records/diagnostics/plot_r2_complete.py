#!/usr/bin/env python
"""StopCube 真实 rollout 线性探针：两张面向非技术读者的 PNG。

图 1 rollout-r2-layers.png：每一层的 R²（含「只用时间和次数」粗虚线、「只用时间」细点线）。
图 2 rollout-r2-fit.png：第 17 层探针猜的步数对真实步数（散点 + 每局等权的加权最小二乘拟合直线）。
数据只读：probe-summary.json、fit-data/fit-summary.json、fit-data/fit-predictions.npz。
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

REPO = "/data/hongzefu/robomme_policy_learning_MotionJEPA"
SRC_PROBE = f"{REPO}/docs/evaluation-doc/stopcube-rollout-1105/records/probe-summary.json"
OUT_DIR = ("/tmp/claude-114466650/-data-hongzefu-robomme-policy-learning-MotionJEPA/"
           "e524da50-0b7d-47ff-ad49-563d85174b52/scratchpad")
SRC_FIT = f"{OUT_DIR}/fit-data/fit-summary.json"
SRC_NPZ = f"{OUT_DIR}/fit-data/fit-predictions.npz"
OUT_LAYERS = f"{OUT_DIR}/rollout-r2-layers.png"
OUT_FIT = f"{OUT_DIR}/rollout-r2-fit.png"

# ---------- 字体（沿用 plot_probe_layers.py：从 Noto ttc 抽 SC 面）----------
FONT = "Noto Sans CJK SC"
from fontTools.ttLib import TTFont

for src_ttc, out_name in (
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "NotoSansCJKsc-Regular.otf"),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "NotoSansCJKsc-Bold.otf")):
    out_path = os.path.join(OUT_DIR, out_name)
    if not os.path.exists(out_path):
        TTFont(src_ttc, fontNumber=2).save(out_path)
    font_manager.fontManager.addfont(out_path)
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11.5

# ---------- 配色 ----------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"
SERIES = {
    # 组名：(颜色, 线宽, 透明度, 数据键)
    "动作 token": ("#2a78d6", 2.6, 1.0, None),
    "文字":       ("#eb6834", 2.6, 1.0, "文字"),
    "当前画面":   ("#1baf7a", 1.8, 1.0, "当前视觉"),
    "历史帧":     ("#eda100", 1.4, 0.75, "历史视觉"),
    "motion":     ("#e87ba4", 1.4, 0.75, "motion"),
}
TIME_N_COLOR = "#52514e"

MODELS = [("ctx", "基线模型"), ("mot", "motion 模型")]
LABELS = [("q3", "离自己按还有几步"), ("q4", "离正确窗口还有几步")]
N_LAYERS = 18
X_INPUT = 0
X_LAYERS = np.arange(1, N_LAYERS + 1)
X_TICKS = [X_INPUT] + [X_LAYERS[i] for i in range(2, N_LAYERS, 3)]
X_TICKLABELS = ["输入"] + [f"第{i}层" for i in range(2, N_LAYERS, 3)]
X_LABEL_AT = N_LAYERS + 0.9


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


# =====================================================================
# 图 1：每一层的 R²
# =====================================================================
def series_values(L, group, metric):
    xs, ys = [], []
    key = SERIES[group][3]
    if group == "动作 token":
        items = L["layers"]["action"]
    else:
        items = L["layers"]["prefix"].get(key)
        if not items:
            return None
        l0 = L["layers"]["prefix_layer0"].get(key)
        if l0:
            xs.append(X_INPUT)
            ys.append(l0[metric])
    for i, it in enumerate(items):
        xs.append(X_LAYERS[i])
        ys.append(it[metric])
    return np.array(xs, float), np.array(ys, float)


def spread_labels(ends, ylim, min_gap, obstacles=()):
    """把线尾标签在 y 方向上推开，避免互相重叠，也避开参考线（obstacles 为 y 值）。"""
    order = sorted(range(len(ends)), key=lambda i: ends[i][1])
    ys = [ends[i][1] for i in order]
    lo, hi = ylim
    keep = min_gap * 0.55

    def avoid(ys):
        out = []
        for y in ys:
            for ob in obstacles:
                if abs(y - ob) < keep:
                    y = ob + keep if y >= ob else ob - keep
            out.append(y)
        return out

    ys = [min(max(y, lo + min_gap * 0.9), hi - min_gap * 0.9) for y in ys]
    for _ in range(6):
        ys = avoid(ys)
        for k in range(1, len(ys)):
            if ys[k] - ys[k - 1] < min_gap:
                ys[k] = ys[k - 1] + min_gap
        over = ys[-1] - (hi - min_gap * 0.9)
        if over > 0:
            ys = [y - over for y in ys]
        for k in range(len(ys) - 2, -1, -1):
            if ys[k + 1] - ys[k] < min_gap:
                ys[k] = ys[k + 1] - min_gap
        ys = avoid(ys)
    for k in range(1, len(ys)):
        if ys[k] - ys[k - 1] < min_gap:
            ys[k] = ys[k - 1] + min_gap
        for ob in obstacles:
            if abs(ys[k] - ob) < keep:
                ys[k] = ob + keep
    return {ends[i][0]: ys[k] for k, i in enumerate(order)}


def draw_layer_panel(ax, L, ylim, label_gap, obstacles=()):
    ends = []
    for name, (color, lw, alpha, _) in SERIES.items():
        sv = series_values(L, name, "r2")
        if sv is None:
            continue
        x, y = sv
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle="round",
                marker="o", ms=3.4 if lw < 2 else 4.2, mfc=color, mec=SURFACE,
                mew=0.6, zorder=3)
        ends.append((name, y[-1]))
    pos = spread_labels(ends, ylim, label_gap, obstacles)
    for name, y_end in ends:
        color = SERIES[name][0]
        y_lab = pos[name]
        if abs(y_lab - y_end) > 1e-9:
            ax.plot([X_LAYERS[-1] + 0.15, X_LABEL_AT - 0.15], [y_end, y_lab],
                    color=color, lw=0.8, alpha=0.8, zorder=2)
        ax.text(X_LABEL_AT, y_lab, name, color=color, fontsize=11.5,
                fontweight="bold" if SERIES[name][1] > 2 else "normal",
                va="center", ha="left", zorder=4)
    ax.set_xlim(-0.6, N_LAYERS + 4.3)
    ax.set_ylim(*ylim)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_TICKLABELS)
    ax.tick_params(colors=INK2, labelsize=11, length=3)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.set_facecolor(SURFACE)


FOOT1 = ("线性探针，50 局真实 rollout，按局 5 折的折外结果。"
         "差值没有区间，小差异不要当结论；各组取了组平均。")
XLABEL1 = "模型层（越往右越靠近输出）"


def fig_layers(p, fit):
    ylim = (-0.2, 1.0)
    label_gap = (ylim[1] - ylim[0]) / 16
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.4), dpi=170, facecolor=SURFACE)
    for r, (lab, lab_name) in enumerate(LABELS):
        for c, (m, m_name) in enumerate(MODELS):
            ax = axes[r][c]
            L = p["models"][m]["labels"][lab]
            tb_n = L["time_baseline"]["r2"]          # 时间 + 次数
            tb_t = fit[m][lab]["time"]["r2"]          # 只用时间
            draw_layer_panel(ax, L, ylim, label_gap, obstacles=(tb_n, tb_t, 0.0))
            ax.axhline(tb_n, color=INK, lw=2.2, ls=(0, (6, 3)), zorder=2)
            ax.axhline(tb_t, color=INK, lw=1.1, ls=(0, (1.5, 2.5)), zorder=2)
            ax.axhline(0, color="#a8a69f", lw=1.0, zorder=1)
            ax.set_title(f"{m_name}：{lab_name}\n", fontsize=13.5,
                         fontweight="bold", color=INK, loc="left", pad=6)
            ax.text(0.0, 1.015,
                    f"粗虚线 = 时间和次数（R² {tb_n:.2f}）；细点线 = 只用时间（R² {tb_t:.2f}）；"
                    f"灰线 = 和瞎猜一样",
                    transform=ax.transAxes, fontsize=10.5, color=INK2,
                    ha="left", va="bottom")
            ax.set_yticks(np.arange(-0.2, 1.01, 0.2))
    for ax in axes[-1]:
        ax.set_xlabel(XLABEL1, color=INK2, fontsize=12)
    for row in axes:
        row[0].set_ylabel("R²（1 = 全猜对，0 = 和瞎猜一样）", color=INK2, fontsize=12)
    fig.suptitle("每一层能读出多少进度（R²）", fontsize=17, fontweight="bold",
                 color=INK, y=0.985)
    fig.text(0.5, 0.012, FOOT1, ha="center", va="bottom", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955), h_pad=2.2, w_pad=1.6)
    fig.savefig(OUT_LAYERS, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# =====================================================================
# 图 2：猜的步数对真实步数（第 17 层）
# =====================================================================
# 列：(标题组名, npz/fit-summary 读数键, 颜色)
COLS = [
    ("动作 token",     "action",   SERIES["动作 token"][0]),
    ("文字",           "文字",     SERIES["文字"][0]),
    ("当前画面",       "当前视觉", SERIES["当前画面"][0]),
    ("历史帧",         "历史视觉", SERIES["历史帧"][0]),
    ("motion",         "motion",   SERIES["motion"][0]),
    ("只用时间和次数", "time_N",   TIME_N_COLOR),
]
ROWS = [(m, lab, f"{m_name}·{lab_name.replace('还有几步', '')}")
        for m, m_name in MODELS for lab, lab_name in LABELS]
FOOT2 = ("R² 是折外结果，1 = 全猜对，0 = 和只猜平均值一样，可为负；"
         "斜率 1 = 不偏，接近 0 = 基本读不出进度。线性探针，50 局真实 rollout，按局 5 折。")


def wls_line(x, yhat, groups):
    """y_hat 对 x 的加权最小二乘直线，权重 = 1/该局样本数（每局等权）。返回 (截距, 斜率)。"""
    _, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv)
    w = 1.0 / counts[inv]
    X = np.column_stack([np.ones_like(x), x])
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X, XtW @ yhat)
    return beta[0], beta[1]


def fig_fit(fit, npz):
    table = []  # (行名, 列名, R², 斜率)
    fig, axes = plt.subplots(4, 6, figsize=(19.5, 18.0), dpi=160, facecolor=SURFACE)
    for r, (m, lab, row_name) in enumerate(ROWS):
        y = npz[f"{m}/{lab}/y"].astype(float)
        groups = npz[f"{m}/{lab}/groups"]
        n_s = fit[m][lab]["n_samples"]
        n_e = fit[m][lab]["n_episodes"]
        # 行内统一范围：覆盖真值与全部有限预测
        vals = [y]
        for _, key, _ in COLS:
            k = f"{m}/{lab}/{key}"
            if k in npz.files:
                v = npz[k].astype(float)
                vals.append(v[np.isfinite(v)])
        allv = np.concatenate(vals)
        lo, hi = allv.min(), allv.max()
        pad = (hi - lo) * 0.05
        lim = (lo - pad, hi + pad)
        for c, (col_name, key, color) in enumerate(COLS):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color=GRID, lw=0.7, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#c3c2b7")
            ax.tick_params(colors=INK2, labelsize=9.5, length=3)
            k = f"{m}/{lab}/{key}"
            if k not in npz.files:
                ax.set_title(f"{col_name}\n", fontsize=12, fontweight="bold",
                             color=INK, pad=4)
                ax.text(0.5, 0.5, "基线没有这一组", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color=MUTED)
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ("left", "bottom"):
                    ax.spines[s].set_visible(False)
                ax.grid(False)
                table.append((row_name, col_name, None, None))
                continue
            pred = npz[k].astype(float)
            ok = np.isfinite(pred) & np.isfinite(y)
            xx, yy, gg = y[ok], pred[ok], groups[ok]
            ax.plot(lim, lim, color="#a8a69f", lw=1.0, zorder=1)
            ax.scatter(xx, yy, s=9, color=color, alpha=0.25, linewidths=0, zorder=2)
            a, b = wls_line(xx, yy, gg)
            xs = np.array([xx.min(), xx.max()])
            ax.plot(xs, a + b * xs, color=color, lw=2.8, solid_capstyle="round", zorder=3)
            r2 = fit[m][lab][key]["r2"]
            ax.set_title(f"{col_name}\nR² {r2:.2f}，斜率 {b:.2f}", fontsize=12,
                         fontweight="bold", color=INK, pad=4)
            table.append((row_name, col_name, r2, b))
            if r == 3:
                ax.set_xlabel("真实还剩几步", color=INK2, fontsize=11)
            if c > 0:
                ax.set_yticklabels([])
        ax0 = axes[r][0]
        ax0.set_ylabel("探针猜还剩几步", color=INK2, fontsize=11)
        ax0.text(-0.46, 0.5, f"{row_name}\n{n_e} 局，{n_s} 个决策点",
                 transform=ax0.transAxes, ha="center", va="center", rotation=90,
                 fontsize=13, fontweight="bold", color=INK, linespacing=1.6)
    # 顶部图例
    handles = [
        Line2D([], [], color="#a8a69f", lw=1.2, label="猜对的位置"),
        Line2D([], [], color=INK, lw=2.8, label="拟合直线（彩色=该组）"),
        Line2D([], [], color=INK2, marker="o", ms=5, lw=0, alpha=0.5, label="一次决策"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.972), fontsize=12.5, handlelength=2.4,
               columnspacing=2.4)
    fig.suptitle("探针猜的步数对真实步数（第 17 层）", fontsize=18, fontweight="bold",
                 color=INK, y=0.99)
    fig.text(0.5, 0.012, FOOT2, ha="center", va="bottom", fontsize=11.5, color=MUTED)
    fig.tight_layout(rect=(0.03, 0.025, 1, 0.945), h_pad=3.4, w_pad=1.2)
    fig.savefig(OUT_FIT, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return table


def main():
    p = load_json(SRC_PROBE)
    fit = load_json(SRC_FIT)
    npz = np.load(SRC_NPZ, allow_pickle=False)
    fp = font_manager.findfont(font_manager.FontProperties(family=FONT))
    print("font:", fp)
    if "NotoSansCJK" not in fp:
        print("警告：未找到 Noto Sans CJK SC", file=sys.stderr)
    fig_layers(p, fit)
    table = fig_fit(fit, npz)
    print("\n图 2 每格（行名 | 组 | R² | 斜率）")
    for row_name, col_name, r2, b in table:
        if r2 is None:
            print(f"{row_name} | {col_name} | 基线没有这一组 | —")
        else:
            print(f"{row_name} | {col_name} | {r2:.2f} | {b:.2f}")
    print("\nwritten:", OUT_LAYERS, OUT_FIT)


if __name__ == "__main__":
    main()
