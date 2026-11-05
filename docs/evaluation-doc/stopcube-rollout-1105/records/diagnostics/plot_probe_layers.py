#!/usr/bin/env python
"""把逐层线性探针结果画成两张面向非技术读者的 PNG。

图 1：时间基线 MAE − 该读数 MAE（越高越好）。
图 2：R²。
数据只读：probe-summary.json。
"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SRC = ("/data/hongzefu/robomme_policy_learning_MotionJEPA/docs/evaluation-doc/"
       "stopcube-rollout-1105/records/probe-summary.json")
OUT_DIR = ("/tmp/claude-114466650/-data-hongzefu-robomme-policy-learning-MotionJEPA/"
           "e524da50-0b7d-47ff-ad49-563d85174b52/scratchpad")
OUT_MAE = f"{OUT_DIR}/probe-vs-time-mae.png"
OUT_R2 = f"{OUT_DIR}/probe-vs-time-r2.png"

# ---------- 字体 ----------
FONT = "Noto Sans CJK SC"
# matplotlib 只读 .ttc 的第 0 面（JP），用 fontTools 把 SC 面（索引 2）抽成单独 .otf 再注册
import os
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

# ---------- 配色（dataviz 参考调色板，按固定顺序分配）----------
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
GREEN_BG = "#e8f5e9"
GREY_BG = "#f1f1ef"

MODELS = [("ctx", "基线模型"), ("mot", "motion 模型")]
LABELS = [("q3", "离自己按还有几步"), ("q4", "离正确窗口还有几步")]
N_LAYERS = 18
X_INPUT = 0            # 「输入」位置
X_LAYERS = np.arange(1, N_LAYERS + 1)   # 第 0～17 层
# 刻度：输入、第2、5、8、11、14、17 层（等距 3 格，首尾都有标）
X_TICKS = [X_INPUT] + [X_LAYERS[i] for i in range(2, N_LAYERS, 3)]
X_TICKLABELS = ["输入"] + [f"第{i}层" for i in range(2, N_LAYERS, 3)]
X_LABEL_AT = N_LAYERS + 0.9  # 线尾标签 x


def load():
    with open(SRC) as fh:
        return json.load(fh)


def series_values(L, group, metric, transform):
    """返回 (x, y) 两个数组，含「输入」点（动作 token 无）。"""
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
            ys.append(transform(l0[metric]))
    for i, it in enumerate(items):
        xs.append(X_LAYERS[i])
        ys.append(transform(it[metric]))
    return np.array(xs, float), np.array(ys, float)


def spread_labels(ends, ylim, min_gap, obstacles=()):
    """把线尾标签在 y 方向上推开，避免互相重叠，也避开参考线（obstacles 为 y 值）。"""
    order = sorted(range(len(ends)), key=lambda i: ends[i][1])
    ys = [ends[i][1] for i in order]
    lo, hi = ylim
    keep = min_gap * 0.55   # 与参考线的最小距离

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
        for k in range(1, len(ys)):            # 从下往上推
            if ys[k] - ys[k - 1] < min_gap:
                ys[k] = ys[k - 1] + min_gap
        over = ys[-1] - (hi - min_gap * 0.9)
        if over > 0:
            ys = [y - over for y in ys]
        for k in range(len(ys) - 2, -1, -1):   # 从上往下推
            if ys[k + 1] - ys[k] < min_gap:
                ys[k] = ys[k + 1] - min_gap
        ys = avoid(ys)
    for k in range(1, len(ys)):                # 收尾：先保证标签不互相重叠，再避开参考线
        if ys[k] - ys[k - 1] < min_gap:
            ys[k] = ys[k - 1] + min_gap
        for ob in obstacles:
            if abs(ys[k] - ob) < keep:
                ys[k] = ob + keep
    return {ends[i][0]: ys[k] for k, i in enumerate(order)}


def draw_panel(ax, L, metric, transform, ylim, label_gap, obstacles=()):
    ends = []
    for name, (color, lw, alpha, _) in SERIES.items():
        sv = series_values(L, name, metric, transform)
        if sv is None:
            continue
        x, y = sv
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle="round",
                marker="o", ms=3.4 if lw < 2 else 4.2, mfc=color, mec=SURFACE,
                mew=0.6, zorder=3)
        ends.append((name, y[-1]))
    # 线尾直接标注，错开并用引导线
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


def common_layout(fig, axes, xlabel, ylabel, title, footnote):
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, color=INK2, fontsize=12)
    for row in axes:
        row[0].set_ylabel(ylabel, color=INK2, fontsize=12)
    fig.suptitle(title, fontsize=17, fontweight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.012, footnote, ha="center", va="bottom", fontsize=11,
             color=MUTED)


FOOT = "线性探针，按局 5 折的折外结果。差值没有区间，小差异不要当结论；各组取了组平均。"
XLABEL = "模型层（越往右越靠近输出）"


def fig_mae(p):
    # 先算全局 y 范围
    vals = []
    for m, _ in MODELS:
        for lab, _ in LABELS:
            L = p["models"][m]["labels"][lab]
            tb = L["time_baseline"]["mae"]
            for name in SERIES:
                sv = series_values(L, name, "mae", lambda v, tb=tb: tb - v)
                if sv is not None:
                    vals.extend(sv[1])
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08
    ylim = (lo - pad * 2.0, hi + pad * 2.2)
    label_gap = (ylim[1] - ylim[0]) / 16

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2), dpi=170,
                             facecolor=SURFACE)
    for r, (lab, lab_name) in enumerate(LABELS):
        for c, (m, m_name) in enumerate(MODELS):
            ax = axes[r][c]
            L = p["models"][m]["labels"][lab]
            tb = L["time_baseline"]["mae"]
            ax.axhspan(0, ylim[1], color=GREEN_BG, zorder=0)
            ax.axhspan(ylim[0], 0, color=GREY_BG, zorder=0)
            draw_panel(ax, L, "mae", lambda v, tb=tb: tb - v, ylim, label_gap,
                       obstacles=(0.0,))
            ax.axhline(0, color=INK, lw=2.2, zorder=2)
            ax.text(0.985, 0.97, "更好", transform=ax.transAxes, ha="right",
                    va="top", fontsize=12, color="#2e7d32", fontweight="bold")
            ax.text(0.985, 0.03, "更差", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=12, color=INK2, fontweight="bold")
            ax.set_title(f"{m_name}：{lab_name}\n", fontsize=13.5,
                         fontweight="bold", color=INK, loc="left", pad=6)
            ax.text(0.0, 1.015, f"黑线 = 只用时间和次数，平均错 {tb:.1f} 步",
                    transform=ax.transAxes, fontsize=11, color=INK2,
                    ha="left", va="bottom")
    common_layout(fig, axes, XLABEL, "比只用时间和次数少错几步（越高越好）",
                  "哪一层的隐状态比『只用时间和次数』更能读出还有几步", FOOT)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955), h_pad=2.2, w_pad=1.6)
    fig.savefig(OUT_MAE, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def fig_r2(p):
    ylim = (-0.2, 1.0)
    label_gap = (ylim[1] - ylim[0]) / 16
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2), dpi=170,
                             facecolor=SURFACE)
    for r, (lab, lab_name) in enumerate(LABELS):
        for c, (m, m_name) in enumerate(MODELS):
            ax = axes[r][c]
            L = p["models"][m]["labels"][lab]
            tb = L["time_baseline"]["r2"]
            draw_panel(ax, L, "r2", lambda v: v, ylim, label_gap,
                       obstacles=(tb, 0.0))
            ax.axhline(tb, color=INK, lw=2.2, ls=(0, (6, 3)), zorder=2)
            ax.axhline(0, color="#a8a69f", lw=1.0, zorder=1)
            ax.set_title(f"{m_name}：{lab_name}\n", fontsize=13.5,
                         fontweight="bold", color=INK, loc="left", pad=6)
            ax.text(0.0, 1.015,
                    f"黑虚线 = 只用时间和次数（R² {tb:.2f}）；灰线 = 和瞎猜一样",
                    transform=ax.transAxes, fontsize=11, color=INK2,
                    ha="left", va="bottom")
            ax.set_yticks(np.arange(-0.2, 1.01, 0.2))
    common_layout(fig, axes, XLABEL, "R²（1 = 全猜对，0 = 和瞎猜一样）",
                  "每一层能读出多少进度（R²）", FOOT)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955), h_pad=2.2, w_pad=1.6)
    fig.savefig(OUT_R2, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = load()
    # 字体自检
    fp = font_manager.findfont(font_manager.FontProperties(family=FONT))
    print("font:", fp)
    if "NotoSansCJK" not in fp:
        print("警告：未找到 Noto Sans CJK SC", file=sys.stderr)
    # 三个核对数值
    Lm4 = p["models"]["mot"]["labels"]["q4"]
    Lc3 = p["models"]["ctx"]["labels"]["q3"]
    print("核对 1 motion 模型/离正确窗口/文字 第0层 MAE 差 =",
          round(Lm4["time_baseline"]["mae"] - Lm4["layers"]["prefix"]["文字"][0]["mae"], 1))
    print("核对 2 基线模型/离自己按/动作 token 第4层 MAE 差 =",
          round(Lc3["time_baseline"]["mae"] - Lc3["layers"]["action"][4]["mae"], 1))
    print("核对 3 motion 模型/离正确窗口/motion 输入处 MAE 差 =",
          round(Lm4["time_baseline"]["mae"] - Lm4["layers"]["prefix_layer0"]["motion"]["mae"], 1))
    fig_mae(p)
    fig_r2(p)
    print("written:", OUT_MAE, OUT_R2)


if __name__ == "__main__":
    main()
