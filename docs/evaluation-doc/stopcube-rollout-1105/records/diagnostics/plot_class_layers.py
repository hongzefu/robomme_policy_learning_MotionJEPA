#!/usr/bin/env python
"""StopCube 真实 rollout：把回归探针换算成分类后，每一层的准确率（4 行 × 2 列折线图）。

数据只读：class-layers.json。输出 rollout-class-layers.png 与抽出的两份 Noto Sans CJK SC 字体文件。
用法：uv run --no-sync python plot_class_layers.py [数据 json 路径] [输出目录]
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = "/data/hongzefu/robomme_policy_learning_MotionJEPA"
SRC_DEFAULT = f"{REPO}/v1-store/evaluation/stopcube-rollout-1105/analysis/fit-data/class-layers.json"
OUT_DIR_DEFAULT = ("/tmp/claude-114466650/-data-hongzefu-robomme-policy-learning-MotionJEPA/"
                   "e524da50-0b7d-47ff-ad49-563d85174b52/scratchpad/class-layers-plot")
SRC = sys.argv[1] if len(sys.argv) > 1 else SRC_DEFAULT
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else OUT_DIR_DEFAULT
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "rollout-class-layers.png")

# ---------- 字体（沿用 plot_r2_complete.py：从 Noto ttc 抽 SC 面）----------
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
    # 图上组名：(颜色, 线宽, 透明度, 数据里的读数名)
    "动作 token": ("#2a78d6", 2.6, 1.0, "action"),
    "文字":       ("#eb6834", 2.6, 1.0, "文字"),
    "当前画面":   ("#1baf7a", 1.8, 1.0, "当前视觉"),
    "历史帧":     ("#eda100", 1.4, 0.75, "历史视觉"),
    "motion":     ("#e87ba4", 1.4, 0.75, "motion"),
}

MODELS = [("ctx", "基线模型"), ("mot", "motion 模型")]
# 行：(问题键, 问题名, 指标键, 指标名)
ROWS = [
    ("press", "该不该按", "pair", "配对准确率"),
    ("c",     "已过次数", "pair", "配对准确率"),
    ("press", "该不该按", "point", "逐点准确率"),
    ("c",     "已过次数", "point", "逐点准确率"),
]
N_LAYERS = 18
X_INPUT = 0
X_LAYERS = np.arange(1, N_LAYERS + 1)
X_TICKS = [X_INPUT] + [X_LAYERS[i] for i in range(2, N_LAYERS, 3)]
X_TICKLABELS = ["输入"] + [f"第{i}层" for i in range(2, N_LAYERS, 3)]
X_LABEL_AT = N_LAYERS + 0.9
YLIM = (0.0, 1.0)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def series_values(readers, group, q, metric):
    """返回 (x, y)：x=0 为输入（层号 -1），x=k+1 为第 k 层；没有这组时返回 None。"""
    key = SERIES[group][3]
    layers = readers.get(key)
    if not layers:
        return None
    xs, ys = [], []
    for lk in sorted(layers.keys(), key=int):
        xs.append(int(lk) + 1)
        ys.append(layers[lk][q][metric]["mean"])
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


def draw_panel(ax, readers, q, metric, label_gap, obstacles=()):
    ends = []
    for name, (color, lw, alpha, _) in SERIES.items():
        sv = series_values(readers, name, q, metric)
        if sv is None:
            continue
        x, y = sv
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle="round",
                marker="o", ms=3.4 if lw < 2 else 4.2, mfc=color, mec=SURFACE,
                mew=0.6, zorder=3)
        ends.append((name, y[-1]))
    pos = spread_labels(ends, YLIM, label_gap, obstacles)
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
    ax.set_ylim(*YLIM)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_TICKLABELS)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.tick_params(colors=INK2, labelsize=11, length=3)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.set_facecolor(SURFACE)


FOOT = ("第 17 层以外同样是『离正确窗口还有几步』的回归，训练时不知道配对和分类目标；预测再用真实速度和次数换算。\n"
        "配对 = 同一局内方块差≤2 cm、TCP 差≤1 cm 且答案不同，两点都对才算对。"
        "按局 5 折折外结果，折线不含区间，小差异不要当结论。")
XLABEL = "模型层（越往右越靠近输出）"


def main():
    d = load_json(SRC)
    fp = font_manager.findfont(font_manager.FontProperties(family=FONT))
    print("font:", fp)
    if "NotoSansCJK" not in fp:
        print("警告：未找到 Noto Sans CJK SC", file=sys.stderr)

    label_gap = (YLIM[1] - YLIM[0]) / 15
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(15.5, 19.0), dpi=170,
                             facecolor=SURFACE, sharex=True)
    for r, (q, q_name, metric, metric_name) in enumerate(ROWS):
        for c, (m, m_name) in enumerate(MODELS):
            ax = axes[r][c]
            M = d[m]
            tb = M["time_N"][q][metric]["mean"]
            draw_panel(ax, M["readers"], q, metric, label_gap, obstacles=(tb,))
            # 横虚线只画到数据区右端，不伸进右侧标签区，免得压住线尾标签
            ax.plot([-0.6, N_LAYERS + 0.6], [tb, tb], color=INK, lw=2.2,
                    ls=(0, (6, 3)), zorder=2, solid_capstyle="butt")
            if metric == "pair":
                pr = M["pairs"][q]
                title = f"{m_name}：{q_name}·{metric_name}（{pr['n']} 对 / {pr['episodes']} 局）"
            else:
                title = f"{m_name}：{q_name}·{metric_name}"
            ax.set_title(title + "\n", fontsize=13.5, fontweight="bold",
                         color=INK, loc="left", pad=6)
            ax.text(0.0, 1.015, f"粗虚线 = 只用时间和次数（准确率 {tb:.2f}）",
                    transform=ax.transAxes, fontsize=10.5, color=INK2,
                    ha="left", va="bottom")
            print(f"{m_name} | {q_name}·{metric_name} | 只用时间和次数 = {tb:.4f}")
    for ax in axes[-1]:
        ax.set_xlabel(XLABEL, color=INK2, fontsize=12)
    for row in axes:
        row[0].set_ylabel("准确率（1 = 全对）", color=INK2, fontsize=12)
    fig.suptitle("把回归换算成分类：每一层的准确率", fontsize=17, fontweight="bold",
                 color=INK, y=0.988)
    fig.text(0.5, 0.008, FOOT, ha="center", va="bottom", fontsize=11, color=MUTED,
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97), h_pad=2.4, w_pad=1.6)
    fig.savefig(OUT_PNG, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print("written:", OUT_PNG)


if __name__ == "__main__":
    main()
