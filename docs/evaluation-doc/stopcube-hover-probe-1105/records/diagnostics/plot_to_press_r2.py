#!/usr/bin/env python
"""距按下步数的逐层 R² 图，版式同 stopcube-rollout-1105/records/diagnostics/plot_probe_layers.py 的 fig_r2。

只读 hover-summary.json，不重算。这次没有动作 token；多一条最近 motion token 和一条 [t] 基线。
用法：python plot_to_press_r2.py <hover-summary.json> <输出 PNG> <字体缓存目录>
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fontTools.ttLib import TTFont
from matplotlib import font_manager


SRC, OUT, FONT_DIR = sys.argv[1:4]
FONT = "Noto Sans CJK SC"
# matplotlib 只读 .ttc 的第 0 面（JP），把 SC 面（索引 2）抽成单独 .otf 再注册
os.makedirs(FONT_DIR, exist_ok=True)
for source, name in (("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "NotoSansCJKsc-Regular.otf"),
                     ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "NotoSansCJKsc-Bold.otf")):
    path = os.path.join(FONT_DIR, name)
    if not os.path.exists(path):
        TTFont(source, fontNumber=2).save(path)
    font_manager.fontManager.addfont(path)
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11.5

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
# 组名：(颜色, 线宽, 透明度, 数据键)，颜色与上一次的图相同；最近 motion token 是这次新增的组
SERIES = {
    "文字": ("#eb6834", 2.6, 1.0, "text"),
    "当前画面": ("#1baf7a", 2.2, 1.0, "current"),
    "历史帧": ("#eda100", 1.4, 0.75, "history"),
    "motion": ("#e87ba4", 1.4, 0.75, "motion"),
    "最近 motion token": ("#7a5fd0", 1.4, 0.75, "latest_motion"),
}
MODELS = (("ctx", "基线模型"), ("mot", "motion 模型"))
N_LAYERS = 18
X_LAYERS = np.arange(1, N_LAYERS + 1)
X_LABEL_AT = N_LAYERS + 0.9
X_TICKS = [0] + [X_LAYERS[index] for index in range(2, N_LAYERS, 3)]
X_TICKLABELS = ["输入"] + [f"第{index}层" for index in range(2, N_LAYERS, 3)]


def spread(ends, ylim, gap, obstacles):
    """把线尾标签在 y 方向推开，也避开参考线。"""
    order = sorted(ends, key=lambda item: item[1])
    keep = gap * 0.55
    ys = [min(max(y, ylim[0] + gap), ylim[1] - gap) for _, y in order]
    for _ in range(8):
        for index, y in enumerate(ys):
            for line in obstacles:
                if abs(y - line) < keep:
                    y = line + keep if y >= line else line - keep
            ys[index] = y
        for index in range(1, len(ys)):
            if ys[index] - ys[index - 1] < gap:
                ys[index] = ys[index - 1] + gap
    return {name: y for (name, _), y in zip(order, ys)}


def main():
    summary = json.load(open(SRC))
    t_r2 = summary["time_baseline"]["t"]["to_press"]["r2"]
    t_n_r2 = summary["time_baseline"]["t_N"]["to_press"]["r2"]
    ylim = (-0.2, 1.0)
    gap = (ylim[1] - ylim[0]) / 19
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), dpi=170, facecolor=SURFACE)
    for axis, (model, model_name) in zip(axes, MODELS):
        features = summary["models"][model]["features"]
        ends = []
        for name, (color, width, alpha, key) in SERIES.items():
            if key not in features:
                continue
            item = features[key]
            ys = [item["input_level"]["to_press"]["r2"]] + [layer["to_press"]["r2"] for layer in item["layers"]]
            axis.plot(np.arange(0, N_LAYERS + 1), ys, color=color, lw=width, alpha=alpha, solid_capstyle="round", marker="o",
                      ms=3.4 if width < 2 else 4.2, mfc=color, mec=SURFACE, mew=0.6, zorder=3)
            ends.append((name, ys[-1]))
            print(f"R2 model={model} group={key} input={ys[0]:.3f} L0={ys[1]:.3f} L17={ys[-1]:.3f} "
                  f"max={max(ys):.3f} at={int(np.argmax(ys)) - 1}", flush=True)
        positions = spread(ends, ylim, gap, (t_n_r2, t_r2, 0.0))
        for name, y_end in ends:
            color, width = SERIES[name][:2]
            axis.plot([X_LAYERS[-1] + 0.15, X_LABEL_AT - 0.15], [y_end, positions[name]], color=color, lw=0.8, alpha=0.8, zorder=2)
            axis.text(X_LABEL_AT, positions[name], name, color=color, fontsize=11.5, fontweight="bold" if width > 2 else "normal",
                      va="center", ha="left", zorder=4)
        axis.axhline(t_n_r2, color=INK, lw=2.2, ls=(0, (6, 3)), zorder=2)
        axis.axhline(t_r2, color=INK, lw=1.2, ls=(0, (1, 2)), zorder=2)
        axis.axhline(0, color="#a8a69f", lw=1.0, zorder=1)
        axis.set_title(f"{model_name}：距按下还有几步\n", fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=6)
        axis.text(0.0, 1.015, f"粗虚线 = 时间和次数（R² {t_n_r2:.2f}）；细点线 = 只用时间（R² {t_r2:.2f}）；灰线 = 和瞎猜一样",
                  transform=axis.transAxes, fontsize=10.5, color=INK2, ha="left", va="bottom")
        axis.set_xlim(-0.6, N_LAYERS + 6.2)
        axis.set_ylim(*ylim)
        axis.set_xticks(X_TICKS)
        axis.set_xticklabels(X_TICKLABELS)
        axis.set_yticks(np.arange(-0.2, 1.01, 0.2))
        axis.tick_params(colors=INK2, labelsize=11, length=3)
        axis.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#c3c2b7")
        axis.set_xlabel("模型层（越往右越靠近输出）", color=INK2, fontsize=12)
    axes[0].set_ylabel("R²（1 = 全猜对，0 = 和瞎猜一样）", color=INK2, fontsize=12)
    figure.suptitle("每一层能读出多少进度（R²）", fontsize=17, fontweight="bold", color=INK, y=0.985)
    figure.text(0.5, 0.012, "线性探针，官方示教 98 局，按局 5 折的折外结果。这些局训练时见过；差值没有区间，小差异不要当结论；各组取了组平均。",
                ha="center", va="bottom", fontsize=11, color=MUTED)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94), w_pad=1.6)
    figure.savefig(OUT, dpi=170, facecolor=SURFACE)
    print(f"PLOT={OUT}", flush=True)


if __name__ == "__main__":
    main()
