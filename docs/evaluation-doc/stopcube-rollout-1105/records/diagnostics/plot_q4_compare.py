#!/usr/bin/env python
"""StopCube 真实 rollout 线性探针，只看 q4（离正确窗口还有几步）的两张 PNG。

图 1 rollout-r2-layers-q4.png：每一层的 R²，1 行 2 列（基线模型 / motion 模型）。
图 2 rollout-r2-fit-q4-compare.png：第 17 层，两个模型叠在同一格对比，2 行 3 列。
字体、配色、数据读取、加权拟合沿用同目录 plot_r2_complete.py。只读数据，只写产出目录。
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.lines import Line2D

sys.path.insert(0, "/tmp/claude-114466650/-data-hongzefu-robomme-policy-learning-MotionJEPA/"
                   "e524da50-0b7d-47ff-ad49-563d85174b52/scratchpad")
import plot_r2_complete as base  # noqa: E402  字体注册、配色、读数函数

OUT_DIR = base.OUT_DIR
OUT_LAYERS = f"{OUT_DIR}/rollout-r2-layers-q4.png"
OUT_FIT = f"{OUT_DIR}/rollout-r2-fit-q4-compare.png"

INK, INK2, MUTED, GRID, SURFACE = base.INK, base.INK2, base.MUTED, base.GRID, base.SURFACE
LAB = "q4"
LAB_NAME = "离正确窗口还有几步"
MODEL_COLOR = {"ctx": "#2a78d6", "mot": "#eb6834"}
MODEL_NAME = {"ctx": "基线模型", "mot": "motion 模型"}


# =====================================================================
# 图 1：每一层的 R²（只有 q4），1 行 2 列
# =====================================================================
def fig_layers(p, fit):
    ylim = (-0.2, 1.0)
    label_gap = (ylim[1] - ylim[0]) / 16
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), dpi=170, facecolor=SURFACE)
    for ax, (m, m_name) in zip(axes, base.MODELS):
        L = p["models"][m]["labels"][LAB]
        tb_n = L["time_baseline"]["r2"]      # 时间 + 次数
        tb_t = fit[m][LAB]["time"]["r2"]      # 只用时间
        base.draw_layer_panel(ax, L, ylim, label_gap, obstacles=(tb_n, tb_t, 0.0))
        ax.axhline(tb_n, color=INK, lw=2.2, ls=(0, (6, 3)), zorder=2)
        ax.axhline(tb_t, color=INK, lw=1.1, ls=(0, (1.5, 2.5)), zorder=2)
        ax.axhline(0, color="#a8a69f", lw=1.0, zorder=1)
        ax.set_title(f"{m_name}：{LAB_NAME}\n", fontsize=13.5, fontweight="bold",
                     color=INK, loc="left", pad=6)
        ax.text(0.0, 1.015,
                f"粗虚线 = 时间和次数（R² {tb_n:.2f}）；细点线 = 只用时间（R² {tb_t:.2f}）；"
                f"灰线 = 和瞎猜一样",
                transform=ax.transAxes, fontsize=10.5, color=INK2, ha="left", va="bottom")
        ax.set_yticks(np.arange(-0.2, 1.01, 0.2))
        ax.set_xlabel(base.XLABEL1, color=INK2, fontsize=12)
    axes[0].set_ylabel("R²（1 = 全猜对，0 = 和瞎猜一样）", color=INK2, fontsize=12)
    fig.suptitle("每一层能读出多少进度：离正确窗口（R²）", fontsize=17, fontweight="bold",
                 color=INK, y=0.985)
    fig.text(0.5, 0.012, "线性探针，真实 rollout（基线 50 局，motion 49 局），按局 5 折的折外结果。差值没有区间，小差异不要当结论；各组取了组平均。", ha="center", va="bottom", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94), w_pad=1.6)
    fig.savefig(OUT_LAYERS, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# =====================================================================
# 图 2：第 17 层，两个模型叠在同一格里对比
# =====================================================================
# (组名, 读数键)
PANELS = [
    ("动作 token", "action"),
    ("文字", "文字"),
    ("当前画面", "当前视觉"),
    ("历史帧", "历史视觉"),
    ("motion", "motion"),
    ("只用时间和次数", "time_N"),
]
FOOT2A = ("R² 是折外结果，1 = 全猜对，0 = 和只猜平均值一样，可为负；"
          "斜率 1 = 不偏，接近 0 = 基本读不出进度。")
FOOT2B = ("线性探针，真实 rollout，按局 5 折。两个模型走出的轨迹不同，决策点不是同一批，"
          "差值没有区间，小差异不要当结论。")


def fig_fit(fit, npz):
    table = []  # (组名, 模型, R², 斜率)
    data = {}
    vals = []
    for m in MODEL_COLOR:
        y = npz[f"{m}/{LAB}/y"].astype(float)
        groups = npz[f"{m}/{LAB}/groups"]
        data[m] = (y, groups)
        vals.append(y)
        for _, key in PANELS:
            k = f"{m}/{LAB}/{key}"
            if k in npz.files:
                v = npz[k].astype(float)
                vals.append(v[np.isfinite(v)])
    allv = np.concatenate(vals)
    lo, hi = allv.min(), allv.max()
    pad = (hi - lo) * 0.05
    lim = (lo - pad, hi + pad)

    # 等比例格子不配合 tight_layout，改为显式布局：每格贴着所在槽位的顶部（紧挨标题），
    # 槽位比格子高出一截留给刻度与轴名，避免下一行标题压到上一行刻度。
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 12.6), dpi=170, facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.85, bottom=0.09,
                        hspace=0.34, wspace=0.16)
    for idx, (name, key) in enumerate(PANELS):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor(SURFACE)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("N")
        ax.grid(color=GRID, lw=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=INK2, labelsize=10.5, length=3)
        ax.plot(lim, lim, color="#a8a69f", lw=1.0, zorder=1)

        title_lines = []
        for m in ("ctx", "mot"):
            k = f"{m}/{LAB}/{key}"
            if k not in npz.files:
                continue
            y, groups = data[m]
            pred = npz[k].astype(float)
            ok = np.isfinite(pred) & np.isfinite(y)
            xx, yy, gg = y[ok], pred[ok], groups[ok]
            color = MODEL_COLOR[m]
            ax.scatter(xx, yy, s=10, color=color, alpha=0.2, linewidths=0, zorder=2)
            a, b = base.wls_line(xx, yy, gg)
            xs = np.array([xx.min(), xx.max()])
            ax.plot(xs, a + b * xs, color=color, lw=3.0, solid_capstyle="round", zorder=4,
                    path_effects=[pe.Stroke(linewidth=5.2, foreground=SURFACE), pe.Normal()])
            r2 = fit[m][LAB][key]["r2"]
            short = "基线" if m == "ctx" else "motion"
            title_lines.append((f"{short} R² {r2:.2f}，斜率 {b:.2f}", color))
            table.append((name, MODEL_NAME[m], r2, b))

        # 标题：第一行组名（黑），后面每行按模型颜色写
        ax.set_title(name, fontsize=13, fontweight="bold", color=INK, pad=4 + 17 * len(title_lines))
        for i, (txt, color) in enumerate(title_lines):
            ax.text(0.5, 1.012 + 0.052 * (len(title_lines) - 1 - i), txt, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=color)
        if key == "motion":
            ax.text(0.04, 0.96, "基线模型没有这一组", transform=ax.transAxes,
                    ha="left", va="top", fontsize=11, color=MUTED, zorder=5)
            table.append((name, MODEL_NAME["ctx"], None, None))
        if idx // 3 == 1:
            ax.set_xlabel("真实还剩几步", color=INK2, fontsize=11.5)
        if idx % 3 == 0:
            ax.set_ylabel("探针猜还剩几步", color=INK2, fontsize=11.5)

    # 顶部图例
    n_c = fit["ctx"][LAB]
    n_m = fit["mot"][LAB]
    handles = [
        Line2D([], [], color=MODEL_COLOR["ctx"], marker="o", ms=7, lw=0, alpha=0.75,
               label=f"基线模型（{n_c['n_episodes']} 局，{n_c['n_samples']} 个决策点）"),
        Line2D([], [], color=MODEL_COLOR["mot"], marker="o", ms=7, lw=0, alpha=0.75,
               label=f"motion 模型（{n_m['n_episodes']} 局，{n_m['n_samples']} 个决策点）"),
        Line2D([], [], color="#a8a69f", lw=1.2, label="猜对的位置"),
        Line2D([], [], color=INK, lw=3.0, label="拟合直线"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.962), fontsize=12, handlelength=2.2,
               columnspacing=2.0, handletextpad=0.6)
    fig.suptitle("第 17 层：两个模型猜『离正确窗口还有几步』的对比", fontsize=17.5,
                 fontweight="bold", color=INK, y=0.99)
    fig.text(0.5, 0.028, FOOT2A, ha="center", va="bottom", fontsize=11, color=MUTED)
    fig.text(0.5, 0.008, FOOT2B, ha="center", va="bottom", fontsize=11, color=MUTED)
    fig.savefig(OUT_FIT, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return table


def main():
    p = base.load_json(base.SRC_PROBE)
    fit = base.load_json(base.SRC_FIT)
    npz = np.load(base.SRC_NPZ, allow_pickle=False)
    fp = font_manager.findfont(font_manager.FontProperties(family=base.FONT))
    print("font:", fp)
    if "NotoSansCJK" not in fp:
        print("警告：未找到 Noto Sans CJK SC", file=sys.stderr)
    fig_layers(p, fit)
    table = fig_fit(fit, npz)
    print("\n图 2 每格（组 | 模型 | R² | 斜率）")
    for name, m_name, r2, b in table:
        if r2 is None:
            print(f"{name} | {m_name} | 没有这一组 | —")
        else:
            print(f"{name} | {m_name} | {r2:.2f} | {b:.2f}")
    print("\nwritten:", OUT_LAYERS, OUT_FIT)


if __name__ == "__main__":
    main()
