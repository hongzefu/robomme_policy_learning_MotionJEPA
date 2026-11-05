#!/usr/bin/env python
"""StopCube 真实 rollout：把第 17 层「还剩几步」的回归结果换算成分类，两张面向非技术读者的 PNG。

图 1 rollout-class-accuracy.png：配对 / 逐点准确率，2 行 2 列分组条形图（带 95% 区间）。
图 2 rollout-class-pairs.png：只看「该不该按」，同一对两个点的预测 ratio 用细线连起来，2 行 6 列。
字体、配色沿用同目录 plot_r2_complete.py。数据只读：records/class-derived.json、fit-data/class-derived.npz。
只写产出目录。
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, "/tmp/claude-114466650/-data-hongzefu-robomme-policy-learning-MotionJEPA/"
                   "e524da50-0b7d-47ff-ad49-563d85174b52/scratchpad")
import plot_r2_complete as base  # noqa: E402  字体注册、配色

OUT_DIR = base.OUT_DIR
SRC_JSON = f"{base.REPO}/docs/evaluation-doc/stopcube-rollout-1105/records/class-derived.json"
SRC_NPZ = f"{OUT_DIR}/fit-data/class-derived.npz"
OUT_ACC = f"{OUT_DIR}/rollout-class-accuracy.png"
OUT_PAIRS = f"{OUT_DIR}/rollout-class-pairs.png"

INK, INK2, MUTED, GRID, SURFACE = base.INK, base.INK2, base.MUTED, base.GRID, base.SURFACE
MODEL_COLOR = {"ctx": "#2a78d6", "mot": "#eb6834"}
MODEL_NAME = {"ctx": "基线模型", "mot": "motion 模型"}
MODEL_SHORT = {"ctx": "基线", "mot": "motion"}
MODELS = ("ctx", "mot")

# (读数中文名, json/npz 读数键)
READERS = [
    ("动作 token", "action"),
    ("文字", "文字"),
    ("当前画面", "当前视觉"),
    ("历史帧", "历史视觉"),
    ("motion", "motion"),
    ("只用时间和次数", "time_N"),
]
TIME_BG = "#eeedea"      # 「只用时间和次数」的浅灰底
BAND_GREEN = "#d8f0dc"   # 图 2 绿带


# =====================================================================
# 图 1：准确率，2 行 2 列分组条形图
# =====================================================================
# 行：(标签键, 行名)；列：(point|pair, 列名)
ROWS1 = [("press", "该不该按"), ("c", "已过次数 / 剩余次数")]
COLS1 = [("pair", "配对准确率（同一对两个点都判对）"), ("point", "逐点准确率（单个点判对）")]
FOOT1A = ("不是专门训练的分类探针：回归猜还剩几步，再用真实的速度和次数换算成类别，"
          "所以既可能低估，也借用了探针没读到的信息。")
FOOT1B = ("配对：同一局内方块相差不超过 2 cm、手臂相差不超过 1 cm，且标签不同。"
          "误差线是按局重采样的 95% 区间；相关性结果。")


def count_text(cd, lab, kind):
    """格子标题里的样本数：配对数写「N 对 M 局」，逐点写「N 个点 M 局」。"""
    parts = []
    for m in MODELS:
        if kind == "pair":
            p = cd[m]["pairs"][lab]
            parts.append(f"{MODEL_SHORT[m]} {p['n']} 对 {p['episodes']} 局")
        else:
            parts.append(f"{MODEL_SHORT[m]} {cd[m]['points']} 个点 {cd[m]['episodes']} 局")
    return "；".join(parts)


def fig_accuracy(cd):
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.2), dpi=170, facecolor=SURFACE)
    width = 0.36
    n_r = len(READERS)
    for r, (lab, row_name) in enumerate(ROWS1):
        for c, (kind, col_name) in enumerate(COLS1):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            # 「只用时间和次数」浅灰底
            ax.axvspan(n_r - 1 - 0.5, n_r - 1 + 0.5, color=TIME_BG, zorder=0, lw=0)
            ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for i, (name, key) in enumerate(READERS):
                avail = [m for m in MODELS if key in cd[m]["readers"]]
                k = len(avail)
                offsets = [0.0] if k == 1 else [-width / 2, width / 2]
                for m, off in zip(avail, offsets):
                    st = cd[m]["readers"][key][lab][kind]
                    mean = st["mean"]
                    lo, hi = st["ci95"]
                    x = i + off
                    ax.bar(x, mean, width=width * 0.94, color=MODEL_COLOR[m], zorder=2, lw=0)
                    ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], color=INK,
                                lw=1.3, capsize=4, capthick=1.3, zorder=3)
                    ax.text(x, hi + 0.018, f"{mean:.2f}", ha="center", va="bottom",
                            fontsize=10.5, fontweight="bold", color=INK, zorder=4)
            ax.set_xlim(-0.6, n_r - 0.4)
            ax.set_ylim(0, 1.0)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.set_xticks(range(n_r))
            ax.set_xticklabels([n for n, _ in READERS], fontsize=11.5)
            ax.tick_params(colors=INK2, labelsize=11, length=3)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#c3c2b7")
            ax.set_title(f"{row_name}：{col_name}", fontsize=13.5, fontweight="bold",
                         color=INK, loc="left", pad=22)
            ax.text(0.0, 1.015, f"（{count_text(cd, lab, kind)}）", transform=ax.transAxes,
                    fontsize=10.5, color=INK2, ha="left", va="bottom")
            if c == 0:
                ax.set_ylabel("准确率（1 = 全判对）", color=INK2, fontsize=12)
    handles = [
        Patch(color=MODEL_COLOR["ctx"], label="基线模型"),
        Patch(color=MODEL_COLOR["mot"], label="motion 模型"),
        Line2D([], [], color=INK, lw=1.3, marker="_", ms=9, label="95% 区间"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.955), fontsize=12.5, handlelength=1.8, columnspacing=2.4)
    fig.suptitle("第 17 层：把『还剩几步』的回归结果换算成分类", fontsize=17.5,
                 fontweight="bold", color=INK, y=0.99)
    fig.text(0.5, 0.030, FOOT1A, ha="center", va="bottom", fontsize=11, color=MUTED)
    fig.text(0.5, 0.010, FOOT1B, ha="center", va="bottom", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.925), h_pad=3.0, w_pad=1.8)
    fig.savefig(OUT_ACC, dpi=170, facecolor=SURFACE)
    plt.close(fig)


# =====================================================================
# 图 2：同位置两个点分不分得开（该不该按）
# =====================================================================
FOOT2 = ("每条线是同一局里位置几乎相同的两个决策点。分得开时：左端在绿带外，右端在绿带内。"
         "回归结果换算，不是专门训练的分类探针。超出纵轴范围的点画在边界上。")


def fig_pairs(cd, npz):
    # 收集所有要画的点，定统一纵轴：1%～99% 分位并留边
    seg = {}   # (m, key) -> (no_vals, yes_vals)
    allv = []
    for m in MODELS:
        pairs = npz[f"{m}/press_pairs_yes_no"]
        yes_idx, no_idx = pairs[:, 0], pairs[:, 1]
        for _, key in READERS:
            k = f"{m}/ratio/{key}"
            if k not in npz.files:
                continue
            ratio = npz[k].astype(float)
            no_v, yes_v = ratio[no_idx], ratio[yes_idx]
            seg[(m, key)] = (no_v, yes_v)
            allv.append(no_v)
            allv.append(yes_v)
    allv = np.concatenate(allv)
    q1, q99 = np.percentile(allv, [1, 99])
    pad = (q99 - q1) * 0.06
    ylim = (min(q1 - pad, -0.5), q99 + pad)
    clipped = {}
    total_clipped = 0

    fig, axes = plt.subplots(2, 6, figsize=(19.5, 9.4), dpi=165, facecolor=SURFACE)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.83, bottom=0.10, hspace=0.50, wspace=0.14)
    for r, m in enumerate(MODELS):
        color = MODEL_COLOR[m]
        n_pairs = cd[m]["pairs"]["press"]["n"]
        n_eps = cd[m]["pairs"]["press"]["episodes"]
        for c, (name, key) in enumerate(READERS):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            if (m, key) not in seg:
                ax.set_title(f"{name}\n", fontsize=12.5, fontweight="bold", color=INK, pad=6)
                ax.text(0.5, 0.5, "基线没有这一组", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color=MUTED)
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                continue
            no_v, yes_v = seg[(m, key)]
            n_clip = int(np.sum((no_v < ylim[0]) | (no_v > ylim[1]) |
                                (yes_v < ylim[0]) | (yes_v > ylim[1])))
            clipped[(m, key)] = n_clip
            total_clipped += n_clip
            no_p = np.clip(no_v, *ylim)
            yes_p = np.clip(yes_v, *ylim)
            ax.axhspan(0, 1, color=BAND_GREEN, zorder=0, lw=0)
            ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
            ax.set_axisbelow(True)
            for a, b in zip(no_p, yes_p):
                ax.plot([0, 1], [a, b], color=color, lw=1.0, alpha=0.25, zorder=2,
                        solid_capstyle="round")
            ax.scatter(np.zeros_like(no_p), no_p, s=12, color=color, alpha=0.55, lw=0, zorder=3)
            ax.scatter(np.ones_like(yes_p), yes_p, s=12, color=color, alpha=0.55, lw=0, zorder=3)
            if c == 0:
                ax.text(0.5, 0.5, "落在这里 = 判为该按", ha="center", va="center",
                        fontsize=10.5, color="#1f7a3a", fontweight="bold", zorder=5,
                        path_effects=[pe.Stroke(linewidth=3.2, foreground=SURFACE), pe.Normal()])
            ax.set_xlim(-0.35, 1.35)
            ax.set_ylim(*ylim)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["不该按", "该按"], fontsize=11.5, color=INK)
            ax.tick_params(colors=INK2, labelsize=10.5, length=3)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#c3c2b7")
            acc = cd[m]["readers"][key]["press"]["pair"]["mean"]
            ax.set_title(f"{name}\n两边都判对 {acc:.2f}", fontsize=12.5, fontweight="bold",
                         color=INK, pad=6)
            if c > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("预测还差几个周期（步数 ÷ 速度）", color=INK2, fontsize=11)
        ax0 = axes[r][0]
        ax0.text(-0.50, 0.5, f"{MODEL_NAME[m]}\n{n_pairs} 对 / {n_eps} 局",
                 transform=ax0.transAxes, ha="center", va="center", rotation=90,
                 fontsize=13, fontweight="bold", color=color, linespacing=1.6)

    handles = [
        Line2D([], [], color=MODEL_COLOR["ctx"], lw=2.2, marker="o", ms=6, label="基线模型：一对决策点"),
        Line2D([], [], color=MODEL_COLOR["mot"], lw=2.2, marker="o", ms=6, label="motion 模型：一对决策点"),
        Patch(color=BAND_GREEN, label="绿带：预测落在 0～1 个周期内，判为该按"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.935), fontsize=12.5, handlelength=2.2, columnspacing=2.4)
    fig.suptitle("同样位置的两个点，能不能分开：该不该按（第 17 层）", fontsize=17.5,
                 fontweight="bold", color=INK, y=0.99)
    fig.text(0.5, 0.012, FOOT2, ha="center", va="bottom", fontsize=11, color=MUTED)
    fig.savefig(OUT_PAIRS, dpi=165, facecolor=SURFACE)
    plt.close(fig)
    return ylim, clipped, total_clipped


def main():
    cd = base.load_json(SRC_JSON)
    npz = np.load(SRC_NPZ, allow_pickle=False)
    fp = font_manager.findfont(font_manager.FontProperties(family=base.FONT))
    print("font:", fp)
    if "NotoSansCJK" not in fp:
        print("警告：未找到 Noto Sans CJK SC", file=sys.stderr)
    fig_accuracy(cd)
    ylim, clipped, total = fig_pairs(cd, npz)
    print(f"图 2 纵轴范围：{ylim[0]:.2f} ～ {ylim[1]:.2f}")
    for (m, key), n in clipped.items():
        print(f"  夹到边界的对数（任一端超出）{MODEL_NAME[m]} / {key}: {n}")
    print(f"  合计 {total}")
    print("written:", OUT_ACC, OUT_PAIRS)


if __name__ == "__main__":
    main()
