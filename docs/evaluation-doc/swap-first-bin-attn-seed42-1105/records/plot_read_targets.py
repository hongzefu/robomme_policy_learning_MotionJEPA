"""把 read-targets.json 和 current-frame-fusion.json 画成三张图：谁读谁的流量、重点读哪段、逐层热力图。

用法：plot_read_targets.py <records 目录>
"""
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import matplotlib.pyplot as plt
import numpy as np

matplotlib.font_manager.fontManager.addfont(subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True))
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
TASKS = {"ButtonUnmaskSwap": "Button", "VideoUnmaskSwap": "Video"}
EVENTS = {"reveal": "开局\n[0,32)", "cover": "盖住\n[32,64)", "swap": "交换", "after": "其后"}
QUERIES = {"text": ("文字", "#2a78d6"), "image_2hop": ("当前图像（经文字两跳估计）", "#eb6834"), "action": ("动作 token", "#1baf7a")}
STREAMS = {"frame": "历史帧", "motion": "motion 窗"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
CMAP = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b3ac")
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def flow_figure(fusion, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharex=True, sharey=True, facecolor=SURFACE, constrained_layout=True)
    rows = (("text", "frame", "文字 → 历史帧", "#2a78d6", "-"), ("text", "motion", "文字 → motion", "#eb6834", "-"), ("image_base", "text", "第三人称图像 → 文字", "#1baf7a", "-"),
            ("image_wrist", "text", "手腕图像 → 文字", "#1baf7a", "--"))
    for r, model in enumerate(("motion", "baseline")):
        for c, task in enumerate(TASKS):
            ax = axes[r, c]
            style(ax)
            cell = fusion["cells"][f"{model}|{task}"]
            for q, k, label, color, ls in rows:
                if model == "baseline" and k == "motion":
                    continue
                stat = cell[q][k]
                ax.fill_between(range(18), stat["lo"], stat["hi"], color=color, alpha=.14, lw=0)
                ax.plot(range(18), stat["mean"], color=color, ls=ls, lw=2, marker="o" if ls == "-" else None, ms=4, label=label)
            ax.plot(range(18), cell["image_base"]["frame"]["mean"], color=INK, lw=1.4, ls=":", label="图像 → 历史帧 / motion（恒为 0）")
            ax.set_title(f"{'motion 模型' if model == 'motion' else '基线'} · {TASKS[task]} · {cell['episodes']} 局", color=INK, fontsize=12)
            ax.set_xticks(range(18))
            ax.set_ylim(0, 100)
            ax.grid(axis="y", color="#e8e7e3", lw=.8)
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel("query 这一组分到的注意力（%）", color=MUTED)
            if r == 1:
                ax.set_xlabel("层", color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=10, labelcolor=MUTED)
    fig.suptitle("信息流的两段：记忆 → 文字（第 0、1 层）→ 当前图像（深层）", color=INK, fontsize=14)
    fig.savefig(out / "fig_fusion_flow.png", dpi=130)


def share_figure(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True, facecolor=SURFACE, constrained_layout=True)
    width = .25
    for r, stream in enumerate(STREAMS):
        for c, task in enumerate(TASKS):
            ax = axes[r, c]
            style(ax)
            cell = data["cells"][f"motion|{task}"]["all"]
            for i, (q, (label, color)) in enumerate(QUERIES.items()):
                x = [j + (i - 1) * (width + .02) for j in range(len(EVENTS))]
                values = [cell["events"][q][stream][e]["share_of_stream"]["mean"] for e in EVENTS]
                bars = ax.bar(x, values, width=width, color=color, label=label)
                for b, v in zip(bars, values):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", fontsize=8.5, color=INK, zorder=5,
                            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": .6})
            for j, e in enumerate(EVENTS):
                u = cell["uniform"][f"{stream}|{e}"] * 100
                ax.plot([j - .42, j + .42], [u, u], color=INK, lw=2, zorder=3, label="均匀水平（token 数占比）" if j == 0 and i == 2 else None)
            ax.set_xticks(range(len(EVENTS)), EVENTS.values())
            ax.set_ylim(0, 100)
            ax.grid(axis="y", color="#e8e7e3", lw=.8)
            ax.set_axisbelow(True)
            ax.set_title(f"{TASKS[task]} · 读{STREAMS[stream]}的注意力分布（{cell['episodes']} 局）", color=INK, fontsize=12)
            if c == 0:
                ax.set_ylabel("占读这一流总量的比例（%）", color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=10, labelcolor=MUTED)
    fig.suptitle("motion 模型：文字、当前图像、动作重点读哪段历史（闭合前 4 次推理合并，全部层）", color=INK, fontsize=14)
    fig.savefig(out / "fig_read_targets_share.png", dpi=130)


def layer_figure(data, out):
    fig, axes = plt.subplots(3, 4, figsize=(15, 11), sharex=True, sharey=True, facecolor=SURFACE, constrained_layout=True)
    for r, (q, (label, _)) in enumerate(QUERIES.items()):
        matrices = {(task, s): np.stack([data["cells"][f"motion|{task}"]["all"]["events"][q][s][e]["by_layer"] for e in EVENTS], 1)
                    for task in TASKS for s in STREAMS}
        vmax = max(m.max() for m in matrices.values())
        norm = PowerNorm(.5, vmin=0, vmax=vmax)
        for c, ((task, s), matrix) in enumerate(matrices.items()):
            ax = axes[r, c]
            style(ax)
            image = ax.imshow(matrix, cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest")
            ax.set_xticks(range(len(EVENTS)), [v.split("\n")[0] for v in EVENTS.values()])
            ax.set_yticks(range(18))
            ax.set_title(f"{label if c == 0 else ''}{chr(10) if c == 0 else chr(10)}{TASKS[task]} · {STREAMS[s]}", color=INK, fontsize=10)
            for layer, column in zip(*np.where(matrix >= .35 * vmax)):
                ax.text(column, layer, f"{matrix[layer, column]:.1f}", ha="center", va="center", fontsize=8, color="white" if norm(matrix[layer, column]) > .55 else INK)
            if c == 0:
                ax.set_ylabel("层", color=MUTED)
        fig.colorbar(image, ax=axes[r].tolist(), shrink=.85, label="占这一层该 query 注意力（%，平方根色标）")
    fig.suptitle("motion 模型逐层读取：每行色标独立，只比较同一行内的格子", color=INK, fontsize=14)
    fig.savefig(out / "fig_read_targets_layers.png", dpi=130)


def main(records):
    data = json.loads((records / "read-targets.json").read_text(encoding="utf-8"))
    fusion = json.loads((records / "current-frame-fusion.json").read_text(encoding="utf-8"))
    flow_figure(fusion, records)
    share_figure(data, records)
    layer_figure(data, records)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
