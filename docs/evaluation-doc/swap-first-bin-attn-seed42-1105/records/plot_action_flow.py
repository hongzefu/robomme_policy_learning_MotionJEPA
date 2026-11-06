"""全链路注意力三张图。输入 action-flow-summary.json 与逐局结果目录。

fig_chain_carry：开局帧、交换帧、交换 motion 的累积比例随层进入各组（alpha=0.5，属注意力近似）。
fig_chain_final：动作的最终来源、记忆来源内部占比与中转组。
fig_event_heads：专门读这两个事件的层 × 头，格子无插值。

用法：plot_action_flow.py <records 目录> <action-flow 目录>
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
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CMAP = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b3ac")
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def carry_figure(s, out):
    g = {n: i for i, n in enumerate(s["groups"])}
    sources = (("frame_reveal", "开局帧"), ("frame_swap", "交换段帧"), ("motion_swap", "交换段 motion 窗"))
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey="row", facecolor=SURFACE, constrained_layout=True)
    for r, (src, name) in enumerate(sources):
        for c, task in enumerate(TASKS):
            ax = axes[r, c]
            style(ax)
            carry = np.array(s["cells"][f"motion|{task}"]["alpha_0.5"]["carry_by_layer"])
            lines = {"其后历史帧 token": (carry[:, g["frame_after"], g[src]], BLUE), "文字 token": (carry[:, g["text"], g[src]], ORANGE),
                     "当前图像 token": ((carry[:, g["image_base"], g[src]] + carry[:, g["image_wrist"], g[src]]) / 2, AQUA), "动作 token": (carry[:, g["action"], g[src]], YELLOW)}
            for label, (values, color) in lines.items():
                ax.plot(range(19), values, color=color, lw=2, marker="o", ms=4, label=label)
                if label in ("其后历史帧 token", "动作 token"):  # 文字、图像的末端和动作几乎重合，只标这两条
                    ax.annotate(f"{values[-1]:.1f}", (18, values[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
            ax.set_xlim(0, 19)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            ax.set_ylim(0, max(l.get_ydata().max() for a in axes[r, :c + 1] for l in a.get_lines()) * 1.08)
            if c == 1:
                axes[r, 0].set_ylim(ax.get_ylim())
            ax.grid(axis="y", color="#e8e7e3", lw=.8)
            ax.set_axisbelow(True)
            ax.set_title(f"{TASKS[task]} · 来源：{name}", color=INK, fontsize=12)
            if c == 0:
                ax.set_ylabel("该组 token 含这一来源（%）", color=MUTED)
            if r == 2:
                ax.set_xlabel("经过的层数（0 为输入）", color=MUTED)
                ax.set_xticks(range(0, 19, 2))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=10, labelcolor=MUTED)
    fig.suptitle("motion 模型：这三类信息一路进入各组 token（注意力累积估计，残差比例 0.5）", color=INK, fontsize=14)
    fig.savefig(out / "fig_chain_carry.png", dpi=130)


def final_figure(s, out):
    g = {n: i for i, n in enumerate(s["groups"])}
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), facecolor=SURFACE, constrained_layout=True)
    cells = (("motion", "ButtonUnmaskSwap", "motion · Button"), ("motion", "VideoUnmaskSwap", "motion · Video"), ("baseline", "ButtonUnmaskSwap", "基线 · Button"), ("baseline", "VideoUnmaskSwap", "基线 · Video"))
    parts = (("文字输入", ["text"], ORANGE), ("手腕图像", ["image_wrist"], AQUA), ("第三人称图像", ["image_base"], YELLOW),
             ("历史帧", [f"frame_{e}" for e in ("reveal", "cover", "swap", "after")], BLUE), ("motion", [f"motion_{e}" for e in ("reveal", "cover", "swap", "after")], MAGENTA))
    ax = axes[0]
    style(ax)
    for i, (m, t, label) in enumerate(cells):
        final = s["cells"][f"{m}|{t}"]["alpha_0.5"]["action_final"]
        left = 0
        for name, members, color in parts:
            value = sum(final[n]["mean"] for n in members)
            ax.barh(i, value, left=left, color=color, height=.7, edgecolor=SURFACE, linewidth=2, label=name if i == 0 else None)
            if value > 2.5:
                ax.text(left + value / 2, i, f"{value:.0f}", ha="center", va="center", fontsize=9, color="white" if color in (BLUE,) else INK)
            left += value
    ax.set_yticks(range(4), [c[2] for c in cells])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("动作 token 的最终来源（%）", color=MUTED)
    ax.set_title("动作最终来源", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12))
    ax = axes[1]
    style(ax)
    sources = ("frame_reveal", "frame_swap", "motion_reveal", "motion_swap")
    names = ("开局帧", "交换帧", "开局 motion", "交换 motion")
    width = .2
    colors = (BLUE, ORANGE, AQUA, YELLOW)
    for i, ((m, t, label), color) in enumerate(zip(cells, colors)):
        final = s["cells"][f"{m}|{t}"]["alpha_0.5"]["action_final"]
        low = {"frame_reveal": .38, "frame_swap": .58, "motion_reveal": .15, "motion_swap": .98}
        values = [final[n]["mean"] for n in sources]
        x = [j + (i - 1.5) * width for j in range(4)]
        bars = ax.bar(x, values, width=width * .92, color=color, label=label)
        for b, v in zip(bars, values):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + .04, f"{v:.2f}", ha="center", fontsize=7.5, color=INK, rotation=90, va="bottom")
    ax.set_xticks(range(4), names)
    ax.set_ylim(0, 3.2)
    ax.grid(axis="y", color="#e8e7e3", lw=.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("占动作最终来源（%）", color=MUTED)
    ax.set_title("两个事件到动作的量", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.08))
    ax = axes[2]
    style(ax)
    via = (("历史帧", range(4), BLUE), ("motion", range(4, 8), MAGENTA), ("手腕图像", [9], AQUA), ("第三人称图像", [8], YELLOW), ("文字", [10], ORANGE))
    rows = [("motion|ButtonUnmaskSwap", "frame_reveal", "Button 开局帧"), ("motion|VideoUnmaskSwap", "frame_reveal", "Video 开局帧"),
            ("motion|ButtonUnmaskSwap", "motion_swap", "Button 交换 motion"), ("motion|VideoUnmaskSwap", "motion_swap", "Video 交换 motion"),
            ("motion|ButtonUnmaskSwap", "frame_swap", "Button 交换帧"), ("motion|VideoUnmaskSwap", "frame_swap", "Video 交换帧")]
    for i, (cell, src, label) in enumerate(rows):
        route = s["cells"][cell]["alpha_0.5"]["route_via_source"]
        total = sum(route[v][g[src]] for v in range(11))
        left = 0
        for name, members, color in via:
            value = sum(route[v][g[src]] for v in members) / total * 100
            ax.barh(i, value, left=left, color=color, height=.7, edgecolor=SURFACE, linewidth=2, label=name if i == 0 else None)
            if value > 5:
                ax.text(left + value / 2, i, f"{value:.0f}", ha="center", va="center", fontsize=9, color="white" if color == BLUE else INK)
            left += value
    ax.set_yticks(range(len(rows)), [r[2] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("到动作前最后经过哪组 token（%）", color=MUTED)
    ax.set_title("motion 模型：最后一跳", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12))
    fig.suptitle("注意力累积估计（残差比例 0.5）：只说明注意力路径，不是真实信息量", color=INK, fontsize=14)
    fig.savefig(out / "fig_chain_final.png", dpi=130)


def heads_figure(s, flow, out):
    g = {n: i for i, n in enumerate(s["groups"])}
    index = json.loads((flow / "index.json").read_text(encoding="utf-8"))
    rows = (("text", "frame_reveal", "文字读开局帧"), ("text", "motion_swap", "文字读交换 motion"), ("action", "frame_reveal", "动作读开局帧"), ("action", "motion_swap", "动作读交换 motion"))
    fig, axes = plt.subplots(4, 2, figsize=(11, 18), sharex=True, sharey=True, facecolor=SURFACE, constrained_layout=True)
    for r, (query, key, name) in enumerate(rows):
        matrices = []
        for task in TASKS:
            subset = [e for e in index["episodes"] if e["model"] == "motion" and e["task"] == task]
            arrays = []
            for e in subset:
                with np.load(flow / e["file"]) as z:
                    arrays.append(z["direct"][:, :, g[query], g[key]])
            matrices.append(np.mean(arrays, 0) * 100)
        vmax = max(m.max() for m in matrices)
        norm = PowerNorm(.5, vmin=0, vmax=vmax)
        for c, (task, matrix) in enumerate(zip(TASKS, matrices)):
            ax = axes[r, c]
            style(ax)
            image = ax.imshow(matrix, cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest")
            ax.set_xticks(range(8))
            ax.set_yticks(range(18))
            ax.set_title(f"{TASKS[task]} · {name}", color=INK, fontsize=11)
            order = np.dstack(np.unravel_index(np.argsort(-matrix, axis=None), matrix.shape))[0]
            for layer, head in order[:4]:
                if matrix[layer, head] > .03 * vmax:
                    ax.text(head, layer, f"{matrix[layer, head]:.1f}", ha="center", va="center", fontsize=8.5, color="white" if norm(matrix[layer, head]) > .55 else INK)
            if c == 0:
                ax.set_ylabel("层", color=MUTED)
            if r == 3:
                ax.set_xlabel("头", color=MUTED)
        fig.colorbar(image, ax=axes[r].tolist(), shrink=.85, label="这个头分给这一块（%，平方根色标）")
    fig.suptitle("motion 模型：哪些头在读开局和交换（4 次推理合并）", color=INK, fontsize=14)
    fig.savefig(out / "fig_event_heads.png", dpi=130)


def main(records, flow):
    s = json.loads((records / "action-flow-summary.json").read_text(encoding="utf-8"))
    carry_figure(s, records)
    final_figure(s, records)
    heads_figure(s, flow, records)
    g = {n: i for i, n in enumerate(s["groups"])}
    for cell, c in s["cells"].items():
        for alpha in s["alphas"]:
            final = c[f"alpha_{alpha}"]["action_final"]
            frames = sum(final[f"frame_{e}"]["mean"] for e in ("reveal", "cover", "swap", "after"))
            motion = sum(final[f"motion_{e}"]["mean"] for e in ("reveal", "cover", "swap", "after"))
            print(cell, alpha, f"帧内 开局{final['frame_reveal']['mean'] / frames * 100:.1f}% 交换{final['frame_swap']['mean'] / frames * 100:.1f}%",
                  f"motion内 开局{final['motion_reveal']['mean'] / max(motion, 1e-9) * 100:.1f}% 交换{final['motion_swap']['mean'] / max(motion, 1e-9) * 100:.1f}%",
                  f"合计 帧{frames:.2f} motion{motion:.2f} 文字{final['text']['mean']:.1f} 图像{final['image_base']['mean'] + final['image_wrist']['mean']:.1f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
