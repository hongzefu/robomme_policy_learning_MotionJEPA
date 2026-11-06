"""motion 模型逐层、逐头注意力：闭合前 4 次推理合并，每局先平均再按局平均。

每个头对全部 token 的注意力和为 1，所以“占这一层注意力”= 该层 8 个头的平均；全局值 = 144 个头的平均。
- fig_layer_attention.png：每层帧流读开局、motion 读交换（有重叠即算，窗 [t,t+32)，演示段截断）；每层 × motion 窗热力图。
- fig_peak_window_heads.png：峰值窗的 18 层 × 8 头热力图，单元格按原始格子显示，无插值。

用法：plot_layer_attention.py <episodes 目录> <输出目录>
"""
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

matplotlib.font_manager.fontManager.addfont(subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True))
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
TASKS = {"ButtonUnmaskSwap": "Button", "VideoUnmaskSwap": "Video"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
CMAP = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])
LAYERS, HEADS = 18, 8
PEAKS = (("ButtonUnmaskSwap", None, 160), ("VideoUnmaskSwap", None, 0), ("VideoUnmaskSwap", 1, 96))


def load(root):
    rows = []
    for path in sorted(glob.glob(f"{root}/with_motion_*/episode_window.json")):
        e = json.loads(Path(path).read_text(encoding="utf-8"))
        swap_end = 64 + 50 * e["swap_times"]
        windows, reveal, swap, reveal_u, swap_u = defaultdict(list), [], [], [], []
        for rec in e["records"]:
            d = rec["exec_start_idx"]
            h = rec["heads"]
            mf = np.nan_to_num(np.asarray(h["m_frame"], float))
            share = np.nan_to_num(np.asarray(h["f_vis_share"], float))
            reveal.append(mf * share)
            frames = rec["streams"]["frame"]
            reveal_u.append(mf * sum(0 <= x["t"] < 32 for x in frames) / len(frames))
            total, hits, mm = np.zeros((LAYERS, HEADS)), 0, np.zeros((LAYERS, HEADS))
            for w in rec["motion_windows"]:
                t = w["t"]
                end = min(t + 32, d) if t < d else t + 32
                mass = np.asarray(w["head_mass"], float)
                windows[t].append(mass)
                mm += mass
                if t < swap_end and end > 64:
                    total += mass
                    hits += 1
            swap.append(total)
            swap_u.append(mm * hits / len(rec["motion_windows"]))
        rows.append({"task": e["task"], "k": e["swap_times"], "key": e["episode_key"], "peak": e["peak"],
                     "windows": {t: np.mean(v, 0) for t, v in windows.items()}, "reveal": np.mean(reveal, 0),
                     "swap": np.mean(swap, 0), "reveal_u": np.mean(reveal_u, 0), "swap_u": np.mean(swap_u, 0)})
    return rows


def boot_layer(values):
    """values: (局, 层)，局级重采样。"""
    picks = np.random.default_rng(42).integers(0, values.shape[0], (2000, values.shape[0]))
    means = values[picks].mean(1)
    return values.mean(0), np.quantile(means, .025, axis=0), np.quantile(means, .975, axis=0)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b3ac")
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def layer_figure(rows, out):
    ks = sorted({r["k"] for r in rows})
    fig = plt.figure(figsize=(14, 15), facecolor=SURFACE, constrained_layout=True)
    grid = fig.add_gridspec(1 + len(ks), 2, height_ratios=[1.05] + [1] * len(ks))
    heat_max = 0
    tables = {}
    for task in TASKS:
        for k in ks:
            sub = [r for r in rows if r["task"] == task and r["k"] == k]
            acc = defaultdict(list)
            for r in sub:
                for t, v in r["windows"].items():
                    acc[t].append(v.mean(1) * 100)
            acc = {t: np.mean(v, 0) for t, v in acc.items() if len(v) >= .5 * len(sub)}
            tables[task, k] = (acc, len(sub))
            heat_max = max(heat_max, max(a.max() for a in acc.values()))
    norm = PowerNorm(.5, vmin=0, vmax=heat_max)
    top_ax = {}
    for col, task in enumerate(TASKS):
        ax = fig.add_subplot(grid[0, col])
        top_ax[col] = ax
        style(ax)
        sub = [r for r in rows if r["task"] == task]
        for key, color, name in (("reveal", BLUE, "帧流读开局"), ("swap", ORANGE, "motion 读交换")):
            values = np.stack([r[key].mean(1) for r in sub]) * 100
            mean, lo, hi = boot_layer(values)
            uniform = np.stack([r[key + "_u"].mean(1) for r in sub]).mean(0) * 100
            ax.fill_between(range(LAYERS), lo, hi, color=color, alpha=.16, lw=0)
            ax.plot(range(LAYERS), uniform, color=color, lw=1.3, ls=(0, (4, 3)), label=f"{name}：均匀水平")
            ax.plot(range(LAYERS), mean, color=color, lw=2, marker="o", ms=5, label=f"{name}：实测")
        ax.set_xticks(range(LAYERS))
        ax.set_xlabel("层", color=MUTED)
        ax.set_ylabel("占这一层注意力（%）", color=MUTED)
        ax.grid(axis="y", color="#e8e7e3", lw=.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 2.1)
        ax.set_title(f"{TASKS[task]} · 每层这两块占多少（{len(sub)} 局）", color=INK, fontsize=12)
        ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper center", ncol=2)
    image = None
    for row, k in enumerate(ks, 1):
        for col, task in enumerate(TASKS):
            acc, n = tables[task, k]
            ax = fig.add_subplot(grid[row, col])
            style(ax)
            starts = sorted(acc)
            matrix = np.stack([acc[t] for t in starts], 1)
            image = ax.imshow(matrix, cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest", origin="upper")
            ax.set_xticks(range(len(starts)), starts, fontsize=8)
            ax.set_yticks(range(0, LAYERS, 2))
            ax.set_ylabel("层", color=MUTED)
            ax.set_title(f"{TASKS[task]} · 交换 {k} 次 · {n} 局", color=INK, fontsize=11, pad=12)
            swap_end = 64 + 50 * k
            for i, t in enumerate(starts):
                color = BLUE if t == 0 else ORANGE if 64 < t + 32 and t < swap_end else None
                if color:
                    ax.add_patch(Rectangle((i - .5, -1.05), 1, .55, clip_on=False, color=color, lw=0))
            for layer, column in zip(*np.where(matrix >= .5 * matrix.max())):
                ax.text(column, layer, f"{matrix[layer, column]:.1f}", ha="center", va="center", fontsize=8, color="white" if norm(matrix[layer, column]) > .55 else INK)
            if row == len(ks):
                ax.set_xlabel("motion 窗起点（环境步）；上方色条：蓝=开局窗，橙=与交换有重叠", color=MUTED)
    fig.colorbar(image, ax=fig.axes[2:], shrink=.5, label="这一窗占这一层注意力（%，颜色按平方根）")
    fig.suptitle("motion 模型逐层注意力（闭合前 4 次推理合并）", color=INK, fontsize=14)
    fig.savefig(out / "fig_layer_attention.png", dpi=130)


def peak_figure(rows, out):
    fig, axes = plt.subplots(1, len(PEAKS), figsize=(14, 7.2), facecolor=SURFACE, constrained_layout=True)
    report = {}
    matrices = []
    for task, k, start in PEAKS:
        sub = [r for r in rows if r["task"] == task and (k is None or r["k"] == k) and start in r["windows"]]
        matrices.append((task, k, start, np.stack([r["windows"][start] for r in sub]) * 100))
    vmax = max(m.mean(0).max() for *_, m in matrices)
    norm = PowerNorm(.5, vmin=0, vmax=vmax)
    for ax, (task, k, start, stack) in zip(axes, matrices):
        style(ax)
        mean = stack.mean(0)
        image = ax.imshow(mean, cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(HEADS))
        ax.set_yticks(range(LAYERS))
        ax.set_xlabel("头", color=MUTED)
        ax.set_ylabel("层", color=MUTED)
        scope = "全部" if k is None else f"交换 {k} 次"
        ax.set_title(f"{TASKS[task]} · {scope} · 第 {start} 窗 · {stack.shape[0]} 局", color=INK, fontsize=12)
        order = np.dstack(np.unravel_index(np.argsort(-mean, axis=None), mean.shape))[0]
        for layer, head in order[:4]:
            ax.text(head, layer, f"{mean[layer, head]:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if mean[layer, head] > .25 * vmax else INK)
        top = [(f"L{l}H{h}", float(mean[l, h])) for l, h in order[:5]]
        report[f"{task}|{k}|{start}"] = {"episodes": stack.shape[0], "window_total_pct": float(mean.mean()), "top_heads_pct": top,
                                         "top_head_share_of_window": float(mean[tuple(order[0])] / mean.sum()),
                                         "by_layer_pct": mean.mean(1).tolist()}
        print(task, scope, start, f"局={stack.shape[0]} 全局{mean.mean():.3f}%", "前5头", [(a, round(b, 1)) for a, b in top],
              f"第一头占该窗全部头之和 {mean[tuple(order[0])] / mean.sum():.0%}", "层前3", [(int(i), round(float(mean.mean(1)[i]), 2)) for i in np.argsort(-mean.mean(1))[:3]])
    fig.colorbar(image, ax=list(axes), shrink=.6, label="这个头分给这一窗的注意力（%，颜色按平方根）")
    fig.suptitle("峰值窗的逐层逐头注意力（每格是一个头；闭合前 4 次推理合并）", color=INK, fontsize=14)
    fig.savefig(out / "fig_peak_window_heads.png", dpi=130)
    return report


def main(root, out):
    rows = load(root)
    layer_figure(rows, out)
    report = {"peak_windows": peak_figure(rows, out), "by_layer": {}}
    for task in TASKS:
        sub = [r for r in rows if r["task"] == task]
        rev, swa = [np.stack([r[n].mean(1) for r in sub]).mean(0) * 100 for n in ("reveal", "swap")]
        ru, su = [np.stack([r[n].mean(1) for r in sub]).mean(0) * 100 for n in ("reveal_u", "swap_u")]
        report["by_layer"][task] = {"reveal": rev.tolist(), "swap": swa.tolist(), "reveal_uniform": ru.tolist(), "swap_uniform": su.tolist()}
        print(task, "开局 按层", np.round(rev, 2).tolist())
        print(task, "开局均匀", np.round(ru, 2).tolist())
        print(task, "交换 按层", np.round(swa, 2).tolist())
        print(task, "交换均匀", np.round(su, 2).tolist())
        print(task, "峰值头计数", Counter(r["peak"]["peak_head"] for r in sub if r["peak"]).most_common(5))
        vals = [(r["peak"]["peak_head_attention"], r["peak"]["peak_head_fraction"]) for r in sub if r["peak"]]
        print(task, f"峰值头注意力均值 {np.mean([a for a, _ in vals]) * 100:.1f}%，占该窗全部头之和 {np.mean([b for _, b in vals]):.0%}，局数 {len(vals)}")
    (out / "layer-attention.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
