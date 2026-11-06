"""motion 模型注意力时间轴：帧流、motion 窗、开局和交换事件画在同一条环境步轴上，并按“有重叠就算”重算两块占比。

口径（用户 2026-11-06 指定）：
- 4 次推理（闭合前 -3～0）合并：每局先平均，再按局平均，不分开画。
- motion 窗内容范围 [t, end)，end = t+32；演示段被 exec_start_idx 截断时 end = min(t+32, exec_start_idx)。
  窗实际 33 帧、32 个位移，最后一帧与下一窗首帧相同，所以 [32,64) 不算碰到从 64 开始的交换。
- 窗只要和交换有重叠，就整窗算读交换。旧口径（重叠/33 计权）一并输出作对照。
- 帧流按帧时刻落在 [0,32) 算开局，按 [64,64+50k) 算交换。

用法：plot_attention_timeline.py <episodes 目录> <输出目录>
"""
from collections import defaultdict
import glob
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

matplotlib.font_manager.fontManager.addfont(subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True))
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
TASKS = {"ButtonUnmaskSwap": "Button", "VideoUnmaskSwap": "Video"}
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BIN = 8
MIN_SHARE = .5


def overlap(a, b, lo, hi):
    return min(b, hi) > max(a, lo)


def boot(values):
    values = np.asarray(values, float)
    means = values[np.random.default_rng(42).integers(0, values.size, (2000, values.size))].mean(1)
    return {"n": int(values.size), "mean": float(values.mean()), "lo": float(np.quantile(means, .025)), "hi": float(np.quantile(means, .975))}


def load(root):
    groups = defaultdict(list)
    for path in sorted(glob.glob(f"{root}/with_motion_*/episode_window.json")):
        e = json.loads(Path(path).read_text(encoding="utf-8"))
        k = e["swap_times"]
        swap_end = 64 + 50 * k
        frames, windows, extents, block = defaultdict(list), defaultdict(list), {}, defaultdict(list)
        closes, execs = [], set()
        for rec in e["records"]:
            d = rec["exec_start_idx"]
            execs.add(d)
            if rec["infer_seq"] == e["G"]:
                closes.append(rec["step_idx"])
            fr, mo = rec["streams"]["frame"], rec["motion_windows"]
            fm, mm = sum(x["mass"] for x in fr), sum(w["mass"] for w in mo)
            for x in fr:
                frames[x["t"] // BIN * BIN].append(x["mass"] * 100)
            reveal = [x for x in fr if 0 <= x["t"] < 32]
            hit, flags = [], []
            for w in mo:
                t = w["t"]
                end = min(t + 32, d) if t < d else t + 32
                extents[t] = end
                windows[t].append(w["mass"] * 100)
                flags.append(overlap(t, end, 64, swap_end))
                hit.append(w["mass"])
            block["reveal_total"].append(sum(x["mass"] for x in reveal) * 100)
            block["reveal_stream"].append(sum(x["mass"] for x in reveal) / fm * 100)
            block["reveal_uniform"].append(len(reveal) / len(fr) * 100)
            swap_mass = sum(m for m, f in zip(hit, flags) if f)
            block["swap_total"].append(swap_mass * 100)
            block["swap_stream"].append(swap_mass / mm * 100)
            block["swap_uniform"].append(sum(flags) / len(flags) * 100)
            block["swap_old_stream"].append(rec["metrics"]["m_swap"] * 100)
            block["swap_old_uniform"].append(rec["metrics"]["m_swap_u"] * 100)
        groups[e["task"], k].append({"key": e["episode_key"], "frames": {t: float(np.mean(v)) for t, v in frames.items()},
                                     "windows": {t: float(np.mean(v)) for t, v in windows.items()}, "extents": extents,
                                     "block": {n: float(np.mean(v)) for n, v in block.items()},
                                     "close": closes[0], "exec": execs.pop()})
    return groups


def summarize(rows):
    out = {n: boot([r["block"][n] for r in rows]) for n in rows[0]["block"]}
    for name in ("reveal", "swap", "swap_old"):
        ratio = [r["block"][f"{name}_stream"] / r["block"][f"{name}_uniform"] for r in rows]
        out[f"{name}_enrich"] = boot(ratio)
        out[f"{name}_enrich"]["above_1"] = int(sum(x > 1 for x in ratio))
    return out


def shade(ax, k, top):
    ax.axvspan(0, 32, color=BLUE, alpha=.12, lw=0)
    for j in range(k):
        ax.axvspan(64 + 50 * j, 114 + 50 * j, color=ORANGE, alpha=.10 + .06 * (j % 2), lw=0)
    if top:
        ylim = ax.get_ylim()[1]
        ax.text(16, ylim * 1.02, "开局", ha="center", va="bottom", fontsize=9, color=INK)
        for j in range(k):
            ax.text(89 + 50 * j, ylim * 1.02, f"交换{j + 1}", ha="center", va="bottom", fontsize=9, color=INK)


def style(ax, xmax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", color="#e8e7e3", lw=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b3ac")
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def draw(groups, out):
    ks = sorted({k for _, k in groups})
    fig = plt.figure(figsize=(15, 3.4 * len(ks) + 1.4), facecolor=SURFACE, constrained_layout=True)
    outer = fig.add_gridspec(len(ks), len(TASKS))
    frame_top, motion_top = 0, 0
    stats = {}
    for key, rows in groups.items():
        fbins, wins = defaultdict(list), defaultdict(list)
        for r in rows:
            for t, v in r["frames"].items():
                fbins[t].append(v)
            for t, v in r["windows"].items():
                wins[t].append(v)
        keep = lambda d: {t: float(np.mean(v)) for t, v in d.items() if len(v) >= MIN_SHARE * len(rows)}
        stats[key] = {"frames": keep(fbins), "windows": keep(wins)}
        frame_top = max(frame_top, max(stats[key]["frames"].values()))
        motion_top = max(motion_top, max(stats[key]["windows"].values()))
    xmax = max(max(t + 32 for t in s["windows"]) for s in stats.values())
    xmax = max(xmax, max(float(np.median([r["close"] for r in rows])) for rows in groups.values()) + 16)
    xmax = int(np.ceil(xmax / 32) * 32)
    for r_index, k in enumerate(ks):
        for c_index, task in enumerate(TASKS):
            rows = groups[task, k]
            inner = outer[r_index, c_index].subgridspec(2, 1, hspace=0)
            top_ax = fig.add_subplot(inner[0])
            mot_ax = fig.add_subplot(inner[1], sharex=top_ax)
            stat = stats[task, k]
            times = sorted(stat["frames"])
            top_ax.step(times + [times[-1] + BIN], [stat["frames"][t] for t in times] + [stat["frames"][times[-1]]],
                        where="post", color=INK, lw=1.5)
            top_ax.set_ylim(0, frame_top * 1.1)
            swap_end = 64 + 50 * k
            for t, v in sorted(stat["windows"].items()):
                end = max(r["extents"][t] for r in rows if t in r["extents"])
                color = BLUE if t == 0 else ORANGE if overlap(t, end, 64, swap_end) else GRAY
                mot_ax.bar(t + 1, v, width=end - t - 2, align="edge", color=color)
            mot_ax.set_ylim(0, motion_top * 1.1)
            for ax, label in ((top_ax, "每帧占比（%）"), (mot_ax, "每窗占比（%）")):
                style(ax, xmax, label)
            shade(top_ax, k, True)
            shade(mot_ax, k, False)
            close = float(np.median([r["close"] for r in rows]))
            for ax in (top_ax, mot_ax):
                ax.axvline(close, color=INK, lw=1, ls=(0, (2, 2)))
            near_edge = close > xmax - 64  # 靠右边的标签放线左侧，否则会把这一格的坐标区挤窄
            mot_ax.text(close - 3 if near_edge else close + 3, motion_top * 1.03, f"闭合中位 {close:.0f}",
                        ha="right" if near_edge else "left", va="top", fontsize=9, color=INK)
            execs = sorted({r["exec"] for r in rows})
            if execs != [0]:
                for ax in (top_ax, mot_ax):
                    ax.axvline(execs[0], color=MUTED, lw=1)
                mot_ax.text(execs[0] + 3, motion_top * 1.03, f"模型接手 {execs[0]}", ha="left", va="top", fontsize=9, color=MUTED)
            summary = summarize(rows)
            top_ax.set_title(f"{TASKS[task]} · 交换 {k} 次 · {len(rows)} 局｜帧流读开局 {summary['reveal_stream']['mean']:.1f}%（均匀 {summary['reveal_uniform']['mean']:.1f}%）"
                             f"｜motion 读交换 {summary['swap_stream']['mean']:.1f}%（均匀 {summary['swap_uniform']['mean']:.1f}%）",
                             color=INK, fontsize=11, pad=18)
            top_ax.tick_params(labelbottom=False)
            mot_ax.set_xticks(range(0, xmax + 1, 32))
            mot_ax.tick_params(axis="x", labelsize=8)
            if r_index == len(ks) - 1:
                mot_ax.set_xlabel("环境步", color=MUTED)
    handles = [Line2D([], [], color=INK, lw=1.5, label="帧流：每 8 步内每帧平均占比"),
               Patch(color=BLUE, label="motion 窗：开局窗 [0,32)"), Patch(color=ORANGE, label="motion 窗：与交换有重叠，整窗算读交换"),
               Patch(color=GRAY, label="motion 窗：其他"), Patch(color=BLUE, alpha=.12, label="真实开局裸露段"),
               Patch(color=ORANGE, alpha=.13, label="真实交换段")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=10, labelcolor=MUTED)
    fig.suptitle("motion 模型注意力时间轴（闭合前 4 次推理合并；纵轴占全部注意力；小标题百分比占本流）", color=INK, fontsize=14)
    fig.savefig(out / "fig_attention_timeline.png", dpi=130)


def main(root, out):
    groups = load(root)
    result = {}
    for task in TASKS:
        rows = [r for (t, _), v in groups.items() if t == task for r in v]
        result[task] = {"all": summarize(rows), "by_k": {str(k): summarize(v) for (t, k), v in sorted(groups.items()) if t == task}}
        for label, block in [("全部", result[task]["all"])] + [(f"k={k}", b) for k, b in result[task]["by_k"].items()]:
            fmt = lambda n: f"{block[n]['mean']:.3f}"
            print(task, label, f"n={block['reveal_enrich']['n']}",
                  f"开局 总{fmt('reveal_total')}% 流内{block['reveal_stream']['mean']:.1f}% 均匀{block['reveal_uniform']['mean']:.1f}% 富集{block['reveal_enrich']['mean']:.2f}[{block['reveal_enrich']['lo']:.2f},{block['reveal_enrich']['hi']:.2f}] >1:{block['reveal_enrich']['above_1']}",
                  f"| 交换(有重叠) 总{fmt('swap_total')}% 流内{block['swap_stream']['mean']:.1f}% 均匀{block['swap_uniform']['mean']:.1f}% 富集{block['swap_enrich']['mean']:.2f}[{block['swap_enrich']['lo']:.2f},{block['swap_enrich']['hi']:.2f}] >1:{block['swap_enrich']['above_1']}",
                  f"| 旧计权 流内{block['swap_old_stream']['mean']:.1f}% 富集{block['swap_old_enrich']['mean']:.2f}")
    (out / "attention-timeline.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    draw(groups, out)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
