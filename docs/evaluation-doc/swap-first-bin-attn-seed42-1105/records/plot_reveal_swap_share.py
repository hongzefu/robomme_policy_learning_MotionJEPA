"""motion 模型闭合前 4 次推理：帧流读开局、motion 流读交换各占多少注意力。

输入 first_bin_attention 写出的 episodes/with_motion_*/episode_window.json。
占全部注意力 = 流内占比 × 流质量；帧流按帧时刻落在 [0,32) 计，motion 流按窗与交换区间的重叠比例计权。
均匀水平 = 流质量 × 该块均匀期望。每局每个相对推理一个值，局级重采样 2000 次（种子 42）。

用法：plot_reveal_swap_share.py <episodes 目录> <输出目录>
"""
import glob
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np

matplotlib.font_manager.fontManager.addfont(subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True))
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
TASKS = {"ButtonUnmaskSwap": "Button", "VideoUnmaskSwap": "Video"}
RELS = (-3, -2, -1, 0)
BLOCKS = {"reveal": ("帧流读开局 [0,32)", "#2a78d6"), "swap": ("motion 流读交换", "#eb6834")}
STREAMS = ("frame", "motion", "image_base", "image_wrist", "text", "action")


def boot(values):
    values = np.asarray(values, float)
    means = values[np.random.default_rng(42).integers(0, values.size, (2000, values.size))].mean(1)
    return {"n": int(values.size), "mean": float(values.mean()), "lo": float(np.quantile(means, .025)), "hi": float(np.quantile(means, .975))}


def main(root, out):
    rows = {t: {b: {r: {"total": [], "total_u": [], "stream": [], "stream_u": []} for r in RELS} for b in BLOCKS} for t in TASKS}
    mix = {t: {r: {s: [] for s in STREAMS} for r in RELS} for t in TASKS}
    for path in sorted(glob.glob(f"{root}/with_motion_*/episode_window.json")):
        e = json.loads(Path(path).read_text(encoding="utf-8"))
        for rec in e["records"]:
            rel = rec["infer_seq"] - e["G"]
            if rel not in RELS:
                continue
            m, mass = rec["metrics"], rec["streams"]["mass"]
            for s in STREAMS:
                mix[e["task"]][rel][s].append(mass[s])
            for block, share, uniform, stream in (("reveal", m["vis"], m["vis_u"], mass["frame"]), ("swap", m["m_swap"], m["m_swap_u"], mass["motion"])):
                cell = rows[e["task"]][block][rel]
                cell["total"].append(share * stream)
                cell["total_u"].append(uniform * stream)
                cell["stream"].append(share)
                cell["stream_u"].append(uniform)
    result = {t: {"blocks": {b: {str(r): {k: boot(np.asarray(v) * 100) for k, v in rows[t][b][r].items()} for r in RELS} for b in BLOCKS},
                  "streams_pct": {str(r): {s: float(np.mean(mix[t][r][s]) * 100) for s in STREAMS} for r in RELS}} for t in TASKS}
    out.mkdir(parents=True, exist_ok=True)
    (out / "reveal-swap-share.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True, sharey="row", constrained_layout=True, facecolor="#fcfcfb")
    for row, (key, ylabel) in enumerate((("total", "占全部注意力（%）"), ("stream", "占本流注意力（%）"))):
        for col, task in enumerate(TASKS):
            ax = axes[row, col]
            ax.set_facecolor("#fcfcfb")
            for block, (name, color) in BLOCKS.items():
                cells = [result[task]["blocks"][block][str(r)] for r in RELS]
                mean = [c[key]["mean"] for c in cells]
                ax.fill_between(RELS, [c[key]["lo"] for c in cells], [c[key]["hi"] for c in cells], color=color, alpha=.16, lw=0)
                ax.plot(RELS, [c[key + "_u"]["mean"] for c in cells], color=color, lw=1.4, ls=(0, (4, 3)), label=f"{name}：均匀水平")
                ax.plot(RELS, mean, color=color, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2, label=f"{name}：实测")
                for x in (RELS[0], RELS[-1]):
                    index = RELS.index(x)
                    # 目视定的避让：这些数字放在点下方，才不压虚线或另一条的数字
                    below = block == "swap" and (key == "total" or task == "VideoUnmaskSwap")
                    ax.annotate(f"{mean[index]:.2f}" if key == "total" else f"{mean[index]:.1f}", (x, mean[index]),
                                xytext=(0, -18 if below else 10), textcoords="offset points", ha="center", fontsize=10, color="#0b0b0b")
            top = max(max(result[t]["blocks"][b][str(r)][key]["hi"], result[t]["blocks"][b][str(r)][key + "_u"]["mean"])
                      for t in TASKS for b in BLOCKS for r in RELS)
            ax.set_ylim(0, top * 1.15)
            ax.set_xticks(RELS, ["-3", "-2", "-1", "0（闭合）"])
            ax.grid(axis="y", color="#e8e7e3", lw=.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#b5b3ac")
            ax.tick_params(colors="#52514e", length=0)
            if col == 0:
                ax.set_ylabel(ylabel, color="#52514e")
            if row == 0:
                ax.set_title(TASKS[task], color="#0b0b0b", fontsize=13)
            else:
                ax.set_xlabel("相对闭合的推理次序", color="#52514e")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=10, labelcolor="#52514e")
    fig.suptitle("motion 模型闭合前 4 次推理：这两块各占多少注意力（阴影为局级 95% 区间）", color="#0b0b0b", fontsize=14)
    fig.savefig(out / "fig_reveal_swap_share.png", dpi=150)
    for task in TASKS:
        for block, (name, _) in BLOCKS.items():
            print(task, name, " ".join(f"{r}:{result[task]['blocks'][block][str(r)]['total']['mean']:.3f}%/u{result[task]['blocks'][block][str(r)]['total_u']['mean']:.3f}%/流内{result[task]['blocks'][block][str(r)]['stream']['mean']:.1f}%(u{result[task]['blocks'][block][str(r)]['stream_u']['mean']:.1f}%)n={result[task]['blocks'][block][str(r)]['total']['n']}" for r in RELS))
        print(task, "全部流%", {r: {s: round(v, 2) for s, v in result[task]["streams_pct"][str(r)].items()} for r in RELS})


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
