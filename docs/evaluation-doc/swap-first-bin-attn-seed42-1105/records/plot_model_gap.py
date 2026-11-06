"""主图初稿：基线与 motion 模型读开局的差距（哑铃图）。

五个指标量纲不同，所以每个指标一个横轴，不共用坐标。数值全部读留档；区间都是局级重采样 2000 次（种子 42）。
输出 fig_model_gap.png、fig_model_gap.pdf、model-gap.json。

用法：plot_model_gap.py <records 目录> <action-flow 目录>
"""
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
MODELS = {"baseline": ("基线", "#eb6834", "s"), "motion": ("motion 模型", "#2a78d6", "o")}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def boot(values):
    values = np.asarray(values, float)
    means = values[np.random.default_rng(42).integers(0, values.size, (2000, values.size))].mean(1)
    return {"mean": float(values.mean()), "lo": float(np.quantile(means, .025)), "hi": float(np.quantile(means, .975)), "n": int(values.size)}


def direct_stat(flow, index, groups, model, task, layer, head, query, key):
    values = []
    for e in index["episodes"]:
        if e["model"] == model and e["task"] == task:
            with np.load(flow / e["file"]) as z:
                d = z["direct"][layer, :, groups[query], groups[key]]
                values.append((d.mean() if head is None else d[head]) * 100)
    return boot(values)


def collect(records, flow):
    index = json.loads((flow / "index.json").read_text(encoding="utf-8"))
    groups = {n: i for i, n in enumerate(index["groups"])}
    pattern = json.loads((records / "pattern-summary-main.json").read_text(encoding="utf-8"))
    reads = json.loads((records / "read-targets.json").read_text(encoding="utf-8"))
    flows = json.loads((records / "action-flow-summary.json").read_text(encoding="utf-8"))
    code = {"baseline": "without_motion", "motion": "with_motion"}
    rows = []
    m1 = {}
    for model in MODELS:
        m1[model] = {}
        for task in TASKS:
            item = next(i for i in pattern["primary"]["intervals"] if i["readout"] == "M1" and i["model"] == code[model] and i["task"] == task)
            m1[model][task] = {"mean": item["mean"], "lo": item["ci95"][0], "hi": item["ci95"][1], "n": item["n_episodes"]}
    rows.append({"title": "帧流读开局帧的富集倍数", "note": "预定主读数；虚线 = 均匀", "unit": "倍", "digits": 2, "ref": 1.0, "data": m1})
    share = {m: {t: reads["cells"][f"{m}|{t}"]["all"]["events"]["text"]["frame"]["reveal"]["share_of_stream"] for t in TASKS} for m in MODELS}
    rows.append({"title": "文字读历史帧时，开局段的占比", "note": "全部层合并", "unit": "%", "digits": 1, "data": share})
    rows.append({"title": "第 0 层文字分给开局帧的注意力", "note": "8 个头平均", "unit": "%", "digits": 1,
                 "data": {m: {t: direct_stat(flow, index, groups, m, t, 0, None, "text", "frame_reveal") for t in TASKS} for m in MODELS}})
    rows.append({"title": "动作头 L3H0 分给开局帧的注意力", "note": "事后挑选的头", "unit": "%", "digits": 1,
                 "data": {m: {t: direct_stat(flow, index, groups, m, t, 3, 0, "action", "frame_reveal") for t in TASKS} for m in MODELS}})
    rows.append({"title": "动作来源中的开局帧（累积估计）", "note": "残差比例 0.5，注意力近似", "unit": "%", "digits": 2,
                 "data": {m: {t: flows["cells"][f"{m}|{t}"]["alpha_0.5"]["action_final"]["frame_reveal"] for t in TASKS} for m in MODELS}})
    return rows


def draw(rows, out):
    fig, axes = plt.subplots(len(rows), 1, figsize=(9.5, 10.5), facecolor=SURFACE, constrained_layout=True)
    for ax, row in zip(axes, rows):
        ax.set_facecolor(SURFACE)
        for y, task in enumerate(TASKS):
            base, motion = row["data"]["baseline"][task], row["data"]["motion"][task]
            ax.plot([base["mean"], motion["mean"]], [y, y], color="#c9c7c0", lw=3, zorder=1, solid_capstyle="round")
            for model, stat in (("baseline", base), ("motion", motion)):
                name, color, marker = MODELS[model]
                ax.plot([stat["lo"], stat["hi"]], [y, y], color=color, lw=1.6, zorder=2)
                ax.plot(stat["mean"], y, marker=marker, ms=10, color=color, mec=SURFACE, mew=2, ls="none", zorder=3)
                left = stat["mean"] <= (base["mean"] + motion["mean"]) / 2
                ax.annotate(f"{stat['mean']:.{row['digits']}f}{row['unit']}", (stat["mean"], y), xytext=(0, 10), textcoords="offset points",
                            ha="center", va="bottom", fontsize=10, color=INK)
        values = [s[k] for m in row["data"].values() for s in m.values() for k in ("lo", "hi")]
        low, high = min(values), max(values)
        pad = (high - low) * .12
        left_edge = low - pad  # 基线可能接近 0，留白避免点贴边、数字压到任务名
        ax.set_xlim(left_edge, high + pad)
        ax.set_xticks([t for t in ax.get_xticks() if left_edge <= t <= high + pad and t >= 0])
        if "ref" in row:
            ax.axvline(row["ref"], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
        ax.set_ylim(-.65, 1.85)
        ax.invert_yaxis()
        ax.set_yticks(range(len(TASKS)), TASKS.values())
        ax.tick_params(axis="y", colors=INK, length=0, labelsize=11)
        ax.tick_params(axis="x", colors=MUTED, length=0, labelsize=9)
        ax.grid(axis="x", color="#e8e7e3", lw=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#b5b3ac")
        ax.set_title(row["title"], loc="left", color=INK, fontsize=12, pad=6)
        ax.set_title(row["note"], loc="right", color=MUTED, fontsize=9, pad=6)
    handles = [plt.Line2D([], [], marker=MODELS[m][2], ms=9, color=MODELS[m][1], mec=SURFACE, ls="none", label=MODELS[m][0]) for m in MODELS]
    handles.append(plt.Line2D([], [], color=MUTED, lw=1.6, label="横线：局级 95% 区间"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=10.5, labelcolor=INK)
    fig.suptitle("motion 模型读开局信息明显多于基线\n只说明读取差异，不说明因果（每任务 49～50 局，闭合前 4 次推理，seed 42）", color=INK, fontsize=13)
    fig.savefig(out / "fig_model_gap.png", dpi=160)
    fig.savefig(out / "fig_model_gap.pdf")


def main(records, flow):
    rows = collect(records, flow)
    (records / "model-gap.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        print(row["title"], {m: {TASKS[t]: f"{s['mean']:.3f}[{s['lo']:.3f},{s['hi']:.3f}]" for t, s in v.items()} for m, v in row["data"].items()})
    draw(rows, records)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
