#!/usr/bin/env python3
"""按任务长度画 aws motion run 与官方 framesamp+context baseline 的成功率对照图。

四个任务各一张子图，横轴是该任务内部按中位 episode 帧长排序的难度档（easy/medium/hard），
纵轴是成功率。所有数字从留档 JSON 现读，不手抄：

- 成功/总数：docs/training-doc/<run>/records/per_episode.json 的 episodes 逐集 success 字段
- 中位帧长：docs/dataset-build-doc/16task-h5-scan/records/memory_axis_16task.json
  （官方 H5 数据集 train split 口径，demo+exec 全长；eval 跑的是 test split，
   逐集实际执行步数未逐集留档，故长度轴用数据集口径的中位值作代理）

用法：uv run python scripts/analysis/plot_eval_success_by_length.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[2]
RUN_AWS = "eval-awsprod40k-b128-motion"
RUN_BASE = "eval-official-framesamp-context"
AXIS_JSON = REPO / "docs/dataset-build-doc/16task-h5-scan/records/memory_axis_16task.json"
OUT_PNG = REPO / "docs/eval-success-by-task-length.png"

TASKS = ["VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"]
DIFFS = ["easy", "medium", "hard"]

C_AWS = "#2f6f9f"      # 我们的 motion run
C_BASE = "#c96a4a"     # 官方 baseline
C_ALL = "#8c8c8c"      # 全任务合计一档的分隔底色


def setup_font() -> None:
    """拼一条 DejaVu Sans → Droid Sans Fallback 的字体回退链。

    Droid Sans Fallback 是纯 CJK 回退字体，不含 ASCII 字形（单独用会丢掉 "+"、"c" 等字符），
    所以拉丁字符交给 DejaVu Sans，中文由它兜底。
    """
    fonts = ["DejaVu Sans"]
    cjk = Path("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf")
    if cjk.exists():
        font_manager.fontManager.addfont(str(cjk))
        fonts.append(font_manager.FontProperties(fname=str(cjk)).get_name())
    plt.rcParams["font.family"] = fonts
    plt.rcParams["axes.unicode_minus"] = False


def load_counts(run_name: str) -> dict[tuple[str, str], list[int]]:
    """读一条 eval run 的逐集结果，聚合成 (任务, 难度) -> [成功数, 总数]。"""
    path = REPO / "docs/training-doc" / run_name / "records/per_episode.json"
    data = json.loads(path.read_text())
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for ep in data["episodes"]:
        key = (ep["task"], ep["difficulty"])
        counts[key][1] += 1
        counts[key][0] += int(ep["success"])
    return counts, data["mean_rate"]


def load_medians() -> dict[tuple[str, str], int]:
    variants = json.loads(AXIS_JSON.read_text())["variants"]
    return {
        (task, var): stats["num_timesteps"]["median"]
        for var, tasks in variants.items()
        for task, stats in tasks.items()
    }


def main() -> None:
    setup_font()
    aws, aws_mean = load_counts(RUN_AWS)
    base, base_mean = load_counts(RUN_BASE)
    med = load_medians()

    # 子图顺序 = 任务按 all 档中位帧长从短到长
    tasks = sorted(TASKS, key=lambda t: med[(t, "all")])

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.4), sharey=True)
    width = 0.36

    for ax, task in zip(axes, tasks):
        # 任务内部按中位帧长排序（多数任务是 medium < easy < hard，不是字面顺序）
        diffs = sorted(DIFFS, key=lambda d: med[(task, d)])
        labels, aws_pct, base_pct, aws_txt, base_txt = [], [], [], [], []
        for d in diffs:
            a, b = aws[(task, d)], base[(task, d)]
            labels.append(f"{d}\n{med[(task, d)]} 帧 · n={a[1]}")
            aws_pct.append(a[0] / a[1] * 100)
            base_pct.append(b[0] / b[1] * 100)
            aws_txt.append(f"{a[0]}/{a[1]}")
            base_txt.append(f"{b[0]}/{b[1]}")
        # 末尾补一档任务合计（50 集）
        a_all = [sum(aws[(task, d)][i] for d in DIFFS) for i in (0, 1)]
        b_all = [sum(base[(task, d)][i] for d in DIFFS) for i in (0, 1)]
        labels.append(f"全部\nn={a_all[1]}")
        aws_pct.append(a_all[0] / a_all[1] * 100)
        base_pct.append(b_all[0] / b_all[1] * 100)
        aws_txt.append(f"{a_all[0]}/{a_all[1]}")
        base_txt.append(f"{b_all[0]}/{b_all[1]}")

        xs = range(len(labels))
        ax.axvspan(len(labels) - 1.5, len(labels) - 0.5, color=C_ALL, alpha=0.10, zorder=0)
        bars_a = ax.bar([x - width / 2 for x in xs], aws_pct, width,
                        label="aws（motion 40k）", color=C_AWS, zorder=3)
        bars_b = ax.bar([x + width / 2 for x in xs], base_pct, width,
                        label="baseline（官方 framesamp+context 80k）", color=C_BASE, zorder=3)
        for bars, txts in ((bars_a, aws_txt), (bars_b, base_txt)):
            for bar, txt in zip(bars, txts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, txt,
                        ha="center", va="bottom", fontsize=7.5, color="#333333", zorder=4)

        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(f"{task}\n数据集中位 {med[(task, 'all')]} 帧", fontsize=11, pad=8)
        ax.set_ylim(0, 68)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("成功率（%）", fontsize=11)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=10.5)
    fig.suptitle(
        f"test split 每任务 50 集 · seed 42 · max_steps=1300　|　"
        f"总成功率 aws {aws_mean * 100:.1f}%　vs　baseline {base_mean * 100:.1f}%　"
        f"（子图从左到右按任务中位帧长递增；柱上数字为 成功/总数）",
        fontsize=9.5, y=0.055, color="#444444")
    fig.tight_layout(rect=(0, 0.055, 1, 0.925))
    fig.savefig(OUT_PNG, dpi=200)
    print(f"WROTE {OUT_PNG}")


if __name__ == "__main__":
    main()
