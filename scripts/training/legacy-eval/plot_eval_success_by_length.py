#!/usr/bin/env python3
"""按任务 × 任务长度画官方 framesamp+context baseline 与 aws motion run 的成功率对照图。

四个任务各一张子图（从左到右按官方数据集中位 episode 帧长递增），子图内横轴是该任务
自己的长度档（easy/medium/hard 按该任务的中位帧长排序，多数任务不是字面顺序），
末尾补一档 50 集合计。柱高是三个 policy seed（42 / 7 / 2024）的成功率均值，
误差棒取三 seed 的 min–max，叠加的散点是三个 seed 各自的值。

所有数字现读留档 JSON、不手抄：

- 逐集成败：eval-3seed-context-vs-motion 的 records/per_episode_by_seed.json
  （两组 × 4 任务 × 50 集 × 3 seed = 1200 集，SEED_AGG=DONE errors=0）。
  该文件只在 origin/v2-motionmem 上，脚本按 --ref 用 `git show` 取，默认读工作区已有的那份。
- 中位帧长：docs/dataset-build-doc/16task-h5-scan/records/memory_axis_16task.json
  （官方 H5 数据集 train split 口径，demo+exec 全长；eval 跑的是 test split，
   逐集实际执行步数未逐集留档，故长度轴用数据集口径的中位值作代理）

用法：uv run python scripts/training/legacy-eval/plot_eval_success_by_length.py [--ref origin/v2-motionmem]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[3]   # 本文件原在 scripts/analysis/ 下（parents[2] 即仓库根）；迁到 scripts/training/legacy-eval/ 后深度 +1
BY_SEED_REL = "docs/training-doc/eval-3seed-context-vs-motion/records/per_episode_by_seed.json"
AXIS_JSON = REPO / "docs/dataset-build-doc/16task-h5-scan/records/memory_axis_16task.json"
OUT_PNG = REPO / "docs/archive/eval-success-by-task-length.png"

TASKS = ["VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"]
DIFFS = ["easy", "medium", "hard"]

# 组名 -> (图例, 柱色, 散点色)
GROUPS = {
    "motion": ("aws（awsprod40k-b128-motion）", "#2f6f9f", "#17364d"),
    "official": ("baseline（官方 framesamp+context）", "#c96a4a", "#6e2f19"),
}
ORDER = ["motion", "official"]


def setup_font() -> None:
    """拼一条 DejaVu Sans → Droid Sans Fallback 的字体回退链。

    Droid Sans Fallback 是纯 CJK 回退字体、不含 ASCII 字形（单独用会丢 "+"、"c" 等字符），
    所以拉丁字符交给 DejaVu Sans、中文由它兜底。回退必须直接写进 font.family 列表——
    走 font.sans-serif 别名时 matplotlib 3.10 的 fallback 不生效（实测中文仍是豆腐块）。
    """
    fonts = ["DejaVu Sans"]
    cjk = Path("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf")
    if cjk.exists():
        font_manager.fontManager.addfont(str(cjk))
        fonts.append(font_manager.FontProperties(fname=str(cjk)).get_name())
    plt.rcParams["font.family"] = fonts
    plt.rcParams["axes.unicode_minus"] = False


def load_by_seed(ref: str | None) -> dict:
    if ref:
        out = subprocess.run(["git", "show", f"{ref}:{BY_SEED_REL}"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    return json.loads((REPO / BY_SEED_REL).read_text())


def load_medians() -> dict[tuple[str, str], int]:
    variants = json.loads(AXIS_JSON.read_text())["variants"]
    return {
        (task, var): stats["num_timesteps"]["median"]
        for var, tasks in variants.items()
        for task, stats in tasks.items()
    }


def tally(data: dict) -> tuple[dict, list[str]]:
    """聚合成 (组, 任务, 长度档) -> {seed: [成功数, 总数]}；"all" 档为该任务 50 集合计。"""
    seeds = [str(s) for s in data["seeds"]]
    cnt: dict[tuple[str, str, str], dict[str, list[int]]] = defaultdict(
        lambda: {s: [0, 0] for s in seeds})
    for ep in data["episodes"]:
        for bucket in (ep["difficulty"], "all"):
            cell = cnt[(ep["group"], ep["task"], bucket)]
            for s in seeds:
                cell[s][1] += 1
                cell[s][0] += int(ep["results"][s]["success"])
    return cnt, seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/v2-motionmem",
                    help="从该 git ref 只读取逐集 JSON；留空则读工作区文件")
    args = ap.parse_args()

    setup_font()
    data = load_by_seed(args.ref or None)
    cnt, seeds = tally(data)
    med = load_medians()
    tasks = sorted(TASKS, key=lambda t: med[(t, "all")])

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.8), sharey=True)
    width = 0.36
    for ax, task in zip(axes, tasks):
        buckets = sorted(DIFFS, key=lambda d: med[(task, d)]) + ["all"]
        labels = [
            (f"全部\nn={cnt[('motion', task, 'all')][seeds[0]][1]}" if b == "all"
             else f"{b}\n{med[(task, b)]} 帧 · n={cnt[('motion', task, b)][seeds[0]][1]}")
            for b in buckets
        ]
        xs = range(len(buckets))
        ax.axvspan(len(buckets) - 1.5, len(buckets) - 0.5, color="#8c8c8c", alpha=0.10, zorder=0)

        for gi, gname in enumerate(ORDER):
            legend, bar_c, dot_c = GROUPS[gname]
            off = (gi - 0.5) * width
            means, lo, hi, per_seed = [], [], [], []
            for b in buckets:
                cell = cnt[(gname, task, b)]
                rates = [cell[s][0] / cell[s][1] * 100 for s in seeds]
                m = sum(rates) / len(rates)
                means.append(m)
                lo.append(m - min(rates))
                hi.append(max(rates) - m)
                per_seed.append(rates)
            bars = ax.bar([x + off for x in xs], means, width, label=legend,
                          color=bar_c, zorder=3)
            ax.errorbar([x + off for x in xs], means, yerr=[lo, hi], fmt="none",
                        ecolor="#444444", elinewidth=1.0, capsize=3, zorder=5)
            for x, rates in zip(xs, per_seed):
                ax.scatter([x + off] * len(rates), rates, s=9, color=dot_c,
                           zorder=6, linewidths=0)
            # 标签抬到误差棒上端之上，否则 min-max 跨度大的格子会把数字压住
            for bar, m, rates in zip(bars, means, per_seed):
                ax.text(bar.get_x() + bar.get_width() / 2, max(max(rates), m) + 1.8,
                        f"{m:.1f}", ha="center", va="bottom", fontsize=7.5,
                        color="#333333", zorder=7)

        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(f"{task}\n数据集中位 {med[(task, 'all')]} 帧", fontsize=11.5, pad=8)
        ax.set_ylim(0, 62)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("成功率（%）　柱高 = 3 seed 均值", fontsize=10.5)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=10.5)
    fig.suptitle(
        "test split 每任务 50 集 · policy seed 42 / 7 / 2024 · max_steps=1300　|　"
        "误差棒 = 三 seed min–max，散点 = 各 seed 单值　|　"
        "子图从左到右按任务中位帧长递增，子图内长度档同样按帧长排序",
        fontsize=9.5, y=0.05, color="#444444")
    fig.tight_layout(rect=(0, 0.055, 1, 0.92))
    fig.savefig(OUT_PNG, dpi=200)
    print(f"WROTE {OUT_PNG}")

    # 复核：与留档 cells 的逐格成功数逐一比对，防止聚合口径写错
    bad = 0
    for key, cell in data["cells"].items():
        g, s, t = key.split("|")
        got = cnt[(g, t, "all")][s]
        if got[0] != cell["success"] or got[1] != cell["n"]:
            print(f"MISMATCH {key}: got {got} want {cell['success']}/{cell['n']}")
            bad += 1
    print(f"CELL_CHECK={'PASS' if bad == 0 else 'FAIL'} cells={len(data['cells'])} mismatch={bad}")


if __name__ == "__main__":
    main()
