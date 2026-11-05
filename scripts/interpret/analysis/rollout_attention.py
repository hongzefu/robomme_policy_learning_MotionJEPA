"""StopCube 真实 rollout 的 Q2 注意力汇总与想按比例 P 曲线（1105 计划 K5、K6、K8，§三.4）。

两个子命令：
  heights    只读，在 sim 环境解释器里运行，把离线记录的 p_actions 换算成 TCP 高度和 P，
             写 <root>/analysis/heights/<模型>/<episode>.npz（文件名里非 \\w.- 的字符换成 _）。
  summarize  主环境 CPU，写 <root>/analysis/attention-summary.json 和四张图，打印 ATTN_ALIGN 判定行。

只描述「看了哪里」，不做干预，不写因果。统计单位是局，不按 task_success 挑局。
层、头都从 0 编号。对齐点缺失（None）时如实计数，绝不用别的推理顶替。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import re

import numpy as np

from scripts.interpret.analysis.oracle_press_prob import (bootstrap, convert_arrays, episode_number, load_table,
                                                          probabilities, write_json)


MODELS = ("ctx", "mot")
FIVE_GROUPS = ("motion", "历史视觉", "当前视觉", "文字", "其余")
# 对齐点取 align.json 的字段名，顺序就是图和表的顺序；I_near 只给未按的局，单独成组。
POINTS = ("I_cross_minus_3", "I_cross_minus_2", "I_cross_minus_1", "I_cross", "I_trigger", "I_near")
POINT_LABEL = {"I_cross_minus_3": "I_cross−3", "I_cross_minus_2": "I_cross−2", "I_cross_minus_1": "I_cross−1",
               "I_cross": "I_cross", "I_trigger": "I_trigger", "I_near": "I_near（未按）"}
CANDIDATE_HEADS = ((13, 6), (15, 1), (16, 0))
CATEGORIES = ("窗口内", "偏早", "偏晚", "未按")
WINDOW_PREFIX = "motion窗:"
WINDOW_FRAMES = 33
SUM_TOL = 1e-3
P_KEY = "h16_cm2"
CURVE_BIN = 16
REPEATS, SEED = 2000, 42
MODEL_STYLE = {"ctx": ("基线 ctx", "#2a78d6", "o"), "mot": ("motion", "#eb6834", "s")}
GROUP_COLOR = dict(zip(FIVE_GROUPS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_align(root):
    return json.loads((Path(root) / "analysis" / "align.json").read_text(encoding="utf-8"))


def offline_path(root, model, nonce, seq):
    return Path(root) / model / "offline" / str(nonce) / f"offline_{seq:04d}.npz"


def heights_path(root, model, key):
    return Path(root) / "analysis" / "heights" / model / (re.sub(r"[^\w.-]", "_", key) + ".npz")


def scalar(value):
    return np.asarray(value).item()


def read_view(path):
    """只读 Q2 需要的键（hidden 不解压）。基线的 motion 组 token 数为 0 或不存在时记 None，不补 0。"""
    with np.load(path, allow_pickle=False) as archive:
        names = [str(name) for name in archive["group_names"]]
        mass = np.asarray(archive["group_mass"], dtype=np.float64)
        tokens = np.asarray(archive["group_tokens_all"]).astype(np.int64)
        view = {"episode_key": str(scalar(archive["episode_key"])), "infer_seq": int(scalar(archive["infer_seq"])),
                "chunk_start": int(scalar(archive["chunk_start_env_step"])),
                "step_idx": int(scalar(archive["step_idx"])),
                "attn_path_maxabs": float(scalar(archive["attn_path_maxabs"])),
                "same_path_maxabs": float(scalar(archive["same_path_maxabs"]))}
    if mass.ndim != 3 or mass.shape[1:] != (18, 8) or mass.shape[0] != len(names) or tokens.shape != (len(names),):
        raise ValueError(f"组质量形状不符：{path} {mass.shape}")
    index = {name: position for position, name in enumerate(names)}
    view["five"], view["tokens"] = {}, {}
    total = np.zeros((18, 8))
    for group in FIVE_GROUPS:
        position = index.get(group)
        count = int(tokens[position]) if position is not None else 0
        view["tokens"][group] = count
        view["five"][group] = mass[position] if count > 0 else None
        if position is not None:
            total = total + mass[position]
    view["sum_err"] = float(np.abs(total - 1.0).max())
    view["total_tokens"] = sum(view["tokens"].values())
    view["offset"] = view["chunk_start"] - view["step_idx"]
    view["windows"] = [{"start": int(name[len(WINDOW_PREFIX):]), "mass": float(mass[position].mean()),
                        "tokens": int(tokens[position])}
                       for name, position in index.items()
                       if name.startswith(WINDOW_PREFIX) and tokens[position] > 0]
    return view


def load_episodes(root, model, align_model, problems):
    """读每个进入 uses.q2_other 的局的全部推理；与 align.json 不一致的记录记问题且不用。"""
    episodes, totals = [], {"sum_err": 0.0, "attn_path_maxabs": 0.0, "same_path_maxabs": 0.0, "records": 0}
    for key, item in align_model.items():
        if not item.get("valid") or not item["uses"]["q2_other"]:
            continue
        records = {}
        for point in item["inferences"]:
            seq = point["infer_seq"]
            path = offline_path(root, model, item["nonce"], seq)
            if not path.exists():
                problems.append(f"{model} {key} 缺离线记录 {path}")
                continue
            view = read_view(path)
            if view["episode_key"] != key or view["infer_seq"] != seq or view["chunk_start"] != point["chunk_start_env_step"]:
                problems.append(f"{model} {key} 记录身份与 align.json 不符 {path}")
                continue
            records[seq] = view
            totals["records"] += 1
            for name in ("sum_err", "attn_path_maxabs", "same_path_maxabs"):
                totals[name] = max(totals[name], view[name])
        episodes.append({"key": key, "item": item, "records": records})
    return episodes, totals


def stat(values, repeats=REPEATS, seed=SEED):
    """局级重采样；没有样本时显式标缺失，不补 0。"""
    result = bootstrap(values, repeats, seed)
    result["missing"] = result["n_episodes"] == 0
    return result


def point_view(episode, point):
    seq = episode["item"].get(point)
    return None if seq is None else episode["records"].get(seq)


def per_token(view, group):
    mass = view["five"][group]
    return None if mass is None else float(mass.mean()) / view["tokens"][group]


def five_table(episodes, point, repeats=REPEATS, seed=SEED):
    """某对齐点五组的总质量、单 token 质量和 token 数。n_missing 含本对齐点不适用的局（如 I_near 对已按的局）。"""
    views = [view for view in (point_view(episode, point) for episode in episodes) if view is not None]
    groups = {}
    for group in FIVE_GROUPS:
        having = [view for view in views if view["five"][group] is not None]
        groups[group] = {"mean_mass": stat([float(view["five"][group].mean()) for view in having], repeats, seed),
                         "per_token": stat([per_token(view, group) for view in having], repeats, seed),
                         "tokens": stat([view["tokens"][group] for view in having], repeats, seed)}
    return {"n_episodes": len(episodes), "n_present": len(views), "n_missing": len(episodes) - len(views), "groups": groups}


def self_change(episodes, group="motion", earlier="I_cross_minus_3", later="I_cross", repeats=REPEATS, seed=SEED):
    """比较①：同局 later 减 earlier 的单 token 质量差和 token 数差；两边都有才配对。"""
    per_token_diffs, token_diffs, keys = [], [], []
    for episode in episodes:
        before, after = point_view(episode, earlier), point_view(episode, later)
        if before is None or after is None or before["five"][group] is None or after["five"][group] is None:
            continue
        per_token_diffs.append(per_token(after, group) - per_token(before, group))
        token_diffs.append(after["tokens"][group] - before["tokens"][group])
        keys.append(episode["key"])
    return {"per_token_diff": stat(per_token_diffs, repeats, seed), "token_count_diff": stat(token_diffs, repeats, seed),
            "episodes": keys}


def uniform_ratio(episodes, point, repeats=REPEATS, seed=SEED):
    """比较②：motion 实际占比 ÷（motion token 数 ÷ 五大组有效 token 总数）。"""
    ratios, fractions = [], []
    for episode in episodes:
        view = point_view(episode, point)
        if view is None or view["five"]["motion"] is None:
            continue
        fraction = view["tokens"]["motion"] / view["total_tokens"]
        ratios.append(float(view["five"]["motion"].mean()) / fraction)
        fractions.append(fraction)
    return {"ratio": stat(ratios, repeats, seed), "token_fraction": stat(fractions, repeats, seed)}


def others_ratio(episodes, point, repeats=REPEATS, seed=SEED):
    """比较③：同一次推理里 motion 单 token ÷ 其他四组单 token；分母为 0 的不计。"""
    result = {}
    for other in FIVE_GROUPS[1:]:
        values = []
        for episode in episodes:
            view = point_view(episode, point)
            if view is None or view["five"]["motion"] is None or view["five"][other] is None:
                continue
            denominator = per_token(view, other)
            if denominator > 0:
                values.append(per_token(view, "motion") / denominator)
        result[other] = stat(values, repeats, seed)
    return result


def head_name(layer, head):
    return f"L{layer}H{head}"


def candidate_table(episodes, repeats=REPEATS, seed=SEED):
    """预定候选头：每个对齐点五组在该头的平均占比和局级区间。"""
    table = {}
    for layer, head in CANDIDATE_HEADS:
        table[head_name(layer, head)] = {}
        for point in POINTS:
            views = [view for view in (point_view(episode, point) for episode in episodes) if view is not None]
            table[head_name(layer, head)][point] = {
                group: stat([float(view["five"][group][layer, head]) for view in views if view["five"][group] is not None],
                            repeats, seed) for group in FIVE_GROUPS}
    return table


def max_head_counts(episodes, point="I_cross"):
    """I_cross 处 motion 最大头的逐局计数（层×头）和候选头命中局数；基线无 motion 时显式缺失。"""
    counts = np.zeros((18, 8), dtype=np.int64)
    for episode in episodes:
        view = point_view(episode, point)
        if view is None or view["five"]["motion"] is None:
            continue
        layer, head = np.unravel_index(int(np.argmax(view["five"]["motion"])), (18, 8))
        counts[layer, head] += 1
    total = int(counts.sum())
    if total == 0:
        return {"missing": True, "n_episodes": 0, "matrix": None, "candidates": None}
    return {"missing": False, "n_episodes": total, "matrix": counts.tolist(),
            "candidates": {head_name(layer, head): int(counts[layer, head]) for layer, head in CANDIDATE_HEADS}}


def head_matrices(episodes, point):
    """逐组 mean by 层×头，只描述，不做显著性。"""
    result = {}
    for group in FIVE_GROUPS:
        arrays = [view["five"][group] for view in (point_view(episode, point) for episode in episodes)
                  if view is not None and view["five"][group] is not None]
        result[group] = np.mean(arrays, axis=0).tolist() if arrays else None
    return result


def classify_windows(view, visits):
    """把 motion 窗分成覆盖与否。窗口覆盖 [start,start+33) 帧；帧号加 chunk_start−step_idx 得环境步。

    只算在该推理之前结束的 visit；visit 的 in、out 都含端点。
    """
    past = [visit for visit in visits if visit["out"] < view["chunk_start"]]
    covered, other = [], []
    for window in view["windows"]:
        begin = window["start"] + view["offset"]
        end = begin + WINDOW_FRAMES
        hit = any(begin <= visit["out"] and visit["in"] < end for visit in past)
        (covered if hit else other).append(window)
    return covered, other


def class_per_token(windows):
    return sum(item["mass"] for item in windows) / sum(item["tokens"] for item in windows) if windows else None


def window_coverage(episodes, visits_by_key, point, repeats=REPEATS, seed=SEED):
    """只用 uses.q2_window 的局。无窗口的局记入 n_no_window，某类无窗口的局该类缺失。"""
    eligible = [episode for episode in episodes if episode["item"]["uses"]["q2_window"]]
    covered, other, gaps, uniforms, no_window, absent = [], [], [], [], 0, 0
    for episode in eligible:
        view = point_view(episode, point)
        if view is None:
            absent += 1
            continue
        if not view["windows"]:
            no_window += 1
            continue
        left, right = classify_windows(view, visits_by_key[episode["key"]])
        left, right = class_per_token(left), class_per_token(right)
        uniforms.append(1.0 / view["total_tokens"])
        if left is not None:
            covered.append(left)
        if right is not None:
            other.append(right)
        if left is not None and right is not None:
            gaps.append(left - right)
    return {"n_episodes": len(eligible), "n_point_missing": absent, "n_no_window": no_window,
            "covered_per_token": stat(covered, repeats, seed), "other_per_token": stat(other, repeats, seed),
            "covered_minus_other": stat(gaps, repeats, seed), "uniform_per_token": stat(uniforms, repeats, seed)}


def summarize_subset(episodes, visits_by_key, repeats=REPEATS, seed=SEED):
    return {"n_episodes": len(episodes),
            "points": {point: five_table(episodes, point, repeats, seed) for point in POINTS},
            "comparisons": {
                # 比较①对五组都算：基线没有 motion，「只比其余几组」。
                "self_change": {group: self_change(episodes, group, repeats=repeats, seed=seed) for group in FIVE_GROUPS},
                "uniform_ratio": {point: uniform_ratio(episodes, point, repeats, seed) for point in POINTS},
                "others_ratio": {point: others_ratio(episodes, point, repeats, seed) for point in POINTS}},
            "candidate_heads": candidate_table(episodes, repeats, seed),
            "max_head": max_head_counts(episodes),
            "window_coverage": {point: window_coverage(episodes, visits_by_key, point, repeats, seed) for point in POINTS}}


def category_subsets(episodes):
    """分组只用 uses.q2_window 的局，未按单列；不按 task_success 筛选。"""
    return {name: [episode for episode in episodes if episode["item"]["uses"]["q2_window"]
                   and episode["item"].get("category") == name] for name in CATEGORIES}


def read_heights(root, model, key):
    path = heights_path(root, model, key)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        return {"chunk_start": np.asarray(archive["chunk_start_env_step"]), "p": np.asarray(archive["p"], dtype=float)}


def curve_rows(pairs):
    """pairs：每局一份 (相对步, P)。先逐局按 16 步分箱取均值，再跨局取中位数和四分位；到位前的 NaN 不计。"""
    bins = {}
    for relative, values in pairs:
        keep = np.isfinite(values)
        relative, values = relative[keep], values[keep]
        for position in np.unique(np.floor(relative / CURVE_BIN)):
            mask = np.floor(relative / CURVE_BIN) == position
            bins.setdefault(int(position), []).append(float(values[mask].mean()))
    rows = []
    for position in sorted(bins):
        array = np.asarray(bins[position])
        rows.append({"bin_center": (position + 0.5) * CURVE_BIN, "n_episodes": len(array),
                     "median": float(np.median(array)), "q25": float(np.quantile(array, 0.25)),
                     "q75": float(np.quantile(array, 0.75))})
    return rows


def p_curves(root, model, episodes):
    """相对窗口中心（依赖窗口，只用 uses.q2_window）和相对 trigger_step（已触发局）两种坐标。"""
    window, trigger, absent = [], [], []
    for episode in episodes:
        item = episode["item"]
        heights = read_heights(root, model, episode["key"])
        if heights is None:
            absent.append(episode["key"])
            continue
        if item["uses"]["q2_window"]:
            window.append((heights["chunk_start"] - item["window_center"], heights["p"]))
        if item.get("trigger_step") is not None:
            trigger.append((heights["chunk_start"] - item["trigger_step"], heights["p"]))
    return {"bin_width": CURVE_BIN, "heights_missing": absent, "n_window": len(window), "n_trigger": len(trigger),
            "relative_window_center": curve_rows(window), "relative_trigger": curve_rows(trigger)}


def count_points(episodes):
    return sum(point_view(episode, point) is not None for episode in episodes for point in POINTS)


def verdict_line(summary, problems):
    models = summary["models"]
    points = sum(item["n_points"] for item in models.values())
    sum_err = max((item["sum_err"] for item in models.values()), default=0.0)
    attn = max((item["attn_path_maxabs"] for item in models.values()), default=0.0)
    passed = (len(models) == len(MODELS) and all(item["n_points"] > 0 for item in models.values())
              and not problems and sum_err <= SUM_TOL)
    return (f"ATTN_ALIGN={'PASS' if passed else 'FAIL'} models={len(models)} points={points} "
            f"sum_err={sum_err:.6g} attn_path_maxabs={attn:.6g}"), passed


def summarize(root, table_path=None, repeats=REPEATS, seed=SEED):
    root = Path(root)
    align = load_align(root)
    table = load_table(table_path or align["table"]["path"])
    visits = {key: entry.get("visits", []) for key, entry in table.items()}
    problems, per_model = [], {}
    for model in MODELS:
        if model not in align["models"]:
            continue
        episodes, totals = load_episodes(root, model, align["models"][model], problems)
        per_model[model] = {
            "n_q2_other": len(episodes), "n_points": count_points(episodes), **totals,
            "all": summarize_subset(episodes, visits, repeats, seed),
            "by_category": {name: summarize_subset(subset, visits, repeats, seed)
                            for name, subset in category_subsets(episodes).items()},
            "head_matrices": {point: head_matrices(episodes, point) for point in POINTS},
            "p_curves": p_curves(root, model, episodes)}
    summary = {"schema_version": 1, "table": align["table"], "models": per_model, "problems": problems,
               "bootstrap": {"repeats": repeats, "seed": seed, "unit": "episode"},
               "candidate_heads": [head_name(*item) for item in CANDIDATE_HEADS],
               "note": "只描述注意力与想按比例；不做干预，不写因果。层、头从 0 编号。"}
    line, passed = verdict_line(summary, problems)
    summary["verdict"] = line
    return summary, line, passed


def setup_plot(output):
    output = Path(output)
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / "matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # TTC 在部分机器上以 JP 名称登记，字形相同；只选实际可用的字体。
    available = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [next(name for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans")
                                        if name in available)]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def clean_axis(axis):
    for name in ("top", "right"):
        axis.spines[name].set_visible(False)
    axis.set_axisbelow(True)


def plot_groups(summary, output):
    """图①：各对齐点五组占比，每个模型一格，只画有样本的对齐点。"""
    plt = setup_plot(output)
    models = list(summary["models"])
    figure, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.6), sharey=True, squeeze=False)
    for axis, model in zip(axes[0], models):
        points = [point for point in POINTS if summary["models"][model]["all"]["points"][point]["n_present"] > 0]
        bottom = np.zeros(len(points))
        for group in FIVE_GROUPS:
            values = []
            for point in points:
                mean = summary["models"][model]["all"]["points"][point]["groups"][group]["mean_mass"]["mean"]
                values.append(0.0 if mean is None else mean)
            bars = axis.bar(range(len(points)), values, bottom=bottom, width=0.62, color=GROUP_COLOR[group],
                            edgecolor="white", linewidth=1.5, label=group)
            if group == "motion":
                for bar, value in zip(bars, values):
                    if value >= 0.03:
                        axis.text(bar.get_x() + bar.get_width() / 2, value / 2, f"{value:.2f}", ha="center",
                                  va="center", fontsize=9, color="white")
            bottom = bottom + np.asarray(values)
        axis.set_xticks(range(len(points)), [f"{POINT_LABEL[point]}\nn={summary['models'][model]['all']['points'][point]['n_present']}"
                                             for point in points], fontsize=9)
        axis.set_title(MODEL_STYLE[model][0], loc="left", fontsize=11, color=INK)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.set_ylim(0, 1.02)
        clean_axis(axis)
    axes[0, 0].set_ylabel("注意力占比（跨层、头平均）", color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[group]) for group in FIVE_GROUPS]
    figure.legend(handles, FIVE_GROUPS, loc="upper center", ncol=len(FIVE_GROUPS), frameon=False, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = Path(output) / "attention-groups.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def format_interval(item):
    if item["missing"]:
        return "缺失"
    return f"{item['mean']:.3f} [{item['ci95'][0]:.3f},{item['ci95'][1]:.3f}]"


def plot_candidates(summary, output):
    """图②：候选头表，均值 [局级 95% 区间]。"""
    plt = setup_plot(output)
    models = list(summary["models"])
    rows = len(CANDIDATE_HEADS) * len(POINTS) + 1
    figure, axes = plt.subplots(len(models), 1, figsize=(13, (0.27 * rows + 0.6) * len(models)), squeeze=False)
    for axis, model in zip(axes[:, 0], models):
        cells = []
        for layer, head in CANDIDATE_HEADS:
            for point in POINTS:
                entry = summary["models"][model]["all"]["candidate_heads"][head_name(layer, head)][point]
                cells.append([head_name(layer, head), POINT_LABEL[point]] + [format_interval(entry[group]) for group in FIVE_GROUPS])
        axis.axis("off")
        axis.set_title(MODEL_STYLE[model][0], loc="left", fontsize=11, color=INK)
        drawn = axis.table(cellText=cells, colLabels=["头", "对齐点", *FIVE_GROUPS], loc="center", cellLoc="center",
                           colWidths=[0.07, 0.13, 0.16, 0.16, 0.16, 0.16, 0.16])
        drawn.auto_set_font_size(False)
        drawn.set_fontsize(8.5)
        drawn.scale(1, 1.05)
        for (row, _), cell in drawn.get_celld().items():
            cell.set_edgecolor(GRID)
            if row == 0:
                cell.set_facecolor("#eeeeea")
    figure.tight_layout()
    path = Path(output) / "attention-candidate-heads.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_heads(summary, output):
    """图③：I_cross 处 144 头热图（层×头），最近邻显示，不插值；缺失的组写明缺失。"""
    plt = setup_plot(output)
    from matplotlib.colors import LinearSegmentedColormap

    colormap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    models = list(summary["models"])
    figure, axes = plt.subplots(len(models), len(FIVE_GROUPS), figsize=(3.0 * len(FIVE_GROUPS), 4.0 * len(models)),
                                squeeze=False, sharex=True, sharey=True)
    for column, group in enumerate(FIVE_GROUPS):
        matrices = [summary["models"][model]["head_matrices"]["I_cross"][group] for model in models]
        upper = max((float(np.max(matrix)) for matrix in matrices if matrix is not None), default=1.0)
        image = None
        for row, model in enumerate(models):
            axis = axes[row, column]
            matrix = matrices[row]
            if matrix is None:
                axis.text(0.5, 0.5, "缺失", ha="center", va="center", transform=axis.transAxes, color=MUTED, fontsize=11)
                axis.set_xlim(-0.5, 7.5)
                axis.set_ylim(17.5, -0.5)
            else:
                image = axis.imshow(np.asarray(matrix), aspect="auto", interpolation="nearest", cmap=colormap,
                                    vmin=0, vmax=upper)
            axis.set_title(f"{MODEL_STYLE[model][0]} · {group}", loc="left", fontsize=10, color=INK)
            axis.set_xticks(range(0, 8))
            axis.set_yticks(range(0, 18, 2))
            axis.tick_params(labelsize=8)
        if image is not None:
            figure.colorbar(image, ax=axes[:, column].tolist(), orientation="horizontal", fraction=0.04, pad=0.07)
    for axis in axes[-1]:
        axis.set_xlabel("头", color=MUTED)
    for axis in axes[:, 0]:
        axis.set_ylabel("层", color=MUTED)
    path = Path(output) / "attention-heads.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_curves(summary, output):
    """图④：P 曲线，逐局中位数与四分位；到位前不计。"""
    plt = setup_plot(output)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = (("relative_window_center", "推理起点步 − 窗口中心（步）"), ("relative_trigger", "推理起点步 − 触发步（步）"))
    handles = {}
    for axis, (name, label) in zip(axes, panels):
        axis.axvline(0, color=MUTED, linewidth=0.8)
        for model, item in summary["models"].items():
            rows = item["p_curves"][name]
            if not rows:
                continue
            title, color, marker = MODEL_STYLE[model]
            x = [row["bin_center"] for row in rows]
            axis.fill_between(x, [row["q25"] for row in rows], [row["q75"] for row in rows], color=color, alpha=0.18, linewidth=0)
            line, = axis.plot(x, [row["median"] for row in rows], color=color, marker=marker, markersize=4, linewidth=1.8, label=title)
            handles.setdefault(title, line)
        axis.set_xlabel(label, color=MUTED)
        axis.grid(color=GRID, linewidth=0.7)
        clean_axis(axis)
    axes[0].set_ylabel("想按比例 P（中位数与四分位）", color=MUTED)
    axes[0].set_ylim(-0.03, 1.03)
    if handles:
        figure.legend(handles=list(handles.values()), loc="upper center", ncol=len(handles), frameon=False, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path = Path(output) / "attention-p-curves.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def write_outputs(summary, root):
    output = Path(root) / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "attention-summary.json", summary)
    return [plot_groups(summary, output), plot_candidates(summary, output), plot_heads(summary, output), plot_curves(summary, output)]


def episode_heights(reader, records, z_hover, reach):
    """records 须含 infer_seq、chunk_start_env_step、raw_state、p_actions。到位前（起点 < reach_step）P 记 NaN。"""
    rows = {"infer_seq": [], "chunk_start_env_step": [], "state_z": [], "action_z": [], "p": [], "pre_reach": []}
    for record in records:
        state_z, action_z = convert_arrays(reader, {"raw_state": record["raw_state"], "actions": record["p_actions"]})
        pre = reach is None or record["chunk_start_env_step"] < reach
        rows["infer_seq"].append(record["infer_seq"])
        rows["chunk_start_env_step"].append(record["chunk_start_env_step"])
        rows["state_z"].append(state_z)
        rows["action_z"].append(action_z)
        rows["pre_reach"].append(pre)
        rows["p"].append(np.nan if pre else probabilities(action_z, z_hover)[P_KEY]["p"])
    return {name: np.asarray(values) for name, values in rows.items()}


def read_height_record(path):
    with np.load(path, allow_pickle=False) as archive:
        return {"infer_seq": int(scalar(archive["infer_seq"])),
                "chunk_start_env_step": int(scalar(archive["chunk_start_env_step"])),
                "raw_state": np.asarray(archive["raw_state"]), "p_actions": np.asarray(archive["p_actions"])}


def _heights_job(job):
    root, model, item, env_seed = job
    from scripts.interpret.read_sim_eef_pose import TcpReader, build_env

    key = item["episode_key"]
    records = [read_height_record(offline_path(root, model, item["nonce"], point["infer_seq"]))
               for point in item["inferences"]]
    env, seed, _ = build_env("StopCube", episode_number(key))
    if int(seed) != int(env_seed):
        env.close()
        raise ValueError(f"环境 seed 与窗口表不符：{key}")
    try:
        env.reset(seed=seed)
        arrays = episode_heights(TcpReader(env), records, float(item["z_hover"]), item["reach_step"])
    finally:
        env.close()
    path = heights_path(root, model, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, episode_key=key, model=model, z_hover=float(item["z_hover"]),
                        reach_step=-1 if item["reach_step"] is None else int(item["reach_step"]), **arrays)
    return str(path)


def heights(root, table_path=None, workers=4):
    align = load_align(root)
    table = load_table(table_path or align["table"]["path"])
    jobs = [(str(root), model, item, table[key]["env_seed"]) for model, episodes in align["models"].items()
            for key, item in episodes.items() if item.get("valid")]
    if workers == 1:
        written = [_heights_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            written = list(pool.map(_heights_job, jobs))
    print(f"HEIGHTS_DONE files={len(written)}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("heights", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True, help="产物根")
        command.add_argument("--table", type=Path, default=None, help="窗口表；缺省用 align.json 记录的路径")
    commands.choices["heights"].add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "heights":
        heights(args.root, args.table, args.workers)
        return 0
    summary, line, passed = summarize(args.root, args.table)
    write_outputs(summary, args.root)
    for problem in summary["problems"]:
        print(f"对齐问题：{problem}", flush=True)
    print(line, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
