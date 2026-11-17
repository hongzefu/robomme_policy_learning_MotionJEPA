"""RouteStick 注意力图 (a)(b)（计划 1117-routestick-progress-probe-attn 第一部分 §6、§7，第二部分 B1 第 5 行）。

只用 CPU。输入：
  * R1 的 rsprog_points.json（schema rsprog-points-1）：顶层 episodes[]（k、L、e0、demo_bounds = 示范第 j′=1..L 段的
    闭区间 [s, e]，H5 timestep，与 mem_t0/mem_t1 同一坐标）、顶层 points[]（k、u、j（0 起））、dropped[]（排除）。
    默认值即 R1 实际键名，--field 仅备用。
  * R2 的每局 npz（<replay-root>/<模型>/episode_<k>.npz）：u、attn_mem (N,18,M)、mem_type/mem_t0/mem_t1 (N,M)、
    mem_len (N,)、attn_main (N,18,3)（lang、image、action）、film_rel (N,18,2)（总量、motion 部分）。
    标量 memmod_maxabs（R2 replay 的 mem_mod_vec 一致性自检）读进 ATTN_AGG；任一局缺这个键即判 FAIL（memmod_maxabs=na）。
  * R1 的 scripts.interpret.rsprog_points.mem_token_segments(mem_type, mem_t0, mem_t1, demo_bounds, exec_start=e0)：
    返回逐 token 示范段号（1 起），0 = 执行帧 / 过渡窗，−1 = 补齐；逐点传入有效 token。

modulation 下记忆只经 MemoryAttention 进动作专家（§3、§6）：
  (a) 横轴 L0–L17，三块：① action←motion、action←frames：MemoryAttention 份额 ÷ 按 token 数均分份额；
      ② film_rel 两行（%）；③ attn_main 三行（% of reader's attention，平方根色标 0–70）。都按局等权：点内（R2 已对头、
      动作位、去噪步、噪声平均）→ 局内各点平均 → 局间平均。
  (b) 行 = 执行中的第 j 步（1 起），列 = 被读的示范段 j′，格子 = share ÷ uniform（层平均 L0–L17），帧流 single 权重
      （按 mem_token_segments 归段），motion 流 overlap 权重（窗与示范段有重叠就整窗计入该段，与 subgoal_attention 原图同口径）。
      share/uniform 的定义、D（对角富集）与 bootstrap 从 analysis/subgoal_attention.py 导入；每格至少 MIN_CELL_EPISODES 局才画。
判定行 ATTN_AGG：attn_mem 在有效 token 上和为 1、attn_main 三组和为 1（误差 ≤ 1e-4），R2 的 memmod 自检（每局必须有）≤ 1e-3，
画出的格子最少局数 ≥ 5。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.interpret.analysis.oracle_press_prob import bootstrap, write_json
from scripts.interpret.analysis.subgoal_attention import (BOOT_REPEATS, BOOT_SEED, D_NULL, D_THRESHOLD, MAIN_WEIGHTING,
                                                          align_matrix, diag_enrichment)

MODELS = ("A", "B")
MODEL_NAME = {"A": "MotionMem", "B": "Frozen Wan pooling"}
LAYERS = 18
STREAMS = ("motion", "frame")          # 子图列顺序：action ← motion、action ← frames
STREAM_TYPE = {"frame": 0, "motion": 1}
MIN_CELL_EPISODES = 5
ROWSUM_TOL, MEMMOD_TOL = 1e-4, 1e-3
MAIN_ROWS = ("lang", "image", "action")
# 默认即 R1 实际输出（schema rsprog-points-1）：顶层 episodes[]、顶层 points[]、dropped[]；demo_bounds 为示范第
# j′=1..L 段的闭区间 [s, e]（H5 timestep），e0 为执行起点。
DEFAULT_FIELDS = {"episodes": "episodes", "k": "k", "L": "L", "difficulty": "difficulty", "demo_bounds": "demo_bounds",
                  "e0": "e0", "points": "points", "u": "u", "j": "j", "dropped": "dropped"}


# ---------------------------------------------------------------- 读入

DEFAULT_SEGMENTER = "scripts.interpret.rsprog_points:mem_token_segments"


def default_segmenter(spec=DEFAULT_SEGMENTER):
    """按「模块:函数」导入归段函数，默认 R1 的 mem_token_segments；R1 未合入时导入失败，可用 --segmenter 指定替身。"""
    import importlib
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def load_points(path, fields=None):
    """每局：L、难度、示范段闭区间、e0 与 {u: j}；排除 dropped[] 里的局。"""
    f = {**DEFAULT_FIELDS, **(fields or {})}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    dropped = {int(entry[f["k"]]) for entry in data.get(f["dropped"], []) or []}
    out = {}
    for ep in data[f["episodes"]]:
        k = int(ep[f["k"]])
        if k in dropped:
            continue
        out[k] = {"k": k, "L": int(ep[f["L"]]), "difficulty": str(ep.get(f["difficulty"], "")),
                  "demo_bounds": [tuple(int(v) for v in pair) for pair in ep[f["demo_bounds"]]],
                  "e0": int(ep[f["e0"]]) if f["e0"] in ep else None, "j_of_u": {}}
    for point in data[f["points"]]:
        k = int(point[f["k"]])
        if k in out:
            out[k]["j_of_u"][int(point[f["u"]])] = int(point[f["j"]])
    return out


def load_episode(replay_root, pattern, model, k):
    path = Path(replay_root) / pattern.format(model=model, k=k)
    if not path.exists():
        return None
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


# ---------------------------------------------------------------- 逐点读数

def overlap_weights(t0, t1, demo_bounds):
    """motion 窗 [t0, t1] 与各示范段 [s, e]（都是闭区间）的重叠：有重叠的段权重 1（overlap 计权）。
    窗口截到 e0 之前（与 subgoal_attention.segments_of_window 同口径），示范段本就止于 e0 − 1。"""
    weights = np.zeros(len(demo_bounds))
    for index, (start, end) in enumerate(demo_bounds):
        if min(t1, end) - max(t0, start) + 1 > 0:
            weights[index] = 1.0
    return weights


def single_weights(segment, m):
    weights = np.zeros(m)
    if 0 <= segment < m:
        weights[segment] = 1.0
    return weights


def token_weights(mem_type, mem_t0, mem_t1, segments, demo_bounds, stream, weighting):
    """一条流的示范 token 下标与 (n, m) 归段权重；没有示范 token 时返回 (空, None)。"""
    m = len(demo_bounds)
    rows, index = [], []
    for i in np.flatnonzero(mem_type == STREAM_TYPE[stream]).tolist():
        if weighting == "overlap":
            w = overlap_weights(int(mem_t0[i]), int(mem_t1[i]), demo_bounds)
        else:
            w = single_weights(int(segments[i]), m)
        if w.any():
            rows.append(w)
            index.append(i)
    if not rows:
        return np.zeros(0, int), None
    return np.asarray(index), np.stack(rows)


def stream_share(mass, index, weights):
    """subgoal_attention.stream_shares 的同一算式：share(j′) = Σ w·mass ÷ Σ mass（示范 token 内），
    uniform(j′) = Σ w ÷ n。mass 为 (M,) 或 (层, M)。"""
    part = mass[..., index]
    total = part.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = (part @ weights) / total
    uniform = weights.sum(0) / len(index)
    return share, uniform


def point_readouts(ep, row, demo_bounds, segmenter, segments_base=1, exec_start=None):
    """一个推理点的 (a) 与 (b) 读数。"""
    n = int(ep["mem_len"][row])
    mem_type = ep["mem_type"][row, :n]
    t0, t1 = ep["mem_t0"][row, :n], ep["mem_t1"][row, :n]
    attn = ep["attn_mem"][row, :, :n].astype(np.float64)                 # (18, n)
    out = {"rowsum_err": float(np.abs(attn.sum(-1) - 1.0).max()),
           "main_err": float(np.abs(ep["attn_main"][row].astype(np.float64).sum(-1) - 1.0).max())}
    ratio = {}
    for stream, code in STREAM_TYPE.items():
        mask = mem_type == code
        count = int(mask.sum())
        ratio[stream] = attn[:, mask].sum(-1) / (count / n) if count else np.full(LAYERS, np.nan)
    out["ratio"] = ratio
    # R1：段号 1 起，0 = 执行帧 / 过渡窗，−1 = 补齐；减去起点后 0..L−1 为示范段，负数都不归段。
    segments = np.asarray(segmenter(mem_type, t0, t1, demo_bounds, exec_start=exec_start)).astype(int) - segments_base
    out["b"] = {}
    for stream in STREAMS:
        for weighting in ("single", "overlap") if stream == "motion" else ("single",):
            index, weights = token_weights(mem_type, t0, t1, segments, demo_bounds, stream, weighting)
            if weights is None:
                continue
            share_l, uniform = stream_share(attn, index, weights)
            share_avg, _ = stream_share(attn.mean(0), index, weights)
            if not np.isfinite(share_avg).all():
                continue
            out["b"][(stream, weighting)] = {"share": share_avg, "uniform": uniform, "share_l": share_l}
    return out


# ---------------------------------------------------------------- 逐局、跨局

def analyse_model(points, replay_root, pattern, model, segmenter, segments_base=1):
    """返回逐局读数列表与 ATTN_AGG 统计。"""
    episodes, rowsum, mainsum, memmod, points_used, memmod_missing = [], 0.0, 0.0, [], 0, 0
    for k, meta in sorted(points.items()):
        ep = load_episode(replay_root, pattern, model, k)
        if ep is None:
            continue
        if "memmod_maxabs" in ep:
            memmod.append(float(np.max(ep["memmod_maxabs"])))
        else:
            memmod_missing += 1
        m = len(meta["demo_bounds"])
        record = {"k": k, "L": meta["L"], "m": m, "difficulty": meta["difficulty"], "ratio": {s: [] for s in STREAM_TYPE},
                  "film": [], "main": [], "records": {}, "records_l": {}}
        for row, u in enumerate(ep["u"].tolist()):
            if int(u) not in meta["j_of_u"]:
                continue
            j = meta["j_of_u"][int(u)]
            readout = point_readouts(ep, row, meta["demo_bounds"], segmenter, segments_base, meta["e0"])
            points_used += 1
            rowsum, mainsum = max(rowsum, readout["rowsum_err"]), max(mainsum, readout["main_err"])
            for stream in STREAM_TYPE:
                record["ratio"][stream].append(readout["ratio"][stream])
            record["film"].append(ep["film_rel"][row].astype(np.float64))
            record["main"].append(ep["attn_main"][row].astype(np.float64))
            if not 0 <= j < m:
                continue
            for key, value in readout["b"].items():
                record["records"].setdefault(key, []).append({"j": j + 1, "share": value["share"], "uniform": value["uniform"]})
                record["records_l"].setdefault(key, []).append({"j": j + 1, "share_l": value["share_l"], "uniform": value["uniform"]})
        if not record["film"]:
            continue
        record["ratio"] = {s: np.nanmean(np.stack(v), axis=0) if np.isfinite(np.stack(v)).any() else np.full(LAYERS, np.nan)
                           for s, v in record["ratio"].items()}
        record["film"] = np.mean(np.stack(record["film"]), axis=0)          # (18, 2)
        record["main"] = np.mean(np.stack(record["main"]), axis=0)          # (18, 3)
        episodes.append(record)
    return episodes, {"rowsum_maxerr": max(rowsum, mainsum), "attn_mem_rowsum_maxerr": rowsum, "attn_main_sum_maxerr": mainsum,
                      "memmod_maxabs": max(memmod) if memmod else None, "memmod_missing_episodes": memmod_missing, "points": points_used, "episodes": len(episodes)}


def _episode_mean(arrays):
    stack = np.stack(arrays)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0) if np.isfinite(stack).any() else np.full(stack.shape[1:], np.nan)


def panel_a(episodes, repeats=BOOT_REPEATS, seed=BOOT_SEED):
    """(a) 三块的逐层值（局等权）与层平均的局级区间。"""
    blocks = {"ratio_motion": _episode_mean([e["ratio"]["motion"] for e in episodes]),
              "ratio_frame": _episode_mean([e["ratio"]["frame"] for e in episodes]),
              "film_total": 100 * _episode_mean([e["film"][:, 0] for e in episodes]),
              "film_motion": 100 * _episode_mean([e["film"][:, 1] for e in episodes])}
    for index, name in enumerate(MAIN_ROWS):
        blocks[f"main_{name}"] = 100 * _episode_mean([e["main"][:, index] for e in episodes])
    intervals = {}
    for key, getter in (("ratio_motion", lambda e: e["ratio"]["motion"]), ("ratio_frame", lambda e: e["ratio"]["frame"]),
                        ("film_total", lambda e: 100 * e["film"][:, 0]), ("film_motion", lambda e: 100 * e["film"][:, 1])):
        values = [float(np.nanmean(getter(e))) for e in episodes if np.isfinite(getter(e)).any()]
        intervals[key] = bootstrap(values, repeats, seed)
    return {"layers": {k: v.tolist() for k, v in blocks.items()}, "layer_mean": intervals, "n_episodes": len(episodes)}


def group_matrix(pairs, min_episodes=MIN_CELL_EPISODES):
    """绝对 (j, j′) 矩阵：每局先按 j 平均（align_matrix），跨局 nanmean；格子 = 平均 share ÷ 平均 uniform。
    局数少于 min_episodes 的格子记 NaN（不画）。"""
    size = max(m for _, m in pairs)
    share = np.full((len(pairs), size, size), np.nan)
    uniform = np.full((len(pairs), size, size), np.nan)
    for i, (records, m) in enumerate(pairs):
        matrix = align_matrix(records, m)
        share[i, :m, :m], uniform[i, :m, :m] = matrix["share"], matrix["uniform"]
    count = np.isfinite(share).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_share = np.nansum(share, 0) / np.maximum(count, 1)
        mean_uniform = np.nansum(uniform, 0) / np.maximum(count, 1)
        enrichment = mean_share / mean_uniform
    enrichment[count < min_episodes] = np.nan
    return {"enrichment": enrichment, "share": mean_share, "uniform": mean_uniform, "n_episodes_per_cell": count}


def panel_b(episodes, stream, weighting, repeats=BOOT_REPEATS, seed=BOOT_SEED, min_episodes=MIN_CELL_EPISODES):
    key = (stream, weighting)
    pairs = [(e["records"][key], e["m"]) for e in episodes if e["records"].get(key) and e["m"] >= 2]
    if not pairs:
        return None
    d_values = [diag_enrichment(records) for records, _ in pairs]
    d_stat = bootstrap([v for v in d_values if v is not None], repeats, seed)
    by_layer = []
    for layer in range(LAYERS):
        values = []
        for e in episodes:
            rows = e["records_l"].get(key)
            if rows and e["m"] >= 2:
                values.append(diag_enrichment([{"j": r["j"], "share": r["share_l"][layer], "uniform": r["uniform"]} for r in rows]))
        by_layer.append(bootstrap([v for v in values if v is not None], repeats, seed))
    matrix = group_matrix(pairs, min_episodes)
    drawn = matrix["n_episodes_per_cell"][np.isfinite(matrix["enrichment"])]
    return {"stream": stream, "weighting": weighting, "main": MAIN_WEIGHTING[stream] == weighting,
            "n_episodes": len(pairs), "D": d_stat, "D_threshold": D_THRESHOLD, "D_null": D_NULL,
            "D_ci_low_ge_threshold": bool(d_stat["ci95"] and d_stat["ci95"][0] >= D_THRESHOLD),
            "D_by_layer": by_layer, "matrix": matrix,
            "min_cell_eps": int(drawn.min()) if drawn.size else None,
            "blank_cells": int((~np.isfinite(matrix["enrichment"])).sum())}


def attn_agg_line(checks, panels_b):
    rowsum = max(c["rowsum_maxerr"] for c in checks.values())
    memmods = [c["memmod_maxabs"] for c in checks.values() if c["memmod_maxabs"] is not None]
    missing = sum(c.get("memmod_missing_episodes", 0) for c in checks.values())
    cells = [p["min_cell_eps"] for model in panels_b.values() for p in model.values() if p and p["main"] and p["min_cell_eps"] is not None]
    min_cell = min(cells) if cells else None
    memmod_ok = bool(memmods) and missing == 0 and max(memmods) <= MEMMOD_TOL     # 缺自检即 FAIL，不静默放行
    ok = rowsum <= ROWSUM_TOL and memmod_ok and min_cell is not None and min_cell >= MIN_CELL_EPISODES
    memmod_text = f"{max(memmods):.2e}" if memmods and missing == 0 else "na"
    return (f"ATTN_AGG={'PASS' if ok else 'FAIL'} rowsum_maxerr={rowsum:.2e} memmod_maxabs={memmod_text} "
            f"min_cell_eps={min_cell if min_cell is not None else 'na'}")


# ---------------------------------------------------------------- 出图（版式沿用 mock/mock_attn.py）

GREY = "#555555"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
    return plt


def _colormaps():
    from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm, TwoSlopeNorm
    div = LinearSegmentedColormap.from_list("bwr_soft", ["#3b7fd9", "#eeeeee", "#e5484d"])
    div.set_bad("white")
    return div, TwoSlopeNorm, PowerNorm, Normalize


def _heat(ax, data, cmap, norm):
    ax.pcolormesh(np.ma.masked_invalid(data), cmap=cmap, norm=norm, edgecolors="white", linewidth=1.2)
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(data.shape[0], 0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=GREY)


def _cbar(fig, rect, cmap, norm, ticks, label, size=7.5):
    from matplotlib.cm import ScalarMappable
    cax = fig.add_axes(rect)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=ticks)
    cb.ax.tick_params(labelsize=size, colors=GREY, length=2.5)
    cb.set_label(label, fontsize=size, color=GREY, labelpad=4, linespacing=1.1)
    cb.outline.set_linewidth(0.8)
    return cb


def _nice_max(values, floor):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    top = float(finite.max()) if finite.size else floor
    return max(floor, float(np.ceil(top / 5.0) * 5.0))


def plot_panel_a(a, model, output, stem):
    plt = _plt()
    div, TwoSlopeNorm, PowerNorm, Normalize = _colormaps()
    layers = a["layers"]
    blk1 = np.vstack([layers["ratio_motion"], layers["ratio_frame"]])
    blk2 = np.vstack([layers["film_total"], layers["film_motion"]])
    blk3 = np.vstack([layers[f"main_{name}"] for name in MAIN_ROWS])
    ratio_max = max(3.0, float(np.ceil(np.nanmax(blk1)))) if np.isfinite(blk1).any() else 3.0
    film_max = _nice_max(blk2, 10.0)
    blocks = [("action reads memory (MemoryAttention)", ["action ← motion", "action ← frames"], blk1, div,
               TwoSlopeNorm(vmin=0, vcenter=1, vmax=ratio_max), list(range(0, int(ratio_max) + 1)), "share ÷\nuniform"),
              ("memory's effect on action (FiLM)", ["memory → action", "of which motion"], blk2, "Purples",
               Normalize(0, film_max), [0, film_max / 2, film_max], "FiLM rel.\nchange (%)"),
              ("action reads prefix (main attention)", ["action ← lang", "action ← image", "action ← action"], blk3, "Blues",
               PowerNorm(0.5, vmin=0, vmax=70), [0, 5, 20, 40, 70], "% of reader's\nattention")]
    row_in, gap_in, top_in, bottom_in = 0.27, 0.40, 0.62, 0.85     # 版式按英寸排，避免留白随行数漂移
    fig_h = top_in + row_in * sum(len(b[1]) for b in blocks) + gap_in * (len(blocks) - 1) + bottom_in
    fig = plt.figure(figsize=(6.5, fig_h))
    x0, width, cbx, cbw = 0.285, 0.535, 0.855, 0.022
    row_h = row_in / fig_h
    y = 1 - top_in / fig_h
    for i, (title, rows, data, cmap, norm, ticks, label) in enumerate(blocks):
        h = row_h * len(rows)
        ax = fig.add_axes([x0, y - h, width, h])
        _heat(ax, data, cmap, norm)
        ax.set_yticks(np.arange(len(rows)) + 0.5)
        ax.set_yticklabels(rows, fontsize=9.5, color="#222222")
        if rows[1].startswith("of which"):
            ax.get_yticklabels()[1].set_color(GREY)
            ax.get_yticklabels()[1].set_style("italic")
        ax.text(0, 1.0, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color="#777777")
        if i == len(blocks) - 1:
            ax.set_xticks([0.5, 5.5, 10.5, 15.5])
            ax.set_xticklabels(["L0", "L5", "L10", "L15"], fontsize=9, color=GREY)
            ax.set_xlabel("layer", fontsize=9.5, color=GREY, labelpad=3)
            bottom = y - h
        else:
            ax.set_xticks([])
        _cbar(fig, [cbx, y - h, cbw, h], cmap, norm, ticks, label)
        y -= h + gap_in / fig_h
    fig.text(0.02, 1 - 0.22 / fig_h, "(a)", fontsize=13, fontweight="bold", va="center")
    fig.text(x0, 1 - 0.22 / fig_h, f"{MODEL_NAME.get(model, model)}  (n = {a['n_episodes']} episodes)", fontsize=10, va="center",
             color="#222222")
    fig.text(x0 + width / 2, bottom - 0.66 / fig_h, "frames←motion, lang←motion, lang←frames: zero by construction under modulation "
             "(not shown)", ha="center", va="center", fontsize=7, color="#777777", style="italic")
    return _save(fig, plt, output, stem)


def plot_panel_b(panels, output, stem):
    """panels[model][stream] = panel_b 结果（主计权）；两行（模型）× 两列（流）。"""
    plt = _plt()
    div, TwoSlopeNorm, _, _ = _colormaps()
    models = [m for m in MODELS if m in panels] or list(panels)
    size = max((p["matrix"]["enrichment"].shape[0] for model in panels.values() for p in model.values() if p), default=1)
    vmax = 3.0
    for model in panels.values():
        for p in model.values():
            if p is not None and np.isfinite(p["matrix"]["enrichment"]).any():
                vmax = max(vmax, float(np.ceil(np.nanmax(p["matrix"]["enrichment"]))))
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax)
    fig_w, cell_w = 6.5, 0.29
    mat_in, title_in, gap_in, top_in, bottom_in = cell_w * 6.5, 0.48, 0.22, 0.30, 0.95
    fig_h = top_in + len(models) * (title_in + mat_in) + (len(models) - 1) * gap_in + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    x0, width = 0.20, 0.62
    cell_h = mat_in / fig_h
    gap = width - 2 * cell_w
    top = 1 - top_in / fig_h
    for r, model in enumerate(models):
        y0 = top - ((r + 1) * (title_in + mat_in) + r * gap_in) / fig_h
        for c, stream in enumerate(STREAMS):
            panel = panels[model].get(stream)
            ax = fig.add_axes([x0 + c * (cell_w + gap), y0, cell_w, cell_h])
            data = np.full((size, size), np.nan)
            if panel is not None:
                matrix = panel["matrix"]["enrichment"]
                data[:matrix.shape[0], :matrix.shape[1]] = matrix
            _heat(ax, data, div, norm)
            source = "motion" if stream == "motion" else "frames"
            if panel is not None and panel["D"]["mean"] is not None:
                low, high = panel["D"]["ci95"]
                dtext = f"$D$ = {panel['D']['mean']:.2f} [{low:.2f}, {high:.2f}]"
            else:
                dtext = "$D$ = n/a"
            ax.set_title((f"action ← {source}\n" if r == 0 else "") + dtext, loc="left", fontsize=9, color="#222222", pad=4,
                         linespacing=1.15)
            step = 1 if size <= 10 else 2
            if c == 0:
                ax.set_yticks(np.arange(0, size, step) + 0.5)
                ax.set_yticklabels(np.arange(1, size + 1, step), fontsize=8)
                ax.set_ylabel(f"{MODEL_NAME.get(model, model)}\nexecuting step $j$", fontsize=9, color="#222222", labelpad=5,
                              linespacing=1.1)
            else:
                ax.set_yticks([])
            if r == len(models) - 1:
                ax.set_xticks(np.arange(0, size, step) + 0.5)
                ax.set_xticklabels(np.arange(1, size + 1, step), fontsize=8)
            else:
                ax.set_xticks([])
            last_bottom = y0
    fig.text(x0 + width / 2, last_bottom - 0.42 / fig_h, "demo segment $j'$ read", ha="center", va="center", fontsize=9.5,
             color=GREY)
    fig.text(x0 + width / 2, last_bottom - 0.70 / fig_h,
             f"MemoryAttention share ÷ uniform, averaged over L0–L17; blank = fewer than {MIN_CELL_EPISODES} episodes",
             ha="center", va="center", fontsize=7, color="#777777", style="italic")
    ticks = sorted({0.0, 1.0, 2.0, vmax})
    bar_h = min(0.5 * (top - last_bottom), 3.0 / fig_h)
    _cbar(fig, [0.87, last_bottom + (top - last_bottom - bar_h) / 2, 0.022, bar_h], div, norm, ticks, "share ÷ uniform", size=8.5)
    fig.text(0.02, 1 - 0.2 / fig_h, "(b)", fontsize=13, fontweight="bold", va="center")
    return _save(fig, plt, output, stem)


def _save(fig, plt, output, stem):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=200 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------- 入口

def _jsonable(value):
    if isinstance(value, dict):
        return {(" / ".join(k) if isinstance(k, tuple) else str(k)): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def run(points_path, replay_root, output, models=MODELS, pattern="{model}/episode_{k}.npz", fields=None, segmenter=None,
        segments_base=1, repeats=BOOT_REPEATS, min_episodes=MIN_CELL_EPISODES, plot=True):
    segmenter = segmenter or default_segmenter()
    points = load_points(points_path, fields)
    summary = {"plan": "1117-routestick-progress-probe-attn", "repeats": repeats, "min_cell_episodes": min_episodes,
               "weighting": dict(MAIN_WEIGHTING), "panel_a": {}, "panel_b": {}, "checks": {}}
    panels_main, figures = {}, []
    for model in models:
        episodes, check = analyse_model(points, replay_root, pattern, model, segmenter, segments_base)
        if not episodes:
            print(f"RSPROG_ATTN 模型 {model} 找不到回放局，跳过", flush=True)
            continue
        summary["checks"][model] = check
        summary["panel_a"][model] = panel_a(episodes, repeats)
        panels = {}
        for stream in STREAMS:
            for weighting in ("single", "overlap") if stream == "motion" else ("single",):
                panels[f"{stream}:{weighting}"] = panel_b(episodes, stream, weighting, repeats, BOOT_SEED, min_episodes)
        summary["panel_b"][model] = panels
        panels_main[model] = {stream: panels[f"{stream}:{MAIN_WEIGHTING[stream]}"] for stream in STREAMS}
        print(f"RSPROG_ATTN_MODEL model={model} episodes={check['episodes']} points={check['points']}", flush=True)
    if not summary["checks"]:
        raise SystemExit("找不到任何回放局")
    line = attn_agg_line(summary["checks"], panels_main)
    summary["verdict"] = line
    if plot:
        for model in summary["panel_a"]:
            stem = "attn-a-routestick" if model == "A" else f"attn-a-routestick-{model}"
            figures += plot_panel_a(summary["panel_a"][model], model, output, stem)
        figures += plot_panel_b(panels_main, output, "attn-b-routestick")
    summary["figures"] = [str(p) for p in figures]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "rsprog-attn-summary.json", _jsonable(summary))
    for model, panels in panels_main.items():
        for stream, panel in panels.items():
            if panel is not None and panel["D"]["mean"] is not None:
                print(f"RSPROG_ATTN_D model={model} stream={stream} D={panel['D']['mean']:.3f} "
                      f"ci95=[{panel['D']['ci95'][0]:.3f},{panel['D']['ci95'][1]:.3f}] n={panel['n_episodes']}", flush=True)
    print(line, flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--points", required=True, help="R1 的 rsprog_points.json")
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--npz-pattern", default="{model}/episode_{k}.npz")
    parser.add_argument("--field", action="append", help="字段名覆盖，如 demo_bounds=demo_segments；可重复")
    parser.add_argument("--segmenter", default=DEFAULT_SEGMENTER, help="归段函数「模块:函数」")
    parser.add_argument("--segments-base", type=int, default=1, help="mem_token_segments 返回段号的起点（R1 为 1）")
    parser.add_argument("--repeats", type=int, default=BOOT_REPEATS)
    parser.add_argument("--min-cell-episodes", type=int, default=MIN_CELL_EPISODES)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    fields = {}
    for item in args.field or []:
        name, _, value = item.partition("=")
        if not value or name not in DEFAULT_FIELDS:
            raise SystemExit(f"--field 格式为 名=json键，名取自 {sorted(DEFAULT_FIELDS)}：{item}")
        fields[name] = value
    run(args.points, args.replay_root, args.output, tuple(args.models), args.npz_pattern, fields,
        segmenter=default_segmenter(args.segmenter), segments_base=args.segments_base, repeats=args.repeats,
        min_episodes=args.min_cell_episodes, plot=not args.no_plot)


if __name__ == "__main__":
    main()
