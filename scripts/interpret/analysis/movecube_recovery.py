"""MoveCube 强推恢复实验的分析与验收（1107 计划第二部分 §2.4）。

主环境 CPU，只用 numpy、标准库与 matplotlib。只读 §2.2 定义的回执与逐步轨迹，不导入 runner 或 run_local。

输入目录约定：<root>/<model>/k<k>/<shard>/ 下递归有 traces/<nonce>/meta/force_receipt.json 与 commands.json。

结局分类（每局恰属一类）：
  error    outcome 为 None，或字段不属于下面三类；不进成功率分母，计入 MFR_RUN 的 errors。
  grasp    grasp_success=1。此时 outcome 的 success 只是 runner 提前结束的记号，不是任务成功。
  fail     grasp_success=0 且 outcome=fail。
  timeout  grasp_success=0 且 outcome=timeout。
reached_release 为假的局不进放手核对，但照样按上面分类进分母（计划口径 7）。

判据只验实验是否按设计执行，恢复率高低是结果不是验收条件。任一判定行 FAIL 时产物照常写出，进程退出码为 1。

⚠ 同一 (模型, k, 局) 出现多份回执时取路径排序第一份，其余每份记 1 个 errors：
不猜哪份有效，基础设施重试后要由主会话把作废的那份移出产物根再重跑本脚本。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np


MODELS = ("mot", "ctx")
MODEL_LABEL = {"mot": "motion", "ctx": "无 motion 基线"}
ALL_K = (0, 16, 32, 48, 64, 80)
DIRECTIONS = ("A", "B")
# 方向 → (正确物体, 强推目标)，对应轨迹字段名。
TARGETS = {"A": ("cube", "peg_tail"), "B": ("peg_tail", "cube")}
DIRECTION_TITLE = {"A": "方向 A：强推到 peg 上方后，抓起 cube", "B": "方向 B：强推到 cube 上方后，抓起 peg"}
ACTION_TOL = 1e-6
Z95 = statistics.NormalDist().inv_cdf(0.975)
MAX_DETAILS = 50

# 配色取 dataviz 参考调色板的前两个分类色，已用其校验脚本在 #fcfcfb 底上验过：CVD ΔE 24.7，常视 ΔE 33.6。
SERIES_COLOR = {"mot": "#2a78d6", "ctx": "#eb6834"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e2"
CJK_FAMILIES = ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback", "WenQuanYi Zen Hei")


def wilson(successes, total):
    """Wilson 95% 区间；total=0 时返回 (None, None)。"""
    if total == 0:
        return None, None
    p = successes / total
    denom = 1 + Z95 ** 2 / total
    centre = (p + Z95 ** 2 / (2 * total)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / total + Z95 ** 2 / (4 * total ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(only_a, only_b):
    """精确 McNemar 双侧 p：不一致对在 H0 下服从 Binomial(n, 1/2)。用整数算，n 再大也没有浮点累积。"""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def classify(receipt):
    outcome = receipt.get("outcome")
    if outcome is None:
        return "error"
    if receipt.get("grasp_success") == 1:
        return "grasp"
    return outcome if outcome in ("fail", "timeout") else "error"


def _episode_sort_key(key):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(key))]


def load_table(path):
    data = json.loads(Path(path).read_text())
    episodes = {}
    for key, item in data["episodes"].items():
        actions = np.asarray(item["actions"], dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise SystemExit(f"动作表 {key} 的 actions 形状应为 (T, 8)，实际 {actions.shape}")
        if item["direction"] not in TARGETS:
            raise SystemExit(f"动作表 {key} 的 direction 非法：{item['direction']}")
        episodes[key] = {"actions": actions, "first_env_step": item["first_env_step"], "way": item["way"],
                         "direction": item["direction"]}
    return episodes


def discover(root):
    """找全部回执。返回 (可用记录, 问题说明)。路径或 JSON 不合约定的回执记为问题，不静默跳过。"""
    records, problems = [], []
    root = Path(root)
    for path in sorted(root.rglob("traces/*/meta/force_receipt.json")):
        parts = path.relative_to(root).parts
        match = re.fullmatch(r"k(\d+)", parts[1]) if len(parts) > 2 else None
        if match is None or parts[0] not in MODELS:
            problems.append(f"{path}: 路径不符合 <model>/k<k>/<shard>/")
            continue
        try:
            receipt = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"{path}: 回执读不了：{error}")
            continue
        if receipt.get("episode_key") is None:
            problems.append(f"{path}: 回执没有 episode_key")
            continue
        records.append({"model": parts[0], "k": int(match.group(1)), "shard": parts[2], "path": path,
                        "run_root": root.joinpath(*parts[:3]), "receipt": receipt, "key": receipt["episode_key"]})
    return records, problems


def _trace_file(record):
    """trace_path 允许绝对路径，也允许相对回执、nonce 目录、分片运行根。"""
    given = record["receipt"].get("trace_path")
    if not given:
        return None
    given = Path(given)
    if given.is_absolute():
        return given if given.is_file() else None
    meta = record["path"].parent
    for base in (meta, meta.parent, record["run_root"]):
        if (base / given).is_file():
            return base / given
    return None


def read_trace(record):
    """返回轨迹行；文件缺失或有坏行返回 None，由调用方记 mismatch。"""
    path = _trace_file(record)
    if path is None:
        return None
    try:
        with open(path) as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None


def _actions_equal(a, b):
    try:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        return False
    # 要求差值确实 ≤ 容差，NaN 因此判不等。
    return a.shape == b.shape and a.size > 0 and bool(np.abs(a - b).max() <= ACTION_TOL)


def check_exec(record, rows, item):
    """单局的 FORCE_EXEC 核对，返回问题说明列表（一条记一个 mismatch）。"""
    label = f"{record['model']} k{record['k']} {record['key']}"
    receipt, k = record["receipt"], record["k"]
    issues = []
    if receipt.get("force_k") != k:
        issues.append(f"{label}: 回执 force_k={receipt.get('force_k')} 与目录 k 不符")
    origin = receipt.get("origin_env_step")
    if origin != item["first_env_step"]:
        issues.append(f"{label}: origin_env_step={origin} 与表 first_env_step={item['first_env_step']} 不符")
    if receipt.get("direction") not in (None, item["direction"]) or receipt.get("way") not in (None, item["way"]):
        issues.append(f"{label}: 回执方向或 way 与表不符")
    if rows is None:
        return issues + [f"{label}: step_trace 缺失或有坏行"]
    forced = [row for row in rows if row.get("control_source") == "forced"]
    unknown = sum(row.get("control_source") not in ("forced", "policy") for row in rows)
    if unknown:
        issues.append(f"{label}: {unknown} 行 control_source 既非 forced 也非 policy")
    expected_count = min(k, len(rows))
    if len(forced) != expected_count:
        issues.append(f"{label}: forced 步数 {len(forced)}，应为 min(k, 实际步数)={expected_count}")
    if not isinstance(origin, int):
        return issues
    for position, row in enumerate(forced):
        offset = row.get("env_step", -1) - origin
        if offset != position:
            issues.append(f"{label}: 第 {position} 个 forced 行在偏移 {offset}，强推应从 origin 起连续")
        if not 0 <= offset < len(item["actions"]) or not _actions_equal(row.get("executed_action"), item["actions"][offset]):
            issues.append(f"{label}: env_step={row.get('env_step')} 的动作与表不符")
    return issues


def check_release(record, rows):
    """单局的 FORCE_RELEASE 核对。调用方只对 reached_release 为真的局调用。"""
    label = f"{record['model']} k{record['k']} {record['key']}"
    receipt = record["receipt"]
    issues = []
    origin = receipt.get("origin_env_step")
    release = None if origin is None else origin + record["k"]
    if release is None or receipt.get("actual_release_step") != release:
        issues.append(f"{label}: actual_release_step={receipt.get('actual_release_step')}，应为 origin+k={release}")
    if rows is None:
        return issues + [f"{label}: step_trace 缺失或有坏行，无法核对放手后控制权"]
    if release is not None:
        late = sum(row.get("control_source") == "forced" and row.get("env_step", -1) >= release for row in rows)
        if late:
            issues.append(f"{label}: 放手步之后仍有 {late} 个 forced 行")
    return issues


def _forced_pair_differs(a, b):
    if len(a) != len(b):
        return True
    return any(x[0] != y[0] or not _actions_equal(x[1], y[1]) for x, y in zip(a, b))


def post_release_distances(receipt, rows, k, direction):
    """放手后 TCP 到正确物体、强推目标的水平最小距离；没到放手步或字段不全返回 (None, None)。"""
    origin = receipt.get("origin_env_step")
    if not receipt.get("reached_release") or rows is None or origin is None:
        return None, None
    after = [row for row in rows if row.get("env_step", -1) >= origin + k]
    if not after:
        return None, None
    try:
        tcp = np.array([row["tcp"][:2] for row in after], dtype=float)
        return tuple(float(np.linalg.norm(tcp - np.array([row[name][:2] for row in after], dtype=float), axis=1).min())
                     for name in TARGETS[direction])
    except (KeyError, TypeError, ValueError):
        return None, None


def check_no_attn(root):
    """产物根无 npz、无 attn 目录，且每份 commands.json 的 server 都是官方 serve_policy。"""
    root = Path(root)
    npz = sum(1 for _ in root.rglob("*.npz"))
    attn_dirs = [str(p) for p in root.rglob("attn") if p.is_dir()]
    bad, commands = [], 0
    for path in sorted(root.rglob("commands.json")):
        commands += 1
        try:
            server = json.loads(path.read_text())["server"]
            joined = " ".join(str(token) for token in server) if isinstance(server, list) else None
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            joined = None
        if joined is None or "scripts.training.serve_policy" not in joined or "serve_policy_attn" in joined or "--attn-out" in joined:
            bad.append(str(path))
    # 一份 commands.json 都没有时无从核对服务模块，不能算过。
    return {"pass": npz == 0 and not attn_dirs and not bad and commands > 0, "npz": npz, "attn_dirs": attn_dirs,
            "commands": commands, "bad_commands": bad}


def _median(values):
    values = [value for value in values if value is not None]
    return (float(statistics.median(values)), len(values)) if values else (None, 0)


def analyze(root, table_path, expected_episodes=33, expected_k=ALL_K):
    table = load_table(table_path)
    expected_k = tuple(expected_k)
    records, problems = discover(root)
    errors = list(problems)

    # 期望局集：全量时就是表里全部局；冒烟时取回执里实际出现的前 n 局，不足的差额记 missing。
    observed = {record["key"] for record in records if record["key"] in table}
    if expected_episodes >= len(table):
        episodes = sorted(table, key=_episode_sort_key)
    else:
        episodes = sorted(observed, key=_episode_sort_key)[:expected_episodes]
    expected_set = set(episodes)
    shortfall = max(0, expected_episodes - len(episodes))

    cells = {}
    for record in records:
        cell = (record["model"], record["k"], record["key"])
        if record["key"] not in expected_set or record["k"] not in expected_k:
            errors.append(f"{record['path']}: 不在期望的局或 k 内")
        elif cell in cells:
            errors.append(f"{record['path']}: {cell} 重复回执，已用 {cells[cell]['path']}")
        else:
            cells[cell] = record

    exec_issues, release_issues, reached = [], [], 0
    forced_by_cell = {}
    for record in cells.values():
        rows = read_trace(record)
        item = table[record["key"]]
        receipt = record["receipt"]
        exec_issues += check_exec(record, rows, item)
        if receipt.get("reached_release"):
            reached += 1
            release_issues += check_release(record, rows)
        forced_by_cell[(record["model"], record["k"], record["key"])] = [
            (row.get("env_step"), row.get("executed_action")) for row in rows or [] if row.get("control_source") == "forced"]
        record["direction"] = item["direction"]
        record["class"] = classify(receipt)
        record["dist_correct"], record["dist_forced"] = post_release_distances(receipt, rows, record["k"], item["direction"])
        origin, grasp_step = receipt.get("origin_env_step"), receipt.get("grasp_step")
        grasped_after = record["class"] == "grasp" and receipt.get("reached_release") and None not in (origin, grasp_step)
        record["release_to_grasp"] = grasp_step - (origin + record["k"]) if grasped_after else None
        if record["class"] == "error":
            errors.append(f"{record['path']}: 结局记为 error（outcome={receipt.get('outcome')}，grasp_success={receipt.get('grasp_success')}）")
    for k in expected_k:
        for key in episodes:
            pair = [forced_by_cell.get((model, k, key)) for model in MODELS]
            if None not in pair and _forced_pair_differs(*pair):
                exec_issues.append(f"k{k} {key}: 两模型的 forced 序列不一致")

    expected_cells = {(model, k, key) for model in MODELS for k in expected_k for key in episodes}
    missing_cells = sorted(expected_cells - set(cells), key=lambda c: (c[0], c[1], _episode_sort_key(c[2])))
    missing = len(missing_cells) + shortfall * len(expected_k) * len(MODELS)
    completed = sum(record["class"] != "error" for record in cells.values())

    groups, pairs = [], []
    for direction in DIRECTIONS:
        direction_eps = [key for key in episodes if table[key]["direction"] == direction]
        if not direction_eps:
            continue
        for model in MODELS:
            baseline = {key for key in direction_eps
                        if (rec := cells.get((model, 0, key))) is not None and rec["class"] == "grasp"}
            for k in expected_k:
                recs = [cells[(model, k, key)] for key in direction_eps if (model, k, key) in cells]
                valid = [rec for rec in recs if rec["class"] != "error"]
                grasp = sum(rec["class"] == "grasp" for rec in valid)
                low, high = wilson(grasp, len(valid))
                among = [rec for rec in valid if rec["key"] in baseline]
                among_grasp = sum(rec["class"] == "grasp" for rec in among)
                among_low, among_high = wilson(among_grasp, len(among))
                dist_correct, dist_n = _median([rec["dist_correct"] for rec in valid])
                dist_forced, _ = _median([rec["dist_forced"] for rec in valid])
                steps, steps_n = _median([rec["release_to_grasp"] for rec in valid])
                groups.append({
                    "direction": direction, "model": model, "k": k, "total": len(valid), "grasp_success": grasp,
                    "fail": sum(rec["class"] == "fail" for rec in valid),
                    "timeout": sum(rec["class"] == "timeout" for rec in valid),
                    "errors": len(recs) - len(valid),
                    "grasp_before_release": sum(rec["class"] == "grasp" and not rec["receipt"].get("reached_release") for rec in valid),
                    "grasp_rate": grasp / len(valid) if valid else None, "wilson_low": low, "wilson_high": high,
                    "recovery_baseline_total": len(among), "recovery_grasp": among_grasp,
                    "recovery_rate": among_grasp / len(among) if among else None,
                    "recovery_wilson_low": among_low, "recovery_wilson_high": among_high,
                    "median_min_dist_correct": dist_correct, "median_min_dist_forced": dist_forced, "n_dist": dist_n,
                    "median_release_to_grasp": steps, "n_release_to_grasp": steps_n,
                })
        for k in expected_k:
            both = [(cells[("mot", k, key)], cells[("ctx", k, key)]) for key in direction_eps
                    if ("mot", k, key) in cells and ("ctx", k, key) in cells
                    and cells[("mot", k, key)]["class"] != "error" and cells[("ctx", k, key)]["class"] != "error"]
            only_mot = sum(a["class"] == "grasp" and b["class"] != "grasp" for a, b in both)
            only_ctx = sum(b["class"] == "grasp" and a["class"] != "grasp" for a, b in both)
            pairs.append({"direction": direction, "k": k, "pairs": len(both), "only_mot": only_mot, "only_ctx": only_ctx,
                          "both": sum(a["class"] == "grasp" and b["class"] == "grasp" for a, b in both),
                          "mcnemar_p": mcnemar_exact(only_mot, only_ctx)})

    no_attn = check_no_attn(root)
    verdicts = {
        "FORCE_EXEC": {"pass": not exec_issues, "runs": len(cells), "mismatches": len(exec_issues), "details": exec_issues[:MAX_DETAILS]},
        "FORCE_RELEASE": {"pass": not release_issues, "reached_release": reached, "mismatches": len(release_issues),
                          "details": release_issues[:MAX_DETAILS]},
        "NO_ATTN": no_attn,
        "MFR_RUN": {"pass": not missing and not errors and completed == expected_episodes * len(expected_k) * len(MODELS),
                    "completed": completed, "missing": missing, "errors": len(errors),
                    "missing_cells": [list(cell) for cell in missing_cells[:MAX_DETAILS]], "details": errors[:MAX_DETAILS]},
    }
    return {"root": str(root), "force_table": str(table_path),
            "expected": {"episodes": expected_episodes, "k": list(expected_k), "models": list(MODELS)},
            "episodes": episodes, "verdicts": verdicts, "groups": groups, "mcnemar": pairs, "warnings": [],
            "per_episode": [_csv_row(cells[cell]) for cell in sorted(cells, key=lambda c: (c[0], c[1], _episode_sort_key(c[2])))]}


CSV_FIELDS = ("direction", "model", "k", "shard", "episode_key", "episode_id", "way", "class", "grasp_success", "grasp_step",
              "outcome", "env_status", "origin_env_step", "reached_release", "actual_release_step", "actions_executed",
              "release_to_grasp", "min_dist_correct", "min_dist_forced", "receipt")


def _csv_row(record):
    receipt = record["receipt"]
    row = {name: receipt.get(name) for name in CSV_FIELDS}
    row.update({"direction": record["direction"], "model": record["model"], "k": record["k"], "shard": record["shard"],
                "episode_key": record["key"], "class": record["class"], "release_to_grasp": record["release_to_grasp"],
                "min_dist_correct": record["dist_correct"], "min_dist_forced": record["dist_forced"], "receipt": str(record["path"])})
    return row


def verdict_lines(report):
    v = report["verdicts"]
    word = lambda item: "PASS" if item["pass"] else "FAIL"
    lines = [
        f"FORCE_EXEC={word(v['FORCE_EXEC'])} runs={v['FORCE_EXEC']['runs']} mismatches={v['FORCE_EXEC']['mismatches']}",
        f"FORCE_RELEASE={word(v['FORCE_RELEASE'])} reached_release={v['FORCE_RELEASE']['reached_release']} mismatches={v['FORCE_RELEASE']['mismatches']}",
        f"NO_ATTN={word(v['NO_ATTN'])} npz={v['NO_ATTN']['npz']}",
        f"MFR_RUN={word(v['MFR_RUN'])} completed={v['MFR_RUN']['completed']} missing={v['MFR_RUN']['missing']} errors={v['MFR_RUN']['errors']}",
    ]
    lines += [f"MFR_RESULT direction={g['direction']} model={g['model']} k={g['k']} grasp_success={g['grasp_success']}/{g['total']}"
              for g in report["groups"]]
    return lines


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def summary_text(report):
    lines = [f"产物根：{report['root']}", f"动作表：{report['force_table']}",
             f"期望规模：{report['expected']['episodes']} 局 × k={report['expected']['k']} × 模型 {report['expected']['models']}", ""]
    lines.append("抓取结局（方向 模型 k：成功/失败/超时/错误，抓取成功率 [Wilson 95%]；k=0 抓取成功局的恢复率；"
                 "放手后 TCP 水平最小距离中位数 正确物体/强推目标（m）；放手到抓起步数中位数）")
    for g in report["groups"]:
        rate = "—" if g["grasp_rate"] is None else f"{g['grasp_rate'] * 100:.1f}% [{g['wilson_low'] * 100:.1f}, {g['wilson_high'] * 100:.1f}]"
        recovery = f"{g['recovery_grasp']}/{g['recovery_baseline_total']}"
        lines.append(f"{g['direction']} {g['model']} k={g['k']:>2}：{g['grasp_success']}/{g['fail']}/{g['timeout']}/{g['errors']}，{rate}；"
                     f"恢复 {recovery}；距离 {_fmt(g['median_min_dist_correct'])}/{_fmt(g['median_min_dist_forced'])}；"
                     f"步数 {_fmt(g['median_release_to_grasp'], 1)}")
    lines += ["", "同局同 k 两模型配对（精确 McNemar）："]
    lines += [f"{p['direction']} k={p['k']:>2}：配对 {p['pairs']}，都抓起 {p['both']}，仅 mot {p['only_mot']}，仅 ctx {p['only_ctx']}，p={p['mcnemar_p']:.4f}"
              for p in report["mcnemar"]]
    for name, item in report["verdicts"].items():
        details = item.get("details") or item.get("bad_commands") or []
        if details:
            lines += ["", f"{name} 明细（至多 {MAX_DETAILS} 条）："] + [f"  {text}" for text in details]
    if report["verdicts"]["NO_ATTN"]["attn_dirs"]:
        lines += ["", "NO_ATTN 发现 attn 目录："] + [f"  {text}" for text in report["verdicts"]["NO_ATTN"]["attn_dirs"]]
    if report["warnings"]:
        lines += ["", "警告："] + [f"  {text}" for text in report["warnings"]]
    return "\n".join(lines + ["", *verdict_lines(report)]) + "\n"


def _pyplot():
    """返回 (plt, 字体警告或 None)。与 subgoal_attention 同法按 fc-list 注册字体，不导入该模块以免带上副作用。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        listing = subprocess.run(["fc-list", ":lang=zh", "file"], capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        listing = ""
    for line in listing.splitlines():
        if line.strip():
            try:
                font_manager.fontManager.addfont(line.split(":")[0].strip())
            except (OSError, RuntimeError):
                pass
    available = {font.name for font in font_manager.fontManager.ttflist}
    families = [name for name in CJK_FAMILIES if name in available]
    plt.rcParams["font.family"] = families + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, None if families else f"未找到中文字体（{'、'.join(CJK_FAMILIES)}），曲线里的中文可能显示为方块"


def _spread_labels(values, gap):
    """把两条线的末端标注拉开到至少 gap，避免文字重叠。"""
    order = sorted(values, key=values.get)
    placed = dict(values)
    if len(order) == 2 and placed[order[1]] - placed[order[0]] < gap:
        middle = (placed[order[0]] + placed[order[1]]) / 2
        placed[order[0]], placed[order[1]] = middle - gap / 2, middle + gap / 2
    return placed


def plot_curve(report, direction, out):
    """每方向一张：横轴强推步数，纵轴抓取成功率，两模型两条线加 Wilson 竖线。

    k 是离散点位，区间用竖线而不是连续色带，并把两模型左右错开 0.8 步，否则 12 局方向的区间几乎完全重叠。
    """
    groups = [g for g in report["groups"] if g["direction"] == direction and g["total"] > 0]
    if not groups:
        return None
    plt, warning = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ends = {}
    for model, dodge in zip(MODELS, (-0.8, 0.8)):
        points = sorted((g for g in groups if g["model"] == model), key=lambda g: g["k"])
        if not points:
            continue
        x = np.array([g["k"] for g in points], dtype=float) + dodge
        y = np.array([g["grasp_rate"] * 100 for g in points])
        below = y - np.array([g["wilson_low"] * 100 for g in points])
        above = np.array([g["wilson_high"] * 100 for g in points]) - y
        color = SERIES_COLOR[model]
        ax.errorbar(x, y, yerr=[np.maximum(below, 0), np.maximum(above, 0)], color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=1.2, capsize=3, label=MODEL_LABEL[model], zorder=3)
        ends[model] = (x[-1], y[-1])
    placed = _spread_labels({model: end[1] for model, end in ends.items()}, 7.0)
    ks = sorted({g["k"] for g in groups})
    span = max(ks[-1] - ks[0], 16)
    label_x = max(end[0] for end in ends.values()) + span * 0.025  # 从靠右那条线的竖线之外起笔，不压另一模型的竖线
    for model in ends:
        ax.text(label_x, placed[model], MODEL_LABEL[model], color=INK, fontsize=9, va="center")
    ax.set_xlim(ks[0] - span * 0.05, ks[-1] + span * 0.24)
    ax.set_xticks(ks)
    ax.set_ylim(-3, 103)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("强推步数（0 为未强推的对照）", color=MUTED, fontsize=10)
    ax.set_ylabel("抓取成功率（%）", color=MUTED, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    totals = sorted({g["total"] for g in groups})
    count = str(totals[0]) if len(totals) == 1 else f"{totals[0]}～{totals[-1]}"
    fig.suptitle(DIRECTION_TITLE[direction], x=0.085, y=0.98, ha="left", fontsize=12, color=INK)
    fig.text(0.085, 0.885, f"每点 {count} 局，竖线为 Wilson 95% 区间", ha="left", fontsize=9, color=MUTED)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.17), ncol=2, frameon=False, fontsize=9, labelcolor=MUTED, handlelength=1.8)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.84, bottom=0.13)
    base = Path(out) / f"recovery_curve_{direction}"
    with warnings.catch_warnings():
        if warning:  # 已写进 summary；不再让 matplotlib 按缺字逐个刷屏。
            warnings.simplefilter("ignore", UserWarning)
        fig.savefig(f"{base}.png", dpi=180, facecolor=SURFACE)
        fig.savefig(f"{base}.pdf", facecolor=SURFACE)
    plt.close(fig)
    return warning


def write_blank_curve(direction, out):
    """某方向没有任何有效局（如冒烟只跑一个方向）时仍写出文件，图上直接说明，产物清单不缺项。"""
    plt, warning = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    ax.axis("off")
    ax.text(0.5, 0.5, f"{DIRECTION_TITLE[direction]}\n本次产物根没有该方向的有效局", ha="center", va="center", color=MUTED, fontsize=11)
    base = Path(out) / f"recovery_curve_{direction}"
    with warnings.catch_warnings():
        if warning:
            warnings.simplefilter("ignore", UserWarning)
        fig.savefig(f"{base}.png", dpi=180, facecolor=SURFACE)
        fig.savefig(f"{base}.pdf", facecolor=SURFACE)
    plt.close(fig)
    return warning


def write_outputs(report, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for direction in DIRECTIONS:
        has_data = any(g["direction"] == direction and g["total"] > 0 for g in report["groups"])
        warning = plot_curve(report, direction, out) if has_data else write_blank_curve(direction, out)
        if warning and warning not in report["warnings"]:
            report["warnings"].append(warning)
            print(f"警告：{warning}", file=sys.stderr)
    with open(out / "per_episode.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(report["per_episode"])
    text = summary_text(report)
    (out / "summary.txt").write_text(text)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description="MoveCube 强推恢复实验的分析与验收")
    parser.add_argument("--root", required=True, help="产物根，下有 <model>/k<k>/<shard>/")
    parser.add_argument("--force-table", required=True, help="force-table.json")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--expected-episodes", type=int, default=33, help="期望局数（默认 33）")
    parser.add_argument("--expected-k", default=",".join(map(str, ALL_K)), help="期望 k，逗号分隔（默认全部 6 个）")
    args = parser.parse_args(argv)
    expected_k = tuple(int(text) for text in args.expected_k.split(",") if text.strip())
    report = analyze(args.root, args.force_table, args.expected_episodes, expected_k)
    print(write_outputs(report, args.out), end="")
    return 0 if all(item["pass"] for item in report["verdicts"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
