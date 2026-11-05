"""StopCube 真实 rollout 的 Q1 触发分布与对齐点（1105 计划 K1、K7、K8）。

只读两个模型的 step_trace.jsonl、持久终态、收据和窗口表，产物写 <root>/analysis/：
align.json（逐局对齐点，供 Q2～Q4 消费）、trigger-summary.json、trigger-distribution.png。
窗口表只取窗口、M、N、悬停高度和比较方块用的 window_probe 轨迹，不用 Oracle 的动作或预测。

触发时刻只认 button_pressed_after 首次为真的 env_step_after；stop_timestep 单独报告。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from scripts.interpret.analysis.oracle_press_prob import bootstrap, episode_number, fingerprint, write_json


MODELS = ("ctx", "mot")
CATEGORIES = ("窗口内", "偏晚", "偏早", "未按")
OUTCOMES = ("success", "fail", "timeout")
CUBE_TOL = 1e-6
CROSS_MARGIN = 0.02
REACH_TOL = 0.01
MAX_BAD_EPISODES = 5
MODEL_STYLE = {"ctx": ("基线 ctx", "#2a78d6", "o"), "mot": ("motion", "#eb6834", "s")}


def load_table(path):
    """只取 K1 允许的字段，其余（Oracle 动作、终局）不进入内存。"""
    episodes = json.loads(Path(path).read_text(encoding="utf-8"))["episodes"]
    result = {}
    for key, item in episodes.items():
        interval, target = item["move_interval"], item["target_pass"]
        result[key] = {"episode_key": item["episode_key"], "move_interval": interval, "target_pass": target,
                       "window_in": item["target_visit_in"], "window_out": item["target_visit_out"],
                       "window_center": interval * (target - 0.5), "z_hover": item["z_hover"],
                       "trace_path": item["trace_path"]}
    return result


def read_steps(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return [row for row in rows if row.get("event") == "step"]


def read_json_or_none(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def judge_attempt(trace_path, rows):
    """判断一次尝试是否有官方终局；返回 (错误列表, outcome, task_success)。语义同 oracle_press_prob 的校准校验。"""
    key, nonce = rows[0]["episode_key"], trace_path.parent.name
    terminal = read_json_or_none(trace_path.parents[2] / f"{key}.terminal.json")
    if terminal is None:
        return ["没有持久终态（无终局）"], None, None
    errors = []
    outcome = terminal.get("outcome")
    if terminal.get("record_type") != "terminal" or terminal.get("episode_key") != key \
            or terminal.get("episode_nonce") != nonce:
        errors.append("持久终态的局身份或 nonce 与轨迹不符")
    if outcome not in OUTCOMES:
        errors.append(f"持久终态 outcome 非法：{outcome!r}")
    if terminal.get("actions_executed") != len(rows):
        errors.append(f"持久终态动作数 {terminal.get('actions_executed')} 与轨迹 {len(rows)} 行不符")
    task_success = int(outcome == "success")
    receipt = read_json_or_none(trace_path.parent / "meta" / "hold_receipt.json")
    if receipt is not None:
        if receipt.get("outcome") != outcome or receipt.get("task_success") != task_success:
            errors.append("收据的 outcome 或 task_success 与持久终态不符")
    return errors, outcome, task_success


def load_model(run_dir, table, episodes=None):
    """逐局选定一次尝试：优先 accepted 里最新的有效尝试，其次最新的有效尝试；选不出就是无效局。"""
    run_dir = Path(run_dir)
    keys = sorted(episodes if episodes is not None else table, key=episode_number)
    candidates = {key: [] for key in keys}
    for path in sorted(run_dir.glob("attempts/*/client/traces/*/step_trace.jsonl")):
        try:
            rows = read_steps(path)
        except (OSError, ValueError) as error:
            # 中断时可能留下半行，这里取不到局身份，从 started 记录反查。
            started = [read_json_or_none(item) or {} for item in path.parents[2].glob("*.started.json")]
            match = [item["episode_key"] for item in started if item.get("episode_nonce") == path.parent.name]
            if match and match[0] in candidates:
                candidates[match[0]].append({"path": path, "rows": [], "errors": [f"轨迹读取失败：{error}"]})
            continue
        if not rows:
            continue
        key = rows[0]["episode_key"]
        if key not in candidates:
            continue
        errors, outcome, success = judge_attempt(path, rows)
        accepted = (run_dir / "accepted" / f"{key}.{path.parent.name[:8]}.terminal.json").exists()
        candidates[key].append({"path": path, "rows": rows, "errors": errors, "outcome": outcome,
                                "task_success": success, "accepted": accepted, "mtime": path.stat().st_mtime_ns})
    started_counts = {}
    for path in run_dir.glob("attempts/*/client/*.started.json"):
        key = path.name[:-len(".started.json")]
        started_counts[key] = started_counts.get(key, 0) + 1
    result = {}
    for key in keys:
        options = candidates[key]
        tries = max(len(options), started_counts.get(key, 0))
        item = {"episode_key": key, "tries": tries, "retries": max(0, tries - 1), "valid": False,
                "invalid_reason": None, "nonce": None, "trace_path": None, "attempt": None,
                "outcome": None, "task_success": None, "rows": []}
        good = [option for option in options if not option["errors"]]
        if good:
            pool = [option for option in good if option["accepted"]] or good
            chosen = max(pool, key=lambda option: option["mtime"])
            item.update(valid=True, nonce=chosen["path"].parent.name, trace_path=str(chosen["path"].resolve()),
                        attempt=str(chosen["path"].parents[3].resolve()), outcome=chosen["outcome"],
                        task_success=chosen["task_success"], rows=chosen["rows"])
        elif options:
            item["invalid_reason"] = "；".join(options[-1]["errors"])
        else:
            item["invalid_reason"] = "没有轨迹"
        result[key] = item
    return result


def load_rollouts(root, table, models=MODELS, episodes=None):
    return {model: load_model(Path(root) / model / "run", table, episodes) for model in models}


def classify(trigger, entry):
    if trigger is None:
        return "未按"
    if trigger < entry["window_in"]:
        return "偏早"
    return "偏晚" if trigger > entry["window_out"] else "窗口内"


def window_table_match(rows, probe_steps):
    """触发前每步的方块位置与窗口表同步号比较；触发那一步起方块被按钮影响，不比。"""
    maximum, compared, uncovered = 0.0, 0, 0
    for row in rows:
        if row.get("button_pressed_after") is True:
            break
        reference = probe_steps.get(row["env_step_after"])
        if reference is None:
            uncovered += 1
            continue
        compared += 1
        delta = np.abs(np.asarray(row["cube_position_after"], dtype=float) - np.asarray(reference, dtype=float))
        maximum = max(maximum, float(delta.max()))
    return {"cube_maxabs": maximum, "cube_compared_steps": compared, "cube_uncovered_steps": uncovered,
            "window_table_match": maximum <= CUBE_TOL}


def group_inferences(rows):
    """按 infer_seq 汇总动作块；state_before 取 row_in_chunk==0 那一行。"""
    groups = {}
    for row in rows:
        group = groups.setdefault(row["infer_seq"], {
            "infer_seq": row["infer_seq"], "chunk_start_env_step": row["chunk_start_env_step"],
            "chunk_end_env_step": row["env_step_after"], "rows": 0, "state_before": None})
        group["rows"] += 1
        group["chunk_end_env_step"] = row["env_step_after"]
        if row["row_in_chunk"] == 0:
            group["state_before"] = row["state_before"]
    return [groups[seq] for seq in sorted(groups)]


def near_inference(rows, center):
    """未按的局：覆盖窗口中心的动作块；每行覆盖 [env_step_before, env_step_after)。"""
    for row in rows:
        if row["env_step_before"] <= center < row["env_step_after"]:
            return row["infer_seq"], None
    return None, f"轨迹在第 {rows[-1]['env_step_after']} 步结束，动作块没有覆盖窗口中心 {center}"


def align_points(rows, entry, match=True):
    """单局对齐点。不符局（match=False）不给依赖窗口的分类、相对中心和 I_near。"""
    z_hover = entry["z_hover"]
    triggered = next((row for row in rows if row.get("button_pressed_after") is True), None)
    crossed = next((row for row in rows if row["tcp_after"][2] < z_hover - CROSS_MARGIN), None)
    reached = next((row for row in rows if abs(row["tcp_after"][2] - z_hover) <= REACH_TOL), None)
    inferences = group_inferences(rows)
    present = {item["infer_seq"] for item in inferences}
    trigger = triggered["env_step_after"] if triggered else None
    cross = crossed["infer_seq"] if crossed else None
    result = {"trigger_step": trigger, "stop_timestep": next(
                  (row["stop_timestep"] for row in rows if row.get("stop_timestep") is not None), None),
              "steps_after_trigger": rows[-1]["env_step_after"] - trigger if triggered else None,
              "last_env_step": rows[-1]["env_step_after"],
              "I_cross": cross, "I_trigger": triggered["infer_seq"] if triggered else None,
              "reach_step": reached["env_step_after"] if reached else None,
              "category": None, "trigger_minus_center": None, "I_near": None, "I_near_reason": None,
              "inferences": inferences}
    for offset in (1, 2, 3):
        earlier = None if cross is None else cross - offset
        result[f"I_cross_minus_{offset}"] = earlier if earlier in present else None
    if not match:
        result["I_near_reason"] = "不符局，不给依赖窗口的对齐点"
        return result
    result["category"] = classify(trigger, entry)
    if triggered:
        result["trigger_minus_center"] = trigger - entry["window_center"]
        result["I_near_reason"] = "只给未按的局"
    else:
        result["I_near"], result["I_near_reason"] = near_inference(rows, entry["window_center"])
    return result


def invalid_uses():
    return {"q1": False, "q2_window": False, "q2_other": False, "q3": False, "q4": False}


def analyze_episode(item, entry, probe_steps):
    public = {name: value for name, value in item.items() if name != "rows"}
    public.update(move_interval=entry["move_interval"], target_pass=entry["target_pass"],
                  window_in=entry["window_in"], window_out=entry["window_out"], window_center=entry["window_center"],
                  z_hover=entry["z_hover"], uses=invalid_uses())
    if not item["valid"]:
        return public
    check = window_table_match(item["rows"], probe_steps)
    match = check["window_table_match"]
    public.update(check)
    public.update(align_points(item["rows"], entry, match))
    # K8：不符局不进 Q1、Q4 和 Q2 依赖窗口的部分，仍进 Q2 其余统计和 Q3。
    public["uses"] = {"q1": match, "q2_window": match, "q2_other": True, "q3": True, "q4": match}
    return public


def demote_invalid(episode, reason):
    """记录不完整按 K7/K8 记为无效局：保留身份和原因，分析字段全部移除。"""
    keep = ("episode_key", "tries", "retries", "move_interval", "target_pass", "window_in", "window_out",
            "window_center", "z_hover", "nonce", "trace_path", "attempt")
    result = {name: episode.get(name) for name in keep}
    result.update(valid=False, invalid_reason=reason, outcome=None, task_success=None, uses=invalid_uses())
    return result


def check_records(attn_dir, episodes):
    """结构核对：每个有效局的 rollout_<seq>.npz 数等于动作块数，step_idx 与 infer_seq 一一对应。"""
    attn_dir = Path(attn_dir)
    problems, inferences = {}, 0
    valid = [episode for episode in episodes if episode["valid"]]
    for episode in valid:
        found = []
        directory = attn_dir / str(episode["nonce"])
        expected = {item["infer_seq"]: item["chunk_start_env_step"] for item in episode["inferences"]}
        inferences += len(expected)
        names = {path.name for path in directory.glob("rollout_*.npz")}
        wanted = {f"rollout_{seq:04d}.npz" for seq in expected}
        for name in sorted(wanted - names):
            found.append(f"缺文件 {directory / name}")
        for name in sorted(names - wanted):
            found.append(f"多余文件 {directory / name}")
        offsets, seen = set(), set()
        for seq, start in expected.items():
            path = directory / f"rollout_{seq:04d}.npz"
            if path.name not in names:
                continue
            try:
                with np.load(path, allow_pickle=False) as record:
                    infer_seq, step_idx = int(record["infer_seq"]), int(record["step_idx"])
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                found.append(f"读取失败 {path}：{error}")
                continue
            if infer_seq != seq:
                found.append(f"{path.name} 的 infer_seq={infer_seq} 与文件序号不符")
            if step_idx in seen:
                found.append(f"{path.name} 的 step_idx={step_idx} 重复")
            seen.add(step_idx)
            offsets.add(step_idx - start)
        if len(offsets) > 1:
            found.append(f"step_idx 与轨迹动作块起点的差不唯一：{sorted(offsets)}")
        if found:
            problems[episode["episode_key"]] = found
    return {"episodes": len(valid), "inferences": inferences, "problems": problems}


def load_probe(entry):
    return {row["env_step_after"]: row["cube_position_after"] for row in read_steps(entry["trace_path"])}


def compute(root, table_path, models=MODELS, episodes=None, require_records=True):
    """读全部模型并逐局分析，不写盘。require_records=False 只用于没有 rollout 记录的中止数据。"""
    table = load_table(table_path)
    loaded = load_rollouts(root, table, models, episodes)
    probes = {}
    results, record_checks = {}, {}
    for model in models:
        analyzed = {}
        for key, item in loaded[model].items():
            if item["valid"] and key not in probes:
                probes[key] = load_probe(table[key])
            analyzed[key] = analyze_episode(item, table[key], probes.get(key, {}))
        if require_records:
            check = check_records(Path(root) / model / "attn", analyzed.values())
            for key, problems in check["problems"].items():
                analyzed[key] = demote_invalid(analyzed[key], "输入记录不完整：" + "；".join(problems))
            record_checks[model] = check
        results[model] = analyzed
    return results, record_checks


def stats(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "min": float(array.min()), "max": float(array.max())}


def distribution(results, repeats=2000, seed=42):
    """四类计数、按 M 分层、逐局配对和局级重采样；分母按 K8 分别列出。"""
    per_model = {}
    for model, episodes in results.items():
        items = list(episodes.values())
        valid = [item for item in items if item["valid"]]
        eligible = [item for item in valid if item["uses"]["q1"]]
        counts = {name: sum(item["category"] == name for item in eligible) for name in CATEGORIES}
        by_interval = {}
        for item in eligible:
            group = by_interval.setdefault(str(item["move_interval"]), dict.fromkeys(CATEGORIES, 0))
            group[item["category"]] += 1
        disagreements = [{"episode_key": item["episode_key"], "category": item["category"],
                          "trigger_step": item["trigger_step"], "stop_timestep": item["stop_timestep"],
                          "task_success": item["task_success"]}
                         for item in eligible if (item["category"] == "窗口内") != bool(item["task_success"])]
        per_model[model] = {
            "n_valid": len(valid), "n_q1": len(eligible), "n_invalid": len(items) - len(valid),
            "invalid": [{"episode_key": item["episode_key"], "reason": item["invalid_reason"]}
                        for item in items if not item["valid"]],
            "mismatched": [{"episode_key": item["episode_key"], "cube_maxabs": item["cube_maxabs"]}
                           for item in valid if not item["uses"]["q1"]],
            "infra_retries": sum(item["retries"] for item in items),
            "cube_maxabs": max((item["cube_maxabs"] for item in valid), default=0.0),
            "cube_uncovered_steps": sum(item["cube_uncovered_steps"] for item in valid),
            "counts": counts, "by_move_interval": by_interval,
            "trigger_minus_center": stats([item["trigger_minus_center"] for item in eligible
                                           if item["trigger_minus_center"] is not None]),
            "stop_timestep": {"n": sum(item["stop_timestep"] is not None for item in valid),
                              "episodes": {item["episode_key"]: item["stop_timestep"] for item in valid}},
            "task_success": {"n": len(valid), "success": sum(bool(item["task_success"]) for item in valid)},
            "window_vs_official_disagree": disagreements}
    summary = {"category_order": list(CATEGORIES), "models": per_model, "paired": paired(results, repeats, seed)}
    return summary


def paired(results, repeats=2000, seed=42):
    """逐局配对只用两个模型都能进 Q1 的局；单模型时为空。"""
    if len(results) != 2:
        return {"direction": None, "episodes": [], "window_in_rate_diff": None, "trigger_minus_center_diff": None}
    left, right = results
    shared = [key for key in results[left]
              if results[left][key]["uses"]["q1"] and results[right].get(key, {}).get("uses", {}).get("q1")]
    table = []
    for key in shared:
        row = {"episode_key": key, "move_interval": results[left][key]["move_interval"],
               "target_pass": results[left][key]["target_pass"]}
        for model in (left, right):
            item = results[model][key]
            row[model] = {name: item[name] for name in ("category", "trigger_step", "trigger_minus_center", "task_success")}
        table.append(row)
    in_diff = [float(row[right]["category"] == "窗口内") - float(row[left]["category"] == "窗口内") for row in table]
    center_diff = [row[right]["trigger_minus_center"] - row[left]["trigger_minus_center"] for row in table
                   if row[right]["trigger_minus_center"] is not None and row[left]["trigger_minus_center"] is not None]
    return {"direction": f"{right}_minus_{left}", "episodes": table,
            "window_in_rate_diff": bootstrap(in_diff, repeats, seed),
            "trigger_minus_center_diff": bootstrap(center_diff, repeats, seed),
            "bootstrap": {"repeats": repeats, "seed": seed, "unit": "episode"}}


def verdicts(summary):
    """三条判定行；单模型无效局或不符局超过 5 局时对应行写 FAIL。"""
    models = summary["models"]
    run_ok = bool(models) and all(item["n_invalid"] <= MAX_BAD_EPISODES and item["n_valid"] > 0 for item in models.values())
    match_ok = bool(models) and all(len(item["mismatched"]) <= MAX_BAD_EPISODES and item["n_valid"] > 0
                                    for item in models.values())
    dist_ok = bool(models) and all(item["n_q1"] > 0 and sum(item["counts"].values()) == item["n_q1"]
                                   for item in models.values())
    lines = [
        f"ROLLOUT_RUN={'PASS' if run_ok else 'FAIL'} models={len(models)} "
        + " ".join(f"valid_{name}={item['n_valid']}" for name, item in models.items())
        + f" invalid={sum(item['n_invalid'] for item in models.values())}"
        + f" infra_retries={sum(item['infra_retries'] for item in models.values())}",
        f"WINDOW_TABLE_MATCH={'PASS' if match_ok else 'FAIL'} models={len(models)} "
        f"checked={sum(item['n_valid'] for item in models.values())} "
        f"mismatched={sum(len(item['mismatched']) for item in models.values())} "
        f"cube_maxabs={max((item['cube_maxabs'] for item in models.values()), default=0.0):.9g} tol=1e-6",
        f"TRIGGER_DISTRIBUTION={'PASS' if dist_ok else 'FAIL'} models={len(models)} "
        + " ".join(f"n_{name}={item['n_q1']}" for name, item in models.items())
        + f" paired={len(summary['paired']['episodes'])} "
        + " ".join(f"{name}={format_counts(item['counts'])}" for name, item in models.items())]
    return lines, run_ok and match_ok and dist_ok


def format_counts(counts):
    return "/".join(str(counts[name]) for name in CATEGORIES)


def plot_distribution(results, output):
    """按 M 分三行，左列相对窗口中心的逐局散点，右列未按局数；图例放在图外不遮挡数据。"""
    eligible = {model: [item for item in episodes.values() if item["valid"] and item["uses"]["q1"]]
                for model, episodes in results.items()}
    intervals = sorted({item["move_interval"] for items in eligible.values() for item in items})
    if not intervals:
        return None
    output = Path(output)
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / "matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # TTC 在部分机器上以 JP 名称登记，字形相同；只选实际可用的字体。
    available = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [next(name for name in (
        "Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans") if name in available)]
    plt.rcParams["axes.unicode_minus"] = False
    ink, muted, grid, band = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d2"
    models = list(results)
    style = {model: MODEL_STYLE.get(model, (model, ("#2a78d6", "#eb6834")[index % 2], "os"[index % 2]))
             for index, model in enumerate(models)}
    figure, axes = plt.subplots(len(intervals), 2, figsize=(11, 2.5 * len(intervals) + 1.1), sharex="col",
                                squeeze=False, gridspec_kw={"width_ratios": [6, 1]})
    generator = np.random.default_rng(42)
    positions = {model: len(models) - 1 - index for index, model in enumerate(models)}
    for row, interval in enumerate(intervals):
        axis, side = axes[row]
        windows = [item for items in eligible.values() for item in items if item["move_interval"] == interval]
        axis.axvspan(min(item["window_in"] - item["window_center"] for item in windows),
                     max(item["window_out"] - item["window_center"] for item in windows),
                     color=band, alpha=0.7, linewidth=0, zorder=0)
        axis.axvline(0, color=muted, linewidth=0.8, zorder=1)
        side_counts = []
        for model in models:
            label, color, marker = style[model]
            items = [item for item in eligible[model] if item["move_interval"] == interval]
            values = [item["trigger_minus_center"] for item in items if item["trigger_minus_center"] is not None]
            jitter = generator.uniform(-0.22, 0.22, len(values))
            axis.scatter(values, positions[model] + jitter, s=36, color=color, marker=marker,
                         edgecolors="white", linewidths=0.8, zorder=3)
            side_counts.append((label, color, sum(item["category"] == "未按" for item in items)))
        axis.set_yticks([positions[model] for model in models], [style[model][0] for model in models])
        axis.set_ylim(-0.6, len(models) - 0.4)
        axis.set_title(f"M={interval}", loc="left", fontsize=11, color=ink)
        axis.grid(axis="x", color=grid, linewidth=0.7)
        axis.set_axisbelow(True)
        for label, color, count in side_counts:
            position = len(models) - 1 - [item[0] for item in side_counts].index(label)
            side.barh(position, max(count, 0), height=0.5, color=color)
            side.text(count + 0.15, position, str(count), va="center", ha="left", fontsize=10, color=ink)
        side.set_yticks([])
        side.set_ylim(-0.6, len(models) - 0.4)
        side.set_title("未按局数", loc="left", fontsize=10, color=muted)
        for name in ("top", "right"):
            axis.spines[name].set_visible(False)
            side.spines[name].set_visible(False)
    most = max(sum(item["category"] == "未按" for item in items) for items in eligible.values())
    for side in axes[:, 1]:
        side.set_xlim(0, max(most, 1) * 1.5 + 0.5)
        side.xaxis.get_major_locator().set_params(integer=True)
    axes[-1, 0].set_xlabel("触发步 − 窗口中心（步）", color=muted)
    axes[-1, 1].set_xlabel("局", color=muted)
    handles = [Line2D([0], [0], linestyle="", marker=style[model][2], color=style[model][1], markersize=7,
                      label=style[model][0]) for model in models]
    handles.append(Patch(color=band, alpha=0.7, label="本 M 的目标窗口范围"))
    figure.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (2.5 * len(intervals) + 1.1)))
    path = output / "trigger-distribution.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def write_outputs(results, record_checks, summary, lines, table_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = fingerprint(table_path)
    write_json(output / "align.json", {"schema_version": 1, "table": table, "models": results})
    write_json(output / "trigger-summary.json", {"schema_version": 1, "table": table, **summary,
               "record_checks": record_checks, "verdicts": lines})
    plot_distribution(results, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="产物根，其下有 <模型>/run 和 <模型>/attn")
    parser.add_argument("--table", type=Path, required=True, help="oracle_table.json")
    parser.add_argument("--episodes", default=None, help="task:id,...：应有的局，缺省为窗口表全部局（冒烟时只给该局）")
    args = parser.parse_args(argv)
    episodes = [item.strip() for item in args.episodes.split(",") if item.strip()] if args.episodes else None
    results, record_checks = compute(args.root, args.table, episodes=episodes)
    summary = distribution(results)
    lines, passed = verdicts(summary)
    write_outputs(results, record_checks, summary, lines, args.table, args.root / "analysis")
    print(f"RECORD_CHECK models={len(record_checks)} "
          f"valid_episodes={sum(item['episodes'] for item in record_checks.values())} "
          f"inferences={sum(item['inferences'] for item in record_checks.values())} "
          f"problem_episodes={sum(len(item['problems']) for item in record_checks.values())}", flush=True)
    for line in lines:
        print(line, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
