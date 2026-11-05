"""汇总 StopCube 第一遍轨迹，生成待人工确认的静态页面与悬停候选表。"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


EPISODES = tuple(f"StopCube:{i}" for i in range(50))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value, length):
    return isinstance(value, list) and len(value) == length and all(number(x) for x in value)


def episode_map(path):
    data = read_json(path)
    if data.get("schema_version") != 1 or not isinstance(data.get("episodes"), dict):
        raise ValueError(f"{path} 须为 schema_version=1，episodes 为按局键控的对象")
    return data["episodes"]


def resolve_path(value, parent):
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else parent / path


def discover(root, kind):
    """只消费逐步轨迹及相邻回执；同局重复记录保留为歧义，不挑成功局。"""
    if root is None:
        return {}
    root = Path(root)
    found, seen = {}, set()
    traces = set(root.rglob("step_trace.jsonl"))
    receipts = {}
    for name in ("meta.json", "hold_receipt.json", "episode.json"):
        for path in root.rglob(name):
            data = read_json(path)
            target = resolve_path(data.get("trace_path"), path.parent)
            if target:
                traces.add(target)
                existing, _ = receipts.get(target.resolve(), ({}, path.parent))
                receipts[target.resolve()] = ({**existing, **data}, path.parent)
                if name == "episode.json":
                    receipts[target.resolve()][0]["attn_episode_dir"] = str(path.parent.resolve())
    for trace in sorted(traces):
        identity = trace.resolve()
        if identity in seen:
            continue
        seen.add(identity)
        meta, parent = receipts.get(identity, ({}, trace.parent))
        for path in (trace.parent / "meta.json", trace.parent / "hold_receipt.json",
                     trace.parent / "meta" / "hold_receipt.json"):
            if path.is_file():
                meta = {**meta, **read_json(path)}
                parent = path.parent
        errors, rows = [], []
        if trace.is_file():
            for line_number, line in enumerate(trace.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"逐步记录第 {line_number} 行不是 JSON")
                    continue
                if item.get("event") == "step":
                    rows.append(item)
        else:
            errors.append("回执指向的逐步记录不存在")
        key = meta.get("episode_key") or (rows[0].get("episode_key") if rows else None)
        trajectory_kind = meta.get("trajectory_kind") or (rows[0].get("trajectory_kind") if rows else kind)
        if not key:
            continue
        if trajectory_kind != kind:
            errors.append(f"轨迹类型不符：期望 {kind}，记录 {trajectory_kind}")
        if any(row.get("episode_key") != key or row.get("trajectory_kind") != kind for row in rows):
            errors.append("逐步记录混入其他局或其他轨迹类型")
        video = resolve_path(meta.get("video_path"), parent)
        attn = Path(meta["attn_episode_dir"]) if meta.get("attn_episode_dir") else None
        infer_files = sorted(attn.glob("infer_*.npz")) if attn else []
        record = {"meta": meta, "rows": rows, "trace_path": str(trace.resolve()),
                  "infer_files": [str(path.resolve()) for path in infer_files],
                  "video_path": str(video.resolve()) if video else None, "errors": errors}
        found.setdefault(key, []).append(record)
    return found


def trajectory_calibration(calibration, kind):
    if kind in calibration.get("trajectories", {}):
        return calibration["trajectories"][kind]
    if kind == calibration.get("trajectory_kind", "normal"):
        return calibration
    return {}


def command_height(row):
    value = row.get("executed_target_tcp_z")
    if number(value):
        return value
    value = row.get("executed_target_tcp")
    return value[2] if vector(value, 3) else None


def analyze(record, window, calibration, kind):
    """命令按发出前的环境步分类，实测运动按完成后的环境步另记。"""
    if record is None:
        return {"trajectory_kind": kind, "present": False, "reasons": ["尚无该轨迹记录"],
                "observability_reasons": [], "events": {}, "window_infers": [], "window_rep": None}
    rows, meta = record["rows"], record["meta"]
    reasons = list(record["errors"])
    cal = trajectory_calibration(calibration, kind)
    events = dict(cal.get("events", {}))
    hover_z, delta_z = calibration.get("hover_z"), calibration.get("delta_z")
    threshold = hover_z - delta_z if number(hover_z) and number(delta_z) else None
    command = next((r for r in rows if threshold is not None and command_height(r) is not None
                    and command_height(r) < threshold), None)
    if "command_press_step" in events:
        pass  # S1 的离线 FK 事件保留为来源，不要求它改写原始逐步记录。
    elif command is not None:
        events.update(command_press_step=command["env_step_before"], press_infer=command.get("infer_seq"))
    elif not number(events.get("command_press_step")):
        if rows and all(command_height(r) is not None for r in rows) and threshold is not None:
            events.update(command_press_step=None, press_infer=None)
        else:
            reasons.append("缺少实际执行动作的目标 TCP 高度或独立命令校准，无法确认命令下压时刻")
    if threshold is None:
        reasons.append("缺少独立悬停高度或下压阈值校准")
    actual = next((r for r in rows if threshold is not None and vector(r.get("tcp_before"), 3)
                   and vector(r.get("tcp_after"), 3) and r["tcp_before"][2] >= threshold
                   and r["tcp_after"][2] < threshold), None)
    button = next((r for r in rows if r.get("button_pressed_before") is False
                   and r.get("button_pressed_after") is True), None)
    for name, bounds_name, row in (("actual_press_onset", "actual_press_onset_bounds", actual),
                                   ("button_trigger_step", "button_trigger_bounds", button)):
        if bounds_name not in events:
            events[bounds_name] = [row["env_step_before"] + 1, row["env_step_after"]] if row else None
            events[name] = row["env_step_after"] if row else None
    events["stop_timestep"] = next((r["stop_timestep"] for r in rows
                                        if r.get("stop_timestep") is not None), events.get("stop_timestep"))
    events["terminal_step"] = rows[-1].get("env_step_after") if rows else None
    infer_intervals = {}
    for row in rows:
        seq, before, after = row.get("infer_seq"), row.get("env_step_before"), row.get("env_step_after")
        if seq is None or not number(before) or not number(after):
            continue
        interval = infer_intervals.setdefault(seq, [before, after])
        interval[0], interval[1] = min(interval[0], before), max(interval[1], after)
    start_range = window.get("start_range", calibration.get("start_range"))
    valid_range = vector(start_range, 2) and start_range[0] <= start_range[1]
    overlaps, representative, classification, observability = [], None, "未校准", []
    if valid_range:
        center = sum(start_range) / 2
        overlaps = [seq for seq, (lo, hi) in sorted(infer_intervals.items())
                    if lo <= start_range[1] and hi > start_range[0]]
        if overlaps:
            representative = min(overlaps, key=lambda seq: (abs(infer_intervals[seq][0] - center),
                                                            infer_intervals[seq][0]))
        else:
            observability.append("合适下压范围没有已执行推理覆盖，可能终局提前截断")
        press = events.get("command_press_step")
        classification = ("提前" if press < start_range[0] else "过晚" if press > start_range[1]
                          else "范围内") if number(press) else "未观察到下压命令"
    else:
        reasons.append("合适下压范围未校准")
    if not rows:
        reasons.append("逐步记录为空")
    if meta.get("outcome") is None:
        reasons.append("缺少终局回执，不能推断官方成败")
    if meta.get("task_success") is None:
        reasons.append("终局回执缺少独立 task_success 字段")
    infer_files = record.get("infer_files", [])
    file_sequences = []
    for path in infer_files:
        try:
            file_sequences.append(int(Path(path).stem.removeprefix("infer_")))
        except ValueError:
            reasons.append("推理文件名缺少合法序号")
    if not infer_files:
        reasons.append("缺少第一遍 infer_*.npz，请传入包含 episode.json 的 attention 根目录")
    elif set(file_sequences) != set(infer_intervals) or len(file_sequences) != len(infer_intervals):
        reasons.append("第一遍推理文件与实际执行推理序号不一致")
    if not record.get("video_path") or not Path(record["video_path"]).is_file():
        reasons.append("缺少可播放原始视频")
    # 录像每次外层动作写一帧；复合物理步不能直接当成视频帧数。
    fps, video_start = meta.get("video_fps"), meta.get("video_start_env_step")
    video_offset = meta.get("video_exec0_frame_index")
    video_timing = (number(fps) and fps > 0 and number(video_start) and number(video_offset)
                    and bool(rows) and all(number(row.get("action_index")) for row in rows))
    if not video_timing:
        reasons.append("缺少视频帧率、执行首帧索引、动作索引或首帧环境步，视频只能人工定位")
    event_video_frames = {}
    if video_timing:
        for name in ("command_press_step", "actual_press_onset", "button_trigger_step",
                     "stop_timestep", "terminal_step", "actual_takeover_step", "actual_release_step"):
            step = events.get(name, meta.get(name))
            if not number(step):
                continue
            before_event = name in ("command_press_step", "actual_takeover_step", "actual_release_step")
            row = next((row for row in rows if (row["env_step_before"] == step if before_event
                        else row["env_step_before"] < step <= row["env_step_after"])), None)
            if row is not None:
                event_video_frames[name] = video_offset + row["action_index"] + (0 if before_event else 1)
    action_samples = []
    selected_steps = {events.get("command_press_step"), meta.get("actual_takeover_step"),
                      meta.get("actual_release_step")}
    for row in rows:
        if row.get("env_step_before") in selected_steps:
            action_samples.append({key: row.get(key) for key in
                                   ("env_step_before", "infer_seq", "command_action", "executed_action",
                                    "state_before", "state_after", "tcp_before", "tcp_after")})
    return {"trajectory_kind": kind, "present": True, "episode_nonce": meta.get("episode_nonce"),
            "outcome": meta.get("outcome"), "task_success": meta.get("task_success"),
            "events": events, "classification": classification,
            "window_infers": overlaps, "window_rep": representative,
            "infer_intervals": {str(k): v for k, v in infer_intervals.items()},
            "observed_infers": sorted(infer_intervals), "trace_path": record["trace_path"],
            "infer_files": infer_files,
            "video_path": record["video_path"], "video_fps": fps, "video_start_env_step": video_start,
            "video_exec0_frame_index": video_offset, "action_samples": action_samples,
            "event_video_frames": event_video_frames,
            "video_timing_verified": video_timing,
            "takeover_step": meta.get("takeover_step"), "release_step": meta.get("release_step"),
            "actual_takeover_step": meta.get("actual_takeover_step"),
            "actual_release_step": meta.get("actual_release_step"),
            "hold_qpos": meta.get("hold_qpos"),
            "D": meta.get("D"), "exec_start_idx": meta.get("exec_start_idx"),
            "observability_reasons": observability,
            "reasons": list(dict.fromkeys(reasons))}


def hold_candidate(record, analysis, window, calibration):
    """仅为正常轨迹提前按错局选取已观测的悬停位置，不生成替代动作。"""
    reasons = []
    if not record or not analysis.get("present"):
        return None, ["正常轨迹缺失"]
    if record["errors"]:
        return None, list(record["errors"])
    if analysis.get("outcome") != "fail":
        return None, ["正常终局不是 fail，不属于提前按错候选"]
    if analysis.get("classification") != "提前":
        return None, ["正常轨迹没有已确认的提前下压命令"]
    if not number(analysis["events"].get("button_trigger_step")):
        return None, ["正常轨迹没有实际按钮触发，不能归为提前按错"]
    if calibration.get("status") != "resolved" or window.get("status") != "resolved":
        reasons.append("窗口或耗时校准尚未解决")
    rows, meta = record["rows"], record["meta"]
    if any(meta.get(field) is None or calibration.get(field) is None or meta[field] != calibration[field]
           for field in ("episode_key", "episode_id", "env_seed")):
        reasons.append("正常轨迹与校准的局身份或 env_seed 缺失或不一致，不能生成接管候选")
    press = analysis["events"]["command_press_step"]
    confirmed = set(trajectory_calibration(calibration, "normal").get("events", {}).get("hover_steps", []))
    threshold = (calibration["hover_z"] - calibration["delta_z"]
                 if number(calibration.get("hover_z")) and number(calibration.get("delta_z")) else None)
    safe = [r for r in rows if r.get("env_step_after") in confirmed and r["env_step_after"] < press
            and vector(r.get("state_after"), 8) and vector(r.get("tcp_after"), 3)
            and threshold is not None and r["tcp_after"][2] >= threshold
            and not r.get("button_pressed_after")]
    if not safe:
        reasons.append("下压前没有校准确认且按钮未触发的安全悬停实测位置")
    initial = meta.get("initial_env_step")
    if not number(initial):
        starts = [r.get("chunk_start_env_step") for r in rows
                  if r.get("infer_seq") == 0 and number(r.get("chunk_start_env_step"))]
        initial = min(starts) if starts else None
    if not number(initial):
        reasons.append("缺少真实初始化环境步，不能假设 16 步边界从零开始")
    if reasons:
        return None, reasons
    chosen = max(safe, key=lambda r: r["env_step_after"])
    takeover = chosen["env_step_after"]
    ideal = sum(window["start_range"]) / 2
    minimum_k = max(0, math.floor((takeover - initial) / 16) + 1)
    center_k = (ideal - initial) / 16
    candidates = {max(minimum_k, math.floor(center_k)), max(minimum_k, math.ceil(center_k))}
    release = min((initial + 16 * k for k in candidates), key=lambda x: (abs(x - ideal), x))
    return {"takeover_step": takeover, "release_step": release, "hold_qpos": chosen["state_after"],
            "ideal_release_step": ideal, "release_offset": release - ideal,
            "initial_env_step": initial, "D": meta.get("D"), "exec_start_idx": meta.get("exec_start_idx"),
            "normal_trace_path": record["trace_path"], "calibration_source": calibration.get("calibration_source")}, []


def build_review(windows, calibrations, normal, extended, expected_keys=EPISODES):
    episodes, candidates, excluded, ready = {}, {}, {}, True
    unexpected = sorted((set(windows) | set(calibrations) | set(normal) | set(extended)) - set(expected_keys))
    for key in expected_keys:
        window, calibration = windows.get(key, {}), calibrations.get(key, {})
        reasons = []
        if not window:
            reasons.append("缺少本局窗口")
        if not calibration:
            reasons.append("缺少本局校准")
        if window.get("status") != "resolved":
            reasons.extend(window.get("reasons") or ["窗口尚未解决"])
        if calibration.get("status") != "resolved":
            reasons.extend(calibration.get("reasons") or ["耗时校准尚未解决"])
        records = {}
        for kind, collection in (("normal", normal), ("hold_extended", extended)):
            matches = collection.get(key, [])
            records[kind] = matches[0] if len(matches) == 1 else None
            if len(matches) > 1:
                reasons.append(f"{kind} 有 {len(matches)} 份重复轨迹，未自动挑选")
        analyses = {kind: analyze(record, window, calibration, kind) for kind, record in records.items()}
        expected_id = int(key.rpartition(":")[2])
        for kind, record in records.items():
            if record is None:
                continue
            meta = record["meta"]
            if meta.get("episode_key") != key or meta.get("episode_id") != expected_id:
                reasons.append(f"{kind} 回执缺少或不匹配本局 episode_key/episode_id")
            if calibration.get("episode_key") != key or not number(calibration.get("env_seed")):
                reasons.append(f"{kind} 缺少可核对的校准局身份或 env_seed")
            elif meta.get("env_seed") != calibration["env_seed"]:
                reasons.append(f"{kind} 回执 env_seed 缺失或与本局校准不一致")
            if calibration.get("episode_id") != expected_id:
                reasons.append("校准 episode_id 缺失或与局键不一致")
        candidate, rejected = hold_candidate(records["normal"], analyses["normal"], window, calibration)
        if candidate:
            candidates[key] = candidate
        else:
            excluded[key] = {"reasons": rejected}
        required_extended = analyses["normal"].get("outcome") == "fail" and analyses["normal"].get("classification") == "提前" and number(analyses["normal"]["events"].get("button_trigger_step"))
        if required_extended and not candidate:
            reasons.extend(rejected)
        if required_extended and not analyses["hold_extended"].get("present"):
            reasons.append("提前按错局尚缺独立悬停延长轨迹")
        if candidate:
            actual = analyses["hold_extended"].get("actual_release_step")
            analyses["hold_extended"]["ideal_release_step"] = candidate["ideal_release_step"]
            analyses["hold_extended"]["planned_release_step"] = candidate["release_step"]
            analyses["hold_extended"]["release_offset"] = actual - candidate["ideal_release_step"] if number(actual) else None
        if required_extended and candidate and records["hold_extended"] is not None:
            record = records["hold_extended"]
            meta, rows = record["meta"], record["rows"]
            takeover, release, qpos = candidate["takeover_step"], candidate["release_step"], candidate["hold_qpos"]
            for field, expected in (("takeover_step", takeover), ("release_step", release),
                                    ("actual_takeover_step", takeover), ("actual_release_step", release),
                                    ("hold_qpos", qpos)):
                if meta.get(field) != expected:
                    reasons.append(f"悬停延长回执 {field} 缺失或与本次候选不一致")
            prefix_fields = ("env_step_before", "env_step_after", "command_action", "executed_action")
            normal_prefix = [tuple(row.get(field) for field in prefix_fields)
                             for row in records["normal"]["rows"] if row.get("env_step_before", takeover) < takeover]
            extended_prefix = [tuple(row.get(field) for field in prefix_fields)
                               for row in rows if row.get("env_step_before", takeover) < takeover]
            if not normal_prefix or normal_prefix != extended_prefix or any(
                    not number(row[0]) or not number(row[1]) or not vector(row[2], 8)
                    or not vector(row[3], 8) or row[1] > takeover
                    for row in normal_prefix):
                reasons.append("正常与延长接管前的环境步、模型命令或实际执行动作不一致或缺失")
            held = [row for row in rows if takeover <= row.get("env_step_before", -1) < release]
            if (not held or held[0].get("env_step_before") != takeover or held[-1].get("env_step_after") != release
                    or any(row.get("executed_action") != qpos for row in held)
                    or any(left.get("env_step_after") != right.get("env_step_before")
                           for left, right in zip(held, held[1:]))):
                reasons.append("实际接管段缺失、不连续或执行动作偏离固定悬停位置")
            resumed = [row for row in rows if row.get("env_step_before") == release]
            if (len(resumed) != 1 or resumed[0].get("row_in_chunk") != 0
                    or resumed[0].get("chunk_start_env_step") != release
                    or not vector(resumed[0].get("command_action"), 8)
                    or resumed[0].get("command_action") != resumed[0].get("executed_action")):
                reasons.append("实际恢复步缺失或未执行该边界新推理的模型命令")
            if any(row.get("command_action") != row.get("executed_action")
                   for row in rows if row.get("env_step_before", -1) >= release):
                reasons.append("恢复后的实际动作与模型命令不一致")
        needed = [analyses["normal"]] + ([analyses["hold_extended"]] if required_extended else [])
        observed = analyses["hold_extended"] if required_extended else analyses["normal"]
        if not observed["window_infers"]:
            reasons.append("所需轨迹未观察到合适下压范围内的真实推理")
        complete = (not reasons and calibration.get("status") == "resolved" and window.get("status") == "resolved"
                    and all(item.get("present") and not item["reasons"] for item in needed))
        ready = ready and complete
        episodes[key] = {"episode_key": key, "window": window, "calibration": calibration,
                         "trajectories": analyses, "hold_required": required_extended,
                         "hold_candidate": candidate, "hold_inapplicable_reasons": rejected,
                         "ready": complete, "reasons": reasons}
    summary = {"expected": len(expected_keys), "normal": sum(e["trajectories"]["normal"]["present"] for e in episodes.values()),
               "extended": sum(e["trajectories"]["hold_extended"]["present"] for e in episodes.values()),
               "hold_required": sum(e["hold_required"] for e in episodes.values()), "hold_candidates": len(candidates),
               "normal_success": sum(e["trajectories"]["normal"].get("outcome") == "success" for e in episodes.values()),
               "extended_success": sum(e["trajectories"]["hold_extended"].get("outcome") == "success" for e in episodes.values()),
               "unexpected_episodes": unexpected, "ready": ready and not unexpected}
    draft = {"schema_version": 1, "approval": "pending", "summary": summary, "episodes": episodes}
    holds = {"schema_version": 1, "episodes": candidates, "excluded_episodes": excluded}
    return draft, holds


def escaped(value):
    if value is None:
        return "未记录"
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape(str(value), quote=True)


def file_url(path, out):
    if not path:
        return None
    return quote(os.path.relpath(Path(path), Path(out)), safe="/.")


def media_section(item, out):
    parts = []
    video = item.get("video_path")
    if video and Path(video).is_file():
        url = file_url(video, out)
        parts.append(f'<video controls preload="none" src="{escaped(url)}"></video><p><a href="{escaped(url)}">原始完整视频</a></p>')
        if item.get("video_timing_verified"):
            media_events = {name: step for name, step in item.get("events", {}).items()
                            if name in ("command_press_step", "actual_press_onset", "button_trigger_step",
                                        "stop_timestep", "terminal_step")}
            media_events.update({name: item.get(name) for name in ("actual_takeover_step", "actual_release_step")})
            for name, step in media_events.items():
                if not number(step):
                    continue
                frame = item.get("event_video_frames", {}).get(name)
                if not number(frame):
                    continue
                seconds = frame / item["video_fps"]
                if seconds < 0:
                    continue
                lo, hi = max(0, seconds - 2), seconds + 2
                bounds = item["events"].get(EVENT_BOUNDS.get(name))
                interval = f'（环境步区间 {escaped(bounds)}）' if vector(bounds, 2) and bounds[0] != bounds[1] else ""
                parts.append(f'<a href="{escaped(url)}#t={lo:.3f},{hi:.3f}">{escaped(EVENT_NAMES.get(name, name))}{interval}附近 {lo:.2f}～{hi:.2f} 秒</a> ')
    trace = file_url(item.get("trace_path"), out)
    if trace:
        parts.append(f'<p><a href="{escaped(trace)}">逐步原始记录</a></p>')
    return "".join(parts)


EVENT_NAMES = {"command_press_step": "下压命令实际发出步", "press_infer": "首条下压命令推理序号",
               "actual_press_onset": "实际 TCP 首跨阈值步", "button_trigger_step": "按钮首次触发步",
               "stop_timestep": "官方停步", "terminal_step": "逐步记录结束步", "hover_steps": "已确认悬停步",
               "actual_takeover_step": "实际接管步", "actual_release_step": "实际恢复步"}
EVENT_BOUNDS = {"actual_press_onset": "actual_press_onset_bounds", "button_trigger_step": "button_trigger_bounds"}


def render_html(draft, out):
    summary = draft["summary"]
    content = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
               '<title>StopCube 第一阶段人工确认</title><style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:24px auto;padding:0 16px;color:#182230;background:#f5f7fa}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #cad2dc;padding:8px;text-align:left;vertical-align:top}code{overflow-wrap:anywhere}section{margin:22px 0;padding:18px;background:white;border:1px solid #cad2dc}video{max-width:100%;width:520px}.missing{color:#a12323}.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}a{color:#124eb0}li{margin:6px 0}</style>',
               '<h1>StopCube 第一阶段人工确认</h1><p>状态：待用户审核。本页不会批准决策点，也不会启动第二阶段。正常与悬停延长分别计分，延长成功不计入官方成功率。</p>',
               f'<p>正常轨迹 {summary["normal"]}/{summary["expected"]}；正常成功 {summary["normal_success"]}；延长轨迹 {summary["extended"]}；延长成功 {summary["extended_success"]}；提前按错需延长 {summary["hold_required"]}。</p>',
               '<p>所有时刻均为环境步；下压命令按 env_step_before 计，实际运动与按钮保留校准区间。视频按 action_index 定位外层动作帧，不声称看到复合调用省略的中间物理步；不改写原视频。没有视频步制时保留缺失说明。</p>',
               '<table><thead><tr><th>局</th><th>合适下压范围</th><th>正常结果／命令分类</th><th>延长结果</th><th>审核状态与缺失</th></tr></thead><tbody>']
    for index, (key, episode) in enumerate(draft["episodes"].items()):
        normal, ext = episode["trajectories"]["normal"], episode["trajectories"]["hold_extended"]
        reasons = episode["reasons"] + normal["reasons"] + normal["observability_reasons"] + (ext["reasons"] if episode["hold_required"] else [])
        content.append(f'<tr><td><a href="#episode-{index}">{escaped(key)}</a></td><td>{escaped(episode["window"].get("start_range"))}</td><td>{escaped(normal.get("outcome"))}／{escaped(normal.get("classification"))}</td><td>{escaped(ext.get("outcome"))}</td><td>{"待人工确认" if episode["ready"] else "证据未齐"}<br>{escaped(reasons)}</td></tr>')
    content.append('</tbody></table>')
    if summary["unexpected_episodes"]:
        content.append(f'<p class="missing">清单外局：{escaped(summary["unexpected_episodes"])}</p>')
    for index, (key, episode) in enumerate(draft["episodes"].items()):
        window, cal = episode["window"], episode["calibration"]
        content.extend([f'<section id="episode-{index}"><h2>{escaped(key)}</h2>',
                        '<table><tr><th>移动间隔／目标经过次数</th><th>目标进入／离开</th><th>下压范围／运动耗时</th><th>悬停参考／阈值差</th></tr>',
                        f'<tr><td>{escaped(window.get("move_interval"))}／{escaped(window.get("target_pass"))}</td><td>{escaped(window.get("target_visit_in"))}／{escaped(window.get("target_visit_out"))}</td><td>{escaped(window.get("start_range"))}／{escaped(cal.get("motion_delay"))}</td><td>{escaped(cal.get("hover_z"))}／{escaped(cal.get("delta_z"))} 米</td></tr></table>',
                        f'<p>校准状态：{escaped(cal.get("status"))}；来源：<code>{escaped(cal.get("calibration_source"))}</code>；缺失原因：{escaped(cal.get("reasons", []))}</p>',
                        f'<p>接管候选：<code>{escaped(episode["hold_candidate"])}</code>；不适用原因：{escaped(episode["hold_inapplicable_reasons"])}</p><div class="pair">'])
        for kind, label in (("normal", "正常轨迹"), ("hold_extended", "悬停延长轨迹（独立成绩）")):
            item = episode["trajectories"][kind]
            content.append(f'<div><h3>{label}</h3><p>终局：{escaped(item.get("outcome"))}；独立成功字段：{escaped(item.get("task_success"))}；轨迹标识：{escaped(item.get("episode_nonce"))}</p>')
            content.append('<ul>')
            for name in EVENT_NAMES:
                bounds = item["events"].get(EVENT_BOUNDS.get(name))
                value = bounds if vector(bounds, 2) and bounds[0] != bounds[1] else item["events"].get(name, item.get(name))
                suffix = "（区间，非精确步）" if vector(bounds, 2) and bounds[0] != bounds[1] else ""
                content.append(f'<li>{escaped(EVENT_NAMES[name])}：{escaped(value)}{suffix}</li>')
            content.append(f'<li>窗口重叠推理：{escaped(item["window_infers"])}；代表推理：{escaped(item["window_rep"])}</li>')
            content.append(f'<li>接管计划／实际：{escaped(item.get("takeover_step"))}／{escaped(item.get("actual_takeover_step"))}</li><li>恢复理想／计划／实际：{escaped(item.get("ideal_release_step"))}／{escaped(item.get("planned_release_step", item.get("release_step")))}／{escaped(item.get("actual_release_step"))}；实际偏差：{escaped(item.get("release_offset"))}</li></ul>')
            content.append(f'<p>关键步模型命令与实际执行动作：<code>{escaped(item.get("action_samples"))}</code></p>')
            content.append(f'<p>第一遍推理文件：{len(item.get("infer_files", []))}；已执行推理序号：{escaped(item.get("observed_infers"))}</p>')
            content.append(f'<p>窗口可观测性：{escaped(item["observability_reasons"])}</p>')
            content.append(f'<p class="missing">{escaped(item["reasons"])}</p>{media_section(item, out)}</div>')
        content.append('</div></section>')
    content.append('</html>')
    return "\n".join(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True, help="正常 attention 根目录，递归读取 episode.json 及其引用的逐步记录与回执")
    parser.add_argument("--extended", type=Path, help="延长 attention 根目录，与正常轨迹独立")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    draft, holds = build_review(episode_map(args.windows), episode_map(args.calibration),
                                discover(args.normal, "normal"), discover(args.extended, "hold_extended"))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("decision_points_draft.json", draft), ("hold_candidates.json", holds)):
        (args.out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "review.html").write_text(render_html(draft, args.out), encoding="utf-8")
    summary = draft["summary"]
    print(f'STAGE1_REVIEW_READY={"PASS" if summary["ready"] else "FAIL"} normal={summary["normal"]} extended={summary["extended"]} approval=pending')
    # 草稿缺证据仍正常写出，验收状态由具名判定行和 JSON 明确承载。
    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
