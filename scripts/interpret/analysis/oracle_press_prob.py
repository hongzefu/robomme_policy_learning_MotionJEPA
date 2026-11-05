"""Oracle 询问的目标高度、固定校准选点及局级配对报告。

convert-z 必须在 official-sim 环境执行；summarize 和 report 只需要 CPU。
目标 TCP 高度不代表实际触发，FK 校验只对独立 trace 的同步 raw_state/TCP。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re

import numpy as np


KINDS = ("fixed_release", "high_probability", "low_probability", "early_press")
THRESHOLDS = (0.01, 0.02, 0.03)


def fingerprint(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def episode_number(key):
    return int(key.rsplit(":", 1)[-1])


def load_table(path):
    table = json.loads(Path(path).read_text())
    episodes = table["episodes"]
    for key, item in episodes.items():
        for field in ("move_interval", "target_pass", "reach_step", "z_hover", "target_visit_in", "target_visit_out"):
            if item.get(field) is None:
                raise ValueError(f"动作表 {key} 缺少 {field}")
    return episodes


def load_query(path):
    with np.load(path, allow_pickle=False) as source:
        result = {key: np.asarray(source[key]) for key in (
            "actions", "raw_state", "model_id", "episode_key", "env_step", "query_seq", "pre_reach")}
    for key in ("model_id", "episode_key", "env_step", "query_seq", "pre_reach"):
        result[key] = result[key].item()
    if result["actions"].shape != (10, 20, 8) or result["raw_state"].shape != (8,):
        raise ValueError(f"询问数组形状不符：{path}")
    if not all(np.isfinite(result[key]).all() for key in ("actions", "raw_state")):
        raise ValueError(f"询问存在非有限值：{path}")
    result["path"] = str(Path(path).resolve())
    return result


def trace_candidates(roots, nonce=None):
    result = {}
    all_paths = []
    for root in roots:
        paths = [Path(root)] if Path(root).is_file() else sorted(Path(root).rglob("step_trace.jsonl"))
        all_paths.extend(paths)
    exact = [path for path in all_paths if path.parent.name == nonce] if nonce else []
    for path in exact or all_paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows = [row for row in rows if row.get("event") == "step"]
        if rows:
            result.setdefault(rows[0]["episode_key"], []).append((path, rows))
    return result


def synchronized_tcp(query, candidates):
    """精确绑定同局同 nonce 的同步状态，不能用 Oracle 悬停高度自证。"""
    options = candidates.get(query["episode_key"], [])
    nonce = Path(query["path"]).parent.name
    exact = [item for item in options if item[0].parent.name == nonce]
    options = exact or options
    matches = []
    for path, rows in options:
        for row in rows:
            for suffix in ("before", "after"):
                if row.get(f"env_step_{suffix}") != query["env_step"]:
                    continue
                state = np.asarray(row.get(f"state_{suffix}", []), dtype=float)
                tcp = np.asarray(row.get(f"tcp_{suffix}", []), dtype=float)
                if state.shape == (8,) and tcp.shape == (3,) and np.isfinite(tcp).all():
                    error = float(np.max(np.abs(state - query["raw_state"])))
                    if error <= 1e-6:
                        matches.append((float(tcp[2]), path))
    if not matches:
        raise ValueError(f"找不到同步 raw_state/TCP：{query['path']} step={query['env_step']}")
    if len({str(path.resolve()) for _, path in matches}) > 1:
        raise ValueError(f"同局有多份 trace，须指定唯一 trace 或匹配 nonce：{query['path']}")
    if max(value for value, _ in matches) - min(value for value, _ in matches) > 1e-6:
        raise ValueError("同一步 trace 前后状态冲突")
    return matches[0]


def convert_arrays(reader, query):
    state_z = float(reader.set_arm(query["raw_state"][:7])[2])
    action_z = np.asarray([reader.set_arm(action[:7])[2] for action in query["actions"].reshape(-1, 8)])
    action_z = action_z.reshape(10, 20)
    if not np.isfinite(action_z).all() or not math.isfinite(state_z):
        raise ValueError("运动学换算存在非有限值")
    return state_z, action_z


def _convert_episode(job):
    paths, roots, output, table = job
    from scripts.interpret.read_sim_eef_pose import TcpReader, build_env

    queries = [load_query(path) for path in paths]
    first = queries[0]
    key, model = first["episode_key"], first["model_id"]
    candidates = trace_candidates(roots, Path(first["path"]).parent.name)
    synchronized = [synchronized_tcp(query, candidates) for query in queries]
    env, seed, _ = build_env("StopCube", episode_number(key))
    if int(seed) != int(table[key]["env_seed"]):
        env.close()
        raise ValueError(f"环境 seed 与动作表不符：{key}")
    try:
        env.reset(seed=seed)
        reader = TcpReader(env)
        values = [convert_arrays(reader, query) for query in queries]
    finally:
        env.close()
    state_z = np.asarray([value[0] for value in values])
    action_z = np.stack([value[1] for value in values])
    actual_z = np.asarray([value[0] for value in synchronized])
    safe = re.sub(r"[^\w.-]", "_", model)
    # 同一模型/局重复采集不得静默覆盖；在调用层拒绝重复步。
    out = Path(output) / f"query_z_{safe}_{episode_number(key)}.npz"
    manifest = {"queries": [fingerprint(query["path"]) for query in queries],
                "traces": [fingerprint(path) for path in sorted({p for _, p in synchronized})]}
    np.savez_compressed(out, model_id=model, episode_key=key,
        env_step=np.asarray([query["env_step"] for query in queries]),
        query_seq=np.asarray([query["query_seq"] for query in queries]),
        pre_reach=np.asarray([query["pre_reach"] for query in queries], dtype=bool),
        query_paths=np.asarray([query["path"] for query in queries]),
        action_z=action_z, state_z=state_z, actual_tcp_z=actual_z,
        provenance=json.dumps(manifest, ensure_ascii=False))
    return {"path": str(out.resolve()), "rows": len(queries),
            "maxabs_z": float(np.max(np.abs(state_z - actual_z)))}


def convert_z(query_roots, trace_roots, table, output, workers=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grouped, seen = {}, set()
    for root in query_roots:
        for path in sorted(Path(root).rglob("query_*.npz")):
            if path.name.startswith("query_z_"):
                continue
            query = load_query(path)
            identity = (query["model_id"], query["episode_key"], query["env_step"])
            if identity in seen:
                raise ValueError(f"重复模型/局/询问步：{identity}")
            seen.add(identity)
            grouped.setdefault(identity[:2], []).append(path)
    if not grouped:
        raise ValueError("没有询问记录")
    jobs = [(paths, trace_roots, str(output), table) for paths in grouped.values()]
    if workers == 1:
        results = [_convert_episode(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_convert_episode, jobs))
    maximum = max(item["maxabs_z"] for item in results)
    verdict = f"FK_PARITY={'PASS' if maximum <= 1e-3 else 'FAIL'} rows={sum(r['rows'] for r in results)} maxabs_z={maximum:.9g} tol=1e-3"
    write_json(output / "conversion.json", {"files": results, "verdict": verdict})
    print(verdict, flush=True)
    return maximum <= 1e-3


def wilson(successes, total, z=1.959963984540054):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson 分母或成功数非法")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def probabilities(action_z, z_hover):
    heights = np.asarray(action_z, dtype=float)
    if heights.shape != (10, 20) or not np.isfinite(heights).all() or not math.isfinite(z_hover):
        raise ValueError("高度必须为有限 (10,20) 数组")
    result = {}
    for horizon in (16, 20):
        minimum = heights[:, :horizon].min(axis=1)
        for threshold in THRESHOLDS:
            count = int(np.count_nonzero(minimum < z_hover - threshold))
            result[f"h{horizon}_cm{round(threshold * 100)}"] = {
                "count": count, "total": 10, "p": count / 10, "wilson95": wilson(count, 10)}
    return result


def visit_number(step, entry):
    for visit in entry["visits"]:
        if visit["in"] <= step <= visit["out"]:
            return int(visit["target_pass"])
    return None


def summarize_caches(cache_root, table):
    rows, seen, fk = [], set(), []
    for path in sorted(Path(cache_root).rglob("query_z_*.npz")):
        with np.load(path, allow_pickle=False) as source:
            model, key = str(source["model_id"].item()), str(source["episode_key"].item())
            entry = table[key]
            steps = source["env_step"]
            if source["action_z"].shape != (len(steps), 10, 20):
                raise ValueError(f"高度缓存形状不符：{path}")
            if any(source[name].shape != (len(steps),) for name in
                   ("pre_reach", "state_z", "actual_tcp_z", "query_paths")):
                raise ValueError(f"高度缓存点数不符：{path}")
            if not all(np.isfinite(source[name]).all() for name in ("state_z", "actual_tcp_z")):
                raise ValueError(f"高度缓存核验值不有限：{path}")
            provenance = json.loads(str(source["provenance"].item()))
            for item in provenance["queries"] + provenance["traces"]:
                if fingerprint(item["path"]) != item:
                    raise ValueError(f"高度缓存的输入已改变：{item['path']}")
            fk.extend(np.abs(source["state_z"] - source["actual_tcp_z"]).tolist())
            for index, step in enumerate(steps.tolist()):
                identity = (model, key, step)
                if identity in seen:
                    raise ValueError(f"高度缓存重复询问步：{identity}")
                seen.add(identity)
                pre = bool(source["pre_reach"][index]) or step < entry["reach_step"]
                stats = probabilities(source["action_z"][index], float(entry["z_hover"]))
                t = entry["move_interval"] * (entry["target_pass"] - 0.5)
                visit = visit_number(step, entry)
                rows.append({"model_id": model, "episode_key": key, "env_step": step,
                    "relative_step": step - t, "move_interval": entry["move_interval"],
                    "target_pass": entry["target_pass"], "visit": visit, "pre_reach": pre,
                    "cadence16": (step - 1) % 16 == 0,
                    "decision": not pre and t - 48 <= step <= entry["target_visit_out"],
                    "early_visit": not pre and visit is not None and visit < entry["target_pass"],
                    "p": stats["h16_cm2"]["p"], "wilson95": stats["h16_cm2"]["wilson95"],
                    "variants": stats, "query_path": str(source["query_paths"][index]),
                    "cache_path": str(path.resolve())})
    if not rows:
        raise ValueError("没有高度缓存")
    return sorted(rows, key=lambda r: (r["model_id"], episode_number(r["episode_key"]), r["env_step"])), fk


def calibration_episodes(table):
    groups = {}
    for key, item in table.items():
        groups.setdefault(item["move_interval"], []).append(key)
    return [key for interval in sorted(groups) for key in sorted(groups[interval], key=episode_number)[:4]]


def select_points(rows, entry):
    eligible = sorted((row for row in rows if not row["pre_reach"] and row["env_step"] >= entry["reach_step"]),
                      key=lambda row: row["env_step"])
    t = entry["move_interval"] * (entry["target_pass"] - 0.5)
    decision = [row for row in eligible if t - 48 <= row["env_step"] <= entry["target_visit_out"]]
    early = [row for row in eligible if visit_number(row["env_step"], entry) is not None
             and visit_number(row["env_step"], entry) < entry["target_pass"]]
    release = [row for row in eligible if row["env_step"] <= t - 48]
    late = False
    fixed = release[-1] if release else None
    if fixed is None:
        fallback = [row for row in eligible if row["env_step"] < entry["target_visit_in"]]
        fixed = fallback[0] if fallback else None
        late = fixed is not None
    high = max(decision, key=lambda row: (row["p"], -row["env_step"])) if decision else None
    low = min(decision, key=lambda row: (row["p"], row["env_step"])) if decision else None
    early_point = max(early, key=lambda row: (row["p"], -row["env_step"])) if early else None
    choices = (fixed, high, low, early_point if early_point and early_point["p"] > 0 else None)
    reasons = ("没有到位后且早于目标区间的询问点", "决策段没有候选", "决策段没有候选",
               "前 N-1 次经过没有候选" if not early else "早按候选全部 P=0")
    result = {}
    for kind, choice, reason in zip(KINDS, choices, reasons):
        result[kind] = {"status": "selected" if choice else "skipped",
            "release_step": choice["env_step"] if choice else None,
            "stop_after_release": None if kind == "fixed_release" else 48,
            "p": choice["p"] if choice else None,
            "reason": "late_release" if kind == "fixed_release" and late else (None if choice else reason),
            "query_path": choice["query_path"] if choice else None}
    return result


def make_plan(rows, table):
    models = sorted({row["model_id"] for row in rows})
    chosen = calibration_episodes(table)
    result = {"schema_version": 1, "threshold_cm": 2, "horizon": 16, "selection_order": list(KINDS),
              "calibration_episodes": chosen, "models": {}}
    for model in models:
        episodes = {}
        for key in chosen:
            points = [row for row in rows if row["model_id"] == model and row["episode_key"] == key]
            episodes[key] = select_points(points, table[key])
            expected = set(range(1, table[key]["move_interval"] * table[key]["target_pass"] + 1, 4))
            actual = {row["env_step"] for row in points}
            if actual != expected:
                for choice in episodes[key].values():
                    choice.update(status="skipped", release_step=None, p=None, query_path=None,
                                  reason="缺少该模型本局询问结果" if not points else "本局询问网格不完整，不能固定极值点")
        result["models"][model] = {"episodes": episodes}
    return result


def bootstrap(values, repeats=2000, seed=42):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n_episodes": 0, "mean": None, "ci95": None}
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return {"n_episodes": len(values), "mean": float(values.mean()),
            "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}


def episode_metrics(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["model_id"], row["episode_key"]), []).append(row)
    result = {}
    for identity, points in grouped.items():
        decision = [row["p"] for row in points if row["decision"]]
        early = [row["p"] for row in points if row["early_visit"]]
        d = float(np.mean(decision)) if decision else None
        e = float(np.mean(early)) if early else None
        result[identity] = {"decision": d, "early": e, "separation": d - e if d is not None and e is not None else None}
    return result


def paired_comparison(rows, repeats=2000):
    metrics = episode_metrics(rows)
    models = sorted({model for model, _ in metrics})
    comparisons = []
    for i, left in enumerate(models):
        for right in models[i + 1:]:
            entry = {"left": left, "right": right, "direction": "left_minus_right", "metrics": {}}
            for name in ("decision", "early", "separation"):
                keys = sorted({key for model, key in metrics if model == left} & {key for model, key in metrics if model == right})
                pairs = [(key, metrics[left, key][name], metrics[right, key][name]) for key in keys]
                pairs = [(key, a, b) for key, a, b in pairs if a is not None and b is not None]
                entry["metrics"][name] = {**bootstrap([a - b for _, a, b in pairs], repeats),
                    "episodes": [key for key, _, _ in pairs]}
            comparisons.append(entry)
    return comparisons


def coverage(rows, table):
    result = {}
    for model in sorted({row["model_id"] for row in rows}):
        episodes = {}
        for key, entry in table.items():
            expected = set(range(1, entry["move_interval"] * entry["target_pass"] + 1, 4))
            actual = {row["env_step"] for row in rows if row["model_id"] == model and row["episode_key"] == key}
            episodes[key] = {"expected": len(expected), "actual": len(actual),
                "missing_steps": sorted(expected - actual), "extra_steps": sorted(actual - expected)}
        result[model] = episodes
    return result


def validate_calibration(item, rows, entry, selection, terminal):
    """核对已批校准要求的交还点、逐步动作证据和真实完成边界。"""
    errors = []

    def fail(reason):
        if reason not in errors:
            errors.append(reason)

    def action_equal(left, right):
        try:
            a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
            return a.shape == b.shape == (8,) and np.isfinite(a).all() and np.array_equal(a, b)
        except (TypeError, ValueError):
            return False

    release = selection.get("release_step")
    kind = item.get("calibration_kind")
    duration = None if kind == "fixed_release" else 48
    if selection.get("status") != "selected" or type(release) is not int:
        return ["该模型本局运行类型未在固定计划中选定"]
    if (item.get("actual_release_step") != release or item.get("release_step") != release
            or (item.get("calibration") or {}).get("release_step") != release
            or (item.get("calibration") or {}).get("stop_after_release") != duration
            or selection.get("stop_after_release") != duration):
        fail("实际交还点或运行时选点与固定计划不符")
    if (release < entry["reach_step"] or (release - entry["first_env_step"]) % 4
            or release > entry["move_interval"] * entry["target_pass"]):
        fail("交还点不是到位后的合法询问点")
    if item.get("trajectory_kind") != "oracle_calibrate":
        fail("收据不是校准运行")
    if not rows:
        return errors + ["缺少逐步轨迹"]
    if rows[0].get("env_step_before") != entry["first_env_step"]:
        fail("轨迹缺少从动作表首步开始的完整前缀")
    post = []
    previous = None
    for row in rows:
        before, after = row.get("env_step_before"), row.get("env_step_after")
        if (type(before) is not int or type(after) is not int or after <= before
                or (previous is not None and before != previous)):
            fail("轨迹步号缺失、重复或不连续")
            continue
        previous = after
        if row.get("episode_key") != item["episode_key"] or row.get("trajectory_kind") != "oracle_calibrate":
            fail("轨迹身份或运行类型不符")
        if any(type(row.get(f"button_pressed_{suffix}")) is not bool for suffix in ("before", "after")):
            fail("轨迹缺少明确的按钮状态，不能判断未触发")
        if before < release:
            index = before - entry["first_env_step"]
            actions = entry["actions"]
            if (not 0 <= index < len(actions) or row.get("control_source") != "oracle"
                    or not action_equal(row.get("executed_action"), actions[index])
                    or not action_equal(row.get("command_action"), actions[index])):
                fail("交还前动作或控制权与Oracle动作表不符")
            if row.get("button_pressed_before") is not False or row.get("button_pressed_after") is not False:
                fail("交还前已触发按钮或缺少未触发证据")
        else:
            post.append(row)
            if (row.get("control_source") != "model"
                    or not action_equal(row.get("executed_action"), row.get("command_action"))):
                fail("交还后执行动作或控制权不是模型输出")
    if not post or post[0].get("env_step_before") != release:
        fail("缺少固定交还步的新模型动作")
    try:
        prediction = np.asarray(item.get("first_release_prediction"), dtype=float)
        prediction_ok = prediction.shape == (20, 8) and np.isfinite(prediction).all()
    except (TypeError, ValueError):
        prediction_ok = False
    if not prediction_ok:
        fail("缺少完整有限的首次交还预测")
    for index, row in enumerate(post):
        if (row.get("row_in_chunk") != index % 16 or row.get("infer_seq") != index // 16
                or row.get("chunk_start_env_step") != post[(index // 16) * 16].get("env_step_before")):
            fail("交还后没有按新推理的前16行连续执行")
        if index < 16 and prediction_ok and not action_equal(row.get("command_action"), prediction[index]):
            fail("首次执行动作与首次交还预测不符")
    if item.get("execution_infers") != math.ceil(len(post) / 16):
        fail("实际执行推理次数与轨迹动作块不符")
    triggers = [row["env_step_after"] for row in rows
                if row.get("button_pressed_after") is True and type(row.get("env_step_after")) is int]
    trigger = min(triggers) if triggers else None
    if item.get("button_trigger_step") != trigger:
        fail("按钮触发步与独立轨迹不符")
    stop, outcome = item.get("calibration_stop"), item.get("outcome")
    if stop == "official_terminal":
        if outcome not in ("success", "fail", "timeout") or item.get("task_success") != int(outcome == "success"):
            fail("官方终局状态或独立任务成功字段不符")
    elif kind == "fixed_release":
        fail("固定放手没有运行到官方终局")
    elif stop == "release_48_steps":
        if outcome != "calibration_stop" or len(post) != 48 or trigger is not None:
            fail("48步截止没有完整48步或已有触发却未及时停止")
    elif stop == "button_trigger":
        if (outcome != "calibration_stop" or trigger is None or not post
                or trigger != post[-1].get("env_step_after") or not release < trigger <= release + 48):
            fail("按钮停止没有交还后48步内的即时触发证据")
    else:
        fail("缺少有效的校准停止或正常终局")
    if kind != "fixed_release" and (len(post) > 48 or (trigger is not None and post
            and trigger != post[-1].get("env_step_after"))):
        fail("非固定放手超过48步或按钮触发后仍继续执行")
    if (terminal.get("record_type") != "terminal" or terminal.get("episode_key") != item["episode_key"]
            or not item.get("episode_nonce") or terminal.get("episode_nonce") != item.get("episode_nonce")
            or terminal.get("outcome") != outcome or terminal.get("actions_executed") != len(rows)
            or item.get("actions_executed") != len(rows)):
        fail("缺少同局同nonce且动作计数一致的持久终态记录")
    return errors


def calibration_results(roots, table, plan):
    records = []
    for root in roots:
        for path in sorted(Path(root).rglob("hold_receipt.json")):
            item = json.loads(path.read_text())
            kind = item.get("calibration_kind") or (item.get("calibration") or {}).get("kind")
            model = item.get("model_id")
            key = item.get("episode_key")
            if kind not in KINDS or key not in table or model not in plan["models"]:
                continue
            release = item.get("actual_release_step")
            trace = Path(item.get("trace_path") or path.parent.parent / "step_trace.jsonl")
            read_errors = []
            try:
                rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
                rows = [row for row in rows if row.get("event") == "step"]
            except (OSError, ValueError) as error:
                rows = []
                read_errors.append(f"无法读取逐步轨迹：{error}")
            try:
                terminal = json.loads(Path(item["terminal_path"]).read_text())
            except (KeyError, OSError, ValueError) as error:
                terminal = {}
                read_errors.append(f"无法读取持久终态：{error}")
            selection = plan["models"][model]["episodes"].get(key, {}).get(kind, {})
            errors = read_errors + validate_calibration(item, rows, table[key], selection, terminal)
            triggers = [row["env_step_after"] for row in rows
                        if row.get("button_pressed_after") is True and type(row.get("env_step_after")) is int]
            trigger = min(triggers) if triggers else None
            entry = table[key]
            category = "没按" if trigger is None else ("早于区间" if trigger < entry["target_visit_in"] else
                "晚于区间" if trigger > entry["target_visit_out"] else "区间内")
            records.append({"model_id": model, "episode_key": key, "kind": kind,
                "release_step": release, "trigger_step": trigger, "trigger_category": category,
                "predicted_p": plan["models"][model]["episodes"].get(key, {}).get(kind, {}).get("p"),
                "trigger_within48": trigger is not None and type(release) is int and 0 <= trigger - release <= 48,
                "task_success": item.get("task_success"), "receipt_path": str(path.resolve()),
                "trace_path": str(trace.resolve()),
                "terminal_path": item.get("terminal_path"), "outcome": item.get("outcome"),
                "valid": not errors, "validation_errors": errors,
                "first_release_prediction": item.get("first_release_prediction"),
                "calibration_stop": item.get("calibration_stop")})
    counts = {}
    for row in records:
        identity = (row["model_id"], row["episode_key"], row["kind"])
        counts[identity] = counts.get(identity, 0) + 1
    for row in records:
        if counts[row["model_id"], row["episode_key"], row["kind"]] > 1:
            row["valid"] = False
            row["validation_errors"].append("同一选点有重复运行，全部保留且不挑成功样本")
    return records


def calibration_summary(records, plan):
    """结果缺失与运行跳过分开；分组按原计划，不用结果重选。"""
    actual = {(r["model_id"], r["episode_key"], r["kind"]): r for r in records if r["valid"]}
    attempted = {(r["model_id"], r["episode_key"], r["kind"]) for r in records}
    result = []
    for model, model_plan in plan["models"].items():
        for kind in KINDS:
            selected = [(key, item[kind]) for key, item in model_plan["episodes"].items()
                        if item[kind]["status"] == "selected"]
            completed = [actual[model, key, kind] for key, _ in selected if (model, key, kind) in actual]
            skipped = [{"episode_key": key, "reason": item[kind]["reason"]}
                       for key, item in model_plan["episodes"].items() if item[kind]["status"] != "selected"]
            result.append({"model_id": model, "kind": kind, "planned": len(selected), "completed": len(completed),
                "missing_episodes": [key for key, _ in selected if (model, key, kind) not in attempted],
                "failed": [{"episode_key": row["episode_key"], "receipt_path": row["receipt_path"],
                            "errors": row["validation_errors"]} for row in records
                           if row["model_id"] == model and row["kind"] == kind and not row["valid"]],
                "skipped": skipped, "trigger_within48": sum(r["trigger_within48"] for r in completed),
                "trigger_any": sum(r["trigger_step"] is not None for r in completed),
                "task_success": sum(r["task_success"] in (True, 1) for r in completed),
                "task_success_known": sum(r["task_success"] is not None for r in completed)})
    return result


def calibration_probability_groups(records):
    groups = {}
    for row in records:
        if row["valid"] and row["predicted_p"] is not None:
            groups.setdefault((row["model_id"], row["kind"], row["predicted_p"]), []).append(row)
    return [{"model_id": model, "kind": kind, "predicted_p": p, "runs": len(rows),
             "trigger_within48": sum(row["trigger_within48"] for row in rows)}
            for (model, kind, p), rows in sorted(groups.items())]


def grouped_visits(rows, repeats):
    groups = {}
    for row in rows:
        if row["pre_reach"] or row["visit"] is None:
            continue
        identity = (row["model_id"], row["move_interval"], row["target_pass"], row["visit"])
        groups.setdefault(identity, {}).setdefault(row["episode_key"], []).append(row["p"])
    return [{"model_id": model, "move_interval": interval, "target_pass": target, "visit": visit,
             **bootstrap([np.mean(points) for points in episodes.values()], repeats)}
            for (model, interval, target, visit), episodes in sorted(groups.items())]


def summarize(cache_root, table_path, output, repeats=2000, calibration_roots=()):
    table = load_table(table_path)
    rows, fk = summarize_caches(cache_root, table)
    plan = make_plan(rows, table)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "calibration_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("已固定的校准计划发生变化，请保留原计划并交主会话裁决")
    write_json(plan_path, plan)
    maximum = max(fk) if fk else None
    summary = {"schema_version": 1, "table": fingerprint(table_path), "rows": rows,
        "coverage": coverage(rows, table), "comparisons": paired_comparison(rows, repeats),
        "fk": {"rows": len(fk), "maxabs_z": maximum, "tolerance": 1e-3,
               "status": "pass" if maximum is not None and maximum <= 1e-3 else "fail"},
        "calibration_results": calibration_results(calibration_roots, table, plan),
        "visit_groups": grouped_visits(rows, repeats),
        "limitations": ["目标高度不是实测 TCP 或任务成功", "单点 10 次预测分辨率 0.1；Wilson 仅描述该点预测比例",
            "相邻询问点不独立；聚合区间按局重采样", "模型训练来源和步数不同，差异不能完全归因于 motion",
            "12 局校准只能看方向；不按结果调整 2 cm 阈值"]}
    summary["calibration_summary"] = calibration_summary(summary["calibration_results"], plan)
    summary["calibration_probability_groups"] = calibration_probability_groups(summary["calibration_results"])
    write_json(output / "summary.json", summary)
    with (output / "probabilities.csv").open("w", newline="") as stream:
        fields = ("model_id", "episode_key", "env_step", "relative_step", "move_interval", "visit", "pre_reach", "cadence16", "p")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary


def plot_report(summary, output, repeats=2000):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(output).resolve() / "matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Linux 的 TTC 集合通常以首个 JP 字体名注册，但同样包含中文字形。
    plt.rcParams["font.family"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = [row for row in summary["rows"] if not row["pre_reach"]]
    models = sorted({row["model_id"] for row in rows})
    intervals = sorted({row["move_interval"] for row in rows})
    if not intervals:
        raise ValueError("没有到位后的有效询问点")
    figure, axes = plt.subplots(len(intervals), 1, figsize=(11, 3.3 * len(intervals)), squeeze=False)
    curves = []
    for axis, interval in zip(axes[:, 0], intervals):
        for model in models:
            group = [row for row in rows if row["model_id"] == model and row["move_interval"] == interval]
            steps = sorted({row["relative_step"] for row in group})
            stats = [bootstrap([row["p"] for row in group if row["relative_step"] == step], repeats) for step in steps]
            if steps:
                line, = axis.plot(steps, [s["mean"] for s in stats], label=model)
                axis.fill_between(steps, [s["ci95"][0] for s in stats], [s["ci95"][1] for s in stats], color=line.get_color(), alpha=0.16)
                subset = [row for row in group if row["cadence16"]]
                subset_steps = sorted({row["relative_step"] for row in subset})
                axis.plot(subset_steps, [np.mean([row["p"] for row in subset if row["relative_step"] == step])
                    for step in subset_steps], linestyle="--", color=line.get_color(), alpha=0.7, label=f"{model} / 1+16k 子集")
                curves.extend({"model_id": model, "move_interval": interval, "relative_step": step, **stat}
                              for step, stat in zip(steps, stats))
        axis.axvline(0, color="black", linewidth=0.8)
        axis.axvline(-48, color="gray", linestyle=":", linewidth=0.8)
        axis.set(title=f"M={interval}", xlabel="询问步减去目标经过中心 T", ylabel="P(t)", ylim=(-0.03, 1.03))
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(Path(output) / "probability_curves.png", dpi=150)
    plt.close(figure)
    write_json(Path(output) / "curve_intervals.json", curves)

    # 先在每局的每次经过内求均值，再跨局计算，避免长局/多点局支配。
    figure, axis = plt.subplots(figsize=(10, 4))
    visit_stats = []
    for model in models:
        visits = sorted({row["visit"] for row in rows if row["model_id"] == model and row["visit"] is not None})
        values = []
        for visit in visits:
            group = [row for row in rows if row["model_id"] == model and row["visit"] == visit]
            keys = sorted({row["episode_key"] for row in group})
            stat = bootstrap([np.mean([row["p"] for row in group if row["episode_key"] == key]) for key in keys], repeats)
            values.append(stat)
            visit_stats.append({"model_id": model, "visit": visit, **stat})
        if visits:
            axis.errorbar(visits, [v["mean"] for v in values],
                yerr=[[v["mean"] - v["ci95"][0] for v in values], [v["ci95"][1] - v["mean"] for v in values]],
                marker="o", capsize=3, label=model)
    axis.set(xlabel="方块经过次数", ylabel="局内平均 P(t)", ylim=(-0.03, 1.03))
    axis.legend(fontsize=7)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(Path(output) / "visit_groups.png", dpi=150)
    plt.close(figure)
    write_json(Path(output) / "visit_intervals.json", visit_stats)


def report(summary_path, output, repeats=2000):
    summary_path, output = Path(summary_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text())
    plot_report(summary, output, repeats)
    escape = lambda value: html.escape(str(value))
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Oracle 下压比例</title>",
        "<style>body{max-width:1100px;margin:30px auto;font:16px/1.65 sans-serif}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}img{width:100%}code{overflow-wrap:anywhere}</style>",
        "<h1>Oracle 下压比例与真实校准</h1><p>主判据：前 16 行存在目标高度低于本局悬停高度 2 cm；到位前排除。另报 20 行及 1/3 cm，见原始汇总。</p>",
        "<p>图中阴影和误差线按局重采样；虚线为 1+16k 子集。点级 Wilson 区间保存在 summary.json。图尚需主会话目视复查。</p>",
        "<h2>覆盖范围</h2><table><tr><th>模型</th><th>有询问局数</th><th>动作表局数</th><th>缺失询问点</th></tr>"]
    for model, episodes in summary["coverage"].items():
        parts.append(f"<tr><td>{escape(model)}</td><td>{sum(v['actual'] > 0 for v in episodes.values())}</td><td>{len(episodes)}</td><td>{sum(len(v['missing_steps']) for v in episodes.values())}</td></tr>")
    parts.extend(["</table><h2>同步状态的高度核验</h2><pre>" + escape(json.dumps(summary["fk"], ensure_ascii=False)) + "</pre>",
        "<h2>两个模型的下压比例</h2><img src='probability_curves.png' alt='按 M 分面的 P(t) 曲线'>",
        "<h2>按方块经过次数分组</h2><img src='visit_groups.png' alt='按经过次数的局均值'>",
        "<h2>局级配对比较</h2><table><tr><th>左减右</th><th>量</th><th>配对局数</th><th>差值</th><th>95% 区间</th></tr>"])
    for comparison in summary["comparisons"]:
        for name, stat in comparison["metrics"].items():
            parts.append(f"<tr><td>{escape(comparison['left'])} − {escape(comparison['right'])}</td><td>{escape(name)}</td><td>{stat['n_episodes']}</td><td>{escape(stat['mean'])}</td><td>{escape(stat['ci95'])}</td></tr>")
    parts.append("</table><h2>校准与固定放手的真实结果</h2><p>是否触发和任务成功分开报告；缺失结果不算失败，不补挑成功局。证据无效的样本保留在表中，不进入完成数和比例分母。</p><table><tr><th>模型</th><th>局</th><th>类型</th><th>证据有效</th><th>无效原因</th><th>选点 P(t)</th><th>交还步</th><th>触发步</th><th>48 步内触发</th><th>区间分类</th><th>任务成功</th></tr>")
    for row in summary["calibration_results"]:
        parts.append("<tr>" + "".join(f"<td>{escape(row[key])}</td>" for key in ("model_id", "episode_key", "kind", "valid", "validation_errors", "predicted_p", "release_step", "trigger_step", "trigger_within48", "trigger_category", "task_success")) + "</tr>")
    parts.append("</table><h2>按选点 P(t) 对照实际触发</h2><table><tr><th>模型</th><th>类型</th><th>P(t)</th><th>运行数</th><th>48 步内触发数</th></tr>")
    for row in summary["calibration_probability_groups"]:
        parts.append("<tr>" + "".join(f"<td>{escape(row[key])}</td>" for key in ("model_id", "kind", "predicted_p", "runs", "trigger_within48")) + "</tr>")
    parts.append("</table><h2>校准完成量与跳过原因</h2><table><tr><th>模型</th><th>类型</th><th>计划/有效完成</th><th>48 步内触发</th><th>任务成功/已知</th><th>无效样本</th><th>缺失局与跳过原因</th></tr>")
    for row in summary["calibration_summary"]:
        parts.append(f"<tr><td>{escape(row['model_id'])}</td><td>{escape(row['kind'])}</td><td>{row['planned']}/{row['completed']}</td><td>{row['trigger_within48']}</td><td>{row['task_success']}/{row['task_success_known']}</td><td>{escape(row['failed'])}</td><td>{escape(row['missing_episodes'])} {escape(row['skipped'])}</td></tr>")
    parts.append("</table><h2>按 M、目标次数和实际经过分组</h2><table><tr><th>模型</th><th>M</th><th>目标 N</th><th>经过次数</th><th>局数</th><th>P 均值</th><th>95% 区间</th></tr>")
    for row in summary["visit_groups"]:
        parts.append("<tr>" + "".join(f"<td>{escape(row[key])}</td>" for key in ("model_id", "move_interval", "target_pass", "visit", "n_episodes", "mean", "ci95")) + "</tr>")
    parts.append("</table><h2>限制</h2><ul>" + "".join(f"<li>{escape(item)}</li>" for item in summary["limitations"]) + "</ul>")
    parts.append(f"<p>来源：<code>{escape(summary_path.resolve())}</code>。逐点 query_path/cache_path 与校准 trace_path 见该文件。</p></html>")
    (output / "report.html").write_text("\n".join(parts))
    print(f"REPORT_RENDERED=PASS models={len(summary['coverage'])} visual_review=pending path={output / 'report.html'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert-z", help="在 official-sim 内换算目标高度")
    convert.add_argument("--query-root", type=Path, action="append", required=True)
    convert.add_argument("--trace-root", type=Path, action="append", required=True)
    convert.add_argument("--oracle-table", type=Path, required=True)
    convert.add_argument("--output-dir", type=Path, required=True)
    convert.add_argument("--workers", type=int, default=1)
    for name in ("summarize", "plan"):
        analyze = commands.add_parser(name, help="固定主判据，生成汇总和校准计划")
        analyze.add_argument("--cache-root", type=Path, required=True)
        analyze.add_argument("--oracle-table", type=Path, required=True)
        analyze.add_argument("--output-dir", type=Path, required=True)
        analyze.add_argument("--calibration-root", type=Path, action="append", default=[])
        analyze.add_argument("--bootstrap", type=int, default=2000)
    render = commands.add_parser("report", help="从汇总生成静态图和 HTML")
    render.add_argument("--summary", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.command == "convert-z":
        if args.workers < 1:
            parser.error("workers 必须为正数")
        return 0 if convert_z(args.query_root, args.trace_root, load_table(args.oracle_table), args.output_dir, args.workers) else 1
    if args.bootstrap < 1:
        parser.error("bootstrap 必须为正数")
    if args.command in ("summarize", "plan"):
        result = summarize(args.cache_root, args.oracle_table, args.output_dir, args.bootstrap, args.calibration_root)
        print(f"PRESS_SUMMARY=PASS models={len(result['coverage'])} points={len(result['rows'])} threshold_cm=2 horizon=16")
    else:
        report(args.summary, args.output_dir, args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
