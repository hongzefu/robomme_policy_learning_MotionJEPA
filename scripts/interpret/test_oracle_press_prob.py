"""Oracle 概率、独立高度核验、固定选点与局级配对的 CPU 短测。"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.interpret.analysis import oracle_press_prob as analysis


def entry(interval=120, target=4):
    middle = int(interval * (target - 0.5))
    return {"env_seed": 521600, "move_interval": interval, "target_pass": target,
        "reach_step": 33, "z_hover": 0.16,
        "target_visit_in": middle - 8, "target_visit_out": middle + 8,
        "visits": [{"target_pass": visit, "in": int(interval * (visit - 0.5)) - 8,
                    "out": int(interval * (visit - 0.5)) + 8} for visit in range(1, target + 1)]}


def point(step, p=0.5, model="motion", key="StopCube:16", definition=None):
    definition = definition or entry()
    t = definition["move_interval"] * (definition["target_pass"] - 0.5)
    visit = analysis.visit_number(step, definition)
    pre = step < definition["reach_step"]
    return {"model_id": model, "episode_key": key, "env_step": step, "p": p,
        "pre_reach": pre, "query_path": f"query_{step}.npz",
        "relative_step": step - t, "move_interval": definition["move_interval"],
        "target_pass": definition["target_pass"], "visit": visit,
        "cadence16": (step - 1) % 16 == 0,
        "decision": not pre and t - 48 <= step <= definition["target_visit_out"],
        "early_visit": not pre and visit is not None and visit < definition["target_pass"]}


def test_probability_strict_threshold_horizon_and_sensitivity():
    z = np.full((10, 20), 0.16)
    z[0, 15] = 0.139
    z[1, 16] = 0.129
    z[2, 0] = 0.149
    z[3, 0] = 0.14
    result = analysis.probabilities(z, 0.16)
    assert result["h16_cm2"]["count"] == 1
    assert result["h20_cm2"]["count"] == 2
    assert result["h16_cm1"]["count"] == 3
    assert result["h20_cm3"]["count"] == 1
    assert result["h16_cm2"]["wilson95"][0] < 0.1 < result["h16_cm2"]["wilson95"][1]


def test_wilson_extremes_and_bad_input():
    assert analysis.wilson(0, 10)[0] == 0
    assert analysis.wilson(10, 10)[1] == pytest.approx(1)
    assert analysis.wilson(10, 10)[0] > 0.7
    with pytest.raises(ValueError):
        analysis.wilson(0, 0)
    with pytest.raises(ValueError):
        analysis.probabilities(np.zeros((9, 20)), 0.16)
    with pytest.raises(ValueError):
        analysis.probabilities(np.full((10, 20), np.nan), 0.16)


def test_fixed_points_tie_breaking_and_pre_reach_exclusion():
    rows = [point(step, p=0.5) for step in range(1, 481, 4)]
    rows[0]["p"] = 1
    selected = analysis.select_points(rows, entry())
    assert list(selected) == list(analysis.KINDS)
    assert selected["fixed_release"]["release_step"] == 369
    assert selected["high_probability"]["release_step"] == 373
    assert selected["low_probability"]["release_step"] == 373
    assert selected["early_press"]["release_step"] == 53
    assert selected["fixed_release"]["stop_after_release"] is None
    assert selected["high_probability"]["stop_after_release"] == 48


def test_late_release_and_no_early_press():
    definition = entry()
    definition["reach_step"] = 390
    rows = [point(step, 0, definition=definition) for step in range(1, 481, 4)]
    selected = analysis.select_points(rows, definition)
    assert selected["fixed_release"]["release_step"] == 393
    assert selected["fixed_release"]["reason"] == "late_release"
    assert selected["early_press"]["status"] == "skipped"
    definition["reach_step"] = 429
    selected = analysis.select_points([], definition)
    assert all(row["status"] == "skipped" for row in selected.values())


def test_zero_early_candidates_and_models_choose_their_own_points():
    first = [point(step, 0, model="baseline") for step in range(1, 481, 4)]
    second = [point(step, 1 if step == 401 else 0, model="motion") for step in range(1, 481, 4)]
    plan = analysis.make_plan(first + second, {"StopCube:16": entry()})
    assert plan["models"]["baseline"]["episodes"]["StopCube:16"]["high_probability"]["release_step"] == 373
    assert plan["models"]["motion"]["episodes"]["StopCube:16"]["high_probability"]["release_step"] == 401
    assert plan["models"]["motion"]["episodes"]["StopCube:16"]["early_press"]["reason"] == "早按候选全部 P=0"


def test_calibration_strata_numeric_order_and_missing_grid():
    table = {f"StopCube:{index}": entry(interval, 2) for interval, ids in
             ((60, [11, 2, 9, 3, 20]), (80, [10, 4, 8, 7, 30]), (120, [16, 5, 1, 6, 40])) for index in ids}
    assert analysis.calibration_episodes(table) == ["StopCube:2", "StopCube:3", "StopCube:9", "StopCube:11",
        "StopCube:4", "StopCube:7", "StopCube:8", "StopCube:10", "StopCube:1", "StopCube:5", "StopCube:6", "StopCube:16"]
    plan = analysis.make_plan([point(369)], table)
    assert len(plan["calibration_episodes"]) == 12
    assert all(choice["status"] == "skipped" for ep in plan["models"]["motion"]["episodes"].values() for choice in ep.values())


def test_bootstrap_pairs_by_episode_and_does_not_weight_query_counts():
    rows = []
    for model, key, decision, early, count in (("a", "StopCube:1", 0.9, 0.2, 9),
            ("a", "StopCube:2", 0.5, 0.1, 1), ("b", "StopCube:1", 0.4, 0.3, 2),
            ("b", "StopCube:2", 0.2, 0.2, 4)):
        rows.extend([point(401, decision, model, key) for _ in range(count)])
        rows.append(point(61, early, model, key))
    # 单侧多出来的一局不能进入配对分母。
    rows.append(point(401, 1, "a", "StopCube:99"))
    result = analysis.paired_comparison(rows, 1000)[0]
    assert result["metrics"]["decision"]["mean"] == pytest.approx(0.4)
    assert result["metrics"]["decision"]["n_episodes"] == 2
    assert result["metrics"]["separation"]["mean"] == pytest.approx(0.5)
    assert analysis.bootstrap([], 10)["mean"] is None
    assert analysis.bootstrap([1, 2], 100) == analysis.bootstrap([1, 2], 100)


def test_independent_trace_matches_state_and_step_without_oracle_z(tmp_path):
    query = {"episode_key": "StopCube:16", "env_step": 9, "raw_state": np.arange(8),
             "path": str(tmp_path / "nonce" / "query_0000.npz")}
    trace = tmp_path / "nonce" / "step_trace.jsonl"
    rows = [{"env_step_before": 9, "state_before": list(range(8)), "tcp_before": [0, 0, 0.37]}]
    assert analysis.synchronized_tcp(query, {"StopCube:16": [(trace, rows)]})[0] == 0.37
    wrong = copy.deepcopy(query)
    wrong["raw_state"] = np.zeros(8)
    with pytest.raises(ValueError, match="同步"):
        analysis.synchronized_tcp(wrong, {"StopCube:16": [(trace, rows)]})
    with pytest.raises(ValueError, match="多份 trace"):
        analysis.synchronized_tcp({**query, "path": str(tmp_path / "other" / "query_0.npz")},
            {"StopCube:16": [(trace, rows), (tmp_path / "another" / "step_trace.jsonl", rows)]})


def test_convert_uses_raw_state_and_every_prediction_row():
    class Reader:
        def __init__(self):
            self.calls = []

        def set_arm(self, q):
            self.calls.append(np.asarray(q))
            return [0, 0, q[0]]

    reader = Reader()
    query = {"raw_state": np.arange(8), "actions": np.arange(1600).reshape(10, 20, 8)}
    state_z, action_z = analysis.convert_arrays(reader, query)
    assert state_z == 0 and action_z.shape == (10, 20)
    assert action_z[9, 19] == 1592
    assert len(reader.calls) == 201


def make_cache(folder):
    table = {"schema_version": 1, "episodes": {"StopCube:16": entry()}}
    table_path = folder / "oracle_table.json"
    table_path.write_text(json.dumps(table))
    folder.mkdir(exist_ok=True)
    source = folder / "query_0000.npz"
    np.savez(source, marker=1)
    provenance = json.dumps({"queries": [analysis.fingerprint(source)], "traces": []})
    for model in ("baseline", "motion"):
        steps = np.arange(1, 481, 4)
        heights = np.full((len(steps), 10, 20), 0.16)
        heights[95:, :5 if model == "baseline" else 8, :16] = 0.12
        np.savez(folder / f"query_z_{model}_16.npz", model_id=model, episode_key="StopCube:16", env_step=steps,
            pre_reach=steps < 33, action_z=heights, state_z=np.full(len(steps), 0.16),
            actual_tcp_z=np.full(len(steps), 0.16), query_paths=np.asarray([str(source)] * len(steps)), provenance=provenance)
    return table_path


def test_summary_cli_cache_provenance_and_fixed_plan(tmp_path):
    table = make_cache(tmp_path)
    output = tmp_path / "analysis"
    assert analysis.main(["plan", "--cache-root", str(tmp_path), "--oracle-table", str(table),
        "--output-dir", str(output), "--bootstrap", "50"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["rows"]) == 240 and summary["fk"]["status"] == "pass"
    assert summary["comparisons"][0]["metrics"]["decision"]["n_episodes"] == 1
    assert all(not e["missing_steps"] for episodes in summary["coverage"].values() for e in episodes.values())
    assert (output / "probabilities.csv").is_file()
    # 固定后只有校准结果增加可以重汇总，不能改变已选点。
    assert analysis.main(["summarize", "--cache-root", str(tmp_path), "--oracle-table", str(table),
        "--output-dir", str(output), "--bootstrap", "50"]) == 0
    plan = json.loads((output / "calibration_plan.json").read_text())
    plan["threshold_cm"] = 3
    (output / "calibration_plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="已固定"):
        analysis.summarize(tmp_path, table, output, 20)
    np.savez(tmp_path / "query_0000.npz", marker=2)
    with pytest.raises(ValueError, match="已改变"):
        analysis.summarize_caches(tmp_path, analysis.load_table(table))


def test_calibration_results_keep_trigger_and_success_separate(tmp_path):
    root = tmp_path / "calibration" / "nonce"
    (root / "meta").mkdir(parents=True)
    trace = root / "step_trace.jsonl"
    trace.write_text(json.dumps({"event": "step", "env_step_after": 410, "button_pressed_after": True}) + "\n")
    receipt = {"model_id": "motion", "episode_key": "StopCube:16", "calibration_kind": "high_probability",
        "release_step": 373, "actual_release_step": 373, "task_success": 0, "trace_path": str(trace)}
    path = root / "meta" / "hold_receipt.json"
    path.write_text(json.dumps(receipt))
    rows = [point(step) for step in range(1, 481, 4)]
    table = {"StopCube:16": entry()}
    plan = analysis.make_plan(rows, table)
    results = analysis.calibration_results([tmp_path / "calibration"], table, plan)
    assert results[0]["trigger_within48"] is True
    assert results[0]["trigger_category"] == "早于区间"
    assert results[0]["task_success"] == 0
    counts = analysis.calibration_summary(results, plan)
    high = next(row for row in counts if row["kind"] == "high_probability")
    assert high["completed"] == 1 and high["task_success"] == 0
    assert next(row for row in counts if row["kind"] == "fixed_release")["missing_episodes"] == ["StopCube:16"]


def test_report_cli_renders_figures_and_escapes_model(tmp_path):
    table = make_cache(tmp_path)
    output = tmp_path / "output"
    analysis.summarize(tmp_path, table, output, 20)
    assert analysis.main(["report", "--summary", str(output / "summary.json"),
        "--output-dir", str(output), "--bootstrap", "20"]) == 0
    assert (output / "probability_curves.png").stat().st_size > 1000
    assert (output / "visit_groups.png").stat().st_size > 1000
    text = (output / "report.html").read_text()
    assert "目视复查" in text and "区间内" not in text  # 没有校准运行就不伪造结果。
    assert "按 M、目标次数和实际经过分组" in text
