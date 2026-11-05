"""Q1 触发分布与对齐点的 CPU 短测：中止数据硬验收 + 合成数据测边界。"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.interpret.analysis import rollout_trigger as rt


STORE = Path("/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation")
ABORTED_ROOT = STORE / "stopcube-decision-1104/normal"
ORACLE_TABLE = STORE / "stopcube-oracle-1104/oracle/oracle_table.json"
Z_HOVER = 0.15
_state = {"fixture": None, "passed": 0, "failed": 0}


class _UnitLine:
    """pytest 汇总行之后再打印判定行，保证它是真正的末行。"""

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            _state["passed"] += 1
        elif report.failed:
            _state["failed"] += 1

    def pytest_unconfigure(self, config):
        ok = _state["failed"] == 0 and _state["fixture"] is not None
        print(f"TRIGGER_UNIT={'PASS' if ok else 'FAIL'} tests={_state['passed']} fixture={_state['fixture']}", flush=True)


@pytest.fixture(scope="session", autouse=True)
def _register_unit_line(request):
    request.config.pluginmanager.register(_UnitLine(), "rollout-trigger-unit-line")


def table_entry(key, interval=60, target=2, half=5):
    center = interval * (target - 0.5)
    return {"episode_key": key, "move_interval": interval, "target_pass": target, "z_hover": Z_HOVER,
            "target_visit_in": int(center) - half, "target_visit_out": int(center) + half}


def cube_at(step, shift=0.0):
    return [0.001 * step + shift, -0.3, 0.01]


def make_rows(key, last_step, trigger=None, cross_step=None, stop=None, shift=0.0, block=16, hover_from=20):
    """env_step 从 1 起逐步推进；hover_from 起到悬停高度（None 表示不到位），cross_step 起低于悬停 2 cm。"""
    rows = []
    for index, step in enumerate(range(1, last_step)):
        after = step + 1
        z = Z_HOVER if hover_from is not None and after >= hover_from else 0.5
        if cross_step is not None and after >= cross_step:
            z = Z_HOVER - 0.05
        pressed = trigger is not None and after >= trigger
        rows.append({"event": "step", "episode_key": key, "infer_seq": index // block, "row_in_chunk": index % block,
                     "chunk_start_env_step": 1 + (index // block) * block, "env_step_before": step, "env_step_after": after,
                     "tcp_after": [0.0, 0.0, z], "state_before": [float(step)] * 8, "button_pressed_after": pressed,
                     "stop_timestep": stop if stop is not None and after >= stop else None,
                     "cube_position_after": cube_at(after, shift)})
    return rows


class World:
    """在临时目录造窗口表、window_probe 和两个模型的 run 目录。"""

    def __init__(self, tmp_path, keys=("StopCube:0",), probe_last=899, **entry):
        self.root = tmp_path / "root"
        self.table_path = tmp_path / "table.json"
        episodes = {}
        for key in keys:
            probe = tmp_path / "probe" / key.replace(":", "_") / "window_probe.jsonl"
            probe.parent.mkdir(parents=True)
            lines = [{"event": "meta"}] + [{"event": "step", "env_step_after": step, "cube_position_after": cube_at(step)}
                                           for step in range(2, probe_last + 1)]
            probe.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
            episodes[key] = {**table_entry(key, **entry), "trace_path": str(probe)}
        self.table_path.write_text(json.dumps({"schema_version": 1, "episodes": episodes}))
        self.attempts = 0

    def add(self, model, key, rows, *, outcome="fail", terminal=True, accepted=True, records=True, nonce=None, mtime=None):
        self.attempts += 1
        nonce = nonce or f"{self.attempts:08d}" + "a" * 24
        attempt = self.root / model / "run" / "attempts" / f"{self.attempts:03d}-chunk-x"
        trace = attempt / "client" / "traces" / nonce / "step_trace.jsonl"
        (trace.parent / "meta").mkdir(parents=True)
        trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        if mtime is not None:
            os.utime(trace, ns=(mtime, mtime))
        client = attempt / "client"
        (client / f"{key}.started.json").write_text(json.dumps({"episode_key": key, "episode_nonce": nonce}))
        if terminal:
            record = {"record_type": "terminal", "episode_key": key, "episode_nonce": nonce, "outcome": outcome,
                      "actions_executed": len(rows)}
            (client / f"{key}.terminal.json").write_text(json.dumps(record))
            (trace.parent / "meta" / "hold_receipt.json").write_text(json.dumps(
                {"outcome": outcome, "task_success": int(outcome == "success")}))
            if accepted:
                directory = self.root / model / "run" / "accepted"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{key}.{nonce[:8]}.terminal.json").write_text(json.dumps(record))
        if records:
            self.write_records(model, nonce, rows)
        return nonce

    def write_records(self, model, nonce, rows, skip=()):
        directory = self.root / model / "attn" / nonce
        directory.mkdir(parents=True, exist_ok=True)
        for item in rt.group_inferences(rows):
            if item["infer_seq"] not in skip:
                np.savez(directory / f"rollout_{item['infer_seq']:04d}.npz", infer_seq=np.int64(item["infer_seq"]),
                         step_idx=np.int64(item["chunk_start_env_step"] - 1))

    def compute(self, models=("ctx", "mot"), **kwargs):
        return rt.compute(self.root, self.table_path, models=models, **kwargs)


def entry_for_align(**kwargs):
    entry = table_entry("StopCube:0", **kwargs)
    return {"move_interval": entry["move_interval"], "target_pass": entry["target_pass"], "z_hover": Z_HOVER,
            "window_in": entry["target_visit_in"], "window_out": entry["target_visit_out"],
            "window_center": entry["move_interval"] * (entry["target_pass"] - 0.5)}


def test_module_points_at_this_checkout():
    assert Path(rt.__file__).resolve().parents[3] == Path(__file__).resolve().parents[2]


def test_aborted_motion_rollouts_reproduce_plan_numbers():
    """硬验收：中止数据的 motion 轨迹，28 局完整为 13/10/4/1，第 28 局中断另计。"""
    keys = [f"StopCube:{index}" for index in range(29)]
    results, _ = rt.compute(ABORTED_ROOT, ORACLE_TABLE, models=("gpu1",), episodes=keys, require_records=False)
    summary = rt.distribution(results)
    item = summary["models"]["gpu1"]
    assert item["n_valid"] == item["n_q1"] == 28 and item["n_invalid"] == 1
    assert [entry["episode_key"] for entry in item["invalid"]] == ["StopCube:28"]
    assert "无终局" in item["invalid"][0]["reason"]
    assert item["counts"] == {"窗口内": 13, "偏晚": 10, "偏早": 4, "未按": 1}
    assert item["cube_maxabs"] == 0.0 and item["uncovered_after_window"] == 0
    # 到位口径：29 条轨迹（含中断的第 28 局）都落在第 46～51 步。
    reach = []
    for path in ABORTED_ROOT.glob("gpu1/run/attempts/*/client/traces/*/step_trace.jsonl"):
        rows = rt.read_steps(path)
        reach.append(rt.reach_step(rows, rt.load_table(ORACLE_TABLE)[rows[0]["episode_key"]]["z_hover"]))
    assert len(reach) == 29 and all(46 <= step <= 51 for step in reach)
    print(rt.reach_line("gpu1", item["reach"]))
    valid =[episode for episode in results["gpu1"].values() if episode["valid"]]
    after = [episode["steps_after_trigger"] for episode in valid if episode["trigger_step"] is not None]
    assert len(after) == 27 and set(after) == {0, 2}
    gaps = [episode["I_trigger"] - episode["I_cross"] for episode in valid if episode["trigger_step"] is not None]
    assert gaps.count(1) == 21 and gaps.count(0) == 6
    for episode in valid:
        starts = [point["chunk_start_env_step"] for point in episode["inferences"]]
        assert set(np.diff(starts).tolist()) == {16}
        assert all(point["state_before"] is not None for point in episode["inferences"])
    # 第 2 局提前按下，stop_timestep 为空也要按触发步分类。
    assert results["gpu1"]["StopCube:2"]["stop_timestep"] is None
    assert results["gpu1"]["StopCube:2"]["category"] == "偏早"
    lines, passed = rt.verdicts(summary)
    assert passed
    assert lines[0] == "ROLLOUT_RUN=PASS models=1 valid_gpu1=28 invalid=1 infra_retries=0"
    assert lines[1] == ("WINDOW_TABLE_MATCH=PASS models=1 checked=28 mismatched=0 "
                        "uncovered_after_window=0 cube_maxabs=0 tol=1e-6")
    assert lines[2].endswith("n_gpu1=28 paired=0 gpu1=13/10/4/1")
    _state["fixture"] = rt.format_counts(item["counts"])


@pytest.mark.parametrize("trigger,expected", [(None, "未按"), (84, "偏早"), (85, "窗口内"), (90, "窗口内"),
                                              (95, "窗口内"), (96, "偏晚")])
def test_category_boundaries(trigger, expected):
    assert rt.classify(trigger, entry_for_align()) == expected


def test_trigger_is_button_not_stop_timestep():
    rows = make_rows("StopCube:0", 100, trigger=88, stop=92)
    point = rt.align_points(rows, entry_for_align())
    assert point["trigger_step"] == 88 and point["stop_timestep"] == 92
    assert point["trigger_minus_center"] == -2 and point["category"] == "窗口内"
    assert point["steps_after_trigger"] == 100 - 88


def test_cross_alignment_and_missing_earlier_inferences():
    late = rt.align_points(make_rows("StopCube:0", 100, trigger=88, cross_step=70), entry_for_align())
    # 第 70 步对应 env_step_before=69，下标 68，属于第 4 个动作块（68 // 16）。
    assert late["I_cross"] == 4 and late["I_trigger"] == 5
    assert [late[f"I_cross_minus_{k}"] for k in (1, 2, 3)] == [3, 2, 1]
    early = rt.align_points(make_rows("StopCube:0", 100, trigger=88, cross_step=25), entry_for_align())
    assert early["I_cross"] == 1
    assert [early[f"I_cross_minus_{k}"] for k in (1, 2, 3)] == [0, None, None]
    none = rt.align_points(make_rows("StopCube:0", 60), entry_for_align())
    assert none["I_cross"] is None and none["I_cross_minus_1"] is None and none["I_trigger"] is None


def test_reach_and_inference_table():
    point = rt.align_points(make_rows("StopCube:0", 40), entry_for_align())
    assert point["reach_step"] == 20
    assert [item["chunk_start_env_step"] for item in point["inferences"]] == [1, 17, 33]
    assert point["inferences"][1]["state_before"] == [17.0] * 8 and point["inferences"][1]["rows"] == 16


def test_near_only_for_unpressed_and_missing_with_reason():
    covered = rt.align_points(make_rows("StopCube:0", 120), entry_for_align())
    # 窗口中心 90：env_step_before=90 是下标 89，属于第 5 个动作块。
    assert covered["category"] == "未按" and covered["I_near"] == 5 and covered["I_near_reason"] is None
    short = rt.align_points(make_rows("StopCube:0", 60), entry_for_align())
    assert short["I_near"] is None and "没有覆盖窗口中心" in short["I_near_reason"]
    pressed = rt.align_points(make_rows("StopCube:0", 120, trigger=88), entry_for_align())
    assert pressed["I_near"] is None and pressed["I_near_reason"] == "只给未按的局"


def two_model_world(tmp_path, count=4):
    keys = [f"StopCube:{index}" for index in range(count)]
    world = World(tmp_path, keys)
    for key in keys:
        world.add("ctx", key, make_rows(key, 100, trigger=90))
        world.add("mot", key, make_rows(key, 100, trigger=90))
    return world, keys


def test_invalid_and_mismatched_denominators(tmp_path):
    keys = [f"StopCube:{index}" for index in range(4)]
    world = World(tmp_path, keys)
    for key in keys[:2]:
        world.add("ctx", key, make_rows(key, 100, trigger=90), outcome="success")
        world.add("mot", key, make_rows(key, 100, trigger=91), outcome="success")
    # ctx 第 2 局方块不符，第 3 局无终局；mot 两局都有效。
    world.add("ctx", keys[2], make_rows(keys[2], 100, trigger=90, shift=1e-3))
    world.add("ctx", keys[3], make_rows(keys[3], 100, trigger=90), terminal=False)
    world.add("mot", keys[2], make_rows(keys[2], 100, trigger=91))
    world.add("mot", keys[3], make_rows(keys[3], 100, trigger=91))
    results, _ = world.compute()
    summary = rt.distribution(results)
    ctx, mot = summary["models"]["ctx"], summary["models"]["mot"]
    assert (ctx["n_valid"], ctx["n_q1"], ctx["n_invalid"]) == (3, 2, 1)
    assert (mot["n_valid"], mot["n_q1"], mot["n_invalid"]) == (4, 4, 0)
    assert [item["episode_key"] for item in ctx["mismatched"]] == [keys[2]]
    assert sum(ctx["counts"].values()) == 2
    assert [row["episode_key"] for row in summary["paired"]["episodes"]] == keys[:2]
    mismatched = results["ctx"][keys[2]]
    assert mismatched["uses"] == {"q1": False, "q2_window": False, "q2_other": True, "q3": True, "q4": False}
    assert mismatched["category"] is None and mismatched["trigger_minus_center"] is None
    assert mismatched["I_near"] is None
    assert mismatched["trigger_step"] == 90 and mismatched["I_trigger"] is not None
    assert not any(results["ctx"][keys[3]]["uses"].values())
    lines, passed = rt.verdicts(summary)
    assert passed
    assert lines[1].startswith("WINDOW_TABLE_MATCH=PASS models=2 checked=7 mismatched=1 "
                               "uncovered_after_window=0 cube_maxabs=0.001 ")
    assert "valid_ctx=3 valid_mot=4 invalid=1" in lines[0]
    assert "n_ctx=2 n_mot=4 paired=2" in lines[2]


def test_cube_compare_stops_at_trigger(tmp_path):
    """触发那一步起方块可以和窗口表不同，不算不符。"""
    world = World(tmp_path)
    rows = make_rows("StopCube:0", 100, trigger=90)
    for row in rows:
        if row["button_pressed_after"]:
            row["cube_position_after"] = [9.0, 9.0, 9.0]
    world.add("ctx", "StopCube:0", rows)
    results, _ = world.compute(models=("ctx",))
    assert results["ctx"]["StopCube:0"]["window_table_match"] is True


def single_episode(tmp_path, rows, **world):
    instance = World(tmp_path, **world)
    instance.add("ctx", "StopCube:0", rows)
    results, _ = instance.compute(models=("ctx",))
    return results, rt.distribution(results)


def test_short_probe_inside_window_is_mismatch(tmp_path):
    """窗口 85～95，探针只到第 80 步，未触发：窗口内有未覆盖步，不能空 PASS。"""
    results, summary = single_episode(tmp_path, make_rows("StopCube:0", 100), probe_last=80)
    item = results["ctx"]["StopCube:0"]
    assert item["cube_maxabs"] == 0.0 and item["cube_uncovered_in_window"] == 15
    assert item["window_table_match"] is False and item["uses"]["q1"] is False
    assert summary["models"]["ctx"]["mismatched"][0]["cube_uncovered_in_window"] == 15


def test_nothing_compared_is_mismatch(tmp_path):
    """第一步就触发，一步都没比到。"""
    results, _ = single_episode(tmp_path, make_rows("StopCube:0", 100, trigger=2))
    item = results["ctx"]["StopCube:0"]
    assert item["cube_compared_steps"] == 0 and item["window_table_match"] is False


def test_uncovered_after_window_only_counted(tmp_path):
    results, summary = single_episode(tmp_path, make_rows("StopCube:0", 120), probe_last=95)
    item = results["ctx"]["StopCube:0"]
    assert item["window_table_match"] is True and item["cube_uncovered_after_window"] == 25
    assert summary["models"]["ctx"]["uncovered_after_window"] == 25
    assert " uncovered_after_window=25 " in rt.verdicts(summary)[0][1]


def test_uncovered_after_trigger_not_counted(tmp_path):
    """探针短于窗口，但触发更早：触发后不比。"""
    results, _ = single_episode(tmp_path, make_rows("StopCube:0", 100, trigger=70), probe_last=80)
    item = results["ctx"]["StopCube:0"]
    assert item["window_table_match"] is True and item["cube_uncovered_in_window"] == 0


def test_reach_tolerance_and_cross_are_independent():
    rows = make_rows("StopCube:0", 40, hover_from=None)
    rows[9]["tcp_after"][2] = Z_HOVER + 0.016
    assert rt.reach_step(rows, Z_HOVER) is None
    rows[9]["tcp_after"][2] = Z_HOVER - 0.014
    assert rt.reach_step(rows, Z_HOVER) == rows[9]["env_step_after"]
    # 低于悬停 1.4 cm 到位了，但还不算下压。
    assert rt.align_points(rows, entry_for_align())["I_cross"] is None


def test_reach_statistics_and_line(tmp_path):
    keys = [f"StopCube:{index}" for index in range(4)]
    world = World(tmp_path, keys)
    world.add("ctx", keys[0], make_rows(keys[0], 100, hover_from=20, cross_step=70))
    world.add("ctx", keys[1], make_rows(keys[1], 100, hover_from=30, cross_step=70))
    # 到位第 84 步不早于 I_cross 那次推理的起点第 81 步。
    world.add("ctx", keys[2], make_rows(keys[2], 100, hover_from=84, cross_step=86))
    world.add("ctx", keys[3], make_rows(keys[3], 100, hover_from=None))
    results, _ = world.compute(models=("ctx",))
    assert results["ctx"][keys[3]]["reach_step"] is None
    reach = rt.distribution(results)["models"]["ctx"]["reach"]
    assert reach == {"min": 20.0, "median": 30.0, "max": 84.0, "not_reached": 1, "reach_after_cross": 1}
    assert rt.reach_line("ctx", reach) == "REACH_STATS model=ctx min=20 median=30 max=84 not_reached=1 reach_after_cross=1"
    assert rt.reach_line("ctx", rt.reach_statistics([])).endswith("min=null median=null max=null not_reached=0 reach_after_cross=0")


@pytest.mark.parametrize("bad,expected", [(5, "PASS"), (6, "FAIL")])
def test_record_problems_reported_and_fail_with_run_line(tmp_path, bad, expected):
    keys = [f"StopCube:{index}" for index in range(8)]
    world = World(tmp_path, keys)
    for index, key in enumerate(keys):
        world.add("ctx", key, make_rows(key, 100, trigger=90), records=index >= bad)
    results, checks = world.compute(models=("ctx",))
    assert rt.record_verdict(checks) == (f"ROLLOUT_RECORD={expected} models=1 valid_episodes={8 - bad} "
                                         f"inferences={(8 - bad) * 7} missing={bad} readback=1")
    assert rt.verdicts(rt.distribution(results))[0][0].startswith(f"ROLLOUT_RUN={expected} ")


@pytest.mark.parametrize("bad,expected", [(5, "PASS"), (6, "FAIL")])
def test_more_than_five_invalid_fails_run_line(tmp_path, bad, expected):
    keys = [f"StopCube:{index}" for index in range(8)]
    world = World(tmp_path, keys)
    for index, key in enumerate(keys):
        world.add("ctx", key, make_rows(key, 100, trigger=90), terminal=index >= bad)
        world.add("mot", key, make_rows(key, 100, trigger=90))
    results, _ = world.compute()
    lines, passed = rt.verdicts(rt.distribution(results))
    assert lines[0].startswith(f"ROLLOUT_RUN={expected} ")
    assert passed is (expected == "PASS")


@pytest.mark.parametrize("bad,expected", [(5, "PASS"), (6, "FAIL")])
def test_more_than_five_mismatched_fails_match_line(tmp_path, bad, expected):
    keys = [f"StopCube:{index}" for index in range(8)]
    world = World(tmp_path, keys)
    for index, key in enumerate(keys):
        world.add("ctx", key, make_rows(key, 100, trigger=90))
        world.add("mot", key, make_rows(key, 100, trigger=90, shift=1e-3 if index < bad else 0.0))
    results, _ = world.compute()
    lines, passed = rt.verdicts(rt.distribution(results))
    assert lines[1].startswith(f"WINDOW_TABLE_MATCH={expected} ")
    assert lines[0].startswith("ROLLOUT_RUN=PASS") and lines[2].startswith("TRIGGER_DISTRIBUTION=PASS")
    assert passed is (expected == "PASS")


def test_missing_episode_counts_as_invalid(tmp_path):
    world = World(tmp_path, ("StopCube:0", "StopCube:1"))
    world.add("ctx", "StopCube:0", make_rows("StopCube:0", 100, trigger=90))
    results, _ = world.compute(models=("ctx",))
    assert results["ctx"]["StopCube:1"]["invalid_reason"] == "没有轨迹"
    # 冒烟只给指定局时，其余局不算无效。
    only, _ = world.compute(models=("ctx",), episodes=["StopCube:0"])
    assert list(only["ctx"]) == ["StopCube:0"]


def test_retry_uses_accepted_latest_and_counts_retries(tmp_path):
    world = World(tmp_path)
    world.add("ctx", "StopCube:0", make_rows("StopCube:0", 60), terminal=False, mtime=10**18)
    second = world.add("ctx", "StopCube:0", make_rows("StopCube:0", 100, trigger=90), mtime=2 * 10**18)
    results, _ = world.compute(models=("ctx",))
    item = results["ctx"]["StopCube:0"]
    assert item["nonce"] == second and item["retries"] == 1 and item["tries"] == 2 and item["valid"]
    summary = rt.distribution(results)
    assert summary["models"]["ctx"]["infra_retries"] == 1


def test_accepted_beats_newer_unaccepted(tmp_path):
    world = World(tmp_path)
    first = world.add("ctx", "StopCube:0", make_rows("StopCube:0", 100, trigger=90), mtime=10**18)
    world.add("ctx", "StopCube:0", make_rows("StopCube:0", 100, trigger=88), accepted=False, mtime=2 * 10**18)
    results, _ = world.compute(models=("ctx",))
    assert results["ctx"]["StopCube:0"]["nonce"] == first


def test_terminal_with_wrong_nonce_is_invalid(tmp_path):
    world = World(tmp_path)
    nonce = world.add("ctx", "StopCube:0", make_rows("StopCube:0", 100, trigger=90))
    path = world.root / "ctx/run/attempts/001-chunk-x/client/StopCube:0.terminal.json"
    record = json.loads(path.read_text())
    record["episode_nonce"] = "other" + nonce[5:]
    path.write_text(json.dumps(record))
    results, _ = world.compute(models=("ctx",))
    assert results["ctx"]["StopCube:0"]["valid"] is False
    assert "nonce" in results["ctx"]["StopCube:0"]["invalid_reason"]


def test_check_records_reports_missing_and_demotes(tmp_path):
    world = World(tmp_path, ("StopCube:0", "StopCube:1"))
    rows = make_rows("StopCube:0", 100, trigger=90)
    nonce = world.add("ctx", "StopCube:0", rows)
    world.add("ctx", "StopCube:1", make_rows("StopCube:1", 100, trigger=90))
    (world.root / "ctx" / "attn" / nonce / "rollout_0002.npz").unlink()
    results, checks = world.compute(models=("ctx",))
    problems = checks["ctx"]["problems"]
    assert list(problems) == ["StopCube:0"] and "rollout_0002.npz" in problems["StopCube:0"][0]
    assert results["ctx"]["StopCube:0"]["valid"] is False
    assert "输入记录不完整" in results["ctx"]["StopCube:0"]["invalid_reason"]
    assert results["ctx"]["StopCube:1"]["valid"] is True
    summary = rt.distribution(results)
    assert summary["models"]["ctx"]["n_valid"] == 1 and summary["models"]["ctx"]["n_invalid"] == 1


def test_check_records_detects_extra_duplicate_and_wrong_seq(tmp_path):
    directory = tmp_path / "abc"
    directory.mkdir()
    episode = {"episode_key": "StopCube:0", "valid": True, "nonce": "abc", "inferences": [
        {"infer_seq": 0, "chunk_start_env_step": 1}, {"infer_seq": 1, "chunk_start_env_step": 17}]}
    assert "缺文件" in rt.check_records(tmp_path, [episode])["problems"]["StopCube:0"][0]
    np.savez(directory / "rollout_0000.npz", infer_seq=np.int64(0), step_idx=np.int64(0))
    np.savez(directory / "rollout_0001.npz", infer_seq=np.int64(1), step_idx=np.int64(16))
    good = rt.check_records(tmp_path, [episode])
    assert good["problems"] == {} and (good["episodes"], good["inferences"]) == (1, 2)
    np.savez(directory / "rollout_0001.npz", infer_seq=np.int64(1), step_idx=np.int64(0))
    assert any("重复" in item for item in rt.check_records(tmp_path, [episode])["problems"]["StopCube:0"])
    np.savez(directory / "rollout_0001.npz", infer_seq=np.int64(7), step_idx=np.int64(16))
    assert any("序号不符" in item for item in rt.check_records(tmp_path, [episode])["problems"]["StopCube:0"])
    np.savez(directory / "rollout_0001.npz", infer_seq=np.int64(1), step_idx=np.int64(16))
    np.savez(directory / "rollout_0002.npz", infer_seq=np.int64(2), step_idx=np.int64(32))
    assert any("多余文件" in item for item in rt.check_records(tmp_path, [episode])["problems"]["StopCube:0"])


def test_paired_bootstrap_and_official_disagreement(tmp_path):
    keys = [f"StopCube:{index}" for index in range(6)]
    world = World(tmp_path, keys)
    for index, key in enumerate(keys):
        # ctx 全部窗口内，mot 前 3 局偏晚；第 0 局 ctx 窗口内却判失败，第 1 局 mot 偏晚却判成功。
        world.add("ctx", key, make_rows(key, 110, trigger=90), outcome="fail" if index == 0 else "success")
        world.add("mot", key, make_rows(key, 110, trigger=100 if index < 3 else 90), outcome="success")
    results, _ = world.compute()
    summary = rt.distribution(results)
    pair = summary["paired"]
    assert pair["direction"] == "mot_minus_ctx" and len(pair["episodes"]) == 6
    assert pair["window_in_rate_diff"]["mean"] == pytest.approx(-0.5)
    assert pair["trigger_minus_center_diff"]["mean"] == pytest.approx(5.0)
    assert pair["bootstrap"] == {"repeats": 2000, "seed": 42, "unit": "episode"}
    assert summary["paired"] == rt.distribution(results)["paired"]
    assert [item["episode_key"] for item in summary["models"]["ctx"]["window_vs_official_disagree"]] == [keys[0]]
    assert [item["episode_key"] for item in summary["models"]["mot"]["window_vs_official_disagree"]] == keys[:3]
    assert summary["models"]["ctx"]["task_success"] == {"n": 6, "success": 5}
    assert summary["models"]["mot"]["by_move_interval"]["60"]["偏晚"] == 3


def test_main_writes_outputs_and_exit_code(tmp_path, capsys):
    world, _ = two_model_world(tmp_path)
    code = rt.main(["--root", str(world.root), "--table", str(world.table_path)])
    captured = capsys.readouterr().out
    assert code == 0
    assert "ROLLOUT_RUN=PASS models=2 valid_ctx=4 valid_mot=4 invalid=0 infra_retries=0" in captured
    assert "TRIGGER_DISTRIBUTION=PASS models=2 n_ctx=4 n_mot=4 paired=4 ctx=4/0/0/0 mot=4/0/0/0" in captured
    assert "ROLLOUT_RECORD=PASS models=2 valid_episodes=8 inferences=56 missing=0 readback=1" in captured
    assert "REACH_STATS model=ctx min=20 median=20 max=20 not_reached=0 reach_after_cross=0" in captured
    assert "RECORD_CHECK" not in captured
    analysis = world.root / "analysis"
    align = json.loads((analysis / "align.json").read_text())
    assert align["models"]["ctx"]["StopCube:0"]["I_trigger"] == 5
    assert "rows" not in align["models"]["ctx"]["StopCube:0"]
    summary = json.loads((analysis / "trigger-summary.json").read_text())
    assert summary["verdicts"][0].startswith("ROLLOUT_RUN=PASS")
    assert summary["verdicts"][3].startswith("ROLLOUT_RECORD=PASS")
    assert summary["models"]["mot"]["reach"]["max"] == 20.0
    assert (analysis / "trigger-distribution.png").stat().st_size > 0


def test_main_nonzero_exit_on_fail(tmp_path, capsys):
    keys = [f"StopCube:{index}" for index in range(8)]
    world = World(tmp_path, keys)
    for key in keys:
        world.add("ctx", key, make_rows(key, 100, trigger=90))
        world.add("mot", key, make_rows(key, 100, trigger=90), terminal=False)
    assert rt.main(["--root", str(world.root), "--table", str(world.table_path)]) == 1
    assert "ROLLOUT_RUN=FAIL" in capsys.readouterr().out
    assert (world.root / "analysis" / "align.json").exists()
