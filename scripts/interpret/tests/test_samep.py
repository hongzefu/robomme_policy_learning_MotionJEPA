"""计划 1115 §4.2 同一局供体的 CPU 单测：spec 重建、变体合法性、stride16 窗口、pred_k←k′、截断参数（不开仿真）。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.interpret import fp_swap_prep as F
from scripts.interpret import run_local as local
from scripts.interpret import samep_rs_variants as V
from scripts.interpret import sp_swap_prep as S
from scripts.interpret import subgoal_trace_runner as runner

# docs/evaluation-doc/steer-4task-1115/records/spec-probe/probe-run1-layout-frames.txt 的 SPEC 行
PROBE = {
    (3100003, "hard"): ([8, 6, 4, 2, 4, 2, 0, 2], [4, 3, 2, 1, 2, 1, 0, 1],
                        ["counterclockwise", "clockwise", "clockwise", "counterclockwise", "clockwise",
                         "counterclockwise", "clockwise"], 6.686),
    (3100010, "medium"): ([4, 2, 0, 2, 4], [2, 1, 0, 1, 2],
                          ["clockwise", "counterclockwise", "counterclockwise", "counterclockwise"], -4.074),
    (3100011, "easy"): ([8, 6, 4], [4, 3, 2], ["clockwise", "clockwise"], 24.25),
}


@pytest.fixture(autouse=True)
def _reset_grid():
    yield
    S.apply_grid()
    F.apply_grid()


# ---------------------------------------------------------------- spec 重建与变体
@pytest.mark.parametrize("key", sorted(PROBE))
def test_rebuild_matches_probe(key):
    nodes, slots, dirs, theta = PROBE[key]
    spec = V.rebuild_spec(*key)
    assert spec["task"] == "RouteStick" and spec["difficulty"] == key[1]
    assert spec["actions"]["nodes"] == nodes and spec["actions"]["node_slots"] == slots
    assert spec["actions"]["directions"] == dirs and spec["objects"]["L"] == len(dirs)
    assert round(spec["layout"]["rotation_deg"], 3) == pytest.approx(theta, abs=1e-3)
    assert len(spec["layout"]["obstacle_rgb"]) == 4 and V.check_spec(spec, key[1]) == []


def test_side_variant_matches_probe_and_is_legal():
    spec = V.rebuild_spec(3100003, "hard")
    v, back = V.make_variant(spec, 2, "S_side")
    assert back and v["actions"]["nodes"] == [8, 6, 8, 6, 8, 6, 4, 6]          # probe flip_side_j
    assert v["actions"]["directions"] == spec["actions"]["directions"]
    assert V.check_spec(v, "hard") == []
    med = V.rebuild_spec(3100010, "medium")
    vm, _ = V.make_variant(med, 2, "S_side")
    assert vm["actions"]["nodes"] == [4, 2, 4, 6, 8]                           # probe flip_side_j
    assert V.check_spec(vm, "medium") != [] and V.check_spec(vm, "medium", allow_backtrack=True) == []


def test_endpoint_skip_and_swing_flip():
    easy = V.rebuild_spec(3100011, "easy")                 # 起点槽 4 → j=1 端点
    assert V.make_variant(easy, 1, "S_side") is None and V.make_variant(easy, 1, "S_both") is None
    sw, back = V.make_variant(easy, 1, "S_swing")
    assert not back and sw["actions"]["directions"] == ["counterclockwise", "clockwise"]
    both, _ = V.make_variant(V.rebuild_spec(3100003, "hard"), 2, "S_both")
    assert both["actions"]["directions"][1] == "counterclockwise" and both["actions"]["nodes"][2] == 8


@pytest.mark.parametrize("slots", [[0, 1, 2, 3, 4], [4, 3, 4, 3, 2, 1, 0, 1, 0], [2, 1, 0, 1, 2, 3]])
def test_side_slots_always_adjacent_in_range(slots):
    for j in range(1, len(slots)):
        new = V.side_slots(slots, j)
        if slots[j - 1] in (0, 4):
            assert new is None
            continue
        assert len(new) == len(slots) and new[:j] == slots[:j] and new[j] != slots[j]
        assert all(0 <= s <= 4 for s in new) and all(abs(a - b) == 1 for a, b in zip(new, new[1:]))


def test_check_spec_catches_bad_slots():
    spec = V.rebuild_spec(3100010, "medium")
    spec["actions"]["node_slots"][0] = -1
    assert any("越界" in p for p in V.check_spec(spec, "medium"))


def test_content_segments_and_seglen():
    labels = [1] * 50 + [2] * 50 + [3] * 50 + [4] * 50 + [7]          # 3100010 medium 实测形状
    segs, why = V.segments_from_labels(labels, 4)
    assert why is None and segs == [(1, 50), (51, 100), (101, 150), (151, 200)] == V.content_segments(4)
    assert V.seglen_ok(201, segs, 4) and not V.seglen_ok(200, segs, 4)
    assert S.content_segments([(0, 49), (50, 99)]) == [(1, 50), (51, 100)]


def test_swing_and_post():
    a, b = (0.0, 0.0), (1.0, 0.0)
    assert V.swing_of([(0.5, 0.2)], a, b) == "clockwise" and V.swing_of([(0.5, -0.2)], a, b) == "counterclockwise"
    assert V.post_of(4, 6) == (2, 5) and V.post_of(2, 0) == (0, 1)
    assert V.first_diff(["a", "b"], ["a", "c"]) == 1 and V.first_diff(["a"], ["a"]) is None


# ---------------------------------------------------------------- 窗口
def test_stride16_windows_all_and_core():
    S.apply_grid({"stride": 16, "window": 33, "demo_min_real": 17})
    w_all, w_core = S.window_sets((51, 100), 150)
    assert w_all == [32, 48, 64, 80, 96] and w_core == [64]
    S.apply_grid()
    assert S.demo_starts(150) == [0, 32, 64, 96, 128] and S.GRID == S.GRID_DEFAULT


def test_fp_stride_param():
    F.apply_grid({"stride": 16, "window": 33, "demo_min_real": 17, "max_steps": 2000})
    assert F.demo_starts(100)[:3] == [0, 16, 32] and F.exec_starts(100)[:2] == [100, 116]
    assert F.exec_starts(100)[-1] + 32 <= 100 + 2000
    F.apply_grid()
    assert F.demo_starts(100) == [0, 32, 64]


def test_motion_from_yaml(tmp_path):
    (tmp_path / "history_config.resolved.yaml").write_text(
        "motion:\n  stride: 16\n  window_frames: 33\n  demo_min_real_frames: 17\n")
    (tmp_path / "50000").mkdir()
    m = S.motion_from_yaml(tmp_path / "50000")
    assert (m["stride"], m["window"], m["demo_min_real"]) == (16, 33, 17)


# ---------------------------------------------------------------- VideoRepick pred_k←k′
def test_pred_index_and_blind():
    sched = [[0, 1, 10, 20], [1, 2, 20, 30]]                    # 3 物体，目标 0
    # 实际：物体 0→槽 1、物体 1→槽 2、物体 2→槽 0
    assert F.pred_index_k(3, 0, sched, 1, 1) == 0               # k′=k：目标本身
    # 第 1 次换成第 2 次那对 (1,2)：两次 (1,2) 抵消，信念目标在槽 0；实际槽 0 上是物体 2
    assert F.pred_index_k(3, 0, sched, 1, 2) == 2
    assert F.swap_blind_k(3, 0, sched, 1) == 2
    # 第 2 次换成第 1 次那对 (0,1)：两次 (0,1) 抵消，信念目标在槽 0 → 物体 2
    assert F.pred_index_k(3, 0, sched, 2, 1) == 2
    # 只漏第 2 次：信念目标在槽 1 → 实际槽 1 上是目标本身（无区分力）
    assert F.swap_blind_k(3, 0, sched, 2) == 0
    assert F.slot_pairs(3, sched) == [(0, 1), (0, 2)]
    assert F.pred_index_slot_k(3, 0, sched, 2, 1) == 2


def test_samep_ep_codes():
    assert F.samep_ep(3, 2, 1) == 20000321 and F.samep_ep(3, 2, 0) == 20000320
    ep = S.samep_ep(7, 10, "S_both_core")
    assert S.parse_samep(ep) == (7, 10, "S_both_core")
    assert len({S.samep_ep(1, j, g) for j in range(1, 11) for g in S.SAMEP_GROUP_NO}) == 70


def test_assign_samep_groups(monkeypatch):
    S.apply_grid({"stride": 16, "window": 33, "demo_min_real": 17})
    own = {s: np.full(4, s, np.float32) for s in S.demo_starts(150)}
    var = {s: v + 1000 for s, v in own.items()}
    info = {"task": "RouteStick", "ep": 0, "m": 3, "status": "ok", "record": "x", "outcome": "fail", "completed": 1,
            "W_all": [S.window_sets(sg, 150)[0] for sg in [(1, 50), (51, 100), (101, 150)]],
            "W_core": [S.window_sets(sg, 150)[1] for sg in [(1, 50), (51, 100), (101, 150)]]}
    vrec = {("RouteStick:0", 1, "S_swing"): var, ("RouteStick:0", 2, "S_swing"): var, ("RouteStick:0", 2, "S_side"): own}
    monkeypatch.setattr(S, "load_record", lambda p: own)        # 接收局自身编码
    runs, notes = S.assign_samep({"RouteStick:0": info}, vrec, [], groups=("S_swing", "S_side", "Z"))
    got = {(r["j"], r["group"]) for r in runs}
    assert (1, "S_swing") in got and (2, "S_swing") in got and (2, "Z") in got and (2, "S_swing_core") in got
    assert (3, "Z") not in got                                     # N 只完成 1 步 → 只到 j=2
    assert "RouteStick:0:2:S_side" in notes["same_as_recv"]
    r = next(x for x in runs if (x["j"], x["group"]) == (2, "S_swing"))
    assert r["w_starts"] == [32, 48, 64, 80, 96] and float(r["embs"][0, 0]) == 1032.0


# ---------------------------------------------------------------- 截断与 spec 注入参数
def base_spec(tmp_path, **changes):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": [{"task_name": "RouteStick"}]}))
    values = dict(repo="/repo", policy_python="/p", sim_python="/s", stage2_python="/s2", checkpoint="/ckpt",
                  policy_config="cfg", model_kind="stage2_pool4_concat", cuda="0", attn_out="/o/a", run_root="/o/r",
                  manifest=str(manifest), xla_mem_fraction=0.5, model_seed=42)
    return values | changes


def load(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return local.load_spec(path)


def test_run_local_truncation_keys(tmp_path):
    spec = load(tmp_path, base_spec(tmp_path, subgoal_trace=True, stop_after_step=3))
    assert spec["stop_after_step"] == 3
    with pytest.raises(SystemExit, match="只许与 subgoal_trace"):
        load(tmp_path, base_spec(tmp_path, stop_after_step=2))
    with pytest.raises(SystemExit, match="正整数"):
        load(tmp_path, base_spec(tmp_path, subgoal_trace=True, stop_after_step=0))
    with pytest.raises(SystemExit, match="互斥"):
        load(tmp_path, base_spec(tmp_path, subgoal_trace=True, stop_after_step=1, stop_after_first_infer=True))
    with pytest.raises(SystemExit, match="只许与 subgoal_trace"):
        load(tmp_path, base_spec(tmp_path, stop_after_first_infer=True))


def test_run_local_episode_specs_requires_plfull(tmp_path):
    table = tmp_path / "specs.json"
    table.write_text("{}")
    with pytest.raises(SystemExit, match="profile=plfull"):
        load(tmp_path, base_spec(tmp_path, subgoal_trace=True, episode_specs=str(table)))


def test_run_local_client_flags_and_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(local.RW, "client_command", lambda *a: ["py", "eval.py"])
    spec = base_spec(tmp_path, subgoal_trace=True, stop_after_step=2, episode_specs="/x/specs.json")
    cmd = local.build_client_command(spec, "g", "o", "n", 1, "d")
    assert cmd[1].endswith("subgoal_trace_runner.py")
    assert cmd[-4:] == ["--stop-after-step", "2", "--episode-specs", "/x/specs.json"]
    spec = base_spec(tmp_path, subgoal_trace=True, stop_after_first_infer=True)
    assert local.build_client_command(spec, "g", "o", "n", 1, "d")[-1] == "--stop-after-first-infer"
    plain = local.build_client_command(base_spec(tmp_path, subgoal_trace=True), "g", "o", "n", 1, "d")
    assert plain == ["py", "/repo/scripts/interpret/subgoal_trace_runner.py"]


def test_runner_truncation_logic():
    with pytest.raises(runner.official.ContractError):
        runner.check_truncation_args(2, True)
    with pytest.raises(runner.official.ContractError):
        runner.check_truncation_args(-1, False)
    assert runner.crossed_step(7, 3, 1) and not runner.crossed_step(6, 3, 1) and not runner.crossed_step(None, 3, 1)
    fake = SimpleNamespace(receipt={"episode_nonce": "n1", "m": 3}, stop_after_first_infer=False, stop_after_step=2)
    obs = ("img", "wrist", "state")
    assert runner.TraceRunner.check_truncation(fake, (obs, False, "unknown"), {"task_index": 7}) == (obs, False, "unknown")
    assert runner.TraceRunner.check_truncation(fake, (obs, False, "unknown"), {"task_index": 8}) == (obs, True, "success")
    assert fake.receipt["truncated_step"] == 2 and runner.TRUNCATED["n1"] == {"truncated_step": 2}
    assert runner.TraceRunner.check_truncation(fake, (obs, True, "fail"), {"task_index": 9}) == (obs, True, "fail")
    first = SimpleNamespace(receipt={"episode_nonce": "n2", "m": 3}, stop_after_first_infer=True, stop_after_step=0)
    assert runner.TraceRunner.check_truncation(first, (obs, False, "unknown"), {"task_index": 6})[1:] == (True, "fail")
    wrote = []
    wrapped = runner.write_record_marking_truncation(lambda p, r: wrote.append(r))
    wrapped("p", {"record_type": "terminal", "episode_nonce": "n2"})
    wrapped("p", {"record_type": "started", "episode_nonce": "n2"})
    assert wrote[0]["truncated_first_infer"] is True and "truncated_first_infer" not in wrote[1]
    runner.TRUNCATED.clear()


def test_load_episode_specs(tmp_path):
    path = tmp_path / "t.json"
    entry = {"seed": 1, "difficulty": "hard", "episode_spec": {"task": "RouteStick"}, "sampling_config": {}}
    path.write_text(json.dumps({"RouteStick:0": entry}))
    assert runner.load_episode_specs(str(path))["RouteStick:0"]["seed"] == 1
    path.write_text(json.dumps({"RouteStick:0": dict(entry, episode_spec={"task": "PatternLock"})}))
    with pytest.raises(runner.official.ContractError):
        runner.load_episode_specs(str(path))
    assert runner.load_episode_specs("") == {}
