"""人工确认页的 CPU 合成验证，不调用模型或仿真。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.interpret import decision_points as D


KEY = "StopCube:0"


def fixture(tmp_path, *, kind="normal", outcome="fail", press=24, end=56, initial=7):
    video = tmp_path / '原始视频"<.mp4'
    video.write_bytes("合成测试占位，不作为真实视频证据".encode())
    rows = []
    for before in range(initial, end):
        seq = (before - initial) // 16
        below = press is not None and before >= press
        button = press is not None and before >= press + 3
        rows.append({"event": "step", "episode_key": KEY, "trajectory_kind": kind,
                     "infer_seq": seq, "row_in_chunk": (before - initial) % 16,
                     "chunk_start_env_step": initial + seq * 16,
                     "env_step_before": before, "env_step_after": before + 1,
                     "command_action": [0.0] * 8, "executed_action": [0.0] * 8,
                     "state_before": [float(before)] * 8, "state_after": [float(before + 1)] * 8,
                     "tcp_before": [0, 0, 0.1 if below else 0.15],
                     "tcp_after": [0, 0, 0.1 if press is not None and before >= press + 1 else 0.15],
                     "executed_target_tcp_z": 0.1 if below else 0.15,
                     "button_pressed_before": button,
                     "button_pressed_after": button or (press is not None and before == press + 2),
                     "stop_timestep": press + 3 if button else None})
    meta = {"episode_key": KEY, "episode_nonce": kind, "trajectory_kind": kind,
            "outcome": outcome, "task_success": outcome == "success", "initial_env_step": initial,
            "D": 35, "exec_start_idx": 19, "video_fps": 20, "video_start_env_step": 0,
            "video_exec0_frame_index": 35}
    record = {"rows": rows, "meta": meta, "trace_path": str(tmp_path / "step_trace.jsonl"),
              "video_path": str(video), "errors": [],
              "infer_files": [str(tmp_path / f"infer_{seq:04d}.npz") for seq in sorted({r["infer_seq"] for r in rows})]}
    window = {"status": "resolved", "start_range": [36, 58], "move_interval": 80,
              "target_pass": 3, "target_visit_in": 39, "target_visit_out": 61}
    calibration = {"status": "resolved", "hover_z": 0.15, "delta_z": 0.02, "motion_delay": 3,
                   "calibration_source": "same_episode_measured", "trajectory_kind": "normal",
                   "events": {"hover_steps": [20, 23]}}
    return record, window, calibration


def test_no_press_does_not_invent_command(tmp_path):
    record, window, cal = fixture(tmp_path, press=None)
    result = D.analyze(record, window, cal, "normal")
    assert result["events"]["command_press_step"] is None
    assert result["events"]["actual_press_onset"] is None
    assert result["events"]["button_trigger_step"] is None
    assert result["classification"] == "未观察到下压命令"
    assert D.hold_candidate(record, result, window, cal)[0] is None


def test_first_inference_press_has_no_previous_hover(tmp_path):
    record, window, cal = fixture(tmp_path, press=7)
    cal["events"]["hover_steps"] = []
    result = D.analyze(record, window, cal, "normal")
    assert result["events"]["press_infer"] == 0
    assert result["events"]["command_press_step"] == 7
    candidate, reasons = D.hold_candidate(record, result, window, cal)
    assert candidate is None
    assert any("安全悬停" in reason for reason in reasons)


def test_window_all_overlapping_infers_and_truncation(tmp_path):
    record, window, cal = fixture(tmp_path, press=None, end=50)
    result = D.analyze(record, window, cal, "normal")
    assert result["window_infers"] == [1, 2]
    assert result["window_rep"] == 2
    window["start_range"] = [70, 100]
    result = D.analyze(record, window, cal, "normal")
    assert result["window_infers"] == [] and result["window_rep"] is None
    assert any("终局提前截断" in reason for reason in result["reasons"])


def test_command_actual_button_and_terminal_are_distinct(tmp_path):
    record, window, cal = fixture(tmp_path)
    result = D.analyze(record, window, cal, "normal")
    assert result["events"]["command_press_step"] == 24
    assert result["events"]["actual_press_onset"] == 26
    assert result["events"]["button_trigger_step"] == 27
    assert result["events"]["terminal_step"] == 56
    assert result["classification"] == "提前"


def test_hold_uses_measured_state_and_nonzero_boundary_tie_earlier(tmp_path):
    record, window, cal = fixture(tmp_path)
    window["start_range"] = [45, 49]  # 中心47，合法边界39/55同距。
    result = D.analyze(record, window, cal, "normal")
    candidate, reasons = D.hold_candidate(record, result, window, cal)
    assert not reasons
    assert candidate["takeover_step"] == 23
    assert candidate["hold_qpos"] == [23.0] * 8
    assert candidate["release_step"] == 39
    assert candidate["release_offset"] == -8
    assert candidate["D"] == 35  # D与环境步不混用。
    assert (candidate["release_step"] - 7) % 16 == 0


def test_hold_boundary_strictly_after_takeover_and_missing_anchor(tmp_path):
    record, window, cal = fixture(tmp_path)
    window["start_range"] = [22, 24]
    result = D.analyze(record, window, cal, "normal")
    # 用真实分类单独核恢复算法的边界约束。
    result["classification"] = "提前"
    candidate, _ = D.hold_candidate(record, result, window, cal)
    assert candidate["release_step"] == 39 > candidate["takeover_step"]
    del record["meta"]["initial_env_step"]
    for row in record["rows"]:
        row.pop("chunk_start_env_step")
    candidate, reasons = D.hold_candidate(record, result, window, cal)
    assert candidate is None and any("初始化环境步" in reason for reason in reasons)


def test_normal_extended_are_independent_and_no_calibration_leak(tmp_path):
    normal, window, cal = fixture(tmp_path)
    extended, _, _ = fixture(tmp_path, kind="hold_extended", outcome="success", press=44)
    extended["meta"].update(actual_release_step=39, release_step=39, actual_takeover_step=23)
    draft, holds = D.build_review({KEY: window}, {KEY: cal}, {KEY: [normal]}, {KEY: [extended]}, [KEY])
    assert draft["summary"]["normal_success"] == 0
    assert draft["summary"]["extended_success"] == 1
    assert draft["approval"] == "pending"
    episode = draft["episodes"][KEY]
    assert episode["trajectories"]["normal"]["events"]["command_press_step"] == 24
    assert episode["trajectories"]["hold_extended"]["events"]["command_press_step"] == 44
    assert episode["trajectories"]["hold_extended"]["release_offset"] == -8
    assert holds["episodes"][KEY]["release_step"] == 39
    for row in extended["rows"]:
        row.pop("executed_target_tcp_z")
    result = D.analyze(extended, window, {**cal, "events": {"command_press_step": 24}}, "hold_extended")
    assert result["events"].get("command_press_step") is None
    assert any("目标 TCP" in reason for reason in result["reasons"])


def test_all_50_rows_and_ambiguous_duplicates_are_preserved(tmp_path):
    record, window, cal = fixture(tmp_path)
    draft, holds = D.build_review({KEY: window}, {KEY: cal}, {KEY: [record, record]}, {})
    assert len(draft["episodes"]) == 50
    assert not draft["summary"]["ready"]
    assert draft["summary"]["normal"] == 0
    assert "重复轨迹" in draft["episodes"][KEY]["reasons"][0]
    assert KEY not in holds["episodes"]
    page = D.render_html(draft, tmp_path)
    assert page.count('<section id="episode-') == 50
    assert "StopCube:49" in page


def test_discover_trace_meta_and_escaping_links(tmp_path):
    record, window, cal = fixture(tmp_path)
    trace_dir = tmp_path / "run" / "client" / "traces" / "nonce"
    trace_dir.mkdir(parents=True)
    trace = trace_dir / "step_trace.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in record["rows"]), encoding="utf-8")
    meta = {**record["meta"], "trace_path": str(trace), "video_path": record["video_path"]}
    (trace_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    attn = tmp_path / "run" / "attn" / "nonce"
    attn.mkdir(parents=True)
    (attn / "episode.json").write_text(json.dumps(meta), encoding="utf-8")
    for seq in {row["infer_seq"] for row in record["rows"]}:
        (attn / f"infer_{seq:04d}.npz").write_bytes(b"npz")
    found = D.discover(tmp_path / "run", "normal")
    assert len(found[KEY]) == 1
    assert len(found[KEY][0]["infer_files"]) == 4
    cal["calibration_source"] = '<script>alert("侵入")</script>'
    draft, _ = D.build_review({KEY: window}, {KEY: cal}, found, {}, [KEY])
    page = D.render_html(draft, tmp_path)
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "%22%3C.mp4" in page
    assert "#t=" in page and "<video controls" in page
    assert 'href="step_trace.jsonl"' not in page


def test_npz_coverage_is_explicit_and_video_prefix_is_counted(tmp_path):
    record, window, cal = fixture(tmp_path)
    record["infer_files"] = []
    result = D.analyze(record, window, cal, "normal")
    assert any("infer_*.npz" in reason for reason in result["reasons"])
    page = D.media_section(result, tmp_path)
    # 命令环境步24，示范前缀35帧，20帧/秒，事件定位为2.95秒。
    assert "#t=0.950,4.950" in page
    assert "首条下压命令推理序号附近" not in page
    record["infer_files"] = [str(tmp_path / "infer_9999.npz")]
    result = D.analyze(record, window, cal, "normal")
    assert any("序号不一致" in reason for reason in result["reasons"])


def test_cli_writes_only_drafts_and_reports_missing(tmp_path, capsys):
    for name in ("windows", "calibration"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"schema_version": 1, "episodes": {}}))
    normal = tmp_path / "normal"
    normal.mkdir()
    out = tmp_path / "review"
    code = D.main(["--windows", str(tmp_path / "windows.json"), "--calibration", str(tmp_path / "calibration.json"),
                   "--normal", str(normal), "--out", str(out)])
    assert code == 1
    assert "STAGE1_REVIEW_READY=FAIL normal=0 extended=0 approval=pending" in capsys.readouterr().out
    assert {p.name for p in out.iterdir()} == {"decision_points_draft.json", "hold_candidates.json", "review.html"}
    assert len(json.loads((out / "decision_points_draft.json").read_text())["episodes"]) == 50
