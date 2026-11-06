"""first_bin_windows 的合成轨迹测试；不依赖 GPU 与 v1-store。"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from scripts.interpret import first_bin_windows as W

BUTTON, VIDEO = "ButtonUnmaskSwap", "VideoUnmaskSwap"
SWAP_END = 10
CHUNK = 8
BIN_XY = [(0.0, 0.0), (0.3, 0.0), (0.6, 0.0)]


def make_rows(n=40, closed=(), lifts=None, hover=None, subgoal=2, opened_at_start=True, exec_start=0):
    """逐步合成轨迹。closed：夹爪闭合的步集合；lifts：{bin: (起, 止)} 这些步之后 z=0.2；hover：{步: bin}；
    exec_start：reset 写入视频的帧数减 1（Button 为 0，Video 为 D）。"""
    lifts, hover, rows, over = lifts or {}, hover or {}, [], 0
    for step in range(n):
        over = hover.get(step, over)
        g = -1.0 if step in closed or not opened_at_start else 1.0
        bins = [[x, y, 0.2 if b in lifts and lifts[b][0] <= step < lifts[b][1] else 0.02]
                for b, (x, y) in enumerate(BIN_XY)]
        rows.append({"event": "step", "episode_key": "x", "infer_seq": step // CHUNK, "row_in_chunk": step % CHUNK,
                     "chunk_start_env_step": step // CHUNK * CHUNK, "state_before": [0.0] * 8,
                     "action_index": step, "exec_start_idx": exec_start, "command_action": [0.0] * 7 + [g],
                     "executed_action": [0.0] * 7 + [g], "state_after": [0.0] * 8,
                     "env_step_before": step, "env_step_after": step + 1,
                     "tcp_before": [BIN_XY[over][0], 0.0, 0.1], "tcp_after": [BIN_XY[over][0], 0.0, 0.1],
                     "bins_before": bins, "bins_after": bins, "subgoal_index": subgoal})
    return rows


def write_episode(run_root, episode_id, rows, task=BUTTON, target=0, outcome="fail", nonce=None, attempt="001", video=None):
    nonce = nonce or f"{task}{episode_id}"
    key = f"{task}:{episode_id}"
    trace_dir = run_root / "attempts" / attempt / "client" / "traces" / nonce
    (trace_dir / "meta").mkdir(parents=True, exist_ok=True)
    (trace_dir / "step_trace.jsonl").write_text(
        json.dumps({"event": "meta"}) + "\n" + "".join(json.dumps(r) + "\n" for r in rows))
    (trace_dir / "meta" / "hold_receipt.json").write_text(json.dumps({
        "difficulty": "hard", "swap_times": 1, "pick_times": 2, "swap_end": SWAP_END, "D": 0, "exec_start_idx": 0,
        "color_names": ["red", "green", "blue"], "prompt": "pick", "first_target_bin_index": target}))
    accepted = run_root / "accepted"
    accepted.mkdir(parents=True, exist_ok=True)
    (accepted / f"{key}.{nonce[:8]}.terminal.json").write_text(json.dumps({
        "task_name": task, "episode_id": episode_id, "episode_key": key, "episode_nonce": nonce, "outcome": outcome}))
    if video:
        write_video(run_root / "attempts" / attempt / "client" / "videos" / f"{task}_ep{episode_id}_{outcome}.mp4", video)


def write_video(path, frames):
    """第 i 帧用 8 条黑白竖条按二进制编码帧号；有损压缩不会让解码差一帧。"""
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    assert writer.isOpened()
    for i in range(frames):
        image = np.zeros((64, 64, 3), np.uint8)
        for bit in range(8):
            if i >> bit & 1:
                image[:, bit * 8:(bit + 1) * 8] = 255
        writer.write(image)
    writer.release()


def decode_frame_number(image):
    """按竖条中线读回 write_video 编码的帧号，与图像尺寸无关。"""
    width = image.shape[1]
    return sum(1 << bit for bit in range(8) if image[:, int((bit + 0.5) * width / 8)].mean() > 127)


def run(tmp_path, capsys, episodes, extra=()):
    """写若干局后跑 main，返回 ({局号: 条目}, 退出码, stdout)。"""
    root = tmp_path / "mot" / "g0" / "run"
    for args, kwargs in episodes:
        write_episode(root, *args, **kwargs)
    output = tmp_path / "out"
    code = W.main(["--run-root", str(root), "--output", str(output), "--expect-episodes", str(len(episodes)), "--no-sheets", *extra])
    data = json.loads((output / "windows.json").read_text())
    return {e["episode_key"]: e for e in data["episodes"]}, code, capsys.readouterr().out


def one(tmp_path, capsys, rows, **kwargs):
    result, code, out = run(tmp_path, capsys, [((0, rows), kwargs)])
    return next(iter(result.values())), code, out


def test_bin_pickup_z_matches_official_source():
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], check=True,
                            capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    source = (Path(common).parent / "third_party/robomme_benchmark/src/robomme/robomme_env/utils"
              / "subgoal_evaluate_func.py").read_text()
    match = re.search(r"def is_bin_pickup\(.*?obj\.pose\.p\[:, 2\] > ([0-9.]+)", source, re.S)
    assert float(match.group(1)) == W.BIN_PICKUP_Z


def test_correct_pick_fields(tmp_path, capsys):
    rows = make_rows(closed=range(27, 40), lifts={0: (35, 40)})
    entry, code, out = one(tmp_path, capsys, rows)
    assert code == 0
    assert entry["label"] == "correct" and entry["picked_bin"] == 0 and entry["first_target_bin"] == 0
    assert entry["C"] == 28 and entry["L"] == 36
    assert entry["G"] == 3 and entry["window"] == [0, 1, 2, 3] and entry["close_row_in_chunk"] == 3
    assert entry["window_status"] == "complete" and entry["nearest_bin_at_C"] == 0
    assert entry["model"] == "mot" and entry["shard"] == "g0"
    assert entry["difficulty"] == "hard" and entry["swap_times"] == 1 and entry["pick_times"] == 2 and entry["bin_count"] == 3
    assert "FIRST_BIN_LABEL=PASS episodes=1 correct=1 wrong=0 no_pick=0" in out
    assert out.strip().splitlines()[-1].startswith("FIRST_BIN_WINDOW=PASS windows=1 complete=1 incomplete=0")


def test_missed_grasp_retry_uses_final_close_and_records_earlier(tmp_path, capsys):
    """抓空重试：第一次闭合在 bin1 上方没抬起，张开后到 bin0 再抓。"""
    rows = make_rows(closed=[*range(14, 18), *range(30, 40)], lifts={0: (36, 40)}, hover={12: 1, 25: 0})
    entry, _, _ = one(tmp_path, capsys, rows)
    assert entry["C"] == 31 and entry["G"] == 3
    assert entry["earlier_closes"] == [{"step": 15, "nearest_bin": 1}]
    assert entry["label"] == "correct"


def test_second_target_does_not_change_label(tmp_path, capsys):
    """双目标：第二次抬起别的 bin 不影响标签，也不改变 L。"""
    rows = make_rows(n=60, closed=[*range(20, 30), *range(45, 60)], lifts={0: (25, 35), 1: (50, 60)}, hover={40: 1})
    entry, code, _ = one(tmp_path, capsys, rows, outcome="success")
    assert entry["label"] == "correct" and entry["L"] == 26 and entry["C"] == 21 and code == 0


def test_no_lift_is_no_pick(tmp_path, capsys):
    entry, code, out = one(tmp_path, capsys, make_rows(closed=range(20, 30)))
    assert entry["label"] == "no_pick" and entry["window"] == [] and entry["G"] is None and entry["picked_bin"] is None
    assert entry["L"] is None and entry["first_target_bin"] == 0 and code == 0
    assert "no_pick=1" in out and "windows=0" in out


def test_swap_lift_is_not_pick(tmp_path, capsys):
    """交换期 bin 抬高不算 L；恰好落在 swap_end 前一行也不算。"""
    rows = make_rows(closed=range(25, 40), lifts={1: (2, SWAP_END - 1), 0: (33, 40)})
    rows[SWAP_END - 2]["bins_after"][1] = [0.3, 0.0, 0.2]
    entry, _, _ = one(tmp_path, capsys, rows)
    assert entry["picked_bin"] == 0 and entry["L"] == 34


def test_lift_at_swap_end_boundary_counts(tmp_path, capsys):
    """env_step_after 恰好等于 swap_end 的行已进入判定范围。"""
    rows = make_rows(closed=range(5, 40), lifts={0: (SWAP_END - 1, 40)})
    entry, _, _ = one(tmp_path, capsys, rows)
    assert entry["L"] == SWAP_END


def test_swap_lift_only_is_no_pick(tmp_path, capsys):
    entry, _, _ = one(tmp_path, capsys, make_rows(lifts={1: (2, SWAP_END - 1)}))
    assert entry["label"] == "no_pick"


def test_button_close_before_swap_end_is_not_c_nor_earlier(tmp_path, capsys):
    rows = make_rows(closed=[*range(3, 7), *range(25, 40)], lifts={0: (33, 40)}, hover={3: 2, 20: 0})
    entry, _, _ = one(tmp_path, capsys, rows)
    assert entry["C"] == 26 and entry["earlier_closes"] == []


def test_button_close_after_swap_end_goes_to_earlier(tmp_path, capsys):
    rows = make_rows(closed=[*range(12, 15), *range(25, 40)], lifts={0: (33, 40)}, hover={12: 2, 20: 0})
    entry, _, _ = one(tmp_path, capsys, rows)
    assert entry["C"] == 26 and entry["earlier_closes"] == [{"step": 13, "nearest_bin": 2}]


def test_lift_before_buttons_is_flagged_and_exempt(tmp_path, capsys):
    """按钮前抬起错 bin 且官方成功：标 before_buttons，不计 wrong_but_success。"""
    rows = make_rows(closed=range(25, 40), lifts={1: (33, 40)}, hover={20: 1}, subgoal=1)
    entry, code, out = one(tmp_path, capsys, rows, outcome="success")
    assert entry["before_buttons"] is True and entry["label"] == "wrong"
    assert code == 0 and "before_buttons=1" in out and "wrong_but_success=0" in out


def test_button_after_buttons_is_false(tmp_path, capsys):
    entry, _, _ = one(tmp_path, capsys, make_rows(closed=range(25, 40), lifts={0: (33, 40)}))
    assert entry["before_buttons"] is False


def test_video_before_buttons_is_null(tmp_path, capsys):
    rows = make_rows(closed=range(25, 40), lifts={0: (33, 40)}, subgoal=0)
    entry, _, out = one(tmp_path, capsys, rows, task=VIDEO)
    assert entry["before_buttons"] is None and "before_buttons=0" in out


def test_terminal_row_counts_as_lift(tmp_path, capsys):
    """判败那一步就是轨迹最后一行时，也要算 L，并且标 wrong。"""
    rows = make_rows(n=36, closed=range(25, 36), lifts={1: (35, 36)}, hover={20: 1})
    entry, code, _ = one(tmp_path, capsys, rows)
    assert entry["label"] == "wrong" and entry["L"] == 36 and entry["picked_bin"] == 1 and code == 0


def test_close_across_chunk_boundary(tmp_path, capsys):
    """上一 chunk 末行张开、下一 chunk 首行闭合。"""
    rows = make_rows(closed=range(24, 40), lifts={0: (33, 40)})
    entry, _, _ = one(tmp_path, capsys, rows)
    assert entry["G"] == 3 and entry["close_row_in_chunk"] == 0 and entry["C"] == 25


def test_wrong_but_success_fails_with_episode(tmp_path, capsys):
    rows = make_rows(closed=range(25, 40), lifts={1: (33, 40)}, hover={20: 1})
    entry, code, out = one(tmp_path, capsys, rows, outcome="success")
    assert entry["before_buttons"] is False and code == 1
    assert "FIRST_BIN_LABEL=FAIL" in out and f"wrong_but_success_episodes={BUTTON}:0" in out


def test_video_wrong_but_success_fails(tmp_path, capsys):
    """Video 的 before_buttons 是 null，不属于按钮前豁免。"""
    rows = make_rows(closed=range(25, 40), lifts={1: (33, 40)}, hover={20: 1}, subgoal=0)
    _, code, out = one(tmp_path, capsys, rows, task=VIDEO, outcome="success")
    assert code == 1 and "wrong_but_success=1" in out


def test_geom_mismatch_reports_without_fail(tmp_path, capsys):
    rows = make_rows(closed=range(25, 40), lifts={1: (33, 40)}, hover={20: 2})
    entry, code, out = one(tmp_path, capsys, rows)
    assert entry["nearest_bin_at_C"] == 2 and entry["picked_bin"] == 1 and code == 0
    assert "geom_mismatch=1" in out and f"GEOM_MISMATCH_EPISODES={BUTTON}:0" in out


def test_short_window_is_incomplete(tmp_path, capsys):
    entry, _, out = one(tmp_path, capsys, make_rows(closed=range(11, 40), lifts={0: (33, 40)}))
    assert entry["G"] == 1 and entry["window"] == [0, 1] and entry["window_status"] == "incomplete"
    assert "windows=1 complete=0 incomplete=1" in out


def test_no_close_before_pick_keeps_episode(tmp_path, capsys):
    rows = make_rows(lifts={0: (33, 40)}, opened_at_start=False)
    entry, code, out = one(tmp_path, capsys, rows)
    assert entry["label"] == "correct" and entry["window_status"] == "no_close_before_pick"
    assert entry["G"] is None and entry["window"] == [] and entry["L"] == 34 and code == 0
    assert "windows=0" in out and "no_close_before_pick=1" in out


def test_close_at_lift_row_is_not_before_pick(tmp_path, capsys):
    rows = make_rows(closed=[*range(14, 18), *range(33, 40)], lifts={0: (33, 40)})
    entry, _, _ = one(tmp_path, capsys, rows)
    assert entry["C"] == 15


def test_missing_infer_in_window_fails(tmp_path, capsys):
    rows = [r for r in make_rows(closed=range(25, 40), lifts={0: (33, 40)}) if not (r["infer_seq"] == 1 and r["row_in_chunk"] == 0)]
    _, code, out = one(tmp_path, capsys, rows)
    assert code == 1 and "FIRST_BIN_WINDOW=FAIL" in out


def test_shards_dedupe_and_count(tmp_path):
    g0, g1 = tmp_path / "mot" / "g0" / "run", tmp_path / "mot" / "g1" / "run"
    write_episode(g0, 0, make_rows())
    write_episode(g1, 1, make_rows())
    write_episode(g1, 0, make_rows())  # 同 nonce 副本，去重
    output = tmp_path / "out"
    args = ["--run-root", str(g0), "--run-root", str(g1), "--output", str(output), "--no-sheets"]
    assert W.main([*args, "--expect-episodes", "2"]) == 0
    entries = json.loads((output / "windows.json").read_text())["episodes"]
    assert [(e["episode_key"], e["shard"]) for e in entries] == [(f"{BUTTON}:0", "g0"), (f"{BUTTON}:1", "g1")]
    with pytest.raises(SystemExit, match="应正好 100 局"):
        W.main(args)


def test_conflicting_nonce_and_error_terminal(tmp_path):
    root = tmp_path / "mot" / "g0" / "run"
    write_episode(root, 0, make_rows(), nonce="aaaaaaaaaa", outcome="error")
    write_episode(root, 0, make_rows(), nonce="bbbbbbbbbb", attempt="002")
    output = tmp_path / "out"
    args = ["--run-root", str(root), "--output", str(output), "--expect-episodes", "1", "--no-sheets"]
    assert W.main(args) == 0  # 基础设施错误局被重试局取代
    write_episode(root, 0, make_rows(), nonce="cccccccccc", attempt="003")
    with pytest.raises(SystemExit, match="不同 nonce"):
        W.main(args)


def test_two_models_rejected(tmp_path):
    with pytest.raises(SystemExit, match="一个模型"):
        W.main(["--run-root", str(tmp_path / "mot/g0/run"), "--run-root", str(tmp_path / "ctx/g1/run"),
                "--output", str(tmp_path / "out"), "--no-sheets"])


def test_read_frames_fails_past_end(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "v.mp4"
    write_video(video, 40)
    with pytest.raises(SystemExit, match="只有 40 帧"):
        W.read_frames(video, [40], cv2)


@pytest.mark.parametrize("task,exec_start", [(BUTTON, 0), (VIDEO, 5)])
def test_tile_uses_reset_offset_frames(tmp_path, capsys, task, exec_start):
    """帧号 = exec_start_idx + action_index + 1；画面编码帧号，错一帧也要被发现。"""
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    rows = make_rows(closed=range(25, 40), lifts={0: (33, 40)}, exec_start=exec_start)
    root = tmp_path / "mot" / "g0" / "run"
    write_episode(root, 0, rows, task=task, video=50)
    output = tmp_path / "out"
    assert W.main(["--run-root", str(root), "--output", str(output), "--expect-episodes", "1", "--no-sheets"]) == 0
    entry = json.loads((output / "windows.json").read_text())["episodes"][0]
    assert entry["C_frame"] == exec_start + 26 and entry["L_frame"] == exec_start + 34
    assert entry["C"] == 26 and entry["L"] == 34
    capsys.readouterr()
    tile = W.episode_tile(entry, W.trace_path(root, entry["episode_nonce"]), cv2, np)
    body = tile[W.CAPTION_HEIGHT:]
    left, right = body[:, :W.TILE_HEIGHT], body[:, W.TILE_HEIGHT:]
    assert decode_frame_number(left) == entry["C_frame"]
    assert decode_frame_number(right) == entry["L_frame"]


def test_no_pick_has_no_frames(tmp_path, capsys):
    entry, _, _ = one(tmp_path, capsys, make_rows())
    assert entry["C_frame"] is None and entry["L_frame"] is None


def test_no_close_has_only_l_frame(tmp_path, capsys):
    entry, _, _ = one(tmp_path, capsys, make_rows(lifts={0: (33, 40)}, opened_at_start=False))
    assert entry["C_frame"] is None and entry["L_frame"] == 34


def test_sheets_written_for_picked_episodes(tmp_path, capsys):
    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "mot" / "g0" / "run"
    for i in range(21):
        write_episode(root, i, make_rows(closed=range(25, 40), lifts={0: (33, 40)}), video=40)
    write_episode(root, 21, make_rows(), video=40)  # no_pick 不进拼图，也不用视频
    write_episode(root, 22, make_rows(lifts={0: (33, 40)}, opened_at_start=False), task=VIDEO, video=40)
    output = tmp_path / "out"
    assert W.main(["--run-root", str(root), "--output", str(output), "--expect-episodes", "23"]) == 0
    assert "FIRST_BIN_SHEETS=DONE sheets=3" in capsys.readouterr().out
    assert sorted(p.name for p in (output / "sheets").iterdir()) == [
        f"{BUTTON}_000.png", f"{BUTTON}_001.png", f"{VIDEO}_000.png"]
    sheet = cv2.imread(str(output / "sheets" / f"{BUTTON}_000.png"))
    # 5 行 4 列，单局拼块 = 说明条 + 帧；合成视频是方形，每局两帧宽 2 * TILE_HEIGHT。
    assert sheet.shape[:2] == (5 * (W.TILE_HEIGHT + W.CAPTION_HEIGHT), 4 * 2 * W.TILE_HEIGHT)


def test_sheet_fails_when_video_missing(tmp_path):
    root = tmp_path / "mot" / "g0" / "run"
    write_episode(root, 0, make_rows(closed=range(25, 40), lifts={0: (33, 40)}))
    with pytest.raises(SystemExit, match="视频应恰有 1 份"):
        W.main(["--run-root", str(root), "--output", str(tmp_path / "out"), "--expect-episodes", "1"])


def test_sheet_fails_when_cv2_unavailable(tmp_path, monkeypatch):
    root = tmp_path / "mot" / "g0" / "run"
    write_episode(root, 0, make_rows(closed=range(25, 40), lifts={0: (33, 40)}))
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(SystemExit, match="拼图需要 cv2"):
        W.main(["--run-root", str(root), "--output", str(tmp_path / "out"), "--expect-episodes", "1"])
