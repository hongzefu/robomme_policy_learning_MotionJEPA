"""悬停期选点与标签的 CPU 测试：合成序列、oracle 表核对、真实 H5 第 0 局。"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from scripts.interpret import hover_points as hp

H5_PATH = "/data/hongzefu/data_0226/record_dataset_StopCube.h5"
METADATA_PATH = "/data/hongzefu/data_0226/record_dataset_StopCube_metadata.json"
ORACLE_TABLE = ("/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/evaluation/"
                "stopcube-oracle-1104/oracle/oracle_table.json")


def test_task_params_match_oracle_table():
    with open(ORACLE_TABLE, encoding="utf-8") as handle:
        episodes = json.load(handle)["episodes"]
    assert len(episodes) == 50
    for key, entry in episodes.items():
        assert hp.task_params(entry["env_seed"]) == (entry["move_interval"], entry["target_pass"]), key


def test_press_step_and_crossings():
    assert hp.press_step(120, 4) == 420
    assert hp.press_step(60, 2) == 90
    assert hp.crossings(80) == [40, 120, 200, 280, 360]


@pytest.mark.parametrize("M", [60, 80, 120])
def test_cube_position_landmarks(M):
    assert hp.cube_position(0, M) == pytest.approx(0)
    assert hp.cube_position(M // 2, M) == pytest.approx(0.5)
    assert hp.cube_position(M, M) == pytest.approx(1)
    assert hp.cube_position(3 * M // 2, M) == pytest.approx(0.5)
    assert hp.cube_position(2 * M, M) == pytest.approx(0)


def test_point_labels_at_crossing_boundaries():
    # M=80 的第 1、2 次经过在 40、120，经过当步已计入。
    assert hp.point_labels(39, 80, 3)["c"] == 0
    assert hp.point_labels(40, 80, 3)["c"] == 1
    assert hp.point_labels(119, 80, 3)["c"] == 1
    labels = hp.point_labels(120, 80, 3)
    assert labels["c"] == 2
    assert labels["press"] == 1 and isinstance(labels["press"], int)
    assert labels["remaining"] == 1
    assert labels["to_press"] == hp.press_step(80, 3) - 120
    assert labels["cube_pos"] == pytest.approx(hp.cube_position(120, 80))
    assert hp.point_labels(119, 80, 3)["press"] == 0


def test_point_labels_are_not_clipped_for_shuffled_params():
    # 打乱对照：t 来自 M=120 的局，配上 M=60、N=2，已过次数超过 5 次也不裁剪。
    labels = hp.point_labels(450, 60, 2)
    assert labels["c"] == 8
    assert labels["press"] == 0
    assert labels["remaining"] == -6
    assert labels["to_press"] == 90 - 450


def test_hover_start_finds_earliest_settled_step():
    T = 90
    z = np.full(T - 30, 0.15)
    g = np.full(T - 30, 0.04)
    z[:20] = 0.3
    g[:30] = 0.01
    assert hp.hover_start(z, g, T) == 30
    g[:] = 0.04
    assert hp.hover_start(z, g, T) == 20


def test_hover_start_uses_tolerance_relative_to_last_step():
    T = 90
    z = np.full(T - 30, 0.15)
    g = np.full(T - 30, 0.04)
    z[10] = 0.15 + 0.0009  # 容差内不打断
    z[5] = 0.15 + 0.0011  # 超容差，从下一步起算
    assert hp.hover_start(z, g, T) == 6


def test_hover_start_interruption_after_settling_resets_start():
    T = 90
    z = np.full(T - 30, 0.15)
    g = np.full(T - 30, 0.04)
    g[40] = 0.0
    assert hp.hover_start(z, g, T) == 41


def test_hover_start_ignores_steps_after_end():
    T = 90
    z = np.concatenate([np.full(T - 30, 0.15), np.full(30, 0.05)])
    g = np.full(T, 0.04)
    assert hp.hover_start(z, g, T) == 0


def test_hover_start_returns_end_when_reference_is_invalid():
    T = 90
    z = np.full(T - 30, np.nan)
    g = np.full(T - 30, 0.04)
    assert hp.hover_start(z, g, T) == T - 30


def test_select_points_range_and_stride():
    # M=120、N=4：T=420，取值上界 390，经过点 60、180、300、420、540。
    points = hp.select_points(49, 120, 4)
    assert points[0] == 56
    assert points[-1] == 384
    assert all(t % 8 == 0 for t in points)
    assert hp.select_points(392, 120, 4) == []


def test_select_points_removes_crossing_neighbors_by_exact_margin():
    # M=60 的经过点 30、90、150：真实取值下 8 的倍数只能差 2 或差得更远。
    points = hp.select_points(0, 60, 4)
    assert 32 not in points  # |32 − 30| = 2，去掉
    assert 88 not in points  # |88 − 90| = 2，去掉
    assert 152 not in points  # |152 − 150| = 2，去掉
    assert 80 in points  # |80 − 90| = 10，保留

    # 真实 M 下差不出 3，这里用非真实的 M=134（经过点 67）钉 |差|=3 不去的边界。
    assert 64 in hp.select_points(0, 134, 5)  # |64 − 67| = 3，保留
    # M=130（经过点 65）下 |64 − 65| = 1，去掉，作为同一 t 的反例。
    assert 64 not in hp.select_points(0, 130, 5)


def test_pair_t_requires_partner_in_points():
    points = {100, 340}
    assert hp.pair_t(100, 120, points) == 340
    assert hp.pair_t(340, 120, points) is None
    assert hp.pair_t(100, 60, points) is None


def test_pair_stride_keeps_partners_for_all_speeds():
    # 2M 都是 8 的倍数，所以 t 和 t+2M 同时落在取点网格上；经过点相对位置也相同。
    for M, N in ((60, 5), (80, 5), (120, 5)):
        points = hp.select_points(0, M, N)
        points_set = set(points)
        for t in points:
            paired = hp.pair_t(t, M, points_set)
            if paired is not None:
                labels_t = hp.point_labels(t, M, N)
                labels_pair = hp.point_labels(paired, M, N)
                assert labels_pair["c"] == labels_t["c"] + 2
                assert labels_pair["cube_pos"] == pytest.approx(labels_t["cube_pos"])


def test_max_abs_diff_does_not_wrap_uint8():
    first = np.array([[0, 255]], dtype=np.uint8)
    second = np.array([[8, 0]], dtype=np.uint8)
    assert hp.max_abs_diff(first, second) == 255
    assert hp.max_abs_diff(first, first) == 0


def test_ordinal_matches():
    assert hp.ordinal_matches("just as it reaches the target for the fourth time", 4)
    assert not hp.ordinal_matches("just as it reaches the target for the fourth time", 3)
    assert not hp.ordinal_matches("press the button", 2)
    assert not hp.ordinal_matches("the second and third time", 2)


def synthetic_entry(episode, M, count):
    return {"episode": episode, "M": M, "points": [{}] * count}


def test_figure_episodes_cover_every_speed_in_episode_order():
    episodes = {}
    for k in range(30):
        M = hp.MOVE_INTERVALS[k % 3]
        episodes[str(k)] = synthetic_entry(k, M, 2 if k < 3 else 5)
    chosen = hp.figure_episodes(episodes)
    assert len(chosen) == 10
    assert all(len(entry["points"]) >= 4 for entry in chosen)
    for M in hp.MOVE_INTERVALS:
        assert sum(entry["M"] == M for entry in chosen) >= 3
    assert [entry["episode"] for entry in chosen] == sorted(entry["episode"] for entry in chosen)
    assert not {0, 1, 2} & {entry["episode"] for entry in chosen}


def test_verdict_lines_report_pass_and_fail():
    def point(pair, strict, front, wrist):
        return {"pair_t": pair, "strict": strict, "front_maxabs": front, "wrist_maxabs": wrist}

    episodes = {"0": {"M": 60, "points": [point(1, True, 0, 8)] * 4 + [point(1, False, 9, 0)]},
                "1": {"M": 80, "points": []}}
    points_line, labels_line, pair_line = hp.verdict_lines(episodes, 0)
    assert points_line == "POINTS total=5 episodes_with_points=1 by_M=60:5,80:0,120:0"
    assert labels_line == "LABELS_PRECHECK=PASS ordinal_mismatch=0 episodes=2"
    assert pair_line == "PAIR_FRAMES=PASS pairs=5 kept=4 front_maxabs=9 wrist_maxabs=8 tol=8"
    _, labels_line, pair_line = hp.verdict_lines({"1": {"M": 80, "points": []}}, 1)
    assert labels_line == "LABELS_PRECHECK=FAIL ordinal_mismatch=1 episodes=1"
    assert pair_line.startswith("PAIR_FRAMES=FAIL pairs=0 kept=0")


def test_real_episode_zero_end_to_end(tmp_path, capsys):
    output = tmp_path / "out" / "hover_points.json"
    figures = tmp_path / "figures"
    hp.main(["--h5", H5_PATH, "--metadata", METADATA_PATH, "--output", str(output),
             "--figures", str(figures), "--episodes", "0"])
    printed = capsys.readouterr().out
    assert "POINTS total=42 episodes_with_points=1 by_M=60:0,80:0,120:42" in printed
    assert "LABELS_PRECHECK=PASS ordinal_mismatch=0 episodes=1" in printed

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert list(payload["episodes"]) == ["0"]
    entry = payload["episodes"]["0"]
    assert (entry["episode"], entry["M"], entry["N"], entry["T"]) == (0, 120, 4, 420)
    assert entry["difficulty"] == "easy"
    assert len(entry["points"]) == 42
    with h5py.File(H5_PATH, "r") as h5:
        assert entry["seed"] == int(h5["episode_0/setup/seed"][()])
        assert entry["prompt"] == h5["episode_0/setup/task_goal"][0].decode("utf-8")

    by_t = {point["t"]: point for point in entry["points"]}
    assert entry["hover_start"] <= min(by_t)
    assert max(by_t) < entry["T"] - 30
    for t, point in by_t.items():
        assert set(point) == {"t", "c", "press", "remaining", "to_press", "cube_pos", "pair_t", "strict",
                              "front_maxabs", "wrist_maxabs"}
        assert {key: point[key] for key in ("c", "press", "remaining", "to_press", "cube_pos")} == hp.point_labels(t, 120, 4)
        if point["pair_t"] is None:
            assert point["strict"] is None and point["front_maxabs"] is None and point["wrist_maxabs"] is None
        else:
            assert point["pair_t"] == t + 240 and point["pair_t"] in by_t
            assert point["strict"] == (point["front_maxabs"] <= 8 and point["wrist_maxabs"] <= 8)
    assert not list(output.parent.glob("*.tmp"))
    assert [path.name for path in figures.glob("*.png")] == ["episode_0.png"]
