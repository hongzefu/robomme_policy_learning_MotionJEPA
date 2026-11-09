"""共同轨迹 Fig. 4 导出入口的 CPU 短测：合成 hover_points.json 与回放 npz，不启动模型、不用 GPU。

覆盖：三张表的列名与行数、层映射（input→-1、Action token 从 0 起）、缺 hidden_action 时没有动作组、
SAME_POINTS 能检出取点差异、ORACLE_EQUAL 两次相等、拟合线在 y=0.5x+3 上恢复斜率截距、REPRO_1105 比较逻辑。
"""
import csv
import json

import numpy as np
import pytest

from scripts.interpret import hover_points as hp
from scripts.interpret.analysis import common_traj_fig4 as fig4
from scripts.interpret.analysis.rollout_probe import metrics

WIDTH = 12
ACTION_WIDTH = 8
EPISODES = 20


def make_points(episodes=EPISODES):
    """按 hover_points 的公开函数造局：M、N 轮换，到位步固定 48；不需要严格配对。"""
    result = {}
    for k in range(episodes):
        M, N = hp.MOVE_INTERVALS[k % 3], 2 + k % 4
        points = [{"t": t, **hp.point_labels(t, M, N), "pair_t": None, "strict": None} for t in hp.select_points(48, M, N)]
        result[str(k)] = {"episode": k, "seed": k, "M": M, "N": N, "T": hp.press_step(M, N), "hover_start": 48,
                          "prompt": "合成", "points": points}
    return result


def write_replay(root, episodes, label, seed, action):
    """文字列含 to_press，当前视觉列含 t；基线的 motion 两列写 NaN 且 token 为 0（一旦进入特征就会出 NaN）。"""
    generator = np.random.default_rng(seed)
    folder = root / label
    folder.mkdir(parents=True)
    for key, item in episodes.items():
        n = len(item["points"])
        if not n:
            continue
        hidden = generator.normal(size=(n, 19, 5, WIDTH)).astype(np.float32) * 0.05
        tokens = np.ones((n, 5), dtype=np.int32)
        to_press = np.asarray([point["to_press"] for point in item["points"]], dtype=np.float32)
        ts = np.asarray([point["t"] for point in item["points"]], dtype=np.int64)
        hidden[:, :, 3, 0] += to_press[:, None] / 50
        hidden[:, :, 2, 1] += ts[:, None] / 50
        if label == "ctx":
            hidden[:, :, 1] = np.nan
            hidden[:, :, 4] = np.nan
            tokens[:, [1, 4]] = 0
        arrays = {"t": ts, "hidden_prefix": hidden, "group_tokens": tokens}
        if action:
            extra = generator.normal(size=(n, 18, ACTION_WIDTH)).astype(np.float32) * 0.05
            extra[:, :, 0] += to_press[:, None] / 50
            arrays["hidden_action"] = extra
        np.savez(folder / f"episode_{key}.npz", **arrays)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("fig4")
    episodes = make_points()
    points = root / "hover_points.json"
    points.write_text(json.dumps({"schema_version": 1, "episodes": episodes}))
    replay = root / "replay"
    write_replay(replay, episodes, "ctx", 1, action=False)
    write_replay(replay, episodes, "mot", 2, action=True)
    output = root / "out"
    lines, passed, _ = fig4.run(points, replay, output, workers=2)
    summary = json.loads((output / "summary.json").read_text())
    return {"root": root, "episodes": episodes, "points": points, "replay": replay, "output": output, "lines": lines,
            "passed": passed, "summary": summary, "samples": fig4.load_samples(points)}


def test_columns_and_rows(dataset):
    header, rows = read_csv(dataset["output"] / "r2_by_layer.csv")
    assert header == ["model", "group", "layer", "layer_index", "r2", "mae_steps"]
    counts = {}
    for row in rows:
        counts[(row[0], row[1])] = counts.get((row[0], row[1]), 0) + 1
    # 基线：没有 Motion（token 为 0）、没有 Action token（库里没有 hidden_action）。
    assert counts == {("baseline", "Text"): 19, ("baseline", "Current frame"): 19, ("baseline", "History frames"): 19,
                      ("motion", "Text"): 19, ("motion", "Current frame"): 19, ("motion", "History frames"): 19,
                      ("motion", "Motion"): 19, ("motion", "Action token"): 18}
    header, rows = read_csv(dataset["output"] / "fit_lines.csv")
    assert header == ["model", "group", "r2", "slope", "slope_ci_low", "slope_ci_high", "intercept", "n_points",
                      "n_episodes"]
    assert [(row[0], row[1]) for row in rows] == [("baseline", "Text"), ("baseline", "Current frame"),
                                                  ("motion", "Text"), ("motion", "Current frame"),
                                                  ("motion", "Action token")]
    n_points = len(dataset["samples"]["t"])
    assert all(int(row[7]) == n_points and int(row[8]) == EPISODES for row in rows)
    header, rows = read_csv(dataset["output"] / "reference_lines.csv")
    assert header == ["model", "reference", "metric", "target", "value", "ci_low", "ci_high", "note"]
    assert [(row[0], row[1], row[2]) for row in rows] == [
        (model, reference, metric) for model in ("baseline", "motion")
        for reference in ("time_only", "time_and_target_count") for metric in ("r2", "mae")]
    assert all(row[3] == "" and row[5] == "" and row[6] == "" and row[7] == "common demo trajectory" for row in rows)


def test_layer_mapping(dataset):
    _, rows = read_csv(dataset["output"] / "r2_by_layer.csv")
    text = [row for row in rows if row[:2] == ["motion", "Text"]]
    assert [(row[2], int(row[3])) for row in text] == [("input", -1)] + [(str(k), k) for k in range(18)]
    action = [row for row in rows if row[:2] == ["motion", "Action token"]]
    assert [(row[2], int(row[3])) for row in action] == [(str(k), k) for k in range(18)]
    # 逐值核对：input 用 hidden_prefix 第 0 项，第 k 层用第 k+1 项，Action token 第 k 层用 hidden_action 第 k 项。
    samples = dataset["samples"]
    prefix, action_hidden = [], []
    for key in samples["keys"].tolist():
        with np.load(dataset["replay"] / "mot" / f"episode_{key}.npz") as archive:
            prefix.append(archive["hidden_prefix"])
            action_hidden.append(archive["hidden_action"])
    prefix, action_hidden = np.concatenate(prefix), np.concatenate(action_hidden)
    truth, groups = samples["labels"]["to_press"], samples["groups"]
    for X, row in ((prefix[:, 0, 3], text[0]), (prefix[:, 6, 3], text[6]), (action_hidden[:, 4], action[4])):
        prediction, _ = fig4.probe(X.astype(np.float64), samples, True)
        assert metrics(truth, prediction, groups)["r2"] == pytest.approx(float(row[4]), abs=1e-12)


def test_no_action_group_without_hidden_action(dataset):
    groups = dataset["summary"]["models"]["ctx"]["groups"]
    assert fig4.ACTION_GROUP not in groups and "Motion" not in groups
    assert fig4.ACTION_GROUP in dataset["summary"]["models"]["mot"]["groups"]


def test_verdicts(dataset):
    lines = dataset["lines"]
    assert dataset["passed"]
    n_points = len(dataset["samples"]["t"])
    assert f"SAME_POINTS=PASS points={n_points} diff=0" in lines
    assert "GROUP_LEAK=PASS leak=0" in lines
    probe_lines = [line for line in lines if line.startswith("PROBE=")]
    # 每个（模型、组）两行：第 17 层与最佳层。
    assert len(probe_lines) == 2 * (3 + 5)
    assert all(line.startswith("PROBE=PASS") for line in probe_lines)
    assert any("model=motion group=Action_token layer=17" in line for line in probe_lines)


def test_oracle_equal(dataset):
    line = next(line for line in dataset["lines"] if line.startswith("ORACLE_EQUAL="))
    assert line.startswith("ORACLE_EQUAL=PASS")
    references = [dataset["summary"]["models"][model]["references"]["time_and_target_count"]["r2"]
                  for model in ("ctx", "mot")]
    assert references[0] == references[1]
    # 两次独立计算也逐位相同。
    samples = dataset["samples"]
    values = []
    for _ in range(2):
        prediction, _ = fig4.probe(fig4.time_features(samples, "t_N"), samples, False)
        values.append(metrics(samples["labels"]["to_press"], prediction, samples["groups"])["r2"])
    assert values[0] == values[1] == references[0]


def test_same_points_detects_difference(dataset, tmp_path):
    keys = dataset["samples"]["keys"].tolist()
    base = fig4.bank_points(dataset["replay"], "mot", keys)
    assert fig4.same_points([base, fig4.bank_points(dataset["replay"], "ctx", keys)]) == (len(base), 0)
    # 改一局的 t：去掉一个点、再多一个点，对称差为 2。
    altered = tmp_path / "replay"
    (altered / "mot").mkdir(parents=True)
    for key in keys:
        with np.load(dataset["replay"] / "mot" / f"episode_{key}.npz") as archive:
            arrays = {name: archive[name] for name in archive.files}
        if key == keys[0]:
            arrays["t"] = arrays["t"].copy()
            arrays["t"][-1] += 1
        np.savez(altered / "mot" / f"episode_{key}.npz", **arrays)
    other = fig4.bank_points(altered, "mot", keys)
    assert fig4.same_points([base, other]) == (len(base) - 1, 2)
    lines, passed = fig4.verdicts(dataset["summary"], len(base) - 1, 2)
    assert "SAME_POINTS=FAIL points={} diff=2".format(len(base) - 1) in lines and not passed
    # 入口本身拒绝与 hover_points 不符的库。
    (altered / "ctx").symlink_to(dataset["replay"] / "ctx")
    with pytest.raises(ValueError):
        fig4.run(dataset["points"], altered, tmp_path / "out", workers=1)


def test_fit_line_recovers_slope_and_intercept():
    generator = np.random.default_rng(0)
    groups = np.repeat(np.arange(12), generator.integers(3, 15, size=12))
    x = generator.uniform(0, 200, size=len(groups))
    result = fig4.fit_line(x, 0.5 * x + 3, groups)
    assert result["slope"] == pytest.approx(0.5, abs=1e-12)
    assert result["intercept"] == pytest.approx(3.0, abs=1e-9)
    assert result["slope_ci"] == pytest.approx([0.5, 0.5], abs=1e-12)
    assert result["n_points"] == len(groups) and result["n_episodes"] == 12
    # 加噪声且局长不同时，点估计等于直接用 episode_weights 的加权最小二乘；区间含点估计且可复现。
    y = 0.5 * x + 3 + generator.normal(scale=20, size=len(x)) + groups * 3.0
    noisy = fig4.fit_line(x, y, groups)
    slope, intercept = fig4.weighted_line_reference(x, y, groups)
    assert noisy["slope"] == pytest.approx(slope, abs=1e-10)
    assert noisy["intercept"] == pytest.approx(intercept, abs=1e-8)
    low, high = noisy["slope_ci"]
    assert low < noisy["slope"] < high
    assert fig4.fit_line(x, y, groups)["slope_ci"] == noisy["slope_ci"]


def test_fit_line_bootstrap_matches_duplicated_episodes():
    """重采样一次的充分统计量写法，与「把抽中的局复制成新局、episode_weights 重新加权」逐式相同。"""
    generator = np.random.default_rng(3)
    groups = np.repeat(np.arange(6), [2, 5, 3, 7, 4, 6])
    x = generator.uniform(0, 100, size=len(groups))
    y = 0.3 * x + generator.normal(scale=5, size=len(x))
    draw = np.random.default_rng(fig4.SEED).choice(6, size=(1, 6), replace=True)[0]
    xs, ys, ids = [], [], []
    for copy, key in enumerate(draw):
        part = groups == key
        xs.append(x[part])
        ys.append(y[part])
        ids.append(np.full(part.sum(), copy))
    expected, _ = fig4.weighted_line_reference(np.concatenate(xs), np.concatenate(ys), np.concatenate(ids))
    first = fig4.fit_line(x, y, groups, repeats=1)
    assert first["slope_ci"][0] == pytest.approx(expected, abs=1e-10)


def test_repro_compare(dataset):
    summary = dataset["summary"]
    reference = {"models": {}, "time_baseline": {}}
    for model, entry in summary["models"].items():
        features = {}
        for name, group in entry["groups"].items():
            if name not in fig4.REPRO_FEATURE:
                continue
            layers = [cell for cell in group["layers"] if cell["layer_index"] >= 0]
            features[fig4.REPRO_FEATURE[name]] = {
                "input_level": {"to_press": {"r2": group["layers"][0]["r2"]}},
                "layers": [{"to_press": {"r2": cell["r2"]}} for cell in layers]}
        reference["models"][model] = {"features": features}
    for name, kind in fig4.REFERENCES.items():
        reference["time_baseline"][kind] = {"to_press": {"r2": summary["models"]["ctx"]["references"][name]["r2"]}}
    result = fig4.compare_1105(summary, reference)
    assert result["pass"] and result["maxabs"] == 0.0 and result["missing"] == 0
    reference["models"]["mot"]["features"]["text"]["layers"][5]["to_press"]["r2"] += 1e-6
    result = fig4.compare_1105(summary, reference)
    assert not result["pass"] and result["maxabs"] == pytest.approx(1e-6, rel=1e-3)
    lines, _ = fig4.verdicts(summary, 1, 0, result)
    assert lines[-1].startswith("REPRO_1105=FAIL")
