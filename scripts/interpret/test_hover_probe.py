"""悬停期探针的 CPU 短测：全部用合成的 hover_points.json 和回放 npz，不启动模型、不用 GPU。"""
import json
import re

import numpy as np
import pytest

from scripts.interpret import hover_points as hp
from scripts.interpret.analysis import hover_probe as probe

WIDTH = 16


def make_points(episodes=45):
    """按 hover_points 的公开函数造局：M、N 轮换，到位步固定 48，配对全部算严格。"""
    result = {}
    for k in range(episodes):
        M, N = hp.MOVE_INTERVALS[k % 3], 2 + k % 4
        chosen = hp.select_points(48, M, N)
        chosen_set = set(chosen)
        points = []
        for t in chosen:
            paired = hp.pair_t(t, M, chosen_set)
            points.append({"t": t, **hp.point_labels(t, M, N), "pair_t": paired, "strict": True if paired is not None else None})
        result[str(k)] = {"episode": k, "seed": k, "M": M, "N": N, "T": hp.press_step(M, N), "hover_start": 48,
                          "prompt": "合成", "points": points}
    return result


def write_replay(root, episodes, label, seed=0):
    """信号：历史视觉含 c，motion 含 c，当前视觉含方块位置，文字含 N，最近 motion token 含 M。

    基线的 motion 两列写 NaN：一旦进入特征，拟合立刻报错或得到 NaN，测试就会发现。
    """
    generator = np.random.default_rng(seed)
    folder = root / label
    folder.mkdir(parents=True)
    for key, item in episodes.items():
        n = len(item["points"])
        if not n:
            continue
        hidden = generator.normal(size=(n, 19, 5, WIDTH)).astype(np.float32) * 0.05
        tokens = np.ones((n, 5), dtype=np.int64)
        for index, point in enumerate(item["points"]):
            hidden[index, :, 0, point["c"]] += 1.0
            hidden[index, :, 1, point["c"]] += 1.0
            hidden[index, :, 2, 0] += 5.0 * point["cube_pos"]
            hidden[index, :, 3, item["N"]] += 1.0
            hidden[index, :, 4, hp.MOVE_INTERVALS.index(item["M"])] += 1.0
        if label == "ctx":
            hidden[:, :, 1] = np.nan
            hidden[:, :, 4] = np.nan
            tokens[:, [1, 4]] = 0
        np.savez(folder / f"episode_{key}.npz", t=np.asarray([point["t"] for point in item["points"]]),
                 hidden_prefix=hidden, group_tokens=tokens)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("hover")
    episodes = make_points()
    path = root / "hover_points.json"
    path.write_text(json.dumps({"schema_version": 1, "episodes": episodes}))
    replay = root / "replay"
    write_replay(replay, episodes, "ctx", 1)
    write_replay(replay, episodes, "mot", 2)
    return {"root": root, "points": path, "replay": replay, "samples": probe.load_samples(path)}


def oof(samples, X, label, use_pca=True):
    Y, layout = probe.build_targets(samples)
    prediction, _, leak, _ = probe.crossfit(lambda fold: X, samples, Y, use_pca)
    return probe.decode(prediction, layout[(label, None)]), leak


def accuracy(samples, prediction, label):
    return probe.readout_value(samples, label, prediction, samples["labels"][label])


def test_folds_and_leak(dataset):
    samples = dataset["samples"]
    assert all(len(np.unique(samples["outer"][samples["groups"] == key])) == 1 for key in samples["keys"])
    _, leak = oof(samples, probe.time_features(samples, "t"), "M", use_pca=False)
    assert leak == 0


def test_leak_counts_actual_overlap(dataset):
    samples = dict(dataset["samples"])
    outer = samples["outer"].copy()
    first = samples["groups"] == samples["keys"][0]
    # 人为把第一局拆到两折：泄漏必须按实际下标集合查出来。
    positions = np.flatnonzero(first)
    outer[positions[0]] = (outer[positions[0]] + 1) % probe.OUTER_FOLDS
    samples["outer"] = outer
    Y, _ = probe.build_targets(samples)
    _, _, leak, _ = probe.crossfit(lambda fold: probe.time_features(samples, "t"), samples, Y, False)
    assert leak > 0


def test_one_hot_time_baseline_recovers_label_of_t(dataset):
    samples = dict(dataset["samples"])
    samples["labels"] = dict(samples["labels"])
    samples["labels"]["c"] = (samples["t"] // 32 % 3).astype(np.float64)
    prediction, _ = oof(samples, probe.time_features(samples, "t"), "c", use_pca=False)
    assert accuracy(samples, prediction, "c") > 0.98


def test_time_features_are_one_hot(dataset):
    samples = dataset["samples"]
    X = probe.time_features(samples, "t_N")
    n_t, n_N = len(np.unique(samples["t"])), len(np.unique(samples["N"]))
    assert X.shape == (len(samples["t"]), n_t + n_N)
    assert set(np.unique(X)) == {0.0, 1.0}
    assert (X.sum(axis=1) == 2).all()


def test_speed_signal_beats_baseline_and_noise_does_not(dataset):
    samples = dataset["samples"]
    baseline, _ = oof(samples, probe.time_features(samples, "t"), "M", use_pca=False)
    baseline_value = accuracy(samples, baseline, "M")
    generator = np.random.default_rng(3)
    noise = generator.normal(size=(len(samples["t"]), WIDTH))
    signal = noise * 0.05
    signal[np.arange(len(signal)), np.searchsorted(hp.MOVE_INTERVALS, samples["M"])] += 1.0
    with_signal, _ = oof(samples, signal, "M")
    without, _ = oof(samples, noise, "M")
    assert accuracy(samples, with_signal, "M") > max(0.95, baseline_value + 0.1)
    assert accuracy(samples, without, "M") <= baseline_value + 0.02


def test_pair_requires_both_points_correct():
    truth = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    pairs = np.array([[0, 1], [2, 3], [4, 5]])
    prediction = np.array([1.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    correct, keep = probe.pair_correct(prediction, truth, pairs)
    assert correct.tolist() == [True, False, False]
    assert keep.all()
    prediction[5] = np.nan
    _, keep = probe.pair_correct(prediction, truth, pairs)
    assert keep.tolist() == [True, True, False]


def test_pair_episode_average():
    samples = {"keys": np.array([7, 9]), "group_index": np.array([0, 0, 0, 0, 1, 1]),
               "pairs": np.array([[0, 1], [2, 3], [4, 5]])}
    truth = np.zeros(6)
    prediction = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    values = probe.pair_episode_array(prediction, truth, samples)
    assert values.tolist() == [0.5, 1.0]


def test_shuffle_recomputes_labels_and_keeps_out_of_class_values(dataset):
    samples = dataset["samples"]
    shuffled = probe.shuffled_params(samples, 5)
    assert len(shuffled) == 5
    assert all(len(labels["c"]) == len(samples["t"]) for labels in shuffled)
    assert any(not np.array_equal(labels["M"], samples["labels"]["M"]) for labels in shuffled)
    for labels in shuffled:
        for key in samples["keys"]:
            part = samples["groups"] == key
            assert len(np.unique(labels["M"][part])) == 1
            M = int(labels["M"][part][0])
            first = np.flatnonzero(part)[0]
            # 由 to_press+t=T 反推 N，再逐标签核对是 point_labels 重算的结果。
            N = (labels["to_press"][first] + samples["t"][first] + M // 2) / M
            assert N == int(N)
            for index in np.flatnonzero(part):
                expected = hp.point_labels(int(samples["t"][index]), M, int(N))
                assert all(labels[name][index] == expected[name] for name in expected)
    # 超出原类别的值各算一类，点一个都不丢。
    pool = np.concatenate([labels["remaining"] for labels in shuffled])
    assert (pool <= 0).any()
    assert np.concatenate([labels["c"] for labels in shuffled]).max() > samples["labels"]["c"].max()
    Y, layout = probe.build_targets(samples, shuffled, ("c",))
    assert len(Y) == len(samples["t"])
    for index, labels in enumerate(shuffled):
        start, stop, classes = layout[("c", index)]
        assert set(classes.tolist()) == set(labels["c"].tolist())
        assert (Y[:, start:stop].sum(axis=1) == 1).all()


def test_baseline_empty_columns_not_used(dataset):
    bank = probe.load_bank(dataset["replay"], "ctx", dataset["samples"])
    assert bank["active"] == ["history", "current", "text"]
    assert probe.load_bank(dataset["replay"], "mot", dataset["samples"])["active"] == list(probe.FEATURES)


def test_partial_token_column_rejected(tmp_path):
    episodes = make_points(6)
    path = tmp_path / "hover_points.json"
    path.write_text(json.dumps({"episodes": episodes}))
    write_replay(tmp_path / "replay", episodes, "mot")
    target = tmp_path / "replay" / "mot" / "episode_0.npz"
    with np.load(target) as archive:
        content = dict(archive)
    content["group_tokens"][0, 4] = 0
    np.savez(target, **content)
    with pytest.raises(ValueError, match="部分点"):
        probe.load_bank(tmp_path / "replay", "mot", probe.load_samples(path))


def test_replay_t_mismatch_rejected(tmp_path):
    episodes = make_points(3)
    path = tmp_path / "hover_points.json"
    path.write_text(json.dumps({"episodes": episodes}))
    write_replay(tmp_path / "replay", episodes, "mot")
    target = tmp_path / "replay" / "mot" / "episode_0.npz"
    with np.load(target) as archive:
        content = dict(archive)
    content["t"] = content["t"] + 8
    np.savez(target, **content)
    with pytest.raises(ValueError, match="不符"):
        probe.load_bank(tmp_path / "replay", "mot", probe.load_samples(path))


def test_pca_ignores_test_fold(dataset):
    samples = dataset["samples"]
    generator = np.random.default_rng(5)
    X = generator.normal(size=(len(samples["t"]), WIDTH))
    test = samples["outer"] == 0
    weights = probe.episode_weights(samples["groups"][~test])
    first = probe.fit_pca(X[~test], weights)
    changed = X.copy()
    changed[test] = generator.normal(size=(int(test.sum()), WIDTH)) * 100
    second = probe.fit_pca(changed[~test], weights)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    # 走 crossfit 全程：只改测试折的特征，别的折的折外预测不变。
    Y, _ = probe.build_targets(samples)
    before = probe.crossfit(lambda fold: X, samples, Y, True)[0]
    after = probe.crossfit(lambda fold: changed if fold == 0 else X, samples, Y, True)[0]
    assert np.array_equal(before[~test], after[~test])


def test_pca_scores_not_restandardized():
    """特征尺度只由外层标准化决定：得分的方差是特征值，不会被再拉回 1。"""
    generator = np.random.default_rng(7)
    X = generator.normal(size=(200, 6))
    X[:, 1] = X[:, 0] + 0.01 * generator.normal(size=200)
    weights = np.ones(200)
    scores = probe.pca_transform(probe.fit_pca(X, weights), X)
    variances = scores.var(axis=0)
    assert variances[0] > 1.5 and variances[-1] < 0.1
    Y = (scores[:, 0] > 0).astype(np.float64)[:, None]
    mean, betas, intercept = probe.ridge_path(scores, Y, weights, np.array([1e-9]))
    assert betas.shape == (1, scores.shape[1], 1)
    direct, *_ = np.linalg.lstsq(scores - mean, Y - intercept, rcond=None)
    assert np.allclose(betas[0], direct, atol=1e-4)


def test_ridge_dual_matches_primal():
    generator = np.random.default_rng(8)
    X = generator.normal(size=(20, 40))
    Y = generator.normal(size=(20, 3))
    weights = generator.uniform(0.5, 1.5, size=20)
    lambdas = np.array([0.1, 10.0])
    mean, betas, intercept = probe.ridge_path(X, Y, weights, lambdas)
    Z = X - mean
    for index, value in enumerate(lambdas):
        expected = np.linalg.solve((Z * weights[:, None]).T @ Z + value * np.eye(40), Z.T @ (weights[:, None] * (Y - intercept)))
        assert np.allclose(betas[index], expected, atol=1e-8)


def test_constant_feature_does_not_crash(dataset):
    samples = dataset["samples"]
    prediction, _ = oof(samples, np.ones((len(samples["t"]), WIDTH)), "cube_pos")
    assert np.isfinite(prediction).all()


def test_select_and_compose_layers():
    scores = np.array([[0.1, np.nan, 0.5], [0.3, np.nan, 0.4]])
    assert probe.select_layers(scores) == [1, None, 0]
    by_layer = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    outer = np.array([0, 1, 2])
    composed = probe.compose_prediction(by_layer, [1, None, 0], outer)
    assert composed[0] == 2.0 and np.isnan(composed[1]) and composed[2] == 1.0
    assert probe.most_common_layer([3, 3, 5, 5, None]) == 3
    assert probe.most_common_layer([None]) is None


def test_curve_filter_takes_prefix_by_hash(dataset):
    samples = dataset["samples"]
    train = samples["outer"] != 0
    n_train = len(np.unique(samples["groups"][train]))
    half = probe.curve_filter(samples, 0.5)(train)
    full = probe.curve_filter(samples, 1.0)(train)
    assert np.array_equal(full, train)
    assert len(np.unique(samples["groups"][half])) == int(np.ceil(n_train * 0.5))
    assert not (half & ~train).any()
    assert all(len(np.unique(half[samples["groups"] == key])) == 1 for key in samples["keys"])


def test_plan_times_checked_only_on_full_data(dataset):
    samples = dataset["samples"]
    assert probe.check_plan_times(samples, [], []) is False
    full = {"t": np.zeros(probe.PLAN_POINTS)}
    assert probe.check_plan_times(full, probe.PLAN_COMMON_T, probe.PLAN_SAME_TIME_T) is True
    assert len(probe.PLAN_COMMON_T) == 18 and len(probe.PLAN_SAME_TIME_T) == 15
    with pytest.raises(ValueError):
        probe.check_plan_times(full, probe.PLAN_COMMON_T[1:], probe.PLAN_SAME_TIME_T)


def test_cli_end_to_end(dataset, capsys):
    output = dataset["root"] / "output"
    code = probe.main(["--points", str(dataset["points"]), "--replay-root", str(dataset["replay"]), "--output", str(output),
                       "--workers", "4"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0, "\n".join(lines[-8:])
    assert re.fullmatch(r"HOVER_PROBE_DONE seconds=\d+\.\d", lines[-1])
    text = "\n".join(lines)
    assert "GROUP_LEAK=PASS leak=0" in lines
    for model in ("mot", "ctx"):
        assert re.search(rf"^POSITIVE_CONTROL=PASS r2=[-\d.]+ tol=0\.8 model={model}$", text, re.M)
    pattern = (r"^PROBE=PASS model=mot label={} feature={} layer=\d+ value=[-\d.]+ minus_time=[-\d.]+ "
               r"ci=\[[-\d.]+,[-\d.]+\] shuffled=[-\d.]+$")
    assert re.search(pattern.format("M", "latest_motion"), text, re.M)
    assert re.search(pattern.format("c_strict", "motion"), text, re.M)

    summary = json.loads((output / "hover-summary.json").read_text())
    assert summary["group_leak"] == 0
    assert [line for line in summary["verdicts"] if line.startswith("PROBE=")]
    motion, baseline = summary["models"]["mot"], summary["models"]["ctx"]
    assert set(motion["active_features"]) == set(probe.FEATURES)
    assert set(baseline["active_features"]) == {"history", "current", "text"}
    for feature in motion["features"].values():
        assert len(feature["layers"]) == probe.LAYERS and "input_level" in feature
    # 有信号的读数明显高于打乱对照与时间基线。
    selected = motion["features"]["latest_motion"]["selected"]["M"]
    assert selected["value"] > 0.9 and selected["minus_t"]["mean"] > 0
    assert set(selected["by_M"]) == {"60", "80", "120"} and selected["common_t"]["n_samples"] > 0
    assert motion["features"]["latest_motion"]["shuffle"]["M"]["mean"] < selected["value"]
    curve = motion["features"]["latest_motion"]["learning_curve"]["M"]
    assert [item["fraction"] for item in curve] == list(probe.CURVE_FRACTIONS)
    assert curve[-1]["value"] == pytest.approx(selected["value"])
    cell = motion["features"]["motion"]["layers"][17]
    assert "minus_current" in cell["c"]["full"] and "minus_text" in cell["press"]
    assert set(cell["c"]["by_window_update"]) <= {"0", "8", "16", "24"}
    assert "by_window_update" not in baseline["features"]["history"]["layers"][17]["c"]
    assert summary["times"]["checked_against_plan"] is False
    assert summary["times"]["same_time_t"]
    first = cell["c"]["same_time"][str(summary["times"]["same_time_t"][0])]
    assert set(first["by_M"]) == {"60", "80", "120"} and "time" in first
    for name in ("hover-layerwise.png", "hover-progress.png", "hover-sametime.png"):
        assert (output / name).stat().st_size > 0


def test_cli_smoke_only_main_readouts(dataset, capsys):
    output = dataset["root"] / "smoke"
    code = probe.main(["--points", str(dataset["points"]), "--replay-root", str(dataset["replay"]), "--output", str(output),
                       "--workers", "2", "--smoke"])
    text = capsys.readouterr().out
    assert code == 0
    assert "POSITIVE_CONTROL" not in text and text.count("PROBE=PASS") == 2
    summary = json.loads((output / "hover-summary.json").read_text())
    assert list(summary["models"]) == ["mot"]
    assert set(summary["models"]["mot"]["active_features"]) == set(probe.MAIN_READOUTS)
    assert not list(output.glob("*.png"))


def test_verdict_fails_on_leak_and_bad_control():
    selected = {"value": 0.5, "layer": 3, "minus_t": {"mean": 0.1, "ci95": [0.0, 0.2]}}
    shuffle = {"M": {"mean": 0.3}, "c_strict": {"mean": 0.3}, "press": {"mean": 0.3}}
    features = {name: {"selected": {readout: selected}, "shuffle": {readout: {"mean": 0.3}}}
                for name, (_, readout) in probe.MAIN_READOUTS.items()}
    summary = {"smoke": True, "group_leak": 0, "time_baseline": {"shuffle_t": shuffle},
               "models": {"mot": {"features": features}}}
    lines, passed = probe.verdicts(summary)
    assert passed and lines[0] == "GROUP_LEAK=PASS leak=0"
    assert lines[1] == ("PROBE=PASS model=mot label=M feature=latest_motion layer=3 value=0.5000 minus_time=0.1000 "
                        "ci=[0.0000,0.2000] shuffled=0.3000")
    summary["group_leak"] = 2
    lines, passed = probe.verdicts(summary)
    assert not passed and lines[0] == "GROUP_LEAK=FAIL leak=2" and lines[1].startswith("PROBE=FAIL")
    summary["group_leak"] = 0
    summary["smoke"] = False
    features["current"] = {"layers": [{"cube_pos": {"r2": 0.5}}] * probe.LAYERS}
    lines, passed = probe.verdicts(summary)
    assert not passed and "POSITIVE_CONTROL=FAIL r2=0.5000 tol=0.8 model=mot" in lines
