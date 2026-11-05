"""Q2 注意力汇总、P 曲线和 Q3/Q4 探针的 CPU 短测，全部用合成数据，不启动 sim、不用 GPU。"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.interpret.analysis import rollout_attention as ra
from scripts.interpret.analysis import rollout_probe as rp


_state = {"passed": 0, "failed": 0}


class _UnitLine:
    """pytest 汇总行之后再打印判定行，保证它是真正的末行。"""

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            _state["passed"] += 1
        elif report.failed:
            _state["failed"] += 1

    def pytest_unconfigure(self, config):
        verdict = "PASS" if _state["failed"] == 0 and _state["passed"] > 0 else "FAIL"
        print(f"ANALYSIS_UNIT={verdict} tests={_state['passed']}", flush=True)


@pytest.fixture(scope="session", autouse=True)
def _register_unit_line(request):
    request.config.pluginmanager.register(_UnitLine(), "rollout-analysis-unit-line")


Z_HOVER = 0.15
STEP_OFFSET = 7  # chunk_start_env_step − step_idx


def share_matrix(motion=0.2, history=0.3, current=0.2, text=0.1):
    """五组在每个头上的占比，其余补足到 1。"""
    values = {"motion": motion, "历史视觉": history, "当前视觉": current, "文字": text}
    values["其余"] = 1.0 - sum(values.values())
    return {name: np.full((18, 8), value) for name, value in values.items()}


def write_offline(root, model, nonce, key, seq, start, shares, tokens, windows=(), width=8, width_action=6,
                  hidden_prefix=None, hidden_action=None):
    """按 W4 约定写一份离线记录；tokens 给 motion 为 0 表示基线。"""
    names, masses, counts = [], [], []
    for group in ra.FIVE_GROUPS:
        names.append(group)
        masses.append(shares[group])
        counts.append(tokens[group])
    for start_frame, mass, count in windows:
        names.append(f"{ra.WINDOW_PREFIX}{start_frame}")
        masses.append(np.full((18, 8), mass))
        counts.append(count)
    path = ra.offline_path(root, model, nonce, seq)
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = np.zeros((19, 4, width), np.float32) if hidden_prefix is None else hidden_prefix
    action = np.zeros((18, width_action), np.float32) if hidden_action is None else hidden_action
    np.savez(path, group_names=np.asarray(names), group_mass=np.asarray(masses, np.float32),
             group_tokens_all=np.asarray(counts), hidden_prefix=prefix, hidden_action=action,
             hidden_group_tokens=np.asarray([10, tokens["motion"], 8, 5]), p_actions=np.zeros((10, 20, 8), np.float32),
             raw_state=np.zeros(8, np.float32), episode_key=key, infer_seq=seq, step_idx=start - STEP_OFFSET,
             chunk_start_env_step=start, same_path_maxabs=0.0, attn_path_maxabs=0.01,
             frame_times=np.arange(4), motion_times=np.arange(2))
    return path


def align_item(key, nonce, *, seqs=range(6), trigger=None, cross=4, near=None, category="窗口内", uses=None, reach=17,
               center=70.0, target=2, interval=60):
    item = {"episode_key": key, "valid": True, "nonce": nonce, "category": category, "move_interval": interval,
            "target_pass": target, "window_in": 65, "window_out": 75, "window_center": center, "z_hover": Z_HOVER,
            "trigger_step": trigger, "I_cross": cross, "I_trigger": cross if trigger is not None else None, "I_near": near,
            "reach_step": reach,
            "inferences": [{"infer_seq": seq, "chunk_start_env_step": 1 + 16 * seq} for seq in seqs],
            "uses": uses or {"q1": True, "q2_window": True, "q2_other": True, "q3": True, "q4": True}}
    for offset in (1, 2, 3):
        earlier = None if cross is None else cross - offset
        item[f"I_cross_minus_{offset}"] = earlier if earlier in set(seqs) else None
    return item


def base_tokens(motion=20):
    return {"motion": motion, "历史视觉": 30, "当前视觉": 40, "文字": 10, "其余": 20}


def write_table(root, keys):
    entries = {}
    for number, key in enumerate(keys):
        entries[key] = {"move_interval": 60, "target_pass": 2, "reach_step": 17, "z_hover": Z_HOVER, "target_visit_in": 65,
                        "target_visit_out": 75, "env_seed": number, "visits": [{"target_pass": 1, "in": 5, "out": 15},
                                                                               {"target_pass": 2, "in": 65, "out": 75}]}
    path = Path(root) / "table.json"
    path.write_text(json.dumps({"episodes": entries}), encoding="utf-8")
    return path


def write_align(root, models, table):
    path = Path(root) / "analysis" / "align.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "table": {"path": str(table), "sha256": "x"}, "models": models},
                               ensure_ascii=False), encoding="utf-8")


def small_root(tmp_path, n=6):
    """两个模型、n 局，基线没有 motion；每局 I_cross=4。motion 单 token 在 I_cross 比 I_cross−3 高。"""
    keys = [f"StopCube:{index}" for index in range(n)]
    table = write_table(tmp_path, keys)
    models = {"ctx": {}, "mot": {}}
    for index, key in enumerate(keys):
        for model in models:
            models[model][key] = align_item(key, f"{model}-{index}", trigger=70, near=None)
            for seq in range(6):
                motion = 0.0 if model == "ctx" else (0.1 + 0.02 * seq)
                shares = share_matrix(motion=motion)
                tokens = base_tokens(0 if model == "ctx" else 20)
                windows = () if model == "ctx" else ((0, 0.04, 8), (40, 0.06, 12))
                write_offline(tmp_path, model, f"{model}-{index}", key, seq, 1 + 16 * seq, shares, tokens, windows)
    write_align(tmp_path, models, table)
    return tmp_path, keys


# ---------------------------- 注意力 ----------------------------


def test_five_groups_sum_to_one_and_verdict(tmp_path):
    root, _ = small_root(tmp_path)
    summary, line, passed = ra.summarize(root, repeats=50)
    assert passed and line.startswith("ATTN_ALIGN=PASS models=2 ")
    assert summary["models"]["mot"]["sum_err"] < 1e-6
    assert summary["models"]["mot"]["attn_path_maxabs"] == pytest.approx(0.01)
    # 每局 5 个有效对齐点（I_near 不适用），两个模型共 60 个。
    assert "points=60 " in line


def test_sum_error_fails_verdict(tmp_path):
    root, keys = small_root(tmp_path, n=2)
    shares = share_matrix()
    shares["其余"] = shares["其余"] + 0.01
    write_offline(root, "mot", "mot-0", keys[0], 2, 33, shares, base_tokens())
    _, line, passed = ra.summarize(root, repeats=10)
    assert not passed and line.startswith("ATTN_ALIGN=FAIL")


def test_align_mismatch_is_not_used(tmp_path):
    """记录的推理起点与 align.json 不一致：记问题、不用，判定 FAIL。"""
    root, keys = small_root(tmp_path, n=2)
    write_offline(root, "mot", "mot-0", keys[0], 4, 999, share_matrix(motion=0.9, history=0.0, current=0.0, text=0.0), base_tokens())
    summary, line, passed = ra.summarize(root, repeats=10)
    assert not passed and summary["problems"]
    assert summary["models"]["mot"]["all"]["points"]["I_cross"]["n_present"] == 1


def test_missing_points_are_counted_not_substituted(tmp_path):
    """I_cross=1 的局没有 I_cross−2、−3：计入缺失，也不会拿别的推理顶替。"""
    key = "StopCube:0"
    item = align_item(key, "n", cross=1)
    assert item["I_cross_minus_2"] is None and item["I_cross_minus_3"] is None
    write_table(tmp_path, [key])
    for seq in range(6):
        write_offline(tmp_path, "mot", "n", key, seq, 1 + 16 * seq, share_matrix(motion=0.01 * (seq + 1)), base_tokens())
    problems = []
    episodes, _ = ra.load_episodes(tmp_path, "mot", {key: item}, problems)
    assert not problems
    table = ra.five_table(episodes, "I_cross_minus_3", repeats=10)
    assert table["n_present"] == 0 and table["n_missing"] == 1
    assert table["groups"]["motion"]["mean_mass"]["missing"] is True
    table = ra.five_table(episodes, "I_cross", repeats=10)
    assert table["groups"]["motion"]["mean_mass"]["mean"] == pytest.approx(0.02)
    assert ra.self_change(episodes, repeats=10)["episodes"] == []


def test_baseline_motion_is_explicitly_missing(tmp_path):
    root, _ = small_root(tmp_path, n=2)
    summary, _, _ = ra.summarize(root, repeats=10)
    ctx = summary["models"]["ctx"]["all"]
    entry = ctx["points"]["I_cross"]["groups"]["motion"]
    assert entry["mean_mass"]["mean"] is None and entry["mean_mass"]["missing"] is True
    assert ctx["comparisons"]["self_change"]["motion"]["per_token_diff"]["missing"] is True
    assert ctx["comparisons"]["uniform_ratio"]["I_cross"]["ratio"]["missing"] is True
    assert ctx["max_head"]["missing"] is True and ctx["max_head"]["candidates"] is None
    assert summary["models"]["ctx"]["head_matrices"]["I_cross"]["motion"] is None
    # 其余几组照常有值。
    assert ctx["points"]["I_cross"]["groups"]["文字"]["mean_mass"]["mean"] == pytest.approx(0.1)


def test_three_comparisons_by_hand(tmp_path):
    key = "StopCube:0"
    item = align_item(key, "n")
    for seq in range(6):
        # I_cross=4，I_cross−3=1。seq1：motion 0.1/10 token；seq4：motion 0.3/20 token。
        motion, count = {1: (0.1, 10), 4: (0.3, 20)}.get(seq, (0.2, 15))
        tokens = base_tokens(count)
        write_offline(tmp_path, "mot", "n", key, seq, 1 + 16 * seq, share_matrix(motion=motion, history=0.2, current=0.2, text=0.1), tokens)
    episodes, _ = ra.load_episodes(tmp_path, "mot", {key: item}, [])
    change = ra.self_change(episodes, repeats=10)
    assert change["per_token_diff"]["mean"] == pytest.approx(0.3 / 20 - 0.1 / 10)
    assert change["token_count_diff"]["mean"] == 10
    ratio = ra.uniform_ratio(episodes, "I_cross", repeats=10)
    # 总 token 数 20+30+40+10+20=120，均匀占比 1/6。
    assert ratio["token_fraction"]["mean"] == pytest.approx(20 / 120)
    assert ratio["ratio"]["mean"] == pytest.approx(0.3 / (20 / 120))
    others = ra.others_ratio(episodes, "I_cross", repeats=10)
    assert others["当前视觉"]["mean"] == pytest.approx((0.3 / 20) / (0.2 / 40))
    assert others["文字"]["mean"] == pytest.approx((0.3 / 20) / (0.1 / 10))


def test_candidate_heads_table_and_max_head(tmp_path):
    keys = ["StopCube:0", "StopCube:1", "StopCube:2"]
    align = {}
    for number, key in enumerate(keys):
        align[key] = align_item(key, f"n{number}")
        for seq in range(6):
            shares = share_matrix(motion=0.1, history=0.3)
            # 第 0、1 局最大头是 L13H6，第 2 局是 L0H0；从历史视觉挪质量保持和为 1。
            layer, head = (13, 6) if number < 2 else (0, 0)
            shares["motion"][layer, head] = 0.4
            shares["历史视觉"][layer, head] = 0.0
            write_offline(tmp_path, "mot", f"n{number}", key, seq, 1 + 16 * seq, shares, base_tokens())
    episodes, totals = ra.load_episodes(tmp_path, "mot", align, [])
    assert totals["sum_err"] < 1e-6
    table = ra.candidate_table(episodes, repeats=50)
    assert set(table) == {"L13H6", "L15H1", "L16H0"}
    entry = table["L13H6"]["I_cross"]["motion"]
    assert entry["mean"] == pytest.approx((0.4 + 0.4 + 0.1) / 3)
    assert entry["ci95"][0] <= entry["mean"] <= entry["ci95"][1]
    assert table["L15H1"]["I_cross"]["motion"]["mean"] == pytest.approx(0.1)
    assert table["L13H6"]["I_cross"]["历史视觉"]["mean"] == pytest.approx(0.1)
    counts = ra.max_head_counts(episodes)
    assert counts["n_episodes"] == 3 and counts["candidates"] == {"L13H6": 2, "L15H1": 0, "L16H0": 0}
    assert counts["matrix"][0][0] == 1


def test_window_coverage_classification():
    """窗口 [start,start+33) 帧加偏移得环境步；只算该推理之前结束的 visit。"""
    view = {"chunk_start": 100, "offset": STEP_OFFSET, "windows": [
        {"start": 0, "mass": 0.04, "tokens": 4},    # 环境步 [7,40)，覆盖第一次 visit
        {"start": 8, "mass": 0.02, "tokens": 4},    # [15,48)，端点 15 恰好覆盖
        {"start": 9, "mass": 0.03, "tokens": 4},    # [16,49)，已过 visit 1，且不到 visit 2
        {"start": 58, "mass": 0.08, "tokens": 4}]}  # [65,98)，只碰 visit 2
    visits = [{"in": 5, "out": 15}, {"in": 65, "out": 75}, {"in": 125, "out": 135}]
    covered, other = ra.classify_windows(view, visits)
    assert [item["start"] for item in covered] == [0, 8, 58]
    assert [item["start"] for item in other] == [9]
    # 还没结束的 visit 不算。
    covered, _ = ra.classify_windows({**view, "chunk_start": 70}, visits)
    assert [item["start"] for item in covered] == [0, 8]
    assert ra.class_per_token(covered) == pytest.approx(0.06 / 8)
    assert ra.class_per_token([]) is None


def test_window_coverage_summary_and_missing(tmp_path):
    root, _ = small_root(tmp_path, n=3)
    summary, _, _ = ra.summarize(root, repeats=20)
    # I_cross=4，起点 65，offset 7；窗口 0 覆盖 [7,40) 碰 visit 1（out=15<65），窗口 40 覆盖 [47,80) 不碰已结束的。
    coverage = summary["models"]["mot"]["all"]["window_coverage"]["I_cross"]
    assert coverage["covered_per_token"]["mean"] == pytest.approx(0.04 / 8)
    assert coverage["other_per_token"]["mean"] == pytest.approx(0.06 / 12)
    assert coverage["uniform_per_token"]["mean"] == pytest.approx(1 / 120)
    baseline = summary["models"]["ctx"]["all"]["window_coverage"]["I_cross"]
    assert baseline["n_no_window"] == 3 and baseline["covered_per_token"]["missing"] is True


def test_k8_uses_filter(tmp_path):
    """不符局不进窗口相关统计和分组，但仍进 Q2 其余统计；无效局什么都不进。"""
    keys = ["StopCube:0", "StopCube:1", "StopCube:2", "StopCube:3"]
    table = write_table(tmp_path, keys)
    other_only = {"q1": False, "q2_window": False, "q2_other": True, "q3": True, "q4": False}
    align = {keys[0]: align_item(keys[0], "a", trigger=70), keys[1]: align_item(keys[1], "b", trigger=70, uses=other_only, category=None),
             keys[2]: align_item(keys[2], "c", category="未按", near=4),
             keys[3]: {**align_item(keys[3], "d"), "valid": False,
                       "uses": {"q1": False, "q2_window": False, "q2_other": False, "q3": False, "q4": False}}}
    for name, key in zip("abc", keys):
        for seq in range(6):
            write_offline(tmp_path, "mot", name, key, seq, 1 + 16 * seq, share_matrix(), base_tokens(),
                          ((0, 0.05, 10), (40, 0.05, 10)))
            write_offline(tmp_path, "ctx", name, key, seq, 1 + 16 * seq, share_matrix(motion=0.0), base_tokens(0))
    write_align(tmp_path, {"ctx": align, "mot": align}, table)
    summary, _, passed = ra.summarize(tmp_path, repeats=10)
    assert passed
    mot = summary["models"]["mot"]
    assert mot["n_q2_other"] == 3
    assert mot["all"]["points"]["I_cross"]["n_present"] == 3
    assert mot["all"]["window_coverage"]["I_cross"]["n_episodes"] == 2
    assert mot["by_category"]["窗口内"]["n_episodes"] == 1
    assert mot["by_category"]["未按"]["n_episodes"] == 1
    assert mot["by_category"]["未按"]["points"]["I_near"]["n_present"] == 1
    assert mot["by_category"]["偏早"]["n_episodes"] == 0
    assert mot["all"]["points"]["I_near"]["n_present"] == 1


def test_bootstrap_reproducible(tmp_path):
    root, _ = small_root(tmp_path, n=4)
    first, _, _ = ra.summarize(root, repeats=100, seed=42)
    second, _, _ = ra.summarize(root, repeats=100, seed=42)
    assert json.dumps(first, sort_keys=True, ensure_ascii=False) == json.dumps(second, sort_keys=True, ensure_ascii=False)


def test_no_task_success_selection():
    """不按成功挑局：代码里不读 task_success 字段。"""
    tree = ast.parse(Path(ra.__file__).read_text(encoding="utf-8"))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value == "task_success"]


def test_summary_and_figures_written(tmp_path, capsys):
    root, keys = small_root(tmp_path, n=4)
    # 给一个模型写 P 曲线缓存，另一个留空：缺失如实记录。
    for key in keys:
        path = ra.heights_path(root, "mot", key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, chunk_start_env_step=1 + 16 * np.arange(6), p=np.array([np.nan, np.nan, 0.0, 0.2, 0.6, 1.0]))
    code = ra.main(["summarize", "--root", str(root)])
    assert code == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("ATTN_ALIGN=PASS models=2")
    payload = json.loads((root / "analysis" / "attention-summary.json").read_text(encoding="utf-8"))
    assert payload["models"]["ctx"]["p_curves"]["heights_missing"] == keys
    rows = payload["models"]["mot"]["p_curves"]["relative_window_center"]
    assert sum(row["n_episodes"] for row in rows) == 4 * 4  # 到位前两点不计
    for name in ("groups", "candidate-heads", "heads", "p-curves"):
        assert (root / "analysis" / f"attention-{name}.png").stat().st_size > 1000


def test_heatmap_uses_nearest():
    """图③的 imshow 只能用 nearest。"""
    tree = ast.parse(Path(ra.__file__).read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "imshow"]
    assert calls
    for call in calls:
        values = {item.arg: item.value.value for item in call.keywords if item.arg == "interpolation"}
        assert values == {"interpolation": "nearest"}


def test_curve_rows_excludes_pre_reach():
    rows = ra.curve_rows([(np.array([-16, 0, 16]), np.array([np.nan, 0.2, 0.4])),
                          (np.array([-16, 0, 16]), np.array([np.nan, 0.4, 0.8]))])
    assert [row["bin_center"] for row in rows] == [8.0, 24.0]
    assert rows[0]["median"] == pytest.approx(0.3) and rows[1]["q75"] == pytest.approx(0.7)
    assert rows[0]["n_episodes"] == 2


class FakeReader:
    """假运动学：TCP 高度等于关节 0 的值，不启动 sim。"""

    def set_arm(self, q7):
        return np.array([0.0, 0.0, float(q7[0])])


def test_heights_pre_reach_is_nan():
    def record(seq, start, height):
        actions = np.full((10, 20, 8), Z_HOVER, np.float32)
        actions[:, :16, 0] = height
        return {"infer_seq": seq, "chunk_start_env_step": start, "raw_state": np.full(8, Z_HOVER, np.float32), "p_actions": actions}

    records = [record(0, 1, Z_HOVER - 0.05), record(1, 17, Z_HOVER - 0.05), record(2, 33, Z_HOVER)]
    arrays = ra.episode_heights(FakeReader(), records, Z_HOVER, 17)
    assert np.isnan(arrays["p"][0]) and arrays["pre_reach"].tolist() == [True, False, False]
    assert arrays["p"][1] == 1.0 and arrays["p"][2] == 0.0
    assert arrays["action_z"].shape == (3, 10, 20)
    # 只有后 4 行低于阈值时，前 16 行判据不算。
    late = np.full((10, 20, 8), Z_HOVER, np.float32)
    late[:, 16:, 0] = Z_HOVER - 0.05
    arrays = ra.episode_heights(FakeReader(), [{**record(3, 49, 0), "p_actions": late}], Z_HOVER, 17)
    assert arrays["p"][0] == 0.0
    # 未到位的局全部 NaN。
    arrays = ra.episode_heights(FakeReader(), records, Z_HOVER, None)
    assert np.isnan(arrays["p"]).all()


# ---------------------------- 探针 ----------------------------


def synthetic_samples(generator, episodes=40, steps=16, keys=None):
    """每局周期 M、次数 N，锚点 M(N−0.5)，每局从到位后走到锚点附近。"""
    groups, t, n_pass, anchors = [], [], [], []
    keys = keys or [f"StopCube:{index}" for index in range(episodes)]
    for key in keys:
        interval = generator.choice([60, 80, 120])
        target = generator.choice([2, 3, 4])
        anchor = interval * (target - 0.5)
        count = int(generator.integers(steps // 2, steps + 1))
        for position in range(count):
            groups.append(key)
            t.append(49 + 16 * position)
            n_pass.append(target)
            anchors.append(anchor)
    t, n_pass, anchors = (np.asarray(item, dtype=np.float64) for item in (t, n_pass, anchors))
    return {"groups": np.asarray(groups), "t": t, "n_pass": n_pass, "anchor": anchors, "y": anchors - t,
            "rows": np.arange(len(t))}


def embed(generator, signal, width=48, noise=0.05):
    """把 signal (n,k) 随机线性嵌入 width 维并加噪声。"""
    signal = (signal - signal.mean(axis=0)) / signal.std(axis=0)
    return signal @ generator.normal(size=(signal.shape[1], width)) + noise * generator.normal(size=(len(signal), width))


def bank_from(samples, generator, action_signal, width=48, motion=True):
    """合成隐状态库：第 17 层动作 token 由 action_signal 嵌入；其余层和前缀都是噪声。"""
    n = len(samples["y"])
    action = generator.normal(size=(n, 18, width)).astype(np.float32)
    action[:, 17] = embed(generator, action_signal, width).astype(np.float32)
    prefix = generator.normal(size=(n, 19, 4, width)).astype(np.float32)
    if not motion:
        prefix[:, :, rp.MOTION_COLUMN] = 0
    return {"action": action, "prefix": prefix, "motion_present": motion,
            "index": {(key, int(row)): int(row) for key, row in zip(samples["groups"].tolist(), samples["rows"])}}


def test_dual_matches_primal():
    generator = np.random.default_rng(0)
    X = generator.normal(size=(30, 6))
    y = X @ generator.normal(size=6) + generator.normal(size=30)
    weights = generator.uniform(0.5, 1.5, 30)
    lambdas = np.array([0.1, 10.0])
    primal = rp.fit_path(X, y, weights, lambdas)
    # 对偶分支：同一数据加恒零列让维度大于样本数；零列 std 归 1，系数必为 0。
    wide = np.concatenate([X, np.zeros((30, 40))], axis=1)
    dual = rp.fit_path(wide, y, weights, lambdas)
    assert np.allclose(dual[2][:6], primal[2], atol=1e-8)
    assert np.allclose(dual[2][6:], 0.0, atol=1e-8)
    # 与闭式解直接比较。
    mean, std = rp.fit_standardizer(X, weights)
    Z = (X - mean) / std
    centered = y - (weights * y).sum() / weights.sum()
    direct = np.linalg.solve((Z * weights[:, None]).T @ Z + 0.1 * np.eye(6), Z.T @ (weights * centered))
    assert np.allclose(primal[2][:, 0], direct)


def test_decodable_beats_time_baseline():
    """隐状态含周期 M 决定的锚点：能显著优于只用 t、N 的时间基线。"""
    generator = np.random.default_rng(1)
    samples = synthetic_samples(generator)
    bank = bank_from(samples, generator, samples["y"][:, None])
    tally = rp.Tally()
    result, _, _ = rp.main_readouts(tally, bank, samples, repeats=500)
    assert result["main"]["mae"] < result["time_baseline"]["mae"] * 0.5
    assert result["main_minus_time"]["ci95"][1] < 0
    assert result["main"]["r2"] > 0.9 and tally.leak == 0


def test_time_only_hidden_does_not_beat_time_baseline():
    """核心用例：隐状态只含 t、N，R² 可以很高，但不优于时间基线。"""
    generator = np.random.default_rng(2)
    samples = synthetic_samples(generator)
    bank = bank_from(samples, generator, np.stack([samples["t"], samples["n_pass"]], axis=1))
    result, _, _ = rp.main_readouts(rp.Tally(), bank, samples, repeats=500)
    assert result["time_baseline"]["r2"] > 0.5  # 标签确实几乎是时间的函数
    assert result["main_minus_time"]["ci95"][1] > 0  # 区间没有整段落在 0 以下
    assert result["main"]["mae"] >= result["time_baseline"]["mae"] * 0.95


def test_shuffle_near_zero_for_episode_information():
    """隐状态只含整局信息（锚点）：真标签可解码，整局打乱后接近 0。"""
    generator = np.random.default_rng(3)
    samples = synthetic_samples(generator, episodes=60)
    bank = bank_from(samples, generator, samples["anchor"][:, None])
    tally = rp.Tally()
    result, _, _ = rp.main_readouts(tally, bank, samples, repeats=100)
    shuffled = rp.shuffle_controls(tally, bank, samples, shuffles=20)
    assert result["main"]["r2"] > 0.5
    assert shuffled["n"] == 20 and len(shuffled["r2"]) == 20
    assert shuffled["r2_mean"] < 0.05
    assert tally.leak == 0


def test_shuffle_keeps_time_and_reports_time_baseline():
    """隐状态含 t：打乱不动 t，R² 可以大于 0，但与时间基线在同一份打乱标签上几乎一样。"""
    generator = np.random.default_rng(4)
    samples = synthetic_samples(generator, episodes=60)
    bank = bank_from(samples, generator, samples["t"][:, None])
    shuffled = rp.shuffle_controls(rp.Tally(), bank, samples, shuffles=10)
    gaps = np.asarray(shuffled["r2"]) - np.asarray(shuffled["time_baseline_r2"])
    assert np.abs(gaps).max() < 0.1


def test_shuffle_moves_whole_episode():
    generator = np.random.default_rng(5)
    samples = synthetic_samples(generator, episodes=20)
    shuffled = rp.shuffle_labels(samples, generator)
    assert not np.allclose(shuffled, samples["y"])
    for key in np.unique(samples["groups"]):
        mask = samples["groups"] == key
        # 同一局 y′+t 是同一个锚点，且来自别的局的锚点集合。
        anchors = shuffled[mask] + samples["t"][mask]
        assert np.ptp(anchors) == 0 and anchors[0] in set(samples["anchor"].tolist())
    assert sorted({float((shuffled[samples["groups"] == key] + samples["t"][samples["groups"] == key])[0])
                   for key in np.unique(samples["groups"])}) == sorted(
        {float(samples["anchor"][samples["groups"] == key][0]) for key in np.unique(samples["groups"])})


def test_no_episode_across_folds_and_shared_mapping():
    generator = np.random.default_rng(6)
    samples = synthetic_samples(generator)
    fit = rp.ridge_cv(generator.normal(size=(len(samples["y"]), 5)), samples["y"], samples["groups"])
    assert fit["group_leak"] == 0
    folds = {}
    for key in samples["groups"].tolist():
        folds.setdefault(key, set()).add(rp.hash_fold(key, 5))
    assert all(len(value) == 1 for value in folds.values())
    assert len({next(iter(value)) for value in folds.values()}) == 5
    # 映射只由 seed 和局编号决定：与样本集合、其他局无关。
    expected = int(hashlib.sha256(b"42:outer:StopCube:7").hexdigest(), 16) % 5
    assert rp.hash_fold("StopCube:7", 5) == expected


def test_episode_equal_weight_metrics_and_fit():
    groups = np.asarray(["a"] * 100 + ["b"] * 2)
    weights = rp.episode_weights(groups)
    assert weights[groups == "a"].sum() == pytest.approx(weights[groups == "b"].sum())
    # 长局零误差、短局误差 10：每局等权的 MAE 是 5，而不是约 0.2。
    y = np.zeros(102)
    prediction = np.where(groups == "b", 10.0, 0.0)
    assert rp.metrics(y, prediction, groups)["mae"] == pytest.approx(5.0)
    # 特征恒定时，截距应是两局均值的等权平均，而不是被长局拉走。
    target = np.where(groups == "a", 0.0, 1.0)
    path = rp.fit_path(np.ones((102, 1)), target, weights, np.array([1.0]))
    assert path[3] == pytest.approx(0.5)


def test_k9_baseline_concatenates_three_groups():
    prefix = np.zeros((2, 19, 4, 3), np.float32)
    for column in range(4):
        prefix[:, 0, column] = column + 1
    assert rp.k9_layer0(prefix, True).shape == (2, 12)
    baseline = rp.k9_layer0(prefix, False)
    assert baseline.shape == (2, 9)
    assert baseline[0].tolist() == [1, 1, 1, 3, 3, 3, 4, 4, 4]  # motion 列（2）被去掉
    assert rp.active_columns(False) == [0, 2, 3]


def test_q3_only_triggered_and_stops_at_trigger():
    item = align_item("StopCube:0", "n", seqs=range(10), trigger=70, cross=4, reach=17)
    chosen, anchor = rp.select_samples(item, "q3")
    assert anchor == 70
    assert [seq for seq, _ in chosen] == [1, 2, 3, 4]  # 第 0 次在到位前，第 5 次起在 I_trigger 之后
    untriggered = align_item("StopCube:1", "n", seqs=range(10), trigger=None, category="未按")
    assert rp.select_samples(untriggered, "q3") == ([], None)
    excluded = align_item("StopCube:2", "n", trigger=70, uses={"q1": True, "q2_window": True, "q2_other": True, "q3": False, "q4": True})
    assert rp.select_samples(excluded, "q3") == ([], None)
    unreached = align_item("StopCube:3", "n", trigger=70, reach=None)
    assert rp.select_samples(unreached, "q3") == ([], None)


def test_q4_runs_to_end_with_negative_labels():
    item = align_item("StopCube:0", "n", seqs=range(10), trigger=70, center=70.0, reach=17)
    chosen, anchor = rp.select_samples(item, "q4")
    assert anchor == 70.0 and [seq for seq, _ in chosen] == list(range(1, 10))
    labels = [anchor - start for _, start in chosen]
    assert min(labels) < 0
    mismatched = align_item("StopCube:1", "n", uses={"q1": False, "q2_window": False, "q2_other": True, "q3": True, "q4": False})
    assert rp.select_samples(mismatched, "q4") == ([], None)


def test_standardization_uses_training_folds_only():
    """只改测试折里一局的特征：若标准化用了测试折，同折另一局的预测就会变。"""
    generator = np.random.default_rng(7)
    samples = synthetic_samples(generator, episodes=40)
    X = generator.normal(size=(len(samples["y"]), 5))
    in_fold = [key for key in np.unique(samples["groups"]).tolist() if rp.hash_fold(key, 5) == 0]
    assert len(in_fold) >= 2
    changed, neighbor = in_fold[0], in_fold[1]
    first = rp.ridge_cv(X, samples["y"], samples["groups"])["prediction"]
    outlier = X.copy()
    outlier[samples["groups"] == changed, 0] = 1e9
    second = rp.ridge_cv(outlier, samples["y"], samples["groups"])["prediction"]
    mask = samples["groups"] == neighbor
    assert np.array_equal(first[mask], second[mask])
    assert not np.array_equal(first[samples["groups"] == changed], second[samples["groups"] == changed])


def test_no_extra_dependencies():
    for module in (ra, rp):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        assert not names & {"sklearn", "scipy", "pandas", "statsmodels", "torch", "jax"}


def probe_root(tmp_path, generator, episodes=24, width=16):
    """按产物约定写合成离线隐状态。基线没有 motion 列；第 17 层动作 token 含标签信息。"""
    keys = [f"StopCube:{index}" for index in range(episodes)]
    table = write_table(tmp_path, keys)
    models = {"ctx": {}, "mot": {}}
    projection = generator.normal(size=(1, width))
    for number, key in enumerate(keys):
        target, interval = int(generator.choice([2, 3])), int(generator.choice([60, 80, 120]))
        center = interval * (target - 0.5)
        count = 8
        trigger = None if number % 6 == 0 else int(center) + 3
        for model in models:
            nonce = f"{model}-{number}"
            models[model][key] = align_item(key, nonce, seqs=range(count), trigger=trigger, cross=count - 2, reach=17,
                                            center=center, target=target, interval=interval,
                                            category="未按" if trigger is None else "窗口内")
            for seq in range(count):
                start = 1 + 16 * seq
                action = generator.normal(size=(18, width), scale=0.1).astype(np.float32)
                action[17] += (projection * (center - start) / 100).astype(np.float32)[0]
                prefix = generator.normal(size=(19, 4, width)).astype(np.float32)
                tokens = base_tokens(0 if model == "ctx" else 20)
                write_offline(tmp_path, model, nonce, key, seq, start, share_matrix(motion=0.0 if model == "ctx" else 0.2),
                              tokens, width=width, width_action=width, hidden_prefix=prefix, hidden_action=action)
    write_align(tmp_path, models, table)
    return tmp_path


def test_probe_pipeline_end_to_end(tmp_path):
    root = probe_root(tmp_path, np.random.default_rng(8))
    line, passed, _ = rp.run(root, shuffles=2)
    assert passed
    assert line.startswith("PROBE_REPORT=PASS models=2 labels=2 group_leak=0 shuffle_r2_max=")
    payload = json.loads((root / "analysis" / "probe-summary.json").read_text(encoding="utf-8"))
    ctx, mot = payload["models"]["ctx"], payload["models"]["mot"]
    assert ctx["motion_present"] is False and mot["motion_present"] is True
    assert ctx["labels"]["q3"]["layers"]["prefix"]["motion"] is None
    assert len(mot["labels"]["q4"]["layers"]["prefix"]["motion"]) == 18
    assert mot["labels"]["q3"]["n_episodes"] == 20 and mot["labels"]["q4"]["n_episodes"] == 24
    assert mot["labels"]["q4"]["negative_labels"] > 0
    assert payload["common_episodes"]["q3"]["episodes"] and "right_minus_left_main" in payload["common_episodes"]["q3"]
    assert mot["labels"]["q4"]["main"]["mae"] < mot["labels"]["q4"]["time_baseline"]["mae"]
    for name in ("main-table", "layers", "scatter"):
        assert (root / "analysis" / f"probe-{name}.png").stat().st_size > 1000


def test_verdict_fails_on_leak_or_missing_control():
    def entry(mae=1.0, r2=0.0):
        block = {"mae": mae, "r2": r2}
        return {"main": block, "time_baseline": block, "layer0": block, "current_only": block, "shuffle": {"r2_max": r2}}

    models = {name: {"labels": {"q3": entry(), "q4": entry()}} for name in ("ctx", "mot")}
    line, passed, warnings = rp.verdict({"models": models, "group_leak": 0})
    assert passed and not warnings and line.startswith("PROBE_REPORT=PASS")
    assert not rp.verdict({"models": models, "group_leak": 1})[1]
    broken = {name: {"labels": {"q3": entry(mae=None), "q4": entry()}} for name in ("ctx", "mot")}
    assert not rp.verdict({"models": broken, "group_leak": 0})[1]
    # 打乱 R² 超过 0.1 只警告，判据不变。
    loud = {name: {"labels": {"q3": entry(r2=0.3), "q4": entry()}} for name in ("ctx", "mot")}
    line, passed, warnings = rp.verdict({"models": loud, "group_leak": 0})
    assert passed and warnings and "shuffle_r2_max=0.3" in line
