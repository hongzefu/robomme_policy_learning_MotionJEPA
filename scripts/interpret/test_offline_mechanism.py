"""固定选点、同输入分组与注意力聚合的CPU短测。"""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from scripts.interpret.analysis import offline_mechanism as M


def summary():
    rows = []
    for model in ("/a/checkpoint", "/b/checkpoint"):
        for episode in (1, 2):
            for step, p in ((1, 0.), (5, .8), (9, .1), (13, .8)):
                rows.append(dict(model_id=model, episode_key=f"StopCube:{episode}", env_step=step,
                    p=p, pre_reach=False, decision=step >= 5, early_visit=step == 1,
                    move_interval=60, query_path=f"/{model}/{episode}/{step}"))
    return dict(rows=rows, calibration_results=[dict(model_id="/b/checkpoint", episode_key=f"StopCube:{e}",
        kind="fixed_release", valid=e == 2, task_success=1) for e in (1, 2)])


def test_selection_uses_valid_success_earliest_ties_and_deduplicates():
    chosen = M.select_cases(summary())
    case = chosen["cases"][0]
    assert case["episode_key"] == "StopCube:2"
    assert case["reference_model"] == "/a/checkpoint"
    assert [p["env_step"] for p in case["points"]] == [5, 9, 1]
    assert case["points"][0]["categories"] == ["决策起点附近", "高比例"]
    assert all(len(p["inputs"]) == 2 for p in case["points"])
    assert len(case["points"]) * 20 <= 240


def test_selection_does_not_substitute_missing_paired_points():
    source = summary()
    source["rows"] = [r for r in source["rows"] if not (r["model_id"] == "/b/checkpoint" and r["env_step"] == 1)]
    chosen = M.select_cases(source)
    assert chosen["missing"][0]["category"] == "较早经过最高比例"
    source["calibration_results"] = []
    assert M.select_cases(source)["cases"] == []


def test_selection_rejects_duplicates_and_non_checkpoint_identity():
    source = summary()
    source["rows"].append(source["rows"][0])
    with pytest.raises(ValueError, match="重复"):
        M.select_cases(source)
    source = summary()
    source["rows"][0]["model_id"] = "ctx"
    with pytest.raises(ValueError, match="绝对"):
        M.select_cases(source)


def test_groups_follow_original_memory_order_and_real_frames():
    obs = SimpleNamespace(static_mask=np.array([[1, 1, 1, 0]], bool),
        motion_mask=np.array([[1, 0]], bool), mem_order=np.array([[4, 0, 1, 5, 2, 3]]),
        tokenized_prompt=np.array([[101, 0]]), tokenized_prompt_mask=np.array([[1, 0]], bool),
        images={"base": None, "wrist": None}, image_masks={"base": np.array([True]), "wrist": np.array([False])})
    layout = dict(P=10, MEM=6, IMG=2, TXT=2)
    groups = M.token_groups(obs, layout, [3, 7], 2)
    assert np.flatnonzero(groups["motion"]).tolist() == [0]
    assert np.flatnonzero(groups["真实帧:3"]).tolist() == [1, 2]
    assert np.flatnonzero(groups["真实帧:7"]).tolist() == [4]
    assert np.flatnonzero(groups["当前视觉"]).tolist() == [6]
    assert np.flatnonzero(groups["其余"]).tolist() == list(range(10, 30))
    partition = np.sum([groups[name] for name in ("motion", "历史视觉", "当前视觉", "文字", "其余")], axis=0)
    assert np.flatnonzero(partition).tolist() == [0, 1, 2, 4, 6, 8] + list(range(10, 30))
    assert partition.max() == 1
    assert np.array_equal(obs.mem_order, [[4, 0, 1, 5, 2, 3]])


def test_missing_prefix_tokens_never_enter_other_denominator():
    obs = SimpleNamespace(static_mask=np.array([[False, False]]), motion_mask=None, mem_order=None,
        tokenized_prompt=np.array([[0]]), tokenized_prompt_mask=np.array([[False]]), images={"腕部": None, "主视角": None},
        image_masks={"腕部": np.array([False]), "主视角": np.array([False])})
    groups = M.token_groups(obs, dict(P=7, MEM=2, IMG=4, TXT=1), [], 2)
    for name in ("motion", "历史视觉", "当前视觉", "文字"):
        assert groups[name].sum() == 0
    assert groups["其余"].sum() == 20
    assert not groups["其余"][:7].any()
    # 即便故意给无效key非零注意力，统计也不能把它们算成有效其余质量。
    values = np.ones((1, 1, 1, 1, 20, 27)) / 27
    stats = M.aggregate_attention(values, groups)
    assert stats["其余"]["tokens"] == 20
    np.testing.assert_allclose(stats["其余"]["mass"], 20 / 27)
    assert not obs.static_mask.any()
    assert not any(mask.any() for mask in obs.image_masks.values())


def test_original_prompt_positions_preserve_ids_and_partition_text_attention():
    obs = SimpleNamespace(static_mask=np.array([[True, False]]), motion_mask=None, mem_order=None,
        tokenized_prompt=np.array([[101, 7, 88, 9001]], dtype=np.int32),
        tokenized_prompt_mask=np.array([[True, False, True, True]]), images={"base": None},
        image_masks={"base": np.array([True])})
    layout = dict(P=8, MEM=2, IMG=2, TXT=4)
    original_ids = obs.tokenized_prompt.copy()
    groups = M.token_groups(obs, layout, [12], 2)
    assert M.original_prompt_positions(obs) == [dict(position=0, token_id=101), dict(position=2, token_id=88),
                                               dict(position=3, token_id=9001)]
    text_groups = {name: mask for name, mask in groups.items() if name.startswith("文字位置:")}
    assert set(text_groups) == {"文字位置:0:词元:101", "文字位置:2:词元:88", "文字位置:3:词元:9001"}
    assert np.flatnonzero(text_groups["文字位置:2:词元:88"]).tolist() == [6]
    np.testing.assert_array_equal(np.sum(list(text_groups.values()), axis=0), groups["文字"])
    weights = np.arange(1, 29, dtype=np.float64)
    values = np.broadcast_to(weights / weights.sum(), (2, 3, 1, 4, 20, 28))
    stats = M.aggregate_attention(values, groups)
    np.testing.assert_allclose(sum(np.asarray(stats[name]["mass"]) for name in text_groups), stats["文字"]["mass"])
    # 修改只读副本中的编号仅改变新子组名字，不影响原五类或帧组的掩码及质量。
    obs.tokenized_prompt = np.array([[11, 22, 33, 44]])
    changed = M.token_groups(obs, layout, [12], 2)
    changed_stats = M.aggregate_attention(values, changed)
    for name in ("motion", "历史视觉", "当前视觉", "文字", "其余", "真实帧:12"):
        np.testing.assert_array_equal(groups[name], changed[name])
        assert stats[name] == changed_stats[name]
    np.testing.assert_array_equal(original_ids, [[101, 7, 88, 9001]])


def test_aggregation_preserves_layer_head_and_mass_partition():
    values = np.ones((2, 3, 1, 4, 20, 30), np.float32) / 30
    groups = {"前": np.arange(30) < 10, "后": np.arange(30) >= 10, "空": np.zeros(30, bool)}
    result = M.aggregate_attention(values, groups)
    assert np.asarray(result["前"]["mass"]).shape == (3, 4)
    np.testing.assert_allclose(np.asarray(result["前"]["mass"]) + result["后"]["mass"], 1, atol=1e-6)
    np.testing.assert_allclose(result["前"]["per_token"], 1 / 30)
    assert result["空"]["per_token"] is None
    values[..., 0] = np.nan
    with pytest.raises(ValueError, match="有限"):
        M.aggregate_attention(values, groups)


def text_probabilities(valid, keys=6):
    """层2×1×头3×文字4×key6；有效行在前 keys 个 key 上按固定权重分配，padding 行置 0（同 attn_adapter）。"""
    weights = np.arange(1, keys + 1, dtype=np.float64)
    values = np.zeros((2, 1, 3, 4, keys))
    values[:, :, :, valid, :] = weights / weights.sum()
    return values


def test_text_attention_shape_slice_and_mass_partition():
    valid = np.array([True, True, True, False])
    values = text_probabilities(valid)
    groups = {"前": np.arange(26) < 2, "中": (np.arange(26) >= 2) & (np.arange(26) < 6),
              "动作": np.arange(26) >= 6, "跨界": np.isin(np.arange(26), [5, 6, 25])}
    result = M.aggregate_text_attention(values, groups, valid)
    assert np.asarray(result["前"]["mass"]).shape == (2, 3)
    np.testing.assert_allclose(result["前"]["mass"], 3 / 21)
    np.testing.assert_allclose(np.asarray(result["前"]["mass"]) + result["中"]["mass"], 1)
    # 只有前 P 位计数和累加：动作 key 不在文字读者的 key 里
    assert result["动作"]["tokens"] == 0 and result["动作"]["per_token"] is None
    np.testing.assert_allclose(result["动作"]["mass"], 0)
    assert result["跨界"]["tokens"] == 1
    np.testing.assert_allclose(result["跨界"]["mass"], 6 / 21)
    np.testing.assert_allclose(result["跨界"]["per_token"], 6 / 21)


def test_text_attention_ignores_padding_rows():
    # 按 mask 选行，不是按「全 0 行」猜：padding 行即使非零也不计，也不进分母
    valid = np.array([True, False, True, False])
    values = text_probabilities(valid)
    values[:, :, :, 2, :] = 0
    values[:, :, :, 2, 0] = 1
    values[:, :, :, 1, 5] = 7
    groups = {"首": np.arange(26) == 0, "前缀": np.arange(26) < 6}
    result = M.aggregate_text_attention(values, groups, valid[None])
    np.testing.assert_allclose(result["首"]["mass"], (1 / 21 + 1) / 2)
    np.testing.assert_allclose(result["前缀"]["mass"], 1)


def test_text_attention_rejects_bad_input():
    valid = np.array([True, True, True, False])
    values = text_probabilities(valid)
    groups = {"前": np.arange(26) < 2}
    with pytest.raises(ValueError, match="key长度"):
        M.aggregate_text_attention(values, {"前": np.arange(6) < 2}, valid)
    with pytest.raises(ValueError, match="行数"):
        M.aggregate_text_attention(values, groups, valid[:3])
    with pytest.raises(ValueError, match="没有有效"):
        M.aggregate_text_attention(values, groups, np.zeros(4, bool))
    with pytest.raises(ValueError, match="有限"):
        M.aggregate_text_attention(np.full_like(values, np.nan), groups, valid)
    with pytest.raises(ValueError, match="有限"):
        M.aggregate_text_attention(values[:, 0], groups, valid)


def test_selection_manifest_cannot_be_changed(tmp_path):
    path = tmp_path / "selection.json"
    M.write_json(path, {"choice": 1}, fixed=True)
    M.write_json(path, {"choice": 1}, fixed=True)
    with pytest.raises(ValueError, match="不能覆盖"):
        M.write_json(path, {"choice": 2}, fixed=True)


def test_select_consumes_s4_fingerprint_and_original_ten_heights(tmp_path):
    source = summary()
    source["rows"] = [r for r in source["rows"] if r["episode_key"] == "StopCube:2"]
    table_path = tmp_path / "table.json"
    table_path.write_text(json.dumps({"episodes": {"StopCube:2": {"z_hover": .16}}}))
    source["table"] = {k: v for k, v in M.fingerprint(table_path).items() if k != "bytes"}
    for i, row in enumerate(source["rows"]):
        query = tmp_path / f"query_{i}.npz"
        np.savez(query, original=np.arange(4))
        cache = tmp_path / f"cache_{i}.npz"
        z = np.full((1, 10, 20), .16)
        z[0, :round(row["p"] * 10), 0] = .1
        np.savez(cache, env_step=[row["env_step"]], query_paths=[str(query)], action_z=z)
        row.update(query_path=str(query), cache_path=str(cache))
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(source))
    selected = M.select(summary_path, tmp_path / "selection.json")
    high = selected["cases"][0]["points"][0]["inputs"][0]
    assert sum(high["press_predictions"]) == 8
    assert len(high["press_predictions"]) == 10


def test_replay_failure_stops_attention_and_preserves_evidence(tmp_path, monkeypatch):
    from scripts.interpret import offline_query
    query = tmp_path / "query.npz"
    cache = tmp_path / "cache.npz"
    np.savez(query, model_id="/a/checkpoint", env_step=9, noise=np.zeros((10, 20, 32)))
    np.savez(cache, dummy=0)
    item = dict(model_id="/a/checkpoint", env_step=9, query_path=str(query),
                query_fingerprint=M.fingerprint(query), cache_fingerprint=M.fingerprint(cache),
                press_predictions=[False] * 10)
    monkeypatch.setattr(offline_query, "replay_query", lambda *a, **k: {"status": "FAIL", "same_path_maxabs": .01})
    result = M.analyze_input(None, None, item, {})
    assert result["replay"]["same_path_maxabs"] == .01
    assert result["predictions"] == []


def test_zero_inputs_write_missing_reports_without_model_loading(tmp_path):
    table_path = tmp_path / "table.json"
    table_path.write_text(json.dumps({"episodes": {}}))
    selection = dict(models=["/a/checkpoint", "/b/checkpoint"], cases=[],
                     missing=[{"reason": "没有有效固定放手成功局"}], selection_rule="固定参考身份",
                     table=M.fingerprint(table_path))
    selection_path = tmp_path / "selection.json"
    M.write_json(selection_path, selection)
    reports = []
    for index, model_id in enumerate(selection["models"]):
        path = tmp_path / f"analysis_{index}.json"
        # 配置名和检查点刻意不可加载；零输入必须在模型加载前返回。
        assert M.analyze(selection_path, model_id, "不存在的配置", path) == 1
        assert json.loads(path.read_text())["results"] == []
        reports.append(path)
    assert M.report(reports, tmp_path / "report") == 1
    text = (tmp_path / "report" / "report.html").read_text()
    assert "分析未执行" in text
    assert "没有有效固定放手成功局" in text
