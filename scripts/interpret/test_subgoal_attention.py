"""subgoal_attention 的 CPU 合成数据测试：对角与均匀读数、两种计权、标签异常、三档边界、样本不足、基线无 motion、入口全流程。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
import subgoal_attention as sg  # noqa: E402


def rows_for(m, chunks, *, key="T:0"):
    """chunks 是每个动作块各行 task_index_before 的列表；after 取下一行 before，局末给 2m+3 由调用方决定。"""
    flat = [v for chunk in chunks for v in chunk]
    rows, index = [], 0
    for seq, chunk in enumerate(chunks):
        for row_in_chunk, before in enumerate(chunk):
            after = flat[index + 1] if index + 1 < len(flat) else before
            rows.append({"event": "step", "episode_key": key, "infer_seq": seq, "row_in_chunk": row_in_chunk,
                         "action_index": index, "chunk_start_env_step": seq * 16, "exec_start_idx": 0,
                         "task_index_before": before, "task_index_after": after,
                         "timestep_before": 0, "timestep_after": 0, "env_step_before": index, "env_step_after": index + 1})
            index += 1
    return rows


def label_episode(m=2, rows=None, demo=None, exec_start=None, outcome="fail"):
    demo = demo if demo is not None else sg.synthetic_demo(m, 3)[0]
    return {"m": m, "demo_task_index": demo, "exec_start_idx": len(demo) - 1 if exec_start is None else exec_start,
            "rows": rows if rows is not None else rows_for(m, [[m + 3, m + 3], [m + 4, m + 4]]),
            "outcome": outcome, "receipt_outcome": outcome}


def analysed(task, model, eid, kind, *, m=4, outcome="fail", difficulty="easy", js=None):
    """不经文件系统造一局分析结果。"""
    demo, exec_start = sg.synthetic_demo(m)
    js = js or list(range(1, m + 1))
    chunks = [[m + 2 + j] * 4 for j in js]
    rows = rows_for(m, chunks, key=f"{task}:{eid}")
    inferences = [sg.synthetic_inference(demo, exec_start, j, kind, seq=seq, with_motion=model == "mot", episode_key=f"{task}:{eid}")
                  for seq, j in enumerate(js)]
    ep = {"episode_key": f"{task}:{eid}", "nonce": f"n{eid}", "task": task, "episode_id": eid, "difficulty": difficulty, "m": m,
          "exec_start_idx": exec_start, "demo_task_index": demo, "outcome": outcome, "success": outcome == "success",
          "receipt_outcome": outcome, "rows": rows, "inferences": inferences}
    return sg.analyse_episode(ep, model)


def test_synthetic_diag_and_uniform():
    result = sg.sanity_check()
    assert result["ok"]
    assert result["sanity_diag"] == pytest.approx(4.0)
    assert result["sanity_uniform"] == pytest.approx(1.0)
    assert result["uniform_sum_err"] < 1e-9
    assert result["results"]["diag/frame/single"][1] == pytest.approx(1.0)
    assert result["results"]["uniform/frame/single"][1] == pytest.approx(0.0, abs=1e-9)
    assert result["results"]["diag/motion/overlap"][0] > 1


def test_overlap_vs_proportional():
    # 段长 10，窗 [0,32) 内 1、2、3 段各 10 帧，第 4 段 2 帧
    demo = [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10 + [7]
    assert sg.segments_of_window(0, demo, 40) == {1: 10, 2: 10, 3: 10, 4: 2}
    # 内容范围截到 exec_start_idx，执行窗不算示范窗
    assert sg.segments_of_window(30, demo, 40) == {4: 10}
    assert sg.segments_of_window(35, demo, 40) == {4: 5}
    assert sg.segments_of_window(40, demo, 40) == {}
    names = ["motion", "历史视觉", "当前视觉", "文字", "其余", "motion窗:0", "motion窗:35"]
    tokens = np.array([16, 0, 32, 16, 20, 8, 8])
    mass = np.zeros((7, 18, 8))
    mass[5] = .3
    mass[6] = .1
    inference = {"path": "x", "names": names, "tokens": tokens, "mass_action": mass, "mass_text": mass}
    overlap = sg.stream_shares(inference, demo, 40, 4, "action", "motion", "overlap")
    prop = sg.stream_shares(inference, demo, 40, 4, "action", "motion", "proportional")
    # 有重叠就算：窗 0 整窗计入第 1～4 段，share 不归一化，和大于 1
    assert overlap["share"] == pytest.approx([.75, .75, .75, 1.0])
    assert overlap["uniform"] == pytest.approx([.5, .5, .5, 1.0])
    assert overlap["share"].sum() > 1
    # 比例计权：窗 0 按 10/32、10/32、10/32、2/32 分，各段和为 1
    assert prop["share"] == pytest.approx([.75 * 10 / 32, .75 * 10 / 32, .75 * 10 / 32, .75 * 2 / 32 + .25])
    assert prop["share"].sum() == pytest.approx(1.0)
    assert prop["uniform"].sum() == pytest.approx(1.0)


def test_frame_stream_sums_to_one_and_uniform_by_tokens():
    demo = [1] * 5 + [2] * 5 + [5]
    names = ["motion", "历史视觉", "当前视觉", "文字", "其余", "真实帧:0", "真实帧:1", "真实帧:6", "真实帧:10"]
    tokens = np.array([0, 64, 32, 16, 20, 16, 16, 16, 16])
    mass = np.zeros((9, 18, 8))
    mass[5:9] = np.array([.1, .1, .6, .2])[:, None, None]
    inference = {"path": "x", "names": names, "tokens": tokens, "mass_action": mass, "mass_text": mass}
    shares = sg.stream_shares(inference, demo, 10, 2, "action", "frame")
    # 执行帧 t=10 不算；uniform 按这次实际进入记忆的 token 数算：段 1 两帧、段 2 一帧
    assert shares["share"] == pytest.approx([.25, .75])
    assert shares["uniform"] == pytest.approx([2 / 3, 1 / 3])
    assert sg.segment_of_frame(10, demo, 10) is None
    assert sg.segment_of_frame(9, demo, 10) == 2
    assert shares["share_lh"].shape == (18, 8, 2)


def test_m1_excluded_from_a2():
    eps = [analysed("RouteStick", "ctx", 1, "uniform", m=1), analysed("RouteStick", "ctx", 2, "diag", m=3)]
    cell = sg.a2_cell(eps, "action", "frame", "single", 200, 42)
    assert cell["n_episodes"] == 1
    assert cell["excluded_m1"] == ["RouteStick:1"]
    # A1 仍含 m=1 的局
    a1 = sg.a1_block(eps, 50, 42)
    assert a1["RouteStick"]["ctx"]["action"]["all"]["all"]["1"]["n_episodes"] == 2


def test_terminal_task_index_legal_and_label_errors():
    m = 2
    good = rows_for(m, [[m + 3, m + 3], [m + 4, m + 4]])
    good[-1]["task_index_after"] = 2 * m + 3
    assert sg.check_labels(label_episode(m, good)) == []
    # 局末以外出现 2m+3 不合法
    bad = rows_for(m, [[m + 3, m + 3], [m + 4, m + 4]])
    bad[0]["task_index_after"] = 2 * m + 3
    assert sg.check_labels(label_episode(m, bad))
    # before 回退、超范围
    assert sg.check_labels(label_episode(m, rows_for(m, [[m + 4], [m + 3]])))
    assert sg.check_labels(label_episode(m, rows_for(m, [[m + 2]])))
    # 示范序号不是 1..m、末帧不是 m+3、切换次数不符
    assert sg.check_labels(label_episode(m, demo=[1, 1, 3, 3, 5]))
    assert sg.check_labels(label_episode(m, demo=[1, 1, 2, 2, 2]))
    assert sg.check_labels(label_episode(m, demo=[1, 2, 1, 2, 5]))
    # 长度与 exec_start_idx 对不上
    assert sg.check_labels(label_episode(m, exec_start=2))
    # 缺第 0 行
    missing = rows_for(m, [[m + 3, m + 3]])
    missing[0]["row_in_chunk"] = 5
    assert sg.check_labels(label_episode(m, missing))


def test_crosses_subgoal():
    m = 3
    rows = rows_for(m, [[m + 3, m + 3], [m + 3, m + 4], [m + 4, m + 5]])
    labels = sg.label_inferences(rows, m)
    assert [labels[i]["j"] for i in range(3)] == [1, 1, 2]
    assert [labels[i]["crosses_subgoal"] for i in range(3)] == [0, 1, 1]
    # 局末 after=2m+3 不算跨越
    rows = rows_for(m, [[2 * m + 2, 2 * m + 2]])
    rows[-1]["task_index_after"] = 2 * m + 3
    assert sg.label_inferences(rows, m)[0]["crosses_subgoal"] == 0


def test_a1_crosses_column_and_text_reader_action_zero():
    m = 2
    demo, exec_start = sg.synthetic_demo(m)
    rows = rows_for(m, [[m + 3, m + 4], [m + 4, m + 4]], key="RouteStick:1")
    infs = [sg.synthetic_inference(demo, exec_start, j, "uniform", seq=s) for s, j in enumerate((1, 2))]
    ep = sg.analyse_episode({"episode_key": "RouteStick:1", "nonce": "n", "task": "RouteStick", "episode_id": 1, "difficulty": "easy", "m": m,
                             "exec_start_idx": exec_start, "demo_task_index": demo, "outcome": "fail", "success": False,
                             "receipt_outcome": "fail", "rows": rows, "inferences": infs}, "mot")
    a1 = sg.a1_block([ep], 50, 42)["RouteStick"]["mot"]
    act = a1["action"]["all"]["all"]["1"]
    assert act["n_crosses_subgoal"] == 1
    assert "crosses_subgoal" in act["blocks"]["示范帧"]
    assert "crosses_subgoal" not in a1["action"]["all"]["all"]["2"]["blocks"]["示范帧"]
    text = a1["text"]["all"]["all"]["1"]["blocks"]["动作 token"]
    assert text["share"] == 0 and text["uniform"] == 0 and text["enrichment"] is None
    # 按 token 均匀的合成注意力：占比等于均匀水平，七块之和为 1
    blocks = act["blocks"]
    assert sum(b["share"] for b in blocks.values()) == pytest.approx(1.0)
    assert blocks["示范帧"]["share"] == pytest.approx(blocks["示范帧"]["uniform"])


@pytest.mark.parametrize("d_ci, slope_ci, tier", [
    ([1.2, 2.0], None, "明显"),               # 恰等于门槛算明显
    ([1.19, 2.0], [0.2, 0.3], "明显"),
    ([1.19, 2.0], None, "弱"),
    (None, [0.19, 0.3], "弱"),
    ([1.0, 1.5], [0.0, 0.1], "未见"),         # 恰等于零假设不算高于
    ([0.5, 0.9], [-0.3, -0.1], "未见"),       # 上界低于零假设也是未见
    (None, None, "未见"),
])
def test_tier_boundaries(d_ci, slope_ci, tier):
    assert sg.judge_tier(d_ci, slope_ci) == tier


def test_a3_insufficient_and_common_j():
    success = [analysed("RouteStick", "mot", i, "diag", outcome="success") for i in range(10)]
    fail = [analysed("RouteStick", "mot", 100 + i, "uniform", js=[1, 2]) for i in range(10)]
    cell = sg.a3_cell(success, fail, "action", "frame", 200, 42)
    assert cell["common_j"] == [1, 2]
    assert cell["D"]["status"] == "可比" and cell["D"]["mean"] > 0
    assert cell["slope"]["n_success"] == 10
    few = sg.a3_cell(success, fail[:9], "action", "frame", 200, 42)
    assert few["D"]["status"] == "样本不足" and few["D"]["mean"] is None and few["D"]["n_fail"] == 9


def test_bootstrap_diff_matches_first_bin():
    import first_bin_attention as fb
    a, b = np.arange(12.) / 7, np.arange(9.) ** 2 / 13
    assert sg.bootstrap_diff(a, b) == fb.bootstrap_diff(a, b)


def test_slope_null_centered_and_diag_clear():
    eps = [analysed("RouteStick", "ctx", i, "diag") for i in range(6)]
    pairs = [(ep["records"]["action", "frame", "single"], ep["m"]) for ep in eps]
    null = sg.slope_null(pairs, 500, 42)
    assert abs(null["mean"]) < .15
    assert sg.centroid_slope(*pairs[0]) == pytest.approx(1.0)
    # 只走过一个子目标的局斜率不可算，但 D 可算
    one = analysed("RouteStick", "ctx", 9, "diag", js=[1, 1, 1])["records"]["action", "frame", "single"]
    assert sg.centroid_slope(one, 4) is None
    assert sg.diag_enrichment(one) == pytest.approx(4.0)


def test_summary_tiers_baseline_without_motion_and_common_pattern():
    eps = []
    for task in sg.TASKS:
        for model in sg.MODELS:
            for i in range(5):
                eps.append(analysed(task, model, i, "diag" if model == "mot" else "uniform"))
    summary = sg.summarize(eps, 200, 42)
    main = summary["A2_main_readouts"]
    assert len(main) == 12 and len(summary["A2_review_proportional"]) == 4
    tier = {(r["task"], r["model"], r["reader"], r["stream"]): r["tier"] for r in main}
    assert tier["RouteStick", "mot", "action", "frame"] == "明显"
    assert tier["RouteStick", "ctx", "action", "frame"] == "未见"
    assert not any(r["stream"] == "motion" for r in main if r["model"] == "ctx")
    common = {(c["model"], c["reader"], c["stream"]): c["common_pattern"] for c in summary["common_patterns"]}
    assert common["mot", "action", "frame"] is True and common["ctx", "action", "frame"] is False
    # 基线的 A1 motion 块为 0，不报错；A4 为 18×8
    block = summary["A1"]["RouteStick"]["ctx"]["action"]["all"]["all"]["1"]["blocks"]["示范 motion"]
    assert block["share"] == 0 and block["enrichment"] is None
    a4 = summary["A4"]["cells"]["mot"]["action"]["frame"]["single"]["RouteStick"]
    assert np.array(a4["excess"]).shape == (18, 8)
    json.dumps(summary, allow_nan=False)
    # 对角只落在第 j 段，对角超额为 1 − 1/4
    assert np.array(a4["excess"]) == pytest.approx(.75)


def test_a4_keeps_per_head_differences():
    ep = analysed("RouteStick", "ctx", 0, "uniform")
    # 让 L3H2 的示范帧质量只落在对角段
    demo, exec_start = sg.synthetic_demo(4)
    infs = []
    for seq, j in enumerate(range(1, 5)):
        inf = sg.synthetic_inference(demo, exec_start, j, "uniform", seq=seq, with_motion=False)
        diag = sg.synthetic_inference(demo, exec_start, j, "diag", seq=seq, with_motion=False)
        inf["mass_action"][:, 3, 2] = diag["mass_action"][:, 3, 2]
        infs.append(inf)
    rows = rows_for(4, [[6 + j] * 2 for j in range(1, 5)], key="RouteStick:0")
    ep = sg.analyse_episode({**{k: ep[k] for k in ("episode_key", "nonce", "task", "episode_id", "difficulty", "m", "exec_start_idx", "outcome", "success")},
                             "demo_task_index": demo, "receipt_outcome": "fail", "rows": rows, "inferences": infs}, "ctx")
    cell = sg.a4_cell([ep], "action", "frame", "single")
    excess = np.array(cell["excess"])
    assert excess[3, 2] == pytest.approx(.75)
    assert excess[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert cell["top_heads"][0]["head"] == "L3H2"


# ---------------------------------------------------------------- 入口全流程（落盘假产物）

def write_episode(root, model, shard, task, eid, kind, *, m=3, outcome="success", offline=True, attempts=1, label_bad=False):
    key = f"{task}:{eid}"
    nonce = f"{model}{task}{eid:04d}" + "a" * 16
    run = root / model / shard / "run"
    (run / "accepted").mkdir(parents=True, exist_ok=True)
    demo, exec_start = sg.synthetic_demo(m)
    for i in range(attempts - 1):
        lost = run / "attempts" / f"lost-{key.replace(':', '_')}-{i}" / "client"
        lost.mkdir(parents=True)
        (lost / f"{key}.started.json").write_text(json.dumps({"episode_key": key}))
    client = run / "attempts" / f"000-{key.replace(':', '_')}" / "client"
    trace_dir = client / "traces" / nonce
    (trace_dir / "meta").mkdir(parents=True)
    (client / f"{key}.started.json").write_text(json.dumps({"episode_key": key}))
    (client.parent / "commands.json").write_text(json.dumps({"episode_keys": [key]}))
    rows = rows_for(m, [[m + 2 + j] * 4 for j in range(1, m + 1)], key=key)
    rows[-1]["task_index_after"] = 2 * m + 3
    if label_bad:
        rows[0]["task_index_before"] = 1
    with (trace_dir / "step_trace.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    receipt = {"episode_key": key, "task_name": task, "episode_id": eid, "difficulty": "easy", "m": m, "exec_start_idx": exec_start,
               "D": exec_start, "demo_task_index": demo, "outcome": outcome, "task_success": int(outcome == "success"), "actions_executed": len(rows)}
    (trace_dir / "meta" / "hold_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False))
    (run / "accepted" / f"{key}.{nonce[:8]}.terminal.json").write_text(
        json.dumps({"episode_key": key, "episode_nonce": nonce, "outcome": outcome, "task_name": task, "episode_id": eid}))
    if not offline:
        return
    directory = root / model / "offline" / nonce
    directory.mkdir(parents=True)
    for seq, j in enumerate(range(1, m + 1)):
        inf = sg.synthetic_inference(demo, exec_start, j, kind, seq=seq, with_motion=model == "mot", episode_key=key)
        np.savez(directory / f"offline_{seq:04d}.npz", group_names=np.asarray(inf["names"]),
                 group_mass_action=inf["mass_action"].astype(np.float32), group_mass_text=inf["mass_text"].astype(np.float32),
                 group_tokens_all=inf["tokens"], episode_key=np.str_(key), infer_seq=np.int64(seq), step_idx=np.int64(seq * 16),
                 chunk_start_env_step=np.int64(seq * 16), same_path_maxabs=np.float64(0), attn_path_maxabs=np.float64(0),
                 input_sha256=np.str_("x"), model_id=np.str_(model), frame_times=np.arange(3), motion_times=np.arange(3))


def build_tree(root, shards=("s0", "s1", "s2", "s3"), episodes=4):
    for model in sg.MODELS:
        for task in sg.TASKS:
            for eid in range(episodes):
                shard = shards[eid % len(shards)]
                write_episode(root, model, shard, task, eid, "diag" if model == "mot" else "uniform",
                              outcome="success" if eid % 2 else "fail")
    return episodes * len(sg.TASKS)


def test_main_end_to_end(tmp_path, capsys):
    total = build_tree(tmp_path / "root")
    code = sg.main(["--root", str(tmp_path / "root"), "--out", str(tmp_path / "out"), "--expected-episodes", str(total), "--repeats", "100"])
    text = capsys.readouterr().out
    assert code == 0, text
    assert f"ROLLOUT_RUN=PASS models=2 valid_ctx={total} valid_mot={total} invalid_ctx=0 invalid_mot=0 infra_retries=0" in text
    assert f"SUBGOAL_LABEL=PASS models=2 episodes={2 * total} mismatches=0" in text
    assert "SUBGOAL_ALIGN=PASS models=2 tasks=2 sanity_diag=4 sanity_uniform=1" in text
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert len(summary["A2_main_readouts"]) == 12
    assert len(list((tmp_path / "out" / "episodes").glob("*.json"))) == 2 * total
    for name in ("fig_a1_block_share", "fig_a2_matrix", "fig_a2_forest", "fig_a4_layer_head"):
        assert (tmp_path / "out" / f"{name}.png").stat().st_size > 0


def test_main_merged_shards(tmp_path, capsys):
    total = build_tree(tmp_path / "root", shards=("s01", "s23"), episodes=2)
    assert sg.main(["--root", str(tmp_path / "root"), "--out", str(tmp_path / "out"), "--expected-episodes", str(total), "--repeats", "50"]) == 0
    assert "SUBGOAL_ALIGN=PASS" in capsys.readouterr().out


def test_main_label_failure(tmp_path, capsys):
    total = build_tree(tmp_path / "root")
    write_episode(tmp_path / "root", "mot", "s0", "RouteStick", 99, "diag", label_bad=True)
    with pytest.raises(SystemExit, match="局数"):      # 多一局时局数先不符
        sg.main(["--root", str(tmp_path / "root"), "--out", str(tmp_path / "out"), "--expected-episodes", str(total)])
    write_episode(tmp_path / "root", "ctx", "s0", "RouteStick", 99, "uniform")
    assert sg.main(["--root", str(tmp_path / "root"), "--out", str(tmp_path / "out"), "--expected-episodes", str(total + 1), "--repeats", "50"]) == 1
    text = capsys.readouterr().out
    assert "SUBGOAL_LABEL=FAIL" in text and "mismatches=1" in text
    assert "SUBGOAL_ALIGN" not in text
    assert (tmp_path / "out" / "label_mismatches.json").exists()
    assert not (tmp_path / "out" / "summary.json").exists()


def test_main_sanity_failure_stops_before_reading(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sg, "sanity_check", lambda: {"ok": False, "sanity_diag": 1.0, "sanity_uniform": 2.0, "results": {}})
    assert sg.main(["--root", str(tmp_path / "不存在"), "--out", str(tmp_path / "out")]) == 1
    assert "SUBGOAL_ALIGN=FAIL" in capsys.readouterr().out


def test_missing_shard_and_offline_report_errors(tmp_path):
    root = tmp_path / "root"
    build_tree(root, episodes=4)
    (root / "mot" / "s3" / "run").rename(root / "mot" / "s3" / "run-x")
    with pytest.raises(SystemExit, match="缺分片"):
        sg.discover_shards(root / "mot")
    import shutil
    shutil.rmtree(root / "mot" / "s3")
    with pytest.raises(SystemExit, match="不全"):
        sg.discover_shards(root / "mot")
    # 显式片名只用于冒烟
    assert len(sg.discover_shards(root / "mot", ["s0"])) == 1
    # 某局 offline 缺失
    write_episode(root, "ctx", "s0", "RouteStick", 50, "uniform", offline=False)
    record = sg.rollout_counts([root / "ctx" / "s0" / "run"])["accepted"]["RouteStick:50"][0]
    with pytest.raises(SystemExit, match="缺离线记录"):
        sg.load_episode(root / "ctx" / "s0" / "run", root / "ctx" / "offline", record)
    # 动作块与离线记录对不上
    nonce = record["episode_nonce"]
    (root / "ctx" / "offline" / nonce).mkdir(parents=True)
    source = next((root / "ctx" / "offline").glob("ctx*/offline_0000.npz"))
    shutil.copy(source, root / "ctx" / "offline" / nonce / "offline_0000.npz")
    with pytest.raises(SystemExit, match="对不上"):
        sg.load_episode(root / "ctx" / "s0" / "run", root / "ctx" / "offline", record)


def test_rollout_counts_invalid_and_retries(tmp_path):
    root = tmp_path / "root"
    write_episode(root, "ctx", "s0", "RouteStick", 1, "uniform", attempts=2)
    run = root / "ctx" / "s0" / "run"
    lost = run / "attempts" / "009" / "client"
    lost.mkdir(parents=True)
    (lost.parent / "commands.json").write_text(json.dumps({"episode_keys": ["RouteStick:2"]}))
    counts = sg.rollout_counts([run])
    assert sorted(counts["accepted"]) == ["RouteStick:1"]
    assert counts["invalid"] == ["RouteStick:2"]
    assert counts["infra_retries"] == 1
    ok, line = sg.run_line({"ctx": counts, "mot": {"accepted": {}, "invalid": ["x"] * 6, "infra_retries": 0}}, ["ctx", "mot"])
    assert not ok and "ROLLOUT_RUN=FAIL" in line and "invalid_mot=6" in line
