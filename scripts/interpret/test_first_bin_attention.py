"""first_bin_attention 的 CPU 合成数据测试：判定规则、局级平均、缺 motion 基线、Video 不比 160、敏感性与全链路。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
import first_bin_attention as fb  # noqa: E402

BUTTON, VIDEO = fb.TASKS


def fake_ep(model, task, eid, value, *, label="correct", swap=None, extra=None):
    """只带汇总所需字段的局；value 同时作为 RB 与 motion 交换富集。"""
    swap = value if swap is None else swap
    readouts = {name: None for name in fb.READOUTS}
    readouts["rb_vis_enrich"] = value
    if model == "with_motion":
        readouts["m_swap_enrich"] = swap
    named = {name: {"m_frame": .1, "f_vis_excess": value - 1, "f_vis_enrich": value, "f_first_excess": 0., "f_first_enrich": 1.}
             for name in fb.NAMED_HEADS}
    heads = {key: np.full((18, 8), .1) if key in ("m_frame", "m_motion") or key.startswith("f_") else None for key in fb.HEAD_KEEP}
    heads["f_vis_excess"] = np.full((18, 8), value - 1.)
    if model == "with_motion":
        heads["m_swap_excess"] = np.full((18, 8), swap - 1.)
        heads["m_vis_excess"] = np.zeros((18, 8))
    else:
        heads["m_motion"] = None
    p7 = {"frame": {"head": "L3H0", "t": 0, "t_rel_exec": 0}, "motion": {"head": "L17H1", "t": 160, "t_rel_exec": 160} if model == "with_motion" else None}
    ep = {"model": model, "task": task, "episode_id": eid, "episode_key": f"{task}:{eid}", "label": label, "outcome": "fail",
          "swap_times": 1, "pick_times": 1, "bin_count": 3, "before_buttons": False, "has_earlier_closes": False, "window": [0, 1, 2, 3],
          "complete": True, "readouts": readouts, "heads": heads, "peak": None, "close_row": None,
          "per_infer": [{"rel": r, "readouts": readouts, "named_heads": named, "P7": p7} for r in fb.RELS]}
    ep.update(extra or {})
    return ep


def cell(model, task, n, value, **kw):
    start = kw.pop("start", 0)
    noise = np.linspace(-.01, .01, n)
    return [fake_ep(model, task, start + i, value + float(noise[i]), **kw) for i in range(n)]


def verdict(block, code, model):
    return next(v for v in block["verdicts"] if v["readout"] == code and v["model"] == model)


def test_common_pattern_requires_same_direction_and_both_excluding():
    eps = cell("with_motion", BUTTON, 12, 2.) + cell("with_motion", VIDEO, 12, 3.)
    block = fb.primary_block(eps)
    assert len(block["intervals"]) == 12
    assert verdict(block, "M1", "with_motion")["verdict"] == "共同 pattern"
    assert verdict(block, "M2", "with_motion")["verdict"] == "共同 pattern"


def test_task_specific_and_unseen():
    eps = cell("with_motion", BUTTON, 12, 2.) + [fake_ep("with_motion", VIDEO, i, v) for i, v in enumerate(np.linspace(.5, 1.5, 12))]
    result = verdict(fb.primary_block(eps), "M1", "with_motion")
    assert result["verdict"] == "任务特有" and result["task"] == BUTTON
    eps = [fake_ep("with_motion", t, i, v) for t in fb.TASKS for i, v in enumerate(np.linspace(.5, 1.5, 12))]
    assert verdict(fb.primary_block(eps), "M1", "with_motion")["verdict"] == "未见"


def test_opposite_directions_are_flagged_not_called_common():
    eps = cell("with_motion", BUTTON, 12, 2.) + cell("with_motion", VIDEO, 12, .5)
    assert "方向相反" in verdict(fb.primary_block(eps), "M1", "with_motion")["verdict"]


def test_fewer_than_ten_episodes_is_unseen():
    eps = cell("with_motion", BUTTON, 12, 2.) + cell("with_motion", VIDEO, 9, 3.)
    block = fb.primary_block(eps)
    assert verdict(block, "M1", "with_motion")["verdict"] == "未见"
    video = next(i for i in block["intervals"] if i["readout"] == "M1" and i["model"] == "with_motion" and i["task"] == VIDEO)
    assert video["insufficient"] and video["ci95"] is None and video["n_episodes"] == 9


def test_difference_readout_group_size_and_selection_related():
    eps = (cell("with_motion", BUTTON, 12, 2., label="correct") + cell("with_motion", BUTTON, 12, 1., label="wrong", start=100)
           + cell("with_motion", VIDEO, 12, 1.5, label="correct") + cell("with_motion", VIDEO, 9, 1.5, label="wrong", start=100))
    block = fb.primary_block(eps)
    m3 = {i["task"]: i for i in block["intervals"] if i["readout"] == "M3" and i["model"] == "with_motion"}
    assert m3[BUTTON]["excludes_ref"] and m3[BUTTON]["mean"] == pytest.approx(1.0)
    assert m3[VIDEO]["insufficient"]
    assert any(s["readout"] == "M3" and s["task"] == BUTTON for s in block["selection_related"])


def test_bootstrap_uses_seed_and_is_deterministic():
    a, b = np.arange(12.), np.arange(12.) + 1
    assert fb.bootstrap_diff(a, b) == fb.bootstrap_diff(a, b)


def test_episode_averaged_first_no_pseudo_replication():
    """一局 4 次推理与一局 1 次推理在局级各算一个值。"""
    many = [{"readouts": dict.fromkeys(fb.READOUTS, None) | {"rb_vis_enrich": v}} for v in (9., 9., 9., 9.)]
    one = [{"readouts": dict.fromkeys(fb.READOUTS, None) | {"rb_vis_enrich": 1.}}]
    assert fb.episode_means(many)["rb_vis_enrich"] == 9.
    eps = [{"episode_key": "a", "readouts": fb.episode_means(many)}, {"episode_key": "b", "readouts": fb.episode_means(one)}]
    stat = fb.describe(eps, "rb_vis_enrich", 1.)
    assert stat["n_episodes"] == 2 and stat["mean"] == 5.
    assert stat["support"] == 1 and stat["counterexamples"] == ["b"]
    assert fb.episode_means([{"readouts": dict.fromkeys(fb.READOUTS, None)}])["rb_vis_enrich"] is None


def test_baseline_has_no_motion_readouts_and_six_baseline_free_intervals():
    eps = cell("without_motion", BUTTON, 12, 2.) + cell("without_motion", VIDEO, 12, 2.)
    block = fb.primary_block(eps)
    assert not [i for i in block["intervals"] if i["model"] == "without_motion" and i["readout"] in ("M2", "M4")]
    assert sum(i["readout"] in ("M1", "M3") for i in block["intervals"]) == 8
    assert verdict(block, "M1", "without_motion")["verdict"] == "共同 pattern"
    assert fb.describe(eps, "m_swap_enrich", 1.)["n_episodes"] == 0
    p7 = fb.p7_block(eps)
    assert p7["two_task_overlap"]["without_motion"]["motion"] is None
    assert fb.head_block(eps)["cells"]["without_motion"][BUTTON]["m_swap_excess"] is None


def peak(start):
    return {"infer_seq": 1, "rel": -1, "window_start": start, "window_start_rel_exec": start - 96, "mass": .01, "motion_stream_share": .5,
            "swap_overlap33": .25, "peak_head": "L17H1", "peak_head_attention": .8, "peak_head_fraction": .8}


def test_video_is_not_compared_with_fixed_160_window():
    eps = (cell("with_motion", BUTTON, 3, 2., extra={"peak": peak(160)})
           + cell("with_motion", VIDEO, 3, 2., extra={"peak": peak(160)}))
    block = fb.p4_block(eps)["motion_peak"]
    assert block[BUTTON]["fixed_window"]["at_160"] == 3
    assert block[BUTTON]["fixed_window"]["at_160_by_peak_head"] == 3
    assert "fixed_window" not in block[VIDEO]
    assert block[VIDEO]["peak_start_counts"] == {"160": 3}
    assert block[VIDEO]["peak_start_rel_exec_counts"] == {"64": 3}
    assert block[VIDEO]["swap_overlap33"]["mean"] == pytest.approx(.25)


def test_close_row_sensitivity_uses_only_queries_up_to_close():
    """第 0～2 行 query 全看帧 0，后面的 query 全看文本：截断后帧流质量应为 1，全部 query 时为 3/20。"""
    mem, p = 512, 512 + 576
    a = np.zeros((1, 1, 1, 20, p + 20), np.float32)
    a[0, 0, 0, :3, 0:16] = 1 / 16
    a[0, 0, 0, 3:, mem + 512] = 1.
    z = fb.sa.Record(P=np.array(p), MEM=np.array(mem), mem_order=np.arange(mem), mem_mask=np.ones(mem, bool),
                     frame_times=np.arange(32) * 8, motion_times=np.zeros(0, int), motion_mask=np.zeros(0, bool),
                     txt_len=np.array(10), step_idx=np.array(0), infer_seq=np.array(0), exec_start_idx=np.array(0), probs_action=a)
    result = fb.close_row_metrics(z, 1, 2)
    assert result["partial"]["frame_mass"] == pytest.approx(1.)
    assert result["full"]["frame_mass"] == pytest.approx(3 / 20)
    assert result["partial"]["motion_mass"] is None or result["partial"]["motion_mass"] == 0


def test_close_row_block_reports_difference():
    row = {"partial": {"frame_mass": .5, "motion_mass": None, "vis_enrich": 2., "m_swap_enrich": None},
           "full": {"frame_mass": .25, "motion_mass": None, "vis_enrich": 1., "m_swap_enrich": None}}
    eps = cell("without_motion", BUTTON, 3, 2., extra={"close_row": row})
    block = fb.close_row_block(eps)["without_motion"][BUTTON]
    assert block["episodes"] == 3
    assert block["frame_mass"]["partial_minus_full"]["mean"] == pytest.approx(.25)
    assert block["m_swap_enrich"]["partial"]["n_episodes"] == 0


def test_exclude_earlier_closes_reported_separately():
    eps = cell("with_motion", BUTTON, 12, 2.) + cell("with_motion", VIDEO, 12, 2.)
    for ep in eps[:3]:
        ep["has_earlier_closes"] = True
    rows = [dict(ep) for ep in eps]
    summary = fb.build_summary(eps, rows)
    excluded = summary["sensitivity"]["exclude_earlier_closes"]
    assert len(excluded["excluded"]) == 3
    button = next(i for i in excluded["intervals"] if i["readout"] == "M1" and i["model"] == "with_motion" and i["task"] == BUTTON)
    assert button["n_episodes"] == 9 and button["insufficient"]
    main = next(i for i in summary["primary"]["intervals"] if i["readout"] == "M1" and i["model"] == "with_motion" and i["task"] == BUTTON)
    assert main["n_episodes"] == 12


def test_top_ten_head_overlap_between_tasks():
    def with_heads(task, heads):
        out = []
        for i, name in enumerate(heads):
            ep = fake_ep("with_motion", task, i, 2.)
            ep["per_infer"] = [{"rel": 0, "readouts": ep["readouts"], "named_heads": {}, "P7": {"frame": {"head": name, "t": 0, "t_rel_exec": 0}, "motion": {"head": name, "t": 160, "t_rel_exec": 64}}}]
            out.append(ep)
        return out
    a = [f"L{i}H0" for i in range(12)]
    b = [f"L{i}H0" for i in range(7)] + [f"L{i}H1" for i in range(3)]
    block = fb.p7_block(with_heads(BUTTON, a) + with_heads(VIDEO, b))["two_task_overlap"]["with_motion"]["motion"]
    assert len(block["top"][BUTTON]) == fb.TOPN == 10
    assert block["overlap"] == 7
    assert "L0H0" in block["shared"]


def test_p1_rates_exclude_no_pick_from_denominator_and_stratify():
    rows = [{"model": "with_motion", "task": BUTTON, "label": label, "outcome": outcome, "pick_times": pick, "bin_count": 3, "swap_times": 1,
             "episode_key": f"{BUTTON}:{i}"}
            for i, (label, outcome, pick) in enumerate([("correct", "success", 1), ("correct", "fail", 2), ("wrong", "fail", 1), ("no_pick", "timeout", 1)])]
    block = fb.p1_block(rows)["cells"]["with_motion"][BUTTON]
    assert (block["correct"], block["wrong"], block["no_pick"]) == (2, 1, 1)
    assert block["selection_rate"]["mean"] == pytest.approx(2 / 3)
    assert block["official_success"]["mean"] == pytest.approx(1 / 4)
    assert block["strata"]["pick_times"]["2"]["correct"] == 1


def test_first_target_prompt_cuts_second_colour():
    prompt = "watch the video carefully, then pick up the container hiding the red cube, finally pick up another container hiding the blue cube"
    pieces = ["▁red", "▁blue"]
    info = fb.sa.target_color_info(fb.first_target_prompt(prompt), pieces)
    assert info["rows"] == [0] and info["color"] == "red"


def test_compare_tree_treats_null_as_nan_and_reports_structure():
    assert fb.compare_tree({"a": [1., None, "x"]}, {"a": [1.5, None, "x"]}) == (.5, [])
    worst, bad = fb.compare_tree({"a": 1., "b": None}, {"a": None, "b": None})
    assert bad and "/a" in bad[0]
    assert fb.compare_tree({"a": 1}, {"c": 1})[1]
    assert fb.compare_tree([1], [1, 2])[1]


# ---------------------------------------------------------------- 全链路：真实形状的合成 NPZ

def write_synthetic(d, seq, motion, target_seq_mass=False):
    """有效 action key 数恰为 1024，使 1/1024 在 f16 下精确，五流和不触发容差。"""
    d.mkdir(parents=True, exist_ok=True)
    frames = 29 if motion else 30
    n_motion = 16 if motion else 0
    mem = 512 + n_motion
    txt = 12
    p = mem + 576
    mask = np.zeros(mem, bool)
    mask[:frames * 16] = True
    mask[512:] = True
    valid = np.zeros(p + 20, bool)
    valid[:mem] = mask
    valid[mem:mem + 512] = True
    valid[mem + 512:mem + 512 + txt] = True
    valid[p:] = True
    assert valid.sum() == 1024
    action = np.zeros((10, 18, 8, 20, p + 20), np.float16)
    action[..., valid] = 1 / 1024
    prefix = np.zeros((18, 8, p, p), np.float16)
    key_valid = valid[:p]
    prefix[:, :, np.ix_(key_valid)[0], :] = 0
    row = key_valid.astype(np.float32) / key_valid.sum()
    prefix[:, :, key_valid, :] = row.astype(np.float16)
    text = np.zeros((18, 8, 64, p), np.float16)
    text[:, :, :txt, :] = row.astype(np.float16)
    frame_times = np.full(32, -1)
    frame_times[:frames] = np.arange(frames) * 8
    np.savez(d / f"infer_{seq:04d}.npz", P=p, MEM=mem, step_idx=seq * 16, exec_start_idx=0, infer_seq=seq,
             mem_order=np.arange(mem), mem_mask=mask, frame_times=frame_times,
             motion_times=np.arange(n_motion) * 16, motion_mask=np.ones(n_motion, bool), txt_len=txt,
             actions=np.zeros((20, 8)), state=np.zeros(8), top_frames=np.zeros(3, int),
             probs_action=action, probs_text=text, attn_path_maxabs=0.01, same_path_maxabs=0.)
    np.savez(d / f"prefix_{seq:04d}.npz", probs_prefix=prefix, P=p, MEM=mem, infer_seq=seq, step_idx=seq * 16)


@pytest.mark.parametrize("workers", [2])
def test_end_to_end_four_cells_with_real_shapes(tmp_path, workers):
    pieces = ["▁pick", "▁up", "▁the", "▁container", "▁hiding", "▁the", "▁green", "▁cube", ",", "▁finally", "\n", "x"]
    prompt = "pick up the container hiding the green cube, finally pick up another container hiding the red cube"
    args = []
    for model in fb.MODELS:
        root = tmp_path / model
        entries = []
        for t, task in enumerate(fb.TASKS):
            nonce = f"n{t}"
            write_synthetic(root / "attn" / nonce, 1, model == "with_motion")
            entries.append({"episode_key": f"{task}:{t}", "episode_nonce": nonce, "label": "correct", "G": 1, "window": [1],
                            "close_row_in_chunk": 5, "outcome": "fail", "difficulty": "easy", "swap_times": 1, "pick_times": 2,
                            "bin_count": 3, "before_buttons": False, "earlier_closes": [], "prompt": prompt, "text_token_pieces": pieces})
        entries.append({"episode_key": f"{fb.TASKS[0]}:9", "label": "no_pick", "G": None, "window": None, "close_row_in_chunk": None,
                        "outcome": "timeout", "swap_times": 1, "pick_times": 1, "bin_count": 3, "before_buttons": None, "earlier_closes": []})
        (root / "windows.json").write_text(json.dumps({"episodes": entries}))
        args += ["--windows", f"{model}={root}", "--attn-root", f"{model}={root / 'attn'}"]
    out = tmp_path / "out"
    assert fb.main(args + ["--out", str(out), "--workers", str(workers)]) == 0
    summary = json.loads((out / "pattern-summary.json").read_text())
    assert len(summary["primary"]["intervals"]) == 12
    assert all(v["verdict"] == "未见" for v in summary["primary"]["verdicts"])
    assert summary["P1"]["cells"]["with_motion"][fb.TASKS[0]]["no_pick"] == 1
    assert summary["checks"]["episodes_with_window"] == 4 and summary["checks"]["attn_path_maxabs"] == .01
    base = summary["P3"]["table"]["without_motion"][fb.TASKS[0]]["all"]["m_swap_enrich"]
    assert base["n_episodes"] == 0
    assert summary["P7"]["two_task_overlap"]["without_motion"]["motion"] is None
    assert summary["P7"]["two_task_overlap"]["with_motion"]["motion"]["overlap"] >= 1
    assert fb.TASKS[1] in summary["P4"]["motion_peak"] and "fixed_window" not in summary["P4"]["motion_peak"][fb.TASKS[1]]
    audit = json.loads((out / "episodes" / f"with_motion_{fb.TASKS[0]}_0" / "episode_window.json").read_text())
    assert audit["R3_target_color_verified"] is True
    assert audit["close_row"]["close_row_in_chunk"] == 5
    assert {r["infer_seq"] for r in audit["records"]} == {1}
    for name in ("fig_selection_rate", "fig_enrich_by_relative", "fig_head_diff", "fig_motion_peaks", "fig_peak_heads", "fig_pattern_table"):
        assert (out / f"{name}.png").stat().st_size > 0


def test_missing_nonce_and_missing_record_fail_loudly(tmp_path):
    entry = {"episode_key": f"{BUTTON}:1", "label": "correct", "G": 1, "window": [1], "outcome": "fail", "swap_times": 1, "pick_times": 1,
             "bin_count": 3, "prompt": "x"}
    path = tmp_path / "windows.json"
    path.write_text(json.dumps([entry]))
    with pytest.raises(SystemExit):
        fb.plan_jobs({"with_motion": [str(path)]}, {"with_motion": [str(tmp_path)]}, {}, None, tmp_path)
    with pytest.raises(FileNotFoundError):
        fb.analyse_episode({"model": "with_motion", "entry": entry, "attn_dir": str(tmp_path), "prompt": "x", "out": str(tmp_path)})
