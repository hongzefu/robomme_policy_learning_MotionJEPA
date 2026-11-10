"""fp_swap_prep 与 fp_swap_readout 的 CPU 测试：合成扫描、局池、终局、回执、轨迹与录制 npz，不起模型与仿真。"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import numpy as np
import pytest

from scripts.interpret import fp_swap_prep as F
from scripts.interpret.analysis import fp_swap_readout as R

DIM = 5
COMMIT = "c" * 40


# ---------------------------------------------------------------- 窗口（F1）
def test_demo_windows_and_tail():
    assert F.demo_starts(200) == [0, 32, 64, 96, 128, 160]          # 192+16 > 199
    assert F.demo_starts(150) == [0, 32, 64, 96, 128]               # 128+16=144 ≤ 149
    assert F.real_range(128, 150) == (128, 149)                     # 尾窗真实末帧 es−1
    assert F.real_range(64, 200) == (64, 96)
    assert F.demo_starts(0) == [] and F.demo_starts(17) == [0] and F.demo_starts(16) == []


def test_exec_windows():
    ex = F.exec_starts(0)
    assert ex[:3] == [0, 32, 64] and ex[-1] == 1248 and len(ex) == 40   # 1248+32 = 1280 ≤ 1300
    assert F.exec_starts(150)[:2] == [150, 182]
    assert F.real_range(150, 150) == (150, 182)
    assert F.all_starts(150) == F.demo_starts(150) + F.exec_starts(150)


def test_swap_window_boundaries_video():
    sched = [[0, 1, 64, 114], [1, 2, 114, 164]]
    assert F.swap_frames(sched) == (65, 165)
    W = F.swap_windows("VideoUnmaskSwap", 200, sched)
    assert W == [64, 96, 128, 160]                                  # 32:[32,64] 不含 65
    assert F.s0_start("VideoUnmaskSwap", 200, sched) == 32
    assert F.last_window_end(W, 200) == 192
    # 交换结束帧 165 恰为某窗真实首帧时也算相交
    assert 160 in F.swap_windows("VideoRepick", 200, [[0, 1, 100, 164]])
    # 交换开始 start=63 → 首帧 64，窗 32 的末帧 64 相交
    assert F.swap_windows("VideoRepick", 200, [[0, 1, 63, 113]])[0] == 32
    # Video 只取示范窗：es−1 == end 时执行窗 es 不进 W
    W2 = F.swap_windows("VideoRepick", 165, [[0, 1, 114, 164]])
    assert all(f < 165 for f in W2) and W2[-1] == 128


def test_swap_window_button_exec_and_offset():
    W = F.swap_windows("ButtonUnmaskSwap", 0, [[0, 1, 64, 114]])
    assert W == [64, 96]
    assert F.s0_start("ButtonUnmaskSwap", 0, [[0, 1, 64, 114]]) == 32
    assert F.swap_windows("ButtonUnmaskSwap", 0, [[0, 1, 64, 114], [1, 2, 114, 164]]) == [64, 96, 128, 160]
    assert F.frame_offset("ButtonUnmaskSwap", 0, 0) == 0
    assert F.frame_offset("ButtonUnmaskSwap", 0, 2) == -2
    assert F.frame_offset("VideoUnmaskSwap", 200, 999) == 0
    # 偏移 −2：交换帧 [63,113]，窗 32 [32,64] 相交
    assert F.swap_windows("ButtonUnmaskSwap", 0, [[0, 1, 64, 114]], offset=-2) == [32, 64, 96]
    assert F.next_start_after(160, 200) == 200 and F.next_start_after(64, 0) == 96


# ---------------------------------------------------------------- moved 与 swap_blind
def test_swap_outcome():
    assert F.swap_outcome(3, 0, [[0, 1, 64, 114]]) == (True, 1)
    assert F.swap_outcome(3, 0, [[0, 1, 0, 0], [1, 0, 0, 0]]) == (False, 0)
    assert F.swap_outcome(3, 0, [[0, 1, 0, 0], [1, 2, 0, 0]]) == (True, 2)
    assert F.swap_outcome(4, 3, [[0, 1, 0, 0]]) == (False, 3)
    with pytest.raises(ValueError):
        F.swap_outcome(3, 0, [[0, 3, 0, 0]])


def test_classify_pick():
    fp = lambda i: {"index": i}
    assert F.classify_pick(None, 0, 1, True) == "none"
    assert F.classify_pick(fp(0), 0, 1, True) == "correct"
    assert F.classify_pick(fp(1), 0, 1, True) == "swap_blind"
    assert F.classify_pick(fp(1), 0, 0, False) == "other"
    assert F.classify_pick(fp(2), 0, 1, True) == "other"


def test_synthetic_key():
    assert F.synthetic_ep(0, "A1") == 10000001 and F.synthetic_ep(37, "Zall") == 10003708
    assert len({F.synthetic_ep(5, s) for s in F.SLOTS}) == 8


def test_aligned_rows_pad():
    rec = {s: np.full(DIM, s, np.float32) for s in (64, 96, 128, 200)}
    embs, pad, srcs, why = F.aligned_rows([32, 64, 96], [64, 96], rec, 150)
    assert why is None and pad == 1 and srcs == [64, 96, 128]       # 96 之后第一个窗 128
    assert embs[:, 0].tolist() == [64, 96, 128]
    embs, pad, srcs, _ = F.aligned_rows([32], [64, 96], rec, 150)   # 供体多：截断
    assert pad == 0 and srcs == [64]
    _, _, _, why = F.aligned_rows([1, 2, 3], [64, 96], {64: rec[64], 96: rec[96]}, 150)
    assert why.startswith("donor_missing")


# ---------------------------------------------------------------- 分层选局
def test_stratified_pick():
    entries = []
    for i in range(40):
        entries.append({"env_seed": 1000 + i, "difficulty": ("easy", "medium")[i % 2], "swap_times": 1 + (i // 2) % 2,
                        "moved": i % 5 != 0})
    picked, counts, avail = F.stratified_pick(entries, 12)
    assert len(picked) == 12 and [e["env_seed"] for e in picked] == sorted(e["env_seed"] for e in picked)
    assert sum(counts.values()) == 12 and max(counts.values()) - min(v for v in counts.values()) <= 2
    picked, counts, _ = F.stratified_pick(entries[:3], 10)          # 不够时全取
    assert len(picked) == 3


# ---------------------------------------------------------------- 回执与标签
def receipt(task, ep, sched, target, fp, es=0, initial=0):
    return {"task_name": task, "episode_id": ep, "env_seed": 1, "difficulty": "easy", "target_index": target,
            "swap_times": len(sched), "swap_schedule": sched, "first_pick": fp, "exec_start_idx": es,
            "initial_env_step": initial}


def test_pick_status_and_first_pick_label():
    sched = [[0, 1, 64, 114]]
    fp = {"index": 1, "env_step": 300, "is_target": True, "subgoal_index_at_pick": 2, "successflag": False,
          "failureflag": False}
    rf = F.receipt_fields(receipt("ButtonUnmaskSwap", 0, sched, 1, fp))
    lab = F.episode_labels("ButtonUnmaskSwap", rf, 3)
    st = F.pick_status("ButtonUnmaskSwap", rf, lab, {"subgoal_index": 0})
    assert st["on_turn"] and st["f3_ok"] and st["pick_frame"] == 300 and lab["w_end"] == 128
    fp2 = dict(fp, env_step=100, subgoal_index_at_pick=1)
    rf2 = F.receipt_fields(receipt("ButtonUnmaskSwap", 0, sched, 1, fp2))
    st2 = F.pick_status("ButtonUnmaskSwap", rf2, lab, {"subgoal_index": 0})
    assert st2["before_buttons"] and not st2["on_turn"] and not st2["f3_ok"]
    # Video：首行子目标即第一次拿起
    rf3 = F.receipt_fields(receipt("VideoRepick", 0, [[0, 1, 100, 150]], 0, dict(fp, subgoal_index_at_pick=7), es=180))
    st3 = F.pick_status("VideoRepick", rf3, F.episode_labels("VideoRepick", rf3, 3), {"subgoal_index": 7})
    assert st3["on_turn"] and st3["f3_ok"] and st3["pick_frame"] is None
    # 回执显式给出 first_pick_subgoal_index 时优先
    assert F.expected_pick_subgoal("VideoRepick", dict(rf3, first_pick_subgoal_index=9), {"subgoal_index": 7}) == 9


def test_receipt_fields_w1_names():
    fp = {"index": 2, "env_step": 310, "action_index": 309, "is_target": False, "subgoal_index_at_pick": 1,
          "subgoal_index_after_pick": 1, "successflag": [False], "failureflag": [True], "outcome": "fail"}
    rf = F.receipt_fields(dict(receipt("VideoUnmaskSwap", 3, [[0, 1, 64, 114]], 0, fp, es=140),
                               first_pick_subgoal_index=1, truncated_first_pick=True))
    assert rf["schedule_complete"] and rf["first_pick"]["failureflag"] is True and rf["first_pick"]["outcome"] == "fail"
    assert F.expected_pick_subgoal("VideoUnmaskSwap", rf, {"subgoal_index": 0}) == 1
    bad = F.receipt_fields(receipt("ButtonUnmaskSwap", 0, [[0, None, 64, 114]], 0, None))
    assert bad["schedule_complete"] is False and bad["first_pick"] is None
    row = F.trace_row_fields({"env_step_before": 4, "env_step_after": 5, "action_index": 4, "subgoal_index": 2,
                              "bins_after": [[0, 0, 0]], "objects": [[1, 1, 1]]})
    assert row["env_step"] == 5 and row["objects"] == [[1, 1, 1]]
    assert F.trace_row_fields({"env_step_after": 5, "bins_after": [[0, 0, 0]]})["objects"] == [[0, 0, 0]]


def test_judge_label_counts():
    def run(key, is_target, failure, sub=2, index=None, outcome=None):
        fp = {"index": (0 if is_target else 1) if index is None else index, "env_step": 300, "is_target": is_target,
              "subgoal_index_at_pick": sub, "successflag": None, "failureflag": failure}
        return {"stage": "n", "status": "ok", "key": key, "truncated": True,
                "outcome": outcome or ("success" if is_target else "fail"), "rf": {"first_pick": fp, "target_index": 0},
                "pick": {"before_buttons": sub < 2, "on_turn": sub == 2}}
    runs = [run("a", True, False), run("b", False, True), run("c", False, False, sub=1)]
    line, _ = R.judge_label(runs, "motion")
    assert line.startswith("FIRST_PICK_LABEL=PASS") and "before_buttons=1" in line and "on_turn=2" in line
    line, d = R.judge_label(runs + [run("d", False, False)], "motion")
    assert "FAIL" in line and "wrong_but_success=1" in line and d["bad"] == ["d"]
    line, _ = R.judge_label([run("e", True, True)], "motion")
    assert "correct_but_fail=1" in line
    line, _ = R.judge_label([run("f", True, False, outcome="fail")], "motion")
    assert "outcome_mismatch=1" in line


def test_scope_check():
    base = {"task": "VideoRepick", "donor": 10000106, "group": "Z", "w_starts": [64, 96]}
    mc = {"mode": "swap_exact", "donor": "VideoRepick:10000106", "replace_log": [[64, 64], [96, 96]], "n_replaced": 2,
          "n_windows": 9}
    assert R.scope_check(dict(base, motion_content=mc)) == (True, True)
    assert R.scope_check(dict(base, motion_content=dict(mc, replace_log=[[64, 64]], n_replaced=1)))[0] is False
    assert R.scope_check(dict(base, motion_content=dict(mc, replace_log=[[64, 64], [96, 64]])))[0] is False
    zall = dict(base, group="Zall", donor=10000108)
    mz = {"mode": "swap_exact", "donor": "VideoRepick:10000108", "replace_log": [[s, s] for s in range(0, 288, 32)],
          "n_replaced": 9, "n_windows": 9}
    assert R.scope_check(dict(zall, motion_content=mz)) == (True, True)
    assert R.scope_check(dict(zall, motion_content=dict(mz, n_windows=10)))[1] is False


def test_framemap_problem():
    rf = {"schedule": [[0, 1, 114, 164]], "es": 190}
    assert R.framemap_problem("VideoRepick", rf, []) is None
    assert R.framemap_problem("VideoRepick", dict(rf, es=230), []) == "video_gap=65"
    rows = [{"env_step": k + 1, "action_index": k} for k in range(5)]
    assert R.framemap_problem("ButtonUnmaskSwap", {"schedule": [[0, 1, 64, 114]], "es": 0}, rows) is None
    rows[2]["env_step"] = 9
    assert R.framemap_problem("ButtonUnmaskSwap", {"schedule": [[0, 1, 64, 114]], "es": 0}, rows).startswith("button")


def test_env_template_keeps_caller_o():
    text = F.ENV_TEMPLATE.format(frac=0.3, run_tree="/t", commit=COMMIT, store="/s", bench="/b", out="/x/motion",
                                 py="/py", sim="/sim")
    assert "\nO=" not in text and "FPS_M=/x/motion" in text


# ---------------------------------------------------------------- 端到端（合成）
SCHED = {"VideoUnmaskSwap": [[[0, 1, 64, 114]], [[0, 1, 64, 114], [1, 2, 114, 164]], [[1, 2, 64, 114]]],
         "ButtonUnmaskSwap": [[[0, 1, 64, 114]], [[0, 1, 64, 114], [0, 1, 114, 164]], [[1, 2, 64, 114]]],
         "VideoRepick": [[[0, 1, 120, 170]], [[0, 1, 120, 170], [0, 2, 170, 220]], [[0, 1, 120, 170], [1, 0, 170, 220]]]}
# W1 回执 first_pick_subgoal_index 的实际值（VideoRepick 按名字找，这里取 6 作样例）
PICK_SUB = {"VideoUnmaskSwap": 1, "ButtonUnmaskSwap": 2, "VideoRepick": 6}
PER_TASK = 9


def fake_scan(out):
    for task in F.TASKS:
        entries = []
        for i in range(12):
            sched = SCHED[task][i % 3]
            target = 0
            moved, blind = F.swap_outcome(3, target, sched)
            entries.append({"scan_index": i, "env_seed": F.SEED0[task] + 100 * i, "difficulty": ("easy", "medium")[i % 2],
                            "status": "ok", "n_objects": 3, "target_index": target, "swap_times": len(sched),
                            "swap_schedule": sched, "moved": moved, "swap_blind_index": blind,
                            # 示范末帧在最后交换结束后 20 步（FP_FRAMEMAP 要求差 ≤ 40）
                            "es": 0 if task in F.BUTTON_TASKS else max(x[3] for x in sched) + 21})
        (out / f"scan_{task}.json").write_text(json.dumps({"task": task, "entries": entries}))


def write_episode(row, pick, sched, target, record_dir=None, mc=None, es=0):
    """为 plan 行造终局、回执、轨迹（与可选的 motion 录制与 episode.json）。"""
    task, ep = row["task"], row["ep"]
    key = F.episode_key(task, ep)
    nonce = uuid.uuid4().hex
    run_root, attn = Path(row["run_root"]), Path(row["attn_out"])
    (run_root / "accepted").mkdir(parents=True, exist_ok=True)
    is_target = pick == target
    sub0 = 0 if task == "ButtonUnmaskSwap" else PICK_SUB[task]
    pick_sub = PICK_SUB[task]
    term = {"episode_key": key, "episode_nonce": nonce, "task_name": task, "episode_id": ep,
            "outcome": "success" if is_target else "fail", "truncated_first_pick": True}
    (run_root / "accepted" / f"{key}.{nonce[:8]}.terminal.json").write_text(json.dumps(term))
    tdir = run_root / "attempts" / "a0" / "client" / "traces" / nonce
    (tdir / "meta").mkdir(parents=True)
    pick_step = 400
    fp = {"index": pick, "env_step": pick_step, "action_index": 2, "is_target": is_target,
          "subgoal_index_at_pick": pick_sub, "subgoal_index_after_pick": pick_sub, "successflag": is_target,
          "failureflag": not is_target, "outcome": "success" if is_target else "fail"}
    # W1 实际字段：env_step_after（步后）、objects、subgoal_index；拿起那一行另带 first_pick
    rows = [{"event": "step", "env_step_before": es + k, "env_step_after": es + k + 1, "action_index": k,
             "subgoal_index": sub0, "objects": [[0.0, 0.1 * j, 0.0] for j in range(3)]} for k in range(3)]
    rows[-1]["first_pick"] = fp
    (tdir / "step_trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rec = {"task_name": task, "episode_id": ep, "env_seed": 1, "difficulty": "easy", "target_index": target,
           "swap_times": len(sched), "swap_schedule": sched, "first_pick": fp, "exec_start_idx": es, "initial_env_step": 0,
           "first_pick_subgoal_index": pick_sub, "stop_after_first_pick": True, "truncated_first_pick": True}
    (tdir / "meta" / "hold_receipt.json").write_text(json.dumps(rec))
    if record_dir is not None:
        starts = F.all_starts(es)[:12] if es == 0 else F.demo_starts(es) + [es]
        rng = np.random.default_rng(1000 * F.TASKS.index(task) + ep)
        Path(record_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(record_dir) / F.record_name(task, ep), start_frames=np.asarray(starts, np.int32),
                 embs=rng.standard_normal((len(starts), DIM)).astype(np.float32), counts=np.ones(len(starts), np.int32))
    if mc is not None:
        (attn / nonce).mkdir(parents=True, exist_ok=True)
        (attn / nonce / "episode.json").write_text(json.dumps({"motion_content": mc}))


def test_end_to_end(tmp_path):
    O = tmp_path / "fp"
    O.mkdir()
    fake_scan(O)
    assert F.main(["pool", "--out", str(O), "--per-task", str(PER_TASK)]) == 0
    pool = json.loads((O / "pool.json").read_text())
    assert len(pool["episodes"]) == 3 * PER_TASK
    meta = json.loads((O / "metadata" / F.meta_name("VideoRepick")).read_text())
    assert meta["record_count"] == PER_TASK and {r["difficulty"] for r in meta["records"]} == {"easy", "medium"}
    Mdir = O / "motion"
    assert F.main(["specs", "--out", str(Mdir), "--model", "motion", "--commit", COMMIT,
                   "--run-tree", str(tmp_path / "tree")]) == 0
    plan = json.loads((Mdir / "plan.json").read_text())
    assert sum(r["stage"] == "n" for r in plan["runs"]) == 3 * PER_TASK
    assert sum(r["stage"] == "smoke" and r["group"] == "N" for r in plan["runs"]) == 3
    spec = json.loads(next((Mdir / "specs").glob("n-*.json")).read_text())
    assert spec["bin_trace"] is True and spec["stop_after_first_pick"] is True and "subgoal_trace" not in spec
    assert spec["profile"] == "plfull" and spec["record_motion"].endswith("motion/record")
    assert "\nO=" not in (Mdir / "env.sh").read_text()
    assert "fps-motion-" in (Mdir / "launch_stage.sh").read_text()
    # N 阶段：目标 0；局号偶数拿对，奇数拿 swap_blind（若 moved）或 2
    by_key = {F.episode_key(e["task_name"], e["episode_id"]): e for e in pool["episodes"]}
    for row in plan["runs"]:
        if row["stage"] != "n":
            continue
        e = by_key[F.episode_key(row["task"], row["ep"])]
        pick = 0 if row["ep"] % 2 == 0 else (e["swap_blind_index"] if e["moved"] else 2)
        write_episode(row, pick, e["swap_schedule"], 0, record_dir=Mdir / "record", es=e["es"])
    assert F.main(["donors", "--out", str(Mdir)]) == 0
    plan = json.loads((Mdir / "plan.json").read_text())
    recv = plan["recv"]
    assert recv and all(F.split_key(k)[1] % 2 == 0 for k in recv)
    swap = [r for r in plan["runs"] if r["stage"] == "swap"]
    groups = {g: sum(1 for r in swap if r["group"] == g) for g in F.SWAP_GROUPS}
    assert groups["Z"] == groups["Zall"] == groups["Rnd"] == len(recv)
    assert groups["S0"] == len(recv)
    assert groups["A"] <= 2 * len(recv) and groups["B"] <= 2 * len(recv)
    # A：同 swap_times 同 moved；B：同 swap_times、moved 相反（VideoRepick 可回退 n=2 moved=否）
    elig = {F.episode_key(e["task_name"], e["episode_id"]): e for e in pool["episodes"]}
    for r in swap:
        if r["group"] in ("A", "B"):
            d, me = elig[r["donor_src"]], elig[F.episode_key(r["task"], r["ep"])]
            assert r["donor_src"] != F.episode_key(r["task"], r["ep"])
            if r["group"] == "A":
                assert (d["swap_times"], d["moved"]) == (me["swap_times"], me["moved"])
            elif not r["fallback"]:
                assert d["swap_times"] == me["swap_times"] and d["moved"] != me["moved"]
            else:
                assert r["task"] == "VideoRepick" and d["swap_times"] == 2 and not d["moved"]
    # 合成供体文件：起点 = 接收局 W；Zall = 全部起点；A/B 编码 = 供体 W 第 k 窗
    for r in swap:
        path = Mdir / F.DONOR_DIR / F.record_name(r["task"], r["donor"])
        assert r["donor"] == F.synthetic_ep(r["ep"], r["slot"])
        with np.load(path) as z:
            starts, embs = z["start_frames"].tolist(), z["embs"]
        if r["group"] == "Zall":
            assert starts == F.all_starts(elig[F.episode_key(r["task"], r["ep"])]["es"]) and not embs.any()
        else:
            assert starts == r["w_starts"]
        if r["group"] in ("A", "B"):
            d = elig[r["donor_src"]]
            dW = F.swap_windows(r["task"], d["es"], d["swap_schedule"])
            with np.load(Mdir / "record" / F.record_name(*F.split_key(r["donor_src"]))) as z:
                src = dict(zip(z["start_frames"].tolist(), z["embs"]))
            k = min(len(dW), len(starts))
            assert np.array_equal(embs[:k], np.stack([src[s] for s in dW[:k]]))
    seeds = [r for r in plan["runs"] if r["stage"] == "seeds"]
    assert {r["seed"] for r in seeds} == {1, 2}
    per = {}
    for r in seeds:
        per[(r["task"], r["group"], r["seed"])] = per.get((r["task"], r["group"], r["seed"]), 0) + 1
    assert max(per.values()) <= F.SEEDCHECK_PER_GROUP
    assert sum(1 for r in plan["runs"] if r["stage"] == "multieq" and r.get("variant") == "single") == \
        sum(1 for r in plan["runs"] if r["stage"] == "multieq" and r.get("variant") == "multi")
    # 交换局：造读数输入（拿对与 swap_blind 交替），replace_log 正好等于 W
    for row in plan["runs"]:
        if row["stage"] not in ("swap", "seeds", "multieq"):
            continue
        e = elig[F.episode_key(row["task"], row["ep"])]
        if row["group"] == "N":
            write_episode(row, 0, e["swap_schedule"], 0, es=e["es"])
            continue
        starts = F.all_starts(e["es"]) if row["group"] == "Zall" else row["w_starts"]
        mc = {"mode": "swap_exact", "donor": F.episode_key(row["task"], row["donor"]), "n_windows": len(starts),
              "n_replaced": len(starts), "replace_log": [[s, s] for s in starts]}
        pick = e["swap_blind_index"] if (row["group"] == "S0" and e["moved"]) else 0
        write_episode(row, pick, e["swap_schedule"], 0, mc=mc, es=e["es"])
    rc = R.main(["--root", str(Mdir)])
    out = json.loads((Mdir / "readout" / "readout.json").read_text())
    lines = out["lines"]
    get = lambda name: [l for l in lines if l.startswith(name)]
    assert get("FIRST_PICK_LABEL=PASS") and get("FP_REPLAY=PASS") and get("SWAP_SCOPE=PASS") and get("FP_FRAMEMAP=PASS")
    assert get("MULTI_SWAP_EQ=PASS")
    swap_lines = get("FP_SWAP=")
    assert len(swap_lines) == 3 and all(l.startswith("FP_SWAP=PASS") and "invalid=0" in l for l in swap_lines)
    s0_task = out["summaries"]["VideoUnmaskSwap"]["groups"]["S0"]
    assert s0_task["blind"] == s0_task["moved"]
    assert rc == 0
    # 交集：同一目录当两个模型
    (O / "wan").symlink_to(Mdir)
    assert R.main(["--root", str(O), "--intersect", "motion,wan"]) == 0
    inter = json.loads((O / "readout" / "intersect.json").read_text())
    assert len(inter["recv"]) == len(recv) and len(inter["lines"]) == 3


def test_wan_gate_few():
    infos = {}
    for ep in range(4):
        infos[f"VideoRepick:{ep}"] = {"task": "VideoRepick", "ep": ep, "key": f"VideoRepick:{ep}", "status": "ok",
                                      "recv_ok": True, "donor_ok": True, "swap_times": 1, "moved": True,
                                      "labels": {"S0": 32, "W": [64], "es": 200}}
    runs, _ = F.assign_groups(infos, "VideoRepick", few=True)
    assert {r["group"] for r in runs} == {"Z", "S0", "Zall"} and len(runs) == 12
    runs, notes = F.assign_groups(infos, "VideoRepick", few=False)
    # 全部 moved、swap_times=1：B 无同 n 相反 moved → 回退 n=2 moved=否，也没有 → 记 short
    assert sum(r["group"] == "A" for r in runs) == 8 and not any(r["group"] == "B" for r in runs)
    assert len(notes["fallback"]) == 4 and len(notes["short"]) == 4
