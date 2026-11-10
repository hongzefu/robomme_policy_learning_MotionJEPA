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
    assert R.scope_check(dict(base, motion_content=mc)) == (True, True, False)
    assert R.scope_check(dict(base, motion_content=dict(mc, replace_log=[[64, 64]], n_replaced=1)))[0] is False
    assert R.scope_check(dict(base, motion_content=dict(mc, replace_log=[[64, 64], [96, 64]])))[0] is False
    zall = dict(base, group="Zall", donor=10000108)
    mz = {"mode": "swap_exact", "donor": "VideoRepick:10000108", "replace_log": [[s, s] for s in range(0, 288, 32)],
          "n_replaced": 9, "n_windows": 9}
    assert R.scope_check(dict(zall, motion_content=mz)) == (True, True, False)
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


# ---------------------------------------------------------------- 审查续改（单局跳过、交换表补全、早拿起、blind）
def _info(task, ep, W, es, swap_times=1, moved=True, s0=32):
    return {"task": task, "ep": ep, "key": f"{task}:{ep}", "status": "ok", "recv_ok": True, "donor_ok": True,
            "swap_times": swap_times, "moved": moved, "labels": {"S0": s0, "W": W, "es": es}}


def test_infeasible_donor_dropped_and_single_skip(tmp_path):
    task = "VideoUnmaskSwap"
    infos = {f"{task}:{i}": _info(task, i, [64, 96], 150) for i in range(4)}
    infos[f"{task}:3"]["labels"]["W"] = [64]          # 只有一个交换窗，补齐窗 96 不在它的录制里
    full = {s: np.full(DIM, s + 1, np.float32) for s in (0, 32, 64, 96, 128)}
    records = {k: dict(full) for k in infos}
    records[f"{task}:3"] = {0: full[0], 32: full[32], 64: full[64]}
    runs, notes = F.assign_groups(infos, task, records=records)
    assert all(r["donor_src"] != f"{task}:3" for r in runs if r["group"] == "A")
    assert len(notes["infeasible"]) == 3                 # 3 个接收局各剔一次
    # 单局问题：接收局 0 的录制缺 S0 窗 → 只跳过 S0，其余照常写出，ok 仍为真
    records[f"{task}:0"] = {k: v for k, v in full.items() if k != 32}
    ok, line, problems, done = F.write_donor_files(tmp_path / "d", runs, infos, records)
    assert ok and line.startswith("DONOR_FILES=PASS") and "skipped=1" in line
    assert problems == [f"{task}:0:S0:s0_missing"]
    assert len(done) == len(runs) - 1


def test_complete_schedule():
    scan = [[0, 1, 64, 114], [1, 2, 114, 164]]
    assert F.complete_schedule([[0, 1, 64, 114], [1, None, 114, 164]], scan) == (scan, [True, False])
    assert F.complete_schedule([[0, 1, 64, 114], [None, None, 114, 164]], scan)[0] == scan
    assert F.complete_schedule([[0, 1, 64, 114], [2, None, 114, 164]], scan)[0] is None   # 已知半对与扫描不符
    assert F.complete_schedule([[0, 1, 64, 114], [1, None, 114, 164]], None)[0] is None
    assert F.complete_schedule([[0, None, 64, 114]], scan)[0] is None                      # 条数不符补不了
    assert F.complete_schedule([[0, 1, 64, 114]], scan)[0] == [[0, 1, 64, 114]]            # 已完整：原样（不符留给 FP_REPLAY）
    rf = F.receipt_fields(receipt("ButtonUnmaskSwap", 0, [[0, 1, 64, 114], [1, None, 114, 164]], 0, None))
    assert not rf["schedule_complete"] and F.apply_completion(rf, {"swap_schedule": scan})
    assert rf["schedule"] == scan and rf["completed_from_scan"] and rf["raw_schedule"][1] == [1, None, 114, 164]
    # FP_REPLAY 只比已确定部分：补全来的不当作不符；已确定部分不符照报
    scan_entry = {"swap_times": 2, "swap_schedule": scan, "target_index": 0, "moved": True}
    assert R.replay_mismatch(scan_entry, rf, moved=False) == []
    rf2 = F.receipt_fields(receipt("ButtonUnmaskSwap", 0, [[0, 2, 64, 114], [1, None, 114, 164]], 0, None))
    F.apply_completion(rf2, {"swap_schedule": [[0, 2, 64, 114], [1, 2, 114, 164]]})
    assert R.replay_mismatch(scan_entry, rf2, moved=True) == ["swap_schedule"]


def test_excluded_episode_info(tmp_path):
    run_root = tmp_path / "run"
    row = {"task": "ButtonUnmaskSwap", "ep": 0, "run_root": str(run_root), "attn_out": str(tmp_path / "attn")}
    write_episode(row, 0, [[0, 1, 64, 114], [1, None, 114, 164]], 0, es=0)
    run = F.find_run(run_root, tmp_path / "attn", "ButtonUnmaskSwap:0")
    assert F.episode_info("ButtonUnmaskSwap", 0, run, tmp_path / "rec", None)["status"] == "excluded"
    info = F.episode_info("ButtonUnmaskSwap", 0, run, tmp_path / "rec",
                          {"swap_schedule": [[0, 1, 64, 114], [1, 2, 114, 164]]})
    assert info["status"] == "ok" and info["labels"]["W"] == [64, 96, 128, 160]
    r = R.load_run(dict(row, stage="n", group="N"), {})
    assert r["status"] == "excluded"


def test_scope_early_pick_button():
    mc = {"mode": "swap_exact", "donor": "ButtonUnmaskSwap:10000106", "replace_log": [[64, 64]], "n_replaced": 1,
          "n_windows": 3}
    base = {"task": "ButtonUnmaskSwap", "donor": 10000106, "group": "Z", "w_starts": [64, 96], "motion_content": mc}
    assert R.scope_check(dict(base, pick={"pick_frame": 120})) == (True, True, True)     # 96+32=128 > 120
    assert R.scope_check(dict(base, pick={"pick_frame": 130}))[0] is False               # 窗 96 本应已编码
    assert R.scope_check(dict(base, task="VideoRepick", pick={"pick_frame": None},
                              motion_content=dict(mc, donor="VideoRepick:10000106")))[0] is False
    empty = dict(base, motion_content=dict(mc, replace_log=[], n_replaced=0), pick={"pick_frame": 90})
    assert R.scope_check(empty) == (True, True, True)
    zall = {"task": "ButtonUnmaskSwap", "donor": 10000108, "group": "Zall", "pick": {"pick_frame": 70},
            "motion_content": {"mode": "swap_exact", "donor": "ButtonUnmaskSwap:10000108",
                               "replace_log": [[0, 0], [32, 32]], "n_replaced": 2, "n_windows": 2}}
    assert R.scope_check(zall) == (True, True, False)


def test_swap_line_blind_and_excluded():
    rows = lambda cat, moved: {"category": cat, "moved": moved}
    s = {"groups": {g: R.group_stats([rows("swap_blind", True), rows("correct", False)]) for g in F.GROUPS},
         "invalid": 0, "excluded": 2, "scope_early_pick": 1, "runs": 14}
    line = R.swap_line("motion", "ButtonUnmaskSwap", s)
    assert line.startswith("FP_SWAP=PASS") and "excluded=2" in line and "scope_early_pick=1" in line
    for g in ("Z", "Rnd", "S0", "A", "B"):
        assert f"{g}_blind=1/1" in line
    runs = [{"task": "VideoRepick", "stage": "swap", "group": "Z", "key": "VideoRepick:0", "status": "excluded"},
            {"task": "VideoRepick", "stage": "swap", "group": "Z", "key": "VideoRepick:1", "status": "invalid"}]
    summ = R.task_summary(runs, "VideoRepick")
    assert summ["excluded"] == 1 and summ["invalid"] == 1


# ---------------------------------------------------------------- SP-W2：难度参数、按 n 均衡局池、W_k、Bsame_k/Z_k、FP_SWAP_K、路径根
def test_scan_difficulties_and_tag(tmp_path):
    assert F.scan_seeds("VideoUnmaskSwap", 100, 3) == [(0, 100, "easy"), (1, 200, "medium"), (2, 300, "easy")]
    hard = F.scan_seeds("VideoUnmaskSwap", 2050000, 2, F.parse_difficulties("hard"))
    assert hard == [(0, 2050000, "hard"), (1, 2050100, "hard")]
    assert F.parse_difficulties("easy, medium,hard") == ("easy", "medium", "hard")
    with pytest.raises(SystemExit):
        F.parse_difficulties("easy,xhard")
    assert F.scan_tag(("easy", "medium")) is None and F.scan_tag(("hard",)) == "hard" and F.scan_tag(("easy",), "x") == "x"
    assert F.scan_file(tmp_path, "VideoRepick").name == "scan_VideoRepick.json"
    assert F.scan_file(tmp_path, "VideoRepick", "hard").name == "scan_VideoRepick_hard.json"


def test_swap_blind_k_and_order():
    sched = [[0, 1, 64, 114], [1, 2, 114, 164]]
    # 实际：物体 0 在槽 1；漏第 1 次 → 信念槽 0，实际占槽 0 的是物体 2；漏第 2 次（不涉及目标）→ 仍是目标
    assert F.swap_blind_k(3, 0, sched, 1) == 2 and F.swap_blind_k(3, 0, sched, 2) == 0
    assert F.swap_blind_k(3, 0, list(reversed(sched)), 1) == 2           # 按 start 排序定义第 k 次
    for s in ([[0, 1, 0, 1]], [[1, 2, 0, 1]], [[0, 2, 0, 1]]):           # n=1 时等于 swap_blind_index
        assert F.swap_blind_k(3, 0, s, 1) == F.swap_outcome(3, 0, s)[1]
    s3 = [[0, 1, 10, 20], [0, 2, 20, 30], [1, 2, 30, 40]]
    # 实际槽 [2,1,0]；漏第 1 次信念槽 2 → 物体 0（目标，第 1 次被第 2 次抵消）；漏第 2 次信念槽 1 → 物体 1
    assert [F.swap_blind_k(3, 0, s3, k) for k in (1, 2, 3)] == [0, 1, 0]
    with pytest.raises(ValueError):
        F.swap_blind_k(3, 0, sched, 3)
    fp = lambda i: {"index": i}
    assert F.classify_pick_k(fp(2), 0, 2) == "swap_blind_k" and F.classify_pick_k(fp(0), 0, 0) == "correct"
    assert F.classify_pick_k(fp(1), 0, 0) == "other" and F.classify_pick_k(None, 0, 2) == "none"


def test_swap_windows_k():
    sched = [[0, 1, 64, 114], [1, 2, 114, 164]]
    assert F.swap_frames_k(sched, 1) == (65, 115) and F.swap_frames_k(sched, 2) == (115, 165)
    assert F.swap_frames_k(sched, 3) is None
    w1, w2 = F.swap_windows_k("VideoUnmaskSwap", 200, sched, 1), F.swap_windows_k("VideoUnmaskSwap", 200, sched, 2)
    assert w1 == [64, 96] and w2 == [96, 128, 160]
    assert sorted(set(w1) | set(w2)) == F.swap_windows("VideoUnmaskSwap", 200, sched)
    # Button 取执行窗、带帧偏移；闭区间端点相交也算
    assert F.swap_windows_k("ButtonUnmaskSwap", 0, sched, 2, offset=-4) == [96, 128, 160]   # [111,161] 含 160
    assert F.swap_windows_k("ButtonUnmaskSwap", 0, sched, 2, offset=-6) == [96, 128]        # [109,159]
    assert F.swap_windows_k("VideoRepick", 200, [[0, 1, 63, 80]], 1) == [32, 64]
    lab = F.episode_labels("VideoUnmaskSwap", {"es": 200, "initial_env_step": None, "schedule": sched,
                                               "target_index": 0}, 3)
    assert lab["Wk"] == {1: [64, 96], 2: [96, 128, 160]} and lab["blind_k"] == {1: 2, 2: 0}


def test_k_slots():
    assert F.synthetic_ep(3, F.k_slot("Bsame_k", 1)) == 10000322 and F.synthetic_ep(3, F.k_slot("Z_k", 3)) == 10000327
    codes = set(F.K_SLOTS.values())
    assert len(codes) == 6 and not codes & set(F.SLOTS.values()) and max(codes) < 100
    assert len(F.SLOTS) == 8


def _scan_entry(task, i, seed, diff, sched):
    moved, blind = F.swap_outcome(3, 0, sched)
    return {"scan_index": i, "env_seed": seed, "difficulty": diff, "status": "ok", "n_objects": 3, "target_index": 0,
            "swap_times": len(sched), "swap_schedule": sched, "moved": moved, "swap_blind_index": blind,
            "es": 0 if task in F.BUTTON_TASKS else max(x[3] for x in sched) + 21}


S3 = [[0, 1, 64, 114], [1, 2, 114, 164], [0, 2, 164, 214]]


def fake_scan_by_n(out, per=4):
    """easy/medium：n=1、2（VR 另有 n=3）；VUS/BUS 另写 hard 扫描文件（n=3 与混进来的 n=1，后者不得被取）。"""
    for task in F.TASKS:
        entries, hard = [], []
        for i in range(4 * per):
            n = 1 + i % (3 if task == "VideoRepick" else 2)
            entries.append(_scan_entry(task, i, F.SEED0[task] + 100 * i, ("easy", "medium")[(i // 3) % 2], S3[:n]))
        (out / f"scan_{task}.json").write_text(json.dumps({"task": task, "entries": entries}))
        if task != "VideoRepick":
            for i in range(2 * per):
                n = 3 if i % 4 else 1
                hard.append(_scan_entry(task, i, F.SEED0[task] + 50000 + 100 * i, "hard", S3[:n]))
            (out / f"scan_{task}_hard.json").write_text(json.dumps({"task": task, "entries": hard}))


def test_by_swap_times_pick_sources():
    spec = [("easy", 1)] * 3 + [("medium", 2)] * 3 + [("hard", 3)] * 3 + [("hard", 1)] * 3 + [("medium", 3)] * 3
    entries = [_scan_entry("VideoUnmaskSwap", i, 1000 + i, d, S3[:n]) for i, (d, n) in enumerate(spec)]
    picked, _, _, bins = F.by_swap_times_pick("VideoUnmaskSwap", entries, 2)
    assert bins == {1: 2, 2: 2, 3: 2}
    assert all(e["difficulty"] == "hard" for e in picked if e["swap_times"] == 3)
    assert all(e["difficulty"] != "hard" for e in picked if e["swap_times"] < 3)
    picked, _, _, bins = F.by_swap_times_pick("VideoRepick", entries, 2)
    assert bins[3] == 2 and all(e["difficulty"] == "medium" for e in picked if e["swap_times"] == 3)


def test_pool_by_swap_times(tmp_path):
    O = tmp_path / "fpk"
    O.mkdir()
    fake_scan_by_n(O)
    assert F.main(["pool", "--out", str(O), "--by-swap-times", "--per-n", "3"]) == 0
    pool = json.loads((O / "pool.json").read_text())
    assert pool["mode"] == "by_swap_times" and pool["per_task"] == 9
    for task in F.TASKS:
        eps = [e for e in pool["episodes"] if e["task_name"] == task]
        assert [e["episode_id"] for e in eps] == list(range(9))
        assert {n: sum(e["swap_times"] == n for e in eps) for n in (1, 2, 3)} == {1: 3, 2: 3, 3: 3}
        n3 = {e["difficulty"] for e in eps if e["swap_times"] == 3}
        assert n3 <= ({"easy", "medium"} if task == "VideoRepick" else {"hard"})
        assert all(e["difficulty"] != "hard" for e in eps if e["swap_times"] < 3)
        assert json.loads((O / "metadata" / F.meta_name(task)).read_text())["record_count"] == 9
    # 旧模式只读 scan_<T>.json：hard 文件不进局池
    O2 = tmp_path / "fpold"
    O2.mkdir()
    fake_scan_by_n(O2)
    F.main(["pool", "--out", str(O2), "--per-task", "6"])
    assert all(e["difficulty"] != "hard" for e in json.loads((O2 / "pool.json").read_text())["episodes"])


def _ns(**kw):
    import argparse
    base = {k: None for k in ("main", "store", "bench_src", "policy_python", "sim_python", "stage2_python",
                              "mmevla_store", "log_dir", "gl_store")}
    base.update(release=False, run_tree="/rel/abc1234")
    base.update(kw)
    return argparse.Namespace(**base)


def test_roots_default_and_gl():
    assert F.resolve_roots(_ns()) == F.default_roots()
    gl = F.resolve_roots(_ns(gl_store="/g"))
    assert gl["policy_python"] == "/g/envs/policy/bin/python" and gl["sim_python"] == "/g/envs/official-sim/bin/python"
    assert gl["stage2_python"] == "/g/envs/stage2/bin/python" and gl["mmevla_store"] == "/g" and gl["release"]
    assert gl["bench_src"] == "/rel/abc1234/third_party/robomme_benchmark/src"
    assert F.resolve_roots(_ns(gl_store="/g", mmevla_store="/m"))["mmevla_store"] == "/m"
    assert F.reroot(str(F.STORE / "assets/x"), gl) == "/g/assets/x" and F.reroot("/other/x", gl) == "/other/x"
    env = F.render_env(gl, 0.3, "/rel/abc1234", "abc1234" + "0" * 33, "/o/motion")
    assert "git rev-parse" not in env and "reason=release_name" in env and "MMEVLA_V1_STORE=/g " in env
    assert "PYTHONPATH=$T:$T/src:/rel/abc1234/third_party/robomme_benchmark/src" in env and "FPS_M=/o/motion" in env
    old = F.ENV_TEMPLATE.format(frac=0.3, run_tree="/t", commit=COMMIT, store=F.STORE, bench=F.BENCH_SRC, out="/o",
                                py=F.SPEC_TEMPLATE["policy_python"], sim=F.SPEC_TEMPLATE["sim_python"])
    assert F.render_env(F.default_roots(), 0.3, "/t", COMMIT, "/o") == old
    assert F.parse_workers(None) == (list(F.P.WORKERS), list(F.P.ASSIGN_ORDER), dict(F.P.WORKER_CUDA))
    assert F.parse_workers("a=0,b=1")[2] == {"a": "0", "b": "1"}
    with pytest.raises(SystemExit):
        F.parse_workers("a=0,a=1")


def test_specs_gl_variant(tmp_path):
    O = tmp_path / "fpgl"
    O.mkdir()
    fake_scan(O)
    assert F.main(["pool", "--out", str(O), "--per-task", str(PER_TASK)]) == 0
    Mdir = O / "motion"
    rel = tmp_path / "releases" / COMMIT[:12]
    assert F.main(["specs", "--out", str(Mdir), "--model", "motion", "--commit", COMMIT, "--run-tree", str(rel),
                   "--gl-store", "/gl/v1-store", "--checkpoint", "/gl/v1-store/assets/vla-o16cc-59999-bd35/59999"]) == 0
    plan = json.loads((Mdir / "plan.json").read_text())
    assert plan["roots"]["release"] and plan["workers"]["order"] == list(F.GL_WORKERS)
    assert {r["worker"] for r in plan["runs"] if r["stage"] == "n"} <= set(F.GL_WORKERS)
    spec = json.loads(next((Mdir / "specs").glob("n-gl*.json")).read_text())
    assert spec["policy_python"] == "/gl/v1-store/envs/policy/bin/python" and spec["cuda"] == "0"
    assert spec["sim_python"] == "/gl/v1-store/envs/official-sim/bin/python"
    assert spec["checkpoint"] == "/gl/v1-store/assets/vla-o16cc-59999-bd35/59999"
    assert spec["target_contract"].startswith("/gl/v1-store/evaluation/") and spec["repo"] == str(rel)
    assert not any(isinstance(v, str) and str(F.STORE) in v for v in spec.values())
    env = (Mdir / "env.sh").read_text()
    assert "git rev-parse" not in env and "MMEVLA_V1_STORE=/gl/v1-store" in env
    assert (Mdir / "queues" / "n-gl15.sh").is_file() and not (Mdir / "queues" / "n-w00.sh").exists()
    assert "/gl/v1-store/logs/fp-swap-1110" in (Mdir / "launch_stage.sh").read_text()


def test_end_to_end_by_k(tmp_path):
    O = tmp_path / "fpk"
    O.mkdir()
    fake_scan(O)
    assert F.main(["pool", "--out", str(O), "--per-task", str(PER_TASK)]) == 0
    pool = json.loads((O / "pool.json").read_text())
    Mdir = O / "motion"
    assert F.main(["specs", "--out", str(Mdir), "--model", "motion", "--commit", COMMIT,
                   "--run-tree", str(tmp_path / "tree")]) == 0
    plan = json.loads((Mdir / "plan.json").read_text())
    assert "roots" not in plan and "workers" not in plan                 # 缺省：plan 与旧版同键
    elig = {F.episode_key(e["task_name"], e["episode_id"]): e for e in pool["episodes"]}
    for row in plan["runs"]:
        if row["stage"] == "n":
            e = elig[F.episode_key(row["task"], row["ep"])]
            pick = 0 if row["ep"] % 2 == 0 else (e["swap_blind_index"] if e["moved"] else 2)
            write_episode(row, pick, e["swap_schedule"], 0, record_dir=Mdir / "record", es=e["es"])
    assert F.main(["donors", "--out", str(Mdir), "--by-k"]) == 0
    plan = json.loads((Mdir / "plan.json").read_text())
    assert not any(r["stage"] in ("swap", "seeds") for r in plan["runs"])   # 旧组不生成
    ks = [r for r in plan["runs"] if r["stage"] == "kswap"]
    assert ks and all(r["seed"] == F.MAIN_SEED for r in ks)
    recv = plan["k_recv"]
    assert recv and all(F.split_key(k)[1] % 2 == 0 for k in recv)
    for key in recv:
        for k in range(1, elig[key]["swap_times"] + 1):
            assert sum(1 for r in ks if F.episode_key(r["task"], r["ep"]) == key and r["k"] == k
                       and r["group"] == "Z_k") == 1
    assert not any(r["k"] > elig[F.episode_key(r["task"], r["ep"])]["swap_times"] for r in ks)
    assert any(r["group"] == "Bsame_k" for r in ks)
    for r in ks:
        e = elig[F.episode_key(r["task"], r["ep"])]
        Wk = F.swap_windows_k(r["task"], e["es"], e["swap_schedule"], r["k"])
        assert r["w_starts"] == Wk and r["donor"] == F.synthetic_ep(r["ep"], F.k_slot(r["group"], r["k"]))
        with np.load(Mdir / F.DONOR_DIR / F.record_name(r["task"], r["donor"])) as z:
            starts, embs = z["start_frames"].tolist(), z["embs"]
        assert starts == Wk
        if r["group"] == "Z_k":
            assert not embs.any()
        else:
            d = elig[r["donor_src"]]
            assert d["swap_times"] == e["swap_times"] and r["donor_src"] != F.episode_key(r["task"], r["ep"])
            dW = F.swap_windows_k(r["task"], d["es"], d["swap_schedule"], r["k"])
            with np.load(Mdir / "record" / F.record_name(*F.split_key(r["donor_src"]))) as z:
                src = dict(zip(z["start_frames"].tolist(), z["embs"]))
            m = min(len(dW), len(Wk))
            assert np.array_equal(embs[:m], np.stack([src[s] for s in dW[:m]]))
    per_spec = {}
    for r in ks:                                                          # 同 (seed,k,组) 装同一 spec
        per_spec.setdefault(r["spec"], set()).add((r["k"], r["group"]))
    assert all(len(v) == 1 for v in per_spec.values())
    kseeds = [r for r in plan["runs"] if r["stage"] == "kseeds"]
    assert kseeds and {r["seed"] for r in kseeds} == {F.KSEED_SEED}
    assert max(sum(1 for r in kseeds if r["task"] == t) for t in F.TASKS) <= F.KSEED_PER_TASK
    # 读数输入：Bsame_k 拿 blind_k（≠目标时即 swap_blind_k），Z_k 拿对；replace_log = W_k
    for row in plan["runs"]:
        if row["stage"] not in ("kswap", "kseeds"):
            continue
        e = elig[F.episode_key(row["task"], row["ep"])]
        bk = F.swap_blind_k(3, 0, e["swap_schedule"], row["k"])
        mc = {"mode": "swap_exact", "donor": F.episode_key(row["task"], row["donor"]), "n_windows": 30,
              "n_replaced": len(row["w_starts"]), "replace_log": [[s, s] for s in row["w_starts"]]}
        write_episode(row, bk if row["group"] == "Bsame_k" else 0, e["swap_schedule"], 0, mc=mc, es=e["es"])
    R.main(["--root", str(Mdir)])
    out = json.loads((Mdir / "readout" / "readout.json").read_text())
    lines = out["lines"]
    assert any(l.startswith("SWAP_SCOPE=PASS") for l in lines)
    klines = [l for l in lines if l.startswith("FP_SWAP_K=")]
    n_cells = sum(len(R.k_cells(ks, t)) for t in F.TASKS)
    assert n_cells and len(klines) == n_cells
    assert all(l.startswith("FP_SWAP_K=PASS") and "invalid=0" in l for l in klines)
    for cellmap in out["k_summaries"].values():
        for s in cellmap.values():
            assert s["groups"]["Bsame_k"]["cats"]["swap_blind_k"] <= s["k_matters"]
            assert s["groups"]["Z_k"]["cats"]["correct"] == s["groups"]["Z_k"]["n"]
    # VUS n=2（[0,1] 后 [1,2]，目标 0）：漏第 1 次时目标在物体 2（matters）；第 2 次不涉及目标
    vus = out["k_summaries"]["VideoUnmaskSwap"]
    assert vus["n2k1"]["k_matters"] == vus["n2k1"]["recv"] > 0 and vus["n2k2"]["k_matters"] == 0
    assert vus["n2k1"]["groups"]["Bsame_k"]["cats"]["swap_blind_k"] == vus["n2k1"]["groups"]["Bsame_k"]["n"]
    assert any(l.startswith("SEED_FIRST_PICK") and "pairs=0" not in l for l in lines)


def test_bootstrap_ci():
    assert R.bootstrap_ci([]) is None
    assert R.bootstrap_ci([1, 1, 1]) == (1.0, 1.0)
    lo, hi = R.bootstrap_ci([0, 1] * 20)
    assert 0.25 < lo < 0.5 < hi < 0.75
    assert R.bootstrap_ci([0, 1] * 20) == (lo, hi)                        # 固定 seed，可复现
