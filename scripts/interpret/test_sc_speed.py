"""sc_speed_prep 与 analysis.sc_speed_readout 的 CPU 合成数据单测（计划 1110 第二部分 1.5）。不跑仿真、不起模型。"""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from scripts.interpret import sc_speed_prep as C
from scripts.interpret.analysis import sc_speed_readout as R
from scripts.interpret.hover_points import cube_position, press_step


# ---------------------------------------------------------------- Δ* 与键
def test_delta_star_table():
    table = {(60, 80): (30, 50, 70, 90), (60, 120): (90, 150, 210, 270), (80, 120): (60, 100, 140, 180)}
    for (M, Mp), want in table.items():
        assert [C.delta_star(M, Mp, n) for n in (2, 3, 4, 5)] == list(want)
        assert [C.delta_star(Mp, M, n) for n in (2, 3, 4, 5)] == [-x for x in want]
        assert [C.delta_star(M, M, n) for n in (2, 3, 4, 5)] == [0, 0, 0, 0]


def test_speed_group_and_horizon():
    assert C.speed_group(60, 120) == "slow" and C.speed_group(120, 60) == "fast" and C.speed_group(80, 80) == "same"
    assert C.horizon(60, 120, 3) == 392 and C.horizon(120, 60, 3) == 392
    assert C.horizon(120, 60, 5) <= 60 * C.STOP_TIME_OVERRIDE


def test_syn_key_unique_and_reversible():
    keys = {C.syn_ep(r, s) for r in range(72) for s in C.SLOTS}
    assert len(keys) == 72 * 3 and min(keys) == 20000000 and max(keys) == 20000712
    for r in range(72):
        for s in C.SLOTS:
            assert C.parse_syn(C.syn_ep(r, s)) == (r, s)
    assert [C.slot_of(m) for m in (60, 80, 120)] == [0, 1, 2]
    with pytest.raises(ValueError):
        C.syn_ep(1, 3)
    with pytest.raises(ValueError):
        C.parse_syn(20000003)


# ---------------------------------------------------------------- pool
def test_pool_balanced_and_excludes_banned():
    cells = list(C.CELLS)
    params = lambda seed: cells[seed % len(cells)]          # 桩：按种子轮转格子
    banned = {100, 113}
    picked, scanned, hit = C.build_pool(100, 2, banned, params_fn=params)
    assert len(picked) == 24 and hit == 2 and scanned == 26
    assert not {e["seed"] for e in picked} & banned
    assert all(sum(1 for e in picked if (e["M"], e["N"]) == c) == 2 for c in cells)
    eps, smoke = C.number_pool(picked, "easy")
    assert [e["ep"] for e in eps] == list(range(24))
    assert [(e["M"], e["N"]) for e in eps] == sorted((e["M"], e["N"]) for e in eps)
    assert [(eps[i]["M"], eps[i]["N"]) for i in smoke] == [(60, 3), (80, 3), (120, 3)]
    line, ok = C.pool_line(eps, 2, banned)
    assert ok and line == "SC_POOL=PASS eps=24 cells=12 per_cell=2"
    payload = C.metadata_payload(eps)
    assert payload["records"][0].keys() == {"task", "episode", "seed", "difficulty"}


def test_pool_real_task_params_small():
    picked, _, _ = C.build_pool(C.SEED_START, 1)
    from scripts.interpret.hover_points import task_params
    assert len(picked) == 12 and all(task_params(e["seed"]) == (e["M"], e["N"]) for e in picked)


# ---------------------------------------------------------------- 回放表与 T_h
BUTTON = [0.1, 0.2, 0.0075]


def fake_rows(n_steps, press_at=None, hover_from=None, first=1):
    rows = []
    for i in range(n_steps):
        b, a = first + i, first + i + 1
        hovering = hover_from is not None and a >= hover_from
        tcp = [BUTTON[0], BUTTON[1], 0.15] if hovering else [0.4, 0.0, 0.5]
        pressed = press_at is not None and a >= press_at
        rows.append({"event": "step", "env_step_before": b, "env_step_after": a, "tcp_after": tcp,
                     "button_pressed_after": pressed, "button_pressed_before": press_at is not None and b >= press_at,
                     "executed_action": [float(b)] * 8})
    return rows


def test_hold_step_rules():
    th, src, press, reach = C.hold_step(fake_rows(200, press_at=150, hover_from=60), BUTTON)
    assert (th, src, press, reach) == (60, "reach", 150, 60)
    th, src, press, _ = C.hold_step(fake_rows(200, press_at=150), BUTTON)
    assert (th, src, press) == (134, "press-16", 150)
    th, src, press, _ = C.hold_step(fake_rows(120), BUTTON)
    assert (th, src, press) == (121, "last", None)
    th, src, _, _ = C.hold_step(fake_rows(20, press_at=10), BUTTON)
    assert th == 2 and src == "press-16"


def test_replay_actions_align_env_step_before():
    rows = fake_rows(200, press_at=150, hover_from=60)
    acts = C.replay_actions(rows, 60)
    assert len(acts) == 59 and acts[0] == [1.0] * 8 and acts[-1] == [59.0] * 8
    entry = C.donor_entry(123, 60, 3, 120, acts, 60)
    assert entry["hold_qpos"] == acts[-1] and entry["first_env_step"] == 1 and entry["horizon"] == 392
    assert entry["move_interval_override"] == 120 and entry["stop_time_override"] == 99
    assert entry["orig_move_interval"] == 60 and entry["target_pass"] == 3 and entry["t_hold"] == 60
    assert len(entry["actions"]) == entry["t_hold"] - entry["first_env_step"]
    # 首步不是 1 时按步前值对齐
    rows2 = fake_rows(50, first=3)
    acts2 = C.replay_actions(rows2, 10, first=3)
    assert len(acts2) == 7 and acts2[0] == [3.0] * 8
    with pytest.raises(ValueError):
        C.donor_entry(1, 60, 3, 60, acts2, 10)                 # first_env_step 不符 → 长度不符
    gap = [r for r in rows if r["env_step_before"] != 30]
    with pytest.raises(ValueError):
        C.replay_actions(gap, 60)


def test_replay_eq_window_selection():
    starts = [0, 32, 64, 96, 128]
    assert C.replay_starts(starts, 32) == []          # 帧 32 由第 32 个动作（步前值 32 ≥ T_h）产生
    assert C.replay_starts(starts, 33) == [0]
    assert C.replay_starts(starts, 65) == [0, 32]
    assert C.replay_starts(starts, 96) == [0, 32]
    assert C.replay_starts(starts, 97) == [0, 32, 64]


def test_write_speed_donor_readback(tmp_path):
    embs = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    path = tmp_path / C.record_name(C.syn_ep(5, 2))
    assert C.write_speed_donor(path, [0, 32, 64, 96], embs) is None
    with np.load(path) as z:
        assert z["start_frames"].dtype == np.int32 and z["counts"].tolist() == [1] * 4
        assert z["embs"].tobytes() == embs.tobytes()


def synthetic_trace(Mp, steps, half=0.02, length=0.6, pressed_at=None):
    rows = []
    for t in range(1, steps + 1):
        def pos(k):
            return [(cube_position(k, Mp) - 0.5) * length, 0.0, 0.01]
        rows.append({"event": "step", "env_step_before": t, "env_step_after": t + 1, "move_interval": Mp,
                     "target_pass": 99, "cube_position_before": pos(t), "cube_position_after": pos(t + 1),
                     "target_position_before": [0.0, 0.0, 0.01], "target_position_after": [0.0, 0.0, 0.01],
                     "button_pressed_before": False, "button_pressed_after": False})
    return {"meta": {"episode_key": "StopCube:0"}, "rows": rows, "source": {"path": "x"}}


def test_move_override_check():
    ok, dev, why = C.move_override_check(synthetic_trace(80, 3 * 80 + 32), {"cube_half_size": 0.02}, 80, 3)
    assert ok and dev <= 1 and why is None
    tr = synthetic_trace(80, 3 * 80 + 32)
    tr["rows"][5]["move_interval"] = 60
    assert C.move_override_check(tr, {"cube_half_size": 0.02}, 80, 3)[2] == "move_interval_rows=1"
    ok, _, _ = C.move_override_check(synthetic_trace(80, 3 * 80 + 32), {"cube_half_size": 0.02}, 60, 3)
    assert not ok


# ---------------------------------------------------------------- spec
def test_specs_n_stage_keys(tmp_path):
    meta = tmp_path / "metadata"
    meta.mkdir()
    eps = [{"task_name": "StopCube", "episode_id": i, "ep": i, "env_seed": 2500000 + i, "M": 60, "N": 3,
            "difficulty": "easy"} for i in range(3)]
    name = "record_dataset_StopCube_metadata.json"
    (meta / name).write_text(json.dumps(C.metadata_payload(eps)))
    roots = {"aspen": {k: f"/a/{k}" for k in C.ASPEN}, "gl": None}
    root = tmp_path / "motion"
    b = C.SCBuilder(root, {"aspen": dict(C.MODELS["motion"])}, roots, C.make_workers(0, 2), 0.3, meta, {name: "x"})
    C.build_n(b, [0, 1, 2], {e["ep"]: e for e in eps})
    specs = {n: s for n, (s, *_rest) in b.specs.items()}
    assert len(specs) == 6
    for n, s in specs.items():
        assert s["profile"] == "plfull" and s["stopcube_mode"] == "normal" and s["save_attn"] is False
        assert "subgoal_trace" not in s and s["plfull_metadata_dir"] == str(meta)
        assert s["model_seed"] == (42 if "-N-" in n else 1)
        assert ("record_motion" in s) == ("-N-" in n)
    assert json.loads((root / C.NORMAL_CFG).read_text()) == {"schema_version": 1, "episodes": {}}
    b.write("n")
    assert (root / "queues" / "n-w00.sh").read_text().count("run_spec ") == 2


def test_slot_stage_unique_recv_per_spec(tmp_path):
    eps = {i: {"task_name": "StopCube", "episode_id": i, "ep": i, "env_seed": 2500000 + i, "M": (60, 80, 120)[i % 3],
               "N": 2 + i % 4, "difficulty": "easy"} for i in range(30)}
    roots = {"aspen": {k: f"/a/{k}" for k in C.ASPEN}, "gl": {k: f"/g/{k}" for k in C.ASPEN}}
    models = {"aspen": dict(C.MODELS["motion"]), "gl": dict(C.MODELS["motion"])}
    b = C.SCBuilder(tmp_path, models, roots, C.make_workers(2, 2), 0.3, tmp_path, {"m": "x"})
    items = [(s, r) for s in C.SLOTS for r in eps]
    C.build_slot_stage(b, "iv", items, eps)
    seen = set()
    for name, (spec, manifest, stage, worker) in b.specs.items():
        recvs = [k for k in spec["swap_map"]]
        assert len(recvs) == len(set(recvs))
        slots = {C.parse_syn(int(v.rpartition(":")[2]))[1] for v in spec["swap_map"].values()}
        assert len(slots) == 1 and spec["ablate"].startswith("motion_swap_exact:")
        for k, v in spec["swap_map"].items():
            r, s = C.parse_syn(int(v.rpartition(":")[2]))
            assert k == f"StopCube:{r}"
            seen.add((s, r))
    assert seen == set(items)
    holds = C.write_holds(tmp_path, "iv", "motion", C.make_workers(2, 2), "/g/logs", lambda st, w: bool(b.queue.get((st, w))))
    text = holds[0].read_text()
    assert "sleep 45" in text and "g0a g0b" in text


# ---------------------------------------------------------------- 读数
def test_classify_and_overlap():
    wt, wn = (140, 160), (290, 310)
    assert R.classify_press(150, wt, wn) == "true"
    assert R.classify_press(300, wt, wn) == "new"
    assert R.classify_press(100, wt, wn) == "neither_early"
    assert R.classify_press(200, wt, wn) == "neither_between"
    assert R.classify_press(400, wt, wn) == "neither_late"
    assert R.classify_press(None, wt, wn) == "nopress"
    assert R.classify_press(150, (140, 160), (155, 175)) == "overlap"


def follow_rows(beta, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(36):
        M = (60, 80, 120)[i % 3]
        for Mp in (60, 80, 120):
            if Mp == M:
                continue
            for N in (2, 4):
                ds = C.delta_star(M, Mp, N)
                rows.append({"recv": f"StopCube:{i}", "dstar": ds, "delta": beta * ds + rng.normal(0, 5)})
    return rows


def test_follow_beta_one_and_zero():
    one = R.follow_stats(follow_rows(1.0))
    zero = R.follow_stats(follow_rows(0.0))
    assert abs(one["beta"] - 1) < 0.05 and one["ci"][0] < 1 < one["ci"][1]
    assert abs(zero["beta"]) < 0.05 and zero["ci"][0] < 0 < zero["ci"][1]
    assert abs(one["beta0"] - 1) < 0.05
    again = R.follow_stats(follow_rows(1.0))
    assert again["ci"] == one["ci"]                       # 固定 rng，重采样可复现


def test_success_stats_overlap_excluded():
    rows = [{"recv": "StopCube:0", "cat": "new", "base_new": False, "true": False},
            {"recv": "StopCube:1", "cat": "true", "base_new": False, "true": True},
            {"recv": "StopCube:2", "cat": "overlap", "base_new": True, "true": True},
            {"recv": "StopCube:3", "cat": "nopress", "base_new": False, "true": False},
            {"recv": "StopCube:4", "cat": "neither_late", "base_new": True, "true": False}]
    st = R.success_stats(rows)
    assert (st["new"], st["base"], st["n"], st["n_all"], st["true"]) == (1, 1, 4, 5, 2)
    assert (st["overlap"], st["nopress"], st["neither"]) == (1, 1, 1) and st["diff"] == 0.0
    assert st == R.success_stats(rows)


def test_analyze_synthetic(monkeypatch):
    """合成两种回合：R 完全按 token（p_R = T(M′,N)+δ）→ β≈1、新速度成功全中。"""
    by_ep, runs = {}, []
    for i in range(12):
        M, N = (60, 80, 120)[i % 3], 2 + i % 4
        by_ep[i] = {"ep": i, "M": M, "N": N}
        T = press_step(M, N)
        common = {"ep": i, "recv": f"StopCube:{i}", "status": "ok", "M": M, "N": N, "move_interval": M,
                  "target_pass": N, "last_frame": None}
        runs.append(dict(common, group="N", press=T + 2, success=True, outcome="success", slot=None))
        runs.append(dict(common, group="S", press=T + 4, success=True, outcome="success", slot=None))
        for slot, Mp in C.SLOTS.items():
            runs.append(dict(common, group="D", slot=slot, M_prime=Mp, outcome=R.DONOR_OUTCOME, press=None))
            Tp = press_step(Mp, N)
            syn = C.syn_ep(i, slot)
            stop = Tp + 2
            runs.append(dict(common, group="R", slot=slot, M_prime=Mp, syn=syn, kind=C.speed_group(M, Mp),
                             press=stop, success=(Mp == M), outcome="success" if Mp == M else "fail",
                             last_frame=stop - 1,
                             motion_content={"mode": "swap_exact", "donor": f"StopCube:{syn}",
                                             "n_windows": len(range(0, stop - 33, 32)),
                                             "n_replaced": len(range(0, stop - 33, 32)),
                                             "replace_log": [[s, s] for s in range(0, stop, 32) if s + 32 <= stop - 1]}))

    def fake_window(d, N):
        T = press_step(d["M_prime"], N)
        return (T - 10, T + 10), None
    monkeypatch.setattr(R, "donor_window", fake_window)
    lines, detail = R.analyze({"model": "motion"}, by_ep, runs)
    text = "\n".join(lines)
    assert "SC_POOL=PASS model=motion eps=12 mismatch=0" in text
    assert "WINDOW_GEOM=PASS model=motion donors=36 resolved=36 formula_dev_max=0.0" in text
    assert "PRESS_LABEL=PASS model=motion runs=60 mismatch=0" in text
    assert re.search(r"^SPEED_SCOPE=PASS model=motion runs=36 unreplaced=0 scope_bad=0$", text, re.M)
    assert "SEED_PRESS model=motion pairs=12 mad=2.0" in text
    follow = next(l for l in lines if l.startswith("SPEED_FOLLOW "))
    assert re.match(r"SPEED_FOLLOW model=motion beta=1\.000 ci=\[1\.000,1\.000\] beta0=1\.000 same_delta=0\.0 "
                    r"ci=\[0\.0,0\.0\] n=24$", follow)
    allg = next(l for l in lines if "group=all" in l)
    assert allg.startswith("SPEED_SUCCESS model=motion group=all new_speed=24/24 base=0/24 diff=1.000")
    assert "true_speed=0/24 neither=0 nopress=0 overlap=0" in allg
    assert "SPEED_SAME model=motion true_speed=12/12" in text


LINE_FORMATS = [
    r"^SC_POOL=(PASS|FAIL) eps=\d+ cells=\d+ per_cell=\d+$",
    r"^SPEED_SUCCESS model=\w+ group=(fast|slow|all) new_speed=\d+/\d+ base=\d+/\d+ diff=\S+ ci=\[\S*\] "
    r"true_speed=\d+/\d+ neither=\d+ nopress=\d+ overlap=\d+$",
    r"^SPEED_FOLLOW model=\w+ beta=\S+ ci=\[\S*\] beta0=\S+ same_delta=\S+ ci=\[\S*\] n=\d+$",
]


def test_line_formats():
    eps, _ = C.number_pool([{"seed": i, "M": c[0], "N": c[1]} for i, c in enumerate(C.CELLS)], "easy")
    st = R.success_stats([{"recv": "a", "cat": "new", "base_new": False, "true": False}])
    fs = R.follow_stats(follow_rows(1.0))
    samples = [C.pool_line(eps, 1, set())[0], R.success_line("motion", "fast", st),
               R.follow_line("motion", fs, {"n": 0})]
    for pat, line in zip(LINE_FORMATS, samples):
        assert re.match(pat, line), line


# ---------------------------------------------------------------- 审查非阻断项修补
def test_hold_step_upper_is_press_minus_one(monkeypatch):
    monkeypatch.setattr(C, "reach_step", lambda rows, button: 200)     # 构造就位晚于按下的极端情况
    th, src, press, _ = C.hold_step(fake_rows(300, press_at=150), BUTTON)
    assert (th, src, press) == (149, "reach", 150)
    acts = C.replay_actions(fake_rows(300, press_at=150), th)
    assert acts[-1] == [148.0] * 8                                     # 不含产生按下的那一步（步前值 149）
    monkeypatch.setattr(C, "reach_step", lambda rows, button: None)
    th, _, _, _ = C.hold_step(fake_rows(300), BUTTON)
    assert th == 301                                                   # 未按下时上界仍为最后一步


def test_replay_line_zero_windows_fail():
    ok = [{"ep": 0, "windows": 2, "maxabs": 1e-6, "missing": []}, {"ep": 1, "windows": 0, "maxabs": 0.0, "missing": []}]
    line = C.replay_line("motion", ok)
    assert line.startswith("REPLAY_EQ=PASS model=motion eps=2 windows=2") and line.endswith("zero_window_eps=1")
    zero = [{"ep": 0, "windows": 0, "maxabs": 0.0, "missing": []}]
    assert C.replay_line("motion", zero).startswith("REPLAY_EQ=FAIL")
    assert C.replay_line("motion", zero).endswith("zero_window_eps=1")
    assert C.replay_line("motion", []).startswith("REPLAY_EQ=FAIL")
    big = [{"ep": 0, "windows": 3, "maxabs": 2e-4, "missing": []}]
    assert C.replay_line("motion", big).startswith("REPLAY_EQ=FAIL")


def test_scope_no_last_frame_is_bad():
    syn = C.syn_ep(1, 0)
    r = {"syn": syn, "last_frame": None,
         "motion_content": {"mode": "swap_exact", "donor": f"StopCube:{syn}", "n_windows": 0, "n_replaced": 0,
                            "replace_log": []}}
    assert R.scope_check(r) == (0, "no_last_frame")
    r["last_frame"] = 70
    assert R.scope_check(r) == (2, None)                               # 起点 0、32 未替换
    r["motion_content"]["replace_log"] = [[0, 0], [32, 32]]
    assert R.scope_check(r) == (0, None)


def test_slot_items_missing_gate(tmp_path, capsys):
    (tmp_path / C.CFG_DIR).mkdir()
    for slot in C.SLOTS:
        eps = {f"StopCube:{r}": {} for r in (0, 1, 2) if not (slot == 1 and r == 2)}
        (tmp_path / C.CFG_DIR / f"slot{slot}.json").write_text(json.dumps({"schema_version": 1, "episodes": eps}))
    items, missing = C.slot_items(tmp_path, [0, 1, 2], {}, "cfg")
    assert len(items) == 8 and missing == [(1, 2)]
    assert C.missing_gate(missing, False) is False
    assert "SCS_SPECS_MISSING=1" in capsys.readouterr().out
    assert C.missing_gate(missing, True) is True
    assert C.missing_gate([], False) is True
    items, missing = C.slot_items(tmp_path, [0, 1], {}, "donors")     # iv：donors_sc 为空 → 全缺
    assert items == [] and len(missing) == 6


def test_hold_script_ignores_skipped_old_logs(tmp_path):
    import subprocess
    logs = tmp_path / "logs"
    logs.mkdir()
    (tmp_path / "queues").mkdir()
    for w in ("g0a", "g0b"):
        (tmp_path / "queues" / f"iv-{w}.sh").write_text("run_spec() { :; }\nrun_spec iv x y\necho ran\n")
    (logs / "scs-motion-iv-g0a.log").write_text("old\nEXIT_CODE=1\n")   # 旧日志：被跳过，不得计入 rc
    text = C.HOLD_TEMPLATE.format(hold=0, stage="iv", stagger=0, logs=logs, prefix="scs-motion", root=tmp_path,
                                  workers="g0a g0b")
    script = tmp_path / "hold.sh"
    script.write_text(text)
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "SKIP_LOG_EXISTS" in out.stdout and "channels=1 rc=0" in out.stdout
    assert (logs / "scs-motion-iv-g0b.log").read_text().endswith("EXIT_CODE=0\n")


# ---------------------------------------------------------------- 实施中调整：干预回合关超时
def test_iv_stage_timeout_off_configs(tmp_path):
    eps = {i: {"task_name": "StopCube", "episode_id": i, "ep": i, "env_seed": 2500000 + i, "M": (60, 80, 120)[i % 3],
               "N": 2 + i % 4, "difficulty": "easy"} for i in range(6)}
    roots = {"aspen": {k: f"/a/{k}" for k in C.ASPEN}, "gl": None}
    b = C.SCBuilder(tmp_path, {"aspen": dict(C.MODELS["motion"])}, roots, C.make_workers(0, 2), 0.3, tmp_path, {"m": "x"})
    items = [(s, r) for s in C.SLOTS for r in eps]
    cfgs = C.build_slot_stage(b, "iv", items, eps)
    assert sorted(p.name for p in cfgs) == ["slot0.json", "slot1.json", "slot2.json"]
    for path, payload in cfgs.items():
        slot = int(path.stem[4:])
        assert path.parent == tmp_path / C.IV_CFG_DIR and len(payload["episodes"]) == len(eps)
        for r, e in eps.items():
            entry = payload["episodes"][f"StopCube:{r}"]
            assert entry == {"env_seed": e["env_seed"], "orig_move_interval": e["M"], "target_pass": e["N"],
                             "stop_time_override": 99,
                             "horizon": max(e["M"], C.SLOTS[slot]) * e["N"] + 32}
            assert entry["stop_time_override"] * e["M"] >= entry["horizon"]
    for name, (spec, *_rest) in b.specs.items():
        slot = int(name.split("-")[1][1:])
        assert spec["stopcube_mode"] == "normal" and spec["stopcube_config"] == str(tmp_path / C.IV_CFG_DIR / f"slot{slot}.json")
    assert C.build_slot_stage(C.SCBuilder(tmp_path, {"aspen": dict(C.MODELS["motion"])}, roots, C.make_workers(0, 2),
                                          0.3, tmp_path, {"m": "x"}), "donor", items, eps) == {}
    assert C.timeout_off_entry(eps[0], 120)["horizon"] == C.horizon(60, 120, 2) == 272


def test_existing_run_dirs(tmp_path):
    (tmp_path / "runs" / "iv-s0-w00-00").mkdir(parents=True)
    assert C.existing_run_dirs(tmp_path, ["iv-s0-w00-00", "iv-s1-w00-00"]) == ["iv-s0-w00-00"]
    assert C.existing_run_dirs(tmp_path / "none", ["a"]) == []


def test_late_stats_and_sym_press():
    rows = [{"press": 130, "M": 60, "N": 2, "timeout_off": True},     # > M·N
            {"press": 310, "M": 60, "N": 2, "timeout_off": True},     # > M·N 且 > 5·M
            {"press": 100, "M": 60, "N": 2, "timeout_off": False},
            {"press": None, "M": 60, "N": 2}]
    assert R.late_stats(rows) == {"runs": 4, "late_after_MN": 2, "after_cube_stop": 1, "timeout_off": 2}
    assert R.late_line("motion", "slow", R.late_stats(rows)) == \
        "SPEED_LATE model=motion group=slow runs=4 late_after_MN=2 after_cube_stop=1 timeout_off=2/4"
    assert R.sym_press(120, 60, 2) == 120 and R.sym_press(121, 60, 2) is None and R.sym_press(None, 60, 2) is None


def test_analyze_sym_and_late(monkeypatch):
    """R 完全按 token：slow 组按下步全部晚于 M·N → 对称口径只剩 fast 组。"""
    by_ep, runs = {}, []
    for i, (M, N) in enumerate(((60, 2), (80, 3), (120, 2))):
        by_ep[i] = {"ep": i, "M": M, "N": N}
        T = press_step(M, N)
        common = {"ep": i, "recv": f"StopCube:{i}", "status": "ok", "M": M, "N": N, "move_interval": M,
                  "target_pass": N, "last_frame": None}
        runs.append(dict(common, group="N", press=T + 2, success=True, outcome="success", slot=None))
        for slot, Mp in C.SLOTS.items():
            runs.append(dict(common, group="D", slot=slot, M_prime=Mp, outcome=R.DONOR_OUTCOME, press=None))
            runs.append(dict(common, group="R", slot=slot, M_prime=Mp, syn=C.syn_ep(i, slot),
                             kind=C.speed_group(M, Mp), press=press_step(Mp, N) + 2, success=(Mp == M),
                             outcome="success" if Mp == M else "fail", timeout_off=(i != 2), motion_content=None))
    monkeypatch.setattr(R, "donor_window", lambda d, N: ((press_step(d["M_prime"], N) - 10,
                                                          press_step(d["M_prime"], N) + 10), None))
    lines, detail = R.analyze({"model": "motion"}, by_ep, runs)
    text = "\n".join(lines)
    assert re.search(r"^SPEED_FOLLOW model=motion beta=1\.000 .* n=6$", text, re.M)
    assert re.search(r"^SPEED_FOLLOW_SYM model=motion beta=1\.000 ci=\[\S+\] beta0=1\.000 same_delta=0\.0 "
                     r"ci=\[\S+\] n=3$", text, re.M)
    assert "SPEED_NEWWIN_LATE model=motion group=slow w_new_after_MN=2/3" in text
    assert "SPEED_LATE model=motion group=slow runs=3 late_after_MN=3 after_cube_stop=0 timeout_off=3/3" in text
    assert "SPEED_LATE model=motion group=fast runs=3 late_after_MN=0 after_cube_stop=0 timeout_off=1/3" in text
    assert "SPEED_LATE model=motion group=same runs=3 late_after_MN=0 after_cube_stop=0 timeout_off=2/3" in text
    assert detail["r_timeout_off"]["StopCube:2:s0"] is False and detail["r_timeout_off"]["StopCube:0:s2"] is True


def test_load_run_press_capped_at_horizon(monkeypatch):
    rows = fake_rows(400, press_at=300)
    monkeypatch.setattr(C, "find_sc_run", lambda row: {"terminal": {"outcome": "timeout_off_horizon"},
                                                       "motion_content": None, "donor_receipt": None,
                                                       "timeout_receipt": {"horizon": 152}, "trace": "x"})
    monkeypatch.setattr(C, "load_trace", lambda path: {"rows": rows})
    row = {"stage": "iv", "group": "R", "ep": 0, "M": 60, "N": 2, "slot": 1, "M_prime": 80, "syn": C.syn_ep(0, 1)}
    out = R.load_run(row)                      # horizon = max(60,80)·2+32 = 192 < 300 → horizon 内未按
    assert out["status"] == "ok" and out["press"] is None and out["timeout_off"] is True
    rows[:] = fake_rows(400, press_at=180)
    assert R.load_run(row)["press"] == 180
    monkeypatch.setattr(C, "find_sc_run", lambda row: {"terminal": {"outcome": "fail"}, "motion_content": None,
                                                       "donor_receipt": None, "trace": "x"})
    assert R.load_run(row)["timeout_off"] is False          # 旧口径（未关超时）冒烟没有回执
