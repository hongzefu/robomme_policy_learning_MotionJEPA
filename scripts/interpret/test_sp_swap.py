"""sp_swap_prep 与 sp_swap_readout 的 CPU 测试：合成扫描、局池、终局、回执、轨迹与录制，不起模型与仿真。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.interpret import sp_swap_prep as S
from scripts.interpret.analysis import sp_swap_readout as R

DIM = 4
COMMIT = "c" * 40


# ---------------------------------------------------------------- 分段与窗口
def test_segments_contiguous_and_rs_len():
    labels = [1] * 50 + [2] * 50 + [3]
    segs, why = S.segments_from_labels(labels, 100, 2)
    assert why is None and segs == [(0, 49), (50, 99)]
    assert S.segment_ok("RouteStick", segs, 100) is None
    segs2, _ = S.segments_from_labels([1] * 40 + [2] * 60 + [3], 100, 2)
    assert S.segment_ok("RouteStick", segs2, 100) == "rs_len"
    assert S.segment_ok("PatternLock", segs2, 100) is None
    assert S.segments_from_labels([1] * 10 + [2] * 10 + [1] * 5, 25, 2)[0] is None       # 不连续
    assert S.segments_from_labels([1] * 25, 25, 2)[0] is None                              # 段数 ≠ m
    assert S.segments_from_labels([0] + [1] * 24, 25, 1)[0] is None                        # 序号须从 1 起


def test_window_sets_all_and_core():
    assert S.demo_starts(100) == [0, 32, 64]
    assert S.window_sets((0, 49), 100) == ([0, 32], [0])
    assert S.window_sets((50, 99), 100) == ([32, 64], [64])
    # 尾窗真实末帧 es−1：es=82 时起点 64 真实帧 [64, 81]；es=80 时 64 不足 17 帧真实帧、不成窗
    assert S.real_range(64, 82) == (64, 81) and S.demo_starts(80) == [0, 32]
    assert S.window_sets((60, 81), 82) == ([32, 64], [64])
    # PL 短段：一个窗都不完全落在段内
    assert S.window_sets((20, 40), 90) == ([0, 32], [])


# ---------------------------------------------------------------- 几何与可执行性
def test_pl_geometry_and_executable():
    e = {"task": "PatternLock", "cells": [0, 4, 8], "grid": 3, "m": 2}
    sgs = S.subgoals(e)
    assert sgs[0] == {"start": 0, "target": 4, "dir": [1, 1]} and sgs[1]["dir"] == [1, 1]
    assert S.pl_step(8, (1, 0), 3) is None and S.pl_step(4, (-1, 0), 3) == 1
    donor = {"start": 0, "target": 3, "dir": [1, 0]}
    assert S.subgoal_differs("PatternLock", sgs[1], donor)
    assert S.donor_dir_target("PatternLock", e, 2, donor) == 7          # 从第 2 步起点 4 向下
    assert S.donor_dir_target("PatternLock", e, 2, {"dir": [-1, -1]}) == 0
    e2 = {"task": "PatternLock", "cells": [3, 6], "grid": 3, "m": 1}
    assert S.donor_dir_target("PatternLock", e2, 1, {"dir": [0, -1]}) is None   # 出盘不可执行
    assert S.pl_xy(0, 3) == pytest.approx((-0.2, -0.1)) and S.pl_xy(12, 5) == pytest.approx((-0.1, 0.0))


def test_rs_geometry_and_differs():
    e = {"task": "RouteStick", "cells": [4, 6, 4], "swings": ["cw", "ccw"], "theta": 0.0, "m": 2}
    sgs = S.subgoals(e)
    assert sgs[0] == {"start": 4, "target": 6, "side": 1, "swing": "cw"} and sgs[1]["side"] == -1
    assert S.subgoal_differs("RouteStick", sgs[0], {"side": 1, "swing": "ccw"})
    assert not S.subgoal_differs("RouteStick", sgs[0], {"side": 1, "swing": "cw"})
    assert S.rs_target(8, 1) is None and S.rs_target(0, 1) == 2
    assert S.donor_dir_target("RouteStick", e, 2, {"side": 1}) == 8
    assert S.rs_xy(4, 0.0) == pytest.approx((-0.1, 0.0))


# ---------------------------------------------------------------- 对齐、合成键、装箱
def test_aligned_rows_pad_and_missing():
    rec = {s: np.full(DIM, s, np.float32) for s in (0, 32, 64, 96)}
    embs, pad, srcs, why = S.aligned_rows([32, 64], [0], rec, 100)
    assert why is None and pad == 1 and srcs == [0, 32] and embs[:, 0].tolist() == [0, 32]
    embs, pad, srcs, why = S.aligned_rows([0], [32, 64], rec, 100)
    assert pad == 0 and srcs == [32]                                                     # 多出的不用
    assert S.aligned_rows([0, 32], [64], {64: rec[64]}, 100)[3].startswith("donor_missing")


def test_synthetic_keys_unique_and_roundtrip():
    keys = set()
    for recv in range(120):
        for j in range(1, 8):
            for variant in ("all", "core"):
                for g in S.SWAP_GROUPS:
                    k = S.synthetic_ep(recv, j, g, variant)
                    assert k not in keys
                    keys.add(k)
                    assert S.parse_synthetic(k) == (recv, j, g, variant)
    assert S.synthetic_ep(3, 2, "Bsame") == 10000000 + 300 + 21


def test_pack_runs_same_j_group():
    runs = [{"recv": f"PatternLock:{i % 5}", "j": 1 + i % 2, "group": "Z" if i % 3 else "Bsame"} for i in range(40)]
    bins = S.pack_runs(runs, lambda r: (r["j"], r["group"]), cap=4)
    total = 0
    for key, members in bins:
        assert len(members) <= 4 and len({m["recv"] for m in members}) == len(members)
        assert all((m["j"], m["group"]) == key for m in members)
        total += len(members)
    assert total == 40


# ---------------------------------------------------------------- 逐步复原
def pl_trace(e, plan_cells, wrong=None, swing=None):
    """合成执行段 step 行：逐格移动到 plan_cells；wrong 给出时最后一段去错格且不跳变。"""
    m = e["m"]
    rows, ti = [], m + 3
    pos = S.pl_xy(e["cells"][0], e["grid"]) if e["task"] == "PatternLock" else S.rs_xy(e["cells"][0], e["theta"])
    targets = list(plan_cells) + ([wrong] if wrong is not None else [])
    for k, cell in enumerate(targets):
        xy = S.pl_xy(cell, e["grid"]) if e["task"] == "PatternLock" else S.rs_xy(cell, e["theta"])
        for t in (0.3, 0.6, 1.0):
            p = [pos[0] + (xy[0] - pos[0]) * t, pos[1] + (xy[1] - pos[1]) * t]
            if swing is not None and t != 1.0:
                p[0] += 0.01 * (1 if swing[k] == "ccw" else -1)
            before = ti
            if t == 1.0 and not (wrong is not None and k == len(targets) - 1):
                ti += 1
            rows.append({"event": "step", "tcp_after": [p[0], p[1], 0.1], "task_index_before": before,
                         "task_index_after": ti})
        pos = xy
    return rows


def test_reconstruct_pl_success_and_wrong():
    e = {"task": "PatternLock", "cells": [0, 1, 2], "grid": 3, "m": 2}
    steps = S.reconstruct("PatternLock", e, pl_trace(e, [1, 2]))
    assert [(s["reached"], s["completed"], s["how"]) for s in steps] == [(1, True, "jump"), (2, True, "jump")]
    steps = S.reconstruct("PatternLock", e, pl_trace(e, [1], wrong=5))
    assert [(s["reached"], s["completed"]) for s in steps] == [(1, True), (5, False)]
    assert S.completed_steps(steps) == 1
    steps = S.reconstruct("PatternLock", e, pl_trace(e, [], wrong=3))
    assert len(steps) == 1 and steps[0]["reached"] == 3 and not steps[0]["completed"]


def test_reconstruct_rs_swing():
    e = {"task": "RouteStick", "cells": [4, 6, 4], "swings": ["cw", "ccw"], "theta": 0.0, "m": 2}
    steps = S.reconstruct("RouteStick", e, pl_trace(e, [6, 4], swing=["cw", "ccw"]))
    assert [s["reached"] for s in steps] == [6, 4] and all(s["completed"] for s in steps)
    assert steps[0]["swing"] == "cw"
    # 绕向按平均叉积：偏 −x 的点在 +y 方向连线上为 cw
    assert S.swing_side([[-0.11, 0.07]], (-0.1, 0.0), (-0.1, 0.14)) == "cw"
    assert S.swing_side([[-0.09, 0.07]], (-0.1, 0.0), (-0.1, 0.14)) == "ccw"


# ---------------------------------------------------------------- 矩阵与重采样
def _row(recv, j, group, follow, base, correct=True, keep=None, later=None, m=3, n_correct=True):
    return {"recv": recv, "task": "PatternLock", "m": m, "j": j, "group": group, "variant": "all", "valid": True,
            "follow": follow, "base_follow": base, "correct": correct, "n_correct": n_correct, "cell_cat": "own",
            "swing_cat": None, "keep": keep or {}, "later": later or {}, "clean": True}


def test_matrix_and_follow_stats():
    rows = [_row("a", 2, "Bsame", True, False, correct=False, keep={1: True}),
            _row("b", 2, "Bsame", False, False, correct=True, keep={1: False}, later={3: True}),
            _row("a", 1, "Bsame", True, True, correct=False),
            _row("c", 1, "Z", False, False, correct=True, later={2: True, 3: False})]
    M = R.matrix(rows, "PatternLock", "Bsame")
    assert M[2][1] == [1, 2] and M[2][2] == [1, 2] and M[2][3] == [1, 1] and M[1][1] == [1, 1]
    Z = R.matrix(rows, "PatternLock", "Z")
    assert Z[1][1] == [1, 1] and Z[1][2] == [1, 1] and Z[1][3] == [0, 1]
    st = R.follow_stats([r for r in rows if r["group"] == "Bsame"])
    assert st["n"] == 3 and st["follow"] == pytest.approx(2 / 3) and st["base"] == pytest.approx(1 / 3)
    assert st["diff"] == pytest.approx(1 / 3) and st["ci"][0] <= st["diff"] <= st["ci"][1]


def test_bootstrap_by_recv_deterministic():
    rows = [_row(f"r{i}", 1, "Bsame", i % 2 == 0, False) for i in range(20)]
    a = R.bootstrap_diff(rows, "follow", "base_follow")
    b = R.bootstrap_diff(rows, "follow", "base_follow")
    assert a == b and a[0] < 0.5 < a[1]
    same = [_row("x", 1, "Bsame", True, False), _row("x", 2, "Bsame", True, False)]
    assert R.bootstrap_diff(same, "follow", "base_follow") == [1.0, 1.0]   # 只有一个接收局


def test_step_view_follow_rs():
    e = {"task": "RouteStick", "cells": [4, 6], "swings": ["cw"], "theta": 0.0, "m": 1}
    de = {"task": "RouteStick", "cells": [2, 4], "swings": ["ccw"], "theta": 0.0, "m": 1}
    steps = [{"i": 1, "start": 4, "reached": 6, "completed": False, "swing": "ccw", "how": "geom"}]
    v = R.step_view("RouteStick", steps, e, 1, de, 1)
    assert v["follow"] and v["cell_cat"] == "own" and v["swing_cat"] == "donor_swing"
    assert R.step_view("RouteStick", [], e, 1, de, 1)["cell_cat"] == "unreached"


# ---------------------------------------------------------------- 路径根
def _args(**kw):
    import argparse
    base = {k: None for k in S.ASPEN}
    base.update(gl_jobs=0, gl_per_job=2, gl_release=None, gl_store=None, gl_bench_src=None, gl_sim_python=None,
                gl_policy_python=None, gl_stage2_python=None, gl_mmevla_store=None, gl_log_dir=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_site_roots_and_remap():
    r = S.site_roots(_args(store="/x/store"))
    assert r["aspen"]["store"] == "/x/store" and r["gl"] is None
    with pytest.raises(SystemExit):
        S.site_roots(_args(gl_jobs=8))
    r = S.site_roots(_args(gl_jobs=8, gl_release="/gl/rel"))
    assert r["gl"]["store"] == S.GL_STORE and r["gl"]["sim_python"] == f"{S.GL_STORE}/envs/official-sim/bin/python"
    assert r["gl"]["stage2_python"].endswith("envs/stage2/bin/python")
    assert r["gl"]["bench_src"] == "/gl/rel/third_party/robomme_benchmark/src"
    model = {"checkpoint": f"{S.ASPEN['store']}/assets/w/59999", "policy_config": "p",
             "target_contract": f"{S.ASPEN['store']}/evaluation/c.json", "extra": {"x": f"{S.MAIN}/a/b", "n": 3}}
    g = S.gl_model(model, r["aspen"], r["gl"], gl_checkpoint="/gl/ckpt")
    assert g["checkpoint"] == "/gl/ckpt" and g["target_contract"] == f"{S.GL_STORE}/evaluation/c.json"
    assert g["extra"] == {"x": "/gl/rel/a/b", "n": 3} and g["policy_config"] == "p"
    assert S.remap("/data/abc", ["/data/ab"], ["/z"]) == "/data/abc"                # 只按目录边界换
    w = S.make_workers(8, 2)
    assert len(w) == 20 and w[4] == ("g0a", "gl", "0", 0) and w[-1] == ("g7b", "gl", "0", 7)


# ---------------------------------------------------------------- 端到端：pool → specs → donors → readout
def _scan_entries(task):
    out, i = [], 0
    for m, q in S.QUOTA[task].items():
        for k in range(q + 2):
            seed = S.SEED0[task] + 100 * i
            diff = "easy" if m <= 2 else ("medium" if m <= 4 else "hard")
            if task == "PatternLock":
                grid = S.pl_grid(diff)
                cells = [c % (grid * grid) for c in range(m + 1)]
                if k % 2:
                    cells = cells[::-1]
                es = 30 * m
                segs = [[30 * (j - 1), 30 * j - 1] for j in range(1, m + 1)]
                e = {"cells": cells, "grid": grid}
            else:
                cells = [S.RS_ROUTE[2 + (t % 2)] for t in range(m + 1)]
                swings = ["cw" if (t + k) % 2 else "ccw" for t in range(m)]
                es = 50 * m
                segs = [[50 * (j - 1), 50 * j - 1] for j in range(1, m + 1)]
                e = {"cells": cells, "swings": swings, "theta": 0.0}
            e.update(task=task, env_seed=seed, difficulty=diff, scan_index=i, m=m, status="ok", es=es, segments=segs,
                     segment_problem=None, replay_match=True)
            out.append(e)
            i += 1
    return out


def _write_n_run(row, e, outcome="success", completed=None):
    """合成一局 N 的终局、回执、轨迹（执行段全对）与录制。"""
    key = S.episode_key(row["task"], row["ep"])
    nonce = f"n{row['task'][:2]}{row['ep']:04d}"
    run_root = Path(row["run_root"])
    acc = run_root / "accepted"
    acc.mkdir(parents=True, exist_ok=True)
    (acc / f"{key}.{nonce[:8]}.terminal.json").write_text(json.dumps({"episode_nonce": nonce, "outcome": outcome}))
    tdir = run_root / "attempts/a0/client/traces" / nonce
    (tdir / "meta").mkdir(parents=True, exist_ok=True)
    m = e["m"]
    labels = [j for j in range(1, m + 1) for _ in range(e["segments"][j - 1][1] - e["segments"][j - 1][0] + 1)] + [m + 1]
    (tdir / "meta/hold_receipt.json").write_text(json.dumps({"m": m, "demo_task_index": labels,
                                                             "exec_start_idx": e["es"]}))
    plan_cells = e["cells"][1:] if completed is None else e["cells"][1:completed + 1]
    with (tdir / "step_trace.jsonl").open("w") as f:
        for r in pl_trace(e, plan_cells):
            f.write(json.dumps(r) + "\n")
    rec = Path(row["record_dir"]) / S.record_name(row["task"], row["ep"])
    rec.parent.mkdir(parents=True, exist_ok=True)
    starts = S.demo_starts(e["es"])
    rng = np.random.default_rng(row["ep"] + 7 * len(row["task"]))
    np.savez(rec, start_frames=np.asarray(starts, np.int32), embs=rng.standard_normal((len(starts), DIM)).astype(np.float32),
             counts=np.ones(len(starts), np.int32))


def _write_swap_run(row, e, donor_ok=True):
    key = S.episode_key(row["task"], row["ep"])
    nonce = f"s{row['task'][:2]}{row['ep']:04d}{row['j']}{row['group'][:2]}"
    run_root = Path(row["run_root"])
    acc = run_root / "accepted"
    acc.mkdir(parents=True, exist_ok=True)
    (acc / f"{key}.{nonce[:8]}.terminal.json").write_text(json.dumps({"episode_nonce": nonce, "outcome": "success"}))
    tdir = run_root / "attempts/a0/client/traces" / nonce
    (tdir / "meta").mkdir(parents=True, exist_ok=True)
    (tdir / "meta/hold_receipt.json").write_text(json.dumps({"m": e["m"]}))
    with (tdir / "step_trace.jsonl").open("w") as f:
        for r in pl_trace(e, e["cells"][1:]):
            f.write(json.dumps(r) + "\n")
    ep_dir = Path(row["attn_out"]) / nonce
    ep_dir.mkdir(parents=True, exist_ok=True)
    log = [[s, s] for s in row["w_starts"]]
    mc = {"mode": "swap_exact", "donor": S.episode_key(row["task"], row["donor"]), "replace_log": log,
          "n_replaced": len(log)}
    (ep_dir / "episode.json").write_text(json.dumps({"motion_content": mc}))


@pytest.fixture()
def built(tmp_path, monkeypatch):
    O = tmp_path / "O"
    O.mkdir()
    for task in S.TASKS:
        S.P._write_json(O / f"scan_{task}.json", {"task": task, "entries": _scan_entries(task)})
    assert S.main(["pool", "--out", str(O)]) == 0
    model_dir = O / "motion"
    rc = S.main(["specs", "--out", str(model_dir), "--model", "motion", "--commit", COMMIT,
                 "--gl-jobs", "2", "--gl-release", "/gl/release/abc", "--gl-checkpoint", "/gl/ckpt/59999",
                 "--log-dir", str(tmp_path / "logs")])
    assert rc == 0
    return O, model_dir


def test_pool_and_specs(built):
    O, model_dir = built
    pool = json.loads((O / "pool.json").read_text())
    assert len(pool["episodes"]) == 240
    pl = [e for e in pool["episodes"] if e["task_name"] == "PatternLock"]
    counts = {}
    for e in pl:
        counts[e["m"]] = counts.get(e["m"], 0) + 1
    assert counts == S.QUOTA["PatternLock"] and [e["episode_id"] for e in pl] == list(range(120))
    plan = json.loads((model_dir / "plan.json").read_text())
    n_rows = [r for r in plan["runs"] if r["stage"] == "n"]
    assert len(n_rows) == 240 and len({r["worker"] for r in n_rows}) == 8
    gl_spec = next(r for r in n_rows if r["site"] == "gl")["spec"]
    spec = json.loads((model_dir / "specs" / f"{gl_spec}.json").read_text())
    assert spec["repo"] == "/gl/release/abc" and spec["checkpoint"] == "/gl/ckpt/59999"
    assert spec["sim_python"] == f"{S.GL_STORE}/envs/official-sim/bin/python"
    assert spec["subgoal_trace"] and spec["profile"] == "plfull" and spec["model_seed"] == 42 and spec["cuda"] == "0"
    assert spec["record_motion"] == str(model_dir / "record")
    as_spec = json.loads((model_dir / "specs" / "n-w10-00.json").read_text())
    assert as_spec["repo"] == S.ASPEN["repo"] and as_spec["cuda"] == "1"
    env_gl = (model_dir / "env-gl.sh").read_text()
    assert f"MMEVLA_V1_STORE={S.GL_STORE}" in env_gl and "SPS_M=" in env_gl and "\nO=" not in env_gl
    gl_launch = (model_dir / "gl_launch_stage.sh").read_text()
    body = "\n".join(l for l in gl_launch.splitlines() if not l.startswith("#"))
    assert "--gpu_cmode=shared" in body and "--qos" not in body and "g1b:1" in body
    assert (model_dir / "queues" / "n-g0a.sh").read_text().count("run_spec ") >= 1
    assert any(r["stage"] == "glsmoke" and r["worker"] == "g0a" for r in plan["runs"])


def test_donors_and_readout(built):
    O, model_dir = built
    pool, by_key = S.load_pool(O)
    plan = json.loads((model_dir / "plan.json").read_text())
    pick = {}
    for task in S.TASKS:
        mine = [e for e in pool["episodes"] if e["task_name"] == task and e["m"] in (2, 3)][:6]
        for e in mine:
            pick[S.episode_key(task, e["episode_id"])] = e
    for row in plan["runs"]:
        key = S.episode_key(row["task"], row["ep"])
        if row["stage"] == "n" and key in pick:
            _write_n_run(row, by_key[key])
    assert S.main(["donors", "--out", str(model_dir)]) == 0
    plan = json.loads((model_dir / "plan.json").read_text())
    swaps = [r for r in plan["runs"] if r["stage"] in ("swap", "cross")]
    assert swaps
    groups = {r["group"] for r in swaps}
    assert {"Z", "Bsame"} <= groups
    for r in swaps:
        f = model_dir / S.DONOR_DIR / S.record_name(r["task"], r["donor"])
        with np.load(f) as z:
            assert z["start_frames"].tolist() == r["w_starts"]
            if r["group"] == "Z":
                assert float(np.abs(z["embs"]).max()) == 0.0
        if r["group"] != "Z":
            rsg = S.subgoals(by_key[S.episode_key(r["task"], r["ep"])])[r["j"] - 1]
            dsg = S.subgoals(by_key[r["donor_src"]])[r["donor_j"] - 1]
            assert S.subgoal_differs(r["task"], rsg, dsg)
            if r["group"] == "Bsame":
                assert r["donor_j"] == r["j"]
            else:
                assert r["donor_j"] != r["j"]
    # spec 内同一 (j, 组)
    for name in {r["spec"] for r in swaps}:
        mine = [r for r in swaps if r["spec"] == name]
        assert len({(r["j"], r["group"]) for r in mine}) == 1
        assert len({r["ep"] for r in mine if r["task"] == mine[0]["task"]}) == len([r for r in mine if r["task"] == mine[0]["task"]])
    for r in swaps:
        _write_swap_run(r, by_key[S.episode_key(r["task"], r["ep"])])
    R.main(["--root", str(model_dir)])
    out = json.loads((model_dir / "readout" / "readout.json").read_text())
    lines = out["lines"]
    assert any(l.startswith("SP_REPLAY=PASS") for l in lines)
    assert any(l.startswith("SWAP_SCOPE=PASS") for l in lines)
    assert any(l.startswith("SP_SEGMENT=PASS") and "PatternLock" in l for l in lines)
    mline = next(l for l in lines if l.startswith("SP_MATRIX") and "task=PatternLock" in l)
    assert "invalid=0" in mline
    zstats = out["summaries"]["PatternLock"]["groups"]["Z"]
    assert zstats["diag"][0] == zstats["diag"][1] > 0          # 合成运行全对
    assert (model_dir / "readout" / "matrix_PatternLock_Bsame.png").is_file()
