"""rs_full_prep 与 rs_full_readout 的 CPU 测试：合成局池、录制 npz、终局、回执与轨迹，不起模型与仿真。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.evaluation import manifest as M
from scripts.interpret import rs_full_prep as P
from scripts.interpret.analysis import rs_full_readout as R

DIM = 6
COMMIT = "b" * 40
THETA = 0.1


# ---------------------------------------------------------------- 类型、几何与配对
def test_sixteen_types():
    types = P.all_types()
    assert len(types) == 16 and len(set(types)) == 16
    assert types[:4] == [(0, 1, "cw"), (0, 1, "ccw"), (1, -1, "cw"), (1, -1, "ccw")]
    assert all(P.target_cell(s, side) is not None for s, side, _ in types)
    assert {(s, side) for s, side, _ in types if s in (0, 4)} == {(0, 1), (4, -1)}   # 端点单向
    assert P.type_label((2, -1, "ccw")) == "2R-ccw" and P.target_cell(2, 1) == 6 and P.target_cell(0, -1) is None


def test_cell_xy_rotation():
    assert P.cell_xy(4, 0.0) == pytest.approx((-0.1, 0.0))
    assert P.cell_xy(8, 0.0) == pytest.approx((-0.1, 0.28))
    x, y = P.cell_xy(8, math.radians(30))
    assert (x, y) == pytest.approx((-0.1 * math.cos(math.radians(30)) - 0.28 * 0.5,
                                    -0.1 * 0.5 + 0.28 * math.cos(math.radians(30))))
    # 下标增大 → y 增大（环境 _stick_side 记 left），|theta|≤30° 时恒成立
    for th in (-math.radians(30), 0.0, math.radians(30)):
        assert P.cell_xy(6, th)[1] > P.cell_xy(4, th)[1]


def test_replay_route_known_seeds():
    pytest.importorskip("torch")
    # 2026-11-09 replay-check 与真实环境对拍通过的两个 seed（含 easy 到端点被迫回头的情形）
    r = P.replay_route(1400000)
    assert r["cells"] == [6, 4, 2] and r["swings"] == ["counterclockwise", "clockwise"] and r["steps"] == 2
    assert P.step1_type(r) == (3, -1, "ccw")
    assert P.replay_route(1400200)["cells"] == [2, 0, 2]
    assert -math.radians(30) <= r["theta"] <= math.radians(30)


def test_pair_counts_and_relations():
    pairs = P.build_pairs()
    assert pairs["counts"] == {"A": 32, "Bsw": 32, "Bside": 48, "C": 96, "D1": 96, "D2": 208}
    assert sum(pairs["counts"].values()) == 32 * 16
    assert pairs["d2_strata"] == {"in": 144, "out": 64}
    for p in pairs["all"]["A"]:
        assert p["recv_type"] == p["donor_type"] and p["donor"] == 32 + p["recv"] // 2
    for p in pairs["all"]["Bsw"]:
        assert p["recv_start"] == p["donor_start"] and p["recv_side"] == p["donor_side"] and p["recv_swing"] != p["donor_swing"]
    for p in pairs["all"]["Bside"]:
        assert p["recv_start"] == p["donor_start"] and p["recv_side"] != p["donor_side"]
        assert p["donor_dir_cell"] == p["donor_abs"] != p["own"]
    for p in pairs["all"]["C"]:
        assert p["recv_start"] != p["donor_start"] and p["donor_dir_cell"] == p["own"] and p["recv_swing"] == p["donor_swing"]
    for p in pairs["all"]["D1"]:
        assert p["recv_side"] == p["donor_side"] and p["recv_swing"] != p["donor_swing"]
    for p in pairs["all"]["D2"]:
        assert p["recv_start"] != p["donor_start"] and p["recv_side"] != p["donor_side"]
        assert (p["stratum"] == "out") == (p["recv_start"] in (0, 4))
    assert len(pairs["w0"]) == 80 and {p["w0_from"] for p in pairs["w0"]} == {"Bsw", "Bside"}
    line, ok = P.pairs_line(pairs)
    assert ok and line == "RS_PAIRS=PASS A=32 Bsw=32 Bside=48 C=96 D1=96 D2=208 W0=80 seedcheck=60"


def test_seedcheck_deterministic_subset():
    a, b = P.build_pairs(), P.build_pairs()
    key = lambda ps: [(p["recv"], p["donor"]) for p in ps]
    for g in P.PAIR_GROUPS:
        assert key(a["seedcheck"][g]) == key(b["seedcheck"][g])
        assert len(a["seedcheck"][g]) == 10 and set(key(a["seedcheck"][g])) <= set(key(a["all"][g]))


def test_multieq_unique_receivers():
    chosen = P.multieq_pairs(P.build_pairs())
    assert len(chosen) == 12 and len({p["recv"] for _, p in chosen}) == 12
    assert [g for g, _ in chosen] == [g for g in P.PAIR_GROUPS for _ in range(2)]


# ---------------------------------------------------------------- 绕向、到达归类
def test_swing_sign_matches_direction_fail():
    prev, curr = (0.0, 0.0), (0.14, 0.0)
    assert R.swing_side([(0.07, 0.05)], prev, curr) == "clockwise"          # 叉积 > 0
    assert R.swing_side([(0.07, -0.05)], prev, curr) == "counterclockwise"
    assert R.swing_side([(0.07, 0.0)], prev, curr) is None                  # 落在连线上
    assert R.swing_side([(0.07, 0.05)], curr, prev) == "counterclockwise"   # 反向连线符号翻转
    assert R.swing_side([], prev, curr) is None


def test_classify_branches():
    assert R.classify_target(None, 4) == "none"
    assert R.classify_target(4, 4, 4, 4) == "own"
    assert R.classify_target(2, 4, 2, 2) == "donor_dir"
    assert R.classify_target(6, 4, None, 6) == "donor_abs"
    assert R.classify_target(8, 4, 2, 6) == "other"
    assert R.classify_swing(None, "cw", "ccw") == "none"
    assert R.classify_swing("cw", "cw", "cw") == "own_swing"
    assert R.classify_swing("ccw", "cw", "ccw") == "donor_swing"
    assert R.classify_swing("ccw", "cw", None) == "other_swing"
    row = {"recv_type": [1, 1, "cw"], "donor_type": [3, -1, "ccw"]}
    assert R.targets(row) == (4, 0, 4, "cw", "ccw")
    assert R.targets({"recv_type": [0, 1, "cw"], "donor_type": [2, -1, "cw"]})[1] is None   # 出盘


def swing_walk(start, cell, swing, theta=THETA):
    """起点 → 中点外偏 0.05 m（cw 偏向连线左法向）→ 目标格；cell == start 时原地不动。"""
    s, t = P.cell_xy(start, theta), P.cell_xy(cell, theta)
    if cell == start:
        return [[*s, 0.05]] * 3
    lx, ly = t[0] - s[0], t[1] - s[1]
    norm = math.hypot(lx, ly)
    k = 0.05 / norm * (1 if swing == "cw" else -1)
    mid = ((s[0] + t[0]) / 2 - ly * k, (s[1] + t[1]) / 2 + lx * k)
    return [[*s, 0.05], [*mid, 0.05], [*t, 0.02], [*t, 0.02]]


def test_first_step_segment_stops_at_first_target():
    pts = swing_walk(2, 4, "cw") + swing_walk(4, 6, "ccw")
    cell, seg = R.first_step_segment(pts, 2, THETA)
    assert cell == 4 and len(seg) == 3                       # 起点格不算到达，到达点含在段内
    assert R.swing_side(seg, P.cell_xy(2, THETA), P.cell_xy(4, THETA)) == "clockwise"
    assert R.first_step_segment([[*P.cell_xy(2, THETA), 0.05]], 2, THETA) == (None, [[*P.cell_xy(2, THETA), 0.05]])
    post = [[*P.cell_xy(3, THETA), 0.05]]                    # 立柱格不是目标
    assert R.first_step_segment(post, 2, THETA)[0] is None


# ---------------------------------------------------------------- 合成局池
def fake_replay(seed, difficulty="easy"):
    """每 20 个 seed：前 16 个依次是 16 种类型的两步局，其余为三步局。"""
    types = P.all_types()
    i = (seed - P.SCAN_START) // P.SCAN_STEP
    if i % 20 < 16:
        s, side, sw = types[i % 20]
        n = s + side
        n2 = n + side if 0 <= n + side < 5 else n - side
        idx = [s, n, n2]
        swings = ["clockwise" if sw == "cw" else "counterclockwise", "clockwise"]
    else:
        idx, swings = [0, 1, 2, 3], ["clockwise"] * 3
    return {"env_seed": seed, "theta": THETA, "steps": len(idx) - 1, "route_idx": idx,
            "cells": [P.ROUTE[j] for j in idx], "swings": swings}


@pytest.fixture
def pool_root(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "replay_route", fake_replay)
    root = tmp_path / "O"
    assert P.main(["pool", "--out", str(root)]) == 0
    return root


def test_pool_files(pool_root, capsys):
    pool = json.loads((pool_root / "pool.json").read_text())
    eps = pool["episodes"]
    assert [e["episode_id"] for e in eps] == list(range(48))
    types = P.all_types()
    for k, t in enumerate(types):
        a, b, s = eps[2 * k], eps[2 * k + 1], eps[32 + k]
        assert {a["type"], b["type"], s["type"]} == {k} and (a["start"], a["side"], a["swing"]) == t
        assert a["env_seed"] < b["env_seed"] < s["env_seed"] and (a["role"], s["role"]) == ("R", "S")
        assert len(a["path"]) == 3
    meta = json.loads((pool_root / "metadata" / P.META_NAME).read_text())
    assert meta["env_id"] == "RouteStick" and meta["record_count"] == 48
    man = M.validate_episodes(json.loads((pool_root / "manifest.json").read_text()), official=False)
    assert man["episodes"][0]["episode_key"] == "RouteStick:0" and man["sha256"] == pool["manifest_seal"]
    capsys.readouterr()
    assert P.main(["pool", "--out", str(pool_root)]) == 0      # 同内容重跑幂等
    out = capsys.readouterr().out
    assert "RS_POOL=PASS eps=48 types=16 overlap=0" in out
    assert "RS_PAIRS=PASS A=32 Bsw=32 Bside=48 C=96 D1=96 D2=208 W0=80 seedcheck=60" in out


def test_pool_refuses_overwrite(pool_root):
    pool = json.loads((pool_root / "pool.json").read_text())
    pool["episodes"][0]["env_seed"] += 1
    (pool_root / "pool.json").write_text(json.dumps(pool))
    with pytest.raises(SystemExit):
        P.main(["pool", "--out", str(pool_root)])


# ---------------------------------------------------------------- specs
@pytest.fixture
def motion(pool_root, capsys):
    out = pool_root / "motion"
    assert P.main(["specs", "--out", str(out), "--model", "motion", "--commit", COMMIT, "--run-tree", "/tmp/tree"]) == 0
    assert "RSF_SPECS model=motion smoke=5 multieq=24 n=48" in capsys.readouterr().out
    return out


def load_specs(out):
    return {p.stem: json.loads(p.read_text()) for p in sorted((out / "specs").glob("*.json"))}


def test_specs_keys(motion, pool_root):
    specs = load_specs(motion)
    plan = json.loads((motion / "plan.json").read_text())
    assert not any(n.startswith("repro") for n in specs) and "repro" not in P.STAGES
    for name, spec in specs.items():
        man = M.validate_episodes(json.loads(Path(spec["manifest"]).read_text()), official=False)
        keys = [e["episode_key"] for e in man["episodes"]]
        assert all(k.startswith("RouteStick:") for k in keys) and len(set(keys)) == len(keys)
        assert spec["profile"] == "plfull" and spec["plfull_metadata_dir"] == str(pool_root / "metadata")
        assert spec["subgoal_trace"] is True and spec["save_attn"] is False and spec["repo"] == "/tmp/tree"
        assert spec["model_kind"] == "stage2_pool4_concat" and Path(spec["run_root"]).parent.name == name
        if "swap_map" in spec:
            assert sorted(spec["swap_map"]) == sorted(keys) and spec["ablate"].startswith("motion_swap_exact:")
        else:
            assert spec["ablate"] == "none"
    assert sorted(r["ep"] for r in plan["runs"] if r["stage"] == "n") == list(range(48))
    assert all(specs[r["spec"]]["record_motion"] == str(motion / "record") for r in plan["runs"] if r["stage"] == "n")
    assert all("theta" in r and "recv_type" in r for r in plan["runs"])
    singles = [s for n, s in specs.items() if n.startswith("multieq-single")]
    assert len(singles) == 12 and {k: v for s in singles for k, v in s["swap_map"].items()} == specs["multieq-multi"]["swap_map"]
    smoke = (motion / "queues" / "smoke-w00.sh").read_text()
    assert smoke.index("smoke-n") < smoke.index("rs_full_prep donors") < smoke.index("smoke-swap") < smoke.index("smoke-zero")
    assert specs["smoke-swap"]["swap_map"] == {"RouteStick:0": "RouteStick:33", "RouteStick:33": "RouteStick:0"}
    launch = (motion / "launch_stage.sh").read_text()
    assert "rsf-motion-$st-$w" in launch and "rs-swap-full-1109" in launch and "EXIT_CODE=" in launch
    assert "repro" not in launch and "smoke|multieq|n|swap|w0|seeds" in launch
    for script in [motion / "launch_stage.sh", motion / "env.sh", *sorted((motion / "queues").glob("*.sh"))]:
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0, script
    assert COMMIT in (motion / "env.sh").read_text()


def test_wan_needs_model_entry(pool_root):
    with pytest.raises(SystemExit):
        P.main(["specs", "--out", str(pool_root / "wan"), "--model", "wan", "--commit", COMMIT])
    assert P.main(["specs", "--out", str(pool_root / "wan"), "--model", "wan", "--commit", COMMIT,
                   "--checkpoint", "/ck", "--policy-config", "cfg", "--model-kind", "wan_fixedpool1024",
                   "--target-contract", "none", "--extra", '{"k": "v"}']) == 0
    spec = json.loads((pool_root / "wan" / "specs" / "n-w00.json").read_text())
    assert spec["model_kind"] == "wan_fixedpool1024" and spec["k"] == "v" and spec["target_contract"] == "none"


# ---------------------------------------------------------------- donors
def emb_of(ep, start):
    return np.arange(DIM, dtype=np.float32) * 0.1 + ep + 0.01 * start + 1.0


def write_record(base, ep, starts=(0, 32, 64), counts=(1, 1, 1)):
    (base / "record").mkdir(parents=True, exist_ok=True)
    np.savez(base / "record" / P.record_name(ep), start_frames=np.asarray(starts, np.int32),
             embs=np.stack([emb_of(ep, s) for s in starts]).astype(np.float32), counts=np.asarray(counts, np.int32))


def write_terminal(row, outcome, nonce):
    run_root = Path(row["run_root"])
    key = f"RouteStick:{row['ep']}"
    (run_root / "accepted").mkdir(parents=True, exist_ok=True)
    (run_root / "accepted" / f"{key}.{nonce[:8]}.terminal.json").write_text(json.dumps(
        {"episode_key": key, "episode_nonce": nonce, "outcome": outcome,
         "budget": {"demo_frames": 100, "demo_windows": 3}}))


def run_donors(motion, success_recv, capsys, extra=()):
    plan = json.loads((motion / "plan.json").read_text())
    for ep in range(48):
        write_record(motion, ep)
    for row in plan["runs"]:
        if row["stage"] == "n":
            ok = row["ep"] >= 32 or row["ep"] in success_recv
            write_terminal(row, "success" if ok else "fail", f"{row['ep']:08d}aaaa")
    code = P.main(["donors", "--out", str(motion), *extra])
    return code, capsys.readouterr().out, json.loads((motion / "plan.json").read_text())


def test_random_rows_same_norm():
    rows = np.stack([emb_of(3, 0), emb_of(3, 32)])
    a, b = P.random_rows(rows, 3), P.random_rows(rows, 3)
    assert np.array_equal(a, b) and a.shape == rows.shape
    assert np.allclose(np.linalg.norm(a, axis=1), np.linalg.norm(rows, axis=1), rtol=1e-6)
    assert not np.allclose(a[0], a[1])


def test_donors_files_and_swap_specs(motion, capsys):
    ok_recv = set(range(0, 32, 2))
    code, out, plan = run_donors(motion, ok_recv, capsys)
    assert code == 0, out
    assert "DONOR_TRIM=PASS donors=16 w0=16 zeros=32 randoms=32 rows=2 maxabs=0" in out
    assert "DONOR_DISTINCT=PASS donors=16 identical_pairs=0" in out
    with np.load(motion / "donor_trim" / P.record_name(40)) as z:
        assert z["start_frames"].tolist() == [0, 32]
        assert np.array_equal(z["embs"], np.stack([emb_of(40, 0), emb_of(40, 32)]))
    with np.load(motion / "donor_w0" / P.record_name(40)) as z:
        assert z["start_frames"].tolist() == [0] and np.array_equal(z["embs"][0], emb_of(40, 0))
    with np.load(motion / "zero" / P.record_name(5)) as z:
        assert z["start_frames"].tolist() == [0, 32] and not z["embs"].any()
    with np.load(motion / "random" / P.record_name(5)) as z:
        assert np.allclose(np.linalg.norm(z["embs"], axis=1),
                           np.linalg.norm(np.stack([emb_of(5, 0), emb_of(5, 32)]), axis=1), rtol=1e-6)
    assert not (motion / "donor_trim" / P.record_name(3)).exists()
    swap = [r for r in plan["runs"] if r["stage"] in ("swap", "w0", "seeds")]
    assert swap and all(r["ep"] in ok_recv for r in swap)
    pairs = P.build_pairs()
    want = {g: sum(1 for p in pairs["all"][g] if p["recv"] in ok_recv) for g in P.PAIR_GROUPS}
    got = {g: sum(1 for r in swap if r["stage"] == "swap" and r["group"] == g) for g in P.PAIR_GROUPS}
    assert got == want
    assert sum(1 for r in swap if r["stage"] == "w0") == sum(1 for p in pairs["w0"] if p["recv"] in ok_recv)
    assert {r["seed"] for r in swap if r["stage"] == "seeds"} == {1, 2}
    specs = load_specs(motion)
    for r in swap:
        spec = specs[r["spec"]]
        assert spec["swap_map"][f"RouteStick:{r['ep']}"] == f"RouteStick:{r['donor']}"
        assert spec["model_seed"] == r["seed"] and spec["profile"] == "plfull"
        assert spec["ablate"] == f"motion_swap_exact:{motion / P.ABLATE_DIR[r['group']]}"
    assert all(r["group"] == "W0" and specs[r["spec"]]["ablate"].endswith("donor_w0") for r in swap if r["stage"] == "w0")
    assert plan["gate"]["few"] is False


def test_donors_wan_gate(motion, capsys):
    code, out, plan = run_donors(motion, {1, 2, 3}, capsys, extra=("--wan-gate", "8"))
    assert code == 0 and "gate=few" in out
    swap = [r for r in plan["runs"] if r["stage"] in ("swap", "w0", "seeds")]
    assert {r["group"] for r in swap} == {"Z", "Rnd"} and len(swap) == 6
    assert plan["gate"]["few"] is True and plan["gate"]["r_success"] == 3


def test_donors_identical_and_missing_row(motion, capsys):
    plan = json.loads((motion / "plan.json").read_text())
    for ep in range(48):
        write_record(motion, ep)
    import shutil
    shutil.copy(motion / "record" / P.record_name(40), motion / "record" / P.record_name(41))
    for row in plan["runs"]:
        if row["stage"] == "n":
            write_terminal(row, "success", f"{row['ep']:08d}bbbb")
    assert P.main(["donors", "--out", str(motion)]) == 1
    out = capsys.readouterr().out
    assert "DONOR_DISTINCT=FAIL donors=16 identical_pairs=1" in out and "DONORS_SPECS=SKIP" in out
    write_record(motion, 41, starts=(0, 64), counts=(1, 1))      # 缺起点 32 行
    assert P.main(["donors", "--out", str(motion)]) == 1
    assert "ep41:start32_rows=0" in capsys.readouterr().out


# ---------------------------------------------------------------- 读数端到端
def synth(row, *, outcome, reach, swing, nonce, mc=None, official=None):
    run_root, attn = Path(row["run_root"]), Path(row["attn_out"])
    key = f"RouteStick:{row['ep']}"
    attempt = run_root / "attempts" / f"000-{nonce[:6]}"
    (attempt / "client").mkdir(parents=True, exist_ok=True)
    (attempt / "commands.json").write_text(json.dumps({"episode_keys": [key]}))
    tdir = attempt / "client" / "traces" / nonce
    (tdir / "meta").mkdir(parents=True)
    pts = swing_walk(row["path"][0], reach, swing, row["theta"])
    steps = [{"event": "step", "tcp_after": p, "task_index_before": 5,
              "task_index_after": 6 if (official and i == len(pts) - 1) else 5} for i, p in enumerate(pts)]
    (tdir / "step_trace.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    (tdir / "meta" / "hold_receipt.json").write_text(json.dumps(
        {"D": 100, "m": 2, "demo_task_index": [1] * 50 + [2] * 50 + [5]}))
    write_terminal(row, outcome, nonce)
    (attn / nonce).mkdir(parents=True, exist_ok=True)
    (attn / nonce / "episode.json").write_text(json.dumps({"episode_nonce": nonce, **({"motion_content": mc} if mc else {})}))


def plan_outcome(row):
    """合成规则：返回 (到达格, 绕向)。"""
    own, ddir, dabs, own_sw, donor_sw = R.targets(row)
    g = row["group"]
    if g in ("N", "Z", "Rnd", "A", "C"):
        return own, own_sw
    if g in ("Bsw", "D1") or (g == "W0" and row.get("w0_from") == "Bsw"):
        return own, donor_sw
    if g == "Bside" or g == "W0":
        return ddir, own_sw
    return (ddir, own_sw) if ddir is not None else (dabs, own_sw)   # D2


def test_readout_end_to_end(motion, capsys):
    code, _, plan = run_donors(motion, set(range(32)), capsys)
    assert code == 0
    for r in [r for r in plan["runs"] if r["stage"] == "n"]:
        for f in (Path(r["run_root"]) / "accepted").glob("*.terminal.json"):
            f.unlink()
    n = 0
    for row in plan["runs"]:
        if row["stage"] not in ("n", "swap", "w0", "multieq"):
            continue
        n += 1
        reach, swing = plan_outcome(row)
        mc = None
        if row["group"] != "N":
            starts = [0] if row["group"] == "W0" else [0, 32]
            mc = {"mode": "swap_exact", "donor": f"RouteStick:{row['donor']}",
                  "replace_log": [[s, s] for s in starts], "n_replaced": len(starts)}
        ok1 = reach == R.targets(row)[0] and swing == row["recv_type"][2]
        synth(row, outcome="success" if ok1 else "fail", reach=reach, swing=swing, nonce=f"{n:032x}", mc=mc, official=ok1)
    capsys.readouterr()
    assert R.main(["--root", str(motion)]) == 0
    out = capsys.readouterr().out
    pairs = P.build_pairs()
    d2 = pairs["all"]["D2"]
    n_abs = sum(1 for p in d2 if p["stratum"] == "out" and p["donor_abs"] not in (p["own"], P.ROUTE[p["recv_start"]]))
    assert "RS_DEMO_LAYOUT=PASS model=motion eps=48 bad=0" in out
    assert "MULTI_SWAP_EQ=PASS runs=12 differ=0" in out
    assert "SWAP_SCOPE=PARTIAL model=motion" in out and "count_mismatch=0 scope_bad=0" in out   # smoke/seeds 未跑
    assert ("RS_SWAP_FULL=PASS model=motion runs=704 invalid=0 N=48/48 Z=32/32 Rnd=32/32 A=32/32 Bsw_swing=32/32 "
            f"Bside_dir=48/48 C_own=96/96 D1_swing=96/96 D2_dir=144/208 D2_abs={n_abs}/208 W0_swing=32/32 W0_dir=48/48") in out
    official = next(l for l in out.splitlines() if l.startswith("OFFICIAL_STEP1"))
    assert "known=704" in official and official.endswith("agree=704")
    data = json.loads((motion / "readout" / "readout.json").read_text())
    assert data["summary"]["strata"]["D2_by_stratum"]["in"]["n"] == 144
    assert data["summary"]["groups"]["Bsw"]["swing"]["donor_swing"] == 32
    assert (motion / "readout" / "readout.md").is_file()


def test_readout_layout_and_scope_fail(motion, capsys):
    plan = json.loads((motion / "plan.json").read_text())
    for ep in range(48):
        write_record(motion, ep)
    n_rows = [r for r in plan["runs"] if r["stage"] == "n"]
    for i, row in enumerate(n_rows):
        synth(row, outcome="success", reach=R.targets(row)[0], swing=row["recv_type"][2], nonce=f"n{i:031d}")
    receipt = next(Path(n_rows[0]["run_root"]).glob("attempts/*/client/traces/*/meta/hold_receipt.json"))
    receipt.write_text(json.dumps({"D": 100, "m": 2, "demo_task_index": [1] * 40 + [2] * 60 + [5]}))
    row = next(r for r in plan["runs"] if r["stage"] == "multieq")
    synth(row, outcome="fail", reach=R.targets(row)[0], swing="cw", nonce="m" * 32,
          mc={"mode": "swap_exact", "donor": f"RouteStick:{row['donor']}", "replace_log": [[0, 0], [64, 64]], "n_replaced": 2})
    capsys.readouterr()
    R.main(["--root", str(motion)])
    out = capsys.readouterr().out
    assert "RS_DEMO_LAYOUT=FAIL model=motion eps=48 bad=1" in out
    assert "scope_bad=1" in out and "SWAP_SCOPE=FAIL" in out


def test_readout_wan_gate_partial(pool_root, capsys, monkeypatch):
    out = pool_root / "wan"
    assert P.main(["specs", "--out", str(out), "--model", "wan", "--commit", COMMIT, "--checkpoint", "/ck",
                   "--policy-config", "c", "--model-kind", "k", "--target-contract", "none"]) == 0
    code, _, plan = run_donors(out, {1, 2}, capsys, extra=("--wan-gate",))
    assert code == 0 and plan["gate"]["min"] == 8
    for i, row in enumerate(r for r in plan["runs"] if r["stage"] == "n"):
        for f in (Path(row["run_root"]) / "accepted").glob("*.terminal.json"):
            f.unlink()
        synth(row, outcome="success", reach=R.targets(row)[0], swing=row["recv_type"][2], nonce=f"w{i:031d}")
    capsys.readouterr()
    R.main(["--root", str(out)])
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("RS_SWAP_FULL"))
    assert line.startswith("RS_SWAP_FULL=PARTIAL model=wan reason=few_success r_success=2 groups=N,Z,Rnd ")


def test_readout_intersect(pool_root, capsys):
    for m in ("motion", "wan"):
        out = pool_root / m
        assert P.main(["specs", "--out", str(out), "--model", m, "--commit", COMMIT,
                       *([] if m == "motion" else ["--checkpoint", "/ck", "--policy-config", "c", "--model-kind", "k",
                                                   "--target-contract", "none"])]) == 0
        plan = json.loads((out / "plan.json").read_text())
        for i, row in enumerate(r for r in plan["runs"] if r["stage"] == "n"):
            good = row["ep"] in ({0, 1, 2, 3} if m == "motion" else {2, 3, 4})
            synth(row, outcome="success" if good else "fail", reach=R.targets(row)[0], swing=row["recv_type"][2],
                  nonce=f"{m}{i:030d}")
    capsys.readouterr()
    assert R.main(["--root", str(pool_root), "--intersect", "motion,wan"]) == 0
    out = capsys.readouterr().out
    assert "RS_SWAP_INTERSECT models=motion,wan recv=2" in out
    assert json.loads((pool_root / "readout" / "intersect.json").read_text())["recv"] == [2, 3]
