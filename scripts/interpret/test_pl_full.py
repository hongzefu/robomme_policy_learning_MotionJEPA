"""pl_full_prep 与 pl_full_readout 的 CPU 测试：合成局池、录制 npz、终局与轨迹，不起模型与仿真、不依赖 torch。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation import manifest as M
from scripts.interpret import pl_full_prep as P
from scripts.interpret.analysis import pl_full_readout as R
from scripts.interpret.analysis import pl_swap_readout as PR

DIM = 6


# ---------------------------------------------------------------- 组合与配对
def test_forty_combos():
    combos = P.all_combos()
    assert len(combos) == 40 and len(set(combos)) == 40
    assert combos == sorted(combos)                    # 起点格升序、再方向升序
    assert combos[:3] == [(0, (0, 1)), (0, (1, 0)), (0, (1, 1))]
    for start, d in combos:
        assert P.step_cell(start, d) is not None


def test_pair_counts_and_relations():
    pairs = P.build_pairs()
    assert pairs["counts"] == {"A": 40, "B": 160, "C": 168, "D": 1232}
    assert sum(pairs["counts"].values()) == 40 * 40
    for p in pairs["all"]["A"]:
        assert p["donor"] == p["recv"] + 40
    for p in pairs["all"]["B"]:
        assert p["recv_start"] == p["donor_start"] and p["recv_dir"] != p["donor_dir"]
        assert p["donor_dir_cell"] == p["donor_abs"]   # 同起点：供体方向推出的格就是供体目标
    for p in pairs["all"]["C"]:
        assert p["recv_start"] != p["donor_start"] and p["recv_dir"] == p["donor_dir"]
        assert p["donor_dir_cell"] == p["own"]          # 同方向：并入 own
    for p in pairs["all"]["D"]:
        assert p["recv_start"] != p["donor_start"] and p["recv_dir"] != p["donor_dir"]
        assert (p["stratum"] == "in") == (P.step_cell(p["recv_start"], p["donor_dir"]) is not None)
    line, ok = P.pairs_line(pairs)
    assert ok and line == "PL_PAIRS=PASS A=40 B=160 C=168 D=1232 D_sel=400 seedcheck=60"


def test_d_sampling_deterministic_and_stratified():
    a, b = P.build_pairs(), P.build_pairs()
    key = lambda ps: [(p["recv"], p["donor"]) for p in ps]
    assert key(a["selected"]["D"]) == key(b["selected"]["D"])
    assert {g: key(v) for g, v in a["seedcheck"].items()} == {g: key(v) for g, v in b["seedcheck"].items()}
    sel = a["selected"]["D"]
    assert len(sel) == 400 and len(set(key(sel))) == 400
    assert sum(p["stratum"] == "in" for p in sel) == 300 and sum(p["stratum"] == "out" for p in sel) == 100
    assert a["d_strata"]["in"] + a["d_strata"]["out"] == 1232
    # seed 复核：每组 15 对且取自已选配对（D 取自已抽的 400）
    for group in P.PAIR_GROUPS:
        chosen = key(a["seedcheck"][group])
        assert len(chosen) == 15 and set(chosen) <= set(key(a["selected"][group]))


def test_multieq_pairs_unique_receivers():
    chosen = P.multieq_pairs(P.build_pairs())
    assert len(chosen) == 12 and [g for g, _ in chosen] == ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3
    assert len({p["recv"] for _, p in chosen}) == 12


def test_pack_unique_receivers_and_cap():
    runs = [("k", r % 7, i) for i, r in enumerate(range(50))] + [("j", 1, 99)]
    bins = P.pack(runs, cap=5)
    assert sum(len(m) for _, m in bins) == len(runs)
    for key, members in bins:
        assert len(members) <= 5 and len({m[1] for m in members}) == len(members)
        assert all(m[0] == key for m in members)


# ---------------------------------------------------------------- 合成局池
def fake_replay(seed):
    """seed → 路径：每 50 个 seed 中前 40 个依次是 40 种组合的单步局，其余为两步局。"""
    combos = P.all_combos()
    idx = (seed // 100) % 50
    if idx < 40:
        start, d = combos[idx]
        return [start, P.step_cell(start, d)]
    return [0, 4, 8]


@pytest.fixture
def pool_root(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "replay_path", fake_replay)
    root = tmp_path / "O"
    assert P.main(["pool", "--out", str(root)]) == 0
    return root


def test_pool_files(pool_root, capsys):
    pool = json.loads((pool_root / "pool.json").read_text())
    assert pool["scan"]["scanned"] == 3950
    eps = pool["episodes"]
    assert [e["episode_id"] for e in eps] == list(range(80))
    combos = P.all_combos()
    for e in eps:
        k = e["episode_id"] % 40
        assert (e["start"], tuple(e["dir"])) == combos[k] and e["role"] == ("R" if e["episode_id"] < 40 else "S")
    for k in range(40):   # 同组合 R 取 env_seed 较小的那局
        assert eps[k]["env_seed"] < eps[40 + k]["env_seed"]
    assert not any(1150000 <= e["env_seed"] <= 1154900 for e in eps)
    meta = json.loads((pool_root / "metadata" / P.META_NAME).read_text())
    assert meta["env_id"] == "PatternLock" and meta["record_count"] == 80
    assert meta["records"][3] == {"task": "PatternLock", "episode": 3, "seed": eps[3]["env_seed"], "difficulty": "easy"}
    man = M.validate_episodes(json.loads((pool_root / "manifest.json").read_text()), official=False)
    assert len(man["episodes"]) == 80 and all(len(c["episode_keys"]) <= 5 for c in man["chunks"])
    assert man["episodes"][0]["episode_key"] == "PatternLock:0" and man["sha256"] == pool["manifest_seal"]
    capsys.readouterr()
    assert P.main(["pool", "--out", str(pool_root)]) == 0   # 同内容重跑幂等
    out = capsys.readouterr().out
    assert "PL_POOL=PASS eps=80 combos=40 overlap=0" in out
    assert "PL_PAIRS=PASS A=40 B=160 C=168 D=1232 D_sel=400 seedcheck=60" in out


def test_pool_refuses_overwrite(pool_root):
    pool = json.loads((pool_root / "pool.json").read_text())
    pool["episodes"][0]["env_seed"] += 1
    (pool_root / "pool.json").write_text(json.dumps(pool))
    with pytest.raises(SystemExit):
        P.main(["pool", "--out", str(pool_root)])


# ---------------------------------------------------------------- specs
COMMIT = "a" * 40


@pytest.fixture
def motion(pool_root, capsys):
    out = pool_root / "motion"
    assert P.main(["specs", "--out", str(out), "--model", "motion", "--commit", COMMIT, "--run-tree", "/tmp/tree"]) == 0
    assert "PLF_SPECS model=motion repro=50 smoke=5 multieq=24 n=80" in capsys.readouterr().out
    return out


def load_specs(out):
    return {p.stem: json.loads(p.read_text()) for p in sorted((out / "specs").glob("*.json"))}


def test_specs_keys(motion, pool_root):
    specs = load_specs(motion)
    plan = json.loads((motion / "plan.json").read_text())
    for name, spec in specs.items():
        man = M.validate_episodes(json.loads(Path(spec["manifest"]).read_text()), official=False)
        assert all(len(c["episode_keys"]) <= 5 for c in man["chunks"])
        assert spec["model_kind"] == "stage2_pool4_concat" and spec["repo"] == "/tmp/tree"
        assert spec["subgoal_trace"] is True and spec["save_attn"] is False
        assert Path(spec["run_root"]).parent.name == name
        if name.startswith("repro"):
            assert "profile" not in spec and spec["model_seed"] == 42 and spec["ablate"] == "none"
            assert all(e["env_seed"] == 650000 + 100 * e["episode_id"] for e in man["episodes"])
            continue
        assert spec["profile"] == "plfull" and spec["plfull_metadata_dir"] == str(pool_root / "metadata")
        keys = [e["episode_key"] for e in man["episodes"]]
        assert len(keys) <= P.SPEC_MAX and len(set(keys)) == len(keys)
        if "swap_map" in spec:
            assert sorted(spec["swap_map"]) == sorted(keys)
            assert spec["ablate"].startswith("motion_swap_exact:")
        else:
            assert spec["ablate"] == "none"
    assert sorted(e["ep"] for e in plan["runs"] if e["stage"] == "n") == list(range(80))
    assert all(specs[r["spec"]].get("record_motion") == str(motion / "record") for r in plan["runs"] if r["stage"] == "n")
    assert sorted(r["ep"] for r in plan["runs"] if r["stage"] == "repro") == list(range(50))
    multi = specs["multieq-multi"]
    assert len(multi["swap_map"]) == 12
    singles = [s for n, s in specs.items() if n.startswith("multieq-single")]
    assert len(singles) == 12 and all(len(s["swap_map"]) == 1 for s in singles)
    assert {k: v for s in singles for k, v in s["swap_map"].items()} == multi["swap_map"]
    # 队列：每个 worker 一个，GPU 对应
    for stage in ("repro", "n"):
        for w in P.WORKERS:
            text = (motion / "queues" / f"{stage}-{w}.sh").read_text()
            assert f"env.sh {P.WORKER_CUDA[w]} 0.30" in text and text.count("run_spec ") == 1
    smoke = (motion / "queues" / "smoke-w00.sh").read_text()
    assert smoke.index("smoke-n") < smoke.index("pl_full_prep donors") < smoke.index("smoke-swap") < smoke.index("smoke-zero")
    launch = (motion / "launch_stage.sh").read_text()
    assert "plf-motion-$st-$w" in launch and "EXIT_CODE=" in launch and "pl-swap-full-1109" in launch
    import subprocess
    for script in [motion / "launch_stage.sh", motion / "env.sh", *sorted((motion / "queues").glob("*.sh"))]:
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0, script
    env = (motion / "env.sh").read_text()
    assert COMMIT in env and "XLA_PYTHON_CLIENT_MEM_FRACTION=$frac" in env


def test_wan_needs_model_entry(pool_root):
    with pytest.raises(SystemExit):
        P.main(["specs", "--out", str(pool_root / "wan"), "--model", "wan", "--commit", COMMIT])
    assert P.main(["specs", "--out", str(pool_root / "wan"), "--model", "wan", "--commit", COMMIT,
                   "--checkpoint", "/ck", "--policy-config", "cfg", "--model-kind", "wan_kind",
                   "--target-contract", "/c.json", "--extra", '{"wan_sidecar": "x"}']) == 0
    spec = json.loads((pool_root / "wan" / "specs" / "n-w00.json").read_text())
    assert spec["model_kind"] == "wan_kind" and spec["wan_sidecar"] == "x" and spec["checkpoint"] == "/ck"


# ---------------------------------------------------------------- donors
def emb_of(ep, start):
    return np.arange(DIM, dtype=np.float32) * 0.1 + ep + 0.5 * start + 1.0


def write_record(base, ep, starts=(0, 33), counts=(2, 1)):
    (base / "record").mkdir(parents=True, exist_ok=True)
    np.savez(base / "record" / P.record_name(ep), start_frames=np.asarray(starts, np.int32),
             embs=np.stack([emb_of(ep, s) for s in starts]).astype(np.float32), counts=np.asarray(counts, np.int32))


def write_terminal(row, outcome, nonce, demo_windows=1):
    run_root = Path(row["run_root"])
    key = f"PatternLock:{row['ep']}"
    (run_root / "accepted").mkdir(parents=True, exist_ok=True)
    (run_root / "accepted" / f"{key}.{nonce[:8]}.terminal.json").write_text(json.dumps(
        {"episode_key": key, "episode_nonce": nonce, "outcome": outcome, "budget": {"demo_windows": demo_windows}}))


def test_random_same_norm():
    row = emb_of(3, 0)[None, :]
    a, b = P.random_like(row, 3), P.random_like(row, 3)
    assert np.array_equal(a, b) and a.shape == row.shape
    assert np.isclose(np.linalg.norm(a), np.linalg.norm(row), rtol=1e-6)
    assert not np.allclose(a, row) and not np.array_equal(P.random_like(row, 4), a)


def run_donors(motion, success_recv, capsys, extra=()):
    plan = json.loads((motion / "plan.json").read_text())
    for ep in range(80):
        write_record(motion, ep)
    for row in plan["runs"]:
        if row["stage"] == "n":
            ok = row["ep"] >= 40 or row["ep"] in success_recv
            write_terminal(row, "success" if ok else "fail", f"{row['ep']:08d}aaaa")
    code = P.main(["donors", "--out", str(motion), *extra])
    return code, capsys.readouterr().out, json.loads((motion / "plan.json").read_text())


def test_donors_files_and_swap_specs(motion, capsys):
    ok_recv = set(range(0, 40, 2))   # 20 个接收局原样成功
    code, out, plan = run_donors(motion, ok_recv, capsys)
    assert code == 0, out
    assert "DONOR_TRIM=PASS donors=40 zeros=40 randoms=40 rows=1 maxabs=0" in out
    assert "DONOR_DISTINCT=PASS donors=40 identical_pairs=0" in out
    with np.load(motion / "donor_trim" / P.record_name(45)) as z:
        assert z["start_frames"].tolist() == [0] and np.array_equal(z["embs"][0], emb_of(45, 0)) and z["counts"].tolist() == [2]
    with np.load(motion / "random" / P.record_name(7)) as z:
        assert np.isclose(np.linalg.norm(z["embs"]), np.linalg.norm(emb_of(7, 0)), rtol=1e-6)
    assert not (motion / "donor_trim" / P.record_name(3)).exists()
    swap = [r for r in plan["runs"] if r["stage"] in ("swap", "seeds")]
    assert swap and all(r["ep"] in ok_recv for r in swap)
    pairs = P.build_pairs()
    want = {g: sum(1 for p in pairs["selected"][g] if p["recv"] in ok_recv) for g in P.PAIR_GROUPS}
    got = {g: sum(1 for r in swap if r["stage"] == "swap" and r["group"] == g) for g in P.PAIR_GROUPS}
    assert got == want
    assert sum(1 for r in swap if r["stage"] == "swap" and r["group"] in ("Z", "Rnd")) == 2 * len(ok_recv)
    assert {r["seed"] for r in swap if r["stage"] == "seeds"} == {1, 2}
    specs = load_specs(motion)
    for r in swap:
        spec = specs[r["spec"]]
        assert spec["swap_map"][f"PatternLock:{r['ep']}"] == f"PatternLock:{r['donor']}"
        assert spec["model_seed"] == r["seed"] and spec["profile"] == "plfull"
        assert spec["ablate"] == f"motion_swap_exact:{motion / P.ABLATE_DIR[r['group']]}"
    assert plan["gate"]["few"] is False


def test_donors_wan_gate(motion, capsys):
    code, out, plan = run_donors(motion, {1, 2, 3}, capsys, extra=("--wan-gate",))
    assert code == 0 and "gate=few" in out
    swap = [r for r in plan["runs"] if r["stage"] in ("swap", "seeds")]
    assert {r["group"] for r in swap} == {"Z", "Rnd"} and len(swap) == 6
    assert plan["gate"]["few"] is True and plan["gate"]["r_success"] == 3


def test_donors_identical_detected(motion, capsys):
    plan = json.loads((motion / "plan.json").read_text())
    for ep in range(80):
        write_record(motion, ep)
    write_record(motion, 41)
    import shutil
    shutil.copy(motion / "record" / P.record_name(41), motion / "record" / P.record_name(42))
    for row in plan["runs"]:
        if row["stage"] == "n":
            write_terminal(row, "success", f"{row['ep']:08d}bbbb")
    assert P.main(["donors", "--out", str(motion)]) == 1
    out = capsys.readouterr().out
    assert "DONOR_DISTINCT=FAIL donors=40 identical_pairs=1" in out and "DONORS_SPECS=SKIP" in out


# ---------------------------------------------------------------- 读数
def test_classify5_branches():
    assert R.classify5(None, 5) == "none"
    assert R.classify5(5, 5, 5, 5) == "own"            # 重合记 own
    assert R.classify5(3, 5, 3, 3) == "donor_dir"      # donor_abs 与 donor_dir 重合不单记
    assert R.classify5(3, 5, None, 3) == "donor_abs"
    assert R.classify5(3, 5, 7, 3) == "donor_abs"
    assert R.classify5(2, 5, 7, 3) == "other"
    # targets：接收 [4,5]（中心向右）、供体 [0,3]（左上角向下）→ donor_dir=4+下=7，donor_abs=3
    assert R.targets([4, 5], [0, 3]) == (5, 7, 3)
    assert R.targets([2, 5], [4, 1]) == (5, None, 1)    # 供体向上，从右上角出盘
    assert R.targets([4, 5], None) == (5, None, None)


def walk_to(start, cell):
    """起点上方一帧，再直接落到目标格（不经过中间按钮）。"""
    return [[*PR.button_xy(start), 0.07], [*PR.button_xy(cell), 0.01]]


def synth(row, *, outcome, reach, nonce, mc=None, demo_windows=1):
    run_root, attn = Path(row["run_root"]), Path(row["attn_out"])
    key = f"PatternLock:{row['ep']}"
    attempt = run_root / "attempts" / f"000-{nonce[:6]}"
    (attempt / "client").mkdir(parents=True, exist_ok=True)
    (attempt / "commands.json").write_text(json.dumps({"episode_keys": [key]}))
    trace = attempt / "client" / "traces" / nonce / "step_trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("".join(json.dumps({"event": "step", "tcp_after": t}) + "\n" for t in walk_to(row["path"][0], reach)))
    write_terminal(row, outcome, nonce, demo_windows)
    (attn / nonce).mkdir(parents=True, exist_ok=True)
    (attn / nonce / "episode.json").write_text(json.dumps({"episode_nonce": nonce, **({"motion_content": mc} if mc else {})}))


def test_readout_end_to_end(motion, capsys):
    ok_recv = set(range(40))
    code, _, plan = run_donors(motion, ok_recv, capsys)
    assert code == 0
    for r in [r for r in plan["runs"] if r["stage"] == "n"]:   # 去掉 donors 测试用的空终局
        for f in (Path(r["run_root"]) / "accepted").glob("*.terminal.json"):
            f.unlink()
    n = 0
    for row in plan["runs"]:
        if row["stage"] not in ("n", "swap", "multieq"):
            continue
        n += 1
        nonce = f"{n:032x}"
        own, ddir, dabs = R.targets(row["path"], row.get("donor_path"))
        if row["stage"] == "n" or row["group"] in ("Z", "Rnd", "A", "C"):
            reach, outcome = own, "success"
        elif row["group"] == "B":
            reach, outcome = ddir, "fail"
        else:   # D：盘内跟供体方向，出盘走到供体绝对目标
            reach, outcome = (ddir, "fail") if ddir is not None else (dabs, "fail")
        mc = None
        if row["group"] != "N":
            mc = {"mode": "swap_exact", "donor": f"PatternLock:{row['donor']}", "replace_log": [[0, 0], [0, 0]],
                  "n_replaced": 2}
        synth(row, outcome=outcome, reach=reach, nonce=nonce, mc=mc)
    capsys.readouterr()
    assert R.main(["--root", str(motion)]) == 0
    out = capsys.readouterr().out
    pairs = P.build_pairs()
    n_d_in = sum(p["stratum"] == "in" for p in pairs["selected"]["D"])
    n_d_abs = sum(p["stratum"] == "out" and p["donor_abs"] not in (p["own"], p["recv_start"]) for p in pairs["selected"]["D"])
    assert "DEMO_WINDOWS=PASS model=motion eps=80 not_one=0" in out
    assert "MULTI_SWAP_EQ=PASS runs=12 differ=0" in out
    assert (f"PL_SWAP_FULL=PASS model=motion runs={80 + 80 + 40 + 160 + 168 + 400} invalid=0 N=80/80 Z=40/40 Rnd=40/40 "
            f"A=40/40 B_donor=160/160 C_own=168/168 D_dir={n_d_in}/400 D_abs={n_d_abs}/400") in out
    assert "SWAP_SCOPE=PARTIAL model=motion" in out   # seeds 阶段未跑
    assert "count_mismatch=0" in out
    data = json.loads((motion / "readout" / "readout.json").read_text())
    assert data["summary"]["strata"]["D_by_stratum"]["in"]["n"] == n_d_in
    assert (motion / "readout" / "readout.md").is_file()


def test_readout_intersect(pool_root, capsys):
    for m in ("motion", "wan"):
        out = pool_root / m
        assert P.main(["specs", "--out", str(out), "--model", "motion" if m == "motion" else "wan", "--commit", COMMIT,
                       *([] if m == "motion" else ["--checkpoint", "/ck", "--policy-config", "c", "--model-kind", "k",
                                                   "--target-contract", "/t"])]) == 0
        plan = json.loads((out / "plan.json").read_text())
        for i, row in enumerate(r for r in plan["runs"] if r["stage"] == "n"):
            good = row["ep"] in ({0, 1, 2, 3} if m == "motion" else {2, 3, 4})
            synth(row, outcome="success" if good else "fail", reach=row["path"][1], nonce=f"{m}{i:030d}")
    capsys.readouterr()
    assert R.main(["--root", str(pool_root), "--intersect", "motion,wan"]) == 0
    out = capsys.readouterr().out
    assert "PL_SWAP_INTERSECT models=motion,wan recv=2" in out
    assert json.loads((pool_root / "readout" / "intersect.json").read_text())["recv"] == [2, 3]
