"""pl_swap_prep 与 pl_swap_readout 的 CPU 测试：合成 manifest / npz / trace / episode.json，不起模型与仿真。"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from scripts.interpret import pl_swap_prep as P
from scripts.interpret import run_local as local
from scripts.interpret.analysis import pl_swap_readout as R

DIM = 5
RUN_LINE = re.compile(r"^run_one (\w) (\d+) (\d+) (\S+)$")


# ---------------------------------------------------------------- 表与条件
def test_episode_table_single_step():
    assert set(P.EP_ORDER) == set(P.EPISODES) and len(P.EP_ORDER) == 8
    for ep, (seed, path) in P.EPISODES.items():
        assert seed == 650000 + 100 * ep
        assert len(path) == 2 and path[0] != path[1]
        dr, dc = P.direction(path)
        assert max(abs(dr), abs(dc)) == 1   # 8 邻一步


def test_direction_labels():
    assert P.direction(P.EPISODES[5][1]) == (1, 1)    # 右下斜
    assert P.direction(P.EPISODES[20][1]) == (0, -1)  # 左
    assert P.direction(P.EPISODES[44][1]) == (0, 1)   # 右
    assert P.direction(P.EPISODES[29][1]) == (-1, 0)  # 上


def test_conditions_relations():
    path = lambda ep: P.EPISODES[ep][1]
    for ep, donor in P.CONDITIONS["A"]:
        assert path(ep)[0] == path(donor)[0] and P.direction(path(ep)) == P.direction(path(donor))
    for ep, donor in P.CONDITIONS["B"]:
        assert path(ep)[0] == path(donor)[0] and P.direction(path(ep)) != P.direction(path(donor))
    for ep, donor in P.CONDITIONS["C"]:
        assert path(ep)[0] != path(donor)[0] and P.direction(path(ep)) == P.direction(path(donor))
    assert sum(len(v) for v in P.CONDITIONS.values()) == 10
    assert P.donor_episodes() == [5, 20, 25, 28, 29, 32, 48]


def test_replay_path_matches_table():
    pytest.importorskip("torch")
    try:
        find_path = P._load_find_path()
    except Exception as exc:   # robomme 只在 official-sim 环境可导入
        pytest.skip(f"robomme 不可导入: {exc}")
    for ep, (seed, path) in P.EPISODES.items():
        assert tuple(P.replay_path(seed, find_path=find_path)) == path


def test_button_xy_consistent():
    for cell in range(9):
        assert P.button_xy(cell) == pytest.approx(R.button_xy(cell))
    assert R.button_xy(0) == pytest.approx((-0.2, -0.1))
    assert R.button_xy(5) == pytest.approx((-0.1, 0.1))


# ---------------------------------------------------------------- specs 与队列
def fake_manifests(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for shard in range(4):
        eps = []
        for task in ("PatternLock", "RouteStick"):
            for i in range(50):
                if shard * 4 <= i % 16 < shard * 4 + 4:
                    eps.append({"task_name": task, "episode_id": i, "env_seed": 650000 + 100 * i,
                                "difficulty": "easy", "episode_key": f"{task}:{i}"})
        (directory / f"manifest-s{shard}.json").write_text(json.dumps({"episodes": eps, "chunks": []}))


@pytest.fixture()
def generated(tmp_path):
    fake_manifests(tmp_path / "man")
    out = tmp_path / "O"
    assert P.main(["specs", "--out", str(out), "--commit", "deadbeefcafe", "--manifest-dir", str(tmp_path / "man"),
                   "--run-tree", str(tmp_path / "tree")]) == 0
    return out


def parse_queues(out):
    rows = []
    for stage in P.STAGES:
        for worker in P.WORKERS:
            path = out / "queues" / f"{stage}-{worker}.sh"
            text = path.read_text()
            assert f"env.sh {P.WORKER_CUDA[worker]} 0.30" in text
            subprocess.run(["bash", "-n", str(path)], check=True)
            for line in text.splitlines():
                m = RUN_LINE.match(line)
                if m:
                    rows.append((stage, worker, m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)))
    return rows


def test_queues_cover_130_unique(generated):
    rows = parse_queues(generated)
    assert len(rows) == 130
    keys = [(g, s, e) for _, _, g, s, e, _ in rows]
    assert len(set(keys)) == 130
    by_stage = {st: [r for r in rows if r[0] == st] for st in P.STAGES}
    assert {k: len(v) for k, v in by_stage.items()} == {"smoke1": 2, "smoke2": 2, "stage2": 6, "stage3a": 8,
                                                        "stage3b": 112}
    assert sorted((g, s, e) for _, _, g, s, e, _ in by_stage["smoke1"]) == [("N", 42, 5), ("N", 42, 20)]
    assert sorted((g, s, e) for _, _, g, s, e, _ in by_stage["smoke2"]) == [("B", 42, 20), ("Z", 42, 20)]
    assert all(g == "N" and s == 42 for _, _, g, s, _, _ in by_stage["stage2"])
    assert sorted(e for *_, e, _ in by_stage["stage3a"]) == sorted(P.EP_ORDER)
    assert all(s == 1 and g == "N" for _, _, g, s, _, _ in by_stage["stage3a"])
    counts = {g: sum(1 for r in rows if r[2] == g) for g in P.GROUPS}
    assert counts == {"N": 40, "Z": 40, "A": 10, "B": 15, "C": 25}
    for stage, members in by_stage.items():   # 每阶段均分到 4 个 worker
        sizes = [sum(1 for r in members if r[1] == w) for w in P.WORKERS]
        assert max(sizes) - min(sizes) <= 1


def test_specs_consistent_with_queues(generated, tmp_path):
    rows = parse_queues(generated)
    by_spec = {}
    for stage, worker, group, seed, ep, spec in rows:
        by_spec.setdefault(spec, []).append((stage, worker, group, seed, ep))
    donor_of = {(g, ep): d for g, pairs in P.CONDITIONS.items() for ep, d in pairs}
    run_roots = {}
    for spec_path, members in by_spec.items():
        raw = json.loads(Path(spec_path).read_text())
        spec = local.load_spec(spec_path)   # run_local 自己的合法性检查
        workers = {m[1] for m in members}
        assert len(workers) == 1
        worker = workers.pop()
        assert spec["cuda"] == P.WORKER_CUDA[worker]
        groups, seeds = {m[2] for m in members}, {m[3] for m in members}
        assert len(groups) == 1 and len(seeds) == 1
        group, seed = groups.pop(), seeds.pop()
        assert raw["model_seed"] == seed and raw["save_attn"] is False and raw["subgoal_trace"] is True
        assert "rollout_record" not in raw and raw["xla_mem_fraction"] == 0.30
        assert raw["repo"] == str(tmp_path / "tree")
        assert raw["attn_out"].endswith(f"/{group}/seed{seed}/{worker}/attn")
        assert raw["run_root"].endswith(f"/{group}/seed{seed}/{worker}/run")
        run_roots.setdefault(raw["run_root"], set()).add(worker)
        manifest_eps = {e["episode_key"] for e in json.loads(Path(raw["manifest"]).read_text())["episodes"]}
        eps = sorted(m[4] for m in members)
        assert all(f"PatternLock:{e}" in manifest_eps for e in eps)
        if group == "N":
            assert raw["ablate"] == "none" and "swap_map" not in raw
            assert ("record_motion" in raw) == (seed == 42)
        else:
            assert raw["ablate"] == "motion_swap_exact:" + str(generated / ("zero" if group == "Z" else "donor_trim"))
            assert sorted(spec["swap_map"]) == sorted(f"PatternLock:{e}" for e in eps)   # swap_map 与队列一致
            for e in eps:
                donor = e if group == "Z" else donor_of[(group, e)]
                assert spec["swap_map"][f"PatternLock:{e}"] == f"PatternLock:{donor}"
            assert "record_motion" not in raw
    assert all(len(ws) == 1 for ws in run_roots.values())   # 同一 run_root 只归一个 worker


def test_specs_profile_by_seed(generated):
    """seed 42 的 spec 不带 profile/max_steps；其余 seed 恰为 testx/1300，且客户端命令带上对应参数。"""
    from scripts.evaluation.resident_worker import client_command
    specs = [json.loads(p.read_text()) for p in sorted((generated / "specs").glob("*.json"))]
    assert any(s["model_seed"] == 42 for s in specs) and any(s["model_seed"] != 42 for s in specs)
    for spec in specs:
        if spec["model_seed"] == 42:
            assert "profile" not in spec and "max_steps" not in spec
        else:
            assert spec["profile"] == "testx" and spec["max_steps"] == 1300
    spec = next(s for s in specs if s["model_seed"] != 42)
    spec = dict(spec, sim_python="py", run_identity_sha256="0" * 64, model_kind="k")
    grant = generated / "grant.json"
    grant.write_text(json.dumps({"chunk_id": "c0"}))
    joined = " ".join(client_command(spec, grant, "out", "nonce", 1234, "drain"))
    assert "--profile testx --max-steps 1300" in joined
    assert f"--model-seed {spec['model_seed']}" in joined


def test_env_and_plan(generated):
    env = (generated / "env.sh").read_text()
    assert "deadbeefcafe" in env and "HF_HUB_OFFLINE=1" in env and "--xla_gpu_deterministic_ops=true" in env
    subprocess.run(["bash", "-n", str(generated / "env.sh")], check=True)
    plan = json.loads((generated / "plan.json").read_text())
    assert len(plan["runs"]) == 130
    assert all(r["cuda"] == P.WORKER_CUDA[r["worker"]] for r in plan["runs"])


def test_manifest_seed_mismatch_rejected(tmp_path):
    fake_manifests(tmp_path / "man")
    path = tmp_path / "man" / "manifest-s1.json"
    data = json.loads(path.read_text())
    for e in data["episodes"]:
        if e["episode_key"] == "PatternLock:5":
            e["env_seed"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        P.manifest_shards(tmp_path / "man")


# ---------------------------------------------------------------- donors
def emb_of(ep, start):
    return np.arange(DIM, dtype=np.float32) + 10 * ep + 0.5 * start


def write_record(out, ep, starts=(0, 33, 65), counts=None):
    starts = np.asarray(starts, np.int32)
    embs = np.stack([emb_of(ep, s) for s in starts]).astype(np.float32)
    counts = np.asarray(counts if counts is not None else [2] + [1] * (len(starts) - 1), np.int32)
    (out / "record").mkdir(parents=True, exist_ok=True)
    np.savez(out / "record" / f"PatternLock_ep{ep}.npz", start_frames=starts, embs=embs, counts=counts,
             mode=np.str_("none"))


def test_donors_smoke_subset(tmp_path, capsys):
    for ep in (5, 20):
        write_record(tmp_path, ep)
    assert P.main(["donors", "--out", str(tmp_path), "--episodes", "5,20"]) == 0
    assert "DONOR_TRIM=PASS donors=2 zeros=2 rows=1 maxabs=0" in capsys.readouterr().out
    with np.load(tmp_path / "donor_trim" / "PatternLock_ep5.npz") as z:
        assert z["start_frames"].tolist() == [0] and z["embs"].shape == (1, DIM)
        assert np.array_equal(z["embs"][0], emb_of(5, 0)) and z["counts"].tolist() == [2]
    with np.load(tmp_path / "zero" / "PatternLock_ep20.npz") as z:
        assert z["start_frames"].tolist() == [0] and z["embs"].shape == (1, DIM) and not z["embs"].any()
    # load_donor 接口能读
    from scripts.interpret import serve_policy_attn as SA
    donor = SA.load_donor(tmp_path / "donor_trim", "PatternLock:5")
    assert donor["start_frames"].tolist() == [0]


def test_donors_full_counts_and_missing(tmp_path, capsys):
    for ep in P.EP_ORDER:
        write_record(tmp_path, ep)
    assert P.main(["donors", "--out", str(tmp_path)]) == 0
    assert "DONOR_TRIM=PASS donors=7 zeros=8 rows=1 maxabs=0" in capsys.readouterr().out
    assert not (tmp_path / "donor_trim" / "PatternLock_ep44.npz").exists()   # 44 不当供体
    (tmp_path / "record" / "PatternLock_ep44.npz").unlink()
    assert P.main(["donors", "--out", str(tmp_path)]) == 1
    assert "DONOR_TRIM=FAIL" in capsys.readouterr().out


def test_donors_rejects_no_start0(tmp_path, capsys):
    write_record(tmp_path, 5, starts=(33, 65))
    assert P.main(["donors", "--out", str(tmp_path), "--episodes", "5"]) == 1
    out = capsys.readouterr().out
    assert "start0_rows=0" in out and "DONOR_TRIM=FAIL" in out


# ---------------------------------------------------------------- 读数
def test_first_reached_and_classify():
    c = R.button_xy
    start = 1
    # 先在起点附近徘徊，再到 5
    tcps = [[*c(1), 0.07], [c(1)[0] + 0.03, c(1)[1] + 0.03, 0.05], [c(5)[0] + 0.01, c(5)[1], 0.01]]
    assert R.first_reached(tcps, start) == 5
    assert R.classify(5, own=5, donor_abs=0) == "own"
    assert R.classify(0, own=5, donor_abs=0) == "donor_abs"
    assert R.classify(3, own=5, donor_abs=0) == "other"
    assert R.classify(None, own=5, donor_abs=0) == "none"
    assert R.classify(5, own=5, donor_abs=5) == "own"   # A 组：供体目标即自己目标
    assert R.first_reached([[*c(1), 0.1], [-0.1 + 0.05, 0.05, 0.0]], start) is None   # 两格中间，未到
    assert R.first_reached([[c(3)[0] + 0.024, c(3)[1], 0]], start) == 3
    assert R.first_reached([[c(3)[0] + 0.026, c(3)[1], 0]], start) is None


def synth_run(row, *, outcome, tcps, nonce, mc=None, extra_attempts=0):
    run_root, attn = Path(row["run_root"]), Path(row["attn_out"])
    key = f"PatternLock:{row['ep']}"
    for k in range(1 + extra_attempts):
        attempt = run_root / "attempts" / f"{k:03d}-{row['group']}{row['ep']}-{nonce[:4]}{k}"
        (attempt / "client").mkdir(parents=True, exist_ok=True)
        (attempt / "commands.json").write_text(json.dumps({"episode_keys": [key]}))
    trace = attempt / "client" / "traces" / nonce / "step_trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("".join(json.dumps({"event": "step", "tcp_after": t, "exec_start_idx": 40}) + "\n" for t in tcps))
    (run_root / "accepted").mkdir(parents=True, exist_ok=True)
    (run_root / "accepted" / f"{key}.{nonce[:8]}.terminal.json").write_text(
        json.dumps({"episode_key": key, "episode_nonce": nonce, "outcome": outcome}))
    (attn / nonce).mkdir(parents=True, exist_ok=True)
    info = {"episode_nonce": nonce}
    if mc is not None:
        info["motion_content"] = mc
    (attn / nonce / "episode.json").write_text(json.dumps(info))


def walk_to(start, cell):
    a, b = np.array(R.button_xy(start)), np.array(R.button_xy(cell))
    return [[*(a + (b - a) * t), 0.02] for t in np.linspace(0, 1, 12)]


def find(plan, group, seed, ep):
    return next(r for r in plan["runs"] if (r["group"], r["seed"], r["ep"]) == (group, seed, ep))


def test_readout_partial_and_scope(generated, capsys):
    plan = json.loads((generated / "plan.json").read_text())
    for ep in (5, 20):
        write_record(generated, ep, counts=[3, 1, 1])
    # seed 42 N 5、20 成功；seed 1 N 5 轨迹不同、20 完全相同
    synth_run(find(plan, "N", 42, 5), outcome="success", tcps=walk_to(1, 5), nonce="a" * 32)
    synth_run(find(plan, "N", 42, 20), outcome="success", tcps=walk_to(1, 0), nonce="b" * 32)
    synth_run(find(plan, "N", 1, 5), outcome="success", tcps=walk_to(1, 5)[:-1] + walk_to(1, 5)[-1:] * 2,
              nonce="c" * 32)
    synth_run(find(plan, "N", 1, 20), outcome="success", tcps=walk_to(1, 0), nonce="d" * 32)
    # B 20←5 走到供体目标格 5；Z 20 停在中间（未到）
    synth_run(find(plan, "B", 42, 20), outcome="fail", tcps=walk_to(1, 5), nonce="e" * 32,
              mc={"mode": "swap_exact", "donor": "PatternLock:5", "n_windows": 3, "n_replaced": 3,
                  "replace_log": [[0, 0]] * 3})
    mid = [[-0.15, -0.05, 0.02]]
    synth_run(find(plan, "Z", 42, 20), outcome="fail", tcps=mid, nonce="f" * 32, extra_attempts=1,
              mc={"mode": "swap_exact", "donor": "PatternLock:20", "n_windows": 3, "n_replaced": 3,
                  "replace_log": [[0, 0]] * 3})
    out = generated / "ro"
    assert R.main(["--root", str(generated), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "RECORD_REPRO=PARTIAL eps=2 success=2" in text
    assert "SEED_EFFECT=PASS seeds=2 eps_differ=1" in text
    assert "SWAP_SCOPE=PARTIAL runs=2 log_all_00=1 count_mismatch=0" in text
    assert "PL_SWAP=PARTIAL runs=6 invalid=1 N=4/4 Z=0/1 A=0/0 B_donor=1/1 C_own=0/0 C_donor=0/0" in text
    data = json.loads((out / "readout.json").read_text())
    cats = {(r["group"], r["seed"], r["ep"]): r.get("category") for r in data["runs"] if r["status"] == "ok"}
    assert cats[("B", 42, 20)] == "donor_abs" and cats[("Z", 42, 20)] == "none" and cats[("N", 42, 5)] == "own"
    assert data["cosine_distance"]["B:20<-5"] is not None
    assert (out / "readout.md").read_text().startswith("# PatternLock")
    # 中间检查：只读 seed 42、1
    assert R.main(["--root", str(generated), "--out", str(out), "--seeds", "42,1"]) == 0
    assert "SEED_EFFECT=PASS seeds=2 eps_differ=1" in capsys.readouterr().out


def test_readout_scope_mismatch_and_other(generated, capsys):
    plan = json.loads((generated / "plan.json").read_text())
    write_record(generated, 20, counts=[2, 1])
    write_record(generated, 25, counts=[1, 1])
    # C 20←25：走到 3（other：自己目标 0，供体绝对目标 3 → donor_abs）；再造一个走到 2 的 other
    synth_run(find(plan, "C", 42, 20), outcome="fail", tcps=walk_to(1, 3), nonce="1" * 32,
              mc={"mode": "swap_exact", "donor": "PatternLock:25", "n_windows": 2, "n_replaced": 1,
                  "replace_log": [[0, 0]]})
    synth_run(find(plan, "C", 1, 20), outcome="fail", tcps=walk_to(1, 2), nonce="2" * 32,
              mc={"mode": "swap_exact", "donor": "PatternLock:25", "n_windows": 2, "n_replaced": 2,
                  "replace_log": [[0, 0], [33, 0]]})
    assert R.main(["--root", str(generated), "--out", str(generated / "ro2")]) == 0
    text = capsys.readouterr().out
    assert "SWAP_SCOPE=FAIL runs=2 log_all_00=0 count_mismatch=1" in text
    assert "C_own=0/2 C_donor=1/2" in text
    data = json.loads((generated / "ro2" / "readout.json").read_text())
    cats = {(r["seed"]): r["category"] for r in data["runs"] if r["status"] == "ok"}
    assert cats == {42: "donor_abs", 1: "other"}


def test_seed_effect_identical_fails(generated, capsys):
    plan = json.loads((generated / "plan.json").read_text())
    for i, seed in enumerate((42, 1)):
        synth_run(find(plan, "N", seed, 44), outcome="success", tcps=walk_to(0, 1), nonce=f"{i}" * 32)
    R.main(["--root", str(generated), "--out", str(generated / "ro3"), "--seeds", "42,1"])
    assert "SEED_EFFECT=FAIL seeds=2 eps_differ=0" in capsys.readouterr().out


def test_sequences_differ():
    a = [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]
    assert not R.sequences_differ(a, [list(x) for x in a])
    assert R.sequences_differ(a, a[:1])
    assert R.sequences_differ(a, [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1 + 2e-6]])
    assert not R.sequences_differ(a, [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1 + 5e-7]])


def test_bootstrap_deterministic():
    flags = [1, 0, 1, 1, 0, 1, 0, 1]
    assert R.bootstrap_ci(flags) == R.bootstrap_ci(flags)
    lo, hi = R.bootstrap_ci(flags)
    assert 0 <= lo <= np.mean(flags) <= hi <= 1
    assert R.bootstrap_ci([]) is None
