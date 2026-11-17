"""rsprog_points 纯 CPU 单测（计划 1117 B4 R1）：spec 被子模块原样加载、200 条路线过 check_spec、spec_sha256 重算一致、
合成 H5 上的取点／标签／边界／配对、mem_token_segments 边界用例。

子模块源码只读导入：默认主检出 third_party/robomme_benchmark，可用环境变量 RSPROG_BENCH_ROOT 覆盖。
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.interpret import rsprog_points as R
from scripts.interpret import samep_rs_variants as V

BENCH = Path(os.environ.get("RSPROG_BENCH_ROOT", str(R.DEFAULT_BENCH_ROOT)))
needs_bench = pytest.mark.skipif(not (BENCH / "scripts/generate_dataset_newseed.py").is_file(),
                                 reason="找不到子模块生成脚本")


@pytest.fixture(scope="module")
def full_pool():
    return R.build_pool()


# ---------------------------------------------------------------- 局池与 spec
def test_pool_layout(full_pool):
    rows, specs = full_pool
    assert len(rows) == 200
    for d in R.DIFFICULTIES:
        sub = [r for r in rows if r["difficulty"] == d]
        assert len(sub) == 50
        assert [r["episode"] for r in sub] == list(range(1000, 1050))
        assert len(specs[d]) == 50
    assert rows[0]["seed"] == 3_400_000 and rows[199]["seed"] == 3_400_000 + 100 * 199
    assert rows[50]["difficulty"] == "medium" and rows[150]["difficulty"] == "xhard"


def test_all_routes_pass_check_spec(full_pool):
    rows, specs = full_pool
    for d in R.DIFFICULTIES:
        lo, hi = V.CFG[d][0]
        for rec in specs[d]:
            assert V.check_spec(rec, d) == []
            assert lo <= rec["objects"]["L"] <= hi
            assert rec["difficulty"] == d and rec["task"] == "RouteStick"


def test_moves_and_sides(full_pool):
    rows, _ = full_pool
    for r in rows:
        assert len(r["sides"]) == len(r["moves"]) == len(r["directions"]) == r["L"]
        for s, d, m in zip(r["sides"], r["directions"], r["moves"]):
            assert R.MOVES[(s, d)] == m
    # theta ∈ [−30°, 30°]，cos > 0：槽位增大即 y 增大即 left
    for r in rows:
        for (a, b), s in zip(zip(r["node_slots"][:-1], r["node_slots"][1:]), r["sides"]):
            assert s == ("left" if b > a else "right")


def test_cell_xy_matches_rs_full_prep():
    try:
        from scripts.interpret import rs_full_prep
    except Exception as exc:  # 该模块导入链较重，导入不了就跳过
        pytest.skip(f"rs_full_prep 导入失败：{exc}")
    for cell in range(9):
        for th in (-0.5, 0.0, 0.3):
            assert np.allclose(R.cell_xy(cell, th), rs_full_prep.cell_xy(cell, th))


@needs_bench
def test_spec_sha_matches_submodule(full_pool):
    gen = R.load_generator(BENCH)
    _, specs = full_pool
    for d in R.DIFFICULTIES:
        for rec in specs[d]:
            assert gen.spec_record_sha256(rec) == rec["spec_sha256"] == R.spec_sha256(rec)
    assert R.canonical_json({"b": 1, "a": [1.5, "中"]}) == gen._spec_canonical_json({"b": 1, "a": [1.5, "中"]})


@needs_bench
def test_specs_load_in_submodule(full_pool, tmp_path):
    rows, specs = full_pool
    sdir = R.write_specs(rows, specs, tmp_path)
    ok, lines = R.spec_load_check(sdir, BENCH)
    assert ok, lines
    gen = R.load_generator(BENCH)
    groups = gen.load_episode_specs(sdir / "manifest.json", BENCH, tmp_path / "gen-out")
    assert sorted(g.difficulty for g in groups) == sorted(R.DIFFICULTIES)
    for g in groups:
        assert g.episodes == tuple(range(1000, 1050))
        assert g.output_root.endswith(f"gen-out/RouteStick/{g.difficulty}")
    for d in ("easy", "medium", "hard"):
        single = gen.load_episode_specs(sdir / f"{d}.json", BENCH, tmp_path / "o", task="RouteStick", difficulty=d)
        assert len(single) == 1 and len(single[0].records) == 50
    pool = json.loads((tmp_path / "pool.json").read_text())["episodes"]
    assert [r["spec_sha256"] for r in pool] == [r["spec_sha256"] for r in rows]


@needs_bench
def test_tampered_spec_rejected(full_pool, tmp_path):
    rows, specs = full_pool
    bad = copy.deepcopy(specs)
    bad["hard"][3]["actions"]["directions"][0] = V.flip_dir(bad["hard"][3]["actions"]["directions"][0])
    sdir = R.write_specs(rows, bad, tmp_path)
    gen = R.load_generator(BENCH)
    with pytest.raises(gen.EpisodeSpecError, match="spec_sha256"):
        gen.load_episode_specs(sdir / "manifest.json", BENCH, tmp_path / "o")


# ---------------------------------------------------------------- 合成 H5
def _img(value):
    return np.full((4, 4, 3), value, dtype=np.uint8)


def write_episode(h5, row, exec_lens=None, choice_override=None, image_of=None):
    """按官方示教的帧结构写一局：示范 50L 帧，执行按 exec_lens（默认每步 50），末步倒数第 7 帧是完成标记边界。"""
    L, moves = row["L"], row["moves"]
    exec_lens = exec_lens or [50] * L
    g = h5.create_group(f"episode_{row['episode']}")
    g.create_dataset("setup/seed", data=np.int64(777))
    frames = []
    for j in range(L):
        for i in range(50):
            frames.append(dict(demo=True, bnd=i == 0, comp=False, choice=moves[j], step=j, first=i == 0))
    for j, n in enumerate(exec_lens):
        for i in range(n):
            frames.append(dict(demo=False, bnd=i == 0, comp=False, choice=moves[j], step=j, first=i == 0))
    frames[-7]["bnd"] = True
    for f in frames[-7:]:
        f["comp"] = True
    if choice_override:
        for t, c in choice_override.items():
            frames[t]["choice"] = c
    e0 = 50 * L
    for t, f in enumerate(frames):
        ts = g.create_group(f"timestep_{t}")
        ts.create_dataset("info/is_video_demo", data=f["demo"])
        ts.create_dataset("info/is_subgoal_boundary", data=f["bnd"])
        ts.create_dataset("info/is_completed", data=f["comp"])
        ts.create_dataset("action/choice_action", data=json.dumps({"choice": f["choice"], "point": []}))
        val = image_of(t, f, row) if image_of else (t % 200)
        ts.create_dataset("obs/front_rgb", data=_img(val))
        ts.create_dataset("obs/wrist_rgb", data=_img(val))
    return e0, len(frames)


def _revisit_row(rows):
    """找一局：同一出发按钮在两个不同执行步起点出现（xhard 必有）。"""
    for r in rows:
        if r["difficulty"] == "xhard":
            starts = r["node_slots"][:-1]
            if len(set(starts)) < len(starts):
                return r
    raise AssertionError("没有回访局")


def _save_pool(tmp_path, rows):
    p = tmp_path / "pool.json"
    p.write_text(json.dumps({"episodes": rows}))
    return p


def test_points_labels_regular(full_pool, tmp_path):
    rows, _ = full_pool
    r = next(x for x in rows if x["difficulty"] == "medium")
    d = tmp_path / "data/RouteStick/medium"
    d.mkdir(parents=True)
    with h5py.File(d / R.H5_NAME, "w") as h5:
        e0, n = write_episode(h5, r)
    res = R.run_points(_save_pool(tmp_path, [r]), tmp_path / "data")
    assert res["verdicts"]["GEN_SPEC"].startswith("GEN_SPEC=PASS eps=1 mismatches=0 dropped=0 irregular=0")
    L = r["L"]
    pts = res["points"]
    assert [p["u"] for p in pts] == list(range(0, 50 * L, 16))
    ep = res["episodes"][0]
    assert ep["e0"] == e0 == 50 * L and ep["n_frames"] == n == 100 * L
    assert ep["demo_bounds"] == [[50 * j, 50 * j + 49] for j in range(L)]
    assert ep["exec_bounds"] == [[e0 + 50 * j, e0 + 50 * j + 49] for j in range(L)]
    assert ep["env_seed"] == 777
    for p in pts:
        assert p["t"] == e0 + p["u"]
        assert p["j"] == p["u"] // 50
        assert p["tau"] == pytest.approx(p["u"] / (50 * L))
        assert p["tau_time"] == pytest.approx(p["u"] / (50 * L))
        assert p["remaining"] == L - p["j"]
        assert p["cur_move"] == r["moves"][p["j"]]
        assert p["next_move"] == (r["moves"][p["j"] + 1] if p["j"] + 1 < L else None)
        assert p["button_slot"] == r["node_slots"][p["j"]] and p["target_slot"] == r["node_slots"][p["j"] + 1]
        assert p["step_start"] == (p["u"] % 50 < 16)
    assert sum(p["step_start"] for p in pts) == L


def test_points_irregular_uses_actual_bounds(full_pool, tmp_path):
    rows, _ = full_pool
    r = next(x for x in rows if x["difficulty"] == "easy" and x["L"] == 2)
    d = tmp_path / "data/easy"                      # 也接受 <root>/<难度>/ 布局
    d.mkdir(parents=True)
    with h5py.File(d / R.H5_NAME, "w") as h5:
        e0, _ = write_episode(h5, r, exec_lens=[60, 45])
    res = R.run_points(_save_pool(tmp_path, [r]), tmp_path / "data")
    assert "irregular=1" in res["verdicts"]["GEN_SPEC"] and "mismatches=0" in res["verdicts"]["GEN_SPEC"]
    pts = {p["u"]: p for p in res["points"]}
    assert max(pts) < 105
    assert pts[48]["j"] == 0 and pts[48]["tau"] == pytest.approx((48 / 60) / 2)
    assert pts[64]["j"] == 1 and pts[64]["tau"] == pytest.approx((1 + 4 / 45) / 2)
    assert pts[64]["step_start"] is True and pts[48]["step_start"] is False


def test_gen_spec_mismatch_and_missing(full_pool, tmp_path):
    rows, _ = full_pool
    hard = [x for x in rows if x["difficulty"] == "hard"][:3]
    d = tmp_path / "data/RouteStick/hard"
    d.mkdir(parents=True)
    with h5py.File(d / R.H5_NAME, "w") as h5:
        write_episode(h5, hard[0])
        wrong = {"A": "B", "B": "A", "C": "D", "D": "C"}[hard[1]["moves"][0]]
        write_episode(h5, hard[1], choice_override={10: wrong})   # 示范第 1 段一帧走法不符
    res = R.run_points(_save_pool(tmp_path, hard), tmp_path / "data")
    v = res["verdicts"]["GEN_SPEC"]
    assert v.startswith("GEN_SPEC=FAIL eps=1 mismatches=1 dropped=2") and "missing=1" in v
    reasons = {x["episode"]: x["reason"] for x in res["dropped"]}
    assert "choice" in reasons[hard[1]["episode"]] and reasons[hard[2]["episode"]] == "missing"


def test_episode_bounds_demo_flip_errors(full_pool):
    rows, _ = full_pool
    r = next(x for x in rows if x["difficulty"] == "easy")
    L = r["L"]
    meta = {"demo": [True] * (50 * L) + [False] * (50 * L), "bnd": [False] * (100 * L),
            "comp": [False] * (100 * L), "choice": [r["moves"][0]] * (100 * L)}
    for j in range(L):
        meta["bnd"][50 * j] = meta["bnd"][50 * L + 50 * j] = True
        meta["choice"][50 * j:50 * j + 50] = [r["moves"][j]] * 50
        meta["choice"][50 * L + 50 * j:50 * L + 50 * j + 50] = [r["moves"][j]] * 50
    info, probs = R.episode_bounds(meta, L, r["moves"])
    assert probs == [] and info["e0"] == 50 * L
    bad = copy.deepcopy(meta)
    bad["demo"][50 * L + 3] = True
    assert any("is_video_demo" in p for p in R.episode_bounds(bad, L, r["moves"])[1])
    short = {k: v[1:] for k, v in meta.items()}               # 示范少一帧
    assert any("示范段长" in p for p in R.episode_bounds(short, L, r["moves"])[1])


def test_same_frame_pairs(full_pool, tmp_path):
    rows, _ = full_pool
    r = _revisit_row(rows)
    L, e0 = r["L"], 50 * r["L"]
    starts = r["node_slots"][:-1]
    dup = next(s for s in set(starts) if starts.count(s) >= 2)
    js = [j for j, s in enumerate(starts) if s == dup]

    def image_of(t, f, row):
        if not f["demo"] and (t - e0) % 50 < 16:
            j = (t - e0) // 50
            return 100 + row["node_slots"][j] + (5 if j == js[-1] else 0)   # 与同按钮首访差 5（≤ 8）
        return t % 200

    d = tmp_path / "data/RouteStick/xhard"
    d.mkdir(parents=True)
    with h5py.File(d / R.H5_NAME, "w") as h5:
        write_episode(h5, r, image_of=image_of)
    res = R.run_points(_save_pool(tmp_path, [r]), tmp_path / "data")
    cand = sum(c * (c - 1) // 2 for c in (starts.count(s) for s in set(starts)))
    assert res["verdicts"]["PAIRS"].startswith(f"PAIRS n={cand} ")
    kept = res["pairs"]
    assert len(kept) == cand
    assert all(p["button_slot"] == r["node_slots"][p["j_a"]] == r["node_slots"][p["j_b"]] for p in kept)
    assert all(p["front_maxdiff"] <= 8 and p["wrist_maxdiff"] <= 8 for p in kept)

    # 像素差 > 8：不进严格档
    def image_far(t, f, row):
        if not f["demo"] and (t - e0) % 50 < 16:
            return 30 * ((t - e0) // 50)
        return t % 200
    d2 = tmp_path / "data2/RouteStick/xhard"
    d2.mkdir(parents=True)
    with h5py.File(d2 / R.H5_NAME, "w") as h5:
        write_episode(h5, r, image_of=image_far)
    res2 = R.run_points(_save_pool(tmp_path, [r]), tmp_path / "data2")
    assert res2["verdicts"]["PAIRS"] == f"PAIRS n={cand} kept=0"


def test_points_cli(full_pool, tmp_path):
    rows, _ = full_pool
    r = rows[0]
    d = tmp_path / "data/RouteStick/easy"
    d.mkdir(parents=True)
    with h5py.File(d / R.H5_NAME, "w") as h5:
        write_episode(h5, r)
    out = tmp_path / "out/rsprog_points.json"
    rc = R.main(["points", "--pool", str(_save_pool(tmp_path, [r])), "--data-root", str(tmp_path / "data"),
                 "--out", str(out)])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["schema"] == R.POINTS_SCHEMA and len(doc["episodes"]) == 1 and doc["points"]


# ---------------------------------------------------------------- mem_token_segments
DEMO = [[0, 49], [50, 99], [100, 149]]          # 3 段，e0 = 150


def test_mem_frames():
    mt = np.array([0, 0, 0, 0, 0, -1])
    t = np.array([0, 49, 50, 149, 150, 0])
    out = R.mem_token_segments(mt, t, t, DEMO)
    assert out.tolist() == [1, 1, 2, 3, 0, -1]


def test_mem_motion_majority():
    mt = np.ones(6, dtype=np.int8)
    t0 = np.array([0, 17, 18, 28, 140, 160])
    t1 = t0 + 32                                  # 33 帧窗口
    out = R.mem_token_segments(mt, t0, t1, DEMO)
    # 0–32 全在段 1；17–49 有 33 帧在段 1；18–50 段 1 有 32 帧；28–60 段 1 22 帧（≥22）→ 1；
    # 140–172 段 3 只 10 帧 → 0；160–192 全在执行段 → 0
    assert out.tolist() == [1, 1, 1, 1, 0, 0]
    t0b = np.array([29, 39, 61])
    out2 = R.mem_token_segments(np.ones(3), t0b, t0b + 32, DEMO)
    # 29–61：段 1 21 帧、段 2 12 帧 → 0（过渡）；39–71：段 2 22 帧 → 2；61–93 全在段 2 → 2
    assert out2.tolist() == [0, 2, 2]


def test_mem_2d_and_padding():
    mt = np.array([[0, 1, -1], [1, 0, -1]])
    t0 = np.array([[10, 100, 0], [117, 160, 0]])
    t1 = np.array([[10, 132, 0], [149, 160, 0]])
    out = R.mem_token_segments(mt, t0, t1, DEMO)
    assert out.shape == (2, 3)
    # 117–149：段 3 共 33 帧 → 3
    assert out.tolist() == [[1, 3, -1], [3, 0, -1]]
    assert R.mem_token_segments(mt, t0, t1, DEMO, exec_start=150).tolist() == out.tolist()


def test_mem_matches_subgoal_attention_primitives():
    from scripts.interpret.analysis import subgoal_attention as S
    dti = R.demo_task_index(DEMO)
    assert len(dti) == 150 and dti[0] == 1 and dti[149] == 3
    assert S.segments_of_window(28, dti, 150, window=33) == {1: 22, 2: 11}
    assert S.segment_of_frame(75, dti, 150) == 2
