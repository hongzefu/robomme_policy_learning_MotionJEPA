"""rsprog_probe / rsprog_attn 的合成数据单测（计划 1117 B1 第 4、5 行、B4 R3）：只用 CPU，不读任何真实数据。

合成数据带已知信号：模型 A 动作组第 10 层线性编码 τ 与下一步走法、第 17 层编码当前步走法（正对照）；前缀 image
只编码执行偏移 u（时间）；模型 B 只有第 17 层的当前步走法。注意力：执行第 j 步时 attn_mem 对示范第 j 段的 token 加权
（对角富集）。R1 的 mem_token_segments 尚未合入，这里用本地 stub（按计划 B4 R1 行的归段规则）。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.interpret.analysis import rsprog_attn as RA
from scripts.interpret.analysis import rsprog_probe as RP

STEP = 50
W, WA, WM = 24, 24, 16
DIFFS = ("easy", "medium", "hard", "xhard")
L_OF = {"easy": (2, 3), "medium": (3, 4), "hard": (3, 4), "xhard": (4, 4)}
MOVES = ("A", "B", "C", "D")
N_EP = 40
ENRICH = 4.0


def stub_segments(mem_type, mem_t0, mem_t1, demo_bounds):
    """计划 B4 R1 行的归段规则：帧按所在段；motion 窗 ≥ 22/33 帧落在某示范段才归该段，否则 −2（过渡）；执行 −1。"""
    out = np.full(len(mem_type), -1, dtype=int)
    for i, (kind, t0, t1) in enumerate(zip(mem_type, mem_t0, mem_t1)):
        for seg, (s, e) in enumerate(demo_bounds):
            if kind == 0 and s <= t0 < e:
                out[i] = seg
            if kind == 1:
                inside = min(t1 + 1, e) - max(t0, s)
                if inside >= 22:
                    out[i] = seg
                elif inside > 0 and out[i] == -1:
                    out[i] = -2
    return out


def make_data(tmp_path, n_episodes=N_EP, seed=0):
    rng = np.random.default_rng(seed)
    basis_tau = rng.normal(size=WA)
    basis_move = rng.normal(size=(4, WA)) * 2.0
    basis_cur = rng.normal(size=(4, WA)) * 2.0
    basis_u = rng.normal(size=W)
    episodes, pairs = [], []
    for k in range(n_episodes):
        difficulty = DIFFS[k % 4]
        L = int(rng.integers(L_OF[difficulty][0], L_OF[difficulty][1] + 1))
        moves = [MOVES[int(v)] for v in rng.integers(0, 4, size=L)]
        demo = [(STEP * i, STEP * (i + 1)) for i in range(L)]
        e0 = STEP * L
        points = []
        for u in range(0, STEP * L, 16):
            j = u // STEP
            points.append({"t": e0 + u, "u": u, "j": j, "tau": u / (STEP * L), "remaining": L - j, "cur_move": moves[j],
                           "next_move": moves[j + 1] if j + 1 < L else None, "button": int(j % 5)})
        starts = [p["u"] for p in points if p["u"] % STEP < 16]
        if len(starts) >= 2:
            pairs.append({"k": k, "u_a": starts[0], "u_b": starts[1]})
        episodes.append({"k": k, "seed": 3400000 + 100 * k, "difficulty": difficulty, "L": L, "route": moves,
                         "demo_bounds": [list(b) for b in demo], "exec_bounds": [[e0 + STEP * i, e0 + STEP * (i + 1)] for i in range(L)],
                         "e0": e0, "points": points})
        for model in ("A", "B"):
            write_npz(tmp_path / model / f"episode_{k}.npz", model, episodes[-1], rng, basis_tau, basis_move, basis_cur, basis_u)
    path = tmp_path / "rsprog_points.json"
    path.write_text(json.dumps({"episodes": episodes, "same_frame_pairs": pairs}), encoding="utf-8")
    return path


def memory_tokens(ep, u):
    """当前推理点可见的记忆：示范与执行帧（每 10 帧一个 token）+ motion 窗（33 帧、步长 16）。"""
    e0, t = ep["e0"], ep["e0"] + u
    types, t0s, t1s = [], [], []
    for frame in range(0, t + 1, 10):
        types.append(0), t0s.append(frame), t1s.append(frame)
    for start in range(0, max(t - 32, 0) + 1, 16):
        types.append(1), t0s.append(start), t1s.append(start + 32)
    return np.asarray(types), np.asarray(t0s), np.asarray(t1s)


def write_npz(path, model, ep, rng, basis_tau, basis_move, basis_cur, basis_u):
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = ep["points"]
    N = len(pts)
    tokens = [memory_tokens(ep, p["u"]) for p in pts]
    M = max(len(t[0]) for t in tokens)
    prefix = rng.normal(size=(N, 19, 2, W)).astype(np.float32)
    action = rng.normal(size=(N, 18, WA)).astype(np.float32)
    memmod = rng.normal(size=(N, 18, WA)).astype(np.float32)
    attn_mem = np.zeros((N, 18, M), np.float32)
    mem_type = np.full((N, M), -1, np.int8)
    mem_t0 = np.zeros((N, M), np.int32)
    mem_t1 = np.zeros((N, M), np.int32)
    mem_len = np.zeros(N, np.int32)
    for i, p in enumerate(pts):
        code_cur = MOVES.index(p["cur_move"])
        prefix[i, :, 0] += 3.0 * (p["u"] / 100.0) * basis_u
        action[i, 17] += basis_cur[code_cur] * 2.0
        if model == "A":
            action[i, 10] += 6.0 * p["tau"] * basis_tau
            if p["next_move"] is not None:
                action[i, 10] += basis_move[MOVES.index(p["next_move"])]
        types, t0, t1 = tokens[i]
        n = len(types)
        segs = stub_segments(types, t0, t1, ep["demo_bounds"])
        weight = np.ones(n)
        lo, hi = ep["demo_bounds"][p["j"]]
        on_diag = ((types == 0) & (t0 >= lo) & (t0 < hi)) | ((types == 1) & (segs == p["j"]))
        weight[on_diag] *= ENRICH
        attn = weight[None, :] * rng.uniform(0.8, 1.2, size=(18, n))
        attn_mem[i, :, :n] = attn / attn.sum(-1, keepdims=True)
        mem_type[i, :n], mem_t0[i, :n], mem_t1[i, :n], mem_len[i] = types, t0, t1, n
    attn_main = rng.dirichlet([3, 2, 3], size=(N, 18)).astype(np.float32)
    film = rng.uniform(0.05, 0.2, size=(N, 18, 1))
    film_rel = np.concatenate([film, film * 0.5], axis=-1).astype(np.float32)
    np.savez(path, t=np.asarray([p["t"] for p in pts], np.int64), u=np.asarray([p["u"] for p in pts], np.int64),
             hidden_prefix=prefix, hidden_action=action, hidden_memmod=memmod, attn_mem=attn_mem, mem_type=mem_type,
             mem_t0=mem_t0, mem_t1=mem_t1, mem_len=mem_len, attn_main=attn_main, film_rel=film_rel,
             mem_frame_mean=rng.normal(size=(N, WM)).astype(np.float32),
             mem_motion_mean=rng.normal(size=(N, WM)).astype(np.float32),
             mem_motion_latest=rng.normal(size=(N, WM)).astype(np.float32), memmod_maxabs=np.float32(2e-4))


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("rsprog")
    return root, make_data(root)


@pytest.fixture(scope="module")
def probe(data):
    root, points = data
    summary, predictions, samples = RP.run(points, root, root / "probe-out", workers=1, shuffles=4, repeats=100)
    return root, summary, predictions, samples


# ---------------------------------------------------------------- 探针

def test_points_load_and_pairs(data):
    _, points = data
    loaded = RP.load_points(points)
    assert len(loaded["rows"]) > 300
    assert all(r["next"] is None or r["next"] in MOVES for r in loaded["rows"])
    assert len(loaded["pairs_raw"]) == N_EP


def test_no_group_leak_and_folds(probe):
    _, summary, _, samples = probe
    assert summary["group_leak"] == 0
    for key in samples["keys"]:
        assert len(set(samples["outer"][samples["groups"] == str(key)].tolist())) == 1
    assert set(samples["outer"].tolist()) == set(range(RP.OUTER_FOLDS))
    assert summary["counts"]["dropped"] == 0


def test_probe_reads_planted_signal(probe):
    _, summary, _, _ = probe
    action_a = summary["models"]["A"]["action"]
    curve = [c["mean"] for c in action_a["tau"]["curve"]]
    assert int(np.argmax(curve)) == 10 and curve[10] > 0.8
    assert action_a["tau"]["selected"]["mean"] > 0.8
    assert set(action_a["tau"]["layer_by_fold"]) == {10}
    assert action_a["next"]["selected"]["mean"] > 0.85
    # 时间基线：u 独热读得出一部分 τ，但不如动作组；oracle (u, L) 接近 1
    base = summary["time_baseline"]["u"]["tau"]["mean"]
    assert 0.2 < base < action_a["tau"]["selected"]["mean"]
    assert summary["time_baseline"]["uL"]["tau"]["mean"] > 0.9
    assert action_a["tau"]["minus_u"]["mean"] > 0.05 and action_a["tau"]["minus_u"]["ci95"][0] > 0
    # 下一步走法时间读不出（四类随机），基线接近随机
    assert summary["time_baseline"]["u"]["next"]["mean"] < 0.5
    # 模型 B 没有 τ 信号
    assert summary["models"]["B"]["action"]["tau"]["selected"]["mean"] < base + 0.1
    # 按难度分层与同帧异史都有
    assert set(action_a["tau"]["by_difficulty"]) == set(DIFFS)
    assert action_a["next"]["pair_next"]["n_pairs"] > 0
    assert action_a["next"]["pair_next"]["mean"] > summary["time_baseline"]["u"]["pair_next"]["mean"]


def test_prefix_reads_time_not_route(probe):
    _, summary, _, _ = probe
    image = summary["models"]["A"]["prefix_image"]
    assert max(c["mean"] for c in image["j"]["curve"]) > 0.5           # 前缀 image 编码了 u
    assert image["next"]["selected"]["mean"] < 0.5                     # 但读不出下一步走法


def test_shuffle_control_destroys_signal(probe):
    _, summary, _, _ = probe
    shuffled = summary["models"]["A"]["action"]["shuffled"]
    assert len(shuffled["next"]["values"]) == 4
    assert shuffled["next"]["mean"] < 0.5
    assert shuffled["tau"]["mean"] < summary["models"]["A"]["action"]["tau"]["selected"]["mean"]


def test_positive_control_and_verdicts(probe):
    _, summary, _, _ = probe
    assert summary["models"]["A"]["positive_control"]["pass"]
    lines = summary["verdicts"]
    assert "GROUP_LEAK=PASS leak=0" in lines
    assert any(line.startswith("POSITIVE_CONTROL=PASS model=A") for line in lines)
    for label in RP.MAIN_LABELS:
        assert any(line.startswith(f"PROBE=PASS model=A label={label}") for line in lines)


def test_probe_figures_and_json(probe):
    root, summary, _, _ = probe
    out = root / "probe-out"
    for stem in ("progress-layerwise", "progress-fit", "progress-fig4"):
        for suffix in ("png", "pdf"):
            assert (out / f"{stem}.{suffix}").stat().st_size > 1000
    stored = json.loads((out / "rsprog-probe-summary.json").read_text(encoding="utf-8"))
    assert stored["group_leak"] == 0 and "A" in stored["models"]


def test_r2_episode_decomposition():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(6).astype(str), [3, 5, 2, 7, 4, 6])
    y = rng.normal(size=len(groups))
    p = y + rng.normal(scale=0.5, size=len(groups))
    values, _ = RP.r2_episodes(y, p, groups)
    w = RP.episode_weights(groups)
    mean = (w * y).sum() / w.sum()
    r2 = 1 - (w * (y - p) ** 2).sum() / (w * (y - mean) ** 2).sum()
    assert np.isclose(values.mean(), r2)


# ---------------------------------------------------------------- 注意力

@pytest.fixture(scope="module")
def attn(data):
    root, points = data
    summary = RA.run(points, root, root / "attn-out", segmenter=stub_segments, repeats=100)
    return root, summary


def test_attn_agg_pass(attn):
    _, summary = attn
    assert summary["verdict"].startswith("ATTN_AGG=PASS")
    for check in summary["checks"].values():
        assert check["rowsum_maxerr"] <= RA.ROWSUM_TOL
        assert check["memmod_maxabs"] == pytest.approx(2e-4, rel=1e-3)


def test_panel_b_diagonal_enrichment(attn):
    _, summary = attn
    for model in ("A", "B"):
        for key in ("motion:overlap", "frame:single", "motion:single"):
            panel = summary["panel_b"][model][key]
            assert panel["D"]["mean"] > RA.D_THRESHOLD and panel["D"]["ci95"][0] > 1.0
            matrix = np.asarray(panel["matrix"]["enrichment"], dtype=float)
            diag = np.nanmean(np.diag(matrix))
            off = np.nanmean(matrix[~np.eye(len(matrix), dtype=bool)])
            assert diag > 1.0 > off
            assert panel["min_cell_eps"] >= RA.MIN_CELL_EPISODES
        assert summary["panel_b"][model]["motion:overlap"]["main"]
        assert len(summary["panel_b"][model]["frame:single"]["D_by_layer"]) == RA.LAYERS


def test_panel_a_rows(attn):
    _, summary = attn
    a = summary["panel_a"]["A"]["layers"]
    assert len(a["ratio_motion"]) == RA.LAYERS
    main = np.stack([a[f"main_{name}"] for name in RA.MAIN_ROWS])
    assert np.allclose(main.sum(0), 100.0, atol=1e-3)
    assert np.allclose(np.asarray(a["film_motion"]) / np.asarray(a["film_total"]), 0.5, atol=1e-4)


def test_ratio_is_one_under_uniform_attention():
    n_frame, n_motion = 30, 10
    ep = {"mem_len": np.array([40]), "mem_type": np.array([[0] * n_frame + [1] * n_motion]),
          "mem_t0": np.array([list(range(0, 300, 10)) + list(range(0, 160, 16))]),
          "mem_t1": np.array([list(range(0, 300, 10)) + list(range(32, 192, 16))]),
          "attn_mem": np.full((1, 18, 40), 1 / 40), "attn_main": np.full((1, 18, 3), 1 / 3)}
    readout = RA.point_readouts(ep, 0, [(0, 50), (50, 100), (100, 150)], stub_segments)
    assert np.allclose(readout["ratio"]["motion"], 1.0) and np.allclose(readout["ratio"]["frame"], 1.0)
    share = readout["b"][("frame", "single")]
    assert np.allclose(share["share"] / share["uniform"], 1.0)


def test_attn_figures(attn):
    root, _ = attn
    out = root / "attn-out"
    for stem in ("attn-a-routestick", "attn-a-routestick-B", "attn-b-routestick"):
        for suffix in ("png", "pdf"):
            assert (out / f"{stem}.{suffix}").stat().st_size > 1000
    assert (out / "rsprog-attn-summary.json").exists()
