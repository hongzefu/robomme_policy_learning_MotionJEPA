"""计划 1116（steer-speedup-1116）S-B 的 CPU 单测：变体存窗帧（契约 C1）与在线切片规则逐字节一致、
donors 读离线编码（契约 C2）、spec 写 demo_cache_map / resident_server（契约 C3）、缺参数时 spec 不变（不开仿真、不用 GPU）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
from scripts.interpret import samep_rs_variants as V
from scripts.interpret import sp_swap_prep as S
from scripts.interpret.test_sp_swap import COMMIT, _write_n_run

GRID16 = {"stride": 16, "window": 33, "demo_min_real": 17, "demo_tail_pad": "repeat_last"}
GRID33 = {"stride": 32, "window": 33, "demo_min_real": 33, "demo_tail_pad": "none"}
PREV = Path("/nfs/turbo/coe-chaijy-unreplicated/hongzefu/robomme_policy_learning_MotionJEPA-eval/v1-store/"
            "evaluation/steer-4task-1115/rs")
YAML = Path("/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/assets/vla-4task-modul-motion-50000/"
            "history_config.resolved.yaml")
# 上一轮 motion/record 实测录制（局号 → (env_seed, 难度, es, 录制 start_frames 中 < es 的示范窗起点)）
RECORDED = {0: (3100000, "easy", 100, [0, 16, 32, 48, 64, 80]),
            20: (3102000, "easy", 150, [0, 16, 32, 48, 64, 80, 96, 112, 128]),
            40: (3104300, "hard", 300, list(range(0, 273, 16)))}


@pytest.fixture(autouse=True)
def _reset_grid():
    yield
    S.apply_grid()


def synth_front(es, tag=0):
    """每帧 2×2×3：通道 0/1 编码帧号，通道 2 编码 tag（区分不同局）。"""
    n = int(es) + 1
    f = np.zeros((n, 2, 2, 3), np.uint8)
    idx = np.arange(n)
    f[:, :, :, 0] = (idx // 256)[:, None, None]
    f[:, :, :, 1] = (idx % 256)[:, None, None]
    f[:, :, :, 2] = tag
    return f


def online_windows(front, grid):
    """真实在线代码 FrameSampMemory._encode_ready_windows 首批（demo 整段）喂给编码器的 (起点, 窗口) 序列。"""
    es = len(front) - 1
    got = []
    mem = FrameSampMemory.__new__(FrameSampMemory)
    mem.motion_window, mem.motion_stride = grid["window"], grid["stride"]
    mem.demo_min_real, mem.demo_tail_pad = grid["demo_min_real"], grid["demo_tail_pad"]
    mem.motion_dim = 2
    mem.exec_start_idx = es
    mem._raw_frames = {i: np.ascontiguousarray(front[i]) for i in range(es + 1)}
    mem._history_feats_motion = {}
    mem._next_grid_start_demo = mem._next_grid_start_exec = 0
    mem.motion_encode_calls, mem.motion_encode_s = 0, 0.0

    def enc(window, f):
        got.append((int(f), window.copy()))
        return np.zeros(2, np.float32)
    mem.motion_enc = enc
    mem._encode_ready_windows(last_frame=es)
    return [(f, w) for f, w in got if f < es]


def load_c1(path):
    with np.load(path) as z:
        return z["start_frames"].copy(), z["frames"].copy(), json.loads(str(z["meta"]))


# ---------------------------------------------------------------- 切片规则 vs 在线
@pytest.mark.parametrize("grid", [GRID16, GRID33])
@pytest.mark.parametrize("es", [100, 150, 200, 300, 350, 500])
def test_demo_full_frames_bytewise_equal_online(tmp_path, grid, es):
    front = synth_front(es, tag=7)
    online = online_windows(front, grid)
    starts = V.demo_window_starts(es, grid)
    assert starts == [f for f, _ in online]
    path, n = V.save_window_frames(tmp_path, "k", front, starts, es, grid, {"scope": "demo_full"})
    sf, frames, meta = load_c1(path)
    assert sf.dtype == np.int64 and frames.dtype == np.uint8 and n == len(starts)
    assert sf.tolist() == starts and frames.shape == (len(starts), grid["window"], 2, 2, 3)
    for (f, w), got in zip(online, frames):
        assert got.tobytes() == w.tobytes()                       # 逐字节等于在线编码器收到的输入（含补帧）
    assert meta["key"] == "k" and meta["es"] == es and len(meta["window_frame_ids"]) == len(starts)


def test_tail_pad_repeats_es_minus_1():
    ids = V.window_frame_ids(80, 100, GRID16)
    assert ids[:20] == list(range(80, 100)) and ids[20:] == [99] * 13 and len(ids) == 33
    with pytest.raises(RuntimeError, match="真实帧数"):
        V.window_frame_ids(96, 100, GRID16)                      # 只剩 4 帧真实帧
    with pytest.raises(RuntimeError, match="真实帧数"):
        V.window_frame_ids(80, 100, GRID33)


def test_segment_windows_match_sp_swap_prep():
    S.apply_grid({"stride": 16, "window": 33, "demo_min_real": 17})
    for es, segs in ((100, [(1, 50), (51, 100)]), (150, [(1, 50), (51, 100), (101, 150)])):
        for seg in segs:
            assert V.segment_windows(seg, es, GRID16) == tuple(S.window_sets(seg, es))
    assert V.segment_windows((51, 100), 150, GRID16) == ([32, 48, 64, 80, 96], [64])


def test_recorded_starts_match_rule():
    """上一轮 N 录制（motion/record）的示范窗起点 = 本规则算出的起点（快照常量；NFS 可见时另读原文件对拍）。"""
    grid = V.window_grid_from_yaml(YAML) if YAML.is_file() else GRID16
    assert {k: grid[k] for k in GRID16} == GRID16
    for ep, (_, _, es, want) in RECORDED.items():
        assert V.demo_window_starts(es, grid) == want
        path = PREV / "motion/record" / f"RouteStick_ep{ep}.npz"
        if path.is_file():
            with np.load(path) as z:
                rec = [int(s) for s in z["start_frames"] if int(s) < es]
            assert rec == want == V.demo_window_starts(es, grid)
    pool = PREV / "pool.json"
    if pool.is_file():
        by = {int(e["episode_id"]): e for e in json.loads(pool.read_text())["episodes"]}
        for ep, (seed, diff, es, _) in RECORDED.items():
            assert (by[ep]["env_seed"], by[ep]["difficulty"], by[ep]["es"]) == (seed, diff, es)


def test_window_grid_from_yaml(tmp_path):
    (tmp_path / "history_config.resolved.yaml").write_text(
        "motion:\n  stride: 16\n  window_frames: 33\n  demo_min_real_frames: 17\n  demo_tail_pad: repeat_last\n")
    (tmp_path / "50000").mkdir()
    g = V.window_grid_from_yaml(tmp_path / "50000")
    assert {k: g[k] for k in GRID16} == GRID16
    (tmp_path / "bad.yaml").write_text("motion:\n  stride: 16\n  window_frames: 33\n  demo_min_real_frames: 17\n")
    with pytest.raises(SystemExit, match="同时提供"):
        V.window_grid_from_yaml(tmp_path / "bad.yaml")
    (tmp_path / "bad2.yaml").write_text(
        "motion:\n  stride: 16\n  window_frames: 33\n  demo_min_real_frames: 17\n  demo_tail_pad: none\n")
    with pytest.raises(ValueError, match="契约不合法"):
        V.window_grid_from_yaml(tmp_path / "bad2.yaml")


def test_variant_key():
    assert V.variant_key(3100000, "easy", 2, "S_side") == "RouteStick_s3100000_easy_j2_S_side"
    assert V.variant_key(3100000, "easy") == "RouteStick_s3100000_easy_demo_full"
    with pytest.raises(ValueError):
        V.variant_key(1, "easy", 1, "Z")


# ---------------------------------------------------------------- process_episode 存窗帧（假仿真）
def fake_reset(seed, difficulty, spec=None, sampling=None, return_front=False):
    """按 spec 合成示范：段 j 内容帧的摘要随前 j 步路线与绕向变；front 通道 2 编码（局, 组）以便区分。"""
    sp = spec or V.rebuild_spec(seed, difficulty)
    L = int(sp["objects"]["L"])
    nodes, dirs = list(sp["actions"]["nodes"]), list(sp["actions"]["directions"])
    n = 1 + V.SEGLEN * L
    seg = [0] + [(k - 1) // V.SEGLEN + 1 for k in range(1, n)]
    hashes = [f"{nodes[:s + 1]}{dirs[:s]}" for s in seg]
    labels = [k // V.SEGLEN + 1 for k in range(n - 1)] + [L + 3]
    out = {"hashes": hashes, "labels": labels, "eef": {"dirs": dirs}, "cell_xy": {}, "cells": nodes, "dirs": dirs,
           "difficulty": difficulty, "n_frames": n}
    if return_front:
        tag = (hash((tuple(nodes), tuple(dirs))) % 200) + 1
        out["front"] = synth_front(n - 1, tag=tag)
    return out


def test_process_episode_saves_c1(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "reset_demo", fake_reset)
    monkeypatch.setattr(V, "native_sampling", lambda difficulty=None, backtrack=False: {"backtrack": bool(backtrack)})
    monkeypatch.setattr(V, "demo_swings", lambda eef, segs, cells, ref, xy: (list(eef["dirs"]), list(eef["dirs"])))
    item = {"seed": 3100010, "difficulty": "medium", "episode_id": 0}
    plain = V.process_episode(item)
    vf = {"dir": str(tmp_path / "vframes"), "grid": GRID16}
    rows = V.process_episode(item, vframes=vf)
    assert [{k: v for k, v in r.items() if not k.startswith("vframes")} for r in rows] == plain   # 只多 vframes 字段
    base = rows[0]
    assert base["usable"] and base["vframes_windows"] == len(V.demo_window_starts(200, GRID16))
    sf, frames, meta = load_c1(base["vframes"])
    online = online_windows(fake_reset(3100010, "medium", return_front=True)["front"], GRID16)
    assert meta["scope"] == "demo_full" and sf.tolist() == [f for f, _ in online]
    assert all(a.tobytes() == w.tobytes() for a, (_, w) in zip(frames, online))
    used = [r for r in rows[1:] if r.get("usable")]
    assert used and all(r.get("vframes") for r in used)
    assert not any(r.get("vframes") for r in rows[1:] if not r.get("usable"))
    for r in used:
        sf, frames, meta = load_c1(r["vframes"])
        w_all, w_core = V.segment_windows(r["segments"][r["j"] - 1], 200, GRID16)
        assert sf.tolist() == w_all and meta["core_starts"] == w_core and meta["scope"] == "all"
        assert meta["window_scope"] == ["core" if s in w_core else "all" for s in w_all]
        assert (meta["j"], meta["group"], meta["seed"]) == (r["j"], r["group"], 3100010)
        assert Path(r["vframes"]).name == V.variant_key(3100010, "medium", r["j"], r["group"]) + ".npz"
        vspec = r["spec"]
        front = fake_reset(3100010, "medium", spec=vspec, return_front=True)["front"]
        on = dict(online_windows(front, GRID16))
        assert all(frames[i].tobytes() == on[s].tobytes() for i, s in enumerate(w_all))
    assert V.vframes_verdict(rows).startswith("VFRAMES=PASS ")


def test_cli_build_requires_yaml(tmp_path):
    with pytest.raises(SystemExit, match="history-yaml"):
        V.main(["build", "--bench-src", "/x", "--episodes", "1:easy", "--out", str(tmp_path / "o"),
                "--save-window-frames"])


# ---------------------------------------------------------------- specs / donors（同一局，合成 N 与离线供体）
SEEDS = (2400000, 2400100, 2400200)


def _scan(seed, i):
    return {"task": "RouteStick", "env_seed": seed, "difficulty": "easy", "scan_index": i, "m": 2, "status": "ok",
            "es": 100, "segments": [[0, 49], [50, 99]], "segment_problem": None, "replay_match": True,
            "cells": [4, 6, 4], "swings": ["cw", "ccw"], "theta": 0.0}


def _variants(path):
    rows = []
    for seed in SEEDS:
        spec = {"task": "RouteStick", "actions": {"node_slots": [2, 3, 2], "nodes": [4, 6, 4],
                                                  "directions": ["clockwise", "counterclockwise"]}}
        rows.append({"kind": "base", "seed": seed, "difficulty": "easy", "usable": True, "spec": spec,
                     "sampling_config": {"s": 1}})
        for j in (1, 2):
            v = json.loads(json.dumps(spec))
            v["actions"]["directions"][j - 1] = V.flip_dir(v["actions"]["directions"][j - 1])
            rows.append({"kind": "variant", "seed": seed, "difficulty": "easy", "j": j, "group": "S_swing",
                         "usable": True, "spec": v, "sampling_config": {"s": 2}, "demo_swings_phys": ["ccw", "cw"]})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_root(tmp_path, name, yaml_path, spec_extra=()):
    O = tmp_path / name
    O.mkdir()
    S.P._write_json(O / "scan_RouteStick.json",
                    {"task": "RouteStick", "entries": [_scan(s, i) for i, s in enumerate(SEEDS)]})
    S.main(["pool", "--out", str(O), "--tasks", "RouteStick", "--rs-quota-by-difficulty", "easy=3"])
    _variants(O / "variants.jsonl")
    M = O / "motion"
    rc = S.main(["specs", "--out", str(M), "--model", "motion", "--commit", COMMIT, "--history-yaml", str(yaml_path),
                 "--same-episode", str(O / "variants.jsonl"), "--log-dir", str(tmp_path / "logs"), *spec_extra])
    assert rc == 0
    return O, M


def write_n_runs(O, M):
    plan = json.loads((M / "plan.json").read_text())
    _, by_key = S.load_pool(O)
    for r in plan["runs"]:
        if r["stage"] == "n":
            e = dict(by_key[S.episode_key("RouteStick", r["ep"])], task="RouteStick")
            _write_n_run(r, e)
    return plan, by_key


def write_offline(folder, plan, by_key, dim=4):
    """按契约 C2 写离线供体：S 组存第 j 段相交窗，demo_full 存全部示范窗；值 = 1000·(j or 0) + 起点 + 局号。"""
    folder.mkdir(parents=True, exist_ok=True)
    for r in plan["samep"]["vrec"]:
        e = by_key[r["recv"]]
        key = V.variant_key(e["env_seed"], e["difficulty"], r["j"], None if r["j"] is None else r["vgroup"])
        if r["j"] is None:
            starts = V.demo_window_starts(100, GRID16)
        else:
            starts = V.segment_windows(S.content_segments(e["segments"])[r["j"] - 1], 100, GRID16)[0]
        embs = np.stack([np.full(dim, 1000 * (r["j"] or 0) + s + e["episode_id"], np.float32) for s in starts])
        np.savez(folder / f"{key}.npz", start_frames=np.asarray(starts, np.int32), embs=embs,
                 counts=np.ones(len(starts), np.int32))
    (folder / "OFFLINE_ENC.txt").write_text("OFFLINE_ENC=PASS model=motion windows=1 files=1\n")


def yaml_file(tmp_path):
    p = tmp_path / "history_config.resolved.yaml"
    p.write_text("motion:\n  stride: 16\n  window_frames: 33\n  demo_min_real_frames: 17\n  demo_tail_pad: repeat_last\n")
    return p


def specs_of(M, stage_prefix=None):
    out = {}
    for p in sorted((M / "specs").glob("*.json")):
        if stage_prefix is None or p.name.startswith(stage_prefix):
            out[p.name] = json.loads(p.read_text())
    return out


def test_no_vrec_and_c3_specs(tmp_path):
    y = yaml_file(tmp_path)
    _, M0 = make_root(tmp_path, "A", y)
    O1, M1 = make_root(tmp_path, "B", y, ("--no-vrec", "--demo-cache-from", "REC", "--resident-server"))
    p0, p1 = json.loads((M0 / "plan.json").read_text()), json.loads((M1 / "plan.json").read_text())
    assert any(r["stage"] == "vrec" for r in p0["runs"]) and not any(r["stage"] == "vrec" for r in p1["runs"])
    assert "no_vrec" not in p0["samep"] and "c3" not in p0["samep"] and p1["samep"]["no_vrec"] is True
    assert [(r["recv"], r["j"], r["vgroup"]) for r in p0["samep"]["vrec"]] == \
        [(r["recv"], r["j"], r["vgroup"]) for r in p1["samep"]["vrec"]]
    assert all(r["record"] is None for r in p1["samep"]["vrec"])
    n0, n1 = specs_of(M0, "noise-"), specs_of(M1, "noise-")
    assert n0.keys() == n1.keys() and n0
    for name in n0:
        a, b = n0[name], n1[name]
        assert "demo_cache_map" not in a and "resident_server" not in a
        assert b["resident_server"] is True
        eps = json.loads(Path(b["manifest"]).read_text())["episodes"]
        assert sorted(b["demo_cache_map"]) == sorted(S.episode_key(e["task_name"], e["episode_id"]) for e in eps)
        for k, v in b["demo_cache_map"].items():
            assert Path(v).is_absolute() and Path(v).name == S.record_name(*S.split_key(k))
        strip = {k: v for k, v in b.items() if k not in ("demo_cache_map", "resident_server")}
        assert json.dumps(strip, sort_keys=True).replace(str(O1), "<O>") == \
            json.dumps(a, sort_keys=True).replace(str(M0.parent), "<O>")
    with pytest.raises(SystemExit, match="只用于 --same-episode"):
        S.main(["specs", "--out", str(M1), "--model", "motion", "--commit", COMMIT, "--no-vrec", "--overwrite"])


def test_donors_offline_matches_vrec_structure(tmp_path):
    y = yaml_file(tmp_path)
    # 旧路径：vrec 录制（逐窗写成与离线同值）→ donors 缺省参数
    O0, M0 = make_root(tmp_path, "A", y)
    plan0, by0 = write_n_runs(O0, M0)
    off0 = tmp_path / "offA"
    write_offline(off0, plan0, by0)
    for r in plan0["samep"]["vrec"]:
        e = by0[r["recv"]]
        key = V.variant_key(e["env_seed"], e["difficulty"], r["j"], None if r["j"] is None else r["vgroup"])
        with np.load(off0 / f"{key}.npz") as z:
            Path(r["record"]).parent.mkdir(parents=True, exist_ok=True)
            np.savez(r["record"], start_frames=z["start_frames"], embs=z["embs"], counts=z["counts"])
    assert S.main(["donors", "--out", str(M0), "--same-episode"]) == 0
    # 新路径：--no-vrec + --offline-donors + C3
    O1, M1 = make_root(tmp_path, "B", y, ("--no-vrec",))
    plan1, by1 = write_n_runs(O1, M1)
    off1 = tmp_path / "offB"
    write_offline(off1, plan1, by1)
    with pytest.raises(SystemExit, match="offline-donors"):
        S.main(["donors", "--out", str(M1), "--same-episode"])
    rc = S.main(["donors", "--out", str(M1), "--same-episode", "--offline-donors", str(off1),
                 "--demo-cache-from", str(M1 / "record"), "--resident-server"])
    assert rc == 0
    d0 = sorted((M0 / S.SAMEP_DONOR_DIR).glob("*.npz"))
    d1 = sorted((M1 / S.SAMEP_DONOR_DIR).glob("*.npz"))
    assert [p.name for p in d0] == [p.name for p in d1] and d0
    groups = {S.parse_samep(int(p.stem.split("_ep")[1]))[2] for p in d1}
    assert {"S_swing", "S_swing_core", "Z"} <= groups                      # core 组也从离线目录生成
    ref = sorted((PREV / "motion/donors_samep").glob("*.npz"))[:1] if (PREV / "motion/donors_samep").is_dir() else []
    for a, b in zip(d0, d1):
        with np.load(a) as za, np.load(b) as zb:
            assert za.files == zb.files == ["start_frames", "embs", "counts"]
            for k in za.files:
                assert za[k].dtype == zb[k].dtype and np.array_equal(za[k], zb[k])
            for path in ref:                                              # 与上一轮 donors_samep 字段 / dtype 一致
                with np.load(path) as zr:
                    assert zr.files == zb.files and all(zr[k].dtype == zb[k].dtype for k in zr.files)
    s0, s1 = specs_of(M0, "samep"), specs_of(M1, "samep")
    assert s0.keys() == s1.keys() and s0
    for name in s0:
        a, b = s0[name], s1[name]
        assert "demo_cache_map" not in a and "resident_server" not in a
        assert b["resident_server"] is True and sorted(b["demo_cache_map"]) == sorted(b["swap_map"])
        for k, v in b["demo_cache_map"].items():
            assert v == str((M1 / "record" / S.record_name(*S.split_key(k))).resolve()) and Path(v).is_file()
        strip = {k: v for k, v in b.items() if k not in ("demo_cache_map", "resident_server")}
        assert json.dumps(strip, sort_keys=True).replace(str(O1), "<O>") == \
            json.dumps(a, sort_keys=True).replace(str(O0), "<O>")
    p1 = json.loads((M1 / "plan.json").read_text())
    assert p1["samep"]["offline_donors"] == str(off1.resolve()) and p1["samep"]["c3_donors"]["resident_server"] is True
    assert "offline_donors" not in json.loads((M0 / "plan.json").read_text())["samep"]


def test_offline_meta_key_mismatch(tmp_path):
    p = tmp_path / "RouteStick_s1_easy_demo_full.npz"
    np.savez(p, start_frames=np.asarray([0], np.int32), embs=np.zeros((1, 2), np.float32),
             meta=np.asarray(json.dumps({"key": "other"})))
    with pytest.raises(SystemExit, match="meta.key"):
        S.load_offline_record(p, "RouteStick_s1_easy_demo_full")
    assert S.load_offline_record(tmp_path / "nope.npz", "x") is None
