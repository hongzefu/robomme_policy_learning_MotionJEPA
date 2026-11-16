"""samep_offline_encode（契约 C1 → C2，批处理 / 分片 / 输出格式）与 samep_enc_tol（ENC_TOL 判定）的合成数据测试。"""

import json

import numpy as np
import pytest

from scripts.interpret import samep_enc_tol as T
from scripts.interpret import samep_offline_encode as O
from scripts.interpret import serve_policy_attn as SPA


def write_vframes(folder, key, starts, seed=0):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, (len(starts), 33, 256, 256, 3), dtype=np.uint8)
    np.savez(folder / f"{key}.npz", start_frames=np.asarray(starts, np.int64), frames=frames,
             meta=np.str_(json.dumps({"key": key, "scope": "all"})))
    return frames


def fake_encoder(calls=None, dim=768):
    """确定性假编码：只依赖每窗自身像素，批量与逐窗逐位相同。"""
    def encode(windows):
        if calls is not None:
            calls.append(len(windows))
        flat = windows.reshape(len(windows), -1)[:, :dim].astype(np.float32)
        return flat / 255.0 + 1.0
    return encode


def test_batching_and_output_matches_donor_format(tmp_path):
    vf = tmp_path / "vframes"
    frames = write_vframes(vf, "RouteStick_ep20000011", [0, 16, 32], seed=1)
    calls = []
    s = O.run(O.list_inputs(vf), tmp_path / "off", fake_encoder(calls), model="motion", batch=2, check_single=1)
    assert s["line"].startswith("OFFLINE_ENC=PASS model=motion windows=3 files=1")
    assert calls == [2, 1, 1]                                   # 两批 + 一次逐窗复核
    out = tmp_path / "off" / "RouteStick_ep20000011.npz"
    with np.load(out) as z:
        assert sorted(z.files) == ["counts", "embs", "start_frames"]
        assert z["start_frames"].dtype == np.int32 and z["embs"].dtype == np.float32 and z["counts"].dtype == np.int32
        np.testing.assert_array_equal(z["embs"], fake_encoder()(frames))
    donor = SPA.load_donor(tmp_path / "off", "RouteStick:20000011")      # 现有供体加载器直接可读
    assert donor["start_frames"].tolist() == [0, 16, 32] and donor["embs"].shape == (3, 768)
    assert (tmp_path / "off" / "OFFLINE_ENC.txt").read_text().startswith("OFFLINE_ENC=PASS")


def test_batch_equals_per_window(tmp_path):
    vf = tmp_path / "vframes"
    write_vframes(vf, "a", [0, 16, 32, 48, 64], seed=2)
    O.run(O.list_inputs(vf), tmp_path / "b16", fake_encoder(), model="motion", batch=16, check_single=0)
    O.run(O.list_inputs(vf), tmp_path / "b1", fake_encoder(), model="motion", batch=1, check_single=0)
    with np.load(tmp_path / "b16/a.npz") as x, np.load(tmp_path / "b1/a.npz") as y:
        np.testing.assert_array_equal(x["embs"], y["embs"])


def test_shards_disjoint_cover_and_merge(tmp_path):
    vf = tmp_path / "vframes"
    keys = [f"k{i}" for i in range(3)]
    for i, k in enumerate(keys):
        write_vframes(vf, k, [0, 16], seed=i)
    parts = [O.list_inputs(vf, (i, 2)) for i in range(2)]
    names = [p.stem for part in parts for p in part]
    assert sorted(names) == keys and len(set(names)) == 3
    out = tmp_path / "off"
    assert O.merge_shards(tmp_path / "empty", 2).startswith("OFFLINE_ENC=FAIL")   # 无片清单即 FAIL
    for i in range(2):
        if i == 1:
            assert O.merge_shards(tmp_path / "off", 2).startswith("OFFLINE_ENC=FAIL")  # 缺一片即 FAIL
            (out / "OFFLINE_ENC.txt").unlink()
        s = O.run(parts[i], out, fake_encoder(), model="wanpool", batch=4, shard=(i, 2))
        assert s["line"].startswith("OFFLINE_ENC=PASS") and not (out / "OFFLINE_ENC.txt").exists()
    line = O.merge_shards(out, 2)
    assert line.startswith("OFFLINE_ENC=PASS model=wanpool windows=6 files=3")
    assert (out / "OFFLINE_ENC.txt").read_text().strip() == line


def test_bad_input_counts_fail(tmp_path):
    vf = tmp_path / "vframes"
    write_vframes(vf, "good", [0, 16])
    np.savez(vf / "dup.npz", start_frames=np.array([0, 0]), frames=np.zeros((2, 33, 256, 256, 3), np.uint8))
    np.savez(vf / "shape.npz", start_frames=np.array([0]), frames=np.zeros((1, 32, 256, 256, 3), np.uint8))
    s = O.run(O.list_inputs(vf), tmp_path / "off", fake_encoder(), model="motion")
    assert s["bad"] == 2 and s["files"] == 1 and s["line"].startswith("OFFLINE_ENC=FAIL")


def test_compressed_c1_with_meta_and_empty_file(tmp_path):
    """对齐 S-B：savez_compressed、0 维 json meta（key 须等于文件名主干）、W=0 空文件。"""
    vf = tmp_path / "vframes"
    vf.mkdir()
    key = "RouteStick_s3100001_easy_j2_S_swing"
    frames = np.random.default_rng(3).integers(0, 256, (2, 33, 256, 256, 3), dtype=np.uint8)
    np.savez_compressed(vf / f"{key}.npz", start_frames=np.array([16, 32], np.int64), frames=frames,
                        meta=np.asarray(json.dumps({"key": key, "scope": "all", "j": 2})))
    np.savez_compressed(vf / "RouteStick_s3100001_easy_j9_Z.npz", start_frames=np.zeros(0, np.int64),
                        frames=np.zeros((0, 33, 256, 256, 3), np.uint8),
                        meta=np.asarray(json.dumps({"key": "RouteStick_s3100001_easy_j9_Z"})))
    np.savez_compressed(vf / "wrongkey.npz", start_frames=np.array([0], np.int64), frames=frames[:1],
                        meta=np.asarray(json.dumps({"key": "other"})))
    s = O.run(O.list_inputs(vf), tmp_path / "off", fake_encoder(), model="motion")
    assert s["files"] == 1 and s["empty"] == 1 and s["bad"] == 1 and "empty=1" in s["line"]
    with np.load(tmp_path / "off" / f"{key}.npz") as z:
        assert z["start_frames"].tolist() == [16, 32] and z["embs"].shape == (2, 768)
    assert not (tmp_path / "off" / "RouteStick_s3100001_easy_j9_Z.npz").exists()


def test_single_check_detects_batch_inconsistency(tmp_path):
    vf = tmp_path / "vframes"
    write_vframes(vf, "a", [0, 16, 32])

    def drifting(windows):                                       # 批量与逐窗差很大
        return fake_encoder()(windows) + (0.0 if len(windows) == 1 else 5.0)
    s = O.run(O.list_inputs(vf), tmp_path / "off", drifting, model="motion", batch=3, check_single=2)
    assert s["single_n"] == 2 and s["line"].startswith("OFFLINE_ENC=FAIL")


def test_skip_existing_reuses(tmp_path):
    vf = tmp_path / "vframes"
    write_vframes(vf, "a", [0, 16])
    O.run(O.list_inputs(vf), tmp_path / "off", fake_encoder(), model="motion")
    calls = []
    s = O.run(O.list_inputs(vf), tmp_path / "off", fake_encoder(calls), model="motion", skip_existing=True)
    assert s["reused"] == 1 and calls == [] and s["line"].startswith("OFFLINE_ENC=PASS")


def test_parse_shard():
    assert O.parse_shard("1/4") == (1, 4)
    for bad in ("4/4", "x", "-1/2", "0/0"):
        with pytest.raises(ValueError):
            O.parse_shard(bad)


# ── ENC_TOL ────────────────────────────────────────────────────────────────────

def save_rec(path, starts, embs):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, start_frames=np.asarray(starts, np.int32), embs=np.asarray(embs, np.float32))


def test_enc_tol_aligns_by_start_frame_and_passes_small_noise(tmp_path):
    rng = np.random.default_rng(0)
    e = rng.normal(size=(4, 768)).astype(np.float32)
    save_rec(tmp_path / "new/a.npz", [0, 16, 32, 48], e)
    perm = [3, 1, 0, 2]                                          # 旧录制顺序不同 + 多一个起点
    save_rec(tmp_path / "old/a.npz", [[0, 16, 32, 48][i] for i in perm] + [64],
             np.concatenate([e[perm] + 1e-3 * rng.normal(size=(4, 768)), rng.normal(size=(1, 768))]))
    pairs, missing = T.pairs_from_dirs(tmp_path / "new", tmp_path / "old")
    r = T.evaluate(pairs, missing=missing)
    assert r["ok"] and r["windows"] == 4 and r["min_cos"] > 0.999 and r["maxabs"] < 0.01
    assert r["line"].startswith("ENC_TOL=PASS windows=4")


@pytest.mark.parametrize("delta,why", [(0.2, "maxabs"), (None, "cos")])
def test_enc_tol_fails(tmp_path, delta, why):
    rng = np.random.default_rng(1)
    e = rng.normal(size=(3, 768)).astype(np.float32)
    old = e.copy()
    if delta is not None:
        old[1, 5] += delta                                       # 单元素大差：cos 仍高但 maxabs 超
    else:
        old[2] = -old[2]                                         # 方向反了
    save_rec(tmp_path / "new/a.npz", [0, 16, 32], e)
    save_rec(tmp_path / "old/a.npz", [0, 16, 32], old)
    r = T.evaluate(T.pairs_from_dirs(tmp_path / "new", tmp_path / "old")[0])
    assert not r["ok"] and r["line"].startswith("ENC_TOL=FAIL")
    if why == "maxabs":
        assert r["min_cos"] >= 0.999 and r["maxabs"] > 0.1


def test_enc_tol_sampling_cap_and_pairs_json(tmp_path):
    e = np.random.default_rng(2).normal(size=(50, 768)).astype(np.float32)
    starts = list(range(0, 800, 16))
    for i in range(8):
        save_rec(tmp_path / f"new/v{i}.npz", starts, e)
        save_rec(tmp_path / f"old/s000/RouteStick_ep{i}.npz", starts, e)
    pairs_file = tmp_path / "pairs.json"
    pairs_file.write_text(json.dumps([{"new": str(tmp_path / f"new/v{i}.npz"),
                                       "old": str(tmp_path / f"old/s000/RouteStick_ep{i}.npz")} for i in range(8)]
                                     + [[str(tmp_path / "new/nope.npz"), str(tmp_path / "old/x.npz")]]))
    rc = T.main(["--pairs", str(pairs_file), "--out", str(tmp_path / "r/ENC_TOL.json")])
    r = json.loads((tmp_path / "r/ENC_TOL.json").read_text())
    assert rc == 0 and r["windows"] == 300 and r["aligned"] == 400 and r["min_cos"] == pytest.approx(1.0)
    assert "missing_old=1" in r["line"] and (tmp_path / "r/ENC_TOL.txt").read_text().startswith("ENC_TOL=PASS")


def test_enc_tol_dim_mismatch_and_empty(tmp_path):
    save_rec(tmp_path / "new/a.npz", [0], np.ones((1, 768)))
    save_rec(tmp_path / "old/a.npz", [0], np.ones((1, 1024)))
    assert not T.evaluate(T.pairs_from_dirs(tmp_path / "new", tmp_path / "old")[0])["ok"]
    assert T.evaluate([])["line"].startswith("ENC_TOL=FAIL windows=0")
