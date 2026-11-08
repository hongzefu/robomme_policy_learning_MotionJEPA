"""1108 计划 G2 与 S1 闸门单测：768 池化公式、导出确定性、G3 校验器、MotionMeta / _motion_gates / 训练快照的池化分支。

全部 CPU、合成夹具；latent 夹具按 scripts/dataset/wan/extract_wan.py 的落盘格式合成：
``<段>.bin``（num_grid × (9,16,32,32) f32 裸字节）+ ``<段>.bin.sha256`` + ``<段>.metadata.json``（schema 3，含 rows），
成员清单按 pack_and_hash.py 的 ``sha256-source-files.txt`` 口径（``<sha256>  wan-latents/<名>``）。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mme_vla_suite.datastore import motion_store as ms  # noqa: E402
from mme_vla_suite.datastore.manifest import manifest_sha256  # noqa: E402
from mme_vla_suite.shared import wan_pool768 as wp  # noqa: E402

EXPORT = ROOT / "scripts/dataset/export_wan_pool768.py"
VERIFY = ROOT / "scripts/dataset/verify_wan_pool768.py"


def _ref(z: np.ndarray) -> np.ndarray:
    a = torch.from_numpy(z.astype(np.float64))[1:9].mean(dim=0)
    return F.adaptive_avg_pool2d(a, (6, 8)).reshape(1, 768).numpy()


def _say(capsys, line: str) -> None:
    with capsys.disabled():
        print(f"\n{line}")


# ══ G2：公式 ══════════════════════════════════════════════════════════════════


def test_pool_contract(capsys):
    rng = np.random.default_rng(1108)
    cases, max_abs = 0, 0.0
    for scale in (1e-3, 1.0, 3.0, 10.0):
        for _ in range(6):
            z = (rng.standard_normal(wp.POOL768_INPUT_SHAPE) * scale).astype(np.float32)
            keep = z.copy()
            out = wp.pool_raw_wan_latent_768(z)
            assert out.shape == wp.POOL768_SHAPE and out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]
            assert out.nbytes == wp.POOL768_OUTPUT_BYTES
            assert np.array_equal(z, keep)                                  # 不改写输入
            assert out.tobytes() == wp.pool_raw_wan_latent_768(z.copy()).tobytes()   # 重跑逐位相同
            max_abs = max(max_abs, float(np.abs(out.astype(np.float64) - _ref(z)).max()))
            cases += 1
    assert max_abs <= 1e-6
    assert wp.POOL768_PARAMETER_COUNT == 0
    assert (wp.POOL768_READOUT, wp.POOL768_ENCODER_KIND) == ("wan_raw_timepool8_adaptive6x8_v1", "wan_fixedpool768")
    _say(capsys, f"POOL768_CONTRACT=PASS cases={cases} max_abs={max_abs:.3e} bins=6x8")


def test_bins_each_boundary():
    assert wp.POOL768_H_BINS == ((0, 6), (5, 11), (10, 16), (16, 22), (21, 27), (26, 32))
    assert wp.POOL768_W_BLOCK == 4
    # 指示输入逐行 / 逐列落点：与 torch adaptive_avg_pool2d 的分箱逐格相同，且每格权重恰为 1/24
    for h in range(32):
        z = np.zeros(wp.POOL768_INPUT_SHAPE, np.float32)
        z[1:, 0, h, :] = 1.0
        got = wp.pool_raw_wan_latent_768(z).reshape(16, 6, 8)[0, :, 0]
        want_bins = [i for i, (s, e) in enumerate(wp.POOL768_H_BINS) if s <= h < e]
        assert [i for i in range(6) if got[i] != 0] == want_bins
        assert np.allclose(got[want_bins], 4 / 24)
        assert np.allclose(_ref(z).reshape(16, 6, 8)[0, :, 0], got)
    assert [h for h in range(32) if sum(s <= h < e for s, e in wp.POOL768_H_BINS) == 2] == [5, 10, 21, 26]
    for w in range(32):
        z = np.zeros(wp.POOL768_INPUT_SHAPE, np.float32)
        z[1:, 3, :, w] = 1.0
        got = wp.pool_raw_wan_latent_768(z).reshape(16, 6, 8)[3]
        assert [j for j in range(8) if got[0, j] != 0] == [w // 4]
    # 第 0 个 latent 被丢弃；展平序 48c+8i+j
    z = np.zeros(wp.POOL768_INPUT_SHAPE, np.float32)
    z[0] = 1e6
    assert not wp.pool_raw_wan_latent_768(z).any()
    z = np.zeros(wp.POOL768_INPUT_SHAPE, np.float32)
    z[1:, 7, 27:32, 28:32] = 24.0                 # 第 26 行同属第 4、5 箱，这里只取仅属第 5 箱的 27–31 行
    flat = wp.pool_raw_wan_latent_768(z)[0]
    assert flat[48 * 7 + 8 * 5 + 7] == 20.0 and np.count_nonzero(flat) == 1


@pytest.mark.parametrize("bad,exc", [
    (lambda: np.zeros(wp.POOL768_INPUT_SHAPE, np.float64), TypeError),
    (lambda: np.zeros(wp.POOL768_INPUT_SHAPE, np.float32).tolist(), TypeError),
    (lambda: np.zeros((8, 16, 32, 32), np.float32), ValueError),
    (lambda: np.zeros((9, 16, 32, 32), ">f4"), TypeError),
    (lambda: np.full(wp.POOL768_INPUT_SHAPE, np.nan, np.float32), ValueError),
    (lambda: np.full(wp.POOL768_INPUT_SHAPE, 3e38, np.float32), ValueError),
])
def test_pool_rejects_bad_input(bad, exc):
    with pytest.raises(exc):
        wp.pool_raw_wan_latent_768(bad())


def test_pool_source_sha256_is_file_bytes():
    path = ROOT / "src/mme_vla_suite/shared/wan_pool768.py"
    assert wp.pool_source_sha256() == hashlib.sha256(path.read_bytes()).hexdigest()


# ══ 夹具：manifest + 旧 encoder 表 + latent 根 + 成员清单 ══════════════════════════

VAE = {"vae_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "vae_state_sha256": "9" * 64, "gpu_name": "fake"}
ENC = {"run_name": "enc-run", "checkpoint_name": "checkpoint_epoch_72.pt", "epoch": 72, "state_key": "encoder"}


def _write_json(p: pathlib.Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def build_lib(tmp: pathlib.Path, seed: int = 0) -> SimpleNamespace:
    """3 个 episode：含 demo 补帧窗、demo 不足 17 帧（0 窗）两种边界。"""
    rng = np.random.default_rng(seed)
    eps_spec = [("record_dataset_TaskA.h5", 0, 40, 110), ("record_dataset_TaskA.h5", 1, 10, 60),
                ("record_dataset_TaskB.h5", 3, 70, 120)]           # (h5, raw_ep, exec_start, num_timesteps)
    episodes = [{"global_episode_idx": g, "h5_file": h5, "raw_ep_idx": ep, "num_timesteps": nt, "exec_start_idx": es}
                for g, (h5, ep, es, nt) in enumerate(eps_spec)]
    manifest = {"version": 1, "raw_dir": "/fake/raw", "episodes": episodes}
    manifest["sha256"] = manifest_sha256(manifest)
    lib = tmp / "lib"
    man_path = lib / "meta/episode_manifest.json"
    _write_json(man_path, manifest)

    spec = ms.LAYOUT_SPECS[ms.LAYOUT]
    entries = ms.build_index_entries(manifest, spec)
    totals = ms.index_totals(entries)
    ref = lib / "motion"
    index = ms.index_payload(manifest, entries, spec=spec, layout=ms.LAYOUT, mj_repo_commit="mjc")
    index_bytes = (json.dumps(index, ensure_ascii=False, indent=1) + "\n").encode()
    (ref / "meta").mkdir(parents=True)
    (ref / ms.INDEX_RELPATH).write_bytes(index_bytes)
    rows = totals["rows"]
    table = rng.standard_normal((rows, 768)).astype(np.float32)
    (ref / ms.MOTION_TABLE_RELPATH).write_bytes(table.tobytes())
    meta = {
        "schema": ms.META_SCHEMA, "layout": ms.LAYOUT, "status": "verified", **spec.fields(),
        "byte_order": ms.BYTE_ORDER, "array_order": ms.ARRAY_ORDER, "grid_stride": ms.GRID_STRIDE,
        "window_frames": ms.WINDOW_FRAMES, "grid_origin": ms.GRID_ORIGIN, "window_direction": ms.WINDOW_DIRECTION,
        "truncation_policy": ms.TRUNCATION_POLICY, "frame_size": ms.FRAME_SIZE,
        "tables": {ms.MOTION_KEY: {"row_shape": [768], "dtype": "float32", "row_bytes": ms.MOTION_ROW_BYTES,
                                   "num_rows": rows, "relpath": ms.MOTION_TABLE_RELPATH,
                                   "byte_count": rows * ms.MOTION_ROW_BYTES,
                                   "sha256": hashlib.sha256(table.tobytes()).hexdigest()}},
        "num_rows": rows, "totals": totals, "row_order": index["row_order"],
        "manifest_sha256": manifest["sha256"], "manifest_path": str(man_path),
        "motion_index_relpath": ms.INDEX_RELPATH, "motion_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "provenance": {"vae": VAE, "encoder": dict(ENC), "mj_repo_commit": "mjc", "source_pin": {"x": 1},
                       "raw_dir": "/fake/raw", "input_manifest_sha256": "1" * 64},
    }
    _write_json(ref / ms.META_RELPATH, meta)

    lat_root = lib / "wan-latents"
    lat_root.mkdir()
    _write_json(lat_root / "metadata.json", {"raw_dir": "/fake/raw", "manifest_sha256": manifest["sha256"]})
    lines = []
    padded = 0
    for e in entries:
        for seg in ms.SEGMENTS:
            s = getattr(e, seg)
            if s.num_grid == 0:
                continue
            key = ms.segment_key(e.h5_file, e.raw_ep_idx, seg)
            start = 0 if seg == "demo" else e.exec_start_idx
            lat = (rng.standard_normal((s.num_grid, 9, 16, 32, 32)) * 2).astype(np.float32)
            data = lat.tobytes()
            sha = hashlib.sha256(data).hexdigest()
            (lat_root / f"{key}.bin").write_bytes(data)
            (lat_root / f"{key}.bin.sha256").write_text(f"{sha}  {key}.bin\n")
            rws = []
            for m in range(s.num_grid):
                off = 16 * m
                real = min(33, s.seg_len - off)
                rws.append({"m": m, "seg_offset": off, "start_global_frame": start + off,
                            "input_frames_sha256": "0" * 64, "real_frames": real, "pad_frames": 33 - real,
                            "pad_source_frame": start + s.seg_len - 1 if real < 33 else None})
                padded += int(real < 33)
            md = {"schema": 3, "segment": key, "g": e.g, "h5_file": e.h5_file, "raw_ep_idx": e.raw_ep_idx,
                  "segment_kind": seg, "seg_start_global": start, "seg_len": s.seg_len, "num_grid": s.num_grid,
                  "grid_stride": 16, "window_frames": 33, "latent_shape": [9, 16, 32, 32],
                  "latent_dtype": "float32", "chunk_bytes": 589824, "rows": rws, "sha256": sha, "bytes": len(data),
                  "vae": dict(VAE, batch=1)}
            _write_json(lat_root / f"{key}.metadata.json", md)
            for name in (f"{key}.bin", f"{key}.bin.sha256", f"{key}.metadata.json"):
                lines.append(f"{hashlib.sha256((lat_root / name).read_bytes()).hexdigest()}  wan-latents/{name}")
    lines.append(f"{hashlib.sha256((lat_root / 'metadata.json').read_bytes()).hexdigest()}  wan-latents/metadata.json")
    lines.append(f"{'a' * 64}  data/0.pkl")
    cks = lib / "checksums"
    cks.mkdir()
    (cks / "sha256-source-files.txt").write_text("\n".join(lines) + "\n")
    (cks / "sha256-shards.txt").write_text(f"{'b' * 64}  wan-latents_tars/wan-latents-00000.tar\n")
    assert padded >= 1 and entries[1].demo.num_grid == 0
    return SimpleNamespace(lib=lib, manifest=man_path, ref=ref, latents=lat_root, checksums=cks, totals=totals,
                           padded=padded, entries=entries)


def run_export(L, out: pathlib.Path, workers: int, checksums: str = "sha256-source-files.txt"):
    cmd = [sys.executable, str(EXPORT), "--latents", str(L.latents), "--motion-ref", str(L.ref),
           "--manifest", str(L.manifest), "--checksums", str(L.checksums / checksums),
           "--workers", str(workers), "--out", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True)


def run_verify(L, table: pathlib.Path):
    cmd = [sys.executable, str(VERIFY), "--table", str(table), "--latents", str(L.latents), "--motion-ref", str(L.ref),
           "--workers", "3", "--expect-rows", str(L.totals["rows"]), "--expect-demo", str(L.totals["demo_rows"]),
           "--expect-exec", str(L.totals["exec_rows"]), "--expect-padded", str(L.padded)]
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w768")
    L = build_lib(tmp)
    outs = {}
    for w in (1, 4):
        out = tmp / f"out-w{w}"
        r = run_export(L, out, w)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "EXPORT_POOL768=PASS" in r.stdout
        outs[w] = out
    return L, outs


def _meta_wo_exporter(p: pathlib.Path) -> dict:
    m = json.loads((p / ms.META_RELPATH).read_text())
    m.pop("exporter")
    return m


def test_export_determinism(exported, capsys):
    L, outs = exported
    a, b = outs[1], outs[4]
    for rel in (ms.MOTION_TABLE_RELPATH, ms.INDEX_RELPATH, ms.ROW_DIGESTS_RELPATH):
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
    assert _meta_wo_exporter(a) == _meta_wo_exporter(b)
    assert (a / ms.INDEX_RELPATH).read_bytes() == (L.ref / ms.INDEX_RELPATH).read_bytes()
    # 表行 = 逐窗直接调池化函数，行序按 index
    table = np.fromfile(a / ms.MOTION_TABLE_RELPATH, dtype=np.float32).reshape(-1, 768)
    assert table.shape[0] == L.totals["rows"]
    for e in L.entries:
        for seg in ms.SEGMENTS:
            s = getattr(e, seg)
            if s.num_grid == 0:
                continue
            key = ms.segment_key(e.h5_file, e.raw_ep_idx, seg)
            lat = np.fromfile(L.latents / f"{key}.bin", dtype=np.float32).reshape(-1, 9, 16, 32, 32)
            for m in range(s.num_grid):
                assert table[s.row_base + m].tobytes() == wp.pool_raw_wan_latent_768(lat[m]).tobytes()
    meta = ms.MotionMeta.load(a)
    p = meta.provenance
    assert meta.status == "verified" and p["encoder_kind"] == wp.POOL768_ENCODER_KIND
    assert p["encoder_ckpt"] is None and p["parameter_count"] == 0 and p["readout"] == wp.POOL768_READOUT
    assert p["pool_source_sha256"] == wp.pool_source_sha256() and p["vae"] == VAE
    assert p["latent_root"]["member_count"] == 3 * sum(
        1 for e in L.entries for s in ms.SEGMENTS if getattr(e, s).num_grid) + 1
    _say(capsys, f"EXPORT_DETERMINISM=PASS workers=1,4 rows={table.shape[0]} "
                 f"table_sha256={hashlib.sha256((a / ms.MOTION_TABLE_RELPATH).read_bytes()).hexdigest()[:16]}")


def test_export_refuses_existing_out_and_wrong_checksums(exported, tmp_path):
    L, outs = exported
    r = run_export(L, outs[1], 1)
    assert r.returncode != 0 and "输出根已存在" in r.stderr
    r = run_export(L, tmp_path / "x", 1, checksums="sha256-shards.txt")
    assert r.returncode != 0 and "sha256-source-files.txt" in r.stderr


def test_export_rejects_member_mismatch(tmp_path):
    L = build_lib(tmp_path)
    p = L.checksums / "sha256-source-files.txt"
    lines = p.read_text().splitlines()
    i = next(k for k, x in enumerate(lines) if x.endswith(".bin"))
    lines[i] = "c" * 64 + lines[i][64:]
    p.write_text("\n".join(lines) + "\n")
    r = run_export(L, tmp_path / "out", 2)
    assert r.returncode != 0 and "成员清单" in r.stderr


def test_verify_g3_pass_and_tamper(exported, tmp_path):
    L, outs = exported
    r = run_verify(L, outs[1])
    assert r.returncode == 0, r.stdout + r.stderr
    line = r.stdout.strip().splitlines()[-1]
    assert line.startswith(f"POOL768_TABLE=PASS rows={L.totals['rows']} demo={L.totals['demo_rows']} "
                           f"exec={L.totals['exec_rows']} padded={L.padded} index_mismatch=0 max_abs=")
    assert float(line.rsplit("=", 1)[1]) <= 1e-6
    # 篡改一行（同步改 meta 里的表 sha，绕过 full 档）→ 独立重算发现
    bad = tmp_path / "bad"
    subprocess.run(["cp", "-r", str(outs[1]), str(bad)], check=True)
    t = bad / ms.MOTION_TABLE_RELPATH
    arr = np.fromfile(t, dtype=np.float32)
    arr[5] += 1e-3
    t.write_bytes(arr.tobytes())
    m = json.loads((bad / ms.META_RELPATH).read_text())
    m["tables"][ms.MOTION_KEY]["sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
    (bad / ms.META_RELPATH).write_text(json.dumps(m))
    r = run_verify(L, bad)
    assert r.returncode == 1 and "POOL768_TABLE=FAIL" in r.stdout


# ══ MotionMeta.load / _motion_gates / init_history_config 的池化分支 ═══════════════


def _rewrite_prov(root: pathlib.Path, **kv) -> None:
    m = json.loads((root / ms.META_RELPATH).read_text())
    m["provenance"].update(kv)
    (root / ms.META_RELPATH).write_text(json.dumps(m))


@pytest.mark.parametrize("kv", [{"readout": "x"}, {"pool_source_sha256": "0" * 64}, {"encoder_ckpt": "ck.pt"},
                                {"parameter_count": 1}, {"parameter_count": False}, {"encoder_kind": "wan_fixedpool1024"}])
def test_motion_meta_pool_branch_rejects_tamper(exported, tmp_path, kv):
    _, outs = exported
    bad = tmp_path / "t"
    subprocess.run(["cp", "-r", str(outs[1]), str(bad)], check=True)
    ms.MotionMeta.load(bad)
    _rewrite_prov(bad, **kv)
    with pytest.raises(ValueError):
        ms.MotionMeta.load(bad)


def test_motion_meta_encoder_table_unchanged(exported):
    L, _ = exported
    meta = ms.MotionMeta.load(L.ref)
    assert "encoder_kind" not in meta.provenance and meta.provenance["encoder"] == ENC


def _gates():
    from omegaconf import OmegaConf
    from mme_vla_suite.training.dataloader import _motion_gates
    return OmegaConf, _motion_gates


def _hcfg(OmegaConf, store, **motion):
    return OmegaConf.create({"motion": {"enabled": True, "store_path": str(store), "stride": 16, **motion}})


def test_motion_gates_pool_and_encoder_branches(exported, monkeypatch):
    L, outs = exported
    OmegaConf, gates = _gates()
    monkeypatch.delenv("MMEVLA_MOTION_STORE", raising=False)
    monkeypatch.delenv("MMEVLA_FRAMESAMP_MANIFEST", raising=False)
    man = json.loads(L.manifest.read_text())
    fm = SimpleNamespace(manifest_sha256=man["sha256"], manifest_path=str(L.manifest))
    pool = dict(encoder_kind=wp.POOL768_ENCODER_KIND, readout=wp.POOL768_READOUT)
    assert gates(_hcfg(OmegaConf, outs[1], **pool), fm) == str(outs[1])
    for bad in (dict(pool, readout="wan_raw_timepool8_spatial8x8_v1"), dict(pool, encoder_kind="x"),
                dict(pool, source_run="enc-run/checkpoint_epoch_72.pt#encoder"), {"encoder_kind": pool["encoder_kind"]}):
        with pytest.raises(ValueError):
            gates(_hcfg(OmegaConf, outs[1], **bad), fm)
    # encoder 分支：旧表照旧通过、source_run 不符照旧拒；encoder YAML 指向池化表即拒
    src = "enc-run/checkpoint_epoch_72.pt#encoder"
    assert gates(_hcfg(OmegaConf, L.ref, source_run=src), fm) == str(L.ref)
    with pytest.raises(ValueError):
        gates(_hcfg(OmegaConf, L.ref, source_run="enc-run/checkpoint_epoch_71.pt#encoder"), fm)
    with pytest.raises(ValueError):
        gates(_hcfg(OmegaConf, outs[1], source_run=src), fm)
    with pytest.raises(ValueError):
        gates(_hcfg(OmegaConf, L.ref, source_run=src, **pool), fm)


ENCODER_PROV_KEYS = ["history_config_name", "motion_enabled", "resolved_sha256", "framesamp_root",
                     "framesamp_manifest_sha256", "framesamp_store_meta_sha256", "motion_root",
                     "motion_manifest_sha256", "motion_index_sha256", "motion_store_meta_sha256",
                     "motion_table_sha256", "vae", "encoder"]


@pytest.fixture(scope="module")
def train_mod():
    spec = importlib.util.spec_from_file_location("w768_train", ROOT / "scripts/training/train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_init_history_config_snapshot(exported, tmp_path, monkeypatch, train_mod):
    L, outs = exported
    from omegaconf import OmegaConf
    import mme_vla_suite.datastore as dsmod
    man = json.loads(L.manifest.read_text())
    frames = tmp_path / "framesamp"
    (frames / "meta").mkdir(parents=True)
    (frames / "meta/store_meta.json").write_text("{}")
    monkeypatch.setattr(dsmod, "StoreMeta", SimpleNamespace(
        load=lambda root: SimpleNamespace(manifest_sha256=man["sha256"], manifest_path=str(L.manifest))))
    monkeypatch.delenv("MMEVLA_MOTION_STORE", raising=False)
    monkeypatch.delenv("MMEVLA_FRAMESAMP_MANIFEST", raising=False)

    def snap(name, hcfg):
        d = tmp_path / name
        d.mkdir()
        cfg = SimpleNamespace(model=SimpleNamespace(history_config="h.yaml"), checkpoint_dir=d)
        train_mod.init_history_config(cfg, hcfg, framesamp_root=frames)
        return json.loads((d / "motion_provenance.json").read_text())

    pool = snap("pool", _hcfg(OmegaConf, outs[1], encoder_kind=wp.POOL768_ENCODER_KIND, readout=wp.POOL768_READOUT))
    assert list(pool) == ENCODER_PROV_KEYS + ["encoder_kind", "readout", "pool_source_sha256"]
    assert pool["encoder"] is None and pool["vae"] == VAE
    assert (pool["encoder_kind"], pool["readout"], pool["pool_source_sha256"]) == (
        wp.POOL768_ENCODER_KIND, wp.POOL768_READOUT, wp.pool_source_sha256())
    enc = snap("enc", _hcfg(OmegaConf, L.ref, source_run="enc-run/checkpoint_epoch_72.pt#encoder"))
    assert list(enc) == ENCODER_PROV_KEYS and enc["encoder"] == ENC
    closed = snap("closed", OmegaConf.create({"motion": {"enabled": False}}))
    assert list(closed) == ENCODER_PROV_KEYS and closed["vae"] is None and closed["encoder"] is None
