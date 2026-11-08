"""S2（在线侧）单测：计划 1108-4task-modul-wantp768-50k-plan.html 第二部分「二」分配表 S2 行。

覆盖：sidecar stub 握手（kind/readout/pool sha）与池化分支 Tensor→NumPy→(768,)；MotionEncoderClient 按 kind 握手；
policy_config 按快照 motion.encoder_kind 构造 client；新旧 YAML 差异只在 motion 来源字段；runner --dry-run 与
--smoke20 --dry-run 的 TRAIN_ARGS 只差步数与 run 名；compare_online_pool768.py --dry-run 抽样含 8 个补帧窗且可复现、
exec 窗走 pkl / demo 窗走 H5 的分派、map 缺失与三种篡改（h5 摘要、帧摘要、表行）各自 FAIL。

S1 的 ``src/mme_vla_suite/shared/wan_pool768.py`` 存在时用真模块，否则用同名接口的最小桩（只验证 S2 的接线）。
运行：``uv run --no-sync python scripts/training/tests/test_wantp768_online.py``，末行 ``S2_ONLINE_UNIT=PASS|FAIL``。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import pickle
import shlex
import socket
import subprocess
import sys
import threading
import time
import types

import numpy as np
import pytest

WT = pathlib.Path(__file__).resolve().parents[3]
POOL_PATH = WT / "src/mme_vla_suite/shared/wan_pool768.py"
SIDECAR_PATH = WT / "scripts/dataset/wan/motion_sidecar.py"
RUNNER = WT / "scripts/training/run_4task_wantp768_50k.sh"
COMPARE_PATH = WT / "scripts/training/g0/compare_online_pool768.py"
YAML_DIR = WT / "src/mme_vla_suite/models/config/robomme"
NEW_YAML = YAML_DIR / "perceptual-framesamp-modul-8frame-8x8-wantp768.yaml"
OLD_YAML = YAML_DIR / "perceptual-framesamp-modul-8frame-8x8-motion.yaml"
KIND, READOUT = "wan_fixedpool768", "wan_raw_timepool8_adaptive6x8_v1"
MAIN_V1 = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store")

_STUB_POOL_SRC = '''"""测试用最小桩：与 S1 wan_pool768 同名接口。"""
import hashlib, pathlib
import numpy as np
POOL768_READOUT = "wan_raw_timepool8_adaptive6x8_v1"
POOL768_ENCODER_KIND = "wan_fixedpool768"
POOL768_SHAPE = (1, 768)
POOL768_PARAMETER_COUNT = 0
def pool_raw_wan_latent_768(z):
    assert isinstance(z, np.ndarray) and z.dtype == np.float32 and z.shape == (9, 16, 32, 32)
    a = z[1:].mean(axis=0, dtype=np.float32)
    bins = [(0, 6), (5, 11), (10, 16), (16, 22), (21, 27), (26, 32)]
    p = np.stack([a[:, s:e].reshape(16, 6, 8, 4).mean(axis=(1, 3), dtype=np.float32) for s, e in bins], axis=1)
    return np.ascontiguousarray(p.reshape(1, 768), dtype=np.float32)
def pool_source_sha256():
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
'''


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                               # dataclass 需要按模块名回查
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def pool_path(tmp_path_factory) -> pathlib.Path:
    if POOL_PATH.is_file():
        return POOL_PATH
    p = tmp_path_factory.mktemp("poolstub") / "wan_pool768.py"
    p.write_text(_STUB_POOL_SRC)
    return p


@pytest.fixture(scope="session")
def PL(pool_path):
    return _load(pool_path, "wan_pool768_under_test")


# ── sidecar ────────────────────────────────────────────────────────────────────

def test_sidecar_stub_handshake_reports_pool_identity(pool_path, monkeypatch):
    from mme_vla_suite.policies import motion_protocol as P
    sc = _load(SIDECAR_PATH, "motion_sidecar_under_test")
    monkeypatch.setattr(sc, "_POOL768_PATH", pool_path)
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    fd = child.detach()                                    # fd 归 sidecar 所有，由它关闭
    monkeypatch.setattr(sys, "argv", ["motion_sidecar.py", "--fd", str(fd), "--stub", "--encoder-kind", KIND])
    rc = []
    t = threading.Thread(target=lambda: rc.append(sc.main()), daemon=True)
    t.start()
    try:
        info = P.recv_handshake(parent, time.monotonic() + 30)
        assert info["stub"] is True and info["encoder_kind"] == KIND and info["readout"] == READOUT
        assert info["pool_source_sha256"] == hashlib.sha256(pool_path.read_bytes()).hexdigest()
        assert info["encoder"] is None
        window = np.stack([P.stub_frame(t) for t in range(5, 5 + 33)])
        P.send_all(parent, P.encode_request(5, window.tobytes()))
        tok = np.frombuffer(P.recv_response(parent, time.monotonic() + 30), np.float32)
        assert np.array_equal(tok, np.full(768, 5.0, np.float32))
        P.send_all(parent, P.encode_shutdown())
        t.join(30)
    finally:
        parent.close()
    assert rc == [0]


def test_sidecar_rejects_unknown_kind(monkeypatch):
    sc = _load(SIDECAR_PATH, "motion_sidecar_under_test2")
    monkeypatch.setattr(sys, "argv", ["motion_sidecar.py", "--fd", "999", "--encoder-kind", "bogus"])
    with pytest.raises(SystemExit):
        sc.main()


def test_sidecar_pool_encode_tensor_to_numpy_bitexact(PL):
    torch = pytest.importorskip("torch")
    sc = _load(SIDECAR_PATH, "motion_sidecar_under_test3")
    rng = np.random.default_rng(0)
    lat_np = rng.standard_normal((9, 16, 32, 32)).astype(np.float32)
    W = types.SimpleNamespace(encode_chunk=lambda vae, window, device: torch.from_numpy(lat_np.copy()).to(device))
    window = np.zeros((33, 256, 256, 3), np.uint8)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() and os.environ.get("S2_TEST_CUDA") == "1" else [])
    want = PL.pool_raw_wan_latent_768(lat_np).reshape(768)
    for dev in devices:
        got = sc.pool_encode(W, PL, None, window, torch.device(dev))
        assert got.shape == (768,) and got.dtype == np.float32
        assert got.tobytes() == want.tobytes()


# ── MotionEncoderClient ───────────────────────────────────────────────────────

def _bare_client(**kw):
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    c = MotionEncoderClient.__new__(MotionEncoderClient)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_client_constructor_validates_kind():
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    with pytest.raises(ValueError):
        MotionEncoderClient(online_gpu=0, encoder_kind="bogus", readout=READOUT, expected_pool_sha256="a" * 64)
    with pytest.raises(ValueError):
        MotionEncoderClient(online_gpu=0, encoder_kind=KIND, readout=READOUT, expected_pool_sha256="a" * 64,
                            source_run="r/checkpoint_epoch_1.pt#encoder")
    with pytest.raises(ValueError):
        MotionEncoderClient(online_gpu=0, encoder_kind=KIND, readout=None, expected_pool_sha256="a" * 64)


@pytest.mark.parametrize("bad", ["encoder_kind", "readout", "pool_source_sha256", None])
def test_client_pool_handshake_by_kind(bad):
    prov = {"encoder_kind": KIND, "readout": READOUT, "pool_source_sha256": "b" * 64}
    if bad:
        prov[bad] = "x"
    c = _bare_client(encoder_kind=KIND, readout=READOUT, expected_pool_sha256="b" * 64, provenance=prov)
    if bad:
        with pytest.raises(RuntimeError):
            c.check_pool_handshake()
    else:
        c.check_pool_handshake()


def test_client_provenance_sections_by_kind():
    vae = {"vae_id": "v", "vae_state_sha256": "s"}
    pool = _bare_client(encoder_kind=KIND, provenance={"vae": dict(vae), "encoder": None})
    pool.check_provenance({"vae": dict(vae)})                    # 池化 kind 不比 encoder 段
    with pytest.raises(RuntimeError):
        pool.check_provenance({"vae": {**vae, "vae_state_sha256": "t"}})
    old = _bare_client(provenance={"vae": dict(vae), "encoder": {"checkpoint_sha256": "c"}})
    with pytest.raises(RuntimeError):                             # 旧路径（无 encoder_kind 属性）仍比 encoder 段
        old.check_provenance({"vae": dict(vae), "encoder": {"checkpoint_sha256": "d"}})
    old.check_provenance({"vae": dict(vae), "encoder": {"checkpoint_sha256": "c"}})


def _wan_venv_ready() -> bool:
    return (MAIN_V1 / "venvs/wan/bin/python").exists()


@pytest.mark.skipif(not _wan_venv_ready(), reason="缺主树 v1-store/venvs/wan")
def test_client_stub_end_to_end_old_path_unchanged():
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    from mme_vla_suite.policies import motion_protocol as P
    c = MotionEncoderClient(online_gpu=0, stub=True, v1_store=MAIN_V1, handshake_timeout=120)
    try:
        assert "--encoder-kind" not in c.argv and c.argv[-1] == "--stub"
        assert c.provenance.get("encoder_kind") is None
        window = np.stack([P.stub_frame(t) for t in range(16, 49)])
        assert np.array_equal(c(window, 16), np.full(768, 16.0, np.float32))
    finally:
        c.close()


@pytest.mark.skipif(not (_wan_venv_ready() and POOL_PATH.is_file()), reason="缺 wan venv 或 S1 wan_pool768.py 尚未合入")
def test_client_stub_end_to_end_pool_kind():
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    PL = _load(POOL_PATH, "wan_pool768_e2e")
    c = MotionEncoderClient(online_gpu=0, stub=True, v1_store=MAIN_V1, handshake_timeout=120, encoder_kind=KIND,
                            readout=READOUT, expected_pool_sha256=PL.pool_source_sha256())
    try:
        assert c.argv[-3:] == ["--stub", "--encoder-kind", KIND] and "--encoder-run-dir" not in c.argv
    finally:
        c.close()
    with pytest.raises(RuntimeError):
        MotionEncoderClient(online_gpu=0, stub=True, v1_store=MAIN_V1, handshake_timeout=120, encoder_kind=KIND,
                            readout=READOUT, expected_pool_sha256="0" * 64)


# ── policy_config ─────────────────────────────────────────────────────────────

def test_policy_config_builds_client_by_kind(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    import mme_vla_suite.policies.motion_client as mc
    from mme_vla_suite.policies import policy_config as pc
    seen = {}

    class Rec:
        def __init__(self, **kw):
            seen.update(kw)
    monkeypatch.setattr(mc, "MotionEncoderClient", Rec)
    mcfg = OmegaConf.load(NEW_YAML).motion
    prov = {"vae": {"vae_state_sha256": "v"}, "encoder": None, "encoder_kind": KIND, "readout": READOUT,
            "pool_source_sha256": "p" * 64}
    (tmp_path / "motion_provenance.json").write_text(json.dumps(prov))
    pc._make_pool_motion_client(mcfg, tmp_path, False)
    assert seen["encoder_kind"] == KIND and seen["readout"] == READOUT and seen["expected_pool_sha256"] == "p" * 64
    assert seen["store_provenance"] == {"vae": prov["vae"]} and "source_run" not in seen
    for key, val in (("pool_source_sha256", None), ("readout", "other")):
        (tmp_path / "motion_provenance.json").write_text(json.dumps({**prov, key: val}))
        with pytest.raises(ValueError):
            pc._make_pool_motion_client(mcfg, tmp_path, False)


# ── YAML ──────────────────────────────────────────────────────────────────────

def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_yaml_differs_only_in_motion_source():
    from omegaconf import OmegaConf
    new = _flat(OmegaConf.to_container(OmegaConf.load(NEW_YAML)))
    old = _flat(OmegaConf.to_container(OmegaConf.load(OLD_YAML)))
    changed = {k for k in set(new) | set(old) if new.get(k, "<缺>") != old.get(k, "<缺>")}
    assert changed == {"motion.store_path", "motion.source_run", "motion.encoder_kind", "motion.readout"}
    assert new["motion.store_path"] == "v1-store/datasets/4task-v2-1600ep-604f16da/motion-wan-pool768"
    assert "motion.source_run" not in new and new["motion.encoder_kind"] == KIND and new["motion.readout"] == READOUT
    for k, v in (("motion.dim", 768), ("motion.budget", 160), ("motion.pos_dim", 256), ("motion.stride", 16),
                 ("motion.window_frames", 33), ("motion.demo_min_real_frames", 17), ("integration_type", "modulation"),
                 ("memory_token_dim", 1024), ("budget", 512), ("token_per_image", 64)):
        assert new[k] == v


# ── runner ────────────────────────────────────────────────────────────────────

def _dry(*extra):
    r = subprocess.run(["bash", str(RUNNER), "0" * 40, *extra, "--dry-run"], capture_output=True, text=True, check=False)
    return r


def _train_args(out: str) -> list[str]:
    line = [l for l in out.splitlines() if l.startswith("TRAIN_ARGS=")]
    assert len(line) == 1
    return shlex.split(line[0][len("TRAIN_ARGS="):])


def test_runner_syntax_and_yaml_sha():
    assert subprocess.run(["bash", "-n", str(RUNNER)], check=False).returncode == 0
    text = RUNNER.read_text()
    assert f"HC_SHA={hashlib.sha256(NEW_YAML.read_bytes()).hexdigest()}" in text


def test_runner_dry_run_formal_vs_smoke_only_steps_and_name():
    formal, smoke = _dry(), _dry("--smoke20", "--run-name", "w768-smoke20")
    assert formal.returncode == 0 and smoke.returncode == 0, formal.stderr + smoke.stderr
    a, b = _train_args(formal.stdout), _train_args(smoke.stdout)
    assert len(a) == len(b)
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    assert diffs == [("v2-1600ep-m8x8-modul-wantp768-b128-50k", "w768-smoke20"),
                     ("--num_train_steps=50001", "--num_train_steps=20")]
    assert a[0] == "mme_vla_suite_b128_80k" and "perceptual-framesamp-modul-8frame-8x8-wantp768.yaml" in a
    assert "EXPECT_CKPTS=19" in smoke.stdout
    assert "EXPECT_CKPTS=5000 10000 15000 20000 25000 30000 35000 40000 45000 50000" in formal.stdout
    assert _dry("--smoke20").stdout == smoke.stdout                 # 冒烟默认 run 名即 w768-smoke20
    assert _dry("--smoke20", "--run-name", "other").returncode == 2


# ── compare_online_pool768.py ─────────────────────────────────────────────────

def _frame(g: int, t: int) -> np.ndarray:
    rng = np.random.default_rng(1000 * g + t)
    return rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)


def _fake_latent(window: np.ndarray) -> np.ndarray:
    seed = int(hashlib.sha256(np.ascontiguousarray(window).tobytes()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).standard_normal((9, 16, 32, 32)).astype(np.float32)


def _window(frames: list, off: int, seg_len: int) -> np.ndarray:
    real = min(33, seg_len - off)
    w = np.stack(frames[off:off + real])
    if real < 33:
        w = np.concatenate([w, np.repeat(np.stack(frames[seg_len - 1:seg_len]), 33 - real, axis=0)])
    return np.ascontiguousarray(w)


def _build_lib(root: pathlib.Path, PL, specs, with_frames=True):
    """specs: [(task, nt, es)]；每个 task 一条 episode（raw_ep_idx=0）。"""
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.datastore.manifest import manifest_sha256
    import h5py
    eps, off_exec = [], 0
    for g, (task, nt, es) in enumerate(specs):
        eps.append({"global_episode_idx": g, "h5_file": f"record_dataset_{task}.h5", "raw_ep_idx": 0,
                    "num_timesteps": nt, "exec_start_idx": es, "exec_samples": nt - es,
                    "exec_sample_offset": off_exec, "total_sample_offset": 0, "shard_idx": 0})
        off_exec += nt - es
    manifest = {"version": 1, "raw_dir": str(root / "raw"), "episodes": eps}
    manifest["sha256"] = manifest_sha256(manifest)
    (root / "meta").mkdir(parents=True)
    (root / "meta/episode_manifest.json").write_text(json.dumps(manifest))
    spec = ms.LAYOUT_SPECS[ms.LAYOUT]
    entries = ms.build_index_entries(manifest, spec)
    idx_bytes = json.dumps(ms.index_payload(manifest, entries, spec=spec, layout=ms.LAYOUT, mj_repo_commit="t")).encode()
    table = root / "motion-wan-pool768"
    (table / "meta").mkdir(parents=True)
    (table / ms.INDEX_RELPATH).write_bytes(idx_bytes)
    rows_total = ms.index_totals(entries)["rows"]
    tab = np.zeros((rows_total, 768), np.float32)
    segments, maps = {}, {}
    for e, (task, nt, es) in zip(entries, specs):
        frames = [_frame(e.g, t) for t in range(nt)] if with_frames else None
        for kind in ms.SEGMENTS:
            seg = getattr(e, kind)
            if not seg.num_grid:
                continue
            key = ms.segment_key(e.h5_file, e.raw_ep_idx, kind)
            s0 = 0 if kind == "demo" else es
            rows = []
            for m in range(seg.num_grid):
                off = 16 * m
                pad = max(0, 33 - (seg.seg_len - off))
                r = {"m": m, "pad_frames": pad, "input_frames_sha256": None}
                if with_frames:
                    w = _window(frames[s0:s0 + seg.seg_len], off, seg.seg_len)
                    r["input_frames_sha256"] = hashlib.sha256(w.tobytes()).hexdigest()
                    tab[seg.row_base + m] = PL.pool_raw_wan_latent_768(_fake_latent(w)).reshape(768)
                rows.append(r)
            segments[key] = {"segment_kind": kind, "seg_len": seg.seg_len, "num_grid": seg.num_grid, "rows": rows}
        if with_frames:
            member = f"{task}/easy/ep{e.g}.h5"
            p = root / "h5" / member
            p.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(p, "w") as f:
                grp = f.create_group(f"episode_{7 + e.g}")
                for t in range(nt):
                    ts = grp.create_group(f"timestep_{t}")
                    ts.create_dataset("obs/front_rgb", data=frames[t])
                    ts.create_dataset("info/is_video_demo", data=np.bool_(t < es))
            maps[task] = [{"task": task, "difficulty": "easy", "episode": 7 + e.g, "seed": 1, "role": "primary",
                           "member": member, "h5_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                           "h5_bytes": p.stat().st_size, "timestep_count": nt, "new_idx": 0,
                           "src_group": f"episode_{7 + e.g}"}]
            (root / "source/data").mkdir(parents=True, exist_ok=True)
            ep = eps[e.g]
            for t in range(es, nt):
                d = {"image": frames[t], "epis_idx": np.array([e.g], np.int32), "step_idx": np.array([t], np.int32),
                     "is_demo": np.array([False])}
                with open(root / "source/data" / f"{ep['exec_sample_offset'] + t - es}.pkl", "wb") as f:
                    pickle.dump(d, f)
    (table / ms.MOTION_TABLE_RELPATH).write_bytes(tab.tobytes())
    meta = {"schema": ms.META_SCHEMA, "layout": ms.LAYOUT, "status": "verified", **spec.fields(),
            "byte_order": ms.BYTE_ORDER, "array_order": ms.ARRAY_ORDER, "grid_stride": ms.GRID_STRIDE,
            "window_frames": ms.WINDOW_FRAMES, "grid_origin": ms.GRID_ORIGIN, "window_direction": ms.WINDOW_DIRECTION,
            "truncation_policy": ms.TRUNCATION_POLICY, "frame_size": ms.FRAME_SIZE,
            "tables": {ms.MOTION_KEY: {"row_shape": [768], "dtype": "float32", "row_bytes": 3072, "num_rows": rows_total,
                                       "relpath": ms.MOTION_TABLE_RELPATH, "byte_count": rows_total * 3072}},
            "num_rows": rows_total, "manifest_sha256": manifest["sha256"],
            "manifest_path": str(root / "meta/episode_manifest.json"),
            "motion_index_sha256": hashlib.sha256(idx_bytes).hexdigest(),
            "provenance": {"encoder_kind": KIND, "readout": READOUT, "encoder_ckpt": None, "parameter_count": 0,
                           "pool_source_sha256": PL.pool_source_sha256(), "vae": {"vae_state_sha256": "v"}}}
    (table / ms.META_RELPATH).write_text(json.dumps(meta))
    (root / "wan-latents").mkdir()
    (root / "wan-latents/metadata.json").write_text(json.dumps({"manifest_sha256": manifest["sha256"], "segments": segments}))
    (root / "maps").mkdir()
    for task, rows in maps.items():
        (root / "maps" / f"record_dataset_{task}_episode_map.json").write_text(json.dumps(rows))
    return table


@pytest.fixture(scope="session")
def G4():
    return _load(COMPARE_PATH, "compare_online_pool768_under_test")


def _args(root, table, **kw):
    base = dict(table=str(table), source_root=str(root / "source"), h5_root=str(root / "h5"),
                episode_map_dir=str(root / "maps"), windows=6, padded=2, seed=0, gpu="0", dry_run=False)
    base.update(kw)
    return argparse.Namespace(**base)


class _FakeClient:
    def __init__(self, PL):
        self.PL, self.calls = PL, []

    def __call__(self, window, start):
        self.calls.append(start)
        return self.PL.pool_raw_wan_latent_768(_fake_latent(np.asarray(window))).reshape(768)

    def close(self):
        pass


def test_compare_dry_run_samples_8_padded_reproducible(tmp_path, PL, G4, capsys):
    specs = [(t, 400, 200) for t in ("BinFill", "RouteStick", "VideoRepick", "VideoUnmaskSwap")] * 3
    table = _build_lib(tmp_path, PL, specs, with_frames=False)
    outs = []
    for _ in range(2):
        assert G4.run(_args(tmp_path, table, windows=64, padded=8, dry_run=True)) == 0
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1]
    last = outs[0].strip().splitlines()[-1]
    assert last.startswith("DRY_RUN windows=64 padded=8 ")
    wins = [json.loads(l[len("WINDOW "):]) for l in outs[0].splitlines() if l.startswith("WINDOW ")]
    assert len(wins) == 64 and sum(w["padded"] for w in wins) == 8
    assert all(w["source"] == ("h5" if w["kind"] == "demo" else "pkl") for w in wins)
    assert all(w["kind"] == "demo" for w in wins if w["padded"])
    assert {w["kind"] for w in wins} == {"demo", "exec"}


def test_compare_real_mode_pass_and_dispatch(tmp_path, PL, G4, capsys, monkeypatch):
    table = _build_lib(tmp_path, PL, [("BinFill", 80, 40), ("RouteStick", 80, 40)])
    used = {"h5": 0, "pkl": 0}
    orig_demo, orig_exec = G4.demo_window, G4.exec_window
    monkeypatch.setattr(G4, "demo_window", lambda *a, **k: (used.__setitem__("h5", used["h5"] + 1), orig_demo(*a, **k))[1])
    monkeypatch.setattr(G4, "exec_window", lambda *a, **k: (used.__setitem__("pkl", used["pkl"] + 1), orig_exec(*a, **k))[1])
    client = _FakeClient(PL)
    assert G4.run(_args(tmp_path, table), client_factory=lambda meta, gpu: client) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ONLINE_POOL768=PASS windows=6 padded=2 bitexact=6"
    assert used == {"h5": 4, "pkl": 2} and len(client.calls) == 6


@pytest.mark.parametrize("tamper", ["map_missing", "h5_sha", "frame_sha", "table_row"])
def test_compare_tamper_fails(tmp_path, PL, G4, capsys, monkeypatch, tamper):
    table = _build_lib(tmp_path, PL, [("BinFill", 80, 40), ("RouteStick", 80, 40)])
    if tamper == "map_missing":
        (tmp_path / "maps/record_dataset_RouteStick_episode_map.json").unlink()
    elif tamper == "h5_sha":
        p = tmp_path / "maps/record_dataset_BinFill_episode_map.json"
        rows = json.loads(p.read_text()); rows[0]["h5_sha256"] = "0" * 64; p.write_text(json.dumps(rows))
    elif tamper == "frame_sha":
        p = tmp_path / "wan-latents/metadata.json"
        lat = json.loads(p.read_text())
        lat["segments"]["BinFill_ep0_demo"]["rows"][1]["input_frames_sha256"] = "0" * 64       # 补帧窗
        p.write_text(json.dumps(lat))
    elif tamper == "table_row":
        tab = np.fromfile(table / "motion_token.f32.bin", np.float32).reshape(-1, 768)
        tab[2, 5] = np.nextafter(tab[2, 5], np.float32(np.inf))
        (table / "motion_token.f32.bin").write_bytes(tab.tobytes())
    client = _FakeClient(PL)
    monkeypatch.setattr(G4, "make_client", lambda meta, gpu: client)
    rc = G4.main([f"--table={table}", f"--source-root={tmp_path / 'source'}", f"--h5-root={tmp_path / 'h5'}",
                  f"--episode-map-dir={tmp_path / 'maps'}", "--windows=6", "--padded=2"])
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert rc == 1 and out.startswith("ONLINE_POOL768=FAIL"), out
    if tamper != "table_row":
        assert client.calls == []                                   # 摘要不过不送 sidecar
    else:
        assert out == "ONLINE_POOL768=FAIL windows=6 padded=2 bitexact=5"


# ── runner 内置 G0（DATA_LINEAGE 两侧 pin 取自独立记录）──────────────────────────

def _sha_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _build_g0_lib(root: pathlib.Path, PL) -> dict:
    """在 _build_lib 小夹具上补齐 G0 各环要读的文件；返回 argv 所需的 manifest identity 与 latent 元数据摘要。"""
    from mme_vla_suite.datastore import motion_store as ms
    table = _build_lib(root, PL, [("BinFill", 80, 40), ("RouteStick", 80, 40)], with_frames=False)
    man = json.loads((root / "meta/episode_manifest.json").read_text())
    raw_dir = man["raw_dir"]
    im = {"raw_dir": raw_dir, "count": 4,
          "files": {f"record_dataset_{t}.h5": {"size": i + 1, "sha256": hashlib.sha256(t.encode()).hexdigest()}
                    for i, t in enumerate(("BinFill", "RouteStick", "VideoRepick", "VideoUnmaskSwap"))}}
    (root / "meta/input_manifest.json").write_text(json.dumps(im, indent=2))
    (root / "meta.bucket").mkdir()
    (root / "meta.bucket/input_manifest.json").write_bytes((root / "meta/input_manifest.json").read_bytes())
    im_sha = _sha_file(root / "meta/input_manifest.json")
    lat_p = root / "wan-latents/metadata.json"
    lat = json.loads(lat_p.read_text()); lat["raw_dir"] = raw_dir
    lat_p.write_text(json.dumps(lat))
    lat_sha = _sha_file(lat_p)
    meta_p = table / ms.META_RELPATH
    meta = json.loads(meta_p.read_text())
    meta["provenance"].update(manifest_sha256=man["sha256"], latent_root={"metadata_sha256": lat_sha},
                              input_manifest_sha256=im_sha)
    meta_p.write_text(json.dumps(meta))
    (root / "motion/meta").mkdir(parents=True)
    (root / "motion/meta/store_meta.json").write_text(json.dumps({"provenance": {"input_manifest_sha256": im_sha}}))
    (root / "source/meta").mkdir(parents=True)
    (root / "source/meta/provenance.json").write_text(json.dumps({"manifest_sha256": man["sha256"], "raw_dir": raw_dir}))
    (root / "framesamp-8x8/meta").mkdir(parents=True)
    (root / "framesamp-8x8/meta/store_meta.json").write_text(json.dumps({
        "manifest_sha256": man["sha256"], "source_dataset_root": str(root / "source"),
        "source_provenance_sha256": _sha_file(root / "source/meta/provenance.json")}))
    return {"table": table, "manifest_id": man["sha256"], "lat_sha": lat_sha, "rows": meta["num_rows"]}


def _run_g0(root: pathlib.Path, info: dict, PL, monkeypatch, capsys) -> tuple[int, str]:
    import re
    import mme_vla_suite.shared as shared
    code = re.findall(r"<<'PY'\n(.*?)\nPY\n", RUNNER.read_text(), re.S)[0]
    code = code.replace("mm.num_rows == 71316", f"mm.num_rows == {info['rows']}")   # 夹具行数替换全库常量
    monkeypatch.setitem(sys.modules, "mme_vla_suite.shared.wan_pool768", PL)
    monkeypatch.setattr(shared, "wan_pool768", PL, raising=False)
    monkeypatch.delenv("MMEVLA_MOTION_STORE", raising=False)
    monkeypatch.setattr(sys, "argv", ["g0", str(root), str(info["table"]), info["manifest_id"], info["lat_sha"]])
    with pytest.raises(SystemExit) as ex:
        exec(compile(code, "runner_g0", "exec"), {"__name__": "__main__"})
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.startswith("DATA_LINEAGE=")]
    assert len(lines) == 1, out
    return ex.value.code, lines[0]


@pytest.mark.parametrize("case", ["pass", "record_differs", "record_missing", "old_record_differs"])
def test_runner_g0_lineage_independent_pins(tmp_path, PL, monkeypatch, capsys, case):
    from mme_vla_suite.datastore import motion_store as ms
    info = _build_g0_lib(tmp_path, PL)
    meta_p = info["table"] / ms.META_RELPATH
    meta = json.loads(meta_p.read_text())
    if case == "record_differs":
        meta["provenance"]["input_manifest_sha256"] = "0" * 64
    elif case == "record_missing":
        del meta["provenance"]["input_manifest_sha256"]
    elif case == "old_record_differs":
        (tmp_path / "motion/meta/store_meta.json").write_text(json.dumps({"provenance": {"input_manifest_sha256": "1" * 64}}))
    meta_p.write_text(json.dumps(meta))
    rc, line = _run_g0(tmp_path, info, PL, monkeypatch, capsys)
    im_sha = _sha_file(tmp_path / "meta/input_manifest.json")
    if case == "pass":
        assert rc == 0 and line.startswith("DATA_LINEAGE=PASS "), line
        fields = dict(kv.split("=", 1) for kv in line.split()[1:])
        assert fields["encoder_source_pin"] == fields["vla_input_pin"] != "None"
        assert fields["files"] == "4" and fields["mismatches"] == "0"
        assert fields["binding"] == f"input_manifest_sha256:{im_sha[:8]}"
        assert fields["enc_record"] == "motion-wan-pool768.provenance+motion.provenance"
    else:
        assert rc == 1 and line.startswith("DATA_LINEAGE=FAIL "), line
        if case in ("record_differs", "record_missing"):
            assert "encoder_source_pin=None" in line


if __name__ == "__main__":
    class _Count:
        def __init__(self):
            self.passed = self.failed = self.skipped = 0

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
                if report.outcome == "passed":
                    self.passed += 1
                elif report.outcome == "skipped":
                    self.skipped += 1
                else:
                    self.failed += 1
    c = _Count()
    rc = pytest.main([__file__, "-q", "-p", "no:cacheprovider"], plugins=[c])
    ok = rc == 0 and c.failed == 0
    print(f"S2_ONLINE_UNIT={'PASS' if ok else 'FAIL'} passed={c.passed} failed={c.failed} skipped={c.skipped}")
    sys.exit(0 if ok else 1)
