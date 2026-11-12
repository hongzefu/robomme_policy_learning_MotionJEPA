"""S3（评估侧）单测：计划 1112-4task-modul-wanraw9-s32-50k-plan.html 第二部分「二」分配表 S3 行、1.1 第 15–24 行。

覆盖：
- sidecar wan_raw9 分支：桩 ``W.encode_chunk`` 返回随机 latent，``raw9_encode`` 输出与直接 ``flatten_raw9`` 逐字节相等；
  stub 档经真实 socket 走「sidecar → motion_protocol（589,824 B）→ MotionEncoderClient.__call__ → (9,16384)」整条链；
- 握手：kind / readout / raw9_source_sha256 任一不符即拒；构造参数缺 readout / expected_raw9_sha256 即拒；
- pool768 分支回归：stub 握手、(768,) 响应、``pool_encode`` 输出与改动前同式；协议缺省长度仍 3,072 B；
- policy_config：raw9 快照按 raw9_source_sha256 构造客户端，缺该键即拒；
- 在线缓冲 ``FrameSampMemory``：合成 episode 上每个决策时刻的 motion 四键与 S2 ``FrameSampDataset._fill_motion_raw9``
  （真 raw9 index 条目 + rows_into 表）逐字节相等（emb / pos / mask / mem_order）；超预算 resample 情形与按计划 C-τ 独立写的
  参考实现（窗步长 32、每窗 9 token、τ_j = min(f+4j, last_real)、pos 取 pos_emb[τ,0,:256]、右填充到 budget×9）逐字节相等；
  tokens_per_window 缺省时旧路径形状不变；
- ``compare_online_raw9.py``：小夹具 exec 窗走 pkl / demo 补帧窗走 H5 的分派、帧摘要取源 latent 第 2m' 行；
  map 缺失、h5_sha256 不符、input_frames_sha256 不符各自 FAIL，表行改 1 ulp 时 bitexact 少 1；
- ``check_shard.py``：不设 env 旧口径 16 不变，``EXPECT_MOTION_STRIDE=32`` 时闭式公式按 32；
- ``check_checkpoint.py``：RUNS 五条登记、raw9 ``assert_motion`` 分支、``WANRAW9_RESOLVED_SHA256`` 与定稿 YAML 复算一致；
- ``compare_train_infer_obs.py``：``--motion-subdir`` 默认 motion。

S1、S2 已合入（v15 73289a3）：全部用真模块 ``mme_vla_suite.shared.wan_raw9``、真 ``motion_store`` raw9 布局与
S2 定稿 YAML ``…-8x8-wanraw9s32.yaml``；在线缓冲直接与 S2 的 ``FrameSampDataset._fill_motion_raw9`` 同一 (episode, t) 逐字节比，
超预算 resample 情形训练侧按零截断契约 raise，只能与按 C-τ 写的参考实现比。
运行：``uv run --no-sync python scripts/training/tests/test_raw9_online.py``，末行 ``S3_EVAL_UNIT=PASS|FAIL``。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import pickle
import socket
import sys
import threading
import time
import types

import numpy as np
import pytest

WT = pathlib.Path(__file__).resolve().parents[3]
RAW9_PATH = WT / "src/mme_vla_suite/shared/wan_raw9.py"
POOL_PATH = WT / "src/mme_vla_suite/shared/wan_pool768.py"
SIDECAR_PATH = WT / "scripts/dataset/wan/motion_sidecar.py"
COMPARE_PATH = WT / "scripts/training/g0/compare_online_raw9.py"
CHECK_SHARD = WT / "scripts/evaluation/check_shard.py"
CHECK_CKPT = WT / "scripts/evaluation/check_checkpoint.py"
TIC_PATH = WT / "scripts/training/g0/compare_train_infer_obs.py"
YAML_DIR = WT / "src/mme_vla_suite/models/config/robomme"
RAW9_YAML = YAML_DIR / "perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml"
WANTP768_YAML = YAML_DIR / "perceptual-framesamp-modul-8frame-8x8-wantp768.yaml"
KIND, READOUT = "wan_raw9", "wan_raw9_flat16384_s32_v1"
SHAPE = (9, 16, 32, 32)



def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                               # dataclass 需要按模块名回查
    spec.loader.exec_module(mod)
    return mod


def test_imports_point_to_worktree():
    import mme_vla_suite
    print("mme_vla_suite.__file__ =", mme_vla_suite.__file__)
    assert pathlib.Path(mme_vla_suite.__file__).resolve().is_relative_to(WT / "src")


@pytest.fixture(scope="session")
def R():
    """S1 真模块（sidecar 按 _RAW9_PATH 按文件加载同一份）。"""
    from mme_vla_suite.shared import wan_raw9 as mod
    assert pathlib.Path(mod.__file__).resolve() == RAW9_PATH
    return mod


@pytest.fixture(scope="session")
def PL():
    return _load(POOL_PATH, "wan_pool768_under_test")


def _latent(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(SHAPE).astype(np.float32)


# ── 协议 ───────────────────────────────────────────────────────────────────────

def test_protocol_lengths_default_unchanged():
    from mme_vla_suite.policies import motion_protocol as P
    assert P.RESPONSE_BYTES == 3072 and P.RAW9_RESPONSE_BYTES == 589824 == 9 * 16 * 32 * 32 * 4
    tok = np.arange(768, dtype=np.float32).tobytes()
    assert P.encode_response(tok) == P.encode_response(tok, P.RESPONSE_BYTES)
    with pytest.raises(P.ProtocolError):
        P.encode_response(tok, P.RAW9_RESPONSE_BYTES)
    with pytest.raises(P.ProtocolError):
        P.encode_response(_latent(0).tobytes())                  # 缺省长度不接受 raw9 响应


# ── sidecar ────────────────────────────────────────────────────────────────────

def _run_stub_sidecar(sc, monkeypatch, kind: str):
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    fd = child.detach()                                    # fd 归 sidecar 所有，由它关闭
    monkeypatch.setattr(sys, "argv", ["motion_sidecar.py", "--fd", str(fd), "--stub", "--encoder-kind", kind])
    rc = []
    t = threading.Thread(target=lambda: rc.append(sc.main()), daemon=True)
    t.start()
    return parent, t, rc


def test_sidecar_raw9_encode_bitexact_vs_flatten(R):
    torch = pytest.importorskip("torch")
    sc = _load(SIDECAR_PATH, "motion_sidecar_raw9_enc")
    lat = _latent(3)
    W = types.SimpleNamespace(encode_chunk=lambda vae, window, device: torch.from_numpy(lat.copy()).to(device))
    got = sc.raw9_encode(W, R, None, np.zeros((33, 256, 256, 3), np.uint8), torch.device("cpu"))
    assert got.shape == SHAPE and got.dtype == np.float32 and got.flags.c_contiguous
    assert got.tobytes() == R.flatten_raw9(lat).tobytes() == lat.tobytes()
    assert len(got.tobytes()) == 589824
    bad = types.SimpleNamespace(encode_chunk=lambda vae, window, device: torch.zeros(1, 9, 16, 32, 32))
    with pytest.raises(RuntimeError):
        sc.raw9_encode(bad, R, None, np.zeros((33, 256, 256, 3), np.uint8), torch.device("cpu"))


def test_sidecar_raw9_info_rejects_wrong_kind(R):
    sc = _load(SIDECAR_PATH, "motion_sidecar_raw9_info")
    info = sc.raw9_info(R, KIND)
    assert info["encoder_kind"] == KIND and info["readout"] == READOUT and info["encoder"] is None
    assert info["raw9_source_sha256"] == R.source_sha256()
    with pytest.raises(RuntimeError):
        sc.raw9_info(R, "wan_fixedpool768")


def _bare_client(**kw):
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    c = MotionEncoderClient.__new__(MotionEncoderClient)
    c._lock, c.n_calls, c.total_s = threading.Lock(), 0, 0.0
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_sidecar_stub_raw9_end_to_end_through_client(R, monkeypatch):
    """stub 档整条在线链：sidecar 握手 → 589,824 B 响应 → 客户端展平为 (9,16384)。"""
    from mme_vla_suite.policies import motion_protocol as P
    sc = _load(SIDECAR_PATH, "motion_sidecar_raw9_e2e")
    assert sc._RAW9_PATH == RAW9_PATH
    parent, t, rc = _run_stub_sidecar(sc, monkeypatch, KIND)
    try:
        info = P.recv_handshake(parent, time.monotonic() + 30)
        assert info["stub"] is True and info["encoder_kind"] == KIND and info["readout"] == READOUT
        assert info["raw9_source_sha256"] == hashlib.sha256(RAW9_PATH.read_bytes()).hexdigest() == R.source_sha256()
        assert info["encoder"] is None and info["protocol_sha256"] == P.protocol_sha256()
        c = _bare_client(encoder_kind=KIND, readout=READOUT, expected_raw9_sha256=R.source_sha256(), provenance=info,
                         _sock=parent, _proc=types.SimpleNamespace(poll=lambda: None, returncode=None))
        c.check_pool_handshake()
        for start in (5, 37):
            window = np.stack([P.stub_frame(i) for i in range(start, start + 33)])
            tok = c(window, start)
            assert tok.shape == (9, 16384) and tok.dtype == np.float32
            assert tok.tobytes() == np.full((9, 16384), float(start), np.float32).tobytes()
        assert c.n_calls == 2
        P.send_all(parent, P.encode_shutdown())
        t.join(30)
        c._sock = c._proc = None
    finally:
        parent.close()
    assert rc == [0]


def test_sidecar_stub_pool768_regression(PL, monkeypatch):
    from mme_vla_suite.policies import motion_protocol as P
    sc = _load(SIDECAR_PATH, "motion_sidecar_pool_reg")
    parent, t, rc = _run_stub_sidecar(sc, monkeypatch, "wan_fixedpool768")
    try:
        info = P.recv_handshake(parent, time.monotonic() + 30)
        assert info["encoder_kind"] == "wan_fixedpool768" and info["pool_source_sha256"] == PL.pool_source_sha256()
        assert "raw9_source_sha256" not in info
        window = np.stack([P.stub_frame(i) for i in range(5, 38)])
        P.send_all(parent, P.encode_request(5, window.tobytes()))
        tok = np.frombuffer(P.recv_response(parent, time.monotonic() + 30), np.float32)
        assert tok.tobytes() == np.full(768, 5.0, np.float32).tobytes()
        P.send_all(parent, P.encode_shutdown())
        t.join(30)
    finally:
        parent.close()
    assert rc == [0]


def test_sidecar_pool_encode_regression(PL):
    torch = pytest.importorskip("torch")
    sc = _load(SIDECAR_PATH, "motion_sidecar_pool_enc")
    lat = _latent(4)
    W = types.SimpleNamespace(encode_chunk=lambda vae, window, device: torch.from_numpy(lat.copy()).to(device))
    got = sc.pool_encode(W, PL, None, np.zeros((33, 256, 256, 3), np.uint8), torch.device("cpu"))
    assert got.shape == (768,) and got.tobytes() == PL.pool_raw_wan_latent_768(lat).reshape(768).tobytes()


def test_sidecar_rejects_unknown_kind(monkeypatch):
    sc = _load(SIDECAR_PATH, "motion_sidecar_bogus")
    monkeypatch.setattr(sys, "argv", ["motion_sidecar.py", "--fd", "999", "--encoder-kind", "bogus"])
    with pytest.raises(SystemExit):
        sc.main()


# ── MotionEncoderClient 握手 ──────────────────────────────────────────────────

def test_client_constructor_validates_raw9_args():
    from mme_vla_suite.policies.motion_client import MotionEncoderClient, POOL_ENCODER_KINDS, RAW9_ENCODER_KINDS
    assert RAW9_ENCODER_KINDS == (KIND,) and POOL_ENCODER_KINDS == ("wan_fixedpool768",)
    for kw in ({"readout": None, "expected_raw9_sha256": "a" * 64},
               {"readout": READOUT, "expected_raw9_sha256": None},
               {"readout": READOUT, "expected_pool_sha256": "a" * 64},            # raw9 不认池化 sha
               {"readout": READOUT, "expected_raw9_sha256": "a" * 64, "source_run": "r/checkpoint_epoch_1.pt#encoder"}):
        with pytest.raises(ValueError):
            MotionEncoderClient(online_gpu=0, encoder_kind=KIND, **kw)
    with pytest.raises(ValueError):                                       # 池化 kind 仍要求 expected_pool_sha256
        MotionEncoderClient(online_gpu=0, encoder_kind="wan_fixedpool768", readout="x", expected_raw9_sha256="a" * 64)


@pytest.mark.parametrize("bad", ["encoder_kind", "readout", "raw9_source_sha256", None])
def test_client_raw9_handshake_rejects_mismatch(bad):
    prov = {"encoder_kind": KIND, "readout": READOUT, "raw9_source_sha256": "b" * 64}
    if bad:
        prov[bad] = "x"
    c = _bare_client(encoder_kind=KIND, readout=READOUT, expected_raw9_sha256="b" * 64, expected_pool_sha256=None,
                     provenance=prov, _sock=None, _proc=None)
    if bad:
        with pytest.raises(RuntimeError):
            c.check_pool_handshake()
    else:
        c.check_pool_handshake()


def test_client_pool_handshake_regression():
    prov = {"encoder_kind": "wan_fixedpool768", "readout": "r", "pool_source_sha256": "p" * 64, "raw9_source_sha256": "z"}
    c = _bare_client(encoder_kind="wan_fixedpool768", readout="r", expected_pool_sha256="p" * 64,
                     expected_raw9_sha256=None, provenance=prov, _sock=None, _proc=None)
    c.check_pool_handshake()
    c.provenance = {**prov, "pool_source_sha256": "q" * 64}
    with pytest.raises(RuntimeError):
        c.check_pool_handshake()


def test_client_raw9_provenance_only_vae_section():
    vae = {"vae_id": "v", "vae_state_sha256": "s", "encoder_src_sha256": "40d8a33b"}
    c = _bare_client(encoder_kind=KIND, provenance={"vae": dict(vae)}, _sock=None, _proc=None)
    c.check_provenance({"vae": {**vae, "encoder_src_sha256": "other"}})      # 与池化 kind 同：不比 encoder 段与该键
    with pytest.raises(RuntimeError):
        c.check_provenance({"vae": {**vae, "vae_state_sha256": "t"}})


# ── policy_config ─────────────────────────────────────────────────────────────

def _raw9_mcfg():
    from omegaconf import OmegaConf
    m = OmegaConf.to_container(OmegaConf.load(WANTP768_YAML).motion)
    m.update(dim=16384, budget=80, stride=32, encoder_kind=KIND, readout=READOUT, tokens_per_window=9,
             store_path="v1-store/datasets/4task-v2-1600ep-604f16da/motion-wan-raw9-s32")
    real = OmegaConf.load(RAW9_YAML).motion
    assert OmegaConf.to_container(real) == m          # 与 S2 定稿 YAML 的 motion 节逐键逐值相同
    return real


def test_policy_config_builds_raw9_client(tmp_path, monkeypatch):
    import mme_vla_suite.policies.motion_client as mc
    from mme_vla_suite.policies import policy_config as pc
    seen = {}

    class Rec:
        def __init__(self, **kw):
            seen.clear()
            seen.update(kw)
    monkeypatch.setattr(mc, "MotionEncoderClient", Rec)
    prov = {"vae": {"vae_state_sha256": "v"}, "encoder": None, "encoder_kind": KIND, "readout": READOUT,
            "raw9_source_sha256": "r" * 64}
    (tmp_path / "motion_provenance.json").write_text(json.dumps(prov))
    pc._make_pool_motion_client(_raw9_mcfg(), tmp_path, False)
    assert seen["encoder_kind"] == KIND and seen["readout"] == READOUT and seen["expected_raw9_sha256"] == "r" * 64
    assert "expected_pool_sha256" not in seen and seen["store_provenance"] == {"vae": prov["vae"]}
    for key, val in (("raw9_source_sha256", None), ("readout", "other"), ("encoder_kind", "wan_fixedpool768")):
        (tmp_path / "motion_provenance.json").write_text(json.dumps({**prov, key: val}))
        with pytest.raises(ValueError):
            pc._make_pool_motion_client(_raw9_mcfg(), tmp_path, False)
    # 池化快照回归：仍按 pool_source_sha256 构造
    from omegaconf import OmegaConf
    pprov = {"vae": {"vae_state_sha256": "v"}, "encoder": None, "encoder_kind": "wan_fixedpool768",
             "readout": "wan_raw_timepool8_adaptive6x8_v1", "pool_source_sha256": "p" * 64}
    (tmp_path / "motion_provenance.json").write_text(json.dumps(pprov))
    pc._make_pool_motion_client(OmegaConf.load(WANTP768_YAML).motion, tmp_path, False)
    assert seen["expected_pool_sha256"] == "p" * 64 and "expected_raw9_sha256" not in seen


# ── 在线缓冲 vs 训练侧参考实现（C-τ）─────────────────────────────────────────────

STRIDE, WIN, DEMO_MIN, POS_DIM = 32, 33, 17, 256


def _latent_of_start(f: int) -> np.ndarray:
    return _latent(10_000 + int(f))


def _ref_train_motion(R, pos_emb, es: int, t: int, budget: int, fsm):
    """训练侧参考实现（计划 1.0 C-B / C-τ，独立于 FrameSampMemory 的取点代码）：
    demo 起点 s=32m 满足 s+16 ≤ es−1；exec 起点 es+u、u=32m 满足 u+32 ≤ t−es；超预算按既有 resample（按窗）；
    每窗 9 token = 源 latent 展平，τ_j = min(f+4j, last_real)（demo 窗 last_real=es−1，exec 窗 f+32）。"""
    demo = list(range(0, max(es - (DEMO_MIN - 1), 0), STRIDE))
    exec_ = [es + u for u in range(0, max(t - es - (WIN - 1) + 1, 0), STRIDE)]
    frames = demo + exec_
    if len(frames) > budget:
        kd, ke = fsm._motion_quota(len(demo), len(exec_), budget)
        frames = [demo[i] for i in fsm._even_pick(len(demo), kd)] + [exec_[i] for i in fsm._even_pick(len(exec_), ke)]
    T = budget * 9
    emb = np.zeros((T, 16384), np.float32)
    pos = np.zeros((T, POS_DIM), np.float32)
    taus = []
    for i, f in enumerate(frames):
        last = es - 1 if f < es else f + 32
        tau = np.array([min(f + 4 * j, last) for j in range(9)], np.int64)
        emb[9 * i:9 * i + 9] = _latent_of_start(f).reshape(9, 16384)
        pos[9 * i:9 * i + 9] = pos_emb[tau, 0, :POS_DIM]
        taus.append(tau)
    mask = np.zeros(T, np.bool_)
    mask[:9 * len(frames)] = True
    from mme_vla_suite.shared.sampling import pad_times
    times = pad_times(np.concatenate(taus) if taus else [], T)
    return emb, pos, mask, times, len(frames)


def _make_mem(R, budget: int, tpw: int | None = 9, overflow: str = "raise"):
    from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
    cfg = {"stride": STRIDE if tpw else 16, "window_frames": WIN, "budget": budget, "frame_size": 4,
           "pos_dim": POS_DIM, "dim": 16384 if tpw else 768, "window_direction": "forward",
           "grid_origin": "segment_start", "demo_min_real_frames": DEMO_MIN, "demo_tail_pad": "repeat_last",
           "overflow": overflow}
    if tpw is not None:
        cfg["tokens_per_window"] = tpw

    def enc(window, start):
        assert window.shape == (33, 4, 4, 3)
        return R.flatten_raw9(_latent_of_start(start)).copy() if tpw else np.full(768, float(start), np.float32)
    return FrameSampMemory(vision_enc_fn=lambda x: x, token_per_image=4, max_steps=512, motion_enc_fn=enc, motion_cfg=cfg)


def _drive(mem, es: int, nt: int):
    """评估节奏：首批整段 demo + 第 es 帧，之后每批 16 帧；每批后编完合法窗，产出该决策时刻 t。"""
    mem.exec_start_idx = es
    have = 0
    t = es
    while t < nt:
        for i in range(have, t + 1):
            mem._raw_frames[i] = np.full((4, 4, 3), i % 251, np.uint8)
        have = t + 1
        mem._encode_ready_windows(last_frame=t)
        yield t
        t += 16


@pytest.mark.parametrize("es,nt,budget,overflow", [(90, 300, 80, "raise"),     # demo 第 3 窗为补帧窗（64+33>90）
                                                    (100, 260, 80, "raise"),
                                                    (90, 400, 5, "resample")])  # 超预算按窗降采样
def test_online_buffer_matches_train_reference(R, es, nt, budget, overflow):
    import mme_vla_suite.policies.framesamp_memory as fsm
    from mme_vla_suite.shared.sampling import memory_order
    mem = _make_mem(R, budget, 9, overflow)
    assert mem.motion_tokens == budget * 9
    n_check, max_k, downs = 0, 0, 0
    for t in _drive(mem, es, nt):
        emb, pos, mask, times = mem._prepare_motion(t)
        r_emb, r_pos, r_mask, r_times, k = _ref_train_motion(R, mem.pos_emb, es, t, budget, fsm)
        assert emb.shape == (budget * 9, 16384) and pos.shape == (budget * 9, POS_DIM)
        assert emb.dtype == np.float32 and pos.dtype == np.float32 and mask.dtype == np.bool_ and times.dtype == np.int64
        assert emb.tobytes() == r_emb.tobytes()
        assert pos.tobytes() == r_pos.tobytes()
        assert mask.tobytes() == r_mask.tobytes()
        assert times.tobytes() == r_times.tobytes()
        ft = np.arange(8, dtype=np.int64) * 16
        assert memory_order(ft, 4, times).tobytes() == memory_order(ft, 4, r_times).tobytes()
        max_k = max(max_k, mem.motion_last_k)
        n_check += 1
    assert n_check >= 5
    if overflow == "resample":
        assert mem.motion_downsample_steps > 0 and max_k > budget
    else:
        assert mem.motion_downsample_steps == 0


class _TableStore:
    """内存版 raw9 表，接口同 S1 ``MotionStore.rows_into``。"""

    def __init__(self, table: np.ndarray):
        self._t = table

    def rows_into(self, rows, out) -> None:
        out[...] = self._t[np.asarray(rows, dtype=np.int64)]


class _PosStore:
    def __init__(self, pos: np.ndarray):
        self._pos = pos

    def pos_rows(self, frames) -> np.ndarray:
        return self._pos[np.asarray(frames, dtype=np.int64)]


@pytest.mark.parametrize("es,nt", [(90, 300), (100, 260), (10, 200), (50, 150)])
def test_online_buffer_matches_s2_dataset(R, es, nt):
    """同一 (episode, t)：在线 FrameSampMemory 的四键 == S2 FrameSampDataset._fill_motion_raw9 的四键（逐字节）。
    训练侧 index 条目由 S1 ``build_index_entries(LAYOUT_RAW9)`` 生成，表第 r 行 = 起点 f 的合成 latent；
    pos 两侧用同一张 pos 表（训练 store.pos_rows 与在线 pos_emb 同表是 v11 既有结论，这里只验 S3 的取行与展开）。"""
    from omegaconf import OmegaConf
    from mme_vla_suite.datastore import motion_store as ms
    from mme_vla_suite.training.framesamp_dataset import FrameSampDataset
    from mme_vla_suite.shared.sampling import even_sampling_indices, memory_order, pad_times
    budget, tpf, max_frames = 80, 4, 8
    eps = [{"global_episode_idx": 0, "h5_file": "record_dataset_BinFill.h5", "raw_ep_idx": 0,
            "num_timesteps": nt, "exec_start_idx": es}]
    entry = ms.build_index_entries({"episodes": eps}, ms.LAYOUT_SPECS[ms.LAYOUT_RAW9])[0]
    rows_all, f_all = ms.visible_motion_rows(entry, nt - 1)
    table = np.zeros((entry.demo.num_grid + entry.exec.num_grid,) + SHAPE, np.float32)
    for r, f in zip(rows_all.tolist(), f_all.tolist()):
        table[r] = _latent_of_start(f)
    ds = FrameSampDataset.__new__(FrameSampDataset)
    ds._motion_budget = budget
    ds._motion_tokens = R.motion_token_budget(OmegaConf.load(RAW9_YAML).motion)
    ds._motion_dim, ds._motion_pos_dim = 16384, POS_DIM
    ds._max_frames, ds._tokens_per_frame = max_frames, tpf
    mem = _make_mem(R, budget, 9)
    assert ds._motion_tokens == mem.motion_tokens == budget * 9
    mstore, pstore = _TableStore(table), _PosStore(mem.pos_emb)
    n_check = 0
    for t in _drive(mem, es, nt):
        emb, pos, mask, times = mem._prepare_motion(t)
        frames = np.asarray(even_sampling_indices(t, max_frames), dtype=np.int64)
        d_emb, d_pos, d_mask, d_order = ds._fill_motion_raw9(entry, t, mstore, pstore, frames)
        assert emb.tobytes() == d_emb.tobytes()
        assert pos.tobytes() == d_pos.tobytes()
        assert mask.tobytes() == d_mask.tobytes()
        on_order = memory_order(pad_times(frames, max_frames), tpf, times)
        assert on_order.dtype == d_order.dtype and on_order.tobytes() == d_order.tobytes()
        n_check += 1
    assert n_check >= 5 and mem.motion_downsample_steps == 0


def test_online_buffer_padded_demo_window_tau_truncated(R):
    mem = _make_mem(R, 80, 9)
    *_, last = list(_drive(mem, 90, 140))
    _, _, mask, times = mem._prepare_motion(last)
    # demo 窗起点 64：τ = 64,68,…,96 截到 es−1=89 → 后两个 token 同刻 89
    assert times[18:27].tolist() == [64, 68, 72, 76, 80, 84, 88, 89, 89]
    assert int(mask.sum()) % 9 == 0


def test_online_buffer_tpw1_shape_unchanged(R):
    from mme_vla_suite.shared.sampling import pad_times
    mem = _make_mem(R, 160, None)
    assert mem.motion_tpw == 1
    for t in _drive(mem, 90, 200):
        emb, pos, mask, times = mem._prepare_motion(t)
        frames = mem.visible_motion_frames(t)
        assert emb.shape == (160, 768) and pos.shape == (160, POS_DIM) and mask.shape == (160,)
        assert times.tobytes() == pad_times(frames, 160).tobytes()
        assert emb[:len(frames), 0].tolist() == [float(f) for f in frames]


def test_online_buffer_rejects_wrong_token_shape(R):
    mem = _make_mem(R, 80, 9)
    mem.motion_enc = lambda window, start: np.zeros(16384, np.float32)     # 旧形状 (dim,) 不再合法
    with pytest.raises(RuntimeError):
        list(_drive(mem, 90, 140))


# ── compare_online_raw9.py ────────────────────────────────────────────────────

def _frame(g: int, t: int) -> np.ndarray:
    return np.random.default_rng(1000 * g + t).integers(0, 256, (256, 256, 3), dtype=np.uint8)


def _fake_latent(window: np.ndarray) -> np.ndarray:
    seed = int(hashlib.sha256(np.ascontiguousarray(window).tobytes()).hexdigest()[:8], 16)
    return _latent(seed)


def _window(frames: list, off: int, seg_len: int) -> np.ndarray:
    real = min(33, seg_len - off)
    w = np.stack(frames[off:off + real])
    if real < 33:
        w = np.concatenate([w, np.repeat(np.stack(frames[seg_len - 1:seg_len]), 33 - real, axis=0)])
    return np.ascontiguousarray(w)


def _raw9_entries(manifest: dict):
    """stride-32 index 条目：S1 的 ms.build_index_entries(LAYOUT_RAW9)，并与独立推算互校。"""
    from mme_vla_suite.datastore import motion_store as ms
    spec = ms.LayoutSpec(17, 33, "repeat_last")
    entries, cursor = [], 0
    for g, ep in enumerate(manifest["episodes"]):
        lens = ms.segment_lengths(ep["num_timesteps"], ep["exec_start_idx"])
        segs = {}
        for seg in ms.SEGMENTS:
            nc = ms.seg_num_chunks(lens[seg], 17 if seg == "demo" else 33)
            ng = len(range(0, nc, 32))
            segs[seg] = ms.SegmentInfo(row_base=cursor if ng else None, num_grid=ng, num_chunks=nc, seg_len=lens[seg])
            cursor += ng
        entries.append(ms.IndexEntry(g=g, h5_file=ep["h5_file"], raw_ep_idx=int(ep["raw_ep_idx"]),
                                     num_timesteps=int(ep["num_timesteps"]), exec_start_idx=int(ep["exec_start_idx"]),
                                     demo=segs["demo"], exec=segs["exec"], spec=spec))
    real = ms.build_index_entries(manifest, ms.LAYOUT_SPECS[ms.LAYOUT_RAW9])
    assert [(e.demo, e.exec) for e in real] == [(e.demo, e.exec) for e in entries]
    assert all(e.spec.grid_stride == 32 for e in real)
    return real


def _build_lib(root: pathlib.Path, R, specs, with_frames=True):
    """specs: [(task, nt, es)]；每个 task 一条 episode（raw_ep_idx=0）。返回 (table, meta)。"""
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
    entries = _raw9_entries(manifest)
    rows_total = sum(e.demo.num_grid + e.exec.num_grid for e in entries)
    table = root / "motion-wan-raw9-s32"
    table.mkdir()
    tab = np.zeros((rows_total, 9, 16384), np.float32)
    segments, maps = {}, {}
    for e, (task, nt, es) in zip(entries, specs):
        frames = [_frame(e.g, t) for t in range(nt)] if with_frames else None
        for kind in ms.SEGMENTS:
            seg = getattr(e, kind)
            key = ms.segment_key(e.h5_file, e.raw_ep_idx, kind)
            s0 = 0 if kind == "demo" else es
            rows = []
            for m in range(len(range(0, seg.num_chunks, 16))):          # 源 wan-latents：stride-16 网格
                off = 16 * m
                r = {"m": m, "pad_frames": max(0, 33 - (seg.seg_len - off)), "input_frames_sha256": None}
                if with_frames:
                    w = _window(frames[s0:s0 + seg.seg_len], off, seg.seg_len)
                    r["input_frames_sha256"] = hashlib.sha256(w.tobytes()).hexdigest()
                    if m % 2 == 0 and m // 2 < seg.num_grid:              # 新表第 m' 行 = 源第 2m' 窗
                        tab[seg.row_base + m // 2] = R.flatten_raw9(_fake_latent(w))
                rows.append(r)
            segments[key] = {"segment_kind": kind, "seg_len": seg.seg_len, "rows": rows}
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
    (root / "wan-latents").mkdir()
    (root / "wan-latents/metadata.json").write_text(json.dumps({"manifest_sha256": manifest["sha256"], "segments": segments}))
    (root / "maps").mkdir()
    for task, rows in maps.items():
        (root / "maps" / f"record_dataset_{task}_episode_map.json").write_text(json.dumps(rows))
    meta = types.SimpleNamespace(
        entries=entries, spec=types.SimpleNamespace(grid_stride=32), num_rows=rows_total,
        manifest_sha256=manifest["sha256"],
        provenance={"encoder_kind": KIND, "readout": READOUT, "encoder_ckpt": None, "parameter_count": 0,
                    "raw9_source_sha256": R.source_sha256(), "vae": {"vae_state_sha256": "v"}})
    return table, meta


@pytest.fixture(scope="session")
def G4(R):
    return _load(COMPARE_PATH, "compare_online_raw9_under_test")


def _args(root, table, **kw):
    base = dict(table=str(table), source_root=str(root / "source"), h5_root=str(root / "h5"),
                episode_map_dir=str(root / "maps"), windows=6, padded=2, seed=0, gpu="0", dry_run=False)
    base.update(kw)
    return argparse.Namespace(**base)


class _FakeClient:
    """替身 sidecar + 客户端：VAE 换成确定性假 latent，返回与真客户端同形的 (9,16384)。"""

    def __init__(self, R):
        self.R, self.calls = R, []

    def __call__(self, window, start):
        self.calls.append(start)
        return self.R.flatten_raw9(_fake_latent(np.asarray(window))).copy()

    def close(self):
        pass


# 每条 episode：demo 段 50 帧 → 起点 0、32（32+33>50 为补帧窗）；exec 段 80 帧 → 起点 es+0、es+32
_SMALL = [("BinFill", 130, 50), ("RouteStick", 130, 50)]


def test_compare_load_table_meta_requires_stride32(G4, monkeypatch):
    import mme_vla_suite.datastore.motion_store as ms
    monkeypatch.setattr(ms.MotionMeta, "load", classmethod(lambda cls, t: types.SimpleNamespace(
        spec=types.SimpleNamespace(grid_stride=16))))
    with pytest.raises(G4.G4Fail):
        G4.load_table_meta(pathlib.Path("x"))


def test_compare_enumerates_stride32_and_src_m(tmp_path, R, G4):
    table, meta = _build_lib(tmp_path, R, _SMALL, with_frames=False)
    wins = G4.enumerate_windows(meta, 0)
    assert len(wins) == 8 == meta.num_rows and sorted(w.row for w in wins) == list(range(8))
    assert [(w.kind, w.start, w.padded, w.src_m) for w in wins if w.g == 1] == [
        ("demo", 0, False, 0), ("demo", 32, True, 2), ("exec", 50, False, 0), ("exec", 82, False, 2)]


def test_compare_dry_run_reproducible(tmp_path, R, G4, capsys, monkeypatch):
    # es ∈ {210,212,218,220}：demo 起点 192 满足 192+16 ≤ es−1 且 192+33 > es，每条恰 1 个补帧窗
    specs = [(t, 400, es) for t, es in zip(("BinFill", "RouteStick", "VideoRepick", "VideoUnmaskSwap"),
                                           (210, 212, 218, 220))] * 3
    table, meta = _build_lib(tmp_path, R, specs, with_frames=False)
    monkeypatch.setattr(G4, "load_table_meta", lambda t: meta)
    outs = []
    for _ in range(2):
        assert G4.run(_args(tmp_path, table, windows=32, padded=8, dry_run=True)) == 0
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1]
    assert outs[0].strip().splitlines()[-1].startswith("DRY_RUN windows=32 padded=8 ")
    wins = [json.loads(l[len("WINDOW "):]) for l in outs[0].splitlines() if l.startswith("WINDOW ")]
    assert len(wins) == 32 and sum(w["padded"] for w in wins) == 8
    assert all(w["source"] == ("h5" if w["kind"] == "demo" else "pkl") for w in wins)
    assert all(w["kind"] == "demo" and w["start"] % 32 == 0 for w in wins if w["padded"])
    es_of = {w["g"]: specs[w["g"]][2] for w in wins}
    assert all((w["start"] - (0 if w["kind"] == "demo" else es_of[w["g"]])) == 32 * w["m"] for w in wins)


def test_compare_real_mode_pass_and_dispatch(tmp_path, R, G4, capsys, monkeypatch):
    table, meta = _build_lib(tmp_path, R, _SMALL)
    monkeypatch.setattr(G4, "load_table_meta", lambda t: meta)
    used = {"h5": 0, "pkl": 0}
    orig_demo, orig_exec = G4.demo_window, G4.exec_window
    monkeypatch.setattr(G4, "demo_window", lambda *a, **k: (used.__setitem__("h5", used["h5"] + 1), orig_demo(*a, **k))[1])
    monkeypatch.setattr(G4, "exec_window", lambda *a, **k: (used.__setitem__("pkl", used["pkl"] + 1), orig_exec(*a, **k))[1])
    # 源 latent 奇数行（stride-16 的 m=1，新表不用）篡改不影响：证明帧摘要取的是第 2m' 行
    p = tmp_path / "wan-latents/metadata.json"
    lat = json.loads(p.read_text())
    for seg in lat["segments"].values():
        seg["rows"][1]["input_frames_sha256"] = "0" * 64
    p.write_text(json.dumps(lat))
    client = _FakeClient(R)
    assert G4.run(_args(tmp_path, table), client_factory=lambda meta, gpu: client) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ONLINE_RAW9=PASS windows=6 padded=2 bitexact=6"
    chosen = G4.select_windows(G4.enumerate_windows(meta, 0), 6, 2)
    assert used == {"h5": sum(w.kind == "demo" for w in chosen), "pkl": sum(w.kind == "exec" for w in chosen)}
    assert used["h5"] >= 2 and used["pkl"] >= 1 and sorted(client.calls) == sorted(w.start for w in chosen)


@pytest.mark.parametrize("tamper", ["map_missing", "h5_sha", "frame_sha", "table_row"])
def test_compare_tamper_fails(tmp_path, R, G4, capsys, monkeypatch, tamper):
    table, meta = _build_lib(tmp_path, R, _SMALL)
    monkeypatch.setattr(G4, "load_table_meta", lambda t: meta)
    if tamper == "map_missing":
        (tmp_path / "maps/record_dataset_RouteStick_episode_map.json").unlink()
    elif tamper == "h5_sha":
        p = tmp_path / "maps/record_dataset_BinFill_episode_map.json"
        rows = json.loads(p.read_text()); rows[0]["h5_sha256"] = "0" * 64; p.write_text(json.dumps(rows))
    elif tamper == "frame_sha":
        p = tmp_path / "wan-latents/metadata.json"
        lat = json.loads(p.read_text())
        lat["segments"]["BinFill_ep0_demo"]["rows"][2]["input_frames_sha256"] = "0" * 64     # 补帧窗 m'=1 ↔ 源 m=2
        p.write_text(json.dumps(lat))
    elif tamper == "table_row":
        tab = np.fromfile(table / "motion_token.f32.bin", np.float32).reshape(-1, 9 * 16384)
        tab[1, 5] = np.nextafter(tab[1, 5], np.float32(np.inf))                          # 补帧窗所在行
        (table / "motion_token.f32.bin").write_bytes(tab.tobytes())
    client = _FakeClient(R)
    monkeypatch.setattr(G4, "make_client", lambda meta, gpu: client)
    rc = G4.main([f"--table={table}", f"--source-root={tmp_path / 'source'}", f"--h5-root={tmp_path / 'h5'}",
                  f"--episode-map-dir={tmp_path / 'maps'}", "--windows=6", "--padded=2"])
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert rc == 1 and out.startswith("ONLINE_RAW9=FAIL"), out
    if tamper != "table_row":
        assert client.calls == []                                   # 摘要不过不送 sidecar
    else:
        assert out == "ONLINE_RAW9=FAIL windows=6 padded=2 bitexact=5"


def test_compare_make_client_checks_source_sha(R, G4):
    meta = types.SimpleNamespace(provenance={"raw9_source_sha256": "0" * 64, "vae": {}})
    with pytest.raises(G4.G4Fail):
        G4.make_client(meta, "0")


# ── check_shard.py ────────────────────────────────────────────────────────────

def _load_check_shard(monkeypatch, stride: str | None):
    pytest.importorskip("imageio")
    if stride is None:
        monkeypatch.delenv("EXPECT_MOTION_STRIDE", raising=False)
    else:
        monkeypatch.setenv("EXPECT_MOTION_STRIDE", stride)
    return _load(CHECK_SHARD, f"check_shard_s{stride}")


def test_check_shard_default_stride16_unchanged(monkeypatch):
    cs = _load_check_shard(monkeypatch, None)
    assert cs.MOTION_STRIDE == 16
    for es in range(0, 400):
        assert cs.k_demo_closed(es) == (0 if es < 17 else (es - 17) // 16 + 1)
    for t in range(0, 400):
        assert cs.k_exec_closed(t) == (0 if t < 32 else (t - 32) // 16 + 1)
    assert cs.k_total_closed(200, 10) == cs.k_demo_closed(200) + cs.k_exec_closed(144)


def test_check_shard_stride32_closed_form(monkeypatch, tmp_path):
    cs = _load_check_shard(monkeypatch, "32")
    assert cs.MOTION_STRIDE == 32
    for es in range(0, 400):        # 与在线 visible_motion_frames 同一合法条件
        assert cs.k_demo_closed(es) == len(range(0, max(es - 16, 0), 32))
    for t in range(0, 400):
        assert cs.k_exec_closed(t) == len(range(0, max(t - 31, 0), 32))
    # MOTION_WINDOWS 闸在 stride 32 / budget 80 窗口口径下的整链判定
    monkeypatch.setenv("EXPECT_MOTION_BUDGET", "80")
    stats = {}
    for i, (es, infers) in enumerate([(90, 40), (1200, 130)]):
        ks = [cs.k_total_closed(es, j) for j in range(1, infers + 1)]
        stats[f"BinFill/easy/{i}"] = {"exec_start_idx": es, "motion_infers": infers, "motion_k_max": max(ks),
                                      "motion_downsample_steps": sum(k > 80 for k in ks), "motion_budget": 80,
                                      "motion_overflow": "resample"}
    (tmp_path / "motion_stats.json").write_text(json.dumps(stats))
    progress = {"BinFill/easy": {"0": True, "1": False}}
    out = cs.check_motion_windows(tmp_path, progress)
    assert out["verdict"] == "PASS" and out["budget"] == 80 and out["downsample_steps"] > 0


# ── check_checkpoint.py ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def CC():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    return _load(CHECK_CKPT, "check_checkpoint_under_test")


RAW9_RUNS = {"v2-1600ep-m8x8-modul-wanraw9s32-b128-50k": ("50000",), "r9-smoke20": ("19",),
             "r9-smoke20-l1": ("19",), "r9-smoke20-l2": ("19",), "r9-smoke20-l3": ("19",)}


def test_check_checkpoint_runs_registered(CC):
    for name, steps in RAW9_RUNS.items():
        run = CC.RUNS[name]
        assert run["steps"] == steps and run["motion"] is True and run["config"] == "mme_vla_suite_b128_80k"
        assert run["budget"] == 512 and run["motion_budget"] == 80
        assert run["motion_kind"] == CC.WANRAW9_KIND == KIND and run["readout"] == CC.WANRAW9_READOUT == READOUT
        assert run["resolved_sha256"] == CC.WANRAW9_RESOLVED_SHA256
        assert run["norm_stats_sha256"] == CC.RUNS["v2-1600ep-m8x8-modul-wantp768-b128-50k"]["norm_stats_sha256"]
    # 旧条目未动
    assert CC.RUNS["w768-smoke20"]["motion_kind"] == "wan_fixedpool768" and CC.RUNS["w768-smoke20"]["motion_budget"] == 160


def _raw9_yaml_text() -> str:
    """S2 定稿 YAML。"""
    return RAW9_YAML.read_text(encoding="utf-8")


def test_check_checkpoint_resolved_sha_matches_yaml(CC, tmp_path):
    p = RAW9_YAML
    assert p.is_file()
    from omegaconf import OmegaConf
    m = OmegaConf.load(p).motion
    assert (int(m.dim), int(m.budget), int(m.stride), int(m.tokens_per_window), m.encoder_kind, m.readout) == \
        (16384, 80, 32, 9, KIND, READOUT)
    assert CC.resolved_sha256_of_yaml(p) == CC.WANRAW9_RESOLVED_SHA256


def test_check_checkpoint_assert_motion_raw9(CC, tmp_path):
    from omegaconf import OmegaConf
    p = tmp_path / "wanraw9s32.yaml"
    p.write_text(_raw9_yaml_text(), encoding="utf-8")
    cfg = OmegaConf.load(p)
    run = CC.RUNS["r9-smoke20"]
    prov = {"encoder_kind": KIND, "readout": READOUT, "raw9_source_sha256": "a" * 64, "vae": {"x": 1}, "encoder": None}
    CC.assert_motion(cfg, run, prov)
    for bad_cfg in ({"stride": 16}, {"dim": 768}, {"tokens_per_window": 1}, {"budget": 160}):
        c2 = OmegaConf.merge(cfg, {"motion": bad_cfg})
        with pytest.raises(AssertionError):
            CC.assert_motion(c2, run, prov)
    for bad_prov in ({"raw9_source_sha256": None}, {"encoder": {"c": 1}}, {"readout": "x"}, {"vae": None}):
        with pytest.raises(AssertionError):
            CC.assert_motion(cfg, run, {**prov, **bad_prov})
    # 池化条目回归：同一函数仍按池化分支断言
    pcfg = OmegaConf.load(WANTP768_YAML)
    CC.assert_motion(pcfg, CC.RUNS["w768-smoke20"], {"encoder_kind": "wan_fixedpool768",
                                                     "readout": "wan_raw_timepool8_adaptive6x8_v1",
                                                     "pool_source_sha256": "p" * 64, "vae": {}, "encoder": None})


# ── compare_train_infer_obs.py ────────────────────────────────────────────────

def test_tic_motion_subdir_default():
    src = TIC_PATH.read_text(encoding="utf-8")
    assert 'ap.add_argument("--motion-subdir", default="motion"' in src
    assert 'os.environ["MMEVLA_MOTION_STORE"] = str(lib / args.motion_subdir)' in src
    assert 'str(lib / "motion")' not in src



# ── compare_train_infer_obs.py：单 episode H5 读法（--h5-root / --episode-map-dir / --dry-run） ──────────

@pytest.fixture(scope="module")
def TIC():
    return _load(TIC_PATH, "compare_train_infer_obs_under_test")


def _write_single_h5(path: pathlib.Path, group: str, T: int, seed: int):
    import h5py
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group(group)
        g.create_group("setup").create_dataset("task_goal", data=np.array([f"goal {seed}".encode()]))
        for t in range(T):
            o = g.create_group(f"timestep_{t}").create_group("obs")
            o.create_dataset("front_rgb", data=rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
            o.create_dataset("wrist_rgb", data=rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
            o.create_dataset("joint_state", data=rng.standard_normal(7).astype(np.float32))
            o.create_dataset("gripper_state", data=rng.standard_normal(2).astype(np.float32))


def _tic_fixture(root: pathlib.Path, local: set[tuple[str, int]]):
    """两个任务、6 集（es 互有重复）；local 中的 (task, new_idx) 才落单 episode H5。返回 (manifest, h5_root, map_dir)。"""
    specs = [("TaskA", 0, 10, 4), ("TaskA", 1, 12, 4), ("TaskA", 2, 8, 6),
             ("TaskB", 0, 9, 4), ("TaskB", 1, 7, 2), ("TaskB", 2, 11, 8)]      # (task, new_idx, T, es)
    h5_root, map_dir = root / "raw-h5", root / "maps"
    map_dir.mkdir(parents=True)
    eps, maps = [], {}
    for g, (task, idx, T, es) in enumerate(specs):
        member = f"record_dataset_{task}_easy/hdf5_files/{task}_ep{idx + 10}_seed{g}.h5"
        row = {"task": task, "difficulty": "easy", "episode": idx + 10, "seed": g, "role": "primary", "member": member,
               "h5_sha256": "0" * 64, "h5_bytes": 1, "timestep_count": T, "new_idx": idx, "src_group": f"episode_{idx + 10}"}
        if (task, idx) in local:
            p = h5_root / member
            _write_single_h5(p, row["src_group"], T, g)
            row["h5_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            row["h5_bytes"] = p.stat().st_size
        maps.setdefault(task, []).append(row)
        eps.append({"global_episode_idx": g, "h5_file": f"record_dataset_{task}.h5", "raw_ep_idx": idx,
                    "num_timesteps": T, "exec_start_idx": es, "exec_samples": T - es, "total_sample_offset": 0})
    for task, rows in maps.items():
        (map_dir / f"record_dataset_{task}_episode_map.json").write_text(json.dumps(rows), encoding="utf-8")
    return {"raw_dir": str(root / "merged"), "episodes": eps}, h5_root, map_dir


def test_tic_single_h5_reads_same_as_merged(tmp_path, TIC):
    import h5py
    src = tmp_path / "single.h5"
    _write_single_h5(src, "episode_17", 6, 3)
    merged = tmp_path / "merged" / "record_dataset_TaskA.h5"
    merged.parent.mkdir()
    with h5py.File(src, "r") as s, h5py.File(merged, "w") as t:
        s.copy("episode_17", t, name="episode_2")                   # merge_v2_h5.py::merge_task 同法
    a = TIC.load_episode_full(merged.parent, merged.name, 2, 6, 4)
    b = TIC.load_episode_full(None, merged.name, 2, 6, 4, single={"path": src, "src_group": "episode_17"})
    for x, y in zip(a, b):
        assert (np.array_equal(x, y) and x.dtype == y.dtype) if isinstance(x, np.ndarray) else x == y
    with pytest.raises(SystemExit):
        TIC.load_episode_full(None, merged.name, 2, 6, 4, single={"path": src, "src_group": "episode_9"})


def test_tic_plan_merged_mode_unchanged(tmp_path, TIC):
    manifest, _, _ = _tic_fixture(tmp_path, set())
    plan = TIC.plan_episodes(manifest, None, None, None)
    assert plan["single"] is None and plan["missing"] == []
    assert (plan["picked"], plan["source"]) == TIC.resolve_episodes(manifest, TIC.DEFAULT_EPISODES)


def test_tic_plan_auto_pick_local(tmp_path, TIC, monkeypatch):
    monkeypatch.setattr(TIC, "N_ES", 3)
    local = {("TaskA", 0), ("TaskA", 1), ("TaskA", 2), ("TaskB", 1), ("TaskB", 2)}
    manifest, h5_root, map_dir = _tic_fixture(tmp_path, local)
    plan = TIC.plan_episodes(manifest, None, h5_root, map_dir)
    # 轮转：TaskA 最小 es=4（T 最长取 idx1），TaskB 最小未占用 es=2（idx1），TaskA 次小 es=6（idx2）
    assert plan["arg"] == "TaskB:1,TaskA:1,TaskA:2" and plan["missing"] == []
    assert len({e["exec_start_idx"] for e in plan["picked"]}) == 3
    assert set(plan["single"]) == {("record_dataset_TaskB.h5", 1), ("record_dataset_TaskA.h5", 1), ("record_dataset_TaskA.h5", 2)}


def test_tic_plan_lists_missing_h5(tmp_path, TIC, monkeypatch, capsys):
    monkeypatch.setattr(TIC, "N_ES", 3)
    manifest, h5_root, map_dir = _tic_fixture(tmp_path, {("TaskA", 0)})
    plan = TIC.plan_episodes(manifest, None, h5_root, map_dir)
    assert len(plan["picked"]) == 3 and len(plan["missing"]) == 2
    assert not TIC.report_plan(plan, pathlib.Path(manifest["raw_dir"]))
    out = capsys.readouterr().out
    lines = [json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("H5_EPISODE ")]
    assert len(lines) == 2
    for rec in lines:
        assert {"task", "raw_ep_idx", "member", "h5_sha256", "h5_bytes", "tar_object"} <= set(rec)
        assert rec["tar_object"] == f"record_dataset_{rec['task']}_easy.h5.tar.xz"
    assert "H5_COVERAGE=MISSING" in out


def test_tic_plan_explicit_episodes_and_sha_check(tmp_path, TIC):
    manifest, h5_root, map_dir = _tic_fixture(tmp_path, {("TaskA", 0), ("TaskB", 2)})
    plan = TIC.plan_episodes(manifest, "TaskA:0,TaskB:2", h5_root, map_dir)
    assert plan["source"] == "args" and plan["missing"] == [] and len(plan["single"]) == 2
    p = h5_root / json.loads((map_dir / "record_dataset_TaskA_episode_map.json").read_text())[0]["member"]
    data = bytearray(p.read_bytes())
    data[-1] ^= 0xFF                                         # 等长篡改：字节数仍对，sha256 不对
    p.write_bytes(bytes(data))
    with pytest.raises(SystemExit, match="sha256"):
        TIC.plan_episodes(manifest, "TaskA:0,TaskB:2", h5_root, map_dir)
    with pytest.raises(SystemExit, match="episode-map-dir"):
        TIC.plan_episodes(manifest, None, h5_root, None)


def test_tic_main_dry_run(tmp_path, TIC, monkeypatch, capsys):
    monkeypatch.setattr(TIC, "N_ES", 2)
    manifest, h5_root, map_dir = _tic_fixture(tmp_path, {("TaskA", 1), ("TaskB", 1)})
    lib = tmp_path / "lib"
    (lib / "meta").mkdir(parents=True)
    (lib / "meta" / "episode_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.chdir(tmp_path)                              # main 会 chdir 到仓库根，测试后还原
    base = ["x", "--lib", str(lib), "--h5-root", str(h5_root), "--episode-map-dir", str(map_dir), "--dry-run"]
    monkeypatch.setattr(sys, "argv", base)
    assert TIC.main() == 0
    out = capsys.readouterr().out
    assert "DRY_RUN episodes=2 complete=yes episodes_arg=TaskB:1,TaskA:1" in out
    monkeypatch.setattr(sys, "argv", base + ["--episodes", "TaskA:0,TaskB:1"])
    assert TIC.main() == 1
    out = capsys.readouterr().out
    assert "DRY_RUN episodes=2 complete=no" in out and out.count("H5_EPISODE ") == 1


# ── compare_train_infer_obs.py：L3 / L4 的 modulation 分支（G5 第 2 轮：f_full 漏传 mem_seq 崩溃） ──────────

def _tiny_hist_model(integration_type: str):
    """最小 history 模型（f32、budget=8、无 motion），CPU 现场 init。

    MemoryAttention 硬编码 width=1024 / head_dim=256，且两路 expert 须同 depth，故把 "dummy" 变体临时换成
    gemma_300m 的单层瘦身版（width 1024、depth 1、mlp 256），只在 init 期间替换（模块持有 config 对象，之后不再查表）。
    """
    import dataclasses
    import jax
    import omegaconf
    from mme_vla_suite.models.integration import history_gemma as HG
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
    hc = omegaconf.OmegaConf.load(YAML_DIR / "perceptual-framesamp-modul.yaml")
    hc.budget = 8
    hc.integration_type = integration_type
    tiny = dataclasses.replace(HG.get_config("gemma_300m"), depth=1, mlp_dim=256)
    orig = HG.get_config
    cfg = HistoryPi0Config(use_history=True, history_config=hc, dtype="float32", action_horizon=4, max_token_len=8,
                           paligemma_variant="dummy", action_expert_variant="dummy", pi05=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(HG, "get_config", lambda v: tiny if v == "dummy" else orig(v))
        return cfg.create(jax.random.key(0))


def _tiny_obs(mem_seed: int = 0):
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_observation import HistAugObservation
    B = 1
    return HistAugObservation(
        images={"base_0_rgb": jnp.zeros((B, 224, 224, 3)), "left_wrist_0_rgb": jnp.zeros((B, 224, 224, 3))},
        image_masks={"base_0_rgb": jnp.ones((B,), bool), "left_wrist_0_rgb": jnp.ones((B,), bool)},
        state=jnp.zeros((B, 32)), tokenized_prompt=jnp.zeros((B, 8), jnp.int32),
        tokenized_prompt_mask=jnp.ones((B, 8), bool), token_ar_mask=None, token_loss_mask=None,
        static_image_emb=jnp.asarray(np.random.default_rng(mem_seed).standard_normal((B, 8, 2048)), jnp.float32),
        static_mask=jnp.ones((B, 8), bool).at[:, 6:].set(False),
        static_pos_emb=jnp.zeros((B, 8, 768)), static_state_emb=jnp.zeros((B, 8, 8)))


@pytest.fixture(scope="module")
def MODUL():
    from flax import nnx
    import jax
    model = _tiny_hist_model("modulation")
    # pi05 adaRMS 门控是零初始化：新 init 时 FFN 支路（含记忆调制）贡献恒为 0，记忆改了 v_t 也不动。
    # 把全零参数换成小随机值，让「记忆确实进入 L4」可被检出（只影响本夹具，不涉及被测代码）。
    params = nnx.state(model, nnx.Param)
    pure = params.to_pure_dict()
    leaves, treedef = jax.tree.flatten(pure)
    keys = jax.random.split(jax.random.key(7), len(leaves))
    leaves = [0.02 * jax.random.normal(k, x.shape, x.dtype) if not np.any(np.asarray(x)) else x for k, x in zip(keys, leaves)]
    params.replace_by_pure_dict(jax.tree.unflatten(treedef, leaves))
    nnx.update(model, params)
    gdef, state = nnx.split(model)
    return model, gdef, state


def test_tic_l4_modulation_runs_and_matches_model(TIC, MODUL):
    """f_full / f_cached 在 modulation 下不崩，且与 history_pi0 自身的 compute_loss 主体 / sample_actions 单步同值。"""
    import jax
    import jax.numpy as jnp
    import einops
    from mme_vla_suite.models.integration.history_observation import preprocess_observation
    from mme_vla_suite.models.integration.history_pi0 import make_attn_mask
    model, gdef, state = MODUL
    assert model.integration_type == "modulation"
    f_full, f_cached = TIC.make_l4_fns(gdef, "modulation")
    obs = _tiny_obs()
    x_t = jax.random.normal(jax.random.key(1), (1, 4, 32))
    ts = jnp.asarray([0.5], jnp.float32)
    fu, ca = f_full(state, obs, x_t, ts), f_cached(state, obs, x_t, ts)
    assert fu["v_t"].shape == ca["v_t"].shape == (1, 4, 32)
    assert np.isfinite(np.asarray(fu["v_t"])).all() and np.isfinite(np.asarray(ca["v_t"])).all()
    # L4 结构判据在 modulation 下仍成立：前缀不含记忆，整段 attn 左上块（带 na）与缓存前缀 attn（不带 na）逐位相同
    P = int(fu["prefix_len"])
    assert np.array_equal(np.asarray(fu["attn"])[:, :P, :P], np.asarray(ca["prefix_attn"]))
    assert np.array_equal(np.asarray(fu["attn"])[:, P:, :], np.asarray(ca["full_attn"]))
    # 参照：sample_actions modulation 分支的一步（逐行照抄 history_pi0），与 f_cached 必须逐位相同
    o = preprocess_observation(None, obs, train=False)
    pt, pm, par_, _ = model.embed_prefix(o)
    _, kv = model.PaliGemma.llm([pt, None], mask=make_attn_mask(pm, par_), positions=jnp.cumsum(pm, axis=1) - 1)
    mem_seq, mem_mask, _, _ = model.embed_memory(o)
    st, sm, sar, _, ad = model.embed_suffix(o, x_t, ts)
    full = jnp.concatenate([einops.repeat(pm, "b p -> b s p", s=st.shape[1]), make_attn_mask(sm, sar)], axis=-1)
    spos = jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
    (_, so), _ = model.PaliGemma.llm([None, st], mask=full, positions=spos, kv_cache=kv, adarms_cond=[None, ad],
                                     mem_seq=[None, mem_seq], mem_mask=[None, mem_mask])
    ref = model.action_out_proj(so[:, -model.action_horizon:])
    assert np.allclose(np.asarray(ref), np.asarray(ca["v_t"]), atol=1e-5)
    # 记忆确实进了 L4 两路：改记忆内容，v_t 必须变
    fu2, ca2 = f_full(state, _tiny_obs(1), x_t, ts), f_cached(state, _tiny_obs(1), x_t, ts)
    assert not np.array_equal(np.asarray(fu["v_t"]), np.asarray(fu2["v_t"]))
    assert not np.array_equal(np.asarray(ca["v_t"]), np.asarray(ca2["v_t"]))
    # 前缀不含记忆：L4 整段与缓存的 v_t 数值同量级（f32 下非逐位，只要求接近）
    assert float(np.abs(np.asarray(fu["v_t"]) - np.asarray(ca["v_t"])).max()) < 1e-3


def test_tic_l3_modulation_prefix_excludes_memory(TIC, MODUL):
    _, gdef, state = MODUL
    o = TIC.make_l3_fn(gdef, "modulation", False)(state, _tiny_obs())
    txt = int(o["txt_len"])
    assert o["mem_tok"].shape[1] == 8 and o["tokens"].shape[1] - txt > 0
    assert (o["tokens"].shape[1] - txt) % 2 == 0                  # 两路视图的图像 token，记忆不在前缀里
    assert np.asarray(o["k"]).shape[2] == o["tokens"].shape[1]


def test_tic_l4_context_path_source_unchanged():
    """context 分支仍是旧写法（前缀 pass 带 na_mask、llm 不传 mem_seq）。"""
    src = TIC_PATH.read_text(encoding="utf-8")
    assert "pattn = make_attn_mask(pm, par_, pna)" in src
    assert "(_, so), kv = m.PaliGemma.llm([pt, st], mask=attn, positions=pos, adarms_cond=[None, ad])" in src
    assert "attn = make_attn_mask(mask, ar, na)" in src


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
    rc = pytest.main([__file__, "-q", "-p", "no:cacheprovider", "-s"], plugins=[c])
    ok = rc == 0 and c.failed == 0 and c.skipped == 0
    print(f"S3_EVAL_UNIT={'PASS' if ok else 'FAIL'} passed={c.passed} failed={c.failed} skipped={c.skipped}")
    sys.exit(0 if ok else 1)
