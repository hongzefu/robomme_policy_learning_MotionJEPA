"""S4（在线侧）单测：计划 1117-e2e-enc-train-plan.html 第二部分「二」分配表 S4 行、「一」在线侧各文件行、三节 G5 / G5b。

覆盖：
- sidecar：`--encoder-kind wan_e2e` 被接受、`raw9_info` 回 e2e 身份（raw9_source_sha256 仍取 wan_raw9.py），stub 档经真实 socket
  握手通过并回 589,824 B fp32；装了 wan venv 时另以真实 `MotionEncoderClient(stub=True)` 子进程启动并握手（「sidecar 以 e2e kind 启动且握手通过」）；
- 客户端：e2e 构造参数校验、握手 kind / readout / raw9_source_sha256 任一不符即拒；`__call__` 对 sidecar fp32 调 `raw9_f32_to_bf16`
  后与直接调用逐字节相等、持有的转换函数对象就是 `raw_bf16.raw9_f32_to_bf16`；
- 在线缓冲：每槽 (9,16384) bf16、时间码 = 窗起点、emb/pos/mask/mem_order 与独立参考实现逐字节相等（含补尾 demo 窗）；
  长缓冲 k=159/160/161（budget 160、resample）窗身份、时间码、mask、重排与一次实际降采样；返回 f32 即拒；
- policy：`_motion_cfg` 拷 e2e 键（旧 YAML 逐键不变），装配出 (1440,16384) bf16 / (160,256) / (160,) / 672 的 mem_order；
  `MMEVLA_PERF_LOG` 逐次落 infer 行、reset 落 episode 行；
- policy_config：e2e 快照不落入 `_make_pool_motion_client`，provenance 契约逐项拒绝；
- check_checkpoint：e2e 分支 PASS、交换新旧 kind / readout 必失败、RUNS 条目；YAML 存在时 resolved sha 与常量一致；
- compare_train_infer_obs：`--acceptance` 指定样本缺失 / (h5,raw) 重复 / exec_start_idx 重复即 FAIL 不回退、不给 --episodes 即 FAIL、
  覆盖条件逐项、显式参数检查、e2e 模式 MAX_STEPS=2000、dry-run 整条入口、`--perf-report` 判定行；
- HF 导出副本：假 bucket 夹具下每 step 读回 PASS 后 `$VERIFY/<step>` 被删并记 `VERIFY_DELETED`，阶段 2 不重下已删 step。

`raw_bf16`（S2 写集）缺失时依赖它的用例 skip；设 `S4_RAW_BF16_STANDIN=1` 时改用按计划 R9 契约写的测试内替身
（`np.asarray(x, np.float32).astype(ml_dtypes.bfloat16)`，只注入 sys.modules，不落盘），S2 合入后不设该变量重跑。
运行：``uv run --no-sync python -m pytest scripts/training/tests/test_e2e_online.py -q``。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types

import ml_dtypes
import numpy as np
import pytest

WT = pathlib.Path(__file__).resolve().parents[3]
RAW9_PATH = WT / "src/mme_vla_suite/shared/wan_raw9.py"
SIDECAR_PATH = WT / "scripts/dataset/wan/motion_sidecar.py"
CHECK_CKPT = WT / "scripts/evaluation/check_checkpoint.py"
TIC_PATH = WT / "scripts/training/g0/compare_train_infer_obs.py"
EXPORT_SH = WT / "scripts/dataset/hf_export/run_e2eenc_ckpt_export.sh"
YAML_DIR = WT / "src/mme_vla_suite/models/config/robomme"
E2E_YAML = YAML_DIR / "perceptual-framesamp-modul-8frame-8x8-e2eenc.yaml"
KIND, READOUT = "wan_e2e", "wan_e2e_enc768_s16_v1"
RAW9_KIND, RAW9_READOUT = "wan_raw9", "wan_raw9_flat16384_s32_v1"
SHAPE = (9, 16, 32, 32)
BF16 = np.dtype(ml_dtypes.bfloat16)
MAIN_V1 = pathlib.Path("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store")


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_standin():
    mod = types.ModuleType("mme_vla_suite.shared.raw_bf16")

    def raw9_f32_to_bf16(x):
        x = np.asarray(x, np.float32)
        if not np.isfinite(x).all():
            raise ValueError("非有限")
        return x.astype(ml_dtypes.bfloat16)
    mod.raw9_f32_to_bf16 = raw9_f32_to_bf16
    mod.__s4_standin__ = True
    sys.modules["mme_vla_suite.shared.raw_bf16"] = mod
    import mme_vla_suite.shared as sh
    sh.raw_bf16 = mod
    return mod


@pytest.fixture(scope="session")
def RB():
    """S2 的 raw_bf16；缺失时 skip（或按 S4_RAW_BF16_STANDIN=1 注入契约替身）。"""
    try:
        from mme_vla_suite.shared import raw_bf16
        return raw_bf16
    except ImportError:
        if os.environ.get("S4_RAW_BF16_STANDIN") == "1":
            return _install_standin()
        pytest.skip("mme_vla_suite.shared.raw_bf16（S2）尚未合入：待合入后重跑")


@pytest.fixture(scope="session")
def R():
    from mme_vla_suite.shared import wan_raw9 as mod
    assert pathlib.Path(mod.__file__).resolve() == RAW9_PATH
    return mod


def test_imports_point_to_worktree():
    import mme_vla_suite
    assert pathlib.Path(mme_vla_suite.__file__).resolve().is_relative_to(WT / "src")


def _latent(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(SHAPE).astype(np.float32)


# ── sidecar ────────────────────────────────────────────────────────────────────

def test_sidecar_e2e_info_and_choices(R):
    sc = _load(SIDECAR_PATH, "motion_sidecar_e2e_info")
    assert sc.E2E_KINDS == (KIND,) and sc.E2E_READOUT == READOUT
    info = sc.raw9_info(R, KIND)
    assert info == {"encoder_kind": KIND, "readout": READOUT, "raw9_source_sha256": R.source_sha256(),
                    "raw9_path": str(RAW9_PATH), "encoder": None}
    old = sc.raw9_info(R, RAW9_KIND)                        # wan_raw9 握手逐值不变
    assert old["encoder_kind"] == RAW9_KIND and old["readout"] == RAW9_READOUT
    with pytest.raises(RuntimeError):
        sc.raw9_info(R, "wan_fixedpool768")


def _run_stub_sidecar(sc, monkeypatch, kind: str):
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    fd = child.detach()
    monkeypatch.setattr(sys, "argv", ["motion_sidecar.py", "--fd", str(fd), "--stub", "--encoder-kind", kind])
    rc = []
    t = threading.Thread(target=lambda: rc.append(sc.main()), daemon=True)
    t.start()
    return parent, t, rc


def _bare_client(**kw):
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    c = MotionEncoderClient.__new__(MotionEncoderClient)
    c._lock, c.n_calls, c.total_s = threading.Lock(), 0, 0.0
    c.bytes_sent = c.bytes_recv = 0
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_sidecar_stub_e2e_handshake_and_client_bf16(R, RB, monkeypatch):
    """stub 档整条链：sidecar 以 e2e kind 握手 → 589,824 B fp32 → 客户端 raw9_f32_to_bf16 → (9,16384) bf16。"""
    from mme_vla_suite.policies import motion_protocol as P
    sc = _load(SIDECAR_PATH, "motion_sidecar_e2e_stub")
    parent, t, rc = _run_stub_sidecar(sc, monkeypatch, KIND)
    try:
        info = P.recv_handshake(parent, time.monotonic() + 30)
        assert info["stub"] is True and info["encoder_kind"] == KIND and info["readout"] == READOUT
        assert info["raw9_source_sha256"] == R.source_sha256() and info["encoder"] is None
        c = _bare_client(encoder_kind=KIND, readout=READOUT, expected_raw9_sha256=R.source_sha256(), provenance=info,
                         _bf16_fn=RB.raw9_f32_to_bf16, keep_last_raw_f32=True, last_raw_f32=None,
                         _sock=parent, _proc=types.SimpleNamespace(poll=lambda: None, returncode=None))
        c.check_pool_handshake()
        for start in (5, 37):
            window = np.stack([P.stub_frame(i) for i in range(start, start + 33)])
            tok = c(window, start)
            assert tok.shape == (9, 16384) and tok.dtype == BF16
            want = RB.raw9_f32_to_bf16(np.full((9, 16384), float(start), np.float32))
            assert tok.tobytes() == want.tobytes()
            assert c.last_raw_f32.dtype == np.float32
        assert c.n_calls == 2 and c.bytes_recv == 2 * P.RAW9_RESPONSE_BYTES and c.bytes_sent > 2 * 33 * 256 * 256 * 3
        P.send_all(parent, P.encode_shutdown())
        t.join(30)
        c._sock = c._proc = None
    finally:
        parent.close()
    assert rc == [0]


@pytest.mark.skipif(os.environ.get("S4_REAL_SIDECAR") != "1" or not (MAIN_V1 / "venvs/wan").is_dir(),
                    reason="真实子进程握手须显式 S4_REAL_SIDECAR=1（会用 v1-store/venvs/wan 起 stub sidecar，合入后由主会话跑）")
def test_real_client_spawns_e2e_sidecar_stub(R, RB, monkeypatch):
    """真实 MotionEncoderClient(stub=True)：Popen(uv run … motion_sidecar.py --stub --encoder-kind wan_e2e) → 握手通过。"""
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    from mme_vla_suite.policies import motion_protocol as P
    monkeypatch.setenv("MMEVLA_V1_STORE", str(MAIN_V1))
    c = MotionEncoderClient(online_gpu="", stub=True, encoder_kind=KIND, readout=READOUT,
                            expected_raw9_sha256=R.source_sha256(), handshake_timeout=120)
    try:
        assert c.provenance["encoder_kind"] == KIND and c.provenance["readout"] == READOUT
        assert "--encoder-kind" in c.argv and c.argv[c.argv.index("--encoder-kind") + 1] == KIND
        window = np.stack([P.stub_frame(i) for i in range(16, 49)])
        tok = c(window, 16)
        assert tok.dtype == BF16 and tok.tobytes() == RB.raw9_f32_to_bf16(np.full((9, 16384), 16.0, np.float32)).tobytes()
    finally:
        c.close()


# ── 客户端 ───────────────────────────────────────────────────────────────────

def test_client_e2e_constructor_validation(RB):
    from mme_vla_suite.policies.motion_client import MotionEncoderClient, E2E_ENCODER_KINDS, RAW_LATENT_KINDS
    assert E2E_ENCODER_KINDS == (KIND,) and RAW_LATENT_KINDS == (RAW9_KIND, KIND)
    for kw in ({"readout": None, "expected_raw9_sha256": "a" * 64},
               {"readout": READOUT, "expected_raw9_sha256": None},
               {"readout": RAW9_READOUT, "expected_raw9_sha256": "a" * 64},           # 旧 readout 配新 kind
               {"readout": READOUT, "expected_raw9_sha256": "a" * 64, "source_run": "r/checkpoint_epoch_1.pt#encoder"}):
        with pytest.raises(ValueError):
            MotionEncoderClient(online_gpu=0, encoder_kind=KIND, **kw)


@pytest.mark.parametrize("bad", ["encoder_kind", "readout", "raw9_source_sha256", None])
def test_client_e2e_handshake_rejects_mismatch(bad):
    prov = {"encoder_kind": KIND, "readout": READOUT, "raw9_source_sha256": "b" * 64}
    if bad == "encoder_kind":
        prov[bad] = RAW9_KIND                                   # 新旧 kind 互换
    elif bad == "readout":
        prov[bad] = RAW9_READOUT
    elif bad:
        prov[bad] = "x"
    c = _bare_client(encoder_kind=KIND, readout=READOUT, expected_raw9_sha256="b" * 64, expected_pool_sha256=None,
                     provenance=prov, _sock=None, _proc=None)
    if bad:
        with pytest.raises(RuntimeError):
            c.check_pool_handshake()
    else:
        c.check_pool_handshake()


def test_client_e2e_call_bytes_equal_direct_conversion(R, RB):
    """客户端对 sidecar 回的 fp32 调 raw9_f32_to_bf16 后，与对同一 fp32 直接调用逐字节相等（含舍入平局与负零）。"""
    from mme_vla_suite.policies import motion_protocol as P
    lat = _latent(11)
    lat.reshape(-1)[:4] = np.array([1.0 + 2 ** -8, -0.0, 3e-39, 1.0 + 3 * 2 ** -8], np.float32)   # RNE 平局 / 负零 / 次正规
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    def serve():
        P.recv_request(child, time.monotonic() + 30)
        P.send_all(child, P.encode_response(lat.tobytes(), P.RAW9_RESPONSE_BYTES))
    th = threading.Thread(target=serve, daemon=True)
    th.start()
    try:
        c = _bare_client(encoder_kind=KIND, _bf16_fn=RB.raw9_f32_to_bf16, keep_last_raw_f32=False,
                         _sock=parent, _proc=types.SimpleNamespace(poll=lambda: None, returncode=None))
        tok = c(np.zeros((33, 256, 256, 3), np.uint8), 0)
        th.join(10)
        direct = RB.raw9_f32_to_bf16(R.flatten_raw9(lat.copy()))
        assert tok.dtype == BF16 and tok.shape == (9, 16384) and tok.tobytes() == direct.tobytes()
        assert c._bf16_fn is RB.raw9_f32_to_bf16
        c._sock = c._proc = None
    finally:
        parent.close()
        child.close()


# ── 在线缓冲 ─────────────────────────────────────────────────────────────────

STRIDE, WIN, DEMO_MIN, POS_DIM = 16, 33, 17, 256


def _slot_of(f: int) -> np.ndarray:
    return _latent(20_000 + int(f)).reshape(9, 16384).astype(ml_dtypes.bfloat16)


def _e2e_cfg(budget=160, overflow="raise", **extra):
    cfg = {"stride": STRIDE, "window_frames": WIN, "budget": budget, "frame_size": 4, "pos_dim": POS_DIM, "dim": 768,
           "window_direction": "forward", "grid_origin": "segment_start", "demo_min_real_frames": DEMO_MIN,
           "demo_tail_pad": "repeat_last", "overflow": overflow, "encoder_kind": KIND, "readout": READOUT,
           "latents_per_window": 9, "raw_dim": 16384, "tokens_per_window": 1}
    cfg.update(extra)
    return cfg


def _make_mem(budget=160, overflow="raise", max_steps=1024, enc=None):
    from mme_vla_suite.policies.framesamp_memory import FrameSampMemory

    def default_enc(window, start):
        assert window.shape == (33, 4, 4, 3)
        return _slot_of(start).copy()
    return FrameSampMemory(vision_enc_fn=lambda x: x, token_per_image=4, max_steps=max_steps,
                           motion_enc_fn=enc or default_enc, motion_cfg=_e2e_cfg(budget, overflow))


def _drive(mem, es: int, nt: int):
    mem.exec_start_idx = es
    have, t = 0, es
    while t < nt:
        for i in range(have, t + 1):
            mem._raw_frames[i] = np.full((4, 4, 3), i % 251, np.uint8)
        have = t + 1
        mem._encode_ready_windows(last_frame=t)
        yield t
        t += 16


def _ref_frames(es, t):
    demo = list(range(0, max(es - (DEMO_MIN - 1), 0), STRIDE))
    exec_ = [es + u for u in range(0, max(t - es - (WIN - 1) + 1, 0), STRIDE)]
    return demo, exec_


def _ref_assemble(pos_emb, frames, budget):
    from mme_vla_suite.shared.sampling import pad_times
    emb = np.zeros((budget * 9, 16384), BF16)
    pos = np.zeros((budget, POS_DIM), np.float32)
    for i, f in enumerate(frames):
        emb[9 * i:9 * i + 9] = _slot_of(f)
        pos[i] = pos_emb[f, 0, :POS_DIM]
    mask = np.zeros(budget, np.bool_)
    mask[:len(frames)] = True
    return emb, pos, mask, pad_times(frames, budget)


@pytest.mark.parametrize("es,nt", [(90, 300), (100, 260), (678, 900)])     # 90/678：最后一个 demo 窗为补尾窗
def test_online_buffer_e2e_matches_reference(es, nt):
    from mme_vla_suite.shared.sampling import memory_order
    mem = _make_mem()
    n_pad = 0
    for t in _drive(mem, es, nt):
        emb, pos, mask, times = mem._prepare_motion(t)
        demo, exec_ = _ref_frames(es, t)
        frames = demo + exec_
        n_pad = sum(1 for f in demo if es - f < WIN)
        r_emb, r_pos, r_mask, r_times = _ref_assemble(mem.pos_emb, frames, 160)
        assert emb.dtype == BF16 and emb.shape == (1440, 16384)
        assert emb.tobytes() == r_emb.tobytes() and pos.tobytes() == r_pos.tobytes()
        assert np.array_equal(mask, r_mask) and np.array_equal(times, r_times)
        assert list(times[:len(frames)]) == frames                      # 时间码 = 窗起点
        ft = np.full(8, 10 ** 9, np.int64)
        assert np.array_equal(memory_order(ft, 4, times), memory_order(ft, 4, r_times))
    assert n_pad >= 1


def test_online_buffer_e2e_rejects_f32_slot():
    mem = _make_mem(enc=lambda window, start: _slot_of(start).astype(np.float32))
    with pytest.raises(RuntimeError, match="bfloat16"):
        list(_drive(mem, 40, 41))


def test_online_buffer_e2e_contract_keys():
    from mme_vla_suite.policies.framesamp_memory import FrameSampMemory
    with pytest.raises(ValueError):
        FrameSampMemory(vision_enc_fn=lambda x: x, token_per_image=4, max_steps=64, motion_enc_fn=lambda w, s: None,
                        motion_cfg=_e2e_cfg(tokens_per_window=9))
    with pytest.raises(KeyError):
        cfg = _e2e_cfg()
        del cfg["latents_per_window"]
        FrameSampMemory(vision_enc_fn=lambda x: x, token_per_image=4, max_steps=64, motion_enc_fn=lambda w, s: None,
                        motion_cfg=cfg)


def test_online_buffer_e2e_long_k159_160_161_downsample(capsys):
    """构造长缓冲（es=600：demo 37 窗）：k=159/160/161 的窗身份、时间码、mask、重排，k=161 恰触发一次实际降采样。"""
    import mme_vla_suite.policies.framesamp_memory as fsm
    from mme_vla_suite.shared.sampling import memory_order
    es, B = 600, 160
    mem = _make_mem(budget=B, overflow="resample", max_steps=4096)
    seen = {}
    for t in _drive(mem, es, 3000):
        k = len(mem.visible_motion_frames(t))
        if k in (159, 160, 161):
            seen[k] = (t, mem._prepare_motion(t), mem.motion_downsample_steps)
        if k > 161:
            break
    assert sorted(seen) == [159, 160, 161]
    for k in (159, 160, 161):
        t, (emb, pos, mask, times), ds = seen[k]
        demo, exec_ = _ref_frames(es, t)
        assert len(demo) == 37 and len(demo) + len(exec_) == k
        if k <= B:
            frames = demo + exec_
            assert ds == 0
        else:
            keep_demo, keep_exec = min(len(demo), B // 2), None
            keep_exec = min(len(exec_), B - keep_demo)
            keep_demo = B - keep_exec
            idx = [int(i) for i in np.linspace(0, len(exec_) - 1, keep_exec).astype(np.int64)]
            frames = demo[:keep_demo] + [exec_[i] for i in idx]
            assert ds == 1 and mem.motion_downsample_max_k == 161
            assert len(frames) == B and len(set(frames)) == B and frames == sorted(frames)
            assert set(demo) <= set(frames)                              # demo 全保
            dropped = sorted(set(demo + exec_) - set(frames))
            assert len(dropped) == 1 and dropped[0] >= es                # 只丢 1 个 exec 窗
        r_emb, r_pos, r_mask, r_times = _ref_assemble(mem.pos_emb, frames, B)
        assert emb.tobytes() == r_emb.tobytes() and pos.tobytes() == r_pos.tobytes()
        assert int(mask.sum()) == min(k, B) and np.array_equal(mask, r_mask)
        assert np.array_equal(times, r_times)
        ft = np.arange(8, dtype=np.int64) * 300
        order = memory_order(ft, 4, times)
        assert np.array_equal(order, memory_order(ft, 4, r_times)) and order.shape == (32 + B,)
    assert "MOTION_DOWNSAMPLE" in capsys.readouterr().out
    assert fsm._motion_quota(37, 124, 160) == (37, 123)


# ── policy：_motion_cfg + 装配 + perf 日志 ───────────────────────────────────────

def _policy_cfg(e2e=True):
    from omegaconf import OmegaConf
    m = {"enabled": True, "dim": 768, "budget": 160, "demo_min_real_frames": 17, "demo_tail_pad": "repeat_last",
         "stride": 16, "window_frames": 33, "window_direction": "forward", "grid_origin": "segment_start",
         "pos_dim": 256, "frame_size": 4}
    if e2e:
        m.update(encoder_kind=KIND, readout=READOUT, raw_dim=16384, latents_per_window=9,
                 encoder={"layers": 1, "width": 768})
    return OmegaConf.create({"budget": 32, "num_views": 1, "token_per_image": 4,
                             "memory_feature": {"img": {"input_dim": 2048}, "pos": {"input_dim": 768},
                                                "state": {"input_dim": 8}}, "motion": m})


def _make_policy(monkeypatch, cfg, enc, perf_path=None):
    import jax.numpy as jnp
    import mme_vla_suite.policies.policy as pol
    from openpi.shared import nnx_utils
    monkeypatch.setattr(nnx_utils, "module_jit", lambda f: f)
    if perf_path:
        monkeypatch.setenv("MMEVLA_PERF_LOG", str(perf_path))
    else:
        monkeypatch.delenv("MMEVLA_PERF_LOG", raising=False)

    class FakeObs:
        @staticmethod
        def from_dict(d):
            return types.SimpleNamespace(state=d["state"], _d=d)
    monkeypatch.setattr(pol, "HistAugObservation", FakeObs)
    model = types.SimpleNamespace(
        sample_actions=lambda rng, obs, **kw: jnp.zeros((1, 4, 8)),
        vision_encode=lambda x: jnp.zeros((x.shape[0], x.shape[1], 256, 2048), jnp.bfloat16),
        history_config=cfg)
    ns = types.SimpleNamespace(mean=np.zeros(8, np.float32), std=np.ones(8, np.float32))
    return pol.MME_VLA_Policy(model, norm_stats={"state": ns}, motion_enc_fn=enc)


def test_policy_motion_cfg_e2e_keys_and_assembly(monkeypatch, tmp_path):
    def enc(window, start):
        return _slot_of(start).copy()
    perf = tmp_path / "perf.jsonl"
    p = _make_policy(monkeypatch, _policy_cfg(True), enc, perf)
    assert p._motion_cfg["encoder_kind"] == KIND and p._motion_cfg["latents_per_window"] == 9
    assert p._motion_cfg["raw_dim"] == 16384 and p._motion_cfg["readout"] == READOUT
    assert p.mem_buffer.motion_e2e
    es = 40
    imgs = np.zeros((es + 1, 1, 4, 4, 3), np.uint8)
    p.add_buffer({"images": imgs, "state": np.zeros((es + 1, 8), np.float32), "exec_start_idx": es})
    for _ in range(3):
        p.add_buffer({"images": np.zeros((16, 1, 4, 4, 3), np.uint8), "state": np.zeros((16, 8), np.float32)})
        inputs = p._prepare_history({"state": np.zeros(8, np.float32)})
        assert inputs["motion_emb"].dtype == BF16 and inputs["motion_emb"].shape == (1440, 16384)
        assert inputs["motion_pos"].shape == (160, 256) and inputs["motion_mask"].shape == (160,)
        assert inputs["mem_order"].shape == (32 + 160,)
        p.infer({"state": np.zeros(8, np.float32)})
    p.reset()
    recs = [json.loads(x) for x in perf.read_text().splitlines()]
    infer = [r for r in recs if r["event"] == "infer"]
    ep = [r for r in recs if r["event"] == "episode"]
    assert len(infer) == 3 and len(ep) == 1
    assert ep[0]["n_add_buffer"] == 4 and len(ep[0]["add_buffer_ms"]) == 4 and ep[0]["n_infer"] == 3
    assert ep[0]["last_step_idx"] == es + 48 and ep[0]["wall_s"] > 0


def test_policy_motion_cfg_old_yaml_unchanged(monkeypatch):
    p = _make_policy(monkeypatch, _policy_cfg(False), lambda w, s: np.zeros(768, np.float32))
    assert set(p._motion_cfg) == {"stride", "window_frames", "budget", "frame_size", "pos_dim", "dim", "window_direction",
                                  "grid_origin", "demo_min_real_frames", "demo_tail_pad", "tokens_per_window", "overflow"}
    assert not p.mem_buffer.motion_e2e and p._perf_path is None


# ── policy_config ────────────────────────────────────────────────────────────

def _e2e_prov(R, **over):
    prov = {"vae": {"vae_state_sha256": "v"}, "encoder": None, "encoder_kind": KIND, "readout": READOUT,
            "table_kind": "wan_raw_bf16_s16_v1", "raw9_source_sha256": R.source_sha256(),
            "policy_encoder": {"init": "random", "trainable": True, "layers": 1, "parameter_count": 20252928,
                               "spec_src": "d3213de"}}
    prov.update(over)
    return prov


def test_policy_config_e2e_dispatch_not_pool(tmp_path, monkeypatch, R):
    import mme_vla_suite.policies.motion_client as mc
    from mme_vla_suite.policies import policy_config as pc
    seen = {}

    class Rec:
        def __init__(self, **kw):
            seen.clear()
            seen.update(kw)
    monkeypatch.setattr(mc, "MotionEncoderClient", Rec)
    monkeypatch.setattr(pc, "_make_pool_motion_client", lambda *a, **k: pytest.fail("e2e 落入了 _make_pool_motion_client"))
    mcfg = _policy_cfg(True).motion
    (tmp_path / "motion_provenance.json").write_text(json.dumps(_e2e_prov(R)))
    pc._make_motion_client(mcfg, tmp_path, False)
    assert seen["encoder_kind"] == KIND and seen["readout"] == READOUT
    assert seen["expected_raw9_sha256"] == R.source_sha256() and seen["store_provenance"] == {"vae": {"vae_state_sha256": "v"}}
    pc._make_motion_client(mcfg, tmp_path / "nope", True)                  # stub：无 prov 也按本仓 sha 构造
    assert seen["expected_raw9_sha256"] == R.source_sha256() and seen["stub"] is True
    bad_cases = [dict(table_kind="wan_raw9_s32"), dict(raw9_source_sha256="0" * 64), dict(encoder_kind=RAW9_KIND),
                 dict(readout=RAW9_READOUT), dict(policy_encoder=None), dict(parameter_count=20252928),
                 dict(encoder={"x": 1}),
                 dict(policy_encoder={"init": "pretrained", "trainable": True, "layers": 1, "parameter_count": 1, "spec_src": "d"}),
                 dict(policy_encoder={"init": "random", "trainable": True, "layers": 8, "parameter_count": 1, "spec_src": "d"}),
                 dict(policy_encoder={"init": "random", "trainable": True, "layers": 1, "parameter_count": 1})]
    for over in bad_cases:
        (tmp_path / "motion_provenance.json").write_text(json.dumps(_e2e_prov(R, **over)))
        with pytest.raises(ValueError):
            pc._make_motion_client(mcfg, tmp_path, False)


# ── check_checkpoint ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def CC():
    return _load(CHECK_CKPT, "check_checkpoint_e2e_under_test")


def _cc_cfg(**motion_over):
    from omegaconf import OmegaConf
    m = dict(_policy_cfg(True).motion)
    m = OmegaConf.to_container(_policy_cfg(True).motion)
    m.update(frame_size=256, store_path="v1-store/datasets/4task-v2-1600ep-604f16da/motion-wan-raw9-s16")
    m.update(motion_over)
    m = {k: v for k, v in m.items() if v is not None}
    return OmegaConf.create({"motion": m})


def test_check_checkpoint_e2e_runs_registered(CC):
    for name, step in (("v2-1600ep-m8x8-modul-e2eenc-s16-b128-50k", "50000"), ("e2e-smoke20", "19")):
        run = CC.RUNS[name]
        assert run["steps"] == (step,) and run["motion_kind"] == KIND and run["readout"] == READOUT
        assert run["motion_budget"] == 160 and run["budget"] == 512 and run["config"] == "mme_vla_suite_b128_80k"
        assert run["policy_encoder"] == {"init": "random", "trainable": True, "layers": 1,
                                         "parameter_count": 20252928, "spec_src": "d3213de"}
    assert CC.RUNS["v2-1600ep-m8x8-modul-wanraw9s32-b128-50k"]["motion_kind"] == RAW9_KIND     # 既有条目不变


def test_check_checkpoint_e2e_branch_and_swaps_fail(CC, R, capsys):
    run = CC.RUNS["e2e-smoke20"]
    prov = _e2e_prov(R)
    CC.assert_motion(_cc_cfg(), run, prov)
    assert "BUDGET_CONSISTENT=PASS budget=160 motion_kind=wan_e2e" in capsys.readouterr().out
    fails = [
        (_cc_cfg(encoder_kind=RAW9_KIND), run, prov),                       # 快照 kind 换成旧的
        (_cc_cfg(readout=RAW9_READOUT), run, prov),                         # 快照 readout 换成旧的
        (_cc_cfg(), run, _e2e_prov(R, encoder_kind=RAW9_KIND)),             # prov kind 换成旧的
        (_cc_cfg(), run, _e2e_prov(R, readout=RAW9_READOUT)),               # prov readout 换成旧的
        (_cc_cfg(), {**run, "readout": RAW9_READOUT}, prov),                # 条目 readout 换成旧的
        (_cc_cfg(), {**CC.RUNS["r9-smoke20"], "motion_budget": 160}, prov),  # 条目 kind 是旧的 wan_raw9
        (_cc_cfg(tokens_per_window=9), run, prov),
        (_cc_cfg(stride=32), run, prov),
        (_cc_cfg(latents_per_window=None), run, prov),
        (_cc_cfg(source_run="x/checkpoint_epoch_72.pt#encoder"), run, prov),
        (_cc_cfg(), run, _e2e_prov(R, table_kind="wan_raw9_s32")),
        (_cc_cfg(), run, _e2e_prov(R, parameter_count=20252928)),           # 表参数量与 policy_encoder 混用
        (_cc_cfg(), run, _e2e_prov(R, policy_encoder={**prov["policy_encoder"], "parameter_count": 69808896})),
        (_cc_cfg(), run, _e2e_prov(R, policy_encoder={**prov["policy_encoder"], "trainable": False})),
    ]
    for cfg, r, pv in fails:
        with pytest.raises(AssertionError):
            CC.assert_motion(cfg, r, pv)
    # 反向：旧 raw9 条目遇到 e2e 快照也失败（raw9 分支要求 encoder_kind == wan_raw9）
    with pytest.raises(AssertionError):
        CC.assert_motion(_cc_cfg(budget=80), CC.RUNS["r9-smoke20"], prov)


def test_check_checkpoint_e2e_resolved_sha_matches_yaml(CC):
    if not E2E_YAML.is_file():
        pytest.skip("S3 的 e2e YAML 尚未合入：合入后重跑，不一致即以 --resolved-sha256 复算值改写 WANE2E_RESOLVED_SHA256")
    assert CC.resolved_sha256_of_yaml(E2E_YAML) == CC.WANE2E_RESOLVED_SHA256


def test_check_checkpoint_drop_none_leaves(CC):
    """G5b 故障：nnx.Linear(use_bias=False) 的 bias=None 叶。两侧剔除 None 后再比对；期望 None 而 ckpt 为数组 → 键集不等。"""
    import jax
    arr = lambda *s: jax.ShapeDtypeStruct(s, "float32")
    k_w, k_b, k_x = ("enc", "lin", "kernel"), ("enc", "lin", "bias"), ("other", "kernel")
    exp = {k_w: arr(4, 3), k_b: None, k_x: arr(2)}
    # 1) ckpt 无该键 → 剔除后键集相等
    e, a, nm, nc = CC.drop_none_leaves(exp, {k_w: arr(4, 3), k_x: arr(2)})
    assert set(e) == set(a) == {k_w, k_x} and (nm, nc) == (1, 0)
    # 2) ckpt 该键也为 None（真实 e2e-smoke20/19 的情形）→ 相等
    e, a, nm, nc = CC.drop_none_leaves(exp, {k_w: arr(4, 3), k_b: None, k_x: arr(2)})
    assert set(e) == set(a) and (nm, nc) == (1, 1)
    # 3) ckpt 该键为数组 → 剔除后 ckpt 多出 k_b，main 的键集断言 FAIL
    e, a, _, _ = CC.drop_none_leaves(exp, {k_w: arr(4, 3), k_b: arr(3), k_x: arr(2)})
    assert set(a) - set(e) == {k_b}
    # 4) ckpt 侧 None 而期望为数组 → 直接 FAIL
    with pytest.raises(AssertionError):
        CC.drop_none_leaves(exp, {k_w: None, k_b: None, k_x: arr(2)})
    # 5) ckpt 侧多出期望里没有的 None 叶 → FAIL
    with pytest.raises(AssertionError):
        CC.drop_none_leaves(exp, {k_w: arr(4, 3), k_x: arr(2), ("stray", "bias"): None})
    # 6) 旧 run（无 None）：原样返回同一内容
    old = {k_w: arr(4, 3), k_x: arr(2)}
    e, a, nm, nc = CC.drop_none_leaves(old, dict(old))
    assert e == old and a == old and (nm, nc) == (0, 0)


# ── compare_train_infer_obs：--acceptance ───────────────────────────────────────

@pytest.fixture(scope="module")
def TIC():
    return _load(TIC_PATH, "compare_train_infer_obs_e2e_under_test")


def _manifest():
    """四任务合成清单：BinFill 最长（c_env 1151）、k 可达 ≥120；RouteStick es=100/150 两局；VideoUnmaskSwap es 重复两局。"""
    specs = [("BinFill", 0, 2304, 1152), ("BinFill", 1, 900, 450), ("RouteStick", 0, 600, 100), ("RouteStick", 1, 600, 150),
             ("VideoRepick", 0, 800, 253), ("VideoUnmaskSwap", 0, 700, 114), ("VideoUnmaskSwap", 1, 650, 114)]
    eps = [{"global_episode_idx": g, "h5_file": f"record_dataset_{t}.h5", "raw_ep_idx": i, "num_timesteps": T,
            "exec_start_idx": es, "exec_samples": T - es, "total_sample_offset": 0} for g, (t, i, T, es) in enumerate(specs)]
    return {"raw_dir": "/nonexistent", "episodes": eps}


MCFG = {"stride": 16, "window_frames": 33, "demo_min_real_frames": 17, "budget": 160, "encoder_kind": KIND}
GOOD = "BinFill:0,RouteStick:0,VideoRepick:0,VideoUnmaskSwap:0"


def test_tic_max_steps_e2e(TIC):
    assert TIC.MAX_STEPS == 1300 and TIC.MAX_STEPS_E2E == 2000
    assert TIC.max_steps_for(MCFG) == 2000 and TIC.max_steps_for({"encoder_kind": RAW9_KIND}) == 1300
    assert TIC.max_steps_for(None) == 1300
    long_ep = _manifest()["episodes"][0]
    p13 = TIC.build_points([long_ep], 0)
    p20 = TIC.build_points([long_ep], 0, 2000)
    assert len(p20[0]["taus"]) == (1151 - 1) // 16 + 1 and p20[0]["taus"][-1] == 1152 + 16 * ((1151 - 1) // 16)
    assert p13[0]["taus"] == p20[0]["taus"]                               # 训练集最长 c_env 1151 < 1300：两口径同
    ep2 = dict(long_ep, num_timesteps=1152 + 2500)                         # 合成超长局：2000 口径比 1300 多点
    assert len(TIC.build_points([ep2], 0, 2000)[0]["taus"]) > len(TIC.build_points([ep2], 0)[0]["taus"])


def test_tic_visible_windows_matches_online(TIC):
    mem = _make_mem()
    for es in (40, 90, 678):
        mem.exec_start_idx = es
        for t in (es, es + 31, es + 32, es + 500):
            assert TIC.visible_windows(es, t, 16, 33, 17) == mem.visible_motion_frames(t)


@pytest.mark.parametrize("spec,why", [("BinFill:0,RouteStick:9,VideoRepick:0,VideoUnmaskSwap:0", "命中 0 条"),
                                      ("BinFill:0,BinFill:0,VideoRepick:0", "重复"),
                                      ("BinFill:0,VideoUnmaskSwap:0,VideoUnmaskSwap:1", "exec_start_idx"),
                                      ("BinFill-0", "不是")])
def test_tic_acceptance_resolve_fails_no_fallback(TIC, capsys, spec, why):
    with pytest.raises(SystemExit) as ei:
        TIC.resolve_episodes(_manifest(), spec, strict=True)
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "TIC_ACCEPTANCE=FAIL" in out and why in out
    picked, source = TIC.resolve_episodes(_manifest(), spec)               # 非验收模式仍回退（旧口径不变）
    assert source.startswith("manifest(")


def test_tic_acceptance_requires_explicit_episodes(TIC, capsys):
    with pytest.raises(SystemExit):
        TIC.plan_episodes(_manifest(), None, None, None, strict=True)
    assert "TIC_ACCEPTANCE=FAIL" in capsys.readouterr().out
    plan = TIC.plan_episodes(_manifest(), GOOD, None, None, strict=True)
    assert plan["source"] == "args" and len(plan["picked"]) == 4


def test_tic_acceptance_coverage(TIC):
    man = _manifest()
    picked, _ = TIC.resolve_episodes(man, GOOD, strict=True)
    cov = TIC.acceptance_coverage(man, picked, MCFG, 2000)
    assert cov["ok"], cov["reasons"]
    assert cov["tasks"] == 4 == cov["tasks_total"] and cov["binfill_tail_pad_windows"] >= 1
    # BinFill:0 es=1152：demo 71 窗（0..1120），c_env 1151 → 末决策时刻 t=2288 → exec 70 窗（u ≤ 1104），k=141（= P10 训练实测最大）
    assert cov["max_visible_k"] == 141 and cov["long_c_env"] == 1151 == cov["manifest_max_c_env"]
    assert TIC.coverage_tag(cov) == "tasks=4/4 tail_pad_windows>=1 max_visible_k=141"
    assert TIC.coverage_line(cov).startswith("TIC_ACCEPTANCE_COVERAGE=PASS tasks=4/4 tail_pad_windows>=1 max_visible_k=141")
    # 逐项失败
    p3, _ = TIC.resolve_episodes(man, "BinFill:0,RouteStick:0,VideoRepick:0", strict=True)
    assert "tasks=3/4" in " ".join(TIC.acceptance_coverage(man, p3, MCFG, 2000)["reasons"])
    p_short, _ = TIC.resolve_episodes(man, "BinFill:1,RouteStick:0,VideoRepick:0,VideoUnmaskSwap:0", strict=True)
    rs = " ".join(TIC.acceptance_coverage(man, p_short, MCFG, 2000)["reasons"])
    assert "max_visible_k" in rs and "最长局" in rs
    assert not TIC.acceptance_coverage(man, picked, MCFG, 2000, max_points=3)["ok"]
    dup = picked + [man["episodes"][6]]
    assert "exec_start_idx 重复" in " ".join(TIC.acceptance_coverage(man, dup, MCFG, 2000)["reasons"])
    nobin = [dict(e) for e in picked]
    nobin[0] = dict(nobin[0], exec_start_idx=1153 + 15)                    # es-s 恰为 33 的倍数边界外：构造无补尾窗
    c2 = TIC.acceptance_coverage(man, [e for e in picked if "BinFill" not in e["h5_file"]], MCFG, 2000)
    assert "BinFill 补尾窗局缺失" in " ".join(c2["reasons"])


def test_tic_check_acceptance_args(TIC, capsys):
    full = ["x"] + sum([[f, "v"] for f in TIC.ACCEPT_REQUIRED_FLAGS], [])
    args = types.SimpleNamespace(motion="sidecar", store_subdir="framesamp-8x8", max_points=0)
    TIC.check_acceptance_args(full, args)
    for drop in ("--norm-stats", "--episodes", "--history-yaml", "--ckpt"):
        argv = [a for i, a in enumerate(full) if not (a == drop or (i > 0 and full[i - 1] == drop))]
        with pytest.raises(SystemExit):
            TIC.check_acceptance_args(argv, args)
    with pytest.raises(SystemExit):
        TIC.check_acceptance_args(full, types.SimpleNamespace(motion="store", store_subdir="framesamp-8x8", max_points=0))
    with pytest.raises(SystemExit):
        TIC.check_acceptance_args(full, types.SimpleNamespace(motion="sidecar", store_subdir="framesamp", max_points=0))
    with pytest.raises(SystemExit):
        TIC.check_acceptance_args(full, types.SimpleNamespace(motion="sidecar", store_subdir="framesamp-8x8", max_points=2))
    assert TIC.explicit_flags(["x", "--a=1", "--b", "2", "c"]) == {"--a", "--b"}


def _write_single_h5(path: pathlib.Path, group: str, T: int, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(hashlib.sha256(f"{group}-{T}-{seed}".encode()).digest() * 4)    # dry-run 只核字节数与 sha256


def test_tic_main_acceptance_dry_run(tmp_path, TIC, monkeypatch, capsys):
    """整条入口（--acceptance --dry-run）：显式参数齐、history YAML 与 run 快照 sha 相等、覆盖 PASS → DRY_RUN complete=yes；
    清单里换成重复 es → TIC_ACCEPTANCE=FAIL 退出 1。"""
    man = _manifest()
    h5_root, map_dir = tmp_path / "raw-h5", tmp_path / "maps"
    map_dir.mkdir()
    rows = {}
    for e in man["episodes"]:
        task = e["h5_file"][len("record_dataset_"):-3]
        member = f"{task}/ep{e['raw_ep_idx']}.h5"
        _write_single_h5(h5_root / member, f"episode_{e['raw_ep_idx']}", e["num_timesteps"], e["global_episode_idx"])
        p = h5_root / member
        rows.setdefault(task, []).append({"task": task, "difficulty": "easy", "episode": e["raw_ep_idx"], "seed": 0,
                                          "role": "primary", "member": member,
                                          "h5_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "h5_bytes": p.stat().st_size,
                                          "timestep_count": e["num_timesteps"], "new_idx": e["raw_ep_idx"],
                                          "src_group": f"episode_{e['raw_ep_idx']}"})
    for task, rs in rows.items():
        (map_dir / f"record_dataset_{task}_episode_map.json").write_text(json.dumps(rs))
    lib = tmp_path / "lib"
    (lib / "meta").mkdir(parents=True)
    (lib / "meta" / "episode_manifest.json").write_text(json.dumps(man))
    hy = tmp_path / "e2e.yaml"
    hy.write_text(textwrap.dedent("""\
        budget: 512
        motion:
          enabled: true
          budget: 160
          demo_min_real_frames: 17
          stride: 16
          window_frames: 33
          encoder_kind: wan_e2e
          readout: wan_e2e_enc768_s16_v1
        """))
    run_root = tmp_path / "run"
    (run_root / "19").mkdir(parents=True)
    (run_root / "history_config.resolved.sha256").write_text(TIC.yaml_resolved_sha256(hy) + "\n")
    monkeypatch.chdir(tmp_path)
    base = ["x", "--acceptance", "--dry-run", "--lib", str(lib), "--ckpt", str(run_root / "19"), "--store-subdir", "framesamp-8x8",
            "--train-config", "mme_vla_suite_b128_80k", "--norm-stats", str(tmp_path / "ns/robomme/norm_stats.json"),
            "--h5-root", str(h5_root), "--episode-map-dir", str(map_dir), "--history-yaml", str(hy),
            "--motion", "sidecar", "--motion-subdir", "motion-wan-raw9-s16", "--motion-gpu", "1", "--out", str(tmp_path / "o.json")]
    monkeypatch.setattr(sys, "argv", base + ["--episodes", GOOD])
    assert TIC.main() == 0
    out = capsys.readouterr().out
    assert "TIC_ACCEPTANCE_COVERAGE=PASS tasks=4/4 tail_pad_windows>=1 max_visible_k=141" in out
    assert "DRY_RUN episodes=4 complete=yes" in out and "snapshot_equal=1" in out
    monkeypatch.setattr(sys, "argv", base + ["--episodes", "BinFill:0,VideoUnmaskSwap:0,VideoUnmaskSwap:1"])
    with pytest.raises(SystemExit) as ei:
        TIC.main()
    assert ei.value.code == 1 and "TIC_ACCEPTANCE=FAIL" in capsys.readouterr().out
    (run_root / "history_config.resolved.sha256").write_text("0" * 64 + "\n")      # 快照 sha 不等即 FAIL
    monkeypatch.setattr(sys, "argv", base + ["--episodes", GOOD])
    with pytest.raises(SystemExit):
        TIC.main()
    assert "TIC_ACCEPTANCE=FAIL" in capsys.readouterr().out


def test_tic_e2e_row_view_and_convert_check(TIC, RB):
    row = _latent(5).astype(ml_dtypes.bfloat16)
    v = TIC.e2e_row_view(row)
    assert v.shape == (9, 16384) and v.dtype == BF16 and v.tobytes() == row.tobytes()
    with pytest.raises(RuntimeError):
        TIC.e2e_row_view(_latent(5))

    class FakeClient:
        def __init__(self, bad=False):
            self._bf16_fn, self.bad, self.last_raw_f32 = RB.raw9_f32_to_bf16, bad, None

        def __call__(self, frames, start):
            self.last_raw_f32 = _latent(start).reshape(9, 16384)
            out = RB.raw9_f32_to_bf16(self.last_raw_f32)
            if self.bad:
                out = out.copy()
                out.view(np.uint16)[0, 0] ^= 1
            return out

        def close(self):
            pass
    ok = TIC.E2EConvertCheck(FakeClient())
    for s in range(3):
        ok(None, s)
    assert ok.checked == 3 and ok.mismatches == 0 and ok.fn_ok
    bad = TIC.E2EConvertCheck(FakeClient(bad=True))
    bad(None, 1)
    assert bad.mismatches == 1


def _perf_log(path: pathlib.Path, n_infer=5, episode=True, step=900):
    with path.open("w") as f:
        for i in range(n_infer):
            f.write(json.dumps({"event": "infer", "episode_idx": 0, "step_idx": 100 + 16 * i, "infer_ms": 100.0 + i,
                                "t_rel_s": 2.0 * (i + 1), "request_bytes": 1000 * (i + 1), "peak_bytes_in_use": 3 << 30}) + "\n")
        if episode:
            f.write(json.dumps({"event": "episode", "episode_idx": 0, "wall_s": 12.5, "add_buffer_ms": [50.0, 20.0, 21.0],
                                "n_infer": n_infer, "infer_ms": [], "last_step_idx": step, "request_bytes": 805_000_000,
                                "motion_downsample_steps": 2, "motion_downsample_max_k": 162, "peak_bytes_in_use": 5 << 30}) + "\n")


def test_tic_perf_report(tmp_path, TIC):
    logs = []
    for i in range(4):
        p = tmp_path / f"s{i}.jsonl"
        _perf_log(p, episode=(i != 3))                                    # 第 4 片服务被外部结束：无 episode 行，用 infer 行补
        logs.append(str(p))
    csv = tmp_path / "gpu.csv"
    csv.write_text("40000, 81920\n70000, 81920\n65000, 81920\n")
    ok, line, detail = TIC.perf_report(logs, str(csv), 4)
    assert ok, line
    assert line.startswith("E2E_ONLINE_PERF=PASS p95_ms=") and " shards_per_gpu=4 peak_gb=68.36 " in line
    assert "motion_downsample_steps=6" in line and "motion_downsample_max_k=162" in line
    assert len(detail["episode_wall_s"]) == 4
    ok3, line3, _ = TIC.perf_report(logs[:3], str(csv), 4)                 # 片数不足
    assert not ok3 and line3.startswith("E2E_ONLINE_PERF=FAIL")
    csv.write_text("81920, 81920\n")
    assert not TIC.perf_report(logs, str(csv), 4)[0]                      # 显存打满
    assert not TIC.perf_report(logs, None, 4)[0]                          # 无卡级证据
    ok24, line24, _ = TIC.perf_report(logs[:3], str(tmp_path / "gpu.csv"), 3)
    assert "shards_per_gpu=3" in line24


def test_tic_perf_report_cli(tmp_path, TIC, monkeypatch, capsys):
    logs = []
    for i in range(3):
        p = tmp_path / f"s{i}.jsonl"
        _perf_log(p)
        logs.append(str(p))
    (tmp_path / "gpu.csv").write_text("50000, 81920\n")
    monkeypatch.setattr(sys, "argv", ["x", "--perf-report", *logs, "--perf-gpu-csv", str(tmp_path / "gpu.csv"),
                                      "--shards-per-gpu", "3", "--out", str(tmp_path / "perf.json")])
    assert TIC.main() == 0
    assert "E2E_ONLINE_PERF=PASS" in capsys.readouterr().out
    assert json.loads((tmp_path / "perf.json").read_text())["line"].startswith("E2E_ONLINE_PERF=PASS")


# ── HF 导出副本：假 bucket 夹具 ───────────────────────────────────────────────

FAKE_HF = r'''#!/usr/bin/env bash
# 假 hf：把 hf://buckets/<id>/<path> 映射到 $FAKE_BUCKET_ROOT/<id>/<path>，每次调用记一行到 $FAKE_HF_LOG
set -euo pipefail
echo "$*" >> "$FAKE_HF_LOG"
map() { local x="$1"; if [[ "$x" == hf://buckets/* ]]; then echo "$FAKE_BUCKET_ROOT/${x#hf://buckets/}"; else echo "$x"; fi; }
case "$1" in
  auth) echo "user: HongzeFu" ;;
  __stat) find "$FAKE_BUCKET_ROOT/$2" -type f 2>/dev/null | wc -l ;;
  buckets)
    case "$2" in
      create) mkdir -p "$FAKE_BUCKET_ROOT/$3" ;;
      cp) s=$(map "$3"); d=$(map "$4"); mkdir -p "$(dirname "$d")"; cp "$s" "$d" ;;
    esac ;;
  sync) s=$(map "$2"); d=$(map "$3"); mkdir -p "$d"; cp -a "$s/." "$d/" ;;
  *) echo "fake hf: 未知子命令 $*" >&2; exit 2 ;;
esac
'''


def test_export_script_fake_bucket_deletes_verify_copies(tmp_path):
    if shutil.which("sha256sum") is None:
        pytest.skip("无 sha256sum")
    src = tmp_path / "run"
    for step, n in (("100", 3), ("200", 2)):
        for j in range(n):
            f = src / step / "params" / f"shard{j}.bin"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(os.urandom(1000 + j))
        (src / step / "_CHECKPOINT_METADATA").write_text("{}")
    (src / "assets" / "robomme").mkdir(parents=True)
    (src / "assets" / "robomme" / "norm_stats.json").write_text("{}")
    (src / "history_config.resolved.yaml").write_text("budget: 512\n")
    (src / "motion_provenance.json").write_text("{}")
    src_sums_before = subprocess.run(["bash", "-c", f"cd {src} && find . -type f | sort | xargs sha256sum"],
                                     capture_output=True, text=True, check=True).stdout
    fake = tmp_path / "hf"
    fake.write_text(FAKE_HF)
    fake.chmod(0o755)
    (tmp_path / "hf.env").write_text("HF_TOKEN=dummy-not-a-secret\n")
    export_root = tmp_path / "export"
    env = {**os.environ, "SRC": str(src), "EXPORT_ROOT": str(export_root), "E2E_EXPORT_FAKE_HF": str(fake),
           "HF_ENV_FILE": str(tmp_path / "hf.env"), "E2E_EXPORT_STEPS": "100 200", "POLL_S": "1",
           "TRAIN_LOG": str(tmp_path / "no-train.log"), "FAKE_BUCKET_ROOT": str(tmp_path / "bucket"),
           "FAKE_HF_LOG": str(tmp_path / "hf.log")}
    r = subprocess.run(["bash", str(EXPORT_SH)], env=env, capture_output=True, text=True, timeout=120)
    out = r.stdout
    assert r.returncode == 0, out + r.stderr
    verify = export_root / "verify"
    lines = out.splitlines()
    for step in ("100", "200"):
        i_pass = next(i for i, l in enumerate(lines) if l.startswith(f"STEP_VERIFY=PASS sub={step} "))
        i_del = next(i for i, l in enumerate(lines) if l.startswith(f"VERIFY_DELETED path={verify}/{step} bytes="))
        assert i_del == i_pass + 1
        want = sum(p.stat().st_size for p in (src / step).rglob("*") if p.is_file())
        assert lines[i_del].endswith(f"bytes={want}")
    i_final = next(i for i, l in enumerate(lines) if l == "HF_VERIFY=PASS ckpts=2")
    assert any(l.startswith(f"VERIFY_DELETED path={verify}/assets") for l in lines[i_final:])
    assert "ROOT_VERIFY=PASS dirs=1 files=2" in out
    assert not verify.exists() or not any(verify.iterdir())
    tsv = (export_root / "logs" / "verify_deleted.tsv").read_text().splitlines()
    assert len(tsv) >= 4 and all(len(x.split("\t")) == 3 for x in tsv)
    calls = (tmp_path / "hf.log").read_text().splitlines()
    bucket = "hf://buckets/HongzeFu/robomme-4task-modul-e2eenc-s16-50k-v1"
    downloads = [c for c in calls if c.startswith(f"sync {bucket}")]
    # 每个 step 恰下载一次（阶段 1 读回）；阶段 2 不整库 sync，不重下已删 step
    assert sum(1 for c in downloads if c.startswith(f"sync {bucket}/100 ")) == 1
    assert sum(1 for c in downloads if c.startswith(f"sync {bucket}/200 ")) == 1
    assert not any(c.split()[1] == bucket for c in downloads)
    assert sum(1 for c in calls if c.startswith(f"buckets cp {bucket}/")) == 2       # 平文件逐个读回
    # 源 checkpoint 不动（stage 是硬链接）
    assert subprocess.run(["bash", "-c", f"cd {src} && find . -type f | sort | xargs sha256sum"],
                          capture_output=True, text=True, check=True).stdout == src_sums_before
    # 续跑：已 .done 的 step 跳过，不重新下载
    r2 = subprocess.run(["bash", str(EXPORT_SH)], env=env, capture_output=True, text=True, timeout=120)
    assert r2.returncode == 0 and "step 100 已校验，跳过" in r2.stdout and "HF_VERIFY=PASS ckpts=2" in r2.stdout


def test_export_script_delete_guard_rejects_outside(tmp_path):
    """delete_verified 只许删 $VERIFY 下的路径。"""
    txt = EXPORT_SH.read_text()
    assert 'delete_verified "$VERIFY/$sub"' in txt and "retry_sync \"$BUCKET\" \"$VERIFY\"" not in txt
    fn = txt[txt.index("delete_verified() {"):txt.index("\n}\n", txt.index("delete_verified() {")) + 3]
    script = f'set -euo pipefail\nVERIFY={tmp_path}/v\nDELETED_TSV={tmp_path}/d.tsv\n{fn}\nmkdir -p {tmp_path}/other\ndelete_verified {tmp_path}/other\n'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode != 0 and "拒绝删除" in r.stdout and (tmp_path / "other").is_dir()
    r = subprocess.run(["bash", "-c", script.replace(f"delete_verified {tmp_path}/other", f"delete_verified {tmp_path}/v/../other")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and (tmp_path / "other").is_dir()
