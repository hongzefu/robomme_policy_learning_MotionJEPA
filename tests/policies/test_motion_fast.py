"""steer-speedup-1116 F3 快速数值模式的 CPU 合同测试：缺省逐字不变、快速模式握手放行 numerics、模型身份照比。"""

import copy
import hashlib
import importlib.util
import json
import sys
import types

import numpy as np
import pytest

from mme_vla_suite.policies import motion_client as C

ROOT = C._REPO_ROOT
WAN = ROOT / "scripts/dataset/wan"


def load_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── 缺省不变：复制件逐字节未改，SOURCE_PIN 仍成立 ─────────────────────────────

def test_wan_motion_infer_bytes_unchanged_vs_source_pin():
    pin = json.loads((WAN / "SOURCE_PIN.json").read_text())
    assert hashlib.sha256((WAN / "wan_motion_infer.py").read_bytes()).hexdigest() == pin["source_sha256"]


def test_sidecar_default_does_not_load_fast_module(monkeypatch):
    S = load_path(WAN / "motion_sidecar.py", "motion_sidecar_fast_test")
    monkeypatch.delenv("MMEVLA_MOTION_FAST", raising=False)
    assert S.fast_numerics(object()) is None
    monkeypatch.setenv("MMEVLA_MOTION_FAST", "0")
    assert S.fast_numerics(object()) is None


# ── wan_motion_fast：开关读回与编码路径（torch CPU，不加载真权重）────────────────

torch = pytest.importorskip("torch")


@pytest.fixture
def fast(monkeypatch):
    F = load_path(WAN / "wan_motion_fast.py", "wan_motion_fast_test")
    yield F
    F.infer_module().pin_numerics()                      # 恢复缺省开关，免得影响其他测试


def test_pin_numerics_default_unchanged_after_fast(fast):
    W = fast.infer_module()
    assert fast.pin_numerics_fast(False) == fast.expected_flags(False)
    assert W.pin_numerics() == W.EXPECTED_FLAGS           # 缺省 pin 仍按原期望断言
    assert W.EXPECTED_FLAGS["matmul_allow_tf32"] is False and W.EXPECTED_FLAGS["cudnn_benchmark"] is False


@pytest.mark.parametrize("fp32", [False, True])
def test_pin_numerics_fast_readback(fast, fp32):
    snap = fast.pin_numerics_fast(fp32)
    assert snap == fast.expected_flags(fp32)
    assert snap["cudnn_benchmark"] is True
    assert snap["matmul_allow_tf32"] is (not fp32) and snap["cudnn_allow_tf32"] is (not fp32)
    assert snap["float32_matmul_precision"] == ("highest" if fp32 else "high")


def test_env_switches(fast):
    assert fast.fast_enabled({"MMEVLA_MOTION_FAST": "1"}) and not fast.fast_enabled({"MMEVLA_MOTION_FAST": "true"})
    assert fast.vae_fp32({"MMEVLA_MOTION_FAST_VAE_FP32": "1"}) and not fast.vae_fp32({})
    info = fast.fast_info(True, {"x": 1})
    assert info["enabled"] is True and info["vae_amp"] == "off" and info["tf32"] == "off"
    assert fast.fast_info(False)["vae_amp"] == "bf16"


class FakeVAE:
    """把 (B,3,33,256,256) 平均池化成 (B,16,9,32,32)，记录是否在 autocast 中被调用。"""

    def __init__(self):
        self.autocast = []

    def encode(self, x):
        self.autocast.append(torch.is_autocast_enabled("cpu") if hasattr(torch, "is_autocast_enabled") else None)
        B = x.shape[0]
        t = torch.cat([x[:, :, :1], x[:, :, 1:]], dim=2)            # 33 帧
        g = torch.stack([t[:, :, 0]] + [t[:, :, 1 + 4 * k:5 + 4 * k].mean(2) for k in range(8)], dim=2)  # (B,3,9,256,256)
        z = torch.nn.functional.avg_pool3d(g, (1, 8, 8))                                                  # (B,3,9,32,32)
        z = torch.cat([z] * 5 + [z[:, :1]], dim=1)[:, :16]
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: z))


def windows(n, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, 33, 256, 256, 3), dtype=np.uint8)


def test_encode_chunks_fp32_equals_default_encode_chunk(fast):
    W = fast.infer_module()
    vae = FakeVAE()
    w = windows(2)
    batch = fast.encode_chunks(vae, w, "cpu", fp32_vae=True)
    assert batch.shape == (2, 9, 16, 32, 32) and batch.dtype == torch.float32
    for i in range(2):
        assert torch.equal(batch[i], W.encode_chunk(vae, w[i], "cpu"))   # 预处理与轴顺序逐位一致


def test_encode_chunks_single_window_and_bad_input(fast):
    vae = FakeVAE()
    assert fast.encode_chunks(vae, windows(1)[0], "cpu", fp32_vae=True).shape == (1, 9, 16, 32, 32)
    with pytest.raises(AssertionError):
        fast.encode_chunks(vae, windows(1).astype(np.int16), "cpu", fp32_vae=True)


def test_motion_tokens_batch_matches_single(fast):
    W = fast.infer_module()
    torch.manual_seed(0)
    lin = torch.nn.Linear(9 * 16 * 32 * 32, 768)
    enc = lambda x: lin(x.reshape(x.shape[0], -1)).reshape(x.shape[0], 1, 768)
    lat = torch.randn(3, 9, 16, 32, 32)
    out = fast.motion_tokens(enc, lat, False, "cpu")
    assert out.shape == (3, 768) and out.dtype == np.float32
    for i in range(3):
        np.testing.assert_allclose(out[i], W.motion_token(enc, lat[i], False, "cpu"), rtol=1e-5, atol=1e-5)


def test_sidecar_fast_numerics_shares_checked_module(monkeypatch):
    S = load_path(WAN / "motion_sidecar.py", "motion_sidecar_fast_test2")
    W = object()
    monkeypatch.setenv("MMEVLA_MOTION_FAST", "1")
    monkeypatch.setenv("MMEVLA_MOTION_FAST_VAE_FP32", "1")
    F, fp32 = S.fast_numerics(W)
    assert F._W is W and fp32 is True


# ── 客户端握手：fast_mode_agreed ────────────────────────────────────────────────

def test_fast_mode_agreed():
    assert C.fast_mode_agreed({}, env={}) is False
    assert C.fast_mode_agreed({"fast": {"enabled": True}}, env={"MMEVLA_MOTION_FAST": "1"}) is True
    with pytest.raises(ValueError):
        C.fast_mode_agreed({"fast": {"enabled": True}}, env={})
    with pytest.raises(ValueError):
        C.fast_mode_agreed({}, env={"MMEVLA_MOTION_FAST": "1"})
    with pytest.raises(ValueError):                                      # 只认 True，不认真值
        C.fast_mode_agreed({"fast": {"enabled": 1}}, env={"MMEVLA_MOTION_FAST": "1"})


# ── v10 / 池化 768 单进程路径 ──────────────────────────────────────────────────

FLAGS = {"matmul_allow_tf32": False, "cudnn_allow_tf32": False, "cudnn_benchmark": False}
FAST_FLAGS = {"matmul_allow_tf32": True, "cudnn_allow_tf32": True, "cudnn_benchmark": True}


def store_prov():
    vae = {k: f"v-{k}" for k in C.PROV_SAME_VAE}
    vae.update(vae_dtype="float32", batch=1, tf32="off", amp="off", flags=dict(FLAGS))
    enc = {k: f"e-{k}" for k in C.PROV_SAME_ENC}
    enc.update(amp="bf16", tf32="off", batch=1, flags=dict(FLAGS))
    return {"vae": vae, "encoder": enc}


def fast_handshake(prov):
    p = copy.deepcopy(prov)
    p["vae"].update(amp="bf16", tf32="on", batch="any", flags=dict(FAST_FLAGS))
    p["encoder"].update(tf32="on", batch="any", flags=dict(FAST_FLAGS))
    p["fast"] = {"enabled": True, "schema": "wan-motion-fast-v1"}
    return p


def v10_client(provenance, kind="v10"):
    client = object.__new__(C.MotionEncoderClient)
    client.encoder_kind = kind
    client.provenance = provenance
    client.closed = False

    def close():
        client.closed = True
    client.close = close
    return client


def test_default_mode_identical_passes_and_numerics_diff_still_rejected(monkeypatch):
    monkeypatch.delenv("MMEVLA_MOTION_FAST", raising=False)
    prov = store_prov()
    c = v10_client(copy.deepcopy(prov))
    c.check_provenance(prov)
    assert c.fast_released == [] and not c.closed
    hs = fast_handshake(prov)
    hs.pop("fast")
    c = v10_client(hs)
    with pytest.raises(RuntimeError, match="不同源"):
        c.check_provenance(prov)
    assert c.closed


def test_fast_mode_releases_numerics(monkeypatch):
    monkeypatch.setenv("MMEVLA_MOTION_FAST", "1")
    prov = store_prov()
    c = v10_client(fast_handshake(prov))
    c.check_provenance(prov)
    released = {line.split(":")[0] for line in c.fast_released}
    assert released == {"vae.amp", "vae.tf32", "vae.batch", "vae.flags", "encoder.tf32", "encoder.batch", "encoder.flags"}
    assert not c.closed


@pytest.mark.parametrize("section,key", [
    ("vae", "vae_state_sha256"), ("vae", "vae_id"), ("vae", "latent_mode"), ("vae", "module_sha256"),
    ("encoder", "checkpoint_sha256"), ("encoder", "checkpoint_epoch"), ("encoder", "arch"),
    ("encoder", "motion_dims"), ("encoder", "encoder_src_sha256"), ("vae", "torch"),
])
def test_fast_mode_still_rejects_model_identity(monkeypatch, section, key):
    monkeypatch.setenv("MMEVLA_MOTION_FAST", "1")
    prov = store_prov()
    hs = fast_handshake(prov)
    hs[section][key] = "other"
    c = v10_client(hs)
    with pytest.raises(RuntimeError, match=f"{section}.{key}"):
        c.check_provenance(prov)
    assert c.closed


def test_fast_marker_without_client_env_rejected(monkeypatch):
    monkeypatch.delenv("MMEVLA_MOTION_FAST", raising=False)
    prov = store_prov()
    c = v10_client(fast_handshake(prov))
    with pytest.raises(RuntimeError, match="快速模式两侧不一致"):
        c.check_provenance(prov)
    assert c.closed


def test_client_env_without_sidecar_marker_rejected(monkeypatch):
    monkeypatch.setenv("MMEVLA_MOTION_FAST", "1")
    prov = store_prov()
    c = v10_client(copy.deepcopy(prov))
    with pytest.raises(RuntimeError, match="快速模式两侧不一致"):
        c.check_provenance(prov)


def test_pool768_fast_releases_vae_numerics_but_checks_identity(monkeypatch):
    monkeypatch.setenv("MMEVLA_MOTION_FAST", "1")
    prov = store_prov()
    c = v10_client(fast_handshake(prov), kind=C.POOL768_KIND)
    c.check_provenance(prov)
    assert all(line.startswith("vae.") for line in c.fast_released)
    hs = fast_handshake(prov)
    hs["vae"]["vae_state_sha256"] = "0" * 64
    c = v10_client(hs, kind=C.POOL768_KIND)
    with pytest.raises(RuntimeError, match="vae_state_sha256"):
        c.check_provenance(prov)


def test_pool768_handshake_identity_unaffected_by_fast(monkeypatch):
    monkeypatch.setenv("MMEVLA_MOTION_FAST", "1")
    c = v10_client({"encoder_kind": C.POOL768_KIND, "readout": "other", "pool_source_sha256": "a" * 64,
                    "fast": {"enabled": True}}, kind=C.POOL768_KIND)
    c.readout, c.expected_pool_sha256 = "wan_raw_timepool8_adaptive6x8_v1", "a" * 64
    with pytest.raises(RuntimeError, match="readout"):
        c.check_pool_handshake()


# ── 会话分支：固定池化 1024 与阶段二 ─────────────────────────────────────────────

def _timepool_helpers():
    return load_path(ROOT / "tests/policies/test_motion_client_timepool.py", "timepool_tests_for_fast")


def test_timepool_fast_releases_numerics_keeps_identity(monkeypatch):
    T = _timepool_helpers()
    prov = T.contract.provenance_for()
    monkeypatch.setenv("MMEVLA_MOTION_FAST", "1")
    client = T.handshake_client(prov)
    client.provenance["numerics"] = {"dtype": "float32", "autocast": "bf16", "batch": 16,
                                     "flags": {"matmul_allow_tf32": True, "cudnn_allow_tf32": True}}
    client.provenance["fast"] = {"enabled": True}
    client._check_timepool_provenance(prov)
    client.provenance["pool"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pool"):
        client._check_timepool_provenance(prov)


def test_timepool_default_still_requires_fp32(monkeypatch):
    T = _timepool_helpers()
    prov = T.contract.provenance_for()
    monkeypatch.delenv("MMEVLA_MOTION_FAST", raising=False)
    client = T.handshake_client(prov)
    client.provenance["numerics"]["flags"]["matmul_allow_tf32"] = True
    with pytest.raises(ValueError, match="TF32"):
        client._check_timepool_provenance(prov)


def stage2_client(monkeypatch):
    from mme_vla_suite.datastore import motion_store as ms
    monkeypatch.setattr(ms, "validate_stage2_provenance", lambda prov: None)
    monkeypatch.setattr(C, "validate_stage2_evidence", lambda prov, root: None)
    online = {"vae_id": "w", "vae_state_sha256": "a" * 64, "vae_dtype": "float32", "latent_mode": "mode",
              "batch": 1, "tf32": "off", "amp": "off", "adapter_sha256": "b" * 64, "preprocess": {"frame_size": 256},
              "numerics": {"amp": "bf16"}}
    store = {"encoder": {"checkpoint_sha256": "c" * 64, "run_dir": "/x"}, "encoder_state_sha256": "d" * 64,
             "source_pin": {"vae_adapter_sha256": "b" * 64}, "numerics": {"tf32": False},
             "stage2_compat": {"online_vae": online}}
    client = object.__new__(C.MotionEncoderClient)
    client.encoder_kind = C.STAGE2_KIND
    client._provenance_root = None
    client._check_session_worker = lambda: None
    hs = copy.deepcopy({k: store[k] for k in ("encoder", "encoder_state_sha256", "source_pin")})
    hs["encoder"]["run_dir"] = "/elsewhere"
    hs.update(numerics={"tf32": True}, fast={"enabled": True},
              online_vae=dict(online, tf32="on", amp="bf16", batch="any", numerics={"amp": "bf16", "tf32": True}))
    client.provenance = hs
    return client, store


def test_stage2_fast_releases_numerics(monkeypatch):
    monkeypatch.setenv("MMEVLA_MOTION_FAST", "1")
    client, store = stage2_client(monkeypatch)
    client._check_stage2_provenance(store)


@pytest.mark.parametrize("mutate", [
    lambda p: p["encoder"].__setitem__("checkpoint_sha256", "0" * 64),
    lambda p: p.__setitem__("encoder_state_sha256", "0" * 64),
    lambda p: p["online_vae"].__setitem__("vae_state_sha256", "0" * 64),
    lambda p: p["online_vae"].__setitem__("adapter_sha256", "0" * 64),
])
def test_stage2_fast_still_rejects_identity(monkeypatch, mutate):
    monkeypatch.setenv("MMEVLA_MOTION_FAST", "1")
    client, store = stage2_client(monkeypatch)
    mutate(client.provenance)
    with pytest.raises(ValueError):
        client._check_stage2_provenance(store)


def test_stage2_default_still_compares_numerics(monkeypatch):
    monkeypatch.delenv("MMEVLA_MOTION_FAST", raising=False)
    client, store = stage2_client(monkeypatch)
    client.provenance.pop("fast")
    with pytest.raises(ValueError, match="numerics"):
        client._check_stage2_provenance(store)
