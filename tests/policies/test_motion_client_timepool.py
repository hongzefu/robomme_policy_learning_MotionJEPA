"""固定池化在线入口的 CPU 合同测试：sidecar 池化路径、协议、身份分派与 run 根快照恢复。"""

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from mme_vla_suite.policies import motion_client as C
from mme_vla_suite.policies import motion_protocol as P
from mme_vla_suite.shared import wan_timepool as tp

ROOT = C._REPO_ROOT
SIDECAR = ROOT / "scripts/dataset/wan_timepool/timepool_sidecar.py"
STAGE2_RUN = "stage2-full144-1025-a/checkpoint_epoch_105.pt#model"
KIND = tp.TIMEPOOL_ENCODER_KIND


def load_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load_path(SIDECAR, "timepool_sidecar_tests")
contract = load_path(ROOT / "tests/datastore/test_motion_timepool_contract.py", "timepool_contract_fixture")
stage2_store = load_path(ROOT / "tests/datastore/test_motion_store_stage2.py", "stage2_store_fixture")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def window(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (33, 256, 256, 3), dtype=np.uint8)


@pytest.fixture
def timepool_prov():
    return contract.provenance_for()


@pytest.fixture
def stub_process(monkeypatch):
    """uv 换成当前解释器；进程、脚本、socket 和封包全部走生产代码。"""
    real_popen = subprocess.Popen
    calls = []

    def spawn(argv, **kwargs):
        if argv[:3] != ["uv", "run", "--project"]:
            return real_popen(argv, **kwargs)
        calls.append((argv, kwargs["env"]))
        return real_popen([sys.executable, *argv[6:]], **kwargs)
    monkeypatch.setattr(C.subprocess, "Popen", spawn)
    return calls


# ── sidecar 池化路径 ───────────────────────────────────────────────────────────

def test_sidecar_top_level_needs_only_stdlib_and_numpy():
    code = ("import sys, importlib.util as u\n"
            "sys.modules['torch'] = None; sys.modules['jax'] = None; sys.modules['mme_vla_suite'] = None\n"
            f"s = u.spec_from_file_location('sc', {str(SIDECAR)!r}); m = u.module_from_spec(s); s.loader.exec_module(m)\n"
            "print(m.RESPONSE_BYTES)")
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "4096"


@pytest.mark.parametrize("seed", range(4))
def test_pool_shared_eq_sidecar_path_matches_offline_bits(seed):
    rng = np.random.default_rng(seed)
    scale = 10.0 ** rng.integers(-4, 5, tp.TIMEPOOL_INPUT_SHAPE)
    latent = (rng.standard_normal(tp.TIMEPOOL_INPUT_SHAPE) * scale).astype(np.float32)
    offline = tp.pool_raw_wan_latent(latent)
    token = S.token_from_latent(latent)
    assert token.shape == (1024,) and token.dtype == np.float32 and token.nbytes == 4096
    assert token.tobytes() == offline.tobytes()


def test_sidecar_pool_is_the_shared_file():
    assert S._pool.pool_source_sha256() == tp.pool_source_sha256()
    assert S._pool.POOL_CONTRACT == tp.POOL_CONTRACT


@pytest.mark.parametrize("bad", [np.zeros((8, 16, 32, 32), np.float32), np.zeros((9, 16, 32, 32), np.float64),
                                 np.full((9, 16, 32, 32), np.nan, np.float32)])
def test_token_rejects_non_contract_latent(bad):
    with pytest.raises((TypeError, ValueError)):
        S.token_from_latent(bad)


class StatefulVAE:
    """假 VAE：不清缓存时内部状态会跨窗累加，输出随调用顺序变化。"""

    def __init__(self):
        self.state = 0.0
        self.events = []

    def clear_cache(self):
        self.state = 0.0
        self.events.append("clear")

    def encode(self, x):
        import torch
        self.events.append("encode")
        assert x.shape == (1, 3, 33, 256, 256) and x.dtype == torch.float32
        base = x.mean(dim=(1, 3, 4), keepdim=True)[:, :, ::4]            # (1,1,9,1,1)
        z = base.expand(1, 16, 9, 32, 32).clone() + self.state
        self.state += 1.0
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))


@pytest.fixture
def adapter():
    """真实 encode_chunk；只替换数值开关，不改写测试进程的全局 torch 状态。"""
    real = load_path(ROOT / "scripts/dataset/wan/wan_motion_infer.py", "wan_adapter_tests")
    events = []
    return SimpleNamespace(encode_chunk=real.encode_chunk, pin_numerics=lambda: events.append("pin"), events=events)


def test_requests_do_not_leak_vae_state_and_each_is_reset(adapter):
    vae = StatefulVAE()
    a, b = window(1), window(2)
    first = S.encode_window(adapter, vae, a, "cpu")
    second = S.encode_window(adapter, vae, b, "cpu")
    third = S.encode_window(adapter, vae, a, "cpu")
    assert first.tobytes() == third.tobytes() and first.tobytes() != second.tobytes()
    assert vae.events == ["clear", "encode", "clear"] * 3
    assert adapter.events == ["pin"] * 3
    fresh = StatefulVAE()
    assert S.encode_window(adapter, fresh, b, "cpu").tobytes() == second.tobytes()


def test_fake_vae_would_expose_a_missing_reset(adapter):
    """负对照：同一假 VAE 不清缓存时输出确实不同，证明上一条测试有检测力。"""
    vae = StatefulVAE()
    a = window(1)
    first = adapter.encode_chunk(vae, a, "cpu").cpu().numpy()
    second = adapter.encode_chunk(vae, a, "cpu").cpu().numpy()
    assert first.tobytes() != second.tobytes()


def test_vae_reset_also_runs_when_encode_fails(adapter):
    vae = StatefulVAE()
    with pytest.raises(AssertionError):
        S.encode_window(adapter, vae, np.zeros((32, 256, 256, 3), np.uint8), "cpu")
    assert vae.events[-1] == "clear"


def test_vae_without_clear_cache_is_rejected(adapter):
    vae = SimpleNamespace(encode=StatefulVAE().encode)
    with pytest.raises(AttributeError):
        S.encode_window(adapter, vae, window(), "cpu")


VINFO = dict(vae_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", vae_state_sha256="e" * 64, vae_dtype="float32", latent_mode="mode",
             batch=1, tf32="off", amp="off", python="3.11", torch="2.9.0+cu128", cuda="12.8", cudnn=91002,
             cublas_pkg="x", cudnn_pkg="y", diffusers="0.39.0", hostname="h", module_sha256="1" * 64,
             encoder_src_sha256="2" * 64, flags={}, env={"NVIDIA_TF32_OVERRIDE": None}, gpu_name="A100",
             compute_cap="8.0", sm_count=1, driver="d")
FLAGS = {"matmul_allow_tf32": False, "cudnn_allow_tf32": False, "cudnn_benchmark": False}


def real_handshake():
    fields = S.real_handshake_fields(VINFO, FLAGS)
    return {**S.base_handshake(P, stub=False), **fields, "gpu_uuid": "GPU-test"}


def test_handshake_has_full_identity_and_no_encoder_fields():
    info = real_handshake()
    assert info["encoder_kind"] == KIND and info["dtype"] == "float32" and info["shape"] == [1024]
    assert info["response_bytes"] == 4096 and info["protocol_sha256"] == P.protocol_sha256()
    assert info["pool"] == {"version": tp.POOL_VERSION, "source_sha256": tp.pool_source_sha256()}
    assert info["wan"] == {"vae_id": VINFO["vae_id"], "vae_state_sha256": VINFO["vae_state_sha256"],
                           "adapter_sha256": sha((ROOT / "scripts/dataset/wan/wan_motion_infer.py").read_bytes()),
                           "dependency_lock_sha256": sha((ROOT / "scripts/dataset/wan_stage2/uv.lock").read_bytes())}
    assert info["numerics"] == {"dtype": "float32", "autocast": False, "batch": 1, "flags": FLAGS}
    assert info["environment"]["torch"] == "2.9.0+cu128" and info["environment"]["numpy"] == np.__version__
    C._reject_encoder_fields(info)
    text = json.dumps({k: v for k, v in info.items() if k != "encoder_kind"})
    assert "encoder" not in text and "ckpt" not in text and "checkpoint" not in text


@pytest.mark.parametrize("field", ["encoder", "encoder_ckpt", "encoder_state_sha256", "checkpoint_sha256", "ckpt"])
def test_encoder_field_detector_catches_nested_fields(field):
    with pytest.raises(ValueError, match="encoder"):
        C._reject_encoder_fields({"wan": {"nested": [{field: "x"}]}})
    C._reject_encoder_fields({"encoder_kind": KIND})


# ── 协议与进程：stub 档 ────────────────────────────────────────────────────────

def stub_client(tmp_path, explicit_python=False, **kwargs):
    return C.MotionEncoderClient(online_gpu="", encoder_kind=KIND, stub=True, v1_store=tmp_path, handshake_timeout=20,
                                 stage2_python=sys.executable if explicit_python else None, **kwargs)


@pytest.mark.parametrize("explicit_python", [False, True])
def test_stub_three_complete_windows_and_cleanup(stub_process, tmp_path, explicit_python):
    client = stub_client(tmp_path, explicit_python)
    try:
        C._reject_encoder_fields(client.provenance)
        assert client.provenance["stub"] is True and client.provenance["shape"] == [1024]
        assert client.worker_pid == client.provenance["pid"]
        for start, real in ((0, 33), (32, 33), (64, 17)):
            ids = list(range(start, start + real)) + [start + real - 1] * (33 - real)
            token = client(np.stack([P.stub_frame(i) for i in ids]), start)
            assert token.shape == (1024,) and token.dtype == np.float32 and token.nbytes == 4096
            np.testing.assert_array_equal(token, np.full(1024, start, np.float32))
        assert client.n_calls == 3 and client.alive
    finally:
        client.close()
    assert not client.alive and client._sock is None
    assert len(stub_process) == (0 if explicit_python else 1)


def test_stub_launch_uses_locked_project_and_sidecar_path(stub_process, tmp_path):
    client = stub_client(tmp_path)
    try:
        argv, env = stub_process[0]
        assert Path(argv[3]).name == "wan_stage2" and "--no-sync" in argv
        assert argv[6] == str(SIDECAR) and "--stub" in argv
        assert env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path.resolve() / "venvs/wan-stage2")
        assert env["HF_HUB_OFFLINE"] == "1" and "PYTHONPATH" not in env
    finally:
        client.close()


def test_worker_script_identity_is_enforced(stub_process, tmp_path):
    client = stub_client(tmp_path)
    original = client.argv[6]
    try:
        client.argv[6] = str(Path(original).with_name("unrelated.py"))
        with pytest.raises(ValueError, match="脚本身份"):
            client.worker_pid
    finally:
        client.argv[6] = original
        client.close()


def test_protocol_error_from_worker_closes_process(stub_process, tmp_path):
    client = stub_client(tmp_path)
    with pytest.raises(P.ProtocolError, match="stub"):
        client(np.zeros((33, 256, 256, 3), dtype=np.uint8), start_frame=17)
    assert client._sock is None and not client.alive


@pytest.mark.parametrize("frames", [np.zeros((32, 256, 256, 3), np.uint8), np.zeros((33, 256, 256, 3), np.float32)])
def test_wrong_input_rejected_before_send(stub_process, tmp_path, frames):
    client = stub_client(tmp_path)
    try:
        with pytest.raises(ValueError, match="33,256,256,3"):
            client(frames, 0)
        assert client.alive
    finally:
        client.close()


def scripted_client(reply):
    """真实 __call__ 接一个按脚本回包的对端；对端先读完请求再发 reply。"""
    client = object.__new__(C.MotionEncoderClient)
    near, far = socket.socketpair()
    client.encoder_kind = KIND
    client._lock = threading.Lock()
    client._sock = near
    client._proc = SimpleNamespace(pid=1, poll=lambda: None, wait=lambda timeout=None: 0)
    client.n_calls, client.total_s = 0, 0.0

    def peer():
        request = P.recv_request(far, time.monotonic() + 10)
        assert request is not None and len(request[1]) == P.PAYLOAD_BYTES
        far.sendall(reply)
        far.shutdown(socket.SHUT_WR)
    thread = threading.Thread(target=peer, daemon=True)
    thread.start()
    return client, far, thread


@pytest.mark.parametrize("reply,match", [
    (struct.pack("<II", 0, 3072) + b"x" * 3072, "length"),
    (struct.pack("<II", 0, 4097), "length"),
    (struct.pack("<I", 0) + b"x" * 3072, "length"),
    (struct.pack("<II", 0, 4096) + b"x" * 4095, "EOF"),
    (struct.pack("<I", 7), "status"),
])
def test_response_length_and_status_negative(reply, match):
    client, far, thread = scripted_client(reply)
    try:
        with pytest.raises(P.ProtocolError, match=match):
            client(window(), 0)
        assert client._sock is None, "协议错误后必须关闭连接"
    finally:
        thread.join(timeout=5)
        far.close()


def test_valid_response_returns_exact_bits():
    latent = np.random.default_rng(3).standard_normal(tp.TIMEPOOL_INPUT_SHAPE).astype(np.float32)
    token = S.token_from_latent(latent)
    client, far, thread = scripted_client(P.encode_response(token.tobytes(), response_bytes=4096))
    try:
        assert client(window(), 0).tobytes() == tp.pool_raw_wan_latent(latent).tobytes()
    finally:
        client._sock.close()
        client._sock = None
        thread.join(timeout=5)
        far.close()


def serve_in_thread(encode):
    near, far = socket.socketpair()
    result = {}

    def run():
        result["rc"] = S.serve(far, P, encode)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return near, far, thread, result


def test_serve_round_trip_through_real_protocol_matches_offline_pool(adapter):
    vae = StatefulVAE()
    near, far, thread, result = serve_in_thread(lambda start, frames: S.encode_window(adapter, vae, frames, "cpu"))
    try:
        frames = window(5)
        offline = tp.pool_raw_wan_latent(adapter.encode_chunk(StatefulVAE(), frames, "cpu").cpu().numpy())
        for _ in range(2):
            P.send_all(near, P.encode_request(0, frames.tobytes()))
            assert P.recv_response(near, time.monotonic() + 30, response_bytes=4096) == offline.tobytes()
        P.send_all(near, P.encode_shutdown())
        thread.join(timeout=10)
        assert result["rc"] == 0
    finally:
        near.close()
        far.close()


def test_serve_reports_encode_failure_and_exits_nonzero():
    def encode(start, frames):
        raise RuntimeError("预期编码失败")
    near, far, thread, result = serve_in_thread(encode)
    try:
        P.send_all(near, P.encode_request(0, window().tobytes()))
        with pytest.raises(P.ProtocolError, match="预期编码失败"):
            P.recv_response(near, time.monotonic() + 10, response_bytes=4096)
        thread.join(timeout=10)
        assert result["rc"] == 3
    finally:
        near.close()
        far.close()


def test_serve_rejects_non_contract_token():
    near, far, thread, result = serve_in_thread(lambda start, frames: np.zeros(768, np.float32))
    try:
        P.send_all(near, P.encode_request(0, window().tobytes()))
        with pytest.raises(P.ProtocolError, match="1024"):
            P.recv_response(near, time.monotonic() + 10, response_bytes=4096)
        thread.join(timeout=10)
        assert result["rc"] == 3
    finally:
        near.close()
        far.close()


def test_real_mode_without_vae_sha_reports_error_handshake():
    near, far = socket.socketpair()
    process = subprocess.Popen([sys.executable, str(SIDECAR), "--fd", str(far.fileno())], pass_fds=[far.fileno()])
    far.close()
    try:
        info = P.recv_handshake(near, time.monotonic() + 30)
        assert "VAE" in info["error"]
        assert process.wait(timeout=10) == 2
    finally:
        near.close()
        if process.poll() is None:
            process.kill()


# ── 身份分派 ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("field,value", [("encoder_kind", C.STAGE2_KIND), ("dtype", "float16"), ("shape", [768]),
                                        ("response_bytes", 3072), ("stub", False),
                                        ("pool", {"version": tp.POOL_VERSION, "source_sha256": "0" * 64}),
                                        ("encoder_ckpt", "/x.pt"),
                                        ("encoder", {"checkpoint_sha256": "a" * 64})])
def test_mixed_handshake_rejected_and_process_cleaned(stub_process, monkeypatch, tmp_path, field, value):
    original = P.recv_handshake
    seen = {}

    def receive(*args):
        result = original(*args)
        seen["pid"] = result["pid"]
        result[field] = value
        return result
    monkeypatch.setattr(P, "recv_handshake", receive)
    with pytest.raises((RuntimeError, ValueError), match="握手"):
        stub_client(tmp_path)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            C._live_process(seen["pid"])
        except ValueError:
            return
        time.sleep(0.01)
    pytest.fail("身份失败后 worker 未被清理")


def handshake_client(timepool_prov):
    client = object.__new__(C.MotionEncoderClient)
    client.encoder_kind = KIND
    client._online_gpu = "0"
    wan = timepool_prov["wan"]
    client.provenance = {
        **S.base_handshake(P, stub=False), "gpu_uuid": "GPU-test", "cuda_visible_devices": "0", "pid": 123,
        "pool": {"version": timepool_prov["pool"]["version"], "source_sha256": timepool_prov["pool"]["source_sha256"]},
        "wan": {key: wan[key] for key in C.TIMEPOOL_WAN_KEYS},
        "numerics": {"dtype": "float32", "autocast": False, "batch": 1, "flags": dict(FLAGS)},
        "environment": {"torch": "2.9.0+cu128"}}
    client._proc = SimpleNamespace(pid=123, poll=lambda: 0)

    def identity():
        if client.provenance.get("pid") != 123:
            raise ValueError("worker 进程身份不符")
        return 123, 0
    client._stage2_worker_identity = identity
    return client


def test_complete_handshake_passes_check_provenance(timepool_prov):
    client = handshake_client(timepool_prov)
    client._close_stage2 = lambda: pytest.fail("完全一致的握手不应关闭进程")
    client.check_provenance(timepool_prov)
    client._close_stage2 = lambda: None


@pytest.mark.parametrize("path,value", [
    (("pool", "source_sha256"), "0" * 64), (("pool", "version"), "other"),
    (("wan", "vae_state_sha256"), "0" * 64), (("wan", "adapter_sha256"), "0" * 64),
    (("wan", "dependency_lock_sha256"), "b" * 64), (("wan", "vae_id"), "other/repo"),
    (("wan", "extra"), "x"),
    (("numerics", "dtype"), "bfloat16"), (("numerics", "autocast"), True), (("numerics", "batch"), 8),
    (("numerics", "flags", "matmul_allow_tf32"), True), (("numerics", "flags", "cudnn_allow_tf32"), True),
    (("environment",), {}), (("gpu_uuid",), ""), (("cuda_visible_devices",), "4"), (("pid",), 999),
    (("encoder",), {"checkpoint_sha256": "a" * 64}),
])
def test_handshake_identity_mismatch_rejected(timepool_prov, path, value):
    client = handshake_client(timepool_prov)
    target = client.provenance
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        client._check_timepool_provenance(timepool_prov)


def test_check_provenance_rejects_stage2_record_and_closes(timepool_prov):
    client = handshake_client(timepool_prov)
    client._sock = None
    closed = []
    client._close_stage2 = lambda: closed.append(True)
    with pytest.raises(ValueError):
        client.check_provenance(stage2_store.stage2_identity())
    assert closed == [True]


def test_timepool_kind_rejects_stage2_provenance_before_spawn(monkeypatch):
    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: pytest.fail("身份混用不应启动子进程"))
    with pytest.raises(ValueError):
        C.MotionEncoderClient(online_gpu=0, encoder_kind=KIND, store_provenance=stage2_store.stage2_identity())


def test_stage2_kind_rejects_timepool_provenance_before_spawn(timepool_prov, tmp_path, monkeypatch):
    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: pytest.fail("身份混用不应启动子进程"))
    with pytest.raises((ValueError, KeyError)):
        C.MotionEncoderClient(online_gpu=0, encoder_kind=C.STAGE2_KIND, source_run=STAGE2_RUN,
                              store_provenance=timepool_prov, provenance_root=tmp_path,
                              expected_ckpt_sha256="a" * 64)


def test_timepool_requires_verified_provenance_before_spawn(monkeypatch):
    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: pytest.fail("来源不全不应启动子进程"))
    with pytest.raises(ValueError, match="store_provenance"):
        C.MotionEncoderClient(online_gpu=0, encoder_kind=KIND)
    with pytest.raises(ValueError):
        C.MotionEncoderClient(online_gpu=0, encoder_kind=KIND, store_provenance=contract.provenance_for(status="packed"))


@pytest.mark.parametrize("kwargs", [dict(source_run=STAGE2_RUN), dict(encoder_run_dir="/x"),
                                    dict(expected_ckpt_sha256="a" * 64), dict(eval_target_contract="/x.json")])
def test_timepool_rejects_any_encoder_or_stage2_eval_option(monkeypatch, timepool_prov, kwargs):
    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: pytest.fail("不应启动子进程"))
    with pytest.raises(ValueError):
        C.MotionEncoderClient(online_gpu=0, encoder_kind=KIND, store_provenance=timepool_prov, **kwargs)


def test_stage2_python_still_rejected_for_v10():
    with pytest.raises(ValueError, match="阶段二评估选项"):
        C.MotionEncoderClient(online_gpu=0, stage2_python="/x/bin/python")


def test_real_launch_argv_has_vae_identity_but_no_encoder(timepool_prov, tmp_path, monkeypatch):
    handshake = handshake_client(timepool_prov).provenance
    handshake["protocol_sha256"] = P.protocol_sha256()
    handshake["pool"] = {"version": tp.POOL_VERSION, "source_sha256": tp.pool_source_sha256()}
    captured = {}

    def spawn(argv, **kwargs):
        captured.update(argv=argv, env=kwargs["env"], session=kwargs["start_new_session"])
        return SimpleNamespace(pid=123, poll=lambda: 0)
    monkeypatch.setattr(C.subprocess, "Popen", spawn)
    monkeypatch.setattr(P, "recv_handshake", lambda *args: handshake)
    monkeypatch.setattr(C.MotionEncoderClient, "_stage2_worker_identity", lambda self: (123, 0))
    monkeypatch.setattr(C.MotionEncoderClient, "_capture_stage2_session", lambda self: None)
    client = C.MotionEncoderClient(online_gpu=0, encoder_kind=KIND, store_provenance=timepool_prov, v1_store=tmp_path)
    try:
        argv = captured["argv"]
        assert argv[argv.index("--expected-vae-sha256") + 1] == timepool_prov["wan"]["vae_state_sha256"]
        assert argv[6] == str(SIDECAR) and captured["session"] is True
        assert not any(flag in argv for flag in ("--ckpt", "--expected-sha256", "--stub"))
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0" and "PYTHONPATH" not in captured["env"]
    finally:
        client.close()


# ── run 根快照与 create_trained_policy ─────────────────────────────────────────

YAML = {"motion": {"enabled": True, "encoder_kind": KIND, "readout": tp.TIMEPOOL_READOUT, "online_gpu": 1,
                   "store_path": "v1-store/datasets/x/motion-wan-timepool1024"}}


def write_snapshot(root, *, resolved=None, ready_schema="wan-timepool-motion-snapshot-v1", mutate=None):
    """按主会话 write_timepool_motion_snapshot 的合同写一份完整快照。"""
    from omegaconf import OmegaConf
    root.mkdir(parents=True, exist_ok=True)
    raw = OmegaConf.to_yaml(OmegaConf.create(resolved or YAML)).encode()
    digest = sha(raw)
    files = {"history_config.txt": b"x.yaml", "history_config.resolved.yaml": raw,
             "history_config.resolved.sha256": digest.encode() + b"\n"}
    window_map = canonical({"schema": "wan-timepool-window-map-v1", "windows": []})
    files["timepool_window_map.json"] = window_map
    timepool = contract.provenance_for(map_sha=sha(window_map))
    for role in ("DATA_LINEAGE", "WAN_RESTORE_EQ", "WINDOW_EQ", "POOL_CONTRACT", "POOL_TABLE_EQ"):
        blob = canonical({"role": role, "status": "PASS"})
        files[f"motion_evidence/{role}.json"] = blob
        timepool["validation"][role]["sha256"] = sha(blob)
    arrays = b"npz-bytes"
    files["online_restore_inputs.npz"] = arrays
    files["online_restore_inputs.json"] = canonical({"schema": "stage2-online-restore-inputs-v1", "payload": {
        "path": "online_restore_inputs.npz", "sha256": sha(arrays)}})
    prov = {"history_config_name": "x.yaml", "motion_enabled": True, "resolved_sha256": digest,
            "framesamp_root": "/f", "framesamp_manifest_sha256": "1" * 64, "framesamp_store_meta_sha256": "2" * 64,
            "motion_root": "/m", "motion_manifest_sha256": timepool["manifests"]["episode_manifest_sha256"],
            "motion_index_sha256": timepool["output"]["motion_index_sha256"], "motion_store_meta_sha256": "3" * 64,
            "motion_table_sha256": timepool["output"]["motion_table_sha256"], "vae": None, "encoder": None,
            "timepool": timepool}
    if mutate is not None:
        mutate(prov, files)
    files["motion_provenance.json"] = canonical(prov) + b"\n"
    for relative, blob in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    ready = {"schema": ready_schema, "files": {relative: sha(blob) for relative, blob in files.items()}}
    (root / "motion_snapshot.ready.json").write_bytes(canonical(ready) + b"\n")
    return prov


def test_snapshot_verifies_and_returns_provenance(tmp_path):
    prov = write_snapshot(tmp_path)
    assert C.verify_timepool_snapshot(tmp_path) == prov
    assert not (tmp_path / "encoder_dependencies.json").exists()


SNAPSHOT_FILES = ["history_config.txt", "history_config.resolved.yaml", "history_config.resolved.sha256",
                  "motion_provenance.json", "timepool_window_map.json", "online_restore_inputs.json",
                  "online_restore_inputs.npz", "motion_evidence/DATA_LINEAGE.json", "motion_evidence/WAN_RESTORE_EQ.json",
                  "motion_evidence/WINDOW_EQ.json", "motion_evidence/POOL_CONTRACT.json",
                  "motion_evidence/POOL_TABLE_EQ.json"]


@pytest.mark.parametrize("relative", SNAPSHOT_FILES)
def test_snapshot_each_file_sha_is_checked(tmp_path, relative):
    write_snapshot(tmp_path)
    path = tmp_path / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        C.verify_timepool_snapshot(tmp_path)


@pytest.mark.parametrize("relative", SNAPSHOT_FILES)
def test_snapshot_each_file_must_exist(tmp_path, relative):
    write_snapshot(tmp_path)
    (tmp_path / relative).unlink()
    with pytest.raises(ValueError, match="SHA"):
        C.verify_timepool_snapshot(tmp_path)


def rewrite_ready(root, change):
    path = root / "motion_snapshot.ready.json"
    ready = json.loads(path.read_text())
    change(ready)
    path.write_bytes(canonical(ready))


def test_snapshot_ready_missing_or_wrong_schema(tmp_path):
    with pytest.raises(ValueError, match="ready"):
        C.verify_timepool_snapshot(tmp_path)
    write_snapshot(tmp_path, ready_schema="stage2-motion-snapshot-v1")
    with pytest.raises(ValueError, match="schema"):
        C.verify_timepool_snapshot(tmp_path)


@pytest.mark.parametrize("change", [
    lambda ready: ready["files"].pop("motion_evidence/WINDOW_EQ.json"),
    lambda ready: ready["files"].update({"encoder_dependencies.json": "a" * 64}),
])
def test_snapshot_file_set_must_be_exact(tmp_path, change):
    write_snapshot(tmp_path)
    rewrite_ready(tmp_path, change)
    with pytest.raises(ValueError, match="清单"):
        C.verify_timepool_snapshot(tmp_path)


def _set_vae(prov, files):
    prov["vae"] = {"vae_state_sha256": "a" * 64}


def _set_encoder(prov, files):
    prov["encoder"] = {"checkpoint_sha256": "a" * 64}


def _drop_key(prov, files):
    del prov["motion_store_meta_sha256"]


def _add_stage2_group(prov, files):
    prov["stage2_compat"] = {}


def _disable(prov, files):
    prov["motion_enabled"] = False


def _table(prov, files):
    prov["motion_table_sha256"] = "9" * 64


def _index(prov, files):
    prov["motion_index_sha256"] = "9" * 64


def _manifest(prov, files):
    prov["motion_manifest_sha256"] = "9" * 64


def _resolved(prov, files):
    prov["resolved_sha256"] = "9" * 64


def _window_map(prov, files):
    prov["timepool"]["window_map"]["sha256"] = "9" * 64


def _evidence(prov, files):
    prov["timepool"]["validation"]["WINDOW_EQ"]["sha256"] = "9" * 64


def _packed(prov, files):
    prov["timepool"]["validation"] = None


def _bad_timepool(prov, files):
    prov["timepool"]["encoder_ckpt"] = "/stage2.pt"


def _online_schema(prov, files):
    files["online_restore_inputs.json"] = canonical({"schema": "other", "payload": {
        "path": "online_restore_inputs.npz", "sha256": sha(files["online_restore_inputs.npz"])}})


def _online_payload(prov, files):
    files["online_restore_inputs.json"] = canonical({"schema": "stage2-online-restore-inputs-v1", "payload": {
        "path": "online_restore_inputs.npz", "sha256": "9" * 64}})


@pytest.mark.parametrize("mutate", [_set_vae, _set_encoder, _drop_key, _add_stage2_group, _disable, _table, _index, _manifest,
                                    _resolved, _window_map, _evidence, _packed, _bad_timepool, _online_schema,
                                    _online_payload])
def test_snapshot_identity_binding_even_with_consistent_hashes(tmp_path, mutate):
    write_snapshot(tmp_path, mutate=mutate)
    with pytest.raises(ValueError):
        C.verify_timepool_snapshot(tmp_path)


@dataclasses.dataclass
class ModelConfig:
    history_config: object
    use_history: bool = True

    def load(self, params, **kwargs):
        return object()


@dataclasses.dataclass
class TrainConfig:
    model: object
    data: object
    assets_dirs: tuple = ()
    policy_metadata: object = None


@pytest.fixture
def policy_env(monkeypatch):
    from mme_vla_suite.policies import policy_config as PC
    captured = {}
    transforms = SimpleNamespace(inputs=[], outputs=[])
    data = SimpleNamespace(create=lambda *a: SimpleNamespace(
        asset_id="fixture", data_transforms=transforms, model_transforms=transforms, use_quantile_norm=False))
    monkeypatch.setattr(PC._model, "restore_params", lambda *a, **k: {})
    monkeypatch.setattr(PC, "_params_have_motion", lambda p: True)
    monkeypatch.setattr(PC, "_assert_param_tree_exact", lambda *a: None)
    monkeypatch.setattr(PC._policy, "MME_VLA_Policy", lambda *a, **k: k)

    def client(**kwargs):
        captured["client"] = kwargs
        return "client"
    monkeypatch.setattr(C, "MotionEncoderClient", client)
    return SimpleNamespace(PC=PC, captured=captured, config=TrainConfig(model=ModelConfig("x"), data=data))


def test_policy_restores_timepool_without_any_encoder_dependency(tmp_path, policy_env):
    prov = write_snapshot(tmp_path)
    result = policy_env.PC.create_trained_policy(policy_env.config, tmp_path / "19", norm_stats={},
                                                 motion_online_gpu=3, stage2_python="/venv/bin/python")
    kwargs = policy_env.captured["client"]
    assert result["motion_enc_fn"] == "client"
    assert result["metadata"]["model_kind"] == KIND
    assert kwargs == dict(online_gpu=3, stub=False, encoder_kind=KIND, store_provenance=prov["timepool"],
                          stage2_python="/venv/bin/python", eval_target_contract=None)


def test_policy_snapshot_sha_failure_blocks_client(tmp_path, policy_env):
    write_snapshot(tmp_path)
    path = tmp_path / "motion_evidence/POOL_TABLE_EQ.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        policy_env.PC.create_trained_policy(policy_env.config, tmp_path / "19", norm_stats={})
    assert "client" not in policy_env.captured


def test_policy_stub_skips_real_provenance_but_keeps_identity(tmp_path, policy_env):
    write_snapshot(tmp_path)
    (tmp_path / "motion_snapshot.ready.json").unlink()
    result = policy_env.PC.create_trained_policy(policy_env.config, tmp_path / "19", norm_stats={}, motion_stub=True)
    assert result["motion_enc_fn"] == "client"
    assert policy_env.captured["client"]["stub"] is True and policy_env.captured["client"]["store_provenance"] is None


STAGE2_MOTION = {"enabled": True, "encoder_kind": C.STAGE2_KIND, "readout": "stage2_pool4_concat_v1",
                 "source_run": STAGE2_RUN, "online_gpu": 1}


@pytest.mark.parametrize("stub", [False, True])
@pytest.mark.parametrize("motion,message", [
    (STAGE2_MOTION, "混用"),
    ({**YAML["motion"], "readout": "stage2_pool4_concat_v1"}, "混用"),
    ({**YAML["motion"], "source_run": STAGE2_RUN}, "source_run"),
    ({**YAML["motion"], "checkpoint_sha256": "a" * 64}, "source_run"),
    ({**STAGE2_MOTION, "readout": tp.TIMEPOOL_READOUT}, "混用"),
])
def test_policy_rejects_config_and_timepool_snapshot_mix(tmp_path, policy_env, motion, message, stub):
    write_snapshot(tmp_path, resolved={"motion": motion})
    with pytest.raises(ValueError, match=message):
        policy_env.PC.create_trained_policy(policy_env.config, tmp_path / "19", norm_stats={}, motion_stub=stub)
    assert "client" not in policy_env.captured


@pytest.mark.parametrize("stub", [False, True])
def test_policy_rejects_timepool_config_with_stage2_snapshot(tmp_path, policy_env, stub):
    def to_stage2(prov, files):
        del prov["timepool"]
        prov.update(stage2_store.stage2_identity())
    write_snapshot(tmp_path, mutate=to_stage2)
    with pytest.raises(ValueError, match="混用"):
        policy_env.PC.create_trained_policy(policy_env.config, tmp_path / "19", norm_stats={}, motion_stub=stub)
    assert "client" not in policy_env.captured


def test_policy_stage2_branch_kwargs_unchanged(tmp_path, policy_env):
    """旧分支照常把完整来源、source_run、run 根和 checkpoint 摘要交给客户端。"""
    from omegaconf import OmegaConf
    raw = OmegaConf.to_yaml(OmegaConf.create({"motion": STAGE2_MOTION})).encode()
    (tmp_path / "history_config.resolved.yaml").write_bytes(raw)
    (tmp_path / "history_config.resolved.sha256").write_text(sha(raw))
    prov = {"motion_enabled": True, "resolved_sha256": sha(raw), "vae": {"vae_state_sha256": "e" * 64},
            "encoder": {"checkpoint_sha256": "a" * 64}}
    (tmp_path / "motion_provenance.json").write_text(json.dumps(prov))
    result = policy_env.PC.create_trained_policy(policy_env.config, tmp_path / "19", norm_stats={})
    assert policy_env.captured["client"] == dict(
        source_run=STAGE2_RUN, online_gpu=1, stub=False, store_provenance=prov, encoder_kind=C.STAGE2_KIND,
        provenance_root=tmp_path, stage2_python=None, eval_target_contract=None, expected_ckpt_sha256="a" * 64)
    assert result["metadata"]["model_kind"] == C.STAGE2_KIND


# ── 显式指定已安装环境时的跨环境验收（不装东西、不用 GPU）─────────────────────────

def stage2_interpreter():
    selected = os.environ.get("MMEVLA_STAGE2_IMPORT_ENV")
    if not selected:
        pytest.skip("跨环境验收需 MMEVLA_STAGE2_IMPORT_ENV 指向已安装的 wan-stage2 环境")
    return Path(selected) / "bin/python"


def cpu_env():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    return env


def test_actual_environment_stub_ipc(tmp_path):
    client = C.MotionEncoderClient(online_gpu="", encoder_kind=KIND, stub=True, v1_store=tmp_path, handshake_timeout=30,
                                   stage2_python=str(stage2_interpreter()))
    try:
        token = client(np.stack([P.stub_frame(i) for i in range(32, 65)]), 32)
        np.testing.assert_array_equal(token, np.full(1024, 32, np.float32))
    finally:
        client.close()


def test_actual_environment_pool_bits_match_main_environment(tmp_path):
    """两个环境的 NumPy 版本不同；同一 raw latent 经 sidecar 路径与主环境离线池化仍逐位相等。"""
    python = stage2_interpreter()
    rng = np.random.default_rng(11)
    latent = (rng.standard_normal(tp.TIMEPOOL_INPUT_SHAPE) * 10.0 ** rng.integers(-4, 5, tp.TIMEPOOL_INPUT_SHAPE)).astype(np.float32)
    source, target = tmp_path / "z.npy", tmp_path / "t.bin"
    np.save(source, latent)
    code = ("import sys, importlib.util as u, numpy as np\n"
            "s = u.spec_from_file_location('sc', sys.argv[1]); m = u.module_from_spec(s); s.loader.exec_module(m)\n"
            "open(sys.argv[3], 'wb').write(m.token_from_latent(np.load(sys.argv[2])).tobytes()); print(np.__version__)")
    result = subprocess.run([str(python), "-I", "-c", code, str(SIDECAR), str(source), str(target)],
                            env=cpu_env(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() != np.__version__, "本测试应跨两个 NumPy 版本"
    assert target.read_bytes() == tp.pool_raw_wan_latent(latent).tobytes()


def test_actual_diffusers_vae_requests_are_independent(tmp_path):
    """锁定环境的真实 AutoencoderKLWan（小通道随机权重，CPU）：A/B/A 顺序下第三次与第一次逐位相等。"""
    python = stage2_interpreter()
    code = """
import importlib.util as u, sys
import numpy as np, torch
from diffusers import AutoencoderKLWan
s = u.spec_from_file_location('sc', sys.argv[1]); m = u.module_from_spec(s); s.loader.exec_module(m)
s = u.spec_from_file_location('ad', sys.argv[2]); a = u.module_from_spec(s); s.loader.exec_module(a)
torch.manual_seed(0)
torch.set_num_threads(4)
vae = AutoencoderKLWan(base_dim=8, num_res_blocks=1).eval()
for p in vae.parameters():
    p.requires_grad_(False)
rng = np.random.default_rng(0)
A, B = (rng.integers(0, 256, (33, 256, 256, 3), dtype=np.uint8) for _ in range(2))
one = m.encode_window(a, vae, A, 'cpu')
two = m.encode_window(a, vae, B, 'cpu')
three = m.encode_window(a, vae, A, 'cpu')
assert one.tobytes() == three.tobytes() and one.tobytes() != two.tobytes()
assert all(item is None for item in vae._enc_feat_map)
print('REAL_VAE_INDEPENDENT=PASS')
"""
    result = subprocess.run([str(python), "-I", "-c", code, str(SIDECAR), str(ROOT / "scripts/dataset/wan/wan_motion_infer.py")],
                            env=cpu_env(), capture_output=True, text=True, timeout=280)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REAL_VAE_INDEPENDENT=PASS" in result.stdout
