"""阶段二实际 socket、来源证据和便携 encoder 注册件的 CPU 合同测试。"""

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from mme_vla_suite.policies import motion_client as C
from mme_vla_suite.policies import motion_protocol as P


SOURCE_RUN = "stage2-full144-1025-a/checkpoint_epoch_105.pt#model"


def test_datastore_dtype_dependency_is_directly_pinned_in_stage2_lock():
    project = C._REPO_ROOT / "scripts/dataset/wan_stage2"
    config = tomllib.loads((project / "pyproject.toml").read_text())
    lock = tomllib.loads((project / "uv.lock").read_text())
    assert "ml-dtypes==0.4.1" in config["project"]["dependencies"]
    packages = {package["name"]: package for package in lock["package"]}
    assert packages["ml-dtypes"]["version"] == "0.4.1"
    assert {"name": "ml-dtypes"} in packages["mme-vla-wan-stage2"]["dependencies"]
    assert packages["ml-dtypes"]["wheels"]
    pin = json.loads((project / "SOURCE_PIN.json").read_text())
    assert pin["dependency_lock_sha256"] == hashlib.sha256((project / "uv.lock").read_bytes()).hexdigest()


def test_datastore_import_in_actual_stage2_environment():
    """显式选择已由主会话安装的独立环境；禁止主环境包或用户路径补足依赖。"""
    selected = os.environ.get("MMEVLA_STAGE2_IMPORT_ENV")
    if not selected:
        pytest.skip("独立环境导入验收需显式设置 MMEVLA_STAGE2_IMPORT_ENV；本测试不负责安装")
    environment = Path(selected).resolve(strict=True)
    project = C._REPO_ROOT / "scripts/dataset/wan_stage2"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)
    env.update(UV_PROJECT_ENVIRONMENT=str(environment), CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    code = """
import importlib.metadata
import pathlib
import sys
assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, sys.argv[2])
import numpy as np
from mme_vla_suite.datastore import motion_store
import ml_dtypes
assert importlib.metadata.version('ml-dtypes') == '0.4.1'
assert pathlib.Path(ml_dtypes.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
assert np.dtype(ml_dtypes.bfloat16).itemsize == 2
assert callable(motion_store.layout_spec)
print('STAGE2_DATASTORE_IMPORT=PASS ml_dtypes=0.4.1 numpy=' + np.__version__)
"""
    result = subprocess.run(["uv", "run", "--project", str(project), "--no-sync", "python", "-I", "-c", code,
                             str(environment), str(C._REPO_ROOT / "src")], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STAGE2_DATASTORE_IMPORT=PASS" in result.stdout


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return {"path": str(path.name), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def provenance(tmp_path):
    pin = json.loads((C._REPO_ROOT / "scripts/dataset/wan_stage2/SOURCE_PIN.json").read_text())
    sha = "a" * 64
    leaves = {f"leaf{i}": {"dtype": "float32", "shape": [1], "sha256": sha} for i in range(65)}
    state = {"algorithm": "named-tensors-v1", "leaves": leaves,
             "sha256": hashlib.sha256(json.dumps(leaves, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    encoder = dict(run_name="stage2-full144-1025-a", checkpoint_name="checkpoint_epoch_105.pt", epoch=105,
                   global_step=123, state_key="model", weight_type="ema",
                   arch="wan-latent-stage2-blockwise-pool4-three-decoder", checkpoint_sha256=sha,
                   config_sha256="b" * 64, fingerprint_sha256=sha, source_times=[0, 3, 5, 8], output_shape=[1, 1024])
    numerics = dict(dtype="float32", autocast=False, matmul_allow_tf32=False, cudnn_allow_tf32=False,
                    batch=1, torch="2.9.0+cu128", cuda="12.8", cudnn=91002, numpy="2.4.4",
                    gpu_name="NVIDIA A100-SXM4-80GB", driver="580.95.05")
    vae = dict(vae_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", vae_state_sha256=sha, vae_dtype="float32",
               latent_mode="mode", batch=1, tf32="off", amp="off", torch="2.9.0+cu128", cuda="12.8",
               cudnn=91002, diffusers="0.39.0", module_sha256=sha, encoder_src_sha256=sha)
    online = {k: vae[k] for k in ("vae_id", "vae_state_sha256", "vae_dtype", "latent_mode", "batch", "tf32", "amp")}
    online.update(adapter_sha256=pin["vae_adapter_sha256"], numerics=numerics,
                  preprocess=dict(frame_size=256, input_dtype="uint8", input_layout="THWC", window_frames=33))
    windows = [dict(task=f"task{i // 4}", frame_ids=list(range(33)), source_frame_indices=[{}] * 33,
                    sources=[{}], source_grid=0, input_frames_sha256=sha, latent_block_sha256=sha) for i in range(64)]
    g0 = dict(schema="stage2-validation-inputs-v1", gate="G0", status="PASS", seed=0,
              manifest_sha256=sha, latent_metadata_sha256=sha, windows=windows,
              counts=dict(windows=64, tasks=16, sha_mismatches=0, demo_tail=16))
    evidence = {"G0": _write(tmp_path / "G0.json", g0)}
    for gate, metrics in (("G2", dict(windows=64, demo_tail=16, worst_mse=0, worst_max=0)),
                          ("G3", dict(windows=32, seed=0, max_abs=0, geometry_all_true=1, leaves=65,
                                      state_key="model", exact=1, live_negative_rejected=1))):
        evidence[gate] = _write(tmp_path / f"{gate}.json", dict(
            schema="stage2-validation-gate-v1", gate=gate, status="PASS", checkpoint_sha256=sha,
            samples_sha256=evidence["G0"]["sha256"], metrics=metrics, source_pin=pin, numerics=numerics,
            **({"online_vae": online} if gate == "G2" else {})))
    receipt = dict(schema_version="mixed8-vla-latent-restore/v1", status="PASS", new_encoder_identity=None,
                   vae=dict(identity=vae, vae_state_sha256=sha, historical_module_sha256=sha,
                            historical_encoder_src_sha256=sha), latents=dict(total_metadata_sha256=sha),
                   parent=dict(manifest_canonical_sha256=sha))
    evidence["restore_receipt"] = _write(tmp_path / "receipt.json", receipt)
    evidence["SOURCE_PIN"] = _write(tmp_path / "pin.json", pin)
    return dict(encoder=encoder, encoder_state_sha256=state, source_pin=pin, numerics=numerics, vae=vae,
                stage2_compat=dict(schema="stage2-compat-v1", evidence=evidence, historical_vae=copy.deepcopy(vae),
                                   online_vae=online, restore_receipt_sha256=evidence["restore_receipt"]["sha256"]))


@pytest.fixture
def stub_process(monkeypatch):
    """只替换测试解释器选择；实际子进程、脚本、socket 和封包全部走生产代码。"""
    real_popen = subprocess.Popen
    calls = []
    def spawn(argv, **kwargs):
        if not isinstance(argv, list) or argv[:3] != ["uv", "run", "--project"]:
            return real_popen(argv, **kwargs)
        calls.append((argv, kwargs["env"]))
        assert argv[:3] == ["uv", "run", "--project"]
        assert Path(argv[3]).name == "wan_stage2"
        assert Path(kwargs["env"]["UV_PROJECT_ENVIRONMENT"]).name == "wan-stage2"
        assert "--no-sync" in argv
        return real_popen([sys.executable, *argv[6:]], **kwargs)
    monkeypatch.setattr(C.subprocess, "Popen", spawn)
    return calls


def test_three_complete_stage2_ipc_windows(stub_process, tmp_path):
    client = C.MotionEncoderClient(online_gpu="", encoder_kind=C.STAGE2_KIND, source_run=SOURCE_RUN,
                                   stub=True, v1_store=tmp_path, handshake_timeout=20)
    try:
        assert client.provenance["shape"] == [1024]
        assert client.provenance["response_bytes"] == 4096
        for start, real in ((0, 33), (32, 33), (64, 17)):
            ids = list(range(start, start + real)) + [start + real - 1] * (33 - real)
            token = client(np.stack([P.stub_frame(i) for i in ids]), start)
            assert token.shape == (1024,) and token.dtype == np.float32 and token.nbytes == 4096
            np.testing.assert_array_equal(token, np.full(1024, start, np.float32))
        assert client.n_calls == 3 and client.alive
    finally:
        client.close()
    assert not client.alive and len(stub_process) == 1


@pytest.mark.parametrize("field,value", [("encoder_kind", "v10"), ("dtype", "float16"),
                                        ("shape", [768]), ("response_bytes", 3072), ("stub", False)])
def test_mixed_handshake_rejected(stub_process, monkeypatch, tmp_path, field, value):
    original = P.recv_handshake
    def receive(*args):
        result = original(*args)
        result[field] = value
        return result
    monkeypatch.setattr(P, "recv_handshake", receive)
    with pytest.raises(RuntimeError, match="握手"):
        C.MotionEncoderClient(online_gpu="", encoder_kind=C.STAGE2_KIND, source_run=SOURCE_RUN,
                              stub=True, v1_store=tmp_path, handshake_timeout=20)


def test_legacy_default_response_unchanged():
    raw = b"x" * 3072
    assert P.RESPONSE_BYTES == 3072 and P.TOKEN_DIM == 768
    assert P.encode_response(raw) == struct.pack("<I", 0) + raw
    with _socket_pair() as pair:
        pair[0].sendall(P.encode_response(raw))
        assert P.recv_response(pair[1], time.monotonic() + 2) == raw


class _socket_pair:
    def __enter__(self):
        self.pair = socket.socketpair()
        return self.pair
    def __exit__(self, *_):
        for sock in self.pair:
            sock.close()


@pytest.mark.parametrize("size", [0, 3072, 4095, 4097])
def test_response_wrong_length_and_short_packet_rejected(size):
    with pytest.raises(P.ProtocolError):
        P.encode_response(b"x" * size, response_bytes=4096)
    if size < 4096:
        with _socket_pair() as (sender, receiver):
            sender.sendall(struct.pack("<II", 0, 4096) + b"x" * size)
            sender.shutdown(socket.SHUT_WR)
            with pytest.raises(P.ProtocolError, match="EOF"):
                P.recv_response(receiver, time.monotonic() + 2, response_bytes=4096)


@pytest.mark.parametrize("declared", [0, 3072, 4095, 4097, 2**32 - 1])
def test_stage2_wire_length_rejected(declared):
    with _socket_pair() as (sender, receiver):
        sender.sendall(struct.pack("<II", 0, declared))
        with pytest.raises(P.ProtocolError, match="length"):
            P.recv_response(receiver, time.monotonic() + 2, response_bytes=4096)


@pytest.mark.parametrize("response_bytes", [3072, 4096])
def test_error_response_readable_in_both_protocols(response_bytes):
    with _socket_pair() as (sender, receiver):
        sender.sendall(P.encode_error("预期编码失败"))
        with pytest.raises(P.ProtocolError, match="预期编码失败"):
            P.recv_response(receiver, time.monotonic() + 2, response_bytes=response_bytes)


def test_trailing_bytes_pollute_next_frame():
    """流式 socket 没有包边界；合法长度后的多字节只能在后续帧读取时拒绝。"""
    packet = P.encode_response(b"x" * 4096, response_bytes=4096)
    assert len(packet) == 4104
    with _socket_pair() as (sender, receiver):
        sender.sendall(packet + b"X" + packet)
        assert len(P.recv_response(receiver, time.monotonic() + 2, response_bytes=4096)) == 4096
        with pytest.raises(P.ProtocolError):
            P.recv_response(receiver, time.monotonic() + 2, response_bytes=4096)


def test_stage2_protocol_error_closes_process(stub_process, tmp_path):
    client = C.MotionEncoderClient(online_gpu="", encoder_kind=C.STAGE2_KIND, source_run=SOURCE_RUN,
                                   stub=True, v1_store=tmp_path, handshake_timeout=20)
    with pytest.raises(P.ProtocolError, match="stub"):
        client(np.zeros((33, 256, 256, 3), dtype=np.uint8), start_frame=17)
    assert client._sock is None and not client.alive


def test_nonstub_requires_provenance_before_spawn(monkeypatch):
    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: pytest.fail("缺来源不应启动子进程"))
    with pytest.raises(ValueError, match="store_provenance"):
        C.MotionEncoderClient(online_gpu=0, encoder_kind=C.STAGE2_KIND, source_run=SOURCE_RUN)


@pytest.mark.parametrize("group", C.STAGE2_GROUPS)
def test_nonstub_rejects_each_missing_group(provenance, tmp_path, group, monkeypatch):
    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: pytest.fail("缺来源不应启动子进程"))
    del provenance[group]
    with pytest.raises(ValueError):
        C.MotionEncoderClient(online_gpu=0, encoder_kind=C.STAGE2_KIND, source_run=SOURCE_RUN,
                              store_provenance=provenance, provenance_root=tmp_path,
                              expected_ckpt_sha256="a" * 64)


def _handshake_client(provenance, root):
    client = object.__new__(C.MotionEncoderClient)
    client._provenance_root = root
    client._online_gpu = "0"
    client.provenance = {k: copy.deepcopy(provenance[k]) for k in
                         ("encoder", "encoder_state_sha256", "source_pin", "numerics")}
    client.provenance.update(online_vae=copy.deepcopy(provenance["stage2_compat"]["online_vae"]),
                             pid=123, gpu_uuid="GPU-test", cuda_visible_devices="0")
    client._proc = SimpleNamespace(pid=123, poll=lambda: 0)
    return client


def test_complete_provenance_and_relative_evidence(provenance, tmp_path):
    client = _handshake_client(provenance, tmp_path)
    client._check_stage2_provenance(provenance)
    client.provenance["encoder"]["run_dir"] = "/different/machine/run"
    client._check_stage2_provenance(provenance)
    client.provenance["encoder"]["state_key"] = "model_live"
    with pytest.raises(ValueError, match="encoder"):
        client._check_stage2_provenance(provenance)


@pytest.mark.parametrize("gate", ["G0", "G2", "G3", "restore_receipt", "SOURCE_PIN"])
def test_modified_evidence_rejected(provenance, tmp_path, gate):
    ref = provenance["stage2_compat"]["evidence"][gate]
    (tmp_path / ref["path"]).write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        C.validate_stage2_evidence(provenance, tmp_path)


def test_rehashed_failing_gate_still_rejected(provenance, tmp_path):
    path = tmp_path / "G2.json"
    record = json.loads(path.read_text())
    record["metrics"]["worst_mse"] = 1
    provenance["stage2_compat"]["evidence"]["G2"] = _write(path, record)
    with pytest.raises(ValueError, match="G2"):
        C.validate_stage2_evidence(provenance, tmp_path)


@pytest.mark.parametrize("gate,group", [("G2", "source_pin"), ("G3", "source_pin"),
                                       ("G2", "numerics"), ("G3", "numerics"), ("G2", "online_vae")])
def test_gate_actual_sources_cannot_be_omitted(provenance, tmp_path, gate, group):
    path = tmp_path / f"{gate}.json"
    record = json.loads(path.read_text())
    del record[group]
    provenance["stage2_compat"]["evidence"][gate] = _write(path, record)
    with pytest.raises(ValueError, match=gate):
        C.validate_stage2_evidence(provenance, tmp_path)


@pytest.mark.parametrize("field,value", [("pid", 999), ("gpu_uuid", ""), ("cuda_visible_devices", "4")])
def test_real_worker_identity_rejected(provenance, tmp_path, field, value):
    client = _handshake_client(provenance, tmp_path)
    client.provenance[field] = value
    with pytest.raises(ValueError, match="身份"):
        client._check_stage2_provenance(provenance)


def test_nonstub_launch_uses_independent_project_and_registered_source(provenance, tmp_path, monkeypatch):
    handshake = _handshake_client(provenance, tmp_path).provenance
    handshake.update(encoder_kind=C.STAGE2_KIND, dtype="float32", shape=[1024], response_bytes=4096,
                      stub=False, protocol_sha256=P.protocol_sha256())
    captured = {}
    def spawn(argv, **kwargs):
        captured.update(argv=argv, env=kwargs["env"])
        return SimpleNamespace(pid=123, poll=lambda: 0)
    monkeypatch.setattr(C.subprocess, "Popen", spawn)
    monkeypatch.setattr(P, "recv_handshake", lambda *args: handshake)
    client = C.MotionEncoderClient(online_gpu=0, encoder_kind=C.STAGE2_KIND, source_run=SOURCE_RUN,
                                   store_provenance=provenance, provenance_root=tmp_path, v1_store=tmp_path,
                                   expected_ckpt_sha256=provenance["encoder"]["checkpoint_sha256"])
    try:
        argv = captured["argv"]
        assert Path(argv[3]).name == "wan_stage2"
        assert argv[argv.index("--ckpt") + 1] == str(tmp_path / "external/motionjepa" / SOURCE_RUN.split("#")[0])
        assert captured["env"]["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "venvs/wan-stage2")
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert "PYTHONPATH" not in captured["env"]
    finally:
        client.close()


def test_relocated_encoder_entities(provenance, tmp_path):
    encoder = provenance["encoder"]
    files = []
    folder = tmp_path / "encoder_dependencies" / encoder["run_name"]
    folder.mkdir(parents=True)
    for role, name, field in (("encoder_checkpoint", encoder["checkpoint_name"], "checkpoint_sha256"),
                               ("encoder_config", "config.yaml", "config_sha256")):
        path = folder / name
        path.write_bytes(role.encode())
        encoder[field] = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(dict(role=role, relative_path=str(path.relative_to(tmp_path)), sha256=encoder[field],
                          bytes=path.stat().st_size, source_path="/unavailable/original/machine"))
    record = dict(schema="stage2-encoder-dependencies-v1", files=files,
                   **{k: provenance[k] for k in ("encoder", "source_pin", "vae")})
    _write(tmp_path / "encoder_dependencies.json", record)
    assert C.resolve_stage2_dependencies(tmp_path, provenance) == folder
    files[0]["relative_path"] = "../escape.pt"
    _write(tmp_path / "encoder_dependencies.json", record)
    with pytest.raises(ValueError, match="相对路径"):
        C.resolve_stage2_dependencies(tmp_path, provenance)


def test_policy_passes_full_sources_and_overrides_gpu_in_copy(provenance, tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from mme_vla_suite.policies import policy_config as PC
    original = OmegaConf.create({"motion": {"enabled": True, "encoder_kind": C.STAGE2_KIND,
                                           "source_run": SOURCE_RUN, "online_gpu": 4}})
    raw = OmegaConf.to_yaml(original).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (tmp_path / "history_config.resolved.yaml").write_bytes(raw)
    (tmp_path / "history_config.resolved.sha256").write_text(digest)
    provenance.update(motion_enabled=True, resolved_sha256=digest)
    _write(tmp_path / "motion_provenance.json", provenance)
    captured = {}

    @dataclasses.dataclass
    class ModelConfig:
        history_config: object
        use_history: bool = True
        def load(self, params, **kwargs):
            captured["history"] = self.history_config
            return object()

    @dataclasses.dataclass
    class TrainConfig:
        model: object
        data: object
        assets_dirs: tuple = ()
        policy_metadata: object = None

    transforms = SimpleNamespace(inputs=[], outputs=[])
    data = SimpleNamespace(create=lambda *a: SimpleNamespace(
        asset_id="fixture", data_transforms=transforms, model_transforms=transforms, use_quantile_norm=False))
    config = TrainConfig(model=ModelConfig(original), data=data)
    monkeypatch.setattr(PC._model, "restore_params", lambda *a, **k: {})
    monkeypatch.setattr(PC, "_params_have_motion", lambda p: True)
    monkeypatch.setattr(PC, "_assert_param_tree_exact", lambda *a: None)
    monkeypatch.setattr(PC._policy, "MME_VLA_Policy", lambda *a, **k: k)
    def client(**kwargs):
        captured["client"] = kwargs
        return "client"
    monkeypatch.setattr(C, "MotionEncoderClient", client)
    result = PC.create_trained_policy(config, tmp_path / "19", norm_stats={}, motion_online_gpu=0)
    assert result["motion_enc_fn"] == "client"
    assert captured["client"]["encoder_kind"] == C.STAGE2_KIND
    assert captured["client"]["provenance_root"] == tmp_path
    assert captured["client"]["store_provenance"] == provenance
    assert captured["client"]["online_gpu"] == 0
    assert captured["history"].motion.online_gpu == 0 and original.motion.online_gpu == 4
    assert (tmp_path / "history_config.resolved.yaml").read_bytes() == raw
