"""合入 v17（fd20feb）后的 CPU 冒烟：导入、两份 modulation YAML、四种 motion kind 的客户端与 model_kind 推断、
EnvRunner 的 xhard + metadata_dir 路径。不加载权重、不起 GPU、不起 sidecar 子进程、不建真实仿真环境。"""

from __future__ import annotations

import pathlib
import sys
import types

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "src/mme_vla_suite/models/config/robomme"
YAML_A = "perceptual-framesamp-modul-8frame-8x8-motion.yaml"       # v10 source_run 旧路径
YAML_B = "perceptual-framesamp-modul-8frame-8x8-wantp768.yaml"     # v17 零参数池化 768

sys.path.insert(0, str(REPO / "examples/robomme"))


def _load(name):
    from omegaconf import OmegaConf
    return OmegaConf.load(CONFIG_DIR / name)


def test_imports():
    import env_runner  # noqa: F401
    from mme_vla_suite.policies import framesamp_memory, motion_client, policy_config  # noqa: F401
    assert pathlib.Path(policy_config.__file__).resolve().is_relative_to(REPO)


@pytest.mark.parametrize("name, kind, model_kind", [
    (YAML_A, "v10", "modul_v10_768"),
    (YAML_B, "wan_fixedpool768", "wan_fixedpool768"),
])
def test_modulation_yaml_configs(name, kind, model_kind):
    from mme_vla_suite.policies import policy_config as pc
    from mme_vla_suite.policies.framesamp_memory import FrameSampMemory

    hc = _load(name)
    assert hc.integration_type == "modulation" and hc.motion.enabled
    assert pc._motion_encoder_kind(hc.motion) == kind
    assert pc._motion_model_kind(hc.motion, hc.integration_type) == model_kind
    mcfg = {k: hc.motion[k] for k in ("stride", "window_frames", "budget", "frame_size", "pos_dim", "dim",
                                      "window_direction", "grid_origin", "demo_min_real_frames", "demo_tail_pad")}
    mem = FrameSampMemory(num_views=1, token_per_image=int(hc.token_per_image), vision_enc_fn=lambda x: x,
                          motion_enc_fn=lambda frames: np.zeros(768, np.float32), motion_cfg=mcfg)
    assert mem.motion_budget == 160 and mem.motion_overflow == "raise"
    assert mem.motion_last_k == 0 and mem.motion_downsample_max_k == 0 and mem.motion_downsample_steps == 0


def test_model_kind_existing_unchanged():
    from omegaconf import OmegaConf
    from mme_vla_suite.policies import policy_config as pc
    for kind in ("stage2_pool4_concat", "wan_fixedpool1024"):
        for integ in ("modulation", "context"):
            assert pc._motion_model_kind(OmegaConf.create({"encoder_kind": kind}), integ) == kind
    # v10 非 modulation 保持 BASE 的 "history"；encoder_kind=None 与 v10 同义
    assert pc._motion_model_kind(OmegaConf.create({"encoder_kind": None}), "context") == "history"
    assert pc._motion_model_kind(OmegaConf.create({"encoder_kind": None}), "modulation") == "modul_v10_768"


class _FakeProc:
    returncode = None

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


@pytest.fixture
def fake_sidecar(monkeypatch):
    """替换子进程与握手：记录 argv，回一份按 kind 构造的 provenance。"""
    from mme_vla_suite.policies import motion_client as mc
    from mme_vla_suite.policies import motion_protocol as P
    from mme_vla_suite.shared import wan_pool768 as wp

    seen = {}

    def popen(argv, **kw):
        seen["argv"] = list(argv)
        return _FakeProc()

    def handshake(sock, deadline):
        prov = {"protocol_sha256": P.protocol_sha256(), "stub": True}
        if "--encoder-kind" in seen["argv"]:
            prov.update(encoder_kind=wp.POOL768_ENCODER_KIND, readout=wp.POOL768_READOUT,
                        pool_source_sha256=wp.pool_source_sha256())
        return prov

    monkeypatch.setattr(mc.subprocess, "Popen", popen)
    monkeypatch.setattr(mc.P, "recv_handshake", handshake)
    monkeypatch.setattr(mc.MotionEncoderClient, "close", lambda self: None)
    return seen


def test_stub_clients_four_kinds(fake_sidecar, monkeypatch, tmp_path):
    from mme_vla_suite.policies import motion_client as mc
    from mme_vla_suite.shared import wan_pool768 as wp

    # v10（encoder_kind=None 同义）
    c = mc.MotionEncoderClient(online_gpu=0, stub=True, encoder_kind=None, v1_store=tmp_path)
    assert c.encoder_kind == "v10" and "--encoder-kind" not in fake_sidecar["argv"]
    # wan_fixedpool768：argv 带 --encoder-kind，握手核 kind / readout / 池化源码 sha
    c = mc.MotionEncoderClient(online_gpu=0, stub=True, encoder_kind="wan_fixedpool768", v1_store=tmp_path,
                               readout=wp.POOL768_READOUT, expected_pool_sha256=wp.pool_source_sha256())
    assert c.encoder_kind == "wan_fixedpool768"
    assert fake_sidecar["argv"][-2:] == ["--encoder-kind", "wan_fixedpool768"]
    with pytest.raises(RuntimeError, match="池化身份"):
        mc.MotionEncoderClient(online_gpu=0, stub=True, encoder_kind="wan_fixedpool768", v1_store=tmp_path,
                               readout=wp.POOL768_READOUT, expected_pool_sha256="0" * 64)
    with pytest.raises(ValueError):
        mc.MotionEncoderClient(online_gpu=0, stub=True, encoder_kind="wan_fixedpool768", v1_store=tmp_path,
                               source_run="x/checkpoint_epoch_1.pt#encoder",
                               readout=wp.POOL768_READOUT, expected_pool_sha256=wp.pool_source_sha256())
    # 阶段二与固定池化 1024：会话初始化替换为记录器，只核分派
    called = []
    monkeypatch.setattr(mc.MotionEncoderClient, "_init_stage2", lambda self, **kw: called.append("stage2"))
    monkeypatch.setattr(mc.MotionEncoderClient, "_init_timepool", lambda self, **kw: called.append("timepool"))
    c = mc.MotionEncoderClient(online_gpu=0, stub=True, encoder_kind="stage2_pool4_concat")
    assert c.encoder_kind == "stage2_pool4_concat"
    c = mc.MotionEncoderClient(online_gpu=0, stub=True, encoder_kind="wan_fixedpool1024")
    assert c.encoder_kind == "wan_fixedpool1024" and called == ["stage2", "timepool"]
    with pytest.raises(ValueError, match="未知"):
        mc.MotionEncoderClient(online_gpu=0, stub=True, encoder_kind="nope")


def test_pool_motion_client_routing(monkeypatch, tmp_path):
    """policy_config 的池化 768 分支：读快照 kind / readout，环境变量覆盖卡号。"""
    from mme_vla_suite.policies import motion_client as mc
    from mme_vla_suite.policies import policy_config as pc

    got = {}
    monkeypatch.setattr(mc, "MotionEncoderClient", lambda **kw: got.update(kw) or "client")
    monkeypatch.setenv("MMEVLA_MOTION_ONLINE_GPU", "3")
    hc = _load(YAML_B)
    assert pc._make_pool_motion_client(hc.motion, tmp_path, motion_stub=True) == "client"
    assert got["encoder_kind"] == "wan_fixedpool768" and got["online_gpu"] == "3"
    assert got["readout"] == "wan_raw_timepool8_adaptive6x8_v1" and "source_run" not in got


def test_env_runner_xhard_metadata_dir(monkeypatch, tmp_path):
    built = []

    class FakeBuilder:
        def __init__(self, **kw):
            built.append(kw)

    robomme = types.ModuleType("robomme")
    robomme_env = types.ModuleType("robomme.robomme_env")
    wrapper = types.ModuleType("robomme.env_record_wrapper")
    wrapper.BenchmarkEnvBuilder = FakeBuilder
    robomme.robomme_env, robomme.env_record_wrapper = robomme_env, wrapper
    monkeypatch.setitem(sys.modules, "robomme", robomme)
    monkeypatch.setitem(sys.modules, "robomme.robomme_env", robomme_env)
    monkeypatch.setitem(sys.modules, "robomme.env_record_wrapper", wrapper)
    monkeypatch.setitem(sys.modules, "robomme_xhard", None)       # 一旦导入即 ImportError
    monkeypatch.delenv("XHARD_METADATA_DIR", raising=False)
    import env_runner

    runner = env_runner.EnvRunner("RouteStick", str(tmp_path), max_steps=1300, dataset="test",
                                  difficulty="xhard", metadata_dir=str(tmp_path))
    assert runner.route == "plfull" and runner.build_id == "RouteStick" and runner.env_id == "RouteStick"
    assert built[-1]["env_id"] == "RouteStick" and built[-1]["override_metadata_path"] == tmp_path
    assert runner.env is None and runner.difficulty is None and runner.expected_difficulty == "xhard"
    # 非 xhard 难度同一路径；缺 difficulty 拒绝
    runner = env_runner.EnvRunner("RouteStick", str(tmp_path), difficulty="hard", metadata_dir=str(tmp_path))
    assert runner.route == "plfull" and runner.build_id == "RouteStick"
    with pytest.raises(ValueError):
        env_runner.EnvRunner("RouteStick", str(tmp_path), metadata_dir=str(tmp_path))
    # SpecEnvRunner 仍可构造（不注入 demo 前缀时不导入 cv2 / h5py）
    spec = env_runner.SpecEnvRunner("RouteStick", "xhard", {}, str(tmp_path))
    assert spec.group_difficulty == "xhard" and spec._demo_store is None
