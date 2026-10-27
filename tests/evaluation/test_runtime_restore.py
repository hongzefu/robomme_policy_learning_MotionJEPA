"""纯CPU恢复检查负控；不启动训练或GPU。"""

import copy
import dataclasses
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation.check_runtime import bf16_parameter_summary, check_restore_tree, check_single_gpu_token, validate_restore_purpose, fixed_noise_repeatability


def test_actual_parameter_tree_and_digest():
    tree = {"model": {"weight": np.arange(6, dtype=np.float32).reshape(2, 3)}}
    result = check_restore_tree(tree, copy.deepcopy(tree))
    assert result["strict_restore"]["leaves"] == 1
    assert result["parameters"]["leaves"]["model/weight"]["dtype"] == "float32"
    assert len(result["parameters"]["sha256"]) == 64


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype", "nan", "empty"])
def test_strict_restore_rejects_mismatch(mutation):
    expected = {"weight": np.zeros((2, 3), np.float32)}
    actual = copy.deepcopy(expected)
    if mutation == "missing":
        actual.pop("weight")
    elif mutation == "extra":
        actual["extra"] = np.zeros(1)
    elif mutation == "shape":
        actual["weight"] = np.zeros((3, 2), np.float32)
    elif mutation == "dtype":
        actual["weight"] = actual["weight"].astype(np.float16)
    elif mutation == "nan":
        actual["weight"][0, 0] = np.nan
    else:
        expected, actual = {}, {}
    with pytest.raises(ValueError):
        check_restore_tree(expected, actual)


@pytest.mark.parametrize("token", ["4", "0", "GPU-some-uuid", "MIG-some-uuid"])
def test_slurm_single_token_is_preserved(token):
    assert check_single_gpu_token(token, environ={"CUDA_VISIBLE_DEVICES": token, "SLURM_JOB_ID": "123", "SLURM_STEP_ID": "2"}) == token


@pytest.mark.parametrize("token,environment", [("0", {"CUDA_VISIBLE_DEVICES": "4"}),
    ("0,1", {"CUDA_VISIBLE_DEVICES": "0,1"}), ("0", {"CUDA_VISIBLE_DEVICES": "0"})])
def test_no_implicit_physical_gpu_fallback(token, environment):
    with pytest.raises(ValueError):
        check_single_gpu_token(token, environ=environment)


def test_validation_is_not_formal_60000():
    validate_restore_purpose("/run/1234", 1234, "validation")
    validate_restore_purpose("/run/60000", 60000, "production")
    with pytest.raises(ValueError):
        validate_restore_purpose("/run/1234", 1234, "production")
    with pytest.raises(ValueError):
        validate_restore_purpose("/run/60000", 1234, "validation")


def test_bf16_summary_matches_actual_rounded_bytes():
    import ml_dtypes
    value = np.array([1.003, 2.007], np.float32)
    assert bf16_parameter_summary({"w": value}) == bf16_parameter_summary({"w": value.astype(ml_dtypes.bfloat16)})
    assert bf16_parameter_summary({"w": np.asarray(1.0, np.float32)})["leaves"]["w"]["shape"] == []


def test_same_machine_replay_uses_real16_and_complete_noise():
    class Policy:
        _sample_kwargs = {"original": True}
        def __init__(self):
            self.counts = []
        def reset(self):
            self.counts.append([])
        def add_buffer(self, obs):
            self.counts[-1].append(len(obs["images"]))
        def infer(self, obs):
            assert self._sample_kwargs["noise"].shape == (1, 20, 32)
            assert self._sample_kwargs["num_steps"] == 10
            return {"actions": np.zeros((20, 8), np.float32)}
    policy = Policy()
    arrays = {"images": np.zeros((131, 2, 2, 3), np.uint8), "states": np.zeros((131, 8), np.float32),
              "wrist_image": np.zeros((2, 2, 3), np.uint8)}
    record = {"exec_start_idx": 66, "exec_batch_lengths": [33, 32], "prompt": "真实输入契约"}
    output, batches = fixed_noise_repeatability(policy, record, arrays, np.zeros((1, 20, 32), np.float32),
        lambda images, states, boundary: {"images": images, "state": states, "exec_start_idx": boundary})
    assert batches == [67, 16, 16, 16, 16]
    assert policy.counts == [batches, batches] and record["exec_batch_lengths"] == [33, 32]
    assert output[0].shape == (20, 8) and policy._sample_kwargs == {"original": True}


def test_binding_rejection_happens_before_model_load(tmp_path, monkeypatch):
    from scripts.training import serve_policy
    def reject_binding(*args, **kwargs):
        assert kwargs["expected_purpose"] == "validation"
        assert kwargs["expected_run_identity_sha256"] == "a" * 64
        raise ValueError("完整模型成员字节与binding不符")
    monkeypatch.setitem(sys.modules, "scripts.evaluation.manifest", SimpleNamespace(verify_model_binding=reject_binding))
    monkeypatch.setattr(serve_policy, "create_policy", lambda *args: pytest.fail("binding失败不得先加载权重"))
    args = serve_policy.Args(policy=serve_policy.Checkpoint(config="mme_vla_suite_b128_80k", dir=tmp_path / "20000"),
        eval_target_contract=tmp_path / "runtime.json", model_binding=tmp_path / "binding.json",
        run_identity_sha256="a" * 64, eval_purpose="validation", checkpoint_step=20000)
    with pytest.raises(ValueError, match="完整模型成员"):
        serve_policy.main(args)


def test_explicit_stage2_interpreter_runs_real_ipc_without_install(tmp_path):
    from mme_vla_suite.policies.motion_client import MotionEncoderClient
    from mme_vla_suite.policies.motion_protocol import stub_frame
    # 直接使用uv管理venv入口；真实子进程/socket，stub不加载权重或GPU。
    client = MotionEncoderClient(online_gpu="", encoder_kind="stage2_pool4_concat", stub=True,
        source_run="test/checkpoint_epoch_1.pt#model", v1_store=tmp_path, stage2_python=sys.executable,
        handshake_timeout=30)
    try:
        interpreter = str(Path(sys.executable).absolute())
        assert client.argv[0] == interpreter
        assert client.argv[1].endswith("/scripts/dataset/wan_stage2/stage2_sidecar.py")
        assert "uv" not in client.argv and "--python" not in client.argv
        assert client.provenance["pid"] == client._proc.pid
        actual_env = dict(item.split(b"=", 1) for item in Path(f"/proc/{client._proc.pid}/environ").read_bytes().split(b"\0") if b"=" in item)
        selected_prefix = Path(interpreter).parent.parent
        assert actual_env[b"UV_PROJECT_ENVIRONMENT"].decode() == str(selected_prefix)
        assert actual_env[b"CUDA_VISIBLE_DEVICES"] == b""
        assert b"PYTHONPATH" not in actual_env and b"PYTHONHOME" not in actual_env
        # 同一入口真实启动对照，防止resolve到uv底层解释器而丢掉venv前缀。
        probe = json.loads(subprocess.check_output([interpreter, "-c",
            "import json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix}))"], text=True))
        assert probe["executable"] == interpreter
        assert Path(probe["prefix"]).resolve() == selected_prefix.resolve()
        output = client(np.stack([stub_frame(i) for i in range(33)]), 0)
        assert output.shape == (1024,) and output.dtype == np.float32
        np.testing.assert_array_equal(output, np.zeros(1024, np.float32))
    finally:
        client.close()
    assert not client.alive


def test_interpreter_and_gpu_only_override_memory_snapshot(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from mme_vla_suite.policies import policy_config as pc, motion_client
    original = OmegaConf.create({"motion": {"enabled": True, "encoder_kind": "stage2_pool4_concat",
                                           "source_run": "test/checkpoint_epoch_1.pt#model", "online_gpu": 4}})
    raw = OmegaConf.to_yaml(original).encode()
    (tmp_path / "history_config.resolved.yaml").write_bytes(raw)
    (tmp_path / "motion_provenance.json").write_text(json.dumps({"vae": {}, "encoder": {"checkpoint_sha256": "a" * 64}}))
    captured = {}
    @dataclasses.dataclass
    class Model:
        history_config: object
        use_history: bool = True
        def load(self, *args, **kwargs):
            captured["history"] = self.history_config
            return object()
    @dataclasses.dataclass
    class Train:
        model: object
        data: object
        assets_dirs: tuple = ()
        policy_metadata: object = None
    group = SimpleNamespace(inputs=[], outputs=[])
    train = Train(Model(original), SimpleNamespace(create=lambda *args: SimpleNamespace(
        asset_id="test", data_transforms=group, model_transforms=group, use_quantile_norm=False)))
    monkeypatch.setattr(pc._model, "restore_params", lambda *args, **kwargs: {})
    monkeypatch.setattr(pc, "_params_have_motion", lambda params: True)
    monkeypatch.setattr(pc, "_load_resolved_snapshot", lambda root: (original, True))
    monkeypatch.setattr(pc, "_assert_param_tree_exact", lambda *args: None)
    monkeypatch.setattr(pc._policy, "MME_VLA_Policy", lambda *args, **kwargs: kwargs)
    def client(**kwargs):
        captured["client"] = kwargs
        return object()
    monkeypatch.setattr(motion_client, "MotionEncoderClient", client)
    result = pc.create_trained_policy(train, tmp_path / "20000", norm_stats={},
        motion_online_gpu="GPU-assigned", stage2_python="/explicit/env/bin/python", eval_target_contract="cert.json")
    assert original.motion.online_gpu == 4 and "stage2_python" not in original.motion
    assert (tmp_path / "history_config.resolved.yaml").read_bytes() == raw
    assert captured["history"].motion.online_gpu == "GPU-assigned"
    assert captured["client"]["stage2_python"] == "/explicit/env/bin/python"
    assert captured["client"]["eval_target_contract"] == "cert.json"
    assert result["metadata"]["model_seed"] == 42
