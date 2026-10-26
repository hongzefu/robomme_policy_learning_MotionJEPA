"""G2例外producer的CPU控制流合同；不加载torch、不编码任何窗口。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dataset/wan_stage2/accept_stage2_g2_exception.py"
SPEC = importlib.util.spec_from_file_location("_stage2_g2_exception_record", SCRIPT)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)

# 此处保留的是冻结失败判定文字，不包含原始图像、latent或权重。
ORIGINAL_FAILURE = ('{"error":"G2 超阈值：ButtonUnmask_ep178_exec:10 mse=1.747542904391321e-07 '
                    'max=0.013544513720267467","gate":"G2","producer":"stage2-numeric-validator-v1",'
                    '"schema":"stage2-validation-gate-v1","status":"FAIL"}\n')


def synthetic_samples():
    windows = []
    for task in range(16):
        for episode in range(4):
            windows.append(dict(key=f"Task{task}_ep{episode}_demo", task=f"Task{task}", episode=episode,
                                segment_kind="demo", source_grid=0, real_frames=17, pad_frames=16,
                                pad_source_frame=16, frame_ids=list(range(17)) + [16] * 16,
                                input_frames_sha256="a" * 64, latent_block_sha256="b" * 64,
                                latent_path="不应读取的latent.bin",
                                sources=[dict(kind="demo_h5", path="不应读取的frames.h5", sha256="c" * 64)],
                                source_frame_indices=[dict(source=0, index=i) for i in range(17)]
                                + [dict(source=0, index=16)] * 16))
    return dict(schema="stage2-validation-inputs-v1", gate="G0", status="PASS", seed=0, windows=windows,
                manifest_sha256="d" * 64, latent_metadata_sha256="e" * 64,
                counts=dict(windows=64, tasks=16, demo_tail=64, sha_mismatches=0))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    from mme_vla_suite.datastore import motion_store as ms

    paths = dict(failure=tmp_path / "G2.json", samples=tmp_path / "windows.json", ckpt=tmp_path / "checkpoint.pt",
                 out=tmp_path / "G2-waiver.json")
    paths["failure"].write_bytes(ORIGINAL_FAILURE.encode("utf-8"))
    paths["samples"].write_bytes(A.V.canonical(synthetic_samples()) + b"\n")
    paths["ckpt"].write_bytes(b"synthetic-checkpoint-only")
    checkpoint_sha, samples_sha = A.V.sha_file(paths["ckpt"]), A.V.sha_file(paths["samples"])
    # 合成文件不可能拥有生产SHA；仅测试中局部替换两摘要，原失败UTF8仍用固定真摘要。
    monkeypatch.setattr(A, "CHECKPOINT_SHA256", checkpoint_sha)
    monkeypatch.setattr(A, "SAMPLES_SHA256", samples_sha)
    if hasattr(ms, "STAGE2_G2_WAIVER_CHECKPOINT_SHA256"):
        monkeypatch.setattr(ms, "STAGE2_G2_WAIVER_CHECKPOINT_SHA256", checkpoint_sha)
        monkeypatch.setattr(ms, "STAGE2_G2_WAIVER_SAMPLES_SHA256", samples_sha)
    calls = []
    pin = A.V.read_json(A.HERE / "SOURCE_PIN.json")
    numerics = dict(dtype="float32", autocast=False, matmul_allow_tf32=False, cudnn_allow_tf32=False,
                    batch=1, torch="2.9.0+cu128", cuda="12.8", cudnn=91002, numpy="2.4.4",
                    gpu_name="NVIDIA A100-SXM4-80GB", driver="synthetic-driver")
    vae_sha = "9980d252230c265cc2869466a74f85f5ee45b01ea9521bbb31159f90b75fe6d0"
    vae = SimpleNamespace(use_tiling=False, use_slicing=False)
    vae_info = dict(vae_id=A.VAE_ID, vae_state_sha256=vae_sha, vae_dtype="float32", latent_mode="mode",
                    batch=1, tf32="off", amp="off")

    def load_vae(vae_id, device, *, expected_state_sha256, strict_versions):
        calls.append("load_vae")
        assert (vae_id, device, expected_state_sha256, strict_versions) == (A.VAE_ID, "cuda:0", vae_sha, True)
        return vae, vae_info

    adapter = SimpleNamespace(VAE_STATE_SHA256_EXPECTED=vae_sha, load_vae=load_vae)
    info = dict(encoder=dict(checkpoint_sha256=checkpoint_sha), numerics=numerics,
                source_pin={key: pin[key] for key in ("mj_repo_commit", "export_source_sha256", "encoder_source_sha256")})

    def runtime(args):
        calls.append("setup_runtime")
        return None, None, adapter, None, info

    monkeypatch.setattr(A.V, "setup_runtime", runtime)

    def forbidden(*args, **kwargs):
        raise AssertionError("例外producer禁止读取原帧、latent或执行数值前向")

    for name in ("gate_g2", "gate_g3", "read_window_frames", "read_latent", "normalized_error"):
        monkeypatch.setattr(A.V, name, forbidden)
    original_validator = getattr(ms, "validate_stage2_g2_evidence", None)

    def validator(record, **kwargs):
        calls.append("validate")
        assert kwargs["demo_tail"] == 64
        if original_validator is not None:
            return original_validator(record, **kwargs)
        # 独立分支尚未合入公共函数时只mock消费接口，不替生产实现提供回退。
        assert record["status"] == "WAIVED" and record.get("metrics") is None
        assert record["source_pin"] == kwargs["source_pin"]
        assert record["numerics"] == kwargs["numerics"]
        assert record["online_vae"] == kwargs["online_vae"]
        return "WAIVED"

    monkeypatch.setattr(ms, "validate_stage2_g2_evidence", validator, raising=False)
    args = SimpleNamespace(**{key: str(value) for key, value in paths.items()},
                           expected_sha256=checkpoint_sha, device="cuda:0")
    return SimpleNamespace(paths=paths, args=args, calls=calls, info=info, vae=vae, vae_info=vae_info,
                           runtime=runtime, ms=ms)


def test_frozen_authorization_and_original_failure_bytes():
    assert A.CHECKPOINT_SHA256 == "27daa8320c9a392616778a7cd3e3a2041f9b52a440181f51df7e996352716035"
    assert A.SAMPLES_SHA256 == "4cb05b6b3097b22a8594576def1505957399a924d4189591980292e3bd320d30"
    assert A.V.sha_bytes(ORIGINAL_FAILURE.encode("utf-8")) == A.FAILURE_SHA256
    assert (A.USER_QUOTE, A.AUTHORIZED_AT, A.SCOPE) == (
        "可以忽略 继续做", "2026-10-26T19:21:45Z", "g2_cross_precision_numerics_only")


def test_accepted_record_preserves_fail_and_has_no_numeric_metrics(fixture, capsys):
    arguments = []
    for key, value in vars(fixture.args).items():
        arguments.extend(["--" + key.replace("_", "-"), value])
    before = {key: path.read_bytes() for key, path in fixture.paths.items() if key != "out"}
    assert A.main(arguments) == 0
    record = json.loads(fixture.paths["out"].read_bytes())
    assert record["status"] == "WAIVED" and record["producer"] == A.PRODUCER
    assert "metrics" not in record
    assert record["exception"]["numeric_validation_complete"] is False
    assert record["exception"]["original_failure_json"].encode("utf-8") == before["failure"]
    assert set(record["source_pin"]) == A.PIN_KEYS
    assert record["online_vae"]["numerics"] == record["numerics"]
    assert fixture.calls == ["setup_runtime", "load_vae", "validate"]
    assert {key: path.read_bytes() for key, path in fixture.paths.items() if key != "out"} == before
    assert capsys.readouterr().out == "G2_USER_EXCEPTION=ACCEPTED original_status=FAIL numeric_validation_complete=0\n"


@pytest.mark.parametrize("key", ["failure", "samples", "ckpt"])
def test_changed_fixed_input_stops_before_identity_load(fixture, key):
    with fixture.paths[key].open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="SHA"):
        A.accept_exception(fixture.args)
    assert not fixture.calls and not fixture.paths["out"].exists()


def test_other_expected_checkpoint_rejected_before_identity_load(fixture):
    fixture.args.expected_sha256 = "0" * 64
    with pytest.raises(ValueError, match="固定"):
        A.accept_exception(fixture.args)
    assert not fixture.calls


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling_symlink"])
def test_existing_output_is_never_overwritten(fixture, kind):
    output = fixture.paths["out"]
    if kind == "file":
        output.write_bytes(b"existing")
    else:
        output.symlink_to(fixture.paths["failure"] if kind == "symlink" else output.parent / "missing")
    with pytest.raises(FileExistsError):
        A.accept_exception(fixture.args)
    assert not fixture.calls
    if kind == "file":
        assert output.read_bytes() == b"existing"
    else:
        assert output.is_symlink()


def test_public_validator_must_accept_before_publish(fixture, monkeypatch):
    monkeypatch.setattr(fixture.ms, "validate_stage2_g2_evidence", lambda *args, **kwargs: "PASS")
    with pytest.raises(ValueError, match="WAIVED"):
        A.accept_exception(fixture.args)
    assert not fixture.paths["out"].exists()


def test_missing_common_validator_has_no_production_fallback(fixture, monkeypatch):
    monkeypatch.delattr(fixture.ms, "validate_stage2_g2_evidence")
    with pytest.raises(ImportError):
        A.accept_exception(fixture.args)
    assert not fixture.calls and not fixture.paths["out"].exists()


@pytest.mark.parametrize("change", ["wrong_vae", "tiling", "wrong_encoder", "wrong_source"])
def test_actual_identity_mismatch_cannot_be_waived(fixture, change):
    if change == "wrong_vae":
        fixture.vae_info["vae_state_sha256"] = "0" * 64
    elif change == "tiling":
        fixture.vae.use_tiling = True
    elif change == "wrong_encoder":
        fixture.info["encoder"]["checkpoint_sha256"] = "0" * 64
    else:
        fixture.info["source_pin"]["mj_repo_commit"] = "0" * 40
    with pytest.raises(ValueError):
        A.accept_exception(fixture.args)
    assert not fixture.paths["out"].exists()


def test_input_mutation_during_identity_collection_is_rejected(fixture, monkeypatch):
    def changed_runtime(args):
        result = fixture.runtime(args)
        fixture.paths["failure"].write_bytes(b"changed")
        return result

    monkeypatch.setattr(A.V, "setup_runtime", changed_runtime)
    with pytest.raises(ValueError, match="期间"):
        A.accept_exception(fixture.args)
    assert not fixture.paths["out"].exists()


def test_atomic_publish_rejects_concurrent_creator(fixture, monkeypatch):
    original_link = A.os.link

    def concurrent_link(source, target, **kwargs):
        staged = json.loads(Path(source).read_bytes())
        assert staged["status"] == "WAIVED"
        Path(target).write_bytes(b"concurrent-owner")
        return original_link(source, target, **kwargs)

    monkeypatch.setattr(A.os, "link", concurrent_link)
    with pytest.raises(FileExistsError):
        A.accept_exception(fixture.args)
    assert fixture.paths["out"].read_bytes() == b"concurrent-owner"
    assert list(fixture.paths["out"].parent.glob(".G2-waiver.json.*")) == []
