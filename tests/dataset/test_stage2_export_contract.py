"""新stage2导出合同：真实文件与模拟远端；没有网络和凭据访问。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[2]
COMMON = REPO / "scripts/dataset/hf_export/_stage2_vla_ckpt_export_common.sh"


@pytest.fixture
def export(monkeypatch):
    monkeypatch.setenv("STAGE2_REPO", str(REPO))
    text = COMMON.read_text()
    source = text.split("<<'STAGE2_PY'\n", 1)[1].rsplit("\nSTAGE2_PY", 1)[0]
    module = types.ModuleType("stage2_export_contract")
    module.__file__ = str(COMMON)
    exec(compile(source, str(COMMON), "exec"), module.__dict__)
    monkeypatch.setattr(module, "DISK_MARGIN", 0)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value, str) else value)
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def snapshot(tmp_path, export):
    e = export
    repo = tmp_path / "repo"
    run = repo / "v1-store/train-runs/mme_vla_suite_b128_80k" / e.RUN
    registered = repo / "v1-store/external/motionjepa" / e.ENC_RUN
    checkpoint = registered / "checkpoint_epoch_105.pt"
    checkpoint.parent.mkdir(parents=True)
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("archive/data.pkl", "可信模型元数据")
    config = write(registered / "config.yaml", "motion: {}\n")
    sha = digest(checkpoint)
    encoder = {"run_name": e.ENC_RUN, "checkpoint_name": checkpoint.name, "epoch": 105,
        "global_step": 7770, "arch": e.ARCH, "state_key": "model", "weight_type": "ema",
        "source_times": [0, 3, 5, 8], "output_shape": [1, 1024], "checkpoint_sha256": sha,
        "config_sha256": digest(config), "fingerprint_sha256": "f" * 64}
    pin = {"mj_repo_commit": "a" * 40, "export_source_sha256": "b" * 64,
           "encoder_source_sha256": "c" * 64, "vae_adapter_sha256": "d" * 64,
           "dependency_lock_sha256": "e" * 64}
    numerics = dict(dtype="float32", autocast=False, matmul_allow_tf32=False, cudnn_allow_tf32=False,
                    batch=1, torch="2.9.0", cuda="12.8", cudnn=91002, numpy="2.4.4", gpu_name="A100", driver="595")
    online_vae = dict(vae_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", vae_state_sha256="f"*64, vae_dtype="float32",
                      latent_mode="mode", batch=1, tf32="off", amp="off", adapter_sha256=pin["vae_adapter_sha256"],
                      numerics=numerics, preprocess=dict(frame_size=256,input_dtype="uint8",input_layout="THWC",window_frames=33))
    evidence = {}
    for gate in ("G0", "G2", "G3", "restore_receipt"):
        content = {"gate": gate, "status": "PASS", "checkpoint_sha256": sha}
        if gate in ("G2", "G3"):
            content["samples_sha256"] = evidence["G0"]["sha256"]
        if gate == "G2":
            content.update(schema="stage2-validation-gate-v1", source_pin=pin, numerics=numerics, online_vae=online_vae,
                           metrics=dict(windows=64,demo_tail=16,worst_mse=0,worst_max=0))
        path = write(run / f"motion_evidence/{gate}.json", content)
        evidence[gate] = {"path": path.relative_to(run).as_posix(), "sha256": digest(path)}
    pin_path = write(run / "motion_evidence/SOURCE_PIN.json", pin)
    evidence["SOURCE_PIN"] = {"path": "motion_evidence/SOURCE_PIN.json", "sha256": digest(pin_path)}
    history = {"budget": 512, "motion": {"enabled": True, "source_shape": [1, 1024],
        "readout": "stage2_pool4_concat_v1", "num_tokens": 1, "dim": 1024, "budget": 96,
        "stride": 32, "window_frames": 33, "demo_min_real_frames": 17,
        "demo_tail_pad": "repeat_last", "pos_dim": 256, "window_direction": "forward",
        "grid_origin": "segment_start", "frame_size": 256,
        "store_path": f"v1-store/datasets/{e.DATASET}/motion-stage2-1024",
        "source_run": f"{e.ENC_RUN}/checkpoint_epoch_105.pt#model",
        "encoder_kind": "stage2_pool4_concat", "checkpoint_sha256": sha, "online_gpu": 4}}
    resolved = write(run / "history_config.resolved.yaml", history)
    write(run / "history_config.resolved.sha256", digest(resolved) + "\n")
    write(run / "history_config.txt", e.HISTORY)
    prov = {"motion_enabled": True, "resolved_sha256": digest(resolved), "encoder": encoder,
        "encoder_state_sha256": {"algorithm": "named-tensors-v1"}, "source_pin": pin,
        "numerics": numerics, "vae": {"vae_id": "fixture"},
        "stage2_compat": {"schema": "stage2-compat-v1", "evidence": evidence, "online_vae": online_vae,
                          "restore_receipt_sha256": evidence["restore_receipt"]["sha256"]}}
    write(run / "motion_provenance.json", prov)
    deps = {"schema": "stage2-encoder-dependencies-v1", "encoder": encoder, "source_pin": pin,
            "vae": prov["vae"], "files": []}
    for role, path in (("encoder_checkpoint", checkpoint), ("encoder_config", config)):
        deps["files"].append({"role": role, "source_path": str(path),
            "relative_path": f"encoder_dependencies/{e.ENC_RUN}/{path.name}",
            "sha256": digest(path), "bytes": path.stat().st_size})
    write(run / "encoder_dependencies.json", deps)
    import numpy as np
    online_path = run / "online_restore_inputs.npz"
    online_arrays = {"images": np.zeros((82, 256, 256, 3), dtype=np.uint8),
                     "states": np.zeros((82, 8), dtype=np.float32),
                     "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8)}
    np.savez_compressed(online_path, **online_arrays)
    write(run / "online_restore_inputs.json", {"schema": "stage2-online-restore-inputs-v1",
        "payload": {"path": online_path.name, "sha256": digest(online_path)},
        "exec_start_idx": 17, "exec_batch_lengths": [33, 32], "prompt": "按下按钮",
        "sources": [{"path": "诊断原始图像", "sha256": "a" * 64}],
        "tokenizer": {"uri": e.TOKENIZER_URI, "sha256": e.TOKENIZER_SHA},
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                       "sha256": hashlib.sha256(v.tobytes()).hexdigest()} for k, v in online_arrays.items()}})
    refresh_ready(run, e)
    return types.SimpleNamespace(repo=repo, run=run, registered=registered, prov=prov, history=history)


def refresh_ready(run, e):
    files = {p.relative_to(run).as_posix(): digest(p) for p in run.rglob("*")
             if p.is_file() and (p.name in e.ROOT_FILES or p.parent.name == "motion_evidence")}
    write(run / e.READY, {"schema": "stage2-motion-snapshot-v1", "files": files})


def add_step(run, step, committed=True):
    write(run / str(step) / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": 123 if committed else None})
    write(run / str(step) / "params/_METADATA", {"tree_metadata": {"params": {}}})
    write(run / str(step) / "params/array.bin", "不可变参数内容")


def test_snapshot_includes_exact_ready_and_registered_dependencies(export, snapshot):
    sources, prov, history = export.snapshot_sources(snapshot.run, snapshot.registered)
    assert export.READY in sources
    assert len([x for x in sources if x.startswith("encoder_dependencies/")]) == 2
    assert prov["encoder"]["state_key"] == "model"
    assert history["motion"]["stride"] == 32


@pytest.mark.parametrize("change", ["missing_root", "unknown_root", "wrong_sha", "wrong_encoder", "wrong_library",
                                   "missing_evidence", "wrong_dependency", "missing_dependency", "wrong_g0", "wrong_online_npz"])
def test_snapshot_rejects_broken_chain(export, snapshot, change):
    run = snapshot.run
    if change == "missing_root":
        (run / "history_config.txt").unlink()
    elif change == "unknown_root":
        ready = export.json_file(run / export.READY)
        ready["files"]["secrets.env"] = "a" * 64
        write(run / export.READY, ready)
    elif change == "wrong_sha":
        (run / "history_config.resolved.yaml").write_text("changed")
    elif change == "wrong_encoder":
        snapshot.prov["encoder"]["state_key"] = "encoder"
        write(run / "motion_provenance.json", snapshot.prov)
        refresh_ready(run, export)
    elif change == "wrong_library":
        snapshot.history["motion"]["store_path"] = f"v1-store/datasets/{export.DATASET}/motion"
        path = write(run / "history_config.resolved.yaml", snapshot.history)
        write(run / "history_config.resolved.sha256", digest(path))
        snapshot.prov["resolved_sha256"] = digest(path)
        write(run / "motion_provenance.json", snapshot.prov)
        refresh_ready(run, export)
    elif change == "missing_evidence":
        (run / "motion_evidence/G3.json").unlink()
        refresh_ready(run, export)
    elif change in ("wrong_dependency", "missing_dependency"):
        path = run / "encoder_dependencies.json"
        deps = export.json_file(path)
        if change == "wrong_dependency":
            deps["files"][0]["source_path"] = "/tmp/unknown.pt"
        else:
            deps["files"].pop()
        write(path, deps)
        refresh_ready(run, export)
    elif change == "wrong_online_npz":
        path = run / "online_restore_inputs.json"
        online = export.json_file(path)
        online["payload"]["sha256"] = "0" * 64
        write(path, online)
        refresh_ready(run, export)
    else:
        path = run / "motion_evidence/G3.json"
        body = export.json_file(path)
        body["samples_sha256"] = "0" * 64
        write(path, body)
        snapshot.prov["stage2_compat"]["evidence"]["G3"]["sha256"] = digest(path)
        write(run / "motion_provenance.json", snapshot.prov)
        refresh_ready(run, export)
    with pytest.raises((ValueError, FileNotFoundError)):
        export.snapshot_sources(run, snapshot.registered)


@pytest.mark.parametrize("path", ["/etc/passwd", "../x", "a/../x", "a//b", "./x", "a\\b", ""])
def test_relative_paths_fail_closed(export, path):
    with pytest.raises(ValueError):
        export.relative(path)


def test_snapshot_symlink_and_unknown_evidence_rejected(export, snapshot):
    path = snapshot.run / "motion_evidence/G0.json"
    target = path.with_suffix(".saved")
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="符号链接"):
        export.snapshot_sources(snapshot.run, snapshot.registered)


def test_committed_steps_and_actual_members(export, snapshot):
    add_step(snapshot.run, 5000, committed=False)
    with pytest.raises(ValueError, match="尚未提交"):
        export.step_sources(snapshot.run, 5000)
    add_step(snapshot.run, 5000)
    files = export.step_sources(snapshot.run, 5000)
    assert set(files) == {"5000/_CHECKPOINT_METADATA", "5000/params/_METADATA", "5000/params/array.bin"}
    with pytest.raises(ValueError):
        export.step_sources(snapshot.run, 60001)


def test_two_encoder_sources_have_exact_checkpoint_sets(export, tmp_path):
    for run, names in ((export.ENC_RUN, [f"checkpoint_epoch_{n}.pt" for n in range(5, 106, 5)]),
                       (export.SCALE_RUN, ["checkpoint_epoch_4.pt", "checkpoint_epoch_8.pt", "checkpoint_final.pt"])):
        for name in [*names, "config.yaml", "train_metrics_epoch.jsonl"]:
            write(tmp_path / "runs" / run / name, "数据")
    write(tmp_path / "runs" / export.ENC_RUN / "motion_ablation/epoch105/summary.json", {})
    sources, ckpts = export.encoder_sources(tmp_path)
    assert len(ckpts) == 24
    assert f"{export.ENC_RUN}/config.yaml" in sources and f"{export.SCALE_RUN}/config.yaml" in sources
    write(tmp_path / "runs" / export.ENC_RUN / "checkpoint_final.pt", "伪造完成")
    with pytest.raises(ValueError, match="集合不符"):
        export.encoder_sources(tmp_path)


def schema_tensor_fixture(export):
    """独立生成185项参考，故障只修改待检权重。"""
    import copy
    import torch
    expected = {}
    for prefix, count in (("encoder", 65), ("rgb_decoder", 40), ("xyz_decoder", 40), ("state_decoder", 40)):
        buffers = 0
        if prefix != "state_decoder":
            expected[prefix + ".latents_mean"] = torch.ones(1, 16, 1, 1)
            expected[prefix + ".latents_std"] = torch.ones(1, 16, 1, 1)
            buffers = 2
        for i in range(count - buffers):
            expected[f"{prefix}.leaf{i}"] = torch.ones(2)
    cfg = {"run_name": export.ENC_RUN}
    fingerprint = {"data_metadata": {"identity": "fixture"}}
    c = {"arch": export.ARCH, "weight_type": "ema", "config": cfg,
         "fingerprint": fingerprint, "fingerprint_sha256": "expected-fingerprint",
         "epoch": 105, "global_step": 7770, "model": copy.deepcopy(expected),
         "model_live": copy.deepcopy(expected), "optimizer": {"state": {}, "param_groups": []}}
    return c, expected, cfg


@pytest.mark.parametrize("optimize", [0, 1, 2])
@pytest.mark.parametrize("fault", [None, "shape", "dtype", "live", "live_dtype", "live_nonfinite", "missing", "fingerprint", "config", "nonfinite"])
def test_real_schema_validation_rejects_corrupt_tensor_contract(export, tmp_path, monkeypatch, fault, optimize):
    import copy
    import io
    import torch
    # 使用真实Tensor和独立模型参考，隔离大模型配置依赖，覆盖各失败分支。
    c, expected, cfg = schema_tensor_fixture(export)
    if fault == "shape":
        c["model"]["encoder.leaf0"] = torch.ones(3)
    elif fault == "dtype":
        c["model"]["encoder.leaf0"] = torch.ones(2, dtype=torch.float64)
    elif fault == "live":
        c["model_live"]["encoder.leaf0"] = torch.ones(3)
    elif fault == "live_dtype":
        c["model_live"]["encoder.leaf0"] = torch.ones(2, dtype=torch.float64)
    elif fault == "live_nonfinite":
        c["model_live"]["encoder.leaf0"][0] = float("nan")
    elif fault == "missing":
        del c["model"]["xyz_decoder.leaf0"]
    elif fault == "fingerprint":
        c["fingerprint_sha256"] = "wrong"
    elif fault == "nonfinite":
        c["model"]["encoder.leaf0"][0] = float("nan")
    source = tmp_path / "checkpoint_epoch_105.pt"
    write(tmp_path / "config.yaml", cfg if fault != "config" else {"run_name": "unknown"})
    factory = types.ModuleType("motion_jepa.models.factory")
    factory.build_stage2_model_from_cfg = lambda *args, **kwargs: types.SimpleNamespace(state_dict=lambda: expected)
    model_module = types.ModuleType("motion_jepa.models.stage2_model")
    model_module.load_stage2_checkpoint = lambda *args, **kwargs: None
    model_module.fingerprint_sha256 = lambda value: "expected-fingerprint"
    package = types.ModuleType("motion_jepa.models")
    package.factory = factory
    for name, module in (("motion_jepa", types.ModuleType("motion_jepa")),
                         ("motion_jepa.models", package), ("motion_jepa.models.factory", factory),
                         ("motion_jepa.models.stage2_model", model_module)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: c)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps([
        {"path": str(source), "run_name": export.ENC_RUN, "name": source.name}])))
    monkeypatch.setenv("PYTHONOPTIMIZE", str(optimize))
    code = compile(export.MJ_SCHEMA_CODE, "stage2-schema", "exec", optimize=optimize)
    if fault:
        with pytest.raises(ValueError):
            exec(code, {})
    else:
        exec(code, {})


def test_existing_stage_cannot_be_overwritten(export, tmp_path):
    src = write(tmp_path / "source/a.json", {"value": 1})
    stage = tmp_path / "stage"
    stage.mkdir()
    result = export.freeze_sources({"a.json": src}, stage)
    assert result["a.json"]["sha256"] == digest(stage / "a.json")
    write(src, {"value": 2})
    with pytest.raises(ValueError, match="已有stage"):
        export.freeze_sources({"a.json": src}, stage)


def test_anonymous_env_never_uses_tokens(export, tmp_path, monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        monkeypatch.setenv(name, "fixture-secret")
    bucket = export.AnonymousBucket(export.VLA_BUCKET, "", cache_root=tmp_path)
    env = bucket._env()
    assert not any(name in env for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN_PATH"))
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""


class FakeBucket:
    def __init__(self):
        self.data = {}
        self.uploads = []
        self.downloads = []
        self.corrupt = False

    def upload(self, path, name):
        self.uploads.append(name)
        self.data[name] = Path(path).read_bytes()
        return types.SimpleNamespace(ok=True)

    def download(self, name, path):
        self.downloads.append(name)
        Path(path).write_bytes(b"broken" if self.corrupt else self.data[name])
        return types.SimpleNamespace(ok=True)

    def list_files(self):
        return {k: len(v) for k, v in self.data.items()}

    def entry(self):
        return {"private": False}


def publisher(export, tmp_path):
    p = export.Publisher.__new__(export.Publisher)
    p.root = tmp_path
    p.bucket_id = export.VLA_BUCKET
    p.identity = {"kind": "stage2-vla", "run_name": export.RUN, "bucket": export.VLA_BUCKET,
                  "root_manifest_sha256": "1" * 64}
    p.marker_root = tmp_path / "logs/uploaded"
    p.marker_root.mkdir(parents=True)
    p.verify = tmp_path / "verify"
    p.verify.mkdir()
    p.bucket = p.anon = FakeBucket()
    return p


def test_incremental_final_share_markers_and_full_readback(export, tmp_path):
    p = publisher(export, tmp_path)
    stage = tmp_path / "stage"
    path = write(stage / "5000/params/array.bin", "参数")
    files = {"5000/params/array.bin": export.file_info(path)}
    p.transfer("5000", stage, files)
    assert (tmp_path / "logs/uploaded/5000.done").is_file()
    assert p.bucket.uploads == ["5000/params/array.bin"]
    p.transfer("5000", stage, files)
    assert len(p.bucket.uploads) == 1
    p.verify_all(stage, files)
    assert len(p.bucket.downloads) == 2
    p.bucket.corrupt = True
    with pytest.raises(ValueError, match="匿名SHA"):
        p.verify_all(stage, files)


def test_upload_marker_identity_and_extra_remote_rejected(export, tmp_path):
    p = publisher(export, tmp_path)
    stage = tmp_path / "stage"
    path = write(stage / "a.json", {})
    files = {"a.json": export.file_info(path)}
    p.transfer("root", stage, files)
    p.bucket.data["unknown.json"] = b"extra"
    with pytest.raises(ValueError, match="集合"):
        p.verify_all(stage, files)
    p.identity["run_name"] = "wrong-run"
    with pytest.raises(ValueError, match="标记身份"):
        p.transfer("root", stage, files)


def test_smoke_gate_bound_to_manifest(export, tmp_path):
    with pytest.raises(ValueError, match="SMOKE=1"):
        export.smoke_gate(tmp_path, "final", "a" * 64, False)
    path, body = export.smoke_gate(tmp_path, "final", "a" * 64, True)
    write(path, body)
    export.smoke_gate(tmp_path, "final", "a" * 64, False)
    with pytest.raises(ValueError):
        export.smoke_gate(tmp_path, "final", "b" * 64, False)


def test_secret_is_never_allowlisted(export, tmp_path):
    path = write(tmp_path / "record.json", {"token": "hf_" + "A" * 32})
    allow = write(tmp_path / "allow.json", {"entries": [{"pattern": "P5", "path_glob": "*", "decision": "accept"}]})
    with pytest.raises(ValueError, match="疑似凭据"):
        export.scan({"record.json": path}, allow)


def test_local_smoke_and_final_flow(export, snapshot, tmp_path, monkeypatch, capsys):
    e = export
    monkeypatch.setattr(e, "REPO", snapshot.repo)
    monkeypatch.setattr(e, "validate_store", lambda *args: None)
    # 此测试隔离S2数据层，实际生产入口始终调用validate_store。
    for step in e.STEPS:
        add_step(snapshot.run, step)
    log = write(snapshot.repo / "v1-store/logs" / f"{e.RUN}.log", f"RUN={e.RUN}\nEXIT_CODE=0\n")
    target = snapshot.repo / "scripts/dataset/hf_export"
    write(target / "stage2_vla_bucket_README.md", "交付说明")
    write(target / "hygiene_allowlist_stage2_vla.json", {"entries": []})
    root = tmp_path / "export"
    (root / "stage").mkdir(parents=True)
    monkeypatch.setattr(e.io, "load_hf_token", lambda *a: pytest.fail("SMOKE触碰凭据"))
    e.run_vla("final", True, root)
    assert "SMOKE=PASS" in capsys.readouterr().out
    assert not (root / "logs/uploaded").exists()
    body = e.json_file(root / "stage/manifests/step-60000.json")
    assert e.READY in body["files"]
    assert any(k.startswith("encoder_dependencies/") for k in body["files"])
    assert "60000/params/array.bin" in body["files"]
    p = publisher(e, root)
    monkeypatch.setattr(e, "Publisher", lambda *a, **kw: p)
    e.run_vla("final", False, root)
    assert "CKPT_UPLOAD=PASS steps=16" in capsys.readouterr().out
    assert set(p.bucket.data) == set(e.json_file(root / "stage/delivery_manifest.json")["files"]) | {"delivery_manifest.json"}
    assert len(list((root / "logs/uploaded").glob("*.done"))) == 18


def test_incremental_contract_failure_never_touches_network(export, snapshot, tmp_path, monkeypatch):
    e = export
    monkeypatch.setattr(e, "REPO", snapshot.repo)
    calls = []
    monkeypatch.setattr(e, "Publisher", lambda *a, **kw: calls.append("network"))
    monkeypatch.setattr(e.io, "load_hf_token", lambda *a: calls.append("credential"))
    def fail_store(*args):
        raise ValueError("stage2库未验证")
    monkeypatch.setattr(e, "validate_store", fail_store)
    root = tmp_path / "export"
    (root / "stage").mkdir(parents=True)
    with pytest.raises(ValueError, match="未验证"):
        e.run_vla("incremental", False, root)
    assert calls == []
    assert not (root / "smoke-incremental.json").exists()


def test_incremental_local_smoke_happens_before_publisher(export, snapshot, tmp_path, monkeypatch):
    e = export
    monkeypatch.setattr(e, "REPO", snapshot.repo)
    monkeypatch.setattr(e, "validate_store", lambda *a: None)
    target = snapshot.repo / "scripts/dataset/hf_export"
    write(target / "stage2_vla_bucket_README.md", "交付说明")
    write(target / "hygiene_allowlist_stage2_vla.json", {"entries": []})
    root = tmp_path / "export"
    (root / "stage").mkdir(parents=True)
    class ReachedNetwork(Exception):
        pass
    def network(*args, **kwargs):
        assert (root / "smoke-incremental.json").is_file()
        assert (root / "stage/manifests/root.json").is_file()
        assert not (root / "logs/uploaded/root.done").exists()
        raise ReachedNetwork
    monkeypatch.setattr(e, "Publisher", network)
    with pytest.raises(ReachedNetwork):
        e.run_vla("incremental", False, root)


@pytest.mark.parametrize("fault", ["tokenizer", "array_sha", "array_shape", "hidden_sources", "batch_lengths"])
def test_online_delivery_inputs_reject_incomplete_portable_contract(export, snapshot, fault):
    record = export.json_file(snapshot.run / "online_restore_inputs.json")
    if fault == "tokenizer":
        record["tokenizer"]["sha256"] = "0" * 64
    elif fault == "array_sha":
        record["arrays"]["states"]["sha256"] = "0" * 64
    elif fault == "array_shape":
        record["arrays"]["images"]["shape"][0] -= 1
    elif fault == "hidden_sources":
        record["sources"] = []
    else:
        record["exec_batch_lengths"] = [32, 33]
    with pytest.raises(ValueError):
        export.check_online_inputs(record, snapshot.run / "online_restore_inputs.npz")


@pytest.mark.parametrize("keep_original", [True, False])
def test_ready_rejects_aliased_source_pin_even_with_identical_bytes(export, snapshot, keep_original):
    run = snapshot.run
    original = run / "motion_evidence/SOURCE_PIN.json"
    extra = run / "motion_evidence/extra.json"
    extra.write_bytes(original.read_bytes())
    snapshot.prov["stage2_compat"]["evidence"]["SOURCE_PIN"]["path"] = "motion_evidence/extra.json"
    write(run / "motion_provenance.json", snapshot.prov)
    if not keep_original:
        original.unlink()
    refresh_ready(run, export)
    ready = export.json_file(run / export.READY)
    assert len(ready["files"]) == (13 if keep_original else 12)
    with pytest.raises(ValueError, match="精确等于12"):
        export.snapshot_sources(run, snapshot.registered)


def test_ready_source_pin_reference_must_match_fixed_file(export, snapshot):
    # 文件集合仍是精确12项；即使引用别的已列举证据且SHA自洽，也不能替换SOURCE_PIN路径。
    snapshot.prov["stage2_compat"]["evidence"]["SOURCE_PIN"] = dict(
        snapshot.prov["stage2_compat"]["evidence"]["G0"])
    write(snapshot.run / "motion_provenance.json", snapshot.prov)
    refresh_ready(snapshot.run, export)
    with pytest.raises(ValueError, match="固定路径"):
        export.snapshot_sources(snapshot.run, snapshot.registered)


@pytest.mark.parametrize("optimize", [1, 2])
@pytest.mark.parametrize("fault", [None, "live_dtype", "live_nonfinite"])
def test_checkpoint_subprocess_enforces_contract_with_pythonoptimize(export, tmp_path, monkeypatch, optimize, fault):
    import os
    import subprocess
    import torch
    c, _, cfg = schema_tensor_fixture(export)
    if fault == "live_dtype":
        c["model_live"]["encoder.leaf0"] = torch.ones(2, dtype=torch.float64)
    elif fault == "live_nonfinite":
        c["model_live"]["encoder.leaf0"][0] = float("nan")
    checkpoint = tmp_path / "checkpoint_epoch_105.pt"
    torch.save(c, checkpoint)
    write(tmp_path / "config.yaml", cfg)
    monkeypatch.setenv("PYTHONOPTIMIZE", str(optimize))
    # 子进程仍运行真实schema及torch读盘，仅隔离MJ大模型工厂，使用独立的main树作结构参考。
    setup = r'''
import os, sys, types, torch
if sys.flags.optimize != int(os.environ["PYTHONOPTIMIZE"]):
    raise RuntimeError("优化级别未传入真实子进程")
reference = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["model"]
factory = types.ModuleType("motion_jepa.models.factory")
factory.build_stage2_model_from_cfg = lambda *a, **kw: types.SimpleNamespace(state_dict=lambda: reference)
loader = types.ModuleType("motion_jepa.models.stage2_model")
loader.load_stage2_checkpoint = lambda *a, **kw: None
loader.fingerprint_sha256 = lambda value: "expected-fingerprint"
package = types.ModuleType("motion_jepa.models")
package.factory = factory
sys.modules.update({"motion_jepa": types.ModuleType("motion_jepa"), "motion_jepa.models": package,
                    "motion_jepa.models.factory": factory, "motion_jepa.models.stage2_model": loader})
'''
    actual_run = subprocess.run
    def run_child(argv, **kwargs):
        assert kwargs["env"]["PYTHONOPTIMIZE"] == str(optimize)
        return actual_run([sys.executable, "-c", setup + "\n" + argv[-1], str(checkpoint)], **kwargs)
    monkeypatch.setattr(export.subprocess, "run", run_child)
    items = [{"path": str(checkpoint), "run_name": export.ENC_RUN, "name": checkpoint.name}]
    if fault:
        with pytest.raises(ValueError, match="schema校验失败"):
            export.checkpoint_schemas(tmp_path, items)
    else:
        result = export.checkpoint_schemas(tmp_path, items)
        assert result[f"{export.ENC_RUN}/{checkpoint.name}"]["has_model_live"] is True
