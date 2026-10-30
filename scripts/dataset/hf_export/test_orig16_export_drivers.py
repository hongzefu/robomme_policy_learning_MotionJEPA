"""官方16×100导出CPU夹具：完整source、双帧表、新证据、12步与匿名回读失败分支。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import types
import zipfile

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
COMMON = HERE / "_stage2_orig16_vla_ckpt_export_common.sh"
DRIVERS = [HERE / name for name in (
    "run_orig100_trainset_export.sh", "run_orig16_stage2_ckpt_inc.sh",
    "run_orig16_ctx32x4x4_split_60k_ckpt_export.sh", "run_orig16_modul32x4x4_split_60k_ckpt_export.sh",
    "run_orig16_ctx32x4x4_concat_60k_ckpt_export.sh", "run_orig16_modul32x4x4_concat_60k_ckpt_export.sh")]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict):
        data = json.dumps(data)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


@pytest.fixture
def exporter(monkeypatch):
    module = types.ModuleType("orig16_export_test")
    code = COMMON.read_text().split("<<'O16_PY'\n", 1)[1].rsplit("\nO16_PY", 1)[0]
    monkeypatch.setenv("O16_REPO", str(REPO))
    exec(compile(code, str(COMMON), "exec"), module.__dict__)
    monkeypatch.setattr(module.io, "load_hf_token", lambda *a: pytest.fail("离线测试不得读取凭据"))
    monkeypatch.setattr(module.io.HFBucket, "_run", lambda *a: pytest.fail("离线测试不得联网"))
    return module


@pytest.fixture
def checkpoint_consumer():
    path = REPO / "scripts/training/check_stage2_checkpoint.py"
    spec = importlib.util.spec_from_file_location("orig16_checkpoint_consumer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_library(root):
    lib = root / "lib"
    for i in range(3):
        write(lib / f"source/data/{i}.pkl", bytes([i]) * 16)
    # 第三份feature未被spot点名，必须仍完整进入tar。
    for i in range(3):
        write(lib / f"source/features/episode_{i}/token_emb_0.npy", bytes([i]) * 32)
    write(lib / "source/meta/provenance.json", {"fixture": True})
    write(lib / "meta/episode_manifest.json", {"fixture": True})
    write(lib / "meta/input_manifest.json", {"fixture": True})
    for i, grid in enumerate(("4x4", "8x8")):
        sub = f"framesamp-{grid}"
        tables = {}
        for name in (f"pos_emb_{grid}", "state_emb"):
            data = f"{grid}-{name}".encode()
            write(lib / sub / f"{name}.f32.bin", data)
            tables[name] = {"sha256": sha(data)}
        write(lib / sub / f"image_emb_{grid}/part_000.bf16.bin", grid.encode())
        write(lib / sub / "meta/store_meta.json", {
            "status": "verified", "layout": f"framesamp-{grid}-v1", "num_exec_samples": 3,
            "tables": tables, "source_spot_sha256": {"entries": [{"relpath": f"features/episode_{i}/token_emb_0.npy"}]}})
    evidence = {}
    for role in ("G0", "G2", "G3", "wan_cache_import", "SOURCE_PIN"):
        path = write(root / "validation" / f"{role}.json", {"role": role})
        evidence[role] = {"path": str(path), "sha256": sha(path.read_bytes())}
    token = bytes(4096)
    write(lib / "motion-stage2-1024/motion_token.f32.bin", token)
    write(lib / "motion-stage2-1024/meta/store_meta.json", {
        "status": "verified", "layout": "motion-stage2-1x1024-grid32-demopad17-v1", "num_rows": 1,
        "tables": {"motion_token": {"relpath": "motion_token.f32.bin", "sha256": sha(token)}},
        "provenance": {"stage2_compat": {"kind": "published-cache-with-local-fill", "evidence": evidence}}})
    for sub in ("wan-latents", "motion-tokens-stage2"):
        write(lib / sub / "segment.bin", bytes(32))
        write(lib / sub / "segment.metadata.json", {"fixture": True})
    norm = write(root / "norm_stats.json", {"norm_stats": {}})
    return lib, norm


def env_for_scripts(**extra):
    env = dict(os.environ)
    env.update(UV_PROJECT_ENVIRONMENT=str(Path(sys.executable).parent.parent),
               PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
               UV_CACHE_DIR="/scratch/hongze/.cache/uv", UV_OFFLINE="1", HF_HUB_OFFLINE="1")
    env.update(extra)
    return env


def run_pack(lib, norm, export):
    return subprocess.run(["uv", "run", "--no-sync", "python", str(HERE / "pack_and_hash.py"),
        "--layout", "orig100-stage2-v1", "--lib", str(lib), "--norm-stats", str(norm),
        "--stage-root", str(export), "--bucket", "hf://buckets/test/fixture", "--workers", "2"],
        cwd=REPO, env=env_for_scripts(), capture_output=True, text=True, timeout=60)


def test_full_pack_and_readback(tmp_path, exporter):
    lib, norm = fixture_library(tmp_path)
    root = tmp_path / "export"
    result = run_pack(lib, norm, root)
    assert result.returncode == 0, result.stdout + result.stderr
    stage = root / "stage"
    restored = tmp_path / "restored"
    seen = set()
    for archive in stage.rglob("*.tar"):
        with tarfile.open(archive) as tar:
            for item in tar:
                if not item.isfile():
                    continue
                prefix = "source/" if archive.parent.name in ("data_tars", "features_tars") else ""
                rel = prefix + item.name
                actual = tar.extractfile(item).read()
                assert actual == (lib / rel).read_bytes()
                write(restored / rel, actual)
                seen.add(rel)
    expected = {p.relative_to(lib).as_posix() for sub in ("source/data", "source/features", "wan-latents", "motion-tokens-stage2")
                for p in (lib / sub).rglob("*") if p.is_file()}
    assert seen == expected
    assert "source/features/episode_2/token_emb_0.npy" in seen
    for grid in ("4x4", "8x8"):
        assert (stage / f"framesamp-{grid}/image_emb_{grid}/part_000.bf16.bin").read_bytes() == grid.encode()
    for role in exporter.EVIDENCE_FILES:
        assert (stage / f"evidence/{role}.json").read_bytes() == (tmp_path / f"validation/{role}.json").read_bytes()
    for line in (stage / "checksums/sha256-plain.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha((stage / name).read_bytes()) == digest
    # 同清单续跑不会改变任一已打包对象。
    before = {p.relative_to(stage).as_posix(): sha(p.read_bytes()) for p in stage.rglob("*") if p.is_file()}
    rerun = run_pack(lib, norm, root)
    assert rerun.returncode == 0, rerun.stderr
    assert before == {p.relative_to(stage).as_posix(): sha(p.read_bytes()) for p in stage.rglob("*") if p.is_file()}


@pytest.mark.parametrize("corrupt_rel", [
    "source/data_tars/data-00000.tar", "framesamp-4x4/image_emb_4x4/part_000.bf16.bin"])
def test_trainset_resume_rejects_same_size_corruption(tmp_path, exporter, monkeypatch, corrupt_rel):
    lib, norm = fixture_library(tmp_path)
    root = tmp_path / "export"
    result = run_pack(lib, norm, root)
    assert result.returncode == 0, result.stdout + result.stderr
    target = root / "stage" / corrupt_rel
    data = bytearray(target.read_bytes())
    # 改tar首个成员的数据字节，仍是有效tar且尺寸不变；表文件同样原位改一个字节。
    offset = 512 if target.suffix == ".tar" else 0
    data[offset] ^= 1
    target.write_bytes(data)
    for key, value in env_for_scripts(O16_LIB=str(lib), O16_NORM_STATS=str(norm)).items():
        monkeypatch.setenv(key, value)
    # 真实重跑pack会因尺寸相同而跳过；正式驱动必须在发布器读取凭据之前拒绝。
    with pytest.raises(ValueError, match="打包清单SHA或字节数不符"):
        exporter.run_trainset(False, root)
    assert not (root / "stage/delivery_manifest.json").exists()


@pytest.mark.parametrize("failure", [None, "extra", "missing", "checksum", "bytes"])
def test_trainset_uses_pack_manifest_object_set(tmp_path, exporter, failure):
    lib, norm = fixture_library(tmp_path)
    root = tmp_path / "export"
    result = run_pack(lib, norm, root)
    assert result.returncode == 0, result.stdout + result.stderr
    stage = root / "stage"
    write(stage / "README.md", "夹具")
    if failure == "extra":
        write(stage / "unexpected.bin", "未列举对象")
    elif failure == "missing":
        (stage / "source/data_tars/data-00000.tar").unlink()
    elif failure == "checksum":
        checksum = stage / "checksums/sha256-shards.txt"
        checksum.write_text("f" * 64 + checksum.read_text()[64:])
    elif failure == "bytes":
        manifest = stage / "manifest/upload_manifest.json"
        packed = json.loads(manifest.read_text())
        packed["shards"][0]["bytes"] += 1
        write(manifest, packed)
    if failure:
        with pytest.raises(ValueError, match="对象集合|校验和|SHA或字节数"):
            exporter.packed_trainset_files(stage)
    else:
        files = exporter.packed_trainset_files(stage)
        assert files == {p.relative_to(stage).as_posix(): exporter.file_info(p) for p in stage.rglob("*") if p.is_file()}


@pytest.mark.parametrize("failure", ["layout", "count", "lock", "evidence", "sha"])
def test_pack_rejects_invalid_library(tmp_path, failure):
    lib, norm = fixture_library(tmp_path)
    path = lib / "framesamp-4x4/meta/store_meta.json"
    meta = json.loads(path.read_text())
    if failure == "layout":
        meta["layout"] = "framesamp-8x8-v1"
    elif failure == "count":
        meta["num_exec_samples"] += 1
    elif failure == "lock":
        write(path.parent / "pack.lock", "锁")
    elif failure == "evidence":
        path = lib / "motion-stage2-1024/meta/store_meta.json"
        meta = json.loads(path.read_text())
        meta["provenance"]["stage2_compat"]["evidence"].pop("wan_cache_import")
    else:
        meta["tables"]["state_emb"]["sha256"] = "f" * 64
    write(path, meta)
    result = run_pack(lib, norm, tmp_path / "export")
    assert result.returncode != 0


def test_trainset_shell_smoke_is_offline(tmp_path):
    lib, norm = fixture_library(tmp_path)
    result = subprocess.run(["bash", str(DRIVERS[0])], cwd=REPO,
        env=env_for_scripts(SMOKE="1", O16_LIB=str(lib), O16_NORM_STATS=str(norm), O16_EXPORT_ROOT=str(tmp_path / "export")),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SMOKE=PASS kind=trainset" in result.stdout and "network=0" in result.stdout


def test_formal_placeholder_fails_before_credentials(exporter, monkeypatch):
    monkeypatch.setattr(exporter, "DATASET", "16task-orig100-1600ep-__FILL_AFTER_BUILD__")
    monkeypatch.setenv("O16_MODE", "trainset")
    monkeypatch.setenv("SMOKE", "0")
    with pytest.raises(ValueError, match="__FILL_AFTER_BUILD__"):
        exporter.main()


@pytest.mark.parametrize("run_index", [0, 1])
def test_bucket_pair_is_fixed(exporter, monkeypatch, run_index):
    runs = list(exporter.RUNS)
    monkeypatch.setattr(exporter, "DATASET", "16task-orig100-1600ep-12345678")
    monkeypatch.setattr(exporter, "RUN", runs[run_index])
    monkeypatch.setenv("O16_MODE", "incremental")
    monkeypatch.setenv("BUCKET", exporter.RUNS[runs[1 - run_index]][0])
    with pytest.raises(ValueError, match="固定配对"):
        exporter.main()


def make_run(tmp_path, exporter, monkeypatch, index=0, run_name=None):
    base = list(exporter.RUNS)[index]
    run_name = run_name or base
    repo = tmp_path / "repo"
    monkeypatch.setattr(exporter, "REPO", repo)
    monkeypatch.setattr(exporter, "RUN", run_name)
    monkeypatch.setattr(exporter, "DATASET", "16task-orig100-1600ep-12345678")
    run = repo / "v1-store/train-runs/mme_vla_suite_orig16_b128_60k" / run_name
    reg = repo / "v1-store/external/motionjepa" / exporter.ENC_RUN
    cfg = write(reg / "config.yaml", "fixture: true\n")
    ckpt = reg / "checkpoint_epoch_105.pt"
    with zipfile.ZipFile(ckpt, "w") as archive:
        archive.writestr("fixture/data.pkl", b"\x80\x04N.")
    enc_sha = sha(ckpt.read_bytes())
    readout = ({"readout": "stage2_pool4_split_v1", "num_tokens": 4, "dim": 256}
               if exporter.RUNS[base][4] == "4x256"
               else {"readout": "stage2_pool4_concat_v1", "num_tokens": 1, "dim": 1024, "budget": 96})
    history = {"motion": {"store_path": f"v1-store/datasets/{exporter.DATASET}/motion-stage2-1024",
                           **readout, "checkpoint_sha256": enc_sha}}
    name = exporter.history_name(base)
    write(repo / "src/mme_vla_suite/models/config/robomme" / name, history)
    write(run / "history_config.txt", name)
    resolved = write(run / "history_config.resolved.yaml", history)
    resolved_sha = sha(resolved.read_bytes())
    write(run / "history_config.resolved.sha256", resolved_sha)
    source_pin = {"fixture": True}
    g0 = write(run / exporter.EVIDENCE_FILES["G0"], {"gate": "G0", "status": "PASS"})
    receipt_record = {"copy": 1, "fill": 1, "manifest_sha256": "c" * 64,
                      "latents": {"metadata_sha256": "d" * 64}, "vae": {"fixture": True},
                      "preprocess": {"frame_size": 256, "input_dtype": "uint8", "input_layout": "THWC", "window_frames": 33},
                      "numeric_modes": {"copy": {"dtype": "float32", "tf32": "on"},
                                        "fill": {"dtype": "float32", "tf32": "off"}}}
    receipt = write(run / exporter.EVIDENCE_FILES["wan_cache_import"], receipt_record)
    # 缓存G2形制来自check_import：绑定输入和receipt，不含encoder字段；完整S2语义不在此夹具中重建。
    write(run / exporter.EVIDENCE_FILES["G2"], {
        "schema": "stage2-validation-gate-v1", "gate": "G2", "status": "PASS", "kind": "published-cache-with-local-fill",
        "samples_sha256": sha(g0.read_bytes()), "cache_import_receipt_sha256": sha(receipt.read_bytes()),
        "manifest_sha256": receipt_record["manifest_sha256"], "latent_metadata_sha256": receipt_record["latents"]["metadata_sha256"],
        "vae": receipt_record["vae"], "preprocess": receipt_record["preprocess"], "numeric_modes": receipt_record["numeric_modes"],
        "output_numerics": {"status": "NOT_REQUIRED", "reason": "user_decision_same_input"},
        "metrics": {"windows": 64, "demo_tail": 16, "copy": 32, "demo_fill": 16, "exec_fill": 16,
                    "sha_mismatches": 0, "frame_mismatches": 0, "shape_mismatches": 0, "nonfinite": 0}})
    write(run / exporter.EVIDENCE_FILES["G3"], {"gate": "G3", "status": "PASS", "checkpoint_sha256": enc_sha,
                                               "samples_sha256": sha(g0.read_bytes())})
    write(run / exporter.EVIDENCE_FILES["SOURCE_PIN"], source_pin)
    evidence = {role: {"path": rel, "sha256": sha((run / rel).read_bytes())} for role, rel in exporter.EVIDENCE_FILES.items()}
    encoder = {"run_name": exporter.ENC_RUN, "checkpoint_name": "checkpoint_epoch_105.pt", "epoch": 105,
               "global_step": 7770, "arch": exporter.ARCH, "state_key": "model", "weight_type": "ema",
               "source_times": [0, 3, 5, 8], "output_shape": [1, 1024], "checkpoint_sha256": enc_sha,
               "config_sha256": sha(cfg.read_bytes()), "fingerprint_sha256": "a" * 64}
    prov = {"motion_enabled": True, "resolved_sha256": resolved_sha, "encoder": encoder,
            "source_pin": source_pin, "encoder_state_sha256": {"fixture": "b" * 64}, "numerics": {"fixture": True},
            "vae": {"fixture": True}, "stage2_compat": {"kind": "published-cache-with-local-fill", "evidence": evidence,
                 "cache_import_receipt_sha256": evidence["wan_cache_import"]["sha256"]}}
    write(run / "motion_provenance.json", prov)
    deps = {"schema": "stage2-encoder-dependencies-v1", "encoder": encoder, "source_pin": source_pin, "vae": prov["vae"],
            "files": [{"role": role, "source_path": str(path), "relative_path": f"encoder_dependencies/{exporter.ENC_RUN}/{path.name}",
                       **exporter.file_info(path)} for role, path in (("encoder_checkpoint", ckpt), ("encoder_config", cfg))]}
    write(run / "encoder_dependencies.json", deps)
    arrays = {"images": np.zeros((82, 256, 256, 3), dtype=np.uint8), "states": np.zeros((82, 8), dtype=np.float32),
              "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8)}
    np.savez_compressed(run / "online_restore_inputs.npz", **arrays)
    online = {"schema": "stage2-online-restore-inputs-v1", "exec_start_idx": 17, "exec_batch_lengths": [1, 16, 16, 16, 16],
              "prompt": "夹具", "sources": [{"path": "fixture", "sha256": "a" * 64}],
              "tokenizer": {"uri": exporter.TOKENIZER_URI, "sha256": exporter.TOKENIZER_SHA},
              "payload": {"path": "online_restore_inputs.npz", "sha256": sha((run / "online_restore_inputs.npz").read_bytes())},
              "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha(value.tobytes())}
                         for key, value in arrays.items()}}
    write(run / "online_restore_inputs.json", online)
    write(run / exporter.READY, {"schema": "stage2-motion-snapshot-v1", "files": {
        rel: sha((run / rel).read_bytes()) for rel in exporter.ROOT_FILES | set(exporter.EVIDENCE_FILES.values())}})
    for step in exporter.STEPS:
        write(run / str(step) / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": 1})
        write(run / str(step) / "params/_METADATA", {"fixture": True})
        write(run / str(step) / "params/part.bin", bytes(8))
        write(run / str(step) / "assets/robomme/norm_stats.json", {"norm_stats": {"state": {"mean": [0.0]}}})
    write(repo / "v1-store/logs" / f"{run_name}.log", f"RUN={run_name}\nEXIT_CODE=0\n")
    return run, reg


def make_producer_delivery(tmp_path, exporter, monkeypatch, index=0):
    """按真实producer函数生成根及59999清单，不替换快照或文件核验函数。"""
    run, reg = make_run(tmp_path, exporter, monkeypatch, index)
    stage = tmp_path / "export/stage"
    stage.mkdir(parents=True)
    sources, _, _ = exporter.snapshot_sources(run, reg)
    sources["README.md"] = exporter.readme_path(exporter.RUN)
    identity = {"run_name": exporter.RUN, "bucket": exporter.RUNS[exporter.RUN][0]}
    root_files = exporter.add_manifest(stage, exporter.freeze_sources(sources, stage), "manifests/root.json", **identity)
    step_files = exporter.freeze_sources(exporter.step_sources(run, 59999), stage)
    path = exporter.write_generated(stage, "manifests/step-59999.json",
                                    exporter.manifest({**root_files, **step_files}, step=59999, **identity))
    # 原run只有依赖来源声明；交付恢复必须使用stage中复制的依赖实体。
    assert not (run / "encoder_dependencies").exists()
    return stage, path


@pytest.mark.parametrize("index", [0, 1])
def test_producer_manifest_consumed_by_restore(tmp_path, exporter, checkpoint_consumer, monkeypatch, index):
    stage, path = make_producer_delivery(tmp_path, exporter, monkeypatch, index)
    record = checkpoint_consumer.verify_delivery_manifest(stage, path, stage / "59999")
    assert record["step"] == 59999
    assert set(record["files"]) == {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file() and p != path}


@pytest.mark.parametrize("failure", ["asset_sha", "asset_omitted", "root_omitted", "unlisted_param"])
def test_producer_delivery_preserves_consumer_checks(tmp_path, exporter, checkpoint_consumer, monkeypatch, failure):
    stage, path = make_producer_delivery(tmp_path, exporter, monkeypatch)
    record = json.loads(path.read_text())
    asset = "59999/assets/robomme/norm_stats.json"
    if failure == "asset_sha":
        data = bytearray((stage / asset).read_bytes())
        data[-2] ^= 1
        (stage / asset).write_bytes(data)
        match = "SHA256"
    elif failure == "unlisted_param":
        write(stage / "59999/params/unlisted.bin", bytes(8))
        match = "完整 checkpoint"
    else:
        del record["files"][asset if failure == "asset_omitted" else "online_restore_inputs.npz"]
        write(path, record)
        match = "完整 checkpoint|缺根文件"
    with pytest.raises(ValueError, match=match):
        checkpoint_consumer.verify_delivery_manifest(stage, path, stage / "59999")


@pytest.mark.parametrize("index", [0, 1])
def test_final_checkpoint_smoke_with_new_evidence(tmp_path, exporter, monkeypatch, capsys, index):
    make_run(tmp_path, exporter, monkeypatch, index)
    # S2验证器另有完整来源合同测试，本用例隔离上传阶段，真实核快照、输入数组、依赖和12步。
    monkeypatch.setattr(exporter, "validate_store", lambda *args: None)
    root = tmp_path / "export"
    (root / "stage").mkdir(parents=True)
    exporter.run_checkpoints("final", True, root)
    manifest = json.loads((root / "stage/delivery_manifest.json").read_text())
    assert manifest["steps"] == list(range(5000, 60000, 5000)) + [59999]
    assert "motion_evidence/wan_cache_import.json" in manifest["files"]
    assert all("restore_receipt" not in path for path in manifest["files"])
    assert "SMOKE=PASS kind=checkpoint mode=final steps=12" in capsys.readouterr().out


def test_snapshot_rejects_missing_or_changed_evidence(tmp_path, exporter, monkeypatch):
    run, reg = make_run(tmp_path, exporter, monkeypatch)
    exporter.snapshot_sources(run, reg)
    write(run / exporter.EVIDENCE_FILES["wan_cache_import"], {"tampered": True})
    with pytest.raises(ValueError, match="根快照SHA不符"):
        exporter.snapshot_sources(run, reg)


def test_snapshot_accepts_cache_g2_without_encoder(tmp_path, exporter, monkeypatch):
    run, reg = make_run(tmp_path, exporter, monkeypatch)
    g2 = json.loads((run / exporter.EVIDENCE_FILES["G2"]).read_text())
    assert "checkpoint_sha256" not in g2
    exporter.snapshot_sources(run, reg)


@pytest.mark.parametrize("role,field,match", [
    ("G2", "samples_sha256", "G2没有绑定同一G0"),
    ("G3", "samples_sha256", "G3没有绑定同一G0"),
    ("G3", "checkpoint_sha256", "证据encoder不符: G3")])
@pytest.mark.parametrize("fault", ["missing", "wrong"])
def test_snapshot_keeps_existing_gate_bindings(tmp_path, exporter, monkeypatch, role, field, match, fault):
    run, reg = make_run(tmp_path, exporter, monkeypatch)
    rel = exporter.EVIDENCE_FILES[role]
    record = json.loads((run / rel).read_text())
    if fault == "missing":
        del record[field]
    else:
        record[field] = "f" * 64
    write(run / rel, record)
    # 更新外层实体摘要，确保负例命中既有samples/encoder合同，而非提前撞上ready的SHA检查。
    prov = json.loads((run / "motion_provenance.json").read_text())
    prov["stage2_compat"]["evidence"][role]["sha256"] = sha((run / rel).read_bytes())
    write(run / "motion_provenance.json", prov)
    ready = json.loads((run / exporter.READY).read_text())
    for name in (rel, "motion_provenance.json"):
        ready["files"][name] = sha((run / name).read_bytes())
    write(run / exporter.READY, ready)
    with pytest.raises(ValueError, match=match):
        exporter.snapshot_sources(run, reg)


def test_restore_batches_keep_legacy_compatibility(tmp_path, exporter, monkeypatch):
    run, _ = make_run(tmp_path, exporter, monkeypatch)
    record = json.loads((run / "online_restore_inputs.json").read_text())
    record["exec_batch_lengths"] = [33, 32]
    exporter.check_online_inputs(record, run / "online_restore_inputs.npz")
    record["exec_batch_lengths"] = [65]
    with pytest.raises(ValueError, match="窗口合同"):
        exporter.check_online_inputs(record, run / "online_restore_inputs.npz")


def test_incremental_only_stages_committed_steps(tmp_path, exporter, monkeypatch, capsys):
    run, _ = make_run(tmp_path, exporter, monkeypatch)
    monkeypatch.setattr(exporter, "validate_store", lambda *args: None)
    write(run / "10000/_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": None})
    shutil.rmtree(run / "59999")
    root = tmp_path / "export"
    (root / "stage").mkdir(parents=True)
    exporter.run_checkpoints("incremental", True, root)
    assert not (root / "stage/10000").exists()
    assert not (root / "stage/59999").exists()
    assert len(list((root / "stage/manifests").glob("step-*.json"))) == 10
    assert "SMOKE=PASS kind=checkpoint mode=incremental steps=10" in capsys.readouterr().out


@pytest.mark.parametrize("fault", ["missing", "wrong"])
def test_snapshot_binds_compat_cache_receipt(tmp_path, exporter, monkeypatch, fault):
    run, reg = make_run(tmp_path, exporter, monkeypatch)
    path = run / "motion_provenance.json"
    prov = json.loads(path.read_text())
    if fault == "missing":
        del prov["stage2_compat"]["cache_import_receipt_sha256"]
    else:
        prov["stage2_compat"]["cache_import_receipt_sha256"] = "f" * 64
    write(path, prov)
    ready = json.loads((run / exporter.READY).read_text())
    ready["files"][path.name] = sha(path.read_bytes())
    write(run / exporter.READY, ready)
    with pytest.raises(ValueError, match="缓存导入收据SHA"):
        exporter.snapshot_sources(run, reg)


def test_checkpoint_commit_and_step_set(tmp_path, exporter, monkeypatch):
    run, _ = make_run(tmp_path, exporter, monkeypatch)
    log = exporter.REPO / "v1-store/logs" / f"{exporter.RUN}.log"
    exporter.strict_final(run, log)
    write(run / "5000/_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": None})
    with pytest.raises(ValueError, match="尚未提交"):
        exporter.step_sources(run, 5000)
    shutil.rmtree(run / "59999")
    with pytest.raises(ValueError, match="12步集合"):
        exporter.strict_final(run, log)


class FakeBucket:
    def __init__(self, files):
        self.files = files

    def download(self, name, target):
        write(target, self.files[name])
        return types.SimpleNamespace(ok=True)

    def list_files(self):
        return {name: len(data) for name, data in self.files.items()}

    def entry(self):
        return {"private": False}


@pytest.mark.parametrize("failure", [None, "sha", "bytes", "extra"])
def test_anonymous_readback_verifies_exact_files(tmp_path, exporter, failure):
    stage = tmp_path / "stage"
    write(stage / "x.bin", b"original")
    files = {"x.bin": exporter.file_info(stage / "x.bin")}
    remote = {"x.bin": b"original"}
    if failure == "sha":
        remote["x.bin"] = b"tampered"
    elif failure == "bytes":
        remote["x.bin"] = b"short"
    elif failure == "extra":
        remote["extra.bin"] = b"extra"
    publisher = exporter.Publisher.__new__(exporter.Publisher)
    publisher.anon = FakeBucket(remote)
    publisher.verify = tmp_path / "verify"
    publisher.verify.mkdir()
    if failure:
        with pytest.raises(ValueError, match="SHA|字节数|集合"):
            publisher.verify_all(stage, files)
    else:
        publisher.verify_all(stage, files)
        assert not list(publisher.verify.iterdir())


def test_anonymous_environment_has_no_credentials(tmp_path, exporter, monkeypatch):
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        monkeypatch.setenv(key, "fixture")
    bucket = exporter.AnonymousBucket("test/fixture", "", cache_root=tmp_path)
    env = bucket._env()
    assert not any(key in env for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN_PATH"))
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert env["HF_HOME"] == str(tmp_path / "anonymous-home")


def test_private_bucket_is_rejected(tmp_path, exporter):
    publisher = exporter.Publisher.__new__(exporter.Publisher)
    publisher.root = tmp_path
    publisher.identity = {"kind": "orig16-stage2-vla"}
    publisher.bucket = types.SimpleNamespace(entry=lambda: {"private": True})
    write(tmp_path / "owner.json", publisher.identity)
    with pytest.raises(ValueError, match="必须公开"):
        publisher.ensure_bucket()


def test_wrapper_syntax_and_runtime_selection():
    for path in [COMMON, *DRIVERS]:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    text = COMMON.read_text()
    assert "python3" not in text and "--private" not in text and "restore_receipt" not in text
    assert "mme_vla_suite_orig16_b128_60k" in text


CONCAT_INDEX = (2, 3)


def test_runs_include_two_concat_pairs(exporter):
    assert list(exporter.RUNS)[2:] == ["orig100-ctx32x4x4-motion-s2concat1x1024-b128-60k",
                                       "orig100-m32x4x4-modul-motion-s2concat1x1024-b128-60k"]
    assert [exporter.RUNS[run][0] for run in list(exporter.RUNS)[2:]] == [
        "HongzeFu/robomme-vla-orig100-ctx32x4x4-motion-s2concat1x1024-60k-v1",
        "HongzeFu/robomme-vla-orig100-m32x4x4-modul-motion-s2concat1x1024-60k-v1"]
    for run in list(exporter.RUNS)[2:]:
        assert exporter.history_name(run).endswith("-32frame-4x4-motion-stage2-1x1024-orig16.yaml")
        assert exporter.readme_path(run).is_file() and "concat" in exporter.readme_path(run).name


@pytest.mark.parametrize("index", CONCAT_INDEX)
def test_concat_final_checkpoint_smoke(tmp_path, exporter, monkeypatch, capsys, index):
    make_run(tmp_path, exporter, monkeypatch, index)
    monkeypatch.setattr(exporter, "validate_store", lambda *args: None)
    root = tmp_path / "export"
    (root / "stage").mkdir(parents=True)
    exporter.run_checkpoints("final", True, root)
    manifest = json.loads((root / "stage/delivery_manifest.json").read_text())
    assert manifest["bucket"] == exporter.RUNS[exporter.RUN][0] and manifest["steps"][-1] == 59999
    assert (root / "stage/README.md").read_bytes() == exporter.readme_path(exporter.RUN).read_bytes()
    assert f"encoder_dependencies/{exporter.ENC_RUN}/checkpoint_epoch_105.pt" in manifest["files"]
    assert "SMOKE=PASS kind=checkpoint mode=final steps=12" in capsys.readouterr().out


@pytest.mark.parametrize("index", CONCAT_INDEX)
def test_concat_producer_manifest_consumed_by_restore(tmp_path, exporter, checkpoint_consumer, monkeypatch, index):
    stage, path = make_producer_delivery(tmp_path, exporter, monkeypatch, index)
    record = checkpoint_consumer.verify_delivery_manifest(stage, path, stage / "59999")
    assert record["step"] == 59999


@pytest.mark.parametrize("pair", [(2, 0), (0, 2), (2, 3), (3, 1)])
def test_concat_run_bucket_mismatch_fails(exporter, monkeypatch, pair):
    runs = list(exporter.RUNS)
    monkeypatch.setattr(exporter, "DATASET", "16task-orig100-1600ep-12345678")
    monkeypatch.setattr(exporter, "RUN", runs[pair[0]])
    monkeypatch.setenv("O16_MODE", "final")
    monkeypatch.setenv("BUCKET", exporter.RUNS[runs[pair[1]]][0])
    monkeypatch.delenv("ATTEMPT_MANIFEST", raising=False)
    with pytest.raises(ValueError, match="固定配对"):
        exporter.main()


@pytest.mark.parametrize("motion,match", [({"budget": 384}, "1×1024"), ({"num_tokens": 4, "dim": 256}, "1×1024"),
                                          ({"readout": "stage2_pool4_split_v1"}, "1×1024")])
def test_concat_history_checked(tmp_path, exporter, monkeypatch, motion, match):
    run, _ = make_run(tmp_path, exporter, monkeypatch, 2)
    history = json.loads((run / "history_config.resolved.yaml").read_text())
    exporter.check_history(history)
    history["motion"].update(motion)
    # yaml与快照同时漂移，负例必须命中concat读法校验而非仅仅是快照不等。
    write(exporter.REPO / "src/mme_vla_suite/models/config/robomme" / exporter.history_name(exporter.RUN), history)
    with pytest.raises(ValueError, match=match):
        exporter.check_history(history)


def test_concat_split_history_cannot_cross(tmp_path, exporter, monkeypatch):
    run, reg = make_run(tmp_path, exporter, monkeypatch, 2)
    write(run / "history_config.txt", exporter.history_name(list(exporter.RUNS)[0]))
    with pytest.raises(ValueError, match="history标签不符|根快照SHA不符"):
        exporter.snapshot_sources(run, reg)


@pytest.mark.parametrize("index", CONCAT_INDEX)
def test_concat_stage_missing_encoder_entity_rejected(tmp_path, exporter, checkpoint_consumer, monkeypatch, index):
    stage, path = make_producer_delivery(tmp_path, exporter, monkeypatch, index)
    external = exporter.REPO / "v1-store/external/motionjepa" / exporter.ENC_RUN / "checkpoint_epoch_105.pt"
    entity = stage / "encoder_dependencies" / exporter.ENC_RUN / "checkpoint_epoch_105.pt"
    assert external.is_file() and entity.is_file()
    entity.unlink()
    # 外部源仍在，但stage缺依赖实体：交付恢复必须拒绝。
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        checkpoint_consumer.verify_delivery_manifest(stage, path, stage / "59999")


def attempt_record(tmp_path, exporter, *, index=2, nn="02", allowed=True, **override):
    base = list(exporter.RUNS)[index]
    run = f"{base}-attempt{nn}"
    decision = write(tmp_path / "decisions/evt.json", {"allowed_runs": [run] if allowed else []})
    record = {"base_run": base, "run": run, "history": exporter.history_name(base), "train_head": "a" * 40,
              "decision": str(decision), "bucket": exporter.RUNS[base][0] + f"-attempt{nn}"}
    record.update(override)
    return write(tmp_path / "attempt.json", record), record


def test_registered_attempt_final_smoke(tmp_path, exporter, monkeypatch, capsys):
    path, record = attempt_record(tmp_path, exporter)
    make_run(tmp_path, exporter, monkeypatch, 2, run_name=record["run"])
    monkeypatch.setattr(exporter, "validate_store", lambda *args: None)
    for key, value in dict(O16_MODE="final", SMOKE="1", BUCKET=record["bucket"], ATTEMPT_MANIFEST=str(path),
                           O16_EXPORT_ROOT=str(tmp_path / "export")).items():
        monkeypatch.setenv(key, value)
    exporter.main()
    out = capsys.readouterr().out
    assert f"ATTEMPT=PASS run={record['run']}" in out and "SMOKE=PASS kind=checkpoint mode=final steps=12" in out
    manifest = json.loads((tmp_path / "export/stage/delivery_manifest.json").read_text())
    assert manifest["run_name"] == record["run"] and manifest["bucket"] == record["bucket"]


@pytest.mark.parametrize("fault", ["unregistered", "no_manifest", "nn01", "bucket", "history", "run_env"])
def test_attempt_faults_rejected(tmp_path, exporter, monkeypatch, fault):
    kwargs = {}
    if fault == "unregistered": kwargs["allowed"] = False
    if fault == "nn01": kwargs["nn"] = "01"
    if fault == "bucket": kwargs["bucket"] = exporter.RUNS[list(exporter.RUNS)[2]][0] + "-attempt03"
    if fault == "history": kwargs["history"] = exporter.history_name(list(exporter.RUNS)[0])
    path, record = attempt_record(tmp_path, exporter, **kwargs)
    monkeypatch.setattr(exporter, "DATASET", "16task-orig100-1600ep-12345678")
    monkeypatch.setattr(exporter, "RUN", record["run"] if fault != "run_env" else list(exporter.RUNS)[2])
    for key, value in dict(O16_MODE="final", SMOKE="1", BUCKET=record["bucket"],
                           O16_EXPORT_ROOT=str(tmp_path / "export")).items():
        monkeypatch.setenv(key, value)
    if fault == "no_manifest":
        monkeypatch.delenv("ATTEMPT_MANIFEST", raising=False)
    else:
        monkeypatch.setenv("ATTEMPT_MANIFEST", str(path))
    with pytest.raises(ValueError, match="固定配对|attempt|决定记录|RUN/BUCKET"):
        exporter.main()
    assert not (tmp_path / "export").exists()


def test_concat_wrapper_rejects_other_role_attempt(tmp_path, exporter):
    path, _ = attempt_record(tmp_path, exporter, index=3)
    result = subprocess.run(["bash", str(HERE / "run_orig16_ctx32x4x4_concat_60k_ckpt_export.sh")],
                            env=env_for_scripts(SMOKE="1", ATTEMPT_MANIFEST=str(path),
                                                O16_EXPORT_ROOT=str(tmp_path / "export")),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and "base_run" in result.stderr
    assert not (tmp_path / "export").exists()
