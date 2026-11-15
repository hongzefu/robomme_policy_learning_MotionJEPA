"""4task stage2 rgb-state两条50k导出CPU夹具：增量只出INC_END、stage离线、upload复用stage、eval/独立范围与stage删除条件。

假bucket只在pytest临时目录里（目录即对象存储），任何用例都不读取凭据、不联网。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import zipfile

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
COMMON = HERE / "_stage2_4task_vla_ckpt_export_common.sh"
INC = HERE / "run_4task_stage2_ckpt_inc.sh"
WRAPPERS = [HERE / "run_4task_modul8x8_concat_50k_ckpt_export.sh", HERE / "run_4task_ctx8x8_concat_50k_ckpt_export.sh"]
# local-built-latents 的证据角色由S2的 motion_store.stage2_evidence_roles 定义；导出器按该函数动态取集合，
# 本夹具在S2合入前用计划B表所列角色（G0/G2/G3、SOURCE_PIN、member-pin血缘收据、逐段bin摘要清单）代入。
LOCAL_ROLES = ("G0", "G2", "G3", "SOURCE_PIN", "lineage", "wan_latents_bin_sha256")
STEPS = list(range(5000, 50001, 5000))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        data = json.dumps(data)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


@pytest.fixture
def exporter(monkeypatch):
    module = types.ModuleType("task4_export_test")
    code = COMMON.read_text().split("<<'O16_PY'\n", 1)[1].rsplit("\nO16_PY", 1)[0]
    monkeypatch.setenv("O16_REPO", str(REPO))
    for key in ("ATTEMPT_MANIFEST", "O16_EXPORT_ROOT", "EVAL_RUN_NAME", "SMOKE"):
        monkeypatch.delenv(key, raising=False)
    exec(compile(code, str(COMMON), "exec"), module.__dict__)
    monkeypatch.setattr(module.io, "load_hf_token", lambda *a: pytest.fail("离线测试不得读取凭据"))
    monkeypatch.setattr(module.io.HFBucket, "_run", lambda *a: pytest.fail("离线测试不得联网"))
    monkeypatch.setattr(module, "evidence_roles", lambda prov: LOCAL_ROLES)
    # S2 的表验证另有完整测试；这里隔离导出与上传阶段。
    monkeypatch.setattr(module, "validate_store", lambda *args: None)
    return module


@pytest.fixture
def checkpoint_consumer():
    path = REPO / "scripts/training/check_stage2_checkpoint.py"
    spec = importlib.util.spec_from_file_location("task4_checkpoint_consumer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeHub:
    """本地目录充当对象存储：bucket 名 → 目录；记录每次上传。"""

    def __init__(self, root):
        self.root = Path(root)
        self.uploads = []

    def bucket_dir(self, bucket):
        return self.root / bucket.replace("/", "__")


class FakeBucket:
    def __init__(self, hub, bucket):
        self.hub, self.bucket = hub, bucket
        self.base = hub.bucket_dir(bucket)

    def entry(self):
        return {"private": False} if self.base.is_dir() else None

    def _run(self, args, timeout):
        assert args == ["buckets", "create", self.bucket]
        self.base.mkdir(parents=True)
        return 0, "", ""

    def list_files(self):
        return {p.relative_to(self.base).as_posix(): p.stat().st_size for p in self.base.rglob("*") if p.is_file()}

    def upload(self, local, name):
        target = self.base / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)
        self.hub.uploads.append(name)
        return types.SimpleNamespace(ok=True)

    def download(self, name, target):
        source = self.base / name
        if not source.is_file():
            return types.SimpleNamespace(ok=False)
        shutil.copyfile(source, target)
        return types.SimpleNamespace(ok=True)


@pytest.fixture
def hub(tmp_path, exporter, monkeypatch):
    fake = FakeHub(tmp_path / "fake-hub")
    monkeypatch.setattr(exporter, "make_buckets", lambda bucket, root: (FakeBucket(fake, bucket), FakeBucket(fake, bucket)))
    return fake


def make_run(tmp_path, exporter, monkeypatch, index=0, run_name=None, log_tail="EXIT_CODE=0\n"):
    base = list(exporter.RUNS)[index]
    run_name = run_name or base
    repo = tmp_path / "repo"
    monkeypatch.setattr(exporter, "REPO", repo)
    monkeypatch.setattr(exporter, "RUN", run_name)
    run = repo / exporter.TRAIN_ROOT / run_name
    reg = repo / "v1-store/external/motionjepa" / exporter.ENC_RUN
    cfg = write(reg / "config.yaml", "fixture: true\n")
    ckpt = reg / exporter.ENC_CKPT
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ckpt, "w") as archive:
        archive.writestr("fixture/data.pkl", b"\x80\x04N.")
    enc_sha = sha(ckpt.read_bytes())
    history = {"motion": {"store_path": f"v1-store/datasets/{exporter.DATASET}/{exporter.MOTION_SUB}",
                          "readout": "stage2_pool4_concat_v1", "num_tokens": 1, "dim": 1024, "budget": 96,
                          "checkpoint_sha256": enc_sha},
               "_build": "/scratch/hongze/robomme_policy_learning_MotionJEPA"}
    name = exporter.history_name(base)
    write(repo / "src/mme_vla_suite/models/config/robomme" / name, history)
    write(run / "history_config.txt", name)
    resolved = write(run / "history_config.resolved.yaml", history)
    resolved_sha = sha(resolved.read_bytes())
    write(run / "history_config.resolved.sha256", resolved_sha)
    source_pin = {"member_pin_sha256": "2cb9026fc874964941ddb8c1cf1c0425a3b8ac94d3bd8e2082779f4e95f5c8bd", "files": 1600}
    g0 = write(run / "motion_evidence/G0.json", {"gate": "G0", "status": "PASS", "host": "ip-10-242-11-177"})
    g0_sha = sha(g0.read_bytes())
    write(run / "motion_evidence/G2.json", {"gate": "G2", "status": "PASS", "samples_sha256": g0_sha,
                                            "gpu": "GPU-0123abcd-ffff"})
    write(run / "motion_evidence/G3.json", {"gate": "G3", "status": "PASS", "checkpoint_sha256": enc_sha,
                                            "samples_sha256": g0_sha})
    write(run / "motion_evidence/SOURCE_PIN.json", source_pin)
    write(run / "motion_evidence/lineage.json", {"verdict": "DATA_LINEAGE=PASS",
                                                 "path": "/scratch/hongze/robomme_policy_learning_MotionJEPA/x"})
    write(run / "motion_evidence/wan_latents_bin_sha256.json",
          {"schema": "bin-sha256-list-v1", "root": "/scratch/hongze/x/wan-latents", "files": {"a.bin": "e" * 64}})
    evidence = {role: {"path": f"motion_evidence/{role}.json",
                       "sha256": sha((run / f"motion_evidence/{role}.json").read_bytes())} for role in LOCAL_ROLES}
    encoder = {"run_name": exporter.ENC_RUN, "checkpoint_name": exporter.ENC_CKPT, "epoch": 144, "global_step": 9999,
               "arch": exporter.ARCH, "state_key": "model", "weight_type": "ema", "source_times": [0, 3, 5, 8],
               "output_shape": [1, 1024], "checkpoint_sha256": enc_sha, "config_sha256": sha(cfg.read_bytes()),
               "fingerprint_sha256": "a" * 64}
    prov = {"motion_enabled": True, "resolved_sha256": resolved_sha, "encoder": encoder, "source_pin": source_pin,
            "encoder_state_sha256": {"fixture": "b" * 64}, "numerics": {"fixture": True}, "vae": {"fixture": True},
            "stage2_compat": {"kind": exporter.KIND, "evidence": evidence}}
    write(run / "motion_provenance.json", prov)
    deps = {"schema": "stage2-encoder-dependencies-v1", "encoder": encoder, "source_pin": source_pin, "vae": prov["vae"],
            "files": [{"role": role, "source_path": str(path),
                       "relative_path": f"encoder_dependencies/{exporter.ENC_RUN}/{path.name}", **exporter.file_info(path)}
                      for role, path in (("encoder_checkpoint", ckpt), ("encoder_config", cfg))]}
    write(run / "encoder_dependencies.json", deps)
    arrays = {"images": np.zeros((82, 256, 256, 3), dtype=np.uint8), "states": np.zeros((82, 8), dtype=np.float32),
              "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8)}
    np.savez_compressed(run / "online_restore_inputs.npz", **arrays)
    online = {"schema": "stage2-online-restore-inputs-v1", "exec_start_idx": 17, "exec_batch_lengths": [1, 16, 16, 16, 16],
              "prompt": "夹具", "sources": [{"path": "/scratch/hongze/fixture.h5", "sha256": "a" * 64}],
              "tokenizer": {"uri": exporter.TOKENIZER_URI, "sha256": exporter.TOKENIZER_SHA},
              "payload": {"path": "online_restore_inputs.npz", "sha256": sha((run / "online_restore_inputs.npz").read_bytes())},
              "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha(value.tobytes())}
                         for key, value in arrays.items()}}
    write(run / "online_restore_inputs.json", online)
    names = set(exporter.ROOT_FILES) | {f"motion_evidence/{role}.json" for role in LOCAL_ROLES}
    write(run / exporter.READY, {"schema": "stage2-motion-snapshot-v1",
                                 "files": {rel: sha((run / rel).read_bytes()) for rel in names}})
    for step in STEPS:
        write(run / str(step) / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": 1})
        write(run / str(step) / "params/_METADATA", {"fixture": True})
        write(run / str(step) / "params/part.bin", bytes(8) + str(step).encode())
        write(run / str(step) / "assets/robomme/norm_stats.json", {"norm_stats": {"state": {"mean": [0.0]}}})
    write(repo / exporter.LOG_DIR / f"{run_name}.log",
          f"RUN={run_name}\nTRAIN_HEAD={'a' * 40}\nhost ip-10-242-11-177 /scratch/hongze/robomme_policy_learning_MotionJEPA\n{log_tail}")
    return run, reg


def set_env(monkeypatch, exporter, mode, **extra):
    monkeypatch.setenv("O16_MODE", mode)
    monkeypatch.setenv("BUCKET", extra.pop("BUCKET", exporter.bucket_id() if exporter.RUN in exporter.RUNS else ""))
    for key, value in extra.items():
        monkeypatch.setenv(key, value)


def ckpt_root(exporter):
    return exporter.REPO / "v1-store/exports" / f"hf-ckpt-{exporter.RUN}"


def eval_root(exporter):
    return exporter.REPO / "v1-store/exports" / f"hf-eval-{exporter.RUN}"


def make_eval(exporter, name=None):
    """真实形态的692评估目录：EVAL692.txt、merged、分片目录、日志、计划、视频、gpu-slots占位。"""
    name = name or f"local692-s2rgbstate-{exporter.role()}-concat1024-50k-s7"
    root = exporter.REPO / "v1-store/evaluation" / name
    write(root / "EVAL692.txt", "EVAL692=PASS episodes=692 success=300 rate=0.4335 preseeded_errors=9 new_errors=0 "
                                "retried=0 expect_motion_stats=1\n")
    write(root / "merged/log.json", {"episodes": 692, "successes": 300})
    write(root / "merged/errors.json", [])
    write(root / "preseeded-errors.json", {"keys": [], "src": "/scratch/hongze/robomme_policy_learning_MotionJEPA/x"})
    write(root / "retries.json", [])
    write(root / "plans/shard0.json", {"groups": {}, "ckpt": f"/scratch/hongze/x/hf-ckpt-{exporter.RUN}/stage/50000"})
    write(root / "logs/s0.log", "LOCAL692 ip-10-242-11-177 GPU-0123abcd-0000-1111 /scratch/hongze/robomme_policy_learning_MotionJEPA\n")
    write(root / "s0/client.log", "DEMO_INJECT ok /scratch/hongze/MotionJEPA/x\n")
    write(root / "s0/progress.json", {"done": 1})
    write(root / "s0/videos/BinFill_ep0.mp4", bytes(64))
    write(root / "gpu-slots/gpu0.lock", "")
    write(exporter.REPO / exporter.LOG_DIR / f"{name}.log", "LOCAL692_START\n" + (root / "EVAL692.txt").read_text())
    return root


def run_main(exporter, monkeypatch, capsys, mode, **extra):
    set_env(monkeypatch, exporter, mode, **extra)
    exporter.main()
    return capsys.readouterr().out


# ── 常量与配对 ──

def test_runs_buckets_steps_and_templates(exporter):
    assert list(exporter.RUNS) == ["v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k",
                                   "v2-1600ep-m8x8-ctx-motion-s2rgbstate-concat1x1024-b128-50k"]
    assert [v[0] for v in exporter.RUNS.values()] == [
        "HongzeFu/robomme-vla-4task-m8x8-modul-s2rgbstate-concat1x1024-50k-v1",
        "HongzeFu/robomme-vla-4task-m8x8-ctx-s2rgbstate-concat1x1024-50k-v1"]
    assert [v[2:] for v in exporter.RUNS.values()] == [("modulation", 1024), ("context", 2048)]
    assert list(exporter.CKPT_STEPS) == STEPS and len(exporter.CKPT_STEPS) == 10
    for run in exporter.RUNS:
        assert exporter.history_name(run).endswith("-8frame-8x8-motion-stage2-1x1024-4task-rgbstate.yaml")
        text = exporter.readme_path(run).read_text()
        assert exporter.RUNS[run][0] in text and run in text and "eval/" in text and "CKPT_UPLOAD=PASS steps=10" in text


@pytest.mark.parametrize("pair", [(0, 1), (1, 0)])
def test_bucket_pair_is_fixed(exporter, monkeypatch, pair):
    runs = list(exporter.RUNS)
    monkeypatch.setattr(exporter, "RUN", runs[pair[0]])
    set_env(monkeypatch, exporter, "stage", BUCKET=exporter.RUNS[runs[pair[1]]][0])
    with pytest.raises(ValueError, match="固定配对"):
        exporter.main()


# ── 增量 ──

@pytest.mark.parametrize("index", [0, 1])
def test_smoke_prints_ten_step_plan_and_encoder_check(tmp_path, exporter, monkeypatch, capsys, index):
    make_run(tmp_path, exporter, monkeypatch, index)
    monkeypatch.setattr(exporter, "make_buckets", lambda *a: pytest.fail("SMOKE不得建发布器"))
    out = run_main(exporter, monkeypatch, capsys, "incremental", SMOKE="1")
    assert f"CKPT_STEPS_PLAN={','.join(map(str, STEPS))} count=10" in out
    assert "ENCODER_CHECK=PASS run=stage2-4task-rgbstate-144ep-1113-a epoch=144" in out
    assert "SMOKE=PASS kind=checkpoint mode=incremental steps=10 network=0" in out
    assert len(list((ckpt_root(exporter) / "stage/manifests").glob("step-*.json"))) == 10


def test_incremental_only_ends_with_inc_end(tmp_path, exporter, hub, monkeypatch, capsys):
    make_run(tmp_path, exporter, monkeypatch)
    out = run_main(exporter, monkeypatch, capsys, "incremental")
    assert f"INC_END run={exporter.RUN} done=10" in out
    assert "HF_VERIFY" not in out and "CKPT_UPLOAD" not in out
    assert out.count("INC_UPLOADED step=") == 10
    assert not (ckpt_root(exporter) / "stage/delivery_manifest.json").exists()


def test_encoder_identity_rejected(tmp_path, exporter, monkeypatch):
    run, reg = make_run(tmp_path, exporter, monkeypatch)
    prov = json.loads((run / "motion_provenance.json").read_text())
    prov["encoder"]["epoch"] = 105
    with pytest.raises(ValueError, match="ep144"):
        exporter.check_encoder(prov["encoder"])


def test_snapshot_requires_local_built_kind_and_role_set(tmp_path, exporter, monkeypatch):
    run, reg = make_run(tmp_path, exporter, monkeypatch)
    exporter.snapshot_sources(run, reg)
    monkeypatch.setattr(exporter, "evidence_roles", lambda prov: LOCAL_ROLES[:-1])
    with pytest.raises(ValueError, match="证据集合"):
        exporter.snapshot_sources(run, reg)


# ── stage 模式 ──

@pytest.mark.parametrize("index", [0, 1])
def test_stage_mode_offline_without_upload_lock(tmp_path, exporter, monkeypatch, capsys, checkpoint_consumer, index):
    make_run(tmp_path, exporter, monkeypatch, index)
    monkeypatch.setattr(exporter, "make_buckets", lambda *a: pytest.fail("stage不得联网"))
    out = run_main(exporter, monkeypatch, capsys, "stage")
    root = ckpt_root(exporter)
    stage = root / "stage"
    assert "STAGE=PASS" in out and "network=0" in out and "EXIT_CODE=0" in out
    assert not (root / "upload.lock").exists() and not (root / "owner.json").exists()
    delivery = json.loads((stage / "delivery_manifest.json").read_text())
    assert delivery["steps"] == STEPS and delivery["bucket"] == exporter.RUNS[exporter.RUN][0]
    assert f"encoder_dependencies/{exporter.ENC_RUN}/{exporter.ENC_CKPT}" in delivery["files"]
    assert "_run-meta/train.log" in delivery["files"] and "motion_evidence/lineage.json" in delivery["files"]
    # 交付检查消费的正是这份 step-50000 清单与 stage/50000（含encoder依赖实体）。
    record = checkpoint_consumer.verify_delivery_manifest(stage, stage / "manifests/step-50000.json", stage / "50000")
    assert record["step"] == 50000
    (stage / f"encoder_dependencies/{exporter.ENC_RUN}/{exporter.ENC_CKPT}").unlink()
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        checkpoint_consumer.verify_delivery_manifest(stage, stage / "manifests/step-50000.json", stage / "50000")


@pytest.mark.parametrize("fault", ["missing_step", "extra_step", "exit_code"])
def test_stage_rejects_incomplete_training(tmp_path, exporter, monkeypatch, fault):
    run, _ = make_run(tmp_path, exporter, monkeypatch, log_tail="EXIT_CODE=1\n" if fault == "exit_code" else "EXIT_CODE=0\n")
    if fault == "missing_step":
        shutil.rmtree(run / "45000")
    elif fault == "extra_step":
        write(run / "50001/_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": 1})
    set_env(monkeypatch, exporter, "stage")
    with pytest.raises(ValueError, match="10步集合|EXIT_CODE|训练退出失败"):
        exporter.main()
    assert not (ckpt_root(exporter) / "stage/delivery_manifest.json").exists()


def test_stage_after_incremental_reuses_frozen_files(tmp_path, exporter, hub, monkeypatch, capsys):
    make_run(tmp_path, exporter, monkeypatch)
    run_main(exporter, monkeypatch, capsys, "incremental")
    out = run_main(exporter, monkeypatch, capsys, "stage")
    assert "STAGE=PASS" in out


# ── upload 模式 ──

def test_upload_reuses_stage_and_verifies(tmp_path, exporter, hub, monkeypatch, capsys):
    make_run(tmp_path, exporter, monkeypatch)
    run_main(exporter, monkeypatch, capsys, "incremental")
    incremental_uploads = len(hub.uploads)
    run_main(exporter, monkeypatch, capsys, "stage")
    stage_before = {p: p.stat().st_mtime_ns for p in (ckpt_root(exporter) / "stage").rglob("*") if p.is_file()}
    out = run_main(exporter, monkeypatch, capsys, "upload")
    assert "STAGE_REUSE=PASS" in out
    assert f"HF_VERIFY=PASS bucket={exporter.bucket_id()} scope=ckpt" in out and "CKPT_UPLOAD=PASS steps=10" in out
    # 增量已传的root与10步不重传，只补 final 组（train.log、delivery_manifest.json）。
    assert sorted(hub.uploads[incremental_uploads:]) == ["_run-meta/train.log", "delivery_manifest.json"]
    assert stage_before == {p: p.stat().st_mtime_ns for p in (ckpt_root(exporter) / "stage").rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", ["no_stage", "no_delivery", "tamper", "extra", "missing"])
def test_upload_rejects_missing_or_mismatched_stage(tmp_path, exporter, hub, monkeypatch, capsys, fault):
    make_run(tmp_path, exporter, monkeypatch)
    if fault != "no_stage":
        run_main(exporter, monkeypatch, capsys, "stage")
    stage = ckpt_root(exporter) / "stage"
    if fault == "no_delivery":
        (stage / "delivery_manifest.json").unlink()
    elif fault == "tamper":
        target = stage / "50000/params/part.bin"
        data = bytearray(target.read_bytes())
        data[0] ^= 1
        target.write_bytes(data)
    elif fault == "extra":
        write(stage / "50000/params/unlisted.bin", bytes(4))
    elif fault == "missing":
        (stage / "manifests/step-25000.json").unlink()
    monkeypatch.setattr(exporter, "make_buckets", lambda *a: pytest.fail("stage不符时不得建发布器"))
    set_env(monkeypatch, exporter, "upload")
    with pytest.raises(ValueError, match="stage"):
        exporter.main()
    assert not hub.uploads


# ── eval/ 范围 ──

def test_eval_upload_runs_while_ckpt_lock_held(tmp_path, exporter, hub, monkeypatch, capsys):
    make_run(tmp_path, exporter, monkeypatch)
    run_main(exporter, monkeypatch, capsys, "incremental")
    make_eval(exporter)
    lock = exporter.io.SingleInstanceLock(ckpt_root(exporter) / "upload.lock")
    with lock:
        out = run_main(exporter, monkeypatch, capsys, "eval")
        # 同一ckpt锁此刻确被占用。
        with pytest.raises(exporter.io.UploaderStop):
            with exporter.io.SingleInstanceLock(ckpt_root(exporter) / "upload.lock"):
                pass
    assert "EVAL_HF_VERIFY=PASS" in out and "scope=eval evals=1" in out
    assert (eval_root(exporter) / "upload.lock").exists() and (eval_root(exporter) / "logs/uploaded").is_dir()
    name = f"local692-s2rgbstate-{exporter.role()}-concat1024-50k-s7"
    remote = FakeBucket(hub, exporter.bucket_id()).list_files()
    assert f"eval/{name}/eval_manifest.json" in remote and f"eval/{name}/s0/videos/BinFill_ep0.mp4" in remote
    assert not any("gpu-slots" in key for key in remote)


def test_eval_requires_ckpt_bucket_ownership(tmp_path, exporter, hub, monkeypatch, capsys):
    make_run(tmp_path, exporter, monkeypatch)
    make_eval(exporter)
    set_env(monkeypatch, exporter, "eval")
    with pytest.raises(ValueError, match="ckpt增量上传"):
        exporter.main()
    assert not hub.uploads


def test_eval_name_must_match_fusion(tmp_path, exporter, monkeypatch):
    make_run(tmp_path, exporter, monkeypatch, 0)
    monkeypatch.setenv("EVAL_RUN_NAME", "local692-s2rgbstate-ctx-concat1024-50k-s7")
    with pytest.raises(ValueError, match="融合方式"):
        exporter.eval_run_name()


@pytest.mark.parametrize("order", ["eval_first", "upload_first", "restart"])
@pytest.mark.parametrize("index", [0, 1])
def test_three_orders_both_verdicts_pass(tmp_path, exporter, hub, monkeypatch, capsys, order, index):
    make_run(tmp_path, exporter, monkeypatch, index)
    make_eval(exporter)
    run_main(exporter, monkeypatch, capsys, "incremental")
    run_main(exporter, monkeypatch, capsys, "stage")
    sequence = {"eval_first": ["eval", "upload"], "upload_first": ["upload", "eval"],
                "restart": ["upload", "eval", "upload", "eval"]}[order]
    outputs = []
    for step, mode in enumerate(sequence):
        if order == "restart" and step == 2:
            uploads_before_restart = len(hub.uploads)
        outputs.append(run_main(exporter, monkeypatch, capsys, mode))
    text = "".join(outputs)
    assert text.count("EVAL_HF_VERIFY=PASS") == sequence.count("eval")
    assert text.count("CKPT_UPLOAD=PASS steps=10") == sequence.count("upload")
    if order == "restart":
        assert len(hub.uploads) == uploads_before_restart
    # 先后顺序之外再各复核一次：ckpt 范围不看 eval/，eval 范围只看 eval/。
    again = run_main(exporter, monkeypatch, capsys, "upload") + run_main(exporter, monkeypatch, capsys, "eval")
    assert "HF_VERIFY=PASS" in again and "EVAL_HF_VERIFY=PASS" in again


@pytest.mark.parametrize("scope", ["ckpt", "eval"])
def test_scope_verify_detects_its_own_drift_only(tmp_path, exporter, hub, monkeypatch, capsys, scope):
    make_run(tmp_path, exporter, monkeypatch)
    make_eval(exporter)
    run_main(exporter, monkeypatch, capsys, "incremental")
    run_main(exporter, monkeypatch, capsys, "stage")
    run_main(exporter, monkeypatch, capsys, "upload")
    run_main(exporter, monkeypatch, capsys, "eval")
    base = hub.bucket_dir(exporter.bucket_id())
    if scope == "ckpt":
        write(base / "stray.bin", b"x")
        assert "EVAL_HF_VERIFY=PASS" in run_main(exporter, monkeypatch, capsys, "eval")
        set_env(monkeypatch, exporter, "upload")
    else:
        name = f"local692-s2rgbstate-{exporter.role()}-concat1024-50k-s7"
        write(base / f"eval/{name}/retries.json", b"[1]")
        assert "HF_VERIFY=PASS" in run_main(exporter, monkeypatch, capsys, "upload")
        set_env(monkeypatch, exporter, "eval")
    with pytest.raises(ValueError, match=f"{scope}远端文件集合|匿名SHA"):
        exporter.main()


# ── hygiene ──

def test_hygiene_pass_on_real_shape_stage_and_eval(tmp_path, exporter, hub, monkeypatch, capsys):
    make_run(tmp_path, exporter, monkeypatch)
    eval_dir = make_eval(exporter)
    run_main(exporter, monkeypatch, capsys, "incremental")
    run_main(exporter, monkeypatch, capsys, "stage")
    run_main(exporter, monkeypatch, capsys, "eval")
    allow = exporter.EXPORT / exporter.ALLOWLIST
    for root in (ckpt_root(exporter) / "stage", eval_root(exporter) / "stage"):
        files = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
        exporter.scan(files, allow)
        assert f"HYGIENE=PASS files={len(files)}" in capsys.readouterr().out
    # 凭据恒为硬停；未裁决的集群路径不被 eval 范围放行。
    for text, match in (("token hf_" + "A" * 34, "疑似凭据"), ("/nfs/turbo/x", "未裁决")):
        write(eval_dir / "logs/s1.log", text)
        with pytest.raises(ValueError, match=match):
            exporter.scan({"eval/x/logs/s1.log": eval_dir / "logs/s1.log"}, allow)


def test_allowlist_scopes_have_own_globs(exporter):
    entries = json.loads((exporter.EXPORT / exporter.ALLOWLIST).read_text())["entries"]
    evals = [e for e in entries if e["path_glob"].startswith("eval/")]
    ckpts = [e for e in entries if not e["path_glob"].startswith("eval/")]
    assert evals and ckpts and all(e["scope"] == "eval" for e in evals)
    assert not any(e["pattern"] == "P5" for e in entries)


# ── stage 删除 ──

@pytest.mark.parametrize("missing", [None, "hf_verify", "delivery", "eval692"])
def test_stage_delete_requires_three_conditions(tmp_path, exporter, monkeypatch, capsys, missing):
    make_run(tmp_path, exporter, monkeypatch)
    run_main(exporter, monkeypatch, capsys, "stage")
    log = exporter.REPO / exporter.LOG_DIR
    short = exporter.SHORT[exporter.role()]
    if missing != "hf_verify":
        write(log / f"up-{short}-final.log", f"HF_VERIFY=PASS bucket={exporter.bucket_id()} scope=ckpt objects=1 "
                                             "bytes=1 mismatches=0\nCKPT_UPLOAD=PASS steps=10\nEXIT_CODE=0\n")
    if missing != "delivery":
        write(log / f"delivery-{short}.log", "CKPT_DELIVERY_READY=PASS step=50000\n")
    if missing != "eval692":
        make_eval(exporter)
    out = run_main(exporter, monkeypatch, capsys, "stage-delete")
    stage = ckpt_root(exporter) / "stage"
    if missing:
        assert f"STAGE_DELETE=SKIP run={exporter.RUN} reason={missing}" in out and stage.is_dir()
    else:
        assert "STAGE_DELETE=PASS" in out and not stage.exists()


def test_stage_delete_refuses_failed_delivery(tmp_path, exporter, monkeypatch, capsys):
    make_run(tmp_path, exporter, monkeypatch)
    run_main(exporter, monkeypatch, capsys, "stage")
    log = exporter.REPO / exporter.LOG_DIR
    write(log / "up-m-final.log", f"HF_VERIFY=PASS bucket={exporter.bucket_id()} scope=ckpt x\nCKPT_UPLOAD=PASS steps=10\n")
    write(log / "delivery-m.log", "CKPT_DELIVERY_READY=FAIL step=50000\n")
    make_eval(exporter)
    out = run_main(exporter, monkeypatch, capsys, "stage-delete")
    assert "reason=delivery" in out and (ckpt_root(exporter) / "stage").is_dir()


# ── attempt 与包装入口 ──

def attempt_record(tmp_path, exporter, *, index=1, nn="02", allowed=True, **override):
    base = list(exporter.RUNS)[index]
    run = f"{base}-attempt{nn}"
    decision = write(tmp_path / "decisions/evt.json", {"allowed_runs": [run] if allowed else []})
    record = {"base_run": base, "run": run, "history": exporter.history_name(base), "train_head": "a" * 40,
              "decision": str(decision), "bucket": exporter.RUNS[base][0] + f"-attempt{nn}"}
    record.update(override)
    return write(tmp_path / "attempt.json", record), record


def test_registered_attempt_stage(tmp_path, exporter, monkeypatch, capsys):
    path, record = attempt_record(tmp_path, exporter)
    make_run(tmp_path, exporter, monkeypatch, 1, run_name=record["run"])
    out = run_main(exporter, monkeypatch, capsys, "stage", BUCKET=record["bucket"], ATTEMPT_MANIFEST=str(path))
    assert f"ATTEMPT=PASS run={record['run']}" in out and "STAGE=PASS" in out
    delivery = json.loads((ckpt_root(exporter) / "stage/delivery_manifest.json").read_text())
    assert delivery["run_name"] == record["run"] and delivery["bucket"] == record["bucket"]


@pytest.mark.parametrize("fault", ["unregistered", "nn01", "bucket", "history"])
def test_attempt_faults_rejected(tmp_path, exporter, monkeypatch, fault):
    kwargs = {"unregistered": {"allowed": False}, "nn01": {"nn": "01"},
              "bucket": {"bucket": exporter.RUNS[list(exporter.RUNS)[1]][0] + "-attempt03"},
              "history": {"history": exporter.history_name(list(exporter.RUNS)[0])}}[fault]
    path, record = attempt_record(tmp_path, exporter, **kwargs)
    monkeypatch.setattr(exporter, "RUN", record["run"])
    set_env(monkeypatch, exporter, "stage", BUCKET=record["bucket"], ATTEMPT_MANIFEST=str(path))
    with pytest.raises(ValueError, match="attempt|决定记录|RUN/BUCKET"):
        exporter.main()


def script_env(**extra):
    env = dict(os.environ)
    env.update(UV_PROJECT_ENVIRONMENT=str(Path(sys.executable).parent.parent), PYTHONPATH=str(REPO / "src"),
               CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", UV_CACHE_DIR="/scratch/hongze/.cache/uv",
               UV_OFFLINE="1", HF_HUB_OFFLINE="1")
    for key in ("O16_MODE", "RUN", "BUCKET", "ATTEMPT_MANIFEST", "SMOKE"):
        env.pop(key, None)
    env.update(extra)
    return env


def test_wrapper_syntax_and_mode_selection(tmp_path):
    for path in [COMMON, INC, *WRAPPERS]:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    text = COMMON.read_text()
    assert "python3" not in text and "--private" not in text and "mme_vla_suite_4task_b128_50k" in text
    for mode in (None, "incremental", "final"):
        env = script_env(**({"O16_MODE": mode} if mode else {}))
        result = subprocess.run(["bash", str(WRAPPERS[1])], env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 2 and "O16_MODE" in result.stderr
    result = subprocess.run(["bash", str(INC)], env=script_env(), capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "RUN" in result.stderr


def test_wrapper_rejects_other_role_attempt(tmp_path, exporter):
    path, _ = attempt_record(tmp_path, exporter, index=0)
    result = subprocess.run(["bash", str(WRAPPERS[1])], env=script_env(O16_MODE="stage", ATTEMPT_MANIFEST=str(path)),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and "base_run" in result.stderr


def test_stage_shell_entry_has_no_lock_or_network(tmp_path):
    """真实包装入口跑 stage：没有训练产物时在任何联网与建锁之前失败。"""
    result = subprocess.run(["bash", str(WRAPPERS[0])], env=script_env(O16_MODE="stage"), cwd=REPO,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 1 and "4TASK_EXPORT=FAIL mode=stage" in result.stdout
    root = REPO / "v1-store/exports/hf-ckpt-v2-1600ep-m8x8-modul-motion-s2rgbstate-concat1x1024-b128-50k"
    assert not root.exists()
