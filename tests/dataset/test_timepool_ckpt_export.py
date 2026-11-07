"""Wan 池化 checkpoint 公开备份的 CPU 夹具：真实快照校验 + 内存假 HF，不读凭据、不联网、不用 GPU。"""
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

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/datastore"))
from test_motion_timepool_contract import build_store  # noqa: E402

from mme_vla_suite.datastore import motion_store as ms  # noqa: E402

EXPORT = ROOT / "scripts/dataset/hf_export"
TOOL = EXPORT / "_timepool_orig16_ckpt_export.py"
ROLES = ("DATA_LINEAGE", "WAN_RESTORE_EQ", "WINDOW_EQ", "POOL_CONTRACT", "POOL_TABLE_EQ")


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def tool(monkeypatch):
    spec = importlib.util.spec_from_file_location("timepool_ckpt_export_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "DISK_MARGIN", 0)
    monkeypatch.setattr(module, "POLL_SECONDS", 0)
    monkeypatch.delenv("SMOKE", raising=False)
    monkeypatch.delenv("O16_STORE_ROOT", raising=False)
    # 进程内夹具：测试结束恢复，避免 main 的环境设置泄漏给其他测试。
    for key in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    return module


class FakeHF:
    """内存 bucket：按 hf CLI 参数应答，记录上传顺序与匿名性，可注入读回损坏或上传失败。"""

    def __init__(self, module, monkeypatch):
        self.objects, self.uploads, self.anon_env = {}, [], []
        self.exists, self.private = False, False
        self.corrupt, self.fail_upload = set(), set()
        self.tokens = []
        fake = self

        def run(bucket, args, timeout):
            return fake.run(bucket, list(args))

        monkeypatch.setattr(module.io.HFBucket, "_run", run)
        monkeypatch.setattr(module.io, "load_hf_token", self.load_token)

    def load_token(self, path):
        self.tokens.append(Path(path))
        return "fake-token-not-real"

    def run(self, bucket, args):
        if args[:2] == ["buckets", "create"]:
            self.exists = True
            return 0, "", ""
        if args[:2] == ["buckets", "list"] and "-R" in args:
            listing = [{"type": "file", "path": k, "size": len(v)} for k, v in self.objects.items()]
            return 0, json.dumps(listing), ""
        if args[:2] == ["buckets", "list"]:
            entries = [{"id": bucket.bucket_id, "private": self.private}] if self.exists else []
            return 0, json.dumps(entries), ""
        assert args[:2] == ["buckets", "cp"], args
        source, target = args[2], args[3]
        prefix = f"hf://buckets/{bucket.bucket_id}/"
        if target.startswith(prefix):
            name = target[len(prefix):]
            assert bucket._token, "上传必须带凭据"
            if name in self.fail_upload:
                return 1, "", "注入的上传失败"
            self.objects[name] = Path(source).read_bytes()
            self.uploads.append(name)
            return 0, "", ""
        name = source[len(prefix):]
        env = bucket._env()
        self.anon_env.append(env)
        data = self.objects[name]
        Path(target).write_bytes(data[:-1] + b"\xff" if name in self.corrupt else data)
        return 0, "", ""


@pytest.fixture
def hf(tool, monkeypatch):
    return FakeHF(tool, monkeypatch)


def write_online_inputs(run):
    arrays = {"images": np.zeros((82, 256, 256, 3), np.uint8), "states": np.zeros((82, 8), np.float32),
              "wrist_image": np.zeros((256, 256, 3), np.uint8)}
    np.savez_compressed(run / "online_restore_inputs.npz", **arrays)
    record = dict(schema="stage2-online-restore-inputs-v1", exec_start_idx=17, exec_batch_lengths=[1, 16, 16, 16, 16],
                  prompt="合成验证提示", sources=[dict(path="仅诊断来源", sha256="a" * 64)],
                  payload=dict(path="online_restore_inputs.npz", sha256=ms.sha256_file(run / "online_restore_inputs.npz")),
                  tokenizer=dict(uri="gs://big_vision/paligemma_tokenizer.model",
                                 sha256="8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"),
                  arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=sha(v.tobytes()))
                          for k, v in arrays.items()})
    (run / "online_restore_inputs.json").write_text(json.dumps(record))


def publish_ready(run):
    """按当前文件内容重写 ready，用于构造 SHA 闭合、只有身份错配的快照。"""
    files = {p.relative_to(run).as_posix(): ms.sha256_file(p) for p in run.rglob("*")
             if p.is_file() and p.parent.name not in ("params", "d", "robomme") and not p.name.startswith("_")
             and p.name not in ("motion_snapshot.ready.json", "wandb_id.txt")}
    (run / "motion_snapshot.ready.json").write_text(json.dumps(dict(schema="wan-timepool-motion-snapshot-v1", files=files)))


def edit_prov(run, change):
    path = run / "motion_provenance.json"
    prov = json.loads(path.read_text())
    change(prov)
    path.write_text(json.dumps(prov))
    publish_ready(run)


def edit_history(run, change):
    """改 resolved yaml 并同步摘要与来源记录，只留身份错配。"""
    path = run / "history_config.resolved.yaml"
    cfg = OmegaConf.load(path)
    change(cfg)
    OmegaConf.save(cfg, path)
    digest = ms.sha256_file(path)
    (run / "history_config.resolved.sha256").write_text(digest + "\n")
    edit_prov(run, lambda p: p.update(resolved_sha256=digest))


def add_step(run, step, *, commit=True):
    root = run / str(step)
    (root / "params/d").mkdir(parents=True)
    (root / "assets/robomme").mkdir(parents=True)
    (root / "params/_METADATA").write_text(json.dumps({"tree": step}))
    (root / "params/d/abc").write_bytes(b"\x00" + bytes([step % 251]) * 64)
    (root / "assets/robomme/norm_stats.json").write_text("{}")
    metadata = {"init_timestamp_nsecs": 1}
    if commit:
        metadata["commit_timestamp_nsecs"] = step + 1
    (root / "_CHECKPOINT_METADATA").write_text(json.dumps(metadata))
    return root


def commit_step(run, step):
    path = run / str(step) / "_CHECKPOINT_METADATA"
    value = json.loads(path.read_text())
    value["commit_timestamp_nsecs"] = step + 1
    path.write_text(json.dumps(value))


@pytest.fixture
def store(tmp_path, tool):
    """--store-root 夹具：迷你 verified motion 表 + 正式 run 根的 12 文件快照。"""
    root = tmp_path / "store"
    motion = root / tool.MOTION_REL
    run = root / tool.TRAIN_PARENT / tool.RUN
    (run / "motion_evidence").mkdir(parents=True)
    (root / "v1-store/logs").mkdir(parents=True)
    motion.mkdir(parents=True)
    evidence = {role: json.dumps({"role": role, "status": "PASS"}).encode() for role in ROLES}

    def bind(prov):
        for role, blob in evidence.items():
            prov["validation"][role].update(sha256=sha(blob))

    meta = build_store(motion, provenance=bind)
    for role, blob in evidence.items():
        (run / f"motion_evidence/{role}.json").write_bytes(blob)
    shutil.copyfile(motion / ms.TIMEPOOL_WINDOW_MAP_RELPATH, run / "timepool_window_map.json")
    write_online_inputs(run)
    (run / "history_config.txt").write_text(tool.HISTORY_NAME + "\n")
    shutil.copyfile(ROOT / "src/mme_vla_suite/models/config/robomme" / tool.HISTORY_NAME,
                    run / "history_config.resolved.yaml")
    resolved = ms.sha256_file(run / "history_config.resolved.yaml")
    (run / "history_config.resolved.sha256").write_text(resolved + "\n")
    tp = meta["provenance"]
    prov = dict(encoder=None, vae=None, motion_enabled=True, timepool=tp, history_config_name=tool.HISTORY_NAME,
                resolved_sha256=resolved, framesamp_manifest_sha256="1" * 64, framesamp_store_meta_sha256="2" * 64,
                framesamp_root=str(root / tool.FRAMESAMP_REL), motion_root=str(motion),
                motion_manifest_sha256=tp["manifests"]["episode_manifest_sha256"],
                motion_table_sha256=tp["output"]["motion_table_sha256"],
                motion_index_sha256=tp["output"]["motion_index_sha256"],
                motion_store_meta_sha256=ms.sha256_file(motion / ms.META_RELPATH))
    (run / "motion_provenance.json").write_text(json.dumps(prov))
    # ready 不列的本地文件：不得进入上传集合。
    (run / "wandb_id.txt").write_text("abc\n")
    publish_ready(run)
    return types.SimpleNamespace(root=root, run=run, motion=motion, log=root / "v1-store/logs" / f"{tool.RUN}.log",
                                 export=root / "v1-store/exports" / f"hf-ckpt-{tool.RUN}")


def write_log(store, tail="", run_name=None):
    run_name = run_name or "orig100-ctx32x4x4-wan-timepool8x8-b128-60k"
    store.log.write_text(f"RUN={run_name}\nSTEPS=60000\nStep 1: loss=1\n{tail}")


def fill_steps(tool, store, steps=None):
    for step in tool.STEPS if steps is None else steps:
        add_step(store.run, step)


def inc(tool, store, *extra):
    return tool.cli(["--mode", "incremental", "--store-root", str(store.root), *extra])


def final(tool, store):
    return tool.cli(["--mode", "final", "--store-root", str(store.root)])


def tree_state(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_mode,
                                            ms.sha256_file(p) if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


def uploaded_steps(hf):
    seen = []
    for name in hf.uploads:
        head = name.split("/")[0]
        if head.isdecimal() and int(head) not in seen:
            seen.append(int(head))
    return seen


# ── 增量排序与等待 ───────────────────────────────────────────────────────────


def test_incremental_uploads_in_order_and_ends(tool, store, hf, capsys):
    write_log(store)
    fill_steps(tool, store)
    assert inc(tool, store) == 0
    out = capsys.readouterr().out
    assert uploaded_steps(hf) == list(tool.STEPS)
    assert f"INC_END run={tool.RUN} done=12" in out and "EXIT_CODE=0" in out
    assert "HF_VERIFY" not in out, "增量不得输出最终判定行"
    assert hf.tokens == [store.root / "v1-store/secrets/hf.env"]
    assert all("HF_TOKEN" not in env and env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1" for env in hf.anon_env)
    assert "wandb_id.txt" not in hf.objects
    assert {"README.md", "motion_snapshot.ready.json", "manifests/root.json", "manifests/step-59999.json"} <= set(hf.objects)
    manifest = json.loads(hf.objects["manifests/step-5000.json"])
    assert manifest["step"] == 5000 and manifest["bucket"] == tool.BUCKET
    assert "motion_provenance.json" in manifest["files"] and "5000/params/_METADATA" in manifest["files"]


def test_incremental_waits_for_earlier_step_before_later(tool, store, hf, capsys, monkeypatch):
    write_log(store)
    add_step(store.run, 5000, commit=False)
    add_step(store.run, 10000)
    polls = []

    def fake_sleep(_):
        polls.append(uploaded_steps(hf))
        if len(polls) == 1:
            commit_step(store.run, 5000)
            return
        fill_steps(tool, store, [s for s in tool.STEPS if s not in (5000, 10000)])

    monkeypatch.setattr(tool, "time", types.SimpleNamespace(sleep=fake_sleep))
    assert inc(tool, store) == 0
    assert polls[0] == [], "10000 已提交但 5000 未提交时不得先传"
    assert uploaded_steps(hf) == list(tool.STEPS)
    assert "INC_WAIT done=0 expected=12 next=5000" in capsys.readouterr().out


def test_resume_skips_done_steps(tool, store, hf, capsys):
    write_log(store)
    fill_steps(tool, store)
    hf.fail_upload.add("10000/params/_METADATA")
    assert inc(tool, store) == 1
    assert "TIMEPOOL_EXPORT=FAIL" in capsys.readouterr().out
    assert uploaded_steps(hf)[0] == 5000
    done = sorted(p.name for p in (store.export / "logs/uploaded").iterdir())
    assert "5000.done" in done and "10000.done" not in done
    hf.fail_upload.clear()
    hf.uploads.clear()
    assert inc(tool, store) == 0
    assert 5000 not in uploaded_steps(hf), "已有 .done 的步不重复上传"
    assert uploaded_steps(hf) == list(tool.STEPS[1:])


def test_resume_rejects_changed_stage(tool, store, hf, capsys):
    write_log(store)
    fill_steps(tool, store, [5000])
    hf.fail_upload.add("5000/params/_METADATA")
    assert inc(tool, store) == 1
    (store.export / "stage/5000/params/d/abc").write_bytes(b"\x01")
    hf.fail_upload.clear()
    capsys.readouterr()
    assert inc(tool, store) == 1
    assert "已有stage内容不符" in capsys.readouterr().out


# ── 未提交 checkpoint ───────────────────────────────────────────────────────


@pytest.mark.parametrize("metadata", ['{"init_timestamp_nsecs": 1}', '{"commit_timestamp_nsecs": 0}', '{"commit',
                                      '{"commit_timestamp_nsecs": "1"}'])
def test_uncommitted_step_not_packed(tool, store, hf, capsys, metadata):
    write_log(store, "EXIT_CODE=1\n")
    add_step(store.run, 5000)
    (store.run / "5000/_CHECKPOINT_METADATA").write_text(metadata)
    assert inc(tool, store) == 1
    assert not (store.export / "stage/5000").exists()
    assert not any(name.startswith("5000/") for name in hf.objects)
    assert "missing=[5000," in capsys.readouterr().out


def test_orbax_tmp_dir_not_packed(tool, store, hf, capsys):
    write_log(store, "EXIT_CODE=0\n")
    tmp_dir = store.run / "5000.orbax-checkpoint-tmp-99"
    tmp_dir.mkdir()
    (tmp_dir / "x").write_text("half")
    assert inc(tool, store) == 1
    assert not any("tmp" in name for name in hf.objects)


def test_tmp_file_inside_committed_step_rejected(tool, store, hf, capsys):
    write_log(store)
    add_step(store.run, 5000)
    (store.run / "5000/params/x.tmp").write_text("half")
    assert inc(tool, store) == 1
    assert "checkpoint含临时文件" in capsys.readouterr().out
    assert not any(name.startswith("5000/") for name in hf.objects)


# ── 身份错配 ─────────────────────────────────────────────────────────────────


def mutate_stage2_file(store, tool):
    # stage2 快照多一个 encoder 依赖文件，文件集合不再恰好相等。
    (store.run / "encoder_dependencies.json").write_text("{}")
    publish_ready(store.run)


def mutate_ready_schema(store, tool):
    ready = json.loads((store.run / "motion_snapshot.ready.json").read_text())
    ready["schema"] = "stage2-motion-snapshot-v1"
    (store.run / "motion_snapshot.ready.json").write_text(json.dumps(ready))


def mutate_file_bytes(store, tool):
    (store.run / "motion_evidence/WINDOW_EQ.json").write_text("{}")


def mutate_readout(store, tool):
    edit_prov(store.run, lambda p: p["timepool"].update(readout="stage2_pool4_concat_v1"))


def mutate_history_name(store, tool):
    edit_prov(store.run, lambda p: p.update(history_config_name="other.yaml"))


def mutate_history_readout(store, tool):
    edit_history(store.run, lambda c: c.motion.update(readout="stage2_pool4_concat_v1"))


def mutate_history_store(store, tool):
    edit_history(store.run, lambda c: c.motion.update(store_path="v1-store/datasets/other/motion-wan-timepool1024"))


def mutate_history_label(store, tool):
    (store.run / "history_config.txt").write_text("other.yaml\n")
    publish_ready(store.run)


def mutate_motion_root(store, tool):
    edit_prov(store.run, lambda p: p.update(motion_root="/somewhere/else"))


def mutate_framesamp_root(store, tool):
    edit_prov(store.run, lambda p: p.update(framesamp_root="/somewhere/else/framesamp-4x4"))


def mutate_meta_binding(store, tool):
    edit_prov(store.run, lambda p: p.update(motion_store_meta_sha256="0" * 64))


def mutate_table(store, tool):
    path = store.motion / ms.MOTION_TABLE_RELPATH
    path.write_bytes(path.read_bytes()[:-1] + b"\x7f")


def mutate_online(store, tool):
    record = json.loads((store.run / "online_restore_inputs.json").read_text())
    record["tokenizer"]["sha256"] = "0" * 64
    (store.run / "online_restore_inputs.json").write_text(json.dumps(record))
    publish_ready(store.run)


@pytest.mark.parametrize("mutate", [mutate_stage2_file, mutate_ready_schema, mutate_file_bytes, mutate_readout,
                                    mutate_history_name, mutate_history_readout, mutate_history_store,
                                    mutate_history_label, mutate_motion_root, mutate_framesamp_root,
                                    mutate_meta_binding, mutate_table, mutate_online],
                         ids=lambda f: f.__name__)
def test_identity_mismatch_rejected_before_upload(tool, store, hf, capsys, mutate):
    write_log(store)
    fill_steps(tool, store)
    mutate(store, tool)
    assert inc(tool, store) == 1
    out = capsys.readouterr().out
    assert "TIMEPOOL_EXPORT=FAIL" in out and "WAIT_USER_DECISION" in out and "EXIT_CODE=1" in out
    assert hf.uploads == [] and hf.tokens == [], "身份不符必须在读凭据和上传之前拒绝"


def test_wrong_run_log_rejected(tool, store, hf, capsys):
    write_log(store, run_name="other-run")
    assert inc(tool, store) == 1
    assert "训练日志run不符" in capsys.readouterr().out


def test_foreign_bucket_not_taken_over(tool, store, hf, capsys):
    write_log(store)
    fill_steps(tool, store)
    hf.exists = True
    assert inc(tool, store) == 1
    assert "拒绝接管" in capsys.readouterr().out and hf.uploads == []


def test_private_bucket_rejected(tool, store, hf, capsys):
    write_log(store)
    fill_steps(tool, store)
    hf.private = True
    assert inc(tool, store) == 1
    assert "bucket必须公开" in capsys.readouterr().out and hf.uploads == []


# ── 卫生扫描 ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("line,word", [("hf_" + "A" * 34, "疑似凭据"), ("/nfs/turbo/内部路径", "未裁决命中")])
def test_hygiene_blocks_step_before_copy(tool, store, hf, capsys, line, word):
    write_log(store)
    add_step(store.run, 5000)
    (store.run / "5000/assets/robomme/norm_stats.json").write_text(json.dumps({"note": line}))
    assert inc(tool, store) == 1
    out = capsys.readouterr().out
    assert word in out and line not in out, "凭据与内部路径原文不得进日志"
    assert not (store.export / "stage/5000").exists()
    assert not any(name.startswith("5000/") for name in hf.objects)


def test_hygiene_blocks_root_before_any_upload(tool, store, hf, capsys):
    write_log(store)
    record = json.loads((store.run / "online_restore_inputs.json").read_text())
    record["prompt"] = "coe-chaijy"
    (store.run / "online_restore_inputs.json").write_text(json.dumps(record))
    publish_ready(store.run)
    assert inc(tool, store) == 1
    assert "未裁决命中" in capsys.readouterr().out and hf.uploads == [] and hf.tokens == []


def test_hygiene_blocks_train_log_in_final(tool, store, hf, capsys):
    fill_steps(tool, store)
    write_log(store, "/data/hongzefu/内部路径\nEXIT_CODE=0\n")
    assert final(tool, store) == 1
    out = capsys.readouterr().out
    assert "未裁决命中: _run-meta/train.log P3" in out and "HF_VERIFY=PASS" not in out
    assert "_run-meta/train.log" not in hf.objects


# ── 读回与最终判定 ───────────────────────────────────────────────────────────


def test_final_pass_lines(tool, store, hf, capsys):
    fill_steps(tool, store)
    write_log(store, "END_UTC=x\nEXIT_CODE=0\n")
    assert final(tool, store) == 0
    out = capsys.readouterr().out
    total = sum(len(v) for v in hf.objects.values())
    assert f"HF_VERIFY=PASS bucket={tool.BUCKET} objects={len(hf.objects)} bytes={total} mismatches=0" in out
    assert "CKPT_UPLOAD=PASS steps=12" in out
    delivery = json.loads(hf.objects["delivery_manifest.json"])
    assert set(delivery["files"]) | {"delivery_manifest.json"} == set(hf.objects)
    assert delivery["steps"] == list(tool.STEPS)
    assert "_run-meta/train.log" in hf.objects


def test_final_after_incremental_reuses_uploads(tool, store, hf, capsys):
    fill_steps(tool, store)
    write_log(store)
    assert inc(tool, store) == 0
    hf.uploads.clear()
    write_log(store, "EXIT_CODE=0\n")
    assert final(tool, store) == 0
    assert set(hf.uploads) == {"_run-meta/train.log", "delivery_manifest.json"}


def test_readback_mismatch_fails_step(tool, store, hf, capsys):
    write_log(store)
    fill_steps(tool, store, [5000])
    hf.corrupt.add("5000/params/_METADATA")
    assert inc(tool, store) == 1
    assert "匿名SHA或字节数不符: 5000/params/_METADATA" in capsys.readouterr().out
    assert not (store.export / "logs/uploaded/5000.done").exists()


def test_final_readback_mismatch_is_fail(tool, store, hf, capsys):
    fill_steps(tool, store)
    write_log(store)
    assert inc(tool, store) == 0
    write_log(store, "EXIT_CODE=0\n")
    hf.corrupt.add("35000/params/d/abc")
    capsys.readouterr()
    assert final(tool, store) == 1
    out = capsys.readouterr().out
    assert "匿名SHA或字节数不符: 35000/params/d/abc" in out and "HF_VERIFY=PASS" not in out and "CKPT_UPLOAD=PASS" not in out


def test_final_extra_remote_object_is_fail(tool, store, hf, capsys):
    fill_steps(tool, store)
    write_log(store, "EXIT_CODE=0\n")
    hf.objects["stray.bin"] = b"x"
    assert final(tool, store) == 1
    out = capsys.readouterr().out
    assert "远端文件集合或字节数不符" in out and "CKPT_UPLOAD=PASS" not in out


def test_upload_failure_is_fail(tool, store, hf, capsys):
    write_log(store)
    fill_steps(tool, store, [5000])
    hf.fail_upload.add("5000/_CHECKPOINT_METADATA")
    assert inc(tool, store) == 1
    assert "上传失败: 5000/_CHECKPOINT_METADATA" in capsys.readouterr().out


# ── 缺步与训练结束 ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("tail", ["EXIT_CODE=0\n", "EXIT_CODE=1\n"])
def test_incremental_missing_step_after_train_end_fails(tool, store, hf, capsys, tail):
    write_log(store, tail)
    fill_steps(tool, store, [5000, 10000, 20000])
    assert inc(tool, store) == 1
    out = capsys.readouterr().out
    assert "增量未齐12步" in out and f"missing=[15000, 20000," in out and tail.strip() in out
    assert uploaded_steps(hf) == [5000, 10000], "已提交的前缀先传完，跳过缺口之后的步"


def test_incremental_last_step_committed_with_train_end_passes(tool, store, hf):
    write_log(store, "EXIT_CODE=0\n")
    fill_steps(tool, store)
    assert inc(tool, store) == 0


def test_killed_session_without_exit_code_fails(tool, store, hf, capsys, monkeypatch):
    write_log(store)
    fill_steps(tool, store, [5000])
    seen = []
    monkeypatch.setattr(tool, "session_alive", lambda name: seen.append(name) or False)
    assert inc(tool, store, "--train-tmux", "o16-tp-train-x") == 1
    assert seen == ["o16-tp-train-x"]
    assert "session_gone_without_exit_code" in capsys.readouterr().out


def test_alive_session_keeps_waiting(tool, store, hf, monkeypatch):
    write_log(store)
    fill_steps(tool, store, [5000])
    monkeypatch.setattr(tool, "session_alive", lambda name: True)

    def fake_sleep(_):
        fill_steps(tool, store, tool.STEPS[1:])

    monkeypatch.setattr(tool, "time", types.SimpleNamespace(sleep=fake_sleep))
    assert inc(tool, store, "--train-tmux", "o16-tp-train-x") == 0


@pytest.mark.parametrize("case", ["no_exit", "bad_exit", "missing_step", "extra_step"])
def test_final_requires_complete_training(tool, store, hf, capsys, case):
    fill_steps(tool, store)
    write_log(store, {"no_exit": "", "bad_exit": "EXIT_CODE=2\n"}.get(case, "EXIT_CODE=0\n"))
    if case == "missing_step":
        shutil.rmtree(store.run / "25000")
    if case == "extra_step":
        add_step(store.run, 6000)
    assert final(tool, store) == 1
    out = capsys.readouterr().out
    assert "HF_VERIFY=PASS" not in out and hf.uploads == [] and hf.tokens == []


def test_ready_missing_after_train_end_fails(tool, store, hf, capsys):
    write_log(store, "EXIT_CODE=1\n")
    (store.run / "motion_snapshot.ready.json").unlink()
    assert inc(tool, store) == 1
    assert "缺原子根快照ready" in capsys.readouterr().out


def test_ready_wait_then_continue(tool, store, hf, capsys, monkeypatch):
    write_log(store)
    fill_steps(tool, store)
    ready = store.run / "motion_snapshot.ready.json"
    blob = ready.read_bytes()
    ready.unlink()
    monkeypatch.setattr(tool, "time", types.SimpleNamespace(sleep=lambda _: ready.write_bytes(blob)))
    assert inc(tool, store) == 0
    assert "ROOT_WAIT=1" in capsys.readouterr().out


# ── 不写 train-runs ──────────────────────────────────────────────────────────


def test_train_runs_untouched(tool, store, hf):
    fill_steps(tool, store)
    write_log(store, "EXIT_CODE=0\n")
    run_parent = store.run.parent
    before = tree_state(run_parent)
    log_before = tree_state(store.log.parent)
    assert inc(tool, store) == 0
    assert final(tool, store) == 0
    assert tree_state(run_parent) == before
    assert tree_state(store.log.parent) == log_before


def test_failed_export_does_not_touch_train_runs(tool, store, hf):
    fill_steps(tool, store)
    write_log(store)
    hf.corrupt.add("5000/params/_METADATA")
    before = tree_state(store.run.parent)
    assert inc(tool, store) == 1
    assert tree_state(store.run.parent) == before


# ── 参数与 SMOKE ─────────────────────────────────────────────────────────────


def test_store_root_from_env(tool, store, hf, monkeypatch):
    write_log(store)
    fill_steps(tool, store)
    monkeypatch.setenv("O16_STORE_ROOT", str(store.root))
    assert tool.cli(["--mode", "incremental"]) == 0


def test_store_root_required(tool, capsys):
    assert tool.cli(["--mode", "incremental"]) == 1
    assert "必须给 --store-root" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [["--export-root", "/tmp/x"], ["--smoke-steps", "19"], ["--smoke-run-root", "/tmp/x"]])
def test_overrides_only_in_smoke(tool, store, hf, capsys, extra):
    assert inc(tool, store, *extra) == 1
    assert "只允许离线SMOKE" in capsys.readouterr().out


def test_smoke_requires_export_root(tool, store, monkeypatch, capsys):
    monkeypatch.setenv("SMOKE", "1")
    assert inc(tool, store) == 1
    assert "SMOKE必须显式给" in capsys.readouterr().out
    assert not store.export.exists()


def smoke_guard(tool, monkeypatch):
    monkeypatch.setenv("SMOKE", "1")
    monkeypatch.setattr(tool.io, "load_hf_token", lambda *a: pytest.fail("SMOKE 不得读取凭据"))
    monkeypatch.setattr(tool.io.HFBucket, "_run", lambda *a: pytest.fail("SMOKE 不得联网"))


def test_smoke_no_credentials_no_network(tool, store, tmp_path, monkeypatch, capsys):
    write_log(store)
    fill_steps(tool, store, [5000])
    smoke_guard(tool, monkeypatch)
    out_root = tmp_path / "smoke-out"
    assert inc(tool, store, "--export-root", str(out_root)) == 0
    out = capsys.readouterr().out
    assert "SMOKE=PASS kind=checkpoint mode=incremental steps=1" in out and "credentials=0" in out
    assert (out_root / "stage/5000/params/_METADATA").is_file()
    assert not store.export.exists()


def test_smoke_other_run_root_and_steps(tool, store, tmp_path, monkeypatch, capsys):
    other = store.run.parent / "o16-smoke-timepool-x"
    shutil.copytree(store.run, other)
    add_step(other, 19)
    smoke_guard(tool, monkeypatch)
    before = tree_state(other)
    code = inc(tool, store, "--export-root", str(tmp_path / "o"), "--smoke-run-root", str(other), "--smoke-steps", "19")
    assert code == 0, capsys.readouterr().out
    assert "SMOKE=PASS kind=checkpoint mode=incremental steps=1" in capsys.readouterr().out
    assert tree_state(other) == before
    assert (tmp_path / "o/stage/manifests/step-19.json").is_file()


def test_entry_script_runs_from_own_tree(store, tmp_path):
    """真实子进程走入口：不带 PYTHONPATH 也必须加载本源码树，且不碰凭据。"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(O16_PYTHON=sys.executable, SMOKE="1", O16_STORE_ROOT=str(store.root))
    add_step(store.run, 5000)
    write_log(store)
    proc = subprocess.run(["bash", str(EXPORT / "run_orig16_ctx_timepool_60k_ckpt_inc.sh"), "--export-root", str(tmp_path / "o")],
                          env=env, cwd=tmp_path, capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"CODE_ROOT={ROOT} " in proc.stdout and "SMOKE=PASS" in proc.stdout


def test_entry_script_requires_explicit_python(store):
    env = {k: v for k, v in os.environ.items() if k != "O16_PYTHON"}
    proc = subprocess.run(["bash", str(EXPORT / "run_orig16_ctx_timepool_60k_ckpt_export.sh"), "--store-root", str(store.root)],
                          env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0 and "O16_PYTHON" in proc.stderr


def test_readme_and_allowlist_are_public_safe(tool):
    tool.scan({"README.md": tool.README}, tool.ALLOWLIST)
    names = {entry["path_glob"] for entry in json.loads(tool.ALLOWLIST.read_text())["entries"]}
    assert "README.md" not in names and "encoder_dependencies.json" not in names
    text = tool.README.read_text()
    assert tool.BUCKET in text and tool.RUN in text and tool.READOUT in text and tool.ENCODER_KIND in text and tool.LAYOUT in text
