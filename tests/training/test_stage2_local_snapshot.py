"""1114 计划短测故障回归：local-built 训练侧根快照来源须带 raw_dir／manifest_sha256，快照与下游消费端都能读通。"""
import copy
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/dataset"))
sys.path.insert(0, str(ROOT / "tests/datastore"))
sys.path.insert(0, str(ROOT / "scripts/training"))
from test_pack_motion_local_built import built, local_fx, certify, verify_args  # noqa: F401  （复用 S2 合成库夹具）
import pack_motion_store as pack
from test_motion_store_stage2 import stage2_identity
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.policies import motion_client as C


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train = _load("train_stage2_local_snapshot", "scripts/training/train.py")
ckcheck = _load("check_stage2_checkpoint_local_snapshot", "scripts/training/check_stage2_checkpoint.py")
TOKENIZER_SHA = json.loads((ROOT / "scripts/assets/ASSETS_LOCK.json").read_text())["assets"]["paligemma_tokenizer"]["sha256"]


def _run_prov(root):
    """仿 init_history_config：run 级字段（只取快照要用的几项）。"""
    (root / "history_config.txt").write_text("perceptual-framesamp-modul-8frame-8x8-motion-stage2-1x1024-4task-rgbstate.yaml")
    (root / "history_config.resolved.yaml").write_text("motion: {enabled: true}\n")
    sha = ms.sha256_file(root / "history_config.resolved.yaml")
    (root / "history_config.resolved.sha256").write_text(sha + "\n")
    return {"history_config_name": "合成", "motion_enabled": True, "resolved_sha256": sha,
            "motion_table_sha256": None, "motion_index_sha256": None, "vae": None, "encoder": None}


def _online_inputs(validation):
    arrays = {"images": np.zeros((82, 256, 256, 3), np.uint8), "states": np.zeros((82, 8), np.float32),
              "wrist_image": np.zeros((256, 256, 3), np.uint8)}
    np.savez_compressed(validation / "online_restore_inputs.npz", **arrays)
    record = dict(schema="stage2-online-restore-inputs-v1", exec_start_idx=17, exec_batch_lengths=[1, 16, 16, 16, 16],
                  prompt="合成验证提示",
                  payload=dict(path="online_restore_inputs.npz", sha256=ms.sha256_file(validation / "online_restore_inputs.npz")),
                  sources=[dict(path="仅诊断来源", sha256="a" * 64)],
                  tokenizer=dict(uri="gs://big_vision/paligemma_tokenizer.model", sha256=TOKENIZER_SHA),
                  arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=ms.sha256_bytes(v.tobytes()))
                          for k, v in arrays.items()})
    (validation / "online_restore_inputs.json").write_text(json.dumps(record))


@pytest.fixture
def local_store(local_fx):
    pack.cmd_pack(local_fx.args)
    certify(local_fx)
    pack.cmd_verify(verify_args(local_fx))
    _online_inputs(local_fx.validation)
    return local_fx, ms.MotionMeta.load(pathlib.Path(local_fx.args.out)).provenance


def test_local_training_provenance_carries_raw_dir_and_manifest(local_store):
    _, store_prov = local_store
    assert "raw_dir" in store_prov and store_prov["raw_dir"] is None
    prov = train.stage2_snapshot_provenance({"motion_enabled": True}, store_prov)
    assert prov["raw_dir"] is None and prov["manifest_sha256"] == store_prov["manifest_sha256"]
    ms.validate_stage2_provenance(prov)
    # 修复前的组装（不带 raw_dir）就是短测实跑的失败形态。
    old = {k: v for k, v in prov.items() if k not in train.STAGE2_LOCAL_SNAPSHOT_KEYS}
    with pytest.raises(ValueError, match="raw_dir 必须为 null"):
        ms.validate_stage2_provenance(old)


def test_local_snapshot_written_and_consumers_pass(local_store, tmp_path):
    fx, store_prov = local_store
    run = tmp_path / "run"
    run.mkdir()
    prov = train.stage2_snapshot_provenance(_run_prov(run), store_prov)
    ready = train.write_stage2_motion_snapshot(run, prov, registered_root=fx.enc_dir.parent)
    assert (run / "motion_snapshot.ready.json").is_file()
    saved = json.loads((run / "motion_provenance.json").read_text())
    assert saved["stage2_compat"]["evidence"]["wan_latents_bin_sha256"]["path"] == ms.BIN_SHA256_LIST_SNAPSHOT_PATH
    assert set(ready["files"]) >= {f"motion_evidence/{r}.json" for r in ms.LOCAL_EVIDENCE_ROLES}
    ms.validate_stage2_provenance(saved)
    C.validate_stage2_evidence(saved, run)               # 评估侧 policy_config → MotionEncoderClient 同一校验
    ckcheck.verify_snapshot(run)                          # check_stage2_checkpoint smoke／delivery 根快照校验
    ckcheck.load_online_inputs(run)


def test_local_bin_list_path_still_strict(local_store):
    _, store_prov = local_store
    prov = train.stage2_snapshot_provenance({}, store_prov)
    prov["stage2_compat"]["evidence"]["wan_latents_bin_sha256"]["path"] = "motion_evidence/other.json"
    with pytest.raises(ValueError, match="逐段摘要清单必须是固定文件名"):
        ms.validate_stage2_provenance(prov)


def test_non_local_sources_add_no_keys():
    identity = stage2_identity()
    assert not ms.stage2_uses_local(identity)
    assert train.stage2_local_snapshot_fields(identity) == {}
    base = {"motion_enabled": True}
    store = dict(copy.deepcopy(identity), raw_dir="/历史/raw", manifest_sha256="c" * 64,
                 source_grid_sha256="d" * 64, latents_metadata_sha256="e" * 64)
    prov = train.stage2_snapshot_provenance(base, store)
    assert "raw_dir" not in prov and "manifest_sha256" not in prov
    assert set(prov) == {"motion_enabled", *ms.STAGE2_GROUPS, "source_grid_sha256", "latents_metadata_sha256"}
