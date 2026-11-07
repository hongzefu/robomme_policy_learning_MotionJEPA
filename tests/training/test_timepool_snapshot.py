"""run 根 timepool 快照：真实写入端必须通过已合入的消费端 verify_timepool_snapshot，并覆盖拒绝路径与入口分派。"""
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/dataset"))
sys.path.insert(0, str(ROOT / "tests/datastore"))
from test_export_wan_timepool import build_tree, run
from test_pack_motion_timepool import cli, pack_argv, verify_argv, write_records
from mme_vla_suite.datastore import framesamp_store as fs
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.policies import motion_client as mc
from mme_vla_suite.shared import wan_timepool as tp
from test_stage2_snapshot import snapshot  # noqa: F401  stage2 夹具，用于回归

spec =importlib.util.spec_from_file_location("train_timepool_snapshot", ROOT / "scripts/training/train.py")
train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train
spec.loader.exec_module(train)

BASE_YAML = ROOT / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-motion-mixed8.yaml"


def write_online_inputs(root):
    arrays = {"images": np.zeros((82, 256, 256, 3), np.uint8), "states": np.zeros((82, 8), np.float32),
              "wrist_image": np.zeros((256, 256, 3), np.uint8)}
    np.savez_compressed(root / "online_restore_inputs.npz", **arrays)
    portable = dict(schema="stage2-online-restore-inputs-v1", exec_start_idx=17, exec_batch_lengths=[33, 32],
                    prompt="合成验证提示",
                    payload=dict(path="online_restore_inputs.npz", sha256=ms.sha256_file(root / "online_restore_inputs.npz")),
                    sources=[dict(path="仅诊断来源", sha256="a" * 64)],
                    tokenizer=dict(uri="gs://big_vision/paligemma_tokenizer.model", sha256="b" * 64),
                    arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype),
                                      sha256=ms.sha256_bytes(value.tobytes())) for key, value in arrays.items()})
    (root / "online_restore_inputs.json").write_text(json.dumps(portable))


def write_frame_store(root, manifest, manifest_path):
    """只造 StoreMeta.load 需要的帧表元数据；快照路径不读帧数据。"""
    (root / "meta").mkdir(parents=True)
    layout = fs.SPECS[fs.LAYOUT]
    rows = manifest["totals"]["timesteps"]
    meta = dict(layout=layout.layout, byte_order=fs.BYTE_ORDER, array_order=fs.ARRAY_ORDER,
                bf16_encoding=fs.BF16_ENCODING, status="verified", manifest_scope="full", num_rows=rows,
                num_exec_samples=manifest["totals"]["exec_samples"], num_pos_rows=114,
                manifest_sha256=manifest["sha256"], manifest_path=str(manifest_path), source_dataset_root=str(root),
                tables={layout.image_key: dict(row_shape=list(layout.image_row_shape), dtype="bfloat16",
                                               row_bytes=layout.image_row_bytes, part_dir=layout.image_part_dir),
                        layout.pos_key: dict(row_shape=list(layout.pos_row_shape), dtype="float32",
                                             row_bytes=layout.pos_row_bytes, relpath=layout.pos_table_relpath),
                        fs.STATE_KEY: dict(row_shape=list(fs.STATE_ROW_SHAPE), dtype="float32", row_bytes=fs.STATE_ROW_BYTES)},
                parts=[dict(index=0, start_row=0, num_rows=rows, bytes=rows * layout.image_row_bytes,
                            path=f"{layout.image_part_dir}/part_000.bf16.bin", sha256="a" * 64,
                            head_tail_digest="b" * 64, full_covered=True)])
    (root / fs.META_RELPATH).write_text(json.dumps(meta))


@pytest.fixture
def item(tmp_path, monkeypatch):
    """真实走 export → pack → verify 得到 verified 迷你库，验证根同时放在线输入。"""
    t = build_tree(tmp_path)
    t.store = tmp_path / "store" / "motion-wan-timepool1024"
    t.records = tmp_path / "records"
    run(t)
    cli(monkeypatch, pack_argv(t))
    write_records(t)
    cli(monkeypatch, verify_argv(t, records=True))
    write_online_inputs(t.records)
    frames = tmp_path / "framesamp"
    write_frame_store(frames, t.manifest, t.manifest_path)
    monkeypatch.delenv("MMEVLA_MOTION_STORE", raising=False)
    monkeypatch.delenv("MMEVLA_FRAMESAMP_MANIFEST", raising=False)
    history = OmegaConf.load(BASE_YAML)
    history.motion.update(source_shape=[1, 1024], readout=tp.TIMEPOOL_READOUT, encoder_kind=tp.TIMEPOOL_ENCODER_KIND,
                          num_tokens=1, dim=1024, budget=96, stride=32, store_path=str(t.store))
    for key in ("source_run", "checkpoint_sha256"):
        history.motion.pop(key, None)
    root = tmp_path / "run"
    root.mkdir()
    config = SimpleNamespace(checkpoint_dir=root, model=SimpleNamespace(history_config="合成配置.yaml"))
    return SimpleNamespace(t=t, frames=frames, history=history, root=root, config=config)


def init(item):
    train.init_history_config(item.config, item.history, framesamp_root=item.frames)


def published_then_reopened(item):
    """走 init 拿到真实来源载荷并撤掉 ready，供直接调用写入端的拒绝用例。"""
    init(item)
    prov = json.loads((item.root / "motion_provenance.json").read_text())
    (item.root / "motion_snapshot.ready.json").unlink()
    return prov


def test_init_history_snapshot_passes_consumer(item):
    init(item)
    # 硬验收：真实写入端的产物必须通过已合入的消费端
    prov = mc.verify_timepool_snapshot(item.root)
    source = ms.MotionMeta.load(item.t.store)
    assert set(prov) == mc.TIMEPOOL_SNAPSHOT_PROV_KEYS and len(prov) == 14
    assert prov["vae"] is None and prov["encoder"] is None
    assert prov["timepool"] == source.provenance
    assert prov["motion_table_sha256"] == source.provenance["output"]["motion_table_sha256"]
    ready = json.loads((item.root / "motion_snapshot.ready.json").read_text())
    assert ready["schema"] == "wan-timepool-motion-snapshot-v1"
    assert len(ready["files"]) == 12 and "motion_snapshot.ready.json" not in ready["files"]
    for role in ms.TIMEPOOL_VALIDATION_ROLES:
        assert (item.root / f"motion_evidence/{role}.json").read_bytes() == (item.t.records / f"{role}.json").read_bytes()
    assert (item.root / "timepool_window_map.json").read_bytes() == (item.t.store / ms.TIMEPOOL_WINDOW_MAP_RELPATH).read_bytes()
    for name in ("online_restore_inputs.json", "online_restore_inputs.npz"):
        assert (item.root / name).read_bytes() == (item.t.records / name).read_bytes()


def test_no_encoder_artifacts(item):
    init(item)
    assert not (item.root / "encoder_dependencies.json").exists()
    assert not (item.root / "encoder_dependencies").exists()
    assert not list(item.root.rglob("*.pt"))


def test_refuses_republish(item):
    init(item)
    ready_bytes = (item.root / "motion_snapshot.ready.json").read_bytes()
    with pytest.raises(FileExistsError):
        init(item)
    assert (item.root / "motion_snapshot.ready.json").read_bytes() == ready_bytes


@pytest.mark.parametrize("role", ms.TIMEPOOL_VALIDATION_ROLES)
def test_changed_evidence_rejected(item, role):
    prov = published_then_reopened(item)
    (item.t.records / f"{role}.json").write_text("{}")
    with pytest.raises(ValueError, match=f"{role} 证据"):
        train.write_timepool_motion_snapshot(item.root, prov)
    assert not (item.root / "motion_snapshot.ready.json").exists()


def test_changed_window_map_rejected(item):
    prov = published_then_reopened(item)
    (item.t.store / ms.TIMEPOOL_WINDOW_MAP_RELPATH).write_text("{}")
    with pytest.raises(ValueError, match="逐窗映射"):
        train.write_timepool_motion_snapshot(item.root, prov)
    assert not (item.root / "motion_snapshot.ready.json").exists()


@pytest.mark.parametrize("fault", ["payload_sha", "array_sha", "schema"])
def test_bad_online_inputs_rejected(item, fault):
    prov = published_then_reopened(item)
    path = item.t.records / "online_restore_inputs.json"
    value = json.loads(path.read_text())
    if fault == "payload_sha":
        value["payload"]["sha256"] = "0" * 64
    elif fault == "array_sha":
        value["arrays"]["states"]["sha256"] = "0" * 64
    else:
        value["schema"] = "其他"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="timepool 可携带在线输入"):
        train.write_timepool_motion_snapshot(item.root, prov)
    assert not (item.root / "motion_snapshot.ready.json").exists()


def corrupt_extra_key(prov):
    prov["额外"] = 1


def corrupt_encoder(prov):
    prov["encoder"] = {"run_name": "x"}


def corrupt_table(prov):
    prov["motion_table_sha256"] = "0" * 64


def corrupt_manifest(prov):
    prov["motion_manifest_sha256"] = "0" * 64


def corrupt_index(prov):
    prov["motion_index_sha256"] = "0" * 64


def corrupt_validation(prov):
    prov["timepool"]["validation"] = None


def corrupt_resolved(prov):
    prov["resolved_sha256"] = "0" * 64


@pytest.mark.parametrize("change", [corrupt_extra_key, corrupt_encoder, corrupt_table, corrupt_manifest, corrupt_index,
                                    corrupt_validation, corrupt_resolved])
def test_bad_payload_rejected(item, change):
    prov = published_then_reopened(item)
    change(prov)
    with pytest.raises(ValueError):
        train.write_timepool_motion_snapshot(item.root, prov)
    assert not (item.root / "motion_snapshot.ready.json").exists()


def test_failed_snapshot_never_publishes_ready(item, monkeypatch):
    original = train._atomic_stage2_file

    def fail(path, data):
        if path.name == "motion_provenance.json":
            raise OSError("故障注入")
        return original(path, data)

    monkeypatch.setattr(train, "_atomic_stage2_file", fail)
    with pytest.raises(OSError):
        init(item)
    assert not (item.root / "motion_snapshot.ready.json").exists()


def test_consumer_rejects_tampered_copy(item):
    init(item)
    (item.root / "motion_evidence/POOL_TABLE_EQ.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA 不符"):
        mc.verify_timepool_snapshot(item.root)


def test_train_head_not_compared_with_current_head(item):
    """train_head 是建库时 HEAD，可以早于当前检出；祖先与零差异由起跑检查器核，写入端不比较。"""
    prov = published_then_reopened(item)
    prov["timepool"]["train_head"] = "0" * 40
    train.write_timepool_motion_snapshot(item.root, prov)
    assert mc.verify_timepool_snapshot(item.root)["timepool"]["train_head"] == "0" * 40


@pytest.mark.parametrize("readout,steps,root,enabled,want", [
    (tp.TIMEPOOL_READOUT, 20, "/记录", True, True),
    ("stage2_pool4_concat_v1", 20, "/记录", True, True),
    ("stage2_pool4_split_v1", 20, "/记录", True, True),
    ("其他", 20, "/记录", True, False),
    (tp.TIMEPOOL_READOUT, 60000, "/记录", True, False),
    (tp.TIMEPOOL_READOUT, 20, None, True, False),
    (tp.TIMEPOOL_READOUT, 20, "/记录", False, False),
])
def test_pre_save_reference_dispatch(readout, steps, root, enabled, want):
    assert train._wants_pre_save_reference(steps, root, dict(enabled=enabled, readout=readout)) is want


def test_pre_save_reference_dispatch_without_motion():
    assert train._wants_pre_save_reference(20, "/记录", None) is False


# ── stage2 路径回归：抽共用校验后产物、报错文本不变，且两种快照互不串用 ──────────────


def test_stage2_snapshot_unchanged_and_refused_by_timepool_consumer(snapshot):
    root, registered, provenance = snapshot
    ready = train.write_stage2_motion_snapshot(root, provenance, registered_root=registered)
    assert json.loads((root / "motion_snapshot.ready.json").read_text())["schema"] == "stage2-motion-snapshot-v1"
    assert len(ready["files"]) == 12 and "encoder_dependencies.json" in ready["files"]
    for name in ("online_restore_inputs.json", "online_restore_inputs.npz"):
        assert (root / name).read_bytes() == (root.parent / name).read_bytes()
    with pytest.raises(ValueError, match="ready schema"):
        mc.verify_timepool_snapshot(root)


def test_stage2_online_input_error_text_unchanged(snapshot):
    root, registered, provenance = snapshot
    path = root.parent / "online_restore_inputs.json"
    value = json.loads(path.read_text())
    value["payload"]["sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="^stage2 可携带在线输入 NPZ SHA 不符$"):
        train.write_stage2_motion_snapshot(root, provenance, registered_root=registered)


def test_timepool_writer_refuses_stage2_payload(snapshot):
    root, _, provenance = snapshot
    with pytest.raises(ValueError, match="键集合"):
        train.write_timepool_motion_snapshot(root, provenance)
    assert not (root / "motion_snapshot.ready.json").exists()
