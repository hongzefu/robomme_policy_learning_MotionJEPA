"""pack/verify 的零参数池化分支：真实走 export→pack→verify 小链路，含错配与崩溃语义。"""
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dataset"))
sys.path.insert(0, str(ROOT / "tests/dataset"))
import pack_motion_store as pack
from test_export_wan_timepool import build_tree, dump, load, run
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.shared import wan_timepool as tp

HEAD = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def pack_argv(t, **extra):
    argv = ["pack", "--layout", "wan-timepool", "--manifest", str(t.manifest_path), "--tokens", str(t.out),
            "--latents", str(t.latents), "--raw-dir", str(t.raw), "--out", str(t.store), "--train-head", HEAD]
    for flag, value in extra.items():
        flag = "--" + flag.replace("_", "-")
        argv += [flag] if value is True else [flag, str(value)]
    return argv


def cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["pack_motion_store.py", *argv])
    pack.main()


def write_records(t, *, skip=(), verdicts=None, extra=None):
    t.records.mkdir(exist_ok=True)
    for role in ms.TIMEPOOL_VALIDATION_ROLES:
        if role in skip:
            continue
        value = {"role": role, "verdict": (verdicts or {}).get(role, f"{role}=PASS rows=8"), **(extra or {})}
        dump(t.records / f"{role}.json", value)


def verify_argv(t, *, records=False, resume=True):
    argv = ["verify", "--store", str(t.store), "--tokens", str(t.out)]
    if records:
        argv += ["--gate-records-dir", str(t.records)]
    if resume:
        argv += ["--resume"]
    return argv


@pytest.fixture
def exported(tmp_path):
    t = build_tree(tmp_path)
    t.store = tmp_path / "store" / "motion-wan-timepool1024"
    t.records = tmp_path / "records"
    run(t)
    return t


@pytest.fixture
def packed(exported, monkeypatch):
    cli(monkeypatch, pack_argv(exported))
    return exported


def test_pack_writes_packed_with_null_validation_and_map(packed):
    meta = ms.MotionMeta.load(packed.store)
    raw = meta.raw
    assert raw["status"] == "packed" and raw["layout"] == tp.TIMEPOOL_LAYOUT and meta.row_shape == (1, 1024)
    prov = meta.provenance
    assert prov["validation"] is None and prov["encoder_ckpt"] is None and prov["train_head"] == HEAD
    table_sha = ms.sha256_file(packed.store / ms.MOTION_TABLE_RELPATH)
    assert prov["output"] == dict(motion_table_sha256=table_sha, motion_index_sha256=meta.motion_index_sha256)
    map_bytes = (packed.store / ms.TIMEPOOL_WINDOW_MAP_RELPATH).read_bytes()
    assert map_bytes == (packed.out / pack.TIMEPOOL_EXPORT_MAP).read_bytes()
    assert prov["window_map"]["sha256"] == ms.sha256_bytes(map_bytes)
    assert prov["wan_cache"]["restored_receipt_file_sha256"] == ms.sha256_file(packed.restored_path)
    assert prov["routes"]["counts"]["fill"] == packed.expected["fill"]
    assert prov["routes"]["numeric_modes"]["copy"] == {"cudnn_conv_tf32": True}
    assert load(packed.store / ms.INDEX_RELPATH)["mj_repo_commit"] is None
    ms.run_full_checks(meta)                    # 逐行字节与映射、池化源码一致
    assert (packed.store / ms.LOCK_RELPATH).exists()


def test_packed_is_not_verified_for_consumers(packed):
    meta = ms.MotionMeta.load(packed.store)
    with pytest.raises(RuntimeError, match="未通过 verify"):
        ms.require_verified(meta)
    with pytest.raises(RuntimeError, match="pack.lock"):
        ms.require_no_pack_lock(packed.store)


@pytest.mark.parametrize("extra", [dict(encoder_run_dir="/tmp/x"), dict(latent_grid_subsample=2),
                                   dict(stage2_validation="/tmp/x"), dict(stage2_source_pin="/tmp/x"),
                                   dict(cache_import_receipt="/tmp/x"), dict(restore_receipt="/tmp/x"),
                                   dict(expected_ckpt_sha256="a" * 64), dict(oracle_report="/tmp/x"),
                                   dict(force_break_lock=True)])
def test_pack_rejects_stage2_and_encoder_arguments(exported, monkeypatch, extra):
    with pytest.raises(ValueError, match="不接受"):
        cli(monkeypatch, pack_argv(exported, **extra))
    assert not exported.store.exists()


def test_cli_spelling_is_short_alias_not_layout_name(exported, monkeypatch):
    argv = pack_argv(exported)
    argv[argv.index("wan-timepool")] = tp.TIMEPOOL_LAYOUT
    with pytest.raises(SystemExit):
        cli(monkeypatch, argv)


@pytest.mark.parametrize("head", ["0" * 40, "abc", None])
def test_pack_train_head_must_equal_git_head(exported, monkeypatch, head):
    argv = pack_argv(exported)
    position = argv.index("--train-head")
    if head is None:
        del argv[position:position + 2]
    else:
        argv[position + 1] = head
    with pytest.raises(ValueError, match="train-head"):
        cli(monkeypatch, argv)
    assert not exported.store.exists()


def test_train_head_rejected_on_other_layouts(exported, monkeypatch):
    argv = ["pack", "--manifest", str(exported.manifest_path), "--tokens", str(exported.out),
            "--latents", str(exported.latents), "--raw-dir", str(exported.raw), "--out", str(exported.store),
            "--encoder-run-dir", "/tmp/x", "--train-head", HEAD]
    with pytest.raises(ValueError, match="train-head"):
        cli(monkeypatch, argv)


def test_legacy_and_stage2_still_require_encoder_run_dir(exported, monkeypatch):
    for layout in ("legacy", "stage2"):
        argv = ["pack", "--layout", layout, "--manifest", str(exported.manifest_path), "--tokens", str(exported.out),
                "--latents", str(exported.latents), "--raw-dir", str(exported.raw), "--out", str(exported.store)]
        with pytest.raises(SystemExit):
            cli(monkeypatch, argv)


def test_pack_rejects_source_h5_change(exported, monkeypatch):
    (exported.raw / "record_dataset_T.h5").write_bytes(b"synthetic-h6")
    with pytest.raises(ValueError, match="原 H5"):
        cli(monkeypatch, pack_argv(exported))


def test_pack_rejects_receipt_changed_after_export(exported, monkeypatch):
    receipt = load(exported.receipt_path)
    receipt["windows"][0]["latent_sha256"] = "0" * 64
    dump(exported.receipt_path, receipt)
    with pytest.raises(ValueError, match="已变动"):
        cli(monkeypatch, pack_argv(exported))


def test_pack_rejects_tampered_token_and_extra_segment(exported, monkeypatch):
    item = exported.segments[0]["key"]
    path = exported.out / f"{item}.f32.bin"
    original = path.read_bytes()
    path.write_bytes(original[:-4] + b"\0\0\0\0")
    with pytest.raises(ValueError, match="sha256"):
        cli(monkeypatch, pack_argv(exported))
    path.write_bytes(original)
    (exported.out / "T_ep9_demo.f32.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="段集合"):
        cli(monkeypatch, pack_argv(exported, resume=True))


def test_read_segment_tokens_keeps_stage2_contract_separate(exported):
    key = exported.segments[0]["key"]
    ng = load(exported.out / f"{key}.metadata.json")["num_grid"]
    # 新 token 不能按 stage2 行合同读，stage2 元数据也不能按新合同读。
    with pytest.raises(ValueError, match="stage2 行合同"):
        pack.read_segment_tokens(exported.out, key, ng, spec=ms.LAYOUT_SPECS[ms.STAGE2_LAYOUT])
    data, meta = pack.read_segment_tokens(exported.out, key, ng, spec=ms.TIMEPOOL_LAYOUT_SPEC)
    assert len(data) == ng * 4096 and meta["readout"] == tp.TIMEPOOL_READOUT
    stage2_meta = {**meta, "schema": "stage2-memory-export-v1"}
    dump(exported.out / f"{key}.metadata.json", stage2_meta)
    with pytest.raises(ValueError, match="零参数池化行合同"):
        pack.read_segment_tokens(exported.out, key, ng, spec=ms.TIMEPOOL_LAYOUT_SPEC)


def test_verify_without_records_stays_packed_and_locked(packed, monkeypatch, capsys):
    before = (packed.store / ms.META_RELPATH).read_bytes()
    cli(monkeypatch, verify_argv(packed))
    assert "VERIFY_MOTION=PASS" in capsys.readouterr().out
    assert (packed.store / ms.META_RELPATH).read_bytes() == before
    assert (packed.store / ms.LOCK_RELPATH).exists()
    assert not (packed.store / ms.ROW_DIGESTS_RELPATH).exists()


def test_verify_with_records_sets_verified_and_backfills_validation(packed, monkeypatch, capsys):
    write_records(packed)
    cli(monkeypatch, verify_argv(packed, records=True))
    meta = ms.MotionMeta.load(packed.store)
    assert meta.status == "verified"
    ms.require_verified(meta)
    ms.require_no_pack_lock(packed.store)
    ms.run_full_checks(meta)
    for role, record in meta.provenance["validation"].items():
        path = packed.records / f"{role}.json"
        assert record == dict(path=str(path.resolve()), sha256=ms.sha256_file(path), verdict=f"{role}=PASS rows=8")
    assert meta.raw["verify"]["demo_tail"] == packed.expected["demo_tail"]
    assert meta.raw["row_digests"]["covered_rows"] == meta.num_rows
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith(f"VERIFY_MOTION=PASS scanned={meta.num_rows} ")


@pytest.mark.parametrize("fault", ["missing", "fail", "wrong_role", "stale_table", "stale_index", "bad_json"])
def test_bad_gate_records_keep_packed(packed, monkeypatch, fault):
    if fault == "missing":
        write_records(packed, skip=("POOL_TABLE_EQ",))
    elif fault == "fail":
        write_records(packed, verdicts={"WINDOW_EQ": "WINDOW_EQ=FAIL windows=1"})
    elif fault == "wrong_role":
        write_records(packed, verdicts={"WINDOW_EQ": "POOL_CONTRACT=PASS"})
    elif fault == "stale_table":
        write_records(packed, extra=dict(store_table_sha256="0" * 64))
    elif fault == "stale_index":
        write_records(packed, extra=dict(motion_index_sha256="0" * 64))
    else:
        write_records(packed)
        (packed.records / "DATA_LINEAGE.json").write_text("{坏")
    before = (packed.store / ms.META_RELPATH).read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        cli(monkeypatch, verify_argv(packed, records=True))
    assert (packed.store / ms.META_RELPATH).read_bytes() == before
    assert ms.MotionMeta.load(packed.store).status == "packed"


def test_crash_at_meta_write_leaves_packed_with_lock(packed, monkeypatch):
    """模拟写 tmp 后、改名前崩溃：磁盘上的 store_meta 仍是原 packed 字节，半成品不可能被当 verified。"""
    write_records(packed)
    before = (packed.store / ms.META_RELPATH).read_bytes()
    real_replace = pack.os.replace

    def crash(src, dst):
        if pathlib.Path(dst).name == pathlib.Path(ms.META_RELPATH).name:
            raise OSError("模拟崩溃")
        return real_replace(src, dst)

    monkeypatch.setattr(pack.os, "replace", crash)
    with pytest.raises(OSError, match="模拟崩溃"):
        cli(monkeypatch, verify_argv(packed, records=True))
    assert (packed.store / ms.META_RELPATH).read_bytes() == before
    meta = ms.MotionMeta.load(packed.store)
    assert meta.status == "packed" and meta.provenance["validation"] is None
    assert (packed.store / ms.LOCK_RELPATH).exists()
    with pytest.raises(RuntimeError):
        ms.require_verified(meta)
    # 恢复后可以重跑到 verified。
    monkeypatch.setattr(pack.os, "replace", real_replace)
    cli(monkeypatch, verify_argv(packed, records=True))
    assert ms.MotionMeta.load(packed.store).status == "verified"


def test_verify_requires_tokens_resume_and_rejects_force_break(packed, monkeypatch):
    with pytest.raises(ValueError, match="--tokens"):
        cli(monkeypatch, ["verify", "--store", str(packed.store), "--resume"])
    with pytest.raises(ValueError, match="--resume"):
        cli(monkeypatch, verify_argv(packed, resume=False))
    with pytest.raises(ValueError, match="强制破锁"):
        cli(monkeypatch, [*verify_argv(packed), "--force-break-lock"])


def test_verified_store_is_not_reverified(packed, monkeypatch):
    write_records(packed)
    cli(monkeypatch, verify_argv(packed, records=True))
    with pytest.raises(ValueError, match="status=packed"):
        cli(monkeypatch, verify_argv(packed, records=True))


def test_verify_detects_table_tamper_and_other_export(packed, monkeypatch, tmp_path):
    write_records(packed)
    path = packed.store / ms.MOTION_TABLE_RELPATH
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="sha256"):
        cli(monkeypatch, verify_argv(packed, records=True))
    assert ms.MotionMeta.load(packed.store).status == "packed"
    path.write_bytes(original)
    record = load(packed.out / pack.TIMEPOOL_EXPORT_RECORD)
    record["table_sha256"] = "0" * 64
    dump(packed.out / pack.TIMEPOOL_EXPORT_RECORD, record)
    with pytest.raises(ValueError, match="同一次产物"):
        cli(monkeypatch, verify_argv(packed, records=True))


def test_gate_records_rejected_on_non_timepool_layout(monkeypatch):
    # 走参数校验分支即可：非 timepool 表给 --gate-records-dir 必须拒绝。
    meta = SimpleNamespace(spec=ms.LAYOUT_SPECS[ms.STAGE2_LAYOUT], manifest_path="x", provenance={})
    monkeypatch.setattr(ms.MotionMeta, "load", staticmethod(lambda root: meta))
    args = SimpleNamespace(store="/tmp/none", manifest=None, tokens=None, gate_records_dir="/tmp/x")
    with pytest.raises(ValueError, match="gate-records-dir"):
        pack.cmd_verify(args)


def test_report_prints_timepool_identity(packed, monkeypatch, capsys):
    cli(monkeypatch, ["report", "--store", str(packed.store)])
    out = capsys.readouterr().out
    assert "encoder_kind=wan_fixedpool1024 encoder_ckpt=none" in out and "status=packed" in out
