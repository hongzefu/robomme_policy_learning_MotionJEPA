"""1114 计划 S2：local-built-latents 来源的 pack / verify / 消费端（motion_store、motion_client）CPU 链路。"""
import copy
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/dataset"))
sys.path.insert(0, str(ROOT / "tests/datastore"))
sys.path.insert(0, str(ROOT / "tests/dataset"))
import check_stage2_validation_inputs as g0
import local_built_fixture as LB
import pack_motion_store as pack
from test_motion_store_stage2 import stage2_identity
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.policies import motion_client as C

PIN_PATH = ROOT / "scripts/dataset/wan_stage2_rgbstate/SOURCE_PIN.json"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def load(path):
    return json.loads(pathlib.Path(path).read_text())


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("lib")
    lib = LB.build(root)
    samples = g0.build_local_inputs(lib.manifest_path, lib.latents, episode_maps=lib.maps,
                                    merge_done=lib.merge_done, member_roots=[lib.raw])
    return lib, samples


@pytest.fixture
def local_fx(built, tmp_path, monkeypatch):
    lib, samples = built
    # 历史绑定的合成副本：总 metadata 摘要与报告文件 sha 都是合同常量，测试改指合成值。
    hist = tmp_path / "hist"
    dump(hist / ms.LOCAL_HISTORY_REPORT, dict(tested_metadata_sha256=lib.total_sha))
    monkeypatch.setattr(ms, "LOCAL_TESTED_METADATA_SHA256", lib.total_sha)
    monkeypatch.setattr(ms, "LOCAL_HISTORY_REPORT_SHA256", ms.sha256_file(hist / ms.LOCAL_HISTORY_REPORT))
    monkeypatch.setattr(pack, "HISTORY_REPORT_ROOT", hist)
    identity = stage2_identity()
    pin = load(PIN_PATH)
    enc_dir = tmp_path / "stage2-4task-rgbstate-144ep-1113-a"
    enc_dir.mkdir()
    (enc_dir / "checkpoint_epoch_144.pt").write_bytes(b"synthetic-rgbstate-EMA")
    (enc_dir / "config.yaml").write_text("run_name: stage2-4task-rgbstate-144ep-1113-a\n")
    enc = identity["encoder"]
    enc.update(run_name=enc_dir.name, checkpoint_name="checkpoint_epoch_144.pt", epoch=144,
               arch=ms.STAGE2_RGBSTATE_ARCH, checkpoint_sha256=ms.sha256_file(enc_dir / "checkpoint_epoch_144.pt"),
               config_sha256=ms.sha256_file(enc_dir / "config.yaml"))
    groups = {k: copy.deepcopy(identity[k]) for k in ("encoder", "encoder_state_sha256", "numerics")}
    groups["source_pin"] = {k: pin[k] for k in ("mj_repo_commit", "export_source_sha256", "encoder_source_sha256")}
    entries = ms.build_index_entries(lib.manifest, ms.LAYOUT_SPECS[ms.STAGE2_LAYOUT])
    tokens = tmp_path / "motion-tokens-stage2-rgbstate-ep144"
    tokens.mkdir()
    content, files = {}, []
    for e, seg, key, base, ng in pack.iter_rows_in_order(entries):
        lm = load(lib.latents / f"{key}.metadata.json")
        selected = list(range(0, lm["num_grid"], 2))
        latent = (lib.latents / f"{key}.bin").read_bytes()
        blob = (np.arange(ng * 1024, dtype=np.float32) + base * 1024).tobytes()
        content[key] = blob
        (tokens / f"{key}.f32.bin").write_bytes(blob)
        (tokens / f"{key}.f32.bin.sha256").write_text(ms.sha256_bytes(blob) + f"  {key}.f32.bin\n")
        metadata = {**groups, "worker": {"pid": 1, "device": "cpu", "hostname": "fixture"},
                    "schema": "stage2-memory-export-v1", "layout": ms.STAGE2_LAYOUT,
                    "manifest_sha256": ms.sha256_file(lib.manifest_path), "segment": key, "segment_kind": seg,
                    "seg_start_global": lm["seg_start_global"], "seg_len": lm["seg_len"], "grid_stride": 32,
                    "window_frames": 33, "row_shape": [1, 1024], "dtype": "float32", "token_dim": 1024,
                    "token_bytes": 4096, "num_grid": ng, "bytes": len(blob), "sha256": ms.sha256_bytes(blob),
                    "source_grid": selected, "rows": [lm["rows"][g] for g in selected],
                    "input_latent_sha256": ms.sha256_bytes(latent),
                    "input_block_sha256": [ms.sha256_bytes(latent[g * pack.CHUNK_BYTES:(g + 1) * pack.CHUNK_BYTES])
                                           for g in selected]}
        dump(tokens / f"{key}.metadata.json", metadata)
        files.append(dict(key=key, sha256=metadata["sha256"], num_grid=ng,
                          metadata_sha256=ms.sha256_file(tokens / f"{key}.metadata.json")))
    dump(tokens / "export_record.json", {**groups, "schema": "stage2-memory-export-v1", "scope": "full",
                                         "segments": len(content), "rows": ms.index_totals(entries)["rows"],
                                         "segment_keys": list(content), "files": sorted(files, key=lambda f: f["key"]),
                                         "manifest_sha256": ms.sha256_file(lib.manifest_path)})
    validation = tmp_path / "validation"
    dump(validation / "windows.json", samples)
    samples_sha = ms.sha256_file(validation / "windows.json")
    vae = load(lib.latents / f"{next(iter(content))}.metadata.json")["vae"]
    online_vae = {k: vae[k] for k in ("vae_id", "vae_state_sha256", "vae_dtype", "latent_mode", "batch", "tf32", "amp")}
    online_vae.update(adapter_sha256=pin["vae_adapter_sha256"], numerics=groups["numerics"],
                      preprocess=dict(frame_size=256, input_dtype="uint8", input_layout="THWC", window_frames=33))
    base = dict(schema="stage2-validation-gate-v1", status="PASS", checkpoint_sha256=enc["checkpoint_sha256"],
                samples_sha256=samples_sha, source_pin=pin, numerics=groups["numerics"], kind=ms.STAGE2_LOCAL_ORIGIN)
    dump(validation / "G2.json", dict(base, gate="G2", comparison=ms.LOCAL_G2_COMPARISON, online_vae=online_vae,
                                      metrics=dict(windows=64, demo_tail=samples["counts"]["demo_tail"],
                                                   worst_mse=1e-9, worst_max=1e-4)))
    dump(validation / "G3.json", dict(base, gate="G3", mask_profile="rgb-state",
                                      encoder_state_sha256=groups["encoder_state_sha256"],
                                      metrics=dict(windows=32, seed=0, max_abs=0, geometry_all_true=1, leaves=65,
                                                   state_key="model", exact=1, live_negative_rejected=1)))
    lineage = tmp_path / "lineage.json"
    dump(lineage, dict(schema=ms.LOCAL_LINEAGE_SCHEMA, mode="member-pin", status="PASS",
                       member_pin_sha256=ms.LOCAL_MEMBER_PIN_SHA256, members=1600))
    args = SimpleNamespace(layout="stage2", latent_grid_subsample=2, stage2_validation=str(validation),
                           stage2_source_pin=str(PIN_PATH), stage2_project="wan_stage2_rgbstate",
                           latents_origin=ms.STAGE2_LOCAL_ORIGIN, lineage_receipt=str(lineage),
                           manifest=str(lib.manifest_path), tokens=str(tokens), latents=str(lib.latents),
                           out=str(tmp_path / "motion-stage2-1024-rgbstate-ep144"), encoder_run_dir=str(enc_dir),
                           raw_dir=None, oracle_report=None, restore_receipt=None, cache_import_receipt=None,
                           expected_ckpt_sha256=enc["checkpoint_sha256"], resume=False, force_break_lock=False)
    return SimpleNamespace(lib=lib, samples=samples, args=args, content=content, validation=validation,
                           enc_dir=enc_dir, tokens=tokens)


def certify(fx):
    root = pathlib.Path(fx.args.out)
    meta = ms.MotionMeta.load(root)
    record = dict(schema="stage2-encoder-bitexact-v1", scope="full", layout=ms.STAGE2_LAYOUT, status="PASS",
                  compared=meta.num_rows, mismatches=0, store_table_sha256=meta.table_sha256,
                  motion_index_sha256=meta.motion_index_sha256,
                  checkpoint_sha256=meta.provenance["encoder"]["checkpoint_sha256"],
                  source_grid_sha256=meta.provenance["source_grid_sha256"])
    dump(root / pack.BITEXACT_RELPATH, record)


def verify_args(fx):
    return SimpleNamespace(store=fx.args.out, manifest=None, tokens=None, resume=True, force_break_lock=False,
                           check_padding=True)


def test_local_built_pack_verify_load_and_consumers(local_fx, capsys):
    fx = local_fx
    pack.cmd_pack(fx.args)
    assert "latents_origin=local-built-latents raw_dir=null" in capsys.readouterr().out
    root = pathlib.Path(fx.args.out)
    meta = ms.MotionMeta.load(root)
    prov = meta.provenance
    assert prov["raw_dir"] is None and prov["stage2_compat"]["kind"] == ms.STAGE2_LOCAL_ORIGIN
    assert prov["encoder"]["arch"] == ms.STAGE2_RGBSTATE_ARCH
    assert set(prov["stage2_compat"]["evidence"]) == set(ms.LOCAL_EVIDENCE_ROLES)
    assert "restore_receipt" not in prov["stage2_compat"]["evidence"]
    # C06：逐段摘要清单固定路径、schema，且被消费端读通。
    listing_path = fx.validation / ms.BIN_SHA256_LIST_NAME
    listing = load(listing_path)
    assert listing["schema"] == "bin-sha256-list-v1" and listing["root"] == str(fx.lib.latents.resolve())
    assert len(listing["files"]) == len(fx.content)
    ms.validate_bin_sha256_list(listing, load(fx.lib.latents / "metadata.json"))
    assert prov["stage2_compat"]["evidence"]["wan_latents_bin_sha256"]["path"] == str(listing_path.resolve())
    C.validate_stage2_evidence(prov, root)              # motion_client 消费端：读全部证据并逐项核验
    ms.validate_stage2_provenance(prov)
    certify(fx)
    pack.cmd_verify(verify_args(fx))
    meta = ms.MotionMeta.load(root)
    assert meta.status == "verified" and meta.raw["verify"]["mismatches"] == 0
    assert not (root / ms.LOCK_RELPATH).exists()
    assert (root / ms.MOTION_TABLE_RELPATH).read_bytes() == b"".join(fx.content.values())


def test_tampered_sha_fails(local_fx):
    fx = local_fx
    key = next(iter(fx.content))
    side = fx.lib.latents / f"{key}.bin.sha256"
    original = side.read_text()
    try:
        side.write_text("0" * 64 + f"  {key}.bin\n")
        with pytest.raises(ValueError, match="sidecar"):
            pack.cmd_pack(fx.args)
    finally:
        side.write_text(original)
    g2 = fx.validation / "G2.json"
    value = load(g2)
    value["samples_sha256"] = "0" * 64
    dump(g2, value)
    with pytest.raises(ValueError):
        pack.cmd_pack(fx.args)
    assert not (pathlib.Path(fx.args.out) / ms.MOTION_TABLE_RELPATH).exists()


def test_out_of_sample_bin_with_updated_digests_fails_history(local_fx):
    fx = local_fx
    # 样本外：G0 只取偶数格，改段内奇数格（第 1 块）的字节。
    key = next(iter(fx.content))
    paths = [fx.lib.latents / f"{key}{s}" for s in (".bin", ".bin.sha256", ".metadata.json")] + [
        fx.lib.latents / "metadata.json"]
    saved = {p: p.read_bytes() for p in paths}
    try:
        data = bytearray(saved[paths[0]])
        data[pack.CHUNK_BYTES] ^= 1
        paths[0].write_bytes(bytes(data))
        digest = ms.sha256_bytes(bytes(data))
        paths[1].write_text(f"{digest}  {key}.bin\n")
        lm = json.loads(saved[paths[2]])
        lm["sha256"] = digest
        dump(paths[2], lm)
        total = json.loads(saved[paths[3]])
        total["segments"][key]["sha256"] = digest
        dump(paths[3], total)
        # 自身摘要链一致，但总 metadata 摘要已不等于历史 tested_metadata_sha256。
        with pytest.raises(ValueError, match="历史 tested_metadata_sha256"):
            pack.cmd_pack(fx.args)
    finally:
        for p, b in saved.items():
            p.write_bytes(b)


@pytest.mark.parametrize("fault", ["raw_dir", "no_lineage", "no_project", "project_mismatch", "lineage_fail"])
def test_local_built_argument_contract(local_fx, fault):
    fx = local_fx
    if fault == "raw_dir":
        fx.args.raw_dir = "/scratch/hongze/whatever"
    elif fault == "no_lineage":
        fx.args.lineage_receipt = None
    elif fault == "no_project":
        fx.args.stage2_project = None
    elif fault == "project_mismatch":
        fx.args.stage2_project = "wan_stage2"
    else:
        value = load(fx.args.lineage_receipt)
        value["member_pin_sha256"] = "01027855e0980c08da617a927e9f6bb5e6a114a720014b498381c3315dad5ed3"
        dump(pathlib.Path(fx.args.lineage_receipt), value)
    with pytest.raises(ValueError):
        pack.cmd_pack(fx.args)
    assert not (pathlib.Path(fx.args.out) / ms.MOTION_TABLE_RELPATH).exists()


def test_non_local_sources_reject_lineage_and_require_raw_dir(local_fx):
    fx = local_fx
    fx.args.latents_origin = None
    fx.args.restore_receipt = "/x/restore_receipt.json"
    with pytest.raises(ValueError, match="lineage-receipt"):
        pack.cmd_pack(fx.args)
    fx.args.lineage_receipt = None
    with pytest.raises(ValueError, match="--raw-dir"):
        pack.cmd_pack(fx.args)


def test_provenance_arch_whitelist_and_local_pairing(local_fx):
    fx = local_fx
    pack.cmd_pack(fx.args)
    prov = ms.MotionMeta.load(pathlib.Path(fx.args.out)).provenance
    unknown = copy.deepcopy(prov)
    unknown["encoder"]["arch"] = "wan-latent-stage2-unknown"
    with pytest.raises(ValueError, match="白名单"):
        ms.validate_stage2_provenance(unknown)
    three = copy.deepcopy(prov)
    three["encoder"]["arch"] = ms.STAGE2_ARCH
    with pytest.raises(ValueError, match="只能配 rgb-state"):
        ms.validate_stage2_provenance(three)
    raw = copy.deepcopy(prov)
    raw["raw_dir"] = "/scratch/hongze/raw"
    with pytest.raises(ValueError, match="raw_dir"):
        ms.validate_stage2_provenance(raw)
    old = stage2_identity()                  # 旧三路历史来源不回归
    ms.validate_stage2_provenance(old)
    assert ms.stage2_evidence_roles(old) == ("G0", "G2", "G3", "restore_receipt", "SOURCE_PIN")


# ─────────────────────────────── motion_client：项目选择

def test_motion_client_selects_new_project_and_rejects_mismatch(local_fx, tmp_path):
    fx = local_fx
    pack.cmd_pack(fx.args)
    meta = ms.MotionMeta.load(pathlib.Path(fx.args.out))
    source_run = f"{fx.enc_dir.name}/checkpoint_epoch_144.pt#model"
    common = dict(online_gpu="", encoder_kind=C.STAGE2_KIND, source_run=source_run, v1_store=tmp_path,
                  handshake_timeout=30, store_provenance=meta.provenance)
    client = C.MotionEncoderClient(**common, stub=True, stage2_python=sys.executable)
    try:
        assert client.stage2_project == "wan_stage2_rgbstate"
        assert client.provenance["shape"] == [1024] and client.alive
        assert client.argv[client.argv.index("--stage2-project") + 1] == "wan_stage2_rgbstate"
        token = client(np.stack([C.P.stub_frame(i) for i in range(33)]), 0)
        assert token.shape == (1024,)
    finally:
        client.close()
    with pytest.raises(ValueError, match="不符"):
        C.MotionEncoderClient(**common, stub=True, stage2_python=sys.executable, stage2_project="wan_stage2")
    # 真实模式：stage2_python 必须是所选项目 venv 的解释器。
    with pytest.raises(ValueError, match="stage2_python"):
        C.MotionEncoderClient(**dict(common, provenance_root=meta.root), stub=False, stage2_python=sys.executable,
                              expected_ckpt_sha256=meta.provenance["encoder"]["checkpoint_sha256"])


def test_compare_and_online_reference_use_selected_project(local_fx, monkeypatch, tmp_path):
    fx = local_fx
    pack.cmd_pack(fx.args)
    root = pathlib.Path(fx.args.out)
    meta = ms.MotionMeta.load(root)
    sys.path.insert(0, str(ROOT / "scripts/dataset/wan_stage2"))
    sys.path.insert(0, str(ROOT / "scripts/training"))
    import compare_stage2 as CMP
    import check_stage2_online as O
    base = dict(store=str(root), latents=str(fx.lib.latents), ckpt=str(fx.enc_dir / "checkpoint_epoch_144.pt"),
                expected_sha256=meta.provenance["encoder"]["checkpoint_sha256"], batch=1,
                expect_rows=meta.num_rows, device="cpu")
    with pytest.raises(ValueError, match="全表"):
        CMP.certify(SimpleNamespace(**base, scope="sampled", stage2_project=None))
    with pytest.raises(ValueError, match="不符"):
        CMP.certify(SimpleNamespace(**base, scope="full", stage2_project="wan_stage2"))
    captured = {}

    def run(argv, env, cwd, check):
        captured.update(argv=argv, env=env)
        raise RuntimeError("截获参考子进程")

    monkeypatch.setattr(O.subprocess, "run", run)
    args = SimpleNamespace(samples=str(fx.validation / "windows.json"), online_gpu="1", stage2_project=None)
    with pytest.raises(RuntimeError, match="截获"):
        O.reference_subprocess(args, meta, tmp_path, tmp_path)
    argv, env = captured["argv"], captured["env"]
    assert pathlib.Path(argv[3]).name == "wan_stage2_rgbstate"
    assert argv[argv.index("--stage2-project") + 1] == "wan_stage2_rgbstate"
    assert pathlib.Path(env["UV_PROJECT_ENVIRONMENT"]) == tmp_path / "venvs/wan-stage2-rgbstate"
    assert argv[6].endswith("scripts/dataset/wan_stage2/validate_stage2.py")
    args.stage2_project = "wan_stage2"
    with pytest.raises(ValueError, match="不符"):
        O.reference_subprocess(args, meta, tmp_path, tmp_path)
