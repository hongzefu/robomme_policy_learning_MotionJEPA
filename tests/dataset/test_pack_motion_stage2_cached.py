"""发布缓存的输入合同、完整性认证及旧来源并存的 CPU 回归。"""
import copy
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/dataset"))
from test_pack_motion_stage2 import dump, load, stage2_fx, verify_args
import pack_motion_store as pack
import test_build_mixed8_v10_motion as legacy
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.manifest import manifest_sha256
from mme_vla_suite.policies.motion_client import validate_stage2_evidence


def cache_receipt_for(stage2):
    """测试专用收据构造器；生产入口不依赖测试 helper。"""
    s = stage2
    raw = pathlib.Path(s.args.raw_dir)
    raw.mkdir()
    source_h5 = []
    for name in s.fx.manifest["canonical_order"]:
        path = raw / name
        path.write_bytes(name.encode())
        source_h5.append(dict(name=name, bytes=path.stat().st_size, sha256=ms.sha256_file(path)))
    input_path = s.fx.manifest_path.parent / "input_manifest.json"
    dump(input_path, dict(raw_dir=str(raw), count=len(source_h5),
                         files={f["name"]: dict(size=f["bytes"], sha256=f["sha256"]) for f in source_h5}))
    total = load(s.fx.latents / "metadata.json")
    published, windows, selected = [], [], []
    for key, segment in total["segments"].items():
        lm = load(s.fx.latents / f"{key}.metadata.json")
        blob = (s.fx.latents / f"{key}.bin").read_bytes()
        task, ep = key.split("_ep", 1)
        episode = int(ep.split("_", 1)[0])
        kind = lm["segment_kind"]
        name = f"record_dataset_{task}.h5"
        published.append(dict(path=f"latents/{key}.bin", sha256="b" * 64, bytes=33 * pack.CHUNK_BYTES))
        source = dict(kind="raw_h5", path=str(raw / name), episode=episode,
                      sha256=ms.sha256_file(raw / name))
        for row in lm["rows"]:
            m, real, pad = row["m"], row["real_frames"], row["pad_frames"]
            start = row["start_global_frame"]
            frames = list(range(start, start + real)) + [start + real - 1] * pad
            route = "fill" if m == lm["num_grid"] - 1 else "copy"
            window = dict(key=key, m=m, route=route, h5_file=name, raw_ep_idx=episode, segment_kind=kind,
                          start_global_frame=start, frame_ids=frames, real_frames=real, pad_frames=pad,
                          pad_source_frame=row["pad_source_frame"], input_frames_sha256=row["input_frames_sha256"],
                          latent_sha256=ms.sha256_bytes(blob[m * pack.CHUNK_BYTES:(m + 1) * pack.CHUNK_BYTES]),
                          output_file=f"{key}.bin", output_offset=m * pack.CHUNK_BYTES,
                          source_file=f"latents/{key}.bin" if route == "copy" else None,
                          source_row=m * 16 if route == "copy" else None,
                          source_offset=m * 16 * pack.CHUNK_BYTES if route == "copy" else None,
                          source_layout="group_first" if route == "copy" else None)
            windows.append(window)
            if m % 2 == 0:
                selected.append(dict(key=key, task=task, episode=episode, segment_kind=kind, source_grid=m,
                                     route=route, latent_path=str(s.fx.latents / f"{key}.bin"),
                                     latent_block_sha256=window["latent_sha256"], frame_ids=frames,
                                     real_frames=real, pad_frames=pad, pad_source_frame=row["pad_source_frame"],
                                     input_frames_sha256=row["input_frames_sha256"], sources=[source],
                                     source_frame_indices=[dict(source=0, index=f) for f in frames]))
    receipt = dict(schema=ms.WAN_CACHE_IMPORT_SCHEMA, kind=ms.STAGE2_CACHE_ORIGIN, status="PASS",
        published=dict(repo="HongzeFu/robomme-original-16x100-wan21-front-33f-dense-v1",
                       revision="384d5db410be713a49357bf3f55ca4f4d2f559a0", sha256sums_sha256="c" * 64, files=published),
        source_h5_files=source_h5, manifest_sha256=s.fx.manifest["sha256"],
        manifest_file_sha256=ms.sha256_file(s.fx.manifest_path), input_manifest_sha256=ms.sha256_file(input_path),
        importer_sha256="d" * 64, manifest_path=str(s.fx.manifest_path), input_manifest_path=str(input_path),
        vae={key: legacy.vae_info()[key] for key in ("vae_id", "vae_state_sha256", "latent_mode")},
        preprocess=dict(frame_size=256, input_dtype="uint8", input_layout="THWC", window_frames=33, augmentation="none"),
        numeric_modes=dict(copy=dict(dtype="float16", cudnn_allow_tf32=True),
                           fill=dict(dtype="float32", cudnn_allow_tf32=False)),
        latents=dict(root=str(s.fx.latents), metadata_sha256=ms.sha256_file(s.fx.latents / "metadata.json"),
                     files=[dict(path=p.name, sha256=ms.sha256_file(p), bytes=p.stat().st_size)
                            for p in sorted(s.fx.latents.iterdir())]),
        windows=windows, counts=dict(windows=len(windows), copy=sum(w["route"] == "copy" for w in windows),
                                    fill=sum(w["route"] == "fill" for w in windows)))
    return receipt, selected


@pytest.fixture
def cached_fx(tmp_path, monkeypatch):
    original = legacy.manifest_for

    def sixteen_tasks(_specs, raw):
        manifest = original([(114, 49)] * 16, raw)
        for g, episode in enumerate(manifest["episodes"]):
            episode.update(h5_file=f"record_dataset_Task{g:02d}.h5", raw_ep_idx=0)
        manifest["canonical_order"] = [e["h5_file"] for e in manifest["episodes"]]
        manifest["sha256"] = manifest_sha256(manifest)
        return manifest

    monkeypatch.setattr(legacy, "manifest_for", sixteen_tasks)
    s = stage2_fx.__wrapped__(tmp_path, monkeypatch)
    receipt, windows = cache_receipt_for(s)
    s.receipt = tmp_path / "cache_import.json"
    dump(s.receipt, receipt)
    s.args.restore_receipt = None
    s.args.latents_origin = ms.STAGE2_CACHE_ORIGIN
    s.args.cache_import_receipt = str(s.receipt)
    g0 = load(s.validation / "windows.json")
    g0.update(windows=windows, cache_import_receipt_sha256=ms.sha256_file(s.receipt))
    dump(s.validation / "windows.json", g0)
    g2 = dict(schema="stage2-validation-gate-v1", gate="G2", status="PASS", kind=ms.STAGE2_CACHE_ORIGIN,
              samples_sha256=ms.sha256_file(s.validation / "windows.json"),
              cache_import_receipt_sha256=ms.sha256_file(s.receipt), manifest_sha256=receipt["manifest_sha256"],
              latent_metadata_sha256=receipt["latents"]["metadata_sha256"], vae=receipt["vae"],
              preprocess=receipt["preprocess"], numeric_modes=receipt["numeric_modes"],
              output_numerics=dict(ms.SAME_INPUT_NOT_REQUIRED),
              metrics=dict(windows=64, demo_tail=16, copy=32, demo_fill=16, exec_fill=16,
                           sha_mismatches=0, frame_mismatches=0, shape_mismatches=0, nonfinite=0))
    dump(s.validation / "G2.json", g2)
    g3 = load(s.validation / "G3.json")
    g3.update(kind=ms.STAGE2_CACHE_ORIGIN, samples_sha256=g2["samples_sha256"],
              cache_import_receipt_sha256=g2["cache_import_receipt_sha256"],
              encoder_state_sha256=s.identity["encoder_state_sha256"], output_numerics=dict(ms.SAME_INPUT_NOT_REQUIRED))
    g3["metrics"].pop("max_abs")
    g3["metrics"].update(shape_mismatches=0, nonfinite=0)
    dump(s.validation / "G3.json", g3)
    return s


def test_cached_pack_verify_and_portable_evidence(cached_fx):
    s = cached_fx
    pack.cmd_pack(s.args)
    root = pathlib.Path(s.args.out)
    assert not (root / pack.BITEXACT_RELPATH).exists()
    pack.cmd_verify(verify_args(s))
    meta = ms.MotionMeta.load(root)
    assert meta.num_rows == 64 and meta.status == "verified"
    compat = meta.provenance["stage2_compat"]
    assert set(compat["evidence"]) == {"G0", "G2", "G3", "wan_cache_import", "SOURCE_PIN"}
    assert compat["numeric_modes"]["copy"] != compat["numeric_modes"]["fill"]
    assert compat["output_numerics"] == ms.SAME_INPUT_NOT_REQUIRED
    assert compat["encoder_validation"]["packed_compared"] == 64
    assert compat["encoder_validation"]["encoder_compared"] == 0
    validate_stage2_evidence(meta.provenance, root)
    pack.cmd_verify(verify_args(s))
    raw = load(root / ms.META_RELPATH)
    del raw["provenance"]["stage2_compat"]["encoder_validation"]
    dump(root / ms.META_RELPATH, raw)
    with pytest.raises(ValueError, match="完整性认证"):
        ms.MotionMeta.load(root)


@pytest.mark.parametrize("fault", ["h5", "frame", "augmentation", "axis", "missing", "duplicate", "bytes", "nan", "weight", "g2_finite", "g3_finite"])
def test_cached_sources_fail_closed(cached_fx, fault):
    s = cached_fx
    receipt = load(s.receipt)
    if fault == "h5":
        (pathlib.Path(s.args.raw_dir) / receipt["source_h5_files"][0]["name"]).write_bytes("换源".encode())
    elif fault == "bytes":
        path = s.fx.latents / receipt["windows"][0]["output_file"]
        blob = bytearray(path.read_bytes()); blob[0] ^= 1; path.write_bytes(blob)
    elif fault == "nan":
        path = s.fx.latents / receipt["windows"][0]["output_file"]
        blob = bytearray(path.read_bytes()); blob[:4] = np.float32(np.nan).tobytes(); path.write_bytes(blob)
        receipt["windows"][0]["latent_sha256"] = ms.sha256_bytes(blob[:pack.CHUNK_BYTES])
        for item in receipt["latents"]["files"]:
            if item["path"] == path.name:
                item["sha256"] = ms.sha256_file(path)
    elif fault == "frame":
        receipt["windows"][0]["frame_ids"][0] += 1
    elif fault == "augmentation":
        receipt["preprocess"]["augmentation"] = "crop"
    elif fault == "axis":
        path = s.fx.latents / receipt["windows"][0]["output_file"]
        values = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(-1, 9, 16, 32, 32)
        path.write_bytes(values.transpose(0, 2, 1, 3, 4).tobytes())
    elif fault == "missing":
        receipt["windows"].pop()
    elif fault == "duplicate":
        receipt["windows"].append(copy.deepcopy(receipt["windows"][0]))
    elif fault == "weight":
        receipt["vae"]["vae_state_sha256"] = "0" * 64
    elif fault in ("g2_finite", "g3_finite"):
        path = s.validation / ("G2.json" if fault == "g2_finite" else "G3.json")
        record = load(path); record["metrics"]["nonfinite"] = 1; dump(path, record)
    dump(s.receipt, receipt)
    with pytest.raises(ValueError):
        pack.cmd_pack(s.args)
    assert not (pathlib.Path(s.args.out) / ms.MOTION_TABLE_RELPATH).exists()


def test_cached_verify_detects_forged_packed_sha(cached_fx):
    s = cached_fx
    pack.cmd_pack(s.args)
    root = pathlib.Path(s.args.out)
    table = root / ms.MOTION_TABLE_RELPATH
    blob = bytearray(table.read_bytes()); blob[0] ^= 1; table.write_bytes(blob)
    meta = load(root / ms.META_RELPATH)
    meta["tables"][ms.MOTION_KEY]["sha256"] = ms.sha256_file(table)
    dump(root / ms.META_RELPATH, meta)
    with pytest.raises(SystemExit):
        pack.cmd_verify(verify_args(s))
    assert load(root / ms.META_RELPATH)["status"] == "packed"
