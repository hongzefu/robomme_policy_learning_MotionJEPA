"""fixed pool 的 layout、readout 与来源记录合同；旧分支与 stage2 路径互不串用。"""
import copy
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest

from mme_vla_suite.datastore import motion_adapter as ma
from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.manifest import manifest_sha256
from mme_vla_suite.shared import wan_timepool as tp

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = "f7b2dc854d5955a08c2c3f9f4e678ce251213745"


def baseline_module(relative, name):
    source = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT, text=True)
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, str(ROOT / relative), "exec"), module.__dict__)
    return module


def load_stage2_identity():
    spec = importlib.util.spec_from_file_location("stage2_store_tests", ROOT / "tests/datastore/test_motion_store_stage2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stage2_identity()


def manifest_for(specs):
    rows, offset, total = [], 0, 0
    for g, (nt, es) in enumerate(specs):
        rows.append(dict(global_episode_idx=g, h5_file="record_dataset_T.h5", raw_ep_idx=g, num_timesteps=nt,
                         exec_start_idx=es, exec_samples=nt - es, exec_sample_offset=offset,
                         total_sample_offset=total, shard_idx=0))
        offset += nt - es
        total += nt
    value = dict(version=1, raw_dir="/scratch/hongze/合成原数据", canonical_order=["record_dataset_T.h5"],
                 num_shards=1, shard_load_timesteps=[total], episodes=rows,
                 totals=dict(episodes=len(rows), timesteps=total, exec_samples=offset))
    value["sha256"] = manifest_sha256(value)
    return value


def window_map_for(entries, table):
    windows, row = [], 0
    for entry in entries:
        for segment in ms.SEGMENTS:
            info = getattr(entry, segment)
            start = 0 if segment == "demo" else entry.exec_start_idx
            for m in range(info.num_grid):
                real = min(33, info.seg_len - 32 * m)
                windows.append(dict(row=row, key=ms.segment_key(entry.h5_file, entry.raw_ep_idx, segment),
                                    segment=segment, m=m, source_m=2 * m, route="copy" if row % 2 else "fill",
                                    start_frame=start + 32 * m, real_frames=real, pad_frames=33 - real,
                                    input_frames_sha256="a" * 64, raw_latent_sha256="b" * 64,
                                    pooled_sha256=hashlib.sha256(table[row].tobytes()).hexdigest()))
                row += 1
    counts = dict(windows=row, copy=sum(w["route"] == "copy" for w in windows),
                  fill=sum(w["route"] == "fill" for w in windows))
    return dict(schema=ms.TIMEPOOL_WINDOW_MAP_SCHEMA, layout=tp.TIMEPOOL_LAYOUT, source_grid_stride=16,
                target_grid_stride=32, counts=counts, windows=windows)


def provenance_for(*, table_sha="1" * 64, index_sha="2" * 64, manifest_sha="3" * 64, counts=None,
                   map_sha="4" * 64, status="verified"):
    counts = counts or dict(windows=3, copy=1, fill=2)
    validation = {role: dict(path=f"/scratch/hongze/{role}.json", sha256="5" * 64, verdict=f"{role}=PASS rows=3")
                  for role in ms.TIMEPOOL_VALIDATION_ROLES}
    return dict(
        schema=ms.TIMEPOOL_PROVENANCE_SCHEMA, layout=tp.TIMEPOOL_LAYOUT, readout=tp.TIMEPOOL_READOUT,
        encoder_kind=tp.TIMEPOOL_ENCODER_KIND, encoder_ckpt=None, parameter_count=0, output_shape=[1, 1024],
        pool=dict(version=tp.POOL_VERSION, source_sha256=tp.pool_source_sha256(), contract=dict(tp.POOL_CONTRACT)),
        source_h5=[dict(name="record_dataset_T.h5", bytes=123, sha256="6" * 64)],
        manifests=dict(episode_manifest_sha256=manifest_sha, episode_manifest_file_sha256="7" * 64,
                       input_manifest_sha256="8" * 64),
        wan_cache=dict(plan_file_sha256="9" * 64, plan_canonical_sha256="a" * 64,
                       import_receipt_file_sha256="b" * 64, restored_receipt_file_sha256="c" * 64),
        published=dict(repo="HongzeFu/robomme-original", revision="d" * 40),
        routes=dict(counts=counts, numeric_modes=dict(copy=dict(tf32=True), fill=dict(tf32=False))),
        window_map=dict(schema=ms.TIMEPOOL_WINDOW_MAP_SCHEMA, relpath=ms.TIMEPOOL_WINDOW_MAP_RELPATH,
                        sha256=map_sha, rows=counts["windows"], source_grid_stride=16, target_grid_stride=32),
        wan=dict(vae_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", latent_mode="mode", vae_state_sha256="e" * 64,
                 adapter_sha256="f" * 64, dependency_lock_sha256="0" * 64,
                 preprocess=dict(frame_size=256, input_dtype="uint8", input_layout="THWC", window_frames=33,
                                 augmentation="none")),
        output=dict(motion_table_sha256=table_sha, motion_index_sha256=index_sha),
        validation=None if status == "packed" else validation,
        train_head="1" * 40,
    )


def build_store(root, *, layout=tp.TIMEPOOL_LAYOUT, status="verified", provenance=None, spec=None):
    """写一份完整迷你库；provenance 给可调用对象时可以改写默认来源后再落盘。"""
    spec = spec or ms.KNOWN_LAYOUT_SPECS[layout]
    manifest = manifest_for([(165, 100), (82, 50)])
    entries = ms.build_index_entries(manifest, spec)
    index = ms.index_payload(manifest, entries, spec=spec, layout=layout, mj_repo_commit="a" * 40)
    (root / "meta").mkdir(parents=True)
    index_path = root / ms.INDEX_RELPATH
    index_path.write_text(json.dumps(index))
    manifest_path = root / "episode_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    rows = ms.index_totals(entries)["rows"]
    table = np.arange(rows * 1024, dtype=np.float32).reshape(rows, 1, 1024)
    (root / ms.MOTION_TABLE_RELPATH).write_bytes(table.tobytes())
    table_sha = ms.sha256_file(root / ms.MOTION_TABLE_RELPATH)
    window_map = window_map_for(entries, table)
    map_path = root / ms.TIMEPOOL_WINDOW_MAP_RELPATH
    map_path.write_text(json.dumps(window_map))
    value = provenance_for(table_sha=table_sha, index_sha=ms.sha256_file(index_path), manifest_sha=manifest["sha256"],
                           counts=window_map["counts"], map_sha=ms.sha256_file(map_path), status=status)
    if callable(provenance):
        provenance(value)
    elif provenance is not None:
        value = provenance
    raw = dict(schema=1, layout=layout, status=status, byte_order="little", array_order="C", manifest_scope="full",
               grid_stride=32, window_frames=33, grid_origin="segment_start", window_direction="forward",
               truncation_policy="none", frame_size=256, num_rows=rows, manifest_sha256=manifest["sha256"],
               manifest_path=str(manifest_path), motion_index_sha256=ms.sha256_file(index_path), provenance=value,
               tables={"motion_token": dict(row_shape=[1, 1024], dtype="float32", row_bytes=4096, num_rows=rows,
                                            relpath=ms.MOTION_TABLE_RELPATH, byte_count=rows * 4096, sha256=table_sha)},
               **spec.fields())
    (root / ms.META_RELPATH).write_text(json.dumps(raw))
    return raw


def rewrite_meta(root, change):
    raw = json.loads((root / ms.META_RELPATH).read_text())
    change(raw)
    (root / ms.META_RELPATH).write_text(json.dumps(raw))


# ── motion_policy_spec ───────────────────────────────────────────────────────


def timepool_cfg(**overrides):
    return {**dict(readout=tp.TIMEPOOL_READOUT, source_shape=[1, 1024], num_tokens=1, dim=1024, budget=96), **overrides}


@pytest.mark.parametrize("budget", [1, 96, 160])
def test_policy_spec_new_readout(budget):
    spec = ma.motion_policy_spec(timepool_cfg(budget=budget))
    assert spec == ma.MotionPolicySpec((1, 1024), tp.TIMEPOOL_READOUT, 1, 1024, budget, budget, (tp.TIMEPOOL_LAYOUT,))


@pytest.mark.parametrize("key", ["source_shape", "num_tokens", "dim", "budget"])
def test_policy_spec_missing_key_rejected(key):
    cfg = timepool_cfg()
    del cfg[key]
    with pytest.raises(ValueError, match="缺字段"):
        ma.motion_policy_spec(cfg)


@pytest.mark.parametrize("overrides", [
    dict(source_shape=[4, 256]), dict(source_shape=[1024]), dict(source_shape=[1.0, 1024]), dict(source_shape=[True, 1024]),
    dict(source_shape=7), dict(num_tokens=4), dict(num_tokens=True), dict(dim=256), dict(dim=1024.0),
    dict(budget=0), dict(budget=-96), dict(budget=96.0), dict(budget=True),
])
def test_policy_spec_wrong_values_rejected(overrides):
    with pytest.raises(ValueError):
        ma.motion_policy_spec(timepool_cfg(**overrides))


def spec_outcome(module, cfg):
    try:
        return ("ok", repr(module.motion_policy_spec(copy.deepcopy(cfg))))
    except Exception as exc:  # noqa: BLE001 — 比较旧新实现的异常类型与原文
        return (type(exc).__name__, str(exc))


OLD_CONFIGS = [
    {}, dict(budget=160), dict(budget=8), dict(budget=32, dim=768), dict(dim=512), dict(num_tokens=4),
    dict(source_shape=[8, 256], readout="adjacent_pair_v1", num_tokens=4, dim=512, budget=160),
    dict(source_shape=[8, 256], readout="adjacent_pair_v1", num_tokens=4, dim=512, budget=161),
    dict(source_shape=[8, 256], readout="unknown", num_tokens=4, dim=512),
    dict(readout="stage2_pool4_concat_v1", source_shape=[1, 1024], num_tokens=1, dim=1024, budget=96),
    dict(readout="stage2_pool4_concat_v1", source_shape=[1, 1024], num_tokens=1, dim=1024, budget=0),
    dict(readout="stage2_pool4_concat_v1", source_shape=[1, 1024], num_tokens=1),
    dict(readout="stage2_pool4_split_v1", source_shape=[1, 1024], num_tokens=4, dim=256, budget=384),
    dict(readout="stage2_pool4_split_v1", source_shape=[1, 1024], num_tokens=4, dim=256, budget=385),
    dict(readout="stage2_pool4_split_v1", source_shape=7, num_tokens=4, dim=256, budget=384),
]


@pytest.mark.parametrize("cfg", OLD_CONFIGS)
def test_old_policy_branches_unchanged_against_baseline(cfg):
    old = baseline_module("src/mme_vla_suite/datastore/motion_adapter.py", "baseline_motion_adapter_a")
    before, after = spec_outcome(old, cfg), spec_outcome(ma, cfg)
    assert before[0] == after[0] and before[1].replace("baseline_motion_adapter_a.", "") == after[1].replace(
        "mme_vla_suite.datastore.motion_adapter.", "")


# ── layout 合同 ──────────────────────────────────────────────────────────────


def test_old_layout_registry_and_repr_unchanged():
    old = baseline_module("src/mme_vla_suite/datastore/motion_store.py", "baseline_motion_store_a")
    assert list(old.LAYOUT_SPECS) == list(ms.LAYOUT_SPECS)
    for name, spec in old.LAYOUT_SPECS.items():
        assert repr(spec) == repr(ms.LAYOUT_SPECS[name])


def test_timepool_spec_is_distinct_from_stage2_with_same_window_numbers():
    spec, stage2 = ms.KNOWN_LAYOUT_SPECS[tp.TIMEPOOL_LAYOUT], ms.LAYOUT_SPECS[ms.STAGE2_LAYOUT]
    assert spec != stage2 and tp.TIMEPOOL_LAYOUT not in ms.LAYOUT_SPECS
    assert (spec.row_shape, spec.row_bytes, spec.grid_stride) == ((1, 1024), 4096, 32)
    assert spec.fields() == stage2.fields()


def test_index_round_trip_and_reuses_window_algorithm():
    spec = ms.KNOWN_LAYOUT_SPECS[tp.TIMEPOOL_LAYOUT]
    manifest = manifest_for([(165, 100), (82, 50), (32, 0)])
    entries = ms.build_index_entries(manifest, spec)
    payload = ms.index_payload(manifest, entries, spec=spec, layout=tp.TIMEPOOL_LAYOUT, mj_repo_commit="a" * 40)
    assert ms.parse_index(payload) == entries
    stage2 = ms.build_index_entries(manifest, ms.LAYOUT_SPECS[ms.STAGE2_LAYOUT])
    for t in (100, 132, 164):
        for a, b in zip(ms.visible_motion_rows(entries[0], t), ms.visible_motion_rows(stage2[0], t)):
            assert a.tobytes() == b.tobytes()
    payload["grid_stride"] = 16
    with pytest.raises(ValueError):
        ms.parse_index(payload)


def test_stage2_encoder_sampling_refuses_timepool_entries():
    manifest = manifest_for([(165, 100)])
    with pytest.raises(ValueError, match="stage2"):
        ms.stage2_encoder_sample_plan(ms.build_index_entries(manifest, ms.KNOWN_LAYOUT_SPECS[tp.TIMEPOOL_LAYOUT]))


# ── provenance ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("status", ["packed", "verified"])
def test_provenance_positive(status):
    value = provenance_for(status=status)
    ms.validate_timepool_provenance(value, status=status)
    assert value["encoder_ckpt"] is None and value["parameter_count"] == 0
    assert json.loads(json.dumps(value)) == value


@pytest.mark.parametrize("key", sorted(provenance_for()))
def test_every_top_level_field_required(key):
    value = provenance_for()
    del value[key]
    with pytest.raises(ValueError):
        ms.validate_timepool_provenance(value)


@pytest.mark.parametrize("group,key", [
    (group, key) for group in ("pool", "manifests", "wan_cache", "published", "routes", "window_map", "wan", "output")
    for key in provenance_for()[group]
])
def test_every_nested_field_required(group, key):
    value = provenance_for()
    del value[group][key]
    with pytest.raises(ValueError):
        ms.validate_timepool_provenance(value)


@pytest.mark.parametrize("key", list(tp.POOL_CONTRACT))
def test_every_pool_contract_key_is_bound(key):
    value = provenance_for()
    del value["pool"]["contract"][key]
    with pytest.raises(ValueError):
        ms.validate_timepool_provenance(value)
    value = provenance_for()
    value["pool"]["contract"][key] = False if value["pool"]["contract"][key] is True else "错"
    with pytest.raises(ValueError):
        ms.validate_timepool_provenance(value)


@pytest.mark.parametrize("change", [
    lambda p: p.update(encoder_ckpt=""),
    lambda p: p.update(encoder_ckpt="a" * 64),
    lambda p: p.update(encoder_ckpt=False),
    lambda p: p.update(encoder_kind="stage2"),
    lambda p: p.update(encoder_kind="wan_fixedpool"),
    lambda p: p.update(parameter_count=1),
    lambda p: p.update(parameter_count=False),
    lambda p: p.update(parameter_count=0.0),
    lambda p: p.update(schema="stage2-compat-v1"),
    lambda p: p.update(layout=ms.STAGE2_LAYOUT),
    lambda p: p.update(readout=ma.STAGE2_READOUT),
    lambda p: p.update(output_shape=[4, 256]),
    lambda p: p.update(train_head="HEAD"),
    lambda p: p.update(extra_field=1),
    lambda p: p.update(source_h5=[]),
    lambda p: p["source_h5"].append(dict(p["source_h5"][0])),
    lambda p: p["source_h5"][0].update(name="/abs/record_dataset_T.h5"),
    lambda p: p["pool"].update(source_sha256=""),
    lambda p: p["pool"].update(version="v0"),
    lambda p: p["published"].update(revision="main"),
    lambda p: p["routes"]["counts"].update(copy=99),
    lambda p: p["routes"]["numeric_modes"].update(fill={}),
    lambda p: p["window_map"].update(rows=99),
    lambda p: p["window_map"].update(relpath="/tmp/map.json"),
    lambda p: p["wan"].update(latent_mode="sample"),
    lambda p: p["wan"]["preprocess"].update(window_frames=17),
    lambda p: p["validation"]["POOL_TABLE_EQ"].update(verdict="POOL_TABLE_EQ=FAIL rows=3"),
    lambda p: p["validation"].pop("DATA_LINEAGE"),
    lambda p: p.update(validation=None),
])
def test_provenance_negative(change):
    value = provenance_for()
    change(value)
    with pytest.raises(ValueError):
        ms.validate_timepool_provenance(value)


def test_packed_must_not_carry_validation():
    with pytest.raises(ValueError, match="null"):
        ms.validate_timepool_provenance(provenance_for(), status="packed")


@pytest.mark.parametrize("kwargs", [
    dict(table_sha256="f" * 64), dict(index_sha256="f" * 64), dict(manifest_sha256="f" * 64), dict(num_rows=4),
])
def test_binding_mismatch_rejected(kwargs):
    with pytest.raises(ValueError, match="当前表不同"):
        ms.validate_timepool_provenance(provenance_for(), **kwargs)


def test_stage2_and_timepool_provenance_are_mutually_exclusive():
    stage2 = load_stage2_identity()
    ms.validate_stage2_provenance(stage2)
    with pytest.raises(ValueError):
        ms.validate_timepool_provenance(stage2)
    with pytest.raises(ValueError):
        ms.validate_stage2_provenance(provenance_for())
    with pytest.raises(ValueError):
        ms.validate_timepool_provenance({**provenance_for(), **stage2})


# ── MotionMeta / MotionStore ─────────────────────────────────────────────────


@pytest.mark.parametrize("status", ["packed", "verified"])
def test_store_round_trip_without_encoder_checkpoint(tmp_path, status):
    build_store(tmp_path, status=status)
    meta = ms.MotionMeta.load(tmp_path)
    assert meta.spec == ms.TIMEPOOL_LAYOUT_SPEC and meta.status == status
    assert meta.provenance["encoder_ckpt"] is None
    ms.run_full_checks(meta)
    with ms.MotionStore(tmp_path, verify_level="full") as store:
        assert store.table.shape == (meta.num_rows, 1, 1024)


@pytest.mark.parametrize("change", [
    lambda p: p.update(encoder_ckpt=""),
    lambda p: p.update(encoder_kind="stage2"),
    lambda p: p.pop("train_head"),
    lambda p: p["output"].update(motion_table_sha256="f" * 64),
    lambda p: p["output"].update(motion_index_sha256="f" * 64),
    lambda p: p["manifests"].update(episode_manifest_sha256="f" * 64),
    lambda p: p["window_map"].update(rows=999),
])
def test_store_load_rejects_bad_provenance(tmp_path, change):
    build_store(tmp_path, provenance=change)
    with pytest.raises(ValueError):
        ms.MotionMeta.load(tmp_path)


def test_store_load_requires_table_sha(tmp_path):
    build_store(tmp_path)
    rewrite_meta(tmp_path, lambda raw: raw["tables"]["motion_token"].pop("sha256"))
    with pytest.raises(ValueError, match="sha256"):
        ms.MotionMeta.load(tmp_path)


def test_store_verified_without_validation_rejected(tmp_path):
    build_store(tmp_path, status="packed")
    rewrite_meta(tmp_path, lambda raw: raw.update(status="verified"))
    with pytest.raises(ValueError):
        ms.MotionMeta.load(tmp_path)


def test_store_requires_full_scope_and_strict_types(tmp_path):
    build_store(tmp_path)
    rewrite_meta(tmp_path, lambda raw: raw.update(manifest_scope="subset_prefix"))
    with pytest.raises(ValueError, match="manifest_scope"):
        ms.MotionMeta.load(tmp_path)
    rewrite_meta(tmp_path, lambda raw: raw.update(manifest_scope="full", grid_stride=32.0))
    with pytest.raises(ValueError, match="grid_stride"):
        ms.MotionMeta.load(tmp_path)


def test_stage2_layout_refuses_timepool_provenance(tmp_path):
    build_store(tmp_path, layout=ms.STAGE2_LAYOUT, provenance=provenance_for())
    with pytest.raises(ValueError, match="stage2"):
        ms.MotionMeta.load(tmp_path)


def test_timepool_layout_refuses_stage2_provenance(tmp_path):
    build_store(tmp_path, provenance=load_stage2_identity())
    with pytest.raises(ValueError, match="timepool"):
        ms.MotionMeta.load(tmp_path)


def test_full_check_detects_table_tamper_with_consistent_sha(tmp_path):
    raw = build_store(tmp_path)
    path = tmp_path / ms.MOTION_TABLE_RELPATH
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    sha = ms.sha256_file(path)
    raw["tables"]["motion_token"]["sha256"] = sha
    raw["provenance"]["output"]["motion_table_sha256"] = sha
    (tmp_path / ms.META_RELPATH).write_text(json.dumps(raw))
    meta = ms.MotionMeta.load(tmp_path)
    with pytest.raises(ValueError, match="逐窗映射不符"):
        ms.run_full_checks(meta)


def test_full_check_detects_window_map_changes(tmp_path):
    build_store(tmp_path)
    meta = ms.MotionMeta.load(tmp_path)
    path = tmp_path / ms.TIMEPOOL_WINDOW_MAP_RELPATH
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="sha256"):
        ms.run_full_checks(meta)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        ms.run_full_checks(meta)
    path.write_bytes(original)
    ms.run_full_checks(meta)


def test_full_check_detects_pool_source_change(tmp_path):
    build_store(tmp_path, provenance=lambda p: p["pool"].update(source_sha256="f" * 64))
    with pytest.raises(ValueError, match="池化源码"):
        ms.run_full_checks(ms.MotionMeta.load(tmp_path))


@pytest.mark.parametrize("change,match", [
    (lambda w: w.__setitem__("source_m", w["source_m"] + 1), "source_m"),
    (lambda w: w.__setitem__("m", w["m"] + 1), "m="),
    (lambda w: w.__setitem__("key", "T_ep9_demo"), "key"),
    (lambda w: w.__setitem__("segment", "exec" if w["segment"] == "demo" else "demo"), "segment"),
    (lambda w: w.__setitem__("start_frame", w["start_frame"] + 1), "start_frame"),
    (lambda w: w.__setitem__("real_frames", 32), "real_frames"),
    (lambda w: w.__setitem__("pad_frames", 1), "pad_frames"),
    (lambda w: w.__setitem__("route", "other"), "route"),
    (lambda w: w.__setitem__("raw_latent_sha256", ""), "raw_latent_sha256"),
    (lambda w: w.pop("input_frames_sha256"), "字段"),
])
def test_window_map_row_negative(tmp_path, change, match):
    build_store(tmp_path)
    meta = ms.MotionMeta.load(tmp_path)
    payload = json.loads((tmp_path / ms.TIMEPOOL_WINDOW_MAP_RELPATH).read_text())
    ms.validate_timepool_window_map(payload, meta.provenance, meta.entries)
    change(payload["windows"][1])
    with pytest.raises(ValueError, match=match):
        ms.validate_timepool_window_map(payload, meta.provenance, meta.entries)


def test_window_map_extra_missing_row_and_counts_negative(tmp_path):
    build_store(tmp_path)
    meta = ms.MotionMeta.load(tmp_path)
    payload = json.loads((tmp_path / ms.TIMEPOOL_WINDOW_MAP_RELPATH).read_text())
    extra = copy.deepcopy(payload)
    extra["windows"].append(extra["windows"][-1])
    with pytest.raises(ValueError, match="行数"):
        ms.validate_timepool_window_map(extra, meta.provenance, meta.entries)
    short = copy.deepcopy(payload)
    short["windows"].pop()
    with pytest.raises(ValueError, match="缺行"):
        ms.validate_timepool_window_map(short, meta.provenance, meta.entries)
    flipped = copy.deepcopy(payload)
    flipped["windows"][0]["route"] = "copy" if flipped["windows"][0]["route"] == "fill" else "fill"
    with pytest.raises(ValueError, match="计数"):
        ms.validate_timepool_window_map(flipped, meta.provenance, meta.entries)
