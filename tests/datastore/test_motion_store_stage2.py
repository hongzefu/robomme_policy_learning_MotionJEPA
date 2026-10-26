"""新布局公式、来源负控与旧三布局逐字节回归。"""
import copy
import json
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest

from mme_vla_suite.datastore import motion_store as ms
from mme_vla_suite.datastore.manifest import manifest_sha256

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = "1383aba0fd77b259f0b32eadb47ea7792190145e"


def baseline_module(relative, name):
    source = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT, text=True)
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, str(ROOT / relative), "exec"), module.__dict__)
    return module


def manifest_for(specs):
    rows, offset, total = [], 0, 0
    for g, (nt, es) in enumerate(specs):
        rows.append(dict(global_episode_idx=g, h5_file="record_dataset_T.h5", raw_ep_idx=g,
                         num_timesteps=nt, exec_start_idx=es, exec_samples=nt-es,
                         exec_sample_offset=offset, total_sample_offset=total, shard_idx=0))
        offset += nt-es
        total += nt
    value = dict(version=1, raw_dir="/scratch/hongze/合成原数据", canonical_order=["record_dataset_T.h5"],
                 num_shards=1, shard_load_timesteps=[total], episodes=rows,
                 totals=dict(episodes=len(rows), timesteps=total, exec_samples=offset))
    value["sha256"] = manifest_sha256(value)
    return value


def stage2_identity():
    leaves = {f"state_{i:02d}": dict(shape=[1], dtype="float32", sha256="a"*64) for i in range(65)}
    encoder = dict(run_name="stage2-full144-1025-a", checkpoint_name="checkpoint_epoch_105.pt", epoch=105,
                   global_step=7770, state_key="model", weight_type="ema", arch=ms.STAGE2_ARCH,
                   checkpoint_sha256="a"*64, config_sha256="b"*64, fingerprint_sha256="c"*64,
                   source_times=[0,3,5,8], output_shape=[1,1024])
    pin = dict(mj_repo_commit="a"*40, export_source_sha256="b"*64, encoder_source_sha256="c"*64,
               vae_adapter_sha256="d"*64, dependency_lock_sha256="e"*64)
    numerics = dict(dtype="float32", autocast=False, matmul_allow_tf32=False, cudnn_allow_tf32=False,
                    batch=1, torch="2.9.0", cuda="12.8", cudnn=91002, numpy="2.4.4", gpu_name="A100", driver="595")
    vae = dict(vae_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", vae_state_sha256="f"*64, vae_dtype="float32",
               latent_mode="mode", batch=1, tf32="off", amp="off", torch="2.9.0", cuda="12.8", cudnn=91002,
               diffusers="0.39.0", module_sha256="1"*64, encoder_src_sha256="2"*64)
    online = dict(vae, adapter_sha256=pin["vae_adapter_sha256"], numerics=numerics,
                  preprocess=dict(frame_size=256, input_dtype="uint8", input_layout="THWC", window_frames=33))
    evidence = {key: dict(path=f"/scratch/hongze/{key}.json",sha256="a"*64)
                for key in ("G0","G2","G3","restore_receipt","SOURCE_PIN")}
    return dict(encoder=encoder, encoder_state_sha256=dict(algorithm="named-tensors-v1", leaves=leaves,
                sha256=ms.sha256_bytes(ms.canonical_json_bytes(leaves))), source_pin=pin, numerics=numerics,
                vae=vae, stage2_compat=dict(schema="stage2-compat-v1", restore_receipt_sha256="a"*64,
                                         historical_vae=vae, online_vae=online, evidence=evidence))


@pytest.mark.parametrize("layout", [name for name in ms.LAYOUT_SPECS if name != ms.STAGE2_LAYOUT])
def test_legacy_baseline_1000_cases_byte_exact(layout):
    old = baseline_module("src/mme_vla_suite/datastore/motion_store.py", "baseline_motion_store")
    rng = np.random.default_rng(3)
    for _ in range(1000):
        es, length = int(rng.integers(0,1301)), int(rng.integers(1,1301))
        manifest = manifest_for([(es+length, es)])
        before = old.build_index_entries(manifest, old.LAYOUT_SPECS[layout])
        after = ms.build_index_entries(manifest, ms.LAYOUT_SPECS[layout])
        t = es + int(rng.integers(0,length))
        for a,b in zip(old.visible_motion_rows(before[0],t), ms.visible_motion_rows(after[0],t)):
            assert a.tobytes() == b.tobytes()
        args = dict(layout=layout, mj_repo_commit="a"*40)
        expected = old.index_payload(manifest,before,spec=old.LAYOUT_SPECS[layout],**args)
        actual = ms.index_payload(manifest,after,spec=ms.LAYOUT_SPECS[layout],**args)
        assert json.dumps(expected).encode() == json.dumps(actual).encode()


def test_stage2_even_grid_round_trip():
    for length in range(1301):
        for minimum in (17,33):
            assert ms.segment_grid_starts(length,minimum,32) == ms.segment_grid_starts(length,minimum)[::2]
    assert ms.segment_grid_starts(100,17,32) == [0,32,64]
    manifest = manifest_for([(165,100),(82,50),(32,0)])
    spec = ms.LAYOUT_SPECS[ms.STAGE2_LAYOUT]
    entries = ms.build_index_entries(manifest,spec)
    payload = ms.index_payload(manifest,entries,spec=spec,layout=ms.STAGE2_LAYOUT,mj_repo_commit="a"*40)
    assert ms.parse_index(payload) == entries
    assert entries[0].demo.num_chunks == 84 and entries[0].demo.num_grid == 3
    assert ms.max_visible_count(entries[0]) == 5
    assert spec.row_bytes == 4096
    payload["grid_stride"] = 16
    with pytest.raises(ValueError): ms.parse_index(payload)


@pytest.mark.parametrize("group", ms.STAGE2_GROUPS)
def test_provenance_missing_group(group):
    provenance = stage2_identity()
    ms.validate_stage2_provenance(provenance)
    del provenance[group]
    with pytest.raises(ValueError): ms.validate_stage2_provenance(provenance)


@pytest.mark.parametrize("group", ["encoder","source_pin","numerics","stage2_compat"])
def test_every_required_identity_field_is_required(group):
    original = stage2_identity()
    for field in original[group]:
        broken = copy.deepcopy(original)
        del broken[group][field]
        with pytest.raises(ValueError, match="stage2|SHA256|encoder"):
            ms.validate_stage2_provenance(broken)


def test_mutated_state_summary_and_live_weights_rejected():
    for mutate in (lambda p: p["encoder"].update(state_key="model_live"),
                   lambda p: p["encoder_state_sha256"]["leaves"]["state_00"].update(sha256="b"*64),
                   lambda p: p["numerics"].update(batch=True),
                   lambda p: p["stage2_compat"]["online_vae"].update(amp="fp16")):
        value = copy.deepcopy(stage2_identity()); mutate(value)
        with pytest.raises(ValueError): ms.validate_stage2_provenance(value)


def test_matching_but_wrong_vae_precision_is_rejected():
    value=stage2_identity()
    for group in (value["vae"],value["stage2_compat"]["historical_vae"],value["stage2_compat"]["online_vae"]):
        group["vae_dtype"]="float16"
    with pytest.raises(ValueError,match="固定模型、精度"):ms.validate_stage2_provenance(value)
