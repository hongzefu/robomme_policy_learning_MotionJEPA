"""原版 H5 的 G0：真实像素、D/C/T 边界、收据映射和三路线抽样。"""
from __future__ import annotations

import copy
import importlib.util
import json
import pickle
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SPEC = importlib.util.spec_from_file_location("legacy_g0_fixture", Path(__file__).with_name("test_stage2_validation_inputs.py"))
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)
g0 = legacy.g0


@pytest.fixture(autouse=True)
def receipt_contract(monkeypatch):
    """S1 先合入时记录外部调用；S2 合入后直接运行正式合同校验器。"""
    validator = getattr(g0.ms, "validate_wan_cache_import_receipt", None)
    calls = []

    def validate(receipt, **kwargs):
        calls.append(kwargs)
        if validator is not None:
            return validator(receipt, **kwargs)
        # 本夹具只替代另一个实施块的入口；S1 自身的像素与索引校验全部真实执行。
        assert kwargs == {"manifest_sha256": receipt["manifest_sha256"],
                          "input_manifest_sha256": receipt["input_manifest_sha256"]}
        return {(row["key"], row["m"]): row for row in receipt["windows"]}

    monkeypatch.setattr(g0.ms, "validate_wan_cache_import_receipt", validate, raising=False)
    return calls


def make_h5_dataset(root, tasks=1):
    sample = legacy.make_dataset(root, [(49,)] * tasks, write_frames=False)
    sample.h5_root = root / "h5"
    sample.h5_root.mkdir()
    sample.input_path = root / "input_manifest.json"
    sample.index_path = root / "index/episodes.parquet"
    sample.index_path.parent.mkdir()
    files, episodes, windows, outputs = {}, [], [], []
    for ep in sample.manifest["episodes"]:
        path = sample.h5_root / ep["h5_file"]
        with h5py.File(path, "w") as handle:
            for frame in range(ep["num_timesteps"]):
                handle.create_dataset(f"episode_1/timestep_{frame}/obs/front_rgb",
                                      data=legacy.pixels([frame])[0], compression="gzip")
        for frame in range(ep["exec_start_idx"], ep["num_timesteps"]):
            sample_index = ep["exec_sample_offset"] + frame - ep["exec_start_idx"]
            record = {"image": legacy.pixels([frame])[0], "epis_idx": np.array([ep["global_episode_idx"]], np.int32),
                      "step_idx": np.array([frame], np.int32), "exec_start_idx": np.array([49], np.int32),
                      "is_demo": np.array([False])}
            (sample.source / "data" / f"{sample_index}.pkl").write_bytes(pickle.dumps(record))
        files[path.name] = {"size": path.stat().st_size, "sha256": g0.ms.sha256_file(path)}
        episodes.append({"h5_file": f"source/h5/{path.name}", "episode": 1,
                         "task": g0.ms.task_of_h5(path.name), "source_sha256": files[path.name]["sha256"],
                         "demo_end": 49, "completed_index": 98, "num_timesteps": 114})
        for kind in ("demo", "exec"):
            key = g0.ms.segment_key(path.name, 1, kind)
            metadata = g0.read_json(sample.latents / f"{key}.metadata.json")
            outputs.append({"path": f"{key}.bin", "bytes": metadata["bytes"], "sha256": metadata["sha256"]})
            data = (sample.latents / f"{key}.bin").read_bytes()
            for row in metadata["rows"]:
                start, real, pad, m = (row[k] for k in ("start_global_frame", "real_frames", "pad_frames", "m"))
                route = "copy" if start + 33 <= (49 if kind == "demo" else 98) else "fill"
                windows.append({"key": key, "m": m, "route": route,
                                "source_file": f"published/{key}.bin" if route == "copy" else None,
                                "source_row": m * 16 if route == "copy" else None,
                                "source_offset": m * 16 * g0.CHUNK_BYTES if route == "copy" else None,
                                "source_layout": "group_first" if route == "copy" else None,
                                "frame_ids": list(range(start, start + real)) + [start + real - 1] * pad,
                                "real_frames": real, "pad_frames": pad, "pad_source_frame": row["pad_source_frame"],
                                "input_frames_sha256": row["input_frames_sha256"],
                                "latent_sha256": g0.ms.sha256_bytes(data[m*g0.CHUNK_BYTES:(m+1)*g0.CHUNK_BYTES]),
                                "output_file": f"{key}.bin", "output_offset": m * g0.CHUNK_BYTES})
    pq.write_table(pa.Table.from_pylist(episodes), sample.index_path)
    legacy.write_json(sample.input_path, {"count": tasks, "files": files})
    sample.receipt_path = root / "receipt.json"
    receipt = {"schema": "orig16-wan-cache-import-v1", "kind": "published-cache-with-local-fill", "status": "PASS",
               "manifest_sha256": sample.manifest["sha256"], "manifest_file_sha256": g0.ms.sha256_file(sample.manifest_path),
               "input_manifest_sha256": g0.ms.sha256_file(sample.input_path), "importer_sha256": "a"*64,
               "source_h5_files": [{"name": name, "bytes": x["size"], "sha256": x["sha256"]} for name, x in files.items()],
               "published": {"repo": "fixture/cache", "revision": "b"*40, "sha256sums_sha256": "c"*64,
                             "files": [{"path": "index/episodes.parquet", "bytes": sample.index_path.stat().st_size,
                                        "sha256": g0.ms.sha256_file(sample.index_path)},
                                       *[{"path": f"published/{x['path']}", "bytes": 64*g0.CHUNK_BYTES,
                                          "sha256": "d"*64} for x in outputs]]},
               "latents": {"root": str(sample.latents), "files": outputs,
                           "metadata_sha256": g0.ms.sha256_file(sample.latents / "metadata.json")},
               "vae": {"vae_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "latent_mode": "mode", "vae_state_sha256": "e"*64},
               "preprocess": {"frame_size": 256, "input_dtype": "uint8", "input_layout": "THWC",
                              "window_frames": 33, "augmentation": "none"},
               "numeric_modes": {"copy": {"dtype": "float16"}, "fill": {"dtype": "float32"}},
               "windows": windows, "counts": {"windows": len(windows),
                   "copy": sum(w["route"] == "copy" for w in windows), "fill": sum(w["route"] == "fill" for w in windows)}}
    legacy.write_json(sample.receipt_path, receipt)
    return sample


@pytest.fixture
def sample(tmp_path):
    return make_h5_dataset(tmp_path)


def context(sample):
    return g0.read_h5_context(sample.manifest, sample.input_path, sample.index_path.parent, sample.receipt_path)


def change_receipt(sample, update):
    receipt = g0.read_json(sample.receipt_path)
    update(receipt)
    legacy.write_json(sample.receipt_path, receipt)


def update_index(sample, update, *, repin=True):
    rows = pq.read_table(sample.index_path).to_pylist()
    update(rows)
    pq.write_table(pa.Table.from_pylist(rows), sample.index_path)
    if repin:
        change_receipt(sample, lambda r: r["published"]["files"][0].update(
            bytes=sample.index_path.stat().st_size, sha256=g0.ms.sha256_file(sample.index_path)))


def test_real_h5_cli_covers_64_windows_and_all_routes(tmp_path, capsys, receipt_contract):
    sample = make_h5_dataset(tmp_path / "inputs", tasks=16)
    out = tmp_path / "G0.json"
    g0.main(["--manifest", str(sample.manifest_path), "--latents", str(sample.latents),
             "--source", str(sample.source), "--h5-root", str(sample.h5_root),
             "--input-manifest", str(sample.input_path), "--mj-index", str(sample.index_path.parent),
             "--cache-import-receipt", str(sample.receipt_path), "--out", str(out)])
    result = g0.read_json(out)
    assert "RAW_WINDOWS_READY=PASS" in capsys.readouterr().out
    assert result["counts"] == {"windows": 64, "tasks": 16, "demo_tail": 16, "sha_mismatches": 0,
                                "copy": 32, "demo-fill": 16, "exec-fill": 16}
    assert result["cache_import_receipt_sha256"] == g0.ms.sha256_file(sample.receipt_path)
    assert receipt_contract[-1] == {"manifest_sha256": sample.manifest["sha256"],
                                    "input_manifest_sha256": g0.ms.sha256_file(sample.input_path)}
    assert all(w["sources"][0]["kind"] == ("raw_h5" if w["segment_kind"] == "demo" else "exec_pkl")
               for w in result["windows"])
    inputs, mapping, routes, _ = context(sample)
    selected = g0.select_windows(sample.manifest, sample.latents, routes=routes)
    reordered = copy.deepcopy(sample.manifest)
    reordered["episodes"].reverse()
    assert selected == g0.select_windows(reordered, sample.latents, routes=routes)
    tail = next(item for item in selected if routes[item[2], item[3]] == "demo-fill")
    pixels, row = g0.materialize_h5_window(tail, h5_root=sample.h5_root, inputs=inputs,
                                          mapping=mapping[tail[2], tail[3]], file_sha_cache={})
    assert row["frame_ids"] == list(range(32, 49)) + [48]*16
    np.testing.assert_array_equal(pixels, legacy.pixels(row["frame_ids"]))


@pytest.mark.parametrize("field,value", [("demo_end", 48), ("num_timesteps", 113),
                                        ("completed_index", 114), ("source_sha256", "f"*64)])
def test_mj_boundaries_and_h5_identity_must_agree(sample, field, value):
    update_index(sample, lambda rows: rows[0].update({field: value}))
    with pytest.raises(ValueError, match="exec_start_idx|num_timesteps|路线|源 H5"):
        context(sample)


def test_index_bytes_are_bound_to_receipt(sample):
    update_index(sample, lambda rows: rows[0].update(completed_index=97), repin=False)
    with pytest.raises(ValueError, match="episodes.parquet"):
        context(sample)


def test_duplicate_episode_is_not_silently_overwritten(sample):
    update_index(sample, lambda rows: rows.append(copy.deepcopy(rows[0])))
    with pytest.raises(ValueError, match="重复"):
        context(sample)


@pytest.mark.parametrize("fault", ["route", "frame_ids", "missing", "extra", "source_h5"])
def test_receipt_mapping_must_cover_exact_windows(sample, fault):
    def mutate(receipt):
        row = receipt["windows"][0]
        if fault == "route": row["route"] = "fill"
        elif fault == "frame_ids": row["frame_ids"][0] += 1
        elif fault == "missing": receipt["windows"].pop()
        elif fault == "extra": receipt["windows"].append({**row, "m": 99})
        else: receipt["source_h5_files"][0]["sha256"] = "f"*64
    change_receipt(sample, mutate)
    with pytest.raises(ValueError):
        context(sample)


@pytest.mark.parametrize("fault", ["file_sha", "pixel_sha", "receipt_sha", "dtype", "shape", "missing_frame"])
def test_raw_h5_and_input_pixels_fail_closed(sample, fault):
    inputs, mapping, routes, _ = context(sample)
    item = next(x for x in g0.select_windows(sample.manifest, sample.latents, tasks=1,
                                            min_demo_tail=1, routes=routes) if x[1] == "demo" and x[3] == 0)
    path = sample.h5_root / item[0]["h5_file"]
    if fault == "receipt_sha":
        mapping[item[2], item[3]]["input_frames_sha256"] = "f"*64
    else:
        with h5py.File(path, "r+") as handle:
            address = "episode_1/timestep_0/obs/front_rgb"
            if fault == "file_sha": handle.attrs["changed"] = True
            elif fault == "pixel_sha": handle[address][...] = 1
            else:
                del handle[address]
                if fault != "missing_frame":
                    handle.create_dataset(address, data=np.zeros((128,256,3) if fault == "shape" else (256,256,3),
                                                                 np.float32 if fault == "dtype" else np.uint8))
        if fault != "file_sha":
            inputs[path.name] = {"bytes": path.stat().st_size, "sha256": g0.ms.sha256_file(path)}
    with pytest.raises((ValueError, KeyError)):
        g0.materialize_h5_window(item, h5_root=sample.h5_root, inputs=inputs,
                                mapping=mapping[item[2], item[3]], file_sha_cache={})


def test_all_three_routes_are_required(sample):
    inputs, mappings, routes, _ = context(sample)
    routes = {key: "copy" if value == "exec-fill" else value for key, value in routes.items()}
    with pytest.raises(ValueError, match="exec-fill"):
        g0.select_windows(sample.manifest, sample.latents, tasks=1, min_demo_tail=1, routes=routes)


@pytest.mark.parametrize("present", range(1, 4))
def test_original_h5_arguments_are_all_or_nothing(sample, tmp_path, present):
    flags = ["--h5-root", str(sample.h5_root), "--input-manifest", str(sample.input_path),
             "--mj-index", str(sample.index_path)]
    out = tmp_path / "G0.json"
    result = subprocess.run([sys.executable, str(legacy.SCRIPT), "--manifest", str(sample.manifest_path),
                             "--latents", str(sample.latents), "--source", str(sample.source),
                             "--out", str(out), *flags[:2*present]], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "四项须齐全" in result.stderr
    assert not out.exists() and "RAW_WINDOWS_READY=PASS" not in result.stdout


@pytest.mark.parametrize("fault", ["exec_identity", "exec_pixels", "latent_bytes", "manifest_file", "metadata_file"])
def test_build_rejects_export_or_binding_corruption(sample, monkeypatch, fault):
    if fault.startswith("exec"):
        path = sample.source / "data/7.pkl"
        value = pickle.loads(path.read_bytes())
        value["step_idx" if fault == "exec_identity" else "image"] += 1
        path.write_bytes(pickle.dumps(value))
    elif fault == "latent_bytes":
        path = sample.latents / "Task00_ep1_demo.bin"
        with path.open("r+b") as stream:
            stream.write(b"corrupt")
    elif fault == "manifest_file":
        change_receipt(sample, lambda r: r.update(manifest_file_sha256="f"*64))
    else:
        change_receipt(sample, lambda r: r["latents"].update(metadata_sha256="f"*64))
    select = g0.select_windows
    monkeypatch.setattr(g0, "select_windows", lambda m, p, **kw: select(m, p, tasks=1, min_demo_tail=1, **kw))
    monkeypatch.setattr(g0, "TASKS", 1)
    monkeypatch.setattr(g0, "MIN_DEMO_TAIL", 1)
    with pytest.raises(ValueError, match="step_idx|原帧 SHA|latent 文件|manifest 或 latent metadata"):
        g0.build_inputs(sample.manifest_path, sample.latents, sample.source, sample.demo_root,
                         h5_root=sample.h5_root, input_manifest=sample.input_path,
                         mj_index=sample.index_path, cache_import_receipt=sample.receipt_path)
