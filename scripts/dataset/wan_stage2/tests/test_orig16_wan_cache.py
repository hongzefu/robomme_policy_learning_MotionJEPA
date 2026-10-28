"""缓存工具 CPU 夹具：验证真实文件映射、原始像素及数值路径边界。"""

import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "orig16_wan_cache.py"
spec = importlib.util.spec_from_file_location("orig16_wan_cache", SCRIPT)
cache = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cache
spec.loader.exec_module(cache)


def fixture_dataset(root, *, tasks=1, layout="group_first"):
    """D=49/C=100/T=115：每任务四个 copy、两个 fill，grid32 四窗。"""
    published, raw, latents = root / "published", root / "raw", root / "latents"
    index = published / "index"
    index.mkdir(parents=True)
    raw.mkdir()
    (published / "wan_chunk_latents").mkdir()
    inputs, sources, episodes, segments, windows, target = {}, {}, [], [], [], []
    latent = np.arange(cache.wc.CHUNK_BYTES // 4, dtype=np.float32).reshape(cache.wc.LAT_SHAPE)
    blob = np.ascontiguousarray(latent if layout == "group_first" else latent.transpose(1, 0, 2, 3)).tobytes()
    for task_id in range(tasks):
        task = f"Task{task_id}"
        name = f"record_dataset_{task}.h5"
        with h5py.File(raw / name, "w") as handle:
            episode = handle.create_group("episode_0")
            for frame in range(115):
                episode.create_dataset(
                    f"timestep_{frame}/obs/front_rgb",
                    shape=(256, 256, 3),
                    dtype="u1",
                    fillvalue=frame,
                    chunks=(256, 256, 3),
                )
        digest = cache.sha_file(raw / name)
        inputs[name] = {"size": (raw / name).stat().st_size, "sha256": digest}
        sources[task] = {"h5": {"file": name, **inputs[name]}}
        episodes.append(
            {
                "task": task,
                "episode": 0,
                "h5_file": "source/h5/" + name,
                "source_sha256": digest,
                "demo_end": 49,
                "completed_index": 100,
                "num_timesteps": 115,
            }
        )
        target.append(
            {
                "h5_file": name,
                "raw_ep_idx": 0,
                "global_episode_idx": task_id,
                "num_timesteps": 115,
                "exec_start_idx": 49,
                "exec_sample_offset": task_id * 66,
            }
        )
        for kind, lo, hi in (("demo", 0, 49), ("exec", 49, 100)):
            key = f"{task}_ep0_{kind}"
            file = f"wan_chunk_latents/{key}.bin"
            chunks = hi - lo - 32
            segments.append(
                {
                    "name": key,
                    "task": task,
                    "episode": 0,
                    "segment": kind,
                    "h5_file": "source/h5/" + name,
                    "lo": lo,
                    "hi": hi,
                    "original_hi": 49 if kind == "demo" else 115,
                    "frames": hi - lo,
                    "num_chunks": chunks,
                    "binfile": file,
                }
            )
            (published / file).write_bytes(blob * chunks)
            windows.extend(
                {
                    "task": task,
                    "episode": 0,
                    "segment": kind,
                    "start": offset,
                    "original_frame_idx": lo + offset,
                    "window_start": lo + offset,
                    "window_end": lo + offset + 33,
                    "valid_frames": 33,
                    "binfile": file,
                    "byte_offset": offset * cache.wc.CHUNK_BYTES,
                }
                for offset in range(chunks)
            )
    for name, rows in (("episodes", episodes), ("segments", segments), ("windows", windows)):
        pq.write_table(pa.Table.from_pylist(rows), index / f"{name}.parquet")
    provenance = {
        "axis_order": layout,
        "latent_shape": list(cache.LATENT_SHAPES[layout]),
        "dtype": "float32",
        "byte_order": "little",
        "chunk_bytes": cache.wc.CHUNK_BYTES,
        "affine_normalized": False,
        "window_frames": 33,
        "stride": 1,
        "numeric_mode": {"weights": "fp32", "cudnn_conv_tf32": True},
        "vae_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "vae_state_sha256": "a" * 64,
    }
    cache.write_json(published / "provenance.json", provenance)
    cache.write_json(published / "source_manifest.json", {"files": sources})
    refresh_sums(published)
    manifest = {"episodes": target, "raw_dir": str(raw)}
    manifest["sha256"] = cache.wc.manifest_sha256(manifest)
    cache.write_json(root / "manifest.json", manifest)
    cache.write_json(root / "input.json", {"files": inputs, "count": tasks})
    plan = cache.build_plan(root / "manifest.json", root / "input.json", index)
    cache.write_json(root / "plan.json", plan)
    return {"plan": plan, "root": root, "published": published, "raw": raw, "latents": latents, "latent": latent}


def refresh_sums(root):
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text("".join(f"{cache.sha_file(p)}  {p.relative_to(root)}\n" for p in files))


def import_fixture(data):
    plan = data["plan"]
    cache.verify_inputs(plan, data["published"], data["raw"])
    calls = []

    def encode(frames):
        calls.append(frames.copy())
        return np.full(cache.wc.LAT_SHAPE, int(frames[-1, 0, 0, 0]), dtype=np.float32)

    for item in plan["segments"]:
        rows = [r for r in plan["windows"] if r["key"] == item["key"]]
        cache.import_segment(
            plan,
            item,
            rows,
            data["published"],
            data["raw"],
            data["latents"],
            encode,
            {"weights": "fp32", "cudnn_conv_tf32": False},
            {"vae_id": plan["provenance"]["vae_id"]},
            {"worker": "cpu-fixture", "hostname": "fixture", "gpu_uuid": "none", "pid": 1},
        )
    receipt = cache.aggregate_import(plan, data["latents"], data["raw"])
    cache.write_json(data["root"] / "receipt.json", receipt)
    return receipt, calls


def test_plan_copy_fill_and_whole_file_sizes(tmp_path):
    data = fixture_dataset(tmp_path)
    plan = data["plan"]
    assert plan["counts"] == {
        "windows": 6,
        "copy": 4,
        "fill": 2,
        "grid32": 4,
        "segments": 2,
        "download_files": 2,
        "download_bytes": 36 * cache.wc.CHUNK_BYTES,
    }
    assert [r["source_row"] for r in plan["windows"]] == [0, 16, None, 0, 16, None]
    assert plan["windows"][2]["frame_ids"] == list(range(32, 49)) + [48] * 16
    assert plan["windows"][-1]["frame_ids"] == list(range(81, 114))
    assert cache.load_plan(tmp_path / "plan.json") == plan


@pytest.mark.parametrize("layout", ["group_first", "channel_first"])
def test_import_only_fills_missing_windows_and_preserves_axes(tmp_path, layout):
    data = fixture_dataset(tmp_path, layout=layout)
    receipt, calls = import_fixture(data)
    assert len(calls) == 2
    assert calls[0][:, 0, 0, 0].tolist() == list(range(32, 49)) + [48] * 16
    assert calls[1][:, 0, 0, 0].tolist() == list(range(81, 114))
    rows = np.fromfile(data["latents"] / "Task0_ep0_demo.bin", dtype=np.float32).reshape((3, *cache.wc.LAT_SHAPE))
    np.testing.assert_array_equal(rows[0], data["latent"])
    np.testing.assert_array_equal(rows[1], data["latent"])
    assert np.all(rows[2] == 48)
    assert receipt["numeric_modes"]["copy"] != receipt["numeric_modes"]["fill"]
    assert any(x["path"] == "index/episodes.parquet" for x in receipt["published"]["files"])
    total = cache.read_json(data["latents"] / "metadata.json")
    assert total["segments"]["Task0_ep0_demo"]["min_real"] == 17


def test_fetch_fixed_revision_and_whole_file_sha(tmp_path):
    data = fixture_dataset(tmp_path)
    seen = []

    def download(**kwargs):
        assert kwargs["repo_id"] == cache.REPO
        assert kwargs["revision"] == cache.REVISION
        seen.append(kwargs["filename"])
        return data["published"] / kwargs["filename"]

    with pytest.raises(ValueError, match="revision"):
        cache.fetch_plan(data["plan"], "main", tmp_path / "fetched", downloader=download)
    result = cache.fetch_plan(data["plan"], cache.REVISION, tmp_path / "fetched", downloader=download)
    assert len(seen) == 8
    assert result["status"] == "PASS"
    file = tmp_path / "fetched/wan_chunk_latents/Task0_ep0_demo.bin"
    with file.open("r+b") as stream:
        # 未命中行也必须参与整文件 SHA。
        stream.seek(cache.wc.CHUNK_BYTES)
        stream.write(b"oops")
    with pytest.raises(ValueError, match="SHA"):
        cache.fetch_plan(data["plan"], cache.REVISION, tmp_path / "fetched", downloader=download)


@pytest.mark.parametrize("fault", ["input_h5", "axis", "window", "duplicate"])
def test_plan_rejects_inconsistent_source_contract(tmp_path, fault):
    data = fixture_dataset(tmp_path)
    if fault == "input_h5":
        manifest = cache.read_json(tmp_path / "input.json")
        manifest["files"]["record_dataset_Task0.h5"]["sha256"] = "f" * 64
        cache.write_json(tmp_path / "input.json", manifest)
    elif fault == "axis":
        path = data["published"] / "provenance.json"
        value = cache.read_json(path)
        value["axis_order"] = "channel_first"
        cache.write_json(path, value)
    else:
        path = data["published"] / "index/windows.parquet"
        rows = pq.read_table(path).to_pylist()
        if fault == "window":
            rows[0]["byte_offset"] += cache.wc.CHUNK_BYTES
        else:
            rows.append(rows[0])
        pq.write_table(pa.Table.from_pylist(rows), path)
    refresh_sums(data["published"])
    with pytest.raises(ValueError, match="不相等|不符"):
        cache.build_plan(tmp_path / "manifest.json", tmp_path / "input.json", data["published"] / "index")


def test_import_rejects_nan_and_source_tampering(tmp_path):
    data = fixture_dataset(tmp_path)
    item = data["plan"]["segments"][0]
    rows = [r for r in data["plan"]["windows"] if r["key"] == item["key"]]
    with pytest.raises(ValueError, match="finite"):
        cache.import_segment(
            data["plan"],
            item,
            rows,
            data["published"],
            data["raw"],
            data["latents"],
            lambda window: np.full(cache.wc.LAT_SHAPE, np.nan, dtype=np.float32),
            {},
            {},
            {},
        )
    path = data["raw"] / "record_dataset_Task0.h5"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        cache.verify_inputs(data["plan"], data["published"], data["raw"])


def test_plan_digest_and_safe_paths_fail_closed(tmp_path):
    data = fixture_dataset(tmp_path)
    data["plan"]["windows"][0]["source_row"] = 1
    cache.write_json(tmp_path / "plan.json", data["plan"])
    with pytest.raises(ValueError, match="摘要"):
        cache.load_plan(tmp_path / "plan.json")
    for name in ("../secret", "/absolute", "x/../../secret"):
        with pytest.raises(ValueError, match="非法相对路径"):
            cache.safe_path(tmp_path, name)
