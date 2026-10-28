"""真实缓存导入器产物与 S2 唯一校验器的互操作测试。"""

import copy
import importlib.util
from pathlib import Path
import pickle

import numpy as np
import pytest

from mme_vla_suite.datastore import motion_store as ms

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "orig16_fixture", ROOT / "scripts/dataset/wan_stage2/tests/test_orig16_wan_cache.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
cache = fixture.cache


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    data = fixture.fixture_dataset(tmp_path_factory.mktemp("cache64"), tasks=16)
    receipt, _ = fixture.import_fixture(data)
    windows = []
    sources = {r["name"]: r for r in receipt["source_h5_files"]}
    for row in receipt["windows"]:
        if row["m"] % 2:
            continue
        sample = {
            "key": row["key"],
            "task": cache.wc.task_of_h5(row["h5_file"]),
            "episode": row["raw_ep_idx"],
            "segment_kind": row["segment_kind"],
            "source_grid": row["m"],
            **{
                k: row[k]
                for k in ("frame_ids", "real_frames", "pad_frames", "pad_source_frame", "input_frames_sha256", "route")
            },
            "cache_route": "copy" if row["route"] == "copy" else row["segment_kind"] + "-fill",
            "sources": [
                {
                    "kind": "raw_h5",
                    "path": str(data["raw"] / row["h5_file"]),
                    "sha256": sources[row["h5_file"]]["sha256"],
                    "episode": row["raw_ep_idx"],
                }
            ],
            "source_frame_indices": [{"source": 0, "index": f} for f in row["frame_ids"]],
            "latent_path": str(data["latents"] / row["output_file"]),
            "latent_block_sha256": row["latent_sha256"],
        }
        if row["segment_kind"] == "exec":
            sample["raw_h5"] = sample["sources"][0]
            sample["sources"] = []
            sample["source_frame_indices"] = [{"source": i, "index": 0} for i in range(33)]
            g = int(sample["task"].removeprefix("Task"))
            for frame in row["frame_ids"]:
                path = data["root"] / f"pkl-{g}-{frame}.pkl"
                if not path.exists():
                    value = {
                        "image": np.full((256, 256, 3), frame, dtype=np.uint8),
                        "epis_idx": np.array([g], dtype=np.int32),
                        "step_idx": np.array([frame], dtype=np.int32),
                        "exec_start_idx": np.array([49], dtype=np.int32),
                        "is_demo": np.array([False], dtype=np.bool_),
                    }
                    path.write_bytes(pickle.dumps(value))
                sample["sources"].append({"kind": "exec_pkl", "path": str(path), "sha256": cache.sha_file(path)})
        windows.append(sample)
    samples = {
        "schema": "stage2-validation-inputs-v1",
        "gate": "G0",
        "status": "PASS",
        "cache_import_receipt_sha256": cache.sha_file(data["root"] / "receipt.json"),
        "manifest_sha256": receipt["manifest_sha256"],
        "latent_metadata_sha256": receipt["latents"]["metadata_sha256"],
        "windows": windows,
    }
    samples_path = data["root"] / "samples.json"
    cache.write_json(samples_path, samples)
    return data, receipt, samples


def test_real_receipt_and_g2_accept_precision_difference(imported):
    data, receipt, samples = imported
    mapping = ms.validate_wan_cache_import_receipt(receipt)
    assert len(mapping) == 96
    assert receipt["numeric_modes"]["copy"] != receipt["numeric_modes"]["fill"]
    result = cache.check_import(data["root"] / "samples.json", data["root"] / "receipt.json", data["raw"])
    assert result["metrics"]["windows"] == 64
    assert result["metrics"]["demo_tail"] == 16
    assert result["output_numerics"]["status"] == "NOT_REQUIRED"
    assert (
        ms.validate_stage2_g2_evidence(
            result,
            checkpoint_sha256=None,
            samples_sha256=cache.sha_file(data["root"] / "samples.json"),
            source_pin=None,
            numerics=None,
            online_vae=receipt["vae"],
            demo_tail=16,
            cache_import_receipt=receipt,
            cache_import_receipt_sha256=cache.sha_file(data["root"] / "receipt.json"),
        )
        == "PASS"
    )


@pytest.mark.parametrize("fault", ["missing", "axis", "augmentation", "offset", "source_h5"])
def test_shared_validator_rejects_bad_contract(imported, fault):
    _, original, _ = imported
    receipt = copy.deepcopy(original)
    if fault == "missing":
        receipt["windows"].pop()
    elif fault == "axis":
        receipt["windows"][0]["source_layout"] = "flat"
    elif fault == "augmentation":
        receipt["preprocess"]["augmentation"] = "crop"
    elif fault == "offset":
        receipt["windows"][0]["source_offset"] += cache.wc.CHUNK_BYTES
    else:
        receipt["windows"][0]["h5_file"] = "record_dataset_Alien.h5"
    with pytest.raises(ValueError, match="不符|不在"):
        ms.validate_wan_cache_import_receipt(receipt)


@pytest.mark.parametrize("fault", ["frame", "frame_order", "source", "input_sha", "latent_sha", "receipt_pin"])
def test_g2_rejects_changed_samples(imported, tmp_path, fault):
    data, _, original = imported
    samples = copy.deepcopy(original)
    row = samples["windows"][0]
    if fault == "frame":
        row["frame_ids"][0] += 1
    elif fault == "frame_order":
        row["source_frame_indices"][0], row["source_frame_indices"][1] = (
            row["source_frame_indices"][1],
            row["source_frame_indices"][0],
        )
    elif fault == "source":
        row["sources"][0]["episode"] += 1
    elif fault == "input_sha":
        row["input_frames_sha256"] = "f" * 64
    elif fault == "latent_sha":
        row["latent_block_sha256"] = "f" * 64
    else:
        samples["cache_import_receipt_sha256"] = "f" * 64
    path = tmp_path / "samples.json"
    cache.write_json(path, samples)
    with pytest.raises(ValueError, match="不符"):
        cache.check_import(path, data["root"] / "receipt.json", data["raw"])


def test_g2_rejects_exec_pixels_even_with_updated_pkl_sha(imported, tmp_path):
    data, _, original = imported
    samples = copy.deepcopy(original)
    row = next(r for r in samples["windows"] if r["segment_kind"] == "exec")
    old = row["sources"][0]
    value = pickle.loads(Path(old["path"]).read_bytes())
    value["image"][0, 0, 0] ^= 1
    changed = tmp_path / "changed.pkl"
    changed.write_bytes(pickle.dumps(value))
    row["sources"][0] = {"kind": "exec_pkl", "path": str(changed), "sha256": cache.sha_file(changed)}
    path = tmp_path / "samples.json"
    cache.write_json(path, samples)
    with pytest.raises(ValueError, match="像素"):
        cache.check_import(path, data["root"] / "receipt.json", data["raw"])
