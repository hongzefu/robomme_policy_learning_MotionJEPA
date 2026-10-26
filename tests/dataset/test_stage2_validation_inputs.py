"""G0 的 CPU 合成输入：真实 H5／pkl 字节、历史 metadata 及失败关闭。"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import pickle
import subprocess
import sys
from collections import Counter
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from mme_vla_suite.datastore.manifest import manifest_sha256


SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts/dataset/check_stage2_validation_inputs.py"
SPEC = importlib.util.spec_from_file_location("stage2_validation_inputs", SCRIPT)
g0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g0)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def pixels(frame_ids):
    values = np.asarray(frame_ids, dtype=np.uint8)
    return np.broadcast_to(values[:, None, None, None], (len(frame_ids), 256, 256, 3)).copy()


def make_dataset(root, demo_lengths_by_task=((49,),), *, exec_length=65, write_frames=True):
    """输出 episode 从 1 开始、源 episode 从 3 开始，故意与全域编号分离。"""
    latents, source, demo_root = root / "latents", root / "source", root / "raw/data-raw"
    latents.mkdir(parents=True)
    (source / "data").mkdir(parents=True)
    manifest = {"episodes": []}
    total = {"schema": 3, "stage": "wan", "grid_stride": 16, "window_frames": 33,
             "frame_size": 256, "demo_min_real_frames": 17, "exec_min_real_frames": 33,
             "demo_tail_pad": "repeat_last", "grid_origin": "segment_start",
             "window_direction": "forward", "truncation_policy": "none", "segments": {}}
    scope = {"status": "PASS", "profile": "synthetic", "entries": []}
    run = {"profile": "synthetic", "raw_sha256": {}}
    offset = 7
    for task_idx, demo_lengths in enumerate(demo_lengths_by_task):
        task = f"Task{task_idx:02d}"
        for episode, demo_length in enumerate(demo_lengths, 1):
            ep = {"global_episode_idx": len(manifest["episodes"]), "h5_file": f"record_dataset_{task}.h5",
                  "raw_ep_idx": episode, "num_timesteps": demo_length + exec_length,
                  "exec_start_idx": demo_length, "exec_sample_offset": offset}
            manifest["episodes"].append(ep)
            for kind, length, start, minimum in (("demo", demo_length, 0, 17),
                                                  ("exec", exec_length, demo_length, 33)):
                key = f"{task}_ep{episode}_{kind}"
                rows = []
                for m, local in enumerate(range(0, max(0, length - minimum + 1), 16)):
                    real = min(33, length - local)
                    frame_ids = list(range(start + local, start + local + real))
                    frame_ids += [frame_ids[-1]] * (33 - real)
                    rows.append({"m": m, "seg_offset": local, "start_global_frame": start + local,
                                 "real_frames": real, "pad_frames": 33 - real,
                                 "pad_source_frame": frame_ids[-1] if real < 33 else None,
                                 "input_shape": [33, 256, 256, 3], "input_dtype": "uint8",
                                 "input_frames_sha256": g0.ms.sha256_bytes(pixels(frame_ids).tobytes())})
                if not rows:
                    continue
                data = b"".join(bytes([m + ep["global_episode_idx"]]) * g0.CHUNK_BYTES for m in range(len(rows)))
                lm = {"schema": 3, "segment": key, "segment_kind": kind, "g": ep["global_episode_idx"],
                      "h5_file": ep["h5_file"], "raw_ep_idx": episode, "seg_start_global": start,
                      "seg_len": length, "num_grid": len(rows), "num_chunks": length - minimum + 1,
                      "min_real": minimum, "grid_stride": 16, "window_frames": 33,
                      "latent_shape": [9, 16, 32, 32], "latent_dtype": "float32",
                      "chunk_bytes": g0.CHUNK_BYTES, "bytes": len(data), "sha256": g0.ms.sha256_bytes(data),
                      "rows": rows}
                write_json(latents / f"{key}.metadata.json", lm)
                total["segments"][key] = copy.deepcopy(lm)
                (latents / f"{key}.bin").write_bytes(data)
            if write_frames and demo_length:
                path = demo_root / f"{task}_ep{episode}" / "video_demo.h5"
                path.parent.mkdir(parents=True)
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("frames", data=pixels(range(demo_length)), compression="gzip")
                identity = {"difficulty": "easy", "episode_group": f"episode_{episode + 2}",
                            "h5_sha256": "a" * 64, "hi": demo_length, "lo": 0,
                            "member": f"{task}_ep{episode + 2}_seed42.h5", "output_episode_index": episode,
                            "seed": 42, "suffix": "demo"}
                sha = g0.ms.sha256_file(path)
                write_json(path.with_suffix(".complete.json"), {"identity": identity, "sha256": sha})
                scope["entries"].append({"name": f"{task}_ep{episode}_demo", "task": task, **identity})
                run["raw_sha256"][str(path.relative_to(demo_root.parent))] = sha
            if write_frames:
                for index in range(exec_length):
                    frame = demo_length + index
                    sample = {"image": pixels([frame])[0], "epis_idx": np.array([ep["global_episode_idx"]], np.int32),
                              "step_idx": np.array([frame], np.int32), "exec_start_idx": np.array([demo_length], np.int32),
                              "is_demo": np.array([False])}
                    (source / "data" / f"{offset + index}.pkl").write_bytes(pickle.dumps(sample))
            offset += exec_length
    manifest["sha256"] = manifest_sha256(manifest)
    total["manifest_sha256"] = manifest["sha256"]
    manifest_path = root / "episode_manifest.json"
    write_json(manifest_path, manifest)
    write_json(latents / "metadata.json", total)
    write_json(demo_root.parent / "control/scope.json", scope)
    write_json(demo_root.parent / "control/run_manifest.json", run)
    return SimpleNamespace(manifest_path=manifest_path, manifest=manifest, latents=latents, source=source,
                           demo_root=demo_root, scope={x["name"]: x for x in scope["entries"]}, run=run)


@pytest.fixture
def sample(tmp_path):
    return make_dataset(tmp_path)


def selected(sample):
    return g0.select_windows(sample.manifest, sample.latents, tasks=1, per_task=4, min_demo_tail=1)


def materialize(sample, item):
    return g0.materialize_window(item, source=sample.source, demo_root=sample.demo_root,
                                 scope=sample.scope, run_manifest=sample.run, file_sha_cache={})


def cli_args(sample, out):
    return ["--manifest", str(sample.manifest_path), "--latents", str(sample.latents),
            "--source", str(sample.source), "--demo-root", str(sample.demo_root), "--out", str(out)]


def test_full_64_window_cli_and_reproducibility(tmp_path, capsys):
    sample = make_dataset(tmp_path / "inputs", [(49,)] * 16)
    out = tmp_path / "result/windows.json"
    g0.main(cli_args(sample, out))
    result = json.loads(out.read_text())
    assert "RAW_WINDOWS_READY=PASS windows=64 tasks=16 demo_tail=16 sha_mismatches=0" in capsys.readouterr().out
    assert result["schema"] == "stage2-validation-inputs-v1"
    assert result["gate"] == "G0" and result["status"] == "PASS" and result["seed"] == 0
    assert result["counts"] == {"windows": 64, "tasks": 16, "demo_tail": 16, "sha_mismatches": 0}
    assert Counter(w["task"] for w in result["windows"]) == {f"Task{i:02d}": 4 for i in range(16)}
    assert all(w["source_grid"] % 2 == 0 and len(w["frame_ids"]) == 33 for w in result["windows"])
    assert all(len(w["source_frame_indices"]) == 33 for w in result["windows"])
    assert len({(w["key"], w["source_grid"]) for w in result["windows"]}) == 64
    for window in result["windows"]:
        path = pathlib.Path(window["latent_path"])
        block = path.read_bytes()[window["source_grid"] * g0.CHUNK_BYTES:][:g0.CHUNK_BYTES]
        assert window["latent_block_sha256"] == g0.ms.sha256_bytes(block)
        assert len({s["path"] for s in window["sources"]}) == len(window["sources"])
    again = g0.build_inputs(sample.manifest_path, sample.latents, sample.source, sample.demo_root)
    assert result == again
    reversed_manifest = {**sample.manifest, "episodes": list(reversed(sample.manifest["episodes"]))}
    assert g0.select_windows(sample.manifest, sample.latents) == g0.select_windows(reversed_manifest, sample.latents)
    original = out.read_bytes()
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        g0.main(cli_args(sample, out))
    assert out.read_bytes() == original
    assert list(out.parent.iterdir()) == [out]


def test_demo_tail_sources_use_output_episode_and_repeat_last(sample):
    item = next(x for x in selected(sample) if x[1] == "demo" and x[4]["pad_frames"])
    frames, row = materialize(sample, item)
    assert row["episode"] == 1
    assert row["sources"][0]["identity"]["episode_group"] == "episode_3"
    assert pathlib.Path(row["sources"][0]["path"]).parent.name == "Task00_ep1"
    assert row["frame_ids"] == list(range(32, 49)) + [48] * 16
    assert row["source_frame_indices"] == [{"source": 0, "index": i} for i in row["frame_ids"]]
    np.testing.assert_array_equal(frames, pixels(row["frame_ids"]))


def test_exec_sample_offset_and_global_episode_mapping(sample):
    item = next(x for x in selected(sample) if x[1] == "exec" and x[3] == 2)
    frames, row = materialize(sample, item)
    assert row["frame_ids"] == list(range(81, 114))
    assert pathlib.Path(row["sources"][0]["path"]).name == "39.pkl"
    assert row["source_frame_indices"] == [{"source": i, "index": 0} for i in range(33)]
    np.testing.assert_array_equal(frames, pixels(range(81, 114)))


@pytest.mark.parametrize("field", ["epis_idx", "step_idx", "exec_start_idx", "image"])
@pytest.mark.parametrize("fault", ["missing", "value", "dtype", "shape"])
def test_exec_rejects_identity_and_image_faults(sample, field, fault):
    item = next(x for x in selected(sample) if x[1] == "exec" and x[3] == 0)
    path = sample.source / "data/7.pkl"
    data = pickle.loads(path.read_bytes())
    if fault == "missing":
        del data[field]
    elif fault == "value":
        data[field] = data[field] + 1
    elif fault == "dtype":
        data[field] = data[field].astype(np.float32)
    else:
        data[field] = data[field][None]
    path.write_bytes(pickle.dumps(data))
    with pytest.raises(ValueError, match="原帧 SHA" if field == "image" and fault == "value" else field):
        materialize(sample, item)


@pytest.mark.parametrize("fault", ["complete_identity", "scope_identity", "complete_sha", "run_sha", "missing_identity"])
def test_demo_control_and_complete_must_agree(sample, fault):
    item = next(x for x in selected(sample) if x[1] == "demo")
    complete = sample.demo_root / "Task00_ep1/video_demo.complete.json"
    record = json.loads(complete.read_text())
    if fault == "complete_identity":
        record["identity"]["output_episode_index"] = 3
    elif fault == "scope_identity":
        sample.scope[item[2]]["episode_group"] = "episode_1"
    elif fault == "complete_sha":
        record["sha256"] = "b" * 64
    elif fault == "run_sha":
        sample.run["raw_sha256"] = {}
    else:
        del record["identity"]["episode_group"]
        del sample.scope[item[2]]["episode_group"]
    write_json(complete, record)
    with pytest.raises(ValueError, match="不同源|摘要"):
        materialize(sample, item)


@pytest.mark.parametrize("fault", ["grid_stride", "source_grid", "pad_source", "sha", "bound_rows", "manifest"])
def test_rejects_changed_historical_metadata(sample, fault):
    path = sample.latents / "Task00_ep1_demo.metadata.json"
    lm = json.loads(path.read_text())
    total = json.loads((sample.latents / "metadata.json").read_text())
    if fault == "grid_stride":
        lm["grid_stride"] = 32
    elif fault == "source_grid":
        lm["rows"][2]["m"] = 1
    elif fault == "pad_source":
        lm["rows"][2]["pad_source_frame"] = 47
    elif fault == "sha":
        lm["rows"][2]["input_frames_sha256"] = "invalid"
    elif fault == "bound_rows":
        total["segments"][lm["segment"]]["rows"][2]["input_frames_sha256"] = "b" * 64
    else:
        total["manifest_sha256"] = "b" * 64
    if fault not in {"bound_rows", "manifest"}:
        total["segments"][lm["segment"]] = copy.deepcopy(lm)
    write_json(path, lm)
    write_json(sample.latents / "metadata.json", total)
    with pytest.raises(ValueError, match="metadata|原始帧合同|SHA256"):
        selected(sample)


def test_tasks_without_demo_still_allow_16_real_demo_tails(tmp_path):
    sample = make_dataset(tmp_path, [(49, 49)] * 8 + [(0,)] * 8, exec_length=129, write_frames=False)
    items = g0.select_windows(sample.manifest, sample.latents)
    assert len(items) == 64
    assert sum(x[1] == "demo" and x[4]["pad_frames"] > 0 for x in items) >= 16
    assert set(Counter(g0.ms.task_of_h5(x[0]["h5_file"]) for x in items).values()) == {4}
    assert len({g0.ms.task_of_h5(x[0]["h5_file"]) for x in items if x[1] == "exec"}) == 16


def test_insufficient_demo_tails_fail(sample):
    with pytest.raises(ValueError, match="demo 补帧尾窗不足"):
        g0.select_windows(sample.manifest, sample.latents, tasks=1, per_task=4, min_demo_tail=2)


@pytest.mark.parametrize("relative", ["raw/data-raw/Task00_ep1/video_demo.h5",
                                      "raw/data-raw/Task00_ep1/video_demo.complete.json", "source/data/7.pkl"])
def test_missing_sources_name_exact_file_and_never_download(sample, relative):
    missing = sample.manifest_path.parent / relative
    missing.unlink()
    kind = "exec" if relative.endswith("pkl") else "demo"
    item = next(x for x in selected(sample) if x[1] == kind and x[3] == 0)
    with pytest.raises(FileNotFoundError, match="不自动下载") as error:
        materialize(sample, item)
    assert str(missing) in str(error.value)


def test_truncated_pickle_names_exact_source(sample):
    item = next(x for x in selected(sample) if x[1] == "exec" and x[3] == 0)
    path = sample.source / "data/7.pkl"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="执行样本无法读取") as error:
        materialize(sample, item)
    assert str(path) in str(error.value)


def test_cli_missing_manifest_reports_failure_without_output(sample, tmp_path):
    sample.manifest_path.unlink()
    out = tmp_path / "failed/windows.json"
    result = subprocess.run([sys.executable, str(SCRIPT), *cli_args(sample, out)],
                            capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 1
    assert "RAW_WINDOWS_READY=FAIL" in result.stderr
    assert "不自动下载" in result.stderr and str(sample.manifest_path) in result.stderr
    assert "RAW_WINDOWS_READY=PASS" not in result.stdout
    assert not out.exists()


def test_failed_cli_does_not_publish_pass(sample, tmp_path, monkeypatch):
    original_select = g0.select_windows
    monkeypatch.setattr(g0, "select_windows", lambda m, p: original_select(m, p, tasks=1, min_demo_tail=1))
    path = sample.latents / "Task00_ep1_demo.bin"
    with path.open("r+b") as stream:
        stream.write(b"corrupted")
    out = tmp_path / "failed/windows.json"
    with pytest.raises(ValueError, match="latent 文件字节数或 SHA"):
        g0.main(cli_args(sample, out))
    assert not out.exists()
    assert not out.parent.exists()


def test_atomic_publication_does_not_overwrite_concurrent_writer(sample, tmp_path, monkeypatch):
    out = tmp_path / "windows.json"
    monkeypatch.setattr(g0, "build_inputs", lambda *args: {"counts": {"demo_tail": 16}})
    original_link = g0.os.link

    def concurrent_link(src, dst):
        out.write_text("先到的结果", encoding="utf-8")
        return original_link(src, dst)

    monkeypatch.setattr(g0.os, "link", concurrent_link)
    with pytest.raises(FileExistsError):
        g0.main(cli_args(sample, out))
    assert out.read_text() == "先到的结果"
    assert not list(tmp_path.glob(".windows.json.*"))
