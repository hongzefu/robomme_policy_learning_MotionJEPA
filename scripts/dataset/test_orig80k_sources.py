"""来源 pin 与清单守卫的合成正负例，另实跑公开清单的 CLI。"""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_orig80k_sources as sources


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def reindex(manifest):
    exec_offset = total_offset = 0
    for index, episode in enumerate(manifest["episodes"]):
        episode.update(global_episode_idx=index, exec_sample_offset=exec_offset, total_sample_offset=total_offset)
        exec_offset += episode["exec_samples"]
        total_offset += episode["num_timesteps"]
    manifest["totals"] = {"episodes": len(manifest["episodes"]), "timesteps": total_offset,
                          "exec_samples": exec_offset}
    manifest["canonical_order"] = sorted({ep["h5_file"] for ep in manifest["episodes"]})
    manifest["sha256"] = sources.manifest_sha256(manifest)
    return manifest


def make_manifest(tasks, *, per_task=2):
    episodes = [{"h5_file": f"record_dataset_{task}.h5", "raw_ep_idx": raw_index,
                 "num_timesteps": 2, "exec_start_idx": 0, "exec_samples": 2, "shard_idx": 0}
                for task in tasks for raw_index in range(per_task)]
    return reindex({"version": 1, "raw_dir": "/tmp/测试原始目录", "episodes": episodes,
                    "num_shards": 1, "shard_load_timesteps": [len(episodes) * 2]})


def configure_mini(monkeypatch):
    # 只缩小进程内常量，生产 CLI 没有绕过固定 16×100 / 4×100 的选项。
    monkeypatch.setattr(sources, "FULL_TASKS", ("Alpha", "Beta"))
    monkeypatch.setattr(sources, "COUNTING_TASKS", ("Beta",))
    monkeypatch.setattr(sources, "EPISODES_PER_TASK", 2)
    monkeypatch.setattr(sources, "FULL_TOTALS", {"episodes": 4, "timesteps": 8, "exec_samples": 8})
    monkeypatch.setattr(sources, "COUNTING_TOTALS", {"episodes": 2, "timesteps": 4, "exec_samples": 4})


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    configure_mini(monkeypatch)
    raw = tmp_path / "raw"
    raw.mkdir()
    files = {}
    for task in sources.FULL_TASKS:
        path = raw / f"record_dataset_{task}.h5"
        with h5py.File(path, "w") as handle:
            handle["values"] = np.arange(4, dtype=np.int32)
        files[path.name] = {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    pin = {"repo_id": "Yinpei/robomme_data_h5", "raw_dir": str(raw), "files": files}
    reference = make_manifest(sources.FULL_TASKS)
    current = copy.deepcopy(reference)
    current.update(raw_dir=str(raw), num_shards=7, shard_load_timesteps=[8, 0, 0, 0, 0, 0, 0])
    for episode in current["episodes"]:
        episode["shard_idx"] = 6
    reindex(current)
    paths = [tmp_path / name for name in ("input.json", "reference-input.json", "manifest.json", "reference.json")]
    for path, value in zip(paths, ({"count": 2, "files": files}, pin, current, reference), strict=True):
        write_json(path, value)
    return paths, raw


def test_recording_and_shard_fields_may_differ(inputs):
    paths, _ = inputs
    result = sources.check_sources(*paths, sources.FULL_TASKS)
    assert result["files"] == 2
    assert result["episodes"] == 4
    assert result["manifest_sha256"] != result["reference_manifest_sha256"]


@pytest.mark.parametrize("mode", ["missing", "extra", "wrong_size", "wrong_sha", "wrong_count"])
def test_input_pin_rejects_changes(inputs, mode):
    paths, _ = inputs
    document = sources.read_json(paths[0])
    name = next(iter(document["files"]))
    if mode == "missing":
        del document["files"][name]
    elif mode == "extra":
        document["files"]["extra.h5"] = dict(document["files"][name])
    elif mode == "wrong_size":
        document["files"][name]["size"] += 1
    elif mode == "wrong_sha":
        document["files"][name]["sha256"] = "0" * 64
    else:
        document["count"] = 9
    write_json(paths[0], document)
    with pytest.raises(ValueError, match="文件集合不符|来源 pin 不符|count 与文件集合不符"):
        sources.check_sources(*paths, sources.FULL_TASKS)


def test_same_size_h5_corruption_rejected_after_fresh_hash(inputs):
    paths, raw = inputs
    document = sources.read_json(paths[0])
    name = next(iter(document["files"]))
    path = raw / name
    with h5py.File(path, "r+") as handle:
        handle["values"][0] = 999
    assert path.stat().st_size == document["files"][name]["size"]
    document["files"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(paths[0], document)
    with pytest.raises(ValueError, match="来源 pin"):
        sources.check_sources(*paths, sources.FULL_TASKS)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "order", "offset", "global_index", "totals",
                                  "content", "identity", "stale_sha", "bool_version"])
def test_episode_guards(inputs, mode):
    paths, _ = inputs
    current = sources.read_json(paths[2])
    if mode == "missing":
        current["episodes"].pop()
        reindex(current)
    elif mode == "duplicate":
        current["episodes"][1]["raw_ep_idx"] = 0
    elif mode == "order":
        current["episodes"].reverse()
        reindex(current)
    elif mode == "offset":
        current["episodes"][1]["exec_sample_offset"] += 1
    elif mode == "global_index":
        current["episodes"][1]["global_episode_idx"] = 9
    elif mode == "totals":
        current["totals"]["timesteps"] = 99
    elif mode == "content":
        current["episodes"][0]["exec_start_idx"] = 1
    elif mode == "identity":
        current["episodes"][0]["raw_ep_idx"] = 99
    elif mode == "bool_version":
        current["version"] = True
    else:
        current["raw_dir"] = "摘要未更新"
    if mode != "stale_sha":
        current["sha256"] = sources.manifest_sha256(current)
    write_json(paths[2], current)
    with pytest.raises(ValueError, match="每任务|身份重复|顺序|不连续|totals|执行区间|物理身份集合|sha256|version"):
        sources.check_sources(*paths, sources.FULL_TASKS)


def test_counting_allows_explicit_reindexing(inputs):
    paths, _ = inputs
    current = sources.read_json(paths[2])
    current["episodes"] = [ep for ep in reversed(current["episodes"]) if ep["h5_file"] in sources.file_names(sources.COUNTING_TASKS)]
    write_json(paths[2], reindex(current))
    document = sources.read_json(paths[0])
    document["files"] = {key: value for key, value in document["files"].items() if key in sources.file_names(sources.COUNTING_TASKS)}
    document["count"] = 1
    write_json(paths[0], document)
    assert sources.check_sources(*paths, sources.COUNTING_TASKS)["episodes"] == 2


@pytest.mark.parametrize("value", ['{"files":{},"files":{}}', '{"value":NaN}'])
def test_ambiguous_json_rejected(tmp_path, value):
    path = tmp_path / "bad.json"
    path.write_text(value)
    with pytest.raises(ValueError, match="JSON"):
        sources.read_json(path)


def test_partial_or_duplicate_tasks_rejected(monkeypatch):
    configure_mini(monkeypatch)
    for value in ("", "Alpha", "Beta,Beta", "Alpha,Beta"):
        with pytest.raises(ValueError, match="--tasks 必须恰为"):
            sources.parse_tasks(value)


@pytest.mark.parametrize("counting", [False, True])
def test_real_cli_with_archived_public_manifests(tmp_path, counting):
    # 使用真实归档清单的全部 1600 / 400 条，仅复制小 JSON，不读取或生成 H5。
    reference = Path(__file__).resolve().parents[2] / "docs/dataset-build-doc/16task-h5-scan/records"
    manifest = sources.read_manifest(reference / "episode_manifest.json")
    document = sources.read_json(reference / "input_manifest.json")
    task_args = []
    if counting:
        names = sources.file_names(sources.COUNTING_TASKS)
        manifest["episodes"] = [ep for ep in manifest["episodes"] if ep["h5_file"] in names]
        document["files"] = {key: value for key, value in document["files"].items() if key in names}
        task_args = ["--tasks", ",".join(reversed(sources.COUNTING_TASKS))]
    manifest.update(num_shards=1, shard_load_timesteps=[sum(ep["num_timesteps"] for ep in manifest["episodes"])])
    for episode in manifest["episodes"]:
        episode["shard_idx"] = 0
    write_json(tmp_path / "manifest.json", reindex(manifest))
    write_json(tmp_path / "input.json", {"count": len(document["files"]), "files": document["files"]})
    command = [sys.executable, str(Path(sources.__file__)), "--input-manifest", str(tmp_path / "input.json"),
               "--reference-input", str(reference / "input_manifest.json"), "--manifest", str(tmp_path / "manifest.json"),
               "--reference-manifest", str(reference / "episode_manifest.json"), *task_args]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stderr
    assert f"INPUT_PIN=PASS files={4 if counting else 16}" in result.stdout
    assert f"episodes={400 if counting else 1600}" in result.stdout
