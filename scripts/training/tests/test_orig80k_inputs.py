"""独立输入量具的 CPU 小文件测试，不启动模型训练或读取生产数据。"""

import argparse
import copy
import json
import multiprocessing
from pathlib import Path
import pickle
import subprocess
import sys
import types
import uuid

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_orig80k_inputs as tool


def manifest(episodes=((0, 40), (35, 45))):
    rows = []
    offset = total = 0
    for index, (start, length) in enumerate(episodes):
        rows.append({"global_episode_idx": index, "h5_file": "record_dataset_Test.h5", "raw_ep_idx": index,
                     "num_timesteps": length, "exec_start_idx": start, "exec_samples": length - start,
                     "exec_sample_offset": offset, "total_sample_offset": total})
        offset += length - start
        total += length
    result = {"version": 1, "episodes": rows,
              "totals": {"episodes": len(rows), "exec_samples": offset, "timesteps": total}}
    result["sha256"] = tool.manifest_content_sha(result)
    return result


def test_sample_boundaries_include_every_episode_and_no_demo():
    result = tool.sample_plan(manifest())
    assert [row["step_idx"] for row in result if row["epis_idx"] == 0] == [0, 1, 30, 31, 32, 39]
    assert [row["step_idx"] for row in result if row["epis_idx"] == 1] == [35, 36, 44]
    assert [row["index"] for row in result if row["epis_idx"] == 1] == [40, 41, 49]


def test_full_pkl_identity_real_files(tmp_path):
    payload = manifest(((1, 4), (2, 4)))
    data = tmp_path / "data"
    data.mkdir()
    for episode in payload["episodes"]:
        for step in range(episode["exec_start_idx"], episode["num_timesteps"]):
            index = episode["exec_sample_offset"] + step - episode["exec_start_idx"]
            values = {"epis_idx": np.asarray([episode["global_episode_idx"]], dtype=np.int32),
                      "step_idx": np.asarray([step], dtype=np.int32),
                      "exec_start_idx": np.asarray([episode["exec_start_idx"]], dtype=np.int32)}
            (data / f"{index}.pkl").write_bytes(pickle.dumps(values))
    assert tool.full_identity(tmp_path, payload)["exec_samples"] == 5
    path = data / "4.pkl"
    values = pickle.loads(path.read_bytes())
    values["step_idx"][0] -= 1
    path.write_bytes(pickle.dumps(values))
    with pytest.raises(ValueError, match="全量 pkl 身份不符"):
        tool.full_identity(tmp_path, payload)


@pytest.mark.parametrize("change", ["dtype", "shape", "signed_zero", "value", "missing_none"])
def test_raw_comparison_never_casts_or_fills(change):
    a = {"state": np.asarray([0.0, 1.0], dtype=np.float32)}
    b = copy.deepcopy(a)
    if change == "dtype":
        b["state"] = b["state"].astype(np.float64)
    elif change == "shape":
        b["state"] = b["state"].reshape(1, 2)
    elif change == "signed_zero":
        b["state"][0] = -0.0
    elif change == "value":
        b["state"][1] += 1
    else:
        b["motion_emb"] = None
    assert tool.first_difference(tool.comparison_tree(a), tool.comparison_tree(b), "样本")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_equal_nonfinite_inputs_still_fail(value):
    tree = tool.describe_tree({"state": np.asarray([value], dtype=np.float32)})
    assert not tool.first_difference(tree, tree, "相同")
    with pytest.raises(ValueError, match="非有限"):
        tool.require_finite_tree(tree, "相同")


def test_only_none_recurrent_keys_are_excluded():
    a = {"state": np.zeros(1), "recur_image_emb": None}
    b = {"state": np.zeros(1)}
    assert tool.comparison_tree(a) == tool.comparison_tree(b)
    a["recur_image_emb"] = np.zeros(1)
    with pytest.raises(ValueError, match="并未关闭"):
        tool.comparison_tree(a)


class TinyDataset:
    """真实 torch worker 调用的确定性数据集，记录单进程模式的解码次数。"""

    def __init__(self, size):
        self.size = size
        self.seen = []
        self.calls = multiprocessing.get_context("spawn").Value("q", 0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        with self.calls.get_lock():
            self.calls.value += 1
        self.seen.append(index)
        return {"index": np.int64(index), "value": np.asarray([index, -index], dtype=np.float32)}


def loader(size, workers):
    generator = torch.Generator().manual_seed(42)
    options = {"multiprocessing_context": "spawn"} if workers else {}
    return torch.utils.data.DataLoader(TinyDataset(size), batch_size=4, shuffle=True, drop_last=True,
                                       generator=generator, num_workers=workers, persistent_workers=bool(workers), **options)


@pytest.mark.parametrize("workers", [0, 2])
def test_real_sampler_two_epochs_match_without_decoding_skipped_batches(workers):
    size = 47
    reference = loader(size, workers)
    probe_dataset = reference.dataset
    probe = tool.capture_epochs(reference, index_only=True)
    assert not probe_dataset.seen
    assert probe_dataset.calls.value == 0
    actual = loader(size, workers)
    dataset = actual.dataset
    selections = tool.batch_selections(size, batch_size=4, prefix=3)
    seen_batches = []

    def callback(epoch, position, batch):
        seen_batches.append((epoch, position, batch["index"].tolist()))

    content = tool.capture_epochs(actual, index_only=False, selections=selections, callback=callback)
    assert probe == content
    assert [epoch["base_seed"] for epoch in probe] == [epoch["base_seed"] for epoch in content]
    tool.validate_sampler_records(content, size, 4)
    assert [(epoch, position) for epoch, position, _ in seen_batches] == [
        (epoch, position) for epoch, selected in enumerate(selections) for position in selected]
    for epoch, position, indices in seen_batches:
        assert indices == probe[epoch]["indices"][position * 4:(position + 1) * 4]
    if workers == 0:
        assert len(dataset.seen) == sum(map(len, selections)) * 4
        assert len(dataset.seen) < 2 * size
    assert dataset.calls.value == sum(map(len, selections)) * 4
    assert dataset.calls.value < 2 * size
    assert len(content[0]["drop_last_tail"]) == 3


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _input_fields(required):
    return {key: None if key.startswith("motion") else np.asarray([1.0], dtype=np.float32) for key in required}


def _ready_record(out, root, side, head, shared):
    out.mkdir()
    batch_size = tool.BATCH_SIZE
    size = batch_size * (tool.PREFIX_BATCHES + 3) + 1
    logical_batches = size // batch_size
    payload = manifest(((0, size),))
    plan = tool.sample_plan(payload)
    tool.write_json(out / "episode_manifest.json", payload)
    tool.write_json(out / "sample_plan.json", plan)
    tool.write_json(out / "identity.json", {**payload["totals"], "identity_sha256": "f" * 64,
                                           "manifest_sha256": payload["sha256"]})
    raw = _input_fields(tool.REQUIRED_RAW)
    transformed = _input_fields(tool.REQUIRED_TRANSFORMED)
    rows = [{"identity": identity, "raw_all": tool.describe_tree(raw), "raw": tool.comparison_tree(raw),
             "transformed_all": tool.describe_tree(transformed), "transformed": tool.comparison_tree(transformed)}
            for identity in plan]
    _jsonl(out / "samples.jsonl", rows)
    indices = [list(range(size)), list(reversed(range(size)))]
    epochs = [{"epoch": epoch, "indices": order, "drop_last_tail": order[logical_batches * batch_size:],
               "logical_batches": logical_batches,
               "generator_before": ("a" if epoch == 0 else "b") * 64,
               "generator_after": ("b" if epoch == 0 else "c") * 64, "base_seed": 42}
              for epoch, order in enumerate(indices)]
    selections = tool.batch_selections(size)
    tool.write_json(out / "sampler.json", {"index_probe": epochs, "real_content": epochs, "selected": selections})
    _jsonl(out / "batches.jsonl", [{"epoch": epoch, "batch": position,
                                    "indices": indices[epoch][position * batch_size:(position + 1) * batch_size],
                                    "inputs_all": tool.describe_tree(transformed), "inputs": tool.comparison_tree(transformed)}
                                   for epoch, selected in enumerate(selections) for position in selected])
    module = {"path": str(root / "src/mme_vla_suite/training/dataloader.py"), "sha256": "1" * 64, "bytes": 10}
    modules = {"mme_vla_suite.training.dataloader": module}
    for phase in ("start", "end"):
        tool.write_json(out / f"provenance_{phase}.json", {"root": str(root), "head": head, "porcelain": "", "modules": modules})
    for phase in ("index", "content"):
        directory = out / "workers" / phase
        directory.mkdir(parents=True)
        for worker in range(tool.WORKERS):
            tool.write_json(directory / f"worker-{worker}.json", {"worker": worker, "prefix": str(root / ".venv"),
                                                                  "pid": worker + 100, "modules": modules})
    common = {key: {"path": str(shared / key), "sha256": "2" * 64, "bytes": 10}
              for key in ("manifest", "input_manifest", "source_stats", "source_provenance", "norm_stats", "tokenizer", "history")}
    common["history"]["path"] = str(root / "history.yaml")
    dataset_type = "dataset.RoboMMEDataset" if side == "upstream" else "framesamp_dataset.FrameSampDataset"
    metadata = {"schema": tool.SCHEMA, "status": "COMPLETE", "side": side, "root": str(root), "head": head,
                "prefix": str(root / ".venv"), "forbid_roots": [], "common_files": common,
                "implementations": {"dataset": "mme_vla_suite.training." + dataset_type},
                "python": "3.11", "python_build": "相同构建", "packages": {"torch": "相同版本"},
                "source": str(shared), "harness": {"sha256": "3" * 64},
                "contract": {"batch_size": batch_size, "workers": tool.WORKERS, "seed": 42, "epochs": 2,
                             "prefix_batches": tool.PREFIX_BATCHES,
                             "drop_last": True, "history": tool.HISTORY, "action_horizon": 20,
                             "endpoint": "collate_before_jax", "excluded_legacy_none_keys": list(tool.LEGACY_NONE_KEYS)}}
    tool.write_json(out / "meta.json", metadata)
    tool.write_record_manifest(out)


@pytest.fixture
def ready_records(tmp_path, monkeypatch):
    # 仅进程内缩小判定规模；生产 CLI 不接受这些覆盖。
    monkeypatch.setattr(tool, "BATCH_SIZE", 4)
    monkeypatch.setattr(tool, "WORKERS", 2)
    monkeypatch.setattr(tool, "PREFIX_BATCHES", 3)
    left, right = tmp_path / "a", tmp_path / "b"
    _ready_record(left, tmp_path / "upstream", "upstream", "a" * 40, tmp_path / "shared")
    _ready_record(right, tmp_path / "current", "current", "b" * 40, tmp_path / "shared")
    return argparse.Namespace(a_dir=left, b_dir=right, expect_head_a="a" * 40, expect_head_b="b" * 40)


def test_complete_judge_records(ready_records, capsys):
    tool.judge(ready_records)
    assert "INPUT_EQ=PASS" in capsys.readouterr().out


@pytest.mark.parametrize("change", ["missing_sample", "duplicate_sample", "missing_batch", "duplicate_batch",
                                   "batch_identity", "sampler", "asset", "head", "worker", "module", "projection",
                                   "dtype", "nan", "motion_schema", "missing_asset"])
def test_judge_rejects_incomplete_or_changed_evidence(ready_records, change):
    out = ready_records.b_dir
    if change in ("missing_sample", "duplicate_sample", "projection", "dtype", "nan", "motion_schema"):
        path = out / "samples.jsonl"
        rows = tool.read_jsonl(path)
        if change == "missing_sample":
            rows.pop()
        elif change == "duplicate_sample":
            rows.append(rows[0])
        elif change == "projection":
            del rows[0]["transformed"]["items"]["actions"]
        else:
            for key in ("transformed_all", "transformed"):
                if change == "dtype":
                    rows[0][key]["items"]["state"]["dtype"] = "float64"
                elif change == "nan":
                    rows[0][key]["items"]["state"]["finite"] = False
                else:
                    rows[0][key]["items"]["motion_emb"] = {"kind": "none"}
        _jsonl(path, rows)
    elif change in ("missing_batch", "duplicate_batch", "batch_identity"):
        path = out / "batches.jsonl"
        rows = tool.read_jsonl(path)
        if change == "missing_batch":
            rows.pop()
        elif change == "duplicate_batch":
            rows.append(rows[0])
        else:
            rows[0]["indices"][0] = 999
        _jsonl(path, rows)
    elif change == "sampler":
        value = tool.read_json(out / "sampler.json")
        value["real_content"][0]["indices"][0] = 99
        tool.write_json(out / "sampler.json", value)
    elif change == "worker":
        (out / "workers/content/worker-1.json").unlink()
    elif change == "module":
        value = tool.read_json(out / "provenance_end.json")
        value["modules"]["mme_vla_suite.training.dataloader"]["path"] = "/tmp/外部污染.py"
        tool.write_json(out / "provenance_end.json", value)
    else:
        value = tool.read_json(out / "meta.json")
        if change == "asset":
            value["common_files"]["norm_stats"]["sha256"] = "0" * 64
        elif change == "missing_asset":
            del value["common_files"]["tokenizer"]
        else:
            value["head"] = "c" * 40
        tool.write_json(out / "meta.json", value)
    tool.write_record_manifest(out)
    with pytest.raises(ValueError, match="缺失|记录|投影|不同|不符|不一致|非有限|污染|未消费|state"):
        tool.judge(ready_records)


def test_artifact_tamper_is_not_accepted(ready_records):
    path = ready_records.a_dir / "samples.jsonl"
    with path.open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="取证文件已变化"):
        tool.judge(ready_records)


@pytest.mark.parametrize("location", ["parent", "index", "content"])
def test_nested_upstream_module_rejected_without_forbid_root(ready_records, location):
    a_out, b_out = ready_records.a_dir, ready_records.b_dir
    a_meta, b_meta = tool.read_json(a_out / "meta.json"), tool.read_json(b_out / "meta.json")
    old_root = Path(a_meta["root"])
    nested_root = Path(b_meta["root"]) / "v1-store/worktrees/orig"
    a_meta.update(root=str(nested_root), prefix=str(nested_root / ".venv"))
    a_meta["common_files"]["history"]["path"] = str(nested_root / "history.yaml")
    tool.write_json(a_out / "meta.json", a_meta)
    for path in [a_out / "provenance_start.json", a_out / "provenance_end.json",
                 *sorted((a_out / "workers").glob("*/*.json"))]:
        value = tool.read_json(path)
        if "root" in value:
            value["root"] = str(nested_root)
        if "prefix" in value:
            value["prefix"] = str(nested_root / ".venv")
        for record in value["modules"].values():
            record["path"] = str(nested_root / Path(record["path"]).relative_to(old_root))
        tool.write_json(path, value)
    assert b_meta["forbid_roots"] == []
    paths = ([b_out / "provenance_start.json", b_out / "provenance_end.json"] if location == "parent"
             else [b_out / "workers" / location / "worker-0.json"])
    for path in paths:
        value = tool.read_json(path)
        value["modules"]["openpi.transforms"] = {"path": str(nested_root / "src/openpi/transforms.py"),
                                                    "sha256": "4" * 64, "bytes": 5}
        tool.write_json(path, value)
    tool.write_record_manifest(a_out)
    tool.write_record_manifest(b_out)
    with pytest.raises(ValueError, match="B 项目模块混入 A worktree"):
        tool.judge(ready_records)


def test_module_origin_guard_rejects_pollution(tmp_path, monkeypatch):
    module = types.ModuleType("mme_vla_suite.input_probe_test")
    module.__file__ = "/tmp/外部模块.py"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="来源污染"):
        tool.module_records(tmp_path)


def test_cli_judge_reports_failure_without_project_imports(tmp_path):
    result = subprocess.run([sys.executable, str(Path(tool.__file__)), "judge", "--a-dir", str(tmp_path / "a"),
                             "--b-dir", str(tmp_path / "b"), "--expect-head-a", "a" * 40,
                             "--expect-head-b", "b" * 40], capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 1
    assert "INPUT_EQ=FAIL" in result.stderr


def test_cli_judge_positive_with_full_contract(tmp_path):
    # 保留正式 b64/w4/100批契约，仅使用小型摘要文件，不伪称真实上游数据取证。
    left, right = tmp_path / "a", tmp_path / "b"
    _ready_record(left, tmp_path / "upstream", "upstream", "a" * 40, tmp_path / "shared")
    _ready_record(right, tmp_path / "current", "current", "b" * 40, tmp_path / "shared")
    result = subprocess.run([sys.executable, str(Path(tool.__file__)), "judge", "--a-dir", str(left),
                             "--b-dir", str(right), "--expect-head-a", "a" * 40, "--expect-head-b", "b" * 40],
                            capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "INPUT_EQ=PASS" in result.stdout
    assert "batches=104" in result.stdout


def test_cli_collect_fails_before_output_when_head_is_wrong(tmp_path):
    root = Path(tool.__file__).resolve().parents[3]
    out = root / "v1-store/bench" / f"test-input-cli-{uuid.uuid4().hex}"
    command = [sys.executable, str(Path(tool.__file__)), "collect", "--side", "current",
               "--expect-root", str(root), "--expect-head", "0" * 40, "--dataset-path", str(tmp_path),
               "--source", str(tmp_path), "--manifest", str(tmp_path / "manifest.json"),
               "--input-manifest", str(tmp_path / "input.json"), "--assets-dir", str(tmp_path), "--out", str(out)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 1
    assert "INPUT_COLLECT=FAIL" in result.stderr
    assert "clean HEAD" in result.stderr
    assert not out.exists()
