"""官方清单的独立数量、顺序与真实元数据核对。"""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.evaluation.manifest import (BENCHMARK_COMMIT, TASKS, build_official_manifest,
                                          bind_model, budget_record, digest, make_manifest, seal, validate_episodes)
from scripts.evaluation.manifest import verify_model_binding


def test_actual_official_manifest():
    root = Path(__file__).resolve().parents[4]
    candidates = [Path(__file__).resolve().parents[2] / "third_party/robomme_benchmark",
                  Path("/data/hongzefu/robomme_policy_learning_MotionJEPA/third_party/robomme_benchmark")]
    benchmark = next((p for p in candidates if (p / "src").exists()), None)
    if benchmark is None:
        pytest.skip("需要已检出的锁定官方子模块")
    result = build_official_manifest(benchmark)
    assert len(result["episodes"]) == 800
    assert len(result["chunks"]) == 160
    assert [e["task_name"] for e in result["episodes"][:16]] == list(TASKS)
    assert result["episodes"][0] == {"task_name": "BinFill", "episode_id": 0,
                                        "episode_key": "BinFill:0", "env_seed": 540000, "difficulty": "easy"}
    assert len(result["metadata_sha256"]) == 16
    assert result["benchmark_commit"] == BENCHMARK_COMMIT
    for mutation in (lambda m: m["episodes"].pop(),
                     lambda m: m["episodes"].__setitem__(1, m["episodes"][0]),
                     lambda m: m["episodes"][0].__setitem__("task_name", "Unknown")):
        bad = copy.deepcopy(result)
        mutation(bad)
        with pytest.raises(ValueError):
            validate_episodes(seal(bad))


def test_hash_and_budget_boundaries():
    episode = {"task_name": "A", "episode_id": 0, "env_seed": 2, "difficulty": "easy"}
    manifest = make_manifest([episode], official=False)
    manifest["episodes"][0]["env_seed"] = 99
    with pytest.raises(ValueError):
        validate_episodes(manifest, official=False)
    assert budget_record(episode, 0)["passed"]
    assert budget_record(episode, 32)["k_demo"] == 1
    for d in range(17):
        assert budget_record(episode, d)["k_demo"] == 0
    assert budget_record(episode, 17)["k_demo"] == 1
    assert budget_record(episode, 48)["k_demo"] == 1
    assert budget_record(episode, 49)["k_demo"] == 2
    assert budget_record(episode, 1793)["k_demo"] == 56
    assert budget_record(episode, 1793)["passed"]
    assert not budget_record(episode, 3000)["passed"]


@pytest.mark.parametrize("sizes", [[800], [4, 6] + [5] * 158, [6, 4] + [5] * 158])
def test_resealed_official_manifest_refuses_nonfive_chunks(sizes):
    episodes = [{"task_name": task, "episode_id": i, "env_seed": i, "difficulty": "easy"}
                for i in range(50) for task in TASKS]
    manifest = make_manifest(episodes)
    keys = [e["episode_key"] for e in manifest["episodes"]]
    offset = 0
    manifest["chunks"] = []
    for size in sizes:
        members = keys[offset:offset + size]
        manifest["chunks"].append({"chunk_id": digest(members), "episode_keys": members})
        offset += size
    with pytest.raises(ValueError, match="160块"):
        validate_episodes(seal(manifest))


def test_model_binding_files_and_validation_identity(tmp_path, monkeypatch):
    # 只隔离B的证书边界，实体文件SHA与production/validation区分由C真实执行。
    monkeypatch.setitem(sys.modules, "scripts.evaluation.runtime_contract", SimpleNamespace(
        descriptor_identity=lambda kind, value: digest([kind, value]),
        validate_target_contract=lambda path, **kwargs: json.loads(Path(path).read_text())))
    episodes = [{"task_name": task, "episode_id": i, "env_seed": i, "difficulty": "easy"}
                for i in range(50) for task in TASKS]
    manifest = make_manifest(episodes)
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "resolved.json").write_text("{}")
    (root / "normstats.json").write_text("{}")
    (root / "motion_provenance.json").write_text("{}")
    (root / "params.bin").write_bytes(b"fixed weights")
    entries = [{"path": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in sorted(root.iterdir())]
    h = hashlib.sha256(b"{}").hexdigest()
    model = {"model_id": "stage2", "checkpoint_step": 60000, "checkpoint_root": str(root),
             "checkpoint_files": entries, "resolved_path": "resolved.json", "resolved_sha256": h,
             "normstats_path": "normstats.json", "normstats_sha256": h,
             "training_provenance_path": "motion_provenance.json", "motion_identity_sha256": "a" * 64,
             "eval_code_commit": "b" * 40, "reference_manifest_sha256": "c" * 64}
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"training_runtime": {}, "motion_identity_sha256": "a" * 64}))
    result = bind_model(manifest, {"models": [model]}, target)
    assert result["purpose"] == "production"
    assert verify_model_binding(result, target, expected_checkpoint_root=root, expected_purpose="production",
                                expected_run_identity_sha256=result["run_identity_sha256"]) == result
    with pytest.raises(ValueError, match="用途"):
        verify_model_binding(result, target, expected_purpose="validation")
    with pytest.raises(ValueError, match="20000"):
        bind_model(manifest, {"models": [model]}, target, validation=True)
    model["checkpoint_step"] = 20000
    validation = bind_model(manifest, {"models": [model]}, target, validation=True)
    assert validation["purpose"] == "validation"
    assert validation["run_identity_sha256"] != result["run_identity_sha256"]
    assert "validation_model_identity_sha256" in validation
    assert verify_model_binding(validation, target, expected_purpose="validation") == validation
    changed = copy.deepcopy(validation)
    changed["run_descriptor"]["model_seed"] = 99
    with pytest.raises(ValueError, match="重算"):
        verify_model_binding(seal(changed), target)
    (root / "params.bin").write_bytes(b"wrong weights")
    with pytest.raises(ValueError, match="SHA"):
        bind_model(manifest, {"models": [model]}, target, validation=True)


def test_model_binding_refuses_extra_file_and_symlink(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "scripts.evaluation.runtime_contract", SimpleNamespace(
        descriptor_identity=lambda kind, value: digest([kind, value]), validate_target_contract=lambda *a, **kw: {}))
    manifest = make_manifest([{"task_name": task, "episode_id": i, "env_seed": i, "difficulty": "easy"}
                              for i in range(50) for task in TASKS])
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "params").write_bytes(b"x")
    (root / "extra").write_bytes(b"y")
    model = {"model_id": "stage2", "checkpoint_step": 60000, "checkpoint_root": str(root),
             "eval_code_commit": "b" * 40, "checkpoint_files": [{"path": "params", "bytes": 1,
             "sha256": hashlib.sha256(b"x").hexdigest()}]}
    with pytest.raises(ValueError, match="集合"):
        bind_model(manifest, {"models": [model]}, tmp_path / "absent")
    (root / "params").unlink()
    (root / "params").symlink_to(root / "extra")
    with pytest.raises(ValueError, match="符号链接"):
        bind_model(manifest, {"models": [model]}, tmp_path / "absent")
