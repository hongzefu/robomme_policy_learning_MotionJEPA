"""原版 80k 对拍量具的真实文件判定、负例、入口包装与历史 argv 回归。"""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parents[2]
sys.path.insert(0, str(TESTS))
import entry_equiv as eq

HEAD = "a" * 40
UPSTREAM = "b" * 40


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_record(directory, root, *, role="upstream", head=HEAD, gpu="GPU-0", start=0.0):
    directory.mkdir()
    scalar = {"dec": 1.0, "hex": 1.0.hex(), "finite": True}
    rows = [{"step": step, "wall_time": start + step,
             **{key: dict(scalar) for key in eq._SCALAR_KEYS}} for step in range(100)]
    write_rows(directory / "metrics.jsonl", rows)
    states = [{"step": step, "strict_global": "strict", "treedef_sha": "tree", "keyset_sha": "keys",
               "finite": True, "n_leaves": 3, "finite_leaves": 3, "nonfinite_keys": []}
              for step in (50, 99)]
    write_rows(directory / "state_digests.jsonl", states)
    entry = {"file": str(root / "train.py"), "sha256": "entry-sha"}
    provenance = {"expect_root": str(root), "git_root": str(root), "entry": entry,
                  "git_head_of_cwd": head, "git_porcelain_of_cwd": "", "modules": {}}
    write_json(directory / "provenance_start.json", provenance)
    provenance["modules"] = {name: {"file": str(root / (name.replace(".", "/") + ".py")), "sha256": name}
                             for name in ("mme_vla_suite.training.config", "openpi.training.checkpoints")}
    write_json(directory / "provenance.json", provenance)
    write_json(directory / "run_status.json", {"exit_code": 0, "entry_run_ok": True})
    write_json(directory / "harness_meta.json", {
        "expect_root": str(root), "entry": entry["file"], "forbid_roots": [], "harness_sha256": "harness-sha",
        "execution_role": role})
    first_gpu = int(gpu.removeprefix("GPU-"))
    fingerprint = {"version": 1, "python": {"version": "3.11"}, "dependencies": {"jax": "0.5.3"},
                   "uv_lock_sha256": "lock", "devices": [{"uuid": f"GPU-{number}"} for number in range(first_gpu, first_gpu + 4)],
                   "environment": {"XLA_FLAGS": "deterministic"}, "jax": {"enable_x64": False},
                   "history": {"path": str(root / "history.yaml"), "bytes": 12, "sha256": "history"},
                   "assets": {"norm_stats": "norm"}, "source": {"path": "/same/source", "records": "manifest"},
                   "dataset": {"path": "/same/framesamp"}}
    for name in ("runtime_fingerprint.json", "runtime_fingerprint_end.json"):
        write_json(directory / name, fingerprint)
    write_json(directory / "resolved_config.json", {"fields": {
        "exp_name": str(directory.name), "checkpoint_base_dir": str(directory),
        "dataset_path": "same-dataset", "overwrite": "False", "ema_decay": "0.999",
        "optimizer": "adamw", "lr_schedule": "original"}})
    return directory


def judge_args(a, b, **overrides):
    fields = dict(a_dir=str(a), b_dir=str(b), expect_steps=100, expect_tentative_a=0,
                  expect_state_steps="50,99", expect_head_a=HEAD, expect_head_b=HEAD,
                  expect_sha256=None, mode="upstream", concurrent_peer_dir=None)
    fields.update(overrides)
    return argparse.Namespace(**fields)


def same_entry_records(tmp_path):
    root = tmp_path / "repo"
    a = make_record(tmp_path / "solo", root, role="solo")
    b = make_record(tmp_path / "concurrent", root, role="concurrent", start=1000)
    peer = make_record(tmp_path / "peer", root, role="concurrent", gpu="GPU-4", start=1010)
    return a, b, peer


def mutate_json(path, fn):
    record = json.loads(path.read_text())
    fn(record)
    write_json(path, record)


def test_upstream_same_root_is_rejected(tmp_path, capsys):
    a, b, _ = same_entry_records(tmp_path)
    assert eq._judge(judge_args(a, b)) == 1
    assert "upstream 两侧 expect_root 相同" in capsys.readouterr().out


def test_upstream_different_roots_pass_with_identical_common_inputs(tmp_path):
    a = make_record(tmp_path / "a", tmp_path / "upstream", head=UPSTREAM)
    b = make_record(tmp_path / "b", tmp_path / "current")
    assert eq._judge(judge_args(a, b, expect_head_a=UPSTREAM)) == 0


def test_same_entry_passes_with_real_overlap(tmp_path, capsys):
    a, b, peer = same_entry_records(tmp_path)
    assert eq._judge(judge_args(a, b, mode="same-entry", concurrent_peer_dir=str(peer))) == 0
    assert "ENTRY_CONCURRENT=PASS" in capsys.readouterr().out


@pytest.mark.parametrize("change", ["head", "module", "data", "dependencies", "assets", "environment", "config",
                                  "start_dirty", "missing_start", "failed_exit", "missing_finite", "state_duplicate"])
def test_same_entry_evidence_changes_fail(tmp_path, change):
    a, b, peer = same_entry_records(tmp_path)
    if change == "head":
        mutate_json(b / "provenance.json", lambda p: p.update(git_head_of_cwd=UPSTREAM))
    elif change == "module":
        mutate_json(b / "provenance.json", lambda p: p["modules"]["openpi.training.checkpoints"].update(sha256="changed"))
    elif change in {"data", "dependencies", "assets", "environment"}:
        key = "dataset" if change == "data" else change
        for name in ("runtime_fingerprint.json", "runtime_fingerprint_end.json"):
            mutate_json(b / name, lambda p: p[key].update(changed="yes"))
    elif change == "config":
        mutate_json(b / "resolved_config.json", lambda p: p["fields"].update(dataset_path="wrong"))
    elif change == "start_dirty":
        mutate_json(b / "provenance_start.json", lambda p: p.update(git_porcelain_of_cwd="?? untracked\n"))
    elif change == "missing_start":
        (b / "provenance_start.json").unlink()
    elif change == "failed_exit":
        mutate_json(b / "run_status.json", lambda p: p.update(exit_code=1, entry_run_ok=False))
    elif change == "missing_finite":
        rows = eq._load_jsonl(b / "state_digests.jsonl")
        rows[0].pop("finite")
        write_rows(b / "state_digests.jsonl", rows)
    else:
        rows = eq._load_jsonl(b / "state_digests.jsonl")
        write_rows(b / "state_digests.jsonl", [*rows, rows[-1]])
    assert eq._judge(judge_args(a, b, mode="same-entry", concurrent_peer_dir=str(peer))) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_identical_nonfinite_scalars_are_rejected(tmp_path, value):
    a = make_record(tmp_path / "a", tmp_path / "upstream", head=UPSTREAM)
    b = make_record(tmp_path / "b", tmp_path / "current")
    for directory in (a, b):
        rows = eq._load_jsonl(directory / "metrics.jsonl")
        rows[12]["loss"] = {"dec": None, "hex": value.hex(), "finite": False}
        write_rows(directory / "metrics.jsonl", rows)
    assert eq._judge(judge_args(a, b, expect_head_a=UPSTREAM)) == 1


@pytest.mark.parametrize("fault", ["missing", "duplicate", "missing_key", "nonfinite_state", "forbidden_root"])
def test_missing_or_duplicate_training_evidence_fails(tmp_path, fault):
    a = make_record(tmp_path / "a", tmp_path / "upstream", head=UPSTREAM)
    b = make_record(tmp_path / "b", tmp_path / "current")
    for directory in (a, b):
        rows = eq._load_jsonl(directory / "metrics.jsonl")
        if fault == "missing":
            rows.pop(20)
        elif fault == "duplicate":
            rows.append(rows[-1])
        elif fault == "missing_key":
            rows[20].pop("loss")
        elif fault == "nonfinite_state":
            states = eq._load_jsonl(directory / "state_digests.jsonl")
            states[0].update(finite=False, finite_leaves=2, nonfinite_keys=["params.p"])
            write_rows(directory / "state_digests.jsonl", states)
        else:
            mutate_json(directory / "harness_meta.json", lambda p: p.update(forbid_roots=[p["expect_root"]]))
        write_rows(directory / "metrics.jsonl", rows)
    assert eq._judge(judge_args(a, b, expect_head_a=UPSTREAM)) == 1


@pytest.mark.parametrize("fault", ["no_overlap", "too_short", "same_gpu", "same_record", "missing_peer", "wrong_role"])
def test_invalid_concurrency_fails(tmp_path, fault):
    a, b, peer = same_entry_records(tmp_path)
    if fault in {"no_overlap", "too_short"}:
        rows = eq._load_jsonl(peer / "metrics.jsonl")
        for row in rows:
            row["wall_time"] += 1000 if fault == "no_overlap" else 45
        write_rows(peer / "metrics.jsonl", rows)
    elif fault == "same_gpu":
        for name in ("runtime_fingerprint.json", "runtime_fingerprint_end.json"):
            mutate_json(peer / name, lambda p: p.update(devices=[{"uuid": "GPU-0"}]))
    elif fault == "same_record":
        peer = b
    elif fault == "wrong_role":
        mutate_json(a / "harness_meta.json", lambda p: p.update(execution_role="concurrent"))
    assert eq._judge(judge_args(a, b, mode="same-entry",
                               concurrent_peer_dir=None if fault == "missing_peer" else str(peer))) == 1


def test_cache_redirect_changes_only_cache_and_forwards_return(tmp_path):
    calls = []
    config = types.SimpleNamespace(update=lambda *a, **kw: calls.append((a, kw)) or "result")
    original, redirected = eq._install_jax_cache_redirect(config, tmp_path)
    assert config.update("jax_compilation_cache_dir", "/home/wrong") == "result"
    assert config.update("jax_enable_x64", False) == "result"
    assert calls == [(("jax_compilation_cache_dir", str(tmp_path)), {}), (("jax_enable_x64", False), {})]
    assert redirected == [{"requested": "/home/wrong", "actual": str(tmp_path)}]
    config.update = original


def test_fingerprint_reads_actual_common_source_and_norm_files(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    source = root / "v1-store/datasets/lib/source"
    packed = source.parent / "framesamp"
    stats = root / "v1-store/train-assets/mme_vla_suite/robomme/norm_stats.json"
    weights = root / "v1-store/models/openpi-assets/checkpoints/pi05_base/params"
    history = root / "src/mme_vla_suite/models/config/robomme/history.yaml"
    paths = {root / "uv.lock": "locked", stats: "norm", weights / "_METADATA": "weights",
             root / "v1-store/models/big_vision/paligemma_tokenizer.model": "tokenizer",
             source / "meta/stats.json": "{}", source.parent / "meta/episode_manifest.json": "{}",
             source.parent / "meta/input_manifest.json": "{}", history: "budget: 512"}
    for path, content in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (packed / "meta").mkdir(parents=True)
    write_json(packed / "meta/store_meta.json", {"source_dataset_root": str(source),
                                                "manifest_path": str(source.parent / "meta/episode_manifest.json")})
    monkeypatch.setenv("OPENPI_DATA_HOME", str(root / "v1-store/models"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("MMEVLA_FRAMESAMP_SOURCE", raising=False)
    monkeypatch.delenv("MMEVLA_FRAMESAMP_MANIFEST", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(
        stdout="".join(f"{n}, GPU-{n}, A100, 550\n" for n in range(4))))
    cfg = types.SimpleNamespace(dataset_path=str(source),
                                data=types.SimpleNamespace(assets=types.SimpleNamespace(
                                    assets_dir=str(stats.parent.parent), asset_id="robomme"), repo_id="robomme"),
                                weight_loader=types.SimpleNamespace(params_path=str(weights)),
                                model=types.SimpleNamespace(history_config="history.yaml"))
    first = eq._runtime_fingerprint(cfg, root)
    cfg.dataset_path = str(packed)
    second = eq._runtime_fingerprint(cfg, root)
    assert first["source"] == second["source"]
    assert first["assets"] == second["assets"]
    assert first["dataset"] != second["dataset"]
    assert first["assets"]["norm_stats"] == eq._file_record(stats)
    stats.write_text("changed norm")
    assert eq._runtime_fingerprint(cfg, root)["assets"] != second["assets"]


def destructive_initializer(checkpoint_dir, *, overwrite=False, resume=False):
    """模拟真实入口的删除路径，确保负例在调用删除之前被守卫挡住。"""
    path = Path(checkpoint_dir)
    if path.exists():
        if not overwrite:
            raise FileExistsError(path)
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / "created.txt").write_text("本次创建")
    return "manager"


def checkpoint_fixture(tmp_path, upstream=True):
    cfg = types.SimpleNamespace(checkpoint_base_dir=str(tmp_path / "runs"), name="cfg", exp_name="fresh")
    path = Path(cfg.checkpoint_base_dir) / cfg.name / cfg.exp_name
    records = tmp_path / "records"
    records.mkdir()
    return eq._CheckpointOwnership(records, upstream=upstream), cfg, path


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling", "parent_symlink"])
def test_first_upstream_init_never_deletes_existing_output(tmp_path, kind):
    guard, cfg, path = checkpoint_fixture(tmp_path)
    path.parent.mkdir(parents=True)
    target = tmp_path / "foreign"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("必须保留的既有数据")
    if kind == "directory":
        path.mkdir()
        (path / "keep.txt").write_bytes(marker.read_bytes())
    elif kind == "file":
        path.write_bytes(marker.read_bytes())
    elif kind == "symlink":
        path.symlink_to(target, target_is_directory=True)
    elif kind == "dangling":
        path.symlink_to(tmp_path / "absent", target_is_directory=True)
    else:
        path.parent.rmdir()
        path.parent.symlink_to(target, target_is_directory=True)
    with pytest.raises((ValueError, FileExistsError)):
        guard.initialize(destructive_initializer, cfg, path.resolve(), overwrite=True, resume=False)
    assert marker.read_text() == "必须保留的既有数据"
    if kind == "directory":
        assert (path / "keep.txt").read_text() == marker.read_text()
    elif kind == "file":
        assert path.read_text() == marker.read_text()
    elif kind in {"symlink", "dangling"}:
        assert path.is_symlink()
    assert not guard.events


def test_only_own_tentative_second_init_may_overwrite(tmp_path):
    guard, cfg, path = checkpoint_fixture(tmp_path)
    assert guard.initialize(destructive_initializer, cfg, path, overwrite=True) == "manager"
    (path / "tentative.txt").write_text("临时段")
    assert guard.initialize(destructive_initializer, cfg, path, overwrite=True) == "manager"
    assert not (path / "tentative.txt").exists()
    guard.check_complete()
    record = json.loads(guard.record_path.read_text())
    assert [event["effective_overwrite"] for event in record["events"]] == [False, True]
    with pytest.raises(ValueError, match="次数"):
        guard.initialize(destructive_initializer, cfg, path, overwrite=True)
    assert (path / "created.txt").read_text() == "本次创建"


@pytest.mark.parametrize("fault", ["changed_path", "changed_inode", "symlink", "no_overwrite"])
def test_second_init_rejects_lost_ownership(tmp_path, fault):
    guard, cfg, path = checkpoint_fixture(tmp_path)
    guard.initialize(destructive_initializer, cfg, path, overwrite=True)
    if fault == "changed_path":
        cfg.exp_name = "another"
        path = path.parent / cfg.exp_name
    elif fault in {"changed_inode", "symlink"}:
        archived = path.parent / "preserved"
        path.rename(archived)
        if fault == "changed_inode":
            path.mkdir()
            (path / "foreign.txt").write_text("他人的目录")
        else:
            path.symlink_to(archived, target_is_directory=True)
    with pytest.raises(ValueError):
        guard.initialize(destructive_initializer, cfg, path, overwrite=fault != "no_overwrite")
    if fault == "changed_inode":
        assert (path / "foreign.txt").read_text() == "他人的目录"
    assert len(guard.events) == 1


def test_b_side_never_accepts_overwrite_or_second_init(tmp_path):
    guard, cfg, path = checkpoint_fixture(tmp_path, upstream=False)
    with pytest.raises(ValueError, match="B 侧禁止"):
        guard.initialize(destructive_initializer, cfg, path, overwrite=True)
    guard.initialize(destructive_initializer, cfg, path, overwrite=False)
    guard.check_complete()
    with pytest.raises(ValueError, match="次数"):
        guard.initialize(destructive_initializer, cfg, path, overwrite=False)


def test_first_init_disables_destructive_flag_even_if_output_appears_late(tmp_path):
    guard, cfg, path = checkpoint_fixture(tmp_path)

    def late_creator(checkpoint_dir, **kwargs):
        assert kwargs["overwrite"] is False
        path.mkdir(parents=True)
        (path / "keep.txt").write_text("检查后创建的目录")
        return destructive_initializer(checkpoint_dir, **kwargs)

    with pytest.raises(FileExistsError):
        guard.initialize(late_creator, cfg, path, overwrite=True)
    assert (path / "keep.txt").read_text() == "检查后创建的目录"


def driver(command, **values):
    keys = ("WORKTREE RECORD_ROOT RECORD_A RECORD_B STEPS EXP_A EXP_B HISTORY_YAML BATCH FSDP GPUS SAVE_INTERVAL "
            "DATA_A DATA_B CHECKPOINT_A CHECKPOINT_B ASSETS_DIR ANCHOR_SHA256 HEAD_A HEAD_B MODE EXPECT_STATE_STEPS "
            "EXPECT_TENTATIVE_A EXECUTION_ROLE FRAMESAMP_SOURCE FRAMESAMP_MANIFEST JAX_CACHE_A MMEVLA_JAX_CACHE_DIR "
            "A_PYTHON CONCURRENT_PEER_DIR").split()
    env = {key: value for key, value in os.environ.items() if key not in keys}
    env.update(values)
    return subprocess.run(["bash", str(TESTS / "run_entry_equiv.sh"), command], env=env,
                          text=True, capture_output=True)


@pytest.mark.parametrize("side", ["a", "b"])
def test_default_training_argv_is_historical(side):
    store = REPO / "v1-store"
    expected = ["mme_vla_suite", "--exp-name", f"entry-eq-{side}", "--num-train-steps", "1000",
                "--log-interval", "1", "--save-interval", "100", "--batch-size", "8", "--num-workers", "4",
                "--seed", "42", "--fsdp-devices", "2", "--dataset-path",
                str(store / "datasets" / ("4task-gl" if side == "a" else "4task-gl-framesamp")),
                "--assets-base-dir", str(store / "train-assets"), "--checkpoint-base-dir", str(store / "entryeq" / f"ckpt-{side}"),
                "--weight-loader.params-path", str(store / "models/openpi-assets/checkpoints/pi05_base/params"),
                "--model.use-history", "--model.history-config", "perceptual-framesamp-context.yaml", "--no-wandb-enabled"]
    if side == "a":
        expected.append("--overwrite")
    result = driver(f"dry-run-{side}")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected


def test_driver_new_parameters_and_empty_anchor():
    values = dict(STEPS="100", BATCH="64", FSDP="4", GPUS="4,5,6,7", SAVE_INTERVAL="50",
                  HISTORY_YAML="perceptual-framesamp-modul.yaml", ASSETS_DIR="/scratch/stats", ANCHOR_SHA256="")
    result = driver("dry-run-b", **values)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    for flag, value in (("--num-train-steps", "100"), ("--batch-size", "64"), ("--fsdp-devices", "4"),
                        ("--save-interval", "50"), ("--data.assets.assets-dir", "/scratch/stats")):
        assert args[args.index(flag) + 1] == value
    result = driver("dry-judge", **values)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert "--expect-sha256" not in args
    assert args[args.index("--expect-state-steps") + 1] == "50,99"
    assert args[args.index("--expect-tentative-a") + 1] == "12"


@pytest.mark.parametrize("values", [{"ASSETS_DIR": "relative"}, {"STEPS": "0"}, {"STEPS": "2"}, {"SAVE_INTERVAL": "0"}])
def test_driver_rejects_invalid_parameters_or_p1_judge(values):
    result = driver("dry-judge", **values)
    assert result.returncode != 0


@pytest.mark.parametrize("nonfinite", [False, True])
def test_real_runpy_p1_keeps_duplicate_states_and_checks_finite(tmp_path, monkeypatch, nonfinite):
    """使用真实 runpy/JAX 叶摘要执行两段两步；只替身外部训练与环境取证。"""
    import openpi.training.checkpoints as checkpoint

    root = tmp_path / "repo"
    root.mkdir()
    entry = root / "scripts/train.py"
    entry.parent.mkdir()
    cfg = types.SimpleNamespace(checkpoint_base_dir=str(root / "runs"), name="cfg", exp_name="fresh")
    checkpoint_path = Path(cfg.checkpoint_base_dir) / cfg.name / cfg.exp_name
    literal = 'float("nan")' if nonfinite else "1.0"
    entry.write_text(
        "import wandb\nimport numpy as np\nfrom types import SimpleNamespace\n"
        "import openpi.training.checkpoints as checkpoint\n"
        "for segment in range(2):\n"
        f"    checkpoint.initialize_checkpoint_dir({str(checkpoint_path)!r}, overwrite=True, resume=False)\n"
        "    for step in range(2):\n"
        "        wandb.log({key: 1.0 for key in ('loss','grad_norm','llm_grad_norm','mem_enc_norm','param_norm')}, step=step)\n"
        f"    value = np.array([{literal}], dtype=np.float32)\n"
        "    state = SimpleNamespace(params={'p': value}, opt_state={'m': value}, step=2, ema_params={'p': value})\n"
        "    checkpoint.save_state(None, state, None, 1)\n")
    provenance = {"expect_root": str(root), "git_root": str(root), "entry": eq._file_record(entry),
                  "git_head_of_cwd": HEAD, "git_porcelain_of_cwd": "", "modules": {}}
    provenance["entry"]["file"] = str(entry)
    monkeypatch.setattr(eq, "_module_provenance", lambda *args: copy.deepcopy(provenance))
    monkeypatch.setattr(eq, "_assert_provenance", lambda *args: None)
    monkeypatch.setattr(eq, "_runtime_fingerprint", lambda *args: {"fixed": True})
    monkeypatch.setattr(eq, "_dump_resolved_config", lambda directory: write_json(directory / "resolved_config.json", {}))
    original_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: types.SimpleNamespace(origin=str(root / "module.py"))
                        if name in eq._PROJECT_TOPLEVELS else original_spec(name))
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))
    monkeypatch.setattr(checkpoint, "initialize_checkpoint_dir", destructive_initializer)
    dummy_wandb = types.ModuleType("wandb")
    dummy_wandb.log = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "wandb", dummy_wandb)
    dummy_config = types.ModuleType("mme_vla_suite.training.config")
    dummy_config.cli = lambda: cfg
    monkeypatch.setitem(sys.modules, "mme_vla_suite.training.config", dummy_config)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("PYTHONHOME", raising=False)
    record = tmp_path / "record"
    args = argparse.Namespace(entry=str(entry), record_dir=str(record), expect_root=str(root), forbid_root=[],
                              expect_steps=2, expect_tentative=2, expect_head=HEAD, execution_role="upstream",
                              jax_cache_dir=None)
    assert eq._run(args, []) == int(nonfinite)
    assert [row["step"] for row in eq._load_jsonl(record / "state_digests.jsonl")] == [1, 1]
    assert len(eq._load_jsonl(record / "metrics_tentative.jsonl")) == 2
    assert len(eq._load_jsonl(record / "metrics.jsonl")) == 2
