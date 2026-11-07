"""Wan零参数池化起跑合同：真实 export→pack→verify 小库、真实CLI解析、实际shell边界；全程CPU。"""
import dataclasses
import difflib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests/dataset"))
sys.path.insert(0, str(REPO / "scripts/dataset"))
spec = importlib.util.spec_from_file_location("o16_tp_launch", REPO / "scripts/training/check_orig16_timepool_launch.py")
launch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launch
spec.loader.exec_module(launch)

YAML_DIR = REPO / "src/mme_vla_suite/models/config/robomme"
NEW_YAML, OLD_YAML = launch.HISTORY_NAME, launch.OLD_HISTORY
LINEAGE = "DATA_LINEAGE=PASS encoder_kind=wan_fixedpool1024 encoder_ckpt=none files=1 mismatches=0"
RUNNER = "run_orig16_timepool_60k.sh"
HEAD = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_fixture(root, *, verified=True):
    """全部读取都绑定临时根；不查主检出产物、真实模型或网络。"""
    from test_export_wan_timepool import build_tree, run as export
    from test_pack_motion_timepool import cli, pack_argv, verify_argv, write_records
    from mme_vla_suite.datastore import motion_store as ms
    parent = root / "v1-store/datasets/fixture"
    tree = build_tree(root / "build")
    tree.store = parent / launch.STORE_DIR
    tree.records = root / "records"
    export(tree)
    with pytest.MonkeyPatch.context() as patch:
        cli(patch, pack_argv(tree))
        if verified:
            write_records(tree, verdicts={"DATA_LINEAGE": LINEAGE})
            cli(patch, verify_argv(tree, records=True))
    (parent / "meta").mkdir(parents=True)
    shutil.copyfile(tree.manifest_path, parent / "meta/episode_manifest.json")
    shutil.copyfile(tree.input_path, parent / "meta/input_manifest.json")
    (parent / "source/data").mkdir(parents=True)
    dump(parent / "framesamp-4x4/meta/store_meta.json", dict(layout="framesamp-4x4-v1", status="verified",
        manifest_scope="full", manifest_sha256=tree.manifest["sha256"]))
    norm = root / "v1-store/train-assets/mme_vla_suite/fixture/robomme/norm_stats.json"
    dump(norm, dict(norm_stats={}))
    bounds = launch.legacy.manifest_bounds(tree.manifest)
    contract = launch.Contract(library="fixture", norm_sha256=launch.sha256_file(norm),
        episode_manifest_sha256=launch.sha256_file(parent / "meta/episode_manifest.json"),
        input_manifest_sha256=launch.sha256_file(parent / "meta/input_manifest.json"),
        canonical_manifest_sha256=tree.manifest["sha256"], motion_rows=bounds["rows"], train_max=bounds["train_max"],
        episodes=len(tree.manifest["episodes"]), tasks=1)
    return SimpleNamespace(root=root, parent=parent, contract=contract, store=tree.store, records=tree.records, ms=ms)


def fixture_history(contract):
    history = launch.legacy.load_yaml(REPO, NEW_YAML)
    history["motion"]["store_path"] = f"{contract.parent_rel}/{launch.STORE_DIR}"
    return history


def inputs(fx, *, steps=20, run=None, train_head=HEAD, repo=REPO):
    run = run or ("o16-smoke-timepool-fixture" if steps == 20 else launch.FORMAL_RUN)
    argv = launch.formal_argv(fx.root, run=run, steps=steps, library="fixture")
    env = dict(OPENPI_DATA_HOME=str(fx.root / "v1-store/models"), CUDA_VISIBLE_DEVICES=fx.contract.gpus,
        XLA_PYTHON_CLIENT_PREALLOCATE="false", MMEVLA_FRAMESAMP_SOURCE=str(fx.parent / "source"),
        MMEVLA_FRAMESAMP_MANIFEST=str(fx.parent / "meta/episode_manifest.json"),
        TRAIN_FINAL_RECORD_DIR=str(fx.root / "v1-store/validation/final-record"),
        TRAIN_RECORD_DIR=str(fx.root / "v1-store/validation/train-record"))
    args = SimpleNamespace(repo=repo, store_root=fx.root, steps=steps, train_head=train_head)
    return args, argv, env, fixture_history(fx.contract)


def contract_run(fx, args, argv, env, history):
    from mme_vla_suite.training import config
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "OPENPI_DATA_HOME", env["OPENPI_DATA_HOME"])
        return launch.core(args, argv, contract=fx.contract, env=env, history=history)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return make_fixture(tmp_path_factory.mktemp("o16-tp"))


# ── YAML ───────────────────────────────────────────────────────────────────


def test_yaml_text_differs_only_in_comment_and_identity_lines():
    old = (YAML_DIR / OLD_YAML).read_text().splitlines()
    new = (YAML_DIR / NEW_YAML).read_text().splitlines()
    removed = [line[2:].strip().split(":")[0] for line in difflib.ndiff(old, new) if line.startswith("- ")]
    added = [line[2:].strip().split(":")[0] for line in difflib.ndiff(old, new) if line.startswith("+ ")]
    assert sorted(k for k in removed if not k.startswith("#")) == sorted(
        ["readout", "store_path", "source_run", "checkpoint_sha256", "encoder_kind"])
    assert sorted(k for k in added if not k.startswith("#")) == sorted(["readout", "store_path", "encoder_kind"])
    assert new[0].startswith("#") and "零参数" in new[0]


def test_yaml_key_diff_is_exact():
    old, new = launch.legacy.load_yaml(REPO, OLD_YAML), launch.legacy.load_yaml(REPO, NEW_YAML)
    assert {k: v for k, v in old.items() if k != "motion"} == {k: v for k, v in new.items() if k != "motion"}
    changed = {k for k in old["motion"].keys() | new["motion"].keys() if old["motion"].get(k, "<x>") != new["motion"].get(k, "<x>")}
    assert changed == {"readout", "encoder_kind", "store_path", "source_run", "checkpoint_sha256"}
    assert "source_run" not in new["motion"] and "checkpoint_sha256" not in new["motion"]
    assert new["motion"]["online_gpu"] == 1
    assert new["motion"]["readout"] == "wan_raw_timepool8_spatial8x8_v1"
    assert new["motion"]["encoder_kind"] == "wan_fixedpool1024"
    assert new["motion"]["store_path"] == "v1-store/datasets/16task-orig100-1600ep-9e109419/motion-wan-timepool1024"


def test_yaml_resolves_to_timepool_policy_spec():
    from mme_vla_suite.datastore.motion_adapter import motion_policy_spec
    from mme_vla_suite.shared import wan_timepool as tp
    got = motion_policy_spec(launch.legacy.load_yaml(REPO, NEW_YAML)["motion"])
    assert (got.source_shape, got.readout, got.num_tokens, got.dim, got.budget, got.window_budget, got.layouts) == (
        (1, 1024), tp.TIMEPOOL_READOUT, 1, 1024, 96, 96, (tp.TIMEPOOL_LAYOUT,))


# ── TRAIN_CONFIG_EQ ────────────────────────────────────────────────────────


def test_train_config_eq_passes_with_exact_allowed_keys(tmp_path):
    result = launch.train_config_eq(launch.formal_argv(tmp_path))
    assert result["ok"], result
    assert set(result["diff_keys"]) == launch.TRAIN_CONFIG_ALLOWED
    assert result["verdict"].startswith("TRAIN_CONFIG_EQ=PASS diff_keys=")


@pytest.mark.parametrize("flag,value,key", [("--batch-size", "64", "batch_size"), ("--seed", "7", "seed"),
    ("--num-workers", "8", "num_workers"), ("--ema-decay", "0.99", "ema_decay"),
    ("--lr-schedule.peak-lr", "1e-4", "lr_schedule.peak_lr"), ("--save-interval", "1000", "save_interval")])
def test_train_config_eq_flags_real_cli_drift(tmp_path, flag, value, key):
    result = launch.train_config_eq(launch.formal_argv(tmp_path) + [flag, value])
    assert not result["ok"]
    assert key in result["unexpected"]


@pytest.mark.parametrize("mutate,key", [
    (lambda h: h["motion"].update(source_run=None), "history.motion.source_run"),
    (lambda h: h["motion"].update(checkpoint_sha256="a" * 64), "history.motion.checkpoint_sha256"),
    (lambda h: h["motion"].update(online_gpu=0), "history.motion.online_gpu"),
    (lambda h: h.update(budget=256), "history.budget"),
    (lambda h: h["motion"].pop("encoder_kind"), "history.motion.encoder_kind"),
    (lambda h: h["motion"].update(store_path="v1-store/x/motion-stage2-1024"), "history.motion.store_path")])
def test_train_config_eq_flags_yaml_drift(tmp_path, mutate, key):
    history = launch.legacy.load_yaml(REPO, NEW_YAML)
    mutate(history)
    result = launch.train_config_eq(launch.formal_argv(tmp_path), new_history=history)
    assert not result["ok"] and key in result["unexpected"]


def test_train_config_eq_missing_key_when_old_equals_new_identity(tmp_path):
    """差异键少了也判 FAIL：拿新 YAML 当旧侧时差分不满集合。"""
    new = launch.legacy.load_yaml(REPO, NEW_YAML)
    result = launch.train_config_eq(launch.formal_argv(tmp_path), new_history=new, old_history=new)
    assert not result["ok"] and "history.motion.readout" in result["missing"]


def test_old_concat_argv_is_canonical_not_copy(tmp_path):
    argv = launch.formal_argv(tmp_path, run="o16-smoke-timepool-a", steps=20) + ["--seed", "7"]
    old = launch.old_concat_argv(argv)
    assert "--seed" not in old and old[old.index("--exp-name") + 1] == launch.OLD_RUN
    assert old[old.index("--model.history-config") + 1] == OLD_YAML and "--num-train-steps" in old


def test_normalize_strips_memory_addresses_and_marks_absent():
    class Thing:
        def __repr__(self):
            return "<Thing at 0xABC123>"
    flat = launch.normalize_train_config(SimpleNamespace(), dict(a=Thing(), b=[1, 2]))
    assert flat["history.a"] == "<Thing>" and flat["history.b.__len__"] == 2
    assert launch.train_config_diff(dict(x="1"), dict(y="1")) == ["x", "y"]


# ── 合同核心 ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("steps", [20, 60000])
def test_contract_positive(fx, steps, capsys):
    rec = contract_run(fx, *inputs(fx, steps=steps))
    assert not rec.failed, rec.failed
    out = capsys.readouterr().out
    assert LINEAGE in out and "TRAIN_CONFIG_EQ=PASS" in out
    assert "MOTION_LAYOUT=PASS readout=timepool tokens_per_window=1 budget=96 total_tokens=608" in out


def faulty(fx, fault):
    args, argv, env, history = inputs(fx, steps=60000 if fault in ("old_run", "smoke_on_formal", "formal_old_weights") else 20)
    if fault == "old_history": argv[argv.index("--model.history-config") + 1] = OLD_YAML
    if fault == "old_run": argv[argv.index("--exp-name") + 1] = launch.OLD_RUN
    if fault == "smoke_on_formal": argv[argv.index("--exp-name") + 1] = "o16-smoke-timepool-x"
    if fault == "bad_smoke_name": argv[argv.index("--exp-name") + 1] = "o16-smoke-x"
    if fault == "resume": argv.append("--resume")
    if fault == "overwrite": argv.append("--overwrite")
    if fault == "resum_id": argv += ["--resum-ckpt-id", "59999"]
    if fault == "formal_old_weights":
        argv[argv.index("--weight-loader.params-path") + 1] = str(fx.root / "v1-store/train-runs/old/59999/params")
    if fault == "steps_mismatch": args.steps = 60000
    if fault == "gpu": env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    if fault == "allow_unverified": env["MMEVLA_MOTION_ALLOW_UNVERIFIED"] = "1"
    if fault == "motion_store_env": env["MMEVLA_MOTION_STORE"] = "/tmp/x"
    if fault == "learned_null": history["motion"]["source_run"] = None
    if fault == "learned_sha": history["motion"]["checkpoint_sha256"] = "a" * 64
    if fault == "stage2_kind": history["motion"]["encoder_kind"] = "stage2_pool4_concat"
    if fault == "stage2_readout": history["motion"]["readout"] = "stage2_pool4_concat_v1"
    if fault == "store_path": history["motion"]["store_path"] = f"{fx.contract.parent_rel}/motion-stage2-1024"
    if fault == "online_gpu": history["motion"]["online_gpu"] = 0
    if fault == "budget": history["motion"]["budget"] = 48
    if fault == "not_ancestor": args.train_head = "0" * 40
    return args, argv, env, history


@pytest.mark.parametrize("fault,check", [
    ("old_history", "HISTORY_NAME"), ("old_run", "CFG_RUN"), ("smoke_on_formal", "CFG_RUN"),
    ("bad_smoke_name", "CFG_RUN"), ("resume", "ARGV_NO_RESUME"), ("overwrite", "ARGV_NO_RESUME"),
    ("resum_id", "ARGV_NO_RESUME"), ("formal_old_weights", "CFG_EXACT"), ("steps_mismatch", "ARGV_STEPS"),
    ("gpu", "ENV_GPUS"), ("allow_unverified", "ENV_NO_ALLOW_OVERRIDES"), ("motion_store_env", "ENV_MOTION_STORE_UNSET"),
    ("learned_null", "HISTORY_KEYS"), ("learned_sha", "HISTORY_KEYS"), ("stage2_kind", "HISTORY_KEYS"),
    ("stage2_readout", "MOTION_READOUT"), ("store_path", "HISTORY_KEYS"), ("online_gpu", "HISTORY_KEYS"),
    ("budget", "MOTION_READOUT"), ("not_ancestor", "STORE_HEAD_ANCESTOR")])
def test_contract_negatives(fx, fault, check):
    assert check in contract_run(fx, *faulty(fx, fault)).failed


def test_packed_store_refused(tmp_path):
    packed = make_fixture(tmp_path, verified=False)
    (packed.store / packed.ms.LOCK_RELPATH).unlink()
    assert "STORE_VERIFIED" in contract_run(packed, *inputs(packed)).failed


def test_pack_lock_refused(tmp_path):
    fixture = make_fixture(tmp_path)
    (fixture.store / fixture.ms.LOCK_RELPATH).write_text("{}")
    rec = contract_run(fixture, *inputs(fixture))
    assert "DATASET" in rec.failed
    assert any("pack.lock" in detail for name, _, detail in rec.rows if name == "DATASET")


def rewrite_input(parent, mutate):
    path = parent / "meta/input_manifest.json"
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


def test_source_h5_set_is_exact():
    provenance = dict(source_h5=[dict(name="a.h5", bytes=1, sha256="x"), dict(name="b.h5", bytes=2, sha256="y")])
    same = dict(files={"a.h5": dict(size=1, sha256="x"), "b.h5": dict(size=2, sha256="y")})
    assert launch.source_h5_mismatches(provenance, same) == []
    assert launch.source_h5_mismatches(provenance, dict(files={"a.h5": dict(size=1, sha256="x")})) == ["b.h5"]
    assert launch.source_h5_mismatches(provenance, dict(files={**same["files"], "c.h5": dict(size=3, sha256="z")})) == ["c.h5"]
    assert launch.source_h5_mismatches(provenance, dict(files={**same["files"], "b.h5": dict(size=2, sha256="w")})) == ["b.h5"]


def test_input_manifest_h5_mismatch_cannot_be_waived(tmp_path):
    fixture = make_fixture(tmp_path)
    rewrite_input(fixture.parent, lambda v: next(iter(v["files"].values())).update(sha256="0" * 64))
    # 即使 pin 被改成新文件的 SHA，集合比较仍独立失败。
    fixture.contract = dataclasses.replace(fixture.contract,
        input_manifest_sha256=launch.sha256_file(fixture.parent / "meta/input_manifest.json"))
    assert "SOURCE_H5_SET" in contract_run(fixture, *inputs(fixture)).failed


@pytest.mark.parametrize("fault", ["tampered", "missing"])
def test_gate_record_read_back(tmp_path, fault):
    fixture = make_fixture(tmp_path)
    path = fixture.records / "POOL_TABLE_EQ.json"
    path.write_text("{}") if fault == "tampered" else path.unlink()
    assert "RECORD_POOL_TABLE_EQ" in contract_run(fixture, *inputs(fixture)).failed


# ── HEAD 规则：真实临时 git 仓库 ───────────────────────────────────────────


def git(cwd, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def history_repo(tmp_path):
    """base ─ docs_only ─ src_change ─ script_change；side 从 base 分叉。"""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "docs").mkdir()
    (root / "src/a.py").write_text("1\n")
    (root / "scripts/b.py").write_text("1\n")
    (root / "docs/c.md").write_text("1\n")
    git(root, "init", "-q", "-b", "main")
    commits = {}

    def commit(name, path=None):
        if path:
            (root / path).write_text(name + "\n")
        git(root, "add", "src", "scripts", "docs")
        git(root, "commit", "-q", "-m", name)
        commits[name] = git(root, "rev-parse", "HEAD")
    commit("base")
    commit("docs_only", "docs/c.md")
    git(root, "branch", "side", commits["base"])
    commit("src_change", "src/a.py")
    git(root, "switch", "-q", "side")
    commit("side_docs", "docs/c.md")
    git(root, "switch", "-q", "main")
    (root / "scripts/b.py").write_text("x\n")
    git(root, "add", "scripts/b.py")
    git(root, "commit", "-q", "-m", "script_change")
    commits["script_change"] = git(root, "rev-parse", "HEAD")
    return SimpleNamespace(root=root, c=commits)


@pytest.mark.parametrize("store,train,failed", [
    ("base", "base", []),
    ("base", "docs_only", []),
    ("docs_only", "src_change", ["STORE_HEAD_CODE_EQ"]),
    ("src_change", "script_change", ["STORE_HEAD_CODE_EQ"]),
    ("src_change", "base", ["STORE_HEAD_ANCESTOR", "STORE_HEAD_CODE_EQ"]),
    ("side_docs", "docs_only", ["STORE_HEAD_ANCESTOR"])])
def test_store_head_rule(history_repo, store, train, failed):
    rec = launch.Recorder()
    launch.check_store_head(rec, history_repo.root, history_repo.c[store], history_repo.c[train])
    assert rec.failed == failed


def test_store_head_unknown_commit_fails(history_repo):
    rec = launch.Recorder()
    launch.check_store_head(rec, history_repo.root, "f" * 40, history_repo.c["base"])
    assert rec.failed == ["STORE_HEAD_ANCESTOR", "STORE_HEAD_CODE_EQ"]


def test_core_uses_store_train_head_against_repo(fx, history_repo):
    """--train-head 在所给仓库里找不到建库 HEAD 时，合同核心必须失败，不靠别的仓库放行。"""
    args, argv, env, history = inputs(fx, train_head=history_repo.c["base"])
    args.repo = history_repo.root
    (history_repo.root / launch.s2.YAML_DIR).mkdir(parents=True)
    for name in (NEW_YAML, OLD_YAML):
        shutil.copyfile(YAML_DIR / name, history_repo.root / launch.s2.YAML_DIR / name)
    assert "STORE_HEAD_ANCESTOR" in contract_run(fx, args, argv, env, history).failed


@pytest.mark.parametrize("fault", ["head", "dirty"])
def test_repo_rule_requires_clean_and_exact_head(history_repo, monkeypatch, fault):
    monkeypatch.chdir(history_repo.root)
    head = history_repo.c["script_change"]
    if fault == "head":
        head = history_repo.c["base"]
    else:
        (history_repo.root / "docs/c.md").write_text("dirty\n")
    rec = launch.Recorder()
    launch.legacy.check_repo(rec, history_repo.root, head)
    assert ("REPO_HEAD" if fault == "head" else "REPO_CLEAN") in rec.failed


# ── CLI 分支 ───────────────────────────────────────────────────────────────


def final_fixture(run, steps):
    scalars = {"loss", "grad_norm", "llm_grad_norm", "param_norm", "mem_enc_norm"}
    return dict(run_name=run, head="a" * 40, num_train_steps=steps, state_step=steps, loop_step=steps - 1,
        checkpoint_step=steps - 1, log_interval=100,
        tail=[dict(step=i, scalars={n: dict(finite=True) for n in scalars}) for i in range((steps - 1) // 100 * 100 + 1, steps)],
        ema_leaves=dict(weight=dict(finite=True)))


@pytest.mark.parametrize("fault", [None, "missing", "ema", "run"])
def test_final_record_cli(tmp_path, capsys, fault):
    record = final_fixture(launch.FORMAL_RUN, 60000)
    if fault == "ema": record["ema_leaves"]["weight"]["finite"] = False
    if fault == "run": record["run_name"] = "other"
    path = tmp_path / "final.json"
    if fault != "missing": dump(path, record)
    code = launch.main(["--check-final-record", str(path), "--run", launch.FORMAL_RUN, "--steps", "60000", "--train-head", "a" * 40])
    assert code == int(fault is not None)
    assert f"TRAIN_FINAL_RECORD={'FAIL' if fault else 'PASS'}" in capsys.readouterr().out


def test_print_library(capsys):
    assert launch.main(["--print-library"]) == 0
    assert capsys.readouterr().out.strip() == "16task-orig100-1600ep-9e109419"


# ── 实际 shell 边界 ────────────────────────────────────────────────────────


def shell_tree(tmp_path):
    root = tmp_path / "repo"
    scripts = root / "scripts/training"
    scripts.mkdir(parents=True)
    for name in (RUNNER, "check_orig16_timepool_launch.py", "check_orig16_stage2_launch.py",
                 "check_mixed8_stage2_1024_launch.py", "final_record.py"):
        shutil.copyfile(REPO / "scripts/training" / name, scripts / name)
    return root


def fake_python(root, *, checker_exit=0):
    py = root / ".venv/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text(f'''#!/usr/bin/env bash
if [[ "$*" == *--print-library* ]]; then echo fixture; exit 0; fi
[[ "$1" == scripts/training/check_orig16_timepool_launch.py ]] || exit 91
[[ "$JAX_PLATFORMS" == cpu ]] || exit 92
echo "CONTRACT_INVOKED=1 $*"
exit {checker_exit}
''')
    py.chmod(0o755)


def shell_env(root, **extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", HISTORY=NEW_YAML, TRAIN_HEAD="a" * 40,
               FINAL_RECORD_DIR=str(root / "v1-store/validation/final-record"), DRY_RUN="1")
    env.pop("MAX_STEPS", None)
    env.update(extra)
    return {k: v for k, v in env.items() if v is not None}


def run_shell(root, env, *args):
    return subprocess.run(["bash", str(root / "scripts/training" / RUNNER), *args], env=env, capture_output=True, text=True)


def test_runner_bash_syntax():
    subprocess.run(["bash", "-n", str(REPO / "scripts/training" / RUNNER)], check=True)


@pytest.mark.parametrize("checker_exit", [0, 7])
@pytest.mark.parametrize("kind", ["smoke", "formal"])
def test_runner_dry_run_creates_nothing(tmp_path, checker_exit, kind):
    root = shell_tree(tmp_path)
    fake_python(root, checker_exit=checker_exit)
    extra = dict(RUN_NAME="o16-smoke-timepool-a", MAX_STEPS="20") if kind == "smoke" else dict(RUN_NAME=launch.FORMAL_RUN)
    before = {str(p.relative_to(root)) for p in root.rglob("*")}
    result = run_shell(root, shell_env(root, **extra))
    assert result.returncode == checker_exit, result.stdout + result.stderr
    assert "TRAIN_CMD=" in result.stdout and "CONTRACT_INVOKED=1" in result.stdout
    assert "--resume" not in result.stdout and "--overwrite" not in result.stdout
    assert ("--num-train-steps" in result.stdout) == (kind == "smoke")
    assert ("--steps 20" if kind == "smoke" else "--steps 60000") in result.stdout
    assert NEW_YAML in result.stdout and "framesamp-4x4" in result.stdout
    assert {str(p.relative_to(root)) for p in root.rglob("*")} == before


@pytest.mark.parametrize("extra,args", [
    (dict(HISTORY=OLD_YAML, RUN_NAME=launch.FORMAL_RUN), ()),
    (dict(HISTORY=f"x/{NEW_YAML}", RUN_NAME=launch.FORMAL_RUN), ()),
    (dict(RUN_NAME=launch.OLD_RUN), ()),
    (dict(RUN_NAME="o16-smoke-timepool-a"), ()),
    (dict(RUN_NAME="o16-smoke-timepool-a", MAX_STEPS="60000"), ()),
    (dict(RUN_NAME="o16-smoke-timepool-a", MAX_STEPS=""), ()),
    (dict(RUN_NAME="o16-smoke-a", MAX_STEPS="20"), ()),
    (dict(RUN_NAME="o16-smoke-timepool-", MAX_STEPS="20"), ()),
    (dict(RUN_NAME=launch.FORMAL_RUN, MAX_STEPS="20"), ()),
    (dict(RUN_NAME=launch.FORMAL_RUN), ("--resume",)),
    (dict(RUN_NAME=launch.FORMAL_RUN, TRAIN_HEAD="abc"), ()),
    (dict(RUN_NAME=launch.FORMAL_RUN, FINAL_RECORD_DIR="relative/final-record"), ()),
    (dict(RUN_NAME=launch.FORMAL_RUN, DRY_RUN="2"), ()),
    (dict(RUN_NAME=launch.FORMAL_RUN, CUDA_VISIBLE_DEVICES="0"), ())])
def test_runner_rejects_before_contract(tmp_path, extra, args):
    root = shell_tree(tmp_path)
    fake_python(root)
    result = run_shell(root, shell_env(root, **extra), *args)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "CONTRACT_INVOKED" not in result.stdout and "TRAIN_CMD" not in result.stdout


def test_runner_clears_override_environment(tmp_path):
    root = shell_tree(tmp_path)
    py = root / ".venv/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text('''#!/usr/bin/env bash
if [[ "$*" == *--print-library* ]]; then echo fixture; exit 0; fi
[[ -z "${MMEVLA_MOTION_STORE+x}${PYTHONPATH+x}${MMEVLA_MOTION_ALLOW_UNVERIFIED+x}" ]] || exit 95
exit 0
''')
    py.chmod(0o755)
    env = shell_env(root, RUN_NAME=launch.FORMAL_RUN, MMEVLA_MOTION_STORE="/x", PYTHONPATH="/x", MMEVLA_MOTION_ALLOW_UNVERIFIED="1")
    assert run_shell(root, env).returncode == 0


def test_runner_real_checker_fails_closed_on_empty_tree(tmp_path):
    """真实检查器走完整 CLI：空夹具树上判 FAIL，不写日志与缓存。"""
    root = shell_tree(tmp_path)
    py = root / ".venv/bin/python"
    py.parent.mkdir(parents=True)
    # 经包装转发才保留 venv 的包；入口会清 PYTHONPATH，这里在包装内补上被测源码。
    py.write_text(f'#!/usr/bin/env bash\nPYTHONPATH={REPO / "src"} exec {sys.executable} "$@"\n')
    py.chmod(0o755)
    before = {str(p.relative_to(root)) for p in root.rglob("*")}
    result = run_shell(root, shell_env(root, RUN_NAME=launch.FORMAL_RUN))
    assert result.returncode == 1, result.stdout + result.stderr
    last = [line for line in result.stdout.splitlines() if line.startswith("LAUNCH_CONTRACT=")]
    assert len(last) == 1 and last[0].startswith("LAUNCH_CONTRACT=FAIL")
    after = {str(p.relative_to(root)) for p in root.rglob("*") if "__pycache__" not in str(p)}
    assert after == {x for x in before if "__pycache__" not in x}


@pytest.mark.parametrize("fault", ["missing_final", "ema", "program_exit", "contract_exit", "nan", None])
def test_runner_records_diagnostics_and_keeps_exit(tmp_path, fault):
    root = shell_tree(tmp_path)
    (root / "v1-store/logs").mkdir(parents=True)
    record = final_fixture("o16-smoke-timepool-a", 20)
    if fault == "ema": record["ema_leaves"]["weight"]["finite"] = False
    if fault != "missing_final": dump(root / "fixture-final.json", record)
    py = root / ".venv/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text(f'''#!{sys.executable}
import os
from pathlib import Path
import shutil
import sys
if "--print-library" in sys.argv:
    print("fixture")
elif "--check-final-record" in sys.argv:
    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
elif sys.argv[1] == "scripts/training/train.py":
    source = Path("fixture-final.json")
    if source.exists():
        final = Path(os.environ["TRAIN_FINAL_RECORD_DIR"]) / "final.json"
        final.parent.mkdir(parents=True)
        shutil.copyfile(source, final)
    print("\\r\\rStep 0: loss={'nan' if fault == 'nan' else '0.2'}, grad_norm=0.1")
    sys.exit({9 if fault == "program_exit" else 0})
else:
    sys.exit({7 if fault == "contract_exit" else 0})
''')
    py.chmod(0o755)
    env = shell_env(root, RUN_NAME="o16-smoke-timepool-a", MAX_STEPS="20", DRY_RUN="0")
    result = run_shell(root, env)
    log = (root / "v1-store/logs/o16-smoke-timepool-a.log").read_text()
    want = {"program_exit": 9, "contract_exit": 7}.get(fault, 0)
    assert result.returncode == want, result.stdout + result.stderr + log
    assert log.splitlines()[-1] == f"EXIT_CODE={want}"
    final_fails = fault in ("missing_final", "ema", "contract_exit")
    assert f"TRAIN_FINAL_RECORD={'FAIL' if final_fails else 'PASS'}" in log
    assert ("TRAIN_DIAGNOSTIC=FAIL check=final_record action=record_only reason=user_decision" in log) == final_fails
    assert ("TRAIN_NUMERICS=FAIL reason=nonfinite_or_oom" in log) == (fault == "nan")
    # 无覆盖日志：同名再跑必须拒绝且不动旧日志。
    again = run_shell(root, env)
    assert again.returncode == 2 and (root / "v1-store/logs/o16-smoke-timepool-a.log").read_text() == log


def test_runner_matches_old_entry_outside_identity_lines():
    """与旧入口的差异只能落在固定身份、运行名校验和帧布局分派上。"""
    old = (REPO / "scripts/training/run_orig16_stage2_60k.sh").read_text().splitlines()
    new = (REPO / "scripts/training" / RUNNER).read_text().splitlines()
    diff = [line for line in difflib.ndiff(old, new) if line[:2] in ("- ", "+ ")]
    allowed = re.compile(r"HISTORY|RUN_FORMAL|CHECKER|check_orig16|MAX_STEPS|RUN.*formal|正式运行名|^. else$|FRAME|官方16×100|case \"\$HC\"|esac|\*\)")
    assert [line for line in diff if not allowed.search(line) and line[2:].strip() not in ("", "fi", "else")] == []
