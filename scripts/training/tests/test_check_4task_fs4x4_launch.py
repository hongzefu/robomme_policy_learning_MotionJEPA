"""check_4task_fs4x4_launch.py（1127 计划 G3 启动合同）正例与反例。

全部用 --skip-dataset 与 tmp 目录里的假 store_meta；反例在 tmp 仓库副本（三份 yaml + runner）上改一处：
改 lr（训练配置 / runner 额外覆盖）、改 pool_type、budget 不整除、步数不是 50001、meta 不符，各自 FAIL。
运行：``JAX_PLATFORMS=cpu uv run --no-sync python -m pytest scripts/training/tests/test_check_4task_fs4x4_launch.py -q``
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import pathlib
import shutil
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

WT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "src"))
_spec = importlib.util.spec_from_file_location("check_4task_fs4x4_launch", WT / "scripts/training/check_4task_fs4x4_launch.py")
CL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CL)

need_ds = pytest.mark.skipif(not (CL.DS / "source").is_dir() or not (CL.DS / "meta/episode_manifest.json").is_file(),
                             reason="本机无 4task-v2-1600ep 库的 source/ 与 manifest")


@pytest.fixture(scope="module")
def real_cfg():
    return CL.get_train_config(CL.TRAIN_CONFIG)


@pytest.fixture
def store(tmp_path):
    def make(**over):
        meta = dict(layout=CL.LAYOUT, status="verified", num_rows=CL.NUM_ROWS, manifest_sha256=CL.MANIFEST_ID)
        meta.update(over)
        root = tmp_path / f"store-{len(list(tmp_path.glob('store-*')))}"
        (root / "meta").mkdir(parents=True)
        (root / "meta/store_meta.json").write_text(json.dumps(meta))
        return root
    return make


@pytest.fixture
def fake_repo(tmp_path):
    """tmp 仓库副本：三份 yaml + runner；返回 (repo, yaml_dir, runner)。"""
    repo = tmp_path / "repo"
    yd = repo / CL.YAML_REL
    yd.mkdir(parents=True)
    for n in (CL.BASE_8X8, CL.YAML_A, CL.YAML_B):
        shutil.copy(WT / CL.YAML_REL / n, yd / n)
    rd = repo / "scripts/training"
    rd.mkdir(parents=True)
    runner = rd / "run_4task_fs4x4_50k.sh"
    shutil.copy(WT / "scripts/training/run_4task_fs4x4_50k.sh", runner)
    return repo, yd, runner


def _run(argv, capsys, cfg):
    rc = CL.main([str(x) for x in argv], train_config_getter=lambda name: cfg)
    out = capsys.readouterr().out
    last = out.strip().splitlines()[-1]
    return rc, last


def _sub(p: pathlib.Path, old: str, new: str):
    t = p.read_text()
    assert t.count(old) == 1, (p, old)
    p.write_text(t.replace(old, new))


@need_ds
@pytest.mark.parametrize("budget,frames", [(512, 32), (1024, 64)])
def test_pass(budget, frames, store, capsys, real_cfg):
    rc, last = _run(["--budget", budget, "--store", store(), "--skip-dataset"], capsys, real_cfg)
    assert rc == 0 and last == f"LAUNCH_CONTRACT=PASS budget={budget} frames={frames} dataset=skipped", last


@need_ds
def test_pass_on_fake_repo_copy(fake_repo, store, capsys, real_cfg):
    repo, _, runner = fake_repo
    rc, last = _run(["--budget", 1024, "--store", store(), "--skip-dataset", "--repo", repo, "--runner", runner],
                    capsys, real_cfg)
    assert rc == 0, last


@need_ds
def test_fail_lr_in_train_config(store, capsys, real_cfg):
    bad = dataclasses.replace(real_cfg, lr_schedule=dataclasses.replace(real_cfg.lr_schedule, peak_lr=1e-4))
    rc, last = _run(["--budget", 512, "--store", store(), "--skip-dataset"], capsys, bad)
    assert rc == 1 and last.startswith("LAUNCH_CONTRACT=FAIL") and "train_config_fields" in last, last


@need_ds
def test_fail_lr_override_in_runner(fake_repo, store, capsys, real_cfg):
    repo, _, runner = fake_repo
    _sub(runner, '  "--num_train_steps=$STEPS"\n', '  "--num_train_steps=$STEPS"\n  --lr-schedule.peak-lr=1e-4\n')
    rc, last = _run(["--budget", 512, "--store", store(), "--skip-dataset", "--repo", repo, "--runner", runner],
                    capsys, real_cfg)
    assert rc == 1 and "train_args_mismatch" in last, last


@need_ds
def test_fail_pool_type(fake_repo, store, capsys, real_cfg):
    repo, yd, runner = fake_repo
    _sub(yd / CL.YAML_A, "pool_type: mean\n", "pool_type: max\n")
    rc, last = _run(["--budget", 512, "--store", store(), "--skip-dataset", "--repo", repo, "--runner", runner],
                    capsys, real_cfg)
    assert rc == 1 and "yaml_A_vs_8x8_diff" in last and "pool_type" in last, last


@need_ds
def test_fail_budget_not_divisible(fake_repo, store, capsys, real_cfg):
    repo, yd, runner = fake_repo
    _sub(yd / CL.YAML_B, "budget: 1024\n", "budget: 1000\n")
    rc, last = _run(["--budget", 1024, "--store", store(), "--skip-dataset", "--repo", repo, "--runner", runner],
                    capsys, real_cfg)
    assert rc == 1 and "budget_not_divisible" in last, last


@need_ds
def test_fail_steps_not_50001(fake_repo, store, capsys, real_cfg):
    repo, _, runner = fake_repo
    _sub(runner, "STEPS=50001\n", "STEPS=50000\n")
    rc, last = _run(["--budget", 512, "--store", store(), "--skip-dataset", "--repo", repo, "--runner", runner],
                    capsys, real_cfg)
    assert rc == 1 and "steps_not_50001" in last, last


@need_ds
def test_fail_yaml_sha_pin(fake_repo, store, capsys, real_cfg):
    repo, yd, runner = fake_repo
    _sub(yd / CL.YAML_A, "（1127 计划 A 组）", "（1127 计划 A 组，改注释）")   # 只改注释：diff 判据放行，sha 钉死拦下
    rc, last = _run(["--budget", 512, "--store", store(), "--skip-dataset", "--repo", repo, "--runner", runner],
                    capsys, real_cfg)
    assert rc == 1 and "yaml_sha_pin_mismatch" in last and "yaml_A_vs_8x8" not in last, last


@need_ds
@pytest.mark.parametrize("over", [dict(layout="framesamp-8x8-v1"), dict(status="packed"),
                                  dict(num_rows=1192917), dict(manifest_sha256="0" * 64)])
def test_fail_store_meta(over, store, capsys, real_cfg):
    rc, last = _run(["--budget", 512, "--store", store(**over), "--skip-dataset"], capsys, real_cfg)
    assert rc == 1 and "store_meta=" in last, last


def test_fail_missing_store(tmp_path, capsys, real_cfg):
    rc, last = _run(["--budget", 512, "--store", tmp_path / "none", "--skip-dataset"], capsys, real_cfg)
    assert rc == 1 and "store_meta_missing" in last, last
