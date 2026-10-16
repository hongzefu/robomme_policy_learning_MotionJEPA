"""train_es_bound.py 的 PASS / FAIL / headroom 边界与源文件映射单测（合成小清单，纯 CPU、零数据依赖）。"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from mme_vla_suite.datastore.manifest import manifest_sha256

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/training/tests"))
import train_es_bound as bound  # noqa: E402

SCRIPT = ROOT / "scripts/training/tests/train_es_bound.py"


def write_manifest(tmp_path, episodes, raw_dir=None):
    """episodes: [(task, raw_ep_idx, exec_start_idx, num_timesteps)]，按给定顺序编 global_episode_idx。"""
    rows = [{"global_episode_idx": g, "h5_file": f"record_dataset_{task}.h5", "raw_ep_idx": idx,
             "num_timesteps": nt, "exec_start_idx": es, "exec_samples": nt - es}
            for g, (task, idx, es, nt) in enumerate(episodes)]
    payload = {"version": 1, "raw_dir": str(raw_dir or tmp_path / "no-such-raw"), "episodes": rows}
    payload["sha256"] = manifest_sha256(payload)
    path = tmp_path / "episode_manifest.json"
    path.write_text(json.dumps(payload))
    return path


def write_map(raw_dir, task, members):
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"new_idx": i, "member": m} for i, m in enumerate(members)]
    (raw_dir / f"record_dataset_{task}_episode_map.json").write_text(json.dumps(rows))


def reference_windows(es, nt, demo_min_real=17):
    """计划第五节的等价写法，只作测试侧独立对照，不进被测脚本。"""
    demo = len(range(0, max(0, es - (demo_min_real - 1)), 16))
    execution = len(range(0, max(0, (nt - es) - 32), 16))
    return demo + execution


def final_line(out):
    return out.strip().splitlines()[-1]


def test_reference_worst_episode_is_123(tmp_path, capsys):
    """建库前最坏集：PickXtimes 输出索引 196 / 源 PickXtimes_ep175_seed18500.h5，demo 0、执行 1989 步 → 123 窗。"""
    raw = tmp_path / "raw"
    members = [f"PickXtimes/PickXtimes_ep{i}_seed{i}.h5" for i in range(197)]
    members[196] = "PickXtimes/PickXtimes_ep175_seed18500.h5"
    write_map(raw, "PickXtimes", members)
    write_map(raw, "BinFill", ["BinFill/BinFill_ep0_seed1.h5"])
    path = write_manifest(tmp_path, [("BinFill", 0, 678, 1356), ("PickXtimes", 196, 0, 1989)], raw_dir=raw)
    assert bound.main(["--manifest", str(path), "--budget", "160"]) == 0
    out = capsys.readouterr().out
    assert final_line(out) == "TRAIN_ES_BOUND=PASS episodes=2 tasks=2 max_windows=123 budget=160 headroom=37"
    assert ("TRAIN_ES_BOUND_WORST task=PickXtimes output_index=196 source=PickXtimes_ep175_seed18500.h5 "
            "g=1 demo_len=0 exec_len=1989 demo_windows=0 exec_windows=123 windows=123") in out
    assert "TRAIN_ES_BOUND_TASK task=PickXtimes episodes=1 max_windows=123 worst=196/PickXtimes_ep175_seed18500.h5" in out
    assert f"TRAIN_ES_BOUND_TASK task=BinFill episodes=1 max_windows={reference_windows(678, 1356)}" in out
    assert "TRAIN_ES_BOUND_NOTE" not in out


@pytest.mark.parametrize("budget,expect", [(124, "PASS"), (123, "PASS"), (122, "FAIL")])
def test_headroom_boundary(tmp_path, capsys, budget, expect):
    path = write_manifest(tmp_path, [("PickXtimes", 196, 0, 1989)])
    code = bound.main(["--manifest", str(path), "--budget", str(budget)])
    line = final_line(capsys.readouterr().out)
    assert code == (0 if expect == "PASS" else 1)
    assert line.startswith(f"TRAIN_ES_BOUND={expect} episodes=1 tasks=1 max_windows=123 budget={budget} "
                           f"headroom={budget - 123}")
    if expect == "FAIL":
        assert line.endswith("over_budget=1 worst=PickXtimes:196/unknown")


def test_fail_names_worst_and_counts_over_budget(tmp_path, capsys):
    """多集超限：worst 取窗数最大者，over_budget 计全部超限集；子进程非 0 退出。"""
    raw = tmp_path / "raw"
    write_map(raw, "MoveCube", ["MoveCube/a.h5", "MoveCube/b.h5", "MoveCube/c.h5"])
    eps = [("MoveCube", 0, 100, 400), ("MoveCube", 1, 200, 1000), ("MoveCube", 2, 50, 700)]
    path = write_manifest(tmp_path, eps, raw_dir=raw)
    windows = [reference_windows(es, nt) for _, _, es, nt in eps]
    budget = sorted(windows)[0]          # 最小那集恰好不超，另两集超
    result = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(path), "--budget", str(budget)],
                            capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    line = final_line(result.stdout)
    assert line == (f"TRAIN_ES_BOUND=FAIL episodes=3 tasks=1 max_windows={max(windows)} budget={budget} "
                    f"headroom={budget - max(windows)} over_budget=2 worst=MoveCube:1/b.h5")


def test_windows_match_reference_formula_over_grid(tmp_path, capsys):
    """逐集窗数（库函数）与计划第五节等价写法一致，覆盖 16 的倍数、demo<17、exec<33 等边界。"""
    eps = []
    for es in (0, 1, 16, 17, 32, 33, 48, 49, 66, 1247):
        for exec_len in (1, 32, 33, 34, 48, 49, 64, 1989):
            eps.append(("VideoPlaceOrder", len(eps), es, es + exec_len))
    path = write_manifest(tmp_path, eps)
    rows = bound.evaluate(json.loads(path.read_text()), bound.ms.LAYOUT_SPECS[bound.ms.LAYOUT])
    assert [r["windows"] for r in rows] == [reference_windows(es, nt) for _, _, es, nt in eps]
    assert all(r["demo_windows"] == len(range(0, max(0, es - 16), 16)) for r, (_, _, es, _) in zip(rows, eps))


def test_legacy_layout_needs_33_demo_frames(tmp_path, capsys):
    """历史布局 demo 也要 33 帧：es=17 时 demopad17 有 1 个 demo 窗，历史布局为 0。"""
    path = write_manifest(tmp_path, [("ButtonUnmask", 0, 17, 17 + 33)])
    bound.main(["--manifest", str(path), "--layout", "motion-768-grid16-demopad17-v1"])
    assert "max_windows=2 " in final_line(capsys.readouterr().out)
    bound.main(["--manifest", str(path), "--layout", "motion-768-grid16-v1"])
    assert "max_windows=1 " in final_line(capsys.readouterr().out)


def test_tie_breaks_to_first_canonical_episode(tmp_path, capsys):
    path = write_manifest(tmp_path, [("StopCube", 0, 10, 300), ("BinFill", 5, 10, 300)])
    bound.main(["--manifest", str(path)])
    assert "TRAIN_ES_BOUND_WORST task=StopCube output_index=0 " in capsys.readouterr().out


def test_tampered_manifest_rejected(tmp_path):
    path = write_manifest(tmp_path, [("PickXtimes", 0, 0, 1989)])
    payload = json.loads(path.read_text())
    payload["episodes"][0]["num_timesteps"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sha256"):
        bound.main(["--manifest", str(path)])


def test_empty_exec_segment_is_loud(tmp_path):
    path = write_manifest(tmp_path, [("PickXtimes", 0, 50, 50)])
    with pytest.raises(ValueError, match="执行段为空"):
        bound.main(["--manifest", str(path)])


def test_explicit_map_dir_overrides_raw_dir(tmp_path, capsys):
    alt = tmp_path / "alt"
    write_map(alt, "PickXtimes", ["x/PickXtimes_ep9_seed9.h5"])
    path = write_manifest(tmp_path, [("PickXtimes", 0, 0, 100)])
    bound.main(["--manifest", str(path), "--episode-map-dir", str(alt)])
    assert "source=PickXtimes_ep9_seed9.h5" in capsys.readouterr().out


# ── eval_es_bound.py 参数化与 SKIP（同属窗口预算闸门，放在这里一起收集）──────────────────

EVAL_SCRIPT = ROOT / "scripts/training/tests/eval_es_bound.py"


def run_check(*args):
    return subprocess.run([sys.executable, str(EVAL_SCRIPT), "check", *args], capture_output=True, text=True)


def test_eval_check_skips_without_sim_testset(tmp_path):
    result = run_check("--tasks", "PickXtimes", "StopCube", "--metadata-dir", str(tmp_path / "none"))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1].startswith("ES_BOUND=SKIP reason=no_sim_testset ")


def test_eval_check_ready_with_parameterized_count(tmp_path):
    for task in ("PickXtimes", "StopCube"):
        records = [{"task": task, "episode": i, "seed": i} for i in range(3)]
        (tmp_path / f"record_dataset_{task}_metadata.json").write_text(json.dumps({"env_id": task, "records": records}))
    ready = run_check("--tasks", "PickXtimes", "StopCube", "--episodes-per-task", "3", "--metadata-dir", str(tmp_path))
    assert ready.stdout.strip().splitlines()[-1].startswith("ES_BOUND_TESTSET=READY tasks=2 episodes_per_task=3 ")
    short = run_check("--tasks", "PickXtimes", "--episodes-per-task", "4", "--metadata-dir", str(tmp_path))
    assert short.returncode == 0 and "ES_BOUND=SKIP reason=no_sim_testset missing_tasks=PickXtimes need=4" in short.stdout


def test_eval_aggregate_parameterized_tasks(tmp_path):
    from types import SimpleNamespace
    import eval_es_bound as ev
    paths = []
    for task in ("PickXtimes", "StopCube"):
        report = dict(task=task, complete=True, episodes=[dict(episode=i, es=100, k_eval=86) for i in range(3)],
                      python="p", packages={}, env_runner_sha256="s", robomme_commit="c", budget=160, max_steps=1300)
        path = tmp_path / f"{task}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    args = SimpleNamespace(reports=paths, out=tmp_path / "agg.json", budget=160, max_steps=1300,
                           tasks=["PickXtimes", "StopCube"], episodes_per_task=3)
    ev.aggregate(args)
    out = json.loads(args.out.read_text())
    assert (out["tasks"], out["episodes"], out["passed"]) == (2, 6, True)
    args.out, args.tasks = tmp_path / "bad.json", ["PickXtimes", "StopCube", "BinFill"]
    with pytest.raises(ValueError, match="3 个任务"):
        ev.aggregate(args)
