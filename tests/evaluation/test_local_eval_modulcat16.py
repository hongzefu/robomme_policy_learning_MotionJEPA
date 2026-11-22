"""local_eval.py 的 modulcat16 profile、--seeds、--dry-run 与单 seed 汇总；不启动任何进程、不碰 GPU。"""
import json

import pytest

from scripts.evaluation import local_eval
from scripts.evaluation import resident_worker

TASKS = ("BinFill", "PickXtimes", "VideoUnmask", "MoveCube")


@pytest.fixture(autouse=True)
def restore_profile():
    yield
    local_eval.select("timepool")


def test_modulcat16_profile_fields():
    p = local_eval.PROFILES["modulcat16"]
    assert str(p["ckpt"]) == ("/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/train-runs/"
                              "mme_vla_suite_orig16_b128_60k/orig100-m32x4x4-modul-motion-s2concat1x1024-s16-b128-60k/59999")
    assert p["model_kind"] == "stage2_pool4_concat" and p["prefix"] == "o16s16-ev"
    assert p["policy_config"] == "mme_vla_suite_orig16_b128_60k"
    assert p["history_yaml"] == "perceptual-framesamp-modul-32frame-4x4-motion-stage2-1x1024-s16-orig16.yaml"
    assert p["stage2_python"] == "/scratch/hongze/robomme_policy_learning_MotionJEPA/v1-store/venvs/wan-stage2/bin/python"
    assert p["motion"] == {"motion_stride": 16, "motion_budget": 192}
    assert list(p["gpus"]) == list(range(8)) and p["late_gpus"] == [] and p["wait_log"] is None


def test_port_ranges_disjoint():
    """各 profile 各 seed 的 32 个端口段互不重叠。"""
    used = {}
    for name, p in local_eval.PROFILES.items():
        for seed, base in p["port_base"].items():
            ports = {base + g * 10 + k for g in range(8) for k in range(local_eval.PER_GPU)}
            for other, taken in used.items():
                assert not ports & taken, (name, seed, other)
            used[(name, seed)] = ports


def test_spec_motion_only_for_modulcat16():
    local_eval.select("ctxcat")
    assert "motion_stride" not in local_eval.make_spec(42, "a" * 64, "c" * 40)
    local_eval.select("modulcat16")
    spec = local_eval.make_spec(42, "a" * 64, "c" * 40)
    assert (spec["motion_stride"], spec["motion_budget"], spec["model_seed"]) == (16, 192, 42)
    assert spec["stage2_python"].endswith("venvs/wan-stage2/bin/python")


def test_client_command_motion_args():
    base = {"sim_python": "sim", "repo": "/r", "run_identity_sha256": "a" * 64, "model_kind": "k", "model_seed": 42}
    grant = {"chunk_id": "c0"}
    import scripts.evaluation.resident_worker as rw
    original = rw.read_json
    rw.read_json = lambda path: grant
    try:
        old = resident_worker.client_command(base, "g.json", "out", "n", 1, "d")
        new = resident_worker.client_command({**base, "motion_stride": 16, "motion_budget": 192}, "g.json", "out", "n", 1, "d")
    finally:
        rw.read_json = original
    assert "--motion-stride" not in old
    assert new[len(old):] == ["--motion-stride", "16", "--motion-budget", "192"]


def test_dry_run_prints_plan(capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run 不得启动进程")
    monkeypatch.setattr(local_eval.subprocess, "run", forbidden)
    monkeypatch.setattr(local_eval.subprocess, "check_output", forbidden)
    assert local_eval.main(["--profile", "modulcat16", "--seeds", "42", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "seeds=(42,)" in out and "workers=32" in out
    assert "orig100-m32x4x4-modul-motion-s2concat1x1024-s16-b128-60k/59999" in out
    assert "ports=9900-9973" in out and "tmux_prefix=o16s16-ev" in out


def test_default_seeds_and_unknown_seed():
    assert local_eval.DEFAULT_SEEDS == (42, 7, 0)
    with pytest.raises(SystemExit):
        local_eval.main(["--profile", "modulcat16", "--seeds", "5", "--dry-run"])


def _write_accepted(out, seed, successes):
    root = out / f"seed{seed}/run800/accepted"
    root.mkdir(parents=True)
    for i in range(800):
        task = TASKS[i % len(TASKS)]
        record = {"episode_key": f"{task}:{i}", "task_name": task, "difficulty": ("easy", "hard")[i % 2],
                  "outcome": "success" if i < successes else ("fail", "timeout")[i % 2]}
        (root / f"{i}.json").write_text(json.dumps(record))


def test_single_seed_summary(tmp_path, monkeypatch, capsys):
    local_eval.select("modulcat16")
    monkeypatch.setattr(local_eval, "OUT", tmp_path)
    monkeypatch.setattr(local_eval.statistics, "stdev", lambda *a: (_ for _ in ()).throw(AssertionError("单 seed 不调 stdev")))
    _write_accepted(tmp_path, 42, 346)
    assert local_eval.summarize((42,)) == 0
    rows = json.loads((tmp_path / "seed42_summary.json").read_text())
    assert rows[0] == {"group": "overall", "name": "overall", "seed": 42, "episodes": 800, "success": 346,
                       "rate": 100 * 346 / 800}
    assert {r["name"] for r in rows if r["group"] == "task"} == set(TASKS)
    assert "SINGLE_SEED=PASS seed=42 overall=346/800=43.25" in capsys.readouterr().out
    assert (tmp_path / "seed42_summary.txt").exists()


def test_single_seed_summary_incomplete(tmp_path, monkeypatch):
    local_eval.select("modulcat16")
    monkeypatch.setattr(local_eval, "OUT", tmp_path)
    (tmp_path / "seed42/run800/accepted").mkdir(parents=True)
    assert local_eval.summarize((42,)) == 1


def test_multi_seed_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(local_eval, "OUT", tmp_path)
    for seed, n in ((42, 340), (7, 350), (0, 360)):
        _write_accepted(tmp_path, seed, n)
    assert local_eval.summarize((42, 7, 0)) == 0
    overall = json.loads((tmp_path / "three_seed_summary.json").read_text())[0]
    assert overall["mean"] == pytest.approx(100 * 350 / 800) and overall["std"] > 0
