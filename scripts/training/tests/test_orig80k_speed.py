"""四卡测速的真实计时路径与报告拒绝条件；不加载模型。"""

import datetime
import json
import sys
from types import SimpleNamespace

import check_orig80k_speed as speed
import pytest


def dump_rows(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def test_timer_keeps_real_save_order(tmp_path):
    timer = speed.HostTiming(tmp_path, 300)
    events = []
    for step in range(300):
        with timer.step(step, flush=lambda step=step: events.append((step, "sync"))):
            with timer.phase("train_dispatch"):
                pass
            with timer.phase("data_next"):
                pass
            if step == 299:
                with timer.phase("checkpoint"):
                    events.append((step, "real_save"))
    timer.close()
    timer.close()
    assert events == [(99, "sync"), (299, "sync"), (299, "real_save")]
    data = speed.rows(tmp_path / "step_timing.jsonl")
    assert len(data) == 300
    assert all(row["completed"] for row in data)
    assert data[-1]["sync"]["wall_end"] <= data[-1]["save_wall_start"] <= data[-1]["wall_end"]
    assert not (tmp_path / "step_trace").exists()


def test_failed_step_not_reported_as_success(tmp_path):
    timer = speed.HostTiming(tmp_path, 300)
    with pytest.raises(RuntimeError, match="训练失败"), timer.step(0, flush=lambda: None):
        raise RuntimeError("训练失败")
    timer.close()
    assert speed.rows(tmp_path / "step_timing.jsonl")[0]["completed"] is False


def test_wrong_step_and_intermediate_save_rejected(tmp_path):
    timer = speed.HostTiming(tmp_path, 300)
    with pytest.raises(ValueError, match="乱序"), timer.step(1):
        pass
    with pytest.raises(ValueError, match="末步299"), timer.step(0), timer.phase("checkpoint"):
        pass
    timer.close()


def fixture(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    base = 1_790_000_000.0
    timing = [{"step": step, "wall_start": base + step, "wall_end": base + step + 1,
               "host_step_s": 1.0, "phases_s": {}, "completed": True} for step in range(300)]
    timing[99]["sync"] = {"location": "warmup_end", "wall_end": base + 100, "seconds": .1}
    timing[299]["sync"] = {"location": "before_final_save", "wall_end": base + 300, "seconds": .1}
    timing[299].update(wall_end=base + 302, host_step_s=3., save_wall_start=base + 300)
    dump_rows(records / "step_timing.jsonl", timing)
    metadata = {"head": "a" * 40, "mode": "perf", "steps": 300, "success": True, "entry_calls": 1, "sampler_stopped": True,
                "profiler": False, "loader": {"batch_size": 64, "workers": 4}, "gpu_ids": [0, 1, 2, 3],
                "start_wall": base - 5, "end_wall": base + 302,
                "save_calls": [{"step": 299, "state_step": 300, "start_wall": base + 300,
                                "return_wall": base + 301, "checkpoint_root": "仅测试夹具"}]}
    speed.write_json(records / "speed_run.json", metadata)
    dump_rows(records / "metrics.jsonl", [{"step": step, "wall_time": base + step,
        **{key: {"dec": .5, "hex": .5.hex()} for key in speed.SCALARS}} for step in (0, 100, 200)])
    dump_rows(records / "host_samples.jsonl", [{"wall_time": base + .5 * index,
        "rss_bytes": {"1": 1024}, "shm_used": 200, "MemAvailable_bytes": 4096} for index in range(607)])
    gpu = records / "gpu.csv"
    with gpu.open("w") as stream:
        for index in range(607):
            stamp = datetime.datetime.fromtimestamp(base + .5 * index, datetime.UTC)
            for device in range(4):
                stream.write(stamp.strftime("%Y/%m/%d %H:%M:%S.%f") + f", {device}, {0 if index % 2 else 100}, 2000\n")
    return records, gpu


def test_report_uses_full_mean_zero_fraction_and_excludes_save(tmp_path):
    records, gpu = fixture(tmp_path)
    result = speed.summarize(records, gpu)
    assert result["steady_seconds"] == 200
    assert result["samples_per_second"] == 64
    assert result["save_and_finish_s"] == 2
    assert result["slow_steps"] == []
    assert result["gpu"]["0"]["zero_pct"] == pytest.approx(100 * 200 / 401)
    assert result["gpu"]["0"]["mean_pct"] == pytest.approx(100 * 201 / 401)


@pytest.mark.parametrize("fault", ["nan", "missing_step", "wrong_gpu", "missing_sync", "sampling_failure"])
def test_report_rejects_invalid_evidence(tmp_path, fault):
    records, gpu = fixture(tmp_path)
    if fault == "nan":
        metrics = speed.rows(records / "metrics.jsonl")
        metrics[0]["loss"] = {"dec": float("nan"), "hex": "nan"}
        dump_rows(records / "metrics.jsonl", metrics)
    elif fault == "missing_step":
        dump_rows(records / "step_timing.jsonl", speed.rows(records / "step_timing.jsonl")[:-1])
    elif fault == "wrong_gpu":
        gpu.write_text(gpu.read_text().replace(", 3,", ", 4,"))
    elif fault == "missing_sync":
        timing = speed.rows(records / "step_timing.jsonl")
        del timing[99]["sync"]
        dump_rows(records / "step_timing.jsonl", timing)
    else:
        metadata = json.loads((records / "speed_run.json").read_text())
        metadata["sampling_error"] = "模拟资源采样失败"
        (records / "speed_run.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="非有限|300步|GPU采样|同步边界|采样未正常"):
        speed.summarize(records, gpu)


def test_run_rejects_deterministic_flags_before_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(speed, "ROOT", tmp_path)
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
    record_path = tmp_path / "v1-store/perf-test"
    with pytest.raises(ValueError, match="XLA_FLAGS"):
        speed.run(SimpleNamespace(records=str(record_path), mode="perf", train_args=[]))
    assert not record_path.exists()


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("mode", ["perf", "smoke"])
def test_wrapper_forwards_real_entry_and_save_and_restores_state(tmp_path, monkeypatch, failed, mode):
    """替换昂贵训练本身，实际执行包装器的安装、转发、采样和异常恢复。"""
    repo = speed.ROOT
    monkeypatch.syspath_prepend(str(repo / "scripts/training"))
    from orig80k_contract import make_train_args

    monkeypatch.setattr(speed, "ROOT", tmp_path)
    monkeypatch.setenv("MMEVLA_EXPECTED_TRAIN_HEAD", "a" * 40)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.setenv("TRAIN_TIMING_STEPS", "0")
    monkeypatch.setattr(speed.subprocess, "check_output", lambda argv, **kw: "" if "status" in argv else "a" * 40)
    calls = []
    def original_save(*a):
        calls.append(("save", a))

    original_timer = object()

    def original_trace(*a):
        return None
    fake_jax = SimpleNamespace(profiler=SimpleNamespace(start_trace=original_trace))
    fake_timer = SimpleNamespace(StepTiming=original_timer)
    fake_checkpoint = SimpleNamespace(save_state=original_save)
    fake_loader = SimpleNamespace(create_data_loader=lambda *a, **kw: SimpleNamespace(
        _data_loader=SimpleNamespace(torch_loader=SimpleNamespace(
            batch_size=64, num_workers=4, prefetch_factor=2, persistent_workers=True))))
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "step_timing", fake_timer)
    monkeypatch.setitem(sys.modules, "openpi.training", SimpleNamespace(checkpoints=fake_checkpoint))
    monkeypatch.setitem(sys.modules, "mme_vla_suite.training", SimpleNamespace(dataloader=fake_loader))
    monkeypatch.setattr(speed, "host_sample", lambda pid: {"wall_time": speed.time.time(), "pid": pid})
    expected_steps = 300 if mode == "perf" else 20
    argv = make_train_args(mode, "fixture", tmp_path / "v1-store/lib", tmp_path / "v1-store/assets", repo=tmp_path)
    old_argv = list(sys.argv)

    def entry(path, run_name):
        calls.append(("entry", list(sys.argv)))
        assert run_name == "__main__"
        assert sys.argv[1:] == argv
        assert fake_timer.StepTiming is speed.HostTiming
        assert os_env("TRAIN_TIMING_STEPS") == str(expected_steps)
        with pytest.raises(RuntimeError, match="profiler"):
            fake_jax.profiler.start_trace("任何路径")
        fake_loader.create_data_loader()
        fake_checkpoint.save_state(SimpleNamespace(directory="fixture"), SimpleNamespace(step=expected_steps), object(), expected_steps - 1)
        if failed:
            raise RuntimeError("模拟真实入口失败")

    monkeypatch.setattr(speed.runpy, "run_path", entry)
    record_path = tmp_path / "v1-store/records"
    args = SimpleNamespace(records=str(record_path), mode=mode, train_args=["--", *argv])
    if failed:
        with pytest.raises(RuntimeError, match="真实入口失败"):
            speed.run(args)
    else:
        speed.run(args)
    assert [kind for kind, _ in calls] == ["entry", "save"]
    assert sys.argv == old_argv
    assert fake_timer.StepTiming is original_timer
    assert fake_checkpoint.save_state is original_save
    assert fake_jax.profiler.start_trace is original_trace
    assert os_env("TRAIN_TIMING_STEPS") == "0"
    record = json.loads((record_path / "speed_run.json").read_text())
    assert bool(record.get("success")) is not failed
    assert record["sampler_stopped"]
    assert len(record["save_calls"]) == 1


def test_smoke_wrapper_runs_twenty_steps_and_keeps_save(tmp_path):
    timer = speed.HostTiming(tmp_path, 20)
    events = []
    for step in range(20):
        with timer.step(step, flush=lambda: events.append("sync")):
            with timer.phase("train_dispatch"):
                pass
            if step == 19:
                with timer.phase("checkpoint"):
                    events.append("save")
    timer.close()
    assert events == ["sync", "save"]
    assert len(speed.rows(tmp_path / "step_timing.jsonl")) == 20


def test_smoke_record_cannot_produce_perf_report(tmp_path):
    records, gpu = fixture(tmp_path)
    metadata = json.loads((records / "speed_run.json").read_text())
    metadata.update(mode="smoke", steps=20)
    (records / "speed_run.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="20步包装验证"):
        speed.summarize(records, gpu)


def os_env(key):
    import os
    return os.environ.get(key)
