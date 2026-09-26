"""四卡测速的真实计时路径与报告拒绝条件；不加载模型。"""

import datetime
import errno
import hashlib
import json
import os
import sys
import threading
import time
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
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint = checkpoint_root / "299"
    checkpoint.mkdir(parents=True)
    speed.write_json(checkpoint / "_CHECKPOINT_METADATA", {
        "init_timestamp_nsecs": 1_790_000_300_001_000_123,
        "commit_timestamp_nsecs": 1_790_000_301_900_000_789})
    (records / "final").mkdir()
    speed.write_json(records / "final/start.json", {
        "run_uuid": "本run", "head": "a" * 40, "checkpoint_dir": str(checkpoint_root)})
    speed.write_json(records / "final/checkpoint_wait_done.json", {
        "run_uuid": "本run", "head": "a" * 40, "wait_until_finished": True,
        "completed_at": datetime.datetime.fromtimestamp(base + 302, datetime.UTC).isoformat()})
    timing = [{"step": step, "wall_start": base + step, "wall_end": base + step + 1,
               "host_step_s": 1.0, "phases_s": {}, "completed": True} for step in range(300)]
    timing[99]["sync"] = {"location": "warmup_end", "wall_end": base + 100, "seconds": .1}
    timing[299]["sync"] = {"location": "before_final_save", "wall_end": base + 300, "seconds": .1}
    timing[299].update(wall_end=base + 302, host_step_s=3., save_wall_start=base + 300)
    dump_rows(records / "step_timing.jsonl", timing)
    metadata = {"head": "a" * 40, "mode": "perf", "steps": 300, "success": True, "entry_calls": 1, "sampler_stopped": True,
                "profiler": False, "loader": {"batch_size": 64, "workers": 4}, "gpu_ids": [0, 1, 2, 3],
                "start_wall": base - 5, "entry_start_wall": base - 4, "end_wall": base + 302,
                "save_calls": [{"step": 299, "state_step": 300, "start_wall": base + 300,
                                "return_wall": base + 301, "checkpoint_root": str(checkpoint_root)}]}
    disk = []
    for index in range(615):
        stamp = base - 4.5 + index * .5
        disk.append({"sequence": index, "kind": "start" if index == 0 else "final" if index == 614 else "periodic",
                     "phase": "save_pending" if stamp >= base + 301 else "training",
                     "wall_start": stamp, "wall_end": stamp + .01, "monotonic_start": stamp,
                     "scheduled_monotonic": stamp, "interval_s": .5 if index else None,
                     "lateness_s": 0., "duration_s": .01, "errors": [], "scratch_available_bytes": 40960 - index,
                     "checkpoint": {"exists": True, "allocated_bytes": 8192 if index == 610 else 4096,
                                    "unique_inodes": 1, "duplicate_inodes": 0, "disappeared": 0,
                                    "changed": 0, "symlinks": 0, "errors": []}})
    dump_rows(records / "disk_samples.jsonl", disk)
    metadata["disk_sampling"] = {"schema": 1, "interval_s": .5, "stopped": True, "error": None,
                                 "checkpoint_root": str(checkpoint_root), "scratch_path": str(speed.SCRATCH),
                                 "samples_file": "disk_samples.jsonl", "samples": len(disk), "missed_ticks": 0,
                                 "sha256": hashlib.sha256((records / "disk_samples.jsonl").read_bytes()).hexdigest(),
                                 "note": "离散采样非连续峰值；保持保守余量"}
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
    assert result["disk"]["sampled_max_allocated_bytes"] == 8192
    assert result["disk"]["final_allocated_bytes"] == 4096
    assert result["disk"]["scratch_available_min_bytes"] == 40960 - 614
    assert result["disk"]["save_samples"] == 4
    assert result["disk"]["checkpoint_save_commit_seconds"] == 1.899000666


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
        checkpoint_root = tmp_path / "v1-store/train-runs/mme_vla_suite/fixture"
        fake_checkpoint.save_state(SimpleNamespace(directory=checkpoint_root),
                                   SimpleNamespace(step=expected_steps), object(), expected_steps - 1)
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
    assert record["disk_sampling"]["stopped"]
    disk = speed.rows(record_path / "disk_samples.jsonl")
    assert disk[0]["kind"] == "start"
    assert disk[-1]["kind"] == "final"
    assert disk[0]["wall_end"] <= record["entry_start_wall"]
    assert disk[-1]["wall_start"] >= record["end_wall"]


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


def test_allocation_counts_blocks_sparse_hardlinks_and_temporary_directories(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    assert speed.checkpoint_allocation(checkpoint)["exists"] is False
    temporary = checkpoint / "299.orbax-checkpoint-tmp"
    temporary.mkdir(parents=True)
    sparse = temporary / "sparse"
    with sparse.open("wb") as stream:
        stream.seek(16 * 1024 * 1024)
        stream.write(b"x")
    os.link(sparse, temporary / "hardlink")
    expected = sum(path.stat().st_blocks * 512 for path in (checkpoint, temporary, sparse))
    record = speed.checkpoint_allocation(checkpoint)
    assert record["allocated_bytes"] == expected < sparse.stat().st_size
    assert record["unique_inodes"] == 3
    assert record["duplicate_inodes"] == 1
    assert not record["errors"]
    temporary.rename(checkpoint / "299")
    assert speed.checkpoint_allocation(checkpoint)["allocated_bytes"] == expected


def test_allocation_does_not_follow_links_and_records_disappearance(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload").write_bytes(b"x" * 8192)
    (checkpoint / "external").symlink_to(outside, target_is_directory=True)
    vanished = checkpoint / "vanished"
    vanished.write_bytes(b"x")
    real_stat = speed.os.stat

    def race(path, *args, **kwargs):
        if path == "vanished" and "dir_fd" in kwargs:
            vanished.unlink()
            raise FileNotFoundError(errno.ENOENT, "模拟扫描期间文件消失")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(speed.os, "stat", race)
    result = speed.checkpoint_allocation(checkpoint)
    assert result["symlinks"] == 1
    assert result["disappeared"] == 1
    assert result["allocated_bytes"] == checkpoint.stat().st_blocks * 512 + (checkpoint / "external").lstat().st_blocks * 512
    assert not result["errors"]


def test_disk_sampler_covers_asynchronous_save_wait_and_final_commit(tmp_path):
    checkpoint = tmp_path / "new-checkpoint"
    sampler = speed.DiskSampler(tmp_path / "disk.jsonl", checkpoint, tmp_path)
    sampler.start()
    save_returned = threading.Event()

    def asynchronous_save():
        temporary = checkpoint / "299.orbax-checkpoint-tmp"
        temporary.mkdir(parents=True)
        (temporary / "params").write_bytes(b"x" * 32768)
        save_returned.set()
        time.sleep(.65)
        temporary.rename(checkpoint / "299")

    worker = threading.Thread(target=asynchronous_save)
    sampler.phase = "save_dispatch"
    worker.start()
    assert save_returned.wait(timeout=2)
    sampler.phase = "save_pending"
    worker.join(timeout=2)  # 模拟原训练入口本来已有的wait_until_finished。
    assert not worker.is_alive()
    entry_end = time.time()
    metadata = sampler.finish(success=True)
    records = speed.rows(tmp_path / "disk.jsonl")
    assert metadata["stopped"]
    assert metadata["error"] is None
    assert records[0]["checkpoint"]["exists"] is False
    assert any(row["kind"] == "periodic" and row["phase"] == "save_pending"
               and row["checkpoint"]["allocated_bytes"] > 0 for row in records)
    assert records[-1]["wall_start"] >= entry_end
    assert records[-1]["checkpoint"] == speed.checkpoint_allocation(checkpoint)
    assert all(row["interval_s"] > 0 for row in records[1:])


def test_disk_sampler_records_io_errors_and_delays(tmp_path, monkeypatch):
    real_allocation = speed.checkpoint_allocation

    def slow_scan(path):
        time.sleep(.02)
        return real_allocation(path)

    def no_space_info(path):
        raise PermissionError(errno.EACCES, "模拟statvfs失败")

    monkeypatch.setattr(speed, "checkpoint_allocation", slow_scan)
    monkeypatch.setattr(speed.os, "statvfs", no_space_info)
    sampler = speed.DiskSampler(tmp_path / "disk.jsonl", tmp_path / "absent", tmp_path)
    sampler.start()
    sampler.finish(success=False)
    records = speed.rows(tmp_path / "disk.jsonl")
    assert all(row["duration_s"] >= .02 for row in records)
    assert all(row["errors"][0]["errno"] == errno.EACCES for row in records)
    assert records[-1]["phase"] == "entry_failed"


def test_disk_sampler_background_failure_is_not_hidden_by_final_sample(tmp_path, monkeypatch):
    real_allocation = speed.checkpoint_allocation
    calls = 0

    def fail_background(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EIO, "模拟后台扫描失败")
        return real_allocation(path)

    monkeypatch.setattr(speed, "checkpoint_allocation", fail_background)
    sampler = speed.DiskSampler(tmp_path / "disk.jsonl", tmp_path / "absent", tmp_path)
    sampler.start()
    sampler.thread.join(timeout=2)
    metadata = sampler.finish(success=True)
    assert metadata["stopped"]
    assert "模拟后台扫描失败" in metadata["error"]
    assert [row["kind"] for row in speed.rows(tmp_path / "disk.jsonl")] == ["start", "final"]


@pytest.mark.parametrize("fault", ["missing", "io_error", "not_stopped", "late_start", "early_final",
                                   "no_save_samples", "changed_bytes", "wrong_root", "final_race", "wrong_wait_run"])
def test_report_rejects_incomplete_disk_evidence(tmp_path, fault):
    records, gpu = fixture(tmp_path)
    metadata_path = records / "speed_run.json"
    meta = json.loads(metadata_path.read_text())
    path = records / "disk_samples.jsonl"
    data = speed.rows(path)
    if fault == "missing":
        del meta["disk_sampling"]
    elif fault == "not_stopped":
        meta["disk_sampling"]["stopped"] = False
    elif fault == "io_error":
        data[100]["checkpoint"]["errors"] = [{"errno": errno.EIO}]
    elif fault == "late_start":
        data[0]["wall_end"] = meta["entry_start_wall"] + 1
    elif fault == "early_final":
        data[-1]["wall_start"] = meta["end_wall"] - 1
    elif fault == "no_save_samples":
        # 所有周期样本都早于保存，最后一个样本只在入口完成后；不能称保存期间取到峰值。
        data = [row for row in data if row["wall_start"] < meta["save_calls"][0]["start_wall"] or row["kind"] == "final"]
        for index, row in enumerate(data):
            row["sequence"] = index
            row["interval_s"] = row["monotonic_start"] - data[index - 1]["monotonic_start"] if index else None
        meta["disk_sampling"]["samples"] = len(data)
    elif fault == "wrong_root":
        meta["disk_sampling"]["checkpoint_root"] = "其它run"
    elif fault == "final_race":
        data[-1]["checkpoint"]["disappeared"] = 1
    elif fault == "changed_bytes":
        data[-1]["checkpoint"]["allocated_bytes"] += 4096
    elif fault == "wrong_wait_run":
        wait_path = records / "final/checkpoint_wait_done.json"
        wait_record = json.loads(wait_path.read_text())
        wait_record["run_uuid"] = "另一run"
        wait_path.write_text(json.dumps(wait_record))
    dump_rows(path, data)
    if fault not in ("missing", "changed_bytes"):
        meta["disk_sampling"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="磁盘|checkpoint|保存目录|保存期间"):
        speed.summarize(records, gpu)


def test_report_rejects_samples_only_after_commit_during_wandb_finish(tmp_path):
    records, gpu = fixture(tmp_path)
    path = records / "disk_samples.jsonl"
    meta_path = records / "speed_run.json"
    meta = json.loads(meta_path.read_text())
    save_start = meta["save_calls"][0]["start_wall"]
    # 保存真正提交之后、入口结束之前只保留这一条周期样本。
    data = [row for row in speed.rows(path) if row["wall_start"] < save_start or row["wall_start"] >= save_start + 2]
    meta["end_wall"] += 1  # 额外W&B收尾时长，不扩大保存窗口。
    data[-1]["wall_start"] += 1
    data[-1]["wall_end"] += 1
    data[-1]["monotonic_start"] += 1
    for index, row in enumerate(data):
        row["sequence"] = index
        row["interval_s"] = row["monotonic_start"] - data[index - 1]["monotonic_start"] if index else None
    dump_rows(path, data)
    meta["disk_sampling"].update(samples=len(data), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="缺保存期间磁盘采样"):
        speed.summarize(records, gpu)


def os_env(key):
    return os.environ.get(key)
