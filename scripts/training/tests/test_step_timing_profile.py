"""StepTiming 的 TRAIN_TIMING_PROFILE 开关单测（CPU、秒级、不跑真实训练）。

背景：raw9 正式训练在第 300 步 jax.profiler.stop_trace 之后继续训练，与 L1 数据管线后台线程冲突 abort
（absl BadStatusOrAccess，EXIT_CODE=134）。修法：TRAIN_TIMING_PROFILE=0 时不调 jax.profiler 任何接口，
step_timing.jsonl 计时行照写。本测试用假 step 循环（含计时窗口之后继续跑的步）验证：
  ① profile=0：start_trace / stop_trace / StepTraceAnnotation / TraceAnnotation 调用次数全为 0，不建 step_trace，
     jsonl 行数与字段与旧行为相同，flush 不被调用（不新增设备同步），close 打印 profile=0；
  ② profile=1（缺省）：旧行为不变——start/stop 各 1 次、每步一次 StepTraceAnnotation、每个 phase 一次 TraceAnnotation、
     close 时 flush 1 次，close 打印 profile=1；
  ③ 非法取值拒绝。
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import step_timing  # noqa: E402

ROW_KEYS = {"step", "wall_start", "phases_s", "completed", "host_step_s", "wall_end"}


@pytest.fixture
def counts(monkeypatch):
    c = {"start": 0, "stop": 0, "step_ann": 0, "ann": 0}

    def start_trace(d):
        c["start"] += 1
        pathlib.Path(d).mkdir(parents=True)

    def stop_trace():
        c["stop"] += 1

    @contextlib.contextmanager
    def step_ann(*a, **k):
        c["step_ann"] += 1
        yield

    @contextlib.contextmanager
    def ann(*a, **k):
        c["ann"] += 1
        yield

    prof = step_timing.jax.profiler
    monkeypatch.setattr(prof, "start_trace", start_trace)
    monkeypatch.setattr(prof, "stop_trace", stop_trace)
    monkeypatch.setattr(prof, "StepTraceAnnotation", step_ann)
    monkeypatch.setattr(prof, "TraceAnnotation", ann)
    return c


def fake_loop(root, steps_timed, total_steps, flushes):
    t = step_timing.StepTiming(str(root), steps_timed)
    for s in range(total_steps):          # 计时窗口结束后继续跑，模拟「停 trace 后继续训练」
        with t.step(s, flush=lambda: flushes.append(s)):
            with t.phase("train_dispatch"):
                pass
            with t.phase("data_next"):
                pass
    t.close()
    return t


def rows(root):
    return [json.loads(x) for x in (root / "step_timing.jsonl").read_text().splitlines() if x.strip()]


def test_profile_off_no_profiler(tmp_path, counts, monkeypatch, capsys):
    monkeypatch.setenv("TRAIN_TIMING_PROFILE", "0")
    flushes = []
    t = fake_loop(tmp_path, 5, 12, flushes)
    assert t.profile is False
    assert counts == {"start": 0, "stop": 0, "step_ann": 0, "ann": 0}
    assert not (tmp_path / "step_trace").exists()
    assert flushes == []
    r = rows(tmp_path)
    assert [x["step"] for x in r] == list(range(5))
    for x in r:
        assert set(x) == ROW_KEYS
        assert x["completed"] is True and set(x["phases_s"]) == {"train_dispatch", "data_next"}
    out = capsys.readouterr().out
    assert "STEP_TIMING_DONE steps=5 " in out and out.rstrip().endswith("profile=0")


def test_profile_default_old_behavior(tmp_path, counts, monkeypatch, capsys):
    monkeypatch.delenv("TRAIN_TIMING_PROFILE", raising=False)
    flushes = []
    t = fake_loop(tmp_path, 5, 12, flushes)
    assert t.profile is True
    assert counts == {"start": 1, "stop": 1, "step_ann": 5, "ann": 10}
    assert (tmp_path / "step_trace").is_dir()
    assert flushes == [4]                  # 只在第 N 步结束时同步一次
    r = rows(tmp_path)
    assert [x["step"] for x in r] == list(range(5))
    assert all(set(x) == ROW_KEYS for x in r)
    out = capsys.readouterr().out
    assert "STEP_TIMING_DONE steps=5 " in out and out.rstrip().endswith("profile=1")


def test_profile_explicit_one(tmp_path, counts, monkeypatch):
    monkeypatch.setenv("TRAIN_TIMING_PROFILE", "1")
    fake_loop(tmp_path, 3, 6, [])
    assert counts == {"start": 1, "stop": 1, "step_ann": 3, "ann": 6}


@pytest.mark.parametrize("bad", ["", "2", "true", "no"])
def test_profile_bad_value(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("TRAIN_TIMING_PROFILE", bad)
    with pytest.raises(ValueError):
        step_timing.StepTiming(str(tmp_path), 3)
