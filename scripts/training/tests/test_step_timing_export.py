"""导出上限只影响查看器文件，验证环境恢复和收尾同步次数。"""

import contextlib
import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import step_timing


@pytest.mark.parametrize("previous",[None,"1","200000000"])
def test_export_limit_is_restored(tmp_path,monkeypatch,previous):
    key=step_timing.TRACE_VIEWER_LIMIT_ENV
    if previous is None: monkeypatch.delenv(key,raising=False)
    else: monkeypatch.setenv(key,previous)
    seen=[]; flushed=[]
    monkeypatch.setattr(step_timing.jax.profiler,"start_trace",lambda _:None)
    monkeypatch.setattr(step_timing.jax.profiler,"stop_trace",lambda:seen.append(int(os.environ[key])))
    monkeypatch.setattr(step_timing.jax.profiler,"StepTraceAnnotation",lambda *a,**k:contextlib.nullcontext())
    timer=step_timing.StepTiming(str(tmp_path),2)
    for step in range(3):
        with timer.step(step,flush=lambda:flushed.append(True)): pass
    assert seen==[max(100000000,int(previous or 0))]
    assert flushed==[True]
    assert os.environ.get(key)==previous
    assert json.loads(timer.trace_metadata.read_text())["trace_viewer_event_limit"]==seen[0]
    assert len(timer.path.read_text().splitlines())==2


def test_export_failure_restores_environment(tmp_path,monkeypatch):
    key=step_timing.TRACE_VIEWER_LIMIT_ENV
    monkeypatch.setenv(key,"3")
    monkeypatch.setattr(step_timing.jax.profiler,"start_trace",lambda _:None)
    monkeypatch.setattr(step_timing.jax.profiler,"StepTraceAnnotation",lambda *a,**k:contextlib.nullcontext())
    def fail():
        assert int(os.environ[key])>=100000000
        raise RuntimeError("模拟导出失败")
    monkeypatch.setattr(step_timing.jax.profiler,"stop_trace",fail)
    timer=step_timing.StepTiming(str(tmp_path),1)
    with pytest.raises(RuntimeError,match="模拟导出失败"):
        with timer.step(0): pass
    assert os.environ[key]=="3" and timer._file.closed
