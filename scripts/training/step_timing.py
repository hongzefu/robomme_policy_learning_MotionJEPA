"""训练首段的可选主线程计时与 JAX trace；不增加逐步设备同步。"""

from __future__ import annotations

import contextlib
import json
import pathlib
import time

import jax


class StepTiming:
    """主线程分段计时与设备 trace 分开留存，不能把异步 dispatch 当设备耗时。"""

    def __init__(self, root: str, steps: int):
        if steps <= 0:
            raise ValueError("计时步数必须为正整数")
        self.root = pathlib.Path(root)
        self.steps = steps
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "step_timing.jsonl"
        self.trace_dir = self.root / "step_trace"
        if self.path.exists() or self.trace_dir.exists():
            raise FileExistsError("计时或 trace 输出已存在，拒绝覆盖")
        self._file = self.path.open("x", encoding="utf-8")
        self._active = False
        self._row = None
        self._count = 0
        self._closed = False

    @contextlib.contextmanager
    def step(self, step: int, *, flush=None):
        if self._count >= self.steps:
            yield
            return
        if not self._active:
            jax.profiler.start_trace(self.trace_dir)
            self._active = True
        start = time.perf_counter()
        self._row = {"step": int(step), "wall_start": time.time(), "phases_s": {}}
        completed = False
        try:
            with jax.profiler.StepTraceAnnotation("motionjepa_train", step_num=int(step)):
                yield
            completed = True
        finally:
            self._row["completed"] = completed
            self._row["host_step_s"] = time.perf_counter() - start
            self._row["wall_end"] = time.time()
            self._file.write(json.dumps(self._row, ensure_ascii=False) + "\n")
            self._file.flush()
            self._row = None
            self._count += 1
            if self._count >= self.steps:
                self.close(flush=flush if completed else None)

    @contextlib.contextmanager
    def phase(self, name: str):
        if self._row is None:
            yield
            return
        start = time.perf_counter()
        try:
            with jax.profiler.TraceAnnotation("motionjepa_" + name, step_num=self._row["step"]):
                yield
        finally:
            self._row["phases_s"][name] = time.perf_counter() - start

    def close(self, *, flush=None):
        if self._closed:
            return
        self._closed = True
        try:
            if self._active:
                try:
                    # 仅在整段 trace 结束时等待一次，计时行已封存；不在每步同步。
                    if flush is not None:
                        flush()
                finally:
                    jax.profiler.stop_trace()
                    self._active = False
        finally:
            self._file.close()
        print(f"STEP_TIMING_DONE steps={self._count} host={self.path} trace={self.trace_dir}", flush=True)
